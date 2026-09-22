//! Pure-Rust quantized loader for the `deepseek4` GGUF architecture.
//!
//! DeepSeek-V4 ("DSV4") is the successor to DeepSeek-V2/V3.  It keeps MLA and
//! fine-grained MoE, and adds three features on top:
//!
//! * **Hyper-Connections (HC).**  The residual stream is expanded to
//!   `hc_mult` parallel copies.  Every block first mixes the copies down to a
//!   single stream (`hc_pre`), runs the attention / MoE body on that single
//!   stream, then re-expands to `hc_mult` copies (`hc_post`).  The mixing
//!   weights come from a learned per-token `hc_*_fn` linear map, and the
//!   combination matrix is made doubly-stochastic by a Sinkhorn iteration.
//! * **KV compression.**  Layers alternate between a pure sliding-window
//!   attention (ratio 0), a learned *compressor* with ratio 4 (the
//!   "CSA" layers), and a learned compressor with ratio 128 (the "HCA"
//!   layers).  The compressor is a gated pooling over `ratio` consecutive
//!   tokens: `softmax(wgate·x + ape) · (wkv·x)`.
//! * **Sparse attention via an indexer.**  The ratio-4 layers select the
//!   `index_topk` best compressed KV positions for each query using a learned
//!   *Lightning Indexer*: a low-rank Q, a Hadamard-rotated compressed KV, a
//!   ReLU bilinear score with a causal mask, and a top-k pick.  Only those
//!   rows plus the sliding window participate in attention.
//!
//! The model also has a "parallel head": the final hidden state is first
//! collapsed from `hc_mult` copies into one via `output_hc_*` weights (a
//! sigmoid gate over a RMS-normalized `hc_mult·d` vector), then the output
//! matrix produces logits.
//!
//! Like [`crate::quantized_deepseek2`], experts are sliced from the 3-D
//! quantized expert tensors (no dequantization) so a 162 B MoE keeps its
//! on-disk footprint, and activations run in f32 on the CPU.
//!
//! Reference: DeepSeek-V4 official modeling code (`.py`) and llama.cpp's
//! `llama_model_deepseek4` (`/tmp/deepseek4.cpp`, `/tmp/kv-dsv4.cpp`).

use std::borrow::Cow;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

use candle_core::quantized::{gguf_file, GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::ops::{sigmoid, silu, softmax, softmax_last_dim};
use candle_transformers::quantized_nn::RmsNorm;

use crate::gguf_ext::GgufHeader;

/// Numerically stable `log(1 + exp(x))`.
fn softplus(x: &Tensor) -> Result<Tensor> {
    let ax = x.abs()?;
    let e = ax.neg()?.exp()?;
    x.maximum(0.0)?.add(&(e + 1.0)?.log()?)
}

const HCA_RATIO: usize = 128;
const CSA_RATIO: usize = 4;
/// Hard cap on the KV context (and rope tables) per model instance.
const KV_CAP: usize = 262_144;
/// Minimum prompt length for the layer-ahead expert prefetch to engage
/// (shorter prompts don't justify streaming whole layers).
const PREFETCH_AHEAD_MIN: usize = 8;

/// Parsed `deepseek4` hyper-parameters.
struct Config {
    n_layer: usize,
    n_head: usize,
    rms_eps: f64,
    // MLA dims.
    q_lora_rank: usize,
    head_dim: usize,
    rope_head_dim: usize,
    nope_head_dim: usize,
    // Compression / indexer.
    compress_ratios: Vec<usize>,
    compress_rope_base: f32,
    window_size: usize,
    index_n_head: usize,
    index_head_dim: usize,
    index_topk: usize,
    // Output grouping.
    o_groups: usize,
    o_lora_rank: usize,
    // Hyper-Connections.
    hc_mult: usize,
    hc_sinkhorn_iters: usize,
    hc_eps: f64,
    // MoE.
    n_expert: usize,
    n_expert_used: usize,
    n_expert_shared: usize,
    n_hash_layer: usize,
    expert_weights_scale: f64,
    swiglu_clamp: Vec<f64>, // per-layer clamp for gate/up
    swiglu_clamp_shexp: Vec<f64>,
    // RoPE (YaRN).
    rope_theta: f32,
    context_length: usize,
    yarn: Option<YarnConfig>,
}

struct YarnConfig {
    factor: f32,
    orig_context_length: usize,
    mscale_all_dim: f32,
}

// ─── Metadata helpers ───────────────────────────────────────────────────────

struct Meta<'a>(&'a std::collections::HashMap<String, gguf_file::Value>);

impl Meta<'_> {
    fn u32(&self, key: &str) -> Result<u32> {
        match self.0.get(key) {
            Some(v) => v.to_u32(),
            None => candle_core::bail!("deepseek4: missing GGUF metadata key `{key}`"),
        }
    }
    fn u32_or(&self, key: &str, default: u32) -> u32 {
        self.0
            .get(key)
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(default)
    }
    fn f32(&self, key: &str) -> Result<f32> {
        match self.0.get(key) {
            Some(v) => v.to_f32(),
            None => candle_core::bail!("deepseek4: missing GGUF metadata key `{key}`"),
        }
    }
    fn f32_or(&self, key: &str, default: f32) -> f32 {
        self.0
            .get(key)
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(default)
    }
    #[allow(dead_code)]
    fn bool_or(&self, key: &str, default: bool) -> bool {
        self.0
            .get(key)
            .and_then(|v| v.to_bool().ok())
            .unwrap_or(default)
    }
    fn array_f64(&self, key: &str, n: usize) -> Vec<f64> {
        match self.0.get(key) {
            Some(gguf_file::Value::Array(arr)) => {
                let mut out = Vec::with_capacity(n);
                for v in arr.iter().take(n) {
                    out.push(v.to_f32().map(|x| x as f64).unwrap_or(0.0));
                }
                while out.len() < n {
                    out.push(0.0);
                }
                out
            }
            _ => vec![0.0; n],
        }
    }
    fn array_u32(&self, key: &str, n: usize) -> Vec<usize> {
        match self.0.get(key) {
            Some(gguf_file::Value::Array(arr)) => {
                let mut out = Vec::with_capacity(n);
                for v in arr.iter().take(n) {
                    out.push(v.to_u32().unwrap_or(0) as usize);
                }
                while out.len() < n {
                    out.push(0);
                }
                out
            }
            _ => vec![0; n],
        }
    }
}

impl Config {
    fn from_metadata(md: &std::collections::HashMap<String, gguf_file::Value>) -> Result<Self> {
        let m = Meta(md);
        let a = "deepseek4";
        let n_layer = m.u32(&format!("{a}.block_count"))? as usize;
        let n_head = m.u32(&format!("{a}.attention.head_count"))? as usize;
        let _n_embd = m.u32(&format!("{a}.embedding_length"))? as usize;
        let rms_eps = m.f32(&format!("{a}.attention.layer_norm_rms_epsilon"))? as f64;

        let q_lora_rank = m.u32(&format!("{a}.attention.q_lora_rank"))? as usize;
        let head_dim = m.u32_or(&format!("{a}.attention.key_length"), 0).max(1) as usize;
        // llama.cpp stores this at `{a}.rope.dimension_count`; older files used
        // `{a}.attention.rope.dimension_count`.  Prefer the current key and
        // fall back to the legacy one.  A file with neither is malformed for
        // this architecture (every real model ships it), so refuse to load —
        // a silent 1-wide rotary would only fail on the first generated token.
        let rope_head_dim = m
            .u32(&format!("{a}.rope.dimension_count"))
            .or_else(|_| m.u32(&format!("{a}.attention.rope.dimension_count")))
            .map(|v| v as usize)
            .map_err(|_| {
                candle_core::Error::Msg(format!(
                    "deepseek4: missing rotary-dimension metadata (`{a}.rope.dimension_count` \
                     or legacy `{a}.attention.rope.dimension_count`)"
                ))
            })?;
        let nope_head_dim = head_dim.saturating_sub(rope_head_dim);

        let compress_ratios = m.array_u32(&format!("{a}.attention.compress_ratios"), n_layer);
        let compress_rope_base =
            m.f32_or(&format!("{a}.attention.compress_rope_freq_base"), 160000.0);
        let window_size = m
            .u32_or(&format!("{a}.attention.sliding_window"), 128)
            .max(1) as usize;

        let index_n_head = m.u32_or(&format!("{a}.attention.indexer.head_count"), 64) as usize;
        let index_head_dim = m.u32_or(&format!("{a}.attention.indexer.key_length"), 128) as usize;
        let index_topk = m.u32_or(&format!("{a}.attention.indexer.top_k"), 512) as usize;

        let o_groups = m.u32_or(&format!("{a}.attention.output_group_count"), 8) as usize;
        let o_lora_rank = m.u32_or(&format!("{a}.attention.output_lora_rank"), 1024) as usize;

        let hc_mult = m.u32_or(&format!("{a}.hyper_connection.count"), 4) as usize;
        let hc_sinkhorn_iters = m
            .u32_or(&format!("{a}.hyper_connection.sinkhorn_iterations"), 20)
            .max(1) as usize;
        let hc_eps = m.f32_or(&format!("{a}.hyper_connection.epsilon"), 1e-6) as f64;

        let n_expert = m.u32(&format!("{a}.expert_count"))? as usize;
        let n_expert_used = m.u32(&format!("{a}.expert_used_count"))? as usize;
        let n_expert_shared = m.u32_or(&format!("{a}.expert_shared_count"), 1).max(1) as usize;
        let n_hash_layer = m.u32_or(&format!("{a}.hash_layer_count"), 0) as usize;
        let expert_weights_scale = m.f32_or(&format!("{a}.expert_weights_scale"), 0.0) as f64;
        let swiglu_clamp = m.array_f64(&format!("{a}.swiglu_clamp_exp"), n_layer);
        // llama.cpp: clamp_shexp defaults to clamp_exp when absent.
        let swiglu_clamp_shexp = if m.0.contains_key(&format!("{a}.swiglu_clamp_shexp")) {
            m.array_f64(&format!("{a}.swiglu_clamp_shexp"), n_layer)
        } else {
            swiglu_clamp.clone()
        };

        let rope_theta = m.f32_or(&format!("{a}.rope.freq_base"), 10000.0);
        let context_length = m.u32_or(&format!("{a}.context_length"), 0).max(1) as usize;
        let yarn =
            m.0.get(&format!("{a}.rope.scaling.type"))
                .and_then(|v| v.to_string().ok().cloned())
                .filter(|s| s == "yarn")
                .map(|_| YarnConfig {
                    factor: m.f32_or(&format!("{a}.rope.scaling.factor"), 16.0),
                    orig_context_length: m
                        .u32_or(&format!("{a}.rope.scaling.original_context_length"), 65536)
                        as usize,
                    // llama.cpp stores 0.1 * mscale_all_dim and divides it back out.
                    mscale_all_dim: m.f32_or(&format!("{a}.rope.scaling.yarn_log_multiplier"), 0.0)
                        / 0.1,
                });

        Ok(Self {
            n_layer,
            n_head,
            rms_eps,
            q_lora_rank,
            head_dim,
            rope_head_dim,
            nope_head_dim,
            compress_ratios,
            compress_rope_base,
            window_size,
            index_n_head,
            index_head_dim,
            index_topk,
            o_groups,
            o_lora_rank,
            hc_mult,
            hc_sinkhorn_iters,
            hc_eps,
            n_expert,
            n_expert_used,
            n_expert_shared,
            n_hash_layer,
            expert_weights_scale,
            swiglu_clamp,
            swiglu_clamp_shexp,
            rope_theta,
            context_length,
            yarn,
        })
    }
}

// ─── RoPE (YaRN-aware, applied to the rope slice only) ─────────────────────
//
// The rope is *per-layer*: pure sliding-window layers (ratio 0) use the plain
// `rope_theta` base with no YaRN (they never look beyond the window), while
// compressed layers use `compress_rope_freq_base` with the model's YaRN
// scaling (matches llama.cpp `build_csa_attention` and the official python,
// which passes `original_seq_len = 0` for raw layers).

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    /// `base`: rope base; `yarn`: enable YaRN interpolation with the model's
    /// yarn config.  Table covers token positions `[0, max_seq)`.
    fn new(cfg: &Config, dev: &Device, base: f32, yarn: bool, max_seq: usize) -> Result<Self> {
        let dim = cfg.rope_head_dim;
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f32 / dim as f32))
            .collect();
        let mscale = if yarn {
            let y = cfg
                .yarn
                .as_ref()
                .expect("yarn rope requested without yarn config");
            yarn_get_mscale(y.factor, y.mscale_all_dim)
        } else {
            1.0
        };
        let inv_freq = if yarn {
            let y = cfg.yarn.as_ref().unwrap();
            let half = dim / 2;
            let freq_inter: Vec<f32> = inv_freq.iter().map(|f| f / y.factor).collect();
            let (low, high) = yarn_correction_range(32.0, 1.0, dim, base, y.orig_context_length);
            let ramp = yarn_linear_ramp(low, high, half);
            (0..half)
                .map(|i| {
                    let mask = 1.0 - ramp[i];
                    freq_inter[i] * (1.0 - mask) + inv_freq[i] * mask
                })
                .collect()
        } else {
            inv_freq
        };
        Self::from_inv_freq(inv_freq, max_seq, mscale, dev)
    }

    fn from_inv_freq(
        inv_freq: Vec<f32>,
        max_seq: usize,
        mscale: f32,
        dev: &Device,
    ) -> Result<Self> {
        let n = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, n), dev)?;
        let t = Tensor::arange(0u32, max_seq as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = (freqs.sin()? * mscale as f64)?;
        let cos = (freqs.cos()? * mscale as f64)?;
        Ok(Self { sin, cos })
    }

    /// Apply interleaved RoPE to a `[b, heads, seq, rope_dim]` tensor.
    fn apply(&self, x: &Tensor, offset: usize, seq_len: usize) -> Result<Tensor> {
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)
    }

    /// Apply inverse interleaved RoPE (conjugate rotation) to a
    /// `[b, heads, seq, rope_dim]` tensor, matching `ggml_rope_ext_back`.
    fn apply_back(&self, x: &Tensor, offset: usize, seq_len: usize) -> Result<Tensor> {
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &(&sin * -1.0)?)
    }

    /// Apply at explicit (compressed) positions: x is `[n, rope_dim]`,
    /// `positions` are `[n]` u32 TOKEN positions of each row (block index
    /// times the compression ratio), matching llama's `comp_pos` +
    /// `ggml_rope_ext` with the compress rope.
    fn apply_at(&self, x: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let n = x.dim(0)?;
        let d = x.dim(1)?;
        let x4 = x.reshape((n, 1, 1, d))?;
        // rope_i accepts 3-D cos/sin as [b, t, d], one row per batch item.
        let cos = self.cos.index_select(positions, 0)?.unsqueeze(1)?; // [n, 1, half]
        let sin = self.sin.index_select(positions, 0)?.unsqueeze(1)?;
        let out = candle_nn::rotary_emb::rope_i(&x4.contiguous()?, &cos, &sin)?;
        out.reshape((n, d))
    }
}

fn yarn_find_correction_dim(num_rot: f32, dim: usize, base: f32, max_pos: usize) -> f32 {
    (dim as f32 * (max_pos as f32 / (num_rot * 2.0 * std::f32::consts::PI)).ln())
        / (2.0 * base.ln())
}

fn yarn_correction_range(
    low_rot: f32,
    high_rot: f32,
    dim: usize,
    base: f32,
    max_pos: usize,
) -> (f32, f32) {
    let low = yarn_find_correction_dim(low_rot, dim, base, max_pos).floor();
    let high = yarn_find_correction_dim(high_rot, dim, base, max_pos).ceil();
    (low.max(0.0), high.min(dim as f32 - 1.0))
}

fn yarn_linear_ramp(min: f32, mut max: f32, dim: usize) -> Vec<f32> {
    if (min - max).abs() < f32::EPSILON {
        max += 0.001;
    }
    (0..dim)
        .map(|i| (((i as f32) - min) / (max - min)).clamp(0.0, 1.0))
        .collect()
}

fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

// ─── Linear helpers ─────────────────────────────────────────────────────────

struct Mlp {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
    clamp: f64,
    /// When the weights are borrowed from the model mapping, handles that
    /// can `MADV_WILLNEED` the gate/up/down byte ranges ahead of a matmul.
    /// `None` for streamed/fallback loads (weights are plain memory then).
    prefetch: Option<crate::residency::ExpertHandles>,
}

/// One expert's weight tensor: the [`QMatMul`] plus an optional handle for
/// prefetching its byte range from the mapping.
struct ExpertTensor {
    qmatmul: QMatMul,
    prefetch: Option<Arc<dyn crate::mmap_tensor::MmapPrefetch>>,
}

/// The swiglu gate clamp, `(-inf, clamp]` when `clamp > 0`.  Shared by the
/// host [`Mlp`] and the device-resident [`Ds4DeviceExpert`] so the two
/// forms cannot drift.
fn clamp_gate(gate: Tensor, clamp: f64) -> Result<Tensor> {
    if clamp > 0.0 {
        gate.clamp(f64::NEG_INFINITY, clamp)
    } else {
        Ok(gate)
    }
}

/// The swiglu up clamp, `[-clamp, clamp]` when `clamp > 0`.
fn clamp_up(up: Tensor, clamp: f64) -> Result<Tensor> {
    if clamp > 0.0 {
        up.clamp(-clamp, clamp)
    } else {
        Ok(up)
    }
}

impl Mlp {
    /// Gate projection + optional clamp.  Split out of [`Mlp::forward`] so
    /// the MoE dispatch can run all experts' gates (then all ups, then all
    /// downs) — reading each weight tensor as one sequential stream instead
    /// of jumping between gate/up/down on every expert.
    fn gate_forward(&self, xs: &Tensor) -> Result<Tensor> {
        clamp_gate(self.gate.forward(xs)?, self.clamp)
    }

    /// Up projection + optional clamp (see [`Mlp::gate_forward`]).
    fn up_forward(&self, xs: &Tensor) -> Result<Tensor> {
        clamp_up(self.up.forward(xs)?, self.clamp)
    }

    /// Combine `silu(gate) * up` and run the down projection.
    fn combine_and_down(&self, gate: Tensor, up: Tensor) -> Result<Tensor> {
        let h = (silu(&gate)? * up)?;
        self.down.forward(&h)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_forward(xs)?;
        let up = self.up_forward(xs)?;
        self.combine_and_down(gate, up)
    }

    /// Ask the kernel to prefetch this expert's weight pages (best effort;
    /// no-op when the weights are not mmap-backed).
    fn prefetch(&self) {
        if let Some(p) = &self.prefetch {
            p.prefetch();
        }
    }
}

/// One routed expert resident in the device expert pool: the same three
/// block-quantized projections as its host [`Mlp`], uploaded as-is
/// (IQ2_XXS gate/up, Q2_K down on the real model) and run through the
/// device's quantized kernels.  An opaque, byte-sized payload for
/// [`crate::residency::DeviceResidency`].
pub(crate) struct Ds4DeviceExpert {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
    clamp: f64,
    bytes: u64,
}

impl crate::residency::DeviceExpertSlot for Ds4DeviceExpert {
    fn device_bytes(&self) -> u64 {
        self.bytes
    }
}

impl Ds4DeviceExpert {
    /// `down(silu(clamp(gate·x)) ⊙ clamp(up·x))`, exactly [`Mlp::forward`].
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = clamp_gate(self.gate.forward(xs)?, self.clamp)?;
        let up = clamp_up(self.up.forward(xs)?, self.clamp)?;
        self.down.forward(&(silu(&gate)? * up)?)
    }
}

/// The host-side view of one routed expert the device pool uploads from:
/// the borrowed (mmap) block tensors, their prefetch handles and the exact
/// byte size a slot will occupy.
struct HostExpert {
    gate: Arc<QTensor>,
    up: Arc<QTensor>,
    down: Arc<QTensor>,
    clamp: f64,
    prefetch: Option<crate::residency::ExpertHandles>,
    bytes: u64,
}

/// How a routed expert that is *not* resident in the device pool runs
/// during decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MissPolicy {
    /// Run the host kernels now (overlapping the device's work on the
    /// resident experts) and queue a background upload, so the next step
    /// that routes here finds the expert resident.  The default.
    Host,
    /// Upload synchronously (a blocking write on the device's transfer
    /// queue) and run on the device.  A measurement mode
    /// (`JOSHUA_EXPERT_MISS=upload`): it stalls the decode thread for the
    /// transfer but warms the pool fastest.
    Upload,
}

impl MissPolicy {
    fn from_env() -> Self {
        match std::env::var("JOSHUA_EXPERT_MISS").as_deref() {
            Ok("upload") | Ok("sync") => Self::Upload,
            _ => Self::Host,
        }
    }
}

/// What happens to a routed expert's *host* pages once it is resident in
/// the device pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HostPagePolicy {
    /// Drop them from the mapping and the page cache
    /// ([`crate::residency::ExpertHandles::release_host_pages`]) a moment
    /// after the upload, so RAM holds experts the card does not: the two
    /// tiers are exclusive and a host smaller than the pool covers more of
    /// it.  The default.
    Drop,
    /// Leave them to the kernel (`JOSHUA_EXPERT_HOST_PAGES=keep`): the page
    /// cache may carry a copy of the card's experts.  For hosts whose RAM
    /// holds the whole pool anyway, or to bisect a regression.
    Keep,
}

impl HostPagePolicy {
    fn from_env() -> Self {
        match std::env::var("JOSHUA_EXPERT_HOST_PAGES").as_deref() {
            Ok("keep") | Ok("0") => Self::Keep,
            _ => Self::Drop,
        }
    }
}

/// Where the time of a forward pass with a device expert pool goes, and
/// how much of what the host had to run was in RAM when it ran.  One
/// accumulator each for decode steps and prefill; snapshot with
/// [`ModelWeights::expert_phase_timing`] and printed in the periodic
/// `debug` line.
#[derive(Default)]
struct PhaseTiming {
    /// Forward passes accumulated (decode steps, or prefill layer-chunks).
    passes: std::sync::atomic::AtomicU64,
    /// Wall time of those passes.
    pass_ns: std::sync::atomic::AtomicU64,
    /// Enqueuing the resident experts' launches.
    device_launch_ns: std::sync::atomic::AtomicU64,
    /// Running the misses on the host expert kernels.
    host_ns: std::sync::atomic::AtomicU64,
    /// Waiting for the device's partial sum after the host part (the
    /// read-back when the block's output lives on the CPU; with the dense
    /// set on the device the wait lands in the next layer's work instead).
    device_wait_ns: std::sync::atomic::AtomicU64,
    /// Host misses (expert visits run on the host).
    miss_experts: std::sync::atomic::AtomicU64,
    /// Host misses whose pages were all resident before the run (RAM, not
    /// disk).  Counted only while page probing is on.
    miss_experts_resident: std::sync::atomic::AtomicU64,
    /// Pages of the probed host misses, and how many were resident.
    miss_pages: std::sync::atomic::AtomicU64,
    miss_pages_resident: std::sync::atomic::AtomicU64,
}

/// A snapshot of one [`PhaseTiming`] accumulator.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpertPhaseTiming {
    /// Forward passes accumulated (decode steps, or prefill layer-chunks).
    pub passes: u64,
    /// Wall time of those passes, nanoseconds.
    pub pass_ns: u64,
    /// Time enqueuing the resident experts' launches, nanoseconds.
    pub device_launch_ns: u64,
    /// Time running the misses on the host, nanoseconds.
    pub host_ns: u64,
    /// Time waiting for the device's partial sum, nanoseconds.
    pub device_wait_ns: u64,
    /// Host misses.
    pub miss_experts: u64,
    /// Host misses whose pages were all resident before the run.
    pub miss_experts_resident: u64,
    /// Pages of the probed host misses.
    pub miss_pages: u64,
    /// …of which resident before the run.
    pub miss_pages_resident: u64,
}

impl PhaseTiming {
    fn add(counter: &std::sync::atomic::AtomicU64, v: u64) {
        counter.fetch_add(v, std::sync::atomic::Ordering::Relaxed);
    }

    fn snapshot(&self) -> ExpertPhaseTiming {
        use std::sync::atomic::Ordering::Relaxed;
        ExpertPhaseTiming {
            passes: self.passes.load(Relaxed),
            pass_ns: self.pass_ns.load(Relaxed),
            device_launch_ns: self.device_launch_ns.load(Relaxed),
            host_ns: self.host_ns.load(Relaxed),
            device_wait_ns: self.device_wait_ns.load(Relaxed),
            miss_experts: self.miss_experts.load(Relaxed),
            miss_experts_resident: self.miss_experts_resident.load(Relaxed),
            miss_pages: self.miss_pages.load(Relaxed),
            miss_pages_resident: self.miss_pages_resident.load(Relaxed),
        }
    }

    /// Snapshot and zero (the prefill accumulator is reported per prefill).
    fn take(&self) -> ExpertPhaseTiming {
        use std::sync::atomic::Ordering::Relaxed;
        ExpertPhaseTiming {
            passes: self.passes.swap(0, Relaxed),
            pass_ns: self.pass_ns.swap(0, Relaxed),
            device_launch_ns: self.device_launch_ns.swap(0, Relaxed),
            host_ns: self.host_ns.swap(0, Relaxed),
            device_wait_ns: self.device_wait_ns.swap(0, Relaxed),
            miss_experts: self.miss_experts.swap(0, Relaxed),
            miss_experts_resident: self.miss_experts_resident.swap(0, Relaxed),
            miss_pages: self.miss_pages.swap(0, Relaxed),
            miss_pages_resident: self.miss_pages_resident.swap(0, Relaxed),
        }
    }
}

impl ExpertPhaseTiming {
    /// The time split as one log-friendly line: per-pass averages in
    /// milliseconds and the host-miss page residency.
    pub fn describe(&self, per: &str) -> String {
        let n = self.passes.max(1) as f64;
        let ms = |ns: u64| ns as f64 / 1e6 / n;
        let rest = self
            .pass_ns
            .saturating_sub(self.device_launch_ns + self.host_ns + self.device_wait_ns);
        let pages = if self.miss_pages > 0 {
            format!(
                ", {:.1}% of their pages resident before the run ({} of {} experts fully resident)",
                100.0 * self.miss_pages_resident as f64 / self.miss_pages as f64,
                self.miss_experts_resident,
                self.miss_experts,
            )
        } else {
            String::new()
        };
        format!(
            "per {per} (avg over {}): total {:.1} ms = device launch {:.1} + host experts {:.1} + \
             device wait {:.1} + rest {:.1}; host misses {:.1} experts{pages}",
            self.passes,
            ms(self.pass_ns),
            ms(self.device_launch_ns),
            ms(self.host_ns),
            ms(self.device_wait_ns),
            ms(rest),
            self.miss_experts as f64 / n,
        )
    }
}

/// The model-wide device expert pool: a byte-budgeted LRU of
/// [`Ds4DeviceExpert`] slots keyed by `(layer, expert)`, the background
/// uploader that fills it, and the device the slots live on.  Shared by
/// every MoE layer (for lookups and warming requests) and by the model's
/// residency handle (hot-set protection).
struct Ds4DevicePool {
    device: Device,
    pool: Arc<crate::residency::DeviceResidency<Ds4DeviceExpert>>,
    /// Decode-step time split and host-miss residency (model-wide,
    /// cumulative).  Prefill timing is per session (`ModelWeights`).
    decode: PhaseTiming,
    /// Probe each host miss's page residency (`mincore`) before it runs.
    /// On with `JOSHUA_EXPERT_STATS=1` or a `debug`-level `joshua` log
    /// filter at load; a few hundred syscalls per decode step otherwise
    /// nobody reads.
    probe_pages: bool,
    uploader: Arc<crate::residency::ExpertUploader<Ds4DeviceExpert>>,
    miss: MissPolicy,
    /// What happens to an uploaded expert's host pages.
    host_pages: HostPagePolicy,
    /// Experts the host kernels are running right now, with how many runs
    /// (sessions) are on each; the release hook declines them and retries
    /// later.  A count, not a set: two sessions missing the same expert
    /// must both finish before its pages may go.
    busy: Arc<BusyExperts>,
}

/// Reference counts of the experts under host kernels (see
/// `Ds4DevicePool::busy`).
type BusyExperts = std::sync::Mutex<std::collections::HashMap<(u32, u32), u32>>;

/// Holds one host run's count on each of `keys` in a [`BusyExperts`] map,
/// released on drop — on every exit path, an error or a panic included, so
/// a failed transfer can never pin an expert's host pages for good.
struct BusyGuard<'a> {
    busy: &'a BusyExperts,
    keys: Vec<(u32, u32)>,
}

impl<'a> BusyGuard<'a> {
    fn new(busy: &'a BusyExperts, keys: Vec<(u32, u32)>) -> Self {
        {
            let mut map = busy.lock().unwrap_or_else(|p| p.into_inner());
            for k in &keys {
                *map.entry(*k).or_insert(0) += 1;
            }
        }
        Self { busy, keys }
    }
}

impl Drop for BusyGuard<'_> {
    fn drop(&mut self) {
        // Only this run's count: another session may still be on it.
        let mut map = self.busy.lock().unwrap_or_else(|p| p.into_inner());
        for k in &self.keys {
            if let Some(n) = map.get_mut(k) {
                *n -= 1;
                if *n == 0 {
                    map.remove(k);
                }
            }
        }
    }
}

/// Per-call dispatch settings a session passes down to its MoE blocks
/// (never stored in the shared pool: concurrent sessions would race).
#[derive(Clone)]
struct DispatchCtx {
    /// A decode step (one token per sequence, batched or not): the pool
    /// warms from its routing.  A prefill only reads the pool.  Explicit,
    /// never derived from the tensor's sequence length: a batched decode
    /// concatenates one token per sequence into a longer one.
    decode: bool,
    /// Whether a prefill's last row seeds the device pool: true for a
    /// whole-prompt forward and for the final chunk of a streamed prefill
    /// only, so a small prefill chunk does not queue an upload per chunk.
    seed_prefill: bool,
    /// Row offset of this call's tokens inside the prompt, for the routing
    /// trace (a streamed prefill is one trace call across its chunks).
    trace_row_base: usize,
    /// The streamed prefill chunk this call runs (0 otherwise), for the
    /// routing trace.
    trace_chunk: usize,
    /// This session's prefill time split (see `ModelWeights::prefill_timing`).
    prefill_timing: Arc<PhaseTiming>,
}

/// Whether an environment flag is set to something other than off
/// (`1`, `true`, `on`, … — not `0`, `false`, `off`, `no` or empty).
fn env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "off" | "no"
        ),
        Err(_) => false,
    }
}

/// The block-quantized tensor behind a `QMatMul`, when it kept one.
fn qtensor_of(m: &QMatMul) -> Option<Arc<QTensor>> {
    match m {
        QMatMul::QTensor(qt) => Some(Arc::clone(qt)),
        QMatMul::Tensor(_) | QMatMul::TensorF16(_) => None,
    }
}

/// Build the device expert pool over every layer's routed experts.
///
/// `None` (with a log line) when the experts are not block-quantized
/// tensors (a streamed load decoded them to f32) or the budget holds no
/// expert.  On a CPU "device" (tests, `--device cpu` with a budget) a slot
/// shares the host tensors — a hit is numerically the host path — so the
/// partition can be exercised without an accelerator.
fn build_device_pool(
    layers: &[Layer],
    dev: &Device,
    budget: u64,
    file: Option<Arc<std::fs::File>>,
) -> Option<Arc<Ds4DevicePool>> {
    let mut host: Vec<Vec<HostExpert>> = Vec::with_capacity(layers.len());
    for layer in layers {
        // Every layer of this architecture carries an MoE block.
        let FeedForward::Moe(moe) = &layer.ffn;
        let mut row = Vec::with_capacity(moe.experts.len());
        for m in &moe.experts {
            let (Some(gate), Some(up), Some(down)) =
                (qtensor_of(&m.gate), qtensor_of(&m.up), qtensor_of(&m.down))
            else {
                tracing::warn!(
                    "deepseek4: the routed experts are not block-quantized tensors (streamed load); \
                     VRAM expert cache disabled"
                );
                return None;
            };
            let bytes = (gate.storage_size_in_bytes()
                + up.storage_size_in_bytes()
                + down.storage_size_in_bytes()) as u64;
            row.push(HostExpert {
                gate,
                up,
                down,
                clamp: m.clamp,
                prefetch: m.prefetch.clone(),
                bytes,
            });
        }
        host.push(row);
    }
    let per_slot = host.first().and_then(|r| r.first()).map_or(0, |h| h.bytes);
    let n_experts: usize = host.iter().map(Vec::len).sum();
    if per_slot == 0 || n_experts == 0 {
        return None;
    }
    let slots = (budget / per_slot) as usize;
    if slots == 0 {
        tracing::warn!(
            "deepseek4: VRAM expert cache budget of {:.1} MiB holds no expert ({:.2} MiB each); cache disabled",
            budget as f64 / 2f64.powi(20),
            per_slot as f64 / 2f64.powi(20),
        );
        return None;
    }
    // Per-expert prefetch handles, for the host-page release hook below.
    let host_handles: Arc<Vec<Vec<Option<crate::residency::ExpertHandles>>>> = Arc::new(
        host.iter()
            .map(|row| row.iter().map(|h| h.prefetch.clone()).collect())
            .collect(),
    );
    let host = Arc::new(host);
    let share_host = dev.is_cpu();
    let dev2 = dev.clone();
    let upload_one = move |l: u32, e: u32| -> Result<Option<Ds4DeviceExpert>> {
        let Some(ex) = host.get(l as usize).and_then(|row| row.get(e as usize)) else {
            return Ok(None);
        };
        if share_host {
            return Ok(Some(Ds4DeviceExpert {
                gate: QMatMul::from_arc(Arc::clone(&ex.gate))?,
                up: QMatMul::from_arc(Arc::clone(&ex.up))?,
                down: QMatMul::from_arc(Arc::clone(&ex.down))?,
                clamp: ex.clamp,
                bytes: ex.bytes,
            }));
        }
        // Advise the three ranges in before the blocking write so it does
        // not page-fault its way through the mapping mid-transfer.
        if let Some(p) = &ex.prefetch {
            p.prefetch();
        }
        let upload = |qt: &Arc<QTensor>| -> Result<QMatMul> {
            // `data()` borrows the mapped bytes for a CPU storage: no host copy.
            let bytes = qt.data()?;
            let storage = QStorage::from_data_transfer(bytes, &dev2, qt.dtype())?;
            QMatMul::from_qtensor(QTensor::new(storage, qt.shape().clone())?)
        };
        Ok(Some(Ds4DeviceExpert {
            gate: upload(&ex.gate)?,
            up: upload(&ex.up)?,
            down: upload(&ex.down)?,
            clamp: ex.clamp,
            bytes: ex.bytes,
        }))
    };
    let upload: Arc<dyn Fn(u32, u32) -> Option<Arc<Ds4DeviceExpert>> + Send + Sync> =
        Arc::new(move |l, e| match upload_one(l, e) {
            Ok(slot) => slot.map(Arc::new),
            Err(err) => {
                tracing::warn!(
                    "deepseek4: expert ({l}, {e}) upload to the VRAM cache failed: {err}"
                );
                None
            }
        });
    let pool = Arc::new(crate::residency::DeviceResidency::new(
        budget, per_slot, upload,
    ));
    // On a real device the host pages of an uploaded expert are released
    // (exclusive tiers).  With the CPU as the "device" the slot *is* the
    // host tensor, so there is nothing to release.
    let host_pages = if share_host {
        HostPagePolicy::Keep
    } else {
        HostPagePolicy::from_env()
    };
    let busy: Arc<BusyExperts> = Default::default();
    let release: Option<crate::residency::ReleaseHook> = match host_pages {
        HostPagePolicy::Drop => {
            let host = Arc::clone(&host_handles);
            let busy = Arc::clone(&busy);
            Some(Arc::new(move |l: u32, e: u32| {
                // Not while the host kernels are reading it: the uploader
                // retries after another delay.
                if busy
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .contains_key(&(l, e))
                {
                    return false;
                }
                if let Some(Some(h)) = host.get(l as usize).and_then(|row| row.get(e as usize)) {
                    h.release_host_pages(file.as_deref());
                }
                true
            }))
        }
        HostPagePolicy::Keep => None,
    };
    let uploader = Arc::new(crate::residency::ExpertUploader::spawn_with_release(
        Arc::clone(&pool),
        release,
        crate::residency::HOST_RELEASE_DELAY,
    ));
    let miss = MissPolicy::from_env();
    tracing::info!(
        "deepseek4: VRAM expert cache on {dev:?}: {:.2} GiB budget = {slots} slots of {:.2} MiB \
         ({n_experts} routed experts in the model, {:.1}% resident at most); decode misses run on the {}; \
         host pages of uploaded experts are {}",
        budget as f64 / 2f64.powi(30),
        per_slot as f64 / 2f64.powi(20),
        100.0 * slots.min(n_experts) as f64 / n_experts as f64,
        match miss {
            MissPolicy::Host => "host and are uploaded in the background",
            MissPolicy::Upload => "device after a synchronous upload (JOSHUA_EXPERT_MISS=upload)",
        },
        match host_pages {
            HostPagePolicy::Drop => "released (RAM and VRAM hold different experts)",
            HostPagePolicy::Keep => "kept (JOSHUA_EXPERT_HOST_PAGES=keep)",
        },
    );
    let probe_pages = env_flag("JOSHUA_EXPERT_STATS")
        || tracing::enabled!(target: "joshua", tracing::Level::DEBUG);
    Some(Arc::new(Ds4DevicePool {
        device: dev.clone(),
        pool,
        decode: PhaseTiming::default(),
        probe_pages,
        uploader,
        miss,
        host_pages,
        busy,
    }))
}

// ─── HC (hyper-computation) stream mixing ───────────────────────────────────

struct HcMix {
    pre: Tensor,  // [b, s, hc]
    post: Tensor, // [b, s, hc]
    comb: Tensor, // [b, s, hc, hc]
}

/// Split the mixes row `[hc, 2*hc, hc*hc]` into pre/post/comb and run the
/// Sinkhorn normalization on the comb block, matching the reference
/// `hc_split_sinkhorn` kernel:
///   softmax over k (last dim), column-normalize over j (dim 1),
///   then `iters - 1` rounds of row (dim 2) then column (dim 1) normalization.
fn hc_split_sinkhorn(
    mixes: &Tensor,
    scale: &Tensor,
    base: &Tensor,
    hc: usize,
    iters: usize,
    eps: f64,
) -> Result<HcMix> {
    let shape = mixes.shape().dims().to_vec();
    let n = shape[..shape.len() - 1].iter().product::<usize>();
    let m = mixes.reshape((n, (2 + hc) * hc))?;

    let s: Vec<f32> = scale.flatten_all()?.to_vec1()?;
    if s.len() < 3 {
        candle_core::bail!(
            "deepseek4: hc scale tensor has {} entries, expected 3",
            s.len()
        );
    }

    let pre = sigmoid(
        &m.narrow(D::Minus1, 0, hc)?
            .affine(s[0] as f64, 0.0)?
            .broadcast_add(&base.narrow(0, 0, hc)?)?,
    )?
    .affine(1.0, eps)?;
    let post = (sigmoid(
        &m.narrow(D::Minus1, hc, hc)?
            .affine(s[1] as f64, 0.0)?
            .broadcast_add(&base.narrow(0, hc, hc)?)?,
    )? * 2.0)?;
    let comb0 = m
        .narrow(D::Minus1, 2 * hc, hc * hc)?
        .affine(s[2] as f64, 0.0)?
        .broadcast_add(&base.narrow(0, 2 * hc, hc * hc)?)?;
    let comb0 = comb0.reshape((n, hc, hc))?;

    let mut comb = softmax_last_dim(&comb0)?.affine(1.0, eps)?;
    comb = comb.broadcast_div(&comb.sum_keepdim(1)?.affine(1.0, eps)?)?;
    for _ in 0..iters.saturating_sub(1) {
        comb = comb.broadcast_div(&comb.sum_keepdim(2)?.affine(1.0, eps)?)?;
        comb = comb.broadcast_div(&comb.sum_keepdim(1)?.affine(1.0, eps)?)?;
    }

    let mut pre_shape = shape[..shape.len() - 1].to_vec();
    pre_shape.push(hc);
    let pre = pre.reshape(pre_shape)?;
    let mut post_shape = shape[..shape.len() - 1].to_vec();
    post_shape.push(hc);
    let post = post.reshape(post_shape)?;
    let mut comb_shape = shape[..shape.len() - 1].to_vec();
    comb_shape.push(hc);
    comb_shape.push(hc);
    let comb = comb.reshape(comb_shape)?;
    Ok(HcMix { pre, post, comb })
}

/// `hc_pre`: mix `hc` copies down to one stream.
/// `x`: `[b, s, hc, d]`, `hc_fn`: `[hc_dim, mix_hc]`, `hc_scale`: `[3]`,
/// `hc_base`: `[mix_hc]`, eps from config.
fn hc_pre(
    x: &Tensor,
    hc_fn: &QMatMul,
    hc_scale: &Tensor,
    hc_base: &Tensor,
    eps: f64,
    sinkhorn_iters: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let shape = x.shape().dims().to_vec(); // [b, s, hc, d]
    let (b, s, hc, d) = (shape[0], shape[1], shape[2], shape[3]);
    let flat = x.reshape((b * s, hc * d))?;
    let rsqrt = flat
        .sqr()?
        .mean_keepdim(D::Minus1)?
        .affine(1.0, eps)?
        .powf(-0.5)?;
    let mixes = hc_fn.forward(&flat)?.broadcast_mul(&rsqrt)?; // [b*s, (2+hc)*hc]
    let mixes = mixes.reshape((b, s, (2 + hc) * hc))?;
    let mix = hc_split_sinkhorn(&mixes, hc_scale, hc_base, hc, sinkhorn_iters, eps)?;
    // y = sum over hc of pre[..., h] * x[..., h, :]
    let y = mix
        .pre
        .unsqueeze(D::Minus1)?
        .broadcast_as((b, s, hc, d))?
        .mul(x)?
        .sum(D::Minus2)?; // [b, s, d]
    Ok((y, mix.post, mix.comb))
}

/// `hc_post`: expand one stream back to `hc` copies.
/// `x`: `[b, s, d]`, `residual`: `[b, s, hc, d]`, `post`: `[b, s, hc]`,
/// `comb`: `[b, s, hc, hc]`.
fn hc_post(x: &Tensor, residual: &Tensor, post: &Tensor, comb: &Tensor) -> Result<Tensor> {
    let shape = residual.shape().dims().to_vec(); // [b, s, hc, d]
    let (b, s, hc, d) = (shape[0], shape[1], shape[2], shape[3]);
    let post_t = post.unsqueeze(D::Minus1)?.broadcast_as((b, s, hc, d))?;
    let comb_t = comb.unsqueeze(D::Minus1)?.broadcast_as((b, s, hc, hc, d))?;
    let x_t = x.unsqueeze(2)?.broadcast_as((b, s, hc, d))?;
    let comb_src =
        (comb_t * residual.unsqueeze(2)?.broadcast_as((b, s, hc, hc, d))?)?.sum(D::Minus2)?; // [b, s, hc, d]
    (post_t * x_t)?.add(&comb_src)
}

// ─── Quantized-rotation helpers ─────────────────────────────────────────────

/// In-place fast Walsh–Hadamard transform (unnormalized).
fn fast_hadamard(data: &mut [f32]) {
    let mut h = 1usize;
    let n = data.len();
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let (a, b) = (data[j], data[j + h]);
                data[j] = a + b;
                data[j + h] = a - b;
            }
        }
        h *= 2;
    }
}

/// Hadamard rotation applied to the trailing dim of a tensor of any rank; the
/// input shape is preserved.  The trailing dim must be a power of two, which is
/// what the fast Walsh–Hadamard transform requires.
fn hadamard_rows(x: &Tensor) -> Result<Tensor> {
    let shape = x.shape().dims().to_vec();
    let d = *shape
        .last()
        .ok_or_else(|| candle_core::Error::Msg("deepseek4: hadamard on a rank-0 tensor".into()))?;
    if !d.is_power_of_two() {
        candle_core::bail!("deepseek4: Hadamard rotation needs a power-of-two head dim, got {d}");
    }
    let n = shape[..shape.len() - 1].iter().product::<usize>();
    let data: Vec<f32> = x.flatten_all()?.to_vec1()?;
    let mut out = vec![0f32; data.len()];
    let scale = 1.0 / (d as f32).sqrt();
    for (row, chunk) in data.chunks_exact(d).enumerate() {
        let mut r = chunk.to_vec();
        fast_hadamard(&mut r);
        for (o, v) in out[row * d..row * d + d].iter_mut().zip(r.iter()) {
            *o = v * scale;
        }
    }
    Tensor::from_vec(out, (n, d), x.device())?.reshape(shape)
}

struct Compressor {
    wkv: QMatMul,
    wgate: QMatMul,
    ape: Tensor, // [ratio, coff*head_dim] (GGUF stores it as [coff*head_dim, ratio])
    norm: RmsNorm,
    ratio: usize,
    coff: usize, // 2 for CSA (overlap), 1 for HCA
    head_dim: usize,
    rope_head_dim: usize,
    rotate: bool, // indexer compressor applies the Hadamard rotation
}

/// Streaming state of a compressor: rows of the current (partial) block plus,
/// for CSA, the previous block kept for the overlap window.
struct CompressorState {
    kv: Tensor,    // [coff*ratio, coff*head_dim]
    score: Tensor, // same, -inf initialized
}

impl CompressorState {
    fn new(ratio: usize, coff: usize, head_dim: usize, dev: &Device) -> Result<Self> {
        Ok(Self {
            kv: Tensor::zeros((coff * ratio, coff * head_dim), DType::F32, dev)?,
            score: Tensor::full(f32::NEG_INFINITY, (coff * ratio, coff * head_dim), dev)?,
        })
    }
}

impl Compressor {
    fn load<R: Read + Seek>(
        rd: &mut Reader<R>,
        prefix: &str, // e.g. "blk.0.attn_compressor" or "blk.0.indexer_compressor"
        ratio: usize,
        head_dim: usize,
        rope_head_dim: usize,
        eps: f64,
        rotate: bool,
    ) -> Result<Self> {
        debug_assert!(ratio == CSA_RATIO || ratio == HCA_RATIO);
        let coff = if ratio == CSA_RATIO { 2 } else { 1 };
        Ok(Self {
            wkv: rd.qmatmul(&format!("{prefix}_kv.weight"))?,
            wgate: rd.qmatmul(&format!("{prefix}_gate.weight"))?,
            ape: rd.f32_tensor(&format!("{prefix}_ape.weight"))?, // [ratio, coff*head_dim] (gguf reader reverses dims)
            norm: rd.rms_norm(&format!("{prefix}_norm.weight"), eps)?,
            ratio,
            coff,
            head_dim,
            rope_head_dim,
            rotate,
        })
    }

    /// Compress `kv`/`score` rows for tokens `[offset, offset+seq)` and update
    /// the streaming state.  Returns the newly produced compressed rows
    /// `[n, head_dim]` and their block positions `[n]` (u32), or `None` when no
    /// block boundary was crossed.  Follows the official modeling code: prefill
    /// compresses whole blocks at once, decode accumulates per token; both
    /// branches push `[n, head_dim]` chunks so the tail can concatenate them.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        kv: &Tensor,    // [seq, coff*head_dim]
        score: &Tensor, // [seq, coff*head_dim]
        state: &mut CompressorState,
        offset: usize,
        seq: usize,
        rotary: &RotaryEmbedding,
        dev: &Device,
    ) -> Result<Option<(Tensor, Tensor)>> {
        let ratio = self.ratio;
        let coff = self.coff;
        let hd = self.head_dim;
        let mut rows: Vec<Tensor> = Vec::new();
        let mut poss: Vec<u32> = Vec::new();

        if offset == 0 {
            let cutoff = seq - seq % ratio;
            let rem = seq % ratio;
            let nb = cutoff / ratio;
            if coff == 2 && cutoff >= ratio {
                // Previous window = the last full block of this prompt.
                let prev = kv.narrow(0, cutoff - ratio, ratio)?;
                let prev_s = (score.narrow(0, cutoff - ratio, ratio)? + &self.ape)?;
                state.kv = state.kv.slice_scatter(&prev, 0, 0)?;
                state.score = state.score.slice_scatter(&prev_s, 0, 0)?;
            }
            if rem > 0 {
                let kv_rest = kv.narrow(0, cutoff, rem)?;
                let sc_rest = (score.narrow(0, cutoff, rem)? + self.ape.narrow(0, 0, rem)?)?;
                let at = if coff == 2 { ratio } else { 0 };
                state.kv = state.kv.slice_scatter(&kv_rest, 0, at)?;
                state.score = state.score.slice_scatter(&sc_rest, 0, at)?;
            }
            if nb > 0 {
                let bkv = kv.narrow(0, 0, cutoff)?.reshape((nb, ratio, coff * hd))?;
                let bsc = score
                    .narrow(0, 0, cutoff)?
                    .reshape((nb, ratio, coff * hd))?
                    .broadcast_add(&self.ape.unsqueeze(0)?)?;
                let (bkv, bsc) = if coff == 2 {
                    // Overlap transform: the pooled window of block b spans
                    // `2*ratio` tokens — block b-1 seen through its
                    // previous-window half, then block b through its own half.
                    // Block 0 has no previous window (zero kv / -inf scores, so
                    // those rows contribute nothing).
                    let own = bkv.narrow(2, hd, hd)?;
                    let own_s = bsc.narrow(2, hd, hd)?;
                    let prev = bkv.narrow(2, 0, hd)?;
                    let prev_s = bsc.narrow(2, 0, hd)?;
                    let zero = Tensor::zeros((1, ratio, hd), DType::F32, dev)?;
                    let zero_s = Tensor::full(f32::NEG_INFINITY, (1, ratio, hd), dev)?;
                    let prev = Tensor::cat(&[zero, prev.narrow(0, 0, nb - 1)?], 0)?;
                    let prev_s = Tensor::cat(&[zero_s, prev_s.narrow(0, 0, nb - 1)?], 0)?;
                    (
                        Tensor::cat(&[prev, own], 1)?, // [nb, 2*ratio, hd]
                        Tensor::cat(&[prev_s, own_s], 1)?,
                    )
                } else {
                    (bkv, bsc)
                };
                let w = softmax(&bsc, 1)?;
                rows.push((bkv * w)?.sum(1)?); // [nb, hd]
                poss.extend(0..nb as u32);
            }
        } else {
            for i in 0..seq {
                let pos = offset + i;
                let k = kv.narrow(0, i, 1)?.reshape((coff * hd,))?;
                let s = (score.narrow(0, i, 1)?.reshape((coff * hd,))?
                    + self.ape.narrow(0, pos % ratio, 1)?.reshape((coff * hd,))?)?;
                if coff == 2 {
                    state.kv = state
                        .kv
                        .slice_scatter(&k.unsqueeze(0)?, 0, ratio + pos % ratio)?;
                    state.score =
                        state
                            .score
                            .slice_scatter(&s.unsqueeze(0)?, 0, ratio + pos % ratio)?;
                    if (pos + 1).is_multiple_of(ratio) {
                        // [2*ratio, hd]: the previous block through its
                        // previous-window half, then this block through its own.
                        let kv_state = Tensor::cat(
                            &[
                                state.kv.narrow(0, 0, ratio)?.narrow(1, 0, hd)?,
                                state.kv.narrow(0, ratio, ratio)?.narrow(1, hd, hd)?,
                            ],
                            0,
                        )?;
                        let sc_state = Tensor::cat(
                            &[
                                state.score.narrow(0, 0, ratio)?.narrow(1, 0, hd)?,
                                state.score.narrow(0, ratio, ratio)?.narrow(1, hd, hd)?,
                            ],
                            0,
                        )?;
                        let w = softmax(&sc_state, 0)?;
                        rows.push((kv_state * w)?.sum_keepdim(0)?); // [1, hd]
                        poss.push((pos / ratio) as u32);
                        // slide the window: current block becomes the previous one
                        state.kv = state.kv.slice_scatter(
                            &state.kv.narrow(0, ratio, ratio)?.contiguous()?,
                            0,
                            0,
                        )?;
                        state.score = state.score.slice_scatter(
                            &state.score.narrow(0, ratio, ratio)?.contiguous()?,
                            0,
                            0,
                        )?;
                    }
                } else {
                    state.kv = state.kv.slice_scatter(&k.unsqueeze(0)?, 0, pos % ratio)?;
                    state.score = state
                        .score
                        .slice_scatter(&s.unsqueeze(0)?, 0, pos % ratio)?;
                    if (pos + 1).is_multiple_of(ratio) {
                        let w = softmax(&state.score, 0)?;
                        rows.push(state.kv.mul(&w)?.sum_keepdim(0)?); // [1, hd]
                        poss.push((pos / ratio) as u32);
                    }
                }
            }
        }

        if rows.is_empty() {
            return Ok(None);
        }
        let mut comp = Tensor::cat(&rows, 0)?; // [n, hd]
        comp = self.norm.forward(&comp)?;
        // RoPE on the trailing rope dims at the compressed positions.
        let rd = self.rope_head_dim;
        let poss_t = Tensor::from_vec(poss, comp.dim(0)?, dev)?;
        // Rotate at the TOKEN position where each block starts (b*ratio), which
        // is what llama passes as `comp_pos` and the python uses via
        // `freqs_cis[:cutoff:ratio]`.
        let poss_tok = Tensor::from_vec(
            poss_t
                .to_vec1::<u32>()?
                .iter()
                .map(|b| b * self.ratio as u32)
                .collect(),
            comp.dim(0)?,
            dev,
        )?;
        let nope = comp.narrow(D::Minus1, 0, hd - rd)?;
        let pe = comp.narrow(D::Minus1, hd - rd, rd)?;
        let pe = rotary.apply_at(&pe, &poss_tok)?;
        let comp = Tensor::cat(&[nope, pe], D::Minus1)?;
        // The indexer stores its compressed KV Hadamard-rotated, matching the
        // official `rotate_activation`; the indexer query is rotated the same
        // way, so the relative scores are unchanged but QAT-aligned.
        let comp = if self.rotate {
            hadamard_rows(&comp)?
        } else {
            comp
        };
        Ok(Some((comp, poss_t)))
    }
}

/// Apply (or invert) RoPE on the trailing `rope_dim` slice of a
/// `[1, heads, seq, dim]` tensor, leaving the leading `nope_dim` slice alone.
fn rope_apply(
    rotary: &RotaryEmbedding,
    x4: &Tensor,
    nope_dim: usize,
    rope_dim: usize,
    offset: usize,
    back: bool,
) -> Result<Tensor> {
    let seq = x4.dim(2)?;
    let nope = x4.narrow(3, 0, nope_dim)?;
    let pe = x4.narrow(3, nope_dim, rope_dim)?;
    let pe = if back {
        rotary.apply_back(&pe, offset, seq)?
    } else {
        rotary.apply(&pe, offset, seq)?
    };
    Tensor::cat(&[nope, pe], 3)
}

struct Indexer {
    proj: QMatMul,          // [n_embd, index_n_head]
    attn_q_b: QMatMul,      // [q_lora_rank, index_n_head * index_head_dim]
    compressor: Compressor, // ratio 4, head_dim = index_head_dim, rotate
    n_head: usize,
    head_dim: usize,
    rope_head_dim: usize,
    nope_head_dim: usize,
    softmax_scale: f64,
    topk: usize,
}

impl Indexer {
    /// Score all compressed KV rows with the lightning indexer and return the
    /// top-`index_topk` block indices per query: `[seq, k]` u32.
    fn score(
        &self,
        x: &Tensor,  // [1, seq, n_embd]
        qr: &Tensor, // [1, seq, q_lora_rank]
        offset: usize,
        seq: usize,
        lid: &Tensor, // [max_blocks, index_head_dim]
        rotary: &RotaryEmbedding,
    ) -> Result<Tensor> {
        let dev = x.device();
        let n_lid = (offset + seq) / self.compressor.ratio;
        if n_lid == 0 {
            // No compressed KV rows yet at this position.  An empty `[seq, 0]`
            // tensor is only ever read back on the host in `build_kv` (dims +
            // flatten), so allocate on CPU: on CUDA, `cuMemAlloc(0)` inside
            // `Tensor::zeros` fails with CUDA_ERROR_INVALID_VALUE.
            return Tensor::zeros((seq, 0), DType::U32, &Device::Cpu);
        }
        let q = self
            .attn_q_b
            .forward(qr)?
            .reshape((seq, self.n_head, self.head_dim))?;
        let q4 = q.unsqueeze(0)?.transpose(1, 2)?; // [1, ih, seq, ihd]
        let q4 = rope_apply(
            rotary,
            &q4,
            self.nope_head_dim,
            self.rope_head_dim,
            offset,
            false,
        )?;
        let q = q4.transpose(1, 2)?.squeeze(0)?.contiguous()?; // [seq, ih, ihd]
                                                               // The official model rotates the indexer query with the Hadamard matrix
                                                               // (same one the compressor applied to the lid rows); llama.cpp does the
                                                               // same with `k_rot`.  Applied to both sides it is an exact orthogonal
                                                               // change of basis, so this only matters for QAT/quantization alignment.
        let q = hadamard_rows(&q)?;

        let k = lid.narrow(0, 0, n_lid)?; // [n_lid, ihd]
        let kt = k
            .t()?
            .unsqueeze(0)?
            .broadcast_as((seq, self.head_dim, n_lid))?;
        let sc = q.matmul(&kt)?; // [seq, ih, n_lid]
        let sc = sc.relu()?;
        let w = self.proj.forward(x)?.squeeze(0)?; // [seq, ih]
        let w = ((w * self.softmax_scale)? * (self.n_head as f64).powf(-0.5))?;
        let sc = sc.broadcast_mul(&w.unsqueeze(D::Minus1)?)?.sum(1)?; // [seq, n_lid]

        // Causal mask: block b is visible to the query at absolute pos p iff
        // b < (p+1)/ratio.
        let mut vals = vec![f32::NEG_INFINITY; seq * n_lid];
        for r in 0..seq {
            let nv = (offset + r + 1) / self.compressor.ratio;
            for b in 0..nv.min(n_lid) {
                vals[r * n_lid + b] = 0.0;
            }
        }
        let mask = Tensor::from_vec(vals, (seq, n_lid), dev)?;
        let sc = (sc + mask)?;
        topk_indices(&sc, self.topk.min(n_lid))
    }
}

struct Attention {
    q_a: QMatMul,
    q_norm: RmsNorm,
    q_b: QMatMul,
    wkv: QMatMul,
    kv_norm: RmsNorm,
    wo_a: QMatMul,
    wo_b: QMatMul,
    attn_sinks: Tensor, // [n_head]
    compressor: Option<Compressor>,
    indexer: Option<Indexer>,
    ratio: usize,
    n_head: usize,
    head_dim: usize,
    rope_head_dim: usize,
    nope_head_dim: usize,
    o_groups: usize,
    o_lora_rank: usize,
    window_size: usize,
    softmax_scale: f64,
    eps: f64,
    rotary: Arc<RotaryEmbedding>,
    compress_rotary: Option<Arc<RotaryEmbedding>>,
}

impl Attention {
    /// RoPE the trailing slice of a `[seq, n_head, head_dim]` tensor.
    fn rope_qkv(&self, q: &Tensor, offset: usize) -> Result<Tensor> {
        let q4 = q.unsqueeze(0)?.transpose(1, 2)?; // [1, h, seq, d]
        let q4 = rope_apply(
            &self.rotary,
            &q4,
            self.nope_head_dim,
            self.rope_head_dim,
            offset,
            false,
        )?;
        q4.transpose(1, 2)?.squeeze(0)
    }

    /// Gather the visible key/value rows (sliding window + compressed blocks)
    /// for every query and return `(k_all, mask)` with shapes
    /// `[seq, n_kv, head_dim]` and `[seq, n_kv]` (0 for valid rows, -inf for
    /// masked-out padding).
    ///
    /// `swa_prev` is the ring cache *before* this chunk was written and
    /// `kv_raw` the keys of the chunk itself: the two are concatenated into a
    /// dense, position-ordered buffer covering tokens
    /// `[offset - window_size, offset + seq)` so that a prompt longer than the
    /// window cannot alias its own history through the ring.
    fn build_kv(
        &self,
        kv: &KvState,
        swa_prev: &Tensor,
        kv_raw: &Tensor,
        offset: usize,
        seq: usize,
        lid_idx: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let win = self.window_size;
        let ratio = self.ratio;
        let d = self.head_dim;
        let dev = kv_raw.device();

        // Dense window source: row j holds token `base + j`, with the first
        // `win` rows read out of the ring in absolute-position order.  Rows for
        // tokens before position 0 hold stale data but are always masked out.
        let base = offset as i64 - win as i64;
        let ring_rows: Vec<u32> = (0..win)
            .map(|j| (base + j as i64).rem_euclid(win as i64) as u32)
            .collect();
        let hist = swa_prev.index_select(&Tensor::from_vec(ring_rows, (win,), dev)?, 0)?;
        let k_src = Tensor::cat(&[hist, kv_raw.clone()], 0)?; // [win + seq, d]

        // Window rows: token t of query p is valid iff
        // max(0, p+1-win) <= t <= p.
        let mut w_rows = vec![0u32; seq * win];
        let mut w_mask = vec![f32::NEG_INFINITY; seq * win];
        for r in 0..seq {
            let p = offset + r;
            let lo = (p + 1).saturating_sub(win);
            let cnt = p + 1 - lo;
            for j in 0..cnt {
                w_rows[r * win + j] = ((lo + j) as i64 - base) as u32;
                w_mask[r * win + j] = 0.0;
            }
        }

        // Compressed rows: either the indexer's top-k blocks (CSA) or every
        // visible block (HCA).
        let n_comp = match lid_idx {
            Some(idx) => idx.dim(1)?,
            None => {
                if ratio > 0 {
                    (offset + seq).div_ceil(ratio)
                } else {
                    0
                }
            }
        };
        let mut c_rows = vec![0u32; seq * n_comp];
        let mut c_mask = vec![f32::NEG_INFINITY; seq * n_comp];
        #[allow(clippy::manual_checked_ops)] // ratio==0 is guarded; the div is intentional
        if ratio > 0 {
            let lid_vals: Option<Vec<u32>> = match lid_idx {
                Some(idx) => Some(idx.flatten_all()?.to_vec1()?),
                None => None,
            };
            for r in 0..seq {
                let nv = (offset + r + 1) / ratio;
                match &lid_vals {
                    Some(vals) => {
                        for j in 0..n_comp {
                            let b = vals[r * n_comp + j] as usize;
                            if b < nv {
                                c_rows[r * n_comp + j] = b as u32;
                                c_mask[r * n_comp + j] = 0.0;
                            }
                        }
                    }
                    None => {
                        for b in 0..nv.min(n_comp) {
                            c_rows[r * n_comp + b] = b as u32;
                            c_mask[r * n_comp + b] = 0.0;
                        }
                    }
                }
            }
        }

        let k_win = k_src
            .index_select(&Tensor::from_vec(w_rows, (seq * win,), dev)?, 0)?
            .reshape((seq, win, d))?;
        // A ratio-0 (sliding-window-only) layer has no compressed keys, so
        // `n_comp` is 0.  Allocating `(seq, 0, d)` as a separate empty tensor
        // is fine on the CPU backend but on CUDA it asks `cuMemAlloc(0)`,
        // which fails with CUDA_ERROR_INVALID_VALUE.  Build the key/mask pair
        // directly from the window rows in that case instead.
        let k_all;
        let mask;
        if n_comp > 0 {
            let c = kv.comp.as_ref().unwrap();
            let k_comp = c
                .index_select(&Tensor::from_vec(c_rows, (seq * n_comp,), dev)?, 0)?
                .reshape((seq, n_comp, d))?;
            k_all = Tensor::cat(&[k_win, k_comp], 1)?;
            mask = Tensor::cat(
                &[
                    Tensor::from_vec(w_mask, (seq, win), dev)?,
                    Tensor::from_vec(c_mask, (seq, n_comp), dev)?,
                ],
                1,
            )?;
        } else {
            k_all = k_win;
            mask = Tensor::from_vec(w_mask, (seq, win), dev)?;
        }
        Ok((k_all, mask))
    }

    fn forward(
        &self,
        kv: &mut KvState,
        x: &Tensor,
        offset: usize,
        max_seq: usize,
    ) -> Result<Tensor> {
        let dev = x.device();
        let seq = x.dim(1)?;
        if offset + seq > max_seq {
            candle_core::bail!(
                "deepseek4: positions {}..{} exceed the KV capacity ({max_seq}); rebuild with a larger context",
                offset,
                offset + seq
            );
        }

        // Q: [seq, n_head, head_dim], RMS per head, then RoPE.
        let qr = self.q_norm.forward(&self.q_a.forward(x)?)?; // [1, seq, q_lora_rank]
        let q = self
            .q_b
            .forward(&qr)?
            .reshape((seq, self.n_head, self.head_dim))?;
        let m = q.sqr()?.mean_keepdim(D::Minus1)?.affine(1.0, self.eps)?;
        let q = q.broadcast_div(&m.sqrt()?)?.contiguous()?;
        let q = self.rope_qkv(&q, offset)?.contiguous()?;

        // Raw KV row (MLA has a single KV head), RoPE on the trailing slice.
        let kv_raw = self.kv_norm.forward(&self.wkv.forward(x)?)?.squeeze(0)?; // [seq, head_dim]
        let kv4 = kv_raw.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, seq, d]
        let kv4 = rope_apply(
            &self.rotary,
            &kv4,
            self.nope_head_dim,
            self.rope_head_dim,
            offset,
            false,
        )?;
        let kv_raw = kv4.squeeze(0)?.squeeze(0)?.contiguous()?; // [seq, d]

        // Write into the sliding-window ring, keeping the pre-chunk contents so
        // the window rows of this chunk can be read from dense positions.  A
        // chunk at least as long as the window overwrites the ring completely,
        // in which case it is rebuilt from the last `window_size` keys (a
        // scatter with repeated ring slots would leave an arbitrary winner).
        let win = self.window_size;
        let swa_prev = kv.swa.as_ref().unwrap().clone();
        kv.swa = Some(if seq >= win {
            let start = offset + seq - win;
            let rows: Vec<u32> = (0..win)
                .map(|slot| {
                    let t = start + (slot + win - start % win) % win;
                    (t - offset) as u32
                })
                .collect();
            kv_raw.index_select(&Tensor::from_vec(rows, (win,), dev)?, 0)?
        } else {
            let ring: Vec<u32> = (offset..offset + seq).map(|p| (p % win) as u32).collect();
            // Broadcast index views are not contiguous; scatter requires a
            // dense index tensor (short prompts land here).
            let ridx = Tensor::from_vec(ring, (seq, 1), dev)?
                .broadcast_as((seq, self.head_dim))?
                .contiguous()?;
            swa_prev.scatter(&ridx, &kv_raw, 0)?
        });

        // Main compressor (CSA / HCA).
        if let Some(c) = &self.compressor {
            let ckv = c.wkv.forward(x)?.squeeze(0)?; // [seq, coff*head_dim]
            let csc = c.wgate.forward(x)?.squeeze(0)?;
            let out = c.forward(
                &ckv,
                &csc,
                kv.comp_state.as_mut().unwrap(),
                offset,
                seq,
                self.compress_rotary.as_ref().unwrap(),
                dev,
            )?;
            if let Some((rows, poss)) = &out {
                let comp = kv.comp.as_ref().unwrap();
                // Broadcast index views are not contiguous; scatter requires
                // a dense index tensor (same as the sliding-window ring
                // above), so materialize it before writing the cache.
                let pidx = poss
                    .unsqueeze(1)?
                    .broadcast_as((rows.dim(0)?, self.head_dim))?
                    .contiguous()?;
                kv.comp = Some(comp.scatter(&pidx, rows, 0)?);
            }
        }

        // Indexer (CSA only): compress into the lid cache, then pick top-k
        // block indices per query.
        let lid_idx: Option<Tensor> = if let Some(ix) = &self.indexer {
            let lkv = ix.compressor.wkv.forward(x)?.squeeze(0)?;
            let lsc = ix.compressor.wgate.forward(x)?.squeeze(0)?;
            let lout = ix.compressor.forward(
                &lkv,
                &lsc,
                kv.lid_state.as_mut().unwrap(),
                offset,
                seq,
                self.compress_rotary.as_ref().unwrap(),
                dev,
            )?;
            if let Some((rows, poss)) = &lout {
                let lid = kv.lid.as_ref().unwrap();
                // Same contiguous-index requirement as the compressor cache:
                // a broadcast view of the position list is not a valid
                // scatter index.
                let pidx = poss
                    .unsqueeze(1)?
                    .broadcast_as((rows.dim(0)?, ix.head_dim))?
                    .contiguous()?;
                kv.lid = Some(lid.scatter(&pidx, rows, 0)?);
            }
            Some(ix.score(x, &qr, offset, seq, kv.lid.as_ref().unwrap(), &self.rotary)?)
        } else {
            None
        };

        // Sparse attention with per-head sink.
        let (k_all, mask) = self.build_kv(kv, &swa_prev, &kv_raw, offset, seq, lid_idx.as_ref())?;
        let scores = q.matmul(&k_all.transpose(1, 2)?)?; // [seq, h, n_kv]
        let scores = (scores * self.softmax_scale)?.broadcast_add(&mask.unsqueeze(1)?)?;
        let max = scores.max_keepdim(D::Minus1)?;
        let e = scores.broadcast_sub(&max)?.exp()?;
        let sink = self.attn_sinks.unsqueeze(0)?.unsqueeze(2)?; // [1, h, 1]
        let den = e
            .sum_keepdim(D::Minus1)?
            .add(&sink.broadcast_sub(&max)?.exp()?)?; // [seq, h, 1]

        // Batched over `seq`; a broadcast_matmul here would materialize the
        // whole key block once per head.
        let num = e.contiguous()?.matmul(&k_all)?; // [seq, h, n_kv] @ [seq, n_kv, d] → [seq, h, d]
        let o = num.broadcast_div(&den)?; // [seq, h, d]

        // Derope the output's trailing slice.
        let o4 = o.unsqueeze(0)?.transpose(1, 2)?; // [1, h, seq, d]
        let o4 = rope_apply(
            &self.rotary,
            &o4,
            self.nope_head_dim,
            self.rope_head_dim,
            offset,
            true,
        )?;
        let o = o4.transpose(1, 2)?.squeeze(0)?; // [seq, h, d]

        // Grouped output projection.  `wo_a` is [o_group_dim, o_groups*o_lora_rank]:
        // per group g the block of columns [g*r, (g+1)*r) applies to the g-th
        // group of heads.  A single matmul produces all groups; the diagonal
        // gather extracts y[s, g, r] = oa[s, g, g*o_lora_rank + r].
        let o_group_dim = (self.n_head / self.o_groups) * self.head_dim;
        let og = o.reshape((seq, self.o_groups, o_group_dim))?;
        let oa = self
            .wo_a
            .forward(&og.reshape((seq * self.o_groups, o_group_dim))?)?; // [seq*g, g*r]
        let oa = oa.reshape((seq, self.o_groups, self.o_groups * self.o_lora_rank))?;
        let mut idxv = Vec::with_capacity(seq * self.o_groups * self.o_lora_rank);
        for _ in 0..seq {
            for g in 0..self.o_groups {
                for r in 0..self.o_lora_rank {
                    idxv.push((g * self.o_lora_rank + r) as u32);
                }
            }
        }
        let idx = Tensor::from_vec(idxv, (seq, self.o_groups, self.o_lora_rank), dev)?;
        let y = oa
            .gather(&idx, D::Minus1)?
            .reshape((seq, self.o_groups * self.o_lora_rank))?;
        self.wo_b.forward(&y)?.reshape((1, seq, ()))
    }
}

// KV caches (owned by the model, reset via clear_kv_cache).
struct KvState {
    /// Sliding-window cache: [window_size, head_dim].
    swa: Option<Tensor>,
    /// Compressed KV for CSA/HCA attention: [max_blocks, head_dim].
    comp: Option<Tensor>,
    /// Indexer compressed KV: [max_blocks, index_head_dim].
    lid: Option<Tensor>,
    /// Main compressor streaming state.
    comp_state: Option<CompressorState>,
    /// Indexer compressor streaming state.
    lid_state: Option<CompressorState>,
}

impl KvState {
    fn new(cfg: &Config, layer: usize, dev: &Device, max_seq: usize) -> Result<Self> {
        let hd = cfg.head_dim;
        let ihd = cfg.index_head_dim;
        let win = cfg.window_size;
        let ratio = *cfg.compress_ratios.get(layer).unwrap_or(&0);
        // The indexer always compresses at the CSA ratio; the main compressor
        // uses this layer's own ratio, so an HCA layer needs 32x fewer rows.
        let max_blocks = max_seq / CSA_RATIO + 1;
        let swa = Tensor::zeros((win, hd), DType::F32, dev)?;
        let (comp, comp_state) = if ratio != 0 {
            let coff = if ratio == CSA_RATIO { 2 } else { 1 };
            (
                Some(Tensor::zeros((max_seq / ratio + 1, hd), DType::F32, dev)?),
                Some(CompressorState::new(ratio, coff, hd, dev)?),
            )
        } else {
            (None, None)
        };
        let (lid, lid_state) = if ratio == CSA_RATIO {
            (
                Some(Tensor::zeros((max_blocks, ihd), DType::F32, dev)?),
                Some(CompressorState::new(CSA_RATIO, 2, ihd, dev)?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            swa: Some(swa),
            comp,
            lid,
            comp_state,
            lid_state,
        })
    }
}

// ─── MoE ────────────────────────────────────────────────────────────────────

#[cfg(feature = "distributed")]
struct DistributedExpert {
    gate: crate::distributed::shard::ShardedTensor,
    up: crate::distributed::shard::ShardedTensor,
    down: crate::distributed::shard::ShardedTensor,
}

#[cfg(feature = "distributed")]
struct DistributedMoe {
    experts: Vec<DistributedExpert>,
    cluster: Arc<crate::distributed::session::ClusterSession>,
    clamp: f64,
    local_bytes: usize,
    full_bytes: usize,
}

#[cfg(feature = "distributed")]
impl DistributedMoe {
    fn dispatch(
        &self,
        xs: &Tensor,
        ids: &[u32],
        weights: &[f32],
        k: usize,
    ) -> Result<(Tensor, Vec<u32>)> {
        let (n, h) = xs.dims2()?;
        let input = xs.flatten_all()?.to_vec1::<f32>()?;
        let mut output = vec![0f32; input.len()];
        let mut buckets = vec![Vec::new(); self.experts.len()];
        for (i, (&id, &weight)) in ids.iter().zip(weights).enumerate() {
            let bucket = buckets.get_mut(id as usize).ok_or_else(|| {
                candle_core::Error::Msg("deepseek4: routed expert id out of range".into())
            })?;
            bucket.push((i / k, weight));
        }
        for (expert, bucket) in self.experts.iter().zip(&buckets) {
            if bucket.is_empty() {
                continue;
            }
            // These handles contain only local rows/columns, not full experts.
            for tensor in [&expert.gate, &expert.up, &expert.down] {
                let _ = tensor.prefetch();
            }
            for &(token, weight) in bucket {
                let x = &input[token * h..(token + 1) * h];
                let gate = expert.gate.forward(x).map_err(candle_core::Error::wrap)?;
                let up = expert.up.forward(x).map_err(candle_core::Error::wrap)?;
                let width = gate.len();
                let gate = clamp_gate(Tensor::from_vec(gate, width, &Device::Cpu)?, self.clamp)?;
                let up = clamp_up(Tensor::from_vec(up, width, &Device::Cpu)?, self.clamp)?;
                let hidden = (silu(&gate)? * up)?.to_vec1::<f32>()?;
                let local = expert.down.forward_local(&hidden).map_err(candle_core::Error::wrap)?;
                for (out, value) in output[token * h..(token + 1) * h].iter_mut().zip(local) {
                    *out += weight * value;
                }
            }
        }
        // Exactly one logical reduction per MoE, before the replicated shared MLP.
        self.cluster.all_reduce_sum(&mut output).map_err(candle_core::Error::wrap)?;
        let mut last = ids[n.saturating_sub(1) * k..].to_vec();
        last.sort_unstable();
        last.dedup();
        Ok((Tensor::from_vec(output, (n, h), &Device::Cpu)?, last))
    }
}

struct Moe {
    /// Router weight, transposed to `[n_embd, n_expert]` and made contiguous
    /// once at load (it was a transpose + copy per forward call).
    gate_t: Tensor,
    gate_bias: Option<Tensor>, // [n_expert]
    tid2eid: Option<Tensor>,   // [n_vocab, n_expert_used] (candle shape after gguf dim reversal)
    experts: Vec<Mlp>,
    #[cfg(feature = "distributed")]
    distributed: Option<DistributedMoe>,
    shared: Option<Mlp>,
    n_expert_used: usize,
    weights_scale: f64,
    hash: bool,
    /// Device the routed-expert weights live on.  This is the model device
    /// for a CPU load, but on an accelerator it is the CPU: the gate/up
    /// experts are IQ2_XXS (no CUDA/Metal kernel) and the expert pool is far
    /// larger than the VRAM of the cards that can run the dense set.
    /// [`Moe::dispatch`] runs the whole block there and moves the result back.
    expert_device: Device,
    /// This block's layer index: the `(layer, expert)` key into the pool.
    layer: u32,
    /// The device expert pool, when a VRAM cache budget was given: routed
    /// experts resident there run on the device, the rest on the host.
    device_pool: Option<Arc<Ds4DevicePool>>,
}

impl Moe {
    fn forward(
        &self,
        xs: &Tensor,
        input_ids: &Tensor,
        ctx: DispatchCtx,
    ) -> Result<(Tensor, Vec<u32>)> {
        let (b, seq_len, h) = xs.dims3()?;
        let n_tokens = b * seq_len;
        let x2 = xs.reshape((n_tokens, h))?;

        let logits = x2.matmul(&self.gate_t)?; // [n_tokens, n_expert]
                                               // sqrt(softplus(x)) scoring; bias shifts selection only.
        let probs = softplus(&logits)?.sqrt()?;
        let (weights, indices) = if self.hash {
            // Hash layers: expert *selection* comes from tid2eid[token_id], but the
            // routing *weights* still come from the gate network's softplus scores
            // (official `Gate.forward` gathers `original_scores` at the hash ids).
            let ids = input_ids.flatten_all()?.to_vec1::<u32>()?;
            let tid = Tensor::from_vec(ids, n_tokens, xs.device())?;
            // tid2eid is [n_vocab, n_expert_used]: select the token-id rows.
            let idx = self
                .tid2eid
                .as_ref()
                .unwrap()
                .index_select(&tid, 0)?
                .to_dtype(DType::U32)?; // [n_tokens, k]
            let mut weights = probs.gather(&idx, D::Minus1)?;
            weights = weights.broadcast_div(
                &weights
                    .sum_keepdim(D::Minus1)?
                    .clamp(6.103_515_6e-5, f32::INFINITY)?,
            )?;
            if self.weights_scale != 0.0 && self.weights_scale != 1.0 {
                weights = (weights * self.weights_scale)?;
            }
            (weights, idx)
        } else {
            let selection = match &self.gate_bias {
                Some(bias) => probs.broadcast_add(&bias.reshape((1, ()))?)?,
                None => probs.clone(),
            };
            let topk_idx = topk_indices(&selection, self.n_expert_used)?;
            let mut weights = probs.gather(&topk_idx, D::Minus1)?;
            weights = weights.broadcast_div(
                &weights
                    .sum_keepdim(D::Minus1)?
                    .clamp(6.103_515_6e-5, f32::INFINITY)?,
            )?;
            if self.weights_scale != 0.0 && self.weights_scale != 1.0 {
                weights = (weights * self.weights_scale)?;
            }
            (weights, topk_idx)
        };

        // A decode step (one token per sequence, batched or not) warms the
        // pool from its routing; a prefill only reads the pool.
        let decode = ctx.decode;
        let (routed, routed_ids) = self.dispatch(&x2, &indices, &weights, n_tokens, decode, ctx)?;
        let mut out = routed;
        if let Some(shared) = &self.shared {
            out = (out + shared.forward(&x2)?)?;
        }
        Ok((out.reshape((b, seq_len, h))?, routed_ids))
    }

    fn dispatch(
        &self,
        x2: &Tensor,
        indices: &Tensor,
        weights: &Tensor,
        n_tokens: usize,
        decode: bool,
        ctx: DispatchCtx,
    ) -> Result<(Tensor, Vec<u32>)> {
        let k = self.n_expert_used;
        let h = x2.dim(1)?;
        let ids: Vec<u32> = indices.flatten_all()?.to_vec1()?;
        let wts: Vec<f32> = weights.flatten_all()?.to_vec1()?;
        crate::route_trace::record(self.layer, &ids, k, ctx.trace_row_base, ctx.trace_chunk);
        #[cfg(feature = "distributed")]
        if let Some(distributed) = &self.distributed {
            return distributed.dispatch(x2, &ids, &wts, k);
        }
        let out_device = x2.device().clone();
        if let Some(pool) = &self.device_pool {
            // Every launch of the previous MoE layer has completed: with the
            // dense set on the device the routing read-back above waited for
            // the in-order queue, and with it on the CPU that layer's pooled
            // dispatch ended with a blocking read of its device sum (or
            // launched nothing).  Buffers evicted since can go.
            pool.pool.reclaim();
        }

        let mut per_expert: Vec<Vec<(u32, f32)>> = vec![Vec::new(); self.experts.len()];
        for t in 0..n_tokens {
            for s in 0..k {
                let e = ids[t * k + s] as usize;
                if e < self.experts.len() {
                    per_expert[e].push((t as u32, wts[t * k + s]));
                }
            }
        }

        // Routing record for the speculative next-step prefetch: during
        // decode, every id this step routed to (routing consistency makes
        // the next step likely to repeat them); during prefill only the
        // final row's ids — the routing the immediately following decode
        // step is most likely to continue from.
        let mut routed_ids: Vec<u32> = ids[n_tokens.saturating_sub(1) * k..].to_vec();
        routed_ids.sort_unstable();
        routed_ids.dedup();

        let y = match &self.device_pool {
            Some(pool) => self.dispatch_pooled(
                pool,
                x2,
                &per_expert,
                &routed_ids,
                n_tokens,
                decode,
                &ctx,
                h,
            )?,
            None => {
                // The routed experts may live on a different device from the
                // activations: on an accelerator the dense set is on the
                // device while the expert pool stays in host memory (see
                // `Moe::expert_device`).  Move the whole MoE block's input
                // across once and its result back once, so the device
                // boundary costs two transfers per layer instead of one per
                // expert matmul.
                let x2 = crate::moe::on_device(x2, &self.expert_device)?;
                let active: Vec<usize> = (0..self.experts.len())
                    .filter(|e| !per_expert[*e].is_empty())
                    .collect();
                self.host_experts(&x2, &per_expert, &active, n_tokens, h)?
            }
        };
        // …and hand the block's output back to the model's device.
        let y = crate::moe::on_device(&y, &out_device)?.into_owned();
        Ok((y, routed_ids))
    }

    /// Run the routed experts `active` over their token buckets on the host
    /// form (`x2` on [`Moe::expert_device`]) and return the weighted sum
    /// `[n_tokens, h]` on that device.
    fn host_experts(
        &self,
        x2: &Tensor,
        per_expert: &[Vec<(u32, f32)>],
        active: &[usize],
        n_tokens: usize,
        h: usize,
    ) -> Result<Tensor> {
        // The gate has just told us which experts this token needs.  Each
        // expert's weights are a contiguous run inside the model mapping, so
        // `MADV_WILLNEED` turns the scattered page faults that would otherwise
        // stall the expert matmuls into sequential background reads — the
        // kernel streams the selected experts while the first matmul runs.
        // Idempotent and best-effort (a no-op for non-mmap loads).
        for &e in active {
            self.experts[e].prefetch();
        }

        let dev = x2.device();
        let mut y = Tensor::zeros((n_tokens, h), DType::F32, dev)?;
        // Select each expert's input rows once; all three phases reuse them.
        let mut sel: Vec<(usize, Vec<u32>, Tensor)> = Vec::with_capacity(active.len());
        for &e in active {
            let token_idx: Vec<u32> = per_expert[e].iter().map(|(t, _)| *t).collect();
            let count = token_idx.len();
            if count == 0 {
                continue;
            }
            let idx = Tensor::from_vec(token_idx.clone(), count, dev)?;
            sel.push((e, token_idx, x2.index_select(&idx, 0)?));
        }

        // Tensor-major MoE: run every expert's gate, then every expert's up,
        // then every expert's down.  Reads of a weight tensor are one
        // sequential pass over the selected experts instead of jumping
        // between gate/up/down regions on each expert — the kernel's
        // readahead streams each tensor at full device bandwidth, which is
        // what the disk-bound prefill path needs.  Gate/up outputs for the
        // whole batch are a few hundred KiB.
        let mut gates: Vec<Tensor> = Vec::with_capacity(sel.len());
        for (e, _, x_sel) in &sel {
            gates.push(self.experts[*e].gate_forward(x_sel)?);
        }
        let mut ups: Vec<Tensor> = Vec::with_capacity(sel.len());
        for (e, _, x_sel) in &sel {
            ups.push(self.experts[*e].up_forward(x_sel)?);
        }
        for (((e, token_idx, _), gate), up) in sel.iter().zip(gates).zip(ups) {
            let out = self.experts[*e].combine_and_down(gate, up)?;
            let idx = Tensor::from_vec(token_idx.clone(), token_idx.len(), dev)?;
            let w: Vec<f32> = per_expert[*e].iter().map(|(_, w)| *w).collect();
            let w = Tensor::from_vec(w, (token_idx.len(), 1), dev)?;
            y = y.index_add(&idx, &out.broadcast_mul(&w)?, 0)?;
        }
        Ok(y)
    }

    /// The MoE block with a device expert pool: routed experts resident in
    /// the pool run on the device, the rest on the host, and the two partial
    /// sums are added on the block's output device.
    ///
    /// * Decode (one token per sequence): the resident experts are enqueued
    ///   first (asynchronous launches), then the misses run on the host while
    ///   the device works; every miss is queued for a background upload so
    ///   the next step that routes here finds it resident
    ///   (`JOSHUA_EXPERT_MISS=upload` uploads synchronously instead).
    /// * Prefill: resident experts only — a prompt routes through nearly
    ///   every expert of every layer, and uploading them all would stream the
    ///   whole pool over the bus and evict the decode working set.  The last
    ///   prompt row's routing is queued, so the decode that follows starts
    ///   with its likely experts resident.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_pooled(
        &self,
        pool: &Ds4DevicePool,
        x2: &Tensor,
        per_expert: &[Vec<(u32, f32)>],
        last_row_ids: &[u32],
        n_tokens: usize,
        decode: bool,
        ctx: &DispatchCtx,
        h: usize,
    ) -> Result<Tensor> {
        let dev = &pool.device;
        let mut hits: Vec<(usize, Arc<Ds4DeviceExpert>)> = Vec::new();
        let mut misses: Vec<usize> = Vec::new();
        for (e, bucket) in per_expert.iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            if decode && pool.miss == MissPolicy::Upload {
                let before = pool.pool.generation(self.layer, e as u32);
                pool.pool.acquire(self.layer, e as u32);
                // Uploaded here, not by the uploader: still release the host
                // pages in due course — for a new upload only, not a hit.
                if pool.pool.generation(self.layer, e as u32) != before {
                    pool.uploader.note_resident(self.layer, e as u32);
                }
            }
            match pool.pool.lookup(self.layer, e as u32) {
                Some(slot) => hits.push((e, slot)),
                None => {
                    misses.push(e);
                    if decode {
                        pool.uploader.request(self.layer, e as u32);
                    }
                }
            }
        }
        if !decode && ctx.seed_prefill {
            for &e in last_row_ids {
                pool.uploader.request(self.layer, e);
            }
        }

        let timing: &PhaseTiming = if decode {
            &pool.decode
        } else {
            &ctx.prefill_timing
        };
        PhaseTiming::add(&timing.miss_experts, misses.len() as u64);
        if pool.probe_pages {
            // Was what the host is about to run in RAM, or is it about to
            // come from disk?  Probed before the prefetch below so the
            // answer is the state the step found, not the one it made.
            for &e in &misses {
                let handle = self.experts[e].prefetch.as_ref();
                if std::env::var("JOSHUA_DS4_PROBE_DEBUG").as_deref() == Ok("1") {
                    let rp = handle.and_then(|h| h.resident_pages());
                    eprintln!(
                        "DBG probe layer={} e={e} handle={} resident_pages={:?}",
                        self.layer,
                        handle.is_some(),
                        rp
                    );
                }
                if let Some((res, total)) = handle.and_then(|h| h.resident_pages()) {
                    PhaseTiming::add(&timing.miss_pages, total as u64);
                    PhaseTiming::add(&timing.miss_pages_resident, res as u64);
                    if res == total {
                        PhaseTiming::add(&timing.miss_experts_resident, 1);
                    }
                }
            }
        }

        // Device part first: the launches are asynchronous, so the host
        // misses below overlap with them.
        let t0 = std::time::Instant::now();
        let y_dev = if hits.is_empty() {
            None
        } else {
            let x_dev = crate::moe::on_device(x2, dev)?;
            Some(if n_tokens == 1 {
                Self::pooled_decode(&hits, per_expert, &x_dev)?
            } else {
                Self::pooled_prefill(&hits, per_expert, &x_dev, n_tokens, h)?
            })
        };
        let t1 = std::time::Instant::now();
        let y_host = if misses.is_empty() {
            None
        } else {
            // The release hook must not drop these experts' pages while the
            // host kernels read them.
            let keys: Vec<(u32, u32)> = misses.iter().map(|&e| (self.layer, e as u32)).collect();
            let _busy = BusyGuard::new(&pool.busy, keys);
            let x_host = crate::moe::on_device(x2, &self.expert_device)?;
            Some(self.host_experts(&x_host, per_expert, &misses, n_tokens, h)?)
        };
        let t2 = std::time::Instant::now();
        // The handles this layer launched on go back to the pool's retire
        // list: freed at the next `reclaim`, after a read-back proved the
        // launches complete, even if the uploader evicts them meanwhile.
        for (_, slot) in hits {
            pool.pool.retire(slot);
        }
        let out_device = x2.device();
        let y = match (y_dev, y_host) {
            (Some(d), Some(hst)) => {
                let d = crate::moe::on_device(&d, out_device)?;
                let hst = crate::moe::on_device(&hst, out_device)?;
                (d.as_ref() + hst.as_ref())?
            }
            (Some(y), None) | (None, Some(y)) => {
                crate::moe::on_device(&y, out_device)?.into_owned()
            }
            (None, None) => Tensor::zeros((n_tokens, h), DType::F32, out_device)?,
        };
        let t3 = std::time::Instant::now();
        PhaseTiming::add(&timing.device_launch_ns, (t1 - t0).as_nanos() as u64);
        PhaseTiming::add(&timing.host_ns, (t2 - t1).as_nanos() as u64);
        PhaseTiming::add(&timing.device_wait_ns, (t3 - t2).as_nanos() as u64);
        Ok(y)
    }

    /// Decode over the resident experts: `x_dev` is the one row `[1, h]`;
    /// each expert's output is scaled by its routing weight (a host scalar,
    /// no upload) and the outputs are summed.
    fn pooled_decode(
        hits: &[(usize, Arc<Ds4DeviceExpert>)],
        per_expert: &[Vec<(u32, f32)>],
        x_dev: &Tensor,
    ) -> Result<Tensor> {
        let mut acc: Option<Tensor> = None;
        for (e, slot) in hits {
            let w = per_expert[*e].first().map_or(0.0, |(_, w)| *w) as f64;
            let out = slot.forward(x_dev)?.affine(w, 0.0)?;
            acc = Some(match acc {
                None => out,
                Some(a) => (a + out)?,
            });
        }
        acc.ok_or_else(|| candle_core::Error::Msg("deepseek4: no resident expert to run".into()))
    }

    /// Prefill over the resident experts: one packed token-index buffer and
    /// one packed routing-weight buffer per layer (a single upload each);
    /// each expert gathers its rows through a narrowed view, and the
    /// weighted outputs are scattered back with one `index_add`.
    fn pooled_prefill(
        hits: &[(usize, Arc<Ds4DeviceExpert>)],
        per_expert: &[Vec<(u32, f32)>],
        x_dev: &Tensor,
        n_tokens: usize,
        h: usize,
    ) -> Result<Tensor> {
        let dev = x_dev.device();
        let mut idx_all: Vec<u32> = Vec::new();
        let mut w_all: Vec<f32> = Vec::new();
        let mut spans: Vec<(usize, usize)> = Vec::with_capacity(hits.len());
        for (e, _) in hits {
            let off = idx_all.len();
            for (t, w) in &per_expert[*e] {
                idx_all.push(*t);
                w_all.push(*w);
            }
            spans.push((off, idx_all.len() - off));
        }
        let total = idx_all.len();
        let idx_t = Tensor::from_vec(idx_all, total, dev)?;
        let w_t = Tensor::from_vec(w_all, (total, 1), dev)?;
        let mut outs: Vec<Tensor> = Vec::with_capacity(hits.len());
        for ((_, slot), (off, count)) in hits.iter().zip(spans) {
            let idx = idx_t.narrow(0, off, count)?;
            let x_sel = x_dev.index_select(&idx, 0)?;
            outs.push(
                slot.forward(&x_sel)?
                    .broadcast_mul(&w_t.narrow(0, off, count)?)?,
            );
        }
        let all = Tensor::cat(&outs, 0)?;
        Tensor::zeros((n_tokens, h), DType::F32, dev)?.index_add(&idx_t, &all, 0)
    }
}

/// Indices of the top-`k` values along the last dim (descending), as u32.
fn topk_indices(t: &Tensor, k: usize) -> Result<Tensor> {
    t.arg_sort_last_dim(false)?
        .narrow(D::Minus1, 0, k)?
        .contiguous()
}

// ─── Layer + model ───────────────────────────────────────────────────────────

struct Layer {
    attn_norm: RmsNorm,
    attn: Attention,
    ffn_norm: RmsNorm,
    ffn: FeedForward,
    hc_attn_fn: QMatMul,
    hc_attn_base: Tensor,
    hc_attn_scale: Tensor,
    hc_ffn_fn: QMatMul,
    hc_ffn_base: Tensor,
    hc_ffn_scale: Tensor,
}

enum FeedForward {
    Moe(Moe),
}

impl FeedForward {
    fn forward(
        &self,
        xs: &Tensor,
        input_ids: &Tensor,
        ctx: DispatchCtx,
    ) -> Result<(Tensor, Vec<u32>)> {
        match self {
            Self::Moe(m) => m.forward(xs, input_ids, ctx),
        }
    }
}

/// A quantized DeepSeek-V4 model loaded from GGUF.
/// The immutable, weight-bearing half of a DeepSeek-V4 model.
///
/// Holds every weight tensor, the rotary/compressor/reference tables, the
/// mmap backing, and the residency backend — nothing here is mutated after
/// load (the qwen3moe/deepseek2 pattern from #60).  Wrapped in an `Arc` and
/// shared by every [`ModelWeights`] session derived via
/// [`ModelWeights::new_session`]: a second concurrent request on an
/// accelerator reuses the already-uploaded dense set instead of reloading it.
struct Shared {
    tok_embeddings: crate::token_embedding::TokenEmbedding,
    layers: Vec<Layer>,

    cfg: Config,
    norm: RmsNorm,
    output: QMatMul,
    hc_head_fn: QMatMul,
    hc_head_base: Tensor,
    hc_head_scale: Tensor,
    hc_mult: usize,
    hc_eps: f64,
    max_seq: usize,
    device: Device,
    /// The model mapping, retained so prefill can prefetch whole layers.
    mmap: Option<std::sync::Arc<memmap2::Mmap>>,
    /// The model file, retained so prefill can run a layer-ahead pread
    /// prefetch thread (see [`crate::mmap_tensor::LayerPrefetcher`]).
    file: Option<std::sync::Arc<std::fs::File>>,
    /// Per-layer expert byte ranges in the mapping (see
    /// [`crate::gguf_ext::GgufHeader::layer_expert_ranges`]).
    layer_expert_ranges: Vec<Option<(usize, usize)>>,
    /// Executes residency for the hot set: CPU madvise, plus the device
    /// pool's uploader when one exists.  Built once at load from the
    /// per-expert handles.
    residency: std::sync::Arc<dyn crate::residency::ExpertResidency>,
    /// The device expert pool (see [`ModelWeights::from_gguf_mmap_placed`]),
    /// kept for diagnostics; the MoE blocks hold their own handles.
    device_pool: Option<Arc<Ds4DevicePool>>,
}

/// A quantized DeepSeek-V4 model: one shared set of weights plus per-session
/// mutable state (KV caches, batched-KV, the speculative-routing predictor and
/// the hot-expert cache).  Sessions are derived from one template via
/// [`ModelWeights::new_session`] (one `Arc` clone of the weights, no re-read,
/// no re-upload).
pub struct ModelWeights {
    shared: std::sync::Arc<Shared>,
    /// Per-layer KV cache — the primary per-session tensor state.
    kv: Vec<KvState>,
    /// Per-sequence KV state for batched `forward_sequences` decode:
    /// `[layer][seq]`.  Persistent across steps (unlike a fresh-per-call
    /// cache) so multi-token batched generation keeps each sequence's
    /// attention history.  Reset via [`ModelWeights::reset_batch_kv`].
    kv_seq: Vec<Vec<KvState>>,
    /// Routed-expert ids of the most recent forward pass, per layer index
    /// (see [`ModelWeights::last_routed_experts`]).  The prediction source
    /// for the speculative decode prefetch.
    last_routed: Vec<Vec<u32>>,
    /// Routing-frequency LRU hot-expert cache (per-session bookkeeping; budget
    /// set after load via [`ModelWeights::set_pin_hot_experts`], CLI flag
    /// `--pin-hot-experts`).  Records routing, re-selects the hot set every
    /// [`crate::hot_experts::REFRESH_STEPS`] decode steps, and reports newly
    /// hot experts for the residency backend.
    hot_experts: crate::hot_experts::HotExpertCache,
    /// Dispatch settings for the streamed prefill in progress (set per
    /// chunk by the streaming runner; per session, never shared).
    stream_ctx: DispatchCtx,
    /// This session's prefill time split (taken and logged at the end of
    /// each prefill).  Per session: concurrent prefills must not take each
    /// other's counters.
    prefill_timing: Arc<PhaseTiming>,
}

// Routed experts (see `crate::moe::is_routed_expert`) are the only weights
// that stay CPU-resident on an accelerator; everything else is dense and can
// be dequantized onto the device (M4).
use crate::moe::is_routed_expert;

/// Small GGUF reader over the memory-mapped file.
struct Reader<R: Read + Seek> {
    ct: gguf_file::Content,
    /// Raw header with every tensor's dtype as its GGUF id, including types
    /// candle cannot represent (IQ2_XXS experts, I32 id tables).
    raw: Option<GgufHeader>,
    reader: R,
    device: Device,
    /// Device the routed-expert tensors are built on: the CPU for every
    /// mapped model (a mapped expert is borrowed as CPU storage; a device
    /// expert cache uploads from it), `device` for a streamed load.  See
    /// [`Moe::expert_device`].
    expert_device: Device,
    mmap: Option<std::sync::Arc<memmap2::Mmap>>,
    /// The model file, for the layer-ahead pread prefetch thread.
    file: Option<std::sync::Arc<std::fs::File>>,
    #[cfg(feature = "distributed")]
    cluster: Option<Arc<crate::distributed::session::ClusterSession>>,
}

impl<R: Read + Seek> Reader<R> {
    fn qtensor(&mut self, name: &str) -> Result<QTensor> {
        // Tensors candle cannot represent never make it into `ct`; borrow them
        // from the mapping (or read + decode) using the raw header instead.
        let raw_info = self.raw.as_ref().and_then(|r| r.tensors.get(name)).cloned();
        if let Some(info) = raw_info {
            if !crate::gguf_ext::is_candle_supported(info.dtype) {
                let tensor_data_offset = self
                    .raw
                    .as_ref()
                    .map(|r| r.tensor_data_offset)
                    .unwrap_or(self.ct.tensor_data_offset);
                // Borrowing yields CPU-backed `QStorage`, so it is only sound
                // when the model itself lives on the CPU.
                if let Some(mmap) = self.mmap.as_ref().filter(|_| self.device.is_cpu()) {
                    if let Some(qt) = crate::mmap_tensor::borrowed_qtensor_raw(
                        mmap,
                        info.dtype,
                        info.offset,
                        tensor_data_offset,
                        info.dims.clone().into(),
                    )? {
                        return Ok(qt);
                    }
                }
                // Accelerator devices cannot borrow (the blocks are CPU
                // storage), and decoding a routed-expert IQ2_XXS tensor to f32
                // is an order-of-magnitude memory blow-up (the experts alone
                // are ~40 GB of 2-bit data, ~640 GB as f32).  Routed experts
                // therefore stay on the CPU (the design always keeps the
                // expert pool in host RAM on an accelerator via `expert_device`).
                // Dense (non-expert) raw-dtype tensors, in contrast, are small
                // enough (~GiB) to dequantize onto the accelerator, so we let
                // those fall through to the decode-to-f32-on-device path below
                // (M4: dense set on OpenCl).
                if !self.device.is_cpu() && is_routed_expert(name) {
                    candle_core::bail!(
                        "deepseek4: tensor `{name}` has GGUF dtype {} which is only supported on the CPU device (routed expert)",
                        info.dtype
                    );
                }
                // No mapping (CPU): decode the weights to f32 and re-quantize
                // as F32 so the QMatMul machinery downstream keeps working
                // unchanged.  This is also what makes the streamed path agree
                // with the mmap path: both end up with f32-activation matmul
                // semantics (candle's own `k_quants::matmul` quantizes
                // activations to Q8_0, which would diverge by ~1% per layer).
                // Note the memory cost: a real IQ2_XXS expert tensor (~40 GB
                // across the model) becomes ~640 GB as f32, so the streamed
                // path is only practical for small models — the mmap path is
                // the production one.
                let bytes = self.raw_bytes_from(name)?;
                if let Some(f32s) =
                    crate::quant_matmul::decode_raw_to_f32(info.dtype, &bytes, info.elem_count())?
                {
                    let t = Tensor::from_vec(f32s, info.dims.clone(), &self.device)?;
                    return QTensor::quantize(&t, GgmlDType::F32);
                }
                candle_core::bail!(
                    "deepseek4: tensor `{name}` has GGUF dtype {} which has no decoder here",
                    info.dtype
                );
            }
        }
        if let Some(mmap) = &self.mmap {
            return crate::mmap_tensor::qtensor_from_mmap(
                &self.ct,
                mmap,
                &mut self.reader,
                name,
                &self.device,
            );
        }
        // No mapping (CPU): quantized/f16 tensors are decoded to f32 so the
        // streamed path shares the mmap path's f32-activation matmul
        // semantics (same rationale as above).  F32 tensors keep candle's
        // reader, which is exact for them.  This requires the raw header to
        // locate + size the tensor's bytes (candle's `TensorInfo` has no
        // on-disk offset); `ModelWeights::from_gguf` (raw = None) falls
        // through to candle's own reader instead.
        if self.raw.is_some() {
            let decode = self.ct.tensor_infos.get(name).map(|info| {
                (
                    crate::gguf_ext::ggml_id_from_dtype(info.ggml_dtype),
                    info.shape.clone(),
                )
            });
            if let Some((dtype, shape)) = decode {
                if let Some(f32s) = crate::quant_matmul::decode_raw_to_f32(
                    dtype,
                    &self.raw_bytes_from(name)?,
                    shape.elem_count(),
                )? {
                    let t = Tensor::from_vec(f32s, shape, &self.device)?;
                    return QTensor::quantize(&t, GgmlDType::F32);
                }
            }
        }
        self.ct.tensor(&mut self.reader, name, &self.device)
    }

    /// Read a tensor's raw bytes from the underlying reader using the raw
    /// header (works for dtypes candle cannot describe).
    fn raw_bytes_from(&mut self, name: &str) -> Result<Vec<u8>> {
        let (offset, tensor_data_offset, size) =
            {
                let raw = self.raw.as_ref().ok_or_else(|| {
                    candle_core::Error::Msg(format!("no raw header for `{name}`"))
                })?;
                let info = raw.tensors.get(name).ok_or_else(|| {
                    candle_core::Error::Msg(format!("deepseek4: tensor `{name}` not in raw header"))
                })?;
                let size = crate::gguf_ext::type_size_bytes(info.dtype, info.elem_count())
                    .ok_or_else(|| {
                        candle_core::Error::Msg(format!(
                            "deepseek4: no size known for GGUF dtype {}",
                            info.dtype
                        ))
                    })?;
                (info.offset, raw.tensor_data_offset, size)
            };
        self.reader
            .seek(SeekFrom::Start(tensor_data_offset + offset))?;
        let mut buf = vec![0u8; size];
        self.reader.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Load an I32 tensor (GGUF dtype 26) as f32.  Used for the routed-expert
    /// id tables in hash layers, which candle's dtype table cannot name.
    fn i32_to_f32_tensor(&mut self, name: &str) -> Result<Tensor> {
        let is_i32 = self
            .raw
            .as_ref()
            .and_then(|r| r.tensors.get(name))
            .map(|info| info.dtype == 26)
            .unwrap_or(false);
        if !is_i32 {
            return self.f32_tensor(name);
        }
        let dims = self
            .raw
            .as_ref()
            .and_then(|r| r.tensors.get(name))
            .map(|info| info.dims.clone())
            .unwrap_or_default();
        let bytes = self.raw_bytes_from(name)?;
        let mut vals = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            vals.push(i32::from_le_bytes(chunk.try_into().unwrap()) as f32);
        }
        Tensor::from_vec(vals, dims, &self.device)
    }
    fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        QMatMul::from_qtensor(self.qtensor(name)?)
    }
    fn qmatmul_opt(&mut self, name: &str) -> Result<Option<QMatMul>> {
        // Distinguish "absent" from "present but failed to load": a tensor
        // that exists but cannot be decoded must error, not silently fall
        // back to a different weight (e.g. tied embeddings for the head).
        if self.has(name) {
            Ok(Some(self.qmatmul(name)?))
        } else {
            Ok(None)
        }
    }
    fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        RmsNorm::from_qtensor(self.qtensor(name)?, eps)
    }
    fn f32_tensor(&mut self, name: &str) -> Result<Tensor> {
        self.qtensor(name)?
            .dequantize(&self.device)?
            .to_dtype(DType::F32)
    }
    fn has(&self, name: &str) -> bool {
        // `ct` only carries tensors whose dtype candle can name; the raw
        // header is the source of truth for the rest (IQ2_XXS, I32, ...),
        // which `to_candle_content` drops.  Without this a real tensor in
        // such a format would be reported absent and silently swapped for
        // a different weight.
        self.ct.tensor_infos.contains_key(name)
            || self
                .raw
                .as_ref()
                .is_some_and(|r| r.tensors.contains_key(name))
    }
}

impl ModelWeights {
    /// Load a `deepseek4` GGUF.
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        Self::from_gguf_mmap(ct, None, reader, device, None, None, 0)
    }

    /// Load with weights borrowed in place from `mmap` where possible.
    ///
    /// `raw` is the GGUF header with raw dtype ids (see [`crate::gguf_ext`]);
    /// it carries the IQ2_XXS and I32 tensors that candle's [`GgmlDType`]
    /// cannot represent and that are therefore absent from `ct`.
    pub fn from_gguf_mmap<R: Read + Seek>(
        ct: gguf_file::Content,
        raw: Option<&GgufHeader>,
        reader: &mut R,
        device: &Device,
        mmap: Option<std::sync::Arc<memmap2::Mmap>>,
        file: Option<std::sync::Arc<std::fs::File>>,
        n_ctx: usize,
    ) -> Result<Self> {
        Self::from_gguf_mmap_placed(ct, raw, reader, device, device, mmap, file, n_ctx, None)
    }

    /// [`ModelWeights::from_gguf_mmap`] with an explicit expert placement and
    /// a VRAM expert-cache budget.
    ///
    /// A mapped model's routed experts are *always* borrowed from the mapping
    /// on the host (IQ2_XXS gate/up + Q2_K down; the ~72 GiB pool of
    /// V4-Flash never fits a card).  `device_expert_cache_bytes` (> 0) adds a
    /// bounded device pool over them: a byte-budgeted LRU of experts
    /// uploaded to `device`, filled by a background uploader from the
    /// routing (decode misses, the routing-frequency hot set, the last
    /// prompt row), consulted by every MoE layer — resident experts run on
    /// the device, the rest on the host, in the same forward pass.  The
    /// pool needs an IQ2_XXS kernel, which only the OpenCL and SYCL backends (and, for
    /// tests, the CPU, where a slot shares the host tensors) has; on other
    /// accelerators the budget is ignored with a log line.
    ///
    /// `expert_device` is the engine's resolved expert placement: an OpenCL or
    /// SYCL
    /// device there means "run the routed experts on the device", which
    /// this loader can only honour through the pool — without a budget it
    /// logs and keeps them on the host.
    ///
    /// `n_ctx` is the engine's configured context length.  The KV caches are
    /// sized to `min(context_length, KV_CAP, n_ctx)`; 0 means "no engine
    /// limit".  This matters on an accelerator: the dense weights already
    /// occupy most of VRAM, so an engine serving 4K tokens must not reserve
    /// the 256K cap (several GiB) and fail the first attention call.
    #[allow(clippy::too_many_arguments)]
    pub fn from_gguf_mmap_placed<R: Read + Seek>(
        ct: gguf_file::Content,
        raw: Option<&GgufHeader>,
        reader: &mut R,
        device: &Device,
        expert_device: &Device,
        mmap: Option<std::sync::Arc<memmap2::Mmap>>,
        file: Option<std::sync::Arc<std::fs::File>>,
        n_ctx: usize,
        device_expert_cache_bytes: Option<u64>,
    ) -> Result<Self> {
        Self::load_gguf(
            ct, raw, reader, device, expert_device, mmap, file, n_ctx,
            device_expert_cache_bytes,
            #[cfg(feature = "distributed")]
            None,
        )
    }

    #[cfg(feature = "distributed")]
    pub(crate) fn from_gguf_distributed<R: Read + Seek>(
        ct: gguf_file::Content,
        raw: &GgufHeader,
        reader: &mut R,
        mmap: Arc<memmap2::Mmap>,
        n_ctx: usize,
        cluster: Arc<crate::distributed::session::ClusterSession>,
    ) -> Result<Self> {
        Self::load_gguf(
            ct, Some(raw), reader, &Device::Cpu, &Device::Cpu, Some(mmap),
            None, n_ctx, None, Some(cluster),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn load_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        raw: Option<&GgufHeader>,
        reader: &mut R,
        device: &Device,
        expert_device: &Device,
        mmap: Option<Arc<memmap2::Mmap>>,
        file: Option<Arc<std::fs::File>>,
        n_ctx: usize,
        device_expert_cache_bytes: Option<u64>,
        #[cfg(feature = "distributed")]
        cluster: Option<Arc<crate::distributed::session::ClusterSession>>,
    ) -> Result<Self> {
        let cfg = Config::from_metadata(&ct.metadata)?;
        if !(expert_device.is_cpu()
            || expert_device.is_opencl()
            || expert_device.is_sycl()
            || expert_device.same_device(device))
        {
            candle_core::bail!(
                "deepseek4: routed experts must live on the model device, an OpenCL, SYCL or CPU device, \
                 not {expert_device:?}"
            );
        }
        // Where the device expert pool lives, if one is built at all (see
        // the doc comment above).  Decided up front so the log lines land
        // before the (long) load, and so the streamed path — which decodes
        // the experts to f32 on the model device — never pretends to cache.
        let cache_bytes = device_expert_cache_bytes.unwrap_or(0);
        let pool_device: Option<Device> = if mmap.is_none() {
            None
        } else if cache_bytes == 0 {
            if expert_device.is_opencl() || expert_device.is_sycl() {
                tracing::warn!(
                    "deepseek4: expert placement `device` requested without a VRAM expert-cache budget \
                     (--vram-expert-cache); the routed experts stay on the host"
                );
            }
            None
        } else if expert_device.is_opencl() || expert_device.is_sycl() {
            // The B50 configuration: dense set on the CPU (or the device),
            // routed experts cached on the card.
            Some(expert_device.clone())
        } else if device.is_opencl() || device.is_sycl() {
            Some(device.clone())
        } else if device.is_cpu() && expert_device.is_cpu() {
            Some(Device::Cpu)
        } else {
            tracing::info!(
                "deepseek4: {device:?} has no IQ2_XXS kernel; VRAM expert cache disabled, routed \
                 experts stay on the host"
            );
            None
        };
        // The host form of every routed expert: borrowed from the mapping on
        // the CPU whenever there is one (the source of truth, the CPU path
        // and what the device pool uploads from); a streamed load decodes
        // them to f32 on the model device instead.
        let expert_device = if mmap.is_some() {
            Device::Cpu
        } else {
            device.clone()
        };
        let mut rd = Reader {
            ct,
            raw: raw.cloned(),
            reader,
            device: device.clone(),
            expert_device,
            mmap,
            file,
            #[cfg(feature = "distributed")]
            cluster,
        };

        let tok_embeddings =
            crate::token_embedding::TokenEmbedding::load(rd.qtensor("token_embd.weight")?, device)?;
        let norm = rd.rms_norm("output_norm.weight", cfg.rms_eps)?;
        let output = match rd.qmatmul_opt("output.weight")? {
            Some(o) => o,
            None => rd.qmatmul("token_embd.weight")?,
        };

        let hc_head_fn = rd.qmatmul("output_hc_fn.weight")?;
        let hc_head_base = rd.f32_tensor("output_hc_base.weight")?;
        let hc_head_scale = rd.f32_tensor("output_hc_scale.weight")?;

        // Rope tables are capped at the same length as the KV caches: a 1M
        // context config would otherwise allocate ~256 MB of sin/cos tables.
        let max_seq = cfg.context_length.min(KV_CAP);
        // Raw (window-only) layers: plain rope_theta, no YaRN.
        let rotary_raw = Arc::new(RotaryEmbedding::new(
            &cfg,
            device,
            cfg.rope_theta,
            false,
            max_seq,
        )?);
        // Compressed layers (CSA/HCA): compress_rope_freq_base + YaRN, used for
        // both the main q/kv and the compressed-KV rows.
        let rotary_compress = Arc::new(RotaryEmbedding::new(
            &cfg,
            device,
            cfg.compress_rope_base,
            cfg.yarn.is_some(),
            max_seq,
        )?);

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = format!("blk.{i}");
            let attn_norm = rd.rms_norm(&format!("{p}.attn_norm.weight"), cfg.rms_eps)?;
            let ffn_norm = rd.rms_norm(&format!("{p}.ffn_norm.weight"), cfg.rms_eps)?;

            let q_a = rd.qmatmul(&format!("{p}.attn_q_a.weight"))?;
            // q_lora_rank is implied by the weight shapes; validate it so a
            // mismatched GGUF fails early instead of misbehaving later.
            let qb_shape = rd
                .qtensor(&format!("{p}.attn_q_b.weight"))?
                .shape()
                .dims()
                .to_vec();
            debug_assert_eq!(
                vec![cfg.n_head * cfg.head_dim, cfg.q_lora_rank],
                qb_shape,
                "blk.{i}.attn_q_b"
            );
            let q_norm = rd.rms_norm(&format!("{p}.attn_q_a_norm.weight"), cfg.rms_eps)?;
            let q_b = rd.qmatmul(&format!("{p}.attn_q_b.weight"))?;
            let wkv = rd.qmatmul(&format!("{p}.attn_kv.weight"))?;
            let kv_norm = rd.rms_norm(&format!("{p}.attn_kv_a_norm.weight"), cfg.rms_eps)?;
            let wo_a = rd.qmatmul(&format!("{p}.attn_output_a.weight"))?;
            let wo_b = rd.qmatmul(&format!("{p}.attn_output_b.weight"))?;
            let attn_sinks = rd.f32_tensor(&format!("{p}.attn_sinks.weight"))?;

            let ratio = cfg.compress_ratios[i];
            let rotary = if ratio != 0 {
                rotary_compress.clone()
            } else {
                rotary_raw.clone()
            };
            let (compressor, indexer) = if ratio != 0 {
                let comp = Compressor::load(
                    &mut rd,
                    &format!("{p}.attn_compressor"),
                    ratio,
                    cfg.head_dim,
                    cfg.rope_head_dim,
                    cfg.rms_eps,
                    false,
                )?;
                if ratio == CSA_RATIO {
                    let indexer = Indexer {
                        proj: rd.qmatmul(&format!("{p}.indexer.proj.weight"))?,
                        attn_q_b: rd.qmatmul(&format!("{p}.indexer.attn_q_b.weight"))?,
                        compressor: Compressor::load(
                            &mut rd,
                            &format!("{p}.indexer_compressor"),
                            ratio,
                            cfg.index_head_dim,
                            cfg.rope_head_dim,
                            cfg.rms_eps,
                            true,
                        )?,
                        n_head: cfg.index_n_head,
                        head_dim: cfg.index_head_dim,
                        rope_head_dim: cfg.rope_head_dim,
                        nope_head_dim: cfg.index_head_dim - cfg.rope_head_dim,
                        softmax_scale: (cfg.index_head_dim as f64).powf(-0.5),
                        topk: cfg.index_topk,
                    };
                    (Some(comp), Some(indexer))
                } else {
                    (Some(comp), None)
                }
            } else {
                (None, None)
            };

            let attn = Attention {
                q_a,
                q_norm,
                q_b,
                wkv,
                kv_norm,
                wo_a,
                wo_b,
                attn_sinks,
                compressor,
                indexer,
                ratio,
                n_head: cfg.n_head,
                head_dim: cfg.head_dim,
                rope_head_dim: cfg.rope_head_dim,
                nope_head_dim: cfg.nope_head_dim,
                o_groups: cfg.o_groups,
                o_lora_rank: cfg.o_lora_rank,
                window_size: cfg.window_size,
                softmax_scale: (cfg.head_dim as f64).powf(-0.5),
                eps: cfg.rms_eps,
                rotary: rotary.clone(),
                compress_rotary: Some(rotary_compress.clone()),
            };

            let ffn = FeedForward::Moe(load_moe(&mut rd, &p, &cfg, i, i < cfg.n_hash_layer)?);

            let hc_attn_fn = rd.qmatmul(&format!("{p}.hc_attn_fn.weight"))?;
            let hc_attn_base = rd.f32_tensor(&format!("{p}.hc_attn_base.weight"))?;
            let hc_attn_scale = rd.f32_tensor(&format!("{p}.hc_attn_scale.weight"))?;
            let hc_ffn_fn = rd.qmatmul(&format!("{p}.hc_ffn_fn.weight"))?;
            let hc_ffn_base = rd.f32_tensor(&format!("{p}.hc_ffn_base.weight"))?;
            let hc_ffn_scale = rd.f32_tensor(&format!("{p}.hc_ffn_scale.weight"))?;

            layers.push(Layer {
                attn_norm,
                attn,
                ffn_norm,
                ffn,
                hc_attn_fn,
                hc_attn_base,
                hc_attn_scale,
                hc_ffn_fn,
                hc_ffn_base,
                hc_ffn_scale,
            });
        }

        // KV caches are sized to the configured context, capped so a 1M-token
        // config does not silently reserve ~20 GB of CPU RAM per instance, and
        // further capped by the engine's own `n_ctx`.  The cap is a hard limit
        // enforced in `forward` (clear error if hit).
        let mut kv_cap = cfg.context_length.min(KV_CAP);
        if n_ctx != 0 {
            kv_cap = kv_cap.min(n_ctx);
        }
        let mut kv_states = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            kv_states.push(KvState::new(&cfg, i, device, kv_cap)?);
        }
        let hc_mult = cfg.hc_mult;
        let hc_eps = cfg.hc_eps;

        let layer_expert_ranges = rd
            .raw
            .as_ref()
            .map(|r| r.layer_expert_ranges(cfg.n_layer))
            .unwrap_or_default();
        let mmap = rd.mmap.clone();
        let file = rd.file.clone();
        // Shards retain their own mappings. Do not expose whole expert ranges
        // to the layer-ahead or residency prefetch paths.
        #[cfg(feature = "distributed")]
        let (mmap, file, layer_expert_ranges) = if rd.cluster.is_some() {
            (None, None, Vec::new())
        } else {
            (mmap, file, layer_expert_ranges)
        };

        let n_expert = cfg.n_expert;
        // The device expert pool spans every layer (one LRU keyed by
        // (layer, expert), so a layer with skewed routing can hold more
        // experts than an even split would give it); every MoE block gets a
        // handle for lookups and warming requests.
        let device_pool =
            pool_device.and_then(|dev| build_device_pool(&layers, &dev, cache_bytes, file.clone()));
        if let Some(pool) = &device_pool {
            for layer in layers.iter_mut() {
                let FeedForward::Moe(moe) = &mut layer.ffn;
                moe.device_pool = Some(Arc::clone(pool));
            }
        }
        let cpu_residency = crate::residency::CpuResidency::new(
            layers
                .iter()
                .map(|layer| {
                    // Every layer of this architecture carries an MoE block.
                    let FeedForward::Moe(moe) = &layer.ffn;
                    moe.experts.iter().map(|m| m.prefetch.clone()).collect()
                })
                .collect(),
        );
        let residency: std::sync::Arc<dyn crate::residency::ExpertResidency> = match &device_pool {
            Some(pool) => std::sync::Arc::new(
                crate::residency::CompositeResidency::with_device_pool(
                    cpu_residency,
                    Arc::clone(&pool.pool),
                    Arc::clone(&pool.uploader),
                )
                // Keeping host pages means the inclusive layout: advise
                // device-resident experts too.
                .advise_device_resident(pool.host_pages == HostPagePolicy::Keep),
            ),
            None => std::sync::Arc::new(cpu_residency),
        };
        let n_layers = layers.len();
        let shared = std::sync::Arc::new(Shared {
            tok_embeddings,
            layers,
            cfg,
            norm,
            output,
            hc_head_fn,
            hc_head_base,
            hc_head_scale,
            hc_mult,
            hc_eps,
            max_seq: kv_cap,
            device: device.clone(),
            mmap,
            file,
            layer_expert_ranges,
            residency,
            device_pool,
        });
        let prefill_timing = Arc::new(PhaseTiming::default());
        Ok(Self {
            shared,
            kv: kv_states,
            kv_seq: Vec::new(),
            last_routed: vec![Vec::new(); n_layers],
            hot_experts: crate::hot_experts::HotExpertCache::new(n_layers, n_expert, 0),
            stream_ctx: DispatchCtx {
                decode: false,
                seed_prefill: true,
                trace_row_base: 0,
                trace_chunk: 0,
                prefill_timing: Arc::clone(&prefill_timing),
            },
            prefill_timing,
        })
    }

    /// Fire the speculative prefetch for each MoE layer's predicted experts
    /// (the ids recorded by the previous forward pass — see
    /// [`ModelWeights::last_routed_experts`]).
    ///
    /// Best-effort and idempotent: a no-op for layers that have not routed
    /// yet, dense layers, and every expert whose weights are not mmap-backed
    /// (streamed loads keep no prefetch handles).
    fn prefetch_speculative(&self) {
        let pool = self.shared.device_pool.as_ref();
        for (i, ids) in self.last_routed.iter().enumerate() {
            if ids.is_empty() {
                continue;
            }
            let Some(layer) = self.shared.layers.get(i) else {
                continue;
            };
            // Every layer of this architecture carries an MoE block.
            let FeedForward::Moe(moe) = &layer.ffn;
            for &e in ids {
                // An expert resident on the device runs there; with
                // exclusive tiers its host pages are not wanted.
                if pool.is_some_and(|p| {
                    p.host_pages == HostPagePolicy::Drop && p.pool.contains(i as u32, e)
                }) {
                    continue;
                }
                if let Some(expert) = moe.experts.get(e as usize) {
                    expert.prefetch();
                }
            }
        }
    }

    /// Routed-expert ids of the most recent forward pass, per layer index
    /// (dense layers and not-yet-run layers report empty vectors).
    ///
    /// Seeded during prefill from the final prompt token's routing and
    /// refreshed by every decode step; the speculative decode prefetch
    /// fires these ids' pages before each step so they stream in behind
    /// compute instead of faulting on demand.  Diagnostics hook — also lets
    /// tests observe that routing is being tracked.
    pub fn last_routed_experts(&self) -> &[Vec<u32>] {
        &self.last_routed
    }

    /// Set the routing-frequency hot-expert cache budget (experts kept
    /// resident).  Call once after load, before serving; routing is recorded
    /// from the first forward pass and the pinned set is re-selected every
    /// [`crate::hot_experts::REFRESH_STEPS`] decode steps.
    pub fn set_pin_hot_experts(&mut self, n: usize) {
        self.hot_experts.set_budget(n);
    }

    /// Number of mmap-backed experts the host residency backend knows
    /// (informational; the host hot-expert auto-sizing input).  The device
    /// pool is reported separately by [`ModelWeights::device_expert_cache`].
    pub fn expert_residency_capacity(&self) -> usize {
        self.shared.residency.capacity()
    }

    /// The device expert pool's budget, occupancy and hit/miss counters,
    /// when a VRAM cache was built at load.
    pub fn device_expert_cache(&self) -> Option<crate::residency::DeviceCacheReport> {
        self.shared
            .device_pool
            .as_ref()
            .map(|p| crate::residency::DeviceCacheReport::of(&p.pool, Some(&p.uploader)))
    }

    /// The decode-step time split of the device expert pool (launch / host
    /// misses / device wait / rest) and the host misses' page residency,
    /// accumulated since load.  `None` without a pool.
    pub fn expert_phase_timing(&self) -> Option<ExpertPhaseTiming> {
        self.shared
            .device_pool
            .as_ref()
            .map(|p| p.decode.snapshot())
    }

    /// The context of a whole forward pass (one prompt, or one decode step
    /// for one or several sequences).
    fn ctx_whole(&self, decode: bool) -> DispatchCtx {
        DispatchCtx {
            decode,
            seed_prefill: true,
            trace_row_base: 0,
            trace_chunk: 0,
            prefill_timing: Arc::clone(&self.prefill_timing),
        }
    }

    /// Account one forward pass's wall time to the pool's decode
    /// accumulator or this session's prefill accumulator; a no-op without
    /// a pool.
    fn note_pass(&self, decode: bool, elapsed: std::time::Duration) {
        if let Some(p) = &self.shared.device_pool {
            let t: &PhaseTiming = if decode {
                &p.decode
            } else {
                &self.prefill_timing
            };
            PhaseTiming::add(&t.passes, 1);
            PhaseTiming::add(&t.pass_ns, elapsed.as_nanos() as u64);
        }
    }

    /// Log and reset this session's prefill accumulator (at the end of a
    /// prefill; `n_tokens` is unknown on the layer-streaming path).
    fn log_prefill_timing(&self, n_tokens: Option<usize>) {
        if self.shared.device_pool.is_some() {
            let t = self.prefill_timing.take();
            if t.passes > 0 && tracing::enabled!(target: "joshua", tracing::Level::DEBUG) {
                let what = match n_tokens {
                    Some(n) => format!("prefill of {n} tokens"),
                    None => "streamed prefill".to_string(),
                };
                tracing::debug!(
                    "deepseek4 {what} with the vram expert cache: {}",
                    t.describe("layer pass")
                );
            }
        }
    }

    /// Block until the background uploader has drained its queue (tests and
    /// diagnostics: makes "the misses of the last step are resident now"
    /// observable).  A no-op without a device pool.
    pub fn wait_for_expert_uploads(&self) {
        if let Some(p) = &self.shared.device_pool {
            p.uploader.wait_idle();
        }
    }

    /// A periodic `debug` line with the device pool's state (every
    /// [`crate::hot_experts::REFRESH_STEPS`] decode steps).
    fn log_device_cache(&self, step: u64) {
        if let Some(r) = self.device_expert_cache() {
            tracing::debug!(
                "deepseek4 vram expert cache @step {step}: {}/{} slots ({:.2}/{:.2} GiB), hits {} misses {} \
                 uploads {} evictions {} refused {} failed {}; upload requests {} (dropped {}), host pages released {}",
                r.resident,
                r.slots,
                r.resident_bytes as f64 / 2f64.powi(30),
                r.budget_bytes as f64 / 2f64.powi(30),
                r.stats.hits,
                r.stats.misses,
                r.stats.uploads,
                r.stats.evictions,
                r.stats.refused,
                r.stats.upload_failures,
                r.upload_requests,
                r.upload_drops,
                r.host_releases,
            );
            if let Some(t) = self.expert_phase_timing() {
                tracing::debug!(
                    "deepseek4 decode time split @step {step}: {}",
                    t.describe("decode step")
                );
            }
        }
    }

    /// Whether the token-embedding table is held quantized (diagnostics).
    pub fn embeddings_quantized(&self) -> bool {
        self.shared.tok_embeddings.is_quantized()
    }

    /// A fresh session over the same weights: shares every weight tensor
    /// (one `Arc` clone — no re-read, no re-upload) and starts with empty KV
    /// caches (per-layer + per-sequence) and a fresh routing record under the
    /// same hot-expert budget.  Two concurrent sessions on an accelerator
    /// therefore reuse the dense set that was uploaded once for the template.
    pub fn new_session(&self) -> Self {
        // A fresh session rebuilds its KV caches from scratch (the documented
        // per-session cost).  Allocation failure is unrecoverable device/CPU
        // OOM — the same condition the loader's own eager KV build fails on —
        // so panic loudly rather than hand forward a half-built cache.  Unlike
        // qwen3moe/deepseek2 (whose KV is a lazy `Option`), deepseek4 allocates
        // its KV eagerly, so `new_session` cannot be infallible by construction.
        let kv: Vec<KvState> = (0..self.shared.layers.len())
            .map(|i| {
                KvState::new(
                    &self.shared.cfg,
                    i,
                    &self.shared.device,
                    self.shared.max_seq,
                )
            })
            .collect::<Result<Vec<_>>>()
            .unwrap_or_else(|e| panic!("deepseek4: new-session KV cache allocation failed: {e}"));
        let prefill_timing = Arc::new(PhaseTiming::default());
        Self {
            shared: std::sync::Arc::clone(&self.shared),
            kv,
            kv_seq: Vec::new(),
            last_routed: vec![Vec::new(); self.shared.layers.len()],
            hot_experts: crate::hot_experts::HotExpertCache::new(
                self.shared.layers.len(),
                self.shared.cfg.n_expert,
                self.hot_experts.budget(),
            ),
            stream_ctx: DispatchCtx {
                decode: false,
                seed_prefill: true,
                trace_row_base: 0,
                trace_chunk: 0,
                prefill_timing: Arc::clone(&prefill_timing),
            },
            prefill_timing,
        }
    }

    /// Number of sessions (including this one) sharing these weights.
    pub fn shared_session_count(&self) -> usize {
        std::sync::Arc::strong_count(&self.shared)
    }

    /// Forward pass. `input` is `[1, seq_len]`; `offset` is the KV-cache
    /// position of the first input token.
    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let t_pass = std::time::Instant::now();
        let (_b, seq_len) = input.dims2()?;
        let logits = self.forward_inner(input, offset, seq_len)?;
        self.note_pass(seq_len == 1, t_pass.elapsed());
        if seq_len > 1 {
            self.log_prefill_timing(Some(seq_len));
        }
        Ok(logits)
    }

    fn forward_inner(&mut self, input: &Tensor, offset: usize, seq_len: usize) -> Result<Tensor> {
        crate::route_trace::begin_call(if seq_len == 1 {
            crate::route_trace::Phase::Decode
        } else {
            crate::route_trace::Phase::Prefill
        });
        let hc = self.shared.hc_mult;
        let d = self.shared.tok_embeddings.hidden()?;

        let tok = self
            .shared
            .tok_embeddings
            .forward(&input.flatten_all()?)?
            .reshape((1, seq_len, d))?;
        // Expand to hc copies.
        let mut xs = tok.unsqueeze(2)?.broadcast_as((1, seq_len, hc, d))?;

        let profile = std::env::var_os("JOSHUA_PROFILE_LAYERS").is_some();
        let mut prof = if profile {
            Some((
                Vec::with_capacity(self.shared.layers.len()),
                Vec::with_capacity(self.shared.layers.len()),
                Vec::with_capacity(self.shared.layers.len()),
            ))
        } else {
            None
        };

        // Prefill reads ~1.35 GB of routed-expert weights per layer.  The
        // dispatch-level prefetch covers the current layer's *selected*
        // experts but fires right before the expert loop — no head start, so
        // the matmuls still stall on page faults.  For real prefill
        // (seq_len ≥ PREFETCH_AHEAD_MIN), a background thread preads this and
        // the next few layers' expert spans into the page cache while the
        // layer loop computes (see [`LayerPrefetcher`]); the kernel streams at
        // full device bandwidth and the layer's mmap faults become cache hits.
        // The madvise hints below only need the mapping; the pread thread
        // additionally needs the model file handle, so loads that supply a
        // mapping but no file (public `from_gguf_mmap(..., Some(mmap), None)`)
        // keep the hints and just skip the thread.
        let prefetch_layers = seq_len >= PREFETCH_AHEAD_MIN
            && !self.shared.layer_expert_ranges.is_empty()
            && self.shared.mmap.is_some();
        let prefetch_thread = prefetch_layers && self.shared.file.is_some();

        let prefetcher = if prefetch_thread {
            Some(crate::mmap_tensor::LayerPrefetcher::spawn(
                self.shared.file.as_ref().expect("checked above").clone(),
                std::sync::Arc::new(self.shared.layer_expert_ranges.clone()),
                crate::mmap_tensor::prefetch_ahead_depth(),
            ))
        } else {
            None
        };

        // Routing-frequency hot-expert cache: every HOT_EXPERTS_REFRESH_STEPS
        // decode steps, re-select the most-used experts (recency as the
        // tie-break) and WILLNEED their pages, so the common routing path
        // stays resident instead of faulting from disk each step.  Runs
        // before the layer loop so the prefetch has a full step of compute
        // to stream in behind.  The per-token routing recorded below feeds
        // the counters.
        let step = self.hot_experts.begin_step(seq_len == 1);
        if self.hot_experts.refresh_due(seq_len == 1) {
            let hot = self.hot_experts.refresh();
            self.shared.residency.replace_hot_set(&hot);
            for (l, e) in hot {
                self.shared.residency.acquire(l, e);
            }
        }
        if seq_len == 1 && step.is_multiple_of(crate::hot_experts::REFRESH_STEPS) {
            self.log_device_cache(step);
        }

        // Speculative next-step expert prefetch (decode only): routing is
        // temporally local — the step now being generated routes to a large
        // extent to the experts the previous step chose (measured across
        // model families).  Firing WILLNEED for each MoE layer's *predicted*
        // experts before any layer runs gives those pages a full step of
        // compute to stream in the background, instead of the dispatch-level
        // prefetch below, which fires right before the expert matmuls with
        // no head start.  Mispredictions are cheap: the advice call is
        // idempotent and best-effort, and the dispatch-level prefetch still
        // covers whatever was missed.
        if seq_len == 1 {
            self.prefetch_speculative();
        }

        let ctx = self.ctx_whole(seq_len == 1);
        // Field-split borrows so the loop can record each layer's routing.
        // `shared.layers` is immutable (Arc) and only ever read here; the mutable
        // per-session state (`last_routed`, `hot_experts`) is captured separately.
        let layers = &self.shared.layers;
        let last_routed = &mut self.last_routed;
        for (i, layer) in layers.iter().enumerate() {
            let t_layer = std::time::Instant::now();
            if let Some(pf) = &prefetcher {
                pf.set_current(i);
            }
            if i == 0 && prefetch_layers {
                if let Some(mmap) = &self.shared.mmap {
                    // The lazy-weights path advises MADV_RANDOM over the
                    // whole mapping, which disables kernel readahead: every
                    // demand fault is a single 4 KiB page read (~175 MB/s
                    // effective).  Prefill walks the expert ranges in file
                    // order, so switch the whole expert span to SEQUENTIAL —
                    // one call — and the kernel's sequential detector streams
                    // it at full device bandwidth (1.9 GB/s here) while the
                    // layer loop computes.  The tensor-major MoE dispatch
                    // below then reads each expert tensor as one clean pass.
                    if let (Some(Some((b0, _))), Some(Some((_, e_n)))) = (
                        self.shared.layer_expert_ranges.first(),
                        self.shared.layer_expert_ranges.last(),
                    ) {
                        let span = e_n.saturating_sub(*b0);
                        if span > 0 && *e_n <= mmap.len() {
                            let _ = mmap.advise_range(memmap2::Advice::Sequential, *b0, span);
                        }
                    }
                }
            }
            // hc_pre with attention weights
            let (x, post, comb) = hc_pre(
                &xs,
                &layer.hc_attn_fn,
                &layer.hc_attn_scale,
                &layer.hc_attn_base,
                self.shared.hc_eps,
                self.shared.cfg.hc_sinkhorn_iters,
            )?;
            let residual = xs;
            let h = layer.attn_norm.forward(&x)?;
            let h = layer
                .attn
                .forward(&mut self.kv[i], &h, offset, self.shared.max_seq)?;
            xs = hc_post(&h, &residual, &post, &comb)?;
            if let Some((a, _, _)) = prof.as_mut() {
                a.push(t_layer.elapsed().as_secs_f64());
            }
            let t_moe = std::time::Instant::now();

            let (x, post, comb) = hc_pre(
                &xs,
                &layer.hc_ffn_fn,
                &layer.hc_ffn_scale,
                &layer.hc_ffn_base,
                self.shared.hc_eps,
                self.shared.cfg.hc_sinkhorn_iters,
            )?;
            let residual = xs;
            let h = layer.ffn_norm.forward(&x)?;
            let (h, routed_ids) = layer.ffn.forward(&h, input, ctx.clone())?;
            last_routed[i] = routed_ids;
            self.hot_experts.record(i, &last_routed[i], step);
            xs = hc_post(&h, &residual, &post, &comb)?;
            if let Some((_, m, _)) = prof.as_mut() {
                m.push(t_moe.elapsed().as_secs_f64());
            }
            if let Some((_, _, t)) = prof.as_mut() {
                t.push(t_layer.elapsed().as_secs_f64());
            }
        }

        if let Some((a, m, t)) = prof {
            let (sa, sm, st): (f64, f64, f64) = (a.iter().sum(), m.iter().sum(), t.iter().sum());
            let (ma, mm) = (
                a.iter().copied().fold(0.0, f64::max),
                m.iter().copied().fold(0.0, f64::max),
            );
            eprintln!(
                "[prof] attn: total {sa:.1}s avg {:.3}s max {ma:.3}s | moe: total {sm:.1}s avg {:.3}s max {mm:.3}s | layer: total {st:.1}s",
                sa / a.len() as f64,
                sm / m.len() as f64
            );
            // print the 6 slowest layers' breakdown
            let mut idx: Vec<usize> = (0..t.len()).collect();
            idx.sort_by(|x, y| t[*y].total_cmp(&t[*x]));
            eprintln!("[prof] slowest layers (layer: attn/moe/total):");
            for i in idx.iter().take(6) {
                eprintln!("  blk.{i}: {:.3}s / {:.3}s / {:.3}s", a[*i], m[*i], t[*i]);
            }
        }

        // Stop the prefetch thread: prefill is over, and decode reads a handful
        // of experts per layer — the thread's sequential stream would fight the
        // random-access hint below.
        if let Some(mut pf) = prefetcher {
            pf.stop();
        }

        // Restore the sparse-access hint after a prefill pass: decode reads a
        // handful of experts per layer, so SEQUENTIAL's aggressive readahead
        // would waste bandwidth pulling the wrong pages.
        if prefetch_layers {
            if let Some(mmap) = &self.shared.mmap {
                if let (Some(Some((b0, _))), Some(Some((_, e_n)))) = (
                    self.shared.layer_expert_ranges.first(),
                    self.shared.layer_expert_ranges.last(),
                ) {
                    let span = e_n.saturating_sub(*b0);
                    if span > 0 && *e_n <= mmap.len() {
                        let _ = mmap.advise_range(memmap2::Advice::Random, *b0, span);
                    }
                }
            }
        }

        // Parallel head: collapse hc copies, RMS, output matmul.
        // pre = sigmoid(mixes * hc_head_scale + hc_head_base) + eps
        let flat = xs.reshape((seq_len, hc * d))?;
        let rsqrt = flat
            .sqr()?
            .mean_keepdim(D::Minus1)?
            .affine(1.0, self.shared.hc_eps)?
            .powf(-0.5)?;
        let mixes = self
            .shared
            .hc_head_fn
            .forward(&flat)?
            .broadcast_mul(&rsqrt)?; // [s, hc]
        let pre = sigmoid(
            &mixes
                .broadcast_mul(&self.shared.hc_head_scale)?
                .broadcast_add(&self.shared.hc_head_base)?,
        )?
        .affine(1.0, self.shared.hc_eps)?; // + eps
        let y = pre
            .unsqueeze(D::Minus1)?
            .broadcast_as((seq_len, hc, d))?
            .mul(&xs.squeeze(0)?)?
            .sum(D::Minus2)?; // [s, d]

        // Only the last position's logits are needed, and the engine's
        // `squeeze_batch_logits` requires a single row.
        let y = y.narrow(0, seq_len - 1, 1)?;
        let y = self.shared.norm.forward(&y)?;
        let logits = self.shared.output.forward(&y)?; // [1, n_vocab]
        logits.to_dtype(DType::F32)
    }

    /// Forward one decode step for `n_seq` **independent** sequences at their
    /// own `(1 token, position)` pairs, amortizing the routed-expert fetch and
    /// the MoE dispatch across the batch (each routed expert is fetched once
    /// and applied to every token that routes to it).
    ///
    /// The **attention** branch stays per-sequence — each sequence reads and
    /// writes its own [`KvState`] at its own position, so sequences never see
    /// one another's keys.  The **MoE/FFN** branch runs once on the
    /// concatenation of the per-sequence layer outputs, which is exact because
    /// `dispatch` is sequence-agnostic (each token's routing only depends on
    /// its own hidden state).  This is the engine-facing entry for
    /// `JOSHUA_BATCH`.
    ///
    /// Returns the per-sequence logits (one `n_vocab` vector each), split back
    /// from the shared output head.
    pub fn forward_sequences(&mut self, seqs: &[(&Tensor, usize)]) -> Result<Vec<Vec<f32>>> {
        let t_pass = std::time::Instant::now();
        let out = self.forward_sequences_inner(seqs)?;
        if !seqs.is_empty() {
            self.note_pass(true, t_pass.elapsed());
        }
        Ok(out)
    }

    fn forward_sequences_inner(&mut self, seqs: &[(&Tensor, usize)]) -> Result<Vec<Vec<f32>>> {
        let n_seq = seqs.len();
        if n_seq == 0 {
            return Ok(Vec::new());
        }
        crate::route_trace::begin_call(crate::route_trace::Phase::Decode);
        let hc = self.shared.hc_mult;
        let d = self.shared.tok_embeddings.hidden()?;

        // Persistent per-sequence KV: reset when the batch size changes (a new
        // batch of sequences starts), otherwise reuse the cache across steps so
        // multi-token generation keeps each sequence's attention history.
        if self.kv_seq.len() != self.shared.layers.len()
            || self.kv_seq.first().unwrap_or(&Vec::new()).len() != n_seq
        {
            self.kv_seq.clear();
            for i in 0..self.shared.layers.len() {
                let mut per_seq: Vec<KvState> = Vec::with_capacity(n_seq);
                for _ in 0..n_seq {
                    per_seq.push(KvState::new(
                        &self.shared.cfg,
                        i,
                        &self.shared.device,
                        self.shared.max_seq,
                    )?);
                }
                self.kv_seq.push(per_seq);
            }
        }

        // Per-sequence embeddings -> [1, 1, hc, d].
        let mut xs_seq: Vec<Tensor> = Vec::with_capacity(n_seq);
        let mut ids_cat: Vec<u32> = Vec::new();
        for (input, _) in seqs {
            let tok = self
                .shared
                .tok_embeddings
                .forward(&input.flatten_all()?)?
                .reshape((1, 1, d))?;
            xs_seq.push(tok.unsqueeze(2)?.broadcast_as((1, 1, hc, d))?);
            ids_cat.push(
                input
                    .flatten_all()?
                    .to_vec1()?
                    .first()
                    .copied()
                    .unwrap_or(0),
            );
        }

        let step = self.hot_experts.begin_step(true);
        if self.hot_experts.refresh_due(true) {
            let hot = self.hot_experts.refresh();
            self.shared.residency.replace_hot_set(&hot);
            for (l, e) in hot {
                self.shared.residency.acquire(l, e);
            }
        }
        if step.is_multiple_of(crate::hot_experts::REFRESH_STEPS) {
            self.log_device_cache(step);
        }

        let ctx = self.ctx_whole(true);
        for i in 0..self.shared.layers.len() {
            let layer = &self.shared.layers[i];

            // Per-sequence attention branch, then stash everything the shared
            // MoE branch needs to hc_post back per-sequence.
            let mut ffn_pre: Vec<Tensor> = Vec::with_capacity(n_seq); // h to feed the MoE
            let mut ffn_residual: Vec<Tensor> = Vec::with_capacity(n_seq);
            let mut ffn_post: Vec<Tensor> = Vec::with_capacity(n_seq);
            let mut ffn_comb: Vec<Tensor> = Vec::with_capacity(n_seq);
            for (s, (_, off)) in seqs.iter().enumerate() {
                let xs_s = &xs_seq[s];
                let (x, post, comb) = hc_pre(
                    xs_s,
                    &layer.hc_attn_fn,
                    &layer.hc_attn_scale,
                    &layer.hc_attn_base,
                    self.shared.hc_eps,
                    self.shared.cfg.hc_sinkhorn_iters,
                )?;
                let residual = xs_s;
                let h = layer.attn_norm.forward(&x)?;
                let h = layer.attn.forward(
                    &mut self.kv_seq.get_mut(i).unwrap().get_mut(s).unwrap(),
                    &h,
                    *off,
                    self.shared.max_seq,
                )?;
                let xs_s2 = hc_post(&h, &residual, &post, &comb)?;

                let (x2, fpost, fcomb) = hc_pre(
                    &xs_s2,
                    &layer.hc_ffn_fn,
                    &layer.hc_ffn_scale,
                    &layer.hc_ffn_base,
                    self.shared.hc_eps,
                    self.shared.cfg.hc_sinkhorn_iters,
                )?;
                ffn_pre.push(layer.ffn_norm.forward(&x2)?);
                ffn_residual.push(xs_s2);
                ffn_post.push(fpost);
                ffn_comb.push(fcomb);
            }

            // Shared MoE: concatenate the per-seq FFN inputs + token ids.
            let ffn_cats: Vec<Tensor> = (0..n_seq)
                .map(|s| ffn_pre.get(s).unwrap().clone())
                .collect();
            let h_cat = Tensor::cat(&ffn_cats, 1)?;
            let input_cat = Tensor::new(ids_cat.as_slice(), &self.shared.device)
                .and_then(|t| t.unsqueeze(0))?;
            let (h_out, _routed) = layer.ffn.forward(&h_cat, &input_cat, ctx.clone())?;

            // Split the MoE output back per-sequence and hc_post each.
            let (_, n_tok, _) = h_out.dims3()?;
            let per_seq = n_tok / n_seq;
            for s in 0..n_seq {
                let h_out_s = h_out.narrow(1, s * per_seq, per_seq)?;
                let xs_s2 = hc_post(&h_out_s, &ffn_residual[s], &ffn_post[s], &ffn_comb[s])?;
                xs_seq[s] = xs_s2;
            }
        }

        // Logits head on the concatenated last-layer output.
        // Concatenate xs_seq along the token dim -> [1, n_seq, hc, d].
        let xs_cats: Vec<Tensor> = (0..n_seq).map(|s| xs_seq.get(s).unwrap().clone()).collect();
        let xs_cat = Tensor::cat(&xs_cats, 1)?;
        let seq_len = n_seq;
        let flat = xs_cat.reshape((seq_len, hc * d))?;
        let rsqrt = flat
            .sqr()?
            .mean_keepdim(D::Minus1)?
            .affine(1.0, self.shared.hc_eps)?
            .powf(-0.5)?;
        let mixes = self
            .shared
            .hc_head_fn
            .forward(&flat)?
            .broadcast_mul(&rsqrt)?;
        let pre = sigmoid(
            &mixes
                .broadcast_mul(&self.shared.hc_head_scale)?
                .broadcast_add(&self.shared.hc_head_base)?,
        )?
        .affine(1.0, self.shared.hc_eps)?;
        let y = pre
            .unsqueeze(D::Minus1)?
            .broadcast_as((seq_len, hc, d))?
            .mul(&xs_cat.squeeze(0)?)?
            .sum(D::Minus2)?; // [n_seq, d]
        let y = self.shared.norm.forward(&y)?;
        let logits_all = self.shared.output.forward(&y)?; // [n_seq, n_vocab]
                                                          // Split logits per sequence (each is 1 row).
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(n_seq);
        for s in 0..n_seq {
            let row = logits_all
                .narrow(0, s, 1)?
                .squeeze(0)?
                .to_dtype(DType::F32)?
                .to_vec1()?;
            out.push(row);
        }
        Ok(out)
    }

    /// Routed expert bytes addressable by this rank versus the unsharded total.
    #[cfg(feature = "distributed")]
    pub(crate) fn distributed_expert_bytes(&self) -> (usize, usize) {
        debug_assert!(self.distributed_prefetch_is_shard_only());
        self.shared.layers.iter().fold((0, 0), |(local, full), layer| {
            let FeedForward::Moe(moe) = &layer.ffn;
            match &moe.distributed {
                Some(shards) => (local + shards.local_bytes, full + shards.full_bytes),
                None => (local, full),
            }
        })
    }

    #[cfg(feature = "distributed")]
    fn distributed_prefetch_is_shard_only(&self) -> bool {
        self.shared.mmap.is_none()
            && self.shared.file.is_none()
            && self.shared.layer_expert_ranges.is_empty()
            && self.shared.device_pool.is_none()
            && self.shared.layers.iter().all(|layer| {
                let FeedForward::Moe(moe) = &layer.ffn;
                moe.experts.is_empty() && moe.distributed.is_some() && moe.device_pool.is_none()
            })
    }

    /// Number of routed experts whose weights are mmap-backed (and therefore
    /// prefetchable) vs the total, for diagnostics.
    pub fn mmap_backed_experts(&self) -> (usize, usize) {
        let (mut backed, mut total) = (0, 0);
        for layer in &self.shared.layers {
            let FeedForward::Moe(moe) = &layer.ffn;
            for e in &moe.experts {
                total += 1;
                if e.prefetch.is_some() {
                    backed += 1;
                }
            }
        }
        (backed, total)
    }

    /// Reset the KV caches so this instance can serve an unrelated prompt.
    pub fn clear_kv_cache(&mut self) {
        let dev = self.shared.device.clone();
        for (i, kv) in self.kv.iter_mut().enumerate() {
            match KvState::new(&self.shared.cfg, i, &dev, self.shared.max_seq) {
                Ok(n) => *kv = n,
                Err(e) => eprintln!("deepseek4: failed to reset KV state for layer {i}: {e}"),
            }
        }
        // The batched path keeps its own per-sequence history and only
        // rebuilds it when the batch size changes, so a same-sized batch
        // after a reset would otherwise continue the previous batch's
        // attention.  Drop it; `forward_sequences` re-creates it empty.
        self.kv_seq.clear();
        // The speculative prefetch predicts from the *previous step's*
        // routing; after a reset that routing belongs to whatever ran on
        // this instance before, so drop it.  (Advice-only either way — a
        // stale prediction can waste readahead, never change logits.)
        for ids in &mut self.last_routed {
            ids.clear();
        }
    }
}

// ─── Layer-streaming prefill (shared framework) ──────────────────────────────
impl crate::stream_prefill::StreamPrefill for ModelWeights {
    fn n_layers(&self) -> usize {
        self.shared.layers.len()
    }

    fn begin_stream(&mut self) {
        crate::route_trace::begin_call(crate::route_trace::Phase::Prefill);
    }

    fn set_stream_progress(&mut self, chunk: usize, n_chunks: usize, row_base: usize) {
        // Only the final chunk's last row is the routing the decode that
        // follows continues from; the trace numbers rows across the prompt.
        self.stream_ctx = DispatchCtx {
            decode: false,
            seed_prefill: chunk + 1 == n_chunks,
            trace_row_base: row_base,
            trace_chunk: chunk,
            prefill_timing: Arc::clone(&self.prefill_timing),
        };
    }

    fn embed_chunk(&self, tokens: &[u32], device: &candle_core::Device) -> Result<Tensor> {
        let hc = self.shared.hc_mult;
        let d = self.shared.tok_embeddings.hidden()?;
        let tok = self
            .shared
            .tok_embeddings
            .forward(&Tensor::new(tokens.to_vec(), device)?.unsqueeze(0)?)?
            .reshape((1, tokens.len(), d))?;
        // `xs` form: expand to the hc copies.
        tok.unsqueeze(2)?.broadcast_as((1, tokens.len(), hc, d))
    }

    fn apply_layer_chunk(
        &mut self,
        l: usize,
        xs: &Tensor,
        pos: usize,
        tokens: &[u32],
    ) -> Result<Tensor> {
        let t_pass = std::time::Instant::now();
        let out = self.apply_layer_chunk_inner(l, xs, pos, tokens)?;
        self.note_pass(false, t_pass.elapsed());
        Ok(out)
    }

    fn final_logits(&self, last: &Tensor) -> Result<Tensor> {
        let logits = self.final_logits_inner(last);
        self.log_prefill_timing(None);
        logits
    }
}

impl ModelWeights {
    fn apply_layer_chunk_inner(
        &mut self,
        l: usize,
        xs: &Tensor,
        pos: usize,
        tokens: &[u32],
    ) -> Result<Tensor> {
        let layer = &self.shared.layers[l];
        let kv = &mut self.kv[l];
        let hc_eps = self.shared.hc_eps;
        let sinkhorn = self.shared.cfg.hc_sinkhorn_iters;
        let max_seq = self.shared.max_seq;

        // hc_pre with attention weights.
        let (x, post, comb) = hc_pre(
            xs,
            &layer.hc_attn_fn,
            &layer.hc_attn_scale,
            &layer.hc_attn_base,
            hc_eps,
            sinkhorn,
        )?;
        let residual = xs.clone();
        let h = layer.attn_norm.forward(&x)?;
        let h = layer.attn.forward(kv, &h, pos, max_seq)?;
        let xs = hc_post(&h, &residual, &post, &comb)?;

        // hc_pre with FFN weights, then the MoE block.
        let (x, post, comb) = hc_pre(
            &xs,
            &layer.hc_ffn_fn,
            &layer.hc_ffn_scale,
            &layer.hc_ffn_base,
            hc_eps,
            sinkhorn,
        )?;
        let residual = xs.clone();
        let h = layer.ffn_norm.forward(&x)?;
        let input = Tensor::new(tokens.to_vec(), &self.shared.device)?.unsqueeze(0)?;
        let (h, routed_ids) = layer.ffn.forward(&h, &input, self.stream_ctx.clone())?;
        self.last_routed[l] = routed_ids;
        // Advisory: routing recording feeds the (hot-expert) prefetch policy,
        // never the logits.  Record against a fixed step; prefilter streaming
        // does not advance the decode clock.
        let step = self.hot_experts.begin_step(false);
        self.hot_experts.record(l, &self.last_routed[l], step);
        hc_post(&h, &residual, &post, &comb)
    }

    fn final_logits_inner(&self, last: &Tensor) -> Result<Tensor> {
        let (_, seq_len, hc, d) = last.dims4()?;
        let flat = last.reshape((seq_len, hc * d))?;
        let rsqrt = flat
            .sqr()?
            .mean_keepdim(D::Minus1)?
            .affine(1.0, self.shared.hc_eps)?
            .powf(-0.5)?;
        let mixes = self
            .shared
            .hc_head_fn
            .forward(&flat)?
            .broadcast_mul(&rsqrt)?;
        let pre = sigmoid(
            &mixes
                .broadcast_mul(&self.shared.hc_head_scale)?
                .broadcast_add(&self.shared.hc_head_base)?,
        )?
        .affine(1.0, self.shared.hc_eps)?;
        let y = pre
            .unsqueeze(D::Minus1)?
            .broadcast_as((seq_len, hc, d))?
            .mul(&last.squeeze(0)?)?
            .sum(D::Minus2)?; // [seq, d]
        let y = y.narrow(0, seq_len - 1, 1)?;
        let y = self.shared.norm.forward(&y)?;
        let logits = self.shared.output.forward(&y)?.to_dtype(DType::F32)?;
        Ok(logits)
    }
}

/// Slice an IQ2_XXS expert tensor (`[n_expert, out, in]`, GGUF dims reversed)
/// into per-expert [`QMatMul`]s whose blocks stay in the mapping.
fn split_mxfp4_experts<R: Read + Seek>(
    rd: &mut Reader<R>,
    name: &str,
    info: &crate::gguf_ext::RawTensorInfo,
    n_expert: usize,
) -> Result<Vec<ExpertTensor>> {
    let dims = info.dims.clone();
    if dims.len() != 3 || dims[0] != n_expert {
        candle_core::bail!(
            "deepseek4: expected expert tensor `{name}` shaped [n_expert, out, in], got {dims:?}"
        );
    }
    let (out, inn) = (dims[1], dims[2]);
    let per_elems = out * inn;
    let mxfp4_block = std::mem::size_of::<crate::mxfp4::BlockMxfp4>();
    if !per_elems.is_multiple_of(crate::mxfp4::QK_MXFP4) {
        candle_core::bail!(
            "deepseek4: MXFP4 expert `{name}` rows {out}x{inn} are not a multiple of {}",
            crate::mxfp4::QK_MXFP4
        );
    }
    let per_bytes = per_elems / crate::mxfp4::QK_MXFP4 * mxfp4_block;
    let tensor_data_offset = rd
        .raw
        .as_ref()
        .map(|r| r.tensor_data_offset)
        .unwrap_or(rd.ct.tensor_data_offset);

    // Zero-copy: one borrowed QTensor per expert, pointing into the mapping.
    // A borrow is `QStorage::Cpu` by construction, so it is only taken when
    // the experts' home device is the CPU — which it is for every mapped
    // model, including one whose dense set runs on an accelerator (see
    // `Reader::expert_device`).  Otherwise decode to f32 and copy onto
    // `rd.expert_device` below.
    if let Some(mmap) = rd.mmap.as_ref().filter(|_| rd.expert_device.is_cpu()) {
        let base = tensor_data_offset.saturating_add(info.offset) as usize;
        let mut experts = Vec::with_capacity(n_expert);
        for e in 0..n_expert {
            match crate::mmap_tensor::borrowed_range_mxfp4(
                mmap,
                base + e * per_bytes,
                (out, inn).into(),
            )? {
                Some(qt) => experts.push(ExpertTensor {
                    qmatmul: QMatMul::from_qtensor(qt)?,
                    prefetch: crate::mmap_tensor::prefetch_handle_mxfp4(
                        mmap,
                        base + e * per_bytes,
                        per_elems / crate::mxfp4::QK_MXFP4,
                    ),
                }),
                None => {
                    experts.clear();
                    break;
                }
            }
        }
        if experts.len() == n_expert {
            return Ok(experts);
        }
        tracing::warn!("deepseek4: could not borrow `{name}` from the mapping, decoding to f32");
    }

    // The fallback below materializes the whole stacked tensor as f32 — an
    // order-of-magnitude blow-up for real model footprints (~9 GB of 4-bit
    // data per stacked tensor becomes ~140 GB), and accelerator devices cannot
    // borrow the blocks (they are CPU storage).  Refuse loudly rather than OOM.
    if !rd.expert_device.is_cpu() {
        candle_core::bail!(
            "deepseek4: MXFP4 expert tensor `{name}` is only supported on the CPU device"
        );
    }

    // No mapping (or borrow declined): decode the whole tensor to f32 and hand
    // each expert over as an f32 QMatMul.  Only reachable in tests for the
    // production footprint of these tensors.
    let bytes = rd.raw_bytes_from(name)?;
    let blocks = crate::mxfp4::blocks_from_bytes(&bytes)?;
    let mut all = vec![0f32; info.elem_count()];
    crate::mxfp4::dequantize(blocks, &mut all)?;
    let mut experts = Vec::with_capacity(n_expert);
    for e in 0..n_expert {
        let t = Tensor::from_vec(
            all[e * per_elems..(e + 1) * per_elems].to_vec(),
            (out, inn),
            &rd.expert_device,
        )?;
        experts.push(ExpertTensor {
            qmatmul: QMatMul::from_qtensor(QTensor::quantize(&t, GgmlDType::F32)?)?,
            prefetch: None,
        });
    }
    Ok(experts)
}

fn split_iq2xxs_experts<R: Read + Seek>(
    rd: &mut Reader<R>,
    name: &str,
    info: &crate::gguf_ext::RawTensorInfo,
    n_expert: usize,
) -> Result<Vec<ExpertTensor>> {
    let dims = info.dims.clone();
    if dims.len() != 3 || dims[0] != n_expert {
        candle_core::bail!(
            "deepseek4: expected expert tensor `{name}` shaped [n_expert, out, in], got {dims:?}"
        );
    }
    let (out, inn) = (dims[1], dims[2]);
    let per_elems = out * inn;
    if !per_elems.is_multiple_of(crate::iq2xxs::QK_IQ2_XXS) {
        candle_core::bail!(
            "deepseek4: IQ2_XXS expert `{name}` rows {out}x{inn} are not a multiple of {}",
            crate::iq2xxs::QK_IQ2_XXS
        );
    }
    let per_bytes = per_elems / crate::iq2xxs::QK_IQ2_XXS * crate::iq2xxs::BLOCK_BYTES;
    let tensor_data_offset = rd
        .raw
        .as_ref()
        .map(|r| r.tensor_data_offset)
        .unwrap_or(rd.ct.tensor_data_offset);

    // Zero-copy: one borrowed QTensor per expert, pointing into the mapping.
    // The borrow is `QStorage::Cpu` by construction and is the source of
    // truth for both the host path and the device expert cache (which
    // uploads an active expert's blocks from the same bytes).
    if let Some(mmap) = rd.mmap.as_ref().filter(|_| rd.expert_device.is_cpu()) {
        let base = tensor_data_offset.saturating_add(info.offset) as usize;
        let mut experts = Vec::with_capacity(n_expert);
        for e in 0..n_expert {
            match crate::mmap_tensor::borrowed_range_iq2xxs(
                mmap,
                base + e * per_bytes,
                (out, inn).into(),
            )? {
                Some(qt) => experts.push(ExpertTensor {
                    qmatmul: QMatMul::from_qtensor(qt)?,
                    prefetch: crate::mmap_tensor::prefetch_handle_iq2xxs(
                        mmap,
                        base + e * per_bytes,
                        per_elems / crate::iq2xxs::QK_IQ2_XXS,
                    ),
                }),
                None => {
                    experts.clear();
                    break;
                }
            }
        }
        if experts.len() == n_expert {
            return Ok(experts);
        }
        tracing::warn!("deepseek4: could not borrow `{name}` from the mapping, decoding to f32");
    }

    // The fallback below materializes the whole stacked tensor as f32 — an
    // order-of-magnitude blow-up for real model footprints (~40 GB of 2-bit
    // data becomes ~640 GB) — and accelerator devices cannot borrow the
    // blocks (they are CPU storage).  Refuse loudly rather than OOM.
    if !rd.expert_device.is_cpu() {
        candle_core::bail!(
            "deepseek4: IQ2_XXS expert tensor `{name}` is only supported on the CPU device"
        );
    }

    // No mapping (or borrow declined): decode the whole tensor to f32 and hand
    // each expert over as an f32 QMatMul.  Only reachable in tests for the
    // production footprint of these tensors.
    let bytes = rd.raw_bytes_from(name)?;
    let blocks = crate::iq2xxs::blocks_from_bytes(&bytes)?;
    let mut all = vec![0f32; info.elem_count()];
    crate::iq2xxs::dequantize(blocks, &mut all)?;
    let mut experts = Vec::with_capacity(n_expert);
    for e in 0..n_expert {
        let t = Tensor::from_vec(
            all[e * per_elems..(e + 1) * per_elems].to_vec(),
            (out, inn),
            &rd.expert_device,
        )?;
        experts.push(ExpertTensor {
            qmatmul: QMatMul::from_qtensor(QTensor::quantize(&t, GgmlDType::F32)?)?,
            prefetch: None,
        });
    }
    Ok(experts)
}

fn load_moe<R: Read + Seek>(
    rd: &mut Reader<R>,
    p: &str,
    cfg: &Config,
    layer: usize,
    hash: bool,
) -> Result<Moe> {
    // Router weight `[n_expert, n_embd]`, kept transposed + contiguous for
    // the per-call `x @ Wᵀ` (numerically identical to transposing per call).
    let gate_t = rd
        .f32_tensor(&format!("{p}.ffn_gate_inp.weight"))?
        .t()?
        .contiguous()?;
    let gate_bias = if hash {
        None
    } else {
        rd.has(&format!("{p}.exp_probs_b.bias"))
            .then(|| rd.f32_tensor(&format!("{p}.exp_probs_b.bias")))
            .transpose()?
    };
    // A hash layer routes through the table on every token, so a missing one is
    // a load error rather than something to discover on the first forward pass.
    let tid2eid = if hash {
        // The id table ships as I32 in every DeepSeek-V4-Flash GGUF; candle
        // cannot name that dtype, so the raw header is used.
        Some(rd.i32_to_f32_tensor(&format!("{p}.ffn_gate_tid2eid.weight"))?)
    } else {
        None
    };
    if let Some(t) = &tid2eid {
        // `dispatch` unflattens the routed ids assuming `n_expert_used` per token.
        let dims = t.shape().dims().to_vec();
        if dims.len() != 2 || dims[1] != cfg.n_expert_used {
            candle_core::bail!(
                "deepseek4: `{p}.ffn_gate_tid2eid.weight` should be [n_vocab, {}], got {dims:?}",
                cfg.n_expert_used
            );
        }
    }

    let clamp = cfg.swiglu_clamp.get(layer).copied().unwrap_or(0.0);
    #[cfg(feature = "distributed")]
    let distributed = if rd.cluster.is_some() {
        Some(load_distributed_experts(rd, p, cfg, gate_t.dim(0)?, clamp)
            .map_err(candle_core::Error::wrap)?)
    } else {
        None
    };
    #[cfg(feature = "distributed")]
    let sharded = distributed.is_some();
    #[cfg(not(feature = "distributed"))]
    let sharded = false;
    let experts = if sharded {
        Vec::new()
    } else {
    let gate_exps = split_experts(rd, &format!("{p}.ffn_gate_exps.weight"), cfg.n_expert)?;
    let up_exps = split_experts(rd, &format!("{p}.ffn_up_exps.weight"), cfg.n_expert)?;
    let down_exps = split_experts(rd, &format!("{p}.ffn_down_exps.weight"), cfg.n_expert)?;
    gate_exps
        .into_iter()
        .zip(up_exps)
        .zip(down_exps)
        .map(|((gate, up), down)| {
            let prefetch = match (&gate.prefetch, &up.prefetch, &down.prefetch) {
                (Some(g), Some(u), Some(d)) => Some(crate::residency::ExpertHandles {
                    gate: g.clone(),
                    up: u.clone(),
                    down: d.clone(),
                }),
                _ => None,
            };
            Mlp {
                gate: gate.qmatmul,
                up: up.qmatmul,
                down: down.qmatmul,
                clamp,
                prefetch,
            }
        })
        .collect()
    };

    let shared = if cfg.n_expert_shared > 0 {
        Some(Mlp {
            gate: rd.qmatmul(&format!("{p}.ffn_gate_shexp.weight"))?,
            up: rd.qmatmul(&format!("{p}.ffn_up_shexp.weight"))?,
            down: rd.qmatmul(&format!("{p}.ffn_down_shexp.weight"))?,
            clamp: cfg.swiglu_clamp_shexp.get(layer).copied().unwrap_or(0.0),
            prefetch: None,
        })
    } else {
        None
    };

    Ok(Moe {
        gate_t,
        gate_bias,
        tid2eid,
        experts,
        #[cfg(feature = "distributed")]
        distributed,
        shared,
        n_expert_used: cfg.n_expert_used,
        weights_scale: cfg.expert_weights_scale,
        hash,
        expert_device: rd.expert_device.clone(),
        layer: layer as u32,
        device_pool: None,
    })
}

#[cfg(feature = "distributed")]
fn load_distributed_experts<R: Read + Seek>(
    rd: &Reader<R>,
    p: &str,
    cfg: &Config,
    hidden: usize,
    clamp: f64,
) -> anyhow::Result<DistributedMoe> {
    use anyhow::{ensure, Context};
    use crate::distributed::shard::{dtype_layout, ShardSpec, ShardedTensor};
    use crate::gguf_ext::RawTensorInfo;
    ensure!(rd.device.is_cpu() && rd.expert_device.is_cpu(), "cluster experts require CPU");
    let raw = rd.raw.as_ref().context("cluster requires raw GGUF header")?;
    let mmap = rd.mmap.as_ref().context("cluster requires mapped weights")?;
    let cluster = rd.cluster.as_ref().context("missing cluster session")?;
    ensure!(cfg.n_expert > 0 && cfg.n_expert_used > 0 && cfg.n_expert_used <= cfg.n_expert,
        "invalid routed expert counts");
    let get = |suffix: &str| raw.tensors.get(&format!("{p}.{suffix}.weight"))
        .with_context(|| format!("missing {p}.{suffix}.weight"));
    let gate = get("ffn_gate_exps")?;
    let up = get("ffn_up_exps")?;
    let down = get("ffn_down_exps")?;
    ensure!(gate.dims.len() == 3 && gate.dims[0] == cfg.n_expert
        && gate.dims[2] == hidden, "invalid gate expert dimensions");
    let intermediate = gate.dims[1];
    ensure!(up.dims == gate.dims, "gate/up expert dimensions differ");
    ensure!(down.dims == [cfg.n_expert, hidden, intermediate],
        "invalid down expert dimensions");
    let range = ShardSpec::new(cluster.rank(), cluster.world_size())?
        .input_range(intermediate, dtype_layout(down.dtype)?.block_elements)?;
    let mut full_bytes = 0usize;
    let mut strides = Vec::new();
    for info in [gate, up, down] {
        let layout = dtype_layout(info.dtype)?;
        ensure!(info.dims[2] > 0 && info.dims[2].is_multiple_of(layout.block_elements),
            "expert input dimension is not block-aligned");
        let row = (info.dims[2] / layout.block_elements).checked_mul(layout.block_bytes)
            .context("expert row bytes overflow")?;
        let stride = row.checked_mul(info.dims[1]).context("expert stride overflow")?;
        let total = stride.checked_mul(cfg.n_expert).context("expert tensor bytes overflow")?;
        let start = usize::try_from(raw.tensor_data_offset.checked_add(info.offset)
            .context("expert file offset overflow")?)?;
        ensure!(start.checked_add(total).context("expert file end overflow")? <= mmap.len(),
            "expert tensor exceeds mapping");
        full_bytes = full_bytes.checked_add(total).context("expert byte accounting overflow")?;
        strides.push((row, stride));
    }
    let matrix = |info: &RawTensorInfo, expert: usize, rows: std::ops::Range<usize>,
                  stride: (usize, usize)| -> anyhow::Result<RawTensorInfo> {
        let relative = expert.checked_mul(stride.1)
            .and_then(|v| rows.start.checked_mul(stride.0).and_then(|r| v.checked_add(r)))
            .context("expert slice offset overflow")?;
        Ok(RawTensorInfo {
            dtype: info.dtype,
            dims: vec![rows.len(), info.dims[2]],
            offset: info.offset.checked_add(u64::try_from(relative)?)
                .context("expert slice file offset overflow")?,
        })
    };
    let mut experts = Vec::with_capacity(cfg.n_expert);
    let mut local_bytes = 0usize;
    for e in 0..cfg.n_expert {
        let g = matrix(gate, e, range.clone(), strides[0])?;
        let u = matrix(up, e, range.clone(), strides[1])?;
        let d = matrix(down, e, 0..hidden, strides[2])?;
        let expert = DistributedExpert {
            gate: ShardedTensor::with_input_range(mmap.clone(), &g, raw.tensor_data_offset, 0..hidden)?,
            up: ShardedTensor::with_input_range(mmap.clone(), &u, raw.tensor_data_offset, 0..hidden)?,
            down: ShardedTensor::with_input_range(mmap.clone(), &d, raw.tensor_data_offset, range.clone())?,
        };
        for t in [&expert.gate, &expert.up, &expert.down] {
            for row in t.row_ranges() {
                local_bytes = local_bytes.checked_add(row.len()).context("shard byte accounting overflow")?;
            }
        }
        experts.push(expert);
    }
    Ok(DistributedMoe { experts, cluster: cluster.clone(), clamp, local_bytes, full_bytes })
}

/// Split a 3-D expert weight `[n_expert, out, in]` into `n_expert` quantized
/// 2-D `QMatMul`s by carving its raw quantized byte-buffer.
fn split_experts<R: Read + Seek>(
    rd: &mut Reader<R>,
    name: &str,
    n_expert: usize,
) -> Result<Vec<ExpertTensor>> {
    // IQ2_XXS expert tensors (how DeepSeek-V4-Flash GGUFs store gate/up):
    // candle cannot represent the dtype, so slice per-expert block ranges
    // straight out of the mapping and let `crate::iq2xxs` decode at matmul
    // time.  Only falls back to a full f32 decode when there is no mapping
    // (tests).
    let raw_info = rd.raw.as_ref().and_then(|r| r.tensors.get(name)).cloned();
    if let Some(info) = raw_info {
        if info.dtype == crate::iq2xxs::GGML_TYPE_IQ2_XXS {
            return split_iq2xxs_experts(rd, name, &info, n_expert);
        }
        if info.dtype == crate::mxfp4::GGML_TYPE_MXFP4 {
            return split_mxfp4_experts(rd, name, &info, n_expert);
        }
    }
    // Read the shape and dtype straight from the header instead of
    // materialising the tensor first: on an accelerator `Reader::qtensor`
    // would upload the whole `[n_expert, out, in]` stack to VRAM (28 GiB for
    // V4-Flash's Q2_K `ffn_down_exps`) only to slice it back into per-expert
    // pieces.
    let (out, inn, dtype) = match rd.ct.tensor_infos.get(name) {
        Some(info) => {
            let dims = info.shape.dims().to_vec();
            if dims.len() != 3 || dims[0] != n_expert {
                candle_core::bail!(
                    "deepseek4: expected expert tensor `{name}` shaped [n_expert, out, in], got {dims:?}"
                );
            }
            (dims[1], dims[2], info.ggml_dtype)
        }
        None => candle_core::bail!("deepseek4: tensor `{name}` is missing from the GGUF header"),
    };

    // The routed experts are built on `rd.expert_device` — the CPU for every
    // mapped model, including a GPU build (see `Reader::expert_device`), so
    // this borrow path is the production one on an accelerator too: the dense
    // set is what gets uploaded, not the 72 GiB expert pool (the VRAM expert
    // cache uploads individual experts from these borrowed tensors).
    if let Some(mmap) = rd.mmap.clone().filter(|_| rd.expert_device.is_cpu()) {
        if let Some(info) = rd.ct.tensor_infos.get(name) {
            let block_size = dtype.block_size();
            let per_elems = out * inn;
            if block_size > 0 && per_elems.is_multiple_of(block_size) {
                let per_bytes = per_elems / block_size * dtype.type_size();
                let base = rd.ct.tensor_data_offset.saturating_add(info.offset) as usize;
                let mut borrowed = Vec::with_capacity(n_expert);
                for e in 0..n_expert {
                    match crate::mmap_tensor::borrowed_range(
                        &mmap,
                        dtype,
                        base + e * per_bytes,
                        (out, inn).into(),
                    )? {
                        Some(t) => borrowed.push(ExpertTensor {
                            qmatmul: QMatMul::from_qtensor(t)?,
                            prefetch: crate::mmap_tensor::prefetch_handle(
                                &mmap,
                                dtype,
                                base + e * per_bytes,
                                per_elems / block_size,
                            ),
                        }),
                        None => {
                            borrowed.clear();
                            break;
                        }
                    }
                }
                if borrowed.len() == n_expert {
                    return Ok(borrowed);
                }
            }
        }
    }

    let qt = rd.qtensor(name)?;
    // Split the whole stacked tensor into per-expert slices for the storage of
    // whatever `qt` actually holds.  The streamed (no-mmap) path decodes the
    // weight to f32, so the storage dtype here is the *decoded* one, not the
    // GGUF header dtype — slicing by the decoded dtype's block size is the only
    // way the per-expert `QStorage::from_data` is block-aligned.  (The
    // production mmap path takes the `borrowed_range` branch above and never
    // reaches this fallback for a well-formed file.)
    let storage_dtype = qt.dtype();
    let bytes = qt.data()?;
    let block = storage_dtype.block_size();
    let per_elems = out * inn;
    let per_bytes = if block > 0 && per_elems.is_multiple_of(block) {
        per_elems / block * storage_dtype.type_size()
    } else {
        // Unquantized (f32/f16/bf16) storage, or a block that doesn't divide the
        // per-expert element count: each expert is `per_elems` contiguous
        // `type_size`-byte elements.
        per_elems
            .checked_mul(storage_dtype.type_size())
            .ok_or_else(|| {
                candle_core::Error::Msg(format!("deepseek4: expert tensor `{name}` size overflow"))
            })?
    };
    let total_needed = per_bytes.checked_mul(n_expert).ok_or_else(|| {
        candle_core::Error::Msg(format!("deepseek4: expert tensor `{name}` size overflow"))
    })?;
    if bytes.len() < total_needed {
        candle_core::bail!(
            "deepseek4: expert tensor `{name}` byte length {} is smaller than the {} experts × {per_bytes} bytes needed",
            bytes.len(),
            n_expert
        );
    }
    let mut experts = Vec::with_capacity(n_expert);
    for e in 0..n_expert {
        let slice = &bytes[e * per_bytes..(e + 1) * per_bytes];
        // Rebuild with the *storage* dtype, not the header dtype, so a decoded
        // (f32) fallback storage is sliced consistently.
        let storage = QStorage::from_data(Cow::Borrowed(slice), &rd.expert_device, storage_dtype)?;
        let qt = QTensor::new(storage, (out, inn))?;
        experts.push(ExpertTensor {
            qmatmul: QMatMul::from_qtensor(qt)?,
            prefetch: None,
        });
    }
    Ok(experts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn md() -> HashMap<String, gguf_file::Value> {
        let mut m = HashMap::new();
        let u32v = |v: u32| gguf_file::Value::U32(v);
        let f32v = |v: f32| gguf_file::Value::F32(v);
        m.insert("deepseek4.block_count".into(), u32v(43));
        m.insert("deepseek4.attention.head_count".into(), u32v(64));
        m.insert("deepseek4.embedding_length".into(), u32v(4096));
        m.insert(
            "deepseek4.attention.layer_norm_rms_epsilon".into(),
            f32v(1e-6),
        );
        m.insert("deepseek4.attention.q_lora_rank".into(), u32v(1024));
        m.insert("deepseek4.attention.key_length".into(), u32v(512));
        m.insert("deepseek4.expert_count".into(), u32v(256));
        m.insert("deepseek4.expert_used_count".into(), u32v(6));
        m
    }

    /// The real DeepSeek-V4-Flash GGUFs store the rope dim at
    /// `deepseek4.rope.dimension_count`; older files used the
    /// `attention.`-prefixed key.  Both must parse to the same config.
    #[test]
    fn rope_dim_reads_both_metadata_keys() {
        let mut m = md();
        m.insert(
            "deepseek4.rope.dimension_count".into(),
            gguf_file::Value::U32(64),
        );
        let cfg = Config::from_metadata(&m).unwrap();
        assert_eq!(cfg.rope_head_dim, 64);
        assert_eq!(cfg.nope_head_dim, 512 - 64);

        let mut m = md();
        m.insert(
            "deepseek4.attention.rope.dimension_count".into(),
            gguf_file::Value::U32(64),
        );
        let cfg = Config::from_metadata(&m).unwrap();
        assert_eq!(cfg.rope_head_dim, 64);
        assert_eq!(cfg.nope_head_dim, 512 - 64);
    }

    /// Missing rope metadata must be a load-time error, not a silent collapse
    /// to a degenerate 1-wide rotary (which would produce an unreadable
    /// `[b, h, t, 1]` rope slice mid-generation).
    #[test]
    fn missing_rope_dim_is_a_load_error() {
        let err = match Config::from_metadata(&md()) {
            Err(e) => e,
            Ok(_) => panic!("from_metadata must reject a file with no rotary-dimension metadata"),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("rope"),
            "error should name the rotary-dimension metadata, got: {msg}"
        );
    }
}
