//! Pure-Rust quantized loader for the Qwen family of GGUF architectures.
//!
//! One loader covers every Qwen decoder llama.cpp can convert:
//!
//! | `general.architecture` | Models | Attention | FFN |
//! |---|---|---|---|
//! | `qwen`       | Qwen (1)                          | fused QKV + bias          | dense |
//! | `qwen2moe`   | Qwen1.5-MoE, Qwen2-57B-A14B       | QKV bias                  | MoE + gated shared expert |
//! | `qwen2vl`    | Qwen2-VL, Qwen2.5-VL (text)       | QKV bias, M-RoPE          | dense |
//! | `qwen3moe`   | Qwen3-30B-A3B, Qwen3-Coder        | Q/K norm                  | MoE |
//! | `qwen3vl`    | Qwen3-VL (text)                   | Q/K norm, IM-RoPE         | dense |
//! | `qwen3vlmoe` | Qwen3-VL-MoE (text)               | Q/K norm, IM-RoPE         | MoE |
//! | `qwen3next`  | Qwen3-Next-80B-A3B                | gated / Gated DeltaNet    | MoE + gated shared expert |
//! | `qwen35`     | Qwen3.5 dense                     | gated / Gated DeltaNet, IM-RoPE | dense |
//! | `qwen35moe`  | Qwen3.5-MoE                       | gated / Gated DeltaNet, IM-RoPE | MoE + gated shared expert |
//!
//! (`qwen2` and `qwen3` dense run through candle's stock loaders.)  The
//! pieces that vary are all optional parts of one decoder layer:
//!
//! * **Attention.**  GQA over a fused (`attn_qkv`) or split Q/K/V projection
//!   with optional biases, optional per-head Q/K RMSNorm, and — for the
//!   Qwen3-Next generation — a sigmoid output gate carried in a doubled Q
//!   projection.  RoPE is NEOX-paired over the leading `rope.dimension_count`
//!   dims; the VL models' multi-section RoPE reduces, for text positions
//!   `[p, p, p, 0]`, to that same rotation with the "extra"-section
//!   frequencies frozen (see [`crate::attention::mrope_text_mask`]).
//!
//! * **Gated DeltaNet** (Qwen3-Next / Qwen3.5 linear-attention layers).  A
//!   causal depthwise conv over the fused Q/K/V stream, L2-normalised Q/K,
//!   and a per-head `d_k × d_v` state updated by the gated delta rule; the
//!   output is RMS-normalised, gated by `silu(z)` and projected back.  The
//!   state is recurrent, so these models cannot rewind their cache to an
//!   arbitrary prefix (edited-context reuse and speculative rollback fall
//!   back to a fresh prefill).
//!
//! * **FFN.**  Dense SwiGLU, or a fine-grained softmax-routed MoE with
//!   optional top-k normalisation / scaling and an optional shared expert
//!   whose output is scaled by `sigmoid(x · ffn_gate_inp_shexp)`.  Experts
//!   stay **quantized**: the 3-D expert tensor is sliced into per-expert
//!   [`QMatMul`]s straight from its quantized bytes (borrowed in place from
//!   the mmap when possible), so the model keeps its on-disk footprint
//!   instead of exploding to f32 in RAM.
//!
//! The math follows llama.cpp's `src/models/qwen*.cpp` graphs on the same
//! GGUF tensors.  Activations run in f32 for CPU accuracy, mirroring the
//! other Joshua quantized loaders (`glm4`, `deepseek2`).

use std::io::{Read, Seek};
use std::sync::Arc;

use candle_core::quantized::{gguf_file, QMatMul, QTensor};
use crate::paged_weights::{PagedWeight, WeightCache};
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::ops::{sigmoid, silu, softmax_last_dim};
use candle_transformers::quantized_nn::RmsNorm;

use crate::attention::{KvCache, Rope, RopeStyle};
use crate::gguf_meta::Meta;
use crate::token_embedding::TokenEmbedding;
use crate::zero_copy_metal::{ZcContext, ZcWeight};

// ─── Opt-in phase profiler (JOSHUA_PROFILE=1) ────────────────────────────────
//
// Diagnostic for the CPU decode path: accumulates nanoseconds per phase
// across forward passes and prints per-step averages every 16 steps on the
// calling thread.  Off by default; zero cost when disabled.
mod prof {
    use std::cell::Cell;
    use std::sync::OnceLock;
    use std::time::Instant;

    type Slot = &'static std::thread::LocalKey<Cell<u128>>;

    pub fn enabled() -> bool {
        static E: OnceLock<bool> = OnceLock::new();
        *E.get_or_init(|| {
            std::env::var("JOSHUA_PROFILE").map(|v| !v.is_empty()).unwrap_or(false)
        })
    }

    thread_local! {
        pub static ATT: Cell<u128> = const { Cell::new(0) };
        pub static MOE: Cell<u128> = const { Cell::new(0) };
        pub static EXPERTS: Cell<u128> = const { Cell::new(0) };
        pub static HEAD: Cell<u128> = const { Cell::new(0) };
        static STEPS: Cell<u64> = const { Cell::new(0) };
    }

    pub struct Phase(Option<(Instant, Slot)>);

    impl Phase {
        pub fn start(slot: Slot) -> Self {
            Self(if enabled() { Some((Instant::now(), slot)) } else { None })
        }
    }

    impl Drop for Phase {
        fn drop(&mut self) {
            if let Some((t0, slot)) = self.0.take() {
                let ns = t0.elapsed().as_nanos();
                slot.with(|c| c.set(c.get() + ns));
            }
        }
    }

    /// Report averages over the steps accumulated so far and reset.
    pub fn report() {
        if !enabled() {
            return;
        }
        STEPS.with(|s| {
            let n = s.get() + 1;
            s.set(n);
            if n % 16 == 0 {
                let a = ATT.with(|c| c.get());
                let m = MOE.with(|c| c.get());
                let e = EXPERTS.with(|c| c.get());
                let h = HEAD.with(|c| c.get());
                eprintln!(
                    "[profile] avg ms/step over last 16: attention {:.1}, moe {:.1} \
                     (expert matmuls {:.1}), lm_head {:.1}",
                    a as f64 / 16e6,
                    m as f64 / 16e6,
                    e as f64 / 16e6,
                    h as f64 / 16e6,
                );
                ATT.with(|c| c.set(0));
                MOE.with(|c| c.set(0));
                EXPERTS.with(|c| c.set(0));
                HEAD.with(|c| c.set(0));
            }
        });
    }
}

// ─── Configuration ───────────────────────────────────────────────────────────

/// Every `general.architecture` this loader serves.
pub const ARCHES: &[&str] = &[
    "qwen",
    "qwen2moe",
    "qwen2vl",
    "qwen3moe",
    "qwen3vl",
    "qwen3vlmoe",
    "qwen3next",
    "qwen35",
    "qwen35moe",
];

/// How a Gated DeltaNet layer's value heads share the (fewer) key heads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeyHeadShare {
    /// Value head `h` reads key head `h / (n_v / n_k)` — Qwen3-Next, whose
    /// GGUF keeps HF's grouped head order (llama.cpp repeat-interleaves).
    Grouped,
    /// Value head `h` reads key head `h % n_k` — Qwen3.5, whose converter
    /// reorders V heads into ggml's tiled broadcast order.
    Tiled,
}

/// Gated DeltaNet (`ssm.*`) dimensions.
#[derive(Debug, Clone)]
struct SsmConfig {
    d_conv: usize,
    n_k_heads: usize,
    n_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    share: KeyHeadShare,
}

impl SsmConfig {
    fn key_dim(&self) -> usize {
        self.n_k_heads * self.head_k_dim
    }
    fn value_dim(&self) -> usize {
        self.n_v_heads * self.head_v_dim
    }
    fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
    /// The key head value head `h` reads.
    fn key_head(&self, h: usize) -> usize {
        match self.share {
            KeyHeadShare::Grouped => h / (self.n_v_heads / self.n_k_heads),
            KeyHeadShare::Tiled => h % self.n_k_heads,
        }
    }
}

struct Config {
    arch: &'static str,
    n_layer: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rms_eps: f64,
    // RoPE.
    n_rot: usize,
    rope_theta: f32,
    /// Linear RoPE scaling (`1 / rope.scaling.factor` for `"linear"`), else 1.
    rope_freq_scale: f32,
    /// Multi-section RoPE `(sections, interleaved)` for the VL / Qwen3.5
    /// models (see [`crate::attention::mrope_text_mask`]).
    mrope: Option<([usize; 4], bool)>,
    context_length: usize,
    attn_scale: f64,
    /// Doubled Q projection carrying a sigmoid output gate (Qwen3-Next+).
    gated_attn: bool,
    /// Per layer: Gated DeltaNet (true) or attention (false).
    recurrent: Vec<bool>,
    ssm: Option<SsmConfig>,
    // MoE (`n_expert == 0` → dense FFN).
    n_expert: usize,
    n_expert_used: usize,
    expert_weights_norm: bool,
    expert_weights_scale: f64,
}

impl Config {
    fn from_metadata(md: &std::collections::HashMap<String, gguf_file::Value>, arch: &str) -> Result<Self> {
        let Some(&arch) = ARCHES.iter().find(|&&a| a == arch) else {
            candle_core::bail!("qwen loader: unsupported architecture `{arch}`");
        };
        let m = Meta::new(md, arch);
        let n_head = m.u32("attention.head_count")? as usize;
        let n_kv_head = m.u32_or("attention.head_count_kv", n_head as u32) as usize;
        let n_layer = m.u32("block_count")? as usize;
        let n_embd = m.u32("embedding_length")? as usize;
        let context_length = m.u32("context_length")? as usize;
        let rms_eps = m.f32("attention.layer_norm_rms_epsilon")? as f64;
        if n_head == 0 {
            candle_core::bail!("{arch}: attention.head_count must be positive");
        }

        // Qwen3-Coder decouples head_dim from the embedding width (2048/32 = 64
        // would be wrong; the real value is 128), so `attention.key_length`
        // must win when present.  Fall back to n_embd/n_head only for GGUFs
        // that omit it.
        let head_dim = m.u32_or("attention.key_length", (n_embd / n_head) as u32) as usize;
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            candle_core::bail!("{arch}: invalid head_dim {head_dim} (must be a positive even number)");
        }
        if n_kv_head == 0 || !n_head.is_multiple_of(n_kv_head) {
            candle_core::bail!(
                "{arch}: head_count {n_head} must be a multiple of head_count_kv {n_kv_head}"
            );
        }
        // Partial rotary (Qwen3-Next / Qwen3.5 rotate a quarter of each head).
        let n_rot = m.u32_or("rope.dimension_count", head_dim as u32) as usize;
        if n_rot == 0 || !n_rot.is_multiple_of(2) || n_rot > head_dim {
            candle_core::bail!("{arch}: invalid rope.dimension_count {n_rot} for head_dim {head_dim}");
        }

        let rope_theta = m.f32_or("rope.freq_base", 10_000.0);
        let rope_freq_scale = match m.string("rope.scaling.type").as_deref() {
            Some("linear") => {
                let factor = m.f32_or("rope.scaling.factor", 1.0);
                if factor > 0.0 { 1.0 / factor } else { 1.0 }
            }
            _ => 1.0,
        };
        let interleaved_mrope = match arch {
            "qwen2vl" => Some(false),
            "qwen3vl" | "qwen3vlmoe" | "qwen35" | "qwen35moe" => Some(true),
            _ => None,
        };
        let mrope = interleaved_mrope.map(|interleaved| {
            let s = m.array_u32("rope.dimension_sections", 4);
            let s = [s[0], s[1], s[2], s[3]];
            // Qwen3.5's converter default when a checkpoint omits the field.
            let s = if s == [0; 4] && arch.starts_with("qwen35") { [11, 11, 10, 0] } else { s };
            (s, interleaved)
        });
        let attn_scale = match m.f32_or("attention.scale", 0.0) {
            s if s != 0.0 => s as f64,
            _ => 1.0 / (head_dim as f64).sqrt(),
        };

        let hybrid = matches!(arch, "qwen3next" | "qwen35" | "qwen35moe");
        let (recurrent, ssm) = if hybrid {
            let recurrent = match m.array_bool("attention.recurrent_layers", n_layer) {
                Some(r) => r,
                None => {
                    let interval = m.u32_or("full_attention_interval", 4).max(1) as usize;
                    (0..n_layer).map(|i| (i + 1) % interval != 0).collect()
                }
            };
            let n_v_heads = m.u32("ssm.time_step_rank")? as usize;
            let n_k_heads = m.u32("ssm.group_count")? as usize;
            let d_inner = m.u32("ssm.inner_size")? as usize;
            if n_k_heads == 0 || n_v_heads == 0 || !n_v_heads.is_multiple_of(n_k_heads) {
                candle_core::bail!(
                    "{arch}: ssm.time_step_rank {n_v_heads} must be a positive multiple of ssm.group_count {n_k_heads}"
                );
            }
            let ssm = SsmConfig {
                d_conv: m.u32("ssm.conv_kernel")? as usize,
                n_k_heads,
                n_v_heads,
                head_k_dim: m.u32("ssm.state_size")? as usize,
                head_v_dim: d_inner / n_v_heads,
                share: if arch == "qwen3next" { KeyHeadShare::Grouped } else { KeyHeadShare::Tiled },
            };
            if ssm.d_conv == 0 || ssm.head_k_dim != ssm.head_v_dim {
                candle_core::bail!(
                    "{arch}: Gated DeltaNet needs a positive conv kernel and equal key/value head dims \
                     (got {} / {})",
                    ssm.head_k_dim,
                    ssm.head_v_dim
                );
            }
            (recurrent, Some(ssm))
        } else {
            (vec![false; n_layer], None)
        };

        let n_expert = m.u32_or("expert_count", 0) as usize;
        let n_expert_used = m.u32_or("expert_used_count", 0) as usize;
        if n_expert > 0 && (n_expert_used == 0 || n_expert_used > n_expert) {
            candle_core::bail!(
                "{arch}: expert_used_count {n_expert_used} must be in 1..=expert_count {n_expert}"
            );
        }
        // llama.cpp normalises the top-k routing weights for every Qwen MoE
        // except Qwen1.5/Qwen2-MoE (`norm_topk_prob = false`).
        let expert_weights_norm = m.bool_or("attention.norm_topk_prob", arch != "qwen2moe");
        let expert_weights_scale = m.f32_or("expert_weights_scale", 0.0) as f64;

        Ok(Self {
            arch,
            n_layer,
            n_head,
            n_kv_head,
            head_dim,
            rms_eps,
            n_rot,
            rope_theta,
            rope_freq_scale,
            mrope,
            context_length,
            attn_scale,
            gated_attn: hybrid,
            recurrent,
            ssm,
            n_expert,
            n_expert_used,
            expert_weights_norm,
            expert_weights_scale,
        })
    }

    /// NEOX RoPE over the leading `n_rot` dims, with the M-RoPE text mask
    /// applied for the multi-section models.
    fn rope(&self, dev: &Device) -> Result<Rope> {
        let mut inv_freq: Vec<f32> = crate::attention::inv_freq(self.n_rot, self.rope_theta)
            .into_iter()
            .map(|f| f * self.rope_freq_scale)
            .collect();
        if let Some((sections, interleaved)) = self.mrope {
            crate::attention::mrope_text_mask(&mut inv_freq, sections, interleaved);
        }
        Rope::from_inv_freq(inv_freq, self.context_length, 1.0, RopeStyle::Neox, dev)
    }
}

// ─── Quantized GGUF reader (with optional mmap borrowing) ───────────────────

/// Small GGUF reader over the (possibly memory-mapped) file.
struct Reader<R: Read + Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
    /// Device the routed-expert tensors are built on.  Same as `device` for
    /// a CPU model or when the experts are uploaded; the CPU when the model
    /// runs on an accelerator that cannot hold the expert pool (see
    /// [`crate::placement::ExpertPlacement`]) — each expert is then borrowed
    /// from the mapping and [`Moe::dispatch`] hops the activations across.
    expert_device: Device,
    /// When present, tensors are borrowed in place from this mapping instead
    /// of being copied onto the heap (see [`crate::mmap_tensor`]).
    mmap: Option<Arc<memmap2::Mmap>>,
    /// Bytes of the bounded VRAM expert cache (#62) to build for each MoE
    /// block (`None` / `Some(0)` disables).  `load_moe` reads this to construct
    /// the per-layer `DeviceResidency`.
    device_expert_cache_bytes: Option<u64>,
    /// When present, quantized weights are bound straight into the mapping
    /// via a no-copy Metal buffer (see [`crate::zero_copy_metal`]) instead of
    /// being uploaded.  `None` on CPU, without a mapping, or when the
    /// no-copy buffer could not be created (the loader then copies).
    zc: Option<Arc<ZcContext>>,
    /// Opt-in bounded GPU weight cache (see `JOSHUA_GPU_WEIGHT_CACHE`).  When
    /// present, quantized experts are kept compressed in the mmap and uploaded
    /// to the device on demand instead of being copied up front.
    gpu_cache: Option<Arc<WeightCache>>,
    /// `general.architecture`, for error context.
    arch: &'static str,
}

impl<R: Read + Seek> Reader<R> {
    fn qtensor(&mut self, name: &str) -> Result<QTensor> {
        if let Some(mmap) = &self.mmap {
            return crate::mmap_tensor::qtensor_from_mmap(
                &self.ct,
                mmap,
                &mut self.reader,
                name,
                &self.device,
            );
        }
        self.ct.tensor(&mut self.reader, name, &self.device)
    }
    fn qmatmul(&mut self, name: &str) -> Result<Weight> {
        if let Some(zc) = &self.zc {
            if let Some(w) = zc.weight(&self.ct, name)? {
                return Ok(Weight::Zc(Arc::new(w)));
            }
        }
        Ok(Weight::Candle(QMatMul::from_qtensor(self.qtensor(name)?)?))
    }
    fn qmatmul_opt(&mut self, name: &str) -> Option<Weight> {
        if self.has(name) {
            self.qmatmul(name).ok()
        } else {
            None
        }
    }
    fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        RmsNorm::from_qtensor(self.qtensor(name)?, eps)
    }
    fn rms_norm_opt(&mut self, name: &str, eps: f64) -> Result<Option<RmsNorm>> {
        self.has(name).then(|| self.rms_norm(name, eps)).transpose()
    }
    fn f32_tensor(&mut self, name: &str) -> Result<Tensor> {
        self.qtensor(name)?.dequantize(&self.device)?.to_dtype(DType::F32)
    }
    fn f32_opt(&mut self, name: &str) -> Result<Option<Tensor>> {
        self.has(name).then(|| self.f32_tensor(name)).transpose()
    }
    fn has(&self, name: &str) -> bool {
        self.ct.tensor_infos.contains_key(name)
    }
}

// ─── Weight carrier (candle upload vs zero-copy mmap binding) ───────────────

/// A quantized linear weight in one of two homes:
///
/// * [`Weight::Candle`] — candle's own `QMatMul` (copied onto the device).
///   Used on CPU, without a mapping, and for the tensors the zero-copy path
///   does not serve.
/// * [`Weight::Zc`] — a [`ZcWeight`] bound at its file offset inside a
///   no-copy Metal buffer.  The GPU reads the mapped pages directly; nothing
///   is copied or uploaded.
#[derive(Clone)]
enum Weight {
    Candle(QMatMul),
    Zc(Arc<ZcWeight>),
    /// Quantized weights kept compressed in mmap and uploaded to the device on
    /// demand through a bounded [`WeightCache`] (opt-in; see
    /// `JOSHUA_GPU_WEIGHT_CACHE`).  Each expert owns one [`PagedWeight`].
    Paged(Arc<PagedWeight>),
}

impl Weight {
    /// Re-upload this weight's quantized blocks onto `device`.
    ///
    /// Only `Weight::Candle` over a `QMatMul::QTensor` is re-uploadable
    /// (copies its raw blocks via `QStorage::from_data`, which is
    /// backend-generic — CPU, Vulkan, Metal, OpenCL, CUDA).  Other variants
    /// (plain `Tensor`, zero-copy Metal, paged) return `None`: they have no
    /// standalone quantized form to re-upload here.  This is what lets the
    /// #62 device-expert cache hold hot experts on a *non-CPU* backend.
    fn to_uploadable(&self, device: &Device) -> Option<Weight> {
        match self {
            Weight::Candle(q) => crate::moe::upload_qmatmul(q, device).map(Weight::Candle),
            _ => None,
        }
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Weight::Candle(q) => {
                // Joshua's SIMD path first: fused dequant+dot kernels
                // (Q8_0/Q2_K/Q4_K) or dequant-row + NEON/AVX2 dot (other
                // k-quants), parallelised across rows — instead of candle's
                // scalar single-threaded kernel.  Anything without a fast
                // path here falls through to candle unchanged.
                if let QMatMul::QTensor(qt) = q {
                    if let Some(res) = crate::quant_matmul::try_fast_cpu_qmatmul(qt, xs) {
                        return res;
                    }
                }
                q.forward(xs)
            }
            Weight::Zc(z) => z.forward(xs),
            Weight::Paged(p) => p.forward(xs),
        }
    }
}

// ─── Linear helpers ─────────────────────────────────────────────────────────

/// A quantized projection plus an optional f32 bias.
struct Linear {
    w: Weight,
    b: Option<Tensor>,
}

impl Linear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let y = self.w.forward(xs)?;
        match &self.b {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}

/// The Q/K/V input projection: one fused `attn_qkv` matmul (Qwen1, the
/// Qwen3-Next generation) or three separate ones.  Outputs are
/// `[b, seq, dim]` each.
enum Qkv {
    Fused { qkv: Linear, q_dim: usize, k_dim: usize, v_dim: usize },
    Split { q: Linear, k: Linear, v: Linear },
}

impl Qkv {
    fn load<R: Read + Seek>(
        rd: &mut Reader<R>,
        p: &str,
        q_dim: usize,
        k_dim: usize,
        v_dim: usize,
    ) -> Result<Self> {
        if rd.has(&format!("{p}.attn_qkv.weight")) {
            let w = rd.qmatmul(&format!("{p}.attn_qkv.weight"))?;
            // A fused weight may still ship separate Q/K/V biases (llama.cpp
            // `create_tensor_qkv`); concatenate them into one.
            let b = match rd.f32_opt(&format!("{p}.attn_qkv.bias"))? {
                Some(b) => Some(b),
                None => {
                    let parts = [
                        rd.f32_opt(&format!("{p}.attn_q.bias"))?,
                        rd.f32_opt(&format!("{p}.attn_k.bias"))?,
                        rd.f32_opt(&format!("{p}.attn_v.bias"))?,
                    ];
                    match parts {
                        [Some(q), Some(k), Some(v)] => Some(Tensor::cat(&[q, k, v], 0)?),
                        _ => None,
                    }
                }
            };
            return Ok(Self::Fused { qkv: Linear { w, b }, q_dim, k_dim, v_dim });
        }
        let mut lin = |name: &str| -> Result<Linear> {
            Ok(Linear {
                w: rd.qmatmul(&format!("{p}.{name}.weight"))?,
                b: rd.f32_opt(&format!("{p}.{name}.bias"))?,
            })
        };
        Ok(Self::Split { q: lin("attn_q")?, k: lin("attn_k")?, v: lin("attn_v")? })
    }

    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        match self {
            Self::Fused { qkv, q_dim, k_dim, v_dim } => {
                let y = qkv.forward(xs)?;
                Ok((
                    y.narrow(D::Minus1, 0, *q_dim)?,
                    y.narrow(D::Minus1, *q_dim, *k_dim)?,
                    y.narrow(D::Minus1, q_dim + k_dim, *v_dim)?,
                ))
            }
            Self::Split { q, k, v } => Ok((q.forward(xs)?, k.forward(xs)?, v.forward(xs)?)),
        }
    }
}

/// Apply a `[head_dim]` RMSNorm to every head of a `[b, heads, seq, d]`
/// tensor independently.
fn per_head_norm(norm: &RmsNorm, x: &Tensor) -> Result<Tensor> {
    let (b, h, s, d) = x.dims4()?;
    norm.forward(&x.flatten(0, 2)?)?.reshape((b, h, s, d))
}

// ─── Attention (GQA; optional biases, Q/K norm, output gate) ────────────────

struct Attention {
    qkv: Qkv,
    o: Linear,
    q_norm: Option<RmsNorm>, // [head_dim] per-head query norm
    k_norm: Option<RmsNorm>, // [head_dim] per-head key norm
    /// Q projection carries `[q ‖ gate]` per head; the attention output is
    /// scaled by `sigmoid(gate)` before `o` (Qwen3-Next / Qwen3.5).
    gated: bool,
    rope: Arc<Rope>,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    fn load<R: Read + Seek>(rd: &mut Reader<R>, p: &str, cfg: &Config, rope: Arc<Rope>) -> Result<Self> {
        let hd = cfg.head_dim;
        let q_dim = cfg.n_head * hd * if cfg.gated_attn { 2 } else { 1 };
        let kv_dim = cfg.n_kv_head * hd;
        Ok(Self {
            qkv: Qkv::load(rd, p, q_dim, kv_dim, kv_dim)?,
            o: Linear {
                w: rd.qmatmul(&format!("{p}.attn_output.weight"))?,
                b: rd.f32_opt(&format!("{p}.attn_output.bias"))?,
            },
            q_norm: rd.rms_norm_opt(&format!("{p}.attn_q_norm.weight"), cfg.rms_eps)?,
            k_norm: rd.rms_norm_opt(&format!("{p}.attn_k_norm.weight"), cfg.rms_eps)?,
            gated: cfg.gated_attn,
            rope,
            n_head: cfg.n_head,
            n_kv_head: cfg.n_kv_head,
            head_dim: hd,
            scale: cfg.attn_scale,
        })
    }

    fn forward(
        &self,
        kv_cache: &mut KvCache,
        xs: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let _p = prof::Phase::start(&prof::ATT);
        let (b, seq_len, _) = xs.dims3()?;
        let hd = self.head_dim;
        let (q, k, v) = self.qkv.forward(xs)?;

        let (q, gate) = if self.gated {
            let q = q.reshape((b, seq_len, self.n_head, 2 * hd))?;
            let gate = q.narrow(3, hd, hd)?.reshape((b, seq_len, self.n_head * hd))?;
            (q.narrow(3, 0, hd)?, Some(gate))
        } else {
            (q.reshape((b, seq_len, self.n_head, hd))?, None)
        };
        let heads = |x: Tensor| -> Result<Tensor> { x.transpose(1, 2)?.contiguous() };
        let mut q = heads(q)?;
        let mut k = heads(k.reshape((b, seq_len, self.n_kv_head, hd))?)?;
        let v = heads(v.reshape((b, seq_len, self.n_kv_head, hd))?)?;
        if let Some(n) = &self.q_norm {
            q = per_head_norm(n, &q)?;
        }
        if let Some(n) = &self.k_norm {
            k = per_head_norm(n, &k)?;
        }
        let q = self.rope.apply_leading(&q, offset)?;
        let k = self.rope.apply_leading(&k, offset)?;

        let mut ctx = crate::attention::cached_attention_heads(kv_cache, &q, &k, &v, mask, self.scale)?;
        if let Some(gate) = gate {
            ctx = (ctx * sigmoid(&gate)?)?;
        }
        self.o.forward(&ctx)
    }
}

// ─── Gated DeltaNet (Qwen3-Next / Qwen3.5 linear attention) ─────────────────

/// The Q/K/V + `z` input projection of a DeltaNet layer.
enum DeltaInput {
    /// `attn_qkv` (`[q ‖ k ‖ v]`) plus `attn_gate` (`z`).
    Split { qkv: Weight, z: Weight },
    /// Legacy `ssm_in`: per key head `[q, k, v, z]`, regrouped at run time.
    Legacy(Weight),
}

/// The per-token β (write strength) and α (decay) projections.
enum BetaAlpha {
    /// Qwen3-Next `ssm_ba`: per key head `[β × r, α × r]`, `r = n_v / n_k`.
    Fused(Weight),
    /// Qwen3.5 `ssm_beta` / `ssm_alpha`.
    Split { beta: Weight, alpha: Weight },
}

/// One layer's recurrent state: the causal-conv tail and the delta-rule
/// matrices.
#[derive(Clone)]
struct DeltaState {
    /// The last `d_conv - 1` conv inputs, `[conv_dim, d_conv - 1]`.
    conv: Tensor,
    /// Per value head `h`, the `head_k_dim × head_v_dim` state `S[h][i][j]`
    /// (row-major, host f32).
    ssm: Vec<f32>,
}

struct GatedDeltaNet {
    input: DeltaInput,
    ba: BetaAlpha,
    /// Depthwise conv kernel, `[conv_dim, d_conv]`.
    conv_w: Tensor,
    dt_bias: Tensor, // [n_v]
    /// `-exp(A_log)` per value head (the converter pre-applies it).
    a: Tensor, // [n_v]
    norm: RmsNorm, // [head_v_dim]
    out: Weight,
    ssm: SsmConfig,
    eps: f64,
}

impl GatedDeltaNet {
    fn load<R: Read + Seek>(rd: &mut Reader<R>, p: &str, cfg: &Config) -> Result<Self> {
        let ssm = cfg.ssm.clone().expect("recurrent layer without ssm config");
        let input = if rd.has(&format!("{p}.attn_qkv.weight")) {
            DeltaInput::Split {
                qkv: rd.qmatmul(&format!("{p}.attn_qkv.weight"))?,
                z: rd.qmatmul(&format!("{p}.attn_gate.weight"))?,
            }
        } else {
            DeltaInput::Legacy(rd.qmatmul(&format!("{p}.ssm_in.weight"))?)
        };
        let ba = if rd.has(&format!("{p}.ssm_ba.weight")) {
            BetaAlpha::Fused(rd.qmatmul(&format!("{p}.ssm_ba.weight"))?)
        } else {
            BetaAlpha::Split {
                beta: rd.qmatmul(&format!("{p}.ssm_beta.weight"))?,
                alpha: rd.qmatmul(&format!("{p}.ssm_alpha.weight"))?,
            }
        };
        let conv_w = rd
            .f32_tensor(&format!("{p}.ssm_conv1d.weight"))?
            .reshape((ssm.conv_dim(), ssm.d_conv))?;
        Ok(Self {
            input,
            ba,
            conv_w,
            dt_bias: rd.f32_tensor(&format!("{p}.ssm_dt.bias"))?,
            a: rd.f32_tensor(&format!("{p}.ssm_a"))?,
            norm: rd.rms_norm(&format!("{p}.ssm_norm.weight"), cfg.rms_eps)?,
            out: rd.qmatmul(&format!("{p}.ssm_out.weight"))?,
            ssm,
            eps: cfg.rms_eps,
        })
    }

    /// `(qkv [t, conv_dim], z [t, value_dim])` for `xs` `[t, hidden]`.
    fn project(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let c = &self.ssm;
        match &self.input {
            DeltaInput::Split { qkv, z } => Ok((qkv.forward(xs)?, z.forward(xs)?)),
            DeltaInput::Legacy(w) => {
                let t = xs.dim(0)?;
                let r = c.n_v_heads / c.n_k_heads;
                let (hk, vr) = (c.head_k_dim, r * c.head_v_dim);
                let y = w.forward(xs)?.reshape((t, c.n_k_heads, 2 * hk + 2 * vr))?;
                let flat = |off: usize, n: usize| -> Result<Tensor> {
                    y.narrow(2, off, n)?.contiguous()?.reshape((t, c.n_k_heads * n))
                };
                let qkv = Tensor::cat(&[flat(0, hk)?, flat(hk, hk)?, flat(2 * hk, vr)?], 1)?;
                Ok((qkv, flat(2 * hk + vr, vr)?))
            }
        }
    }

    /// `(β, g)` per token and value head, `[t, n_v]` each: `β = sigmoid(b)`,
    /// `g = softplus(α + dt_bias) · a` (the log-decay).
    fn gates(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let c = &self.ssm;
        let t = xs.dim(0)?;
        let (b, alpha) = match &self.ba {
            BetaAlpha::Fused(w) => {
                let r = c.n_v_heads / c.n_k_heads;
                let y = w.forward(xs)?.reshape((t, c.n_k_heads, 2 * r))?;
                (
                    y.narrow(2, 0, r)?.contiguous()?.reshape((t, c.n_v_heads))?,
                    y.narrow(2, r, r)?.contiguous()?.reshape((t, c.n_v_heads))?,
                )
            }
            BetaAlpha::Split { beta, alpha } => (beta.forward(xs)?, alpha.forward(xs)?),
        };
        let g = crate::moe::softplus(&alpha.broadcast_add(&self.dt_bias)?)?.broadcast_mul(&self.a)?;
        Ok((sigmoid(&b)?, g))
    }

    /// Causal depthwise conv over `qkv` (`[t, conv_dim]`) continuing from the
    /// state's tail, followed by SiLU.  Updates the tail.
    fn conv(&self, state: &mut DeltaState, qkv: &Tensor) -> Result<Tensor> {
        let t = qkv.dim(0)?;
        let k = self.ssm.d_conv;
        let inp = Tensor::cat(&[&state.conv, &qkv.t()?], 1)?; // [conv_dim, k - 1 + t]
        let mut acc = inp.narrow(1, 0, t)?.broadcast_mul(&self.conv_w.narrow(1, 0, 1)?)?;
        for j in 1..k {
            acc = (acc + inp.narrow(1, j, t)?.broadcast_mul(&self.conv_w.narrow(1, j, 1)?)?)?;
        }
        state.conv = inp.narrow(1, t, k - 1)?.contiguous()?;
        silu(&acc.t()?.contiguous()?)
    }

    fn fresh_state(&self, dev: &Device) -> Result<DeltaState> {
        let c = &self.ssm;
        Ok(DeltaState {
            conv: Tensor::zeros((c.conv_dim(), c.d_conv - 1), DType::F32, dev)?,
            ssm: vec![0.0; c.n_v_heads * c.head_k_dim * c.head_v_dim],
        })
    }

    fn forward(&self, state: &mut Option<DeltaState>, xs: &Tensor) -> Result<Tensor> {
        let _p = prof::Phase::start(&prof::ATT);
        let c = &self.ssm;
        let (b, t, h) = xs.dims3()?;
        if b != 1 {
            candle_core::bail!("Gated DeltaNet supports batch size 1, got {b}");
        }
        let x2 = xs.reshape((t, h))?;
        let state = match state {
            Some(s) => s,
            None => state.insert(self.fresh_state(xs.device())?),
        };

        let (qkv, z) = self.project(&x2)?;
        let (beta, g) = self.gates(&x2)?;
        let mixed = self.conv(state, &qkv)?; // [t, conv_dim]
        let (kd, vd) = (c.key_dim(), c.value_dim());
        let q = l2_norm(&mixed.narrow(1, 0, kd)?.reshape((t, c.n_k_heads, c.head_k_dim))?, self.eps)?;
        let k = l2_norm(&mixed.narrow(1, kd, kd)?.reshape((t, c.n_k_heads, c.head_k_dim))?, self.eps)?;
        let v = mixed.narrow(1, 2 * kd, vd)?;

        let host = |x: &Tensor| -> Result<Vec<f32>> { x.flatten_all()?.to_vec1::<f32>() };
        let o = gated_delta_rule(
            c,
            &host(&q)?,
            &host(&k)?,
            &host(&v)?,
            &host(&g)?,
            &host(&beta)?,
            &mut state.ssm,
            t,
        );
        let o = Tensor::from_vec(o, (t, c.n_v_heads, c.head_v_dim), xs.device())?;

        // Gated RMSNorm over each head: norm(o) · silu(z).
        let z = z.reshape((t, c.n_v_heads, c.head_v_dim))?;
        let o = (self.norm.forward(&o)? * silu(&z)?)?.reshape((1, t, vd))?;
        self.out.forward(&o)
    }
}

/// `x / sqrt(sum(x²) + eps)` over the last dim (llama.cpp
/// `build_gdn_l2_norm`).
fn l2_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    x.broadcast_div(&(x.sqr()?.sum_keepdim(D::Minus1)? + eps)?.sqrt()?)
}

/// The gated delta rule over `t` tokens, updating `state` in place and
/// returning the outputs `[t, n_v, d_v]` (row-major).
///
/// Per value head `h` (reading key head `c.key_head(h)`), with `S` the
/// `d_k × d_v` state and `q` pre-scaled by `1/sqrt(d_k)`:
///
/// ```text
/// S  ← exp(g) · S
/// u  ← β · (v − Sᵀk)
/// S  ← S + k uᵀ
/// o  ← Sᵀq
/// ```
///
/// — llama.cpp `build_delta_net_autoregressive`, applied token by token (its
/// chunked prefill form computes the same recurrence).  Runs on the host,
/// one head per task: each head's state stays cache-resident across the
/// whole sequence.
#[allow(clippy::too_many_arguments)]
fn gated_delta_rule(
    c: &SsmConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut [f32],
    t: usize,
) -> Vec<f32> {
    use rayon::prelude::*;

    let (dk, dv, nk, nv) = (c.head_k_dim, c.head_v_dim, c.n_k_heads, c.n_v_heads);
    let scale = 1.0 / (dk as f32).sqrt();
    let per_head: Vec<Vec<f32>> = state
        .par_chunks_mut(dk * dv)
        .enumerate()
        .map(|(h, s)| {
            let kh = c.key_head(h);
            let mut out = vec![0f32; t * dv];
            let mut u = vec![0f32; dv];
            for tok in 0..t {
                let qv = &q[(tok * nk + kh) * dk..][..dk];
                let kv = &k[(tok * nk + kh) * dk..][..dk];
                let vv = &v[(tok * nv + h) * dv..][..dv];
                let decay = g[tok * nv + h].exp();
                let b = beta[tok * nv + h];
                // Decay, then Sᵀk.
                u.iter_mut().for_each(|x| *x = 0.0);
                for (i, &ki) in kv.iter().enumerate() {
                    let row = &mut s[i * dv..(i + 1) * dv];
                    for (j, sij) in row.iter_mut().enumerate() {
                        *sij *= decay;
                        u[j] += *sij * ki;
                    }
                }
                for (j, uj) in u.iter_mut().enumerate() {
                    *uj = b * (vv[j] - *uj);
                }
                // Rank-1 update, then Sᵀq.
                let o = &mut out[tok * dv..(tok + 1) * dv];
                for i in 0..dk {
                    let (ki, qi) = (kv[i], qv[i] * scale);
                    let row = &mut s[i * dv..(i + 1) * dv];
                    for j in 0..dv {
                        row[j] += ki * u[j];
                        o[j] += row[j] * qi;
                    }
                }
            }
            out
        })
        .collect();
    // [n_v][t][d_v] → [t][n_v][d_v]
    let mut o = vec![0f32; t * nv * dv];
    for (h, head) in per_head.iter().enumerate() {
        for tok in 0..t {
            o[(tok * nv + h) * dv..][..dv].copy_from_slice(&head[tok * dv..(tok + 1) * dv]);
        }
    }
    o
}

/// A layer's token mixer.
enum Mixer {
    Attention(Attention),
    DeltaNet(GatedDeltaNet),
}

// ─── Feed-forward (dense SwiGLU or softmax-routed MoE) ──────────────────────

#[derive(Clone)]
struct Mlp {
    gate: Weight,
    up: Weight,
    down: Weight,
    /// Per-tensor byte-range handles for best-effort page prefetch, present
    /// only when the weights are borrowed from the model mapping (absent for
    /// zero-copy Metal buffers and streamed loads).
    prefetch: Option<crate::residency::ExpertHandles>,
}

impl Mlp {
    fn load<R: Read + Seek>(rd: &mut Reader<R>, p: &str, suffix: &str) -> Result<Self> {
        Ok(Self {
            gate: rd.qmatmul(&format!("{p}.ffn_gate{suffix}.weight"))?,
            up: rd.qmatmul(&format!("{p}.ffn_up{suffix}.weight"))?,
            down: rd.qmatmul(&format!("{p}.ffn_down{suffix}.weight"))?,
            prefetch: None,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.gate.forward(xs)?;
        let w3 = self.up.forward(xs)?;
        self.down.forward(&(silu(&w1)? * w3)?)
    }

    /// This expert with every weight re-uploaded onto `device` (see
    /// [`Weight::to_uploadable`]); `None` when a weight has no standalone
    /// quantized form.
    fn to_device(&self, device: &Device) -> Option<Self> {
        Some(Self {
            gate: self.gate.to_uploadable(device)?,
            up: self.up.to_uploadable(device)?,
            down: self.down.to_uploadable(device)?,
            prefetch: None,
        })
    }
}

impl crate::moe::Expert for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Mlp::forward(self, xs)
    }
}

/// The device-resident form of one routed expert: the same gate/up/down
/// projections, uploaded to the expert device, wrapped in an opaque
/// [`crate::residency::DeviceExpertSlot`] payload so the
/// [`crate::residency::DeviceResidency`] pool can hold it and dispatch can run
/// it on the GPU.
struct DeviceExpert {
    mlp: Mlp,
    bytes: u64,
}

impl crate::residency::DeviceExpertSlot for DeviceExpert {
    fn device_bytes(&self) -> u64 {
        self.bytes
    }
}

/// An always-on shared expert, optionally scaled per token by
/// `sigmoid(x · ffn_gate_inp_shexp)` (Qwen2-MoE, Qwen3-Next, Qwen3.5-MoE).
struct SharedExpert {
    mlp: Mlp,
    /// The gate vector as a `[n_embd, 1]` column.
    gate: Option<Tensor>,
}

impl SharedExpert {
    fn forward(&self, x2: &Tensor) -> Result<Tensor> {
        let out = self.mlp.forward(x2)?;
        match &self.gate {
            Some(g) => out.broadcast_mul(&sigmoid(&x2.matmul(g)?)?),
            None => Ok(out),
        }
    }
}

struct Moe {
    gate_t: Tensor, // router weight transposed to [n_embd, n_expert], contiguous (cached)
    experts: Vec<Mlp>,
    shared: Option<SharedExpert>,
    n_expert_used: usize,
    weights_norm: bool,
    /// Scale applied to the routing weights (`0` / `1`: none).
    weights_scale: f64,
    /// This MoE block's layer index (for the residency key).
    layer: u32,
    /// `general.architecture`, for error context.
    arch: &'static str,
    /// Device the routed-expert weights live on.  The model device when the
    /// experts were uploaded; the CPU when the pool stays in host RAM on an
    /// accelerator (see [`crate::placement::ExpertPlacement`]).  `dispatch`
    /// moves the block's input across once and its result back once.
    expert_device: Device,
    /// Bounded VRAM expert cache (#62): `Some` when a device-resident subset of
    /// the routed experts is enabled.  `dispatch`/`dispatch_decode` call
    /// [`crate::residency::DeviceResidency::lookup`] per expert and run the
    /// resident device form on `expert_device`, falling back to the host
    /// `experts[e]` on a miss.  `None` (the default) keeps today's all-host or
    /// all-device path byte-for-byte.
    residency: Option<std::sync::Arc<crate::residency::DeviceResidency<DeviceExpert>>>,
}

impl Moe {
    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Vec<u32>)> {
        let _p = prof::Phase::start(&prof::MOE);
        let (b, seq_len, h) = xs.dims3()?;
        let n_tokens = b * seq_len;
        let x2 = xs.reshape((n_tokens, h))?;
        let (topk_idx, weights) = self.route(&x2)?;
        // dispatch already drains the routed ids to the host for its own
        // bucketing; reuse them for the hot-expert cache instead of a second
        // device-to-host sync (which would cost a synchronization per layer
        // even when the cache is disabled).
        let (mut out, ids) = self.dispatch(&x2, &topk_idx, &weights, n_tokens)?;
        if let Some(shared) = &self.shared {
            out = (out + shared.forward(&x2)?)?;
        }
        Ok((out.reshape((b, seq_len, h))?, ids))
    }

    /// Router logits → softmax probs → top-k ids and gathered weights, with
    /// llama.cpp's `norm_topk_prob` normalisation (min-clamp to the f16
    /// epsilon so a degenerate routing can never divide by zero).
    fn route(&self, x2: &Tensor) -> Result<(Tensor, Tensor)> {
        let logits = x2.matmul(&self.gate_t)?; // [n_tokens, n_expert]
        let probs = softmax_last_dim(&logits)?;
        let topk_idx = crate::moe::topk_indices(&probs, self.n_expert_used)?; // [n_tokens, k]
        let mut weights = probs.gather(&topk_idx, D::Minus1)?; // [n_tokens, k]
        if self.weights_norm {
            let denom = weights
                .sum_keepdim(D::Minus1)?
                .clamp(6.103_515_6e-5, f32::INFINITY)?;
            weights = weights.broadcast_div(&denom)?;
        }
        if self.weights_scale != 0.0 && self.weights_scale != 1.0 {
            weights = (weights * self.weights_scale)?;
        }
        Ok((topk_idx, weights))
    }

    /// Run each selected expert over its routed tokens and accumulate the
    /// weighted outputs. Experts stay quantized.
    ///
    /// Prefill (`n_tokens > 1`) buckets tokens per expert and runs one batched
    /// matmul per expert; decode (`n_tokens == 1`) runs the `k` experts over
    /// the one row with no per-expert host round-trip.  Both live in
    /// [`crate::moe`], shared with the other MoE loaders.
    fn dispatch(
        &self,
        x2: &Tensor,
        topk_idx: &Tensor,
        weights: &Tensor,
        n_tokens: usize,
    ) -> Result<(Tensor, Vec<u32>)> {
        let _p = prof::Phase::start(&prof::EXPERTS);
        // Residency-aware per-expert forward: on a `DeviceResidency` hit run the
        // device-resident (uploaded) weights, moving the activation to that
        // expert's device; on a miss run the host form. `None` residency keeps
        // today's single-device path byte-for-byte (`dispatch_decode`/`_prefill`).
        let fwd: crate::moe::ResidencyForward<'_> = match &self.residency {
            Some(res) => Some(&(|e: usize, x: &Tensor| {
                if let Some(dev) = res.lookup(self.layer, e as u32) {
                    let xd = crate::moe::on_device(x, &self.expert_device)?;
                    let out = dev.mlp.forward(&xd)?;
                    crate::moe::on_device(&out, x.device()).map(|c| c.into_owned())
                } else {
                    self.experts[e].forward(x)
                }
            })),
            None => None,
        };
        if n_tokens == 1 {
            crate::moe::dispatch_decode_with(
                self.arch,
                &self.experts,
                &self.expert_device,
                x2,
                topk_idx,
                weights,
                self.n_expert_used,
                fwd,
            )
        } else {
            crate::moe::dispatch_prefill_with(
                self.arch,
                &self.experts,
                &self.expert_device,
                x2,
                topk_idx,
                weights,
                n_tokens,
                self.n_expert_used,
                fwd,
            )
        }
    }
}

// ─── Layer + model ───────────────────────────────────────────────────────────

enum FeedForward {
    Dense(Mlp),
    Moe(Moe),
}

impl FeedForward {
    /// The block's output plus the routed-expert ids this input used (empty
    /// for dense blocks) — the routing-frequency cache's input.
    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Vec<u32>)> {
        match self {
            Self::Dense(m) => Ok((m.forward(xs)?, Vec::new())),
            Self::Moe(m) => m.forward(xs),
        }
    }
}

struct Layer {
    attn_norm: RmsNorm,
    mixer: Mixer,
    /// `ffn_norm`, or `post_attention_norm` in the Qwen3-Next generation.
    ffn_norm: RmsNorm,
    ffn: FeedForward,
}

/// One layer's per-session state: the KV cache of an attention layer or the
/// recurrent state of a Gated DeltaNet layer.
#[derive(Clone, Default)]
pub struct LayerState {
    kv: KvCache,
    delta: Option<DeltaState>,
}

/// The immutable half of a loaded model: every weight, the expert residency
/// backend and the device.  Sessions ([`ModelWeights`]) share one `Arc` of
/// it and own only their per-layer state, so a second concurrent
/// conversation costs its cache rather than a second copy of the weights —
/// on an accelerator, a second upload of the whole model.
pub struct Weights {
    tok_embeddings: TokenEmbedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    output: Linear,
    device: Device,
    /// Executes residency for the hot set (CPU madvise today; a device slot
    /// cache later).  Built once at load from the per-expert handles.
    residency: Arc<dyn crate::residency::ExpertResidency>,
    n_expert: usize,
}

impl crate::native_session::LayerStack for Weights {
    type State = LayerState;

    fn n_layers(&self) -> usize {
        self.layers.len()
    }
    fn n_expert(&self) -> usize {
        self.n_expert
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn tok_embeddings(&self) -> &TokenEmbedding {
        &self.tok_embeddings
    }
    fn residency(&self) -> &Arc<dyn crate::residency::ExpertResidency> {
        &self.residency
    }

    fn layer(
        &self,
        l: usize,
        state: &mut LayerState,
        xs: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<(Tensor, Vec<u32>)> {
        let layer = &self.layers[l];
        let h = layer.attn_norm.forward(xs)?;
        let h = match &layer.mixer {
            Mixer::Attention(a) => a.forward(&mut state.kv, &h, mask, offset)?,
            Mixer::DeltaNet(d) => d.forward(&mut state.delta, &h)?,
        };
        let xs = (xs + h)?;
        let (h, routed) = layer.ffn.forward(&layer.ffn_norm.forward(&xs)?)?;
        Ok(((xs + h)?, routed))
    }

    fn head(&self, xs: &Tensor) -> Result<Tensor> {
        let _p = prof::Phase::start(&prof::HEAD);
        let out = self.output.forward(&self.norm.forward(xs)?)?.to_dtype(DType::F32);
        drop(_p);
        prof::report();
        out
    }

    fn can_truncate(&self) -> bool {
        !self.layers.iter().any(|l| matches!(l.mixer, Mixer::DeltaNet(_)))
    }

    fn truncate_state(state: &mut LayerState, keep: usize) -> Result<()> {
        if keep == 0 {
            *state = LayerState::default();
            return Ok(());
        }
        crate::moe::truncate_kv(&mut state.kv, keep, crate::attention::KV_SEQ_DIM)
    }
}

/// A quantized Qwen-family model loaded from GGUF: one session over shared
/// [`Weights`].
pub type ModelWeights = crate::native_session::Session<Weights>;

/// Slice a 3-D expert tensor `[n_expert, out, in]` into per-expert [`Weight`]s
/// plus an optional prefetch handle for the expert's bytes in the mapping.
///
/// Preferred paths point each expert at its own bytes **without reading or
/// copying anything**: on Metal via a no-copy buffer at the expert's file
/// offset, on CPU via a borrowed slice of the mapping.  Either way an expert
/// costs a pointer and a length until tokens actually route to it.  The
/// fallback (no mapping, or a tensor that cannot be borrowed) reads the whole
/// tensor and copies per-expert slices onto the device.
/// One routed expert's weight plus, when it is borrowed from the mapping,
/// the handle that can prefetch its byte range.
type ExpertPart = (Weight, Option<Arc<dyn crate::mmap_tensor::MmapPrefetch>>);

fn split_experts<R: Read + Seek>(
    rd: &mut Reader<R>,
    name: &str,
    n_expert: usize,
) -> Result<Vec<ExpertPart>> {
    let et = crate::moe::ExpertTensor::lookup(&rd.ct, rd.arch, name, n_expert)?;
    let host_experts = rd.expert_device.is_cpu();
    let candle = |qt: QTensor| -> Result<Weight> { Ok(Weight::Candle(QMatMul::from_qtensor(qt)?)) };

    // Host-resident experts with a mapping: borrow each expert's slice of the
    // mapping (see `ExpertTensor::borrow_host`).  On an accelerator this is
    // the `ExpertPlacement::Host` layout: dense set on the device, experts in
    // host RAM, activations hopping across in `Moe::dispatch`.
    if host_experts {
        if let Some(mmap) = rd.mmap.as_ref() {
            if let Some(borrowed) = et.borrow_host(mmap)? {
                return borrowed
                    .into_iter()
                    .map(|b| Ok((candle(b.tensor)?, b.prefetch)))
                    .collect();
            }
        }
    }

    // Zero-copy Metal path: each expert is a byte range inside the shared
    // no-copy buffer.  No reads, no uploads.
    if !host_experts {
        if let (Some(per_bytes), Some(zc), Some(info)) =
            (et.bytes_per_expert(), &rd.zc, rd.ct.tensor_infos.get(name))
        {
            let [out, inn] = [et.expert_shape().0, et.expert_shape().1];
            let mut experts = Vec::with_capacity(n_expert);
            for e in 0..n_expert {
                experts.push((
                    Weight::Zc(Arc::new(ZcWeight::expert(
                        zc,
                        info,
                        rd.ct.tensor_data_offset,
                        [out, inn],
                        e * per_bytes,
                    )?)),
                    None, // Metal no-copy buffers have no pages to advise.
                ));
            }
            return Ok(experts);
        }
    }

    // GPU paging path: keep each expert's quantized block compressed in the
    // mmap and upload it to the device on demand through the bounded cache,
    // instead of copying every expert up front.
    if !host_experts {
        if let (Some(per_bytes), Some(cache)) = (et.bytes_per_expert(), rd.gpu_cache.clone()) {
            let (out, inn) = et.expert_shape();
            let mut paged = Vec::with_capacity(n_expert);
            for e in 0..n_expert {
                match cache.weight(et.base() + e * per_bytes, out, inn, et.dtype()) {
                    Ok(p) => paged.push((Weight::Paged(Arc::new(p)), None)),
                    Err(_) => {
                        paged.clear();
                        break;
                    }
                }
            }
            if paged.len() == n_expert {
                return Ok(paged);
            }
        }
    }

    // Device-resident experts with a mapping: upload each expert straight
    // from its bytes in the mapping (see `ExpertTensor::upload_from_mmap`).
    if !host_experts {
        if let Some(mmap) = rd.mmap.as_ref() {
            if let Some(uploaded) = et.upload_from_mmap(mmap, &rd.expert_device)? {
                return uploaded.into_iter().map(|qt| Ok((candle(qt)?, None))).collect();
            }
        }
    }

    // No mapping (streamed load) or a tensor that cannot be sliced.
    et.read_and_split(&rd.ct, &mut rd.reader, rd.arch, &rd.expert_device)?
        .into_iter()
        .map(|qt| Ok((candle(qt)?, None)))
        .collect()
}

fn load_moe<R: Read + Seek>(rd: &mut Reader<R>, p: &str, cfg: &Config) -> Result<Moe> {
    let gate_t = rd
        .f32_tensor(&format!("{p}.ffn_gate_inp.weight"))? // [n_expert, n_embd]
        .t()?
        .contiguous()?; // [n_embd, n_expert]

    if rd.has(&format!("{p}.ffn_gate_up_exps.weight")) {
        candle_core::bail!(
            "{}: fused `ffn_gate_up_exps` expert tensors are not supported; \
             convert without --fuse-gate-up-exps",
            rd.arch
        );
    }
    let gate_exps = split_experts(rd, &format!("{p}.ffn_gate_exps.weight"), cfg.n_expert)?;
    let up_exps = split_experts(rd, &format!("{p}.ffn_up_exps.weight"), cfg.n_expert)?;
    let down_exps = split_experts(rd, &format!("{p}.ffn_down_exps.weight"), cfg.n_expert)?;
    let experts: Vec<Mlp> = gate_exps
        .into_iter()
        .zip(up_exps)
        .zip(down_exps)
        .map(|((gate, up), down)| Mlp {
            gate: gate.0,
            up: up.0,
            down: down.0,
            prefetch: crate::residency::ExpertHandles::from_parts(gate.1, up.1, down.1),
        })
        .collect();

    let layer = p
        .rsplit('.')
        .next()
        .and_then(|id| id.parse::<u32>().ok())
        .unwrap_or(0);

    // Build the bounded VRAM expert cache (#62) when a byte budget is supplied.
    // The upload closure makes an expert's device form resident on demand.  On
    // the CPU (the only backend this box can run) it wraps the already-loaded
    // host weights — identical numbers, so a hit is numerically the host path,
    // which the mixed-residency parity test asserts.  On a real accelerator the
    // device branch uploads the expert's bytes (see `cuda_io`) — that is the
    // GPU-validated follow-up; until then the same code path runs and a hit is a
    // no-op device copy we cannot benchmark without hardware.
    let residency = match rd.device_expert_cache_bytes {
        Some(cap_bytes) if cap_bytes > 0 => {
            let host = experts.clone(); // fallback / CPU-parity source
            // Per-slot uploaded size estimate; used only by `capacity()`.
            let per_slot = crate::moe::EXPERT_SLOT_BYTES_ESTIMATE;
            let expert_dev = rd.expert_device.clone();
            let upload: std::sync::Arc<
                dyn Fn(u32, u32) -> Option<std::sync::Arc<DeviceExpert>> + Send + Sync,
            > = std::sync::Arc::new(move |l, e| {
                if l != layer {
                    return None;
                }
                let m: &Mlp = host.get(e as usize)?;
                // On a non-CPU expert device, upload each weight's quantized
                // blocks onto that device so a residency hit runs for real on
                // the accelerator (#62; backend-generic via QStorage::from_data).
                // On the CPU we keep sharing the host weights — a hit is
                // numerically the host path, which the parity tests assert.
                let mlp = if expert_dev.is_cpu() { m.clone() } else { m.to_device(&expert_dev)? };
                Some(std::sync::Arc::new(DeviceExpert { mlp, bytes: per_slot }))
            });
            Some(std::sync::Arc::new(crate::residency::DeviceResidency::<DeviceExpert>::new(
                cap_bytes,
                per_slot,
                upload,
            )))
        }
        _ => None,
    };

    let shared = if rd.has(&format!("{p}.ffn_up_shexp.weight")) {
        let gate = rd
            .f32_opt(&format!("{p}.ffn_gate_inp_shexp.weight"))?
            .map(|g| g.flatten_all()?.unsqueeze(1))
            .transpose()?;
        Some(SharedExpert { mlp: Mlp::load(rd, p, "_shexp")?, gate })
    } else {
        None
    };

    Ok(Moe {
        gate_t,
        experts,
        shared,
        n_expert_used: cfg.n_expert_used,
        weights_norm: cfg.expert_weights_norm,
        weights_scale: cfg.expert_weights_scale,
        layer,
        arch: rd.arch,
        expert_device: rd.expert_device.clone(),
        residency,
    })
}

impl ModelWeights {
    /// Load a Qwen-family GGUF (see [`ARCHES`]).
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        Self::from_gguf_mmap(ct, reader, device, None)
    }

    /// Load with weights borrowed in place from `mmap` where possible.
    ///
    /// Without a mapping every tensor is copied onto the heap and the whole
    /// model must fit in RAM.  With one, weights are referenced directly in
    /// the page cache and fault in on demand.  The routed experts go to
    /// `device` too; see [`ModelWeights::from_gguf_mmap_placed`] to keep them
    /// in host RAM on an accelerator.
    pub fn from_gguf_mmap<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        mmap: Option<Arc<memmap2::Mmap>>,
    ) -> Result<Self> {
        Self::from_gguf_mmap_placed(ct, reader, device, device, mmap, None)
    }

    /// [`ModelWeights::from_gguf_mmap`] with an explicit device for the routed
    /// experts.
    ///
    /// The dense set always goes to `device`.  With `expert_device` the CPU
    /// on an accelerator model, the routed-expert pool stays in host RAM —
    /// borrowed in place from `mmap` (or read onto the heap without one) and
    /// run through the CPU expert kernels, with the MoE block's activations
    /// hopping across the bus once per layer.  That is the layout for a
    /// model larger than the device's memory (see
    /// [`crate::placement::ExpertPlacement`]).  Any other `expert_device`
    /// must equal `device`.
    pub fn from_gguf_mmap_placed<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        expert_device: &Device,
        mmap: Option<Arc<memmap2::Mmap>>,
        device_expert_cache_bytes: Option<u64>,
    ) -> Result<Self> {
        let arch = crate::model::Architecture::arch_name(&ct.metadata).unwrap_or_default();
        let cfg = Config::from_metadata(&ct.metadata, &arch)?;
        if !expert_device.is_cpu() && !expert_device.same_device(device) {
            candle_core::bail!(
                "{}: routed experts must live on the model device or the CPU, not {expert_device:?}",
                cfg.arch
            );
        }
        // Zero-copy Metal: bind the mapped weights into no-copy GPU buffers
        // so quantized weights are never uploaded.  Best effort — if the
        // mapping is not page-aligned or the device refuses the buffers, fall
        // back to candle's copying path.  Chunk boundaries follow tensor
        // boundaries so no weight straddles two buffers.
        let zc = match (&device, &mmap) {
            (Device::Metal(md), Some(mmap)) if !expert_device.is_cpu() => {
                match ZcContext::new_for_tensors(md, mmap.clone(), &ct.tensor_infos, ct.tensor_data_offset)
                {
                    Ok(zc) => {
                        tracing::info!(
                            "zero-copy Metal: binding {} bytes of weights into {} no-copy GPU buffers",
                            zc.len(),
                            zc.num_chunks()
                        );
                        Some(Arc::new(zc))
                    }
                    Err(e) => {
                        tracing::warn!(
                            "zero-copy Metal unavailable ({}); copying weights onto the GPU",
                            e
                        );
                        None
                    }
                }
            }
            _ => None,
        };
        // Opt-in bounded GPU weight cache.  Heed the env budget (MiB); 0/unset
        // disables.  Only meaningful with an mmap-backed model on a device that
        // can hold quantized blocks; building it is best-effort.
        let gpu_cache = match std::env::var("JOSHUA_GPU_WEIGHT_CACHE") {
            Ok(v) if !v.is_empty() && v != "0" => {
                let mib = v.parse::<usize>().unwrap_or(0);
                if mib == 0 {
                    None
                } else if let Some(m) = mmap.clone() {
                    crate::paged_weights::WeightCache::new(
                        m.clone(),
                        device.clone(),
                        mib * 1024 * 1024,
                    ).ok()
                } else {
                    tracing::warn!("JOSHUA_GPU_WEIGHT_CACHE set but no mmap-backed model; disabled");
                    None
                }
            }
            _ => None,
        };
        // The Content owns metadata; move it into our reader together with the
        // underlying file handle (borrowed for the lifetime of the load).
        let mut rd = Reader {
            ct,
            reader,
            device: device.clone(),
            expert_device: expert_device.clone(),
            mmap,
            device_expert_cache_bytes,
            zc,
            gpu_cache,
            arch: cfg.arch,
        };

        // Kept quantized: dequantizing the table to f32 costs vocab × hidden
        // × 4 bytes of anonymous memory per instance (1.2 GiB on 30B-A3B).
        let tok_embeddings = TokenEmbedding::load(rd.qtensor("token_embd.weight")?, device)?;
        let norm = rd.rms_norm("output_norm.weight", cfg.rms_eps)?;
        let output = Linear {
            w: match rd.qmatmul_opt("output.weight") {
                Some(q) => q,
                // tie_word_embeddings conversions ship no output head.
                None => rd.qmatmul("token_embd.weight")?,
            },
            b: rd.f32_opt("output.bias")?,
        };

        let rope = Arc::new(cfg.rope(device)?);

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for layer_idx in 0..cfg.n_layer {
            let p = format!("blk.{layer_idx}");
            let attn_norm = rd.rms_norm(&format!("{p}.attn_norm.weight"), cfg.rms_eps)?;
            let ffn_norm = match rd.rms_norm_opt(&format!("{p}.post_attention_norm.weight"), cfg.rms_eps)? {
                Some(n) => n,
                None => rd.rms_norm(&format!("{p}.ffn_norm.weight"), cfg.rms_eps)?,
            };
            let mixer = if cfg.recurrent[layer_idx] {
                Mixer::DeltaNet(GatedDeltaNet::load(&mut rd, &p, &cfg)?)
            } else {
                Mixer::Attention(Attention::load(&mut rd, &p, &cfg, rope.clone())?)
            };
            let ffn = if cfg.n_expert > 0 {
                FeedForward::Moe(load_moe(&mut rd, &p, &cfg)?)
            } else {
                FeedForward::Dense(Mlp::load(&mut rd, &p, "")?)
            };
            layers.push(Layer {
                attn_norm,
                mixer,
                ffn_norm,
                ffn,
            });
        }

        let residency: std::sync::Arc<dyn crate::residency::ExpertResidency> =
            std::sync::Arc::new(crate::residency::CpuResidency::new(
                layers
                    .iter()
                    .map(|layer| match &layer.ffn {
                        FeedForward::Moe(moe) => moe.experts.iter().map(|m| m.prefetch.clone()).collect(),
                        FeedForward::Dense(_) => Vec::new(),
                    })
                    .collect(),
            ));
        Ok(Self::from_weights(Weights {
            tok_embeddings,
            layers,
            norm,
            output,
            device: device.clone(),
            residency,
            n_expert: cfg.n_expert,
        }))
    }
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::{GgmlDType, QStorage};
    use std::borrow::Cow;

    fn lin(_rows: usize, _cols: usize, t: &Tensor) -> Weight {
        Weight::Candle(QMatMul::Tensor(t.clone()))
    }

    fn rms_from_f32(weights: &[f32], dim: usize, dev: &Device) -> Result<RmsNorm> {
        let bytes: Vec<u8> = weights.iter().flat_map(|x| x.to_le_bytes()).collect();
        let storage = QStorage::from_data(Cow::Owned(bytes), dev, GgmlDType::F32)?;
        RmsNorm::from_qtensor(QTensor::new(storage, dim)?, 1e-5)
    }

    fn tiny_moe(dev: &Device, weights_norm: bool) -> Result<Moe> {
        let (h, ne, nfe) = (8usize, 4usize, 16usize);
        let gate_t = Tensor::randn(0f32, 1f32, (h, ne), dev)?.contiguous()?; // [n_embd, n_expert]
        let mut experts = Vec::with_capacity(ne);
        for _ in 0..ne {
            experts.push(Mlp {
                gate: lin(nfe, h, &Tensor::randn(0f32, 1f32, (nfe, h), dev)?),
                up: lin(nfe, h, &Tensor::randn(0f32, 1f32, (nfe, h), dev)?),
                down: lin(h, nfe, &Tensor::randn(0f32, 1f32, (h, nfe), dev)?),
                prefetch: None,
            });
        }
        Ok(Moe {
            gate_t,
            experts,
            shared: None,
            n_expert_used: 2,
            weights_norm,
            weights_scale: 0.0,
            layer: 0,
            arch: "qwen3moe",
            expert_device: dev.clone(),
            residency: None,
        })
    }

    /// A mixed-residency `Moe` (some experts resident in a `DeviceResidency`,
    /// the rest host) must produce **same logits** as the all-host `Moe` from
    /// the same weights.  The fake "device" form wraps the same host weights,
    /// so the assertion exercises the dispatch *partition* (per-expert lookup /
    /// hit-vs-miss routing), not real device math — the actual VRAM copy and
    /// GPU matmul still need a real accelerator to validate (issue #62).
    #[test]
    fn mixed_residency_dispatch_matches_all_host() -> Result<()> {
        let dev = Device::Cpu;
        let moe = tiny_moe(&dev, false)?; // all-host reference

        // A cache whose slots wrap the **same** expert weights (as "device
        // form"), so a hit runs numerically-identical math on the CPU.
        let experts: Vec<Mlp> = moe.experts.clone();
        let upload: std::sync::Arc<
            dyn Fn(u32, u32) -> Option<std::sync::Arc<DeviceExpert>> + Send + Sync,
        > = std::sync::Arc::new(move |l, e| {
            experts
                .get(e as usize)
                .map(|m| std::sync::Arc::new(DeviceExpert { mlp: m.clone(), bytes: 1024 }))
                .filter(|_| l == 0)
        });
        let res = std::sync::Arc::new(crate::residency::DeviceResidency::<DeviceExpert>::new(
            2048, 1024, upload,
        ));
        // Experts 1,2 resident; 0,3 are host misses -> a genuine mixed layout.
        res.mark_hot(0, 1);
        res.mark_hot(0, 2);
        res.acquire(0, 1);
        res.acquire(0, 2);

        // mixed shares moe's exact weights (same clones), differing only in the
        // residency: hits run the (numerically identical) device form, misses run
        // the host form.
        let mixed = Moe {
            gate_t: moe.gate_t.clone(),
            experts: moe.experts.clone(),
            shared: None,
            n_expert_used: moe.n_expert_used,
            weights_norm: moe.weights_norm,
            weights_scale: moe.weights_scale,
            layer: moe.layer,
            arch: moe.arch,
            expert_device: moe.expert_device.clone(),
            residency: Some(std::sync::Arc::clone(&res)),
        };
        let _ = &res;

        let xs = Tensor::randn(0f32, 1f32, (1, 4, 8), &dev)?; // [b, seq, h]
        let (a, _) = moe.forward(&xs)?;
        let (b, _) = mixed.forward(&xs)?;
        let av = a.flatten_all()?.to_vec1::<f32>()?;
        let bv = b.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(av.len(), bv.len());
        for (i, (x, y)) in av.iter().zip(bv.iter()).enumerate() {
            assert!(
                (x - y).abs() < 1e-5,
                "mixed/host logit {i} diverges: {x} vs {y}"
            );
        }
        Ok(())
    }

    /// Half-split RoPE must pair `(i, i + d/2)` with the `(d/2)`-long
    /// frequency table — the exact convention llama.cpp uses for qwen3moe.
    #[test]
    fn rope_half_split_matches_manual() -> Result<()> {
        let dev = Device::Cpu;
        let dim = 8usize;
        let theta = 10_000f32;
        let rope = Rope::new(dim, theta, 64, RopeStyle::Neox, &dev)?;
        let x = Tensor::randn(0f32, 1f32, (1, 1, 1, dim), &dev)?; // [b, h, seq, d]
        let rotated = rope.apply(&x, 5)?;

        let xv: Vec<f32> = x.flatten_all()?.to_vec1()?;
        let rv: Vec<f32> = rotated.flatten_all()?.to_vec1()?;
        let half = dim / 2;
        let mut expected = vec![0f32; dim];
        for i_d in 0..half {
            let freq = 1f32 / theta.powf((2 * i_d) as f32 / dim as f32);
            let c = (5f32 * freq).cos();
            let s = (5f32 * freq).sin();
            expected[i_d] = xv[i_d] * c - xv[i_d + half] * s;
            expected[i_d + half] = xv[i_d] * s + xv[i_d + half] * c;
        }
        for i in 0..dim {
            assert!(
                (rv[i] - expected[i]).abs() < 1e-5,
                "rope element {i}: got {}, expected {}",
                rv[i],
                expected[i]
            );
        }
        Ok(())
    }

    /// With `norm_topk_prob`, the gathered routing weights must sum to 1 for
    /// every token (llama.cpp `build_moe_ffn` normalisation).
    #[test]
    fn moe_routing_normalizes_weights() -> Result<()> {
        let dev = Device::Cpu;
        let moe = tiny_moe(&dev, true)?;
        let xs = Tensor::randn(0f32, 1f32, (1, 3, 8), &dev)?;
        let (_, weights) = moe.route(&xs.reshape((3, 8))?)?;
        let sums: Vec<f32> = weights.sum_keepdim(D::Minus1)?.flatten_all()?.to_vec1()?;
        for s in sums {
            assert!((s - 1.0).abs() < 1e-4, "routing weights must sum to 1, got {s}");
        }
        Ok(())
    }

    /// The decode dispatch path must agree with the prefill bucketing path for
    /// the same token (they compute the same weighted expert sum).
    #[test]
    fn moe_prefill_matches_decode_path() -> Result<()> {
        let dev = Device::Cpu;
        let moe = tiny_moe(&dev, true)?;
        let row = Tensor::randn(0f32, 1f32, (1, 1, 8), &dev)?;
        let two = Tensor::cat(&[&row, &row], 1)?; // [1, 2, 8], identical rows
        let (out2, _) = moe.forward(&two)?; // prefill path (n_tokens = 2)
        let (out1, _) = moe.forward(&row)?; // decode path (n_tokens = 1)
        let a: Vec<f32> = out2.narrow(1, 1, 1)?.flatten_all()?.to_vec1()?;
        let b: Vec<f32> = out1.flatten_all()?.to_vec1()?;
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-4, "prefill vs decode diverge: {x} vs {y}");
        }
        Ok(())
    }

    fn tiny_attention(dev: &Device, fused: bool, gated: bool) -> Result<Attention> {
        let (h, nh, nkv, hd) = (8usize, 2usize, 1usize, 4usize);
        let qd = nh * hd * if gated { 2 } else { 1 };
        let mk = |r: usize, c: usize| Tensor::randn(0f32, 1f32, (r, c), dev).unwrap();
        let bias = |n: usize| Some(Tensor::randn(0f32, 1f32, n, dev).unwrap());
        let (wq, wk, wv) = (mk(qd, h), mk(nkv * hd, h), mk(nkv * hd, h));
        let (bq, bk, bv) = (bias(qd), bias(nkv * hd), bias(nkv * hd));
        let qkv = if fused {
            Qkv::Fused {
                qkv: Linear {
                    w: lin(0, 0, &Tensor::cat(&[&wq, &wk, &wv], 0)?),
                    b: Some(Tensor::cat(&[bq.unwrap(), bk.unwrap(), bv.unwrap()], 0)?),
                },
                q_dim: qd,
                k_dim: nkv * hd,
                v_dim: nkv * hd,
            }
        } else {
            Qkv::Split {
                q: Linear { w: lin(0, 0, &wq), b: bq },
                k: Linear { w: lin(0, 0, &wk), b: bk },
                v: Linear { w: lin(0, 0, &wv), b: bv },
            }
        };
        Ok(Attention {
            qkv,
            o: Linear { w: lin(h, nh * hd, &mk(h, nh * hd)), b: None },
            q_norm: Some(rms_from_f32(&vec![1.0; hd], hd, dev)?),
            k_norm: Some(rms_from_f32(&vec![1.0; hd], hd, dev)?),
            gated,
            rope: Arc::new(Rope::new(hd, 10_000.0, 64, RopeStyle::Neox, dev)?),
            n_head: nh,
            n_kv_head: nkv,
            head_dim: hd,
            scale: 1.0 / (hd as f64).sqrt(),
        })
    }

    /// Per-head Q/K norms reshape correctly and the KV cache appends along the
    /// sequence dim with the transposed per-head layout, with and without the
    /// output gate.
    #[test]
    fn attention_qk_norm_and_kv_cache_shapes() -> Result<()> {
        let dev = Device::Cpu;
        let (h, nkv, hd) = (8usize, 1usize, 4usize);
        for gated in [false, true] {
            let attn = tiny_attention(&dev, false, gated)?;
            let mut kv_cache: KvCache = None;

            let xs = Tensor::randn(0f32, 1f32, (1, 3, h), &dev)?;
            let out = attn.forward(&mut kv_cache, &xs, None, 0)?;
            assert_eq!(out.dims(), &[1, 3, h]);
            let (k, v) = kv_cache.as_ref().expect("cache after prefill");
            assert_eq!(k.dims(), &[1, nkv, hd, 3], "k cache must be [b, n_kv_head, head_dim, seq]");
            assert_eq!(v.dims(), &[1, nkv, hd, 3]);

            // Decode appends one more position.
            let _ = attn.forward(&mut kv_cache, &Tensor::randn(0f32, 1f32, (1, 1, h), &dev)?, None, 3)?;
            let (k, v) = kv_cache.as_ref().expect("cache after decode");
            assert_eq!(k.dims(), &[1, nkv, hd, 4]);
            assert_eq!(v.dims(), &[1, nkv, hd, 4]);
        }
        Ok(())
    }

    /// A fused `attn_qkv` (+ bias) is the same projection as split Q/K/V
    /// (+ biases) with the weights stacked.
    #[test]
    fn fused_qkv_matches_split_qkv() -> Result<()> {
        let dev = Device::Cpu;
        let split = tiny_attention(&dev, false, false)?;
        let Qkv::Split { q, k, v } = &split.qkv else { unreachable!() };
        let (Weight::Candle(QMatMul::Tensor(wq)), Weight::Candle(QMatMul::Tensor(wk)), Weight::Candle(QMatMul::Tensor(wv))) =
            (&q.w, &k.w, &v.w)
        else {
            unreachable!()
        };
        let fused = Qkv::Fused {
            qkv: Linear {
                w: lin(0, 0, &Tensor::cat(&[wq, wk, wv], 0)?),
                b: Some(Tensor::cat(&[q.b.clone().unwrap(), k.b.clone().unwrap(), v.b.clone().unwrap()], 0)?),
            },
            q_dim: wq.dim(0)?,
            k_dim: wk.dim(0)?,
            v_dim: wv.dim(0)?,
        };
        let xs = Tensor::randn(0f32, 1f32, (1, 3, 8), &dev)?;
        let (a, b) = (split.qkv.forward(&xs)?, fused.forward(&xs)?);
        for (x, y) in [(a.0, b.0), (a.1, b.1), (a.2, b.2)] {
            let d = (x - y)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            assert!(d < 1e-5, "fused vs split projection diverge: {d}");
        }
        Ok(())
    }

    /// The legacy `ssm_in` projection (per key head `[q, k, v, z]`) must
    /// regroup into exactly the `attn_qkv` + `attn_gate` split the current
    /// converter writes.
    #[test]
    fn legacy_ssm_in_regroups_into_split_projection() -> Result<()> {
        let dev = Device::Cpu;
        let c = SsmConfig { d_conv: 4, n_k_heads: 2, n_v_heads: 4, head_k_dim: 3, head_v_dim: 3, share: KeyHeadShare::Grouped };
        let (h, r) = (5usize, 2usize);
        let (hk, vr) = (c.head_k_dim, r * c.head_v_dim);
        let qkv = Tensor::randn(0f32, 1f32, (c.conv_dim(), h), &dev)?;
        let z = Tensor::randn(0f32, 1f32, (c.value_dim(), h), &dev)?;
        // Build the legacy per-key-head row layout from the split weights.
        let kd = c.key_dim();
        let mut rows = Vec::new();
        for g in 0..c.n_k_heads {
            rows.push(qkv.narrow(0, g * hk, hk)?);
            rows.push(qkv.narrow(0, kd + g * hk, hk)?);
            rows.push(qkv.narrow(0, 2 * kd + g * vr, vr)?);
            rows.push(z.narrow(0, g * vr, vr)?);
        }
        let legacy_w = Tensor::cat(&rows, 0)?;

        let net = |input: DeltaInput| GatedDeltaNet {
            input,
            ba: BetaAlpha::Fused(lin(0, 0, &Tensor::zeros((2 * c.n_v_heads, h), DType::F32, &dev).unwrap())),
            conv_w: Tensor::zeros((c.conv_dim(), c.d_conv), DType::F32, &dev).unwrap(),
            dt_bias: Tensor::zeros(c.n_v_heads, DType::F32, &dev).unwrap(),
            a: Tensor::zeros(c.n_v_heads, DType::F32, &dev).unwrap(),
            norm: rms_from_f32(&vec![1.0; c.head_v_dim], c.head_v_dim, &dev).unwrap(),
            out: lin(0, 0, &Tensor::zeros((h, c.value_dim()), DType::F32, &dev).unwrap()),
            ssm: c.clone(),
            eps: 1e-6,
        };
        let split = net(DeltaInput::Split { qkv: lin(0, 0, &qkv), z: lin(0, 0, &z) });
        let legacy = net(DeltaInput::Legacy(lin(0, 0, &legacy_w)));
        let xs = Tensor::randn(0f32, 1f32, (3, h), &dev)?;
        let (a, b) = (split.project(&xs)?, legacy.project(&xs)?);
        for (x, y) in [(a.0, b.0), (a.1, b.1)] {
            let d = (x - y)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            assert!(d < 1e-6, "legacy regrouping diverges: {d}");
        }
        Ok(())
    }

    /// The host delta rule must match a direct transcription of llama.cpp's
    /// autoregressive recurrence (`S ← e^g S; u = β(v − Sᵀk); S += k uᵀ;
    /// o = Sᵀq/√d_k`) for both key-head sharing layouts, and continue
    /// identically when a sequence is split across calls.
    #[test]
    fn gated_delta_rule_matches_reference_recurrence() {
        for share in [KeyHeadShare::Grouped, KeyHeadShare::Tiled] {
            let c = SsmConfig {
                d_conv: 4,
                n_k_heads: 2,
                n_v_heads: 4,
                head_k_dim: 3,
                head_v_dim: 3,
                share,
            };
            let (t, nk, nv, dk, dv) = (5usize, 2usize, 4usize, 3usize, 3usize);
            let mut seed = 7u32;
            let mut rnd = |n: usize| -> Vec<f32> {
                (0..n)
                    .map(|_| {
                        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
                    })
                    .collect()
            };
            let (q, k, v) = (rnd(t * nk * dk), rnd(t * nk * dk), rnd(t * nv * dv));
            let g: Vec<f32> = rnd(t * nv).iter().map(|x| -x.abs()).collect();
            let beta: Vec<f32> = rnd(t * nv).iter().map(|x| x + 0.5).collect();

            // Reference.
            let mut s_ref = vec![0f32; nv * dk * dv];
            let mut o_ref = vec![0f32; t * nv * dv];
            for tok in 0..t {
                for h in 0..nv {
                    let kh = match share {
                        KeyHeadShare::Grouped => h / (nv / nk),
                        KeyHeadShare::Tiled => h % nk,
                    };
                    let s = &mut s_ref[h * dk * dv..(h + 1) * dk * dv];
                    let kk = &k[(tok * nk + kh) * dk..][..dk];
                    let qq = &q[(tok * nk + kh) * dk..][..dk];
                    let vv = &v[(tok * nv + h) * dv..][..dv];
                    let e = g[tok * nv + h].exp();
                    s.iter_mut().for_each(|x| *x *= e);
                    let mut u = vec![0f32; dv];
                    for j in 0..dv {
                        let sk: f32 = (0..dk).map(|i| s[i * dv + j] * kk[i]).sum();
                        u[j] = beta[tok * nv + h] * (vv[j] - sk);
                    }
                    for i in 0..dk {
                        for j in 0..dv {
                            s[i * dv + j] += kk[i] * u[j];
                        }
                    }
                    for j in 0..dv {
                        o_ref[(tok * nv + h) * dv + j] =
                            (0..dk).map(|i| s[i * dv + j] * qq[i]).sum::<f32>() / (dk as f32).sqrt();
                    }
                }
            }

            // One call over the whole sequence…
            let mut s1 = vec![0f32; nv * dk * dv];
            let o1 = gated_delta_rule(&c, &q, &k, &v, &g, &beta, &mut s1, t);
            // …and two calls continuing the state.
            let mut s2 = vec![0f32; nv * dk * dv];
            let split = 2usize;
            let mut o2 = gated_delta_rule(
                &c,
                &q[..split * nk * dk],
                &k[..split * nk * dk],
                &v[..split * nv * dv],
                &g[..split * nv],
                &beta[..split * nv],
                &mut s2,
                split,
            );
            o2.extend(gated_delta_rule(
                &c,
                &q[split * nk * dk..],
                &k[split * nk * dk..],
                &v[split * nv * dv..],
                &g[split * nv..],
                &beta[split * nv..],
                &mut s2,
                t - split,
            ));
            for (name, got) in [("single", &o1), ("split", &o2)] {
                for (i, (a, b)) in got.iter().zip(&o_ref).enumerate() {
                    assert!((a - b).abs() < 1e-5, "{share:?} {name} output {i}: {a} vs {b}");
                }
            }
            for (a, b) in s1.iter().zip(&s_ref) {
                assert!((a - b).abs() < 1e-5, "{share:?} state diverges: {a} vs {b}");
            }
        }
    }

    /// `attention.key_length` must win over `embedding_length / head_count`
    /// (Qwen3-Coder: 128 vs 2048/32 = 64), and `norm_topk_prob` defaults to
    /// true when a conversion omits it.
    #[test]
    fn config_prefers_key_length_and_defaults_topk_norm() -> Result<()> {
        use std::collections::HashMap;
        let mut md = HashMap::new();
        let u = |v: u32| gguf_file::Value::U32(v);
        md.insert("qwen3moe.attention.head_count".into(), u(32));
        md.insert("qwen3moe.attention.head_count_kv".into(), u(4));
        md.insert("qwen3moe.block_count".into(), u(48));
        md.insert("qwen3moe.embedding_length".into(), u(2048));
        md.insert("qwen3moe.context_length".into(), u(262_144));
        md.insert(
            "qwen3moe.attention.layer_norm_rms_epsilon".into(),
            gguf_file::Value::F32(1e-6),
        );
        md.insert("qwen3moe.expert_count".into(), u(128));
        md.insert("qwen3moe.expert_used_count".into(), u(8));
        md.insert("qwen3moe.expert_feed_forward_length".into(), u(768));

        // No key_length, no norm_topk_prob: fallbacks apply.
        let cfg = Config::from_metadata(&md, "qwen3moe")?;
        assert_eq!(cfg.head_dim, 2048 / 32);
        assert!(cfg.expert_weights_norm, "norm_topk_prob must default to true");
        assert_eq!(cfg.n_expert, 128);
        assert_eq!(cfg.n_expert_used, 8);

        // Qwen3-Coder advertises key_length 128 explicitly.
        md.insert("qwen3moe.attention.key_length".into(), u(128));
        let cfg = Config::from_metadata(&md, "qwen3moe")?;
        assert_eq!(cfg.head_dim, 128);
        Ok(())
    }
}
