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
use std::io::{Read, Seek};
use std::sync::Arc;

use candle_core::quantized::{gguf_file, GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::ops::{sigmoid, silu, softmax, softmax_last_dim};
use candle_transformers::quantized_nn::RmsNorm;

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
        let rope_head_dim = m
            .u32_or(&format!("{a}.attention.rope.dimension_count"), 0)
            .max(1) as usize;
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
                    orig_context_length: m.u32_or(
                        &format!("{a}.rope.scaling.original_max_position_embeddings"),
                        65536,
                    ) as usize,
                    mscale_all_dim: m.f32_or(&format!("{a}.rope.scaling.yarn_log_multiplier"), 1.0),
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
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut gate = self.gate.forward(xs)?;
        let mut up = self.up.forward(xs)?;
        if self.clamp > 0.0 {
            gate = gate.clamp(f64::NEG_INFINITY, self.clamp)?;
            up = up.clamp(-self.clamp, self.clamp)?;
        }
        let h = (silu(&gate)? * up)?;
        self.down.forward(&h)
    }
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

    let pre = sigmoid(&((&m.narrow(D::Minus1, 0, hc)? * scale.i(0)?)? + base.narrow(0, 0, hc)?)?)?
        .affine(1.0, eps)?;
    let post =
        (sigmoid(&((&m.narrow(D::Minus1, hc, hc)? * scale.i(1)?)? + base.narrow(0, hc, hc)?)?)?
            * 2.0)?;
    let comb0 = (&m.narrow(D::Minus1, 2 * hc, hc * hc)? * scale.i(2)?)?.add(&base.narrow(
        0,
        2 * hc,
        hc * hc,
    )?)?;
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
    let mixes = hc_fn.forward(&flat)?.mul(&rsqrt)?; // [b*s, (2+hc)*hc]
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

/// Hadamard rotation applied to the trailing dim of a `[n, d]` tensor.
fn hadamard_rows(x: &Tensor) -> Result<Tensor> {
    let (n, d) = x.dims2()?;
    let data: Vec<f32> = x.to_vec1()?;
    let mut out = vec![0f32; data.len()];
    let scale = 1.0 / (d as f32).sqrt();
    for (row, chunk) in data.chunks_exact(d).enumerate() {
        let mut r = chunk.to_vec();
        fast_hadamard(&mut r);
        for (o, v) in out[row * d..row * d + d].iter_mut().zip(r.iter()) {
            *o = v * scale;
        }
    }
    Tensor::from_vec(out, (n, d), x.device())
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
    /// compresses whole blocks at once, decode accumulates per token.
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
                let bsc = (score
                    .narrow(0, 0, cutoff)?
                    .reshape((nb, ratio, coff * hd))?
                    + self.ape.unsqueeze(0)?)?;
                let (bkv, bsc) = if coff == 2 {
                    // Overlap transform: rows [0, ratio) come from the previous
                    // block's first half, rows [ratio, 2*ratio) from this block's
                    // second half.  Block 0 has no previous window (zero kv / -inf
                    // scores, so the zero row contributes nothing).
                    let own = bkv.narrow(2, hd, hd)?;
                    let own_s = bsc.narrow(2, hd, hd)?;
                    let prev = bkv.narrow(2, 0, hd)?;
                    let prev_s = bsc.narrow(2, 0, hd)?;
                    let zero = Tensor::zeros((1, ratio, hd), DType::F32, dev)?;
                    let zero_s = Tensor::full(f32::NEG_INFINITY, (1, ratio, hd), dev)?;
                    let prev = Tensor::cat(&[zero, prev.narrow(0, 0, nb - 1)?], 0)?;
                    let prev_s = Tensor::cat(&[zero_s, prev_s.narrow(0, 0, nb - 1)?], 0)?;
                    (
                        Tensor::cat(&[prev, own], 2)?,
                        Tensor::cat(&[prev_s, own_s], 2)?,
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
                        let kv_state = Tensor::cat(
                            &[
                                state.kv.narrow(0, 0, ratio)?.narrow(1, 0, hd)?,
                                state.kv.narrow(0, ratio, ratio)?.narrow(1, hd, hd)?,
                            ],
                            1,
                        )?;
                        let sc_state = Tensor::cat(
                            &[
                                state.score.narrow(0, 0, ratio)?.narrow(1, 0, hd)?,
                                state.score.narrow(0, ratio, ratio)?.narrow(1, hd, hd)?,
                            ],
                            1,
                        )?;
                        let w = softmax(&sc_state, 0)?;
                        rows.push((kv_state * w)?.sum(0)?); // [hd]
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
                        rows.push(state.kv.mul(&w)?.sum(0)?); // [hd]
                        poss.push((pos / ratio) as u32);
                    }
                }
            }
        }

        if rows.is_empty() {
            return Ok(None);
        }
        let mut comp = Tensor::stack(&rows, 0)?; // [n, hd]
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
            return Tensor::zeros((seq, 0), DType::U32, dev);
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
        let sc = (sc * w.unsqueeze(D::Minus1)?)?.sum(1)?; // [seq, n_lid]

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
    fn build_kv(
        &self,
        kv: &KvState,
        offset: usize,
        seq: usize,
        lid_idx: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let win = self.window_size;
        let ratio = self.ratio;
        let d = self.head_dim;
        let dev = kv.swa.as_ref().unwrap().device();

        // Window rows: token t of query p is valid iff
        // max(0, p+1-win) <= t <= p; the cache is a ring of size `win`.
        let mut w_rows = vec![0u32; seq * win];
        let mut w_mask = vec![f32::NEG_INFINITY; seq * win];
        for r in 0..seq {
            let p = offset + r;
            let lo = (p + 1).saturating_sub(win);
            let cnt = p + 1 - lo;
            for j in 0..cnt {
                w_rows[r * win + j] = ((lo + j) % win) as u32;
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

        let swa = kv.swa.as_ref().unwrap();
        let k_win = swa
            .index_select(&Tensor::from_vec(w_rows, (seq * win,), dev)?, 0)?
            .reshape((seq, win, d))?;
        let k_comp = match (kv.comp.as_ref(), n_comp) {
            (Some(c), n) if n > 0 => c
                .index_select(&Tensor::from_vec(c_rows, (seq * n_comp,), dev)?, 0)?
                .reshape((seq, n_comp, d))?,
            _ => Tensor::zeros((seq, 0, d), DType::F32, dev)?,
        };
        let k_all = Tensor::cat(&[k_win, k_comp], 1)?;
        let mask = Tensor::cat(
            &[
                Tensor::from_vec(w_mask, (seq, win), dev)?,
                Tensor::from_vec(c_mask, (seq, n_comp), dev)?,
            ],
            1,
        )?;
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

        // Write into the sliding-window ring.
        let ring: Vec<u32> = (offset..offset + seq)
            .map(|p| (p % self.window_size) as u32)
            .collect();
        let ridx = Tensor::from_vec(ring, (seq, 1), dev)?.broadcast_as((seq, self.head_dim))?;
        let swa = kv.swa.as_ref().unwrap();
        kv.swa = Some(swa.scatter(&ridx, &kv_raw, 0)?);

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
                let pidx = poss
                    .unsqueeze(1)?
                    .broadcast_as((rows.dim(0)?, self.head_dim))?;
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
                let pidx = poss
                    .unsqueeze(1)?
                    .broadcast_as((rows.dim(0)?, ix.head_dim))?;
                kv.lid = Some(lid.scatter(&pidx, rows, 0)?);
            }
            Some(ix.score(x, &qr, offset, seq, kv.lid.as_ref().unwrap(), &self.rotary)?)
        } else {
            None
        };

        // Sparse attention with per-head sink.
        let (k_all, mask) = self.build_kv(kv, offset, seq, lid_idx.as_ref())?;
        let scores = q.matmul(&k_all.transpose(1, 2)?)?; // [seq, h, n_kv]
        let scores = (scores * self.softmax_scale)?.add(&mask.unsqueeze(1)?)?;
        let max = scores.max_keepdim(D::Minus1)?;
        let e = (scores - &max)?.exp()?;
        let sink = self.attn_sinks.unsqueeze(0)?.unsqueeze(2)?; // [1, h, 1]
        let den = e.sum_keepdim(D::Minus1)?.add(&(sink - &max)?.exp()?)?; // [seq, h, 1]
        let num = e.unsqueeze(2)?.broadcast_matmul(&k_all.unsqueeze(1)?)?; // [seq, h, 1, n_kv] @ [seq, 1, n_kv, d] → [seq, h, 1, d]
        let o = num.broadcast_div(&den.unsqueeze(D::Minus1)?)?.squeeze(2)?; // [seq, h, d]

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
        self.wo_b.forward(&y)
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
        let max_blocks = max_seq / CSA_RATIO + 1;
        let swa = Tensor::zeros((win, hd), DType::F32, dev)?;
        let (comp, comp_state) = if ratio != 0 {
            let coff = if ratio == CSA_RATIO { 2 } else { 1 };
            (
                Some(Tensor::zeros((max_blocks, hd), DType::F32, dev)?),
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

struct Moe {
    gate: Tensor,              // [n_expert, n_embd] (f32)
    gate_bias: Option<Tensor>, // [n_expert]
    tid2eid: Option<Tensor>,   // [n_vocab, n_expert_used] (candle shape after gguf dim reversal)
    experts: Vec<Mlp>,
    shared: Option<Mlp>,
    n_expert_used: usize,
    weights_scale: f64,
    hash: bool,
}

impl Moe {
    fn forward(&self, xs: &Tensor, input_ids: &Tensor) -> Result<Tensor> {
        let (b, seq_len, h) = xs.dims3()?;
        let n_tokens = b * seq_len;
        let x2 = xs.reshape((n_tokens, h))?;

        let logits = x2.matmul(&self.gate.t()?.contiguous()?)?; // [n_tokens, n_expert]
                                                                // sqrt(softplus(x)) scoring; bias shifts selection only.
        let probs = softplus(&logits)?.sqrt()?;
        let (weights, indices) = if self.hash {
            // Hash layers: expert *selection* comes from tid2eid[token_id], but the
            // routing *weights* still come from the gate network's softplus scores
            // (official `Gate.forward` gathers `original_scores` at the hash ids).
            let ids = input_ids.flatten_all()?.to_vec1::<u32>()?;
            let tid = Tensor::from_vec(ids, n_tokens, xs.device())?;
            let idx = self
                .tid2eid
                .as_ref()
                .unwrap()
                .index_select(&tid, 1)?
                .to_dtype(DType::U32)?; // [n_tokens, k]
            let mut weights = probs.gather(&idx.to_dtype(DType::F32)?, D::Minus1)?;
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

        let routed = self.dispatch(&x2, &indices, &weights, n_tokens)?;
        let mut out = routed;
        if let Some(shared) = &self.shared {
            out = (out + shared.forward(&x2)?)?;
        }
        out.reshape((b, seq_len, h))
    }

    fn dispatch(
        &self,
        x2: &Tensor,
        indices: &Tensor,
        weights: &Tensor,
        n_tokens: usize,
    ) -> Result<Tensor> {
        let k = self.n_expert_used;
        let h = x2.dim(1)?;
        let ids: Vec<u32> = indices.flatten_all()?.to_vec1()?;
        let wts: Vec<f32> = weights.flatten_all()?.to_vec1()?;

        let mut per_expert: Vec<Vec<(u32, f32)>> = vec![Vec::new(); self.experts.len()];
        for t in 0..n_tokens {
            for s in 0..k {
                let e = ids[t * k + s] as usize;
                if e < self.experts.len() {
                    per_expert[e].push((t as u32, wts[t * k + s]));
                }
            }
        }

        let dev = x2.device();
        let mut y = Tensor::zeros((n_tokens, h), DType::F32, dev)?;
        for (e, bucket) in per_expert.iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            let token_idx: Vec<u32> = bucket.iter().map(|(t, _)| *t).collect();
            let w: Vec<f32> = bucket.iter().map(|(_, w)| *w).collect();
            let count = token_idx.len();
            let idx = Tensor::from_vec(token_idx, count, dev)?;
            let x_sel = x2.index_select(&idx, 0)?;
            let out = self.experts[e].forward(&x_sel)?;
            let w = Tensor::from_vec(w, (count, 1), dev)?;
            y = y.index_add(&idx, &out.broadcast_mul(&w)?, 0)?;
        }
        Ok(y)
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
    fn forward(&self, xs: &Tensor, input_ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Moe(m) => m.forward(xs, input_ids),
        }
    }
}

/// A quantized DeepSeek-V4 model loaded from GGUF.
pub struct ModelWeights {
    tok_embeddings: Tensor,
    layers: Vec<Layer>,
    kv: Vec<KvState>,
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
}

/// Small GGUF reader over the memory-mapped file.
struct Reader<R: Read + Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
    mmap: Option<std::sync::Arc<memmap2::Mmap>>,
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
    fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        QMatMul::from_qtensor(self.qtensor(name)?)
    }
    fn qmatmul_opt(&mut self, name: &str) -> Option<QMatMul> {
        if self.has(name) {
            self.qmatmul(name).ok()
        } else {
            None
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
        self.ct.tensor_infos.contains_key(name)
    }
}

impl ModelWeights {
    /// Load a `deepseek4` GGUF.
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        Self::from_gguf_mmap(ct, reader, device, None)
    }

    /// Load with weights borrowed in place from `mmap` where possible.
    pub fn from_gguf_mmap<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        mmap: Option<std::sync::Arc<memmap2::Mmap>>,
    ) -> Result<Self> {
        let cfg = Config::from_metadata(&ct.metadata)?;
        let mut rd = Reader {
            ct,
            reader,
            device: device.clone(),
            mmap,
        };

        let tok_embeddings = rd.f32_tensor("token_embd.weight")?;
        let norm = rd.rms_norm("output_norm.weight", cfg.rms_eps)?;
        let output = match rd.qmatmul_opt("output.weight") {
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
            let qa_shape = rd
                .qtensor(&format!("{p}.attn_q_a.weight"))?
                .shape()
                .dims()
                .to_vec();
            debug_assert_eq!(
                vec![cfg.n_head * cfg.head_dim, cfg.q_lora_rank],
                qa_shape,
                "blk.{i}.attn_q_a"
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
        // config does not silently reserve ~20 GB of CPU RAM per instance.
        // The cap is a hard limit enforced in `forward` (clear error if hit).
        let kv_cap = cfg.context_length.min(KV_CAP);
        let mut kv_states = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            kv_states.push(KvState::new(&cfg, i, device, kv_cap)?);
        }
        let hc_mult = cfg.hc_mult;
        let hc_eps = cfg.hc_eps;

        Ok(Self {
            tok_embeddings,
            layers,
            kv: kv_states,
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
        })
    }

    /// Forward pass. `input` is `[1, seq_len]`; `offset` is the KV-cache
    /// position of the first input token.
    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_b, seq_len) = input.dims2()?;
        let hc = self.hc_mult;
        let d = self.tok_embeddings.dim(1)?;

        let tok = self
            .tok_embeddings
            .index_select(&input.flatten_all()?, 0)?
            .reshape((1, seq_len, d))?;
        // Expand to hc copies.
        let mut xs = tok.unsqueeze(2)?.broadcast_as((1, seq_len, hc, d))?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            // hc_pre with attention weights
            let (x, post, comb) = hc_pre(
                &xs,
                &layer.hc_attn_fn,
                &layer.hc_attn_scale,
                &layer.hc_attn_base,
                self.hc_eps,
                self.cfg.hc_sinkhorn_iters,
            )?;
            let residual = xs;
            let h = layer.attn_norm.forward(&x)?;
            let h = layer
                .attn
                .forward(&mut self.kv[i], &h, offset, self.max_seq)?;
            xs = hc_post(&h, &residual, &post, &comb)?;

            let (x, post, comb) = hc_pre(
                &xs,
                &layer.hc_ffn_fn,
                &layer.hc_ffn_scale,
                &layer.hc_ffn_base,
                self.hc_eps,
                self.cfg.hc_sinkhorn_iters,
            )?;
            let residual = xs;
            let h = layer.ffn_norm.forward(&x)?;
            let h = layer.ffn.forward(&h, input)?;
            xs = hc_post(&h, &residual, &post, &comb)?;
        }

        // Parallel head: collapse hc copies, RMS, output matmul.
        // pre = sigmoid(mixes * hc_head_scale + hc_head_base) + eps
        let flat = xs.reshape((seq_len, hc * d))?;
        let rsqrt = flat
            .sqr()?
            .mean_keepdim(D::Minus1)?
            .affine(1.0, self.hc_eps)?
            .powf(-0.5)?;
        let mixes = self.hc_head_fn.forward(&flat)?.mul(&rsqrt)?; // [s, hc]
        let pre = sigmoid(
            &mixes
                .broadcast_mul(&self.hc_head_scale)?
                .broadcast_add(&self.hc_head_base)?,
        )?
        .affine(1.0, self.hc_eps)?; // + eps
        let y = pre
            .unsqueeze(D::Minus1)?
            .broadcast_as((seq_len, hc, d))?
            .mul(&xs.squeeze(0)?)?
            .sum(D::Minus2)?; // [s, d]
        let y = self.norm.forward(&y)?;
        let logits = self.output.forward(&y)?; // [s, n_vocab]
        logits.to_dtype(DType::F32)
    }

    /// Reset the KV caches so this instance can serve an unrelated prompt.
    pub fn clear_kv_cache(&mut self) {
        let dev = self.device.clone();
        for (i, kv) in self.kv.iter_mut().enumerate() {
            match KvState::new(&self.cfg, i, &dev, self.max_seq) {
                Ok(n) => *kv = n,
                Err(e) => eprintln!("deepseek4: failed to reset KV state for layer {i}: {e}"),
            }
        }
    }
}

fn load_moe<R: Read + Seek>(
    rd: &mut Reader<R>,
    p: &str,
    cfg: &Config,
    layer: usize,
    hash: bool,
) -> Result<Moe> {
    let gate = rd.f32_tensor(&format!("{p}.ffn_gate_inp.weight"))?;
    let gate_bias = if hash {
        None
    } else {
        rd.has(&format!("{p}.exp_probs_b.bias"))
            .then(|| rd.f32_tensor(&format!("{p}.exp_probs_b.bias")))
            .transpose()?
    };
    let tid2eid = if hash {
        rd.has(&format!("{p}.ffn_gate_tid2eid.weight"))
            .then(|| rd.f32_tensor(&format!("{p}.ffn_gate_tid2eid.weight")))
            .transpose()?
    } else {
        None
    };

    let gate_exps = split_experts(rd, &format!("{p}.ffn_gate_exps.weight"), cfg.n_expert)?;
    let up_exps = split_experts(rd, &format!("{p}.ffn_up_exps.weight"), cfg.n_expert)?;
    let down_exps = split_experts(rd, &format!("{p}.ffn_down_exps.weight"), cfg.n_expert)?;
    let clamp = cfg.swiglu_clamp.get(layer).copied().unwrap_or(0.0);
    let experts = gate_exps
        .into_iter()
        .zip(up_exps)
        .zip(down_exps)
        .map(|((gate, up), down)| Mlp {
            gate,
            up,
            down,
            clamp,
        })
        .collect();

    let shared = if cfg.n_expert_shared > 0 {
        Some(Mlp {
            gate: rd.qmatmul(&format!("{p}.ffn_gate_shexp.weight"))?,
            up: rd.qmatmul(&format!("{p}.ffn_up_shexp.weight"))?,
            down: rd.qmatmul(&format!("{p}.ffn_down_shexp.weight"))?,
            clamp: cfg.swiglu_clamp_shexp.get(layer).copied().unwrap_or(0.0),
        })
    } else {
        None
    };

    Ok(Moe {
        gate,
        gate_bias,
        tid2eid,
        experts,
        shared,
        n_expert_used: cfg.n_expert_used,
        weights_scale: cfg.expert_weights_scale,
        hash,
    })
}

/// Split a 3-D expert weight `[n_expert, out, in]` into `n_expert` quantized
/// 2-D `QMatMul`s by carving its raw quantized byte-buffer.
fn split_experts<R: Read + Seek>(
    rd: &mut Reader<R>,
    name: &str,
    n_expert: usize,
) -> Result<Vec<QMatMul>> {
    let qt = rd.qtensor(name)?;
    let dims = qt.shape().dims().to_vec();
    if dims.len() != 3 || dims[0] != n_expert {
        candle_core::bail!(
            "deepseek4: expected expert tensor `{name}` shaped [n_expert, out, in], got {dims:?}"
        );
    }
    let (out, inn) = (dims[1], dims[2]);
    let dtype: GgmlDType = qt.dtype();

    if let Some(mmap) = rd.mmap.clone().filter(|_| rd.device.is_cpu()) {
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
                        Some(t) => borrowed.push(QMatMul::from_qtensor(t)?),
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

    let bytes = qt.data()?;
    if bytes.len() % n_expert != 0 {
        candle_core::bail!(
            "deepseek4: expert tensor `{name}` byte length {} not divisible by n_expert {n_expert}",
            bytes.len()
        );
    }
    let per = bytes.len() / n_expert;
    let mut experts = Vec::with_capacity(n_expert);
    for e in 0..n_expert {
        let slice = &bytes[e * per..(e + 1) * per];
        let storage = QStorage::from_data(Cow::Borrowed(slice), &rd.device, dtype)?;
        let qt = QTensor::new(storage, (out, inn))?;
        experts.push(QMatMul::from_qtensor(qt)?);
    }
    Ok(experts)
}
