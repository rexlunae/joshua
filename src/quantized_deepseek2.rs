//! Pure-Rust quantized loader for the `deepseek2` GGUF architecture.
//!
//! Covers DeepSeek-V2, DeepSeek-V2-Lite, DeepSeek-V3 and **Kimi-K2** — every
//! model llama.cpp labels `general.architecture = "deepseek2"`.  candle ships a
//! *full-precision* `deepseek2` model but no quantized/GGUF one, and its gate
//! only implements DeepSeek-V2 softmax routing; this module adds the GGUF path
//! plus the DeepSeek-V3 / Kimi-K2 sigmoid-with-bias, group-limited routing.
//!
//! Two things make this architecture unusual:
//!
//! * **MLA (Multi-head Latent Attention).**  Q and KV are produced through
//!   low-rank projections; only a small `qk_rope_head_dim` slice of each head
//!   carries RoPE, the rest ("nope") is un-rotated.  We implement the
//!   *unabsorbed* (full-MHA) form, reconstructing per-head K/V from the
//!   compressed latent — numerically identical to llama.cpp and matching
//!   candle's reference `deepseek2` math.  Modern GGUFs that pre-split the KV
//!   up-projection into `attn_k_b`/`attn_v_b` are supported by folding those
//!   back into the combined projection at load.
//!
//! * **Fine-grained MoE.**  Most layers route each token to a few of many
//!   experts, with a handful of always-on shared experts.  DeepSeek-V3 / Kimi
//!   add a per-expert selection bias (aux-loss-free balancing) and group-limited
//!   top-k.  Experts stay **quantized**: the 3-D expert tensor is sliced into
//!   per-expert [`QMatMul`]s from its raw quantized bytes, so a 1 T-parameter
//!   MoE keeps its on-disk footprint instead of exploding to f32 in RAM.
//!
//! Activations run in f32 for CPU accuracy, mirroring the other Joshua
//! quantized loaders (`glm4`, `qwen3moe`).

use std::borrow::Cow;
use std::io::{Read, Seek};
use std::sync::Arc;

use candle_core::quantized::{gguf_file, GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::ops::{sigmoid, softmax_last_dim};
use candle_transformers::quantized_nn::RmsNorm;

/// Expert gating (scoring) function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gating {
    /// DeepSeek-V2 / V2.5: softmax over the router logits.
    Softmax,
    /// DeepSeek-V3 / Kimi-K2: independent sigmoid per expert.
    Sigmoid,
}

/// Parsed `deepseek2` hyper-parameters.
struct Config {
    n_layer: usize,
    n_head: usize,
    rms_eps: f64,
    // MLA dims.
    q_lora_rank: Option<usize>,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    softmax_scale: f64,
    // RoPE.
    rope_theta: f32,
    context_length: usize,
    yarn: Option<YarnConfig>,
    // FFN / MoE.
    leading_dense: usize,
    n_expert: usize,
    n_expert_used: usize,
    n_expert_shared: usize,
    expert_weights_scale: f64,
    expert_weights_norm: bool,
    gating: Gating,
    n_group: usize,
    topk_group: usize,
}

impl Config {
    /// Per-head Q/K dimension (`qk_nope_head_dim + qk_rope_head_dim`).
    fn q_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
}

struct YarnConfig {
    factor: f32,
    orig_context_length: usize,
    /// `mscale_all_dim` recovered from `rope.scaling.yarn_log_multiplier`.
    mscale_all_dim: f32,
}

// ─── Metadata helpers ───────────────────────────────────────────────────────

struct Meta<'a>(&'a std::collections::HashMap<String, gguf_file::Value>);

impl Meta<'_> {
    fn u32(&self, key: &str) -> Result<u32> {
        match self.0.get(key) {
            Some(v) => v.to_u32(),
            None => candle_core::bail!("deepseek2: missing GGUF metadata key `{key}`"),
        }
    }
    fn u32_or(&self, key: &str, default: u32) -> u32 {
        self.0.get(key).and_then(|v| v.to_u32().ok()).unwrap_or(default)
    }
    fn f32(&self, key: &str) -> Result<f32> {
        match self.0.get(key) {
            Some(v) => v.to_f32(),
            None => candle_core::bail!("deepseek2: missing GGUF metadata key `{key}`"),
        }
    }
    fn f32_or(&self, key: &str, default: f32) -> f32 {
        self.0.get(key).and_then(|v| v.to_f32().ok()).unwrap_or(default)
    }
    fn bool_or(&self, key: &str, default: bool) -> bool {
        self.0.get(key).and_then(|v| v.to_bool().ok()).unwrap_or(default)
    }
    fn string(&self, key: &str) -> Option<String> {
        self.0.get(key).and_then(|v| v.to_string().ok().cloned())
    }
}

impl Config {
    fn from_metadata(md: &std::collections::HashMap<String, gguf_file::Value>) -> Result<Self> {
        let m = Meta(md);
        let a = "deepseek2";
        let n_head = m.u32(&format!("{a}.attention.head_count"))? as usize;
        let n_embd = m.u32(&format!("{a}.embedding_length"))? as usize;
        let n_layer = m.u32(&format!("{a}.block_count"))? as usize;
        let rms_eps = m.f32(&format!("{a}.attention.layer_norm_rms_epsilon"))? as f64;

        // Head dims. `is_mla` (pre-split k_b/v_b) advertises key_length_mla;
        // otherwise fall back to key_length / value_length (default n_embd/n_head).
        let key_length_mla = m.u32_or(&format!("{a}.attention.key_length_mla"), 0) as usize;
        let value_length_mla = m.u32_or(&format!("{a}.attention.value_length_mla"), 0) as usize;
        let is_mla = key_length_mla != 0 && value_length_mla != 0;
        let n_embd_head_k = if is_mla {
            key_length_mla
        } else {
            m.u32_or(&format!("{a}.attention.key_length"), (n_embd / n_head) as u32) as usize
        };
        let v_head_dim = if is_mla {
            value_length_mla
        } else {
            m.u32_or(&format!("{a}.attention.value_length"), (n_embd / n_head) as u32) as usize
        };
        let qk_rope_head_dim =
            m.u32_or(&format!("{a}.rope.dimension_count"), n_embd_head_k as u32) as usize;
        let qk_nope_head_dim = n_embd_head_k.saturating_sub(qk_rope_head_dim);

        let q_lora_rank = m
            .0
            .get(&format!("{a}.attention.q_lora_rank"))
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .filter(|&v| v > 0);
        let kv_lora_rank = m.u32(&format!("{a}.attention.kv_lora_rank"))? as usize;

        let rope_theta = m.f32_or(&format!("{a}.rope.freq_base"), 10_000.0);
        let context_length = m.u32(&format!("{a}.context_length"))? as usize;

        // YaRN long-context scaling (optional).
        let scaling_type = m.string(&format!("{a}.rope.scaling.type"));
        let yarn = if scaling_type.as_deref() == Some("yarn") {
            let factor = m.f32_or(&format!("{a}.rope.scaling.factor"), 1.0);
            let orig_context_length = m.u32_or(
                &format!("{a}.rope.scaling.original_context_length"),
                context_length as u32,
            ) as usize;
            // llama.cpp stores 0.1 * mscale_all_dim and divides it back out.
            let log_mul = m.f32_or(&format!("{a}.rope.scaling.yarn_log_multiplier"), 0.0);
            Some(YarnConfig {
                factor,
                orig_context_length,
                mscale_all_dim: log_mul / 0.1,
            })
        } else {
            None
        };

        // Softmax scale: 1/sqrt(q_head_dim), YaRN-corrected by mscale².
        let q_head_dim = qk_nope_head_dim + qk_rope_head_dim;
        let mut softmax_scale = 1.0f64 / (q_head_dim as f64).sqrt();
        if let Some(y) = &yarn {
            let mscale = yarn_get_mscale(y.factor, y.mscale_all_dim) as f64;
            softmax_scale *= mscale * mscale;
        }

        let leading_dense = m.u32_or(&format!("{a}.leading_dense_block_count"), 0) as usize;
        let n_expert = m.u32_or(&format!("{a}.expert_count"), 0) as usize;
        let n_expert_used = m.u32_or(&format!("{a}.expert_used_count"), 0) as usize;
        let n_expert_shared = m.u32_or(&format!("{a}.expert_shared_count"), 0) as usize;
        let expert_weights_scale =
            m.f32_or(&format!("{a}.expert_weights_scale"), 0.0) as f64;
        let expert_weights_norm = m.bool_or(&format!("{a}.expert_weights_norm"), false);
        // Gating: 1=softmax, 2=sigmoid. Absent → softmax (DeepSeek-V2).
        let gating = match m.u32_or(&format!("{a}.expert_gating_func"), 1) {
            2 => Gating::Sigmoid,
            _ => Gating::Softmax,
        };
        let n_group = m.u32_or(&format!("{a}.expert_group_count"), 0) as usize;
        let topk_group = m.u32_or(&format!("{a}.expert_group_used_count"), 0) as usize;

        Ok(Self {
            n_layer,
            n_head,
            rms_eps,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            softmax_scale,
            rope_theta,
            context_length,
            yarn,
            leading_dense,
            n_expert,
            n_expert_used,
            n_expert_shared,
            expert_weights_scale,
            expert_weights_norm,
            gating,
            n_group,
            topk_group,
        })
    }
}

/// YaRN attention/temperature scale: `0.1 * mscale * ln(scale) + 1` for
/// `scale > 1`, else 1.
fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

// ─── RoPE (YaRN-aware, applied to the qk_rope slice only) ────────────────────

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.qk_rope_head_dim;
        let max_seq = cfg.context_length.max(1);
        let theta = cfg.rope_theta;
        match &cfg.yarn {
            None => {
                let inv_freq: Vec<f32> = (0..dim)
                    .step_by(2)
                    .map(|i| 1f32 / theta.powf(i as f32 / dim as f32))
                    .collect();
                Self::from_inv_freq(inv_freq, max_seq, 1.0, dev)
            }
            Some(y) => {
                // Interpolated vs extrapolated frequencies blended by a ramp
                // over the YaRN correction range (see DeepSeek modeling code).
                let half = dim / 2;
                let freq_extra: Vec<f32> = (0..dim)
                    .step_by(2)
                    .map(|i| 1f32 / theta.powf(i as f32 / dim as f32))
                    .collect();
                let freq_inter: Vec<f32> = (0..dim)
                    .step_by(2)
                    .map(|i| 1f32 / (y.factor * theta.powf(i as f32 / dim as f32)))
                    .collect();
                let (low, high) = yarn_correction_range(
                    32.0,
                    1.0,
                    dim,
                    theta,
                    y.orig_context_length,
                );
                let ramp = yarn_linear_ramp(low, high, half);
                let inv_freq: Vec<f32> = (0..half)
                    .map(|i| {
                        let mask = 1.0 - ramp[i];
                        freq_inter[i] * (1.0 - mask) + freq_extra[i] * mask
                    })
                    .collect();
                let mscale = yarn_get_mscale(y.factor, y.mscale_all_dim);
                Self::from_inv_freq(inv_freq, max_seq, mscale, dev)
            }
        }
    }

    fn from_inv_freq(inv_freq: Vec<f32>, max_seq: usize, mscale: f32, dev: &Device) -> Result<Self> {
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

    /// Apply interleaved RoPE (llama.cpp `NORM` type) to `q`/`k`, each shaped
    /// `[b, heads, seq, qk_rope_head_dim]`.
    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let q = candle_nn::rotary_emb::rope_i(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope_i(&k.contiguous()?, &cos, &sin)?;
        Ok((q, k))
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

// ─── Linear helpers ─────────────────────────────────────────────────────────

/// SwiGLU MLP over quantized weights (dense layers and shared experts).
struct Mlp {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
    /// Per-tensor byte-range handles for best-effort page prefetch, present
    /// only when the weights are borrowed from the model mapping.
    prefetch: Option<crate::residency::ExpertHandles>,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate.forward(xs)?)?;
        let up = self.up.forward(xs)?;
        self.down.forward(&(gate * up)?)
    }

    /// Ask the kernel to prefetch this expert's weight pages (best effort;
    /// no-op when the weights are not mmap-backed).
    fn prefetch(&self) {
        if let Some(p) = &self.prefetch {
            p.gate.prefetch();
            p.up.prefetch();
            p.down.prefetch();
        }
    }
}

/// The Q projection: a plain linear (V2-Lite) or a LoRA a→norm→b stack.
enum QProj {
    Plain(QMatMul),
    Lora { a: QMatMul, norm: RmsNorm, b: QMatMul },
}

impl QProj {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Plain(l) => l.forward(xs),
            Self::Lora { a, norm, b } => b.forward(&norm.forward(&a.forward(xs)?)?),
        }
    }
}

// ─── Attention (MLA, unabsorbed / full-MHA form) ─────────────────────────────

struct Attention {
    q: QProj,
    kv_a_mqa: QMatMul,
    kv_a_norm: RmsNorm,
    /// Combined KV up-projection: kv_lora_rank → n_head*(qk_nope + v_head_dim).
    kv_b: KvB,
    o_proj: QMatMul,
    rotary: Arc<RotaryEmbedding>,
    n_head: usize,
    kv_lora_rank: usize,
    qk_nope: usize,
    qk_rope: usize,
    v_head_dim: usize,
    q_head_dim: usize,
    softmax_scale: f64,
    /// MLA cache: reconstructed per-head K/V kept transposed —
    /// `(k [b, n_head, q_head_dim, seq], v [b, n_head, v_head_dim, seq])` —
    /// and appended incrementally.
    ///
    /// The compressed latent is only `kv_lora_rank + qk_rope` ≈ 576 elems/token
    /// vs `n_head·(qk_nope + v_head_dim)` ≈ 40,960 for the full per-head K/V, so
    /// caching the latent saves ~70x memory — but reconstructing the full K/V
    /// from it every forward is O(seq) work per step, which makes decode
    /// degrade linearly with context length.  Instead we reconstruct K/V for
    /// the *new* tokens only (a linear map over the latent) and append the
    /// per-head result to the cache: decode stays O(1) in reconstruction, at
    /// the cost of caching the ~8x larger per-head form (still far below a
    /// plain MHA model, since MLA keeps Q and the latent low-rank).
    kv_cache: Option<(Tensor, Tensor)>,
}

/// The KV up-projection, either a native combined weight or one reconstructed
/// from the pre-split MLA `attn_k_b`/`attn_v_b` tensors.
enum KvB {
    Quantized(QMatMul),
    Dense(Tensor),
}

impl KvB {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Quantized(q) => q.forward(xs),
            // xs: [..., kv_lora_rank] · W^T where W is [out, kv_lora_rank].
            Self::Dense(w) => xs.broadcast_matmul(&w.t()?),
        }
    }
}

impl Attention {
    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;

        // Q → [b, n_head, seq, q_head_dim], split into nope/rope slices.
        let q = self
            .q
            .forward(xs)?
            .reshape((b, seq_len, self.n_head, self.q_head_dim))?
            .transpose(1, 2)?;
        let q_nope = q.narrow(D::Minus1, 0, self.qk_nope)?;
        let q_pe = q.narrow(D::Minus1, self.qk_nope, self.qk_rope)?;

        // Compressed KV: [b, seq, kv_lora_rank + qk_rope]; the trailing slice is
        // a single-head RoPE key shared (MQA-style) across all query heads.
        let compressed = self.kv_a_mqa.forward(xs)?;
        let kv_cmpr = compressed
            .narrow(D::Minus1, 0, self.kv_lora_rank)?
            .contiguous()?;
        let k_pe = compressed
            .narrow(D::Minus1, self.kv_lora_rank, self.qk_rope)?
            .reshape((b, seq_len, 1, self.qk_rope))?
            .transpose(1, 2)?;

        // RoPE the *_pe slices, then reassemble Q (nope ‖ rope).  The single-head
        // key is RoPE'd here too; the cached copy below is already post-RoPE.
        let (q_pe, k_pe) = self.rotary.apply(&q_pe, &k_pe, offset)?;
        let q = Tensor::cat(&[&q_nope.contiguous()?, &q_pe.contiguous()?], D::Minus1)?;

        // Reconstruct per-head K/V for the *new* tokens only and append it to
        // the cache.  kv_a_norm is a per-row norm and kv_b a linear map, so
        // applying them to this step's latent is bit-identical to the old
        // whole-cache reconstruction — just O(seq_len) instead of O(seq_total)
        // per step, which is what made decode slow down with context length.
        let kv = self
            .kv_b
            .forward(&self.kv_a_norm.forward(&kv_cmpr)?)?
            .reshape((b, seq_len, self.n_head, self.qk_nope + self.v_head_dim))?
            .transpose(1, 2)?; // [b, n_head, seq_len, qk_nope + v_head_dim]
        let k_nope = kv.narrow(D::Minus1, 0, self.qk_nope)?; // [b, n_head, seq_len, qk_nope]
        let v = kv
            .narrow(D::Minus1, self.qk_nope, self.v_head_dim)?
            .contiguous()?; // [b, n_head, seq_len, v_head_dim]
        // The single-head RoPE'd key is shared (MQA-style) across query heads.
        let k_pe = k_pe.broadcast_as((b, self.n_head, seq_len, self.qk_rope))?;
        let k_new = Tensor::cat(&[&k_nope.contiguous()?, &k_pe], D::Minus1)?; // [b, n_head, seq_len, q_head_dim]

        // The cache lives transposed — [b, n_head, dim, seq] — so the attention
        // matmuls below run directly on it: `q · kᵀ` becomes `q · k_cache` and
        // `probs · v` becomes `v_cache · probsᵀ`.  A decode step then copies
        // only the two append cats (the accumulated context), never an extra
        // transpose of it.
        let k_new = k_new.transpose(2, 3)?.contiguous()?; // [b, n_head, q_head_dim, seq_len]
        let v_new = v.transpose(2, 3)?.contiguous()?; // [b, n_head, v_head_dim, seq_len]
        let (k_cache, v_cache) = match &self.kv_cache {
            None => (k_new, v_new),
            Some((kc, vc)) => (
                Tensor::cat(&[kc, &k_new], 3)?.contiguous()?,
                Tensor::cat(&[vc, &v_new], 3)?.contiguous()?,
            ),
        };
        self.kv_cache = Some((k_cache.clone(), v_cache.clone()));

        // Scaled dot-product attention over the whole cached context.
        let scores = (q.contiguous()?.matmul(&k_cache)? * self.softmax_scale)?; // [b, n_head, seq_len, seq_total]
        let scores = match mask {
            Some(m) => scores.broadcast_add(m)?,
            None => scores,
        };
        let probs = softmax_last_dim(&scores)?;
        let ctx = v_cache.matmul(&probs.transpose(2, 3)?.contiguous()?)?; // [b, n_head, v_head_dim, seq_len]
        let ctx = ctx
            .transpose(1, 3)?
            .transpose(2, 3)?
            .contiguous()?
            .reshape((b, seq_len, self.n_head * self.v_head_dim))?;
        self.o_proj.forward(&ctx)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }

    /// Keep only the first `keep` sequence positions of the KV cache.
    ///
    /// Edited-context prefix reuse: positions `[0..keep)` were produced by
    /// exactly the tokens both conversations share, so they remain valid;
    /// everything from `keep` on is dropped and recomputed by the next
    /// prefill (which continues at absolute position `keep`).  The prefix is
    /// materialised (`contiguous`) so the full-length buffer is actually
    /// freed instead of lingering behind a view until the next append.
    fn truncate_kv_cache(&mut self, keep: usize) -> Result<()> {
        match &mut self.kv_cache {
            None => Ok(()),
            Some((_k, _v)) if keep == 0 => {
                self.kv_cache = None;
                Ok(())
            }
            Some((k, v)) => {
                // The cache lives transposed — [b, n_head, dim, seq] — so
                // the sequence dimension is dim 3 here (see `forward`).
                let len = k.dim(3)?;
                if keep >= len {
                    return Ok(());
                }
                let k = k.narrow(3, 0, keep)?.contiguous()?;
                let v = v.narrow(3, 0, keep)?.contiguous()?;
                self.kv_cache = Some((k, v));
                Ok(())
            }
        }
    }
}

// ─── Mixture of experts (DeepSeek routing + shared experts) ──────────────────

struct Moe {
    gate_t: Tensor,        // router weight transposed to [n_embd, n_expert], contiguous (cached)
    gate_bias: Option<Tensor>, // exp_probs_b [n_expert] (f32), V3/K2 only
    experts: Vec<Mlp>,     // per-expert quantized SwiGLU
    shared: Option<Mlp>,
    gating: Gating,
    n_expert_used: usize,
    n_group: usize,
    topk_group: usize,
    weights_norm: bool,
    weights_scale: f64,
}

impl Moe {
    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Vec<u32>)> {
        let (b, seq_len, h) = xs.dims3()?;
        let n_tokens = b * seq_len;
        let x2 = xs.reshape((n_tokens, h))?;

        // Router logits → per-expert scores.
        let logits = x2.matmul(&self.gate_t)?; // [n_tokens, n_expert]
        let probs = match self.gating {
            Gating::Softmax => softmax_last_dim(&logits)?,
            Gating::Sigmoid => sigmoid(&logits)?,
        };

        // Selection scores add the aux-loss-free bias; final weights use the
        // *unbiased* probs.
        let selection = match &self.gate_bias {
            Some(bias) => probs.broadcast_add(&bias.reshape((1, ()))?)?,
            None => probs.clone(),
        };
        let selection = self.group_limit(&selection, n_tokens)?;

        let topk_idx = topk_indices(&selection, self.n_expert_used)?; // [n_tokens, k]
        let mut weights = probs.gather(&topk_idx, D::Minus1)?; // [n_tokens, k]
        if self.weights_norm {
            // Min-clamp to the f16 epsilon, matching llama.cpp.
            let denom = weights.sum_keepdim(D::Minus1)?.clamp(6.103_515_6e-5, f32::INFINITY)?;
            weights = weights.broadcast_div(&denom)?;
        }
        if self.weights_scale != 0.0 && self.weights_scale != 1.0 {
            weights = (weights * self.weights_scale)?;
        }

        // dispatch already drains the routed ids to the host for its own
        // bucketing; reuse them for the hot-expert cache instead of a second
        // device-to-host sync (which would cost a synchronization per layer
        // even when the cache is disabled).
        let (routed, ids) = self.dispatch(&x2, &topk_idx, &weights, n_tokens)?;
        let mut out = routed;
        if let Some(shared) = &self.shared {
            out = (out + shared.forward(&x2)?)?;
        }
        Ok((out.reshape((b, seq_len, h))?, ids))
    }

    /// Zero out experts outside the top `topk_group` groups (scored by the sum
    /// of their two best experts). No-op unless `n_group > 1`.
    fn group_limit(&self, selection: &Tensor, n_tokens: usize) -> Result<Tensor> {
        if self.n_group <= 1 {
            return Ok(selection.clone());
        }
        let n_expert = selection.dim(D::Minus1)?;
        let per = n_expert / self.n_group;
        let grouped = selection.reshape((n_tokens, self.n_group, per))?;
        // Score each group, then keep the best `topk_group` groups. The scoring
        // rule differs by model variant (both give [n_tokens, n_group]):
        //   * DeepSeek-V3 / Kimi-K2 (sigmoid, "noaux_tc"): sum of the group's
        //     top-2 experts — matches llama.cpp's build_moe_ffn.
        //   * DeepSeek-V2 (softmax, "group_limited_greedy"): the single best
        //     expert in the group — matches HF modeling_deepseek.py and
        //     candle's reference. (llama.cpp applies the V3 sum rule here too,
        //     so this path intentionally follows the model definition, not
        //     llama.cpp.)
        let group_score = match self.gating {
            Gating::Sigmoid => topk_values(&grouped, 2)?.sum(D::Minus1)?,
            Gating::Softmax => grouped.max(D::Minus1)?,
        };
        let group_idx = topk_indices(&group_score, self.topk_group)?; // [n_tokens, topk_group]
        // Mask: 1.0 for selected groups.
        let ones = group_idx.ones_like()?.to_dtype(DType::F32)?;
        let group_mask = Tensor::zeros((n_tokens, self.n_group), DType::F32, selection.device())?
            .scatter_add(&group_idx, &ones, 1)?;
        let expert_mask = group_mask
            .reshape((n_tokens, self.n_group, 1))?
            .broadcast_as((n_tokens, self.n_group, per))?
            .reshape((n_tokens, n_expert))?
            .contiguous()?;
        // Add a large negative penalty to experts outside the selected groups
        // (mask 0 → −1e30, mask 1 → 0) so they never survive the top-k.
        let penalty = expert_mask.affine(1e30, -1e30)?;
        selection.add(&penalty)
    }

    /// Run each selected expert over its routed tokens and accumulate the
    /// weighted outputs. Experts stay quantized.
    ///
    /// Prefill (`n_tokens > 1`) buckets tokens per expert and runs one batched
    /// matmul per expert.  Decode (`n_tokens == 1`) takes a separate path that
    /// avoids every per-expert host round-trip: no `to_vec1` drain on the
    /// weights, no `Tensor::from_vec` uploads, no `index_select`/`index_add`.
    /// Each single-token layer forward then costs one host drain (the expert
    /// ids, needed to pick weights) plus `n_expert_used ×` the expert MLPs and
    /// a stack–scale–sum, instead of per-expert uploads and gathers.
    fn dispatch(
        &self,
        x2: &Tensor,
        topk_idx: &Tensor,
        weights: &Tensor,
        n_tokens: usize,
    ) -> Result<(Tensor, Vec<u32>)> {
        if n_tokens == 1 {
            return self.dispatch_decode(x2, topk_idx, weights);
        }
        let k = self.n_expert_used;
        let h = x2.dim(1)?;
        let ids: Vec<u32> = topk_idx.flatten_all()?.to_vec1()?;
        let wts: Vec<f32> = weights.flatten_all()?.to_vec1()?;

        // Bucket (token, weight) pairs by expert.
        let mut per_expert: Vec<Vec<(u32, f32)>> = vec![Vec::new(); self.experts.len()];
        for t in 0..n_tokens {
            for s in 0..k {
                let e = ids[t * k + s] as usize;
                per_expert[e].push((t as u32, wts[t * k + s]));
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
            let x_sel = x2.index_select(&idx, 0)?; // [count, h]
            let out = self.experts[e].forward(&x_sel)?; // [count, h]
            let w = Tensor::from_vec(w, (count, 1), dev)?;
            y = y.index_add(&idx, &out.broadcast_mul(&w)?, 0)?;
        }
        Ok((y, ids))
    }

    /// Single-token decode dispatch.  With one token every selected expert
    /// consumes the same input row, so there is nothing to gather: run the
    /// `k` expert MLPs over `x2`, stack their outputs, scale by the routing
    /// weights (still on the device — no copy back) and sum.
    fn dispatch_decode(
        &self,
        x2: &Tensor,
        topk_idx: &Tensor,
        weights: &Tensor,
    ) -> Result<(Tensor, Vec<u32>)> {
        let k = self.n_expert_used;
        let ids: Vec<u32> = topk_idx.flatten_all()?.to_vec1()?; // [k] — the one host sync
        let mut outs = Vec::with_capacity(k);
        for &e in ids.iter() {
            outs.push(self.experts[e as usize].forward(x2)?); // [1, h] each
        }
        let out = Tensor::stack(&outs, 0)?; // [k, 1, h]
        let w = weights.reshape((k, 1, 1))?; // GPU view [k, 1, 1]
        let y = out.broadcast_mul(&w)?.sum(0)?; // [1, h]
        Ok((y, ids))
    }
}

/// Indices of the top-`k` values along the last dim (descending), as u32.
fn topk_indices(t: &Tensor, k: usize) -> Result<Tensor> {
    t.arg_sort_last_dim(false)?
        .narrow(D::Minus1, 0, k)?
        .contiguous()
}

/// Top-`k` values along the last dim (descending).
fn topk_values(t: &Tensor, k: usize) -> Result<Tensor> {
    let idx = topk_indices(t, k)?;
    t.gather(&idx, D::Minus1)
}

// ─── Layer + model ───────────────────────────────────────────────────────────

enum FeedForward {
    Dense(Mlp),
    Moe(Moe),
}

impl FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(m) => m.forward(xs),
            Self::Moe(m) => m.forward(xs).map(|(t, _)| t),
        }
    }

    /// [`FeedForward::forward`] plus the routed-expert ids this input used
    /// (empty for dense blocks) — the routing-frequency cache's input.
    fn forward_routed(&self, xs: &Tensor) -> Result<(Tensor, Vec<u32>)> {
        match self {
            Self::Dense(m) => Ok((m.forward(xs)?, Vec::new())),
            Self::Moe(m) => m.forward(xs),
        }
    }
}

struct Layer {
    attn_norm: RmsNorm,
    attn: Attention,
    ffn_norm: RmsNorm,
    ffn: FeedForward,
}

/// A quantized DeepSeek-V2/V3/Kimi-K2 model loaded from GGUF.
pub struct ModelWeights {
    tok_embeddings: Tensor,
    layers: Vec<Layer>,
    norm: RmsNorm,
    output: QMatMul,
    device: Device,
    /// Routing-frequency LRU hot-expert cache (shared bookkeeping; budget set
    /// after load via [`ModelWeights::set_pin_hot_experts`], CLI flag
    /// `--pin-hot-experts`).  Records routing, re-selects the hot set every
    /// [`crate::hot_experts::REFRESH_STEPS`] decode steps, and reports newly
    /// hot experts for the residency backend.
    hot_experts: crate::hot_experts::HotExpertCache,
    /// Executes residency for the hot set (CPU madvise today; a device slot
    /// cache later).  Built once at load from the per-expert handles.
    residency: std::sync::Arc<dyn crate::residency::ExpertResidency>,
}

/// Small GGUF reader over the memory-mapped file.
struct Reader<R: Read + Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
    /// When present, tensors are borrowed in place from this mapping instead
    /// of being copied onto the heap (see [`crate::mmap_tensor`]).  This is
    /// what makes a model far larger than RAM loadable: pages fault in only
    /// as weights are actually touched.
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
        self.qtensor(name)?.dequantize(&self.device)?.to_dtype(DType::F32)
    }
    fn has(&self, name: &str) -> bool {
        self.ct.tensor_infos.contains_key(name)
    }
    fn mlp(&mut self, prefix: &str) -> Result<Mlp> {
        Ok(Mlp {
            gate: self.qmatmul(&format!("{prefix}_gate.weight"))?,
            up: self.qmatmul(&format!("{prefix}_up.weight"))?,
            down: self.qmatmul(&format!("{prefix}_down.weight"))?,
            prefetch: None,
        })
    }
}

impl ModelWeights {
    /// Load a `deepseek2` GGUF (DeepSeek-V2/V3, Kimi-K2).
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
    /// the page cache and fault in on demand — the difference between a
    /// trillion-parameter MoE being unloadable and merely slow.
    pub fn from_gguf_mmap<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        mmap: Option<std::sync::Arc<memmap2::Mmap>>,
    ) -> Result<Self> {
        let cfg = Config::from_metadata(&ct.metadata)?;
        // The Content owns metadata; move it into our reader together with the
        // underlying file handle (borrowed for the lifetime of the load).
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
            None => rd.qmatmul("token_embd.weight")?, // tied
        };

        let rotary = Arc::new(RotaryEmbedding::new(&cfg, device)?);

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = format!("blk.{i}");
            let attn_norm = rd.rms_norm(&format!("{p}.attn_norm.weight"), cfg.rms_eps)?;
            let ffn_norm = rd.rms_norm(&format!("{p}.ffn_norm.weight"), cfg.rms_eps)?;

            // Q projection: LoRA (V2-full/V3/K2) or plain (V2-Lite).
            let q = if cfg.q_lora_rank.is_some() {
                QProj::Lora {
                    a: rd.qmatmul(&format!("{p}.attn_q_a.weight"))?,
                    norm: rd.rms_norm(&format!("{p}.attn_q_a_norm.weight"), cfg.rms_eps)?,
                    b: rd.qmatmul(&format!("{p}.attn_q_b.weight"))?,
                }
            } else {
                QProj::Plain(rd.qmatmul(&format!("{p}.attn_q.weight"))?)
            };

            let kv_a_mqa = rd.qmatmul(&format!("{p}.attn_kv_a_mqa.weight"))?;
            let kv_a_norm = rd.rms_norm(&format!("{p}.attn_kv_a_norm.weight"), cfg.rms_eps)?;
            let kv_b = load_kv_b(&mut rd, &p, &cfg)?;
            let o_proj = rd.qmatmul(&format!("{p}.attn_output.weight"))?;

            let attn = Attention {
                q,
                kv_a_mqa,
                kv_a_norm,
                kv_b,
                o_proj,
                rotary: rotary.clone(),
                n_head: cfg.n_head,
                kv_lora_rank: cfg.kv_lora_rank,
                qk_nope: cfg.qk_nope_head_dim,
                qk_rope: cfg.qk_rope_head_dim,
                v_head_dim: cfg.v_head_dim,
                q_head_dim: cfg.q_head_dim(),
                softmax_scale: cfg.softmax_scale,
                kv_cache: None,
            };

            let ffn = if cfg.n_expert > 0 && i >= cfg.leading_dense {
                FeedForward::Moe(load_moe(&mut rd, &p, &cfg)?)
            } else {
                FeedForward::Dense(rd.mlp(&format!("{p}.ffn"))?)
            };

            layers.push(Layer {
                attn_norm,
                attn,
                ffn_norm,
                ffn,
            });
        }

        let n_layers = layers.len();
        let residency: std::sync::Arc<dyn crate::residency::ExpertResidency> =
            std::sync::Arc::new(crate::residency::CpuResidency::new(
                layers
                    .iter()
                    .map(|layer| match &layer.ffn {
                        FeedForward::Moe(moe) => {
                            moe.experts.iter().map(|m| m.prefetch.clone()).collect()
                        }
                        FeedForward::Dense(_) => Vec::new(),
                    })
                    .collect(),
            ));
        Ok(Self {
            tok_embeddings,
            layers,
            norm,
            output,
            device: device.clone(),
            hot_experts: crate::hot_experts::HotExpertCache::new(
                n_layers,
                cfg.n_expert,
                0,
            ),
            residency,
        })
    }

    fn causal_mask(&self, seq_len: usize, offset: usize) -> Result<Tensor> {
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..seq_len + offset).map(move |j| {
                    if j > i + offset {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        Tensor::from_slice(&mask, (1, 1, seq_len, seq_len + offset), &self.device)
    }

    /// Forward pass. `input` is `[1, seq_len]`; `offset` is the KV-cache
    /// position of the first input token.
    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_b, seq_len) = input.dims2()?;
        let mut xs = self.tok_embeddings.index_select(&input.flatten_all()?, 0)?
            .reshape((1, seq_len, self.tok_embeddings.dim(1)?))?;

        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.causal_mask(seq_len, offset)?)
        };

        // Routing-frequency hot-expert cache: every
        // crate::hot_experts::REFRESH_STEPS decode steps, re-select the
        // most-used experts (recency as the tie-break) and WILLNEED their
        // pages, so the common routing path stays resident instead of
        // faulting from disk each step.  Runs before the layer loop so the
        // prefetch has a full step of compute to stream in behind.
        let step = self.hot_experts.begin_step(seq_len == 1);
        if self.hot_experts.refresh_due(seq_len == 1) {
            for (l, e) in self.hot_experts.refresh() {
                self.residency.acquire(l, e);
            }
        }

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let residual = &xs;
            let h = layer.attn_norm.forward(&xs)?;
            let h = layer.attn.forward(&h, mask.as_ref(), offset)?;
            let xs2 = (residual + h)?;

            let residual = &xs2;
            let h = layer.ffn_norm.forward(&xs2)?;
            let (h, routed) = layer.ffn.forward_routed(&h)?;
            self.hot_experts.record(i, &routed, step);
            xs = (residual + h)?;
        }

        let xs = xs.narrow(1, seq_len - 1, 1)?;
        let xs = self.norm.forward(&xs)?;
        self.output.forward(&xs)?.to_dtype(DType::F32)?.squeeze(1)
    }

    /// Set the routing-frequency hot-expert cache budget (experts kept
    /// resident).  Call once after load, before serving; routing is recorded
    /// from the first forward pass and the pinned set is re-selected every
    /// [`crate::hot_experts::REFRESH_STEPS`] decode steps.
    pub fn set_pin_hot_experts(&mut self, n: usize) {
        self.hot_experts.set_budget(n);
    }

    /// Reset the KV cache so this instance can serve an unrelated prompt.
    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.attn.clear_kv_cache();
        }
    }

    /// Keep only the first `keep` fed tokens of every layer's KV cache.
    ///
    /// See [`Attention::truncate_kv_cache`] for the semantics; used by the
    /// engine's edited-context prefix reuse (a follow-up prompt that shares
    /// a prefix with the cached history after an agent harness truncated or
    /// replaced middle blocks).
    pub fn truncate_kv_cache(&mut self, keep: usize) -> Result<()> {
        for layer in self.layers.iter_mut() {
            layer.attn.truncate_kv_cache(keep)?;
        }
        Ok(())
    }
}

/// Load the KV up-projection, folding pre-split `attn_k_b`/`attn_v_b` back into
/// the combined `kv_lora_rank → n_head*(qk_nope + v_head_dim)` weight when the
/// GGUF ships the MLA-split form.
fn load_kv_b<R: Read + Seek>(rd: &mut Reader<R>, p: &str, cfg: &Config) -> Result<KvB> {
    if rd.has(&format!("{p}.attn_kv_b.weight")) {
        return Ok(KvB::Quantized(rd.qmatmul(&format!("{p}.attn_kv_b.weight"))?));
    }
    // Reconstruct from the split tensors.
    //   attn_k_b: candle shape [n_head, kv_lora_rank, qk_nope]
    //   attn_v_b: candle shape [n_head, v_head_dim, kv_lora_rank]
    let k_b = rd.f32_tensor(&format!("{p}.attn_k_b.weight"))?;
    let v_b = rd.f32_tensor(&format!("{p}.attn_v_b.weight"))?;
    let h = cfg.n_head;
    let lkv = cfg.kv_lora_rank;
    let np = cfg.qk_nope_head_dim;
    let vh = cfg.v_head_dim;
    let k_b = k_b.reshape((h, lkv, np))?;
    let v_b = v_b.reshape((h, vh, lkv))?;
    // Per head, stack [k_nope-proj; v-proj] as rows → [(np+vh), lkv], then over
    // heads → [h*(np+vh), lkv].
    let mut rows = Vec::with_capacity(h);
    for head in 0..h {
        let k_head = k_b.i(head)?.t()?.contiguous()?; // [np, lkv]
        let v_head = v_b.i(head)?.contiguous()?; // [vh, lkv]
        rows.push(Tensor::cat(&[&k_head, &v_head], 0)?); // [(np+vh), lkv]
    }
    let w = Tensor::cat(&rows.iter().collect::<Vec<_>>(), 0)?; // [h*(np+vh), lkv]
    Ok(KvB::Dense(w))
}

/// Load a MoE feed-forward block: quantized per-expert SwiGLU experts, the f32
/// router (+ optional bias), and any shared experts.
fn load_moe<R: Read + Seek>(rd: &mut Reader<R>, p: &str, cfg: &Config) -> Result<Moe> {
    let gate = rd.f32_tensor(&format!("{p}.ffn_gate_inp.weight"))?; // [n_expert, n_embd]
    let gate_bias = rd
        .has(&format!("{p}.exp_probs_b.bias"))
        .then(|| rd.f32_tensor(&format!("{p}.exp_probs_b.bias")))
        .transpose()?;

    // Slice the 3-D expert tensors into per-expert quantized QMatMuls.
    let gate_exps = split_experts(rd, &format!("{p}.ffn_gate_exps.weight"), cfg.n_expert)?;
    let up_exps = split_experts(rd, &format!("{p}.ffn_up_exps.weight"), cfg.n_expert)?;
    let down_exps = split_experts(rd, &format!("{p}.ffn_down_exps.weight"), cfg.n_expert)?;
    let experts = gate_exps
        .into_iter()
        .zip(up_exps)
        .zip(down_exps)
        .map(|((gate, up), down)| Mlp {
            gate: gate.qmatmul,
            up: up.qmatmul,
            down: down.qmatmul,
            prefetch: match (gate.prefetch, up.prefetch, down.prefetch) {
                (Some(gate), Some(up), Some(down)) => {
                    Some(crate::residency::ExpertHandles { gate, up, down })
                }
                _ => None,
            },
        })
        .collect();

    let shared = if cfg.n_expert_shared > 0 {
        Some(Mlp {
            gate: rd.qmatmul(&format!("{p}.ffn_gate_shexp.weight"))?,
            up: rd.qmatmul(&format!("{p}.ffn_up_shexp.weight"))?,
            down: rd.qmatmul(&format!("{p}.ffn_down_shexp.weight"))?,
            prefetch: None,
        })
    } else {
        None
    };

    Ok(Moe {
        gate_t: gate.t()?.contiguous()?,
        gate_bias,
        experts,
        shared,
        gating: cfg.gating,
        n_expert_used: cfg.n_expert_used,
        n_group: cfg.n_group,
        topk_group: cfg.topk_group,
        weights_norm: cfg.expert_weights_norm,
        weights_scale: cfg.expert_weights_scale,
    })
}

/// Split a 3-D expert weight `[n_expert, out, in]` into `n_expert` quantized
/// 2-D `QMatMul`s by carving its raw quantized byte-buffer — no dequantization,
/// so the experts keep their on-disk size.
/// One expert's weight: the [`QMatMul`] plus an optional handle for
/// prefetching its byte range from the mapping.
struct ExpertWeight {
    qmatmul: QMatMul,
    prefetch: Option<Arc<dyn crate::mmap_tensor::MmapPrefetch>>,
}

fn split_experts<R: Read + Seek>(
    rd: &mut Reader<R>,
    name: &str,
    n_expert: usize,
) -> Result<Vec<ExpertWeight>> {
    let qt = rd.qtensor(name)?;
    let dims = qt.shape().dims().to_vec();
    if dims.len() != 3 || dims[0] != n_expert {
        candle_core::bail!(
            "deepseek2: expected expert tensor `{name}` shaped [n_expert, out, in], got {dims:?}"
        );
    }
    let (out, inn) = (dims[1], dims[2]);
    let dtype: GgmlDType = qt.dtype();

    // Preferred path: point each expert straight at its own slice of the
    // mapping.  Building all of them reads nothing — an expert is a pointer
    // and a length — so a layer with hundreds of experts costs almost no
    // memory until tokens actually route to them, at which point the kernel
    // faults in just those pages and can evict them again later.
    // Borrowed storage is always CPU-resident, so this is only valid when the
    // model itself runs on the CPU — otherwise the experts would sit in host
    // memory while activations live on the accelerator. `qtensor_from_mmap`
    // applies the same guard for ordinary tensors.
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
                        Some(t) => borrowed.push(ExpertWeight {
                            qmatmul: QMatMul::from_qtensor(t)?,
                            prefetch: crate::mmap_tensor::prefetch_handle(
                                &mmap,
                                dtype,
                                base + e * per_bytes,
                                per_elems / block_size,
                            ),
                        }),
                        // Any expert that cannot be borrowed (misalignment,
                        // truncated file) drops the whole layer to the copying
                        // path rather than mixing the two.
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
            "deepseek2: expert tensor `{name}` byte length {} not divisible by n_expert {n_expert}",
            bytes.len()
        );
    }
    let per = bytes.len() / n_expert;
    let mut experts = Vec::with_capacity(n_expert);
    for e in 0..n_expert {
        let slice = &bytes[e * per..(e + 1) * per];
        let storage = QStorage::from_data(Cow::Borrowed(slice), &rd.device, dtype)?;
        let qt = QTensor::new(storage, (out, inn))?;
        experts.push(ExpertWeight {
            qmatmul: QMatMul::from_qtensor(qt)?,
            prefetch: None,
        });
    }
    Ok(experts)
}


#[cfg(test)]
mod tests {
    use super::*;

    /// Shared random weights for a tiny `Attention` (`n_head=2`,
    /// `kv_lora_rank=6`, `qk_nope=4`, `qk_rope=4`, `v_head_dim=4`, `n_embd=8`).
    /// Built once and shared so two instances can be compared exactly.
    struct TinyW {
        q: Tensor, // [n_head*q_head_dim, n_embd]
        kv_a: Tensor, // [kv_lora_rank + qk_rope, n_embd]
        norm_w: Vec<f32>, // [kv_lora_rank]
        kv_b: Tensor, // [n_head*(qk_nope+v_head_dim), kv_lora_rank]
        o: Tensor, // [n_embd, n_head*v_head_dim]
        nh: usize,
        lkv: usize,
        np: usize,
        r: usize,
        vh: usize,
        qd: usize,
        h: usize,
    }

    fn tiny_weights(dev: &Device) -> Result<TinyW> {
        let h = 8usize; // n_embd
        let nh = 2usize; // n_head
        let lkv = 6usize; // kv_lora_rank
        let np = 4usize; // qk_nope
        let r = 4usize; // qk_rope
        let vh = 4usize; // v_head_dim
        let qd = np + r;
        let norm_w: Vec<f32> = Tensor::randn(0f32, 1f32, (lkv,), dev)?.to_vec1()?;
        Ok(TinyW {
            q: Tensor::randn(0f32, 1f32, (nh * qd, h), dev)?,
            kv_a: Tensor::randn(0f32, 1f32, (lkv + r, h), dev)?,
            norm_w,
            kv_b: Tensor::randn(0f32, 1f32, (nh * (np + vh), lkv), dev)?,
            o: Tensor::randn(0f32, 1f32, (h, nh * vh), dev)?,
            nh,
            lkv,
            np,
            r,
            vh,
            qd,
            h,
        })
    }

    fn tiny_attention(dev: &Device, w: &TinyW) -> Result<Attention> {
        use candle_core::quantized::{QStorage, QTensor};
        use std::borrow::Cow;

        let lin = |_rows: usize, _cols: usize, t: &Tensor| -> QMatMul {
            QMatMul::Tensor(t.clone())
        };
        // RMS norm weight as an f32 QTensor (candle-transformers' RmsNorm only
        // builds from a QTensor).
        let bytes: Vec<u8> = w.norm_w.iter().flat_map(|x| x.to_le_bytes()).collect();
        let storage = QStorage::from_data(Cow::Owned(bytes), dev, GgmlDType::F32)?;
        let norm = RmsNorm::from_qtensor(QTensor::new(storage, w.lkv)?, 1e-5)?;
        let cfg = Config {
            n_layer: 1,
            n_head: w.nh,
            rms_eps: 1e-5,
            q_lora_rank: None,
            kv_lora_rank: w.lkv,
            qk_nope_head_dim: w.np,
            qk_rope_head_dim: w.r,
            v_head_dim: w.vh,
            softmax_scale: 1.0 / (w.qd as f64).sqrt(),
            rope_theta: 10_000.0,
            context_length: 4096,
            yarn: None,
            leading_dense: 0,
            n_expert: 0,
            n_expert_used: 0,
            n_expert_shared: 0,
            expert_weights_scale: 1.0,
            expert_weights_norm: false,
            gating: Gating::Sigmoid,
            n_group: 1,
            topk_group: 1,
        };
        Ok(Attention {
            q: QProj::Plain(lin(w.nh * w.qd, w.h, &w.q)),
            kv_a_mqa: lin(w.lkv + w.r, w.h, &w.kv_a),
            kv_a_norm: norm,
            kv_b: KvB::Dense(w.kv_b.clone()),
            o_proj: lin(w.h, w.nh * w.vh, &w.o),
            rotary: Arc::new(RotaryEmbedding::new(&cfg, dev)?),
            n_head: w.nh,
            kv_lora_rank: w.lkv,
            qk_nope: w.np,
            qk_rope: w.r,
            v_head_dim: w.vh,
            q_head_dim: w.qd,
            softmax_scale: cfg.softmax_scale,
            kv_cache: None,
        })
    }

    /// The cache must hold the reconstructed per-head K/V (appended
    /// incrementally), not the compressed latent — decode then costs O(1)
    /// reconstruction per step instead of the O(seq) full rebuild.
    #[test]
    fn mla_cache_stores_reconstructed_per_head_kv() -> Result<()> {
        let dev = Device::Cpu;
        let w = tiny_weights(&dev)?;
        let mut attn = tiny_attention(&dev, &w)?;

        // Prefill: 3 tokens at once.
        let xs = Tensor::randn(0f32, 1f32, (1, 3, 8), &dev)?;
        attn.forward(&xs, None, 0)?;
        let (k, v) = attn.kv_cache.as_ref().expect("cache populated after prefill");
        assert_eq!(
            k.dims(),
            &[1, 2, 8, 3],
            "k cache must be [b, n_head, qk_nope + qk_rope, seq]"
        );
        assert_eq!(v.dims(), &[1, 2, 4, 3], "v cache must be [b, n_head, v_head_dim, seq]");

        // Decode: one more token appends along seq.
        let xs2 = Tensor::randn(0f32, 1f32, (1, 1, 8), &dev)?;
        attn.forward(&xs2, None, 3)?;
        let (k, v) = attn.kv_cache.as_ref().expect("cache after decode");
        assert_eq!(k.dims(), &[1, 2, 8, 4]);
        assert_eq!(v.dims(), &[1, 2, 4, 4]);
        Ok(())
    }

    /// The reconstructed K/V must agree whether the cache was filled by one
    /// prefill call or by repeated single-token calls (regression guard for the
    /// latent-cache concat/decompress ordering).
    #[test]
    fn mla_latent_cache_prefill_matches_incremental_reconstruction() -> Result<()> {
        let dev = Device::Cpu;

        // Prefill path: 3 tokens in one forward (causal mask, as the engine does).
        let w = tiny_weights(&dev)?;
        let mut a = tiny_attention(&dev, &w)?;
        let xs = Tensor::randn(0f32, 1f32, (1, 3, 8), &dev)?;
        let mask: Vec<f32> = (0..3)
            .flat_map(|i| (0..3).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        let mask = Tensor::from_slice(&mask, (1, 1, 3, 3), &dev)?;
        let out_prefill = a.forward(&xs, Some(&mask), 0)?;
        let (c_pre, k_pre) = a.kv_cache.as_ref().unwrap();

        // Incremental path: 1 + 1 + 1 tokens with growing offset.
        let mut b = tiny_attention(&dev, &w)?;
        let mut out_inc = Vec::new();
        for off in 0..3 {
            let t = xs.narrow(1, off, 1)?;
            out_inc.push(b.forward(&t, None, off)?);
        }
        let out_inc = Tensor::cat(&out_inc, 1)?;
        let (c_inc, k_inc) = b.kv_cache.as_ref().unwrap();

        assert_eq!(c_pre.dims(), c_inc.dims());
        assert_eq!(k_pre.dims(), k_inc.dims());
        let diff = (c_pre.to_dtype(DType::F32)? - c_inc.to_dtype(DType::F32)?)?
            .abs()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let max = diff.into_iter().fold(0.0f32, f32::max);
        assert!(max < 1e-6, "latent caches diverge: max diff {max}");
        let out_diff = (out_prefill.to_dtype(DType::F32)? - out_inc.to_dtype(DType::F32)?)?
            .abs()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let max = out_diff.into_iter().fold(0.0f32, f32::max);
        assert!(max < 1e-5, "attention outputs diverge: max diff {max}");
        Ok(())
    }

    /// The single-token decode dispatch path must agree exactly with the
    /// batched prefill path: each token is routed to the same experts and the
    /// per-expert MLPs are applied to the same input, just gathered/stacked
    /// differently.
    #[test]
    fn moe_decode_dispatch_matches_batched_dispatch() -> Result<()> {
        use candle_core::quantized::QMatMul;

        let dev = Device::Cpu;
        let h = 6usize; // n_embd
        let n_expert = 8usize;
        let k = 2usize; // n_expert_used
        let ffn = 10usize; // intermediate size

        let lin = |rows: usize, cols: usize| -> QMatMul {
            QMatMul::Tensor(Tensor::randn(0f32, 1f32, (rows, cols), &dev).unwrap())
        };
        let gate = Tensor::randn(0f32, 1f32, (n_expert, h), &dev)?;
        let experts = (0..n_expert)
            .map(|_| Mlp {
                gate: lin(ffn, h),
                up: lin(ffn, h),
                down: lin(h, ffn),
                prefetch: None,
            })
            .collect();
        let moe = Moe {
            gate_t: gate.t()?.contiguous()?,
            gate_bias: None,
            experts,
            shared: None,
            gating: Gating::Softmax,
            n_expert_used: k,
            n_group: 1,
            topk_group: 1,
            weights_norm: false,
            weights_scale: 0.0,
        };

        // Batched path: 3 tokens in one forward (softmax routing per token).
        let xs = Tensor::randn(0f32, 1f32, (1, 3, h), &dev)?;
        let (out_batch, _) = moe.forward(&xs)?; // [1, 3, h]

        // Decode path: same 3 tokens one at a time, concatenated.
        let mut out_inc = Vec::new();
        for t in 0..3 {
            out_inc.push(moe.forward(&xs.narrow(1, t, 1)?)?.0); // [1, 1, h]
        }
        let out_inc = Tensor::cat(&out_inc, 1)?; // [1, 3, h]

        let diff = (out_batch - out_inc)?
            .abs()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let max = diff.into_iter().fold(0.0f32, f32::max);
        assert!(max < 1e-5, "decode vs batched dispatch diverge: max diff {max}");
        Ok(())
    }
}
