//! Pure-Rust quantized loader for the `deepseek`, `deepseek2` and `glm-dsa`
//! GGUF architectures.
//!
//! Covers DeepSeek-MoE (`deepseek`), DeepSeek-V2, DeepSeek-V2-Lite,
//! DeepSeek-V2.5, DeepSeek-V3 / V3.1 / R1, **Kimi-K2** and GLM-4.7-Flash —
//! every model llama.cpp labels `general.architecture = "deepseek"` or
//! `"deepseek2"` — plus GLM-5 / 5.1 / 5.2 (`glm-dsa`), which add DeepSeek
//! Sparse Attention to the same MLA + MoE stack (see [`dsa`]), and
//! GLM-5.3-Flash (`glm5next`, also written `glm5-next`), which further swaps
//! three in four attention layers for Kimi Delta Attention (see [`kda`]),
//! drops RoPE, pools the indexer's keys, clamps its SwiGLUs and carries the
//! residual as manifold-constrained hyper-connections ([`crate::mhc`]).
//! candle ships a *full-precision* `deepseek2` model but no quantized/GGUF
//! one, and its gate only implements DeepSeek-V2 softmax routing; this module
//! adds the GGUF path plus the DeepSeek-V3 / Kimi-K2 sigmoid-with-bias,
//! group-limited routing.
//!
//! The two architectures share the whole MoE stack (leading dense layers,
//! fine-grained routed experts, always-on shared experts) and differ only in
//! attention: `deepseek` (DeepSeek-MoE 16B, the V1 generation) uses plain
//! GQA with full-width RoPE, `deepseek2` uses MLA.  Both run through one
//! cached-attention kernel ([`cached_attention`]) over the same transposed
//! KV-cache layout.
//!
//! Two things make the `deepseek2` architecture unusual:
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

use std::io::{Read, Seek};
use std::sync::Arc;

use candle_core::quantized::{gguf_file, QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_transformers::quantized_nn::RmsNorm;

use crate::attention::{cached_attention, KvCache, Rope, RopeStyle};
use crate::gguf_meta::Meta;
use crate::tensor_source::TensorSource;
use crate::moe::{topk_indices, topk_values, Gating};
use crate::token_embedding::TokenEmbedding;
use crate::yarn::YarnConfig;

mod dsa;
mod kda;
use dsa::{DsaConfig, IndexCache, Indexer, Selection};
use kda::{Kda, KdaConfig, KdaState};
use crate::mhc::HyperConnection;

/// Parsed `deepseek` / `deepseek2` hyper-parameters.
///
/// A `deepseek` (GQA) model reuses the MLA head-dim fields with no un-rotated
/// slice: `qk_nope_head_dim = 0`, `qk_rope_head_dim = head_dim`, and
/// `kv_lora_rank = 0` marks the absence of the latent projection.
struct Config {
    n_layer: usize,
    n_head: usize,
    /// KV heads (GQA); equal to `n_head` for MLA.
    n_kv_head: usize,
    rms_eps: f64,
    // MLA dims (`kv_lora_rank == 0` for GQA).
    q_lora_rank: Option<usize>,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    softmax_scale: f64,
    // RoPE.
    rope_theta: f32,
    /// Linear RoPE scaling (`1 / rope.scaling.factor` for `"linear"`), else 1.
    rope_freq_scale: f32,
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
    /// DeepSeek Sparse Attention (`glm-dsa`, `glm5next`).
    dsa: Option<DsaConfig>,
    /// Per layer: Kimi Delta Attention (`glm5next`) instead of MLA.
    recurrent: Vec<bool>,
    kda: Option<KdaConfig>,
    /// Hyper-connection `(streams, Sinkhorn iterations, eps)` (`glm5next`).
    hc: Option<(usize, usize, f64)>,
    /// Per-layer SwiGLU clamps for the routed experts and for the shared
    /// experts / dense FFNs (`0`: none).
    swiglu_exp: Vec<f64>,
    swiglu_shexp: Vec<f64>,
}

impl Config {
    /// Per-head Q/K dimension (`qk_nope_head_dim + qk_rope_head_dim`).
    fn q_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Whether attention is MLA (`deepseek2`) rather than GQA (`deepseek`).
    fn is_mla(&self) -> bool {
        self.kv_lora_rank > 0
    }
}

impl Config {
    /// Parse the hyper-parameters of a `deepseek`, `deepseek2` or `glm-dsa`
    /// GGUF.
    fn from_metadata(
        md: &std::collections::HashMap<String, gguf_file::Value>,
        arch: &str,
    ) -> Result<Self> {
        let m = Meta::new(md, arch);
        let n_head = m.u32("attention.head_count")? as usize;
        let n_embd = m.u32("embedding_length")? as usize;
        // `block_count` includes any appended NextN / MTP blocks
        // (GLM-4.7-Flash, GLM-5); those are draft heads, not trunk layers.
        let block_count = m.u32("block_count")?;
        let Some(n_layer) = block_count.checked_sub(m.u32_or("nextn_predict_layers", 0)) else {
            candle_core::bail!("{arch}: nextn_predict_layers exceeds block_count {block_count}");
        };
        let n_layer = n_layer as usize;
        let glm5next = matches!(arch, "glm5next" | "glm5-next");
        let rms_eps = m.f32("attention.layer_norm_rms_epsilon")? as f64;
        let context_length = m.u32("context_length")? as usize;
        if n_head == 0 {
            candle_core::bail!("{arch}: attention.head_count must be positive");
        }

        // MLA advertises its latent rank (required for `deepseek2`);
        // DeepSeek-MoE (`deepseek`) has none.
        let kv_lora_rank = if arch != "deepseek" {
            m.u32("attention.kv_lora_rank")?
        } else {
            m.u32_or("attention.kv_lora_rank", 0)
        } as usize;
        let mla = kv_lora_rank > 0;

        // Head dims. `is_mla` (pre-split k_b/v_b) advertises key_length_mla;
        // otherwise fall back to key_length / value_length (default n_embd/n_head).
        let key_length_mla = m.u32_or("attention.key_length_mla", 0) as usize;
        let value_length_mla = m.u32_or("attention.value_length_mla", 0) as usize;
        let split_mla = key_length_mla != 0 && value_length_mla != 0;
        let n_embd_head_k = if split_mla {
            key_length_mla
        } else {
            m.u32_or("attention.key_length", (n_embd / n_head) as u32) as usize
        };
        let v_head_dim = if split_mla {
            value_length_mla
        } else {
            m.u32_or("attention.value_length", (n_embd / n_head) as u32) as usize
        };
        let qk_rope_head_dim = m.u32_or("rope.dimension_count", n_embd_head_k as u32) as usize;
        let qk_nope_head_dim = n_embd_head_k.saturating_sub(qk_rope_head_dim);

        let n_kv_head = if mla {
            n_head
        } else {
            m.u32_or("attention.head_count_kv", n_head as u32) as usize
        };
        if !mla {
            if n_kv_head == 0 || !n_head.is_multiple_of(n_kv_head) {
                candle_core::bail!(
                    "{arch}: head_count {n_head} must be a multiple of head_count_kv {n_kv_head}"
                );
            }
            if qk_nope_head_dim != 0 {
                candle_core::bail!(
                    "{arch}: partial rotary embeddings (rope.dimension_count {qk_rope_head_dim} < \
                     head dim {n_embd_head_k}) are not supported"
                );
            }
        }
        // GLM-5.3-Flash is NoPE: no rotary slice at all.
        if (qk_rope_head_dim == 0 && !glm5next) || !qk_rope_head_dim.is_multiple_of(2) {
            candle_core::bail!(
                "{arch}: invalid rotary dimension {qk_rope_head_dim} (must be a positive even number)"
            );
        }

        let q_lora_rank = m.u32_opt("attention.q_lora_rank").map(|v| v as usize).filter(|&v| v > 0);

        let rope_theta = m.f32_or("rope.freq_base", 10_000.0);
        // YaRN long-context scaling (optional), or plain linear interpolation.
        let yarn = YarnConfig::from_meta(&m, 1.0, context_length);
        let rope_freq_scale = match m.string("rope.scaling.type").as_deref() {
            Some("linear") => {
                let factor = m.f32_or("rope.scaling.factor", 1.0);
                if factor > 0.0 { 1.0 / factor } else { 1.0 }
            }
            _ => 1.0,
        };

        // Softmax scale: 1/sqrt(q_head_dim), YaRN-corrected by mscale².
        let q_head_dim = qk_nope_head_dim + qk_rope_head_dim;
        let mut softmax_scale = 1.0f64 / (q_head_dim as f64).sqrt();
        if let Some(y) = &yarn {
            let mscale = y.mscale() as f64;
            softmax_scale *= mscale * mscale;
        }

        let leading_dense = m.u32_or("leading_dense_block_count", 0) as usize;
        let n_expert = m.u32_or("expert_count", 0) as usize;
        let n_expert_used = m.u32_or("expert_used_count", 0) as usize;
        let n_expert_shared = m.u32_or("expert_shared_count", 0) as usize;
        let expert_weights_scale = m.f32_or("expert_weights_scale", 0.0) as f64;
        let expert_weights_norm = m.bool_or("expert_weights_norm", false);
        // Absent → softmax (DeepSeek-MoE / V2); GLM-5 routes by sigmoid.
        let glm = arch == "glm-dsa" || glm5next;
        let gating = Gating::from_meta(&m, if glm { Gating::Sigmoid } else { Gating::Softmax });
        let n_group = m.u32_or("expert_group_count", 0) as usize;
        let topk_group = m.u32_or("expert_group_used_count", 0) as usize;
        if n_expert > 0 && (n_expert_used == 0 || n_expert_used > n_expert) {
            candle_core::bail!(
                "{arch}: expert_used_count {n_expert_used} must be in 1..=expert_count {n_expert}"
            );
        }

        let dsa = if glm {
            if q_lora_rank.is_none() {
                candle_core::bail!("{arch}: sparse attention needs the Q LoRA projection (attention.q_lora_rank)");
            }
            Some(DsaConfig::from_meta(&m, n_layer, qk_rope_head_dim, context_length)?)
        } else {
            None
        };

        // GLM-5.3-Flash marks its KDA layers with a zero KV head count.
        let recurrent: Vec<bool> = if glm5next {
            m.array_u32("attention.head_count_kv", n_layer).iter().map(|&n| n == 0).collect()
        } else {
            vec![false; n_layer]
        };
        let kda = glm5next.then(|| KdaConfig::from_meta(&m, n_head)).transpose()?;
        let hc = if glm5next {
            let n = m.u32("hyper_connection.count")? as usize;
            if n < 2 {
                candle_core::bail!("{arch}: hyper_connection.count must be at least 2, got {n}");
            }
            Some((
                n,
                m.u32_or("hyper_connection.sinkhorn_iterations", 20) as usize,
                m.f32_or("hyper_connection.epsilon", 1e-6) as f64,
            ))
        } else {
            None
        };
        let swiglu_exp = m.array_f64("swiglu_clamp_exp", n_layer);
        let swiglu_shexp = if m.contains("swiglu_clamp_shexp") {
            m.array_f64("swiglu_clamp_shexp", n_layer)
        } else {
            swiglu_exp.clone()
        };

        Ok(Self {
            n_layer,
            n_head,
            n_kv_head,
            rms_eps,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            softmax_scale,
            rope_theta,
            rope_freq_scale,
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
            dsa,
            recurrent,
            kda,
            hc,
            swiglu_exp,
            swiglu_shexp,
        })
    }
}

// ─── RoPE (YaRN-aware, applied to the qk_rope slice only) ────────────────────

/// Interleaved (llama.cpp `NORM`) RoPE over the `qk_rope` slice, with YaRN
/// frequency blending and magnitude scaling when configured.
fn rope_table(cfg: &Config, dev: &Device) -> Result<Rope> {
    let dim = cfg.qk_rope_head_dim;
    let max_seq = cfg.context_length;
    let theta = cfg.rope_theta;
    match &cfg.yarn {
        None => {
            let inv_freq = crate::attention::inv_freq(dim, theta)
                .into_iter()
                .map(|f| f * cfg.rope_freq_scale)
                .collect();
            Rope::from_inv_freq(inv_freq, max_seq, 1.0, RopeStyle::Interleaved, dev)
        }
        // Interpolated vs extrapolated frequencies blended by a ramp over
        // the YaRN correction range (see DeepSeek modeling code).
        Some(y) => Rope::from_inv_freq(
            crate::yarn::inv_freq(dim, theta, y),
            max_seq,
            y.mscale(),
            RopeStyle::Interleaved,
            dev,
        ),
    }
}

// ─── Linear helpers ─────────────────────────────────────────────────────────

/// SwiGLU MLP over quantized weights (dense layers and shared experts).
#[derive(Clone)]
struct Mlp {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
    /// SwiGLU clamp (`0`: none; see [`swiglu`]).
    limit: f64,
    /// Per-tensor byte-range handles for best-effort page prefetch, present
    /// only when the weights are borrowed from the model mapping.
    prefetch: Option<crate::residency::ExpertHandles>,
}

/// `down(silu(gate(x)) * up(x))` — the one SwiGLU body shared by dense
/// layers, shared experts and both forms of a routed expert.  A positive
/// `limit` first clamps `gate(x)` from above and `up(x)` to `±limit`
/// (GLM-5.3-Flash, llama.cpp `ggml_swiglu_clamp`).
fn swiglu(gate: &QMatMul, up: &QMatMul, down: &QMatMul, xs: &Tensor, limit: f64) -> Result<Tensor> {
    let (mut gate, mut up) = (gate.forward(xs)?, up.forward(xs)?);
    if limit > 0.0 {
        gate = gate.clamp(f64::NEG_INFINITY, limit)?;
        up = up.clamp(-limit, limit)?;
    }
    down.forward(&(candle_nn::ops::silu(&gate)? * up)?)
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        swiglu(&self.gate, &self.up, &self.down, xs, self.limit)
    }
}

impl crate::moe::Expert for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Mlp::forward(self, xs)
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

    /// The LoRA stack split at its latent `norm(a(x))` (which the sparse
    /// attention indexer also reads): `(latent, b(latent))`.
    fn forward_latent(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Lora { a, norm, b } => {
                let latent = norm.forward(&a.forward(xs)?)?;
                let q = b.forward(&latent)?;
                Ok((latent, q))
            }
            Self::Plain(_) => candle_core::bail!("sparse attention needs a LoRA Q projection"),
        }
    }
}

// ─── Attention ───────────────────────────────────────────────────────────────

// The layer KV cache ([`KvCache`]) holds the *reconstructed* per-head K/V
// for MLA rather than the compressed latent.  The latent is only
// `kv_lora_rank + qk_rope` ≈ 576 elems/token vs `n_head·(qk_nope +
// v_head_dim)` ≈ 40,960 for the full per-head K/V, so caching it would save
// ~70x memory — but reconstructing the full K/V from it every forward is
// O(seq) work per step, which makes decode degrade linearly with context
// length.  Instead K/V is reconstructed for the *new* tokens only (a linear
// map over the latent) and appended: decode stays O(1) in reconstruction, at
// the cost of caching the ~8x larger per-head form (still far below a plain
// MHA model, since MLA keeps Q and the latent low-rank).

/// A layer's attention block: MLA (`deepseek2`), GQA (`deepseek`) or Kimi
/// Delta Attention (`glm5next`'s linear layers).
enum Attention {
    Mla(MlaAttention),
    Gqa(GqaAttention),
    Kda(Kda),
}

impl Attention {
    fn forward(
        &self,
        kv_cache: &mut KvCache,
        xs: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        match self {
            Self::Mla(a) => a.forward(kv_cache, xs, mask, offset),
            Self::Gqa(a) => a.forward(kv_cache, xs, mask, offset),
            Self::Kda(_) => unreachable!("KDA layers run through their recurrent state"),
        }
    }
}

/// Grouped-query attention with full-width interleaved RoPE (DeepSeek-MoE,
/// llama.cpp `LLM_ARCH_DEEPSEEK`).
struct GqaAttention {
    q: QMatMul,
    k: QMatMul,
    v: QMatMul,
    o_proj: QMatMul,
    rotary: Arc<Rope>,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    v_head_dim: usize,
    softmax_scale: f64,
}

impl GqaAttention {
    fn forward(
        &self,
        kv_cache: &mut KvCache,
        xs: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;
        // Project and split into heads: [b, n, seq, d].
        let heads = |w: &QMatMul, n: usize, d: usize| -> Result<Tensor> {
            w.forward(xs)?.reshape((b, seq_len, n, d))?.transpose(1, 2)
        };
        let q = heads(&self.q, self.n_head, self.head_dim)?;
        let k = heads(&self.k, self.n_kv_head, self.head_dim)?;
        let v = heads(&self.v, self.n_kv_head, self.v_head_dim)?;
        let (q, k) = self.rotary.apply_pair(&q, &k, offset)?;
        let ctx = crate::attention::cached_attention_heads(kv_cache, &q, &k, &v, mask, self.softmax_scale)?;
        self.o_proj.forward(&ctx)
    }
}

/// Multi-head latent attention in the unabsorbed (full-MHA) form.
struct MlaAttention {
    q: QProj,
    kv_a_mqa: QMatMul,
    kv_a_norm: RmsNorm,
    /// Combined KV up-projection: kv_lora_rank → n_head*(qk_nope + v_head_dim).
    kv_b: KvB,
    o_proj: QMatMul,
    /// `None` for NoPE MLA (GLM-5.3-Flash: `qk_rope == 0`).
    rotary: Option<Arc<Rope>>,
    n_head: usize,
    kv_lora_rank: usize,
    qk_nope: usize,
    qk_rope: usize,
    v_head_dim: usize,
    q_head_dim: usize,
    softmax_scale: f64,
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

impl MlaAttention {
    fn forward(
        &self,
        kv_cache: &mut KvCache,
        xs: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        self.forward_q(kv_cache, xs, self.q.forward(xs)?, mask, offset)
    }

    /// [`Self::forward`] with the Q projection `q` (`[b, seq, n_head ·
    /// q_head_dim]`) already applied.
    fn forward_q(
        &self,
        kv_cache: &mut KvCache,
        xs: &Tensor,
        q: Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;

        // Q → [b, n_head, seq, q_head_dim], split into nope/rope slices.
        let q = q
            .reshape((b, seq_len, self.n_head, self.q_head_dim))?
            .transpose(1, 2)?;

        // Compressed KV: [b, seq, kv_lora_rank + qk_rope]; the trailing slice is
        // a single-head RoPE key shared (MQA-style) across all query heads.
        let compressed = self.kv_a_mqa.forward(xs)?;
        let kv_cmpr = compressed
            .narrow(D::Minus1, 0, self.kv_lora_rank)?
            .contiguous()?;
        // RoPE the *_pe slices, then reassemble Q (nope ‖ rope).  The single-head
        // key is RoPE'd here too; the cached copy is already post-RoPE.
        let rope = match &self.rotary {
            Some(rotary) => {
                let q_pe = q.narrow(D::Minus1, self.qk_nope, self.qk_rope)?;
                let k_pe = compressed
                    .narrow(D::Minus1, self.kv_lora_rank, self.qk_rope)?
                    .reshape((b, seq_len, 1, self.qk_rope))?
                    .transpose(1, 2)?;
                Some(rotary.apply_pair(&q_pe, &k_pe, offset)?)
            }
            None => None,
        };
        let q = match &rope {
            Some((q_pe, _)) => {
                let q_nope = q.narrow(D::Minus1, 0, self.qk_nope)?;
                Tensor::cat(&[&q_nope.contiguous()?, &q_pe.contiguous()?], D::Minus1)?
            }
            None => q.contiguous()?,
        };

        // Reconstruct per-head K/V for the *new* tokens only (see the cache note
        // above).
        // kv_a_norm is a per-row norm and kv_b a linear map, so applying them
        // to this step's latent is bit-identical to a whole-cache
        // reconstruction — just O(seq_len) instead of O(seq_total) per step.
        let kv = self
            .kv_b
            .forward(&self.kv_a_norm.forward(&kv_cmpr)?)?
            .reshape((b, seq_len, self.n_head, self.qk_nope + self.v_head_dim))?
            .transpose(1, 2)?; // [b, n_head, seq_len, qk_nope + v_head_dim]
        let k_nope = kv.narrow(D::Minus1, 0, self.qk_nope)?; // [b, n_head, seq_len, qk_nope]
        let v = kv.narrow(D::Minus1, self.qk_nope, self.v_head_dim)?; // [b, n_head, seq_len, v_head_dim]
        // The single-head RoPE'd key is shared (MQA-style) across query heads.
        let k_new = match &rope {
            Some((_, k_pe)) => {
                let k_pe = k_pe.broadcast_as((b, self.n_head, seq_len, self.qk_rope))?;
                Tensor::cat(&[&k_nope.contiguous()?, &k_pe], D::Minus1)? // [b, n_head, seq_len, q_head_dim]
            }
            None => k_nope.contiguous()?,
        };

        let ctx = cached_attention(
            kv_cache,
            &q,
            k_new.transpose(2, 3)?.contiguous()?,
            v.transpose(2, 3)?.contiguous()?,
            mask,
            self.softmax_scale,
        )?;
        self.o_proj.forward(&ctx)
    }
}

// ─── Mixture of experts (DeepSeek routing + shared experts) ──────────────────

/// The device-resident form of one routed expert (deepseek2): the same
/// gate/up/down `QMatMul`s uploaded to the expert device, as an opaque
/// [`crate::residency::DeviceExpertSlot`] for the slot pool.  Forwarding
/// mirrors [`Mlp::forward`].
struct DeviceExpert {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
    limit: f64,
    bytes: u64,
}

impl crate::residency::DeviceExpertSlot for DeviceExpert {
    fn device_bytes(&self) -> u64 {
        self.bytes
    }
}

impl DeviceExpert {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        swiglu(&self.gate, &self.up, &self.down, xs, self.limit)
    }
}

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
    /// This MoE block's layer index (for the residency key).
    layer: u32,
    /// `general.architecture`, for error context.
    arch: &'static str,
    /// Device the routed-expert weights live on.  The model device when the
    /// experts were uploaded; the CPU when the pool stays in host RAM on an
    /// accelerator (see [`crate::placement::ExpertPlacement`]).  `dispatch`
    /// moves the block's input across once and its result back once.
    expert_device: Device,
    /// Bounded VRAM expert cache (#62): see `qwen3moe::Moe::residency`.
    residency: Option<std::sync::Arc<crate::residency::DeviceResidency<DeviceExpert>>>,
}

impl Moe {
    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Vec<u32>)> {
        let (b, seq_len, h) = xs.dims3()?;
        let n_tokens = b * seq_len;
        let x2 = xs.reshape((n_tokens, h))?;

        // Router logits → per-expert scores; selection adds the aux-loss-free
        // bias and the group limit, the weights use the unbiased scores.
        let probs = self.gating.scores(&x2.matmul(&self.gate_t)?)?; // [n_tokens, n_expert]
        let (topk_idx, weights) = crate::moe::route(
            &probs,
            self.gate_bias.as_ref(),
            self.n_expert_used,
            self.weights_norm,
            self.weights_scale,
            |sel| self.group_limit(&sel, n_tokens),
        )?;

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
            Gating::Softmax => grouped.max(D::Minus1)?,
            _ => topk_values(&grouped, 2)?.sum(D::Minus1)?,
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
        // Residency-aware per-expert forward: on a `DeviceResidency` hit run the
        // uploaded weights on the expert device; on a miss run the host form.
        // `None` residency keeps today's single-device path byte-for-byte.
        let fwd: crate::moe::ResidencyForward<'_> = match &self.residency {
            Some(res) => Some(&(|e: usize, x: &Tensor| {
                if let Some(dev) = res.lookup(self.layer, e as u32) {
                    let xd = crate::moe::on_device(x, &self.expert_device)?;
                    let out = dev.forward(&xd)?;
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
    fn forward_routed(&self, xs: &Tensor) -> Result<(Tensor, Vec<u32>)> {
        match self {
            Self::Dense(m) => Ok((m.forward(xs)?, Vec::new())),
            Self::Moe(m) => m.forward(xs),
        }
    }
}

/// How a layer's attention picks its keys.
enum Sparse {
    /// Every causally visible key.
    Dense,
    /// Its own lightning indexer (`glm-dsa` full layers).  `publish` when
    /// later layers share the selection.
    Full { indexer: Indexer, publish: bool },
    /// The selection published by full layer `source`; `last` for the final
    /// layer sharing it, which retires it.
    Shared { source: usize, last: bool },
}

struct Layer {
    attn_norm: RmsNorm,
    attn: Attention,
    sparse: Sparse,
    ffn_norm: RmsNorm,
    ffn: FeedForward,
    /// The attention and FFN hyper-connections (`glm5next`); `None`: a plain
    /// pre-norm residual.
    hyper: Option<[HyperConnection; 2]>,
}

/// One layer's per-session state.
#[derive(Clone, Default)]
pub struct LayerState {
    kv: KvCache,
    /// Sparse-attention indexer state (full layers).
    index: IndexCache,
    /// Kimi Delta Attention state (`glm5next` linear layers).
    kda: Option<KdaState>,
    /// Selections a full layer has published and its sharers have not yet
    /// all read, oldest first (one per input chunk; see
    /// [`crate::native_session::LayerStack::layer`]).
    published: Vec<Selection>,
}

/// The immutable half of a loaded model: every weight, the expert residency
/// backend and the device.  Sessions ([`ModelWeights`]) share one `Arc` of it
/// and own only their KV caches, so a second concurrent conversation costs
/// its KV cache rather than a second copy of the weights — on an
/// accelerator, a second upload of the whole model.
pub struct Weights {
    tok_embeddings: TokenEmbedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    output: QMatMul,
    device: Device,
    /// Executes residency for the hot set (CPU madvise today; a device slot
    /// cache later).  Built once at load from the per-expert handles.
    residency: Arc<dyn crate::residency::ExpertResidency>,
    n_expert: usize,
    /// Hyper-connection `(streams, Sinkhorn iterations, eps)` and the RMS
    /// eps the mixers normalise with (`glm5next`).
    hc: Option<(usize, usize, f64)>,
    rms_eps: f64,
}

impl Weights {
    /// The attention sublayer over its normalised input `x`.
    fn attend(
        &self,
        l: usize,
        states: &mut [LayerState],
        x: &Tensor,
        input: &crate::native_session::LayerInput<'_>,
    ) -> Result<Tensor> {
        let layer = &self.layers[l];
        let offset = input.offset;
        match (&layer.attn, &layer.sparse) {
            (Attention::Kda(k), _) => k.forward(&mut states[l].kda, x),
            (Attention::Mla(a), Sparse::Full { .. } | Sparse::Shared { .. }) => {
                let (latent, q) = a.q.forward_latent(x)?;
                let selection = match &layer.sparse {
                    Sparse::Full { indexer, publish } => {
                        let sel = indexer.select(&mut states[l].index, x, &latent, offset)?;
                        if *publish {
                            states[l].published.push(sel.clone());
                        }
                        sel
                    }
                    Sparse::Shared { source, last } => {
                        let published = &mut states[*source].published;
                        let Some(i) = published.iter().position(|s| s.offset == offset) else {
                            candle_core::bail!(
                                "layer {l} found no sparse-attention selection from layer {source} at \
                                 position {offset}"
                            );
                        };
                        if *last {
                            published.remove(i)
                        } else {
                            published[i].clone()
                        }
                    }
                    Sparse::Dense => unreachable!(),
                };
                let sparse = selection.mask(x.device())?;
                a.forward_q(&mut states[l].kv, x, q, sparse.as_ref().or(input.mask), offset)
            }
            (attn, _) => attn.forward(&mut states[l].kv, x, input.mask, offset),
        }
    }
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

    fn lift(&self, emb: Tensor) -> Result<Tensor> {
        match self.hc {
            // The streams start as identical copies of the embedding.
            Some((hc, ..)) => {
                let (b, t, n) = emb.dims3()?;
                emb.unsqueeze(2)?.broadcast_as((b, t, hc, n))?.reshape((b, t, hc * n))
            }
            None => Ok(emb),
        }
    }

    fn layer(
        &self,
        l: usize,
        states: &mut [LayerState],
        xs: &Tensor,
        input: &crate::native_session::LayerInput<'_>,
    ) -> Result<(Tensor, Vec<u32>)> {
        let layer = &self.layers[l];
        let (Some([hc_attn, hc_ffn]), Some((hc, iters, eps))) = (&layer.hyper, self.hc) else {
            let h = self.attend(l, states, &layer.attn_norm.forward(xs)?, input)?;
            let xs = (xs + h)?;
            let (h, routed) = layer.ffn.forward_routed(&layer.ffn_norm.forward(&xs)?)?;
            return Ok(((xs + h)?, routed));
        };
        // Each sublayer reads the streams collapsed by its mixer and writes
        // back through it.
        let (b, t, wide) = xs.dims3()?;
        let res = xs.reshape((b, t, hc, wide / hc))?;
        let (x, mix) = hc_attn.enter(&res, self.rms_eps, iters, eps)?;
        let h = self.attend(l, states, &layer.attn_norm.forward(&x)?, input)?;
        let res = crate::mhc::post(&h, &res, &mix.post, &mix.comb)?;
        let (x, mix) = hc_ffn.enter(&res, self.rms_eps, iters, eps)?;
        let (h, routed) = layer.ffn.forward_routed(&layer.ffn_norm.forward(&x)?)?;
        let out = crate::mhc::post(&h, &res, &mix.post, &mix.comb)?;
        Ok((out.reshape((b, t, wide))?, routed))
    }

    fn head(&self, xs: &Tensor) -> Result<Tensor> {
        // GLM-5.3-Flash collapses the streams by their plain mean.
        let xs = match self.hc {
            Some((hc, ..)) => {
                let (b, t, wide) = xs.dims3()?;
                xs.reshape((b, t, hc, wide / hc))?.mean(2)?
            }
            None => xs.clone(),
        };
        self.output.forward(&self.norm.forward(&xs)?)?.to_dtype(DType::F32)
    }

    fn can_truncate(&self) -> bool {
        !self.layers.iter().any(|l| matches!(l.attn, Attention::Kda(_)))
    }

    fn truncate_state(state: &mut LayerState, keep: usize) -> Result<()> {
        if keep == 0 {
            *state = LayerState::default();
            return Ok(());
        }
        state.published.clear();
        state.index.truncate(keep)?;
        crate::moe::truncate_kv(&mut state.kv, keep, crate::attention::KV_SEQ_DIM)
    }
}

/// A quantized DeepSeek-MoE / V2 / V3 / Kimi-K2 model loaded from GGUF: one
/// session over shared [`Weights`].
pub type ModelWeights = crate::native_session::Session<Weights>;

/// Small GGUF reader over the memory-mapped file.
struct Reader<R: Read + Seek> {
    /// The tensors themselves (raw-header formats, mmap borrow, or copy),
    /// with the dense `device` and the architecture name (`deepseek` or
    /// `deepseek2`).
    src: TensorSource<R>,
    /// Device the routed-expert tensors are built on.  Same as `device` for
    /// a CPU model or when the experts are uploaded; the CPU when the model
    /// runs on an accelerator that cannot hold the expert pool (see
    /// [`crate::placement::ExpertPlacement`]) — each expert is then borrowed
    /// from the mapping and [`Moe::dispatch`] hops the activations across.
    expert_device: Device,
    /// Bytes of the bounded VRAM expert cache (#62) to build for each MoE
    /// block (`None` / `Some(0)` disables); `load_moe` reads it.
    device_expert_cache_bytes: Option<u64>,
}

impl<R: Read + Seek> std::ops::Deref for Reader<R> {
    type Target = TensorSource<R>;
    fn deref(&self) -> &TensorSource<R> {
        &self.src
    }
}

impl<R: Read + Seek> std::ops::DerefMut for Reader<R> {
    fn deref_mut(&mut self) -> &mut TensorSource<R> {
        &mut self.src
    }
}

impl<R: Read + Seek> Reader<R> {
    fn qmatmul_opt(&mut self, name: &str) -> Option<QMatMul> {
        if self.has(name) {
            self.qmatmul(name).ok()
        } else {
            None
        }
    }
    /// A dense SwiGLU block: `{p}.ffn_{gate,up,down}{suffix}.weight`
    /// (`suffix` is `""` for a dense layer, `"_shexp"` for shared experts).
    fn mlp(&mut self, p: &str, suffix: &str, limit: f64) -> Result<Mlp> {
        Ok(Mlp {
            gate: self.qmatmul(&format!("{p}.ffn_gate{suffix}.weight"))?,
            up: self.qmatmul(&format!("{p}.ffn_up{suffix}.weight"))?,
            down: self.qmatmul(&format!("{p}.ffn_down{suffix}.weight"))?,
            limit,
            prefetch: None,
        })
    }

    /// A sublayer's hyper-connection mixer `{p}_{fn,base,scale}`.
    fn hyper_connection(&mut self, p: &str) -> Result<HyperConnection> {
        Ok(HyperConnection {
            hc_fn: self.qmatmul(&format!("{p}_fn.weight"))?,
            base: self.f32_tensor(&format!("{p}_base.weight"))?,
            scale: self.f32_tensor(&format!("{p}_scale.weight"))?,
        })
    }

    /// Layer `p`'s attention block, MLA or GQA per `cfg`.
    fn attention(&mut self, p: &str, cfg: &Config, rotary: &Option<Arc<Rope>>) -> Result<Attention> {
        let o_proj = self.qmatmul(&format!("{p}.attn_output.weight"))?;
        if !cfg.is_mla() {
            let Some(rotary) = rotary else {
                candle_core::bail!("{}: GQA attention needs rotary dims", self.arch);
            };
            return Ok(Attention::Gqa(GqaAttention {
                q: self.qmatmul(&format!("{p}.attn_q.weight"))?,
                k: self.qmatmul(&format!("{p}.attn_k.weight"))?,
                v: self.qmatmul(&format!("{p}.attn_v.weight"))?,
                o_proj,
                rotary: rotary.clone(),
                n_head: cfg.n_head,
                n_kv_head: cfg.n_kv_head,
                head_dim: cfg.q_head_dim(),
                v_head_dim: cfg.v_head_dim,
                softmax_scale: cfg.softmax_scale,
            }));
        }
        // Q projection: LoRA (V2-full/V3/K2) or plain (V2-Lite).
        let q = if cfg.q_lora_rank.is_some() {
            QProj::Lora {
                a: self.qmatmul(&format!("{p}.attn_q_a.weight"))?,
                norm: self.rms_norm(&format!("{p}.attn_q_a_norm.weight"), cfg.rms_eps)?,
                b: self.qmatmul(&format!("{p}.attn_q_b.weight"))?,
            }
        } else {
            QProj::Plain(self.qmatmul(&format!("{p}.attn_q.weight"))?)
        };
        Ok(Attention::Mla(MlaAttention {
            q,
            kv_a_mqa: self.qmatmul(&format!("{p}.attn_kv_a_mqa.weight"))?,
            kv_a_norm: self.rms_norm(&format!("{p}.attn_kv_a_norm.weight"), cfg.rms_eps)?,
            kv_b: load_kv_b(self, p, cfg)?,
            o_proj,
            rotary: rotary.clone(),
            n_head: cfg.n_head,
            kv_lora_rank: cfg.kv_lora_rank,
            qk_nope: cfg.qk_nope_head_dim,
            qk_rope: cfg.qk_rope_head_dim,
            v_head_dim: cfg.v_head_dim,
            q_head_dim: cfg.q_head_dim(),
            softmax_scale: cfg.softmax_scale,
        }))
    }
}

impl ModelWeights {
    /// Load a `deepseek` (DeepSeek-MoE) or `deepseek2` (DeepSeek-V2/V3,
    /// Kimi-K2) GGUF.
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
    /// trillion-parameter MoE being unloadable and merely slow.  The routed
    /// experts go to `device` too; see
    /// [`ModelWeights::from_gguf_mmap_placed`] to keep them in host RAM on an
    /// accelerator.
    pub fn from_gguf_mmap<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        mmap: Option<std::sync::Arc<memmap2::Mmap>>,
    ) -> Result<Self> {
        Self::from_gguf_mmap_placed(ct, None, reader, device, device, mmap, None)
    }

    /// [`ModelWeights::from_gguf_mmap`] with an explicit device for the
    /// routed experts.
    ///
    /// The dense set always goes to `device`.  With `expert_device` the CPU
    /// on an accelerator model, the routed-expert pool stays in host RAM —
    /// borrowed in place from `mmap` (or read onto the heap without one) and
    /// run through the CPU expert kernels, with the MoE block's activations
    /// hopping across the bus once per layer.  That is the layout for a
    /// model larger than the device's memory (see
    /// [`crate::placement::ExpertPlacement`]).  Any other `expert_device`
    /// must equal `device`.
    ///
    /// `raw` is the header with raw dtype ids (see [`crate::gguf_ext`]),
    /// carrying the tensors in formats candle cannot name.
    pub fn from_gguf_mmap_placed<R: Read + Seek>(
        ct: gguf_file::Content,
        raw: Option<crate::gguf_ext::GgufHeader>,
        reader: &mut R,
        device: &Device,
        expert_device: &Device,
        mmap: Option<std::sync::Arc<memmap2::Mmap>>,
        device_expert_cache_bytes: Option<u64>,
    ) -> Result<Self> {
        let arch = match crate::model::Architecture::arch_name(&ct.metadata).as_deref() {
            Some("deepseek") => "deepseek",
            Some("glm-dsa") => "glm-dsa",
            // llama.cpp's open GLM-5.3-Flash pull requests name the
            // architecture differently; the files are otherwise the same.
            Some("glm5next") => "glm5next",
            Some("glm5-next") => "glm5-next",
            _ => "deepseek2",
        };
        if !expert_device.is_cpu() && !expert_device.same_device(device) {
            candle_core::bail!(
                "{arch}: routed experts must live on the model device or the CPU, not {expert_device:?}"
            );
        }
        let cfg = Config::from_metadata(&ct.metadata, arch)?;
        // The Content owns metadata; move it into our reader together with the
        // underlying file handle (borrowed for the lifetime of the load).
        let mut rd = Reader {
            src: TensorSource {
                ct,
                raw,
                reader,
                device: device.clone(),
                mmap,
                arch,
            },
            expert_device: expert_device.clone(),
            device_expert_cache_bytes,
        };

        // Kept quantized: dequantizing the table to f32 costs vocab × hidden
        // × 4 bytes of anonymous memory per instance (3.7 GiB on V3, 4.7 GiB
        // on Kimi-K2).
        let tok_embeddings = TokenEmbedding::load(rd.qtensor("token_embd.weight")?, device)?;
        let norm = rd.rms_norm("output_norm.weight", cfg.rms_eps)?;
        let output = match rd.qmatmul_opt("output.weight") {
            Some(o) => o,
            None => rd.qmatmul("token_embd.weight")?, // tied
        };

        let rotary = (cfg.qk_rope_head_dim > 0)
            .then(|| rope_table(&cfg, device).map(Arc::new))
            .transpose()?;
        // The sparse-attention role of each MLA layer: full layers run their
        // indexer, shared ones reuse the nearest earlier full layer's
        // selection (skipping any KDA layers in between).
        let mla_layers: Vec<usize> = (0..cfg.n_layer).filter(|&i| !cfg.recurrent[i]).collect();

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = format!("blk.{i}");
            let attn_norm = rd.rms_norm(&format!("{p}.attn_norm.weight"), cfg.rms_eps)?;
            let ffn_norm = rd.rms_norm(&format!("{p}.ffn_norm.weight"), cfg.rms_eps)?;

            let attn = match &cfg.kda {
                Some(k) if cfg.recurrent[i] => Attention::Kda(Kda::load(&mut rd, &p, k, cfg.rms_eps)?),
                _ => rd.attention(&p, &cfg, &rotary)?,
            };
            let sparse = match &cfg.dsa {
                Some(d) if !cfg.recurrent[i] => {
                    let at = mla_layers.iter().position(|&j| j == i).expect("an MLA layer");
                    let next_shares = mla_layers.get(at + 1).is_some_and(|&j| !d.full[j]);
                    if d.full[i] {
                        Sparse::Full {
                            indexer: Indexer::load(&mut rd, &p, d, rotary.clone())?,
                            publish: next_shares,
                        }
                    } else {
                        let Some(&source) = mla_layers[..at].iter().rev().find(|&&j| d.full[j]) else {
                            candle_core::bail!(
                                "{}: layer {i} shares a sparse-attention selection but no earlier layer \
                                 makes one",
                                rd.arch
                            );
                        };
                        Sparse::Shared { source, last: !next_shares }
                    }
                }
                _ => Sparse::Dense,
            };

            let ffn = if cfg.n_expert > 0 && i >= cfg.leading_dense {
                FeedForward::Moe(load_moe(&mut rd, &p, &cfg, i)?)
            } else {
                FeedForward::Dense(rd.mlp(&p, "", cfg.swiglu_shexp[i])?)
            };
            let hyper = match cfg.hc {
                Some(_) => Some([
                    rd.hyper_connection(&format!("{p}.hc_attn"))?,
                    rd.hyper_connection(&format!("{p}.hc_ffn"))?,
                ]),
                None => None,
            };

            layers.push(Layer {
                attn_norm,
                attn,
                sparse,
                ffn_norm,
                ffn,
                hyper,
            });
        }

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
        Ok(Self::from_weights(Weights {
            tok_embeddings,
            layers,
            norm,
            output,
            device: device.clone(),
            residency,
            n_expert: cfg.n_expert,
            hc: cfg.hc,
            rms_eps: cfg.rms_eps,
        }))
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
fn load_moe<R: Read + Seek>(rd: &mut Reader<R>, p: &str, cfg: &Config, il: usize) -> Result<Moe> {
    let gate = rd.f32_tensor(&format!("{p}.ffn_gate_inp.weight"))?; // [n_expert, n_embd]
    let gate_bias = rd
        .has(&format!("{p}.exp_probs_b.bias"))
        .then(|| rd.f32_tensor(&format!("{p}.exp_probs_b.bias")))
        .transpose()?;

    // Slice the 3-D expert tensors into per-expert quantized QMatMuls.
    let gate_exps = split_experts(rd, &format!("{p}.ffn_gate_exps.weight"), cfg.n_expert)?;
    let up_exps = split_experts(rd, &format!("{p}.ffn_up_exps.weight"), cfg.n_expert)?;
    let down_exps = split_experts(rd, &format!("{p}.ffn_down_exps.weight"), cfg.n_expert)?;
    let experts: Vec<Mlp> = gate_exps
        .into_iter()
        .zip(up_exps)
        .zip(down_exps)
        .map(|((gate, up), down)| Mlp {
            gate: gate.qmatmul,
            up: up.qmatmul,
            down: down.qmatmul,
            limit: cfg.swiglu_exp[il],
            prefetch: crate::residency::ExpertHandles::from_parts(
                gate.prefetch,
                up.prefetch,
                down.prefetch,
            ),
        })
        .collect();

    let shared = if cfg.n_expert_shared > 0 {
        Some(rd.mlp(p, "_shexp", cfg.swiglu_shexp[il])?)
    } else {
        None
    };

    let layer = p
        .rsplit('.')
        .next()
        .and_then(|id| id.parse::<u32>().ok())
        .unwrap_or(0);

    // Build the bounded VRAM expert cache (#62) when a byte budget is given.
    // See qwen3moe::load_moe: on the CPU (this box) the "device form" wraps the
    // already-loaded host weights (identical numbers) so a hit == the host path;
    // on a real accelerator the device branch uploads the expert's bytes.
    let residency = match rd.device_expert_cache_bytes {
        Some(cap_bytes) if cap_bytes > 0 => {
            let host = experts.clone();
            let expert_dev = rd.expert_device.clone();
            let per_slot = crate::moe::EXPERT_SLOT_BYTES_ESTIMATE;
            let upload: std::sync::Arc<
                dyn Fn(u32, u32) -> Option<std::sync::Arc<DeviceExpert>> + Send + Sync,
            > = std::sync::Arc::new(move |l, e| {
                if l != layer {
                    return None;
                }
                let m: &Mlp = host.get(e as usize)?;
                // Non-CPU expert device: upload each expert's quantized blocks
                // onto it (backend-generic via QStorage::from_data) so a hit
                // runs for real on the accelerator (#62). CPU keeps sharing host
                // weights (parity).
                if !expert_dev.is_cpu() {
                    let gate = crate::moe::upload_qmatmul(&m.gate, &expert_dev)?;
                    let up = crate::moe::upload_qmatmul(&m.up, &expert_dev)?;
                    let down = crate::moe::upload_qmatmul(&m.down, &expert_dev)?;
                    return Some(std::sync::Arc::new(DeviceExpert {
                        gate,
                        up,
                        down,
                        limit: m.limit,
                        bytes: per_slot,
                    }));
                }
                Some(std::sync::Arc::new(DeviceExpert {
                    gate: m.gate.clone(),
                    up: m.up.clone(),
                    down: m.down.clone(),
                    limit: m.limit,
                    bytes: per_slot,
                }))
            });
            Some(std::sync::Arc::new(crate::residency::DeviceResidency::<DeviceExpert>::new(
                cap_bytes,
                per_slot,
                upload,
            )))
        }
        _ => None,
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
        layer,
        arch: rd.arch,
        expert_device: rd.expert_device.clone(),
        residency,
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
    let et = rd.expert_tensor(name, n_expert)?;
    let host_experts = rd.expert_device.is_cpu();
    let uploaded = |qt: QTensor| -> Result<ExpertWeight> {
        Ok(ExpertWeight {
            qmatmul: QMatMul::from_qtensor(qt)?,
            prefetch: None,
        })
    };

    // Host-resident experts with a mapping: borrow each expert's slice of the
    // mapping (see `ExpertTensor::borrow_host`).  On an accelerator this is
    // the `ExpertPlacement::Host` layout: dense set on the device, experts in
    // host RAM, activations hopping across in `Moe::dispatch`.
    if host_experts {
        if let Some(mmap) = rd.mmap.as_ref() {
            if let Some(borrowed) = et.borrow_host(mmap)? {
                return borrowed
                    .into_iter()
                    .map(|b| {
                        Ok(ExpertWeight {
                            qmatmul: QMatMul::from_qtensor(b.tensor)?,
                            prefetch: b.prefetch,
                        })
                    })
                    .collect();
            }
        }
    }

    // Device-resident experts with a mapping: upload each expert straight
    // from its bytes in the mapping (see `ExpertTensor::upload_from_mmap`).
    if !host_experts {
        if let Some(mmap) = rd.mmap.as_ref() {
            if let Some(tensors) = et.upload_from_mmap(mmap, &rd.expert_device)? {
                return tensors.into_iter().map(uploaded).collect();
            }
        }
    }

    // No mapping (streamed load) or a tensor that cannot be sliced.
    et.read_and_split(&rd.src.ct, &mut rd.src.reader, rd.src.arch, &rd.expert_device)?
        .into_iter()
        .map(uploaded)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::GgmlDType;

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
            n_kv_head: w.nh,
            rms_eps: 1e-5,
            q_lora_rank: None,
            kv_lora_rank: w.lkv,
            qk_nope_head_dim: w.np,
            qk_rope_head_dim: w.r,
            v_head_dim: w.vh,
            softmax_scale: 1.0 / (w.qd as f64).sqrt(),
            rope_theta: 10_000.0,
            rope_freq_scale: 1.0,
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
            dsa: None,
            recurrent: vec![false],
            kda: None,
            hc: None,
            swiglu_exp: vec![0.0],
            swiglu_shexp: vec![0.0],
        };
        Ok(Attention::Mla(MlaAttention {
            q: QProj::Plain(lin(w.nh * w.qd, w.h, &w.q)),
            kv_a_mqa: lin(w.lkv + w.r, w.h, &w.kv_a),
            kv_a_norm: norm,
            kv_b: KvB::Dense(w.kv_b.clone()),
            o_proj: lin(w.h, w.nh * w.vh, &w.o),
            rotary: Some(Arc::new(rope_table(&cfg, dev)?)),
            n_head: w.nh,
            kv_lora_rank: w.lkv,
            qk_nope: w.np,
            qk_rope: w.r,
            v_head_dim: w.vh,
            q_head_dim: w.qd,
            softmax_scale: cfg.softmax_scale,
        }))
    }

    /// The cache must hold the reconstructed per-head K/V (appended
    /// incrementally), not the compressed latent — decode then costs O(1)
    /// reconstruction per step instead of the O(seq) full rebuild.
    #[test]
    fn mla_cache_stores_reconstructed_per_head_kv() -> Result<()> {
        let dev = Device::Cpu;
        let w = tiny_weights(&dev)?;
        let attn = tiny_attention(&dev, &w)?;
        let mut kv: KvCache = None;

        // Prefill: 3 tokens at once.
        let xs = Tensor::randn(0f32, 1f32, (1, 3, 8), &dev)?;
        attn.forward(&mut kv, &xs, None, 0)?;
        let (k, v) = kv.as_ref().expect("cache populated after prefill");
        assert_eq!(
            k.dims(),
            &[1, 2, 8, 3],
            "k cache must be [b, n_head, qk_nope + qk_rope, seq]"
        );
        assert_eq!(v.dims(), &[1, 2, 4, 3], "v cache must be [b, n_head, v_head_dim, seq]");

        // Decode: one more token appends along seq.
        let xs2 = Tensor::randn(0f32, 1f32, (1, 1, 8), &dev)?;
        attn.forward(&mut kv, &xs2, None, 3)?;
        let (k, v) = kv.as_ref().expect("cache after decode");
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
        let a = tiny_attention(&dev, &w)?;
        let mut kv_a: KvCache = None;
        let xs = Tensor::randn(0f32, 1f32, (1, 3, 8), &dev)?;
        let mask: Vec<f32> = (0..3)
            .flat_map(|i| (0..3).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        let mask = Tensor::from_slice(&mask, (1, 1, 3, 3), &dev)?;
        let out_prefill = a.forward(&mut kv_a, &xs, Some(&mask), 0)?;
        let (c_pre, k_pre) = kv_a.as_ref().unwrap();

        // Incremental path: 1 + 1 + 1 tokens with growing offset.
        let b = tiny_attention(&dev, &w)?;
        let mut kv_b: KvCache = None;
        let mut out_inc = Vec::new();
        for off in 0..3 {
            let t = xs.narrow(1, off, 1)?;
            out_inc.push(b.forward(&mut kv_b, &t, None, off)?);
        }
        let out_inc = Tensor::cat(&out_inc, 1)?;
        let (c_inc, k_inc) = kv_b.as_ref().unwrap();

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
                limit: 0.0,
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
            layer: 0,
            arch: "deepseek2",
            expert_device: dev.clone(),
            residency: None,
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
