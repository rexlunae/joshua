//! Pure-Rust quantized loader for the Nemotron-H family (NVIDIA Nemotron 3.5).
//!
//! `nemotron_h` is a **hybrid Mamba-2 / Transformer MoE**: each layer is one of
//!
//! | layer kind  | mixer                            | FFN |
//! |---|---|---|
//! | `mamba`     | Mamba-2 selective scan           | — (the mixer *is* the block) |
//! | `attention` | GQA + RoPE                       | — |
//! | `moe`       | none (identity mix)              | sigmoid-routed MoE + shared expert |
//!
//! so the residual stream is advanced by *whichever* sub-block the layer owns:
//! a `mamba` layer contributes only a mixer, an `attention` layer only a mixer,
//! and an `moe` layer only an FFN.  The layer kind is derived from the GGUF
//! hyperparameters exactly as llama.cpp does it (`nemotron-h.cpp`): **a layer is
//! recurrent iff `attention.head_count_kv == 0 && feed_forward_length == 0`**;
//! otherwise it is an attention layer when `feed_forward_length == 0`, and an
//! MoE layer when it has a router.
//!
//! # Mamba-2 block
//!
//! Per token, from `x` (`[hidden]`):
//!
//! ```text
//! zxBCdt = ssm_in · x                       # one fused projection
//! z   = zxBCdt[0 .. d_inner]                # gate, reshaped [n_head, head_dim]
//! xBC = zxBCdt[d_inner .. d_inner+conv_dim] # conv input
//! dt  = zxBCdt[d_inner+conv_dim ..]         # per-head time step (raw)
//!
//! xBC = silu(conv1d(xBC) + conv_bias)       # causal depthwise conv, d_conv taps
//! x, B, C = split(xBC)                      # [head_dim,n_head], [d_state,n_group]×2
//! dt  = softplus(dt + dt_bias)
//! A   = -exp(ssm_a)                         # per head
//! h_t = exp(dt·A) · h_{t-1} + dt·B ⊗ x      # per-head state [head_dim, d_state]
//! y   = C · h_t + ssm_d · x
//! y   = z · silu(y)                         # "swiglu_split" gate
//! y   = grouped RMSNorm(y)                  # over (d_inner/n_group, n_group)
//! out = ssm_out · y
//! ```
//!
//! The recurrence is [`crate::scan::ssm_scan`], which is written as a chunked
//! associative scan so it is GEMM-bound (and therefore worth running on an
//! accelerator) rather than a per-token loop.
//!
//! # Relation to the other loaders
//!
//! The recurrent state has the same *shape of problem* as Qwen3-Next's Gated
//! DeltaNet (see [`crate::quantized_qwen`]), so the state layout, the
//! conv-tail handling and the snapshot/restore protocol for speculative
//! rollback follow that loader.  The recurrence itself is different: Mamba-2 is
//! a linear state-space scan with scalar-per-head decay, not a delta rule.
//!
//! Activations run in f32, mirroring the other quantized loaders.

use std::io::{Read, Seek};
use std::sync::Arc;

use candle_core::quantized::{gguf_file, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::ops::silu;
use candle_transformers::quantized_nn::RmsNorm;

use crate::attention::{KvCache, Rope, RopeStyle};
use crate::gguf_ext::GgufHeader;
use crate::gguf_meta::Meta;
use crate::moe::Gating;
use crate::native_session::{LayerInput, LayerStack};
use crate::tensor_source::TensorSource;
use crate::token_embedding::TokenEmbedding;

/// `general.architecture` values this loader serves.
pub const ARCHES: &[&str] = &["nemotron_h", "nemotron_h_moe"];

// ─── Configuration ───────────────────────────────────────────────────────────

/// Mamba-2 (`ssm.*`) dimensions.
#[derive(Debug, Clone)]
struct SsmConfig {
    /// Taps in the causal depthwise conv (`ssm.conv_kernel`, 4 on Nemotron 3.5).
    d_conv: usize,
    /// Per-head state width (`ssm.state_size`, 128).
    d_state: usize,
    /// `d_inner = mamba_num_heads * mamba_head_dim` (`ssm.inner_size`).
    d_inner: usize,
    /// Number of SSM heads (`ssm.time_step_rank`).
    n_head: usize,
    /// B/C groups (`ssm.group_count`); each B/C row is shared by
    /// `n_head / n_group` heads.
    n_group: usize,
    /// Precomputed `-(ssm_a)` per head, i.e. the negative decay rate.
    /// `None` until [`Self::with_a`]; the loaded value lives on the layer.
    time_step_min: f32,
}

impl SsmConfig {
    fn head_dim(&self) -> usize {
        self.d_inner / self.n_head
    }
    /// `d_inner + 2 * n_group * d_state` — the conv/projection width.
    fn conv_dim(&self) -> usize {
        self.d_inner + 2 * self.n_group * self.d_state
    }
    /// The fused projection's output width: `[z ‖ xBC ‖ dt]`.
    fn d_in_proj(&self) -> usize {
        2 * self.d_inner + 2 * self.n_group * self.d_state + self.n_head
    }
    /// The group `head h` reads its `B`/`C` from.
    fn group(&self, h: usize) -> usize {
        h / (self.n_head / self.n_group)
    }
}

/// A per-layer key that may also be a single scalar applying to every layer.
///
/// llama.cpp accepts both encodings for `attention.head_count_kv` and
/// `feed_forward_length`; a uniform model writes one `U32`, Nemotron-H writes
/// an array.  Reading a scalar through [`Meta::array_u32`] yields zeros, which
/// would classify every layer as recurrent, so the scalar is replicated here.
fn per_layer_or_scalar(m: &Meta<'_>, suffix: &str, n: usize) -> Result<Vec<usize>> {
    match m.get(suffix) {
        Some(gguf_file::Value::Array(_)) => Ok(m.array_u32(suffix, n)),
        Some(_) => match m.u32_opt(suffix) {
            Some(scalar) => Ok(vec![scalar as usize; n]),
            None => candle_core::bail!(
                "{}: `{suffix}` must be an integer or an integer array",
                m.arch()
            ),
        },
        None => Ok(vec![0; n]),
    }
}

/// How a layer advances the residual stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerKind {
    /// Mamba-2 mixer only.
    Mamba,
    /// GQA + RoPE mixer only.
    Attention,
    /// MoE FFN only (no mixer).
    Moe,
}

/// One layer's declared shape, in the order the GGUF declares them.
#[derive(Debug, Clone)]
struct LayerSpec {
    kind: LayerKind,
    /// `n_head_kv == 0` when the layer is recurrent (used to *derive* `kind`).
    n_head_kv: usize,
    n_ff: usize,
}

/// The model geometry.
#[derive(Debug, Clone)]
struct Config {
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    n_rot: usize,
    rope_theta: f32,
    /// The RoPE table is precomputed for this many positions, so a full
    /// `context_length` pass never runs off the end of it.
    context_length: usize,
    rms_eps: f64,
    n_vocab: usize,
    n_expert: usize,
    n_expert_used: usize,
    /// `routed_scaling_factor` (2.5 on Nemotron 3.5).
    expert_weights_scale: f64,
    expert_weights_norm: bool,
    /// Latent MoE width (`moe_latent_size`); `None` = no latent projection.
    moe_latent: Option<usize>,
    ssm: SsmConfig,
    layers: Vec<LayerSpec>,
}

impl Config {
    fn from_meta(m: &Meta<'_>) -> Result<Option<Self>> {
        if !ARCHES.contains(&m.arch()) {
            return Ok(None);
        }
        let n_layer = m.u32("block_count")? as usize;
        let n_embd = m.u32("embedding_length")? as usize;
        let n_head = m.u32("attention.head_count")? as usize;
        let head_dim = m
            .u32_opt("attention.key_length")
            .map(|v| v as usize)
            .unwrap_or(n_embd / n_head.max(1));
        let n_rot = m
            .u32_opt("rope.dimension_count")
            .map(|v| v as usize)
            .unwrap_or(head_dim);
        let d_inner = m.u32("ssm.inner_size")? as usize;
        let ssm = SsmConfig {
            d_conv: m.u32("ssm.conv_kernel")? as usize,
            d_state: m.u32("ssm.state_size")? as usize,
            d_inner,
            n_head: m.u32("ssm.time_step_rank")? as usize,
            n_group: m.u32("ssm.group_count")? as usize,
            time_step_min: 0.0,
        };
        if ssm.n_head == 0 || ssm.n_head % ssm.n_group != 0 {
            candle_core::bail!(
                "{}: ssm.time_step_rank {} must be a positive multiple of ssm.group_count {}",
                m.arch(),
                ssm.n_head,
                ssm.n_group
            );
        }
        if ssm.d_inner % ssm.n_head != 0 {
            candle_core::bail!(
                "{}: ssm.inner_size {d_inner} is not a multiple of ssm.time_step_rank {}",
                m.arch(),
                ssm.n_head
            );
        }

        // Per-layer `head_count_kv` / `feed_forward_length` may be scalars (the
        // uniform case) or arrays (Nemotron-H's default, and "Puzzle" models
        // that vary the expert FFN per layer).
        // Either key may be a **scalar** (a uniform model) or a per-layer
        // array (what Nemotron-H ships, and the only form that can express
        // `Puzzle`'s heterogeneous layers).  `Meta::array_u32` reads arrays
        // only and returns zeros for a scalar, which would silently make
        // every layer recurrent, so the scalar case is handled explicitly.
        let kv = per_layer_or_scalar(m, "attention.head_count_kv", n_layer)?;
        let n_ff = per_layer_or_scalar(m, "feed_forward_length", n_layer)?;
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let n_head_kv_i = kv.get(i).copied().unwrap_or(0);
            let n_ff_i = n_ff.get(i).copied().unwrap_or(0);
            // llama.cpp `nemotron-h.cpp`: recurrent iff n_head_kv == 0 && n_ff == 0.
            let kind = if n_head_kv_i == 0 && n_ff_i == 0 {
                LayerKind::Mamba
            } else if n_ff_i == 0 {
                LayerKind::Attention
            } else {
                LayerKind::Moe
            };
            layers.push(LayerSpec {
                kind,
                n_head_kv: n_head_kv_i,
                n_ff: n_ff_i,
            });
        }
        let n_attn = layers
            .iter()
            .filter(|l| l.kind == LayerKind::Attention)
            .count();
        // The uniform scalar `head_count_kv` is the only one present for the
        // attention rows; take it from the first attention layer.
        let n_head_kv = layers
            .iter()
            .find(|l| l.kind == LayerKind::Attention)
            .map(|l| l.n_head_kv)
            .unwrap_or_else(|| m.u32_or("attention.head_count_kv", 0) as usize);
        if n_attn > 0 && n_head_kv == 0 {
            candle_core::bail!("{}: attention layers declare 0 kv heads", m.arch());
        }

        Ok(Some(Self {
            n_layer,
            n_embd,
            n_head,
            n_head_kv,
            head_dim,
            n_rot,
            rope_theta: m.f32_or("rope.freq_base", 10_000.0),
            context_length: m.u32_or("context_length", 4096) as usize,
            rms_eps: m.f32_or("attention.layer_norm_rms_epsilon", 1e-5) as f64,
            n_vocab: m.u32_or("vocab_size", 0) as usize,
            n_expert: m.u32_or("expert_count", 0) as usize,
            n_expert_used: m.u32_or("expert_used_count", 0) as usize,
            expert_weights_scale: m.f32_or("expert_weights_scale", 0.0) as f64,
            expert_weights_norm: m.bool_or("expert_weights_norm", false),
            moe_latent: m.u32_opt("moe_latent_size").map(|v| v as usize),
            ssm,
            layers,
        }))
    }

    fn rope(&self, dev: &Device) -> Result<Rope> {
        let inv_freq = crate::attention::inv_freq(self.n_rot, self.rope_theta);
        Rope::from_inv_freq(inv_freq, self.context_length, 1.0, RopeStyle::Neox, dev)
    }
}

// ─── Weights ─────────────────────────────────────────────────────────────────

/// A quantized linear weight, in candle's own form.
#[derive(Clone)]
struct Weight(QMatMul);

impl Weight {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.0.forward(xs)
    }
}

/// A linear layer: weight plus an optional bias.
#[derive(Clone)]
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

/// The recurrent state of one Mamba-2 layer.
#[derive(Clone, Default)]
struct SsmState {
    /// The causal conv's tail, `[d_conv - 1, conv_dim]` (time-major).
    conv: Option<Tensor>,
    /// The scan's state, `n_head * head_dim * d_state` f32, head-major.
    scan: Vec<f32>,
}

/// The Mamba-2 mixer of one `mamba` layer.
struct Mamba2 {
    /// Fused `[z ‖ xBC ‖ dt]` projection.
    in_proj: Weight,
    /// Depthwise conv taps, `[conv_dim, d_conv]`.
    conv_w: Tensor,
    /// `[conv_dim]`, optional (`use_conv_bias`).
    conv_b: Option<Tensor>,
    /// `[n_head]`.
    dt_b: Tensor,
    /// `-[exp(ssm_a)]`, `[n_head]`.
    a: Tensor,
    /// `[n_head]` skip weight.
    d_skip: Tensor,
    /// Grouped RMS-norm weight, `[d_inner / n_group]` (llama.cpp stores it
    /// as `[d_inner/n_group, n_group]`, normalised over dim 0).
    norm_w: Tensor,
    eps: f64,
    out: Weight,
    cfg: SsmConfig,
}

impl Mamba2 {
    fn load<R: Read + Seek>(src: &mut TensorSource<R>, p: &str, cfg: &SsmConfig) -> Result<Self> {
        let conv_w = src
            .f32_tensor(&format!("{p}.ssm_conv1d.weight"))?
            .reshape((cfg.conv_dim(), cfg.d_conv))?;
        let conv_b = src.f32_opt(&format!("{p}.ssm_conv1d.bias"))?;
        // `ssm_a` ships as the raw log-rate `[1, n_head]`; the block wants
        // `-exp(a)` (llama.cpp negates at load).
        let a_raw = src
            .f32_tensor(&format!("{p}.ssm_a"))?
            .reshape((cfg.n_head,))?;
        let a = a_raw.exp()?.neg()?;
        let d_skip = src
            .f32_tensor(&format!("{p}.ssm_d"))?
            .reshape((cfg.n_head,))?;
        let dt_b = src
            .f32_tensor(&format!("{p}.ssm_dt.bias"))?
            .reshape((cfg.n_head,))?;
        // llama.cpp stores `ssm_norm` as `[d_inner/n_group, n_group]`: entry
        // `(i, g)` weighs element `i` of group `g`'s block.  The whole matrix
        // is kept — each group has its own column, and collapsing to column 0
        // would silently scale every group by the first one's weights.
        let norm_w = src
            .f32_tensor(&format!("{p}.ssm_norm.weight"))?
            .reshape((cfg.d_inner / cfg.n_group, cfg.n_group))?;
        Ok(Self {
            in_proj: Weight(src.qmatmul(&format!("{p}.ssm_in.weight"))?),
            conv_w,
            conv_b,
            dt_b,
            a,
            d_skip,
            norm_w,
            eps: 1e-5,
            out: Weight(src.qmatmul(&format!("{p}.ssm_out.weight"))?),
            cfg: cfg.clone(),
        })
    }

    /// The fused conv + bias + SiLU, continuing from the state's tail.
    ///
    /// Returns `[t, conv_dim]`.  Delegates to [`crate::conv1d`] when that
    /// module provides a native device kernel; the CPU path here is the
    /// reference it is validated against.
    fn conv(&self, state: &mut SsmState, xbc: &Tensor) -> Result<Tensor> {
        let c = &self.cfg;
        let dev = xbc.device();
        let tail = match &state.conv {
            Some(t) => t.clone(),
            None => Tensor::zeros((c.d_conv - 1, c.conv_dim()), DType::F32, dev)?,
        };
        let t = xbc.dim(0)?;
        // `[d_conv-1 + t, conv_dim]`: the previous tokens' tail then this chunk.
        let inp = Tensor::cat(&[&tail, xbc], 0)?.contiguous()?;
        // y[t,c] = sum_k w[c,k] * inp[t+k, c]
        let mut acc: Option<Tensor> = None;
        for k in 0..c.d_conv {
            let slice = inp.narrow(0, k, t)?;
            let tap = self.conv_w.narrow(1, k, 1)?.reshape((1, c.conv_dim()))?;
            let term = slice.broadcast_mul(&tap)?;
            acc = Some(match acc {
                Some(a) => (a + term)?,
                None => term,
            });
        }
        let mut y = acc.expect("d_conv >= 1");
        if let Some(b) = &self.conv_b {
            y = y.broadcast_add(&b.reshape((1, c.conv_dim()))?)?;
        }
        state.conv = Some(inp.narrow(0, t, c.d_conv - 1)?.contiguous()?);
        silu(&y)
    }

    /// Grouped RMS norm over blocks of `g` consecutive values: each block of
    /// the `d_inner` row is scaled by `1/sqrt(mean(x²) + eps) · weight` with
    /// its own weight entry (llama.cpp's `[d_inner/n_group, n_group]`
    /// `ssm_norm`, normalised over dim 0).
    fn grouped_rms_norm(&self, x: &Tensor, t: usize) -> Result<Tensor> {
        let c = &self.cfg;
        // `n` values per group; the row splits into `n_group` such blocks.
        let n = c.d_inner / c.n_group;
        let rows = x.reshape((t, c.n_group, n))?; // [t, n_group, n]
                                                  // mean of squares over each group's block, then 1/sqrt(ms + eps)
        let ms = (rows.sqr()?.sum(D::Minus1)? / n as f64)?; // [t, n_group]
        let inv = (ms + self.eps)?.powf(-0.5)?.reshape((t, c.n_group, 1))?;
        let x = rows.broadcast_mul(&inv)?;
        // `norm_w` is `[n, n_group]`; transpose to `[n_group, n]` so each
        // group is weighted by its own column.
        let w = self.norm_w.t()?.contiguous()?.reshape((1, c.n_group, n))?;
        (x.broadcast_mul(&w)?).reshape((t, c.d_inner))
    }

    fn fresh_state(&self, dev: &Device) -> Result<SsmState> {
        Ok(SsmState {
            conv: Some(Tensor::zeros(
                (self.cfg.d_conv - 1, self.cfg.conv_dim()),
                DType::F32,
                dev,
            )?),
            scan: vec![0.0; self.cfg.n_head * self.cfg.head_dim() * self.cfg.d_state],
        })
    }

    fn forward(
        &self,
        state: &mut Option<SsmState>,
        cap: &mut Option<RecurrentSnapshot>,
        xs: &Tensor,
    ) -> Result<Tensor> {
        let c = &self.cfg;
        let (b, t, _) = xs.dims3()?;
        if b != 1 {
            candle_core::bail!("Nemotron-H Mamba-2 supports batch size 1, got {b}");
        }
        let x2 = xs.reshape((t, xs.dim(2)?))?;
        if state.is_none() {
            *state = Some(self.fresh_state(xs.device())?);
        }
        let st = state.as_mut().expect("just filled");
        // Clone the running state *before* the conv advances its tail and the
        // scan advances its state: a snapshot taken afterwards would carry the
        // rejected tokens' history and re-seed it on rollback.
        let pre = match cap.as_ref() {
            Some(c) if c.armed => Some((st.conv.clone(), st.scan.clone())),
            _ => None,
        };

        let proj = self.in_proj.forward(&x2)?; // [t, d_in_proj]
        let (hd, nh) = (c.head_dim(), c.n_head);
        let z = proj.narrow(1, 0, c.d_inner)?; // [t, d_inner]
        let xbc = proj.narrow(1, c.d_inner, c.conv_dim())?;
        let dt_raw = proj.narrow(1, c.d_inner + c.conv_dim(), c.n_head)?; // [t, n_head]

        let mixed = self.conv(st, &xbc)?; // [t, conv_dim] contiguous, silu applied
                                          // `mixed` is handed to the scan as one fused buffer (x ‖ B ‖ C), the
                                          // same shape llama.cpp feeds `ggml_ssm_scan` — a strided view per part
                                          // would force a contiguous() copy of the whole prefill.
        if let (Some((pre_conv, pre_scan)), Some(cap)) = (pre, cap.as_mut()) {
            cap.pre_conv = pre_conv;
            cap.pre_scan = pre_scan;
            cap.mixed = Some(mixed.clone());
            cap.dt = Some(dt_raw.contiguous()?);
            cap.xbc = Some(xbc.contiguous()?);
            cap.params = Some(ScanParams {
                cfg: self.cfg.clone(),
                dt_b: self.dt_b.clone(),
                a: self.a.clone(),
                d_skip: self.d_skip.clone(),
            });
        }

        let y = crate::scan::ssm_scan(
            crate::scan::ScanConfig {
                d_inner: c.d_inner,
                n_head: nh,
                d_state: c.d_state,
                n_group: c.n_group,
                chunk: 128,
            },
            &mixed,
            &dt_raw,
            &self.dt_b,
            &self.a,
            &self.d_skip,
            &mut st.scan,
        )?; // [t, d_inner]

        // y = z * silu(y); grouped RMS norm; project out.
        //
        // ggml normalises `ssm_norm` over **dim 0** of the tensor reshaped to
        // `[d_inner/n_group, n_group, t]` — i.e. each group's `d_inner/n_group`
        // consecutive values — not candle's usual last-dim norm.  Transposing
        // that block to `[.., d_inner/n_group]` and using the stock RmsNorm
        // would normalise a different axis, so the grouped reduction is done
        // explicitly here.
        let gated = (z * silu(&y)?)?;
        let normed = self.grouped_rms_norm(&gated, t)?;
        let _ = hd;
        self.out.forward(&normed)?.reshape((1, t, xs.dim(2)?))
    }
}

/// The GQA attention mixer of an `attention` layer (no Q/K norm, optional
/// biases: Nemotron-H uses plain GQA + RoPE).
struct AttentionBlock {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    rope: Arc<Rope>,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    scale: f64,
}

impl AttentionBlock {
    fn load<R: Read + Seek>(
        src: &mut TensorSource<R>,
        p: &str,
        cfg: &Config,
        rope: Arc<Rope>,
    ) -> Result<Self> {
        let dim = |n: usize| format!("{n}");
        let _ = dim(cfg.n_embd);
        Ok(Self {
            q: Linear {
                w: Weight(src.qmatmul(&format!("{p}.attn_q.weight"))?),
                b: src.f32_opt(&format!("{p}.attn_q.bias"))?,
            },
            k: Linear {
                w: Weight(src.qmatmul(&format!("{p}.attn_k.weight"))?),
                b: src.f32_opt(&format!("{p}.attn_k.bias"))?,
            },
            v: Linear {
                w: Weight(src.qmatmul(&format!("{p}.attn_v.weight"))?),
                b: src.f32_opt(&format!("{p}.attn_v.bias"))?,
            },
            o: Linear {
                w: Weight(src.qmatmul(&format!("{p}.attn_output.weight"))?),
                b: src.f32_opt(&format!("{p}.attn_output.bias"))?,
            },
            rope,
            n_head: cfg.n_head,
            n_kv: cfg.n_head_kv,
            head_dim: cfg.head_dim,
            scale: 1.0 / (cfg.head_dim as f64).sqrt(),
        })
    }

    fn forward(
        &self,
        kv: &mut KvCache,
        xs: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        let (nh, nkv, hd) = (self.n_head, self.n_kv, self.head_dim);
        let q = self
            .q
            .forward(xs)?
            .reshape((b, t, nh, hd))?
            .transpose(1, 2)?;
        let k = self
            .k
            .forward(xs)?
            .reshape((b, t, nkv, hd))?
            .transpose(1, 2)?;
        let v = self
            .v
            .forward(xs)?
            .reshape((b, t, nkv, hd))?
            .transpose(1, 2)?;
        let (q, k) = self.rope.apply_pair(&q, &k, offset)?;
        let ctx = crate::attention::cached_attention(kv, &q, k, v, mask, self.scale)?;
        self.o.forward(&ctx.reshape((b, t, nh * hd))?)
    }
}

/// One routed expert: `relu2(x·W_up) · W_down` (Nemotron-H experts are
/// **ungated** — a single up projection, not SwiGLU).
struct Expert {
    up: Weight,
    down: Weight,
}

impl crate::moe::Expert for Expert {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = relu2(&self.up.forward(xs)?)?;
        self.down.forward(&h)
    }
}

/// `relu(x)^2` (llama.cpp `LLM_FFN_RELU_SQR`).
fn relu2(x: &Tensor) -> Result<Tensor> {
    let r = x.relu()?;
    r.sqr()
}

/// Each expert of a stacked `[n_expert, out, in]` tensor, borrowed from the
/// mapping when there is one and read-and-split otherwise.
///
/// Borrowing keeps the pool at its on-disk footprint; the read path is the
/// streamed-load fallback `QuantizedModel::from_gguf` needs (it passes no
/// mapping), and it copies each expert onto the device rather than expanding
/// the whole tensor.
fn expert_parts<R: Read + Seek>(
    src: &mut TensorSource<R>,
    et: &crate::moe::ExpertTensor,
    name: &str,
    candle: &impl Fn(QTensor) -> Result<Weight>,
) -> Result<Vec<Weight>> {
    if let Some(mmap) = src.mmap.clone() {
        if let Some(borrowed) = et.borrow_host(&mmap)? {
            return borrowed.into_iter().map(|b| candle(b.tensor)).collect();
        }
    }
    // No mapping, or a tensor the mmap slicer cannot borrow (streamed load).
    let parts = et
        .read_and_split(&src.ct, &mut src.reader, src.arch, &src.device)
        .map_err(|e| {
            candle_core::Error::Msg(format!(
                "{}: routed-expert tensor `{name}` could be neither borrowed nor split: {e}",
                src.arch
            ))
        })?;
    parts.into_iter().map(candle).collect()
}

/// The MoE FFN of a `moe` layer: sigmoid-routed top-k experts over an
/// optional latent projection, plus an always-on shared expert.
struct MoeBlock {
    router: Tensor, // [n_embd, n_expert] transposed for matmul
    experts: Vec<Expert>,
    shared_up: Weight,
    shared_down: Weight,
    latent_down: Option<Weight>,
    latent_up: Option<Weight>,
    n_expert: usize,
    k: usize,
    scale: f64,
    norm_weights: bool,
    n_embd: usize,
    /// The device the routed experts actually live on.  They are borrowed
    /// from the mapping, so this is the CPU even when the activations run on
    /// an accelerator; the shared dispatch moves the block input across once
    /// and the result back once, instead of feeding GPU activations to
    /// CPU-resident weights.
    expert_device: Device,
}

impl MoeBlock {
    fn load<R: Read + Seek>(
        src: &mut TensorSource<R>,
        p: &str,
        cfg: &Config,
        expert_device: &Device,
    ) -> Result<Self> {
        let router = src
            .f32_tensor(&format!("{p}.ffn_gate_inp.weight"))? // [n_expert, n_embd]
            .t()?
            .contiguous()?; // [n_embd, n_expert]
        let up = src.expert_tensor(&format!("{p}.ffn_up_exps.weight"), cfg.n_expert)?;
        let down = src.expert_tensor(&format!("{p}.ffn_down_exps.weight"), cfg.n_expert)?;
        let candle = |qt: QTensor| -> Result<Weight> { Ok(Weight(QMatMul::from_qtensor(qt)?)) };
        // Borrow each expert's slice of the mapping when there is one, exactly
        // as `quantized_qwen::split_experts` prefers: an expert then costs a
        // pointer and a length until tokens route to it, and the pool keeps its
        // on-disk footprint instead of expanding to f32.  Without a mapping
        // (a streamed load, see `QuantizedModel::from_gguf`) read and split the
        // stacked tensor instead — refusing here would make the public
        // no-mmap loader unusable for every Nemotron MoE model.
        let ups = expert_parts(src, &up, &format!("{p}.ffn_up_exps.weight"), &candle)?;
        let downs = expert_parts(src, &down, &format!("{p}.ffn_down_exps.weight"), &candle)?;
        let experts = ups
            .into_iter()
            .zip(downs)
            .map(|(up, down)| Expert { up, down })
            .collect();
        Ok(Self {
            router,
            experts,
            shared_up: Weight(src.qmatmul(&format!("{p}.ffn_up_shexp.weight"))?),
            shared_down: Weight(src.qmatmul(&format!("{p}.ffn_down_shexp.weight"))?),
            latent_down: if src.has(&format!("{p}.ffn_latent_down.weight")) {
                Some(Weight(src.qmatmul(&format!("{p}.ffn_latent_down.weight"))?))
            } else {
                None
            },
            latent_up: if src.has(&format!("{p}.ffn_latent_up.weight")) {
                Some(Weight(src.qmatmul(&format!("{p}.ffn_latent_up.weight"))?))
            } else {
                None
            },
            n_expert: cfg.n_expert,
            k: cfg.n_expert_used,
            scale: cfg.expert_weights_scale,
            norm_weights: cfg.expert_weights_norm,
            n_embd: cfg.n_embd,
            expert_device: expert_device.clone(),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, h) = xs.dims3()?;
        let x2 = xs.reshape((b * t, h))?;
        let logits = x2.matmul(&self.router)?; // [n, n_expert]
                                               // Nemotron-H routes with an independent sigmoid per expert.
        let scores = Gating::Sigmoid.scores(&logits)?;
        let idx = crate::moe::topk_indices(&scores, self.k)?;
        let weights = crate::moe::routing_weights(&scores, &idx, self.norm_weights, self.scale)?;

        // The routed experts read the (optionally latent) block input.
        let routed_in = match &self.latent_down {
            Some(w) => w.forward(&x2)?,
            None => x2.clone(),
        };
        let (out, _ids) = if b * t == 1 {
            crate::moe::dispatch_decode(
                "nemotron_h",
                &self.experts,
                &self.expert_device,
                &routed_in,
                &idx,
                &weights,
                self.k,
            )?
        } else {
            crate::moe::dispatch_prefill(
                "nemotron_h",
                &self.experts,
                &self.expert_device,
                &routed_in,
                &idx,
                &weights,
                b * t,
                self.k,
            )?
        };
        let out = match &self.latent_up {
            Some(w) => w.forward(&out)?,
            None => out,
        };
        // Shared expert, added to the routed output (no gate on it).
        let shared = relu2(&self.shared_up.forward(&x2)?)?;
        let shared = self.shared_down.forward(&shared)?;
        self.n_embd_guard(&out)?;
        (out + shared)?.reshape((b, t, h))
    }

    /// The latent projection must preserve the residual width.
    fn n_embd_guard(&self, out: &Tensor) -> Result<()> {
        let d = out.dim(D::Minus1)?;
        if d != self.n_embd {
            candle_core::bail!(
                "nemotron_h: MoE output width {d} != embedding length {}",
                self.n_embd
            );
        }
        Ok(())
    }
}

// ─── Layers ──────────────────────────────────────────────────────────────────

/// One decoder layer: a pre-norm residual around exactly one sub-block.
enum SubBlock {
    Mamba(Box<Mamba2>),
    Attention(Box<AttentionBlock>),
    Moe(Box<MoeBlock>),
}

struct Layer {
    norm: RmsNorm,
    block: SubBlock,
}

impl Layer {
    fn forward(
        &self,
        state: &mut LayerState,
        xs: &Tensor,
        input: &LayerInput<'_>,
    ) -> Result<Tensor> {
        let h = self.norm.forward(xs)?;
        let out = match &self.block {
            SubBlock::Mamba(m) => m.forward(&mut state.ssm, &mut state.capture, &h)?,
            SubBlock::Attention(a) => a.forward(&mut state.kv, &h, input.mask, input.offset)?,
            SubBlock::Moe(f) => f.forward(&h)?,
        };
        xs + out
    }
}

/// One layer's per-session state.
#[derive(Clone, Default)]
pub struct LayerState {
    kv: KvCache,
    ssm: Option<SsmState>,
    /// Armed by [`LayerStack::begin_verify`], filled by a recurrent layer.
    capture: Option<RecurrentSnapshot>,
}

/// What a verification pass would overwrite in a recurrent layer, plus the
/// inputs needed to *replay* the accepted prefix (see
/// [`LayerStack::restore_state`]).
#[derive(Clone, Default)]
pub struct RecurrentSnapshot {
    /// Conv tail and scan state as they were before the pass.
    pre_conv: Option<Tensor>,
    pre_scan: Vec<f32>,
    /// The pass's post-conv fused rows, `[t, conv_dim]`, and its raw `dt`,
    /// so a rollback replays only the accepted prefix without recomputing the
    /// projection or the conv.
    mixed: Option<Tensor>,
    dt: Option<Tensor>,
    /// The pass's **pre-conv** `xBC` rows, `[t, conv_dim]`.  The conv tail is
    /// defined on these raw projection outputs, not on the post-conv
    /// activations, so a rollback can only rebuild it from here.
    xbc: Option<Tensor>,
    /// The scan's own parameters, carried because
    /// [`LayerStack::restore_state`] is a static method with no handle on the
    /// weights — the same reason `quantized_qwen`'s snapshot carries its
    /// `SsmConfig`.
    params: Option<ScanParams>,
    /// Set by `begin_verify` so the layer knows to record; cleared on fill.
    armed: bool,
}

/// The per-layer scalars the scan needs to replay a prefix.
#[derive(Clone)]
struct ScanParams {
    cfg: SsmConfig,
    dt_b: Tensor,
    a: Tensor,
    d_skip: Tensor,
}

/// A prompt-boundary checkpoint of a recurrent layer's running state.
#[derive(Clone, Default)]
pub struct RunningState {
    conv: Option<Tensor>,
    scan: Vec<f32>,
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// The immutable half: every weight, the device and the config.  Sessions
/// ([`ModelWeights`]) share one `Arc` of it and own only their state.
pub struct Weights {
    tok_embeddings: TokenEmbedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    output: Weight,
    device: Device,
    residency: Arc<dyn crate::residency::ExpertResidency>,
    n_expert: usize,
    cfg: Config,
}

impl Weights {
    /// Load a Nemotron-H GGUF.
    pub fn load<R: Read + Seek>(
        ct: gguf_file::Content,
        raw: Option<GgufHeader>,
        reader: R,
        device: &Device,
        mmap: Option<Arc<memmap2::Mmap>>,
        arch: &'static str,
    ) -> Result<Self> {
        // Where the routed experts actually live.  On the borrowed path they
        // are CPU-resident regardless of the dense set's device; on the
        // streamed path `expert_parts` materialises them on `device`.  Both
        // are recorded so the shared MoE dispatch moves the block input to
        // the experts' device instead of assuming it is the activations'.
        let expert_device = if mmap.is_some() {
            Device::Cpu
        } else {
            device.clone()
        };
        let meta = Meta::new(&ct.metadata, arch);
        let cfg = Config::from_meta(&meta)?.ok_or_else(|| {
            candle_core::Error::Msg(format!("{arch} is not a Nemotron-H architecture"))
        })?;
        let mut src = TensorSource {
            ct,
            raw,
            reader,
            device: device.clone(),
            mmap,
            arch,
        };
        let tok_embeddings = TokenEmbedding::load(src.qtensor("token_embd.weight")?, device)?;
        let rope = Arc::new(cfg.rope(device)?);
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for (i, spec) in cfg.layers.iter().enumerate() {
            let p = format!("blk.{i}");
            let norm = src.rms_norm(&format!("{p}.attn_norm.weight"), cfg.rms_eps)?;
            let block = match spec.kind {
                LayerKind::Mamba => {
                    SubBlock::Mamba(Box::new(Mamba2::load(&mut src, &p, &cfg.ssm)?))
                }
                LayerKind::Attention => SubBlock::Attention(Box::new(AttentionBlock::load(
                    &mut src,
                    &p,
                    &cfg,
                    rope.clone(),
                )?)),
                LayerKind::Moe => SubBlock::Moe(Box::new(MoeBlock::load(
                    &mut src,
                    &p,
                    &cfg,
                    &expert_device,
                )?)),
            };
            layers.push(Layer { norm, block });
        }
        let norm = src.rms_norm("output_norm.weight", cfg.rms_eps)?;
        let output = if src.has("output.weight") {
            Weight(src.qmatmul("output.weight")?)
        } else {
            // Tied embeddings: reuse the (quantized) embedding table.
            Weight(src.qmatmul("token_embd.weight")?)
        };
        Ok(Self {
            tok_embeddings,
            layers,
            norm,
            output,
            device: device.clone(),
            residency: Arc::new(crate::residency::NoopResidency),
            n_expert: cfg.n_expert,
            cfg,
        })
    }
}

impl LayerStack for Weights {
    type State = LayerState;
    type Snapshot = RecurrentSnapshot;
    type Checkpoint = RunningState;

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
        states: &mut [LayerState],
        xs: &Tensor,
        input: &LayerInput<'_>,
    ) -> Result<(Tensor, Vec<u32>)> {
        let out = self.layers[l].forward(&mut states[l], xs, input)?;
        // Routing ids are not surfaced here: the routed experts of a Mamba
        // layer have no residency pool of their own yet (the MoE layers do,
        // via `residency`), so nothing consumes them.
        Ok((out, Vec::new()))
    }

    fn head(&self, xs: &Tensor) -> Result<Tensor> {
        let normed = self.norm.forward(xs)?;
        self.output.forward(&normed)?.to_dtype(DType::F32)
    }

    /// The scan state is running rather than append-only, so a verification
    /// pass must snapshot before it advances.  A mamba layer cannot be cut
    /// back to an arbitrary prefix, which is exactly why it snapshots.
    fn can_truncate(&self) -> bool {
        !self
            .layers
            .iter()
            .any(|l| matches!(l.block, SubBlock::Mamba(_)))
    }

    fn truncate_state(state: &mut LayerState, keep: usize) -> Result<()> {
        // A mamba layer's scan state cannot be cut back, so only the KV cache
        // of an attention layer is truncated here; a recurrent layer rewinds
        // through `snapshot_state`/`restore_state` instead.
        crate::attention::truncate_kv(&mut state.kv, keep);
        Ok(())
    }

    fn begin_verify(state: &mut LayerState) {
        // Arm unconditionally.  A fresh Mamba layer has `ssm == None` until its
        // first forward creates the state, so keying this off `ssm.is_some()`
        // would leave a verification pass at offset 0 with nothing to roll
        // back to — the rejected tokens would stay folded into the recurrence.
        // An attention layer never fills the capture (it has no `ssm`), so
        // `snapshot_state` still reports nothing for it and the checkpoint
        // stays layer-aligned.
        state.capture = Some(RecurrentSnapshot {
            armed: true,
            ..Default::default()
        });
    }

    fn clear_verify(state: &mut LayerState) {
        state.capture = None;
    }

    fn snapshot_state(state: &LayerState) -> Option<RecurrentSnapshot> {
        state.capture.clone()
    }

    fn restore_state(
        state: &mut LayerState,
        snap: RecurrentSnapshot,
        retained: usize,
    ) -> Result<()> {
        let Some(st) = state.ssm.as_mut() else {
            return Ok(());
        };
        // A fresh layer (verification at offset 0) had no running state before
        // the pass: rolling all the way back means there is nothing to keep, so
        // dropping the state is the correct "empty" result — otherwise the
        // rejected tokens stay folded into a state that never existed.
        if retained == 0 && snap.pre_conv.is_none() && snap.pre_scan.is_empty() {
            state.ssm = None;
            return Ok(());
        }
        // Back to the pre-pass state first, discarding the whole pass...
        st.conv = snap.pre_conv.clone();
        st.scan = snap.pre_scan.clone();
        if retained == 0 {
            return Ok(());
        }
        // ...then *replay* the accepted prefix.  The scan is stateful, so the
        // accepted tokens must be re-applied or the recurrence ends up behind
        // plain decoding.
        let (Some(mixed), Some(dt), Some(params)) =
            (snap.mixed.as_ref(), snap.dt.as_ref(), snap.params.as_ref())
        else {
            return Ok(());
        };
        let kept = retained.min(mixed.dim(0)?);
        let mixed = mixed.narrow(0, 0, kept)?.contiguous()?;
        let dt = dt.narrow(0, 0, kept)?.contiguous()?;
        // The conv tail is the last `d_conv-1` **pre-conv** rows of the pass
        // (llama.cpp keeps the raw `xBC`, not the silu'd activations), so it
        // is rebuilt from `xbc`; a shorter accepted prefix is completed from
        // the pre-pass tail.
        let c = &params.cfg;
        if c.d_conv > 1 && kept > 0 {
            let Some(xbc) = snap.xbc.as_ref() else {
                return Ok(());
            };
            let tail_from = kept.saturating_sub(c.d_conv - 1);
            let width = kept - tail_from;
            if width < c.d_conv - 1 {
                let pre = st.conv.clone().ok_or_else(|| {
                    candle_core::Error::Msg("nemotron_h: no conv tail to continue".into())
                })?;
                let need = (c.d_conv - 1) - width;
                let from_pre = pre.narrow(0, (c.d_conv - 1) - need, need)?;
                st.conv = Some(Tensor::cat(
                    &[from_pre, xbc.narrow(0, tail_from, width)?],
                    0,
                )?);
            } else {
                st.conv = Some(xbc.narrow(0, tail_from, c.d_conv - 1)?.contiguous()?);
            }
        }
        let _ = crate::scan::ssm_scan(
            crate::scan::ScanConfig {
                d_inner: c.d_inner,
                n_head: c.n_head,
                d_state: c.d_state,
                n_group: c.n_group,
                chunk: 128,
            },
            &mixed,
            &dt,
            &params.dt_b,
            &params.a,
            &params.d_skip,
            &mut st.scan,
        )?;
        Ok(())
    }

    fn can_rewind(&self) -> bool {
        true
    }

    fn checkpoint_state(state: &LayerState) -> Option<RunningState> {
        let ssm = state.ssm.as_ref()?;
        Some(RunningState {
            conv: ssm.conv.clone(),
            scan: ssm.scan.clone(),
        })
    }

    fn restore_checkpoint(state: &mut LayerState, checkpoint: &RunningState) -> Result<()> {
        if let Some(st) = state.ssm.as_mut() {
            st.conv = checkpoint.conv.clone();
            st.scan = checkpoint.scan.clone();
        }
        Ok(())
    }

    fn fork_state(state: &LayerState, keep: usize, device: &Device) -> Result<LayerState> {
        use crate::native_session::tensor_on;
        if keep == 0 {
            return Ok(LayerState::default());
        }
        // The forward replaces the conv tail and the scan state rather than
        // writing into them, so only the KV rows need a copy.
        let ssm = state.ssm.as_ref().map(|s| {
            Ok::<_, candle_core::Error>(SsmState {
                conv: s.conv.as_ref().map(|t| tensor_on(t, device)).transpose()?,
                scan: s.scan.clone(),
            })
        });
        Ok(LayerState {
            kv: crate::attention::fork_kv(&state.kv, keep, device)?,
            ssm: ssm.transpose()?,
            capture: None,
        })
    }

    fn state_bytes(state: &LayerState) -> u64 {
        use crate::attention::tensor_bytes;
        crate::attention::kv_bytes(&state.kv)
            + state.ssm.as_ref().map_or(0, |s| {
                s.conv.as_ref().map_or(0, tensor_bytes)
                    + (s.scan.len() * std::mem::size_of::<f32>()) as u64
            })
    }
}

/// A Nemotron-H session over shared [`Weights`].
pub type ModelWeights = crate::native_session::Session<Weights>;
