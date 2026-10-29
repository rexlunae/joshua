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
    /// Trunk layers (excludes the appended NextN block).
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
        // `block_count` **includes** any appended NextN / MTP blocks (llama.cpp
        // `num_nextn_predict_layers`), so the trunk is the remainder.  Reading
        // it raw would try to load the draft block as an ordinary layer.
        let block_count = m.u32("block_count")? as usize;
        let n_nextn = m.u32_or("nextn_predict_layers", 0) as usize;
        let n_layer = block_count.checked_sub(n_nextn).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "{}: block_count {block_count} is smaller than nextn_predict_layers {n_nextn}",
                m.arch()
            ))
        })?;
        if n_layer == 0 {
            candle_core::bail!("{}: no trunk layers (block_count {block_count})", m.arch());
        }
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
        // The KV head count normally comes from the trunk's attention rows, but
        // the appended NextN block *always* has an attention sub-layer, so an
        // attention-free trunk still needs a usable value.  Two sources, in
        // order: the first attention row's `head_count_kv`, then the scalar
        // form (which covers a file whose arrays are all zeros).
        let kv_from_array = kv.iter().copied().find(|&k| k > 0);
        let kv_from_scalar = m.u32_opt("attention.head_count_kv").filter(|&k| k > 0);
        let n_head_kv = kv_from_array
            .or_else(|| kv_from_scalar.map(|k| k as usize))
            .unwrap_or(0);
        if (n_attn > 0 || m.u32_opt("nextn_predict_layers").unwrap_or(0) > 0) && n_head_kv == 0 {
            candle_core::bail!(
                "{}: attention layers (or the NextN block) need a nonzero \
                 attention.head_count_kv, found none",
                m.arch()
            );
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

/// Load a linear weight, preferring NVFP4 blocks when the file stores them.
///
/// NVFP4 is kept as blocks (see [`Weight::Nvfp4`]); every other dtype goes
/// through candle's quantized matmul, which covers the formats it can name and
/// decodes the rest on the host.
fn load_weight<R: Read + Seek>(src: &mut TensorSource<R>, name: &str) -> Result<Weight> {
    // Own the device up front so the immutable borrow of `src.device` does not
    // overlap the mutable borrow the loader needs.
    let device = src.device.clone();
    if let Some(w) = Weight::nvfp4(src, name, &device)? {
        return Ok(w);
    }
    Ok(Weight::Quant(src.qmatmul(name)?))
}

/// A linear weight: either candle's quantized matmul, or an **NVFP4** matrix
/// kept as its raw blocks so the device GEMV decodes it in-kernel.
///
/// Nemotron 3.5 ships NVFP4 checkpoints.  candle has no NVFP4 dtype
/// (`GgmlDType::from_u32` rejects id 40), so the only alternatives were to
/// decode the whole matrix to f32 at load — 8x the bytes streamed per token,
/// and the model no longer fits its intended footprint — or to keep the 36-byte
/// blocks and let [`crate::nvfp4_gemv::nvfp4_matmul_qt`] decode them where they
/// are used.  This is the second.
#[derive(Clone)]
enum Weight {
    /// candle's own quantized matmul (every dtype it can name).
    Quant(QMatMul),
    /// NVFP4 blocks, `[out_dim, 36 * blocks_per_row]` u8, plus the logical
    /// `[out_dim, in_dim]` shape the caller sees.
    Nvfp4 { blocks: Tensor, out_dim: usize, in_dim: usize },
}

impl Weight {
    /// Load `name` as an NVFP4 weight when the file stores it as
    /// `GGML_TYPE_NVFP4` blocks, else `None`.
    ///
    /// The blocks are kept verbatim (36 bytes per 64 elements) and placed on
    /// `device`, so a device GEMV decodes them where they are used and the
    /// weights stream at their on-disk size instead of as f32.
    fn nvfp4<R: Read + Seek>(
        src: &mut TensorSource<R>,
        name: &str,
        device: &Device,
    ) -> Result<Option<Self>> {
        let Some(info) = src.raw.as_ref().and_then(|r| r.tensors.get(name)) else {
            return Ok(None);
        };
        if info.dtype != <crate::mxfp4::BlockNvfp4 as crate::raw_block::RawBlock>::GGML_TYPE {
            return Ok(None);
        }
        // `RawTensorInfo.dims` is already in row-major order — `read_header`
        // reverses GGUF's fastest-varying-first order — so it is `[out, in]`
        // exactly as the op wants.  Swapping here would reinterpret the same
        // block bytes as the transposed matrix.
        let (out_dim, in_dim) = match info.dims.as_slice() {
            [o, i] => (*o, *i),
            d => candle_core::bail!(
                "{}: NVFP4 tensor `{name}` must be 2-D, got dims {d:?}",
                src.arch
            ),
        };
        let row_bytes = crate::nvfp4_gemv::row_bytes(in_dim)?;
        let bytes = src.raw.as_ref().unwrap().read_tensor_bytes(&mut src.reader, name)?;
        let blocks = Tensor::from_vec(bytes, (out_dim, row_bytes), device)?;
        Ok(Some(Self::Nvfp4 { blocks, out_dim, in_dim }))
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Quant(q) => q.forward(xs),
            Self::Nvfp4 { blocks, .. } => {
                // The op takes `[in]` or `[rows, in]` with `rows <= MAX_ROWS`.
                // This loader always runs with a leading batch of one, so a
                // `[1, t, hidden]` activation is flattened to `[t, hidden]` and
                // the batch axis restored after.
                let dims = xs.dims().to_vec();
                let flat = if dims.len() == 3 {
                    xs.reshape((dims[1], dims[2]))?
                } else {
                    xs.clone()
                };
                let rows = flat.dim(0)?;
                // A prefill chunk is up to `--prefill-chunk` tokens (512 by
                // default), far past the op's row limit, so the rows are
                // processed in batches and concatenated in token order.  The
                // split is on the *token* axis, so every output row still
                // corresponds to the input row at the same index.
                let mut y = if rows <= crate::nvfp4_gemv::MAX_ROWS {
                    crate::nvfp4_gemv::nvfp4_matmul_qt(blocks, &flat)?
                } else {
                    let mut parts = Vec::with_capacity(rows.div_ceil(crate::nvfp4_gemv::MAX_ROWS));
                    let mut start = 0;
                    while start < rows {
                        let n = (rows - start).min(crate::nvfp4_gemv::MAX_ROWS);
                        let chunk = flat.narrow(0, start, n)?.contiguous()?;
                        parts.push(crate::nvfp4_gemv::nvfp4_matmul_qt(blocks, &chunk)?);
                        start += n;
                    }
                    Tensor::cat(&parts, 0)?
                };
                if dims.len() == 3 {
                    let c = y.dim(1)?;
                    y = y.reshape((1, dims[1], c))?;
                }
                Ok(y)
            }
        }
    }

    /// The logical weight shape, for load-time validation.
    fn shape(&self) -> Option<(usize, usize)> {
        match self {
            Self::Quant(q) => match q {
                QMatMul::QTensor(t) => t.shape().dims().first().map(|_| ()).and_then(|_| {
                    let d = t.shape().dims();
                    (d.len() == 2).then(|| (d[0], d[1]))
                }),
                _ => None,
            },
            Self::Nvfp4 { out_dim, in_dim, .. } => Some((*out_dim, *in_dim)),
        }
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
    /// Grouped RMS-norm weight, pre-transposed at load to
    /// `[1, n_group, d_inner / n_group]` so `grouped_rms_norm` broadcasts it
    /// with no per-token transpose or copy.  llama.cpp stores it as
    /// `[d_inner/n_group, n_group]`, normalised over dim 0, with column `g`
    /// weighting group `g`.
    norm_w: Tensor,
    eps: f64,
    out: Weight,
    cfg: SsmConfig,
}

impl Mamba2 {
    fn load<R: Read + Seek>(src: &mut TensorSource<R>, p: &str, cfg: &SsmConfig) -> Result<Self> {
        let conv_w = src
            .f32_tensor(&format!("{p}.ssm_conv1d.weight"))?
            // ggml stores `ssm_conv1d` as `[d_conv, conv_dim]` (tap-major,
            // `w[k*C + c]`), which is exactly what `conv1d` and its reference
            // tap ladder expect, so keep it and make it contiguous once here
            // rather than slicing/reshaping a tap column per token.
            .reshape((cfg.d_conv, cfg.conv_dim()))?
            .contiguous()?;
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
        // Stored already transposed into the `[1, n_group, d_inner/n_group]`
        // form `grouped_rms_norm` broadcasts, so the hot path does no transpose
        // or copy per token — one layout change at load instead of one per
        // layer per token.
        let norm_w = src
            .f32_tensor(&format!("{p}.ssm_norm.weight"))?
            .reshape((cfg.d_inner / cfg.n_group, cfg.n_group))?
            .t()?
            .contiguous()?
            .reshape((1, cfg.n_group, cfg.d_inner / cfg.n_group))?;
        Ok(Self {
            in_proj: load_weight(src, &format!("{p}.ssm_in.weight"))?,
            conv_w,
            conv_b,
            dt_b,
            a,
            d_skip,
            norm_w,
            eps: 1e-5,
            out: load_weight(src, &format!("{p}.ssm_out.weight"))?,
            cfg: cfg.clone(),
        })
    }

    /// The fused conv + bias + SiLU, continuing from the state's tail.
    ///
    /// Returns `[t, conv_dim]`.  Delegates to [`crate::conv1d`] when that
    /// module provides a native device kernel; the CPU path here is the
    /// reference it is validated against.
    /// The fused causal depthwise conv + bias + SiLU, continuing from the
    /// state's stored tail.
    ///
    /// Delegates to [`crate::conv1d::conv1d`], which runs one native kernel on
    /// an OpenCL device (bias and SiLU folded into the same launch, where
    /// llama.cpp runs `ggml_ssm_conv`/`ggml_add`/`ggml_silu` separately) and
    /// the same arithmetic on the CPU.  The per-tap tensor loop this replaced
    /// ran `d_conv` broadcast multiplies plus a bias add plus a SiLU per layer
    /// per token, all of them host round trips on a device.
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
        let bias = match &self.conv_b {
            Some(b) => Some(b.reshape((c.conv_dim(),))?),
            None => None,
        };
        let y = crate::conv1d::conv1d(
            &inp,
            &self.conv_w,
            bias.as_ref(),
            crate::conv1d::Conv1d::mamba2(),
        )?;
        state.conv = Some(inp.narrow(0, t, c.d_conv - 1)?.contiguous()?);
        Ok(y)
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
        // `norm_w` is pre-transposed to `[1, n_group, n]` at load; broadcast it
        // directly.
        (x.broadcast_mul(&self.norm_w)?).reshape((t, c.d_inner))
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
                w: load_weight(src, &format!("{p}.attn_q.weight"))?,
                b: src.f32_opt(&format!("{p}.attn_q.bias"))?,
            },
            k: Linear {
                w: load_weight(src, &format!("{p}.attn_k.weight"))?,
                b: src.f32_opt(&format!("{p}.attn_k.bias"))?,
            },
            v: Linear {
                w: load_weight(src, &format!("{p}.attn_v.weight"))?,
                b: src.f32_opt(&format!("{p}.attn_v.bias"))?,
            },
            o: Linear {
                w: load_weight(src, &format!("{p}.attn_output.weight"))?,
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
        // The released Nemotron 3.5 RoPE is **partial**: the GGUF declares
        // `rope.dimension_count` 84 against `head_dim` 128, so only the leading
        // 84 dims rotate (`partial_rotary_factor`).  `apply_pair` assumes the
        // full width and rejects the mismatch; `apply_leading` rotates the first
        // `n_rot` and passes the rest through.
        let (q, k) = (self.rope.apply_leading(&q, offset)?, self.rope.apply_leading(&k, offset)?);
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
/// Each expert of a stacked `[n_expert, out, in]` tensor.
///
/// Three cases, in order:
///
/// 1. **A raw block format candle cannot name** (Q2_0, Q1_0, IQ2_XXS, MXFP4,
///    NVFP4, …): borrowed per expert from the mapping and decoded at matmul
///    time.  This is not optional — a real Nemotron 3.5 conversion uses Q2_0
///    for its routed experts, and refusing it would make the model unloadable.
/// 2. **A candle-typed tensor with a mapping**: borrowed in place, so the pool
///    keeps its on-disk footprint.
/// 3. **A streamed load**: read and split onto `split_device`.
fn expert_parts_raw<R: Read + Seek>(
    src: &mut TensorSource<R>,
    name: &str,
    n_expert: usize,
    split_device: &Device,
    candle: &impl Fn(QTensor) -> Result<Weight>,
) -> Result<Option<(Vec<Weight>, Device)>> {
    let Some(info) = src
        .raw_dtype(name)
        .and_then(|_| src.raw.as_ref()?.tensors.get(name).cloned())
    else {
        return Ok(None);
    };
    let raw = src.raw.as_ref().expect("raw_dtype implies a raw header");
    let map = |b: crate::moe::BorrowedExpert| -> Result<Weight> { candle(b.tensor) };
    let split = crate::with_raw_block!(info.dtype, B => {
        crate::moe::split_raw_experts::<B, R>(
            src.arch,
            raw,
            &mut src.reader,
            src.mmap.as_ref(),
            split_device,
            name,
            &info,
            n_expert,
        )
    });
    match split {
        Some(experts) => {
            let parts = experts?.into_iter().map(map).collect::<Result<Vec<_>>>()?;
            // Raw blocks are borrowed from the mapping when one exists, and
            // decoded on the split device otherwise.
            let dev = if src.mmap.is_some() { Device::Cpu } else { split_device.clone() };
            Ok(Some((parts, dev)))
        }
        None => candle_core::bail!(
            "{}: routed-expert tensor `{name}` has GGUF dtype {}, which has no decoder here",
            src.arch,
            info.dtype
        ),
    }
}

fn expert_parts<R: Read + Seek>(
    src: &mut TensorSource<R>,
    et: &crate::moe::ExpertTensor,
    name: &str,
    split_device: &Device,
    candle: &impl Fn(QTensor) -> Result<Weight>,
) -> Result<(Vec<Weight>, Device)> {
    // Borrowed experts are CPU-resident slices of the mapping, whatever the
    // dense set's device: report that so the MoE dispatch moves the block's
    // input to the CPU rather than handing accelerator tensors to CPU
    // weights.
    if let Some(mmap) = src.mmap.clone() {
        if let Some(borrowed) = et.borrow_host(&mmap)? {
            let parts = borrowed.into_iter().map(|b| candle(b.tensor)).collect::<Result<Vec<_>>>()?;
            return Ok((parts, Device::Cpu));
        }
    }
    // No mapping, or a tensor the mmap slicer cannot borrow (streamed load).
    // Not borrowable: split onto `split_device`, which the caller chose to
    // match where it will dispatch (the CPU for a mapped-but-unaligned tensor,
    // the model device for a streamed load).
    let parts = et
        .read_and_split(&src.ct, &mut src.reader, src.arch, split_device)
        .map_err(|e| {
            candle_core::Error::Msg(format!(
                "{}: routed-expert tensor `{name}` could be neither borrowed nor split: {e}",
                src.arch
            ))
        })?;
    Ok((parts.into_iter().map(candle).collect::<Result<Vec<_>>>()?, split_device.clone()))
}

/// The MoE FFN of a `moe` layer: sigmoid-routed top-k experts over an
/// optional latent projection, plus an always-on shared expert.
struct MoeBlock {
    router: Tensor, // [n_embd, n_expert] transposed for matmul
    /// `ffn_exp_probs_b` — a per-expert bias that steers *selection* only.
    /// Nemotron 3.5 ships one (`mixer.gate.e_score_correction_bias`); the
    /// routing weights still come from the unbiased scores.
    gate_bias: Option<Tensor>,
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
        // The router's selection bias.  llama.cpp reads this as
        // `ffn_exp_probs_b`; without it the top-k picks differ on the real
        // model even though every tensor loads.
        // Centred at load, as the other MoE loaders do: the bias only steers
        // *selection*, so a large common offset carries no information but can
        // round distinct scores into ties in f32.
        let gate_bias = match src.f32_opt(&format!("{p}.exp_probs_b.bias"))? {
            Some(b) => Some(crate::moe::center_selection_bias(&b)?),
            None => None,
        };
        // A real conversion may store the routed experts in a raw block format
        // (Q2_0 in the published Lightning GGUF), so the stacked-tensor lookup
        // is only needed for the candle-typed path; `expert_parts` handles the
        // raw case itself.  A placeholder keeps the borrow simple.
        let up_name = format!("{p}.ffn_up_exps.weight");
        let down_name = format!("{p}.ffn_down_exps.weight");

        let candle = |qt: QTensor| -> Result<Weight> { Ok(Weight::Quant(QMatMul::from_qtensor(qt)?)) };
        // Borrow each expert's slice of the mapping when there is one, exactly
        // as `quantized_qwen::split_experts` prefers: an expert then costs a
        // pointer and a length until tokens route to it, and the pool keeps its
        // on-disk footprint instead of expanding to f32.  Without a mapping
        // (a streamed load, see `QuantizedModel::from_gguf`) read and split the
        // stacked tensor instead — refusing here would make the public
        // no-mmap loader unusable for every Nemotron MoE model.
        // `expert_parts` needs the `ExpertTensor` descriptor only for the
        // candle-typed path; a raw-format stack (Q2_0 in the published
        // Lightning GGUF) takes the borrowed-block path and never uses it, so a
        // failed lookup is passed through and only reported if it is needed.
        let split_device = expert_device;
        let (ups, up_dev) = match expert_parts_raw(src, &up_name, cfg.n_expert, split_device, &candle)? {
            Some(parts) => parts,
            None => {
                let et = src.expert_tensor(&up_name, cfg.n_expert)?;
                expert_parts(src, &et, &up_name, split_device, &candle)?
            }
        };
        let (downs, down_dev) = match expert_parts_raw(src, &down_name, cfg.n_expert, split_device, &candle)? {
            Some(parts) => parts,
            None => {
                let et = src.expert_tensor(&down_name, cfg.n_expert)?;
                expert_parts(src, &et, &down_name, split_device, &candle)?
            }
        };
        // Both halves of an expert must live together, or neither the gate/up
        // nor the down matmul could run.
        if !up_dev.same_device(&down_dev) {
            candle_core::bail!(
                "{}: `{p}` routed experts resolved to two different devices ({up_dev:?}, {down_dev:?})",
                src.arch
            );
        }
        let experts = ups
            .into_iter()
            .zip(downs)
            .map(|(up, down)| Expert { up, down })
            .collect();
        Ok(Self {
            router,
            gate_bias,
            experts,
            shared_up: load_weight(src, &format!("{p}.ffn_up_shexp.weight"))?,
            shared_down: load_weight(src, &format!("{p}.ffn_down_shexp.weight"))?,
            latent_down: if src.has(&format!("{p}.ffn_latent_down.weight")) {
                Some(load_weight(src, &format!("{p}.ffn_latent_down.weight"))?)
            } else {
                None
            },
            latent_up: if src.has(&format!("{p}.ffn_latent_up.weight")) {
                Some(load_weight(src, &format!("{p}.ffn_latent_up.weight"))?)
            } else {
                None
            },
            n_expert: cfg.n_expert,
            k: cfg.n_expert_used,
            scale: cfg.expert_weights_scale,
            norm_weights: cfg.expert_weights_norm,
            n_embd: cfg.n_embd,
            expert_device: up_dev,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, h) = xs.dims3()?;
        let x2 = xs.reshape((b * t, h))?;
        let logits = x2.matmul(&self.router)?; // [n, n_expert]
                                               // Nemotron-H routes with an independent sigmoid per expert.
        let scores = Gating::Sigmoid.scores(&logits)?;
        // `moe::route` selects on `scores + bias` and weights by the unbiased
        // scores, which is llama.cpp's contract for `ffn_exp_probs_b`.
        let (idx, weights) = crate::moe::route(
            &scores,
            self.gate_bias.as_ref(),
            self.k,
            self.norm_weights,
            self.scale,
            Ok,
        )?;

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

// ─── MTP / NextN draft block ─────────────────────────────────────────────────

/// The NextN / MTP draft block Nemotron 3.5 appends after the trunk
/// (`num_nextn_predict_layers = 1`; `blk.{n_layer}.*`).
///
/// The block folds an **attention** sub-layer and a **MoE** sub-layer into one
/// trailing block, and reads a token's embedding together with the trunk's
/// normed hidden of the *previous* position:
///
/// ```text
/// row q = eh_proj([enorm(emb(t_q)) ‖ hnorm(h_{q-1})]) → attn → MoE
/// ```
///
/// then `shared_head_norm` and the trunk's output head.  A chained draft seeds
/// the next row with that normed output instead of a trunk hidden.  Used as a
/// speculative drafter (see [`crate::native_session::Session::mtp_draft`]); the
/// state is the pair of sub-layer states this struct keeps.
struct MtpBlock {
    enorm: RmsNorm,
    hnorm: RmsNorm,
    eh_proj: Weight,
    attn_norm: RmsNorm,
    attn: AttentionBlock,
    post_norm: RmsNorm,
    moe: MoeBlock,
    /// `nextn.shared_head_norm`; the trunk's output norm when absent.
    head_norm: Option<RmsNorm>,
}

impl MtpBlock {
    fn load<R: Read + Seek>(
        src: &mut TensorSource<R>,
        cfg: &Config,
        rope: Arc<Rope>,
        expert_device: &Device,
    ) -> Result<Option<Self>> {
        let p = format!("blk.{}", cfg.n_layer);
        if !src.has(&format!("{p}.nextn.eh_proj.weight")) {
            return Ok(None);
        }
        let device = src.device.clone();
        Ok(Some(Self {
            enorm: src.rms_norm(&format!("{p}.nextn.enorm.weight"), cfg.rms_eps)?,
            hnorm: src.rms_norm(&format!("{p}.nextn.hnorm.weight"), cfg.rms_eps)?,
            eh_proj: load_weight(src, &format!("{p}.nextn.eh_proj.weight"))?,
            attn_norm: src.rms_norm(&format!("{p}.attn_norm.weight"), cfg.rms_eps)?,
            attn: AttentionBlock::load(src, &p, cfg, rope)?,
            post_norm: src.rms_norm(&format!("{p}.attn_post_norm.weight"), cfg.rms_eps)?,
            moe: MoeBlock::load(src, &p, cfg, expert_device)?,
            head_norm: src.rms_norm_opt(
                &format!("{p}.nextn.shared_head_norm.weight"),
                cfg.rms_eps,
            )?,
        }))
    }

    /// The block's normed output for `n` rows (`[1, n, hidden]`), advancing
    /// `state` (its attention KV).
    fn forward(
        &self,
        state: &mut LayerState,
        emb: &Tensor,
        seeds: &Tensor,
        trunk_norm: &RmsNorm,
        input: &LayerInput<'_>,
    ) -> Result<Tensor> {
        // eh_proj([enorm(emb) ‖ hnorm(seed)])
        let e = self.enorm.forward(emb)?;
        let h = self.hnorm.forward(seeds)?;
        let merged = Tensor::cat(&[&e, &h], D::Minus1)?;
        let mut x = self.eh_proj.forward(&merged)?;
        // attention sub-layer, then the MoE sub-layer, each pre-normed.
        let a = self.attn_norm.forward(&x)?;
        x = x.add(&self.attn.forward(&mut state.kv, &a, input.mask, input.offset)?)?;
        let m = self.post_norm.forward(&x)?;
        x = x.add(&self.moe.forward(&m)?)?;
        match &self.head_norm {
            Some(n) => n.forward(&x),
            // No `shared_head_norm`: the *trunk's* output norm is the
            // fallback, not "no norm" — `mtp_draft` treats this tensor as
            // already normed (it goes straight through the output head and
            // seeds the next draft), so returning a raw residual would skew
            // both the logits and every chained draft.
            None => trunk_norm.forward(&x),
        }
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
    /// The NextN / MTP draft block, when the file has one and the caller
    /// asked for it (see [`crate::quantized_qwen::with_mtp_block`]).
    mtp: Option<MtpBlock>,
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
            Weight::Quant(src.qmatmul("output.weight")?)
        } else {
            // Tied embeddings: reuse the (quantized) embedding table.
            Weight::Quant(src.qmatmul("token_embd.weight")?)
        };
        // The MTP block is a whole extra attention + MoE layer, so it is only
        // loaded when a caller says it will draft with it.
        let mtp = if crate::quantized_qwen::mtp_wanted() {
            MtpBlock::load(&mut src, &cfg, rope, &expert_device)?
        } else {
            None
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
            mtp,
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
        self.head_logits(&self.head_norm(xs)?)
    }

    fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    fn head_norm(&self, xs: &Tensor) -> Result<Tensor> {
        self.norm.forward(xs)
    }

    fn head_logits(&self, normed: &Tensor) -> Result<Tensor> {
        self.output.forward(normed)?.to_dtype(DType::F32)
    }

    fn mtp_forward(
        &self,
        state: &mut LayerState,
        emb: &Tensor,
        seeds: &Tensor,
        input: &LayerInput<'_>,
    ) -> Result<Tensor> {
        let Some(m) = &self.mtp else {
            candle_core::bail!("{}: no MTP block loaded", ARCHES[0]);
        };
        m.forward(state, emb, seeds, &self.norm, input)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// An NVFP4 `Weight` must decode in-kernel and agree with the reference
    /// decoder — routing NVFP4 through this loader must not change what the
    /// matmul computes, only where the decoding happens.
    #[test]
    fn an_nvfp4_weight_matches_the_reference_matmul() {
        use crate::mxfp4::{BlockNvfp4, QK_NVFP4};
        use crate::raw_block::RawBlock;
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (16usize, 128usize);
        let blocks_per_row = in_dim / QK_NVFP4;
        let row_bytes = blocks_per_row * std::mem::size_of::<BlockNvfp4>();
        // Deterministic pseudo-random blocks; every byte pattern is a valid
        // NVFP4 block (all 4-bit codes are in range, all scales finite).
        let mut state = 0x1234_5678u32;
        let raw: Vec<u8> = (0..out_dim * row_bytes)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                (state >> 16) as u8
            })
            .collect();
        let blocks = Tensor::from_vec(raw.clone(), (out_dim, row_bytes), &dev).unwrap();
        let w = Weight::Nvfp4 { blocks: blocks.clone(), out_dim, in_dim };

        let acts = Tensor::randn(0f32, 1.0, (4, in_dim), &dev).unwrap();
        let got = w.forward(&acts).unwrap();

        // The reference: the same block bytes through the scalar decoder.
        let act_host = acts.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut want = vec![0f32; 4 * out_dim];
        crate::nvfp4_gemv::matmul_t(&raw, &act_host, &mut want, 4, out_dim, in_dim).unwrap();

        let a = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got.dims(), &[4, out_dim]);
        // Compare against a reference computed from the same bytes; both are
        // the same decode, so this is exact up to summation order.
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let worst = a.iter().zip(&want).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max) / scale;
        assert!(worst < 1e-6, "NVFP4 weight vs reference matmul: rel {worst:e}");
        let _ = <BlockNvfp4 as RawBlock>::GGML_TYPE;
    }

    /// A **non-square** NVFP4 projection must be read with `out_dim` from the
    /// first (already row-major) dimension.
    ///
    /// `RawTensorInfo.dims` is `[out, in]` because `read_header` reverses
    /// GGUF's fastest-varying-first order.  Reading it as disk order and
    /// swapping back transposes the matrix: the same bytes become a different
    /// shape, and the GEMV then rejects ordinary activations.  A square matrix
    /// cannot tell the two apart, so this uses one where out != in and checks
    /// both the accepted activation width and the values.
    #[test]
    fn a_non_square_nvfp4_weight_keeps_its_orientation() {
        use crate::mxfp4::{BlockNvfp4, QK_NVFP4};
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (48usize, 128usize); // deliberately non-square
        let row_bytes = (in_dim / QK_NVFP4) * std::mem::size_of::<BlockNvfp4>();
        let mut state = 0xdead_beefu32;
        let raw: Vec<u8> = (0..out_dim * row_bytes)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                (state >> 16) as u8
            })
            .collect();
        let blocks = Tensor::from_vec(raw.clone(), (out_dim, row_bytes), &dev).unwrap();
        let w = Weight::Nvfp4 { blocks, out_dim, in_dim };
        assert_eq!(w.shape(), Some((out_dim, in_dim)), "shape is [out, in]");

        // An activation of the declared input width must be accepted and give
        // `[rows, out_dim]`.
        let acts = Tensor::randn(0f32, 1.0, (3, in_dim), &dev).unwrap();
        let got = w.forward(&acts).unwrap();
        assert_eq!(got.dims(), &[3, out_dim], "out_dim, not in_dim, is the width");

        // And it must equal the reference over the same bytes.
        let act_host = acts.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut want = vec![0f32; 3 * out_dim];
        crate::nvfp4_gemv::matmul_t(&raw, &act_host, &mut want, 3, out_dim, in_dim).unwrap();
        let a = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let worst = a.iter().zip(&want).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max) / scale;
        assert!(worst < 1e-6, "non-square NVFP4 weight: rel {worst:e}");

        // The transposed reading would have demanded `out_dim`-wide input.
        let wrong = Tensor::randn(0f32, 1.0, (3, out_dim), &dev).unwrap();
        assert!(
            w.forward(&wrong).is_err(),
            "an activation of the wrong width must be rejected, not reinterpreted"
        );
    }

    /// A prefill chunk is far longer than the op's 16-row limit, so the
    /// wrapper must batch the rows rather than hand the whole chunk over.  The
    /// default chunk is 512 tokens, i.e. 32 batches — a prompt of 17 tokens is
    /// enough to fail without this.
    #[test]
    fn an_nvfp4_weight_handles_a_full_prefill_chunk() {
        use crate::mxfp4::{BlockNvfp4, QK_NVFP4};
        let dev = Device::Cpu;
        let (out_dim, in_dim, tokens) = (24usize, 128usize, 37usize);
        assert!(tokens > crate::nvfp4_gemv::MAX_ROWS, "exceeds the op's row limit");
        let row_bytes = (in_dim / QK_NVFP4) * std::mem::size_of::<BlockNvfp4>();
        let mut state = 0x5eed_u32;
        let raw: Vec<u8> = (0..out_dim * row_bytes)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                (state >> 16) as u8
            })
            .collect();
        let blocks = Tensor::from_vec(raw.clone(), (out_dim, row_bytes), &dev).unwrap();
        let w = Weight::Nvfp4 { blocks, out_dim, in_dim };

        let acts = Tensor::randn(0f32, 1.0, (1, tokens, in_dim), &dev).unwrap();
        let got = w.forward(&acts).expect("a 37-token chunk must be batched, not rejected");
        assert_eq!(got.dims(), &[1, tokens, out_dim]);

        // Batched must equal the single-pass reference over all rows: the split
        // is on the token axis, so no row may be reordered or dropped.
        let act_host = acts.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut want = vec![0f32; tokens * out_dim];
        crate::nvfp4_gemv::matmul_t(&raw, &act_host, &mut want, tokens, out_dim, in_dim).unwrap();
        let a = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let worst = a.iter().zip(&want).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max) / scale;
        assert!(worst < 1e-6, "batched prefill vs single-pass: rel {worst:e}");
    }

    /// The loader must derive the released checkpoint's dimensions from that
    /// checkpoint's own metadata.
    ///
    /// This drives `Config::from_meta` with the real GGUF hyperparameters
    /// (taken from `NVIDIA-Nemotron-3.5-Lightning-30B-A3B`'s config, converted
    /// the way llama.cpp's `nemotron-h.cpp` reads them) and asserts every
    /// derived width against the shape the checkpoint actually stores.  That
    /// matters because the identity comparisons an earlier version used —
    /// literal against literal — could not catch a change to `from_meta`;
    /// reading a real checkpoint is also how the missing router bias surfaced.
    #[test]
    fn config_derives_the_released_checkpoint_shapes() {
        use candle_core::quantized::gguf_file::Value;
        let mut md: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
        let k = |s: &str| format!("nemotron_h.{s}");
        let mut put = |s: &str, v: Value| {
            md.insert(k(s), v);
        };
        put("block_count", Value::U32(53)); // 52 trunk + 1 NextN
        put("nextn_predict_layers", Value::U32(1));
        put("embedding_length", Value::U32(2688));
        put("attention.head_count", Value::U32(32));
        put("attention.key_length", Value::U32(128));
        put("attention.layer_norm_rms_epsilon", Value::F32(1e-5));
        put("rope.freq_base", Value::F32(10_000.0));
        put("ssm.conv_kernel", Value::U32(4));
        put("ssm.inner_size", Value::U32(4096));
        put("ssm.state_size", Value::U32(128));
        put("ssm.time_step_rank", Value::U32(64));
        put("ssm.group_count", Value::U32(8));
        put("expert_count", Value::U32(128));
        put("expert_used_count", Value::U32(6));
        put("expert_weights_scale", Value::F32(2.5));
        put("expert_weights_norm", Value::Bool(true));
        // The released `layers_block_type`: 23 `mamba`, 23 `moe`, 6
        // `attention`, at exactly these indexes.  Each is a *single* kind, so a
        // `mamba` layer has both `head_count_kv` and `feed_forward_length` zero
        // — that pair is llama.cpp's whole recurrent test.
        const ATTN: [u32; 6] = [5, 12, 19, 26, 33, 42];
        const MOE: [u32; 23] = [
            1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31, 34, 36, 38, 40, 43, 45, 47, 49,
            51,
        ];
        let kv: Vec<Value> = (0..52u32)
            .map(|i| Value::U32(if ATTN.contains(&i) { 2 } else { 0 }))
            .collect();
        let ff: Vec<Value> = (0..52u32)
            .map(|i| Value::U32(if MOE.contains(&i) { 1856 } else { 0 }))
            .collect();
        put("attention.head_count_kv", Value::Array(kv));
        put("feed_forward_length", Value::Array(ff));

        let meta = Meta::new(&md, "nemotron_h");
        let cfg = Config::from_meta(&meta)
            .expect("metadata parses")
            .expect("recognised architecture");

        // Trunk excludes the NextN block.
        assert_eq!(cfg.n_layer, 52, "block_count 53 - nextn 1");
        // What the checkpoint stores, per its index.json and shard headers.
        assert_eq!(cfg.ssm.d_in_proj(), 10_304); // in_proj [10304, 2688]
        assert_eq!(cfg.ssm.conv_dim(), 6_144); // conv1d [6144, 1, 4]
        assert_eq!(cfg.ssm.d_inner, 4_096); // out_proj [2688, 4096]
        assert_eq!(
            (cfg.ssm.d_inner / cfg.ssm.n_group, cfg.ssm.n_group),
            (512, 8) // norm [4096] -> {512, 8}
        );
        assert_eq!(cfg.ssm.n_head, 64); // A_log/D/dt_bias [64]
        assert_eq!(cfg.ssm.head_dim(), 64); // mamba_head_dim, not attention's 128
        // Attention: q [4096, 2688], k/v [256, 2688].
        assert_eq!((cfg.n_head, cfg.head_dim, cfg.n_embd), (32, 128, 2688));
        assert_eq!(cfg.n_head * cfg.head_dim, 4_096);
        assert_eq!(cfg.n_head_kv * cfg.head_dim, 256);
        // Layer kinds: 6 attention, MoE for the rest, none left as mamba by
        // the kv/ffn rule.
        let attn = cfg.layers.iter().filter(|l| l.kind == LayerKind::Attention).count();
        let mamba = cfg.layers.iter().filter(|l| l.kind == LayerKind::Mamba).count();
        let moe = cfg.layers.iter().filter(|l| l.kind == LayerKind::Moe).count();
        assert_eq!((attn, mamba, moe), (6, 23, 23), "the released layer pattern");
        // Spot-check the classification at the boundaries rather than trusting
        // the counts alone.
        assert_eq!(cfg.layers[0].kind, LayerKind::Mamba);
        assert_eq!(cfg.layers[1].kind, LayerKind::Moe);
        assert_eq!(cfg.layers[5].kind, LayerKind::Attention);
        assert_eq!(cfg.layers[33].kind, LayerKind::Attention);
        assert_eq!(cfg.layers[42].kind, LayerKind::Attention);
        assert_eq!(cfg.layers[51].kind, LayerKind::Moe);

        // And the MTP block is addressed past the trunk.
        assert_eq!(format!("blk.{}", cfg.n_layer), "blk.52");
    }

    /// `Mamba2::conv` must dispatch the native kernel, not fall back to a
    /// per-tap tensor loop.  The op has its own unit tests, but only this
    /// reaches the layer's call site: a layer that stopped calling
    /// `crate::conv1d::conv1d` would still pass those.
    #[cfg(feature = "opencl")]
    #[test]
    fn the_mamba_layer_convolves_natively() {
        let Ok(dev) = Device::new_opencl(0) else {
            eprintln!("no OpenCL device; skipping");
            return;
        };
        let cfg = SsmConfig {
            d_conv: 4,
            d_state: 8,
            d_inner: 32,
            n_head: 4,
            n_group: 2,
            time_step_min: 0.0,
        };
        // Only the conv fields are exercised, so the projection weights are
        // never touched and can be trivial.
        let conv_dim = cfg.conv_dim();
        let layer = Mamba2 {
            in_proj: Weight::Quant(candle_core::quantized::QMatMul::QTensor(std::sync::Arc::new(
                candle_core::quantized::QTensor::quantize(
                    &Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
                    candle_core::quantized::GgmlDType::F32,
                )
                .unwrap(),
            ))),
            conv_w: Tensor::randn(0f32, 0.1, (cfg.d_conv, conv_dim), &dev).unwrap(),
            conv_b: Some(Tensor::randn(0f32, 0.1, (conv_dim,), &dev).unwrap()),
            dt_b: Tensor::zeros((cfg.n_head,), DType::F32, &dev).unwrap(),
            a: Tensor::zeros((cfg.n_head,), DType::F32, &dev).unwrap(),
            d_skip: Tensor::zeros((cfg.n_head,), DType::F32, &dev).unwrap(),
            norm_w: Tensor::ones((1, cfg.n_group, cfg.d_inner / cfg.n_group), DType::F32, &dev)
                .unwrap(),
            eps: 1e-5,
            out: Weight::Quant(candle_core::quantized::QMatMul::QTensor(std::sync::Arc::new(
                candle_core::quantized::QTensor::quantize(
                    &Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
                    candle_core::quantized::GgmlDType::F32,
                )
                .unwrap(),
            ))),
            cfg,
        };
        let mut state = Some(layer.fresh_state(&dev).unwrap());
        let xbc = Tensor::randn(0f32, 1.0, (5, conv_dim), &dev).unwrap();

        let before = candle_core::opencl_backend::kernels::native_exec_count();
        let fb_before: usize = candle_core::opencl_backend::kernels::fallback_counts_by_op()
            .iter()
            .filter(|(n, _)| n == crate::conv1d::OP_NAME)
            .map(|(_, c)| *c)
            .sum();
        let y = layer.conv(state.as_mut().unwrap(), &xbc).unwrap();
        let launched = candle_core::opencl_backend::kernels::native_exec_count() - before;
        let fell_back: usize = candle_core::opencl_backend::kernels::fallback_counts_by_op()
            .iter()
            .filter(|(n, _)| n == crate::conv1d::OP_NAME)
            .map(|(_, c)| *c)
            .sum::<usize>()
            - fb_before;

        assert_eq!(y.dims(), &[5, conv_dim]);
        assert!(launched > 0, "the layer's conv must launch a native kernel");
        assert_eq!(fell_back, 0, "the layer's conv must not round-trip");
    }
}
