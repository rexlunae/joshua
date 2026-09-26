//! Building blocks shared by the mixture-of-experts loaders.
//!
//! `quantized_qwen` and `quantized_deepseek2` (and, for the parts that
//! apply, `quantized_deepseek4`) grew the same code side by side: the
//! per-layer KV cache and its truncation, the causal mask, the routed
//! expert dispatch with its host/device hop, and the slicing of a stacked
//! `[n_expert, out, in]` expert tensor into per-expert weights.  This module
//! holds the one copy.  What stays in each loader is what genuinely differs:
//! the router (softmax vs sigmoid, group limits, shared experts), the
//! attention (GQA vs MLA), and the weight representation (`qwen3moe` also
//! has zero-copy Metal and paged variants).

use std::borrow::Cow;
use std::io::{Read, Seek};
use std::sync::Arc;

use candle_core::quantized::{gguf_file, GgmlDType, QStorage, QTensor};
use candle_core::{DType, Device, Result, Tensor, D};
use memmap2::Mmap;

use crate::mmap_tensor::MmapPrefetch;

// ─── Tensor naming ───────────────────────────────────────────────────────────

/// Whether a GGUF tensor name belongs to the routed-expert set
/// (`.ffn_gate_exps` / `.ffn_up_exps` / `.ffn_down_exps`).
///
/// These are the only weights touched sparsely: a token routes through a
/// handful of the model's experts per layer, so readahead/prefetch drags in
/// far more than a token will use, and they are the only weights an
/// accelerator may leave in host RAM.  Everything else — embeddings, norms,
/// attention, routers, shared experts, indexer/compressor, output — is dense
/// and touched on every token.
pub fn is_routed_expert(name: &str) -> bool {
    name.contains(".ffn_gate_exps")
        || name.contains(".ffn_down_exps")
        || name.contains(".ffn_up_exps")
}

// ─── Per-session state ───────────────────────────────────────────────────────

/// One layer's KV cache: `(k, v)`, appended along the layer's sequence
/// dimension.  Owned by the session, not the (shared) weights.
pub type KvCache = Option<(Tensor, Tensor)>;

/// Keep only the first `keep` positions of a layer's cache, where the
/// sequence runs along `seq_dim` of both `k` and `v`.  `keep == 0` clears
/// the cache; `keep` at or past the cached length is a no-op.
pub fn truncate_kv(kv_cache: &mut KvCache, keep: usize, seq_dim: usize) -> Result<()> {
    match kv_cache {
        None => Ok(()),
        Some(_) if keep == 0 => {
            *kv_cache = None;
            Ok(())
        }
        Some((k, v)) => {
            let len = k.dim(seq_dim)?;
            if keep >= len {
                return Ok(());
            }
            let k = k.narrow(seq_dim, 0, keep)?.contiguous()?;
            let v = v.narrow(seq_dim, 0, keep)?.contiguous()?;
            *kv_cache = Some((k, v));
            Ok(())
        }
    }
}

/// Additive causal mask `[1, 1, seq_len, seq_len + offset]` for `seq_len`
/// new positions appended after `offset` cached ones: `0` where a query may
/// attend, `-inf` where it may not.
pub fn causal_mask(seq_len: usize, offset: usize, device: &Device) -> Result<Tensor> {
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
    Tensor::from_slice(&mask, (1, 1, seq_len, seq_len + offset), device)
}

// ─── Routing helpers ─────────────────────────────────────────────────────────

/// Indices of the top-`k` values along the last dim (descending), as u32.
pub fn topk_indices(t: &Tensor, k: usize) -> Result<Tensor> {
    t.arg_sort_last_dim(false)?
        .narrow(D::Minus1, 0, k)?
        .contiguous()
}

/// Numerically stable `log(1 + exp(x))`.
pub fn softplus(x: &Tensor) -> Result<Tensor> {
    let e = x.abs()?.neg()?.exp()?;
    x.maximum(0.0)?.add(&(e + 1.0)?.log()?)
}

/// Top-`k` values along the last dim (descending).
pub fn topk_values(t: &Tensor, k: usize) -> Result<Tensor> {
    let idx = topk_indices(t, k)?;
    t.gather(&idx, D::Minus1)
}

/// How router logits become per-expert scores (llama.cpp
/// `expert_gating_func`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gating {
    /// Softmax over the experts (DeepSeek-V2, the Qwen MoEs).
    Softmax,
    /// Independent sigmoid per expert (DeepSeek-V3, Kimi-K2, the GLM MoEs).
    Sigmoid,
    /// `sqrt(softplus(x))` (DeepSeek-V4).
    SqrtSoftplus,
}

impl Gating {
    /// Parse `expert_gating_func` (1 softmax, 2 sigmoid); `default` when the
    /// key is absent or 0 ("none").
    pub fn from_meta(m: &crate::gguf_meta::Meta<'_>, default: Self) -> Self {
        match m.u32_or("expert_gating_func", 0) {
            1 => Self::Softmax,
            2 => Self::Sigmoid,
            _ => default,
        }
    }

    pub fn scores(self, logits: &Tensor) -> Result<Tensor> {
        match self {
            Self::Softmax => candle_nn::ops::softmax_last_dim(logits),
            Self::Sigmoid => candle_nn::ops::sigmoid(logits),
            Self::SqrtSoftplus => softplus(logits)?.sqrt(),
        }
    }
}

/// The routing weights of the experts `idx` picked (`[n_tokens, k]`): their
/// scores, optionally renormalised to sum to one (min-clamped to the f16
/// epsilon, as llama.cpp does, so a degenerate routing never divides by
/// zero) and scaled (`0` / `1`: no scaling).
pub fn routing_weights(scores: &Tensor, idx: &Tensor, norm: bool, scale: f64) -> Result<Tensor> {
    let mut weights = scores.gather(idx, D::Minus1)?;
    if norm {
        let denom = weights.sum_keepdim(D::Minus1)?.clamp(6.103_515_6e-5, f32::INFINITY)?;
        weights = weights.broadcast_div(&denom)?;
    }
    if scale != 0.0 && scale != 1.0 {
        weights = (weights * scale)?;
    }
    Ok(weights)
}

/// Top-`k` routing over per-expert `scores` (`[n_tokens, n_expert]`):
/// experts are *selected* on `select(scores + bias)` — the aux-loss-free
/// balancing bias plus any group limit — and *weighted* by their unbiased
/// scores ([`routing_weights`]).  Returns `(ids, weights)`, both
/// `[n_tokens, k]`.
pub fn route(
    scores: &Tensor,
    bias: Option<&Tensor>,
    k: usize,
    norm: bool,
    scale: f64,
    select: impl FnOnce(Tensor) -> Result<Tensor>,
) -> Result<(Tensor, Tensor)> {
    let selection = match bias {
        Some(b) => scores.broadcast_add(&b.reshape((1, ()))?)?,
        None => scores.clone(),
    };
    let idx = topk_indices(&select(selection)?, k)?;
    let weights = routing_weights(scores, &idx, norm, scale)?;
    Ok((idx, weights))
}

// ─── Device expert cache helpers ─────────────────────────────────────────────

/// Per-slot byte estimate for a routed expert in a
/// [`crate::residency::DeviceResidency`] pool (the f32-dense form of a
/// ~400k-element expert).  `QMatMul` hides the raw sizes, so this is a round
/// heuristic; an overestimate only shrinks the reported capacity, never the
/// correctness.
pub const EXPERT_SLOT_BYTES_ESTIMATE: u64 = 400_000 * 4;

/// Upload a quantized `QMatMul`'s blocks onto `device` (backend-generic via
/// `QStorage::from_data` — CPU, Vulkan, Metal, OpenCL, CUDA).  `None` when
/// the matmul is not a raw `QMatMul::QTensor` or the upload fails; the
/// caller then keeps the host form for that expert (#62).
pub fn upload_qmatmul(q: &candle_core::quantized::QMatMul, device: &Device) -> Option<candle_core::quantized::QMatMul> {
    use candle_core::quantized::QMatMul;
    let QMatMul::QTensor(qt) = q else {
        return None;
    };
    if device.same_device(&qt.device()) {
        return Some(q.clone());
    }
    let bytes = qt.data().ok()?;
    let storage = QStorage::from_data(Cow::Borrowed(&bytes), device, qt.dtype()).ok()?;
    let qt = QTensor::new(storage, qt.shape().clone()).ok()?;
    Some(QMatMul::QTensor(Arc::new(qt)))
}

// ─── Routed-expert dispatch ──────────────────────────────────────────────────

/// `x` on `device`: borrowed when it already lives there, a copy otherwise.
///
/// The routed experts may live on a different device from the activations
/// (a host-resident expert pool on an accelerator, see
/// [`crate::placement::ExpertPlacement`]).  The MoE dispatch moves the
/// whole block's input across once and its result back once, so the
/// boundary costs two transfers per layer instead of one per expert matmul.
pub fn on_device<'a>(x: &'a Tensor, device: &Device) -> Result<Cow<'a, Tensor>> {
    if x.device().same_device(device) {
        Ok(Cow::Borrowed(x))
    } else {
        Ok(Cow::Owned(x.to_device(device)?))
    }
}

/// One routed expert: a gate/up/down MLP over `[n, hidden]` rows.
pub trait Expert {
    /// Run the expert over `xs` (`[n, hidden]` → `[n, hidden]`).
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

/// Per-expert forward used by the MoE dispatch, with optional residency
/// (device-resident hot experts, #62).
///
/// The resolver receives the expert index and the block activation and runs
/// that expert's *correct form*: the device-resident weights on the expert
/// device when a hit, the host weights otherwise.  `None` (the default) makes
/// the dispatch call `experts[e].forward` exactly as before, so an
/// unprefixed model is byte-for-byte unchanged.
pub type ResidencyForward<'a> = Option<&'a dyn Fn(usize, &Tensor) -> Result<Tensor>>;

/// Router output for one MoE block, drained to the host once.
struct Routing {
    /// Selected expert ids, `[n_tokens * k]`, token-major.
    ids: Vec<u32>,
    /// Matching routing weights.
    weights: Vec<f32>,
}

impl Routing {
    fn read(topk_idx: &Tensor, weights: &Tensor) -> Result<Self> {
        Ok(Self {
            ids: topk_idx.flatten_all()?.to_vec1()?,
            weights: weights.flatten_all()?.to_vec1()?,
        })
    }
}

/// A router id that names no expert must fail loudly with context rather
/// than panic on the slice index below.
fn check_expert_id(
    arch: &str,
    e: usize,
    n_experts: usize,
    token: usize,
    slot: usize,
) -> Result<()> {
    if e >= n_experts {
        candle_core::bail!(
            "{arch}: router selected expert {e} out of {n_experts} (token {token}, slot {slot})"
        );
    }
    Ok(())
}

/// Run a routed MoE block over `n_tokens` rows (prefill).
///
/// `x2` is `[n_tokens, hidden]`, `topk_idx` / `weights` are `[n_tokens, k]`;
/// returns the weighted expert sum `[n_tokens, hidden]` on `x2`'s device
/// plus the routed ids (token-major, `n_tokens * k`) for the caller's
/// hot-expert bookkeeping.  Tokens are bucketed by expert so each expert
/// runs once over its token block.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_prefill<E: Expert>(
    arch: &str,
    experts: &[E],
    expert_device: &Device,
    x2: &Tensor,
    topk_idx: &Tensor,
    weights: &Tensor,
    n_tokens: usize,
    k: usize,
) -> Result<(Tensor, Vec<u32>)> {
    dispatch_prefill_with(arch, experts, expert_device, x2, topk_idx, weights, n_tokens, k, None)
}

/// [`dispatch_prefill`] with an optional residency-aware per-expert forward.
/// When `forward_expert` is `None` this is byte-for-byte [`dispatch_prefill`].
#[allow(clippy::too_many_arguments)]
pub fn dispatch_prefill_with<E: Expert>(
    arch: &str,
    experts: &[E],
    expert_device: &Device,
    x2: &Tensor,
    topk_idx: &Tensor,
    weights: &Tensor,
    n_tokens: usize,
    k: usize,
    forward_expert: ResidencyForward<'_>,
) -> Result<(Tensor, Vec<u32>)> {
    let h = x2.dim(1)?;
    let routing = Routing::read(topk_idx, weights)?;
    let out_device = x2.device().clone();
    let x2 = on_device(x2, expert_device)?;

    // Bucket (token, weight) pairs by expert.
    let mut per_expert: Vec<Vec<(u32, f32)>> = vec![Vec::new(); experts.len()];
    for t in 0..n_tokens {
        for s in 0..k {
            let e = routing.ids[t * k + s] as usize;
            check_expert_id(arch, e, experts.len(), t, s)?;
            per_expert[e].push((t as u32, routing.weights[t * k + s]));
        }
    }

    let default_fwd = |e: usize, x: &Tensor| experts[e].forward(x);
    let fwd = forward_expert.unwrap_or(&default_fwd);
    let mut y = Tensor::zeros((n_tokens, h), DType::F32, expert_device)?;
    for (e, bucket) in per_expert.iter().enumerate() {
        if bucket.is_empty() {
            continue;
        }
        let token_idx: Vec<u32> = bucket.iter().map(|(t, _)| *t).collect();
        let w: Vec<f32> = bucket.iter().map(|(_, w)| *w).collect();
        let count = token_idx.len();
        let idx = Tensor::from_vec(token_idx, count, expert_device)?;
        let x_sel = x2.index_select(&idx, 0)?; // [count, h]
        let out = fwd(e, &x_sel)?; // [count, h]
        let w = Tensor::from_vec(w, (count, 1), expert_device)?;
        y = y.index_add(&idx, &out.broadcast_mul(&w)?, 0)?;
    }
    let y = on_device(&y, &out_device)?.into_owned();
    Ok((y, routing.ids))
}

/// Run a routed MoE block over one row (decode).
///
/// `x2` is `[1, hidden]`, `topk_idx` / `weights` are `[1, k]`; returns the
/// weighted expert sum `[1, hidden]` on `x2`'s device plus the `k` routed
/// ids.  Deliberately serial: candle's own CPU kernels already fan out
/// through its global rayon pool per op, and an outer parallel loop over
/// experts measurably *regresses* decode (~20 %) through nested-pool
/// contention; the memory streams of `k` experts overlap fine within that
/// pool.
pub fn dispatch_decode<E: Expert>(
    arch: &str,
    experts: &[E],
    expert_device: &Device,
    x2: &Tensor,
    topk_idx: &Tensor,
    weights: &Tensor,
    k: usize,
) -> Result<(Tensor, Vec<u32>)> {
    dispatch_decode_with(arch, experts, expert_device, x2, topk_idx, weights, k, None)
}

/// [`dispatch_decode`] with an optional residency-aware per-expert forward.
///
/// When `forward_expert` is `None`, this is byte-for-byte [`dispatch_decode`].
/// When present, each routed expert runs through it: on a device-resident
/// hit it runs the uploaded weights (moving `xs` to the expert device); on a
/// miss it runs the host form. This is what lets the #62 device-expert cache
/// serve hot experts from a non-CUDA device while the rest stay host.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_decode_with<E: Expert>(
    arch: &str,
    experts: &[E],
    expert_device: &Device,
    x2: &Tensor,
    topk_idx: &Tensor,
    weights: &Tensor,
    k: usize,
    forward_expert: ResidencyForward<'_>,
) -> Result<(Tensor, Vec<u32>)> {
    let ids: Vec<u32> = topk_idx.flatten_all()?.to_vec1()?; // [k] — the one host sync
    let out_device = x2.device().clone();
    let x2 = on_device(x2, expert_device)?;
    let default_fwd = |e: usize, x: &Tensor| experts[e].forward(x);
    let fwd = forward_expert.unwrap_or(&default_fwd);
    let mut outs = Vec::with_capacity(k);
    for (s, &e) in ids.iter().enumerate() {
        check_expert_id(arch, e as usize, experts.len(), 0, s)?;
        outs.push(fwd(e as usize, &x2)?); // [1, h] each
    }
    let out = Tensor::stack(&outs, 0)?; // [k, 1, h]
    let out = on_device(&out, &out_device)?;
    let w = weights.reshape((k, 1, 1))?; // [k, 1, 1]
    let y = out.broadcast_mul(&w)?.sum(0)?; // [1, h]
    Ok((y, ids))
}

// ─── Stacked expert tensors ──────────────────────────────────────────────────

/// One expert borrowed in place from the model mapping: the quantized
/// tensor and, when the dtype has a prefetch implementation, a handle that
/// can `MADV_WILLNEED` its byte range ahead of a matmul.
pub struct BorrowedExpert {
    pub tensor: QTensor,
    pub prefetch: Option<Arc<dyn MmapPrefetch>>,
}

/// Slice a stacked routed-expert tensor in a format candle cannot name
/// ([`crate::raw_block::RawBlock`]: IQ2_XXS, MXFP4, …; `[n_expert, out,
/// in]`) into per-expert weights whose blocks stay in the mapping and are
/// decoded at matmul time.  Without a mapping (tests) the tensor is decoded
/// to f32 once; on an accelerator expert device that would materialise the
/// whole pool, so it is refused.
#[allow(clippy::too_many_arguments)]
pub fn split_raw_experts<B: crate::raw_block::RawBlock, R: Read + Seek>(
    arch: &str,
    raw: &crate::gguf_ext::GgufHeader,
    reader: &mut R,
    mmap: Option<&Arc<Mmap>>,
    expert_device: &Device,
    name: &str,
    info: &crate::gguf_ext::RawTensorInfo,
    n_expert: usize,
) -> Result<Vec<BorrowedExpert>> {
    let dims = info.dims.clone();
    if dims.len() != 3 || dims[0] != n_expert {
        candle_core::bail!("{arch}: expected expert tensor `{name}` shaped [n_expert, out, in], got {dims:?}");
    }
    let (out, inn) = (dims[1], dims[2]);
    let per_elems = out * inn;
    let Some(per_bytes) = crate::raw_block::size_bytes::<B>(per_elems) else {
        candle_core::bail!(
            "{arch}: {} expert `{name}` rows {out}x{inn} are not a multiple of {}",
            B::NAME,
            B::QK
        );
    };

    // Zero-copy: one borrowed QTensor per expert, pointing into the mapping.
    // A borrow is `QStorage::Cpu` by construction, so it is only taken when
    // the experts' home device is the CPU — which it is for every mapped
    // model, including one whose dense set runs on an accelerator; it is
    // also the source a device expert cache uploads an active expert from.
    if let Some(mmap) = mmap.filter(|_| expert_device.is_cpu()) {
        let base = raw.tensor_data_offset.saturating_add(info.offset) as usize;
        let mut experts = Vec::with_capacity(n_expert);
        for e in 0..n_expert {
            let at = base + e * per_bytes;
            match crate::mmap_tensor::borrowed_range_raw::<B>(mmap, at, (out, inn).into())? {
                Some(tensor) => experts.push(BorrowedExpert {
                    tensor,
                    prefetch: crate::mmap_tensor::prefetch_handle_raw::<B>(mmap, at, per_elems / B::QK),
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
        tracing::warn!("{arch}: could not borrow `{name}` from the mapping, decoding to f32");
    }

    // The fallback below materializes the whole stacked tensor as f32 — an
    // order-of-magnitude blow-up for real model footprints — and accelerator
    // devices cannot borrow the blocks (they are CPU storage).  Refuse
    // loudly rather than OOM.
    if !expert_device.is_cpu() {
        candle_core::bail!("{arch}: {} expert tensor `{name}` is only supported on the CPU device", B::NAME);
    }
    let all = crate::raw_block::decode_bytes::<B>(&raw.read_tensor_bytes(reader, name)?, info.elem_count())?;
    (0..n_expert)
        .map(|e| {
            let t = Tensor::from_vec(all[e * per_elems..(e + 1) * per_elems].to_vec(), (out, inn), expert_device)?;
            Ok(BorrowedExpert { tensor: QTensor::quantize(&t, GgmlDType::F32)?, prefetch: None })
        })
        .collect()
}

/// A stacked routed-expert tensor `[n_expert, out, in]` as the GGUF header
/// describes it, with the three ways of turning it into per-expert weights:
/// borrow each expert from the mapping ([`ExpertTensor::borrow_host`]),
/// upload each expert straight from the mapping
/// ([`ExpertTensor::upload_from_mmap`]), or read the whole tensor once and
/// split it on the host ([`ExpertTensor::read_and_split`]).
pub struct ExpertTensor {
    name: String,
    dtype: GgmlDType,
    n_expert: usize,
    out: usize,
    inn: usize,
    /// Absolute byte offset of the tensor's data in the file / mapping.
    base: usize,
}

impl ExpertTensor {
    /// Describe tensor `name`, which must be `[n_expert, out, in]`.
    pub fn lookup(
        ct: &gguf_file::Content,
        arch: &str,
        name: &str,
        n_expert: usize,
    ) -> Result<Self> {
        let Some(info) = ct.tensor_infos.get(name) else {
            candle_core::bail!("{arch}: missing tensor `{name}`");
        };
        let dims = info.shape.dims();
        if dims.len() != 3 || dims[0] != n_expert {
            candle_core::bail!(
                "{arch}: expected expert tensor `{name}` shaped [n_expert, out, in], got {dims:?}"
            );
        }
        Ok(Self {
            name: name.to_string(),
            dtype: info.ggml_dtype,
            n_expert,
            out: dims[1],
            inn: dims[2],
            base: ct.tensor_data_offset.saturating_add(info.offset) as usize,
        })
    }

    /// The tensor's name in the header.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Quantized dtype of the blocks.
    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    /// Number of experts stacked in the tensor.
    pub fn n_expert(&self) -> usize {
        self.n_expert
    }

    /// One expert's `[out, in]` shape.
    pub fn expert_shape(&self) -> (usize, usize) {
        (self.out, self.inn)
    }

    /// Elements per expert.
    pub fn per_elems(&self) -> usize {
        self.out * self.inn
    }

    /// Absolute byte offset of the tensor's data.
    pub fn base(&self) -> usize {
        self.base
    }

    /// Blocks per expert, when an expert is a whole number of blocks
    /// (a `None` means the tensor cannot be sliced per expert at all).
    pub fn blocks_per_expert(&self) -> Option<usize> {
        let block_size = self.dtype.block_size();
        (block_size > 0 && self.per_elems().is_multiple_of(block_size))
            .then(|| self.per_elems() / block_size)
    }

    /// Bytes per expert, when an expert is a whole number of blocks.
    pub fn bytes_per_expert(&self) -> Option<usize> {
        self.blocks_per_expert()
            .and_then(|blocks| blocks.checked_mul(self.dtype.type_size()))
    }

    /// Absolute byte offset of expert `e`, when the tensor slices per expert.
    pub fn expert_offset(&self, e: usize) -> Option<usize> {
        self.bytes_per_expert()
            .and_then(|per| per.checked_mul(e))
            .and_then(|off| self.base.checked_add(off))
    }

    /// Point each expert straight at its own slice of `mmap`.
    ///
    /// Building all of them reads nothing — an expert is a pointer and a
    /// length — so a layer with hundreds of experts costs almost no memory
    /// until tokens actually route to them, at which point the kernel faults
    /// in just those pages and can evict them again later.  Borrowed storage
    /// is always CPU-resident; on an accelerator this is the
    /// `ExpertPlacement::Host` layout.  Returns `None` when any expert
    /// cannot be borrowed (misalignment, truncated file, a dtype without a
    /// borrow implementation): the whole layer then takes a copying path
    /// rather than mixing the two.
    pub fn borrow_host(&self, mmap: &Arc<Mmap>) -> Result<Option<Vec<BorrowedExpert>>> {
        let Some(blocks) = self.blocks_per_expert() else {
            return Ok(None);
        };
        let mut borrowed = Vec::with_capacity(self.n_expert);
        for e in 0..self.n_expert {
            let Some(offset) = self.expert_offset(e) else {
                return Ok(None);
            };
            match crate::mmap_tensor::borrowed_range(
                mmap,
                self.dtype,
                offset,
                self.expert_shape().into(),
            )? {
                Some(tensor) => borrowed.push(BorrowedExpert {
                    tensor,
                    prefetch: crate::mmap_tensor::prefetch_handle(mmap, self.dtype, offset, blocks),
                }),
                None => return Ok(None),
            }
        }
        Ok(Some(borrowed))
    }

    /// Upload each expert to `device` straight from its bytes in `mmap`.
    ///
    /// Each expert is one copy from the page cache: no host `Vec` of the
    /// whole tensor, no transient whole-tensor device buffer, no download
    /// and re-upload per expert.  Returns `None` when the tensor cannot be
    /// sliced per expert (see [`crate::mmap_tensor::expert_slices`]).
    pub fn upload_from_mmap(&self, mmap: &Mmap, device: &Device) -> Result<Option<Vec<QTensor>>> {
        let Some(slices) = crate::mmap_tensor::expert_slices(
            mmap,
            self.dtype,
            self.base,
            self.n_expert,
            self.per_elems(),
        ) else {
            return Ok(None);
        };
        slices
            .into_iter()
            .map(|slice| self.expert_from_bytes(slice, device))
            .collect::<Result<Vec<_>>>()
            .map(Some)
    }

    /// Read the stacked tensor once onto the host (streamed load, or a
    /// tensor that cannot be sliced from the mapping) and copy per-expert
    /// slices to `device`.  The host copy is the only staging buffer.
    pub fn read_and_split<R: Read + Seek>(
        &self,
        ct: &gguf_file::Content,
        reader: &mut R,
        arch: &str,
        device: &Device,
    ) -> Result<Vec<QTensor>> {
        let qt = ct.tensor(reader, &self.name, &Device::Cpu)?;
        let bytes = qt.data()?;
        if bytes.len() % self.n_expert != 0 {
            candle_core::bail!(
                "{arch}: expert tensor `{}` byte length {} not divisible by n_expert {}",
                self.name,
                bytes.len(),
                self.n_expert
            );
        }
        let per = bytes.len() / self.n_expert;
        bytes
            .chunks_exact(per)
            .map(|slice| self.expert_from_bytes(slice, device))
            .collect()
    }

    fn expert_from_bytes(&self, bytes: &[u8], device: &Device) -> Result<QTensor> {
        let storage = QStorage::from_data(Cow::Borrowed(bytes), device, self.dtype)?;
        QTensor::new(storage, self.expert_shape())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::QMatMul;
    use candle_core::Module;

    /// A plain quantized MLP-free expert: one matmul, enough to exercise the
    /// dispatch bookkeeping.
    struct Linear(QMatMul);

    impl Expert for Linear {
        fn forward(&self, xs: &Tensor) -> Result<Tensor> {
            self.0.forward(xs)
        }
    }

    fn experts(n: usize, h: usize) -> Result<Vec<Linear>> {
        (0..n)
            .map(|e| {
                let w = Tensor::arange(0f32, (h * h) as f32, &Device::Cpu)?
                    .reshape((h, h))?
                    .affine(1.0 / (h * h) as f64, e as f64 / 10.0)?;
                Ok(Linear(QMatMul::from_qtensor(QTensor::quantize(
                    &w,
                    GgmlDType::F32,
                )?)?))
            })
            .collect()
    }

    /// Prefill over identical rows equals the decode path over one row, and
    /// the routed ids come back token-major.
    #[test]
    fn prefill_matches_decode() -> Result<()> {
        let (n, h, k) = (4usize, 32usize, 2usize);
        let ex = experts(n, h)?;
        let row = Tensor::randn(0f32, 1f32, (1, h), &Device::Cpu)?;
        let two = Tensor::cat(&[&row, &row], 0)?;
        let idx1 = Tensor::new(&[[3u32, 1]], &Device::Cpu)?;
        let idx2 = Tensor::new(&[[3u32, 1], [3, 1]], &Device::Cpu)?;
        let w1 = Tensor::new(&[[0.75f32, 0.25]], &Device::Cpu)?;
        let w2 = Tensor::new(&[[0.75f32, 0.25], [0.75, 0.25]], &Device::Cpu)?;
        let (y2, ids2) = dispatch_prefill("t", &ex, &Device::Cpu, &two, &idx2, &w2, 2, k)?;
        let (y1, ids1) = dispatch_decode("t", &ex, &Device::Cpu, &row, &idx1, &w1, k)?;
        assert_eq!(ids2, vec![3, 1, 3, 1]);
        assert_eq!(ids1, vec![3, 1]);
        let a: Vec<f32> = y2.narrow(0, 1, 1)?.flatten_all()?.to_vec1()?;
        let b: Vec<f32> = y1.flatten_all()?.to_vec1()?;
        for (x, y) in a.iter().zip(&b) {
            assert!(
                (x - y).abs() < 1e-5,
                "prefill vs decode diverge: {x} vs {y}"
            );
        }
        Ok(())
    }

    /// A router id past the expert count is an error naming the token and
    /// slot, on both paths.
    #[test]
    fn out_of_range_expert_is_an_error() -> Result<()> {
        let ex = experts(2, 8)?;
        let x = Tensor::zeros((1, 8), DType::F32, &Device::Cpu)?;
        let idx = Tensor::new(&[[0u32, 7]], &Device::Cpu)?;
        let w = Tensor::new(&[[0.5f32, 0.5]], &Device::Cpu)?;
        let err = dispatch_decode("t", &ex, &Device::Cpu, &x, &idx, &w, 2).unwrap_err();
        assert!(err.to_string().contains("expert 7 out of 2"), "{err}");
        let err = dispatch_prefill("t", &ex, &Device::Cpu, &x, &idx, &w, 1, 2).unwrap_err();
        assert!(err.to_string().contains("(token 0, slot 1)"), "{err}");
        Ok(())
    }

    #[test]
    fn truncate_kv_honours_the_sequence_dim() -> Result<()> {
        let k = Tensor::zeros((1, 2, 6, 4), DType::F32, &Device::Cpu)?;
        let mut kv = Some((k.clone(), k));
        truncate_kv(&mut kv, 9, 2)?;
        assert_eq!(
            kv.as_ref().unwrap().0.dim(2)?,
            6,
            "keep past the end is a no-op"
        );
        truncate_kv(&mut kv, 3, 2)?;
        assert_eq!(kv.as_ref().unwrap().0.dims(), &[1, 2, 3, 4]);
        // Transposed layout: sequence on dim 3.
        let mut kv = Some((
            Tensor::zeros((1, 2, 4, 6), DType::F32, &Device::Cpu)?,
            Tensor::zeros((1, 2, 4, 6), DType::F32, &Device::Cpu)?,
        ));
        truncate_kv(&mut kv, 2, 3)?;
        assert_eq!(kv.as_ref().unwrap().1.dims(), &[1, 2, 4, 2]);
        truncate_kv(&mut kv, 0, 3)?;
        assert!(kv.is_none(), "keep 0 clears the cache");
        Ok(())
    }

    #[test]
    fn causal_mask_offsets_the_cached_prefix() -> Result<()> {
        let m: Vec<f32> = causal_mask(2, 1, &Device::Cpu)?.flatten_all()?.to_vec1()?;
        // Row 0 (position 1) sees positions 0..=1; row 1 sees all three.
        assert_eq!(m, vec![0.0, 0.0, f32::NEG_INFINITY, 0.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn is_routed_expert_matches_moe_names() {
        for dense in [
            "token_embd.weight",
            "output.weight",
            "blk.0.attn_q.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.ffn_gate_inp.weight",
            "blk.0.ffn_gate_shexp.weight",
            "blk.0.indexer_compressor_gate.weight",
        ] {
            assert!(!is_routed_expert(dense), "{dense} should be dense");
        }
        for expert in [
            "blk.0.ffn_gate_exps.weight",
            "blk.12.ffn_up_exps.weight",
            "blk.60.ffn_down_exps.weight",
        ] {
            assert!(is_routed_expert(expert), "{expert} should be an expert");
        }
    }

    /// The three slicing paths agree with each other and with candle's own
    /// reading of the stacked tensor.
    #[test]
    fn expert_tensor_paths_agree() -> Result<()> {
        use candle_core::quantized::gguf_file::{Content, TensorInfo};
        use std::collections::HashMap;
        use std::io::{Cursor, Write};

        let (n_expert, out, inn) = (3usize, 4usize, 64usize);
        let t = Tensor::arange(0f32, (n_expert * out * inn) as f32, &Device::Cpu)?
            .reshape((n_expert, out, inn))?
            .affine(0.01, -3.0)?;
        let q = QTensor::quantize(&t, GgmlDType::Q8_0)?;
        let data = q.data()?.into_owned();
        // A minimal "file": a 64-byte header pad then the tensor bytes.
        let mut file = vec![0u8; 64];
        file.write_all(&data).unwrap();
        let mmap = {
            let path = std::env::temp_dir().join(format!("joshua-moe-{}", std::process::id()));
            std::fs::write(&path, &file).unwrap();
            let f = std::fs::File::open(&path).unwrap();
            let m = unsafe { Mmap::map(&f) }.unwrap();
            std::fs::remove_file(&path).ok();
            Arc::new(m)
        };
        let mut tensor_infos = HashMap::new();
        tensor_infos.insert(
            "blk.0.ffn_up_exps.weight".to_string(),
            TensorInfo {
                ggml_dtype: GgmlDType::Q8_0,
                shape: (n_expert, out, inn).into(),
                offset: 0,
            },
        );
        let ct = Content {
            magic: gguf_file::VersionedMagic::GgufV3,
            metadata: HashMap::new(),
            tensor_infos,
            tensor_data_offset: 64,
            external: None,
        };
        let et = ExpertTensor::lookup(&ct, "t", "blk.0.ffn_up_exps.weight", n_expert)?;
        assert_eq!(et.expert_shape(), (out, inn));
        assert_eq!(et.bytes_per_expert(), Some(data.len() / n_expert));
        assert!(ExpertTensor::lookup(&ct, "t", "blk.0.ffn_up_exps.weight", 2).is_err());
        assert!(ExpertTensor::lookup(&ct, "t", "missing", n_expert).is_err());

        let expected: Vec<Vec<f32>> = (0..n_expert)
            .map(|e| t.get(e)?.flatten_all()?.to_vec1())
            .collect::<Result<_>>()?;
        let check = |tensors: Vec<QTensor>, what: &str| -> Result<()> {
            assert_eq!(tensors.len(), n_expert, "{what}");
            for (e, qt) in tensors.iter().enumerate() {
                let got: Vec<f32> = qt.dequantize(&Device::Cpu)?.flatten_all()?.to_vec1()?;
                let want: Vec<f32> = q
                    .dequantize(&Device::Cpu)?
                    .get(e)?
                    .flatten_all()?
                    .to_vec1()?;
                assert_eq!(got, want, "{what}: expert {e}");
                assert_eq!(got.len(), expected[e].len());
            }
            Ok(())
        };
        let borrowed = et.borrow_host(&mmap)?.expect("Q8_0 borrows");
        assert!(borrowed.iter().all(|b| b.prefetch.is_some()));
        check(
            borrowed.into_iter().map(|b| b.tensor).collect(),
            "borrow_host",
        )?;
        check(
            et.upload_from_mmap(&mmap, &Device::Cpu)?.expect("slices"),
            "upload_from_mmap",
        )?;
        let mut cursor = Cursor::new(&file[..]);
        check(
            et.read_and_split(&ct, &mut cursor, "t", &Device::Cpu)?,
            "read_and_split",
        )?;
        Ok(())
    }
}
