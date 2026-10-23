//! SYCL decode megakernel: one persistent kernel launch per generated token
//! for a dense pre-norm decoder (see `mega.hpp` for the scheduling and the
//! numerics).  A loader describes its weights once as a [`ModelSpec`] plus
//! one [`LayerSpec`] per layer; [`DecodePlan::decode`] then runs a whole
//! single-token step (every layer, the final norm and the head) and returns
//! the logits.
//!
//! The plan borrows nothing at run time: it keeps clones of every weight it
//! addresses, so the device buffers stay alive as long as the plan does.
use super::bridge::{self, MegaKvDesc, MegaLayerDesc, MegaModelDesc, MegaVDesc, MegaWDesc};
use super::kernels;
use super::SyclDevice;
use crate::op::BackpropOp;
use crate::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use crate::{DType, Device, Error, Result, Storage, Tensor};
use std::sync::Arc;

/// Rows `row0 .. row0 + rows` of a `[n, k]` projection weight (a view into
/// a fused matrix such as `attn_qkv` or a `[gate ‖ up]` FFN).
#[derive(Clone)]
pub struct RowRange {
    pub weight: QMatMul,
    pub row0: usize,
    pub rows: usize,
}

impl RowRange {
    pub fn all(weight: &QMatMul) -> Result<Self> {
        let rows = match weight {
            QMatMul::QTensor(q) => q.shape().dims2()?.0,
            QMatMul::Tensor(t) | QMatMul::TensorF16(t) => t.dims2()?.0,
        };
        Ok(Self { weight: weight.clone(), row0: 0, rows })
    }
    pub fn slice(weight: &QMatMul, row0: usize, rows: usize) -> Self {
        Self { weight: weight.clone(), row0, rows }
    }
}

/// One decoder layer: `x += o(attn(norm(x)))`, `x += down(silu(gate) * up)`
/// over `norm(x)`.  Vectors are f32 on the device (views are fine).
#[derive(Clone)]
pub struct LayerSpec {
    pub attn_norm: Tensor,
    pub ffn_norm: Tensor,
    pub q: RowRange,
    pub k: RowRange,
    pub v: RowRange,
    pub q_bias: Option<Tensor>,
    pub k_bias: Option<Tensor>,
    pub v_bias: Option<Tensor>,
    pub q_norm: Option<Tensor>,
    pub k_norm: Option<Tensor>,
    pub o: RowRange,
    pub o_bias: Option<Tensor>,
    pub gate: RowRange,
    pub up: RowRange,
    pub down: RowRange,
}

#[derive(Clone)]
pub struct ModelSpec {
    pub hidden: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    /// Rotated leading dims of each head (`<= head_dim`).
    pub n_rot: usize,
    /// NEOX pairing (`i`, `i + n_rot/2`); otherwise adjacent pairs.
    pub neox: bool,
    pub ffn: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    pub attn_scale: f32,
    /// RoPE tables `[max_pos, n_rot / 2]`.
    pub cos: Tensor,
    pub sin: Tensor,
    pub out_norm: Tensor,
    pub out: RowRange,
    pub out_bias: Option<Tensor>,
}

/// A built megakernel plan for one model on one SYCL device.
pub struct DecodePlan {
    device: SyclDevice,
    plan: usize,
    n_layers: usize,
    n_kv_head: usize,
    head_dim: usize,
    hidden: usize,
    vocab: usize,
    max_pos: usize,
    /// Keeps every addressed buffer alive.
    _keep: (ModelSpec, Vec<LayerSpec>),
}

impl Drop for DecodePlan {
    fn drop(&mut self) {
        unsafe { bridge::mega_free(self.plan) };
    }
}

fn sycl_buf(t: &Tensor, what: &str) -> Result<MegaVDesc> {
    if t.dtype() != DType::F32 {
        crate::bail!("sycl megakernel: {what} must be f32, got {:?}", t.dtype());
    }
    let (storage, layout) = t.storage_and_layout();
    if !layout.is_contiguous() {
        crate::bail!("sycl megakernel: {what} is not contiguous");
    }
    match &*storage {
        Storage::Sycl(s) => Ok(MegaVDesc { buf: s.buffer, off: (layout.start_offset() * 4) as u64 }),
        _ => crate::bail!("sycl megakernel: {what} is not on a SYCL device"),
    }
}

fn opt_buf(t: &Option<Tensor>, what: &str) -> Result<MegaVDesc> {
    match t {
        Some(t) => sycl_buf(t, what),
        None => Ok(MegaVDesc::default()),
    }
}

/// Whether a weight's layout is one the megakernel addresses: the formats
/// the device GEMV kernels take, and not the f16-activation `TensorF16` path
/// (whose numerics differ).
pub fn weight_supported(w: &QMatMul) -> bool {
    match w {
        QMatMul::QTensor(q) => {
            matches!(q.storage(), QStorage::Sycl(_))
                && !matches!(q.dtype(), GgmlDType::Q8_1 | GgmlDType::Q8K)
        }
        QMatMul::Tensor(t) => t.dtype() == DType::F32 && t.is_contiguous() && t.device().is_sycl(),
        QMatMul::TensorF16(_) => false,
    }
}

fn weight(r: &RowRange, k_expect: usize, what: &str) -> Result<MegaWDesc> {
    match &r.weight {
        QMatMul::QTensor(q) => {
            let (n, k) = q.shape().dims2()?;
            weight_desc(q, n, k, r, k_expect, what)
        }
        QMatMul::Tensor(t) => {
            let (n, k) = t.dims2()?;
            if r.row0 + r.rows > n || k != k_expect {
                crate::bail!("sycl megakernel: {what} rows {}..{} of [{n}, {k}] (expected k = {k_expect})", r.row0, r.row0 + r.rows);
            }
            let v = sycl_buf(t, what)?;
            Ok(MegaWDesc {
                buf: v.buf,
                off: v.off + (r.row0 * k * 4) as u64,
                qt: kernels::qtype_code(GgmlDType::F32),
                qk: 1,
                bsz: 4,
                n: r.rows as i32,
                k: k as i32,
                cols: 0,
            })
        }
        QMatMul::TensorF16(_) => crate::bail!("sycl megakernel: {what} is an f16-activation weight"),
    }
}

fn weight_desc(q: &Arc<QTensor>, n: usize, k: usize, r: &RowRange, k_expect: usize, what: &str) -> Result<MegaWDesc> {
    if r.row0 + r.rows > n || k != k_expect {
        crate::bail!("sycl megakernel: {what} rows {}..{} of [{n}, {k}] (expected k = {k_expect})", r.row0, r.row0 + r.rows);
    }
    let QStorage::Sycl(s) = q.storage() else {
        crate::bail!("sycl megakernel: {what} is not on a SYCL device");
    };
    let dtype = q.dtype();
    let (qk, bsz) = (dtype.block_size(), dtype.type_size());
    if !k.is_multiple_of(qk) {
        crate::bail!("sycl megakernel: {what} inner dim {k} is not a whole number of {dtype:?} blocks");
    }
    if kernels::is_block_dtype(dtype) && !k.is_multiple_of(32) {
        crate::bail!("sycl megakernel: {what} inner dim {k} is not a multiple of 32");
    }
    // The layout the op-at-a-time path would pick for one row (`run_qgemv`):
    // the multi-row kernel's lanes for the formats it takes.
    let cols = if kernels::qgemv_multirow(dtype) { kernels::qgemv_mr_cols(k, 64) } else { 0 };
    let row_bytes = (k / qk) * bsz;
    Ok(MegaWDesc {
        buf: s.buffer,
        off: s.byte_offset + (r.row0 * row_bytes) as u64,
        qt: kernels::qtype_code(dtype),
        qk: qk as i32,
        bsz: bsz as i32,
        n: r.rows as i32,
        k: k as i32,
        cols: cols as i32,
    })
}

fn i32_of(v: usize, what: &str) -> Result<i32> {
    i32::try_from(v).map_err(|_| Error::Msg(format!("sycl megakernel: {what} {v} exceeds i32")))
}

impl DecodePlan {
    /// Whether the loaded SYCL bridge has the megakernel.
    pub fn available() -> bool {
        bridge::mega_available()
    }

    /// Build a plan.  `groups` overrides the work-groups launched (0: one per
    /// compute unit).
    pub fn new(device: &Device, spec: ModelSpec, layers: Vec<LayerSpec>, groups: usize) -> Result<Self> {
        let Device::Sycl(dev) = device else {
            crate::bail!("sycl megakernel: not a SYCL device");
        };
        let (h, hd) = (spec.hidden, spec.head_dim);
        let q_dim = spec.n_head * hd;
        let kv_dim = spec.n_kv_head * hd;
        let max_pos = spec.cos.dim(0)?;
        if spec.cos.dims2()? != (max_pos, spec.n_rot / 2) || spec.sin.dims2()? != (max_pos, spec.n_rot / 2) {
            crate::bail!("sycl megakernel: RoPE tables must be [max_pos, n_rot / 2]");
        }
        let mut descs = Vec::with_capacity(layers.len());
        for (l, ls) in layers.iter().enumerate() {
            let what = |s: &str| format!("layer {l} {s}");
            for (r, n) in [(&ls.q, q_dim), (&ls.k, kv_dim), (&ls.v, kv_dim), (&ls.o, h), (&ls.gate, spec.ffn), (&ls.up, spec.ffn), (&ls.down, h)] {
                if r.rows != n {
                    crate::bail!("sycl megakernel: {} has {} rows, expected {n}", what("projection"), r.rows);
                }
            }
            descs.push(MegaLayerDesc {
                attn_norm: sycl_buf(&ls.attn_norm, &what("attn norm"))?,
                ffn_norm: sycl_buf(&ls.ffn_norm, &what("ffn norm"))?,
                qkv: [weight(&ls.q, h, &what("q"))?, weight(&ls.k, h, &what("k"))?, weight(&ls.v, h, &what("v"))?],
                qkv_b: [opt_buf(&ls.q_bias, &what("q bias"))?, opt_buf(&ls.k_bias, &what("k bias"))?, opt_buf(&ls.v_bias, &what("v bias"))?],
                q_norm: opt_buf(&ls.q_norm, &what("q norm"))?,
                k_norm: opt_buf(&ls.k_norm, &what("k norm"))?,
                o: weight(&ls.o, q_dim, &what("o"))?,
                o_b: opt_buf(&ls.o_bias, &what("o bias"))?,
                gate: weight(&ls.gate, h, &what("gate"))?,
                up: weight(&ls.up, h, &what("up"))?,
                down: weight(&ls.down, spec.ffn, &what("down"))?,
            });
        }
        let model = MegaModelDesc {
            n_layers: i32_of(layers.len(), "layers")?,
            h: i32_of(h, "hidden")?,
            n_head: i32_of(spec.n_head, "heads")?,
            n_kv: i32_of(spec.n_kv_head, "kv heads")?,
            hd: i32_of(hd, "head dim")?,
            n_rot: i32_of(spec.n_rot, "n_rot")?,
            neox: spec.neox as i32,
            ffn: i32_of(spec.ffn, "ffn")?,
            vocab: i32_of(spec.vocab, "vocab")?,
            max_pos: i32_of(max_pos, "max pos")?,
            groups: i32_of(groups, "groups")?,
            pad: 0,
            eps: spec.rms_eps,
            scale: spec.attn_scale,
            cos: sycl_buf(&spec.cos, "rope cos")?,
            sin: sycl_buf(&spec.sin, "rope sin")?,
            out_norm: sycl_buf(&spec.out_norm, "output norm")?,
            out_b: opt_buf(&spec.out_bias, "output bias")?,
            out: weight(&spec.out, h, "output")?,
        };
        let plan = bridge::mega_new(dev.context(), &model, &descs)?;
        Ok(Self {
            device: dev.clone(),
            plan,
            n_layers: layers.len(),
            n_kv_head: spec.n_kv_head,
            head_dim: hd,
            hidden: h,
            vocab: spec.vocab,
            max_pos,
            _keep: (spec, layers),
        })
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Positions the RoPE table covers.
    pub fn max_pos(&self) -> usize {
        self.max_pos
    }

    /// Work-groups each launch uses.
    pub fn groups(&self) -> usize {
        bridge::mega_groups(self.plan)
    }

    /// One decode step at position `pos`: `x` is the token's embedding
    /// (`hidden` f32), `kv[l]` layer `l`'s key and value buffers
    /// `[1, n_kv_head, cap, head_dim]` with `cap > pos`; the kernel writes
    /// row `pos` of each.  Returns the logits `[1, vocab]`.
    pub fn decode(&self, kv: &[(Tensor, Tensor)], pos: usize, x: &Tensor) -> Result<Tensor> {
        if kv.len() != self.n_layers {
            crate::bail!("sycl megakernel: {} KV caches for {} layers", kv.len(), self.n_layers);
        }
        if x.elem_count() != self.hidden {
            crate::bail!("sycl megakernel: input has {} elements, expected {}", x.elem_count(), self.hidden);
        }
        let mut descs = Vec::with_capacity(kv.len());
        for (l, (k, v)) in kv.iter().enumerate() {
            let (b, n_kv, cap, d) = k.dims4()?;
            if (b, n_kv, d) != (1, self.n_kv_head, self.head_dim) || v.dims4()? != (b, n_kv, cap, d) {
                crate::bail!("sycl megakernel: layer {l} KV cache {:?} / {:?}", k.dims(), v.dims());
            }
            descs.push(MegaKvDesc {
                k: sycl_buf(k, "key cache")?,
                v: sycl_buf(v, "value cache")?,
                cap: i32_of(cap, "kv capacity")?,
                pad: 0,
            });
        }
        let x_in = sycl_buf(x, "input")?;
        let out = self.device.alloc(DType::F32, self.vocab)?;
        let logits = MegaVDesc { buf: out.buffer, off: 0 };
        bridge::mega_decode(self.device.context(), self.plan, &descs, i32_of(pos, "position")?, x_in, logits)?;
        kernels::note_native_exec();
        Ok(Tensor::from_storage(Storage::Sycl(out), (1, self.vocab), BackpropOp::none(), false))
    }
}
