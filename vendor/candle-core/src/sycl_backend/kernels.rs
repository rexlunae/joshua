//! SYCL kernel launchers — a 1:1 mirror of the OpenCL backend's
//! `opencl_backend/kernels.rs`: the same kernel names, the same argument
//! orders (the SYCL `dispatch.inc` was generated against these orders), the
//! same grid conventions.  `k.run(&[g], local)` becomes
//! `dev.launch(name, args, global, local)`; the bridge reverses the
//! dimensions for SYCL exactly as the OpenCL x/y/z layout expects.

use super::{SyclDevice, WG};
use crate::{Error, Layout, Result};

pub const MAXD: usize = 8;

/// Stride descriptor for the strided kernels — same field order as
/// `kernels.hpp`'s `Idx` (`nd; dims[8]; s0[8]; s1[8]; s2[8]; o0; o1; o2`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Idx {
    pub nd: i32,
    pub dims: [i32; MAXD],
    pub s0: [i32; MAXD],
    pub s1: [i32; MAXD],
    pub s2: [i32; MAXD],
    pub o0: i32,
    pub o1: i32,
    pub o2: i32,
}

impl Idx {
    pub fn new(dims: &[usize]) -> Result<Self> {
        if dims.len() > MAXD {
            return Err(Error::Msg(format!(
                "sycl: rank {} exceeds the kernel limit of {MAXD}",
                dims.len()
            )));
        }
        let mut ix = Idx {
            nd: dims.len() as i32,
            dims: [1; MAXD],
            s0: [0; MAXD],
            s1: [0; MAXD],
            s2: [0; MAXD],
            o0: 0,
            o1: 0,
            o2: 0,
        };
        for (i, &d) in dims.iter().enumerate() {
            ix.dims[i] = to_i32(d)?;
        }
        Ok(ix)
    }

    /// Fill stride set `which` (0..3) from a layout whose shape is this
    /// descriptor's output shape.
    pub fn with_layout(mut self, which: usize, l: &Layout) -> Result<Self> {
        if l.dims().len() != self.nd as usize {
            return Err(Error::Msg("sycl: layout rank mismatch".into()));
        }
        let (s, o) = match which {
            0 => (&mut self.s0, &mut self.o0),
            1 => (&mut self.s1, &mut self.o1),
            _ => (&mut self.s2, &mut self.o2),
        };
        for (i, &st) in l.stride().iter().enumerate() {
            s[i] = to_i32(st)?;
        }
        *o = to_i32(l.start_offset())?;
        Ok(self)
    }

    /// Stride set `which` from explicit strides and offset.
    pub fn with_strides(mut self, which: usize, strides: &[usize], offset: usize) -> Result<Self> {
        let (s, o) = match which {
            0 => (&mut self.s0, &mut self.o0),
            1 => (&mut self.s1, &mut self.o1),
            _ => (&mut self.s2, &mut self.o2),
        };
        for (i, &st) in strides.iter().enumerate() {
            s[i] = to_i32(st)?;
        }
        *o = to_i32(offset)?;
        Ok(self)
    }
}

pub fn to_i32(v: usize) -> Result<i32> {
    i32::try_from(v).map_err(|_| Error::Msg(format!("sycl: dimension {v} exceeds the 32-bit kernel index range")))
}

/// Strides of one operand of a (batched) matmul — identical to the OpenCL
/// backend's `MatStrides`.
#[derive(Clone, Copy, Debug)]
pub struct MatStrides {
    pub row: usize,
    pub col: usize,
    pub offset: usize,
    pub batch: usize,
}

pub fn binary_code(name: &str) -> Option<i32> {
    Some(match name {
        "add" => 0,
        "sub" => 1,
        "mul" => 2,
        "div" => 3,
        "minimum" => 4,
        "maximum" => 5,
        _ => return None,
    })
}

pub fn cmp_code(op: crate::op::CmpOp) -> i32 {
    use crate::op::CmpOp;
    match op {
        CmpOp::Eq => 0,
        CmpOp::Ne => 1,
        CmpOp::Le => 2,
        CmpOp::Ge => 3,
        CmpOp::Lt => 4,
        CmpOp::Gt => 5,
    }
}

pub fn unary_code(name: &str) -> Option<i32> {
    Some(match name {
        "exp" => 0,
        "log" => 1,
        "sin" => 2,
        "cos" => 3,
        "tanh" => 4,
        "neg" => 5,
        "recip" => 6,
        "sqr" => 7,
        "sqrt" => 8,
        "gelu" => 9,
        "gelu_erf" => 10,
        "erf" => 11,
        "silu" => 12,
        "relu" => 13,
        "sigmoid" => 14,
        "abs" => 15,
        "ceil" => 16,
        "floor" => 17,
        "round" => 18,
        _ => return None,
    })
}

pub const RED_SUM: i32 = 0;
pub const RED_MAX: i32 = 1;
pub const RED_MIN: i32 = 2;

// ─── Launchers ───────────────────────────────────────────────────────────────

pub fn run_unary(dev: &SyclDevice, op: i32, x: usize, out: usize, n: usize, l: &Layout) -> Result<()> {
    if let Some((o1, _)) = l.contiguous_offsets() {
        let mut b = super::ArgBuilder::new();
        b.buf(x).buf(out).i32(n as i32).i32(o1 as i32).i32(op);
        dev.launch("k_unary_c", &mut b, [n, 1, 1], [WG, 1, 1])
    } else {
        let ix = Idx::new(l.dims())?.with_layout(0, l)?;
        let mut b = super::ArgBuilder::new();
        b.buf(x).buf(out).i32(n as i32).idx(ix).i32(op);
        dev.launch("k_unary_s", &mut b, [n, 1, 1], [WG, 1, 1])
    }
}

pub fn run_affine(dev: &SyclDevice, x: usize, out: usize, n: usize, l: &Layout, mul: f32, add: f32) -> Result<()> {
    if let Some((o1, _)) = l.contiguous_offsets() {
        let mut b = super::ArgBuilder::new();
        b.buf(x).buf(out).i32(n as i32).i32(o1 as i32).f32(mul).f32(add);
        dev.launch("k_affine_c", &mut b, [n, 1, 1], [WG, 1, 1])
    } else {
        let ix = Idx::new(l.dims())?.with_layout(0, l)?;
        let mut b = super::ArgBuilder::new();
        b.buf(x).buf(out).i32(n as i32).idx(ix).f32(mul).f32(add);
        dev.launch("k_affine_s", &mut b, [n, 1, 1], [WG, 1, 1])
    }
}

pub fn run_powf(dev: &SyclDevice, x: usize, out: usize, n: usize, l: &Layout, e: f32) -> Result<()> {
    let ix = Idx::new(l.dims())?.with_layout(0, l)?;
    let mut b = super::ArgBuilder::new();
    b.buf(x).buf(out).i32(n as i32).idx(ix).f32(e);
    dev.launch("k_powf_s", &mut b, [n, 1, 1], [WG, 1, 1])
}

pub fn run_elu(dev: &SyclDevice, x: usize, out: usize, n: usize, l: &Layout, alpha: f32) -> Result<()> {
    let ix = Idx::new(l.dims())?.with_layout(0, l)?;
    let mut b = super::ArgBuilder::new();
    b.buf(x).buf(out).i32(n as i32).idx(ix).f32(alpha);
    dev.launch("k_elu_s", &mut b, [n, 1, 1], [WG, 1, 1])
}

/// f32 binary op; `u32` selects the integer kernel.
pub fn run_binary(dev: &SyclDevice, op: i32, a: usize, b: usize, out: usize, n: usize, la: &Layout, lb: &Layout, u32: bool) -> Result<()> {
    if !u32 {
        if let (Some((oa, _)), Some((ob, _))) = (la.contiguous_offsets(), lb.contiguous_offsets()) {
            let mut ab = super::ArgBuilder::new();
            ab.buf(a).buf(b).buf(out).i32(n as i32).i32(oa as i32).i32(ob as i32).i32(op);
            return dev.launch("k_binary_c", &mut ab, [n, 1, 1], [WG, 1, 1]);
        }
    }
    let ix = Idx::new(la.dims())?.with_layout(0, la)?.with_layout(1, lb)?;
    let mut ab = super::ArgBuilder::new();
    ab.buf(a).buf(b).buf(out).i32(n as i32).idx(ix).i32(op);
    dev.launch(if u32 { "k_binary_u32_s" } else { "k_binary_s" }, &mut ab, [n, 1, 1], [WG, 1, 1])
}

pub fn run_cmp(dev: &SyclDevice, op: i32, a: usize, b: usize, out: usize, n: usize, la: &Layout, lb: &Layout, u32: bool) -> Result<()> {
    let ix = Idx::new(la.dims())?.with_layout(0, la)?.with_layout(1, lb)?;
    let mut b = super::ArgBuilder::new();
    b.buf(a).buf(b).buf(out).i32(n as i32).idx(ix).i32(op);
    dev.launch(if u32 { "k_cmp_u32" } else { "k_cmp_f32" }, &mut b, [n, 1, 1], [WG, 1, 1])
}

/// `where_cond` with a u8 or u32 condition over 4- or 8-byte payloads.
pub fn run_where(dev: &SyclDevice, cond: usize, t: usize, f: usize, out: usize, n: usize, lc: &Layout, lt: &Layout, lf: &Layout, cond_u32: bool, elem8: bool) -> Result<()> {
    let ix = Idx::new(lc.dims())?.with_layout(0, lc)?.with_layout(1, lt)?.with_layout(2, lf)?;
    let name = match (cond_u32, elem8) {
        (false, false) => "k_where_4",
        (false, true) => "k_where_8",
        (true, false) => "k_where_u32c_4",
        (true, true) => return Err(Error::Msg("sycl: u32 condition with 8-byte payload has no kernel".into())),
    };
    let mut b = super::ArgBuilder::new();
    b.buf(cond).buf(t).buf(f).buf(out).i32(n as i32).idx(ix);
    dev.launch(name, &mut b, [n, 1, 1], [WG, 1, 1])
}

/// Strided copy of `n` elements (`elem` bytes each) described by `l` into
/// `dst` at element offset `dst_off`, contiguous.
pub fn run_copy_strided(dev: &SyclDevice, elem: usize, src: usize, dst: usize, n: usize, l: &Layout, dst_off: usize) -> Result<()> {
    let name = match elem {
        1 => "k_copy_s1",
        2 => "k_copy_s2",
        4 => "k_copy_s4",
        8 => "k_copy_s8",
        _ => return Err(Error::Msg(format!("sycl: no copy kernel for {elem}-byte elements"))),
    };
    let ix = Idx::new(l.dims())?.with_layout(0, l)?;
    let mut b = super::ArgBuilder::new();
    b.buf(src).buf(dst).i32(n as i32).idx(ix).i32(dst_off as i32);
    dev.launch(name, &mut b, [n, 1, 1], [WG, 1, 1])
}

#[allow(clippy::too_many_arguments)]
pub fn run_copy2d(dev: &SyclDevice, elem: usize, src: usize, dst: usize, d1: usize, d2: usize, src_s: usize, dst_s: usize, src_o: usize, dst_o: usize) -> Result<()> {
    let name = match elem {
        1 => "k_copy2d_1",
        2 => "k_copy2d_2",
        4 => "k_copy2d_4",
        8 => "k_copy2d_8",
        _ => return Err(Error::Msg(format!("sycl: no copy2d kernel for {elem}-byte elements"))),
    };
    let mut b = super::ArgBuilder::new();
    b.buf(src).buf(dst).i32(d1 as i32).i32(d2 as i32).i32(src_s as i32).i32(dst_s as i32).i32(src_o as i32).i32(dst_o as i32);
    // OpenCL runs &[d2, d1]: x = d2 (row length), y = d1 (rows).
    dev.launch(name, &mut b, [d2, d1, 1], [WG, 1, 1])
}

/// Fill the elements addressed by `l` with the bit pattern `bits`.
pub fn run_fill(dev: &SyclDevice, elem: usize, dst: usize, n: usize, l: &Layout, bits: u64) -> Result<()> {
    let ix = Idx::new(l.dims())?.with_layout(0, l)?;
    let (name, size) = match elem {
        1 => ("k_fill_1", 1usize),
        2 => ("k_fill_2", 2),
        4 => ("k_fill_4", 4),
        8 => ("k_fill_8", 8),
        _ => return Err(Error::Msg(format!("sycl: no fill kernel for {elem}-byte elements"))),
    };
    let bytes: [u8; 8] = bits.to_ne_bytes();
    let mut b = super::ArgBuilder::new();
    b.buf(dst).i32(n as i32).idx(ix).push_raw(&bytes[..size]);
    dev.launch(name, &mut b, [n, 1, 1], [WG, 1, 1])
}

/// Cast kernel name for a (from, to) dtype pair, when one exists.
pub fn cast_kernel(from: crate::DType, to: crate::DType) -> Option<&'static str> {
    use crate::DType::*;
    Some(match (from, to) {
        (F32, U32) => "k_cast_f32_u32",
        (U32, F32) => "k_cast_u32_f32",
        (F32, U8) => "k_cast_f32_u8",
        (U8, F32) => "k_cast_u8_f32",
        (F32, I64) => "k_cast_f32_i64",
        (I64, F32) => "k_cast_i64_f32",
        (U32, I64) => "k_cast_u32_i64",
        (I64, U32) => "k_cast_i64_u32",
        (U8, U32) => "k_cast_u8_u32",
        (U32, U8) => "k_cast_u32_u8",
        (F32, F16) => "k_cast_f32_f16",
        (F16, F32) => "k_cast_f16_f32",
        (I32, F32) => "k_cast_i32_f32",
        (F32, I32) => "k_cast_f32_i32",
        (U32, I32) => "k_cast_u32_i32",
        (I32, U32) => "k_cast_i32_u32",
        _ => return None,
    })
}

pub fn run_cast(dev: &SyclDevice, name: &str, x: usize, out: usize, n: usize, l: &Layout) -> Result<()> {
    let ix = Idx::new(l.dims())?.with_layout(0, l)?;
    let mut b = super::ArgBuilder::new();
    b.buf(x).buf(out).i32(n as i32).idx(ix);
    dev.launch(name, &mut b, [n, 1, 1], [WG, 1, 1])
}

// ─── Reductions ──────────────────────────────────────────────────────────────

pub fn run_reduce_last(dev: &SyclDevice, op: i32, x: usize, out: usize, rows: usize, cols: usize, off: usize) -> Result<()> {
    let mut b = super::ArgBuilder::new();
    b.buf(x).buf(out).i32(rows as i32).i32(cols as i32).i32(off as i32).i32(op);
    dev.launch("k_reduce_last", &mut b, [rows * WG, 1, 1], [WG, 1, 1])
}

pub fn run_arg_last(dev: &SyclDevice, is_max: bool, x: usize, out: usize, rows: usize, cols: usize, off: usize) -> Result<()> {
    let mut b = super::ArgBuilder::new();
    b.buf(x).buf(out).i32(rows as i32).i32(cols as i32).i32(off as i32).i32(is_max as i32);
    dev.launch("k_arg_last", &mut b, [rows * WG, 1, 1], [WG, 1, 1])
}

pub fn run_reduce_generic(dev: &SyclDevice, op: i32, x: usize, out: usize, n: usize, ix: Idx, rd: Idx, count: usize) -> Result<()> {
    let mut b = super::ArgBuilder::new();
    b.buf(x).buf(out).i32(n as i32).idx(ix).idx(rd).i32(count as i32).i32(op);
    dev.launch("k_reduce_generic", &mut b, [n, 1, 1], [WG, 1, 1])
}

// ─── Indexing ────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub fn run_index_select(dev: &SyclDevice, elem8: bool, ids_i64: bool, src: usize, ids: usize, out: usize, n: usize, left: usize, n_ids: usize, right: usize, dim_size: usize, src_off: usize, ids_off: usize, fault: usize, fslot: i32) -> Result<()> {
    let name = match (elem8, ids_i64) {
        (false, false) => "k_index_select_u32_4",
        (false, true) => "k_index_select_i64_4",
        (true, false) => "k_index_select_u32_8",
        (true, true) => return Err(Error::Msg("sycl: i64 ids with 8-byte elements has no kernel".into())),
    };
    let mut b = super::ArgBuilder::new();
    b.buf(src).buf(ids).buf(out)
        .i32(n as i32).i32(left as i32).i32(n_ids as i32).i32(right as i32).i32(dim_size as i32)
        .i32(src_off as i32).i32(ids_off as i32).buf(fault).i32(fslot);
    dev.launch(name, &mut b, [n, 1, 1], [WG, 1, 1])
}

#[allow(clippy::too_many_arguments)]
pub fn run_gather(dev: &SyclDevice, src: usize, ids: usize, out: usize, n: usize, ix: Idx, src_dim_stride: usize, dim_size: usize, fault: usize, fslot: i32) -> Result<()> {
    let mut b = super::ArgBuilder::new();
    b.buf(src).buf(ids).buf(out).i32(n as i32).idx(ix).i32(src_dim_stride as i32).i32(dim_size as i32).buf(fault).i32(fslot);
    dev.launch("k_gather_4", &mut b, [n, 1, 1], [WG, 1, 1])
}

#[allow(clippy::too_many_arguments)]
pub fn run_scatter(dev: &SyclDevice, add: bool, dst: usize, ids: usize, src: usize, n: usize, ix: Idx, n_j: usize, ids_ds: usize, src_ds: usize, dst_ds: usize, dim_size: usize, fault: usize, fslot: i32) -> Result<()> {
    let mut b = super::ArgBuilder::new();
    b.buf(dst).buf(ids).buf(src).i32(n as i32).idx(ix).i32(n_j as i32).i32(ids_ds as i32).i32(src_ds as i32).i32(dst_ds as i32).i32(dim_size as i32).buf(fault).i32(fslot);
    dev.launch(if add { "k_scatter_add_f32" } else { "k_scatter_set_4" }, &mut b, [n, 1, 1], [WG, 1, 1])
}

#[allow(clippy::too_many_arguments)]
pub fn run_index_add(dev: &SyclDevice, dst: usize, ids: usize, src: usize, left: usize, n_ids: usize, right: usize, dim_size: usize, src_off: usize, ids_off: usize, fault: usize, fslot: i32) -> Result<()> {
    let n_lr = left * right;
    let mut b = super::ArgBuilder::new();
    b.buf(dst).buf(ids).buf(src)
        .i32(n_lr as i32).i32(left as i32).i32(n_ids as i32).i32(right as i32).i32(dim_size as i32)
        .i32(src_off as i32).i32(ids_off as i32).buf(fault).i32(fslot);
    dev.launch("k_index_add_f32", &mut b, [n_lr, 1, 1], [WG, 1, 1])
}

// ─── Dense GEMM / GEMV ───────────────────────────────────────────────────────

/// `C[bz] = A[bz] @ B[bz]` — mirrors the OpenCL launcher exactly (same
/// kernel choice, same grid formula).
#[allow(clippy::too_many_arguments)]
pub fn run_matmul(dev: &SyclDevice, a: usize, b: usize, out: usize, (batch, m, n, k): (usize, usize, usize, usize), sa: MatStrides, sb: MatStrides) -> Result<()> {
    if batch == 0 || m == 0 || n == 0 {
        return Ok(());
    }
    if k == 0 {
        let n_out = batch * m * n;
        return run_fill(dev, 4, out, n_out, &Layout::contiguous(n_out), 0);
    }
    let (m32, n32, k32) = (to_i32(m)?, to_i32(n)?, to_i32(k)?);
    if m == 1 && sa.col == 1 {
        let (oa, ob) = (to_i32(sa.offset)?, to_i32(sb.offset)?);
        let (ba, bb, bc) = (to_i32(sa.batch)?, to_i32(sb.batch)?, to_i32(n)?);
        if sb.row == 1 {
            let mut kn = super::ArgBuilder::new();
            kn.buf(a).buf(b).buf(out).i32(n32).i32(k32).i32(to_i32(sb.col)?)
                .i32(oa).i32(ob).i32(0).i32(ba).i32(bb).i32(bc);
            return dev.launch("k_gemv_nt", &mut kn, [n * WG, batch, 1], [WG, 1, 1]);
        }
        if sb.col == 1 {
            let mut kn = super::ArgBuilder::new();
            kn.buf(a).buf(b).buf(out).i32(n32).i32(k32).i32(to_i32(sb.row)?)
                .i32(oa).i32(ob).i32(0).i32(ba).i32(bb).i32(bc);
            return dev.launch("k_gemv_nn", &mut kn, [n, batch, 1], [WG, 1, 1]);
        }
    }
    let mut kn = super::ArgBuilder::new();
    kn.buf(a).buf(b).buf(out)
        .i32(m32).i32(n32).i32(k32)
        .i32(to_i32(sa.row)?).i32(to_i32(sa.col)?).i32(to_i32(sb.row)?).i32(to_i32(sb.col)?)
        .i32(to_i32(sa.offset)?).i32(to_i32(sb.offset)?).i32(0)
        .i32(to_i32(sa.batch)?).i32(to_i32(sb.batch)?).i32(to_i32(m * n)?)
        .i32((sb.row == 1) as i32);
    let gx = n.div_ceil(64) * 16;
    let gy = m.div_ceil(64) * 16;
    dev.launch("k_gemm", &mut kn, [gx, gy, batch], [16, 16, 1])
}

// ─── Fused attention-path ops ────────────────────────────────────────────────

pub fn run_softmax_last(dev: &SyclDevice, x: usize, out: usize, rows: usize, cols: usize, off: usize) -> Result<()> {
    let mut b = super::ArgBuilder::new();
    b.buf(x).buf(out).i32(rows as i32).i32(cols as i32).i32(off as i32);
    dev.launch("k_softmax_last", &mut b, [rows * WG, 1, 1], [WG, 1, 1])
}

#[allow(clippy::too_many_arguments)]
pub fn run_rmsnorm(dev: &SyclDevice, x: usize, alpha: usize, out: usize, rows: usize, cols: usize, off: usize, aoff: usize, eps: f32) -> Result<()> {
    let mut b = super::ArgBuilder::new();
    b.buf(x).buf(alpha).buf(out).i32(rows as i32).i32(cols as i32).i32(off as i32).i32(aoff as i32).f32(eps);
    dev.launch("k_rmsnorm", &mut b, [rows * WG, 1, 1], [WG, 1, 1])
}

/// RoPE over `x [b, h, t, d]` with `cos`/`sin` `[t, d/2]` (or `[b, t, d/2]`
/// when `cs_batched`); `interleaved` selects candle's `rope_i` pairing.
#[allow(clippy::too_many_arguments)]
pub fn run_rope(dev: &SyclDevice, interleaved: bool, x: usize, cos: usize, sin: usize, out: usize, (b, h, t, d): (usize, usize, usize, usize), xoff: usize, coff: usize, soff: usize, cs_batched: bool) -> Result<()> {
    let n_pairs = b * h * t * (d / 2);
    let mut b = super::ArgBuilder::new();
    b.buf(x).buf(cos).buf(sin).buf(out)
        .i32(to_i32(n_pairs)?).i32(to_i32(h)?).i32(to_i32(t)?).i32(to_i32(d)?)
        .i32(xoff as i32).i32(coff as i32).i32(soff as i32).i32(cs_batched as i32);
    dev.launch(if interleaved { "k_rope_i" } else { "k_rope" }, &mut b, [n_pairs, 1, 1], [WG, 1, 1])
}

// ─── Quantized weights ───────────────────────────────────────────────────────

/// GGUF dtype code the kernels switch on (the on-disk ggml ids).
pub fn qtype_code(dtype: crate::quantized::GgmlDType) -> i32 {
    use crate::quantized::GgmlDType::*;
    match dtype {
        F32 => 0,
        F16 => 1,
        Q4_0 => 2,
        Q4_1 => 3,
        Q5_0 => 6,
        Q5_1 => 7,
        Q8_0 => 8,
        Q8_1 => 9,
        Q2K => 10,
        Q3K => 11,
        Q4K => 12,
        Q5K => 13,
        Q6K => 14,
        Q8K => 15,
        Iq2Xxs => 16,
        BF16 => 30,
    }
}

/// Whether the quantized kernels handle `dtype` as blocks.
pub fn is_block_dtype(dtype: crate::quantized::GgmlDType) -> bool {
    use crate::quantized::GgmlDType::*;
    !matches!(dtype, F32 | F16 | BF16)
}

/// The block kernels stride whole 32-element sub-blocks and whole blocks;
/// refuse anything else and let the op take the CPU path.
fn check_block_k(k: usize, block: usize, what: &str) -> Result<()> {
    if k == 0 || !k.is_multiple_of(32) || !k.is_multiple_of(block) {
        return Err(Error::Msg(format!(
            "sycl {what}: inner dimension {k} is not a multiple of the {block}-element block"
        )));
    }
    Ok(())
}

pub const QGEMV_MR_MAX_ROWS: usize = 16;

/// Whether `dtype` has a lane-level decoder in `k_qgemv_mr` — mirror of the
/// OpenCL routing (IQ2/Q2K take the multi-row kernel for decode).
pub fn qgemv_multirow(dtype: crate::quantized::GgmlDType) -> bool {
    use crate::quantized::GgmlDType::*;
    static V1: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let forced_v1 = *V1.get_or_init(|| {
        matches!(std::env::var("JOSHUA_SYCL_QGEMV"), Ok(s) if s.eq_ignore_ascii_case("v1"))
    });
    !forced_v1 && matches!(dtype, Iq2Xxs | Q2K)
}

fn qgemv_mr_cols(k: usize, wg: usize) -> usize {
    let groups = k / 8;
    let mut cols = 1;
    while cols < 8 && groups * cols < wg {
        cols *= 2;
    }
    cols
}

/// `C[m, n] = sum_k X[m, k] * W[n, k]` over block-quantized `W` (`[N, K]`).
#[allow(clippy::too_many_arguments)]
pub fn run_qgemv(dev: &SyclDevice, dtype: crate::quantized::GgmlDType, x: usize, w: usize, out: usize, m: usize, n: usize, k: usize, woff: u64, xoff: usize) -> Result<()> {
    check_block_k(k, dtype.block_size(), "qgemv")?;
    if m <= QGEMV_MR_MAX_ROWS && qgemv_multirow(dtype) {
        let cols = qgemv_mr_cols(k, WG);
        let mut kn = super::ArgBuilder::new();
        kn.buf(x).buf(w).buf(out)
            .i32(to_i32(n)?).i32(to_i32(k)?)
            .i32(qtype_code(dtype)).i32(to_i32(dtype.type_size())?)
            .u64(woff).i32(to_i32(xoff)?).i32(0).i32(to_i32(m)?).i32(to_i32(cols)?);
        return dev.launch("k_qgemv_mr", &mut kn, [n.div_ceil(cols) * WG, 1, 1], [WG, 1, 1]);
    }
    let mut kn = super::ArgBuilder::new();
    kn.buf(x).buf(w).buf(out)
        .i32(to_i32(n)?).i32(to_i32(k)?)
        .i32(qtype_code(dtype)).i32(to_i32(dtype.block_size())?).i32(to_i32(dtype.type_size())?)
        .u64(woff).i32(to_i32(xoff)?).i32(0).i32(to_i32(m)?);
    dev.launch("k_qgemv", &mut kn, [n * WG, m, 1], [WG, 1, 1])
}

/// `C[m, n] = sum_k X[m, k] * W[n, k]` over f16 / bf16 `W`.
#[allow(clippy::too_many_arguments)]
pub fn run_hgemv(dev: &SyclDevice, bf16: bool, x: usize, w: usize, out: usize, m: usize, n: usize, k: usize, woff: u64, xoff: usize) -> Result<()> {
    let mut kn = super::ArgBuilder::new();
    kn.buf(x).buf(w).buf(out)
        .i32(to_i32(n)?).i32(to_i32(k)?).i32(bf16 as i32)
        .u64(woff).i32(to_i32(xoff)?).i32(0).i32(to_i32(m)?);
    dev.launch("k_hgemv", &mut kn, [n * WG, m, 1], [WG, 1, 1])
}

/// Dequantize `elem_count` elements of blocks to f32.
pub fn run_dequant(dev: &SyclDevice, dtype: crate::quantized::GgmlDType, w: usize, out: usize, elem_count: usize, woff: u64) -> Result<()> {
    use crate::quantized::GgmlDType::*;
    match dtype {
        F16 | BF16 => {
            let mut kn = super::ArgBuilder::new();
            kn.buf(w).buf(out).i32(to_i32(elem_count)?).i32((dtype == BF16) as i32).u64(woff);
            dev.launch("k_dequant_half", &mut kn, [elem_count, 1, 1], [WG, 1, 1])
        }
        F32 => Err(Error::Msg("sycl: f32 weights need no dequantization".into())),
        _ => {
            check_block_k(elem_count, dtype.block_size(), "dequant")?;
            let nsub = elem_count / 32;
            let mut kn = super::ArgBuilder::new();
            kn.buf(w).buf(out)
                .i32(to_i32(nsub)?).i32(qtype_code(dtype)).i32(to_i32(dtype.block_size())?).i32(to_i32(dtype.type_size())?)
                .u64(woff);
            dev.launch("k_dequant", &mut kn, [nsub, 1, 1], [WG, 1, 1])
        }
    }
}

/// Gather rows of an f16 / bf16 `[vocab, K]` table into f32 `[n_ids, K]`.
#[allow(clippy::too_many_arguments)]
pub fn run_hembed(dev: &SyclDevice, bf16: bool, w: usize, ids: usize, out: usize, n_ids: usize, k: usize, vocab: usize, woff: u64, ids_off: usize, fault: usize, fslot: i32) -> Result<()> {
    let n = n_ids * k;
    let mut kn = super::ArgBuilder::new();
    kn.buf(w).buf(ids).buf(out)
        .i32(to_i32(n)?).i32(to_i32(k)?).i32(bf16 as i32)
        .u64(woff).i32(to_i32(ids_off)?).i32(to_i32(vocab)?).buf(fault).i32(fslot);
    dev.launch("k_hembed", &mut kn, [n, 1, 1], [WG, 1, 1])
}

/// Gather rows of a block-quantized `[vocab, K]` table into f32 `[n_ids, K]`.
#[allow(clippy::too_many_arguments)]
pub fn run_qembed(dev: &SyclDevice, dtype: crate::quantized::GgmlDType, w: usize, ids: usize, out: usize, n_ids: usize, k: usize, vocab: usize, woff: u64, ids_off: usize, fault: usize, fslot: i32) -> Result<()> {
    check_block_k(k, dtype.block_size(), "qembed")?;
    let mut kn = super::ArgBuilder::new();
    kn.buf(w).buf(ids).buf(out)
        .i32(to_i32(n_ids)?).i32(to_i32(k)?)
        .i32(qtype_code(dtype)).i32(to_i32(dtype.block_size())?).i32(to_i32(dtype.type_size())?)
        .u64(woff).i32(to_i32(ids_off)?).i32(to_i32(vocab)?).buf(fault).i32(fslot);
    dev.launch("k_qembed", &mut kn, [n_ids * WG, 1, 1], [WG, 1, 1])
}
