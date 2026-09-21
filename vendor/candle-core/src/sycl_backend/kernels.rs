//! Native SYCL kernel launchers and explicit CPU fallback diagnostics.
#![allow(clippy::too_many_arguments)]
use crate::{Error, Layout, Result};
use std::ffi::c_void;
use super::bridge;
const WG: usize = 64; // matches kernels.hpp
// ─── Gating / diagnostics ────────────────────────────────────────────────────

/// Native kernels are on unless `JOSHUA_SYCL_NATIVE=0`.
pub fn native_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("JOSHUA_SYCL_NATIVE") {
        Ok(s) => !(s == "0" || s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("off")),
        Err(_) => true,
    })
}

fn trace_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("JOSHUA_SYCL_TRACE"), Ok(s) if !s.is_empty() && s != "0"))
}

static NATIVE_EXEC: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static FALLBACKS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn note_native_exec() {
    NATIVE_EXEC.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Count of native kernel launches so far (tests assert compute really ran).
pub fn native_exec_count() -> usize {
    NATIVE_EXEC.load(std::sync::atomic::Ordering::Relaxed)
}

/// Count of ops that went through the CPU round-trip.
pub fn fallback_count() -> usize {
    FALLBACKS.load(std::sync::atomic::Ordering::Relaxed)
}

static FALLBACKS_BY_OP: std::sync::Mutex<Vec<(String, usize)>> = std::sync::Mutex::new(Vec::new());

/// Per-op CPU round-trip counts (op name as passed to `note_fallback`), so a
/// test can assert *which* ops fell back rather than diffing the global
/// count, which every test in a binary shares.
pub fn fallback_counts_by_op() -> Vec<(String, usize)> {
    FALLBACKS_BY_OP.lock().unwrap_or_else(|p| p.into_inner()).clone()
}

/// Whether `JOSHUA_SYCL_CHECK_NAN=1` is set: every native launch that
/// produces an f32 storage is followed by a NaN count on the device (a
/// sync per op) and the first op whose output holds a NaN is reported.
/// A diagnostic for a result that is NaN only on one driver; never on by
/// default.  Only NaN counts: infinities are legitimate (mask fills,
/// reduction identities).
pub fn check_nan_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("JOSHUA_SYCL_CHECK_NAN"), Ok(s) if !s.is_empty() && s != "0"))
}

/// Count the NaNs in the f32 buffer `x` (`n` elements) on the device and
/// report the first op whose output has any; see [`check_nan_enabled`].
/// Errors of the check itself are swallowed (it is a diagnostic).
pub fn debug_check_nan(c: &Ctx, op: &str, x: usize, n: usize) {
    if n == 0 {
        return;
    }
    match count_nan(c, x, n) {
        Ok(0) => {}
        Ok(v) => {
            static FIRST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            let first = !FIRST.swap(true, std::sync::atomic::Ordering::Relaxed);
            eprintln!(
                "[sycl] check-nan: op `{op}` produced {v} NaN(s) in {n} elements{}",
                if first { " (first NaN-producing op)" } else { "" }
            );
        }
        Err(e) => eprintln!("[sycl] check-nan: could not check `{op}`: {e}"),
    }
}

/// Number of NaNs in the f32 device buffer `x` (`n` elements): one kernel
/// launch plus a 4-byte blocking read-back.
pub fn count_nan(c: &Ctx, x: usize, n: usize) -> Result<u32> {
    let zero = 0u32;
    let cnt = crate::sycl_backend::create_buffer(c.context, 4, 0x01)?; // CL_MEM_READ_WRITE
    let guard = scopeguard_release(cnt);
    unsafe { crate::sycl_backend::write_buffer_at(c.queue, cnt, 0, 4, &zero as *const u32 as *const u8) }?;
    let mut kn = c.kernel("k_count_nan")?;
    kn.buf(x)?.buf(cnt)?.val(to_i32(n)?)?;
    kn.run(&[n], None)?;
    let mut v = 0u32;
    unsafe { crate::sycl_backend::read_buffer(c.queue, cnt, 0, 4, &mut v as *mut u32 as *mut u8) }?;
    drop(guard);
    Ok(v)
}

struct ReleaseOnDrop(usize);
impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        crate::sycl_backend::release_mem(self.0);
    }
}
fn scopeguard_release(buffer: usize) -> ReleaseOnDrop {
    ReleaseOnDrop(buffer)
}

/// Record (and, under `JOSHUA_SYCL_TRACE`, log) an op that fell back to the
/// CPU path.  `why` is `None` when no native kernel exists for the case and
/// `Some(err)` when the kernel failed.
pub fn note_fallback(op: &str, why: Option<&Error>) {
    FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    {
        let mut by_op = FALLBACKS_BY_OP.lock().unwrap_or_else(|p| p.into_inner());
        match by_op.iter_mut().find(|(name, _)| name == op) {
            Some((_, n)) => *n += 1,
            None => by_op.push((op.to_string(), 1)),
        }
    }
    if trace_enabled() {
        match why {
            Some(e) => eprintln!("[sycl] {op}: native kernel failed, using CPU round-trip: {e}"),
            None => eprintln!("[sycl] {op}: no native kernel for this case, using CPU round-trip"),
        }
    }
}

// ─── Index descriptor ────────────────────────────────────────────────────────

pub const MAXD: usize = 8;

/// Output shape plus up to three input stride sets, passed to kernels by
/// value.  Layout must match the `Idx` struct in `kernels.hpp` exactly.
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
            return Err(Error::Msg(format!("sycl: rank {} exceeds the kernel limit of {MAXD}", dims.len())));
        }
        let mut ix = Idx { nd: dims.len() as i32, dims: [1; MAXD], s0: [0; MAXD], s1: [0; MAXD], s2: [0; MAXD], o0: 0, o1: 0, o2: 0 };
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

// ─── Launch helper ───────────────────────────────────────────────────────────

/// Arguments are copied here and captured by value by the SYCL submission.
/// Each launch owns its argument vector, so concurrent sessions cannot race.
pub struct Kernel {
    queue: usize,
    name: std::ffi::CString,
    args: Vec<Vec<u8>>,
    pub wg: usize,
}
impl Kernel {
    pub fn new(_ctx: usize, _dev: usize, queue: usize, name: &str) -> Result<Self> {
        Ok(Self { queue, name: std::ffi::CString::new(name).map_err(Error::wrap)?, args: Vec::new(), wg: WG })
    }
    fn set(&mut self, size: usize, ptr: *const c_void) -> Result<&mut Self> {
        self.args.push(unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), size) }.to_vec());
        Ok(self)
    }
    /// A SYCL buffer argument.
    pub fn buf(&mut self, buffer: usize) -> Result<&mut Self> {
        self.set(std::mem::size_of::<usize>(), &buffer as *const usize as *const c_void)
    }

    /// A plain-old-data argument passed by value (`i32`, `f32`, `u64`, [`Idx`], …).
    pub fn val<T: Copy>(&mut self, v: T) -> Result<&mut Self> {
        self.set(std::mem::size_of::<T>(), &v as *const T as *const c_void)
    }

    pub fn run(&mut self, global: &[usize], local: Option<&[usize]>) -> Result<()> {
        if global.contains(&0) { return Ok(()); }
        if global.is_empty() || global.len() > 3 { crate::bail!("sycl: invalid launch rank"); }
        let mut g = [1usize; 3]; let mut l = [1usize; 3];
        g[..global.len()].copy_from_slice(global);
        if let Some(local) = local {
            if local.len() != global.len() { crate::bail!("sycl: local/global rank mismatch"); }
            l[..local.len()].copy_from_slice(local);
        } else {
            l[0] = WG;
            g[0] = g[0].checked_add(WG - 1).ok_or_else(|| Error::Msg("sycl: launch size overflow".into()))? / WG * WG;
        }
        let args: Vec<_> = self.args.iter().map(|a| bridge::Arg { data: a.as_ptr().cast(), size: a.len() }).collect();
        unsafe { bridge::launch(self.queue, &self.name, &args, &g, &l) }?;
        note_native_exec();
        Ok(())
    }
}

/// Device handles a launch needs.
#[derive(Clone, Copy)]
pub struct Ctx {
    pub context: usize,
    pub device: usize,
    pub queue: usize,
    /// The device's fault buffer (see `kernels.hpp`): indexing kernels set
    /// the calling thread's slot on an out-of-range id; the host reports
    /// and clears it at the next read-back.
    pub fault: usize,
    /// The calling thread's slot in `fault`.
    pub fslot: i32,
}

impl Ctx {
    pub fn kernel(&self, name: &str) -> Result<Kernel> {
        Kernel::new(self.context, self.device, self.queue, name)
    }
}

// ─── Op codes (must match kernels.hpp) ────────────────────────────────────────

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
        "abs" => 13,
        "ceil" => 14,
        "floor" => 15,
        "round" => 16,
        "relu" => 17,
        "sign" => 18,
        "sigmoid" => 19,
        _ => return None,
    })
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

pub const RED_SUM: i32 = 0;
pub const RED_MAX: i32 = 1;
pub const RED_MIN: i32 = 2;

// ─── Elementwise ─────────────────────────────────────────────────────────────

pub fn run_unary(c: &Ctx, op: i32, x: usize, out: usize, n: usize, l: &Layout) -> Result<()> {
    if let Some((o1, _)) = l.contiguous_offsets() {
        let mut k = c.kernel("k_unary_c")?;
        k.buf(x)?.buf(out)?.val(to_i32(n)?)?.val(to_i32(o1)?)?.val(op)?;
        k.run(&[n], None)
    } else {
        let ix = Idx::new(l.dims())?.with_layout(0, l)?;
        let mut k = c.kernel("k_unary_s")?;
        k.buf(x)?.buf(out)?.val(to_i32(n)?)?.val(ix)?.val(op)?;
        k.run(&[n], None)
    }
}

pub fn run_affine(c: &Ctx, x: usize, out: usize, n: usize, l: &Layout, mul: f32, add: f32) -> Result<()> {
    if let Some((o1, _)) = l.contiguous_offsets() {
        let mut k = c.kernel("k_affine_c")?;
        k.buf(x)?.buf(out)?.val(to_i32(n)?)?.val(to_i32(o1)?)?.val(mul)?.val(add)?;
        k.run(&[n], None)
    } else {
        let ix = Idx::new(l.dims())?.with_layout(0, l)?;
        let mut k = c.kernel("k_affine_s")?;
        k.buf(x)?.buf(out)?.val(to_i32(n)?)?.val(ix)?.val(mul)?.val(add)?;
        k.run(&[n], None)
    }
}

pub fn run_powf(c: &Ctx, x: usize, out: usize, n: usize, l: &Layout, e: f32) -> Result<()> {
    let ix = Idx::new(l.dims())?.with_layout(0, l)?;
    let mut k = c.kernel("k_powf_s")?;
    k.buf(x)?.buf(out)?.val(to_i32(n)?)?.val(ix)?.val(e)?;
    k.run(&[n], None)
}

pub fn run_elu(c: &Ctx, x: usize, out: usize, n: usize, l: &Layout, alpha: f32) -> Result<()> {
    let ix = Idx::new(l.dims())?.with_layout(0, l)?;
    let mut k = c.kernel("k_elu_s")?;
    k.buf(x)?.buf(out)?.val(to_i32(n)?)?.val(ix)?.val(alpha)?;
    k.run(&[n], None)
}

/// f32 binary op; `u32` selects the integer kernel.
pub fn run_binary(c: &Ctx, op: i32, a: usize, b: usize, out: usize, n: usize, la: &Layout, lb: &Layout, u32: bool) -> Result<()> {
    if !u32 {
        if let (Some((oa, _)), Some((ob, _))) = (la.contiguous_offsets(), lb.contiguous_offsets()) {
            let mut k = c.kernel("k_binary_c")?;
            k.buf(a)?.buf(b)?.buf(out)?.val(to_i32(n)?)?.val(to_i32(oa)?)?.val(to_i32(ob)?)?.val(op)?;
            return k.run(&[n], None);
        }
    }
    let ix = Idx::new(la.dims())?.with_layout(0, la)?.with_layout(1, lb)?;
    let mut k = c.kernel(if u32 { "k_binary_u32_s" } else { "k_binary_s" })?;
    k.buf(a)?.buf(b)?.buf(out)?.val(to_i32(n)?)?.val(ix)?.val(op)?;
    k.run(&[n], None)
}

pub fn run_cmp(c: &Ctx, op: i32, a: usize, b: usize, out: usize, n: usize, la: &Layout, lb: &Layout, u32: bool) -> Result<()> {
    let ix = Idx::new(la.dims())?.with_layout(0, la)?.with_layout(1, lb)?;
    let mut k = c.kernel(if u32 { "k_cmp_u32" } else { "k_cmp_f32" })?;
    k.buf(a)?.buf(b)?.buf(out)?.val(to_i32(n)?)?.val(ix)?.val(op)?;
    k.run(&[n], None)
}

/// `where_cond` with a u8 or u32 condition over 4- or 8-byte payloads.
pub fn run_where(c: &Ctx, cond: usize, t: usize, f: usize, out: usize, n: usize, lc: &Layout, lt: &Layout, lf: &Layout, cond_u32: bool, elem8: bool) -> Result<()> {
    let ix = Idx::new(lc.dims())?.with_layout(0, lc)?.with_layout(1, lt)?.with_layout(2, lf)?;
    let name = match (cond_u32, elem8) {
        (false, false) => "k_where_4",
        (false, true) => "k_where_8",
        (true, false) => "k_where_u32c_4",
        (true, true) => return Err(Error::Msg("sycl: u32 condition with 8-byte payload has no kernel".into())),
    };
    let mut k = c.kernel(name)?;
    k.buf(cond)?.buf(t)?.buf(f)?.buf(out)?.val(to_i32(n)?)?.val(ix)?;
    k.run(&[n], None)
}

/// Strided copy of `n` elements (`elem` bytes each) described by `l` into
/// `dst` at element offset `dst_off`, contiguous.
pub fn run_copy_strided(c: &Ctx, elem: usize, src: usize, dst: usize, n: usize, l: &Layout, dst_off: usize) -> Result<()> {
    let name = match elem {
        1 => "k_copy_s1",
        2 => "k_copy_s2",
        4 => "k_copy_s4",
        8 => "k_copy_s8",
        _ => return Err(Error::Msg(format!("sycl: no copy kernel for {elem}-byte elements"))),
    };
    let ix = Idx::new(l.dims())?.with_layout(0, l)?;
    let mut k = c.kernel(name)?;
    k.buf(src)?.buf(dst)?.val(to_i32(n)?)?.val(ix)?.val(to_i32(dst_off)?)?;
    k.run(&[n], None)
}

#[allow(clippy::too_many_arguments)]
pub fn run_copy2d(c: &Ctx, elem: usize, src: usize, dst: usize, d1: usize, d2: usize, src_s: usize, dst_s: usize, src_o: usize, dst_o: usize) -> Result<()> {
    let name = match elem {
        1 => "k_copy2d_1",
        2 => "k_copy2d_2",
        4 => "k_copy2d_4",
        8 => "k_copy2d_8",
        _ => return Err(Error::Msg(format!("sycl: no copy2d kernel for {elem}-byte elements"))),
    };
    let mut k = c.kernel(name)?;
    k.buf(src)?.buf(dst)?.val(to_i32(d1)?)?.val(to_i32(d2)?)?.val(to_i32(src_s)?)?.val(to_i32(dst_s)?)?.val(to_i32(src_o)?)?.val(to_i32(dst_o)?)?;
    k.run(&[d2, d1], None)
}

/// Fill the elements addressed by `l` with the bit pattern `bits` (element
/// size `elem`).
pub fn run_fill(c: &Ctx, elem: usize, dst: usize, n: usize, l: &Layout, bits: u64) -> Result<()> {
    let ix = Idx::new(l.dims())?.with_layout(0, l)?;
    match elem {
        1 => { let mut k = c.kernel("k_fill_1")?; k.buf(dst)?.val(to_i32(n)?)?.val(ix)?.val(bits as u8)?; k.run(&[n], None) }
        2 => { let mut k = c.kernel("k_fill_2")?; k.buf(dst)?.val(to_i32(n)?)?.val(ix)?.val(bits as u16)?; k.run(&[n], None) }
        4 => { let mut k = c.kernel("k_fill_4")?; k.buf(dst)?.val(to_i32(n)?)?.val(ix)?.val(bits as u32)?; k.run(&[n], None) }
        8 => { let mut k = c.kernel("k_fill_8")?; k.buf(dst)?.val(to_i32(n)?)?.val(ix)?.val(bits)?; k.run(&[n], None) }
        _ => Err(Error::Msg(format!("sycl: no fill kernel for {elem}-byte elements"))),
    }
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

pub fn run_cast(c: &Ctx, name: &str, x: usize, out: usize, n: usize, l: &Layout) -> Result<()> {
    let ix = Idx::new(l.dims())?.with_layout(0, l)?;
    let mut k = c.kernel(name)?;
    k.buf(x)?.buf(out)?.val(to_i32(n)?)?.val(ix)?;
    k.run(&[n], None)
}

// ─── Reductions ──────────────────────────────────────────────────────────────

/// Reduce the contiguous last dim: `rows` rows of `cols` starting at element `off`.
pub fn run_reduce_last(c: &Ctx, op: i32, x: usize, out: usize, rows: usize, cols: usize, off: usize) -> Result<()> {
    let mut k = c.kernel("k_reduce_last")?;
    let wg = k.wg;
    k.buf(x)?.buf(out)?.val(to_i32(rows)?)?.val(to_i32(cols)?)?.val(to_i32(off)?)?.val(op)?;
    k.run(&[rows * wg], Some(&[wg]))
}

/// ArgMax / ArgMin over the contiguous last dim (u32 output).
pub fn run_arg_last(c: &Ctx, is_max: bool, x: usize, out: usize, rows: usize, cols: usize, off: usize) -> Result<()> {
    let mut k = c.kernel("k_arg_last")?;
    let wg = k.wg;
    k.buf(x)?.buf(out)?.val(to_i32(rows)?)?.val(to_i32(cols)?)?.val(to_i32(off)?)?.val(is_max as i32)?;
    k.run(&[rows * wg], Some(&[wg]))
}

/// General reduction: `ix` maps each output element to its input base
/// (reduced dims get stride 0 and dim 1); `rd` enumerates the reduced
/// sub-space (`count` elements).
pub fn run_reduce_generic(c: &Ctx, op: i32, x: usize, out: usize, n: usize, ix: Idx, rd: Idx, count: usize) -> Result<()> {
    let mut k = c.kernel("k_reduce_generic")?;
    k.buf(x)?.buf(out)?.val(to_i32(n)?)?.val(ix)?.val(rd)?.val(to_i32(count)?)?.val(op)?;
    k.run(&[n], None)
}

// ─── Indexing ────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub fn run_index_select(c: &Ctx, elem8: bool, ids_i64: bool, src: usize, ids: usize, out: usize, n: usize, left: usize, n_ids: usize, right: usize, dim_size: usize, src_off: usize, ids_off: usize) -> Result<()> {
    let name = match (elem8, ids_i64) {
        (false, false) => "k_index_select_u32_4",
        (false, true) => "k_index_select_i64_4",
        (true, false) => "k_index_select_u32_8",
        (true, true) => return Err(Error::Msg("sycl: i64 ids with 8-byte elements has no kernel".into())),
    };
    let mut k = c.kernel(name)?;
    k.buf(src)?.buf(ids)?.buf(out)?.val(to_i32(n)?)?.val(to_i32(left)?)?.val(to_i32(n_ids)?)?.val(to_i32(right)?)?.val(to_i32(dim_size)?)?.val(to_i32(src_off)?)?.val(to_i32(ids_off)?)?.buf(c.fault)?.val(c.fslot)?;
    k.run(&[n], None)
}

#[allow(clippy::too_many_arguments)]
pub fn run_gather(c: &Ctx, src: usize, ids: usize, out: usize, n: usize, ix: Idx, src_dim_stride: usize, dim_size: usize) -> Result<()> {
    let mut k = c.kernel("k_gather_4")?;
    k.buf(src)?.buf(ids)?.buf(out)?.val(to_i32(n)?)?.val(ix)?.val(to_i32(src_dim_stride)?)?.val(to_i32(dim_size)?)?.buf(c.fault)?.val(c.fslot)?;
    k.run(&[n], None)
}

/// scatter set / add: `ix` enumerates the ids space with `dim` collapsed
/// (s0 ids, s1 src, s2 dst); the `*_ds` are the strides along `dim`, which
/// the kernel walks in order (`n_j` positions).
#[allow(clippy::too_many_arguments)]
pub fn run_scatter(c: &Ctx, add: bool, dst: usize, ids: usize, src: usize, n: usize, ix: Idx, n_j: usize, ids_ds: usize, src_ds: usize, dst_ds: usize, dim_size: usize) -> Result<()> {
    let mut k = c.kernel(if add { "k_scatter_add_f32" } else { "k_scatter_set_4" })?;
    k.buf(dst)?.buf(ids)?.buf(src)?.val(to_i32(n)?)?.val(ix)?.val(to_i32(n_j)?)?.val(to_i32(ids_ds)?)?.val(to_i32(src_ds)?)?.val(to_i32(dst_ds)?)?.val(to_i32(dim_size)?)?.buf(c.fault)?.val(c.fslot)?;
    k.run(&[n], None)
}

#[allow(clippy::too_many_arguments)]
pub fn run_index_add(c: &Ctx, dst: usize, ids: usize, src: usize, left: usize, n_ids: usize, right: usize, dim_size: usize, src_off: usize, ids_off: usize) -> Result<()> {
    let mut k = c.kernel("k_index_add_f32")?;
    let n_lr = left * right;
    k.buf(dst)?.buf(ids)?.buf(src)?.val(to_i32(n_lr)?)?.val(to_i32(left)?)?.val(to_i32(n_ids)?)?.val(to_i32(right)?)?.val(to_i32(dim_size)?)?.val(to_i32(src_off)?)?.val(to_i32(ids_off)?)?.buf(c.fault)?.val(c.fslot)?;
    k.run(&[n_lr], None)
}

// ─── Dense GEMM / GEMV ───────────────────────────────────────────────────────

/// Strides of one operand of a (batched) matmul.
#[derive(Clone, Copy, Debug)]
pub struct MatStrides {
    /// Stride along the row index (m for A, k for B).
    pub row: usize,
    /// Stride along the column index (k for A, n for B).
    pub col: usize,
    /// Offset of the first element.
    pub offset: usize,
    /// Stride between consecutive batch matrices (0 = broadcast).
    pub batch: usize,
}

/// `C[bz] = A[bz] @ B[bz]` for `batch` matrices of `(m, k) @ (k, n)`; C is
/// written contiguous `[batch, m, n]`.  Picks the GEMV kernels for `m == 1`.
pub fn run_matmul(c: &Ctx, a: usize, b: usize, out: usize, (batch, m, n, k): (usize, usize, usize, usize), sa: MatStrides, sb: MatStrides) -> Result<()> {
    if batch == 0 || m == 0 || n == 0 {
        return Ok(());
    }
    if k == 0 {
        // An empty contraction is a zero matrix on every backend; the freshly
        // allocated output holds whatever the device had there, so fill it.
        let n_out = batch * m * n;
        return run_fill(c, 4, out, n_out, &Layout::contiguous(n_out), 0);
    }
    let (m32, n32, k32) = (to_i32(m)?, to_i32(n)?, to_i32(k)?);
    if m == 1 && sa.col == 1 {
        let (oa, ob) = (to_i32(sa.offset)?, to_i32(sb.offset)?);
        let (ba, bb, bc) = (to_i32(sa.batch)?, to_i32(sb.batch)?, to_i32(n)?);
        if sb.row == 1 {
            // B contiguous along k: x @ W^T with W [N, K].
            let mut kn = c.kernel("k_gemv_nt")?;
            let wg = kn.wg;
            kn.buf(a)?.buf(b)?.buf(out)?.val(n32)?.val(k32)?.val(to_i32(sb.col)?)?.val(oa)?.val(ob)?.val(0i32)?.val(ba)?.val(bb)?.val(bc)?;
            return kn.run(&[n * wg, batch], Some(&[wg, 1]));
        }
        if sb.col == 1 {
            let mut kn = c.kernel("k_gemv_nn")?;
            kn.buf(a)?.buf(b)?.buf(out)?.val(n32)?.val(k32)?.val(to_i32(sb.row)?)?.val(oa)?.val(ob)?.val(0i32)?.val(ba)?.val(bb)?.val(bc)?;
            return kn.run(&[n, batch], None);
        }
    }
    let mut kn = c.kernel("k_gemm")?;
    kn.buf(a)?.buf(b)?.buf(out)?
        .val(m32)?.val(n32)?.val(k32)?
        .val(to_i32(sa.row)?)?.val(to_i32(sa.col)?)?.val(to_i32(sb.row)?)?.val(to_i32(sb.col)?)?
        .val(to_i32(sa.offset)?)?.val(to_i32(sb.offset)?)?.val(0i32)?
        .val(to_i32(sa.batch)?)?.val(to_i32(sb.batch)?)?.val(to_i32(m * n)?)?
        .val((sb.row == 1) as i32)?;
    let gx = n.div_ceil(64) * 16;
    let gy = m.div_ceil(64) * 16;
    kn.run(&[gx, gy, batch], Some(&[16, 16, 1]))
}

// ─── Fused attention-path ops ────────────────────────────────────────────────

pub fn run_softmax_last(c: &Ctx, x: usize, out: usize, rows: usize, cols: usize, off: usize) -> Result<()> {
    let mut k = c.kernel("k_softmax_last")?;
    let wg = k.wg;
    k.buf(x)?.buf(out)?.val(to_i32(rows)?)?.val(to_i32(cols)?)?.val(to_i32(off)?)?;
    k.run(&[rows * wg], Some(&[wg]))
}

#[allow(clippy::too_many_arguments)]
pub fn run_rmsnorm(c: &Ctx, x: usize, alpha: usize, out: usize, rows: usize, cols: usize, off: usize, aoff: usize, eps: f32) -> Result<()> {
    let mut k = c.kernel("k_rmsnorm")?;
    let wg = k.wg;
    k.buf(x)?.buf(alpha)?.buf(out)?.val(to_i32(rows)?)?.val(to_i32(cols)?)?.val(to_i32(off)?)?.val(to_i32(aoff)?)?.val(eps)?;
    k.run(&[rows * wg], Some(&[wg]))
}

/// RoPE over `x [b, h, t, d]` with `cos`/`sin` `[t, d/2]` (or `[b, t, d/2]`
/// when `cs_batched`); `interleaved` selects candle's `rope_i` pairing.
#[allow(clippy::too_many_arguments)]
pub fn run_rope(c: &Ctx, interleaved: bool, x: usize, cos: usize, sin: usize, out: usize, (b, h, t, d): (usize, usize, usize, usize), xoff: usize, coff: usize, soff: usize, cs_batched: bool) -> Result<()> {
    let n_pairs = b * h * t * (d / 2);
    let mut k = c.kernel(if interleaved { "k_rope_i" } else { "k_rope" })?;
    k.buf(x)?.buf(cos)?.buf(sin)?.buf(out)?
        .val(to_i32(n_pairs)?)?.val(to_i32(h)?)?.val(to_i32(t)?)?.val(to_i32(d)?)?
        .val(to_i32(xoff)?)?.val(to_i32(coff)?)?.val(to_i32(soff)?)?.val(cs_batched as i32)?;
    k.run(&[n_pairs], None)
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

/// Whether the quantized kernels handle `dtype` as blocks (`k_qgemv` /
/// `k_dequant`).  F16/BF16 go through the half kernels and F32 is dense.
pub fn is_block_dtype(dtype: crate::quantized::GgmlDType) -> bool {
    use crate::quantized::GgmlDType::*;
    !matches!(dtype, F32 | F16 | BF16)
}

/// The block kernels stride whole 32-element sub-blocks and whole blocks;
/// an inner dimension that is not a multiple of both would silently drop its
/// tail, so refuse it here and let the op take the CPU path.
fn check_block_k(k: usize, block: usize, what: &str) -> Result<()> {
    if k == 0 || !k.is_multiple_of(32) || !k.is_multiple_of(block) {
        return Err(Error::Msg(format!("sycl {what}: inner dimension {k} is not a multiple of the {block}-element block")));
    }
    Ok(())
}

/// Rows the multi-row kernel accumulates per launch (`QGEMV_MR_MAX_ROWS`).
pub const QGEMV_MR_MAX_ROWS: usize = 16;

/// Whether `dtype` has a lane-level decoder in `k_qgemv_mr` (the routed
/// experts' formats).  `JOSHUA_SYCL_QGEMV=v1` forces the one-row kernel
/// for every dtype (a bisect switch for a wrong result on one driver).
pub fn qgemv_multirow(dtype: crate::quantized::GgmlDType) -> bool {
    use crate::quantized::GgmlDType::*;
    static V1: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let forced_v1 = *V1.get_or_init(|| matches!(std::env::var("JOSHUA_SYCL_QGEMV"), Ok(s) if s.eq_ignore_ascii_case("v1")));
    !forced_v1 && matches!(dtype, Iq2Xxs | Q2K)
}

/// Output columns one `k_qgemv_mr` work-group handles: enough that every
/// lane owns at least one 8-element group of its column (`K / 8` groups per
/// column), capped at 8 so a group never spans more than 8 columns.
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
pub fn run_qgemv(c: &Ctx, dtype: crate::quantized::GgmlDType, x: usize, w: usize, out: usize, m: usize, n: usize, k: usize, woff: u64, xoff: usize) -> Result<()> {
    check_block_k(k, dtype.block_size(), "qgemv")?;
    if m <= QGEMV_MR_MAX_ROWS && qgemv_multirow(dtype) {
        let mut kn = c.kernel("k_qgemv_mr")?;
        let wg = kn.wg;
        let cols = qgemv_mr_cols(k, wg);
        kn.buf(x)?.buf(w)?.buf(out)?
            .val(to_i32(n)?)?.val(to_i32(k)?)?
            .val(qtype_code(dtype))?.val(to_i32(dtype.type_size())?)?
            .val(woff)?.val(to_i32(xoff)?)?.val(0i32)?.val(to_i32(m)?)?.val(to_i32(cols)?)?;
        return kn.run(&[n.div_ceil(cols) * wg], Some(&[wg]));
    }
    let mut kn = c.kernel("k_qgemv")?;
    let wg = kn.wg;
    kn.buf(x)?.buf(w)?.buf(out)?
        .val(to_i32(n)?)?.val(to_i32(k)?)?
        .val(qtype_code(dtype))?.val(to_i32(dtype.block_size())?)?.val(to_i32(dtype.type_size())?)?
        .val(woff)?.val(to_i32(xoff)?)?.val(0i32)?.val(to_i32(m)?)?;
    kn.run(&[n * wg, m], Some(&[wg, 1]))
}

/// `C[m, n] = sum_k X[m, k] * W[n, k]` over f16 / bf16 `W`.
#[allow(clippy::too_many_arguments)]
pub fn run_hgemv(c: &Ctx, bf16: bool, x: usize, w: usize, out: usize, m: usize, n: usize, k: usize, woff: u64, xoff: usize) -> Result<()> {
    let mut kn = c.kernel("k_hgemv")?;
    let wg = kn.wg;
    kn.buf(x)?.buf(w)?.buf(out)?
        .val(to_i32(n)?)?.val(to_i32(k)?)?.val(bf16 as i32)?
        .val(woff)?.val(to_i32(xoff)?)?.val(0i32)?.val(to_i32(m)?)?;
    kn.run(&[n * wg, m], Some(&[wg, 1]))
}

/// Dequantize `elem_count` elements of blocks to f32.
pub fn run_dequant(c: &Ctx, dtype: crate::quantized::GgmlDType, w: usize, out: usize, elem_count: usize, woff: u64) -> Result<()> {
    use crate::quantized::GgmlDType::*;
    match dtype {
        F16 | BF16 => {
            let mut kn = c.kernel("k_dequant_half")?;
            kn.buf(w)?.buf(out)?.val(to_i32(elem_count)?)?.val((dtype == BF16) as i32)?.val(woff)?;
            kn.run(&[elem_count], None)
        }
        F32 => Err(Error::Msg("sycl: f32 weights need no dequantization".into())),
        _ => {
            check_block_k(elem_count, dtype.block_size(), "dequant")?;
            let nsub = elem_count / 32;
            let mut kn = c.kernel("k_dequant")?;
            kn.buf(w)?.buf(out)?.val(to_i32(nsub)?)?.val(qtype_code(dtype))?.val(to_i32(dtype.block_size())?)?.val(to_i32(dtype.type_size())?)?.val(woff)?;
            kn.run(&[nsub], None)
        }
    }
}

/// Gather rows of an f16 / bf16 `[vocab, K]` table into f32 `[n_ids, K]`.
#[allow(clippy::too_many_arguments)]
pub fn run_hembed(c: &Ctx, bf16: bool, w: usize, ids: usize, out: usize, n_ids: usize, k: usize, vocab: usize, woff: u64, ids_off: usize) -> Result<()> {
    let n = n_ids * k;
    let mut kn = c.kernel("k_hembed")?;
    kn.buf(w)?.buf(ids)?.buf(out)?
        .val(to_i32(n)?)?.val(to_i32(k)?)?.val(bf16 as i32)?
        .val(woff)?.val(to_i32(ids_off)?)?.val(to_i32(vocab)?)?.buf(c.fault)?.val(c.fslot)?;
    kn.run(&[n], None)
}

/// Gather rows of a block-quantized `[vocab, K]` table into f32 `[n_ids, K]`.
#[allow(clippy::too_many_arguments)]
pub fn run_qembed(c: &Ctx, dtype: crate::quantized::GgmlDType, w: usize, ids: usize, out: usize, n_ids: usize, k: usize, vocab: usize, woff: u64, ids_off: usize) -> Result<()> {
    check_block_k(k, dtype.block_size(), "qembed")?;
    let mut kn = c.kernel("k_qembed")?;
    let wg = kn.wg;
    kn.buf(w)?.buf(ids)?.buf(out)?
        .val(to_i32(n_ids)?)?.val(to_i32(k)?)?
        .val(qtype_code(dtype))?.val(to_i32(dtype.block_size())?)?.val(to_i32(dtype.type_size())?)?
        .val(woff)?.val(to_i32(ids_off)?)?.val(to_i32(vocab)?)?.buf(c.fault)?.val(c.fslot)?;
    kn.run(&[n_ids * wg], Some(&[wg]))
}
