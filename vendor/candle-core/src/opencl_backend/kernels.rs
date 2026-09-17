//! Native OpenCL kernels: program cache, argument marshalling and one launch
//! helper per kernel in `kernels.cl`.
//!
//! The program is compiled once per context (cached in a global map) with
//! the reduction work-group size (`WG`) chosen from the device's limit.
//! Kernel objects are created per launch: `clSetKernelArg` on a shared
//! kernel is not thread-safe and sessions run ops concurrently, whereas a
//! kernel object is a few microseconds to create.  Launches are asynchronous
//! on the device's in-order queue; the only synchronisation points are the
//! blocking reads in `to_cpu_storage` and an explicit `synchronize`.
//!
//! Native execution is on by default; `JOSHUA_OPENCL_NATIVE=0` forces every
//! op through the CPU round-trip (a correctness reference), and
//! `JOSHUA_OPENCL_TRACE=1` logs each op that still falls back, which is how
//! to find the next kernel worth writing.
// Kernel launchers mirror their kernel's argument list one-to-one.
#![allow(clippy::too_many_arguments)]

use crate::{Error, Layout, Result};
use std::ffi::c_void;

extern "C" {
    fn clCreateProgramWithSource(context: usize, count: u32, strings: *const *const c_void, lengths: *const usize, err: *mut i32) -> usize;
    fn clBuildProgram(program: usize, ndev: u32, devices: *const usize, opts: *const c_void, cb: *const c_void, ud: *const c_void) -> i32;
    fn clGetProgramBuildInfo(program: usize, device: usize, param: u32, size: usize, value: *mut c_void, ret: *mut usize) -> i32;
    fn clCreateKernel(program: usize, name: *const c_void, err: *mut i32) -> usize;
    fn clSetKernelArg(kernel: usize, index: u32, size: usize, value: *const c_void) -> i32;
    fn clEnqueueNDRangeKernel(q: usize, kernel: usize, dim: u32, off: *const usize, gsz: *const usize, lsz: *const usize, nev: u32, ev: *const usize, e: *mut usize) -> i32;
    fn clReleaseProgram(program: usize) -> i32;
    fn clReleaseKernel(kernel: usize) -> i32;
    fn clGetDeviceInfo(device: usize, param: u32, size: usize, value: *mut c_void, ret: *mut usize) -> i32;
}

const CL_SUCCESS: i32 = 0;
const CL_PROGRAM_BUILD_LOG: u32 = 0x1183;
const CL_DEVICE_MAX_WORK_GROUP_SIZE: u32 = 0x1004;

const KERNEL_SRC: &str = include_str!("kernels.cl");

/// Largest reduction work-group the kernels are built for.
const MAX_WG: usize = 256;

fn err(code: i32, op: &str) -> Error {
    Error::Msg(format!("opencl kernel {op} failed with status {code}"))
}

// ─── Program cache ───────────────────────────────────────────────────────────

struct Prog {
    context: usize,
    program: usize,
    /// Reduction work-group size the program was compiled with.
    wg: usize,
}

impl Drop for Prog {
    fn drop(&mut self) {
        if self.program != 0 {
            unsafe { clReleaseProgram(self.program) };
        }
    }
}

static PROGS: std::sync::OnceLock<std::sync::Mutex<Vec<Prog>>> = std::sync::OnceLock::new();

/// Largest power of two ≤ the device's max work-group size, capped at [`MAX_WG`].
fn reduction_wg(device_id: usize) -> usize {
    let mut max: usize = 0;
    let rc = unsafe {
        clGetDeviceInfo(
            device_id,
            CL_DEVICE_MAX_WORK_GROUP_SIZE,
            std::mem::size_of::<usize>(),
            &mut max as *mut usize as *mut c_void,
            std::ptr::null_mut(),
        )
    };
    let max = if rc == CL_SUCCESS && max > 0 { max } else { 64 };
    let mut wg = 1usize;
    while wg * 2 <= max.min(MAX_WG) {
        wg *= 2;
    }
    wg
}

/// Compile (or fetch the cached) program for `(context, device_id)`.  Returns
/// the program handle and the reduction work-group size it was built with.
pub fn program_for(context: usize, device_id: usize) -> Result<(usize, usize)> {
    let cache = PROGS.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    let mut list = cache.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(p) = list.iter().find(|p| p.context == context) {
        return Ok((p.program, p.wg));
    }
    let wg = reduction_wg(device_id);
    let src = KERNEL_SRC.as_bytes();
    let srcs = [src.as_ptr() as *const c_void];
    let lens = [src.len()];
    let mut e: i32 = 0;
    let program = unsafe { clCreateProgramWithSource(context, 1, srcs.as_ptr(), lens.as_ptr(), &mut e) };
    if e != CL_SUCCESS || program == 0 {
        return Err(err(e, "clCreateProgramWithSource"));
    }
    let guard = Prog { context, program, wg };
    let opts = std::ffi::CString::new(format!("-D WG={wg} -cl-std=CL1.2")).unwrap();
    let bc = unsafe {
        clBuildProgram(program, 1, &device_id, opts.as_ptr() as *const c_void, std::ptr::null(), std::ptr::null())
    };
    if bc != CL_SUCCESS {
        // Fetch the build log so a broken kernel is diagnosable.
        let mut len: usize = 0;
        unsafe { clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, std::ptr::null_mut(), &mut len) };
        let mut log = vec![0u8; len.max(1)];
        unsafe {
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log.len(), log.as_mut_ptr() as *mut c_void, std::ptr::null_mut())
        };
        let log = String::from_utf8_lossy(&log).trim_end_matches('\0').to_string();
        drop(guard);
        return Err(Error::Msg(format!("opencl clBuildProgram failed with status {bc}:\n{log}")));
    }
    list.push(guard);
    Ok((program, wg))
}

// ─── Gating / diagnostics ────────────────────────────────────────────────────

/// Native kernels are on unless `JOSHUA_OPENCL_NATIVE=0`.
pub fn native_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("JOSHUA_OPENCL_NATIVE") {
        Ok(s) => !(s == "0" || s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("off")),
        Err(_) => true,
    })
}

fn trace_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("JOSHUA_OPENCL_TRACE"), Ok(s) if !s.is_empty() && s != "0"))
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

/// Record (and, under `JOSHUA_OPENCL_TRACE`, log) an op that fell back to the
/// CPU path.  `why` is `None` when no native kernel exists for the case and
/// `Some(err)` when the kernel failed.
pub fn note_fallback(op: &str, why: Option<&Error>) {
    FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if trace_enabled() {
        match why {
            Some(e) => eprintln!("[opencl] {op}: native kernel failed, using CPU round-trip: {e}"),
            None => eprintln!("[opencl] {op}: no native kernel for this case, using CPU round-trip"),
        }
    }
}

// ─── Index descriptor ────────────────────────────────────────────────────────

pub const MAXD: usize = 8;

/// Output shape plus up to three input stride sets, passed to kernels by
/// value.  Layout must match the `Idx` struct in `kernels.cl` exactly.
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
            return Err(Error::Msg(format!("opencl: rank {} exceeds the kernel limit of {MAXD}", dims.len())));
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
            return Err(Error::Msg("opencl: layout rank mismatch".into()));
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
    i32::try_from(v).map_err(|_| Error::Msg(format!("opencl: dimension {v} exceeds the 32-bit kernel index range")))
}

// ─── Launch helper ───────────────────────────────────────────────────────────

/// One kernel launch: `Kernel::new(...)`, `arg*` in parameter order, then
/// `run`.  The kernel object is released on drop, on every path.
pub struct Kernel {
    k: usize,
    next: u32,
    queue: usize,
    /// Reduction work-group size the program was compiled with.
    pub wg: usize,
}

/// Serialises kernel-object creation and release.  pocl keeps one
/// reference-counted dlhandle per compiled program hash across contexts,
/// and its count is not thread-safe: concurrent `clCreateKernel` /
/// `clReleaseKernel` from several threads (each with its own context) trips
/// `pocl_release_dlhandle_cache: Assertion found->ref_count > 0` and aborts
/// the process.  Both calls take microseconds, so the lock costs nothing
/// measurable and is harmless on other ICDs.
static KERNEL_OBJECTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

impl Drop for Kernel {
    fn drop(&mut self) {
        if self.k != 0 {
            let _guard = KERNEL_OBJECTS.lock().unwrap_or_else(|p| p.into_inner());
            unsafe { clReleaseKernel(self.k) };
        }
    }
}

impl Kernel {
    pub fn new(ctx: usize, dev: usize, queue: usize, name: &str) -> Result<Self> {
        let (program, wg) = program_for(ctx, dev)?;
        let cname = std::ffi::CString::new(name).map_err(|_| Error::Msg("opencl: kernel name has a NUL".into()))?;
        let mut e: i32 = 0;
        let k = {
            let _guard = KERNEL_OBJECTS.lock().unwrap_or_else(|p| p.into_inner());
            unsafe { clCreateKernel(program, cname.as_ptr() as *const c_void, &mut e) }
        };
        if e != CL_SUCCESS || k == 0 {
            return Err(Error::Msg(format!("opencl clCreateKernel({name}) failed with status {e}")));
        }
        Ok(Self { k, next: 0, queue, wg })
    }

    fn set(&mut self, size: usize, ptr: *const c_void) -> Result<&mut Self> {
        let rc = unsafe { clSetKernelArg(self.k, self.next, size, ptr) };
        if rc != CL_SUCCESS {
            return Err(Error::Msg(format!("opencl clSetKernelArg(index {}) failed with status {rc}", self.next)));
        }
        self.next += 1;
        Ok(self)
    }

    /// A `cl_mem` buffer argument.
    pub fn buf(&mut self, buffer: usize) -> Result<&mut Self> {
        self.set(std::mem::size_of::<usize>(), &buffer as *const usize as *const c_void)
    }

    /// A plain-old-data argument passed by value (`i32`, `f32`, `u64`, [`Idx`], …).
    pub fn val<T: Copy>(&mut self, v: T) -> Result<&mut Self> {
        self.set(std::mem::size_of::<T>(), &v as *const T as *const c_void)
    }

    /// Enqueue with the given global size and (optionally) local size.
    pub fn run(&mut self, global: &[usize], local: Option<&[usize]>) -> Result<()> {
        if global.contains(&0) {
            return Ok(());
        }
        let rc = unsafe {
            clEnqueueNDRangeKernel(
                self.queue,
                self.k,
                global.len() as u32,
                std::ptr::null(),
                global.as_ptr(),
                local.map_or(std::ptr::null(), |l| l.as_ptr()),
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        if rc != CL_SUCCESS {
            return Err(err(rc, "clEnqueueNDRangeKernel"));
        }
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
    /// The device's fault word (see `kernels.cl`): indexing kernels set it
    /// on an out-of-range id; the host reports and clears it at the next
    /// read-back.
    pub fault: usize,
}

impl Ctx {
    pub fn kernel(&self, name: &str) -> Result<Kernel> {
        Kernel::new(self.context, self.device, self.queue, name)
    }
}

// ─── Op codes (must match kernels.cl) ────────────────────────────────────────

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
        (true, true) => return Err(Error::Msg("opencl: u32 condition with 8-byte payload has no kernel".into())),
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
        _ => return Err(Error::Msg(format!("opencl: no copy kernel for {elem}-byte elements"))),
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
        _ => return Err(Error::Msg(format!("opencl: no copy2d kernel for {elem}-byte elements"))),
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
        _ => Err(Error::Msg(format!("opencl: no fill kernel for {elem}-byte elements"))),
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
        (true, true) => return Err(Error::Msg("opencl: i64 ids with 8-byte elements has no kernel".into())),
    };
    let mut k = c.kernel(name)?;
    k.buf(src)?.buf(ids)?.buf(out)?.val(to_i32(n)?)?.val(to_i32(left)?)?.val(to_i32(n_ids)?)?.val(to_i32(right)?)?.val(to_i32(dim_size)?)?.val(to_i32(src_off)?)?.val(to_i32(ids_off)?)?.buf(c.fault)?;
    k.run(&[n], None)
}

#[allow(clippy::too_many_arguments)]
pub fn run_gather(c: &Ctx, src: usize, ids: usize, out: usize, n: usize, ix: Idx, src_dim_stride: usize, dim_size: usize) -> Result<()> {
    let mut k = c.kernel("k_gather_4")?;
    k.buf(src)?.buf(ids)?.buf(out)?.val(to_i32(n)?)?.val(ix)?.val(to_i32(src_dim_stride)?)?.val(to_i32(dim_size)?)?.buf(c.fault)?;
    k.run(&[n], None)
}

/// scatter set / add: `ix` enumerates the ids space with `dim` collapsed
/// (s0 ids, s1 src, s2 dst); the `*_ds` are the strides along `dim`, which
/// the kernel walks in order (`n_j` positions).
#[allow(clippy::too_many_arguments)]
pub fn run_scatter(c: &Ctx, add: bool, dst: usize, ids: usize, src: usize, n: usize, ix: Idx, n_j: usize, ids_ds: usize, src_ds: usize, dst_ds: usize, dim_size: usize) -> Result<()> {
    let mut k = c.kernel(if add { "k_scatter_add_f32" } else { "k_scatter_set_4" })?;
    k.buf(dst)?.buf(ids)?.buf(src)?.val(to_i32(n)?)?.val(ix)?.val(to_i32(n_j)?)?.val(to_i32(ids_ds)?)?.val(to_i32(src_ds)?)?.val(to_i32(dst_ds)?)?.val(to_i32(dim_size)?)?.buf(c.fault)?;
    k.run(&[n], None)
}

#[allow(clippy::too_many_arguments)]
pub fn run_index_add(c: &Ctx, dst: usize, ids: usize, src: usize, left: usize, n_ids: usize, right: usize, dim_size: usize, src_off: usize, ids_off: usize) -> Result<()> {
    let mut k = c.kernel("k_index_add_f32")?;
    let n_lr = left * right;
    k.buf(dst)?.buf(ids)?.buf(src)?.val(to_i32(n_lr)?)?.val(to_i32(left)?)?.val(to_i32(n_ids)?)?.val(to_i32(right)?)?.val(to_i32(dim_size)?)?.val(to_i32(src_off)?)?.val(to_i32(ids_off)?)?.buf(c.fault)?;
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
    if batch == 0 || m == 0 || n == 0 || k == 0 {
        return Ok(());
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
        BF16 => 30,
    }
}

/// Whether the quantized kernels handle `dtype` as blocks (`k_qgemv` /
/// `k_dequant`).  F16/BF16 go through the half kernels and F32 is dense.
pub fn is_block_dtype(dtype: crate::quantized::GgmlDType) -> bool {
    use crate::quantized::GgmlDType::*;
    !matches!(dtype, F32 | F16 | BF16)
}

/// `C[m, n] = sum_k X[m, k] * W[n, k]` over block-quantized `W` (`[N, K]`).
#[allow(clippy::too_many_arguments)]
pub fn run_qgemv(c: &Ctx, dtype: crate::quantized::GgmlDType, x: usize, w: usize, out: usize, m: usize, n: usize, k: usize, woff: u64, xoff: usize) -> Result<()> {
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
        F32 => Err(Error::Msg("opencl: f32 weights need no dequantization".into())),
        _ => {
            let nsub = elem_count / 32;
            let mut kn = c.kernel("k_dequant")?;
            kn.buf(w)?.buf(out)?.val(to_i32(nsub)?)?.val(qtype_code(dtype))?.val(to_i32(dtype.block_size())?)?.val(to_i32(dtype.type_size())?)?.val(woff)?;
            kn.run(&[nsub], None)
        }
    }
}

/// Gather rows of a block-quantized `[vocab, K]` table into f32 `[n_ids, K]`.
#[allow(clippy::too_many_arguments)]
pub fn run_qembed(c: &Ctx, dtype: crate::quantized::GgmlDType, w: usize, ids: usize, out: usize, n_ids: usize, k: usize, vocab: usize, woff: u64, ids_off: usize) -> Result<()> {
    let mut kn = c.kernel("k_qembed")?;
    let wg = kn.wg;
    kn.buf(w)?.buf(ids)?.buf(out)?
        .val(to_i32(n_ids)?)?.val(to_i32(k)?)?
        .val(qtype_code(dtype))?.val(to_i32(dtype.block_size())?)?.val(to_i32(dtype.type_size())?)?
        .val(woff)?.val(to_i32(ids_off)?)?.val(to_i32(vocab)?)?.buf(c.fault)?;
    kn.run(&[n_ids * wg], Some(&[wg]))
}
