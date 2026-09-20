//! Rust bridge for the SYCL backend (Intel oneAPI / DPC++ target, e.g. the
//! Arc Pro B50).
//!
//! Phase 1 (this module): the C ABI bridge loader and kernel launch plumbing.
//! The C++ side lives in `sycl_backend/{bridge.cpp, kernels.hpp, dispatch.inc,
//! CMakeLists.txt}` and builds out-of-tree into `libjoshua_sycl.so` (DPC++
//! `clang++ -fsycl`); the kernels are a 1:1 port of the OpenCL backend's
//! `kernels.cl` (same names, same argument orders — `dispatch.inc` consumes
//! the launch arguments in the order the OpenCL launchers supply them).
//!
//! Phase 2 (follow-up): a full `BackendStorage`/`BackendDevice` integration
//! wired into `Device`, mirroring `opencl_backend`.  Until then the bridge is
//! exercised directly by `tests/sycl_bridge_tests.rs`, which proves the
//! kernels are numerically correct on real SYCL hardware.
//!
//! Library loading: the bridge is `dlopen`ed at runtime (no hard link) with
//! `RTLD_GLOBAL`, and the SYCL runtime (`libsycl.so.9`) is pre-loaded
//! `RTLD_GLOBAL` from the oneAPI toolchain so the bridge's dependency resolves
//! without LD_LIBRARY_PATH.  Override the locations with `JOSHUA_SYCL_LIBRARY`
//! (bridge) and `JOSHUA_SYCL_RUNTIME` (libsycl.so.9).
#![cfg(all(feature = "sycl", target_os = "linux"))]

use std::ffi::{c_char, c_void, CString};
use std::path::PathBuf;
use std::sync::OnceLock;

const WG: usize = 64; // matches kernels.hpp `constexpr int WG`

#[repr(C)]
struct SyclArg {
    data: *const u8,
    size: usize,
}

/// C ABI function pointers, resolved from the dlopened bridge at startup.
/// The bridge is loaded with `RTLD_GLOBAL` but the extern "C" declarations
/// would still be undefined symbols at Rust link time — resolve them via
/// libloading like every other runtime-loaded backend (vulkan, npu shim).
#[derive(Clone, Copy)]
pub struct SyclFns {
    pub error: unsafe extern "C" fn() -> *const c_char,
    pub open: unsafe extern "C" fn(ordinal: usize, out: *mut usize) -> i32,
    pub close: unsafe extern "C" fn(h: usize) -> i32,
    pub info: unsafe extern "C" fn(h: usize, name: *mut u8, len: usize, memory: *mut u64) -> i32,
    pub alloc: unsafe extern "C" fn(h: usize, size: usize, out: *mut usize) -> i32,
    pub free: unsafe extern "C" fn(h: usize) -> i32,
    pub finish: unsafe extern "C" fn(h: usize) -> i32,
    pub write: unsafe extern "C" fn(h: usize, dst: usize, off: usize, size: usize, src: *const u8) -> i32,
    pub read: unsafe extern "C" fn(h: usize, src: usize, off: usize, size: usize, dst: *mut u8) -> i32,
    pub copy: unsafe extern "C" fn(h: usize, src: usize, dst: usize, so: usize, d: usize, size: usize) -> i32,
    pub launch: unsafe extern "C" fn(h: usize, name: *const c_char, args: *const SyclArg, count: usize, global: *const usize, local: *const usize) -> i32,
}

unsafe fn resolve(lib: &libloading::os::unix::Library) -> Result<SyclFns, String> {
    macro_rules! sym {
        ($lib:expr, $name:literal) => {
            *$lib.get::<unsafe extern "C" fn($($unused)*) -> ()>(b"joshua_sycl_dummy").unwrap_or_else(|_| unreachable!())
        };
    }
    Ok(SyclFns {
        error: *lib.get(b"joshua_sycl_error\0").map_err(|e| format!("joshua_sycl_error: {e}"))?,
        open: *lib.get(b"joshua_sycl_open\0").map_err(|e| format!("joshua_sycl_open: {e}"))?,
        close: *lib.get(b"joshua_sycl_close\0").map_err(|e| format!("joshua_sycl_close: {e}"))?,
        info: *lib.get(b"joshua_sycl_info\0").map_err(|e| format!("joshua_sycl_info: {e}"))?,
        alloc: *lib.get(b"joshua_sycl_alloc\0").map_err(|e| format!("joshua_sycl_alloc: {e}"))?,
        free: *lib.get(b"joshua_sycl_free\0").map_err(|e| format!("joshua_sycl_free: {e}"))?,
        finish: *lib.get(b"joshua_sycl_finish\0").map_err(|e| format!("joshua_sycl_finish: {e}"))?,
        write: *lib.get(b"joshua_sycl_write\0").map_err(|e| format!("joshua_sycl_write: {e}"))?,
        read: *lib.get(b"joshua_sycl_read\0").map_err(|e| format!("joshua_sycl_read: {e}"))?,
        copy: *lib.get(b"joshua_sycl_copy\0").map_err(|e| format!("joshua_sycl_copy: {e}"))?,
        launch: *lib.get(b"joshua_sycl_launch\0").map_err(|e| format!("joshua_sycl_launch: {e}"))?,
    })
}

unsafe fn last_error(fns: &SyclFns) -> String {
    let p = (fns.error)();
    if p.is_null() {
        "unknown SYCL error".to_string()
    } else {
        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

unsafe fn check(rc: i32, fns: &SyclFns) -> crate::Result<()> {
    if rc == 0 {
        return Ok(());
    }
    crate::bail!("sycl: {}", last_error(fns))
}

/// The dlopened bridge and SYCL runtime handles, intentionally leaked for the
/// life of the process (the bridge's Context handles stay valid regardless).
struct Bridge {
    _runtime: Option<libloading::os::unix::Library>,
    _bridge: libloading::os::unix::Library,
    fns: SyclFns,
}
// Libraries are safe to share across threads; handles are used behind the
// bridge's own internal mutex.
unsafe impl Send for Bridge {}
unsafe impl Sync for Bridge {}

unsafe fn dlopen_global(path: &std::path::Path) -> Result<libloading::os::unix::Library, String> {
    libloading::os::unix::Library::open(
        Some(path),
        libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL,
    )
    .map_err(|e| format!("{}: {e}", path.display()))
}

fn sycl_runtime_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(p) = std::env::var("JOSHUA_SYCL_RUNTIME") {
        candidates.push(PathBuf::from(p));
    }
    if let Ok(home) = std::env::var("HOME") {
        // The oneAPI toolchain layout used on the SYCL boxes.
        if let Ok(entries) = std::fs::read_dir(std::path::Path::new(&home)) {
            let mut roots: Vec<PathBuf> = entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with("sycl-toolchain-"))
                        .unwrap_or(false)
                })
                .collect();
            roots.sort();
            if let Some(dir) = roots.pop() {
                candidates.push(dir.join("lib").join("libsycl.so.9"));
            }
        }
        candidates.push(PathBuf::from(&home).join("sycl-toolchain-20260918/lib/libsycl.so.9"));
    }
    candidates
}

fn bridge_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(p) = std::env::var("JOSHUA_SYCL_LIBRARY") {
        candidates.push(PathBuf::from(p));
    }
    if let Ok(home) = std::env::var("HOME") {
        candidates.push(PathBuf::from(&home).join("joshua-sycl/libjoshua_sycl.so"));
    }
    candidates
}

unsafe fn dlopen_bridge() -> Result<Bridge, String> {
    let mut runtime = None;
    for path in sycl_runtime_candidates() {
        if path.exists() {
            if let Ok(lib) = dlopen_global(&path) {
                runtime = Some(lib);
                break;
            }
        }
    }
    let mut last = "no candidate path exists".to_string();
    for path in bridge_candidates() {
        if !path.exists() {
            last = format!("{}: not found", path.display());
            continue;
        }
        match dlopen_global(&path) {
            Ok(lib) => {
                let fns = resolve(&lib).map_err(|e| format!("symbol resolution: {e}"))?;
                return Ok(Bridge { _runtime: runtime, _bridge: lib, fns });
            }
            Err(e) => last = e,
        }
    }
    Err(last)
}

fn bridge() -> crate::Result<&'static Bridge> {
    static BRIDGE: OnceLock<Result<&'static Bridge, String>> = OnceLock::new();
    let bridge = BRIDGE.get_or_init(|| unsafe { dlopen_bridge().map(|b| &*Box::leak(Box::new(b))) });
    bridge.clone().map_err(|e| crate::Error::Msg(format!("sycl: {e}")))
}

/// An opened SYCL device: one bridge context handle.
pub struct SyclDevice {
    handle: usize,
    name: String,
    memory: u64,
    fns: SyclFns,
}

impl SyclDevice {
    /// Open SYCL device `ordinal` (the bridge prefers GPUs when present).
    pub fn new(ordinal: usize) -> crate::Result<Self> {
        let b = bridge()?;
        let fns = b.fns;
        unsafe {
            let mut h: usize = 0;
            check((fns.open)(ordinal, &mut h), &fns)?;
            let mut name_buf = [0u8; 128];
            let mut memory: u64 = 0;
            check((fns.info)(h, name_buf.as_mut_ptr(), name_buf.len(), &mut memory), &fns)?;
            let end = name_buf.iter().position(|&b| b == 0).unwrap_or(name_buf.len());
            Ok(Self {
                handle: h,
                name: String::from_utf8_lossy(&name_buf[..end]).into_owned(),
                memory,
                fns,
            })
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn memory(&self) -> u64 {
        self.memory
    }

    /// Allocate `bytes` on the device; returns the buffer handle used by the
    /// launchers and the read/write helpers.
    pub fn alloc(&self, bytes: usize) -> crate::Result<usize> {
        unsafe {
            let mut h: usize = 0;
            check((self.fns.alloc)(self.handle, bytes, &mut h), &self.fns)?;
            Ok(h)
        }
    }

    pub fn free(&self, buffer: usize) -> crate::Result<()> {
        unsafe { check((self.fns.free)(buffer), &self.fns) }
    }

    pub fn write(&self, buffer: usize, off: usize, bytes: &[u8]) -> crate::Result<()> {
        unsafe { check((self.fns.write)(self.handle, buffer, off, bytes.len(), bytes.as_ptr()), &self.fns) }
    }

    pub fn read(&self, buffer: usize, off: usize, out: &mut [u8]) -> crate::Result<()> {
        unsafe { check((self.fns.read)(self.handle, buffer, off, out.len(), out.as_mut_ptr()), &self.fns) }
    }

    pub fn copy(&self, src: usize, dst: usize, so: usize, d: usize, size: usize) -> crate::Result<()> {
        unsafe { check((self.fns.copy)(self.handle, src, dst, so, d, size), &self.fns) }
    }

    /// Block until all queued work completes.
    pub fn finish(&self) -> crate::Result<()> {
        unsafe { check((self.fns.finish)(self.handle), &self.fns) }
    }

    fn launch(
        &self,
        kernel: &str,
        b: &mut ArgBuilder,
        global: [usize; 3],
        local: [usize; 3],
    ) -> crate::Result<()> {
        let name = CString::new(kernel).unwrap();
        let mut g = global;
        // The bridge requires global % local == 0 on every dimension; round
        // the 1-D global up so padded launches never fail the check.
        if local[1] == 1 && local[2] == 1 && local[0] > 1 {
            g[0] = g[0].div_ceil(local[0]) * local[0];
        }
        unsafe {
            check((self.fns.launch)(
                self.handle,
                name.as_ptr(),
                b.args.as_ptr(),
                b.args.len(),
                g.as_ptr(),
                local.as_ptr(),
            ), &self.fns)
        }
    }

    // ── Kernel launchers (arg order mirrors dispatch.inc / kernels.hpp) ──

    /// `o = x * mul + add` over `n` elements (kernel `k_affine_c`).
    pub fn run_affine_c(&self, x: usize, o: usize, n: usize, off: usize, mul: f32, add: f32) -> crate::Result<()> {
        let mut b = ArgBuilder::new();
        b.buf(x).buf(o).i32(n as i32).i32(off as i32).f32(mul).f32(add);
        self.launch("k_affine_c", &mut b, [n, 1, 1], [WG, 1, 1])
    }

    /// Unary op `op` over `n` elements (kernel `k_unary_c`; op codes match the
    /// OpenCL backend: 0=exp, 1=log, 2=sin, 3=cos, ...).
    pub fn run_unary_c(&self, x: usize, o: usize, n: usize, off: usize, op: i32) -> crate::Result<()> {
        let mut b = ArgBuilder::new();
        b.buf(x).buf(o).i32(n as i32).i32(off as i32).i32(op);
        self.launch("k_unary_c", &mut b, [n, 1, 1], [WG, 1, 1])
    }

    /// `o = a op b` elementwise (kernel `k_binary_c`; op codes match the
    /// OpenCL backend: 0=add, 1=sub, 2=mul, 3=div).
    #[allow(clippy::too_many_arguments)]
    pub fn run_binary_c(&self, a: usize, b: usize, o: usize, n: usize, off_a: usize, off_b: usize, op: i32) -> crate::Result<()> {
        let mut builder = ArgBuilder::new();
        builder.buf(a).buf(b).buf(o).i32(n as i32).i32(off_a as i32).i32(off_b as i32).i32(op);
        self.launch("k_binary_c", &mut builder, [n, 1, 1], [WG, 1, 1])
    }

    /// Row softmax (kernel `k_softmax_last`): one work-group of WG lanes per
    /// row, `[rows, cols]` contiguous input at `off`.
    pub fn run_softmax_last(&self, x: usize, o: usize, rows: usize, cols: usize, off: usize) -> crate::Result<()> {
        let mut b = ArgBuilder::new();
        b.buf(x).buf(o).i32(rows as i32).i32(cols as i32).i32(off as i32);
        self.launch("k_softmax_last", &mut b, [WG * rows, 1, 1], [WG, 1, 1])
    }

    /// Row RMSNorm (kernel `k_rmsnorm`).
    #[allow(clippy::too_many_arguments)]
    pub fn run_rmsnorm(&self, x: usize, alpha: usize, o: usize, rows: usize, cols: usize, off: usize, aoff: usize, eps: f32) -> crate::Result<()> {
        let mut b = ArgBuilder::new();
        b.buf(x).buf(alpha).buf(o).i32(rows as i32).i32(cols as i32).i32(off as i32).i32(aoff as i32).f32(eps);
        self.launch("k_rmsnorm", &mut b, [WG * rows, 1, 1], [WG, 1, 1])
    }

    /// Tiled GEMM (kernel `k_gemm`): `C[m, n] = Σ_k A[m, k] * B[k, n]` with
    /// arbitrary strides and batch strides (16×16 tiles, 256 lanes).
    #[allow(clippy::too_many_arguments)]
    pub fn run_gemm(
        &self,
        a: usize,
        b: usize,
        c: usize,
        m: usize,
        n: usize,
        k: usize,
        sam: usize,
        sak: usize,
        sbk: usize,
        sbn: usize,
        oa: usize,
        ob: usize,
        oc: usize,
        ba: usize,
        bb: usize,
        bc: usize,
        b_kc: i32,
    ) -> crate::Result<()> {
        let mut builder = ArgBuilder::new();
        builder.buf(a).buf(b).buf(c)
            .i32(m as i32).i32(n as i32).i32(k as i32)
            .i32(sam as i32).i32(sak as i32).i32(sbk as i32).i32(sbn as i32)
            .i32(oa as i32).i32(ob as i32).i32(oc as i32)
            .i32(ba as i32).i32(bb as i32).i32(bc as i32).i32(b_kc);
        let batch = ba.max(bb).max(bc);
        let gx = n.div_ceil(16) * 16;
        let gy = m.div_ceil(16) * 16;
        self.launch("k_gemm", &mut builder, [gx, gy, batch], [16, 16, 1])
    }

    /// GEMV with transposed weight rows (kernel `k_gemv_nt`):
    /// `C[bz, n] = Σ_k A[bz, k] * B[n, k]`, one work-group per output.
    #[allow(clippy::too_many_arguments)]
    pub fn run_gemv_nt(&self, a: usize, b: usize, c: usize, n: usize, k: usize, sbn: usize, oa: usize, ob: usize, oc: usize, ba: usize, bb: usize, bc: usize) -> crate::Result<()> {
        let mut builder = ArgBuilder::new();
        builder.buf(a).buf(b).buf(c)
            .i32(n as i32).i32(k as i32).i32(sbn as i32)
            .i32(oa as i32).i32(ob as i32).i32(oc as i32)
            .i32(ba as i32).i32(bb as i32).i32(bc as i32);
        let batch = ba.max(bb).max(bc);
        self.launch("k_gemv_nt", &mut builder, [n * WG, batch, 1], [WG, 1, 1])
    }

    /// Dequantize-in-kernel quantized GEMV (kernel `k_qgemv`), the
    /// single-token decode path: `C[m, n] = Σ_k X[m, k] * dequant(W[n, k])`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_qgemv(&self, x: usize, w: usize, c: usize, n: usize, k: usize, qt: i32, qk: i32, bsz: i32, woff: u64, xoff: i32, coff: i32, m: usize) -> crate::Result<()> {
        let mut builder = ArgBuilder::new();
        builder.buf(x).buf(w).buf(c)
            .i32(n as i32).i32(k as i32).i32(qt).i32(qk).i32(bsz)
            .u64(woff).i32(xoff).i32(coff).i32(m as i32);
        self.launch("k_qgemv", &mut builder, [n * WG, m, 1], [WG, 1, 1])
    }

    /// Dequantize a block-quantized tensor to f32 (kernel `k_dequant`).
    pub fn run_dequant(&self, w: usize, o: usize, nsub: usize, qt: i32, qk: i32, bsz: i32, woff: u64) -> crate::Result<()> {
        let mut b = ArgBuilder::new();
        b.buf(w).buf(o).i32(nsub as i32).i32(qt).i32(qk).i32(bsz).u64(woff);
        self.launch("k_dequant", &mut b, [nsub, 1, 1], [WG, 1, 1])
    }

    /// F16/BF16 embedding gather (kernel `k_hembed`).
    #[allow(clippy::too_many_arguments)]
    pub fn run_hembed(&self, w: usize, ids: usize, o: usize, n: usize, k: usize, bf16: bool, woff: u64, ids_off: usize, vocab: usize) -> crate::Result<()> {
        let mut b = ArgBuilder::new();
        b.buf(w).buf(ids).buf(o).i32(n as i32).i32(k as i32)
            .i32(bf16 as i32).u64(woff).i32(ids_off as i32).i32(vocab as i32);
        self.launch("k_hembed", &mut b, [n, 1, 1], [WG, 1, 1])
    }
}

/// Launch-argument builder: an arena of 8-byte-aligned slots plus the
/// `SyclArg` descriptors.  `Arg.size` must match the C++ `sizeof(T)` for the
/// slot (8 for pointer/u64, 4 for int/float), which the bridge checks.
pub struct ArgBuilder {
    arena: Vec<u8>,
    args: Vec<SyclArg>,
}

impl ArgBuilder {
    fn new() -> Self {
        Self { arena: Vec::new(), args: Vec::new() }
    }

    fn push(&mut self, bytes: &[u8]) {
        while self.arena.len() % 8 != 0 {
            self.arena.push(0);
        }
        let off = self.arena.len();
        self.arena.extend_from_slice(bytes);
        self.args.push(SyclArg {
            data: unsafe { self.arena.as_ptr().add(off) },
            size: bytes.len(),
        });
    }

    /// A kernel pointer argument: carries the buffer handle; the bridge
    /// resolves it to the device pointer.
    pub fn buf(&mut self, handle: usize) -> &mut Self {
        self.push(&handle.to_ne_bytes());
        self
    }

    pub fn i32(&mut self, v: i32) -> &mut Self {
        self.push(&v.to_ne_bytes());
        self
    }

    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.push(&v.to_ne_bytes());
        self
    }

    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.push(&v.to_ne_bytes());
        self
    }

    pub fn f32(&mut self, v: f32) -> &mut Self {
        self.push(&v.to_ne_bytes());
        self
    }
}



fn batch_max(ba: usize, bb: usize, bc: usize) -> usize {
    ba.max(bb).max(bc)
}

impl Drop for SyclDevice {
    fn drop(&mut self) {
        if self.handle != 0 {
            unsafe { (self.fns.close)(self.handle) };
        }
    }
}

unsafe impl Send for SyclDevice {}
unsafe impl Sync for SyclDevice {}
