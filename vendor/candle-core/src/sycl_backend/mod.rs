//! Rust bridge + full candle backend for SYCL (Intel oneAPI / DPC++ target,
//! e.g. the Arc Pro B50).
//!
//! Phase 2: a full `BackendDevice`/`BackendStorage` implementation mirroring
//! `opencl_backend/mod.rs` — the same native op set (all 68 kernels are a 1:1
//! port of the OpenCL ones, and the launchers in `kernels.rs` mirror the
//! OpenCL launcher argument orders), the same CPU-fallback contract for
//! anything a launcher refuses, and the same block-quantized device weights
//! (`QSyclStorage`, `fwd` / `embedding`).
//!
//! Library loading: the bridge is `dlopen`ed at runtime (no hard link) with
//! `RTLD_GLOBAL`, and the SYCL runtime (`libsycl.so.9`) is pre-loaded
//! `RTLD_GLOBAL` from the oneAPI toolchain so the bridge's dependency
//! resolves without LD_LIBRARY_PATH.  Override the locations with
//! `JOSHUA_SYCL_LIBRARY` (bridge) and `JOSHUA_SYCL_RUNTIME` (libsycl.so.9).
#![cfg(all(feature = "sycl", target_os = "linux"))]

use std::ffi::{c_char, CString};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use crate::backend::{BackendDevice, BackendStorage};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape, WithDType};

pub mod kernels;

use kernels::{Idx, MatStrides};

const WG: usize = 64; // matches kernels.hpp `constexpr int WG`

#[repr(C)]
pub struct SyclArg {
    data: *const u8,
    size: usize,
}

/// C ABI function pointers, resolved from the dlopened bridge at startup
/// (the same pattern as every runtime-loaded backend).
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

unsafe fn resolve(lib: &libloading::os::unix::Library) -> std::result::Result<SyclFns, String> {
    unsafe fn get<T: Copy>(lib: &libloading::os::unix::Library, name: &[u8]) -> std::result::Result<T, String> {
        let sym = lib
            .get::<T>(name)
            .map_err(|e| format!("{}: {e}", std::str::from_utf8(&name[..name.len() - 1]).unwrap_or("?")))?;
        Ok(*sym)
    }
    Ok(SyclFns {
        error: get(lib, b"joshua_sycl_error\0")?,
        open: get(lib, b"joshua_sycl_open\0")?,
        close: get(lib, b"joshua_sycl_close\0")?,
        info: get(lib, b"joshua_sycl_info\0")?,
        alloc: get(lib, b"joshua_sycl_alloc\0")?,
        free: get(lib, b"joshua_sycl_free\0")?,
        finish: get(lib, b"joshua_sycl_finish\0")?,
        write: get(lib, b"joshua_sycl_write\0")?,
        read: get(lib, b"joshua_sycl_read\0")?,
        copy: get(lib, b"joshua_sycl_copy\0")?,
        launch: get(lib, b"joshua_sycl_launch\0")?,
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

unsafe fn check(rc: i32, fns: &SyclFns) -> Result<()> {
    if rc == 0 {
        return Ok(());
    }
    crate::bail!("sycl: {}", last_error(fns))
}

/// Serializes ALL bridge operations: the DPC++ runtime is not thread-safe
/// for concurrent kernel submissions from multiple threads.
static GLOBAL_LOCK: Mutex<()> = Mutex::new(());

struct Bridge {
    _runtime: Option<libloading::os::unix::Library>,
    _bridge: libloading::os::unix::Library,
    fns: SyclFns,
}

unsafe impl Send for Bridge {}
unsafe impl Sync for Bridge {}

unsafe fn dlopen_global(path: &std::path::Path) -> std::result::Result<libloading::os::unix::Library, String> {
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
    }
    candidates
}

fn bridge_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(p) = std::env::var("JOSHUA_SYCL_LIBRARY") {
        candidates.push(PathBuf::from(p));
    }
    if let Ok(home) = std::env::var("HOME") {
        candidates.push(PathBuf::from(&home).join("joshua-sycl/build/libjoshua_sycl.so"));
        candidates.push(PathBuf::from(&home).join("joshua-sycl/libjoshua_sycl.so"));
    }
    candidates
}

unsafe fn dlopen_bridge() -> Result<&'static Bridge> {
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
                let bridge = Bridge { _runtime: runtime, _bridge: lib, fns };
                return Ok(&*Box::leak(Box::new(bridge)));
            }
            Err(e) => last = e,
        }
    }
    Err(Error::Msg(format!("sycl: {last}")))
}

fn bridge() -> Result<&'static Bridge> {
    static BRIDGE: OnceLock<std::result::Result<&'static Bridge, String>> = OnceLock::new();
    let _init = GLOBAL_LOCK.lock().unwrap();
    let b = BRIDGE.get_or_init(|| unsafe { dlopen_bridge().map_err(|e| e.to_string()) });
    b.clone().map_err(|e| Error::Msg(format!("sycl: {e}")))
}

/// Launch-argument builder: an arena of 8-byte-aligned slots plus the
/// `SyclArg` descriptors, materialised once after the arena is final (taking
/// `arena.as_ptr()` during the pushes would dangle on reallocation).
pub struct ArgBuilder {
    arena: Vec<u8>,
    layout: Vec<(usize, usize)>,
}

impl ArgBuilder {
    pub fn new() -> Self {
        Self { arena: Vec::new(), layout: Vec::new() }
    }

    fn push(&mut self, bytes: &[u8]) {
        while self.arena.len() % 8 != 0 {
            self.arena.push(0);
        }
        let off = self.arena.len();
        self.arena.extend_from_slice(bytes);
        self.layout.push((off, bytes.len()));
    }

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

    /// A stride-descriptor argument (the `Idx` struct, 144 bytes, `repr(C)`).
    pub fn idx(&mut self, ix: Idx) -> &mut Self {
        // # Safety: the reference points at `size_of::<Idx>()` plain bytes.
        self.push(unsafe {
            std::slice::from_raw_parts(&ix as *const Idx as *const u8, std::mem::size_of::<Idx>())
        });
        self
    }

    pub fn push_raw(&mut self, bytes: &[u8]) -> &mut Self {
        self.push(bytes);
        self
    }

    fn args(&self) -> Vec<SyclArg> {
        self.layout
            .iter()
            .map(|&(off, size)| SyclArg {
                data: unsafe { self.arena.as_ptr().add(off) },
                size,
            })
            .collect()
    }
}

/// An opened SYCL device: one bridge context handle.
#[derive(Debug)]
pub struct SyclDevice {
    bridge: &'static Bridge,
    handle: usize,
    name: String,
    memory: u64,
    /// The kernels' shared index-fault buffer (u32 per slot; a gather or
    /// embedding sets its thread's slot on an out-of-range id).
    fault: usize,
    /// Reusable scratch for the quantized prefill path (dequantize-then-GEMM
    /// must not queue one fresh n*k*4-byte buffer per matmul).
    scratch: Mutex<Option<(usize, Option<Box<SyclStorage>>)>>,
}

impl SyclDevice {
    /// Open SYCL device `ordinal` (the bridge prefers GPUs when present).
    pub fn new(ordinal: usize) -> Result<Self> {
        let b = bridge()?;
        let fns = b.fns;
        static DEV_LOCK: Mutex<()> = Mutex::new(());
        let _dev = DEV_LOCK.lock().unwrap();
        unsafe {
            let mut h: usize = 0;
            check((fns.open)(ordinal, &mut h), &fns)?;
            let mut name_buf = [0u8; 128];
            let mut memory: u64 = 0;
            check((fns.info)(h, name_buf.as_mut_ptr(), name_buf.len(), &mut memory), &fns)?;
            let end = name_buf.iter().position(|&x| x == 0).unwrap_or(name_buf.len());
            Ok(Self {
                bridge: b,
                handle: h,
                name: String::from_utf8_lossy(&name_buf[..end]).into_owned(),
                memory,
                fault: 0,
                scratch: Mutex::new(None),
            })
        }
    }

    /// Allocate the fault-checker buffer (once, after construction).
    fn init_fault(&mut self) -> Result<()> {
        let storage = self.alloc(DType::U32, 1024)?;
        self.fault = storage.buffer;
        Ok(())
    }

    fn alloc_bytes(&self, bytes: usize) -> Result<usize> {
        let fns = self.bridge.fns;
        unsafe {
            let mut h: usize = 0;
            check((fns.alloc)(self.handle, bytes, &mut h), &fns)?;
            Ok(h)
        }
    }

    /// Allocate `numel` elements of `dtype` on the device.
    pub fn alloc(&self, dtype: DType, numel: usize) -> Result<SyclStorage> {
        let bytes = numel
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| Error::Msg("sycl: overflow in storage size".into()))?;
        let buffer = self.alloc_bytes(bytes.max(1))?;
        Ok(SyclStorage { buffer, dtype, numel, device: self.clone() })
    }

    /// Allocate raw bytes (quantized blocks, etc.).
    pub fn alloc_raw(&self, bytes: usize) -> Result<usize> {
        self.alloc_bytes(bytes.max(1))
    }

    pub fn write_bytes(&self, buffer: usize, off: usize, bytes: &[u8]) -> Result<()> {
        let fns = self.bridge.fns;
        unsafe { check((fns.write)(self.handle, buffer, off, bytes.len(), bytes.as_ptr()), &fns) }
    }

    pub fn read_bytes(&self, buffer: usize, off: usize, out: &mut [u8]) -> Result<()> {
        let fns = self.bridge.fns;
        unsafe { check((fns.read)(self.handle, buffer, off, out.len(), out.as_mut_ptr()), &fns) }
    }

    pub fn copy(&self, src: usize, dst: usize, so: usize, d: usize, size: usize) -> Result<()> {
        let fns = self.bridge.fns;
        unsafe { check((fns.copy)(self.handle, src, dst, so, d, size), &fns) }
    }

    /// Launch `kernel` with marshalled arguments (global/local in the OpenCL
    /// x/y/z convention; the bridge reverses for SYCL).
    pub fn launch(&self, kernel: &str, b: &mut ArgBuilder, global: [usize; 3], local: [usize; 3]) -> Result<()> {
        let name = CString::new(kernel).unwrap();
        let mut g = global;
        if local[1] == 1 && local[2] == 1 && local[0] > 1 {
            g[0] = g[0].div_ceil(local[0]) * local[0];
        }
        let args = b.args();
        let _lock = GLOBAL_LOCK.lock().unwrap();
        unsafe {
            check(
                (self.bridge.fns.launch)(self.handle, name.as_ptr(), args.as_ptr(), args.len(), g.as_ptr(), local.as_ptr()),
                &self.bridge.fns,
            )
        }
    }

    /// A reusable device buffer of at least `bytes` (the quantized prefill's
    /// dequantize scratch).
    pub fn scratch(&self, bytes: usize) -> Result<SyclStorage> {
        let mut guard = self.scratch.lock().unwrap();
        if let Some((cap, Some(st))) = guard.as_mut() {
            if *cap >= bytes {
                return Ok(SyclStorage {
                    buffer: st.buffer,
                    dtype: DType::F32,
                    numel: bytes / 4,
                    device: self.clone(),
                });
            }
        }
        let st = self.alloc(DType::F32, bytes.div_ceil(4))?;
        *guard = Some((bytes, Some(Box::new(st.clone()))));
        Ok(st)
    }

    /// Check (and clear) this thread's index-fault slot.
    pub fn check_fault(&self) -> Result<()> {
        if self.fault == 0 {
            return Ok(());
        }
        let slot = crate::fault_slot::current();
        let mut buf = [0u8; 4];
        self.read_bytes(self.fault, slot * 4, &mut buf)?;
        let v = u32::from_ne_bytes(buf);
        if v != 0 {
            self.write_bytes(self.fault, slot * 4, &0u32.to_ne_bytes())?;
            crate::bail!("sycl: an index kernel faulted on an out-of-range id")
        }
        Ok(())
    }

    /// Block until all queued work completes.
    pub fn finish(&self) -> Result<()> {
        let fns = self.bridge.fns;
        let _lock = GLOBAL_LOCK.lock().unwrap();
        unsafe { check((fns.finish)(self.handle), &fns) }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn global_mem_size(&self) -> Option<u64> {
        Some(self.memory)
    }
}

impl Drop for SyclDevice {
    fn drop(&mut self) {
        if self.handle != 0 {
            let fns = self.bridge.fns;
            unsafe { (fns.close)(self.handle) };
        }
    }
}

unsafe impl Send for SyclDevice {}
unsafe impl Sync for SyclDevice {}

impl Clone for SyclDevice {
    fn clone(&self) -> Self {
        Self {
            bridge: self.bridge,
            handle: self.handle,
            name: self.name.clone(),
            memory: self.memory,
            fault: self.fault,
            scratch: Mutex::new(None),
        }
    }
}

/// Storage on the SYCL device.
#[derive(Debug)]
pub struct SyclStorage {
    pub buffer: usize,
    pub dtype: DType,
    pub numel: usize,
    pub device: SyclDevice,
}

impl Drop for SyclStorage {
    fn drop(&mut self) {
        if self.buffer != 0 {
            let fns = self.device.bridge.fns;
            unsafe { (fns.free)(self.buffer) };
        }
    }
}

fn transmute_bytes<T: Copy>(raw: &[u8], numel: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(numel);
    let size = std::mem::size_of::<T>();
    for i in 0..numel {
        // # Safety: raw holds numel * size bytes, T is plain data.
        let v = unsafe { std::ptr::read_unaligned(raw.as_ptr().add(i * size) as *const T) };
        out.push(v);
    }
    out
}

impl SyclStorage {
    pub fn from_vec<T: WithDType>(slice: Vec<T>, device: &SyclDevice) -> Result<Self> {
        let dtype = T::DTYPE;
        let bytes = dtype
            .size_in_bytes()
            .checked_mul(slice.len())
            .ok_or_else(|| Error::Msg("sycl: overflow in storage size".into()))?;
        let buffer = device.alloc_bytes(bytes.max(1))?;
        if bytes > 0 {
            // # Safety: slice.as_ptr() points to `bytes` valid bytes.
            let data = unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, bytes) };
            if let Err(e) = device.write_bytes(buffer, 0, data) {
                let fns = device.bridge.fns;
                unsafe { (fns.free)(buffer) };
                return Err(e);
            }
        }
        Ok(Self { buffer, dtype, numel: slice.len(), device: device.clone() })
    }

    fn elem_size(&self) -> usize {
        self.dtype.size_in_bytes()
    }

    /// A contiguous copy of the elements addressed by `l`, on the device.
    pub fn contiguous_copy(&self, l: &Layout) -> Result<Self> {
        let n = l.shape().elem_count();
        let out = self.device.alloc(self.dtype, n)?;
        kernels::run_copy_strided(&self.device, self.elem_size(), self.buffer, out.buffer, n, l, 0)?;
        Ok(out)
    }

    /// `(storage, offset)`: the storage itself when `l` is contiguous, else
    /// a contiguous copy at offset 0.
    pub fn as_contiguous(&self, l: &Layout) -> Result<(std::borrow::Cow<'_, Self>, usize)> {
        match l.contiguous_offsets() {
            Some((o1, _)) => Ok((std::borrow::Cow::Borrowed(self), o1)),
            None => Ok((std::borrow::Cow::Owned(self.contiguous_copy(l)?), 0)),
        }
    }

    /// u32 ids from a u32 / i64 / u8 id storage (cast on the device when
    /// needed).
    pub fn ids_u32(&self, l: &Layout) -> Result<(std::borrow::Cow<'_, Self>, usize)> {
        match self.dtype {
            DType::U32 => self.as_contiguous(l),
            DType::I64 | DType::U8 => {
                let n = l.shape().elem_count();
                let out = self.device.alloc(DType::U32, n)?;
                let name = if self.dtype == DType::I64 {
                    "k_ids_i64"
                } else {
                    kernels::cast_kernel(self.dtype, DType::U32)
                        .ok_or_else(|| Error::Msg("sycl: no id cast".into()))?
                };
                kernels::run_cast(&self.device, name, self.buffer, out.buffer, n, l)?;
                Ok((std::borrow::Cow::Owned(out), 0))
            }
            d => Err(Error::Msg(format!("sycl: unsupported index dtype {d:?}"))),
        }
    }
}

impl Clone for SyclStorage {
    fn clone(&self) -> Self {
        self.try_clone(&Layout::contiguous(self.numel))
            .expect("sycl: device buffer copy failed")
    }
}

impl BackendStorage for SyclStorage {
    type Device = SyclDevice;

    fn try_clone(&self, _layout: &Layout) -> Result<Self> {
        let bytes = self.numel * self.elem_size();
        let buffer = self.device.alloc_bytes(bytes.max(1))?;
        if bytes > 0 {
            self.device.copy(self.buffer, buffer, 0, 0, bytes)?;
        }
        Ok(Self { buffer, dtype: self.dtype, numel: self.numel, device: self.device.clone() })
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn const_set(&mut self, s: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        if let Some(bits) = scalar_bits(s) {
            let n = layout.shape().elem_count();
            return kernels::run_fill(&self.device, self.elem_size(), self.buffer, n, layout, bits);
        }
        let mut cpu = self.to_cpu_storage()?;
        cpu.const_set(s, layout)?;
        let dev = self.device.clone();
        *self = dev.storage_from_cpu_storage(&cpu)?;
        Ok(())
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        let bytes = self.numel * self.elem_size();
        let mut raw = vec![0u8; bytes];
        self.device.read_bytes(self.buffer, 0, &mut raw)?;
        self.device.check_fault()?;
        Ok(match self.dtype {
            DType::U8 => CpuStorage::U8(raw),
            DType::U32 => CpuStorage::U32(transmute_bytes(&raw, self.numel)),
            DType::I16 => CpuStorage::I16(transmute_bytes(&raw, self.numel)),
            DType::I32 => CpuStorage::I32(transmute_bytes(&raw, self.numel)),
            DType::I64 => CpuStorage::I64(transmute_bytes(&raw, self.numel)),
            DType::F32 => CpuStorage::F32(transmute_bytes(&raw, self.numel)),
            DType::F64 => CpuStorage::F64(transmute_bytes(&raw, self.numel)),
            DType::F16 => CpuStorage::F16(
                transmute_bytes::<u16>(&raw, self.numel).into_iter().map(half::f16::from_bits).collect(),
            ),
            DType::BF16 => CpuStorage::BF16(
                transmute_bytes::<u16>(&raw, self.numel).into_iter().map(half::bf16::from_bits).collect(),
            ),
            other => return Err(Error::Msg(format!("sycl to_cpu_storage: dtype {other:?} not supported"))),
        })
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        if self.dtype == DType::F32 {
            let n = layout.shape().elem_count();
            let out = self.device.alloc(DType::F32, n)?;
            if let Ok(()) = kernels::run_affine(&self.device, self.buffer, out.buffer, n, layout, mul as f32, add as f32) {
                return Ok(out);
            }
        }
        let cpu = self.to_cpu_storage()?;
        let out = cpu.affine(layout, mul, add)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        if self.dtype == DType::F32 {
            let n = layout.shape().elem_count();
            let out = self.device.alloc(DType::F32, n)?;
            if let Ok(()) = kernels::run_powf(&self.device, self.buffer, out.buffer, n, layout, e as f32) {
                return Ok(out);
            }
        }
        let cpu = self.to_cpu_storage()?;
        let out = cpu.powf(layout, e)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        if self.dtype == DType::F32 {
            let n = layout.shape().elem_count();
            let out = self.device.alloc(DType::F32, n)?;
            if let Ok(()) = kernels::run_elu(&self.device, self.buffer, out.buffer, n, layout, alpha as f32) {
                return Ok(out);
            }
        }
        let cpu = self.to_cpu_storage()?;
        let out = cpu.elu(layout, alpha)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn reduce_op(&self, op: crate::op::ReduceOp, layout: &Layout, s: &[usize]) -> Result<Self> {
        if self.dtype == DType::F32 {
            if let Ok(Some(out)) = self.reduce_native(op, layout, s) {
                return Ok(out);
            }
        }
        let cpu = self.to_cpu_storage()?;
        let out = cpu.reduce_op(op, layout, s)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn cmp(&self, op: crate::op::CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        if self.dtype == rhs.dtype && matches!(self.dtype, DType::F32 | DType::U32) {
            let n = lhs_l.shape().elem_count();
            let out = self.device.alloc(DType::U8, n)?;
            if let Ok(()) = kernels::run_cmp(
                &self.device,
                kernels::cmp_code(op),
                self.buffer,
                rhs.buffer,
                out.buffer,
                n,
                lhs_l,
                rhs_l,
                self.dtype == DType::U32,
            ) {
                return Ok(out);
            }
        }
        let lhs = self.to_cpu_storage()?;
        let rhs = rhs.to_cpu_storage()?;
        let out = lhs.cmp(op, &rhs, lhs_l, rhs_l)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        if dtype == self.dtype {
            return self.contiguous_copy(layout);
        }
        if let Some(name) = kernels::cast_kernel(self.dtype, dtype) {
            let n = layout.shape().elem_count();
            let out = self.device.alloc(dtype, n)?;
            if let Ok(()) = kernels::run_cast(&self.device, name, self.buffer, out.buffer, n, layout) {
                return Ok(out);
            }
        }
        let cpu = self.to_cpu_storage()?;
        let out = cpu.to_dtype(layout, dtype)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn unary_impl<B: crate::op::UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        if self.dtype == DType::F32 {
            if let Some(code) = kernels::unary_code(B::NAME) {
                let n = layout.shape().elem_count();
                let out = self.device.alloc(DType::F32, n)?;
                if let Ok(()) = kernels::run_unary(&self.device, code, self.buffer, out.buffer, n, layout) {
                    return Ok(out);
                }
            }
        }
        let cpu = self.to_cpu_storage()?;
        let out = cpu.unary_impl::<B>(layout)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn binary_impl<B: crate::op::BinaryOpT>(&self, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        if self.dtype == rhs.dtype && matches!(self.dtype, DType::F32 | DType::U32) {
            if let Some(code) = kernels::binary_code(B::NAME) {
                let n = lhs_l.shape().elem_count();
                let out = self.device.alloc(self.dtype, n)?;
                if let Ok(()) = kernels::run_binary(&self.device, code, self.buffer, rhs.buffer, out.buffer, n, lhs_l, rhs_l, self.dtype == DType::U32) {
                    return Ok(out);
                }
            }
        }
        let lhs = self.to_cpu_storage()?;
        let rhs = rhs.to_cpu_storage()?;
        let out = lhs.binary_impl::<B>(&rhs, lhs_l, rhs_l)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn where_cond(&self, layout: &Layout, t: &Self, t_l: &Layout, f: &Self, f_l: &Layout) -> Result<Self> {
        if t.dtype == f.dtype && matches!(self.dtype, DType::U8 | DType::U32) {
            let elem = t.elem_size();
            if elem == 4 || (elem == 8 && self.dtype == DType::U8) {
                let n = layout.shape().elem_count();
                let out = self.device.alloc(t.dtype, n)?;
                if let Ok(()) = kernels::run_where(
                    &self.device,
                    self.buffer,
                    t.buffer,
                    f.buffer,
                    out.buffer,
                    n,
                    layout,
                    t_l,
                    f_l,
                    self.dtype == DType::U32,
                    elem == 8,
                ) {
                    return Ok(out);
                }
            }
        }
        let cond = self.to_cpu_storage()?;
        let t = t.to_cpu_storage()?;
        let f = f.to_cpu_storage()?;
        let out = cond.where_cond(layout, &t, t_l, &f, f_l)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn conv1d(&self, l: &Layout, kernel: &Self, kernel_l: &Layout, params: &crate::conv::ParamsConv1D) -> Result<Self> {
        let inp = self.to_cpu_storage()?;
        let kernel = kernel.to_cpu_storage()?;
        let out = inp.conv1d(l, &kernel, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn conv_transpose1d(&self, l: &Layout, kernel: &Self, kernel_l: &Layout, params: &crate::conv::ParamsConvTranspose1D) -> Result<Self> {
        let inp = self.to_cpu_storage()?;
        let kernel = kernel.to_cpu_storage()?;
        let out = inp.conv_transpose1d(l, &kernel, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn conv2d(&self, l: &Layout, kernel: &Self, kernel_l: &Layout, params: &crate::conv::ParamsConv2D) -> Result<Self> {
        let inp = self.to_cpu_storage()?;
        let kernel = kernel.to_cpu_storage()?;
        let out = inp.conv2d(l, &kernel, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn conv_transpose2d(&self, l: &Layout, kernel: &Self, kernel_l: &Layout, params: &crate::conv::ParamsConvTranspose2D) -> Result<Self> {
        let inp = self.to_cpu_storage()?;
        let kernel = kernel.to_cpu_storage()?;
        let out = inp.conv_transpose2d(l, &kernel, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        if matches!(self.elem_size(), 4 | 8) && matches!(ids.dtype, DType::U32 | DType::I64 | DType::U8) {
            match self.index_select_native(ids, l, ids_l, dim) {
                Ok(out) => return Ok(out),
                Err(_) => {}
            }
        }
        let src = self.to_cpu_storage()?;
        let ids = ids.to_cpu_storage()?;
        let out = src.index_select(&ids, l, ids_l, dim)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        if self.elem_size() == 4 && matches!(ids.dtype, DType::U32 | DType::I64 | DType::U8) {
            match self.gather_native(l, ids, ids_l, dim) {
                Ok(out) => return Ok(out),
                Err(_) => {}
            }
        }
        let src = self.to_cpu_storage()?;
        let ids = ids.to_cpu_storage()?;
        let out = src.gather(l, &ids, ids_l, dim)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn scatter_set(&mut self, l: &Layout, ids: &Self, ids_l: &Layout, src: &Self, src_l: &Layout, dim: usize) -> Result<()> {
        if self.elem_size() == 4 && src.dtype == self.dtype && l.is_contiguous() {
            if self.scatter_native(false, l, ids, ids_l, src, src_l, dim).is_ok() {
                return Ok(());
            }
        }
        let mut tgt = self.to_cpu_storage()?;
        let ids = ids.to_cpu_storage()?;
        let src = src.to_cpu_storage()?;
        tgt.scatter_set(l, &ids, ids_l, &src, src_l, dim)?;
        let dev = self.device.clone();
        *self = dev.storage_from_cpu_storage(&tgt)?;
        Ok(())
    }

    fn scatter_add_set(&mut self, l: &Layout, ids: &Self, ids_l: &Layout, src: &Self, src_l: &Layout, dim: usize) -> Result<()> {
        if self.dtype == DType::F32 && src.dtype == DType::F32 && l.is_contiguous() {
            if self.scatter_native(true, l, ids, ids_l, src, src_l, dim).is_ok() {
                return Ok(());
            }
        }
        let mut tgt = self.to_cpu_storage()?;
        let ids = ids.to_cpu_storage()?;
        let src = src.to_cpu_storage()?;
        tgt.scatter_add_set(l, &ids, ids_l, &src, src_l, dim)?;
        let dev = self.device.clone();
        *self = dev.storage_from_cpu_storage(&tgt)?;
        Ok(())
    }

    fn index_add(&self, l: &Layout, ids: &Self, ids_l: &Layout, src: &Self, src_l: &Layout, dim: usize) -> Result<Self> {
        if self.dtype == DType::F32 && src.dtype == DType::F32 {
            match self.index_add_native(l, ids, ids_l, src, src_l, dim) {
                Ok(out) => return Ok(out),
                Err(_) => {}
            }
        }
        let tgt = self.to_cpu_storage()?;
        let ids = ids.to_cpu_storage()?;
        let src = src.to_cpu_storage()?;
        let out = tgt.index_add(l, &ids, ids_l, &src, src_l, dim)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn matmul(&self, rhs: &Self, bmnk: (usize, usize, usize, usize), lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        if self.dtype == DType::F32 && rhs.dtype == DType::F32 {
            match self.matmul_native(rhs, bmnk, lhs_l, rhs_l) {
                Ok(out) => return Ok(out),
                Err(_) => {}
            }
        }
        let lhs = self.to_cpu_storage()?;
        let rhs = rhs.to_cpu_storage()?;
        let out = lhs.matmul(&rhs, bmnk, lhs_l, rhs_l)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        if dst.dtype == self.dtype {
            let n = src_l.shape().elem_count();
            if let Ok(()) = kernels::run_copy_strided(&self.device, self.elem_size(), self.buffer, dst.buffer, n, src_l, dst_offset) {
                return Ok(());
            }
        }
        let src = self.to_cpu_storage()?;
        let mut dst_cpu = dst.to_cpu_storage()?;
        src.copy_strided_src(&mut dst_cpu, dst_offset, src_l)?;
        let dev = dst.device.clone();
        *dst = dev.storage_from_cpu_storage(&dst_cpu)?;
        Ok(())
    }

    fn copy2d(&self, dst: &mut Self, d1: usize, d2: usize, src_s: usize, dst_s: usize, src_o: usize, dst_o: usize) -> Result<()> {
        if dst.dtype == self.dtype {
            if let Ok(()) = kernels::run_copy2d(&self.device, self.elem_size(), self.buffer, dst.buffer, d1, d2, src_s, dst_s, src_o, dst_o) {
                return Ok(());
            }
        }
        let src = self.to_cpu_storage()?;
        let mut dst_cpu = dst.to_cpu_storage()?;
        src.copy2d(&mut dst_cpu, d1, d2, src_s, dst_s, src_o, dst_o)?;
        let dev = dst.device.clone();
        *dst = dev.storage_from_cpu_storage(&dst_cpu)?;
        Ok(())
    }

    fn avg_pool2d(&self, layout: &Layout, k: (usize, usize), s: (usize, usize)) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let out = cpu.avg_pool2d(layout, k, s)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn max_pool2d(&self, layout: &Layout, k: (usize, usize), s: (usize, usize)) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let out = cpu.max_pool2d(layout, k, s)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn upsample_nearest1d(&self, layout: &Layout, sz: usize) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let out = cpu.upsample_nearest1d(layout, sz)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn upsample_nearest2d(&self, layout: &Layout, h: usize, w: usize) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let out = cpu.upsample_nearest2d(layout, h, w)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn upsample_bilinear2d(&self, layout: &Layout, h: usize, w: usize, align_corners: bool, scale_h: Option<f64>, scale_w: Option<f64>) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let out = cpu.upsample_bilinear2d(layout, h, w, align_corners, scale_h, scale_w)?;
        self.device.storage_from_cpu_storage(&out)
    }
}

/// Strides of a matmul operand `[batch..., rows, cols]` under `l`, when the
/// batch dims collapse to one stride — identical to the OpenCL helper.
pub fn mat_strides(l: &Layout, batch: usize) -> Option<MatStrides> {
    let dims = l.dims();
    let st = l.stride();
    let r = dims.len();
    if r < 2 {
        return None;
    }
    let row = st[r - 2];
    let col = st[r - 1];
    let batch_stride = if r == 2 {
        0
    } else {
        let bdims = &dims[..r - 2];
        let bst = &st[..r - 2];
        if bdims.iter().product::<usize>() != batch {
            return None;
        }
        if bst.iter().all(|&s| s == 0) {
            0
        } else {
            let inner = bst[bdims.len() - 1];
            for i in 0..bdims.len() - 1 {
                if bdims[i] > 1 && bst[i] != bdims[i + 1] * bst[i + 1] {
                    return None;
                }
            }
            inner
        }
    };
    Some(MatStrides { row, col, offset: l.start_offset(), batch: batch_stride })
}

fn scalar_bits(s: crate::scalar::Scalar) -> Option<u64> {
    use crate::scalar::Scalar::*;
    Some(match s {
        U8(v) => v as u64,
        U32(v) => v as u64,
        I16(v) => v as u16 as u64,
        I32(v) => v as u32 as u64,
        I64(v) => v as u64,
        BF16(v) => v.to_bits() as u64,
        F16(v) => v.to_bits() as u64,
        F32(v) => v.to_bits() as u64,
        F64(v) => v.to_bits(),
        _ => return None,
    })
}

impl SyclStorage {
    fn reduce_native(&self, op: crate::op::ReduceOp, layout: &Layout, s: &[usize]) -> Result<Option<Self>> {
        let dims = layout.dims();
        let rank = dims.len();
        let code = match op {
            crate::op::ReduceOp::Sum => kernels::RED_SUM,
            crate::op::ReduceOp::Max => kernels::RED_MAX,
            crate::op::ReduceOp::Min => kernels::RED_MIN,
            crate::op::ReduceOp::ArgMax | crate::op::ReduceOp::ArgMin => {
                if rank == 0 || s != [rank - 1] {
                    return Ok(None);
                }
                let Some((o1, _)) = layout.contiguous_offsets() else { return Ok(None) };
                let cols = dims[rank - 1];
                let rows = dims[..rank - 1].iter().product::<usize>();
                if cols == 0 {
                    return Ok(None);
                }
                let out = self.device.alloc(DType::U32, rows)?;
                kernels::run_arg_last(
                    &self.device,
                    matches!(op, crate::op::ReduceOp::ArgMax),
                    self.buffer,
                    out.buffer,
                    rows,
                    cols,
                    o1,
                )?;
                return Ok(Some(out));
            }
        };
        let mut reduced = s.to_vec();
        reduced.sort_unstable();
        reduced.dedup();
        if reduced.iter().any(|&d| d >= rank) {
            return Ok(None);
        }
        let out_dims: Vec<usize> = dims
            .iter()
            .enumerate()
            .map(|(i, &d)| if reduced.contains(&i) { 1 } else { d })
            .collect();
        let n_out = out_dims.iter().product::<usize>();
        if let Some((o1, _)) = layout.contiguous_offsets() {
            let n_trailing = reduced.len();
            let trailing_ok = reduced.iter().enumerate().all(|(i, &d)| d == rank - n_trailing + i);
            if trailing_ok && n_trailing > 0 {
                let cols = dims[rank - n_trailing..].iter().product::<usize>();
                let rows = n_out;
                if cols == 0 && code != kernels::RED_SUM {
                    return Ok(None);
                }
                let out = self.device.alloc(DType::F32, rows)?;
                if cols == 0 {
                    kernels::run_fill(&self.device, 4, out.buffer, rows, &Layout::contiguous(rows), 0)?;
                } else {
                    kernels::run_reduce_last(&self.device, code, self.buffer, out.buffer, rows, cols, o1)?;
                }
                return Ok(Some(out));
            }
        }
        let count = reduced.iter().map(|&d| dims[d]).product::<usize>();
        if count == 0 && code != kernels::RED_SUM {
            return Ok(None);
        }
        let st = layout.stride();
        let out_strides: Vec<usize> = st
            .iter()
            .enumerate()
            .map(|(i, &v)| if reduced.contains(&i) { 0 } else { v })
            .collect();
        let ix = Idx::new(&out_dims)?.with_strides(0, &out_strides, layout.start_offset())?;
        let rdims: Vec<usize> = reduced.iter().map(|&d| dims[d]).collect();
        let rstrides: Vec<usize> = reduced.iter().map(|&d| st[d]).collect();
        let rd = if rdims.is_empty() {
            Idx::new(&[1])?.with_strides(0, &[0], 0)?
        } else {
            Idx::new(&rdims)?.with_strides(0, &rstrides, 0)?
        };
        let out = self.device.alloc(DType::F32, n_out)?;
        kernels::run_reduce_generic(
            &self.device,
            code,
            self.buffer,
            out.buffer,
            n_out,
            ix,
            rd,
            count.max(if rdims.is_empty() { 1 } else { 0 }),
        )?;
        Ok(Some(out))
    }

    fn index_select_native(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        let dims = l.dims();
        if dim >= dims.len() {
            return Err(Error::Msg("sycl index_select: bad dim".into()));
        }
        let (src, src_off) = self.as_contiguous(l)?;
        let n_ids = ids_l.shape().elem_count();
        let left = dims[..dim].iter().product::<usize>();
        let dim_size = dims[dim];
        let right = dims[dim + 1..].iter().product::<usize>();
        let n = left * n_ids * right;
        let out = self.device.alloc(self.dtype, n)?;
        let elem8 = self.elem_size() == 8;
        let (ids_s, ids_off, i64_ids) = match ids.dtype {
            DType::I64 if !elem8 => {
                let (s, o) = ids.as_contiguous(ids_l)?;
                (s, o, true)
            }
            _ => {
                let (s, o) = ids.ids_u32(ids_l)?;
                (s, o, false)
            }
        };
        kernels::run_index_select(
            &self.device,
            elem8,
            i64_ids,
            src.buffer,
            ids_s.buffer,
            out.buffer,
            n,
            left,
            n_ids,
            right,
            dim_size,
            src_off,
            ids_off,
            self.device.fault,
            crate::fault_slot::current() as i32,
        )?;
        Ok(out)
    }

    fn gather_native(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        let dims = l.dims();
        if dim >= dims.len() || ids_l.dims().len() != dims.len() {
            return Err(Error::Msg("sycl gather: bad dim or rank".into()));
        }
        let (ids_s, ids_off) = ids.ids_u32(ids_l)?;
        let ids_layout = Layout::new(ids_l.shape().clone(), ids_l.stride().to_vec(), ids_off);
        let n = ids_l.shape().elem_count();
        let mut src_strides = l.stride().to_vec();
        let dim_stride = src_strides[dim];
        src_strides[dim] = 0;
        let ix = Idx::new(ids_l.dims())?
            .with_layout(0, if ids_s.buffer == ids.buffer { ids_l } else { &ids_layout })?
            .with_strides(1, &src_strides, l.start_offset())?;
        let out = self.device.alloc(self.dtype, n)?;
        kernels::run_gather(
            &self.device,
            self.buffer,
            ids_s.buffer,
            out.buffer,
            n,
            ix,
            dim_stride,
            dims[dim],
            self.device.fault,
            crate::fault_slot::current() as i32,
        )?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn scatter_native(&mut self, add: bool, l: &Layout, ids: &Self, ids_l: &Layout, src: &Self, src_l: &Layout, dim: usize) -> Result<()> {
        let dims = l.dims();
        if dim >= dims.len() || ids_l.dims().len() != dims.len() || src_l.dims().len() != dims.len() {
            return Err(Error::Msg("sycl scatter: bad dim or rank".into()));
        }
        let (ids_s, ids_off) = ids.ids_u32(ids_l)?;
        let ids_layout = Layout::new(ids_l.shape().clone(), ids_l.stride().to_vec(), ids_off);
        let ids_l = if ids_s.buffer == ids.buffer { ids_l } else { &ids_layout };
        let mut cdims = ids_l.dims().to_vec();
        let n_j = cdims[dim];
        cdims[dim] = 1;
        let n = cdims.iter().product::<usize>();
        let ix = Idx::new(&cdims)?.with_layout(0, ids_l)?.with_layout(1, src_l)?.with_layout(2, l)?;
        kernels::run_scatter(
            &self.device,
            add,
            self.buffer,
            ids_s.buffer,
            src.buffer,
            n,
            ix,
            n_j,
            ids_l.stride()[dim],
            src_l.stride()[dim],
            l.stride()[dim],
            dims[dim],
            self.device.fault,
            crate::fault_slot::current() as i32,
        )
    }

    fn index_add_native(&self, l: &Layout, ids: &Self, ids_l: &Layout, src: &Self, src_l: &Layout, dim: usize) -> Result<Self> {
        let dims = l.dims();
        if dim >= dims.len() {
            return Err(Error::Msg("sycl index_add: bad dim".into()));
        }
        let dst = self.contiguous_copy(l)?;
        let (src_c, src_off) = src.as_contiguous(src_l)?;
        let (ids_s, ids_off) = ids.ids_u32(ids_l)?;
        let n_ids = ids_l.shape().elem_count();
        let left = dims[..dim].iter().product::<usize>();
        let right = dims[dim + 1..].iter().product::<usize>();
        kernels::run_index_add(
            &self.device,
            dst.buffer,
            ids_s.buffer,
            src_c.buffer,
            left,
            n_ids,
            right,
            dims[dim],
            src_off,
            ids_off,
            self.device.fault,
            crate::fault_slot::current() as i32,
        )?;
        Ok(dst)
    }

    fn matmul_native(&self, rhs: &Self, bmnk: (usize, usize, usize, usize), lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let (batch, m, n, k) = bmnk;
        let out = self.device.alloc(DType::F32, batch * m * n)?;
        let (lhs_buf, sa, _lhs_keep) = match mat_strides(lhs_l, batch) {
            Some(s) => (self.buffer, s, None),
            None => {
                let c = self.contiguous_copy(lhs_l)?;
                (c.buffer, MatStrides { row: k, col: 1, offset: 0, batch: m * k }, Some(c))
            }
        };
        let (rhs_buf, sb, _rhs_keep) = match mat_strides(rhs_l, batch) {
            Some(s) => (rhs.buffer, s, None),
            None => {
                let c = rhs.contiguous_copy(rhs_l)?;
                (c.buffer, MatStrides { row: n, col: 1, offset: 0, batch: k * n }, Some(c))
            }
        };
        kernels::run_matmul(&self.device, lhs_buf, rhs_buf, out.buffer, (batch, m, n, k), sa, sb)?;
        Ok(out)
    }
}

impl BackendDevice for SyclDevice {
    type Storage = SyclStorage;

    fn new(ordinal: usize) -> Result<Self> {
        Self::new(ordinal)
    }

    fn set_seed(&self, _seed: u64) -> Result<()> {
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        Ok(0)
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Sycl { ordinal: 0 }
    }

    fn same_device(&self, other: &Self) -> bool {
        self.handle == other.handle
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let numel = shape.elem_count();
        let storage = self.alloc(dtype, numel)?;
        if numel > 0 {
            let l = Layout::contiguous(numel);
            if matches!(dtype.size_in_bytes(), 1 | 2 | 4 | 8) {
                kernels::run_fill(self, dtype.size_in_bytes(), storage.buffer, numel, &l, 0)?;
            } else {
                let bytes = numel * dtype.size_in_bytes();
                let zeros = vec![0u8; bytes];
                self.write_bytes(storage.buffer, 0, &zeros)?;
            }
        }
        Ok(storage)
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        self.alloc(dtype, shape.elem_count())
    }

    fn storage_from_slice<T: WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        SyclStorage::from_vec(s.to_vec(), self)
    }

    fn storage_from_cpu_storage(&self, cpu: &CpuStorage) -> Result<Self::Storage> {
        match cpu {
            CpuStorage::U8(v) => SyclStorage::from_vec(v.clone(), self),
            CpuStorage::U32(v) => SyclStorage::from_vec(v.clone(), self),
            CpuStorage::I16(v) => SyclStorage::from_vec(v.clone(), self),
            CpuStorage::I32(v) => SyclStorage::from_vec(v.clone(), self),
            CpuStorage::I64(v) => SyclStorage::from_vec(v.clone(), self),
            CpuStorage::F32(v) => SyclStorage::from_vec(v.clone(), self),
            CpuStorage::F64(v) => SyclStorage::from_vec(v.clone(), self),
            CpuStorage::F16(v) => SyclStorage::from_vec(v.clone(), self),
            CpuStorage::BF16(v) => SyclStorage::from_vec(v.clone(), self),
            other => Err(Error::Msg(format!(
                "sycl storage_from_cpu_storage: dtype {:?} not supported",
                other.dtype()
            ))),
        }
    }

    fn storage_from_cpu_storage_owned(&self, cpu: CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&cpu)
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, hi: f64) -> Result<Self::Storage> {
        let cpu = crate::cpu_backend::CpuDevice.rand_uniform(shape, dtype, lo, hi)?;
        self.storage_from_cpu_storage(&cpu)
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<Self::Storage> {
        let cpu = crate::cpu_backend::CpuDevice.rand_normal(shape, dtype, mean, std)?;
        self.storage_from_cpu_storage(&cpu)
    }

    fn synchronize(&self) -> Result<()> {
        self.finish()?;
        self.check_fault()
    }
}

/// Rows up to which `fwd` runs the fused dequantize-and-dot GEMV instead of
/// dequantizing the whole weight once and running the tiled GEMM.
fn qgemv_max_rows(dtype: crate::quantized::GgmlDType) -> usize {
    use crate::quantized::GgmlDType::*;
    if kernels::qgemv_multirow(dtype) {
        return kernels::QGEMV_MR_MAX_ROWS;
    }
    match dtype {
        F16 | BF16 => 16,
        Q2K | Q3K | Q4K | Q5K | Q6K | Q8K | Iq2Xxs => 4,
        _ => 8,
    }
}

/// A GGUF-block-quantized tensor held on the SYCL device in its on-disk
/// format (mirror of `QOpenClStorage` without the zero-copy host mappings).
#[derive(Debug)]
pub struct QSyclStorage {
    pub buffer: usize,
    pub byte_offset: u64,
    pub dtype: crate::quantized::GgmlDType,
    pub elem_count: usize,
    pub device: SyclDevice,
}


impl Drop for QSyclStorage {
    fn drop(&mut self) {
        if self.buffer != 0 {
            let fns = self.device.bridge.fns;
            unsafe { (fns.free)(self.buffer) };
        }
    }
}

impl QSyclStorage {
    fn bytes_for(dtype: crate::quantized::GgmlDType, elem_count: usize) -> Result<usize> {
        let bs = dtype.block_size();
        if !elem_count.is_multiple_of(bs) {
            return Err(Error::Msg(format!(
                "sycl: {elem_count} elements is not a whole number of {dtype:?} blocks"
            )));
        }
        Ok(elem_count / bs * dtype.type_size())
    }

    pub fn zeros(device: &SyclDevice, elem_count: usize, dtype: crate::quantized::GgmlDType) -> Result<Self> {
        let bytes = Self::bytes_for(dtype, elem_count)?;
        let buffer = device.alloc_raw(bytes)?;
        if bytes > 0 {
            device.write_bytes(buffer, 0, &vec![0u8; bytes])?;
        }
        Ok(Self { buffer, byte_offset: 0, dtype, elem_count, device: device.clone() })
    }

    /// Upload raw block bytes (a blocking write on the compute queue).
    pub fn from_bytes(device: &SyclDevice, dtype: crate::quantized::GgmlDType, elem_count: usize, data: &[u8]) -> Result<Self> {
        let bytes = Self::bytes_for(dtype, elem_count)?;
        if data.len() < bytes {
            return Err(Error::Msg(format!(
                "sycl: {} bytes given for a {dtype:?} tensor needing {bytes}",
                data.len()
            )));
        }
        let buffer = device.alloc_raw(bytes)?;
        if let Err(e) = device.write_bytes(buffer, 0, &data[..bytes]) {
            let fns = device.bridge.fns;
            unsafe { (fns.free)(buffer) };
            return Err(e);
        }
        Ok(Self { buffer, byte_offset: 0, dtype, elem_count, device: device.clone() })
    }

    pub fn dtype(&self) -> crate::quantized::GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &SyclDevice {
        &self.device
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        Self::bytes_for(self.dtype, self.elem_count).unwrap_or(0)
    }

    /// Read the block bytes back to the host.
    pub fn data(&self) -> Result<Vec<u8>> {
        let bytes = self.storage_size_in_bytes();
        let mut out = vec![0u8; bytes];
        self.device.read_bytes(self.buffer, self.byte_offset as usize, &mut out)?;
        self.device.check_fault()?;
        Ok(out)
    }

    /// Dequantize to an f32 storage on the device.
    pub fn dequantize(&self, elem_count: usize) -> Result<SyclStorage> {
        let out = self.device.alloc(DType::F32, elem_count)?;
        self.dequantize_into(out.buffer, elem_count)?;
        Ok(out)
    }

    fn dequantize_into(&self, out: usize, elem_count: usize) -> Result<()> {
        use crate::quantized::GgmlDType;
        match self.dtype {
            GgmlDType::F32 => {
                // Dense f32 weights: a straight device-to-device copy.
                let bytes = self.storage_size_in_bytes();
                let mut tmp = vec![0u8; bytes];
                self.device.read_bytes(self.buffer, self.byte_offset as usize, &mut tmp)?;
                self.device.write_bytes(out, 0, &tmp)
            }
            _ => kernels::run_dequant(&self.device, self.dtype, self.buffer, out, elem_count, self.byte_offset),
        }
    }

    /// Quantized matmul: `out[m, n] = x[m, k] @ W[n, k]` — the same routing
    /// as `QOpenClStorage::fwd` (GEMV for decode rows, dequantize + GEMM for
    /// prefill).
    pub fn fwd(&self, self_shape: &Shape, storage: &SyclStorage, layout: &Layout) -> Result<(SyclStorage, Shape)> {
        if storage.dtype != DType::F32 {
            return Err(Error::Msg(format!(
                "sycl qmatmul: input must be f32, got {:?}",
                storage.dtype
            )));
        }
        let Some((o1, _)) = layout.contiguous_offsets() else {
            return Err(Error::Msg(format!("sycl qmatmul: input tensor is not contiguous {layout:?}")));
        };
        let (n, k) = self_shape.dims2()?;
        let src_shape = layout.shape();
        if src_shape.rank() < 2 {
            return Err(Error::Msg(format!("sycl qmatmul: input has only one dimension {layout:?}")));
        }
        let mut dst_dims = src_shape.dims().to_vec();
        let last_k = dst_dims.pop().unwrap();
        if last_k != k {
            return Err(Error::Msg(format!(
                "sycl qmatmul: input {layout:?} incompatible with {self_shape:?}"
            )));
        }
        dst_dims.push(n);
        let dst_shape = Shape::from(dst_dims);
        let m = src_shape.elem_count() / k;
        let out = self.device.alloc(DType::F32, m * n)?;
        match self.dtype {
            crate::quantized::GgmlDType::F32 => {
                let sa = MatStrides { row: k, col: 1, offset: o1, batch: 0 };
                let sb = MatStrides { row: 1, col: k, offset: (self.byte_offset / 4) as usize, batch: 0 };
                kernels::run_matmul(&self.device, storage.buffer, self.buffer, out.buffer, (1, m, n, k), sa, sb)?;
            }
            crate::quantized::GgmlDType::F16 | crate::quantized::GgmlDType::BF16 if m <= qgemv_max_rows(self.dtype) => {
                kernels::run_hgemv(
                    &self.device,
                    self.dtype == crate::quantized::GgmlDType::BF16,
                    storage.buffer,
                    self.buffer,
                    out.buffer,
                    m,
                    n,
                    k,
                    self.byte_offset,
                    o1,
                )?;
            }
            _ if m <= qgemv_max_rows(self.dtype) => {
                kernels::run_qgemv(&self.device, self.dtype, storage.buffer, self.buffer, out.buffer, m, n, k, self.byte_offset, o1)?;
            }
            _ => {
                let sa = MatStrides { row: k, col: 1, offset: o1, batch: 0 };
                let sb = MatStrides { row: 1, col: k, offset: 0, batch: 0 };
                let bytes = n * k * 4;
                if bytes <= 1 << 30 {
                    // ManuallyDrop: the alias must not free the device's
                    // shared scratch buffer on scope exit.
                    let scratch = std::mem::ManuallyDrop::new(self.device.scratch(bytes)?);
                    self.dequantize_into(scratch.buffer, n * k)?;
                    kernels::run_matmul(&self.device, storage.buffer, scratch.buffer, out.buffer, (1, m, n, k), sa, sb)?;
                } else {
                    let w = self.dequantize(n * k)?;
                    kernels::run_matmul(&self.device, storage.buffer, w.buffer, out.buffer, (1, m, n, k), sa, sb)?;
                }
            }
        }
        Ok((out, dst_shape))
    }

    /// Gather rows `ids` of this `[rows, hidden]` table as f32 `[n_ids, hidden]`.
    pub fn embedding(&self, rows: usize, hidden: usize, ids: &SyclStorage, ids_l: &Layout) -> Result<SyclStorage> {
        let (ids_s, ids_off) = ids.ids_u32(ids_l)?;
        let n_ids = ids_l.shape().elem_count();
        let out = self.device.alloc(DType::F32, n_ids * hidden)?;
        let fslot = crate::fault_slot::current() as i32;
        match self.dtype {
            crate::quantized::GgmlDType::F32 => {
                kernels::run_index_select(
                    &self.device,
                    false,
                    false,
                    self.buffer,
                    ids_s.buffer,
                    out.buffer,
                    n_ids * hidden,
                    1,
                    n_ids,
                    hidden,
                    rows,
                    (self.byte_offset / 4) as usize,
                    ids_off,
                    self.device.fault,
                    fslot,
                )?;
            }
            crate::quantized::GgmlDType::F16 | crate::quantized::GgmlDType::BF16 => {
                kernels::run_hembed(
                    &self.device,
                    self.dtype == crate::quantized::GgmlDType::BF16,
                    self.buffer,
                    ids_s.buffer,
                    out.buffer,
                    n_ids,
                    hidden,
                    rows,
                    self.byte_offset,
                    ids_off,
                    self.device.fault,
                    fslot,
                )?;
            }
            _ => {
                kernels::run_qembed(
                    &self.device,
                    self.dtype,
                    self.buffer,
                    ids_s.buffer,
                    out.buffer,
                    n_ids,
                    hidden,
                    rows,
                    self.byte_offset,
                    ids_off,
                    self.device.fault,
                    fslot,
                )?
            }
        }
        Ok(out)
    }
}

/// Open a SYCL device and initialise its fault buffer.
pub fn new_sycl_device(ordinal: usize) -> Result<SyclDevice> {
    let mut dev = SyclDevice::new(ordinal)?;
    dev.init_fault()?;
    Ok(dev)
}
