//! OpenCL accelerator backend: `OpenClDevice` / `OpenClStorage` over a
//! hand-rolled `libOpenCL` FFI, plus `QOpenClStorage` for block-quantized
//! weights.
//!
//! Every `BackendStorage` op has a native kernel path (see `kernels.cl`)
//! with a CPU round-trip fallback for the cases no kernel covers.  Kernels
//! are enqueued asynchronously on the device's in-order queue, so a whole
//! transformer layer runs without a host synchronisation; the blocking
//! reads in `to_cpu_storage` are the only sync points.  This is what makes
//! an integrated GPU worth using: with per-op host round-trips its
//! shared-memory bandwidth advantage over the CPU is spent on transfers.
//!
//! Quantized weights stay in their GGUF block format on the device and are
//! dequantized inside the matmul kernel (`QOpenClStorage`), so a decode
//! step reads Q4_K bytes, not f32 — on an iGPU that shares DRAM with the
//! CPU, bandwidth is the whole game.  A mapped model file can be handed to
//! the device zero-copy (`CL_MEM_USE_HOST_PTR`), so the dense set is never
//! duplicated in RAM.

#![allow(clippy::missing_safety_doc)]

pub mod kernels;
pub use kernels::{fallback_count, native_enabled, native_exec_count};

use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::quantized::GgmlDType;
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};
use kernels::{Ctx, Idx, MatStrides};
use std::sync::Arc;

mod cl {
    pub const CL_SUCCESS: i32 = 0;
    pub const CL_DEVICE_TYPE_GPU: u64 = 1 << 2;
    pub const CL_DEVICE_TYPE_ACCELERATOR: u64 = 1 << 3;
    pub const CL_DEVICE_TYPE_CPU: u64 = 1 << 1;
    pub const CL_MEM_READ_WRITE: u64 = 1 << 0;
    pub const CL_MEM_READ_ONLY: u64 = 1 << 2;
    pub const CL_MEM_USE_HOST_PTR: u64 = 1 << 3;
    pub const CL_TRUE: u8 = 1;
    pub const CL_DEVICE_NAME: u32 = 0x102B;
    pub const CL_DEVICE_GLOBAL_MEM_SIZE: u32 = 0x101F;
    pub const CL_DEVICE_HOST_UNIFIED_MEMORY: u32 = 0x1035;
}

#[link(name = "OpenCL")]
extern "C" {
    fn clGetPlatformIDs(num_entries: u32, platforms: *mut usize, num_platforms: *mut u32) -> i32;
    fn clGetDeviceIDs(
        platform: usize,
        device_type: u64,
        num_entries: u32,
        devices: *mut usize,
        num_devices: *mut u32,
    ) -> i32;
    fn clGetDeviceInfo(device: usize, param: u32, size: usize, value: *mut std::ffi::c_void, ret: *mut usize) -> i32;
    fn clCreateContext(
        properties: *const std::ffi::c_void,
        num_devices: u32,
        devices: *const usize,
        pfn_notify: Option<unsafe extern "C" fn(*const std::ffi::c_void, *const std::ffi::c_void, usize, *mut std::ffi::c_void)>,
        user_data: *mut std::ffi::c_void,
        errcode_ret: *mut i32,
    ) -> usize;
    fn clCreateCommandQueue(
        context: usize,
        device: usize,
        properties: u64,
        errcode_ret: *mut i32,
    ) -> usize;
    fn clCreateBuffer(
        context: usize,
        flags: u64,
        size: usize,
        host_ptr: *mut std::ffi::c_void,
        errcode_ret: *mut i32,
    ) -> usize;
    fn clEnqueueWriteBuffer(
        command_queue: usize,
        buffer: usize,
        blocking_write: u8,
        offset: usize,
        size: usize,
        ptr: *const std::ffi::c_void,
        num_events_in_wait_list: u32,
        event_wait_list: *const usize,
        event: *mut usize,
    ) -> i32;
    fn clEnqueueReadBuffer(
        command_queue: usize,
        buffer: usize,
        blocking_read: u8,
        offset: usize,
        size: usize,
        ptr: *mut std::ffi::c_void,
        num_events_in_wait_list: u32,
        event_wait_list: *const usize,
        event: *mut usize,
    ) -> i32;
    fn clEnqueueCopyBuffer(
        command_queue: usize,
        src: usize,
        dst: usize,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
        num_events_in_wait_list: u32,
        event_wait_list: *const usize,
        event: *mut usize,
    ) -> i32;
    fn clFinish(command_queue: usize) -> i32;
    fn clReleaseMemObject(memobj: usize) -> i32;
    fn clReleaseCommandQueue(command_queue: usize) -> i32;
    fn clReleaseContext(context: usize) -> i32;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

#[derive(Debug)]
struct OpenClContext {
    context: usize,
    queue: usize,
    /// One `uint` per thread slot (`crate::fault_slot`) that the indexing
    /// kernels set on an out-of-range id (see `kernels.cl`); read and
    /// cleared by [`OpenClDevice::check_fault`].
    fault: usize,
}

impl OpenClContext {
    /// Complete every queued command and clear fault `slot`: the drain hook
    /// `crate::fault_slot` runs before a thread's slot is recycled.
    fn drain_slot(&self, slot: usize) {
        unsafe { clFinish(self.queue) };
        let zero = 0u32;
        let _ = unsafe { write_buffer_at(self.queue, self.fault, slot * 4, 4, &zero as *const u32 as *const u8) };
    }
}

impl Drop for OpenClContext {
    fn drop(&mut self) {
        kernels::forget_context(self.context);
        if self.fault != 0 {
            unsafe { clReleaseMemObject(self.fault) };
        }
        if self.queue != 0 {
            unsafe { clReleaseCommandQueue(self.queue) };
        }
        if self.context != 0 {
            unsafe { clReleaseContext(self.context) };
        }
    }
}

#[derive(Debug, Clone)]
pub struct OpenClDevice {
    gpu_id: usize,
    #[allow(dead_code)]
    platform_id: usize,
    device_id: usize,
    inner: Arc<OpenClContext>,
}

fn opencl_error(code: i32, op: &str) -> Error {
    Error::Msg(format!("opencl {op} failed with status {code}"))
}

fn device_info_string(device: usize, param: u32) -> String {
    let mut len: usize = 0;
    let rc = unsafe { clGetDeviceInfo(device, param, 0, std::ptr::null_mut(), &mut len) };
    if rc != cl::CL_SUCCESS || len == 0 {
        return String::new();
    }
    let mut buf = vec![0u8; len];
    let rc = unsafe { clGetDeviceInfo(device, param, len, buf.as_mut_ptr() as *mut std::ffi::c_void, std::ptr::null_mut()) };
    if rc != cl::CL_SUCCESS {
        return String::new();
    }
    String::from_utf8_lossy(&buf).trim_end_matches('\0').trim().to_string()
}

fn device_info_u64(device: usize, param: u32) -> Option<u64> {
    let mut v: u64 = 0;
    let rc = unsafe { clGetDeviceInfo(device, param, 8, &mut v as *mut u64 as *mut std::ffi::c_void, std::ptr::null_mut()) };
    (rc == cl::CL_SUCCESS).then_some(v)
}

fn device_info_u32(device: usize, param: u32) -> Option<u32> {
    let mut v: u32 = 0;
    let rc = unsafe { clGetDeviceInfo(device, param, 4, &mut v as *mut u32 as *mut std::ffi::c_void, std::ptr::null_mut()) };
    (rc == cl::CL_SUCCESS).then_some(v)
}

/// Enumerate every device of `device_type` across all platforms, in
/// platform order.
fn devices_of_type(device_type: u64) -> Vec<(usize, usize)> {
    let mut num_platforms: u32 = 0;
    let mut platforms = [0usize; 8];
    let rc = unsafe { clGetPlatformIDs(8, platforms.as_mut_ptr(), &mut num_platforms) };
    if rc != cl::CL_SUCCESS {
        return Vec::new();
    }
    let mut out = Vec::new();
    for &platform in platforms.iter().take(num_platforms.min(8) as usize) {
        let mut num_devices: u32 = 0;
        let mut devices = [0usize; 16];
        let d_rc = unsafe { clGetDeviceIDs(platform, device_type, 16, devices.as_mut_ptr(), &mut num_devices) };
        if d_rc == cl::CL_SUCCESS {
            for &d in devices.iter().take(num_devices.min(16) as usize) {
                out.push((platform, d));
            }
        }
    }
    out
}

fn init_device(gpu_id: usize) -> Result<OpenClDevice> {
    // Some ICDs (pocl among them) mis-enumerate when several threads open
    // their first context at once; device creation is rare, so serialise it.
    static INIT: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = INIT.lock().unwrap_or_else(|p| p.into_inner());
    // GPUs first; an accelerator or a CPU runtime (pocl, Intel's CPU
    // driver) only when the machine has no GPU platform at all, so tests and
    // GPU-less hosts still exercise the real kernels.
    let mut found = devices_of_type(cl::CL_DEVICE_TYPE_GPU);
    if found.is_empty() {
        found = devices_of_type(cl::CL_DEVICE_TYPE_ACCELERATOR);
    }
    if found.is_empty() {
        found = devices_of_type(cl::CL_DEVICE_TYPE_CPU);
    }
    if found.is_empty() {
        return Err(Error::Msg("opencl: no OpenCL device found".to_string()));
    }
    let (platform_id, device) = found[gpu_id.min(found.len() - 1)];
    let mut err: i32 = 0;
    let context = unsafe { clCreateContext(std::ptr::null(), 1, &device, None, std::ptr::null_mut(), &mut err) };
    if err != cl::CL_SUCCESS || context == 0 {
        return Err(opencl_error(err, "clCreateContext"));
    }
    let queue = unsafe { clCreateCommandQueue(context, device, 0, &mut err) };
    if err != cl::CL_SUCCESS || queue == 0 {
        unsafe { clReleaseContext(context) };
        return Err(opencl_error(err, "clCreateCommandQueue"));
    }
    let fault = match create_buffer(context, crate::fault_slot::BYTES, cl::CL_MEM_READ_WRITE) {
        Ok(b) => b,
        Err(e) => {
            unsafe {
                clReleaseCommandQueue(queue);
                clReleaseContext(context);
            }
            return Err(e);
        }
    };
    let zeros = vec![0u8; crate::fault_slot::BYTES];
    if let Err(e) = unsafe { write_buffer(queue, fault, zeros.len(), zeros.as_ptr()) } {
        unsafe {
            clReleaseMemObject(fault);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
        }
        return Err(e);
    }
    let inner = Arc::new(OpenClContext { context, queue, fault });
    let weak = Arc::downgrade(&inner);
    crate::fault_slot::register_drain(Box::new(move |slot| match weak.upgrade() {
        Some(ctx) => {
            ctx.drain_slot(slot);
            true
        }
        None => false,
    }));
    Ok(OpenClDevice { gpu_id, platform_id, device_id: device, inner })
}

impl OpenClDevice {
    pub fn new(gpu_id: usize) -> Result<Self> {
        init_device(gpu_id)
    }

    pub fn new_with_stream(gpu_id: usize) -> Result<Self> {
        Self::new(gpu_id)
    }

    pub fn id(&self) -> DeviceId {
        DeviceId(self.gpu_id)
    }

    /// The device's `CL_DEVICE_NAME`.
    pub fn name(&self) -> String {
        device_info_string(self.device_id, cl::CL_DEVICE_NAME)
    }

    /// `CL_DEVICE_GLOBAL_MEM_SIZE` in bytes.
    pub fn global_mem_size(&self) -> Option<u64> {
        device_info_u64(self.device_id, cl::CL_DEVICE_GLOBAL_MEM_SIZE)
    }

    /// Whether the device shares memory with the host (an iGPU / CPU
    /// runtime), i.e. `CL_DEVICE_HOST_UNIFIED_MEMORY`.
    pub fn host_unified_memory(&self) -> bool {
        device_info_u32(self.device_id, cl::CL_DEVICE_HOST_UNIFIED_MEMORY).unwrap_or(0) != 0
    }

    pub(crate) fn queue(&self) -> usize {
        self.inner.queue
    }

    pub(crate) fn context(&self) -> usize {
        self.inner.context
    }

    /// Handles a kernel launch needs.
    pub fn ctx(&self) -> Ctx {
        Ctx { context: self.context(), device: self.device_id, queue: self.queue(), fault: self.inner.fault, fslot: crate::fault_slot::current() as i32 }
    }

    /// Report (and clear) an out-of-range id an indexing kernel launched by
    /// this thread flagged since the last check.  Called at every host
    /// read-back and `synchronize`, the points where the CPU backend's
    /// error for the same input would have been observed.
    pub fn check_fault(&self) -> Result<()> {
        let slot = crate::fault_slot::current() * 4;
        let mut v = 0u32;
        unsafe { read_buffer(self.queue(), self.inner.fault, slot, 4, &mut v as *mut u32 as *mut u8) }?;
        if v == 0 {
            return Ok(());
        }
        let zero = 0u32;
        unsafe { write_buffer_at(self.queue(), self.inner.fault, slot, 4, &zero as *const u32 as *const u8) }?;
        Err(Error::Msg(
            "opencl: an index_select / gather / scatter / index_add / embedding id was out of range for the indexed dimension (reported at the next host read-back)".into(),
        ))
    }

    /// Uninitialised device storage for `numel` elements of `dtype`.
    pub fn alloc(&self, dtype: DType, numel: usize) -> Result<OpenClStorage> {
        let elem = dtype.size_in_bytes();
        if elem == 0 {
            return Err(Error::Msg("opencl alloc: unsupported (sub-byte) dtype".into()));
        }
        let bytes = numel.checked_mul(elem).ok_or_else(|| Error::Msg("opencl: overflow in storage size".into()))?;
        let buffer = create_buffer(self.context(), bytes.max(1), cl::CL_MEM_READ_WRITE)?;
        Ok(OpenClStorage { buffer, dtype, numel, device: self.clone() })
    }
}

/// An OpenCL storage: a raw `cl_mem` device buffer + dtype + element count.
#[derive(Debug)]
pub struct OpenClStorage {
    pub buffer: usize,
    pub dtype: DType,
    pub numel: usize,
    pub device: OpenClDevice,
}

impl Drop for OpenClStorage {
    fn drop(&mut self) {
        if self.buffer != 0 {
            unsafe { clReleaseMemObject(self.buffer) };
        }
    }
}

impl OpenClStorage {
    pub fn transfer_to_device(&self, dst: &OpenClDevice) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        dst.storage_from_cpu_storage(&cpu)
    }

    pub fn from_vec<T: crate::WithDType>(slice: Vec<T>, device: &OpenClDevice) -> Result<Self> {
        let dtype = T::DTYPE;
        let bytes = dtype
            .size_in_bytes()
            .checked_mul(slice.len())
            .ok_or_else(|| Error::Msg("opencl: overflow in storage size".into()))?;
        let buffer = create_buffer(device.context(), bytes.max(1), cl::CL_MEM_READ_WRITE)?;
        if bytes > 0 {
            // # Safety: slice.as_ptr() points to `bytes` valid bytes.
            let data = unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, bytes) };
            if let Err(e) = unsafe { write_buffer(device.queue(), buffer, bytes, data.as_ptr()) } {
                unsafe { clReleaseMemObject(buffer) };
                return Err(e);
            }
        }
        Ok(Self { buffer, dtype, numel: slice.len(), device: device.clone() })
    }

    fn ctx(&self) -> Ctx {
        self.device.ctx()
    }

    fn elem_size(&self) -> usize {
        self.dtype.size_in_bytes()
    }

    /// A contiguous copy of the elements addressed by `l`, on the device.
    fn contiguous_copy(&self, l: &Layout) -> Result<Self> {
        let n = l.shape().elem_count();
        let out = self.device.alloc(self.dtype, n)?;
        kernels::run_copy_strided(&self.ctx(), self.elem_size(), self.buffer, out.buffer, n, l, 0)?;
        Ok(out)
    }

    /// `(storage, offset)`: the storage itself when `l` is contiguous
    /// (with its start offset), otherwise a contiguous copy at offset 0.
    fn as_contiguous(&self, l: &Layout) -> Result<(std::borrow::Cow<'_, Self>, usize)> {
        match l.contiguous_offsets() {
            Some((o1, _)) => Ok((std::borrow::Cow::Borrowed(self), o1)),
            None => Ok((std::borrow::Cow::Owned(self.contiguous_copy(l)?), 0)),
        }
    }

    /// u32 ids from a u32 / i64 / u8 id storage (cast on the device when
    /// needed); returns the storage and the offset of its first element.
    fn ids_u32(&self, l: &Layout) -> Result<(std::borrow::Cow<'_, Self>, usize)> {
        match self.dtype {
            DType::U32 => self.as_contiguous(l),
            DType::I64 | DType::U8 => {
                let n = l.shape().elem_count();
                let out = self.device.alloc(DType::U32, n)?;
                // i64 ids saturate (negative / beyond u32 → u32::MAX) so the
                // consuming kernel's bounds check faults instead of a
                // truncated id selecting the wrong element.
                let name = if self.dtype == DType::I64 {
                    "k_ids_i64"
                } else {
                    kernels::cast_kernel(self.dtype, DType::U32).ok_or_else(|| Error::Msg("opencl: no id cast".into()))?
                };
                kernels::run_cast(&self.ctx(), name, self.buffer, out.buffer, n, l)?;
                Ok((std::borrow::Cow::Owned(out), 0))
            }
            d => Err(Error::Msg(format!("opencl: unsupported index dtype {d:?}"))),
        }
    }
}

impl Clone for OpenClStorage {
    /// Whole-buffer device-to-device copy.
    fn clone(&self) -> Self {
        self.try_clone(&Layout::contiguous(self.numel)).expect("opencl: device buffer copy failed")
    }
}

fn create_buffer(context: usize, bytes: usize, flags: u64) -> Result<usize> {
    let mut err: i32 = 0;
    let buffer = unsafe { clCreateBuffer(context, flags, bytes, std::ptr::null_mut(), &mut err) };
    if err != cl::CL_SUCCESS || buffer == 0 {
        return Err(opencl_error(err, "clCreateBuffer"));
    }
    Ok(buffer)
}

/// # Safety
/// `ptr` must point to at least `bytes` valid bytes.
unsafe fn write_buffer(queue: usize, buffer: usize, bytes: usize, ptr: *const u8) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    let rc = clEnqueueWriteBuffer(queue, buffer, cl::CL_TRUE, 0, bytes, ptr as *const std::ffi::c_void, 0, std::ptr::null(), std::ptr::null_mut());
    if rc != cl::CL_SUCCESS {
        return Err(opencl_error(rc, "clEnqueueWriteBuffer"));
    }
    Ok(())
}

unsafe fn write_buffer_at(queue: usize, buffer: usize, offset: usize, bytes: usize, ptr: *const u8) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    let rc = clEnqueueWriteBuffer(queue, buffer, cl::CL_TRUE, offset, bytes, ptr as *const std::ffi::c_void, 0, std::ptr::null(), std::ptr::null_mut());
    if rc != cl::CL_SUCCESS {
        return Err(opencl_error(rc, "clEnqueueWriteBuffer"));
    }
    Ok(())
}

/// # Safety
/// `ptr` must point to at least `bytes` valid bytes.
unsafe fn read_buffer(queue: usize, buffer: usize, offset: usize, bytes: usize, ptr: *mut u8) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    let rc = clEnqueueReadBuffer(queue, buffer, cl::CL_TRUE, offset, bytes, ptr as *mut std::ffi::c_void, 0, std::ptr::null(), std::ptr::null_mut());
    if rc != cl::CL_SUCCESS {
        return Err(opencl_error(rc, "clEnqueueReadBuffer"));
    }
    Ok(())
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

fn native() -> bool {
    kernels::native_enabled()
}

/// Strides of a matmul operand `[batch..., rows, cols]` under `l`, when
/// the batch dims collapse to one stride (contiguous, transposed, or fully
/// broadcast batches).  `None` means the operand must be materialised.
fn mat_strides(l: &Layout, batch: usize) -> Option<MatStrides> {
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
            // Linear batch: stride[i] == dims[i+1] * stride[i+1] for the batch dims.
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

impl BackendStorage for OpenClStorage {
    type Device = OpenClDevice;

    fn try_clone(&self, _layout: &Layout) -> Result<Self> {
        let bytes = self.numel * self.elem_size();
        let out = self.device.alloc(self.dtype, self.numel)?;
        if bytes > 0 {
            let rc = unsafe { clEnqueueCopyBuffer(self.device.queue(), self.buffer, out.buffer, 0, 0, bytes, 0, std::ptr::null(), std::ptr::null_mut()) };
            if rc != cl::CL_SUCCESS {
                return Err(opencl_error(rc, "clEnqueueCopyBuffer"));
            }
        }
        Ok(out)
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn const_set(&mut self, s: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        if native() {
            if let Some(bits) = scalar_bits(s) {
                let n = layout.shape().elem_count();
                match kernels::run_fill(&self.ctx(), self.elem_size(), self.buffer, n, layout, bits) {
                    Ok(()) => return Ok(()),
                    Err(e) => kernels::note_fallback("const_set", Some(&e)),
                }
            } else {
                kernels::note_fallback("const_set", None);
            }
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
        unsafe { read_buffer(self.device.queue(), self.buffer, 0, bytes, raw.as_mut_ptr()) }?;
        self.device.check_fault()?;
        Ok(match self.dtype {
            DType::U8 => CpuStorage::U8(raw),
            DType::U32 => CpuStorage::U32(transmute_bytes(&raw, self.numel)),
            DType::I16 => CpuStorage::I16(transmute_bytes(&raw, self.numel)),
            DType::I32 => CpuStorage::I32(transmute_bytes(&raw, self.numel)),
            DType::I64 => CpuStorage::I64(transmute_bytes(&raw, self.numel)),
            DType::F32 => CpuStorage::F32(transmute_bytes(&raw, self.numel)),
            DType::F64 => CpuStorage::F64(transmute_bytes(&raw, self.numel)),
            DType::F16 => CpuStorage::F16(transmute_bytes::<u16>(&raw, self.numel).into_iter().map(half::f16::from_bits).collect()),
            DType::BF16 => CpuStorage::BF16(transmute_bytes::<u16>(&raw, self.numel).into_iter().map(half::bf16::from_bits).collect()),
            other => return Err(Error::Msg(format!("opencl to_cpu_storage: dtype {other:?} not supported"))),
        })
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        if native() && self.dtype == DType::F32 {
            let n = layout.shape().elem_count();
            let out = self.device.alloc(DType::F32, n)?;
            match kernels::run_affine(&self.ctx(), self.buffer, out.buffer, n, layout, mul as f32, add as f32) {
                Ok(()) => return Ok(out),
                Err(e) => kernels::note_fallback("affine", Some(&e)),
            }
        } else if native() {
            kernels::note_fallback("affine", None);
        }
        let cpu = self.to_cpu_storage()?;
        let out = cpu.affine(layout, mul, add)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        if native() && self.dtype == DType::F32 {
            let n = layout.shape().elem_count();
            let out = self.device.alloc(DType::F32, n)?;
            match kernels::run_powf(&self.ctx(), self.buffer, out.buffer, n, layout, e as f32) {
                Ok(()) => return Ok(out),
                Err(err) => kernels::note_fallback("powf", Some(&err)),
            }
        } else if native() {
            kernels::note_fallback("powf", None);
        }
        let cpu = self.to_cpu_storage()?;
        let out = cpu.powf(layout, e)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        if native() && self.dtype == DType::F32 {
            let n = layout.shape().elem_count();
            let out = self.device.alloc(DType::F32, n)?;
            match kernels::run_elu(&self.ctx(), self.buffer, out.buffer, n, layout, alpha as f32) {
                Ok(()) => return Ok(out),
                Err(err) => kernels::note_fallback("elu", Some(&err)),
            }
        } else if native() {
            kernels::note_fallback("elu", None);
        }
        let cpu = self.to_cpu_storage()?;
        let out = cpu.elu(layout, alpha)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, s: &[usize]) -> Result<Self> {
        if native() && self.dtype == DType::F32 {
            match self.reduce_native(op, layout, s) {
                Ok(Some(out)) => return Ok(out),
                Ok(None) => kernels::note_fallback("reduce_op", None),
                Err(e) => kernels::note_fallback("reduce_op", Some(&e)),
            }
        } else if native() {
            kernels::note_fallback("reduce_op", None);
        }
        let cpu = self.to_cpu_storage()?;
        let out = cpu.reduce_op(op, layout, s)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        if native() && self.dtype == rhs.dtype && matches!(self.dtype, DType::F32 | DType::U32) {
            let n = lhs_l.shape().elem_count();
            let out = self.device.alloc(DType::U8, n)?;
            match kernels::run_cmp(&self.ctx(), kernels::cmp_code(op), self.buffer, rhs.buffer, out.buffer, n, lhs_l, rhs_l, self.dtype == DType::U32) {
                Ok(()) => return Ok(out),
                Err(e) => kernels::note_fallback("cmp", Some(&e)),
            }
        } else if native() {
            kernels::note_fallback("cmp", None);
        }
        let lhs = self.to_cpu_storage()?;
        let rhs = rhs.to_cpu_storage()?;
        let out = lhs.cmp(op, &rhs, lhs_l, rhs_l)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        if native() {
            if dtype == self.dtype {
                return self.contiguous_copy(layout);
            }
            match kernels::cast_kernel(self.dtype, dtype) {
                Some(name) => {
                    let n = layout.shape().elem_count();
                    let out = self.device.alloc(dtype, n)?;
                    match kernels::run_cast(&self.ctx(), name, self.buffer, out.buffer, n, layout) {
                        Ok(()) => return Ok(out),
                        Err(e) => kernels::note_fallback("to_dtype", Some(&e)),
                    }
                }
                None => kernels::note_fallback("to_dtype", None),
            }
        }
        let cpu = self.to_cpu_storage()?;
        let out = cpu.to_dtype(layout, dtype)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        if native() && self.dtype == DType::F32 {
            if let Some(code) = kernels::unary_code(B::NAME) {
                let n = layout.shape().elem_count();
                let out = self.device.alloc(DType::F32, n)?;
                match kernels::run_unary(&self.ctx(), code, self.buffer, out.buffer, n, layout) {
                    Ok(()) => return Ok(out),
                    Err(e) => kernels::note_fallback(B::NAME, Some(&e)),
                }
            } else {
                kernels::note_fallback(B::NAME, None);
            }
        } else if native() {
            kernels::note_fallback(B::NAME, None);
        }
        let cpu = self.to_cpu_storage()?;
        let out = cpu.unary_impl::<B>(layout)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn binary_impl<B: BinaryOpT>(&self, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        if native() && self.dtype == rhs.dtype && matches!(self.dtype, DType::F32 | DType::U32) {
            if let Some(code) = kernels::binary_code(B::NAME) {
                let n = lhs_l.shape().elem_count();
                let out = self.device.alloc(self.dtype, n)?;
                match kernels::run_binary(&self.ctx(), code, self.buffer, rhs.buffer, out.buffer, n, lhs_l, rhs_l, self.dtype == DType::U32) {
                    Ok(()) => return Ok(out),
                    Err(e) => kernels::note_fallback(B::NAME, Some(&e)),
                }
            } else {
                kernels::note_fallback(B::NAME, None);
            }
        } else if native() {
            kernels::note_fallback(B::NAME, None);
        }
        let lhs = self.to_cpu_storage()?;
        let rhs = rhs.to_cpu_storage()?;
        let out = lhs.binary_impl::<B>(&rhs, lhs_l, rhs_l)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn where_cond(&self, layout: &Layout, t: &Self, t_l: &Layout, f: &Self, f_l: &Layout) -> Result<Self> {
        if native() && t.dtype == f.dtype && matches!(self.dtype, DType::U8 | DType::U32) {
            let elem = t.elem_size();
            if elem == 4 || (elem == 8 && self.dtype == DType::U8) {
                let n = layout.shape().elem_count();
                let out = self.device.alloc(t.dtype, n)?;
                match kernels::run_where(&self.ctx(), self.buffer, t.buffer, f.buffer, out.buffer, n, layout, t_l, f_l, self.dtype == DType::U32, elem == 8) {
                    Ok(()) => return Ok(out),
                    Err(e) => kernels::note_fallback("where_cond", Some(&e)),
                }
            } else {
                kernels::note_fallback("where_cond", None);
            }
        } else if native() {
            kernels::note_fallback("where_cond", None);
        }
        let cond = self.to_cpu_storage()?;
        let t = t.to_cpu_storage()?;
        let f = f.to_cpu_storage()?;
        let out = cond.where_cond(layout, &t, t_l, &f, f_l)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn conv1d(&self, l: &Layout, kernel: &Self, kernel_l: &Layout, params: &crate::conv::ParamsConv1D) -> Result<Self> {
        if native() {
            kernels::note_fallback("conv1d", None);
        }
        let inp = self.to_cpu_storage()?;
        let kernel = kernel.to_cpu_storage()?;
        let out = inp.conv1d(l, &kernel, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn conv_transpose1d(&self, l: &Layout, kernel: &Self, kernel_l: &Layout, params: &crate::conv::ParamsConvTranspose1D) -> Result<Self> {
        if native() {
            kernels::note_fallback("conv_transpose1d", None);
        }
        let inp = self.to_cpu_storage()?;
        let kernel = kernel.to_cpu_storage()?;
        let out = inp.conv_transpose1d(l, &kernel, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn conv2d(&self, l: &Layout, kernel: &Self, kernel_l: &Layout, params: &crate::conv::ParamsConv2D) -> Result<Self> {
        if native() {
            kernels::note_fallback("conv2d", None);
        }
        let inp = self.to_cpu_storage()?;
        let kernel = kernel.to_cpu_storage()?;
        let out = inp.conv2d(l, &kernel, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn conv_transpose2d(&self, l: &Layout, kernel: &Self, kernel_l: &Layout, params: &crate::conv::ParamsConvTranspose2D) -> Result<Self> {
        if native() {
            kernels::note_fallback("conv_transpose2d", None);
        }
        let inp = self.to_cpu_storage()?;
        let kernel = kernel.to_cpu_storage()?;
        let out = inp.conv_transpose2d(l, &kernel, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        if native() && matches!(self.elem_size(), 4 | 8) && matches!(ids.dtype, DType::U32 | DType::I64 | DType::U8) {
            match self.index_select_native(ids, l, ids_l, dim) {
                Ok(out) => return Ok(out),
                Err(e) => kernels::note_fallback("index_select", Some(&e)),
            }
        } else if native() {
            kernels::note_fallback("index_select", None);
        }
        let src = self.to_cpu_storage()?;
        let ids = ids.to_cpu_storage()?;
        let out = src.index_select(&ids, l, ids_l, dim)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        if native() && self.elem_size() == 4 && matches!(ids.dtype, DType::U32 | DType::I64 | DType::U8) {
            match self.gather_native(l, ids, ids_l, dim) {
                Ok(out) => return Ok(out),
                Err(e) => kernels::note_fallback("gather", Some(&e)),
            }
        } else if native() {
            kernels::note_fallback("gather", None);
        }
        let src = self.to_cpu_storage()?;
        let ids = ids.to_cpu_storage()?;
        let out = src.gather(l, &ids, ids_l, dim)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn scatter_set(&mut self, l: &Layout, ids: &Self, ids_l: &Layout, src: &Self, src_l: &Layout, dim: usize) -> Result<()> {
        if native() && self.elem_size() == 4 && src.dtype == self.dtype && l.is_contiguous() {
            match self.scatter_native(false, l, ids, ids_l, src, src_l, dim) {
                Ok(()) => return Ok(()),
                Err(e) => kernels::note_fallback("scatter_set", Some(&e)),
            }
        } else if native() {
            kernels::note_fallback("scatter_set", None);
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
        if native() && self.dtype == DType::F32 && src.dtype == DType::F32 && l.is_contiguous() {
            match self.scatter_native(true, l, ids, ids_l, src, src_l, dim) {
                Ok(()) => return Ok(()),
                Err(e) => kernels::note_fallback("scatter_add", Some(&e)),
            }
        } else if native() {
            kernels::note_fallback("scatter_add", None);
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
        if native() && self.dtype == DType::F32 && src.dtype == DType::F32 {
            match self.index_add_native(l, ids, ids_l, src, src_l, dim) {
                Ok(out) => return Ok(out),
                Err(e) => kernels::note_fallback("index_add", Some(&e)),
            }
        } else if native() {
            kernels::note_fallback("index_add", None);
        }
        let tgt = self.to_cpu_storage()?;
        let ids = ids.to_cpu_storage()?;
        let src = src.to_cpu_storage()?;
        let out = tgt.index_add(l, &ids, ids_l, &src, src_l, dim)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn matmul(&self, rhs: &Self, bmnk: (usize, usize, usize, usize), lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        if native() && self.dtype == DType::F32 && rhs.dtype == DType::F32 {
            match self.matmul_native(rhs, bmnk, lhs_l, rhs_l) {
                Ok(out) => return Ok(out),
                Err(e) => kernels::note_fallback("matmul", Some(&e)),
            }
        } else if native() {
            kernels::note_fallback("matmul", None);
        }
        let lhs = self.to_cpu_storage()?;
        let rhs = rhs.to_cpu_storage()?;
        let out = lhs.matmul(&rhs, bmnk, lhs_l, rhs_l)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        if native() && dst.dtype == self.dtype {
            let n = src_l.shape().elem_count();
            match kernels::run_copy_strided(&self.ctx(), self.elem_size(), self.buffer, dst.buffer, n, src_l, dst_offset) {
                Ok(()) => return Ok(()),
                Err(e) => kernels::note_fallback("copy_strided_src", Some(&e)),
            }
        } else if native() {
            kernels::note_fallback("copy_strided_src", None);
        }
        let src = self.to_cpu_storage()?;
        let mut dst_cpu = dst.to_cpu_storage()?;
        src.copy_strided_src(&mut dst_cpu, dst_offset, src_l)?;
        let dev = dst.device.clone();
        *dst = dev.storage_from_cpu_storage(&dst_cpu)?;
        Ok(())
    }

    fn copy2d(&self, dst: &mut Self, d1: usize, d2: usize, src_s: usize, dst_s: usize, src_o: usize, dst_o: usize) -> Result<()> {
        if native() && dst.dtype == self.dtype {
            match kernels::run_copy2d(&self.ctx(), self.elem_size(), self.buffer, dst.buffer, d1, d2, src_s, dst_s, src_o, dst_o) {
                Ok(()) => return Ok(()),
                Err(e) => kernels::note_fallback("copy2d", Some(&e)),
            }
        } else if native() {
            kernels::note_fallback("copy2d", None);
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

// ─── Native op bodies ────────────────────────────────────────────────────────

impl OpenClStorage {
    fn reduce_native(&self, op: ReduceOp, layout: &Layout, s: &[usize]) -> Result<Option<Self>> {
        let dims = layout.dims();
        let rank = dims.len();
        let code = match op {
            ReduceOp::Sum => kernels::RED_SUM,
            ReduceOp::Max => kernels::RED_MAX,
            ReduceOp::Min => kernels::RED_MIN,
            ReduceOp::ArgMax | ReduceOp::ArgMin => {
                // Only the contiguous last-dim case.
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
                kernels::run_arg_last(&self.ctx(), matches!(op, ReduceOp::ArgMax), self.buffer, out.buffer, rows, cols, o1)?;
                return Ok(Some(out));
            }
        };
        let mut reduced = s.to_vec();
        reduced.sort_unstable();
        reduced.dedup();
        if reduced.iter().any(|&d| d >= rank) {
            return Ok(None);
        }
        let out_dims: Vec<usize> = dims.iter().enumerate().map(|(i, &d)| if reduced.contains(&i) { 1 } else { d }).collect();
        let n_out = out_dims.iter().product::<usize>();
        // Fast path: a contiguous input reduced over its trailing dims is
        // `rows` rows of `cols` contiguous elements.
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
                    kernels::run_fill(&self.ctx(), 4, out.buffer, rows, &Layout::contiguous(rows), 0)?;
                } else {
                    kernels::run_reduce_last(&self.ctx(), code, self.buffer, out.buffer, rows, cols, o1)?;
                }
                return Ok(Some(out));
            }
        }
        // General: one work-item per output element.
        let count = reduced.iter().map(|&d| dims[d]).product::<usize>();
        if count == 0 && code != kernels::RED_SUM {
            return Ok(None);
        }
        let st = layout.stride();
        let out_strides: Vec<usize> = st.iter().enumerate().map(|(i, &v)| if reduced.contains(&i) { 0 } else { v }).collect();
        let ix = Idx::new(&out_dims)?.with_strides(0, &out_strides, layout.start_offset())?;
        let rdims: Vec<usize> = reduced.iter().map(|&d| dims[d]).collect();
        let rstrides: Vec<usize> = reduced.iter().map(|&d| st[d]).collect();
        let rd = if rdims.is_empty() { Idx::new(&[1])?.with_strides(0, &[0], 0)? } else { Idx::new(&rdims)?.with_strides(0, &rstrides, 0)? };
        let out = self.device.alloc(DType::F32, n_out)?;
        kernels::run_reduce_generic(&self.ctx(), code, self.buffer, out.buffer, n_out, ix, rd, count.max(if rdims.is_empty() { 1 } else { 0 }))?;
        Ok(Some(out))
    }

    fn index_select_native(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        let dims = l.dims();
        if dim >= dims.len() {
            return Err(Error::Msg("opencl index_select: bad dim".into()));
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
        kernels::run_index_select(&self.ctx(), elem8, i64_ids, src.buffer, ids_s.buffer, out.buffer, n, left, n_ids, right, dim_size, src_off, ids_off)?;
        Ok(out)
    }

    fn gather_native(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        let dims = l.dims();
        if dim >= dims.len() || ids_l.dims().len() != dims.len() {
            return Err(Error::Msg("opencl gather: bad dim or rank".into()));
        }
        let (ids_s, ids_off) = ids.ids_u32(ids_l)?;
        let ids_layout = Layout::new(ids_l.shape().clone(), ids_l.stride().to_vec(), ids_off);
        let n = ids_l.shape().elem_count();
        let mut src_strides = l.stride().to_vec();
        let dim_stride = src_strides[dim];
        src_strides[dim] = 0;
        let ix = Idx::new(ids_l.dims())?.with_layout(0, if ids_s.buffer == ids.buffer { ids_l } else { &ids_layout })?.with_strides(1, &src_strides, l.start_offset())?;
        let out = self.device.alloc(self.dtype, n)?;
        kernels::run_gather(&self.ctx(), self.buffer, ids_s.buffer, out.buffer, n, ix, dim_stride, dims[dim])?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn scatter_native(&mut self, add: bool, l: &Layout, ids: &Self, ids_l: &Layout, src: &Self, src_l: &Layout, dim: usize) -> Result<()> {
        let dims = l.dims();
        if dim >= dims.len() || ids_l.dims().len() != dims.len() || src_l.dims().len() != dims.len() {
            return Err(Error::Msg("opencl scatter: bad dim or rank".into()));
        }
        let (ids_s, ids_off) = ids.ids_u32(ids_l)?;
        let ids_layout = Layout::new(ids_l.shape().clone(), ids_l.stride().to_vec(), ids_off);
        let ids_l = if ids_s.buffer == ids.buffer { ids_l } else { &ids_layout };
        // Enumerate the ids space with `dim` collapsed; the kernel walks
        // `dim` itself, in order.
        let mut cdims = ids_l.dims().to_vec();
        let n_j = cdims[dim];
        cdims[dim] = 1;
        let n = cdims.iter().product::<usize>();
        let ix = Idx::new(&cdims)?.with_layout(0, ids_l)?.with_layout(1, src_l)?.with_layout(2, l)?;
        kernels::run_scatter(
            &self.ctx(),
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
        )
    }

    fn index_add_native(&self, l: &Layout, ids: &Self, ids_l: &Layout, src: &Self, src_l: &Layout, dim: usize) -> Result<Self> {
        let dims = l.dims();
        if dim >= dims.len() {
            return Err(Error::Msg("opencl index_add: bad dim".into()));
        }
        // The result starts as a contiguous copy of self.
        let dst = self.contiguous_copy(l)?;
        let (src_c, src_off) = src.as_contiguous(src_l)?;
        let (ids_s, ids_off) = ids.ids_u32(ids_l)?;
        let n_ids = ids_l.shape().elem_count();
        let left = dims[..dim].iter().product::<usize>();
        let right = dims[dim + 1..].iter().product::<usize>();
        kernels::run_index_add(&self.ctx(), dst.buffer, ids_s.buffer, src_c.buffer, left, n_ids, right, dims[dim], src_off, ids_off)?;
        Ok(dst)
    }

    fn matmul_native(&self, rhs: &Self, bmnk: (usize, usize, usize, usize), lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let (batch, m, n, k) = bmnk;
        let out = self.device.alloc(DType::F32, batch * m * n)?;
        // Operands whose batch dims do not collapse to one stride are
        // materialised contiguous on the device first.
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
        kernels::run_matmul(&self.ctx(), lhs_buf, rhs_buf, out.buffer, (batch, m, n, k), sa, sb)?;
        Ok(out)
    }
}

impl BackendDevice for OpenClDevice {
    type Storage = OpenClStorage;

    fn new(gpu_id: usize) -> Result<Self> {
        Self::new(gpu_id)
    }

    fn set_seed(&self, _seed: u64) -> Result<()> {
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        Ok(0)
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::OpenCl { gpu_id: self.gpu_id }
    }

    fn same_device(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let numel = shape.elem_count();
        let storage = self.alloc(dtype, numel)?;
        if numel > 0 {
            let l = Layout::contiguous(numel);
            if native() && matches!(dtype.size_in_bytes(), 1 | 2 | 4 | 8) {
                kernels::run_fill(&self.ctx(), dtype.size_in_bytes(), storage.buffer, numel, &l, 0)?;
            } else {
                let bytes = numel * dtype.size_in_bytes();
                let zeros = vec![0u8; bytes];
                unsafe { write_buffer(self.queue(), storage.buffer, bytes, zeros.as_ptr()) }?;
            }
        }
        Ok(storage)
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        self.alloc(dtype, shape.elem_count())
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        OpenClStorage::from_vec(s.to_vec(), self)
    }

    fn storage_from_cpu_storage(&self, cpu: &CpuStorage) -> Result<Self::Storage> {
        match cpu {
            CpuStorage::U8(v) => OpenClStorage::from_vec(v.clone(), self),
            CpuStorage::U32(v) => OpenClStorage::from_vec(v.clone(), self),
            CpuStorage::I16(v) => OpenClStorage::from_vec(v.clone(), self),
            CpuStorage::I32(v) => OpenClStorage::from_vec(v.clone(), self),
            CpuStorage::I64(v) => OpenClStorage::from_vec(v.clone(), self),
            CpuStorage::F32(v) => OpenClStorage::from_vec(v.clone(), self),
            CpuStorage::F64(v) => OpenClStorage::from_vec(v.clone(), self),
            CpuStorage::F16(v) => OpenClStorage::from_vec(v.clone(), self),
            CpuStorage::BF16(v) => OpenClStorage::from_vec(v.clone(), self),
            other => Err(Error::Msg(format!("opencl storage_from_cpu_storage: dtype {:?} not supported", other.dtype()))),
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
        let rc = unsafe { clFinish(self.queue()) };
        if rc != cl::CL_SUCCESS {
            return Err(opencl_error(rc, "clFinish"));
        }
        self.check_fault()
    }
}

// ─── Block-quantized weights on the device ───────────────────────────────────

/// Rows below which a quantized matmul dequantizes inside the GEMV kernel;
/// larger inputs (prefill) dequantize the weight to a scratch f32 buffer
/// once and run the tiled GEMM.
const QGEMV_MAX_ROWS: usize = 16;

/// A GGUF-block-quantized tensor held on the device in its on-disk format.
///
/// `buffer` may be a private device copy or a zero-copy wrapper over host
/// memory (`CL_MEM_USE_HOST_PTR`, page-aligned), in which case `byte_offset`
/// locates the tensor inside the wrapped range and `_host` keeps that host
/// memory alive.
pub struct QOpenClStorage {
    pub buffer: usize,
    pub byte_offset: u64,
    pub dtype: GgmlDType,
    pub elem_count: usize,
    pub device: OpenClDevice,
    _host: Option<Arc<dyn std::any::Any + Send + Sync>>,
}

impl std::fmt::Debug for QOpenClStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QOpenClStorage({:?}, {} elems, zero_copy={})", self.dtype, self.elem_count, self._host.is_some())
    }
}

impl Drop for QOpenClStorage {
    fn drop(&mut self) {
        if self.buffer != 0 {
            unsafe { clReleaseMemObject(self.buffer) };
        }
    }
}

/// The system base page size, the granularity `CL_MEM_USE_HOST_PTR`
/// ranges are widened to (every mapping is page-aligned, whatever the
/// page size).
fn page_size() -> usize {
    static PAGE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *PAGE.get_or_init(|| {
        let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if v > 0 { v as usize } else { 4096 }
    })
}

/// Whether zero-copy host buffers are enabled (`JOSHUA_OPENCL_ZERO_COPY=0`
/// disables them; useful when a driver copies anyway).
pub fn zero_copy_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("JOSHUA_OPENCL_ZERO_COPY") {
        Ok(s) => !(s == "0" || s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("off")),
        Err(_) => true,
    })
}

impl QOpenClStorage {
    fn bytes_for(dtype: GgmlDType, elem_count: usize) -> Result<usize> {
        let bs = dtype.block_size();
        if !elem_count.is_multiple_of(bs) {
            return Err(Error::Msg(format!("opencl: {elem_count} elements is not a whole number of {dtype:?} blocks")));
        }
        Ok(elem_count / bs * dtype.type_size())
    }

    pub fn zeros(device: &OpenClDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let bytes = Self::bytes_for(dtype, elem_count)?;
        let buffer = create_buffer(device.context(), bytes.max(1), cl::CL_MEM_READ_WRITE)?;
        if bytes > 0 {
            let zeros = vec![0u8; bytes];
            if let Err(e) = unsafe { write_buffer(device.queue(), buffer, bytes, zeros.as_ptr()) } {
                unsafe { clReleaseMemObject(buffer) };
                return Err(e);
            }
        }
        Ok(Self { buffer, byte_offset: 0, dtype, elem_count, device: device.clone(), _host: None })
    }

    /// Upload raw block bytes.
    pub fn from_bytes(device: &OpenClDevice, dtype: GgmlDType, elem_count: usize, data: &[u8]) -> Result<Self> {
        let bytes = Self::bytes_for(dtype, elem_count)?;
        if data.len() < bytes {
            return Err(Error::Msg(format!("opencl: {} bytes given for a {dtype:?} tensor needing {bytes}", data.len())));
        }
        let buffer = create_buffer(device.context(), bytes.max(1), cl::CL_MEM_READ_WRITE)?;
        if let Err(e) = unsafe { write_buffer(device.queue(), buffer, bytes, data.as_ptr()) } {
            unsafe { clReleaseMemObject(buffer) };
            return Err(e);
        }
        Ok(Self { buffer, byte_offset: 0, dtype, elem_count, device: device.clone(), _host: None })
    }

    /// Wrap `len` bytes at `offset` inside a host mapping (`base`, `base_len`)
    /// as a read-only zero-copy device buffer (`CL_MEM_USE_HOST_PTR`).  The
    /// wrapped range is widened to page boundaries, which every mapping
    /// covers, so the driver can alias the pages instead of copying.
    /// `keepalive` must own the mapping for as long as the storage lives.
    ///
    /// # Safety
    /// `base` must point to `base_len` readable bytes that stay valid and
    /// unmodified while `keepalive` is held.
    pub unsafe fn from_host_mapping(
        device: &OpenClDevice,
        dtype: GgmlDType,
        elem_count: usize,
        base: *const u8,
        base_len: usize,
        offset: usize,
        keepalive: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<Self> {
        let bytes = Self::bytes_for(dtype, elem_count)?;
        if offset.checked_add(bytes).is_none_or(|end| end > base_len) {
            return Err(Error::Msg("opencl: tensor range exceeds the host mapping".into()));
        }
        let page = page_size();
        let start = offset & !(page - 1);
        let end = (offset + bytes).div_ceil(page) * page;
        let end = end.min(base_len.div_ceil(page) * page).max(offset + bytes);
        let mut err: i32 = 0;
        let buffer = unsafe {
            clCreateBuffer(
                device.context(),
                cl::CL_MEM_READ_ONLY | cl::CL_MEM_USE_HOST_PTR,
                end - start,
                base.add(start) as *mut std::ffi::c_void,
                &mut err,
            )
        };
        if err != cl::CL_SUCCESS || buffer == 0 {
            return Err(opencl_error(err, "clCreateBuffer(CL_MEM_USE_HOST_PTR)"));
        }
        Ok(Self { buffer, byte_offset: (offset - start) as u64, dtype, elem_count, device: device.clone(), _host: Some(keepalive) })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &OpenClDevice {
        &self.device
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        Self::bytes_for(self.dtype, self.elem_count).unwrap_or(0)
    }

    /// Whether the weights alias host memory rather than a device copy.
    pub fn is_zero_copy(&self) -> bool {
        self._host.is_some()
    }

    /// Read the block bytes back to the host.
    pub fn data(&self) -> Result<Vec<u8>> {
        let bytes = self.storage_size_in_bytes();
        let mut out = vec![0u8; bytes];
        unsafe { read_buffer(self.device.queue(), self.buffer, self.byte_offset as usize, bytes, out.as_mut_ptr()) }?;
        self.device.check_fault()?;
        Ok(out)
    }

    /// Dequantize to an f32 storage on the device.
    pub fn dequantize(&self, elem_count: usize) -> Result<OpenClStorage> {
        let out = self.device.alloc(DType::F32, elem_count)?;
        match self.dtype {
            GgmlDType::F32 => {
                let rc = unsafe {
                    clEnqueueCopyBuffer(self.device.queue(), self.buffer, out.buffer, self.byte_offset as usize, 0, elem_count * 4, 0, std::ptr::null(), std::ptr::null_mut())
                };
                if rc != cl::CL_SUCCESS {
                    return Err(opencl_error(rc, "clEnqueueCopyBuffer"));
                }
            }
            _ => kernels::run_dequant(&self.device.ctx(), self.dtype, self.buffer, out.buffer, elem_count, self.byte_offset)?,
        }
        Ok(out)
    }

    /// `x @ W^T` for `x` on the device (f32, contiguous) and this `[n, k]` weight.
    pub fn fwd(&self, self_shape: &Shape, storage: &OpenClStorage, layout: &Layout) -> Result<(OpenClStorage, Shape)> {
        if storage.dtype != DType::F32 {
            return Err(Error::Msg(format!("opencl qmatmul: input must be f32, got {:?}", storage.dtype)));
        }
        let Some((o1, _)) = layout.contiguous_offsets() else {
            return Err(Error::Msg(format!("opencl qmatmul: input tensor is not contiguous {layout:?}")));
        };
        let (n, k) = self_shape.dims2()?;
        let src_shape = layout.shape();
        if src_shape.rank() < 2 {
            return Err(Error::Msg(format!("opencl qmatmul: input has only one dimension {layout:?}")));
        }
        let mut dst_dims = src_shape.dims().to_vec();
        let last_k = dst_dims.pop().unwrap();
        if last_k != k {
            return Err(Error::Msg(format!("opencl qmatmul: input {layout:?} incompatible with {self_shape:?}")));
        }
        dst_dims.push(n);
        let dst_shape = Shape::from(dst_dims);
        let m = src_shape.elem_count() / k;
        let out = self.device.alloc(DType::F32, m * n)?;
        let c = self.device.ctx();
        match self.dtype {
            GgmlDType::F32 => {
                // Dense weight stored [n, k]: a transposed-B GEMM / GEMV.
                let sa = MatStrides { row: k, col: 1, offset: o1, batch: 0 };
                let sb = MatStrides { row: 1, col: k, offset: (self.byte_offset / 4) as usize, batch: 0 };
                kernels::run_matmul(&c, storage.buffer, self.buffer, out.buffer, (1, m, n, k), sa, sb)?;
            }
            GgmlDType::F16 | GgmlDType::BF16 if m <= QGEMV_MAX_ROWS => {
                kernels::run_hgemv(&c, self.dtype == GgmlDType::BF16, storage.buffer, self.buffer, out.buffer, m, n, k, self.byte_offset, o1)?;
            }
            _ if m <= QGEMV_MAX_ROWS => {
                kernels::run_qgemv(&c, self.dtype, storage.buffer, self.buffer, out.buffer, m, n, k, self.byte_offset, o1)?;
            }
            _ => {
                // Prefill: dequantize once, then the tiled GEMM.
                let w = self.dequantize(n * k)?;
                let sa = MatStrides { row: k, col: 1, offset: o1, batch: 0 };
                let sb = MatStrides { row: 1, col: k, offset: 0, batch: 0 };
                kernels::run_matmul(&c, storage.buffer, w.buffer, out.buffer, (1, m, n, k), sa, sb)?;
            }
        }
        Ok((out, dst_shape))
    }

    /// Gather rows `ids` of this `[rows, hidden]` table as f32 `[n_ids, hidden]`.
    pub fn embedding(&self, rows: usize, hidden: usize, ids: &OpenClStorage, ids_l: &Layout) -> Result<OpenClStorage> {
        let (ids_s, ids_off) = ids.ids_u32(ids_l)?;
        let n_ids = ids_l.shape().elem_count();
        let out = self.device.alloc(DType::F32, n_ids * hidden)?;
        let c = self.device.ctx();
        match self.dtype {
            GgmlDType::F32 => {
                kernels::run_index_select(&c, false, false, self.buffer, ids_s.buffer, out.buffer, n_ids * hidden, 1, n_ids, hidden, rows, (self.byte_offset / 4) as usize, ids_off)?;
            }
            GgmlDType::F16 | GgmlDType::BF16 => {
                kernels::run_hembed(&c, self.dtype == GgmlDType::BF16, self.buffer, ids_s.buffer, out.buffer, n_ids, hidden, rows, self.byte_offset, ids_off)?;
            }
            _ => kernels::run_qembed(&c, self.dtype, self.buffer, ids_s.buffer, out.buffer, n_ids, hidden, rows, self.byte_offset, ids_off)?,
        }
        Ok(out)
    }
}

#[cfg(all(test, feature = "opencl"))]
mod tests {
    use super::*;
    use crate::{Device, Tensor};

    /// One device shared by every test in the module (they run on parallel
    /// threads; a context per test is not a real configuration).
    fn device() -> Option<Device> {
        static DEVICE: std::sync::OnceLock<Option<Device>> = std::sync::OnceLock::new();
        DEVICE
            .get_or_init(|| match Device::new_opencl(0) {
                Ok(d) => Some(d),
                Err(e) => {
                    eprintln!("skipping opencl test: {e}");
                    None
                }
            })
            .clone()
    }

    fn close(a: &Tensor, b: &Tensor, tol: f32, what: &str) {
        let a: Vec<f32> = a.to_device(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(a.len(), b.len(), "{what}: length");
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            let d = (x - y).abs();
            let scale = x.abs().max(y.abs()).max(1.0);
            assert!(d <= tol * scale, "{what}: element {i} differs: device {x} vs cpu {y}");
        }
    }

    /// M1 round-trip: write f32 to an OpenCL device buffer and read it back.
    #[test]
    fn f32_roundtrip_via_tensor() -> crate::Result<()> {
        let Some(dev) = device() else { return Ok(()) };
        let v: Vec<f32> = (0..1000).map(|i| i as f32 * 0.25 - 3.0).collect();
        let t = Tensor::from_vec(v.clone(), 1000, &dev)?;
        let back: Vec<f32> = t.to_device(&Device::Cpu)?.to_vec1()?;
        assert_eq!(back, v);
        Ok(())
    }

    /// Every native op agrees with the CPU backend, including strided,
    /// broadcast and offset views.
    #[test]
    fn opencl_parity_native() -> crate::Result<()> {
        let Some(dev) = device() else { return Ok(()) };
        let cpu = Device::Cpu;
        let before = kernels::native_exec_count();
        let x = Tensor::arange(0f32, 96f32, &cpu)?.reshape((4, 24))?.affine(0.05, -1.7)?;
        let y = Tensor::arange(0f32, 96f32, &cpu)?.reshape((4, 24))?.affine(-0.03, 0.9)?;
        let xd = x.to_device(&dev)?;
        let yd = y.to_device(&dev)?;

        // Elementwise on contiguous, broadcast and transposed views.
        close(&xd.affine(1.5, -0.25)?, &x.affine(1.5, -0.25)?, 1e-6, "affine");
        close(&xd.exp()?, &x.exp()?, 1e-5, "exp");
        close(&xd.t()?.contiguous()?, &x.t()?.contiguous()?, 0.0, "transpose copy");
        close(&xd.silu()?, &x.silu()?, 1e-5, "silu");
        close(&xd.gelu()?, &x.gelu()?, 1e-5, "gelu");
        close(&(&xd + &yd)?, &(&x + &y)?, 1e-6, "add");
        close(&xd.broadcast_mul(&yd.narrow(0, 0, 1)?)?, &x.broadcast_mul(&y.narrow(0, 0, 1)?)?, 1e-6, "broadcast mul");
        close(&xd.t()?.broadcast_add(&yd.narrow(1, 3, 1)?.t()?)?, &x.t()?.broadcast_add(&y.narrow(1, 3, 1)?.t()?)?, 1e-6, "strided broadcast add");
        close(&xd.narrow(1, 5, 7)?.sqr()?, &x.narrow(1, 5, 7)?.sqr()?, 1e-6, "narrow sqr");
        close(&xd.abs()?.powf(2.5)?, &x.abs()?.powf(2.5)?, 1e-5, "powf");
        close(&xd.clamp(-0.5f32, 0.5f32)?, &x.clamp(-0.5f32, 0.5f32)?, 0.0, "clamp");

        // Reductions.
        close(&xd.sum_keepdim(1)?, &x.sum_keepdim(1)?, 1e-5, "sum last");
        close(&xd.max_keepdim(1)?, &x.max_keepdim(1)?, 0.0, "max last");
        close(&xd.min_keepdim(1)?, &x.min_keepdim(1)?, 0.0, "min last");
        close(&xd.sum_keepdim(0)?, &x.sum_keepdim(0)?, 1e-5, "sum dim0 (generic)");
        close(&xd.sum_all()?, &x.sum_all()?, 1e-4, "sum all");
        close(&xd.t()?.sum_keepdim(1)?, &x.t()?.sum_keepdim(1)?, 1e-5, "sum over strided");
        let am: Vec<u32> = xd.argmax(1)?.to_device(&cpu)?.to_vec1()?;
        assert_eq!(am, x.argmax(1)?.to_vec1::<u32>()?, "argmax");

        // Comparisons, where, casts.
        let mask = xd.gt(&yd)?;
        assert_eq!(mask.to_device(&cpu)?.to_vec2::<u8>()?, x.gt(&y)?.to_vec2::<u8>()?, "gt");
        close(&mask.where_cond(&xd, &yd)?, &x.gt(&y)?.where_cond(&x, &y)?, 0.0, "where");
        let u: Vec<u32> = xd.abs()?.to_dtype(DType::U32)?.to_device(&cpu)?.flatten_all()?.to_vec1()?;
        assert_eq!(u, x.abs()?.to_dtype(DType::U32)?.flatten_all()?.to_vec1::<u32>()?, "cast u32");
        close(&xd.to_dtype(DType::F16)?.to_dtype(DType::F32)?, &x.to_dtype(DType::F16)?.to_dtype(DType::F32)?, 0.0, "f16 round trip");

        // Cat / narrow materialise through copy kernels.
        close(&Tensor::cat(&[&xd, &yd], 1)?, &Tensor::cat(&[&x, &y], 1)?, 0.0, "cat dim1");
        close(&Tensor::cat(&[&xd.narrow(0, 1, 2)?, &yd.narrow(0, 0, 1)?], 0)?, &Tensor::cat(&[&x.narrow(0, 1, 2)?, &y.narrow(0, 0, 1)?], 0)?, 0.0, "cat dim0");

        // Gathers.
        let ids = Tensor::new(&[3u32, 0, 2, 3], &cpu)?;
        close(&xd.index_select(&ids.to_device(&dev)?, 0)?, &x.index_select(&ids, 0)?, 0.0, "index_select dim0");
        close(&xd.index_select(&ids.to_device(&dev)?, 1)?, &x.index_select(&ids, 1)?, 0.0, "index_select dim1");
        let ids64 = Tensor::new(&[1i64, 1, 0], &cpu)?;
        close(&xd.index_select(&ids64.to_device(&dev)?, 0)?, &x.index_select(&ids64, 0)?, 0.0, "index_select i64");
        let gids = Tensor::new(&[[0u32, 5, 23], [1, 1, 2], [22, 0, 7], [3, 4, 5]], &cpu)?;
        close(&xd.gather(&gids.to_device(&dev)?, 1)?, &x.gather(&gids, 1)?, 0.0, "gather");
        let add_ids = Tensor::new(&[1u32, 3, 1], &cpu)?;
        let src = Tensor::arange(0f32, 72f32, &cpu)?.reshape((3, 24))?;
        close(&xd.index_add(&add_ids.to_device(&dev)?, &src.to_device(&dev)?, 0)?, &x.index_add(&add_ids, &src, 0)?, 1e-6, "index_add");
        let sc_ids = Tensor::new(&[[0u32, 1, 2], [3, 2, 1]], &cpu)?;
        let sc_src = Tensor::new(&[[10f32, 11., 12.], [13., 14., 15.]], &cpu)?;
        let base = Tensor::zeros((4, 3), DType::F32, &cpu)?;
        close(&base.to_device(&dev)?.scatter_add(&sc_ids.to_device(&dev)?, &sc_src.to_device(&dev)?, 0)?, &base.scatter_add(&sc_ids, &sc_src, 0)?, 0.0, "scatter_add");
        close(&base.to_device(&dev)?.scatter(&sc_ids.to_device(&dev)?, &sc_src.to_device(&dev)?, 0)?, &base.scatter(&sc_ids, &sc_src, 0)?, 0.0, "scatter");
        // Duplicate ids resolve like the sequential CPU loop (last write wins).
        let dup_ids = Tensor::new(&[[0u32, 1, 1], [0, 1, 1]], &cpu)?;
        close(&base.to_device(&dev)?.scatter(&dup_ids.to_device(&dev)?, &sc_src.to_device(&dev)?, 0)?, &base.scatter(&dup_ids, &sc_src, 0)?, 0.0, "scatter duplicate ids");
        close(&base.to_device(&dev)?.scatter_add(&dup_ids.to_device(&dev)?, &sc_src.to_device(&dev)?, 0)?, &base.scatter_add(&dup_ids, &sc_src, 0)?, 0.0, "scatter_add duplicate ids");
        // i64 → f32 keeps the high word.
        let big = Tensor::new(&[4_294_967_296i64, -4_294_967_297, 5, -1], &cpu)?;
        close(&big.to_device(&dev)?.to_dtype(DType::F32)?, &big.to_dtype(DType::F32)?, 0.0, "cast large i64 to f32");
        // An out-of-range id is reported at the next host read-back; the
        // device stays usable afterwards.
        let bad = Tensor::new(&[1u32, 9], &dev)?;
        assert!(xd.index_select(&bad, 0)?.to_device(&cpu).is_err(), "out-of-range index_select must fail");
        // An i64 id beyond u32 (or negative) is out of range, not truncated.
        let big_id = Tensor::new(&[4_294_967_297i64, 1], &dev)?;
        assert!(xd.gather(&big_id.reshape((2, 1))?.broadcast_as((2, 24))?.contiguous()?, 0).and_then(|t| t.to_device(&cpu)).is_err(), "i64 id beyond u32 must fail");
        let neg_id = Tensor::new(&[-1i64, 1], &dev)?;
        assert!(xd.index_select(&neg_id, 0)?.to_device(&cpu).is_err(), "negative i64 id must fail");
        close(&xd.index_select(&ids.to_device(&dev)?, 0)?, &x.index_select(&ids, 0)?, 0.0, "index_select after a fault");

        // Matmul: NN, NT (weight transpose), batched with broadcast rhs, decode GEMV.
        let a = Tensor::arange(0f32, 24f32 * 40f32, &cpu)?.reshape((24, 40))?.affine(1e-3, -0.4)?;
        let w = Tensor::arange(0f32, 40f32 * 17f32, &cpu)?.reshape((17, 40))?.affine(-2e-3, 0.3)?;
        let (ad, wd) = (a.to_device(&dev)?, w.to_device(&dev)?);
        close(&ad.matmul(&wd.t()?)?, &a.matmul(&w.t()?)?, 1e-4, "gemm NT");
        close(&ad.t()?.contiguous()?.t()?.matmul(&wd.t()?)?, &a.matmul(&w.t()?)?, 1e-4, "gemm NT strided lhs");
        close(&wd.matmul(&ad.t()?)?, &w.matmul(&a.t()?)?, 1e-4, "gemm NT 2");
        let b = Tensor::arange(0f32, 40f32 * 9f32, &cpu)?.reshape((40, 9))?.affine(3e-3, -0.2)?;
        close(&ad.matmul(&b.to_device(&dev)?)?, &a.matmul(&b)?, 1e-4, "gemm NN");
        let q = Tensor::arange(0f32, 2f32 * 3f32 * 40f32, &cpu)?.reshape((2, 3, 1, 40))?.affine(1e-2, -0.5)?;
        let kk = Tensor::arange(0f32, 2f32 * 3f32 * 5f32 * 40f32, &cpu)?.reshape((2, 3, 5, 40))?.affine(-1e-2, 0.2)?;
        close(&q.to_device(&dev)?.matmul(&kk.to_device(&dev)?.transpose(2, 3)?)?, &q.matmul(&kk.transpose(2, 3)?)?, 1e-4, "batched decode attention QK^T");
        let scores = q.matmul(&kk.transpose(2, 3)?)?;
        close(&scores.to_device(&dev)?.matmul(&kk.to_device(&dev)?)?, &scores.matmul(&kk)?, 1e-4, "batched decode attention scores V");
        let row = a.narrow(0, 2, 1)?;
        close(&row.to_device(&dev)?.matmul(&wd.t()?)?, &row.matmul(&w.t()?)?, 1e-4, "gemv NT");
        close(&row.to_device(&dev)?.matmul(&b.to_device(&dev)?)?, &row.matmul(&b)?, 1e-4, "gemv NN");
        let big = Tensor::arange(0f32, 130f32 * 70f32, &cpu)?.reshape((130, 70))?.affine(1e-3, -0.05)?;
        let bw = Tensor::arange(0f32, 70f32 * 131f32, &cpu)?.reshape((131, 70))?.affine(-1e-3, 0.04)?;
        close(&big.to_device(&dev)?.matmul(&bw.to_device(&dev)?.t()?)?, &big.matmul(&bw.t()?)?, 1e-4, "gemm edges");

        // Fused fills / clone.
        let z = Tensor::zeros((3, 5), DType::F32, &dev)?;
        close(&z, &Tensor::zeros((3, 5), DType::F32, &cpu)?, 0.0, "zeros");
        close(&xd.copy()?, &x, 0.0, "clone");

        let n = kernels::native_exec_count() - before;
        eprintln!("opencl parity: {n} native kernel launches, {} fallbacks", kernels::fallback_count());
        assert!(n >= 40, "native kernels must actually run (got {n})");
        Ok(())
    }

    /// Block-quantized weights: dequantize, GEMV, GEMM and embedding gather
    /// on the device match candle's CPU reference for every block format.
    #[test]
    fn opencl_quantized_parity() -> crate::Result<()> {
        use crate::quantized::{GgmlDType, QMatMul, QTensor};
        let Some(dev) = device() else { return Ok(()) };
        let cpu = Device::Cpu;
        let (n, k) = (12usize, 512usize);
        let w = Tensor::arange(0f32, (n * k) as f32, &cpu)?.reshape((n, k))?.affine(7e-4, -2.1)?.sin()?;
        let x1 = Tensor::arange(0f32, k as f32, &cpu)?.reshape((1, k))?.affine(3e-3, -0.7)?.cos()?;
        let xm = Tensor::arange(0f32, (40 * k) as f32, &cpu)?.reshape((40, k))?.affine(1e-3, -0.9)?.sin()?;
        for dtype in [
            GgmlDType::F32, GgmlDType::F16, GgmlDType::BF16, GgmlDType::Q8_0, GgmlDType::Q8_1, GgmlDType::Q4_0, GgmlDType::Q4_1,
            GgmlDType::Q5_0, GgmlDType::Q5_1, GgmlDType::Q2K, GgmlDType::Q3K, GgmlDType::Q4K, GgmlDType::Q5K, GgmlDType::Q6K, GgmlDType::Q8K,
        ] {
            let q_cpu = QTensor::quantize(&w, dtype)?;
            let bytes = q_cpu.data()?.into_owned();
            let q_dev = QTensor::new(crate::quantized::QStorage::from_data(std::borrow::Cow::Borrowed(&bytes), &dev, dtype)?, (n, k))?;
            let what = format!("{dtype:?}");
            close(&q_dev.dequantize(&dev)?, &q_cpu.dequantize(&cpu)?, 1e-6, &format!("{what} dequantize"));
            // The device multiplies f32 activations against the dequantized
            // weights, so it should agree tightly with an f32 reference.  The
            // CPU quantized path quantizes the activations to 8 bits first,
            // which perturbs each dot product by roughly 1e-3 relative; the
            // CPU comparison is a loose sanity check only.
            let w_ref = q_cpu.dequantize(&cpu)?;
            let mm_cpu = QMatMul::from_qtensor(q_cpu)?;
            let mm_dev = QMatMul::from_qtensor(q_dev)?;
            use crate::Module;
            let ref1 = x1.matmul(&w_ref.t()?)?;
            let refm = xm.matmul(&w_ref.t()?)?;
            let dev1 = mm_dev.forward(&x1.to_device(&dev)?)?;
            let devm = mm_dev.forward(&xm.to_device(&dev)?)?;
            close(&dev1, &ref1, 1e-4, &format!("{what} gemv vs f32"));
            close(&devm, &refm, 1e-4, &format!("{what} gemm vs f32"));
            close(&dev1, &mm_cpu.forward(&x1)?, 2e-2, &format!("{what} gemv vs cpu"));
            close(&devm, &mm_cpu.forward(&xm)?, 2e-2, &format!("{what} gemm vs cpu"));
            let ids = Tensor::new(&[3u32, 0, 11, 3], &cpu)?;
            close(&mm_dev.embedding(&ids.to_device(&dev)?)?, &mm_cpu.embedding(&ids)?, 1e-6, &format!("{what} embedding"));
        }
        Ok(())
    }

    /// A zero-copy host mapping serves the same numbers as an uploaded copy.
    #[test]
    fn opencl_zero_copy_weights() -> crate::Result<()> {
        use crate::quantized::{GgmlDType, QMatMul, QTensor};
        let Some(dev) = device() else { return Ok(()) };
        let cpu = Device::Cpu;
        let (n, k) = (8usize, 256usize);
        let w = Tensor::arange(0f32, (n * k) as f32, &cpu)?.reshape((n, k))?.affine(1e-3, -1.0)?;
        let q_cpu = QTensor::quantize(&w, GgmlDType::Q4K)?;
        let bytes = q_cpu.data()?.into_owned();
        // Place the tensor at an unaligned offset inside a larger host block.
        let mut host = vec![0u8; 3 * page_size() + bytes.len() + 100];
        let off = page_size() + 96;
        host[off..off + bytes.len()].copy_from_slice(&bytes);
        let host: Arc<Vec<u8>> = Arc::new(host);
        let ocl = dev.as_opencl_device()?;
        let storage = unsafe {
            QOpenClStorage::from_host_mapping(ocl, GgmlDType::Q4K, n * k, host.as_ptr(), host.len(), off, host.clone())
        }?;
        assert!(storage.is_zero_copy());
        let q_dev = QTensor::new(crate::quantized::QStorage::OpenCl(storage), (n, k))?;
        close(&q_dev.dequantize(&dev)?, &q_cpu.dequantize(&cpu)?, 0.0, "zero-copy dequantize");
        let x = Tensor::arange(0f32, k as f32, &cpu)?.reshape((1, k))?.affine(2e-3, -0.3)?;
        use crate::Module;
        let w_ref = q_cpu.dequantize(&cpu)?;
        close(&QMatMul::from_qtensor(q_dev)?.forward(&x.to_device(&dev)?)?, &x.matmul(&w_ref.t()?)?, 1e-4, "zero-copy gemv");
        Ok(())
    }

    /// Importance-weighted quantization on the device yields the CPU's blocks.
    #[test]
    fn opencl_imatrix_quantize() -> crate::Result<()> {
        let Some(dev) = device() else { return Ok(()) };
        let cpu = Device::Cpu;
        let x = Tensor::arange(0f32, 512f32, &cpu)?.affine(0.01, -2.0)?.reshape((2, 256))?;
        let w: Vec<f32> = (0..256).map(|i| 1.0 + (i % 7) as f32).collect();
        for dtype in [GgmlDType::Q4K, GgmlDType::Q6K] {
            let q_cpu = crate::quantized::QTensor::quantize_imatrix(&x, &w, dtype)?;
            let q_dev = crate::quantized::QTensor::quantize_imatrix(&x.to_device(&dev)?, &w, dtype)?;
            assert_eq!(q_cpu.data()?.as_ref(), q_dev.data()?.as_ref(), "{dtype:?}: imatrix blocks");
            let q_onto = crate::quantized::QTensor::quantize_imatrix_onto(&x, &w, dtype, &dev)?;
            assert_eq!(q_cpu.data()?.as_ref(), q_onto.data()?.as_ref(), "{dtype:?}: imatrix blocks (onto)");
        }
        Ok(())
    }

    /// A fault raised by one thread's launch is reported to that thread only:
    /// a thread doing valid work on the same device never sees it.
    #[test]
    fn opencl_fault_is_per_thread() -> crate::Result<()> {
        let Some(dev) = device() else { return Ok(()) };
        let cpu = Device::Cpu;
        let x = Tensor::arange(0f32, 32f32, &cpu)?.reshape((4, 8))?.to_device(&dev)?;
        let bad = Tensor::new(&[1u32, 9], &dev)?;
        let good = Tensor::new(&[1u32, 3], &dev)?;
        let faulty = std::thread::spawn({
            let (x, bad, cpu) = (x.clone(), bad.clone(), cpu.clone());
            move || (0..20).all(|_| x.index_select(&bad, 0).and_then(|t| t.to_device(&cpu)).is_err())
        });
        let clean = std::thread::spawn({
            let (x, good, cpu) = (x.clone(), good.clone(), cpu.clone());
            move || (0..200).all(|_| x.index_select(&good, 0).and_then(|t| t.to_device(&cpu)).is_ok())
        });
        assert!(faulty.join().unwrap(), "the faulting thread must see its error every time");
        assert!(clean.join().unwrap(), "the clean thread must never see another thread's fault");
        Ok(())
    }

    /// A thread that exits with a faulting launch still in flight does not
    /// hand its fault to the next owner of its slot.
    #[test]
    fn opencl_fault_slot_is_drained_on_thread_exit() -> crate::Result<()> {
        let Some(dev) = device() else { return Ok(()) };
        let cpu = Device::Cpu;
        let x = Tensor::arange(0f32, 32f32, &cpu)?.reshape((4, 8))?.to_device(&dev)?;
        let bad = Tensor::new(&[1u32, 9], &dev)?;
        let good = Tensor::new(&[1u32, 3], &dev)?;
        for _ in 0..4 {
            std::thread::spawn({
                let (x, bad) = (x.clone(), bad.clone());
                // Launch and drop without any read-back, then exit.
                move || drop(x.index_select(&bad, 0))
            })
            .join()
            .unwrap();
            let clean = std::thread::spawn({
                let (x, good, cpu) = (x.clone(), good.clone(), cpu.clone());
                move || (0..20).all(|_| x.index_select(&good, 0).and_then(|t| t.to_device(&cpu)).is_ok())
            });
            assert!(clean.join().unwrap(), "a recycled slot must come back clean");
        }
        Ok(())
    }
}
