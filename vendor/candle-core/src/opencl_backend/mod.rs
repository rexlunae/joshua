//! OpenCL backend for candle-core.
//!
//! Enabled with the `opencl` feature. This is a hand-rolled FFI to the system
//! OpenCL ICD loader (`-lOpenCL`, i.e. `libOpenCL.so.1`), following the
//! "dependency-lean" recommendation in docs/opencl-backend-plan.md (as cudarc
//! binds CUDA, we bind OpenCL directly).
//!
//! M1 scope (this module): device + storage types wired into the `Device` /
//! `Storage` / `DeviceLocation` enums, with a working **host<->device buffer
//! round-trip** (the raw FFI was verified on the joshua host's Intel UHD 730
//! iGPU; see docs/ocl_roundtrip.c). Compute operators (matmul, elementwise,
//! conv, ...) are explicit M2 work and return a clear "not implemented" error
//! for now rather than a silent wrong result.
#![allow(clippy::missing_safety_doc)]

pub mod kernels;

use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};

/// OpenCL constants we care about (CL 1.2 core subset).
mod cl {
    pub const CL_SUCCESS: i32 = 0;
    pub const CL_DEVICE_TYPE_GPU: u64 = 1 << 2;
    pub const CL_MEM_READ_WRITE: u64 = 1 << 0;
    pub const CL_TRUE: u8 = 1;
}

// --- Minimal hand-rolled OpenCL FFI (CL 1.2 buffer-transfer subset). ---
// The `#[link]` directive makes rustc link against `-lOpenCL` (the ICD loader,
// libOpenCL.so on Linux / OpenCL.framework or libOpenCL on macOS).
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
    fn clFinish(command_queue: usize) -> i32;
    fn clReleaseMemObject(memobj: usize) -> i32;
    fn clReleaseCommandQueue(command_queue: usize) -> i32;
    fn clReleaseContext(context: usize) -> i32;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

/// An OpenCL device: a `cl_context` + `cl_command_queue` pair for one GPU.
#[derive(Debug)]
pub struct OpenClDevice {
    gpu_id: usize,
    context: usize,
    queue: usize,
    #[allow(dead_code)]
    platform_id: usize,
    #[allow(dead_code)]
    device_id: usize,
}

fn opencl_error(code: i32, op: &str) -> Error {
    Error::Msg(format!("opencl {op} failed with status {code}"))
}

/// Connect to the first OpenCL GPU and build a context + queue.
fn init_device(gpu_id: usize) -> Result<(usize, usize, usize, usize)> {
    let mut num_platforms: u32 = 0;
    let mut platforms = [0usize; 4];
    let rc = unsafe { clGetPlatformIDs(4, platforms.as_mut_ptr(), &mut num_platforms) };
    if rc != cl::CL_SUCCESS || num_platforms == 0 {
        return Err(Error::Msg("opencl: no OpenCL platform found".to_string()));
    }
    for i in 0..num_platforms.min(4) as usize {
        let mut num_devices: u32 = 0;
        let mut devices = [0usize; 8];
        let d_rc = unsafe {
            clGetDeviceIDs(
                platforms[i],
                cl::CL_DEVICE_TYPE_GPU,
                8,
                devices.as_mut_ptr(),
                &mut num_devices,
            )
        };
        if d_rc == cl::CL_SUCCESS && num_devices > 0 {
            let dev_idx = gpu_id.min(num_devices as usize - 1);
            let device = devices[dev_idx];
            let mut err: i32 = 0;
            let context = unsafe {
                clCreateContext(
                    std::ptr::null(),
                    1,
                    &device,
                    None,
                    std::ptr::null_mut(),
                    &mut err,
                )
            };
            if err != cl::CL_SUCCESS || context == 0 {
                return Err(opencl_error(err, "clCreateContext"));
            }
            let queue = unsafe { clCreateCommandQueue(context, device, 0, &mut err) };
            if err != cl::CL_SUCCESS || queue == 0 {
                return Err(opencl_error(err, "clCreateCommandQueue"));
            }
            return Ok((context, queue, platforms[i], device));
        }
    }
    Err(Error::Msg("opencl: no OpenCL GPU device found".to_string()))
}

impl Drop for OpenClDevice {
    fn drop(&mut self) {
        if self.queue != 0 {
            unsafe { clReleaseCommandQueue(self.queue) };
        }
        if self.context != 0 {
            unsafe { clReleaseContext(self.context) };
        }
    }
}

impl Clone for OpenClDevice {
    fn clone(&self) -> Self {
        match init_device(self.gpu_id) {
            Ok((context, queue, platform_id, device_id)) => Self {
                gpu_id: self.gpu_id,
                context,
                queue,
                platform_id,
                device_id,
            },
            Err(_) => Self {
                gpu_id: self.gpu_id,
                context: 0,
                queue: 0,
                platform_id: self.platform_id,
                device_id: self.device_id,
            },
        }
    }
}

impl OpenClDevice {
    pub fn new(gpu_id: usize) -> Result<Self> {
        let (context, queue, platform_id, device_id) = init_device(gpu_id)?;
        Ok(Self {
            gpu_id,
            context,
            queue,
            platform_id,
            device_id,
        })
    }

    pub fn new_with_stream(gpu_id: usize) -> Result<Self> {
        Self::new(gpu_id)
    }

    pub fn id(&self) -> DeviceId {
        DeviceId(self.gpu_id)
    }

    pub(crate) fn queue(&self) -> usize {
        self.queue
    }

    pub(crate) fn context(&self) -> usize {
        self.context
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
        let buffer = create_buffer(device.context(), bytes, cl::CL_MEM_READ_WRITE)?;
        // # Safety: slice.as_ptr() points to `bytes` valid bytes.
        let data = unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, bytes) };
        unsafe { write_buffer(device.queue(), buffer, bytes, data.as_ptr()) }?;
        Ok(Self {
            buffer,
            dtype,
            numel: slice.len(),
            device: device.clone(),
        })
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
    let rc = clEnqueueWriteBuffer(
        queue,
        buffer,
        cl::CL_TRUE,
        0,
        bytes,
        ptr as *const std::ffi::c_void,
        0,
        std::ptr::null(),
        std::ptr::null_mut(),
    );
    if rc != cl::CL_SUCCESS {
        return Err(opencl_error(rc, "clEnqueueWriteBuffer"));
    }
    Ok(())
}

/// # Safety
/// `ptr` must point to at least `bytes` valid bytes.
unsafe fn read_buffer(queue: usize, buffer: usize, bytes: usize, ptr: *mut u8) -> Result<()> {
    let rc = clEnqueueReadBuffer(
        queue,
        buffer,
        cl::CL_TRUE,
        0,
        bytes,
        ptr as *mut std::ffi::c_void,
        0,
        std::ptr::null(),
        std::ptr::null_mut(),
    );
    if rc != cl::CL_SUCCESS {
        return Err(opencl_error(rc, "clEnqueueReadBuffer"));
    }
    Ok(())
}

impl BackendStorage for OpenClStorage {
    type Device = OpenClDevice;

    fn try_clone(&self, _layout: &Layout) -> Result<Self> {
        // Whole-buffer clone via a host round-trip (M1; no P2P).
        let cpu = self.to_cpu_storage()?;
        self.device.storage_from_cpu_storage(&cpu)
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn const_set(&mut self, s: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        let mut cpu = self.to_cpu_storage()?;
        cpu.const_set(s, layout)?;
        let dev = self.device.clone();
        *self = dev.storage_from_cpu_storage(&cpu)?;
        Ok(())
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        let bytes = self.dtype.size_in_bytes() * self.numel;
        // Read the raw device bytes back into a CPU buffer, then transpose into
        // the matching CpuStorage variant.
        let mut raw = vec![0u8; bytes];
        unsafe { read_buffer(self.device.queue(), self.buffer, bytes, raw.as_mut_ptr()) }?;
        Ok(match self.dtype {
            DType::U8 => {
                let mut v = vec![0u8; self.numel];
                v.copy_from_slice(&raw);
                CpuStorage::U8(v)
            }
            DType::U32 => CpuStorage::U32(transmute_bytes(&raw, self.numel)),
            DType::I16 => CpuStorage::I16(transmute_bytes(&raw, self.numel)),
            DType::I32 => CpuStorage::I32(transmute_bytes(&raw, self.numel)),
            DType::I64 => CpuStorage::I64(transmute_bytes(&raw, self.numel)),
            DType::F32 => CpuStorage::F32(transmute_bytes(&raw, self.numel)),
            DType::F64 => CpuStorage::F64(transmute_bytes(&raw, self.numel)),
            _ => {
                return Err(Error::Msg(format!(
                    "opencl to_cpu_storage: dtype {:?} not supported in M1",
                    self.dtype
                )))
            }
        })
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        // M5 native kernel path (opt-in, contiguous f32): run on the iGPU.
        if kernels::native_enabled() && layout.is_contiguous() && self.dtype == DType::F32 {
            let n = self.numel;
            let out_buf = create_buffer(self.device.context(), n * 4, cl::CL_MEM_READ_WRITE)?;
            match kernels::run_affine(
                self.device.context(),
                self.device.device_id,
                self.device.queue(),
                self.buffer,
                out_buf,
                n,
                mul as f32,
                add as f32,
            ) {
                Ok(()) => {
                    return Ok(OpenClStorage {
                        buffer: out_buf,
                        dtype: self.dtype,
                        numel: self.numel,
                        device: self.device.clone(),
                    });
                }
                Err(_) => {
                    unsafe { clReleaseMemObject(out_buf) };
                    // fall through to CPU fallback below
                }
            }
        }
        let cpu = self.to_cpu_storage()?;
        let out = cpu.affine(layout, mul, add)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let out = cpu.powf(layout, e)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let out = cpu.elu(layout, alpha)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, s: &[usize]) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let out = cpu.reduce_op(op, layout, s)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let lhs = self.to_cpu_storage()?;
        let rhs = rhs.to_cpu_storage()?;
        let out = lhs.cmp(op, &rhs, lhs_l, rhs_l)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let out = cpu.to_dtype(layout, dtype)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        if kernels::native_enabled()
            && layout.is_contiguous()
            && self.dtype == DType::F32
            && kernels::has_unary(B::NAME)
        {
            let n = self.numel;
            let out_buf = create_buffer(self.device.context(), n * 4, cl::CL_MEM_READ_WRITE)?;
            match kernels::run_unary(
                self.device.context(),
                self.device.device_id,
                self.device.queue(),
                B::NAME,
                self.buffer,
                out_buf,
                n,
            ) {
                Ok(()) => {
                    return Ok(OpenClStorage {
                        buffer: out_buf,
                        dtype: self.dtype,
                        numel: self.numel,
                        device: self.device.clone(),
                    });
                }
                Err(_) => {
                    unsafe { clReleaseMemObject(out_buf) };
                }
            }
        }
        let cpu = self.to_cpu_storage()?;
        let out = cpu.unary_impl::<B>(layout)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn binary_impl<B: BinaryOpT>(&self, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        if kernels::native_enabled()
            && lhs_l.is_contiguous()
            && rhs_l.is_contiguous()
            && self.dtype == DType::F32
            && rhs.dtype == DType::F32
            && self.numel == rhs.numel
            && kernels::has_binary(B::NAME)
        {
            let n = self.numel;
            let out_buf = create_buffer(self.device.context(), n * 4, cl::CL_MEM_READ_WRITE)?;
            match kernels::run_binary(
                self.device.context(),
                self.device.device_id,
                self.device.queue(),
                B::NAME,
                self.buffer,
                rhs.buffer,
                out_buf,
                n,
            ) {
                Ok(()) => {
                    return Ok(OpenClStorage {
                        buffer: out_buf,
                        dtype: self.dtype,
                        numel: self.numel,
                        device: self.device.clone(),
                    });
                }
                Err(_) => {
                    unsafe { clReleaseMemObject(out_buf) };
                }
            }
        }
        let lhs = self.to_cpu_storage()?;
        let rhs = rhs.to_cpu_storage()?;
        let out = lhs.binary_impl::<B>(&rhs, lhs_l, rhs_l)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        let cond = self.to_cpu_storage()?;
        let t = t.to_cpu_storage()?;
        let f = f.to_cpu_storage()?;
        let out = cond.where_cond(layout, &t, t_l, &f, f_l)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        let inp = self.to_cpu_storage()?;
        let kernel = kernel.to_cpu_storage()?;
        let out = inp.conv1d(l, &kernel, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        let inp = self.to_cpu_storage()?;
        let kernel = kernel.to_cpu_storage()?;
        let out = inp.conv_transpose1d(l, &kernel, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        let inp = self.to_cpu_storage()?;
        let kernel = kernel.to_cpu_storage()?;
        let out = inp.conv2d(l, &kernel, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        let inp = self.to_cpu_storage()?;
        let kernel = kernel.to_cpu_storage()?;
        let out = inp.conv_transpose2d(l, &kernel, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        let src = self.to_cpu_storage()?;
        let ids = ids.to_cpu_storage()?;
        let out = src.index_select(&ids, l, ids_l, dim)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        let src = self.to_cpu_storage()?;
        let ids = ids.to_cpu_storage()?;
        let out = src.gather(l, &ids, ids_l, dim)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn scatter_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let mut tgt = self.to_cpu_storage()?;
        let ids = ids.to_cpu_storage()?;
        let src = src.to_cpu_storage()?;
        tgt.scatter_set(l, &ids, ids_l, &src, src_l, dim)?;
        let dev = self.device.clone();
        *self = dev.storage_from_cpu_storage(&tgt)?;
        Ok(())
    }

    fn scatter_add_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let mut tgt = self.to_cpu_storage()?;
        let ids = ids.to_cpu_storage()?;
        let src = src.to_cpu_storage()?;
        tgt.scatter_add_set(l, &ids, ids_l, &src, src_l, dim)?;
        let dev = self.device.clone();
        *self = dev.storage_from_cpu_storage(&tgt)?;
        Ok(())
    }

    fn index_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let tgt = self.to_cpu_storage()?;
        let ids = ids.to_cpu_storage()?;
        let src = src.to_cpu_storage()?;
        let out = tgt.index_add(l, &ids, ids_l, &src, src_l, dim)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let lhs = self.to_cpu_storage()?;
        let rhs = rhs.to_cpu_storage()?;
        let out = lhs.matmul(&rhs, bmnk, lhs_l, rhs_l)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let src = self.to_cpu_storage()?;
        let mut dst_cpu = dst.to_cpu_storage()?;
        src.copy_strided_src(&mut dst_cpu, dst_offset, src_l)?;
        let dev = dst.device.clone();
        *dst = dev.storage_from_cpu_storage(&dst_cpu)?;
        Ok(())
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
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

    fn upsample_bilinear2d(
        &self,
        layout: &Layout,
        h: usize,
        w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let out = cpu.upsample_bilinear2d(layout, h, w, align_corners, scale_h, scale_w)?;
        self.device.storage_from_cpu_storage(&out)
    }
}

/// Reinterpret `raw` (a byte buffer of `numel * size` bytes) as a `Vec<T>` by
/// copying. The caller must guarantee `raw.len() == numel * size_of::<T>()`
/// and that the bytes are a valid bit pattern for `T`.
fn transmute_bytes<T: Copy>(raw: &[u8], numel: usize) -> Vec<T> {
    let t = std::mem::size_of::<T>();
    debug_assert_eq!(raw.len(), numel * t);
    let mut v = vec![unsafe { std::mem::zeroed() }; numel];
    let dst = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, numel * t) };
    dst.copy_from_slice(raw);
    v
}

impl BackendDevice for OpenClDevice {
    type Storage = OpenClStorage;

    fn new(gpu_id: usize) -> Result<Self> {
        Self::new(gpu_id)
    }

    fn set_seed(&self, _seed: u64) -> Result<()> {
        crate::bail!("opencl set_seed not implemented yet (M2)")
    }

    fn get_current_seed(&self) -> Result<u64> {
        crate::bail!("opencl get_current_seed not implemented yet (M2)")
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::OpenCl { gpu_id: self.gpu_id }
    }

    fn same_device(&self, other: &Self) -> bool {
        self.gpu_id == other.gpu_id
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let numel = shape.elem_count();
        let storage = unsafe { self.alloc_uninit(shape, dtype) }?;
        if numel > 0 && dtype.size_in_bytes() > 0 {
            let bytes = numel * dtype.size_in_bytes();
            let zeros = vec![0u8; bytes];
            unsafe { write_buffer(self.queue, storage.buffer, bytes, zeros.as_ptr()) }?;
        }
        Ok(storage)
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let numel = shape.elem_count();
        if dtype.size_in_bytes() == 0 {
            return Err(Error::Msg(
                "opencl alloc_uninit: unsupported (sub-byte) dtype".into(),
            ));
        }
        let bytes = numel * dtype.size_in_bytes();
        let buffer = create_buffer(self.context, bytes, cl::CL_MEM_READ_WRITE)?;
        Ok(OpenClStorage {
            buffer,
            dtype,
            numel,
            device: self.clone(),
        })
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        OpenClStorage::from_vec(s.to_vec(), self)
    }

    fn storage_from_cpu_storage(&self, cpu: &CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage_owned(cpu.clone())
    }

    fn storage_from_cpu_storage_owned(&self, cpu: CpuStorage) -> Result<Self::Storage> {
        let dtype = cpu.dtype();
        match &cpu {
            CpuStorage::U8(v) => OpenClStorage::from_vec(v.clone(), self),
            CpuStorage::U32(v) => OpenClStorage::from_vec(v.clone(), self),
            CpuStorage::I16(v) => OpenClStorage::from_vec(v.clone(), self),
            CpuStorage::I32(v) => OpenClStorage::from_vec(v.clone(), self),
            CpuStorage::I64(v) => OpenClStorage::from_vec(v.clone(), self),
            CpuStorage::F32(v) => OpenClStorage::from_vec(v.clone(), self),
            CpuStorage::F64(v) => OpenClStorage::from_vec(v.clone(), self),
            _ => Err(Error::Msg(format!(
                "opencl storage_from_cpu_storage: dtype {:?} not supported in M1",
                dtype
            ))),
        }
    }

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        crate::bail!("opencl rand_uniform not implemented yet (M2)")
    }

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        crate::bail!("opencl rand_normal not implemented yet (M2)")
    }

    fn synchronize(&self) -> Result<()> {
        let rc = unsafe { clFinish(self.queue) };
        if rc != cl::CL_SUCCESS {
            return Err(opencl_error(rc, "clFinish"));
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "opencl"))]
mod tests {
    use super::*;
    use crate::{Device, Tensor};

    /// M1 round-trip: write f32 to an OpenCL device buffer and read it back.
    ///
    /// Marked `#[ignore]` so it is skipped in normal CI / GPU-less runs; execute on
    /// an OpenCL machine (the joshua host's Intel iGPU) with:
    ///   cargo test --features opencl -p candle-core -- --ignored opencl
    #[test]
    #[ignore]
    fn f32_roundtrip_via_tensor() -> crate::Result<()> {
        let dev = match crate::OpenClDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping opencl round-trip: {e}");
                return Ok(());
            }
        };
        let dev = Device::OpenCl(dev);
        let a = Tensor::from_vec(vec![1.5f32, 2.0, 3.0, -4.5], (4,), &dev)?;
        let cpu = a.to_device(&Device::Cpu)?;
        let v = cpu.to_vec1::<f32>()?;
        assert_eq!(v, vec![1.5f32, 2.0, 3.0, -4.5]);
        Ok(())
    }
}
