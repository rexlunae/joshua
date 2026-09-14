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

    fn const_set(&mut self, _s: crate::scalar::Scalar, _layout: &Layout) -> Result<()> {
        crate::bail!("opencl const_set not implemented yet (M2)")
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

    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self> {
        crate::bail!("opencl affine not implemented yet (M2)")
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        crate::bail!("opencl powf not implemented yet (M2)")
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        crate::bail!("opencl elu not implemented yet (M2)")
    }

    fn reduce_op(&self, _: ReduceOp, _: &Layout, _: &[usize]) -> Result<Self> {
        crate::bail!("opencl reduce_op not implemented yet (M2)")
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        crate::bail!("opencl cmp not implemented yet (M2)")
    }

    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
        crate::bail!("opencl to_dtype not implemented yet (M2)")
    }

    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self> {
        crate::bail!("opencl unary not implemented yet (M2)")
    }

    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        crate::bail!("opencl binary not implemented yet (M2)")
    }

    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        crate::bail!("opencl where_cond not implemented yet (M2)")
    }

    fn conv1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        crate::bail!("opencl conv1d not implemented yet (M2)")
    }

    fn conv_transpose1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        crate::bail!("opencl conv_transpose1d not implemented yet (M2)")
    }

    fn conv2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        crate::bail!("opencl conv2d not implemented yet (M2)")
    }

    fn conv_transpose2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        crate::bail!("opencl conv_transpose2d not implemented yet (M2)")
    }

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        crate::bail!("opencl index_select not implemented yet (M2)")
    }

    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        crate::bail!("opencl gather not implemented yet (M2)")
    }

    fn scatter_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()> {
        crate::bail!("opencl scatter_set not implemented yet (M2)")
    }

    fn scatter_add_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()> {
        crate::bail!("opencl scatter_add_set not implemented yet (M2)")
    }

    fn index_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        crate::bail!("opencl index_add not implemented yet (M2)")
    }

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        crate::bail!("opencl matmul not implemented yet (M2)")
    }

    fn copy_strided_src(&self, _dst: &mut Self, _dst_offset: usize, _src_l: &Layout) -> Result<()> {
        crate::bail!("opencl copy_strided_src not implemented yet (M2)")
    }

    fn copy2d(
        &self,
        _: &mut Self,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
    ) -> Result<()> {
        crate::bail!("opencl copy2d not implemented yet (M2)")
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        crate::bail!("opencl avg_pool2d not implemented yet (M2)")
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        crate::bail!("opencl max_pool2d not implemented yet (M2)")
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        crate::bail!("opencl upsample_nearest1d not implemented yet (M2)")
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        crate::bail!("opencl upsample_nearest2d not implemented yet (M2)")
    }

    fn upsample_bilinear2d(
        &self,
        _: &Layout,
        _: usize,
        _: usize,
        _: bool,
        _: Option<f64>,
        _: Option<f64>,
    ) -> Result<Self> {
        crate::bail!("opencl upsample_bilinear2d not implemented yet (M2)")
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
