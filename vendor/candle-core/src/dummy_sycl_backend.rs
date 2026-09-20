//! Implementation of the OpenCL backend when OpenCL support has not been compiled in.
//!
#![allow(dead_code)]
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};

#[derive(Debug, Clone)]
pub struct SyclDevice;

#[derive(Debug)]
pub struct SyclStorage {
    #[allow(dead_code)]
    pub numel: usize,
    #[allow(dead_code)]
    pub dtype: DType,
    #[allow(dead_code)]
    pub device: SyclDevice,
}

impl SyclStorage {
    pub fn transfer_to_device(&self, _dst: &SyclDevice) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    pub fn from_vec<T: crate::WithDType>(_slice: Vec<T>, _device: &SyclDevice) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
    }
}

macro_rules! fail {
    () => {
        unimplemented!("sycl support has not been enabled, add `SYCL` feature to enable.")
    };
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl SyclDevice {
    pub fn new_with_stream(_: usize) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
    }
    pub fn name(&self) -> &str {
        "sycl (not compiled)"
    }
    pub fn global_mem_size(&self) -> Option<u64> {
        None
    }
    pub fn id(&self) -> DeviceId {
        DeviceId(0)
    }
}

impl crate::backend::BackendStorage for SyclStorage {
    type Device = SyclDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn dtype(&self) -> DType {
        fail!()
    }

    fn device(&self) -> &Self::Device {
        fail!()
    }

    fn const_set(&mut self, _: crate::scalar::Scalar, _: &Layout) -> Result<()> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn reduce_op(&self, _: ReduceOp, _: &Layout, _: &[usize]) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn conv1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn conv_transpose1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn conv2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
    }
    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
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
        Err(Error::NotCompiledWithSyclSupport)
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
        Err(Error::NotCompiledWithSyclSupport)
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
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        Err(Error::NotCompiledWithSyclSupport)
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
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
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
        Err(Error::NotCompiledWithSyclSupport)
    }
}

impl crate::backend::BackendDevice for SyclDevice {
    type Storage = SyclStorage;
    fn new(_: usize) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn set_seed(&self, _: u64) -> Result<()> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn get_current_seed(&self) -> Result<u64> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn location(&self) -> crate::DeviceLocation {
        fail!()
    }

    fn same_device(&self, _: &Self) -> bool {
        fail!()
    }

    fn zeros_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    unsafe fn alloc_uninit(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn storage_from_cpu_storage_owned(&self, _: CpuStorage) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithSyclSupport)
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with f16 GEMMs.
pub fn gemm_reduced_precision_f16() -> bool {
    true
}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with f16 GEMMs.
pub fn set_gemm_reduced_precision_f16(_: bool) {}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with bf16 GEMMs.
pub fn gemm_reduced_precision_bf16() -> bool {
    true
}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with bf16 GEMMs.
pub fn set_gemm_reduced_precision_bf16(_: bool) {}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with f32 GEMMs.
pub fn gemm_reduced_precision_f32() -> bool {
    true
}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with f32 GEMMs.
pub fn set_gemm_reduced_precision_f32(_b: bool) {}

/// Block-quantized OpenCL storage stub (the `SYCL` feature is off).
#[derive(Debug)]
pub struct QSyclStorage {
    pub elem_count: usize,
    pub device: SyclDevice,
}

impl QSyclStorage {
    pub fn zeros(_: &SyclDevice, _: usize, _: crate::quantized::GgmlDType) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
    }
    pub fn from_bytes(_: &SyclDevice, _: crate::quantized::GgmlDType, _: usize, _: &[u8]) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
    }
    pub fn from_bytes_transfer(_: &SyclDevice, _: crate::quantized::GgmlDType, _: usize, _: &[u8]) -> Result<Self> {
        Err(Error::NotCompiledWithSyclSupport)
    }
    pub fn dtype(&self) -> crate::quantized::GgmlDType {
        fail!()
    }
    pub fn device(&self) -> &SyclDevice {
        &self.device
    }
    pub fn storage_size_in_bytes(&self) -> usize {
        fail!()
    }
    pub fn is_zero_copy(&self) -> bool {
        false
    }
    pub fn data(&self) -> Result<Vec<u8>> {
        Err(Error::NotCompiledWithSyclSupport)
    }
    pub fn dequantize(&self, _: usize) -> Result<SyclStorage> {
        Err(Error::NotCompiledWithSyclSupport)
    }
    pub fn fwd(&self, _: &Shape, _: &SyclStorage, _: &Layout) -> Result<(SyclStorage, Shape)> {
        Err(Error::NotCompiledWithSyclSupport)
    }
    pub fn embedding(&self, _: usize, _: usize, _: &SyclStorage, _: &Layout) -> Result<SyclStorage> {
        Err(Error::NotCompiledWithSyclSupport)
    }
}

pub fn new_sycl_device(_ordinal: usize) -> Result<SyclDevice> {
    Err(Error::NotCompiledWithSyclSupport)
}
