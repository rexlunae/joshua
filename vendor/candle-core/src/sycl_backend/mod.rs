//! SYCL 2020 backend, using an optional dynamically loaded C++ bridge.
//! Native tensor and block-quantized kernels run on an in-order SYCL queue.
//! Unsupported operation/dtype combinations use the CPU reference path.
//! Device memory stays alive until pending kernels have finished; the bridge
//! bounds deferred frees and submissions to avoid unbounded in-flight memory.
#![allow(clippy::missing_safety_doc)]
mod bridge;
pub mod kernels;
pub use kernels::{fallback_count, native_enabled, native_exec_count};
use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::quantized::GgmlDType;
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};
use kernels::{Ctx, Idx, MatStrides};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);
#[derive(Debug, Default)]
struct Scratch { buffer: usize, bytes: usize }
const SCRATCH_MAX_BYTES: usize = 256 << 20;
#[derive(Debug)]
struct SyclContext {
    context: usize,
    fault: usize,
    name: String,
    memory: u64,
    scratch: std::sync::Mutex<Scratch>,
}
impl SyclContext {
    fn drain_slot(&self, slot: usize) {
        unsafe { bridge::finish(self.context); }
        let zero = 0u32;
        let _ = unsafe { write_buffer_at(self.context, self.fault, slot * 4, 4, &zero as *const u32 as *const u8) };
    }
    fn scratch(&self, bytes: usize) -> Result<std::sync::MutexGuard<'_, Scratch>> {
        let mut g = self.scratch.lock().unwrap_or_else(|p| p.into_inner());
        if g.bytes < bytes {
            let buffer = bridge::alloc(self.context, bytes.max(1))?;
            unsafe { bridge::free(g.buffer); }
            g.buffer = buffer; g.bytes = bytes;
        }
        Ok(g)
    }
}
impl Drop for SyclContext {
    fn drop(&mut self) {
        let scratch = self.scratch.get_mut().unwrap_or_else(|p| p.into_inner());
        unsafe {
            bridge::free(scratch.buffer);
            bridge::free(self.fault);
            bridge::close(self.context);
        }
    }
}
#[derive(Clone, Debug)]
pub struct SyclDevice { gpu_id: usize, inner: Arc<SyclContext> }
fn sycl_error(_code: i32, op: &str) -> Error { bridge::error(op) }
impl SyclDevice {
    pub fn new(gpu_id: usize) -> Result<Self> {
        let context = bridge::open(gpu_id)?;
        let result = (|| {
            let (name, memory) = bridge::info(context)?;
            let fault = bridge::alloc(context, crate::fault_slot::BYTES)?;
            let inner = Arc::new(SyclContext { context, fault, name, memory, scratch: Default::default() });
            let zeros = vec![0u8; crate::fault_slot::BYTES];
            // From this point the context is owned by inner, including on error.
            unsafe { write_buffer(context, fault, zeros.len(), zeros.as_ptr()) }?;
            let weak = Arc::downgrade(&inner);
            crate::fault_slot::register_drain(Box::new(move |slot| match weak.upgrade() {
                Some(ctx) => { ctx.drain_slot(slot); true }
                None => false,
            }));
            Ok(Self { gpu_id, inner })
        })();
        result
    }
    pub fn new_with_stream(gpu_id: usize) -> Result<Self> { Self::new(gpu_id) }
    pub fn id(&self) -> DeviceId { DeviceId(self.gpu_id) }
    pub fn name(&self) -> &str { &self.inner.name }
    pub fn global_mem_size(&self) -> Option<u64> { Some(self.inner.memory) }
    /// This backend uses explicit device USM allocations, including on iGPUs.
    pub fn host_unified_memory(&self) -> bool { false }
    pub(crate) fn context(&self) -> usize { self.inner.context }
    pub(crate) fn queue(&self) -> usize { self.inner.context }
    pub(crate) fn transfer_queue(&self) -> usize { self.inner.context }
    pub fn ctx(&self) -> Ctx { Ctx { context: self.context(), device: self.context(), queue: self.queue(), fault: self.inner.fault, fslot: crate::fault_slot::current() as i32 } }
    pub fn check_fault(&self) -> Result<()> {
        let slot = crate::fault_slot::current() * 4;
        let mut value = 0u32;
        unsafe { read_buffer(self.queue(), self.inner.fault, slot, 4, &mut value as *mut u32 as *mut u8) }?;
        if value == 0 { return Ok(()); }
        let zero = 0u32;
        unsafe { write_buffer_at(self.queue(), self.inner.fault, slot, 4, &zero as *const u32 as *const u8) }?;
        Err(Error::Msg("sycl: indexing or embedding id out of range (reported at host read-back)".into()))
    }
    pub fn alloc(&self, dtype: DType, numel: usize) -> Result<SyclStorage> {
        let elem = dtype.size_in_bytes();
        if elem == 0 { crate::bail!("sycl: sub-byte storage is unsupported"); }
        let bytes = numel.checked_mul(elem).ok_or_else(|| Error::Msg("sycl: storage size overflow".into()))?;
        let buffer = bridge::alloc(self.context(), bytes.max(1))?;
        Ok(SyclStorage { buffer, dtype, numel, device: self.clone() })
    }
    pub fn synchronize(&self) -> Result<()> {
        if unsafe { bridge::finish(self.queue()) } != 0 { return Err(bridge::error("synchronize")); }
        self.check_fault()
    }
}

/// SYCL device USM storage, with dtype and element count.
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
            unsafe { bridge::free(self.buffer) };
        }
    }
}

impl SyclStorage {
    pub fn transfer_to_device(&self, dst: &SyclDevice) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        dst.storage_from_cpu_storage(&cpu)
    }

    pub fn from_vec<T: crate::WithDType>(slice: Vec<T>, device: &SyclDevice) -> Result<Self> {
        let dtype = T::DTYPE;
        let bytes = dtype
            .size_in_bytes()
            .checked_mul(slice.len())
            .ok_or_else(|| Error::Msg("sycl: overflow in storage size".into()))?;
        let buffer = create_buffer(device.context(), bytes.max(1), 0)?;
        if bytes > 0 {
            // # Safety: slice.as_ptr() points to `bytes` valid bytes.
            let data = unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, bytes) };
            if let Err(e) = unsafe { write_buffer(device.queue(), buffer, bytes, data.as_ptr()) } {
                unsafe { bridge::free(buffer) };
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
                    kernels::cast_kernel(self.dtype, DType::U32).ok_or_else(|| Error::Msg("sycl: no id cast".into()))?
                };
                kernels::run_cast(&self.ctx(), name, self.buffer, out.buffer, n, l)?;
                Ok((std::borrow::Cow::Owned(out), 0))
            }
            d => Err(Error::Msg(format!("sycl: unsupported index dtype {d:?}"))),
        }
    }
}

impl Clone for SyclStorage {
    /// Whole-buffer device-to-device copy.
    fn clone(&self) -> Self {
        self.try_clone(&Layout::contiguous(self.numel)).expect("sycl: device buffer copy failed")
    }
}

pub(crate) fn release_mem(buffer: usize) {
    if buffer != 0 {
        unsafe { bridge::free(buffer) };
    }
}

pub(crate) fn create_buffer(context: usize, bytes: usize, _flags: u64) -> Result<usize> {
    bridge::alloc(context, bytes)
}
unsafe fn write_buffer(queue: usize, buffer: usize, bytes: usize, ptr: *const u8) -> Result<()> {
    bridge::write(queue, buffer, 0, bytes, ptr)
}
pub(crate) unsafe fn write_buffer_at(queue: usize, buffer: usize, offset: usize, bytes: usize, ptr: *const u8) -> Result<()> {
    bridge::write(queue, buffer, offset, bytes, ptr)
}
pub(crate) unsafe fn read_buffer(queue: usize, buffer: usize, offset: usize, bytes: usize, ptr: *mut u8) -> Result<()> {
    bridge::read(queue, buffer, offset, bytes, ptr)
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

/// Hand a native op's f32 result on, after the `JOSHUA_SYCL_CHECK_NAN`
/// device-side NaN count when that diagnostic is enabled.
fn nan_checked(op: &str, out: SyclStorage) -> SyclStorage {
    if out.dtype == DType::F32 && kernels::check_nan_enabled() {
        kernels::debug_check_nan(&out.device.ctx(), op, out.buffer, out.numel);
    }
    out
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

impl BackendStorage for SyclStorage {
    type Device = SyclDevice;

    fn try_clone(&self, _layout: &Layout) -> Result<Self> {
        let bytes = self.numel * self.elem_size();
        let out = self.device.alloc(self.dtype, self.numel)?;
        if bytes > 0 {
            let rc = unsafe { bridge::copy(self.device.queue(), self.buffer, out.buffer, 0, 0, bytes) };
            if rc != 0 {
                return Err(sycl_error(rc, "bridge::copy"));
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
            other => return Err(Error::Msg(format!("sycl to_cpu_storage: dtype {other:?} not supported"))),
        })
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        if native() && self.dtype == DType::F32 {
            let n = layout.shape().elem_count();
            let out = self.device.alloc(DType::F32, n)?;
            match kernels::run_affine(&self.ctx(), self.buffer, out.buffer, n, layout, mul as f32, add as f32) {
                Ok(()) => return Ok(nan_checked("affine", out)),
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
                Ok(()) => return Ok(nan_checked("powf", out)),
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
                Ok(()) => return Ok(nan_checked("elu", out)),
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
                Ok(Some(out)) => return Ok(nan_checked("reduce_op", out)),
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
                Ok(()) => return Ok(nan_checked("cmp", out)),
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
                        Ok(()) => return Ok(nan_checked("to_dtype", out)),
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
                    Ok(()) => return Ok(nan_checked(B::NAME, out)),
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
                    Ok(()) => return Ok(nan_checked(B::NAME, out)),
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
                    Ok(()) => return Ok(nan_checked("where_cond", out)),
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
                Ok(out) => return Ok(nan_checked("index_select", out)),
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
                Ok(out) => return Ok(nan_checked("gather", out)),
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
                Ok(out) => return Ok(nan_checked("index_add", out)),
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
                Ok(out) => return Ok(nan_checked("matmul", out)),
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

impl SyclStorage {
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
        kernels::run_index_select(&self.ctx(), elem8, i64_ids, src.buffer, ids_s.buffer, out.buffer, n, left, n_ids, right, dim_size, src_off, ids_off)?;
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
        let ix = Idx::new(ids_l.dims())?.with_layout(0, if ids_s.buffer == ids.buffer { ids_l } else { &ids_layout })?.with_strides(1, &src_strides, l.start_offset())?;
        let out = self.device.alloc(self.dtype, n)?;
        kernels::run_gather(&self.ctx(), self.buffer, ids_s.buffer, out.buffer, n, ix, dim_stride, dims[dim])?;
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
            return Err(Error::Msg("sycl index_add: bad dim".into()));
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

impl BackendDevice for SyclDevice {
    type Storage = SyclStorage;

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
        crate::DeviceLocation::Sycl { gpu_id: self.gpu_id }
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
            other => Err(Error::Msg(format!("sycl storage_from_cpu_storage: dtype {:?} not supported", other.dtype()))),
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
        let rc = unsafe { bridge::finish(self.queue()) };
        if rc != 0 {
            return Err(sycl_error(rc, "bridge::finish"));
        }
        self.check_fault()
    }
}

// ─── Block-quantized weights on the device ───────────────────────────────────

/// Rows below which a quantized matmul dequantizes inside the GEMV kernel;
/// larger inputs (prefill) dequantize the weight to a scratch f32 buffer
/// once and run the tiled GEMM.
/// Rows up to which `fwd` runs the fused dequantize-and-dot GEMV instead of
/// dequantizing the whole weight once and running the tiled GEMM.
///
/// `k_qgemv` decodes every block again for every row (its second grid
/// dimension is the row), so its cost grows linearly with `m` while the
/// GEMM path pays one decode of the matrix plus a GEMM that is nearly free at
/// these sizes; for the 256-element block formats (the K-quants and IQ2_XXS,
/// whose decode dominates) the break-even sits at a handful of rows.  The
/// half-precision `k_hgemv` reads its weights as-is and keeps the wider window.
fn qgemv_max_rows(dtype: GgmlDType) -> usize {
    use GgmlDType::*;
    if kernels::qgemv_multirow(dtype) {
        // `k_qgemv_mr` decodes each weight once for up to 16 rows.
        return kernels::QGEMV_MR_MAX_ROWS;
    }
    match dtype {
        F16 | BF16 => 16,
        Q2K | Q3K | Q4K | Q5K | Q6K | Q8K | Iq2Xxs => 4,
        _ => 8,
    }
}

/// A GGUF tensor stored as block-quantized bytes in device USM.
pub struct QSyclStorage {
    pub buffer: usize,
    pub byte_offset: u64,
    pub dtype: GgmlDType,
    pub elem_count: usize,
    pub device: SyclDevice,
    _host: Option<Arc<dyn std::any::Any + Send + Sync>>,
}

impl std::fmt::Debug for QSyclStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QSyclStorage({:?}, {} elems, zero_copy={})", self.dtype, self.elem_count, self._host.is_some())
    }
}

impl Drop for QSyclStorage {
    fn drop(&mut self) {
        if self.buffer != 0 {
            unsafe { bridge::free(self.buffer) };
        }
    }
}

/// Arbitrary mmap pointers are not portable SYCL USM allocations.
/// Quantized weights are uploaded in their original block format.
pub fn zero_copy_enabled() -> bool { false }

impl QSyclStorage {
    fn bytes_for(dtype: GgmlDType, elem_count: usize) -> Result<usize> {
        let bs = dtype.block_size();
        if !elem_count.is_multiple_of(bs) {
            return Err(Error::Msg(format!("sycl: {elem_count} elements is not a whole number of {dtype:?} blocks")));
        }
        Ok(elem_count / bs * dtype.type_size())
    }

    pub fn zeros(device: &SyclDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let bytes = Self::bytes_for(dtype, elem_count)?;
        let buffer = create_buffer(device.context(), bytes.max(1), 0)?;
        if bytes > 0 {
            let zeros = vec![0u8; bytes];
            if let Err(e) = unsafe { write_buffer(device.queue(), buffer, bytes, zeros.as_ptr()) } {
                unsafe { bridge::free(buffer) };
                return Err(e);
            }
        }
        Ok(Self { buffer, byte_offset: 0, dtype, elem_count, device: device.clone(), _host: None })
    }

    /// Upload raw block bytes (a blocking write on the compute queue).
    pub fn from_bytes(device: &SyclDevice, dtype: GgmlDType, elem_count: usize, data: &[u8]) -> Result<Self> {
        Self::from_bytes_on(device, device.queue(), dtype, elem_count, data)
    }

    /// Blocking upload. This initial backend uses the compute queue for all
    /// transfers; returned storage is immediately safe to use.
    pub fn from_bytes_transfer(device: &SyclDevice, dtype: GgmlDType, elem_count: usize, data: &[u8]) -> Result<Self> {
        Self::from_bytes_on(device, device.transfer_queue(), dtype, elem_count, data)
    }

    fn from_bytes_on(device: &SyclDevice, queue: usize, dtype: GgmlDType, elem_count: usize, data: &[u8]) -> Result<Self> {
        let bytes = Self::bytes_for(dtype, elem_count)?;
        if data.len() < bytes {
            return Err(Error::Msg(format!("sycl: {} bytes given for a {dtype:?} tensor needing {bytes}", data.len())));
        }
        let buffer = create_buffer(device.context(), bytes.max(1), 0)?;
        if let Err(e) = unsafe { write_buffer(queue, buffer, bytes, data.as_ptr()) } {
            unsafe { bridge::free(buffer) };
            return Err(e);
        }
        Ok(Self { buffer, byte_offset: 0, dtype, elem_count, device: device.clone(), _host: None })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &SyclDevice {
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
    pub fn dequantize(&self, elem_count: usize) -> Result<SyclStorage> {
        let out = self.device.alloc(DType::F32, elem_count)?;
        self.dequantize_into(out.buffer, elem_count)?;
        Ok(out)
    }

    /// Dequantize `elem_count` elements into the f32 device buffer `out`.
    fn dequantize_into(&self, out: usize, elem_count: usize) -> Result<()> {
        match self.dtype {
            GgmlDType::F32 => {
                let rc = unsafe {
                    bridge::copy(self.device.queue(), self.buffer, out, self.byte_offset as usize, 0, elem_count * 4)
                };
                if rc != 0 {
                    return Err(sycl_error(rc, "bridge::copy"));
                }
                Ok(())
            }
            _ => kernels::run_dequant(&self.device.ctx(), self.dtype, self.buffer, out, elem_count, self.byte_offset),
        }
    }

    /// `x @ W^T` for `x` on the device (f32, contiguous) and this `[n, k]` weight.
    pub fn fwd(&self, self_shape: &Shape, storage: &SyclStorage, layout: &Layout) -> Result<(SyclStorage, Shape)> {
        if storage.dtype != DType::F32 {
            return Err(Error::Msg(format!("sycl qmatmul: input must be f32, got {:?}", storage.dtype)));
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
            return Err(Error::Msg(format!("sycl qmatmul: input {layout:?} incompatible with {self_shape:?}")));
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
            GgmlDType::F16 | GgmlDType::BF16 if m <= qgemv_max_rows(self.dtype) => {
                kernels::run_hgemv(&c, self.dtype == GgmlDType::BF16, storage.buffer, self.buffer, out.buffer, m, n, k, self.byte_offset, o1)?;
            }
            _ if m <= qgemv_max_rows(self.dtype) => {
                kernels::run_qgemv(&c, self.dtype, storage.buffer, self.buffer, out.buffer, m, n, k, self.byte_offset, o1)?;
            }
            _ => {
                // Prefill: dequantize once, then the tiled GEMM.  The f32
                // copy lives in the device's shared scratch (held for the
                // two launches) so a long prefill cannot queue one fresh
                // `n*k*4`-byte buffer per matmul ahead of the GPU.
                let sa = MatStrides { row: k, col: 1, offset: o1, batch: 0 };
                let sb = MatStrides { row: 1, col: k, offset: 0, batch: 0 };
                let bytes = n * k * 4;
                if bytes <= SCRATCH_MAX_BYTES {
                    let scratch = self.device.inner.scratch(bytes)?;
                    self.dequantize_into(scratch.buffer, n * k)?;
                    kernels::run_matmul(&c, storage.buffer, scratch.buffer, out.buffer, (1, m, n, k), sa, sb)?;
                } else {
                    let w = self.dequantize(n * k)?;
                    kernels::run_matmul(&c, storage.buffer, w.buffer, out.buffer, (1, m, n, k), sa, sb)?;
                }
            }
        }
        Ok((nan_checked("qmatmul", out), dst_shape))
    }

    /// Gather rows `ids` of this `[rows, hidden]` table as f32 `[n_ids, hidden]`.
    pub fn embedding(&self, rows: usize, hidden: usize, ids: &SyclStorage, ids_l: &Layout) -> Result<SyclStorage> {
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

