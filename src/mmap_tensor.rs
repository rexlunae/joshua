//! Zero-copy quantized tensors borrowed directly from the memory-mapped GGUF.
//!
//! Joshua has always `mmap`ed the model file, but candle's GGUF reader took a
//! `Read + Seek` and *copied* every tensor out of the mapping into the heap —
//! twice, in fact: once into a `Vec<u8>` staging buffer
//! (`TensorInfo::read`) and again when the blocks were re-collected with
//! `to_vec()`.  The mapping was doing nothing but standing in for a file
//! handle, and the whole model ended up resident in anonymous memory.
//!
//! That is invisible for a 4 GB model and fatal for a 2.8 T-parameter one.
//! This module closes the gap: it reinterprets the mapped bytes *in place* as
//! quantized blocks and hands candle a [`QTensor`] that points straight at the
//! mapping.  Nothing is read until a matmul actually touches a page, the
//! kernel can evict clean pages under memory pressure, and several engine
//! instances share one copy through the page cache.
//!
//! # Safety model
//!
//! Borrowing is only sound because the mapping is read-only and the GGUF file
//! is treated as immutable for the lifetime of the process (the same
//! assumption llama.cpp makes, and the one [`crate::engine`] already
//! documents).  Every borrow additionally checks, at runtime, that the tensor
//! lies fully inside the mapping and that its start address satisfies the
//! block type's alignment.  When either check fails — a truncated file, an
//! exotic `general.alignment`, a dtype with no block type here — the borrow is
//! declined and the caller falls back to candle's copying reader, so a
//! pathological file degrades in performance rather than in correctness.
//!
//! One residual risk cannot be checked away: if the file is truncated or
//! rewritten *while* the mapping lives, a later access can fault (SIGBUS on
//! Unix) or observe torn data.  That is inherent to memory-mapped I/O and is
//! the price of zero-copy weight loading — the same tradeoff llama.cpp makes.
//! The contract is that no party modifies the model file after mapping; a
//! well-formed loader never does (the mapping is read-only), and a hostile
//! actor with write access to the file can already corrupt the process in
//! more direct ways.  Within the checked range, reads are in-bounds and every
//! byte pattern is a valid block, so there is no out-of-bounds access
//! *unless* the file changes under the mapping.

use std::fs::File;
use std::marker::PhantomData;
use std::sync::Arc;

use candle_core::quantized::QuantizedType;
use candle_core::quantized::{gguf_file, k_quants, GgmlDType, GgmlType, QStorage, QTensor};
use candle_core::{CpuStorage, Result};
use half::{bf16, f16};
use memmap2::Mmap;

/// A run of quantized blocks borrowed from the memory-mapped model file.
///
/// Holds an `Arc<Mmap>` so the mapping outlives every tensor cut from it, and
/// a raw pointer into that mapping.  No block data is ever copied.
pub struct MmapBlocks<T: GgmlType> {
    /// Keeps the mapping alive; never dereferenced directly.
    _mmap: Arc<Mmap>,
    ptr: *const T,
    len: usize,
    _marker: PhantomData<T>,
}

/// A handle that can ask the kernel to prefetch a borrowed block range into
/// the page cache.
///
/// The routed experts of a MoE are contiguous runs inside the model mapping,
/// so once the gate has picked the experts a token needs, `MADV_WILLNEED` on
/// each selected run turns the scattered page faults that would otherwise
/// stall the expert matmuls (each 4 KiB fault is an independent random read)
/// into sequential background streams the kernel reads ahead at full device
/// bandwidth.
pub trait MmapPrefetch: Send + Sync + 'static {
    /// Issue a best-effort `MADV_WILLNEED` for this block range.  Never
    /// blocks and never fails the caller.
    fn prefetch(&self);
}

impl<T: GgmlType + 'static> MmapPrefetch for MmapBlocks<T> {
    fn prefetch(&self) {
        let base = self._mmap.as_ptr() as usize;
        let off = self.ptr as usize - base;
        let len = self.len * std::mem::size_of::<T>();
        let _ = self._mmap.advise_range(memmap2::Advice::WillNeed, off, len);
    }
}

// SAFETY: the mapping is read-only and the file is immutable for the lifetime
// of the process, so the pointed-to blocks are never mutated or moved.  `_mmap`
// keeps the mapping alive for at least as long as `ptr` is valid, and `T` is
// itself `Send + Sync`.
unsafe impl<T: GgmlType> Send for MmapBlocks<T> {}
unsafe impl<T: GgmlType> Sync for MmapBlocks<T> {}

impl<T: GgmlType> MmapBlocks<T> {
    /// The borrowed blocks.
    ///
    /// Reading from this slice is what actually faults the model pages in.
    fn blocks(&self) -> &[T] {
        // SAFETY: `ptr`/`len` were bounds- and alignment-checked in `borrow`
        // against a mapping that `_mmap` keeps alive and that is never mutated.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Borrow `n_blocks` blocks starting `byte_offset` into `mmap`, or return
    /// `None` when the range is out of bounds or insufficiently aligned.
    fn borrow(mmap: &Arc<Mmap>, byte_offset: usize, n_blocks: usize) -> Option<Self> {
        let size = n_blocks.checked_mul(std::mem::size_of::<T>())?;
        let end = byte_offset.checked_add(size)?;
        if end > mmap.len() {
            return None;
        }
        // SAFETY: `byte_offset <= end <= mmap.len()`, so this stays inside the
        // mapping's allocation.
        let ptr = unsafe { mmap.as_ptr().add(byte_offset) };
        if !(ptr as usize).is_multiple_of(std::mem::align_of::<T>()) {
            return None;
        }
        Some(Self {
            _mmap: Arc::clone(mmap),
            ptr: ptr as *const T,
            len: n_blocks,
            _marker: PhantomData,
        })
    }
}

/// Mirrors candle's `impl QuantizedType for Vec<T>`, but over borrowed blocks.
impl<T: GgmlType + Send + Sync> QuantizedType for MmapBlocks<T> {
    fn dtype(&self) -> GgmlDType {
        T::DTYPE
    }

    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()> {
        crate::quant_matmul::matmul_kquant(mkn, lhs, self.blocks(), dst)
    }

    fn matmul_t_f16(&self, mkn: (usize, usize, usize), lhs: &[f16], dst: &mut [f16]) -> Result<()> {
        k_quants::matmul_f16(mkn, lhs, self.blocks(), dst)
    }

    fn embedding(&self, ids: &[u32], rows: usize, hidden: usize) -> Result<CpuStorage> {
        if !hidden.is_multiple_of(T::BLCK_SIZE) {
            candle_core::bail!(
                "quantized embedding hidden size {hidden} is not divisible by block size {}",
                T::BLCK_SIZE
            )
        }
        let blocks = self.blocks();
        let row_blocks = hidden / T::BLCK_SIZE;
        if blocks.len() != rows * row_blocks {
            candle_core::bail!(
                "quantized tensor has {} blocks, expected {}",
                blocks.len(),
                rows * row_blocks
            )
        }
        let mut out = vec![0f32; ids.len() * hidden];
        for (out_row, &row_id) in ids.iter().enumerate() {
            let row = row_id as usize;
            if row >= rows {
                candle_core::bail!("embedding id {row} is out of range for {rows} rows")
            }
            let src = &blocks[row * row_blocks..(row + 1) * row_blocks];
            let dst = &mut out[out_row * hidden..(out_row + 1) * hidden];
            T::to_float(src, dst);
        }
        Ok(CpuStorage::F32(out))
    }

    fn dequantize(&self, elem_count: usize) -> Result<CpuStorage> {
        let mut ys = vec![0.0f32; elem_count];
        T::to_float(self.blocks(), &mut ys);
        Ok(CpuStorage::F32(ys))
    }

    fn storage_size_in_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    fn block_size(&self) -> usize {
        T::BLCK_SIZE
    }

    fn size(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    fn from_float(&mut self, _xs: &[f32]) {
        // Structurally unreachable: borrowed tensors are only ever produced by
        // `borrowed_qtensor` from a read-only mapping, and candle only calls
        // this on storage it allocated itself via `QTensor::quantize`.
        // Writing through the mapping would be unsound, so refuse loudly
        // rather than silently corrupting or no-oping.
        panic!("cannot quantize into a read-only memory-mapped tensor")
    }

    fn from_float_imatrix(&mut self, _xs: &[f32], _imatrix_weights: &[f32], _n_per_row: usize) {
        panic!("cannot quantize into a read-only memory-mapped tensor")
    }
}

/// IQ2_XXS blocks borrowed from the mapping.
///
/// Candle's `GgmlDType` has no IQ2_XXS variant, so this cannot be expressed
/// as `MmapBlocks<T: GgmlType>`.  `dtype()` returns a placeholder: the only
/// consumer comparing it is `QMatMul::from_qtensor`, which distinguishes
/// f32/f16/bf16 (dequantise eagerly) from everything else (keep as QTensor),
/// and the aarch64 Q4K repack path that this deliberately is not.  The real
/// dtype travels alongside the raw header instead.
pub struct MmapBlocksIq2Xxs {
    /// Keeps the mapping alive; never dereferenced directly.
    _mmap: Arc<Mmap>,
    ptr: *const crate::iq2xxs::BlockIq2Xxs,
    len: usize,
}

// SAFETY: same argument as `MmapBlocks`: read-only mapping kept alive by
// `_mmap`, blocks never mutated or moved.
unsafe impl Send for MmapBlocksIq2Xxs {}
unsafe impl Sync for MmapBlocksIq2Xxs {}

impl MmapPrefetch for MmapBlocksIq2Xxs {
    fn prefetch(&self) {
        let base = self._mmap.as_ptr() as usize;
        let off = self.ptr as usize - base;
        let len = self.len * crate::iq2xxs::BLOCK_BYTES;
        let _ = self._mmap.advise_range(memmap2::Advice::WillNeed, off, len);
    }
}

impl MmapBlocksIq2Xxs {
    fn blocks(&self) -> &[crate::iq2xxs::BlockIq2Xxs] {
        // SAFETY: bounds- and alignment-checked in `borrow` (alignment is 1
        // for the packed block type, so only bounds matter) against a mapping
        // that `_mmap` keeps alive and that is never mutated.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    fn borrow(mmap: &Arc<Mmap>, byte_offset: usize, n_blocks: usize) -> Option<Self> {
        let size = n_blocks.checked_mul(crate::iq2xxs::BLOCK_BYTES)?;
        let end = byte_offset.checked_add(size)?;
        if end > mmap.len() {
            return None;
        }
        // SAFETY: `byte_offset <= end <= mmap.len()`.
        let ptr = unsafe { mmap.as_ptr().add(byte_offset) };
        // The block type is packed (align 1), so any offset is aligned.
        Some(Self {
            _mmap: Arc::clone(mmap),
            ptr: ptr as *const crate::iq2xxs::BlockIq2Xxs,
            len: n_blocks,
        })
    }
}

/// Prefetch handle for `n_blocks` k-quant blocks at `byte_offset` in `mmap`.
///
/// Re-borrows the range with the same bounds/alignment checks as
/// `MmapBlocks::borrow`; returns `None` exactly when a borrow would be
/// declined, so callers can pair it with the borrow they already performed.
/// The `dtype` picks the block type, mirroring [`borrowed_range`].
pub fn prefetch_handle(
    mmap: &Arc<Mmap>,
    dtype: GgmlDType,
    byte_offset: usize,
    n_blocks: usize,
) -> Option<Arc<dyn MmapPrefetch>> {
    macro_rules! handle {
        ($ty:ty) => {
            MmapBlocks::<$ty>::borrow(mmap, byte_offset, n_blocks)
                .map(|b| Arc::new(b) as Arc<dyn MmapPrefetch>)
        };
    }
    match dtype {
        GgmlDType::F32 => handle!(f32),
        GgmlDType::F16 => handle!(f16),
        GgmlDType::BF16 => handle!(bf16),
        GgmlDType::Q4_0 => handle!(k_quants::BlockQ4_0),
        GgmlDType::Q4_1 => handle!(k_quants::BlockQ4_1),
        GgmlDType::Q5_0 => handle!(k_quants::BlockQ5_0),
        GgmlDType::Q5_1 => handle!(k_quants::BlockQ5_1),
        GgmlDType::Q8_0 => handle!(k_quants::BlockQ8_0),
        GgmlDType::Q8_1 => handle!(k_quants::BlockQ8_1),
        GgmlDType::Q2K => handle!(k_quants::BlockQ2K),
        GgmlDType::Q3K => handle!(k_quants::BlockQ3K),
        GgmlDType::Q4K => handle!(k_quants::BlockQ4K),
        GgmlDType::Q5K => handle!(k_quants::BlockQ5K),
        GgmlDType::Q6K => handle!(k_quants::BlockQ6K),
        GgmlDType::Q8K => handle!(k_quants::BlockQ8K),
    }
}

/// Prefetch handle for `n_blocks` IQ2_XXS blocks; see [`prefetch_handle`].
pub fn prefetch_handle_iq2xxs(
    mmap: &Arc<Mmap>,
    byte_offset: usize,
    n_blocks: usize,
) -> Option<Arc<dyn MmapPrefetch>> {
    MmapBlocksIq2Xxs::borrow(mmap, byte_offset, n_blocks)
        .map(|b| Arc::new(b) as Arc<dyn MmapPrefetch>)
}

/// MXFP4 blocks borrowed from the mapping.
///
/// Same shape as [`MmapBlocksIq2Xxs`]: candle's `GgmlDType` has no MXFP4
/// variant either, so this is a bespoke `QuantizedType` with a placeholder
/// `dtype()` (only `QMatMul::from_qtensor` compares it, and only to tell
/// eager-dequantise f32/f16/bf16 apart from everything else).
pub struct MmapBlocksMxfp4 {
    /// Keeps the mapping alive; never dereferenced directly.
    _mmap: Arc<Mmap>,
    ptr: *const crate::mxfp4::BlockMxfp4,
    len: usize,
}

// SAFETY: same argument as `MmapBlocks`: read-only mapping kept alive by
// `_mmap`, blocks never mutated or moved.
unsafe impl Send for MmapBlocksMxfp4 {}
unsafe impl Sync for MmapBlocksMxfp4 {}

impl MmapPrefetch for MmapBlocksMxfp4 {
    fn prefetch(&self) {
        let base = self._mmap.as_ptr() as usize;
        let off = self.ptr as usize - base;
        let len = self.len * std::mem::size_of::<crate::mxfp4::BlockMxfp4>();
        let _ = self._mmap.advise_range(memmap2::Advice::WillNeed, off, len);
    }
}

impl MmapBlocksMxfp4 {
    fn blocks(&self) -> &[crate::mxfp4::BlockMxfp4] {
        // SAFETY: bounds- and alignment-checked in `borrow` (alignment is 1
        // for the packed block type, so only bounds matter) against a mapping
        // that `_mmap` keeps alive and that is never mutated.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    fn borrow(mmap: &Arc<Mmap>, byte_offset: usize, n_blocks: usize) -> Option<Self> {
        let block_bytes = std::mem::size_of::<crate::mxfp4::BlockMxfp4>();
        let size = n_blocks.checked_mul(block_bytes)?;
        let end = byte_offset.checked_add(size)?;
        if end > mmap.len() {
            return None;
        }
        // SAFETY: `byte_offset <= end <= mmap.len()`.
        let ptr = unsafe { mmap.as_ptr().add(byte_offset) };
        // The block type is packed (align 1), so any offset is aligned.
        Some(Self {
            _mmap: Arc::clone(mmap),
            ptr: ptr as *const crate::mxfp4::BlockMxfp4,
            len: n_blocks,
        })
    }
}

/// Prefetch handle for `n_blocks` MXFP4 blocks; see [`prefetch_handle`].
pub fn prefetch_handle_mxfp4(
    mmap: &Arc<Mmap>,
    byte_offset: usize,
    n_blocks: usize,
) -> Option<Arc<dyn MmapPrefetch>> {
    MmapBlocksMxfp4::borrow(mmap, byte_offset, n_blocks)
        .map(|b| Arc::new(b) as Arc<dyn MmapPrefetch>)
}

impl QuantizedType for MmapBlocksMxfp4 {
    fn dtype(&self) -> GgmlDType {
        GgmlDType::Q2K // placeholder; see struct docs
    }

    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()> {
        crate::mxfp4::matmul_t(mkn, lhs, self.blocks(), dst)
    }

    fn matmul_t_f16(&self, mkn: (usize, usize, usize), lhs: &[f16], dst: &mut [f16]) -> Result<()> {
        crate::mxfp4::matmul_t_f16(mkn, lhs, self.blocks(), dst)
    }

    fn embedding(&self, ids: &[u32], rows: usize, hidden: usize) -> Result<CpuStorage> {
        if !hidden.is_multiple_of(crate::mxfp4::QK_MXFP4) {
            candle_core::bail!(
                "quantized embedding hidden size {hidden} is not divisible by block size {}",
                crate::mxfp4::QK_MXFP4
            )
        }
        let blocks = self.blocks();
        let row_blocks = hidden / crate::mxfp4::QK_MXFP4;
        if blocks.len() != rows * row_blocks {
            candle_core::bail!(
                "quantized tensor has {} blocks, expected {}",
                blocks.len(),
                rows * row_blocks
            )
        }
        let mut out = vec![0f32; ids.len() * hidden];
        for (out_row, &row_id) in ids.iter().enumerate() {
            let row = row_id as usize;
            if row >= rows {
                candle_core::bail!("embedding id {row} is out of range for {rows} rows")
            }
            let src = &blocks[row * row_blocks..(row + 1) * row_blocks];
            let dst = &mut out[out_row * hidden..(out_row + 1) * hidden];
            crate::mxfp4::dequantize(src, dst)?;
        }
        Ok(CpuStorage::F32(out))
    }

    fn dequantize(&self, elem_count: usize) -> Result<CpuStorage> {
        let mut ys = vec![0.0f32; elem_count];
        crate::mxfp4::dequantize(self.blocks(), &mut ys)?;
        Ok(CpuStorage::F32(ys))
    }

    fn storage_size_in_bytes(&self) -> usize {
        self.len * std::mem::size_of::<crate::mxfp4::BlockMxfp4>()
    }

    fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    fn block_size(&self) -> usize {
        crate::mxfp4::QK_MXFP4
    }

    fn size(&self) -> usize {
        self.len * std::mem::size_of::<crate::mxfp4::BlockMxfp4>()
    }

    fn from_float(&mut self, _xs: &[f32]) {
        panic!("cannot quantize into a read-only memory-mapped tensor")
    }

    fn from_float_imatrix(&mut self, _xs: &[f32], _imatrix_weights: &[f32], _n_per_row: usize) {
        panic!("cannot quantize into a read-only memory-mapped tensor")
    }
}

impl QuantizedType for MmapBlocksIq2Xxs {
    fn dtype(&self) -> GgmlDType {
        GgmlDType::Q2K // placeholder; see struct docs
    }

    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()> {
        crate::iq2xxs::matmul_t(mkn, lhs, self.blocks(), dst)
    }

    fn matmul_t_f16(&self, mkn: (usize, usize, usize), lhs: &[f16], dst: &mut [f16]) -> Result<()> {
        crate::iq2xxs::matmul_t_f16(mkn, lhs, self.blocks(), dst)
    }

    fn embedding(&self, ids: &[u32], rows: usize, hidden: usize) -> Result<CpuStorage> {
        if !hidden.is_multiple_of(crate::iq2xxs::QK_IQ2_XXS) {
            candle_core::bail!(
                "quantized embedding hidden size {hidden} is not divisible by block size {}",
                crate::iq2xxs::QK_IQ2_XXS
            )
        }
        let blocks = self.blocks();
        let row_blocks = hidden / crate::iq2xxs::QK_IQ2_XXS;
        if blocks.len() != rows * row_blocks {
            candle_core::bail!(
                "quantized tensor has {} blocks, expected {}",
                blocks.len(),
                rows * row_blocks
            )
        }
        let mut out = vec![0f32; ids.len() * hidden];
        for (out_row, &row_id) in ids.iter().enumerate() {
            let row = row_id as usize;
            if row >= rows {
                candle_core::bail!("embedding id {row} is out of range for {rows} rows")
            }
            let src = &blocks[row * row_blocks..(row + 1) * row_blocks];
            let dst = &mut out[out_row * hidden..(out_row + 1) * hidden];
            crate::iq2xxs::dequantize(src, dst)?;
        }
        Ok(CpuStorage::F32(out))
    }

    fn dequantize(&self, elem_count: usize) -> Result<CpuStorage> {
        let mut ys = vec![0.0f32; elem_count];
        crate::iq2xxs::dequantize(self.blocks(), &mut ys)?;
        Ok(CpuStorage::F32(ys))
    }

    fn storage_size_in_bytes(&self) -> usize {
        self.len * crate::iq2xxs::BLOCK_BYTES
    }

    fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    fn block_size(&self) -> usize {
        crate::iq2xxs::QK_IQ2_XXS
    }

    fn size(&self) -> usize {
        self.len * crate::iq2xxs::BLOCK_BYTES
    }

    fn from_float(&mut self, _xs: &[f32]) {
        panic!("cannot quantize into a read-only memory-mapped tensor")
    }

    fn from_float_imatrix(&mut self, _xs: &[f32], _imatrix_weights: &[f32], _n_per_row: usize) {
        panic!("cannot quantize into a read-only memory-mapped tensor")
    }
}

/// Borrow a tensor by raw GGUF dtype id, falling back to candle's table.
///
/// `dtype` is the raw header id, so IQ2_XXS (16) — which candle cannot name —
/// can still be borrowed.  Returns `Ok(None)` when the dtype is unknown or the
/// range cannot be borrowed safely.
pub fn borrowed_qtensor_raw(
    mmap: &Arc<Mmap>,
    dtype: u32,
    offset: u64,
    tensor_data_offset: u64,
    shape: candle_core::Shape,
) -> Result<Option<QTensor>> {
    if dtype == crate::iq2xxs::GGML_TYPE_IQ2_XXS {
        let Ok(off) = usize::try_from(tensor_data_offset.saturating_add(offset)) else {
            return Ok(None);
        };
        return borrowed_range_iq2xxs(mmap, off, shape);
    }
    if dtype == crate::mxfp4::GGML_TYPE_MXFP4 {
        let Ok(off) = usize::try_from(tensor_data_offset.saturating_add(offset)) else {
            return Ok(None);
        };
        return borrowed_range_mxfp4(mmap, off, shape);
    }
    let Some(ggml_dtype) = crate::gguf_ext::ggml_dtype_from_id(dtype) else {
        return Ok(None);
    };
    let info = gguf_file::TensorInfo {
        ggml_dtype,
        shape,
        offset,
    };
    borrowed_qtensor(mmap, &info, tensor_data_offset)
}

/// Borrow an IQ2_XXS tensor (or per-expert slice of one) from the mapping.
pub fn borrowed_range_iq2xxs(
    mmap: &Arc<Mmap>,
    offset: usize,
    shape: candle_core::Shape,
) -> Result<Option<QTensor>> {
    let elem_count = shape.elem_count();
    if !elem_count.is_multiple_of(crate::iq2xxs::QK_IQ2_XXS) {
        return Ok(None);
    }
    let n_blocks = elem_count / crate::iq2xxs::QK_IQ2_XXS;
    let Some(blocks) = MmapBlocksIq2Xxs::borrow(mmap, offset, n_blocks) else {
        return Ok(None);
    };
    let storage: Box<dyn QuantizedType> = Box::new(blocks);
    QTensor::new(QStorage::Cpu(storage), shape).map(Some)
}

/// Borrow an MXFP4 tensor (or per-expert slice of one) from the mapping.
pub fn borrowed_range_mxfp4(
    mmap: &Arc<Mmap>,
    offset: usize,
    shape: candle_core::Shape,
) -> Result<Option<QTensor>> {
    let elem_count = shape.elem_count();
    if !elem_count.is_multiple_of(crate::mxfp4::QK_MXFP4) {
        return Ok(None);
    }
    let n_blocks = elem_count / crate::mxfp4::QK_MXFP4;
    let Some(blocks) = MmapBlocksMxfp4::borrow(mmap, offset, n_blocks) else {
        return Ok(None);
    };
    let storage: Box<dyn QuantizedType> = Box::new(blocks);
    QTensor::new(QStorage::Cpu(storage), shape).map(Some)
}

/// Build a [`QTensor`] that borrows `info`'s bytes from the mapping.
///
/// Returns `Ok(None)` when the tensor cannot be borrowed safely — an unknown
/// dtype, a misaligned start address, or a range running past the end of the
/// file — leaving the caller to fall back to candle's copying reader.
pub fn borrowed_qtensor(
    mmap: &Arc<Mmap>,
    info: &gguf_file::TensorInfo,
    tensor_data_offset: u64,
) -> Result<Option<QTensor>> {
    let Ok(offset) = usize::try_from(tensor_data_offset.saturating_add(info.offset)) else {
        return Ok(None);
    };
    borrowed_range(mmap, info.ggml_dtype, offset, info.shape.clone())
}

/// Borrow an arbitrary byte range of the mapping as a quantized tensor.
///
/// This is what makes fine-grained mixture-of-experts models viable: a stacked
/// `[n_expert, out, in]` expert tensor can be sliced into per-expert matrices
/// that each point at their own offset inside the mapping.  Building all of
/// them is nearly free — an expert is a pointer and a length, not a buffer —
/// and the kernel pages in only the experts a token actually routes to, then
/// evicts them under pressure.  No explicit cache is required; the page cache
/// *is* the cache.
///
/// Returns `Ok(None)` if the range is out of bounds or misaligned, so callers
/// can fall back to copying.
pub fn borrowed_range(
    mmap: &Arc<Mmap>,
    dtype: GgmlDType,
    offset: usize,
    shape: candle_core::Shape,
) -> Result<Option<QTensor>> {
    let elem_count = shape.elem_count();
    let block_size = dtype.block_size();
    if block_size == 0 || !elem_count.is_multiple_of(block_size) {
        return Ok(None);
    }
    let n_blocks = elem_count / block_size;

    // Reinterpret the mapped bytes as the block type matching this dtype.
    macro_rules! borrow {
        ($ty:ty) => {
            match MmapBlocks::<$ty>::borrow(mmap, offset, n_blocks) {
                Some(blocks) => blocks,
                None => return Ok(None),
            }
        };
    }
    let storage: Box<dyn QuantizedType> = match dtype {
        GgmlDType::F32 => Box::new(borrow!(f32)),
        GgmlDType::F16 => Box::new(borrow!(f16)),
        GgmlDType::BF16 => Box::new(borrow!(bf16)),
        GgmlDType::Q4_0 => Box::new(borrow!(k_quants::BlockQ4_0)),
        GgmlDType::Q4_1 => Box::new(borrow!(k_quants::BlockQ4_1)),
        GgmlDType::Q5_0 => Box::new(borrow!(k_quants::BlockQ5_0)),
        GgmlDType::Q5_1 => Box::new(borrow!(k_quants::BlockQ5_1)),
        GgmlDType::Q8_0 => Box::new(borrow!(k_quants::BlockQ8_0)),
        GgmlDType::Q8_1 => Box::new(borrow!(k_quants::BlockQ8_1)),
        GgmlDType::Q2K => Box::new(borrow!(k_quants::BlockQ2K)),
        GgmlDType::Q3K => Box::new(borrow!(k_quants::BlockQ3K)),
        GgmlDType::Q4K => Box::new(borrow!(k_quants::BlockQ4K)),
        GgmlDType::Q5K => Box::new(borrow!(k_quants::BlockQ5K)),
        GgmlDType::Q6K => Box::new(borrow!(k_quants::BlockQ6K)),
        GgmlDType::Q8K => Box::new(borrow!(k_quants::BlockQ8K)),
    };

    QTensor::new(QStorage::Cpu(storage), shape).map(Some)
}

/// Load a tensor by name, borrowing from the mapping when possible and
/// falling back to candle's copying reader otherwise.
pub fn qtensor_from_mmap<R: std::io::Read + std::io::Seek>(
    content: &gguf_file::Content,
    mmap: &Arc<Mmap>,
    reader: &mut R,
    name: &str,
    device: &candle_core::Device,
) -> Result<QTensor> {
    // Borrowing is only sound on CPU; accelerator storage must be copied over.
    if device.is_cpu() {
        if let Some(info) = content.tensor_infos.get(name) {
            if let Some(t) = borrowed_qtensor(mmap, info, content.tensor_data_offset)? {
                return Ok(t);
            }
            tracing::debug!(
                tensor = name,
                "tensor not borrowable from the mapping, copying instead"
            );
        }
    }
    content.tensor(reader, name, device)
}

// ─── Layer-ahead pread prefetch thread ───────────────────────────────────────

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::JoinHandle;

/// How many layers ahead of the compute thread the prefetcher keeps streamed.
///
/// One layer of routed-expert weights is ~0.75 GB here; at ~1.9 GB/s device
/// bandwidth a layer takes ~0.4 s to read, about the same as the layer loop's
/// compute time, so staying 2–3 layers ahead gives the disk a full layer's
/// worth of lead while bounding how much of the page cache the stream occupies
/// (~2.5 GB at depth 3).
pub const PREFETCH_AHEAD_DEPTH: usize = 3;

/// Size of the scratch buffer used per `pread` syscall.  The bytes are
/// discarded — the page cache is the transport — so this only bounds syscall
/// size.  4 MiB keeps syscall overhead negligible without pinning much RAM.
const PREFETCH_CHUNK: usize = 4 * 1024 * 1024;

/// How long the thread naps when it has caught up with the compute thread.
const PREFETCH_POLL: std::time::Duration = std::time::Duration::from_millis(1);

/// A background thread that pre-reads upcoming layer byte ranges into the page
/// cache while the caller computes the current layer.
///
/// The other prefetch paths are *hints*: `MADV_WILLNEED`/`MADV_SEQUENTIAL`
/// ask the kernel to start readahead, but the reads are driven by the compute
/// thread's fault stream, so the matmuls still stall on the first touches of
/// each page.  This thread issues actual `pread(2)` calls through its own file
/// descriptor — its own readahead context, independent of the mmap's — so the
/// kernel streams each range at full device bandwidth *ahead* of the layer
/// loop, and the later mmap faults are pure page-cache hits.  The read data is
/// discarded; the page cache is the transport.
///
/// The thread is best-effort: an I/O error stops the current range and the
/// loop continues; prefill never fails because prefetch failed.
#[cfg(unix)]
pub struct LayerPrefetcher {
    /// The layer the compute thread is on; the thread keeps `[cur+1, cur+depth)`
    /// streamed (plus the current layer, so a cold start warms it too).
    current: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

#[cfg(unix)]
impl LayerPrefetcher {
    /// Spawn the prefetch thread for `ranges` (per-layer `(start, end)` byte
    /// offsets into `file`, absolute in the file, as produced by
    /// [`crate::gguf_ext::GgufHeader::layer_expert_ranges`]).
    pub fn spawn(
        file: Arc<File>,
        ranges: Arc<Vec<Option<(usize, usize)>>>,
        depth: usize,
    ) -> Self {
        let current = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let join = {
            let file = Arc::clone(&file);
            let ranges = Arc::clone(&ranges);
            let current = Arc::clone(&current);
            let stop = Arc::clone(&stop);
            std::thread::Builder::new()
                .name("layer-prefetch".into())
                .spawn(move || run_prefetch(file, ranges, current, stop, depth))
                .ok()
        };
        Self {
            current,
            stop,
            join,
        }
    }

    /// Tell the thread which layer the compute thread is on.  Called once per
    /// layer, at its start.
    pub fn set_current(&self, layer: usize) {
        self.current.store(layer, Ordering::Release);
    }

    /// Signal the thread to stop and wait for it to exit.
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

#[cfg(unix)]
impl Drop for LayerPrefetcher {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(unix)]
fn run_prefetch(
    file: Arc<File>,
    ranges: Arc<Vec<Option<(usize, usize)>>>,
    current: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    depth: usize,
) {
    use std::os::unix::fs::FileExt;
    let mut buf = vec![0u8; PREFETCH_CHUNK];
    let file_len = file.metadata().map(|m| m.len() as usize).unwrap_or(0);
    // Next layer index whose range we still need to stream.  Monotonic: the
    // thread never re-reads a range, only catches up when the caller advances.
    let mut next = 0usize;
    while !stop.load(Ordering::Acquire) {
        let cur = current.load(Ordering::Acquire);
        let target = cur.saturating_add(depth).min(ranges.len());
        let mut advanced = false;
        while next < target {
            if let Some((b0, e1)) = ranges[next] {
                if b0 < e1 && e1 <= file_len {
                    let mut off = b0;
                    // Best effort: on error, skip the rest of this range.
                    let mut ok = true;
                    while ok && off < e1 {
                        let n = (e1 - off).min(buf.len());
                        match file.read_at(&mut buf[..n], off as u64) {
                            Ok(0) => break,
                            Ok(k) => off += k,
                            Err(_) => ok = false,
                        }
                    }
                }
            }
            next += 1;
            advanced = true;
        }
        if !advanced {
            // Caught up with cur + depth: wait for the compute thread to move.
            std::thread::sleep(PREFETCH_POLL);
        }
    }
}

/// Non-unix builds get a no-op handle so the model code compiles unchanged.
#[cfg(not(unix))]
pub struct LayerPrefetcher;

#[cfg(not(unix))]
impl LayerPrefetcher {
    pub fn spawn(
        _file: Arc<File>,
        _ranges: Arc<Vec<Option<(usize, usize)>>>,
        _depth: usize,
    ) -> Self {
        Self
    }
    pub fn set_current(&self, _layer: usize) {}
    pub fn stop(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::QTensor as QT;
    use candle_core::{Device, Tensor};

    /// Write a one-tensor GGUF and map it.
    fn gguf_with_tensor(dir: &std::path::Path, data: &[f32], shape: &[usize]) -> Arc<Mmap> {
        let path = dir.join("t.gguf");
        let t = Tensor::from_vec(data.to_vec(), shape, &Device::Cpu).unwrap();
        let q = QT::quantize(&t, GgmlDType::F32).unwrap();
        let mut f = std::fs::File::create(&path).unwrap();
        gguf_file::write(&mut f, &[], &[("w", &q)]).unwrap();
        drop(f);
        let f = std::fs::File::open(&path).unwrap();
        Arc::new(unsafe { Mmap::map(&f) }.unwrap())
    }

    #[test]
    fn borrowed_tensor_matches_copied_tensor_and_shares_the_mapping() {
        let dir = std::env::temp_dir().join(format!("joshua-mmapt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.25 - 8.0).collect();
        let mmap = gguf_with_tensor(&dir, &data, &[8, 8]);

        let mut cursor = std::io::Cursor::new(&mmap[..]);
        let content = gguf_file::Content::read(&mut cursor).unwrap();
        let info = content.tensor_infos.get("w").unwrap();

        let borrowed = borrowed_qtensor(&mmap, info, content.tensor_data_offset)
            .unwrap()
            .expect("F32 tensor should be borrowable");

        // Same values as candle's copying path.
        let copied = content.tensor(&mut cursor, "w", &Device::Cpu).unwrap();
        let a: Vec<f32> = borrowed
            .dequantize(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let b: Vec<f32> = copied
            .dequantize(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(a, b, "borrowed and copied tensors must agree");
        assert_eq!(a, data);

        // The borrow really points into the mapping rather than a copy.
        let base = mmap.as_ptr() as usize;
        let ptr = borrowed.data().unwrap().as_ptr() as usize;
        assert!(
            ptr >= base && ptr < base + mmap.len(),
            "borrowed tensor must point inside the mapping"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn out_of_bounds_tensor_declines_the_borrow() {
        let dir = std::env::temp_dir().join(format!("joshua-mmapt-oob-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let data: Vec<f32> = vec![1.0; 32];
        let mmap = gguf_with_tensor(&dir, &data, &[4, 8]);
        let mut cursor = std::io::Cursor::new(&mmap[..]);
        let content = gguf_file::Content::read(&mut cursor).unwrap();
        let info = content.tensor_infos.get("w").unwrap();

        // An offset past the end of the file must decline, not read garbage.
        let bogus = gguf_file::TensorInfo {
            ggml_dtype: info.ggml_dtype,
            shape: info.shape.clone(),
            offset: mmap.len() as u64,
        };
        assert!(borrowed_qtensor(&mmap, &bogus, content.tensor_data_offset)
            .unwrap()
            .is_none());

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Fraction of the mapping's pages currently resident (0.0–1.0).
    #[cfg(target_os = "linux")]
    fn resident_fraction(mmap: &Mmap) -> f64 {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let len = mmap.len();
        let n = len.div_ceil(page);
        let mut vec = vec![0u8; n];
        let rc = unsafe {
            libc::mincore(
                mmap.as_ptr() as *mut libc::c_void,
                len,
                vec.as_mut_ptr() as *mut libc::c_uchar,
            )
        };
        assert_eq!(rc, 0, "mincore failed");
        let resident = vec.iter().filter(|b| *b & 0x1 != 0).count();
        resident as f64 / n as f64
    }

    /// The prefetch thread must stream a dropped range back into the page cache
    /// so that later mmap faults are hits.
    #[cfg(target_os = "linux")]
    #[test]
    fn prefetcher_warms_page_cache() {
        use std::os::unix::fs::FileExt;
        use std::os::unix::io::AsRawFd;

        let dir = std::env::temp_dir().join(format!("joshua-pf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.bin");
        let len = 32 * 1024 * 1024;
        let mut f = std::fs::File::create(&path).unwrap();
        // Write real data: sparse holes are served from the shared zero page
        // and never populate the page cache, which would make the residency
        // assertion vacuous.  Real GGUFs are dense.
        {
            use std::io::Write;
            let chunk = vec![0x5au8; 1024 * 1024];
            let mut written = 0usize;
            while written < len {
                f.write_all(&chunk).unwrap();
                written += chunk.len();
            }
        }
        f.sync_all().unwrap();
        drop(f);
        // Reopen read-only before mapping, like the GGUF tests do.
        let file = Arc::new(std::fs::File::open(&path).unwrap());
        let mmap = Arc::new(unsafe { Mmap::map(&file).unwrap() });

        // Drop the file's pages from the cache so the test starts cold.
        let rc = unsafe {
            libc::posix_fadvise(
                file.as_raw_fd(),
                0,
                len as libc::off_t,
                libc::POSIX_FADV_DONTNEED,
            )
        };
        assert_eq!(rc, 0, "posix_fadvise failed");
        // Give the kernel a moment to actually drop them.
        std::thread::sleep(std::time::Duration::from_millis(50));
        let cold = resident_fraction(&mmap);
        if cold > 0.9 {
            // Cache not droppable here (e.g. tmpfs-backed /tmp): the residency
            // assertion would be vacuous, so skip rather than flake.
            eprintln!("prefetch test skipped: pages not droppable (resident {cold:.2})");
            drop(mmap);
            std::fs::remove_dir_all(&dir).ok();
            return;
        }

        let ranges: Vec<Option<(usize, usize)>> = vec![Some((0, len))];
        let mut pf = LayerPrefetcher::spawn(
            Arc::clone(&file),
            Arc::new(ranges),
            PREFETCH_AHEAD_DEPTH,
        );
        pf.set_current(0);

        // Poll until the thread has streamed the whole range (or timeout).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        let mut resident = 0.0;
        while std::time::Instant::now() < deadline {
            resident = resident_fraction(&mmap);
            if resident > 0.99 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        pf.stop();
        assert!(
            resident > 0.99,
            "prefetch thread did not warm the cache: resident {resident:.3}"
        );

        // Sanity: the warm pages are readable via the mapping.
        let mut probe = vec![0u8; 4096];
        let n = file.read_at(&mut probe, 0).unwrap();
        assert_eq!(n, 4096);

        drop(mmap);
        std::fs::remove_dir_all(&dir).ok();
    }
}
