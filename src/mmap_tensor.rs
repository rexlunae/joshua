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
        k_quants::matmul(mkn, lhs, self.blocks(), dst)
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
}
