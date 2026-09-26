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
use crate::raw_block::{self, RawBlock};

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

    /// The byte range this handle covers inside its mapping, for page
    /// accounting and release ([`MappedRange`]).  `None` for handles that
    /// are not backed by a mapping (test doubles).
    fn mapped_range(&self) -> Option<MappedRange> {
        None
    }
}

/// A byte range inside a read-only file mapping: the unit the expert
/// residency code accounts for and releases.
///
/// The mapping covers the model file from offset 0, so `offset` is also the
/// file offset — which is what [`MappedRange::evict_from_cache`] needs.
#[derive(Clone)]
pub struct MappedRange {
    mmap: Arc<Mmap>,
    /// Byte offset of the range inside the mapping (and the file).
    pub offset: usize,
    /// Length in bytes.
    pub len: usize,
}

impl std::fmt::Debug for MappedRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedRange")
            .field("offset", &self.offset)
            .field("len", &self.len)
            .finish()
    }
}

/// The system page size (bytes).
fn page_size() -> usize {
    #[cfg(unix)]
    {
        // SAFETY: sysconf has no preconditions.
        let n = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if n > 0 {
            return n as usize;
        }
    }
    4096
}

impl MappedRange {
    fn new(mmap: &Arc<Mmap>, ptr: *const u8, len: usize) -> Self {
        let offset = ptr as usize - mmap.as_ptr() as usize;
        Self {
            mmap: Arc::clone(mmap),
            offset,
            len,
        }
    }

    /// The whole pages strictly inside the range, `(offset, len)`; `None`
    /// when the range holds no whole page.  Partial pages at either end are
    /// shared with a neighbouring tensor, so page-granular release leaves
    /// them alone.
    fn inner_pages(&self) -> Option<(usize, usize)> {
        let page = page_size();
        let start = self.offset.div_ceil(page) * page;
        let end = (self.offset + self.len) / page * page;
        (end > start).then_some((start, end - start))
    }

    /// The whole pages that cover the range (start rounded down, end up).
    fn covering_pages(&self) -> (usize, usize) {
        let page = page_size();
        let start = self.offset / page * page;
        let end = (self.offset + self.len).div_ceil(page) * page;
        (start, end - start)
    }

    /// How many of the pages covering this range are resident in memory
    /// right now, as `(resident, total)`.  `None` where the kernel cannot
    /// tell us (non-Linux, or a `mincore` failure).
    pub fn resident_pages(&self) -> Option<(usize, usize)> {
        #[cfg(target_os = "linux")]
        {
            let (start, len) = self.covering_pages();
            if len == 0 || start + len > self.mmap.len().div_ceil(page_size()) * page_size() {
                return None;
            }
            let page = page_size();
            let n = len.div_ceil(page);
            let mut vec = vec![0u8; n];
            // SAFETY: `start` is page-aligned and `[start, start + len)` lies
            // within the mapping (checked above, rounded to whole pages the
            // mapping itself occupies); `vec` holds one byte per page.
            let rc = unsafe {
                libc::mincore(
                    self.mmap.as_ptr().add(start) as *mut libc::c_void,
                    len,
                    vec.as_mut_ptr() as *mut libc::c_uchar,
                )
            };
            if rc != 0 {
                return None;
            }
            let resident = vec.iter().filter(|b| *b & 0x1 != 0).count();
            return Some((resident, n));
        }
        #[allow(unreachable_code)]
        None
    }

    /// Drop this range's whole pages from the process (`MADV_DONTNEED`), so
    /// the mapping no longer holds them and the page cache may reclaim them.
    /// Safe on a read-only file mapping: the next touch re-faults the same
    /// bytes from the file.  Best-effort; a no-op off Unix.
    pub fn drop_pages(&self) {
        #[cfg(unix)]
        if let Some((start, len)) = self.inner_pages() {
            if start + len <= self.mmap.len() {
                // SAFETY: page-aligned range inside a read-only, file-backed
                // mapping; DONTNEED on such a mapping only discards clean
                // pages that re-fault from the file.
                let _ = unsafe {
                    libc::madvise(
                        self.mmap.as_ptr().add(start) as *mut libc::c_void,
                        len,
                        libc::MADV_DONTNEED,
                    )
                };
            }
        }
    }

    /// Ask the kernel to evict this range's whole pages from the page cache
    /// (`posix_fadvise(POSIX_FADV_DONTNEED)` on `file`, the mapped file).
    /// Only pages no mapping still holds are dropped, so call
    /// [`MappedRange::drop_pages`] first.  Best-effort; a no-op off Linux.
    pub fn evict_from_cache(&self, file: &std::fs::File) {
        #[cfg(target_os = "linux")]
        if let Some((start, len)) = self.inner_pages() {
            use std::os::unix::io::AsRawFd;
            // SAFETY: a plain advisory syscall on an open descriptor.
            let _ = unsafe {
                libc::posix_fadvise(
                    file.as_raw_fd(),
                    start as libc::off_t,
                    len as libc::off_t,
                    libc::POSIX_FADV_DONTNEED,
                )
            };
        }
        #[cfg(not(target_os = "linux"))]
        let _ = file;
    }
}

impl<T: GgmlType + 'static> MmapPrefetch for MmapBlocks<T> {
    fn prefetch(&self) {
        let base = self._mmap.as_ptr() as usize;
        let off = self.ptr as usize - base;
        let len = self.len * std::mem::size_of::<T>();
        let _ = self._mmap.advise_range(memmap2::Advice::WillNeed, off, len);
    }

    fn mapped_range(&self) -> Option<MappedRange> {
        Some(MappedRange::new(
            &self._mmap,
            self.ptr as *const u8,
            self.len * std::mem::size_of::<T>(),
        ))
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

/// Blocks of a [`RawBlock`] format (the i-quants, TQ1_0 / TQ2_0, MXFP4 /
/// NVFP4, Q1_0 / Q2_0) borrowed from the mapping — or, for a tensor read
/// through a copying reader, held on the heap in the same compact form.
///
/// Deliberately not `MmapBlocks<T>`: that runs candle's `GgmlType` kernels
/// (Q8-quantized activations; and candle has no type at all for most of
/// these formats), while this storage runs [`RawBlock::matmul_t`] — f32
/// activations, a block decoded at a time.  IQ2_XXS reports its real dtype
/// so a device upload of these bytes (`QStorage::from_data` with
/// `qt.dtype()`) lands as IQ2_XXS device storage.
pub struct RawBlocks<B: RawBlock> {
    backing: RawBacking<B>,
}

enum RawBacking<B> {
    Mapped {
        /// Keeps the mapping alive; never dereferenced directly.
        mmap: Arc<Mmap>,
        ptr: *const B,
        len: usize,
    },
    Owned(Vec<B>),
}

// SAFETY: same argument as `MmapBlocks`: a read-only mapping kept alive by
// `mmap`, blocks never mutated or moved; the owned form is a plain `Vec`.
unsafe impl<B: RawBlock> Send for RawBlocks<B> {}
unsafe impl<B: RawBlock> Sync for RawBlocks<B> {}

impl<B: RawBlock> MmapPrefetch for RawBlocks<B> {
    fn prefetch(&self) {
        if let RawBacking::Mapped { mmap, ptr, .. } = &self.backing {
            let off = *ptr as usize - mmap.as_ptr() as usize;
            let _ = mmap.advise_range(memmap2::Advice::WillNeed, off, self.size());
        }
    }

    fn mapped_range(&self) -> Option<MappedRange> {
        match &self.backing {
            RawBacking::Mapped { mmap, ptr, .. } => Some(MappedRange::new(mmap, *ptr as *const u8, self.size())),
            RawBacking::Owned(_) => None,
        }
    }
}

impl<B: RawBlock> RawBlocks<B> {
    fn blocks(&self) -> &[B] {
        match &self.backing {
            // SAFETY: bounds-checked in `borrow` (blocks are align 1, so only
            // bounds matter) against a mapping that `mmap` keeps alive and
            // that is never mutated.
            RawBacking::Mapped { ptr, len, .. } => unsafe { std::slice::from_raw_parts(*ptr, *len) },
            RawBacking::Owned(v) => v,
        }
    }

    fn borrow(mmap: &Arc<Mmap>, byte_offset: usize, n_blocks: usize) -> Option<Self> {
        let size = n_blocks.checked_mul(raw_block::block_bytes::<B>())?;
        let end = byte_offset.checked_add(size)?;
        if end > mmap.len() {
            return None;
        }
        // SAFETY: `byte_offset <= end <= mmap.len()`.
        let ptr = unsafe { mmap.as_ptr().add(byte_offset) };
        Some(Self {
            backing: RawBacking::Mapped {
                mmap: Arc::clone(mmap),
                ptr: ptr as *const B,
                len: n_blocks,
            },
        })
    }
}

/// A [`RawBlock`] tensor held on the heap as its file bytes (`shape`'s
/// element count must be whole blocks).
pub fn owned_qtensor_raw<B: RawBlock>(bytes: &[u8], shape: candle_core::Shape) -> Result<QTensor> {
    let blocks = raw_block::blocks_from_bytes::<B>(bytes)?.to_vec();
    if blocks.len() * B::QK != shape.elem_count() {
        candle_core::bail!("{}: {} blocks for a tensor of shape {shape:?}", B::NAME, blocks.len());
    }
    let storage: Box<dyn QuantizedType> = Box::new(RawBlocks { backing: RawBacking::Owned(blocks) });
    QTensor::new(QStorage::Cpu(storage), shape)
}

impl<B: RawBlock> QuantizedType for RawBlocks<B> {
    fn dtype(&self) -> GgmlDType {
        B::CANDLE_DTYPE
    }

    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()> {
        B::matmul_t(mkn, lhs, self.blocks(), dst)
    }

    fn matmul_t_f16(&self, mkn: (usize, usize, usize), lhs: &[f16], dst: &mut [f16]) -> Result<()> {
        raw_block::matmul_t_f16(mkn, lhs, self.blocks(), dst)
    }

    fn embedding(&self, ids: &[u32], rows: usize, hidden: usize) -> Result<CpuStorage> {
        if !hidden.is_multiple_of(B::QK) {
            candle_core::bail!(
                "quantized embedding hidden size {hidden} is not divisible by block size {}",
                B::QK
            )
        }
        let blocks = self.blocks();
        let row_blocks = hidden / B::QK;
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
            raw_block::dequantize(src, dst)?;
        }
        Ok(CpuStorage::F32(out))
    }

    fn dequantize(&self, elem_count: usize) -> Result<CpuStorage> {
        let mut ys = vec![0.0f32; elem_count];
        raw_block::dequantize(self.blocks(), &mut ys)?;
        Ok(CpuStorage::F32(ys))
    }

    fn storage_size_in_bytes(&self) -> usize {
        self.size()
    }

    fn as_ptr(&self) -> *const u8 {
        self.blocks().as_ptr() as *const u8
    }

    fn block_size(&self) -> usize {
        B::QK
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self.blocks())
    }

    fn from_float(&mut self, _xs: &[f32]) {
        panic!("cannot quantize into a raw-format tensor (decode-only)")
    }

    fn from_float_imatrix(&mut self, _xs: &[f32], _imatrix_weights: &[f32], _n_per_row: usize) {
        panic!("cannot quantize into a raw-format tensor (decode-only)")
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
        // The IQ2_XXS mmap form is `RawBlocks` (`prefetch_handle_raw`).
        GgmlDType::Iq2Xxs => None,
    }
}

/// Prefetch handle for `n_blocks` [`RawBlock`] blocks; see [`prefetch_handle`].
pub fn prefetch_handle_raw<B: RawBlock>(
    mmap: &Arc<Mmap>,
    byte_offset: usize,
    n_blocks: usize,
) -> Option<Arc<dyn MmapPrefetch>> {
    RawBlocks::<B>::borrow(mmap, byte_offset, n_blocks)
        .map(|b| Arc::new(b) as Arc<dyn MmapPrefetch>)
}

/// Borrow a tensor by raw GGUF dtype id, falling back to candle's table.
///
/// `dtype` is the raw header id, so [`RawBlock`] formats candle cannot name
/// (IQ2_XXS, MXFP4, Q1_0, Q2_0) can still be borrowed.  Returns `Ok(None)`
/// when the dtype is unknown or the range cannot be borrowed safely.
pub fn borrowed_qtensor_raw(
    mmap: &Arc<Mmap>,
    dtype: u32,
    offset: u64,
    tensor_data_offset: u64,
    shape: candle_core::Shape,
) -> Result<Option<QTensor>> {
    if raw_block::is_raw_block(dtype) {
        let Ok(off) = usize::try_from(tensor_data_offset.saturating_add(offset)) else {
            return Ok(None);
        };
        return crate::with_raw_block!(dtype, B => borrowed_range_raw::<B>(mmap, off, shape))
            .unwrap_or(Ok(None));
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

/// Borrow a [`RawBlock`] tensor (or per-expert slice of one) from the
/// mapping.
pub fn borrowed_range_raw<B: RawBlock>(
    mmap: &Arc<Mmap>,
    offset: usize,
    shape: candle_core::Shape,
) -> Result<Option<QTensor>> {
    let elem_count = shape.elem_count();
    if !elem_count.is_multiple_of(B::QK) {
        return Ok(None);
    }
    let Some(blocks) = RawBlocks::<B>::borrow(mmap, offset, elem_count / B::QK) else {
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
        // The IQ2_XXS mmap form is `RawBlocks` (`borrowed_range_raw`),
        // which keeps the fused AVX2 matmul instead of candle's reference decode.
        GgmlDType::Iq2Xxs => candle_core::bail!(
            "IQ2_XXS tensors are borrowed with `borrowed_range_raw`, not the generic block borrow"
        ),
    };

    QTensor::new(QStorage::Cpu(storage), shape).map(Some)
}

/// The byte slices of each expert in a stacked `[n_expert, out, in]` expert
/// tensor, straight out of the mapping.
///
/// This is the *upload* counterpart of [`borrowed_range`]: where a CPU model
/// points each expert at its bytes in place, a model running on an
/// accelerator needs those same bytes copied onto the device.  Slicing the
/// mapping directly hands the device copy its source without any host-side
/// staging — no `Vec` read of the whole tensor, no round trip through a
/// whole-tensor device buffer.  The bytes at `[e * per_bytes, (e + 1) *
/// per_bytes)` of the tensor are exactly expert `e`'s quantized blocks, in
/// the order candle's own `QTensor::data` would return them.
///
/// Returns `None` when the tensor's element count is not a whole number of
/// blocks, or the range runs past the mapping (a truncated file) — callers
/// then fall back to candle's copying reader.
pub fn expert_slices(
    mmap: &Mmap,
    dtype: GgmlDType,
    byte_offset: usize,
    n_expert: usize,
    per_expert_elems: usize,
) -> Option<Vec<&[u8]>> {
    let block_size = dtype.block_size();
    if block_size == 0 || n_expert == 0 || !per_expert_elems.is_multiple_of(block_size) {
        return None;
    }
    let per_bytes = (per_expert_elems / block_size).checked_mul(dtype.type_size())?;
    let total = per_bytes.checked_mul(n_expert)?;
    let end = byte_offset.checked_add(total)?;
    if end > mmap.len() {
        return None;
    }
    Some(
        (0..n_expert)
            .map(|e| &mmap[byte_offset + e * per_bytes..byte_offset + (e + 1) * per_bytes])
            .collect(),
    )
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
    if let Some(info) = content.tensor_infos.get(name) {
        // Borrowing CPU block storage is only sound on the CPU device.
        if device.is_cpu() {
            if let Some(t) = borrowed_qtensor(mmap, info, content.tensor_data_offset)? {
                return Ok(t);
            }
            tracing::debug!(
                tensor = name,
                "tensor not borrowable from the mapping, copying instead"
            );
        }
        // An OpenCL device with host-unified memory (an iGPU, or a CPU
        // runtime) can alias the mapped pages directly instead of holding a
        // second copy of the weights.
        #[cfg(feature = "opencl")]
        if let Some(t) = opencl_zero_copy_qtensor(mmap, info, content.tensor_data_offset, device)? {
            return Ok(t);
        }
    }
    content.tensor(reader, name, device)
}

/// Wrap a tensor's mapped bytes as a zero-copy OpenCL buffer
/// (`CL_MEM_USE_HOST_PTR`) when the device shares memory with the host.
///
/// On an iGPU the device reads DRAM either way; aliasing the page cache saves
/// the whole second copy of the weights and lets the kernel evict clean pages
/// under pressure, exactly as on the CPU.  A discrete GPU (no unified memory)
/// is better served by an explicit upload, so `Ok(None)` sends the caller to
/// the copying reader.  `JOSHUA_OPENCL_ZERO_COPY=0` disables aliasing.
#[cfg(feature = "opencl")]
fn opencl_zero_copy_qtensor(
    mmap: &Arc<Mmap>,
    info: &gguf_file::TensorInfo,
    tensor_data_offset: u64,
    device: &candle_core::Device,
) -> Result<Option<QTensor>> {
    let Ok(ocl) = device.as_opencl_device() else {
        return Ok(None);
    };
    if !candle_core::opencl_backend::zero_copy_enabled() || !ocl.host_unified_memory() {
        return Ok(None);
    }
    let Ok(offset) = usize::try_from(tensor_data_offset.saturating_add(info.offset)) else {
        return Ok(None);
    };
    let dtype = info.ggml_dtype;
    let elem_count = info.shape.elem_count();
    let block_size = dtype.block_size();
    if block_size == 0 || !elem_count.is_multiple_of(block_size) {
        return Ok(None);
    }
    let bytes = (elem_count / block_size).checked_mul(dtype.type_size());
    if bytes
        .and_then(|b| offset.checked_add(b))
        .is_none_or(|end| end > mmap.len())
    {
        return Ok(None);
    }
    let keepalive: Arc<dyn std::any::Any + Send + Sync> = mmap.clone();
    // SAFETY: the mapping is read-only and immutable for the life of the
    // process (the module-level safety model), the range was bounds-checked
    // above, and `keepalive` holds the mapping for as long as the buffer.
    let storage = unsafe {
        candle_core::QOpenClStorage::from_host_mapping(
            ocl,
            dtype,
            elem_count,
            mmap.as_ptr(),
            mmap.len(),
            offset,
            keepalive,
        )
    };
    match storage {
        Ok(storage) => QTensor::new(QStorage::OpenCl(storage), info.shape.clone()).map(Some),
        Err(e) => {
            tracing::debug!(error = %e, "opencl zero-copy mapping declined, uploading instead");
            Ok(None)
        }
    }
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
///
/// The depth is **byte-budgeted by available RAM** (architecture finding #3):
/// on a small machine 3 layers of read-ahead can evict useful pages, so the
/// lead shrinks to 2 / 1 / 0 as free memory drops.  The budget floor is chosen
/// so the default (plenty of RAM) keeps the full 3-layer lead.
pub fn prefetch_ahead_depth() -> usize {
    match crate::placement::available_ram_bytes() {
        Some(free) => {
            let free_gib = free / (1024 * 1024 * 1024);
            if free_gib < 2 {
                0
            } else if free_gib < 4 {
                1
            } else if free_gib < 8 {
                2
            } else {
                3
            }
        }
        None => 3, // cannot probe; keep the default lead
    }
}

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
    // Next layer index whose range we still need to stream.  Never re-reads a
    // range; if the compute thread overtakes us we skip forward to it rather
    // than streaming layers that have already been consumed.
    let mut next = 0usize;
    while !stop.load(Ordering::Acquire) {
        let cur = current.load(Ordering::Acquire);
        next = next.max(cur);
        let target = cur.saturating_add(depth).min(ranges.len());
        let mut advanced = false;
        while next < target && !stop.load(Ordering::Acquire) {
            if let Some((b0, e1)) = ranges[next] {
                if b0 < e1 && e1 <= file_len {
                    let mut off = b0;
                    // Best effort: on error, skip the rest of this range.
                    let mut ok = true;
                    while ok && off < e1 && !stop.load(Ordering::Acquire) {
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
    use candle_core::quantized::QMatMul;
    use candle_core::quantized::QTensor as QT;
    use candle_core::{Device, Module, Tensor};

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

    /// Map a raw byte blob as if it were a (synthetic) model file.
    fn mmap_bytes(dir: &std::path::Path, bytes: &[u8]) -> Arc<Mmap> {
        let path = dir.join("t.bin");
        std::fs::write(&path, bytes).unwrap();
        let f = std::fs::File::open(&path).unwrap();
        Arc::new(unsafe { Mmap::map(&f) }.unwrap())
    }

    /// Build `n_rows` MXFP4 weight rows of `k` elements with deterministic
    /// data, returning (blocks, dequantized weights).
    fn mxfp4_rows(n_rows: usize, k: usize) -> (Vec<crate::mxfp4::BlockMxfp4>, Vec<f32>) {
        use crate::mxfp4::{BlockMxfp4, QK_MXFP4};
        let blocks_per_row = k / QK_MXFP4;
        let mut blocks = Vec::new();
        for r in 0..n_rows {
            for b in 0..blocks_per_row {
                let mut qs = [0u8; 16];
                for (j, q) in qs.iter_mut().enumerate() {
                    *q = (((j + r * 3 + b * 5) % 16) as u8) | ((((j + r) % 16) as u8) << 4);
                }
                blocks.push(BlockMxfp4 {
                    e: 120 + (r as u8 % 7), // scales 2^-7 .. 2^-1
                    qs,
                });
            }
        }
        let mut weights = vec![0f32; n_rows * k];
        crate::raw_block::dequantize(&blocks, &mut weights).unwrap();
        (blocks, weights)
    }

    #[test]
    fn borrowed_mxfp4_qtensor_matmul_matches_reference() {
        let dir = std::env::temp_dir().join(format!("joshua-mxfp4-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let (n_rows, k) = (6usize, 64usize); // 2 blocks per row
        let (blocks, weights) = mxfp4_rows(n_rows, k);
        let bytes: Vec<u8> = blocks
            .iter()
            .flat_map(|b| {
                let mut v = Vec::with_capacity(17);
                v.push(b.e);
                v.extend_from_slice(&b.qs);
                v
            })
            .collect();
        let mmap = mmap_bytes(&dir, &bytes);

        let qt = borrowed_range_raw::<crate::mxfp4::BlockMxfp4>(&mmap, 0, (n_rows, k).into())
            .unwrap()
            .expect("MXFP4 range should borrow");
        let qmm = QMatMul::from_qtensor(qt).unwrap();

        let lhs: Vec<f32> = (0..2 * k).map(|i| (i as f32 % 7.0) - 3.0).collect();
        let xs = Tensor::from_vec(lhs.clone(), (2, k), &Device::Cpu).unwrap();
        let got = qmm.forward(&xs).unwrap();
        let got: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();

        // Reference: dequantized weights, plain dot products.
        let mut expect = vec![0f32; 2 * n_rows];
        for i in 0..2 {
            for r in 0..n_rows {
                expect[i * n_rows + r] =
                    (0..k).map(|j| lhs[i * k + j] * weights[r * k + j]).sum();
            }
        }
        assert!(
            got.iter().zip(&expect).all(|(g, e)| (g - e).abs() < 1e-3),
            "got {got:?}, expected {expect:?}"
        );

        // A per-expert slice (3 rows at a 3-row offset) must also borrow and
        // agree with the same reference rows.
        let qt2 = borrowed_range_raw::<crate::mxfp4::BlockMxfp4>(&mmap, 3 * 2 * 17, (3, k).into())
            .unwrap()
            .expect("sliced MXFP4 range should borrow");
        let qmm2 = QMatMul::from_qtensor(qt2).unwrap();
        let got2 = qmm2.forward(&xs).unwrap();
        let got2: Vec<f32> = got2.flatten_all().unwrap().to_vec1().unwrap();
        for i in 0..2 {
            for r in 0..3 {
                assert!(
                    (got2[i * 3 + r] - expect[i * 6 + (3 + r)]).abs() < 1e-3,
                    "slice mismatch at [{i},{r}]"
                );
            }
        }

        // Out-of-bounds borrows decline.
        assert!(borrowed_range_raw::<crate::mxfp4::BlockMxfp4>(&mmap, bytes.len(), (n_rows, k).into())
            .unwrap()
            .is_none());

        std::fs::remove_dir_all(&dir).ok();
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

    /// A mapped range reports its page residency, `drop_pages` +
    /// `evict_from_cache` release it, and touching it again re-faults the
    /// same bytes.
    ///
    /// `mincore` reports page-cache residency for a file mapping, and
    /// `posix_fadvise(DONTNEED)` only frees the page-cache folios that lie
    /// wholly inside the range (a large folio straddling an edge is
    /// deactivated instead), so the assertion is "most of a 6 MiB range
    /// goes", not "every page".
    #[cfg(target_os = "linux")]
    #[test]
    fn mapped_range_accounts_and_releases_pages() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("joshua-mr-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("blocks.bin");
        let len = 12 * 1024 * 1024;
        {
            let mut f = std::fs::File::create(&path).unwrap();
            let chunk = vec![0x3cu8; 1024 * 1024];
            for _ in 0..(len / chunk.len()) {
                f.write_all(&chunk).unwrap();
            }
            f.sync_all().unwrap();
        }
        let file = Arc::new(std::fs::File::open(&path).unwrap());
        let mmap = Arc::new(unsafe { Mmap::map(&file).unwrap() });
        // An f32 handle over the middle 6 MiB, at a deliberately unaligned
        // (but 4-byte aligned) offset so partial edge pages exist.
        let n_blocks = 6 * 1024 * 1024 / 4;
        let handle = MmapBlocks::<f32>::borrow(&mmap, 3 * 1024 * 1024 + 100, n_blocks).unwrap();
        let range = handle.mapped_range().unwrap();
        assert_eq!(range.offset, 3 * 1024 * 1024 + 100);
        assert_eq!(range.len, n_blocks * 4);

        // Touch every byte: the pages are resident afterwards.
        let expect = u64::from(0x3c3c3c3cu32) * n_blocks as u64;
        let sum: u64 = handle.blocks().iter().map(|v| u64::from(v.to_bits())).sum();
        assert_eq!(sum, expect);
        let (res, total) = range.resident_pages().expect("mincore works on linux");
        assert!(total > 0);
        assert_eq!(res, total, "just-touched pages are resident");

        range.drop_pages();
        range.evict_from_cache(&file);
        let (res_after, _) = range.resident_pages().unwrap();
        if res_after == res {
            // Can this environment drop file pages at all?  (tmpfs cannot.)
            // A whole-file DONTNEED is the reference; skip rather than
            // flake when even that keeps everything resident.
            use std::os::unix::io::AsRawFd;
            unsafe {
                libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
            }
            if range.resident_pages().unwrap().0 == res {
                eprintln!("mapped_range test skipped: pages not droppable here");
                drop(handle);
                drop(mmap);
                std::fs::remove_dir_all(&dir).ok();
                return;
            }
            panic!("range release dropped nothing although a whole-file release does");
        }
        // At least the whole 2 MiB-aligned folios inside the range are gone:
        // two of them fit in a 6 MiB range at any alignment.
        assert!(
            res_after + 2 * 512 <= res,
            "resident after release: {res_after}/{total}"
        );

        // Re-touching re-faults the same bytes.
        let sum: u64 = handle.blocks().iter().map(|v| u64::from(v.to_bits())).sum();
        assert_eq!(sum, expect);
        let (res, _) = range.resident_pages().unwrap();
        assert_eq!(res, total);

        drop(handle);
        drop(mmap);
        std::fs::remove_dir_all(&dir).ok();
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
            prefetch_ahead_depth(),
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

    /// `expert_slices` must hand out exactly the bytes candle's own reader
    /// would produce for each expert of a stacked expert tensor — this is
    /// what lets an accelerator load upload experts straight from the
    /// mapping instead of staging the whole tensor on the host and device.
    #[test]
    fn expert_slices_match_candle_tensor_data() {
        let dir = std::env::temp_dir().join(format!("joshua-exps-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (n_expert, out, inn) = (3usize, 4usize, 64usize);
        let data: Vec<f32> = (0..n_expert * out * inn).map(|i| (i % 23) as f32 - 11.0).collect();
        let t = Tensor::from_vec(data, (n_expert, out, inn), &Device::Cpu).unwrap();
        let q = QT::quantize(&t, GgmlDType::Q8_0).unwrap();
        let path = dir.join("e.gguf");
        let mut f = std::fs::File::create(&path).unwrap();
        gguf_file::write(&mut f, &[], &[("blk.0.ffn_up_exps.weight", &q)]).unwrap();
        drop(f);
        let f = std::fs::File::open(&path).unwrap();
        let mmap = Arc::new(unsafe { Mmap::map(&f) }.unwrap());
        let content = gguf_file::Content::read(&mut std::io::Cursor::new(&mmap[..])).unwrap();
        let info = &content.tensor_infos["blk.0.ffn_up_exps.weight"];
        let base = (content.tensor_data_offset + info.offset) as usize;

        let slices = expert_slices(&mmap, GgmlDType::Q8_0, base, n_expert, out * inn).unwrap();
        let expected = q.data().unwrap();
        let per = expected.len() / n_expert;
        assert_eq!(slices.len(), n_expert);
        for (e, s) in slices.iter().enumerate() {
            assert_eq!(s.len(), per);
            assert_eq!(*s, &expected[e * per..(e + 1) * per], "expert {e} bytes differ");
        }
        // Each slice is a valid expert matrix that matmuls like the reference.
        let xs = Tensor::from_vec((0..inn).map(|i| (i % 5) as f32).collect::<Vec<_>>(), (1, inn), &Device::Cpu)
            .unwrap();
        for (e, s) in slices.iter().enumerate() {
            let st = QStorage::from_data(std::borrow::Cow::Borrowed(s), &Device::Cpu, GgmlDType::Q8_0).unwrap();
            let qt = QT::new(st, (out, inn)).unwrap();
            let got = QMatMul::from_qtensor(qt).unwrap().forward(&xs).unwrap();
            // `get` yields an offset view; rebuild it so quantize sees only
            // this expert's elements.
            let ref_e: Vec<f32> = t.get(e).unwrap().flatten_all().unwrap().to_vec1().unwrap();
            let ref_e = Tensor::from_vec(ref_e, (out, inn), &Device::Cpu).unwrap();
            let want = QMatMul::from_qtensor(QT::quantize(&ref_e, GgmlDType::Q8_0).unwrap())
                .unwrap()
                .forward(&xs)
                .unwrap();
            let diff = (got - want).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
            assert!(diff < 1e-5, "expert {e} matmul differs by {diff}");
        }

        // Truncated ranges and non-block element counts are declined.
        assert!(expert_slices(&mmap, GgmlDType::Q8_0, base, n_expert + 1, out * inn).is_none());
        assert!(expert_slices(&mmap, GgmlDType::Q8_0, base, n_expert, out * inn + 1).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }
}
