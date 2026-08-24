//! Zero-copy quantized weights on Metal.
//!
//! candle's Metal path copies every weight into a fresh GPU buffer at load
//! time — and MoE splitters add more transient copies while carving out
//! per-expert tensors — so a memory-mapped model ends up resident **three
//! times** (file pages + CPU copy + GPU buffer).  On unified-memory Apple
//! Silicon that is what tips a 24 GB machine over the edge with an 18.5 GB
//! model.
//!
//! This module instead wraps the mmap itself in a single no-copy Metal buffer
//! (`newBufferWithBytesNoCopy`) and binds each quantized tensor at its own
//! file offset via `setBuffer:offset:` — the same trick llama.cpp and MLX
//! use.  The GPU reads the file-backed pages directly: one copy of the
//! weights total, paged in on demand.
//!
//! Requires the vendored `candle-metal-kernels` (see `vendor/`), which adds
//! offset-aware variants of the quantized matmul entry points.  The Metal
//! shaders themselves are untouched: `setBuffer:offset:` moves the base
//! pointer the shader sees, exactly like the activation/dst offsets candle
//! already passes.
//!
//! Without the `metal` feature the types degrade to inert stubs so the
//! loaders can keep their `Option<Arc<ZcContext>>` plumbing unconditionally.

#[cfg(feature = "metal")]
mod imp {
    use candle_core::metal_backend::MetalStorage;
    use candle_core::op::BackpropOp;
    use candle_core::quantized::{gguf_file, GgmlDType};
    use candle_core::{DType, Layout, MetalDevice, Result, Shape, Storage, Tensor, D};
    use candle_metal_kernels::metal::{Buffer, MTLResourceOptions};
    use candle_metal_kernels::{
        call_quantized_get_rows_zc, call_quantized_matmul_mm_t_zc,
        call_quantized_matmul_mv_t_zc,
    };
    use objc2_metal::MTLDevice as _;
    use std::ffi::c_void;
    use std::ptr::NonNull;
    use std::sync::Arc;

    /// System page size in bytes (16 KiB on Apple Silicon).
    pub fn page_size() -> usize {
        // SAFETY: sysconf does not touch Rust state.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page <= 0 {
            16 * 1024
        } else {
            page as usize
        }
    }

/// No-copy Metal buffers backing a model mapping, in chunks.
///
/// One instance per model load.  Keeps the `Arc<Mmap>` alive for as long as
/// any weight built from it exists, so the pointer handed to Metal never
/// dangles.
///
/// Metal's driver refuses a single shared no-copy buffer above a device- and
/// memory-dependent cap (measured ≈13.3 GiB on a 24 GB Mac), so the mapping
/// is split into chunks of at most [`CHUNK_MAX`] bytes, each its own no-copy
/// buffer.  Chunk boundaries are placed at tensor boundaries when the tensor
/// layout is known ([`Self::new_for_tensors`]), guaranteeing no single
/// tensor — and therefore no [`ZcWeight`] — ever straddles two buffers.
pub struct ZcContext {
    device: MetalDevice,
    buffers: Vec<Arc<Buffer>>,
    /// File offset where each chunk starts; chunk `i` covers
    /// `chunk_offsets[i] .. chunk_offsets[i] + buffers[i].length()`.
    chunk_offsets: Vec<usize>,
    /// Mapping lifetime guard: must outlive every GPU command reading the
    /// buffers.
    _mmap: Arc<memmap2::Mmap>,
    len: usize,
}

/// Maximum bytes per no-copy buffer.  4 GiB leaves 3× headroom under the
/// measured ~13.3 GiB single-buffer cap while keeping the buffer count
/// single-digit for an 18.5 GB model.
const CHUNK_MAX: usize = 4 << 30;

impl ZcContext {
    /// Wrap `mmap` in shared-mode no-copy Metal buffers, chunked in fixed
    /// [`CHUNK_MAX`] slices.
    ///
    /// The mapping must start on a page boundary (guaranteed by `mmap`) and
    /// its length must be a whole number of pages — map the file with padded
    /// length (see [`crate::engine`]).  Any violation returns `Err`, letting
    /// the caller fall back to candle's copying path.
    pub fn new(device: &MetalDevice, mmap: Arc<memmap2::Mmap>) -> Result<Self> {
        let options = objc2_metal::MTLResourceOptions(
            MTLResourceOptions::StorageModeShared.0
                | MTLResourceOptions::HazardTrackingModeUntracked.0,
        );
        Self::new_chunked(device, mmap, options, None)
    }

    /// Like [`Self::new`], but chunk boundaries are aligned to tensor
    /// boundaries: each tensor's data lives entirely inside one chunk, so any
    /// [`ZcWeight`] resolves to a single buffer.  `tensor_data_offset` is the
    /// file offset where tensor data begins (GGUF `tensor_data_offset`);
    /// `ti.offset` values are relative to it.
    pub fn new_for_tensors(
        device: &MetalDevice,
        mmap: Arc<memmap2::Mmap>,
        tensors: &std::collections::HashMap<String, gguf_file::TensorInfo>,
        tensor_data_offset: u64,
    ) -> Result<Self> {
        let options = objc2_metal::MTLResourceOptions(
            MTLResourceOptions::StorageModeShared.0
                | MTLResourceOptions::HazardTrackingModeUntracked.0,
        );
        // Tensor byte ranges in file coordinates, sorted by start.
        let mut ranges: Vec<(usize, usize)> = tensors
            .values()
            .filter_map(|ti| {
                let n: usize = ti.shape.dims().iter().product();
                let bytes = n / ti.ggml_dtype.block_size().max(1) * ti.ggml_dtype.type_size();
                let start = usize::try_from(tensor_data_offset.saturating_add(ti.offset)).ok()?;
                Some((start, start.saturating_add(bytes)))
            })
            .collect();
        ranges.sort_unstable();
        // Coalesce into chunks ≤ CHUNK_MAX, starting a new chunk at a tensor
        // start whenever the running chunk would overflow.
        let mut chunks: Vec<(usize, usize)> = Vec::new();
        let mut cur_start = 0usize;
        let mut cur_end = 0usize;
        for (start, end) in ranges {
            if cur_end == 0 {
                cur_start = start;
                cur_end = end;
                continue;
            }
            if end <= cur_end {
                continue; // contained in the running chunk
            }
            if end - cur_start > CHUNK_MAX {
                chunks.push((cur_start, cur_end));
                cur_start = start;
            }
            cur_end = end;
        }
        if cur_end > 0 {
            chunks.push((cur_start, cur_end));
        }
        Self::new_chunked(device, mmap, options, Some(chunks))
    }

    fn new_chunked(
        device: &MetalDevice,
        mmap: Arc<memmap2::Mmap>,
        options: objc2_metal::MTLResourceOptions,
        chunks: Option<Vec<(usize, usize)>>,
    ) -> Result<Self> {
        let page = page_size();
        let base = mmap.as_ptr() as usize;
        if !base.is_multiple_of(page) {
            candle_core::bail!(
                "zero-copy Metal: mmap base {base:#x} is not {page}-byte aligned"
            );
        }
        let len = mmap.len();
        if !len.is_multiple_of(page) {
            candle_core::bail!(
                "zero-copy Metal: mmap length {len} is not a multiple of the \
                 {page}-byte page size; map the model file with padded length"
            );
        }
        // Chunk ranges: caller tensor-aligned ranges, else fixed CHUNK_MAX
        // slices.  Ends are clamped to the mapping; the final chunk always
        // runs to `len` so the whole mapping is covered.
        let ranges: Vec<(usize, usize)> = match chunks {
            Some(c) => c,
            None => (0..len)
                .step_by(CHUNK_MAX)
                .map(|s| (s, s.saturating_add(CHUNK_MAX).min(len)))
                .collect(),
        };
        let mut buffers = Vec::with_capacity(ranges.len());
        let mut chunk_offsets = Vec::with_capacity(ranges.len());
        for (start, mut end) in ranges {
            // Round the chunk down/up to page boundaries (tensor starts are
            // only 32-byte aligned).  Chunks must be page-aligned, page-sized
            // slices of the mapping for newBufferWithBytesNoCopy.
            let lo = start / page * page;
            end = end.min(len);
            end = (end + page - 1) / page * page;
            if end <= lo {
                continue;
            }
            // SAFETY: `base+lo .. base+end` is inside the mapping, held alive
            // by `_mmap`; the deallocator is None because we never free it.
            let raw = unsafe {
                device
                    .metal_device()
                    .as_ref()
                    .newBufferWithBytesNoCopy_length_options_deallocator(
                        NonNull::new((base + lo) as *mut c_void).expect("mmap base is never null"),
                        end - lo,
                        options,
                        None,
                    )
            }
            .ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "zero-copy Metal: newBufferWithBytesNoCopy failed for a {}-byte chunk \
                     at file offset {lo}",
                    end - lo
                ))
            })?;
            buffers.push(Arc::new(Buffer::new(raw)));
            chunk_offsets.push(lo);
        }
        if buffers.is_empty() {
            candle_core::bail!("zero-copy Metal: empty mapping has no chunks");
        }
        debug_assert_eq!(buffers.len(), chunk_offsets.len());
        Ok(Self {
            device: device.clone(),
            buffers,
            chunk_offsets,
            _mmap: mmap,
            len,
        })
    }

    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    /// The no-copy buffer for chunk `i`.
    pub fn buffer(&self, chunk: usize) -> &Buffer {
        &self.buffers[chunk]
    }

    pub fn num_chunks(&self) -> usize {
        self.buffers.len()
    }

    /// Total mapped (and therefore bufferable) length, in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// A zero-copy mapping is never empty; present for API symmetry.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Resolve a `[offset, offset+bytes)` file range to a chunk index and the
    /// offset of `offset` within that chunk's buffer.  Errors when the range
    /// straddles a chunk boundary.
    pub(crate) fn locate(&self, offset: usize, bytes: usize) -> Result<(usize, usize)> {
        if offset.saturating_add(bytes) > self.len {
            candle_core::bail!(
                "zero-copy Metal: range {offset}..{} exceeds mapping length {}",
                offset.saturating_add(bytes),
                self.len
            );
        }
        for (i, &start) in self.chunk_offsets.iter().enumerate() {
            let end = start + self.buffers[i].length();
            if offset >= start && offset.saturating_add(bytes) <= end {
                return Ok((i, offset - start));
            }
        }
        candle_core::bail!(
            "zero-copy Metal: range {offset}..{} straddles a chunk boundary \
             ({} chunks)",
            offset.saturating_add(bytes),
            self.chunk_offsets.len()
        );
    }

    /// Build a zero-copy weight for `name`, or `None` when the tensor is not
    /// borrowable this way (missing, non-quantized dtype, or a shape/range
    /// this path cannot handle) — the caller then falls back to candle.
    pub fn weight(self: &Arc<Self>, content: &gguf_file::Content, name: &str) -> Result<Option<ZcWeight>> {
        let Some(ti) = content.tensor_infos.get(name) else {
            return Ok(None);
        };
        // F32/F16/BF16 weights are tiny in practice (routers, norms) and
        // candle already handles them as plain tensors; zero-copying them
        // would only add paths to maintain.
        if matches!(
            ti.ggml_dtype,
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
        ) {
            return Ok(None);
        }
        // The Metal kernels have no Q8_1/Q8K matrix-multiply variants.
        if matches!(ti.ggml_dtype, GgmlDType::Q8_1 | GgmlDType::Q8K) {
            return Ok(None);
        }
        ZcWeight::new(self, ti, content.tensor_data_offset).map(Some)
    }
}

/// A quantized weight living at `offset` inside a [`ZcContext`] chunk.
///
/// Drop-in for candle's `QMatMul` on the shapes the qwen3moe loader uses
/// (2-D weights, f32 activations on Metal).  `forward` mirrors candle's
/// `QMetalStorage::fwd`/`fwd_mv` dispatch exactly: single-row inputs take
/// the `matmul_mv` path, wider inputs the bucketed `matmul_mm` path.
#[derive(Clone)]
pub struct ZcWeight {
    ctx: Arc<ZcContext>,
    dtype: GgmlDType,
    /// Candle-order dims `[n, k]`: output width then contraction width.
    dims: [usize; 2],
    /// Which chunk buffer the weight lives in.
    chunk: usize,
    /// Byte offset of the tensor data inside that chunk's buffer.
    offset: usize,
}

impl ZcWeight {
    fn new(
        ctx: &Arc<ZcContext>,
        ti: &gguf_file::TensorInfo,
        tensor_data_offset: u64,
    ) -> Result<Self> {
        let dims = ti.shape.dims();
        if dims.len() != 2 {
            candle_core::bail!(
                "zero-copy Metal: weight has rank {} (expected 2)",
                dims.len(),
            );
        }
        Self::from_parts(ctx, ti, tensor_data_offset, [dims[0], dims[1]], 0)
    }

    /// Slice one expert out of a `[n_expert, out, in]` expert tensor.
    ///
    /// `byte_offset` is the expert's byte offset **within** the tensor data
    /// (expert `e` of `per_bytes`-sized rows).  The resulting weight has the
    /// expert's own `[out, in]` dims.
    pub fn expert(
        ctx: &Arc<ZcContext>,
        ti: &gguf_file::TensorInfo,
        tensor_data_offset: u64,
        dims: [usize; 2],
        byte_offset: usize,
    ) -> Result<Self> {
        Self::from_parts(ctx, ti, tensor_data_offset, dims, byte_offset)
    }

    fn from_parts(
        ctx: &Arc<ZcContext>,
        ti: &gguf_file::TensorInfo,
        tensor_data_offset: u64,
        dims: [usize; 2],
        byte_offset: usize,
    ) -> Result<Self> {
        let offset = usize::try_from(tensor_data_offset.saturating_add(ti.offset))
            .map_err(|_| candle_core::Error::Msg("zero-copy Metal: tensor offset overflow".into()))?
            .saturating_add(byte_offset);
        let bytes = dims[0] * dims[1] * ti.ggml_dtype.type_size() / ti.ggml_dtype.block_size();
        let (chunk, offset_in_chunk) = ctx.locate(offset, bytes)?;
        Ok(Self {
            ctx: ctx.clone(),
            dtype: ti.ggml_dtype,
            dims,
            chunk,
            offset: offset_in_chunk,
        })
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    /// `xs @ W^T` with `W` the `[n, k]` weight, activations f32 on Metal.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (storage_guard, layout) = xs.storage_and_layout();
        let storage = match &*storage_guard {
            Storage::Metal(s) => s,
            _ => candle_core::bail!("zero-copy Metal: activations must be Metal tensors"),
        };
        if xs.dtype() != DType::F32 {
            candle_core::bail!(
                "zero-copy Metal: activations must be f32, got {:?}",
                xs.dtype()
            );
        }
        if !layout.is_contiguous() {
            candle_core::bail!("zero-copy Metal: input tensor is not contiguous {layout:?}");
        }
        let src_shape = layout.shape();
        if src_shape.rank() < 2 {
            candle_core::bail!("zero-copy Metal: input tensor has only one dimension {layout:?}");
        }
        let [n, k] = self.dims;
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().expect("rank >= 2 checked above");
        if last_k != k {
            candle_core::bail!(
                "zero-copy Metal: input {layout:?} incompatible with weight [{n}, {k}]"
            );
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        if src_shape.dim(D::Minus2)? == 1 {
            self.forward_mv(storage, layout, n, k, &dst_shape)
        } else {
            self.forward_mm(storage, layout, n, k, &dst_shape)
        }
    }

    /// Decode path (single-row activations, possibly batched): one `matmul_mv`
    /// dispatch per row, exactly like candle's `fwd_mv`.
    fn forward_mv(
        &self,
        storage: &MetalStorage,
        layout: &Layout,
        n: usize,
        k: usize,
        dst_shape: &Shape,
    ) -> Result<Tensor> {
        let src_shape = layout.shape();
        let m = match src_shape.dims() {
            &[b, s, _] => b * s,
            &[s, _] => s,
            d => candle_core::bail!(
                "zero-copy Metal: unsupported input rank {} for decode ({d:?})",
                d.len()
            ),
        };
        let device = self.ctx.device();
        let dst = device
            .new_buffer_builder()
            .with_size_for(dst_shape.elem_count(), DType::F32)
            .with_label("zc_qmatmul_mv")
            .build()?;
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            call_quantized_matmul_mv_t_zc(
                device.metal_device(),
                &encoder,
                device.kernels(),
                self.dtype.into(),
                (1, 1, n, k),
                storage.buffer(),
                (layout.start_offset() + batch_id * k) * DType::F32.size_in_bytes(),
                self.ctx.buffer(self.chunk),
                self.offset,
                batch_id * n * DType::F32.size_in_bytes(),
                &dst,
            )
            .map_err(|e| candle_core::Error::Msg(format!("zero-copy metal kernel (mv): {e}")))?;
        }
        let dst_storage =
            MetalStorage::new(dst, device.clone(), dst_shape.elem_count(), DType::F32);
        Ok(Tensor::from_storage(
            Storage::Metal(dst_storage),
            dst_shape.clone(),
            BackpropOp::none(),
            false,
        ))
    }

    /// Prefill path (multi-row activations): the bucketed `matmul_mm` kernel,
    /// mirroring candle's `fwd`.
    fn forward_mm(
        &self,
        storage: &MetalStorage,
        layout: &Layout,
        n: usize,
        k: usize,
        dst_shape: &Shape,
    ) -> Result<Tensor> {
        let src_shape = layout.shape();
        if src_shape.rank() > 4 {
            candle_core::bail!(
                "zero-copy Metal: input rank {} must be <= 4",
                src_shape.rank()
            );
        }
        // Weights are 2-D, so pad to rank 4 like candle does.
        let src0_l = Layout::contiguous(vec![1, 1, n, k]);
        let block = self.dtype.type_size() as f32 / self.dtype.block_size() as f32;
        let src0_stride = src0_l
            .stride()
            .iter()
            .map(|x| (*x as f32 * block) as usize)
            .collect::<Vec<_>>();
        let src1_l = Layout::contiguous(
            [vec![1; 4 - src_shape.rank()], src_shape.dims().to_vec()].concat(),
        );
        let device = self.ctx.device();
        let dst = device
            .new_buffer_builder()
            .with_size_for(dst_shape.elem_count(), DType::F32)
            .with_label("zc_qmatmul_mm")
            .build()?;
        let encoder = device.command_encoder()?;
        call_quantized_matmul_mm_t_zc(
            device.metal_device(),
            &encoder,
            device.kernels(),
            self.dtype.into(),
            src0_l.dims(),
            &src0_stride,
            self.ctx.buffer(self.chunk),
            self.offset,
            src1_l.dims(),
            &src1_l
                .stride()
                .iter()
                .map(|x| x * DType::F32.size_in_bytes())
                .collect::<Vec<_>>(),
            storage.buffer(),
            src1_l.start_offset() * DType::F32.size_in_bytes(),
            dst_shape.dims(),
            0,
            &dst,
        )
        .map_err(|e| candle_core::Error::Msg(format!("zero-copy metal kernel (mm): {e}")))?;
        let dst_storage =
            MetalStorage::new(dst, device.clone(), dst_shape.elem_count(), DType::F32);
        Ok(Tensor::from_storage(
            Storage::Metal(dst_storage),
            dst_shape.clone(),
            BackpropOp::none(),
            false,
        ))
    }
}

/// A quantized embedding table backed by a [`ZcContext`] buffer.
///
/// The qwen3moe loader currently dequantizes `token_embd` to an f32 tensor
/// (it is small), so nothing uses this yet — kept for the architectures that
/// keep embeddings quantized.
#[allow(dead_code)]
pub struct ZcEmbedding {
    ctx: Arc<ZcContext>,
    dtype: GgmlDType,
    rows: usize,
    hidden: usize,
    chunk: usize,
    offset: usize,
    bytes: usize,
}

#[allow(dead_code)]
impl ZcEmbedding {
    fn new(ctx: &Arc<ZcContext>, ti: &gguf_file::TensorInfo, data_offset: u64) -> Result<Self> {
        let dims = ti.shape.dims();
        if dims.len() != 2 {
            candle_core::bail!("zero-copy Metal: embedding must be 2-D, got {dims:?}");
        }
        if !dims[1].is_multiple_of(ti.ggml_dtype.block_size()) {
            candle_core::bail!(
                "zero-copy Metal: embedding hidden size {} not divisible by block size {}",
                dims[1],
                ti.ggml_dtype.block_size()
            );
        }
        let offset =
            usize::try_from(data_offset.saturating_add(ti.offset)).map_err(|_| {
                candle_core::Error::Msg("zero-copy Metal: embedding offset overflow".into())
            })?;
        let bytes = dims[0] * dims[1] * ti.ggml_dtype.type_size() / ti.ggml_dtype.block_size();
        let (chunk, offset_in_chunk) = ctx.locate(offset, bytes)?;
        Ok(Self {
            ctx: ctx.clone(),
            dtype: ti.ggml_dtype,
            rows: dims[0],
            hidden: dims[1],
            chunk,
            offset: offset_in_chunk,
            bytes,
        })
    }

    /// Gather `ids` rows from the table, returning f32 `[ids_len, hidden]`.
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let (ids_guard, ids_layout) = ids.storage_and_layout();
        let ids_storage = match &*ids_guard {
            Storage::Metal(s) => s,
            _ => candle_core::bail!("zero-copy Metal: embedding ids must be Metal tensors"),
        };
        if ids.dtype() != DType::U32 {
            candle_core::bail!("zero-copy Metal: embedding ids must be u32");
        }
        if !ids_layout.is_contiguous() {
            candle_core::bail!("zero-copy Metal: embedding ids must be contiguous");
        }
        let ids_len = ids_layout.shape().elem_count();
        let row_stride = self.hidden * self.dtype.type_size() / self.dtype.block_size();
        let device = self.ctx.device();
        let dst = device
            .new_buffer_builder()
            .with_size_for(ids_len * self.hidden, DType::F32)
            .with_label("zc_qembedding")
            .build()?;
        let encoder = device.command_encoder()?;
        call_quantized_get_rows_zc(
            device.metal_device(),
            &encoder,
            device.kernels(),
            self.dtype.into(),
            self.hidden,
            row_stride,
            ids_len,
            self.ctx.buffer(self.chunk),
            self.offset,
            ids_storage.buffer(),
            ids_layout.start_offset() * DType::U32.size_in_bytes(),
            &dst,
        )
        .map_err(|e| candle_core::Error::Msg(format!("zero-copy metal kernel (get_rows): {e}")))?;
        let dst_storage =
            MetalStorage::new(dst, device.clone(), ids_len * self.hidden, DType::F32);
        Ok(Tensor::from_storage(
            Storage::Metal(dst_storage),
            Shape::from_dims(&[ids_len, self.hidden]),
            BackpropOp::none(),
            false,
        ))
    }
}

    #[cfg(test)]
    mod tests {
        use super::*;
        use candle_core::Device;

        #[test]
        fn page_size_is_sane() {
            let page = page_size();
            assert!(page.is_power_of_two() && page >= 4096, "page size {page}");
        }

        #[test]
        fn zc_requires_page_aligned_mapping() {
            let dev = match Device::new_metal(0) {
                Ok(d) => d,
                Err(_) => return, // skip without Metal
            };
            let Device::Metal(md) = &dev else {
                unreachable!()
            };
            // A deliberately tiny, non-page-multiple mapping must be refused.
            let tmp =
                std::env::temp_dir().join(format!("joshua-zc-{}", std::process::id()));
            std::fs::write(&tmp, vec![0u8; 4097]).unwrap();
            let file = std::fs::File::open(&tmp).unwrap();
            let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
            assert!(ZcContext::new(md, Arc::new(mmap)).is_err());
            std::fs::remove_file(&tmp).ok();
        }
    }
}

#[cfg(not(feature = "metal"))]
mod imp {
    //! Inert stand-ins so loaders can hold `Option<Arc<ZcContext>>` and call
    //! `ZcContext::new` without cfg noise.  The `metal` build replaces this
    //! whole module.

    use candle_core::quantized::gguf_file;
    use std::sync::Arc;

    pub struct ZcContext;

    impl ZcContext {
        pub fn new(
            _device: &candle_core::MetalDevice,
            _mmap: Arc<memmap2::Mmap>,
        ) -> candle_core::Result<Self> {
            candle_core::bail!("zero-copy Metal requires the `metal` feature")
        }

        pub fn new_for_tensors(
            _device: &candle_core::MetalDevice,
            _mmap: Arc<memmap2::Mmap>,
            _tensors: &std::collections::HashMap<String, gguf_file::TensorInfo>,
            _tensor_data_offset: u64,
        ) -> candle_core::Result<Self> {
            candle_core::bail!("zero-copy Metal requires the `metal` feature")
        }

        pub fn num_chunks(&self) -> usize {
            0
        }

        pub fn len(&self) -> usize {
            0
        }

        pub fn is_empty(&self) -> bool {
            true
        }

        pub fn weight(
            &self,
            _content: &gguf_file::Content,
            _name: &str,
        ) -> candle_core::Result<Option<ZcWeight>> {
            Ok(None)
        }
    }

    pub struct ZcWeight;

    impl ZcWeight {
        pub fn expert(
            _ctx: &Arc<ZcContext>,
            _ti: &gguf_file::TensorInfo,
            _tensor_data_offset: u64,
            _dims: [usize; 2],
            _byte_offset: usize,
        ) -> candle_core::Result<Self> {
            candle_core::bail!("zero-copy Metal requires the `metal` feature")
        }

        pub fn forward(&self, _xs: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
            candle_core::bail!("zero-copy Metal requires the `metal` feature")
        }
    }

    pub fn page_size() -> usize {
        16 * 1024
    }
}

pub use imp::{page_size, ZcContext, ZcWeight};
