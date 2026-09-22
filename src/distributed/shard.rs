//! Input-column shards of a row-major matrix (`RawTensorInfo.dims = [output, input]`).
//! The GGUF file stores `[input, output]`; `read_header` reverses those dimensions.
//!
//! Every output row contributes a separate byte range. A shard never owns a
//! contiguous fraction of the entire tensor. The full local file is mapped
//! lazily; only these row ranges are read or explicitly prefetched.
//!
//! As with `mmap_tensor`, mapped files must not be modified or truncated while
//! any mapping is alive. The owning `Arc<Mmap>` keeps storage alive, but cannot
//! protect against another process changing the underlying file.

use std::{fs::File, ops::Range, sync::Arc};

use anyhow::{ensure, Context, Result};
use candle_core::quantized::{k_quants::*, GgmlType};
use memmap2::{Mmap, MmapOptions};

use crate::gguf_ext::RawTensorInfo;

/// On-disk storage for one GGUF quantization block (one scalar for floats).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DtypeLayout {
    pub block_elements: usize,
    pub block_bytes: usize,
}

/// Layouts of every GGUF weight dtype supported by Joshua's CPU decoders.
/// Unknown IDs are rejected rather than guessed.
pub fn dtype_layout(dtype: u32) -> Result<DtypeLayout> {
    macro_rules! layout {
        ($ty:ty) => {
            DtypeLayout {
                block_elements: <$ty as GgmlType>::BLCK_SIZE,
                block_bytes: std::mem::size_of::<$ty>(),
            }
        };
    }
    Ok(match dtype {
        0 => layout!(f32),
        1 => layout!(half::f16),
        30 => layout!(half::bf16),
        2 => layout!(BlockQ4_0),
        3 => layout!(BlockQ4_1),
        6 => layout!(BlockQ5_0),
        7 => layout!(BlockQ5_1),
        8 => layout!(BlockQ8_0),
        9 => layout!(BlockQ8_1),
        10 => layout!(BlockQ2K),
        11 => layout!(BlockQ3K),
        12 => layout!(BlockQ4K),
        13 => layout!(BlockQ5K),
        14 => layout!(BlockQ6K),
        15 => layout!(BlockQ8K),
        16 => DtypeLayout {
            block_elements: crate::iq2xxs::QK_IQ2_XXS,
            block_bytes: crate::iq2xxs::BLOCK_BYTES,
        },
        39 => DtypeLayout {
            block_elements: crate::mxfp4::QK_MXFP4,
            block_bytes: std::mem::size_of::<crate::mxfp4::BlockMxfp4>(),
        },
        _ => anyhow::bail!("unsupported GGUF shard dtype {dtype}"),
    })
}

/// A nonempty rank in a block-balanced input-column partition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShardSpec {
    rank: usize,
    count: usize,
}

impl ShardSpec {
    pub fn new(rank: usize, count: usize) -> Result<Self> {
        ensure!(count > 0, "shard count must be positive");
        ensure!(rank < count, "shard rank {rank} is outside count {count}");
        Ok(Self { rank, count })
    }

    pub fn rank(self) -> usize {
        self.rank
    }

    pub fn count(self) -> usize {
        self.count
    }

    /// Partition whole blocks; the first `blocks % count` ranks get one extra.
    pub fn input_range(self, input_width: usize, block_elements: usize) -> Result<Range<usize>> {
        ensure!(block_elements > 0, "block size must be positive");
        ensure!(
            input_width > 0 && input_width.is_multiple_of(block_elements),
            "input width must be positive and block-aligned"
        );
        let blocks = input_width / block_elements;
        ensure!(self.count <= blocks, "more shards than input blocks");
        let base = blocks / self.count;
        let remainder = blocks % self.count;
        let first = self
            .rank
            .checked_mul(base)
            .and_then(|n| n.checked_add(self.rank.min(remainder)))
            .context("shard block offset overflow")?;
        let end = first
            .checked_add(base)
            .and_then(|n| n.checked_add(usize::from(self.rank < remainder)))
            .context("shard block end overflow")?;
        Ok(first
            .checked_mul(block_elements)
            .context("shard input offset overflow")?
            ..end
                .checked_mul(block_elements)
                .context("shard input end overflow")?)
    }
}

/// Checked metadata and an owning view of a local file's input-column shard.
///
/// Construction validates metadata without reading tensor bytes. Execution
/// decodes at most one quantization block at a time, never the full weights.
pub struct ShardedTensor {
    mmap: Arc<Mmap>,
    dtype: u32,
    layout: DtypeLayout,
    input_width: usize,
    output_width: usize,
    input_range: Range<usize>,
    first_byte: usize,
    row_bytes: usize,
    local_row_bytes: usize,
}

impl ShardedTensor {
    pub fn new(
        mmap: Arc<Mmap>,
        tensor: &RawTensorInfo,
        data_offset: u64,
        rank: usize,
        count: usize,
    ) -> Result<Self> {
        let spec = ShardSpec::new(rank, count)?;
        let layout = dtype_layout(tensor.dtype)?;
        let input_width = *tensor
            .dims
            .get(1)
            .context("tensor requires two dimensions")?;
        let range = spec.input_range(input_width, layout.block_elements)?;
        Self::with_input_range(mmap, tensor, data_offset, range)
    }

    /// Explicit block-aligned columns for heterogeneous device scheduling.
    /// Callers are responsible for making multiple explicit ranges a partition.
    pub fn with_input_range(
        mmap: Arc<Mmap>,
        tensor: &RawTensorInfo,
        data_offset: u64,
        input_range: Range<usize>,
    ) -> Result<Self> {
        ensure!(tensor.dims.len() == 2, "shards require a 2D GGUF matrix");
        let (output_width, input_width) = (tensor.dims[0], tensor.dims[1]);
        let layout = dtype_layout(tensor.dtype)?;
        ensure!(input_width > 0 && output_width > 0, "empty tensor");
        ensure!(
            input_width.is_multiple_of(layout.block_elements),
            "tensor input width is not block-aligned"
        );
        ensure!(
            input_range.start < input_range.end && input_range.end <= input_width,
            "shard input range is empty, reversed, or outside the tensor"
        );
        ensure!(
            input_range.start.is_multiple_of(layout.block_elements)
                && input_range.end.is_multiple_of(layout.block_elements),
            "shard input range is not block-aligned"
        );
        input_width
            .checked_mul(output_width)
            .context("tensor element count overflow")?;
        let row_bytes = (input_width / layout.block_elements)
            .checked_mul(layout.block_bytes)
            .context("tensor row byte count overflow")?;
        let tensor_bytes = row_bytes
            .checked_mul(output_width)
            .context("tensor byte count overflow")?;
        let tensor_start = usize::try_from(
            data_offset
                .checked_add(tensor.offset)
                .context("tensor file offset overflow")?,
        )
        .context("tensor file offset exceeds address space")?;
        let tensor_end = tensor_start
            .checked_add(tensor_bytes)
            .context("tensor file end overflow")?;
        ensure!(
            tensor_end <= mmap.len(),
            "tensor exceeds mapping (truncated file)"
        );
        let local_offset = (input_range.start / layout.block_elements)
            .checked_mul(layout.block_bytes)
            .context("local byte offset overflow")?;
        let first_byte = tensor_start
            .checked_add(local_offset)
            .context("local file offset overflow")?;
        let local_row_bytes = ((input_range.end - input_range.start) / layout.block_elements)
            .checked_mul(layout.block_bytes)
            .context("local row byte count overflow")?;
        Ok(Self {
            mmap,
            dtype: tensor.dtype,
            layout,
            input_width,
            output_width,
            input_range,
            first_byte,
            row_bytes,
            local_row_bytes,
        })
    }

    /// Lazily map a local file from offset zero, without populate or prefetch.
    ///
    /// The file must remain immutable while this object or its mapping lives,
    /// matching the read-only mapping convention of `mmap_tensor`.
    pub fn map_file(
        file: &File,
        tensor: &RawTensorInfo,
        data_offset: u64,
        rank: usize,
        count: usize,
    ) -> Result<Self> {
        // SAFETY: the caller observes the immutable-file contract documented
        // above. No writable mapping or borrowed block pointer is exposed.
        let mmap = unsafe { MmapOptions::new().map(file) }.context("mapping local shard file")?;
        Self::new(Arc::new(mmap), tensor, data_offset, rank, count)
    }

    pub fn dtype(&self) -> u32 {
        self.dtype
    }

    pub fn layout(&self) -> DtypeLayout {
        self.layout
    }

    pub fn input_width(&self) -> usize {
        self.input_width
    }

    pub fn output_width(&self) -> usize {
        self.output_width
    }

    pub fn input_range(&self) -> Range<usize> {
        self.input_range.clone()
    }

    /// Absolute file byte ranges, one per output row, in output order.
    pub fn row_ranges(&self) -> impl ExactSizeIterator<Item = Range<usize>> + '_ {
        (0..self.output_width).map(|row| {
            // Construction checked the entire matrix end and row stride.
            let start = self.first_byte + row * self.row_bytes;
            start..start + self.local_row_bytes
        })
    }

    /// Advise only the local row slices, never the full file or bounding span.
    /// OS advice is page-granular, so boundary pages can contain other columns.
    /// A no-op on platforms without Unix mapping advice.
    pub fn prefetch(&self) -> Result<()> {
        #[cfg(unix)]
        for range in self.row_ranges() {
            self.mmap
                .advise_range(memmap2::Advice::WillNeed, range.start, range.len())
                .context("prefetching local shard row")?;
        }
        Ok(())
    }

    /// Multiply only local columns, returning a partial full-width output.
    ///
    /// `input` is the full activation vector; values outside `input_range()`
    /// are never accessed. Sum peer outputs to recover the full matvec.
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        ensure!(input.len() == self.input_width, "incorrect input width");
        self.forward_local(&input[self.input_range.clone()])
    }

    /// Multiply a compact activation vector containing only this shard's columns.
    pub fn forward_local(&self, input: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            input.len() == self.input_range.len(),
            "incorrect local input width"
        );
        let mut output = Vec::new();
        output
            .try_reserve_exact(self.output_width)
            .context("allocating shard output")?;
        let mut decoded = [0.0; 256];
        for range in self.row_ranges() {
            let mut sum = 0.0;
            for (block, activations) in self.mmap[range]
                .chunks_exact(self.layout.block_bytes)
                .zip(input.chunks_exact(self.layout.block_elements))
            {
                let weights = &mut decoded[..self.layout.block_elements];
                decode_block(self.dtype, block, weights)?;
                for (&weight, &activation) in weights.iter().zip(activations) {
                    sum += weight * activation;
                }
            }
            output.push(sum);
        }
        Ok(output)
    }
}

fn decode_block(dtype: u32, bytes: &[u8], out: &mut [f32]) -> Result<()> {
    macro_rules! decode {
        ($ty:ty) => {{
            ensure!(
                bytes.len() == std::mem::size_of::<$ty>(),
                "invalid block bytes"
            );
            // SAFETY: these concrete candle blocks contain only integer/float
            // fields, so all bit patterns are valid. The size was checked.
            // read_unaligned copies into an aligned local value, never forms
            // an unaligned reference to mmap storage.
            let block = unsafe { std::ptr::read_unaligned(bytes.as_ptr().cast::<$ty>()) };
            <$ty as GgmlType>::to_float(std::slice::from_ref(&block), out);
        }};
    }
    match dtype {
        0 => out[0] = f32::from_le_bytes(bytes.try_into()?),
        1 => out[0] = half::f16::from_le_bytes(bytes.try_into()?).to_f32(),
        30 => out[0] = half::bf16::from_le_bytes(bytes.try_into()?).to_f32(),
        // Candle's quantized block representation follows GGUF little endian.
        2 | 3 | 6..=15 => {
            ensure!(
                cfg!(target_endian = "little"),
                "quantized shards require little endian"
            );
            match dtype {
                2 => decode!(BlockQ4_0),
                3 => decode!(BlockQ4_1),
                6 => decode!(BlockQ5_0),
                7 => decode!(BlockQ5_1),
                8 => decode!(BlockQ8_0),
                9 => decode!(BlockQ8_1),
                10 => decode!(BlockQ2K),
                11 => decode!(BlockQ3K),
                12 => decode!(BlockQ4K),
                13 => decode!(BlockQ5K),
                14 => decode!(BlockQ6K),
                15 => decode!(BlockQ8K),
                _ => unreachable!(),
            }
        }
        16 => crate::iq2xxs::dequantize(crate::iq2xxs::blocks_from_bytes(bytes)?, out)?,
        39 => crate::mxfp4::dequantize(crate::mxfp4::blocks_from_bytes(bytes)?, out)?,
        _ => anyhow::bail!("unsupported GGUF shard dtype {dtype}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use memmap2::MmapMut;

    fn mapping(bytes: &[u8]) -> Arc<Mmap> {
        let mut mmap = MmapMut::map_anon(bytes.len()).unwrap();
        mmap.copy_from_slice(bytes);
        Arc::new(mmap.make_read_only().unwrap())
    }

    fn fixture(dtype: u32, blocks: usize, rows: usize) -> (Arc<Mmap>, RawTensorInfo, Vec<f32>) {
        let layout = dtype_layout(dtype).unwrap();
        // Odd absolute offset exercises unaligned quantized block handling.
        let mut bytes = vec![0u8; 3 + rows * blocks * layout.block_bytes];
        let width = blocks * layout.block_elements;
        let mut weights = Vec::new();
        for (i, block) in bytes[3..].chunks_exact_mut(layout.block_bytes).enumerate() {
            let scale = half::f16::from_f32(0.125 * (i % 5 + 1) as f32).to_le_bytes();
            match dtype {
                8 => {
                    block[..2].copy_from_slice(&scale);
                    for (j, value) in block[2..].iter_mut().enumerate() {
                        *value = ((i + j) % 17) as u8;
                    }
                }
                10 => {
                    block[..16].fill(0x21);
                    block[16..80].fill((i * 13 + 3) as u8);
                    block[80..82].copy_from_slice(&scale);
                    block[82..84].copy_from_slice(&half::f16::from_f32(0.0625).to_le_bytes());
                }
                12 => {
                    block[..2].copy_from_slice(&scale);
                    block[2..4].copy_from_slice(&half::f16::from_f32(0.0625).to_le_bytes());
                    block[4..16].fill(1);
                    block[16..].fill((i * 11 + 7) as u8);
                }
                16 => {
                    block[..2].copy_from_slice(&scale);
                    for (j, value) in block[2..].iter_mut().enumerate() {
                        *value = (i * 13 + j * 7) as u8;
                    }
                }
                39 => {
                    block[0] = 124 + (i % 5) as u8;
                    for (j, value) in block[1..].iter_mut().enumerate() {
                        *value = (i * 11 + j * 3) as u8;
                    }
                }
                _ => panic!("unsupported fixture dtype"),
            }
            let mut decoded = vec![0.0; layout.block_elements];
            decode_block(dtype, block, &mut decoded).unwrap();
            weights.extend(decoded);
        }
        (
            mapping(&bytes),
            RawTensorInfo {
                dtype,
                dims: vec![rows, width],
                offset: 1,
            },
            weights,
        )
    }

    #[test]
    fn five_quant_types_match_reference_and_cover_each_row_once() {
        for dtype in [8, 12, 10, 16, 39] {
            let (mmap, tensor, weights) = fixture(dtype, 7, 3);
            let width = tensor.dims[1];
            let input: Vec<_> = (0..width).map(|i| (i % 19) as f32 / 13.0 - 0.5).collect();
            let expected: Vec<f32> = weights
                .chunks_exact(width)
                .map(|row| row.iter().zip(&input).map(|(a, b)| a * b).sum())
                .collect();
            let full = ShardedTensor::new(mmap.clone(), &tensor, 2, 0, 1).unwrap();
            let full_ranges: Vec<_> = full.row_ranges().collect();
            let mut sum = vec![0.0; 3];
            let mut ranges = vec![Vec::new(); 3];
            for rank in 0..3 {
                let shard = ShardedTensor::new(mmap.clone(), &tensor, 2, rank, 3).unwrap();
                assert_eq!(
                    shard.input_range().len() / shard.layout().block_elements,
                    if rank == 0 { 3 } else { 2 }
                );
                for (row, range) in shard.row_ranges().enumerate() {
                    ranges[row].push(range);
                }
                // Poison non-local activations: they must never contribute.
                let mut local_input = vec![f32::NAN; width];
                let local = shard.input_range();
                local_input[local.clone()].copy_from_slice(&input[local]);
                assert_eq!(
                    shard.forward(&local_input).unwrap(),
                    shard.forward_local(&input[shard.input_range()]).unwrap()
                );
                assert!(shard.forward_local(&[]).is_err());
                assert!(shard.forward_local(&input).is_err());
                for (total, value) in sum.iter_mut().zip(shard.forward(&local_input).unwrap()) {
                    *total += value;
                }
                shard.prefetch().unwrap();
            }
            for (actual, expected) in sum.into_iter().zip(expected) {
                assert!(actual.is_finite());
                assert!(
                    (actual - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                    "dtype {dtype}: {actual} != {expected}"
                );
            }
            for (row_ranges, full) in ranges.into_iter().zip(full_ranges) {
                assert_eq!(row_ranges[0].start, full.start);
                assert_eq!(row_ranges[2].end, full.end);
                assert!(row_ranges.windows(2).all(|w| w[0].end == w[1].start));
            }
        }
    }

    #[test]
    fn actual_gguf_header_reverses_dimensions_before_sharding() {
        use std::io::Cursor;
        for dtype in [8, 12, 10, 16, 39] {
            let (source, tensor, weights) = fixture(dtype, 8, 4);
            let width = tensor.dims[1];
            let mut bytes = b"GGUF".to_vec();
            bytes.extend_from_slice(&3u32.to_le_bytes());
            bytes.extend_from_slice(&1u64.to_le_bytes());
            bytes.extend_from_slice(&0u64.to_le_bytes());
            bytes.extend_from_slice(&1u64.to_le_bytes());
            bytes.push(b'w');
            bytes.extend_from_slice(&2u32.to_le_bytes());
            bytes.extend_from_slice(&(width as u64).to_le_bytes());
            bytes.extend_from_slice(&4u64.to_le_bytes());
            bytes.extend_from_slice(&dtype.to_le_bytes());
            bytes.extend_from_slice(&0u64.to_le_bytes());
            bytes.resize(bytes.len().div_ceil(32) * 32, 0);
            let data_offset = bytes.len() as u64;
            bytes.extend_from_slice(&source[3..]);
            let header = crate::gguf_ext::read_header(&mut Cursor::new(&bytes)).unwrap();
            assert_eq!(header.tensor_data_offset, data_offset);
            let parsed = &header.tensors["w"];
            assert_eq!(parsed.dims, vec![4, width]);
            let mmap = mapping(&bytes);
            let input = vec![0.25; width];
            let mut actual = vec![0.0f32; 4];
            for rank in 0..3 {
                let shard = ShardedTensor::new(mmap.clone(), parsed, data_offset, rank, 3).unwrap();
                assert_eq!(shard.input_width(), width);
                assert_eq!(shard.output_width(), 4);
                for (sum, partial) in actual.iter_mut().zip(shard.forward(&input).unwrap()) {
                    *sum += partial;
                }
            }
            for (actual, row) in actual.iter().zip(weights.chunks_exact(width)) {
                let expected: f32 = row.iter().map(|w| w * 0.25).sum();
                assert!((actual - expected).abs() <= 1e-4 * expected.abs().max(1.0));
            }
        }
    }

    #[test]
    fn all_supported_layouts_and_decoders() {
        for (dtype, elements, bytes) in [
            (0, 1, 4),
            (1, 1, 2),
            (30, 1, 2),
            (2, 32, 18),
            (3, 32, 20),
            (6, 32, 22),
            (7, 32, 24),
            (8, 32, 34),
            (9, 32, 36),
            (10, 256, 84),
            (11, 256, 110),
            (12, 256, 144),
            (13, 256, 176),
            (14, 256, 210),
            (15, 256, 292),
            (16, 256, 66),
            (39, 32, 17),
        ] {
            let layout = dtype_layout(dtype).unwrap();
            assert_eq!(
                layout,
                DtypeLayout {
                    block_elements: elements,
                    block_bytes: bytes
                }
            );
            let tensor = RawTensorInfo {
                dtype,
                dims: vec![2, elements * 3],
                offset: 1,
            };
            let shard =
                ShardedTensor::new(mapping(&vec![0; 1 + bytes * 6]), &tensor, 0, 1, 2).unwrap();
            assert_eq!(
                shard.row_ranges().collect::<Vec<_>>(),
                vec![1 + 2 * bytes..1 + 3 * bytes, 1 + 5 * bytes..1 + 6 * bytes,]
            );
            assert!(shard
                .forward(&vec![1.0; elements * 3])
                .unwrap()
                .iter()
                .all(|v| v.is_finite()));
        }
        assert!(dtype_layout(999).is_err());
    }

    #[test]
    fn invalid_specs_ranges_metadata_and_truncation() {
        assert!(ShardSpec::new(0, 0).is_err());
        assert!(ShardSpec::new(2, 2).is_err());
        let spec = ShardSpec::new(0, 2).unwrap();
        assert!(spec.input_range(32, 32).is_err());
        assert!(spec.input_range(0, 32).is_err());
        assert!(spec.input_range(33, 32).is_err());
        assert!(spec.input_range(32, 0).is_err());
        assert_eq!(
            ShardSpec::new(1, 2)
                .unwrap()
                .input_range(usize::MAX, 1)
                .unwrap()
                .end,
            usize::MAX
        );

        let (mmap, tensor, _) = fixture(8, 3, 2);
        for range in [0..0, 32..0, 1..32, 0..33, 0..128] {
            assert!(ShardedTensor::with_input_range(mmap.clone(), &tensor, 2, range).is_err());
        }
        let shard = ShardedTensor::with_input_range(mmap.clone(), &tensor, 2, 32..96).unwrap();
        assert_eq!(shard.input_range(), 32..96);
        assert!(shard.forward(&[0.0]).is_err());
        assert!(ShardedTensor::new(mapping(&mmap[..mmap.len() - 1]), &tensor, 2, 0, 3).is_err());
        assert!(ShardedTensor::new(mmap.clone(), &tensor, u64::MAX, 0, 1).is_err());
        for dims in [
            vec![],
            vec![96],
            vec![2, 96, 1],
            vec![2, 0],
            vec![0, 96],
            vec![2, 95],
        ] {
            let mut bad = tensor.clone();
            bad.dims = dims;
            assert!(ShardedTensor::new(mmap.clone(), &bad, 2, 0, 1).is_err());
        }
        for dims in [vec![2, usize::MAX], vec![1, usize::MAX / 4 + 1]] {
            let bad = RawTensorInfo {
                dtype: 0,
                dims,
                offset: 0,
            };
            assert!(ShardedTensor::new(mmap.clone(), &bad, 0, 0, 1).is_err());
        }
        let bad = RawTensorInfo {
            dtype: 0,
            dims: vec![1, 1],
            offset: u64::MAX - 1,
        };
        assert!(ShardedTensor::new(mmap, &bad, 0, 0, 1).is_err());
    }

    #[test]
    fn owns_mapping_and_maps_local_file_without_retaining_file_handle() {
        use std::io::Write;
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let (mmap, tensor, _) = fixture(8, 3, 2);
        let weak = Arc::downgrade(&mmap);
        let shard = ShardedTensor::new(mmap.clone(), &tensor, 2, 0, 2).unwrap();
        drop(mmap);
        assert!(weak.upgrade().is_some());
        let expected = shard.forward(&vec![1.0; 96]).unwrap();
        let path = format!(
            "target/shard-fixture-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        struct RemoveFile(String);
        impl Drop for RemoveFile {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        let cleanup = RemoveFile(path.clone());
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all(&shard.mmap).unwrap();
        file.flush().unwrap();
        let local = ShardedTensor::map_file(&file, &tensor, 2, 0, 2).unwrap();
        drop(file);
        assert_eq!(local.forward(&vec![1.0; 96]).unwrap(), expected);
        drop(shard);
        assert!(weak.upgrade().is_none());
        drop(local);
        drop(cleanup);
    }
}
