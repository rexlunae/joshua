//! Block quantization formats candle cannot name, behind one trait.
//!
//! Candle's GGUF reader rejects any tensor dtype outside its own
//! [`GgmlDType`] table, and its CPU kernels quantize activations before the
//! dot.  Joshua instead keeps these weights as their GGUF blocks — borrowed
//! straight from the memory mapping — and decodes a block at a time inside
//! an f32-activation matmul.  Every such format (IQ2_XXS, MXFP4, and
//! Bonsai's Q1_0 / Q2_0) needs the same plumbing: reinterpreting bytes as
//! blocks, bulk decode, the matmul, its f16 wrapper, byte sizes, and
//! dispatch from a raw GGUF dtype id.  [`RawBlock`] is the per-format part
//! (layout, id, one-block decode, optionally a faster matmul); everything
//! else lives here once.
//!
//! Blocks must be byte-aligned (`align_of == 1`), so a run of them can be
//! borrowed from any offset in a mapping.

use candle_core::quantized::GgmlDType;
use candle_core::{bail, Result};
use half::f16;

/// Largest block any [`RawBlock`] format uses (IQ2_XXS's 256), sizing the
/// per-block decode scratch buffer.
pub const MAX_QK: usize = 256;

/// One block of a quantization format decoded by Joshua rather than candle.
pub trait RawBlock: Copy + Send + Sync + 'static {
    /// The GGUF / ggml dtype id (`GGML_TYPE_*`).
    const GGML_TYPE: u32;
    /// Elements per block.
    const QK: usize;
    /// Human-readable format name for errors.
    const NAME: &'static str;
    /// The [`GgmlDType`] a borrowed tensor of these blocks reports.  Formats
    /// candle has no variant for use a placeholder: only
    /// `QMatMul::from_qtensor` inspects it, and only to tell eagerly
    /// dequantized float types apart from everything else.
    const CANDLE_DTYPE: GgmlDType;

    /// Decode this block into `out` (`out.len() == Self::QK`).
    fn dequantize_block(&self, out: &mut [f32]);

    /// `dst[m, n] = lhs[m, k] · rhs[n, k]ᵀ`; a format with a fused SIMD
    /// kernel overrides the portable [`matmul_t`].
    fn matmul_t(
        mkn: (usize, usize, usize),
        lhs: &[f32],
        rhs: &[Self],
        dst: &mut [f32],
    ) -> Result<()> {
        matmul_t(mkn, lhs, rhs, dst)
    }
}

/// Bytes per block of `B`.
pub const fn block_bytes<B: RawBlock>() -> usize {
    std::mem::size_of::<B>()
}

/// Bytes taken by `elems` elements of `B`, or `None` for a partial block.
pub fn size_bytes<B: RawBlock>(elems: usize) -> Option<usize> {
    elems
        .is_multiple_of(B::QK)
        .then(|| elems / B::QK * block_bytes::<B>())
}

/// Reinterpret `bytes` as blocks of `B`.
///
/// Blocks are byte-aligned with no invalid bit patterns, so only the length
/// has to be a whole number of blocks.  The source must stay immutable for
/// the slice's lifetime (a heap buffer or a read-only mapping, see
/// `mmap_tensor`'s safety model).
pub fn blocks_from_bytes<B: RawBlock>(bytes: &[u8]) -> Result<&[B]> {
    const { assert!(std::mem::align_of::<B>() == 1) };
    let sz = block_bytes::<B>();
    if !bytes.len().is_multiple_of(sz) {
        bail!(
            "{}: {} bytes is not a whole number of {sz}-byte blocks",
            B::NAME,
            bytes.len()
        );
    }
    // SAFETY: `B` is align 1 (asserted above), made of plain bytes with no
    // padding or invalid patterns, and the length is an exact multiple.
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const B, bytes.len() / sz) })
}

/// Decode a run of blocks into `out`, which must hold `blocks.len() * QK`
/// values.
pub fn dequantize<B: RawBlock>(blocks: &[B], out: &mut [f32]) -> Result<()> {
    if out.len() != blocks.len() * B::QK {
        bail!(
            "{}: output holds {} values, expected {}",
            B::NAME,
            out.len(),
            blocks.len() * B::QK
        );
    }
    for (block, chunk) in blocks.iter().zip(out.chunks_exact_mut(B::QK)) {
        block.dequantize_block(chunk);
    }
    Ok(())
}

/// Decode raw tensor bytes of `B` holding `elems` elements to f32.
pub fn decode_bytes<B: RawBlock>(bytes: &[u8], elems: usize) -> Result<Vec<f32>> {
    let mut out = vec![0f32; elems];
    dequantize(blocks_from_bytes::<B>(bytes)?, &mut out)?;
    Ok(out)
}

/// Check a `matmul_t` call's shapes; returns the blocks per weight row.
pub fn validate_matmul_t<B: RawBlock>(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs: &[B],
    dst: &[f32],
) -> Result<usize> {
    if !k.is_multiple_of(B::QK) {
        bail!(
            "{} matmul: k={k} is not a multiple of the block size {}",
            B::NAME,
            B::QK
        );
    }
    let blocks_per_row = k / B::QK;
    if rhs.len() != n * blocks_per_row {
        bail!(
            "{} matmul: rhs holds {} blocks, expected {}",
            B::NAME,
            rhs.len(),
            n * blocks_per_row
        );
    }
    if lhs.len() != m * k || dst.len() != m * n {
        bail!(
            "{} matmul: lhs/dst sized {}/{}, expected {}/{}",
            B::NAME,
            lhs.len(),
            dst.len(),
            m * k,
            m * n
        );
    }
    Ok(blocks_per_row)
}

/// Portable `dst[m, n] = lhs[m, k] · rhs[n, k]ᵀ` with `rhs` held as blocks.
///
/// Each weight row is decoded a block at a time into a stack buffer and
/// accumulated against every lhs row, so the weights are never materialised
/// as f32 in bulk — the point when the weight matrix is larger than RAM.
/// Output rows are spread across the rayon pool.  `dst` is assigned, not
/// accumulated into (candle's convention).
pub fn matmul_t<B: RawBlock>(
    mkn: (usize, usize, usize),
    lhs: &[f32],
    rhs: &[B],
    dst: &mut [f32],
) -> Result<()> {
    matmul_t_rows(mkn, lhs, rhs, dst, true)
}

/// [`matmul_t`] with a serial row loop — the determinism tests use it to
/// prove the parallel path is bit-identical.
pub fn matmul_t_serial<B: RawBlock>(
    mkn: (usize, usize, usize),
    lhs: &[f32],
    rhs: &[B],
    dst: &mut [f32],
) -> Result<()> {
    matmul_t_rows(mkn, lhs, rhs, dst, false)
}

fn matmul_t_rows<B: RawBlock>(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs: &[B],
    dst: &mut [f32],
    parallel: bool,
) -> Result<()> {
    let blocks_per_row = validate_matmul_t((m, k, n), lhs, rhs, dst)?;
    if n == 0 || m == 0 {
        return Ok(());
    }
    // Each row is summed a block at a time, so the destination has to start
    // from zero.
    dst.fill(0.0);
    let dst_ptr = crate::simd::DstPtr::new(dst);
    let worker = |row: usize| matmul_row_scalar((m, k, n), lhs, rhs, blocks_per_row, row, &dst_ptr);
    if parallel {
        crate::simd::for_each_row(n, worker);
    } else {
        (0..n).for_each(worker);
    }
    Ok(())
}

/// Scalar per-row worker: decode each block of weight row `row` and
/// accumulate `lhs · block` into `dst[i*n + row]`.  `dst` must be zeroed.
pub(crate) fn matmul_row_scalar<B: RawBlock>(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs: &[B],
    blocks_per_row: usize,
    row: usize,
    dst: &crate::simd::DstPtr,
) {
    const { assert!(B::QK <= MAX_QK) };
    let row_blocks = &rhs[row * blocks_per_row..(row + 1) * blocks_per_row];
    let mut scratch = [0f32; MAX_QK];
    let decoded = &mut scratch[..B::QK];
    for (b, block) in row_blocks.iter().enumerate() {
        block.dequantize_block(decoded);
        let base = b * B::QK;
        for i in 0..m {
            let a = &lhs[i * k + base..i * k + base + B::QK];
            let mut acc = 0f32;
            for (x, w) in a.iter().zip(decoded.iter()) {
                acc += x * w;
            }
            // SAFETY: row `row` owns dst[i*n + row] for all i (disjoint per
            // row, see `crate::simd`); the buffer was zero-filled first.
            unsafe { dst.add(i * n + row, acc) };
        }
    }
}

/// f16 variant of [`RawBlock::matmul_t`], for candle's `QuantizedType`
/// contract: converts through f32, so the result is the f16-rounded f32
/// answer.
pub fn matmul_t_f16<B: RawBlock>(
    (m, k, n): (usize, usize, usize),
    lhs: &[f16],
    rhs: &[B],
    dst: &mut [f16],
) -> Result<()> {
    let lhs_f32: Vec<f32> = lhs.iter().map(|v| v.to_f32()).collect();
    let mut dst_f32 = vec![0f32; m * n];
    B::matmul_t((m, k, n), &lhs_f32, rhs, &mut dst_f32)?;
    for (o, v) in dst.iter_mut().zip(dst_f32.iter()) {
        *o = f16::from_f32(*v);
    }
    Ok(())
}

/// Evaluate `$body` with `$B` bound to the [`RawBlock`] type for GGUF dtype
/// id `$dtype`: `Some(body)` for a raw block format, `None` otherwise.
///
/// ```ignore
/// let bytes = with_raw_block!(dtype, B => raw_block::size_bytes::<B>(n));
/// ```
#[macro_export]
macro_rules! with_raw_block {
    ($dtype:expr, $B:ident => $body:expr) => {{
        match $dtype {
            <$crate::iq2xxs::BlockIq2Xxs as $crate::raw_block::RawBlock>::GGML_TYPE => {
                type $B = $crate::iq2xxs::BlockIq2Xxs;
                Some($body)
            }
            <$crate::mxfp4::BlockMxfp4 as $crate::raw_block::RawBlock>::GGML_TYPE => {
                type $B = $crate::mxfp4::BlockMxfp4;
                Some($body)
            }
            <$crate::low_bit::BlockQ1_0 as $crate::raw_block::RawBlock>::GGML_TYPE => {
                type $B = $crate::low_bit::BlockQ1_0;
                Some($body)
            }
            <$crate::low_bit::BlockQ2_0 as $crate::raw_block::RawBlock>::GGML_TYPE => {
                type $B = $crate::low_bit::BlockQ2_0;
                Some($body)
            }
            _ => None,
        }
    }};
}
pub use with_raw_block;

/// `(elements, bytes)` per block for a raw block format.
pub fn layout(dtype: u32) -> Option<(usize, usize)> {
    with_raw_block!(dtype, B => (B::QK, block_bytes::<B>()))
}

/// Whether GGUF dtype id `dtype` is a [`RawBlock`] format.
pub fn is_raw_block(dtype: u32) -> bool {
    layout(dtype).is_some()
}

#[cfg(test)]
pub(crate) mod testing {
    //! Format-independent checks every [`RawBlock`] implementation runs.
    use super::*;

    /// `matmul_t` agrees with a full dequantize + plain dot products, assigns
    /// rather than accumulates into `dst`, and parallel == serial bit for bit.
    pub fn check_matmul<B: RawBlock>(rhs: &[B], k: usize) {
        let n = rhs.len() * B::QK / k;
        let m = 3;
        let lhs: Vec<f32> = (0..m * k)
            .map(|i| ((i * 104729) % 1000) as f32 / 100.0 - 5.0)
            .collect();
        let mut weights = vec![0f32; n * k];
        dequantize(rhs, &mut weights).unwrap();

        let mut got = vec![123.75f32; m * n];
        B::matmul_t((m, k, n), &lhs, rhs, &mut got).unwrap();
        for i in 0..m {
            for r in 0..n {
                let want: f32 = (0..k).map(|j| lhs[i * k + j] * weights[r * k + j]).sum();
                let g = got[i * n + r];
                assert!(
                    (g - want).abs() <= 1e-3 * want.abs().max(1.0),
                    "{} matmul[{i},{r}] = {g}, reference {want}",
                    B::NAME
                );
            }
        }
        let mut par = vec![0f32; m * n];
        matmul_t((m, k, n), &lhs, rhs, &mut par).unwrap();
        let mut ser = vec![0f32; m * n];
        matmul_t_serial((m, k, n), &lhs, rhs, &mut ser).unwrap();
        assert_eq!(
            par,
            ser,
            "{}: parallel and serial must be bit-identical",
            B::NAME
        );

        // Shape errors are explicit.
        assert!(B::matmul_t((1, k + 1, n), &lhs[..k + 1], rhs, &mut got[..n]).is_err());
        assert!(B::matmul_t((1, k, n + 1), &lhs[..k], rhs, &mut got[..n + 1]).is_err());
    }

    /// A partial block is rejected; whole blocks reinterpret.
    pub fn check_blocks_from_bytes<B: RawBlock>() {
        let sz = block_bytes::<B>();
        assert!(blocks_from_bytes::<B>(&vec![0u8; 2 * sz]).is_ok());
        assert!(blocks_from_bytes::<B>(&vec![0u8; sz + 1]).is_err());
        assert_eq!(size_bytes::<B>(2 * B::QK), Some(2 * sz));
        assert_eq!(size_bytes::<B>(B::QK + 1), None);
        assert_eq!(layout(B::GGML_TYPE), Some((B::QK, sz)));
    }
}
