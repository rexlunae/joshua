//! MXFP4 (OCP Microscaling FP4) support — `GGML_TYPE_MXFP4`, type id 39.
//!
//! Kimi-K3-class models ship their weights natively in MXFP4, and candle has
//! no notion of the format: [`candle_core::quantized::GgmlDType`] stops at
//! `BF16`/`Q8K`, and its `from_u32` *rejects* type 39 outright, so candle
//! cannot even parse the header of such a GGUF, let alone its tensors.  This
//! module supplies the missing piece: the block layout, an exact decoder, and
//! a matmul that keeps weights in the memory mapping.
//!
//! # Format
//!
//! A block covers 32 elements in 17 bytes: one E8M0 shared-exponent byte plus
//! sixteen bytes of packed 4-bit E2M1 elements.
//!
//! * **Scale** — E8M0 is a bare 8-bit exponent, bias 127, decoding to
//!   `2^(e - 127)`.  Note `e == 0` denotes `2^-127`, which is *not* expressible
//!   by shifting the exponent field into place (that yields zero), so it is
//!   special-cased to the float32 denormal bit pattern.  `e == 0xFF` is NaN in
//!   the OCP spec and is treated here as corrupt data.
//! * **Elements** — 4-bit E2M1 codes indexing [`E2M1`].  ggml's own
//!   `kvalues_mxfp4` table stores these *doubled* so it can be `int8_t`, and
//!   compensates with a `* 0.5` (or by folding a half into the scale).  We
//!   use the true values directly, so no such factor appears here.
//! * **Packing** — split-half, matching ggml and every backend: within a
//!   block, byte `j` holds element `j` in its low nibble and element `j + 16`
//!   in its high nibble.  This is *not* sequential pairing, and getting it
//!   wrong silently scrambles every weight, so it is asserted in the tests.
//!
//! Blocks are `align_of == 1`, so unlike the k-quants they are always
//! borrowable straight out of a memory mapping regardless of file alignment.

use candle_core::Result;
use half::f16;

/// Elements per MXFP4 block.
pub const QK_MXFP4: usize = 32;

/// `GGML_TYPE_MXFP4` — the dtype id written in GGUF tensor info.
pub const GGML_TYPE_MXFP4: u32 = 39;

/// The sixteen E2M1 values, indexed by 4-bit code.
///
/// Sign in bit 3, 2-bit exponent (bias 1) in bits 2..1, 1-bit mantissa in bit
/// 0; exponent 0 is subnormal.  No infinities, no NaN.
pub const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// One MXFP4 block: an E8M0 scale byte followed by 16 packed nibble pairs.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockMxfp4 {
    /// Shared exponent, E8M0.
    pub e: u8,
    /// 32 elements as 16 packed nibble pairs (split-half order).
    pub qs: [u8; QK_MXFP4 / 2],
}

const _: () = assert!(std::mem::size_of::<BlockMxfp4>() == 17);
const _: () = assert!(std::mem::align_of::<BlockMxfp4>() == 1);

/// Decode an E8M0 shared exponent to its float multiplier, `2^(e - 127)`.
///
/// Returns `None` for `e == 0xFF`, which the OCP spec reserves for NaN and
/// which therefore indicates corrupt data rather than a usable scale.
#[inline]
pub fn e8m0_to_f32(e: u8) -> Option<f32> {
    match e {
        // 2^-127 is a float32 denormal; shifting the exponent into place would
        // produce 0.0, so use the bit pattern directly.
        0 => Some(f32::from_bits(0x0040_0000)),
        0xFF => None,
        _ => Some(f32::from_bits((e as u32) << 23)),
    }
}

impl BlockMxfp4 {
    /// Decode this block's 32 elements into `out`.
    ///
    /// A block whose scale byte is the reserved NaN encoding decodes to zeros
    /// rather than propagating infinities through the whole model.
    #[inline]
    pub fn dequantize(&self, out: &mut [f32; QK_MXFP4]) {
        let Some(d) = e8m0_to_f32(self.e) else {
            out.fill(0.0);
            return;
        };
        for j in 0..QK_MXFP4 / 2 {
            let byte = self.qs[j];
            // Split-half: low nibble → element j, high nibble → element j + 16.
            out[j] = d * E2M1[(byte & 0x0F) as usize];
            out[j + QK_MXFP4 / 2] = d * E2M1[(byte >> 4) as usize];
        }
    }
}

/// Decode a run of blocks into `out`, which must hold `blocks.len() * 32`
/// values.
pub fn dequantize(blocks: &[BlockMxfp4], out: &mut [f32]) -> Result<()> {
    if out.len() != blocks.len() * QK_MXFP4 {
        return Err(candle_core::Error::Msg(format!(
            "mxfp4: output holds {} values, expected {}",
            out.len(),
            blocks.len() * QK_MXFP4
        )));
    }
    let mut scratch = [0f32; QK_MXFP4];
    for (block, chunk) in blocks.iter().zip(out.chunks_exact_mut(QK_MXFP4)) {
        block.dequantize(&mut scratch);
        chunk.copy_from_slice(&scratch);
    }
    Ok(())
}

/// `dst[m, n] = lhs[m, k] · rhs[n, k]ᵀ`, with `rhs` held as MXFP4 blocks.
///
/// Rows are decoded a block at a time into a small stack buffer, so the
/// weights stay MXFP4 in the mapping and are never materialised as f32 in
/// bulk — which is the entire point when the weight matrix is larger than RAM.
pub fn matmul_t(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs: &[BlockMxfp4],
    dst: &mut [f32],
) -> Result<()> {
    let blocks_per_row = validate_matmul_t((m, k, n), lhs, rhs, dst)?;
    matmul_t_impl((m, k, n), blocks_per_row, lhs, rhs, dst, true)
}

/// Same as [`matmul_t`] with a serial row loop — used by the determinism
/// tests to prove the parallel path is bit-identical to serial.
pub fn matmul_t_serial(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs: &[BlockMxfp4],
    dst: &mut [f32],
) -> Result<()> {
    let blocks_per_row = validate_matmul_t((m, k, n), lhs, rhs, dst)?;
    matmul_t_impl((m, k, n), blocks_per_row, lhs, rhs, dst, false)
}

fn validate_matmul_t(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs: &[BlockMxfp4],
    dst: &mut [f32],
) -> Result<usize> {
    let err = |msg: String| candle_core::Error::Msg(msg);
    if !k.is_multiple_of(QK_MXFP4) {
        return Err(err(format!(
            "mxfp4 matmul: k={k} is not a multiple of the block size {QK_MXFP4}"
        )));
    }
    let blocks_per_row = k / QK_MXFP4;
    if rhs.len() != n * blocks_per_row {
        return Err(err(format!(
            "mxfp4 matmul: rhs holds {} blocks, expected {}",
            rhs.len(),
            n * blocks_per_row
        )));
    }
    if lhs.len() != m * k || dst.len() != m * n {
        return Err(err(format!(
            "mxfp4 matmul: lhs/dst sized {}/{}, expected {}/{}",
            lhs.len(),
            dst.len(),
            m * k,
            m * n
        )));
    }
    Ok(blocks_per_row)
}

fn matmul_t_impl(
    (m, k, n): (usize, usize, usize),
    blocks_per_row: usize,
    lhs: &[f32],
    rhs: &[BlockMxfp4],
    dst: &mut [f32],
    parallel: bool,
) -> Result<()> {
    if n == 0 || m == 0 {
        return Ok(());
    }
    // Each weight row is summed a block at a time, so the destination has to
    // start from zero. Clear it rather than requiring callers to — candle's
    // own `k_quants::matmul` assigns, and silently accumulating into a reused
    // buffer would corrupt results with no error.
    dst.fill(0.0);
    let dst_ptr = crate::simd::DstPtr::new(dst);
    let worker = |row: usize| {
        matmul_row_scalar(m, k, n, lhs, rhs, blocks_per_row, row, &dst_ptr);
    };
    if parallel {
        crate::simd::for_each_row(n, worker);
    } else {
        for row in 0..n {
            worker(row);
        }
    }
    Ok(())
}

/// Scalar per-row worker: decode each block and accumulate `lhs · block`
/// into `dst[i*n + row]` one block at a time (bit-compatible with the
/// original single-threaded loop).
#[allow(clippy::too_many_arguments)] // (m, k, n) is the matmul shape; kept flat for the hot loop
fn matmul_row_scalar(
    m: usize,
    k: usize,
    n: usize,
    lhs: &[f32],
    rhs: &[BlockMxfp4],
    blocks_per_row: usize,
    row: usize,
    dst: &crate::simd::DstPtr,
) {
    let row_blocks = &rhs[row * blocks_per_row..(row + 1) * blocks_per_row];
    let mut decoded = [0f32; QK_MXFP4];
    for (b, block) in row_blocks.iter().enumerate() {
        block.dequantize(&mut decoded);
        let base = b * QK_MXFP4;
        for i in 0..m {
            let a = &lhs[i * k + base..i * k + base + QK_MXFP4];
            let mut acc = 0f32;
            for (x, w) in a.iter().zip(decoded.iter()) {
                acc += x * w;
            }
            // SAFETY: row `row` owns dst[i*n + row] for all i (disjoint per
            // row, see `crate::simd`); the buffer was zero-filled by the
            // caller before any worker ran.
            unsafe { dst.add(i * n + row, acc) };
        }
    }
}

/// f16 variant of [`matmul_t`], for candle's `QuantizedType` contract.
///
/// Converts through f32: the weights are decoded exactly the same way and the
/// products accumulate in f32, so the result is the f16-rounded f32 answer —
/// matching what the IQ2_XXS and k-quant kernels do.
pub fn matmul_t_f16(
    (m, k, n): (usize, usize, usize),
    lhs: &[f16],
    rhs: &[BlockMxfp4],
    dst: &mut [f16],
) -> Result<()> {
    let mut lhs_f32 = vec![0f32; lhs.len()];
    for (o, v) in lhs_f32.iter_mut().zip(lhs.iter()) {
        *o = v.to_f32();
    }
    let mut dst_f32 = vec![0f32; m * n];
    matmul_t((m, k, n), &lhs_f32, rhs, &mut dst_f32)?;
    for (o, v) in dst.iter_mut().zip(dst_f32.iter()) {
        *o = f16::from_f32(*v);
    }
    Ok(())
}

/// Reinterpret `bytes` as MXFP4 blocks.
///
/// Blocks are byte-aligned, so this only has to check that the length is a
/// whole number of blocks — which makes MXFP4 weights borrowable directly from
/// a memory mapping.
pub fn blocks_from_bytes(bytes: &[u8]) -> Result<&[BlockMxfp4]> {
    let sz = std::mem::size_of::<BlockMxfp4>();
    if !bytes.len().is_multiple_of(sz) {
        return Err(candle_core::Error::Msg(format!(
            "mxfp4: {} bytes is not a whole number of {sz}-byte blocks",
            bytes.len()
        )));
    }
    // SAFETY: `BlockMxfp4` is `repr(C)`, all-`u8`, has alignment 1 and no
    // padding or invalid bit patterns, so any byte run of the right length is
    // a valid block sequence.
    Ok(unsafe {
        std::slice::from_raw_parts(bytes.as_ptr() as *const BlockMxfp4, bytes.len() / sz)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e8m0_decodes_to_powers_of_two() {
        // Bias 127: e == 127 is 2^0.
        assert_eq!(e8m0_to_f32(127), Some(1.0));
        assert_eq!(e8m0_to_f32(128), Some(2.0));
        assert_eq!(e8m0_to_f32(126), Some(0.5));
        assert_eq!(e8m0_to_f32(130), Some(8.0));
        // e == 0 is 2^-127, a denormal that the naive shift would flush to 0.
        let tiny = e8m0_to_f32(0).unwrap();
        assert!(tiny > 0.0, "e=0 must decode to 2^-127, not zero");
        assert_eq!(tiny, f32::from_bits(0x0040_0000));
        // e == 0xFF is the reserved NaN encoding.
        assert_eq!(e8m0_to_f32(0xFF), None);
    }

    #[test]
    fn block_uses_split_half_nibble_order() {
        // Element 0 in the low nibble of byte 0, element 16 in its high
        // nibble. If this were sequential pairing, element 1 would be 6.0.
        let mut qs = [0u8; 16];
        qs[0] = 0x7 | (0x1 << 4); // low = code 7 (+6.0), high = code 1 (+0.5)
        let block = BlockMxfp4 { e: 127, qs }; // scale 2^0 = 1.0
        let mut out = [0f32; QK_MXFP4];
        block.dequantize(&mut out);

        assert_eq!(out[0], 6.0, "low nibble of byte 0 is element 0");
        assert_eq!(out[16], 0.5, "high nibble of byte 0 is element 16");
        assert_eq!(out[1], 0.0, "element 1 comes from byte 1, not byte 0's high nibble");
    }

    #[test]
    fn scale_multiplies_every_element() {
        let mut qs = [0u8; 16];
        qs[3] = 0xF; // low nibble code 15 => -6.0
        let block = BlockMxfp4 { e: 129, qs }; // 2^2 = 4.0
        let mut out = [0f32; QK_MXFP4];
        block.dequantize(&mut out);
        assert_eq!(out[3], -24.0, "-6.0 * 4.0");
    }

    #[test]
    fn nan_scale_decodes_to_zeros_not_infinities() {
        let block = BlockMxfp4 {
            e: 0xFF,
            qs: [0xFF; 16],
        };
        let mut out = [0f32; QK_MXFP4];
        block.dequantize(&mut out);
        assert!(out.iter().all(|v| *v == 0.0), "corrupt scale must not poison the model");
    }

    #[test]
    fn full_e2m1_table_round_trips_through_a_block() {
        // Codes 0..15 across the low nibbles of the first 16 bytes.
        let mut qs = [0u8; 16];
        for (j, q) in qs.iter_mut().enumerate() {
            *q = j as u8;
        }
        let block = BlockMxfp4 { e: 127, qs };
        let mut out = [0f32; QK_MXFP4];
        block.dequantize(&mut out);
        for j in 0..16 {
            assert_eq!(out[j], E2M1[j], "code {j} must decode to its E2M1 value");
        }
    }

    #[test]
    fn matmul_matches_an_explicit_dequantized_reference() {
        // k = 64 (two blocks), n = 3 rows, m = 2.
        let (m, k, n) = (2usize, 64usize, 3usize);
        let blocks_per_row = k / QK_MXFP4;
        let mut rhs = Vec::new();
        for r in 0..n {
            for b in 0..blocks_per_row {
                let mut qs = [0u8; 16];
                for (j, q) in qs.iter_mut().enumerate() {
                    // Deterministic spread over the code space.
                    *q = (((j + r * 3 + b * 5) % 16) as u8) | ((((j + r) % 16) as u8) << 4);
                }
                rhs.push(BlockMxfp4 {
                    e: 127 + (r as u8 % 3),
                    qs,
                });
            }
        }
        let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 % 7.0) - 3.0).collect();

        let mut dst = vec![0f32; m * n];
        matmul_t((m, k, n), &lhs, &rhs, &mut dst).unwrap();

        // Reference: fully dequantize the weights, then plain dot products.
        let mut weights = vec![0f32; n * k];
        dequantize(&rhs, &mut weights).unwrap();
        for i in 0..m {
            for r in 0..n {
                let expect: f32 = (0..k).map(|j| lhs[i * k + j] * weights[r * k + j]).sum();
                let got = dst[i * n + r];
                assert!(
                    (got - expect).abs() < 1e-3,
                    "matmul[{i},{r}] = {got}, reference {expect}"
                );
            }
        }
    }

    #[test]
    fn matmul_assigns_rather_than_accumulating_into_a_dirty_buffer() {
        // Reusing an output buffer must give the same answer as a fresh one:
        // the routine is documented as assigning the product, and candle's
        // matmul does the same, so a caller following that convention must
        // not silently get sums of unrelated results.
        let (m, k, n) = (1usize, 32usize, 2usize);
        let rhs = vec![
            BlockMxfp4 {
                e: 127,
                qs: [0x21; 16],
            },
            BlockMxfp4 {
                e: 128,
                qs: [0x34; 16],
            },
        ];
        let lhs: Vec<f32> = (0..k).map(|i| (i % 5) as f32 - 2.0).collect();

        let mut fresh = vec![0f32; m * n];
        matmul_t((m, k, n), &lhs, &rhs, &mut fresh).unwrap();

        let mut dirty = vec![123.75f32; m * n];
        matmul_t((m, k, n), &lhs, &rhs, &mut dirty).unwrap();

        assert_eq!(fresh, dirty, "a pre-filled destination must be overwritten");
    }

    /// Parallel and serial execution must agree bit-for-bit: rows are
    /// independent and each is computed with identical per-element
    /// operations, so the thread count must not change a single bit.
    #[test]
    fn threaded_matches_serial_bit_exact() {
        let (m, k, n) = (4, 1024, 23);
        let blocks_per_row = k / QK_MXFP4;
        let mut rhs = Vec::new();
        for r in 0..n {
            for b in 0..blocks_per_row {
                let mut qs = [0u8; 16];
                for (j, q) in qs.iter_mut().enumerate() {
                    // Deterministic spread over the code space.
                    *q = (((j + r * 3 + b * 5) % 16) as u8) | ((((j + r) % 16) as u8) << 4);
                }
                rhs.push(BlockMxfp4 {
                    e: 127 + (r as u8 % 3),
                    qs,
                });
            }
        }
        let lhs: Vec<f32> = (0..m * k)
            .map(|i| ((i * 104729) % 1000) as f32 / 100.0 - 5.0)
            .collect();

        let mut par = vec![0f32; m * n];
        matmul_t((m, k, n), &lhs, &rhs, &mut par).unwrap();

        let mut ser = vec![0f32; m * n];
        matmul_t_serial((m, k, n), &lhs, &rhs, &mut ser).unwrap();

        assert_eq!(par, ser, "parallel and serial matmul must be bit-identical");
    }

    #[test]
    fn blocks_from_bytes_rejects_a_partial_block() {
        assert!(blocks_from_bytes(&[0u8; 17]).is_ok());
        assert!(blocks_from_bytes(&[0u8; 34]).is_ok());
        assert!(blocks_from_bytes(&[0u8; 20]).is_err());
    }
}
