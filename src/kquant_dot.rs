//! Fused AVX2 dequant+dot kernels for candle's k-quant block types used by
//! the DeepSeek-V4-Flash GGUF (Q8_0, Q2_K, Q4_K).
//!
//! The generic fast path in [`crate::quant_matmul`] dequantizes each weight
//! row into an f32 scratch buffer (via candle's scalar `GgmlType::to_float`)
//! and only then SIMD-dots it against the activations.  That materializes
//! k×4 bytes of f32 per row — ~16x the weight's on-disk bytes — and the
//! dequant itself runs scalar.  The kernels here do what llama.cpp's
//! `ggml-cpu` backend does instead: the quantized blocks are read straight
//! from the mmap, decoded *inside* the dot in 8-lane registers, and FMA-
//! accumulated against the activations.  No f32 weight row is ever written.
//!
//! Block layouts mirror candle's `k_quants` types exactly (the GGUF format
//! is fixed; candle's `to_float`/`vec_dot_unopt` are the reference):
//!
//! * **Q8_0** — 32 i8 values, one f16 scale `d`; `y = d·q`.
//! * **Q2_K** — 256 values in 2-bit fields.  Value `v` of a block lives in
//!   byte `v % 32` (within its 128-value super-group), field
//!   `2·((v/32) mod 4)`.  Each of the 16 scale bytes covers a 16-value
//!   group: `dl = s & 0xF`, `dh = s >> 4`; `y = d·dl·q − dmin·dh`.
//! * **Q4_K** — 256 4-bit values.  Value `v` lives in byte
//!   `32·(v/64) + (v mod 32)`, nibble `(v/32) mod 2`.  Each of the 8 scale
//!   pairs (6 bits each, packed into 12 bytes) covers a 32-value group;
//!   `y = d·sc·q − dmin·m`.
//!
//! # Safety model
//!
//! Same contract as the rest of joshua's SIMD matmuls (see `crate::simd`):
//! rows are independent, each worker writes exactly `dst[i·n + row]` for
//! `i in 0..m` through a [`crate::simd::DstPtr`], and every byte pattern is
//! a valid block, so reinterpreting a checked byte slice as blocks is sound.
//! The kernels are only entered after `avx2_fma_available()` and carry
//! `#[target_feature(enable = "avx2,fma")]`.

use candle_core::quantized::GgmlDType;
use half::f16;

/// Elements per Q8_0 block (candle `QK8_0`).
const QK8_0: usize = 32;
/// Elements per K-quant super-block (candle `QK_K`).
const QK_K: usize = 256;

/// Raw Q8_0 block — byte-identical to candle's `BlockQ8_0` (`d`, 32 i8).
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct BlockQ8_0Raw {
    /// f16 block scale.
    pub d: [u8; 2],
    /// 32 signed values.
    pub qs: [i8; QK8_0],
}

/// Raw Q2_K block — byte-identical to candle's `BlockQ2K`
/// (`scales[16]`, `qs[64]`, `d`, `dmin`).
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct BlockQ2KRaw {
    /// 16 scale bytes: `dl = s & 0xF`, `dh = s >> 4` per 16-value group.
    pub scales: [u8; 16],
    /// 256 values packed 4 per byte (2-bit fields).
    pub qs: [u8; 64],
    /// f16 super-block scale.
    pub d: [u8; 2],
    /// f16 super-block min.
    pub dmin: [u8; 2],
}

/// Raw Q4_K block — byte-identical to candle's `BlockQ4K`
/// (`d`, `dmin`, `scales[12]`, `qs[128]`).  The 12 scale bytes pack 8
/// scale/min pairs with 6 bits each (llama.cpp's scheme, see
/// `get_scale_min_k4`); each pair covers 32 values.
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct BlockQ4KRaw {
    /// f16 super-block scale.
    pub d: [u8; 2],
    /// f16 super-block min.
    pub dmin: [u8; 2],
    /// 12 packed scale/min bytes (6 bits each, 8 pairs).
    pub scales: [u8; 12],
    /// 256 values packed 2 per byte (4-bit nibbles).
    pub qs: [u8; 128],
}

const _: () = {
    assert!(std::mem::size_of::<BlockQ8_0Raw>() == 34);
    assert!(std::mem::size_of::<BlockQ2KRaw>() == 84);
    assert!(std::mem::size_of::<BlockQ4KRaw>() == 144);
    // These must stay in lockstep with candle's block structs (the GGUF
    // reader and the streamed loader size reads through these layouts).
    assert!(std::mem::size_of::<BlockQ8_0Raw>() == std::mem::size_of::<candle_core::quantized::k_quants::BlockQ8_0>());
    assert!(std::mem::size_of::<BlockQ2KRaw>() == std::mem::size_of::<candle_core::quantized::k_quants::BlockQ2K>());
    assert!(std::mem::size_of::<BlockQ4KRaw>() == std::mem::size_of::<candle_core::quantized::k_quants::BlockQ4K>());
};

/// Decode scale pair `j` (0..8) of a Q4_K block's 12 packed scale bytes
/// (candle `get_scale_min_k4`): `(d, m)` with 6 bits each.
fn q4k_scale_min(scales: &[u8; 12], j: usize) -> (u8, u8) {
    if j < 4 {
        let d = scales[j] & 63;
        let m = scales[j + 4] & 63;
        (d, m)
    } else {
        let d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// Try the fused AVX2 matmul for `dtype`.  Returns `Ok(true)` when the
/// kernel ran and `dst` was fully written, `Ok(false)` when `dtype` has no
/// fused kernel here (the caller keeps its generic fallback).
///
/// `block_bytes` must hold `n · k/BLOCK_ELEMS` raw blocks in GGUF byte
/// order; `lhs`/`dst` are `m×k` / `m×n` f32 row-major.
pub fn try_matmul_fused_avx2(
    dtype: GgmlDType,
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    block_bytes: &[u8],
    dst: &mut [f32],
    parallel: bool,
) -> bool {
    if !crate::simd::avx2_fma_available() {
        return false;
    }
    match dtype {
        GgmlDType::Q8_0 => try_fused::<BlockQ8_0Raw, QK8_0>(m, k, n, lhs, block_bytes, dst, parallel, fused_row_q8_0),
        GgmlDType::Q2K => try_fused::<BlockQ2KRaw, QK_K>(m, k, n, lhs, block_bytes, dst, parallel, fused_row_q2k),
        GgmlDType::Q4K => try_fused::<BlockQ4KRaw, QK_K>(m, k, n, lhs, block_bytes, dst, parallel, fused_row_q4k),
        _ => false,
    }
}

/// Row-kernel signature: process one output row of the `(m, k, n)` matmul.
#[cfg(target_arch = "x86_64")]
type RowKernel<B> = unsafe fn(
    m: usize,
    k: usize,
    n: usize,
    lhs: &[f32],
    blocks: &[B],
    blocks_per_row: usize,
    row: usize,
    dst: &crate::simd::DstPtr,
);

/// Shared driver: reinterpret the bytes as `B` blocks, then run the fused
/// row kernel over `n` rows (parallel across the rayon pool when asked).
///
/// Returns `false` (and writes nothing) if the byte length does not line up
/// with whole blocks — the caller's generic path then takes over.  `k` must
/// be a multiple of `BLOCK_ELEMS` (guaranteed by `quant_matmul::validate`
/// before this is reached).
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)] // (m, k, n) is the matmul shape; kept flat for the hot loop
fn try_fused<B: Sync, const BLOCK_ELEMS: usize>(
    m: usize,
    k: usize,
    n: usize,
    lhs: &[f32],
    block_bytes: &[u8],
    dst: &mut [f32],
    parallel: bool,
    row_kernel: RowKernel<B>,
) -> bool {
    let block_size = std::mem::size_of::<B>();
    if !block_bytes.len().is_multiple_of(block_size) {
        return false;
    }
    let blocks_per_row = k / BLOCK_ELEMS;
    if blocks_per_row == 0 {
        return false;
    }
    // SAFETY: B is packed and every byte pattern is a valid block, so any
    // byte slice of whole-block length is a valid block slice; the caller
    // keeps the underlying buffer (an mmap borrow) alive.  The row kernel
    // only reads.
    let blocks: &[B] = unsafe {
        std::slice::from_raw_parts(
            block_bytes.as_ptr() as *const B,
            block_bytes.len() / block_size,
        )
    };
    let dst_ptr = crate::simd::DstPtr::new(dst);
    let worker = |row: usize| {
        // SAFETY: avx2_fma_available() was checked by the caller
        // (`try_matmul_fused_avx2`), so the target_feature kernel may run;
        // row `row` writes exactly dst[i*n + row] for i in 0..m, disjoint
        // from every other row (see `crate::simd`'s safety model).
        unsafe { row_kernel(m, k, n, lhs, blocks, blocks_per_row, row, &dst_ptr) }
    };
    if parallel {
        crate::simd::for_each_row(n, worker);
    } else {
        for row in 0..n {
            worker(row);
        }
    }
    true
}

#[cfg(not(target_arch = "x86_64"))]
fn try_fused<B: Sync, const BLOCK_ELEMS: usize>(
    _m: usize,
    _k: usize,
    _n: usize,
    _lhs: &[f32],
    _block_bytes: &[u8],
    _dst: &mut [f32],
    _parallel: bool,
    _row_kernel: unsafe fn(
        m: usize,
        k: usize,
        n: usize,
        lhs: &[f32],
        blocks: &[B],
        blocks_per_row: usize,
        row: usize,
        dst: &crate::simd::DstPtr,
    ),
) -> bool {
    false
}

// ─── Row kernels ──────────────────────────────────────────────────────────
//
// Each kernel processes one weight row: loop over the row's blocks, decode
// each block into 8-lane registers, FMA-accumulate against the m-tile of
// activation rows.  Results differ from the scalar dequant+dot path by at
// most FMA rounding (the dequant `d·dl·q − dmin·dh` is a single fmsub
// instead of two rounded ops), far below the tests' tolerances.

/// # Safety
/// Caller must have verified AVX2+FMA and row-disjointness (see
/// [`try_fused`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)] // (m, k, n) is the matmul shape; kept flat for the hot loop
unsafe fn fused_row_q8_0(
    m: usize,
    k: usize,
    n: usize,
    lhs: &[f32],
    blocks: &[BlockQ8_0Raw],
    blocks_per_row: usize,
    row: usize,
    dst: &crate::simd::DstPtr,
) {
    use std::arch::x86_64::*;

    const MTILE: usize = 4;
    let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
    let mut m0 = 0;
    while m0 < m {
        let mcnt = (m - m0).min(MTILE);
        let mut acc = [_mm256_setzero_ps(); MTILE];
        for (b, block) in row_blocks.iter().enumerate() {
            let d = f16::from_le_bytes(block.d).to_f32();
            let d4 = _mm256_set1_ps(d);
            // 32 i8 → four 8-lane f32 vectors (cvtepi8_epi32 sign-extends
            // only the low 8 bytes of its 128-bit operand), scaled by d.
            let lo = _mm_loadu_si128(block.qs.as_ptr().add(0) as *const __m128i);
            let hi = _mm_loadu_si128(block.qs.as_ptr().add(16) as *const __m128i);
            let w0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), d4);
            let w1 = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))),
                d4,
            );
            let w2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)), d4);
            let w3 = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))),
                d4,
            );
            let base = b * QK8_0;
            for i in 0..mcnt {
                let a = lhs[(m0 + i) * k + base..].as_ptr();
                acc[i] = _mm256_fmadd_ps(_mm256_loadu_ps(a), w0, acc[i]);
                acc[i] = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(8)), w1, acc[i]);
                acc[i] = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(16)), w2, acc[i]);
                acc[i] = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(24)), w3, acc[i]);
            }
        }
        for (i, acc_i) in acc.iter().enumerate().take(mcnt) {
            // SAFETY: row `row` owns dst[i*n + row] for all i (see
            // `crate::simd`); disjoint per row.
            dst.write((m0 + i) * n + row, crate::simd::hsum256(*acc_i));
        }
        m0 += MTILE;
    }
}

/// # Safety
/// See [`fused_row_q8_0`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)] // (m, k, n) is the matmul shape; kept flat for the hot loop
unsafe fn fused_row_q2k(
    m: usize,
    k: usize,
    n: usize,
    lhs: &[f32],
    blocks: &[BlockQ2KRaw],
    blocks_per_row: usize,
    row: usize,
    dst: &crate::simd::DstPtr,
) {
    use std::arch::x86_64::*;

    const MTILE: usize = 4;
    let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
    let m3 = _mm256_set1_epi8(3);
    let mut m0 = 0;
    while m0 < m {
        let mcnt = (m - m0).min(MTILE);
        let mut acc = [_mm256_setzero_ps(); MTILE];
        for (b, block) in row_blocks.iter().enumerate() {
            let d = f16::from_le_bytes(block.d).to_f32();
            let dmin = f16::from_le_bytes(block.dmin).to_f32();
            let base = b * QK_K;
            for sg in 0..2usize {
                // 32 bytes = 128 values.  Sub-group j (0..8) covers values
                // [16j, 16j+16) with scale byte sg*8+j: shift 2*(j/2),
                // byte window low half for even j, high half for odd j
                // (candle `BlockQ2K::to_float`, matching llama.cpp).
                let x = _mm256_loadu_si256(block.qs[sg * 32..sg * 32 + 32].as_ptr() as *const __m256i);
                // Field extraction: the 16-bit-lane shift trick llama.cpp
                // uses; `& 3` cleans the spill into the adjacent byte.
                let xf = [
                    _mm256_and_si256(x, m3),
                    _mm256_and_si256(_mm256_srli_epi16(x, 2), m3),
                    _mm256_and_si256(_mm256_srli_epi16(x, 4), m3),
                    _mm256_and_si256(_mm256_srli_epi16(x, 6), m3),
                ];
                for j in 0..8usize {
                    let sc = block.scales[sg * 8 + j];
                    let d1 = _mm256_set1_ps(d * (sc & 0xF) as f32);
                    let m1 = _mm256_set1_ps(dmin * (sc >> 4) as f32);
                    let xv = xf[j / 2];
                    let bytes = if j % 2 == 0 {
                        _mm256_castsi256_si128(xv)
                    } else {
                        _mm256_extracti128_si256(xv, 1)
                    };
                    // Widen 16 bytes to 2× 8-lane f32 and dequantize in
                    // registers: w' = d1·q − m1 (one fmsub, no f32 row).
                    let w0 = _mm256_fmsub_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes)),
                        d1,
                        m1,
                    );
                    let w1 = _mm256_fmsub_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(bytes, 8))),
                        d1,
                        m1,
                    );
                    let a = base + sg * 128 + 16 * j;
                    for i in 0..mcnt {
                        let p = lhs[(m0 + i) * k + a..].as_ptr();
                        acc[i] = _mm256_fmadd_ps(_mm256_loadu_ps(p), w0, acc[i]);
                        acc[i] = _mm256_fmadd_ps(_mm256_loadu_ps(p.add(8)), w1, acc[i]);
                    }
                }
            }
        }
        for (i, acc_i) in acc.iter().enumerate().take(mcnt) {
            // SAFETY: row `row` owns dst[i*n + row] for all i; disjoint per row.
            dst.write((m0 + i) * n + row, crate::simd::hsum256(*acc_i));
        }
        m0 += MTILE;
    }
}

/// # Safety
/// See [`fused_row_q8_0`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)] // (m, k, n) is the matmul shape; kept flat for the hot loop
unsafe fn fused_row_q4k(
    m: usize,
    k: usize,
    n: usize,
    lhs: &[f32],
    blocks: &[BlockQ4KRaw],
    blocks_per_row: usize,
    row: usize,
    dst: &crate::simd::DstPtr,
) {
    use std::arch::x86_64::*;

    const MTILE: usize = 4;
    let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
    let mf = _mm256_set1_epi8(0x0F);
    let mut m0 = 0;
    while m0 < m {
        let mcnt = (m - m0).min(MTILE);
        let mut acc = [_mm256_setzero_ps(); MTILE];
        for (b, block) in row_blocks.iter().enumerate() {
            let d = f16::from_le_bytes(block.d).to_f32();
            let dmin = f16::from_le_bytes(block.dmin).to_f32();
            let base = b * QK_K;
            for c in 0..4usize {
                // 32 bytes = 64 values.  Low nibbles = values [64c, 64c+32)
                // with scale pair 2c; high nibbles = [64c+32, 64c+64) with
                // scale pair 2c+1.
                let x = _mm256_loadu_si256(block.qs[c * 32..c * 32 + 32].as_ptr() as *const __m256i);
                let x_lo = _mm256_and_si256(x, mf);
                let x_hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), mf);
                for r in 0..2usize {
                    let xv = if r == 0 { x_lo } else { x_hi };
                    let (sc, m) = q4k_scale_min(&block.scales, 2 * c + r);
                    let d1 = _mm256_set1_ps(d * sc as f32);
                    let m1 = _mm256_set1_ps(dmin * m as f32);
                    // Widen the 32 bytes to 4× 8-lane f32 and dequantize in
                    // registers: w' = d1·q − m1 (one fmsub, no f32 row).
                    let low = _mm256_castsi256_si128(xv);
                    let high = _mm256_extracti128_si256(xv, 1);
                    let w0 = _mm256_fmsub_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(low)),
                        d1,
                        m1,
                    );
                    let w1 = _mm256_fmsub_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(low, 8))),
                        d1,
                        m1,
                    );
                    let w2 = _mm256_fmsub_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(high)),
                        d1,
                        m1,
                    );
                    let w3 = _mm256_fmsub_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(high, 8))),
                        d1,
                        m1,
                    );
                    let a = base + 64 * c + 32 * r;
                    for i in 0..mcnt {
                        let p = lhs[(m0 + i) * k + a..].as_ptr();
                        acc[i] = _mm256_fmadd_ps(_mm256_loadu_ps(p), w0, acc[i]);
                        acc[i] = _mm256_fmadd_ps(_mm256_loadu_ps(p.add(8)), w1, acc[i]);
                        acc[i] = _mm256_fmadd_ps(_mm256_loadu_ps(p.add(16)), w2, acc[i]);
                        acc[i] = _mm256_fmadd_ps(_mm256_loadu_ps(p.add(24)), w3, acc[i]);
                    }
                }
            }
        }
        for (i, acc_i) in acc.iter().enumerate().take(mcnt) {
            // SAFETY: row `row` owns dst[i*n + row] for all i; disjoint per row.
            dst.write((m0 + i) * n + row, crate::simd::hsum256(*acc_i));
        }
        m0 += MTILE;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::k_quants::{BlockQ2K, BlockQ4K, BlockQ8_0, GgmlType};
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device, Tensor};

    /// Quantize a random `[n, k]` f32 tensor with `dtype` and return the raw
    /// block bytes (mirrors `quant_matmul`'s test helper).
    fn quantized_block_bytes<T: GgmlType>(n: usize, k: usize, dtype: GgmlDType) -> Vec<u8> {
        let data: Vec<f32> = (0..n * k)
            .map(|i| ((((i as u64) * 2654435761) % 100000) as f32 / 1000.0) - 50.0)
            .collect();
        let t = Tensor::from_vec(data, (n, k), &Device::Cpu).unwrap();
        let qt = QTensor::quantize(&t, dtype).unwrap();
        qt.data().unwrap().to_vec()
    }

    fn run_case(dtype: GgmlDType, m: usize, k: usize, n: usize) {
        let block_bytes = quantized_block_bytes::<BlockQ8_0>(n, k, dtype); // any GgmlType works
        let lhs: Vec<f32> = (0..m * k)
            .map(|i| (((i * 40503) % 1000) as f32 / 100.0) - 5.0)
            .collect();
        let mut fast = vec![0f32; m * n];
        let ran = try_matmul_fused_avx2(dtype, (m, k, n), &lhs, &block_bytes, &mut fast, false);
        assert!(ran, "{dtype:?}: fused kernel must handle this dtype");

        // Reference weights: candle's own dequant (the authoritative GGUF
        // decode).
        let w = match dtype {
            GgmlDType::Q8_0 => {
                let blocks = blocks_of::<BlockQ8_0>(&block_bytes);
                let mut w = vec![0f32; n * k];
                BlockQ8_0::to_float(&blocks, &mut w);
                w
            }
            GgmlDType::Q2K => {
                let blocks = blocks_of::<BlockQ2K>(&block_bytes);
                let mut w = vec![0f32; n * k];
                BlockQ2K::to_float(&blocks, &mut w);
                w
            }
            GgmlDType::Q4K => {
                let blocks = blocks_of::<BlockQ4K>(&block_bytes);
                let mut w = vec![0f32; n * k];
                BlockQ4K::to_float(&blocks, &mut w);
                w
            }
            _ => unreachable!(),
        };

        // Reference A — exact accumulation order.  The fused kernel keeps one
        // 8-lane FMA accumulator per m-row and hsums at the end, so this
        // reference replicates that lane order with `mul_add` (single
        // rounding).  The only remaining difference is the dequant rounding
        // (fused: one fmsub; to_float: two rounded ops), ≤ 1 ulp per
        // element.  This is what pins the numerics to FMA-rounding level.
        for (i, fastv) in fast.iter().enumerate() {
            let (ii, jj) = (i / n, i % n);
            let mut acc = [0f32; 8];
            let mut c = 0;
            while c + 8 <= k {
                for l in 0..8 {
                    acc[l] = lhs[ii * k + c + l].mul_add(w[jj * k + c + l], acc[l]);
                }
                c += 8;
            }
            let s = acc.iter().sum::<f32>();
            let tol = 1e-3 * s.abs().max(1.0);
            assert!(
                (fastv - s).abs() <= tol,
                "{dtype:?}: dst[{i}] fused={} same-order-dot={} (tol {tol})",
                fast[i],
                s
            );
        }

        // Reference B — true f32 GEMM over the dequantised weights, a decode
        // sanity check.  Its accumulation order differs (scalar sequential),
        // so the comparison absorbs the order-dependent rounding; a decode
        // bug shifts results by ≳1%, far beyond the 2% tolerance.
        for (i, fastv) in fast.iter().enumerate() {
            let mut acc = 0f32;
            let (ii, jj) = (i / n, i % n);
            for (kk, wv) in w[jj * k..(jj + 1) * k].iter().enumerate() {
                acc += lhs[ii * k + kk] * wv;
            }
            let tol = 2e-2 * acc.abs().max(1.0);
            assert!(
                (fastv - acc).abs() <= tol,
                "{dtype:?}: dst[{i}] fused={} f32-gemm={} (tol {tol})",
                fast[i],
                acc
            );
        }
    }

    fn blocks_of<T: Clone>(bytes: &[u8]) -> Vec<T> {
        let size = std::mem::size_of::<T>();
        assert!(bytes.len() % size == 0);
        let ptr = bytes.as_ptr() as *const T;
        let len = bytes.len() / size;
        let blocks = unsafe { std::slice::from_raw_parts(ptr, len) };
        blocks.to_vec()
    }

    #[test]
    fn fused_matches_f32_gemm() {
        run_case(GgmlDType::Q8_0, 1, 256, 40);
        run_case(GgmlDType::Q8_0, 3, 512, 17);
        run_case(GgmlDType::Q2K, 1, 256, 40);
        run_case(GgmlDType::Q2K, 3, 512, 11);
        run_case(GgmlDType::Q2K, 5, 768, 9);
        run_case(GgmlDType::Q4K, 1, 256, 40);
        run_case(GgmlDType::Q4K, 3, 512, 13);
        run_case(GgmlDType::Q4K, 4, 768, 25);
    }

    /// The real model's shapes: decode m=1 and m=32 against f32 GEMM.
    #[test]
    fn fused_matches_f32_gemm_at_model_shapes() {
        // expert gate/up: k=7680 (30 Q2_K blocks), n=2048; down: k=2048.
        run_case(GgmlDType::Q2K, 1, 7680, 2048);
        run_case(GgmlDType::Q4K, 1, 2048, 7680);
        // attention: k=4096 Q8_0.
        run_case(GgmlDType::Q8_0, 1, 4096, 4096);
        run_case(GgmlDType::Q8_0, 32, 4096, 4096);
    }

    /// Parallel and serial fused execution must agree bit-for-bit.
    #[test]
    fn fused_parallel_matches_serial_bit_exact() {
        for dtype in [GgmlDType::Q8_0, GgmlDType::Q2K, GgmlDType::Q4K] {
            let block_bytes = quantized_block_bytes::<BlockQ8_0>(24, 256, dtype);
            let lhs: Vec<f32> = (0..3 * 256).map(|i| (i as f32) * 0.01 - 1.0).collect();
            let mut par = vec![0f32; 3 * 24];
            assert!(try_matmul_fused_avx2(dtype, (3, 256, 24), &lhs, &block_bytes, &mut par, true));
            let mut ser = vec![0f32; 3 * 24];
            assert!(try_matmul_fused_avx2(dtype, (3, 256, 24), &lhs, &block_bytes, &mut ser, false));
            assert_eq!(par, ser, "{dtype:?}: parallel and serial must be bit-identical");
        }
    }

    /// Dtypes without a fused kernel must be declined (caller falls back).
    #[test]
    fn unknown_dtype_is_declined() {
        let block_bytes = quantized_block_bytes::<BlockQ8_0>(2, 256, GgmlDType::Q8_0);
        let mut dst = vec![0f32; 2];
        // Q3K has no fused kernel here.
        assert!(!try_matmul_fused_avx2(
            GgmlDType::Q3K,
            (1, 256, 2),
            &vec![0f32; 256],
            &block_bytes,
            &mut dst,
            false
        ));
    }

    /// Bad byte length must be declined, not panic.
    #[test]
    fn mismatched_byte_length_is_declined() {
        let mut dst = vec![0f32; 8];
        assert!(!try_matmul_fused_avx2(
            GgmlDType::Q8_0,
            (1, 256, 8),
            &vec![0f32; 256],
            &[0u8; 7], // not a whole number of 34-byte blocks
            &mut dst,
            false
        ));
    }
}
