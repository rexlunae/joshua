//! IQ2_XXS — the 2.0625-bit trellis quantisation DeepSeek-V4-Flash GGUFs use
//! for their routed expert gate/up projections.
//!
//! The vendored candle knows the dtype (`GgmlDType::Iq2Xxs`, with the block
//! type and tables in `candle_core::quantized::iq2xxs`) so device backends can
//! hold the blocks as ordinary quantized storage; on the host the tensors
//! that use it are exactly the ones too large to ever materialise as f32
//! (both expert projections across 256 experts are ~40 GB of 2-bit data; as
//! f32 they would be ~640 GB).  This module therefore mirrors what
//! [`crate::mxfp4`] does for MXFP4: the blocks stay in the memory mapping and
//! are decoded a block at a time inside the fused matmul below.  The GGUF
//! reader keeps IQ2_XXS on its raw-header path (see
//! `gguf_ext::is_candle_supported`) so the CPU form is this module's, never
//! candle's reference decode.
//!
//! The format and tables are a direct port of llama.cpp's
//! `ggml-quants.c`/`ggml-common.h`:
//!   block of 256 elements, 66 bytes: 16-bit fp16 scale, then 32 uint16
//!   codes.  Each 32-element group indexes the 256-entry `iq2xxs_grid`
//!   trellis with 4 codes of 8 values, and stores sign bits packed 7 per
//!   code in the upper 28 bits of the second 32-bit word.

use candle_core::Result;

/// GGML type id for IQ2_XXS (ggml.h `GGML_TYPE_IQ2_XXS`).
pub const GGML_TYPE_IQ2_XXS: u32 = 16;
// The block type, its decode and the trellis tables are candle-core's (the
// dtype is `GgmlDType::Iq2Xxs` there, so device backends can hold the blocks
// as ordinary `QStorage`); this module keeps the host-side fused AVX2 matmul
// and the mmap helpers around them.
pub use candle_core::quantized::iq2xxs::{
    BlockIq2Xxs, BLOCK_BYTES, IQ2XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS, QK_IQ2_XXS,
};

impl crate::raw_block::RawBlock for BlockIq2Xxs {
    const GGML_TYPE: u32 = GGML_TYPE_IQ2_XXS;
    const QK: usize = QK_IQ2_XXS;
    const NAME: &'static str = "iq2_xxs";
    const CANDLE_DTYPE: candle_core::quantized::GgmlDType =
        candle_core::quantized::GgmlDType::Iq2Xxs;

    fn dequantize_block(&self, out: &mut [f32]) {
        self.dequantize(out.try_into().expect("one IQ2_XXS block"));
    }

    const DECODE_AVX512: bool = true;

    /// Two 8-value grid codes per 16-lane register: the codes' grid bytes
    /// widen straight to f32 (`vpmovzxbd`) and their sign patterns become
    /// one 16-bit lane mask that negates in place (`vsubps` under mask) — no
    /// ±1 multiply and no cross-lane shuffles, so the decode stays cheap
    /// even at one activation row.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")]
    #[inline]
    unsafe fn decode_avx512(&self, ib32: usize) -> [std::arch::x86_64::__m512; 2] {
        use std::arch::x86_64::*;
        let zero = _mm512_setzero_ps();
        let d = half::f16::from_le_bytes(self.d).to_f32();
        let q = &self.qs[ib32 * 8..ib32 * 8 + 8];
        let lo = u32::from_le_bytes([q[0], q[1], q[2], q[3]]);
        let hi = u32::from_le_bytes([q[4], q[5], q[6], q[7]]);
        let db = _mm512_set1_ps(d * (0.5 + ((hi >> 28) as f32)) * 0.25);
        let mut w = [zero; 2];
        // (A loop rather than `array::map`: a closure called through a
        // non-target-feature function cannot inline these intrinsics.)
        for (h, wh) in w.iter_mut().enumerate() {
            let l = 2 * h as u32;
            let (c0, c1) = ((lo >> (8 * l)) & 0xFF, (lo >> (8 * (l + 1))) & 0xFF);
            let grid = _mm_set_epi64x(IQ2XXS_GRID[c1 as usize] as i64, IQ2XXS_GRID[c0 as usize] as i64);
            let v = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(grid));
            // Bit j of a sign byte negates lane j of its code.
            let s0 = KSIGNS_IQ2XS[((hi >> (7 * l)) & 0x7F) as usize] as u16;
            let s1 = KSIGNS_IQ2XS[((hi >> (7 * (l + 1))) & 0x7F) as usize] as u16;
            *wh = _mm512_mul_ps(_mm512_mask_sub_ps(v, s0 | (s1 << 8), zero, v), db);
        }
        w
    }

    fn matmul_t(
        mkn: (usize, usize, usize),
        lhs: &[f32],
        rhs: &[Self],
        dst: &mut [f32],
    ) -> Result<()> {
        matmul_t(mkn, lhs, rhs, dst)
    }
}

/// `dst[m, n] = lhs[m, k] · rhs[n, k]ᵀ`, with `rhs` held as IQ2_XXS blocks.
///
/// At the AVX2 level this runs the fused AVX2 dequant+dot kernel (see
/// `try_avx2_matmul`), spreading the independent output rows across the
/// rayon pool; every other level goes through [`crate::raw_block::matmul_t`],
/// whose AVX-512 form decodes with [`RawBlock::decode_avx512`].  Every path
/// is deterministic and they agree to FMA / accumulation-order rounding.
///
/// [`RawBlock::decode_avx512`]: crate::raw_block::RawBlock::decode_avx512
pub fn matmul_t(
    mkn: (usize, usize, usize),
    lhs: &[f32],
    rhs: &[BlockIq2Xxs],
    dst: &mut [f32],
) -> Result<()> {
    matmul_t_dispatch(mkn, lhs, rhs, dst, true)
}

fn matmul_t_dispatch(
    mkn: (usize, usize, usize),
    lhs: &[f32],
    rhs: &[BlockIq2Xxs],
    dst: &mut [f32],
    parallel: bool,
) -> Result<()> {
    let blocks_per_row = crate::raw_block::validate_matmul_t(mkn, lhs, rhs, dst)?;
    if mkn.0 == 0 || mkn.2 == 0 || try_avx2_matmul(mkn, blocks_per_row, lhs, rhs, dst, parallel) {
        return Ok(());
    }
    if parallel {
        crate::raw_block::matmul_t(mkn, lhs, rhs, dst)
    } else {
        crate::raw_block::matmul_t_serial(mkn, lhs, rhs, dst)
    }
}

/// The fused AVX2 kernel for every row, at the AVX2 level only.  Returns
/// `true` if it ran; the kernel writes each dst element exactly once, so no
/// zero-fill is needed.
fn try_avx2_matmul(
    mkn: (usize, usize, usize),
    blocks_per_row: usize,
    lhs: &[f32],
    rhs: &[BlockIq2Xxs],
    dst: &mut [f32],
    parallel: bool,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    if crate::simd::simd_level() == crate::simd::SimdLevel::Avx2 {
        let (m, k, n) = mkn;
        let dst_ptr = crate::simd::DstPtr::new(dst);
        let worker = |row: usize| {
            // SAFETY: `simd_level` only reports levels this CPU supports;
            // row `row` writes exactly dst[i*n + row] for i in 0..m,
            // disjoint from every other row (see `crate::simd`).
            unsafe { matmul_row_avx2(m, k, n, lhs, rhs, blocks_per_row, row, &dst_ptr) }
        };
        if parallel {
            crate::simd::for_each_row(n, worker);
        } else {
            (0..n).for_each(worker);
        }
        return true;
    }
    let _ = (mkn, blocks_per_row, lhs, rhs, dst, parallel);
    false
}

// ─── AVX2/FMA fused dequant+dot kernel ────────────────────────────────────

/// Grid values expanded to f32 lanes (8 values per code), for the AVX2
/// kernel.  Mirrors `IQ2XXS_GRID` (one u8 value per byte) as directly
/// loadable 8-lane vectors.
#[cfg(target_arch = "x86_64")]
const IQ2XXS_GRID_F32: [[f32; 8]; 256] = build_grid_f32();

#[cfg(target_arch = "x86_64")]
const fn build_grid_f32() -> [[f32; 8]; 256] {
    let mut out = [[0.0f32; 8]; 256];
    let mut i = 0;
    while i < 256 {
        let bytes = IQ2XXS_GRID[i].to_le_bytes();
        let mut j = 0;
        while j < 8 {
            out[i][j] = bytes[j] as f32;
            j += 1;
        }
        i += 1;
    }
    out
}

/// Sign patterns expanded to ±1.0 f32 lanes, for the AVX2 kernel.  Mirrors
/// `KSIGNS_IQ2XS` (one bit per value) as directly loadable 8-lane vectors.
#[cfg(target_arch = "x86_64")]
const IQ2XXS_SIGNS_F32: [[f32; 8]; 128] = build_signs_f32();

#[cfg(target_arch = "x86_64")]
const fn build_signs_f32() -> [[f32; 8]; 128] {
    let mut out = [[1.0f32; 8]; 128];
    let mut i = 0;
    while i < 128 {
        let b = KSIGNS_IQ2XS[i];
        let mut j = 0;
        while j < 8 {
            if b & KMASK_IQ2XS[j] != 0 {
                out[i][j] = -1.0;
            }
            j += 1;
        }
        i += 1;
    }
    out
}

/// Fused IQ2_XXS dequant+dot for one weight row, 4 lhs rows at a time.
///
/// Each 32-value sub-group decodes straight into four 8-lane vectors
/// (`grid × signs × db`) and is FMA-accumulated against the activations —
/// no intermediate f32 buffer, and the decode cost is amortized across the
/// m-tile.  Results differ from the scalar path by at most FMA rounding
/// (≤ 1 ulp per element, far below the 1e-4 tolerances the tests use).
///
/// # Safety
/// The caller must have verified `avx2_fma_available()` (this kernel uses
/// AVX2+FMA instructions) and must only hand out disjoint rows per [`DstPtr`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)] // (m, k, n) is the matmul shape; kept flat for the hot loop
unsafe fn matmul_row_avx2(
    m: usize,
    k: usize,
    n: usize,
    lhs: &[f32],
    rhs: &[BlockIq2Xxs],
    blocks_per_row: usize,
    row: usize,
    dst: &crate::simd::DstPtr,
) {
    use std::arch::x86_64::*;

    const MTILE: usize = 4;
    let row_blocks = &rhs[row * blocks_per_row..(row + 1) * blocks_per_row];
    let mut m0 = 0;
    while m0 < m {
        let mcnt = (m - m0).min(MTILE);
        let mut acc = [_mm256_setzero_ps(); MTILE];
        for (b, block) in row_blocks.iter().enumerate() {
            let d = half::f16::from_le_bytes(block.d).to_f32();
            for ib32 in 0..8usize {
                let base = ib32 * 8;
                let lo = u32::from_le_bytes([
                    block.qs[base],
                    block.qs[base + 1],
                    block.qs[base + 2],
                    block.qs[base + 3],
                ]);
                let hi = u32::from_le_bytes([
                    block.qs[base + 4],
                    block.qs[base + 5],
                    block.qs[base + 6],
                    block.qs[base + 7],
                ]);
                let dbv = _mm256_set1_ps(d * (0.5 + ((hi >> 28) as f32)) * 0.25);
                let lane = ib32 * 32;
                for l in 0..4usize {
                    let code = ((lo >> (8 * l)) & 0xFF) as usize;
                    let si = ((hi >> (7 * l)) & 0x7F) as usize;
                    let g = _mm256_loadu_ps(IQ2XXS_GRID_F32[code].as_ptr());
                    let s = _mm256_loadu_ps(IQ2XXS_SIGNS_F32[si].as_ptr());
                    let v = _mm256_mul_ps(_mm256_mul_ps(g, s), dbv);
                    // `b * QK_IQ2_XXS` is the block's offset within the row
                    // (the scalar worker's `base = b * QK_IQ2_XXS`).
                    let c = b * QK_IQ2_XXS + lane + l * 8;
                    for i in 0..mcnt {
                        let a = _mm256_loadu_ps(lhs[(m0 + i) * k + c..].as_ptr());
                        acc[i] = _mm256_fmadd_ps(a, v, acc[i]);
                    }
                }
            }
        }
        // `mcnt` can be < MTILE at the tail; iterate only the live lanes.
        for (i, acc_i) in acc.iter().enumerate().take(mcnt) {
            // SAFETY: row `row` owns dst[i*n + row] for all i; disjoint per
            // row (see `crate::simd`).
            dst.write((m0 + i) * n + row, crate::simd::hsum256(*acc_i));
        }
        m0 += MTILE;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw_block::dequantize;

    fn blocks_from_bytes(bytes: &[u8]) -> Result<&[BlockIq2Xxs]> {
        crate::raw_block::blocks_from_bytes(bytes)
    }

    /// Golden vectors produced by compiling llama.cpp's own
    /// `dequantize_row_iq2_xxs` (ggml-quants.c, master) with the reference
    /// tables from ggml-common.h and running it over the fixture block bytes.
    /// Guards the port against transcription errors in the tables or the bit
    /// packing.
    #[test]
    fn dequant_matches_llamacpp_reference() {
        let blocks = include_bytes!("../tests/data/iq2xxs_blocks.bin");
        let expected = include_str!("../tests/data/iq2xxs_ref.txt");
        let blocks = blocks_from_bytes(blocks).unwrap();
        let mut out = vec![0f32; blocks.len() * QK_IQ2_XXS];
        dequantize(blocks, &mut out).unwrap();
        for (i, line) in expected.lines().enumerate() {
            let want: f32 = line.trim().parse().unwrap();
            assert!(
                (out[i] - want).abs() <= 1e-6 * want.abs().max(1.0),
                "element {i}: got {} want {}",
                out[i],
                want
            );
        }
    }

    /// The packed layout really is 66 bytes per 256 elements, with the fp16
    /// scale first — the offset math in the loader depends on it.
    #[test]
    fn block_layout_is_66_bytes() {
        assert_eq!(std::mem::size_of::<BlockIq2Xxs>(), BLOCK_BYTES);
        assert_eq!(std::mem::align_of::<BlockIq2Xxs>(), 1);
        assert_eq!(BLOCK_BYTES, 66);
        let b = BlockIq2Xxs {
            d: [0, 0x3c],
            qs: [0; 64],
        }; // 1.0 in fp16
        let mut out = [0f32; QK_IQ2_XXS];
        b.dequantize(&mut out);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "all decoded values must be finite"
        );
    }

    /// `matmul_t` equals a naive f32 GEMM over the dequantised weights.
    #[test]
    fn matmul_matches_naive_f32() {
        let m = 3;
        let k = 512; // 2 blocks per row
        let n = 7;
        let blocks = include_bytes!("../tests/data/iq2xxs_blocks.bin");
        let n_blocks = blocks.len() / BLOCK_BYTES; // 4
                                                   // 4 blocks = 1 row of k=1024; build rhs from the fixture blocks,
                                                   // reusing them cyclically to fill n rows of 2 blocks each.
        let mut rhs: Vec<BlockIq2Xxs> = Vec::new();
        for i in 0..n * (k / QK_IQ2_XXS) {
            let blk = blocks_from_bytes(
                &blocks[(i % n_blocks) * BLOCK_BYTES..(i % n_blocks + 1) * BLOCK_BYTES],
            )
            .unwrap();
            rhs.push(blk[0]);
        }
        let lhs: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7919) % 1000) as f32 / 100.0 - 5.0)
            .collect();
        let mut dst = vec![0f32; m * n];
        matmul_t((m, k, n), &lhs, &rhs, &mut dst).unwrap();

        // Naive reference.
        let mut want = vec![0f32; m * n];
        let mut w = vec![0f32; n * k];
        dequantize(&rhs, &mut w).unwrap();
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for kk in 0..k {
                    acc += lhs[i * k + kk] * w[j * k + kk];
                }
                want[i * n + j] = acc;
            }
        }
        for i in 0..m * n {
            let tol = 1e-4 * want[i].abs().max(1.0);
            assert!(
                (dst[i] - want[i]).abs() <= tol,
                "dst[{i}]={} want {} (tol {tol})",
                dst[i],
                want[i]
            );
        }
    }

    /// Error paths are explicit rather than silently wrong.
    #[test]
    fn matmul_rejects_bad_shapes() {
        let blocks = include_bytes!("../tests/data/iq2xxs_blocks.bin");
        let rhs = blocks_from_bytes(blocks).unwrap();
        assert!(matmul_t((1, 100, 1), &[0.0; 100], rhs, &mut [0.0]).is_err()); // k not multiple
        assert!(matmul_t((1, 256, 1), &[0.0; 256], rhs, &mut [0.0]).is_err()); // too few blocks
        assert!(matmul_t((2, 256, 1), &[0.0; 256], rhs, &mut [0.0; 2]).is_err());
        // lhs too short
    }

    /// Build a reusable `rhs` of `n` weight rows × `k` columns from the
    /// fixture blocks (cycled, as the naive-GEMM test does).
    fn fixture_rhs(k: usize, n: usize) -> Vec<BlockIq2Xxs> {
        let blocks = include_bytes!("../tests/data/iq2xxs_blocks.bin");
        let n_blocks = blocks.len() / BLOCK_BYTES; // 4
        let mut rhs = Vec::new();
        for i in 0..n * (k / QK_IQ2_XXS) {
            let blk = blocks_from_bytes(
                &blocks[(i % n_blocks) * BLOCK_BYTES..(i % n_blocks + 1) * BLOCK_BYTES],
            )
            .unwrap();
            rhs.push(blk[0]);
        }
        rhs
    }

    /// Every fused kernel and the scalar worker must agree on the same
    /// weights/activations: each fused decode is checked against the
    /// reference dequant (with 11 activation rows, a full and a partial m-tile
    /// on every kernel).
    #[test]
    fn simd_and_scalar_paths_agree() {
        let (m, k, n) = (11, 1024, 11);
        let rhs = fixture_rhs(k, n);
        let lhs: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7919) % 1000) as f32 / 100.0 - 5.0)
            .collect();

        // Scalar worker, forced.
        let mut scalar = vec![0f32; m * n];
        let dstp = crate::simd::DstPtr::new(&mut scalar);
        for row in 0..n {
            crate::raw_block::matmul_row_scalar((m, k, n), &lhs, &rhs, k / QK_IQ2_XXS, row, &dstp);
        }

        // The AVX2 kernel, plus the dispatched matmul (AVX-512's fused worker
        // is checked against the portable one in `avx512_decode_matches_reference`).
        type RowKernel = unsafe fn(usize, usize, usize, &[f32], &[BlockIq2Xxs], usize, usize, &crate::simd::DstPtr);
        let mut kernels: Vec<(&str, RowKernel)> = Vec::new();
        #[cfg(target_arch = "x86_64")]
        {
            if crate::simd::avx2_fma_available() {
                kernels.push(("avx2", matmul_row_avx2));
            }
        }
        let mut results = Vec::new();
        for (name, kernel) in kernels {
            let mut out = vec![0f32; m * n];
            let dstp = crate::simd::DstPtr::new(&mut out);
            for row in 0..n {
                // SAFETY: the kernel's ISA was checked above; rows are disjoint.
                unsafe { kernel(m, k, n, &lhs, &rhs, k / QK_IQ2_XXS, row, &dstp) };
            }
            results.push((name, out));
        }
        let mut dispatched = vec![0f32; m * n];
        matmul_t((m, k, n), &lhs, &rhs, &mut dispatched).unwrap();
        results.push(("dispatched", dispatched));

        for (name, fast) in &results {
            for i in 0..m * n {
                let tol = 1e-4 * scalar[i].abs().max(1.0);
                assert!(
                    (fast[i] - scalar[i]).abs() <= tol,
                    "{name}: dst[{i}] fast={} scalar={} (tol {tol})",
                    fast[i],
                    scalar[i]
                );
            }
        }
    }

    /// The AVX-512 decode reproduces candle's reference dequant exactly.
    #[test]
    fn avx512_decode_matches_reference() {
        crate::raw_block::testing::check_decode_avx512(&fixture_rhs(1024, 12));
    }

    /// Parallel and serial execution must agree bit-for-bit: rows are
    /// independent and each is computed with identical per-element
    /// operations, so the thread count must not change a single bit.
    #[test]
    fn threaded_matches_serial_bit_exact() {
        let (m, k, n) = (4, 1024, 23);
        let rhs = fixture_rhs(k, n);
        let lhs: Vec<f32> = (0..m * k)
            .map(|i| ((i * 104729) % 1000) as f32 / 100.0 - 5.0)
            .collect();

        let mut par = vec![0f32; m * n];
        matmul_t((m, k, n), &lhs, &rhs, &mut par).unwrap();

        let mut ser = vec![0f32; m * n];
        matmul_t_dispatch((m, k, n), &lhs, &rhs, &mut ser, false).unwrap();

        assert_eq!(par, ser, "parallel and serial matmul must be bit-identical");
    }

    /// Dequantised values respect the format's bounds: |v| <= d * 15.5 * 43/4.
    #[test]
    fn dequant_is_bounded() {
        let mut b = BlockIq2Xxs {
            d: [0, 0x3c],
            qs: [0; 64],
        }; // d = 1.0
           // Highest scale index (15) and max grid value (43).
        b.qs = [0xff; 64];
        let mut out = [0f32; QK_IQ2_XXS];
        b.dequantize(&mut out);
        for v in out {
            assert!(v.abs() <= 43.0 * 0.25 * 15.5 + 1e-6, "unbounded value {v}");
        }
    }
}
