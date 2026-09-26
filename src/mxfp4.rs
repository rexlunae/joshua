//! MXFP4 (OCP Microscaling FP4) support — `GGML_TYPE_MXFP4`, type id 39.
//!
//! Kimi-K3-class models ship their weights natively in MXFP4, and candle has
//! no notion of the format: [`candle_core::quantized::GgmlDType`] stops at
//! `BF16`/`Q8K`, and its `from_u32` *rejects* type 39 outright, so candle
//! cannot even parse the header of such a GGUF, let alone its tensors.  This
//! module supplies the missing piece: the block layout and an exact decoder;
//! [`crate::raw_block`] provides the rest (bulk decode, the in-mapping
//! matmul, dispatch by dtype id).
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
//!
//! # NVFP4
//!
//! NVIDIA's FP4 (`GGML_TYPE_NVFP4`, id 40) uses the same E2M1 elements with
//! a finer scale: a block covers 64 elements in 36 bytes — four UE4M3 scale
//! bytes (unsigned, 4 exponent bits with bias 7, 3 mantissa bits; one per 16
//! elements) then 32 bytes of codes, packed split-half within each 16.

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

impl crate::raw_block::RawBlock for BlockMxfp4 {
    const GGML_TYPE: u32 = GGML_TYPE_MXFP4;
    const QK: usize = QK_MXFP4;
    const NAME: &'static str = "mxfp4";
    const CANDLE_DTYPE: candle_core::quantized::GgmlDType = candle_core::quantized::GgmlDType::Q2K;
    const DECODE_AVX512: bool = true;

    /// The 16 low nibbles (elements 0..16) and 16 high nibbles (16..32)
    /// each index the E2M1 table in one `vpermps`, then scale.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")]
    #[inline]
    unsafe fn decode_avx512(&self, _c: usize) -> [std::arch::x86_64::__m512; 2] {
        use std::arch::x86_64::*;
        let Some(d) = e8m0_to_f32(self.e) else {
            return [_mm512_setzero_ps(); 2];
        };
        let table = _mm512_loadu_ps(E2M1.as_ptr());
        let d = _mm512_set1_ps(d);
        let bytes = _mm512_cvtepu8_epi32(_mm_loadu_si128(self.qs.as_ptr() as *const __m128i));
        let lo = _mm512_and_si512(bytes, _mm512_set1_epi32(0x0F));
        let hi = _mm512_srli_epi32(bytes, 4);
        [
            _mm512_mul_ps(_mm512_permutexvar_ps(lo, table), d),
            _mm512_mul_ps(_mm512_permutexvar_ps(hi, table), d),
        ]
    }

    fn dequantize_block(&self, out: &mut [f32]) {
        self.dequantize(out.try_into().expect("one MXFP4 block"));
    }
}

/// Elements per NVFP4 block.
pub const QK_NVFP4: usize = 64;
/// Elements per NVFP4 scale.
pub const QK_NVFP4_SUB: usize = 16;

/// One NVFP4 block: a UE4M3 scale per 16 elements, then their E2M1 codes.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockNvfp4 {
    pub d: [u8; QK_NVFP4 / QK_NVFP4_SUB],
    pub qs: [u8; QK_NVFP4 / 2],
}

const _: () = assert!(std::mem::size_of::<BlockNvfp4>() == 36);

/// Decode a UE4M3 scale byte.  The sign bit is ignored (the format is
/// unsigned) and `0x7F`, E4M3's NaN, decodes to zero like llama.cpp's
/// `ggml_ue4m3_to_fp32` (which returns this value halved, to pair with its
/// doubled E2M1 table).
pub fn ue4m3_to_f32(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let exp = ((x >> 3) & 0xF) as i32;
    let man = (x & 0x7) as f32;
    if exp == 0 {
        man * 2f32.powi(-9)
    } else {
        (1.0 + man / 8.0) * 2f32.powi(exp - 7)
    }
}

impl crate::raw_block::RawBlock for BlockNvfp4 {
    const GGML_TYPE: u32 = 40;
    const QK: usize = QK_NVFP4;
    const NAME: &'static str = "nvfp4";
    const CANDLE_DTYPE: candle_core::quantized::GgmlDType = candle_core::quantized::GgmlDType::Q2K;

    fn dequantize_block(&self, out: &mut [f32]) {
        for (s, (y, qs)) in out.chunks_exact_mut(QK_NVFP4_SUB).zip(self.qs.chunks_exact(QK_NVFP4_SUB / 2)).enumerate() {
            let d = ue4m3_to_f32(self.d[s]);
            for (j, &q) in qs.iter().enumerate() {
                y[j] = E2M1[(q & 0x0F) as usize] * d;
                y[j + QK_NVFP4_SUB / 2] = E2M1[(q >> 4) as usize] * d;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvfp4_decodes_ue4m3_scales_per_16() {
        assert_eq!(ue4m3_to_f32(0x38), 1.0); // exp 7, man 0
        assert_eq!(ue4m3_to_f32(0x3C), 1.5);
        assert_eq!(ue4m3_to_f32(0x01), 2f32.powi(-9)); // subnormal
        assert_eq!(ue4m3_to_f32(0x7F), 0.0);
        let mut b = BlockNvfp4 { d: [0x38, 0x40, 0, 0], qs: [0; 32] };
        b.qs[0] = 0x7 | (0x1 << 4); // element 0 → +6, element 8 → +0.5
        b.qs[8] = 0xF; // first element of the second 16 → −6, scale 2
        let mut out = [0f32; QK_NVFP4];
        crate::raw_block::RawBlock::dequantize_block(&b, &mut out);
        assert_eq!((out[0], out[8], out[16]), (6.0, 0.5, -12.0));
        crate::raw_block::testing::check_matmul(&crate::raw_block::testing::synthetic::<BlockNvfp4>(6, 5), 128);
        crate::raw_block::testing::check_blocks_from_bytes::<BlockNvfp4>();
    }

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

    /// A deterministic spread over the code space, `n` rows of `k`.
    fn fixture_rows(n: usize, k: usize) -> Vec<BlockMxfp4> {
        let mut rhs = Vec::new();
        for r in 0..n {
            for b in 0..k / QK_MXFP4 {
                let mut qs = [0u8; 16];
                for (j, q) in qs.iter_mut().enumerate() {
                    *q = (((j + r * 3 + b * 5) % 16) as u8) | ((((j + r) % 16) as u8) << 4);
                }
                rhs.push(BlockMxfp4 {
                    e: 127 + (r as u8 % 3),
                    qs,
                });
            }
        }
        rhs
    }

    #[test]
    fn matmul_matches_an_explicit_dequantized_reference() {
        crate::raw_block::testing::check_matmul(&fixture_rows(23, 1024), 1024);
        let mut rows = fixture_rows(23, 1024);
        rows[5].e = 0xFF; // the NaN scale decodes to zeros on every path
        crate::raw_block::testing::check_decode_avx512(&rows);
    }

    #[test]
    fn blocks_from_bytes_rejects_a_partial_block() {
        assert_eq!(crate::raw_block::block_bytes::<BlockMxfp4>(), 17);
        crate::raw_block::testing::check_blocks_from_bytes::<BlockMxfp4>();
    }
}
