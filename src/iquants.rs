//! The i-quants llama.cpp's "small model" GGUFs are built from — IQ1_S,
//! IQ1_M, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL and IQ4_XS (IQ2_XXS lives
//! in [`crate::iq2xxs`]).
//!
//! Candle has no variant for any of them, so like MXFP4 they are decoded by
//! Joshua through [`crate::raw_block`]: borrowed from the mapping (or held
//! as compact blocks) and decoded a block at a time inside an
//! f32-activation matmul.  The layouts and decodes are a direct port of
//! llama.cpp's `ggml-common.h` / `dequantize_row_*` in `ggml-quants.c`, with
//! the codebooks in [`crate::iq_grids`]:
//!
//! * **IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S** — each group of 8 (IQ2) or 4
//!   (IQ3) values is an index into a codebook of unsigned magnitudes, with
//!   one sign bit per value and a 4-bit scale per 16 or 32 values.
//! * **IQ1_S / IQ1_M** — groups of 8 index a codebook of `-1 / 0 / +1`
//!   patterns, shifted by `±1/8` and scaled per 32 (IQ1_S) or 16 (IQ1_M)
//!   values.
//! * **IQ4_NL / IQ4_XS** — 4-bit indices into a fixed non-linear 16-value
//!   table, with an f16 scale per 32 values (IQ4_NL) or a 6-bit sub-scale
//!   per 32 under an f16 super-scale per 256 (IQ4_XS).
//!
//! llama.cpp's CPU kernels dot these against Q8_K-quantized activations;
//! Joshua keeps activations in f32 (the contract of every raw format), so
//! results differ from llama.cpp by that activation rounding only.

use candle_core::quantized::iq2xxs::{KMASK_IQ2XS, KSIGNS_IQ2XS};
use candle_core::quantized::GgmlDType;
use half::f16;

use crate::iq_grids::{IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ3S_GRID, IQ3XXS_GRID};
use crate::raw_block::RawBlock;

/// Super-block size shared by every format here except IQ4_NL.
pub const QK_K: usize = 256;
/// Elements per IQ4_NL block.
pub const QK4_NL: usize = 32;
/// IQ1_S / IQ1_M's offset of every codebook value.
const IQ1_DELTA: f32 = 0.125;
/// The IQ4 non-linear value table (`kvalues_iq4nl`).
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// One IQ2_XS block (2.3125 bpw): f16 scale, 32 codes (9-bit grid index +
/// 7-bit sign pattern), one byte of two 4-bit scales per 32 values.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockIq2Xs {
    pub d: [u8; 2],
    pub qs: [u8; QK_K / 8 * 2],
    pub scales: [u8; QK_K / 32],
}

/// One IQ2_S block (2.5625 bpw): f16 scale, 32 low grid-index bytes then
/// 32 sign bytes, 8 bytes of high index bits, 8 scale bytes.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockIq2S {
    pub d: [u8; 2],
    pub qs: [u8; QK_K / 4],
    pub qh: [u8; QK_K / 32],
    pub scales: [u8; QK_K / 32],
}

/// One IQ3_XXS block (3.0625 bpw): f16 scale, 64 grid indices, then per 32
/// values a u32 of four 7-bit sign patterns and a 4-bit scale.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockIq3Xxs {
    pub d: [u8; 2],
    pub qs: [u8; 3 * QK_K / 8],
}

/// One IQ3_S block (3.4375 bpw): f16 scale, 64 low grid-index bytes, 8
/// bytes of high index bits, 32 sign bytes, 4 bytes of 4-bit scales.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockIq3S {
    pub d: [u8; 2],
    pub qs: [u8; QK_K / 4],
    pub qh: [u8; QK_K / 32],
    pub signs: [u8; QK_K / 8],
    pub scales: [u8; QK_K / 64],
}

/// One IQ1_S block (1.5625 bpw): f16 scale, 32 low grid-index bytes, then
/// per 32 values a u16 of four 3-bit high index parts, a 3-bit scale and
/// the delta's sign.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockIq1S {
    pub d: [u8; 2],
    pub qs: [u8; QK_K / 8],
    pub qh: [u8; QK_K / 32 * 2],
}

/// One IQ1_M block (1.75 bpw): 32 low grid-index bytes, 16 bytes of high
/// index bits and delta signs, then four u16 holding 3-bit scales per 16
/// values with the f16 super-scale spread over their top nibbles.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockIq1M {
    pub qs: [u8; QK_K / 8],
    pub qh: [u8; QK_K / 16],
    pub scales: [u8; QK_K / 32],
}

/// One IQ4_NL block (4.5 bpw): f16 scale, 16 bytes of 4-bit table indices.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockIq4Nl {
    pub d: [u8; 2],
    pub qs: [u8; QK4_NL / 2],
}

/// One IQ4_XS block (4.25 bpw): f16 scale, the high 2 bits of eight 6-bit
/// sub-scales, their low 4 bits, then 128 bytes of 4-bit table indices.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockIq4Xs {
    pub d: [u8; 2],
    pub scales_h: [u8; 2],
    pub scales_l: [u8; QK_K / 64],
    pub qs: [u8; QK_K / 2],
}

const _: () = assert!(std::mem::size_of::<BlockIq2Xs>() == 74);
const _: () = assert!(std::mem::size_of::<BlockIq2S>() == 82);
const _: () = assert!(std::mem::size_of::<BlockIq3Xxs>() == 98);
const _: () = assert!(std::mem::size_of::<BlockIq3S>() == 110);
const _: () = assert!(std::mem::size_of::<BlockIq1S>() == 50);
const _: () = assert!(std::mem::size_of::<BlockIq1M>() == 56);
const _: () = assert!(std::mem::size_of::<BlockIq4Nl>() == 18);
const _: () = assert!(std::mem::size_of::<BlockIq4Xs>() == 136);

fn f16_at(b: [u8; 2]) -> f32 {
    f16::from_le_bytes(b).to_f32()
}

fn u16_at(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[2 * i], b[2 * i + 1]])
}

/// Write `n` codebook bytes of `grid` (little-endian) scaled by `d`, each
/// negated where bit `j` of `signs` is set.
fn signed_grid(out: &mut [f32], grid: u64, n: usize, signs: u8, d: f32) {
    let bytes = grid.to_le_bytes();
    for (j, o) in out[..n].iter_mut().enumerate() {
        let sign = if signs & KMASK_IQ2XS[j] != 0 {
            -1.0
        } else {
            1.0
        };
        *o = d * bytes[j] as f32 * sign;
    }
}

/// Write the 8 values of IQ1 codebook entry `idx`: `d · (grid + delta)`.
fn iq1_group(out: &mut [f32], idx: usize, d: f32, delta: f32) {
    for (o, g) in out[..8].iter_mut().zip(IQ1S_GRID[idx].to_le_bytes()) {
        *o = d * (g as i8 as f32 + delta);
    }
}

macro_rules! raw_block_consts {
    ($id:expr, $qk:expr, $name:expr) => {
        const GGML_TYPE: u32 = $id;
        const QK: usize = $qk;
        const NAME: &'static str = $name;
        const CANDLE_DTYPE: GgmlDType = GgmlDType::Q2K; // placeholder, see trait docs
    };
}

impl RawBlock for BlockIq2Xs {
    raw_block_consts!(17, QK_K, "iq2_xs");

    fn dequantize_block(&self, out: &mut [f32]) {
        let d = f16_at(self.d);
        for ib32 in 0..QK_K / 32 {
            let s = self.scales[ib32];
            let db = [
                d * (0.5 + (s & 0xf) as f32) * 0.25,
                d * (0.5 + (s >> 4) as f32) * 0.25,
            ];
            for l in 0..4 {
                let q = u16_at(&self.qs, 4 * ib32 + l);
                let y = &mut out[32 * ib32 + 8 * l..];
                signed_grid(
                    y,
                    IQ2XS_GRID[(q & 511) as usize],
                    8,
                    KSIGNS_IQ2XS[(q >> 9) as usize],
                    db[l / 2],
                );
            }
        }
    }
}

impl RawBlock for BlockIq2S {
    raw_block_consts!(22, QK_K, "iq2_s");

    fn dequantize_block(&self, out: &mut [f32]) {
        let d = f16_at(self.d);
        let (qs, signs) = self.qs.split_at(QK_K / 8);
        for ib32 in 0..QK_K / 32 {
            let s = self.scales[ib32];
            let db = [
                d * (0.5 + (s & 0xf) as f32) * 0.25,
                d * (0.5 + (s >> 4) as f32) * 0.25,
            ];
            for l in 0..4 {
                let hi = ((self.qh[ib32] as usize) << (8 - 2 * l)) & 0x300;
                let idx = qs[4 * ib32 + l] as usize | hi;
                let y = &mut out[32 * ib32 + 8 * l..];
                signed_grid(y, IQ2S_GRID[idx], 8, signs[4 * ib32 + l], db[l / 2]);
            }
        }
    }
}

impl RawBlock for BlockIq3Xxs {
    raw_block_consts!(18, QK_K, "iq3_xxs");

    fn dequantize_block(&self, out: &mut [f32]) {
        let d = f16_at(self.d);
        let (qs, scales_and_signs) = self.qs.split_at(QK_K / 4);
        for ib32 in 0..QK_K / 32 {
            let a = &scales_and_signs[4 * ib32..4 * ib32 + 4];
            let aux = u32::from_le_bytes([a[0], a[1], a[2], a[3]]);
            let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
            for l in 0..4 {
                let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                let y = &mut out[32 * ib32 + 8 * l..];
                let g1 = IQ3XXS_GRID[qs[8 * ib32 + 2 * l] as usize] as u64;
                let g2 = IQ3XXS_GRID[qs[8 * ib32 + 2 * l + 1] as usize] as u64;
                signed_grid(y, g1 | (g2 << 32), 8, signs, db);
            }
        }
    }
}

impl RawBlock for BlockIq3S {
    raw_block_consts!(21, QK_K, "iq3_s");

    fn dequantize_block(&self, out: &mut [f32]) {
        let d = f16_at(self.d);
        for ib32 in 0..QK_K / 32 {
            let s = self.scales[ib32 / 2];
            let db = d * (1 + 2 * (if ib32 % 2 == 0 { s & 0xf } else { s >> 4 }) as u32) as f32;
            let qh = self.qh[ib32] as usize;
            for l in 0..4 {
                let q = &self.qs[8 * ib32 + 2 * l..];
                let g1 = IQ3S_GRID[q[0] as usize | ((qh << (8 - 2 * l)) & 256)] as u64;
                let g2 = IQ3S_GRID[q[1] as usize | ((qh << (7 - 2 * l)) & 256)] as u64;
                let y = &mut out[32 * ib32 + 8 * l..];
                signed_grid(y, g1 | (g2 << 32), 8, self.signs[4 * ib32 + l], db);
            }
        }
    }
}

impl RawBlock for BlockIq1S {
    raw_block_consts!(19, QK_K, "iq1_s");

    fn dequantize_block(&self, out: &mut [f32]) {
        let d = f16_at(self.d);
        for ib in 0..QK_K / 32 {
            let qh = u16_at(&self.qh, ib);
            let dl = d * (2 * ((qh >> 12) & 7) + 1) as f32;
            let delta = if qh & 0x8000 != 0 {
                -IQ1_DELTA
            } else {
                IQ1_DELTA
            };
            for l in 0..4 {
                let idx = self.qs[4 * ib + l] as usize | ((((qh >> (3 * l)) & 7) as usize) << 8);
                iq1_group(&mut out[32 * ib + 8 * l..], idx, dl, delta);
            }
        }
    }
}

impl RawBlock for BlockIq1M {
    raw_block_consts!(29, QK_K, "iq1_m");

    fn dequantize_block(&self, out: &mut [f32]) {
        let sc: [u16; 4] = std::array::from_fn(|i| u16_at(&self.scales, i));
        let scale =
            (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        let d = f16::from_bits(scale).to_f32();
        for ib in 0..QK_K / 32 {
            let shift = 6 * (ib % 2) as u16;
            let dl1 = d * (2 * ((sc[ib / 2] >> shift) & 7) + 1) as f32;
            let dl2 = d * (2 * ((sc[ib / 2] >> (shift + 3)) & 7) + 1) as f32;
            let qs = &self.qs[4 * ib..];
            let (h0, h1) = (self.qh[2 * ib] as usize, self.qh[2 * ib + 1] as usize);
            let idx = [
                qs[0] as usize | ((h0 << 8) & 0x700),
                qs[1] as usize | ((h0 << 4) & 0x700),
                qs[2] as usize | ((h1 << 8) & 0x700),
                qs[3] as usize | ((h1 << 4) & 0x700),
            ];
            let delta = |h: usize, bit: usize| if h & bit != 0 { -IQ1_DELTA } else { IQ1_DELTA };
            let deltas = [
                delta(h0, 0x08),
                delta(h0, 0x80),
                delta(h1, 0x08),
                delta(h1, 0x80),
            ];
            for l in 0..4 {
                iq1_group(
                    &mut out[32 * ib + 8 * l..],
                    idx[l],
                    if l < 2 { dl1 } else { dl2 },
                    deltas[l],
                );
            }
        }
    }
}

/// Decode 16 bytes of IQ4 nibbles: low nibbles to `out[..16]`, high to
/// `out[16..32]`.
fn iq4_nibbles(out: &mut [f32], qs: &[u8], d: f32) {
    for (j, &q) in qs[..16].iter().enumerate() {
        out[j] = d * KVALUES_IQ4NL[(q & 0xf) as usize] as f32;
        out[j + 16] = d * KVALUES_IQ4NL[(q >> 4) as usize] as f32;
    }
}

/// [`KVALUES_IQ4NL`] as f32, the lookup table of [`iq4_decode_avx512`].
#[cfg(target_arch = "x86_64")]
static KVALUES_IQ4NL_F32: [f32; 16] = {
    let mut t = [0f32; 16];
    let mut i = 0;
    while i < 16 {
        t[i] = KVALUES_IQ4NL[i] as f32;
        i += 1;
    }
    t
};

/// [`iq4_nibbles`] into two 16-lane registers: the 16 bytes widen to
/// lanes, their low and high nibbles each index the value table in one
/// `vpermps`, then one multiply by `d` — the same product as the scalar
/// decode, so the two agree bit for bit.
///
/// # Safety
/// AVX-512F must be available and `qs` must hold at least 16 bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")]
#[inline]
unsafe fn iq4_decode_avx512(qs: &[u8], d: f32) -> [std::arch::x86_64::__m512; 2] {
    use std::arch::x86_64::*;
    debug_assert!(qs.len() >= 16);
    let table = _mm512_loadu_ps(KVALUES_IQ4NL_F32.as_ptr());
    let d = _mm512_set1_ps(d);
    let bytes = _mm512_cvtepu8_epi32(_mm_loadu_si128(qs.as_ptr() as *const __m128i));
    let lo = _mm512_and_si512(bytes, _mm512_set1_epi32(0x0F));
    let hi = _mm512_srli_epi32(bytes, 4);
    [
        _mm512_mul_ps(_mm512_permutexvar_ps(lo, table), d),
        _mm512_mul_ps(_mm512_permutexvar_ps(hi, table), d),
    ]
}

impl RawBlock for BlockIq4Nl {
    raw_block_consts!(20, QK4_NL, "iq4_nl");
    const DECODE_AVX512: bool = true;

    fn dequantize_block(&self, out: &mut [f32]) {
        iq4_nibbles(out, &self.qs, f16_at(self.d));
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")]
    #[inline]
    unsafe fn decode_avx512(&self, _c: usize) -> [std::arch::x86_64::__m512; 2] {
        iq4_decode_avx512(&self.qs, f16_at(self.d))
    }
}

impl BlockIq4Xs {
    /// The scale of 32-value group `ib`: the super-scale times the 6-bit
    /// sub-scale, offset by 32.
    fn group_scale(&self, ib: usize) -> f32 {
        let scales_h = u16::from_le_bytes(self.scales_h);
        let lo = (self.scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf;
        let ls = lo as i32 | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
        f16_at(self.d) * (ls - 32) as f32
    }
}

impl RawBlock for BlockIq4Xs {
    raw_block_consts!(23, QK_K, "iq4_xs");

    const DECODE_AVX512: bool = true;

    fn dequantize_block(&self, out: &mut [f32]) {
        for ib in 0..QK_K / 32 {
            iq4_nibbles(
                &mut out[32 * ib..],
                &self.qs[16 * ib..],
                self.group_scale(ib),
            );
        }
    }

    /// Group `c`'s 32 values: its 16 bytes of nibbles under its scale.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")]
    #[inline]
    unsafe fn decode_avx512(&self, c: usize) -> [std::arch::x86_64::__m512; 2] {
        iq4_decode_avx512(&self.qs[16 * c..16 * c + 16], self.group_scale(c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw_block::testing;

    fn check<B: RawBlock>() {
        testing::check_matmul(&testing::synthetic::<B>(6, 7), 2 * B::QK);
        testing::check_blocks_from_bytes::<B>();
    }

    #[test]
    fn matmuls_match_their_decode() {
        check::<BlockIq2Xs>();
        check::<BlockIq2S>();
        check::<BlockIq3Xxs>();
        check::<BlockIq3S>();
        check::<BlockIq1S>();
        check::<BlockIq1M>();
        check::<BlockIq4Nl>();
        check::<BlockIq4Xs>();
    }

    /// The AVX-512 decodes reproduce the scalar ones exactly, and the fused
    /// kernel matches the portable one.
    #[test]
    fn iq4_avx512_decodes_match_scalar() {
        testing::check_decode_avx512(&testing::synthetic::<BlockIq4Xs>(8, 11));
        testing::check_decode_avx512(&testing::synthetic::<BlockIq4Nl>(64, 11));
    }

    #[test]
    fn iq4_nl_decodes_its_table() {
        let mut b = BlockIq4Nl {
            d: f16::from_f32(0.5).to_le_bytes(),
            qs: [0; 16],
        };
        b.qs[0] = 0xF0; // element 0 → index 0, element 16 → index 15
        let mut out = [0f32; QK4_NL];
        b.dequantize_block(&mut out);
        assert_eq!(out[0], -63.5);
        assert_eq!(out[16], 56.5);
        assert_eq!(out[1], -63.5);
    }
}
