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

#[inline(always)]
fn f16_at(b: [u8; 2]) -> f32 {
    f16::from_le_bytes(b).to_f32()
}

#[inline(always)]
fn u16_at(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[2 * i], b[2 * i + 1]])
}

/// One 32-value group of an IQ2 / IQ3 block, unpacked: four 8-value
/// codebook entries of unsigned magnitudes (little-endian bytes), their
/// sign bytes (bit `j` negates value `j`), and the scale of each 16-value
/// half.  The scalar and AVX-512 decodes both run from it.
struct SignedGroup {
    grids: [u64; 4],
    signs: [u8; 4],
    scale: [f32; 2],
}

impl SignedGroup {
    fn decode(&self, out: &mut [f32]) {
        for l in 0..4 {
            let bytes = self.grids[l].to_le_bytes();
            for (j, o) in out[8 * l..8 * l + 8].iter_mut().enumerate() {
                let sign = if self.signs[l] & KMASK_IQ2XS[j] != 0 {
                    -1.0
                } else {
                    1.0
                };
                *o = self.scale[l / 2] * bytes[j] as f32 * sign;
            }
        }
    }

    /// Each half's two entries widen straight to f32 lanes (`vpmovzxbd`),
    /// their sign bytes form one 16-bit mask that negates in place, then
    /// one multiply by the half's scale — the scalar product up to the
    /// order of an exact sign flip, so bit-identical.
    ///
    /// # Safety
    /// AVX-512F/BW/VL/DQ must be available.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")]
    #[inline]
    unsafe fn decode_avx512(&self) -> [std::arch::x86_64::__m512; 2] {
        use std::arch::x86_64::*;
        let zero = _mm512_setzero_ps();
        let mut w = [zero; 2];
        // (A loop rather than `array::map`: a closure cannot inline these
        // intrinsics.)
        for (h, wh) in w.iter_mut().enumerate() {
            let grid = _mm_set_epi64x(self.grids[2 * h + 1] as i64, self.grids[2 * h] as i64);
            let v = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(grid));
            let mask = self.signs[2 * h] as u16 | (self.signs[2 * h + 1] as u16) << 8;
            *wh = _mm512_mul_ps(
                _mm512_mask_sub_ps(v, mask, zero, v),
                _mm512_set1_ps(self.scale[h]),
            );
        }
        w
    }
}

/// One 32-value group of an IQ1 block, unpacked: four 8-value codebook
/// entries of `-1 / 0 / +1` (as little-endian `i8` bytes), each entry's
/// `±1/8` shift (bit `l` of `neg_delta` set: entry `l` shifts down), and the
/// scale of each 16-value half.  Values are `scale · (grid + delta)`.
struct Iq1Group {
    grids: [u64; 4],
    neg_delta: u8,
    scale: [f32; 2],
}

impl Iq1Group {
    fn decode(&self, out: &mut [f32]) {
        for l in 0..4 {
            let delta = if self.neg_delta >> l & 1 != 0 {
                -IQ1_DELTA
            } else {
                IQ1_DELTA
            };
            for (o, g) in out[8 * l..8 * l + 8]
                .iter_mut()
                .zip(self.grids[l].to_le_bytes())
            {
                *o = self.scale[l / 2] * (g as i8 as f32 + delta);
            }
        }
    }

    /// Sign-extend each half's two entries to f32 lanes, add the per-entry
    /// delta (a lane mask choosing `±1/8` per 8 lanes, kept in registers),
    /// multiply by the scale — the scalar operations in the scalar order.
    ///
    /// # Safety
    /// AVX-512F/BW/VL/DQ must be available.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")]
    #[inline]
    unsafe fn decode_avx512(&self) -> [std::arch::x86_64::__m512; 2] {
        use std::arch::x86_64::*;
        let mut w = [_mm512_setzero_ps(); 2];
        for (h, wh) in w.iter_mut().enumerate() {
            let grid = _mm_set_epi64x(self.grids[2 * h + 1] as i64, self.grids[2 * h] as i64);
            let v = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(grid));
            let bits = (self.neg_delta >> (2 * h)) & 3;
            let lanes = ((bits & 1) as u16 * 0x00FF) | ((bits >> 1) as u16 * 0xFF00);
            let delta =
                _mm512_mask_blend_ps(lanes, _mm512_set1_ps(IQ1_DELTA), _mm512_set1_ps(-IQ1_DELTA));
            *wh = _mm512_mul_ps(_mm512_add_ps(v, delta), _mm512_set1_ps(self.scale[h]));
        }
        w
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

/// `RawBlock` for a format whose 32-value groups unpack through
/// `$Block::group` into a [`SignedGroup`] or [`Iq1Group`]: the scalar
/// decode walks the groups, and the fused AVX-512 kernel decodes group `c`
/// straight into registers.
macro_rules! grouped_raw_block {
    ($Block:ty, $id:expr, $name:expr) => {
        impl RawBlock for $Block {
            raw_block_consts!($id, QK_K, $name);
            const DECODE_AVX512: bool = true;

            fn dequantize_block(&self, out: &mut [f32]) {
                for ib in 0..QK_K / 32 {
                    self.group(ib).decode(&mut out[32 * ib..32 * ib + 32]);
                }
            }

            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")]
            #[inline]
            unsafe fn decode_avx512(&self, c: usize) -> [std::arch::x86_64::__m512; 2] {
                self.group(c).decode_avx512()
            }
        }
    };
}

/// The two IQ2_XS / IQ2_S half scales packed in one byte.
#[inline(always)]
fn iq2_scales(d: f32, s: u8) -> [f32; 2] {
    [
        d * (0.5 + (s & 0xf) as f32) * 0.25,
        d * (0.5 + (s >> 4) as f32) * 0.25,
    ]
}

impl BlockIq2Xs {
    #[inline(always)]
    fn group(&self, ib: usize) -> SignedGroup {
        let q: [u16; 4] = std::array::from_fn(|l| u16_at(&self.qs, 4 * ib + l));
        SignedGroup {
            grids: q.map(|q| IQ2XS_GRID[(q & 511) as usize]),
            signs: q.map(|q| KSIGNS_IQ2XS[(q >> 9) as usize]),
            scale: iq2_scales(f16_at(self.d), self.scales[ib]),
        }
    }
}
grouped_raw_block!(BlockIq2Xs, 17, "iq2_xs");

impl BlockIq2S {
    #[inline(always)]
    fn group(&self, ib: usize) -> SignedGroup {
        let (qs, signs) = self.qs.split_at(QK_K / 8);
        let qh = self.qh[ib] as usize;
        SignedGroup {
            grids: std::array::from_fn(|l| {
                IQ2S_GRID[qs[4 * ib + l] as usize | ((qh << (8 - 2 * l)) & 0x300)]
            }),
            signs: std::array::from_fn(|l| signs[4 * ib + l]),
            scale: iq2_scales(f16_at(self.d), self.scales[ib]),
        }
    }
}
grouped_raw_block!(BlockIq2S, 22, "iq2_s");

impl BlockIq3Xxs {
    #[inline(always)]
    fn group(&self, ib: usize) -> SignedGroup {
        let (qs, scales_and_signs) = self.qs.split_at(QK_K / 4);
        let a = &scales_and_signs[4 * ib..4 * ib + 4];
        let aux = u32::from_le_bytes([a[0], a[1], a[2], a[3]]);
        let db = f16_at(self.d) * (0.5 + (aux >> 28) as f32) * 0.5;
        let q = &qs[8 * ib..8 * ib + 8];
        SignedGroup {
            grids: std::array::from_fn(|l| {
                IQ3XXS_GRID[q[2 * l] as usize] as u64
                    | (IQ3XXS_GRID[q[2 * l + 1] as usize] as u64) << 32
            }),
            signs: std::array::from_fn(|l| KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize]),
            scale: [db; 2],
        }
    }
}
grouped_raw_block!(BlockIq3Xxs, 18, "iq3_xxs");

impl BlockIq3S {
    #[inline(always)]
    fn group(&self, ib: usize) -> SignedGroup {
        let s = self.scales[ib / 2];
        let db = f16_at(self.d)
            * (1 + 2
                * (if ib.is_multiple_of(2) {
                    s & 0xf
                } else {
                    s >> 4
                }) as u32) as f32;
        let qh = self.qh[ib] as usize;
        let q = &self.qs[8 * ib..8 * ib + 8];
        SignedGroup {
            grids: std::array::from_fn(|l| {
                let g1 = IQ3S_GRID[q[2 * l] as usize | ((qh << (8 - 2 * l)) & 256)] as u64;
                let g2 = IQ3S_GRID[q[2 * l + 1] as usize | ((qh << (7 - 2 * l)) & 256)] as u64;
                g1 | (g2 << 32)
            }),
            signs: std::array::from_fn(|l| self.signs[4 * ib + l]),
            scale: [db; 2],
        }
    }
}
grouped_raw_block!(BlockIq3S, 21, "iq3_s");

impl BlockIq1S {
    #[inline(always)]
    fn group(&self, ib: usize) -> Iq1Group {
        let qh = u16_at(&self.qh, ib);
        let dl = f16_at(self.d) * (2 * ((qh >> 12) & 7) + 1) as f32;
        Iq1Group {
            grids: std::array::from_fn(|l| {
                IQ1S_GRID[self.qs[4 * ib + l] as usize | ((((qh >> (3 * l)) & 7) as usize) << 8)]
            }),
            neg_delta: if qh & 0x8000 != 0 { 0xF } else { 0 },
            scale: [dl; 2],
        }
    }
}
grouped_raw_block!(BlockIq1S, 19, "iq1_s");

impl BlockIq1M {
    #[inline(always)]
    fn group(&self, ib: usize) -> Iq1Group {
        let sc: [u16; 4] = std::array::from_fn(|i| u16_at(&self.scales, i));
        let scale =
            (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        let d = f16::from_bits(scale).to_f32();
        let shift = 6 * (ib % 2) as u16;
        let qs = &self.qs[4 * ib..4 * ib + 4];
        let (h0, h1) = (self.qh[2 * ib] as usize, self.qh[2 * ib + 1] as usize);
        Iq1Group {
            grids: [
                IQ1S_GRID[qs[0] as usize | ((h0 << 8) & 0x700)],
                IQ1S_GRID[qs[1] as usize | ((h0 << 4) & 0x700)],
                IQ1S_GRID[qs[2] as usize | ((h1 << 8) & 0x700)],
                IQ1S_GRID[qs[3] as usize | ((h1 << 4) & 0x700)],
            ],
            neg_delta: (h0 >> 3 & 1 | h0 >> 6 & 2 | h1 >> 1 & 4 | h1 >> 4 & 8) as u8,
            scale: [
                d * (2 * ((sc[ib / 2] >> shift) & 7) + 1) as f32,
                d * (2 * ((sc[ib / 2] >> (shift + 3)) & 7) + 1) as f32,
            ],
        }
    }
}
grouped_raw_block!(BlockIq1M, 29, "iq1_m");

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
    #[inline(always)]
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
    fn avx512_decodes_match_scalar() {
        testing::check_decode_avx512(&testing::synthetic::<BlockIq1S>(8, 11));
        testing::check_decode_avx512(&testing::synthetic::<BlockIq1M>(8, 11));
        testing::check_decode_avx512(&testing::synthetic::<BlockIq2Xs>(8, 11));
        testing::check_decode_avx512(&testing::synthetic::<BlockIq2S>(8, 11));
        testing::check_decode_avx512(&testing::synthetic::<BlockIq3Xxs>(8, 11));
        testing::check_decode_avx512(&testing::synthetic::<BlockIq3S>(8, 11));
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
