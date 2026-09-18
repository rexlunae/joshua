//! IQ2_XXS (ggml type 16): 2.0625-bit trellis blocks of 256 elements, the
//! format DeepSeek-V4-Flash GGUFs use for their routed expert gate/up
//! projections.
//!
//! Layout (llama.cpp `block_iq2_xxs`): a 16-bit fp16 scale followed by 32
//! little-endian `u16` codes — 66 bytes.  Each 32-element group is described
//! by two 32-bit words: the low word holds 4 codes of 8 values indexing the
//! 256-entry [`IQ2XXS_GRID`] codebook, the high word packs 4×7 sign-pattern
//! indices into [`KSIGNS_IQ2XS`] plus a 4-bit group scale.
//!
//! This module supplies the dtype's CPU reference form so `QTensor` /
//! `QStorage` can carry IQ2_XXS blocks on every backend: [`to_float`]
//! decodes, `vec_dot` is the plain decode-and-dot reference (the fused
//! decode kernels — AVX2 on the host, `k_qgemv` on OpenCL — live in the
//! consumers), and `from_float` refuses: IQ2_XXS is an importance-matrix
//! quantiser and a float-only encoder would silently produce a different
//! model.  `QTensor::quantize` rejects it before reaching here.
//!
//! [`to_float`]: GgmlType::to_float

use super::k_quants::{BlockQ8K, GgmlType, QK_K};
use super::GgmlDType;
use half::f16;

/// Elements per block.
pub const QK_IQ2_XXS: usize = QK_K;
/// Bytes per block: fp16 scale + 32 codes.
pub const BLOCK_BYTES: usize = 2 + 32 * 2;

/// A single IQ2_XXS block.  Packed so it can be cut out of a memory mapping
/// at any offset; fields are read via `from_le_bytes` so no alignment or
/// endianness assumptions hide here.
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct BlockIq2Xxs {
    /// fp16 block scale.
    pub d: [u8; 2],
    /// 32 uint16 codes (little-endian).
    pub qs: [u8; 64],
}
const _: () = assert!(std::mem::size_of::<BlockIq2Xxs>() == BLOCK_BYTES);

impl BlockIq2Xxs {
    /// Decode this block into `out`, following `dequantize_row_iq2_xxs`
    /// from llama.cpp.
    pub fn dequantize(&self, out: &mut [f32; QK_IQ2_XXS]) {
        let d = f16::from_le_bytes(self.d).to_f32();
        for ib32 in 0..QK_IQ2_XXS / 32 {
            let base = ib32 * 8;
            let lo = u16::from_le_bytes([self.qs[base], self.qs[base + 1]]);
            let hi = u16::from_le_bytes([self.qs[base + 2], self.qs[base + 3]]);
            let lo = u32::from(lo) | (u32::from(hi) << 16);
            let hi = u16::from_le_bytes([self.qs[base + 4], self.qs[base + 5]]);
            let hi2 = u16::from_le_bytes([self.qs[base + 6], self.qs[base + 7]]);
            let hi = u32::from(hi) | (u32::from(hi2) << 16);
            // Group scale: top 4 bits of the high word.
            let db = d * (0.5f32 + ((hi >> 28) as f32)) * 0.25;
            let codes = lo.to_le_bytes();
            for l in 0..4 {
                let grid = IQ2XXS_GRID[codes[l] as usize].to_le_bytes();
                let signs = KSIGNS_IQ2XS[((hi >> (7 * l)) & 127) as usize];
                let dst = &mut out[ib32 * 32 + l * 8..ib32 * 32 + l * 8 + 8];
                for j in 0..8 {
                    let s = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    dst[j] = db * grid[j] as f32 * s;
                }
            }
        }
    }
}

impl GgmlType for BlockIq2Xxs {
    const DTYPE: GgmlDType = GgmlDType::Iq2Xxs;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK_K), "dequantize_row_iq2_xxs: {k} is not divisible by {QK_K}");
        let mut buf = [0f32; QK_IQ2_XXS];
        for (x, y) in xs.iter().zip(ys.chunks_exact_mut(QK_IQ2_XXS)) {
            x.dequantize(&mut buf);
            y.copy_from_slice(&buf);
        }
    }

    fn from_float(_xs: &[f32], _ys: &mut [Self]) {
        // Reached only through a direct `QuantizedType::from_float` call:
        // `QTensor::quantize` refuses IQ2_XXS with an error first.
        panic!("IQ2_XXS cannot be quantized from float: it needs an importance matrix (use llama.cpp's quantizer)");
    }

    /// Reference decode-and-dot against Q8_K activations (what
    /// `k_quants::matmul` quantizes the LHS to).  Only a fallback: the
    /// production CPU path is the consumer's fused AVX2 kernel over f32.
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(n.is_multiple_of(QK_K), "vec_dot_iq2xxs_q8k: {n} is not divisible by {QK_K}");
        let mut buf = [0f32; QK_IQ2_XXS];
        let mut sumf = 0f32;
        for (x, y) in xs.iter().zip(ys.iter()) {
            x.dequantize(&mut buf);
            let mut s = 0f32;
            for (v, &q) in buf.iter().zip(y.qs.iter()) {
                s += v * q as f32;
            }
            sumf += s * y.d;
        }
        sumf
    }
}

/// Sign mask per 8-value group (llama.cpp `kmask_iq2xs`).
pub const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// Sign-bit patterns (llama.cpp `ksigns_iq2xs`).
pub const KSIGNS_IQ2XS: [u8; 128] = [
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147, 20, 149,
    150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170,
    43, 172, 45, 46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207, 80, 209, 210, 83, 212,
    85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99, 228, 101, 102, 231, 232,
    105, 106, 235, 108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123,
    252, 125, 126, 255,
];

/// Trellis codebook (llama.cpp `iq2xxs_grid`).  Each entry's 8 bytes are the
/// 8 values (8/25/43) of one 8-element group.
pub const IQ2XXS_GRID: [u64; 256] = [
    0x0808080808080808,
    0x080808080808082b,
    0x0808080808081919,
    0x0808080808082b08,
    0x0808080808082b2b,
    0x0808080808190819,
    0x0808080808191908,
    0x08080808082b0808,
    0x08080808082b082b,
    0x08080808082b2b08,
    0x08080808082b2b2b,
    0x0808080819080819,
    0x0808080819081908,
    0x0808080819190808,
    0x0808080819192b08,
    0x08080808192b0819,
    0x08080808192b1908,
    0x080808082b080808,
    0x080808082b08082b,
    0x080808082b082b2b,
    0x080808082b2b082b,
    0x0808081908080819,
    0x0808081908081908,
    0x0808081908190808,
    0x0808081908191919,
    0x0808081919080808,
    0x080808192b081908,
    0x080808192b192b08,
    0x0808082b08080808,
    0x0808082b0808082b,
    0x0808082b082b082b,
    0x0808082b2b08082b,
    0x0808190808080819,
    0x0808190808081908,
    0x0808190808190808,
    0x08081908082b0819,
    0x08081908082b1908,
    0x0808190819080808,
    0x080819081908082b,
    0x0808190819082b08,
    0x08081908192b0808,
    0x080819082b080819,
    0x080819082b081908,
    0x080819082b190808,
    0x080819082b2b1908,
    0x0808191908080808,
    0x080819190808082b,
    0x0808191908082b08,
    0x08081919082b0808,
    0x080819191908192b,
    0x08081919192b2b19,
    0x080819192b080808,
    0x080819192b190819,
    0x0808192b08082b19,
    0x0808192b08190808,
    0x0808192b19080808,
    0x0808192b2b081908,
    0x0808192b2b2b1908,
    0x08082b0808080808,
    0x08082b0808081919,
    0x08082b0808082b08,
    0x08082b0808191908,
    0x08082b08082b2b08,
    0x08082b0819080819,
    0x08082b0819081908,
    0x08082b0819190808,
    0x08082b081919082b,
    0x08082b082b082b08,
    0x08082b1908081908,
    0x08082b1919080808,
    0x08082b2b0808082b,
    0x08082b2b08191908,
    0x0819080808080819,
    0x0819080808081908,
    0x0819080808190808,
    0x08190808082b0819,
    0x0819080819080808,
    0x08190808192b0808,
    0x081908082b081908,
    0x081908082b190808,
    0x081908082b191919,
    0x0819081908080808,
    0x0819081908082b08,
    0x08190819082b0808,
    0x0819081919190808,
    0x0819081919192b2b,
    0x081908192b080808,
    0x0819082b082b1908,
    0x0819082b19081919,
    0x0819190808080808,
    0x0819190808082b08,
    0x08191908082b0808,
    0x08191908082b1919,
    0x0819190819082b19,
    0x081919082b080808,
    0x0819191908192b08,
    0x08191919192b082b,
    0x0819192b08080808,
    0x0819192b0819192b,
    0x08192b0808080819,
    0x08192b0808081908,
    0x08192b0808190808,
    0x08192b0819080808,
    0x08192b082b080819,
    0x08192b1908080808,
    0x08192b1908081919,
    0x08192b192b2b0808,
    0x08192b2b19190819,
    0x082b080808080808,
    0x082b08080808082b,
    0x082b080808082b2b,
    0x082b080819081908,
    0x082b0808192b0819,
    0x082b08082b080808,
    0x082b08082b08082b,
    0x082b0819082b2b19,
    0x082b081919082b08,
    0x082b082b08080808,
    0x082b082b0808082b,
    0x082b190808080819,
    0x082b190808081908,
    0x082b190808190808,
    0x082b190819080808,
    0x082b19081919192b,
    0x082b191908080808,
    0x082b191919080819,
    0x082b1919192b1908,
    0x082b192b2b190808,
    0x082b2b0808082b08,
    0x082b2b08082b0808,
    0x082b2b082b191908,
    0x082b2b2b19081908,
    0x1908080808080819,
    0x1908080808081908,
    0x1908080808190808,
    0x1908080808192b08,
    0x19080808082b0819,
    0x19080808082b1908,
    0x1908080819080808,
    0x1908080819082b08,
    0x190808081919192b,
    0x19080808192b0808,
    0x190808082b080819,
    0x190808082b081908,
    0x190808082b190808,
    0x1908081908080808,
    0x19080819082b0808,
    0x19080819192b0819,
    0x190808192b080808,
    0x190808192b081919,
    0x1908082b08080819,
    0x1908082b08190808,
    0x1908082b19082b08,
    0x1908082b1919192b,
    0x1908082b192b2b08,
    0x1908190808080808,
    0x1908190808082b08,
    0x19081908082b0808,
    0x190819082b080808,
    0x190819082b192b19,
    0x190819190819082b,
    0x19081919082b1908,
    0x1908192b08080808,
    0x19082b0808080819,
    0x19082b0808081908,
    0x19082b0808190808,
    0x19082b0819080808,
    0x19082b0819081919,
    0x19082b1908080808,
    0x19082b1919192b08,
    0x19082b19192b0819,
    0x19082b192b08082b,
    0x19082b2b19081919,
    0x19082b2b2b190808,
    0x1919080808080808,
    0x1919080808082b08,
    0x1919080808190819,
    0x1919080808192b19,
    0x19190808082b0808,
    0x191908082b080808,
    0x191908082b082b08,
    0x1919081908081908,
    0x191908191908082b,
    0x191908192b2b1908,
    0x1919082b2b190819,
    0x191919082b190808,
    0x191919082b19082b,
    0x1919191908082b2b,
    0x1919192b08080819,
    0x1919192b19191908,
    0x19192b0808080808,
    0x19192b0808190819,
    0x19192b0808192b19,
    0x19192b08192b1908,
    0x19192b1919080808,
    0x19192b2b08082b08,
    0x192b080808081908,
    0x192b080808190808,
    0x192b080819080808,
    0x192b0808192b2b08,
    0x192b081908080808,
    0x192b081919191919,
    0x192b082b08192b08,
    0x192b082b192b0808,
    0x192b190808080808,
    0x192b190808081919,
    0x192b191908190808,
    0x192b19190819082b,
    0x192b19192b081908,
    0x192b2b081908082b,
    0x2b08080808080808,
    0x2b0808080808082b,
    0x2b08080808082b2b,
    0x2b08080819080819,
    0x2b0808082b08082b,
    0x2b08081908081908,
    0x2b08081908192b08,
    0x2b08081919080808,
    0x2b08082b08190819,
    0x2b08190808080819,
    0x2b08190808081908,
    0x2b08190808190808,
    0x2b08190808191919,
    0x2b08190819080808,
    0x2b081908192b0808,
    0x2b08191908080808,
    0x2b0819191908192b,
    0x2b0819192b191908,
    0x2b08192b08082b19,
    0x2b08192b19080808,
    0x2b08192b192b0808,
    0x2b082b080808082b,
    0x2b082b1908081908,
    0x2b082b2b08190819,
    0x2b19080808081908,
    0x2b19080808190808,
    0x2b190808082b1908,
    0x2b19080819080808,
    0x2b1908082b2b0819,
    0x2b1908190819192b,
    0x2b1908192b080808,
    0x2b19082b19081919,
    0x2b19190808080808,
    0x2b191908082b082b,
    0x2b19190819081908,
    0x2b19191919190819,
    0x2b192b082b080819,
    0x2b192b19082b0808,
    0x2b2b08080808082b,
    0x2b2b080819190808,
    0x2b2b08082b081919,
    0x2b2b081908082b19,
    0x2b2b082b08080808,
    0x2b2b190808192b08,
    0x2b2b2b0819190808,
    0x2b2b2b1908081908,
];

/// A block whose codes/signs/scale are pseudo-random but valid (tests).
#[cfg(test)]
pub(crate) fn synthetic_block(seed: u32) -> BlockIq2Xxs {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    let mut qs = [0u8; 64];
    for ib32 in 0..8 {
        let lo: u32 = next();
        // 4 sign indices of 7 bits + a 4-bit scale.
        let signs = next() & 0x0FFF_FFFF;
        let scale = next() & 0xF;
        let hi = signs | (scale << 28);
        qs[ib32 * 8..ib32 * 8 + 4].copy_from_slice(&lo.to_le_bytes());
        qs[ib32 * 8 + 4..ib32 * 8 + 8].copy_from_slice(&hi.to_le_bytes());
    }
    let d = f16::from_f32(0.001 + (next() % 1000) as f32 * 1e-5).to_le_bytes();
    BlockIq2Xxs { d, qs }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_layout_is_66_bytes_and_decodes_bounded_values() {
        assert_eq!(std::mem::size_of::<BlockIq2Xxs>(), 66);
        let b = synthetic_block(7);
        let mut out = [0f32; QK_IQ2_XXS];
        b.dequantize(&mut out);
        let d = f16::from_le_bytes(b.d).to_f32();
        // |value| <= d * (0.5 + 15) * 0.25 * 43.
        let bound = d * 15.5 * 0.25 * 43.0 + 1e-6;
        assert!(out.iter().all(|v| v.abs() <= bound && v.is_finite()));
        assert!(out.iter().any(|v| *v != 0.0));
    }

    #[test]
    fn to_float_matches_per_block_decode() {
        let blocks: Vec<BlockIq2Xxs> = (0..3).map(synthetic_block).collect();
        let mut ys = vec![0f32; 3 * QK_IQ2_XXS];
        BlockIq2Xxs::to_float(&blocks, &mut ys);
        for (i, b) in blocks.iter().enumerate() {
            let mut one = [0f32; QK_IQ2_XXS];
            b.dequantize(&mut one);
            assert_eq!(&ys[i * QK_IQ2_XXS..(i + 1) * QK_IQ2_XXS], &one[..]);
        }
    }

    #[test]
    fn vec_dot_is_the_decoded_dot_with_q8k_activations() {
        let blocks: Vec<BlockIq2Xxs> = (10..12).map(synthetic_block).collect();
        let xs: Vec<f32> = (0..2 * QK_K).map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0).collect();
        let mut q8 = vec![BlockQ8K::zeros(); 2];
        BlockQ8K::from_float(&xs, &mut q8);
        let got = BlockIq2Xxs::vec_dot(2 * QK_K, &blocks, &q8);
        let mut w = vec![0f32; 2 * QK_K];
        BlockIq2Xxs::to_float(&blocks, &mut w);
        // Reference: dot with the *dequantised* Q8_K activations.
        let mut want = 0f32;
        for (b, q) in q8.iter().enumerate() {
            for i in 0..QK_K {
                want += w[b * QK_K + i] * q.d * q.qs[i] as f32;
            }
        }
        assert!((got - want).abs() <= 1e-3 * want.abs().max(1.0), "{got} vs {want}");
    }
}
