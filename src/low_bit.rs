//! Q1_0 and Q2_0 — the 1-bit and 2-bit block formats Bonsai models ship in
//! (`GGML_TYPE_Q1_0 = 41`, `GGML_TYPE_Q2_0 = 42`).
//!
//! Candle has no variant for either, so like MXFP4 they are decoded by
//! Joshua through [`crate::raw_block`]: borrowed from the mapping and
//! decoded a block at a time inside an f32-activation matmul.  The layouts
//! are llama.cpp's (`ggml-common.h`, `dequantize_row_q{1,2}_0` in
//! `ggml-quants.c`); both are an f16 scale followed by 16 bytes of codes:
//!
//! * **Q1_0** — 128 elements per block; element `j` is bit `j % 8` of byte
//!   `j / 8`, decoding to `+d` when set and `-d` when clear.
//! * **Q2_0** — 64 elements per block; element `j` is the 2-bit field at
//!   bit `(j % 4) * 2` of byte `j / 4`, decoding to `(q - 1) · d`
//!   (`00 → -1, 01 → 0, 10 → +1, 11 → +2`).
//!
//! llama.cpp's CPU kernels dot these against Q8_0-quantized activations;
//! Joshua keeps activations in f32 (the same contract as every other raw
//! format), so results differ from llama.cpp by that activation rounding.

use candle_core::quantized::GgmlDType;
use half::f16;

use crate::raw_block::RawBlock;

/// Elements per Q1_0 block.
pub const QK1_0: usize = 128;
/// Elements per Q2_0 block.
pub const QK2_0: usize = 64;

/// One Q1_0 block: f16 scale, then one sign bit per element.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockQ1_0 {
    pub d: [u8; 2],
    pub qs: [u8; QK1_0 / 8],
}

/// One Q2_0 block: f16 scale, then four 2-bit codes per byte.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BlockQ2_0 {
    pub d: [u8; 2],
    pub qs: [u8; QK2_0 / 4],
}

const _: () = assert!(std::mem::size_of::<BlockQ1_0>() == 18);
const _: () = assert!(std::mem::size_of::<BlockQ2_0>() == 18);

impl RawBlock for BlockQ1_0 {
    const GGML_TYPE: u32 = 41;
    const QK: usize = QK1_0;
    const NAME: &'static str = "q1_0";
    const CANDLE_DTYPE: GgmlDType = GgmlDType::Q2K; // placeholder, see trait docs

    fn dequantize_block(&self, out: &mut [f32]) {
        let d = f16::from_le_bytes(self.d).to_f32();
        for (bits, chunk) in self.qs.iter().zip(out.chunks_exact_mut(8)) {
            for (i, o) in chunk.iter_mut().enumerate() {
                *o = if bits >> i & 1 == 1 { d } else { -d };
            }
        }
    }
}

impl RawBlock for BlockQ2_0 {
    const GGML_TYPE: u32 = 42;
    const QK: usize = QK2_0;
    const NAME: &'static str = "q2_0";
    const CANDLE_DTYPE: GgmlDType = GgmlDType::Q2K; // placeholder, see trait docs

    fn dequantize_block(&self, out: &mut [f32]) {
        let d = f16::from_le_bytes(self.d).to_f32();
        for (byte, chunk) in self.qs.iter().zip(out.chunks_exact_mut(4)) {
            for (i, o) in chunk.iter_mut().enumerate() {
                *o = ((byte >> (2 * i) & 3) as f32 - 1.0) * d;
            }
        }
    }
}

/// Quantize `xs` to Q1_0 the way llama.cpp's `quantize_row_q1_0_ref` does
/// (scale = mean |x|, sign bit per element) — for building test fixtures.
pub fn quantize_q1_0(xs: &[f32]) -> Vec<BlockQ1_0> {
    xs.chunks_exact(QK1_0)
        .map(|x| {
            let d = x.iter().map(|v| v.abs()).sum::<f32>() / QK1_0 as f32;
            let mut qs = [0u8; QK1_0 / 8];
            for (j, v) in x.iter().enumerate() {
                if *v >= 0.0 {
                    qs[j / 8] |= 1 << (j % 8);
                }
            }
            BlockQ1_0 {
                d: f16::from_f32(d).to_le_bytes(),
                qs,
            }
        })
        .collect()
}

/// Quantize `xs` to Q2_0 the way llama.cpp's `quantize_row_q2_0_ref` does
/// (scale = max |x|, codes `round(x / d)` clamped to `-1..=2`) — for
/// building test fixtures.
pub fn quantize_q2_0(xs: &[f32]) -> Vec<BlockQ2_0> {
    xs.chunks_exact(QK2_0)
        .map(|x| {
            let d = x.iter().fold(0f32, |a, v| a.max(v.abs()));
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            let mut qs = [0u8; QK2_0 / 4];
            for (j, v) in x.iter().enumerate() {
                let q = ((v * id).round() as i32 + 1).clamp(0, 3) as u8;
                qs[j / 4] |= q << ((j % 4) * 2);
            }
            BlockQ2_0 {
                d: f16::from_f32(d).to_le_bytes(),
                qs,
            }
        })
        .collect()
}

/// The raw bytes of a run of blocks, as written to a GGUF tensor.
pub fn blocks_as_bytes<B: RawBlock>(blocks: &[B]) -> &[u8] {
    // SAFETY: `RawBlock`s are plain byte structs (align 1, no padding).
    unsafe {
        std::slice::from_raw_parts(blocks.as_ptr() as *const u8, std::mem::size_of_val(blocks))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw_block::{blocks_from_bytes, dequantize, testing};

    /// Bit / field order and the value mapping match llama.cpp's
    /// `dequantize_row_q1_0` / `dequantize_row_q2_0`.
    #[test]
    fn decode_matches_llama_cpp_layout() {
        let mut q1 = BlockQ1_0 {
            d: f16::from_f32(0.5).to_le_bytes(),
            qs: [0; 16],
        };
        q1.qs[0] = 0b0000_0101; // elements 0 and 2 set
        q1.qs[15] = 0x80; // element 127 set
        let mut out = [0f32; QK1_0];
        q1.dequantize_block(&mut out);
        assert_eq!(&out[..4], &[0.5, -0.5, 0.5, -0.5]);
        assert_eq!(out[127], 0.5);
        assert_eq!(out[126], -0.5);

        let mut q2 = BlockQ2_0 {
            d: f16::from_f32(3.0).to_le_bytes(),
            qs: [0x55; 16], // every code 01 → 0
        };
        q2.qs[0] = 0b11_10_01_00; // codes 0,1,2,3 for elements 0..4
        let mut out = [0f32; QK2_0];
        q2.dequantize_block(&mut out);
        assert_eq!(&out[..5], &[-3.0, 0.0, 3.0, 6.0, 0.0]);
    }

    /// Golden output of llama.cpp's own `dequantize_row_q1_0` /
    /// `dequantize_row_q2_0` (ggml-quants.c, compiled unchanged) over two
    /// blocks each of deterministic bytes (`i*37+11` / `i*53+7`, scale
    /// 0.75): 256 Q1_0 values, then 128 Q2_0 values.
    #[test]
    fn decode_matches_llama_cpp_reference() {
        let want: Vec<f32> = include_str!("../tests/data/low_bit_ref.txt")
            .lines()
            .map(|l| l.parse().unwrap())
            .collect();
        let bytes = |mul: usize, add: usize| {
            let mut b: Vec<u8> = (0..36).map(|i| (i * mul + add) as u8).collect();
            for k in 0..2 {
                b[k * 18..k * 18 + 2].copy_from_slice(&f16::from_f32(0.75).to_le_bytes());
            }
            b
        };
        let q1 = crate::raw_block::decode_bytes::<BlockQ1_0>(&bytes(37, 11), 256).unwrap();
        let q2 = crate::raw_block::decode_bytes::<BlockQ2_0>(&bytes(53, 7), 128).unwrap();
        assert_eq!([q1, q2].concat(), want);
    }

    #[test]
    fn quantizers_round_trip_their_grids() {
        // Values already on the Q2_0 grid survive exactly.
        let xs: Vec<f32> = (0..QK2_0).map(|j| ((j % 3) as f32 - 1.0) * 0.5).collect();
        let mut back = vec![0f32; QK2_0];
        dequantize(&quantize_q2_0(&xs), &mut back).unwrap();
        assert_eq!(xs, back);
        // Q1_0 keeps every sign and the mean magnitude.
        let xs: Vec<f32> = (0..QK1_0)
            .map(|j| if j % 3 == 0 { -1.0 } else { 1.0 })
            .collect();
        let mut back = vec![0f32; QK1_0];
        dequantize(&quantize_q1_0(&xs), &mut back).unwrap();
        assert_eq!(xs, back);
    }

    #[test]
    fn bytes_round_trip() {
        let xs: Vec<f32> = (0..2 * QK1_0).map(|j| (j as f32 * 0.37).sin()).collect();
        let blocks = quantize_q1_0(&xs);
        let again: &[BlockQ1_0] = blocks_from_bytes(blocks_as_bytes(&blocks)).unwrap();
        assert_eq!(blocks_as_bytes(again), blocks_as_bytes(&blocks));
        testing::check_blocks_from_bytes::<BlockQ1_0>();
        testing::check_blocks_from_bytes::<BlockQ2_0>();
    }

    #[test]
    fn matmul_matches_dequantized_reference() {
        let w: Vec<f32> = (0..7 * 256)
            .map(|j| ((j * 7919) % 97) as f32 / 48.0 - 1.0)
            .collect();
        testing::check_matmul(&quantize_q1_0(&w), 256);
        testing::check_matmul(&quantize_q2_0(&w), 256);
    }
}
