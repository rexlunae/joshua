#![cfg(feature = "vulkan")]
//! Parity check: the Vulkan IQ2_XXS `k_qgemv` (my added `dequant_sub`
//! branch + trellis tables) must match the CPU reference decode of the same
//! quantized blocks. The reference uses candle-core's own
//! `BlockIq2Xxs::dequantize`, so any mismatch is a Vulkan-kernel bug.

use candle_core::backend::{BackendDevice, BackendStorage};
use candle_core::quantized::iq2xxs::{BlockIq2Xxs, QK_IQ2_XXS};
use candle_core::quantized::GgmlDType;
use candle_core::vulkan_backend::kernels::{self, Qgemv};
use candle_core::{DType, VulkanDevice};

const BLOCK_BYTES: usize = std::mem::size_of::<BlockIq2Xxs>();

fn block_from(chunk: &[u8]) -> BlockIq2Xxs {
    let mut d = [0u8; 2];
    d.copy_from_slice(&chunk[..2]);
    let mut qs = [0u8; 64];
    qs.copy_from_slice(&chunk[2..2 + 64]);
    BlockIq2Xxs { d, qs }
}

#[test]
fn vulkan_iq2xxs_gemv_matches_cpu_reference() {
    let dev = VulkanDevice::new(0).expect("Vulkan device 0 required");
    let fixture = include_bytes!("data/iq2xxs_blocks.bin");
    let n_fixture = fixture.len() / BLOCK_BYTES;
    assert!(n_fixture >= 1, "need at least one IQ2_XXS fixture block");

    let m = 1usize;
    let n = 5usize;
    let k = 1024usize; // whole number of 256-element blocks
    let blocks_per_row = k / QK_IQ2_XXS;

    // Weight bytes [n, k] built cyclically from the fixture.
    let mut weight_bytes = Vec::with_capacity(n * blocks_per_row * BLOCK_BYTES);
    for j in 0..n {
        for b in 0..blocks_per_row {
            let c = (j * blocks_per_row + b) % n_fixture;
            weight_bytes.extend_from_slice(&fixture[c * BLOCK_BYTES..(c + 1) * BLOCK_BYTES]);
        }
    }
    let mut blocks: Vec<BlockIq2Xxs> = Vec::with_capacity(n * blocks_per_row);
    for chunk in weight_bytes.chunks_exact(BLOCK_BYTES) {
        blocks.push(block_from(chunk));
    }

    // Activations.
    let x: Vec<f32> = (0..k)
        .map(|i| ((i as i64 * 7919) % 1000) as f32 / 100.0 - 5.0)
        .collect();

    // Vulkan device.
    let words: Vec<u32> = weight_bytes
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes([v[0], v[1], v[2], v[3]]))
        .collect();
    let w = dev.storage_from_slice(&words).expect("upload weights");
    let xw: Vec<u32> = x.iter().map(|v| v.to_bits()).collect();
    let xdev = dev.storage_from_slice(&xw).expect("upload activations");

    let out = dev.alloc(DType::F32, m * n).expect("alloc out");
    kernels::run_qgemv_with(
        &dev,
        Qgemv::Generic,
        GgmlDType::Iq2Xxs,
        xdev.buf(),
        w.buf(),
        out.buf(),
        m,
        n,
        k,
        0,
    )
    .expect("vulkan qgemv");
    dev.synchronize().expect("sync");
    let got = out
        .to_cpu_storage()
        .expect("to cpu")
        .as_slice::<f32>()
        .expect("f32")
        .to_vec();

    // CPU reference: dequantize each block, dot with activation.
    for j in 0..n {
        let mut acc = 0.0f32;
        for r in 0..blocks_per_row {
            let mut vals = [0f32; QK_IQ2_XXS];
            blocks[j * blocks_per_row + r].dequantize(&mut vals);
            for t in 0..QK_IQ2_XXS {
                acc += x[r * QK_IQ2_XXS + t] * vals[t];
            }
        }
        let tol = 1e-3 * acc.abs().max(1.0) + 1e-3;
        assert!(
            (got[j] - acc).abs() <= tol,
            "vulkan IQ2_XXS GEMV mismatch col {j}: vulkan={} cpu={acc} (tol {tol})",
            got[j]
        );
    }

    // Multi-row path (m=2): same weights, two activation rows.
    let m2 = 2usize;
    let x2: Vec<f32> = (0..m2 * k)
        .map(|i| ((i as i64 * 8123) % 1000) as f32 / 100.0 - 5.0)
        .collect();
    let x2w: Vec<u32> = x2.iter().map(|v| v.to_bits()).collect();
    let x2dev = dev.storage_from_slice(&x2w).expect("upload x2");
    let out2 = dev.alloc(DType::F32, m2 * n).expect("alloc out2");
    kernels::run_qgemv_with(
        &dev,
        Qgemv::Generic,
        GgmlDType::Iq2Xxs,
        x2dev.buf(),
        w.buf(),
        out2.buf(),
        m2,
        n,
        k,
        0,
    )
    .expect("vulkan qgemv 2-row");
    dev.synchronize().expect("sync2");
    let got2 = out2
        .to_cpu_storage()
        .expect("to cpu2")
        .as_slice::<f32>()
        .expect("f32 2")
        .to_vec();
    for mm in 0..m2 {
        for j in 0..n {
            let mut acc = 0.0f32;
            for r in 0..blocks_per_row {
                let mut vals = [0f32; QK_IQ2_XXS];
                blocks[j * blocks_per_row + r].dequantize(&mut vals);
                for t in 0..QK_IQ2_XXS {
                    acc += x2[mm * k + r * QK_IQ2_XXS + t] * vals[t];
                }
            }
            let gotv = got2[mm * n + j];
            let tol = 1e-3 * acc.abs().max(1.0) + 1e-3;
            assert!(
                (gotv - acc).abs() <= tol,
                "vulkan IQ2_XXS 2-row GEMV mismatch [{mm},{j}]: vulkan={gotv} cpu={acc} (tol {tol})"
            );
        }
    }
}
