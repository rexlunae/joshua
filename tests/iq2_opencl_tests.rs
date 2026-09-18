//! IQ2_XXS on an OpenCL device: `QMatMul` over `QStorage::from_data(..,
//! GgmlDType::Iq2Xxs)` (the form the deepseek4 expert cache uploads) must
//! reproduce joshua's fused CPU kernel (`iq2xxs::matmul_t`, the host expert
//! path) at the real expert shapes, for the decode GEMV and the prefill
//! dequantize-then-GEMM paths alike.  Prints resident-weight timings so a run
//! on real hardware (an Arc B50) shows the per-matmul cost next to the CPU.
//! Skips when no OpenCL device (or runtime) is available.

#![cfg(feature = "opencl")]

use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{Device, Module, Tensor};
use half::f16;
use joshua::iq2xxs::{matmul_t, BlockIq2Xxs, BLOCK_BYTES, QK_IQ2_XXS};

fn device() -> Option<Device> {
    static D: std::sync::OnceLock<Option<Device>> = std::sync::OnceLock::new();
    D.get_or_init(|| {
        let dev = candle_core::OpenClDevice::new(0).ok()?;
        Some(Device::OpenCl(dev))
    })
    .clone()
}

/// `n * k / 256` blocks with pseudo-random codes and signs and a *finite*
/// fp16 scale (random scale bits would make NaN/Inf blocks, which say nothing
/// about parity).
fn blocks_for(k: usize, n: usize) -> Vec<BlockIq2Xxs> {
    let n_blocks = n * k / QK_IQ2_XXS;
    let mut seed: u64 = 0xdead_beef_cafe_f00d;
    let mut rnd = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    (0..n_blocks)
        .map(|_| {
            let mut qs = [0u8; 64];
            for c in qs.chunks_exact_mut(4) {
                c.copy_from_slice(&rnd().to_le_bytes());
            }
            let scale = f16::from_f32(1e-3 + (rnd() % 1000) as f32 * 1e-5);
            BlockIq2Xxs {
                d: scale.to_le_bytes(),
                qs,
            }
        })
        .collect()
}

fn block_bytes(blocks: &[BlockIq2Xxs]) -> Vec<u8> {
    let mut out = Vec::with_capacity(blocks.len() * BLOCK_BYTES);
    for b in blocks {
        out.extend_from_slice(&b.d);
        out.extend_from_slice(&b.qs);
    }
    out
}

/// Worst error relative to the larger of the element and the output's mean
/// magnitude: a 4096-term dot product that cancels to near zero differs
/// between two f32 accumulation orders by the rounding noise of its *terms*,
/// so a per-element relative error there measures nothing about the decode.
fn worst_rel(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let mean_abs = want.iter().map(|w| w.abs()).sum::<f32>() / want.len().max(1) as f32;
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs() / w.abs().max(mean_abs).max(1e-6))
        .fold(0.0f32, f32::max)
}

/// The uploaded IQ2_XXS weight agrees with the fused CPU kernel at the real
/// expert shapes (gate/up: [2048, 4096]) for one row (decode), a few rows
/// (the fused GEMV path) and many rows (the dequantize-then-GEMM path), and
/// every device result is finite.
#[test]
fn opencl_iq2xxs_qmatmul_matches_cpu_kernel() {
    let Some(dev) = device() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    let cpu = Device::Cpu;
    for (k, n) in [(4096usize, 2048usize), (2048, 1024), (256, 64)] {
        let blocks = blocks_for(k, n);
        let bytes = block_bytes(&blocks);
        let q = QTensor::new(
            QStorage::from_data(std::borrow::Cow::Borrowed(&bytes), &dev, GgmlDType::Iq2Xxs)
                .unwrap(),
            (n, k),
        )
        .unwrap();
        assert_eq!(q.dtype(), GgmlDType::Iq2Xxs);
        let mm = QMatMul::from_qtensor(q).unwrap();
        for m in [1usize, 6, 16, 17, 64] {
            let x: Vec<f32> = (0..m * k)
                .map(|i| ((i % 29) as f32 - 14.0) * 0.03)
                .collect();
            let mut want = vec![0f32; m * n];
            matmul_t((m, k, n), &x, &blocks, &mut want).unwrap();
            let xs = Tensor::from_vec(x, (m, k), &cpu)
                .unwrap()
                .to_device(&dev)
                .unwrap();
            let got: Vec<f32> = mm
                .forward(&xs)
                .unwrap()
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert!(
                got.iter().all(|v| v.is_finite()),
                "({m},{k},{n}): non-finite device output"
            );
            let worst = worst_rel(&got, &want);
            eprintln!("iq2xxs ({m},{k},{n}): worst rel diff = {worst:.3e}");
            assert!(
                worst < 1e-3,
                "({m},{k},{n}) parity failed: worst={worst:.3e}"
            );
        }
    }
}

/// Resident-weight timing on the decode shape: weights uploaded once, the
/// activation already on the device, launches batched with one sync — the
/// per-visit cost the expert cache pays on a hit — next to the CPU kernel.
#[test]
fn opencl_iq2xxs_resident_timing() {
    let Some(dev) = device() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    let cpu = Device::Cpu;
    for (m, k, n) in [
        (1usize, 4096usize, 2048usize),
        (6, 4096, 2048),
        (12, 4096, 2048),
    ] {
        let blocks = blocks_for(k, n);
        let bytes = block_bytes(&blocks);
        let upload = std::time::Instant::now();
        let q = QTensor::new(
            QStorage::from_data_transfer(
                std::borrow::Cow::Borrowed(&bytes),
                &dev,
                GgmlDType::Iq2Xxs,
            )
            .unwrap(),
            (n, k),
        )
        .unwrap();
        let upload_ms = upload.elapsed().as_secs_f64() * 1e3;
        let mm = QMatMul::from_qtensor(q).unwrap();
        let x: Vec<f32> = (0..m * k)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.02)
            .collect();
        let xs = Tensor::from_vec(x.clone(), (m, k), &cpu)
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let mut want = vec![0f32; m * n];
        matmul_t((m, k, n), &x, &blocks, &mut want).unwrap();
        let got: Vec<f32> = mm
            .forward(&xs)
            .unwrap()
            .to_device(&cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(worst_rel(&got, &want) < 1e-3);

        let cpu_start = std::time::Instant::now();
        for _ in 0..30 {
            let mut c2 = vec![0.0f32; m * n];
            matmul_t((m, k, n), &x, &blocks, &mut c2).unwrap();
        }
        let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1e3 / 30.0;
        let gpu_start = std::time::Instant::now();
        let mut last = None;
        for _ in 0..30 {
            last = Some(mm.forward(&xs).unwrap());
        }
        dev.synchronize().unwrap();
        drop(last);
        let gpu_ms = gpu_start.elapsed().as_secs_f64() * 1e3 / 30.0;
        eprintln!(
            "resident iq2xxs ({m},{k},{n}): upload {upload_ms:.3} ms ({:.1} MB); per matmul cpu {cpu_ms:.3} ms  gpu {gpu_ms:.3} ms  ratio {:.2}x",
            bytes.len() as f64 / 1e6,
            cpu_ms / gpu_ms.max(1e-9)
        );
    }
}
