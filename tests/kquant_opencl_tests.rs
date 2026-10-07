//! The k-quant weights (Q2_K..Q6_K) on an OpenCL device: a resident
//! `QMatMul` must reproduce the f32 product with the CPU-dequantized weight
//! for the decode GEMV (one row), the GEMV a speculative verify or short
//! prompt chunk takes (a few rows), and the prefill dequantize-then-GEMM path
//! (many rows).  Q2_K, Q4_K, Q5_K and Q6_K take `k_qgemv_mr` for the GEMV
//! rows, each with its own lane-level decoder; Q3_K takes the one-row
//! `k_qgemv` (see `qgemv_multirow`).  Every format is checked at a shape
//! whose rows span several 256-element blocks.  Skips when no OpenCL device
//! (or runtime) is available.

#![cfg(feature = "opencl")]

use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{Device, Module, Tensor};

fn device() -> Option<Device> {
    static D: std::sync::OnceLock<Option<Device>> = std::sync::OnceLock::new();
    D.get_or_init(|| {
        let dev = candle_core::OpenClDevice::new(0).ok()?;
        Some(Device::OpenCl(dev))
    })
    .clone()
}

fn pseudo_random(len: usize, mut seed: u64) -> Vec<f32> {
    (0..len)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Worst error relative to the larger of the element and the output's mean
/// magnitude (a dot product that cancels to near zero differs between two
/// f32 accumulation orders by the rounding noise of its terms).
fn worst_rel(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let mean_abs = want.iter().map(|w| w.abs()).sum::<f32>() / want.len().max(1) as f32;
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs() / w.abs().max(mean_abs).max(1e-6))
        .fold(0.0f32, f32::max)
}

#[test]
fn opencl_kquant_qmatmul_matches_dequantized_reference() {
    let Some(dev) = device() else {
        eprintln!("no OpenCL device; skipping");
        return;
    };
    // Under `JOSHUA_OPENCL_NATIVE=0` the op runs candle's CPU reference,
    // which quantizes the activations to Q8_K first (a few percent).
    let tol = if candle_core::opencl_backend::native_enabled() {
        1e-4
    } else {
        5e-2
    };
    let (n, k) = (96usize, 1024usize);
    let w = Tensor::from_vec(pseudo_random(n * k, 7), (n, k), &Device::Cpu).unwrap();
    for dtype in [
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
    ] {
        let q = QTensor::quantize(&w, dtype).unwrap();
        let wd = q.dequantize(&Device::Cpu).unwrap();
        let storage = QStorage::from_data(q.data().unwrap(), &dev, dtype).unwrap();
        let mm = QMatMul::from_qtensor(QTensor::new(storage, q.shape().clone()).unwrap()).unwrap();
        for m in [1usize, 3, 16, 40] {
            let x = Tensor::from_vec(pseudo_random(m * k, 11 + m as u64), (m, k), &Device::Cpu)
                .unwrap();
            let want: Vec<f32> = x
                .matmul(&wd.t().unwrap())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let got: Vec<f32> = mm
                .forward(&x.to_device(&dev).unwrap())
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert!(
                got.iter().all(|v| v.is_finite()),
                "{dtype:?} m={m}: non-finite output"
            );
            let err = worst_rel(&got, &want);
            assert!(
                err < tol,
                "{dtype:?} m={m}: worst relative error {err:.2e} >= {tol:.0e}"
            );
        }
    }
}

/// Resident-weight decode timing for every k-quant at a dense-projection
/// shape ([4096, 4096], one row): weights uploaded once, the activation on
/// the device, 30 launches with one sync — what a decode step pays per
/// projection once the dense set lives on the device.  Prints GFLOPS so a
/// run on real hardware (an Arc B50) shows each format's kernel cost; run it
/// again under `JOSHUA_OPENCL_QGEMV=v1` for the one-row kernel's figures.
#[test]
fn opencl_kquant_decode_timing() {
    let Some(dev) = device() else {
        eprintln!("no OpenCL device; skipping");
        return;
    };
    let (n, k, iters) = (4096usize, 4096usize, 30usize);
    let w = Tensor::from_vec(pseudo_random(n * k, 3), (n, k), &Device::Cpu).unwrap();
    let x = Tensor::from_vec(pseudo_random(k, 5), (1, k), &Device::Cpu)
        .unwrap()
        .to_device(&dev)
        .unwrap();
    for dtype in [
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
    ] {
        let q = QTensor::quantize(&w, dtype).unwrap();
        let storage = QStorage::from_data(q.data().unwrap(), &dev, dtype).unwrap();
        let mm = QMatMul::from_qtensor(QTensor::new(storage, q.shape().clone()).unwrap()).unwrap();
        mm.forward(&x).unwrap();
        dev.synchronize().unwrap();
        let start = std::time::Instant::now();
        let mut last = None;
        for _ in 0..iters {
            last = Some(mm.forward(&x).unwrap());
        }
        dev.synchronize().unwrap();
        drop(last);
        let secs = start.elapsed().as_secs_f64() / iters as f64;
        eprintln!(
            "{dtype:?} [{n}, {k}] m=1: {:.3} ms ({:.1} GFLOPS)",
            secs * 1e3,
            2.0 * (n * k) as f64 / secs / 1e9
        );
    }
}
