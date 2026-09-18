//! OpenCL IQ2_XXS matmul parity (runs on an OpenCL host, e.g. the Arc B50 Pro).
//! Validates that joshua::iq2xxs::try_opencl_matmul reproduces the CPU
//! matmul_t reference for decode-shaped matmuls. Skips when no OpenCL device.

#![cfg(feature = "opencl")]

use candle_core::Device;
use joshua::iq2xxs::{matmul_t, try_opencl_matmul, BlockIq2Xxs, QK_IQ2_XXS};

fn device() -> Option<Device> {
    static D: std::sync::OnceLock<Option<Device>> = std::sync::OnceLock::new();
    D.get_or_init(|| {
        let dev = candle_core::OpenClDevice::new(0).ok()?;
        Some(Device::OpenCl(dev))
    })
    .clone()
}

fn blocks_for(dims: (usize, usize, usize)) -> (Vec<BlockIq2Xxs>, usize) {
    let (m, k, n) = dims;
    let bpr = k / QK_IQ2_XXS;
    let n_blocks = n * bpr;
    let mut seed: u64 = 0xdead_beef_cafe_f00d;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as u8
    };
    // 66-byte block: scale(2) + codes(64).  Any bytes decode deterministically.
    let bytes: Vec<u8> = (0..n_blocks * 66).map(|_| rnd()).collect();
    let blocks: Vec<BlockIq2Xxs> = bytes
        .chunks_exact(66)
        .map(|c| BlockIq2Xxs {
            d: [c[0], c[1]],
            qs: c[2..66].try_into().unwrap(),
        })
        .collect();
    let _ = (m, bpr);
    (blocks, n_blocks)
}

#[test]
fn opencl_iq2xxs_matches_cpu() {
    let Some(dev) = device() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    for (m, k, n) in [(1usize, 512usize, 1024usize), (1, 2048, 2048), (4, 4096, 2048)] {
        let (blocks, _) = blocks_for((m, k, n));
        let x: Vec<f32> = (0..m * k).map(|i| ((i % 29) as f32 - 14.0) * 0.03).collect();

        let mut cpu = vec![0.0f32; m * n];
        matmul_t((m, k, n), &x, &blocks, &mut cpu).unwrap();

        let mut gpu = vec![0.0f32; m * n];
        let ran = try_opencl_matmul(&dev, (m, k, n), &x, &blocks, &mut gpu).unwrap();
        assert!(ran, "opencl matmul should run on an OpenCl device");

        let mut worst = 0.0f32;
        for i in 0..m * n {
            let rel = (gpu[i] - cpu[i]).abs() / cpu[i].abs().max(1e-4);
            worst = worst.max(rel);
        }
        eprintln!("shape ({m},{k},{n}): worst rel diff = {worst:.3e}");
        assert!(worst < 2e-3, "({m},{k},{n}) parity failed: worst={worst:.3e}");

        // Rough CPU-vs-GPU throughput on the expert decode shape (m small).
        let cpu_start = std::time::Instant::now();
        for _ in 0..20 {
            let mut c2 = vec![0.0f32; m * n];
            matmul_t((m, k, n), &x, &blocks, &mut c2).unwrap();
        }
        let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1e3 / 20.0;
        let gpu_start = std::time::Instant::now();
        for _ in 0..20 {
            let mut g2 = vec![0.0f32; m * n];
            try_opencl_matmul(&dev, (m, k, n), &x, &blocks, &mut g2).unwrap();
        }
        let gpu_ms = gpu_start.elapsed().as_secs_f64() * 1e3 / 20.0;
        eprintln!("  timing ({m},{k},{n}): cpu {cpu_ms:.3} ms  gpu {gpu_ms:.3} ms  speedup {:.2}x", cpu_ms / gpu_ms.max(1e-9));
    }
    dev.synchronize().unwrap();
}

/// The resident-device path (`Iq2OpenClWeight`) — weights uploaded once, the
/// activation already on the device — must match the CPU reference on the
/// exact decode shape, and shows the transfer-amortized speedup.
#[test]
fn opencl_iq2xxs_resident_matches_cpu() {
    use joshua::iq2xxs::Iq2OpenClWeight;
    let Some(dev) = device() else {
        eprintln!("SKIP: no OpenCL device");
        return;
    };
    let odev = match dev.as_opencl_device() {
        Ok(d) => d.clone(),
        Err(_) => return,
    };
    for (m, k, n) in [(1usize, 2048usize, 2048usize), (1, 4096, 2048)] {
        let (blocks, _) = blocks_for((m, k, n));
        // Resident weight (upload once).
        let w = Iq2OpenClWeight::upload(&odev, &blocks, n, k).unwrap();
        // Activation as an OpenCl tensor.
        let x_v: Vec<f32> = (0..m * k).map(|i| ((i % 29) as f32 - 14.0) * 0.03).collect();
        let xs = candle_core::Tensor::from_vec(x_v.clone(), (m, k), &dev).unwrap();

        // Device result (resident path, activation on device).
        let out = w.matmul(&odev, &xs).unwrap();
        odev.synchronize().unwrap();
        let gpu: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();

        // CPU reference.
        let mut cpu = vec![0.0f32; m * n];
        matmul_t((m, k, n), &x_v, &blocks, &mut cpu).unwrap();

        let mut worst = 0.0f32;
        for i in 0..m * n {
            let rel = (gpu[i] - cpu[i]).abs() / cpu[i].abs().max(1e-4);
            worst = worst.max(rel);
        }
        eprintln!("resident ({m},{k},{n}): worst rel diff = {worst:.3e}");
        assert!(worst < 2e-3, "resident ({m},{k},{n}) parity failed: worst={worst:.3e}");

        // Transfer-amortized timing: repeated matmuls, weights resident.
        let cpu_start = std::time::Instant::now();
        for _ in 0..30 {
            let mut c2 = vec![0.0f32; m * n];
            matmul_t((m, k, n), &x_v, &blocks, &mut c2).unwrap();
        }
        let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1e3 / 30.0;
        let gpu_start = std::time::Instant::now();
        for _ in 0..30 {
            w.matmul(&odev, &xs).unwrap();
        }
        odev.synchronize().unwrap(); // include completion
        let gpu_ms = gpu_start.elapsed().as_secs_f64() * 1e3 / 30.0;
        eprintln!("  resident timing ({m},{k},{n}): cpu {cpu_ms:.3} ms  gpu {gpu_ms:.3} ms  speedup {:.2}x", cpu_ms / gpu_ms.max(1e-9));
    }
}