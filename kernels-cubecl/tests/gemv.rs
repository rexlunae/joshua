use candle_core::{
    DType, Device, VulkanDevice,
    backend::{BackendDevice, BackendStorage},
    quantized::{GgmlDType, QStorage, QTensor},
};
use cubecl::{
    device::{WgpuBackend, WgpuDevice, WgpuDeviceKind},
    prelude::*,
};
use joshua_kernels_cubecl::GemvShape;
use std::{borrow::Cow, time::Instant};

fn client() -> Client {
    cubecl::Device::Wgpu(WgpuDevice::new(WgpuDeviceKind::DiscreteGpu(0)).on(WgpuBackend::Vulkan))
        .client()
}

// Arbitrary valid packed blocks exercise all scale/min high bits and nibbles,
// independently of the quantizer's chosen scales. Include half subnormals.
fn fixture(n: usize, k: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let mut bytes = vec![0u8; n * (k / 256) * 144];
    let mut seed = 42u32;
    for (b, block) in bytes.as_chunks_mut::<144>().0.iter_mut().enumerate() {
        for byte in block.iter_mut() {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *byte = seed as u8;
        }
        let d = if b % 11 == 0 {
            half::f16::from_bits(7)
        } else {
            half::f16::from_f32(0.003)
        };
        let min = half::f16::from_f32(0.002);
        block[..2].copy_from_slice(&d.to_le_bytes());
        block[2..4].copy_from_slice(&min.to_le_bytes());
    }
    let x: Vec<f32> = (0..k)
        .map(|i| ((i * 17 % 97) as f32 - 48.0) / 49.0)
        .collect();
    let qt = QTensor::new(
        QStorage::from_data(Cow::Borrowed(&bytes), &Device::Cpu, GgmlDType::Q4K).unwrap(),
        (n, k),
    )
    .unwrap();
    let w = qt
        .dequantize(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    // Independent scalar f64 dot over Candle's dequantized GGUF data.
    let reference = w
        .chunks_exact(k)
        .map(|row| {
            row.iter()
                .zip(&x)
                .map(|(&w, &x)| w as f64 * x as f64)
                .sum::<f64>() as f32
        })
        .collect();
    (bytes, x, reference)
}

fn close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() <= 1e-4 + 1e-5 * b.abs(),
            "row {i}: {a} != {b}"
        );
    }
}

#[test]
fn rejects_invalid_shapes() {
    for (n, k, wg) in [
        (0, 256, 64),
        (1, 0, 64),
        (1, 257, 64),
        (65536, 256, 64),
        (1, 256, 48),
        (65535, usize::MAX - 255, 64),
    ] {
        assert!(GemvShape::new(n, k, wg).is_err());
    }
}

#[test]
fn q4k_parity_offsets_and_reused_buffers() {
    let client = client();
    println!(
        "CubeCL: {} on {}",
        client.name(),
        client.properties().identity.name
    );
    for (n, k) in [(1, 256), (7, 512), (33, 2304), (17, 8960)] {
        let (bytes, x, reference) = fixture(n, k);
        // Prefix/suffix sentinels verify offset addressing and partial rows.
        let mut wb = vec![0x55u8; 12];
        wb.extend_from_slice(&bytes);
        wb.extend_from_slice(&[0xaa; 4]);
        let mut xb = vec![999f32; 3];
        xb.extend_from_slice(&x);
        xb.push(-999.);
        let w = client.create_from_slice(&wb);
        let input = client.create_from_slice(f32::as_bytes(&xb));
        let out = client.empty(n * 4);
        for wg in [32, 64, 128, 256] {
            let shape = GemvShape::new(n, k, wg).unwrap();
            for sign in [1.0f32, -1.0, 0.0] {
                for (dst, src) in xb[3..3 + k].iter_mut().zip(&x) {
                    *dst = sign * src;
                }
                client.write(
                    &input,
                    cubecl::bytes::Bytes::from_bytes_vec(f32::as_bytes(&xb).to_vec()),
                );
                // SAFETY: allocations above satisfy the documented lengths,
                // offsets, device ownership, layout and non-aliasing contract.
                unsafe {
                    shape.enqueue(&client, &w, &input, &out, 12, 3);
                }
                let result = client.read_one(out.clone()).unwrap();
                close(
                    f32::from_bytes(&result),
                    &reference.iter().map(|v| sign * v).collect::<Vec<_>>(),
                );
            }
        }
    }
}

/// Opt in explicitly: microbenchmark only, not model token throughput.
#[test]
#[ignore = "GPU timing; run with --release --ignored --nocapture"]
fn benchmark_q4k_against_native_vulkan() {
    use candle_core::vulkan_backend::kernels;
    let client = client();
    let dev = VulkanDevice::new(0).unwrap();
    println!(
        "CubeCL: {} on {}; Candle: {}",
        client.name(),
        client.properties().identity.name,
        dev.name()
    );
    assert_eq!(
        client.properties().identity.name,
        dev.name(),
        "benchmark must use the same GPU"
    );
    assert!(kernels::native_enabled());
    for (n, k) in [(512, 2048), (2048, 2048), (2048, 8960)] {
        let (bytes, x, reference) = fixture(n, k);
        let w = client.create_from_slice(&bytes);
        let input = client.create_from_slice(f32::as_bytes(&x));
        let out = client.empty(n * 4);
        let words: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| u32::from_le_bytes(*v))
            .collect();
        let native_w = dev.storage_from_slice(&words).unwrap();
        let native_x = dev.storage_from_slice(&x).unwrap();
        let native_out = dev.alloc(DType::F32, n).unwrap();
        let native = || {
            kernels::run_qgemv(
                &dev,
                GgmlDType::Q4K,
                native_x.buf(),
                native_w.buf(),
                native_out.buf(),
                1,
                n,
                k,
                0,
            )
            .unwrap()
        };
        native();
        dev.synchronize().unwrap();
        close(
            native_out
                .to_cpu_storage()
                .unwrap()
                .as_slice::<f32>()
                .unwrap(),
            &reference,
        );
        let native_us = measure(|| {
            native();
            dev.synchronize().unwrap();
        });
        println!("n={n} k={k}: native={native_us:.2} us");
        for wg in [32, 64, 128, 256] {
            let shape = GemvShape::new(n, k, wg).unwrap();
            let launch = || unsafe {
                shape.enqueue(&client, &w, &input, &out, 0, 0);
            };
            launch();
            pollster::block_on(client.sync()).unwrap();
            close(
                f32::from_bytes(&client.read_one(out.clone()).unwrap()),
                &reference,
            );
            let us = measure(|| {
                launch();
                pollster::block_on(client.sync()).unwrap();
            });
            println!(
                "  wg={wg}: {us:.2} us; native/CubeCL={:.3}x",
                native_us / us
            );
        }
    }
}

fn measure(mut call: impl FnMut()) -> f64 {
    for _ in 0..5 {
        call();
    }
    let mut samples = Vec::with_capacity(5);
    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..20 {
            call();
        }
        samples.push(start.elapsed().as_secs_f64() * 1e6 / 20.0);
    }
    samples.sort_by(f64::total_cmp);
    samples[2]
}
