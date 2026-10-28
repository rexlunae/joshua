#![cfg(feature = "vulkan")]
mod common;
use candle_core::vulkan_backend::kernels::{self, Qgemv};
use candle_core::{
    backend::{BackendDevice, BackendStorage},
    quantized::{GgmlDType, QTensor},
    DType, Device, Tensor, VulkanDevice,
};

#[test]
fn cubecl_uses_native_buffers_queue_and_deferred_release() {
    let dev = VulkanDevice::new(0).expect("Vulkan required for this test");
    for dtype in [
        GgmlDType::Q4K,
        GgmlDType::Q8_0,
        GgmlDType::Q2K,
        GgmlDType::Q6K,
    ] {
        let (n, k) = (65, 2560);
        let weights = Tensor::from_vec(common::weights(n * k, 123), (n, k), &Device::Cpu).unwrap();
        let qt = QTensor::quantize(&weights, dtype).unwrap();
        let bytes = qt.data().unwrap();
        let words: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| u32::from_le_bytes(*v))
            .collect();
        let w = dev.storage_from_slice(&words).unwrap();
        let mut x = vec![999.; 3];
        x.extend(common::weights(k, 17));
        let x = dev.storage_from_slice(&x).unwrap();
        let producer = dev.alloc(DType::F32, k + 3).unwrap();
        kernels::run_affine(
            &dev,
            x.buf(),
            producer.buf(),
            k + 3,
            &candle_core::Layout::contiguous(k + 3),
            1.5,
            0.25,
        )
        .unwrap();
        let a = dev.alloc(DType::F32, n).unwrap();
        let b = dev.alloc(DType::F32, n).unwrap();
        // Both kernels read the same allocations, queued without a host fence.
        kernels::run_qgemv_with(
            &dev,
            Qgemv::Generic,
            dtype,
            producer.buf(),
            w.buf(),
            a.buf(),
            1,
            n,
            k,
            3,
        )
        .unwrap();
        kernels::run_qgemv_with(
            &dev,
            Qgemv::CubeClNext,
            dtype,
            producer.buf(),
            w.buf(),
            b.buf(),
            1,
            n,
            k,
            3,
        )
        .unwrap();
        let consumer = dev.alloc(DType::F32, n).unwrap();
        kernels::run_affine(
            &dev,
            b.buf(),
            consumer.buf(),
            n,
            &candle_core::Layout::contiguous(n),
            2.,
            1.,
        )
        .unwrap();
        drop(producer); // Still referenced by the queued GEMVs.
        let av = a.to_cpu_storage().unwrap();
        let bv = b.to_cpu_storage().unwrap();
        common::assert_close(
            av.as_slice::<f32>().unwrap(),
            bv.as_slice::<f32>().unwrap(),
            1e-5,
            "shared-buffer GEMV",
        );
        let expected: Vec<f32> = av
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|v| 2. * v + 1.)
            .collect();
        common::assert_close(
            consumer
                .to_cpu_storage()
                .unwrap()
                .as_slice::<f32>()
                .unwrap(),
            &expected,
            2e-5,
            "queued consumer",
        );
        dev.synchronize().unwrap();
        let before = dev.memory_stats();
        // Cross the 1024-dispatch batch boundary. Dropped outputs must stay alive
        // until the common recorder completes; no retained model/scratch copies.
        for _ in 0..1100 {
            let out = dev.alloc(DType::F32, n).unwrap();
            kernels::run_qgemv_with(
                &dev,
                Qgemv::CubeClNext,
                dtype,
                x.buf(),
                w.buf(),
                out.buf(),
                1,
                n,
                k,
                3,
            )
            .unwrap();
            drop(out);
        }
        dev.synchronize().unwrap();
        assert_eq!(
            before,
            dev.memory_stats(),
            "resident allocations grew after completion"
        );
    }
}

fn load(path: &std::path::Path, dev: &Device, mmap: bool) -> joshua::model::QuantizedModel {
    let bytes = std::fs::read(path).unwrap();
    let mut cursor = std::io::Cursor::new(&bytes[..]);
    let header = joshua::gguf_ext::read_header(&mut cursor).unwrap();
    let mapping = mmap.then(|| {
        std::sync::Arc::new(unsafe {
            memmap2::Mmap::map(&std::fs::File::open(path).unwrap()).unwrap()
        })
    });
    let content = header.to_candle_content().unwrap();
    let expert_device = match content.metadata.get("general.architecture") {
        Some(candle_core::quantized::gguf_file::Value::String(arch))
            if arch.starts_with("deepseek4") =>
        {
            &Device::Cpu
        }
        _ => dev,
    };
    joshua::model::QuantizedModel::from_gguf_mmap_placed(
        content,
        &mut cursor,
        dev,
        expert_device,
        mapping,
        None,
        0,
        None,
        joshua::EngramTier::Auto,
    )
    .unwrap()
}

fn logits(
    model: &mut joshua::model::QuantizedModel,
    tokens: &[u32],
    offset: usize,
    dev: &Device,
    all: bool,
) -> Vec<f32> {
    let input = Tensor::new(tokens, dev).unwrap().unsqueeze(0).unwrap();
    let y = if all {
        model.forward_all_logits(&input, offset)
    } else {
        model.forward(&input, offset)
    }
    .unwrap();
    y.to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

#[test]
fn tiny_production_models_match_generic_prefill_decode_and_all_rows() {
    let base = Device::Vulkan(VulkanDevice::new_with_qgemv(0, Qgemv::Generic).unwrap());
    let candidate = Device::Vulkan(VulkanDevice::new_with_qgemv(0, Qgemv::CubeClProduction).unwrap());
    let dir = common::model_dir("cubecl-q4k-parity");
    type Fixture = (&'static str, fn(&std::path::Path), &'static [bool]);
    // DeepSeek4's raw IQ2_XXS experts require mmap for accelerator execution;
    // its streamed/heap form intentionally rejects device expert uploads.
    let writers: &[Fixture] = &[
        (
            "qwen3",
            |p| common::write_tiny_qwen3_q4k_layers(p, 2),
            &[false, true],
        ),
        (
            "qwen3q8",
            |p| common::write_tiny_qwen3_quantized_layers(p, 2),
            &[false, true],
        ),
        (
            "qwen3moe",
            common::write_tiny_qwen3moe_metal_gguf,
            &[false, true],
        ),
        (
            "qwen3q6",
            |p| common::write_tiny_qwen3_q6k_layers(p, 2),
            &[false, true],
        ),
        ("deepseek41", common::write_tiny_deepseek41_gguf, &[true]),
        (
            "deepseek41group",
            |p| {
                common::write_tiny_deepseek4_gguf_opts(
                    p,
                    common::TinyDeepseek4Opts {
                        v41: true,
                        grouped_q8: true,
                        ..Default::default()
                    },
                )
            },
            &[true],
        ),
        (
            "deepseek4",
            common::write_tiny_deepseek4_gguf_kquant,
            &[true],
        ),
    ];
    for (name, writer, modes) in writers {
        let path = dir.join(format!("{name}.gguf"));
        writer(&path);
        for &mmap in *modes {
            let mut a = load(&path, &base, mmap);
            let mut b = load(&path, &candidate, mmap);
            for (tokens, offset, all) in [
                (&[1, 4, 2][..], 0, false),
                (&[7][..], 3, false),
                (&[5][..], 4, false),
                (&[2, 3][..], 5, true),
                (&[1][..], 7, false),
            ] {
                let before = kernels::fallback_count();
                let expected = logits(&mut a, tokens, offset, &base, all);
                let baseline_fallbacks = kernels::fallback_count() - before;
                let before = kernels::fallback_count();
                let got = logits(&mut b, tokens, offset, &candidate, all);
                assert_eq!(
                    baseline_fallbacks,
                    kernels::fallback_count() - before,
                    "candidate introduced CPU fallbacks/transfers"
                );
                common::assert_close(
                    &got,
                    &expected,
                    1e-5,
                    &format!("{name} mmap={mmap} offset={offset} all={all}"),
                );
                let argmax = |v: &[f32]| {
                    v.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .unwrap()
                        .0
                };
                assert_eq!(argmax(&got), argmax(&expected));
            }
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}

/// Full model forward + host logits, with fixed tokens to give both paths the
/// same workload. Alternates timing order and checks quiescent allocator use.
#[test]
#[ignore = "end-to-end GPU benchmark; run release, serial, with --ignored --nocapture"]
fn benchmark_decode_and_session_memory() {
    let dir = common::model_dir("cubecl-q4k-benchmark");
    let path = dir.join("model.gguf");
    common::write_small_qwen3_q4k_layers(&path, 4);
    let base_vk = VulkanDevice::new_with_qgemv(0, Qgemv::Generic).unwrap();
    let candidate_vk = VulkanDevice::new_with_qgemv(0, Qgemv::CubeClNext).unwrap();
    assert_eq!(base_vk.name(), candidate_vk.name());
    println!("GPU: {}", base_vk.name());
    let base = Device::Vulkan(base_vk.clone());
    let candidate = Device::Vulkan(candidate_vk.clone());
    let a = load(&path, &base, true);
    let b = load(&path, &candidate, true);
    base_vk.synchronize().unwrap();
    candidate_vk.synchronize().unwrap();
    assert_eq!(
        base_vk.memory_stats(),
        candidate_vk.memory_stats(),
        "extra model storage"
    );
    let run = |model: &joshua::model::QuantizedModel, dev: &Device| {
        let mut session = model.new_session().unwrap();
        let _ = logits(&mut session, &[1, 4, 2, 7], 0, dev, false);
        let start = std::time::Instant::now();
        let mut outputs = Vec::new();
        for step in 0..32 {
            outputs.push(logits(
                &mut session,
                &[(step % 15 + 1) as u32],
                4 + step,
                dev,
                false,
            ));
        }
        (start.elapsed().as_secs_f64() * 1e3 / 32.0, outputs)
    };
    run(&a, &base);
    run(&b, &candidate);
    base_vk.synchronize().unwrap();
    candidate_vk.synchronize().unwrap();
    let memory_a = base_vk.memory_stats();
    let memory_b = candidate_vk.memory_stats();
    assert_eq!(memory_a, memory_b);
    let rss_before = rss_kib();
    let mut times_a = Vec::new();
    let mut times_b = Vec::new();
    for pass in 0..7 {
        let (ra, rb) = if pass % 2 == 0 {
            (run(&a, &base), run(&b, &candidate))
        } else {
            let b = run(&b, &candidate);
            (run(&a, &base), b)
        };
        for (a, b) in ra.1.iter().zip(&rb.1) {
            common::assert_close(a, b, 1e-5, "benchmark token logits");
        }
        times_a.push(ra.0);
        times_b.push(rb.0);
        base_vk.synchronize().unwrap();
        candidate_vk.synchronize().unwrap();
        assert_eq!(
            memory_a,
            base_vk.memory_stats(),
            "baseline retained session allocations"
        );
        assert_eq!(
            memory_b,
            candidate_vk.memory_stats(),
            "candidate retained session allocations"
        );
    }
    let rss_after = rss_kib();
    println!("process RSS KiB after warmup / after 7 sessions: {rss_before:?} / {rss_after:?}");
    if let (Some(before), Some(after)) = (rss_before, rss_after) {
        assert!(
            after <= before + 1024,
            "host RSS grew by more than 1 MiB after warmup"
        );
    }
    times_a.sort_by(f64::total_cmp);
    times_b.sort_by(f64::total_cmp);
    println!(
        "4 layers, width 512, FFN 1536; 4-token prompt + 32 decode tokens, 7 alternating passes"
    );
    println!("generic ms/token: {times_a:?}; CubeCL ms/token: {times_b:?}");
    println!(
        "median {:.3} vs {:.3} ms/token; speedup {:.3}x",
        times_a[3],
        times_b[3],
        times_a[3] / times_b[3]
    );
    println!("reserved/used bytes before and after 7 sessions: generic={memory_a:?}, CubeCL={memory_b:?}");
    std::fs::remove_dir_all(dir).unwrap();
}

fn rss_kib() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find(|line| line.starts_with("VmRSS:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

#[test]
fn validated_adapter_enables_production_dispatch() {
    let dev = VulkanDevice::new(0).unwrap();
    if dev.name() == "Intel(R) Arc(tm) Pro B50 Graphics (BMG G21)"
        && std::env::var_os("JOSHUA_VULKAN_QGEMV").is_none()
    {
        assert_eq!(dev.limits().qgemv, Qgemv::CubeClProduction);
        assert!(kernels::grouped_q8_eligible(&dev, 8, 8192, 4096));
        for (groups, n, k) in [(4, 8192, 4096), (8, 4096, 8192),
            (8, 8200, 4096), (8, 8192, 8192)] {
            assert!(!kernels::grouped_q8_eligible(&dev, groups, n, k),
                "unvalidated geometry must retain its fallback");
        }
        assert_eq!(Qgemv::for_adapter(0x8086, 0xE20B, false), Qgemv::Generic);
        let reference = VulkanDevice::new_with_qgemv(0, Qgemv::CubeClPacked).unwrap();
        assert!(!kernels::grouped_q8_eligible(&reference, 8, 8192, 4096));
    }
}

#[test]
fn cubecl_packed_gemv_tiled_output_and_unaligned_q8_blocks() {
    let vk = VulkanDevice::new_with_qgemv(0, Qgemv::CubeClNext).unwrap();
    for (dtype, n, k) in [
        (GgmlDType::Q8_0, 129280, 64),
        (GgmlDType::Q8_0, 34, 32),
        (GgmlDType::Q2K, 65537, 256),
        (GgmlDType::Q6K, 65538, 256),
        (GgmlDType::Q6K, 65, 256),
    ] {
        let weights = Tensor::from_vec(common::weights(n * k, 41), (n, k), &Device::Cpu).unwrap();
        let qt = QTensor::quantize(&weights, dtype).unwrap();
        let w = vk.storage_from_slice(qt.data().unwrap().as_ref()).unwrap();
        let mut input = vec![99.; 7];
        input.extend(common::weights(k, 37));
        let x = vk.storage_from_slice(&input).unwrap();
        let outputs: Vec<_> = [Qgemv::Generic, Qgemv::CubeClNext]
            .into_iter()
            .map(|kernel| {
                let out = vk.alloc(DType::F32, n).unwrap();
                kernels::run_qgemv_with(
                    &vk,
                    kernel,
                    dtype,
                    x.buf(),
                    w.buf(),
                    out.buf(),
                    1,
                    n,
                    k,
                    7,
                )
                .unwrap();
                out.to_cpu_storage()
                    .unwrap()
                    .as_slice::<f32>()
                    .unwrap()
                    .to_vec()
            })
            .collect();
        common::assert_close(&outputs[0], &outputs[1], 1e-5, "tiled packed GEMV");
    }
}

#[test]
#[ignore = "GPU timing: run release and serial"]
fn benchmark_deepseek_packed_projections() {
    let vk = VulkanDevice::new_with_qgemv(0, Qgemv::CubeClNext).unwrap();
    println!("GPU={}", vk.name());
    for (dtype, n, k) in [
        (GgmlDType::Q4K, 2304, 5120),
        (GgmlDType::Q8_0, 1280, 5120),
        (GgmlDType::Q8_0, 32768, 1280),
        (GgmlDType::Q8_0, 129280, 5120),
        (GgmlDType::Q2K, 24, 20480),
        (GgmlDType::Q2K, 512, 4096),
    ] {
        let weights = Tensor::from_vec(common::weights(n * k, 41), (n, k), &Device::Cpu).unwrap();
        let qt = QTensor::quantize(&weights, dtype).unwrap();
        drop(weights);
        let w = vk.storage_from_slice(qt.data().unwrap().as_ref()).unwrap();
        let x = vk.storage_from_slice(&common::weights(k, 37)).unwrap();
        let out = vk.alloc(DType::F32, n).unwrap();
        let mut timings = [Vec::new(), Vec::new()];
        let mut reference = Vec::new();
        let memory = vk.memory_stats();
        for pass in 0..7 {
            for idx in if pass % 2 == 0 { [0, 1] } else { [1, 0] } {
                let kernel = [Qgemv::Generic, Qgemv::CubeClNext][idx];
                kernels::run_qgemv_with(
                    &vk,
                    kernel,
                    dtype,
                    x.buf(),
                    w.buf(),
                    out.buf(),
                    1,
                    n,
                    k,
                    0,
                )
                .unwrap();
                let got = out
                    .to_cpu_storage()
                    .unwrap()
                    .as_slice::<f32>()
                    .unwrap()
                    .to_vec();
                if reference.is_empty() {
                    reference = got;
                } else {
                    common::assert_close(&reference, &got, 1e-5, "projection benchmark parity");
                }
                let start = std::time::Instant::now();
                for _ in 0..10 {
                    kernels::run_qgemv_with(
                        &vk,
                        kernel,
                        dtype,
                        x.buf(),
                        w.buf(),
                        out.buf(),
                        1,
                        n,
                        k,
                        0,
                    )
                    .unwrap();
                }
                vk.synchronize().unwrap();
                timings[idx].push(start.elapsed().as_secs_f64() * 1000. / 10.);
            }
        }
        assert_eq!(
            memory,
            vk.memory_stats(),
            "kernel allocated extra device memory"
        );
        for values in &mut timings {
            values.sort_by(f64::total_cmp);
        }
        println!(
            "{dtype:?} n={n} k={k} generic_ms={:.6} cubecl_ms={:.6} speedup={:.3}",
            timings[0][3],
            timings[1][3],
            timings[0][3] / timings[1][3]
        );
    }
}

#[test]
fn packed_gemv_preserves_generic_accumulation_order() {
    let vk = VulkanDevice::new_with_qgemv(0, Qgemv::CubeClNext).unwrap();
    for dtype in [
        GgmlDType::Q4K,
        GgmlDType::Q8_0,
        GgmlDType::Q2K,
        GgmlDType::Q6K,
    ] {
        for k in [1280, 5120, 8192, 20480] {
            let n = 66;
            let weights: Vec<_> = common::weights(n * k, 391)
                .into_iter()
                .map(|v| v * 32.)
                .collect();
            let weights = Tensor::from_vec(weights, (n, k), &Device::Cpu).unwrap();
            let qt = QTensor::quantize(&weights, dtype).unwrap();
            let w = vk.storage_from_slice(qt.data().unwrap().as_ref()).unwrap();
            let input: Vec<_> = common::weights(k, 77)
                .into_iter()
                .map(|v| v * 10.)
                .collect();
            let x = vk.storage_from_slice(&input).unwrap();
            let mut outputs = Vec::new();
            for kernel in [Qgemv::Generic, Qgemv::CubeClFormat(dtype)] {
                let out = vk.alloc(DType::F32, n).unwrap();
                kernels::run_qgemv_with(
                    &vk,
                    kernel,
                    dtype,
                    x.buf(),
                    w.buf(),
                    out.buf(),
                    1,
                    n,
                    k,
                    0,
                )
                .unwrap();
                outputs.push(
                    out.to_cpu_storage()
                        .unwrap()
                        .as_slice::<f32>()
                        .unwrap()
                        .to_vec(),
                );
            }
            let max_abs = outputs[0]
                .iter()
                .zip(&outputs[1])
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let different = outputs[0]
                .iter()
                .zip(&outputs[1])
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                different, 0,
                "{dtype:?} k={k}: changed accumulation order; max_abs={max_abs:e}"
            );
        }
    }
}

#[test]
fn tiny_grouped_q8_matches_packed_cache_continuation_and_all_rows() {
    let base = Device::Vulkan(VulkanDevice::new_with_qgemv(0, Qgemv::CubeClPacked).unwrap());
    let candidate = Device::Vulkan(VulkanDevice::new_with_qgemv(0, Qgemv::CubeClNext).unwrap());
    let dir = common::model_dir("cubecl-grouped-parity");
    let path = dir.join("deepseek41.gguf");
    common::write_tiny_deepseek4_gguf_opts(
        &path,
        common::TinyDeepseek4Opts {
            v41: true,
            grouped_q8: true,
            ..Default::default()
        },
    );
    let mut a = load(&path, &base, true);
    let mut b = load(&path, &candidate, true);
    for (tokens, offset, all) in [
        (&[1, 4, 2][..], 0, false),
        (&[7][..], 3, false),
        (&[5][..], 4, false),
        (&[2, 3][..], 5, true),
        (&[1][..], 7, false),
    ] {
        let before = kernels::fallback_count();
        let expected = logits(&mut a, tokens, offset, &base, all);
        let baseline_fallbacks = kernels::fallback_count() - before;
        let before = kernels::fallback_count();
        let got = logits(&mut b, tokens, offset, &candidate, all);
        assert_eq!(kernels::fallback_count() - before, baseline_fallbacks);
        common::assert_close(
            &got,
            &expected,
            1e-5,
            &format!("grouped DeepSeek offset={offset} all={all}"),
        );
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn grouped_q8_matches_diagonal_projection_and_preserves_fallbacks() {
    use candle_core::{
        quantized::{QMatMul, QStorage},
        Module,
    };
    let vk = VulkanDevice::new_with_qgemv(0, Qgemv::CubeClNext).unwrap();
    let dev = Device::Vulkan(vk.clone());
    for groups in [1, 2, 8, 16, 17] {
        for k in [32, 1280, 8192] {
            let rank = 65;
            let n = groups * rank;
            let cpu = Tensor::from_vec(common::weights(n * k, 391), (n, k), &Device::Cpu).unwrap();
            let qt = QTensor::quantize(&cpu, GgmlDType::Q8_0).unwrap();
            let w = QTensor::new(
                QStorage::from_data(qt.data().unwrap(), &dev, GgmlDType::Q8_0).unwrap(),
                (n, k),
            )
            .unwrap();
            let weight = QMatMul::from_qtensor(w).unwrap();
            for seq in [1, 3] {
                let mut values = vec![99.; 3];
                values.extend(common::weights(seq * groups * k, 71));
                let x = Tensor::from_vec(values, 3 + seq * groups * k, &dev)
                    .unwrap()
                    .narrow(0, 3, seq * groups * k)
                    .unwrap()
                    .reshape((seq, groups, k))
                    .unwrap();
                let full = weight
                    .forward(&x.reshape((seq * groups, k)).unwrap())
                    .unwrap()
                    .reshape((seq, groups, n))
                    .unwrap();
                let blocks: Vec<_> = (0..groups)
                    .map(|g| {
                        full.narrow(1, g, 1)
                            .unwrap()
                            .narrow(2, g * rank, rank)
                            .unwrap()
                            .squeeze(1)
                            .unwrap()
                    })
                    .collect();
                let expected = Tensor::cat(&blocks, 1)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let got = joshua::attention::grouped_output_projection(&weight, &x, groups, rank)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                assert_eq!(expected, got, "groups={groups} seq={seq} k={k}");
                let eligible = groups <= 16 && (k / 32 % 2 == 0 || n % 2 == 0);
                assert_eq!(
                    weight
                        .try_grouped_q8(&x.reshape((seq * groups, k)).unwrap(), groups)
                        .unwrap()
                        .is_some(),
                    seq == 1 && eligible
                );
            }
        }
    }
}

#[test]
fn grouped_q8_deepseek_shape_preserves_exact_reduction() {
    use candle_core::{
        quantized::{QMatMul, QStorage},
        Module,
    };
    let vk = VulkanDevice::new_with_qgemv(0, Qgemv::CubeClProduction).unwrap();
    let dev = Device::Vulkan(vk);
    // Real DeepSeek output-A geometry; larger magnitudes make changes to
    // FMA/reduction order visible beyond the full-model 1e-5 gate.
    let (groups, rank, k) = (8, 1024, 4096);
    let n = groups * rank;
    let values = common::weights(n * k, 391)
        .into_iter()
        .map(|v| v * 70.)
        .collect();
    let cpu = Tensor::from_vec(values, (n, k), &Device::Cpu).unwrap();
    let qt = QTensor::quantize(&cpu, GgmlDType::Q8_0).unwrap();
    let w = QTensor::new(
        QStorage::from_data(qt.data().unwrap(), &dev, GgmlDType::Q8_0).unwrap(),
        (n, k),
    )
    .unwrap();
    let weight = QMatMul::from_qtensor(w).unwrap();
    for seed in [71, 193] {
        let mut values = vec![99.; 3];
        values.extend(
            common::weights(groups * k, seed)
                .into_iter()
                .map(|v| v * 80.),
        );
        let x = Tensor::from_vec(values, 3 + groups * k, &dev)
            .unwrap()
            .narrow(0, 3, groups * k)
            .unwrap()
            .reshape((groups, k))
            .unwrap();
        let full = weight.forward(&x).unwrap();
        let blocks: Vec<_> = (0..groups)
            .map(|g| {
                full.narrow(0, g, 1)
                    .unwrap()
                    .narrow(1, g * rank, rank)
                    .unwrap()
            })
            .collect();
        let expected = Tensor::cat(&blocks, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let got = weight
            .try_grouped_q8(&x, groups)
            .unwrap()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(expected, got, "DeepSeek grouped projection seed={seed}");
    }
}

#[test]
fn grouped_q8_queued_buffers_are_released_across_batch_boundary() {
    let vk = VulkanDevice::new_with_qgemv(0, Qgemv::CubeClNext).unwrap();
    let (groups, n, k) = (8, 520, 256);
    let cpu = Tensor::from_vec(common::weights(n * k, 12), (n, k), &Device::Cpu).unwrap();
    let qt = QTensor::quantize(&cpu, GgmlDType::Q8_0).unwrap();
    let w = vk.storage_from_slice(qt.data().unwrap().as_ref()).unwrap();
    let x = vk
        .storage_from_slice(&common::weights(groups * k + 3, 15))
        .unwrap();
    // Warm the allocator before checking stable resident allocation counts.
    for _ in 0..2 {
        for _ in 0..1100 {
            let out = vk.alloc(DType::F32, n).unwrap();
            kernels::run_grouped_q8(&vk, x.buf(), w.buf(), out.buf(), groups, n, k, 3).unwrap();
        }
        vk.synchronize().unwrap();
    }
    let before = vk.memory_stats();
    for _ in 0..1100 {
        let out = vk.alloc(DType::F32, n).unwrap();
        kernels::run_grouped_q8(&vk, x.buf(), w.buf(), out.buf(), groups, n, k, 3).unwrap();
    }
    vk.synchronize().unwrap();
    assert_eq!(before, vk.memory_stats());
}

#[test]
#[ignore = "requires B50/native float64 Vulkan and JOSHUA_VULKAN_PREFILL_OPT=1"]
fn precise_batched_grouped_q8_preserves_selected_outputs_and_bounds() {
    use candle_core::{
        quantized::{QMatMul, QStorage},
        Module,
    };
    let vk = VulkanDevice::new_with_qgemv(0, Qgemv::CubeClNext)
        .unwrap()
        .with_precise_math()
        .unwrap();
    let dev = Device::Vulkan(vk.clone());
    assert!(kernels::precise_prefill_optimized(&vk, 32));
    let before = kernels::fallback_count();
    // Odd output tiles and partial four-row tails, with an activation offset.
    for (groups, rank, k) in [(3, 7, 256), (8, 1024, 4096)] {
        let n = groups * rank;
        let cpu = Tensor::from_vec(
            common::weights(n * k, 391)
                .into_iter()
                .map(|v| v * 70.)
                .collect::<Vec<_>>(),
            (n, k),
            &Device::Cpu,
        )
        .unwrap();
        let qt = QTensor::quantize(&cpu, GgmlDType::Q8_0).unwrap();
        let weight = QMatMul::from_qtensor(
            QTensor::new(
                QStorage::from_data(qt.data().unwrap(), &dev, GgmlDType::Q8_0).unwrap(),
                (n, k),
            )
            .unwrap(),
        )
        .unwrap();
        for rows in [2usize, 4, 25, 32] {
            let count = rows * groups * k;
            let mut values = vec![99.; 3];
            values.extend(common::weights(count, 193).into_iter().map(|v| v * 80.));
            let x = Tensor::from_vec(values, count + 3, &dev)
                .unwrap()
                .narrow(0, 3, count)
                .unwrap()
                .reshape((rows, groups, k))
                .unwrap();
            let full = weight
                .forward(&x.reshape((rows * groups, k)).unwrap())
                .unwrap()
                .reshape((rows, groups, n))
                .unwrap();
            let blocks: Vec<_> = (0..groups)
                .map(|g| {
                    full.narrow(1, g, 1)
                        .unwrap()
                        .narrow(2, g * rank, rank)
                        .unwrap()
                        .squeeze(1)
                        .unwrap()
                })
                .collect();
            let expected = Tensor::cat(&blocks, 1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let got = weight
                .try_grouped_q8(&x, groups)
                .unwrap()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(expected, got, "groups={groups}, rows={rows}");
        }
        for rows in [0usize, 1, 33] {
            let x = Tensor::zeros((rows, groups, k), DType::F32, &dev).unwrap();
            assert!(weight.try_grouped_q8(&x, groups).unwrap().is_none());
        }
    }
    assert_eq!(before, kernels::fallback_count());
}
