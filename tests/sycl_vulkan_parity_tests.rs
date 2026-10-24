//! B50 regressions for the arithmetic contract used by production Vulkan.
#![cfg(all(feature = "sycl", feature = "vulkan", target_os = "linux"))]
mod common;
use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{Device, Module, Tensor};
use std::{borrow::Cow, sync::Arc};

fn values(x: Tensor) -> Vec<f32> {
    x.flatten_all().unwrap().to_vec1().unwrap()
}
fn assert_same(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let max = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        a.iter().chain(b).all(|v| v.is_finite()),
        "{label}: nonfinite"
    );
    assert!(max <= 1e-5, "{label}: max absolute difference {max:e}");
}
fn assert_same_bits(label: &str, a: &[f32], b: &[f32]) {
    assert_same(label, a, b);
    let first = a
        .iter()
        .zip(b)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    assert!(
        first.is_none(),
        "{label}: first different bits at {first:?}"
    );
}
fn weight(bytes: &[u8], dtype: GgmlDType, n: usize, k: usize, dev: &Device) -> QMatMul {
    QMatMul::from_qtensor(
        QTensor::new(
            QStorage::from_data(Cow::Borrowed(bytes), dev, dtype).unwrap(),
            (n, k),
        )
        .unwrap(),
    )
    .unwrap()
}
#[test]
fn b50_reductions_and_qwen_continuation_match_production_vulkan() {
    // Keep context creation, submissions and teardown on one test thread.
    let Ok(sycl) = Device::new_sycl(0) else {
        eprintln!("SKIP: SYCL unavailable");
        return;
    };
    let Device::Sycl(s) = &sycl else {
        unreachable!()
    };
    if !s.name().contains("Pro B50") {
        eprintln!("SKIP: B50-specific rounding regression");
        return;
    }
    assert!(
        s.ctx().canonical_quantized,
        "B50 requires the current bridge's numerical contract"
    );
    assert_eq!(
        s.ctx().kernel("k_qgemv").unwrap().wg,
        64,
        "retain bounded physical groups"
    );
    let vk = Device::new_vulkan(0).unwrap();
    for cols in [128, 200, 4096, 12288] {
        let data: Vec<f32> = (0..3 * cols)
            .map(|i| ((i * 17 % 199) as f32 - 99.) * 0.00137)
            .collect();
        let alpha: Vec<f32> = (0..cols).map(|i| 30. + (i % 31) as f32 * 0.37).collect();
        let outputs: Vec<_> = [&vk, &sycl]
            .into_iter()
            .map(|dev| {
                values(
                    candle_nn::ops::rms_norm(
                        &Tensor::from_vec(data.clone(), (3, cols), dev).unwrap(),
                        &Tensor::from_vec(alpha.clone(), cols, dev).unwrap(),
                        1e-6,
                    )
                    .unwrap(),
                )
            })
            .collect();
        // Bit equality pins the two Qwen normalization widths; other
        // widths retain the unchanged absolute 1e-5 contract.
        if matches!(cols, 128 | 4096) {
            assert_same_bits(&format!("rms cols={cols}"), &outputs[0], &outputs[1]);
        } else {
            assert_same(&format!("rms cols={cols}"), &outputs[0], &outputs[1]);
        }
    }
    for cols in [31, 126, 254, 513] {
        let data: Vec<f32> = (0..3 * cols)
            .map(|i| {
                if i % cols > cols / 2 {
                    f32::NEG_INFINITY
                } else {
                    ((i * 13 % 71) as f32 - 35.) * 0.173
                }
            })
            .collect();
        let outputs: Vec<_> = [&vk, &sycl]
            .into_iter()
            .map(|dev| {
                values(
                    candle_nn::ops::softmax_last_dim(
                        &Tensor::from_vec(data.clone(), (3, cols), dev).unwrap(),
                    )
                    .unwrap(),
                )
            })
            .collect();
        assert_same_bits(&format!("softmax cols={cols}"), &outputs[0], &outputs[1]);
    }
    for dtype in [GgmlDType::Q4K, GgmlDType::Q6K] {
        for k in [1024, 4096, 12288] {
            let n = 17;
            let ws: Vec<f32> = (0..n * k)
                .map(|i| ((i * 19 % 173) as f32 - 86.) * 0.0091)
                .collect();
            let q = QTensor::quantize(&Tensor::from_vec(ws, (n, k), &Device::Cpu).unwrap(), dtype)
                .unwrap();
            let bytes = q.data().unwrap();
            let v = weight(&bytes, dtype, n, k, &vk);
            let s = weight(&bytes, dtype, n, k, &sycl);
            for m in [1, 2, 16, 32] {
                let x: Vec<f32> = (0..m * k)
                    .map(|i| ((i * 11 % 139) as f32 - 69.) * 0.0037)
                    .collect();
                let a = values(
                    v.forward(&Tensor::from_vec(x.clone(), (m, k), &vk).unwrap())
                        .unwrap(),
                );
                let b = values(
                    s.forward(&Tensor::from_vec(x, (m, k), &sycl).unwrap())
                        .unwrap(),
                );
                assert_same_bits(&format!("{dtype:?} m={m} k={k}"), &a, &b);
            }
        }
    }
    let dir = common::model_dir("sycl-vulkan-qwen-parity");
    let path = dir.join("model.gguf");
    common::write_tiny_qwen3_q4k_layers(&path, 2);
    let load = |dev: &Device| {
        let mut f = std::fs::File::open(&path).unwrap();
        let h = joshua::gguf_ext::read_header(&mut f).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&f) }.unwrap();
        joshua::model::QuantizedModel::from_gguf_mmap(
            h.to_candle_content().unwrap(),
            &mut f,
            dev,
            Some(Arc::new(mmap)),
            None,
            0,
        )
        .unwrap()
    };
    let mut a = load(&vk);
    let mut b = load(&sycl);
    let mut offset = 0;
    for ids in [&[1u32, 4, 2, 7][..], &[5, 9, 3][..], &[6][..], &[8][..]] {
        let av = values(
            a.forward(
                &Tensor::new(ids, &vk).unwrap().unsqueeze(0).unwrap(),
                offset,
            )
            .unwrap(),
        );
        let bv = values(
            b.forward(
                &Tensor::new(ids, &sycl).unwrap().unsqueeze(0).unwrap(),
                offset,
            )
            .unwrap(),
        );
        assert_same(&format!("qwen continuation offset={offset}"), &av, &bv);
        offset += ids.len();
    }
    let verify = [3u32, 2];
    let av = values(
        a.forward_all_logits(
            &Tensor::new(&verify, &vk).unwrap().unsqueeze(0).unwrap(),
            offset,
        )
        .unwrap(),
    );
    let bv = values(
        b.forward_all_logits(
            &Tensor::new(&verify, &sycl).unwrap().unsqueeze(0).unwrap(),
            offset,
        )
        .unwrap(),
    );
    assert_same("qwen all-row verification", &av, &bv);
    assert!(a.truncate_kv_cache(offset).unwrap());
    assert!(b.truncate_kv_cache(offset).unwrap());
    let av = values(
        a.forward(
            &Tensor::new(&[5u32], &vk).unwrap().unsqueeze(0).unwrap(),
            offset,
        )
        .unwrap(),
    );
    let bv = values(
        b.forward(
            &Tensor::new(&[5u32], &sycl).unwrap().unsqueeze(0).unwrap(),
            offset,
        )
        .unwrap(),
    );
    assert_same("qwen verification rewind", &av, &bv);
    std::fs::remove_dir_all(dir).unwrap();
}
