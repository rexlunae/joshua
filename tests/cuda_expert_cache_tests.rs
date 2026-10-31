#![cfg(feature = "cuda")]

mod common;

use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{Device, Module, Tensor};
use joshua::model::QuantizedModel;
use std::borrow::Cow;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

const SLOT_BYTES: u64 = 2 * 128 * 66 + 1024 * 34;

fn cuda() -> Device {
    Device::new_cuda(0).expect("CUDA tests require a working CUDA device")
}

fn load(path: &Path, device: &Device, budget: Option<u64>) -> QuantizedModel {
    let bytes = std::fs::read(path).unwrap();
    let mut cursor = Cursor::new(&bytes);
    let header = joshua::gguf_ext::read_header(&mut cursor).unwrap();
    let mapping = unsafe { memmap2::Mmap::map(&std::fs::File::open(path).unwrap()) }.unwrap();
    QuantizedModel::from_gguf_mmap_placed(
        header.to_candle_content().unwrap(),
        &mut cursor,
        &Device::Cpu,
        device,
        Some(Arc::new(mapping)),
        None,
        0,
        budget,
        joshua::EngramTier::Auto,
    )
    .unwrap()
}

fn logits(model: &mut QuantizedModel, tokens: &[u32], offset: usize) -> Vec<f32> {
    let input = Tensor::new(tokens, &Device::Cpu)
        .unwrap()
        .reshape((1, tokens.len()))
        .unwrap();
    let output = model.forward_all_logits(&input, offset).unwrap();
    let (batch, rows, vocab) = output.dims3().unwrap();
    assert_eq!((batch, rows), (1, tokens.len()));
    assert!(vocab > 0);
    output.flatten_all().unwrap().to_vec1().unwrap()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    let spread = expected.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - expected.iter().copied().fold(f32::INFINITY, f32::min);
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(actual.is_finite() && expected.is_finite());
        assert!(
            (actual - expected).abs() <= 0.05 * spread.max(1e-3),
            "{actual} != {expected}"
        );
    }
}

#[test]
fn cuda_transfer_preserves_blocks_and_compute_visibility() {
    let device = cuda();
    let Device::Cuda(cuda) = &device else {
        unreachable!()
    };
    let transfer = cuda.cuda_transfer_stream().unwrap();
    assert!(!Arc::ptr_eq(&transfer, &cuda.cuda_stream()));
    assert!(Arc::ptr_eq(
        &transfer,
        &cuda.clone().cuda_transfer_stream().unwrap()
    ));
    for dtype in [
        GgmlDType::Q4K,
        GgmlDType::Q6K,
        GgmlDType::Q8_0,
        GgmlDType::Q2K,
    ] {
        let values: Vec<f32> = (0..512).map(|index| (index as f32 * 0.03).sin()).collect();
        let source = Tensor::from_vec(values, (2, 256), &Device::Cpu).unwrap();
        let host = QTensor::quantize(&source, dtype).unwrap();
        let expected_bytes = host.data().unwrap().into_owned();
        let regular = QTensor::new(
            QStorage::from_data(Cow::Borrowed(&expected_bytes), &device, dtype).unwrap(),
            (2, 256),
        )
        .unwrap();
        let input = Tensor::ones((1, 256), candle_core::DType::F32, &device).unwrap();
        let upload_device = device.clone();
        let bytes = expected_bytes.clone();
        let uploaded = std::thread::spawn(move || {
            let storage =
                QStorage::from_data_transfer(Cow::Owned(bytes), &upload_device, dtype).unwrap();
            QTensor::new(storage, (2, 256)).unwrap()
        })
        .join()
        .unwrap();
        assert!(uploaded.device().same_device(&device));
        let uploaded = Arc::new(uploaded);
        let actual = QMatMul::QTensor(Arc::clone(&uploaded))
            .forward(&input)
            .unwrap();
        let regular = QMatMul::from_qtensor(regular)
            .unwrap()
            .forward(&input)
            .unwrap();
        assert_eq!(
            actual
                .to_device(&Device::Cpu)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            regular
                .to_device(&Device::Cpu)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
        );
        assert_eq!(uploaded.data().unwrap().as_ref(), expected_bytes);
    }
}

#[test]
fn cuda_tiny_cache_prefill_decode_and_shared_sessions() {
    let device = cuda();
    let dir = common::model_dir("cuda-expert-cache");
    let path = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&path);
    let template = load(&path, &device, Some(3 * SLOT_BYTES));
    let report = template.device_expert_cache().unwrap();
    assert_eq!(report.slots, 3);
    assert_eq!(report.resident, 0);
    let mut first = template.new_session().unwrap();
    let mut second = template.new_session().unwrap();
    let mut first_host = load(&path, &Device::Cpu, None);
    let mut second_host = first_host.new_session().unwrap();
    std::thread::scope(|scope| {
        for (cached, host, prompt) in [
            (&mut first, &mut first_host, [1u32, 4, 2, 7, 5]),
            (&mut second, &mut second_host, [5u32, 7, 2, 4, 1]),
        ] {
            scope.spawn(move || {
                cached.set_pin_hot_experts(2);
                assert_close(&logits(cached, &prompt, 0), &logits(host, &prompt, 0));
                cached.wait_for_expert_uploads();
                for step in 0..70 {
                    let token = if step % 7 == 2 { 8 } else { 3 };
                    assert_close(
                        &logits(cached, &[token], prompt.len() + step),
                        &logits(host, &[token], prompt.len() + step),
                    );
                    cached.wait_for_expert_uploads();
                    let report = cached.device_expert_cache().unwrap();
                    assert!(report.resident_bytes <= report.budget_bytes, "{report:?}");
                    assert!(report.resident <= 3, "{report:?}");
                }
                assert_close(
                    &logits(cached, &[3, 8, 3], prompt.len() + 70),
                    &logits(host, &[3, 8, 3], prompt.len() + 70),
                );
                cached.wait_for_expert_uploads();
            });
        }
    });
    let report = template.device_expert_cache().unwrap();
    eprintln!("CUDA shared tiny cache: {report:?}");
    assert!(
        report.stats.uploads > 0 && report.stats.hits > 0,
        "{report:?}"
    );
    assert_eq!(report.stats.upload_failures, 0, "{report:?}");
    assert_eq!(report.stats, first.device_expert_cache().unwrap().stats);
    assert_eq!(report.stats, second.device_expert_cache().unwrap().stats);
    let mut zero = load(&path, &device, Some(0));
    let mut host = load(&path, &Device::Cpu, None);
    assert!(zero.device_expert_cache().is_none());
    assert_eq!(
        logits(&mut zero, &[1, 4, 2], 0),
        logits(&mut host, &[1, 4, 2], 0)
    );
    assert_eq!(logits(&mut zero, &[3], 3), logits(&mut host, &[3], 3));
    let mut too_small = load(&path, &device, Some(SLOT_BYTES - 1));
    assert!(too_small.device_expert_cache().is_none());
    host.clear_kv_cache();
    assert_eq!(
        logits(&mut too_small, &[1, 4, 2], 0),
        logits(&mut host, &[1, 4, 2], 0)
    );
    std::fs::remove_dir_all(dir).unwrap();
}
