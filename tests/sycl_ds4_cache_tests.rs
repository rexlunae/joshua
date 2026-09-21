#![cfg(all(feature = "sycl", target_os = "linux"))]

mod common;

use candle_core::{Device, Tensor};
use joshua::model::QuantizedModel;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

const SLOT_BYTES: u64 = 2 * 128 * 66 + 1024 * 34;

fn sycl_or_skip() -> Option<Device> {
    match Device::new_sycl(0) {
        Ok(dev) => Some(dev),
        Err(e) => {
            eprintln!("SKIP: no SYCL device/bridge available: {e}");
            None
        }
    }
}

fn load(
    model: &Path,
    device: &Device,
    expert_device: &Device,
    cache_bytes: Option<u64>,
) -> QuantizedModel {
    let bytes = std::fs::read(model).unwrap();
    let mut cursor = Cursor::new(&bytes[..]);
    let header = joshua::gguf_ext::read_header(&mut cursor).unwrap();
    let content = header.to_candle_content().unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&std::fs::File::open(model).unwrap()) }.unwrap();
    let mut cursor = Cursor::new(&bytes[..]);
    QuantizedModel::from_gguf_mmap_placed(
        content,
        &mut cursor,
        device,
        expert_device,
        Some(Arc::new(mmap)),
        None,
        0,
        cache_bytes,
    )
    .unwrap()
}

fn logits(model: &mut QuantizedModel, tokens: &[u32], offset: usize, device: &Device) -> Vec<f32> {
    let input = Tensor::new(tokens, device)
        .unwrap()
        .reshape((1, tokens.len()))
        .unwrap();
    model
        .forward(&input, offset)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

fn assert_close(what: &str, dev: &[f32], cpu: &[f32]) {
    assert_eq!(dev.len(), cpu.len(), "{what}: logit count");
    assert!(
        dev.iter().all(|v| v.is_finite()),
        "{what}: device logits not finite: {dev:?}"
    );
    let lo = cpu.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = cpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let spread = (hi - lo).max(1e-3);
    let worst = dev
        .iter()
        .zip(cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        worst <= 0.05 * spread,
        "{what}: device logits diverge from cpu by {worst} (spread {spread}):\n dev {dev:?}\n cpu {cpu:?}"
    );
}

#[test]
fn deepseek4_sycl_vram_expert_cache_matches_cpu() {
    let Some(sycl) = sycl_or_skip() else { return };
    let dir = common::model_dir("sycl-ds4-cache");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&model);

    let mut plain = load(&model, &Device::Cpu, &Device::Cpu, None);
    let mut cached = load(&model, &Device::Cpu, &sycl, Some(3 * SLOT_BYTES));

    let report = cached
        .device_expert_cache()
        .expect("SYCL expert placement should build a device cache");
    assert_eq!(
        report.slots, 3,
        "exact per-expert byte accounting: {report:?}"
    );

    let tokens = [1u32, 4, 2, 7, 5];
    assert_close(
        "sycl ds4 prefill",
        &logits(&mut cached, &tokens, 0, &Device::Cpu),
        &logits(&mut plain, &tokens, 0, &Device::Cpu),
    );
    cached.wait_for_expert_uploads();
    let after_prefill = cached.device_expert_cache().unwrap();
    assert!(
        after_prefill.resident >= 1,
        "prefill should seed the SYCL cache: {after_prefill:?}"
    );

    for (i, t) in [3u32, 3, 3, 3, 8, 3, 3].iter().enumerate() {
        assert_close(
            &format!("sycl ds4 decode step {i}"),
            &logits(&mut cached, &[*t], tokens.len() + i, &Device::Cpu),
            &logits(&mut plain, &[*t], tokens.len() + i, &Device::Cpu),
        );
        cached.wait_for_expert_uploads();
    }

    let final_report = cached.device_expert_cache().unwrap();
    assert!(
        final_report.stats.hits > 0,
        "resident experts should be reused on SYCL: {final_report:?}"
    );
    assert!(
        final_report.stats.uploads >= 1,
        "the SYCL cache should upload experts: {final_report:?}"
    );

    std::fs::remove_dir_all(&dir).ok();
}
