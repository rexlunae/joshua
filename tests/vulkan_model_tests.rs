#![cfg(feature = "vulkan")]
//! End-to-end Vulkan tests: tiny models run on the Vulkan device (native
//! kernels, quantized weights on device) must agree with the CPU path.  Skips when no Vulkan platform is installed (any device type
//! serves: a GPU, or a CPU runtime such as llvmpipe in CI).
//!
//!   cargo test --features vulkan --test vulkan_model_tests

mod common;

use candle_core::{Device, Tensor};
use joshua::model::QuantizedModel;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

fn vulkan_or_skip() -> Option<Device> {
    match Device::vulkan_if_available(0) {
        Ok(Device::Cpu) => {
            eprintln!("SKIP: no Vulkan device available on this host");
            None
        }
        Ok(dev) => Some(dev),
        Err(e) => {
            eprintln!("SKIP: vulkan init failed: {e}");
            None
        }
    }
}

fn load_heap(model: &Path, device: &Device) -> QuantizedModel {
    let bytes = std::fs::read(model).unwrap();
    let mut cursor = Cursor::new(&bytes[..]);
    let content = candle_core::quantized::gguf_file::Content::read(&mut cursor).unwrap();
    QuantizedModel::from_gguf(content, &mut cursor, device).unwrap()
}

fn load_mmap(model: &Path, device: &Device) -> QuantizedModel {
    let bytes = std::fs::read(model).unwrap();
    let mut cursor = Cursor::new(&bytes[..]);
    // The tolerant header, as the engine reads it: it names the raw dtypes
    // (IQ2_XXS, I32) candle's own `Content::read` rejects.
    let header = joshua::gguf_ext::read_header(&mut cursor).unwrap();
    let content = header.to_candle_content().unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&std::fs::File::open(model).unwrap()) }.unwrap();
    let mut cursor = Cursor::new(&bytes[..]);
    QuantizedModel::from_gguf_mmap(content, &mut cursor, device, Some(Arc::new(mmap)), None, 0)
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

/// The device and CPU paths differ in rounding only: the CPU quantized
/// matmul quantizes activations to 8 bits, the device multiplies f32
/// activations against dequantized blocks.  Compare against the spread of
/// the logits rather than a fixed epsilon.
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
    let arg = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
    };
    assert_eq!(arg(dev), arg(cpu), "{what}: argmax differs");
}

/// Run `tokens` through the tiny model written by `write` on the CPU and on
/// the Vulkan device (heap-loaded when `heap`, and memory-mapped) and compare
/// the prefill and single-token decode logits.
fn run_model(name: &str, write: fn(&Path), tokens: &[u32], heap: bool) {
    let Some(ocl) = vulkan_or_skip() else { return };
    let dir = common::model_dir(&format!("vulkan-{name}"));
    let model = dir.join("model.gguf");
    write(&model);

    let cpu = Device::Cpu;
    let mut ref_model = load_mmap(&model, &cpu);
    let ref_prefill = logits(&mut ref_model, tokens, 0, &cpu);
    let ref_decode = logits(
        &mut ref_model,
        &tokens[tokens.len() - 1..],
        tokens.len(),
        &cpu,
    );

    let mut runs = vec![("mmap", load_mmap(&model, &ocl))];
    if heap {
        runs.push(("heap", load_heap(&model, &ocl)));
    }
    for (path, mut m) in runs {
        let before = candle_core::vulkan_backend::fallback_count();
        let prefill = logits(&mut m, tokens, 0, &ocl);
        assert_close(&format!("{name} {path} prefill"), &prefill, &ref_prefill);
        let decode = logits(&mut m, &tokens[tokens.len() - 1..], tokens.len(), &ocl);
        assert_close(&format!("{name} {path} decode"), &decode, &ref_decode);
        eprintln!(
            "{name} {path}: {} native launches so far, {} fallbacks during this model",
            candle_core::vulkan_backend::native_exec_count(),
            candle_core::vulkan_backend::fallback_count() - before
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn vulkan_qwen3moe_matches_cpu() {
    run_model(
        "qwen3moe",
        common::write_tiny_qwen3moe_gguf,
        &[1, 4, 2, 7, 5],
        true,
    );
}

#[test]
fn vulkan_deepseek2_matches_cpu() {
    run_model(
        "deepseek2",
        common::write_tiny_deepseek2_gguf,
        &[1, 4, 2, 7, 5],
        true,
    );
}

#[test]
fn vulkan_llama_matches_cpu() {
    run_model(
        "llama",
        common::write_tiny_llama_gguf,
        &[1, 4, 2, 7, 5],
        true,
    );
}

#[test]
fn vulkan_deepseek4_matches_cpu() {
    // deepseek4 needs the raw header (IQ2_XXS experts), so only the mmap
    // loader applies; the experts stay on the CPU, the dense set runs on
    // the device.
    run_model(
        "deepseek4",
        common::write_tiny_deepseek4_gguf,
        &[1, 4, 2, 7, 5],
        false,
    );
    run_model(
        "deepseek4-kquant",
        common::write_tiny_deepseek4_gguf_kquant,
        &[1, 4, 2, 7, 5],
        false,
    );
}
