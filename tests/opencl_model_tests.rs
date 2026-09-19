#![cfg(feature = "opencl")]
//! End-to-end OpenCL tests: tiny models run on the OpenCL device (native
//! kernels, quantized weights on device, zero-copy mmap) must agree with the
//! CPU path.  Skips when no OpenCL platform is installed (any device type
//! serves: a GPU, or a CPU runtime such as pocl in CI).
//!
//!   cargo test --features opencl --test opencl_model_tests

mod common;

use candle_core::{Device, Tensor};
use joshua::model::QuantizedModel;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

/// The one OpenCL device every test in this binary shares, as a server
/// shares one device across its sessions.  Opened once: the tests run on
/// parallel threads, and creating and tearing down a context per test while
/// other threads run kernels is a configuration no real user has (and one
/// that trips reference-counting bugs in pocl).
fn opencl_or_skip() -> Option<Device> {
    static DEVICE: std::sync::OnceLock<Option<Device>> = std::sync::OnceLock::new();
    DEVICE
        .get_or_init(|| match Device::opencl_if_available(0) {
            Ok(Device::Cpu) => {
                eprintln!("SKIP: no OpenCL device available on this host");
                None
            }
            Ok(dev) => Some(dev),
            Err(e) => {
                eprintln!("SKIP: opencl init failed: {e}");
                None
            }
        })
        .clone()
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
/// the OpenCL device (heap-loaded when `heap`, and memory-mapped) and compare
/// the prefill and single-token decode logits.
fn run_model(name: &str, write: fn(&Path), tokens: &[u32], heap: bool) {
    let Some(ocl) = opencl_or_skip() else { return };
    let dir = common::model_dir(&format!("opencl-{name}"));
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
        let before = candle_core::opencl_backend::fallback_count();
        let prefill = logits(&mut m, tokens, 0, &ocl);
        assert_close(&format!("{name} {path} prefill"), &prefill, &ref_prefill);
        let decode = logits(&mut m, &tokens[tokens.len() - 1..], tokens.len(), &ocl);
        assert_close(&format!("{name} {path} decode"), &decode, &ref_decode);
        eprintln!(
            "{name} {path}: {} native launches so far, {} fallbacks during this model",
            candle_core::opencl_backend::native_exec_count(),
            candle_core::opencl_backend::fallback_count() - before
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn opencl_qwen3moe_matches_cpu() {
    run_model(
        "qwen3moe",
        common::write_tiny_qwen3moe_gguf,
        &[1, 4, 2, 7, 5],
        true,
    );
}

#[test]
fn opencl_deepseek2_matches_cpu() {
    run_model(
        "deepseek2",
        common::write_tiny_deepseek2_gguf,
        &[1, 4, 2, 7, 5],
        true,
    );
}

#[test]
fn opencl_llama_matches_cpu() {
    run_model(
        "llama",
        common::write_tiny_llama_gguf,
        &[1, 4, 2, 7, 5],
        true,
    );
}

#[test]
fn opencl_deepseek4_matches_cpu() {
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

/// Long-prompt parity: at prompt lengths where real-model long-context
/// degradation was observed, the OpenCL path must track the CPU path —
/// same logits (within device rounding) and same greedy continuation.
/// Complements the short-prompt `run_model` cases, which cannot reach the
/// window/compressed-cache paths that only engage with longer prompts.
#[test]
fn opencl_deepseek4_long_prompt_matches_cpu() {
    let Some(ocl) = opencl_or_skip() else { return };
    let dir = common::model_dir("opencl-ds4-longprompt");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&model);

    // ~200 prompt tokens cycling the tiny model's vocab (ids 1..=15).
    let tokens: Vec<u32> = (0..200).map(|i| 1 + (i % 15) as u32).collect();

    let cpu = Device::Cpu;
    let mut ref_model = load_mmap(&model, &cpu);
    let ref_prefill = logits(&mut ref_model, &tokens, 0, &cpu);
    let ref_decode = logits(
        &mut ref_model,
        &[tokens[tokens.len() - 1]],
        tokens.len(),
        &cpu,
    );

    let mut dev_model = load_mmap(&model, &ocl);
    let before = candle_core::opencl_backend::fallback_count();
    let dev_prefill = logits(&mut dev_model, &tokens, 0, &ocl);
    assert_close("opencl long-prompt prefill", &dev_prefill, &ref_prefill);
    let dev_decode = logits(
        &mut dev_model,
        &[tokens[tokens.len() - 1]],
        tokens.len(),
        &ocl,
    );
    assert_close("opencl long-prompt decode", &dev_decode, &ref_decode);
    eprintln!(
        "opencl long prompt: {} native launches, {} fallbacks",
        candle_core::opencl_backend::native_exec_count(),
        candle_core::opencl_backend::fallback_count() - before
    );

    // Greedy continuation must agree too: 8 tokens of argmax decode on each
    // device must pick the same ids.
    let mut cpu_ids = tokens.clone();
    let mut dev_ids = tokens.clone();
    let mut cpu_last = ref_decode.clone();
    let mut dev_last = dev_decode.clone();
    for step in 0..8 {
        let cpu_next = cpu_last
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        let dev_next = dev_last
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        assert_eq!(cpu_next, dev_next, "greedy step {step}: argmax diverged");
        cpu_ids.push(cpu_next);
        dev_ids.push(dev_next);
        cpu_last = logits(&mut ref_model, &[cpu_next], cpu_ids.len() - 1, &cpu);
        dev_last = logits(&mut dev_model, &[dev_next], dev_ids.len() - 1, &ocl);
    }

    std::fs::remove_dir_all(&dir).ok();
}
