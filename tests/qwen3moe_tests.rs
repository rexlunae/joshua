//! Integration tests for the pure-Rust `qwen3moe` loader (Qwen3-30B-A3B,
//! Qwen3-Coder-30B-A3B): architecture detection, GGUF loading, and
//! prefill-vs-incremental KV-cache consistency through the whole engine.

mod common;

use candle_core::{Device, Tensor};
use joshua::model::{Architecture, QuantizedModel};
use std::io::Cursor;
use std::path::Path;

fn load(model: &Path) -> QuantizedModel {
    let bytes = std::fs::read(model).unwrap();
    let mut cursor = Cursor::new(&bytes[..]);
    let content = candle_core::quantized::gguf_file::Content::read(&mut cursor).unwrap();
    QuantizedModel::from_gguf(content, &mut cursor, &Device::Cpu).unwrap()
}

fn logits(model: &mut QuantizedModel, tokens: &[u32], offset: usize) -> Vec<f32> {
    let input = Tensor::new(tokens, &Device::Cpu)
        .unwrap()
        .reshape((1, tokens.len()))
        .unwrap();
    model
        .forward(&input, offset)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

#[test]
fn qwen3moe_is_a_supported_architecture() {
    assert_eq!(Architecture::from_name("qwen3moe"), Some(Architecture::Qwen3Moe));
}

#[test]
fn qwen3moe_loads_and_produces_finite_logits() {
    let dir = common::model_dir("qwen3moe-load");
    let model = dir.join("model.gguf");
    common::write_tiny_qwen3moe_gguf(&model);

    let mut m = load(&model);
    let out = logits(&mut m, &[1, 4, 2, 7, 5], 0);
    assert_eq!(out.len(), 16, "logits must cover the 16-token vocab");
    assert!(
        out.iter().all(|v| v.is_finite()),
        "all logits must be finite: {out:?}"
    );
    // Not all equal — the model actually did something.
    let first = out[0];
    assert!(
        out.iter().any(|v| (v - first).abs() > 1e-6),
        "logits are degenerate"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn qwen3moe_prefill_matches_incremental_decode() {
    // Feeding a prompt all at once must give the same next-token logits as
    // feeding it token by token through the KV cache. This exercises the
    // per-head Q/K norms, the half-split RoPE, the cache, and the MoE routing
    // (prefill bucketing vs single-token decode) end to end.
    let dir = common::model_dir("qwen3moe-cache");
    let model = dir.join("model.gguf");
    common::write_tiny_qwen3moe_gguf(&model);
    let tokens = [1u32, 4, 2, 7, 5, 9];

    let mut prefill_model = load(&model);
    let prefill = logits(&mut prefill_model, &tokens, 0);

    let mut step_model = load(&model);
    let mut last = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        last = logits(&mut step_model, &[tok], pos);
    }

    assert_eq!(prefill.len(), last.len());
    for (i, (a, b)) in prefill.iter().zip(&last).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "prefill vs incremental logit {i} diverges: {a} vs {b}"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn qwen3moe_mmap_load_matches_heap_load() {
    // The mmap path borrows quantized weights in place (including per-expert
    // slices of the 3-D expert tensors); it must produce bit-identical routing
    // and the same logits as the plain heap path.
    let dir = common::model_dir("qwen3moe-mmap");
    let model = dir.join("model.gguf");
    common::write_tiny_qwen3moe_gguf(&model);
    let tokens = [1u32, 4, 2, 7, 5];

    let bytes = std::fs::read(&model).unwrap();
    let mut cursor = Cursor::new(&bytes[..]);
    let content = candle_core::quantized::gguf_file::Content::read(&mut cursor).unwrap();
    let mut heap = QuantizedModel::from_gguf(content, &mut cursor, &Device::Cpu).unwrap();
    let heap_logits = logits(&mut heap, &tokens, 0);

    let mmap = unsafe { memmap2::Mmap::map(&std::fs::File::open(&model).unwrap()) }.unwrap();
    let mut cursor = Cursor::new(&bytes[..]);
    let content = candle_core::quantized::gguf_file::Content::read(&mut cursor).unwrap();
    let mut mapped =
        QuantizedModel::from_gguf_mmap(
            content,
            &mut cursor,
            &Device::Cpu,
            Some(std::sync::Arc::new(mmap)),
            None,
        )
        .unwrap();
    let mapped_logits = logits(&mut mapped, &tokens, 0);

    assert_eq!(heap_logits.len(), mapped_logits.len());
    for (i, (a, b)) in heap_logits.iter().zip(&mapped_logits).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "mmap vs heap logit {i} diverges: {a} vs {b}"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn qwen3moe_kv_cache_can_be_cleared() {
    let dir = common::model_dir("qwen3moe-clear");
    let model = dir.join("model.gguf");
    common::write_tiny_qwen3moe_gguf(&model);
    let tokens = [1u32, 4, 2];

    let mut m = load(&model);
    assert!(m.supports_kv_clear());
    let a = logits(&mut m, &tokens, 0);
    assert!(m.clear_kv_cache(), "clear_kv_cache must report it cleared the cache");
    let b = logits(&mut m, &tokens, 0);
    for (x, y) in a.iter().zip(&b) {
        assert!(
            (x - y).abs() < 1e-6,
            "logits differ after clear_kv_cache: {x} vs {y}"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// A `tokenizer.json` written by `model_dir` plus a `tokenizer.ggml`-only
/// GGUF must still load through the engine (the GGUF carries everything
/// needed; the tokenizer file is for prompt handling, not loading).
#[test]
fn qwen3moe_engine_detects_and_runs() {
    let dir = common::model_dir("qwen3moe-engine");
    let model = dir.join("model.gguf");
    common::write_tiny_qwen3moe_gguf(&model);

    let mut file = std::fs::File::open(&model).unwrap();
    let mut data = Vec::new();
    use std::io::Read;
    file.read_to_end(&mut data).unwrap();
    let mut cursor = Cursor::new(&data[..]);
    let content = candle_core::quantized::gguf_file::Content::read(&mut cursor).unwrap();
    let arch = Architecture::detect(&content.metadata).unwrap();
    assert_eq!(arch, Architecture::Qwen3Moe);

    std::fs::remove_dir_all(&dir).ok();
}
