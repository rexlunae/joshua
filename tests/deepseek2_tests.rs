//! Native (pure-Rust) validation for the `deepseek2` quantized loader:
//! DeepSeek-V2/V3 and Kimi-K2. These run on the default `cargo test` — no
//! llama.cpp, no network. A numeric cross-check against llama.cpp lives in
//! `llamacpp_adapter_tests.rs` (gated on the built adapter plugin).

mod common;

use candle_core::{Device, Tensor};
use joshua::model::{Architecture, QuantizedModel};

fn load(model: &std::path::Path) -> QuantizedModel {
    let bytes = std::fs::read(model).unwrap();
    let mut cursor = std::io::Cursor::new(&bytes[..]);
    let content = candle_core::quantized::gguf_file::Content::read(&mut cursor).unwrap();
    QuantizedModel::from_gguf(content, &mut cursor, &Device::Cpu).unwrap()
}

fn logits(model: &mut QuantizedModel, tokens: &[u32], offset: usize) -> Vec<f32> {
    let input = Tensor::new(tokens, &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    model
        .forward(&input, offset)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_vec1()
        .unwrap()
}

#[test]
fn deepseek2_is_a_supported_architecture() {
    // No longer reported as "known but unimplemented".
    assert_eq!(Architecture::from_name("deepseek2"), Some(Architecture::DeepSeek2));
}

#[test]
fn deepseek2_loads_and_produces_finite_logits() {
    let dir = common::model_dir("deepseek2-load");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek2_gguf(&model);

    let mut m = load(&model);
    let out = logits(&mut m, &[1, 4, 2, 7, 5], 0);
    assert_eq!(out.len(), 16, "logits must cover the 16-token vocab");
    assert!(
        out.iter().all(|v| v.is_finite()),
        "all logits must be finite: {out:?}"
    );
    // Not all equal — the model actually did something.
    let first = out[0];
    assert!(out.iter().any(|v| (v - first).abs() > 1e-6), "logits are degenerate");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn deepseek2_prefill_matches_incremental_decode() {
    // Feeding a prompt all at once must give the same next-token logits as
    // feeding it token by token through the KV cache. This exercises MLA
    // attention, the RoPE split, and the cache across the MoE + dense layers.
    let dir = common::model_dir("deepseek2-cache");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek2_gguf(&model);
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
fn deepseek2_mla_split_matches_legacy_form() {
    // The two GGUF encodings of the KV up-projection — legacy combined
    // `attn_kv_b` and modern pre-split `attn_k_b`/`attn_v_b` (Kimi-K2) — are
    // written from the *same* source weights, so they are the same model.
    // Joshua reconstructs the combined projection from the split tensors; its
    // logits must match the legacy path exactly. Since the legacy path is
    // cross-checked against llama.cpp, this validates the reconstruction
    // numerically without depending on llama.cpp's (tiny-model-fragile) MLA.
    let dir = common::model_dir("deepseek2-equiv");
    let legacy = dir.join("legacy.gguf");
    let split = dir.join("split.gguf");
    common::write_tiny_deepseek2_gguf(&legacy);
    common::write_tiny_deepseek2_mla_gguf(&split);
    let tokens = [1u32, 4, 2, 7, 5];

    let legacy_logits = logits(&mut load(&legacy), &tokens, 0);
    let split_logits = logits(&mut load(&split), &tokens, 0);

    assert_eq!(legacy_logits.len(), split_logits.len());
    for (i, (a, b)) in legacy_logits.iter().zip(&split_logits).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "split vs legacy KV encoding diverges at logit {i}: {a} vs {b}"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn deepseek2_kv_cache_clear_allows_reuse() {
    let dir = common::model_dir("deepseek2-reset");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek2_gguf(&model);

    let mut m = load(&model);
    assert!(m.supports_kv_clear());
    let first = logits(&mut m, &[1, 4, 2], 0);
    assert!(m.clear_kv_cache());
    let again = logits(&mut m, &[1, 4, 2], 0);
    for (a, b) in first.iter().zip(&again) {
        assert!((a - b).abs() < 1e-4, "reset run diverged: {a} vs {b}");
    }

    std::fs::remove_dir_all(&dir).ok();
}
