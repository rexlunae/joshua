//! Speculative token generation: the multi-position verification pass, KV
//! rollback, and end-to-end equivalence with plain decoding.

mod common;

use candle_core::{Device, Tensor};
use joshua::model::QuantizedModel;
use joshua::{Engine, EngineOptions, GenerationOptions, SpeculativeConfig};
use std::io::Cursor;
use std::path::Path;

fn load(model: &Path) -> QuantizedModel {
    let bytes = std::fs::read(model).unwrap();
    let mut cursor = Cursor::new(&bytes[..]);
    let content = candle_core::quantized::gguf_file::Content::read(&mut cursor).unwrap();
    QuantizedModel::from_gguf(content, &mut cursor, &Device::Cpu).unwrap()
}

fn input(tokens: &[u32]) -> Tensor {
    Tensor::new(tokens, &Device::Cpu)
        .unwrap()
        .reshape((1, tokens.len()))
        .unwrap()
}

fn last_logits(model: &mut QuantizedModel, tokens: &[u32], offset: usize) -> Vec<f32> {
    model
        .forward(&input(tokens), offset)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

fn all_logits(model: &mut QuantizedModel, tokens: &[u32], offset: usize) -> Vec<Vec<f32>> {
    let out = model.forward_all_logits(&input(tokens), offset).unwrap();
    assert_eq!(out.dims()[..2], [1, tokens.len()]);
    out.squeeze(0).unwrap().to_vec2().unwrap()
}

fn assert_close(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() < 1e-4, "{what}: logit {i}: {x} vs {y}");
    }
}

/// Every row of a verification pass must equal the logits one-token decode
/// produces at that position, and truncating the rejected tail must leave a
/// cache that continues exactly like one that never saw it.
fn check_verification_and_rollback(model_path: &Path) {
    let prompt = [1u32, 4, 2, 7, 5];
    let block = [6u32, 9, 11, 4];

    let mut m = load(model_path);
    assert!(m.supports_speculative());
    last_logits(&mut m, &prompt, 0);
    let rows = all_logits(&mut m, &block, prompt.len());

    let mut reference = load(model_path);
    last_logits(&mut reference, &prompt, 0);
    for (i, &t) in block.iter().enumerate() {
        let want = last_logits(&mut reference, &[t], prompt.len() + i);
        assert_close(&rows[i], &want, &format!("row {i}"));
    }

    // Keep the first two block tokens, drop the rest, and continue with a
    // different token: must match a cache that only ever saw the prefix.
    m.truncate_kv_cache(prompt.len() + 2).unwrap();
    let got = last_logits(&mut m, &[13], prompt.len() + 2);
    let mut fresh = load(model_path);
    last_logits(&mut fresh, &prompt, 0);
    last_logits(&mut fresh, &block[..2], prompt.len());
    let want = last_logits(&mut fresh, &[13], prompt.len() + 2);
    assert_close(&got, &want, "after rollback");
}

#[test]
fn qwen3moe_verification_rows_match_incremental_decode() {
    let dir = common::model_dir("spec-qwen3moe-rows");
    let model = dir.join("model.gguf");
    common::write_tiny_qwen3moe_gguf(&model);
    check_verification_and_rollback(&model);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn deepseek2_verification_rows_match_incremental_decode() {
    let dir = common::model_dir("spec-deepseek2-rows");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek2_gguf(&model);
    check_verification_and_rollback(&model);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn dense_candle_models_do_not_claim_speculative_support() {
    let dir = common::model_dir("spec-llama");
    let model = dir.join("model.gguf");
    common::write_tiny_llama_gguf(&model);
    let m = load(&model);
    assert!(!m.supports_speculative());
    std::fs::remove_dir_all(&dir).ok();
}

fn greedy(max_tokens: u32) -> GenerationOptions {
    GenerationOptions {
        max_tokens,
        temperature: 0.0,
        ..GenerationOptions::default()
    }
}

fn engine(dir: &Path, speculative: Option<SpeculativeConfig>) -> Engine {
    Engine::with_options(
        dir,
        EngineOptions::with_n_ctx(128).speculative(speculative),
    )
    .expect("engine should load tiny model")
}

/// Greedy speculative decoding must be token-for-token identical to plain
/// decoding, across a follow-up request that reuses the pooled KV cache the
/// speculative run left behind.
fn check_engine_equivalence(dir: &Path) {
    // A repetitive prompt so prompt lookup has n-grams to match.
    let prompt = "a b c d a b c d a b c d a b c d a b";
    let config = SpeculativeConfig {
        max_draft: 4,
        max_ngram: 3,
        min_ngram: 1,
    };

    let plain = engine(dir, None);
    let spec = engine(dir, Some(config));
    assert_eq!(spec.speculative_config(), Some(config));

    let (want, want_usage, _, _) = plain.complete_raw(prompt, &greedy(24)).unwrap();
    let (got, got_usage, _, _) = spec.complete_raw(prompt, &greedy(24)).unwrap();
    assert_eq!(got, want, "speculative output diverged from plain decoding");
    assert_eq!(got_usage.completion_tokens, want_usage.completion_tokens);

    // A prompt whose history predicts a continuation the model does not
    // produce, so drafts get rejected and the KV cache rolled back.
    let misleading = "a b k c d e k c d e a b";
    let (want_m, _, _, _) = plain.complete_raw(misleading, &greedy(24)).unwrap();
    let (got_m, _, _, _) = spec.complete_raw(misleading, &greedy(24)).unwrap();
    assert_eq!(got_m, want_m, "speculative output diverged after rejections");

    let stats = spec.speculative_stats();
    assert!(stats.drafted > 0, "nothing was drafted: {stats:?}");
    assert!(stats.accepted > 0, "nothing was accepted: {stats:?}");
    assert!(stats.accepted < stats.drafted, "nothing was rejected: {stats:?}");
    assert!(stats.verify_steps > 0);
    assert_eq!(plain.speculative_stats().drafted, 0);

    // Follow-up turn extending prompt + response: the speculative engine
    // continues from its pooled cache, which must hold exactly the tokens
    // it reported.
    let follow = format!("{prompt} {got} c d");
    let (want2, _, _, _) = plain.complete_raw(&follow, &greedy(16)).unwrap();
    let (got2, _, _, _) = spec.complete_raw(&follow, &greedy(16)).unwrap();
    assert_eq!(got2, want2, "follow-up diverged after speculative decoding");
}

#[test]
fn qwen3moe_speculative_greedy_matches_plain_decoding() {
    let dir = common::model_dir("spec-qwen3moe-engine");
    common::write_tiny_qwen3moe_gguf(&dir.join("model.gguf"));
    check_engine_equivalence(&dir);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn deepseek2_speculative_greedy_matches_plain_decoding() {
    let dir = common::model_dir("spec-deepseek2-engine");
    common::write_tiny_deepseek2_gguf(&dir.join("model.gguf"));
    check_engine_equivalence(&dir);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn speculative_sampling_runs_and_respects_max_tokens() {
    let dir = common::model_dir("spec-qwen3moe-sampled");
    common::write_tiny_qwen3moe_gguf(&dir.join("model.gguf"));
    let spec = engine(&dir, Some(SpeculativeConfig::with_max_draft(6)));
    let options = GenerationOptions {
        max_tokens: 20,
        temperature: 0.8,
        ..GenerationOptions::default()
    };
    for _ in 0..4 {
        let (_, usage, _, _) = spec
            .complete_raw("a b c a b c a b c a b", &options)
            .unwrap();
        assert!(usage.completion_tokens <= 20);
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn unsupported_architecture_falls_back_to_plain_decoding() {
    let dir = common::model_dir("spec-llama-engine");
    common::write_tiny_llama_gguf(&dir.join("model.gguf"));
    let plain = engine(&dir, None);
    let spec = engine(&dir, Some(SpeculativeConfig::default()));
    let prompt = "a b c a b c a b";
    let (want, _, _, _) = plain.complete_raw(prompt, &greedy(12)).unwrap();
    let (got, _, _, _) = spec.complete_raw(prompt, &greedy(12)).unwrap();
    assert_eq!(got, want);
    assert_eq!(spec.speculative_stats().drafted, 0);
    std::fs::remove_dir_all(&dir).ok();
}
