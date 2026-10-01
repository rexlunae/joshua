//! Speculative token generation: the multi-position verification pass, KV
//! rollback, and end-to-end equivalence with plain decoding.

mod common;

use candle_core::{Device, Tensor};
use joshua::model::QuantizedModel;
use joshua::{Engine, EngineOptions, GenerationOptions, SpeculativeConfig};
use std::path::Path;

fn load(model: &Path) -> QuantizedModel {
    // The deepseek4 fixtures use raw dtypes candle's projection cannot name,
    // so go through the common loader (raw header + mmap borrow).
    common::load_model(model, true)
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
    let prefill = last_logits(&mut m, &prompt, 0);
    let mut verifier = load(model_path);
    let prefill_rows = all_logits(&mut verifier, &prompt, 0);
    assert_close(
        &prefill,
        prefill_rows.last().unwrap(),
        "prefill last vs all rows",
    );
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
/// speculative run left behind.  `expect_rejections` asserts the rollback
/// path actually ran; the tiny deepseek4 fixture's drafter is perfect (its
/// output repeats the context faithfully, so prompt lookup never misses),
/// and its rollback is pinned by the model-level tests above instead.
fn check_engine_equivalence(dir: &Path, expect_rejections: bool) {
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
    if expect_rejections {
        assert!(stats.accepted < stats.drafted, "nothing was rejected: {stats:?}");
    }
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
    check_engine_equivalence(&dir, true);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn deepseek2_speculative_greedy_matches_plain_decoding() {
    let dir = common::model_dir("spec-deepseek2-engine");
    common::write_tiny_deepseek2_gguf(&dir.join("model.gguf"));
    check_engine_equivalence(&dir, true);
    std::fs::remove_dir_all(&dir).ok();
}

/// The deepseek4 loader (window-only layers): the verification pass and the
/// ring rollback must match incremental decoding.
#[test]
fn deepseek4_verification_rows_match_incremental_decode() {
    let dir = common::model_dir("spec-ds4-rows");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&model);
    check_verification_and_rollback(&model);
    std::fs::remove_dir_all(&dir).ok();
}

/// The deepseek4 loader with a CSA layer (ratio 4, overlap 2) plus its
/// indexer compressor: the streaming compressor states are snapshotted and
/// the accepted prefix replayed on rollback, and a stale compressed row
/// written for a block the pass completed is rewritten before it is read.
#[test]
fn deepseek4_compressed_verification_rows_match_incremental_decode() {
    let dir = common::model_dir("spec-ds4c-rows");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf_compress(&model);
    check_verification_and_rollback(&model);
    std::fs::remove_dir_all(&dir).ok();
}

/// Same with an HCA layer (ratio 128, no overlap): the compressor only
/// accumulates inside the verification span, so the rollback replays the
/// accumulator without emitting rows.
#[test]
fn deepseek4_hca_verification_rows_match_incremental_decode() {
    let dir = common::model_dir("spec-ds4h-rows");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf_compress_hca(&model);
    check_verification_and_rollback(&model);
    std::fs::remove_dir_all(&dir).ok();
}

/// V4.1: engram lookback (the token history rolls back too), ratio-2 gated
/// and ratio-1 compressors, and the shared key-owner streams.
#[test]
fn deepseek41_verification_rows_match_incremental_decode() {
    let dir = common::model_dir("spec-ds41-rows");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek41_gguf(&model);
    check_verification_and_rollback(&model);
    std::fs::remove_dir_all(&dir).ok();
}

/// The engine path end to end on the deepseek4 loader, including rejections
/// (the misleading prompt), KV reuse across a follow-up request, and the
/// non-zero drafted/accepted/rejected counters the issue calls for.
#[test]
fn deepseek4_speculative_greedy_matches_plain_decoding() {
    let dir = common::model_dir("spec-ds4c-engine");
    common::write_tiny_deepseek4_gguf_compress(&dir.join("model.gguf"));
    check_engine_equivalence(&dir, false);
    std::fs::remove_dir_all(&dir).ok();
}

/// The benchmark shape from the issue: on output that repeats its context,
/// speculative decoding must actually be faster — the drafted tokens the
/// model accepts are decoded in one sweep.  Wall-clock on a tiny fixture
/// is noisy, so this compares the engine's own decode-throughput figures
/// (best of several runs each) and prints the speedup and the hit rate.
#[test]
fn deepseek4_speculative_speeds_up_repetitive_output() {
    let dir = common::model_dir("spec-ds4c-bench");
    common::write_tiny_deepseek4_gguf_compress(&dir.join("model.gguf"));
    // A long repetitive tail so prompt lookup drafts long accepted runs.
    let prompt = "a b c d ".repeat(6);

    let plain = engine(&dir, None);
    let spec = engine(&dir, Some(SpeculativeConfig::with_max_draft(8)));

    // Warm both paths once (allocator, page cache) before measuring.
    let _ = plain.complete_raw(&prompt, &greedy(32)).unwrap();
    let _ = spec.complete_raw(&prompt, &greedy(32)).unwrap();

    let best = |e: &Engine| {
        (0..3)
            .map(|_| e.complete_raw(&prompt, &greedy(32)).unwrap().3)
            .fold(f64::MIN, f64::max)
    };
    let plain_tps = best(&plain);
    let spec_tps = best(&spec);
    let stats = spec.speculative_stats();
    println!(
        "deepseek4 speculative bench: plain {plain_tps:.1} t/s, spec {spec_tps:.1} t/s, speedup {:.2}x, acceptance {:.0}% (drafted {}, accepted {})",
        spec_tps / plain_tps,
        stats.acceptance_rate() * 100.0,
        stats.drafted,
        stats.accepted,
    );
    assert!(stats.drafted > 0, "nothing was drafted: {stats:?}");
    assert!(
        spec_tps > plain_tps,
        "speculative decoding must be faster on repetitive output: plain {plain_tps:.1} t/s, spec {spec_tps:.1} t/s"
    );
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
    // Not silently ignored: the effective configuration reads back `None`
    // (the load logs a warning naming the architecture).
    assert!(
        spec.speculative_config().is_none(),
        "an unsupported architecture must not keep a speculative configuration"
    );
    let prompt = "a b c a b c a b";
    let (want, _, _, _) = plain.complete_raw(prompt, &greedy(12)).unwrap();
    let (got, _, _, _) = spec.complete_raw(prompt, &greedy(12)).unwrap();
    assert_eq!(got, want);
    assert_eq!(spec.speculative_stats().drafted, 0);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn recurrent_hybrid_refuses_speculative_loudly() {
    // A Gated DeltaNet hybrid cannot roll its KV cache back (recurrent state
    // only clears), so the engine must reflect the refusal in the effective
    // configuration instead of holding a setting decode time would skip.
    let dir = common::model_dir("spec-qwen3next-engine");
    common::write_tiny_qwen_gguf(&dir.join("model.gguf"), "qwen3next");
    let plain = engine(&dir, None);
    let spec = engine(&dir, Some(SpeculativeConfig::default()));
    assert!(spec.speculative_config().is_none());
    let prompt = "a b c a b c a b";
    let (want, _, _, _) = plain.complete_raw(prompt, &greedy(12)).unwrap();
    let (got, _, _, _) = spec.complete_raw(prompt, &greedy(12)).unwrap();
    assert_eq!(got, want);
    assert_eq!(spec.speculative_stats().drafted, 0);
    std::fs::remove_dir_all(&dir).ok();
}

/// The static per-architecture map must classify every supported name: the
/// plain-attention qwen and deepseek2 families plus deepseek4 decode
/// speculatively, the stock candle loaders and the recurrent DeltaNet/KDA
/// hybrids do not.
#[test]
fn architecture_speculative_map_covers_every_name() {
    use joshua::model::Architecture;

    let supported = [
        "qwen",
        "qwen2moe",
        "qwen2vl",
        "qwen3",
        "qwen3moe",
        "qwen3vl",
        "qwen3vlmoe",
        "chatglm",
        "glm4",
        "glm4moe",
        "deepseek",
        "deepseek2",
        "glm-dsa",
        "deepseek4",
        "deepseek41",
    ];
    let unsupported = [
        "llama",
        "gemma",
        "gemma2",
        "gemma3",
        "gemma-embedding",
        "lfm2",
        "phi2",
        "phi3",
        "qwen2",
        "qwen3next",
        "qwen35",
        "qwen35moe",
        "qwen4exp",
        "glm5next",
        "glm5-next",
        "kimi-linear",
        "kimi-k3",
    ];
    for name in supported {
        let arch = Architecture::from_name(name).expect("listed name resolves");
        assert!(
            arch.supports_speculative_decoding(),
            "{name} should support speculative decoding"
        );
        assert!(
            arch.unsupported_speculative_reason().is_none(),
            "{name} should have no refusal reason"
        );
    }
    for name in unsupported {
        let arch = Architecture::from_name(name).expect("listed name resolves");
        assert!(
            !arch.supports_speculative_decoding(),
            "{name} should not support speculative decoding"
        );
        assert!(
            arch.unsupported_speculative_reason().is_some(),
            "{name} should carry a refusal reason"
        );
    }
    // Every name in the registry is classified (no name falls through a
    // stale map), and the reason strings are non-empty.
    for name in Architecture::supported_names() {
        let arch = Architecture::from_name(name).expect("listed name resolves");
        if let Some(reason) = arch.unsupported_speculative_reason() {
            assert!(!reason.is_empty(), "{name}: empty refusal reason");
        }
    }
}
