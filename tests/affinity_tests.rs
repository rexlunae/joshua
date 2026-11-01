mod common;

use joshua::{types::GenerationOptions, Engine};

#[test]
fn unscoped_default_and_explicit_contexts_reuse_compatible_prefixes() {
    let directory = common::model_dir("affinity-unscoped-context");
    common::write_tiny_qwen3_layers(&directory.join("model.gguf"), 1);
    for (initial, continuation) in [(None, Some(64)), (Some(64), None), (Some(64), Some(96))] {
        let engine = Engine::with_n_ctx(&directory, 128).unwrap();
        let fresh = Engine::with_n_ctx(&directory, 128).unwrap();
        let mut options = GenerationOptions {
            context_window: initial,
            max_tokens: 0,
            temperature: 0.0,
            repetition_penalty: 1.0,
            ..Default::default()
        };
        engine.complete_raw("hello a b c", &options).unwrap();
        options.context_window = continuation;
        options.max_tokens = 4;
        let prompt = "hello a b c world";
        assert_eq!(
            engine.complete_raw(prompt, &options).unwrap().0,
            fresh.complete_raw(prompt, &options).unwrap().0
        );
        assert_eq!(
            engine.kv_reuse_count(),
            1,
            "{initial:?} -> {continuation:?}"
        );
    }
}

#[test]
fn scoped_snapshot_residency_survives_pool_eviction_with_a_bounded_report() {
    let directory = common::model_dir("affinity-snapshot-residency");
    common::write_tiny_qwen3_layers(&directory.join("model.gguf"), 1);
    let options = joshua::EngineOptions::with_n_ctx(128)
        .prefix_cache_bytes(1 << 20)
        .prefix_cache_root_min(1);
    let engine = Engine::with_options(&directory, options).unwrap();
    let mut options = GenerationOptions {
        cache_scope: Some([1; 32]),
        max_tokens: 0,
        temperature: 0.0,
        repetition_penalty: 1.0,
        ..Default::default()
    };
    for scope in 1..10 {
        options.cache_scope = Some([scope; 32]);
        engine.complete_raw("hello a b c", &options).unwrap();
    }
    assert!(engine.prefix_cache_stats().entries > 0);
    let report = engine.cache_residency(128);
    assert!(report.complete);
    let first = "01".repeat(32);
    assert!(report
        .entries
        .iter()
        .any(|entry| entry.scope == first && entry.tokens > 0));
    let bounded = engine.cache_residency(1);
    assert_eq!(bounded.entries.len(), 1);
    assert!(!bounded.complete);
    options.cache_scope = Some([1; 32]);
    options.max_tokens = 4;
    let before = engine.prefix_cache_stats().hits;
    let got = engine.complete_raw("hello a b c world", &options).unwrap();
    let fresh = Engine::with_n_ctx(&directory, 128).unwrap();
    assert_eq!(
        got.0,
        fresh.complete_raw("hello a b c world", &options).unwrap().0
    );
    assert_eq!(engine.prefix_cache_stats().hits, before + 1);
    assert!(engine.prefix_cache_stats().bytes <= engine.prefix_cache_stats().budget);
}

#[test]
fn scoped_continuation_edit_context_and_concurrency_match_fresh_prefill() {
    let directory = common::model_dir("affinity-scopes");
    common::write_tiny_qwen3_layers(&directory.join("model.gguf"), 1);
    let engine = std::sync::Arc::new(Engine::with_n_ctx(&directory, 128).unwrap());
    let fresh = Engine::with_n_ctx(&directory, 128).unwrap();
    assert_ne!(engine.cache_epoch(), fresh.cache_epoch());
    let mut options = GenerationOptions {
        cache_scope: Some([1; 32]),
        max_tokens: 0,
        temperature: 0.0,
        repetition_penalty: 1.0,
        ..Default::default()
    };
    engine.complete_raw("hello a b c", &options).unwrap();
    options.max_tokens = 4;
    let continuation = "hello a b c d world";
    let got = engine.complete_raw(continuation, &options).unwrap();
    let want = fresh.complete_raw(continuation, &options).unwrap();
    assert_eq!(got.0, want.0);
    assert_eq!(got.1.prompt_tokens, want.1.prompt_tokens);
    assert_eq!(engine.kv_reuse_count(), 1);
    let edited = "hello a b hello world";
    assert_eq!(
        engine.complete_raw(edited, &options).unwrap().0,
        fresh.complete_raw(edited, &options).unwrap().0
    );
    assert_eq!(engine.kv_edit_reuse_count(), 1);
    let before = engine.kv_reuse_count();
    options.cache_scope = Some([2; 32]);
    assert_eq!(
        engine.complete_raw(continuation, &options).unwrap().0,
        want.0
    );
    assert_eq!(engine.kv_reuse_count(), before);
    options.context_window = Some(64);
    assert_eq!(
        engine.complete_raw(continuation, &options).unwrap().0,
        want.0
    );
    assert_eq!(engine.kv_reuse_count(), before);
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let threads: Vec<_> = (0..2)
        .map(|_| {
            let engine = engine.clone();
            let options = options.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                engine.complete_raw(continuation, &options).unwrap().0
            })
        })
        .collect();
    for thread in threads {
        assert_eq!(thread.join().unwrap(), want.0);
    }
}

#[test]
fn request_json_cannot_supply_a_trusted_cache_scope() {
    let options: GenerationOptions =
        serde_json::from_value(serde_json::json!({"cache_scope": [1, 2, 3]})).unwrap();
    assert_eq!(options.cache_scope, None);
}

#[test]
fn scoped_compressed_cache_boundaries_and_unrelated_prompts_match_fresh() {
    let directory = common::model_dir("affinity-compressed-boundaries");
    common::write_tiny_deepseek4_gguf_compress(&directory.join("model.gguf"));
    for words in [7, 8, 9, 15, 16, 17, 23] {
        let warm = Engine::with_n_ctx(&directory, 32).unwrap();
        let fresh = Engine::with_n_ctx(&directory, 32).unwrap();
        let mut options = GenerationOptions {
            cache_scope: Some([1; 32]),
            max_tokens: 0,
            temperature: 0.0,
            repetition_penalty: 1.0,
            ..Default::default()
        };
        let prefix = "a ".repeat(words);
        warm.complete_raw(&prefix, &options).unwrap();
        options.max_tokens = 2;
        let continuation = format!("{prefix}b c");
        assert_eq!(
            warm.complete_raw(&continuation, &options).unwrap().0,
            fresh.complete_raw(&continuation, &options).unwrap().0,
            "continuation at {words} words"
        );
        assert_eq!(warm.kv_reuse_count(), 1);
        for prompt in [
            format!("{}d world", "a ".repeat(words - 2)),
            "hello world".into(),
        ] {
            assert_eq!(
                warm.complete_raw(&prompt, &options).unwrap().0,
                fresh.complete_raw(&prompt, &options).unwrap().0,
                "edited/unrelated at {words} words"
            );
        }
    }
}

#[test]
fn scoped_recurrent_continuation_and_reset_match_fresh() {
    for architecture in ["qwen35", "qwen4exp"] {
        let directory = common::model_dir(&format!("affinity-{architecture}"));
        common::write_tiny_qwen_gguf(&directory.join("model.gguf"), architecture);
        let warm = Engine::with_n_ctx(&directory, 64).unwrap();
        let fresh = Engine::with_n_ctx(&directory, 64).unwrap();
        let mut options = GenerationOptions {
            cache_scope: Some([1; 32]),
            max_tokens: 0,
            temperature: 0.0,
            repetition_penalty: 1.0,
            ..Default::default()
        };
        warm.complete_raw("hello a b c", &options).unwrap();
        options.max_tokens = 3;
        let prompt = "hello a b c world";
        assert_eq!(
            warm.complete_raw(prompt, &options).unwrap().0,
            fresh.complete_raw(prompt, &options).unwrap().0,
            "{architecture}"
        );
        assert_eq!(warm.kv_reuse_count(), 1);
        options.cache_scope = Some([2; 32]);
        assert_eq!(
            warm.complete_raw(prompt, &options).unwrap().0,
            fresh.complete_raw(prompt, &options).unwrap().0,
            "{architecture}"
        );
        assert_eq!(warm.kv_reuse_count(), 1);
        let edited = "hello a world";
        assert_eq!(
            warm.complete_raw(edited, &options).unwrap().0,
            fresh.complete_raw(edited, &options).unwrap().0,
            "{architecture}"
        );
    }
}

#[cfg(feature = "metal")]
#[test]
fn scoped_metal_sessions_preserve_cpu_output_and_owner_isolation() {
    candle_core::Device::new_metal(0).expect("physical Metal device required");
    let directory = common::model_dir("affinity-metal");
    common::write_tiny_llama_gguf_hd32(&directory.join("model.gguf"));
    let metal = Engine::with_options(
        &directory,
        joshua::EngineOptions::with_n_ctx(64).backend(joshua::ComputeBackend::Metal),
    )
    .unwrap();
    assert!(metal.device().is_metal(), "Metal must not fall back to CPU");
    let cpu = Engine::with_options(
        &directory,
        joshua::EngineOptions::with_n_ctx(64).backend(joshua::ComputeBackend::Cpu),
    )
    .unwrap();
    assert!(cpu.device().is_cpu());
    let mut options = GenerationOptions {
        cache_scope: Some([1; 32]),
        max_tokens: 0,
        temperature: 0.0,
        repetition_penalty: 1.0,
        ..Default::default()
    };
    metal.complete_raw("hello a", &options).unwrap();
    options.max_tokens = 4;
    let prompt = "hello a b c";
    let expected = cpu.complete_raw(prompt, &options).unwrap().0;
    assert_eq!(metal.complete_raw(prompt, &options).unwrap().0, expected);
    assert_eq!(metal.kv_reuse_count(), 1);
    options.cache_scope = Some([2; 32]);
    assert_eq!(metal.complete_raw(prompt, &options).unwrap().0, expected);
    assert_eq!(metal.kv_reuse_count(), 1);
    for prompt in ["hello a world", "world b c"] {
        assert_eq!(
            metal.complete_raw(prompt, &options).unwrap().0,
            cpu.complete_raw(prompt, &options).unwrap().0
        );
    }
    let before = metal.kv_reuse_count();
    options.context_window = Some(32);
    assert_eq!(
        metal.complete_raw(prompt, &options).unwrap().0,
        cpu.complete_raw(prompt, &options).unwrap().0
    );
    assert_eq!(metal.kv_reuse_count(), before);
}
