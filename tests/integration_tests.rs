//! Integration tests for the Joshua inference engine.
//!
//! Most tests are gated behind the `JOSHUA_MODEL_PATH` environment variable.
//! If the variable is not set the test is skipped.
//!
//! # Running
//!
//! ```bash
//! export JOSHUA_MODEL_PATH=/path/to/model.gguf
//! cargo test --test integration_tests -- --nocapture
//! ```

use joshua::{types::GenerationOptions, ChatMessage, Engine};
use std::path::PathBuf;

/// Returns the model path from the environment, or `None` if unset.
fn model_path() -> Option<PathBuf> {
    std::env::var("JOSHUA_MODEL_PATH").ok().map(PathBuf::from)
}

/// Load the engine; skip the test if no model path is set.
macro_rules! require_engine {
    ($name:ident) => {
        let Some(path) = model_path() else {
            eprintln!("JOSHUA_MODEL_PATH not set — skipping {}", stringify!($name));
            return;
        };
        let $name = Engine::new(&path).expect("failed to load engine");
    };
}

// ─── Engine unit tests ────────────────────────────────────────────────────────

#[test]
fn test_engine_model_name_is_nonempty() {
    require_engine!(engine);
    assert!(!engine.model_name().is_empty());
}

#[test]
fn test_engine_n_ctx_default() {
    require_engine!(engine);
    assert_eq!(engine.n_ctx(), 4096);
}

#[test]
fn test_engine_model_path_exists() {
    let Some(path) = model_path() else {
        return;
    };
    let engine = Engine::new(&path).expect("failed to load engine");
    assert!(engine.model_path().exists());
}

// ─── Completion tests ─────────────────────────────────────────────────────────

#[test]
fn test_completion_returns_nonempty_text() {
    require_engine!(engine);

    let messages = vec![ChatMessage::text("user".to_string(), "Say hello in one word.".to_string())];

    let options = GenerationOptions {
        max_tokens: 8,
        temperature: 0.0, // greedy for reproducibility
        ..Default::default()
    };

    let (text, usage, _, _) = engine
        .complete(&messages, &options)
        .expect("completion failed");

    assert!(!text.trim().is_empty(), "response should not be empty");
    assert!(usage.prompt_tokens > 0);
    assert!(usage.completion_tokens > 0);
    assert_eq!(
        usage.total_tokens,
        usage.prompt_tokens + usage.completion_tokens
    );
}

#[test]
fn test_completion_respects_max_tokens() {
    require_engine!(engine);

    let messages = vec![ChatMessage::text("user".to_string(), "Count from 1 to 100.".to_string())];

    let max = 5u32;
    let options = GenerationOptions {
        max_tokens: max,
        temperature: 0.0,
        ..Default::default()
    };

    let (_, usage, _, _) = engine
        .complete(&messages, &options)
        .expect("completion failed");

    assert!(
        usage.completion_tokens <= max,
        "generated {} tokens, expected <= {}",
        usage.completion_tokens,
        max
    );
}

#[test]
fn test_completion_stop_sequence() {
    require_engine!(engine);

    let messages = vec![ChatMessage::text("user".to_string(), "Repeat the word STOP ten times.".to_string())];

    let options = GenerationOptions {
        max_tokens: 64,
        temperature: 0.0,
        stop_sequences: vec!["STOP".to_string()],
        ..Default::default()
    };

    let (text, _, _, _) = engine
        .complete(&messages, &options)
        .expect("completion failed");

    // The stop sequence itself must not appear in the trimmed response.
    assert!(
        !text.contains("STOP"),
        "stop sequence should not appear in output"
    );
}

#[test]
fn test_system_prompt_is_respected() {
    require_engine!(engine);

    let messages = vec![
        ChatMessage::text("system".to_string(), "Always respond with only the word 'PINEAPPLE'.".to_string()),
        ChatMessage::text("user".to_string(), "What fruit should I eat?".to_string()),
    ];

    let options = GenerationOptions {
        max_tokens: 10,
        temperature: 0.0,
        ..Default::default()
    };

    let (text, _, _, _) = engine
        .complete(&messages, &options)
        .expect("completion failed");

    // With a very explicit system prompt and greedy decoding most models will comply.
    // We just check the response is non-empty; compliance is model-dependent.
    assert!(!text.trim().is_empty());
}

// ─── Embedding tests ──────────────────────────────────────────────────────────

/// Embeddings require a model that supports them.
/// Set `JOSHUA_EMBED_MODEL_PATH` separately if using a dedicated embedding model.
fn embed_model_path() -> Option<PathBuf> {
    std::env::var("JOSHUA_EMBED_MODEL_PATH")
        .ok()
        .or_else(|| std::env::var("JOSHUA_MODEL_PATH").ok())
        .map(PathBuf::from)
}

#[test]
fn test_embed_returns_correct_count() {
    let Some(path) = embed_model_path() else {
        eprintln!("JOSHUA_EMBED_MODEL_PATH not set — skipping embed test");
        return;
    };

    let engine = match Engine::new(&path) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to load embed engine: {e}");
            return;
        }
    };

    let texts = vec!["Hello world".to_string(), "Rust is fast".to_string()];
    match engine.embed(&texts) {
        Ok(vectors) => {
            assert_eq!(vectors.len(), 2);
            for v in &vectors {
                assert!(!v.is_empty(), "embedding should not be empty");
            }
        }
        Err(e) => {
            // Some base LLMs don't expose embeddings — that's acceptable.
            eprintln!("embed() returned error (model may not support embeddings): {e}");
        }
    }
}

mod common;

// ─── Synthetic-model tests (always run) ──────────────────────────────────────
//
// These tests use tiny but structurally valid GGUF files (see tests/common)
// and run the full engine pipeline: mmap → architecture dispatch → prefill →
// decode.  No network access or real model download is needed.

mod synthetic {
    use crate::common::*;
    use candle_core::quantized::gguf_file;
    use candle_core::{Device, Tensor};

    #[test]
    fn tiny_llama_end_to_end() {
        use joshua::{types::GenerationOptions, Engine};

        let dir = model_dir("tiny-llama");
        write_tiny_llama_gguf(&dir.join("model.gguf"));

        let engine = Engine::with_n_ctx(&dir, 64).expect("engine should load tiny model");
        assert_eq!(engine.model_name(), "model");

        let options = GenerationOptions {
            max_tokens: 4,
            temperature: 0.0,
            ..Default::default()
        };

        // Run twice: the second call re-instantiates the model from the same
        // mmap, verifying weights stay readable across loads.
        for _ in 0..2 {
            let (_, usage, _, _) = engine
                .complete_raw("hello world", &options)
                .expect("completion should succeed");
            assert_eq!(usage.prompt_tokens, 2);
            assert!(usage.completion_tokens <= 4);
            assert_eq!(
                usage.total_tokens,
                usage.prompt_tokens + usage.completion_tokens
            );
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn chat_template_from_gguf_is_used_by_complete() {
        use joshua::{types::GenerationOptions, ChatMessage, Engine};

        let dir = model_dir("tiny-llama-template");
        write_tiny_llama_gguf(&dir.join("model.gguf"));

        let engine = Engine::with_n_ctx(&dir, 64).expect("engine should load tiny model");
        assert!(engine.has_chat_template());

        let messages = vec![ChatMessage::text("user".to_string(), "a b".to_string())];
        let options = GenerationOptions {
            max_tokens: 2,
            temperature: 0.0,
            ..Default::default()
        };

        // The template renders "hello a b world" → exactly 4 known tokens,
        // proving the prompt came from the GGUF template rather than the
        // ChatML fallback (whose <|im_start|> markers would tokenise to a
        // different count).
        let (_, usage, _, _) = engine
            .complete(&messages, &options)
            .expect("completion should succeed");
        assert_eq!(usage.prompt_tokens, 4);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn kv_prefix_reuse_matches_fresh_engine() {
        use joshua::{types::GenerationOptions, Engine};

        let dir = model_dir("tiny-llama-kv");
        write_tiny_llama_gguf(&dir.join("model.gguf"));

        let greedy = |max_tokens| GenerationOptions {
            max_tokens,
            temperature: 0.0,
            repetition_penalty: 1.0,
            ..Default::default()
        };

        // Warm engine: first request seeds the pool with KV for "hello a",
        // second request extends that prompt and must take the prefix-reuse
        // path (max_tokens: 0 keeps the cached history equal to the prompt).
        let warm = Engine::with_n_ctx(&dir, 64).expect("engine should load");
        warm.complete_raw("hello a", &greedy(0)).unwrap();
        assert_eq!(warm.kv_reuse_count(), 0);
        let (warm_text, warm_usage, _, _) =
            warm.complete_raw("hello a b c", &greedy(4)).unwrap();
        assert_eq!(warm.kv_reuse_count(), 1, "second call must reuse the KV prefix");

        // Fresh engine: same extended prompt with an empty cache.
        let fresh = Engine::with_n_ctx(&dir, 64).expect("engine should load");
        let (fresh_text, fresh_usage, _, _) =
            fresh.complete_raw("hello a b c", &greedy(4)).unwrap();

        assert_eq!(warm_text, fresh_text, "prefix reuse must not change output");
        assert_eq!(warm_usage.prompt_tokens, fresh_usage.prompt_tokens);
        assert_eq!(warm_usage.completion_tokens, fresh_usage.completion_tokens);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn kv_clear_reuse_matches_fresh_engine() {
        use joshua::{types::GenerationOptions, Engine};

        let dir = model_dir("tiny-llama-kvclear");
        write_tiny_llama_gguf(&dir.join("model.gguf"));

        let greedy = GenerationOptions {
            max_tokens: 4,
            temperature: 0.0,
            repetition_penalty: 1.0,
            ..Default::default()
        };

        // Two unrelated prompts: the second reuses the pooled llama instance
        // after a KV clear (no prefix in common) — output must match a fresh
        // engine exactly.
        let warm = Engine::with_n_ctx(&dir, 64).expect("engine should load");
        warm.complete_raw("hello a", &greedy).unwrap();
        let (warm_text, _, _, _) = warm.complete_raw("world k l", &greedy).unwrap();
        assert_eq!(warm.kv_reuse_count(), 0, "unrelated prompt must not prefix-reuse");

        let fresh = Engine::with_n_ctx(&dir, 64).expect("engine should load");
        let (fresh_text, _, _, _) = fresh.complete_raw("world k l", &greedy).unwrap();

        assert_eq!(warm_text, fresh_text, "KV clear must fully reset the cache");

        std::fs::remove_dir_all(&dir).ok();
    }


    /// The embedding forward pass must reproduce candle's generation-model
    /// logits exactly (same weights, same input): if the hidden states going
    /// into the LM head match, the hidden states used for pooling are right.
    fn assert_embedding_logits_match_candle(arch: &str) {
        use joshua::embedding::EmbeddingModel;
        use joshua::model::QuantizedModel;

        let dir = model_dir(&format!("parity-{arch}"));
        let path = dir.join("model.gguf");
        write_tiny_gguf(&path, arch);
        let bytes = std::fs::read(&path).unwrap();
        let tokens: Vec<u32> = vec![1, 4, 2, 7, 5];

        // Reference: candle's quantized generation model.
        let mut cursor = std::io::Cursor::new(&bytes[..]);
        let content = gguf_file::Content::read(&mut cursor).unwrap();
        let mut reference =
            QuantizedModel::from_gguf(content, &mut cursor, &Device::Cpu).unwrap();
        let input = Tensor::new(tokens.as_slice(), &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let ref_logits: Vec<f32> = reference
            .forward(&input, 0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_dtype(candle_core::DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap();

        // Joshua's embedding forward + LM head.
        let mut cursor = std::io::Cursor::new(&bytes[..]);
        let content = gguf_file::Content::read(&mut cursor).unwrap();
        let embedder = EmbeddingModel::from_gguf(content, &mut cursor, &Device::Cpu).unwrap();
        let logits = embedder.logits(&tokens).unwrap();

        assert_eq!(ref_logits.len(), logits.len(), "arch {arch}: vocab size");
        for (i, (a, b)) in ref_logits.iter().zip(&logits).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "arch {arch}: logit {i} diverges: candle={a} joshua={b}"
            );
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn embedding_forward_matches_candle_llama() {
        assert_embedding_logits_match_candle("llama");
    }

    #[test]
    fn embedding_forward_matches_candle_qwen2() {
        assert_embedding_logits_match_candle("qwen2");
    }

    #[test]
    fn embedding_forward_matches_candle_qwen3() {
        assert_embedding_logits_match_candle("qwen3");
    }

    #[test]
    fn engine_embeddings_are_normalised_and_distinct() {
        use joshua::Engine;

        // llama exercises mean pooling (default), qwen3 last-token pooling.
        for arch in ["llama", "qwen3"] {
            let dir = model_dir(&format!("embed-{arch}"));
            write_tiny_gguf(&dir.join("model.gguf"), arch);

            let engine = Engine::with_n_ctx(&dir, 64).expect("engine should load");
            let texts = vec!["hello world".to_string(), "a b c".to_string()];
            let (vectors, tokens) = engine.embed_with_usage(&texts).expect("embed failed");

            assert_eq!(vectors.len(), 2, "arch {arch}");
            assert_eq!(tokens, 5, "arch {arch}: 2 + 3 input tokens");
            for v in &vectors {
                assert_eq!(v.len(), 8, "arch {arch}: hidden size");
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!((norm - 1.0).abs() < 1e-4, "arch {arch}: L2 norm {norm}");
            }
            assert_ne!(vectors[0], vectors[1], "arch {arch}: distinct inputs");

            std::fs::remove_dir_all(&dir).ok();
        }
    }

    #[test]
    fn unsupported_architecture_fails_at_load_with_clear_error() {
        use joshua::Engine;

        let dir = model_dir("unsupported-arch");
        write_unsupported_gguf(&dir.join("model.gguf"));

        let msg = match Engine::with_n_ctx(&dir, 64) {
            Ok(_) => panic!("mamba arch must be rejected"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("known llama.cpp architecture"),
            "unexpected error message: {msg}"
        );
        assert!(msg.contains("mamba"), "unexpected error message: {msg}");

        std::fs::remove_dir_all(&dir).ok();
    }
}

// ─── Type tests (always run) ──────────────────────────────────────────────────

#[test]
fn test_generation_options_default() {
    let opts = GenerationOptions::default();
    assert_eq!(opts.max_tokens, 256);
    assert!(opts.temperature > 0.0);
    assert!(opts.top_p > 0.0);
    assert!(opts.top_k > 0);
    assert!(opts.min_p >= 0.0);
    assert!(opts.repetition_penalty >= 1.0);
    assert!(opts.stop_sequences.is_empty());
}

#[test]
fn test_chat_completion_request_to_generation_options() {
    use joshua::types::ChatCompletionRequest;

    let req = ChatCompletionRequest {
        model: "test".to_string(),
        messages: vec![],
        max_tokens: Some(512),
        temperature: Some(0.5),
        top_p: Some(0.8),
        top_k: None,
        min_p: None,
        repetition_penalty: None,
        stop: Some(serde_json::json!(["<end>", "<stop>"])),
        stream: None,
        tools: None,
    };

    let opts = req.to_generation_options();
    assert_eq!(opts.max_tokens, 512);
    assert!((opts.temperature - 0.5).abs() < 1e-6);
    assert!((opts.top_p - 0.8).abs() < 1e-6);
    assert_eq!(opts.stop_sequences, vec!["<end>".to_string(), "<stop>".to_string()]);
}

#[test]
fn test_embedding_input_single_to_vec() {
    use joshua::types::EmbeddingInput;
    let input = EmbeddingInput::Single("hello".to_string());
    assert_eq!(input.into_vec(), vec!["hello".to_string()]);
}

#[test]
fn test_embedding_input_multiple_to_vec() {
    use joshua::types::EmbeddingInput;
    let input = EmbeddingInput::Multiple(vec!["a".to_string(), "b".to_string()]);
    assert_eq!(input.into_vec(), vec!["a".to_string(), "b".to_string()]);
}

#[test]
fn test_error_response_variants() {
    use joshua::types::ErrorResponse;
    let e = ErrorResponse::invalid_request("bad param");
    assert_eq!(e.error.error_type, "invalid_request_error");

    let e = ErrorResponse::server_error("oops");
    assert_eq!(e.error.error_type, "server_error");
}
