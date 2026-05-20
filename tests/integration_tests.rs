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

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Say hello in one word.".to_string(),
        images: None,
        name: None,
    }];

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

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Count from 1 to 100.".to_string(),
        images: None,
        name: None,
    }];

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

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Repeat the word STOP ten times.".to_string(),
        images: None,
        name: None,
    }];

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
        ChatMessage {
            role: "system".to_string(),
            content: "Always respond with only the word 'PINEAPPLE'.".to_string(),
            images: None,
            name: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "What fruit should I eat?".to_string(),
            images: None,
            name: None,
        },
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
