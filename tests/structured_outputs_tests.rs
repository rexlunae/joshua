//! Structured-output (`response_format`) tests.
//!
//! The fixture's vocabulary contains only the characters of `{"a":1}` (and
//! filler), so a schema that admits only that document leaves the sampler no
//! choice: the test fails if the mask, the grammar state, or token consumption
//! is wrong, not merely if the model happens to prefer invalid tokens.

mod common;

use std::path::Path;

use joshua::types::ConstraintSpec;
use joshua::{Engine, EngineOptions, GenerationOptions, JoshuaError};
use serde_json::json;

fn engine(dir: &Path) -> Engine {
    Engine::with_options(dir, EngineOptions::with_n_ctx(128))
        .expect("engine should load the tiny JSON model")
}

fn options(constraint: Option<ConstraintSpec>) -> GenerationOptions {
    GenerationOptions {
        max_tokens: 8,
        temperature: 0.0,
        constraint,
        ..GenerationOptions::default()
    }
}

fn forced_schema() -> ConstraintSpec {
    ConstraintSpec::JsonSchema {
        schema: json!({
            "type": "object",
            "properties": {"a": {"const": 1}},
            "required": ["a"],
            "additionalProperties": false
        }),
    }
}

fn setup(name: &str) -> (std::path::PathBuf, Engine) {
    let dir = common::model_dir(name);
    common::write_tiny_llama_gguf(&dir.join("model.gguf"));
    std::fs::write(dir.join("tokenizer.json"), common::JSON_TOKENIZER_JSON).unwrap();
    let engine = engine(&dir);
    (dir, engine)
}

#[test]
fn json_schema_forces_the_only_valid_document() {
    let (dir, engine) = setup("structured-forced");
    let (text, usage, _, _) = engine
        .complete_raw("a", &options(Some(forced_schema())))
        .expect("constrained completion");
    assert_eq!(text, r#"{"a":1}"#);
    assert_eq!(usage.completion_tokens, 7, "seven JSON tokens were emitted");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn json_object_starts_an_object() {
    let (dir, engine) = setup("structured-object");
    // Any object is legal, so the model may produce a long one; one token is
    // enough to check that `json_object` reached the sampler and that the
    // first token is the object brace (the vocabulary has no whitespace).
    let opts = GenerationOptions {
        max_tokens: 1,
        ..options(Some(ConstraintSpec::JsonObject))
    };
    let (text, _, _, _) = engine
        .complete_raw("a", &opts)
        .expect("json_object completion");
    assert_eq!(text, "{");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn an_invalid_schema_is_a_bad_request() {
    let (dir, engine) = setup("structured-invalid");
    let bad = ConstraintSpec::JsonSchema {
        schema: json!({"type": "not-a-json-type"}),
    };
    let err = engine
        .complete_raw("a", &options(Some(bad)))
        .expect_err("invalid schema must be rejected");
    match err {
        JoshuaError::InvalidRequest(message) => {
            assert!(
                message.contains("response_format") || message.contains("schema"),
                "unexpected message: {message}"
            );
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn unconstrained_output_is_unchanged() {
    let (dir, engine) = setup("structured-unconstrained");
    let options = options(None);
    let (a, ua, _, _) = engine.complete_raw("a", &options).unwrap();
    let (b, ub, _, _) = engine.complete_raw("a", &options).unwrap();
    // Same seed (the default is deterministic greedy here) and no grammar
    // state: the two requests must match, including token counts.
    assert_eq!(a, b);
    assert_eq!(ua.completion_tokens, ub.completion_tokens);
    std::fs::remove_dir_all(&dir).ok();
}
