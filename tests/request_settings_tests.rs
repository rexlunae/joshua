//! Per-request settings (#144) through the full engine on the tiny llama
//! fixture: the per-request context window, the sampler seed, and reasoning
//! switches reaching the GGUF chat template.

mod common;

use candle_core::quantized::gguf_file;
use candle_core::Device;
use joshua::types::reasoning_template_kwargs;
use joshua::{ChatMessage, Engine, GenerationOptions, JoshuaError, ReasoningEffort};
use std::path::Path;

fn greedy(max_tokens: u32) -> GenerationOptions {
    GenerationOptions {
        max_tokens,
        temperature: 0.0,
        repetition_penalty: 1.0,
        ..GenerationOptions::default()
    }
}

#[test]
fn context_window_bounds_prompt_and_generation() {
    let dir = common::model_dir("req-ctx-window");
    common::write_tiny_llama_gguf(&dir.join("model.gguf"));
    let engine = Engine::with_n_ctx(&dir, 64).expect("engine should load tiny model");

    // Prompt (2 tokens) + generation must fit a 4-token window.
    let opts = GenerationOptions {
        context_window: Some(4),
        ..greedy(16)
    };
    let (_, usage, _, _) = engine.complete_raw("hello world", &opts).unwrap();
    assert_eq!(usage.prompt_tokens, 2);
    assert!(usage.total_tokens <= 4, "window exceeded: {usage:?}");

    // A prompt that does not fit the request's window is rejected...
    match engine.complete_raw("hello a b c d", &opts) {
        Err(JoshuaError::PromptTooLong(5, 4)) => {}
        other => panic!("expected PromptTooLong(5, 4), got {other:?}"),
    }
    // ...but fits the engine's own window.
    engine.complete_raw("hello a b c d", &greedy(1)).unwrap();

    // A window larger than the loaded one cannot be granted.
    let too_big = GenerationOptions {
        context_window: Some(65),
        ..greedy(1)
    };
    match engine.complete_raw("hello", &too_big) {
        Err(JoshuaError::InvalidRequest(msg)) => assert!(msg.contains("64"), "{msg}"),
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
    // Exactly the loaded window is fine and matches no setting at all.
    let exact = GenerationOptions {
        context_window: Some(64),
        ..greedy(6)
    };
    assert_eq!(
        engine.complete_raw("hello a", &exact).unwrap().0,
        engine.complete_raw("hello a", &greedy(6)).unwrap().0
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn seed_reproduces_sampled_output() {
    let dir = common::model_dir("req-seed");
    common::write_tiny_llama_gguf(&dir.join("model.gguf"));
    let sampled = |seed| GenerationOptions {
        max_tokens: 12,
        temperature: 1.5,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        seed: Some(seed),
        ..GenerationOptions::default()
    };
    let a = Engine::with_n_ctx(&dir, 64).unwrap();
    let b = Engine::with_n_ctx(&dir, 64).unwrap();
    for seed in [1, 7, 12345] {
        let first = a.complete_raw("hello a", &sampled(seed)).unwrap().0;
        let again = a.complete_raw("hello a", &sampled(seed)).unwrap().0;
        let other_engine = b.complete_raw("hello a", &sampled(seed)).unwrap().0;
        assert_eq!(first, again, "seed {seed}: same engine");
        assert_eq!(first, other_engine, "seed {seed}: fresh engine");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Rewrite `path` with a different `tokenizer.chat_template`, keeping every
/// other metadata entry and tensor.
fn replace_chat_template(path: &Path, template: &str) {
    let mut file = std::fs::File::open(path).unwrap();
    let content = gguf_file::Content::read(&mut file).unwrap();
    let mut metadata = content.metadata.clone();
    metadata.insert(
        "tokenizer.chat_template".to_string(),
        gguf_file::Value::String(template.to_string()),
    );
    let tensors: Vec<(String, _)> = content
        .tensor_infos
        .keys()
        .map(|name| {
            let t = content.tensor(&mut file, name, &Device::Cpu).unwrap();
            (name.clone(), t)
        })
        .collect();
    drop(file);
    let metadata_refs: Vec<(&str, &gguf_file::Value)> =
        metadata.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let tensor_refs: Vec<(&str, &_)> = tensors.iter().map(|(k, t)| (k.as_str(), t)).collect();
    let mut out = std::fs::File::create(path).unwrap();
    gguf_file::write(&mut out, &metadata_refs, &tensor_refs).unwrap();
}

#[test]
fn reasoning_settings_reach_the_gguf_chat_template() {
    let dir = common::model_dir("req-reasoning");
    let model = dir.join("model.gguf");
    common::write_tiny_llama_gguf(&model);
    // Qwen3-style switch (thinking off adds "a b") plus a gpt-oss-style
    // effort marker ("c" for high, "d" otherwise), in the fixture's tiny
    // vocabulary so the prompt token count shows what was rendered.
    replace_chat_template(
        &model,
        "{% for message in messages %}hello {{ message.content }} {% endfor %}\
         {% if add_generation_prompt %}\
         {% if enable_thinking is defined and enable_thinking is false %}a b {% endif %}\
         {% if reasoning_effort is defined %}{% if reasoning_effort == 'high' %}c{% else %}d{% endif %} {% endif %}\
         world{% endif %}",
    );
    let engine = Engine::with_n_ctx(&dir, 64).unwrap();
    assert!(engine.has_chat_template());
    let messages = [ChatMessage::text("user", "e")];
    let prompt_tokens = |effort, enabled| {
        let opts = GenerationOptions {
            chat_template_kwargs: reasoning_template_kwargs(effort, enabled, None),
            ..greedy(1)
        };
        engine.complete(&messages, &opts).unwrap().1.prompt_tokens
    };
    // "hello e world"
    assert_eq!(prompt_tokens(None, None), 3);
    // "hello e c world"
    assert_eq!(prompt_tokens(Some(ReasoningEffort::High), None), 4);
    // "hello e a b d world": thinking off, effort low
    assert_eq!(prompt_tokens(Some(ReasoningEffort::None), None), 6);
    assert_eq!(prompt_tokens(None, Some(false)), 6);
    // Thinking on, no effort: template default.
    assert_eq!(prompt_tokens(None, Some(true)), 3);
    std::fs::remove_dir_all(&dir).ok();
}
