//! Engine-level API compatibility primitives (#254): the raw-prompt
//! generation API, completion metadata (stop type, cached tokens, timings),
//! the unlimited `max_tokens` default, sampling defaults and the sampler
//! extras, template controls, the reasoning budget and FIM prompts — all on
//! the tiny synthetic models in `tests/common`.

mod common;

use std::sync::atomic::AtomicBool;

use candle_core::quantized::gguf_file::Value;
use common::{
    model_dir, think_model_dir, write_tiny_gguf_with, write_tiny_llama_gguf, THINK_TEMPLATE,
};
use joshua::fim::InfillChunk;
use joshua::reasoning::{split, ReasoningFormat};
use joshua::types::{LogitBias, TokenRef, MAX_TOKENS_UNLIMITED, TRUNCATE_TO_CONTEXT};
use joshua::{
    ChatMessage, Engine, EngineOptions, GenerationOptions, JoshuaError, PromptPart, RawPrompt,
    SamplingOverrides, SamplingProfile, StopType,
};

/// EOS in the tiny vocabulary, and the ordinary words `a`, `b`, `c`.
const EOS: u32 = 3;
const A: u32 = 4;
const B: u32 = 5;
const C: u32 = 6;

fn engine(name: &str) -> Engine {
    let dir = model_dir(name);
    write_tiny_llama_gguf(&dir.join("model.gguf"));
    Engine::with_n_ctx(&dir, 64).expect("tiny model loads")
}

fn greedy(max_tokens: u32) -> GenerationOptions {
    GenerationOptions {
        max_tokens,
        temperature: 0.0,
        ..Default::default()
    }
}

fn bias(token: u32, bias: f32) -> LogitBias {
    LogitBias {
        token: TokenRef::Id(token),
        bias,
    }
}

/// The text the engine writes for `ids`: the word-level test tokenizer has
/// no decoder, so each token is decoded alone and appended.
fn text_of(engine: &Engine, ids: &[u32]) -> String {
    ids.iter()
        .map(|&t| engine.detokenize(&[t], false).unwrap())
        .collect()
}

#[test]
fn raw_prompt_text_token_and_mixed_forms_agree() {
    let engine = engine("core-raw-forms");
    let text = "hello a b";
    let ids = engine.tokenize(text, true).unwrap();
    let options = greedy(6);

    let from_text = engine
        .complete_prompt(&RawPrompt::text(text), &options)
        .unwrap();
    let from_ids = engine
        .complete_prompt(&RawPrompt::tokens(ids.clone()), &options)
        .unwrap();
    let mixed = engine
        .complete_prompt(
            &RawPrompt::mixed(vec![
                PromptPart::Text("hello".into()),
                PromptPart::Tokens(engine.tokenize("a b", false).unwrap()),
            ]),
            &options,
        )
        .unwrap();
    for other in [&from_ids, &mixed] {
        assert_eq!(other.text, from_text.text);
        assert_eq!(other.token_ids, from_text.token_ids);
        assert_eq!(other.usage.prompt_tokens, ids.len() as u32);
    }
    // The tuple API is the same generation.
    assert_eq!(
        engine.complete_raw(text, &options).unwrap().0,
        from_text.text
    );
    // The completion's token ids decode to its text.
    assert_eq!(text_of(&engine, &from_text.token_ids), from_text.text);
    assert_eq!(engine.tokenize_prompt(&RawPrompt::text(text)).unwrap(), ids);

    match engine.complete_prompt(&RawPrompt::tokens(vec![1, 9999]), &options) {
        Err(JoshuaError::InvalidRequest(message)) => assert!(message.contains("9999")),
        other => panic!("an out-of-vocabulary id must be rejected, got {other:?}"),
    }
}

#[test]
fn raw_prompt_streams_and_cancels() {
    let engine = engine("core-raw-stream");
    let prompt = RawPrompt::text("hello a b");
    let options = greedy(6);
    let whole = engine.complete_prompt(&prompt, &options).unwrap();

    let cancel = AtomicBool::new(false);
    let mut streamed = String::new();
    let completion = engine
        .complete_prompt_streaming(&prompt, &options, &cancel, &mut |piece| {
            streamed.push_str(piece)
        })
        .unwrap();
    assert_eq!(streamed, completion.text);
    assert_eq!(completion.text, whole.text);
    let cancellable = engine
        .complete_prompt_cancellable(&prompt, &options, &cancel)
        .unwrap();
    assert_eq!(cancellable.text, whole.text);

    let cancelled = AtomicBool::new(true);
    let completion = engine
        .complete_prompt_cancellable(&prompt, &options, &cancelled)
        .unwrap();
    assert_eq!(completion.stop_type, StopType::Cancelled);
    assert!(completion.text.is_empty());
}

#[test]
fn stop_types_and_finish_reasons() {
    let engine = engine("core-stop-types");
    let prompt = RawPrompt::text("hello a b");
    let run = |options: GenerationOptions| engine.complete_prompt(&prompt, &options).unwrap();

    // A forced EOS: nothing generated, a natural stop.
    let c = run(GenerationOptions {
        logit_bias: vec![bias(EOS, 100.0)],
        ..greedy(8)
    });
    assert_eq!((c.stop_type, c.finish_reason()), (StopType::Eos, "stop"));
    assert!(c.token_ids.is_empty() && c.text.is_empty());
    assert_eq!(c.stop_token, None);

    // The cap: `length`.
    let c = run(GenerationOptions {
        logit_bias: vec![bias(A, 100.0)],
        ..greedy(3)
    });
    assert_eq!(
        (c.stop_type, c.finish_reason()),
        (StopType::Limit, "length")
    );
    assert_eq!(c.token_ids, vec![A; 3]);
    assert_eq!(c.text, "aaa");

    // A stop string: trimmed, or kept with include_stop_str_in_output.
    let words = GenerationOptions {
        logit_bias: vec![bias(A, 100.0)],
        stop_sequences: vec!["aa".into()],
        ..greedy(8)
    };
    let c = run(words.clone());
    assert_eq!(
        (c.stop_type, c.stop_sequence.as_deref()),
        (StopType::Word, Some("aa"))
    );
    assert_eq!(c.text, "");
    let c = run(GenerationOptions {
        include_stop_str_in_output: true,
        ..words
    });
    assert_eq!(c.text, "aa");

    // A request stop token ends generation like EOS and is reported.
    let c = run(GenerationOptions {
        logit_bias: vec![bias(B, 100.0)],
        stop_token_ids: vec![B],
        ..greedy(8)
    });
    assert_eq!((c.stop_type, c.stop_token), (StopType::Eos, Some(B)));
    assert!(c.text.is_empty());

    // min_tokens holds EOS back; ignore_eos holds it back for good.
    let c = run(GenerationOptions {
        logit_bias: vec![bias(EOS, 100.0)],
        min_tokens: 3,
        ..greedy(8)
    });
    assert_eq!((c.stop_type, c.usage.completion_tokens), (StopType::Eos, 3));
    let c = run(GenerationOptions {
        logit_bias: vec![bias(EOS, 100.0)],
        ignore_eos: true,
        ..greedy(4)
    });
    assert_eq!(
        (c.stop_type, c.usage.completion_tokens),
        (StopType::Limit, 4)
    );
    assert!(!c.token_ids.contains(&EOS));
}

#[test]
fn token_restrictions_and_text_biases() {
    let engine = engine("core-restrictions");
    let prompt = RawPrompt::text("hello a b");
    let run = |options: GenerationOptions| engine.complete_prompt(&prompt, &options).unwrap();
    let base = GenerationOptions {
        ignore_eos: true,
        ..greedy(4)
    };
    let c = run(GenerationOptions {
        allowed_token_ids: Some(vec![B]),
        ..base.clone()
    });
    assert_eq!(c.token_ids, vec![B; 4]);
    let c = run(GenerationOptions {
        allowed_token_ids: Some(vec![B, C]),
        bad_words: vec!["b".into()],
        ..base.clone()
    });
    assert_eq!(c.token_ids, vec![C; 4]);
    // A text key biases every token it tokenizes to.
    let c = run(GenerationOptions {
        logit_bias: vec![LogitBias {
            token: TokenRef::Text("c".into()),
            bias: 100.0,
        }],
        ..base
    });
    assert_eq!(c.token_ids, vec![C; 4]);
}

#[test]
fn unlimited_max_tokens_runs_to_the_context_or_the_server_cap() {
    let dir = model_dir("core-unlimited");
    write_tiny_llama_gguf(&dir.join("model.gguf"));
    let options = GenerationOptions {
        max_tokens: MAX_TOKENS_UNLIMITED,
        ignore_eos: true,
        logit_bias: vec![bias(A, 100.0)],
        temperature: 0.0,
        ..Default::default()
    };
    let prompt = RawPrompt::text("hello a b");
    let engine = Engine::with_n_ctx(&dir, 64).unwrap();
    let c = engine.complete_prompt(&prompt, &options).unwrap();
    assert_eq!(c.stop_type, StopType::Limit);
    assert_eq!(c.usage.prompt_tokens + c.usage.completion_tokens, 64);
    let capped = Engine::with_n_ctx(&dir, 64)
        .unwrap()
        .with_max_output_tokens(10);
    let c = capped.complete_prompt(&prompt, &options).unwrap();
    assert_eq!(
        (c.stop_type, c.usage.completion_tokens),
        (StopType::Limit, 10)
    );
}

#[test]
fn cached_tokens_and_timings_are_reported() {
    let engine = engine("core-cached");
    // The follow-up extends the first prompt, the warm pool's reuse case.
    let first = engine
        .complete_prompt(&RawPrompt::text("hello a b c d"), &greedy(0))
        .unwrap();
    assert_eq!(first.cached_tokens, 0);
    let second = engine
        .complete_prompt(&RawPrompt::text("hello a b c d e f"), &greedy(2))
        .unwrap();
    assert_eq!(
        second.cached_tokens, 5,
        "the warm session's prefix is reused"
    );
    for c in [&first, &second] {
        let details = c.usage.prompt_tokens_details.as_ref().unwrap();
        assert_eq!(details.cached_tokens, c.cached_tokens);
        assert_eq!(c.timings.cache_n, c.cached_tokens);
        assert_eq!(
            c.timings.cache_n + c.timings.prompt_n,
            c.usage.prompt_tokens
        );
        assert_eq!(c.timings.predicted_n, c.usage.completion_tokens);
        assert!(c.timings.prompt_ms >= 0.0 && c.timings.predicted_ms >= 0.0);
    }
}

#[test]
fn sampled_extras_are_reproducible_with_a_seed() {
    let engine = engine("core-extras");
    let prompt = RawPrompt::text("hello a b");
    let base = GenerationOptions {
        temperature: 0.9,
        seed: Some(7),
        ignore_eos: true,
        ..greedy(8)
    };
    let dry = joshua::sampling::DryOptions {
        multiplier: 0.8,
        ..Default::default()
    };
    for options in [
        GenerationOptions {
            typical_p: 0.5,
            ..base.clone()
        },
        GenerationOptions {
            top_n_sigma: 1.0,
            ..base.clone()
        },
        GenerationOptions {
            xtc_probability: 1.0,
            xtc_threshold: 0.05,
            ..base.clone()
        },
        GenerationOptions {
            mirostat: 1,
            ..base.clone()
        },
        GenerationOptions {
            mirostat: 2,
            ..base.clone()
        },
        GenerationOptions {
            dry: dry.clone(),
            ..base.clone()
        },
        GenerationOptions {
            repeat_last_n: 0,
            repetition_penalty: 1.5,
            ..base.clone()
        },
        GenerationOptions {
            repeat_last_n: -1,
            presence_penalty: 0.5,
            ..base.clone()
        },
    ] {
        let a = engine.complete_prompt(&prompt, &options).unwrap();
        let b = engine.complete_prompt(&prompt, &options).unwrap();
        assert_eq!(a.token_ids, b.token_ids, "{options:?}");
        assert_eq!(a.usage.completion_tokens, 8);
    }
}

#[test]
fn sampling_defaults_layer_server_flags_over_gguf_over_profile() {
    let dir = model_dir("core-gguf-sampling");
    write_tiny_gguf_with(
        &dir.join("model.gguf"),
        "llama",
        &[
            ("general.sampling.temp".to_string(), Value::F32(0.6)),
            ("general.sampling.top_k".to_string(), Value::I32(20)),
            ("general.sampling.top_p".to_string(), Value::F32(0.95)),
        ],
    );
    let engine = Engine::with_n_ctx(&dir, 64).unwrap();
    assert_eq!(engine.gguf_sampling().temperature, Some(0.6));
    let d = engine.sampling_defaults(SamplingProfile::OpenAi);
    assert_eq!(
        (d.temperature, d.top_k, d.top_p, d.min_p),
        (0.6, 20, 0.95, 0.0)
    );
    let d = engine.sampling_defaults(SamplingProfile::LlamaCpp);
    assert_eq!((d.temperature, d.top_k, d.min_p), (0.6, 20, 0.05));

    let engine = engine.with_sampling_overrides(SamplingOverrides {
        temperature: Some(0.3),
        ..Default::default()
    });
    let d = engine.sampling_defaults(SamplingProfile::Ollama);
    assert_eq!((d.temperature, d.top_k, d.top_p), (0.3, 20, 0.95));

    // A model without recommendations keeps each API's own defaults.
    let plain = self::engine("core-plain-sampling");
    assert!(plain.gguf_sampling().is_empty());
    assert_eq!(
        plain.sampling_defaults(SamplingProfile::Ollama),
        SamplingProfile::Ollama.defaults()
    );
}

#[test]
fn template_controls_shape_the_prompt() {
    let engine = engine("core-template");
    let options = GenerationOptions::default();
    let render = |messages: &[ChatMessage], options: &GenerationOptions| {
        engine.apply_chat_template(messages, None, options).unwrap()
    };
    let user = ChatMessage::text("user", "a");
    let prefill = ChatMessage::text("assistant", "b");
    // The default template: "hello <content> " per message, then "world".
    assert_eq!(render(std::slice::from_ref(&user), &options), "hello a world");
    // A final assistant message is continued...
    assert_eq!(
        render(&[user.clone(), prefill.clone()], &options),
        "hello a hello b"
    );
    // ...unless the request says not to.
    let fresh = GenerationOptions {
        continue_final_message: Some(false),
        ..options.clone()
    };
    assert_eq!(
        render(&[user.clone(), prefill.clone()], &fresh),
        "hello a hello b world"
    );
    let bare = GenerationOptions {
        add_generation_prompt: false,
        ..options.clone()
    };
    assert_eq!(render(std::slice::from_ref(&user), &bare), "hello a ");

    // A per-request template is code: refused unless the server trusts it.
    let custom = GenerationOptions {
        chat_template: Some(
            "{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}".into(),
        ),
        ..options.clone()
    };
    let developer = [ChatMessage::text("developer", "x"), user.clone()];
    for refused in [
        engine.apply_chat_template(&developer, None, &custom),
        engine.validate_options(&custom).map(|_| String::new()),
        engine
            .complete_chat_detailed(&developer, None, &custom)
            .map(|c| c.text),
    ] {
        match refused {
            Err(JoshuaError::InvalidRequest(message)) => {
                assert!(
                    message.contains("--trust-request-chat-template"),
                    "{message}"
                )
            }
            other => panic!("an untrusted template must be refused, got {other:?}"),
        }
    }

    // Trusted, it renders, which also shows the developer role mapped.
    let dir = model_dir("core-template-trusted");
    write_tiny_llama_gguf(&dir.join("model.gguf"));
    let trusting = Engine::with_options(
        &dir,
        EngineOptions::with_n_ctx(64).trust_request_chat_template(true),
    )
    .unwrap();
    assert_eq!(
        trusting
            .apply_chat_template(&developer, None, &custom)
            .unwrap(),
        "system:x;user:a;"
    );
    assert!(trusting.validate_options(&custom).is_ok());
    let broken = GenerationOptions {
        chat_template: Some("{% for m in messages %}".into()),
        ..options.clone()
    };
    assert!(matches!(
        trusting.apply_chat_template(std::slice::from_ref(&user), None, &broken),
        Err(JoshuaError::InvalidRequest(_))
    ));
    // Even a trusted template cannot loop for long.
    let looping = GenerationOptions {
        chat_template: Some(
            "{% for a in range(100000) %}{% for b in range(100000) %}{% endfor %}{% endfor %}"
                .into(),
        ),
        ..options
    };
    match trusting.apply_chat_template(&[user], None, &looping) {
        Err(JoshuaError::InvalidRequest(message)) => assert!(message.contains("fuel"), "{message}"),
        other => panic!("a looping template must stop, got {other:?}"),
    }
}

#[test]
fn requests_that_leave_nothing_to_sample_are_client_errors() {
    let engine = engine("core-nothing-to-sample");
    let prompt = RawPrompt::text("hello a b");
    let refused = |options: GenerationOptions, needle: &str| {
        for result in [
            engine.validate_options(&options),
            engine.complete_prompt(&prompt, &options).map(|_| ()),
        ] {
            match result {
                Err(JoshuaError::InvalidRequest(message)) => {
                    assert!(message.contains(needle), "{needle}: {message}")
                }
                other => panic!("{needle}: expected a client error, got {other:?}"),
            }
        }
    };
    // vLLM's checks on the list itself, before any work.
    refused(
        GenerationOptions {
            allowed_token_ids: Some(vec![]),
            ..greedy(4)
        },
        "empty",
    );
    refused(
        GenerationOptions {
            allowed_token_ids: Some(vec![A, 999_999]),
            ..greedy(4)
        },
        "999999",
    );
    // Controls that contradict each other, sampled or greedy: the only
    // allowed token is EOS, which `min_tokens` holds back.
    for temperature in [0.0, 0.9] {
        let options = GenerationOptions {
            allowed_token_ids: Some(vec![EOS]),
            min_tokens: 2,
            temperature,
            seed: Some(1),
            ..greedy(4)
        };
        assert!(engine.validate_options(&options).is_ok());
        match engine.complete_prompt(&prompt, &options) {
            Err(JoshuaError::InvalidRequest(message)) => {
                assert!(message.contains("no token"), "{message}")
            }
            other => panic!("expected a client error, got {other:?}"),
        }
    }
    // The engine still serves the next request.
    let ok = engine
        .complete_prompt(
            &prompt,
            &GenerationOptions {
                allowed_token_ids: Some(vec![B]),
                ignore_eos: true,
                ..greedy(2)
            },
        )
        .unwrap();
    assert_eq!(ok.token_ids, vec![B, B]);
}

#[test]
fn truncate_prompt_tokens_follows_vllm() {
    let engine = engine("core-truncate");
    let prompt = RawPrompt::tokens(vec![1; 80]);
    let truncate = |keep: u32| GenerationOptions {
        truncate_prompt_tokens: Some(keep),
        ignore_eos: true,
        ..greedy(1)
    };
    // `-1` keeps all but one token of the 64-token window.
    let c = engine
        .complete_prompt(&prompt, &truncate(TRUNCATE_TO_CONTEXT))
        .unwrap();
    assert_eq!(c.usage.prompt_tokens, 63);
    assert_eq!(c.usage.completion_tokens, 1);
    let c = engine.complete_prompt(&prompt, &truncate(10)).unwrap();
    assert_eq!(c.usage.prompt_tokens, 10);
    // Zero would leave no prompt; more than the window cannot fit.
    for (keep, needle) in [(0, "at least 1"), (65, "context window")] {
        for result in [
            engine.validate_options(&truncate(keep)),
            engine.complete_prompt(&prompt, &truncate(keep)).map(|_| ()),
        ] {
            match result {
                Err(JoshuaError::InvalidRequest(message)) => {
                    assert!(message.contains(needle), "{keep}: {message}")
                }
                other => panic!("{keep}: expected a client error, got {other:?}"),
            }
        }
    }
}

fn think_engine(name: &str) -> Engine {
    Engine::with_n_ctx(think_model_dir(name), 64)
        .unwrap()
        .with_chat_template(THINK_TEMPLATE)
}

/// Greedy, fixed-length output in which the model cannot write the
/// reasoning tags itself, so only the budget can close the block.
fn think_options(max_tokens: u32, budget: Option<u32>) -> GenerationOptions {
    GenerationOptions {
        logit_bias: vec![bias(14, f32::NEG_INFINITY), bias(15, f32::NEG_INFINITY)],
        ignore_eos: true,
        reasoning_budget: budget,
        ..greedy(max_tokens)
    }
}

#[test]
fn reasoning_budget_forces_the_end_tag() {
    let engine = think_engine("core-think-budget");
    assert_eq!(
        *engine.reasoning_format(),
        ReasoningFormat::Tags {
            start: "<think>".into(),
            end: "</think>".into()
        }
    );
    let messages = [ChatMessage::text("user", "a b")];

    let open = engine
        .complete_chat_detailed(&messages, None, &think_options(5, None))
        .unwrap();
    assert!(open.reasoning_open, "the template opened <think>");
    assert_eq!(open.reasoning_tokens, Some(5));
    assert!(!open.token_ids.contains(&15));
    let (reasoning, content) = split(engine.reasoning_format(), true, &open.text);
    assert_eq!(
        (reasoning.as_deref(), content.as_str()),
        (Some(open.text.as_str()), "")
    );

    let budgeted = engine
        .complete_chat_detailed(&messages, None, &think_options(5, Some(2)))
        .unwrap();
    assert_eq!(
        budgeted.token_ids[2], 15,
        "the end tag is forced after 2 tokens"
    );
    assert_eq!(budgeted.reasoning_tokens, Some(2));
    assert_eq!(
        budgeted
            .usage
            .completion_tokens_details
            .as_ref()
            .unwrap()
            .reasoning_tokens,
        2
    );
    let (reasoning, content) = split(engine.reasoning_format(), true, &budgeted.text);
    assert_eq!(
        reasoning.unwrap(),
        text_of(&engine, &budgeted.token_ids[..2])
    );
    assert_eq!(content, text_of(&engine, &budgeted.token_ids[3..]));

    // A zero budget closes the block at once; a message comes before it.
    let zero = engine
        .complete_chat_detailed(&messages, None, &think_options(3, Some(0)))
        .unwrap();
    assert_eq!(zero.token_ids[0], 15);
    let message = engine
        .complete_chat_detailed(
            &messages,
            None,
            &GenerationOptions {
                reasoning_budget_message: Some("c".into()),
                ..think_options(4, Some(1))
            },
        )
        .unwrap();
    assert_eq!(&message.token_ids[1..3], &[C, 15]);

    // The streaming splitter starts inside the reasoning for this prompt.
    let splitter = engine
        .reasoning_splitter(&messages, None, &think_options(5, None))
        .unwrap();
    assert!(splitter.in_reasoning());
}

#[test]
fn fim_tokens_and_infill_prompt() {
    let dir = model_dir("core-fim");
    write_tiny_gguf_with(
        &dir.join("model.gguf"),
        "llama",
        &[
            (
                "tokenizer.ggml.fim_pre_token_id".to_string(),
                Value::U32(10),
            ),
            (
                "tokenizer.ggml.fim_suf_token_id".to_string(),
                Value::U32(11),
            ),
            (
                "tokenizer.ggml.fim_mid_token_id".to_string(),
                Value::U32(12),
            ),
            (
                "tokenizer.ggml.fim_sep_token_id".to_string(),
                Value::U32(13),
            ),
        ],
    );
    // llama-server gives extra files the context beyond one 2048-token
    // batch, so this needs a window larger than that.
    let engine = Engine::with_n_ctx(&dir, 4096).unwrap();
    let fim = engine.fim_tokens();
    assert_eq!(
        (fim.pre, fim.suf, fim.mid, fim.sep),
        (Some(10), Some(11), Some(12), Some(13))
    );
    let extra = [InfillChunk {
        filename: "c".into(),
        text: "d".into(),
    }];
    let prompt = engine
        .format_infill("hello", "world", &extra, "a", -1, false)
        .unwrap();
    // [SEP] c \n d [SEP] filename \n [PRE] hello a [SUF] world [MID]; the
    // word-level test vocabulary maps `filename` to <unk>.
    let pre = prompt.iter().position(|&t| t == 10).unwrap();
    assert_eq!(&prompt[pre..], &[10, 1, A, 11, 2, 12]);
    assert_eq!(prompt[0], 13);
    assert!(
        prompt[..pre].contains(&7),
        "the extra file's text comes first"
    );

    // A model without FIM tokens cannot infill.
    let plain = self::engine("core-no-fim");
    assert!(matches!(
        plain.format_infill("a", "b", &[], "", -1, false),
        Err(JoshuaError::InvalidRequest(_))
    ));
}
