mod common;

use std::path::Path;
use std::time::Duration;

use joshua::engine::external_draft::{
    DraftSession, FinishReason, VerifyRequest, MAX_EXTERNAL_DRAFT,
};
use joshua::{Engine, EngineOptions, GenerationOptions, SpeculativeConfig};
use uuid::Uuid;

const PROMPT: &[u32] = &[4, 5, 6, 7, 4, 5];
const ARCHITECTURES: &[&str] = &[
    "qwen3",
    "qwen3moe",
    "qwen3next",
    "qwen35",
    "qwen35moe",
    "deepseek",
    "deepseek2",
    "glm-dsa",
    "deepseek4",
    "deepseek41",
    "kimi-linear",
    "kimi-k3",
    "glm5next",
    "nemotron_h",
];

fn write_model(path: &Path, architecture: &str) {
    match architecture {
        "deepseek" => common::write_tiny_deepseek_gguf(path),
        "deepseek2" => common::write_tiny_deepseek2_gguf(path),
        "glm-dsa" => common::write_tiny_glm_dsa_gguf(path),
        "deepseek4" => common::write_tiny_deepseek4_gguf(path),
        "deepseek41" => common::write_tiny_deepseek41_gguf(path),
        "kimi-linear" | "kimi-k3" => common::write_tiny_kimi_gguf(path, architecture, false),
        "glm5next" => common::write_tiny_glm5next_gguf(path, architecture),
        "nemotron_h" => common::write_tiny_nemotron_h_gguf(path),
        _ => common::write_tiny_qwen_gguf(path, architecture),
    }
}

fn set_eos(path: &Path, token: u32) {
    let mut bytes = std::fs::read(path).unwrap();
    let key = b"tokenizer.ggml.eos_token_id";
    let offsets: Vec<_> = bytes
        .windows(key.len())
        .enumerate()
        .filter_map(|(offset, window)| (window == key).then_some(offset + key.len()))
        .collect();
    assert!(!offsets.is_empty());
    for offset in offsets {
        assert_eq!(&bytes[offset..offset + 4], &4u32.to_le_bytes());
        bytes[offset + 4..offset + 8].copy_from_slice(&token.to_le_bytes());
    }
    std::fs::write(path, bytes).unwrap();
}

fn options() -> GenerationOptions {
    GenerationOptions {
        temperature: 0.0,
        max_tokens: 24,
        seed: Some(42),
        ..Default::default()
    }
}

fn request(session: &DraftSession<'_>, sequence: u64, tokens: &[u32]) -> VerifyRequest {
    VerifyRequest {
        generation: session.generation(),
        sequence,
        offset: session.offset(),
        tokens: tokens.to_vec(),
    }
}

fn begin<'a>(engine: &'a Engine, options: &GenerationOptions) -> DraftSession<'a> {
    engine
        .begin_draft_session(PROMPT, options, 4, Duration::from_secs(60))
        .unwrap()
}

fn baseline(engine: &Engine, options: &GenerationOptions) -> (Vec<u32>, String) {
    let mut session = begin(engine, options);
    let mut tokens = Vec::new();
    for sequence in 0..100 {
        let response = session.verify(&request(&session, sequence, &[])).unwrap();
        tokens.extend(&response.tokens);
        if response.finish.is_some() {
            return (tokens, session.text().into());
        }
    }
    panic!("generation was not bounded");
}

fn check_blocks(engine: &Engine, options: &GenerationOptions) {
    let (expected, expected_text) = baseline(engine, options);
    let prompt = engine.detokenize(PROMPT, false).unwrap();
    let (plain, usage, _, _) = engine.complete_raw(&prompt, options).unwrap();
    assert_eq!(expected_text, plain);
    assert_eq!(expected.len(), usage.completion_tokens as usize);
    let speculative = Engine::with_options(
        engine.model_path(),
        EngineOptions::with_n_ctx(64).speculative(Some(SpeculativeConfig {
            max_draft: 4,
            min_ngram: 1,
            max_ngram: 3,
            cost_gate: false,
        })),
    );
    let speculative = speculative.unwrap();
    let (spec_text, spec_usage, _, _) = speculative.complete_raw(&prompt, options).unwrap();
    assert_eq!(expected_text, spec_text);
    assert_eq!(expected.len(), spec_usage.completion_tokens as usize);
    if expected.len() <= 6 {
        let mut session = begin(engine, options);
        let mut got = Vec::<u32>::new();
        for sequence in 0..32 {
            let end = (got.len() + 4).min(expected.len());
            let mut draft = expected[got.len()..end].to_vec();
            draft.resize(4, 0);
            let next = request(&session, sequence, &draft);
            let response = session.verify(&next).unwrap();
            assert_eq!(session.verify(&next).unwrap(), response);
            got.extend_from_slice(&response.tokens);
            if response.finish.is_some() {
                assert_eq!(got, expected);
                assert_eq!(session.text(), expected_text);
                return;
            }
        }
        panic!("short generation must terminate");
    }
    for rejection in 0..=4 {
        let mut session = begin(engine, options);
        let mut draft = expected[..4].to_vec();
        if rejection < 4 {
            draft[rejection] = (draft[rejection] + 1) % 16;
        }
        let first = request(&session, 0, &draft);
        let response = session.verify(&first).unwrap();
        assert_eq!(response.accepted, rejection);
        assert_eq!(response.tokens, expected[..rejection + 1]);
        assert_eq!(
            session.verify(&first).unwrap(),
            response,
            "lost-response retry"
        );
        assert_eq!(session.offset(), PROMPT.len() + response.tokens.len());
        let mut got = response.tokens;
        for sequence in 1..100 {
            let end = (got.len() + 4).min(expected.len());
            let next = request(&session, sequence, &expected[got.len()..end]);
            let response = session.verify(&next).unwrap();
            assert_eq!(session.verify(&next).unwrap(), response);
            got.extend(&response.tokens);
            if response.finish.is_some() {
                assert_eq!(got, expected, "rejection at {rejection}");
                assert_eq!(session.text(), expected_text);
                break;
            }
        }
        assert_eq!(got, expected);
    }
}

#[test]
fn tiny_architecture_blocks_match_plain_and_continue_after_every_rejection() {
    for &arch in ARCHITECTURES {
        let dir = common::model_dir(&format!("external-{arch}"));
        let path = dir.join("model.gguf");
        write_model(&path, arch);
        let engine = Engine::with_n_ctx(&dir, 64).unwrap();
        check_blocks(&engine, &options());
        let sampled = GenerationOptions {
            temperature: 0.85,
            top_p: 0.9,
            top_k: 10,
            min_p: 0.03,
            repetition_penalty: 1.1,
            presence_penalty: 0.1,
            frequency_penalty: 0.05,
            ..options()
        };
        let sampled = (42..100)
            .find_map(|seed| {
                let candidate = GenerationOptions {
                    seed: Some(seed),
                    ..sampled.clone()
                };
                (baseline(&engine, &candidate).0.len() > 6).then_some(candidate)
            })
            .expect("sampled fixture must exercise continuation");
        check_blocks(&engine, &sampled);
        drop(engine);
        let chunked =
            Engine::with_options(&dir, EngineOptions::with_n_ctx(64).prefill_chunk(2)).unwrap();
        check_blocks(&chunked, &sampled);
        drop(chunked);
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn validation_replay_window_and_cleanup_are_bounded() {
    let dir = common::model_dir("external-lifecycle");
    common::write_tiny_qwen_gguf(&dir.join("model.gguf"), "qwen3");
    let engine = Engine::with_n_ctx(&dir, 64)
        .unwrap()
        .with_max_concurrency(1);
    let mut session = begin(&engine, &options());
    assert_eq!(engine.in_flight(), 1);
    assert!(engine
        .begin_draft_session(PROMPT, &options(), 4, Duration::from_secs(1))
        .is_err());
    let original = request(&session, 0, &[]);
    let mut invalid = original.clone();
    invalid.generation = Uuid::new_v4();
    assert!(session.verify(&invalid).is_err());
    invalid = original.clone();
    invalid.sequence = 1;
    assert!(session.verify(&invalid).is_err());
    invalid = original.clone();
    invalid.offset += 1;
    assert!(session.verify(&invalid).is_err());
    invalid = original.clone();
    invalid.tokens = vec![0; 5];
    assert!(session.verify(&invalid).is_err());
    invalid = original.clone();
    invalid.tokens = vec![16];
    assert!(session.verify(&invalid).is_err());
    assert_eq!(session.offset(), original.offset);
    let first = session.verify(&original).unwrap();
    assert_eq!(session.verify(&original).unwrap(), first);
    invalid = original.clone();
    invalid.tokens = vec![4];
    assert!(session.verify(&invalid).is_err(), "conflicting duplicate");
    let second = request(&session, 1, &[]);
    session.verify(&second).unwrap();
    assert!(session.verify(&original).is_err(), "old replay was evicted");
    session.cancel();
    session.cancel();
    assert_eq!(engine.in_flight(), 0);
    assert!(session.verify(&second).is_err(), "cancel destroys replay");
    let mut next = begin(&engine, &options());
    assert!(
        next.verify(&original).is_err(),
        "fresh generation rejects old request"
    );
    next.end();
    next.end();
    assert_eq!(engine.in_flight(), 0);
    drop(begin(&engine, &options()));
    assert_eq!(engine.in_flight(), 0);
    let mut expired = engine
        .begin_draft_session(PROMPT, &options(), 4, Duration::from_nanos(1))
        .unwrap();
    let expired_request = request(&expired, 0, &[]);
    assert!(expired.expire());
    assert!(expired.verify(&expired_request).is_err());
    assert_eq!(engine.in_flight(), 0);
    assert!(engine
        .begin_draft_session(
            PROMPT,
            &options(),
            MAX_EXTERNAL_DRAFT + 1,
            Duration::from_secs(1)
        )
        .is_err());
    assert!(engine
        .begin_draft_session(PROMPT, &options(), 4, Duration::from_secs(301))
        .is_err());
    assert!(engine
        .begin_draft_session(&[], &options(), 4, Duration::from_secs(1))
        .is_err());
    assert!(engine
        .begin_draft_session(&[99], &options(), 4, Duration::from_secs(1))
        .is_err());
    assert!(engine
        .begin_draft_session(PROMPT, &options(), 0, Duration::from_secs(1))
        .is_err());
    assert!(engine
        .begin_draft_session(PROMPT, &options(), 4, Duration::ZERO)
        .is_err());
    let constrained = GenerationOptions {
        constraint: Some(joshua::types::ConstraintSpec::JsonObject),
        ..options()
    };
    assert!(engine
        .begin_draft_session(PROMPT, &constrained, 4, Duration::from_secs(1))
        .is_err());
    for context in [PROMPT.len() as u32, 65] {
        let opts = GenerationOptions {
            context_window: Some(context),
            ..options()
        };
        assert!(engine
            .begin_draft_session(PROMPT, &opts, 4, Duration::from_secs(1))
            .is_err());
    }
    assert_eq!(engine.in_flight(), 0);
    drop((expired, next, session));
    drop(engine);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn max_context_and_stop_boundaries_replay_final_response() {
    for &arch in ARCHITECTURES {
        let dir = common::model_dir(&format!("external-finish-{arch}"));
        write_model(&dir.join("model.gguf"), arch);
        let engine = Engine::with_n_ctx(&dir, 64).unwrap();
        let opts = (0..100)
            .find_map(|seed| {
                let candidate = GenerationOptions {
                    temperature: 0.85,
                    seed: Some(seed),
                    ..options()
                };
                (baseline(&engine, &candidate).0.len() >= 6).then_some(candidate)
            })
            .unwrap();
        let (expected, _) = baseline(&engine, &opts);
        for (limit, context, finish) in [
            (0, None, FinishReason::MaxTokens),
            (1, None, FinishReason::MaxTokens),
            (3, None, FinishReason::MaxTokens),
            (4, None, FinishReason::MaxTokens),
            (5, None, FinishReason::MaxTokens),
            (24, Some(PROMPT.len() as u32 + 3), FinishReason::Context),
        ] {
            let opts = GenerationOptions {
                max_tokens: limit,
                context_window: context,
                ..opts.clone()
            };
            let mut session = begin(&engine, &opts);
            let first = request(&session, 0, &expected[..4]);
            let result = session.verify(&first).unwrap();
            assert_eq!(result.finish, Some(finish));
            let count = limit.min(context.map_or(24, |size| size - PROMPT.len() as u32)) as usize;
            assert_eq!(result.tokens, expected[..count]);
            assert_eq!(result.offset, PROMPT.len() + count);
            assert_eq!(session.verify(&first).unwrap(), result);
            assert_eq!(engine.in_flight(), 0);
            session.end();
            assert!(session.verify(&first).is_err());
        }
        let stop = engine
            .detokenize(&expected[..2], false)
            .unwrap()
            .replace(' ', "");
        let opts = GenerationOptions {
            stop_sequences: vec![stop.clone()],
            ..opts
        };
        let (want, text) = baseline(&engine, &opts);
        let mut session = begin(&engine, &opts);
        let result = session
            .verify(&request(&session, 0, &expected[..4]))
            .unwrap();
        assert_eq!(result.finish, Some(FinishReason::Stop));
        assert_eq!(result.stop_sequence, Some(stop));
        assert_eq!(result.tokens, want);
        assert_eq!(session.text(), text);
        assert_eq!(result.offset, PROMPT.len() + want.len());
        assert_eq!(engine.in_flight(), 0);
        for drafts in [&expected[..0], &expected[..1]] {
            let mut session = begin(&engine, &opts);
            let first = request(&session, 0, drafts);
            let first_response = session.verify(&first).unwrap();
            let result = if first_response.finish.is_some() {
                assert_eq!(session.verify(&first).unwrap(), first_response);
                first_response
            } else {
                let second = request(&session, 1, &[]);
                let result = session.verify(&second).unwrap();
                assert_eq!(session.verify(&second).unwrap(), result);
                result
            };
            assert_eq!(result.finish, Some(FinishReason::Stop));
            assert_eq!(result.completion_tokens, want.len() as u32);
            assert_eq!(result.offset, PROMPT.len() + want.len() - 1);
            assert_eq!(session.text(), text);
            assert_eq!(engine.in_flight(), 0);
        }
        drop(session);
        drop(engine);
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn eos_at_first_or_later_position_replays_without_emitting_or_committing_eos() {
    for &arch in ARCHITECTURES {
        let dir = common::model_dir(&format!("external-eos-{arch}"));
        let path = dir.join("model.gguf");
        write_model(&path, arch);
        let engine = Engine::with_n_ctx(&dir, 64).unwrap();
        let (opts, expected, eos_position) = (42..100)
            .find_map(|seed| {
                let opts = GenerationOptions {
                    temperature: 0.85,
                    seed: Some(seed),
                    ..options()
                };
                let expected = baseline(&engine, &opts).0;
                let position = (1..expected.len().min(4))
                    .find(|&index| !expected[..index].contains(&expected[index]))?;
                Some((opts, expected, position))
            })
            .expect("fixture needs a later unique EOS candidate");
        drop(engine);
        for position in [0, eos_position] {
            set_eos(&path, expected[position]);
            let engine = Engine::with_n_ctx(&dir, 64).unwrap();
            for draft_length in [0, position, position + 1, 4] {
                for reject in [false, true] {
                    let mut session = begin(&engine, &opts);
                    let mut emitted = Vec::<u32>::new();
                    for sequence in 0..8 {
                        let start = emitted.len();
                        let end = (start + draft_length).min(expected.len());
                        let mut draft = expected[start..end].to_vec();
                        if reject && !draft.is_empty() {
                            draft[0] = (draft[0] + 1) % 16;
                        }
                        let next = request(&session, sequence, &draft);
                        let response = session.verify(&next).unwrap();
                        assert_eq!(session.verify(&next).unwrap(), response);
                        emitted.extend_from_slice(&response.tokens);
                        if response.finish.is_some() {
                            assert_eq!(response.finish, Some(FinishReason::Eos));
                            assert_eq!(
                                response.accepted,
                                if reject {
                                    0
                                } else {
                                    draft.len().min(position - start + 1)
                                }
                            );
                            assert_eq!(response.offset, PROMPT.len() + position);
                            assert_eq!(response.completion_tokens, position as u32);
                            assert_eq!(emitted, expected[..position]);
                            assert_eq!(engine.in_flight(), 0);
                            session.cancel();
                            assert!(session.verify(&next).is_err());
                            break;
                        }
                    }
                    assert_eq!(emitted, expected[..position]);
                    assert_eq!(engine.in_flight(), 0);
                }
            }
            drop(engine);
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[cfg(feature = "metal")]
#[test]
fn metal_draft_verification_matches_local_decode_and_replays() {
    use joshua::engine::ComputeBackend;
    use joshua::placement::DensePlacement;

    let dir = common::model_dir("external-metal");
    common::write_tiny_qwen_gguf(&dir.join("model.gguf"), "qwen3");
    let engine = Engine::with_options(
        &dir,
        EngineOptions::with_n_ctx(64)
            .backend(ComputeBackend::Metal)
            .dense_placement(DensePlacement::Device),
    )
    .unwrap();
    assert!(engine.dense_device().is_metal());
    check_blocks(&engine, &options());
    drop(engine);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn inference_failure_releases_kv_and_admission_and_cannot_be_replayed() {
    for &arch in ARCHITECTURES {
        let dir = common::model_dir(&format!("external-failure-{arch}"));
        let path = dir.join("model.gguf");
        write_model(&path, arch);
        let mut bytes = std::fs::read(&path).unwrap();
        let header = joshua::gguf_ext::read_header(&mut std::io::Cursor::new(&bytes)).unwrap();
        let tensor = &header.tensors["output_norm.weight"];
        assert_eq!(tensor.dtype, 0);
        let start = (header.tensor_data_offset + tensor.offset) as usize;
        let length = tensor.dims.iter().product::<usize>() * 4;
        for value in bytes[start..start + length].chunks_exact_mut(4) {
            value.copy_from_slice(&f32::NAN.to_le_bytes());
        }
        std::fs::write(&path, bytes).unwrap();
        let engine = Engine::with_n_ctx(&dir, 64)
            .unwrap()
            .with_max_concurrency(1);
        let mut session = begin(&engine, &options());
        let first = request(&session, 0, &[4, 5]);
        let error = session.verify(&first).unwrap_err();
        assert!(error.to_string().contains("non-finite"), "{arch}: {error}");
        assert_eq!(engine.in_flight(), 0);
        assert!(session.verify(&first).is_err());
        session.cancel();
        session.end();
        drop(begin(&engine, &options()));
        assert_eq!(engine.in_flight(), 0);
        drop(session);
        drop(engine);
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn unsupported_architecture_does_not_leak_admission() {
    let dir = common::model_dir("external-unsupported");
    common::write_tiny_llama_gguf(&dir.join("model.gguf"));
    let engine = Engine::with_n_ctx(&dir, 64).unwrap();
    assert!(engine
        .begin_draft_session(PROMPT, &options(), 4, Duration::from_secs(1))
        .is_err());
    assert_eq!(engine.in_flight(), 0);
    drop(engine);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn maximum_draft_bound_honors_server_ceiling_and_validates_the_unused_tail() {
    let dir = common::model_dir("external-maximum-draft");
    common::write_tiny_qwen_gguf(&dir.join("model.gguf"), "qwen3");
    let engine = Engine::with_n_ctx(&dir, 64)
        .unwrap()
        .with_max_output_tokens(3);
    let expected = baseline(&engine, &options()).0;
    assert_eq!(expected.len(), 3);
    let mut session = engine
        .begin_draft_session(
            PROMPT,
            &options(),
            MAX_EXTERNAL_DRAFT,
            Duration::from_secs(60),
        )
        .unwrap();
    let mut draft = expected.clone();
    draft.resize(MAX_EXTERNAL_DRAFT, 4);
    let first = request(&session, 0, &draft);
    let mut invalid = first.clone();
    invalid.tokens[MAX_EXTERNAL_DRAFT - 1] = 99;
    assert!(session.verify(&invalid).is_err());
    assert_eq!(session.offset(), PROMPT.len());
    let response = session.verify(&first).unwrap();
    assert_eq!(response.tokens, expected);
    assert_eq!(response.accepted, 3);
    assert_eq!(response.finish, Some(FinishReason::MaxTokens));
    assert_eq!(session.verify(&first).unwrap(), response);
    assert_eq!(engine.in_flight(), 0);
    drop(session);
    drop(engine);
    std::fs::remove_dir_all(dir).unwrap();
}
