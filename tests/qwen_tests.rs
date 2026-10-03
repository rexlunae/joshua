//! Native (pure-Rust) validation for the Qwen family loader
//! (`quantized_qwen`) across every architecture it serves, the GLM ones
//! (`chatglm`, `glm4`, `glm4moe`) included.  These run on the
//! default `cargo test` — no llama.cpp, no network.

mod common;

use candle_core::{Device, Tensor};
use common::{assert_close, logits};
use joshua::model::Architecture;

fn load(model: &std::path::Path) -> joshua::model::QuantizedModel {
    common::load_model(model, false)
}

/// Every architecture the Qwen loader serves, with whether its layers
/// include recurrent (Gated DeltaNet) state.
const ARCHES: &[(&str, bool)] = &[
    ("qwen", false),
    ("qwen2moe", false),
    ("qwen2vl", false),
    ("qwen3", false),
    ("qwen3moe", false),
    ("qwen3vl", false),
    ("qwen3vlmoe", false),
    ("qwen3next", true),
    ("qwen35", true),
    ("qwen35moe", true),
    ("qwen4exp", true),
    ("chatglm", false),
    ("glm4", false),
    ("glm4moe", false),
];

#[test]
fn every_qwen_architecture_is_supported() {
    for &(name, _) in ARCHES {
        let arch = Architecture::from_name(name).unwrap_or_else(|| panic!("{name} unsupported"));
        assert!(
            arch.is_native(),
            "{name} should load through the native Qwen loader"
        );
        assert!(arch.shares_weights(), "{name}");
    }
    // The remaining Qwen-branded registry entries are recognised with a
    // specific "known but not loadable" error rather than "unrecognised".
    for name in ["qwen3tts", "rwkv6qwen2"] {
        assert_eq!(Architecture::from_name(name), None, "{name}");
        assert!(Architecture::is_known_llama_cpp_arch(name), "{name}");
    }
}

/// Every architecture loads, produces finite non-degenerate logits, and
/// gives the same next-token logits whether the prompt is prefilled at once
/// or fed token by token through the cache / recurrent state — across a
/// fresh session derived from the same weights.
#[test]
fn qwen_family_prefill_matches_incremental_decode() {
    for &(arch, _) in ARCHES {
        let dir = common::model_dir(&format!("qwen-family-{arch}"));
        let model = dir.join("model.gguf");
        common::write_tiny_qwen_gguf(&model, arch);
        let tokens = [1u32, 4, 2, 7, 5, 9];

        let mut prefill_model = load(&model);
        let prefill = logits(&mut prefill_model, &tokens, 0);
        assert_eq!(
            prefill.len(),
            16,
            "{arch}: logits must cover the 16-token vocab"
        );
        assert!(
            prefill.iter().all(|v| v.is_finite()),
            "{arch}: logits not finite: {prefill:?}"
        );
        let first = prefill[0];
        assert!(
            prefill.iter().any(|v| (v - first).abs() > 1e-6),
            "{arch}: logits are degenerate"
        );

        let mut step_model = prefill_model
            .new_session()
            .expect("native loaders share weights");
        let mut last = Vec::new();
        for (pos, &tok) in tokens.iter().enumerate() {
            last = logits(&mut step_model, &[tok], pos);
        }
        assert_close(
            &prefill,
            &last,
            1e-4,
            &format!("{arch} prefill vs incremental"),
        );

        // Two-part prefill (a chunk boundary mid-prompt) agrees as well.
        let mut chunked = prefill_model.new_session().unwrap();
        let _ = logits(&mut chunked, &tokens[..4], 0);
        let tail = logits(&mut chunked, &tokens[4..], 4);
        assert_close(&prefill, &tail, 1e-4, &format!("{arch} chunked prefill"));

        std::fs::remove_dir_all(&dir).ok();
    }
}

/// Clearing the cache resets recurrent state too, so an unrelated prompt
/// gets exactly a fresh model's logits.
#[test]
fn qwen_family_clear_resets_all_state() {
    for &(arch, _) in ARCHES {
        let dir = common::model_dir(&format!("qwen-family-clear-{arch}"));
        let model = dir.join("model.gguf");
        common::write_tiny_qwen_gguf(&model, arch);

        let mut m = load(&model);
        assert!(m.supports_kv_clear(), "{arch}");
        let fresh = logits(&mut m, &[3, 8, 1], 0);
        let _ = logits(&mut m, &[5, 6], 3);
        assert!(m.clear_kv_cache());
        let again = logits(&mut m, &[3, 8, 1], 0);
        assert_close(&fresh, &again, 1e-5, &format!("{arch} after clear"));

        std::fs::remove_dir_all(&dir).ok();
    }
}

/// Rolling a verification pass back to `keep` must leave the recurrent layers
/// in exactly the state plain incremental decoding would have produced.
///
/// This is the invariant the whole snapshot/replay path exists to protect: a
/// DeltaNet layer's matrices *and* its causal conv tail are running state, and
/// a rollback that restores only the pre-pass copy would also discard the
/// tokens that were accepted.  Unlike an end-to-end generation test it forces
/// the case deterministically, because the draft only has to be *checked* and
/// then rolled back — nothing has to guess it.
///
/// For each `keep` in `0..=n` the test runs one multi-position pass over `n`
/// tokens, rolls back to `keep`, then decodes token `keep` and compares against
/// a reference model fed the same tokens one at a time.
#[test]
fn qwen_recurrent_verify_rollback_matches_incremental_decode() {
    for &(arch, recurrent) in ARCHES {
        if !recurrent {
            continue;
        }
        let dir = common::model_dir(&format!("qwen-verify-rollback-{arch}"));
        let model = dir.join("model.gguf");
        common::write_tiny_qwen_gguf(&model, arch);
        // qwen4exp is not speculating yet: its PLE conv window and QSA indexer
        // keys are captured and replayed, but the rollback still diverges from
        // incremental decode, so it stays off rather than shipping a silent
        // mismatch.  See `Weights::can_rewind`.
        if !load(&model).supports_speculative() {
            std::fs::remove_dir_all(&dir).ok();
            continue;
        }
        let toks: [u32; 5] = [1, 4, 2, 7, 3];
        // The verification pass covers the first `n`; `toks[keep]` is the token
        // that follows the rolled-back prefix, so it must be in range for every
        // `keep` including `n` itself.
        let n = 4;

        for keep in 0..=n {
            // Reference: plain incremental decode of exactly the retained
            // prefix, then decode the token that follows it.
            let mut reference = load(&model);
            for (i, &t) in toks.iter().take(keep).enumerate() {
                common::logits(&mut reference, &[t], i);
            }
            let want = common::logits(&mut reference, &[toks[keep]], keep);

            // Subject: one multi-position verification pass over all `n` tokens,
            // then roll it back to `keep` and decode the same next token.
            let mut m = load(&model);
            let input = Tensor::new(&toks[..n], &Device::Cpu).unwrap().unsqueeze(0).unwrap();
            m.forward_all_logits(&input, 0).unwrap();
            assert!(
                m.truncate_kv_cache(keep).unwrap(),
                "{arch}: a pending verification pass must be rollable back"
            );
            let got = common::logits(&mut m, &[toks[keep]], keep);
            common::assert_close(&want, &got, 1e-4, &format!("{arch} rollback to {keep}"));
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
/// reuse, speculative rollback); recurrent models report that they cannot
/// and refuse rather than silently keeping stale state.
///
/// Speculative decoding is the one exception: a *just-run* verification pass
/// is always rolled back inside its own span, and a recurrent layer snapshots
/// its running state and replays the accepted prefix, so every architecture
/// here can verify and reject a draft.  An *arbitrary* backward rewind is a
/// different request and stays refused for recurrent models.
#[test]
fn qwen_family_truncation_support_matches_state_kind() {
    for &(arch, recurrent) in ARCHES {
        let dir = common::model_dir(&format!("qwen-family-trunc-{arch}"));
        let model = dir.join("model.gguf");
        common::write_tiny_qwen_gguf(&model, arch);

        let mut m = load(&model);
        assert_eq!(m.supports_kv_truncate(), !recurrent, "{arch}");
        // Attention-only and snapshotting DeltaNet hybrids can roll back their
        // own verification pass; a PLE block's convolution history is not
        // rewound yet, so a model carrying one does not speculate.
        let has_ple = arch == "qwen4exp";
        assert_eq!(m.supports_speculative(), !has_ple, "{arch}");
        let full = logits(&mut m, &[1, 4, 2, 7], 0);
        if recurrent {
            assert!(
                !m.truncate_kv_cache(2).unwrap(),
                "{arch}: truncation must be refused"
            );
        } else {
            // Rewind to the shared 2-token prefix and diverge from there.
            assert!(m.truncate_kv_cache(2).unwrap());
            let edited = logits(&mut m, &[2, 7], 2);
            assert_close(&full, &edited, 1e-4, &format!("{arch} truncate + re-feed"));
        }

        std::fs::remove_dir_all(&dir).ok();
    }
}

/// Every architecture's logits match golden values from an independent
/// float64 NumPy transcription of llama.cpp's `src/models/qwen*.cpp` graphs
/// (`tests/data/qwen_family_reference_logits.txt`), run on the same tiny
/// fixtures.  The fixtures are built so each architecture's distinguishing
/// feature moves the logits well past the tolerance (the M-RoPE sections
/// freeze a frequency; the DeltaNet layers have more key heads than one, so
/// grouped vs tiled head sharing differ; …).
#[test]
fn qwen_family_matches_llama_cpp_reference() {
    let golden = include_str!("data/qwen_family_reference_logits.txt");
    let mut checked = 0;
    for line in golden
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let mut parts = line.split_whitespace();
        let arch = parts.next().unwrap();
        let want: Vec<f32> = parts.map(|x| x.parse().unwrap()).collect();

        let dir = common::model_dir(&format!("qwen-family-ref-{arch}"));
        let model = dir.join("model.gguf");
        common::write_tiny_qwen_gguf(&model, arch);
        let got = logits(&mut load(&model), &[1, 4, 2, 7, 5, 9], 0);
        assert_close(&got, &want, 1e-6, &format!("{arch} vs llama.cpp reference"));
        checked += 1;
        std::fs::remove_dir_all(&dir).ok();
    }
    // qwen4exp has its own all-position reference below.
    assert_eq!(checked, ARCHES.len() - 1, "every architecture has a reference");
}

/// `qwen4exp` (hyper-connections, QSA block-sparse attention, PLE n-gram
/// hash embeddings) matches an independent float64 NumPy transcription of
/// llama.cpp's `src/models/qwen4exp.cpp` at **every** position
/// (`tests/data/qwen4exp_reference_logits.txt`).  The prompt is longer than
/// the fixture's QSA budget (2 cells + tail), so most queries attend
/// sparsely, and it contains an EOS to exercise the PLE hash reset.  Every
/// position is checked because the QSA layer is the last layer: only the
/// earlier rows expose each query's own selection.
#[test]
fn qwen4exp_matches_llama_cpp_reference_at_every_position() {
    let tokens = [1u32, 4, 2, 7, 5, 9, 3, 6, 8, 4, 1];
    let golden = include_str!("data/qwen4exp_reference_logits.txt");
    let want: Vec<Vec<f32>> = golden
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| l.split_whitespace().map(|x| x.parse().unwrap()).collect())
        .collect();
    assert_eq!(want.len(), tokens.len());

    let dir = common::model_dir("qwen4exp-ref");
    let model = dir.join("model.gguf");
    common::write_tiny_qwen_gguf(&model, "qwen4exp");
    let mut m = load(&model);
    let input = Tensor::new(&tokens[..], &Device::Cpu).unwrap().unsqueeze(0).unwrap();
    let got: Vec<Vec<f32>> = m
        .forward_all_logits(&input, 0)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_vec2()
        .unwrap();
    for (pos, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_close(g, w, 1e-6, &format!("qwen4exp position {pos} vs llama.cpp reference"));
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// The layer-streaming prefill (layer-outer, chunk-inner) reproduces a
/// single forward for every architecture — including the ones whose layers
/// carry recurrent state and read the token history (qwen4exp's PLE), which
/// the sweep feeds chunk by chunk.
#[test]
fn qwen_family_streamed_prefill_matches_forward() {
    use joshua::stream_prefill::Chunk;
    let tokens = [1u32, 4, 2, 7, 5, 9, 3, 6, 8, 4, 1];
    for &(arch, _) in ARCHES {
        let dir = common::model_dir(&format!("qwen-family-stream-{arch}"));
        let model = dir.join("model.gguf");
        common::write_tiny_qwen_gguf(&model, arch);

        let mut single = load(&model);
        let want = logits(&mut single, &tokens, 0);
        let mut streamed = single.new_session().unwrap();
        let chunks = [
            Chunk { tokens: &tokens[..4], pos: 0 },
            Chunk { tokens: &tokens[4..7], pos: 4 },
            Chunk { tokens: &tokens[7..], pos: 7 },
        ];
        let got: Vec<f32> = streamed
            .prefill_streamed(&chunks, &Device::Cpu)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_close(&want, &got, 1e-4, &format!("{arch} streamed prefill"));
        // The streamed session continues decoding like the single one.
        let a = logits(&mut single, &[2], tokens.len());
        let b = logits(&mut streamed, &[2], tokens.len());
        assert_close(&a, &b, 1e-4, &format!("{arch} decode after streamed prefill"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
