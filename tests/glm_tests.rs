//! Native (pure-Rust) validation for the GLM architectures that are not
//! covered by the Qwen family tests (`chatglm`, `glm4` and `glm4moe` run
//! through that loader and are checked alongside the Qwens in
//! `qwen_tests.rs`): `glm-dsa` (GLM-5 / 5.1 / 5.2) and `glm5next` /
//! `glm5-next` (GLM-5.3-Flash) on the `deepseek2` loader.  These run on the
//! default `cargo test` — no llama.cpp, no network.

mod common;

use candle_core::{Device, Tensor};
use common::{assert_close, logits};
use joshua::model::Architecture;

fn load(model: &std::path::Path) -> joshua::model::QuantizedModel {
    common::load_model(model, false)
}

/// Longer than the fixture's indexer top-k (3), so most queries attend
/// sparsely.
const TOKENS: [u32; 11] = [1, 4, 2, 7, 5, 9, 3, 6, 8, 4, 1];

#[test]
fn every_glm_architecture_is_supported() {
    for name in [
        "chatglm",
        "glm4",
        "glm4moe",
        "glm-dsa",
        "glm5next",
        "glm5-next",
    ] {
        let arch = Architecture::from_name(name).unwrap_or_else(|| panic!("{name} unsupported"));
        assert!(
            arch.is_native(),
            "{name} should load through a native loader"
        );
        assert!(arch.shares_weights(), "{name}");
    }
}

fn glm_dsa_model(name: &str) -> (std::path::PathBuf, joshua::model::QuantizedModel) {
    let dir = common::model_dir(name);
    let model = dir.join("model.gguf");
    common::write_tiny_glm_dsa_gguf(&model);
    let m = load(&model);
    (dir, m)
}

/// `glm-dsa` matches an independent float64 NumPy transcription of
/// llama.cpp's `src/models/glm-dsa.cpp` at every position
/// (`tests/data/glm_dsa_reference_logits.txt`): each query past the third
/// sees only its indexer's top-3 keys, layers 1 and 3 reuse the selection
/// of layers 0 and 2, and the trailing NextN block is skipped.
#[test]
fn glm_dsa_matches_llama_cpp_reference_at_every_position() {
    let want: Vec<Vec<f32>> = include_str!("data/glm_dsa_reference_logits.txt")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| {
            l.split_whitespace()
                .skip(1)
                .map(|x| x.parse().unwrap())
                .collect()
        })
        .collect();
    assert_eq!(want.len(), TOKENS.len());

    let (dir, mut m) = glm_dsa_model("glm-dsa-ref");
    let input = Tensor::new(&TOKENS[..], &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let got: Vec<Vec<f32>> = m
        .forward_all_logits(&input, 0)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_vec2()
        .unwrap();
    for (pos, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_close(
            g,
            w,
            1e-6,
            &format!("glm-dsa position {pos} vs llama.cpp reference"),
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Decoding token by token — each step publishing and reusing a one-query
/// selection — reproduces the prefill at every position, and a session
/// cut back to a prefix re-decodes the rest identically.
#[test]
fn glm_dsa_incremental_decode_and_truncation_match_prefill() {
    let (dir, mut prefill) = glm_dsa_model("glm-dsa-decode");
    let input = Tensor::new(&TOKENS[..], &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let want: Vec<Vec<f32>> = prefill
        .forward_all_logits(&input, 0)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_vec2()
        .unwrap();

    let mut step = prefill.new_session().expect("native loaders share weights");
    for (pos, &t) in TOKENS.iter().enumerate() {
        let got = logits(&mut step, &[t], pos);
        assert_close(&got, &want[pos], 1e-5, &format!("glm-dsa decode at {pos}"));
    }

    assert!(step.supports_kv_truncate());
    step.truncate_kv_cache(5).unwrap();
    let got = logits(&mut step, &TOKENS[5..], 5);
    assert_close(
        &got,
        &want[TOKENS.len() - 1],
        1e-5,
        "glm-dsa after truncation",
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// The layer-streaming prefill runs each layer over every chunk before the
/// next layer starts, so a shared layer reads selections its full layer
/// published chunks earlier; it must still reproduce a single forward.
#[test]
fn glm_dsa_streamed_prefill_matches_forward() {
    use joshua::stream_prefill::Chunk;
    let (dir, mut single) = glm_dsa_model("glm-dsa-stream");
    let want = logits(&mut single, &TOKENS, 0);
    let mut streamed = single.new_session().unwrap();
    let chunks = [
        Chunk {
            tokens: &TOKENS[..4],
            pos: 0,
        },
        Chunk {
            tokens: &TOKENS[4..7],
            pos: 4,
        },
        Chunk {
            tokens: &TOKENS[7..],
            pos: 7,
        },
    ];
    let got: Vec<f32> = streamed
        .prefill_streamed(&chunks, &Device::Cpu)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_close(&want, &got, 1e-5, "glm-dsa streamed prefill");
    let a = logits(&mut single, &[2], TOKENS.len());
    let b = logits(&mut streamed, &[2], TOKENS.len());
    assert_close(&a, &b, 1e-5, "glm-dsa decode after streamed prefill");
    std::fs::remove_dir_all(&dir).ok();
}

/// Both architecture names llama.cpp's open GLM-5.3-Flash pull requests
/// write: `glm5-next` (#27773, with `indexer.types`) and `glm5next`
/// (#27754).
const GLM5NEXT_ARCHES: [&str; 2] = ["glm5-next", "glm5next"];

fn glm5next_model(test: &str, arch: &str) -> (std::path::PathBuf, joshua::model::QuantizedModel) {
    let dir = common::model_dir(&format!("glm5next-{test}-{arch}"));
    let model = dir.join("model.gguf");
    common::write_tiny_glm5next_gguf(&model, arch);
    let m = load(&model);
    (dir, m)
}

fn all_logits(m: &mut joshua::model::QuantizedModel, tokens: &[u32]) -> Vec<Vec<f32>> {
    let input = Tensor::new(tokens, &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    m.forward_all_logits(&input, 0)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_vec2()
        .unwrap()
}

/// GLM-5.3-Flash matches an independent float64 NumPy transcription of HF
/// transformers' `modeling_glm5_next.py` (with llama.cpp's GGUF
/// conventions) at every position (`tests/data/glm5next_reference_logits.txt`):
/// hyper-connection streams, Kimi Delta Attention, NoPE MLA over the k-pool
/// indexer's pools and tails (shared across a KDA layer under `glm5-next`'s
/// `indexer.types`), clamped SwiGLUs and the stream-mean head.
#[test]
fn glm5next_matches_reference_at_every_position() {
    let golden = include_str!("data/glm5next_reference_logits.txt");
    for arch in GLM5NEXT_ARCHES {
        let want: Vec<Vec<f32>> = golden
            .lines()
            .filter(|l| l.starts_with(&format!("{arch} ")))
            .map(|l| {
                l.split_whitespace()
                    .skip(2)
                    .map(|x| x.parse().unwrap())
                    .collect()
            })
            .collect();
        assert_eq!(want.len(), TOKENS.len(), "{arch}");
        let (dir, mut m) = glm5next_model("ref", arch);
        for (pos, (g, w)) in all_logits(&mut m, &TOKENS).iter().zip(&want).enumerate() {
            assert_close(g, w, 1e-6, &format!("{arch} position {pos} vs reference"));
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// Token-by-token decode through the recurrent KDA state, the k-pool cache
/// and the shared selections reproduces the prefill; the recurrent layers
/// cannot rewind, so truncation is refused.
#[test]
fn glm5next_incremental_decode_matches_prefill() {
    for arch in GLM5NEXT_ARCHES {
        let (dir, mut prefill) = glm5next_model("decode", arch);
        let want = all_logits(&mut prefill, &TOKENS);
        let mut step = prefill.new_session().expect("native loaders share weights");
        for (pos, &t) in TOKENS.iter().enumerate() {
            let got = logits(&mut step, &[t], pos);
            assert_close(&got, &want[pos], 1e-5, &format!("{arch} decode at {pos}"));
        }
        assert!(
            !step.supports_kv_truncate(),
            "{arch}: KDA state cannot be truncated"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// The layer-streaming prefill reproduces a single forward (the KDA state
/// and the pooled index keys carry across chunks).
#[test]
fn glm5next_streamed_prefill_matches_forward() {
    use joshua::stream_prefill::Chunk;
    for arch in GLM5NEXT_ARCHES {
        let (dir, mut single) = glm5next_model("stream", arch);
        let want = logits(&mut single, &TOKENS, 0);
        let mut streamed = single.new_session().unwrap();
        let chunks = [
            Chunk {
                tokens: &TOKENS[..3],
                pos: 0,
            },
            Chunk {
                tokens: &TOKENS[3..8],
                pos: 3,
            },
            Chunk {
                tokens: &TOKENS[8..],
                pos: 8,
            },
        ];
        let got: Vec<f32> = streamed
            .prefill_streamed(&chunks, &Device::Cpu)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_close(&want, &got, 1e-5, &format!("{arch} streamed prefill"));
        let a = logits(&mut single, &[2], TOKENS.len());
        let b = logits(&mut streamed, &[2], TOKENS.len());
        assert_close(
            &a,
            &b,
            1e-5,
            &format!("{arch} decode after streamed prefill"),
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
