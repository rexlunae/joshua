//! Native (pure-Rust) validation for the Kimi hybrids on the `deepseek2`
//! loader: `kimi-linear` (Kimi-Linear) and `kimi-k3` (Kimi K3).  Kimi-K2,
//! K2.5 and Moonlight are plain `deepseek2` files and are covered by
//! `deepseek2_tests.rs`.  These run on the default `cargo test` — no
//! llama.cpp, no network.

mod common;

use candle_core::{Device, Tensor};
use common::{assert_close, logits};
use joshua::model::Architecture;

const TOKENS: [u32; 11] = [1, 4, 2, 7, 5, 9, 3, 6, 8, 4, 1];
const ARCHES: [&str; 2] = ["kimi-linear", "kimi-k3"];

#[test]
fn every_kimi_architecture_is_supported() {
    for name in ["deepseek2", "kimi-linear", "kimi-k3"] {
        let arch = Architecture::from_name(name).unwrap_or_else(|| panic!("{name} unsupported"));
        assert!(
            arch.is_native(),
            "{name} should load through a native loader"
        );
        assert!(arch.shares_weights(), "{name}");
    }
}

fn kimi_model(
    test: &str,
    arch: &str,
    mxfp4: bool,
) -> (std::path::PathBuf, joshua::model::QuantizedModel) {
    let dir = common::model_dir(&format!("kimi-{test}-{arch}"));
    let model = dir.join("model.gguf");
    common::write_tiny_kimi_gguf(&model, arch, mxfp4);
    let m = common::load_model(&model, false);
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

fn reference(arch: &str) -> Vec<Vec<f32>> {
    let want: Vec<Vec<f32>> = include_str!("data/kimi_reference_logits.txt")
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
    want
}

/// Both hybrids match an independent float64 NumPy transcription of
/// llama.cpp's `kimi-linear.cpp` / `kimi-k3.cpp` at every position
/// (`tests/data/kimi_reference_logits.txt`): KDA with both decay gates and
/// output gates, fused and split QKV, NoPE MLA over an unrotated rope slice
/// (combined and split KV up-projections, K3's output gate), sigmoid
/// routing, and K3's `situ`, latent MoE and attention residuals.
#[test]
fn kimi_matches_llama_cpp_reference_at_every_position() {
    for arch in ARCHES {
        let want = reference(arch);
        let (dir, mut m) = kimi_model("ref", arch, false);
        for (pos, (g, w)) in all_logits(&mut m, &TOKENS).iter().zip(&want).enumerate() {
            assert_close(
                g,
                w,
                1e-5,
                &format!("{arch} position {pos} vs llama.cpp reference"),
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// K3's routed experts in MXFP4 (as its converter emits them) are sliced
/// per expert from the mapping and decoded at matmul time — or, without a
/// mapping, decoded once at load; either way the file whose experts are
/// their exact f32 decode is the same model.
#[test]
fn kimi_k3_mxfp4_experts_match_their_decode() {
    let want = reference("kimi-k3");
    let (dir, _) = kimi_model("mxfp4", "kimi-k3", true);
    for mmap in [true, false] {
        let mut m = common::load_model(&dir.join("model.gguf"), mmap);
        for (pos, (g, w)) in all_logits(&mut m, &TOKENS).iter().zip(&want).enumerate() {
            assert_close(
                g,
                w,
                1e-5,
                &format!("kimi-k3 MXFP4 (mmap {mmap}) position {pos} vs reference"),
            );
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Token-by-token decode through the recurrent KDA state, the MLA cache and
/// K3's banked residual checkpoints reproduces the prefill; the recurrent
/// layers cannot rewind, so truncation is refused.
#[test]
fn kimi_incremental_decode_matches_prefill() {
    for arch in ARCHES {
        let (dir, mut prefill) = kimi_model("decode", arch, false);
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
/// carries across chunks, and K3's residual bank grows layer by layer).
#[test]
fn kimi_streamed_prefill_matches_forward() {
    use joshua::stream_prefill::Chunk;
    for arch in ARCHES {
        let (dir, mut single) = kimi_model("stream", arch, false);
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
