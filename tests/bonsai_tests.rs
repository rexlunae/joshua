//! Bonsai models: `qwen3` GGUFs whose weights are in the 1-bit Q1_0 and
//! 2-bit Q2_0 block formats, which candle cannot name.  They load through
//! the native Qwen loader with the blocks read from the raw GGUF header
//! (borrowed from the mapping, or decoded to f32 on a streamed load).

mod common;

use common::{assert_close, load_model, logits, BonsaiWeights};

const TOKENS: [u32; 6] = [1, 4, 2, 7, 5, 9];

fn bonsai(name: &str, weights: BonsaiWeights) -> (std::path::PathBuf, std::path::PathBuf) {
    let dir = common::model_dir(name);
    let model = dir.join("model.gguf");
    common::write_tiny_bonsai_gguf(&model, weights);
    (dir, model)
}

/// candle's own reader rejects the file outright, which is why the raw
/// header path exists.
#[test]
fn candle_cannot_parse_bonsai_ggufs() {
    let (dir, model) = bonsai("bonsai-candle", BonsaiWeights::LowBit);
    let bytes = std::fs::read(&model).unwrap();
    let parsed =
        candle_core::quantized::gguf_file::Content::read(&mut std::io::Cursor::new(&bytes[..]));
    assert!(parsed.is_err());
    std::fs::remove_dir_all(&dir).ok();
}

/// The mapped path (blocks borrowed and decoded inside the matmul), the
/// streamed path (decoded to f32 at load) and an F32 twin holding exactly
/// the decoded values all agree.
#[test]
fn bonsai_low_bit_matches_its_dequantized_twin() {
    let (dir, model) = bonsai("bonsai-lowbit", BonsaiWeights::LowBit);
    let twin = dir.join("twin.gguf");
    common::write_tiny_bonsai_gguf(&twin, BonsaiWeights::Dequantized);

    let want = logits(&mut load_model(&twin, false), &TOKENS, 0);
    let mapped = logits(&mut load_model(&model, true), &TOKENS, 0);
    let streamed = logits(&mut load_model(&model, false), &TOKENS, 0);
    assert_close(&mapped, &want, 1e-5, "mapped Q1_0/Q2_0 vs f32 twin");
    assert_close(&streamed, &want, 1e-5, "streamed Q1_0/Q2_0 vs f32 twin");
    std::fs::remove_dir_all(&dir).ok();
}

/// Logits match an independent float64 NumPy transcription of llama.cpp's
/// `src/models/qwen3.cpp` with the blocks decoded per `ggml-quants.c`
/// (`tests/data/bonsai_reference_logits.txt`).
#[test]
fn bonsai_matches_llama_cpp_reference() {
    let golden = include_str!("data/bonsai_reference_logits.txt");
    let line = golden.lines().find(|l| l.starts_with("bonsai ")).unwrap();
    let want: Vec<f32> = line
        .split_whitespace()
        .skip(1)
        .map(|x| x.parse().unwrap())
        .collect();

    let (dir, model) = bonsai("bonsai-ref", BonsaiWeights::LowBit);
    for mmap in [true, false] {
        let got = logits(&mut load_model(&model, mmap), &TOKENS, 0);
        assert_close(
            &got,
            &want,
            1e-5,
            &format!("bonsai (mmap={mmap}) vs llama.cpp reference"),
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Incremental decode through the KV cache agrees with a full prefill.
#[test]
fn bonsai_incremental_decode_matches_prefill() {
    let (dir, model) = bonsai("bonsai-decode", BonsaiWeights::LowBit);
    let mut m = load_model(&model, true);
    let prefill = logits(&mut m, &TOKENS, 0);
    let mut step = m.new_session().expect("native loaders share weights");
    let mut last = Vec::new();
    for (pos, &tok) in TOKENS.iter().enumerate() {
        last = logits(&mut step, &[tok], pos);
    }
    assert_close(&prefill, &last, 1e-4, "bonsai prefill vs incremental");
    std::fs::remove_dir_all(&dir).ok();
}

/// The engine accepts the file (its tolerant header keeps the Q1_0 / Q2_0
/// tensors) and generates.
#[test]
fn bonsai_runs_through_the_engine() {
    use joshua::types::GenerationOptions;

    let (dir, _model) = bonsai("bonsai-engine", BonsaiWeights::LowBit);
    let engine = joshua::Engine::with_n_ctx(&dir, 64).expect("engine must load Bonsai GGUFs");
    let opts = GenerationOptions {
        max_tokens: 4,
        temperature: 0.0,
        ..Default::default()
    };
    let (_, usage, _, _) = engine
        .complete_raw("hello world", &opts)
        .expect("generation");
    assert!(usage.completion_tokens > 0, "{usage:?}");
    std::fs::remove_dir_all(&dir).ok();
}
