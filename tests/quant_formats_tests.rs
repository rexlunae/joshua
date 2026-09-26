//! Every raw block format — the i-quants (IQ1_S … IQ4_XS), the ternary
//! TQ1_0 / TQ2_0, MXFP4 / NVFP4 and Q1_0 / Q2_0 — loads end to end, both
//! through candle's stock `llama` loader (via the GGUF content's
//! external-tensor hook) and through Joshua's native `qwen3` loader, and
//! runs the same model as the file holding its exact f32 decode.  That the
//! decodes themselves match llama.cpp bit for bit is
//! `raw_block::tests::every_format_matches_llama_cpp_bit_for_bit`.

mod common;

use common::{assert_close, load_model, logits};

const TOKENS: [u32; 6] = [1, 4, 2, 7, 5, 9];

/// `(GGUF dtype id, name)` of every raw block format.
const FORMATS: [(u32, &str); 15] = [
    (16, "IQ2_XXS"),
    (17, "IQ2_XS"),
    (18, "IQ3_XXS"),
    (19, "IQ1_S"),
    (20, "IQ4_NL"),
    (21, "IQ3_S"),
    (22, "IQ2_S"),
    (23, "IQ4_XS"),
    (29, "IQ1_M"),
    (34, "TQ1_0"),
    (35, "TQ2_0"),
    (39, "MXFP4"),
    (40, "NVFP4"),
    (41, "Q1_0"),
    (42, "Q2_0"),
];

/// Prefill logits and one decode step.
fn run(model: &std::path::Path, mmap: bool) -> Vec<f32> {
    let mut m = load_model(model, mmap);
    let mut out = logits(&mut m, &TOKENS, 0);
    out.extend(logits(&mut m, &[3], TOKENS.len()));
    out
}

#[test]
fn every_raw_format_runs_like_its_f32_decode() {
    for arch in ["llama", "qwen3"] {
        for (dtype, name) in FORMATS {
            let dir = common::model_dir(&format!("quant-{arch}-{dtype}"));
            let (raw, twin) = (dir.join("raw.gguf"), dir.join("twin.gguf"));
            common::write_tiny_raw_format_gguf(&raw, arch, dtype, false);
            common::write_tiny_raw_format_gguf(&twin, arch, dtype, true);
            let want = run(&twin, false);
            let scale = want.iter().fold(0f32, |a, v| a.max(v.abs()));
            // Mapped (blocks borrowed in place) and read through the copying
            // reader (blocks held on the heap).
            for mmap in [true, false] {
                let got = run(&raw, mmap);
                assert_close(
                    &got,
                    &want,
                    1e-4 * scale.max(1.0),
                    &format!("{arch} {name} (mmap {mmap})"),
                );
            }
            std::fs::remove_dir_all(&dir).ok();
        }
    }
}

/// The engine accepts a stock-architecture file in these formats (it used
/// to refuse any dtype candle cannot name) and generates from it.
#[test]
fn engine_serves_a_llama_in_an_i_quant() {
    use joshua::{types::GenerationOptions, Engine};
    let dir = common::model_dir("quant-engine");
    common::write_tiny_raw_format_gguf(&dir.join("model.gguf"), "llama", 23, false);
    let engine = Engine::with_n_ctx(&dir, 64).expect("an IQ4_XS llama loads");
    let options = GenerationOptions {
        max_tokens: 3,
        temperature: 0.0,
        ..Default::default()
    };
    let (_, usage, _, _) = engine.complete_raw("hello a", &options).unwrap();
    assert!(usage.completion_tokens > 0);
    std::fs::remove_dir_all(&dir).ok();
}
