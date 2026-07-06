//! End-to-end tests for the llama.cpp compatibility plugin.
//!
//! These run only when `crates/joshua-llamacpp-npu` has been built (it
//! compiles llama.cpp, so it is excluded from the default workspace build):
//!
//! ```bash
//! cargo build -p joshua-llamacpp-npu
//! cargo test --test llamacpp_adapter_tests
//! ```
//!
//! Without the artifact the tests skip with a notice, keeping the default
//! `cargo test` pure Rust.

mod common;

use std::path::PathBuf;
use std::time::Duration;

use joshua::npu::{NpuBackend, ShimBackend};

/// Locate the built adapter cdylib, if any.
fn adapter_plugin() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("JOSHUA_LLAMACPP_PLUGIN") {
        let path = PathBuf::from(path);
        return path.exists().then_some(path);
    }
    let target = std::env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target"));
    let name = if cfg!(target_os = "macos") {
        "libjoshua_llamacpp_npu.dylib"
    } else if cfg!(windows) {
        "joshua_llamacpp_npu.dll"
    } else {
        "libjoshua_llamacpp_npu.so"
    };
    ["debug", "release"]
        .iter()
        .map(|profile| target.join(profile).join(name))
        .find(|p| p.exists())
}

macro_rules! require_adapter {
    ($name:ident) => {
        let Some($name) = adapter_plugin() else {
            eprintln!(
                "joshua-llamacpp-npu not built — skipping \
                 (cargo build -p joshua-llamacpp-npu to enable)"
            );
            return;
        };
    };
}

#[test]
fn llamacpp_plugin_serves_the_same_gguf_through_the_shim() {
    require_adapter!(plugin);

    let dir = common::model_dir("llamacpp-adapter");
    let model = dir.join("model.gguf");
    common::write_tiny_gguf(&model, "llama");

    let backend = ShimBackend::new(env!("CARGO_BIN_EXE_joshua-npu-shim"), plugin)
        .init_timeout(Duration::from_secs(120))
        .forward_timeout(Duration::from_secs(60));
    let mut session = backend
        .create_session(&model, 64)
        .expect("llama.cpp should load the same GGUF joshua uses");

    assert_eq!(session.vocab_size(), 16);

    // llama.cpp computes the same math candle does on this F32 model — the
    // logits must agree closely, and the argmax exactly.
    let tokens: Vec<u32> = vec![1, 4, 2, 7, 5];
    let llama_logits = session.forward(&tokens, 0).expect("llama.cpp forward");

    let bytes = std::fs::read(&model).unwrap();
    let mut cursor = std::io::Cursor::new(&bytes[..]);
    let content = candle_core::quantized::gguf_file::Content::read(&mut cursor).unwrap();
    let mut reference =
        joshua::model::QuantizedModel::from_gguf(content, &mut cursor, &candle_core::Device::Cpu)
            .unwrap();
    let input = candle_core::Tensor::new(tokens.as_slice(), &candle_core::Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let candle_logits: Vec<f32> = reference
        .forward(&input, 0)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_vec1()
        .unwrap();

    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    assert_eq!(
        argmax(&llama_logits),
        argmax(&candle_logits),
        "llama.cpp and candle disagree on the argmax:\n  llama.cpp: {llama_logits:?}\n  candle:    {candle_logits:?}"
    );
    for (i, (a, b)) in llama_logits.iter().zip(&candle_logits).enumerate() {
        assert!(
            (a - b).abs() < 5e-2,
            "logit {i} diverges: llama.cpp={a} candle={b}"
        );
    }

    // Incremental decode and reset work across the process boundary too.
    session.forward(&[9], 5).expect("incremental decode");
    assert!(session.reset());
    let again = session.forward(&tokens, 0).expect("forward after reset");
    assert_eq!(argmax(&again), argmax(&llama_logits));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn engine_generates_through_llamacpp_plugin() {
    require_adapter!(plugin);

    let dir = common::model_dir("llamacpp-engine");
    common::write_tiny_gguf(&dir.join("model.gguf"), "llama");

    let backend = ShimBackend::new(env!("CARGO_BIN_EXE_joshua-npu-shim"), plugin)
        .init_timeout(Duration::from_secs(120))
        .forward_timeout(Duration::from_secs(60));
    let engine = joshua::Engine::with_n_ctx(&dir, 64)
        .expect("engine should load")
        .with_npu_backend(std::sync::Arc::new(backend));

    let options = joshua::types::GenerationOptions {
        max_tokens: 4,
        temperature: 0.0,
        repetition_penalty: 1.0,
        ..Default::default()
    };

    // Greedy output through llama.cpp must match the candle path on the
    // same weights (both compute the same model).
    let (npu_text, npu_usage, _, _) = engine
        .complete_raw("hello a b", &options)
        .expect("completion via llama.cpp");
    assert!(engine.npu_active(), "backend should still be healthy");

    let cpu_engine = joshua::Engine::with_n_ctx(&dir, 64).expect("cpu engine");
    let (cpu_text, cpu_usage, _, _) = cpu_engine
        .complete_raw("hello a b", &options)
        .expect("completion via candle");

    assert_eq!(npu_text, cpu_text, "llama.cpp and candle outputs differ");
    assert_eq!(npu_usage.prompt_tokens, cpu_usage.prompt_tokens);

    std::fs::remove_dir_all(&dir).ok();
}
