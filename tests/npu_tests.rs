//! Tests for the three-layer NPU backend architecture.
//!
//! Uses `crates/joshua-mock-npu` — a pure-Rust cdylib implementing the
//! `joshua_npu_*` plugin ABI with deterministic logits — to exercise the
//! full stack with no real NPU:
//!
//! - layer 2: `InProcessBackend` dlopens the plugin into this process;
//! - layer 3: `ShimBackend` hosts the same plugin in a killable subprocess
//!   (crash containment, hang timeouts, restart);
//! - layer 1: the engine falls back to the candle path when the backend
//!   fails, and the circuit breaker disables it after repeated failures.

mod common;

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use joshua::npu::{InProcessBackend, NpuBackend, ShimBackend};
use joshua::types::GenerationOptions;
use joshua::Engine;

/// Magic tokens understood by the mock plugin (see its crate docs).
const CRASH_TOKEN: u32 = 4242;
const HANG_TOKEN: u32 = 5150;

/// Build (once) and locate the mock vendor plugin cdylib.
fn mock_plugin() -> PathBuf {
    static PLUGIN: OnceLock<PathBuf> = OnceLock::new();
    PLUGIN
        .get_or_init(|| {
            let status = Command::new(env!("CARGO"))
                .args(["build", "-p", "joshua-mock-npu"])
                .status()
                .expect("failed to run cargo build for the mock plugin");
            assert!(status.success(), "mock plugin build failed");

            let target = std::env::var("CARGO_TARGET_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target"));
            let name = if cfg!(target_os = "macos") {
                "libjoshua_mock_npu.dylib"
            } else if cfg!(windows) {
                "joshua_mock_npu.dll"
            } else {
                "libjoshua_mock_npu.so"
            };
            let path = target.join("debug").join(name);
            assert!(path.exists(), "mock plugin not found at {path:?}");
            path
        })
        .clone()
}

/// Shim executable built alongside the tests.
fn shim_path() -> &'static str {
    env!("CARGO_BIN_EXE_joshua-npu-shim")
}

fn shim_backend() -> ShimBackend {
    ShimBackend::new(shim_path(), mock_plugin())
        .init_timeout(Duration::from_secs(30))
        .forward_timeout(Duration::from_secs(30))
}

/// Expected mock argmax after feeding `history`: (sum % 12) + 4.
fn mock_argmax(history: &[u32]) -> u32 {
    ((history.iter().map(|&t| t as u64).sum::<u64>() % 12) + 4) as u32
}

/// A model directory the mock init can point at (contents unused by the
/// mock, but the engine needs a loadable GGUF + tokenizer).
fn tiny_model_dir(name: &str) -> PathBuf {
    let dir = common::model_dir(name);
    common::write_tiny_gguf(&dir.join("model.gguf"), "llama");
    dir
}

// ─── Layer 2: in-process dlopen ─────────────────────────────────────────────

#[test]
fn inprocess_backend_is_deterministic_and_resettable() {
    let backend = InProcessBackend::load(&mock_plugin()).expect("plugin should load");
    let dir = tiny_model_dir("npu-inproc");
    let mut session = backend
        .create_session(&dir.join("model.gguf"), 64)
        .expect("session should initialise");

    assert_eq!(session.vocab_size(), 16);

    let logits = session.forward(&[1, 4], 0).expect("forward");
    assert_eq!(logits.len(), 16);
    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32;
    assert_eq!(argmax, mock_argmax(&[1, 4]));

    // Positions must be contiguous: skipping ahead is rejected by the mock.
    assert!(session.forward(&[5], 7).is_err());

    // Reset clears the history: the same first batch gives the same logits.
    assert!(session.reset());
    let again = session.forward(&[1, 4], 0).expect("forward after reset");
    assert_eq!(logits, again);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn inprocess_backend_rejects_non_plugin_library() {
    let err = match InProcessBackend::load(std::path::Path::new("/nonexistent/libnpu.so")) {
        Ok(_) => panic!("loading a nonexistent plugin must fail"),
        Err(e) => e,
    };
    assert!(err.contains("failed to load"), "got: {err}");
}

// ─── Layer 3: shim isolation ────────────────────────────────────────────────

#[test]
fn shim_backend_matches_inprocess_results() {
    let dir = tiny_model_dir("npu-shim-parity");
    let model = dir.join("model.gguf");

    let inproc = InProcessBackend::load(&mock_plugin()).unwrap();
    let mut a = inproc.create_session(&model, 64).unwrap();
    let mut b = shim_backend().create_session(&model, 64).expect("shim session");

    assert_eq!(a.vocab_size(), b.vocab_size());
    // Same plugin, same ABI, two hosting modes: identical logits.
    for (tokens, pos) in [(&[1u32, 4, 2][..], 0usize), (&[9][..], 3), (&[6, 7][..], 4)] {
        let la = a.forward(tokens, pos).unwrap();
        let lb = b.forward(tokens, pos).unwrap();
        assert_eq!(la, lb, "tokens {tokens:?} at pos {pos}");
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn shim_survives_plugin_crash_and_new_sessions_work() {
    let dir = tiny_model_dir("npu-shim-crash");
    let model = dir.join("model.gguf");
    let backend = shim_backend();

    let mut session = backend.create_session(&model, 64).expect("shim session");
    // The mock aborts its process on this token — the *shim* dies, not us.
    let err = session.forward(&[CRASH_TOKEN], 0).unwrap_err();
    assert!(
        err.contains("exited") || err.contains("dead"),
        "expected a process-death error, got: {err}"
    );
    // The session is dead for good...
    assert!(session.forward(&[1], 0).is_err());
    assert!(!session.reset());

    // ...but a fresh session (fresh shim process) works immediately.
    let mut fresh = backend.create_session(&model, 64).expect("fresh shim session");
    assert!(fresh.forward(&[1, 4], 0).is_ok());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn shim_kills_hung_plugin_on_timeout() {
    let dir = tiny_model_dir("npu-shim-hang");
    let model = dir.join("model.gguf");
    let backend = ShimBackend::new(shim_path(), mock_plugin())
        .init_timeout(Duration::from_secs(30))
        .forward_timeout(Duration::from_millis(500));

    let mut session = backend.create_session(&model, 64).expect("shim session");
    let start = std::time::Instant::now();
    // The mock sleeps 5 s on this token; the host must give up at ~500 ms.
    let err = session.forward(&[HANG_TOKEN], 0).unwrap_err();
    assert!(err.contains("did not respond"), "got: {err}");
    assert!(
        start.elapsed() < Duration::from_secs(3),
        "timeout was not enforced (took {:?})",
        start.elapsed()
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ─── Layer 1: engine integration, fallback, circuit breaker ─────────────────

#[test]
fn engine_generates_through_the_shim_backend() {
    let dir = tiny_model_dir("npu-engine");
    let engine = Engine::with_n_ctx(&dir, 64)
        .expect("engine should load")
        .with_npu_backend(Arc::new(shim_backend()));
    assert!(engine.npu_active());

    let options = GenerationOptions {
        max_tokens: 3,
        temperature: 0.0,
        repetition_penalty: 1.0,
        ..Default::default()
    };
    // "hello a" tokenises to [1, 4]; replicate the mock's greedy chain.
    let mut history = vec![1u32, 4];
    let mut letters: Vec<String> = Vec::new();
    for _ in 0..3 {
        let next = mock_argmax(&history);
        letters.push(((b'a' + (next as u8 - 4)) as char).to_string());
        history.push(next);
    }
    let expected = letters.concat();

    let (text, usage, _, _) = engine
        .complete_raw("hello a", &options)
        .expect("completion via NPU");
    assert_eq!(text, expected, "greedy decode must follow the mock's logits");
    assert_eq!(usage.prompt_tokens, 2);
    assert_eq!(usage.completion_tokens, 3);

    // A follow-up request extending the conversation reuses the pooled NPU
    // session's state (prefix reuse works across the process boundary).
    // Space the generated letters so they retokenise to the same IDs the
    // session already holds, and extend past them with a new user token.
    let extended = format!("hello a {} b", letters.join(" "));
    let before = engine.kv_reuse_count();
    engine
        .complete_raw(&extended, &options)
        .expect("second completion via NPU");
    assert_eq!(engine.kv_reuse_count(), before + 1);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn engine_falls_back_to_cpu_and_trips_the_circuit_breaker() {
    let dir = tiny_model_dir("npu-engine-fallback");
    // A backend whose plugin path is bogus: every session creation fails.
    let backend = ShimBackend::new(shim_path(), "/nonexistent/libnpu.so");
    let engine = Engine::with_n_ctx(&dir, 64)
        .expect("engine should load")
        .with_npu_backend(Arc::new(backend));
    assert!(engine.npu_active());

    let options = GenerationOptions {
        max_tokens: 2,
        temperature: 0.0,
        ..Default::default()
    };

    // Requests succeed anyway (candle fallback), and after three failed
    // session creations the breaker disables the backend.
    for _ in 0..3 {
        engine
            .complete_raw("hello a", &options)
            .expect("fallback completion");
    }
    assert!(!engine.npu_active(), "circuit breaker should have tripped");

    // Still healthy afterwards.
    engine
        .complete_raw("world b", &options)
        .expect("post-breaker completion");

    std::fs::remove_dir_all(&dir).ok();
}
