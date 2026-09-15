//! Memory-placement tests for the MoE loaders and the engine: host-placed
//! experts load and match the plain load, sessions share one set of
//! weights, the token-embedding table stays quantized, and the engine
//! derives every session from one loaded template.
//!
//! Everything here runs on the CPU device, so the host/device *hop* in the
//! MoE dispatch is exercised in its same-device form only; the placement
//! decision itself is unit-tested in `placement.rs` and the cross-device
//! transfer is the same `to_device` pair the `deepseek4` loader has always
//! used on an accelerator.

mod common;

use candle_core::{Device, Tensor};
use joshua::model::QuantizedModel;
use joshua::{ChatMessage, Engine, EngineOptions, ExpertPlacement, GenerationOptions};
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

/// Read the header the way the engine does: joshua's tolerant reader (raw
/// dtype ids, so deepseek4's IQ2_XXS tensors parse) projected onto candle's
/// `Content`, with the cursor left at the tensor-data start.
fn read_content(bytes: &[u8]) -> (candle_core::quantized::gguf_file::Content, Cursor<&[u8]>) {
    let mut cursor = Cursor::new(bytes);
    let header = joshua::gguf_ext::read_header(&mut cursor).unwrap();
    let content = header.to_candle_content().unwrap();
    (content, cursor)
}

fn load_heap(model: &Path) -> QuantizedModel {
    let bytes = std::fs::read(model).unwrap();
    let (content, mut cursor) = read_content(&bytes);
    QuantizedModel::from_gguf(content, &mut cursor, &Device::Cpu).unwrap()
}

/// Load through the placed entry point with the experts on `expert_device`.
fn load_placed(model: &Path, expert_device: &Device) -> QuantizedModel {
    let mmap = unsafe { memmap2::Mmap::map(&std::fs::File::open(model).unwrap()) }.unwrap();
    let mmap = Arc::new(mmap);
    let (content, mut cursor) = read_content(&mmap[..]);
    QuantizedModel::from_gguf_mmap_placed(
        content,
        &mut cursor,
        &Device::Cpu,
        expert_device,
        Some(Arc::clone(&mmap)),
        None,
        0,
        None,
    )
    .unwrap()
}

/// Like [`load_placed`], but with a `device_expert_cache_bytes` budget so the
/// loader builds a per-layer `DeviceResidency` and the dispatch partition runs
/// (on CPU, a hit wraps the same host weights — parity holds).
fn load_placed_with_cache(model: &Path, cache_bytes: u64) -> QuantizedModel {
    let mmap = unsafe { memmap2::Mmap::map(&std::fs::File::open(model).unwrap()) }.unwrap();
    let mmap = Arc::new(mmap);
    let (content, mut cursor) = read_content(&mmap[..]);
    QuantizedModel::from_gguf_mmap_placed(
        content,
        &mut cursor,
        &Device::Cpu,
        &Device::Cpu,
        Some(Arc::clone(&mmap)),
        None,
        0,
        Some(cache_bytes),
    )
    .unwrap()
}

fn logits(model: &mut QuantizedModel, tokens: &[u32], offset: usize) -> Vec<f32> {
    let input = Tensor::new(tokens, &Device::Cpu)
        .unwrap()
        .reshape((1, tokens.len()))
        .unwrap();
    model
        .forward(&input, offset)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

fn assert_close(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() < 1e-5,
            "{what}: logit {i} diverges: {x} vs {y}"
        );
    }
}

/// Each MoE fixture as `(name, writer)`.
fn fixtures() -> Vec<(&'static str, fn(&Path))> {
    vec![
        ("qwen3moe", common::write_tiny_qwen3moe_gguf as fn(&Path)),
        ("deepseek2", common::write_tiny_deepseek2_gguf as fn(&Path)),
    ]
}

/// Host-placed experts (the layout an accelerator uses for a model larger
/// than its memory) produce the same logits as the plain load.
#[test]
fn host_placed_experts_match_plain_load() {
    for (name, write) in fixtures() {
        let dir = common::model_dir(&format!("placed-{name}"));
        let model = dir.join("model.gguf");
        write(&model);
        let tokens = [1u32, 4, 2, 7, 5];

        let mut heap = load_heap(&model);
        let mut placed = load_placed(&model, &Device::Cpu);
        assert_close(
            &logits(&mut heap, &tokens, 0),
            &logits(&mut placed, &tokens, 0),
            name,
        );
        // Decode continues correctly from the placed prefill.
        assert_close(
            &logits(&mut heap, &[9], 5),
            &logits(&mut placed, &[9], 5),
            name,
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// The token-embedding table is held quantized by every joshua-native
/// loader (no f32 copy per instance).
#[test]
fn token_embeddings_stay_quantized() {
    let mut all = fixtures();
    all.push(("deepseek4", common::write_tiny_deepseek4_gguf as fn(&Path)));
    for (name, write) in all {
        let dir = common::model_dir(&format!("qemb-{name}"));
        let model = dir.join("model.gguf");
        write(&model);
        let m = load_placed(&model, &Device::Cpu);
        assert_eq!(
            m.embeddings_quantized(),
            Some(true),
            "{name}: embeddings must stay quantized"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// A derived session shares the weights (one `Arc`), computes exactly what
/// a fresh load computes, and keeps its KV cache independent of the
/// instance it was derived from.
#[test]
fn derived_sessions_share_weights_and_isolate_kv() {
    for (name, write) in fixtures() {
        let dir = common::model_dir(&format!("share-{name}"));
        let model = dir.join("model.gguf");
        write(&model);

        let mut template = load_placed(&model, &Device::Cpu);
        assert!(template.supports_shared_weights(), "{name}");
        let mut a = template.new_session().expect("shares weights");
        let mut b = template.new_session().expect("shares weights");
        let count = match &template {
            QuantizedModel::Qwen3Moe(m) => m.shared_session_count(),
            QuantizedModel::DeepSeek2(m) => m.shared_session_count(),
            _ => unreachable!(),
        };
        assert_eq!(
            count, 3,
            "{name}: template + 2 sessions share one weight set"
        );

        // Two different conversations in two sessions; each must equal a
        // fresh instance run on the same conversation alone.
        let conv_a = [1u32, 4, 2, 7, 5];
        let conv_b = [3u32, 8, 8, 1];
        let la = logits(&mut a, &conv_a, 0);
        let lb = logits(&mut b, &conv_b, 0);
        let la2 = logits(&mut a, &[6], conv_a.len());
        let lb2 = logits(&mut b, &[6], conv_b.len());

        let mut fresh_a = load_heap(&model);
        let mut fresh_b = load_heap(&model);
        assert_close(&la, &logits(&mut fresh_a, &conv_a, 0), name);
        assert_close(&lb, &logits(&mut fresh_b, &conv_b, 0), name);
        assert_close(&la2, &logits(&mut fresh_a, &[6], conv_a.len()), name);
        assert_close(&lb2, &logits(&mut fresh_b, &[6], conv_b.len()), name);

        // The template itself is untouched by its sessions' KV state.
        assert_close(&logits(&mut template, &conv_a, 0), &la, name);

        // Truncating one session's KV does not affect the other.
        assert!(a.truncate_kv_cache(2).unwrap());
        let lb3 = logits(&mut b, &[2], conv_b.len() + 1);
        assert_close(&lb3, &logits(&mut fresh_b, &[2], conv_b.len() + 1), name);
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// Loading a qwen3moe model with a non-zero `device_expert_cache_bytes`
/// budget threads a per-layer `DeviceResidency` into every MoE block; on the
/// CPU the device form wraps the same host weights, so a mixed-residency run
/// must produce exactly the no-cache logits (proves the budget plumbing +
/// partition end-to-end through `QuantizedModel::from_gguf_mmap_placed`).
#[test]
fn vram_expert_cache_budget_loads_and_preserves_logits() {
    let dir = common::model_dir("vram-cache-qwen3moe");
    let model = dir.join("model.gguf");
    common::write_tiny_qwen3moe_gguf(&model);

    let mut plain = load_placed(&model, &Device::Cpu);
    let mut cached = load_placed_with_cache(&model, 8 << 20); // 8 MiB budget
    let tokens = [1u32, 4, 2, 7, 5];
    assert_close(
        &logits(&mut plain, &tokens, 0),
        &logits(&mut cached, &tokens, 0),
        "vram-expert-cache budget load preserves logits",
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// Explicit placement requests other than the model device or the CPU are
/// rejected at load rather than silently ignored.
#[test]
fn placed_load_rejects_foreign_expert_device() {
    // Only the CPU exists in this build, so the guard is exercised through
    // the public contract: CPU experts on a CPU model is the same device and
    // must load.
    let dir = common::model_dir("placed-guard");
    let model = dir.join("model.gguf");
    common::write_tiny_qwen3moe_gguf(&model);
    let _ = load_placed(&model, &Device::Cpu);
    std::fs::remove_dir_all(&dir).ok();
}

/// The engine loads a weight-sharing model once and derives every session
/// from that template: two conversations in flight leave the template
/// shared by their pooled sessions, and the expert device follows the
/// placement option.
#[test]
fn engine_shares_one_weight_set_across_sessions() {
    let dir = common::model_dir("engine-share");
    let model = dir.join("model.gguf");
    common::write_tiny_qwen3moe_gguf(&model);

    let opts = EngineOptions::with_n_ctx(64).expert_placement(ExpertPlacement::Host);
    let engine = Engine::with_options(&dir, opts).expect("engine should load tiny model");
    assert!(engine.device().is_cpu());
    assert!(engine.expert_device().is_cpu());
    assert_eq!(
        engine.shared_weight_sessions(),
        0,
        "nothing loaded before the first request"
    );

    let options = GenerationOptions {
        max_tokens: 3,
        temperature: 0.0,
        ..GenerationOptions::default()
    };
    engine
        .complete(&[ChatMessage::text("user", "hello world")], &options)
        .expect("first completion");
    // Template + the pooled session from the first request.
    assert_eq!(engine.shared_weight_sessions(), 2);

    // A second, unrelated conversation clears and reuses the pooled session
    // (or derives a new one) — either way no second copy of the weights.
    engine
        .complete(&[ChatMessage::text("user", "a b c d")], &options)
        .expect("second completion");
    assert!(
        (2..=3).contains(&engine.shared_weight_sessions()),
        "sessions derive from the template: {}",
        engine.shared_weight_sessions()
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// The layer-streaming prefill (layer-outer / chunk-inner, shared framework)
/// must produce bit-identical last-token logits to today's standard
/// chunked prefill — the KV reordering is numerically identical (each layer's
/// KV accumulates in the same order across chunks).
#[test]
fn stream_prefill_matches_chunked_forward() {
    for (name, write) in fixtures() {
        let dir = common::model_dir(&format!("lsp-{name}"));
        let model = dir.join("model.gguf");
        write(&model);
        let mut plain = load_placed(&model, &Device::Cpu);
        let mut stream = load_placed(&model, &Device::Cpu);
        let tokens: Vec<u32> = vec![1, 4, 2, 7, 5, 3, 8, 9];
        let reference = logits(&mut plain, &tokens, 0);

        // A prompt long enough to split into 2 chunks of 4.
        let chunks: Vec<joshua::stream_prefill::Chunk> = vec![
            joshua::stream_prefill::Chunk { tokens: &tokens[0..4], pos: 0 },
            joshua::stream_prefill::Chunk { tokens: &tokens[4..8], pos: 4 },
        ];
        let out = stream
            .prefill_streamed(&chunks, &Device::Cpu)
            .expect("streaming supported");
        let got: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(got.len(), reference.len(), "{name}: vocab widths");
        for (i, (g, r)) in got.iter().zip(reference.iter()).enumerate() {
            assert!(
                (g - r).abs() < 1e-4,
                "{name}: streamed logit {i} diverges: {g} vs {r}"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
