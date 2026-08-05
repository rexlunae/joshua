//! Native (pure-Rust) validation for the `deepseek4` quantized loader with the
//! dtypes candle cannot represent: IQ2_XXS routed experts and I32 routed-id
//! tables.  These run on the default `cargo test` — no llama.cpp, no network.

mod common;

use candle_core::{Device, Tensor};
use joshua::model::{Architecture, QuantizedModel};

fn load(model: &std::path::Path, mmap: bool) -> QuantizedModel {
    let bytes = std::fs::read(model).unwrap();
    let mut cursor = std::io::Cursor::new(&bytes[..]);
    // Load exactly the way the engine does: the tolerant header (raw dtype
    // ids) projected onto candle's Content.  `Content::read` itself would
    // reject the file on the first IQ2_XXS tensor.  `from_gguf_mmap` re-reads
    // the raw header from the cursor internally.
    let header = joshua::gguf_ext::read_header(&mut cursor).unwrap();
    let content = header.to_candle_content().unwrap();
    let _ = header;
    let mmap = if mmap {
        // Safety: the file is read-only for the lifetime of the mapping.
        unsafe { memmap2::Mmap::map(&std::fs::File::open(model).unwrap()) }
            .ok()
            .map(std::sync::Arc::new)
    } else {
        None
    };
    let mut cursor = std::io::Cursor::new(&bytes[..]);
    QuantizedModel::from_gguf_mmap(content, &mut cursor, &Device::Cpu, mmap).unwrap()
}

fn logits(model: &mut QuantizedModel, tokens: &[u32], offset: usize) -> Vec<f32> {
    let input = Tensor::new(tokens, &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    model
        .forward(&input, offset)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_vec1()
        .unwrap()
}

#[test]
fn deepseek4_is_a_supported_architecture() {
    assert_eq!(
        Architecture::from_name("deepseek4"),
        Some(Architecture::DeepSeek4)
    );
}

/// The full loader path: IQ2_XXS experts (mmap-borrowed), I32 tid2eid table,
/// hash + regular MoE layers — produces finite, non-degenerate logits.
#[test]
fn deepseek4_loads_iq2xxs_and_i32_and_produces_finite_logits() {
    let dir = common::model_dir("deepseek4-load");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&model);

    let mut m = load(&model, true);
    let out = logits(&mut m, &[1, 4, 2, 7, 5], 0);
    assert_eq!(out.len(), 16, "logits must cover the 16-token vocab");
    assert!(
        out.iter().all(|v| v.is_finite()),
        "all logits must be finite: {out:?}"
    );
    let first = out[0];
    assert!(
        out.iter().any(|v| (v - first).abs() > 1e-6),
        "logits are degenerate"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The mmap path (blocks borrowed, decoded inside the matmul) and the
/// streamed path (whole tensor decoded to f32 at load) must agree.
#[test]
fn deepseek4_mmap_matches_streamed_path() {
    let dir = common::model_dir("deepseek4-mmap");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&model);

    let mut mmapped = load(&model, true);
    let mut streamed = load(&model, false);
    let tokens = [1, 4, 2, 7, 5, 3, 8];
    let a = logits(&mut mmapped, &tokens, 0);
    let b = logits(&mut streamed, &tokens, 0);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let tol = 1e-3 * y.abs().max(1.0);
        assert!(
            (x - y).abs() <= tol,
            "logit {i}: mmap={x} streamed={y} (tol {tol})"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// Incremental decode must give the same next-token logits as prefill.
#[test]
fn deepseek4_prefill_matches_incremental_decode() {
    let dir = common::model_dir("deepseek4-incr");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&model);

    let tokens = [1u32, 4, 2, 7, 5];
    let mut m = load(&model, true);
    let prefill = logits(&mut m, &tokens, 0);

    let mut m = load(&model, true);
    let mut last = vec![0f32; 16];
    for (i, &t) in tokens.iter().enumerate() {
        last = logits(&mut m, &[t], i);
    }
    for i in 0..16 {
        let tol = 1e-3 * prefill[i].abs().max(1.0);
        assert!(
            (last[i] - prefill[i]).abs() <= tol,
            "logit {i}: prefill={} incremental={}",
            prefill[i],
            last[i]
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// The engine must accept a GGUF whose dtypes candle cannot name — the
/// tolerant header is what makes the whole file reachable at all.
#[test]
fn deepseek4_engine_accepts_unknown_dtypes() {
    let dir = common::model_dir("deepseek4-engine");
    let model = dir.join("model.gguf");
    common::write_tiny_deepseek4_gguf(&model);

    let engine = joshua::Engine::new(&model).expect("engine must load IQ2_XXS GGUFs");
    assert_eq!(engine.model_name(), "DeepSeek-V4");

    std::fs::remove_dir_all(&dir).ok();
}
