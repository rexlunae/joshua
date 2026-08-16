//! Metal regression test for the qwen3moe MoE routing path.
//!
//! The tiny CPU model uses head_dim=4, which Metal rejects (needs >= 32).
//! `common::write_tiny_qwen3moe_metal_gguf` builds a head_dim=128 model so the
//! full quantized forward (including expert routing) can run on Metal and be
//! checked for the out-of-bounds expert-id panic seen on real
//! Qwen3-30B-A3B Q4_K_M.

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use joshua::model::QuantizedModel;
use std::io::Cursor;

mod common;

#[test]
fn qwen3moe_metal_routing_stays_in_range() {
    // Skip cleanly when no Metal device is available.
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("no Metal device, skipping");
            return;
        }
    };

    let dir = common::model_dir("qwen3moe-metal-routing");
    let model = dir.join("model.gguf");
    common::write_tiny_qwen3moe_metal_gguf(&model);

    let bytes = std::fs::read(&model).unwrap();
    let mut cursor = Cursor::new(&bytes[..]);
    let content = gguf_file::Content::read(&mut cursor).unwrap();
    let mut m = QuantizedModel::from_gguf(content, &mut cursor, &dev).unwrap();

    // Prefill 5 tokens (n_tokens > 1 → bucketed dispatch path).
    let input = Tensor::new(vec![1u32, 4, 2, 7, 5], &dev)
        .unwrap()
        .reshape((1, 5))
        .unwrap();
    let out = m.forward(&input, 0).unwrap();
    let logits: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(logits.len(), 16);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "Metal prefill logits must be finite: {logits:?}"
    );

    // Decode 3 more tokens (n_tokens == 1 → dispatch_decode path).
    for tok in [2u32, 7, 5] {
        let t = Tensor::new(vec![tok], &dev).unwrap().reshape((1, 1)).unwrap();
        let out = m.forward(&t, 6).unwrap();
        let logits: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "Metal decode logits must be finite: {logits:?}"
        );
    }

    // ── Same model through the mmap loader (the path the CLI uses) ──
    let file = std::fs::File::open(&model).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let mmap_arc = std::sync::Arc::new(mmap);
    let mut cursor = Cursor::new(&mmap_arc[..]);
    let content = gguf_file::Content::read(&mut cursor).unwrap();
    let mut mm = QuantizedModel::from_gguf_mmap(
        content,
        &mut cursor,
        &dev,
        Some(std::sync::Arc::clone(&mmap_arc)),
    )
    .unwrap();
    let out = mm.forward(&input, 0).unwrap();
    let logits: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(logits.len(), 16);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "Metal mmap prefill logits must be finite: {logits:?}"
    );
    for tok in [2u32, 7, 5] {
        let t = Tensor::new(vec![tok], &dev).unwrap().reshape((1, 1)).unwrap();
        let out = mm.forward(&t, 6).unwrap();
        let logits: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "Metal mmap decode logits must be finite: {logits:?}"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}
