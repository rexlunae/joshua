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
    // The mapping is padded to a whole page so the zero-copy Metal path can
    // wrap it in a no-copy buffer; without the padding it falls back to
    // uploading the weights (still correct, just not exercising the path).
    let file = std::fs::File::open(&model).unwrap();
    let len = file.metadata().unwrap().len() as usize;
    let page = joshua::zero_copy_metal::page_size();
    let map_len = len.next_multiple_of(page);
    let mmap = unsafe {
        memmap2::MmapOptions::new()
            .len(map_len)
            .map(&file)
            .unwrap()
    };
    let mmap_arc = std::sync::Arc::new(mmap);
    let mut cursor = Cursor::new(&mmap_arc[..]);
    let content = gguf_file::Content::read(&mut cursor).unwrap();
    let mut mm = QuantizedModel::from_gguf_mmap(
        content,
        &mut cursor,
        &dev,
        Some(std::sync::Arc::clone(&mmap_arc)),
        None,
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

    // ── The zero-copy path must agree with the copy path exactly ──
    // Same weights, same kernels, same dispatch order — a load through the
    // no-copy mmap buffer has to produce the same logits as a load that
    // uploads the weights.  Any mismatch means an offset or shape is wrong.
    // (Fresh models: the KV cache appends, so models already used for decode
    // cannot be re-run at offset 0.)
    let mut cursor = Cursor::new(&bytes[..]);
    let content = gguf_file::Content::read(&mut cursor).unwrap();
    let mut cref = QuantizedModel::from_gguf(content, &mut cursor, &dev).unwrap();
    let mut cursor = Cursor::new(&mmap_arc[..]);
    let content = gguf_file::Content::read(&mut cursor).unwrap();
    let mut mref = QuantizedModel::from_gguf_mmap(
        content,
        &mut cursor,
        &dev,
        Some(std::sync::Arc::clone(&mmap_arc)),
        None,
    )
    .unwrap();
    let ref_prefill: Vec<f32> = cref
        .forward(&input, 0)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let mm_prefill: Vec<f32> = mref
        .forward(&input, 0)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let prefill_diff = ref_prefill
        .iter()
        .zip(&mm_prefill)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        prefill_diff < 1e-4,
        "zero-copy vs copy prefill logits diverge (max diff {prefill_diff})"
    );
    let t = Tensor::new(vec![2u32], &dev).unwrap().reshape((1, 1)).unwrap();
    let ref_decode: Vec<f32> = cref
        .forward(&t, 5)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let mm_decode: Vec<f32> = mref
        .forward(&t, 5)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let decode_diff = ref_decode
        .iter()
        .zip(&mm_decode)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        decode_diff < 1e-4,
        "zero-copy vs copy decode logits diverge (max diff {decode_diff})"
    );

    std::fs::remove_dir_all(&dir).ok();
}
