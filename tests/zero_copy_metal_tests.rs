//! Zero-copy Metal weight tests.
//!
//! A Q4_K weight bound at its file offset inside a no-copy Metal buffer must
//! produce results identical to candle's copy path, on both kernel paths the
//! model uses: `matmul_mv` (decode, single-row) and `matmul_mm` (prefill,
//! multi-row).  Any mistake in the offset binding or the dispatch mirror
//! shows up here as garbage, not as a subtle slowdown.

use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{Device, Module, Tensor};
use joshua::zero_copy_metal::{page_size, ZcContext};
use std::io::Cursor;
use std::sync::Arc;

mod common;

/// Map a file with its length rounded up to a whole page, as the engine now
/// does — required for `newBufferWithBytesNoCopy`.
fn map_padded(file: &std::fs::File) -> memmap2::Mmap {
    let len = file.metadata().unwrap().len() as usize;
    let map_len = len.next_multiple_of(page_size());
    unsafe { memmap2::MmapOptions::new().len(map_len).map(file).unwrap() }
}

/// Write a GGUF whose only tensor is a Q4_K `[n, k]` weight named `w`.
fn write_q4k_gguf(path: &std::path::Path, n: usize, k: usize) {
    let w: Vec<f32> = common::weights(n * k, 0x5EED);
    let qt = common::qtensor_q4k(w, &[n, k]);
    let metadata: [(&str, &gguf_file::Value); 0] = [];
    gguf_file::write(
        &mut std::fs::File::create(path).unwrap(),
        &metadata,
        &[("w", &qt)],
    )
    .unwrap();
}

#[test]
fn zc_weight_matches_candle_qmatmul() {
    // Skip cleanly when no Metal device is available.
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("no Metal device, skipping");
            return;
        }
    };
    let Device::Metal(md) = &dev else {
        unreachable!()
    };

    let dir = common::model_dir("zc-parity");
    let path = dir.join("w.gguf");
    // 256 must divide `k` (Q4_K block size); 512 rows for a realistic shape.
    let (n, k) = (512usize, 256usize);
    write_q4k_gguf(&path, n, k);

    let file = std::fs::File::open(&path).unwrap();
    let mmap = Arc::new(map_padded(&file));
    let mut cursor = Cursor::new(&mmap[..]);
    let content = gguf_file::Content::read(&mut cursor).unwrap();

    // Zero-copy weight: bound inside the no-copy buffer over the mapping.
    let zc = Arc::new(ZcContext::new(md, Arc::clone(&mmap)).unwrap());
    let zw = zc
        .weight(&content, "w")
        .unwrap()
        .expect("Q4_K weight must be zero-copy-able");

    // Reference: candle's own copy path on the same bytes.
    let mut cursor = Cursor::new(&mmap[..]);
    let content = gguf_file::Content::read(&mut cursor).unwrap();
    let qt = content.tensor(&mut cursor, "w", &dev).unwrap();
    let cw = QMatMul::from_qtensor(qt).unwrap();

    let assert_close = |a: &Tensor, b: &Tensor, what: &str| {
        let a: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(a.len(), b.len());
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "{what}: max diff {max_diff} (zc vs candle) on {} elements",
            a.len()
        );
    };

    // Decode path: single-row input → matmul_mv.
    let x1 = Tensor::randn(0f32, 1f32, (1, k), &dev).unwrap();
    assert_close(&zw.forward(&x1).unwrap(), &cw.forward(&x1).unwrap(), "decode");

    // Prefill path: multi-row input → matmul_mm.
    let x4 = Tensor::randn(0f32, 1f32, (4, k), &dev).unwrap();
    assert_close(&zw.forward(&x4).unwrap(), &cw.forward(&x4).unwrap(), "prefill");

    // Batched single-row (the `[b, s, k]` decode shape the model feeds).
    let xb = Tensor::randn(0f32, 1f32, (2, 1, k), &dev).unwrap();
    assert_close(
        &zw.forward(&xb).unwrap(),
        &cw.forward(&xb).unwrap(),
        "batched decode",
    );

    std::fs::remove_dir_all(&dir).ok();
}
