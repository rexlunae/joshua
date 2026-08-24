//! Dump the metadata and tensor layout of a qwen3moe GGUF file.
//!
//! Prints config keys plus, per tensor, dtype / shape / file offset, and
//! flags tensors whose offset is not page-aligned (zero-copy Metal buffers
//! require page-aligned pointers, so this shows what the loader would need
//! to remap).
//!
//! Usage: `cargo run --release --example dump_cfg [model.gguf]`
use candle_core::quantized::gguf_file;
use std::io::Cursor;

const PAGE: u64 = 16 * 1024;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/Users/tserica/models/Qwen3-30B-A3B-Q4_K_M.gguf".to_string());
    let bytes = std::fs::read(&path).unwrap();
    let mut cursor = Cursor::new(&bytes[..]);
    let ct = gguf_file::Content::read(&mut cursor).unwrap();

    println!("== metadata ==");
    for (k, v) in ct.metadata.iter() {
        if k.starts_with("qwen3moe") || k == "general.architecture" {
            println!("{k} = {v:?}");
        }
    }

    println!("\n== tensor data ==");
    println!(
        "tensor_data_offset = {} (page-aligned: {})",
        ct.tensor_data_offset,
        ct.tensor_data_offset.is_multiple_of(PAGE)
    );
    let mut misaligned = 0usize;
    for (name, ti) in ct.tensor_infos.iter() {
        let off = ct.tensor_data_offset.saturating_add(ti.offset);
        let aligned = off % PAGE == 0;
        if !aligned {
            misaligned += 1;
        }
        println!(
            "{name} {:?} {:?} offset={} (page-aligned: {aligned})",
            ti.ggml_dtype, ti.shape, off
        );
    }
    println!(
        "{} tensors, {} with non-page-aligned offsets",
        ct.tensor_infos.len(),
        misaligned
    );
}
