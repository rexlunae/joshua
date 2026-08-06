//! Dump a GGUF's metadata and tensor inventory using the tolerant header
//! reader (keeps dtypes candle cannot name, e.g. IQ2_XXS / I32), so files
//! like the DeepSeek-V4-Flash GGUFs can be inspected without loading them.
//!
//! Usage: `cargo run --release --example dump_meta -- <model.gguf>`

use std::io::{Cursor, Read};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: dump_meta <model.gguf>");
    let mut f = std::fs::File::open(&path).unwrap();
    // A single `read` may return fewer bytes than requested (signals, pipe
    // semantics), so drain up to the header cap with `read_to_end`.
    let mut header_bytes = Vec::with_capacity(16 * 1024 * 1024);
    f.by_ref()
        .take(16 * 1024 * 1024)
        .read_to_end(&mut header_bytes)
        .unwrap();
    let mut c = Cursor::new(&header_bytes[..]);
    let h = joshua::gguf_ext::read_header(&mut c).unwrap();

    let mut keys: Vec<&String> = h.metadata.keys().collect();
    keys.sort();
    for k in keys {
        let v = h.metadata.get(k).unwrap();
        println!("{k} = {v:?}");
    }

    println!("--- tensors: {}", h.tensors.len());
    let mut names: Vec<&String> = h.tensors.keys().collect();
    names.sort();
    for n in names.iter().take(80) {
        let t = &h.tensors[*n];
        println!(
            "{n} dtype={} dims={:?} offset={}",
            t.dtype, t.dims, t.offset
        );
    }
}
