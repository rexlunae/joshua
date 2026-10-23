//! Print compressor-related tensor names and compression metadata from a
//! GGUF header, for comparing conversion conventions (e.g. llama.cpp-style
//! DeepSeek-V4.1 files against ds4-convert output).
//!
//! Usage: cargo run --release --example dump_compressors -- <model.gguf> [substring]

use std::io::{Cursor, Read};

fn main() {
    let path = std::env::args().nth(1).expect("usage: dump_compressors <model.gguf> [substring]");
    let needle = std::env::args().nth(2).unwrap_or_else(|| "compressor".to_string());
    let mut f = std::fs::File::open(&path).unwrap();
    let mut header_bytes = Vec::with_capacity(16 * 1024 * 1024);
    f.by_ref()
        .take(16 * 1024 * 1024)
        .read_to_end(&mut header_bytes)
        .unwrap();
    let mut c = Cursor::new(&header_bytes[..]);
    let h = joshua::gguf_ext::read_header(&mut c).unwrap();

    for k in h.metadata.keys() {
        if k.contains("compress") || k.contains("hyper") || k.contains("index") {
            println!("{k} = {:?}", h.metadata.get(k).unwrap());
        }
    }
    let mut names: Vec<&String> = h.tensors.keys().filter(|n| n.contains(&needle)).collect();
    names.sort();
    for n in &names {
        println!("{n} dtype={} dims={:?}", h.tensors[*n].dtype, h.tensors[*n].dims);
    }
    println!("--- {} matching tensors", names.len());
}
