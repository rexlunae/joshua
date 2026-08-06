//! Dump all tensors of a GGUF with sizes computed from file offsets, and
//! classify them into dense (always-touched) vs routed-expert weights.
//!
//! Usage: `cargo run --release --example tensor_sizes -- <model.gguf>`

use std::io::{Cursor, Read};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: tensor_sizes <model.gguf>");
    let mut f = std::fs::File::open(&path).unwrap();
    let mut header_bytes = Vec::with_capacity(64 * 1024 * 1024);
    f.by_ref()
        .take(64 * 1024 * 1024)
        .read_to_end(&mut header_bytes)
        .unwrap();
    let mut c = Cursor::new(&header_bytes[..]);
    let h = joshua::gguf_ext::read_header(&mut c).unwrap();

    let mut v: Vec<(&String, u64)> = h
        .tensors
        .iter()
        .map(|(n, t)| (n, t.offset))
        .collect();
    v.sort_by_key(|(_, o)| *o);

    // Sizes from offset deltas (tensor data is stored back-to-back, 32-byte aligned).
    let mut sizes: Vec<(u64, &String)> = Vec::new();
    for i in 0..v.len() {
        let (name, off) = v[i];
        let sz = if i + 1 < v.len() { v[i + 1].1 - off } else { 0 };
        sizes.push((sz, name));
    }

    let mut dense: u64 = 0;
    let mut experts: u64 = 0;
    let (mut dc, mut ec) = (0usize, 0usize);
    for (sz, name) in &sizes {
        if name.contains(".ffn_gate_exps")
            || name.contains(".ffn_down_exps")
            || name.contains(".ffn_up_exps")
        {
            experts += sz;
            ec += 1;
        } else {
            dense += sz;
            dc += 1;
        }
    }
    println!("dense   : {:8.2} GiB  ({} tensors)", dense as f64 / 2f64.powi(30), dc);
    println!("experts : {:8.2} GiB  ({} tensors)", experts as f64 / 2f64.powi(30), ec);
    println!("total   : {:8.2} GiB  ({} tensors)", (dense + experts) as f64 / 2f64.powi(30), sizes.len());

    sizes.sort_by_key(|(sz, _)| std::cmp::Reverse(*sz));
    println!("\nlargest dense tensors:");
    for (sz, name) in sizes.iter().take(12) {
        if name.contains(".ffn_gate_exps") || name.contains(".ffn_down_exps") || name.contains(".ffn_up_exps") {
            continue;
        }
        println!("  {:6.2} GiB  {}", *sz as f64 / 2f64.powi(30), name);
    }
    println!("largest expert tensors:");
    for (sz, name) in sizes.iter().take(8) {
        if name.contains(".ffn_gate_exps") || name.contains(".ffn_down_exps") || name.contains(".ffn_up_exps") {
            println!("  {:6.2} GiB  {}", *sz as f64 / 2f64.powi(30), name);
        }
    }
}
