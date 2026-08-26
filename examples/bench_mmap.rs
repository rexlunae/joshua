//! Reproduce the in-model expert matmul context: borrowed mmap slices,
//! round-robin routing pattern — vs owned quantized copies.
//!
//! Usage: cargo run --release --example bench_mmap -- <model.gguf>

use std::fs::File;
use std::io::{Cursor, Read, Seek};
use std::sync::Arc;
use std::time::Instant;

use candle_core::quantized::gguf_file;
use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{Device, Module, Shape, Tensor};
use joshua::mmap_tensor;

fn bench(name: &str, iters: usize, mut f: impl FnMut()) {
    f();
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    println!(
        "  {name:<52} {:>8.3} ms/call",
        t.elapsed().as_secs_f64() * 1e3 / iters as f64
    );
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: bench_mmap <model.gguf>");
    let dev = Device::Cpu;

    let mut f = File::open(&path).unwrap();
    let mut hb = Vec::with_capacity(16 * 1024 * 1024);
    f.by_ref().take(16 * 1024 * 1024).read_to_end(&mut hb).unwrap();
    let mut c = Cursor::new(&hb[..]);
    let content = gguf_file::Content::read(&mut c).unwrap();
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&f).unwrap() });

    // Expert gate tensor: [128, 768, 2048] Q4_K.
    let name = "blk.10.ffn_gate_exps.weight";
    let info = content.tensor_infos.get(name).expect(name);
    let gdims = info.shape.dims();
    let (ne, out_d, in_d) = (gdims[0], gdims[1], gdims[2]);
    println!("{name}: dtype={:?} dims={gdims:?}", info.ggml_dtype);
    let per_elems = out_d * in_d;
    let per_bytes = per_elems / info.ggml_dtype.block_size() * info.ggml_dtype.type_size();
    let base = (content.tensor_data_offset + info.offset) as usize;

    // Borrow all experts like the loader does.
    let borrowed: Vec<QMatMul> = (0..ne)
        .map(|e| {
            let qt = mmap_tensor::borrowed_range(
                &mmap,
                info.ggml_dtype,
                base + e * per_bytes,
                Shape::from((out_d, in_d)),
            )
            .unwrap()
            .expect("aligned borrow");
            QMatMul::from_qtensor(qt).unwrap()
        })
        .collect();

    let xs = Tensor::randn(0f32, 1f32, (1, in_d), &dev)
        .unwrap()
        .contiguous()
        .unwrap();

    // (a) same borrowed expert repeatedly (cache-hot single slice).
    bench("borrowed, SAME expert every call", 100, || {
        borrowed[0].forward(&xs).unwrap();
    });

    // (b) round-robin over all 128 (mimics routing spreading across experts).
    bench("borrowed, round-robin all 128 experts", 2, || {
        for e in borrowed.iter() {
            e.forward(&xs).unwrap();
        }
    });

    // (c) owned copy of one expert for comparison.
    let bytes = mmap[base..base + ne * per_bytes].to_vec();
    let storage =
        QStorage::from_data(std::borrow::Cow::Owned(bytes[0..per_bytes].to_vec()), &dev, GgmlDType::Q4K)
            .unwrap();
    let owned = QMatMul::from_qtensor(QTensor::new(storage, (out_d, in_d)).unwrap()).unwrap();
    bench("owned copy, same expert every call", 100, || {
        owned.forward(&xs).unwrap();
    });

    // (d) attention-weight-shaped borrowed tensor [4096, 2048] Q4_K.
    let aname = "blk.10.attn_q.weight";
    let ainfo = content.tensor_infos.get(aname).expect(aname);
    let aqt = mmap_tensor::borrowed_qtensor(&mmap, ainfo, content.tensor_data_offset)
        .unwrap()
        .expect("borrow attn_q");
    let adims = ainfo.shape.dims();
    let axs = Tensor::randn(0f32, 1f32, (1, adims[1]), &dev)
        .unwrap()
        .contiguous()
        .unwrap();
    let aqm = QMatMul::from_qtensor(aqt).unwrap();
    bench("borrowed attn_q [4096,2048] m=1", 200, || {
        aqm.forward(&axs).unwrap();
    });

    stream_probe::run(Arc::clone(&mmap), base, per_bytes, ne as usize);
}

// ── appended: raw streaming probe over expert regions ────────────────────
mod stream_probe {
    use std::sync::Arc;
    use std::time::Instant;

    /// Sum u32 words across `experts_per_thread` distinct expert regions.
    fn stream(mmap: &Arc<memmap2::Mmap>, bases: &[usize], len: usize) -> u64 {
        let mut acc = 0u64;
        for &b in bases {
            let chunk = &mmap[b..b + len];
            let words = chunk.chunks_exact(4);
            for w in words {
                acc += u32::from_le_bytes([w[0], w[1], w[2], w[3]]) as u64;
            }
        }
        acc
    }

    pub fn run(mmap: Arc<memmap2::Mmap>, base0: usize, per_bytes: usize, n_experts: usize) {
        println!("stream probe: {n_experts} experts x {per_bytes} B regions");
        for threads in [1usize, 4, 8] {
            // Each thread gets a disjoint slice of the expert list.
            let per = n_experts / threads;
            let mut handles = Vec::new();
            let t0 = Instant::now();
            let iters = 4;
            for t in 0..threads {
                let m = Arc::clone(&mmap);
                handles.push(std::thread::spawn(move || {
                    let mut acc = 0u64;
                    let bases: Vec<usize> =
                        (0..per).map(|i| base0 + (t * per + i) * per_bytes).collect();
                    for _ in 0..iters {
                        acc += stream(&m, &bases, per_bytes);
                    }
                    acc
                }));
            }
            let total: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
            let el = t0.elapsed().as_secs_f64();
            let bytes = total.saturating_mul(0) + iters * (n_experts * per_bytes) as u64;
            println!(
                "  {threads:>2} threads: {el:>7.3} s  -> {:>6.1} GB/s  (sum {total})",
                bytes as f64 / el / 1e9
            );
        }
    }
}
