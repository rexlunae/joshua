//! Append cost of the layer KV cache over a decode-length generation: the
//! old exact-size cache (per-step `Tensor::cat` + `contiguous`) against the
//! in-place append into preallocated `[b, n_kv, cap, d]` buffers.
//!
//! Both arms append the same single-token rows for the same number of steps
//! at the same geometry — deepseek2-style reconstructed per-head K/V for one
//! request — and time nothing but the append.  The attention itself reads
//! both caches identically and is not part of the measurement.  CPU only:
//! `cargo build --release --example bench_kv_cache_append`.

use std::time::Instant;

use joshua::attention::SeqKvCache;

fn main() -> candle_core::Result<()> {
    let dev = candle_core::Device::Cpu;
    let (b, n_head, dk, dv, steps) = (1usize, 16usize, 192usize, 128usize, 2048usize);

    // One token's keys and values, reused for every step: the append work
    // (and not row construction) is what is being timed.
    let row = |d: usize| -> candle_core::Result<candle_core::Tensor> {
        let data: Vec<f32> = (0..b * n_head * d)
            .map(|i| ((i * 37 + d) % 251) as f32 / 7.0 - 17.0)
            .collect();
        candle_core::Tensor::from_vec(data, (b, n_head, 1, d), &dev)
    };
    let k_row = row(dk)?;
    let v_row = row(dv)?;

    // Old cache: exact-size tensors re-cat'd and made contiguous every step.
    let start = Instant::now();
    let mut k = k_row.clone();
    let mut v = v_row.clone();
    for _ in 1..steps {
        k = candle_core::Tensor::cat(&[&k, &k_row], 3)?.contiguous()?;
        v = candle_core::Tensor::cat(&[&v, &v_row], 3)?.contiguous()?;
    }
    let cat = start.elapsed();

    // New cache: in-place append into preallocated buffers, doubling when
    // full (the growth copies are part of the mechanism).
    let start = Instant::now();
    let mut cache = SeqKvCache::new(&k_row, &v_row)?;
    for _ in 1..steps {
        cache.append(&k_row, &v_row)?;
    }
    let inplace = start.elapsed();

    println!("geometry: b={b} n_head={n_head} dk={dk} dv={dv} steps={steps}");
    println!(
        "cat (old exact-size cache): {:>10.2?} total, {:>8.2?} / step",
        cat,
        cat / steps as u32
    );
    println!(
        "in-place append:            {:>10.2?} total, {:>8.2?} / step",
        inplace,
        inplace / steps as u32
    );
    println!(
        "final cache: {steps} rows, {} MiB per arm (k+v, f32)",
        (n_head * (dk + dv) * steps * 4) / (1024 * 1024)
    );
    Ok(())
}
