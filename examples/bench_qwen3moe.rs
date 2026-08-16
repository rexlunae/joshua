//! Benchmark the qwen3moe loader: load time, prefill throughput and decode
//! throughput, on Metal (zero-copy), the Metal copying path, or CPU.
//!
//! Usage:
//! ```sh
//! cargo run --release --features metal --example bench_qwen3moe -- \
//!     /path/to/model.gguf --device metal --prefill 512 --decode 128
//! ```
//!
//! `--copy` loads through candle's copy path (no mmap, weights uploaded) —
//! the behaviour before the zero-copy work.  Peak RSS is sampled in-process
//! throughout and reported alongside the timings.

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use joshua::model::QuantizedModel;
use std::io::Cursor;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

fn usage() -> ! {
    eprintln!(
        "usage: bench_qwen3moe <model.gguf> [--device metal|cpu] [--prefill N] \
         [--decode N] [--copy]"
    );
    std::process::exit(2);
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }
    let mut model_path = None;
    let mut device = "metal".to_string();
    let mut prefill = 256usize;
    let mut decode = 64usize;
    let mut copy = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--device" => {
                i += 1;
                device = args.get(i).cloned().unwrap_or_else(|| usage());
            }
            "--prefill" => {
                i += 1;
                prefill = args.get(i).and_then(|s| s.parse().ok()).unwrap_or_else(|| usage());
            }
            "--decode" => {
                i += 1;
                decode = args.get(i).and_then(|s| s.parse().ok()).unwrap_or_else(|| usage());
            }
            "--copy" => copy = true,
            s if s.starts_with('-') => usage(),
            s => model_path = Some(s.to_string()),
        }
        i += 1;
    }
    let model_path = model_path.unwrap_or_else(|| usage());

    let dev = match device.as_str() {
        "metal" => Device::new_metal(0).expect("Metal device unavailable"),
        "cpu" => Device::Cpu,
        other => {
            eprintln!("unknown device {other}");
            usage();
        }
    };
    println!(
        "device: {device} | prefill: {prefill} tok | decode: {decode} tok | \
         path: {model_path} ({} bytes)",
        std::fs::metadata(&model_path)?.len()
    );

    // Peak RSS sampler (macOS/Linux: `ps -o rss`).
    let pid = std::process::id();
    let peak: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let sampler = {
        let peak = Arc::clone(&peak);
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let out = std::process::Command::new("ps")
                    .args(["-o", "rss=", "-p", &pid.to_string()])
                    .output();
                if let Ok(out) = out {
                    if let Ok(s) = String::from_utf8(out.stdout) {
                        if let Ok(kb) = s.trim().parse::<u64>() {
                            let _ = peak.fetch_max(kb, Ordering::Relaxed);
                        }
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        })
    };

    let file = std::fs::File::open(&model_path)?;
    let file_len = file.metadata()?.len() as usize;
    let page = joshua::zero_copy_metal::page_size();
    let t = std::time::Instant::now();
    let mmap = unsafe {
        memmap2::MmapOptions::new()
            .len(file_len.next_multiple_of(page))
            .map(&file)?
    };
    println!("mmap: {:.2}s", t.elapsed().as_secs_f64());
    let mmap_arc = Arc::new(mmap);

    // ── Load ──
    let t0 = Instant::now();
    let mut cursor = Cursor::new(&mmap_arc[..]);
    let content = gguf_file::Content::read(&mut cursor)?;
    println!(
        "Content::read: {:.2}s (header)",
        t0.elapsed().as_secs_f64()
    );
    let t0 = Instant::now();
    let model = if copy {
        QuantizedModel::from_gguf(content, &mut cursor, &dev)?
    } else {
        QuantizedModel::from_gguf_mmap(
            content,
            &mut cursor,
            &dev,
            Some(Arc::clone(&mmap_arc)),
        )?
    };
    let load_s = t0.elapsed().as_secs_f64();
    println!(
        "load: {load_s:.2}s | peak RSS so far: {:.1} GiB",
        peak.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0)
    );

    let mut model = model;

    // ── Prefill ──
    let input = Tensor::new(vec![42u32; prefill], &dev)?
        .reshape((1, prefill))?;
    let t0 = Instant::now();
    let logits = model.forward(&input, 0)?;
    let prefill_s = t0.elapsed().as_secs_f64();
    let prefill_tps = prefill as f64 / prefill_s;
    println!("prefill: {prefill} tok in {prefill_s:.2}s = {prefill_tps:.1} tok/s");
    dump_topk(&logits, 5, "prefill");

    // ── Decode ──
    let t0 = Instant::now();
    let mut last_top = 0u32;
    for step in 0..decode {
        let tok = Tensor::new(vec![42u32], &dev)?.reshape((1, 1))?;
        let logits = model.forward(&tok, prefill + step)?;
        last_top = logits.argmax(candle_core::D::Minus1)?.flatten_all()?.to_vec1()?[0];
        if step == 0 {
            dump_topk(&logits, 5, "decode-step-0");
        }
    }
    let decode_s = t0.elapsed().as_secs_f64();
    let decode_tps = decode as f64 / decode_s;
    println!("decode: {decode} tok in {decode_s:.2}s = {decode_tps:.1} tok/s");
    println!("  top token after decode: {last_top}");

    let peak_gib = peak.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0);
    println!("peak RSS: {peak_gib:.2} GiB");
    stop.store(true, Ordering::Relaxed);
    sampler.join().ok();
    Ok(())
}

/// Print the top-`k` token ids and logits from `[1, vocab]` logits, plus the
/// logits of a fixed probe set so CPU and Metal runs can be compared directly.
fn dump_topk(logits: &candle_core::Tensor, k: usize, label: &str) -> anyhow::Result<()> {
    let flat: Vec<f32> = logits.flatten_all()?.to_vec1()?;
    let mut order: Vec<usize> = (0..flat.len()).collect();
    order.sort_by(|&a, &b| flat[b].total_cmp(&flat[a]));
    let top: Vec<(usize, f32)> = order[..k.min(order.len())].iter().map(|&i| (i, flat[i])).collect();
    println!("  top-{k} {label}: {top:?}");
    // Fixed probe set: the near-tie tokens observed on both backends.
    for &t in &[323usize, 525, 61516, 5122, 760, 2073, 20412, 313] {
        println!("    logit[{t}] = {:.4}", flat[t]);
    }
    Ok(())
}
