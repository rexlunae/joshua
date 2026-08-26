//! Microbenchmark: quantized matmul variants at real Qwen3-Coder-30B-A3B
//! decode/prefill shapes, on the dtypes the Q4_K_M GGUF actually uses.
//!
//! Compares:
//!   1. candle's `QMatMul::forward` (what joshua used before the fast-path
//!      hook; includes candle's aarch64 dotprod Q4Kx8 repack when compiled
//!      in via `target_feature = "dotprod"`, e.g. `-C target-cpu=native`),
//!   2. joshua's `try_fast_cpu_qmatmul` (the hook: fused kernels for
//!      Q8_0/Q2K/Q4K, dequant+SIMD-dot otherwise),
//!   3. `try_matmul_fused` directly, serial and parallel,
//!   4. the generic k-quant path (`matmul_kquant`) for Q6K.
//!
//! Usage: cargo run --release --example bench_neon

use std::sync::Arc;
use std::time::Instant;

use candle_core::quantized::k_quants::BlockQ6K;
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Tensor};

fn bench(name: &str, iters: usize, mut f: impl FnMut()) {
    f(); // warmup
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    let el = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    println!("  {name:<44} {el:>9.3} ms/call");
}

fn main() {
    let dev = Device::Cpu;
    println!(
        "neon_available={} avx2={}",
        joshua::simd::neon_available(),
        joshua::simd::avx2_fma_available()
    );

    // (dtype, n_out, k_in) — shapes straight from the GGUF.
    let cases: Vec<(GgmlDType, usize, usize)> = vec![
        (GgmlDType::Q4K, 768, 2048), // expert gate/up slice
        (GgmlDType::Q6K, 2048, 768), // early-layer ffn_down
        (GgmlDType::Q6K, 512, 2048), // early-layer attn_v
        (GgmlDType::Q4K, 4096, 2048), // attn_q
        // The LM head — 311M MACs EVERY generated token; n huge enough
        // that row-parallelism should finally pay off.
        (GgmlDType::Q6K, 151936, 2048),
    ];

    for (dtype, n, k) in cases {
        println!("{dtype:?} [{n},{k}]:");
        let data: Vec<f32> = (0..n * k)
            .map(|i| ((((i as u64) * 2654435761) % 100000) as f32 / 1000.0) - 50.0)
            .collect();
        let t = Tensor::from_vec(data, (n, k), &dev).unwrap();
        let qt = Arc::new(QTensor::quantize(&t, dtype).unwrap());

        for &m in &[1usize, 32usize] {
            println!("  m = {m}");
            let xs = Tensor::randn(0f32, 1f32, (m, k), &dev)
                .unwrap()
                .contiguous()
                .unwrap();
            let lhs = xs.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let bytes = qt.data().unwrap().to_vec();
            let iters = if m == 1 { 30 } else { 10 };

            // 1. candle's own path (pre-hook behaviour).
            let qm = QMatMul::from_arc(qt.clone()).unwrap();
            bench("candle QMatMul::forward", iters, || {
                qm.forward(&xs).unwrap();
            });

            // 2. joshua's hooked fast path (may legitimately defer).
            if joshua::quant_matmul::try_fast_cpu_qmatmul(&qt, &xs).is_some() {
                bench("joshua try_fast_cpu_qmatmul", iters, || {
                    joshua::quant_matmul::try_fast_cpu_qmatmul(&qt, &xs)
                        .expect("just checked")
                        .unwrap();
                });
            } else {
                println!("  {:<44} deferred to candle", "joshua try_fast_cpu_qmatmul");
            }

            // 3. fused kernels directly, serial and parallel (only the
            // dtypes that have fused kernels).
            let mut dst = vec![0f32; m * n];
            if matches!(dtype, GgmlDType::Q8_0 | GgmlDType::Q2K | GgmlDType::Q4K) {
                bench(
                    "try_matmul_fused serial",
                    iters,
                    || {
                        assert!(joshua::kquant_dot::try_matmul_fused(
                            dtype,
                            (m, k, n),
                            &lhs,
                            &bytes,
                            &mut dst,
                            false
                        ));
                    },
                );
                bench(
                    "try_matmul_fused parallel",
                    iters,
                    || {
                        assert!(joshua::kquant_dot::try_matmul_fused(
                            dtype,
                            (m, k, n),
                            &lhs,
                            &bytes,
                            &mut dst,
                            true
                        ));
                    },
                );
            }

            // 4. generic k-quant path (Q6K et al).
            if dtype == GgmlDType::Q6K {
                let blocks: &[BlockQ6K] = unsafe {
                    std::slice::from_raw_parts(
                        bytes.as_ptr() as *const BlockQ6K,
                        bytes.len() / std::mem::size_of::<BlockQ6K>(),
                    )
                };
                bench("generic matmul_kquant::<BlockQ6K>", iters, || {
                    joshua::quant_matmul::matmul_kquant((m, k, n), &lhs, blocks, &mut dst)
                        .unwrap();
                });
            }
        }
    }
}
