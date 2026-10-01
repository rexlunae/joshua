//! Microbenchmark for the activation path of [`joshua::quant_matmul::try_fast_cpu_qmatmul`]
//! (PR: borrow the f32 activations from candle's storage instead of
//! materializing an owned `Vec` per call).
//!
//! The old per-call sequence copied every activation (`to_vec1::<f32>` =
//! malloc + memcpy of m·k f32) before the kernel; the new path borrows the
//! flat rows from candle's storage.  The kernel is unchanged, so the delta
//! between the two sequences below is exactly the removed copy.
//!
//! **Microbenchmark, not an end-to-end measurement**: per-call times at the
//! shapes the hook sees (decode: one token; prefill: one 512-token chunk),
//! CPU backend, `synchronize`-free (CPU has no queue).  Run on hardware you
//! own — do not run on the dev machine (see AGENTS.md).

use candle_core::quantized::k_quants::BlockQ4K;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Result, Tensor};
use joshua::quant_matmul::{matmul_kquant, try_fast_cpu_qmatmul};

const CALLS: usize = 200;
const PASSES: usize = 5;

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn bench(backend: &str, dev: &Device, m: usize, k: usize, n: usize) -> Result<()> {
    // A real Q4_K weight and f32 activations, like the hook sees.
    let w = Tensor::randn(0f32, 1f32, (n, k), dev)?;
    let qt = QTensor::quantize(&w, GgmlDType::Q4K)?;
    let xs = Tensor::randn(0f32, 1f32, (m, k), dev)?;

    // Parity: the borrowed path (try_fast_cpu_qmatmul) must match the old
    // copy-then-kernel sequence exactly (same kernel, same inputs).
    let bytes = qt.data()?;
    let blocks = unsafe {
        std::slice::from_raw_parts(bytes.as_ptr() as *const BlockQ4K, bytes.len() / std::mem::size_of::<BlockQ4K>())
    };
    let v = xs.flatten_all()?.to_vec1::<f32>()?;
    let mut dst = vec![0f32; m * n];
    matmul_kquant::<BlockQ4K>((m, k, n), &v, blocks, &mut dst)?;
    let old_ref = Tensor::from_vec(dst, (m, n), dev)?;
    let new = try_fast_cpu_qmatmul(&qt, &xs).expect("fast path applies")?;
    assert_eq!(
        old_ref.flatten_all()?.to_vec1::<f32>()?,
        new.flatten_all()?.to_vec1::<f32>()?,
        "borrowed path diverged from copy+kernel at m={m}"
    );

    let mut old_us = Vec::with_capacity(PASSES);
    let mut new_us = Vec::with_capacity(PASSES);
    for _ in 0..PASSES {
        let t = std::time::Instant::now();
        for _ in 0..CALLS {
            let v = xs.flatten_all()?.to_vec1::<f32>()?;
            let mut dst = vec![0f32; m * n];
            matmul_kquant::<BlockQ4K>((m, k, n), &v, blocks, &mut dst)?;
            drop(dst);
        }
        old_us.push(t.elapsed().as_secs_f64() * 1e6 / CALLS as f64);
        let t = std::time::Instant::now();
        for _ in 0..CALLS {
            drop(try_fast_cpu_qmatmul(&qt, &xs).expect("fast path applies")?);
        }
        new_us.push(t.elapsed().as_secs_f64() * 1e6 / CALLS as f64);
    }
    println!(
        "{backend} m={m:>4} k={k} n={n}: copy+kernel {:9.2} us/call   borrowed {:9.2} us/call   (median of {PASSES} x {CALLS})",
        median(&mut old_us),
        median(&mut new_us),
    );
    Ok(())
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    // Decode (one token) and a 512-token prefill chunk, Qwen3-30B-ish dims.
    bench("cpu  ", &dev, 1, 2048, 2048)?;
    bench("cpu  ", &dev, 512, 2048, 2048)?;
    Ok(())
}
