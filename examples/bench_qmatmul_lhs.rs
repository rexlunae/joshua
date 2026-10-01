//! Microbenchmark for the activation path of [`joshua::quant_matmul::try_fast_cpu_qmatmul`]
//! (PR: borrow the f32 activations from candle's storage instead of
//! materializing an owned `Vec` per call).
//!
//! The two timed arms are identical except for the input handling — both
//! run the same kernel and construct the output the same way — so the
//! delta is exactly the removed activation copy (malloc + memcpy of m·k
//! f32):
//!
//! * *copy + kernel*: `xs.flatten_all().to_vec1()` → `matmul_kquant` →
//!   `Tensor::from_vec(dst)`
//! * *borrowed*: borrow `buf[start..start+m*k]` from candle's storage →
//!   `matmul_kquant` → `Tensor::from_vec(dst)`
//!
//! When the production hook takes the fast path on this architecture
//! (it defers on aarch64 dotprod builds for Q4_K with n%8==0 and for
//! m < 8), the borrowed arm is also asserted bit-identical to
//! [`try_fast_cpu_qmatmul`]'s output.
//!
//! **Microbenchmark, not an end-to-end measurement**: per-call times at the
//! shapes the hook sees (decode: one token; prefill: one 512-token chunk),
//! CPU backend.  Run on hardware you own — do not run on the dev machine
//! (see AGENTS.md).

use candle_core::quantized::k_quants::BlockQ4K;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Result, Storage, Tensor};
use joshua::quant_matmul::{matmul_kquant, try_fast_cpu_qmatmul};

const CALLS: usize = 200;
const PASSES: usize = 5;

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

/// The production hook's borrow, inlined.  The storage read guard must stay
/// alive in the caller's scope for the slice to remain valid, so the macro
/// binds `$guard` and `$lhs` there instead of yielding a reference.
macro_rules! borrowed_lhs {
    ($xs:expr, $m:expr, $k:expr, $guard:ident, $lhs:ident) => {
        let ($guard, layout) = $xs.storage_and_layout();
        let cpu = match &*$guard {
            Storage::Cpu(cpu) => cpu,
            _ => unreachable!("CPU tensor"),
        };
        let buf = cpu.as_slice::<f32>()?;
        let start = layout.start_offset();
        let $lhs = &buf[start..start + $m * $k];
    };
}

fn bench(backend: &str, dev: &Device, m: usize, k: usize, n: usize) -> Result<()> {
    // A real Q4_K weight and f32 activations, like the hook sees.
    let w = Tensor::randn(0f32, 1f32, (n, k), dev)?;
    let qt = QTensor::quantize(&w, GgmlDType::Q4K)?;
    let xs = Tensor::randn(0f32, 1f32, (m, k), dev)?;
    let bytes = qt.data()?;
    let blocks = unsafe {
        std::slice::from_raw_parts(
            bytes.as_ptr() as *const BlockQ4K,
            bytes.len() / std::mem::size_of::<BlockQ4K>(),
        )
    };

    let kernel_from = |lhs: &[f32]| -> Result<Tensor> {
        let mut dst = vec![0f32; m * n];
        matmul_kquant::<BlockQ4K>((m, k, n), lhs, blocks, &mut dst)?;
        Tensor::from_vec(dst, (m, n), dev)
    };

    // Parity: the borrowed arm must match the copy arm exactly, and both
    // must match the production hook whenever it takes the fast path here.
    {
        let copy = kernel_from(&xs.flatten_all()?.to_vec1::<f32>()?)?;
        borrowed_lhs!(xs, m, k, guard, lhs_p);
        let borrow = kernel_from(lhs_p)?;
        assert_eq!(
            copy.flatten_all()?.to_vec1::<f32>()?,
            borrow.flatten_all()?.to_vec1::<f32>()?,
            "borrowed arm diverged at m={m}"
        );
        match try_fast_cpu_qmatmul(&qt, &xs) {
            Some(Ok(prod)) => assert_eq!(
                prod.flatten_all()?.to_vec1::<f32>()?,
                borrow.flatten_all()?.to_vec1::<f32>()?,
                "production hook diverged at m={m}"
            ),
            // The hook defers on some architectures (aarch64 dotprod: Q4_K
            // with n%8==0, or m<8) — nothing to compare against there.
            Some(Err(e)) => return Err(e),
            None => println!("  (hook defers on this arch; timing the borrow inline)"),
        }
    }

    let mut copy_us = Vec::with_capacity(PASSES);
    let mut borrow_us = Vec::with_capacity(PASSES);
    for _ in 0..PASSES {
        let t = std::time::Instant::now();
        for _ in 0..CALLS {
            drop(kernel_from(&xs.flatten_all()?.to_vec1::<f32>()?)?);
        }
        copy_us.push(t.elapsed().as_secs_f64() * 1e6 / CALLS as f64);
        let t = std::time::Instant::now();
        for _ in 0..CALLS {
            borrowed_lhs!(xs, m, k, guard, lhs);
            drop(kernel_from(lhs)?);
        }
        borrow_us.push(t.elapsed().as_secs_f64() * 1e6 / CALLS as f64);
    }
    println!(
        "{backend} m={m:>4} k={k} n={n}: copy+kernel {:9.2} us/call   borrowed {:9.2} us/call   (median of {PASSES} x {CALLS})",
        median(&mut copy_us),
        median(&mut borrow_us),
    );
    Ok(())
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    // Decode (one token) and a 512-token prefill chunk, Qwen3-30B-ish dims.
    // n=2048 is divisible by 8: on aarch64 dotprod builds the hook defers
    // (noted above) instead of panicking.
    bench("cpu  ", &dev, 1, 2048, 2048)?;
    bench("cpu  ", &dev, 512, 2048, 2048)?;
    Ok(())
}
