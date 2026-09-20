//! Decode-path quantized GEMV benchmark, SYCL vs OpenCL, on the same GPU
//! (Arc Pro B50) at DeepSeek 4 Flash's real expert shapes.
//!
//! The engine's decode streams ~2.3 GiB of IQ2_XXS expert weights per token
//! (6 experts × 3 matrices × ~60 layers), so the achievable decode t/s is
//! weight-streaming-bandwidth-bound: t/s ≈ GB/s / 2.3.  This benchmark times
//! each backend's canonical decode launcher (`k_qgemv_mr` for IQ2/Q2K) at the
//! gate shape [N=2048, K=4096] and the down shape [N=4096, K=2048], m=1.
//!
//!   cargo run --release --features opencl --example quant_bench


use candle_core::quantized::GgmlDType;

const QT_IQ2_XXS: i32 = 16;
const IQ2_XXS_TSIZE: i32 = 66;
const IQ2_XXS_QK: usize = 256;

#[cfg(all(feature = "sycl", target_os = "linux"))]
fn sycl_bandwidth(n: usize, k: usize, iters: usize) -> Option<f64> {
    let dev = match candle_core::sycl_backend::SyclDevice::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP sycl: {e}");
            return None;
        }
    };
    let wbytes = n * (k / IQ2_XXS_QK) * IQ2_XXS_TSIZE as usize;
    let wb = dev.alloc(wbytes).unwrap();
    // Data content does not affect GEMV streaming bandwidth; fill with a
    // repeating byte so the dequant path stays on its normal branch.
    let fill: Vec<u8> = (0..wbytes).map(|i| (i % 251) as u8).collect();
    dev.write(wb, 0, &fill).unwrap();
    let xb = dev.alloc(k * 4).unwrap();
    let x: Vec<u8> = (0..k * 4).map(|i| (i % 7) as u8).collect();
    dev.write(xb, 0, &x).unwrap();
    let ob = dev.alloc(n * 4).unwrap();

    // warmup
    for _ in 0..5 {
        dev.run_qgemv_routed(QT_IQ2_XXS, IQ2_XXS_QK as i32, IQ2_XXS_TSIZE, true, xb, wb, ob, n, k, 0, 0, 1).unwrap();
    }
    dev.finish().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        dev.run_qgemv_routed(QT_IQ2_XXS, IQ2_XXS_QK as i32, IQ2_XXS_TSIZE, true, xb, wb, ob, n, k, 0, 0, 1).unwrap();
    }
    dev.finish().unwrap();
    let el = t0.elapsed().as_secs_f64();
    let gbs = (wbytes as f64) * (iters as f64) / el / 1e9;
    println!("  sycl  k_qgemv_mr [{n}, {k}]: {gbs:.1} GB/s ({iters} iters, {wbytes} B/launch)");
    Some(gbs)
}

#[cfg(not(all(feature = "sycl", target_os = "linux")))]
fn sycl_bandwidth(_n: usize, _k: usize, _iters: usize) -> Option<f64> {
    eprintln!("SKIP sycl: needs --features sycl on linux");
    None
}

#[cfg(feature = "opencl")]
fn opencl_bandwidth(n: usize, k: usize, iters: usize) -> Option<f64> {
    use candle_core::opencl_backend::{kernels as ocl, OpenClDevice};
    let dev = match OpenClDevice::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP opencl: {e}");
            return None;
        }
    };
    let wbytes = n * (k / IQ2_XXS_QK) * IQ2_XXS_TSIZE as usize;
    // Buffers: uninitialized device memory is fine for a bandwidth benchmark
    // (values do not change the streaming cost).
    // F32 storage sized to hold the same bytes (U8 alloc isn't a supported
    // device-storage path); the kernel streams raw bytes within it.
    let wb = dev.alloc(candle_core::DType::F32, wbytes / 4).unwrap().buffer;
    let xb = dev.alloc(candle_core::DType::F32, k).unwrap().buffer;
    let ob = dev.alloc(candle_core::DType::F32, n).unwrap().buffer;
    let ctx = dev.ctx();
    for _ in 0..5 {
        ocl::run_qgemv(&ctx, GgmlDType::Iq2Xxs, xb, wb, ob, 1, n, k, 0, 0).unwrap();
    }
    dev.synchronize().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        ocl::run_qgemv(&ctx, GgmlDType::Iq2Xxs, xb, wb, ob, 1, n, k, 0, 0).unwrap();
    }
    dev.synchronize().unwrap();
    let el = t0.elapsed().as_secs_f64();
    let gbs = (wbytes as f64) * (iters as f64) / el / 1e9;
    println!("  opencl k_qgemv_mr [{n}, {k}]: {gbs:.1} GB/s ({iters} iters, {wbytes} B/launch)");
    Some(gbs)
}

#[cfg(not(feature = "opencl"))]
fn opencl_bandwidth(_n: usize, _k: usize, _iters: usize) -> Option<f64> {
    eprintln!("SKIP opencl: needs --features opencl");
    None
}

fn main() {
    println!("deepseek4-flash expert shapes, IQ2_XXS, m=1 (decode):");
    let mut s = (0f64, 0f64);
    let mut o = (0f64, 0f64);
    for (n, k, label) in [(2048usize, 4096usize, "gate/up"), (4096, 2048, "down")] {
        println!("{label} [{n}, {k}]:");
        if let Some(b) = sycl_bandwidth(n, k, 60) { s.0 += b; }
        if let Some(b) = opencl_bandwidth(n, k, 60) { o.0 += b; }
    }
    println!("aggregate streaming bandwidth: sycl {:.1} GB/s, opencl {:.1} GB/s", s.0, o.0);
    // The engine streams ~2.3 GiB of expert weights per decode token.
    let expert_gib = 2.3;
    println!("projected expert-streaming decode: sycl ≈ {:.2} t/s, opencl ≈ {:.2} t/s",
        s.0 / expert_gib, o.0 / expert_gib);
    println!("(OpenCL engine measured end-to-end decode: 0.8 t/s on this model)");
}
