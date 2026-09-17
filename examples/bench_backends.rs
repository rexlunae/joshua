//! Cross-backend throughput benchmark: CPU vs OpenCL vs Vulkan.
//!
//! Runs the same LLM-shaped ops on a chosen device and reports ms/op and an
//! effective decode tokens/s.  The decode path of a quantized model is
//! dominated by `(1, H) · Wᵀ` matmuls against block-quantized weights, so
//! the quantized figures are the ones that predict per-token latency; the
//! f32 GEMM figures show raw compute.
//!
//! Usage:
//!   cargo run --release --features opencl,vulkan --example bench_backends -- \
//!       --device cpu|opencl|vulkan [--H 4096] [--seq 512] [--iters 50]
//!
//! Native kernels run by default on opencl/vulkan; `JOSHUA_OPENCL_NATIVE=0` /
//! `JOSHUA_VULKAN_NATIVE=0` measure the CPU round-trip path instead.

use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{Device, Module, Tensor};

fn main() -> anyhow::Result<()> {
    let mut device = "cpu".to_string();
    let mut h = 4096usize;
    let mut seq = 512usize;
    let mut iters = 50usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--device" => device = args.next().unwrap_or("cpu".into()),
            "--H" => h = args.next().and_then(|s| s.parse().ok()).unwrap_or(4096),
            "--seq" => seq = args.next().and_then(|s| s.parse().ok()).unwrap_or(512),
            "--iters" => iters = args.next().and_then(|s| s.parse().ok()).unwrap_or(50),
            _ => {}
        }
    }

    let dev = match device.as_str() {
        "cpu" => Device::Cpu,
        #[cfg(feature = "opencl")]
        "opencl" => match candle_core::OpenClDevice::new(0) {
            Ok(d) => Device::OpenCl(d),
            Err(e) => {
                eprintln!("opencl unavailable: {e}");
                return Ok(());
            }
        },
        #[cfg(feature = "vulkan")]
        "vulkan" => match candle_core::VulkanDevice::new(0) {
            Ok(d) => Device::Vulkan(d),
            Err(e) => {
                eprintln!("vulkan unavailable: {e}");
                return Ok(());
            }
        },
        // Without the matching feature, report it explicitly.
        #[cfg(not(feature = "opencl"))]
        "opencl" => {
            eprintln!("opencl backend not built (enable --features opencl)");
            return Ok(());
        }
        #[cfg(not(feature = "vulkan"))]
        "vulkan" => {
            eprintln!("vulkan backend not built (enable --features vulkan)");
            return Ok(());
        }
        other => {
            eprintln!("unknown device {other} (cpu|opencl|vulkan)");
            return Ok(());
        }
    };
    println!("backend: {device}  H={h} seq={seq} iters={iters}");
    #[cfg(feature = "opencl")]
    if dev.is_opencl() {
        println!("  (opencl native kernels: {})", candle_core::opencl_backend::native_enabled());
    }
    #[cfg(feature = "vulkan")]
    if dev.is_vulkan() {
        println!("  (vulkan native kernels: {})", candle_core::vulkan_backend::native_enabled());
    }

    // ---- Quantized matmuls: what a GGUF model actually runs. ----
    let wq_v: Vec<f32> = (0..h * h).map(|i| ((i % 11) as f32 - 5.0) * 0.25).collect();
    let wq = QTensor::quantize(&Tensor::from_vec(wq_v, (h, h), &Device::Cpu)?, GgmlDType::Q4K)?;
    let wq_bytes = wq.data()?.into_owned();
    let wq = QMatMul::from_qtensor(QTensor::new(
        QStorage::from_data(std::borrow::Cow::Borrowed(&wq_bytes), &dev, GgmlDType::Q4K)?,
        (h, h),
    )?)?;
    let xq_v: Vec<f32> = (0..h).map(|i| ((i % 17) as f32 - 8.0) * 0.5).collect();
    let xq = Tensor::from_vec(xq_v, (1, h), &dev)?;
    let _ = wq.forward(&xq)?;
    candle_core::Device::synchronize(&dev)?;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let c = wq.forward(&xq)?;
        candle_core::Device::synchronize(&dev)?;
        drop(c);
    }
    let qdec_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!(
        "  q4_k decode (1,{h})·W^T: {qdec_ms:.3} ms/op -> ~{:.1} t/s (ref: 6 matmuls/token), {:.1} GB/s of weights",
        1000.0 / (qdec_ms * 6.0),
        (h * h) as f64 * 0.5625 / (qdec_ms / 1000.0) / 1e9
    );
    let xp_v: Vec<f32> = (0..seq * h).map(|i| ((i % 13) as f32 - 6.0) * 0.4).collect();
    let xp = Tensor::from_vec(xp_v, (seq, h), &dev)?;
    let _ = wq.forward(&xp)?;
    candle_core::Device::synchronize(&dev)?;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let c = wq.forward(&xp)?;
        candle_core::Device::synchronize(&dev)?;
        drop(c);
    }
    let qpre_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!(
        "  q4_k prefill ({seq},{h})·W^T: {qpre_ms:.3} ms/op, {:.1} GFLOPS",
        2.0 * seq as f64 * h as f64 * h as f64 / (qpre_ms / 1000.0) / 1e9
    );

    // ---- Decode-shaped matmul: (1, H) @ (H, H) — the per-token cost. ----
    let a_v: Vec<f32> = (0..h).map(|i| ((i % 17) as f32 - 8.0) * 0.5).collect();
    let b_v: Vec<f32> = (0..h * h).map(|i| ((i % 11) as f32 - 5.0) * 0.25).collect();
    let a = Tensor::from_vec(a_v, (1, h), &dev)?;
    let b = Tensor::from_vec(b_v, (h, h), &dev)?;
    let _ = a.matmul(&b)?; // warm-up
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let c = a.matmul(&b)?;
        candle_core::Device::synchronize(&dev)?;
        drop(c);
    }
    let dec_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    // A decode step's dense cost ~ a few (1,H)@(H,H) matmuls; report t/s for a
    // reference token assuming ~6 such matmuls (attention QK/OV + MLP dense
    // projections), i.e. per_token_ms = dec_ms * 6.
    let dec_tps = 1000.0 / (dec_ms * 6.0);
    println!(
        "  decode (1,{h})@({h},{h}): {dec_ms:.3} ms/op -> ~{dec_tps:.1} t/s (ref: 6 dense matmuls/token)"
    );

    // ---- Prefill-shaped matmul: (seq, H) @ (H, H). ----
    let pa_v: Vec<f32> = (0..seq * h).map(|i| ((i % 13) as f32 - 6.0) * 0.4).collect();
    let pa = Tensor::from_vec(pa_v, (seq, h), &dev)?;
    let _ = pa.matmul(&b)?;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let c = pa.matmul(&b)?;
        candle_core::Device::synchronize(&dev)?;
        drop(c);
    }
    let pre_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let pre_gflops = 2.0 * seq as f64 * h as f64 * h as f64 / (pre_ms / 1000.0) / 1e9;
    println!(
        "  prefill ({seq},{h})@({h},{h}): {pre_ms:.3} ms/op, {pre_gflops:.1} GFLOPS"
    );

    // ---- Elementwise affine (1M elements). ----
    let x_v: Vec<f32> = (0..1_048_576).map(|i| i as f32 * 0.5).collect();
    let x = Tensor::from_vec(x_v, (1_048_576,), &dev)?;
    let _ = x.affine(1.5, -0.25)?;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let y = x.affine(1.5, -0.25)?;
        candle_core::Device::synchronize(&dev)?;
        drop(y);
    }
    let e_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!(
        "  affine 1M: {e_ms:.3} ms/op, {:.1} M elem/s",
        1048576.0 / (e_ms / 1000.0) / 1e6
    );

    Ok(())
}
