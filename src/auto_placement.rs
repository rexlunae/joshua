//! Startup micro-benchmark that picks the dense-set placement automatically.
//!
//! The right place to run the dense set depends on the *actual hardware*, so an
//! `Auto` dense-placement (the default) probes the accelerator once, briefly,
//! before loading the model.  It measures the GEMM throughput of CPU-BLAS
//! versus the requested compute device on the two shapes that dominate a
//! transformer:
//!
//! * **decode** — `(1, H) @ (H, H)`, one token's dense projections; and
//! * **prefill** — `(B, H) @ (H, H)`, a chunk of prompt tokens.
//!
//! If the device's GEMM is faster than CPU-BLAS by a usable margin it is worth
//! offloading the dense set; if not (a weak iGPU or a software OpenCL/Vulkan
//! path — e.g. Renoir runs dense OpenCL at ~21 GFLOPS vs ~384 GFLOPS of
//! 16-core BLAS), the dense set should stay on CPU-BLAS.  Because the rule is a
//! measured *ratio* it adapts automatically to every kind of card: a big
//! discrete GPU scores high and keeps dense on-device, a weak iGPU scores low
//! and keeps it on CPU, with no operator guesswork.
//!
//! The probe is small and bounded (see [`BENCH_H`]/[`BENCH_B`]) so startup stays
//! fast; the CPU-vs-device ratio is stable across widths even when absolute
//! GFLOPS is not.

use candle_core::{Device, Tensor};

use crate::placement::{DensePlacement, ResolvedDense};

/// Reference hidden width for the probe GEMMs.  Kept moderate so the probe
/// finishes in well under a second even on a slow integrated GPU.
const BENCH_H: usize = 1024;
/// Prompt-chunk length for the prefill probe (small on purpose so the probe
/// never trips a weak GPU's ring-timeout watchdog).
const BENCH_B: usize = 32;
/// How many timed matmuls per shape per device.
const BENCH_ITERS: usize = 3;
/// Total wall-clock budget for the whole probe (all shapes, both devices).
const BENCH_DEADLINE: std::time::Duration = std::time::Duration::from_secs(8);

/// Measured dense GEMM throughput of CPU-BLAS vs an accelerator.
#[derive(Debug, Clone, Copy)]
pub struct DenseBench {
    /// Accelerator decode `(1,H)` GFLOPS (0 when the device could not be probed).
    pub dev_decode_gflops: f64,
    /// Accelerator prefill `(B,H)` GFLOPS.
    pub dev_prefill_gflops: f64,
    /// CPU-BLAS decode `(1,H)` GFLOPS.
    pub cpu_decode_gflops: f64,
    /// CPU-BLAS prefill `(B,H)` GFLOPS.
    pub cpu_prefill_gflops: f64,
}

impl DenseBench {
    /// Device-to-CPU speedup on the decode shape (the common single-token path).
    pub fn decode_speedup(&self) -> f64 {
        if self.cpu_decode_gflops > 0.0 {
            self.dev_decode_gflops / self.cpu_decode_gflops
        } else {
            0.0
        }
    }

    /// Device-to-CPU speedup on the prefill shape.
    pub fn prefill_speedup(&self) -> f64 {
        if self.cpu_prefill_gflops > 0.0 {
            self.dev_prefill_gflops / self.cpu_prefill_gflops
        } else {
            0.0
        }
    }

    /// Best of the two speedups — used for the placement decision.
    fn best_speedup(&self) -> f64 {
        self.decode_speedup().max(self.prefill_speedup())
    }
}

/// Measure GEMM GFLOPS for shape `(m, H) @ (H, H)` on `device`, bailing with
/// `None` (as "unmeasurable") if `deadline` passes — a cooperative guard so a
/// wedged GPU cannot hang startup forever.
fn measure_gemm(device: &Device, m: usize, deadline: std::time::Instant) -> Option<f64> {
    let h = BENCH_H;
    let a: Vec<f32> = (0..m * h).map(|i| ((i % 17) as f32 - 8.0) * 0.5).collect();
    let b: Vec<f32> = (0..h * h).map(|i| ((i % 11) as f32 - 5.0) * 0.25).collect();
    let a = Tensor::from_vec(a, (m, h), device).ok()?;
    let b = Tensor::from_vec(b, (h, h), device).ok()?;

    // Warm-up (best-effort; a failure here means "can't probe", not 0 GFLOPS).
    match a.matmul(&b) {
        Ok(_) => {}
        Err(_) => return None,
    }
    let _ = candle_core::Device::synchronize(device).ok();
    let mut samples = 0usize;
    let mut total_secs = 0.0f64;
    for _ in 0..BENCH_ITERS {
        if std::time::Instant::now() >= deadline {
            break;
        }
        let start = std::time::Instant::now();
        let c = match a.matmul(&b) {
            Ok(c) => c,
            Err(_) => break,
        };
        if candle_core::Device::synchronize(device).is_err() {
            break;
        }
        total_secs += start.elapsed().as_secs_f64();
        samples += 1;
        drop(c);
    }
    if samples == 0 || total_secs <= 0.0 {
        return None;
    }
    let secs = total_secs / samples as f64;
    Some(2.0 * m as f64 * h as f64 * h as f64 / secs / 1e9)
}

/// Run the quick probe: time CPU-BLAS and the accelerator on decode + prefill
/// shapes and return the measured throughput.  A device that cannot be probed
/// (fails or runs past the deadline) yields `0.0` for its figures, which the
/// decision logic treats as "no faster than CPU".
pub fn benchmark(device: &Device) -> DenseBench {
    let cpu = Device::Cpu;
    let deadline = std::time::Instant::now() + BENCH_DEADLINE;
    let (cpu_dec, cpu_pre) = if device.is_cpu() {
        (0.0, 0.0)
    } else {
        (
            measure_gemm(&cpu, 1, deadline).unwrap_or(0.0),
            measure_gemm(&cpu, BENCH_B, deadline).unwrap_or(0.0),
        )
    };
    let (dev_dec, dev_pre) = (
        measure_gemm(device, 1, deadline).unwrap_or(0.0),
        measure_gemm(device, BENCH_B, deadline).unwrap_or(0.0),
    );
    DenseBench {
        dev_decode_gflops: dev_dec,
        dev_prefill_gflops: dev_pre,
        cpu_decode_gflops: cpu_dec,
        cpu_prefill_gflops: cpu_pre,
    }
}

/// Pure decision: given the measured bench and the operator's request, resolve
/// where the dense set should live.  Unit-testable (benchmarking is the only
/// part that touches a device).
///
/// * An explicit `Device` / `Cpu` request wins (the operator overrides Auto).
/// * `Auto` keeps dense on the accelerator when it is at least
///   [`MIN_DEVICE_SPEEDUP`] faster than CPU-BLAS on either dominant shape, and
///   moves it to CPU-BLAS otherwise.  An unmeasurable device (0 GFLOPS) counts
///   as "no faster" and dense goes to CPU.
/// * On the CPU device, `Auto` is CPU.
pub fn recommend_dense(requested: DensePlacement, bench: &DenseBench, is_cpu_device: bool) -> ResolvedDense {
    match requested {
        DensePlacement::Device => ResolvedDense::Device,
        DensePlacement::Cpu => ResolvedDense::Cpu,
        DensePlacement::Auto => {
            if is_cpu_device {
                return ResolvedDense::Cpu;
            }
            if bench.best_speedup() >= MIN_DEVICE_SPEEDUP {
                ResolvedDense::Device
            } else {
                ResolvedDense::Cpu
            }
        }
    }
}

/// The minimum device-vs-CPU GEMM speedup for `Auto` to keep the dense set on
/// the accelerator.  Below this the bus + driver overhead of offloading is not
/// worth it and CPU-BLAS is the safer choice.  (1.5x leaves headroom for the
/// copy cost on each layer; a real GPU scores far above this, a weak iGPU far
/// below.)
pub const MIN_DEVICE_SPEEDUP: f64 = 1.5;

/// Whether to run the probe at startup.  Set `JOSHUA_SKIP_PLACEMENT_BENCH=1` to
/// skip it for fast startup; `Auto` then falls back to the historical default
/// (dense on the device) without measuring.
pub fn probe_enabled() -> bool {
    !matches!(
        std::env::var("JOSHUA_SKIP_PLACEMENT_BENCH"),
        Ok(s) if !s.is_empty() && s != "0"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bench(dev_dec: f64, dev_pre: f64, cpu_dec: f64, cpu_pre: f64) -> DenseBench {
        DenseBench {
            dev_decode_gflops: dev_dec,
            dev_prefill_gflops: dev_pre,
            cpu_decode_gflops: cpu_dec,
            cpu_prefill_gflops: cpu_pre,
        }
    }

    #[test]
    fn auto_keeps_dense_when_device_is_much_faster() {
        // A8070-class device: ~30x faster decode, ~15x faster prefill.
        let b = bench(1200.0, 900.0, 40.0, 60.0);
        assert_eq!(recommend_dense(DensePlacement::Auto, &b, false), ResolvedDense::Device);
    }

    #[test]
    fn auto_moves_dense_to_cpu_on_weak_igpu() {
        // Renoir-class device: far slower than CPU-BLAS.
        let b = bench(5.0, 8.0, 40.0, 120.0);
        assert_eq!(recommend_dense(DensePlacement::Auto, &b, false), ResolvedDense::Cpu);
    }

    #[test]
    fn auto_is_cpu_on_cpu_device() {
        let b = bench(0.0, 0.0, 40.0, 120.0);
        assert_eq!(recommend_dense(DensePlacement::Auto, &b, true), ResolvedDense::Cpu);
    }

    #[test]
    fn unmeasurable_device_falls_back_to_cpu() {
        let b = bench(0.0, 0.0, 40.0, 120.0);
        assert_eq!(recommend_dense(DensePlacement::Auto, &b, false), ResolvedDense::Cpu);
    }

    #[test]
    fn explicit_requests_win() {
        let weak = bench(5.0, 8.0, 40.0, 120.0);
        // Operator can still force the device even when Auto would pick CPU.
        assert_eq!(recommend_dense(DensePlacement::Device, &weak, false), ResolvedDense::Device);
        let strong = bench(1200.0, 900.0, 40.0, 60.0);
        assert_eq!(recommend_dense(DensePlacement::Cpu, &strong, false), ResolvedDense::Cpu);
    }

    #[test]
    fn boundary_is_min_speedup() {
        // Exactly at the threshold keeps dense on the device.
        let b = bench(MIN_DEVICE_SPEEDUP * 40.0, 1.0, 40.0, 120.0);
        assert_eq!(recommend_dense(DensePlacement::Auto, &b, false), ResolvedDense::Device);
        // Just under the threshold on every shape moves dense to CPU.
        let b = bench(MIN_DEVICE_SPEEDUP * 40.0 - 1.0, 1.0, 40.0, 120.0);
        assert_eq!(recommend_dense(DensePlacement::Auto, &b, false), ResolvedDense::Cpu);
    }
}