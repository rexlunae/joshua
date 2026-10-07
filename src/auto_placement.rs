//! Startup micro-benchmark that picks the dense-set placement automatically.
//!
//! The right place to run the dense set depends on the *actual hardware*, so an
//! `Auto` dense-placement (the default) probes the accelerator once, briefly,
//! before loading the model.  It measures the throughput of the CPU versus the
//! requested compute device on the operation that dominates a quantized
//! transformer — a block-quantized weight matrix (Q4_K here, the most common
//! GGUF dtype) applied to f32 activations — on the two shapes that matter:
//!
//! * **decode** — `(1, H) · Wᵀ`, one token's dense projections; and
//! * **prefill** — `(B, H) · Wᵀ`, a chunk of prompt tokens.
//!
//! Probing the quantized matmul rather than an f32 GEMM is deliberate: it is
//! what the loaders actually run (the CPU's SIMD `k_quants` kernels against
//! the device's dequantize-in-kernel GEMV / tiled GEMM), and on an iGPU that
//! shares DRAM with the CPU the difference between streaming 4-bit blocks and
//! streaming f32 weights *is* the whole result.
//!
//! If the device is faster than the CPU by a usable margin it is worth
//! offloading the dense set; if not (a weak iGPU or a software OpenCL/Vulkan
//! path), the dense set should stay on the CPU.  Because the rule is a
//! measured *ratio* it adapts automatically to every kind of card: a big
//! discrete GPU scores high and keeps dense on-device, a weak iGPU scores low
//! and keeps it on CPU, with no operator guesswork.
//!
//! The probe is bounded (see [`BENCH_H`]/[`BENCH_B`] and the deadline) so
//! startup stays fast, but it runs at a real projection width over more
//! weight than a CPU cache holds: the ratio is *not* stable across widths.
//! A small probe matrix lives in the CPU's cache while a discrete GPU spends
//! the whole GEMV on launch latency, which inverted the decision on an Arc
//! Pro B50 (#164).

use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{Device, Module, Tensor};

use crate::placement::{DensePlacement, ResolvedDense};

/// Reference hidden width for the probe GEMMs: a real dense projection
/// (4096 is the hidden size of most 7B-class and the DeepSeek-V4 models).
/// At 1024 the decode GEMV is ~2 MFLOP, which a discrete GPU finishes faster
/// than one launch-and-sync round trip and which a desktop CPU serves from
/// its L2: the probe then measured launch latency against cache bandwidth
/// and read 0.2-0.3x for an Arc B50 against a Ryzen 9950X (#164).
const BENCH_H: usize = 4096;
/// Distinct weight matrices the decode probe cycles through, launched back to
/// back with one sync at the end, as a layer's projections are.  At ~9 MiB of
/// Q4_K each, eight (~72 MiB) exceed any desktop CPU's last-level cache, so
/// the CPU streams weights from DRAM as it does on a real model, and the
/// device's per-launch overhead is amortized the way a decode step does.
const BENCH_DECODE_MATS: usize = 8;
/// Distinct matrices for the prefill probe.  A prompt chunk reuses each weight
/// across its rows, so the prefill shape is compute-bound and one matrix
/// measures it; more would only lengthen the probe on a weak GPU.
const BENCH_PREFILL_MATS: usize = 1;
/// Prompt-chunk length for the prefill probe (small on purpose so the probe
/// never trips a weak GPU's ring-timeout watchdog).
const BENCH_B: usize = 32;
/// How many timed passes per shape per device.
const BENCH_ITERS: usize = 3;
/// Timed passes stop early once one shape has spent this long, so a slow
/// device gives one or two samples instead of `BENCH_ITERS`.
const BENCH_SHAPE_BUDGET: std::time::Duration = std::time::Duration::from_millis(250);
/// Total wall-clock budget for the whole probe (all shapes, both devices).
const BENCH_DEADLINE: std::time::Duration = std::time::Duration::from_secs(8);

/// Measured quantized-matmul throughput of the CPU vs an accelerator.
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

    /// The speedup that decides dense placement: the **decode** ratio.
    ///
    /// Decode is the phase where keeping dense on the device costs the most —
    /// one host<->device round trip per layer, paid on the critical path of
    /// every generated token. Prefill feeds many tokens per copy, so a device
    /// that is excellent at prefill and poor at decode still amortises the
    /// transfers well there.
    ///
    /// Taking the best of the two (the old behaviour) let a prefill-only win
    /// qualify dense for the device. Measured on an Intel Arc Pro B50 with
    /// DeepSeek-V4-Flash (experts in host RAM): the device probed at
    /// **0.46x** CPU-BLAS on decode but **1.68x** on prefill, so `max()` read
    /// 1.68 and placed dense on the GPU — making decode *slower* than leaving
    /// dense on the CPU (1.7 vs 2.0 tok/s). Decode is the tighter constraint,
    /// so it is the one that gates.
    fn best_speedup(&self) -> f64 {
        self.decode_speedup()
    }
}

/// Rows of the probe weight that are actually quantized; the rest of the
/// matrix repeats them.  Quantizing a full 4096x4096 Q4_K matrix takes a
/// large fraction of a second, which startup should not pay, and the matmul
/// cost does not depend on the values.
const PROBE_QUANT_ROWS: usize = 64;

/// The probe weight `[BENCH_H, BENCH_H]` as raw Q4_K blocks, starting at a
/// 2-byte boundary (the block's `f16` alignment, which the CPU storage
/// checks).  Returns the backing buffer and the range of the blocks in it.
fn probe_weight_bytes() -> Option<(Vec<u8>, std::ops::Range<usize>)> {
    let h = BENCH_H;
    let w: Vec<f32> = (0..PROBE_QUANT_ROWS * h)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.25)
        .collect();
    let w = Tensor::from_vec(w, (PROBE_QUANT_ROWS, h), &Device::Cpu).ok()?;
    let slab = QTensor::quantize(&w, GgmlDType::Q4K).ok()?;
    let slab = slab.data().ok()?;
    let len = slab.len() * (h / PROBE_QUANT_ROWS);
    let mut buf = vec![0u8; len + 1];
    let off = buf.as_ptr() as usize % 2;
    for chunk in buf[off..off + len].chunks_exact_mut(slab.len()) {
        chunk.copy_from_slice(&slab);
    }
    Some((buf, off..off + len))
}

/// `count` separate copies of the probe weight as `QMatMul`s on `device`.
fn probe_matmuls(device: &Device, weight: &[u8], count: usize) -> Option<Vec<QMatMul>> {
    (0..count)
        .map(|_| {
            let storage =
                QStorage::from_data(std::borrow::Cow::Borrowed(weight), device, GgmlDType::Q4K)
                    .ok()?;
            let q = QTensor::new(storage, (BENCH_H, BENCH_H)).ok()?;
            QMatMul::from_qtensor(q).ok()
        })
        .collect()
}

/// Measure the quantized-matmul throughput (in f32-equivalent GFLOPS) for
/// shape `(m, H) · Wᵀ` on `device`, over `mats` distinct weights launched back
/// to back with one sync per pass, bailing with `None` (as "unmeasurable") if
/// `deadline` passes — a cooperative guard so a wedged GPU cannot hang startup
/// forever.
fn measure_gemm(
    device: &Device,
    weight: &[u8],
    m: usize,
    mats: usize,
    deadline: std::time::Instant,
) -> Option<f64> {
    let h = BENCH_H;
    let a: Vec<f32> = (0..m * h).map(|i| ((i % 17) as f32 - 8.0) * 0.5).collect();
    let a = Tensor::from_vec(a, (m, h), device).ok()?;
    let ws = probe_matmuls(device, weight, mats)?;

    // Warm-up (best-effort; a failure here means "can't probe", not 0 GFLOPS).
    // On a device this also compiles the kernels, which must not be timed,
    // and faults every weight page in on the CPU.
    for w in &ws {
        w.forward(&a).ok()?;
    }
    let _ = candle_core::Device::synchronize(device).ok();
    let mut samples = 0usize;
    let mut total_secs = 0.0f64;
    'passes: for _ in 0..BENCH_ITERS {
        if std::time::Instant::now() >= deadline
            || (samples > 0 && total_secs >= BENCH_SHAPE_BUDGET.as_secs_f64())
        {
            break;
        }
        let start = std::time::Instant::now();
        let mut outs = Vec::with_capacity(ws.len());
        for w in &ws {
            match w.forward(&a) {
                Ok(c) => outs.push(c),
                Err(_) => break 'passes,
            }
        }
        if candle_core::Device::synchronize(device).is_err() {
            break;
        }
        total_secs += start.elapsed().as_secs_f64();
        samples += 1;
        drop(outs);
    }
    if samples == 0 || total_secs <= 0.0 {
        return None;
    }
    let secs = total_secs / samples as f64;
    Some(2.0 * (m * mats) as f64 * h as f64 * h as f64 / secs / 1e9)
}

/// Run the quick probe: time the CPU and the accelerator on decode + prefill
/// shapes and return the measured throughput.  A device that cannot be probed
/// (fails or runs past the deadline) yields `0.0` for its figures, which the
/// decision logic treats as "no faster than CPU".
pub fn benchmark(device: &Device) -> DenseBench {
    let cpu = Device::Cpu;
    let deadline = std::time::Instant::now() + BENCH_DEADLINE;
    let Some((buf, range)) = probe_weight_bytes() else {
        return DenseBench {
            dev_decode_gflops: 0.0,
            dev_prefill_gflops: 0.0,
            cpu_decode_gflops: 0.0,
            cpu_prefill_gflops: 0.0,
        };
    };
    let weight = &buf[range];
    let measure = |dev: &Device, m: usize, mats: usize| {
        measure_gemm(dev, weight, m, mats, deadline).unwrap_or(0.0)
    };
    let (cpu_dec, cpu_pre) = if device.is_cpu() {
        (0.0, 0.0)
    } else {
        (
            measure(&cpu, 1, BENCH_DECODE_MATS),
            measure(&cpu, BENCH_B, BENCH_PREFILL_MATS),
        )
    };
    let (dev_dec, dev_pre) = (
        measure(device, 1, BENCH_DECODE_MATS),
        measure(device, BENCH_B, BENCH_PREFILL_MATS),
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
///   [`MIN_DEVICE_SPEEDUP`] faster than CPU-BLAS on the **decode** shape, and
///   moves it to CPU-BLAS otherwise.  Decode is the gating phase because it is
///   where the per-layer host<->device round trip sits on the critical path; a
///   device that only wins at prefill does not qualify (see [`DenseBench`]).
///   An unmeasurable device (0 GFLOPS) counts as "no faster" and dense goes to
///   CPU.
/// * On the CPU device, `Auto` is CPU.
pub fn recommend_dense(
    requested: DensePlacement,
    bench: &DenseBench,
    is_cpu_device: bool,
) -> ResolvedDense {
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
    fn probe_weight_is_a_full_aligned_q4k_matrix() {
        let (buf, range) = probe_weight_bytes().expect("probe weight");
        assert_eq!(range.start % 2, 0, "Q4_K blocks need f16 alignment");
        assert_eq!(
            range.len(),
            BENCH_H * BENCH_H / 256 * GgmlDType::Q4K.type_size()
        );
        // The CPU storage accepts it (it asserts the alignment) and every
        // copy is a separate allocation, so the decode probe really streams
        // `BENCH_DECODE_MATS` matrices' worth of weight.
        let ws = probe_matmuls(&Device::Cpu, &buf[range], 2).expect("cpu matmuls");
        assert_eq!(ws.len(), 2);
    }

    #[test]
    fn cpu_probe_measures_both_shapes_within_the_deadline() {
        let (buf, range) = probe_weight_bytes().expect("probe weight");
        let start = std::time::Instant::now();
        let deadline = start + BENCH_DEADLINE;
        let dec = measure_gemm(
            &Device::Cpu,
            &buf[range.clone()],
            1,
            BENCH_DECODE_MATS,
            deadline,
        );
        let pre = measure_gemm(
            &Device::Cpu,
            &buf[range],
            BENCH_B,
            BENCH_PREFILL_MATS,
            deadline,
        );
        assert!(dec.is_some_and(|g| g.is_finite() && g > 0.0), "{dec:?}");
        assert!(pre.is_some_and(|g| g.is_finite() && g > 0.0), "{pre:?}");
        assert!(start.elapsed() < BENCH_DEADLINE);
    }

    #[test]
    fn auto_keeps_dense_when_device_is_much_faster() {
        // A8070-class device: ~30x faster decode, ~15x faster prefill.
        let b = bench(1200.0, 900.0, 40.0, 60.0);
        assert_eq!(
            recommend_dense(DensePlacement::Auto, &b, false),
            ResolvedDense::Device
        );
    }

    #[test]
    fn auto_moves_dense_to_cpu_on_weak_igpu() {
        // Renoir-class device: far slower than CPU-BLAS.
        let b = bench(5.0, 8.0, 40.0, 120.0);
        assert_eq!(
            recommend_dense(DensePlacement::Auto, &b, false),
            ResolvedDense::Cpu
        );
    }

    #[test]
    fn auto_is_cpu_on_cpu_device() {
        let b = bench(0.0, 0.0, 40.0, 120.0);
        assert_eq!(
            recommend_dense(DensePlacement::Auto, &b, true),
            ResolvedDense::Cpu
        );
    }

    #[test]
    fn unmeasurable_device_falls_back_to_cpu() {
        let b = bench(0.0, 0.0, 40.0, 120.0);
        assert_eq!(
            recommend_dense(DensePlacement::Auto, &b, false),
            ResolvedDense::Cpu
        );
    }

    #[test]
    fn prefill_only_win_does_not_qualify_dense_for_the_device() {
        // The measured Intel Arc Pro B50 profile: prefill is clearly faster
        // than CPU-BLAS, decode is clearly slower. A prefill-only win used to
        // place dense on the GPU and cost decode throughput.
        let b = bench(18.4, 235.4, 40.0, 140.1); // 0.46x decode, 1.68x prefill
        assert!(
            b.prefill_speedup() >= MIN_DEVICE_SPEEDUP,
            "fixture must be a prefill win, else it tests nothing"
        );
        assert!(b.decode_speedup() < MIN_DEVICE_SPEEDUP);
        assert_eq!(
            recommend_dense(DensePlacement::Auto, &b, false),
            ResolvedDense::Cpu
        );
    }

    #[test]
    fn decode_win_still_qualifies_dense_for_the_device() {
        // The complementary case: fast at decode clears the bar on its own,
        // even with a prefill ratio below the threshold.
        let b = bench(200.0, 60.0, 40.0, 120.0); // 5.0x decode, 0.5x prefill
        assert_eq!(
            recommend_dense(DensePlacement::Auto, &b, false),
            ResolvedDense::Device
        );
    }

    #[test]
    fn explicit_requests_win() {
        let weak = bench(5.0, 8.0, 40.0, 120.0);
        // Operator can still force the device even when Auto would pick CPU.
        assert_eq!(
            recommend_dense(DensePlacement::Device, &weak, false),
            ResolvedDense::Device
        );
        let strong = bench(1200.0, 900.0, 40.0, 60.0);
        assert_eq!(
            recommend_dense(DensePlacement::Cpu, &strong, false),
            ResolvedDense::Cpu
        );
    }

    #[test]
    fn boundary_is_min_speedup() {
        // Exactly at the threshold keeps dense on the device.
        let b = bench(MIN_DEVICE_SPEEDUP * 40.0, 1.0, 40.0, 120.0);
        assert_eq!(
            recommend_dense(DensePlacement::Auto, &b, false),
            ResolvedDense::Device
        );
        // Just under the threshold on every shape moves dense to CPU.
        let b = bench(MIN_DEVICE_SPEEDUP * 40.0 - 1.0, 1.0, 40.0, 120.0);
        assert_eq!(
            recommend_dense(DensePlacement::Auto, &b, false),
            ResolvedDense::Cpu
        );
    }
}
