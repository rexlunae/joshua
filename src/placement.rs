//! Automatic placement: sizing and tuning decisions for the hot-expert cache.
//!
//! Phase 5 of the device expert cache design.  This module is **pure math** —
//! no I/O, no device calls — so it is unit-testable with synthetic inputs.
//! The device backend (CPU now, a CUDA slot cache later) supplies measured
//! inputs (free memory, per-expert bytes, bus bandwidth, routing hit ratio)
//! through the load path; the functions below turn them into a cache budget.
//!
//! Two decisions live here:
//!
//! 1. **Static sizing** ([`expert_budget_for_memory`]): how many experts the
//!    available memory can hold, leaving headroom for the KV cache and the OS.
//! 2. **Adaptive tuning** ([`adaptive_budget`], FreeToken's "bandwidth-adaptive
//!    execution"): grow the cache while routing misses saturate the bus, shrink
//!    it once misses are rare (freeing memory for the KV cache / dense set).

/// Fraction of the memory above `headroom` the expert cache may consume.
pub const CACHE_MEMORY_FRACTION: f64 = 0.5;

/// Hit ratio below which a growing cache is worth it (misses are frequent).
pub const ADAPTIVE_LOW_HIT_RATIO: f64 = 0.80;

/// Hit ratio above which the cache is oversized and should shrink.
pub const ADAPTIVE_HIGH_HIT_RATIO: f64 = 0.98;

/// Static sizing: how many experts fit in `free_bytes`, keeping
/// `headroom_bytes` for the KV cache and the OS and capping at
/// `max_experts` (the backend's capacity).
pub fn expert_budget_for_memory(
    free_bytes: u64,
    expert_bytes: u64,
    max_experts: usize,
    headroom_bytes: u64,
) -> usize {
    if expert_bytes == 0 {
        return 0;
    }
    let usable = (free_bytes.saturating_sub(headroom_bytes)) as f64 * CACHE_MEMORY_FRACTION;
    let n = (usable / expert_bytes as f64).floor();
    if !n.is_finite() || n <= 0.0 {
        return 0;
    }
    (n as u64).min(max_experts as u64) as usize
}

/// Adaptive tuning (bandwidth-adaptive execution).
///
/// - If the routing hit ratio is low **and** the per-step miss traffic would
///   occupy a meaningful slice of the bus, grow the cache (up to
///   `max_budget`): the bus is the limiter and more residency helps.
/// - If the hit ratio is high, shrink (down to `min_budget`): misses are
///   rare, so resident-but-cold experts waste memory that the KV cache or the
///   dense set could use.
/// - Otherwise hold.
///
/// `bus_bytes_per_sec == 0` means "no bus" (pure CPU): never grow on misses,
/// because there is no transfer bottleneck to relieve.
pub fn adaptive_budget(
    current: usize,
    hit_ratio: f64,
    miss_bytes_per_step: f64,
    bus_bytes_per_sec: f64,
    min_budget: usize,
    max_budget: usize,
) -> usize {
    if max_budget == 0 {
        return 0;
    }
    let current = current.clamp(min_budget, max_budget);
    if hit_ratio < ADAPTIVE_LOW_HIT_RATIO {
        // The bus is the bottleneck only if there is a bus *and* the misses
        // would take a non-trivial slice of a step's budget (1 ms of bus
        // time).  With no bus (`bus_bytes_per_sec == 0`) there is no transfer
        // bottleneck to relieve, so a low hit ratio alone never grows.
        let has_bus = bus_bytes_per_sec > 0.0;
        let bus_time = if has_bus {
            miss_bytes_per_step / bus_bytes_per_sec
        } else {
            0.0
        };
        if has_bus && bus_time > 0.001 {
            // saturating_mul: `current` can exceed usize::MAX / 2 when the
            // cap is huge; a plain `* 2` would panic in debug builds and
            // wrap below min_budget in release.
            return current.saturating_mul(2).min(max_budget);
        }
        return current;
    }
    if hit_ratio > ADAPTIVE_HIGH_HIT_RATIO {
        return (current / 2).max(min_budget);
    }
    current
}

/// Budget (in **miBytes** of the expert device) for the bounded VRAM expert
/// cache (#62): the part of the device's free memory left after the dense set,
/// the static placement headroom and a KV reserve.  The device form of a routed
/// expert is larger than its on-disk quantized bytes (e.g. Q4_K → as uploaded),
/// so callers pass the per-expert *uploaded* byte size; `expert_bytes` here is
/// that uploaded figure.
///
/// Returns a byte budget; 0 when `free_bytes` leaves nothing after the
/// reservations.  The engine divides this by the per-expert upload size to get a
/// slot count (and passes it to `--vram-expert-cache auto`).
pub fn device_expert_cache_bytes(
    free_bytes: u64,
    dense_upper_bytes: u64,
    headroom_bytes: u64,
    kv_reserve_bytes: u64,
) -> u64 {
    // The entire leftover of the device budget belongs to the cache (the 0.5
    // RAM fraction is the host-cache heuristic; the device budget already
    // reserved dense + headroom + KV, so there is nothing left to discount).
    free_bytes
        .saturating_sub(dense_upper_bytes)
        .saturating_sub(headroom_bytes)
        .saturating_sub(kv_reserve_bytes)
}

/// Like [`device_expert_cache_bytes`] but returns a slot count (experts) that
/// fit in the budget.  `uploaded_expert_bytes` is the device form's per-expert
/// upload size; `max_experts` caps it.
pub fn device_expert_slots(
    free_bytes: u64,
    dense_upper_bytes: u64,
    headroom_bytes: u64,
    kv_reserve_bytes: u64,
    uploaded_expert_bytes: u64,
    max_experts: usize,
) -> usize {
    if uploaded_expert_bytes == 0 {
        return 0;
    }
    let budget = device_expert_cache_bytes(free_bytes, dense_upper_bytes, headroom_bytes, kv_reserve_bytes);
    (budget / uploaded_expert_bytes).min(max_experts as u64) as usize
}

/// Bytes of memory available for allocation, from `/proc/meminfo`'s
/// `MemAvailable` (Linux).  `None` where unavailable; callers fall back to a
/// conservative assumption (no auto sizing).
#[cfg(target_os = "linux")]
pub fn available_ram_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = meminfo.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

#[cfg(not(target_os = "linux"))]
pub fn available_ram_bytes() -> Option<u64> {
    None
}

// ─── Expert placement (where the routed-expert pool lives) ───────────────────

/// Where a sparse MoE model's routed-expert weights live when inference runs
/// on an accelerator.
///
/// The dense set (embeddings, norms, attention, routers, shared experts,
/// output) is small and touched on every token, so it always goes to the
/// compute device.  The routed experts are the bulk of the file and are
/// touched sparsely, so they can either be uploaded too (fastest, needs the
/// whole model in device memory) or stay in host RAM, borrowed in place from
/// the model mapping and run through the CPU expert kernels, with only the
/// per-layer activation hopping across the bus.  The host variant is what
/// makes a model larger than VRAM runnable at all; it is the layout the
/// `deepseek4` loader has always used, extended to `qwen3moe` / `deepseek2`.
///
/// On the CPU device the placement is moot (everything is host memory) and
/// the setting is ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExpertPlacement {
    /// Decide from the device's memory: keep experts on the device when the
    /// whole model fits with headroom, otherwise keep them in host RAM.  When
    /// no memory probe is available the experts go to the device (the
    /// historical behaviour) unless the backend cannot hold quantized blocks
    /// at all (OpenCL), which always keeps them on the host.
    #[default]
    Auto,
    /// Always upload the routed experts to the compute device.
    Device,
    /// Always keep the routed experts in host RAM (mmap-borrowed on CPU).
    Host,
}

impl std::str::FromStr for ExpertPlacement {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "device" | "gpu" | "vram" => Ok(Self::Device),
            "host" | "cpu" | "ram" => Ok(Self::Host),
            other => Err(format!(
                "unknown expert placement `{other}` (expected auto, device or host)"
            )),
        }
    }
}

/// A resolved placement: [`ExpertPlacement`] minus `Auto`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedPlacement {
    /// Routed experts uploaded to the compute device.
    Device,
    /// Routed experts kept in host RAM.
    Host,
}

/// Memory the routed-expert upload must leave free on the device, for the KV
/// cache, activations and allocator slack (bytes).
pub const DEVICE_PLACEMENT_HEADROOM: u64 = 1024 * 1024 * 1024; // 1 GiB

/// What is known about the compute device when placing the experts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceProfile {
    /// The device is the CPU: placement is irrelevant.
    pub is_cpu: bool,
    /// The routed experts stay on the host unless explicitly requested:
    /// either the backend cannot hold quantized blocks (Vulkan's storage is
    /// dense f32, so uploading the experts would dequantize them — 8–16× the
    /// on-disk size), or it is OpenCL, where the dense set is what the
    /// device speeds up and the experts run on the CPU SIMD expert kernels
    /// with the hot-expert cache and prefetch machinery.
    pub dense_only: bool,
    /// Bytes of device memory available for weights, when known: a probe
    /// (`cudaMemGetInfo`), or the operator's `--vram-budget`.
    pub free_bytes: Option<u64>,
}

/// Decide where the routed experts live.  Pure: every input is a number, so
/// the rule is unit-testable without a device.
///
/// * `Device` / `Host` requests are honoured as-is (a `Device` request on a
///   dense-only backend is honoured too — it is the operator's explicit
///   choice — but logged by the caller).
/// * `Auto` on the CPU is `Host` (moot).  On a dense-only backend it is
///   `Host`.  With a memory figure it is `Device` only when
///   `dense + experts + headroom` fits; without one it is `Device`, the
///   historical behaviour, so a machine with no probe changes nothing.
pub fn resolve_expert_placement(
    requested: ExpertPlacement,
    device: DeviceProfile,
    dense_bytes: u64,
    expert_bytes: u64,
    headroom_bytes: u64,
) -> ResolvedPlacement {
    if device.is_cpu {
        return ResolvedPlacement::Host;
    }
    match requested {
        ExpertPlacement::Device => ResolvedPlacement::Device,
        ExpertPlacement::Host => ResolvedPlacement::Host,
        ExpertPlacement::Auto => {
            if device.dense_only {
                return ResolvedPlacement::Host;
            }
            match device.free_bytes {
                Some(free) => {
                    let need = dense_bytes
                        .saturating_add(expert_bytes)
                        .saturating_add(headroom_bytes);
                    if need <= free {
                        ResolvedPlacement::Device
                    } else {
                        ResolvedPlacement::Host
                    }
                }
                None => ResolvedPlacement::Device,
            }
        }
    }
}

// ─── Dense placement (where the dense set lives) ─────────────────────────────

/// Where a model's dense set — embeddings, norms, attention, routers, shared
/// experts, output — runs when inference is requested on an accelerator.
///
/// These weights are touched on *every* token, so on a backend whose GEMM is
/// faster than CPU-BLAS (big discrete GPUs) they belong on the device.  But a
/// weak integrated GPU / software OpenCL path can be far *slower* than CPU-BLAS
/// (e.g. the Renoir iGPU runs dense OpenCL at ~21 GFLOPS vs ~384 GFLOPS for
/// 16-core BLAS, ~50x worse on attention).  On such systems the best placement
/// keeps the dense set on the CPU and only uses the accelerator where it helps,
/// which is what this knob enables.
///
/// On the CPU device the setting is moot (everything is host memory) and ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DensePlacement {
    /// Dense set goes to the requested compute device.  This is the historical
    /// and recommended behaviour on a real accelerator.
    #[default]
    Auto,
    /// The dense set goes to the requested compute device (explicit).
    Device,
    /// The dense set runs on the CPU (CPU-BLAS), even when an accelerator is
    /// requested.  The routed-expert placement is unaffected.  Use this on
    /// systems where the accelerator's GEMM is slower than CPU-BLAS.
    Cpu,
}

impl std::str::FromStr for DensePlacement {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "device" | "gpu" | "accelerator" => Ok(Self::Device),
            "cpu" | "host" | "blas" => Ok(Self::Cpu),
            other => Err(format!(
                "unknown dense placement `{other}` (expected auto, device or cpu)"
            )),
        }
    }
}

/// A resolved dense placement: [`DensePlacement`] minus `Auto`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedDense {
    /// Dense set runs on the accelerator.
    Device,
    /// Dense set runs on the CPU (CPU-BLAS).
    Cpu,
}

/// Decide where the dense set lives.  Pure: every input is a number, so the
/// rule is unit-testable without a device.
///
/// * `Device` / `Cpu` requests are honoured as-is.
/// * `Auto` on the CPU is `Cpu` (moot).  On any accelerator it is `Device`,
///   keeping the historical "dense always on the device" behaviour so a real
///   GPU is unchanged unless the operator opts into `cpu`.
pub fn resolve_dense_placement(
    requested: DensePlacement,
    is_cpu_device: bool,
) -> ResolvedDense {
    if is_cpu_device {
        return ResolvedDense::Cpu;
    }
    match requested {
        DensePlacement::Device => ResolvedDense::Device,
        DensePlacement::Cpu => ResolvedDense::Cpu,
        DensePlacement::Auto => ResolvedDense::Device,
    }
}

/// Whether the dense set — which, when not forced to the CPU, goes to the
/// compute device — fits the device's memory with headroom.  Expert placement
/// can only move the routed experts; a budget below this cannot be honoured by
/// any placement, so the engine refuses the load instead of deferring an
/// oversized upload to the first request.
pub fn dense_set_fits(dense_device_bytes: u64, headroom_bytes: u64, budget_bytes: u64) -> bool {
    // An overflowing requirement is "does not fit", never a wrap into fitting.
    dense_device_bytes
        .checked_add(headroom_bytes)
        .is_some_and(|need| need <= budget_bytes)
}

/// How many full copies of `instance_bytes` fit in `free_bytes` once
/// `headroom_bytes` is set aside — the number of model sessions an
/// accelerator can hold when each session carries its own weight copy.
/// Always at least 1 (the first session is loaded regardless) and never more
/// than `cap`.
pub fn instances_for_memory(free_bytes: u64, instance_bytes: u64, headroom_bytes: u64, cap: usize) -> usize {
    if instance_bytes == 0 {
        return cap.max(1);
    }
    let usable = free_bytes.saturating_sub(headroom_bytes);
    let n = (usable / instance_bytes) as usize;
    n.clamp(1, cap.max(1))
}

/// Free and total memory of the compute device in bytes, when the backend
/// can report it.
///
/// * CUDA: `cuMemGetInfo` through cudarc (the `cuda` feature).
/// * Metal: unified memory — the GPU shares system RAM, so the caller's
///   system-RAM figure is the right budget and this returns `None`.
/// * OpenCL / CPU: `None`.
pub fn device_memory_info(device: &candle_core::Device) -> Option<(u64, u64)> {
    match device {
        #[cfg(feature = "cuda")]
        candle_core::Device::Cuda(dev) => cuda_memory_info(dev),
        _ => None,
    }
}

#[cfg(feature = "cuda")]
fn cuda_memory_info(dev: &candle_core::CudaDevice) -> Option<(u64, u64)> {
    use candle_core::cuda::cudarc;
    // cudarc's `CudaContext::mem_get_info` wraps `cuMemGetInfo` for this
    // device's own context, so the figures are for the GPU joshua runs on.
    let stream = dev.cuda_stream();
    let ctx: &std::sync::Arc<cudarc::driver::CudaContext> = stream.context();
    let (free, total) = ctx.mem_get_info().ok()?;
    Some((free as u64, total as u64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_fits_available_memory() {
        // 8 GiB free, 1 MiB experts, half usable → 4096 experts.
        let b = expert_budget_for_memory(8 << 30, 1 << 20, 10_000, 0);
        assert_eq!(b, 4096);
    }

    #[test]
    fn device_expert_cache_leaves_only_the_untouched_leftover() {
        // 12 GiB free, 1.1 GiB dense, 1 GiB headroom, 2 GiB KV -> 7.9 GiB cache.
        let b = device_expert_cache_bytes(12 << 30, 1_100 << 20, 1 << 30, 2 << 30);
        assert_eq!(b, (12 << 30) - (1_100 << 20) - (1 << 30) - (2 << 30));
        // Dense + headroom + KV >= free -> 0 (never go negative / overcommit).
        let b0 = device_expert_cache_bytes(3 << 30, 2 << 30, 1 << 30, 1 << 30);
        assert_eq!(b0, 0);
    }

    #[test]
    fn device_expert_slots_fit_uploaded_size_and_cap() {
        // 8 GiB budget, 2 MiB per-expert upload -> 4096 slots, capped at 1000.
        let n = device_expert_slots(8 << 30, 1 << 30, 1 << 30, 1 << 30, 2 << 20, 1000);
        assert_eq!(n, 1000);
        // Uploaded size 1 MiB -> 5120 slots (uncappeduene cap).
        let n2 = device_expert_slots(8 << 30, 1 << 30, 1 << 30, 1 << 30, 1 << 20, 10_000);
        assert_eq!(n2, (5 << 30) / (1 << 20));
    }

    #[test]
    fn budget_reserves_headroom_and_respects_capacity() {
        // 8 GiB free, 2 GiB headroom → 3 GiB usable → 3072 experts of 1 MiB.
        let b = expert_budget_for_memory(8 << 30, 1 << 20, 10_000, 2 << 30);
        assert_eq!(b, 3072);
        // Capacity caps the budget.
        let b = expert_budget_for_memory(8 << 30, 1 << 20, 100, 0);
        assert_eq!(b, 100);
    }

    #[test]
    fn budget_handles_degenerate_inputs() {
        assert_eq!(expert_budget_for_memory(1 << 30, 0, 100, 0), 0);
        assert_eq!(expert_budget_for_memory(0, 1 << 20, 100, 0), 0);
        // Headroom larger than free memory.
        assert_eq!(expert_budget_for_memory(1 << 30, 1 << 20, 100, 2 << 30), 0);
    }

    #[test]
    fn adaptive_grows_when_misses_saturate_the_bus() {
        // 40% hits, 1 GiB of misses per step on a 63 GB/s bus ≈ 17 ms/step.
        let b = adaptive_budget(100, 0.40, (1 << 30) as f64, 63.0 * 1e9, 10, 10_000);
        assert_eq!(b, 200, "low hit ratio + busy bus grows the cache");
    }

    #[test]
    fn adaptive_does_not_grow_without_a_bus() {
        // Same misses but no bus (pure CPU): growing cannot help.
        let b = adaptive_budget(100, 0.40, (1 << 30) as f64, 0.0, 10, 10_000);
        assert_eq!(b, 100);
    }

    #[test]
    fn adaptive_shrinks_when_misses_are_rare() {
        let b = adaptive_budget(100, 0.99, (1 << 20) as f64, 63.0 * 1e9, 10, 10_000);
        assert_eq!(b, 50);
        // Never below the floor.
        let b = adaptive_budget(20, 0.99, (1 << 20) as f64, 63.0 * 1e9, 10, 10_000);
        assert_eq!(b, 10);
    }

    #[test]
    fn adaptive_holds_in_the_middle_and_respects_the_cap() {
        assert_eq!(adaptive_budget(100, 0.90, (1 << 20) as f64, 63.0 * 1e9, 10, 10_000), 100);
        // Growth is capped at max_budget.
        assert_eq!(adaptive_budget(6_000, 0.40, (1 << 30) as f64, 63.0 * 1e9, 10, 10_000), 10_000);
        // A huge budget must saturate rather than wrap: usize::MAX / 2 + 10
        // doubled would overflow a plain `* 2` (panic in debug, wrap below
        // the floor in release).
        assert_eq!(
            adaptive_budget(usize::MAX / 2 + 10, 0.40, (1 << 30) as f64, 63.0 * 1e9, 10, usize::MAX),
            usize::MAX,
            "saturating growth must not wrap"
        );
        // A zero cap disables the cache entirely.
        assert_eq!(adaptive_budget(100, 0.40, (1 << 30) as f64, 63.0 * 1e9, 0, 0), 0);
    }

    // ── Expert placement ────────────────────────────────────────────────

    const GIB: u64 = 1 << 30;

    fn gpu(free: Option<u64>) -> DeviceProfile {
        DeviceProfile { is_cpu: false, dense_only: false, free_bytes: free }
    }

    #[test]
    fn placement_parses_aliases() {
        assert_eq!("auto".parse::<ExpertPlacement>().unwrap(), ExpertPlacement::Auto);
        assert_eq!("Device".parse::<ExpertPlacement>().unwrap(), ExpertPlacement::Device);
        assert_eq!("gpu".parse::<ExpertPlacement>().unwrap(), ExpertPlacement::Device);
        assert_eq!("host".parse::<ExpertPlacement>().unwrap(), ExpertPlacement::Host);
        assert_eq!("cpu".parse::<ExpertPlacement>().unwrap(), ExpertPlacement::Host);
        assert!("sideways".parse::<ExpertPlacement>().is_err());
    }

    #[test]
    fn placement_on_cpu_is_always_host() {
        let cpu = DeviceProfile { is_cpu: true, dense_only: false, free_bytes: Some(100 * GIB) };
        for req in [ExpertPlacement::Auto, ExpertPlacement::Device, ExpertPlacement::Host] {
            assert_eq!(resolve_expert_placement(req, cpu, GIB, GIB, 0), ResolvedPlacement::Host);
        }
    }

    #[test]
    fn placement_explicit_requests_are_honoured() {
        // Even a model that plainly does not fit goes to the device on request…
        assert_eq!(
            resolve_expert_placement(ExpertPlacement::Device, gpu(Some(4 * GIB)), 8 * GIB, 60 * GIB, GIB),
            ResolvedPlacement::Device
        );
        // …and a tiny model stays on the host on request.
        assert_eq!(
            resolve_expert_placement(ExpertPlacement::Host, gpu(Some(80 * GIB)), GIB, GIB, GIB),
            ResolvedPlacement::Host
        );
    }

    #[test]
    fn placement_auto_follows_the_memory_probe() {
        // Qwen3-30B-A3B Q4_K_M on a 12 GiB card: dense ~1 GiB + experts ~17 GiB.
        assert_eq!(
            resolve_expert_placement(ExpertPlacement::Auto, gpu(Some(12 * GIB)), GIB, 17 * GIB, GIB),
            ResolvedPlacement::Host
        );
        // The same model on a 24 GiB card fits with headroom.
        assert_eq!(
            resolve_expert_placement(ExpertPlacement::Auto, gpu(Some(24 * GIB)), GIB, 17 * GIB, GIB),
            ResolvedPlacement::Device
        );
        // Exactly at the limit still fits; one byte over does not.
        assert_eq!(
            resolve_expert_placement(ExpertPlacement::Auto, gpu(Some(19 * GIB)), GIB, 17 * GIB, GIB),
            ResolvedPlacement::Device
        );
        assert_eq!(
            resolve_expert_placement(ExpertPlacement::Auto, gpu(Some(19 * GIB - 1)), GIB, 17 * GIB, GIB),
            ResolvedPlacement::Host
        );
    }

    #[test]
    fn placement_auto_without_a_probe_keeps_the_historical_layout() {
        assert_eq!(
            resolve_expert_placement(ExpertPlacement::Auto, gpu(None), GIB, 100 * GIB, GIB),
            ResolvedPlacement::Device
        );
    }

    #[test]
    fn placement_auto_on_a_dense_only_backend_is_host() {
        let ocl = DeviceProfile { is_cpu: false, dense_only: true, free_bytes: Some(1000 * GIB) };
        assert_eq!(
            resolve_expert_placement(ExpertPlacement::Auto, ocl, GIB, GIB, 0),
            ResolvedPlacement::Host
        );
    }

    #[test]
    fn placement_handles_overflowing_sizes() {
        assert_eq!(
            resolve_expert_placement(ExpertPlacement::Auto, gpu(Some(u64::MAX)), u64::MAX, u64::MAX, u64::MAX),
            ResolvedPlacement::Device,
            "saturating add must not wrap below the probe"
        );
    }

    #[test]
    fn dense_set_fit_is_checked_with_headroom() {
        assert!(dense_set_fits(6 * GIB, GIB, 7 * GIB));
        assert!(!dense_set_fits(6 * GIB, GIB, 7 * GIB - 1));
        assert!(!dense_set_fits(6 * GIB, GIB, 4 * GIB));
        assert!(!dense_set_fits(u64::MAX, GIB, u64::MAX), "saturating add must not wrap into a fit");
    }

    #[test]
    fn instances_for_memory_counts_whole_copies() {
        // 24 GiB free, 1 GiB headroom, 5 GiB per instance → 4 sessions.
        assert_eq!(instances_for_memory(24 * GIB, 5 * GIB, GIB, 8), 4);
        // Capped by the pool limit.
        assert_eq!(instances_for_memory(24 * GIB, GIB, 0, 2), 2);
        // Never below one, even when nothing fits.
        assert_eq!(instances_for_memory(GIB, 5 * GIB, GIB, 8), 1);
        // Unknown instance size: no constraint.
        assert_eq!(instances_for_memory(GIB, 0, 0, 3), 3);
        assert_eq!(instances_for_memory(GIB, 0, 0, 0), 1);
    }
}
