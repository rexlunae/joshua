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
    /// The routed experts stay on the host unless explicitly requested
    /// (OpenCL and Vulkan): the dense set is what the device speeds up,
    /// while the experts run on the CPU SIMD expert kernels with the
    /// hot-expert cache and prefetch machinery, sharing DRAM with an iGPU
    /// either way.
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

// ─── Auto-engaged VRAM expert cache (#110) ──────────────────────────────────

/// The operator's `--vram-expert-cache` setting.
///
/// The CLI parses `auto` (size the cache from the device's free memory at
/// load), a fixed MiB budget, and `0`; no flag is its own state.  Unset and
/// `0` must stay distinct now that `--expert-placement auto` may engage a
/// partial cache by itself (#110): `0` is the explicit opt-out, while unset
/// lets the engine decide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VramExpertCache {
    /// No `--vram-expert-cache` flag: `--expert-placement auto` may engage a
    /// bounded VRAM expert cache over the host pool when the routed experts
    /// do not fit the device and the leftover memory is worth a cache
    /// ([`auto_vram_expert_cache`]).
    #[default]
    Unset,
    /// `--vram-expert-cache 0`: never build a device expert pool.
    Disabled,
    /// `--vram-expert-cache auto`: size the cache from the device's free
    /// memory at load (device free − dense − headroom − scratch − KV).
    Auto,
    /// `--vram-expert-cache <MiB>`: a fixed byte budget.
    Bytes(u64),
}

/// Smallest partial cache `--expert-placement auto` engages by default
/// (#110): the byte budget must hold at least this share (numerator,
/// denominator) of the routed-expert device footprint.
///
/// The Arc Pro B50 in the #110 discussion measured a 2.4 GiB budget against
/// a ~72 GiB routed-expert pool — 3.3% residency: every decode step missed,
/// the background uploader saturated the bus, and decode regressed hard.
/// Below the floor the safer default is all-host placement, unchanged from
/// before #110.
pub const AUTO_VRAM_CACHE_MIN_RESIDENCY: (u64, u64) = (1, 10);

/// Why `--expert-placement auto` kept the all-host expert placement instead
/// of engaging a partial VRAM expert cache (#110).
///
/// [`AutoVramCacheDecline::NotAuto`], [`AutoVramCacheDecline::NotHost`] and
/// [`AutoVramCacheDecline::NotCacheCapable`] mean the decision was not
/// applicable (the caller state made it moot — nothing to log); the rest
/// are real declines worth telling the operator about, with the lever that
/// forces the cache anyway.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AutoVramCacheDecline {
    /// The expert placement was chosen explicitly (not `auto`).
    NotAuto,
    /// The routed experts did not fall back to the host: the whole model
    /// fits the device (or the operator chose `device`), so they are
    /// already on the card.
    NotHost,
    /// The architecture has no device form for its routed experts.
    NotCacheCapable,
    /// No device memory budget: no probe and no `--vram-budget`.
    NoBudget,
    /// The model has no routed-expert device footprint (nothing to cache).
    NoExperts,
    /// The device memory left after the dense set, the placement headroom,
    /// the scratch and the KV reserve is zero.
    ZeroLeftover,
    /// The budget would hold less than [`AUTO_VRAM_CACHE_MIN_RESIDENCY`] of
    /// the routed experts (the measured percent) — the cache would turn
    /// over every step.
    TooSmallResidency(u64),
    /// The placement probe measured the device *decoding* slower than
    /// CPU-BLAS (the measured speedup): resident experts would run on
    /// slower kernels than the host's SIMD ones.
    DeviceSlower(f64),
}

impl AutoVramCacheDecline {
    /// Whether the decline applies to a real cache candidate (worth a log
    /// line); `false` means the decision was simply not applicable.
    pub fn applicable(&self) -> bool {
        !matches!(
            self,
            AutoVramCacheDecline::NotAuto
                | AutoVramCacheDecline::NotHost
                | AutoVramCacheDecline::NotCacheCapable
        )
    }

    /// One log-friendly clause explaining the decline.
    pub fn describe(&self) -> String {
        match self {
            AutoVramCacheDecline::NotAuto => "the expert placement was not `auto`".to_string(),
            AutoVramCacheDecline::NotHost => {
                "the routed experts did not fall back to the host".to_string()
            }
            AutoVramCacheDecline::NotCacheCapable => {
                "the architecture has no device form for its routed experts".to_string()
            }
            AutoVramCacheDecline::NoBudget => {
                "no device memory budget (no probe and no --vram-budget)".to_string()
            }
            AutoVramCacheDecline::NoExperts => {
                "the model has no routed experts to cache".to_string()
            }
            AutoVramCacheDecline::ZeroLeftover => {
                "no device memory is left after the dense set, headroom, scratch and KV reserve"
                    .to_string()
            }
            AutoVramCacheDecline::TooSmallResidency(pct) => format!(
                "the leftover device memory holds {pct}% of the routed experts (below the {:.0}% floor); the cache would turn over every step",
                100.0 * AUTO_VRAM_CACHE_MIN_RESIDENCY.0 as f64 / AUTO_VRAM_CACHE_MIN_RESIDENCY.1 as f64,
            ),
            AutoVramCacheDecline::DeviceSlower(speedup) => format!(
                "the placement probe measured decode at {speedup:.2}x of CPU-BLAS (the device is slower than the CPU for the dense set; the probe is a Q4_K proxy for the expert kernels)"
            ),
        }
    }
}

/// Decide whether `--expert-placement auto` — which resolved to
/// [`ResolvedPlacement::Host`] because the routed experts do not fit the
/// device's memory — should still engage a *bounded* VRAM expert cache over
/// the host pool, sized from the leftover device memory at load (#110), and
/// if not, why not.  `Ok(())` engages (the caller sizes it as
/// `--vram-expert-cache auto`); the error names the decline.
///
/// Conservative by design — the #110 discussion measured a hard decode
/// regression on an Arc Pro B50 whose 2.4 GiB leftover held only 3.3% of
/// the routed experts:
///
/// * only an `Auto` request engages (an explicit `host` is the documented
///   opt-out; an explicit `device` is honoured elsewhere);
/// * only a *fallback* to `Host` engages — the `Device` placement already
///   runs the experts on the card;
/// * only an architecture whose device expert form *is* the cache engages
///   (`deepseek4` on OpenCL/SYCL); the caller passes that as
///   `cache_capable`;
/// * the leftover device budget (`cache_budget_bytes`, computed by
///   [`device_expert_cache_bytes`] from the probe or `--vram-budget`) must
///   be known and non-zero;
/// * it must hold at least [`AUTO_VRAM_CACHE_MIN_RESIDENCY`] of the
///   routed-expert device footprint (`expert_device_bytes`), so the cache
///   does not turn over every step;
/// * a placement probe that measured the device *decoding* slower than
///   CPU-BLAS (`probe_decode_speedup` below 1.0) keeps the cache off — the
///   cache only affects decode, and resident experts would run on slower
///   kernels than the host's.  No probe (`None`) leaves the size rule in
///   charge.  The probe is a *proxy*: it times the dense set's Q4_K GEMM,
///   while the cached deepseek4 experts run IQ2_XXS gate/up and Q2_K down
///   device kernels whose device-vs-CPU speedup can differ — treat this
///   gate as a coarse filter and confirm on hardware (the decode time-split
///   log); `--vram-expert-cache auto` overrides it either way.
pub fn auto_vram_expert_cache(
    requested: ExpertPlacement,
    resolved: ResolvedPlacement,
    cache_capable: bool,
    cache_budget_bytes: Option<u64>,
    expert_device_bytes: u64,
    probe_decode_speedup: Option<f64>,
) -> Result<(), AutoVramCacheDecline> {
    if requested != ExpertPlacement::Auto {
        return Err(AutoVramCacheDecline::NotAuto);
    }
    if resolved != ResolvedPlacement::Host {
        return Err(AutoVramCacheDecline::NotHost);
    }
    if !cache_capable {
        return Err(AutoVramCacheDecline::NotCacheCapable);
    }
    let Some(budget) = cache_budget_bytes else {
        return Err(AutoVramCacheDecline::NoBudget);
    };
    if expert_device_bytes == 0 {
        return Err(AutoVramCacheDecline::NoExperts);
    }
    if budget == 0 {
        return Err(AutoVramCacheDecline::ZeroLeftover);
    }
    let (num, den) = AUTO_VRAM_CACHE_MIN_RESIDENCY;
    // budget/expert >= num/den  is  budget*den >= expert*num; saturating on
    // both sides keeps a huge probe figure from wrapping into a decline.
    if budget.saturating_mul(den) < expert_device_bytes.saturating_mul(num) {
        // The measured percent, in u128 so a huge budget cannot overflow.
        let pct = (100u128 * budget as u128 / expert_device_bytes as u128) as u64;
        return Err(AutoVramCacheDecline::TooSmallResidency(pct));
    }
    if probe_decode_speedup.is_some_and(|s| s < 1.0) {
        return Err(AutoVramCacheDecline::DeviceSlower(
            probe_decode_speedup.unwrap_or(0.0),
        ));
    }
    Ok(())
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
/// * OpenCL: a *discrete* device reports `CL_DEVICE_GLOBAL_MEM_SIZE` as both
///   figures (OpenCL has no free-memory query; the engine's headroom and KV
///   reserve are what absorb the difference).  A device sharing host memory
///   (an iGPU, a CPU runtime such as pocl) returns `None` like Metal: its
///   "global memory" is system RAM, which the caller already budgets.
/// * SYCL: the device's global memory as both figures, as for a discrete
///   OpenCL device (the backend allocates explicit device USM even on an
///   iGPU, so its global memory is the right budget).
/// * CPU / Vulkan: `None`.
pub fn device_memory_info(device: &candle_core::Device) -> Option<(u64, u64)> {
    match device {
        #[cfg(feature = "cuda")]
        candle_core::Device::Cuda(dev) => cuda_memory_info(dev),
        #[cfg(feature = "opencl")]
        candle_core::Device::OpenCl(dev) => {
            if dev.host_unified_memory() {
                return None;
            }
            let total = dev.global_mem_size()?;
            (total > 0).then_some((total, total))
        }
        #[cfg(feature = "sycl")]
        candle_core::Device::Sycl(dev) => {
            let total = dev.global_mem_size()?;
            (total > 0).then_some((total, total))
        }
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
    // ── Auto-engaged VRAM expert cache (#110) ───────────────────────────

    fn auto_request(
        cache_budget: Option<u64>,
        expert_bytes: u64,
        probe: Option<f64>,
    ) -> Result<(), AutoVramCacheDecline> {
        auto_vram_expert_cache(
            ExpertPlacement::Auto,
            ResolvedPlacement::Host,
            true,
            cache_budget,
            expert_bytes,
            probe,
        )
    }

    /// The engaged case: a leftover that holds well over the residency floor
    /// on a device that decodes faster than CPU-BLAS.
    #[test]
    fn auto_vram_cache_engages_on_a_meaningful_budget() {
        // 24 GiB card, 9.7 GiB dense on the device, 2 GiB headroom+scratch,
        // 1 GiB KV → ~11.3 GiB left against a 72 GiB expert pool ≈ 15%.
        let leftover = device_expert_cache_bytes(24 * GIB, 9_700 << 20, 2 * GIB, GIB);
        assert_eq!(leftover, (24 * GIB) - (9_700 << 20) - 2 * GIB - GIB);
        auto_request(Some(leftover), 72 * GIB, Some(2.0)).expect("engage");
        // No probe verdict (the probe was skipped): the size rule decides.
        auto_request(Some(leftover), 72 * GIB, None).expect("engage without a probe");
        // Exactly at the floor (10%) engages, and exactly "not slower than
        // CPU" (1.0x) engages; one expert byte short of the floor declines.
        auto_request(Some(7_200 << 20), 72_000 << 20, Some(1.0)).expect("engage at the floor");
        let err = auto_request(Some(7_199 << 20), 72_000 << 20, Some(1.0)).unwrap_err();
        assert!(
            matches!(err, AutoVramCacheDecline::TooSmallResidency(9)),
            "one byte short of the floor: {err:?}"
        );
    }

    /// The B50 shape from the #110 discussion: a ~2.3 GiB leftover against a
    /// ~72 GiB expert pool (3% residency) must stay all-host, and a device
    /// that decodes slower than CPU-BLAS stays all-host even with a large
    /// leftover.
    #[test]
    fn auto_vram_cache_keeps_host_when_the_cache_is_not_worth_it() {
        let leftover = device_expert_cache_bytes(15 * GIB + (130 << 20), 9_730 << 20, 2 * GIB, GIB);
        let err = auto_request(Some(leftover), 72 * GIB, Some(2.0)).unwrap_err();
        assert!(
            matches!(err, AutoVramCacheDecline::TooSmallResidency(3)),
            "2.3 GiB of 72 GiB is 3% residency: {err:?}"
        );
        // A device slower than CPU-BLAS on decode never engages, however
        // large the leftover (the Arc dense-decode measurement was 0.39x).
        let err = auto_request(Some(72 * GIB), 72 * GIB, Some(0.39)).unwrap_err();
        assert!(
            matches!(err, AutoVramCacheDecline::DeviceSlower(s) if (s - 0.39).abs() < 1e-9),
            "{err:?}"
        );
    }

    /// The explicit opt-outs and the not-applicable states.
    #[test]
    fn auto_vram_cache_honours_explicit_choices_and_states() {
        // An explicit `host` request is the documented opt-out.
        assert_eq!(
            auto_vram_expert_cache(
                ExpertPlacement::Host,
                ResolvedPlacement::Host,
                true,
                Some(72 * GIB),
                72 * GIB,
                None,
            ),
            Err(AutoVramCacheDecline::NotAuto)
        );
        // A device placement never engages (the experts are already there).
        assert_eq!(
            auto_vram_expert_cache(
                ExpertPlacement::Auto,
                ResolvedPlacement::Device,
                true,
                Some(72 * GIB),
                72 * GIB,
                None,
            ),
            Err(AutoVramCacheDecline::NotHost)
        );
        // An architecture without a device expert form never engages.
        assert_eq!(
            auto_vram_expert_cache(
                ExpertPlacement::Auto,
                ResolvedPlacement::Host,
                false,
                Some(72 * GIB),
                72 * GIB,
                None,
            ),
            Err(AutoVramCacheDecline::NotCacheCapable)
        );
        // No budget (no probe, no --vram-budget), no experts, no leftover.
        assert_eq!(
            auto_request(None, 72 * GIB, None),
            Err(AutoVramCacheDecline::NoBudget)
        );
        assert_eq!(
            auto_request(Some(GIB), 0, None),
            Err(AutoVramCacheDecline::NoExperts)
        );
        assert_eq!(
            auto_request(Some(0), 72 * GIB, None),
            Err(AutoVramCacheDecline::ZeroLeftover)
        );
        // Only the non-applicable declines are silent; the rest are logged.
        assert!(!AutoVramCacheDecline::NotAuto.applicable());
        assert!(!AutoVramCacheDecline::NotHost.applicable());
        assert!(!AutoVramCacheDecline::NotCacheCapable.applicable());
        assert!(AutoVramCacheDecline::TooSmallResidency(3).applicable());
        assert!(AutoVramCacheDecline::DeviceSlower(0.39).applicable());
        // The decline text names the measured figures.
        assert!(AutoVramCacheDecline::TooSmallResidency(3)
            .describe()
            .contains("3%"));
        assert!(AutoVramCacheDecline::DeviceSlower(0.39)
            .describe()
            .contains("0.39x"));
    }

    /// The residency comparison saturates instead of wrapping: a huge budget
    /// against a small expert pool engages, never declines through overflow.
    #[test]
    fn auto_vram_cache_residency_handles_huge_inputs() {
        auto_request(Some(u64::MAX), GIB, None).expect("a huge budget engages");
        // The percent fits u64 even for the largest budgets.
        let err = auto_request(Some(1 << 20), u64::MAX, None).unwrap_err();
        assert!(
            matches!(err, AutoVramCacheDecline::TooSmallResidency(0)),
            "{err:?}"
        );
    }
}
