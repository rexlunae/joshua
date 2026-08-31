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
            return (current * 2).min(max_budget);
        }
        return current;
    }
    if hit_ratio > ADAPTIVE_HIGH_HIT_RATIO {
        return (current / 2).max(min_budget);
    }
    current
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
        // A zero cap disables the cache entirely.
        assert_eq!(adaptive_budget(100, 0.40, (1 << 30) as f64, 63.0 * 1e9, 0, 0), 0);
    }
}
