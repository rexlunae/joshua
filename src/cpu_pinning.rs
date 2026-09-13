//! P-core (`performance`) CPU pinning for hybrid-core machines (opt-in).
//!
//! Modern Intel client chips (Alder / Raptor Lake, e.g. the i5-14400) ship a
//! mix of Performance (P) cores and Efficiency (E) cores.  The P-cores clock
//! far higher and carry all the SIMD/AVX weight, yet on Linux the scheduler
//! treats every logical CPU as equal, so worker threads can end up parked on
//! an E-core.  For a CPU-bound quantized-dequant matmul that can leave
//! throughput on the table.
//!
//! This is **opt-in** via the `JOSHUA_CPU_AFFINITY` environment variable:
//!
//! * `off` / `0` / unset  — leave affinity alone (the default).
//! * `auto`               — detect the P-core logical set heuristically and pin.
//! * `0-11`, `0,2,4`, …   — pin to an explicit logical-CPU list.
//!
//! It is disabled by default because on the reference hardware (an
//! i5-14400, 61 GiB RAM, model on lz4 ZFS) the decode workload proved
//! memory/mmap-bound rather than CPU-clock-bound: prefill/decode measured
//! ~0.5 t/s both pinned to the P-cores and unconstrained.  On a machine where
//! the expert path is genuinely CPU-bound the pin can still win, and this
//! module is the hook to enable it per-host without a code change.
//!
//! When enabled, affinity is applied to the *process* early in startup via
//! `sched_setaffinity` (Linux only).  On Linux a new pthread inherits its
//! creator's affinity mask, so every pool created afterwards — joshua's own
//! `joshua-matmul` ray pool, plus candle-core's barrier/ray pools — inherits
//! the mask.  Detection heuristic: on current Intel hybrids the P-core
//! hyper-thread pairs occupy the low logical-CPU indices and the E-cores the
//! high ones (verified on an i5-14400: logical 0-11 are the six P-cores,
//! 12-15 the four E-cores).

/// Resolve the requested P-core logical-CPU list, or `None` to leave affinity
/// alone (feature off, or undetectable).
pub fn p_core_list() -> Option<Vec<u32>> {
    let v = match std::env::var("JOSHUA_CPU_AFFINITY") {
        Ok(s) => Some(str::to_lowercase(&s)),
        Err(_) => None,
    };
    match v {
        // Explicitly off, or unset (default off).
        Some(s)
            if s == "0" || s == "off" || s == "false" || s == "none" || s == "no" || s == "" =>
            None,
        // Automatic detection.
        Some(s) if s == "auto" => detect_p_cores(),
        // Explicit comma/range list.
        Some(s) => parse_cpu_list(&s),
        None => None,
    }
}

/// Detect the P-cores as the low logical-CPU prefix.
fn detect_p_cores() -> Option<Vec<u32>> {
    match hybrid_p_cores() {
        Some(v) if v.is_empty() => None,
        Some(v) => Some(v),
        None => None,
    }
}

/// Parse `/proc/cpuinfo` and return the sorted logical-CPU list of the
/// P-cores, or an empty list when not a hybrid / unreadable.
///
/// Discriminator: on Intel hybrid generations hyper-threading is only enabled
/// on the Performance cores, so a physical core whose `core id` is shared by
/// two (or more) logical `processor`s is a P-core, while a single-thread core
/// (one logical) is an E-core.  This is robust to P/E core interleaving,
/// unlike a naive "upper half are E-cores" split.
fn hybrid_p_cores() -> Option<Vec<u32>> {
    let src = match std::fs::read_to_string("/proc/cpuinfo") {
        Ok(s) => s,
        Err(_) => return None,
    };
    // `core id` -> its logical `processor` indices.  Few physical cores, so a
    // linear scan of a small vec of (core_id, logicals) is fine.
    let mut core_ids: Vec<u32> = Vec::new();
    let mut core_logs: Vec<Vec<u32>> = Vec::new();
    let mut cur_proc: Option<u32> = None;
    for line in src.lines() {
        if line.starts_with("processor") {
            // "processor\t: N"
            if let Some(nstr) = line.split_whitespace().last() {
                if let Ok(n) = (*nstr).parse::<u32>() {
                    cur_proc = Some(n);
                }
            }
        } else if line.starts_with("core id") {
            if let Some(cstr) = line.split_whitespace().last() {
                if let Ok(core) = (*cstr).parse::<u32>() {
                    if let Some(p) = cur_proc {
                        let i = core_ids.iter().position(|c| *c == core);
                        if let Some(i) = i {
                            if let Some(logs) = core_logs.get_mut(i) {
                                logs.push(p);
                            }
                        } else {
                            core_ids.push(core);
                            core_logs.push(vec![p]);
                        }
                    }
                }
            }
        }
    }
    if core_ids.is_empty() {
        return None;
    }
    // P-cores = cores with >1 logical (hyper-threaded).  Collect their
    // logicals and sort.
    let mut out: Vec<u32> = Vec::new();
    for logs in core_logs {
        if logs.len() > 1 {
            for l in logs {
                out.push(l);
            }
        }
    }
    out.sort_unstable();
    Some(out)
}

fn parse_cpu_list(s: &str) -> Option<Vec<u32>> {
    let mut out: Vec<u32> = Vec::new();
    for part in s.split(",") {
        let p = part.trim();
        if p.contains("-") {
            let r = p.split("-").collect::<Vec<&str>>();
            let lo = r.get(0)?.trim().parse::<u32>().ok();
            let hi = r.get(1)?.trim().parse::<u32>().ok();
            if let Some(lo) = lo {
                if let Some(hi) = hi {
                    for i in lo..=hi {
                        out.push(i);
                    }
                }
            }
        } else {
            let v = p.parse::<u32>().ok();
            if let Some(v) = v {
                out.push(v);
            }
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Apply P-core affinity to the calling process (best effort, opt-in).
///
/// Call early in startup, before any worker pool is created.  Returns `true`
/// when a mask was actually applied.  Off-Linux, and with the feature off by
/// default, this is a no-op.
pub fn apply_p_core_affinity() -> bool {
    #[cfg(target_os = "linux")]
    {
        match p_core_list() {
            None => false,
            Some(list) if list.is_empty() => false,
            Some(list) => {
                // Build a fresh zeroed cpu_set (1024 bits = 16 x u64) and drop
                // the P-core bits.  We pass raw pointers to sched_setaffinity;
                // libc::CPU_SET would need a concrete &mut cpu_set_t, so we set
                // the bits ourselves.
                let mut bits = [0u64; 16];
                for c in &list {
                    let log = *c as usize;
                    bits[log / 64] |= 1u64 << (log % 64);
                }
                let applied = unsafe {
                    libc::sched_setaffinity(
                        0,
                        (16 * std::mem::size_of::<u64>()) as libc::size_t,
                        bits.as_ptr() as *const libc::cpu_set_t,
                    ) == 0
                };
                if applied {
                    tracing::info!(
                        "pinning process to {} P-core logical CPU(s): {list:?}",
                        list.len(),
                    );
                }
                applied
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}