//! Loader-agnostic routing-frequency LRU hot-expert cache.
//!
//! Sparse mixture-of-experts models far larger than RAM (DeepSeek-V2/V3/V4,
//! Qwen3-MoE, …) demand-fault their routed-expert weights from the model
//! mapping on every token.  Routing is temporally local — consecutive tokens
//! route through heavily overlapping expert sets — so a small fraction of
//! experts carries most of the traffic.  This module records which experts
//! each token routes through and, every [`REFRESH_STEPS`] decode steps,
//! re-selects the `budget` most-used `(layer, expert)` pairs (recency as the
//! tie-break) and reports the *newly-hot* pairs, so the owning loader can
//! keep their pages resident (best-effort `MADV_WILLNEED`, or `mlock` for a
//! hard guarantee).
//!
//! The cache itself performs no I/O and knows nothing about weights: loaders
//! feed it routing and prefetch the reported experts with their own
//! per-expert handles.  Every joshua-native MoE loader (`deepseek2`,
//! `deepseek4`, `qwen3moe`) integrates it; the vendored candle `llama`
//! loader (Mixtral) has no per-expert handles and ignores it.

/// How often (in decode steps) the hot set is re-selected.
///
/// A refresh is a cheap sort over the per-expert hit counters, so the
/// interval mostly bounds `madvise`/prefetch churn rather than compute.
pub const REFRESH_STEPS: u64 = 64;

/// Loader-agnostic routing-frequency LRU bookkeeping for a hot-expert cache.
#[derive(Debug)]
pub struct HotExpertCache {
    /// Maximum number of `(layer, expert)` pairs kept hot.  `0` disables the
    /// cache (recording and refresh become no-ops).
    budget: usize,
    /// Per-layer per-expert routing hit counts, `[layer][expert]`.  The
    /// frequency side of the routing-frequency LRU.
    hits: Vec<Vec<u64>>,
    /// Per-layer per-expert step of the last route.  The recency clock that
    /// breaks frequency ties (the LRU side).
    last_used: Vec<Vec<u64>>,
    /// Total forward passes completed; the shared recency clock.
    step: u64,
    /// The currently hot experts, `(layer, expert)`, in priority order.
    /// Diffed against the recomputed top set on refresh, so a stable hot set
    /// never re-prefetches.
    hot_set: Vec<(u32, u32)>,
}

impl HotExpertCache {
    /// Build a cache for `n_layers` layers of `n_experts` routed experts,
    /// keeping `budget` of them hot (`0` disables).
    pub fn new(n_layers: usize, n_experts: usize, budget: usize) -> Self {
        Self {
            budget,
            hits: vec![vec![0u64; n_experts]; n_layers],
            last_used: vec![vec![0u64; n_experts]; n_layers],
            step: 0,
            hot_set: Vec::with_capacity(budget.min(n_layers.saturating_mul(n_experts))),
        }
    }

    /// The current budget (`0` = disabled).
    pub fn budget(&self) -> usize {
        self.budget
    }

    /// Total routed experts tracked (for diagnostics).
    pub fn total_experts(&self) -> usize {
        self.hits.iter().map(Vec::len).sum()
    }

    /// Change the budget (`0` disables) and drop the current hot set so the
    /// next refresh re-selects under the new budget.
    pub fn set_budget(&mut self, n: usize) {
        if self.budget == n {
            return;
        }
        self.budget = n;
        self.hot_set.clear();
        if n > 0 {
            tracing::info!(
                "expert cache: pinning up to {n} hot experts ({} available)",
                self.total_experts()
            );
        }
    }

    /// Advance the recency clock; call once per forward pass and pass the
    /// returned step to [`HotExpertCache::record`].
    ///
    /// The clock counts **decode steps only**: pass `decode = seq_len == 1`.
    /// Prefill forwards record routing against the current step but do not
    /// advance it, so the refresh interval and the recency tie-break measure
    /// decode activity — a long prompt cannot consume the interval and
    /// trigger premature refresh churn.
    pub fn begin_step(&mut self, decode: bool) -> u64 {
        if decode {
            self.step = self.step.wrapping_add(1);
        }
        self.step
    }

    /// Record that one layer routed through `ids` at `step` (the value
    /// returned by [`HotExpertCache::begin_step`] for this forward pass).
    /// Bounds-checked per layer/expert, so dense layers and hash layers that
    /// report a different expert count are safe no-ops.  A disabled cache
    /// (budget 0) is fully inert.
    pub fn record(&mut self, layer: usize, ids: &[u32], step: u64) {
        if self.budget == 0 {
            return;
        }
        let Some(hits) = self.hits.get_mut(layer) else {
            return;
        };
        let used = &mut self.last_used[layer];
        for &e in ids {
            if let Some(h) = hits.get_mut(e as usize) {
                *h = h.saturating_add(1);
                used[e as usize] = step;
            }
        }
    }

    /// Whether a decode-step refresh is due.  The clock only advances on
    /// decode steps, so the interval boundary can only be reached by decode
    /// activity.
    pub fn refresh_due(&self) -> bool {
        self.budget > 0 && self.step > 0 && self.step % REFRESH_STEPS == 0
    }

    /// Re-select the hot set from routing frequency (recency as the
    /// tie-break) and return it in priority order for the owner to prefetch.
    ///
    /// The **whole** hot set is returned on every refresh, not just the
    /// members that changed: `MADV_WILLNEED` is an advisory hint whose effect
    /// can be evicted by memory pressure, so a stable set must re-assert its
    /// residency periodically or evicted pages would return through demand
    /// faults indefinitely.  Re-advising already-resident pages is
    /// effectively free; the refresh cadence bounds how stale a resident
    /// hint may get.
    pub fn refresh(&mut self) -> Vec<(u32, u32)> {
        if self.budget == 0 {
            return Vec::new();
        }
        // Sort key `(hits desc, last_used desc)`: frequency first, recency
        // breaks ties — the routing-frequency LRU policy.
        let mut cand: Vec<(u64, u64, u32, u32)> = Vec::new();
        for (l, layer_hits) in self.hits.iter().enumerate() {
            for (e, &h) in layer_hits.iter().enumerate() {
                if h > 0 {
                    cand.push((h, self.last_used[l][e], l as u32, e as u32));
                }
            }
        }
        cand.sort_unstable_by(|a, b| b.cmp(a));
        cand.truncate(self.budget);
        let new_hot: Vec<(u32, u32)> = cand.into_iter().map(|(_, _, l, e)| (l, e)).collect();
        self.hot_set = new_hot;

        if !self.hot_set.is_empty() {
            tracing::info!(
                "expert cache: {} hot experts (budget {}) — re-asserting residency",
                self.hot_set.len(),
                self.budget,
            );
        }
        self.hot_set.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Selection follows frequency first, then recency (the LRU tie-break).
    #[test]
    fn refresh_selects_by_frequency_then_recency() {
        let mut c = HotExpertCache::new(2, 4, 3);
        c.set_budget(3);
        // Layer 0: expert 1 used twice, expert 0 once (old), expert 2 once (new).
        // Layer 1: expert 3 used once (newest).
        let s1 = c.begin_step(true);
        c.record(0, &[1], s1);
        let s2 = c.begin_step(true);
        c.record(0, &[1], s2);
        let s3 = c.begin_step(true);
        c.record(0, &[0], s3);
        let s4 = c.begin_step(true);
        c.record(0, &[2], s4);
        let s5 = c.begin_step(true);
        c.record(1, &[3], s5);

        let hot = c.refresh();
        assert_eq!(hot.len(), 3);
        // Most frequent first.
        assert_eq!(hot[0], (0, 1));
        // The hits=1 group is ordered by recency: expert 3 (step 5) newest,
        // then expert 2 (step 4), then expert 0 (step 3) — which misses the
        // budget of 3.
        assert_eq!(hot[1], (1, 3));
        assert_eq!(hot[2], (0, 2));
    }

    /// A stable hot set is returned in full on every refresh: residency
    /// hints are advisory and can be evicted, so the whole set must be
    /// re-asserted periodically rather than only when members change.
    #[test]
    fn refresh_reasserts_the_full_hot_set() {
        let mut c = HotExpertCache::new(1, 4, 2);
        c.set_budget(2);
        let s1 = c.begin_step(true);
        c.record(0, &[1, 2], s1);
        let first = c.refresh();
        assert_eq!(first.len(), 2);

        let s2 = c.begin_step(true);
        c.record(0, &[1, 2], s2);
        let second = c.refresh();
        assert_eq!(second.len(), 2, "stable hot set must be re-asserted in full");
        assert_eq!(second, first);
    }

    /// The recency clock advances on decode steps only: prefills record
    /// routing without consuming the refresh interval.
    #[test]
    fn step_counts_decode_only() {
        let mut c = HotExpertCache::new(1, 4, 2);
        c.set_budget(2);
        // 60 decode steps then 5000 prefill tokens: refresh must NOT be due.
        for _ in 0..60 {
            let s = c.begin_step(true);
            c.record(0, &[1], s);
        }
        for _ in 0..5000 {
            let s = c.begin_step(false);
            c.record(0, &[1, 2], s);
        }
        assert!(!c.refresh_due(), "prefills must not advance the decode clock");
        // Exactly four more decode steps reach the interval boundary.
        for _ in 0..4 {
            let s = c.begin_step(true);
            c.record(0, &[1], s);
        }
        assert!(c.refresh_due(), "refresh due after exactly 64 decode steps");
    }

    /// Budget zero disables refresh; the cache stays inert.
    #[test]
    fn disabled_cache_noops() {
        let mut c = HotExpertCache::new(1, 4, 0);
        assert!(!c.refresh_due());
        let s = c.begin_step(true);
        c.record(0, &[1], s);
        assert!(c.refresh().is_empty());
        assert_eq!(c.budget(), 0);
    }

    /// Out-of-range layers and experts are safe no-ops.
    #[test]
    fn record_bounds_checks() {
        let mut c = HotExpertCache::new(2, 4, 1);
        let s = c.begin_step(true);
        c.record(9, &[0], s);
        c.record(0, &[99], s);
        c.record(0, &[1], s);
        let hot = c.refresh();
        assert_eq!(hot, vec![(0, 1)]);
    }
}
