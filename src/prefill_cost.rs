//! Online cost model for choosing the next prefill chunk.
//!
//! A prefill chunk costs a per-chunk term (the weights and kernels touched
//! once for the piece) plus a per-token term (attention, MoE routing, the
//! expert matmuls).  On architectures whose prefill re-reads the whole model
//! per chunk, a small chunk trades throughput for a bounded step; a large one
//! does the opposite.  With a time budget, the useful chunk is the largest
//! one that fits, which depends on the model, backend, and prompt prefix
//! length — not something a fixed constant can know.
//!
//! This model learns `seconds ≈ a·rows + b` from observed chunks with an
//! ordinary least-squares fit over a short recent window, solved for the rows
//! that fit a target.  It deliberately has no per-request state: concurrent
//! prefills feed the same model, and a change in cache pressure shows up as a
//! change in the fit.  Before two samples exist the caller probes with a
//! small chunk rather than guessing.
//!
//! The model only chooses a chunk size; it never changes tokens, attention
//! order, or accumulation order within a chunk.  Enabling it can change
//! floating-point results between runs for the same reason changing
//! `--prefill-chunk` can, so it is opt-in.

use std::collections::VecDeque;
use std::time::Duration;

/// Observations kept in the fit.  Enough to smooth noise, short enough to
/// follow a change in machine load.
const MAX_SAMPLES: usize = 32;

/// Smallest chunk the model selects, even for a tiny budget.  A one-token
/// chunk pays the per-chunk overhead for almost no progress.
pub const MIN_CHUNK: usize = 32;

/// Chunk used until the model has two observations.  Large enough to measure
/// the per-token term, small enough not to blow a latency budget on the first
/// step.
pub const PROBE_CHUNK: usize = 64;

#[derive(Debug)]
pub struct PrefillCostModel {
    min: usize,
    max: usize,
    samples: VecDeque<(f64, f64)>,
}

impl PrefillCostModel {
    /// Build a model whose choice is clamped to `min..=max`.
    pub fn new(min: usize, max: usize) -> Self {
        let max = max.max(1);
        Self {
            min: min.clamp(1, max),
            max,
            samples: VecDeque::with_capacity(MAX_SAMPLES),
        }
    }

    /// Record one completed chunk: `rows` tokens in `secs` seconds.
    ///
    /// Invalid or non-finite measurements are ignored rather than poisoning
    /// the fit (a suspended process or a clock adjustment can produce them).
    pub fn observe(&mut self, rows: usize, secs: f64) {
        if rows == 0 || !secs.is_finite() || secs <= 0.0 {
            return;
        }
        if self.samples.len() == MAX_SAMPLES {
            self.samples.pop_front();
        }
        self.samples.push_back((rows as f64, secs));
    }

    /// Rows to prefetch next.
    ///
    /// `fallback` is the configured chunk (also the ceiling when the model
    /// has no useful fit).  `remaining` bounds the choice to the rest of the
    /// prompt; a zero budget disables adaptation and returns `fallback`.
    pub fn next_chunk(&self, remaining: usize, budget: Duration, fallback: usize) -> usize {
        let cap = remaining.min(self.max).max(1);
        if remaining == 0 {
            return 0;
        }
        if budget.is_zero() {
            return fallback.clamp(1, cap);
        }
        if self.samples.len() < 2 {
            // Probe two *different* sizes: a fit needs two x values, and a
            // prompt whose suffix is a multiple of one probe size would
            // otherwise only ever see that size.  The first chunk stays
            // small so it cannot blow the budget before there is a fit.
            let first = fallback.min(PROBE_CHUNK).max(1);
            let probe = match self.samples.back() {
                None => first,
                Some(&(rows, _)) if rows as usize == first => {
                    (PROBE_CHUNK.saturating_mul(2)).max(first + 1)
                }
                Some(_) => PROBE_CHUNK,
            };
            return probe.min(fallback).clamp(self.min.min(cap), cap);
        }
        let Some(rows) = self.predict(budget.as_secs_f64()) else {
            return fallback.clamp(1, cap);
        };
        let rows = rows.floor().max(0.0) as usize;
        rows.clamp(self.min.min(cap), cap)
    }

    /// Solve the fitted line for the rows that take `budget` seconds.
    fn predict(&self, budget: f64) -> Option<f64> {
        if !(budget > 0.0) {
            return None;
        }
        let n = self.samples.len() as f64;
        let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
        for &(x, y) in &self.samples {
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        let denom = n * sxx - sx * sx;
        if denom.abs() < 1e-9 {
            // Every observation used the same chunk size: there is no slope,
            // but the average rate still turns a budget into a row count.
            // (Falling back to the ceiling here would overshoot a small
            // latency budget on every prompt with that shape.)
            let rate = sy / sx;
            return (rate > 0.0).then(|| budget / rate);
        }
        let slope = (n * sxy - sx * sy) / denom;
        let intercept = (sy - slope * sx) / n;
        if slope <= 1e-12 {
            // No measurable per-token cost (or noise dominates it).  Fall
            // back to the observed average rate, which still respects the
            // budget.
            let rate = sy / sx;
            return (rate > 0.0).then(|| budget / rate);
        }
        let rows = (budget - intercept) / slope;
        // A budget below the per-chunk intercept yields zero rows; the
        // caller clamps that to its floor rather than treating it as a
        // failed fit.
        rows.is_finite().then_some(rows.max(0.0))
    }

    /// Observations currently in the fit (diagnostics/tests).
    pub fn samples(&self) -> usize {
        self.samples.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_a_synthetic_line() {
        let mut model = PrefillCostModel::new(32, 512);
        // 1 ms/token + 10 ms per chunk.
        model.observe(100, 0.110);
        model.observe(200, 0.210);
        // 100 ms budget: 90 rows.
        assert_eq!(
            model.next_chunk(10_000, Duration::from_millis(100), 512),
            90
        );
    }

    #[test]
    fn first_chunks_probe_then_clamp() {
        let model = PrefillCostModel::new(32, 512);
        assert_eq!(model.next_chunk(10_000, Duration::from_millis(100), 512), 64);
        assert_eq!(model.next_chunk(20, Duration::from_millis(100), 512), 20);

        let mut model = PrefillCostModel::new(32, 128);
        model.observe(100, 0.110);
        model.observe(200, 0.210);
        // A huge budget cannot exceed the configured ceiling.
        assert_eq!(
            model.next_chunk(10_000, Duration::from_secs(10), 128),
            128
        );
        // A tiny budget cannot go below the floor.
        assert_eq!(
            model.next_chunk(10_000, Duration::from_millis(1), 128),
            32
        );
    }

    #[test]
    fn zero_budget_and_remaining_are_respected() {
        let model = PrefillCostModel::new(32, 512);
        assert_eq!(model.next_chunk(10_000, Duration::ZERO, 200), 200);
        assert_eq!(model.next_chunk(0, Duration::from_millis(100), 512), 0);
        assert_eq!(model.next_chunk(7, Duration::from_millis(100), 512), 7);
    }

    #[test]
    fn invalid_samples_are_ignored() {
        let mut model = PrefillCostModel::new(32, 512);
        model.observe(0, 0.1);
        model.observe(10, f64::NAN);
        model.observe(10, -1.0);
        assert_eq!(model.samples(), 0);
        assert_eq!(model.next_chunk(1000, Duration::from_millis(10), 128), 64);
    }

    #[test]
    fn probes_two_distinct_sizes_then_fits() {
        let mut model = PrefillCostModel::new(32, 512);
        assert_eq!(
            model.next_chunk(10_000, Duration::from_millis(100), 512),
            64
        );
        model.observe(64, 0.064);
        // The second probe must not repeat 64 rows.
        assert_eq!(
            model.next_chunk(10_000, Duration::from_millis(100), 512),
            128
        );
        model.observe(128, 0.128);
        // 1 ms/token: a 100 ms budget is 100 rows.
        assert_eq!(
            model.next_chunk(10_000, Duration::from_millis(100), 512),
            100
        );
    }

    #[test]
    fn equal_observations_use_the_average_rate() {
        let mut model = PrefillCostModel::new(32, 512);
        model.observe(64, 0.064);
        model.observe(64, 0.064);
        // No slope is defined, but ~1000 rows/s still respects the budget.
        assert_eq!(
            model.next_chunk(10_000, Duration::from_millis(100), 512),
            100
        );
    }

    #[test]
    fn noisy_negative_slope_uses_the_average_rate() {
        let mut model = PrefillCostModel::new(32, 512);
        model.observe(100, 0.2);
        model.observe(101, 0.1);
        // Average is ~1.5 ms/token, so 30 ms is about 20 rows, clamped to 32.
        assert_eq!(model.next_chunk(1000, Duration::from_millis(30), 512), 32);
    }
}
