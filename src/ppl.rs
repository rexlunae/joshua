//! Cross-entropy / perplexity scoring of a token stream.
//!
//! Every other number joshua reports is about *speed* or *memory*. This module
//! is about *quality*: given a model and a corpus, how surprised is the model
//! by the next token? For a compressed checkpoint (a pruned expert pool, a
//! re-quantised table) that is the measurement that tells you what the
//! compression cost — and it is the only one that compares a derived model
//! against its source on equal terms.
//!
//! The convention is llama.cpp's: split the token stream into non-overlapping
//! windows, and inside each window score every token except the first, because
//! position `i` predicts token `i + 1`. The first token of a window has no
//! preceding context *within that window*, so scoring it would charge the model
//! for a cold start. With non-overlapping windows each token is therefore
//! scored exactly once, except the first token of each window.
//!
//! Nothing here touches a model: windowing, log-softmax and the running
//! accumulation are pure, so they are unit-tested without loading weights.

/// Token windows `[start, end)` covering `n` tokens, `window` wide.
///
/// A trailing shorter window is kept (dropping it would silently discard part
/// of the corpus). `window` is clamped to at least 2 so a window always holds
/// one scored token.
pub fn windows(n: usize, window: usize) -> Vec<(usize, usize)> {
    let window = window.max(2);
    let mut out = Vec::new();
    let mut start = 0;
    while start < n {
        let end = (start + window).min(n);
        if end - start >= 2 {
            out.push((start, end));
        }
        start = end;
    }
    out
}

/// How many logit rows one forward pass may materialise while scoring.
///
/// 256 rows of a 129k-token vocabulary is ~132 MB of `f32`: enough to amortise
/// each pass over the weights, while keeping the logits buffer bounded no matter
/// how wide the scored window is.
pub const SCORE_ROWS_PER_PASS: usize = 256;

/// One forward pass inside a scored window.
///
/// A window is fed in slices so the logits of a whole window never have to be
/// materialised at once: a 4096-token window over a 129k vocabulary is 2.1 GB
/// of `f32` rows. Slices are contiguous — the KV cache carries across them, so
/// the model still sees the full window context — and only the *materialised
/// rows* are bounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slice {
    /// First token of this slice (absolute, in the scored token stream).
    pub from: usize,
    /// One past the last token fed in this slice.
    pub to: usize,
    /// Position to feed `from` at, relative to the start of the window.
    pub pos: usize,
    /// One past the last token that can be scored from this slice's rows.
    ///
    /// Usually `to + 1`, because the last row predicts the first token of the
    /// next slice — that token is known, so the row is scoreable. At the end of
    /// a window it is `to`: the final row has no successor inside the window.
    pub target_end: usize,
}

/// Split the window `[start, end)` into forward passes of at most `max_rows`
/// tokens each.
pub fn slices(start: usize, end: usize, max_rows: usize) -> Vec<Slice> {
    let max_rows = max_rows.max(1);
    let mut out = Vec::new();
    let mut from = start;
    while from < end {
        let to = (from + max_rows).min(end);
        // A slice that does not reach the window end can still score its last
        // row, because the next slice's first token is its target.
        let target_end = if to < end { to + 1 } else { to };
        out.push(Slice {
            from,
            to,
            pos: from - start,
            target_end,
        });
        from = to;
    }
    out
}

/// Negative log-likelihood of `target` under `logits`, in nats.
///
/// `None` when the row cannot produce a finite value: an empty/one-class
/// distribution, a masked (`-inf`) target, or a NaN anywhere in the row. A
/// caller that silently substituted a number for those would report a
/// flattering perplexity, so the accumulator counts them separately instead.
pub fn nll(logits: &[f32], target: u32) -> Option<f64> {
    let t = *logits.get(target as usize)? as f64;
    if !t.is_finite() || logits.is_empty() {
        return None;
    }
    let mut max = f64::NEG_INFINITY;
    for &v in logits {
        let v = v as f64;
        if v.is_nan() {
            return None;
        }
        if v > max {
            max = v;
        }
    }
    if !max.is_finite() {
        return None;
    }
    let mut sum = 0.0f64;
    for &v in logits {
        sum += (v as f64 - max).exp();
    }
    if !(sum > 0.0) || !sum.is_finite() {
        return None;
    }
    let log_z = max + sum.ln();
    let nll = log_z - t;
    nll.is_finite().then_some(nll)
}

/// Running negative log-likelihood over scored tokens.
#[derive(Debug, Default, Clone)]
pub struct Accumulator {
    nll: f64,
    scored: u64,
    skipped: u64,
}

impl Accumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Score one window: `logits[i]` predicts `tokens[i + 1]`.
    ///
    /// Returns the number of tokens this window contributed, so a caller can
    /// report progress without re-deriving the windowing.
    pub fn push_window(&mut self, tokens: &[u32], logits: &[Vec<f32>]) -> u64 {
        let n = tokens.len().saturating_sub(1).min(logits.len());
        let mut added = 0;
        for i in 0..n {
            match nll(&logits[i], tokens[i + 1]) {
                Some(v) => {
                    self.nll += v;
                    self.scored += 1;
                    added += 1;
                }
                None => self.skipped += 1,
            }
        }
        added
    }

    /// Tokens scored (denominator of the mean).
    pub fn scored(&self) -> u64 {
        self.scored
    }

    /// Tokens a window produced logits for but could not be scored.
    pub fn skipped(&self) -> u64 {
        self.skipped
    }

    /// Mean negative log-likelihood in nats per token.
    pub fn mean_nll(&self) -> Option<f64> {
        (self.scored > 0).then(|| self.nll / self.scored as f64)
    }

    /// Perplexity, `exp(mean NLL)`.
    ///
    /// Saturates at [`f64::INFINITY`] rather than overflowing to NaN for a
    /// catastrophically bad model (mean NLL above ~709).
    pub fn perplexity(&self) -> Option<f64> {
        self.mean_nll().map(|m| m.exp())
    }

    /// Bits per token — the same measurement in the units a compression
    /// argument is usually made in (`ppl = 2^bpt`).
    pub fn bits_per_token(&self) -> Option<f64> {
        self.mean_nll().map(|m| m / std::f64::consts::LN_2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Logits are f32, so a value derived from them only matches an f64
    /// reference to ~1e-7; 1e-6 keeps the comparisons meaningful without
    /// pretending f32 arithmetic is exact.
    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn windows_cover_every_token_exactly_once() {
        let w = windows(25, 10);
        assert_eq!(w, vec![(0, 10), (10, 20), (20, 25)]);
        // contiguous, no overlap, no gaps
        for pair in w.windows(2) {
            assert_eq!(pair[0].1, pair[1].0);
        }
        assert_eq!(w.last().unwrap().1, 25);
    }

    #[test]
    fn windows_drop_a_single_leftover_token() {
        // A 1-token tail has nothing to score (position i predicts i+1), so it
        // is not a window at all.
        assert_eq!(windows(21, 10), vec![(0, 10), (10, 20)]);
        assert!(windows(1, 10).is_empty());
        assert!(windows(0, 10).is_empty());
    }

    #[test]
    fn windows_clamp_a_degenerate_width() {
        // Width 0/1 is clamped to 2, and the 1-token tail is still dropped.
        assert_eq!(windows(5, 0), vec![(0, 2), (2, 4)]);
        assert_eq!(windows(5, 1), vec![(0, 2), (2, 4)]);
        assert_eq!(windows(6, 0), vec![(0, 2), (2, 4), (4, 6)]);
    }

    #[test]
    fn slices_cover_the_window_contiguously_with_growing_positions() {
        let s = slices(100, 350, 100);
        assert_eq!(s.len(), 3);
        assert_eq!(s[0], Slice { from: 100, to: 200, pos: 0, target_end: 201 });
        assert_eq!(s[1], Slice { from: 200, to: 300, pos: 100, target_end: 301 });
        // Last slice: no successor inside the window, so nothing extra to score.
        assert_eq!(s[2], Slice { from: 300, to: 350, pos: 200, target_end: 350 });
        // Contiguous, and every slice except the last hands its last row a target.
        for pair in s.windows(2) {
            assert_eq!(pair[0].to, pair[1].from);
            assert_eq!(pair[0].target_end, pair[1].from + 1);
        }
    }

    #[test]
    fn slices_bound_the_rows_materialised_per_pass() {
        // A 4096-token window at 256 rows/pass is 16 passes, never one 2 GiB
        // logits buffer.
        let s = slices(0, 4096, 256);
        assert_eq!(s.len(), 16);
        assert!(s.iter().all(|x| x.to - x.from == 256));
        // A window smaller than the budget stays a single pass.
        assert_eq!(slices(0, 10, 256), vec![Slice { from: 0, to: 10, pos: 0, target_end: 10 }]);
    }

    #[test]
    fn slices_degenerate_budget_still_makes_progress() {
        // A zero budget is clamped to one row per pass; a non-final slice still
        // hands its last row a target (the next slice's first token).
        assert_eq!(slices(0, 3, 0).len(), 3);
        assert_eq!(slices(0, 3, 0)[0], Slice { from: 0, to: 1, pos: 0, target_end: 2 });
        assert_eq!(slices(0, 3, 0)[2], Slice { from: 2, to: 3, pos: 2, target_end: 3 });
        assert!(slices(5, 5, 8).is_empty());
    }

    #[test]
    fn sliced_scoring_matches_one_shot_scoring() {
        // The point of slicing is that it must not change the score: score the
        // same 4-token window as one pass and as 2-row slices, feeding the
        // concatenation of rows to the accumulator either way.
        let tokens = [1u32, 2, 3, 4];
        let vocab = 5;
        let rows: Vec<Vec<f32>> = (0..4)
            .map(|i| {
                let mut r = vec![0.0f32; vocab];
                r[(i + 1) % vocab] = 5.0;      // strongly predict token i+1
                r
            })
            .collect();

        let mut one = Accumulator::new();
        one.push_window(&tokens, &rows);

        let mut sliced = Accumulator::new();
        for s in slices(0, tokens.len(), 2) {
            sliced.push_window(&tokens[s.from..s.target_end], &rows[s.from..s.to]);
        }
        assert_eq!(one.scored(), sliced.scored());
        assert_eq!(one.skipped(), sliced.skipped());
        assert!(close(one.mean_nll().unwrap(), sliced.mean_nll().unwrap()));
    }

    #[test]
    fn nll_matches_a_hand_computed_log_softmax() {
        // logits [0, ln 3] over two classes: p0 = 1/4, p1 = 3/4.
        let logits = [0.0f32, (3.0f64).ln() as f32];
        assert!(close(nll(&logits, 0).unwrap(), (4.0f64).ln()));
        assert!(close(nll(&logits, 1).unwrap(), (4.0f64 / 3.0).ln()));
    }

    #[test]
    fn nll_of_a_certain_prediction_is_zero_and_of_an_impossible_one_is_inf() {
        // A massively dominant logit: NLL → 0.
        assert!(nll(&[100.0, 0.0], 0).unwrap() < 1e-30);
        // A masked-out (-inf) target cannot be scored at all.
        assert!(nll(&[0.0, f32::NEG_INFINITY], 1).is_none());
        // Non-finite elsewhere in the row poisons the normaliser too.
        assert!(nll(&[f32::NAN, 0.0], 0).is_none());
        // Out-of-range target.
        assert!(nll(&[0.0, 0.0], 5).is_none());
    }

    #[test]
    fn accumulator_scores_all_but_the_first_token_of_a_window() {
        let mut acc = Accumulator::new();
        // Token ids must be inside the logit vocabulary (4 wide here).
        let tokens = [1u32, 2, 3];
        // Uniform logits over a 4-token vocab: every scored token costs ln 4.
        let logits = vec![vec![0.0f32; 4]; 3];
        assert_eq!(acc.push_window(&tokens, &logits), 2);
        assert_eq!(acc.scored(), 2);
        assert_eq!(acc.skipped(), 0);
        assert!(close(acc.mean_nll().unwrap(), (4.0f64).ln()));
        assert!(close(acc.perplexity().unwrap(), 4.0));
        assert!(close(acc.bits_per_token().unwrap(), 2.0));
    }

    #[test]
    fn accumulator_ignores_surplus_logit_rows_and_counts_unscorable_tokens() {
        let mut acc = Accumulator::new();
        // Three logit rows for two tokens: only row 0 can be used (it predicts
        // token 1), and it is masked, so the token is counted as skipped
        // rather than scored at some invented value.
        let logits = vec![vec![f32::NEG_INFINITY; 3]; 3];
        assert_eq!(acc.push_window(&[1u32, 2], &logits), 0);
        assert_eq!(acc.scored(), 0);
        assert_eq!(acc.skipped(), 1);
        assert!(acc.perplexity().is_none());
    }

    #[test]
    fn accumulator_pools_windows_into_one_estimate() {
        let mut acc = Accumulator::new();
        // Two 2-token-vocab windows: one certain (logit 1e4 on the target makes
        // the residual e^-1e4 exactly 0 in f64), one flat (ln 2).
        acc.push_window(&[0, 0], &[vec![1e4, 0.0]]);
        acc.push_window(&[0, 1], &[vec![0.0, 0.0]]);
        assert_eq!(acc.scored(), 2);
        let mean = acc.mean_nll().unwrap();
        assert!(close(mean, (2.0f64).ln() / 2.0), "mean {mean}");
        assert!(close(acc.perplexity().unwrap(), mean.exp()));
        assert!(close(acc.perplexity().unwrap(), (2.0f64).sqrt()));
    }

    #[test]
    fn perplexity_rejects_an_empty_corpus() {
        let acc = Accumulator::new();
        assert!(acc.perplexity().is_none());
        assert!(acc.mean_nll().is_none());
        assert!(acc.bits_per_token().is_none());
    }
}
