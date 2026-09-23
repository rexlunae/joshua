//! Speculative token generation (draft-and-verify decoding).
//!
//! A single-token decode step is dominated by reading the weights (and, for a
//! sparse MoE model, faulting in the routed experts), not by arithmetic.
//! Speculative decoding exploits that: a cheap *drafter* proposes the next
//! `k` tokens, the target model scores all of them in **one** forward pass of
//! `k + 1` tokens, and every prefix of the draft that the target agrees with
//! is emitted at once.  A rejected token is replaced by a correction sampled
//! from the target, and the KV cache is truncated back to the accepted
//! prefix, so the output is exactly what plain decoding would produce:
//!
//! * greedy (`temperature <= 0`) — a draft token is accepted iff it is the
//!   target's argmax, so the text is token-for-token identical;
//! * sampled — standard speculative sampling (Leviathan et al. 2023, Chen et
//!   al. 2023) with a deterministic draft: accept draft token `x` with
//!   probability `p(x)`, otherwise sample from `p` with `x` removed.  The
//!   emitted tokens are distributed exactly as under ordinary sampling.
//!
//! The drafter here is **prompt lookup** ([`NgramDrafter`]): it finds the most
//! recent earlier occurrence of the context's trailing n-gram and proposes the
//! tokens that followed it.  It needs no draft model and costs a hash lookup
//! per step, and it is very effective on the agentic workloads joshua targets
//! — code edits, tool-call arguments and summaries that quote the prompt
//! repeat long spans verbatim.
//!
//! The engine enables this only for sessions that can both score every
//! position of a multi-token input and truncate their KV cache (see
//! `QuantizedModel::supports_speculative`); everything else keeps the plain
//! one-token loop.

use std::collections::HashMap;

use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;

use crate::error::{JoshuaError, Result};

/// Default maximum number of draft tokens verified per step.
pub const DEFAULT_MAX_DRAFT: usize = 8;
/// Default longest n-gram the drafter tries to match.
pub const DEFAULT_MAX_NGRAM: usize = 4;
/// Default shortest n-gram the drafter accepts as a match.
pub const DEFAULT_MIN_NGRAM: usize = 2;

/// Speculative decoding settings (see [`crate::EngineOptions::speculative`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeculativeConfig {
    /// Upper bound on draft tokens verified per forward pass.  The engine
    /// adapts the live draft length between 1 and this bound from the
    /// acceptance it observes.
    pub max_draft: usize,
    /// Longest trailing n-gram the drafter tries to match (tried first).
    pub max_ngram: usize,
    /// Shortest trailing n-gram accepted as a match.  `1` drafts more often
    /// at a lower acceptance rate.
    pub min_ngram: usize,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            max_draft: DEFAULT_MAX_DRAFT,
            max_ngram: DEFAULT_MAX_NGRAM,
            min_ngram: DEFAULT_MIN_NGRAM,
        }
    }
}

impl SpeculativeConfig {
    /// Settings with an explicit draft bound and the default n-gram range.
    pub fn with_max_draft(max_draft: usize) -> Self {
        Self {
            max_draft,
            ..Self::default()
        }
    }

    /// Clamp the settings into a usable range: at least one draft token and
    /// `1 <= min_ngram <= max_ngram`.
    pub fn normalized(self) -> Self {
        let min_ngram = self.min_ngram.max(1);
        Self {
            max_draft: self.max_draft.max(1),
            max_ngram: self.max_ngram.max(min_ngram),
            min_ngram,
        }
    }
}

/// Aggregate speculative-decoding counters (see
/// [`crate::Engine::speculative_stats`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpeculativeStats {
    /// Draft tokens sent to the target model for verification.
    pub drafted: u64,
    /// Draft tokens the target model accepted.
    pub accepted: u64,
    /// Multi-token verification passes run.
    pub verify_steps: u64,
}

impl SpeculativeStats {
    /// Fraction of drafted tokens that were accepted (0 when nothing was
    /// drafted).
    pub fn acceptance_rate(&self) -> f64 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f64 / self.drafted as f64
        }
    }
}

// ─── Drafter ────────────────────────────────────────────────────────────────

/// Prompt-lookup drafter: proposes the continuation of the most recent
/// earlier occurrence of the context's trailing n-gram.
///
/// The context is the prompt followed by every emitted token.  An index from
/// each n-gram (for every `n` in `min_ngram..=max_ngram`) to the position
/// just past its latest occurrence is maintained incrementally, so drafting
/// costs `O(max_ngram)` hash lookups per step instead of a scan of the
/// context.  The n-gram ending at the very end of the context is indexed only
/// once another token follows it, so a lookup never finds the suffix itself.
pub struct NgramDrafter {
    config: SpeculativeConfig,
    tokens: Vec<u32>,
    /// One map per n-gram length (`index[n - min_ngram]`): n-gram hash →
    /// end (exclusive) of its latest occurrence.
    index: Vec<HashMap<u64, usize>>,
    /// Number of end positions indexed so far (positions `1..=indexed`).
    indexed: usize,
}

impl NgramDrafter {
    /// A drafter over `context` (typically the prompt tokens).
    pub fn new(config: SpeculativeConfig, context: &[u32]) -> Self {
        let config = config.normalized();
        let n_maps = config.max_ngram - config.min_ngram + 1;
        Self {
            config,
            tokens: context.to_vec(),
            index: vec![HashMap::new(); n_maps],
            indexed: 0,
        }
    }

    /// Append an emitted token to the context.
    pub fn push(&mut self, token: u32) {
        self.tokens.push(token);
    }

    /// The context the drafter matches against.
    pub fn context(&self) -> &[u32] {
        &self.tokens
    }

    /// Propose `max_tokens` tokens continuing the context, or none when no
    /// trailing n-gram of at least `min_ngram` tokens has occurred before.
    pub fn draft(&mut self, max_tokens: usize) -> Vec<u32> {
        if max_tokens == 0 {
            return Vec::new();
        }
        self.index_pending();
        let len = self.tokens.len();
        let cfg = self.config;
        for n in (cfg.min_ngram..=cfg.max_ngram).rev() {
            if n > len {
                continue;
            }
            let suffix = &self.tokens[len - n..];
            let Some(&end) = self.index[n - cfg.min_ngram].get(&hash_ngram(suffix)) else {
                continue;
            };
            // Guard against hash collisions: the stored occurrence must
            // really be this n-gram.
            if end < n || end >= len || self.tokens[end - n..end] != *suffix {
                continue;
            }
            // Continue from the match.  When the continuation runs into the
            // end of the context it carries on through the draft itself, so
            // a periodic tail (`x y x y` → `x y x y …`) drafts a full run
            // instead of a single token.
            let mut draft = Vec::with_capacity(max_tokens);
            for j in 0..max_tokens {
                let i = end + j;
                let t = if i < len { self.tokens[i] } else { draft[i - len] };
                draft.push(t);
            }
            return draft;
        }
        Vec::new()
    }

    /// Index every n-gram ending before the last context position.
    fn index_pending(&mut self) {
        let len = self.tokens.len();
        // End positions `1..len` are safe to index; `len` itself (the
        // suffix being looked up) is not.
        while self.indexed + 1 < len {
            let end = self.indexed + 1;
            for n in self.config.min_ngram..=self.config.max_ngram {
                if n > end {
                    break;
                }
                let h = hash_ngram(&self.tokens[end - n..end]);
                // Later occurrences overwrite earlier ones: the most recent
                // match is the best predictor of what comes next.
                self.index[n - self.config.min_ngram].insert(h, end);
            }
            self.indexed = end;
        }
    }
}

/// FNV-1a over the token ids.
fn hash_ngram(tokens: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &t in tokens {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

// ─── Verification ───────────────────────────────────────────────────────────

/// The target model's next-token distribution after the request's sampling
/// transforms (repetition penalty, temperature, top-k, min-p, top-p).
#[derive(Debug, Clone, PartialEq)]
pub enum TokenDist {
    /// Deterministic choice (greedy decoding, or a degenerate distribution).
    Greedy(u32),
    /// Normalised probabilities over the vocabulary.
    Probs(Vec<f32>),
}

impl TokenDist {
    /// Draw a token.
    pub fn sample(&self, rng: &mut impl Rng) -> Result<u32> {
        match self {
            Self::Greedy(t) => Ok(*t),
            Self::Probs(p) => sample_probs(p, rng),
        }
    }
}

/// Outcome of verifying one draft token against the target distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The draft token is emitted as is.
    Accept,
    /// The draft token is rejected; emit this correction instead and discard
    /// the rest of the draft.
    Reject(u32),
}

/// Verify a (deterministically drafted) token against the target's
/// distribution with speculative sampling.
///
/// The draft distribution is a point mass on `draft`, so the acceptance
/// probability `min(1, p(x) / q(x))` is just `p(draft)`, and on rejection
/// the residual `max(0, p - q)` is `p` with `draft` removed.  Together they
/// emit exactly one token distributed as `p`.
pub fn verify_token(dist: &TokenDist, draft: u32, rng: &mut impl Rng) -> Result<Verdict> {
    match dist {
        TokenDist::Greedy(t) => Ok(if *t == draft {
            Verdict::Accept
        } else {
            Verdict::Reject(*t)
        }),
        TokenDist::Probs(p) => {
            let p_draft = p.get(draft as usize).copied().unwrap_or(0.0);
            if p_draft > 0.0 && rng.gen::<f32>() < p_draft {
                return Ok(Verdict::Accept);
            }
            let mut residual = p.clone();
            if let Some(x) = residual.get_mut(draft as usize) {
                *x = 0.0;
            }
            if residual.iter().sum::<f32>() <= 0.0 {
                // Only reachable through rounding when p(draft) ≈ 1.
                return Ok(Verdict::Accept);
            }
            Ok(Verdict::Reject(sample_probs(&residual, rng)?))
        }
    }
}

fn sample_probs(p: &[f32], rng: &mut impl Rng) -> Result<u32> {
    let dist = WeightedIndex::new(p).map_err(|e| JoshuaError::Inference(e.to_string()))?;
    Ok(dist.sample(rng) as u32)
}

/// Adapt the live draft length to the acceptance just observed: grow after a
/// fully accepted draft, shrink after a fully rejected one.
pub fn next_draft_len(current: usize, drafted: usize, accepted: usize, max: usize) -> usize {
    if drafted == 0 {
        current
    } else if accepted == drafted {
        (current * 2).min(max).max(1)
    } else if accepted == 0 {
        (current / 2).max(1)
    } else {
        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn cfg(max_ngram: usize, min_ngram: usize) -> SpeculativeConfig {
        SpeculativeConfig {
            max_draft: 8,
            max_ngram,
            min_ngram,
        }
    }

    #[test]
    fn drafts_continuation_of_earlier_ngram() {
        // "1 2 3 4 5 ... 1 2 3" → propose "4 5 9".
        let mut d = NgramDrafter::new(cfg(3, 2), &[1, 2, 3, 4, 5, 9, 7, 1, 2, 3]);
        assert_eq!(d.draft(3), vec![4, 5, 9]);
        assert_eq!(d.draft(1), vec![4]);
    }

    #[test]
    fn no_draft_without_a_match() {
        let mut d = NgramDrafter::new(cfg(3, 2), &[1, 2, 3, 4, 5]);
        assert!(d.draft(4).is_empty());
        // A lone unigram match is below min_ngram.
        let mut d = NgramDrafter::new(cfg(3, 2), &[7, 1, 8, 1]);
        assert!(d.draft(4).is_empty());
        // ...but is enough with min_ngram = 1.
        let mut d = NgramDrafter::new(cfg(3, 1), &[7, 1, 8, 1]);
        assert_eq!(d.draft(4), vec![8, 1, 8, 1]);
    }

    #[test]
    fn prefers_most_recent_and_longest_match() {
        // [5 6] occurs twice; the later occurrence is followed by 8.
        let mut d = NgramDrafter::new(cfg(2, 2), &[5, 6, 7, 5, 6, 8, 0, 5, 6]);
        assert_eq!(d.draft(1), vec![8]);
        // Longest match wins: [3 5 6] only occurs once, followed by 7.
        let mut d = NgramDrafter::new(cfg(3, 2), &[3, 5, 6, 7, 5, 6, 8, 0, 3, 5, 6]);
        assert_eq!(d.draft(1), vec![7]);
    }

    #[test]
    fn index_tracks_pushed_tokens() {
        let mut d = NgramDrafter::new(cfg(2, 2), &[1, 2, 3]);
        assert!(d.draft(2).is_empty());
        d.push(1);
        d.push(2);
        assert_eq!(d.draft(2), vec![3, 1]);
        // A repeating pattern drafts through its own tail.
        let mut d = NgramDrafter::new(cfg(2, 2), &[4, 4, 4]);
        assert_eq!(d.draft(5), vec![4; 5]);
        let mut d = NgramDrafter::new(cfg(2, 2), &[9, 1, 2, 1, 2]);
        assert_eq!(d.draft(5), vec![1, 2, 1, 2, 1]);
    }

    #[test]
    fn draft_is_capped() {
        let mut d = NgramDrafter::new(cfg(2, 2), &[1, 2, 3, 4, 5, 6, 1, 2]);
        assert_eq!(d.draft(2), vec![3, 4]);
        assert!(d.draft(0).is_empty());
    }

    #[test]
    fn config_normalizes() {
        let c = SpeculativeConfig {
            max_draft: 0,
            max_ngram: 1,
            min_ngram: 0,
        }
        .normalized();
        assert_eq!(c.max_draft, 1);
        assert_eq!(c.min_ngram, 1);
        assert_eq!(c.max_ngram, 1);
    }

    #[test]
    fn greedy_verification_matches_argmax() {
        let mut rng = StdRng::seed_from_u64(0);
        let d = TokenDist::Greedy(3);
        assert_eq!(verify_token(&d, 3, &mut rng).unwrap(), Verdict::Accept);
        assert_eq!(verify_token(&d, 4, &mut rng).unwrap(), Verdict::Reject(3));
    }

    #[test]
    fn zero_probability_draft_is_always_rejected() {
        let mut rng = StdRng::seed_from_u64(1);
        let d = TokenDist::Probs(vec![0.5, 0.5, 0.0]);
        for _ in 0..100 {
            match verify_token(&d, 2, &mut rng).unwrap() {
                Verdict::Reject(t) => assert!(t < 2),
                Verdict::Accept => panic!("accepted a zero-probability token"),
            }
        }
    }

    #[test]
    fn speculative_sampling_preserves_the_distribution() {
        // Whatever the draft, the emitted token must be distributed as p.
        let p = vec![0.1_f32, 0.6, 0.3];
        let d = TokenDist::Probs(p.clone());
        let mut rng = StdRng::seed_from_u64(42);
        let trials = 200_000;
        for draft in 0..3u32 {
            let mut counts = [0usize; 3];
            for _ in 0..trials {
                let t = match verify_token(&d, draft, &mut rng).unwrap() {
                    Verdict::Accept => draft,
                    Verdict::Reject(t) => {
                        assert_ne!(t, draft);
                        t
                    }
                };
                counts[t as usize] += 1;
            }
            for (i, &c) in counts.iter().enumerate() {
                let freq = c as f32 / trials as f32;
                assert!(
                    (freq - p[i]).abs() < 0.01,
                    "draft {draft}: token {i} freq {freq} vs p {}",
                    p[i]
                );
            }
        }
    }

    #[test]
    fn draft_length_adapts() {
        assert_eq!(next_draft_len(4, 4, 4, 8), 8);
        assert_eq!(next_draft_len(8, 8, 8, 8), 8);
        assert_eq!(next_draft_len(4, 4, 0, 8), 2);
        assert_eq!(next_draft_len(1, 1, 0, 8), 1);
        assert_eq!(next_draft_len(4, 4, 2, 8), 4);
        assert_eq!(next_draft_len(4, 0, 0, 8), 4);
    }

    #[test]
    fn stats_acceptance_rate() {
        assert_eq!(SpeculativeStats::default().acceptance_rate(), 0.0);
        let s = SpeculativeStats {
            drafted: 10,
            accepted: 7,
            verify_steps: 3,
        };
        assert!((s.acceptance_rate() - 0.7).abs() < 1e-9);
    }
}
