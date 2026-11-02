//! Sampling defaults per API surface, the GGUF `general.sampling.*`
//! overrides, and the llama.cpp samplers the engine adds to its own
//! temperature / top-k / min-p / top-p chain.
//!
//! # Defaults
//!
//! A request falls back, field by field, to:
//!
//! 1. the server's command-line sampling flags (`--temp`, `--top-k`, …);
//! 2. the model's `general.sampling.*` GGUF metadata (the values the model's
//!    publisher recommends, read the way llama.cpp's
//!    `common_init_sampler_from_model` reads them);
//! 3. the upstream defaults of the API it came in through
//!    ([`SamplingProfile`]): OpenAI / vLLM / Anthropic routes use neutral
//!    sampling, llama.cpp-native routes llama.cpp's `common_params_sampling`
//!    defaults and Ollama routes Ollama's `api.DefaultOptions`.
//!
//! [`crate::Engine::sampling_defaults`] resolves the three layers.
//!
//! # Extra samplers
//!
//! Typical-p, top-n-sigma, XTC, Mirostat (v1 and v2) and DRY follow
//! llama.cpp's `src/llama-sampler.cpp`.  Each is off by default, and an off
//! sampler never touches the logits, so a request that sets none of them
//! samples exactly as before.

use std::collections::{HashMap, VecDeque};

use candle_core::quantized::gguf_file;
use serde::{Deserialize, Serialize};

/// The API whose upstream sampling defaults a request falls back to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingProfile {
    /// OpenAI, vLLM and Anthropic: temperature 1.0, top-p 1.0, no top-k or
    /// min-p, no repetition penalty (vLLM's `_DEFAULT_SAMPLING_PARAMS`).
    OpenAi,
    /// llama-server's native routes: `common_params_sampling` in llama.cpp's
    /// `common/common.h` (temperature 0.8, top-k 40, top-p 0.95, min-p 0.05).
    LlamaCpp,
    /// Ollama's `api.DefaultOptions` (temperature 0.8, top-k 40, top-p 0.9).
    Ollama,
}

/// Fully resolved sampler settings a request uses for every field it leaves
/// unset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingDefaults {
    /// Temperature (0 = greedy).
    pub temperature: f32,
    /// Top-k (0 = off).
    pub top_k: i32,
    /// Nucleus threshold (1.0 = off).
    pub top_p: f32,
    /// Min-p (0 = off).
    pub min_p: f32,
    /// Locally typical sampling (1.0 = off).
    pub typical_p: f32,
    /// Multiplicative repetition penalty (1.0 = off).
    pub repetition_penalty: f32,
    /// Penalty window in tokens (0 = off, -1 = the context window).
    pub repeat_last_n: i32,
    /// Additive presence penalty.
    pub presence_penalty: f32,
    /// Additive frequency penalty.
    pub frequency_penalty: f32,
    /// XTC trigger probability (0 = off).
    pub xtc_probability: f32,
    /// XTC probability threshold (> 0.5 = off).
    pub xtc_threshold: f32,
    /// Mirostat version (0 = off, 1, 2).
    pub mirostat: u8,
    /// Mirostat target entropy.
    pub mirostat_tau: f32,
    /// Mirostat learning rate.
    pub mirostat_eta: f32,
}

impl SamplingProfile {
    /// The API's own defaults, before GGUF metadata or server flags.
    pub fn defaults(self) -> SamplingDefaults {
        // The fields every profile shares: llama.cpp and Ollama agree on
        // them, and the OpenAI-shaped APIs have no such samplers.
        let base = SamplingDefaults {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            typical_p: 1.0,
            repetition_penalty: 1.0,
            repeat_last_n: 64,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            xtc_probability: 0.0,
            xtc_threshold: 0.1,
            mirostat: 0,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
        };
        match self {
            Self::OpenAi => base,
            Self::LlamaCpp => SamplingDefaults {
                temperature: 0.8,
                top_k: 40,
                top_p: 0.95,
                min_p: 0.05,
                ..base
            },
            Self::Ollama => SamplingDefaults {
                temperature: 0.8,
                top_k: 40,
                top_p: 0.9,
                ..base
            },
        }
    }
}

/// A partial set of sampler settings layered over a profile's defaults:
/// the server's command-line flags, or a model's `general.sampling.*`
/// metadata.  `None` leaves the value underneath in place.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SamplingOverrides {
    /// `--temp` / `general.sampling.temp`.
    pub temperature: Option<f32>,
    /// `--top-k` / `general.sampling.top_k`.
    pub top_k: Option<i32>,
    /// `--top-p` / `general.sampling.top_p`.
    pub top_p: Option<f32>,
    /// `--min-p` / `general.sampling.min_p`.
    pub min_p: Option<f32>,
    /// `--typical`.
    pub typical_p: Option<f32>,
    /// `--repeat-penalty` / `general.sampling.penalty_repeat`.
    pub repetition_penalty: Option<f32>,
    /// `--repeat-last-n` / `general.sampling.penalty_last_n`.
    pub repeat_last_n: Option<i32>,
    /// `--presence-penalty`.
    pub presence_penalty: Option<f32>,
    /// `--frequency-penalty`.
    pub frequency_penalty: Option<f32>,
    /// `--xtc-probability` / `general.sampling.xtc_probability`.
    pub xtc_probability: Option<f32>,
    /// `--xtc-threshold` / `general.sampling.xtc_threshold`.
    pub xtc_threshold: Option<f32>,
    /// `--mirostat` / `general.sampling.mirostat`.
    pub mirostat: Option<u8>,
    /// `--mirostat-ent` / `general.sampling.mirostat_tau`.
    pub mirostat_tau: Option<f32>,
    /// `--mirostat-lr` / `general.sampling.mirostat_eta`.
    pub mirostat_eta: Option<f32>,
}

/// GGUF keys of the recommended sampling settings (llama.cpp's
/// `LLM_KV_GENERAL_SAMPLING_*`, gguf-py `Keys.General.SAMPLING_*`).
pub const GGUF_SAMPLING_TEMP: &str = "general.sampling.temp";
/// See [`GGUF_SAMPLING_TEMP`].
pub const GGUF_SAMPLING_TOP_K: &str = "general.sampling.top_k";
/// See [`GGUF_SAMPLING_TEMP`].
pub const GGUF_SAMPLING_TOP_P: &str = "general.sampling.top_p";
/// See [`GGUF_SAMPLING_TEMP`].
pub const GGUF_SAMPLING_MIN_P: &str = "general.sampling.min_p";
/// See [`GGUF_SAMPLING_TEMP`].
pub const GGUF_SAMPLING_XTC_PROBABILITY: &str = "general.sampling.xtc_probability";
/// See [`GGUF_SAMPLING_TEMP`].
pub const GGUF_SAMPLING_XTC_THRESHOLD: &str = "general.sampling.xtc_threshold";
/// See [`GGUF_SAMPLING_TEMP`].
pub const GGUF_SAMPLING_PENALTY_LAST_N: &str = "general.sampling.penalty_last_n";
/// See [`GGUF_SAMPLING_TEMP`].
pub const GGUF_SAMPLING_PENALTY_REPEAT: &str = "general.sampling.penalty_repeat";
/// See [`GGUF_SAMPLING_TEMP`].
pub const GGUF_SAMPLING_MIROSTAT: &str = "general.sampling.mirostat";
/// See [`GGUF_SAMPLING_TEMP`].
pub const GGUF_SAMPLING_MIROSTAT_TAU: &str = "general.sampling.mirostat_tau";
/// See [`GGUF_SAMPLING_TEMP`].
pub const GGUF_SAMPLING_MIROSTAT_ETA: &str = "general.sampling.mirostat_eta";

/// A GGUF metadata value as a number.  llama.cpp reads these keys through
/// `llama_model_meta_val_str` and parses the text, so any numeric type —
/// and a numeric string — is accepted.
fn gguf_number(value: &gguf_file::Value) -> Option<f64> {
    use gguf_file::Value as V;
    Some(match value {
        V::U8(v) => *v as f64,
        V::I8(v) => *v as f64,
        V::U16(v) => *v as f64,
        V::I16(v) => *v as f64,
        V::U32(v) => *v as f64,
        V::I32(v) => *v as f64,
        V::U64(v) => *v as f64,
        V::I64(v) => *v as f64,
        V::F32(v) => *v as f64,
        V::F64(v) => *v,
        V::Bool(v) => *v as u8 as f64,
        V::String(s) => s.trim().parse().ok()?,
        V::Array(_) => return None,
    })
    .filter(|v: &f64| v.is_finite())
}

impl SamplingOverrides {
    /// The `general.sampling.*` values a GGUF carries.  `general.sampling
    /// .sequence` (llama.cpp's sampler order) is not read: the engine's
    /// chain has a fixed order.
    pub fn from_gguf(metadata: &HashMap<String, gguf_file::Value>) -> Self {
        let float = |key: &str| metadata.get(key).and_then(gguf_number).map(|v| v as f32);
        let int = |key: &str| {
            metadata
                .get(key)
                .and_then(gguf_number)
                .filter(|v| v.fract() == 0.0 && (i32::MIN as f64..=i32::MAX as f64).contains(v))
                .map(|v| v as i32)
        };
        Self {
            temperature: float(GGUF_SAMPLING_TEMP),
            top_k: int(GGUF_SAMPLING_TOP_K),
            top_p: float(GGUF_SAMPLING_TOP_P),
            min_p: float(GGUF_SAMPLING_MIN_P),
            typical_p: None,
            repetition_penalty: float(GGUF_SAMPLING_PENALTY_REPEAT),
            repeat_last_n: int(GGUF_SAMPLING_PENALTY_LAST_N),
            presence_penalty: None,
            frequency_penalty: None,
            xtc_probability: float(GGUF_SAMPLING_XTC_PROBABILITY),
            xtc_threshold: float(GGUF_SAMPLING_XTC_THRESHOLD),
            mirostat: int(GGUF_SAMPLING_MIROSTAT)
                .filter(|v| (0..=2).contains(v))
                .map(|v| v as u8),
            mirostat_tau: float(GGUF_SAMPLING_MIROSTAT_TAU),
            mirostat_eta: float(GGUF_SAMPLING_MIROSTAT_ETA),
        }
    }

    /// Whether any value is set.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// Replace each value in `defaults` this override sets.
    pub fn apply(&self, defaults: &mut SamplingDefaults) {
        macro_rules! layer {
            ($($field:ident),*) => {
                $(if let Some(v) = self.$field {
                    defaults.$field = v;
                })*
            };
        }
        layer!(
            temperature,
            top_k,
            top_p,
            min_p,
            typical_p,
            repetition_penalty,
            repeat_last_n,
            presence_penalty,
            frequency_penalty,
            xtc_probability,
            xtc_threshold,
            mirostat,
            mirostat_tau,
            mirostat_eta
        );
    }
}

// ─── Truncation samplers on the probability vector ──────────────────────────
//
// The engine's dense sampling path keeps one unnormalised probability per
// vocabulary entry, with filtered tokens at exactly zero.  Each helper below
// zeroes more entries; none reorders or rescales the survivors, so the final
// normalisation is the engine's own.  `order` is scratch space.

/// Positive entries of `probs`, most probable first (ties by token id).
fn survivors_descending(probs: &[f32], order: &mut Vec<usize>) {
    order.clear();
    order.extend((0..probs.len()).filter(|&i| probs[i] > 0.0));
    order.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
}

/// Top-n-sigma: drop tokens whose logit is more than `n` standard
/// deviations below the largest one.  Statistics are over the finite logits
/// the grammar `allowed` (llama.cpp skips the `-inf` ones).  Applied to the
/// pre-temperature logits; scaling by a temperature moves the maximum and
/// the deviation together, so the cut is the same.
pub(crate) fn top_n_sigma(
    logits: &[f32],
    allowed: impl Fn(usize) -> bool,
    probs: &mut [f32],
    n: f32,
) {
    if n <= 0.0 || logits.len() <= 1 {
        return;
    }
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for (i, &l) in logits.iter().enumerate() {
        if l != f32::NEG_INFINITY && allowed(i) {
            max = max.max(l);
            sum += l as f64;
            count += 1;
        }
    }
    if count == 0 {
        return;
    }
    let mean = sum / count as f64;
    let var = logits
        .iter()
        .enumerate()
        .filter(|&(i, &l)| l != f32::NEG_INFINITY && allowed(i))
        .map(|(_, &l)| (l as f64 - mean).powi(2))
        .sum::<f64>()
        / count as f64;
    let threshold = max - n * var.sqrt() as f32;
    for (p, &l) in probs.iter_mut().zip(logits) {
        if l < threshold {
            *p = 0.0;
        }
    }
}

/// Locally typical sampling (llama.cpp `llama_sampler_typical_apply`):
/// keep the tokens whose surprise is closest to the distribution's entropy
/// until their mass exceeds `p`.
pub(crate) fn typical(probs: &mut [f32], p: f32, order: &mut Vec<usize>) {
    if p >= 1.0 {
        return;
    }
    let total: f32 = probs.iter().sum();
    if total <= 0.0 {
        return;
    }
    order.clear();
    order.extend((0..probs.len()).filter(|&i| probs[i] > 0.0));
    let entropy: f32 = order
        .iter()
        .map(|&i| {
            let q = probs[i] / total;
            -q * q.ln()
        })
        .sum();
    let score = |i: usize| (-(probs[i] / total).ln() - entropy).abs();
    order.sort_unstable_by(|&a, &b| score(a).total_cmp(&score(b)).then(a.cmp(&b)));
    let mut cum = 0.0f32;
    let mut keep = order.len();
    for (rank, &i) in order.iter().enumerate() {
        cum += probs[i] / total;
        if cum > p {
            keep = rank + 1;
            break;
        }
    }
    for &i in &order[keep..] {
        probs[i] = 0.0;
    }
}

/// XTC, "exclude top choices" (llama.cpp `llama_sample_xtc_apply`): with
/// chance `probability` (decided by `u`, uniform in `[0, 1)`), remove every
/// token at or above `threshold` except the least probable of them.
pub(crate) fn xtc(
    probs: &mut [f32],
    probability: f32,
    threshold: f32,
    u: f64,
    order: &mut Vec<usize>,
) {
    if probability <= 0.0 || threshold > 0.5 || u > probability as f64 {
        return;
    }
    survivors_descending(probs, order);
    if order.len() < 2 {
        return;
    }
    let total: f32 = order.iter().map(|&i| probs[i]).sum();
    let mut last = 0;
    for (rank, &i) in order.iter().enumerate() {
        if probs[i] / total >= threshold {
            last = rank;
        } else {
            break;
        }
    }
    if last > 0 {
        for &i in &order[..last] {
            probs[i] = 0.0;
        }
    }
}

/// Mirostat's running state: the version, its target and learning rate, and
/// the surprise budget `mu` (llama.cpp starts it at `2 * tau`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Mirostat {
    pub version: u8,
    pub tau: f32,
    pub eta: f32,
    pub mu: f32,
}

impl Mirostat {
    pub(crate) fn new(version: u8, tau: f32, eta: f32) -> Self {
        Self {
            version,
            tau,
            eta,
            mu: 2.0 * tau,
        }
    }

    /// Truncate `probs` (temperature applied) the way the active version
    /// does before its draw.
    pub(crate) fn truncate(&self, probs: &mut [f32], order: &mut Vec<usize>) {
        survivors_descending(probs, order);
        if order.is_empty() {
            return;
        }
        let total: f32 = order.iter().map(|&i| probs[i]).sum();
        let keep = if self.version == 1 {
            // Estimate the Zipf exponent from the top m = 100 tokens.
            const M: usize = 100;
            let (mut sum_ti_bi, mut sum_ti_sq) = (0.0f32, 0.0f32);
            for i in 0..(M - 1).min(order.len() - 1) {
                let t_i = ((i + 2) as f32 / (i + 1) as f32).ln();
                let b_i = (probs[order[i]] / probs[order[i + 1]]).ln();
                sum_ti_bi += t_i * b_i;
                sum_ti_sq += t_i * t_i;
            }
            let s_hat = sum_ti_bi / sum_ti_sq;
            let epsilon_hat = s_hat - 1.0;
            let n_vocab = probs.len() as f32;
            let k = ((epsilon_hat * 2f32.powf(self.mu)) / (1.0 - n_vocab.powf(-epsilon_hat)))
                .powf(1.0 / s_hat);
            // A NaN k (one candidate, a flat distribution) keeps one token,
            // as llama.cpp's `std::max(int(k), 1)` does.
            if k.is_finite() && k >= 1.0 {
                (k as usize).min(order.len())
            } else {
                1
            }
        } else {
            order
                .iter()
                .position(|&i| -(probs[i] / total).log2() > self.mu)
                .unwrap_or(order.len())
                .max(1)
        };
        for &i in &order[keep..] {
            probs[i] = 0.0;
        }
    }

    /// Move `mu` towards the target after a token of normalised probability
    /// `p` was drawn.
    pub(crate) fn update(&mut self, p: f32) {
        let observed = -p.log2();
        self.mu -= self.eta * (observed - self.tau);
    }
}

// ─── DRY ────────────────────────────────────────────────────────────────────

/// DRY's limits on sequence breakers (llama.cpp `llama_sampler_init_dry`).
const DRY_MAX_CHAR_LEN: usize = 40;
const DRY_MAX_SEQ_LEN: usize = 20;
/// llama.cpp's `LLAMA_DRY_MAX_SEQ_BREAKERS`.
pub const DRY_MAX_SEQ_BREAKERS: usize = 64;

/// The default DRY sequence breakers (llama.cpp `dry_sequence_breakers`).
pub fn default_dry_sequence_breakers() -> Vec<String> {
    ["\n", ":", "\"", "*"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// The token sequences that spell `breaker` (or a piece of it) — llama.cpp's
/// `get_overlapping_token_sequences`.  A token whose text contains the whole
/// breaker maps to an empty tail; a token ending in a prefix of it maps to
/// the tokenization of the rest.
fn breaker_sequences(
    vocab: &[String],
    breaker: &str,
    tokenize: &dyn Fn(&str) -> Vec<u32>,
    out: &mut HashMap<u32, Vec<Vec<u32>>>,
) {
    let needle = breaker.as_bytes();
    let Some(&first) = needle.first() else {
        return;
    };
    let mut tails: HashMap<usize, Vec<u32>> = HashMap::new();
    for (id, word) in vocab.iter().enumerate() {
        let id = id as u32;
        if word.contains(breaker) {
            let entry = out.entry(id).or_default();
            if !entry.iter().any(Vec::is_empty) {
                entry.push(Vec::new());
            }
            continue;
        }
        let bytes = word.as_bytes();
        for pos in (0..bytes.len()).filter(|&p| bytes[p] == first) {
            let overlap = (bytes.len() - pos).min(needle.len());
            if bytes[pos..pos + overlap] != needle[..overlap] || !breaker.is_char_boundary(overlap)
            {
                continue;
            }
            let tail = tails.entry(overlap).or_insert_with(|| {
                let mut t = tokenize(&breaker[overlap..]);
                t.truncate(DRY_MAX_SEQ_LEN);
                t
            });
            let entry = out.entry(id).or_default();
            if !entry.contains(tail) {
                entry.push(tail.clone());
            }
        }
    }
}

/// DRY ("don't repeat yourself") repetition penalty, ported from llama.cpp
/// (originally Koboldcpp): a token that would extend a sequence already seen
/// in the recent context loses `multiplier * base^(length - allowed_length)`
/// from its logit.  Sequence breakers cap how far back a repeat may reach.
#[derive(Debug, Clone)]
pub(crate) struct Dry {
    multiplier: f32,
    base: f32,
    allowed_length: usize,
    penalty_last_n: usize,
    /// Head token → the tails that complete a breaker after it.
    breakers: HashMap<u32, Vec<Vec<u32>>>,
    /// The last `penalty_last_n` tokens, oldest first.
    last_tokens: VecDeque<u32>,
    repeat_count: Vec<usize>,
    max_token_repeat: HashMap<u32, usize>,
}

/// DRY settings from a request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DryOptions {
    /// Penalty multiplier (0 = off).
    pub multiplier: f32,
    /// Penalty base (values below 1 fall back to 1.75).
    pub base: f32,
    /// Repeats up to this many tokens are free.
    pub allowed_length: u32,
    /// How many recent tokens to scan (0 = off, -1 = the context window).
    pub penalty_last_n: i32,
    /// Strings that end a repeated sequence.
    pub sequence_breakers: Vec<String>,
}

impl Default for DryOptions {
    fn default() -> Self {
        Self {
            multiplier: 0.0,
            base: 1.75,
            allowed_length: 2,
            penalty_last_n: -1,
            sequence_breakers: default_dry_sequence_breakers(),
        }
    }
}

impl DryOptions {
    /// Whether DRY changes anything.
    pub fn enabled(&self) -> bool {
        self.multiplier != 0.0 && self.base >= 1.0 && self.penalty_last_n != 0
    }
}

impl Dry {
    /// Build the sampler.  `vocab` holds each token's decoded text and
    /// `tokenize` encodes a breaker tail without special tokens.
    /// `n_ctx` resolves a `penalty_last_n` of -1.
    pub(crate) fn new(
        options: &DryOptions,
        n_ctx: usize,
        vocab: &[String],
        tokenize: &dyn Fn(&str) -> Vec<u32>,
    ) -> Self {
        let penalty_last_n = if options.penalty_last_n < 0 {
            n_ctx
        } else {
            options.penalty_last_n as usize
        };
        let mut breakers = HashMap::new();
        let mut seen: Vec<String> = Vec::new();
        for breaker in &options.sequence_breakers {
            if breaker.is_empty() {
                continue;
            }
            let mut end = breaker.len().min(DRY_MAX_CHAR_LEN);
            while !breaker.is_char_boundary(end) {
                end -= 1;
            }
            let breaker = &breaker[..end];
            if seen.iter().any(|s| s == breaker) {
                continue;
            }
            if seen.len() >= DRY_MAX_SEQ_BREAKERS {
                break;
            }
            seen.push(breaker.to_string());
            breaker_sequences(vocab, breaker, tokenize, &mut breakers);
        }
        Self {
            multiplier: options.multiplier,
            base: if options.base < 1.0 {
                1.75
            } else {
                options.base
            },
            allowed_length: options.allowed_length as usize,
            penalty_last_n,
            breakers,
            last_tokens: VecDeque::with_capacity(penalty_last_n.min(1 << 16)),
            repeat_count: Vec::new(),
            max_token_repeat: HashMap::new(),
        }
    }

    /// Record a context token (prompt or generated).
    pub(crate) fn accept(&mut self, token: u32) {
        if self.penalty_last_n == 0 {
            return;
        }
        if self.last_tokens.len() == self.penalty_last_n {
            self.last_tokens.pop_front();
        }
        self.last_tokens.push_back(token);
    }

    /// The `i`-th most recent token (0 = the last one).
    fn rat(&self, i: usize) -> u32 {
        self.last_tokens[self.last_tokens.len() - 1 - i]
    }

    /// Penalise `logits` for the next position.
    pub(crate) fn apply(&mut self, logits: &mut [f32]) {
        let n = self.last_tokens.len().min(self.penalty_last_n);
        if n <= self.allowed_length {
            return;
        }
        // Step 1: a breaker in the context limits how far back a repeat may
        // reach.
        let mut rep_limit = n;
        for i in 0..n {
            let Some(tails) = self.breakers.get(&self.rat(i)) else {
                continue;
            };
            let mut longest: Option<usize> = None;
            for tail in tails {
                let len = tail.len();
                if longest.is_some_and(|l| len <= l) || len > i {
                    continue;
                }
                if (0..len).all(|offset| tail[offset] == self.rat(i - offset - 1)) {
                    longest = Some(len);
                }
            }
            if let Some(len) = longest {
                rep_limit = i - len;
                break;
            }
        }
        if rep_limit < self.allowed_length {
            return;
        }

        // Step 2: reverse Z-algorithm — for each position, how long a suffix
        // of the context repeats ending there.
        self.repeat_count.clear();
        self.repeat_count.resize(n, 0);
        let last = n - 1;
        let (mut lt, mut rt) = (0usize, 0usize);
        for k in 1..n {
            if k > rt {
                let mut len = 0;
                while len + k < n && self.rat(len) == self.rat(len + k) {
                    len += 1;
                }
                self.repeat_count[last - k] = len.min(rep_limit);
                if len > 0 {
                    lt = k;
                    rt = k + len - 1;
                }
            } else {
                let p = k - lt;
                let right_part_len = rt - k + 1;
                if self.repeat_count[last - p] < right_part_len {
                    self.repeat_count[last - k] = self.repeat_count[last - p].min(rep_limit);
                } else {
                    let mut i = rt + 1;
                    while i < n && self.rat(i) == self.rat(i - k) {
                        i += 1;
                    }
                    self.repeat_count[last - k] = (i - k).min(rep_limit);
                    lt = k;
                    rt = i - 1;
                }
            }
        }

        // Step 3: the token after each repeat would extend it.
        self.max_token_repeat.clear();
        for i in 0..n - 1 {
            let len = self.repeat_count[i];
            if len >= self.allowed_length {
                let token = self.rat(n - 2 - i);
                let best = self.max_token_repeat.entry(token).or_insert(0);
                *best = (*best).max(len);
            }
        }

        // Step 4: penalise, clamping the exponent so `base^exp` stays finite.
        const FLOAT_MAX_LOG: f32 = 88.722_84;
        let max_exponent = if self.base > 1.000_001 {
            (FLOAT_MAX_LOG / self.base.ln()) as usize
        } else {
            0
        };
        for (&token, &len) in &self.max_token_repeat {
            let Some(logit) = logits.get_mut(token as usize) else {
                continue;
            };
            let single_token_breaker = self
                .breakers
                .get(&token)
                .is_some_and(|tails| tails.iter().any(Vec::is_empty));
            if single_token_breaker {
                continue;
            }
            let mut exponent = len - self.allowed_length;
            if max_exponent > 0 && exponent > max_exponent {
                exponent = max_exponent;
            }
            *logit -= self.multiplier * self.base.powi(exponent as i32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_match_upstream_defaults() {
        // llama.cpp common/common.h `common_params_sampling`.
        let llama = SamplingProfile::LlamaCpp.defaults();
        assert_eq!(
            (llama.temperature, llama.top_k, llama.top_p, llama.min_p),
            (0.8, 40, 0.95, 0.05)
        );
        assert_eq!((llama.repetition_penalty, llama.repeat_last_n), (1.0, 64));
        assert_eq!((llama.xtc_probability, llama.xtc_threshold), (0.0, 0.1));
        assert_eq!(
            (llama.mirostat, llama.mirostat_tau, llama.mirostat_eta),
            (0, 5.0, 0.1)
        );
        assert_eq!(llama.typical_p, 1.0);
        // Ollama api/types.go `DefaultOptions` (min_p is unset, so 0).
        let ollama = SamplingProfile::Ollama.defaults();
        assert_eq!(
            (ollama.temperature, ollama.top_k, ollama.top_p, ollama.min_p),
            (0.8, 40, 0.9, 0.0)
        );
        assert_eq!((ollama.repetition_penalty, ollama.repeat_last_n), (1.0, 64));
        assert_eq!(ollama.typical_p, 1.0);
        // vLLM `_DEFAULT_SAMPLING_PARAMS` / OpenAI.
        let openai = SamplingProfile::OpenAi.defaults();
        assert_eq!(
            (openai.temperature, openai.top_k, openai.top_p, openai.min_p),
            (1.0, 0, 1.0, 0.0)
        );
        assert_eq!(openai.repetition_penalty, 1.0);
        assert_eq!(
            (openai.presence_penalty, openai.frequency_penalty),
            (0.0, 0.0)
        );
    }

    #[test]
    fn gguf_sampling_keys_override_the_profile_and_accept_any_numeric_type() {
        use gguf_file::Value as V;
        let mut md = HashMap::new();
        md.insert(GGUF_SAMPLING_TEMP.to_string(), V::F32(0.6));
        md.insert(GGUF_SAMPLING_TOP_K.to_string(), V::U32(20));
        md.insert(GGUF_SAMPLING_TOP_P.to_string(), V::F64(0.95));
        md.insert(GGUF_SAMPLING_MIN_P.to_string(), V::String("0.01".into()));
        md.insert(GGUF_SAMPLING_PENALTY_LAST_N.to_string(), V::I32(-1));
        md.insert(GGUF_SAMPLING_MIROSTAT.to_string(), V::U8(7)); // out of range: ignored
        md.insert(
            "general.sampling.sequence".to_string(),
            V::String("kt".into()),
        );
        let o = SamplingOverrides::from_gguf(&md);
        assert_eq!(o.temperature, Some(0.6));
        assert_eq!(o.top_k, Some(20));
        assert_eq!(o.top_p, Some(0.95));
        assert_eq!(o.min_p, Some(0.01));
        assert_eq!(o.repeat_last_n, Some(-1));
        assert_eq!(o.mirostat, None);
        let mut d = SamplingProfile::OpenAi.defaults();
        o.apply(&mut d);
        assert_eq!(
            (d.temperature, d.top_k, d.top_p, d.min_p),
            (0.6, 20, 0.95, 0.01)
        );
        // Untouched fields keep the profile's value.
        assert_eq!(d.repetition_penalty, 1.0);
        assert!(SamplingOverrides::from_gguf(&HashMap::new()).is_empty());
    }

    #[test]
    fn typical_keeps_the_tokens_nearest_the_entropy() {
        let mut order = Vec::new();
        let mut probs = vec![0.5, 0.3, 0.15, 0.05];
        typical(&mut probs, 0.5, &mut order);
        // Entropy ≈ 1.14 nats; surprise: 0.69, 1.20, 1.90, 3.00.  The 0.3
        // token is nearest, then 0.5 — together past 0.5.
        assert_eq!(probs, vec![0.5, 0.3, 0.0, 0.0]);
        let mut off = vec![0.5, 0.3, 0.15, 0.05];
        typical(&mut off, 1.0, &mut order);
        assert_eq!(off, vec![0.5, 0.3, 0.15, 0.05]);
    }

    #[test]
    fn xtc_removes_all_but_the_least_likely_top_choice() {
        let mut order = Vec::new();
        let mut probs = vec![0.4, 0.3, 0.2, 0.1];
        xtc(&mut probs, 1.0, 0.15, 0.5, &mut order);
        assert_eq!(probs, vec![0.0, 0.0, 0.2, 0.1]);
        // Not triggered: u above the probability.
        let mut probs = vec![0.4, 0.3, 0.2, 0.1];
        xtc(&mut probs, 0.3, 0.15, 0.5, &mut order);
        assert_eq!(probs, vec![0.4, 0.3, 0.2, 0.1]);
        // Only one token above the threshold: nothing to exclude.
        let mut probs = vec![0.7, 0.1, 0.1, 0.1];
        xtc(&mut probs, 1.0, 0.5, 0.0, &mut order);
        assert_eq!(probs, vec![0.7, 0.1, 0.1, 0.1]);
    }

    #[test]
    fn top_n_sigma_cuts_far_below_the_maximum() {
        let logits = [10.0, 9.5, 0.0, -1.0, 0.5];
        let mut probs = vec![1.0; 5];
        top_n_sigma(&logits, |_| true, &mut probs, 1.0);
        assert_eq!(probs, vec![1.0, 1.0, 0.0, 0.0, 0.0]);
        let mut probs = vec![1.0; 5];
        top_n_sigma(&logits, |_| true, &mut probs, -1.0);
        assert_eq!(probs, vec![1.0; 5]);
    }

    #[test]
    fn mirostat_v2_truncates_by_surprise_and_tracks_tau() {
        let mut order = Vec::new();
        let mut m = Mirostat::new(2, 1.0, 0.5);
        assert_eq!(m.mu, 2.0);
        // Surprise: 1, 2, 3, 3 bits; mu = 2 keeps the first two.
        let mut probs = vec![0.5, 0.25, 0.125, 0.125];
        m.truncate(&mut probs, &mut order);
        assert_eq!(probs, vec![0.5, 0.25, 0.0, 0.0]);
        // Drawing a p = 0.25 token (2 bits) overshoots tau by 1.
        m.update(0.25);
        assert!((m.mu - 1.5).abs() < 1e-6);
        // Mirostat v1 always keeps at least one token.
        let m1 = Mirostat::new(1, 5.0, 0.1);
        let mut one = vec![0.0, 1.0, 0.0];
        m1.truncate(&mut one, &mut order);
        assert_eq!(one, vec![0.0, 1.0, 0.0]);
    }

    fn dry(breakers: &[&str], vocab: &[&str]) -> Dry {
        let vocab: Vec<String> = vocab.iter().map(|s| s.to_string()).collect();
        let lookup = vocab.clone();
        let tokenize = move |s: &str| -> Vec<u32> {
            lookup
                .iter()
                .position(|w| w == s)
                .map(|i| vec![i as u32])
                .unwrap_or_default()
        };
        Dry::new(
            &DryOptions {
                multiplier: 1.0,
                base: 2.0,
                allowed_length: 2,
                penalty_last_n: -1,
                sequence_breakers: breakers.iter().map(|s| s.to_string()).collect(),
            },
            64,
            &vocab,
            &tokenize,
        )
    }

    #[test]
    fn dry_penalises_the_token_that_extends_a_repeat() {
        // Context a b c x a b c: emitting `x` again would extend a repeat of
        // length 3 (`a b c`), one past the allowed 2.
        let vocab = ["a", "b", "c", "x", "\n"];
        let mut d = dry(&["\n"], &vocab);
        for t in [0, 1, 2, 3, 0, 1, 2] {
            d.accept(t);
        }
        let mut logits = vec![0.0f32; 5];
        d.apply(&mut logits);
        assert_eq!(logits, vec![0.0, 0.0, 0.0, -2.0, 0.0]);

        // A breaker before the second copy leaves it whole.
        let mut d = dry(&["\n"], &vocab);
        for t in [0, 1, 2, 3, 4, 0, 1, 2] {
            d.accept(t);
        }
        let mut logits = vec![0.0f32; 5];
        d.apply(&mut logits);
        assert_eq!(logits, vec![0.0, 0.0, 0.0, -2.0, 0.0]);
        // A breaker inside it caps the repeat at `b c`: the allowed length,
        // so the penalty is the bare multiplier.
        let mut d = dry(&["\n"], &vocab);
        for t in [0, 1, 2, 3, 0, 4, 1, 2] {
            d.accept(t);
        }
        let mut logits = vec![0.0f32; 5];
        d.apply(&mut logits);
        assert_eq!(logits, vec![0.0, 0.0, 0.0, -1.0, 0.0]);
        // Nothing repeats: no penalty.
        let mut d = dry(&["\n"], &vocab);
        for t in [0, 1, 2, 3] {
            d.accept(t);
        }
        let mut logits = vec![0.0f32; 5];
        d.apply(&mut logits);
        assert_eq!(logits, vec![0.0; 5]);
    }

    #[test]
    fn dry_breakers_cover_tokens_containing_or_starting_them() {
        let vocab: Vec<String> = ["a", "x:y", "q:", ":", "b"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut out = HashMap::new();
        breaker_sequences(&vocab, ":", &|_| Vec::new(), &mut out);
        let mut heads: Vec<u32> = out.keys().copied().collect();
        heads.sort_unstable();
        assert_eq!(heads, vec![1, 2, 3]);
        assert!(out[&3].iter().all(Vec::is_empty));
    }
}
