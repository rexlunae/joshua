//! Core LLM inference engine for Joshua.
//!
//! The engine loads a GGUF model file and tokenises input using a
//! `tokenizer.json` file placed alongside the model.  Both the GGUF weights
//! and the tokenizer are loaded entirely in pure Rust — no C or C++ runtime
//! is required.
//!
//! A fresh [`ModelWeights`](candle_transformers::models::quantized_llama::ModelWeights)
//! is created for every inference call so that each request has an isolated
//! KV cache.  After the initial cold start the operating system's page cache
//! keeps the GGUF file hot in RAM, making subsequent loads fast.
//!
//! # Model directory layout
//!
//! ```text
//! my-model/
//! ├── model.gguf          ← quantised weights (any GGUF-compatible architecture)
//! └── tokenizer.json      ← HuggingFace tokenizer (download from the model card)
//! ```
//!
//! You can also point directly at a `.gguf` file; `tokenizer.json` is then
//! looked up in the same directory.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;
use tokenizers::Tokenizer;

use crate::error::{JoshuaError, Result};
use crate::types::{ChatMessage, GenerationOptions, UsageInfo};

// ─── Engine ───────────────────────────────────────────────────────────────────

/// The Joshua inference engine.
///
/// Instances are cheaply clonable (the tokenizer is `Arc`-wrapped) and are
/// `Send + Sync`, so a single `Arc<Engine>` can be shared across threads.
pub struct Engine {
    /// Path to the `.gguf` file.
    model_path: PathBuf,
    /// Stateless tokenizer, shared across all inference calls.
    tokenizer: Arc<Tokenizer>,
    /// EOS token IDs derived from the GGUF metadata and common special tokens.
    eos_token_ids: Vec<u32>,
    /// Stem of the model file (used as the model identifier in API responses).
    model_name: String,
    /// Context-window size in tokens.
    n_ctx: u32,
}

// `PathBuf`, `Arc<Tokenizer>`, `Vec<u32>`, `String`, and `u32` are all
// `Send + Sync`, so Engine is automatically `Send + Sync`.

impl Engine {
    /// Load a GGUF model using a 4 096-token context window.
    ///
    /// `model_path` can be either the path to a `.gguf` file or a directory
    /// that contains one.  A `tokenizer.json` must exist in the same directory
    /// as the `.gguf` file.
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        Self::with_n_ctx(model_path, 4096)
    }

    /// Load a GGUF model with a custom context-window size.
    pub fn with_n_ctx(model_path: impl AsRef<Path>, n_ctx: u32) -> Result<Self> {
        let raw_path = model_path.as_ref().to_path_buf();

        // Resolve the actual .gguf file path.
        let gguf_path = if raw_path.is_dir() {
            find_gguf_in_dir(&raw_path)?
        } else {
            raw_path
        };

        let model_name = gguf_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        tracing::info!("Loading model from {:?}", gguf_path);

        // Locate tokenizer.json in the same directory as the GGUF file.
        let model_dir = gguf_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(JoshuaError::ModelLoad(format!(
                "tokenizer.json not found at {:?}.\n\
                 Place it alongside the .gguf file \
                 (download from the model's HuggingFace repository).",
                tokenizer_path
            )));
        }

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| JoshuaError::ModelLoad(format!("tokenizer load failed: {e}")))?;

        // Read GGUF metadata once to extract EOS token IDs.
        let mut file = File::open(&gguf_path)?;
        let gguf = gguf_file::Content::read(&mut file)
            .map_err(|e| JoshuaError::ModelLoad(format!("GGUF read failed: {e}")))?;

        let eos_token_ids = extract_eos_ids(&gguf, &tokenizer);

        tracing::info!(
            "Model '{}' ready (ctx={}, eos_ids={:?})",
            model_name,
            n_ctx,
            eos_token_ids
        );

        Ok(Self {
            model_path: gguf_path,
            tokenizer: Arc::new(tokenizer),
            eos_token_ids,
            model_name,
            n_ctx,
        })
    }

    /// The stem of the loaded model file name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Absolute path of the loaded `.gguf` file.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Context-window size in tokens.
    pub fn n_ctx(&self) -> u32 {
        self.n_ctx
    }

    // ─── Prompt formatting ────────────────────────────────────────────────────

    /// Format messages as a ChatML prompt and append the assistant turn header.
    fn format_chatml_prompt(messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();
        for msg in messages {
            prompt.push_str("<|im_start|>");
            prompt.push_str(&msg.role);
            prompt.push('\n');
            prompt.push_str(&msg.content);
            prompt.push_str("<|im_end|>\n");
        }
        prompt.push_str("<|im_start|>assistant\n");
        prompt
    }

    // ─── Completion ───────────────────────────────────────────────────────────

    /// Run a chat completion.
    ///
    /// Returns `(generated_text, usage, prefill_tps, decode_tps)`.
    pub fn complete(
        &self,
        messages: &[ChatMessage],
        options: &GenerationOptions,
    ) -> Result<(String, UsageInfo, f64, f64)> {
        let prompt = Self::format_chatml_prompt(messages);
        self.complete_raw(&prompt, options)
    }

    /// Run completion from an arbitrary raw prompt string.
    pub fn complete_raw(
        &self,
        prompt: &str,
        options: &GenerationOptions,
    ) -> Result<(String, UsageInfo, f64, f64)> {
        // Load model weights from the GGUF file.  After the first call the OS
        // page cache keeps the file data hot, so this is I/O-free in practice.
        let mut model = self.load_model()?;

        // ── Tokenise ─────────────────────────────────────────────────────────
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| JoshuaError::Tokenization(e.to_string()))?;
        let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
        let n_prompt = prompt_tokens.len();

        if n_prompt >= self.n_ctx as usize {
            return Err(JoshuaError::PromptTooLong(n_prompt, self.n_ctx as usize));
        }

        // ── Prefill ───────────────────────────────────────────────────────────
        // Process all prompt tokens in a single forward pass.
        // Input shape: [1, n_prompt].
        let input = Tensor::new(prompt_tokens.as_slice(), &Device::Cpu)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| JoshuaError::Inference(e.to_string()))?;

        let prefill_start = Instant::now();
        // index_pos = 0: the KV cache is empty; start building from position 0.
        // forward() internally selects the last-token logits; output: [1, vocab_size].
        let logits = model
            .forward(&input, 0)
            .map_err(|e| JoshuaError::Inference(e.to_string()))?;
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        let mut logits_vec = squeeze_batch_logits(&logits)?;

        // ── Repetition-penalty history ────────────────────────────────────────
        // Seed the recent-token window with the tail of the prompt (up to 64 tokens).
        const REP_WINDOW: usize = 64;
        let mut recent_tokens: Vec<u32> = prompt_tokens
            .iter()
            .rev()
            .take(REP_WINDOW)
            .copied()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        // ── Decode loop ───────────────────────────────────────────────────────
        let mut rng = thread_rng();
        let mut response = String::new();
        let mut n_decoded: u32 = 0;
        let mut n_cur = n_prompt;
        let decode_start = Instant::now();

        loop {
            if n_decoded >= options.max_tokens {
                break;
            }

            let next_token = sample_token(&logits_vec, options, &mut rng, &recent_tokens)?;

            if self.eos_token_ids.contains(&next_token) {
                break;
            }

            // Decode the new token to text.
            let piece = self
                .tokenizer
                .decode(&[next_token], false)
                .map_err(|e| JoshuaError::Inference(e.to_string()))?;
            response.push_str(&piece);
            n_decoded += 1;

            // Maintain sliding-window token history for repetition penalty.
            if recent_tokens.len() >= REP_WINDOW {
                recent_tokens.remove(0);
            }
            recent_tokens.push(next_token);

            if Self::check_stop_sequences(&mut response, &options.stop_sequences) {
                break;
            }

            // Single-token step: input [1, 1], output [1, vocab_size].
            let step_input = Tensor::new(&[next_token], &Device::Cpu)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| JoshuaError::Inference(e.to_string()))?;

            let step_logits = model
                .forward(&step_input, n_cur)
                .map_err(|e| JoshuaError::Inference(e.to_string()))?;

            logits_vec = squeeze_batch_logits(&step_logits)?;
            n_cur += 1;
        }

        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

        let prefill_tps = if prefill_ms > 0.0 {
            n_prompt as f64 / (prefill_ms / 1000.0)
        } else {
            0.0
        };
        let decode_tps = if decode_ms > 0.0 && n_decoded > 0 {
            n_decoded as f64 / (decode_ms / 1000.0)
        } else {
            0.0
        };

        tracing::debug!(
            prefill_tokens = n_prompt,
            prefill_tps,
            decode_tokens = n_decoded,
            decode_tps,
            "Completion finished"
        );

        let usage = UsageInfo {
            prompt_tokens: n_prompt as u32,
            completion_tokens: n_decoded,
            total_tokens: n_prompt as u32 + n_decoded,
        };

        Ok((response, usage, prefill_tps, decode_tps))
    }

    // ─── Embeddings ───────────────────────────────────────────────────────────

    /// Compute dense embeddings for one or more texts.
    ///
    /// Standard causal language models do not expose sentence-level embeddings
    /// without a pooling head.  Use a dedicated embedding model
    /// (e.g. nomic-embed-text, BGE, E5) whose GGUF variant is built with
    /// embedding support.
    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Err(JoshuaError::InvalidRequest(format!(
            "Dense embedding extraction requires a model built with pooling support \
             (e.g. nomic-embed-text-v2, BGE-M3). \
             Standard language models cannot produce sentence embeddings via this API. \
             ({} input text(s) provided)",
            texts.len()
        )))
    }

    // ─── Private helpers ─────────────────────────────────────────────────────

    /// Load `ModelWeights` from the GGUF file.
    ///
    /// Each call re-opens the file so the KV cache starts empty — the OS page
    /// cache ensures subsequent opens are fast.
    fn load_model(&self) -> Result<ModelWeights> {
        let mut file = File::open(&self.model_path)?;
        let gguf = gguf_file::Content::read(&mut file)
            .map_err(|e| JoshuaError::ModelLoad(format!("GGUF read failed: {e}")))?;
        ModelWeights::from_gguf(gguf, &mut file, &Device::Cpu)
            .map_err(|e| JoshuaError::ModelLoad(format!("model init failed: {e}")))
    }

    /// Scan `response` for any configured stop sequence and truncate it.
    fn check_stop_sequences(response: &mut String, stops: &[String]) -> bool {
        for stop in stops {
            if stop.is_empty() {
                continue;
            }
            if response.ends_with(stop.as_str()) {
                response.truncate(response.len() - stop.len());
                return true;
            }
        }
        false
    }
}

// ─── GGUF / tokenizer helpers ─────────────────────────────────────────────────

/// Walk `dir` and return the first `.gguf` file found.
fn find_gguf_in_dir(dir: &Path) -> Result<PathBuf> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
            return Ok(path);
        }
    }
    Err(JoshuaError::ModelLoad(format!(
        "No .gguf file found in {:?}",
        dir
    )))
}

/// Derive EOS token IDs from GGUF metadata and well-known special token strings.
fn extract_eos_ids(gguf: &gguf_file::Content, tokenizer: &Tokenizer) -> Vec<u32> {
    let mut ids: Vec<u32> = Vec::new();

    // Primary: explicit EOS from GGUF metadata.
    let eos_key = "tokenizer.ggml.eos_token_id";
    match gguf.metadata.get(eos_key) {
        Some(gguf_file::Value::U32(id)) => ids.push(*id),
        Some(gguf_file::Value::I32(id)) => ids.push(*id as u32),
        Some(gguf_file::Value::U64(id)) => ids.push(*id as u32),
        _ => {}
    }

    // Fallback: common EOS token strings for popular model families.
    for token_str in &[
        "</s>",
        "<|endoftext|>",
        "<|im_end|>",
        "<end_of_turn>",
        "<eos>",
        "<|eot_id|>",
        "<|end|>",
    ] {
        if let Some(id) = tokenizer.token_to_id(token_str) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }

    ids
}

// ─── Tensor helpers ───────────────────────────────────────────────────────────

/// Convert a `[1, vocab_size]` logits tensor to a flat `Vec<f32>`.
///
/// `ModelWeights::forward()` always returns shape `[batch, vocab_size]`
/// because it selects the last sequence position internally.
fn squeeze_batch_logits(logits: &Tensor) -> Result<Vec<f32>> {
    // Remove the batch dimension (index 0) to get [vocab_size].
    logits
        .squeeze(0)
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| JoshuaError::Inference(e.to_string()))
}

// ─── Sampling ────────────────────────────────────────────────────────────────

/// Sample the next token from a raw logit vector.
///
/// Implements repetition penalty, temperature scaling, top-k filtering,
/// min-p filtering, top-p (nucleus) filtering, and weighted random sampling,
/// all in pure Rust.
fn sample_token(
    logits: &[f32],
    opts: &GenerationOptions,
    rng: &mut impl rand::Rng,
    recent_tokens: &[u32],
) -> Result<u32> {
    if logits.is_empty() {
        return Ok(0);
    }

    // ── Repetition penalty ────────────────────────────────────────────────────
    // For tokens present in the recent window, divide positive logits and
    // multiply negative logits by `repetition_penalty` (> 1.0 discourages
    // repetition; 1.0 is a no-op).  Applied before temperature so the penalty
    // is independent of the temperature scale.
    let logits: Vec<f32> = if opts.repetition_penalty != 1.0 {
        let mut v = logits.to_vec();
        for &token in recent_tokens {
            if let Some(l) = v.get_mut(token as usize) {
                if *l > 0.0 {
                    *l /= opts.repetition_penalty;
                } else {
                    *l *= opts.repetition_penalty;
                }
            }
        }
        v
    } else {
        logits.to_vec()
    };

    // ── Greedy ────────────────────────────────────────────────────────────────
    if opts.temperature <= 0.0 {
        return Ok(logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0));
    }

    // ── Temperature scaling ───────────────────────────────────────────────────
    let inv_temp = 1.0_f32 / opts.temperature;
    // Subtract max for numerical stability before exp.
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits
        .iter()
        .map(|&l| ((l - max_logit) * inv_temp).exp())
        .collect();

    // ── Top-k ─────────────────────────────────────────────────────────────────
    let k = opts.top_k as usize;
    if k > 0 && k < probs.len() {
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|(_, a), (_, b)| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });
        for &(idx, _) in indexed.iter().skip(k) {
            probs[idx] = 0.0;
        }
    }

    // ── Min-p ─────────────────────────────────────────────────────────────────
    if opts.min_p > 0.0 {
        let max_p = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let threshold = max_p * opts.min_p;
        for p in &mut probs {
            if *p < threshold {
                *p = 0.0;
            }
        }
    }

    // ── Top-p (nucleus) ───────────────────────────────────────────────────────
    if opts.top_p < 1.0 && opts.top_p > 0.0 {
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            let mut sorted_idx: Vec<usize> = (0..probs.len()).collect();
            sorted_idx.sort_unstable_by(|&a, &b| {
                probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut cumsum = 0.0_f32;
            let mut cut_from = probs.len();
            for (rank, &idx) in sorted_idx.iter().enumerate() {
                cumsum += probs[idx] / sum;
                if cumsum > opts.top_p {
                    cut_from = rank + 1;
                    break;
                }
            }
            for &idx in sorted_idx.iter().skip(cut_from) {
                probs[idx] = 0.0;
            }
        }
    }

    // ── Normalise & sample ────────────────────────────────────────────────────
    let total: f32 = probs.iter().sum();
    if total <= 0.0 {
        // Fallback: greedy from original (penalty-adjusted) logits.
        return Ok(logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0));
    }

    for p in &mut probs {
        *p /= total;
    }

    let dist =
        WeightedIndex::new(&probs).map_err(|e| JoshuaError::Inference(e.to_string()))?;
    Ok(dist.sample(rng) as u32)
}

