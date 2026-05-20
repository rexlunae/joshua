//! Core LLM inference engine for Joshua.
//!
//! The engine loads a GGUF model file via memory-mapped I/O (delegated to
//! llama.cpp), keeping resident RAM usage low even for large models.  A fresh
//! [`llama_cpp_2::context::LlamaContext`] is created for every call so that
//! multiple requests can run concurrently without sharing mutable state.

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::time::Instant;

use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
    sampling::LlamaSampler,
    token::LlamaToken,
};

use crate::error::{JoshuaError, Result};
use crate::types::{ChatMessage, GenerationOptions, UsageInfo};

// ─── Engine ───────────────────────────────────────────────────────────────────

/// The Joshua inference engine.
///
/// Thread-safe: multiple threads may call [`Engine::complete`] and
/// [`Engine::embed`] concurrently because each call allocates its own
/// llama.cpp context.
pub struct Engine {
    // `model` must be listed before `backend` so it is dropped first.
    model: LlamaModel,
    _backend: LlamaBackend,
    model_path: PathBuf,
    model_name: String,
    /// Context-window size passed to every newly created [`LlamaContext`].
    n_ctx: u32,
}

// Safety: LlamaModel explicitly implements Send + Sync upstream.
// LlamaBackend is a zero-sized sentinel struct that is also Send + Sync.
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

impl Engine {
    /// Load a GGUF model from `model_path` using a 4 096-token context window.
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        Self::with_n_ctx(model_path, 4096)
    }

    /// Load a GGUF model with a custom context-window size.
    pub fn with_n_ctx(model_path: impl AsRef<Path>, n_ctx: u32) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();

        tracing::info!("Initialising llama.cpp backend");
        let backend = LlamaBackend::init()
            .map_err(|e| JoshuaError::ModelLoad(e.to_string()))?;

        tracing::info!("Loading model from {:?}", model_path);
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
            .map_err(|e| JoshuaError::ModelLoad(format!("{}: {e}", model_path.display())))?;

        let model_name = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        tracing::info!("Model '{}' loaded (ctx={})", model_name, n_ctx);

        Ok(Self {
            model,
            _backend: backend,
            model_path,
            model_name,
            n_ctx,
        })
    }

    /// The stem of the loaded model's file name (e.g. `"gemma-3-270m-it-q4_k_m"`).
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Absolute path of the loaded model file.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Context-window size in tokens.
    pub fn n_ctx(&self) -> u32 {
        self.n_ctx
    }

    // ─── Prompt formatting ───────────────────────────────────────────────────

    /// Format messages as a ChatML prompt and append the assistant turn opener.
    ///
    /// Example output:
    /// ```text
    /// <|im_start|>system
    /// You are a helpful assistant.<|im_end|>
    /// <|im_start|>user
    /// Hello!<|im_end|>
    /// <|im_start|>assistant
    /// ```
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

    /// Run completion starting from an arbitrary raw prompt string.
    pub fn complete_raw(
        &self,
        prompt: &str,
        options: &GenerationOptions,
    ) -> Result<(String, UsageInfo, f64, f64)> {
        // ── Context ───────────────────────────────────────────────────────────
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.n_ctx));

        let mut ctx = self
            .model
            .new_context(&self._backend, ctx_params)
            .map_err(|e| JoshuaError::ContextCreation(e.to_string()))?;

        // ── Tokenise ─────────────────────────────────────────────────────────
        #[allow(deprecated)]
        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| JoshuaError::Tokenization(e.to_string()))?;

        let max_ctx = self.n_ctx as usize;
        if tokens.len() >= max_ctx {
            return Err(JoshuaError::PromptTooLong(tokens.len(), max_ctx));
        }

        let n_prompt_tokens = tokens.len() as u32;

        // ── Prefill ───────────────────────────────────────────────────────────
        let mut batch = LlamaBatch::new(max_ctx, 1);
        batch
            .add_sequence(&tokens, 0, false)
            .map_err(|e| JoshuaError::Inference(e.to_string()))?;

        let prefill_start = Instant::now();
        ctx.decode(&mut batch)
            .map_err(|e| JoshuaError::Inference(e.to_string()))?;
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        // ── Sampler ───────────────────────────────────────────────────────────
        let mut sampler = Self::build_sampler(options);

        // Accept all prompt tokens so the sampler's repetition history is warm.
        sampler.accept_many(tokens.iter().copied());

        // ── Decode loop ───────────────────────────────────────────────────────
        let eos = self.model.token_eos();
        let mut response = String::new();
        let mut n_cur = n_prompt_tokens as usize;
        let mut n_decoded: u32 = 0;
        let decode_start = Instant::now();

        loop {
            if n_decoded >= options.max_tokens {
                break;
            }

            // Sample the next token from the last logits row.
            let next_token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(next_token);

            // End-of-generation: EOS or any other end-of-generation token.
            if next_token == eos || self.model.is_eog_token(next_token) {
                break;
            }

            // Decode token → text.
            let piece = token_to_str(&self.model, next_token)?;
            response.push_str(&piece);
            n_decoded += 1;

            // Check stop sequences.
            if Self::check_stop_sequences(&mut response, &options.stop_sequences) {
                break;
            }

            // Prepare single-token batch for the next step.
            batch.clear();
            batch
                .add(next_token, n_cur as i32, &[0], true)
                .map_err(|e| JoshuaError::Inference(e.to_string()))?;
            n_cur += 1;

            ctx.decode(&mut batch)
                .map_err(|e| JoshuaError::Inference(e.to_string()))?;
        }

        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

        let prefill_tps = if prefill_ms > 0.0 {
            n_prompt_tokens as f64 / (prefill_ms / 1000.0)
        } else {
            0.0
        };
        let decode_tps = if decode_ms > 0.0 && n_decoded > 0 {
            n_decoded as f64 / (decode_ms / 1000.0)
        } else {
            0.0
        };

        tracing::debug!(
            prefill_tokens = n_prompt_tokens,
            prefill_tps,
            decode_tokens = n_decoded,
            decode_tps,
            "Completion finished"
        );

        let usage = UsageInfo {
            prompt_tokens: n_prompt_tokens,
            completion_tokens: n_decoded,
            total_tokens: n_prompt_tokens + n_decoded,
        };

        Ok((response, usage, prefill_tps, decode_tps))
    }

    // ─── Embeddings ───────────────────────────────────────────────────────────

    /// Compute dense embeddings for one or more texts.
    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.n_ctx))
            .with_embeddings(true);

        let mut ctx = self
            .model
            .new_context(&self._backend, ctx_params)
            .map_err(|e| JoshuaError::ContextCreation(e.to_string()))?;

        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            #[allow(deprecated)]
            let tokens = self
                .model
                .str_to_token(text, AddBos::Always)
                .map_err(|e| JoshuaError::Tokenization(e.to_string()))?;

            let n = tokens.len();
            if n == 0 {
                results.push(vec![]);
                continue;
            }

            let mut batch = LlamaBatch::new(n, 1);
            batch
                .add_sequence(&tokens, 0, false)
                .map_err(|e| JoshuaError::Inference(e.to_string()))?;

            ctx.decode(&mut batch)
                .map_err(|e| JoshuaError::Inference(e.to_string()))?;

            let embedding = ctx
                .embeddings_seq_ith(0)
                .map_err(|e| JoshuaError::Inference(e.to_string()))?
                .to_vec();

            results.push(embedding);
        }

        Ok(results)
    }

    // ─── Helpers ─────────────────────────────────────────────────────────────

    /// Build a [`LlamaSampler`] chain from generation options.
    fn build_sampler(options: &GenerationOptions) -> LlamaSampler {
        // Repetition penalty (last 64 tokens, no frequency/presence penalty)
        let penalty = LlamaSampler::penalties(64, options.repetition_penalty, 0.0, 0.0);

        if options.temperature <= 0.0 {
            // Greedy decoding.
            return LlamaSampler::chain_simple([penalty, LlamaSampler::greedy()]);
        }

        // Temperature → top-k → min-p → top-p → distribution sampler.
        let samplers: Vec<LlamaSampler> = vec![
            penalty,
            LlamaSampler::temp(options.temperature),
            LlamaSampler::top_k(options.top_k),
            LlamaSampler::min_p(options.min_p, 1),
            LlamaSampler::top_p(options.top_p, 1),
            LlamaSampler::dist(0xDEAD_BEEF),
        ];
        LlamaSampler::chain_simple(samplers)
    }

    /// Scan the generated text for any stop sequence.  If one is found, trim it
    /// from the end of `response` and return `true`.
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

// ─── Token decoding helper ────────────────────────────────────────────────────

/// Convert a single token to its string representation.
///
/// Uses `token_to_str` (which is `#[deprecated]` upstream but still the
/// simplest way to get a UTF-8 string for a token).  The deprecation warning
/// is suppressed here so callers remain clean.
fn token_to_str(model: &LlamaModel, token: LlamaToken) -> Result<String> {
    #[allow(deprecated)]
    use llama_cpp_2::model::Special;
    #[allow(deprecated)]
    model
        .token_to_str(token, Special::Tokenize)
        .map_err(|e| JoshuaError::Inference(e.to_string()))
}
