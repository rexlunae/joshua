//! OpenAI-compatible request and response types used by Joshua.

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// The current Unix time in seconds, as the `created` fields carry it.
pub fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// `time` as an RFC 3339 UTC timestamp with nanoseconds
/// (`2026-11-06T07:48:02.123456789Z`), the form Ollama's Go clients parse.
pub fn rfc3339(time: SystemTime) -> String {
    let since = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = since.as_secs();
    let (days, rem) = (secs / 86_400, secs % 86_400);
    // Howard Hinnant's civil_from_days, for days since 1970-01-01.
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:09}Z",
        rem / 3_600,
        rem % 3_600 / 60,
        rem % 60,
        since.subsec_nanos()
    )
}

// ─── Chat messages ───────────────────────────────────────────────────────────

/// A single message in a chat conversation.
///
/// Deserialisation accepts both plain-string `content` and OpenAI content
/// parts (`[{"type":"text",…},{"type":"image_url",…}]`): text parts are
/// joined and image URLs land in `images`, so multimodal clients work
/// unchanged.  `content: null` (assistant tool-call turns) becomes `""`.
#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    /// The role of the author (`"system"`, `"user"`, `"assistant"`, `"tool"`).
    pub role: String,
    /// The text content of the message.
    pub content: String,
    /// Attached images: `data:` URLs or local file paths.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    /// Optional author name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Tool calls previously emitted by the assistant, passed back verbatim
    /// by clients during multi-turn tool use so chat templates can render
    /// the earlier assistant turn.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
    /// For `role: "tool"` messages: the ID of the call being answered.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// An earlier assistant turn's reasoning, passed back by clients so
    /// templates that render it (Qwen3, GLM-4.5+, Kimi-K2-Thinking,
    /// DeepSeek-V3.1+) see it as `message.reasoning_content`.  Accepted on
    /// input as `reasoning_content` (llama.cpp, DeepSeek), `reasoning`
    /// (vLLM) or `thinking` (Ollama, gpt-oss templates).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

impl<'de> Deserialize<'de> for ChatMessage {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            role: String,
            #[serde(default)]
            content: Option<serde_json::Value>,
            #[serde(default)]
            images: Option<Vec<String>>,
            #[serde(default)]
            name: Option<String>,
            #[serde(default)]
            tool_calls: Option<serde_json::Value>,
            #[serde(default)]
            tool_call_id: Option<String>,
            #[serde(default)]
            reasoning_content: Option<String>,
            #[serde(default)]
            reasoning: Option<String>,
            #[serde(default)]
            thinking: Option<String>,
        }

        let raw = Raw::deserialize(deserializer)?;
        let mut images = raw.images.unwrap_or_default();
        let content = match raw.content {
            None | Some(serde_json::Value::Null) => String::new(),
            Some(serde_json::Value::String(s)) => s,
            Some(serde_json::Value::Array(parts)) => {
                let mut text = String::new();
                for part in parts {
                    match part.get("type").and_then(|t| t.as_str()) {
                        Some("text") => {
                            if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                                if !text.is_empty() {
                                    text.push('\n');
                                }
                                text.push_str(t);
                            }
                        }
                        Some("image_url") => {
                            if let Some(url) = part
                                .get("image_url")
                                .and_then(|i| i.get("url"))
                                .and_then(|u| u.as_str())
                            {
                                images.push(url.to_string());
                            }
                        }
                        // Unknown part types are ignored rather than rejected.
                        _ => {}
                    }
                }
                text
            }
            Some(other) => {
                return Err(serde::de::Error::custom(format!(
                    "message content must be a string or an array of parts, got: {other}"
                )))
            }
        };

        Ok(ChatMessage {
            role: raw.role,
            content,
            images: if images.is_empty() { None } else { Some(images) },
            name: raw.name,
            tool_calls: raw.tool_calls,
            tool_call_id: raw.tool_call_id,
            reasoning_content: raw
                .reasoning_content
                .or(raw.reasoning)
                .or(raw.thinking)
                .filter(|r| !r.is_empty()),
        })
    }
}

impl ChatMessage {
    /// Construct a plain text message.
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            images: None,
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }
}


// ─── Generation options ───────────────────────────────────────────────────────

/// Most alternatives a request may ask to see per token (OpenAI's and
/// vLLM's limit).
pub const MAX_TOP_LOGPROBS: u32 = 20;

/// One generated token's log-probability, with the likeliest alternatives.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TokenLogprob {
    /// The token generated.
    pub token: u32,
    /// Its natural-log probability under the model.
    pub logprob: f32,
    /// The likeliest tokens at that step, most likely first, as many as
    /// [`GenerationOptions::logprobs`] asked for.
    pub top: Vec<(u32, f32)>,
    /// The token's rank among all of them, the likeliest being 1 (vLLM's
    /// `rank`).
    pub rank: u32,
    /// Where the token's text starts in the generated text, in bytes.  A
    /// token that only completes a character another token began (a byte
    /// fragment) starts where that character does.
    pub start: usize,
    /// The same place in characters (the legacy completions API's
    /// `text_offset`).
    pub offset: usize,
    /// Whether the token's text reaches a chat response's answer once the
    /// reasoning (and its markup) is split off: false for reasoning, its
    /// tags, and the whitespace or headers dropped around them.
    pub answer: bool,
}

/// A shared buffer the decode loop pushes [`TokenLogprob`]s into, for a
/// streaming caller to take with the text they belong to.  The decode loop
/// releases entries as it releases their text, so an entry whose text is
/// still held back (it may begin a stop sequence) is not taken early, and
/// one whose text a stop sequence removes is never taken.
#[derive(Clone, Default)]
pub struct LogprobSink(std::sync::Arc<std::sync::Mutex<LogprobQueue>>);

#[derive(Default)]
struct LogprobQueue {
    /// The prompt's entries, once scored, until taken.
    prompt: Vec<TokenLogprob>,
    entries: std::collections::VecDeque<TokenLogprob>,
    /// Entries starting before this byte of the text may be taken.
    released: usize,
}

impl LogprobSink {
    fn queue(&self) -> std::sync::MutexGuard<'_, LogprobQueue> {
        self.0.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Add one token's figures.
    pub fn push(&self, entry: TokenLogprob) {
        self.queue().entries.push_back(entry);
    }

    /// Hand over the prompt's entries, scored before decoding starts.
    pub fn set_prompt(&self, entries: Vec<TokenLogprob>) {
        self.queue().prompt = entries;
    }

    /// The prompt's entries, once: empty until scored, and after.
    pub fn take_prompt(&self) -> Vec<TokenLogprob> {
        std::mem::take(&mut self.queue().prompt)
    }

    /// Let entries whose text starts before byte `end` be taken (`usize::MAX`
    /// releases all of them).
    pub fn release(&self, end: usize) {
        let mut queue = self.queue();
        queue.released = queue.released.max(end);
    }

    /// The released entries not taken yet, in order.
    pub fn take(&self) -> Vec<TokenLogprob> {
        let mut queue = self.queue();
        let released = queue.released;
        let n = queue
            .entries
            .iter()
            .take_while(|e| e.start < released)
            .count();
        queue.entries.drain(..n).collect()
    }
}

impl std::fmt::Debug for LogprobSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LogprobSink")
    }
}

/// Parameters that control the token-generation process.
///
/// Every field added after the first release defaults to "off", so a caller
/// that does not set it samples exactly as before.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationOptions {
    /// Trusted caller's cache namespace; never accepted from request JSON.
    #[serde(skip)]
    pub cache_scope: Option<[u8; 32]>,
    /// Maximum number of tokens to generate.  [`MAX_TOKENS_UNLIMITED`]
    /// generates until end of sequence or the end of the context window;
    /// the server's `--max-output-tokens` caps every request either way.
    #[serde(default = "GenerationOptions::default_max_tokens")]
    pub max_tokens: u32,
    /// Sampling temperature (0 = greedy).
    #[serde(default = "GenerationOptions::default_temperature")]
    pub temperature: f32,
    /// Nucleus (top-p) sampling threshold.
    #[serde(default = "GenerationOptions::default_top_p")]
    pub top_p: f32,
    /// Top-k sampling limit (0 = disabled).
    #[serde(default = "GenerationOptions::default_top_k")]
    pub top_k: i32,
    /// Min-p threshold relative to the highest-probability token.
    #[serde(default = "GenerationOptions::default_min_p")]
    pub min_p: f32,
    /// Locally typical sampling threshold (1.0 = disabled).
    #[serde(default = "GenerationOptions::default_typical_p")]
    pub typical_p: f32,
    /// Repetition penalty (1.0 = disabled).
    #[serde(default = "GenerationOptions::default_repetition_penalty")]
    pub repetition_penalty: f32,
    /// How many recent prompt/generated tokens the repetition, presence and
    /// frequency penalties look at (llama.cpp's `repeat_last_n`): 64 by
    /// default, 0 turns the penalties off, -1 covers the whole context.
    #[serde(default = "GenerationOptions::default_repeat_last_n")]
    pub repeat_last_n: i32,
    /// OpenAI presence penalty (0 = disabled, range `-2.0..=2.0`).
    ///
    /// Subtracted once from the logit of every token that appears in the
    /// same recent-token window the repetition penalty uses (the last
    /// [`GenerationOptions::repeat_last_n`] prompt/generated tokens).
    /// Unlike the repetition penalty it is also applied at temperature 0,
    /// since it is never on by default.
    #[serde(default)]
    pub presence_penalty: f32,
    /// OpenAI frequency penalty (0 = disabled, range `-2.0..=2.0`).
    ///
    /// Subtracted from a token's logit once per occurrence in the recent
    /// window (see [`GenerationOptions::presence_penalty`]).
    #[serde(default)]
    pub frequency_penalty: f32,
    /// Seed for the sampler's random draws.  `None` seeds from OS entropy.
    /// A fixed seed reproduces the same sample stream for the same prompt,
    /// model, backend and options.  Each draw hashes `(seed, absolute token
    /// position, stream)`, so enabling speculative decoding, changing the
    /// draft length, or batching requests does not change the token at a
    /// given position; greedy decoding never consumes randomness.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Per-request context window in tokens: prompt plus generated tokens
    /// may not exceed it.  `None` uses the engine's whole window
    /// ([`crate::Engine::n_ctx`], the server's `--n-ctx`); a value above
    /// that is rejected with [`crate::JoshuaError::InvalidRequest`], since
    /// the KV cache and RoPE tables are sized at load.
    #[serde(default)]
    pub context_window: Option<u32>,
    /// Extra variables passed to the model's GGUF chat template — the
    /// `chat_template_kwargs` convention of vLLM and llama.cpp.
    ///
    /// This is how reasoning settings reach the model: Qwen3 / GLM-4.5 /
    /// SmolLM3 / Hunyuan templates read `enable_thinking`, DeepSeek-V3.1 /
    /// Granite read `thinking`, gpt-oss reads `reasoning_effort` and
    /// Seed-OSS reads `thinking_budget` (see [`reasoning_template_kwargs`]).
    /// Keys a template does not reference are ignored by it.  The reserved
    /// names `messages`, `tools`, `add_generation_prompt`, `bos_token` and
    /// `eos_token` cannot be overridden.  Ignored by the ChatML fallback used
    /// when the GGUF has no chat template, and by raw-prompt completion.
    #[serde(default)]
    pub chat_template_kwargs: serde_json::Map<String, serde_json::Value>,
    /// A chat template (Jinja source) to render this request with instead
    /// of the model's own (vLLM's per-request `chat_template`).
    #[serde(default)]
    pub chat_template: Option<String>,
    /// Append the template's assistant header after the messages (the
    /// default).  `false` renders the conversation as it is.
    #[serde(default = "GenerationOptions::default_add_generation_prompt")]
    pub add_generation_prompt: bool,
    /// Continue the final assistant message instead of opening a new turn
    /// (assistant prefill).  `None` continues it whenever the conversation
    /// ends with an assistant message without tool calls, as llama-server
    /// and Anthropic's API do.
    #[serde(default)]
    pub continue_final_message: Option<bool>,
    /// Strings that will terminate generation when encountered.
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    /// Keep the matched stop string at the end of the output (vLLM's
    /// `include_stop_str_in_output`).
    #[serde(default)]
    pub include_stop_str_in_output: bool,
    /// Token ids that end generation like EOS (vLLM's `stop_token_ids`).
    /// The stop token is not part of the output.
    #[serde(default)]
    pub stop_token_ids: Vec<u32>,
    /// Do not stop at the model's end-of-sequence tokens: they are masked
    /// out, so generation runs to `max_tokens` or the end of the context.
    #[serde(default)]
    pub ignore_eos: bool,
    /// Mask end-of-sequence tokens, stop tokens and stop strings until this
    /// many tokens were generated.
    #[serde(default)]
    pub min_tokens: u32,
    /// Additive logit biases, applied before every other sampler.  A bias
    /// of `-inf` bans the token.
    #[serde(default)]
    pub logit_bias: Vec<LogitBias>,
    /// Only these token ids may be sampled (vLLM's `allowed_token_ids`).
    #[serde(default)]
    pub allowed_token_ids: Option<Vec<u32>>,
    /// Words that may not be generated (vLLM's `bad_words`): the last token
    /// of each word's tokenization, with and without a leading space, is
    /// masked whenever the tokens before it were just generated.
    #[serde(default)]
    pub bad_words: Vec<String>,
    /// Keep only the last this-many prompt tokens (vLLM's
    /// `truncate_prompt_tokens`): at least 1 and at most the context
    /// window, or [`TRUNCATE_TO_CONTEXT`].
    #[serde(default)]
    pub truncate_prompt_tokens: Option<u32>,
    /// Top-n-sigma: drop tokens more than this many standard deviations
    /// below the best logit (≤ 0 = disabled).
    #[serde(default = "GenerationOptions::default_top_n_sigma")]
    pub top_n_sigma: f32,
    /// XTC trigger probability (0 = disabled).
    #[serde(default)]
    pub xtc_probability: f32,
    /// XTC probability threshold (> 0.5 disables XTC).
    #[serde(default = "GenerationOptions::default_xtc_threshold")]
    pub xtc_threshold: f32,
    /// Mirostat version: 0 = off, 1 = Mirostat, 2 = Mirostat 2.0.  It
    /// replaces top-k, typical-p, top-p, min-p, top-n-sigma and XTC.
    #[serde(default)]
    pub mirostat: u8,
    /// Mirostat target entropy (tau).
    #[serde(default = "GenerationOptions::default_mirostat_tau")]
    pub mirostat_tau: f32,
    /// Mirostat learning rate (eta).
    #[serde(default = "GenerationOptions::default_mirostat_eta")]
    pub mirostat_eta: f32,
    /// DRY repetition penalty (off unless `multiplier` is non-zero).
    #[serde(default)]
    pub dry: crate::sampling::DryOptions,
    /// End the model's reasoning after this many tokens by forcing its
    /// end-of-reasoning tag (vLLM's `thinking_token_budget`, llama.cpp's
    /// `reasoning_budget_tokens`).  Only models whose reasoning has
    /// start/end tags (`<think>…</think>`) can be cut short.
    #[serde(default)]
    pub reasoning_budget: Option<u32>,
    /// Text forced before the end-of-reasoning tag when the budget runs out
    /// (llama.cpp's `reasoning_budget_message`).
    #[serde(default)]
    pub reasoning_budget_message: Option<String>,
    /// A structured-output constraint applied while sampling.
    ///
    /// Parsed from the request's `response_format`; `None` leaves decoding
    /// unconstrained and byte-identical to earlier releases.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constraint: Option<ConstraintSpec>,
    /// Report each generated token's log-probability with this many of the
    /// most likely alternatives (at most [`MAX_TOP_LOGPROBS`]): OpenAI's
    /// `top_logprobs`, vLLM's `logprobs` and llama-server's `n_probs`.
    /// `None` reports none.  Probabilities are the model's own (a softmax of
    /// its logits, before temperature and the other samplers), as vLLM
    /// reports them by default.
    #[serde(default)]
    pub logprobs: Option<u32>,
    /// Score the prompt too, reporting each prompt token's
    /// [`TokenLogprob`] with this many alternatives (vLLM's
    /// `prompt_logprobs`, and `echo` with `logprobs`): see
    /// [`crate::engine::Completion::prompt_logprobs`].
    #[serde(default)]
    pub prompt_logprobs: Option<u32>,
    /// Where a streaming caller collects each token's [`TokenLogprob`] as
    /// it is decoded, to send with the text that follows; never from
    /// request JSON.
    #[serde(skip)]
    pub logprob_sink: Option<LogprobSink>,
}

/// `max_tokens` value that means "until end of sequence or the end of the
/// context window" (still capped by the server's `--max-output-tokens`).
pub const MAX_TOKENS_UNLIMITED: u32 = u32::MAX;

/// `truncate_prompt_tokens` value that keeps as much of the prompt as the
/// context window holds with room for one generated token (vLLM's `-1`).
pub const TRUNCATE_TO_CONTEXT: u32 = u32::MAX;

/// One logit bias: a token id, or text whose every token gets the bias
/// (llama.cpp accepts both).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogitBias {
    /// The token(s) to bias.
    pub token: TokenRef,
    /// Added to the logit; `-inf` bans the token.
    pub bias: f32,
}

/// A token named by id or by text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TokenRef {
    /// A token id.
    Id(u32),
    /// Text, tokenized without special tokens.
    Text(String),
}

/// A syntax constraint for generated tokens.
///
/// Only JSON is supported: `json_object` accepts any object, and
/// `json_schema` accepts documents valid under the given schema.  The
/// constraint is compiled once per schema and cached by the engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConstraintSpec {
    /// Any JSON object (`response_format: {"type": "json_object"}`).
    JsonObject,
    /// A JSON document matching this schema.
    JsonSchema {
        /// The JSON Schema document.
        schema: serde_json::Value,
    },
}

impl GenerationOptions {
    pub(crate) fn cache_namespace(&self) -> Option<[u8; 32]> {
        crate::affinity::namespace(self.cache_scope, self.context_window)
    }

    /// Options whose sampler settings come from `defaults` (see
    /// [`crate::Engine::sampling_defaults`]), with no output cap and every
    /// other setting off.
    pub fn with_defaults(defaults: &crate::sampling::SamplingDefaults) -> Self {
        Self {
            max_tokens: MAX_TOKENS_UNLIMITED,
            temperature: defaults.temperature,
            top_k: defaults.top_k,
            top_p: defaults.top_p,
            min_p: defaults.min_p,
            typical_p: defaults.typical_p,
            repetition_penalty: defaults.repetition_penalty,
            repeat_last_n: defaults.repeat_last_n,
            presence_penalty: defaults.presence_penalty,
            frequency_penalty: defaults.frequency_penalty,
            xtc_probability: defaults.xtc_probability,
            xtc_threshold: defaults.xtc_threshold,
            mirostat: defaults.mirostat,
            mirostat_tau: defaults.mirostat_tau,
            mirostat_eta: defaults.mirostat_eta,
            ..Self::default()
        }
    }

    /// Whether a setting depends on the tokens generated so far in a way
    /// speculative verification does not reproduce, so decoding goes one
    /// token at a time: a minimum length, bad words, DRY or Mirostat's
    /// running state.  A reasoning budget does too, on a model with
    /// reasoning tags (the engine checks that).
    pub(crate) fn needs_plain_decode(&self) -> bool {
        // Log-probabilities are read from the logits each token was sampled
        // from, which a verified draft never has on its own.
        self.min_tokens > 0
            || self.logprobs.is_some()
            || !self.bad_words.is_empty()
            || self.dry.enabled()
            || self.mirostat != 0
    }

    fn default_max_tokens() -> u32 {
        256
    }
    fn default_temperature() -> f32 {
        0.7
    }
    fn default_top_p() -> f32 {
        0.9
    }
    fn default_top_k() -> i32 {
        40
    }
    fn default_min_p() -> f32 {
        0.05
    }
    fn default_typical_p() -> f32 {
        1.0
    }
    fn default_repetition_penalty() -> f32 {
        1.1
    }
    fn default_repeat_last_n() -> i32 {
        64
    }
    fn default_add_generation_prompt() -> bool {
        true
    }
    fn default_top_n_sigma() -> f32 {
        -1.0
    }
    fn default_xtc_threshold() -> f32 {
        0.1
    }
    fn default_mirostat_tau() -> f32 {
        5.0
    }
    fn default_mirostat_eta() -> f32 {
        0.1
    }
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            cache_scope: None,
            max_tokens: Self::default_max_tokens(),
            temperature: Self::default_temperature(),
            top_p: Self::default_top_p(),
            top_k: Self::default_top_k(),
            min_p: Self::default_min_p(),
            typical_p: Self::default_typical_p(),
            repetition_penalty: Self::default_repetition_penalty(),
            repeat_last_n: Self::default_repeat_last_n(),
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: None,
            context_window: None,
            chat_template_kwargs: serde_json::Map::new(),
            chat_template: None,
            add_generation_prompt: Self::default_add_generation_prompt(),
            continue_final_message: None,
            stop_sequences: vec![],
            include_stop_str_in_output: false,
            stop_token_ids: vec![],
            ignore_eos: false,
            min_tokens: 0,
            logit_bias: vec![],
            allowed_token_ids: None,
            bad_words: vec![],
            truncate_prompt_tokens: None,
            top_n_sigma: Self::default_top_n_sigma(),
            xtc_probability: 0.0,
            xtc_threshold: Self::default_xtc_threshold(),
            mirostat: 0,
            mirostat_tau: Self::default_mirostat_tau(),
            mirostat_eta: Self::default_mirostat_eta(),
            dry: crate::sampling::DryOptions::default(),
            reasoning_budget: None,
            reasoning_budget_message: None,
            constraint: None,
            logprobs: None,
            prompt_logprobs: None,
            logprob_sink: None,
        }
    }
}

// ─── Reasoning settings ───────────────────────────────────────────────────────

/// Template variables the engine itself supplies; `chat_template_kwargs`
/// may not override them.
pub const RESERVED_TEMPLATE_VARS: [&str; 5] = [
    "messages",
    "tools",
    "add_generation_prompt",
    "bos_token",
    "eos_token",
];

/// How hard a reasoning model should think — the OpenAI `reasoning_effort`
/// values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// No reasoning: turns thinking off where the model can.
    None,
    /// The least reasoning short of none.
    Minimal,
    /// Low effort.
    Low,
    /// Medium effort.
    Medium,
    /// High effort.
    High,
    /// Extra-high effort.
    Xhigh,
    /// The most a model offers (vLLM's `max`, for DeepSeek-V4).
    Max,
}

impl ReasoningEffort {
    /// Whether this effort level asks for thinking at all.
    pub fn enables_thinking(self) -> bool {
        self != Self::None
    }

    /// The `reasoning_effort` template value: the three levels effort-aware
    /// templates (gpt-oss) understand.  `none` and `minimal` map to `low`
    /// (gpt-oss cannot switch reasoning off), `xhigh` and `max` to `high`.
    /// A template with a `max` level gets it through `chat_template_kwargs`.
    pub fn template_level(self) -> &'static str {
        match self {
            Self::None | Self::Minimal | Self::Low => "low",
            Self::Medium => "medium",
            Self::High | Self::Xhigh | Self::Max => "high",
        }
    }
}

/// The `reasoning` request object (OpenAI Responses / OpenRouter style).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReasoningConfig {
    /// Effort level; same meaning as top-level `reasoning_effort`.
    #[serde(default)]
    pub effort: Option<ReasoningEffort>,
    /// Turn thinking on or off.
    #[serde(default)]
    pub enabled: Option<bool>,
    /// Thinking-token budget, passed to templates as `thinking_budget`
    /// (Seed-OSS).  Advisory: the model self-limits; it is not enforced.
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

/// Map reasoning settings to the chat-template variables model families
/// read.
///
/// * `enabled` sets both `enable_thinking` (Qwen3, GLM-4.5/4.6, SmolLM3,
///   Hunyuan-A13B) and `thinking` (DeepSeek-V3.1/V3.2, Granite 3.x);
/// * `effort` sets `reasoning_effort` (gpt-oss; see
///   [`ReasoningEffort::template_level`]) and implies `enabled` (`none` →
///   off, anything else → on);
/// * thinking turned off without an effort also sets `reasoning_effort` to
///   `low`, the nearest a gpt-oss model gets to no reasoning;
/// * `budget` sets `thinking_budget` (Seed-OSS).
///
/// Templates ignore variables they never reference, so a model without a
/// given control (Llama, Gemma, Mistral, the always-thinking DeepSeek-R1 /
/// Kimi-K2-Thinking, the Qwen3-2507 split models) renders unchanged.
pub fn reasoning_template_kwargs(
    effort: Option<ReasoningEffort>,
    enabled: Option<bool>,
    budget: Option<u32>,
) -> serde_json::Map<String, serde_json::Value> {
    use serde_json::Value;
    let mut kwargs = serde_json::Map::new();
    let enabled = enabled.or(effort.map(ReasoningEffort::enables_thinking));
    if let Some(on) = enabled {
        kwargs.insert("enable_thinking".into(), Value::Bool(on));
        kwargs.insert("thinking".into(), Value::Bool(on));
    }
    let level = match (effort, enabled) {
        (Some(e), _) => Some(e.template_level()),
        (None, Some(false)) => Some(ReasoningEffort::None.template_level()),
        _ => None,
    };
    if let Some(level) = level {
        kwargs.insert("reasoning_effort".into(), Value::from(level));
    }
    if let Some(budget) = budget {
        kwargs.insert("thinking_budget".into(), Value::from(budget));
    }
    kwargs
}

/// [`reasoning_template_kwargs`] for callers outside the HTTP request path
/// (the CLI): rejects an `effort` and `enabled` that disagree about whether
/// reasoning is on, as [`ChatCompletionRequest`]'s validation does, instead of
/// rendering both and letting each template pick a different one.
pub fn checked_reasoning_template_kwargs(
    effort: Option<ReasoningEffort>,
    enabled: Option<bool>,
    budget: Option<u32>,
) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    if let (Some(e), Some(on)) = (effort, enabled) {
        if e.enables_thinking() != on {
            return Err(
                "reasoning_effort and enable_thinking disagree about whether reasoning is on"
                    .to_string(),
            );
        }
    }
    Ok(reasoning_template_kwargs(effort, enabled, budget))
}

// ─── Usage statistics ─────────────────────────────────────────────────────────

/// Token-usage statistics returned with every response.
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct UsageInfo {
    /// Number of tokens in the prompt.
    pub prompt_tokens: u32,
    /// Number of tokens generated.
    pub completion_tokens: u32,
    /// Total tokens processed.
    pub total_tokens: u32,
    /// Prompt tokens served from cached state (OpenAI / vLLM / llama.cpp
    /// `prompt_tokens_details.cached_tokens`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    /// Generated tokens spent reasoning
    /// (`completion_tokens_details.reasoning_tokens`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

/// `usage.prompt_tokens_details`.
#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq, Eq)]
pub struct PromptTokensDetails {
    /// Prompt tokens whose state was reused instead of recomputed.
    pub cached_tokens: u32,
}

/// `usage.completion_tokens_details`.
#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq, Eq)]
pub struct CompletionTokensDetails {
    /// Generated tokens inside the model's reasoning.
    pub reasoning_tokens: u32,
}

// ─── OpenAI chat completions ──────────────────────────────────────────────────

/// Sampling and length settings shared by the generation request bodies
/// (chat completions, text completions), in OpenAI's, vLLM's and
/// llama.cpp's spellings.  Flattened into each request type.
///
/// Every field is optional; [`SamplingParams::apply`] layers the ones a
/// request sets over options built from the API's
/// [`crate::sampling::SamplingDefaults`].
#[derive(Debug, Default, Clone, Deserialize)]
pub struct SamplingParams {
    /// Upper bound on generated tokens.  Absent or `-1` generates until end
    /// of sequence or the end of the context window, still capped by the
    /// server's `--max-output-tokens`.
    #[serde(default)]
    pub max_tokens: Option<i64>,
    /// OpenAI's newer name for `max_tokens`; wins over it.
    #[serde(default)]
    pub max_completion_tokens: Option<i64>,
    /// llama.cpp's name for the cap; wins over both OpenAI spellings, as in
    /// llama-server.
    #[serde(default)]
    pub n_predict: Option<i64>,
    /// Sampling temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Top-p sampling.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Top-k sampling (0 or -1 = off).
    #[serde(default)]
    pub top_k: Option<i32>,
    /// Min-p sampling.
    #[serde(default)]
    pub min_p: Option<f32>,
    /// Locally typical sampling (1.0 = off).
    #[serde(default)]
    pub typical_p: Option<f32>,
    /// Repetition penalty (also llama.cpp's `repeat_penalty`).
    #[serde(default, alias = "repeat_penalty")]
    pub repetition_penalty: Option<f32>,
    /// Penalty window (0 = off, -1 = the context window).
    #[serde(default)]
    pub repeat_last_n: Option<i32>,
    /// OpenAI presence penalty, `-2.0..=2.0`.
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// OpenAI frequency penalty, `-2.0..=2.0`.
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// Sampler seed; a negative value (llama.cpp's `-1`) means random.
    #[serde(default)]
    pub seed: Option<i64>,
    /// Stop sequences — either a single string or an array.
    #[serde(default)]
    pub stop: Option<serde_json::Value>,
    /// Keep the matched stop string in the output (vLLM).
    #[serde(default)]
    pub include_stop_str_in_output: Option<bool>,
    /// Token ids that end generation (vLLM).
    #[serde(default)]
    pub stop_token_ids: Option<Vec<u32>>,
    /// Keep generating past end-of-sequence tokens.
    #[serde(default)]
    pub ignore_eos: Option<bool>,
    /// Generate at least this many tokens before any stop applies (vLLM).
    #[serde(default)]
    pub min_tokens: Option<u32>,
    /// Logit biases: OpenAI's `{"<id>": bias}` object, or llama.cpp's
    /// `[[id or text, bias], …]` array; a key may also be text, and a bias
    /// of `false` bans the token.
    #[serde(default)]
    pub logit_bias: Option<serde_json::Value>,
    /// Only these token ids may be sampled (vLLM).
    #[serde(default)]
    pub allowed_token_ids: Option<Vec<u32>>,
    /// Words that may not be generated (vLLM).
    #[serde(default)]
    pub bad_words: Option<Vec<String>>,
    /// Keep only the last N prompt tokens (vLLM; `-1` keeps as many as the
    /// context window holds).
    #[serde(default)]
    pub truncate_prompt_tokens: Option<i64>,
    /// Top-n-sigma (≤ 0 = off).
    #[serde(default)]
    pub top_n_sigma: Option<f32>,
    /// XTC trigger probability.
    #[serde(default)]
    pub xtc_probability: Option<f32>,
    /// XTC threshold.
    #[serde(default)]
    pub xtc_threshold: Option<f32>,
    /// Mirostat version (0, 1, 2).
    #[serde(default)]
    pub mirostat: Option<i32>,
    /// Mirostat target entropy.
    #[serde(default)]
    pub mirostat_tau: Option<f32>,
    /// Mirostat learning rate.
    #[serde(default)]
    pub mirostat_eta: Option<f32>,
    /// DRY multiplier (0 = off).
    #[serde(default)]
    pub dry_multiplier: Option<f32>,
    /// DRY base.
    #[serde(default)]
    pub dry_base: Option<f32>,
    /// DRY allowed repeat length.
    #[serde(default)]
    pub dry_allowed_length: Option<u32>,
    /// DRY scan window (0 = off, -1 = the context window).
    #[serde(default)]
    pub dry_penalty_last_n: Option<i32>,
    /// DRY sequence breakers.
    #[serde(default)]
    pub dry_sequence_breakers: Option<Vec<String>>,
    /// llama.cpp dynamic temperature; not implemented, so a non-zero range
    /// is rejected rather than ignored.
    #[serde(default)]
    pub dynatemp_range: Option<f32>,
}

impl SamplingParams {
    /// The output cap: [`MAX_TOKENS_UNLIMITED`] when the request leaves it
    /// unset or sends `-1`.
    pub fn max_tokens(&self) -> Result<u32, String> {
        match self
            .n_predict
            .or(self.max_completion_tokens)
            .or(self.max_tokens)
        {
            None | Some(-1) => Ok(MAX_TOKENS_UNLIMITED),
            Some(n) if n < 0 => {
                Err("max_tokens must be -1 (no limit) or a non-negative count".to_string())
            }
            Some(n) => Ok(u32::try_from(n).unwrap_or(MAX_TOKENS_UNLIMITED)),
        }
    }

    /// Stop sequences from `stop`: a string, an array of strings, or null.
    pub fn stop_sequences(&self) -> Vec<String> {
        match &self.stop {
            Some(serde_json::Value::String(s)) => vec![s.clone()],
            Some(serde_json::Value::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .filter(|s| !s.is_empty())
                .collect(),
            _ => vec![],
        }
    }

    /// Write the settings this request sets into `options`, leaving every
    /// other field as it is.  Returns a client-facing message for values
    /// out of range.
    pub fn apply(&self, options: &mut GenerationOptions) -> Result<(), String> {
        for (name, value) in [
            ("presence_penalty", self.presence_penalty),
            ("frequency_penalty", self.frequency_penalty),
        ] {
            if value.is_some_and(|v| !(-2.0..=2.0).contains(&v)) {
                return Err(format!("{name} must be between -2.0 and 2.0"));
            }
        }
        if self.mirostat.is_some_and(|m| !(0..=2).contains(&m)) {
            return Err("mirostat must be 0, 1 or 2".to_string());
        }
        if self.repeat_last_n.is_some_and(|n| n < -1)
            || self.dry_penalty_last_n.is_some_and(|n| n < -1)
        {
            return Err("repeat_last_n and dry_penalty_last_n must be -1 or more".to_string());
        }
        if self.truncate_prompt_tokens.is_some_and(|n| n < -1) {
            return Err("truncate_prompt_tokens must be -1 or more".to_string());
        }
        if self.dynatemp_range.is_some_and(|r| r != 0.0) {
            return Err("dynatemp_range is not supported".to_string());
        }
        if self
            .dry_sequence_breakers
            .as_ref()
            .is_some_and(|b| b.is_empty())
        {
            return Err("dry_sequence_breakers must be a non-empty array of strings".to_string());
        }
        if self
            .dry_sequence_breakers
            .as_ref()
            .is_some_and(|b| b.len() > crate::sampling::DRY_MAX_SEQ_BREAKERS)
        {
            return Err(format!(
                "dry_sequence_breakers must contain at most {} entries",
                crate::sampling::DRY_MAX_SEQ_BREAKERS
            ));
        }

        options.max_tokens = self.max_tokens()?;
        macro_rules! set {
            ($($field:ident),*) => {
                $(if let Some(v) = self.$field {
                    options.$field = v;
                })*
            };
        }
        set!(
            temperature,
            top_p,
            top_k,
            min_p,
            typical_p,
            repetition_penalty,
            repeat_last_n,
            presence_penalty,
            frequency_penalty,
            min_tokens,
            ignore_eos,
            include_stop_str_in_output,
            top_n_sigma,
            xtc_probability,
            xtc_threshold,
            mirostat_tau,
            mirostat_eta
        );
        if let Some(seed) = self.seed {
            options.seed = u64::try_from(seed).ok();
        }
        if let Some(m) = self.mirostat {
            options.mirostat = m as u8;
        }
        if self.stop.is_some() {
            options.stop_sequences = self.stop_sequences();
        }
        if let Some(ids) = &self.stop_token_ids {
            options.stop_token_ids = ids.clone();
        }
        if let Some(ids) = &self.allowed_token_ids {
            options.allowed_token_ids = Some(ids.clone());
        }
        if let Some(words) = &self.bad_words {
            options.bad_words = words.clone();
        }
        if let Some(n) = self.truncate_prompt_tokens {
            // The engine refuses 0 and counts past the context window.
            let most = TRUNCATE_TO_CONTEXT - 1;
            options.truncate_prompt_tokens = Some(match n {
                -1 => TRUNCATE_TO_CONTEXT,
                n => u32::try_from(n).map_or(most, |n| n.min(most)),
            });
        }
        if let Some(bias) = &self.logit_bias {
            options.logit_bias = parse_logit_bias(bias)?;
        }
        if let Some(v) = self.dry_multiplier {
            options.dry.multiplier = v;
        }
        if let Some(v) = self.dry_base {
            // llama.cpp: a base below 1 falls back to the default.
            if v >= 1.0 {
                options.dry.base = v;
            }
        }
        if let Some(v) = self.dry_allowed_length {
            options.dry.allowed_length = v;
        }
        if let Some(v) = self.dry_penalty_last_n {
            options.dry.penalty_last_n = v;
        }
        if let Some(v) = &self.dry_sequence_breakers {
            options.dry.sequence_breakers = v.clone();
        }
        if options.min_tokens > 0 && options.min_tokens > options.max_tokens {
            return Err(format!(
                "min_tokens ({}) must not exceed max_tokens ({})",
                options.min_tokens, options.max_tokens
            ));
        }
        Ok(())
    }
}

/// The largest logit bias a request can set: OpenAI's documented range,
/// to which vLLM clamps every bias.
pub const MAX_LOGIT_BIAS: f64 = 100.0;

/// Parse `logit_bias`: OpenAI's object of token id → bias, or llama.cpp's
/// array of `[token, bias]` pairs, where a token may be text and a bias of
/// `false` bans it.  Entries llama.cpp would skip (a non-numeric bias, a
/// malformed pair) are skipped here too.  Numeric biases are clamped to
/// ±[`MAX_LOGIT_BIAS`] as vLLM does: a larger one decides nothing more,
/// and one past `f32`'s range would turn the sampled distribution into
/// NaN.
pub fn parse_logit_bias(value: &serde_json::Value) -> Result<Vec<LogitBias>, String> {
    use serde_json::Value;
    let bias = |v: &Value| match v {
        Value::Number(n) => n
            .as_f64()
            .map(|b| b.clamp(-MAX_LOGIT_BIAS, MAX_LOGIT_BIAS) as f32),
        Value::Bool(false) => Some(f32::NEG_INFINITY),
        _ => None,
    };
    let token = |key: &str| match key.parse::<u32>() {
        Ok(id) => TokenRef::Id(id),
        Err(_) => TokenRef::Text(key.to_string()),
    };
    let mut out = Vec::new();
    match value {
        Value::Null => {}
        Value::Object(map) => {
            for (key, v) in map {
                if let Some(bias) = bias(v) {
                    out.push(LogitBias {
                        token: token(key),
                        bias,
                    });
                }
            }
        }
        Value::Array(pairs) => {
            for pair in pairs {
                let Some([t, v]) = pair.as_array().map(Vec::as_slice) else {
                    continue;
                };
                let Some(bias) = bias(v) else {
                    continue;
                };
                let token = match t {
                    Value::Number(n) => match n.as_u64().and_then(|id| u32::try_from(id).ok()) {
                        Some(id) => TokenRef::Id(id),
                        None => continue,
                    },
                    Value::String(s) => TokenRef::Text(s.clone()),
                    _ => continue,
                };
                out.push(LogitBias { token, bias });
            }
        }
        _ => return Err("logit_bias must be an object or an array of pairs".to_string()),
    }
    Ok(out)
}

/// OpenAI-compatible `POST /v1/chat/completions` request body.
///
/// Unknown fields are ignored.  Fields that are accepted but cannot be
/// honoured are rejected by [`ChatCompletionRequest::to_generation_options`]
/// rather than silently dropped: `n > 1`, `top_logprobs` above 20 or
/// without `logprobs: true`, and `response_format` types other than `text`,
/// `json_object`, or `json_schema`.
#[derive(Debug, Default, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model identifier (e.g. `"joshua"` or path-derived name).
    pub model: String,
    /// Conversation history including the new user turn.
    pub messages: Vec<ChatMessage>,
    /// Sampling, length and stop settings.
    #[serde(flatten)]
    pub sampling: SamplingParams,
    /// Whether to stream token-by-token via SSE.
    #[serde(default)]
    pub stream: Option<bool>,
    /// OpenAI's cache-routing key: requests sharing it share a cache
    /// namespace and, behind `joshua route`, a worker.
    #[serde(default)]
    pub prompt_cache_key: Option<String>,
    /// Accepted for compatibility; the final streamed chunk always carries
    /// usage, so `include_usage` is effectively always on.
    #[serde(default)]
    pub stream_options: Option<serde_json::Value>,
    /// Tool/function definitions for tool-calling.
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    /// `"none"` hides `tools` from the model; `"auto"` (the default) offers
    /// them; a named function (`{"type": "function", "function": {"name"}}`)
    /// offers that tool alone.  `"required"` offers them all: whether the
    /// model calls one is not enforced.
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// Number of choices; only `1` is supported.
    #[serde(default)]
    pub n: Option<u32>,
    /// Report each generated token's log-probability.
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// How many of the likeliest alternatives to report with each token
    /// (0 to 20); needs `logprobs: true`.
    #[serde(default)]
    pub top_logprobs: Option<u32>,
    /// vLLM's: the prompt's log-probabilities, with this many alternatives
    /// (0 to 20); not when streaming, as vLLM.
    #[serde(default)]
    pub prompt_logprobs: Option<i64>,
    /// Output format; only `{"type": "text"}` is supported.
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    /// Reasoning effort (`none`, `minimal`, `low`, `medium`, `high`,
    /// `xhigh`, `max`) — see [`reasoning_template_kwargs`] for the
    /// per-model mapping.
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Reasoning settings object: `{"effort", "enabled", "max_tokens"}`.
    #[serde(default)]
    pub reasoning: Option<ReasoningConfig>,
    /// Thinking on/off switch (Qwen / DashScope top-level spelling).
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    /// Cut the reasoning short after this many tokens (vLLM's
    /// `thinking_token_budget`, llama.cpp's `reasoning_budget_tokens` /
    /// `thinking_budget_tokens`; `-1` = no budget).
    #[serde(
        default,
        alias = "reasoning_budget_tokens",
        alias = "thinking_budget_tokens"
    )]
    pub thinking_token_budget: Option<i64>,
    /// Text forced before the end-of-reasoning tag when the budget runs out
    /// (llama.cpp).
    #[serde(default)]
    pub reasoning_budget_message: Option<String>,
    /// How to report reasoning: llama.cpp's `none` (inline in `content`),
    /// `auto` / `deepseek` (separate, the default) or `deepseek-legacy`
    /// (separate unless streaming).
    #[serde(default)]
    pub reasoning_format: Option<String>,
    /// vLLM's switch: `false` drops the reasoning from the response.
    #[serde(default)]
    pub include_reasoning: Option<bool>,
    /// Extra chat-template variables (vLLM / llama.cpp); these win over
    /// the values derived from the reasoning settings above.
    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
    /// A chat template to render this request with (vLLM).
    #[serde(default)]
    pub chat_template: Option<String>,
    /// Append the assistant header after the messages (default `true`).
    #[serde(default)]
    pub add_generation_prompt: Option<bool>,
    /// Continue the final assistant message (vLLM, llama.cpp): a boolean,
    /// or llama.cpp's `"content"` / `"reasoning_content"`.  Unset, a final
    /// assistant message is continued as llama-server does.
    #[serde(default)]
    pub continue_final_message: Option<serde_json::Value>,
    /// Documents for retrieval-aware templates, passed as `documents`.
    #[serde(default)]
    pub documents: Option<Vec<serde_json::Value>>,
    /// vLLM: prefix the response with the continued final message.
    #[serde(default)]
    pub echo: Option<bool>,
    /// Per-request context window in tokens (also accepted as Ollama's
    /// `num_ctx` or `n_ctx`); must not exceed the server's `--n-ctx`.
    #[serde(default, alias = "num_ctx", alias = "n_ctx")]
    pub context_window: Option<u32>,
}

/// Parse the OpenAI `response_format` value into a decoding constraint.
pub(crate) fn parse_response_format(format: &serde_json::Value) -> Result<Option<ConstraintSpec>, String> {
    let kind = format
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "response_format requires a string `type`".to_string())?;
    match kind {
        "text" => Ok(None),
        "json_object" => Ok(Some(ConstraintSpec::JsonObject)),
        "json_schema" => {
            let wrapper = format.get("json_schema").ok_or_else(|| {
                "response_format type `json_schema` requires `json_schema`".to_string()
            })?;
            // OpenAI wraps the schema as `{name, strict, schema}`; also
            // accept a bare schema document for convenience.  JSON Schema
            // does not require a top-level `type` (`properties`, `allOf`,
            // `$ref`, `enum`, … are valid on their own), so the wrapper's
            // `schema` key — not the presence of `type` — discriminates the
            // two shapes.
            let schema = match wrapper.get("schema") {
                Some(schema) => schema,
                // Wrapper-shaped but with no schema body: report the missing
                // key rather than compiling `{name, strict}` as a schema.
                None if wrapper.get("name").is_some() || wrapper.get("strict").is_some() => {
                    return Err("response_format.json_schema requires `schema`".to_string());
                }
                None => wrapper,
            };
            if !schema.is_object() {
                return Err("response_format.json_schema must be a JSON object".to_string());
            }
            Ok(Some(ConstraintSpec::JsonSchema {
                schema: schema.clone(),
            }))
        }
        other => Err(format!(
            "response_format type `{other}` is not supported; use `text`, \
             `json_object`, or `json_schema`"
        )),
    }
}

impl ChatCompletionRequest {
    /// vLLM's `prompt_logprobs`: how many alternatives to report with each
    /// prompt token, or `None` for none; refused when streaming, as vLLM
    /// does.
    pub fn prompt_logprob_count(&self) -> Result<Option<u32>, String> {
        let count = self
            .prompt_logprobs
            .map(|n| crate::logprobs::count("prompt_logprobs", n))
            .transpose()?;
        if count.is_some() && self.stream == Some(true) {
            return Err("`prompt_logprobs` is not available when streaming".to_string());
        }
        Ok(count)
    }

    /// [`ChatCompletionRequest::to_generation_options_with`] over the
    /// OpenAI profile's defaults alone (no server flags or GGUF values).
    pub fn to_generation_options(&self) -> Result<GenerationOptions, String> {
        self.to_generation_options_with(&crate::sampling::SamplingProfile::OpenAi.defaults())
    }

    /// Derive [`GenerationOptions`] from the request, taking every sampler
    /// setting it leaves unset from `defaults` (see
    /// [`crate::Engine::sampling_defaults`]).
    ///
    /// Returns a client-facing error message for values that are out of
    /// range, contradict each other, or ask for something the engine cannot
    /// do (see the type-level docs).
    pub fn to_generation_options_with(
        &self,
        defaults: &crate::sampling::SamplingDefaults,
    ) -> Result<GenerationOptions, String> {
        if self.n.is_some_and(|n| n != 1) {
            return Err("only n = 1 is supported".to_string());
        }
        self.named_tool()?;
        let logprobs = match (self.logprobs, self.top_logprobs) {
            (Some(true), top) => Some(crate::logprobs::count(
                "top_logprobs",
                i64::from(top.unwrap_or(0)),
            )?),
            (_, Some(n)) if n > 0 => {
                return Err("`top_logprobs` needs `logprobs: true`".to_string())
            }
            _ => None,
        };
        let constraint = match &self.response_format {
            Some(format) => parse_response_format(format)?,
            None => None,
        };
        if self.context_window == Some(0) {
            return Err("context_window must be at least 1".to_string());
        }

        // ── Reasoning: reconcile the equivalent spellings ───────────────────
        let reasoning = self.reasoning.clone().unwrap_or_default();
        let effort = match (self.reasoning_effort, reasoning.effort) {
            (Some(a), Some(b)) if a != b => {
                return Err("reasoning_effort and reasoning.effort disagree".to_string())
            }
            (a, b) => a.or(b),
        };
        let mut enabled = None;
        for (name, value) in [
            (
                "reasoning_effort",
                effort.map(ReasoningEffort::enables_thinking),
            ),
            ("reasoning.enabled", reasoning.enabled),
            ("enable_thinking", self.enable_thinking),
        ] {
            match (enabled, value) {
                (Some((first, a)), Some(b)) if a != b => {
                    return Err(format!(
                        "{first} and {name} disagree about whether reasoning is on"
                    ))
                }
                (None, Some(b)) => enabled = Some((name, b)),
                _ => {}
            }
        }
        let mut chat_template_kwargs =
            reasoning_template_kwargs(effort, enabled.map(|(_, on)| on), reasoning.max_tokens);
        if let Some(documents) = &self.documents {
            chat_template_kwargs.insert(
                "documents".to_string(),
                serde_json::Value::Array(documents.clone()),
            );
        }
        if let Some(extra) = &self.chat_template_kwargs {
            if let Some(key) = extra
                .keys()
                .find(|k| RESERVED_TEMPLATE_VARS.contains(&k.as_str()))
            {
                return Err(format!("chat_template_kwargs may not set `{key}`"));
            }
            chat_template_kwargs.extend(extra.iter().map(|(k, v)| (k.clone(), v.clone())));
        }

        // ── Template controls ───────────────────────────────────────────────
        let continue_final_message = match &self.continue_final_message {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::Bool(b)) => Some(*b),
            Some(serde_json::Value::String(s)) if s == "content" || s == "reasoning_content" => {
                Some(true)
            }
            Some(_) => {
                return Err("continue_final_message must be a boolean, \"content\" or \
                            \"reasoning_content\""
                    .to_string())
            }
        };
        if continue_final_message == Some(true) && self.add_generation_prompt == Some(true) {
            return Err(
                "Cannot set both add_generation_prompt and continue_final_message to true."
                    .to_string(),
            );
        }
        let reasoning_budget = match self.thinking_token_budget {
            None | Some(-1) => reasoning.max_tokens,
            Some(n) if n < 0 => {
                return Err("thinking_token_budget must be -1 or a non-negative count".to_string())
            }
            Some(n) => Some(u32::try_from(n).unwrap_or(u32::MAX)),
        };

        let mut options = GenerationOptions::with_defaults(defaults);
        self.sampling.apply(&mut options)?;
        options.context_window = self.context_window;
        options.chat_template_kwargs = chat_template_kwargs;
        options.chat_template = self.chat_template.clone();
        options.add_generation_prompt =
            self.add_generation_prompt.unwrap_or(true) && continue_final_message != Some(true);
        options.continue_final_message = continue_final_message;
        options.reasoning_budget = reasoning_budget;
        options.reasoning_budget_message = self.reasoning_budget_message.clone();
        options.constraint = constraint;
        options.logprobs = logprobs;
        Ok(options)
    }

    /// How the response reports reasoning, from `reasoning_format` and
    /// `include_reasoning`; `default` (the server's `--reasoning-format`)
    /// when the request names no format.
    pub fn reasoning_output(
        &self,
        default: crate::reasoning::ReasoningOutput,
    ) -> Result<crate::reasoning::ReasoningOutput, String> {
        use crate::reasoning::ReasoningOutput;
        if self.include_reasoning == Some(false) {
            return Ok(ReasoningOutput::Hidden);
        }
        self.reasoning_format
            .as_deref()
            .map_or(Ok(default), ReasoningOutput::parse)
    }

    /// The function a named `tool_choice` picks, after checking
    /// `tool_choice`: `"none"`, `"auto"`, `"required"` (which needs tools)
    /// or a named function among `tools`.
    fn named_tool(&self) -> Result<Option<&str>, String> {
        let has = |name: &str| {
            self.tools
                .iter()
                .flatten()
                .any(|tool| tool.function.name == name)
        };
        match &self.tool_choice {
            None | Some(serde_json::Value::Null) => Ok(None),
            Some(serde_json::Value::String(choice)) => match choice.as_str() {
                "none" | "auto" => Ok(None),
                "required" if self.tools.as_ref().is_some_and(|t| !t.is_empty()) => Ok(None),
                "required" => Err("tool_choice `required` needs `tools`".to_string()),
                other => Err(format!(
                    "unknown tool_choice `{other}`; use none, auto, required or a named function"
                )),
            },
            Some(choice) => {
                let name = choice
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(serde_json::Value::as_str)
                    .filter(|_| {
                        choice.get("type").and_then(serde_json::Value::as_str) == Some("function")
                    })
                    .ok_or_else(|| {
                        "tool_choice must be a string or {\"type\": \"function\", \"function\": {\"name\": ...}}"
                            .to_string()
                    })?;
                if has(name) {
                    Ok(Some(name))
                } else {
                    Err(format!("tool_choice names `{name}`, which is not in `tools`"))
                }
            }
        }
    }

    /// The tools to offer the model: `None` when no tools were supplied or
    /// `tool_choice` is `"none"`, and only the named one when `tool_choice`
    /// names a function.
    pub fn offered_tools(&self) -> Option<Vec<Tool>> {
        if self.tool_choice.as_ref().and_then(|c| c.as_str()) == Some("none") {
            return None;
        }
        match self.named_tool() {
            Ok(Some(name)) => self.tools.as_ref().map(|tools| {
                tools
                    .iter()
                    .filter(|tool| tool.function.name == name)
                    .cloned()
                    .collect()
            }),
            _ => self.tools.clone(),
        }
    }
}

/// An OpenAI tool (function) definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Must be `"function"`.
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Function metadata.
    pub function: FunctionDef,
}

/// Metadata for a callable function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    /// Function name.
    pub name: String,
    /// Human-readable description shown to the model.
    pub description: Option<String>,
    /// JSON schema describing the parameters.
    pub parameters: Option<serde_json::Value>,
}

/// OpenAI-compatible `POST /v1/chat/completions` response body.
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    /// Unique completion identifier.
    pub id: String,
    /// Always `"chat.completion"`.
    pub object: String,
    /// Unix timestamp of when the completion was created.
    pub created: u64,
    /// The model that generated the response.
    pub model: String,
    /// One or more generated alternatives.
    pub choices: Vec<ChatChoice>,
    /// Token usage breakdown.
    pub usage: UsageInfo,
    /// llama-server's prompt and generation timings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings: Option<Timings>,
    /// vLLM's `prompt_logprobs`, when the request asked for them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_logprobs: Option<serde_json::Value>,
}

/// llama-server's `timings` object (`server_slot_stats::to_json`): prompt
/// and generation token counts, wall times and rates.
///
/// `prompt_n` counts the prompt tokens actually processed (the reused
/// prefix is `cache_n`).  The generation rate is over every generated
/// token, the same figure as the engine's `decode_tps`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Timings {
    /// Prompt tokens reused from cached state.
    pub cache_n: u32,
    /// Prompt tokens processed.
    pub prompt_n: u32,
    /// Prompt processing time.
    pub prompt_ms: f64,
    /// `prompt_ms / prompt_n`.
    pub prompt_per_token_ms: f64,
    /// Prompt tokens per second.
    pub prompt_per_second: f64,
    /// Generated tokens.
    pub predicted_n: u32,
    /// Generation time.
    pub predicted_ms: f64,
    /// `predicted_ms / predicted_n`.
    pub predicted_per_token_ms: f64,
    /// Generated tokens per second.
    pub predicted_per_second: f64,
}

impl Timings {
    /// Derive the rates from the counts and times (0 where a divisor is 0,
    /// as llama-server reports them).
    pub fn new(
        cache_n: u32,
        prompt_n: u32,
        prompt_ms: f64,
        predicted_n: u32,
        predicted_ms: f64,
    ) -> Self {
        let per_token = |ms: f64, n: u32| if n > 0 { ms / n as f64 } else { 0.0 };
        let per_second = |ms: f64, n: u32| if ms > 0.0 { 1e3 / ms * n as f64 } else { 0.0 };
        Self {
            cache_n,
            prompt_n,
            prompt_ms,
            prompt_per_token_ms: per_token(prompt_ms, prompt_n),
            prompt_per_second: per_second(prompt_ms, prompt_n),
            predicted_n,
            predicted_ms,
            predicted_per_token_ms: per_token(predicted_ms, predicted_n),
            predicted_per_second: per_second(predicted_ms, predicted_n),
        }
    }
}

impl ChatCompletionResponse {
    /// Construct a new response with the current timestamp.
    pub fn new(id: String, model: String, choices: Vec<ChatChoice>, usage: UsageInfo) -> Self {
        let created = unix_time();
        Self {
            id,
            object: "chat.completion".to_string(),
            created,
            model,
            choices,
            usage,
            timings: None,
            prompt_logprobs: None,
        }
    }
}

/// A single choice in a [`ChatCompletionResponse`].
#[derive(Debug, Serialize)]
pub struct ChatChoice {
    /// Zero-based index of this choice.
    pub index: u32,
    /// The generated assistant message.
    pub message: AssistantMessage,
    /// Each generated token's log-probability (`{"content": [...]}`), when
    /// the request asked for them; `null` otherwise, as OpenAI sends it.
    pub logprobs: Option<serde_json::Value>,
    /// Why generation stopped (`"stop"`, `"length"`, `"tool_calls"`).
    pub finish_reason: String,
}

/// An assistant message inside a response choice.
#[derive(Debug, Serialize)]
pub struct AssistantMessage {
    /// Always `"assistant"`.
    pub role: String,
    /// Text content (may be `None` when tool calls are present).
    pub content: Option<String>,
    /// The model's reasoning, split from `content` (llama.cpp and the
    /// DeepSeek API's field name).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// The same reasoning under vLLM's field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Parsed tool calls from the model output, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// A single tool call emitted by the assistant.
#[derive(Debug, Serialize)]
pub struct ToolCall {
    /// Unique call identifier.
    pub id: String,
    /// Always `"function"`.
    #[serde(rename = "type")]
    pub call_type: String,
    /// The function invocation details.
    pub function: FunctionCallResult,
}

/// Details of a function call inside a [`ToolCall`].
#[derive(Debug, Serialize)]
pub struct FunctionCallResult {
    /// Name of the function to call.
    pub name: String,
    /// JSON-encoded argument object.
    pub arguments: String,
}

// ─── Streaming types ──────────────────────────────────────────────────────────

/// A single SSE chunk returned when `stream: true` is requested.
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    /// Matches the parent completion ID.
    pub id: String,
    /// Always `"chat.completion.chunk"`.
    pub object: String,
    /// Unix timestamp.
    pub created: u64,
    /// Model name.
    pub model: String,
    /// Delta choices.
    pub choices: Vec<StreamChoice>,
}

impl ChatCompletionChunk {
    /// Create a new chunk with the current timestamp.
    pub fn new(
        id: String,
        model: String,
        delta: DeltaContent,
        finish_reason: Option<String>,
    ) -> Self {
        let created = unix_time();
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created,
            model,
            choices: vec![StreamChoice {
                index: 0,
                delta,
                finish_reason,
            }],
        }
    }
}

/// A choice delta inside a streaming chunk.
#[derive(Debug, Serialize)]
pub struct StreamChoice {
    /// Zero-based index.
    pub index: u32,
    /// The incremental content for this step.
    pub delta: DeltaContent,
    /// Non-null on the final chunk.
    pub finish_reason: Option<String>,
}

/// Incremental content in a streaming chunk.
#[derive(Debug, Serialize, Default)]
pub struct DeltaContent {
    /// Present only on the first chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// The new token(s) generated in this step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// New reasoning text (llama.cpp / DeepSeek field name).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// The same reasoning text under vLLM's field name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Tool-call deltas (OpenAI wire format, including `index` per entry).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
}

// ─── Embeddings ───────────────────────────────────────────────────────────────

/// OpenAI-compatible `POST /v1/embeddings` request body.
#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    /// Model identifier.
    pub model: String,
    /// What to embed: a string, an array of this model's token ids, or an
    /// array of either (OpenAI's and vLLM's forms).
    pub input: serde_json::Value,
    /// `"float"` (default) or `"base64"`: each vector's little-endian
    /// float32 bytes, base64-encoded, which the OpenAI SDKs ask for by
    /// default.
    #[serde(default)]
    pub encoding_format: Option<String>,
    /// Keep only the vector's first `dimensions` values, L2-normalized
    /// again (OpenAI's `dimensions`).  Meaningful only for a model trained
    /// for it (Matryoshka, as Qwen3-Embedding is).
    #[serde(default)]
    pub dimensions: Option<usize>,
    /// vLLM's switch for the tokenizer's BOS on a text input (on by
    /// default).
    #[serde(default)]
    pub add_special_tokens: Option<bool>,
    /// vLLM's `normalize`.  Vectors are always L2-normalized, so `false` is
    /// refused.
    #[serde(default)]
    pub normalize: Option<bool>,
}

impl EmbeddingRequest {
    /// The inputs as plain text, when every one is text (the form a `joshua
    /// route` coordinator splits across workers).
    pub fn texts(&self) -> Option<Vec<String>> {
        EmbeddingInput::deserialize(&self.input)
            .ok()
            .map(EmbeddingInput::into_vec)
    }
}

/// Embedding input — either a single string or an array of strings.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// A single text to embed.
    Single(String),
    /// Multiple texts to embed.
    Multiple(Vec<String>),
}

impl EmbeddingInput {
    /// Convert into a `Vec<String>`.
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::Single(s) => vec![s],
            Self::Multiple(v) => v,
        }
    }
}

/// OpenAI-compatible `POST /v1/embeddings` response body.
#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    /// Always `"list"`.
    pub object: String,
    /// One embedding per input text.
    pub data: Vec<EmbeddingData>,
    /// Model name.
    pub model: String,
    /// Token usage.
    pub usage: UsageInfo,
}

/// A single embedding vector in an [`EmbeddingResponse`].
#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    /// Always `"embedding"`.
    pub object: String,
    /// The vector, as the request's `encoding_format` asked.
    pub embedding: EmbeddingVector,
    /// Zero-based index into the input array.
    pub index: u32,
}

/// An embedding vector on the wire.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum EmbeddingVector {
    /// `encoding_format: "float"`: the values as JSON numbers.
    Float(Vec<f32>),
    /// `encoding_format: "base64"`: the little-endian float32 bytes,
    /// base64-encoded.
    Base64(String),
}

impl EmbeddingVector {
    /// `vector` as `encoding_format` (`float` when absent) asks, or why
    /// that format is not one.
    pub fn encode(vector: Vec<f32>, encoding_format: Option<&str>) -> Result<Self, String> {
        use base64::Engine as _;
        match encoding_format {
            None | Some("float") => Ok(Self::Float(vector)),
            Some("base64") => {
                let bytes: Vec<u8> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
                Ok(Self::Base64(
                    base64::engine::general_purpose::STANDARD.encode(bytes),
                ))
            }
            Some(other) => Err(format!(
                "unknown encoding_format `{other}`; use `float` or `base64`"
            )),
        }
    }
}

// ─── Models list ─────────────────────────────────────────────────────────────

/// OpenAI-compatible `GET /v1/models` response body.  A `joshua route`
/// coordinator lists its workers' entries as they sent them
/// (`serde_json::Value`), so fields it does not know pass through.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelListResponse<T = ModelCard> {
    /// Always `"list"`.
    pub object: String,
    /// Available models.
    pub data: Vec<T>,
}

impl<T> ModelListResponse<T> {
    /// The list holding `data`.
    pub fn new(data: Vec<T>) -> Self {
        Self {
            object: "list".to_string(),
            data,
        }
    }
}

/// One `/v1/models` entry, also returned by `GET /v1/models/{id}`:
/// OpenAI's four fields, the limits a client needs to size a request
/// (#260), vLLM's `ModelCard` fields, llama-server's `aliases`, `tags` and
/// `meta`, and joshua's own facts under `joshua`.  Everything past OpenAI's
/// fields is additive.  Built by [`crate::model_info::ModelInfo::card`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCard {
    /// Model identifier.
    pub id: String,
    /// Always `"model"`.
    #[serde(default)]
    pub object: String,
    /// Unix timestamp of when the model was loaded (for a file that is not
    /// loaded, when it was last modified).
    #[serde(default)]
    pub created: u64,
    /// Always `"joshua"`.
    #[serde(default)]
    pub owned_by: String,
    /// Most tokens, prompt and output together, one request may use: the
    /// context window the model is loaded with (`--n-ctx`), which also caps
    /// a request's `context_window` (`n_ctx`, `num_ctx`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    /// The context the model was trained with (`<arch>.context_length`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_context_length: Option<u32>,
    /// Largest value `max_tokens` may take for this model: the server's
    /// `--max-output-tokens` ceiling.  Omitted for entries that cannot
    /// report it (e.g. the Whisper model).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// vLLM: the model's file, or its id when it has none.
    #[serde(default)]
    pub root: Option<String>,
    /// vLLM: the base model of an adapter; `null` for a base model.
    #[serde(default)]
    pub parent: Option<String>,
    /// vLLM: the same as `context_length`.
    #[serde(default)]
    pub max_model_len: Option<u32>,
    /// vLLM: one permissive permission record.
    #[serde(default)]
    pub permission: Vec<ModelPermission>,
    /// llama-server: other names the model answers to.
    #[serde(default)]
    pub aliases: Vec<String>,
    /// llama-server: informational tags.
    #[serde(default)]
    pub tags: Vec<String>,
    /// llama-server: what the GGUF says about the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<LlamaCppModelMeta>,
    /// joshua's own facts: whether the model is loaded, and where it runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub joshua: Option<JoshuaModelExtras>,
}

/// vLLM's `ModelPermission`, with its defaults.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelPermission {
    /// `modelperm-` and a random hex id.
    pub id: String,
    /// Always `"model_permission"`.
    pub object: String,
    pub created: u64,
    pub allow_create_engine: bool,
    pub allow_sampling: bool,
    pub allow_logprobs: bool,
    pub allow_search_indices: bool,
    pub allow_view: bool,
    pub allow_fine_tuning: bool,
    pub organization: String,
    pub group: Option<String>,
    pub is_blocking: bool,
}

impl ModelPermission {
    /// vLLM's default permission record, created at `created`.
    pub fn new(created: u64) -> Self {
        Self {
            id: format!("modelperm-{}", uuid::Uuid::new_v4().simple()),
            object: "model_permission".to_string(),
            created,
            allow_create_engine: false,
            allow_sampling: true,
            allow_logprobs: true,
            allow_search_indices: false,
            allow_view: true,
            allow_fine_tuning: false,
            organization: "*".to_string(),
            group: None,
            is_blocking: false,
        }
    }
}

/// llama-server's `meta` object.  Values the GGUF does not give are left
/// out.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlamaCppModelMeta {
    /// llama.cpp's `llama_vocab_type` (1 SPM, 2 BPE, 3 WPM, 4 UGM, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocab_type: Option<u32>,
    /// Tokens in the vocabulary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_vocab: Option<u32>,
    /// The context one request may use (`context_length`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_ctx: Option<u32>,
    /// The context the model was trained with.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_ctx_train: Option<u32>,
    /// Embedding width.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_embd: Option<u32>,
    /// Parameters across every tensor.
    #[serde(default)]
    pub n_params: u64,
    /// Bytes of tensor data.
    #[serde(default)]
    pub size: u64,
    /// llama.cpp's name for the file type (`"Q4_K - Medium"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ftype: Option<String>,
}

/// joshua's own facts about a `/v1/models` entry, under a key of their own
/// so no upstream field is shadowed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoshuaModelExtras {
    /// Whether the model is loaded.  With `--model-dir`, the directory's
    /// files are listed too; `POST /v1/models/load` loads one.
    pub loaded: bool,
    /// `general.architecture` (`"qwen3"`, `"deepseek2"`, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    /// The quantization, as Ollama names it (`"Q4_K_M"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    /// The model file's size in bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_size: Option<u64>,
    /// Where a loaded model runs.
    #[serde(default, flatten, skip_serializing_if = "Option::is_none")]
    pub placement: Option<crate::model_info::Placement>,
}

// ─── Error response ───────────────────────────────────────────────────────────

/// OpenAI-compatible error envelope.
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    /// Error details.
    pub error: ErrorDetail,
}

/// Inner error detail returned by the API.
#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    /// Human-readable message.
    pub message: String,
    /// Machine-readable error type (e.g. `"invalid_request_error"`).
    #[serde(rename = "type")]
    pub error_type: String,
    /// The parameter that caused the error, if applicable.
    pub param: Option<String>,
    /// Machine-readable error code, if applicable.
    pub code: Option<String>,
}

impl ErrorResponse {
    /// Create a generic error response.
    pub fn new(message: impl Into<String>, error_type: impl Into<String>) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
                error_type: error_type.into(),
                param: None,
                code: None,
            },
        }
    }

    /// Create an `invalid_request_error` response.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(message, "invalid_request_error")
    }

    /// Create a `server_error` response.
    pub fn server_error(message: impl Into<String>) -> Self {
        Self::new(message, "server_error")
    }

    /// OpenAI's answer for a model it cannot serve, saying why in
    /// `message`: `invalid_request_error`, `param: "model"`,
    /// `code: "model_not_found"`.
    pub fn model_not_found(message: impl Into<String>) -> Self {
        let mut body = Self::invalid_request(message);
        body.error.param = Some("model".to_string());
        body.error.code = Some("model_not_found".to_string());
        body
    }

    /// [`Self::model_not_found`] for an id that names no model, in OpenAI's
    /// words.
    pub fn unknown_model(id: &str) -> Self {
        Self::model_not_found(format!("The model `{id}` does not exist."))
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeddings_encode_as_float_or_little_endian_base64() {
        let vector = vec![1.0f32, -0.5];
        assert_eq!(
            EmbeddingVector::encode(vector.clone(), None),
            Ok(EmbeddingVector::Float(vector.clone()))
        );
        // 1.0 = 00 00 80 3f, -0.5 = 00 00 00 bf.
        assert_eq!(
            EmbeddingVector::encode(vector.clone(), Some("base64")),
            Ok(EmbeddingVector::Base64("AACAPwAAAL8=".to_string()))
        );
        assert!(EmbeddingVector::encode(vector, Some("int8")).is_err());
        let request: EmbeddingRequest =
            serde_json::from_value(serde_json::json!({"model": "m", "input": "a"})).unwrap();
        assert_eq!(request.texts(), Some(vec!["a".to_string()]));
        let request: EmbeddingRequest =
            serde_json::from_value(serde_json::json!({"model": "m", "input": [1, 2]})).unwrap();
        assert_eq!(request.texts(), None);
    }
    use serde_json::json;

    #[test]
    fn rfc3339_writes_utc_calendar_time() {
        use std::time::Duration;
        let at = |secs: u64, nanos: u32| rfc3339(UNIX_EPOCH + Duration::new(secs, nanos));
        assert_eq!(at(0, 0), "1970-01-01T00:00:00.000000000Z");
        assert_eq!(at(951_782_400, 5), "2000-02-29T00:00:00.000000005Z");
        assert_eq!(at(1_793_951_282, 123_456_789), "2026-11-06T07:48:02.123456789Z");
        assert_eq!(at(4_102_444_799, 0), "2099-12-31T23:59:59.000000000Z");
    }

    fn request(extra: serde_json::Value) -> ChatCompletionRequest {
        let mut body = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        });
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        serde_json::from_value(body).expect("request deserialises")
    }

    fn options(extra: serde_json::Value) -> GenerationOptions {
        request(extra)
            .to_generation_options()
            .expect("valid request")
    }

    fn rejected(extra: serde_json::Value) -> String {
        request(extra)
            .to_generation_options()
            .expect_err("request should be rejected")
    }

    #[test]
    fn defaults_leave_new_settings_off() {
        let o = options(json!({}));
        // No max_tokens: until end of sequence (the server caps it).
        assert_eq!(o.max_tokens, MAX_TOKENS_UNLIMITED);
        assert_eq!(o.presence_penalty, 0.0);
        assert_eq!(o.frequency_penalty, 0.0);
        assert_eq!(o.seed, None);
        assert_eq!(o.context_window, None);
        assert!(
            o.chat_template_kwargs.is_empty(),
            "no reasoning vars by default"
        );
        // The OpenAI profile: neutral sampling.
        assert_eq!(
            (o.temperature, o.top_p, o.top_k, o.min_p),
            (1.0, 1.0, 0, 0.0)
        );
        assert_eq!(o.repetition_penalty, 1.0);
        assert_eq!(o.repeat_last_n, 64);
        // Every sampler extra is off.
        assert_eq!(o.typical_p, 1.0);
        assert!(o.top_n_sigma <= 0.0);
        assert_eq!((o.xtc_probability, o.mirostat), (0.0, 0));
        assert!(!o.dry.enabled());
        assert!(o.logit_bias.is_empty() && o.allowed_token_ids.is_none());
        assert!(o.bad_words.is_empty() && o.stop_token_ids.is_empty());
        assert!(!o.ignore_eos && o.min_tokens == 0 && !o.include_stop_str_in_output);
        assert_eq!(o.reasoning_budget, None);
        assert!(o.add_generation_prompt);
        assert_eq!(o.continue_final_message, None);
        assert!(!o.needs_plain_decode());
    }

    #[test]
    fn max_tokens_spellings_and_unlimited_values() {
        assert_eq!(
            options(json!({"max_tokens": -1})).max_tokens,
            MAX_TOKENS_UNLIMITED
        );
        assert_eq!(options(json!({"max_tokens": 0})).max_tokens, 0);
        // llama.cpp's n_predict wins over both OpenAI spellings.
        assert_eq!(
            options(json!({"max_tokens": 5, "max_completion_tokens": 6, "n_predict": 7}))
                .max_tokens,
            7
        );
        assert_eq!(
            options(json!({"n_predict": -1, "max_tokens": 5})).max_tokens,
            MAX_TOKENS_UNLIMITED
        );
        assert!(rejected(json!({"max_tokens": -2})).contains("max_tokens"));
        // Huge values saturate rather than fail.
        assert_eq!(
            options(json!({"max_tokens": 1u64 << 40})).max_tokens,
            MAX_TOKENS_UNLIMITED
        );
    }

    #[test]
    fn server_and_gguf_defaults_apply_under_request_values() {
        let defaults = crate::sampling::SamplingDefaults {
            temperature: 0.6,
            top_k: 20,
            ..crate::sampling::SamplingProfile::OpenAi.defaults()
        };
        let o = request(json!({}))
            .to_generation_options_with(&defaults)
            .unwrap();
        assert_eq!((o.temperature, o.top_k), (0.6, 20));
        let o = request(json!({"temperature": 0.1}))
            .to_generation_options_with(&defaults)
            .unwrap();
        assert_eq!((o.temperature, o.top_k), (0.1, 20));
    }

    #[test]
    fn sampler_extras_map_from_the_request() {
        let o = options(json!({
            "repeat_penalty": 1.2,
            "repeat_last_n": -1,
            "typical_p": 0.9,
            "top_n_sigma": 1.5,
            "xtc_probability": 0.5,
            "xtc_threshold": 0.2,
            "mirostat": 2,
            "mirostat_tau": 4.0,
            "mirostat_eta": 0.2,
            "dry_multiplier": 0.8,
            "dry_base": 0.5,
            "dry_allowed_length": 3,
            "dry_penalty_last_n": 128,
            "dry_sequence_breakers": ["\n", "."],
            "min_tokens": 4,
            "ignore_eos": true,
            "stop_token_ids": [7, 8],
            "include_stop_str_in_output": true,
            "allowed_token_ids": [1, 2, 3],
            "bad_words": ["foo"],
            "truncate_prompt_tokens": 100,
            "stop": ["a", "", "b"],
        }));
        assert_eq!(o.repetition_penalty, 1.2);
        assert_eq!(o.repeat_last_n, -1);
        assert_eq!(o.typical_p, 0.9);
        assert_eq!(o.top_n_sigma, 1.5);
        assert_eq!((o.xtc_probability, o.xtc_threshold), (0.5, 0.2));
        assert_eq!((o.mirostat, o.mirostat_tau, o.mirostat_eta), (2, 4.0, 0.2));
        assert_eq!(o.dry.multiplier, 0.8);
        assert_eq!(o.dry.base, 1.75, "a base below 1 keeps the default");
        assert_eq!((o.dry.allowed_length, o.dry.penalty_last_n), (3, 128));
        assert_eq!(
            o.dry.sequence_breakers,
            vec!["\n".to_string(), ".".to_string()]
        );
        assert_eq!(o.min_tokens, 4);
        assert!(o.ignore_eos && o.include_stop_str_in_output);
        assert_eq!(o.stop_token_ids, vec![7, 8]);
        assert_eq!(o.allowed_token_ids, Some(vec![1, 2, 3]));
        assert_eq!(o.bad_words, vec!["foo".to_string()]);
        assert_eq!(o.truncate_prompt_tokens, Some(100));
        assert_eq!(o.stop_sequences, vec!["a".to_string(), "b".to_string()]);
        assert!(o.needs_plain_decode());

        assert!(rejected(json!({"mirostat": 3})).contains("mirostat"));
        assert!(rejected(json!({"dynatemp_range": 0.5})).contains("dynatemp"));
        assert!(rejected(json!({"dry_sequence_breakers": []})).contains("dry_sequence_breakers"));
        assert!(rejected(json!({"min_tokens": 9, "max_tokens": 4})).contains("min_tokens"));
        assert!(rejected(json!({"repeat_last_n": -2})).contains("repeat_last_n"));
        assert!(rejected(json!({"truncate_prompt_tokens": -2})).contains("truncate"));
        // Zero dynamic-temperature range is accepted (SillyTavern sends it).
        options(json!({"dynatemp_range": 0.0}));
        // vLLM's `-1` keeps what the context holds; a huge count stays a
        // count, for the engine to refuse.
        let truncate =
            |n: i64| options(json!({ "truncate_prompt_tokens": n })).truncate_prompt_tokens;
        assert_eq!(truncate(-1), Some(TRUNCATE_TO_CONTEXT));
        assert_eq!(truncate(1 << 40), Some(TRUNCATE_TO_CONTEXT - 1));
        assert_eq!(truncate(0), Some(0));
        // DRY scans llama.cpp's 64 tokens unless the request says otherwise.
        let dry = options(json!({"dry_multiplier": 0.8})).dry;
        assert_eq!(
            (dry.penalty_last_n, dry.allowed_length, dry.base),
            (64, 2, 1.75)
        );
    }

    #[test]
    fn logit_bias_accepts_openai_and_llama_cpp_shapes() {
        let o = options(json!({"logit_bias": {"15": 2.5, "hello": -1, "7": false, "x": "bad"}}));
        let mut got = o.logit_bias.clone();
        got.sort_by(|a, b| format!("{:?}", a.token).cmp(&format!("{:?}", b.token)));
        assert_eq!(
            got,
            vec![
                LogitBias {
                    token: TokenRef::Id(15),
                    bias: 2.5
                },
                LogitBias {
                    token: TokenRef::Id(7),
                    bias: f32::NEG_INFINITY
                },
                LogitBias {
                    token: TokenRef::Text("hello".into()),
                    bias: -1.0
                },
            ]
        );
        let o = options(json!({"logit_bias": [[3, 1.0], ["world", false], [1], "junk"]}));
        assert_eq!(
            o.logit_bias,
            vec![
                LogitBias {
                    token: TokenRef::Id(3),
                    bias: 1.0
                },
                LogitBias {
                    token: TokenRef::Text("world".into()),
                    bias: f32::NEG_INFINITY
                },
            ]
        );
        assert!(rejected(json!({"logit_bias": 5})).contains("logit_bias"));
        // vLLM's clamp: past f32's range a bias would be infinite, and an
        // infinite one makes the sampled distribution NaN.  `false` is
        // still the only ban.
        let o = options(json!({"logit_bias": [[1, 1e39], [2, -1e39], [3, 150], [4, -100.0]]}));
        let biases: Vec<f32> = o.logit_bias.iter().map(|b| b.bias).collect();
        assert_eq!(biases, vec![100.0, -100.0, 100.0, -100.0]);
    }

    #[test]
    fn template_and_reasoning_controls() {
        let o = options(json!({
            "continue_final_message": true,
            "add_generation_prompt": false,
            "chat_template": "{{ messages }}",
            "documents": [{"title": "t", "text": "x"}],
            "thinking_token_budget": 64,
            "reasoning_budget_message": "Enough.",
        }));
        assert_eq!(o.continue_final_message, Some(true));
        assert!(!o.add_generation_prompt);
        assert_eq!(o.chat_template.as_deref(), Some("{{ messages }}"));
        assert_eq!(o.chat_template_kwargs["documents"][0]["text"], json!("x"));
        assert_eq!(o.reasoning_budget, Some(64));
        assert_eq!(o.reasoning_budget_message.as_deref(), Some("Enough."));
        assert_eq!(
            options(json!({"continue_final_message": "content"})).continue_final_message,
            Some(true)
        );
        // llama.cpp spellings of the budget, and -1 = none.
        assert_eq!(
            options(json!({"reasoning_budget_tokens": 8})).reasoning_budget,
            Some(8)
        );
        assert_eq!(
            options(json!({"thinking_budget_tokens": 9})).reasoning_budget,
            Some(9)
        );
        assert_eq!(
            options(json!({"thinking_token_budget": -1})).reasoning_budget,
            None
        );
        // reasoning.max_tokens is now enforced too.
        assert_eq!(
            options(json!({"reasoning": {"max_tokens": 12}})).reasoning_budget,
            Some(12)
        );
        assert!(
            rejected(json!({"continue_final_message": true, "add_generation_prompt": true}))
                .contains("Cannot set both")
        );
        assert!(rejected(json!({"continue_final_message": 3})).contains("continue_final_message"));
        assert!(rejected(json!({"thinking_token_budget": -5})).contains("thinking_token_budget"));

        use crate::reasoning::ReasoningOutput;
        let output = |v: serde_json::Value| request(v).reasoning_output(Default::default());
        assert_eq!(output(json!({})), Ok(ReasoningOutput::Separate));
        assert_eq!(
            output(json!({"reasoning_format": "none"})),
            Ok(ReasoningOutput::Inline)
        );
        assert_eq!(
            output(json!({"reasoning_format": "deepseek-legacy"})),
            Ok(ReasoningOutput::Legacy)
        );
        assert_eq!(
            output(json!({"include_reasoning": false})),
            Ok(ReasoningOutput::Hidden)
        );
        assert!(output(json!({"reasoning_format": "xml"})).is_err());
        // The server's --reasoning-format applies when the request names none.
        let inline = |v: serde_json::Value| request(v).reasoning_output(ReasoningOutput::Inline);
        assert_eq!(inline(json!({})), Ok(ReasoningOutput::Inline));
        assert_eq!(
            inline(json!({"reasoning_format": "auto"})),
            Ok(ReasoningOutput::Separate)
        );
        // `max` is a valid effort, mapped to the 3-level templates' `high`.
        let o = options(json!({"reasoning_effort": "max"}));
        assert_eq!(o.chat_template_kwargs["reasoning_effort"], json!("high"));
    }

    #[test]
    fn reasoning_history_is_accepted_under_every_spelling() {
        for key in ["reasoning_content", "reasoning", "thinking"] {
            let m: ChatMessage = serde_json::from_value(json!({
                "role": "assistant", "content": "Hi", key: "greet"
            }))
            .unwrap();
            assert_eq!(m.reasoning_content.as_deref(), Some("greet"), "{key}");
        }
        let m: ChatMessage = serde_json::from_value(
            json!({"role": "assistant", "content": "Hi", "reasoning_content": ""}),
        )
        .unwrap();
        assert_eq!(m.reasoning_content, None);
    }

    #[test]
    fn timings_follow_llama_server() {
        let t = Timings::new(3, 10, 20.0, 4, 8.0);
        assert_eq!((t.cache_n, t.prompt_n, t.predicted_n), (3, 10, 4));
        assert_eq!(t.prompt_per_token_ms, 2.0);
        assert_eq!(t.prompt_per_second, 500.0);
        assert_eq!(t.predicted_per_token_ms, 2.0);
        assert_eq!(t.predicted_per_second, 500.0);
        let none = Timings::new(0, 0, 0.0, 0, 0.0);
        assert_eq!(
            (none.prompt_per_second, none.predicted_per_token_ms),
            (0.0, 0.0)
        );
        let v = serde_json::to_value(t).unwrap();
        for key in [
            "cache_n",
            "prompt_n",
            "prompt_ms",
            "prompt_per_token_ms",
            "prompt_per_second",
            "predicted_n",
            "predicted_ms",
            "predicted_per_token_ms",
            "predicted_per_second",
        ] {
            assert!(v.get(key).is_some(), "{key}");
        }
    }

    #[test]
    fn sampling_parameters_deserialise() {
        let o = options(json!({
            "max_tokens": 10,
            "max_completion_tokens": 20,
            "presence_penalty": 0.5,
            "frequency_penalty": -0.25,
            "seed": 42,
            "n": 1,
            "logprobs": false,
            "response_format": {"type": "text"},
            "stream_options": {"include_usage": true},
            "user": "ignored-unknown-field",
        }));
        assert_eq!(o.max_tokens, 20, "max_completion_tokens wins");
        assert_eq!(o.presence_penalty, 0.5);
        assert_eq!(o.frequency_penalty, -0.25);
        assert_eq!(o.seed, Some(42));
        // llama.cpp's -1 means "random".
        assert_eq!(options(json!({"seed": -1})).seed, None);
    }

    #[test]
    fn unsupported_parameters_are_rejected() {
        assert!(rejected(json!({"n": 2})).contains("n = 1"));
        assert!(rejected(json!({"top_logprobs": 3})).contains("logprobs: true"));
        assert!(rejected(json!({"logprobs": true, "top_logprobs": 21})).contains("between"));
        assert_eq!(options(json!({"logprobs": true})).logprobs, Some(0));
        assert_eq!(options(json!({"logprobs": true, "top_logprobs": 5})).logprobs, Some(5));
        assert_eq!(options(json!({"logprobs": false})).logprobs, None);
        assert_eq!(
            options(json!({"response_format": {"type": "json_object"}})).constraint,
            Some(ConstraintSpec::JsonObject)
        );
        let schema = json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"]
        });
        assert_eq!(
            options(json!({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "answer", "strict": true, "schema": schema}
                }
            }))
            .constraint,
            Some(ConstraintSpec::JsonSchema { schema })
        );
        assert!(
            rejected(json!({"response_format": {"type": "xml"}})).contains("xml"),
            "unknown response_format types are rejected"
        );
        assert!(rejected(json!({"presence_penalty": 2.5})).contains("presence_penalty"));
        assert!(rejected(json!({"frequency_penalty": -3})).contains("frequency_penalty"));
        assert!(rejected(json!({"context_window": 0})).contains("context_window"));
    }

    #[test]
    fn bare_schemas_need_no_top_level_type() {
        // JSON Schema allows `properties`, `allOf`, `$ref` … without `type`.
        let bare = json!({"properties": {"a": {"type": "integer"}}});
        assert_eq!(
            options(json!({
                "response_format": {"type": "json_schema", "json_schema": bare}
            }))
            .constraint,
            Some(ConstraintSpec::JsonSchema { schema: bare.clone() }),
            "a bare schema without `type` must be accepted"
        );
        let all_of = json!({"allOf": [{"type": "object"}]});
        assert_eq!(
            options(json!({
                "response_format": {"type": "json_schema", "json_schema": all_of}
            }))
            .constraint,
            Some(ConstraintSpec::JsonSchema { schema: all_of.clone() })
        );
        // A wrapper that carries `name`/`strict` but no schema body is still
        // an error rather than an empty schema.
        assert!(
            rejected(json!({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "answer", "strict": true}
                }
            }))
            .contains("schema")
        );
    }

    #[test]
    fn context_window_accepts_ollama_and_llama_cpp_spellings() {
        assert_eq!(
            options(json!({"context_window": 2048})).context_window,
            Some(2048)
        );
        assert_eq!(options(json!({"num_ctx": 1024})).context_window, Some(1024));
        assert_eq!(options(json!({"n_ctx": 512})).context_window, Some(512));
    }

    #[test]
    fn reasoning_effort_maps_to_template_vars() {
        let o = options(json!({"reasoning_effort": "high"}));
        assert_eq!(o.chat_template_kwargs["enable_thinking"], json!(true));
        assert_eq!(o.chat_template_kwargs["thinking"], json!(true));
        assert_eq!(o.chat_template_kwargs["reasoning_effort"], json!("high"));

        let o = options(json!({"reasoning_effort": "none"}));
        assert_eq!(o.chat_template_kwargs["enable_thinking"], json!(false));
        assert_eq!(o.chat_template_kwargs["thinking"], json!(false));
        assert_eq!(o.chat_template_kwargs["reasoning_effort"], json!("low"));

        let o = options(json!({"reasoning": {"effort": "minimal", "max_tokens": 256}}));
        assert_eq!(o.chat_template_kwargs["enable_thinking"], json!(true));
        assert_eq!(o.chat_template_kwargs["reasoning_effort"], json!("low"));
        assert_eq!(o.chat_template_kwargs["thinking_budget"], json!(256));

        // Unknown levels fail deserialisation (axum answers 422).
        assert!(serde_json::from_value::<ChatCompletionRequest>(json!({
            "model": "m", "messages": [], "reasoning_effort": "extreme"
        }))
        .is_err());
    }

    #[test]
    fn thinking_switch_spellings_and_conflicts() {
        for body in [
            json!({"enable_thinking": false}),
            json!({"reasoning": {"enabled": false}}),
            json!({"chat_template_kwargs": {"enable_thinking": false}}),
        ] {
            let o = options(body.clone());
            assert_eq!(
                o.chat_template_kwargs["enable_thinking"],
                json!(false),
                "{body}"
            );
        }
        // Thinking on with no effort leaves gpt-oss at its own default.
        let o = options(json!({"enable_thinking": true}));
        assert!(!o.chat_template_kwargs.contains_key("reasoning_effort"));

        assert!(
            rejected(json!({"reasoning_effort": "high", "enable_thinking": false}))
                .contains("disagree")
        );
        assert!(
            rejected(json!({"reasoning_effort": "none", "reasoning": {"enabled": true}}))
                .contains("disagree")
        );
        assert!(
            rejected(json!({"reasoning_effort": "low", "reasoning": {"effort": "high"}}))
                .contains("disagree")
        );
        // Agreeing spellings are fine.
        options(
            json!({"reasoning_effort": "low", "reasoning": {"effort": "low", "enabled": true}}),
        );
    }

    /// Regression (Devin review on #172): the CLI's `--reasoning-effort` and
    /// `--enable-thinking` must agree, as the HTTP spellings must.
    #[test]
    fn checked_reasoning_kwargs_reject_contradictory_cli_flags() {
        let err = checked_reasoning_template_kwargs(Some(ReasoningEffort::High), Some(false), None)
            .unwrap_err();
        assert!(err.contains("disagree"), "{err}");
        assert!(
            checked_reasoning_template_kwargs(Some(ReasoningEffort::None), Some(true), None)
                .is_err()
        );
        // Agreeing or single flags render exactly as the unchecked helper.
        for (effort, enabled) in [
            (Some(ReasoningEffort::High), Some(true)),
            (Some(ReasoningEffort::None), Some(false)),
            (Some(ReasoningEffort::Low), None),
            (None, Some(false)),
            (None, None),
        ] {
            assert_eq!(
                checked_reasoning_template_kwargs(effort, enabled, Some(7)).unwrap(),
                reasoning_template_kwargs(effort, enabled, Some(7))
            );
        }
    }

    #[test]
    fn chat_template_kwargs_override_derived_vars_but_not_reserved_ones() {
        let o = options(json!({
            "reasoning_effort": "high",
            "chat_template_kwargs": {"reasoning_effort": "medium", "custom_flag": 3},
        }));
        assert_eq!(o.chat_template_kwargs["reasoning_effort"], json!("medium"));
        assert_eq!(o.chat_template_kwargs["custom_flag"], json!(3));
        assert_eq!(o.chat_template_kwargs["enable_thinking"], json!(true));

        for key in RESERVED_TEMPLATE_VARS {
            let mut kwargs = serde_json::Map::new();
            kwargs.insert(key.to_string(), json!(1));
            let err = rejected(json!({ "chat_template_kwargs": kwargs }));
            assert!(err.contains(key), "{err}");
        }
    }

    #[test]
    fn tool_choice_none_hides_tools() {
        let tools = json!([{"type": "function", "function": {"name": "f"}}]);
        assert!(request(json!({"tools": tools})).offered_tools().is_some());
        assert!(request(json!({"tools": tools, "tool_choice": "auto"}))
            .offered_tools()
            .is_some());
        assert!(request(json!({"tools": tools, "tool_choice": "required"}))
            .offered_tools()
            .is_some());
        assert!(request(json!({"tools": tools, "tool_choice": "none"}))
            .offered_tools()
            .is_none());
    }

    #[test]
    fn a_named_tool_choice_offers_that_tool_alone() {
        let tools = json!([
            {"type": "function", "function": {"name": "f"}},
            {"type": "function", "function": {"name": "g"}},
        ]);
        let named = json!({"type": "function", "function": {"name": "g"}});
        let offered = request(json!({"tools": tools, "tool_choice": named}))
            .offered_tools()
            .unwrap();
        assert_eq!(offered.len(), 1);
        assert_eq!(offered[0].function.name, "g");
        // A choice that cannot be honoured is a client error.
        for (choice, needle) in [
            (json!({"type": "function", "function": {"name": "h"}}), "not in `tools`"),
            (json!("sometimes"), "unknown tool_choice"),
            (json!({"name": "f"}), "tool_choice must be"),
        ] {
            let err = request(json!({"tools": tools, "tool_choice": choice}))
                .to_generation_options()
                .unwrap_err();
            assert!(err.contains(needle), "{err}");
        }
        let err = request(json!({"tool_choice": "required"}))
            .to_generation_options()
            .unwrap_err();
        assert!(err.contains("needs `tools`"), "{err}");
    }

    #[test]
    fn generation_options_deserialise_with_new_fields_defaulted() {
        let o: GenerationOptions = serde_json::from_value(json!({"max_tokens": 5})).unwrap();
        assert_eq!(o.max_tokens, 5);
        assert_eq!(o.seed, None);
        assert!(o.chat_template_kwargs.is_empty());
    }
}

#[cfg(test)]
mod model_info_tests {
    use super::*;

    use crate::model_info::{ModelInfo, ModelLimits};

    fn info(limits: Option<ModelLimits>) -> ModelCard {
        let mut info = ModelInfo::new("Qwen3-14B-Q4_K_M", 0);
        info.limits = limits;
        info.card()
    }

    #[test]
    fn model_entry_reports_context_and_output_limits() {
        let limits = ModelLimits::new(16384, None, 4096);
        let value = serde_json::to_value(info(Some(limits))).unwrap();
        assert_eq!(value["context_length"], 16384);
        assert_eq!(value["max_output_tokens"], 4096);
        // Existing OpenAI fields must not move or disappear.
        assert_eq!(value["id"], "Qwen3-14B-Q4_K_M");
        assert_eq!(value["owned_by"], "joshua");
    }

    #[test]
    fn model_entry_omits_limits_it_cannot_report() {
        let value = serde_json::to_value(info(None)).unwrap();
        assert!(value.get("context_length").is_none());
        assert!(value.get("max_output_tokens").is_none());
    }
}
