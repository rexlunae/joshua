//! OpenAI-compatible request and response types used by Joshua.

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

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
        }
    }
}


// ─── Generation options ───────────────────────────────────────────────────────

/// Parameters that control the token-generation process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationOptions {
    /// Maximum number of tokens to generate.
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
    /// Repetition penalty (1.0 = disabled).
    #[serde(default = "GenerationOptions::default_repetition_penalty")]
    pub repetition_penalty: f32,
    /// OpenAI presence penalty (0 = disabled, range `-2.0..=2.0`).
    ///
    /// Subtracted once from the logit of every token that appears in the
    /// same recent-token window the repetition penalty uses (the last 64
    /// prompt/generated tokens).  Unlike the repetition penalty it is also
    /// applied at temperature 0, since it is never on by default.
    #[serde(default)]
    pub presence_penalty: f32,
    /// OpenAI frequency penalty (0 = disabled, range `-2.0..=2.0`).
    ///
    /// Subtracted from a token's logit once per occurrence in the recent
    /// window (see [`GenerationOptions::presence_penalty`]).
    #[serde(default)]
    pub frequency_penalty: f32,
    /// Seed for the sampler's random number generator.  `None` seeds from
    /// OS entropy.  A fixed seed reproduces the same sample stream for the
    /// same prompt, model, backend and options; greedy decoding never
    /// consumes randomness.
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
    /// Strings that will terminate generation when encountered.
    #[serde(default)]
    pub stop_sequences: Vec<String>,
}

impl GenerationOptions {
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
    fn default_repetition_penalty() -> f32 {
        1.1
    }
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: Self::default_max_tokens(),
            temperature: Self::default_temperature(),
            top_p: Self::default_top_p(),
            top_k: Self::default_top_k(),
            min_p: Self::default_min_p(),
            repetition_penalty: Self::default_repetition_penalty(),
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: None,
            context_window: None,
            chat_template_kwargs: serde_json::Map::new(),
            stop_sequences: vec![],
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
}

impl ReasoningEffort {
    /// Whether this effort level asks for thinking at all.
    pub fn enables_thinking(self) -> bool {
        self != Self::None
    }

    /// The `reasoning_effort` template value: the three levels effort-aware
    /// templates (gpt-oss) understand.  `none` and `minimal` map to `low`
    /// (gpt-oss cannot switch reasoning off), `xhigh` to `high`.
    pub fn template_level(self) -> &'static str {
        match self {
            Self::None | Self::Minimal | Self::Low => "low",
            Self::Medium => "medium",
            Self::High | Self::Xhigh => "high",
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
}

// ─── OpenAI chat completions ──────────────────────────────────────────────────

/// OpenAI-compatible `POST /v1/chat/completions` request body.
///
/// Unknown fields are ignored.  Fields that are accepted but cannot be
/// honoured are rejected by [`ChatCompletionRequest::to_generation_options`]
/// rather than silently dropped: `n > 1`, `logprobs: true`,
/// `top_logprobs > 0` and a `response_format` other than `text`.
#[derive(Debug, Default, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model identifier (e.g. `"joshua"` or path-derived name).
    pub model: String,
    /// Conversation history including the new user turn.
    pub messages: Vec<ChatMessage>,
    /// Upper bound on generated tokens.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// OpenAI's newer name for `max_tokens`; wins when both are set.
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    /// Sampling temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Top-p sampling.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Top-k sampling.
    #[serde(default)]
    pub top_k: Option<i32>,
    /// Min-p sampling.
    #[serde(default)]
    pub min_p: Option<f32>,
    /// Repetition penalty.
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
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
    /// Whether to stream token-by-token via SSE.
    #[serde(default)]
    pub stream: Option<bool>,
    /// Accepted for compatibility; the final streamed chunk always carries
    /// usage, so `include_usage` is effectively always on.
    #[serde(default)]
    pub stream_options: Option<serde_json::Value>,
    /// Tool/function definitions for tool-calling.
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    /// `"none"` hides `tools` from the model; `"auto"` (the default) offers
    /// them.  `"required"` and a named function cannot be enforced (there is
    /// no constrained decoding) and behave like `"auto"`.
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// Number of choices; only `1` is supported.
    #[serde(default)]
    pub n: Option<u32>,
    /// Token log-probabilities; not supported (`true` is rejected).
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// Top log-probabilities per token; not supported (`> 0` is rejected).
    #[serde(default)]
    pub top_logprobs: Option<u32>,
    /// Output format; only `{"type": "text"}` is supported.
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    /// Reasoning effort (`none`, `minimal`, `low`, `medium`, `high`,
    /// `xhigh`) — see [`reasoning_template_kwargs`] for the per-model
    /// mapping.
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Reasoning settings object: `{"effort", "enabled", "max_tokens"}`.
    #[serde(default)]
    pub reasoning: Option<ReasoningConfig>,
    /// Thinking on/off switch (Qwen / DashScope top-level spelling).
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    /// Extra chat-template variables (vLLM / llama.cpp); these win over
    /// the values derived from the reasoning settings above.
    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
    /// Per-request context window in tokens (also accepted as Ollama's
    /// `num_ctx` or `n_ctx`); must not exceed the server's `--n-ctx`.
    #[serde(default, alias = "num_ctx", alias = "n_ctx")]
    pub context_window: Option<u32>,
}

impl ChatCompletionRequest {
    /// Derive [`GenerationOptions`] from the request, applying defaults for
    /// missing fields.
    ///
    /// Returns a client-facing error message for values that are out of
    /// range, contradict each other, or ask for something the engine cannot
    /// do (see the type-level docs).
    pub fn to_generation_options(&self) -> Result<GenerationOptions, String> {
        if self.n.is_some_and(|n| n != 1) {
            return Err("only n = 1 is supported".to_string());
        }
        if self.logprobs == Some(true) || self.top_logprobs.is_some_and(|n| n > 0) {
            return Err("logprobs are not supported".to_string());
        }
        if let Some(format) = &self.response_format {
            let kind = format.get("type").and_then(|t| t.as_str());
            if kind != Some("text") {
                return Err(format!(
                    "response_format {} is not supported (no constrained decoding); \
                     only {{\"type\": \"text\"}} is accepted",
                    kind.unwrap_or("<missing type>")
                ));
            }
        }
        for (name, value) in [
            ("presence_penalty", self.presence_penalty),
            ("frequency_penalty", self.frequency_penalty),
        ] {
            if value.is_some_and(|v| !(-2.0..=2.0).contains(&v)) {
                return Err(format!("{name} must be between -2.0 and 2.0"));
            }
        }
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
        if let Some(extra) = &self.chat_template_kwargs {
            if let Some(key) = extra
                .keys()
                .find(|k| RESERVED_TEMPLATE_VARS.contains(&k.as_str()))
            {
                return Err(format!("chat_template_kwargs may not set `{key}`"));
            }
            chat_template_kwargs.extend(extra.iter().map(|(k, v)| (k.clone(), v.clone())));
        }

        let defaults = GenerationOptions::default();
        let stop_sequences = match &self.stop {
            Some(serde_json::Value::String(s)) => vec![s.clone()],
            Some(serde_json::Value::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect(),
            _ => vec![],
        };
        Ok(GenerationOptions {
            max_tokens: self
                .max_completion_tokens
                .or(self.max_tokens)
                .unwrap_or(defaults.max_tokens),
            temperature: self.temperature.unwrap_or(defaults.temperature),
            top_p: self.top_p.unwrap_or(defaults.top_p),
            top_k: self.top_k.unwrap_or(defaults.top_k),
            min_p: self.min_p.unwrap_or(defaults.min_p),
            repetition_penalty: self.repetition_penalty.unwrap_or(defaults.repetition_penalty),
            presence_penalty: self.presence_penalty.unwrap_or(0.0),
            frequency_penalty: self.frequency_penalty.unwrap_or(0.0),
            seed: self.seed.and_then(|s| u64::try_from(s).ok()),
            context_window: self.context_window,
            chat_template_kwargs,
            stop_sequences,
        })
    }

    /// The tools to offer the model: `None` when no tools were supplied or
    /// `tool_choice` is `"none"`.
    pub fn offered_tools(&self) -> Option<Vec<Tool>> {
        if self.tool_choice.as_ref().and_then(|c| c.as_str()) == Some("none") {
            return None;
        }
        self.tools.clone()
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
}

impl ChatCompletionResponse {
    /// Construct a new response with the current timestamp.
    pub fn new(id: String, model: String, choices: Vec<ChatChoice>, usage: UsageInfo) -> Self {
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            id,
            object: "chat.completion".to_string(),
            created,
            model,
            choices,
            usage,
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
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
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
    /// One or more texts to embed.
    pub input: EmbeddingInput,
    /// `"float"` (default) or `"base64"`.
    #[serde(default)]
    pub encoding_format: Option<String>,
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
    /// Dense float vector.
    pub embedding: Vec<f32>,
    /// Zero-based index into the input array.
    pub index: u32,
}

// ─── Models list ─────────────────────────────────────────────────────────────

/// OpenAI-compatible `GET /v1/models` response body.
#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    /// Always `"list"`.
    pub object: String,
    /// Available models.
    pub data: Vec<ModelInfo>,
}

/// Metadata for a single model.
#[derive(Debug, Serialize)]
pub struct ModelInfo {
    /// Model identifier.
    pub id: String,
    /// Always `"model"`.
    pub object: String,
    /// Unix timestamp of when the model was registered.
    pub created: u64,
    /// Always `"joshua"`.
    pub owned_by: String,
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
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
        assert_eq!(o.max_tokens, 256);
        assert_eq!(o.presence_penalty, 0.0);
        assert_eq!(o.frequency_penalty, 0.0);
        assert_eq!(o.seed, None);
        assert_eq!(o.context_window, None);
        assert!(
            o.chat_template_kwargs.is_empty(),
            "no reasoning vars by default"
        );
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
        assert!(rejected(json!({"logprobs": true})).contains("logprobs"));
        assert!(rejected(json!({"top_logprobs": 3})).contains("logprobs"));
        assert!(
            rejected(json!({"response_format": {"type": "json_object"}})).contains("json_object")
        );
        assert!(rejected(json!({"presence_penalty": 2.5})).contains("presence_penalty"));
        assert!(rejected(json!({"frequency_penalty": -3})).contains("frequency_penalty"));
        assert!(rejected(json!({"context_window": 0})).contains("context_window"));
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
    fn generation_options_deserialise_with_new_fields_defaulted() {
        let o: GenerationOptions = serde_json::from_value(json!({"max_tokens": 5})).unwrap();
        assert_eq!(o.max_tokens, 5);
        assert_eq!(o.seed, None);
        assert!(o.chat_template_kwargs.is_empty());
    }
}
