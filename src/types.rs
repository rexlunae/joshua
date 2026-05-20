//! OpenAI-compatible request and response types used by Joshua.

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

// ─── Chat messages ───────────────────────────────────────────────────────────

/// A single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// The role of the author (`"system"`, `"user"`, `"assistant"`, `"tool"`).
    pub role: String,
    /// The text content of the message.
    pub content: String,
    /// Optional image paths for vision models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    /// Optional author name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
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
            stop_sequences: vec![],
        }
    }
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
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model identifier (e.g. `"joshua"` or path-derived name).
    pub model: String,
    /// Conversation history including the new user turn.
    pub messages: Vec<ChatMessage>,
    /// Upper bound on generated tokens.
    #[serde(default)]
    pub max_tokens: Option<u32>,
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
    /// Stop sequences — either a single string or an array.
    #[serde(default)]
    pub stop: Option<serde_json::Value>,
    /// Whether to stream token-by-token via SSE.
    #[serde(default)]
    pub stream: Option<bool>,
    /// Tool/function definitions for tool-calling.
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
}

impl ChatCompletionRequest {
    /// Derive [`GenerationOptions`] from the request, applying defaults for missing fields.
    pub fn to_generation_options(&self) -> GenerationOptions {
        let defaults = GenerationOptions::default();
        let stop_sequences = match &self.stop {
            Some(serde_json::Value::String(s)) => vec![s.clone()],
            Some(serde_json::Value::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect(),
            _ => vec![],
        };
        GenerationOptions {
            max_tokens: self.max_tokens.unwrap_or(defaults.max_tokens),
            temperature: self.temperature.unwrap_or(defaults.temperature),
            top_p: self.top_p.unwrap_or(defaults.top_p),
            top_k: self.top_k.unwrap_or(defaults.top_k),
            min_p: self.min_p.unwrap_or(defaults.min_p),
            repetition_penalty: self.repetition_penalty.unwrap_or(defaults.repetition_penalty),
            stop_sequences,
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
