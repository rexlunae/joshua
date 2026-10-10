//! Transport-free client logic for the Joshua HTTP API.
//!
//! Everything here is plain Rust with no I/O so it can be unit-tested on the
//! host: URL normalisation, request bodies, response parsing and the
//! incremental Server-Sent-Events decoder used for streamed chat
//! completions.  [`crate::client`] wires these pieces to an HTTP client.
//!
//! The shapes mirror what `src/server.rs` / `src/types.rs` in the main
//! `joshua` crate actually serve:
//!
//! | Method | Path                       | Auth                    |
//! |--------|----------------------------|-------------------------|
//! | GET    | `/health`                  | never                   |
//! | GET    | `/v1/models`               | bearer key when set     |
//! | POST   | `/v1/chat/completions`     | bearer key when set     |
//! | POST   | `/v1/completions`          | bearer key when set     |
//! | POST   | `/v1/embeddings`           | bearer key when set     |
//! | POST   | `/v1/audio/transcriptions` | bearer key when set     |

use serde::{Deserialize, Serialize};

/// Address `joshua serve` listens on by default.
pub const DEFAULT_BASE_URL: &str = "http://127.0.0.1:8080";

/// Errors produced while building requests or interpreting responses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApiError {
    /// The base URL could not be used.
    InvalidBaseUrl(String),
    /// The server answered 401: no key, or the wrong key.
    Unauthorized(String),
    /// Any other non-success HTTP status, with the server's message.
    Status { code: u16, message: String },
    /// The body was not the JSON shape we expected.
    Decode(String),
    /// The request never completed (connection refused, CORS, ...).
    Transport(String),
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiError::InvalidBaseUrl(m) => write!(f, "invalid base URL: {m}"),
            ApiError::Unauthorized(m) => write!(f, "unauthorized: {m}"),
            ApiError::Status { code, message } => write!(f, "HTTP {code}: {message}"),
            ApiError::Decode(m) => write!(f, "could not decode response: {m}"),
            ApiError::Transport(m) => write!(f, "request failed: {m}"),
        }
    }
}

impl std::error::Error for ApiError {}

/// Endpoints of a Joshua server the UI talks to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endpoint {
    Health,
    Models,
    ChatCompletions,
}

impl Endpoint {
    /// Path relative to the server root.
    pub fn path(self) -> &'static str {
        match self {
            Endpoint::Health => "/health",
            Endpoint::Models => "/v1/models",
            Endpoint::ChatCompletions => "/v1/chat/completions",
        }
    }

    /// Whether the server's bearer-key middleware guards this endpoint.
    pub fn requires_auth(self) -> bool {
        !matches!(self, Endpoint::Health)
    }
}

/// Every route the server mounts, for display in the monitor view.
pub const SERVER_ROUTES: &[(&str, &str, &str)] = &[
    ("GET", "/health", "Liveness probe (never needs a key)"),
    (
        "GET",
        "/v1/models",
        "Loaded chat model, plus Whisper when configured",
    ),
    (
        "POST",
        "/v1/chat/completions",
        "Chat completion, optionally SSE-streamed",
    ),
    ("POST", "/v1/completions", "Legacy text completion"),
    ("POST", "/v1/embeddings", "Dense text embeddings"),
    (
        "POST",
        "/v1/audio/transcriptions",
        "Whisper speech-to-text (when loaded)",
    ),
];

/// Normalise what a user typed into a base URL: trims whitespace, defaults
/// the scheme to `http://`, and drops trailing slashes and a trailing `/v1`
/// (people often paste an OpenAI-style base URL).
pub fn normalize_base_url(input: &str) -> Result<String, ApiError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(ApiError::InvalidBaseUrl("empty".into()));
    }
    if trimmed.chars().any(char::is_whitespace) {
        return Err(ApiError::InvalidBaseUrl(format!(
            "'{trimmed}' contains whitespace"
        )));
    }
    let with_scheme = if let Some((scheme, _)) = trimmed.split_once("://") {
        if scheme != "http" && scheme != "https" {
            return Err(ApiError::InvalidBaseUrl(format!(
                "unsupported scheme '{scheme}' (use http or https)"
            )));
        }
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    };
    let mut url = with_scheme.trim_end_matches('/').to_string();
    if let Some(stripped) = url.strip_suffix("/v1") {
        url = stripped.trim_end_matches('/').to_string();
    }
    let host = url.split_once("://").map(|(_, rest)| rest).unwrap_or("");
    if host.is_empty() {
        return Err(ApiError::InvalidBaseUrl("missing host".into()));
    }
    Ok(url)
}

/// Absolute URL of `endpoint` on an already-normalised base URL.
pub fn endpoint_url(base: &str, endpoint: Endpoint) -> String {
    format!("{}{}", base.trim_end_matches('/'), endpoint.path())
}

/// `Authorization` header value for an API key, if one is set.  Blank keys
/// count as unset so an empty form field never sends `Bearer `.
pub fn bearer_header(api_key: Option<&str>) -> Option<String> {
    api_key
        .map(str::trim)
        .filter(|k| !k.is_empty())
        .map(|k| format!("Bearer {k}"))
}

// ─── Responses ───────────────────────────────────────────────────────────────

/// `GET /health` body.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Health {
    pub status: String,
}

impl Health {
    pub fn is_ok(&self) -> bool {
        self.status == "ok"
    }
}

/// One entry of `GET /v1/models`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Model {
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created: u64,
    #[serde(default)]
    pub owned_by: String,
}

/// `GET /v1/models` body.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ModelList {
    pub data: Vec<Model>,
}

/// Token accounting returned with completions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// The subset of `POST /v1/chat/completions` (non-streaming) the UI uses.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ChatChoice {
    pub message: ResponseMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ResponseMessage {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<serde_json::Value>,
}

impl ChatResponse {
    /// Text of the first choice (empty when the model only called tools).
    pub fn text(&self) -> &str {
        self.choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .unwrap_or("")
    }

    pub fn finish_reason(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|c| c.finish_reason.as_deref())
    }
}

pub fn parse_health(body: &str) -> Result<Health, ApiError> {
    serde_json::from_str(body).map_err(|e| ApiError::Decode(e.to_string()))
}

pub fn parse_models(body: &str) -> Result<ModelList, ApiError> {
    serde_json::from_str(body).map_err(|e| ApiError::Decode(e.to_string()))
}

pub fn parse_chat_response(body: &str) -> Result<ChatResponse, ApiError> {
    serde_json::from_str(body).map_err(|e| ApiError::Decode(e.to_string()))
}

/// Turn a non-2xx response into an [`ApiError`], pulling `error.message` out
/// of the OpenAI error envelope the server uses when it is present.
pub fn error_from_status(code: u16, body: &str) -> ApiError {
    #[derive(Deserialize)]
    struct Envelope {
        error: Detail,
    }
    #[derive(Deserialize)]
    struct Detail {
        message: String,
    }
    let message = serde_json::from_str::<Envelope>(body)
        .map(|e| e.error.message)
        .unwrap_or_else(|_| {
            let t = body.trim();
            if t.is_empty() {
                "(empty body)".to_string()
            } else {
                t.chars().take(300).collect()
            }
        });
    if code == 401 {
        ApiError::Unauthorized(message)
    } else {
        ApiError::Status { code, message }
    }
}

// ─── Requests ────────────────────────────────────────────────────────────────

/// Chat roles the server's template renderer accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

/// Sampling knobs the server honours (see `ChatCompletionRequest` in the
/// main crate).  `None` leaves the server default in place.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SamplingParams {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub stop: Vec<String>,
}

/// `POST /v1/chat/completions` body.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    pub stream: bool,
}

impl ChatRequest {
    /// Build a request.  An optional non-blank system prompt is placed first.
    pub fn new(
        model: impl Into<String>,
        system_prompt: Option<&str>,
        history: &[ChatMessage],
        params: &SamplingParams,
        stream: bool,
    ) -> Self {
        let mut messages = Vec::with_capacity(history.len() + 1);
        if let Some(sys) = system_prompt.map(str::trim).filter(|s| !s.is_empty()) {
            messages.push(ChatMessage::new(Role::System, sys));
        }
        messages.extend(history.iter().cloned());
        Self {
            model: model.into(),
            messages,
            max_tokens: params.max_tokens,
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            min_p: params.min_p,
            repetition_penalty: params.repetition_penalty,
            stop: params.stop.clone(),
            stream,
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("ChatRequest always serialises")
    }
}

/// Parse an optional numeric form field: blank means "server default".
pub fn parse_optional<T: std::str::FromStr>(field: &str, name: &str) -> Result<Option<T>, String> {
    let t = field.trim();
    if t.is_empty() {
        return Ok(None);
    }
    t.parse::<T>()
        .map(Some)
        .map_err(|_| format!("{name}: '{t}' is not a valid number"))
}

/// Split a comma-separated stop-sequence field, dropping blanks.
pub fn parse_stop_list(field: &str) -> Vec<String> {
    field
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

// ─── Streaming ───────────────────────────────────────────────────────────────

/// Incremental Server-Sent-Events decoder.
///
/// Bytes are buffered until a full line arrives, so a multi-byte UTF-8
/// character split across network chunks is never decoded in halves.
/// Each completed event yields its `data:` payload (multiple `data:` lines
/// are joined with `\n`, per the SSE spec).
#[derive(Debug, Default)]
pub struct SseDecoder {
    buf: Vec<u8>,
    data: Vec<String>,
    has_data: bool,
}

impl SseDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed raw bytes; returns the data payloads of every event completed.
    pub fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(bytes);
        let mut out = Vec::new();
        while let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
            let mut line: Vec<u8> = self.buf.drain(..=pos).collect();
            line.pop(); // '\n'
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            let line = String::from_utf8_lossy(&line);
            self.handle_line(&line, &mut out);
        }
        out
    }

    /// Flush an event left unterminated when the stream closes.
    pub fn finish(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        if !self.buf.is_empty() {
            let rest = std::mem::take(&mut self.buf);
            let line = String::from_utf8_lossy(&rest);
            let line = line.trim_end_matches('\r').to_string();
            self.handle_line(&line, &mut out);
        }
        self.handle_line("", &mut out);
        out
    }

    fn handle_line(&mut self, line: &str, out: &mut Vec<String>) {
        if line.is_empty() {
            if self.has_data {
                out.push(self.data.join("\n"));
                self.data.clear();
                self.has_data = false;
            }
            return;
        }
        if line.starts_with(':') {
            return; // comment / keep-alive
        }
        let (field, value) = match line.split_once(':') {
            Some((f, v)) => (f, v.strip_prefix(' ').unwrap_or(v)),
            None => (line, ""),
        };
        if field == "data" {
            self.data.push(value.to_string());
            self.has_data = true;
        }
    }
}

/// What one streamed chat event contributes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    /// A chunk; any of its fields may be absent.
    Chunk {
        content: Option<String>,
        finish_reason: Option<String>,
        usage: Option<Usage>,
        has_tool_calls: bool,
    },
    /// The `[DONE]` sentinel.
    Done,
}

/// Interpret one SSE `data:` payload of a chat-completion stream.
pub fn parse_stream_data(data: &str) -> Result<StreamEvent, ApiError> {
    let data = data.trim();
    if data == "[DONE]" {
        return Ok(StreamEvent::Done);
    }
    #[derive(Deserialize)]
    struct Chunk {
        #[serde(default)]
        choices: Vec<Choice>,
        #[serde(default)]
        usage: Option<Usage>,
    }
    #[derive(Deserialize)]
    struct Choice {
        #[serde(default)]
        delta: Delta,
        #[serde(default)]
        finish_reason: Option<String>,
    }
    #[derive(Deserialize, Default)]
    struct Delta {
        #[serde(default)]
        content: Option<String>,
        #[serde(default)]
        tool_calls: Option<serde_json::Value>,
    }
    let chunk: Chunk = serde_json::from_str(data).map_err(|e| ApiError::Decode(e.to_string()))?;
    let first = chunk.choices.into_iter().next();
    let (content, finish_reason, has_tool_calls) = match first {
        Some(c) => (
            c.delta.content,
            c.finish_reason,
            c.delta.tool_calls.is_some(),
        ),
        None => (None, None, false),
    };
    Ok(StreamEvent::Chunk {
        content,
        finish_reason,
        usage: chunk.usage,
        has_tool_calls,
    })
}

/// Generation throughput measured on the client.
pub fn tokens_per_second(completion_tokens: u32, elapsed_ms: f64) -> Option<f64> {
    (elapsed_ms > 0.0 && completion_tokens > 0)
        .then(|| f64::from(completion_tokens) * 1000.0 / elapsed_ms)
}

/// `HH:MM:SS UTC` for a Unix timestamp, for "last checked" labels.
pub fn format_utc_hms(unix_secs: u64) -> String {
    let s = unix_secs % 86_400;
    format!("{:02}:{:02}:{:02} UTC", s / 3600, (s / 60) % 60, s % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_url_normalisation() {
        assert_eq!(
            normalize_base_url("localhost:8080").unwrap(),
            "http://localhost:8080"
        );
        assert_eq!(
            normalize_base_url("  https://node1.lan:9000/ ").unwrap(),
            "https://node1.lan:9000"
        );
        assert_eq!(normalize_base_url("http://h:1/v1/").unwrap(), "http://h:1");
        assert_eq!(
            normalize_base_url("http://h/prefix").unwrap(),
            "http://h/prefix"
        );
        assert!(matches!(
            normalize_base_url(""),
            Err(ApiError::InvalidBaseUrl(_))
        ));
        assert!(matches!(
            normalize_base_url("ftp://h"),
            Err(ApiError::InvalidBaseUrl(_))
        ));
        assert!(matches!(
            normalize_base_url("http://"),
            Err(ApiError::InvalidBaseUrl(_))
        ));
        assert!(matches!(
            normalize_base_url("a b"),
            Err(ApiError::InvalidBaseUrl(_))
        ));
    }

    #[test]
    fn endpoint_urls_and_auth() {
        let base = normalize_base_url(DEFAULT_BASE_URL).unwrap();
        assert_eq!(
            endpoint_url(&base, Endpoint::Health),
            "http://127.0.0.1:8080/health"
        );
        assert_eq!(
            endpoint_url(&base, Endpoint::Models),
            "http://127.0.0.1:8080/v1/models"
        );
        assert_eq!(
            endpoint_url(&base, Endpoint::ChatCompletions),
            "http://127.0.0.1:8080/v1/chat/completions"
        );
        assert!(!Endpoint::Health.requires_auth());
        assert!(Endpoint::Models.requires_auth());
        assert_eq!(bearer_header(Some(" k ")).as_deref(), Some("Bearer k"));
        assert_eq!(bearer_header(Some("  ")), None);
        assert_eq!(bearer_header(None), None);
    }

    #[test]
    fn routes_table_matches_endpoints() {
        for ep in [
            Endpoint::Health,
            Endpoint::Models,
            Endpoint::ChatCompletions,
        ] {
            assert!(SERVER_ROUTES.iter().any(|(_, p, _)| *p == ep.path()));
        }
    }

    #[test]
    fn parses_health_and_models() {
        assert!(parse_health(r#"{"status":"ok"}"#).unwrap().is_ok());
        assert!(!parse_health(r#"{"status":"degraded"}"#).unwrap().is_ok());
        assert!(parse_health("not json").is_err());

        let body = r#"{"object":"list","data":[
            {"id":"qwen3-0.6b","object":"model","created":1700000000,"owned_by":"joshua"},
            {"id":"whisper-tiny","object":"model","created":1700000000,"owned_by":"joshua"}]}"#;
        let models = parse_models(body).unwrap();
        assert_eq!(models.data.len(), 2);
        assert_eq!(models.data[0].id, "qwen3-0.6b");
        assert_eq!(models.data[1].owned_by, "joshua");
        assert_eq!(models.data[0].created, 1_700_000_000);
    }

    #[test]
    fn parses_chat_response() {
        let body = r#"{"id":"chatcmpl-1","object":"chat.completion","created":1,
            "model":"m","choices":[{"index":0,"message":{"role":"assistant",
            "content":"Hello!"},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}"#;
        let r = parse_chat_response(body).unwrap();
        assert_eq!(r.text(), "Hello!");
        assert_eq!(r.finish_reason(), Some("stop"));
        assert_eq!(r.usage.total_tokens, 7);

        let tools = r#"{"id":"c","model":"m","choices":[{"message":{"content":null,
            "tool_calls":[{"id":"x"}]},"finish_reason":"tool_calls"}],
            "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
        let r = parse_chat_response(tools).unwrap();
        assert_eq!(r.text(), "");
        assert!(r.choices[0].message.tool_calls.is_some());
    }

    #[test]
    fn error_envelopes() {
        let body = r#"{"error":{"message":"invalid or missing API key","type":"invalid_request_error","param":null,"code":null}}"#;
        assert_eq!(
            error_from_status(401, body),
            ApiError::Unauthorized("invalid or missing API key".into())
        );
        assert_eq!(
            error_from_status(500, "boom"),
            ApiError::Status {
                code: 500,
                message: "boom".into()
            }
        );
        assert_eq!(
            error_from_status(502, ""),
            ApiError::Status {
                code: 502,
                message: "(empty body)".into()
            }
        );
    }

    #[test]
    fn chat_request_body() {
        let history = vec![ChatMessage::new(Role::User, "hi")];
        let params = SamplingParams {
            max_tokens: Some(64),
            temperature: Some(0.5),
            stop: vec!["</s>".into()],
            ..Default::default()
        };
        let req = ChatRequest::new("m", Some(" be brief "), &history, &params, true);
        let v: serde_json::Value = serde_json::from_str(&req.to_json()).unwrap();
        assert_eq!(v["model"], "m");
        assert_eq!(v["stream"], true);
        assert_eq!(v["max_tokens"], 64);
        assert_eq!(v["temperature"], 0.5);
        assert_eq!(v["stop"][0], "</s>");
        assert_eq!(v["messages"][0]["role"], "system");
        assert_eq!(v["messages"][0]["content"], "be brief");
        assert_eq!(v["messages"][1]["role"], "user");
        // Unset knobs are omitted so the server applies its own defaults.
        assert!(v.get("top_p").is_none());
        assert!(v.get("top_k").is_none());

        let req = ChatRequest::new(
            "m",
            Some("   "),
            &history,
            &SamplingParams::default(),
            false,
        );
        let v: serde_json::Value = serde_json::from_str(&req.to_json()).unwrap();
        assert_eq!(v["messages"].as_array().unwrap().len(), 1);
        assert!(v.get("stop").is_none());
        assert_eq!(v["stream"], false);
    }

    #[test]
    fn form_field_parsing() {
        assert_eq!(parse_optional::<u32>("", "max").unwrap(), None);
        assert_eq!(parse_optional::<u32>(" 12 ", "max").unwrap(), Some(12));
        assert_eq!(parse_optional::<f32>("0.7", "t").unwrap(), Some(0.7));
        assert!(parse_optional::<u32>("-1", "max").is_err());
        assert_eq!(parse_stop_list(" a, ,b ,"), vec!["a", "b"]);
    }

    #[test]
    fn sse_decoder_handles_split_chunks() {
        let mut d = SseDecoder::new();
        assert!(d.push(b"data: {\"a\"").is_empty());
        assert!(d.push(b":1}\r\n").is_empty());
        assert_eq!(
            d.push(b"\r\n: keep-alive\n\ndata: [DONE]\n\n"),
            vec!["{\"a\":1}".to_string(), "[DONE]".to_string()]
        );
        // Multi-line data and an unterminated tail.
        let mut d = SseDecoder::new();
        assert_eq!(
            d.push(b"event: x\ndata: one\ndata:two\n\n"),
            vec!["one\ntwo"]
        );
        assert!(d.push(b"data: tail").is_empty());
        assert_eq!(d.finish(), vec!["tail"]);
    }

    #[test]
    fn sse_decoder_keeps_utf8_split_across_chunks() {
        let payload = "data: {\"c\":\"é\"}\n\n".as_bytes();
        let split = payload.iter().position(|&b| b == 0xC3).unwrap() + 1;
        let mut d = SseDecoder::new();
        assert!(d.push(&payload[..split]).is_empty());
        assert_eq!(d.push(&payload[split..]), vec!["{\"c\":\"é\"}"]);
    }

    #[test]
    fn stream_events_match_server_wire_format() {
        // Shapes copied from chat_completions' streaming path in src/server.rs.
        let first = r#"{"id":"chatcmpl-x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"H"},"finish_reason":null}]}"#;
        assert_eq!(
            parse_stream_data(first).unwrap(),
            StreamEvent::Chunk {
                content: Some("H".into()),
                finish_reason: None,
                usage: None,
                has_tool_calls: false,
            }
        );
        let stop = r#"{"id":"chatcmpl-x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}"#;
        assert_eq!(
            parse_stream_data(stop).unwrap(),
            StreamEvent::Chunk {
                content: None,
                finish_reason: Some("stop".into()),
                usage: Some(Usage {
                    prompt_tokens: 3,
                    completion_tokens: 4,
                    total_tokens: 7
                }),
                has_tool_calls: false,
            }
        );
        let tool = r#"{"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0}]},"finish_reason":null}]}"#;
        assert!(matches!(
            parse_stream_data(tool).unwrap(),
            StreamEvent::Chunk {
                has_tool_calls: true,
                ..
            }
        ));
        assert_eq!(parse_stream_data(" [DONE] ").unwrap(), StreamEvent::Done);
        assert!(parse_stream_data("{").is_err());
    }

    #[test]
    fn utc_clock() {
        assert_eq!(format_utc_hms(0), "00:00:00 UTC");
        assert_eq!(format_utc_hms(86_400 + 3_661), "01:01:01 UTC");
    }

    #[test]
    fn throughput() {
        assert_eq!(tokens_per_second(50, 1000.0), Some(50.0));
        assert_eq!(tokens_per_second(0, 1000.0), None);
        assert_eq!(tokens_per_second(5, 0.0), None);
    }
}
