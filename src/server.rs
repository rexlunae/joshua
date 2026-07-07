//! OpenAI-compatible HTTP API server for Joshua.
//!
//! Implements the following endpoints:
//! - `GET  /health`                   — liveness check
//! - `GET  /v1/models`                — list loaded model
//! - `POST /v1/chat/completions`      — chat completion (stream or non-stream)
//! - `POST /v1/completions`           — legacy text completion
//! - `POST /v1/embeddings`            — dense text embeddings
//! - `POST /v1/audio/transcriptions`  — Whisper speech-to-text (when a
//!   whisper model is configured)

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::{self, Next},
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use futures_util::{stream, StreamExt};
use serde_json::json;
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;
use uuid::Uuid;

use crate::engine::Engine;
use crate::error::JoshuaError;
use crate::whisper::WhisperEngine;
use crate::tools::parse_tool_calls;
use crate::types::{
    AssistantMessage, ChatChoice, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, ChatMessage, DeltaContent, EmbeddingData, EmbeddingRequest,
    EmbeddingResponse, ErrorResponse, FunctionCallResult, GenerationOptions, ModelInfo,
    ModelListResponse, ToolCall, UsageInfo,
};

/// Shared application state.
pub struct ServerState {
    /// The chat/embedding engine.
    pub engine: Arc<Engine>,
    /// Optional Whisper model for `/v1/audio/transcriptions`.
    pub whisper: Option<Arc<WhisperEngine>>,
    /// When set, every `/v1` request must carry this key as
    /// `Authorization: Bearer <key>`.  `/health` stays open for probes.
    pub api_key: Option<String>,
}

/// Shared application state handle.
pub type AppState = Arc<ServerState>;

// ─── Router ───────────────────────────────────────────────────────────────────

/// Build the Axum router with all API routes mounted.
pub fn create_router(state: AppState) -> Router {
    let api = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/audio/transcriptions", post(transcriptions))
        .layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            require_api_key,
        ));
    Router::new()
        .route("/health", get(health))
        .merge(api)
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Reject `/v1` requests that lack the configured bearer API key.
///
/// A no-op when no key is configured.  Comparison is constant-time so the
/// key can't be recovered byte-by-byte through response timing.
async fn require_api_key(State(state): State<AppState>, req: Request, next: Next) -> Response {
    if let Some(expected) = &state.api_key {
        let provided = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));
        if !provided.is_some_and(|key| api_keys_match(key.as_bytes(), expected.as_bytes())) {
            let body = ErrorResponse::new(
                "invalid or missing API key — pass the key as 'Authorization: Bearer <key>'",
                "invalid_request_error",
            );
            return (StatusCode::UNAUTHORIZED, Json(body)).into_response();
        }
    }
    next.run(req).await
}

/// Byte-wise equality whose runtime depends only on the length of the
/// caller-supplied `provided` key, revealing nothing about `expected` —
/// not even its length.  On a mismatch the whole loop still runs, cycling
/// through `expected` so no early exit correlates with the secret.
fn api_keys_match(provided: &[u8], expected: &[u8]) -> bool {
    if expected.is_empty() {
        return provided.is_empty();
    }
    let mut diff = provided.len() ^ expected.len();
    for (i, &byte) in provided.iter().enumerate() {
        diff |= usize::from(byte ^ expected[i % expected.len()]);
    }
    diff == 0
}

/// Start the server on `addr` (e.g. `"0.0.0.0:8080"`) with just a chat
/// engine.  Use [`serve_with_state`] to also mount a Whisper model.
pub async fn serve(engine: Arc<Engine>, addr: &str) -> std::io::Result<()> {
    serve_with_state(
        Arc::new(ServerState {
            engine,
            whisper: None,
            api_key: None,
        }),
        addr,
    )
    .await
}

/// Start the server with a fully configured [`ServerState`].
pub async fn serve_with_state(state: AppState, addr: &str) -> std::io::Result<()> {
    let app = create_router(state);
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("Joshua server listening on http://{}", addr);
    axum::serve(listener, app).await
}

/// Start the server over HTTPS (the `tls` cargo feature).
///
/// `cert` and `key` are paths to a PEM-encoded certificate chain and
/// PKCS#8/RSA/SEC1 private key.  TLS is terminated in-process by rustls —
/// no reverse proxy needed.
#[cfg(feature = "tls")]
pub async fn serve_with_state_tls(
    state: AppState,
    addr: &str,
    cert: &std::path::Path,
    key: &std::path::Path,
) -> std::io::Result<()> {
    use std::io::{Error, ErrorKind};

    // axum-server is built without a default crypto provider; install ring
    // process-wide.  Err means a provider is already installed — fine.
    let _ = rustls::crypto::ring::default_provider().install_default();

    let addr: std::net::SocketAddr = addr
        .parse()
        .map_err(|e| Error::new(ErrorKind::InvalidInput, format!("invalid address: {e}")))?;
    let config = axum_server::tls_rustls::RustlsConfig::from_pem_file(cert, key).await?;
    let app = create_router(state);
    tracing::info!("Joshua server listening on https://{}", addr);
    axum_server::bind_rustls(addr, config)
        .serve(app.into_make_service())
        .await
}

// ─── Handlers ────────────────────────────────────────────────────────────────

/// `GET /health` — returns `{"status":"ok"}`.
async fn health() -> Json<serde_json::Value> {
    Json(json!({"status": "ok"}))
}

/// `GET /v1/models` — returns the single loaded model.
async fn list_models(State(state): State<AppState>) -> Json<ModelListResponse> {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let mut data = vec![ModelInfo {
        id: state.engine.model_name().to_string(),
        object: "model".to_string(),
        created,
        owned_by: "joshua".to_string(),
    }];
    if let Some(whisper) = &state.whisper {
        data.push(ModelInfo {
            id: whisper.model_name().to_string(),
            object: "model".to_string(),
            created,
            owned_by: "joshua".to_string(),
        });
    }
    Json(ModelListResponse {
        object: "list".to_string(),
        data,
    })
}

/// `POST /v1/chat/completions` — OpenAI chat completions.
async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let engine = Arc::clone(&state.engine);
    let stream = req.stream.unwrap_or(false);
    let options = req.to_generation_options();
    let messages = req.messages.clone();
    let tools = req.tools.clone();
    let model = engine.model_name().to_string();

    if stream {
        // ── Streaming path ────────────────────────────────────────────────────
        let id = format!("chatcmpl-{}", Uuid::new_v4().simple());

        // Run inference in a blocking thread to avoid stalling the async runtime.
        let (text, usage, _, _) = tokio::task::spawn_blocking({
            let engine = Arc::clone(&engine);
            let tools = tools.clone();
            move || engine.complete_chat(&messages, tools.as_deref(), &options)
        })
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?
        .map_err(ApiError::from)?;

        // When tools were requested and the model emitted calls, send them as
        // a single delta chunk (OpenAI wire format) instead of char-streaming
        // the raw markup.
        if tools.is_some() {
            let (prose, calls) = parse_tool_calls(&text);
            if !calls.is_empty() {
                let delta = DeltaContent {
                    role: Some("assistant".to_string()),
                    content: if prose.is_empty() { None } else { Some(prose) },
                    tool_calls: Some(tool_call_deltas(&calls)),
                };
                let first = ChatCompletionChunk::new(id.clone(), model.clone(), delta, None);
                let stop = ChatCompletionChunk::new(
                    id.clone(),
                    model.clone(),
                    DeltaContent::default(),
                    Some("tool_calls".to_string()),
                );
                let events = stream::iter([first, stop].into_iter().map(|chunk| {
                    let data = serde_json::to_string(&chunk).unwrap_or_default();
                    Ok::<Event, Infallible>(Event::default().data(data))
                }))
                .chain(stream::once(async {
                    Ok::<Event, Infallible>(Event::default().data("[DONE]"))
                }));
                return Ok(Sse::new(events).into_response());
            }
        }

        // Stream the response character-by-character (word-level chunks are
        // possible too, but char chunks give the smoothest streaming experience).
        let chunks: Vec<String> = text
            .char_indices()
            .map(|(_, c)| c.to_string())
            .collect();

        let id2 = id.clone();
        let model2 = model.clone();
        let n_chunks = chunks.len();

        // Content chunks — include the role header on the very first chunk.
        let content_events =
            stream::iter(chunks.into_iter().enumerate().map(move |(i, chunk)| {
                let delta = if i == 0 {
                    DeltaContent {
                        role: Some("assistant".to_string()),
                        content: Some(chunk),
                        tool_calls: None,
                    }
                } else {
                    DeltaContent {
                        role: None,
                        content: Some(chunk),
                        tool_calls: None,
                    }
                };
                let payload =
                    ChatCompletionChunk::new(id2.clone(), model2.clone(), delta, None);
                let data = serde_json::to_string(&payload).unwrap_or_default();
                Ok::<Event, Infallible>(Event::default().data(data))
            }));

        // Final "stop" chunk — includes usage statistics as per the OpenAI spec
        // (`stream_options.include_usage`). We always include them so that clients
        // that inspect this chunk get accurate token counts.
        let stop_payload = {
            let chunk =
                ChatCompletionChunk::new(id.clone(), model.clone(), DeltaContent::default(), Some("stop".to_string()));
            // Attach usage as an extra field via serde_json (ChatCompletionChunk
            // doesn't have a `usage` field to keep streaming chunks lean, so we
            // serialise it manually here and embed it).
            let mut value = serde_json::to_value(&chunk).unwrap_or_default();
            value["usage"] = serde_json::json!({
                "prompt_tokens":     usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens":      usage.total_tokens,
            });
            serde_json::to_string(&value).unwrap_or_default()
        };

        let sse_stream = content_events
            .chain(stream::once(async move {
                Ok::<Event, Infallible>(Event::default().data(stop_payload))
            }))
            .chain(stream::once(async {
                Ok::<Event, Infallible>(Event::default().data("[DONE]"))
            }));

        let _ = n_chunks; // consumed above

        return Ok(Sse::new(sse_stream).into_response());
    }

    // ── Non-streaming path ────────────────────────────────────────────────────
    let (text, usage, _, _) = tokio::task::spawn_blocking({
        let engine = Arc::clone(&engine);
        let tools = tools.clone();
        move || engine.complete_chat(&messages, tools.as_deref(), &options)
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?
    .map_err(ApiError::from)?;

    // Extract tool calls from the output when the request offered tools.
    let (content, tool_calls, finish_reason) = if tools.is_some() {
        let (prose, calls) = parse_tool_calls(&text);
        if calls.is_empty() {
            (Some(text), None, "stop")
        } else {
            let calls: Vec<ToolCall> = calls
                .into_iter()
                .map(|c| ToolCall {
                    id: format!("call_{}", Uuid::new_v4().simple()),
                    call_type: "function".to_string(),
                    function: FunctionCallResult {
                        name: c.name,
                        arguments: c.arguments,
                    },
                })
                .collect();
            (
                if prose.is_empty() { None } else { Some(prose) },
                Some(calls),
                "tool_calls",
            )
        }
    } else {
        (Some(text), None, "stop")
    };

    let id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let response = ChatCompletionResponse::new(
        id,
        model,
        vec![ChatChoice {
            index: 0,
            message: AssistantMessage {
                role: "assistant".to_string(),
                content,
                tool_calls,
            },
            finish_reason: finish_reason.to_string(),
        }],
        usage,
    );
    Ok(Json(response).into_response())
}

/// Build the OpenAI streaming `delta.tool_calls` payload (index per entry).
fn tool_call_deltas(calls: &[crate::tools::ParsedToolCall]) -> serde_json::Value {
    serde_json::Value::Array(
        calls
            .iter()
            .enumerate()
            .map(|(i, c)| {
                json!({
                    "index": i,
                    "id": format!("call_{}", Uuid::new_v4().simple()),
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                })
            })
            .collect(),
    )
}

/// `POST /v1/completions` — legacy (non-chat) text completion.
async fn completions(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let engine = Arc::clone(&state.engine);
    let prompt = body
        .get("prompt")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::bad_request("Missing required field 'prompt'"))?
        .to_string();

    let options = GenerationOptions {
        max_tokens: body
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(256),
        temperature: body
            .get("temperature")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(0.7),
        ..GenerationOptions::default()
    };

    let messages = vec![ChatMessage::text("user".to_string(), prompt)];

    let (text, usage, _, _) = tokio::task::spawn_blocking({
        let engine = Arc::clone(&engine);
        move || engine.complete(&messages, &options)
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?
    .map_err(ApiError::from)?;

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Ok(Json(json!({
        "id": format!("cmpl-{}", Uuid::new_v4().simple()),
        "object": "text_completion",
        "created": created,
        "model": engine.model_name(),
        "choices": [{
            "text": text,
            "index": 0,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
        }
    })))
}

/// `POST /v1/embeddings` — dense text embeddings.
async fn embeddings(
    State(state): State<AppState>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, ApiError> {
    let engine = Arc::clone(&state.engine);
    let texts: Vec<String> = req.input.into_vec();
    let model = engine.model_name().to_string();

    let (vectors, prompt_tokens) = tokio::task::spawn_blocking({
        let engine = Arc::clone(&engine);
        move || engine.embed_with_usage(&texts)
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?
    .map_err(ApiError::from)?;

    let data = vectors
        .into_iter()
        .enumerate()
        .map(|(i, embedding)| EmbeddingData {
            object: "embedding".to_string(),
            embedding,
            index: i as u32,
        })
        .collect();

    Ok(Json(EmbeddingResponse {
        object: "list".to_string(),
        data,
        model,
        usage: UsageInfo {
            prompt_tokens,
            completion_tokens: 0,
            total_tokens: prompt_tokens,
        },
    }))
}

/// `POST /v1/audio/transcriptions` — OpenAI-compatible Whisper STT.
///
/// Multipart form fields: `file` (required, WAV), `language` (optional
/// two-letter code), `response_format` (`json` default, or `text`), and
/// `model` (accepted and ignored — the loaded whisper model is used).
async fn transcriptions(
    State(state): State<AppState>,
    mut multipart: axum::extract::Multipart,
) -> Result<Response, ApiError> {
    let Some(whisper) = state.whisper.clone() else {
        return Err(ApiError::bad_request(
            "no whisper model is loaded — start the server with --whisper-model",
        ));
    };

    let mut file: Option<Vec<u8>> = None;
    let mut language: Option<String> = None;
    let mut response_format = "json".to_string();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ApiError::bad_request(format!("invalid multipart body: {e}")))?
    {
        match field.name().unwrap_or_default() {
            "file" => {
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|e| ApiError::bad_request(format!("file upload failed: {e}")))?;
                file = Some(bytes.to_vec());
            }
            "language" => {
                language = Some(field.text().await.unwrap_or_default());
            }
            "response_format" => {
                response_format = field.text().await.unwrap_or_default();
            }
            // `model`, `prompt`, `temperature`, … accepted and ignored.
            _ => {
                let _ = field.bytes().await;
            }
        }
    }
    let file = file.ok_or_else(|| ApiError::bad_request("missing required field 'file'"))?;

    let transcription = tokio::task::spawn_blocking({
        let language = language.clone();
        move || whisper.transcribe_wav(&file, language.as_deref(), false)
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?
    .map_err(ApiError::from)?;

    if response_format == "text" {
        return Ok(transcription.text.into_response());
    }
    Ok(Json(json!({
        "text": transcription.text,
        "duration": transcription.duration,
        "language": transcription.language,
    }))
    .into_response())
}

// ─── API error type ───────────────────────────────────────────────────────────

/// Internal helper that maps [`JoshuaError`] to HTTP responses.
pub struct ApiError {
    status: StatusCode,
    body: ErrorResponse,
}

impl ApiError {
    fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            body: ErrorResponse::invalid_request(msg),
        }
    }

    /// Internal failure: the detail is logged server-side, and the client
    /// receives only a generic message (no internal strings / panic text).
    fn internal(msg: impl Into<String>) -> Self {
        tracing::error!("internal error serving request: {}", msg.into());
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: ErrorResponse::server_error("internal error"),
        }
    }
}

impl From<JoshuaError> for ApiError {
    fn from(err: JoshuaError) -> Self {
        match &err {
            // Client errors carry a caller-actionable message and are safe
            // to echo verbatim.
            JoshuaError::InvalidRequest(_) | JoshuaError::PromptTooLong(_, _) => Self {
                status: StatusCode::BAD_REQUEST,
                body: ErrorResponse::invalid_request(err.to_string()),
            },
            JoshuaError::Overloaded(_) => Self {
                status: StatusCode::SERVICE_UNAVAILABLE,
                body: ErrorResponse::new(err.to_string(), "overloaded"),
            },
            // Everything else may embed internal detail (tokenizer/candle
            // messages, io errors). Log it server-side; return a generic body.
            _ => {
                tracing::error!("internal error serving request: {err}");
                Self {
                    status: StatusCode::INTERNAL_SERVER_ERROR,
                    body: ErrorResponse::server_error("internal error"),
                }
            }
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::api_keys_match;

    #[test]
    fn api_keys_match_accepts_only_the_exact_key() {
        assert!(api_keys_match(b"sekret", b"sekret"));
        assert!(!api_keys_match(b"Sekret", b"sekret"));
        // Shorter, longer, and cyclic-repeat inputs all fail: a provided key
        // that is `expected` repeated would zero every byte XOR, so the
        // length term must reject it.
        assert!(!api_keys_match(b"sek", b"sekret"));
        assert!(!api_keys_match(b"sekretsekret", b"sekret"));
        assert!(!api_keys_match(b"", b"sekret"));
    }

    #[test]
    fn api_keys_match_handles_an_empty_expected_key() {
        assert!(api_keys_match(b"", b""));
        assert!(!api_keys_match(b"a", b""));
    }
}
