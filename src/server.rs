//! OpenAI-compatible HTTP API server for Joshua.
//!
//! Implements the following endpoints:
//! - `GET  /health`                   — liveness check
//! - `GET  /v1/models`                — list loaded model
//! - `POST /v1/chat/completions`      — chat completion (stream or non-stream)
//! - `POST /v1/completions`           — legacy text completion
//! - `POST /v1/embeddings`            — dense text embeddings

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::State,
    http::StatusCode,
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
use crate::tools::parse_tool_calls;
use crate::types::{
    AssistantMessage, ChatChoice, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, ChatMessage, DeltaContent, EmbeddingData, EmbeddingRequest,
    EmbeddingResponse, ErrorResponse, FunctionCallResult, GenerationOptions, ModelInfo,
    ModelListResponse, ToolCall, UsageInfo,
};

/// Shared application state (the loaded engine).
pub type AppState = Arc<Engine>;

// ─── Router ───────────────────────────────────────────────────────────────────

/// Build the Axum router with all API routes mounted.
pub fn create_router(engine: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .layer(CorsLayer::permissive())
        .with_state(engine)
}

/// Start the server on `addr` (e.g. `"0.0.0.0:8080"`).
pub async fn serve(engine: AppState, addr: &str) -> std::io::Result<()> {
    let app = create_router(engine);
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("Joshua server listening on {}", addr);
    axum::serve(listener, app).await
}

// ─── Handlers ────────────────────────────────────────────────────────────────

/// `GET /health` — returns `{"status":"ok"}`.
async fn health() -> Json<serde_json::Value> {
    Json(json!({"status": "ok"}))
}

/// `GET /v1/models` — returns the single loaded model.
async fn list_models(State(engine): State<AppState>) -> Json<ModelListResponse> {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: engine.model_name().to_string(),
            object: "model".to_string(),
            created,
            owned_by: "joshua".to_string(),
        }],
    })
}

/// `POST /v1/chat/completions` — OpenAI chat completions.
async fn chat_completions(
    State(engine): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
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
    State(engine): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, ApiError> {
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
    State(engine): State<AppState>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, ApiError> {
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

    fn internal(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: ErrorResponse::server_error(msg),
        }
    }
}

impl From<JoshuaError> for ApiError {
    fn from(err: JoshuaError) -> Self {
        match &err {
            JoshuaError::InvalidRequest(_) | JoshuaError::PromptTooLong(_, _) => Self {
                status: StatusCode::BAD_REQUEST,
                body: ErrorResponse::invalid_request(err.to_string()),
            },
            _ => Self {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                body: ErrorResponse::server_error(err.to_string()),
            },
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body)).into_response()
    }
}
