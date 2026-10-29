//! Ollama-compatible API (`/api/*`) on top of joshua's model registry.
//!
//! Ollama's native protocol differs from OpenAI's in three ways that matter:
//!
//! 1. **Streaming is newline-delimited JSON**, not SSE.  Each line is a whole
//!    object; the last carries `"done": true`.
//! 2. **Generation parameters live under `options`** (`num_ctx`, `num_predict`,
//!    `temperature`, `top_k`, `top_p`, `repeat_penalty`, `stop`).
//! 3. **Models are addressed by bare name** from a listing endpoint
//!    (`/api/tags`) that must work before anything is loaded.
//!
//! This is a translation layer: names resolve through the loaded
//! [`crate::server::ModelRegistry`] (which already dedups concurrent loads and
//! confines paths to the model directory), then drive the same engine the
//! OpenAI routes use.
//!
//! **Model management** (`/api/delete`, `/api/copy`, `/api/create`,
//! `/api/pull`) is provided against the model *directory* rather than Ollama's
//! content-addressed blob store, so scripts that manage models through the API
//! keep working:
//!
//! * `delete` / `copy` / `create` (the `from` form) act on `<name>.gguf` inside
//!   the model directory.  Names are confined to that directory — `..`,
//!   absolute paths and names that canonicalise outside it are rejected,
//!   because these endpoints delete and overwrite files.
//! * `create` with `files` (SHA256 blob digests) is **rejected**, not ignored:
//!   there is no blob store to name files in.
//! * `pull` does not speak the Ollama registry protocol, so a bare library name
//!   such as `llama3` fails with guidance instead of hanging.  Place the
//!   `.gguf` in the model directory and use `POST /v1/models/load`, or move the
//!   file in by hand.
//!
//! Ollama's `:tag` suffix names a manifest, which has no joshua equivalent:
//! every tag of a name resolves to the same `<name>.gguf`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::server::{ApiError, AppState, CancelOnDrop};
use crate::types::{ChatMessage, GenerationOptions, Tool};

fn now_rfc3339() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{secs}.000000000Z")
}

// ─── /api/version ───────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct VersionResponse {
    pub version: String,
}

pub async fn version() -> Json<VersionResponse> {
    Json(VersionResponse {
        version: "0.1.0".to_string(),
    })
}

// ─── shared model listing (also used by /v1/models) ─────────────────────────

/// One model joshua can serve, whether or not it is currently loaded.
pub struct AvailableModel {
    pub id: String,
    pub path: Option<PathBuf>,
    pub size: u64,
    pub loaded: bool,
}

impl AvailableModel {
    /// Everything the server can serve: loaded models first, then any `.gguf`
    /// under `model_dir` not already loaded.  Ids are the GGUF stem.
    pub fn discover(state: &AppState) -> Vec<AvailableModel> {
        let mut out: Vec<AvailableModel> = Vec::new();
        for loaded in state.models.list() {
            out.push(AvailableModel {
                id: loaded.id.clone(),
                path: None,
                size: 0,
                loaded: true,
            });
        }
        if let Some(manager) = &state.manager {
            for file in discover_gguf(&manager.model_dir) {
                let id = file
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("model")
                    .to_string();
                if out.iter().any(|m| m.id == id) {
                    continue;
                }
                let size = std::fs::metadata(&file).map(|m| m.len()).unwrap_or(0);
                out.push(AvailableModel {
                    id,
                    path: Some(file),
                    size,
                    loaded: false,
                });
            }
        }
        out
    }
}

fn discover_gguf(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect_gguf(dir, 0, &mut out);
    out.sort();
    out
}

fn collect_gguf(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > 2 {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_gguf(&path, depth + 1, out);
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            out.push(path);
        }
    }
}

/// llama.cpp's `general.file_type` enum → quantisation label.
fn quant_label(ft: u32) -> Option<String> {
    let s = match ft {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ2_S",
        28 => "IQ4_XS",
        29 => "I8",
        30 => "I16",
        31 => "I32",
        32 => "I64",
        33 => "F64",
        34 => "IQ1_M",
        35 => "BF16",
        39 => "TQ1_0",
        40 => "TQ2_0",
        _ => return None,
    };
    Some(s.to_string())
}

#[derive(Serialize, Clone, Default)]
pub struct TagDetails {
    parent_model: String,
    format: String,
    family: String,
    families: Vec<String>,
    parameter_size: String,
    quantization_level: String,
}

fn details_for(path: Option<&Path>) -> TagDetails {
    let mut d = TagDetails {
        format: "gguf".to_string(),
        ..Default::default()
    };
    let Some(path) = path else { return d };
    let Ok(file) = std::fs::File::open(path) else { return d };
    let Ok(content) = crate::gguf_ext::read_header(&mut std::io::BufReader::new(file)) else {
        return d;
    };
    let md = &content.metadata;
    if let Some(family) = md.get("general.architecture").and_then(|v| v.to_string().ok()) {
        d.families = vec![family.clone()];
        d.family = family.clone();
    }
    if let Some(ft) = md.get("general.file_type").and_then(|v| v.to_u32().ok()) {
        d.quantization_level = quant_label(ft).unwrap_or_default();
    }
    d
}

fn normalise_name(id: &str) -> String {
    if id.contains(':') {
        id.to_string()
    } else {
        format!("{id}:latest")
    }
}

/// Drop Ollama's `:tag` suffix, which names a *manifest* rather than a file.
/// joshua serves one `.gguf` per name, so `llama3:latest` and `llama3:13b` both
/// address `llama3.gguf`; a caller that needs two variants keeps two files
/// under distinct names.
fn strip_tag(name: &str) -> &str {
    name.split(':').next().unwrap_or(name).trim()
}

// ─── /api/tags ──────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct TagModel {
    name: String,
    model: String,
    modified_at: String,
    size: u64,
    digest: String,
    details: TagDetails,
}

#[derive(Serialize)]
pub struct TagsResponse {
    models: Vec<TagModel>,
}

pub async fn tags(State(state): State<AppState>) -> Json<TagsResponse> {
    let models = AvailableModel::discover(&state)
        .into_iter()
        .map(|m| TagModel {
            name: normalise_name(&m.id),
            model: normalise_name(&m.id),
            modified_at: now_rfc3339(),
            size: m.size,
            digest: String::new(),
            details: details_for(m.path.as_deref()),
        })
        .collect();
    Json(TagsResponse { models })
}

// ─── /api/ps ────────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct PsModel {
    name: String,
    model: String,
    size: u64,
    digest: String,
    details: TagDetails,
    expires_at: String,
    size_vram: u64,
}

#[derive(Serialize)]
pub struct PsResponse {
    models: Vec<PsModel>,
}

pub async fn ps(State(state): State<AppState>) -> Json<PsResponse> {
    let models = AvailableModel::discover(&state)
        .into_iter()
        .filter(|m| m.loaded)
        .map(|m| PsModel {
            name: normalise_name(&m.id),
            model: normalise_name(&m.id),
            size: m.size,
            digest: String::new(),
            details: details_for(m.path.as_deref()),
            // No idle eviction yet: a loaded model stays resident.
            expires_at: "2318-01-01T00:00:00Z".to_string(),
            size_vram: 0,
        })
        .collect();
    Json(PsResponse { models })
}

// ─── /api/show ──────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ShowRequest {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Serialize)]
pub struct ShowResponse {
    details: TagDetails,
    #[serde(skip_serializing_if = "Option::is_none")]
    modelfile: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    template: Option<String>,
    capabilities: Vec<String>,
}

pub async fn show(
    State(state): State<AppState>,
    Json(req): Json<ShowRequest>,
) -> Result<Json<ShowResponse>, ApiError> {
    let requested = req.model.unwrap_or(req.name);
    let (id, _engine) = state.models.resolve(&requested)?;
    Ok(Json(ShowResponse {
        details: details_for(None),
        modelfile: Some(format!("# joshua model: {id}")),
        parameters: None,
        template: None,
        capabilities: vec!["completion".to_string()],
    }))
}

// ─── options translation ────────────────────────────────────────────────────

#[derive(Debug, Default, Deserialize, Clone)]
pub struct OllamaOptions {
    #[serde(default)]
    pub num_ctx: Option<u32>,
    #[serde(default)]
    pub num_predict: Option<i32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
}

impl OllamaOptions {
    fn to_generation_options(&self) -> GenerationOptions {
        let mut o = GenerationOptions::default();
        if let Some(t) = self.temperature {
            o.temperature = t;
        }
        if let Some(k) = self.top_k {
            o.top_k = k;
        }
        if let Some(p) = self.top_p {
            o.top_p = p;
        }
        if let Some(m) = self.min_p {
            o.min_p = m;
        }
        if let Some(r) = self.repeat_penalty {
            o.repetition_penalty = r;
        }
        if let Some(n) = self.num_predict {
            // Ollama: -1 = until stop, -2 = fill context.  Only a positive
            // value maps cleanly onto joshua's max_tokens.
            if n > 0 {
                o.max_tokens = n as u32;
            }
        }
        if let Some(stop) = &self.stop {
            if !stop.is_empty() {
                o.stop_sequences = stop.clone();
            }
        }
        o
    }
}

// ─── /api/chat ──────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct OllamaMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<OllamaMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: OllamaOptions,
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
}

#[derive(Serialize)]
pub struct ChatMessageOut {
    role: String,
    content: String,
}

#[derive(Serialize)]
pub struct ChatChunk {
    model: String,
    created_at: String,
    message: ChatMessageOut,
    done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    done_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eval_count: Option<u32>,
}

pub async fn chat(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<axum::response::Response, ApiError> {
    let (id, engine) = state.models.resolve(&req.model)?;
    let options = req.options.to_generation_options();
    let messages: Vec<ChatMessage> = req
        .messages
        .iter()
        .map(|m| ChatMessage::text(&m.role, &m.content))
        .collect();
    let tools = req.tools;

    if !req.stream.unwrap_or(true) {
        let (text, usage, ..) = tokio::task::spawn_blocking({
            let engine: Arc<Engine> = Arc::clone(&engine);
            let messages = messages.clone();
            let tools = tools.clone();
            let options = options.clone();
            move || engine.complete_chat(&messages, tools.as_deref(), &options)
        })
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?
        .map_err(ApiError::from)?;

        return Ok(Json(ChatChunk {
            model: id,
            created_at: now_rfc3339(),
            message: ChatMessageOut {
                role: "assistant".to_string(),
                content: text,
            },
            done: true,
            done_reason: Some("stop".to_string()),
            prompt_eval_count: Some(usage.prompt_tokens),
            eval_count: Some(usage.completion_tokens),
        })
        .into_response());
    }

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let cancel = CancelOnDrop::new();
    let flag = cancel.flag();
    state.metrics.record_stream();

    tokio::task::spawn_blocking({
        let engine = Arc::clone(&engine);
        let model = id.clone();
        move || {
            let result = engine.complete_chat_streaming(
                &messages,
                tools.as_deref(),
                &options,
                &flag,
                &mut |piece| {
                    let chunk = ChatChunk {
                        model: model.clone(),
                        created_at: now_rfc3339(),
                        message: ChatMessageOut {
                            role: "assistant".to_string(),
                            content: piece.to_string(),
                        },
                        done: false,
                        done_reason: None,
                        prompt_eval_count: None,
                        eval_count: None,
                    };
                    if let Ok(line) = serde_json::to_string(&chunk) {
                        let _ = tx.send(format!("{line}\n"));
                    }
                },
            );
            match result {
                Ok((_text, usage, _p, _d)) => {
                    let fin = ChatChunk {
                        model: model.clone(),
                        created_at: now_rfc3339(),
                        message: ChatMessageOut {
                            role: "assistant".to_string(),
                            content: String::new(),
                        },
                        done: true,
                        done_reason: Some("stop".to_string()),
                        prompt_eval_count: Some(usage.prompt_tokens),
                        eval_count: Some(usage.completion_tokens),
                    };
                    if let Ok(line) = serde_json::to_string(&fin) {
                        let _ = tx.send(format!("{line}\n"));
                    }
                }
                Err(e) => {
                    let err = serde_json::json!({ "error": e.to_string() });
                    let _ = tx.send(format!("{err}\n"));
                }
            }
        }
    });

    // The `cancel` guard must live as long as the response body: its Drop sets
    // the flag that stops generation when the client disconnects.  Moving it
    // into the unfold state ties its lifetime to the stream.
    let stream = futures_util::stream::unfold((rx, cancel), |(mut rx, cancel)| async move {
        rx.recv()
            .await
            .map(|line| (Ok::<String, std::convert::Infallible>(line), (rx, cancel)))
    });
    Ok(axum::body::Body::from_stream(stream).into_response())
}

// ─── /api/generate ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub model: String,
    #[serde(default)]
    pub prompt: String,
    #[serde(default)]
    pub suffix: Option<String>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: OllamaOptions,
}

#[derive(Serialize)]
pub struct GenerateChunk {
    model: String,
    created_at: String,
    response: String,
    done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    done_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eval_count: Option<u32>,
}

pub async fn generate(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> Result<axum::response::Response, ApiError> {
    let (id, engine) = state.models.resolve(&req.model)?;
    let options = req.options.to_generation_options();
    let prompt = match &req.suffix {
        Some(suffix) => format!("{}{suffix}", req.prompt),
        None => req.prompt.clone(),
    };
    let messages = vec![ChatMessage::text("user", &prompt)];

    if !req.stream.unwrap_or(true) {
        let (text, usage, ..) = tokio::task::spawn_blocking({
            let engine = Arc::clone(&engine);
            let messages = messages.clone();
            let options = options.clone();
            move || engine.complete_chat(&messages, None, &options)
        })
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?
        .map_err(ApiError::from)?;

        return Ok(Json(GenerateChunk {
            model: id,
            created_at: now_rfc3339(),
            response: text,
            done: true,
            done_reason: Some("stop".to_string()),
            prompt_eval_count: Some(usage.prompt_tokens),
            eval_count: Some(usage.completion_tokens),
        })
        .into_response());
    }

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let cancel = CancelOnDrop::new();
    let flag = cancel.flag();
    state.metrics.record_stream();

    tokio::task::spawn_blocking({
        let engine = Arc::clone(&engine);
        let model = id.clone();
        move || {
            let result =
                engine.complete_chat_streaming(&messages, None, &options, &flag, &mut |piece| {
                    let chunk = GenerateChunk {
                        model: model.clone(),
                        created_at: now_rfc3339(),
                        response: piece.to_string(),
                        done: false,
                        done_reason: None,
                        prompt_eval_count: None,
                        eval_count: None,
                    };
                    if let Ok(line) = serde_json::to_string(&chunk) {
                        let _ = tx.send(format!("{line}\n"));
                    }
                });
            match result {
                Ok((_text, usage, _p, _d)) => {
                    let fin = GenerateChunk {
                        model: model.clone(),
                        created_at: now_rfc3339(),
                        response: String::new(),
                        done: true,
                        done_reason: Some("stop".to_string()),
                        prompt_eval_count: Some(usage.prompt_tokens),
                        eval_count: Some(usage.completion_tokens),
                    };
                    if let Ok(line) = serde_json::to_string(&fin) {
                        let _ = tx.send(format!("{line}\n"));
                    }
                }
                Err(e) => {
                    let err = serde_json::json!({ "error": e.to_string() });
                    let _ = tx.send(format!("{err}\n"));
                }
            }
        }
    });

    // The `cancel` guard must live as long as the response body: its Drop sets
    // the flag that stops generation when the client disconnects.  Moving it
    // into the unfold state ties its lifetime to the stream.
    let stream = futures_util::stream::unfold((rx, cancel), |(mut rx, cancel)| async move {
        rx.recv()
            .await
            .map(|line| (Ok::<String, std::convert::Infallible>(line), (rx, cancel)))
    });
    Ok(axum::body::Body::from_stream(stream).into_response())
}

// ─── /api/embed + /api/embeddings ───────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct EmbedRequest {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub input: Option<serde_json::Value>,
    /// `/api/embeddings` (legacy) uses `prompt` instead of `input`.
    #[serde(default)]
    pub prompt: Option<String>,
}

#[derive(Serialize)]
pub struct EmbedResponse {
    model: String,
    embeddings: Vec<Vec<f32>>,
}

pub async fn embed(
    State(state): State<AppState>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, ApiError> {
    let (_id, engine) = state.models.resolve(&req.model)?;

    let inputs: Vec<String> = match (&req.input, &req.prompt) {
        (Some(serde_json::Value::String(s)), _) => vec![s.clone()],
        (Some(serde_json::Value::Array(a)), _) => a
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect(),
        (_, Some(p)) => vec![p.clone()],
        _ => Vec::new(),
    };
    if inputs.is_empty() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "`input` or `prompt` is required",
            "invalid_request_error",
        ));
    }

    let vectors = tokio::task::spawn_blocking({
        let engine = Arc::clone(&engine);
        let inputs = inputs.clone();
        move || engine.embed(&inputs)
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?
    .map_err(|e| ApiError::internal(e.to_string()))?;

    Ok(Json(EmbedResponse {
        model: req.model,
        embeddings: vectors,
    }))
}

// ─── Model management: /api/delete, /api/copy, /api/create, /api/pull ────────
//
// Ollama runs a content-addressed blob store and a remote library; joshua
// reads plain `.gguf` + `tokenizer.json` from a directory.  These endpoints
// keep the *protocol* working for clients and scripts that manage models
// through the API, while operating on that directory instead of a blob store:
//
// * `delete`/`copy`/`create(from:)` act on `.gguf` files inside the model
//   directory, and never on a path that resolves outside it.
// * `create(files: ...)` — Ollama's blob-digest form — is not supported: there
//   is no blob store to name files in.  It returns a clear 400 rather than
//   silently ignoring the field.
// * `pull` fetches from an explicit https URL (or a `hf.co/<repo>` shorthand
//   is rejected with guidance); it does not speak the Ollama registry
//   protocol, so `ollama pull llama3` cannot resolve.

/// Resolve a requested model name to a `.gguf` inside the model directory.
///
/// Confinement matters here: these endpoints delete and overwrite files, so a
/// name containing `..` or an absolute path must not escape the directory.
fn model_file(state: &AppState, name: &str) -> Result<PathBuf, ApiError> {
    let manager = state.manager.as_ref().ok_or_else(|| {
        ApiError::new(
            StatusCode::FORBIDDEN,
            "model management is disabled on this server",
            "management_disabled",
        )
    })?;
    let stem = strip_tag(name);
    if stem.is_empty() || stem.contains('/') || stem.contains('\\') || stem.contains("..") {
        return Err(ApiError::bad_request(&format!(
            "invalid model name `{name}`: expected a bare name inside the model directory"
        )));
    }
    let path = manager.model_dir.join(format!("{stem}.gguf"));
    // A name that canonicalises outside the directory (a symlink pointing out,
    // for instance) is refused even when the joined path looks contained.
    if let Ok(dir) = manager.model_dir.canonicalize() {
        if let Ok(real) = path.canonicalize() {
            if !real.starts_with(&dir) {
                return Err(ApiError::bad_request(&format!(
                    "model `{name}` resolves outside the model directory"
                )));
            }
        }
    }
    Ok(path)
}

/// `DELETE /api/delete` — remove a model's file.  404 when it does not exist,
/// matching Ollama.
pub async fn delete(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<StatusCode, ApiError> {
    let name = body
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::bad_request("missing 'model'"))?;
    let path = model_file(&state, name)?;
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(StatusCode::OK),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(ApiError::new(
            StatusCode::NOT_FOUND,
            &format!("model `{name}` not found"),
            "model_not_found",
        )),
        Err(e) => Err(ApiError::internal(format!(
            "removing {}: {e}",
            path.display()
        ))),
    }
}

/// `POST /api/copy` — duplicate a model file under a new name.
pub async fn copy(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<StatusCode, ApiError> {
    let source = body
        .get("source")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::bad_request("missing 'source'"))?;
    let dest = body
        .get("destination")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::bad_request("missing 'destination'"))?;
    let from = model_file(&state, source)?;
    let to = model_file(&state, dest)?;
    if !from.exists() {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            &format!("model `{source}` not found"),
            "model_not_found",
        ));
    }
    std::fs::copy(&from, &to)
        .map_err(|e| ApiError::internal(format!("copying to {}: {e}", to.display())))?;
    Ok(StatusCode::OK)
}

/// `POST /api/create` — create a model.
///
/// Only the `from` form is meaningful without a blob store: it copies an
/// existing model under a new name (Ollama's "derive a model from another").
/// The `files` form names blobs by SHA256, which joshua has no store for, so it
/// is rejected explicitly instead of being ignored.
pub async fn create(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<StatusCode, ApiError> {
    let name = body
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::bad_request("missing 'model'"))?;
    if body.get("files").is_some() {
        return Err(ApiError::bad_request(
            "creating from blob digests is not supported: joshua reads plain .gguf              files from its model directory, so put the file there and use \
             `\"from\": \"<name>\"` or POST /v1/models/load instead",
        ));
    }
    let source = body.get("from").and_then(|v| v.as_str()).ok_or_else(|| {
        ApiError::bad_request("missing 'from': joshua cannot build a model from template parameters alone")
    })?;
    let from = model_file(&state, source)?;
    let to = model_file(&state, name)?;
    if !from.exists() {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            &format!("model `{source}` not found"),
            "model_not_found",
        ));
    }
    std::fs::copy(&from, &to)
        .map_err(|e| ApiError::internal(format!("writing {}: {e}", to.display())))?;
    Ok(StatusCode::OK)
}

/// `POST /api/pull` — fetch a model file from an explicit https URL.
///
/// Ollama's registry protocol (`model:tag` resolved against its library) is not
/// implemented: it needs the registry manifest/blob API, which joshua does not
/// speak.  A bare library name therefore fails with guidance rather than
/// hanging, while an explicit https URL downloads into the model directory.
/// The response is a single object (`stream: false` shape), which every client
/// accepts, and the final `status` is `success` on completion.
pub async fn pull(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let name = body
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::bad_request("missing 'model'"))?;
    if !name.starts_with("https://") {
        return Err(ApiError::bad_request(&format!(
            "cannot pull `{name}`: joshua does not implement the Ollama registry protocol.              Pass an explicit https:// URL to a .gguf file, or place the file in the              model directory and load it with POST /v1/models/load."
        )));
    }
    Err(ApiError::new(
        StatusCode::NOT_IMPLEMENTED,
        "downloading from a URL is not implemented yet; place the .gguf in the model \
         directory and POST /v1/models/load",
        "pull_not_implemented",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_label_matches_llamacpp_enum() {
        assert_eq!(quant_label(15).as_deref(), Some("Q4_K_M"));
        assert_eq!(quant_label(35).as_deref(), Some("BF16"));
        assert_eq!(quant_label(9999), None);
    }

    #[test]
    fn names_gain_latest_tag_once() {
        assert_eq!(normalise_name("qwen3"), "qwen3:latest");
        assert_eq!(normalise_name("qwen3:8b"), "qwen3:8b");
    }

    #[test]
    fn options_map_onto_generation_options() {
        let o = OllamaOptions {
            num_predict: Some(64),
            temperature: Some(0.0),
            top_k: Some(20),
            repeat_penalty: Some(1.1),
            stop: Some(vec!["</s>".into()]),
            ..Default::default()
        };
        let g = o.to_generation_options();
        assert_eq!(g.max_tokens, 64);
        assert_eq!(g.temperature, 0.0);
        assert_eq!(g.top_k, 20);
        assert_eq!(g.repetition_penalty, 1.1);
        assert_eq!(g.stop_sequences, vec!["</s>".to_string()]);
    }

    #[test]
    fn tags_are_stripped_for_file_names() {
        assert_eq!(strip_tag("llama3"), "llama3");
        assert_eq!(strip_tag("llama3:latest"), "llama3");
        assert_eq!(strip_tag("llama3:13b"), "llama3");
        assert_eq!(strip_tag("  llama3  "), "llama3");
    }

    /// Model management deletes and overwrites files, so a name must never
    /// escape the model directory.  These are the shapes that would otherwise
    /// reach outside it.
    #[test]
    fn unsafe_model_names_are_rejected() {
        for bad in ["", "../etc/passwd", "a/b", "..", "dir/../x", "/abs/path"] {
            let stem = strip_tag(bad);
            let rejected = stem.is_empty()
                || stem.contains('/')
                || stem.contains('\\')
                || stem.contains("..");
            assert!(rejected, "`{bad}` must be rejected as a model name");
        }
        // The bare names clients actually send must pass.
        for good in ["llama3", "llama3:13b", "my-model_1", "Qwen3-30B-A3B"] {
            let stem = strip_tag(good);
            assert!(
                !stem.is_empty() && !stem.contains('/') && !stem.contains(".."),
                "`{good}` must be accepted"
            );
        }
    }

    #[test]
    fn unlimited_predict_does_not_clobber_default() {
        // -1 / -2 mean "unbounded" / "fill context" in Ollama; neither maps to
        // a positive max_tokens, so joshua's default must survive.
        let o = OllamaOptions { num_predict: Some(-1), ..Default::default() };
        assert_eq!(o.to_generation_options().max_tokens, GenerationOptions::default().max_tokens);
        let o = OllamaOptions { num_predict: Some(-2), ..Default::default() };
        assert_eq!(o.to_generation_options().max_tokens, GenerationOptions::default().max_tokens);
    }
}
