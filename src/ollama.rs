//! Ollama-compatible API (`/api/*`) on top of joshua's model registry.
//!
//! Ollama's native protocol differs from OpenAI's in four ways that matter:
//!
//! 1. **Streaming is newline-delimited JSON**, not SSE.  Each line is a whole
//!    object; the last carries `"done": true` and the timings.
//! 2. **Generation parameters live under `options`** (`num_ctx`, `num_predict`,
//!    `temperature`, `top_k`, `top_p`, `min_p`, `typical_p`, `repeat_penalty`,
//!    `repeat_last_n`, `presence_penalty`, `frequency_penalty`, `mirostat*`,
//!    `seed`, `stop`).
//! 3. **Models are addressed by `name:tag`** from a listing endpoint
//!    (`/api/tags`) that must work before anything is loaded.
//! 4. **Errors are `{"error": "<message>"}`**, and the CLI checks the server
//!    with `HEAD /` before every command.
//!
//! This is a translation layer: names resolve through the loaded
//! [`crate::server::ModelRegistry`] (which already dedups concurrent loads and
//! confines paths to the model directory), then drive the same engine and the
//! same output post-processing (reasoning split, tool-call parsing) the
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
//!   file in by hand.  An explicit `https://` URL is answered with a 501:
//!   URL downloads are not implemented.
//!
//! Ollama's `:tag` suffix names a manifest, which has no joshua equivalent:
//! a name whose tag is not part of a model id resolves to the id without it,
//! so `llama3:latest` and `llama3:13b` both address `llama3.gguf`.
//!
//! **`keep_alive`**: `0` unloads the model once the request is answered (when
//! the server manages a model directory, so it can load it again); a request
//! with no messages or prompt only loads or unloads the model, as Ollama's
//! does.  Other values are accepted and change nothing: joshua keeps a model
//! resident until it is unloaded, so `/api/ps` reports no expiry.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use axum::{
    extract::{Request, State},
    http::{header, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::engine::{Completion, Engine};
use crate::model_info::{GgufFacts, ModelInfo};
use crate::sampling::{SamplingDefaults, SamplingProfile};
use crate::server::{select_model, ApiError, AppState, CancelOnDrop, ChatPiece};
use crate::types::{rfc3339, ChatMessage, ConstraintSpec, GenerationOptions, SamplingParams, Tool};

fn now() -> String {
    rfc3339(SystemTime::now())
}

/// `GET /` and `HEAD /`: Ollama's heartbeat.
pub async fn heartbeat() -> &'static str {
    "Ollama is running"
}

/// Give every error an `/api/*` route returns Ollama's shape,
/// `{"error": "<message>"}`: the OpenAI envelope the shared handlers build,
/// and the plain-text rejections of a body that does not parse.
pub async fn error_shape(req: Request, next: Next) -> Response {
    let ollama = req.uri().path().starts_with("/api/");
    let response = next.run(req).await;
    let status = response.status();
    if !ollama || !(status.is_client_error() || status.is_server_error()) {
        return response;
    }
    let (mut parts, body) = response.into_parts();
    let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap_or_default();
    let message = error_message(&bytes);
    parts.headers.remove(header::CONTENT_LENGTH);
    parts.headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    (parts, Json(json!({ "error": message }))).into_response()
}

/// The message in an error body: an OpenAI envelope's `error.message`, an
/// Ollama body's `error`, or the body's text.
fn error_message(body: &[u8]) -> String {
    let parsed = serde_json::from_slice::<Value>(body).ok();
    let from_json = parsed.as_ref().and_then(|v| match v.get("error")? {
        Value::String(message) => Some(message.clone()),
        error => error.get("message")?.as_str().map(str::to_string),
    });
    from_json.unwrap_or_else(|| String::from_utf8_lossy(body).trim().to_string())
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

// ─── model names ────────────────────────────────────────────────────────────

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

/// The model id an Ollama name stands for: the name itself when a model
/// (loaded, or in the model directory) has that id, else the name without
/// its tag when one does, else the name as given (so the shared resolver
/// reports it).
fn model_id(state: &AppState, name: &str) -> String {
    let name = name.trim();
    let known = |id: &str| {
        state.models.list().iter().any(|m| m.id == id)
            || state.unloaded().iter().any(|(listed, _)| listed == id)
    };
    if known(name) {
        return name.to_string();
    }
    let bare = strip_tag(name);
    if bare != name && known(bare) {
        return bare.to_string();
    }
    name.to_string()
}

/// [`select_model`] for an Ollama name, loading the model if it is not
/// loaded.  Also returns how long that took (Ollama's `load_duration`).
async fn select(state: &AppState, name: &str) -> Result<(String, Arc<Engine>, Duration), ApiError> {
    let start = Instant::now();
    let (id, engine) = select_model(state, &model_id(state, name)).await?;
    Ok((id, engine, start.elapsed()))
}

// ─── model details (listing: `ServerState::inventory`) ──────────────────────

#[derive(Serialize, Clone, Default)]
pub struct TagDetails {
    parent_model: String,
    format: String,
    family: String,
    families: Vec<String>,
    parameter_size: String,
    quantization_level: String,
}

fn details_for(facts: Option<&GgufFacts>) -> TagDetails {
    let mut d = TagDetails {
        format: "gguf".to_string(),
        ..Default::default()
    };
    let Some(facts) = facts else { return d };
    if let Some(family) = &facts.architecture {
        d.families = vec![family.clone()];
        d.family = family.clone();
    }
    d.parameter_size = parameter_size(facts.n_params);
    d.quantization_level = facts.quantization().unwrap_or_default().to_string();
    d
}

/// A parameter count as Ollama writes it: `14.8B`, `596.05M`.
fn parameter_size(n: u64) -> String {
    let n = n as f64;
    match n {
        n if n >= 1e9 => format!("{:.1}B", n / 1e9),
        n if n >= 1e6 => format!("{:.2}M", n / 1e6),
        n if n >= 1e3 => format!("{:.2}K", n / 1e3),
        0.0 => String::new(),
        n => format!("{n}"),
    }
}

/// When the model's file was last written (now, when that is unknown).
fn modified_at(path: Option<&Path>) -> String {
    let modified = path
        .and_then(|p| std::fs::metadata(p).ok())
        .and_then(|m| m.modified().ok())
        .unwrap_or_else(SystemTime::now);
    rfc3339(modified)
}

/// A stable 64-hex-digit digest standing in for Ollama's manifest digest,
/// which clients print a prefix of and expect to be present.  It hashes the
/// model's id, file, size and modification time rather than its contents,
/// so listing a directory of large files reads no weights; it changes when
/// the file is replaced.
fn digest(info: &ModelInfo) -> String {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    hash.update(info.id.as_bytes());
    if let Some(path) = &info.path {
        hash.update(path.to_string_lossy().as_bytes());
        hash.update(modified_at(Some(path)).as_bytes());
    }
    hash.update(info.facts.as_ref().map_or(0, |f| f.file_size).to_le_bytes());
    format!("{:x}", hash.finalize())
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

impl TagModel {
    fn of(info: &ModelInfo) -> Self {
        let mut details = details_for(info.facts.as_ref());
        details.parent_model = info.parent.clone().unwrap_or_default();
        Self {
            name: normalise_name(&info.id),
            model: normalise_name(&info.id),
            modified_at: modified_at(info.path.as_deref()),
            size: info.facts.as_ref().map_or(0, |f| f.file_size),
            digest: digest(info),
            details,
        }
    }
}

#[derive(Serialize)]
pub struct TagsResponse {
    models: Vec<TagModel>,
}

pub async fn tags(State(state): State<AppState>) -> Result<Json<TagsResponse>, ApiError> {
    let inventory = crate::server::inventory(&state).await?;
    let models = tokio::task::spawn_blocking(move || inventory.iter().map(TagModel::of).collect())
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;
    Ok(Json(TagsResponse { models }))
}

// ─── /api/ps ────────────────────────────────────────────────────────────────

/// What `/api/ps` reports when a model never expires: joshua keeps a model
/// resident until it is unloaded.
const NEVER: &str = "2318-01-01T00:00:00Z";

#[derive(Serialize)]
pub struct PsModel {
    #[serde(flatten)]
    tag: TagModel,
    expires_at: String,
    size_vram: u64,
    context_length: u32,
}

/// Bytes of the model on an accelerator: all of it when its dense weights
/// and experts run there, its expert cache when only the dense weights do
/// (the experts stay in host memory), none on the CPU.
fn size_vram(info: &ModelInfo) -> u64 {
    let size = info.facts.as_ref().map_or(0, |f| f.file_size);
    match &info.placement {
        Some(p) if p.dense != "cpu" && p.experts.as_deref().is_none_or(|e| e != "cpu") => size,
        Some(p) if p.dense != "cpu" => p.expert_cache_bytes.unwrap_or(0),
        _ => 0,
    }
}

#[derive(Serialize)]
pub struct PsResponse {
    models: Vec<PsModel>,
}

pub async fn ps(State(state): State<AppState>) -> Result<Json<PsResponse>, ApiError> {
    let loaded = state.models.list();
    let models = tokio::task::spawn_blocking(move || {
        loaded
            .iter()
            .map(crate::server::LoadedModel::info)
            .map(|info| PsModel {
                tag: TagModel::of(&info),
                expires_at: NEVER.to_string(),
                size_vram: size_vram(&info),
                context_length: info.limits.as_ref().map_or(0, |l| l.context_length),
            })
            .collect()
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?;
    Ok(Json(PsResponse { models }))
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
    license: String,
    modelfile: String,
    parameters: String,
    template: String,
    details: TagDetails,
    model_info: serde_json::Map<String, Value>,
    capabilities: Vec<String>,
    modified_at: String,
}

/// `POST /api/show` — a model's details, header facts, template, default
/// parameters and capabilities.  A model in the directory that is not
/// loaded is described from its header alone: its template, parameters and
/// capabilities beyond `completion` are known once it is loaded.
pub async fn show(
    State(state): State<AppState>,
    Json(req): Json<ShowRequest>,
) -> Result<Json<ShowResponse>, ApiError> {
    let requested = req.model.filter(|m| !m.is_empty()).unwrap_or(req.name);
    let id = model_id(&state, &requested);
    let inventory = crate::server::inventory(&state).await?;
    let info = match inventory.into_iter().find(|m| m.id == id) {
        Some(info) => info,
        // A name no model has: the shared resolver's answer (the one loaded
        // model, or why there is none).
        None => {
            let (id, _) = state.resolve(&id)?;
            let loaded = state.models.list();
            let model = loaded
                .iter()
                .find(|m| m.id == id)
                .ok_or_else(|| ApiError::unknown_model(&requested))?;
            model.info()
        }
    };
    let engine = state
        .models
        .list()
        .into_iter()
        .find(|m| m.id == info.id)
        .map(|m| m.engine);
    let response = tokio::task::spawn_blocking(move || show_response(&info, engine.as_deref()))
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;
    Ok(Json(response))
}

fn show_response(info: &ModelInfo, engine: Option<&Engine>) -> ShowResponse {
    let mut details = details_for(info.facts.as_ref());
    details.parent_model = info.parent.clone().unwrap_or_default();
    let template = engine
        .and_then(Engine::chat_template_source)
        .unwrap_or_default()
        .to_string();
    let parameters = engine
        .map(|e| parameters(&e.sampling_defaults(SamplingProfile::Ollama)))
        .unwrap_or_default();
    let modelfile = format!(
        "# Modelfile generated by joshua\nFROM {}\n",
        info.path.as_deref().map_or(info.id.clone(), |p| p.display().to_string())
    );
    ShowResponse {
        license: String::new(),
        modelfile,
        parameters,
        template,
        details,
        model_info: model_info(info),
        capabilities: engine.map_or_else(|| vec!["completion".to_string()], capabilities),
        modified_at: modified_at(info.path.as_deref()),
    }
}

/// The GGUF header facts under their metadata keys, as Ollama's
/// `model_info` carries them.
fn model_info(info: &ModelInfo) -> serde_json::Map<String, Value> {
    let mut out = serde_json::Map::new();
    let Some(facts) = &info.facts else { return out };
    let arch = facts.architecture.as_deref();
    if let Some(arch) = arch {
        out.insert("general.architecture".into(), arch.into());
    }
    out.insert("general.parameter_count".into(), facts.n_params.into());
    if let Some(ft) = facts.file_type {
        out.insert("general.file_type".into(), ft.into());
    }
    if let Some(arch) = arch {
        for (key, value) in [
            ("context_length", facts.n_ctx_train),
            ("embedding_length", facts.n_embd),
            ("vocab_size", facts.n_vocab),
            ("expert_count", facts.expert_count),
        ] {
            if let Some(value) = value {
                out.insert(format!("{arch}.{key}"), value.into());
            }
        }
    }
    out
}

/// The sampler defaults a request gets, one `name value` per line as a
/// Modelfile's `PARAMETER`s read.
fn parameters(d: &SamplingDefaults) -> String {
    [
        ("temperature", d.temperature.to_string()),
        ("top_k", d.top_k.to_string()),
        ("top_p", d.top_p.to_string()),
        ("min_p", d.min_p.to_string()),
        ("repeat_penalty", d.repetition_penalty.to_string()),
    ]
    .iter()
    .map(|(name, value)| format!("{name:<30} {value}"))
    .collect::<Vec<_>>()
    .join("\n")
}

/// What a loaded model can do, in Ollama's capability names.
fn capabilities(engine: &Engine) -> Vec<String> {
    let mut caps = vec!["completion"];
    if engine
        .chat_template_source()
        .is_some_and(|t| t.contains("tools"))
    {
        caps.push("tools");
    }
    if engine.fim_tokens().missing().is_none() {
        caps.push("insert");
    }
    if *engine.reasoning_format() != crate::reasoning::ReasoningFormat::None {
        caps.push("thinking");
    }
    if engine.supports_embeddings() {
        caps.push("embedding");
    }
    caps.into_iter().map(str::to_string).collect()
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
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub typical_p: Option<f32>,
    #[serde(default)]
    pub repeat_last_n: Option<i32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub mirostat: Option<i32>,
    #[serde(default)]
    pub mirostat_tau: Option<f32>,
    #[serde(default)]
    pub mirostat_eta: Option<f32>,
}

impl OllamaOptions {
    /// Generation options over `defaults` (the Ollama profile: see
    /// [`crate::Engine::sampling_defaults`]), through the OpenAI routes'
    /// request mapping.
    ///
    /// `num_ctx` narrows the context window to at most `loaded_ctx`, the
    /// server's `--n-ctx`.  Ollama treats it as a load parameter and clients
    /// send their own default (often larger than the server's window), so a
    /// larger value is clamped rather than refused as the OpenAI routes'
    /// `context_window` is; 0 leaves the window alone.
    fn to_generation_options(
        &self,
        defaults: &SamplingDefaults,
        loaded_ctx: u32,
    ) -> Result<GenerationOptions, String> {
        let mut o = GenerationOptions::with_defaults(defaults);
        o.context_window = self
            .num_ctx
            .filter(|&n| n > 0)
            .map(|n| n.min(loaded_ctx.max(1)));
        SamplingParams {
            // Ollama: -1 = until stop, -2 = fill context; both are the
            // unlimited default (still capped by --max-output-tokens).
            n_predict: self.num_predict.filter(|&n| n > 0).map(i64::from),
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            repetition_penalty: self.repeat_penalty,
            stop: self.stop.as_ref().map(|stop| serde_json::Value::from(stop.clone())),
            seed: self.seed,
            typical_p: self.typical_p,
            repeat_last_n: self.repeat_last_n,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            mirostat: self.mirostat,
            mirostat_tau: self.mirostat_tau,
            mirostat_eta: self.mirostat_eta,
            ..Default::default()
        }
        .apply(&mut o)?;
        Ok(o)
    }
}

// ─── request fields shared by /api/chat and /api/generate ───────────────────

/// Ollama's `format`: `"json"` for any JSON value (an object, as the
/// OpenAI routes' `json_object`), or a JSON schema; `""` or `null` for
/// free text.
fn constraint(format: Option<&Value>) -> Result<Option<ConstraintSpec>, String> {
    match format {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if s.is_empty() => Ok(None),
        Some(Value::String(s)) if s == "json" => Ok(Some(ConstraintSpec::JsonObject)),
        Some(Value::Object(_)) => Ok(Some(ConstraintSpec::JsonSchema {
            schema: format.cloned().unwrap_or_default(),
        })),
        Some(other) => Err(format!(
            "invalid format {other}: expected \"json\" or a JSON schema object"
        )),
    }
}

/// Whether `keep_alive` asks for the model to be unloaded once the request
/// is answered: `0`, or a duration of zero (`"0"`, `"0s"`, `"0m"`).
fn unload_after(keep_alive: Option<&Value>) -> bool {
    match keep_alive {
        Some(Value::Number(n)) => n.as_f64() == Some(0.0),
        Some(Value::String(s)) => {
            let number = s.trim().trim_end_matches(|c: char| c.is_ascii_alphabetic());
            number.parse::<f64>().is_ok_and(|n| n == 0.0)
        }
        _ => false,
    }
}

/// Ollama's images are bare base64; the engine reads `data:` URLs.
fn image_url(image: &str) -> String {
    if image.starts_with("data:") {
        image.to_string()
    } else {
        format!("data:application/octet-stream;base64,{image}")
    }
}

/// An Ollama chat message as the engine's [`ChatMessage`]: its images as
/// `data:` URLs, a tool result's `tool_name` as the message's name, and
/// earlier tool calls (whose arguments are objects) passed on as sent.
fn chat_message(mut message: Value) -> Result<ChatMessage, String> {
    if let Some(obj) = message.as_object_mut() {
        if let Some(Value::Array(images)) = obj.get_mut("images") {
            for image in images.iter_mut() {
                if let Value::String(s) = image {
                    *s = image_url(s);
                }
            }
        }
        if !obj.contains_key("name") {
            if let Some(tool_name) = obj.remove("tool_name") {
                obj.insert("name".to_string(), tool_name);
            }
        }
    }
    serde_json::from_value(message).map_err(|e| format!("invalid message: {e}"))
}

/// Which Ollama wire a reply goes out on: `/api/chat` puts text under
/// `message.content`, `/api/generate` under `response`.
#[derive(Clone, Copy)]
enum Wire {
    Chat,
    Generate,
}

impl Wire {
    /// One object of the reply: `text` and, for chat, any `tool_calls`.
    fn chunk(self, model: &str, text: &str, tool_calls: Option<Vec<Value>>) -> Value {
        let mut out = json!({ "model": model, "created_at": now() });
        match self {
            Wire::Chat => {
                let mut message = json!({ "role": "assistant", "content": text });
                if let Some(calls) = tool_calls {
                    message["tool_calls"] = Value::Array(calls);
                }
                out["message"] = message;
            }
            Wire::Generate => out["response"] = Value::from(text),
        }
        out["done"] = Value::Bool(false);
        out
    }

    /// The last object: `done`, why it ended, the token counts and the
    /// durations in nanoseconds.
    fn done(self, model: &str, text: &str, end: &End<'_>) -> Value {
        let mut out = self.chunk(model, text, end.tool_calls.clone());
        out["done"] = Value::Bool(true);
        out["done_reason"] = Value::from(end.reason);
        if let Some(c) = end.completion {
            let ns = |ms: f64| (ms * 1e6).round() as u64;
            out["total_duration"] = Value::from(end.started.elapsed().as_nanos() as u64);
            out["load_duration"] = Value::from(end.load.as_nanos() as u64);
            out["prompt_eval_count"] = Value::from(c.usage.prompt_tokens);
            out["prompt_eval_duration"] = Value::from(ns(c.timings.prompt_ms));
            out["eval_count"] = Value::from(c.usage.completion_tokens);
            out["eval_duration"] = Value::from(ns(c.timings.predicted_ms));
        } else {
            out["total_duration"] = Value::from(end.started.elapsed().as_nanos() as u64);
            out["load_duration"] = Value::from(end.load.as_nanos() as u64);
        }
        out
    }
}

/// How a reply ended.
struct End<'a> {
    reason: &'static str,
    completion: Option<&'a Completion>,
    tool_calls: Option<Vec<Value>>,
    started: Instant,
    load: Duration,
}

/// Parsed tool calls in Ollama's shape: arguments as an object.
fn ollama_calls(calls: &[crate::tools::ParsedToolCall]) -> Option<Vec<Value>> {
    (!calls.is_empty()).then(|| {
        calls
            .iter()
            .enumerate()
            .map(|(index, call)| {
                let arguments = serde_json::from_str::<serde_json::Map<String, Value>>(&call.arguments)
                    .unwrap_or_default();
                json!({
                    "id": call.id.clone().unwrap_or_default(),
                    "function": { "index": index, "name": call.name, "arguments": arguments },
                })
            })
            .collect()
    })
}

/// A request for `/api/chat` or `/api/generate`, translated.
struct Turn {
    wire: Wire,
    model: String,
    messages: Vec<ChatMessage>,
    tools: Option<Vec<Tool>>,
    options: OllamaOptions,
    format: Option<Value>,
    keep_alive: Option<Value>,
    stream: bool,
}

/// Answer a [`Turn`]: load or unload only, when it has nothing to say, else
/// generate, streaming newline-delimited JSON or in one object.
async fn answer(state: AppState, turn: Turn) -> Result<Response, ApiError> {
    let started = Instant::now();
    let unload = unload_after(turn.keep_alive.as_ref()) && state.manager.is_some();
    if turn.messages.is_empty() {
        let (id, reason, load) = if unload {
            let id = model_id(&state, &turn.model);
            if state.models.list().iter().any(|m| m.id == id) {
                crate::server::unload(&state, &id).await?;
            }
            (id, "unload", Duration::ZERO)
        } else {
            let (id, _engine, load) = select(&state, &turn.model).await?;
            (id, "load", load)
        };
        let end = End { reason, completion: None, tool_calls: None, started, load };
        return Ok(Json(turn.wire.done(&id, "", &end)).into_response());
    }

    let (id, engine, load) = select(&state, &turn.model).await?;
    let mut options = turn
        .options
        .to_generation_options(&engine.sampling_defaults(SamplingProfile::Ollama), engine.n_ctx())
        .map_err(ApiError::bad_request)?;
    options.constraint = constraint(turn.format.as_ref()).map_err(ApiError::bad_request)?;
    // Answer a request the model cannot serve before a stream starts.
    engine.validate_options(&options).map_err(ApiError::from)?;
    let Turn { wire, messages, tools, stream, .. } = turn;
    // `keep_alive: 0`: unload once the engine is no longer in use.
    let release = move |state: AppState, id: String| async move {
        if unload {
            if let Err(e) = crate::server::unload(&state, &id).await {
                tracing::debug!(model = %id, "keep_alive unload: {}", e.into_response().status());
            }
        }
    };

    let cancel = CancelOnDrop::new();
    if !stream {
        let completion = tokio::task::spawn_blocking({
            let engine = Arc::clone(&engine);
            let messages = messages.clone();
            let tools = tools.clone();
            let flag = cancel.flag();
            move || {
                let format = match &tools {
                    Some(tools) => engine.tool_call_format_for_request(&messages, Some(tools), &options)?,
                    None => engine.tool_call_format(),
                };
                let completion = engine.complete_chat_cancellable_detailed(
                    &messages,
                    tools.as_deref(),
                    &options,
                    &flag,
                )?;
                Ok::<_, crate::JoshuaError>((completion, format))
            }
        })
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?
        .map_err(ApiError::from)?;
        let (completion, format) = completion;
        state.metrics.record_generation(&completion.usage);
        let answer = crate::server::chat_answer(
            &engine,
            &completion,
            false,
            tools.as_deref(),
            format,
            "call_",
        );
        let end = End {
            reason: completion.stop_type.finish_reason(),
            completion: Some(&completion),
            tool_calls: ollama_calls(&answer.calls),
            started,
            load,
        };
        let body = wire.done(&id, &answer.prose, &end);
        drop(engine);
        release(Arc::clone(&state), id).await;
        return Ok(Json(body).into_response());
    }

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let flag = cancel.flag();
    state.metrics.record_stream();
    let runtime = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking({
        let state = Arc::clone(&state);
        move || {
            let send = |value: Value| {
                let _ = tx.send(format!("{value}\n"));
            };
            let result = crate::server::stream_chat_output(
                &engine,
                &messages,
                tools.as_deref(),
                &options,
                &flag,
                false,
                "call_",
                &mut |piece| {
                    // Calls go out whole, once the answer is read (below).
                    if let ChatPiece::Text(text) = piece {
                        send(wire.chunk(&id, text, None));
                    }
                },
            );
            match result {
                Ok((completion, calls)) => {
                    state.metrics.record_generation(&completion.usage);
                    let tool_calls = ollama_calls(&calls);
                    if tool_calls.is_some() {
                        send(wire.chunk(&id, "", tool_calls.clone()));
                    }
                    let end = End {
                        reason: completion.stop_type.finish_reason(),
                        completion: Some(&completion),
                        tool_calls: None,
                        started,
                        load,
                    };
                    send(wire.done(&id, "", &end));
                }
                Err(e) => send(json!({ "error": e.to_string() })),
            }
            drop(engine);
            runtime.spawn(release(state, id));
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
    let mut response = axum::body::Body::from_stream(stream).into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/x-ndjson"),
    );
    Ok(response)
}

// ─── /api/chat ──────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<Value>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: OllamaOptions,
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    #[serde(default)]
    pub format: Option<Value>,
    #[serde(default)]
    pub keep_alive: Option<Value>,
}

pub async fn chat(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Response, ApiError> {
    let messages = req
        .messages
        .into_iter()
        .map(chat_message)
        .collect::<Result<Vec<_>, _>>()
        .map_err(ApiError::bad_request)?;
    let turn = Turn {
        wire: Wire::Chat,
        model: req.model,
        messages,
        tools: req.tools.filter(|t| !t.is_empty()),
        options: req.options,
        format: req.format,
        keep_alive: req.keep_alive,
        stream: req.stream.unwrap_or(true),
    };
    answer(state, turn).await
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
    pub images: Option<Vec<String>>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: OllamaOptions,
    #[serde(default)]
    pub format: Option<Value>,
    #[serde(default)]
    pub keep_alive: Option<Value>,
}

pub async fn generate(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> Result<Response, ApiError> {
    let prompt = match &req.suffix {
        Some(suffix) => format!("{}{suffix}", req.prompt),
        None => req.prompt.clone(),
    };
    let images = req.images.unwrap_or_default();
    // An empty prompt only loads (or, with `keep_alive: 0`, unloads).
    let messages = if prompt.is_empty() && images.is_empty() {
        Vec::new()
    } else {
        let mut message = ChatMessage::text("user", &prompt);
        if !images.is_empty() {
            message.images = Some(images.iter().map(|i| image_url(i)).collect());
        }
        vec![message]
    };
    let turn = Turn {
        wire: Wire::Generate,
        model: req.model,
        messages,
        tools: None,
        options: req.options,
        format: req.format,
        keep_alive: req.keep_alive,
        stream: req.stream.unwrap_or(true),
    };
    answer(state, turn).await
}

// ─── /api/embed + /api/embeddings ───────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct EmbedRequest {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub input: Option<Value>,
    /// `/api/embeddings` (legacy) uses `prompt` instead of `input`.
    #[serde(default)]
    pub prompt: Option<String>,
}

/// Embed a request's inputs: its vectors, prompt token count, model id and
/// load time.
async fn embed_inputs(
    state: &AppState,
    req: &EmbedRequest,
) -> Result<(Vec<Vec<f32>>, u32, String, Duration), ApiError> {
    let inputs: Vec<String> = match (&req.input, &req.prompt) {
        (Some(Value::String(s)), _) => vec![s.clone()],
        (Some(Value::Array(a)), _) => a
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect(),
        (_, Some(p)) => vec![p.clone()],
        _ => Vec::new(),
    };
    if inputs.is_empty() {
        return Err(ApiError::bad_request("`input` or `prompt` is required"));
    }
    let (id, engine, load) = select(state, &req.model).await?;
    let (vectors, tokens) = tokio::task::spawn_blocking(move || engine.embed_with_usage(&inputs))
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?
        .map_err(ApiError::from)?;
    Ok((vectors, tokens, id, load))
}

/// `POST /api/embed`.
pub async fn embed(
    State(state): State<AppState>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<Value>, ApiError> {
    let started = Instant::now();
    let (embeddings, tokens, model, load) = embed_inputs(&state, &req).await?;
    Ok(Json(json!({
        "model": model,
        "embeddings": embeddings,
        "total_duration": started.elapsed().as_nanos() as u64,
        "load_duration": load.as_nanos() as u64,
        "prompt_eval_count": tokens,
    })))
}

/// `POST /api/embeddings` (legacy): one `prompt`, one `embedding`.
pub async fn embeddings(
    State(state): State<AppState>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<Value>, ApiError> {
    let (mut vectors, ..) = embed_inputs(&state, &req).await?;
    Ok(Json(json!({ "embedding": vectors.pop().unwrap_or_default() })))
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

/// The file stem a management request's model name stands for, or `None`
/// for a name that could reach outside the model directory (empty, a path,
/// or containing `..`).
fn bare_model_stem(name: &str) -> Option<&str> {
    let stem = strip_tag(name);
    let unsafe_name =
        stem.is_empty() || stem.contains('/') || stem.contains('\\') || stem.contains("..");
    (!unsafe_name).then_some(stem)
}

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
    let Some(stem) = bare_model_stem(name) else {
        return Err(ApiError::bad_request(&format!(
            "invalid model name `{name}`: expected a bare name inside the model directory"
        )));
    };
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
/// Neither source is implemented yet.  Ollama's registry protocol
/// (`model:tag` resolved against its library) needs the registry
/// manifest/blob API, which joshua does not speak, so a bare library name
/// fails with a 400 and guidance rather than hanging; an explicit https URL
/// is answered with a 501 until URL downloads land.  Either way the client
/// is told to place the file in the model directory and load it.
pub async fn pull(
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let name = body
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::bad_request("missing 'model'"))?;
    if !name.starts_with("https://") {
        return Err(ApiError::bad_request(&format!(
            "cannot pull `{name}`: joshua does not implement the Ollama registry protocol. \
             Place the .gguf file in the model directory and load it with \
             POST /v1/models/load."
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
    fn details_name_the_family_and_ollama_quantization() {
        let facts = GgufFacts {
            architecture: Some("qwen3".to_string()),
            file_type: Some(15),
            ..GgufFacts::default()
        };
        let d = details_for(Some(&facts));
        assert_eq!(d.family, "qwen3");
        assert_eq!(d.families, ["qwen3"]);
        assert_eq!(d.quantization_level, "Q4_K_M");
        assert_eq!(details_for(None).format, "gguf");
    }

    #[test]
    fn format_is_json_or_a_schema() {
        assert_eq!(constraint(None).unwrap(), None);
        assert_eq!(constraint(Some(&json!(""))).unwrap(), None);
        assert_eq!(constraint(Some(&json!("json"))).unwrap(), Some(ConstraintSpec::JsonObject));
        let schema = json!({"type": "object", "properties": {"a": {"type": "integer"}}});
        assert_eq!(
            constraint(Some(&schema)).unwrap(),
            Some(ConstraintSpec::JsonSchema { schema: schema.clone() })
        );
        assert!(constraint(Some(&json!("xml"))).is_err());
        assert!(constraint(Some(&json!(3))).is_err());
    }

    #[test]
    fn keep_alive_zero_in_any_spelling_unloads() {
        for zero in [json!(0), json!(0.0), json!("0"), json!("0s"), json!("0m")] {
            assert!(unload_after(Some(&zero)), "{zero}");
        }
        for keep in [json!(-1), json!("5m"), json!("-1"), json!(300), json!(null)] {
            assert!(!unload_after(Some(&keep)), "{keep}");
        }
        assert!(!unload_after(None));
    }

    #[test]
    fn ollama_messages_become_engine_messages() {
        let m = chat_message(json!({
            "role": "user", "content": "what is this?", "images": ["iVBORw0KGgo="],
        }))
        .unwrap();
        assert_eq!(
            m.images.unwrap(),
            ["data:application/octet-stream;base64,iVBORw0KGgo="]
        );
        let m = chat_message(json!({"role": "tool", "content": "22", "tool_name": "weather"}))
            .unwrap();
        assert_eq!(m.name.as_deref(), Some("weather"));
        let calls = json!([{"function": {"name": "weather", "arguments": {"city": "Paris"}}}]);
        let m = chat_message(json!({"role": "assistant", "content": "", "tool_calls": calls}))
            .unwrap();
        assert_eq!(m.tool_calls.unwrap(), calls);
        let m = chat_message(json!({"role": "assistant", "content": "hi", "thinking": "hmm"}))
            .unwrap();
        assert_eq!(m.reasoning_content.as_deref(), Some("hmm"));
    }

    #[test]
    fn tool_calls_go_out_with_object_arguments() {
        let calls = [crate::tools::ParsedToolCall {
            name: "weather".into(),
            arguments: r#"{"city":"Paris"}"#.into(),
            id: Some("call_1".into()),
        }];
        let out = ollama_calls(&calls).unwrap();
        assert_eq!(
            out[0],
            json!({"id": "call_1", "function": {"index": 0, "name": "weather", "arguments": {"city": "Paris"}}})
        );
        assert!(ollama_calls(&[]).is_none());
    }

    #[test]
    fn error_bodies_reduce_to_their_message() {
        let openai = br#"{"error":{"message":"model 'x' not found","type":"model_not_found"}}"#;
        assert_eq!(error_message(openai), "model 'x' not found");
        assert_eq!(error_message(br#"{"error":"already"}"#), "already");
        assert_eq!(error_message(b"Failed to parse the request body\n"), "Failed to parse the request body");
    }

    #[test]
    fn parameter_sizes_read_as_ollama_writes_them() {
        assert_eq!(parameter_size(14_768_307_200), "14.8B");
        assert_eq!(parameter_size(596_049_920), "596.05M");
        assert_eq!(parameter_size(12_345), "12.35K");
        assert_eq!(parameter_size(0), "");
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
        let g = o
            .to_generation_options(&SamplingProfile::Ollama.defaults(), 4096)
            .unwrap();
        assert_eq!(g.max_tokens, 64);
        assert_eq!(g.temperature, 0.0);
        assert_eq!(g.top_k, 20);
        assert_eq!(g.repetition_penalty, 1.1);
        assert_eq!(g.stop_sequences, vec!["</s>".to_string()]);
        // Unset values come from Ollama's DefaultOptions.
        assert_eq!(g.top_p, 0.9);
        assert_eq!(g.min_p, 0.0);
        assert_eq!(g.context_window, None);
    }

    #[test]
    fn num_ctx_and_the_other_sampler_options_are_honoured() {
        let o = OllamaOptions {
            num_ctx: Some(1024),
            seed: Some(7),
            typical_p: Some(0.5),
            repeat_last_n: Some(32),
            presence_penalty: Some(0.25),
            frequency_penalty: Some(0.5),
            mirostat: Some(2),
            mirostat_tau: Some(4.0),
            mirostat_eta: Some(0.2),
            ..Default::default()
        };
        let defaults = SamplingProfile::Ollama.defaults();
        let g = o.to_generation_options(&defaults, 4096).unwrap();
        assert_eq!(g.context_window, Some(1024));
        assert_eq!(g.seed, Some(7));
        assert_eq!(g.typical_p, 0.5);
        assert_eq!(g.repeat_last_n, 32);
        assert_eq!(g.presence_penalty, 0.25);
        assert_eq!(g.frequency_penalty, 0.5);
        assert_eq!((g.mirostat, g.mirostat_tau, g.mirostat_eta), (2, 4.0, 0.2));
        // A client default larger than the loaded window is clamped to it,
        // and 0 leaves the window alone.
        let wide = OllamaOptions { num_ctx: Some(32_768), ..Default::default() };
        assert_eq!(wide.to_generation_options(&defaults, 4096).unwrap().context_window, Some(4096));
        let zero = OllamaOptions { num_ctx: Some(0), ..Default::default() };
        assert_eq!(zero.to_generation_options(&defaults, 4096).unwrap().context_window, None);
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
        for bad in ["", "../etc/passwd", "a/b", "..", "dir/../x", "/abs/path", "a\\b", ":tag"] {
            assert_eq!(bare_model_stem(bad), None, "`{bad}` must be rejected as a model name");
        }
        // The bare names clients actually send must pass.
        for (good, stem) in [
            ("llama3", "llama3"),
            ("llama3:13b", "llama3"),
            ("my-model_1", "my-model_1"),
            ("Qwen3-30B-A3B", "Qwen3-30B-A3B"),
        ] {
            assert_eq!(bare_model_stem(good), Some(stem), "`{good}` must be accepted");
        }
    }

    #[test]
    fn unlimited_predict_runs_to_the_end_of_sequence() {
        // -1 / -2 mean "unbounded" / "fill context" in Ollama, as does an
        // unset num_predict (DefaultOptions' -1); all leave the output to
        // the server's --max-output-tokens.
        let defaults = SamplingProfile::Ollama.defaults();
        for num_predict in [None, Some(-1), Some(-2)] {
            let o = OllamaOptions {
                num_predict,
                ..Default::default()
            };
            assert_eq!(
                o.to_generation_options(&defaults, 4096).unwrap().max_tokens,
                crate::types::MAX_TOKENS_UNLIMITED
            );
        }
    }
}
