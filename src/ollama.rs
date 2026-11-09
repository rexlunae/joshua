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
//! `/api/pull`, `/api/blobs`) works against the model *directory*, so
//! scripts that manage models through the API keep working:
//!
//! * `delete` / `copy` / `create` (the `from` form) act on `<name>.gguf` inside
//!   the model directory.  Names are confined to that directory — `..`,
//!   absolute paths and names that canonicalise outside it are rejected,
//!   because these endpoints delete and overwrite files.
//! * `pull` fetches from an Ollama registry or an https URL, `/api/blobs`
//!   keeps uploads in a small blob store, `create` with `files` places
//!   an uploaded GGUF as a model, and `create` with `from` and `adapters`
//!   keeps a LoRA adapter as a model on its base: see [`crate::ollama_pull`].
//!
//! Ollama's `:tag` suffix names a manifest, which has no joshua equivalent:
//! a name whose tag is not part of a model id resolves to the id without it,
//! so `llama3:latest` and `llama3:13b` both address `llama3.gguf`.
//!
//! **Generation**: `/api/chat` and `/api/generate` read `think` (`true`,
//! `false` or a gpt-oss level), which reaches the chat template as the
//! reasoning switches, and return the reasoning apart from the answer under
//! `thinking`; `logprobs` and `top_logprobs` report each answer token.
//! `/api/generate` also reads `system`, `raw` (the prompt as it is),
//! `suffix` (a fill-in-the-middle prompt) and `context` (the tokens an
//! earlier reply gave back, followed by the new turn).  `template` is
//! refused: Ollama's Go templates are not something joshua can run.
//!
//! **`keep_alive`**: `0` unloads the model once the request is answered (when
//! the server manages a model directory, so it can load it again); a request
//! with no messages or prompt only loads or unloads the model, as Ollama's
//! does.  Other values are accepted and change nothing: joshua keeps a model
//! resident until it is unloaded, so `/api/ps` reports no expiry.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
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

use crate::engine::{Completion, Engine, RawPrompt};
use crate::model_info::{GgufFacts, ModelInfo};
use crate::sampling::{SamplingDefaults, SamplingProfile};
use crate::server::{select_generation, ApiError, Selected, AppState, CancelOnDrop, ChatAnswer, ChatPiece};
use crate::types::{
    rfc3339, ChatMessage, ConstraintSpec, GenerationOptions, ReasoningEffort, SamplingParams, TokenLogprob, Tool,
};

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
            || state.models.adapter(id).is_some()
            || state.adapter_model(id).is_some()
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

/// [`select_generation`] for an Ollama name, loading the model (and the
/// adapter of an adapter model `create` made) if it is not loaded.  Also
/// returns how long that took (Ollama's `load_duration`).
async fn select(state: &AppState, name: &str) -> Result<(Selected, Duration), ApiError> {
    let start = Instant::now();
    let selected = select_generation(state, &model_id(state, name)).await?;
    Ok((selected, start.elapsed()))
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
/// and experts run there; its dense weights plus its expert cache when the
/// routed experts stay in host memory; none on the CPU.
fn size_vram(info: &ModelInfo) -> u64 {
    let Some(facts) = &info.facts else { return 0 };
    match &info.placement {
        Some(p) if p.dense != "cpu" && p.experts.as_deref().is_none_or(|e| e != "cpu") => {
            facts.file_size
        }
        Some(p) if p.dense != "cpu" => {
            facts.tensor_bytes.saturating_sub(facts.expert_bytes)
                + p.expert_cache_bytes.unwrap_or(0)
        }
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
    // An adapter model shows its base's template and parameters.
    let runs_on = info.parent.clone().unwrap_or_else(|| info.id.clone());
    let engine = state
        .models
        .list()
        .into_iter()
        .find(|m| m.id == runs_on)
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
    let path = || info.path.as_deref().map_or(info.id.clone(), |p| p.display().to_string());
    let modelfile = match &info.parent {
        Some(base) => format!("# Modelfile generated by joshua\nFROM {base}\nADAPTER {}\n", path()),
        None => format!("# Modelfile generated by joshua\nFROM {}\n", path()),
    };
    ShowResponse {
        license: String::new(),
        modelfile,
        parameters,
        template,
        details,
        model_info: model_info(info),
        capabilities: engine.map_or_else(
            || vec!["completion".to_string()],
            |e| {
                let mut caps = capabilities(e);
                // `embed` refuses an adapter model: it pools the base alone.
                if info.parent.is_some() {
                    caps.retain(|c| c != "embedding");
                }
                caps
            },
        ),
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
    // A template that renders tools reads its `tools` variable (Ollama's
    // own test); without a template, the ChatML fallback teaches them.
    if engine
        .chat_template_source()
        .is_none_or(|t| t.contains("tools"))
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
/// `message.content` (and reasoning under `message.thinking`),
/// `/api/generate` under `response` (and `thinking`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Wire {
    Chat,
    Generate,
}

/// What one reply object carries besides its text.
#[derive(Default)]
struct Extra {
    /// Reasoning text.
    thinking: String,
    /// Chat tool calls, in Ollama's shape.
    tool_calls: Option<Vec<Value>>,
    /// The log-probabilities of the tokens behind the text.
    logprobs: Option<Value>,
}

impl Wire {
    /// One object of the reply: `text`, and what `extra` carries.
    fn chunk(self, model: &str, text: &str, extra: Extra) -> Value {
        let mut out = json!({ "model": model, "created_at": now() });
        match self {
            Wire::Chat => {
                let mut message = json!({ "role": "assistant", "content": text });
                if !extra.thinking.is_empty() {
                    message["thinking"] = Value::from(extra.thinking);
                }
                if let Some(calls) = extra.tool_calls {
                    message["tool_calls"] = Value::Array(calls);
                }
                out["message"] = message;
            }
            Wire::Generate => {
                out["response"] = Value::from(text);
                if !extra.thinking.is_empty() {
                    out["thinking"] = Value::from(extra.thinking);
                }
            }
        }
        if let Some(logprobs) = extra.logprobs {
            out["logprobs"] = logprobs;
        }
        out["done"] = Value::Bool(false);
        out
    }

    /// The last object: `done`, why it ended, the token counts, the
    /// durations in nanoseconds and, for generate, the `context`.
    fn done(self, model: &str, text: &str, extra: Extra, end: &End<'_>) -> Value {
        let mut out = self.chunk(model, text, extra);
        out["done"] = Value::Bool(true);
        out["done_reason"] = Value::from(end.reason);
        out["total_duration"] = Value::from(end.started.elapsed().as_nanos() as u64);
        out["load_duration"] = Value::from(end.load.as_nanos() as u64);
        if let Some(c) = end.completion {
            let ns = |ms: f64| (ms * 1e6).round() as u64;
            out["prompt_eval_count"] = Value::from(c.usage.prompt_tokens);
            out["prompt_eval_duration"] = Value::from(ns(c.timings.prompt_ms));
            out["eval_count"] = Value::from(c.usage.completion_tokens);
            out["eval_duration"] = Value::from(ns(c.timings.predicted_ms));
        }
        if let Some(context) = end.context {
            out["context"] = Value::from(context);
        }
        out
    }
}

/// How a reply ended.
struct End<'a> {
    reason: &'static str,
    completion: Option<&'a Completion>,
    /// Generate's `context`: the prompt's tokens and the answer's.
    context: Option<&'a [u32]>,
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

/// Ollama's `think`: on or off, or a level (`"low"`, `"medium"`, `"high"`,
/// gpt-oss's), which turns it on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Think {
    on: bool,
    effort: Option<ReasoningEffort>,
}

impl Think {
    fn parse(value: Option<&Value>) -> Result<Option<Self>, String> {
        let effort = |effort| Some(Self { on: true, effort: Some(effort) });
        match value {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Bool(on)) => Ok(Some(Self { on: *on, effort: None })),
            Some(Value::String(level)) => match level.as_str() {
                "low" => Ok(effort(ReasoningEffort::Low)),
                "medium" => Ok(effort(ReasoningEffort::Medium)),
                "high" => Ok(effort(ReasoningEffort::High)),
                other => Err(format!(
                    "invalid think value {other:?}: expected true, false, \"low\", \"medium\" or \"high\""
                )),
            },
            Some(other) => Err(format!(
                "invalid think value {other}: expected true, false, \"low\", \"medium\" or \"high\""
            )),
        }
    }
}

/// Ollama's `logprobs` and `top_logprobs`: the count of alternatives to
/// report with each token, when log-probabilities are asked for.
fn logprob_count(logprobs: Option<bool>, top: Option<i64>) -> Result<Option<u32>, String> {
    let top = top
        .map(|n| crate::logprobs::count("top_logprobs", n))
        .transpose()?;
    match (logprobs.unwrap_or(false), top) {
        (true, top) => Ok(Some(top.unwrap_or(0))),
        (false, Some(n)) if n > 0 => Err("`top_logprobs` needs `logprobs: true`".to_string()),
        (false, _) => Ok(None),
    }
}

/// What a turn reads.
enum Source {
    /// A conversation, rendered by the model's chat template.
    Chat(Vec<ChatMessage>),
    /// Generate's `raw` prompt, or its `suffix` infill prompt: read as it
    /// is, with no reasoning split off the answer.
    Raw(RawPrompt),
    /// Generate's `prompt` and `suffix`, as the model's fill-in-the-middle
    /// prompt.
    Infill { prefix: String, suffix: String },
    /// Generate's `context` (the tokens an earlier reply gave back),
    /// followed by the new turn, rendered.
    Continue {
        context: Vec<u32>,
        messages: Vec<ChatMessage>,
    },
}

/// A request for `/api/chat` or `/api/generate`, translated.
struct Turn {
    wire: Wire,
    model: String,
    /// What to read; none only loads or unloads the model.
    source: Option<Source>,
    tools: Option<Vec<Tool>>,
    options: OllamaOptions,
    format: Option<Value>,
    keep_alive: Option<Value>,
    stream: bool,
    think: Option<Think>,
    /// Report log-probabilities, with this many alternatives.
    logprobs: Option<u32>,
}

/// Answer a [`Turn`]: load or unload only, when it has nothing to say, else
/// generate, streaming newline-delimited JSON or in one object.  With
/// `keep_alive: 0` the model is unloaded once the request is done with it,
/// however the request ends.
async fn answer(state: AppState, turn: Turn) -> Result<Response, ApiError> {
    let started = Instant::now();
    let unload = unload_after(turn.keep_alive.as_ref()) && state.manager.is_some();
    if turn.source.is_none() {
        let (id, reason, load) = if unload {
            (unload_named(&state, &turn.model).await?, "unload", Duration::ZERO)
        } else {
            let (selected, load) = select(&state, &turn.model).await?;
            (selected.model, "load", load)
        };
        let end = End { reason, completion: None, context: None, started, load };
        return Ok(Json(turn.wire.done(&id, "", Extra::default(), &end)).into_response());
    }

    let (selected, load) = select(&state, &turn.model).await?;
    let release = unload.then(|| (Arc::clone(&state), selected.model.clone()));
    let (result, streaming) =
        generate_turn(&state, turn, selected, started, load, release.clone()).await;
    // A stream unloads once its generation ends; every other exit here.
    if !streaming {
        if let Some((state, id)) = release {
            unload_quietly(state, id).await;
        }
    }
    result
}

/// Unload the model a request names with `keep_alive: 0` and nothing to
/// say: the loaded model the name resolves to, as generation would pick it.
/// A name for a directory model that is not loaded has nothing to unload.
async fn unload_named(state: &AppState, name: &str) -> Result<String, ApiError> {
    let id = model_id(state, name);
    if let Some(unloaded) = unload_adapter(state, &id).await? {
        return Ok(unloaded);
    }
    if state.adapter_model(&id).is_some() {
        return Ok(id);
    }
    match state.resolve(&id) {
        Ok((loaded, engine)) => {
            drop(engine);
            crate::server::unload(state, &loaded).await?;
            Ok(loaded)
        }
        Err(_) if state.unloaded().iter().any(|(listed, _)| *listed == id) => Ok(id),
        Err(e) => Err(e),
    }
}

/// Unload the loaded LoRA adapter named `name` (an adapter model's, which
/// leaves its base loaded); `None` when no adapter has that name or a
/// model's id is it.
async fn unload_adapter(state: &AppState, name: &str) -> Result<Option<String>, ApiError> {
    let held = crate::lora_api::RUNTIME_LOADS.lock().await;
    unload_adapter_held(state, name, &held).await
}

/// [`unload_adapter`] for a caller already holding the runtime-load lock,
/// so a check, a change on disk and this unload are one step to the
/// runtime routes.
async fn unload_adapter_held(
    state: &AppState,
    name: &str,
    _held: &tokio::sync::MutexGuard<'_, ()>,
) -> Result<Option<String>, ApiError> {
    let Some((engine, _)) = state.models.adapter(name) else {
        return Ok(None);
    };
    if crate::server::names_a_model(state, name).await? {
        return Ok(None);
    }
    match engine.unload_lora(name) {
        Ok(_) => Ok(Some(name.to_string())),
        // Another request let it go first.
        Err(_) if engine.lora_id(name).is_none() => Ok(Some(name.to_string())),
        Err(e) => Err(ApiError::from(e)),
    }
}

/// `keep_alive: 0`'s unload, once nothing holds the engine; a model that
/// another request already unloaded is not an error.
async fn unload_quietly(state: AppState, id: String) {
    match unload_adapter(&state, &id).await {
        Ok(Some(_)) => return,
        Ok(None) => {}
        Err(e) => {
            tracing::debug!(model = %id, "keep_alive unload: {}", e.into_response().status());
            return;
        }
    }
    if let Err(e) = crate::server::unload(&state, &id).await {
        tracing::debug!(model = %id, "keep_alive unload: {}", e.into_response().status());
    }
}

/// A turn's prompt, ready to generate from.
enum Ready {
    Chat(Vec<ChatMessage>),
    /// A prompt read without the template; `split` reads reasoning off the
    /// answer (a continued conversation's) or not (a raw or infill one).
    Raw { prompt: RawPrompt, split: bool },
}

/// Ollama's context shift for a continued conversation: when the earlier
/// tokens and a new turn of `turn` tokens leave less than `answer` tokens
/// of the `window` for the answer, the oldest earlier tokens are dropped
/// (keeping a leading `bos`) until that much is free.  A turn too long for
/// that is left for generation to refuse.
fn shift_context(
    context: Vec<u32>,
    turn: usize,
    window: usize,
    answer: usize,
    bos: Option<u32>,
) -> Vec<u32> {
    let budget = window.saturating_sub(answer.min(window / 4).max(1));
    if context.len() + turn <= budget || turn >= budget {
        return context;
    }
    let head = usize::from(bos.is_some() && context.first() == bos.as_ref());
    let tail = (budget - turn).saturating_sub(head).min(context.len() - head);
    let mut kept = context[..head].to_vec();
    kept.extend_from_slice(&context[context.len() - tail..]);
    kept
}

/// What a turn is generated from: its prompt and, when generate's reply
/// gives back its `context`, the prompt's tokens.
struct Plan {
    wire: Wire,
    ready: Ready,
    prompt_tokens: Option<Vec<u32>>,
    tools: Option<Vec<Tool>>,
    options: GenerationOptions,
}

impl Plan {
    /// Work out the prompt (on the blocking pool: it tokenizes).  A
    /// generate reply gives back its `context` unless its prompt was raw or
    /// an infill, or it carries images (which are not tokens).
    fn new(
        engine: &Engine,
        wire: Wire,
        source: Source,
        tools: Option<Vec<Tool>>,
        options: GenerationOptions,
    ) -> crate::Result<Self> {
        let context = wire == Wire::Generate;
        let (ready, prompt_tokens) = match source {
            Source::Chat(messages) => {
                let images = messages.iter().any(|m| m.images.as_ref().is_some_and(|i| !i.is_empty()));
                let tokens = if context && !images {
                    Some(engine.tokenize_chat(&messages, tools.as_deref(), &options)?)
                } else {
                    None
                };
                (Ready::Chat(messages), tokens)
            }
            Source::Raw(prompt) => (Ready::Raw { prompt, split: false }, None),
            Source::Infill { prefix, suffix } => {
                // Ollama answers a model without fill-in-the-middle tokens so.
                if engine.fim_tokens().missing().is_some() {
                    return Err(crate::JoshuaError::InvalidRequest(
                        "this model does not support insert: it has no fill-in-the-middle tokens"
                            .to_string(),
                    ));
                }
                let n_predict = match options.max_tokens {
                    crate::types::MAX_TOKENS_UNLIMITED => -1,
                    n => i64::from(n),
                };
                let tokens = engine.format_infill(&prefix, &suffix, &[], "", n_predict, false, true)?;
                (Ready::Raw { prompt: RawPrompt::tokens(tokens), split: false }, None)
            }
            Source::Continue { context, messages } => {
                let turn = engine.tokenize_chat_turn(&messages, &options)?;
                let window = options.context_window.unwrap_or(engine.n_ctx()) as usize;
                // Room for the answer: what was asked for, up to a quarter
                // of the window.
                let answer = match options.max_tokens {
                    crate::types::MAX_TOKENS_UNLIMITED => window / 4,
                    n => n as usize,
                };
                let mut tokens = shift_context(context, turn.len(), window, answer, engine.bos_token());
                tokens.extend(turn);
                let prompt = RawPrompt::tokens(tokens.clone());
                (Ready::Raw { prompt, split: true }, Some(tokens))
            }
        };
        Ok(Self { wire, ready, prompt_tokens, tools, options })
    }

    /// Whether the reasoning is read off the answer.
    fn splits(&self) -> bool {
        match self.ready {
            Ready::Chat(_) => true,
            Ready::Raw { split, .. } => split,
        }
    }

    /// The reply's `context`: the prompt's tokens and the answer's.
    fn context(&self, completion: &Completion) -> Option<Vec<u32>> {
        self.prompt_tokens.as_ref().map(|prompt| {
            let mut tokens = prompt.clone();
            tokens.extend_from_slice(&completion.token_ids);
            tokens
        })
    }

    /// Generate the whole answer.
    fn whole(&self, engine: &Engine, cancel: &AtomicBool) -> crate::Result<(Completion, ChatAnswer)> {
        let tools = self.tools.as_deref();
        match &self.ready {
            Ready::Chat(messages) => {
                let format = match tools {
                    Some(tools) => engine.tool_call_format_for_request(messages, Some(tools), &self.options)?,
                    None => engine.tool_call_format(),
                };
                let completion =
                    engine.complete_chat_cancellable_detailed(messages, tools, &self.options, cancel)?;
                let answer = crate::server::chat_answer(engine, &completion, true, tools, format, "call_");
                Ok((completion, answer))
            }
            Ready::Raw { prompt, split } => {
                let completion = engine.complete_prompt_cancellable(prompt, &self.options, cancel)?;
                let answer =
                    crate::server::chat_answer(engine, &completion, *split, None, engine.tool_call_format(), "call_");
                Ok((completion, answer))
            }
        }
    }

    /// Generate the answer piece by piece: `emit` gets reasoning (`true`)
    /// and answer text (`false`) as they are decoded.  Returns the
    /// completion and the tool calls read from it.
    fn stream(
        &self,
        engine: &Engine,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(bool, &str),
    ) -> crate::Result<(Completion, Vec<crate::tools::ParsedToolCall>)> {
        match &self.ready {
            Ready::Chat(messages) => crate::server::stream_chat_output(
                engine,
                messages,
                self.tools.as_deref(),
                &self.options,
                cancel,
                true,
                "call_",
                &mut |piece| match piece {
                    ChatPiece::Reasoning(text) => emit(true, text),
                    ChatPiece::Text(text) => emit(false, text),
                    // Calls go out whole, once the answer is read.
                    ChatPiece::CallStart { .. } | ChatPiece::CallArgs { .. } => {}
                },
            ),
            Ready::Raw { prompt, split: true } => engine
                .complete_prompt_streaming_split(prompt, &self.options, cancel, &mut |split| {
                    emit(true, &split.reasoning);
                    emit(false, &split.content);
                })
                .map(|completion| (completion, Vec::new())),
            Ready::Raw { prompt, split: false } => engine
                .complete_prompt_streaming(prompt, &self.options, cancel, &mut |text| emit(false, text))
                .map(|completion| (completion, Vec::new())),
        }
    }
}

/// Generate a [`Turn`] on `engine`.  Returns the response and whether it
/// streams; a stream unloads the model itself (`release`) once its
/// generation ends.  Every engine reference this holds is dropped by the
/// time a non-streaming response returns.
#[allow(clippy::too_many_arguments)]
async fn generate_turn(
    state: &AppState,
    turn: Turn,
    selected: Selected,
    started: Instant,
    load: Duration,
    release: Option<(AppState, String)>,
) -> (Result<Response, ApiError>, bool) {
    let id = selected.model.clone();
    let engine = Arc::clone(&selected.engine);
    let prepared = (|| {
        let mut options = turn
            .options
            .to_generation_options(&engine.sampling_defaults(SamplingProfile::Ollama), engine.n_ctx())
            .map_err(ApiError::bad_request)?;
        // An adapter model runs its adapter (Ollama has no per-request list).
        selected.apply(&mut options);
        options.constraint = constraint(turn.format.as_ref()).map_err(ApiError::bad_request)?;
        if let Some(think) = turn.think {
            // Ollama's answer to a model that cannot think being asked to.
            if think.on && *engine.reasoning_format() == crate::reasoning::ReasoningFormat::None {
                return Err(ApiError::bad_request(format!("\"{id}\" does not support thinking")));
            }
            options.chat_template_kwargs.extend(crate::types::reasoning_template_kwargs(
                think.effort,
                Some(think.on),
                None,
            ));
        }
        options.logprobs = turn.logprobs;
        if turn.stream && turn.logprobs.is_some() {
            options.logprob_sink = Some(Default::default());
        }
        // Answer a request the model cannot serve before a stream starts.
        engine.validate_options(&options).map_err(ApiError::from)?;
        Ok::<_, ApiError>(options)
    })();
    let options = match prepared {
        Ok(options) => options,
        Err(e) => return (Err(e), false),
    };
    let Turn { wire, source, tools, stream, .. } = turn;
    let Some(source) = source else {
        return (Err(ApiError::internal("a turn with nothing to read")), false);
    };
    let plan = {
        let engine = Arc::clone(&engine);
        tokio::task::spawn_blocking(move || Plan::new(&engine, wire, source, tools, options)).await
    };
    let plan = match plan {
        Ok(Ok(plan)) => plan,
        Ok(Err(e)) => return (Err(ApiError::from(e)), false),
        Err(e) => return (Err(ApiError::internal(e.to_string())), false),
    };
    if stream {
        (Ok(stream_turn(state, plan, id, engine, started, load, release)), true)
    } else {
        (whole_turn(state, plan, &id, engine, started, load).await, false)
    }
}

/// The log-probabilities of the answer tokens among `entries`, in Ollama's
/// shape (OpenAI chat's entries: `token`, `logprob`, `bytes`,
/// `top_logprobs`).
fn ollama_logprobs(engine: &Engine, entries: &[TokenLogprob], split: bool) -> Value {
    crate::logprobs::chat(engine, entries, split)["content"].take()
}

/// One whole answer, generated on the blocking pool, which holds the
/// engine until it is read.
async fn whole_turn(
    state: &AppState,
    plan: Plan,
    id: &str,
    engine: Arc<Engine>,
    started: Instant,
    load: Duration,
) -> Result<Response, ApiError> {
    let cancel = CancelOnDrop::new();
    let flag = cancel.flag();
    let generated = tokio::task::spawn_blocking(move || {
        let (completion, answer) = plan.whole(&engine, &flag)?;
        let logprobs = plan
            .options
            .logprobs
            .map(|_| ollama_logprobs(&engine, &completion.logprobs, plan.splits()));
        let context = plan.context(&completion);
        Ok::<_, crate::JoshuaError>((plan.wire, completion, answer, logprobs, context))
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?;
    let (wire, completion, answer, logprobs, context) = generated.map_err(ApiError::from)?;
    state.metrics.record_generation(&completion.usage);
    let end = End {
        reason: completion.stop_type.finish_reason(),
        completion: Some(&completion),
        context: context.as_deref(),
        started,
        load,
    };
    let extra = Extra {
        thinking: answer.reasoning.unwrap_or_default(),
        tool_calls: ollama_calls(&answer.calls),
        logprobs,
    };
    Ok(Json(wire.done(id, &answer.prose, extra, &end)).into_response())
}

/// A streamed answer: newline-delimited JSON from a blocking thread, which
/// unloads the model (`release`) once generation ends.  Each answer chunk
/// carries the log-probabilities of the tokens released with it, when
/// asked for; the last chunk carries the rest.
fn stream_turn(
    state: &AppState,
    plan: Plan,
    id: String,
    engine: Arc<Engine>,
    started: Instant,
    load: Duration,
    release: Option<(AppState, String)>,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Value>();
    let cancel = CancelOnDrop::new();
    let flag = cancel.flag();
    state.metrics.record_stream();
    let runtime = tokio::runtime::Handle::current();
    let metrics = Arc::clone(&state.metrics);
    tokio::task::spawn_blocking(move || {
        let wire = plan.wire;
        let split = plan.splits();
        let sink = plan.options.logprob_sink.clone();
        let send = |value: Value| {
            let _ = tx.send(value);
        };
        // The entries released so far, when any is an answer token's.
        let take_logprobs = || {
            let entries = sink.as_ref()?.take();
            entries
                .iter()
                .any(|e| !split || e.answer)
                .then(|| ollama_logprobs(&engine, &entries, split))
        };
        let result = plan.stream(&engine, &flag, &mut |thinking, text| {
            if text.is_empty() {
                return;
            }
            let extra = if thinking {
                Extra { thinking: text.to_string(), ..Default::default() }
            } else {
                Extra { logprobs: take_logprobs(), ..Default::default() }
            };
            send(wire.chunk(&id, if thinking { "" } else { text }, extra));
        });
        match result {
            Ok((completion, calls)) => {
                metrics.record_generation(&completion.usage);
                let tool_calls = ollama_calls(&calls);
                if tool_calls.is_some() {
                    send(wire.chunk(&id, "", Extra { tool_calls, ..Default::default() }));
                }
                let context = plan.context(&completion);
                let end = End {
                    reason: completion.stop_type.finish_reason(),
                    completion: Some(&completion),
                    context: context.as_deref(),
                    started,
                    load,
                };
                let extra = Extra { logprobs: sink.as_ref().map(|_| take_logprobs().unwrap_or_else(|| json!([]))), ..Default::default() };
                send(wire.done(&id, "", extra, &end));
            }
            Err(e) => send(json!({ "error": e.to_string() })),
        }
        drop(plan);
        drop(engine);
        if let Some((state, id)) = release {
            runtime.spawn(unload_quietly(state, id));
        }
    });

    ndjson(rx, cancel)
}

/// Ollama's streamed answer: each object sent on `rx` as one line of
/// newline-delimited JSON, the one framer every streaming Ollama route
/// uses.  `guard` lives as long as the response body, so a guard such as
/// [`CancelOnDrop`] stops the work behind the stream when the client
/// disconnects.
pub(crate) fn ndjson<G: Send + 'static>(
    rx: tokio::sync::mpsc::UnboundedReceiver<Value>,
    guard: G,
) -> Response {
    let stream = futures_util::stream::unfold((rx, guard), |(mut rx, guard)| async move {
        rx.recv().await.map(|line| {
            (Ok::<String, std::convert::Infallible>(format!("{line}\n")), (rx, guard))
        })
    });
    let mut response = axum::body::Body::from_stream(stream).into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/x-ndjson"),
    );
    response
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
    /// Think before answering: `true`, `false` or a level.
    #[serde(default)]
    pub think: Option<Value>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<i64>,
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
        source: (!messages.is_empty()).then_some(Source::Chat(messages)),
        tools: req.tools.filter(|t| !t.is_empty()),
        options: req.options,
        format: req.format,
        keep_alive: req.keep_alive,
        stream: req.stream.unwrap_or(true),
        think: Think::parse(req.think.as_ref()).map_err(ApiError::bad_request)?,
        logprobs: logprob_count(req.logprobs, req.top_logprobs).map_err(ApiError::bad_request)?,
    };
    answer(state, turn).await
}

// ─── /api/generate ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub model: String,
    #[serde(default)]
    pub prompt: String,
    /// Text after the answer: the prompt and it are read as a
    /// fill-in-the-middle prompt.
    #[serde(default)]
    pub suffix: Option<String>,
    /// The system message, put before the prompt.
    #[serde(default)]
    pub system: Option<String>,
    /// A Go template to render the prompt with, which joshua cannot run.
    #[serde(default)]
    pub template: Option<String>,
    /// Read the prompt as it is, without the chat template.
    #[serde(default)]
    pub raw: Option<bool>,
    /// The `context` an earlier reply gave back, to continue from.
    #[serde(default)]
    pub context: Option<Vec<u32>>,
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
    #[serde(default)]
    pub think: Option<Value>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<i64>,
}

impl GenerateRequest {
    /// What the request reads: the prompt as it is (`raw`), an infill
    /// prompt (`suffix`), the earlier `context` and the new turn, or the
    /// new turn (`system` and the prompt with its images) alone.  `None`
    /// when there is nothing to read.
    fn source(&self) -> Result<Option<Source>, ApiError> {
        if self.template.as_deref().is_some_and(|t| !t.is_empty()) {
            return Err(ApiError::bad_request(
                "`template` is not supported: prompts are rendered with the model's own chat template; use `raw` to send a prompt formatted by hand",
            ));
        }
        let images = self.images.as_deref().unwrap_or_default();
        let context = self.context.as_deref().unwrap_or_default();
        let suffix = self.suffix.as_deref().filter(|s| !s.is_empty());
        let raw = self.raw.unwrap_or(false);
        if self.prompt.is_empty() && images.is_empty() && suffix.is_none() {
            return Ok(None);
        }
        if !images.is_empty() && (raw || suffix.is_some() || !context.is_empty()) {
            return Err(ApiError::bad_request(
                "`images` cannot be combined with `raw`, `suffix` or `context`",
            ));
        }
        if let Some(suffix) = suffix {
            if raw || !context.is_empty() {
                return Err(ApiError::bad_request(
                    "`suffix` cannot be combined with `raw` or `context`",
                ));
            }
            return Ok(Some(Source::Infill {
                prefix: self.prompt.clone(),
                suffix: suffix.to_string(),
            }));
        }
        if raw {
            // As Ollama, `system` and `context` mean nothing to a raw prompt.
            return Ok(Some(Source::Raw(RawPrompt::text(self.prompt.clone()))));
        }
        let mut messages = Vec::new();
        if let Some(system) = self.system.as_deref().filter(|s| !s.is_empty()) {
            messages.push(ChatMessage::text("system", system));
        }
        let mut message = ChatMessage::text("user", &self.prompt);
        if !images.is_empty() {
            message.images = Some(images.iter().map(|i| image_url(i)).collect());
        }
        messages.push(message);
        if context.is_empty() {
            Ok(Some(Source::Chat(messages)))
        } else {
            Ok(Some(Source::Continue { context: context.to_vec(), messages }))
        }
    }
}

pub async fn generate(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> Result<Response, ApiError> {
    let think = Think::parse(req.think.as_ref()).map_err(ApiError::bad_request)?;
    let logprobs = logprob_count(req.logprobs, req.top_logprobs).map_err(ApiError::bad_request)?;
    let source = req.source()?;
    let turn = Turn {
        wire: Wire::Generate,
        model: req.model,
        source,
        tools: None,
        options: req.options,
        format: req.format,
        keep_alive: req.keep_alive,
        stream: req.stream.unwrap_or(true),
        think,
        logprobs,
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
    let (selected, load) = select(state, &req.model).await?;
    if selected.adapter.is_some() {
        return Err(ApiError::bad_request(format!(
            "\"{}\" is a LoRA adapter model, which does not embed",
            selected.model
        )));
    }
    let (id, engine) = (selected.model, selected.engine);
    let inputs = inputs.into_iter().map(RawPrompt::text).collect();
    let (vectors, tokens) = crate::server::embed_prompts(state, &engine, inputs, None).await?;
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
    // An adapter model `create` made, when no model file has its name.
    if !path.exists() {
        if let Some(created) = state.adapter_model(strip_tag(name)) {
            // Held until the files are gone, so a request cannot load the
            // adapter again in between.
            let held = crate::lora_api::RUNTIME_LOADS.lock().await;
            unload_adapter_held(&state, &created.name, &held).await?;
            let dir = model_dir_of(&state)?;
            crate::ollama_pull::delete_adapter(&dir, &created.name)
                .map_err(|e| ApiError::internal(format!("removing {}: {e}", created.dir.display())))?;
            return Ok(StatusCode::OK);
        }
    }
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
    if !from.exists() && state.adapter_model(strip_tag(source)).is_some() {
        let stem = bare_model_stem(dest).unwrap_or_default();
        if to.exists() || state.models.list().iter().any(|m| m.id == stem) {
            return Err(ApiError::bad_request(format!(
                "`{dest}` is a model; delete it before copying an adapter model to its name"
            )));
        }
        // One step to the runtime routes: no adapter of the destination's
        // name can be loaded between the check and the unload.
        let held = crate::lora_api::RUNTIME_LOADS.lock().await;
        refuse_runtime_adapter(&state, stem)?;
        let dir = model_dir_of(&state)?;
        let (from, to) = (strip_tag(source).to_string(), stem.to_string());
        tokio::task::spawn_blocking(move || crate::ollama_pull::copy_adapter(&dir, &from, &to))
            .await
            .map_err(|e| ApiError::internal(e.to_string()))?
            .map_err(|e| ApiError::internal(format!("copying `{source}`: {e}")))?;
        // A destination adapter already loaded holds its old weights; the
        // next request loads the copied files.
        unload_adapter_held(&state, stem, &held).await?;
        return Ok(StatusCode::OK);
    }
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
/// The `from` form copies an existing model under a new name (Ollama's
/// "derive a model from another").  The `files` form names a GGUF blob
/// uploaded with `POST /api/blobs/{digest}` and places it in the model
/// directory (see [`crate::ollama_pull`]).  Either answers with Ollama's
/// status lines, streamed unless `stream` is false.
pub async fn create(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<axum::response::Response, ApiError> {
    let name = body
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::bad_request("missing 'model'"))?;
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(true);
    let to = model_file(&state, name)?;
    if crate::ollama_pull::sets(&body, "adapters") {
        return create_adapter_model(&state, name, &to, &body, stream).await;
    }
    if body.get("files").is_some_and(|v| !v.is_null()) {
        let stem = to
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| ApiError::bad_request("invalid model name"))?
            .to_string();
        let lines = crate::ollama_pull::create_from_files(&state, &stem, &body)?;
        return Ok(crate::ollama_pull::status_response(lines, stream));
    }
    let source = body.get("from").and_then(|v| v.as_str()).ok_or_else(|| {
        ApiError::bad_request("missing 'from' or 'files': joshua cannot build a model from template parameters alone")
    })?;
    let from = model_file(&state, source)?;
    if !from.exists() {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            format!("model `{source}` not found"),
            "model_not_found",
        ));
    }
    if from != to {
        std::fs::copy(&from, &to)
            .map_err(|e| ApiError::internal(format!("writing {}: {e}", to.display())))?;
    }
    Ok(crate::ollama_pull::status_response(
        vec![
            serde_json::json!({ "status": "using existing layer" }),
            serde_json::json!({ "status": "writing manifest" }),
            serde_json::json!({ "status": "success" }),
        ],
        stream,
    ))
}

/// The model directory, or the refusal a server without one gives.
fn model_dir_of(state: &AppState) -> Result<PathBuf, ApiError> {
    state.manager.as_ref().map(|m| m.model_dir.clone()).ok_or_else(|| {
        ApiError::new(StatusCode::FORBIDDEN, "model management is disabled on this server", "management_disabled")
    })
}

/// `POST /api/create` with `adapters` (a Modelfile's `ADAPTER`): keep the
/// adapter as a model of its own name on the model `from` names, which it
/// runs on as an adapter-named model.  The base must be a model this server
/// serves; a `files` base is refused (create the base first, then the
/// adapter model `from` it).  A loaded adapter of the same name made this
/// way is let go, so the next request loads the new one.
/// Refuse to write the adapter model `stem` over a LoRA adapter of that
/// name that a runtime load (not an adapter model's first use) put on an
/// engine: the new files would sit behind an adapter they do not describe.
fn refuse_runtime_adapter(state: &AppState, stem: &str) -> Result<(), ApiError> {
    if let Some((engine, _)) = state.models.adapter(stem) {
        let ours = state.adapter_model(stem).is_some_and(|created| {
            engine
                .lora_adapters()
                .iter()
                .any(|a| a.name == stem && a.path == created.path)
        });
        if !ours {
            return Err(ApiError::bad_request(format!(
                "a LoRA adapter named `{stem}` is already loaded; unload it or give the adapter \
                 model another name"
            )));
        }
    }
    Ok(())
}

async fn create_adapter_model(
    state: &AppState,
    name: &str,
    to: &Path,
    body: &Value,
    stream: bool,
) -> Result<Response, ApiError> {
    if body.get("files").is_some_and(|v| !v.is_null()) {
        return Err(ApiError::bad_request(
            "`adapters` with `files` is not supported: create the base model from its files \
             first, then create the adapter model with `from` naming it",
        ));
    }
    let from = body.get("from").and_then(Value::as_str).unwrap_or_default();
    if from.is_empty() {
        return Err(ApiError::bad_request("`adapters` needs `from` naming the base model"));
    }
    let base = model_id(state, from);
    if !crate::server::names_a_model(state, &base).await? {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            format!("base model `{from}` not found"),
            "model_not_found",
        ));
    }
    let stem = bare_model_stem(name).unwrap_or_default().to_string();
    if to.exists() || crate::server::names_a_model(state, &stem).await? {
        return Err(ApiError::bad_request(format!(
            "`{name}` is a model; delete it first or give the adapter model another name"
        )));
    }
    // The check, the write and the unload are one step to the runtime
    // routes, which take the same lock.
    let held = crate::lora_api::RUNTIME_LOADS.lock().await;
    refuse_runtime_adapter(state, &stem)?;
    let lines = tokio::task::spawn_blocking({
        let (state, stem, base, body) = (Arc::clone(state), stem.clone(), base.clone(), body.clone());
        move || crate::ollama_pull::create_adapter(&state, &stem, &base, &body)
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))??;
    unload_adapter_held(state, &stem, &held).await?;
    Ok(crate::ollama_pull::status_response(lines, stream))
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
    fn vram_counts_what_runs_on_the_accelerator() {
        let facts = GgufFacts {
            tensor_bytes: 1_000,
            expert_bytes: 800,
            file_size: 1_100,
            ..GgufFacts::default()
        };
        let info = |dense: &str, experts: Option<&str>, cache: Option<u64>| ModelInfo {
            facts: Some(facts.clone()),
            placement: Some(crate::model_info::Placement {
                backend: dense.to_string(),
                dense: dense.to_string(),
                experts: experts.map(str::to_string),
                expert_cache_bytes: cache,
                npu: false,
                remote: false,
                workers: Vec::new(),
                stages: Vec::new(),
            }),
            ..ModelInfo::new("m", 0)
        };
        assert_eq!(size_vram(&info("cuda", None, None)), 1_100);
        assert_eq!(size_vram(&info("cuda", Some("cuda"), None)), 1_100);
        // Dense weights on the device, routed experts in host memory.
        assert_eq!(size_vram(&info("cuda", Some("cpu"), None)), 200);
        assert_eq!(size_vram(&info("cuda", Some("cpu"), Some(50))), 250);
        assert_eq!(size_vram(&info("cpu", Some("cpu"), None)), 0);
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

    #[test]
    fn a_full_context_drops_its_oldest_tokens() {
        let context: Vec<u32> = (0..10).collect();
        // Room to spare for a 4-token answer: nothing is dropped.
        assert_eq!(shift_context(context.clone(), 2, 16, 4, Some(0)), context);
        // Too little: the BOS stays, then the newest tokens, leaving the 4
        // the answer asks for.
        assert_eq!(shift_context(context.clone(), 5, 16, 4, Some(0)), vec![0, 4, 5, 6, 7, 8, 9]);
        // Without a leading BOS only the newest are kept.
        assert_eq!(shift_context(context.clone(), 6, 16, 4, None), (4..10).collect::<Vec<u32>>());
        // An answer asked to be longer gets a quarter of the window.
        assert_eq!(shift_context(context.clone(), 6, 16, 100, None), (4..10).collect::<Vec<u32>>());
        // A turn too long to leave room is left alone.
        assert_eq!(shift_context(context.clone(), 12, 16, 4, Some(0)), context);
    }
}
