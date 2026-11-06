//! OpenAI-compatible HTTP API server for Joshua.
//!
//! Implements the following endpoints:
//! - `GET  /health`                   — liveness check with a load summary
//! - `GET  /status`                   — model inventory and counters
//! - `GET  /metrics`                  — JSON counters (`/metrics/prometheus` for text)
//! - `GET  /props`                    — llama.cpp-compatible server properties
//! - `GET  /slots`                    — lightweight per-model slot records
//! - `GET  /load`                     — vLLM-compatible server-load metrics
//! - `POST /v1/messages/count_tokens`  — Anthropic token count without running the model
//! - `POST /tokenize`                 — tokenize with the loaded model
//!   (also `/v1/tokenize`, and `/api/tokens` for Ollama clients)
//! - `POST /v1/detokenize`            — ids back to text (vLLM / llama.cpp)
//! - `GET  /v1/models`                — list the models, with their limits
//!   (also `/models`; `GET /v1/models/{id}` for one, see
//!   [`crate::model_info`])
//! - `POST /v1/chat/completions`      — chat completion (stream or non-stream)
//! - `POST /v1/messages`              — Anthropic Messages API
//! - `POST /v1/completions`           — legacy text completion
//! - `POST /v1/embeddings`            — dense text embeddings
//! - `POST /v1/audio/transcriptions`  — Whisper speech-to-text (when a
//!   whisper model is configured)
//! - `GET  /v1/worker/info`           — worker identity and live load for a
//!   `joshua route` coordinator (see [`crate::coordinator`])
//! - `POST /v1/models/load`           — load a model from the controller's
//!   model directory, locally or onto its workers (when enabled)
//! - `POST /v1/models/unload`         — unload a model (when enabled)
//!
//! With `--model-dir` the directory is the model store: every model in it is
//! listed, and a request selects one by naming its id — [`select_model`] loads
//! it on first use, so a client need not call `POST /v1/models/load`.  An
//! explicit load is still how a file the directory does not list (added after
//! startup) is served, and how a model is pinned to a chosen id.
//!
//! Chat and text completions stop decoding, and embeddings stop between
//! input texts, when the HTTP request is dropped (the client, or a
//! coordinator proxying for it, disconnected), so a cancelled request
//! releases its concurrency permit promptly. Transcription admission waits
//! asynchronously; dropped requests cancel WAV ingest and Whisper decoding.
//! Candle mel/encoder calls are non-preemptible, with cancellation checks on
//! either side. The blocking job owns its permit until it actually exits.

use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use axum::{
    extract::{Extension, Request, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    routing::{delete, get, post},
    Json, Router,
};
use futures_util::stream;
use serde_json::json;
use tokio::net::TcpListener;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use uuid::Uuid;

use crate::anthropic::{self, MessagesRequest};
use crate::coordinator::{JobModel, ModelSlots, WorkerCapabilities, WorkerInfo, JOB_OWNER_HEADER, WORKER_PROTOCOL_VERSION};
use crate::engine::{Completion, Engine, EngineOptions};
use crate::error::JoshuaError;
use crate::model_info::{device_label, GgufFacts, ModelInfo, ModelLimits, Placement};
use crate::reasoning::{ReasoningOutput, Split};
use crate::sampling::SamplingProfile;
use crate::whisper::WhisperEngine;
use crate::tools::{ParsedToolCall, ToolCallFormat, ToolCallStream, ToolEvent};
use crate::types::{
    unix_time, AssistantMessage, ChatChoice, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, ChatMessage, DeltaContent, EmbeddingData, EmbeddingRequest,
    EmbeddingResponse, ErrorResponse, FunctionCallResult, GenerationOptions, ModelCard,
    ModelListResponse, Tool, ToolCall, UsageInfo,
};

/// Process-lifetime counters exposed by `/status` and `/metrics`.
///
/// The counters are advisory: they are updated with relaxed atomics on the
/// request path, where a lost update under extreme concurrency is less
/// important than keeping the hot path free of locks.
pub struct ServerMetrics {
    started: Instant,
    http_requests: AtomicU64,
    http_errors: AtomicU64,
    prompt_tokens: AtomicU64,
    completion_tokens: AtomicU64,
    embed_tokens: AtomicU64,
    streamed_requests: AtomicU64,
}

impl Default for ServerMetrics {
    fn default() -> Self {
        Self {
            started: Instant::now(),
            http_requests: AtomicU64::new(0),
            http_errors: AtomicU64::new(0),
            prompt_tokens: AtomicU64::new(0),
            completion_tokens: AtomicU64::new(0),
            embed_tokens: AtomicU64::new(0),
            streamed_requests: AtomicU64::new(0),
        }
    }
}

impl ServerMetrics {
    /// Count a streaming response (shared by the OpenAI and Ollama routes).
    pub(crate) fn record_stream(&self) {
        self.streamed_requests.fetch_add(1, Ordering::Relaxed);
    }

    fn record_http(&self, status: StatusCode) {
        self.http_requests.fetch_add(1, Ordering::Relaxed);
        if status.is_client_error() || status.is_server_error() {
            self.http_errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn record_generation(&self, usage: &UsageInfo) {
        self.prompt_tokens
            .fetch_add(usage.prompt_tokens as u64, Ordering::Relaxed);
        self.completion_tokens
            .fetch_add(usage.completion_tokens as u64, Ordering::Relaxed);
    }

    fn snapshot(&self) -> serde_json::Value {
        json!({
            "uptime_seconds": self.started.elapsed().as_secs(),
            "http_requests_total": self.http_requests.load(Ordering::Relaxed),
            "http_errors_total": self.http_errors.load(Ordering::Relaxed),
            "prompt_tokens_total": self.prompt_tokens.load(Ordering::Relaxed),
            "completion_tokens_total": self.completion_tokens.load(Ordering::Relaxed),
            "embedding_tokens_total": self.embed_tokens.load(Ordering::Relaxed),
            "streaming_requests_total": self.streamed_requests.load(Ordering::Relaxed),
        })
    }

    fn prometheus(&self, in_flight: usize, max_concurrency: usize, models: usize) -> String {
        let mut out = String::new();
        let uptime = self.started.elapsed().as_secs();
        out.push_str("# HELP joshua_uptime_seconds Seconds since the server started.\n");
        out.push_str("# TYPE joshua_uptime_seconds gauge\n");
        out.push_str(&format!("joshua_uptime_seconds {uptime}\n"));
        out.push_str("# HELP joshua_http_requests_total HTTP requests received.\n");
        out.push_str("# TYPE joshua_http_requests_total counter\n");
        out.push_str(&format!(
            "joshua_http_requests_total {}\n",
            self.http_requests.load(Ordering::Relaxed)
        ));
        out.push_str("# HELP joshua_http_errors_total HTTP requests answered 4xx/5xx.\n");
        out.push_str("# TYPE joshua_http_errors_total counter\n");
        out.push_str(&format!(
            "joshua_http_errors_total {}\n",
            self.http_errors.load(Ordering::Relaxed)
        ));
        out.push_str("# HELP joshua_prompt_tokens_total Prompt tokens processed.\n");
        out.push_str("# TYPE joshua_prompt_tokens_total counter\n");
        out.push_str(&format!(
            "joshua_prompt_tokens_total {}\n",
            self.prompt_tokens.load(Ordering::Relaxed)
        ));
        out.push_str("# HELP joshua_completion_tokens_total Tokens generated.\n");
        out.push_str("# TYPE joshua_completion_tokens_total counter\n");
        out.push_str(&format!(
            "joshua_completion_tokens_total {}\n",
            self.completion_tokens.load(Ordering::Relaxed)
        ));
        out.push_str("# HELP joshua_embedding_tokens_total Tokens embedded.\n");
        out.push_str("# TYPE joshua_embedding_tokens_total counter\n");
        out.push_str(&format!(
            "joshua_embedding_tokens_total {}\n",
            self.embed_tokens.load(Ordering::Relaxed)
        ));
        out.push_str("# HELP joshua_streaming_requests_total Requests served over SSE.\n");
        out.push_str("# TYPE joshua_streaming_requests_total counter\n");
        out.push_str(&format!(
            "joshua_streaming_requests_total {}\n",
            self.streamed_requests.load(Ordering::Relaxed)
        ));
        out.push_str("# HELP joshua_in_flight_requests Requests holding a generation permit.\n");
        out.push_str("# TYPE joshua_in_flight_requests gauge\n");
        out.push_str(&format!("joshua_in_flight_requests {in_flight}\n"));
        out.push_str("# HELP joshua_max_concurrency Configured admission cap.\n");
        out.push_str("# TYPE joshua_max_concurrency gauge\n");
        out.push_str(&format!("joshua_max_concurrency {max_concurrency}\n"));
        out.push_str("# HELP joshua_models_loaded Models currently loaded.\n");
        out.push_str("# TYPE joshua_models_loaded gauge\n");
        out.push_str(&format!("joshua_models_loaded {models}\n"));
        out
    }
}

/// Shared application state.
pub struct ServerState {
    /// Loaded chat/embedding models.
    pub models: ModelRegistry,
    /// Enables `/v1/models/load` and `/v1/models/unload` when set.
    pub manager: Option<ModelManager>,
    /// Optional Whisper model for `/v1/audio/transcriptions`.
    pub whisper: Option<Arc<WhisperEngine>>,
    /// When set, every `/v1` request must carry this key as
    /// `Authorization: Bearer <key>` or `x-api-key: <key>`.  A
    /// comma-separated list accepts any of the listed keys, which makes
    /// rotation possible without a restart.  Empty entries are ignored; a
    /// value with no non-empty keys rejects every request as a configuration
    /// error.  `/health` stays open for probes.
    pub api_key: Option<String>,
    /// Exact origins allowed by CORS.  Empty keeps the historical
    /// permissive policy for localhost development; set one or more
    /// origins (or `JOSHUA_CORS_ORIGINS`) in a deployment.
    pub cors_origins: Vec<String>,
    /// Ops counters.
    pub metrics: Arc<ServerMetrics>,
}

/// Shared application state handle.
pub type AppState = Arc<ServerState>;

impl ServerState {
    /// State serving `engine` alone, without model management.
    pub fn new(engine: Arc<Engine>) -> Self {
        Self {
            models: ModelRegistry::with_model(engine),
            manager: None,
            whisper: None,
            api_key: None,
            cors_origins: Vec::new(),
            metrics: Arc::new(ServerMetrics::default()),
        }
    }

    /// Every model this server can serve: the loaded ones, then (with
    /// `--model-dir`) the directory's files that are not loaded
    /// ([`ModelManager::unloaded`]), described from their headers alone,
    /// less the GGUFs that are not models on their own (adapters,
    /// projectors).  This is the inventory `/v1/models` and Ollama's
    /// `/api/tags` list.  It reads every such file's header, so handlers
    /// build it off the async runtime ([`inventory`]).
    pub fn inventory(&self) -> Vec<ModelInfo> {
        let loaded = self.models.list();
        let mut out: Vec<ModelInfo> = loaded.iter().map(LoadedModel::info).collect();
        if let Some(manager) = &self.manager {
            out.extend(
                manager
                    .unloaded(&loaded)
                    .iter()
                    .map(|(id, file)| ModelInfo::of_file(id, file, manager))
                    .filter(|model| model.facts.as_ref().is_none_or(GgufFacts::is_model)),
            );
        }
        out
    }

    /// The model directory's files that are not loaded, under the ids the
    /// inventory lists them by ([`ModelManager::unloaded`]).  Reads the
    /// directory only, never a header.
    fn unloaded(&self) -> Vec<(String, PathBuf)> {
        self.manager
            .as_ref()
            .map(|manager| manager.unloaded(&self.models.list()))
            .unwrap_or_default()
    }

    /// The model a request names, by id (see [`ModelRegistry::resolve`]).
    /// The id of a model-directory file that is not loaded is a `404`,
    /// never answered by the one model that is.  That is checked against
    /// the directory alone ([`Self::unloaded`]), so a request never waits
    /// on a header read.
    pub fn resolve(&self, requested: &str) -> Result<(String, Arc<Engine>), ApiError> {
        self.models.resolve(requested, |name| {
            self.unloaded().iter().any(|(id, _)| id == name)
        })
    }

    /// The Whisper model's entry, when one is configured.
    fn whisper_info(&self) -> Option<ModelInfo> {
        self.whisper.as_ref().map(|whisper| ModelInfo {
            placement: Some(Placement::on(whisper.device())),
            ..ModelInfo::new(whisper.model_name(), unix_time())
        })
    }
}

/// [`ServerState::inventory`], built off the async runtime: a file in the
/// model directory is described by reading its header.
pub(crate) async fn inventory(state: &AppState) -> Result<Vec<ModelInfo>, ApiError> {
    let state = Arc::clone(state);
    tokio::task::spawn_blocking(move || state.inventory())
        .await
        .map_err(|e| ApiError::internal(e.to_string()))
}

// ─── Model registry ─────────────────────────────────────────────────────────

/// A loaded model and where it runs.
#[derive(Clone)]
pub struct LoadedModel {
    pub embedding_job: JobModel,
    pub id: String,
    pub engine: Arc<Engine>,
    /// Workers holding its pipeline stages; empty for a local model.
    pub workers: Vec<std::net::SocketAddr>,
    /// Each worker's layer range, in worker order.
    pub stages: Vec<(usize, usize)>,
    pub created: u64,
}

impl LoadedModel {
    /// What `/v1/models` reports about this model.
    pub fn info(&self) -> ModelInfo {
        let mut info = ModelInfo::of_engine(&self.id, self.created, &self.engine);
        if let Some(placement) = &mut info.placement {
            placement.workers = self.workers.iter().map(|w| w.to_string()).collect();
            placement.stages = self.stages.clone();
        }
        info
    }
}

/// The models a server answers for, by id.
#[derive(Default)]
pub struct ModelRegistry {
    models: RwLock<Vec<LoadedModel>>,
    /// Ids being loaded, so concurrent loads of one id cannot both run.
    loading: Mutex<Vec<String>>,
    /// Gates serialising request-initiated loads of one id: requests that name
    /// a model the directory lists wait for its one load instead of failing.
    /// An entry is created the first time a request names that id.
    load_gates: Mutex<std::collections::HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    /// Workers claimed by a model being loaded, loaded, or being unloaded.
    /// A worker serves one controller connection at a time, so a claim
    /// lasts until the model's pipeline has released it.
    workers: Mutex<Vec<std::net::SocketAddr>>,
}

impl ModelRegistry {
    /// A registry holding `engine` under its model name.
    pub fn with_model(engine: Arc<Engine>) -> Self {
        let registry = Self::default();
        registry.models.write().expect("fresh lock").push(LoadedModel {
            embedding_job: embedding_job(&engine, engine.model_name()),
            id: engine.model_name().to_string(),
            engine,
            workers: Vec::new(),
            stages: Vec::new(),
            created: unix_time(),
        });
        registry
    }

    pub fn list(&self) -> Vec<LoadedModel> {
        self.models.read().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// The model a request names. With a single model loaded, any name
    /// selects it, as when a server only ever held one model, unless
    /// `names_another` says the name belongs to a model that is not loaded.
    /// Returns the model's id and engine.
    pub fn resolve(
        &self,
        requested: &str,
        names_another: impl FnOnce(&str) -> bool,
    ) -> Result<(String, Arc<Engine>), ApiError> {
        let only = {
            let models = self.models.read().unwrap_or_else(|p| p.into_inner());
            if let Some(model) = models.iter().find(|m| m.id == requested) {
                return Ok((model.id.clone(), Arc::clone(&model.engine)));
            }
            match models.as_slice() {
                [only] => (only.id.clone(), Arc::clone(&only.engine)),
                [] => {
                    return Err(ApiError::new(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "no model is loaded; load one with POST /v1/models/load",
                        "model_not_loaded",
                    ))
                }
                _ => {
                    return Err(ApiError::model_not_found(format!(
                        "model '{requested}' is not loaded"
                    )))
                }
            }
        };
        // Asked after the lock is released: the answer may list the models.
        if names_another(requested) {
            return Err(ApiError::model_not_found(format!(
                "model '{requested}' is not loaded; load it with POST /v1/models/load"
            )));
        }
        Ok(only)
    }

    /// The loaded model with exactly this id, if it is loaded.
    fn loaded(&self, id: &str) -> Option<(String, Arc<Engine>)> {
        let models = self.models.read().unwrap_or_else(|p| p.into_inner());
        models
            .iter()
            .find(|m| m.id == id)
            .map(|m| (m.id.clone(), Arc::clone(&m.engine)))
    }

    /// The gate serialising request-initiated loads of `id`.
    fn load_gate(&self, id: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut gates = self.load_gates.lock().unwrap_or_else(|p| p.into_inner());
        Arc::clone(gates.entry(id.to_string()).or_default())
    }

    /// Reserve `id` for a load; dropping the guard releases it.
    fn reserve(&self, id: &str) -> Result<LoadGuard<'_>, ApiError> {
        let mut loading = self.loading.lock().unwrap_or_else(|p| p.into_inner());
        let loaded = self
            .models
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .any(|m| m.id == id);
        if loaded || loading.iter().any(|l| l == id) {
            return Err(ApiError::new(
                StatusCode::CONFLICT,
                format!("model '{id}' is already loaded or loading"),
                "model_exists",
            ));
        }
        loading.push(id.to_string());
        Ok(LoadGuard {
            registry: self,
            id: id.to_string(),
        })
    }

    /// Remove a model. Requests already running on it finish first; its
    /// memory (and, for a pipeline, its workers) is released after them.
    pub fn remove(&self, id: &str) -> Option<LoadedModel> {
        let mut models = self.models.write().unwrap_or_else(|p| p.into_inner());
        let index = models.iter().position(|m| m.id == id)?;
        Some(models.remove(index))
    }

    /// Claim `workers` for a load, failing if any is claimed already.
    /// Dropping the claim releases them unless it is kept.
    #[cfg(feature = "distributed")]
    fn claim_workers(&self, workers: &[std::net::SocketAddr]) -> crate::Result<WorkerClaim<'_>> {
        let mut claimed = self.workers.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(w) = workers.iter().find(|w| claimed.contains(w)) {
            return Err(crate::JoshuaError::InvalidRequest(format!(
                "worker {w} is in use by another model; unload it first"
            )));
        }
        claimed.extend_from_slice(workers);
        Ok(WorkerClaim {
            registry: self,
            workers: workers.to_vec(),
        })
    }

    /// Release workers a fully unloaded model held.
    fn release_workers(&self, workers: &[std::net::SocketAddr]) {
        self.workers
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .retain(|w| !workers.contains(w));
    }
}

#[cfg(feature = "distributed")]
struct WorkerClaim<'a> {
    registry: &'a ModelRegistry,
    workers: Vec<std::net::SocketAddr>,
}

#[cfg(feature = "distributed")]
impl WorkerClaim<'_> {
    /// Keep the workers claimed; unloading the model releases them.
    fn keep(mut self) {
        self.workers.clear();
    }
}

#[cfg(feature = "distributed")]
impl Drop for WorkerClaim<'_> {
    fn drop(&mut self) {
        self.registry.release_workers(&self.workers);
    }
}

struct LoadGuard<'a> {
    registry: &'a ModelRegistry,
    id: String,
}

impl LoadGuard<'_> {
    fn finish(self, model: LoadedModel) {
        self.registry
            .models
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .push(model);
    }
}

impl Drop for LoadGuard<'_> {
    fn drop(&mut self) {
        self.registry
            .loading
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .retain(|l| *l != self.id);
    }
}

/// What `/v1/models/load` may load and how.
pub struct ModelManager {
    /// Load requests name model files relative to this directory and cannot
    /// leave it.
    pub model_dir: PathBuf,
    /// Settings for a model run on this host, given its file (so defaults
    /// can depend on its size).
    pub engine_options: Arc<dyn Fn(&Path) -> EngineOptions + Send + Sync>,
    /// NPU backend attached to models run on this host, as at startup.
    pub npu_backend: Option<Arc<dyn crate::npu::NpuBackend>>,
    pub max_concurrency: Option<usize>,
    pub max_output_tokens: Option<u32>,
    /// Workers started without a model that this controller may place
    /// models on.
    #[cfg(feature = "distributed")]
    pub workers: Vec<std::net::SocketAddr>,
    /// Shared key authenticating this controller to its workers.
    #[cfg(feature = "distributed")]
    pub cluster_key: Option<Vec<u8>>,
}

/// `POST /v1/models/load` request body.
#[derive(Debug, Default, serde::Deserialize)]
pub struct LoadModelRequest {
    /// GGUF file (or a directory holding one) under the model directory,
    /// or the id `/v1/models` lists a file there by.
    pub model: String,
    /// Name to serve it under; defaults to the id `/v1/models` lists the
    /// file by (its file stem unless that is ambiguous).  Another model
    /// file's path in the directory (`a/model.gguf`) is refused.
    #[serde(default)]
    pub id: Option<String>,
    /// Run the model on this host even when workers are configured.
    #[serde(default)]
    pub local: bool,
    /// Configured workers to place the model on; defaults to all of them.
    #[serde(default)]
    pub workers: Option<Vec<std::net::SocketAddr>>,
    /// Exclusive layer ends per worker; defaults to splitting by free memory.
    #[serde(default)]
    pub ends: Option<Vec<usize>>,
    /// Context window; defaults to the server's, capped by the model.
    #[serde(default)]
    pub n_ctx: Option<usize>,
    /// Concurrent remote sessions the workers reserve KV for; defaults to one.
    #[serde(default)]
    pub sessions: Option<usize>,
    /// Prompt tokens per pipeline step.
    #[serde(default)]
    pub chunk: Option<usize>,
}

/// `POST /v1/models/unload` request body.
#[derive(Debug, serde::Deserialize)]
pub struct UnloadModelRequest {
    pub id: String,
}

impl ModelManager {
    /// The directory's GGUFs (two levels deep) that no model in `loaded`
    /// was loaded from, each under the id the inventory lists it by: its
    /// file stem or, when another file in the directory has the same stem
    /// (or a loaded model has it as its id), its path in the directory
    /// ([`Self::path_id`], `b/model.gguf`).  So loading one file does not
    /// change another's id, unless the load takes that file's stem as its
    /// id.  `POST /v1/models/load` takes either as `model`, and refuses
    /// another file's path as an `id`.  Left out: files a load would refuse (a symlink leading out of the
    /// directory), a second link to one file, and a file whose path a
    /// loaded model has as its id (it was loaded before the file appeared):
    /// no two entries share an id, so each id answers for its own entry.
    /// Only the directory is read, never a header: a request's model name
    /// is checked against these ids, and the inventory, off the async
    /// runtime, leaves out the GGUFs that are not models on their own.
    pub fn unloaded(&self, loaded: &[LoadedModel]) -> Vec<(String, PathBuf)> {
        let Ok(root) = self.model_dir.canonicalize() else {
            return Vec::new();
        };
        let mut targets = std::collections::HashSet::new();
        let files: Vec<(PathBuf, PathBuf)> = crate::model_info::discover_gguf(&self.model_dir)
            .into_iter()
            .filter_map(|file| {
                let target = file.canonicalize().ok()?;
                (target.starts_with(&root) && targets.insert(target.clone()))
                    .then_some((file, target))
            })
            .collect();
        let stem = |file: &Path| -> String {
            file.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model")
                .to_string()
        };
        let mut stems = std::collections::HashMap::<String, usize>::new();
        for (file, _) in &files {
            *stems.entry(stem(file)).or_default() += 1;
        }
        let loaded_from: Vec<PathBuf> = loaded
            .iter()
            .filter_map(|m| m.engine.model_path().canonicalize().ok())
            .collect();
        let taken = |id: &str| loaded.iter().any(|m| m.id == id);
        files
            .into_iter()
            .filter(|(_, target)| !loaded_from.contains(target))
            .filter_map(|(file, _)| {
                let stem = stem(&file);
                let id = if stems[&stem] > 1 || taken(&stem) {
                    self.path_id(&file)
                } else {
                    stem
                };
                (!taken(&id)).then_some((id, file))
            })
            .collect()
    }

    /// `file`'s path in the model directory, as an id: `b/model.gguf`.
    fn path_id(&self, file: &Path) -> String {
        let relative = file.strip_prefix(&self.model_dir).unwrap_or(file);
        let parts: Vec<_> = relative.iter().map(|part| part.to_string_lossy()).collect();
        parts.join("/")
    }

    /// Whether `id` is the path ([`Self::path_id`]) of a model the directory
    /// lists other than `target` (a canonical path).  A file is listed by its
    /// path once its stem is taken, so a load cannot take another listed
    /// file's path as its id: that file's id would answer with this model.
    /// Files the inventory leaves out (a link leading out of the directory,
    /// a GGUF that is not a model on its own) reserve nothing.  Reads the
    /// header of a file whose path is `id`, so call it off the async runtime.
    fn names_another_file(&self, id: &str, target: Option<&Path>) -> bool {
        let Ok(root) = self.model_dir.canonicalize() else {
            return false;
        };
        crate::model_info::discover_gguf(&self.model_dir)
            .iter()
            .filter(|file| self.path_id(file) == id)
            .any(|file| {
                file.canonicalize()
                    .is_ok_and(|real| real.starts_with(&root) && Some(real.as_path()) != target)
                    && GgufFacts::read(file)
                        .ok()
                        .is_none_or(|facts| facts.is_model())
            })
    }

    /// Resolve a requested model inside `model_dir`.
    fn resolve_path(&self, requested: &str) -> Result<PathBuf, ApiError> {
        let not_found = || {
            ApiError::model_not_found(format!(
                "model '{requested}' was not found in the model directory"
            ))
        };
        let root = self
            .model_dir
            .canonicalize()
            .map_err(|e| ApiError::internal(format!("model directory: {e}")))?;
        let inside = |path: &Path| -> Result<PathBuf, ApiError> {
            // Resolve symlinks first, so none can lead outside the root.
            let path = path.canonicalize().map_err(|_| not_found())?;
            if path.starts_with(&root) {
                Ok(path)
            } else {
                Err(not_found())
            }
        };
        let path = inside(&root.join(requested))?;
        if path.is_dir() {
            // The file picked inside a directory may itself be a symlink:
            // its target must stay inside the root, but the engine gets the
            // link so it finds the tokenizer beside it.
            let file = crate::engine::find_gguf_in_dir(&path).map_err(|_| not_found())?;
            inside(&file)?;
            Ok(file)
        } else {
            Ok(path)
        }
    }

    /// Load the model described by `req` (blocking).
    /// Workers it places the model on are claimed in `registry`.
    fn load(
        &self,
        req: &LoadModelRequest,
        path: &Path,
        id: String,
        #[cfg_attr(not(feature = "distributed"), allow(unused_variables))] registry: &ModelRegistry,
    ) -> crate::Result<LoadedModel> {
        let n_ctx = req
            .n_ctx
            .map(|n_ctx| {
                u32::try_from(n_ctx)
                    .map_err(|_| crate::JoshuaError::InvalidRequest("n_ctx too large".into()))
            })
            .transpose()?;
        #[cfg(feature = "distributed")]
        if self.on_workers(req) {
            return self.load_distributed(req, n_ctx, path, id, registry);
        }
        if req.workers.is_some() || req.ends.is_some() {
            return Err(crate::JoshuaError::InvalidRequest(
                "this server has no workers configured; omit 'workers' and 'ends'".into(),
            ));
        }
        let mut options = (self.engine_options)(path);
        options.n_ctx = Self::load_context(&options, n_ctx, None, false);
        let mut engine = Engine::with_options(path, options)?;
        if let Some(backend) = &self.npu_backend {
            engine = engine.with_npu_backend(Arc::clone(backend));
        }
        let engine = self.finish_engine(engine);
        Ok(LoadedModel {
            embedding_job: embedding_job(&engine, &id),
            id,
            engine: Arc::new(engine),
            workers: Vec::new(),
            stages: Vec::new(),
            created: unix_time(),
        })
    }

    /// Whether `req` places the model on workers rather than on this host.
    #[cfg(feature = "distributed")]
    fn on_workers(&self, req: &LoadModelRequest) -> bool {
        !req.local && (req.workers.is_some() || !self.workers.is_empty())
    }

    /// Whether `req` places the model on workers rather than on this host.
    #[cfg(not(feature = "distributed"))]
    fn on_workers(&self, _req: &LoadModelRequest) -> bool {
        false
    }

    /// The context window a load gives a model trained with `n_ctx_train`:
    /// the load's own `n_ctx`, else the server's `--n-ctx` in `options`;
    /// placed on workers, no more than `n_ctx_train`.  Every load, and
    /// [`Self::default_limits`], sizes the context here.
    fn load_context(
        options: &EngineOptions,
        n_ctx: Option<u32>,
        n_ctx_train: Option<u32>,
        on_workers: bool,
    ) -> u32 {
        let context = crate::engine::resolve_n_ctx(n_ctx.unwrap_or(options.n_ctx));
        match n_ctx_train {
            Some(trained) if on_workers => context.min(trained),
            _ => context,
        }
    }

    fn finish_engine(&self, mut engine: Engine) -> Engine {
        if let Some(max) = self.max_concurrency {
            engine = engine.with_max_concurrency(max);
        }
        if let Some(max) = self.max_output_tokens {
            engine = engine.with_max_output_tokens(max);
        }
        engine
    }

    /// The limits a load of `path` that names nothing else gives the
    /// model, as [`Self::load`] sets them: the context from
    /// [`Self::load_context`] and the output cap from
    /// `--max-output-tokens` ([`crate::engine::output_cap`]).
    pub fn default_limits(&self, path: &Path, facts: Option<&GgufFacts>) -> ModelLimits {
        let n_ctx_train = facts.and_then(|f| f.n_ctx_train);
        // Only the context is read here.  The options also settle (and log)
        // the file's page-cache plan, which is a load's business, not a
        // listing's: keep that out of the log on every `/v1/models`.
        let options =
            tracing::subscriber::with_default(tracing::subscriber::NoSubscriber::default(), || {
                (self.engine_options)(path)
            });
        let on_workers = self.on_workers(&LoadModelRequest::default());
        let context = Self::load_context(&options, None, n_ctx_train, on_workers);
        let output_cap = crate::engine::output_cap(self.max_output_tokens);
        ModelLimits::new(context, n_ctx_train, output_cap)
    }

    /// Place the model on workers: they receive only their stages, from this
    /// controller, and this host keeps just the tokenizer and sampling.
    /// `n_ctx` is the load's own context window, if it names one.
    #[cfg(feature = "distributed")]
    fn load_distributed(
        &self,
        req: &LoadModelRequest,
        n_ctx: Option<u32>,
        path: &Path,
        id: String,
        registry: &ModelRegistry,
    ) -> crate::Result<LoadedModel> {
        use crate::distributed::{
            pipeline::{Deployment, Limits, Pipeline},
            remote::PipelineBackend,
        };
        use crate::JoshuaError::{InvalidRequest, ModelLoad};
        let key = self.cluster_key.as_deref().ok_or_else(|| {
            InvalidRequest("set JOSHUA_CLUSTER_KEY on the controller to use workers".into())
        })?;
        let workers = match &req.workers {
            Some(workers) => {
                if let Some(w) = workers.iter().find(|w| !self.workers.contains(w)) {
                    return Err(InvalidRequest(format!("{w} is not a configured worker")));
                }
                workers.clone()
            }
            None => self.workers.clone(),
        };
        if workers.is_empty() {
            return Err(InvalidRequest("no workers to place the model on".into()));
        }
        // A worker holds one model at a time and serves one controller.
        let claim = registry.claim_workers(&workers)?;
        let facts = GgufFacts::read(path)?;
        let model_ctx = facts.n_ctx_train.ok_or_else(|| {
            let arch = facts.architecture.as_deref().unwrap_or_default();
            ModelLoad(format!(
                "{arch}: missing GGUF metadata key `{arch}.context_length`"
            ))
        })?;
        let defaults = Limits::default();
        let server = (self.engine_options)(path);
        let context = Self::load_context(&server, n_ctx, Some(model_ctx), true) as usize;
        let limits = Limits {
            context,
            chunk: req.chunk.unwrap_or(defaults.chunk).min(context),
            sessions: req.sessions.unwrap_or(defaults.sessions),
            ..defaults
        };
        // The controller maps the file only to tokenize and to read slices;
        // nothing is prefetched or loaded for compute here.  It renders the
        // chat, samples and reads tool calls, so the server's template,
        // sampling and tool-call parser flags apply.
        let options = EngineOptions::with_n_ctx(context as u32)
            .backend(crate::engine::ComputeBackend::Cpu)
            .lazy_weights(true)
            .prefill_chunk(limits.chunk)
            .sampling(server.sampling)
            .chat_template(server.chat_template)
            .trust_request_chat_template(server.trust_request_chat_template)
            .tool_call_parser(server.tool_call_parser);
        let engine = Engine::with_options(path, options)?;
        let pipeline = Pipeline::deploy(
            path,
            &workers,
            key,
            Deployment {
                ends: req.ends.clone(),
                limits,
            },
        )
        .map_err(|e| ModelLoad(format!("placing the model on workers: {e:#}")))?;
        let stages = pipeline
            .plan()
            .stages
            .iter()
            .map(|s| (s.start, s.end))
            .collect();
        let backend = PipelineBackend::new(pipeline);
        let capacity = backend.capacity();
        let engine = self
            .finish_engine(engine)
            .with_remote_backend(Arc::new(backend), capacity);
        claim.keep();
        Ok(LoadedModel {
            embedding_job: embedding_job(&engine, &id),
            id,
            engine: Arc::new(engine),
            workers,
            stages,
            created: unix_time(),
        })
    }
}

/// Map a model-load failure to the error a caller sees.
///
/// A load the caller asked for reports its cause (`UNPROCESSABLE_ENTITY` with
/// the engine's own message); anything else is an internal failure.
fn load_failure(err: crate::JoshuaError) -> ApiError {
    match err {
        crate::JoshuaError::ModelLoad(msg) => {
            ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, msg, "model_load_failed")
        }
        e => ApiError::from(e),
    }
}

/// Load `req`'s model file, already resolved to `path`, as `id`.
///
/// `target` is `path` canonicalised when the caller has it: the id may not be
/// another listed file's path (see [`ModelManager::names_another_file`]), which
/// needs a header read and so stays inside the blocking load.
///
/// The reservation in `state.models` keeps any other load of the same id out
/// while this one runs, and the caller gets the loaded model back to answer
/// with (dropping its engine releases the weights).
async fn load_model_file(
    state: &AppState,
    req: LoadModelRequest,
    path: PathBuf,
    id: String,
    target: Option<PathBuf>,
) -> Result<LoadedModel, ApiError> {
    let state = Arc::clone(state);
    tokio::task::spawn_blocking(move || {
        let manager = manager(&state)?;
        if manager.names_another_file(&id, target.as_deref()) {
            return Err(ApiError::new(
                StatusCode::CONFLICT,
                format!("id '{id}' is the path of another model file in the model directory"),
                "model_exists",
            ));
        }
        let guard = state.models.reserve(&id)?;
        tracing::info!(model = %id, path = %path.display(), "loading model");
        let model = manager
            .load(&req, &path, id, &state.models)
            .map_err(load_failure)?;
        guard.finish(model.clone());
        Ok::<_, ApiError>(model)
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?
}

/// `POST /v1/models/load`'s answer: the model's `/v1/models` entry, with
/// its workers and stages also at the top level, where they have always
/// been.
fn model_json(model: &LoadedModel) -> serde_json::Value {
    let mut body = json!(model.info().card());
    let workers: Vec<String> = model.workers.iter().map(|w| w.to_string()).collect();
    body["workers"] = json!(workers);
    body["stages"] = json!(model.stages);
    body
}

/// The engine for the model `requested` names, loading it on first use.
///
/// A `--model-dir` server lists every valid model in its directory, so the
/// directory is the model store: a request selects a model by naming it, and
/// the first request for one that is not loaded maps it in — no
/// `POST /v1/models/load` needed first.  Loads of one id are serialised by
/// [`ModelRegistry::load_gate`], so a burst of requests for a cold model shares
/// its single load instead of colliding in [`ModelRegistry::reserve`].
///
/// A name the directory does not list keeps the registry's own diagnosis,
/// which reports what *is* loaded ([`ModelRegistry::resolve`]).
pub(crate) async fn select_model(
    state: &AppState,
    requested: &str,
) -> Result<(String, Arc<Engine>), ApiError> {
    let unselected = match state.resolve(requested) {
        Ok(found) => return Ok(found),
        Err(e) => e,
    };
    // Only a file the inventory lists is loadable here.
    let requested = requested.trim();
    let Some((id, path)) = state
        .unloaded()
        .into_iter()
        .find(|(id, _)| id == requested)
    else {
        return Err(unselected);
    };
    let gate = state.models.load_gate(&id);
    let _held = gate.lock().await;
    // Another request may have finished loading it while this one waited.
    if let Some(found) = state.models.loaded(&id) {
        return Ok(found);
    }
    // Loading is logged by `load_model_file` with the file it maps; this says
    // why the load happened.
    tracing::info!(model = %id, "loading the model a request named");
    let target = path.canonicalize().ok();
    let model = load_model_file(
        state,
        LoadModelRequest {
            model: id.clone(),
            ..Default::default()
        },
        path,
        id.clone(),
        target,
    )
    .await?;
    Ok((model.id, Arc::clone(&model.engine)))
}

fn embedding_job(engine: &Engine, model: &str) -> JobModel {
    JobModel {
        model: model.to_string(),
        backend: device_label(engine.device()).to_string(),
        input_format: "text".to_string(),
        owner: Uuid::new_v4().to_string(),
        revision: if engine.supports_embeddings() { engine.embedding_revision() } else { None },
        max_inputs: 32,
        max_input_bytes: 256 << 10,
        max_output_bytes: 2 << 20,
    }
}

fn manager(state: &ServerState) -> Result<&ModelManager, ApiError> {
    state.manager.as_ref().ok_or_else(|| {
        ApiError::new(
            StatusCode::FORBIDDEN,
            "model management is disabled; start the server with --model-dir",
            "model_management_disabled",
        )
    })
}

/// `POST /v1/models/load` — load a model and serve it under its id.
async fn load_model(
    State(state): State<AppState>,
    Json(req): Json<LoadModelRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let unloaded = state.unloaded();
    let path = match manager(&state)?.resolve_path(&req.model) {
        Ok(path) => path,
        // A file the inventory lists by its stem, e.g. in a subdirectory.
        Err(e) => unloaded
            .iter()
            .find(|(id, _)| *id == req.model)
            .map(|(_, file)| file.clone())
            .ok_or(e)?,
    };
    // By default the model is served under the id the inventory lists its
    // file by, so the list and the served id agree; a file the inventory
    // does not list goes by its stem.
    let target = path.canonicalize().ok();
    let id = req.id.clone().unwrap_or_else(|| {
        unloaded
            .into_iter()
            .find(|(_, file)| target.is_some() && file.canonicalize().ok() == target)
            .map(|(id, _)| id)
            .unwrap_or_else(|| {
                path.file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default()
            })
    });
    if id.is_empty() {
        return Err(ApiError::bad_request("empty model id"));
    }
    let model = load_model_file(&state, req, path, id, target).await?;
    Ok(Json(model_json(&model)))
}

/// `POST /v1/models/unload` — stop serving a model and release it.
async fn unload_model(
    State(state): State<AppState>,
    Json(req): Json<UnloadModelRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    manager(&state)?;
    let model = state
        .models
        .remove(&req.id)
        .ok_or_else(|| ApiError::model_not_found(format!("model '{}' is not loaded", req.id)))?;
    tracing::info!(model = %model.id, "unloading model");
    // New requests no longer find it. Wait for the ones already running,
    // then drop the engine here: that frees its weights or releases its
    // workers, so they are free for the next load once this returns.
    tokio::task::spawn_blocking(move || {
        let mut engine = model.engine;
        loop {
            match Arc::try_unwrap(engine) {
                Ok(engine) => break drop(engine),
                Err(shared) => engine = shared,
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        state.models.release_workers(&model.workers);
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?;
    Ok(Json(json!({"id": req.id, "object": "model", "unloaded": true})))
}

// ─── Router ───────────────────────────────────────────────────────────────────

/// Build the Axum router with all API routes mounted.
pub fn create_router(state: AppState) -> Router {
    let api = Router::new()
        .route("/v1/models", get(list_models))
        .route("/models", get(list_models))
        .route("/v1/models/{*id}", get(retrieve_model))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(messages_count_tokens))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/audio/transcriptions", post(transcriptions))
        .route(crate::coordinator::WORKER_INFO_PATH, get(worker_info))
        // Ops endpoints sit behind the API key when one is configured:
        // they expose model names and load figures, which are not useful to
        // an unauthenticated caller.  `/health` below stays open for probes.
        .route("/status", get(server_status))
        .route("/metrics", get(metrics_json))
        .route("/metrics/prometheus", get(metrics_prometheus))
        .route("/props", get(props))
        .route("/slots", get(slots))
        .layer(middleware::from_fn_with_state(
            crate::job_admission::Admission::default(),
            crate::job_admission::admit,
        ))
        .route("/tokenize", post(tokenize))
        .route("/v1/tokenize", post(tokenize))
        .route("/v1/detokenize", post(detokenize))
        .route("/api/tokens", post(tokenize))
        .route("/load", get(server_load))
        .layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            require_api_key,
        ));
    // Model management stays same-origin: without CORS headers a browser
    // will not send these JSON requests from another site's page, even to
    // a localhost server with no API key.
    let manage = Router::new()
        .route("/v1/models/load", post(load_model))
        .route("/v1/models/unload", post(unload_model))
        .route("/api/version", get(crate::ollama::version))
        .route("/api/tags", get(crate::ollama::tags))
        .route("/api/ps", get(crate::ollama::ps))
        .route("/api/show", post(crate::ollama::show))
        .route("/api/chat", post(crate::ollama::chat))
        .route("/api/generate", post(crate::ollama::generate))
        .route("/api/embed", post(crate::ollama::embed))
        .route("/api/embeddings", post(crate::ollama::embed))
        .route("/api/delete", delete(crate::ollama::delete))
        .route("/api/copy", post(crate::ollama::copy))
        .route("/api/create", post(crate::ollama::create))
        .route("/api/pull", post(crate::ollama::pull))
        .layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            require_api_key,
        ));
    let cors = cors_layer(&state.cors_origins);
    Router::new()
        .route("/health", get(health))
        .merge(api)
        .layer(cors)
        .merge(manage)
        .layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            count_requests,
        ))
        .with_state(state)
}

fn cors_layer(origins: &[String]) -> CorsLayer {
    if origins.is_empty() {
        // Historical behaviour: a localhost development server is callable
        // from any browser origin.  Deployments should set --cors-origin.
        return CorsLayer::permissive();
    }
    let origins: Vec<HeaderValue> = origins
        .iter()
        .filter_map(|origin| origin.trim().parse().ok())
        .collect();
    CorsLayer::new()
        .allow_origin(AllowOrigin::list(origins))
        .allow_methods(Any)
        .allow_headers(Any)
}

/// Count every response, including ones rejected by auth.
async fn count_requests(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Response {
    let response = next.run(req).await;
    state.metrics.record_http(response.status());
    response
}

/// Reject `/v1` requests that lack the configured bearer API key.
///
/// Without a configured key, accept public requests but discard internal cache
/// scopes. Comparison is constant-time so the key can't be recovered byte-by-byte
/// through response timing.
async fn require_api_key(State(state): State<AppState>, mut req: Request, next: Next) -> Response {
    if let Some(expected) = &state.api_key {
        let provided = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .or_else(|| {
                req.headers()
                    .get("x-api-key")
                    .and_then(|v| v.to_str().ok())
            });
        // A comma-separated configuration supports key rotation: any listed
        // key is accepted.  Empty entries are ignored so a trailing comma or
        // a stray space does not accidentally accept an empty key.  A value
        // that contains no non-empty key is a configuration error: refuse
        // every request rather than silently disabling authentication.
        let expected_keys: Vec<&str> = expected
            .split(',')
            .map(str::trim)
            .filter(|candidate| !candidate.is_empty())
            .collect();
        if expected_keys.is_empty() {
            let body = ErrorResponse::new(
                "server misconfigured: --api-key contains no non-empty keys",
                "server_error",
            );
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response();
        }
        if !provided.is_some_and(|key| {
            expected_keys
                .iter()
                .any(|candidate| api_keys_match(key.as_bytes(), candidate.as_bytes()))
        }) {
            let body = ErrorResponse::new(
                "invalid or missing API key — pass the key as \
                 'Authorization: Bearer <key>' or 'x-api-key: <key>'",
                "invalid_request_error",
            );
            return (StatusCode::UNAUTHORIZED, Json(body)).into_response();
        }
    } else {
        req.headers_mut().remove(crate::affinity::CACHE_SCOPE_HEADER);
    }
    next.run(req).await
}

/// Key equality that leaks nothing about the configured key through timing.
///
/// Both keys are hashed first, so the comparison always runs over two
/// fixed-size digests regardless of either key's length or contents; the
/// digests are then compared with a branch-free fold.  Timing can therefore
/// only reveal information about SHA-256 digests, which is useless without
/// inverting the hash.
pub(crate) fn api_keys_match(provided: &[u8], expected: &[u8]) -> bool {
    use sha2::{Digest, Sha256};

    let provided = Sha256::digest(provided);
    let expected = Sha256::digest(expected);
    provided
        .iter()
        .zip(expected.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}

/// Start the server on `addr` (e.g. `"0.0.0.0:8080"`) with just a chat
/// engine.  Use [`serve_with_state`] to also mount a Whisper model.
pub async fn serve(engine: Arc<Engine>, addr: &str) -> std::io::Result<()> {
    serve_with_state(Arc::new(ServerState::new(engine)), addr).await
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

/// `GET /health` — liveness plus a compact model/load summary.
async fn health(State(state): State<AppState>) -> Json<serde_json::Value> {
    let models = state.models.list();
    let in_flight: usize = models.iter().map(|m| m.engine.in_flight()).sum();
    let max_concurrency: usize = models.iter().map(|m| m.engine.max_concurrency()).sum();
    Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_seconds": state.metrics.started.elapsed().as_secs(),
        "models": models.len(),
        "in_flight": in_flight,
        "max_concurrency": max_concurrency,
    }))
}

/// `GET /status` — model inventory and live load.
async fn server_status(State(state): State<AppState>) -> Json<serde_json::Value> {
    let models = state.models.list();
    let model_values: Vec<serde_json::Value> = models
        .iter()
        .map(|m| {
            json!({
                "id": m.id,
                "n_ctx": m.engine.n_ctx(),
                "backend": device_label(m.engine.device()),
                "in_flight": m.engine.in_flight(),
                "max_concurrency": m.engine.max_concurrency(),
                "embeddings": m.engine.supports_embeddings(),
                "remote": m.engine.remote_only(),
                "workers": m.workers.iter().map(|w| w.to_string()).collect::<Vec<_>>(),
                "stages": m.stages,
                "created": m.created,
                "has_chat_template": m.engine.has_chat_template(),
            })
        })
        .collect();
    let in_flight: usize = models.iter().map(|m| m.engine.in_flight()).sum();
    let max_concurrency: usize = models.iter().map(|m| m.engine.max_concurrency()).sum();
    Json(json!({
        "object": "joshua.status",
        "version": env!("CARGO_PKG_VERSION"),
        "models": model_values,
        "in_flight": in_flight,
        "max_concurrency": max_concurrency,
        "metrics": state.metrics.snapshot(),
    }))
}

/// `GET /metrics` — JSON counters.
async fn metrics_json(State(state): State<AppState>) -> Json<serde_json::Value> {
    let mut metrics = state.metrics.snapshot();
    metrics["prefix_cache"] = serde_json::Value::Array(
        state
            .models
            .list()
            .iter()
            .map(|model| {
                let cache = model.engine.prefix_cache_stats();
                let (warm_sessions, warm_state_bytes) = model.engine.warm_cache_stats();
                json!({
                    "model": model.id,
                    "reuses": model.engine.kv_reuse_count(),
                    "rewinds": model.engine.kv_edit_reuse_count(),
                    "tokens_reused": model.engine.kv_tokens_reused(),
                    "snapshot_entries": cache.entries,
                    "snapshot_bytes": cache.bytes,
                    "snapshot_budget": cache.budget,
                    "snapshot_evictions": cache.evicted,
                    "snapshot_expirations": cache.expired,
                    "warm_sessions": warm_sessions,
                    "warm_state_bytes": warm_state_bytes,
                })
            })
            .collect(),
    );
    Json(metrics)
}

/// `GET /metrics/prometheus` — Prometheus text exposition format.
async fn metrics_prometheus(State(state): State<AppState>) -> Response {
    let models = state.models.list();
    let in_flight: usize = models.iter().map(|m| m.engine.in_flight()).sum();
    let max_concurrency: usize = models.iter().map(|m| m.engine.max_concurrency()).sum();
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        state
            .metrics
            .prometheus(in_flight, max_concurrency, models.len()),
    )
        .into_response()
}

/// `GET /props` — llama.cpp-compatible server properties.
async fn props(State(state): State<AppState>) -> Result<Json<serde_json::Value>, ApiError> {
    let models = state.models.list();
    let Some(first) = models.first() else {
        return Err(ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "no model is loaded",
            "model_not_loaded",
        ));
    };
    // What a llama.cpp-native request gets when it sets nothing.
    let defaults = first.engine.sampling_defaults(SamplingProfile::LlamaCpp);
    let total_slots: usize = models.iter().map(|m| m.engine.max_concurrency()).sum();
    Ok(Json(json!({
        "model_path": first.engine.model_path(),
        "model_name": first.id,
        "n_ctx": first.engine.n_ctx(),
        "chat_template": first.engine.has_chat_template().then_some("<gguf chat template>"),
        "total_slots": total_slots,
        "default_generation_settings": {
            "n_predict": -1,
            "temperature": defaults.temperature,
            "top_p": defaults.top_p,
            "top_k": defaults.top_k,
            "min_p": defaults.min_p,
            "repeat_penalty": defaults.repetition_penalty,
        },
    })))
}

/// `GET /slots` — one lightweight slot record per loaded model.
async fn slots(State(state): State<AppState>) -> Json<serde_json::Value> {
    let slots: Vec<serde_json::Value> = state
        .models
        .list()
        .into_iter()
        .enumerate()
        .map(|(id, model)| {
            json!({
                "id": id,
                "model": model.id,
                "n_ctx": model.engine.n_ctx(),
                "in_flight": model.engine.in_flight(),
                "max_concurrency": model.engine.max_concurrency(),
                "state": if model.engine.in_flight() == 0 { "idle" } else { "busy" },
            })
        })
        .collect();
    Json(serde_json::Value::Array(slots))
}

/// `POST /tokenize` (also `/api/tokens`) — count a prompt with the loaded
/// tokenizer.  Body: `{"model": "...", "prompt": "..."}` or llama.cpp's
/// `{"content": "..."}`; `add_special_tokens` defaults to true.
async fn tokenize(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let (_, engine) = select_model(
        &state,
        body.get("model")
            .and_then(|v| v.as_str())
            .unwrap_or_default(),
    )
    .await?;
    let prompt = body
        .get("prompt")
        .or_else(|| body.get("content"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::bad_request("missing 'prompt' or 'content'"))?;
    let add_special = body
        .get("add_special_tokens")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let tokens = engine
        .tokenize(prompt, add_special)
        .map_err(ApiError::from)?;
    Ok(Json(json!({
        "tokens": tokens,
        "count": tokens.len(),
    })))
}

/// `POST /v1/detokenize` — turn token ids back into text.
///
/// vLLM and llama.cpp both expose this next to tokenize, and OpenAI-compatible
/// tooling uses it to reconstruct a prompt from ids.  Body:
/// `{"model": "...", "tokens": [1, 2, 3], "skip_special_tokens": true}`.
/// `skip_special_tokens` defaults to **true**, matching vLLM (which drops
/// BOS/EOS unless asked otherwise); llama.cpp's `/detokenize` only takes
/// `tokens`, and this body is a superset of that.
async fn detokenize(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let (_, engine) = select_model(
        &state,
        body.get("model")
            .and_then(|v| v.as_str())
            .unwrap_or_default(),
    )
    .await?;
    let tokens = body
        .get("tokens")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ApiError::bad_request("missing 'tokens' array"))?;
    let mut ids = Vec::with_capacity(tokens.len());
    for (i, t) in tokens.iter().enumerate() {
        let id = t
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| {
                ApiError::bad_request(&format!("token at index {i} is not a non-negative integer"))
            })?;
        ids.push(id);
    }
    let skip_special = body
        .get("skip_special_tokens")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let text = engine
        .detokenize(&ids, skip_special)
        .map_err(ApiError::from)?;
    Ok(Json(json!({ "prompt": text })))
}

/// `GET /load` — vLLM's server-load metrics.
///
/// vLLM reports the queue depth and the running/waiting request counts so a
/// client or a load balancer can pick a replica.  joshua admits per engine and
/// has no prefill queue of its own, so `waiting` is the requests this server
/// has accepted but not yet admitted (the admission cap's backlog) and
/// `running` is the in-flight count.
async fn server_load(State(state): State<AppState>) -> Json<serde_json::Value> {
    let models = state.models.list();
    let running: usize = models.iter().map(|m| m.engine.in_flight()).sum();
    let capacity: usize = models.iter().map(|m| m.engine.max_concurrency()).sum();
    // Anything already counted as in-flight is running; the remainder of the
    // configured concurrency is free.  joshua rejects over-capacity requests
    // rather than queueing them unboundedly, so the backlog is the difference
    // between what has been accepted and what is admitted, which the engine
    // tracks as `in_flight` against `max_concurrency`.
    let waiting = 0usize;
    Json(json!({
        "running": running,
        "waiting": waiting,
        "capacity": capacity,
        "free": capacity.saturating_sub(running),
        "models": models.len(),
    }))
}

/// `GET /v1/worker/info` — what a `joshua route` coordinator needs to admit
/// requests here: protocol version, model identity, context limit, backend,
/// and the admission cap with the current in-flight count.
///
/// With several models loaded, `model`, `n_ctx` and `backend` describe the
/// first, `models` gives each one's own admission figures (each engine
/// admits its own requests), and the worker-wide figures are their sums.
async fn worker_info(State(state): State<AppState>) -> Result<Json<WorkerInfo>, ApiError> {
    let models = state.models.list();
    let first = models.first();
    if first.is_none() && state.whisper.is_none() {
        return Err(ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "no model is loaded; load one with POST /v1/models/load",
            "model_not_loaded",
        ));
    }
    let engines = || models.iter().map(|m| &m.engine);
    Ok(Json(WorkerInfo {
        transcription_job: state.whisper.as_ref().map(|whisper| JobModel {
            model: whisper.model_name().to_string(), owner: whisper.job_owner.clone(),
            backend: device_label(whisper.device()).to_string(), input_format: "wav".to_string(),
            revision: None, max_inputs: 1, max_input_bytes: crate::job_admission::AUDIO_BYTES,
            max_output_bytes: 2 << 20,
        }),
        transcription_in_flight: state.whisper.as_ref().map_or(0, |whisper| {
            1usize.saturating_sub(whisper.admission.available_permits())
        }),
        embedding_jobs: models.iter().filter(|model| model.engine.supports_embeddings())
            .map(|model| model.embedding_job.clone()).collect(),
        cards: models
            .iter()
            .map(LoadedModel::info)
            .chain(state.whisper_info())
            .map(|model| json!(model.card()))
            .collect(),
        protocol_version: WORKER_PROTOCOL_VERSION,
        cache_epochs: models
            .iter()
            .map(|model| (model.id.clone(), model.engine.cache_epoch().to_owned()))
            .collect(),
        cache_residency: models
            .iter()
            .map(|model| {
                (
                    model.id.clone(),
                    model.engine.cache_residency(128 / models.len().max(1)),
                )
            })
            .collect(),
        model: first.map(|model| model.id.clone()).unwrap_or_else(|| state.whisper.as_ref().unwrap().model_name().to_string()),
        models: models
                .iter()
                .map(|m| ModelSlots {
                    id: m.id.clone(),
                    n_ctx: Some(m.engine.n_ctx()),
                    max_concurrency: m.engine.max_concurrency(),
                    in_flight: m.engine.in_flight(),
                    embeddings: m.engine.supports_embeddings(),
                })
                .collect(),
        n_ctx: first.map_or(0, |model| model.engine.n_ctx()),
        backend: first.map(|model| device_label(model.engine.device()))
            .unwrap_or_else(|| device_label(state.whisper.as_ref().unwrap().device())).to_string(),
        max_concurrency: engines().map(|e| e.max_concurrency()).sum::<usize>().max(1),
        in_flight: engines().map(|e| e.in_flight()).sum(),
        capabilities: WorkerCapabilities {
            chat: first.is_some(),
            completions: first.is_some(),
            embeddings: engines().any(|e| e.supports_embeddings()),
            transcriptions: state.whisper.is_some(),
            streaming: first.is_some(),
            tools: first.is_some(),
        },
    }))
}

/// Sets its flag when dropped.  Held by a handler across the blocking
/// generation, so when hyper drops the handler future (the client hung up)
/// the engine sees the flag and stops decoding.
pub(crate) struct CancelOnDrop(Arc<AtomicBool>);

impl CancelOnDrop {
    pub(crate) fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    pub(crate) fn flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.0)
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// The `/v1/models` entries: [`ServerState::inventory`], then the Whisper
/// model.
async fn model_cards(state: AppState) -> Result<Vec<ModelCard>, ApiError> {
    Ok(inventory(&state)
        .await?
        .into_iter()
        .chain(state.whisper_info())
        .map(|model| model.card())
        .collect())
}

/// `GET /v1/models` (also `GET /models`, as llama-server serves it) — the
/// loaded models, the model directory's other files, and the Whisper model.
async fn list_models(State(state): State<AppState>) -> Result<Json<ModelListResponse>, ApiError> {
    Ok(Json(ModelListResponse::new(model_cards(state).await?)))
}

/// `GET /v1/models/{id}` — one model's `/v1/models` entry; an id no entry
/// has is OpenAI's `404`.
async fn retrieve_model(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<ModelCard>, ApiError> {
    model_cards(state)
        .await?
        .into_iter()
        .find(|card| card.id == id)
        .map(Json)
        .ok_or_else(|| ApiError::unknown_model(&id))
}

/// `POST /v1/chat/completions` — OpenAI chat completions.
async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let (model, engine) = select_model(&state, &req.model).await?;
    let stream = req.stream.unwrap_or(false);
    let defaults = engine.sampling_defaults(SamplingProfile::OpenAi);
    let mut options = req
        .to_generation_options_with(&defaults)
        .map_err(ApiError::bad_request)?;
    // Answer a request the model cannot serve before a stream starts.
    engine.validate_options(&options).map_err(ApiError::from)?;
    let reasoning_output = req.reasoning_output().map_err(ApiError::bad_request)?;
    options.cache_scope = Some(crate::affinity::cache_scope(
        &headers,
        options.context_window,
    ));
    let messages = req.messages.clone();
    let tools = req.offered_tools();
    // vLLM's `echo`: repeat the assistant message being continued.
    let echo = (req.echo == Some(true))
        .then(|| messages.last().filter(|m| m.role == "assistant"))
        .flatten()
        .map(|m| m.content.clone());

    if stream {
        // ── Streaming path ────────────────────────────────────────────────────
        let id = format!("chatcmpl-{}", Uuid::new_v4().simple());
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let cancel = CancelOnDrop::new();
        let flag = cancel.flag();
        state.metrics.streamed_requests.fetch_add(1, Ordering::Relaxed);

        // Inference runs on a blocking thread and reports text through the
        // engine's per-token callback.  The SSE body is returned immediately;
        // hyper pulls chunks as the channel fills.
        tokio::task::spawn_blocking({
            let engine = Arc::clone(&engine);
            let model = model.clone();
            let id = id.clone();
            let tools = tools.clone();
            let metrics = Arc::clone(&state.metrics);
            move || {
                stream_chat_completion(
                    &engine,
                    &model,
                    &id,
                    &messages,
                    tools.as_deref(),
                    &options,
                    ChatStreamOutput {
                        reasoning: reasoning_output,
                        echo,
                    },
                    &flag,
                    tx,
                    &metrics,
                );
            }
        });

        let events = stream::unfold(
            (rx, cancel),
            |(mut rx, cancel)| async move {
                rx.recv()
                    .await
                    .map(|data| (Ok::<Event, Infallible>(Event::default().data(data)), (rx, cancel)))
            },
        );
        return Ok(Sse::new(events).into_response());
    }

    // ── Non-streaming path ────────────────────────────────────────────────────
    let tool_call_format = if tools.is_some() {
        engine
            .tool_call_format_for_request(&messages, tools.as_deref(), &options)
            .map_err(ApiError::from)?
    } else {
        engine.tool_call_format()
    };
    let cancel = CancelOnDrop::new();
    let completion = tokio::task::spawn_blocking({
        let engine = Arc::clone(&engine);
        let tools = tools.clone();
        let flag = cancel.flag();
        move || {
            engine.complete_chat_cancellable_detailed(&messages, tools.as_deref(), &options, &flag)
        }
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?
    .map_err(ApiError::from)?;
    state.metrics.record_generation(&completion.usage);

    let answer = chat_answer(
        &engine,
        &completion,
        reasoning_output.splits(false),
        tools.as_deref(),
        tool_call_format,
        OPENAI_CALL_IDS,
    );
    let reasoning = answer
        .reasoning
        .filter(|_| reasoning_output != ReasoningOutput::Hidden);
    let finish_reason = completion
        .stop_type
        .chat_finish_reason(!answer.calls.is_empty());
    let (mut content, tool_calls) = if answer.calls.is_empty() {
        (Some(answer.content), None)
    } else {
        let calls: Vec<ToolCall> = answer
            .calls
            .into_iter()
            .map(|c| ToolCall {
                id: c.id.unwrap_or_default(),
                call_type: "function".to_string(),
                function: FunctionCallResult {
                    name: c.name,
                    arguments: c.arguments,
                },
            })
            .collect();
        let prose = Some(answer.prose).filter(|p| !p.is_empty());
        (prose, Some(calls))
    };
    // vLLM's `echo` goes in front of the content once the calls are out of
    // it, as the stream sends it first.
    if let Some(prefix) = &echo {
        content = Some(format!("{prefix}{}", content.unwrap_or_default()));
    }

    let id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let mut response = ChatCompletionResponse::new(
        id,
        model,
        vec![ChatChoice {
            index: 0,
            message: AssistantMessage {
                role: "assistant".to_string(),
                content,
                reasoning_content: reasoning.clone(),
                reasoning,
                tool_calls,
            },
            finish_reason: finish_reason.to_string(),
        }],
        completion.usage.clone(),
    );
    response.timings = Some(completion.timings);
    Ok(Json(response).into_response())
}
/// One piece of a chat completion's output, in the order to send it.
pub(crate) enum ChatPiece<'a> {
    /// Reasoning text.
    Reasoning(&'a str),
    /// Answer prose that is safe to send.
    Text(&'a str),
    /// Tool call `index` begins, with its id (the model's own, or one
    /// minted for it) and its function name.
    CallStart {
        index: usize,
        id: &'a str,
        name: &'a str,
    },
    /// More of call `index`'s JSON arguments.
    CallArgs { index: usize, delta: &'a str },
}

/// The prefix of the ids minted for an OpenAI response's tool calls.
const OPENAI_CALL_IDS: &str = "call_";
/// The prefix of the ids minted for an Anthropic response's `tool_use`
/// blocks.
const ANTHROPIC_CALL_IDS: &str = "toolu_";

/// The output post-processing every chat API shares, streaming or not: the
/// reasoning is split off first (by the engine, as it decodes), then the
/// answer goes through the model's tool-call reader, which passes prose on
/// at once and each call as it is written, without its markup.  Each API
/// maps the pieces onto its own wire events.
struct ChatOutput<'a> {
    calls: ToolCallStream<'a>,
    format: ToolCallFormat,
    /// The prefix of the ids minted for calls the model gives none.
    id_prefix: &'static str,
    /// Each call's id, as sent.
    ids: Vec<String>,
}

impl<'a> ChatOutput<'a> {
    fn new(format: ToolCallFormat, tools: Option<&'a [Tool]>, id_prefix: &'static str) -> Self {
        Self {
            calls: ToolCallStream::new(format, tools),
            format,
            id_prefix,
            ids: Vec::new(),
        }
    }

    /// One step of the split output: its reasoning, then what of the answer
    /// is safe to send now.
    fn push(&mut self, split: &Split, emit: &mut dyn FnMut(ChatPiece<'_>)) {
        if !split.reasoning.is_empty() {
            emit(ChatPiece::Reasoning(&split.reasoning));
        }
        if !split.content.is_empty() {
            let events = self.calls.push(&split.content);
            self.send(events, emit);
        }
    }

    /// After decoding: the rest of the answer, then the tool calls read
    /// from it, under the ids they were sent with.
    fn finish(mut self, emit: &mut dyn FnMut(ChatPiece<'_>)) -> Vec<ParsedToolCall> {
        let events = self.calls.finish();
        self.send(events, emit);
        let mut calls = self.calls.calls();
        for (call, id) in calls.iter_mut().zip(self.ids) {
            call.id = Some(id);
        }
        calls
    }

    fn send(&mut self, events: Vec<ToolEvent>, emit: &mut dyn FnMut(ChatPiece<'_>)) {
        for event in events {
            match event {
                ToolEvent::Text(text) => emit(ChatPiece::Text(&text)),
                ToolEvent::CallStart { index, id, name } => {
                    let id = id.unwrap_or_else(|| self.format.call_id(self.id_prefix));
                    emit(ChatPiece::CallStart {
                        index,
                        id: &id,
                        name: &name,
                    });
                    self.ids.push(id);
                }
                ToolEvent::CallArgs { index, delta } => emit(ChatPiece::CallArgs {
                    index,
                    delta: &delta,
                }),
            }
        }
    }
}

/// Run a streaming chat completion through the shared post-processing:
/// `emit` gets each piece as it is decoded, with the reasoning split off
/// when `split` (otherwise all of it is answer).  Calls the model gives no
/// id get one starting `id_prefix`.  Returns the completion and the tool
/// calls read from the answer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn stream_chat_output(
    engine: &Engine,
    messages: &[ChatMessage],
    tools: Option<&[Tool]>,
    options: &GenerationOptions,
    cancel: &AtomicBool,
    split: bool,
    id_prefix: &'static str,
    emit: &mut dyn FnMut(ChatPiece<'_>),
) -> crate::error::Result<(Completion, Vec<ParsedToolCall>)> {
    let format = if tools.is_some() {
        engine.tool_call_format_for_request(messages, tools, options)?
    } else {
        engine.tool_call_format()
    };
    let mut output = ChatOutput::new(format, tools, id_prefix);
    let completion = if split {
        engine.complete_chat_streaming_split(messages, tools, options, cancel, &mut |s| {
            output.push(&s, emit)
        })
    } else {
        engine.complete_chat_streaming_detailed(messages, tools, options, cancel, &mut |delta| {
            let s = Split {
                content: delta.to_string(),
                ..Default::default()
            };
            output.push(&s, emit)
        })
    }?;
    Ok((completion, output.finish(emit)))
}

/// A finished chat completion after the shared post-processing.
pub(crate) struct ChatAnswer {
    /// The reasoning split off the output.
    pub(crate) reasoning: Option<String>,
    /// The answer as a stream sends it: the output after its reasoning,
    /// without its tool-call markup.
    pub(crate) content: String,
    /// The answer around its tool calls, trimmed as the tool-call parser
    /// leaves it (the answer itself when no tools were offered).
    pub(crate) prose: String,
    /// The tool calls in the answer, each with its id.
    pub(crate) calls: Vec<ParsedToolCall>,
}

/// [`stream_chat_output`] for a finished completion: the reasoning split
/// off its text when `split`, then the tool calls read from the answer when
/// `tools` were offered, with ids starting `id_prefix` for those the model
/// gives none.
pub(crate) fn chat_answer(
    engine: &Engine,
    completion: &Completion,
    split: bool,
    tools: Option<&[Tool]>,
    format: crate::tools::ToolCallFormat,
    id_prefix: &str,
) -> ChatAnswer {
    let (reasoning, text) = if split {
        crate::reasoning::split(
            engine.reasoning_format(),
            completion.reasoning_open,
            &completion.text,
        )
    } else {
        (None, completion.text.clone())
    };
    let Some(tools) = tools else {
        return ChatAnswer {
            reasoning,
            prose: text.clone(),
            content: text,
            calls: Vec::new(),
        };
    };
    let (content, mut calls) = crate::tools::read_tool_calls(format, tools, &text);
    for call in &mut calls {
        call.id.get_or_insert_with(|| format.call_id(id_prefix));
    }
    ChatAnswer {
        reasoning,
        prose: content.trim().to_string(),
        content,
        calls,
    }
}

/// Writes OpenAI streaming chunks to the SSE channel.
struct ChatStreamEmitter {
    id: String,
    model: String,
    tx: tokio::sync::mpsc::UnboundedSender<String>,
    role_sent: bool,
}

impl ChatStreamEmitter {
    fn new(
        id: String,
        model: String,
        tx: tokio::sync::mpsc::UnboundedSender<String>,
    ) -> Self {
        Self {
            id,
            model,
            tx,
            role_sent: false,
        }
    }

    fn text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        let delta = DeltaContent {
            role: self.take_role(),
            content: Some(text.to_string()),
            ..Default::default()
        };
        self.send(delta, None, None, None);
    }

    /// Reasoning text, under both llama.cpp's and vLLM's field names.
    fn reasoning(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        let delta = DeltaContent {
            role: self.take_role(),
            reasoning_content: Some(text.to_string()),
            reasoning: Some(text.to_string()),
            ..Default::default()
        };
        self.send(delta, None, None, None);
    }

    /// A call's first chunk carries its id, type and name (OpenAI's and
    /// vLLM's shape); its arguments follow in [`Self::tool_args`] chunks.
    fn tool_start(&mut self, index: usize, id: &str, name: &str) {
        let delta = DeltaContent {
            role: self.take_role(),
            tool_calls: Some(json!([{
                "index": index,
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": ""},
            }])),
            ..Default::default()
        };
        self.send(delta, None, None, None);
    }

    /// More of call `index`'s arguments.
    fn tool_args(&mut self, index: usize, delta: &str) {
        let delta = DeltaContent {
            tool_calls: Some(json!([{
                "index": index,
                "function": {"arguments": delta},
            }])),
            ..Default::default()
        };
        self.send(delta, None, None, None);
    }

    /// Send one piece of the shared post-processing's output; reasoning
    /// only when `show_reasoning`.
    fn piece(&mut self, piece: ChatPiece<'_>, show_reasoning: bool) {
        match piece {
            ChatPiece::Reasoning(text) if show_reasoning => self.reasoning(text),
            ChatPiece::Reasoning(_) => {}
            ChatPiece::Text(text) => self.text(text),
            ChatPiece::CallStart { index, id, name } => self.tool_start(index, id, name),
            ChatPiece::CallArgs { index, delta } => self.tool_args(index, delta),
        }
    }

    fn finish(&mut self, reason: &str, usage: &UsageInfo, timings: &crate::types::Timings) {
        self.send(
            DeltaContent::default(),
            Some(reason.to_string()),
            Some(usage),
            Some(timings),
        );
        let _ = self.tx.send("[DONE]".to_string());
    }

    fn error(&mut self, message: &str) {
        let payload = json!({
            "error": {
                "message": message,
                "type": "server_error",
            }
        });
        let _ = self.tx.send(payload.to_string());
        let _ = self.tx.send("[DONE]".to_string());
    }

    fn take_role(&mut self) -> Option<String> {
        if self.role_sent {
            None
        } else {
            self.role_sent = true;
            Some("assistant".to_string())
        }
    }

    fn send(
        &self,
        delta: DeltaContent,
        finish_reason: Option<String>,
        usage: Option<&UsageInfo>,
        timings: Option<&crate::types::Timings>,
    ) {
        let chunk =
            ChatCompletionChunk::new(self.id.clone(), self.model.clone(), delta, finish_reason);
        let mut value = serde_json::to_value(&chunk).unwrap_or_default();
        if let Some(usage) = usage {
            value["usage"] = serde_json::to_value(usage).unwrap_or_default();
        }
        if let Some(timings) = timings {
            // llama-server puts its timings on the final chunk.
            value["timings"] = serde_json::to_value(timings).unwrap_or_default();
        }
        let _ = self.tx.send(value.to_string());
    }
}

/// How a streamed chat response reports reasoning and echoes the prefill.
struct ChatStreamOutput {
    reasoning: ReasoningOutput,
    /// The continued assistant message to repeat first (vLLM's `echo`).
    echo: Option<String>,
}

/// Run one streaming chat completion on the blocking pool.
#[allow(clippy::too_many_arguments)]
fn stream_chat_completion(
    engine: &Engine,
    model: &str,
    id: &str,
    messages: &[ChatMessage],
    tools: Option<&[Tool]>,
    options: &GenerationOptions,
    output: ChatStreamOutput,
    cancel: &AtomicBool,
    tx: tokio::sync::mpsc::UnboundedSender<String>,
    metrics: &ServerMetrics,
) {
    let mut emitter = ChatStreamEmitter::new(id.to_string(), model.to_string(), tx);
    let show_reasoning = output.reasoning != ReasoningOutput::Hidden;
    if let Some(prefix) = &output.echo {
        emitter.text(prefix);
    }
    let result = stream_chat_output(
        engine,
        messages,
        tools,
        options,
        cancel,
        output.reasoning.splits(true),
        OPENAI_CALL_IDS,
        &mut |piece| emitter.piece(piece, show_reasoning),
    );

    match result {
        Ok((completion, calls)) => {
            metrics.record_generation(&completion.usage);
            emitter.finish(
                completion.stop_type.chat_finish_reason(!calls.is_empty()),
                &completion.usage,
                &completion.timings,
            );
        }
        Err(error) => {
            tracing::error!("streaming completion failed: {error}");
            emitter.error("generation failed");
        }
    }
}

/// One Anthropic SSE event.
struct AnthropicSse {
    event: &'static str,
    data: String,
}

/// `POST /v1/messages` — Anthropic Messages API.
/// `POST /v1/messages/count_tokens` — Anthropic's token-count endpoint.
///
/// Accepts the same body as `/v1/messages` and reports the input token count
/// without running the model, so a client can budget a context before sending.
/// The count is the engine's own chat-token accounting (`count_chat_tokens`),
/// which is the same figure the request would consume as its prompt.
async fn messages_count_tokens(
    State(state): State<AppState>,
    Json(req): Json<MessagesRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let (_, engine) = select_model(&state, &req.model).await?;
    let defaults = engine.sampling_defaults(SamplingProfile::OpenAi);
    let (messages, tools, options) =
        anthropic::to_chat_request(&req, &defaults).map_err(ApiError::bad_request)?;
    let count = engine
        .count_chat_tokens(&messages, tools.as_deref(), &options)
        .map_err(ApiError::from)?;
    Ok(Json(json!({ "input_tokens": count })))
}

async fn messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<MessagesRequest>,
) -> Result<Response, ApiError> {
    let (resolved_model, engine) = select_model(&state, &req.model).await?;
    // Anthropic's defaults are the neutral OpenAI-family ones.
    let defaults = engine.sampling_defaults(SamplingProfile::OpenAi);
    let (messages, tools, mut options) =
        anthropic::to_chat_request(&req, &defaults).map_err(ApiError::bad_request)?;
    options.cache_scope = Some(crate::affinity::cache_scope(
        &headers,
        options.context_window,
    ));
    let tools_disabled = matches!(
        req.tool_choice
            .as_ref()
            .and_then(|choice| choice.get("type"))
            .and_then(|kind| kind.as_str()),
        Some("none")
    );
    let tools = if tools_disabled { None } else { tools };
    let model = if req.model.is_empty() {
        resolved_model
    } else {
        req.model.clone()
    };

    if req.stream.unwrap_or(false) {
        let id = format!("msg_{}", Uuid::new_v4().simple());
        let input_tokens = engine
            .count_chat_tokens(&messages, tools.as_deref(), &options)
            .unwrap_or(0);
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<AnthropicSse>();
        let cancel = CancelOnDrop::new();
        let flag = cancel.flag();
        state
            .metrics
            .streamed_requests
            .fetch_add(1, Ordering::Relaxed);
        tokio::task::spawn_blocking({
            let engine = Arc::clone(&engine);
            let model = model.clone();
            let id = id.clone();
            let tools = tools.clone();
            let metrics = Arc::clone(&state.metrics);
            move || {
                stream_messages(
                    &engine,
                    &model,
                    &id,
                    &messages,
                    tools.as_deref(),
                    &options,
                    &flag,
                    tx,
                    &metrics,
                    input_tokens,
                );
            }
        });
        let events = stream::unfold(
            (rx, cancel),
            |(mut rx, cancel)| async move {
                rx.recv().await.map(|event| {
                    (
                        Ok::<Event, Infallible>(
                            Event::default().event(event.event).data(event.data),
                        ),
                        (rx, cancel),
                    )
                })
            },
        );
        return Ok(Sse::new(events).into_response());
    }

    let tool_call_format = if tools.is_some() {
        engine
            .tool_call_format_for_request(&messages, tools.as_deref(), &options)
            .map_err(ApiError::from)?
    } else {
        engine.tool_call_format()
    };
    let cancel = CancelOnDrop::new();
    let completion = tokio::task::spawn_blocking({
        let engine = Arc::clone(&engine);
        let tools = tools.clone();
        let flag = cancel.flag();
        move || engine.complete_chat_cancellable_detailed(&messages, tools.as_deref(), &options, &flag)
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?
    .map_err(ApiError::from)?;
    state.metrics.record_generation(&completion.usage);

    let answer = chat_answer(
        &engine,
        &completion,
        true,
        tools.as_deref(),
        tool_call_format,
        ANTHROPIC_CALL_IDS,
    );
    let id = format!("msg_{}", Uuid::new_v4().simple());
    let response = anthropic::response(
        id,
        model,
        &answer.prose,
        answer.reasoning.as_deref(),
        &answer.calls,
        &completion,
    );
    Ok(Json(response).into_response())
}

/// Writes Anthropic streaming events to the SSE channel.
///
/// Content blocks open and close as the output streams, in llama-server's
/// order: a `thinking` block for the model's reasoning, then a `text` block
/// for prose and one `tool_use` block per call, whose input arrives as
/// `input_json_delta`s while the model writes it.  Prose after a call opens
/// a new `text` block.
struct AnthropicStreamEmitter {
    id: String,
    model: String,
    tx: tokio::sync::mpsc::UnboundedSender<AnthropicSse>,
    /// The open content block's index and kind.
    open: Option<(usize, AnthropicBlock)>,
    /// Whether a `text` or `tool_use` block has opened: reasoning after the
    /// answer has begun is not sent.
    answered: bool,
    next_index: usize,
}

/// The kind of an open Anthropic content block.
enum AnthropicBlock {
    Thinking,
    Text,
    /// Tool call `call`'s `tool_use` block.  Its arguments stream as they
    /// arrive once they start a JSON value (`streaming`); arguments the
    /// model wrote as plain text are `held` and sent as one JSON string when
    /// the block closes, like the non-streaming `tool_use.input`.
    Tool {
        call: usize,
        streaming: bool,
        held: Option<String>,
    },
}

impl AnthropicStreamEmitter {
    fn new(
        id: String,
        model: String,
        tx: tokio::sync::mpsc::UnboundedSender<AnthropicSse>,
    ) -> Self {
        Self {
            id,
            model,
            tx,
            open: None,
            answered: false,
            next_index: 0,
        }
    }

    fn send(&self, event: &'static str, data: serde_json::Value) {
        let _ = self.tx.send(AnthropicSse {
            event,
            data: data.to_string(),
        });
    }

    fn message_start(&mut self, input_tokens: u32) {
        self.send(
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": self.id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                }
            }),
        );
    }

    /// Open a content block, closing the one before it.
    fn open_block(&mut self, kind: AnthropicBlock, content_block: serde_json::Value) -> usize {
        self.close_block();
        let index = self.next_index;
        self.next_index += 1;
        self.send(
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": index,
                "content_block": content_block,
            }),
        );
        self.open = Some((index, kind));
        index
    }

    /// Close the open content block: a `thinking` block ends with an empty
    /// `signature_delta` (llama-server's extended thinking stream), and a
    /// `tool_use` block with arguments held as text sends them now.
    fn close_block(&mut self) {
        let Some((index, kind)) = self.open.take() else {
            return;
        };
        match kind {
            AnthropicBlock::Thinking => {
                self.delta(index, json!({"type": "signature_delta", "signature": ""}))
            }
            AnthropicBlock::Tool {
                held: Some(held), ..
            } => self.delta(
                index,
                json!({
                    "type": "input_json_delta",
                    "partial_json": anthropic::tool_arguments_json(&held),
                }),
            ),
            _ => {}
        }
        self.send(
            "content_block_stop",
            json!({"type": "content_block_stop", "index": index}),
        );
    }

    fn delta(&self, index: usize, delta: serde_json::Value) {
        self.send(
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": index,
                "delta": delta,
            }),
        );
    }

    /// Reasoning text, in a `thinking` block.
    fn thinking(&mut self, text: &str) {
        if text.is_empty() || self.answered {
            return;
        }
        let index = match self.open {
            Some((index, AnthropicBlock::Thinking)) => index,
            _ => self.open_block(
                AnthropicBlock::Thinking,
                json!({"type": "thinking", "thinking": ""}),
            ),
        };
        self.delta(index, json!({"type": "thinking_delta", "thinking": text}));
    }

    fn text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        let index = match self.open {
            Some((index, AnthropicBlock::Text)) => index,
            _ => {
                self.answered = true;
                self.open_block(AnthropicBlock::Text, json!({"type": "text", "text": ""}))
            }
        };
        self.delta(index, json!({"type": "text_delta", "text": text}));
    }

    fn tool_start(&mut self, index: usize, id: &str, name: &str) {
        self.answered = true;
        self.open_block(
            AnthropicBlock::Tool {
                call: index,
                streaming: false,
                held: None,
            },
            json!({"type": "tool_use", "id": id, "name": name, "input": {}}),
        );
    }

    fn tool_args(&mut self, index: usize, delta: &str) {
        let block = match &mut self.open {
            Some((
                block,
                AnthropicBlock::Tool {
                    call,
                    streaming,
                    held,
                },
            )) if *call == index => {
                if !*streaming && held.is_none() {
                    if delta.trim_start().starts_with(['{', '[']) {
                        *streaming = true;
                    } else {
                        *held = Some(String::new());
                    }
                }
                if let Some(held) = held {
                    held.push_str(delta);
                    return;
                }
                *block
            }
            _ => {
                tracing::debug!("arguments for tool call {index} after its block closed");
                return;
            }
        };
        self.delta(
            block,
            json!({"type": "input_json_delta", "partial_json": delta}),
        );
    }

    /// Send one piece of the shared post-processing's output.
    fn piece(&mut self, piece: ChatPiece<'_>) {
        match piece {
            ChatPiece::Reasoning(text) => self.thinking(text),
            ChatPiece::Text(text) => self.text(text),
            ChatPiece::CallStart { index, id, name } => self.tool_start(index, id, name),
            ChatPiece::CallArgs { index, delta } => self.tool_args(index, delta),
        }
    }

    fn finish(&mut self, has_calls: bool, completion: &crate::engine::Completion) {
        self.close_block();
        let stop_reason = anthropic::stop_reason(completion.stop_type, has_calls);
        let usage = anthropic::MessagesUsage::from_completion(completion);
        self.send(
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason,
                    "stop_sequence": completion.stop_sequence,
                },
                // The final counts, including the prompt tokens read from
                // cache (Anthropic's SDKs take the latest value of each).
                "usage": usage,
            }),
        );
        self.send("message_stop", json!({"type": "message_stop"}));
    }

    fn error(&mut self, message: &str) {
        self.send(
            "error",
            json!({
                "type": "error",
                "error": {"type": "api_error", "message": message},
            }),
        );
    }
}

/// Run one streaming Anthropic message on the blocking pool.
fn stream_messages(
    engine: &Engine,
    model: &str,
    id: &str,
    messages: &[ChatMessage],
    tools: Option<&[Tool]>,
    options: &GenerationOptions,
    cancel: &AtomicBool,
    tx: tokio::sync::mpsc::UnboundedSender<AnthropicSse>,
    metrics: &ServerMetrics,
    input_tokens: u32,
) {
    let mut emitter = AnthropicStreamEmitter::new(id.to_string(), model.to_string(), tx);
    emitter.message_start(input_tokens);
    let result = stream_chat_output(
        engine,
        messages,
        tools,
        options,
        cancel,
        true,
        ANTHROPIC_CALL_IDS,
        &mut |piece| emitter.piece(piece),
    );
    match result {
        Ok((completion, calls)) => {
            metrics.record_generation(&completion.usage);
            emitter.finish(!calls.is_empty(), &completion);
        }
        Err(error) => {
            tracing::error!("streaming Anthropic message failed: {error}");
            emitter.error("generation failed");
        }
    }
}

/// `POST /v1/completions` — legacy (non-chat) text completion.
async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let (model, engine) = select_model(
        &state,
        body.get("model")
            .and_then(|v| v.as_str())
            .unwrap_or_default(),
    )
    .await?;
    let prompt = body
        .get("prompt")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::bad_request("Missing required field 'prompt'"))?
        .to_string();

    // The OpenAI route's defaults, then the request's own sampling fields
    // (`max_tokens` absent or -1 runs to end of sequence).
    let sampling: crate::types::SamplingParams =
        serde_json::from_value(body.clone()).map_err(|e| ApiError::bad_request(e.to_string()))?;
    let mut options =
        GenerationOptions::with_defaults(&engine.sampling_defaults(SamplingProfile::OpenAi));
    sampling
        .apply(&mut options)
        .map_err(ApiError::bad_request)?;
    options.cache_scope = Some(crate::affinity::cache_scope(&headers, None));

    let messages = vec![ChatMessage::text("user".to_string(), prompt)];

    let cancel = CancelOnDrop::new();
    let completion = tokio::task::spawn_blocking({
        let engine = Arc::clone(&engine);
        let flag = cancel.flag();
        move || engine.complete_chat_cancellable_detailed(&messages, None, &options, &flag)
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?
    .map_err(ApiError::from)?;

    let usage = &completion.usage;
    state.metrics.record_generation(usage);
    let created = unix_time();

    Ok(Json(json!({
        "id": format!("cmpl-{}", Uuid::new_v4().simple()),
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{
            "text": completion.text,
            "index": 0,
            "finish_reason": completion.finish_reason()
        }],
        "usage": usage,
        "timings": completion.timings,
    })))
}

/// `POST /v1/embeddings` — dense text embeddings.
async fn embeddings(
    State(state): State<AppState>,
    Extension(lease): Extension<Arc<crate::job_admission::JobLease>>,
    headers: HeaderMap,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Response, ApiError> {
    let (model, engine) = select_model(&state, &req.model).await?;
    if let Some(owner) = headers.get(JOB_OWNER_HEADER) {
        let matches = state.models.list().iter().any(|loaded| {
            Arc::ptr_eq(&loaded.engine, &engine)
                && owner.as_bytes() == loaded.embedding_job.owner.as_bytes()
        });
        if !matches {
            return Err(ApiError::new(StatusCode::CONFLICT, "job model ownership changed", "job_owner_changed"));
        }
    }
    let texts: Vec<String> = req.input.into_vec();
    if texts.len() > 2048 {
        return Err(ApiError::new(StatusCode::PAYLOAD_TOO_LARGE, "too many embedding inputs", "job_input_limit"));
    }
    if headers.contains_key(JOB_OWNER_HEADER) && texts.len() > 32 {
        return Err(ApiError::new(StatusCode::PAYLOAD_TOO_LARGE, "too many inputs in embedding subbatch", "job_input_limit"));
    }

    // Like generation, stop embedding a batch once the requester is gone.
    let cancel = CancelOnDrop::new();
    let (vectors, prompt_tokens) = tokio::task::spawn_blocking({
        let engine = Arc::clone(&engine);
        let flag = cancel.flag();
        move || {
            let _lease = lease;
            engine.embed_with_usage_cancellable(&texts, &flag)
        }
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?
    .map_err(ApiError::from)?;
    state
        .metrics
        .embed_tokens
        .fetch_add(prompt_tokens as u64, Ordering::Relaxed);

    let data = vectors
        .into_iter()
        .enumerate()
        .map(|(i, embedding)| EmbeddingData {
            object: "embedding".to_string(),
            embedding,
            index: i as u32,
        })
        .collect();

    let response = EmbeddingResponse {
        object: "list".to_string(),
        data,
        model,
        usage: UsageInfo {
            prompt_tokens,
            completion_tokens: 0,
            total_tokens: prompt_tokens,
            ..Default::default()
        },
    };
    let limit = if headers.contains_key(JOB_OWNER_HEADER) { 2 << 20 } else { 8 << 20 };
    crate::job_admission::json_response(&response, limit).map_err(|_| {
        ApiError::new(StatusCode::PAYLOAD_TOO_LARGE, "embedding response exceeds job buffer limit", "job_response_limit")
    })
}

/// `POST /v1/audio/transcriptions` — OpenAI-compatible Whisper STT.
///
/// Multipart form fields: `file` (required, WAV), `language` (optional
/// two-letter code), `response_format` (`json` default, or `text`), and
/// `model` (accepted and ignored — the loaded whisper model is used).
async fn transcriptions(
    State(state): State<AppState>,
    Extension(lease): Extension<Arc<crate::job_admission::JobLease>>,
    headers: HeaderMap,
    mut multipart: axum::extract::Multipart,
) -> Result<Response, ApiError> {
    let Some(whisper) = state.whisper.clone() else {
        return Err(ApiError::bad_request(
            "no whisper model is loaded — start the server with --whisper-model",
        ));
    };
    if headers.get(JOB_OWNER_HEADER).is_some_and(|owner| owner.as_bytes() != whisper.job_owner.as_bytes()) {
        return Err(ApiError::new(StatusCode::CONFLICT, "job model ownership changed", "job_owner_changed"));
    }

    let mut file: Option<axum::body::Bytes> = None;
    let mut language: Option<String> = None;
    let mut response_format = "json".to_string();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ApiError::new(e.status(), format!("invalid multipart body: {e}"), "invalid_request_error"))?
    {
        match field.name().unwrap_or_default() {
            "file" => {
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|e| ApiError::new(e.status(), format!("file upload failed: {e}"), "invalid_request_error"))?;
                file = Some(bytes);
            }
            "language" => {
                language = Some(field.text().await.map_err(|e| ApiError::new(e.status(), format!("invalid language field: {e}"), "invalid_request_error"))?);
            }
            "response_format" => {
                response_format = field.text().await.map_err(|e| ApiError::new(e.status(), format!("invalid response_format field: {e}"), "invalid_request_error"))?;
            }
            // `model`, `prompt`, `temperature`, … accepted and ignored.
            _ => {
                field.bytes().await.map_err(|e| ApiError::new(e.status(), format!("invalid multipart field: {e}"), "invalid_request_error"))?;
            }
        }
    }
    let file = file.ok_or_else(|| ApiError::bad_request("missing required field 'file'"))?;

    let transcription = run_transcription_job(whisper.admission.clone(), move |cancel| {
        let _lease = lease;
        whisper.transcribe_wav_cancellable(&file, language.as_deref(), false, cancel)
    })
    .await?;

    if response_format == "text" {
        if transcription.text.len() > 2 << 20 {
            return Err(ApiError::new(StatusCode::PAYLOAD_TOO_LARGE, "transcription response exceeds job buffer limit", "job_response_limit"));
        }
        return Ok(transcription.text.into_response());
    }
    crate::job_admission::json_response(&json!({
        "text": transcription.text,
        "duration": transcription.duration,
        "language": transcription.language,
    }), 2 << 20).map_err(|_| {
        ApiError::new(StatusCode::PAYLOAD_TOO_LARGE, "transcription response exceeds job buffer limit", "job_response_limit")
    })
}

async fn run_transcription_job<T, F>(
    admission: Arc<tokio::sync::Semaphore>,
    job: F,
) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: FnOnce(&AtomicBool) -> crate::error::Result<T> + Send + 'static,
{
    let cancel = CancelOnDrop::new();
    let permit = admission
        .acquire_owned()
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;
    let flag = cancel.flag();
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        if flag.load(Ordering::Relaxed) {
            return Err(JoshuaError::Cancelled);
        }
        job(&flag)
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?
    .map_err(ApiError::from)
}

#[cfg(test)]
mod transcription_cancellation_tests {
    use super::*;
    use tokio::sync::{oneshot, Semaphore};
    use tokio::time::{timeout, Duration};

    struct Payload(Option<oneshot::Sender<()>>);

    impl Drop for Payload {
        fn drop(&mut self) {
            if let Some(sender) = self.0.take() {
                let _ = sender.send(());
            }
        }
    }

    #[tokio::test]
    async fn transcription_cancelled_while_waiting_drops_payload_without_running() {
        let admission = Arc::new(Semaphore::new(1));
        let holder = admission.clone().acquire_owned().await.unwrap();
        let ran = Arc::new(AtomicBool::new(false));
        let job_ran = ran.clone();
        let (dropped_tx, dropped_rx) = oneshot::channel();
        let payload = Payload(Some(dropped_tx));
        let mut request = Box::pin(run_transcription_job(admission.clone(), move |_| {
            let _payload = payload;
            job_ran.store(true, Ordering::Relaxed);
            Ok(())
        }));
        assert!(futures_util::poll!(request.as_mut()).is_pending());
        drop(request);
        timeout(Duration::from_secs(2), dropped_rx)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(admission.available_permits(), 0);
        assert!(!ran.load(Ordering::Relaxed));
        drop(holder);
        assert_eq!(admission.available_permits(), 1);
    }

    #[tokio::test]
    async fn transcription_drop_cancels_active_job_but_keeps_permit_until_exit() {
        let admission = Arc::new(Semaphore::new(1));
        let (started_tx, started_rx) = oneshot::channel();
        let (cancelled_tx, cancelled_rx) = oneshot::channel();
        let (finish_tx, finish_rx) = std::sync::mpsc::channel();
        let (dropped_tx, dropped_rx) = oneshot::channel();
        let payload = Payload(Some(dropped_tx));
        let request = tokio::spawn(run_transcription_job(admission.clone(), move |cancel| {
            let _payload = payload;
            started_tx.send(()).unwrap();
            while !cancel.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(1));
            }
            cancelled_tx.send(()).unwrap();
            finish_rx.recv_timeout(Duration::from_secs(2)).unwrap();
            Err::<(), _>(JoshuaError::Cancelled)
        }));
        timeout(Duration::from_secs(2), started_rx)
            .await
            .unwrap()
            .unwrap();
        request.abort();
        assert!(matches!(request.await, Err(error) if error.is_cancelled()));
        timeout(Duration::from_secs(2), cancelled_rx)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(admission.available_permits(), 0);
        finish_tx.send(()).unwrap();
        timeout(Duration::from_secs(2), dropped_rx)
            .await
            .unwrap()
            .unwrap();
        assert!(run_transcription_job(admission, |_| Ok(())).await.is_ok());
    }

    #[test]
    fn transcription_cancelled_in_blocking_queue_never_starts() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        runtime.block_on(async {
            let (started_tx, started_rx) = oneshot::channel();
            let (finish_tx, finish_rx) = std::sync::mpsc::channel();
            let blocker = tokio::task::spawn_blocking(move || {
                started_tx.send(()).unwrap();
                finish_rx.recv_timeout(Duration::from_secs(2)).unwrap();
            });
            started_rx.await.unwrap();
            let admission = Arc::new(Semaphore::new(1));
            let ran = Arc::new(AtomicBool::new(false));
            let job_ran = ran.clone();
            let (dropped_tx, dropped_rx) = oneshot::channel();
            let payload = Payload(Some(dropped_tx));
            let mut request = Box::pin(run_transcription_job(admission.clone(), move |_| {
                let _payload = payload;
                job_ran.store(true, Ordering::Relaxed);
                Ok(())
            }));
            assert!(futures_util::poll!(request.as_mut()).is_pending());
            assert_eq!(admission.available_permits(), 0);
            drop(request);
            assert_eq!(admission.available_permits(), 0);
            finish_tx.send(()).unwrap();
            blocker.await.unwrap();
            timeout(Duration::from_secs(2), dropped_rx)
                .await
                .unwrap()
                .unwrap();
            let _permit = timeout(Duration::from_secs(2), admission.acquire())
                .await
                .unwrap()
                .unwrap();
            assert!(!ran.load(Ordering::Relaxed));
        });
    }

    #[tokio::test]
    async fn transcription_success_error_and_panic_release_permit_and_payload() {
        let admission = Arc::new(Semaphore::new(1));
        for outcome in 0..3 {
            let (dropped_tx, dropped_rx) = oneshot::channel();
            let payload = Payload(Some(dropped_tx));
            let result = run_transcription_job(admission.clone(), move |_| {
                let _payload = payload;
                match outcome {
                    0 => Ok("complete"),
                    1 => Err(JoshuaError::InvalidRequest("invalid WAV".into())),
                    _ => panic!("simulate decoder panic"),
                }
            })
            .await;
            match outcome {
                0 => assert_eq!(result.ok(), Some("complete")),
                1 => assert_eq!(result.unwrap_err().status, StatusCode::BAD_REQUEST),
                _ => assert_eq!(
                    result.unwrap_err().status,
                    StatusCode::INTERNAL_SERVER_ERROR
                ),
            }
            dropped_rx.await.unwrap();
            assert_eq!(admission.available_permits(), 1);
        }
    }

    #[tokio::test]
    async fn transcription_http_disconnect_cancels_blocking_job() {
        use tokio::io::AsyncWriteExt;

        let admission = Arc::new(Semaphore::new(1));
        let (started_tx, started_rx) = oneshot::channel();
        let started = Arc::new(Mutex::new(Some(started_tx)));
        let (cancelled_tx, cancelled_rx) = oneshot::channel();
        let cancelled = Arc::new(Mutex::new(Some(cancelled_tx)));
        let (dropped_tx, dropped_rx) = oneshot::channel();
        let payload = Arc::new(Mutex::new(Some(Payload(Some(dropped_tx)))));
        let app = Router::new().route(
            "/transcription",
            post({
                let admission = admission.clone();
                move || {
                    let admission = admission.clone();
                    let started = started.clone();
                    let cancelled = cancelled.clone();
                    let payload = payload.clone();
                    async move {
                        run_transcription_job(admission, move |cancel| {
                            let _payload = payload.lock().unwrap().take().unwrap();
                            started.lock().unwrap().take().unwrap().send(()).unwrap();
                            let deadline = std::time::Instant::now() + Duration::from_secs(2);
                            while !cancel.load(Ordering::Relaxed) {
                                assert!(
                                    std::time::Instant::now() < deadline,
                                    "HTTP disconnect did not cancel job"
                                );
                                std::thread::sleep(Duration::from_millis(1));
                            }
                            cancelled.lock().unwrap().take().unwrap().send(()).unwrap();
                            Err::<String, _>(JoshuaError::Cancelled)
                        })
                        .await
                    }
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let mut client = tokio::net::TcpStream::connect(address).await.unwrap();
        client
            .write_all(
                b"POST /transcription HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n",
            )
            .await
            .unwrap();
        timeout(Duration::from_secs(2), started_rx)
            .await
            .unwrap()
            .unwrap();
        drop(client);
        timeout(Duration::from_secs(3), cancelled_rx)
            .await
            .unwrap()
            .unwrap();
        timeout(Duration::from_secs(3), dropped_rx)
            .await
            .unwrap()
            .unwrap();
        let _permit = timeout(Duration::from_secs(2), admission.acquire())
            .await
            .unwrap()
            .unwrap();
        server.abort();
    }
}
// ─── API error type ───────────────────────────────────────────────────────────

/// Internal helper that maps [`JoshuaError`] to HTTP responses.
pub struct ApiError {
    status: StatusCode,
    body: ErrorResponse,
}

impl ApiError {
    pub(crate) fn new(status: StatusCode, msg: impl Into<String>, error_type: &str) -> Self {
        Self {
            status,
            body: ErrorResponse::new(msg, error_type),
        }
    }

    pub(crate) fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            body: ErrorResponse::invalid_request(msg),
        }
    }

    /// `404` for a model this server cannot serve, saying why (see
    /// [`ErrorResponse::model_not_found`]).
    pub(crate) fn model_not_found(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            body: ErrorResponse::model_not_found(msg),
        }
    }

    /// `404` for an id that names no model (see
    /// [`ErrorResponse::unknown_model`]).
    pub(crate) fn unknown_model(id: &str) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            body: ErrorResponse::unknown_model(id),
        }
    }

    /// Internal failure: the detail is logged server-side, and the client
    /// receives only a generic message (no internal strings / panic text).
    pub(crate) fn internal(msg: impl Into<String>) -> Self {
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
        assert!(!api_keys_match(b"sek", b"sekret"));
        assert!(!api_keys_match(b"sekretsekret", b"sekret"));
        assert!(!api_keys_match(b"", b"sekret"));
    }

    #[test]
    fn api_keys_match_handles_an_empty_expected_key() {
        assert!(api_keys_match(b"", b""));
        assert!(!api_keys_match(b"a", b""));
    }

    use super::{
        AnthropicSse, AnthropicStreamEmitter, ChatOutput, ChatPiece, ChatStreamEmitter, UsageInfo,
        ANTHROPIC_CALL_IDS, OPENAI_CALL_IDS,
    };
    use crate::engine::{Completion, StopType};
    use crate::reasoning::Split;
    use crate::tools::{ParsedToolCall, ToolCallFormat};
    use serde_json::Value;

    /// Feed `pieces` of answer text (or of reasoning, for pieces starting
    /// `reasoning:`) through the shared chat post-processing in `format`
    /// with tools offered, into `emit`.  Returns the calls it read.
    fn feed(
        format: ToolCallFormat,
        id_prefix: &'static str,
        pieces: &[&str],
        emit: &mut dyn FnMut(ChatPiece<'_>),
    ) -> Vec<ParsedToolCall> {
        let mut output = ChatOutput::new(format, Some(&[]), id_prefix);
        for piece in pieces {
            let split = match piece.strip_prefix("reasoning:") {
                Some(reasoning) => Split {
                    reasoning: reasoning.to_string(),
                    ..Default::default()
                },
                None => Split {
                    content: piece.to_string(),
                    ..Default::default()
                },
            };
            output.push(&split, emit);
        }
        output.finish(emit)
    }

    fn openai_chunks(format: ToolCallFormat, pieces: &[&str], stop_type: StopType) -> Vec<Value> {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut emitter = ChatStreamEmitter::new("chatcmpl-1".into(), "m".into(), tx);
        let calls = feed(format, OPENAI_CALL_IDS, pieces, &mut |piece| {
            emitter.piece(piece, true)
        });
        emitter.finish(
            stop_type.chat_finish_reason(!calls.is_empty()),
            &UsageInfo::default(),
            &Default::default(),
        );
        drop(emitter);
        let mut chunks = Vec::new();
        while let Ok(data) = rx.try_recv() {
            if data != "[DONE]" {
                chunks.push(serde_json::from_str(&data).unwrap());
            }
        }
        chunks
    }

    /// The content and per-call `(id, name, arguments)` a client rebuilds
    /// from OpenAI stream chunks.
    fn openai_rebuild(chunks: &[Value]) -> (String, Vec<(String, String, String)>) {
        let mut content = String::new();
        let mut calls: Vec<(String, String, String)> = Vec::new();
        for chunk in chunks {
            let delta = &chunk["choices"][0]["delta"];
            if let Some(text) = delta["content"].as_str() {
                content.push_str(text);
            }
            for call in delta["tool_calls"].as_array().into_iter().flatten() {
                let index = call["index"].as_u64().unwrap() as usize;
                if index == calls.len() {
                    assert_eq!(call["type"], "function");
                    calls.push((
                        call["id"].as_str().unwrap().to_string(),
                        call["function"]["name"].as_str().unwrap().to_string(),
                        String::new(),
                    ));
                } else {
                    assert!(call.get("id").is_none(), "only the first chunk has the id");
                }
                calls[index]
                    .2
                    .push_str(call["function"]["arguments"].as_str().unwrap());
            }
        }
        (content, calls)
    }

    fn finish_reason(chunks: &[Value]) -> &str {
        chunks.last().unwrap()["choices"][0]["finish_reason"]
            .as_str()
            .unwrap()
    }

    /// Tool calls stream as OpenAI deltas while the model writes them: the
    /// call's id, type and name once, then its arguments in pieces, with no
    /// markup in the content.
    #[test]
    fn openai_stream_sends_tool_call_deltas_as_they_are_generated() {
        let pieces = [
            "Let me look.",
            "<tool_call>\n{\"name\": \"get_",
            "weather\", \"arguments\": {\"ci",
            "ty\": \"Par",
            "is\"}}\n</tool_call>",
        ];
        let chunks = openai_chunks(ToolCallFormat::Generic, &pieces, StopType::Eos);
        let (content, calls) = openai_rebuild(&chunks);
        assert_eq!(content, "Let me look.");
        assert_eq!(calls.len(), 1);
        assert!(calls[0].0.starts_with("call_"));
        assert_eq!(calls[0].1, "get_weather");
        let args: Value = serde_json::from_str(&calls[0].2).unwrap();
        assert_eq!(args, serde_json::json!({"city": "Paris"}));
        // The arguments arrive over several chunks, before the call closes.
        let arg_chunks = chunks
            .iter()
            .filter(|c| c["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] != "")
            .count();
        assert!(arg_chunks >= 3, "{chunks:?}");
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(finish_reason(&chunks), "tool_calls");
        for chunk in &chunks {
            let text = chunk["choices"][0]["delta"]["content"]
                .as_str()
                .unwrap_or("");
            assert!(!text.contains('<'), "markup leaked: {text}");
        }
    }

    /// A call cut off by the token cap is still streamed as far as it got,
    /// but the response ends with `length`, not `tool_calls`.
    #[test]
    fn openai_stream_reports_length_for_a_cut_off_call() {
        let pieces = ["<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Pa"];
        let chunks = openai_chunks(ToolCallFormat::Generic, &pieces, StopType::Limit);
        let (_, calls) = openai_rebuild(&chunks);
        assert_eq!(calls.len(), 1);
        assert_eq!(finish_reason(&chunks), "length");
    }

    /// A model's own call ids (Kimi's `functions.name:index`) are reported
    /// as the call ids, since its template expects them back.
    #[test]
    fn openai_stream_keeps_the_model_call_ids() {
        let text = "<|tool_calls_section_begin|><|tool_call_begin|>functions.f:0\
                    <|tool_call_argument_begin|>{\"a\": 1}<|tool_call_end|>\
                    <|tool_call_begin|>functions.g:1<|tool_call_argument_begin|>{}<|tool_call_end|>\
                    <|tool_calls_section_end|>";
        let chunks = openai_chunks(ToolCallFormat::KimiK2, &[text], StopType::Eos);
        let (content, calls) = openai_rebuild(&chunks);
        assert_eq!(content, "");
        assert_eq!(calls.len(), 2);
        assert_eq!(
            (calls[0].0.as_str(), calls[0].1.as_str()),
            ("functions.f:0", "f")
        );
        assert_eq!(
            (calls[1].0.as_str(), calls[1].1.as_str()),
            ("functions.g:1", "g")
        );
    }

    /// Mistral's templates take back only 9-alphanumeric call ids, so those
    /// are the ids minted for a Mistral model's calls, streamed or not.
    #[test]
    fn mistral_calls_get_ids_their_template_accepts() {
        let text = "[TOOL_CALLS][{\"name\": \"f\", \"arguments\": {}}, {\"name\": \"g\", \"arguments\": {}}]";
        let chunks = openai_chunks(ToolCallFormat::Mistral, &[text], StopType::Eos);
        let (_, calls) = openai_rebuild(&chunks);
        assert_eq!(calls.len(), 2);
        for (id, _, _) in &calls {
            assert_eq!(id.len(), 9, "{id}");
            assert!(id.chars().all(|c| c.is_ascii_alphanumeric()), "{id}");
        }
        assert_ne!(calls[0].0, calls[1].0);
        // The calls the stream returns carry the ids it sent.
        let mut sent = Vec::new();
        let returned = feed(ToolCallFormat::Mistral, ANTHROPIC_CALL_IDS, &[text], &mut |piece| {
            if let ChatPiece::CallStart { id, .. } = piece {
                sent.push(id.to_string());
            }
        });
        let returned: Vec<String> = returned.into_iter().map(|c| c.id.unwrap()).collect();
        assert_eq!(returned, sent);
        assert!(sent.iter().all(|id| id.len() == 9));
    }

    #[test]
    fn openai_stream_without_tool_markup_is_plain_content() {
        let chunks = openai_chunks(
            ToolCallFormat::Generic,
            &["Hello", ", world"],
            StopType::Eos,
        );
        let (content, calls) = openai_rebuild(&chunks);
        assert_eq!(content, "Hello, world");
        assert!(calls.is_empty());
        assert_eq!(finish_reason(&chunks), "stop");
    }

    /// No whitespace-only content arrives between two calls.
    #[test]
    fn openai_stream_sends_no_whitespace_between_calls() {
        let pieces = [
            "Let me check both.\n",
            "<tool_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call>",
            "\n",
            "<tool_call>\n{\"name\": \"b\", \"arguments\": {}}\n</tool_call>",
            "\n",
        ];
        let chunks = openai_chunks(ToolCallFormat::Generic, &pieces, StopType::Eos);
        let contents: Vec<&str> = chunks
            .iter()
            .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(contents, ["Let me check both.\n"]);
        assert_eq!(openai_rebuild(&chunks).1.len(), 2);
    }

    fn anthropic_events(format: ToolCallFormat, pieces: &[&str]) -> Vec<(String, Value)> {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AnthropicSse>();
        let mut emitter = AnthropicStreamEmitter::new("msg_1".into(), "m".into(), tx);
        emitter.message_start(3);
        let calls = feed(format, ANTHROPIC_CALL_IDS, pieces, &mut |piece| {
            emitter.piece(piece)
        });
        emitter.finish(!calls.is_empty(), &Completion::for_tests("", StopType::Eos));
        drop(emitter);
        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push((
                event.event.to_string(),
                serde_json::from_str(&event.data).unwrap(),
            ));
        }
        events
    }

    /// The content blocks a client rebuilds from Anthropic stream events:
    /// `(type, text or name, input JSON text)`.  Checks that every block
    /// starts, takes deltas and stops in order.
    fn anthropic_rebuild(events: &[(String, Value)]) -> Vec<(String, String, String)> {
        let mut blocks: Vec<(String, String, String)> = Vec::new();
        let mut open = None;
        for (event, data) in events {
            assert_eq!(event.as_str(), data["type"].as_str().unwrap());
            match event.as_str() {
                "content_block_start" => {
                    assert_eq!(open, None, "a block opened inside another");
                    let index = data["index"].as_u64().unwrap() as usize;
                    assert_eq!(index, blocks.len());
                    let block = &data["content_block"];
                    let kind = block["type"].as_str().unwrap().to_string();
                    let name = block["name"].as_str().unwrap_or("").to_string();
                    if kind == "tool_use" {
                        assert!(block["id"].as_str().is_some_and(|id| !id.is_empty()));
                        assert_eq!(block["input"], serde_json::json!({}));
                    }
                    blocks.push((kind, name, String::new()));
                    open = Some(index);
                }
                "content_block_delta" => {
                    let index = data["index"].as_u64().unwrap() as usize;
                    assert_eq!(open, Some(index));
                    let delta = &data["delta"];
                    match delta["type"].as_str().unwrap() {
                        "text_delta" => blocks[index].1.push_str(delta["text"].as_str().unwrap()),
                        "thinking_delta" => blocks[index]
                            .1
                            .push_str(delta["thinking"].as_str().unwrap()),
                        "signature_delta" => assert_eq!(blocks[index].0, "thinking"),
                        "input_json_delta" => blocks[index]
                            .2
                            .push_str(delta["partial_json"].as_str().unwrap()),
                        other => panic!("unexpected delta {other}"),
                    }
                }
                "content_block_stop" => {
                    assert_eq!(open, data["index"].as_u64().map(|i| i as usize));
                    open = None;
                }
                _ => {}
            }
        }
        assert_eq!(open, None, "a block was left open");
        blocks
    }

    /// Anthropic streaming: a thinking block, a text block, then a
    /// `tool_use` block whose input arrives as `input_json_delta`s, then
    /// `stop_reason: tool_use`.
    #[test]
    fn anthropic_stream_sends_tool_use_blocks_as_they_are_generated() {
        let pieces = [
            "reasoning:Add them.",
            "Checking.",
            "<tool_call>\n<function=add>\n<parameter=a>\n2\n</parameter>\n",
            "<parameter=b>\n3\n</parameter>\n</function>\n</tool_call>",
        ];
        let events = anthropic_events(ToolCallFormat::Generic, &pieces);
        let blocks = anthropic_rebuild(&events);
        assert_eq!(blocks.len(), 3);
        assert_eq!(
            (blocks[0].0.as_str(), blocks[0].1.as_str()),
            ("thinking", "Add them.")
        );
        assert_eq!(
            (blocks[1].0.as_str(), blocks[1].1.as_str()),
            ("text", "Checking.")
        );
        assert_eq!(
            (blocks[2].0.as_str(), blocks[2].1.as_str()),
            ("tool_use", "add")
        );
        let (_, start) = events
            .iter()
            .find(|(_, data)| data["content_block"]["type"] == "tool_use")
            .unwrap();
        assert!(start["content_block"]["id"]
            .as_str()
            .unwrap()
            .starts_with("toolu_"));
        let input: Value = serde_json::from_str(&blocks[2].2)
            .unwrap_or_else(|e| panic!("{e}: {} in {events:?}", blocks[2].2));
        assert_eq!(input, serde_json::json!({"a": 2, "b": 3}));
        let input_deltas = events
            .iter()
            .filter(|(_, data)| data["delta"]["type"] == "input_json_delta")
            .count();
        assert!(input_deltas >= 2, "{events:?}");
        let (_, message_delta) = events.iter().find(|(e, _)| e == "message_delta").unwrap();
        assert_eq!(message_delta["delta"]["stop_reason"], "tool_use");
        assert_eq!(events.last().unwrap().0, "message_stop");
    }

    /// Prose after a call opens a new text block; arguments the model
    /// wrote as plain text arrive as one JSON string, as the non-streaming
    /// `tool_use.input` reports them.
    #[test]
    fn anthropic_stream_text_after_a_call_and_plain_text_arguments() {
        let text = "<|tool_calls_section_begin|><|tool_call_begin|>functions.f:0\
                    <|tool_call_argument_begin|>not json<|tool_call_end|><|tool_calls_section_end|> Done.";
        let blocks = anthropic_rebuild(&anthropic_events(ToolCallFormat::KimiK2, &[text]));
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].0, "tool_use");
        assert_eq!(
            serde_json::from_str::<Value>(&blocks[0].2).unwrap(),
            "not json"
        );
        assert_eq!(
            (blocks[1].0.as_str(), blocks[1].1.as_str()),
            ("text", " Done.")
        );
    }

    /// Two calls in a row open no text block between them.
    #[test]
    fn anthropic_stream_opens_no_whitespace_block_between_calls() {
        let pieces = [
            "Let me check both.\n<tool_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call>\n",
            "<tool_call>\n{\"name\": \"b\", \"arguments\": {}}\n</tool_call>",
        ];
        let blocks = anthropic_rebuild(&anthropic_events(ToolCallFormat::Generic, &pieces));
        let kinds: Vec<&str> = blocks.iter().map(|b| b.0.as_str()).collect();
        assert_eq!(kinds, ["text", "tool_use", "tool_use"]);
    }

    #[test]
    fn prometheus_exports_embedding_usage() {
        use std::sync::atomic::Ordering;
        let metrics = super::ServerMetrics::default();
        metrics.embed_tokens.fetch_add(7, Ordering::Relaxed);
        let text = metrics.prometheus(0, 0, 0);
        assert!(
            text.contains("joshua_embedding_tokens_total 7"),
            "missing embedding counter:\n{text}"
        );
    }

    #[cfg(feature = "distributed")]
    #[test]
    fn workers_stay_claimed_from_load_until_unload_completes() {
        let registry = super::ModelRegistry::default();
        let a: std::net::SocketAddr = "127.0.0.1:7001".parse().unwrap();
        let b: std::net::SocketAddr = "127.0.0.1:7002".parse().unwrap();
        // A load in progress holds its workers against an overlapping load.
        let claim = registry.claim_workers(&[a, b]).unwrap();
        assert!(registry.claim_workers(&[b]).is_err());
        // A failed load releases them.
        drop(claim);
        // A loaded model keeps them until its unload releases them.
        registry.claim_workers(&[a, b]).unwrap().keep();
        assert!(registry.claim_workers(&[a]).is_err());
        registry.release_workers(&[a, b]);
        registry.claim_workers(&[a, b]).unwrap();
    }
}
