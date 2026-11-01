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
//! - `GET  /v1/models`                — list loaded model
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
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Request, State},
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
use crate::coordinator::{ModelSlots, WorkerCapabilities, WorkerInfo, WORKER_PROTOCOL_VERSION};
use crate::engine::{Engine, EngineOptions};
use crate::error::JoshuaError;
use crate::whisper::WhisperEngine;
use crate::tools::parse_tool_calls;
use crate::types::{
    AssistantMessage, ChatChoice, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, ChatMessage, DeltaContent, EmbeddingData, EmbeddingRequest,
    EmbeddingResponse, ErrorResponse, FunctionCallResult, GenerationOptions, ModelInfo,
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
}

// ─── Model registry ─────────────────────────────────────────────────────────

/// A loaded model and where it runs.
#[derive(Clone)]
pub struct LoadedModel {
    pub id: String,
    pub engine: Arc<Engine>,
    /// Workers holding its pipeline stages; empty for a local model.
    pub workers: Vec<std::net::SocketAddr>,
    /// Each worker's layer range, in worker order.
    pub stages: Vec<(usize, usize)>,
    pub created: u64,
}

/// The models a server answers for, by id.
#[derive(Default)]
pub struct ModelRegistry {
    models: RwLock<Vec<LoadedModel>>,
    /// Ids being loaded, so concurrent loads of one id cannot both run.
    loading: Mutex<Vec<String>>,
    /// Workers claimed by a model being loaded, loaded, or being unloaded.
    /// A worker serves one controller connection at a time, so a claim
    /// lasts until the model's pipeline has released it.
    workers: Mutex<Vec<std::net::SocketAddr>>,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

impl ModelRegistry {
    /// A registry holding `engine` under its model name.
    pub fn with_model(engine: Arc<Engine>) -> Self {
        let registry = Self::default();
        registry.models.write().expect("fresh lock").push(LoadedModel {
            id: engine.model_name().to_string(),
            engine,
            workers: Vec::new(),
            stages: Vec::new(),
            created: now(),
        });
        registry
    }

    pub fn list(&self) -> Vec<LoadedModel> {
        self.models.read().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// The model a request names. With a single model loaded, any name
    /// selects it, as when a server only ever held one model.
    /// Returns the model's id and engine.
    pub fn resolve(&self, requested: &str) -> Result<(String, Arc<Engine>), ApiError> {
        let models = self.models.read().unwrap_or_else(|p| p.into_inner());
        let found = models
            .iter()
            .find(|m| m.id == requested)
            .or(match models.as_slice() {
                [only] => Some(only),
                _ => None,
            });
        if let Some(model) = found {
            return Ok((model.id.clone(), Arc::clone(&model.engine)));
        }
        match models.as_slice() {
            [] => Err(ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "no model is loaded; load one with POST /v1/models/load",
                "model_not_loaded",
            )),
            _ => Err(ApiError::new(
                StatusCode::NOT_FOUND,
                format!("model '{requested}' is not loaded"),
                "model_not_found",
            )),
        }
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
#[derive(Debug, serde::Deserialize)]
pub struct LoadModelRequest {
    /// GGUF file (or a directory holding one) under the model directory.
    pub model: String,
    /// Name to serve it under; defaults to the file stem.
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
    /// Concurrent remote sessions the workers reserve KV for.
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
    /// Resolve a requested model inside `model_dir`.
    fn resolve_path(&self, requested: &str) -> Result<PathBuf, ApiError> {
        let not_found = || {
            ApiError::new(
                StatusCode::NOT_FOUND,
                format!("model '{requested}' was not found in the model directory"),
                "model_not_found",
            )
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
        #[cfg(feature = "distributed")]
        if !req.local && (req.workers.is_some() || !self.workers.is_empty()) {
            return self.load_distributed(req, path, id, registry);
        }
        if req.workers.is_some() || req.ends.is_some() {
            return Err(crate::JoshuaError::InvalidRequest(
                "this server has no workers configured; omit 'workers' and 'ends'".into(),
            ));
        }
        let mut options = (self.engine_options)(path);
        if let Some(n_ctx) = req.n_ctx {
            options.n_ctx = u32::try_from(n_ctx)
                .map_err(|_| crate::JoshuaError::InvalidRequest("n_ctx too large".into()))?;
        }
        let mut engine = Engine::with_options(path, options)?;
        if let Some(backend) = &self.npu_backend {
            engine = engine.with_npu_backend(Arc::clone(backend));
        }
        let engine = self.finish_engine(engine);
        Ok(LoadedModel {
            id,
            engine: Arc::new(engine),
            workers: Vec::new(),
            stages: Vec::new(),
            created: now(),
        })
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

    /// Place the model on workers: they receive only their stages, from this
    /// controller, and this host keeps just the tokenizer and sampling.
    #[cfg(feature = "distributed")]
    fn load_distributed(
        &self,
        req: &LoadModelRequest,
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
        let header = crate::gguf_ext::read_header(&mut std::io::BufReader::new(
            std::fs::File::open(path)?,
        ))?;
        let arch = header.architecture().unwrap_or_default();
        let model_ctx = crate::gguf_meta::Meta::new(&header.metadata, &arch)
            .u32("context_length")
            .map_err(|e| ModelLoad(e.to_string()))? as usize;
        let defaults = Limits::default();
        let server_ctx = match (self.engine_options)(path).n_ctx {
            0 => defaults.context,
            n => n as usize,
        };
        let context = req.n_ctx.unwrap_or(server_ctx).min(model_ctx);
        let limits = Limits {
            context,
            chunk: req.chunk.unwrap_or(defaults.chunk).min(context),
            sessions: req.sessions.unwrap_or(defaults.sessions),
            ..defaults
        };
        // The controller maps the file only to tokenize and to read slices;
        // nothing is prefetched or loaded for compute here.
        let options = EngineOptions::with_n_ctx(context as u32)
            .backend(crate::engine::ComputeBackend::Cpu)
            .lazy_weights(true)
            .prefill_chunk(limits.chunk);
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
            id,
            engine: Arc::new(engine),
            workers,
            stages,
            created: now(),
        })
    }
}

fn model_json(model: &LoadedModel) -> serde_json::Value {
    json!({
        "id": model.id,
        "object": "model",
        "created": model.created,
        "owned_by": "joshua",
        "workers": model.workers.iter().map(|w| w.to_string()).collect::<Vec<_>>(),
        "stages": model.stages,
    })
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
    let path = manager(&state)?.resolve_path(&req.model)?;
    let id = req.id.clone().unwrap_or_else(|| {
        path.file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    });
    if id.is_empty() {
        return Err(ApiError::bad_request("empty model id"));
    }
    let loaded = tokio::task::spawn_blocking(move || {
        let manager = manager(&state)?;
        let guard = state.models.reserve(&id)?;
        tracing::info!(model = %id, path = %path.display(), "loading model");
        let model = manager
            .load(&req, &path, id, &state.models)
            .map_err(|e| match e {
                // The caller asked for this load; tell them why it failed.
                crate::JoshuaError::ModelLoad(msg) => {
                    ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, msg, "model_load_failed")
                }
                e => ApiError::from(e),
            })?;
        let body = model_json(&model);
        guard.finish(model);
        Ok::<_, ApiError>(body)
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))??;
    Ok(Json(loaded))
}

/// `POST /v1/models/unload` — stop serving a model and release it.
async fn unload_model(
    State(state): State<AppState>,
    Json(req): Json<UnloadModelRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    manager(&state)?;
    let model = state.models.remove(&req.id).ok_or_else(|| {
        ApiError::new(
            StatusCode::NOT_FOUND,
            format!("model '{}' is not loaded", req.id),
            "model_not_found",
        )
    })?;
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
                "embeddings": !m.engine.remote_only(),
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
    let defaults = GenerationOptions::default();
    let total_slots: usize = models.iter().map(|m| m.engine.max_concurrency()).sum();
    Ok(Json(json!({
        "model_path": first.engine.model_path(),
        "model_name": first.id,
        "n_ctx": first.engine.n_ctx(),
        "chat_template": first.engine.has_chat_template().then_some("<gguf chat template>"),
        "total_slots": total_slots,
        "default_generation_settings": {
            "n_predict": defaults.max_tokens,
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
    let (_, engine) = state
        .models
        .resolve(body.get("model").and_then(|v| v.as_str()).unwrap_or_default())?;
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
    let (_, engine) = state
        .models
        .resolve(body.get("model").and_then(|v| v.as_str()).unwrap_or_default())?;
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
    let Some(first) = models.first() else {
        return Err(ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "no model is loaded; load one with POST /v1/models/load",
            "model_not_loaded",
        ));
    };
    let engines = || models.iter().map(|m| &m.engine);
    Ok(Json(WorkerInfo {
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
        model: first.id.clone(),
        models: match models.len() {
            1 => Vec::new(),
            _ => models
                .iter()
                .map(|m| ModelSlots {
                    id: m.id.clone(),
                    n_ctx: Some(m.engine.n_ctx()),
                    max_concurrency: m.engine.max_concurrency(),
                    in_flight: m.engine.in_flight(),
                    embeddings: !m.engine.remote_only(),
                })
                .collect(),
        },
        n_ctx: first.engine.n_ctx(),
        backend: device_label(first.engine.device()).to_string(),
        max_concurrency: engines().map(|e| e.max_concurrency()).sum(),
        in_flight: engines().map(|e| e.in_flight()).sum(),
        capabilities: WorkerCapabilities {
            chat: true,
            completions: true,
            embeddings: engines().any(|e| !e.remote_only()),
            transcriptions: state.whisper.is_some(),
            streaming: true,
            tools: true,
        },
    }))
}

/// Short backend name for a candle device.
fn device_label(device: &candle_core::Device) -> &'static str {
    use candle_core::DeviceLocation as L;
    match device.location() {
        L::Cpu => "cpu",
        L::Cuda { .. } => "cuda",
        L::Metal { .. } => "metal",
        L::OpenCl { .. } => "opencl",
        L::Sycl { .. } => "sycl",
        L::Vulkan { .. } => "vulkan",
    }
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

/// `GET /v1/models` — returns the loaded models.
async fn list_models(State(state): State<AppState>) -> Json<ModelListResponse> {
    let created = now();
    let mut data: Vec<ModelInfo> = state
        .models
        .list()
        .into_iter()
        .map(|m| ModelInfo {
            id: m.id,
            object: "model".to_string(),
            created: m.created,
            owned_by: "joshua".to_string(),
        })
        .collect();
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
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let (model, engine) = state.models.resolve(&req.model)?;
    let stream = req.stream.unwrap_or(false);
    let mut options = req.to_generation_options().map_err(ApiError::bad_request)?;
    options.cache_scope = Some(crate::affinity::cache_scope(
        &headers,
        options.context_window,
    ));
    let messages = req.messages.clone();
    let tools = req.offered_tools();

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
    let cancel = CancelOnDrop::new();
    let max_tokens = options.max_tokens;
    let (text, usage, _, _) = tokio::task::spawn_blocking({
        let engine = Arc::clone(&engine);
        let tools = tools.clone();
        let flag = cancel.flag();
        move || engine.complete_chat_cancellable(&messages, tools.as_deref(), &options, &flag)
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?
    .map_err(ApiError::from)?;
    state.metrics.record_generation(&usage);

    // Extract tool calls from the output when the request offered tools.
    let (content, tool_calls, finish_reason) = if tools.is_some() {
        let (prose, calls) = parse_tool_calls(&text);
        if calls.is_empty() {
            let reason = if usage.completion_tokens >= max_tokens {
                "length"
            } else {
                "stop"
            };
            (Some(text), None, reason)
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
        let reason = if usage.completion_tokens >= max_tokens {
            "length"
        } else {
            "stop"
        };
        (Some(text), None, reason)
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

/// Markers that begin a tool call in the model's textual output.  The
/// longest marker sets how much trailing text the stream must withhold,
/// because a marker can be split across two decoded tokens.
const TOOL_MARKERS: [&str; 3] = ["<tool_call>", "[TOOL_CALLS]", "<function="];

/// Filters decoded text for the chat SSE stream.
///
/// Prose is forwarded as soon as it is safe.  A marker that may begin a tool
/// call (or a prefix of one split across decoded tokens) is held back, and
/// the text between a call's opening and closing marker is never emitted as
/// prose.  Once the closing marker arrives the filter resumes streaming the
/// prose that follows it.  Malformed markers are forwarded as ordinary text
/// instead of swallowing the rest of the response, and a bare JSON tool call
/// (which has no marker) is buffered until the parser can tell it apart from
/// ordinary JSON prose.
struct ChatStreamFilter {
    tools: bool,
    all: String,
    /// Next byte of `all` that has not been scanned/emitted.
    pos: usize,
    holdback: usize,
    /// A leading `{` may be a bare JSON tool call; hold the whole response
    /// until the final parse decides.
    bare_json: bool,
}

/// How a tool-call marker is terminated.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ToolMarkerKind {
    /// `<tool_call> … </tool_call>`
    Tagged,
    /// `[TOOL_CALLS][…]` — consumes the rest of the output.
    Mistral,
    /// `<function=…> … </function>` without a wrapper.
    Function,
}

fn tool_marker_kind(marker: &str) -> ToolMarkerKind {
    match marker {
        "<tool_call>" => ToolMarkerKind::Tagged,
        "[TOOL_CALLS]" => ToolMarkerKind::Mistral,
        _ => ToolMarkerKind::Function,
    }
}

impl ChatStreamFilter {
    fn new(tools: bool) -> Self {
        let holdback = if tools {
            TOOL_MARKERS
                .iter()
                .map(|m| m.len())
                .max()
                .unwrap_or(0)
                .saturating_sub(1)
        } else {
            0
        };
        Self {
            tools,
            all: String::new(),
            pos: 0,
            holdback,
            bare_json: false,
        }
    }

    /// Returns the prose that is safe to send now.
    fn push(&mut self, delta: &str) -> Vec<String> {
        if !self.tools {
            return vec![delta.to_string()];
        }
        self.all.push_str(delta);
        if self.pos == 0 && self.all.trim_start().starts_with('{') {
            self.bare_json = true;
        }
        self.drain(false)
    }

    /// Flush any withheld prose after decoding ends.
    fn finish(&mut self, final_text: &str) -> Vec<String> {
        if !self.tools {
            return Vec::new();
        }
        if self.all.is_empty() {
            self.all.push_str(final_text);
        }
        if self.pos == 0 && self.all.trim_start().starts_with('{') {
            self.bare_json = true;
        }
        let calls = parse_tool_calls(&self.all).1;
        if !calls.is_empty() {
            // A bare JSON call consumes the whole response; there is no prose
            // to flush.  Marked calls were already skipped while streaming.
            if self.bare_json {
                return Vec::new();
            }
            return self.drain(true);
        }
        // No call was recognised, so any marker-like text is ordinary prose:
        // release everything that is still buffered.
        self.bare_json = false;
        self.drain(true)
    }

    /// Parse the complete output.  With tools, `all` is the exact stream the
    /// engine reported; the return value from the engine is the fallback for
    /// an empty stream.
    fn parse(&self, final_text: &str) -> (String, Vec<crate::tools::ParsedToolCall>) {
        if !self.tools {
            // Without offered tools the model's markup is ordinary text.
            return (final_text.to_string(), Vec::new());
        }
        if !self.all.is_empty() {
            parse_tool_calls(&self.all)
        } else {
            parse_tool_calls(final_text)
        }
    }

    fn drain(&mut self, flushing: bool) -> Vec<String> {
        if self.bare_json {
            return Vec::new();
        }
        let mut out = String::new();
        loop {
            let rest = &self.all[self.pos..];
            let next = TOOL_MARKERS
                .iter()
                .filter_map(|marker| rest.find(marker).map(|at| (at, *marker)))
                .min_by_key(|(at, _)| *at);
            let Some((at, marker)) = next else {
                let end = if flushing {
                    self.all.len()
                } else {
                    floor_char_boundary(
                        &self.all,
                        self.all.len().saturating_sub(self.holdback),
                    )
                };
                if end > self.pos {
                    out.push_str(&self.all[self.pos..end]);
                    self.pos = end;
                }
                break;
            };

            let marker_start = self.pos + at;
            if marker_start > self.pos {
                out.push_str(&self.all[self.pos..marker_start]);
                self.pos = marker_start;
            }

            match tool_marker_kind(marker) {
                ToolMarkerKind::Mistral => {
                    // The payload runs to the end of the response, so there
                    // is no prose after it to stream.
                    if flushing {
                        let region = &self.all[marker_start..];
                        if parse_tool_calls(region).1.is_empty() {
                            out.push_str(region);
                        }
                        self.pos = self.all.len();
                    }
                    break;
                }
                kind => {
                    let terminator = if kind == ToolMarkerKind::Tagged {
                        "</tool_call>"
                    } else {
                        "</function>"
                    };
                    let body_start = marker_start + marker.len();
                    if let Some(found) = self.all[body_start..].find(terminator) {
                        let end = body_start + found + terminator.len();
                        let region = &self.all[marker_start..end];
                        if parse_tool_calls(region).1.is_empty() {
                            // Not a call: forward the marker text as prose.
                            out.push_str(region);
                        }
                        self.pos = end;
                        continue;
                    }
                    if flushing {
                        let region = &self.all[marker_start..];
                        if parse_tool_calls(region).1.is_empty() {
                            out.push_str(region);
                        }
                        self.pos = self.all.len();
                    }
                    // Otherwise wait for the closing marker to arrive.
                    break;
                }
            }
        }
        if out.is_empty() {
            Vec::new()
        } else {
            vec![out]
        }
    }
}

/// Largest char boundary at or below `index`.
fn floor_char_boundary(s: &str, mut index: usize) -> usize {
    while index > 0 && !s.is_char_boundary(index) {
        index -= 1;
    }
    index
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
            tool_calls: None,
        };
        self.send(delta, None, None);
    }

    fn tool_start(&mut self, index: usize, id: &str, name: &str) {
        let delta = DeltaContent {
            role: self.take_role(),
            content: None,
            tool_calls: Some(json!([{
                "index": index,
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": ""},
            }])),
        };
        self.send(delta, None, None);
    }

    fn tool_args(&mut self, index: usize, arguments: &str) {
        let delta = DeltaContent {
            role: None,
            content: None,
            tool_calls: Some(json!([{
                "index": index,
                "function": {"arguments": arguments},
            }])),
        };
        self.send(delta, None, None);
    }

    fn finish(&mut self, reason: &str, usage: &UsageInfo) {
        self.send(DeltaContent::default(), Some(reason.to_string()), Some(usage));
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
    ) {
        let chunk = ChatCompletionChunk::new(
            self.id.clone(),
            self.model.clone(),
            delta,
            finish_reason,
        );
        let mut value = serde_json::to_value(&chunk).unwrap_or_default();
        if let Some(usage) = usage {
            value["usage"] = json!({
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            });
        }
        let _ = self.tx.send(value.to_string());
    }
}

/// Run one streaming chat completion on the blocking pool.
fn stream_chat_completion(
    engine: &Engine,
    model: &str,
    id: &str,
    messages: &[ChatMessage],
    tools: Option<&[Tool]>,
    options: &GenerationOptions,
    cancel: &AtomicBool,
    tx: tokio::sync::mpsc::UnboundedSender<String>,
    metrics: &ServerMetrics,
) {
    let mut filter = ChatStreamFilter::new(tools.is_some());
    let mut emitter = ChatStreamEmitter::new(id.to_string(), model.to_string(), tx);
    let result = {
        let mut on_text = |delta: &str| {
            for text in filter.push(delta) {
                emitter.text(&text);
            }
        };
        engine.complete_chat_streaming(messages, tools, options, cancel, &mut on_text)
    };

    match result {
        Ok((text, usage, _, _)) => {
            metrics.record_generation(&usage);
            for withheld in filter.finish(&text) {
                emitter.text(&withheld);
            }
            let (_, calls) = filter.parse(&text);
            if calls.is_empty() {
                let reason = if usage.completion_tokens >= options.max_tokens {
                    "length"
                } else {
                    "stop"
                };
                emitter.finish(reason, &usage);
            } else {
                for (index, call) in calls.iter().enumerate() {
                    let call_id = format!("call_{}", Uuid::new_v4().simple());
                    emitter.tool_start(index, &call_id, &call.name);
                    emitter.tool_args(index, &call.arguments);
                }
                emitter.finish("tool_calls", &usage);
            }
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
    let (_, engine) = state.models.resolve(&req.model)?;
    let (messages, tools, options) =
        anthropic::to_chat_request(&req).map_err(ApiError::bad_request)?;
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
    let (resolved_model, engine) = state.models.resolve(&req.model)?;
    let (messages, tools, mut options) =
        anthropic::to_chat_request(&req).map_err(ApiError::bad_request)?;
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
    let tools_offered = tools.is_some();
    let max_tokens = options.max_tokens;
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

    let id = format!("msg_{}", Uuid::new_v4().simple());
    let (response, _) = anthropic::response(
        id,
        model,
        &completion.text,
        &completion.usage,
        tools_offered,
        max_tokens,
        completion.stop_sequence.as_deref(),
    );
    Ok(Json(response).into_response())
}

/// Writes Anthropic streaming events to the SSE channel.
struct AnthropicStreamEmitter {
    id: String,
    model: String,
    tx: tokio::sync::mpsc::UnboundedSender<AnthropicSse>,
    text_started: bool,
    next_index: usize,
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
            text_started: false,
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

    fn text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        if !self.text_started {
            self.text_started = true;
            self.send(
                "content_block_start",
                json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }),
            );
        }
        self.send(
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            }),
        );
    }

    fn finish(
        &mut self,
        calls: &[crate::tools::ParsedToolCall],
        usage: &UsageInfo,
        max_tokens: u32,
        stop_sequence: Option<&str>,
    ) {
        if self.text_started {
            self.send(
                "content_block_stop",
                json!({"type": "content_block_stop", "index": 0}),
            );
            self.next_index = 1;
        }
        for call in calls {
            let index = self.next_index;
            self.next_index += 1;
            let id = format!("toolu_{}", Uuid::new_v4().simple());
            self.send(
                "content_block_start",
                json!({
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": id,
                        "name": call.name,
                        "input": {},
                    },
                }),
            );
            self.send(
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        // Must be valid JSON text even when the model emitted
                        // malformed arguments; the non-streaming response
                        // normalises the same value to a JSON string.
                        "partial_json": anthropic::tool_arguments_json(&call.arguments),
                    },
                }),
            );
            self.send(
                "content_block_stop",
                json!({"type": "content_block_stop", "index": index}),
            );
        }
        let stop_reason = if !calls.is_empty() {
            "tool_use"
        } else if stop_sequence.is_some() {
            "stop_sequence"
        } else if usage.completion_tokens >= max_tokens {
            "max_tokens"
        } else {
            "end_turn"
        };
        self.send(
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason,
                    "stop_sequence": stop_sequence,
                },
                "usage": {"output_tokens": usage.completion_tokens},
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
    let mut filter = ChatStreamFilter::new(tools.is_some());
    let mut emitter = AnthropicStreamEmitter::new(id.to_string(), model.to_string(), tx);
    emitter.message_start(input_tokens);
    let result = {
        let mut on_text = |delta: &str| {
            for text in filter.push(delta) {
                emitter.text(&text);
            }
        };
        engine.complete_chat_streaming_detailed(messages, tools, options, cancel, &mut on_text)
    };
    match result {
        Ok(completion) => {
            metrics.record_generation(&completion.usage);
            for withheld in filter.finish(&completion.text) {
                emitter.text(&withheld);
            }
            let (_, calls) = filter.parse(&completion.text);
            emitter.finish(
                &calls,
                &completion.usage,
                options.max_tokens,
                completion.stop_sequence.as_deref(),
            );
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
    let (model, engine) = state
        .models
        .resolve(body.get("model").and_then(|v| v.as_str()).unwrap_or_default())?;
    let prompt = body
        .get("prompt")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::bad_request("Missing required field 'prompt'"))?
        .to_string();

    let options = GenerationOptions {
        cache_scope: Some(crate::affinity::cache_scope(&headers, None)),
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

    let cancel = CancelOnDrop::new();
    let max_tokens = options.max_tokens;
    let (text, usage, _, _) = tokio::task::spawn_blocking({
        let engine = Arc::clone(&engine);
        let flag = cancel.flag();
        move || engine.complete_chat_cancellable(&messages, None, &options, &flag)
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?
    .map_err(ApiError::from)?;

    state.metrics.record_generation(&usage);
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let finish_reason = if usage.completion_tokens >= max_tokens {
        "length"
    } else {
        "stop"
    };

    Ok(Json(json!({
        "id": format!("cmpl-{}", Uuid::new_v4().simple()),
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{
            "text": text,
            "index": 0,
            "finish_reason": finish_reason
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
    let (model, engine) = state.models.resolve(&req.model)?;
    let texts: Vec<String> = req.input.into_vec();

    // Like generation, stop embedding a batch once the requester is gone.
    let cancel = CancelOnDrop::new();
    let (vectors, prompt_tokens) = tokio::task::spawn_blocking({
        let engine = Arc::clone(&engine);
        let flag = cancel.flag();
        move || engine.embed_with_usage_cancellable(&texts, &flag)
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

    let mut file: Option<axum::body::Bytes> = None;
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
                file = Some(bytes);
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

    let transcription = run_transcription_job(whisper.admission.clone(), move |cancel| {
        whisper.transcribe_wav_cancellable(&file, language.as_deref(), false, cancel)
    })
    .await?;

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

    #[test]
    fn stream_filter_hides_a_tool_marker_split_across_chunks() {
        let mut filter = super::ChatStreamFilter::new(true);
        let mut prose = String::new();
        for piece in [
            "I will call ",
            "<tool",
            "_call>{\"name\":",
            "\"f\",\"arguments\":{}}</tool_call>",
        ] {
            for text in filter.push(piece) {
                prose.push_str(&text);
            }
        }
        assert_eq!(prose, "I will call ");
        let (_, calls) = filter.parse("");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "f");
        assert_eq!(calls[0].arguments, "{}");
    }

    #[test]
    fn stream_filter_forwards_everything_without_tools() {
        let mut filter = super::ChatStreamFilter::new(false);
        let mut prose = String::new();
        for piece in ["a ", "b ", "c"] {
            for text in filter.push(piece) {
                prose.push_str(&text);
            }
        }
        assert_eq!(prose, "a b c");
        assert!(filter.parse("a b c").1.is_empty());
    }

    #[test]
    fn stream_filter_resumes_prose_after_a_tool_call() {
        let mut filter = super::ChatStreamFilter::new(true);
        let mut prose = String::new();
        for piece in [
            "Checking ",
            "<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>",
            " Done",
        ] {
            for text in filter.push(piece) {
                prose.push_str(&text);
            }
        }
        for text in filter.finish("") {
            prose.push_str(&text);
        }
        assert_eq!(prose, "Checking  Done");
        let (parsed_prose, calls) = filter.parse("");
        assert_eq!(calls.len(), 1);
        assert_eq!(parsed_prose, "Checking  Done");
    }

    #[test]
    fn stream_filter_does_not_duplicate_a_bare_json_tool_call() {
        let mut filter = super::ChatStreamFilter::new(true);
        let mut prose = String::new();
        for piece in ["{\"name\":\"lookup\",", "\"arguments\":{\"q\":\"rust\"}}"] {
            for text in filter.push(piece) {
                prose.push_str(&text);
            }
        }
        for text in filter.finish("") {
            prose.push_str(&text);
        }
        assert_eq!(prose, "");
        let (parsed_prose, calls) = filter.parse("");
        assert_eq!(parsed_prose, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "lookup");
    }

    #[test]
    fn stream_filter_releases_plain_json_instead_of_dropping_it() {
        let mut filter = super::ChatStreamFilter::new(true);
        let mut prose = String::new();
        for piece in ["{\"answer\":", "42}"] {
            for text in filter.push(piece) {
                prose.push_str(&text);
            }
        }
        for text in filter.finish("") {
            prose.push_str(&text);
        }
        assert_eq!(prose, "{\"answer\":42}");
        assert!(filter.parse("").1.is_empty());
    }

    #[test]
    fn stream_filter_does_not_swallow_prose_after_an_invalid_marker() {
        let mut filter = super::ChatStreamFilter::new(true);
        let mut prose = String::new();
        for piece in [
            "before ",
            "<tool_call>not a call</tool_call>",
            " after",
        ] {
            for text in filter.push(piece) {
                prose.push_str(&text);
            }
        }
        for text in filter.finish("") {
            prose.push_str(&text);
        }
        assert_eq!(prose, "before <tool_call>not a call</tool_call> after");
        assert!(filter.parse("").1.is_empty());
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
