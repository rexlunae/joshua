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
//! - `GET  /v1/worker/info`           — worker identity and live load for a
//!   `joshua route` coordinator (see [`crate::coordinator`])
//! - `POST /v1/models/load`           — load a model from the controller's
//!   model directory, locally or onto its workers (when enabled)
//! - `POST /v1/models/unload`         — unload a model (when enabled)
//!
//! Chat and text completions stop decoding when the HTTP request is dropped
//! (the client, or a coordinator proxying for it, disconnected), so a
//! cancelled request releases its concurrency permit promptly.

use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
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

use crate::coordinator::{WorkerCapabilities, WorkerInfo, WORKER_PROTOCOL_VERSION};
use crate::engine::{Engine, EngineOptions};
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
    /// Loaded chat/embedding models.
    pub models: ModelRegistry,
    /// Enables `/v1/models/load` and `/v1/models/unload` when set.
    pub manager: Option<ModelManager>,
    /// Optional Whisper model for `/v1/audio/transcriptions`.
    pub whisper: Option<Arc<WhisperEngine>>,
    /// When set, every `/v1` request must carry this key as
    /// `Authorization: Bearer <key>`.  `/health` stays open for probes.
    pub api_key: Option<String>,
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

    /// The only loaded model, for callers that describe a single model.
    pub fn single(&self) -> Result<(String, Arc<Engine>), ApiError> {
        let models = self.models.read().unwrap_or_else(|p| p.into_inner());
        match models.as_slice() {
            [only] => Ok((only.id.clone(), Arc::clone(&only.engine))),
            [] => Err(ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "no model is loaded; load one with POST /v1/models/load",
                "model_not_loaded",
            )),
            several => Err(ApiError::new(
                StatusCode::CONFLICT,
                format!(
                    "{} models are loaded; worker info describes one",
                    several.len()
                ),
                "multiple_models",
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
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/audio/transcriptions", post(transcriptions))
        .route(crate::coordinator::WORKER_INFO_PATH, get(worker_info))
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
        .layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            require_api_key,
        ));
    Router::new()
        .route("/health", get(health))
        .merge(api)
        .layer(CorsLayer::permissive())
        .merge(manage)
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

/// `GET /health` — returns `{"status":"ok"}`.
async fn health() -> Json<serde_json::Value> {
    Json(json!({"status": "ok"}))
}

/// `GET /v1/worker/info` — what a `joshua route` coordinator needs to admit
/// requests here: protocol version, model identity, context limit, backend,
/// and the admission cap with the current in-flight count.
///
/// The protocol describes one model, so this answers only while exactly one
/// is loaded.
async fn worker_info(State(state): State<AppState>) -> Result<Json<WorkerInfo>, ApiError> {
    let (model, engine) = state.models.single()?;
    Ok(Json(WorkerInfo {
        protocol_version: WORKER_PROTOCOL_VERSION,
        model,
        n_ctx: engine.n_ctx(),
        backend: device_label(engine.device()).to_string(),
        max_concurrency: engine.max_concurrency(),
        in_flight: engine.in_flight(),
        capabilities: WorkerCapabilities {
            chat: true,
            completions: true,
            embeddings: !engine.remote_only(),
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
struct CancelOnDrop(Arc<AtomicBool>);

impl CancelOnDrop {
    fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    fn flag(&self) -> Arc<AtomicBool> {
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
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let (model, engine) = state.models.resolve(&req.model)?;
    let stream = req.stream.unwrap_or(false);
    let options = req.to_generation_options().map_err(ApiError::bad_request)?;
    let messages = req.messages.clone();
    let tools = req.offered_tools();

    if stream {
        // ── Streaming path ────────────────────────────────────────────────────
        let id = format!("chatcmpl-{}", Uuid::new_v4().simple());

        // Run inference in a blocking thread to avoid stalling the async runtime.
        let cancel = CancelOnDrop::new();
        let (text, usage, _, _) = tokio::task::spawn_blocking({
            let engine = Arc::clone(&engine);
            let tools = tools.clone();
            let flag = cancel.flag();
            move || engine.complete_chat_cancellable(&messages, tools.as_deref(), &options, &flag)
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
    let cancel = CancelOnDrop::new();
    let (text, usage, _, _) = tokio::task::spawn_blocking({
        let engine = Arc::clone(&engine);
        let tools = tools.clone();
        let flag = cancel.flag();
        move || engine.complete_chat_cancellable(&messages, tools.as_deref(), &options, &flag)
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
    let (model, engine) = state
        .models
        .resolve(body.get("model").and_then(|v| v.as_str()).unwrap_or_default())?;
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

    let cancel = CancelOnDrop::new();
    let (text, usage, _, _) = tokio::task::spawn_blocking({
        let engine = Arc::clone(&engine);
        let flag = cancel.flag();
        move || engine.complete_chat_cancellable(&messages, None, &options, &flag)
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
        "model": model,
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
    let (model, engine) = state.models.resolve(&req.model)?;
    let texts: Vec<String> = req.input.into_vec();

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
    fn new(status: StatusCode, msg: impl Into<String>, error_type: &str) -> Self {
        Self {
            status,
            body: ErrorResponse::new(msg, error_type),
        }
    }

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
        assert!(!api_keys_match(b"sek", b"sekret"));
        assert!(!api_keys_match(b"sekretsekret", b"sekret"));
        assert!(!api_keys_match(b"", b"sekret"));
    }

    #[test]
    fn api_keys_match_handles_an_empty_expected_key() {
        assert!(api_keys_match(b"", b""));
        assert!(!api_keys_match(b"a", b""));
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
