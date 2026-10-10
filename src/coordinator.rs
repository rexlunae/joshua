//! Request-level routing across independent joshua workers (`joshua route`).
//!
//! A coordinator accepts OpenAI-style requests and forwards each *whole*
//! request to one of several ordinary `joshua serve` processes.  The request
//! runs entirely on that worker — prefill, decode, KV cache and sampling
//! never leave it — so this mode scales request throughput, not the size of
//! a single model (that is the separate `distributed` cluster work).
//!
//! # Worker protocol (version [`WORKER_PROTOCOL_VERSION`])
//!
//! The data plane is the worker's existing OpenAI HTTP API
//! (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`); request
//! bodies are forwarded byte-for-byte, so every request parameter, streaming
//! event, stop behaviour, error body and usage figure is the worker's own.
//! The control plane is one endpoint, [`WORKER_INFO_PATH`], returning a
//! [`WorkerInfo`]: protocol version, model identity, context limit, backend,
//! admission cap and live in-flight count.
//!
//! * **Membership** is the configured worker list; nothing joins on its own.
//! * **Health**: each worker's info endpoint is polled; a worker is routable
//!   only while its last probe succeeded with a matching protocol version.  A
//!   transport failure during a request also marks it unhealthy until the
//!   next good probe.
//! * **Admission**: the coordinator keeps at most `slots` requests in flight
//!   per worker (the worker's own `max_concurrency` unless overridden) and
//!   picks the matching healthy worker with the lowest load.  When every
//!   slot is taken, requests wait in a bounded queue; a full queue answers
//!   `429`, a queue wait that times out answers `503`, both with
//!   `Retry-After`.
//! * **Admission vs execution**: a request is retried on another worker only
//!   when nothing has reached the client yet *and* the failure is a refused
//!   connection, a lost connection, or the worker's own `503` admission
//!   rejection (which it returns before any generation starts).  Any other
//!   worker response — a `400` for a bad request, a `500` — is the request's
//!   answer and is passed through, so a deterministic failure is not run
//!   again elsewhere.
//! * **Streams are never restarted.**  Once the first SSE event has been
//!   forwarded, a lost worker ends the stream with an OpenAI-style error
//!   event (and no `[DONE]`) instead of replaying the request elsewhere.
//!   Events are forwarded whole and in order.
//! * **Cancellation**: each forwarded request owns its worker connection.
//!   When the client disconnects, the connection to the worker is closed,
//!   and the worker stops decoding (see [`crate::server`]).
//! * **Records**: every request gets an `x-request-id` (the client's, when it
//!   sends a well-formed one) that is forwarded to the worker and returned
//!   with `x-joshua-worker`.  Logs carry the request id, worker, attempt,
//!   queue time and outcome — never prompt or output text.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{header, HeaderMap, HeaderValue, Method, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use http_body_util::{BodyExt, Full, Limited};
use hyper::body::Incoming;
use hyper_util::rt::TokioIo;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tower_http::cors::CorsLayer;
use uuid::Uuid;

use crate::types::{ErrorResponse, ModelInfo, ModelListResponse};

/// Version of the worker protocol this build speaks.  Bumped on any
/// incompatible change to [`WorkerInfo`] or to how requests are forwarded;
/// a worker reporting another version is never routed to.
pub const WORKER_PROTOCOL_VERSION: u32 = 1;

/// Worker endpoint that reports a [`WorkerInfo`].
pub const WORKER_INFO_PATH: &str = "/v1/worker/info";

/// Response header naming the worker that served a routed request.
pub const WORKER_HEADER: &str = "x-joshua-worker";

/// Request/response header carrying the request id.
pub const REQUEST_ID_HEADER: &str = "x-request-id";

/// Ceiling on a buffered (non-streaming) worker response.
const MAX_BUFFERED_BODY: usize = 64 << 20;

/// Ceiling on a worker error or info body.
const MAX_SMALL_BODY: usize = 1 << 20;

/// Paths routed to workers.
const ROUTED_PATHS: [&str; 3] = ["/v1/chat/completions", "/v1/completions", "/v1/embeddings"];

/// What a worker reports at [`WORKER_INFO_PATH`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// The worker's [`WORKER_PROTOCOL_VERSION`].
    pub protocol_version: u32,
    /// Model identity (the id the worker reports in `/v1/models`).
    pub model: String,
    /// Every model the worker serves, `model` first, with its own admission
    /// figures, when it serves more than one.  Older workers leave it out and
    /// serve only `model`, admitting by the worker-wide figures.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<ModelSlots>,
    /// Context window in tokens.
    pub n_ctx: u32,
    /// Compute backend (`cpu`, `cuda`, `metal`, `opencl`, `sycl`, `vulkan`).
    pub backend: String,
    /// Admission cap: requests beyond this many in flight get a `503`.
    pub max_concurrency: usize,
    /// Requests executing now, from every client.
    pub in_flight: usize,
    /// Which request kinds the worker serves.
    #[serde(default)]
    pub capabilities: WorkerCapabilities,
}

/// One model on a worker that serves several, each admitting its own
/// requests.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSlots {
    /// The model's id.
    pub id: String,
    /// Admission cap for this model's requests.
    pub max_concurrency: usize,
    /// This model's requests executing now, from every client.
    pub in_flight: usize,
    /// Whether this model serves `/v1/embeddings` (the worker-wide
    /// capability says whether any of its models does).
    pub embeddings: bool,
}

/// Request kinds a worker serves.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkerCapabilities {
    /// `/v1/chat/completions`.
    pub chat: bool,
    /// `/v1/completions`.
    pub completions: bool,
    /// `/v1/embeddings`.
    pub embeddings: bool,
    /// `/v1/audio/transcriptions` (not routed by the coordinator).
    pub transcriptions: bool,
    /// SSE streaming on the chat route.
    pub streaming: bool,
    /// Tool calls on the chat route.
    pub tools: bool,
}

impl WorkerInfo {
    /// Every model id the worker serves.
    pub fn model_ids(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.model.as_str()).chain(self.models.iter().map(|m| m.id.as_str()))
    }

    /// `model`'s own admission figures, when the worker reports them.
    fn slots_of(&self, model: Option<&str>) -> Option<&ModelSlots> {
        model.and_then(|m| self.models.iter().find(|s| s.id == m))
    }
}

impl WorkerCapabilities {
    fn serves(&self, path: &str) -> bool {
        match path {
            "/v1/chat/completions" => self.chat,
            "/v1/completions" => self.completions,
            "/v1/embeddings" => self.embeddings,
            _ => false,
        }
    }
}

/// Coordinator settings (the `joshua route` flags).
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// Worker HTTP addresses (`joshua serve --addr`).  Same shape as the
    /// controller's `--worker` list (#171), which names pipeline agents
    /// instead; here every address is an ordinary `joshua serve` reached
    /// over plain HTTP.
    pub workers: Vec<SocketAddr>,
    /// Bearer key clients must present on `/v1` routes.
    pub api_key: Option<String>,
    /// Bearer key presented to workers.
    pub worker_api_key: Option<String>,
    /// Per-worker in-flight cap; `None` uses each worker's `max_concurrency`.
    pub worker_slots: Option<usize>,
    /// Requests allowed to wait for a slot when every worker is full.
    pub queue_depth: usize,
    /// Longest a request waits in the queue before a `503`.
    pub queue_timeout: Duration,
    /// Interval between worker health probes.
    pub health_interval: Duration,
    /// Timeout for connecting to a worker (and for a whole health probe).
    pub connect_timeout: Duration,
}

impl CoordinatorConfig {
    /// Defaults for the given workers: no keys, worker-reported slots, a
    /// 64-request queue with a 30 s wait, 2 s probes and connect timeout.
    pub fn new(workers: Vec<SocketAddr>) -> Self {
        Self {
            workers,
            api_key: None,
            worker_api_key: None,
            worker_slots: None,
            queue_depth: 64,
            queue_timeout: Duration::from_secs(30),
            health_interval: Duration::from_secs(2),
            connect_timeout: Duration::from_secs(2),
        }
    }
}

/// A worker's address as the HTTP client uses it.
#[derive(Debug, Clone)]
struct Endpoint {
    /// `http://addr`, used as the worker's name.
    base: String,
    /// `host:port` to connect to and send as `Host`.
    authority: String,
}

impl Endpoint {
    fn new(addr: SocketAddr) -> Self {
        // `SocketAddr`'s Display brackets IPv6 hosts, as `Host` requires.
        let authority = addr.to_string();
        Self {
            base: format!("http://{authority}"),
            authority,
        }
    }
}

/// Last known state of one worker.
#[derive(Debug, Clone, Default)]
struct WorkerStatus {
    healthy: bool,
    info: Option<WorkerInfo>,
    /// Requests the worker runs for clients other than this coordinator, as
    /// of the last probe.
    external_in_flight: usize,
    /// The same per model, for a worker that reports its models' figures.
    external_by_model: HashMap<String, usize>,
    last_error: Option<String>,
}

struct Worker {
    endpoint: Endpoint,
    status: Mutex<WorkerStatus>,
    /// Requests this coordinator has in flight on the worker.
    dispatched: AtomicUsize,
    /// The same by requested model.
    dispatched_by_model: Mutex<HashMap<String, usize>>,
}

impl Worker {
    fn status(&self) -> MutexGuard<'_, WorkerStatus> {
        self.status.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn by_model(&self) -> MutexGuard<'_, HashMap<String, usize>> {
        self.dispatched_by_model
            .lock()
            .unwrap_or_else(|p| p.into_inner())
    }

    fn mark_unhealthy(&self, reason: String) {
        let mut st = self.status();
        st.healthy = false;
        st.last_error = Some(reason);
    }
}

/// Why no worker slot could be had.
#[derive(Debug)]
enum Reject {
    /// The request names a model no configured worker reports.
    UnknownModel(String, Vec<String>),
    /// Workers serve the model, but none is healthy.
    Unavailable,
    /// Every slot is busy and the wait queue is full.
    QueueFull,
    /// Waited `queue_timeout` without a free slot.
    QueueTimeout,
    /// Every eligible worker was already tried for this request.
    Exhausted,
    /// Every eligible worker refused admission (each answered `503`).
    AllRefused,
}

enum Pick {
    Slot(SlotGuard),
    Busy,
    Reject(Reject),
}

/// One reserved worker slot; released (and queued requests woken) on drop.
struct SlotGuard {
    coordinator: Arc<Coordinator>,
    index: usize,
    /// The requested model, counted against its own admission figures.
    model: Option<String>,
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        let w = &self.coordinator.workers[self.index];
        if let Some(model) = &self.model {
            if let Some(n) = w.by_model().get_mut(model) {
                *n = n.saturating_sub(1);
            }
        }
        w.dispatched.fetch_sub(1, Ordering::AcqRel);
        self.coordinator.released.notify_waiters();
    }
}

struct QueueGuard<'a>(&'a AtomicUsize);

impl Drop for QueueGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Aborts a spawned task when dropped.  Owns a worker connection, so
/// dropping a forwarded request closes the socket to the worker.
struct AbortOnDrop(JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// The request router: worker registry, admission and forwarding.
pub struct Coordinator {
    config: CoordinatorConfig,
    workers: Vec<Worker>,
    /// Serialises slot selection so check-and-reserve is atomic.
    pick_lock: Mutex<()>,
    /// Signalled when a slot frees up or worker health changes.
    released: Notify,
    queued: AtomicUsize,
    /// Rotating start index, so equally loaded workers share requests.
    rotation: AtomicUsize,
}

/// Status row for `GET /v1/workers`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStatusReport {
    /// Worker base URL.
    pub url: String,
    /// Routable right now.
    pub healthy: bool,
    /// Last reported info, when a probe has succeeded.
    pub info: Option<WorkerInfo>,
    /// Slot cap the coordinator applies.
    pub slots: Option<usize>,
    /// Requests this coordinator has in flight there.
    pub dispatched: usize,
    /// Requests from other clients, as of the last probe.
    pub external_in_flight: usize,
    /// Last probe or transport error.
    pub last_error: Option<String>,
}

/// What went wrong forwarding one attempt.
enum Failure {
    /// Nothing reached the client and the worker did not (or can no longer)
    /// run the request: try another worker.
    Retry { reason: String, admission: bool },
    /// The answer to send the client as-is.
    Final(Response),
}

impl Coordinator {
    /// Build a coordinator over the configured workers.  No worker is
    /// routable until [`Coordinator::refresh`] has probed it.
    pub fn new(config: CoordinatorConfig) -> Result<Arc<Self>, String> {
        if config.workers.is_empty() {
            return Err("at least one worker address is required".to_string());
        }
        let mut workers = Vec::with_capacity(config.workers.len());
        for &addr in &config.workers {
            let endpoint = Endpoint::new(addr);
            if workers
                .iter()
                .any(|w: &Worker| w.endpoint.base == endpoint.base)
            {
                return Err(format!("worker '{}' is listed twice", endpoint.base));
            }
            workers.push(Worker {
                endpoint,
                status: Mutex::new(WorkerStatus::default()),
                dispatched: AtomicUsize::new(0),
                dispatched_by_model: Mutex::new(HashMap::new()),
            });
        }
        Ok(Arc::new(Self {
            config,
            workers,
            pick_lock: Mutex::new(()),
            released: Notify::new(),
            queued: AtomicUsize::new(0),
            rotation: AtomicUsize::new(0),
        }))
    }

    /// Probe every worker once and update the registry.
    pub async fn refresh(&self) {
        let probes = self.workers.iter().map(|w| async move {
            let before = w.dispatched.load(Ordering::Acquire);
            let mut ours_by_model = w.by_model().clone();
            let result = self.probe(w).await;
            let after = w.dispatched.load(Ordering::Acquire);
            for (model, n) in w.by_model().iter() {
                let ours = ours_by_model.entry(model.clone()).or_default();
                *ours = (*ours).max(*n);
            }
            (w, result, before.max(after), ours_by_model)
        });
        for (w, result, ours, ours_by_model) in futures_util::future::join_all(probes).await {
            let mut st = w.status();
            match result {
                Ok(info) if info.protocol_version != WORKER_PROTOCOL_VERSION => {
                    if st.healthy || st.last_error.is_none() {
                        tracing::warn!(
                            worker = %w.endpoint.base,
                            version = info.protocol_version,
                            "worker speaks another protocol version; not routing to it"
                        );
                    }
                    st.healthy = false;
                    st.last_error = Some(format!(
                        "protocol version {} (coordinator speaks {})",
                        info.protocol_version, WORKER_PROTOCOL_VERSION
                    ));
                    st.info = Some(info);
                }
                Ok(info) => {
                    if !st.healthy {
                        tracing::info!(
                            worker = %w.endpoint.base,
                            model = %info.model,
                            backend = %info.backend,
                            slots = info.max_concurrency,
                            "worker healthy"
                        );
                    }
                    st.external_in_flight = info.in_flight.saturating_sub(ours);
                    st.external_by_model = info
                        .models
                        .iter()
                        .map(|m| {
                            let ours = ours_by_model.get(&m.id).copied().unwrap_or(0);
                            (m.id.clone(), m.in_flight.saturating_sub(ours))
                        })
                        .collect();
                    st.healthy = true;
                    st.last_error = None;
                    st.info = Some(info);
                }
                Err(e) => {
                    if st.healthy || st.last_error.is_none() {
                        tracing::warn!(worker = %w.endpoint.base, error = %e, "worker unhealthy");
                    }
                    st.healthy = false;
                    st.last_error = Some(e);
                }
            }
        }
        // Health may have changed: let queued requests look again.
        self.released.notify_waiters();
    }

    /// Probe forever at `health_interval`.
    pub fn spawn_health_loop(self: &Arc<Self>) -> JoinHandle<()> {
        let this = Arc::clone(self);
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(this.config.health_interval);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                tick.tick().await;
                this.refresh().await;
            }
        })
    }

    /// The registry, as served at `GET /v1/workers`.
    pub fn status_report(&self) -> Vec<WorkerStatusReport> {
        self.workers
            .iter()
            .map(|w| {
                let st = w.status().clone();
                WorkerStatusReport {
                    url: w.endpoint.base.clone(),
                    healthy: st.healthy,
                    slots: st.info.as_ref().map(|i| self.slots_for(i)),
                    info: st.info,
                    dispatched: w.dispatched.load(Ordering::Acquire),
                    external_in_flight: st.external_in_flight,
                    last_error: st.last_error,
                }
            })
            .collect()
    }

    fn slots_for(&self, info: &WorkerInfo) -> usize {
        self.config
            .worker_slots
            .unwrap_or(info.max_concurrency)
            .max(1)
    }

    async fn probe(&self, w: &Worker) -> Result<WorkerInfo, String> {
        let attempt = async {
            let (response, _conn) = self
                .send(
                    w,
                    Method::GET,
                    WORKER_INFO_PATH,
                    &HeaderMap::new(),
                    Bytes::new(),
                )
                .await
                .map_err(|(_, e)| e)?;
            let status = response.status();
            let body = Limited::new(response.into_body(), MAX_SMALL_BODY)
                .collect()
                .await
                .map_err(|e| format!("reading worker info: {e}"))?
                .to_bytes();
            if !status.is_success() {
                return Err(format!("worker info returned {status}"));
            }
            serde_json::from_slice::<WorkerInfo>(&body)
                .map_err(|e| format!("malformed worker info: {e}"))
        };
        tokio::time::timeout(self.config.connect_timeout, attempt)
            .await
            .unwrap_or_else(|_| Err("worker info probe timed out".to_string()))
    }

    /// Reserve the least-loaded healthy worker that serves `model` on
    /// `path`, skipping `exclude`.
    fn try_pick(self: &Arc<Self>, path: &str, model: Option<&str>, exclude: &[usize]) -> Pick {
        let _lock = self.pick_lock.lock().unwrap_or_else(|p| p.into_inner());
        let n = self.workers.len();
        let start = self.rotation.fetch_add(1, Ordering::Relaxed) % n;
        let mut model_known = false;
        let mut any_eligible = false;
        let mut any_excluded = false;
        // (index, load ratio, load)
        let mut best: Option<(usize, f64, usize)> = None;
        for k in 0..n {
            let i = (start + k) % n;
            let w = &self.workers[i];
            let st = w.status();
            let Some(info) = st.info.as_ref() else {
                continue;
            };
            let own = info.slots_of(model);
            if model.is_some_and(|m| !info.model_ids().any(|id| id == m))
                || !info.capabilities.serves(path)
                || (path == "/v1/embeddings" && own.is_some_and(|s| !s.embeddings))
            {
                continue;
            }
            model_known = true;
            if !st.healthy {
                continue;
            }
            if exclude.contains(&i) {
                any_excluded = true;
                continue;
            }
            any_eligible = true;
            let worker_load = w.dispatched.load(Ordering::Acquire) + st.external_in_flight;
            // A model with its own figures is admitted by them; the
            // worker's other models do not use its slots.  A configured
            // per-worker cap still bounds the worker as a whole.
            let (slots, load) = match own {
                Some(s) => {
                    if self
                        .config
                        .worker_slots
                        .is_some_and(|cap| worker_load >= cap.max(1))
                    {
                        continue;
                    }
                    (
                        s.max_concurrency.max(1),
                        w.by_model().get(&s.id).copied().unwrap_or(0)
                            + st.external_by_model.get(&s.id).copied().unwrap_or(0),
                    )
                }
                None => (self.slots_for(info), worker_load),
            };
            if load >= slots {
                continue;
            }
            let ratio = load as f64 / slots as f64;
            if best.is_none_or(|(_, r, l)| ratio < r || (ratio == r && load < l)) {
                best = Some((i, ratio, load));
            }
        }
        if let Some((index, _, _)) = best {
            let w = &self.workers[index];
            w.dispatched.fetch_add(1, Ordering::AcqRel);
            if let Some(m) = model {
                *w.by_model().entry(m.to_string()).or_default() += 1;
            }
            return Pick::Slot(SlotGuard {
                coordinator: Arc::clone(self),
                index,
                model: model.map(str::to_string),
            });
        }
        if any_eligible {
            return Pick::Busy;
        }
        if any_excluded {
            return Pick::Reject(Reject::Exhausted);
        }
        if !model_known {
            if let Some(m) = model {
                // Only a definite answer when every worker has reported.
                if self.workers.iter().all(|w| w.status().info.is_some()) {
                    return Pick::Reject(Reject::UnknownModel(m.to_string(), self.models()));
                }
            }
        }
        Pick::Reject(Reject::Unavailable)
    }

    /// Reserve a slot, waiting in the bounded queue while every eligible
    /// worker is full.
    async fn acquire(
        self: &Arc<Self>,
        path: &str,
        model: Option<&str>,
        exclude: &[usize],
    ) -> Result<SlotGuard, Reject> {
        match self.try_pick(path, model, exclude) {
            Pick::Slot(slot) => return Ok(slot),
            Pick::Reject(r) => return Err(r),
            Pick::Busy => {}
        }
        if self.queued.fetch_add(1, Ordering::AcqRel) >= self.config.queue_depth {
            self.queued.fetch_sub(1, Ordering::AcqRel);
            return Err(Reject::QueueFull);
        }
        let _queued = QueueGuard(&self.queued);
        let deadline = tokio::time::Instant::now() + self.config.queue_timeout;
        loop {
            // Register for the wake-up before looking, so a release between
            // the look and the wait is not missed.
            let notified = self.released.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            match self.try_pick(path, model, exclude) {
                Pick::Slot(slot) => return Ok(slot),
                Pick::Reject(r) => return Err(r),
                Pick::Busy => {}
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                return Err(Reject::QueueTimeout);
            }
        }
    }

    /// Model ids reported by healthy workers, sorted and deduplicated.
    fn models(&self) -> Vec<String> {
        let mut models: Vec<String> = self
            .workers
            .iter()
            .flat_map(|w| {
                let st = w.status();
                st.info
                    .as_ref()
                    .filter(|_| st.healthy)
                    .map(|i| i.model_ids().map(str::to_string).collect::<Vec<_>>())
                    .unwrap_or_default()
            })
            .collect();
        models.sort();
        models.dedup();
        models
    }

    /// Open a connection to `w` and send one request.  The returned guard
    /// owns the connection; dropping it closes the socket.  The error's flag
    /// says whether the request may have reached the worker.
    async fn send(
        &self,
        w: &Worker,
        method: Method,
        path: &str,
        extra: &HeaderMap,
        body: Bytes,
    ) -> Result<(hyper::Response<Incoming>, AbortOnDrop), (bool, String)> {
        let connect = TcpStream::connect(&w.endpoint.authority);
        let tcp = match tokio::time::timeout(self.config.connect_timeout, connect).await {
            Ok(Ok(tcp)) => tcp,
            Ok(Err(e)) => return Err((false, format!("connect failed: {e}"))),
            Err(_) => return Err((false, "connect timed out".to_string())),
        };
        let _ = tcp.set_nodelay(true);
        let (mut sender, conn) = hyper::client::conn::http1::handshake(TokioIo::new(tcp))
            .await
            .map_err(|e| (false, format!("handshake failed: {e}")))?;
        let conn = AbortOnDrop(tokio::spawn(async move {
            let _ = conn.await;
        }));
        let mut builder = hyper::Request::builder()
            .method(method)
            .uri(path)
            .header(header::HOST, &w.endpoint.authority);
        for (name, value) in extra {
            builder = builder.header(name, value);
        }
        if let Some(key) = &self.config.worker_api_key {
            builder = builder.header(header::AUTHORIZATION, format!("Bearer {key}"));
        }
        let request = builder
            .body(Full::new(body))
            .map_err(|e| (false, format!("building request: {e}")))?;
        let response = sender
            .send_request(request)
            .await
            .map_err(|e| (true, format!("request failed: {e}")))?;
        Ok((response, conn))
    }

    /// Forward one attempt of a routed request to the reserved worker.
    async fn forward(
        &self,
        slot: SlotGuard,
        path: &str,
        body: Bytes,
        stream: bool,
        log: RequestLog,
    ) -> Result<Response, Failure> {
        let w = &self.workers[slot.index];
        let mut extra = HeaderMap::new();
        extra.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        if let Ok(id) = HeaderValue::from_str(&log.request_id) {
            extra.insert(REQUEST_ID_HEADER, id);
        }
        let (response, conn) = match self.send(w, Method::POST, path, &extra, body).await {
            Ok(ok) => ok,
            Err((_, reason)) => {
                w.mark_unhealthy(reason.clone());
                return Err(Failure::Retry {
                    reason,
                    admission: false,
                });
            }
        };
        let status = response.status();
        if !status.is_success() {
            let content_type = response.headers().get(header::CONTENT_TYPE).cloned();
            let body = Limited::new(response.into_body(), MAX_SMALL_BODY)
                .collect()
                .await
                .map(|c| c.to_bytes())
                .unwrap_or_default();
            if status == StatusCode::SERVICE_UNAVAILABLE {
                // The worker refused admission before starting generation.
                return Err(Failure::Retry {
                    reason: "worker at capacity".to_string(),
                    admission: true,
                });
            }
            if matches!(status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) {
                w.mark_unhealthy(format!("worker rejected credentials ({status})"));
                return Err(Failure::Final(error_response(
                    StatusCode::BAD_GATEWAY,
                    "worker rejected the coordinator's credentials",
                    "server_error",
                    None,
                )));
            }
            log.finish(status, "worker_error");
            let mut out = Response::builder().status(status);
            if let Some(ct) = content_type {
                out = out.header(header::CONTENT_TYPE, ct);
            }
            let out = out.body(Body::from(body)).unwrap_or_default();
            return Err(Failure::Final(log.decorate(out)));
        }

        if !stream {
            let content_type = response.headers().get(header::CONTENT_TYPE).cloned();
            let body = match Limited::new(response.into_body(), MAX_BUFFERED_BODY)
                .collect()
                .await
            {
                Ok(c) => c.to_bytes(),
                Err(e) => {
                    let reason = format!("worker response lost: {e}");
                    w.mark_unhealthy(reason.clone());
                    return Err(Failure::Retry {
                        reason,
                        admission: false,
                    });
                }
            };
            drop(conn);
            drop(slot);
            log.finish(status, "complete");
            let mut out = Response::builder().status(status);
            if let Some(ct) = content_type {
                out = out.header(header::CONTENT_TYPE, ct);
            }
            let out = out.body(Body::from(body)).unwrap_or_default();
            return Ok(log.decorate(out));
        }

        // Streaming: hold the response until the first whole event arrives,
        // so a worker lost before any output can still be retried.
        let mut body = response.into_body();
        let mut framer = SseFramer::default();
        let first = loop {
            match body.frame().await {
                Some(Ok(frame)) => {
                    if let Ok(data) = frame.into_data() {
                        if let Some(events) = framer.push(&data) {
                            break events;
                        }
                    }
                }
                Some(Err(e)) => {
                    let reason = format!("worker stream lost before output: {e}");
                    w.mark_unhealthy(reason.clone());
                    return Err(Failure::Retry {
                        reason,
                        admission: false,
                    });
                }
                None => match std::mem::take(&mut framer).finish() {
                    Some(rest) => break rest,
                    None => {
                        return Err(Failure::Retry {
                            reason: "worker closed the stream before output".to_string(),
                            admission: false,
                        })
                    }
                },
            }
        };
        let headers = log.headers();
        let state = StreamState {
            body: Some(body),
            framer,
            first: Some(first),
            events: 0,
            outcome: "client_disconnected",
            log,
            _conn: conn,
            slot,
        };
        let events = futures_util::stream::unfold(state, |mut st| async move {
            let chunk = st.next_chunk().await?;
            Some((Ok::<Bytes, std::io::Error>(chunk), st))
        });
        let mut out = Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache");
        for (name, value) in headers.iter() {
            out = out.header(name, value);
        }
        Ok(out.body(Body::from_stream(events)).unwrap_or_default())
    }

    /// Route one request: admission, forwarding, and the retry policy.
    async fn route(
        self: &Arc<Self>,
        path: &'static str,
        headers: &HeaderMap,
        body: Bytes,
    ) -> Response {
        let request_id = request_id_from(headers);
        let parsed: serde_json::Value = match serde_json::from_slice(&body) {
            Ok(v) => v,
            Err(e) => {
                return with_request_id(
                    error_response(
                        StatusCode::BAD_REQUEST,
                        &format!("request body is not valid JSON: {e}"),
                        "invalid_request_error",
                        None,
                    ),
                    &request_id,
                )
            }
        };
        let model = parsed
            .get("model")
            .and_then(|m| m.as_str())
            .filter(|m| !m.is_empty())
            .map(str::to_string);
        let stream = parsed
            .get("stream")
            .and_then(|s| s.as_bool())
            .unwrap_or(false);
        drop(parsed);

        let arrived = Instant::now();
        let mut tried: Vec<usize> = Vec::new();
        let mut last_admission_only = true;
        loop {
            let slot = match self.acquire(path, model.as_deref(), &tried).await {
                Ok(slot) => slot,
                Err(reject) => {
                    let reject = match reject {
                        Reject::Exhausted if last_admission_only => Reject::AllRefused,
                        other => other,
                    };
                    tracing::info!(
                        request_id = %request_id,
                        path,
                        attempts = tried.len(),
                        queue_ms = arrived.elapsed().as_millis() as u64,
                        outcome = ?reject,
                        "request rejected"
                    );
                    return with_request_id(reject_response(reject), &request_id);
                }
            };
            let index = slot.index;
            let log = RequestLog {
                request_id: request_id.clone(),
                worker: self.workers[index].endpoint.base.clone(),
                path,
                attempt: tried.len() + 1,
                queue_ms: arrived.elapsed().as_millis() as u64,
                started: Instant::now(),
            };
            match self.forward(slot, path, body.clone(), stream, log).await {
                Ok(response) | Err(Failure::Final(response)) => return response,
                Err(Failure::Retry { reason, admission }) => {
                    tracing::warn!(
                        request_id = %request_id,
                        worker = %self.workers[index].endpoint.base,
                        attempt = tried.len() + 1,
                        reason = %reason,
                        "attempt failed before output; trying another worker"
                    );
                    last_admission_only &= admission;
                    tried.push(index);
                }
            }
        }
    }
}

/// Per-attempt record for logs and response headers (no request content).
struct RequestLog {
    request_id: String,
    worker: String,
    path: &'static str,
    attempt: usize,
    queue_ms: u64,
    started: Instant,
}

impl RequestLog {
    fn finish(&self, status: StatusCode, outcome: &str) {
        tracing::info!(
            request_id = %self.request_id,
            worker = %self.worker,
            path = self.path,
            attempt = self.attempt,
            queue_ms = self.queue_ms,
            duration_ms = self.started.elapsed().as_millis() as u64,
            status = status.as_u16(),
            outcome,
            "routed request finished"
        );
    }

    fn headers(&self) -> HeaderMap {
        let mut h = HeaderMap::new();
        if let Ok(v) = HeaderValue::from_str(&self.request_id) {
            h.insert(REQUEST_ID_HEADER, v);
        }
        if let Ok(v) = HeaderValue::from_str(&self.worker) {
            h.insert(WORKER_HEADER, v);
        }
        h
    }

    fn decorate(&self, mut response: Response) -> Response {
        response.headers_mut().extend(self.headers());
        response
    }
}

/// A forwarded SSE stream: owns the worker connection and the slot, so the
/// client hanging up (axum dropping the body) closes the worker connection
/// and frees the slot.
struct StreamState {
    body: Option<Incoming>,
    framer: SseFramer,
    first: Option<Bytes>,
    events: usize,
    outcome: &'static str,
    log: RequestLog,
    _conn: AbortOnDrop,
    slot: SlotGuard,
}

impl StreamState {
    async fn next_chunk(&mut self) -> Option<Bytes> {
        if let Some(first) = self.first.take() {
            self.events += 1;
            return Some(first);
        }
        let body = self.body.as_mut()?;
        loop {
            match body.frame().await {
                Some(Ok(frame)) => {
                    if let Ok(data) = frame.into_data() {
                        if let Some(events) = self.framer.push(&data) {
                            self.events += 1;
                            return Some(events);
                        }
                    }
                }
                Some(Err(e)) => {
                    // Output already reached the client: never restart.
                    // End with an error event (and no `[DONE]`).
                    self.body = None;
                    self.outcome = "worker_lost";
                    self.slot.coordinator.workers[self.slot.index]
                        .mark_unhealthy(format!("stream lost mid-response: {e}"));
                    return Some(Bytes::from_static(MID_STREAM_ERROR));
                }
                None => {
                    self.body = None;
                    self.outcome = "complete";
                    return std::mem::take(&mut self.framer).finish();
                }
            }
        }
    }
}

impl Drop for StreamState {
    fn drop(&mut self) {
        tracing::info!(
            request_id = %self.log.request_id,
            worker = %self.log.worker,
            path = self.log.path,
            attempt = self.log.attempt,
            queue_ms = self.log.queue_ms,
            duration_ms = self.log.started.elapsed().as_millis() as u64,
            chunks = self.events,
            outcome = self.outcome,
            "routed stream finished"
        );
    }
}

/// SSE event sent when a worker is lost after output started.
const MID_STREAM_ERROR: &[u8] = b"data: {\"error\":{\"message\":\"the worker serving this request was lost after output started; the response is incomplete\",\"type\":\"server_error\",\"param\":null,\"code\":\"worker_lost\"}}\n\n";

/// Splits a byte stream into whole SSE events (terminated by a blank line),
/// so a client never sees half an event — in particular not when a worker
/// is lost mid-event.
#[derive(Default)]
struct SseFramer {
    buf: Vec<u8>,
}

impl SseFramer {
    /// Add bytes; return every event completed so far.
    fn push(&mut self, data: &[u8]) -> Option<Bytes> {
        self.buf.extend_from_slice(data);
        let end = self.buf.windows(2).rposition(|w| w == b"\n\n")? + 2;
        let rest = self.buf.split_off(end);
        Some(Bytes::from(std::mem::replace(&mut self.buf, rest)))
    }

    /// Whatever is left at a clean end of stream.
    fn finish(self) -> Option<Bytes> {
        (!self.buf.is_empty()).then(|| Bytes::from(self.buf))
    }
}

/// The client's `x-request-id` when well formed, else a fresh one.
fn request_id_from(headers: &HeaderMap) -> String {
    headers
        .get(REQUEST_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .filter(|v| {
            !v.is_empty()
                && v.len() <= 128
                && v.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        })
        .map(str::to_string)
        .unwrap_or_else(|| format!("req-{}", Uuid::new_v4().simple()))
}

fn with_request_id(mut response: Response, request_id: &str) -> Response {
    if let Ok(v) = HeaderValue::from_str(request_id) {
        response.headers_mut().insert(REQUEST_ID_HEADER, v);
    }
    response
}

fn error_response(status: StatusCode, message: &str, kind: &str, code: Option<&str>) -> Response {
    let mut body = ErrorResponse::new(message, kind);
    body.error.code = code.map(str::to_string);
    let mut response = (status, Json(body)).into_response();
    if matches!(
        status,
        StatusCode::TOO_MANY_REQUESTS | StatusCode::SERVICE_UNAVAILABLE
    ) {
        response
            .headers_mut()
            .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
    }
    response
}

fn reject_response(reject: Reject) -> Response {
    match reject {
        Reject::UnknownModel(model, available) => error_response(
            StatusCode::NOT_FOUND,
            &format!(
                "model '{model}' is not served by any worker (available: {})",
                if available.is_empty() {
                    "none".to_string()
                } else {
                    available.join(", ")
                }
            ),
            "invalid_request_error",
            Some("model_not_found"),
        ),
        Reject::Unavailable => error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "no healthy worker serves this request",
            "unavailable",
            Some("no_worker"),
        ),
        Reject::QueueFull => error_response(
            StatusCode::TOO_MANY_REQUESTS,
            "all workers are busy and the request queue is full; retry shortly",
            "overloaded",
            Some("queue_full"),
        ),
        Reject::QueueTimeout => error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "all workers stayed busy; retry shortly",
            "overloaded",
            Some("queue_timeout"),
        ),
        Reject::AllRefused => error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "every eligible worker is at capacity; retry shortly",
            "overloaded",
            Some("workers_busy"),
        ),
        Reject::Exhausted => error_response(
            StatusCode::BAD_GATEWAY,
            "every eligible worker failed before producing output",
            "server_error",
            Some("workers_failed"),
        ),
    }
}

// ─── HTTP front end ──────────────────────────────────────────────────────────

/// The coordinator's router: the routed `/v1` endpoints plus `GET /health`,
/// `GET /v1/models` (union of healthy workers' models) and `GET /v1/workers`
/// (registry status).
pub fn create_router(coordinator: Arc<Coordinator>) -> Router {
    let mut api = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/workers", get(list_workers));
    for path in ROUTED_PATHS {
        api = api.route(
            path,
            post(
                move |State(c): State<Arc<Coordinator>>, headers: HeaderMap, body: Bytes| async move {
                    c.route(path, &headers, body).await
                },
            ),
        );
    }
    let api = api.layer(middleware::from_fn_with_state(
        Arc::clone(&coordinator),
        require_api_key,
    ));
    Router::new()
        .route("/health", get(health))
        .merge(api)
        .layer(CorsLayer::permissive())
        .with_state(coordinator)
}

async fn require_api_key(State(c): State<Arc<Coordinator>>, req: Request, next: Next) -> Response {
    if let Some(expected) = &c.config.api_key {
        let provided = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));
        if !provided
            .is_some_and(|k| crate::server::api_keys_match(k.as_bytes(), expected.as_bytes()))
        {
            return error_response(
                StatusCode::UNAUTHORIZED,
                "invalid or missing API key — pass the key as 'Authorization: Bearer <key>'",
                "invalid_request_error",
                None,
            );
        }
    }
    next.run(req).await
}

async fn health(State(c): State<Arc<Coordinator>>) -> Response {
    let healthy = c.workers.iter().filter(|w| w.status().healthy).count();
    let status = if healthy > 0 {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    let body = json!({
        "status": if healthy > 0 { "ok" } else { "unavailable" },
        "healthy_workers": healthy,
        "workers": c.workers.len(),
    });
    (status, Json(body)).into_response()
}

async fn list_models(State(c): State<Arc<Coordinator>>) -> Json<ModelListResponse> {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    Json(ModelListResponse {
        object: "list".to_string(),
        data: c
            .models()
            .into_iter()
            .map(|id| ModelInfo {
                id,
                object: "model".to_string(),
                created,
                owned_by: "joshua".to_string(),
            })
            .collect(),
    })
}

async fn list_workers(State(c): State<Arc<Coordinator>>) -> Json<serde_json::Value> {
    Json(json!({ "object": "list", "data": c.status_report() }))
}

/// Run a coordinator on `addr`: probe the workers once, keep probing in the
/// background, and serve until the listener fails.
pub async fn serve(config: CoordinatorConfig, addr: &str) -> std::io::Result<()> {
    let coordinator = Coordinator::new(config)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    coordinator.refresh().await;
    let healthy = coordinator
        .workers
        .iter()
        .filter(|w| w.status().healthy)
        .count();
    tracing::info!(
        "joshua route: {healthy}/{} workers healthy",
        coordinator.workers.len()
    );
    let _health = AbortOnDrop(coordinator.spawn_health_loop());
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("Joshua request router listening on http://{}", addr);
    axum::serve(listener, create_router(coordinator)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_endpoints_name_the_http_address() {
        let e = Endpoint::new("127.0.0.1:8081".parse().unwrap());
        assert_eq!(e.base, "http://127.0.0.1:8081");
        assert_eq!(e.authority, "127.0.0.1:8081");
        let e = Endpoint::new("[::1]:9000".parse().unwrap());
        assert_eq!(e.authority, "[::1]:9000");
    }

    #[test]
    fn duplicate_or_missing_workers_are_rejected() {
        assert!(Coordinator::new(CoordinatorConfig::new(vec![])).is_err());
        let addr: SocketAddr = "127.0.0.1:1".parse().unwrap();
        let dup = vec![addr, addr];
        assert!(Coordinator::new(CoordinatorConfig::new(dup)).is_err());
    }

    #[test]
    fn sse_framer_only_releases_whole_events() {
        let mut f = SseFramer::default();
        assert_eq!(f.push(b"data: a"), None);
        assert_eq!(
            f.push(b"bc\n\ndata: d").as_deref(),
            Some(&b"data: abc\n\n"[..])
        );
        assert_eq!(
            f.push(b"\n\ndata: e\n\ndata").as_deref(),
            Some(&b"data: d\n\ndata: e\n\n"[..])
        );
        assert_eq!(f.finish().as_deref(), Some(&b"data"[..]));
        assert_eq!(SseFramer::default().finish(), None);
    }

    #[test]
    fn request_ids_are_kept_only_when_well_formed() {
        let mut h = HeaderMap::new();
        h.insert(REQUEST_ID_HEADER, HeaderValue::from_static("abc-123_x.y"));
        assert_eq!(request_id_from(&h), "abc-123_x.y");
        h.insert(REQUEST_ID_HEADER, HeaderValue::from_static("bad id"));
        assert!(request_id_from(&h).starts_with("req-"));
        assert!(request_id_from(&HeaderMap::new()).starts_with("req-"));
    }
}
