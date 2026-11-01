//! `joshua route` (request-level routing across independent workers, #155).
//!
//! Most tests drive the coordinator against in-process fake workers that
//! speak the worker protocol (`/v1/worker/info` plus the OpenAI routes), so
//! routing, saturation, worker loss, cancellation and stream ordering are
//! exercised without a model.  The last tests put a real `joshua` server on
//! a tiny synthetic GGUF behind the router and compare greedy output direct
//! versus routed.

mod common;

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::StreamExt;
use http_body_util::{BodyExt, Full};
use hyper_util::rt::TokioIo;
use joshua::coordinator::{
    create_router, Coordinator, CoordinatorConfig, JobModel, ModelSlots, WorkerCapabilities, WorkerInfo,
    REQUEST_ID_HEADER, WORKER_HEADER, WORKER_PROTOCOL_VERSION,
};
use serde_json::{json, Value};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Notify;

// ─── Fake workers ────────────────────────────────────────────────────────────

/// What a fake worker does with a chat/completion request.
#[derive(Clone)]
enum Behavior {
    /// Answer JSON echoing the received body (or SSE events for a stream).
    Echo,
    /// Wait for `release` before answering (holds a slot).
    Gate,
    /// Never answer; record when the request is dropped (cancellation).
    Hang,
    /// Stream one event, then hang; record when the stream is dropped.
    StreamThenHang,
    /// Reject admission with 503, as an engine at capacity does.
    Reject503,
    /// Fail with a 500 (an execution error).
    Fail500,
    LoseJsonBody,
    /// Start a 200 stream and lose the connection before any event.
    LoseBeforeOutput,
    /// Stream two events, then lose the connection.
    LoseMidStream,
    /// Send stream headers, then nothing; record when dropped.
    HeadersThenHang,
    /// Stream an event, pause `gap`, then finish normally.
    SlowStream(Duration),
    /// Stream two events and half of a third, then close cleanly without
    /// `[DONE]`.
    CloseWithoutDone,
    /// Send half an event, then close cleanly.
    PartialThenClose,
}

struct Fake {
    audio_active: AtomicUsize,
    external_audio: AtomicUsize,
    embedding_jobs: Mutex<Vec<JobModel>>,
    completed: Mutex<Vec<(String, std::time::Instant)>>,
    bad_indexes: AtomicBool,
    residency: Mutex<Option<joshua::affinity::CacheResidency>>,
    epoch: Mutex<String>,
    name: String,
    model: String,
    max_concurrency: usize,
    n_ctx: AtomicUsize,
    models: Mutex<Vec<ModelSlots>>,
    behavior: Mutex<Behavior>,
    /// Requests received on the routed endpoints.
    received: AtomicUsize,
    active: AtomicUsize,
    release: Notify,
    /// Set when a hanging request/stream is dropped by the server.
    dropped: AtomicBool,
    last_body: Mutex<Option<Bytes>>,
    last_request_id: Mutex<Option<String>>,
    /// Whether the fake advertises `/v1/audio/transcriptions` (a real worker
    /// does so only when a Whisper model is loaded).
    transcriptions: AtomicBool,
    /// The `Content-Type` of the last transcription upload.
    last_content_type: Mutex<Option<String>>,
}

struct ActiveGuard(Arc<Fake>);

struct AudioGuard(Arc<Fake>);

impl Drop for AudioGuard {
    fn drop(&mut self) {
        self.0.audio_active.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::SeqCst);
        self.0.dropped.store(true, Ordering::SeqCst);
    }
}

fn sse(data: &str) -> Bytes {
    Bytes::from(format!("data: {data}\n\n"))
}

async fn fake_info(State(f): State<Arc<Fake>>) -> Json<WorkerInfo> {
    Json(WorkerInfo {
        transcription_job: None,
        transcription_in_flight: f.audio_active.load(Ordering::SeqCst)
            + f.external_audio.load(Ordering::SeqCst),
        embedding_jobs: f.embedding_jobs.lock().unwrap().clone(),
        protocol_version: WORKER_PROTOCOL_VERSION,
        cache_residency: f
            .residency
            .lock()
            .unwrap()
            .clone()
            .map(|residency| [(f.model.clone(), residency)].into())
            .unwrap_or_default(),
        cache_epochs: [(f.model.clone(), f.epoch.lock().unwrap().clone())].into(),
        model: f.model.clone(),
        models: f.models.lock().unwrap().clone(),
        n_ctx: f.n_ctx.load(Ordering::SeqCst) as u32,
        backend: "cpu".to_string(),
        max_concurrency: f.max_concurrency,
        in_flight: f.active.load(Ordering::SeqCst)
            .saturating_sub(f.audio_active.load(Ordering::SeqCst)),
        capabilities: WorkerCapabilities {
            chat: true,
            completions: true,
            embeddings: true,
            transcriptions: f.transcriptions.load(Ordering::SeqCst),
            streaming: true,
            tools: true,
        },
    })
}

/// `POST /v1/audio/transcriptions` on a fake worker: record the raw upload
/// and the `Content-Type` it arrived with, then answer like Whisper does.
async fn fake_transcribe(
    State(f): State<Arc<Fake>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    f.received.fetch_add(1, Ordering::SeqCst);
    *f.last_body.lock().unwrap() = Some(body.clone());
    *f.last_request_id.lock().unwrap() = headers
        .get(REQUEST_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    *f.last_content_type.lock().unwrap() = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    f.active.fetch_add(1, Ordering::SeqCst);
    let _guard = ActiveGuard(f.clone());
    f.audio_active.fetch_add(1, Ordering::SeqCst);
    let _audio_guard = AudioGuard(f.clone());
    let behavior = f.behavior.lock().unwrap().clone();
    match behavior {
        Behavior::Hang => std::future::pending::<()>().await,
        Behavior::Gate => f.release.notified().await,
        _ => {},
    }
    Json(json!({"text": format!("from {}", f.name), "worker": f.name})).into_response()
}

async fn fake_generate(
    State(f): State<Arc<Fake>>,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    f.received.fetch_add(1, Ordering::SeqCst);
    *f.last_body.lock().unwrap() = Some(body.clone());
    *f.last_request_id.lock().unwrap() = headers
        .get(REQUEST_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let req: Value = serde_json::from_slice(&body).unwrap();
    if let Some(expected) = f.embedding_jobs.lock().unwrap().first() {
        if headers.get(joshua::coordinator::JOB_OWNER_HEADER)
            .is_some_and(|owner| owner.as_bytes() != expected.owner.as_bytes()) {
            return StatusCode::CONFLICT.into_response();
        }
    }
    // Like a joshua worker, only chat completions stream; the other routes
    // answer JSON whatever `stream` says.
    let stream = uri.path() == "/v1/chat/completions" && req["stream"].as_bool().unwrap_or(false);
    let behavior = f.behavior.lock().unwrap().clone();
    f.active.fetch_add(1, Ordering::SeqCst);
    f.dropped.store(false, Ordering::SeqCst);
    let guard = ActiveGuard(Arc::clone(&f));
    let events = |n: usize| -> Vec<Bytes> {
        (0..n)
            .map(|i| sse(&json!({"worker": f.name, "i": i}).to_string()))
            .chain(std::iter::once(sse("[DONE]")))
            .collect()
    };
    let sse_response =
        |s: futures_util::stream::BoxStream<'static, Result<Bytes, std::io::Error>>| {
            Response::builder()
                .header(header::CONTENT_TYPE, "text/event-stream")
                .body(Body::from_stream(s))
                .unwrap()
        };
    match behavior {
        Behavior::Echo | Behavior::Gate => {
            if matches!(behavior, Behavior::Gate) {
                f.release.notified().await;
            }
            drop(guard);
            if stream {
                // Split every event across two body chunks to exercise
                // the router's event framing.
                let chunks: Vec<Result<Bytes, std::io::Error>> = events(20)
                    .into_iter()
                    .flat_map(|e| {
                        let (a, b) = e.split_at(e.len() / 2);
                        [Ok(Bytes::copy_from_slice(a)), Ok(Bytes::copy_from_slice(b))]
                    })
                    .collect();
                sse_response(futures_util::stream::iter(chunks).boxed())
            } else {
                if uri.path() == "/v1/embeddings" && !f.embedding_jobs.lock().unwrap().is_empty() {
                    let inputs = req["input"].as_array().unwrap();
                    if inputs[0].as_str().unwrap().starts_with("slow") {
                        tokio::time::sleep(Duration::from_millis(80)).await;
                    }
                    let data: Vec<_> = inputs.iter().enumerate().rev().map(|(index, input)| {
                        let text = input.as_str().unwrap();
                        json!({"object": "embedding", "index": if f.bad_indexes.load(Ordering::SeqCst) { 0 } else { index },
                            "embedding": [text.len() as f32, text.bytes().map(u64::from).sum::<u64>() as f32]})
                    }).collect();
                    f.completed.lock().unwrap().push((inputs[0].as_str().unwrap().to_string(), std::time::Instant::now()));
                    let tokens: usize = inputs.iter().map(|input| input.as_str().unwrap().len()).sum();
                    return Json(json!({"object": "list", "model": f.model, "data": data,
                        "usage": {"prompt_tokens": tokens, "completion_tokens": 0, "total_tokens": tokens}})).into_response();
                }
                Json(json!({
                    "worker": f.name,
                    "echo": req,
                    "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
                }))
                .into_response()
            }
        }
        Behavior::Hang => {
            let _guard = guard;
            std::future::pending::<Response>().await
        }
        Behavior::StreamThenHang => {
            let first = futures_util::stream::once(async { Ok(sse("{\"i\":0}")) });
            let rest = futures_util::stream::once(async move {
                let _guard = guard;
                std::future::pending::<()>().await;
                Ok(Bytes::new())
            });
            sse_response(first.chain(rest).boxed())
        }
        Behavior::Reject503 => {
            drop(guard);
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": {"message": "at capacity", "type": "overloaded"}})),
            )
                .into_response()
        }
        Behavior::Fail500 => {
            drop(guard);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": {"message": "internal error", "type": "server_error"}})),
            )
                .into_response()
        }
        Behavior::LoseJsonBody => {
            drop(guard);
            let first = futures_util::stream::once(async { Ok::<_, std::io::Error>(Bytes::from_static(b"{\"data\":")) });
            let last = futures_util::stream::once(async {
                tokio::time::sleep(Duration::from_millis(25)).await;
                Err::<Bytes, _>(std::io::Error::other("worker crashed after execution"))
            });
            Response::builder().header(header::CONTENT_TYPE, "application/json")
                .body(Body::from_stream(first.chain(last))).unwrap()
        }
        Behavior::LoseBeforeOutput => {
            drop(guard);
            let s = futures_util::stream::once(async {
                Err::<Bytes, _>(std::io::Error::other("worker crashed"))
            });
            sse_response(s.boxed())
        }
        Behavior::LoseMidStream => {
            drop(guard);
            let items: Vec<Result<Bytes, std::io::Error>> =
                events(2).into_iter().take(2).map(Ok).collect();
            // Pause so the events are flushed before the connection drops
            // (hyper discards buffered output when a body errors).
            let crash = futures_util::stream::once(async {
                tokio::time::sleep(Duration::from_millis(100)).await;
                Err(std::io::Error::other("worker crashed"))
            });
            sse_response(futures_util::stream::iter(items).chain(crash).boxed())
        }
        Behavior::HeadersThenHang => {
            let s = futures_util::stream::once(async move {
                let _guard = guard;
                std::future::pending::<()>().await;
                Ok(Bytes::new())
            });
            sse_response(s.boxed())
        }
        Behavior::SlowStream(gap) => {
            drop(guard);
            let mut events = events(2).into_iter();
            let first = events.next().unwrap();
            let rest: Vec<Result<Bytes, std::io::Error>> = events.map(Ok).collect();
            let s = futures_util::stream::once(async { Ok(first) })
                .chain(futures_util::stream::once(async move {
                    tokio::time::sleep(gap).await;
                    Ok(Bytes::new())
                }))
                .chain(futures_util::stream::iter(rest));
            sse_response(s.boxed())
        }
        Behavior::CloseWithoutDone => {
            drop(guard);
            let mut items: Vec<Result<Bytes, std::io::Error>> =
                events(2).into_iter().take(2).map(Ok).collect();
            items.push(Ok(Bytes::from_static(b"data: {\"partial\":")));
            sse_response(futures_util::stream::iter(items).boxed())
        }
        Behavior::PartialThenClose => {
            drop(guard);
            let s = futures_util::stream::once(async {
                Ok(Bytes::from_static(b"data: {\"partial\":"))
            });
            sse_response(s.boxed())
        }
    }
}

async fn spawn_fake(
    name: &str,
    model: &str,
    max_concurrency: usize,
    behavior: Behavior,
) -> (Arc<Fake>, String) {
    let (fake, url, _) = spawn_fake_handle(name, model, max_concurrency, behavior).await;
    (fake, url)
}

async fn spawn_fake_handle(
    name: &str,
    model: &str,
    max_concurrency: usize,
    behavior: Behavior,
) -> (Arc<Fake>, String, tokio::task::JoinHandle<()>) {
    let fake = Arc::new(Fake {
        audio_active: AtomicUsize::new(0),
        external_audio: AtomicUsize::new(0),
        embedding_jobs: Mutex::new(Vec::new()),
        completed: Mutex::new(Vec::new()),
        bad_indexes: AtomicBool::new(false),
        residency: Mutex::new(None),
        epoch: Mutex::new(uuid::Uuid::new_v4().to_string()),
        name: name.to_string(),
        model: model.to_string(),
        max_concurrency,
        n_ctx: AtomicUsize::new(4096),
        models: Mutex::new(Vec::new()),
        behavior: Mutex::new(behavior),
        received: AtomicUsize::new(0),
        active: AtomicUsize::new(0),
        release: Notify::new(),
        dropped: AtomicBool::new(false),
        last_body: Mutex::new(None),
        last_request_id: Mutex::new(None),
        transcriptions: AtomicBool::new(false),
        last_content_type: Mutex::new(None),
    });
    let app = Router::new()
        .route("/v1/worker/info", get(fake_info))
        .route("/v1/chat/completions", post(fake_generate))
        .route("/v1/completions", post(fake_generate))
        .route("/v1/embeddings", post(fake_generate))
        .route("/v1/audio/transcriptions", post(fake_transcribe))
        .with_state(Arc::clone(&fake));
    let (url, handle) = serve_app_handle(app).await;
    (fake, url, handle)
}

async fn serve_app(app: Router) -> String {
    serve_app_handle(app).await.0
}

/// Serve `app`; aborting the handle closes the listener (a lost worker).
async fn serve_app_handle(app: Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{addr}"), handle)
}

/// A URL nothing listens on.
async fn dead_url() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    format!("http://{addr}")
}

async fn spawn_router(config: CoordinatorConfig) -> (Arc<Coordinator>, String) {
    let coordinator = Coordinator::new(config).unwrap();
    coordinator.refresh().await;
    let url = serve_app(create_router(Arc::clone(&coordinator))).await;
    (coordinator, url)
}

fn config(workers: &[&String]) -> CoordinatorConfig {
    let mut c = CoordinatorConfig::new(
        workers
            .iter()
            .map(|url| url.trim_start_matches("http://").parse().unwrap())
            .collect(),
    );
    c.connect_timeout = Duration::from_millis(500);
    c
}

// ─── Minimal HTTP client ─────────────────────────────────────────────────────

struct Reply {
    status: StatusCode,
    headers: HeaderMap,
    body: Bytes,
}

impl Reply {
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body)
            .unwrap_or_else(|e| panic!("not JSON ({e}): {}", String::from_utf8_lossy(&self.body)))
    }

    fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).and_then(|v| v.to_str().ok())
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).to_string()
    }
}

async fn open(url: &str) -> hyper::client::conn::http1::SendRequest<Full<Bytes>> {
    let addr: SocketAddr = url.trim_start_matches("http://").parse().unwrap();
    let tcp = TcpStream::connect(addr).await.unwrap();
    let (sender, conn) = hyper::client::conn::http1::handshake(TokioIo::new(tcp))
        .await
        .unwrap();
    tokio::spawn(async move {
        let _ = conn.await;
    });
    sender
}

fn request(method: &str, path: &str, body: &str) -> hyper::Request<Full<Bytes>> {
    hyper::Request::builder()
        .method(method)
        .uri(path)
        .header(header::HOST, "localhost")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Full::new(Bytes::from(body.to_string())))
        .unwrap()
}

async fn call(url: &str, method: &str, path: &str, body: &str) -> Reply {
    let mut sender = open(url).await;
    let res = sender
        .send_request(request(method, path, body))
        .await
        .unwrap();
    let status = res.status();
    let headers = res.headers().clone();
    let body = res.into_body().collect().await.unwrap().to_bytes();
    Reply {
        status,
        headers,
        body,
    }
}

async fn chat(url: &str, body: Value) -> Reply {
    call(url, "POST", "/v1/chat/completions", &body.to_string()).await
}

async fn affinity_chat(url: &str, handle: &str, owner: &str) -> Reply {
    affinity_request(
        url,
        Some(handle),
        owner,
        &json!({
            "model": "alpha", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 4
        }),
    )
    .await
}

async fn affinity_request(url: &str, handle: Option<&str>, owner: &str, body: &Value) -> Reply {
    timed_affinity_request(url, handle, owner, body).await.0
}

async fn timed_affinity_request(
    url: &str,
    handle: Option<&str>,
    owner: &str,
    body: &Value,
) -> (Reply, f64, f64) {
    let started = std::time::Instant::now();
    let mut sender = open(url).await;
    let mut req = request("POST", "/v1/chat/completions", &body.to_string());
    if let Some(handle) = handle {
        req.headers_mut()
            .insert("x-joshua-affinity", handle.parse().unwrap());
    }
    req.headers_mut().insert(
        header::AUTHORIZATION,
        format!("Bearer {owner}").parse().unwrap(),
    );
    let response = sender.send_request(req).await.unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let mut incoming = response.into_body();
    let mut bytes = Vec::new();
    let mut ttft = None;
    while let Some(frame) = incoming.frame().await {
        if let Ok(data) = frame.unwrap().into_data() {
            bytes.extend_from_slice(&data);
            if ttft.is_none()
                && sse_data(&String::from_utf8_lossy(&bytes))
                    .iter()
                    .any(|data| {
                        serde_json::from_str::<Value>(data)
                            .ok()
                            .is_some_and(|event| {
                                ["content", "reasoning_content"].iter().any(|field| {
                                    event["choices"][0]["delta"][field]
                                        .as_str()
                                        .is_some_and(|value| !value.is_empty())
                                }) || !event["choices"][0]["finish_reason"].is_null()
                            })
                    })
            {
                ttft = Some(started.elapsed().as_secs_f64() * 1000.0);
            }
        }
    }
    let elapsed = started.elapsed().as_secs_f64() * 1000.0;
    (
        Reply {
            status,
            headers,
            body: Bytes::from(bytes),
        },
        ttft.unwrap_or(elapsed),
        elapsed,
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_worker_affinity_reuses_only_its_owned_scope_and_matches_cold_output() {
    let engine = Arc::new(tiny_engine("affinity-http"));
    let worker = serve_app(joshua::server::create_router(Arc::new(
        joshua::server::ServerState {
            api_key: Some("affinity-worker-test-key".into()),
            ..joshua::server::ServerState::new(engine.clone())
        },
    )))
    .await;
    let mut settings = config(&[&worker]);
    settings.worker_api_key = Some("affinity-worker-test-key".into());
    let (coordinator, router) = spawn_router(settings).await;
    let mut body = json!({"model": engine.model_name(), "messages": [{"role": "user", "content": "hello a b c"}], "temperature": 0.0, "max_tokens": 0});
    let first = affinity_request(&router, Some("new"), "alice", &body).await;
    assert_eq!(first.status, StatusCode::OK);
    let handle = first.header("x-joshua-affinity").unwrap();
    body["max_tokens"] = json!(4);
    body["messages"][0]["content"] = json!("hello a b c world");
    let warm = affinity_request(&router, Some(handle), "alice", &body).await;
    assert_eq!(warm.status, StatusCode::OK);
    assert!(engine.kv_reuse_count() > 0);
    assert!(engine.kv_tokens_reused() > 0);
    let before = engine.kv_reuse_count();
    let cold = affinity_request(&router, Some(handle), "bob", &body).await;
    assert_eq!(cold.status, StatusCode::OK);
    assert_eq!(comparable(&warm.json()), comparable(&cold.json()));
    assert_eq!(engine.kv_reuse_count(), before);
    assert_eq!(coordinator.affinity_stats().entries, 2);
    body["stream"] = json!(true);
    let streamed = affinity_request(&router, Some(handle), "alice", &body).await;
    assert_eq!(streamed.status, StatusCode::OK);
    assert_eq!(streamed.header("x-joshua-affinity"), Some(handle));
    let fresh_stream = affinity_request(&router, Some("new"), "alice", &body).await;
    assert_eq!(
        stream_text(&streamed.text()),
        stream_text(&fresh_stream.text())
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "manual tiny-model CPU locality measurement"]
async fn benchmark_tiny_affinity_policies() {
    use sha2::{Digest, Sha256};
    let mut rows = Vec::new();
    for repeat in 0..3 {
        for prefix_words in [8, 32, 64] {
            for concurrency in [1, 2, 4] {
                let mut policies = vec!["round_robin", "load", "session", "affinity"];
                if repeat % 2 == 1 {
                    policies.reverse();
                }
                for policy in policies {
                    let mut engines = Vec::new();
                    let mut workers = Vec::new();
                    let mut services = Vec::new();
                    let mut model_sha256 = String::new();
                    for index in 0..2 {
                        let directory = common::model_dir(&format!(
                            "affinity-bench-{repeat}-{prefix_words}-{concurrency}-{policy}-{index}"
                        ));
                        common::write_tiny_qwen3_layers(&directory.join("model.gguf"), 1);
                        model_sha256 = format!(
                            "{:x}",
                            Sha256::digest(std::fs::read(directory.join("model.gguf")).unwrap())
                        );
                        let options = joshua::EngineOptions::with_n_ctx(256)
                            .backend(joshua::ComputeBackend::Cpu)
                            .prefix_cache_bytes(1 << 20)
                            .prefix_cache_root_min(16);
                        let engine = Arc::new(
                            joshua::Engine::with_options(&directory, options)
                                .unwrap()
                                .with_max_concurrency(1),
                        );
                        let (url, service) = serve_app_handle(joshua::server::create_router(
                            Arc::new(joshua::server::ServerState {
                                api_key: Some("affinity-worker-test-key".into()),
                                ..joshua::server::ServerState::new(engine.clone())
                            }),
                        ))
                        .await;
                        engines.push(engine);
                        workers.push(url);
                        services.push(service);
                    }
                    let mut settings = config(&[&workers[0], &workers[1]]);
                    settings.worker_api_key = Some("affinity-worker-test-key".into());
                    let coordinator = Coordinator::new(settings).unwrap();
                    let (router, service) =
                        serve_app_handle(joshua::coordinator::create_router(coordinator.clone()))
                            .await;
                    services.push(service);
                    let handles: Vec<_> = (0..concurrency)
                        .map(|_| uuid::Uuid::new_v4().to_string())
                        .collect();
                    let baseline_directory = common::model_dir(&format!(
                        "affinity-bench-baseline-{repeat}-{prefix_words}-{concurrency}-{policy}"
                    ));
                    common::write_tiny_qwen3_layers(&baseline_directory.join("model.gguf"), 1);
                    let baseline_engine = Arc::new(
                        joshua::Engine::with_options(
                            &baseline_directory,
                            joshua::EngineOptions::with_n_ctx(256)
                                .backend(joshua::ComputeBackend::Cpu)
                                .prefix_cache_bytes(0),
                        )
                        .unwrap(),
                    );
                    let (baseline_url, baseline_service) =
                        serve_app_handle(joshua::server::create_router(Arc::new(
                            joshua::server::ServerState::new(baseline_engine),
                        )))
                        .await;
                    services.push(baseline_service);
                    let mut histories: Vec<Vec<Value>> = (0..concurrency).map(|conversation| vec![json!({"role": "system", "content": format!("{} b {}", "a ".repeat(conversation + 1), "hello world ".repeat(prefix_words / 2))}), json!({"role": "user", "content": "hello"})]).collect();
                    let mut durations = Vec::new();
                    let mut ttfts = Vec::new();
                    let queues: Vec<_> = (0..2)
                        .map(|_| Arc::new(tokio::sync::Semaphore::new(1)))
                        .collect();
                    let mut queue_delays = Vec::new();
                    let mut total_prompt = 0;
                    let mut requests_per_worker = [0usize; 2];
                    let mut request_hash = Sha256::new();
                    let mut output_hash = Sha256::new();
                    for turn in 0..6 {
                        coordinator.refresh().await;
                        let mut pending = Vec::new();
                        for conversation in 0..concurrency {
                            let worker = if policy == "round_robin" {
                                (turn * concurrency + conversation) % 2
                            } else {
                                conversation % 2
                            };
                            let direct = matches!(policy, "round_robin" | "session");
                            let target = if direct {
                                workers[worker].clone()
                            } else {
                                router.clone()
                            };
                            let handle = handles[conversation].clone();
                            let body = json!({"model": engines[0].model_name(), "messages": histories[conversation], "temperature": 0.0, "max_tokens": 1, "stream": true, "stream_options": {"include_usage": true}});
                            let encoded = serde_json::to_vec(&body).unwrap();
                            request_hash.update((encoded.len() as u64).to_le_bytes());
                            request_hash.update(encoded);
                            let queue = queues[worker].clone();
                            pending.push(tokio::spawn(async move {
                                let started = std::time::Instant::now();
                                let permit = if direct {
                                    Some(queue.acquire_owned().await.unwrap())
                                } else {
                                    None
                                };
                                let client_queue_ms = started.elapsed().as_secs_f64() * 1000.0;
                                let (response, ttft, _) = timed_affinity_request(
                                    &target,
                                    (policy != "load").then_some(handle.as_str()),
                                    if direct {
                                        "affinity-worker-test-key"
                                    } else {
                                        "bench"
                                    },
                                    &body,
                                )
                                .await;
                                let queue_ms = client_queue_ms
                                    + response
                                        .header("x-joshua-queue-ms")
                                        .and_then(|value| value.parse::<f64>().ok())
                                        .unwrap_or(0.0);
                                drop(permit);
                                assert_eq!(response.status, StatusCode::OK, "{}", response.text());
                                (
                                    conversation,
                                    worker,
                                    response.header(WORKER_HEADER).map(str::to_owned),
                                    started.elapsed().as_secs_f64() * 1000.0,
                                    response.text(),
                                    body,
                                    ttft + client_queue_ms,
                                    queue_ms,
                                )
                            }));
                        }
                        let mut wave = Vec::new();
                        for pending in pending {
                            wave.push(pending.await.unwrap());
                        }
                        for (
                            conversation,
                            direct_worker,
                            worker_header,
                            elapsed,
                            response,
                            body,
                            ttft,
                            queue_ms,
                        ) in wave
                        {
                            durations.push(elapsed);
                            ttfts.push(ttft);
                            queue_delays.push(queue_ms);
                            let worker = worker_header
                                .and_then(|worker| workers.iter().position(|url| url == &worker))
                                .unwrap_or(direct_worker);
                            requests_per_worker[worker] += 1;
                            total_prompt += sse_data(&response)
                                .into_iter()
                                .filter_map(|event| serde_json::from_str::<Value>(&event).ok())
                                .find_map(|event| event["usage"]["prompt_tokens"].as_u64())
                                .unwrap();
                            let baseline = affinity_request(
                                &baseline_url,
                                Some(&uuid::Uuid::new_v4().to_string()),
                                "cold-baseline",
                                &body,
                            )
                            .await;
                            assert_eq!(baseline.status, StatusCode::OK);
                            let output = stream_text(&response);
                            assert_eq!(output, stream_text(&baseline.text()));
                            let encoded = serde_json::to_vec(&output).unwrap();
                            output_hash.update((encoded.len() as u64).to_le_bytes());
                            output_hash.update(encoded);
                            histories[conversation]
                                .push(json!({"role": "assistant", "content": output.0}));
                            histories[conversation].push(json!({"role": "user", "content": format!("world {}", "c ".repeat(turn + 1))}));
                        }
                    }
                    durations.sort_by(f64::total_cmp);
                    ttfts.sort_by(f64::total_cmp);
                    queue_delays.sort_by(f64::total_cmp);
                    let metrics: Vec<_> = engines.iter().map(|engine| json!({"reuses": engine.kv_reuse_count(), "tokens_reused": engine.kv_tokens_reused(), "snapshot_bytes": engine.prefix_cache_stats().bytes, "warm_sessions": engine.warm_cache_stats().0, "warm_state_bytes": engine.warm_cache_stats().1})).collect();
                    rows.push(json!({"repeat": repeat, "prefix_words": prefix_words, "concurrency": concurrency, "policy": policy, "request_count": durations.len(), "ttft_ms_p50": ttfts[ttfts.len()/2], "ttft_ms_max": ttfts.last(), "queue_ms_p50": queue_delays[queue_delays.len()/2], "queue_ms_max": queue_delays.last(), "completion_latency_ms_p50": durations[durations.len()/2], "completion_latency_ms_max": durations.last(), "prompt_tokens": total_prompt, "workers": metrics, "requests_per_worker": requests_per_worker, "directory": coordinator.affinity_stats(), "outputs_checked": durations.len(), "request_sha256": format!("{:x}", request_hash.finalize()), "output_sha256": format!("{:x}", output_hash.finalize()), "model_sha256": model_sha256}));
                    for service in services {
                        service.abort();
                        let _ = service.await;
                    }
                }
            }
        }
    }
    for row in &rows {
        let peers: Vec<_> = rows
            .iter()
            .filter(|peer| {
                peer["repeat"] == row["repeat"]
                    && peer["prefix_words"] == row["prefix_words"]
                    && peer["concurrency"] == row["concurrency"]
            })
            .collect();
        for peer in peers {
            assert_eq!(row["request_sha256"], peer["request_sha256"]);
            assert_eq!(row["output_sha256"], peer["output_sha256"]);
        }
    }
    let mut source_files = serde_json::Map::new();
    for path in [
        "Cargo.lock",
        "src/affinity.rs",
        "src/coordinator.rs",
        "src/engine.rs",
        "src/lib.rs",
        "src/prefix_cache.rs",
        "src/server.rs",
        "src/types.rs",
        "tests/affinity_tests.rs",
        "tests/npu_tests.rs",
        "tests/route_tests.rs",
        "tests/common/mod.rs",
    ] {
        let bytes =
            std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(path)).unwrap();
        source_files.insert(path.into(), json!(format!("{:x}", Sha256::digest(bytes))));
    }
    let report = json!({"backend": "cpu", "host_arch": std::env::consts::ARCH, "host_os": std::env::consts::OS, "profile": "debug", "model": "synthetic one-layer Qwen3 F32 GGUF", "context": 256, "max_tokens": 1, "turns": 6, "workers": 2, "slots_per_worker": 1, "prefix_cache_bytes_per_worker": 1 << 20, "source_files_sha256": source_files, "rows": rows});
    let output = std::env::temp_dir().join("joshua-156-tiny-affinity.json");
    std::fs::write(&output, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    eprintln!("affinity measurements: {}", output.display());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn affinity_pins_spills_and_old_completions_cannot_restore_an_old_owner() {
    let (first, first_url) = spawn_fake("first", "alpha", 1, Behavior::Echo).await;
    let (second, second_url) = spawn_fake("second", "alpha", 1, Behavior::Echo).await;
    let (coordinator, router) = spawn_router(config(&[&first_url, &second_url])).await;
    let initial = affinity_chat(&router, "new", "alice").await;
    assert_eq!(initial.status, StatusCode::OK);
    let handle = initial.header("x-joshua-affinity").unwrap().to_string();
    let owner = initial.header(WORKER_HEADER).unwrap().to_string();
    for _ in 0..4 {
        assert_eq!(
            affinity_chat(&router, &handle, "alice")
                .await
                .header(WORKER_HEADER),
            Some(owner.as_str())
        );
    }
    let busy = if owner == first_url { &first } else { &second };
    *busy.behavior.lock().unwrap() = Behavior::Gate;
    let pending = tokio::spawn({
        let router = router.clone();
        let handle = handle.clone();
        async move { affinity_chat(&router, &handle, "alice").await }
    });
    eventually("affinity owner busy", || {
        busy.active.load(Ordering::SeqCst) == 1
    })
    .await;
    let spill = affinity_chat(&router, &handle, "alice").await;
    let next_owner = spill.header(WORKER_HEADER).unwrap().to_string();
    assert_ne!(next_owner, owner);
    busy.release.notify_one();
    assert_eq!(pending.await.unwrap().status, StatusCode::OK);
    *busy.behavior.lock().unwrap() = Behavior::Echo;
    assert_eq!(
        affinity_chat(&router, &handle, "alice")
            .await
            .header(WORKER_HEADER),
        Some(next_owner.as_str())
    );
    assert!(coordinator.affinity_stats().spillovers >= 1);
    assert_eq!(coordinator.affinity_stats().entries, 1);
    affinity_chat(&router, &handle, "bob").await;
    assert_eq!(coordinator.affinity_stats().entries, 2);
    let restarted = if next_owner == first_url {
        &first
    } else {
        &second
    };
    *restarted.epoch.lock().unwrap() = uuid::Uuid::new_v4().to_string();
    coordinator.refresh().await;
    assert_eq!(
        affinity_chat(&router, &handle, "alice").await.status,
        StatusCode::OK
    );
    assert_eq!(coordinator.affinity_stats().stale, 1);
    assert_eq!(
        affinity_chat(&router, "not-a-handle", "alice").await.status,
        StatusCode::BAD_REQUEST
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn affinity_auth_and_admission_failover_preserve_the_handle() {
    let (first, first_url) = spawn_fake("first", "alpha", 1, Behavior::Echo).await;
    let (second, second_url) = spawn_fake("second", "alpha", 1, Behavior::Echo).await;
    let mut cfg = config(&[&first_url, &second_url]);
    cfg.api_key = Some("alice".into());
    let (coordinator, router) = spawn_router(cfg).await;
    assert_eq!(
        affinity_chat(&router, "new", "bob").await.status,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(coordinator.affinity_stats().entries, 0);
    let initial = affinity_chat(&router, "new", "alice").await;
    let handle = initial.header("x-joshua-affinity").unwrap();
    let previous = initial.header(WORKER_HEADER).unwrap();
    let rejected = if previous == first_url {
        &first
    } else {
        &second
    };
    *rejected.behavior.lock().unwrap() = Behavior::Reject503;
    let retried = affinity_chat(&router, handle, "alice").await;
    assert_eq!(retried.status, StatusCode::OK);
    assert_eq!(retried.header("x-joshua-affinity"), Some(handle));
    assert_ne!(retried.header(WORKER_HEADER), Some(previous));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn affinity_residency_feedback_is_bounded_and_older_probes_do_not_erase_new_leases() {
    let (worker, url) = spawn_fake("first", "alpha", 1, Behavior::Echo).await;
    *worker.residency.lock().unwrap() = Some(joshua::affinity::CacheResidency {
        entries: Vec::new(),
        complete: true,
    });
    let (coordinator, router) = spawn_router(config(&[&url])).await;
    let initial = affinity_chat(&router, "new", "alice").await;
    let handle = initial.header("x-joshua-affinity").unwrap();
    assert_eq!(
        affinity_chat(&router, handle, "alice").await.status,
        StatusCode::OK
    );
    assert_eq!(coordinator.affinity_stats().residency_misses, 0);
    coordinator.refresh().await;
    assert_eq!(
        affinity_chat(&router, handle, "alice").await.status,
        StatusCode::OK
    );
    assert_eq!(coordinator.affinity_stats().residency_misses, 1);
    worker.residency.lock().unwrap().as_mut().unwrap().complete = false;
    coordinator.refresh().await;
    assert_eq!(
        affinity_chat(&router, handle, "alice").await.status,
        StatusCode::OK
    );
    assert_eq!(coordinator.affinity_stats().residency_misses, 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn affinity_worker_connection_loss_reconstructs_on_another_worker() {
    let (_, first_url, first_task) = spawn_fake_handle("first", "alpha", 1, Behavior::Echo).await;
    let (_, second_url, second_task) =
        spawn_fake_handle("second", "alpha", 1, Behavior::Echo).await;
    let (_, router) = spawn_router(config(&[&first_url, &second_url])).await;
    let initial = affinity_chat(&router, "new", "alice").await;
    let handle = initial.header("x-joshua-affinity").unwrap();
    let previous = initial.header(WORKER_HEADER).unwrap();
    if previous == first_url {
        first_task.abort();
    } else {
        second_task.abort();
    }
    tokio::task::yield_now().await;
    let next = affinity_chat(&router, handle, "alice").await;
    assert_eq!(next.status, StatusCode::OK);
    assert_ne!(next.header(WORKER_HEADER), Some(previous));
    assert_eq!(next.header("x-joshua-affinity"), Some(handle));
    first_task.abort();
    second_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_worker_eviction_feedback_invalidates_affinity_and_recomputes() {
    let directory = common::model_dir("affinity-real-eviction");
    common::write_tiny_qwen3_layers(&directory.join("model.gguf"), 1);
    let engine = Arc::new(
        joshua::Engine::with_options(
            &directory,
            joshua::EngineOptions::with_n_ctx(128).prefix_cache_bytes(0),
        )
        .unwrap(),
    );
    let worker = serve_app(joshua::server::create_router(Arc::new(
        joshua::server::ServerState {
            api_key: Some("affinity-worker-test-key".into()),
            ..joshua::server::ServerState::new(engine.clone())
        },
    )))
    .await;
    let mut settings = config(&[&worker]);
    settings.worker_api_key = Some("affinity-worker-test-key".into());
    let (coordinator, router) = spawn_router(settings).await;
    let mut body = json!({"model": engine.model_name(), "messages": [{"role": "user", "content": "hello a b c"}], "temperature": 0.0, "max_tokens": 0});
    let initial = affinity_request(&router, Some("new"), "alice", &body).await;
    assert_eq!(initial.status, StatusCode::OK);
    let handle = initial.header("x-joshua-affinity").unwrap();
    for _ in 0..8 {
        assert_eq!(
            affinity_request(&router, Some("new"), "alice", &body)
                .await
                .status,
            StatusCode::OK
        );
    }
    assert!(engine.warm_cache_stats().0 <= 4);
    assert_eq!(engine.prefix_cache_stats().bytes, 0);
    coordinator.refresh().await;
    body["messages"][0]["content"] = json!("hello a b c world");
    body["max_tokens"] = json!(4);
    let before = engine.kv_reuse_count();
    let recomputed = affinity_request(&router, Some(handle), "alice", &body).await;
    assert_eq!(recomputed.status, StatusCode::OK);
    assert_eq!(coordinator.affinity_stats().residency_misses, 1);
    assert_eq!(engine.kv_reuse_count(), before);
    let cold = affinity_request(&router, Some("new"), "alice", &body).await;
    assert_eq!(comparable(&recomputed.json()), comparable(&cold.json()));
}

#[tokio::test]
async fn affinity_handles_are_scoped_to_explicit_model_and_context() {
    let (_, alpha) = spawn_fake("alpha", "alpha", 1, Behavior::Echo).await;
    let (_, beta) = spawn_fake("beta", "beta", 1, Behavior::Echo).await;
    let (coordinator, router) = spawn_router(config(&[&alpha, &beta])).await;
    let mut body = json!({"model": "alpha", "messages": [{"role": "user", "content": "hello"}]});
    let initial = affinity_request(&router, Some("new"), "alice", &body).await;
    let handle = initial.header("x-joshua-affinity").unwrap();
    body["model"] = json!("beta");
    assert_eq!(
        affinity_request(&router, Some(handle), "alice", &body)
            .await
            .header(WORKER_HEADER),
        Some(beta.as_str())
    );
    body["context_window"] = json!(64);
    assert_eq!(
        affinity_request(&router, Some(handle), "alice", &body)
            .await
            .status,
        StatusCode::OK
    );
    assert_eq!(coordinator.affinity_stats().entries, 3);
}

/// Poll `cond` for up to five seconds.
async fn eventually(what: &str, cond: impl Fn() -> bool) {
    for _ in 0..500 {
        if cond() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("timed out waiting for: {what}");
}

fn sse_data(body: &str) -> Vec<String> {
    body.split("\n\n")
        .filter(|e| !e.trim().is_empty())
        .map(|e| e.strip_prefix("data: ").unwrap_or(e).to_string())
        .collect()
}

// ─── Routing ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn routes_by_model_and_forwards_the_request_unchanged() {
    let (alpha, alpha_url) = spawn_fake("a", "alpha", 4, Behavior::Echo).await;
    let (beta, beta_url) = spawn_fake("b", "beta", 4, Behavior::Echo).await;
    let (_, router) = spawn_router(config(&[&alpha_url, &beta_url])).await;

    let body = json!({
        "model": "beta",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
        "max_tokens": 7,
        "stop": ["\n"],
        "seed": 42,
    });
    let mut sender = open(&router).await;
    let mut req = request("POST", "/v1/chat/completions", &body.to_string());
    req.headers_mut()
        .insert(REQUEST_ID_HEADER, "client-id-1".parse().unwrap());
    let res = sender.send_request(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(res.headers()[WORKER_HEADER], beta_url.as_str());
    assert_eq!(res.headers()[REQUEST_ID_HEADER], "client-id-1");
    let out: Value =
        serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(out["worker"], "b");
    // Every parameter (including ones joshua ignores) reaches the worker,
    // byte for byte; usage comes back as the worker reported it.
    assert_eq!(
        beta.last_body.lock().unwrap().as_deref(),
        Some(body.to_string().as_bytes())
    );
    assert_eq!(
        beta.last_request_id.lock().unwrap().as_deref(),
        Some("client-id-1")
    );
    assert_eq!(out["usage"]["total_tokens"], 8);
    assert_eq!(alpha.received.load(Ordering::SeqCst), 0);

    // The legacy completion route is routed the same way.
    let res = call(
        &router,
        "POST",
        "/v1/completions",
        &json!({"model": "alpha", "prompt": "x"}).to_string(),
    )
    .await;
    assert_eq!(res.status, StatusCode::OK);
    assert_eq!(res.json()["worker"], "a");
    assert!(res.header(REQUEST_ID_HEADER).unwrap().starts_with("req-"));

    let duplicated = r#"{"model":"alpha","prompt":"old","prompt":"new","max_tokens":1,"max_tokens":2}"#;
    let res = call(&router, "POST", "/v1/completions", duplicated).await;
    assert_eq!(res.status, StatusCode::OK);
    assert_eq!(res.json()["echo"]["prompt"], "new");
    assert_eq!(alpha.last_body.lock().unwrap().as_deref(), Some(duplicated.as_bytes()));

    // A model nobody serves is a 404 naming what is available.
    let res = chat(&router, json!({"model": "gamma", "messages": []})).await;
    assert_eq!(res.status, StatusCode::NOT_FOUND);
    let err = res.json();
    assert_eq!(err["error"]["code"], "model_not_found");
    let msg = err["error"]["message"].as_str().unwrap();
    assert!(msg.contains("alpha") && msg.contains("beta"), "{msg}");

    // /v1/models lists the union of the workers' models.
    let models = call(&router, "GET", "/v1/models", "").await.json();
    let ids: Vec<&str> = models["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["alpha", "beta"]);

    // Malformed JSON is rejected at the router.
    let res = call(&router, "POST", "/v1/chat/completions", "{nope").await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST);
}

/// `POST /v1/audio/transcriptions` reaches only a worker that advertises
/// Whisper, and the multipart upload is forwarded untouched, boundary and
/// all (#157).
#[tokio::test]
async fn transcriptions_go_to_a_whisper_worker_with_the_upload_unchanged() {
    let (plain, plain_url) = spawn_fake("a", "m", 4, Behavior::Echo).await;
    let (whisper, whisper_url) = spawn_fake("w", "m", 4, Behavior::Echo).await;
    whisper.transcriptions.store(true, Ordering::SeqCst);
    let (_, router) = spawn_router(config(&[&plain_url, &whisper_url])).await;

    let boundary = "----joshua-boundary-1234";
    let content_type = format!("multipart/form-data; boundary={boundary}");
    // A WAV-ish upload with a JSON-looking field and a `model` field: none of
    // it may be parsed or rewritten by the router.
    let body = format!(
        "--{boundary}\r\n\
         Content-Disposition: form-data; name=\"model\"\r\n\r\n\
         whisper-1\r\n\
         --{boundary}\r\n\
         Content-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\
         Content-Type: audio/wav\r\n\r\n\
         RIFF\x00\x00\x00\x00WAVE\r\n\
         --{boundary}--\r\n"
    );
    let mut sender = open(&router).await;
    let mut req = request("POST", "/v1/audio/transcriptions", &body);
    req.headers_mut()
        .insert(header::CONTENT_TYPE, content_type.parse().unwrap());
    let res = sender.send_request(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(res.headers()[WORKER_HEADER], whisper_url.as_str());
    let out: Value =
        serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(out["worker"], "w");
    // The upload arrived byte for byte, with the client's boundary.
    assert_eq!(
        whisper.last_body.lock().unwrap().as_deref(),
        Some(body.as_bytes())
    );
    assert_eq!(
        whisper.last_content_type.lock().unwrap().as_deref(),
        Some(content_type.as_str())
    );
    assert_eq!(plain.received.load(Ordering::SeqCst), 0);

    // No worker advertises Whisper: a definite 503, not a bad request.
    let (_, other_router) = spawn_router(config(&[&plain_url])).await;
    let mut sender = open(&other_router).await;
    let mut req = request("POST", "/v1/audio/transcriptions", &body);
    req.headers_mut()
        .insert(header::CONTENT_TYPE, content_type.parse().unwrap());
    let res = sender.send_request(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    let out: Value =
        serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(out["error"]["code"], "no_worker");
    assert_eq!(plain.received.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn spreads_load_and_backpressures_when_every_slot_is_busy() {
    let (a, a_url) = spawn_fake("a", "m", 1, Behavior::Gate).await;
    let (b, b_url) = spawn_fake("b", "m", 1, Behavior::Gate).await;
    let mut cfg = config(&[&a_url, &b_url]);
    cfg.queue_depth = 0;
    let (coordinator, router) = spawn_router(cfg).await;

    let req = json!({"model": "m", "messages": []});
    let first = tokio::spawn({
        let (router, req) = (router.clone(), req.clone());
        async move { chat(&router, req).await }
    });
    let second = tokio::spawn({
        let (router, req) = (router.clone(), req.clone());
        async move { chat(&router, req).await }
    });
    eventually("both workers busy", || {
        a.active.load(Ordering::SeqCst) == 1 && b.active.load(Ordering::SeqCst) == 1
    })
    .await;

    // Both one-slot workers are busy and the queue holds nothing: 429.
    let res = chat(&router, req.clone()).await;
    assert_eq!(res.status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(res.header("retry-after"), Some("1"));
    assert_eq!(res.json()["error"]["code"], "queue_full");
    assert_eq!(
        a.received.load(Ordering::SeqCst) + b.received.load(Ordering::SeqCst),
        2
    );

    a.release.notify_waiters();
    b.release.notify_waiters();
    let (first, second) = (first.await.unwrap(), second.await.unwrap());
    assert_eq!(first.status, StatusCode::OK);
    assert_eq!(second.status, StatusCode::OK);
    let mut served = [
        first.json()["worker"].clone(),
        second.json()["worker"].clone(),
    ];
    served.sort_by_key(|v| v.to_string());
    assert_eq!(served, [json!("a"), json!("b")]);

    // Slots are released once the responses are done.
    eventually("slots released", || {
        coordinator
            .status_report()
            .iter()
            .all(|w| w.dispatched == 0)
    })
    .await;
}

#[tokio::test]
async fn queued_requests_wait_for_a_slot_or_time_out() {
    let (w, url) = spawn_fake("a", "m", 1, Behavior::Gate).await;
    let mut cfg = config(&[&url]);
    cfg.queue_depth = 1;
    cfg.queue_timeout = Duration::from_millis(300);
    let (_, router) = spawn_router(cfg).await;
    let req = json!({"messages": []});

    let holder = tokio::spawn({
        let (router, req) = (router.clone(), req.clone());
        async move { chat(&router, req).await }
    });
    eventually("worker busy", || w.active.load(Ordering::SeqCst) == 1).await;

    // Queued, but the slot stays taken past queue_timeout: 503 + Retry-After.
    let res = chat(&router, req.clone()).await;
    assert_eq!(res.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(res.header("retry-after"), Some("1"));
    assert_eq!(res.json()["error"]["code"], "queue_timeout");

    // A queued request proceeds as soon as the slot frees.
    let waiter = tokio::spawn({
        let (router, req) = (router.clone(), req.clone());
        async move { chat(&router, req).await }
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        w.received.load(Ordering::SeqCst),
        1,
        "waiter must be queued"
    );
    w.release.notify_waiters();
    assert_eq!(holder.await.unwrap().status, StatusCode::OK);
    eventually("queued request admitted", || {
        w.active.load(Ordering::SeqCst) == 1
    })
    .await;
    w.release.notify_waiters();
    assert_eq!(waiter.await.unwrap().status, StatusCode::OK);
    assert_eq!(w.received.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn unavailable_workers_answer_503() {
    let dead = dead_url().await;
    let (coordinator, router) = spawn_router(config(&[&dead])).await;
    let res = chat(&router, json!({"messages": []})).await;
    assert_eq!(res.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(res.json()["error"]["code"], "no_worker");
    assert_eq!(res.header("retry-after"), Some("1"));
    let health = call(&router, "GET", "/health", "").await;
    assert_eq!(health.status, StatusCode::SERVICE_UNAVAILABLE);
    let report = coordinator.status_report();
    assert!(!report[0].healthy);
    assert!(report[0].last_error.is_some());
}

// ─── Failure handling ────────────────────────────────────────────────────────

#[tokio::test]
async fn worker_lost_before_output_is_retried_elsewhere() {
    // A worker that passed its probe and then went away.
    let (gone, gone_url) = spawn_fake("gone", "m", 4, Behavior::LoseBeforeOutput).await;
    let (good, good_url) = spawn_fake("good", "m", 1, Behavior::Echo).await;
    // Equally idle workers are tried in list order, so the first attempt of
    // each request goes to the failing worker.
    let (coordinator, router) = spawn_router(config(&[&gone_url, &good_url])).await;

    for stream in [true, false] {
        let res = chat(&router, json!({"messages": [], "stream": stream})).await;
        assert_eq!(res.status, StatusCode::OK, "{}", res.text());
        assert_eq!(res.header(WORKER_HEADER), Some(good_url.as_str()));
        // Bring the lost worker back for the second round.
        coordinator.refresh().await;
    }
    assert!(
        gone.received.load(Ordering::SeqCst) >= 1,
        "first attempt hit the lost worker"
    );
    assert_eq!(good.received.load(Ordering::SeqCst), 2);

    // A worker that passed its probe and then stopped listening (process
    // gone): the refused connection is retried on the other worker.
    let (_, killed_url, server) = spawn_fake_handle("killed", "m", 4, Behavior::Echo).await;
    let (good2, good2_url) = spawn_fake("good2", "m", 4, Behavior::Echo).await;
    let (coordinator, router) = spawn_router(config(&[&killed_url, &good2_url])).await;
    assert!(coordinator.status_report()[0].healthy);
    server.abort();
    let _ = server.await;
    let res = chat(&router, json!({"messages": []})).await;
    assert_eq!(res.status, StatusCode::OK);
    assert_eq!(res.header(WORKER_HEADER), Some(good2_url.as_str()));
    assert_eq!(good2.received.load(Ordering::SeqCst), 1);
    let report = coordinator.status_report();
    assert!(!report[0].healthy);
    assert!(report[0].last_error.as_deref().unwrap().contains("connect"));
}

#[tokio::test]
async fn admission_rejection_retries_but_execution_errors_do_not() {
    let (busy, busy_url) = spawn_fake("busy", "m", 8, Behavior::Reject503).await;
    let (ok, ok_url) = spawn_fake("ok", "m", 1, Behavior::Echo).await;
    let (_, router) = spawn_router(config(&[&busy_url, &ok_url])).await;
    // Equally idle: `busy` (listed first) is tried first and rejects.
    let res = chat(&router, json!({"messages": []})).await;
    assert_eq!(res.status, StatusCode::OK);
    assert_eq!(res.json()["worker"], "ok");
    assert_eq!(busy.received.load(Ordering::SeqCst), 1);
    assert_eq!(ok.received.load(Ordering::SeqCst), 1);

    // Every worker refusing admission is a 503 with Retry-After.
    *ok.behavior.lock().unwrap() = Behavior::Reject503;
    let res = chat(&router, json!({"messages": []})).await;
    assert_eq!(res.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(res.header("retry-after"), Some("1"));
    assert_eq!(res.json()["error"]["code"], "workers_busy");

    // A worker that executed the request and failed is the answer: passed
    // through, not run again on another worker.
    let (failing, failing_url) = spawn_fake("failing", "m", 8, Behavior::Fail500).await;
    let (spare, spare_url) = spawn_fake("spare", "m", 1, Behavior::Echo).await;
    let (_, router) = spawn_router(config(&[&failing_url, &spare_url])).await;
    let res = chat(&router, json!({"messages": []})).await;
    assert_eq!(res.status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(res.json()["error"]["type"], "server_error");
    assert_eq!(failing.received.load(Ordering::SeqCst), 1);
    assert_eq!(spare.received.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn a_stream_lost_after_output_ends_with_an_error_and_is_not_restarted() {
    let (lossy, lossy_url) = spawn_fake("lossy", "m", 8, Behavior::LoseMidStream).await;
    let (spare, spare_url) = spawn_fake("spare", "m", 1, Behavior::Echo).await;
    let (coordinator, router) = spawn_router(config(&[&lossy_url, &spare_url])).await;

    let res = chat(&router, json!({"messages": [], "stream": true})).await;
    assert_eq!(res.status, StatusCode::OK);
    let events = sse_data(&res.text());
    assert_eq!(events.len(), 3, "{events:?}");
    assert_eq!(events[0], json!({"worker": "lossy", "i": 0}).to_string());
    assert_eq!(events[1], json!({"worker": "lossy", "i": 1}).to_string());
    let err: Value = serde_json::from_str(&events[2]).unwrap();
    assert_eq!(err["error"]["code"], "worker_lost");
    assert!(!res.text().contains("[DONE]"));
    assert_eq!(lossy.received.load(Ordering::SeqCst), 1);
    assert_eq!(
        spare.received.load(Ordering::SeqCst),
        0,
        "stream must not restart"
    );
    let report = coordinator.status_report();
    assert!(
        !report[0].healthy,
        "the lost worker is taken out of rotation"
    );
    assert_eq!(report[0].dispatched, 0);
}

// ─── Streaming and cancellation ──────────────────────────────────────────────

#[tokio::test]
async fn streams_are_forwarded_whole_and_in_order() {
    let (_, url) = spawn_fake("a", "m", 4, Behavior::Echo).await;
    let (_, router) = spawn_router(config(&[&url])).await;

    let mut sender = open(&router).await;
    let res = sender
        .send_request(request(
            "POST",
            "/v1/chat/completions",
            &json!({"messages": [], "stream": true}).to_string(),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(res.headers()[header::CONTENT_TYPE], "text/event-stream");
    let mut body = res.into_body();
    let mut all = Vec::new();
    while let Some(frame) = body.frame().await {
        let data = frame.unwrap().into_data().unwrap();
        // The worker splits every event in two; the router only forwards
        // whole events.
        assert!(data.ends_with(b"\n\n"), "partial event forwarded: {data:?}");
        all.extend_from_slice(&data);
    }
    let events = sse_data(&String::from_utf8(all).unwrap());
    let expected: Vec<String> = (0..20)
        .map(|i| json!({"worker": "a", "i": i}).to_string())
        .chain(std::iter::once("[DONE]".to_string()))
        .collect();
    assert_eq!(events, expected);
}

#[tokio::test]
async fn client_disconnect_cancels_the_worker_request() {
    let (w, url) = spawn_fake("a", "m", 1, Behavior::Hang).await;
    let (coordinator, router) = spawn_router(config(&[&url])).await;

    // Disconnect while the worker is still generating (no output yet).
    let client = tokio::spawn({
        let router = router.clone();
        async move { chat(&router, json!({"messages": []})).await }
    });
    eventually("worker running", || w.active.load(Ordering::SeqCst) == 1).await;
    client.abort();
    eventually("worker request dropped", || {
        w.dropped.load(Ordering::SeqCst) && w.active.load(Ordering::SeqCst) == 0
    })
    .await;
    eventually("router slot released", || {
        coordinator.status_report()[0].dispatched == 0
    })
    .await;

    // Disconnect after the stream has started.
    *w.behavior.lock().unwrap() = Behavior::StreamThenHang;
    let mut sender = open(&router).await;
    let res = sender
        .send_request(request(
            "POST",
            "/v1/chat/completions",
            &json!({"messages": [], "stream": true}).to_string(),
        ))
        .await
        .unwrap();
    let mut body = res.into_body();
    let first = body.frame().await.unwrap().unwrap().into_data().unwrap();
    assert_eq!(&first[..], b"data: {\"i\":0}\n\n");
    assert_eq!(w.active.load(Ordering::SeqCst), 1);
    drop(body);
    drop(sender);
    eventually("worker stream dropped", || {
        w.dropped.load(Ordering::SeqCst) && w.active.load(Ordering::SeqCst) == 0
    })
    .await;
    eventually("router slot released", || {
        coordinator.status_report()[0].dispatched == 0
    })
    .await;

    // The slot is reusable afterwards.
    *w.behavior.lock().unwrap() = Behavior::Echo;
    assert_eq!(
        chat(&router, json!({"messages": []})).await.status,
        StatusCode::OK
    );
}

#[tokio::test]
async fn the_router_api_key_is_enforced() {
    // (The worker-side key is covered by the real-worker parity test.)
    let (_, url) = spawn_fake("a", "m", 1, Behavior::Echo).await;
    let mut cfg = config(&[&url]);
    cfg.api_key = Some("client-key".to_string());
    let (_, router) = spawn_router(cfg).await;
    let res = chat(&router, json!({"messages": []})).await;
    assert_eq!(res.status, StatusCode::UNAUTHORIZED);
    let mut sender = open(&router).await;
    let mut req = request("POST", "/v1/chat/completions", "{\"messages\":[]}");
    req.headers_mut()
        .insert(header::AUTHORIZATION, "Bearer client-key".parse().unwrap());
    let res = sender.send_request(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    // /health stays open.
    assert_eq!(
        call(&router, "GET", "/health", "").await.status,
        StatusCode::OK
    );
}

// ─── Review fixes (#174) ─────────────────────────────────────────────────────

#[tokio::test]
async fn a_stream_flag_on_a_json_route_gets_json() {
    let (_, url) = spawn_fake("a", "m", 4, Behavior::Echo).await;
    let (_, router) = spawn_router(config(&[&url])).await;
    for (path, body) in [
        ("/v1/completions", json!({"prompt": "x", "stream": true})),
        ("/v1/embeddings", json!({"input": "x", "stream": true})),
    ] {
        let res = call(&router, "POST", path, &body.to_string()).await;
        assert_eq!(res.status, StatusCode::OK, "{path}");
        assert_eq!(
            res.header("content-type"),
            Some("application/json"),
            "{path}"
        );
        assert_eq!(res.json()["worker"], "a", "{path}");
    }
}

#[tokio::test]
async fn every_worker_failing_in_transit_is_a_502() {
    // One worker, lost before output: the attempt marks it unhealthy, but
    // the answer is still that the request failed in transit.
    let (lost, lost_url) = spawn_fake("lost", "m", 4, Behavior::LoseBeforeOutput).await;
    let (coordinator, router) = spawn_router(config(&[&lost_url])).await;
    for stream in [true, false] {
        let res = chat(&router, json!({"messages": [], "stream": stream})).await;
        assert_eq!(res.status, StatusCode::BAD_GATEWAY, "{}", res.text());
        assert_eq!(res.json()["error"]["code"], "workers_failed");
        assert!(!coordinator.status_report()[0].healthy);
        coordinator.refresh().await;
    }
    assert_eq!(lost.received.load(Ordering::SeqCst), 2);

    // Two workers: one stopped listening, one lost mid-request.
    let (_, killed_url, server) = spawn_fake_handle("killed", "m", 4, Behavior::Echo).await;
    let (_, lost_url) = spawn_fake("lost", "m", 4, Behavior::LoseBeforeOutput).await;
    let (coordinator, router) = spawn_router(config(&[&killed_url, &lost_url])).await;
    server.abort();
    let _ = server.await;
    let res = chat(&router, json!({"messages": []})).await;
    assert_eq!(res.status, StatusCode::BAD_GATEWAY);
    assert_eq!(res.json()["error"]["code"], "workers_failed");
    assert!(coordinator.status_report().iter().all(|w| !w.healthy));

    // With no attempt possible, it is still a 503.
    let res = chat(&router, json!({"messages": []})).await;
    assert_eq!(res.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(res.json()["error"]["code"], "no_worker");
}

#[tokio::test]
async fn a_silent_worker_times_out_and_frees_its_slot() {
    let (w, url) = spawn_fake("a", "m", 1, Behavior::Hang).await;
    let mut cfg = config(&[&url]);
    cfg.response_timeout = Some(Duration::from_millis(300));
    let (coordinator, router) = spawn_router(cfg).await;

    // No response at all, and stream headers with no event: both 504.
    for (behavior, stream) in [(Behavior::Hang, false), (Behavior::HeadersThenHang, true)] {
        *w.behavior.lock().unwrap() = behavior;
        let res = chat(&router, json!({"messages": [], "stream": stream})).await;
        assert_eq!(res.status, StatusCode::GATEWAY_TIMEOUT, "{}", res.text());
        assert_eq!(res.json()["error"]["code"], "worker_timeout");
        assert_eq!(res.header(WORKER_HEADER), Some(url.as_str()));
        // The worker connection is closed (cancelling the work), the slot
        // is free, and the worker leaves rotation until its next probe.
        eventually("worker request dropped", || {
            w.dropped.load(Ordering::SeqCst) && w.active.load(Ordering::SeqCst) == 0
        })
        .await;
        let report = coordinator.status_report();
        assert_eq!(report[0].dispatched, 0);
        assert!(!report[0].healthy);
        assert!(report[0].last_error.as_deref().unwrap().contains("300 ms"));
        coordinator.refresh().await;
        assert!(coordinator.status_report()[0].healthy);
    }

    // Once output flows, a pause longer than the deadline is not cut off.
    *w.behavior.lock().unwrap() = Behavior::SlowStream(Duration::from_millis(700));
    let res = chat(&router, json!({"messages": [], "stream": true})).await;
    assert_eq!(res.status, StatusCode::OK);
    let events = sse_data(&res.text());
    assert_eq!(events.len(), 3, "{events:?}");
    assert_eq!(events[2], "[DONE]");
    assert!(coordinator.status_report()[0].healthy);
}

#[tokio::test]
async fn a_stream_closed_without_done_ends_with_worker_lost() {
    let (_, url) = spawn_fake("a", "m", 4, Behavior::CloseWithoutDone).await;
    let (spare, spare_url) = spawn_fake("spare", "m", 1, Behavior::Echo).await;
    let (coordinator, router) = spawn_router(config(&[&url, &spare_url])).await;

    let res = chat(&router, json!({"messages": [], "stream": true})).await;
    assert_eq!(res.status, StatusCode::OK);
    let text = res.text();
    assert!(
        !text.contains("partial"),
        "unfinished event forwarded: {text}"
    );
    let events = sse_data(&text);
    assert_eq!(events.len(), 3, "{events:?}");
    assert_eq!(events[1], json!({"worker": "a", "i": 1}).to_string());
    let err: Value = serde_json::from_str(&events[2]).unwrap();
    assert_eq!(err["error"]["code"], "worker_lost");
    assert!(!text.contains("[DONE]"));
    assert_eq!(spare.received.load(Ordering::SeqCst), 0);
    // The truncating worker leaves rotation: the next request goes to the
    // spare instead of being cut short again.
    let report = coordinator.status_report();
    assert!(!report[0].healthy);
    assert!(report[0].last_error.as_deref().unwrap().contains("[DONE]"));
    let res = chat(&router, json!({"messages": [], "stream": true})).await;
    assert_eq!(res.header(WORKER_HEADER), Some(spare_url.as_str()));
    assert_eq!(
        sse_data(&res.text()).last().map(String::as_str),
        Some("[DONE]")
    );

    // Closed after only half an event: nothing reached the client, so the
    // request is retried, and the half event is never forwarded.
    let (_, partial_url) = spawn_fake("p", "m", 4, Behavior::PartialThenClose).await;
    let (spare, spare_url) = spawn_fake("spare", "m", 1, Behavior::Echo).await;
    let (_, router) = spawn_router(config(&[&partial_url, &spare_url])).await;
    let res = chat(&router, json!({"messages": [], "stream": true})).await;
    assert_eq!(res.status, StatusCode::OK);
    assert_eq!(res.header(WORKER_HEADER), Some(spare_url.as_str()));
    assert!(!res.text().contains("partial"));
    assert_eq!(
        sse_data(&res.text()).last().map(String::as_str),
        Some("[DONE]")
    );
    assert_eq!(spare.received.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn an_oversized_response_is_a_502_without_failover() {
    let (big, big_url) = spawn_fake("big", "m", 4, Behavior::Echo).await;
    let (spare, spare_url) = spawn_fake("spare", "m", 4, Behavior::Echo).await;
    let mut cfg = config(&[&big_url, &spare_url]);
    cfg.max_response_bytes = 16;
    let (coordinator, router) = spawn_router(cfg).await;
    let res = call(
        &router,
        "POST",
        "/v1/embeddings",
        &json!({"input": vec!["x"; 8]}).to_string(),
    )
    .await;
    assert_eq!(res.status, StatusCode::BAD_GATEWAY);
    assert_eq!(res.json()["error"]["code"], "response_too_large");
    assert_eq!(res.header(WORKER_HEADER), Some(big_url.as_str()));
    assert_eq!(big.received.load(Ordering::SeqCst), 1);
    assert_eq!(spare.received.load(Ordering::SeqCst), 0, "not retried");
    let report = coordinator.status_report();
    assert!(report.iter().all(|w| w.healthy && w.dispatched == 0));
}

#[tokio::test]
async fn cross_origin_access_needs_an_api_key() {
    let (_, url) = spawn_fake("a", "m", 1, Behavior::Echo).await;
    let cors_get = |router: String, key: Option<&'static str>| async move {
        let mut sender = open(&router).await;
        let mut req = request("GET", "/v1/models", "");
        req.headers_mut()
            .insert(header::ORIGIN, "https://elsewhere.example".parse().unwrap());
        if let Some(key) = key {
            req.headers_mut().insert(
                header::AUTHORIZATION,
                format!("Bearer {key}").parse().unwrap(),
            );
        }
        let res = sender.send_request(req).await.unwrap();
        (
            res.status(),
            res.headers()
                .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
                .cloned(),
        )
    };
    // No key: no CORS headers, so browsers keep other sites from reading.
    let (_, open_router) = spawn_router(config(&[&url])).await;
    let (status, allow) = cors_get(open_router, None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(allow, None);
    // With a key, cross-origin clients are allowed but must present it.
    let mut cfg = config(&[&url]);
    cfg.api_key = Some("k".to_string());
    let (_, keyed_router) = spawn_router(cfg).await;
    let (status, allow) = cors_get(keyed_router, Some("k")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(allow.is_some());
}

// ─── Real worker (tiny GGUF) ─────────────────────────────────────────────────

fn tiny_engine(dir_name: &str) -> joshua::Engine {
    let dir = common::model_dir(dir_name);
    common::write_tiny_llama_gguf(&dir.join("model.gguf"));
    joshua::Engine::with_n_ctx(&dir, 64).expect("engine should load tiny model")
}

async fn spawn_real_worker(dir_name: &str, api_key: Option<&str>) -> String {
    let state = Arc::new(joshua::server::ServerState {
        api_key: api_key.map(str::to_string),
        ..joshua::server::ServerState::new(Arc::new(tiny_engine(dir_name)))
    });
    serve_app(joshua::server::create_router(state)).await
}

/// The parts of a chat response that must match direct vs routed (ids and
/// timestamps differ per request).
fn comparable(v: &Value) -> Value {
    json!({"choices": v["choices"], "usage": v["usage"], "model": v["model"]})
}

fn stream_text(body: &str) -> (String, Vec<Value>) {
    let mut text = String::new();
    let mut tail = Vec::new();
    for data in sse_data(body) {
        if data == "[DONE]" {
            tail.push(json!("[DONE]"));
            continue;
        }
        let v: Value = serde_json::from_str(&data).unwrap();
        if let Some(c) = v["choices"][0]["delta"]["content"].as_str() {
            text.push_str(c);
        }
        if !v["choices"][0]["finish_reason"].is_null() {
            tail.push(json!({"finish": v["choices"][0]["finish_reason"], "usage": v["usage"]}));
        }
    }
    (text, tail)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn routed_greedy_output_matches_the_worker_directly() {
    let worker = spawn_real_worker("route-parity", Some("worker-key")).await;
    let mut cfg = config(&[&worker]);
    cfg.worker_api_key = Some("worker-key".to_string());
    let (coordinator, router) = spawn_router(cfg).await;

    // The worker protocol endpoint reports the loaded model.
    let report = coordinator.status_report();
    assert!(report[0].healthy, "{:?}", report[0].last_error);
    let info = report[0].info.clone().unwrap();
    assert_eq!(info.protocol_version, WORKER_PROTOCOL_VERSION);
    assert_eq!(info.n_ctx, 64);
    assert_eq!(info.backend, "cpu");
    assert_eq!(info.in_flight, 0);
    assert!(info.capabilities.chat && info.capabilities.streaming);
    assert!(!info.capabilities.transcriptions);

    let direct_call = |body: Value| {
        let worker = worker.clone();
        async move {
            let mut sender = open(&worker).await;
            let mut req = request("POST", "/v1/chat/completions", &body.to_string());
            req.headers_mut()
                .insert(header::AUTHORIZATION, "Bearer worker-key".parse().unwrap());
            let res = sender.send_request(req).await.unwrap();
            assert_eq!(res.status(), StatusCode::OK);
            String::from_utf8(res.into_body().collect().await.unwrap().to_bytes().to_vec()).unwrap()
        }
    };

    let body = json!({
        "model": info.model,
        "messages": [{"role": "user", "content": "hello there"}],
        "temperature": 0.0,
        "max_tokens": 8,
    });
    let direct: Value = serde_json::from_str(&direct_call(body.clone()).await).unwrap();
    let routed = chat(&router, body.clone()).await;
    assert_eq!(routed.status, StatusCode::OK);
    assert_eq!(comparable(&routed.json()), comparable(&direct));
    assert!(direct["usage"]["completion_tokens"].as_u64().unwrap() > 0);

    let mut stream_body = body.clone();
    stream_body["stream"] = json!(true);
    let direct = stream_text(&direct_call(stream_body.clone()).await);
    let routed = chat(&router, stream_body).await;
    assert_eq!(routed.status, StatusCode::OK);
    assert_eq!(stream_text(&routed.text()), direct);
    assert!(!direct.0.is_empty());
    assert_eq!(direct.1.last(), Some(&json!("[DONE]")));
}

/// A generation whose cancel flag is already set stops before decoding and
/// leaves the engine fully usable: the next request matches a fresh engine.
#[test]
fn cancelled_generation_stops_and_leaves_the_engine_reusable() {
    use joshua::types::GenerationOptions;
    use joshua::ChatMessage;

    let engine = tiny_engine("route-cancel");
    let messages = vec![ChatMessage::text("user", "hello there")];
    let greedy = GenerationOptions {
        max_tokens: 8,
        temperature: 0.0,
        ..GenerationOptions::default()
    };
    let cancelled = AtomicBool::new(true);
    let (text, usage, _, _) = engine
        .complete_chat_cancellable(&messages, None, &greedy, &cancelled)
        .unwrap();
    assert_eq!(usage.completion_tokens, 0);
    assert!(text.is_empty());
    assert_eq!(engine.in_flight(), 0, "the permit is released");

    let live = AtomicBool::new(false);
    let (after, usage, _, _) = engine
        .complete_chat_cancellable(&messages, None, &greedy, &live)
        .unwrap();
    assert!(usage.completion_tokens > 0);
    let fresh = tiny_engine("route-cancel-fresh");
    let (expected, _, _, _) = fresh.complete(&messages, &greedy).unwrap();
    assert_eq!(after, expected);
}

/// A worker serving two models, each admitting one request; `alpha`'s
/// requests wait for `release`.
async fn spawn_two_model_worker() -> (Arc<(AtomicUsize, Notify)>, String) {
    let shared = Arc::new((AtomicUsize::new(0), Notify::new()));
    let info = || async {
        Json(WorkerInfo {
            transcription_job: None,
            transcription_in_flight: 0,
            embedding_jobs: Vec::new(),
            protocol_version: WORKER_PROTOCOL_VERSION,
            cache_residency: Default::default(),
            cache_epochs: Default::default(),
            model: "alpha".to_string(),
            models: vec![
                ModelSlots {
                    id: "alpha".to_string(),
                    n_ctx: Some(4096),
                    max_concurrency: 1,
                    in_flight: 0,
                    embeddings: true,
                },
                ModelSlots {
                    id: "beta".to_string(),
                    n_ctx: Some(8192),
                    max_concurrency: 1,
                    in_flight: 0,
                    embeddings: false,
                },
            ],
            n_ctx: 4096,
            backend: "cpu".to_string(),
            max_concurrency: 2,
            in_flight: 0,
            capabilities: WorkerCapabilities {
                chat: true,
                completions: true,
                embeddings: true,
                transcriptions: false,
                streaming: true,
                tools: true,
            },
        })
    };
    let generate = |State(s): State<Arc<(AtomicUsize, Notify)>>, body: Bytes| async move {
        s.0.fetch_add(1, Ordering::SeqCst);
        let req: Value = serde_json::from_slice(&body).unwrap();
        if req["model"] == "alpha" {
            s.1.notified().await;
        }
        Json(json!({"model": req["model"]}))
    };
    let app = Router::new()
        .route("/v1/worker/info", get(info))
        .route("/v1/chat/completions", post(generate))
        .route("/v1/embeddings", post(generate))
        .with_state(Arc::clone(&shared));
    (shared, serve_app(app).await)
}

#[tokio::test]
async fn each_model_on_a_worker_is_admitted_by_its_own_slots() {
    let (worker, url) = spawn_two_model_worker().await;
    let mut cfg = config(&[&url]);
    cfg.queue_depth = 0;
    let (_coordinator, router) = spawn_router(cfg).await;

    let held = tokio::spawn({
        let router = router.clone();
        async move { chat(&router, json!({"model": "alpha", "messages": []})).await }
    });
    eventually("alpha running", || worker.0.load(Ordering::SeqCst) == 1).await;
    // alpha's one slot is taken, though the worker has a free beta slot.
    let res = chat(&router, json!({"model": "alpha", "messages": []})).await;
    assert_eq!(res.status, StatusCode::TOO_MANY_REQUESTS, "{}", res.text());
    // beta is admitted by its own slot.
    let res = chat(&router, json!({"model": "beta", "messages": []})).await;
    assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    // beta does not serve embeddings here, so none reach the worker.
    let res = call(
        &router,
        "POST",
        "/v1/embeddings",
        &json!({"model": "beta", "input": "x"}).to_string(),
    )
    .await;
    assert_ne!(res.status, StatusCode::OK, "{}", res.text());
    assert_eq!(worker.0.load(Ordering::SeqCst), 2);

    worker.1.notify_waiters();
    assert_eq!(held.await.unwrap().status, StatusCode::OK);
}

#[tokio::test]
async fn a_configured_worker_slot_cap_bounds_all_its_models_together() {
    let (worker, url) = spawn_two_model_worker().await;
    let mut cfg = config(&[&url]);
    cfg.queue_depth = 0;
    cfg.worker_slots = Some(1);
    let (_coordinator, router) = spawn_router(cfg).await;

    let held = tokio::spawn({
        let router = router.clone();
        async move { chat(&router, json!({"model": "alpha", "messages": []})).await }
    });
    eventually("alpha running", || worker.0.load(Ordering::SeqCst) == 1).await;
    // beta has a free slot of its own, but the worker's one slot is taken.
    let res = chat(&router, json!({"model": "beta", "messages": []})).await;
    assert_eq!(res.status, StatusCode::TOO_MANY_REQUESTS, "{}", res.text());

    worker.1.notify_waiters();
    assert_eq!(held.await.unwrap().status, StatusCode::OK);
}

#[tokio::test]
async fn context_fit_uses_each_models_window_and_refreshes_registry() {
    let (worker, url) = spawn_fake("registry", "alpha", 3, Behavior::Echo).await;
    worker.n_ctx.store(64, Ordering::SeqCst);
    *worker.models.lock().unwrap() = vec![
        ModelSlots {
            id: "alpha".into(),
            n_ctx: Some(64),
            max_concurrency: 1,
            in_flight: 0,
            embeddings: true,
        },
        ModelSlots {
            id: "beta".into(),
            n_ctx: Some(8192),
            max_concurrency: 2,
            in_flight: 0,
            embeddings: false,
        },
    ];
    let (coordinator, router) = spawn_router(config(&[&url])).await;
    for alias in ["context_window", "num_ctx", "n_ctx"] {
        let body = json!({"model": "beta", "messages": [], alias: 8192});
        let res = chat(&router, body.clone()).await;
        assert_eq!(res.status, StatusCode::OK, "{}", res.text());
        assert_eq!(
            worker.last_body.lock().unwrap().as_deref(),
            Some(body.to_string().as_bytes())
        );
    }
    let res = chat(&router, json!({"model": "alpha", "context_window": 65})).await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST);
    assert_eq!(res.json()["error"]["code"], "context_window_exceeded");
    assert!(res.header(WORKER_HEADER).is_none());
    assert!(res.header(REQUEST_ID_HEADER).is_some());
    assert_eq!(worker.received.load(Ordering::SeqCst), 3);

    worker.models.lock().unwrap()[1].n_ctx = Some(128);
    coordinator.refresh().await;
    let res = chat(&router, json!({"model": "beta", "context_window": 129})).await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST);
    worker.models.lock().unwrap().pop();
    coordinator.refresh().await;
    let res = chat(&router, json!({"model": "beta", "messages": []})).await;
    assert_eq!(res.status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn old_secondary_context_is_unknown_not_the_primary_context() {
    let (worker, url) = spawn_fake("old", "alpha", 2, Behavior::Echo).await;
    worker.n_ctx.store(64, Ordering::SeqCst);
    *worker.models.lock().unwrap() = vec![
        ModelSlots {
            id: "alpha".into(),
            n_ctx: None,
            max_concurrency: 1,
            in_flight: 0,
            embeddings: true,
        },
        ModelSlots {
            id: "beta".into(),
            n_ctx: None,
            max_concurrency: 1,
            in_flight: 0,
            embeddings: true,
        },
    ];
    let (coordinator, router) = spawn_router(config(&[&url])).await;
    let res = chat(&router, json!({"model": "beta", "context_window": 8192})).await;
    assert_eq!(res.status, StatusCode::OK);
    let res = chat(&router, json!({"model": "alpha", "context_window": 65})).await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST);
    let res = chat(&router, json!({"messages": []})).await;
    assert_eq!(res.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(worker.received.load(Ordering::SeqCst), 1);

    worker.models.lock().unwrap().clear();
    coordinator.refresh().await;
    let res = chat(&router, json!({"messages": [], "context_window": 64})).await;
    assert_eq!(res.status, StatusCode::OK);
}

#[tokio::test]
async fn fitting_workers_saturate_even_when_small_context_slots_are_free() {
    let (small, small_url) = spawn_fake("small", "m", 4, Behavior::Echo).await;
    small.n_ctx.store(64, Ordering::SeqCst);
    let (large, large_url) = spawn_fake("large", "m", 1, Behavior::Gate).await;
    let mut cfg = config(&[&small_url, &large_url]);
    cfg.queue_depth = 1;
    cfg.queue_timeout = Duration::from_millis(100);
    let (coordinator, router) = spawn_router(cfg).await;
    let held = tokio::spawn({
        let router = router.clone();
        async move { chat(&router, json!({"model": "m", "context_window": 4096})).await }
    });
    eventually("large context slot held", || {
        large.active.load(Ordering::SeqCst) == 1
    })
    .await;
    let queued = tokio::spawn({
        let router = router.clone();
        async move { chat(&router, json!({"model": "m", "context_window": 4096})).await }
    });
    let res = queued.await.unwrap();
    assert_eq!(res.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(res.json()["error"]["code"], "queue_timeout");
    assert_eq!(res.header("retry-after"), Some("1"));

    let mut cfg = config(&[&small_url, &large_url]);
    cfg.queue_depth = 0;
    let (_, saturated_router) = spawn_router(cfg).await;
    let res = chat(
        &saturated_router,
        json!({"model": "m", "context_window": 4096}),
    )
    .await;
    assert_eq!(res.status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(res.header("retry-after"), Some("1"));
    let res = chat(&router, json!({"model": "m", "context_window": 4097})).await;
    assert_eq!(res.status, StatusCode::BAD_REQUEST);
    assert_eq!(small.received.load(Ordering::SeqCst), 0);
    large.release.notify_waiters();
    assert_eq!(held.await.unwrap().status, StatusCode::OK);
    eventually("context slot released", || {
        coordinator
            .status_report()
            .iter()
            .all(|worker| worker.dispatched == 0)
    })
    .await;
    *large.behavior.lock().unwrap() = Behavior::Echo;
    assert_eq!(
        chat(&router, json!({"model": "m", "context_window": 4096}))
            .await
            .status,
        StatusCode::OK
    );
}

#[tokio::test]
async fn outstanding_work_breaks_equal_request_count_ties_and_is_released() {
    let (heavy, heavy_url) = spawn_fake("heavy", "m", 3, Behavior::Gate).await;
    let (light, light_url) = spawn_fake("light", "m", 3, Behavior::Gate).await;
    let (coordinator, router) = spawn_router(config(&[&heavy_url, &light_url])).await;
    let first = tokio::spawn({
        let router = router.clone();
        async move {
            chat(
                &router,
                json!({"model": "m", "messages": [{"content": "x".repeat(32768)}]}),
            )
            .await
        }
    });
    eventually("heavy work admitted", || {
        heavy.active.load(Ordering::SeqCst) == 1
    })
    .await;
    let second = tokio::spawn({
        let router = router.clone();
        async move { chat(&router, json!({"model": "m", "messages": []})).await }
    });
    eventually("light work admitted", || {
        light.active.load(Ordering::SeqCst) == 1
    })
    .await;
    *heavy.behavior.lock().unwrap() = Behavior::Echo;
    *light.behavior.lock().unwrap() = Behavior::Echo;
    let res = chat(&router, json!({"model": "m", "messages": []})).await;
    assert_eq!(res.header(WORKER_HEADER), Some(light_url.as_str()));
    heavy.release.notify_waiters();
    light.release.notify_waiters();
    assert_eq!(first.await.unwrap().status, StatusCode::OK);
    assert_eq!(second.await.unwrap().status, StatusCode::OK);
    eventually("all work released", || {
        coordinator
            .status_report()
            .iter()
            .all(|worker| worker.dispatched == 0)
    })
    .await;
    let res = chat(&router, json!({"model": "m", "messages": []})).await;
    assert_eq!(res.header(WORKER_HEADER), Some(light_url.as_str()));
    let res = chat(&router, json!({"model": "m", "messages": []})).await;
    assert_eq!(res.header(WORKER_HEADER), Some(heavy_url.as_str()));
}

#[tokio::test]
async fn context_hints_do_not_reject_unmeasured_prompts_or_ignored_route_fields() {
    let (worker, url) = spawn_fake("small", "m", 1, Behavior::Echo).await;
    worker.n_ctx.store(8, Ordering::SeqCst);
    let (_, router) = spawn_router(config(&[&url])).await;
    let body =
        json!({"model": "m", "messages": [{"content": "x".repeat(32768)}], "max_tokens": u32::MAX});
    assert_eq!(chat(&router, body).await.status, StatusCode::OK);
    for path in ["/v1/completions", "/v1/embeddings"] {
        let res = call(
            &router,
            "POST",
            path,
            r#"{"model":"m","prompt":"x","input":"x","context_window":99999,"num_ctx":88888}"#,
        )
        .await;
        assert_eq!(res.status, StatusCode::OK, "{}", res.text());
    }
}

#[tokio::test]
async fn heterogeneous_model_slots_are_independent_of_the_primary_window() {
    let (worker, url) = spawn_fake("multi", "alpha", 3, Behavior::Gate).await;
    worker.n_ctx.store(64, Ordering::SeqCst);
    *worker.models.lock().unwrap() = vec![
        ModelSlots {
            id: "alpha".into(),
            n_ctx: Some(64),
            max_concurrency: 1,
            in_flight: 0,
            embeddings: true,
        },
        ModelSlots {
            id: "beta".into(),
            n_ctx: Some(8192),
            max_concurrency: 2,
            in_flight: 0,
            embeddings: true,
        },
    ];
    let mut cfg = config(&[&url]);
    cfg.queue_depth = 0;
    let (coordinator, router) = spawn_router(cfg).await;
    let mut held = Vec::new();
    for active in 1..=2 {
        held.push(tokio::spawn({
            let router = router.clone();
            async move { chat(&router, json!({"model": "beta", "context_window": 8192})).await }
        }));
        eventually("beta slot held", || {
            worker.active.load(Ordering::SeqCst) == active
        })
        .await;
    }
    coordinator.refresh().await;
    let res = chat(&router, json!({"model": "beta", "context_window": 8192})).await;
    assert_eq!(res.status, StatusCode::TOO_MANY_REQUESTS);
    *worker.behavior.lock().unwrap() = Behavior::Echo;
    let res = chat(&router, json!({"model": "alpha", "context_window": 64})).await;
    assert_eq!(res.status, StatusCode::OK);
    worker.release.notify_waiters();
    for request in held {
        assert_eq!(request.await.unwrap().status, StatusCode::OK);
    }
}

#[tokio::test]
async fn context_unavailability_is_not_misreported_as_an_invalid_window() {
    let (small, small_url) = spawn_fake("small", "m", 1, Behavior::Echo).await;
    small.n_ctx.store(64, Ordering::SeqCst);
    let (_, large_url, handle) = spawn_fake_handle("large", "m", 1, Behavior::Echo).await;
    let (coordinator, router) = spawn_router(config(&[&small_url, &large_url])).await;
    handle.abort();
    let _ = handle.await;
    coordinator.refresh().await;
    let res = chat(&router, json!({"model": "m", "context_window": 4096})).await;
    assert_eq!(
        res.status,
        StatusCode::SERVICE_UNAVAILABLE,
        "{}",
        res.text()
    );
    assert_eq!(res.json()["error"]["code"], "no_worker");
    let dead = dead_url().await;
    let (_, router) = spawn_router(config(&[&small_url, &dead])).await;
    let res = chat(&router, json!({"model": "m", "context_window": 4096})).await;
    assert_eq!(res.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(small.received.load(Ordering::SeqCst), 0);
}

/// A batch embedding whose cancel flag is set stops before the next input,
/// releasing the engine's permit, and an uncancelled call is unchanged.
#[test]
fn cancelled_embedding_stops_and_releases_the_permit() {
    let engine = tiny_engine("route-embed-cancel");
    let texts = vec!["hello world".to_string(), "a b c".to_string()];
    let cancelled = AtomicBool::new(true);
    assert!(matches!(
        engine.embed_with_usage_cancellable(&texts, &cancelled),
        Err(joshua::JoshuaError::Cancelled)
    ));
    assert_eq!(engine.in_flight(), 0, "the permit is released");
    let live = AtomicBool::new(false);
    let (vectors, tokens) = engine.embed_with_usage_cancellable(&texts, &live).unwrap();
    assert_eq!(
        (vectors.clone(), tokens),
        engine.embed_with_usage(&texts).unwrap()
    );
    assert_eq!(vectors.len(), 2);
}

fn enable_embedding_jobs(worker: &Fake, revision: Option<&str>) {
    *worker.embedding_jobs.lock().unwrap() = vec![JobModel {
        model: worker.model.clone(), owner: format!("{}-owner", worker.name),
        backend: "cpu".to_string(), input_format: "text".to_string(), max_output_bytes: 2 << 20,
        revision: revision.map(str::to_string), max_inputs: 2, max_input_bytes: 256 << 10,
    }];
}

#[tokio::test]
async fn embedding_jobs_reconstruct_reordered_batches_and_preserve_options() {
    let (first, first_url) = spawn_fake("first", "m", 2, Behavior::Echo).await;
    let (second, second_url) = spawn_fake("second", "m", 2, Behavior::Echo).await;
    enable_embedding_jobs(&first, Some("same-weights"));
    enable_embedding_jobs(&second, Some("same-weights"));
    let (_, router) = spawn_router(config(&[&first_url, &second_url])).await;
    let texts = ["slow", "", "fast", "longer text", "last"];
    let response = call(&router, "POST", "/v1/embeddings", &json!({
        "model": "m", "input": texts, "encoding_format": "float", "normalize": false,
    }).to_string()).await;
    assert_eq!(response.status, StatusCode::OK, "{}", response.text());
    let result = response.json();
    for (index, text) in texts.iter().enumerate() {
        assert_eq!(result["data"][index]["index"], index);
        assert_eq!(result["data"][index]["embedding"][0], json!(text.len() as f32));
    }
    let tokens: usize = texts.iter().map(|text| text.len()).sum();
    assert_eq!(result["usage"]["prompt_tokens"], tokens);
    assert_eq!(result["usage"]["total_tokens"], tokens);
    assert_eq!(first.received.load(Ordering::SeqCst) + second.received.load(Ordering::SeqCst), 3);
    assert!(first.received.load(Ordering::SeqCst) > 0 && second.received.load(Ordering::SeqCst) > 0);
    for worker in [&first, &second] {
        let body: Value = serde_json::from_slice(worker.last_body.lock().unwrap().as_ref().unwrap()).unwrap();
        assert_eq!(body["normalize"], false);
        assert_eq!(body["encoding_format"], "float");
        assert!(worker.last_request_id.lock().unwrap().as_ref().unwrap().starts_with("job-"));
    }
    let mut completions: Vec<_> = first.completed.lock().unwrap().iter().cloned()
        .chain(second.completed.lock().unwrap().iter().cloned()).collect();
    assert_eq!(completions.len(), 3);
    completions.sort_by_key(|completion| completion.1);
    assert_ne!(completions[0].0, "slow");
    assert_eq!(completions[2].0, "slow");
}

#[tokio::test]
async fn embedding_jobs_never_mix_unknown_or_incompatible_revisions() {
    for (revision, backend) in [(None, "cpu"), (Some("different-weights"), "cpu"), (Some("weights"), "metal")] {
        let (first, first_url) = spawn_fake("first", "m", 1, Behavior::Echo).await;
        let (second, second_url) = spawn_fake("second", "m", 1, Behavior::Echo).await;
        enable_embedding_jobs(&first, Some("weights"));
        enable_embedding_jobs(&second, revision);
        second.embedding_jobs.lock().unwrap()[0].backend = backend.to_string();
        let (_, router) = spawn_router(config(&[&first_url, &second_url])).await;
        let response = call(&router, "POST", "/v1/embeddings", &json!({
            "model": "m", "input": ["one", "two", "three", "four", "five"],
        }).to_string()).await;
        assert_eq!(response.status, StatusCode::OK, "{}", response.text());
        let counts = [first.received.load(Ordering::SeqCst), second.received.load(Ordering::SeqCst)];
        assert!(counts == [3, 0] || counts == [0, 3], "{counts:?}");
    }
}

#[tokio::test]
async fn embedding_jobs_reject_duplicate_indexes_and_aggregate_overflow() {
    let (worker, url) = spawn_fake("worker", "m", 1, Behavior::Echo).await;
    enable_embedding_jobs(&worker, Some("weights"));
    worker.bad_indexes.store(true, Ordering::SeqCst);
    let (_, router) = spawn_router(config(&[&url])).await;
    let body = json!({"model": "m", "input": ["a", "b"]}).to_string();
    assert_eq!(call(&router, "POST", "/v1/embeddings", &body).await.status, StatusCode::BAD_GATEWAY);
    worker.bad_indexes.store(false, Ordering::SeqCst);
    let mut cfg = config(&[&url]);
    cfg.max_response_bytes = 300;
    let (_, router) = spawn_router(cfg).await;
    let body = json!({"model": "m", "input": ["a", "b", "c", "d"]}).to_string();
    assert_eq!(call(&router, "POST", "/v1/embeddings", &body).await.status, StatusCode::BAD_GATEWAY);
}

#[tokio::test]
async fn embedding_jobs_honor_worker_output_limit_and_skip_zero_budget_peers() {
    let (first, first_url) = spawn_fake("first", "m", 1, Behavior::Echo).await;
    let (second, second_url) = spawn_fake("second", "m", 1, Behavior::Echo).await;
    enable_embedding_jobs(&first, Some("weights"));
    enable_embedding_jobs(&second, Some("weights"));
    second.embedding_jobs.lock().unwrap()[0].max_inputs = 0;
    let (_, router) = spawn_router(config(&[&first_url, &second_url])).await;
    let body = json!({"model": "m", "input": ["a", "b", "c"]}).to_string();
    assert_eq!(call(&router, "POST", "/v1/embeddings", &body).await.status, StatusCode::OK);
    assert_eq!(second.received.load(Ordering::SeqCst), 0);
    first.embedding_jobs.lock().unwrap()[0].max_output_bytes = 1;
    let (_, router) = spawn_router(config(&[&first_url])).await;
    assert_eq!(call(&router, "POST", "/v1/embeddings", &body).await.status, StatusCode::BAD_GATEWAY);
}

#[tokio::test]
async fn embedding_jobs_bound_inputs_and_do_not_retry_execution_errors() {
    let (worker, url) = spawn_fake("worker", "m", 1, Behavior::Fail500).await;
    enable_embedding_jobs(&worker, Some("weights"));
    let (_, router) = spawn_router(config(&[&url])).await;
    let response = call(&router, "POST", "/v1/embeddings", &json!({
        "model": "m", "input": ["x".repeat(256 << 10)],
    }).to_string()).await;
    assert_eq!(response.status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(worker.received.load(Ordering::SeqCst), 0);
    let response = call(&router, "POST", "/v1/embeddings", &json!({
        "model": "m", "input": vec!["x"; 2049],
    }).to_string()).await;
    assert_eq!(response.status, StatusCode::PAYLOAD_TOO_LARGE);
    let response = call(&router, "POST", "/v1/embeddings", r#"{"model":"m","input":["x"]}"#).await;
    assert_eq!(response.status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(worker.received.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn embedding_jobs_cancel_all_dispatched_batches_on_disconnect() {
    let (worker, url) = spawn_fake("worker", "m", 4, Behavior::Hang).await;
    enable_embedding_jobs(&worker, Some("weights"));
    let (coordinator, router) = spawn_router(config(&[&url])).await;
    let client = tokio::spawn(async move {
        call(&router, "POST", "/v1/embeddings", &json!({"model": "m", "input": vec!["x"; 12]}).to_string()).await
    });
    eventually("four bounded subbatches running", || worker.active.load(Ordering::SeqCst) == 4).await;
    assert_eq!(worker.received.load(Ordering::SeqCst), 4);
    client.abort();
    eventually("all subbatches cancelled", || worker.active.load(Ordering::SeqCst) == 0).await;
    eventually("all slots released", || coordinator.status_report()[0].dispatched == 0).await;
    assert_eq!(worker.received.load(Ordering::SeqCst), 4);
}

#[tokio::test]
async fn embedding_jobs_match_direct_real_worker_and_enforce_owner() {
    let worker = spawn_real_worker("route-job-parity", Some("worker-secret")).await;
    let second = spawn_real_worker("route-job-parity-second", Some("worker-secret")).await;
    let info: WorkerInfo = {
        let mut sender = open(&worker).await;
        let mut req = request("GET", "/v1/worker/info", "");
        req.headers_mut().insert(header::AUTHORIZATION, "Bearer worker-secret".parse().unwrap());
        let response = sender.send_request(req).await.unwrap();
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap()
    };
    assert!(info.embedding_jobs[0].revision.is_some());
    let mut cfg = config(&[&worker, &second]);
    cfg.worker_api_key = Some("worker-secret".to_string());
    let (coordinator, router) = spawn_router(cfg).await;
    let reports = coordinator.status_report();
    assert_eq!(reports[0].info.as_ref().unwrap().embedding_jobs[0].revision,
        reports[1].info.as_ref().unwrap().embedding_jobs[0].revision);
    let body = json!({"model": info.model, "input": vec!["hello world"; 67], "encoding_format": "base64"}).to_string();
    let routed = call(&router, "POST", "/v1/embeddings", &body).await;
    let mut sender = open(&worker).await;
    let mut req = request("POST", "/v1/embeddings", &body);
    req.headers_mut().insert(header::AUTHORIZATION, "Bearer worker-secret".parse().unwrap());
    let response = sender.send_request(req).await.unwrap();
    let direct: Value = serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(routed.status, StatusCode::OK, "{}", routed.text());
    assert_eq!(routed.json(), direct);
    let mut req = request("POST", "/v1/embeddings", &body);
    req.headers_mut().insert(header::AUTHORIZATION, "Bearer worker-secret".parse().unwrap());
    req.headers_mut().insert(joshua::coordinator::JOB_OWNER_HEADER, "stale-owner".parse().unwrap());
    assert_eq!(sender.send_request(req).await.unwrap().status(), StatusCode::CONFLICT);
    let invalid = call(&router, "POST", "/v1/embeddings", &json!({"model": info.model, "input": [1]}).to_string()).await;
    assert_eq!(invalid.status, StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn embedding_jobs_retry_admission_but_never_reexecute_a_lost_result() {
    for lost in [false, true] {
        let (first, first_url) = spawn_fake("first", "m", 1,
            if lost { Behavior::LoseJsonBody } else { Behavior::Reject503 }).await;
        let (second, second_url) = spawn_fake("second", "m", 1, Behavior::Echo).await;
        enable_embedding_jobs(&first, Some("same"));
        enable_embedding_jobs(&second, Some("same"));
        let (_, router) = spawn_router(config(&[&first_url, &second_url])).await;
        let response = call(&router, "POST", "/v1/embeddings", r#"{"model":"m","input":["x"]}"#).await;
        assert_eq!(first.received.load(Ordering::SeqCst), 1);
        if lost {
            assert_eq!(response.status, StatusCode::BAD_GATEWAY);
            assert_eq!(second.received.load(Ordering::SeqCst), 0);
        } else {
            assert_eq!(response.status, StatusCode::OK, "{}", response.text());
            assert_eq!(response.json()["usage"]["prompt_tokens"], 1);
            assert_eq!(second.received.load(Ordering::SeqCst), 1);
            assert_eq!(*first.last_request_id.lock().unwrap(), *second.last_request_id.lock().unwrap());
        }
    }
}

#[tokio::test]
async fn embedding_jobs_fail_closed_after_resident_owner_changes() {
    let (worker, url) = spawn_fake("worker", "m", 1, Behavior::Echo).await;
    enable_embedding_jobs(&worker, Some("weights"));
    let (_, router) = spawn_router(config(&[&url])).await;
    worker.embedding_jobs.lock().unwrap()[0].owner = "replacement-model".to_string();
    let response = call(&router, "POST", "/v1/embeddings", r#"{"model":"m","input":["x"]}"#).await;
    assert_eq!(response.status, StatusCode::CONFLICT);
}

#[tokio::test]
async fn embedding_subbatches_do_not_overload_their_own_zero_depth_queue() {
    let (worker, url) = spawn_fake("worker", "m", 1, Behavior::Echo).await;
    enable_embedding_jobs(&worker, Some("weights"));
    let mut cfg = config(&[&url]);
    cfg.queue_depth = 0;
    let (_, router) = spawn_router(cfg).await;
    let response = call(&router, "POST", "/v1/embeddings", r#"{"model":"m","input":["one","two","three","four","five"]}"#).await;
    assert_eq!(response.status, StatusCode::OK, "{}", response.text());
    assert_eq!(worker.received.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn whole_audio_has_its_own_slot_and_does_not_block_embeddings() {
    let (worker, url) = spawn_fake("worker", "m", 1, Behavior::Hang).await;
    worker.transcriptions.store(true, Ordering::SeqCst);
    *worker.models.lock().unwrap() = vec![ModelSlots { id: "m".to_string(), n_ctx: Some(4096),
        max_concurrency: 1, in_flight: 0, embeddings: true }];
    enable_embedding_jobs(&worker, Some("weights"));
    let (coordinator, router) = spawn_router(config(&[&url])).await;
    let audio = tokio::spawn({
        let router = router.clone();
        async move { call(&router, "POST", "/v1/audio/transcriptions", "unmodified multipart").await }
    });
    eventually("audio admitted", || worker.active.load(Ordering::SeqCst) == 1).await;
    *worker.behavior.lock().unwrap() = Behavior::Echo;
    let embedding = tokio::time::timeout(Duration::from_secs(2),
        call(&router, "POST", "/v1/embeddings", r#"{"model":"m","input":["x"]}"#)).await.unwrap();
    assert_eq!(embedding.status, StatusCode::OK, "{}", embedding.text());
    assert_eq!(coordinator.status_report()[0].dispatched, 1);
    audio.abort();
    eventually("audio lease released", || coordinator.status_report()[0].dispatched == 0).await;
}

#[tokio::test]
async fn completed_audio_is_not_retained_as_external_occupancy() {
    let (worker, url) = spawn_fake("worker", "m", 1, Behavior::Gate).await;
    worker.transcriptions.store(true, Ordering::SeqCst);
    let mut settings = config(&[&url]);
    settings.queue_depth = 0;
    settings.worker_slots = Some(1);
    let (coordinator, router) = spawn_router(settings).await;
    let audio = tokio::spawn({
        let router = router.clone();
        async move { call(&router, "POST", "/v1/audio/transcriptions", "audio").await }
    });
    eventually("audio active", || worker.audio_active.load(Ordering::SeqCst) == 1).await;
    coordinator.refresh().await;
    assert_eq!(coordinator.status_report()[0].info.as_ref().unwrap().transcription_in_flight, 1);
    assert_eq!(coordinator.status_report()[0].external_in_flight, 1);
    worker.release.notify_one();
    assert_eq!(audio.await.unwrap().status, StatusCode::OK);
    eventually("audio released", || coordinator.status_report()[0].dispatched == 0).await;
    *worker.behavior.lock().unwrap() = Behavior::Echo;
    assert_eq!(call(&router, "POST", "/v1/audio/transcriptions", "next").await.status, StatusCode::OK);
    worker.external_audio.store(1, Ordering::SeqCst);
    coordinator.refresh().await;
    assert_eq!(coordinator.status_report()[0].external_in_flight, 1);
    let received = worker.received.load(Ordering::SeqCst);
    assert_eq!(call(&router, "POST", "/v1/audio/transcriptions", "blocked").await.status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(worker.received.load(Ordering::SeqCst), received);
    worker.external_audio.store(0, Ordering::SeqCst);
    coordinator.refresh().await;
    assert_eq!(call(&router, "POST", "/v1/audio/transcriptions", "available").await.status, StatusCode::OK);
}

#[tokio::test]
async fn sequential_audio_rotates_workers_and_honors_worker_cap() {
    let (first, first_url) = spawn_fake("first", "m", 1, Behavior::Echo).await;
    let (second, second_url) = spawn_fake("second", "m", 1, Behavior::Echo).await;
    first.transcriptions.store(true, Ordering::SeqCst);
    second.transcriptions.store(true, Ordering::SeqCst);
    let (_, router) = spawn_router(config(&[&first_url, &second_url])).await;
    for _ in 0..6 {
        assert_eq!(call(&router, "POST", "/v1/audio/transcriptions", "audio").await.status, StatusCode::OK);
    }
    assert_eq!(first.received.load(Ordering::SeqCst), 3);
    assert_eq!(second.received.load(Ordering::SeqCst), 3);
    *first.behavior.lock().unwrap() = Behavior::Gate;
    let mut settings = config(&[&first_url]);
    settings.worker_slots = Some(1);
    settings.queue_depth = 0;
    let (coordinator, router) = spawn_router(settings).await;
    let generation = tokio::spawn({
        let router = router.clone();
        async move { chat(&router, json!({"model":"m","messages":[]})).await }
    });
    eventually("generation admitted", || first.active.load(Ordering::SeqCst) == 1).await;
    assert_eq!(call(&router, "POST", "/v1/audio/transcriptions", "blocked").await.status, StatusCode::TOO_MANY_REQUESTS);
    generation.abort();
    eventually("worker lease released", || coordinator.status_report()[0].dispatched == 0).await;
}

#[tokio::test]
async fn completed_audio_wakes_queued_upload_without_periodic_probe() {
    let (worker, url) = spawn_fake("worker", "m", 1, Behavior::Gate).await;
    worker.transcriptions.store(true, Ordering::SeqCst);
    let (coordinator, router) = spawn_router(config(&[&url])).await;
    let first = tokio::spawn({
        let router = router.clone();
        async move { call(&router, "POST", "/v1/audio/transcriptions", "first").await }
    });
    eventually("first audio active", || worker.audio_active.load(Ordering::SeqCst) == 1).await;
    coordinator.refresh().await;
    let mut next = tokio::spawn({
        let router = router.clone();
        async move { call(&router, "POST", "/v1/audio/transcriptions", "next").await }
    });
    assert!(tokio::time::timeout(Duration::from_millis(30), &mut next).await.is_err());
    assert_eq!(worker.received.load(Ordering::SeqCst), 1);
    *worker.behavior.lock().unwrap() = Behavior::Echo;
    worker.release.notify_one();
    assert_eq!(first.await.unwrap().status, StatusCode::OK);
    let response = tokio::time::timeout(Duration::from_secs(2), next).await.unwrap().unwrap();
    assert_eq!(response.status, StatusCode::OK);
    assert_eq!(worker.received.load(Ordering::SeqCst), 2);
    assert_eq!(coordinator.status_report()[0].external_in_flight, 0);
}

#[tokio::test]
async fn embedding_limits_ignore_unhealthy_and_ineligible_cached_owners() {
    for unhealthy in [true, false] {
        let (healthy, healthy_url) = spawn_fake("healthy", "m", 2, Behavior::Echo).await;
        let (other, other_url, handle) = spawn_fake_handle("other", "m", 2, Behavior::Echo).await;
        enable_embedding_jobs(&healthy, Some("same"));
        enable_embedding_jobs(&other, Some("same"));
        other.embedding_jobs.lock().unwrap()[0].max_input_bytes = 1;
        other.embedding_jobs.lock().unwrap()[0].max_inputs = 1;
        if !unhealthy {
            *other.models.lock().unwrap() = vec![ModelSlots { id: "m".to_string(), n_ctx: Some(4096),
                max_concurrency: 2, in_flight: 0, embeddings: false }];
        }
        let (coordinator, router) = spawn_router(config(&[&healthy_url, &other_url])).await;
        if unhealthy {
            handle.abort();
            let _ = handle.await;
            coordinator.refresh().await;
            assert!(!coordinator.status_report()[1].healthy);
            assert!(coordinator.status_report()[1].info.is_some());
        }
        let response = call(&router, "POST", "/v1/embeddings", r#"{"model":"m","input":["one","two","three","four"]}"#).await;
        assert_eq!(response.status, StatusCode::OK, "{}", response.text());
        assert_eq!(healthy.received.load(Ordering::SeqCst), 2);
        assert_eq!(other.received.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test]
async fn embedding_job_identity_is_independent_of_reused_client_request_id() {
    let (worker, url) = spawn_fake("worker", "m", 1, Behavior::Echo).await;
    enable_embedding_jobs(&worker, Some("same"));
    let (_, router) = spawn_router(config(&[&url])).await;
    let mut job_ids = Vec::new();
    for _ in 0..2 {
        let mut sender = open(&router).await;
        let mut req = request("POST", "/v1/embeddings", r#"{"model":"m","input":["same"]}"#);
        req.headers_mut().insert(REQUEST_ID_HEADER, HeaderValue::from_static("reused-client-id"));
        let response = sender.send_request(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers().get(REQUEST_ID_HEADER).unwrap(), "reused-client-id");
        response.into_body().collect().await.unwrap();
        job_ids.push(worker.last_request_id.lock().unwrap().clone().unwrap());
    }
    assert_ne!(job_ids[0], job_ids[1]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn benchmark_embedding_jobs_loopback_acceptance_matrix() {
    for workers in [1, 2, 4] {
        let mut urls = Vec::new();
        for index in 0..workers {
            urls.push(spawn_real_worker(&format!("route-bench-{workers}-{index}"), None).await);
        }
        let info = call(&urls[0], "GET", "/v1/worker/info", "").await.json();
        let refs: Vec<_> = urls.iter().collect();
        let (_, router) = spawn_router(config(&refs)).await;
        for inputs in [1, 32, 128] {
            for words in [1, 8] {
                let body = json!({"model": info["model"], "input": vec!["hello ".repeat(words); inputs]}).to_string();
                for concurrency in [1, 4] {
                    for (mode, endpoint) in [("direct", &urls[0]), ("routed", &router)] {
                        let warm = call(endpoint, "POST", "/v1/embeddings", &body).await;
                        assert_eq!(warm.status, StatusCode::OK, "{}", warm.text());
                        let start = std::time::Instant::now();
                        let mut latencies: Vec<_> = futures_util::stream::iter(0..8).map(|_| {
                            let body = &body;
                            async move {
                                let start = std::time::Instant::now();
                                let result = call(endpoint, "POST", "/v1/embeddings", body).await;
                                assert_eq!(result.status, StatusCode::OK, "{}", result.text());
                                assert_eq!(result.json()["data"].as_array().unwrap().len(), inputs);
                                start.elapsed().as_secs_f64() * 1000.0
                            }
                        }).buffer_unordered(concurrency).collect().await;
                        let elapsed = start.elapsed().as_secs_f64();
                        latencies.sort_by(f64::total_cmp);
                        println!("{}", json!({"benchmark": "embedding-jobs-loopback", "backend": "cpu",
                            "fixture": "tiny-synthetic-llama", "workers": workers, "inputs": inputs,
                            "words": words, "concurrency": concurrency, "mode": mode,
                            "requests": 8, "documents_per_second": (8 * inputs) as f64 / elapsed,
                            "p50_ms": latencies[4], "p95_ms": latencies[7],
                            "client_upload_bytes": body.len() * 8}));
                    }
                }
            }
        }
    }
}

// ─── TLS (only with `cargo test --features tls`) ─────────────────────────────

#[cfg(feature = "tls")]
mod tls {
    use super::*;
    use joshua::rustls;

    /// Self-signed ECDSA P-256 certificate for `localhost` / `127.0.0.1`,
    /// valid for 100 years.  Test fixture only — the key is public.
    const CERT_PEM: &str = "-----BEGIN CERTIFICATE-----
MIIBpjCCAUugAwIBAgIUQtfwrB59vzm5SX8cDi9lvDDD3ZUwCgYIKoZIzj0EAwIw
FDESMBAGA1UEAwwJbG9jYWxob3N0MCAXDTI2MDcwNzA0MzcxMFoYDzIxMjYwNjEz
MDQzNzEwWjAUMRIwEAYDVQQDDAlsb2NhbGhvc3QwWTATBgcqhkjOPQIBBggqhkjO
PQMBBwNCAARZekIrwPUM1Wk7oq843hLDcXB7btwevAAgG105dwYP0yuZkqDAzaPJ
A2jHb6cV4BcMkMIMtyb4dWcn+I8bcnwIo3kwdzAdBgNVHQ4EFgQUU+jI5KG2ZY85
4lCBs3YuN/DTT/UwHwYDVR0jBBgwFoAUU+jI5KG2ZY854lCBs3YuN/DTT/UwGgYD
VR0RBBMwEYIJbG9jYWxob3N0hwR/AAABMAwGA1UdEwEB/wQCMAAwCwYDVR0PBAQD
AgeAMAoGCCqGSM49BAMCA0kAMEYCIQDIzu89kXXI3eb1SkPW2/qvW8H9r4BwGOp8
ecN8+G8rbQIhAKTKCRCf+CmzDkexWKQMy/bbFL2HI1rFluCMgWE0ZGiB
-----END CERTIFICATE-----
";
    const KEY_PEM: &str = "-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgPECih+wbHTjOIFNr
x1PL8bSQJI9KVrDmHx7F+RjRKsGhRANCAARZekIrwPUM1Wk7oq843hLDcXB7btwe
vAAgG105dwYP0yuZkqDAzaPJA2jHb6cV4BcMkMIMtyb4dWcn+I8bcnwI
-----END PRIVATE KEY-----
";

    fn pem_to_der(pem: &str) -> Vec<u8> {
        use base64::Engine as _;
        let b64: String = pem
            .lines()
            .filter(|line| !line.starts_with("-----"))
            .collect();
        base64::engine::general_purpose::STANDARD
            .decode(b64)
            .expect("test PEM should decode")
    }

    /// One request over TLS to `127.0.0.1:port`, trusting only `CERT_PEM`.
    fn tls_request(port: u16, request: &str) -> std::io::Result<String> {
        use std::io::{Read, Write};

        let mut roots = rustls::RootCertStore::empty();
        roots
            .add(rustls::pki_types::CertificateDer::from(pem_to_der(
                CERT_PEM,
            )))
            .unwrap();
        let config = Arc::new(
            rustls::ClientConfig::builder()
                .with_root_certificates(roots)
                .with_no_client_auth(),
        );
        let server_name = rustls::pki_types::ServerName::try_from("localhost").unwrap();
        let sock = std::net::TcpStream::connect(("127.0.0.1", port))?;
        sock.set_read_timeout(Some(Duration::from_secs(10)))?;
        let conn = rustls::ClientConnection::new(config, server_name)
            .map_err(std::io::Error::other)?;
        let mut tls = rustls::StreamOwned::new(conn, sock);
        tls.write_all(request.as_bytes())?;
        tls.flush()?;
        let mut buf = Vec::new();
        tls.read_to_end(&mut buf)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    /// `tls_request`, retried until the router's listener is up.
    async fn request_until_up(port: u16, request: String) -> String {
        for _ in 0..100 {
            let request = request.clone();
            let answer = tokio::task::spawn_blocking(move || tls_request(port, &request))
                .await
                .unwrap();
            if let Ok(answer) = answer {
                if !answer.is_empty() {
                    return answer;
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("router never answered over TLS");
    }

    /// `serve_tls` terminates TLS in-process: the same routes answer over
    /// HTTPS, and the port does not serve them in the clear.
    #[tokio::test]
    async fn https_router_round_trip() {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let (_fake, worker, _handle) = spawn_fake_handle("w", "tiny", 2, Behavior::Echo).await;

        let dir = common::model_dir("route-tls");
        let cert_path = dir.join("cert.pem");
        let key_path = dir.join("key.pem");
        std::fs::write(&cert_path, CERT_PEM).unwrap();
        std::fs::write(&key_path, KEY_PEM).unwrap();

        // Racy in theory (the probe listener is dropped before the router
        // rebinds) but reliable in a test process, as in server_tests.
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let addr = format!("127.0.0.1:{port}");
        let server = tokio::spawn({
            let addr = addr.clone();
            let cfg = config(&[&worker]);
            let cert_path = cert_path.clone();
            let key_path = key_path.clone();
            async move { joshua::coordinator::serve_tls(cfg, &addr, &cert_path, &key_path).await }
        });

        let answer = request_until_up(
            port,
            "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".to_string(),
        )
        .await;
        assert!(answer.starts_with("HTTP/1.1 200"), "{answer}");
        assert!(answer.contains("\"status\":\"ok\""), "{answer}");

        // The routed chat request crosses the TLS hop and comes back with the
        // worker's answer.
        let body = json!({"model": "tiny", "messages": [{"role": "user", "content": "hi"}]});
        let body = body.to_string();
        let answer = request_until_up(
            port,
            format!(
                "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\
                 Content-Type: application/json\r\nContent-Length: {}\r\n\
                 Connection: close\r\n\r\n{body}",
                body.len()
            ),
        )
        .await;
        assert!(answer.starts_with("HTTP/1.1 200"), "{answer}");
        assert!(answer.contains("tiny"), "worker answer missing: {answer}");

        // A plaintext request to the TLS port gets nothing usable.
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut plain = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        plain
            .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();
        let mut buf = [0u8; 32];
        let read = tokio::time::timeout(Duration::from_secs(2), plain.read(&mut buf)).await;
        if let Ok(Ok(n)) = read {
            assert!(
                !String::from_utf8_lossy(&buf[..n]).starts_with("HTTP/1.1 200"),
                "the TLS port served a plaintext request"
            );
        }

        server.abort();
    }
}
