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
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::StreamExt;
use http_body_util::{BodyExt, Full};
use hyper_util::rt::TokioIo;
use joshua::coordinator::{
    create_router, Coordinator, CoordinatorConfig, WorkerCapabilities, WorkerInfo,
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
    /// Start a 200 stream and lose the connection before any event.
    LoseBeforeOutput,
    /// Stream two events, then lose the connection.
    LoseMidStream,
}

struct Fake {
    name: String,
    model: String,
    max_concurrency: usize,
    behavior: Mutex<Behavior>,
    /// Requests received on the routed endpoints.
    received: AtomicUsize,
    active: AtomicUsize,
    release: Notify,
    /// Set when a hanging request/stream is dropped by the server.
    dropped: AtomicBool,
    last_body: Mutex<Option<Bytes>>,
    last_request_id: Mutex<Option<String>>,
}

struct ActiveGuard(Arc<Fake>);

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
        protocol_version: WORKER_PROTOCOL_VERSION,
        model: f.model.clone(),
        models: Vec::new(),
        n_ctx: 4096,
        backend: "cpu".to_string(),
        max_concurrency: f.max_concurrency,
        in_flight: f.active.load(Ordering::SeqCst),
        capabilities: WorkerCapabilities {
            chat: true,
            completions: true,
            embeddings: true,
            transcriptions: false,
            streaming: true,
            tools: true,
        },
    })
}

async fn fake_generate(State(f): State<Arc<Fake>>, headers: HeaderMap, body: Bytes) -> Response {
    f.received.fetch_add(1, Ordering::SeqCst);
    *f.last_body.lock().unwrap() = Some(body.clone());
    *f.last_request_id.lock().unwrap() = headers
        .get(REQUEST_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let req: Value = serde_json::from_slice(&body).unwrap();
    let stream = req["stream"].as_bool().unwrap_or(false);
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
        name: name.to_string(),
        model: model.to_string(),
        max_concurrency,
        behavior: Mutex::new(behavior),
        received: AtomicUsize::new(0),
        active: AtomicUsize::new(0),
        release: Notify::new(),
        dropped: AtomicBool::new(false),
        last_body: Mutex::new(None),
        last_request_id: Mutex::new(None),
    });
    let app = Router::new()
        .route("/v1/worker/info", get(fake_info))
        .route("/v1/chat/completions", post(fake_generate))
        .route("/v1/completions", post(fake_generate))
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
