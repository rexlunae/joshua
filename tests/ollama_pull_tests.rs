//! Ollama's model fetching: `/api/pull` from a registry and from a URL,
//! `/api/blobs` and `/api/create` with `files`.
mod common;

use std::sync::Arc;

use axum::{
    body::Body,
    extract::Path as UrlPath,
    http::{header, Request, StatusCode},
    response::IntoResponse,
    routing::get,
    Router,
};
use joshua::{
    server::{create_router, ModelManager, ModelRegistry, ServerState},
    EngineOptions,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tower::ServiceExt;

fn manager(dir: &std::path::Path, allow_private: bool) -> ModelManager {
    ModelManager {
        model_dir: dir.to_path_buf(),
        engine_options: Arc::new(|_: &std::path::Path| EngineOptions::with_n_ctx(64)),
        npu_backend: None,
        max_concurrency: None,
        max_output_tokens: None,
        allow_private_pull: allow_private,
        #[cfg(feature = "distributed")]
        workers: Vec::new(),
        #[cfg(feature = "distributed")]
        cluster_key: None,
    }
}

/// The server under test, allowed to pull from the local mock registry.
fn app(dir: &std::path::Path) -> Router {
    server(dir, true)
}

fn server(dir: &std::path::Path, allow_private: bool) -> Router {
    create_router(Arc::new(ServerState {
        models: ModelRegistry::default(),
        manager: Some(manager(dir, allow_private)),
        whisper: None,
        api_key: None,
        cors_origins: Vec::new(),
        reasoning_format: Default::default(),
        metrics: Arc::new(joshua::server::ServerMetrics::default()),
        responses: Default::default(),
        allow_runtime_lora: false,
    }))
}

/// Send `body` (JSON, or raw bytes) and read the answer: the status and the
/// body's lines, each parsed as JSON.
async fn send(app: &Router, method: &str, uri: &str, body: Body) -> (StatusCode, Vec<Value>) {
    let req = Request::builder()
        .method(method)
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(body)
        .unwrap();
    let res = app.clone().oneshot(req).await.unwrap();
    let status = res.status();
    let bytes = axum::body::to_bytes(res.into_body(), 1 << 20).await.unwrap();
    let lines = std::str::from_utf8(&bytes)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    (status, lines)
}

async fn post(app: &Router, uri: &str, body: Value) -> (StatusCode, Vec<Value>) {
    send(app, "POST", uri, Body::from(body.to_string())).await
}

fn sha256(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

/// A registry on a local port serving `library/tiny:latest` (whose model
/// layer is `gguf`, under `digest`), a manifest whose layer's digest is
/// wrong (`library/broken`), and the GGUF as a plain file at `/files/`.
async fn registry(gguf: Vec<u8>) -> String {
    let digest = sha256(&gguf);
    let manifest = |digest: &str, size: usize| {
        json!({
            "schemaVersion": 2,
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "config": {"mediaType": "application/vnd.docker.container.image.v1+json",
                       "digest": format!("sha256:{}", "0".repeat(64)), "size": 2},
            "layers": [
                {"mediaType": "application/vnd.ollama.image.model", "digest": digest, "size": size},
                {"mediaType": "application/vnd.ollama.image.template",
                 "digest": format!("sha256:{}", "1".repeat(64)), "size": 10},
                {"mediaType": "application/vnd.ollama.image.projector",
                 "digest": format!("sha256:{}", "2".repeat(64)), "size": 10},
            ],
        })
    };
    let good = manifest(&digest, gguf.len());
    let broken = manifest(&format!("sha256:{}", "f".repeat(64)), gguf.len());
    let blob = gguf.clone();
    let file = gguf;
    let router = Router::new()
        .route(
            "/v2/library/tiny/manifests/{tag}",
            get(move || async move { axum::Json(good) }),
        )
        .route(
            "/v2/library/broken/manifests/{tag}",
            get(move || async move { axum::Json(broken) }),
        )
        .route(
            "/v2/library/{model}/blobs/{digest}",
            get(move |UrlPath((_, _)): UrlPath<(String, String)>| async move { blob.clone() }),
        )
        .route(
            "/files/{name}",
            get(move || async move { file.clone().into_response() }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("127.0.0.1:{}", addr.port())
}

fn tiny_gguf(dir: &std::path::Path) -> Vec<u8> {
    let path = dir.join("source.gguf.bin");
    common::write_tiny_llama_gguf(&path);
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    bytes
}

#[tokio::test(flavor = "multi_thread")]
async fn pull_fetches_a_registry_model_and_checks_its_digest() {
    let dir = common::model_dir("ollama-pull-registry");
    let gguf = tiny_gguf(&dir);
    let host = registry(gguf.clone()).await;
    let app = app(&dir);

    // Streamed: the manifest, the layer's progress, then success.
    let (status, lines) = post(
        &app,
        "/api/pull",
        json!({"model": format!("{host}/library/tiny:latest"), "insecure": true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(lines.first().unwrap()["status"], "pulling manifest", "{lines:?}");
    assert_eq!(lines.last().unwrap()["status"], "success", "{lines:?}");
    let layer = lines
        .iter()
        .rev()
        .find(|l| l["digest"] == sha256(&gguf))
        .unwrap_or_else(|| panic!("no progress for the model layer: {lines:?}"));
    assert_eq!(layer["completed"], gguf.len(), "{layer}");
    assert_eq!(layer["total"], gguf.len(), "{layer}");
    assert!(
        lines.iter().any(|l| l["status"].as_str().unwrap_or_default().contains("projector")),
        "{lines:?}"
    );
    assert_eq!(std::fs::read(dir.join("tiny.gguf")).unwrap(), gguf);

    // The pulled model is listed and answers.
    let (_, tags) = send(&app, "GET", "/api/tags", Body::empty()).await;
    let names: Vec<&str> = tags[0]["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"tiny:latest"), "{names:?}");

    // Pulled again, the stored blob is reused; not streamed, one answer.
    std::fs::remove_file(dir.join("tiny.gguf")).unwrap();
    let (status, lines) = post(
        &app,
        "/api/pull",
        json!({"model": format!("{host}/library/tiny"), "insecure": true, "stream": false}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{lines:?}");
    assert_eq!(lines, [json!({"status": "success"})]);
    assert!(dir.join("tiny.gguf").is_file());

    // A layer whose bytes are not its digest is refused and not placed.
    let (status, lines) = post(
        &app,
        "/api/pull",
        json!({"model": format!("{host}/library/broken"), "insecure": true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let error = lines.last().unwrap()["error"].as_str().unwrap_or_default();
    assert!(error.contains("digest"), "{lines:?}");
    assert!(!dir.join("broken.gguf").exists());
    let (status, lines) = post(
        &app,
        "/api/pull",
        json!({"model": format!("{host}/library/broken"), "insecure": true, "stream": false}),
    )
    .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{lines:?}");
    assert!(lines[0]["error"].as_str().unwrap().contains("digest"), "{lines:?}");

    // A missing model says so.
    let (_, lines) = post(
        &app,
        "/api/pull",
        json!({"model": format!("{host}/library/absent"), "insecure": true}),
    )
    .await;
    assert!(lines.last().unwrap()["error"].as_str().unwrap().contains("not found"), "{lines:?}");

    // Plain http needs `insecure`.
    let (status, lines) =
        post(&app, "/api/pull", json!({"model": format!("http://{host}/files/x.gguf")})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{lines:?}");
    assert!(lines[0]["error"].as_str().unwrap().contains("insecure"), "{lines:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn pull_downloads_a_gguf_url() {
    let dir = common::model_dir("ollama-pull-url");
    let gguf = tiny_gguf(&dir);
    let host = registry(gguf.clone()).await;
    let app = app(&dir);
    let (status, lines) = post(
        &app,
        "/api/pull",
        json!({"model": format!("http://{host}/files/fetched-q4.gguf"), "insecure": true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(lines.last().unwrap()["status"], "success", "{lines:?}");
    assert_eq!(std::fs::read(dir.join("fetched-q4.gguf")).unwrap(), gguf);
    // No download is left behind under a temporary name.
    let leftovers: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with('.') && name != ".blobs")
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}

#[tokio::test]
async fn blobs_upload_and_create_places_one() {
    let dir = common::model_dir("ollama-blobs");
    let gguf = tiny_gguf(&dir);
    let digest = sha256(&gguf);
    let app = app(&dir);

    let (status, _) = send(&app, "HEAD", &format!("/api/blobs/{digest}"), Body::empty()).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    // A body that is not the digest is refused.
    let (status, lines) =
        send(&app, "POST", &format!("/api/blobs/{digest}"), Body::from(b"GGUF nope".to_vec())).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{lines:?}");
    assert!(lines[0]["error"].as_str().unwrap().contains("digest mismatch"), "{lines:?}");
    let (status, _) = send(&app, "HEAD", &format!("/api/blobs/{digest}"), Body::empty()).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) =
        send(&app, "POST", &format!("/api/blobs/{digest}"), Body::from(gguf.clone())).await;
    assert_eq!(status, StatusCode::CREATED);
    let (status, _) = send(&app, "HEAD", &format!("/api/blobs/{digest}"), Body::empty()).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = send(&app, "HEAD", "/api/blobs/sha256:..", Body::empty()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    // An upload that breaks off leaves nothing behind either.
    let broken = futures_util::stream::iter([
        Ok::<_, std::io::Error>(axum::body::Bytes::from_static(b"GGUF")),
        Err(std::io::Error::other("connection reset")),
    ]);
    let other = format!("sha256:{}", "c".repeat(64));
    let (status, _) =
        send(&app, "POST", &format!("/api/blobs/{other}"), Body::from_stream(broken)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let mut stored: Vec<String> = std::fs::read_dir(dir.join(".blobs"))
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    stored.sort();
    assert_eq!(stored, [format!("sha256-{}", &digest["sha256:".len()..])]);

    // `files` naming the blob creates the model, streamed or not.
    let (status, lines) = post(
        &app,
        "/api/create",
        json!({"model": "made", "files": {"made.gguf": digest}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{lines:?}");
    assert_eq!(lines.last().unwrap()["status"], "success", "{lines:?}");
    assert_eq!(std::fs::read(dir.join("made.gguf")).unwrap(), gguf);
    let (status, lines) = post(
        &app,
        "/api/create",
        json!({"model": "again:latest", "files": {"x.gguf": digest}, "stream": false}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{lines:?}");
    assert_eq!(lines, [json!({"status": "success"})]);
    assert!(dir.join("again.gguf").is_file());

    // The `from` form answers with status lines too.
    let (status, lines) =
        post(&app, "/api/create", json!({"model": "copied", "from": "made", "stream": false})).await;
    assert_eq!(status, StatusCode::OK, "{lines:?}");
    assert_eq!(lines, [json!({"status": "success"})]);
    assert!(dir.join("copied.gguf").is_file());

    // What create cannot honour is refused.
    let missing = format!("sha256:{}", "e".repeat(64));
    for (body, says) in [
        (json!({"model": "m", "files": {"m.gguf": missing}}), "not found"),
        (json!({"model": "m", "files": {"m.gguf": digest}, "template": "{{ .Prompt }}"}), "template"),
        (json!({"model": "m", "files": {"m.gguf": digest}, "system": "be terse"}), "system"),
        (json!({"model": "m", "files": {"m.gguf": digest}, "parameters": {"temperature": 0}}), "parameters"),
        (json!({"model": "m", "files": {"m.gguf": digest}, "from": "made"}), "not both"),
        (json!({"model": "m", "files": {"a.gguf": digest, "b.gguf": digest}}), "exactly one"),
        (json!({"model": "m", "files": {"m.safetensors": digest}}), "not a .gguf"),
        (json!({"model": "../m", "files": {"m.gguf": digest}}), "invalid model name"),
    ] {
        let (status, lines) = post(&app, "/api/create", body.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}: {lines:?}");
        let error = lines[0]["error"].as_str().unwrap();
        assert!(error.contains(says), "{body}: {error}");
    }
    assert!(!dir.join("m.gguf").exists());

    // A blob that is not a GGUF is not made a model.
    let text = b"not a model".to_vec();
    let text_digest = sha256(&text);
    let (status, _) =
        send(&app, "POST", &format!("/api/blobs/{text_digest}"), Body::from(text)).await;
    assert_eq!(status, StatusCode::CREATED);
    let (status, lines) = post(
        &app,
        "/api/create",
        json!({"model": "m", "files": {"m.gguf": text_digest}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{lines:?}");
    assert!(lines[0]["error"].as_str().unwrap().contains("not a GGUF"), "{lines:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn pull_stays_off_private_addresses_unless_allowed() {
    let dir = common::model_dir("ollama-pull-private");
    let gguf = tiny_gguf(&dir);
    let host = registry(gguf).await;
    let app = server(&dir, false);
    for name in [format!("{host}/library/tiny"), format!("http://{host}/files/tiny.gguf")] {
        let (status, lines) = post(&app, "/api/pull", json!({"model": name, "insecure": true})).await;
        assert_eq!(status, StatusCode::OK);
        let error = lines.last().unwrap()["error"].as_str().unwrap_or_default();
        assert!(error.contains("--allow-private-pull"), "{name}: {lines:?}");
    }
    assert!(!dir.join("tiny.gguf").exists());
}
