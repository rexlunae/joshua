//! HTTP server tests: the API-key middleware and (when built with
//! `--features tls`) rustls HTTPS termination.
//!
//! These use tiny synthetic GGUF models (see `tests/common`), so no network
//! access or real model download is needed.

mod common;

use std::sync::Arc;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use joshua::server::{create_router, ServerState};
use joshua::Engine;
use tower::ServiceExt;

use common::{model_dir, write_tiny_llama_gguf};

/// Build a `ServerState` around a tiny synthetic llama model.
fn tiny_state(dir_name: &str, api_key: Option<&str>) -> Arc<ServerState> {
    tiny_state_with_cors(dir_name, api_key, Vec::new())
}

fn tiny_state_with_cors(
    dir_name: &str,
    api_key: Option<&str>,
    cors_origins: Vec<String>,
) -> Arc<ServerState> {
    let dir = model_dir(dir_name);
    write_tiny_llama_gguf(&dir.join("model.gguf"));
    let engine = Engine::with_n_ctx(&dir, 64).expect("engine should load tiny model");
    Arc::new(ServerState {
        api_key: api_key.map(str::to_string),
        cors_origins,
        ..ServerState::new(Arc::new(engine))
    })
}

fn get(uri: &str, bearer: Option<&str>) -> Request<Body> {
    let mut req = Request::get(uri);
    if let Some(key) = bearer {
        req = req.header(header::AUTHORIZATION, format!("Bearer {key}"));
    }
    req.body(Body::empty()).unwrap()
}

#[tokio::test]
async fn v1_routes_require_the_configured_api_key() {
    let app = create_router(tiny_state("server-auth", Some("sekret")));

    // Missing Authorization header → 401 with an OpenAI-style error body.
    let res = app
        .clone()
        .oneshot(get("/v1/models", None))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
    let body = axum::body::to_bytes(res.into_body(), 64 * 1024).await.unwrap();
    let body = String::from_utf8_lossy(&body).to_string();
    assert!(
        body.contains("invalid_request_error"),
        "unexpected 401 body: {body}"
    );

    // Wrong key → 401.
    let res = app
        .clone()
        .oneshot(get("/v1/models", Some("wrong")))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

    // Key sent without the Bearer scheme → 401.
    let res = app
        .clone()
        .oneshot(
            Request::get("/v1/models")
                .header(header::AUTHORIZATION, "sekret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

    // POST routes are covered by the same middleware.
    let res = app
        .clone()
        .oneshot(
            Request::post("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"messages":[]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

    // The liveness probe stays open.
    let res = app.clone().oneshot(get("/health", None)).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // The right key gets through.
    let res = app.oneshot(get("/v1/models", Some("sekret"))).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[tokio::test]
async fn a_key_list_with_no_non_empty_entries_rejects_every_request() {
    // A misconfigured key list must never silently disable authentication.
    let app = create_router(tiny_state("server-empty-auth", Some(" , ,")));

    let res = app
        .clone()
        .oneshot(get("/v1/models", None))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::INTERNAL_SERVER_ERROR);

    // Empty credentials must not match the empty entries either.
    let res = app
        .oneshot(get("/v1/models", Some("")))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::INTERNAL_SERVER_ERROR);
}

#[tokio::test]
async fn routes_stay_open_when_no_api_key_is_configured() {
    let app = create_router(tiny_state("server-noauth", None));

    let res = app
        .clone()
        .oneshot(get("/v1/models", None))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // A stray Authorization header is ignored rather than rejected.
    let res = app
        .oneshot(get("/v1/models", Some("anything")))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

/// POST a chat completion body; returns the status and the response body.
async fn chat(app: axum::Router, body: serde_json::Value) -> (StatusCode, String) {
    let req = Request::post("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    let status = res.status();
    let body = axum::body::to_bytes(res.into_body(), 64 * 1024)
        .await
        .unwrap();
    (status, String::from_utf8_lossy(&body).to_string())
}

/// Reasoning, sampling and context-window settings (#144): accepted values
/// generate, unsupported or out-of-range ones are a 400.
#[tokio::test]
async fn chat_request_settings_are_validated() {
    let app = create_router(tiny_state("server-settings", None));
    let base = |extra: serde_json::Value| {
        let mut body = serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "a b"}],
            "max_tokens": 2,
        });
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        body
    };

    let (status, body) = chat(
        app.clone(),
        base(serde_json::json!({
            "reasoning_effort": "low",
            "chat_template_kwargs": {"enable_thinking": false},
            "seed": 3,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.5,
            "max_completion_tokens": 2,
            "num_ctx": 32,
            "tool_choice": "none",
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    for (extra, needle) in [
        (serde_json::json!({"context_window": 65}), "context window"),
        (serde_json::json!({"n": 2}), "n = 1"),
        (serde_json::json!({"logprobs": true}), "logprobs"),
        (
            serde_json::json!({"presence_penalty": 9}),
            "presence_penalty",
        ),
        (
            serde_json::json!({"reasoning_effort": "high", "enable_thinking": false}),
            "disagree",
        ),
        (
            serde_json::json!({"chat_template_kwargs": {"messages": []}}),
            "messages",
        ),
    ] {
        let (status, body) = chat(app.clone(), base(extra.clone())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{extra}: {body}");
        assert!(body.contains(needle), "{extra}: {body}");
    }
}

fn post_json(uri: &str, key: Option<&str>, body: serde_json::Value) -> Request<Body> {
    let mut req = Request::post(uri).header(header::CONTENT_TYPE, "application/json");
    if let Some(key) = key {
        req = req.header(header::AUTHORIZATION, format!("Bearer {key}"));
    }
    req.body(Body::from(body.to_string())).unwrap()
}

async fn body_text(res: axum::response::Response) -> String {
    let body = axum::body::to_bytes(res.into_body(), 1024 * 1024)
        .await
        .unwrap();
    String::from_utf8_lossy(&body).to_string()
}

#[tokio::test]
async fn ops_endpoints_report_models_load_and_metrics() {
    let app = create_router(tiny_state("server-ops", Some("sekret")));

    // Liveness is deliberately open, but carries enough to identify a build.
    let res = app.clone().oneshot(get("/health", None)).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let health: serde_json::Value = serde_json::from_str(&body_text(res).await).unwrap();
    assert_eq!(health["status"], "ok");
    assert_eq!(health["models"], 1);
    assert!(health["version"].is_string());

    // Inventory endpoints require the key.
    let res = app.clone().oneshot(get("/status", None)).await.unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
    let res = app
        .clone()
        .oneshot(get("/status", Some("sekret")))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let status: serde_json::Value = serde_json::from_str(&body_text(res).await).unwrap();
    assert_eq!(status["models"][0]["n_ctx"], 64);
    assert!(status["metrics"]["http_requests_total"].as_u64().unwrap() >= 1);

    let res = app
        .clone()
        .oneshot(get("/metrics", Some("sekret")))
        .await
        .unwrap();
    let metrics: serde_json::Value = serde_json::from_str(&body_text(res).await).unwrap();
    assert!(metrics["uptime_seconds"].is_u64());
    assert!(metrics["http_requests_total"].is_u64());

    let res = app
        .clone()
        .oneshot(get("/metrics/prometheus", Some("sekret")))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let prometheus = body_text(res).await;
    assert!(prometheus.contains("joshua_uptime_seconds"));
    assert!(prometheus.contains("joshua_in_flight_requests"));

    let res = app
        .clone()
        .oneshot(get("/props", Some("sekret")))
        .await
        .unwrap();
    let props: serde_json::Value = serde_json::from_str(&body_text(res).await).unwrap();
    assert_eq!(props["n_ctx"], 64);
    assert_eq!(props["total_slots"].as_u64().unwrap() >= 1, true);

    let res = app
        .clone()
        .oneshot(get("/slots", Some("sekret")))
        .await
        .unwrap();
    let slots: serde_json::Value = serde_json::from_str(&body_text(res).await).unwrap();
    assert_eq!(slots.as_array().unwrap().len(), 1);

    let res = app
        .clone()
        .oneshot(post_json(
            "/tokenize",
            Some("sekret"),
            serde_json::json!({"prompt": "hello world"}),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let tokens: serde_json::Value = serde_json::from_str(&body_text(res).await).unwrap();
    assert_eq!(tokens["count"], tokens["tokens"].as_array().unwrap().len());
    assert!(tokens["count"].as_u64().unwrap() > 0);

    let res = app
        .oneshot(post_json(
            "/api/tokens",
            Some("sekret"),
            serde_json::json!({"content": "hello"}),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[tokio::test]
async fn x_api_key_and_rotation_keys_are_accepted() {
    let app = create_router(tiny_state("server-key-rotation", Some("one, two")));

    let res = app
        .clone()
        .oneshot(
            Request::get("/v1/models")
                .header("x-api-key", "two")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let res = app
        .clone()
        .oneshot(get("/v1/models", Some("one")))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let res = app
        .oneshot(
            Request::get("/v1/models")
                .header("x-api-key", "three")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn configured_cors_origins_are_exact() {
    let app = create_router(tiny_state_with_cors(
        "server-cors",
        None,
        vec!["https://app.example".to_string()],
    ));

    let res = app
        .clone()
        .oneshot(
            Request::get("/v1/models")
                .header(header::ORIGIN, "https://app.example")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(
        res.headers()
            .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
            .and_then(|v| v.to_str().ok()),
        Some("https://app.example")
    );

    let res = app
        .oneshot(
            Request::get("/v1/models")
                .header(header::ORIGIN, "https://elsewhere.example")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert!(res
        .headers()
        .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
        .is_none());
}

#[tokio::test]
async fn streaming_chat_reconstructs_the_non_streaming_text() {
    let app = create_router(tiny_state("server-stream", None));
    let request = serde_json::json!({
        "model": "model",
        "messages": [{"role": "user", "content": "a b"}],
        "max_tokens": 6,
        "temperature": 0.0,
    });

    let (status, body) = chat(app.clone(), request.clone()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let response: serde_json::Value = serde_json::from_str(&body).unwrap();
    let expected = response["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string();

    let mut streaming = request;
    streaming["stream"] = serde_json::json!(true);
    let (status, body) = chat(app, streaming).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let mut text = String::new();
    let mut content_events = 0;
    let mut saw_done = false;
    let mut finish_reason = None;
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            saw_done = true;
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(data).unwrap();
        if let Some(content) = value["choices"][0]["delta"]["content"].as_str() {
            text.push_str(content);
            content_events += 1;
        }
        if let Some(reason) = value["choices"][0]["finish_reason"].as_str() {
            finish_reason = Some(reason.to_string());
        }
    }
    assert!(saw_done);
    assert!(content_events >= 1, "expected at least one content event");
    assert_eq!(text, expected);
    assert!(finish_reason.is_some());
}

#[tokio::test]
async fn anthropic_messages_supports_text_and_streaming() {
    let app = create_router(tiny_state("server-anthropic", None));
    let request = serde_json::json!({
        "model": "model",
        "max_tokens": 6,
        "messages": [{"role": "user", "content": "a b"}],
    });

    let res = app
        .clone()
        .oneshot(post_json("/v1/messages", None, request.clone()))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let response: serde_json::Value = serde_json::from_str(&body_text(res).await).unwrap();
    assert_eq!(response["type"], "message");
    assert_eq!(response["role"], "assistant");
    assert!(response["content"].is_array());
    assert!(response["usage"]["input_tokens"].as_u64().unwrap() > 0);

    let mut streaming = request;
    streaming["stream"] = serde_json::json!(true);
    let res = app
        .oneshot(post_json("/v1/messages", None, streaming))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = body_text(res).await;
    assert!(body.contains("event: message_start"), "{body}");
    assert!(body.contains("event: content_block_delta"), "{body}");
    assert!(body.contains("event: message_stop"), "{body}");
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

    /// Full round trip: serve_with_state_tls terminates TLS in-process and a
    /// rustls client that trusts the self-signed cert can GET /health.
    #[tokio::test]
    async fn https_round_trip() {
        let _ = rustls::crypto::ring::default_provider().install_default();

        let state = tiny_state("server-tls", None);
        let dir = model_dir("server-tls");
        let cert_path = dir.join("cert.pem");
        let key_path = dir.join("key.pem");
        std::fs::write(&cert_path, CERT_PEM).unwrap();
        std::fs::write(&key_path, KEY_PEM).unwrap();

        // Grab an ephemeral port. Racy in theory (the listener is dropped
        // before the server rebinds) but reliable in a test process.
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let addr = format!("127.0.0.1:{port}");

        let server = tokio::spawn({
            let addr = addr.clone();
            async move {
                joshua::server::serve_with_state_tls(state, &addr, &cert_path, &key_path).await
            }
        });

        let response = tokio::task::spawn_blocking(move || {
            use std::io::{Read, Write};

            let mut roots = rustls::RootCertStore::empty();
            roots
                .add(rustls::pki_types::CertificateDer::from(pem_to_der(CERT_PEM)))
                .unwrap();
            let config = Arc::new(
                rustls::ClientConfig::builder()
                    .with_root_certificates(roots)
                    .with_no_client_auth(),
            );
            let server_name = rustls::pki_types::ServerName::try_from("localhost").unwrap();

            // The server task binds asynchronously — retry until it's up.
            let mut sock = None;
            for _ in 0..50 {
                match std::net::TcpStream::connect(&addr) {
                    Ok(s) => {
                        sock = Some(s);
                        break;
                    }
                    Err(_) => std::thread::sleep(std::time::Duration::from_millis(100)),
                }
            }
            let sock = sock.expect("server did not start listening");

            let conn = rustls::ClientConnection::new(config, server_name)
                .expect("client connection should build");
            let mut tls = rustls::StreamOwned::new(conn, sock);
            tls.write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
                .unwrap();
            let mut buf = Vec::new();
            if let Err(e) = tls.read_to_end(&mut buf) {
                // Tolerate a missing close_notify as long as we got the response.
                assert!(!buf.is_empty(), "TLS read failed with no data: {e}");
            }
            String::from_utf8_lossy(&buf).to_string()
        })
        .await
        .unwrap();

        assert!(
            response.starts_with("HTTP/1.1 200"),
            "unexpected response: {response}"
        );
        assert!(response.contains(r#""status":"ok""#), "unexpected body: {response}");

        server.abort();
    }
}
