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
    let dir = model_dir(dir_name);
    write_tiny_llama_gguf(&dir.join("model.gguf"));
    let engine = Engine::with_n_ctx(&dir, 64).expect("engine should load tiny model");
    Arc::new(ServerState {
        engine: Arc::new(engine),
        whisper: None,
        api_key: api_key.map(str::to_string),
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
