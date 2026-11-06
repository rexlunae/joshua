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
async fn internal_cache_scope_requires_authenticated_worker_access() {
    for configured in [None, Some("alice, bob")] {
        let state = tiny_state("server-internal-cache-scope", configured);
        let engine = state.models.list()[0].engine.clone();
        let app = create_router(state);
        let mut observed_scopes = std::collections::HashSet::new();
        for scope in ["coordinator-one", "coordinator-two"] {
            for (credential, expected) in [
                (None, StatusCode::UNAUTHORIZED),
                (Some("invalid"), StatusCode::UNAUTHORIZED),
                (Some("alice"), StatusCode::OK),
                (Some("bob"), StatusCode::OK),
            ] {
                for key_header in [header::AUTHORIZATION.as_str(), "x-api-key"] {
                    let mut request = Request::post("/v1/completions")
                        .header(header::CONTENT_TYPE, "application/json")
                        .header("x-joshua-cache-scope", scope);
                    if let Some(credential) = credential {
                        request = request.header(
                            key_header,
                            if key_header == "x-api-key" {
                                credential.to_string()
                            } else {
                                format!("Bearer {credential}")
                            },
                        );
                    }
                    let response = app
                        .clone()
                        .oneshot(
                            request
                                .body(Body::from(r#"{"prompt":"hello a b c","max_tokens":0}"#))
                                .unwrap(),
                        )
                        .await
                        .unwrap();
                    assert_eq!(
                        response.status(),
                        if configured.is_none() {
                            StatusCode::OK
                        } else {
                            expected
                        }
                    );
                    if configured.is_some() && expected == StatusCode::OK {
                        let report = engine.cache_residency(128);
                        assert!(report.complete);
                        observed_scopes.extend(report.entries.into_iter().map(|entry| entry.scope));
                    }
                }
            }
        }
        if configured.is_some() {
            assert_eq!(
                observed_scopes.len(),
                4,
                "keys and internal scopes isolate; auth header forms share"
            );
        }
    }
}

#[tokio::test]
async fn unauthenticated_internal_scope_is_ignored_without_breaking_public_hints() {
    let mut public_scopes = std::collections::HashSet::new();
    for public_hint in [None, Some("public-conversation")] {
        let state = tiny_state("server-ignore-internal-cache-scope", None);
        let engine = state.models.list()[0].engine.clone();
        let app = create_router(state);
        let mut observed_scopes = std::collections::HashSet::new();
        for (internal_scope, prompt) in [
            (None, "hello a b c"),
            (Some("spoofed-one"), "hello a b c world"),
            (Some("spoofed-two"), "hello a b c world d"),
        ] {
            let mut request =
                Request::post("/v1/completions").header(header::CONTENT_TYPE, "application/json");
            if let Some(hint) = public_hint {
                request = request.header("x-joshua-affinity", hint);
            }
            if let Some(scope) = internal_scope {
                request = request.header("x-joshua-cache-scope", scope);
            }
            let response = app
                .clone()
                .oneshot(
                    request
                        .body(Body::from(
                            serde_json::json!({"prompt": prompt, "max_tokens": 0}).to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let report = engine.cache_residency(128);
            assert!(report.complete);
            observed_scopes.extend(report.entries.into_iter().map(|entry| entry.scope));
        }
        assert_eq!(observed_scopes.len(), 1, "internal hints must be ignored");
        assert!(engine.kv_reuse_count() > 0);
        public_scopes.extend(observed_scopes);
    }
    assert_eq!(
        public_scopes.len(),
        2,
        "public affinity semantics are preserved"
    );
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

/// `/detokenize` is llama-server's path (`/v1/detokenize` is vLLM's), and
/// the two read the text from different keys.
#[tokio::test]
async fn detokenize_answers_on_both_paths_with_both_keys() {
    let app = create_router(tiny_state("server-detokenize", None));
    let res = app
        .clone()
        .oneshot(post_json("/tokenize", None, serde_json::json!({"prompt": "hello world"})))
        .await
        .unwrap();
    let tokens: serde_json::Value = serde_json::from_str(&body_text(res).await).unwrap();
    let ids = tokens["tokens"].clone();
    for path in ["/detokenize", "/v1/detokenize"] {
        let res = app
            .clone()
            .oneshot(post_json(path, None, serde_json::json!({"tokens": ids})))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK, "{path}");
        let body: serde_json::Value = serde_json::from_str(&body_text(res).await).unwrap();
        assert!(body["prompt"].is_string(), "{path}: {body}");
        assert_eq!(body["prompt"], body["content"], "{path}");
    }
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
        // Exercise content events deterministically instead of sampling EOS
        // before the first token and intermittently producing an empty stream.
        "temperature": 0.0,
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

// ─── Engine-core API semantics (#254) ────────────────────────────────────────

/// A reasoning model: the tiny llama with `<think>`/`</think>` tokens and a
/// template that opens the block in its generation prompt.
fn think_state(dir_name: &str) -> Arc<ServerState> {
    let dir = common::think_model_dir(dir_name);
    let engine = Engine::with_n_ctx(&dir, 64)
        .expect("engine should load tiny model")
        .with_chat_template(common::THINK_TEMPLATE);
    Arc::new(ServerState::new(Arc::new(engine)))
}

/// Greedy, fixed-length chat in which only the budget can close the
/// reasoning block (the model may not write the tags itself).
fn think_body(extra: serde_json::Value) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": "model",
        "messages": [{"role": "user", "content": "a b"}],
        "max_tokens": 5,
        "temperature": 0.0,
        "ignore_eos": true,
        "logit_bias": {"14": false, "15": false},
        "thinking_token_budget": 2,
    });
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    body
}

/// The `data:` payloads of an SSE body, without `[DONE]`.
fn sse_chunks(body: &str) -> Vec<serde_json::Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect()
}

#[tokio::test]
async fn chat_splits_reasoning_and_reports_usage_details() {
    let app = create_router(think_state("server-think"));
    let (status, body) = chat(app.clone(), think_body(serde_json::json!({}))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let response: serde_json::Value = serde_json::from_str(&body).unwrap();
    let message = &response["choices"][0]["message"];
    let reasoning = message["reasoning_content"].as_str().unwrap().to_string();
    let content = message["content"].as_str().unwrap().to_string();
    assert!(!reasoning.is_empty() && !content.is_empty(), "{body}");
    assert_eq!(message["reasoning"], message["reasoning_content"]);
    assert!(!content.contains("</think>"));
    assert_eq!(response["choices"][0]["finish_reason"], "length");
    let usage = &response["usage"];
    assert_eq!(usage["completion_tokens"], 5);
    assert_eq!(usage["completion_tokens_details"]["reasoning_tokens"], 2);
    assert!(usage["prompt_tokens_details"]["cached_tokens"].is_u64());
    assert_eq!(response["timings"]["predicted_n"], 5);
    assert_eq!(
        response["timings"]["prompt_n"].as_u64().unwrap()
            + response["timings"]["cache_n"].as_u64().unwrap(),
        usage["prompt_tokens"].as_u64().unwrap()
    );

    // Streaming carries the same split in deltas, then usage and timings.
    let (status, body) = chat(app.clone(), think_body(serde_json::json!({"stream": true}))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let chunks = sse_chunks(&body);
    let (mut streamed_reasoning, mut streamed_content) = (String::new(), String::new());
    for chunk in &chunks {
        let delta = &chunk["choices"][0]["delta"];
        if let Some(r) = delta["reasoning_content"].as_str() {
            assert_eq!(delta["reasoning"].as_str(), Some(r));
            streamed_reasoning.push_str(r);
        }
        if let Some(c) = delta["content"].as_str() {
            streamed_content.push_str(c);
        }
    }
    assert_eq!(streamed_reasoning, reasoning);
    assert_eq!(streamed_content, content);
    let last = chunks.last().unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "length");
    assert_eq!(
        last["usage"]["completion_tokens_details"]["reasoning_tokens"],
        2
    );
    assert_eq!(last["timings"]["predicted_n"], 5);

    // vLLM's include_reasoning=false drops it; llama.cpp's
    // reasoning_format=none leaves it inline.
    let (_, body) = chat(
        app.clone(),
        think_body(serde_json::json!({"include_reasoning": false})),
    )
    .await;
    let response: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(response["choices"][0]["message"]
        .get("reasoning_content")
        .is_none());
    assert_eq!(
        response["choices"][0]["message"]["content"],
        content.as_str()
    );
    let (_, body) = chat(
        app,
        think_body(serde_json::json!({"reasoning_format": "none"})),
    )
    .await;
    let response: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        response["choices"][0]["message"]["content"],
        format!("{reasoning}</think>{content}")
    );
}

#[tokio::test]
async fn anthropic_messages_return_thinking_blocks() {
    let state = think_state("server-think-anthropic");
    let engine = state.models.list()[0].engine.clone();
    let app = create_router(state);
    let request = serde_json::json!({
        "model": "model",
        "max_tokens": 5,
        "temperature": 0.0,
        "thinking": {"type": "enabled", "budget_tokens": 2},
        "messages": [{"role": "user", "content": "a b"}],
    });
    // What the engine itself produces for this request.
    let (messages, _, options) = joshua::anthropic::to_chat_request(
        &serde_json::from_value(request.clone()).unwrap(),
        &engine.sampling_defaults(joshua::SamplingProfile::OpenAi),
    )
    .unwrap();
    let expected = engine
        .complete_chat_detailed(&messages, None, &options)
        .unwrap();
    let (reasoning, content) = joshua::reasoning::split(
        engine.reasoning_format(),
        expected.reasoning_open,
        &expected.text,
    );
    assert!(
        expected.reasoning_open,
        "the template opened the reasoning block"
    );

    let res = app
        .clone()
        .oneshot(post_json("/v1/messages", None, request.clone()))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let response: serde_json::Value = serde_json::from_str(&body_text(res).await).unwrap();
    let mut blocks = response["content"].as_array().unwrap().iter();
    if let Some(reasoning) = &reasoning {
        let block = blocks.next().unwrap();
        assert_eq!(block["type"], "thinking");
        assert_eq!(block["thinking"], reasoning.as_str());
        assert_eq!(block["signature"], "");
    }
    if !content.is_empty() {
        let block = blocks.next().unwrap();
        assert_eq!(block["type"], "text");
        assert_eq!(block["text"], content.as_str());
    }
    assert!(response["usage"]["cache_read_input_tokens"].is_u64());
    let reasoning = reasoning.unwrap_or_default();

    let mut streaming = request;
    streaming["stream"] = serde_json::json!(true);
    let res = app
        .oneshot(post_json("/v1/messages", None, streaming))
        .await
        .unwrap();
    let body = body_text(res).await;
    let (mut thinking, mut text, mut signed) = (String::new(), String::new(), false);
    for event in sse_chunks(&body) {
        if event["type"] == "content_block_delta" {
            let delta = &event["delta"];
            match delta["type"].as_str().unwrap() {
                "thinking_delta" => thinking.push_str(delta["thinking"].as_str().unwrap()),
                "text_delta" => text.push_str(delta["text"].as_str().unwrap()),
                "signature_delta" => signed = true,
                _ => {}
            }
        }
        if event["type"] == "message_delta" {
            assert!(
                event["usage"]["cache_read_input_tokens"].is_u64(),
                "{event}"
            );
        }
    }
    assert_eq!(thinking, reasoning);
    assert_eq!(text, content);
    assert_eq!(
        signed,
        !reasoning.is_empty(),
        "a thinking block closes with a signature_delta"
    );
}

#[tokio::test]
async fn chat_lengths_follow_llama_cpp_and_vllm_conventions() {
    let app = create_router(tiny_state("server-lengths", None));
    let base = |extra: serde_json::Value| {
        let mut body = serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "a b"}],
            "temperature": 0.0,
            "ignore_eos": true,
        });
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        body
    };
    // No cap (or -1): the rest of the 64-token context, reported `length`.
    for extra in [serde_json::json!({}), serde_json::json!({"max_tokens": -1})] {
        let (status, body) = chat(app.clone(), base(extra)).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let response: serde_json::Value = serde_json::from_str(&body).unwrap();
        let usage = &response["usage"];
        assert_eq!(usage["total_tokens"], 64, "{body}");
        assert_eq!(response["choices"][0]["finish_reason"], "length");
    }
    // llama.cpp's n_predict wins over max_tokens.
    let (_, body) = chat(
        app.clone(),
        base(serde_json::json!({"max_tokens": 9, "n_predict": 3})),
    )
    .await;
    let response: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(response["usage"]["completion_tokens"], 3);
    // A stop token ends generation with `stop`.
    let (_, body) = chat(
        app.clone(),
        base(serde_json::json!({"logit_bias": {"5": 100}, "stop_token_ids": [5]})),
    )
    .await;
    let response: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(response["choices"][0]["finish_reason"], "stop", "{body}");
    assert_eq!(response["usage"]["completion_tokens"], 0);
    // Out-of-range values are client errors.
    for extra in [
        serde_json::json!({"max_tokens": -2}),
        serde_json::json!({"mirostat": 5}),
        serde_json::json!({"continue_final_message": true, "add_generation_prompt": true}),
        serde_json::json!({"reasoning_format": "bogus"}),
        serde_json::json!({"chat_template": "{% for m in messages %}"}),
        // vLLM's token-id checks, and controls that leave nothing to sample
        // (the only allowed token is EOS, which `ignore_eos` bans): a 400,
        // not a backend failure.
        serde_json::json!({"allowed_token_ids": []}),
        serde_json::json!({"allowed_token_ids": [999999]}),
        serde_json::json!({"allowed_token_ids": [3]}),
        serde_json::json!({"allowed_token_ids": [3], "temperature": 0.8}),
        serde_json::json!({"truncate_prompt_tokens": 0}),
        serde_json::json!({"truncate_prompt_tokens": 65}),
    ] {
        let (status, body) = chat(app.clone(), base(extra.clone())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{extra}: {body}");
    }
    // vLLM's `-1` keeps what the context holds.
    let (status, body) = chat(
        app.clone(),
        base(serde_json::json!({"truncate_prompt_tokens": -1, "max_tokens": 2})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // The developer role and the llama.cpp sampler fields are accepted.
    let (status, body) = chat(
        app,
        serde_json::json!({
            "model": "model",
            "messages": [
                {"role": "developer", "content": "a"},
                {"role": "user", "content": "b"},
                {"role": "assistant", "content": "c"}
            ],
            "max_tokens": 2,
            "repeat_penalty": 1.1,
            "repeat_last_n": 32,
            "typical_p": 0.9,
            "top_n_sigma": 2.0,
            "xtc_probability": 0.1,
            "dry_multiplier": 0.5,
            "mirostat": 2,
            "seed": 1,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

/// Token controls are held to vLLM's limits: a `bad_words` list past them
/// is a 400 before any stream starts (it used to cost time quadratic in
/// its length while holding a session), and a `logit_bias` past `f32`'s
/// range is clamped to OpenAI's ±100 instead of turning the sampled
/// distribution into NaN, a 500 that counted against the NPU.
#[tokio::test]
async fn token_controls_stay_within_vllm_limits() {
    let app = create_router(tiny_state("server-token-limits", None));
    let base = |extra: serde_json::Value| {
        let mut body = serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "a b"}],
            "max_tokens": 3,
            "temperature": 0.7,
            "seed": 1,
            "ignore_eos": true,
        });
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        body
    };
    let many: Vec<String> = (0..20_000).map(|i| format!("w{i}")).collect();
    for stream in [false, true] {
        for (extra, needle) in [
            (serde_json::json!({"bad_words": ["b", ""]}), "empty string"),
            (
                serde_json::json!({"bad_words": ["  "]}),
                "at least one token",
            ),
            (serde_json::json!({"bad_words": many}), "Too many bad words"),
            (
                serde_json::json!({"bad_words": ["a ".repeat(1025)]}),
                "Too many total bad word tokens",
            ),
        ] {
            let mut body = base(extra.clone());
            body["stream"] = serde_json::json!(stream);
            let (status, text) = chat(app.clone(), body).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{stream}: {text}");
            assert!(text.contains(needle), "{needle}: {text}");
        }
        // Sampled, with a bias far past f32's range on `b` (id 5).
        let mut body = base(serde_json::json!({"logit_bias": {"5": 1e39}}));
        body["stream"] = serde_json::json!(stream);
        let (status, text) = chat(app.clone(), body).await;
        assert_eq!(status, StatusCode::OK, "{stream}: {text}");
        assert!(!text.contains("error"), "{stream}: {text}");
        let content: String = if stream {
            sse_chunks(&text)
                .iter()
                .filter_map(|c| {
                    c["choices"][0]["delta"]["content"]
                        .as_str()
                        .map(String::from)
                })
                .collect()
        } else {
            let response: serde_json::Value = serde_json::from_str(&text).unwrap();
            response["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .to_string()
        };
        assert_eq!(content.replace(' ', ""), "bbb", "{stream}: {text}");
    }
}

/// A request's own chat template is code the server would run: refused,
/// before any stream starts, unless the server trusts them (vLLM's
/// `--trust-request-chat-template`).
#[tokio::test]
async fn request_chat_templates_need_the_servers_trust() {
    let body = |stream: bool| {
        serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "a b"}],
            "max_tokens": 2,
            "stream": stream,
            "chat_template": "{% for m in messages %}{{ m.content }} {% endfor %}",
        })
    };
    let app = create_router(tiny_state("server-template-untrusted", None));
    for stream in [false, true] {
        let (status, text) = chat(app.clone(), body(stream)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "stream={stream}: {text}");
        assert!(text.contains("--trust-request-chat-template"), "{text}");
    }

    let dir = model_dir("server-template-trusted");
    write_tiny_llama_gguf(&dir.join("model.gguf"));
    let engine = Engine::with_options(
        &dir,
        joshua::EngineOptions::with_n_ctx(64).trust_request_chat_template(true),
    )
    .expect("engine should load tiny model");
    let app = create_router(Arc::new(ServerState::new(Arc::new(engine))));
    for stream in [false, true] {
        let (status, text) = chat(app.clone(), body(stream)).await;
        assert_eq!(status, StatusCode::OK, "stream={stream}: {text}");
    }
}

#[tokio::test]
async fn chat_reports_cached_prompt_tokens_on_a_follow_up() {
    let app = create_router(tiny_state("server-cached", None));
    // Without a generation prompt the second conversation's prompt extends
    // the first one's, the warm pool's reuse case.
    let request = |content: &str, max_tokens: u32| {
        serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": content}],
            "add_generation_prompt": false,
            "max_tokens": max_tokens,
        })
    };
    let (status, body) = chat(app.clone(), request("a b", 0)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let first: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(first["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
    let (_, body) = chat(app, request("a b c", 1)).await;
    let second: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(
        second["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap()
            > 0,
        "{body}"
    );
}


/// With tools offered, the streamed text goes through the tool-call parser:
/// output without call markup streams as the same content the non-streaming
/// response returns.
#[tokio::test]
async fn streaming_chat_with_tools_reconstructs_the_non_streaming_text() {
    let app = create_router(tiny_state("server-stream-tools", None));
    let request = serde_json::json!({
        "model": "model",
        "messages": [{"role": "user", "content": "a b"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
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
    assert!(response["choices"][0]["message"]["tool_calls"].is_null());

    let mut streaming = request;
    streaming["stream"] = serde_json::json!(true);
    let (status, body) = chat(app, streaming).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let mut text = String::new();
    let mut finish_reason = None;
    for data in body.lines().filter_map(|line| line.strip_prefix("data: ")) {
        if data == "[DONE]" {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(data).unwrap();
        assert!(
            value["choices"][0]["delta"]["tool_calls"].is_null(),
            "{data}"
        );
        if let Some(content) = value["choices"][0]["delta"]["content"].as_str() {
            text.push_str(content);
        }
        if let Some(reason) = value["choices"][0]["finish_reason"].as_str() {
            finish_reason = Some(reason.to_string());
        }
    }
    assert_eq!(text, expected);
    assert_ne!(finish_reason.as_deref(), Some("tool_calls"));
}

/// A tiny llama whose chat template iterates an assistant call's arguments
/// (`arguments|items`, as Qwen3-Coder's and GLM's do), in the fixture's
/// vocabulary so the prompt token count shows what rendered.
fn tool_history_state(dir_name: &str) -> Arc<ServerState> {
    let dir = model_dir(dir_name);
    let model = dir.join("model.gguf");
    write_tiny_llama_gguf(&model);
    common::replace_chat_template(
        &model,
        "{% for m in messages %}{% if m.tool_calls %}{% for c in m.tool_calls %}\
         {% for k, v in c.function.arguments|items %}{{ k }} {% endfor %}{% endfor %}\
         {% else %}hello {{ m.content }} {% endif %}{% endfor %}\
         {% if add_generation_prompt %}world{% endif %}",
    );
    let engine = Engine::with_n_ctx(&dir, 64).expect("engine should load tiny model");
    Arc::new(ServerState::new(Arc::new(engine)))
}

/// Assistant tool calls sent back with OpenAI's JSON-string arguments (and
/// Anthropic `tool_use` blocks) reach the GGUF template as objects, so it
/// renders instead of falling back to ChatML: "hello a b c hello d world"
/// is 7 tokens.
#[tokio::test]
async fn tool_call_history_renders_with_the_gguf_template() {
    let app = create_router(tool_history_state("server-tool-history"));
    let (status, body) = chat(
        app.clone(),
        serde_json::json!({
            "model": "model",
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "f", "arguments": "{\"b\": 1, \"c\": 2}"}
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "d"}
            ],
            "max_tokens": 1,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let response: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(response["usage"]["prompt_tokens"], 7, "{body}");

    let res = app
        .oneshot(post_json(
            "/v1/messages/count_tokens",
            None,
            serde_json::json!({
                "model": "model",
                "max_tokens": 1,
                "messages": [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": [
                        {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {"b": 1, "c": 2}}
                    ]},
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "d"}
                    ]}
                ]
            }),
        ))
        .await
        .unwrap();
    let status = res.status();
    let body = body_text(res).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let count: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(count["input_tokens"], 7, "{count}");
}

// ─── Model cards (/v1/models, /models, /v1/models/{id}) ─────────────────────

async fn get_json(
    app: &axum::Router,
    uri: &str,
    key: Option<&str>,
) -> (StatusCode, serde_json::Value) {
    let res = app.clone().oneshot(get(uri, key)).await.unwrap();
    let status = res.status();
    (status, serde_json::from_str(&body_text(res).await).unwrap())
}

/// The fields that differ between two builds of the same card: vLLM's
/// permission record is minted per response.
fn without_permission(mut card: serde_json::Value) -> serde_json::Value {
    card.as_object_mut().unwrap().remove("permission");
    card
}

#[tokio::test]
async fn models_list_carries_limits_and_upstream_card_fields() {
    let app = create_router(tiny_state("server-model-card", Some("sekret")));
    let (status, list) = get_json(&app, "/v1/models", Some("sekret")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(list["object"], "list");
    let card = &list["data"][0];

    // OpenAI's fields, unchanged.
    assert_eq!(card["id"], "model");
    assert_eq!(card["object"], "model");
    assert_eq!(card["owned_by"], "joshua");
    assert!(card["created"].as_u64().unwrap() > 0);
    // #260: the per-request ceiling (--n-ctx), the trained context, and the
    // output cap, reported as configured (the default 4096) even though a
    // 64-token context runs out first, as #261 reports it.
    assert_eq!(card["context_length"], 64);
    assert_eq!(card["max_context_length"], 512);
    assert_eq!(card["max_output_tokens"], 4096, "{card}");
    // vLLM's ModelCard.
    assert!(
        card["root"].as_str().unwrap().ends_with("model.gguf"),
        "{card}"
    );
    assert!(card.get("parent").is_some_and(|p| p.is_null()));
    assert_eq!(card["max_model_len"], 64);
    // llama-server's entry; the tiny GGUF is all F32 with no file type.
    assert_eq!(card["aliases"], serde_json::json!([]));
    assert_eq!(card["tags"], serde_json::json!([]));
    assert_eq!(
        card["meta"],
        serde_json::json!({
            "vocab_type": 1, "n_vocab": 16, "n_ctx": 64, "n_ctx_train": 512,
            "n_embd": 8, "n_params": 792, "size": 792 * 4,
        })
    );
    // joshua's own facts: where the model runs.
    let joshua = &card["joshua"];
    assert_eq!(joshua["loaded"], true);
    assert_eq!(joshua["architecture"], "llama");
    assert!(joshua["file_size"].as_u64().unwrap() > 792 * 4);
    assert_eq!(joshua["backend"], "cpu");
    assert_eq!(joshua["dense"], "cpu");
    assert!(joshua.get("experts").is_none(), "a dense model: {joshua}");

    // llama-server also serves the list at /models; GET /v1/models/{id}
    // returns the same card.  All of them sit behind the key.
    let (status, models) = get_json(&app, "/models", Some("sekret")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        without_permission(models["data"][0].clone()),
        without_permission(card.clone())
    );
    let (status, one) = get_json(&app, "/v1/models/model", Some("sekret")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(without_permission(one), without_permission(card.clone()));
    for uri in ["/models", "/v1/models/model"] {
        let res = app.clone().oneshot(get(uri, None)).await.unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED, "{uri}");
    }

    // An id no entry has is OpenAI's 404, even though a single loaded model
    // answers any name on the inference routes.
    let (status, err) = get_json(&app, "/v1/models/gpt-4o", Some("sekret")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(
        err,
        serde_json::json!({"error": {
            "message": "The model `gpt-4o` does not exist.",
            "type": "invalid_request_error",
            "param": "model",
            "code": "model_not_found",
        }})
    );
}

#[tokio::test]
async fn model_cards_use_vllm_and_llama_cpp_field_names() {
    let app = create_router(tiny_state("server-model-card-names", None));
    let (_, list) = get_json(&app, "/v1/models", None).await;
    let card = list["data"][0].as_object().unwrap();
    fn keys(v: &serde_json::Map<String, serde_json::Value>) -> std::collections::BTreeSet<&str> {
        v.keys().map(String::as_str).collect()
    }
    // vLLM `ModelCard` (vllm/entrypoints/serve/engine/protocol.py).
    for field in [
        "id",
        "object",
        "created",
        "owned_by",
        "root",
        "parent",
        "max_model_len",
        "permission",
    ] {
        assert!(
            card.contains_key(field),
            "vLLM field {field} missing: {card:?}"
        );
    }
    // vLLM `ModelPermission`, every field and nothing else.
    let permission = card["permission"][0].as_object().unwrap();
    assert_eq!(
        keys(permission),
        [
            "allow_create_engine",
            "allow_fine_tuning",
            "allow_logprobs",
            "allow_sampling",
            "allow_search_indices",
            "allow_view",
            "created",
            "group",
            "id",
            "is_blocking",
            "object",
            "organization",
        ]
        .into()
    );
    assert_eq!(permission["object"], "model_permission");
    assert!(permission["id"].as_str().unwrap().starts_with("modelperm-"));
    assert_eq!(permission["organization"], "*");
    assert_eq!(permission["allow_sampling"], true);
    assert_eq!(permission["allow_view"], true);
    assert_eq!(permission["allow_create_engine"], false);
    // llama-server's `get_res_model_info` (tools/server/server-context.cpp).
    for field in [
        "id", "aliases", "tags", "object", "created", "owned_by", "meta",
    ] {
        assert!(
            card.contains_key(field),
            "llama.cpp field {field} missing: {card:?}"
        );
    }
    let meta = card["meta"].as_object().unwrap();
    let llama_cpp_meta: std::collections::BTreeSet<_> = [
        "vocab_type",
        "n_vocab",
        "n_ctx",
        "n_ctx_train",
        "n_embd",
        "n_params",
        "size",
        "ftype",
    ]
    .into();
    assert!(keys(meta).is_subset(&llama_cpp_meta), "{meta:?}");
    // Nothing joshua adds shadows an upstream name: its own facts sit under
    // `joshua`, and the rest is #260's three limits.
    let upstream: std::collections::BTreeSet<_> = [
        "id",
        "object",
        "created",
        "owned_by",
        "root",
        "parent",
        "max_model_len",
        "permission",
        "aliases",
        "tags",
        "meta",
    ]
    .into();
    let added: Vec<_> = keys(card).difference(&upstream).copied().collect();
    assert_eq!(
        added,
        [
            "context_length",
            "joshua",
            "max_context_length",
            "max_output_tokens"
        ]
    );
}

#[tokio::test]
async fn model_cards_report_the_output_cap_and_expert_placement() {
    // An output cap below the context is reported.
    let dir = model_dir("server-model-card-cap");
    write_tiny_llama_gguf(&dir.join("model.gguf"));
    let engine = Engine::with_n_ctx(&dir, 64)
        .unwrap()
        .with_max_output_tokens(16);
    let app = create_router(Arc::new(ServerState::new(Arc::new(engine))));
    let (_, list) = get_json(&app, "/v1/models", None).await;
    assert_eq!(list["data"][0]["max_output_tokens"], 16);
    assert_eq!(list["data"][0]["context_length"], 64);

    // A mixture-of-experts model says where its routed experts live.
    let dir = model_dir("server-model-card-moe");
    common::write_tiny_qwen3moe_gguf(&dir.join("moe.gguf"));
    let engine = Engine::with_n_ctx(&dir, 32).unwrap();
    let app = create_router(Arc::new(ServerState::new(Arc::new(engine))));
    let (_, card) = get_json(&app, "/v1/models/moe", None).await;
    assert_eq!(card["joshua"]["architecture"], "qwen3moe");
    assert_eq!(card["joshua"]["experts"], "cpu");
    assert_eq!(card["context_length"], 32);
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
