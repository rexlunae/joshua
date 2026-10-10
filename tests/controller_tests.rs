//! Loading and unloading models through the API, locally and onto
//! model-less pipeline workers.
mod common;

use std::sync::Arc;

use axum::{
    body::Body,
    http::{header, Request, StatusCode},
    Router,
};
use joshua::{
    server::{create_router, ModelManager, ModelRegistry, ServerState},
    EngineOptions,
};
use tower::ServiceExt;

async fn call(
    app: &Router,
    method: &str,
    uri: &str,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let req = Request::builder()
        .method(method)
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(if method == "GET" {
            Body::empty()
        } else {
            Body::from(body.to_string())
        })
        .unwrap();
    let res = app.clone().oneshot(req).await.unwrap();
    let status = res.status();
    let bytes = axum::body::to_bytes(res.into_body(), 1 << 20)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or_default())
}

fn manager(dir: &std::path::Path) -> ModelManager {
    ModelManager {
        model_dir: dir.to_path_buf(),
        engine_options: Arc::new(|_: &std::path::Path| EngineOptions::with_n_ctx(64)),
        npu_backend: None,
        max_concurrency: None,
        max_output_tokens: None,
        #[cfg(feature = "distributed")]
        workers: Vec::new(),
        #[cfg(feature = "distributed")]
        cluster_key: None,
    }
}

fn state(models: ModelRegistry, manager: Option<ModelManager>) -> Arc<ServerState> {
    Arc::new(ServerState {
        models,
        manager,
        whisper: None,
        api_key: None,
    })
}

fn completion(model: &str) -> serde_json::Value {
    serde_json::json!({"model": model, "prompt": "hello world", "max_tokens": 6, "temperature": 0.0})
}

#[tokio::test]
async fn a_server_without_a_model_loads_and_unloads_through_the_api() {
    let dir = common::model_dir("controller-local");
    common::write_tiny_llama_gguf(&dir.join("tiny.gguf"));
    let app = create_router(state(ModelRegistry::default(), Some(manager(&dir))));

    let (status, body) = call(&app, "POST", "/v1/completions", completion("tiny")).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");

    let (status, body) = call(
        &app,
        "POST",
        "/v1/models/load",
        serde_json::json!({"model": "tiny.gguf"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["id"], "tiny");
    let (status, _) = call(
        &app,
        "POST",
        "/v1/models/load",
        serde_json::json!({"model": "tiny.gguf"}),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT);
    let (status, body) = call(
        &app,
        "POST",
        "/v1/models/load",
        serde_json::json!({"model": "tiny.gguf", "id": "second"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let (_, body) = call(&app, "GET", "/v1/models", serde_json::Value::Null).await;
    let ids: Vec<_> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].clone())
        .collect();
    assert_eq!(ids, ["tiny", "second"]);
    let (status, body) = call(&app, "POST", "/v1/completions", completion("second")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // Responses name the model by the id it was loaded under.
    assert_eq!(body["model"], "second");
    // With two models loaded, a request must name one of them.
    let (status, _) = call(&app, "POST", "/v1/completions", completion("other")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let (status, _) = call(
        &app,
        "POST",
        "/v1/models/unload",
        serde_json::json!({"id": "tiny"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = call(
        &app,
        "POST",
        "/v1/models/unload",
        serde_json::json!({"id": "tiny"}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (_, body) = call(&app, "GET", "/v1/models", serde_json::Value::Null).await;
    assert_eq!(body["data"].as_array().unwrap().len(), 1);
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn model_paths_stay_inside_the_model_directory_and_management_is_opt_in() {
    let dir = common::model_dir("controller-paths");
    common::write_tiny_llama_gguf(&dir.join("tiny.gguf"));
    let inner = dir.join("models");
    std::fs::create_dir_all(&inner).unwrap();
    let app = create_router(state(ModelRegistry::default(), Some(manager(&inner))));
    for model in ["../tiny.gguf", "/etc/passwd", "missing.gguf"] {
        let (status, body) = call(
            &app,
            "POST",
            "/v1/models/load",
            serde_json::json!({"model": model}),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{model}: {body}");
    }
    let app = create_router(state(ModelRegistry::default(), None));
    let (status, _) = call(
        &app,
        "POST",
        "/v1/models/load",
        serde_json::json!({"model": "tiny.gguf"}),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    // Other sites' pages may call the inference API, not model management.
    let preflight = |uri: &str| {
        Request::builder()
            .method("OPTIONS")
            .uri(uri)
            .header(header::ORIGIN, "https://example.com")
            .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
            .body(Body::empty())
            .unwrap()
    };
    for (uri, allowed) in [
        ("/v1/chat/completions", true),
        ("/v1/models/load", false),
        ("/v1/models/unload", false),
    ] {
        let res = app.clone().oneshot(preflight(uri)).await.unwrap();
        assert_eq!(
            res.headers()
                .contains_key(header::ACCESS_CONTROL_ALLOW_ORIGIN),
            allowed,
            "{uri}"
        );
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(feature = "distributed")]
#[tokio::test(flavor = "multi_thread")]
async fn controller_places_a_model_on_model_less_workers_and_unloads_it() {
    use joshua::distributed::pipeline::{serve_controller, AgentConfig};
    const KEY: &[u8] = b"controller-test-key-32-bytes-long!";

    let dir = common::model_dir("controller-distributed");
    let path = dir.join("qwen.gguf");
    common::write_tiny_qwen3_layers(&path, 3);
    // The same model run locally is the reference for greedy output.
    let local = create_router(state(
        ModelRegistry::with_model(Arc::new(
            joshua::Engine::with_options(&path, EngineOptions::with_n_ctx(64)).unwrap(),
        )),
        None,
    ));
    let (status, expected) = call(&local, "POST", "/v1/completions", completion("qwen")).await;
    assert_eq!(status, StatusCode::OK, "{expected}");
    assert!(
        expected["usage"]["completion_tokens"].as_u64().unwrap() > 0,
        "{expected}"
    );

    // Workers start empty and accept two controller connections each.
    let mut workers = Vec::new();
    let mut handles = Vec::new();
    for _ in 0..2 {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        workers.push(listener.local_addr().unwrap());
        handles.push(std::thread::spawn(move || {
            (0..2)
                .map(|_| {
                    let (stream, _) = listener.accept().unwrap();
                    serve_controller(stream, KEY, &AgentConfig::default())
                })
                .collect::<Vec<_>>()
        }));
    }
    let mut manager = manager(&dir);
    manager.workers = workers.clone();
    manager.cluster_key = Some(KEY.to_vec());
    let app = create_router(state(ModelRegistry::default(), Some(manager)));

    for round in 0..2 {
        let request = if round == 0 {
            serde_json::json!({"model": "qwen.gguf", "n_ctx": 64, "chunk": 16, "sessions": 4})
        } else {
            serde_json::json!({"model": "qwen.gguf", "n_ctx": 64, "ends": [2, 3]})
        };
        let (status, body) = call(&app, "POST", "/v1/models/load", request).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["workers"].as_array().unwrap().len(), 2);
        if round == 1 {
            assert_eq!(body["stages"], serde_json::json!([[0, 2], [2, 3]]));
        }
        let (status, got) = call(&app, "POST", "/v1/completions", completion("qwen")).await;
        assert_eq!(status, StatusCode::OK, "{got}");
        assert_eq!(got["choices"][0]["text"], expected["choices"][0]["text"]);
        assert_eq!(got["usage"], expected["usage"]);
        // A second, multi-turn request reuses a warm remote session.
        let (status, again) = call(&app, "POST", "/v1/completions", completion("qwen")).await;
        assert_eq!(status, StatusCode::OK, "{again}");
        assert_eq!(again["choices"][0]["text"], expected["choices"][0]["text"]);
        // Embeddings would need the weights on this host.
        let (status, _) = call(
            &app,
            "POST",
            "/v1/embeddings",
            serde_json::json!({"model": "qwen", "input": "hello"}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        // A worker holds one model at a time.
        let (status, body) = call(
            &app,
            "POST",
            "/v1/models/load",
            serde_json::json!({"model": "qwen.gguf", "id": "other", "n_ctx": 64}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        // Unload returns once the workers are free, so the next round's
        // load reaches them immediately.
        let (status, _) = call(
            &app,
            "POST",
            "/v1/models/unload",
            serde_json::json!({"id": "qwen"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }
    // Unloading released every worker cleanly.
    for handle in handles {
        for result in tokio::task::spawn_blocking(move || handle.join().unwrap())
            .await
            .unwrap()
        {
            result.unwrap();
        }
    }
    // Only configured workers can be named.
    let (status, body) = call(
        &app,
        "POST",
        "/v1/models/load",
        serde_json::json!({"model": "qwen.gguf", "workers": ["127.0.0.1:9"]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    std::fs::remove_dir_all(dir).unwrap();
}
