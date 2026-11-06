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
        cors_origins: Vec::new(),
        metrics: Arc::new(joshua::server::ServerMetrics::default()),
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

    // Nothing is loaded, and `missing` is no model the directory lists.
    let (status, body) = call(&app, "POST", "/v1/completions", completion("missing")).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    let (status, _) = call(&app, "GET", "/v1/worker/info", serde_json::Value::Null).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);

    // A request that names a listed model loads it: no explicit load first.
    let (status, body) = call(&app, "POST", "/v1/completions", completion("tiny")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["model"], "tiny");
    // A `joshua route` coordinator sees the loaded model.
    let (status, body) = call(&app, "GET", "/v1/worker/info", serde_json::Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["model"], "tiny");
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
        serde_json::json!({"model": "tiny.gguf", "id": "second", "n_ctx": 128}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let (status, info) = call(&app, "GET", "/v1/worker/info", serde_json::Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{info}");
    assert_eq!(info["n_ctx"], 64);
    assert_eq!(info["models"][0]["n_ctx"], 64);
    assert_eq!(info["models"][1]["n_ctx"], 128);

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
    let (status, body) = call(&app, "POST", "/v1/completions", completion("other")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
    assert_eq!(body["error"]["param"], "model", "{body}");
    assert_eq!(body["error"]["code"], "model_not_found", "{body}");

    let (status, _) = call(
        &app,
        "POST",
        "/v1/models/unload",
        serde_json::json!({"id": "tiny"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, body) = call(
        &app,
        "POST",
        "/v1/models/unload",
        serde_json::json!({"id": "tiny"}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "model_not_found", "{body}");
    let listed = |body: serde_json::Value| -> Vec<serde_json::Value> {
        body["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| serde_json::json!([m["id"], m["joshua"]["loaded"]]))
            .collect()
    };
    // `second` stays loaded from `tiny.gguf`, so the file is not listed
    // again; once nothing serves it, it is, as a file to load.
    let (_, body) = call(&app, "GET", "/v1/models", serde_json::Value::Null).await;
    assert_eq!(listed(body), [serde_json::json!(["second", true])]);
    let (status, _) = call(
        &app,
        "POST",
        "/v1/models/unload",
        serde_json::json!({"id": "second"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (_, body) = call(&app, "GET", "/v1/models", serde_json::Value::Null).await;
    assert_eq!(listed(body), [serde_json::json!(["tiny", false])]);
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn the_model_directory_lists_unloaded_files_from_their_headers() {
    let dir = common::model_dir("controller-cards");
    common::write_tiny_llama_gguf(&dir.join("tiny.gguf"));
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    common::write_tiny_gguf(&dir.join("sub").join("other.gguf"), "qwen3");
    let mut manager = manager(&dir);
    manager.max_output_tokens = Some(16);
    let app = create_router(state(ModelRegistry::default(), Some(manager)));

    // Nothing is loaded: both files are listed from their headers, with the
    // limits a load with the server's flags (n_ctx 64, max output 16) gives.
    let (status, body) = call(&app, "GET", "/v1/models", serde_json::Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let cards = body["data"].as_array().unwrap();
    let ids: Vec<_> = cards.iter().map(|m| m["id"].as_str().unwrap()).collect();
    assert_eq!(ids, ["other", "tiny"]);
    for card in cards {
        assert_eq!(card["joshua"]["loaded"], false, "{card}");
        assert_eq!(card["context_length"], 64, "{card}");
        assert_eq!(card["max_model_len"], 64, "{card}");
        assert_eq!(card["max_context_length"], 512, "{card}");
        assert_eq!(card["max_output_tokens"], 16, "{card}");
        assert_eq!(card["meta"]["n_ctx_train"], 512, "{card}");
        assert_eq!(card["meta"]["n_vocab"], 16, "{card}");
        assert!(
            card["joshua"].get("backend").is_none(),
            "not placed yet: {card}"
        );
        assert!(card["joshua"]["file_size"].as_u64().unwrap() > 0);
    }
    assert_eq!(cards[0]["joshua"]["architecture"], "qwen3");
    assert!(cards[0]["root"].as_str().unwrap().ends_with("other.gguf"));
    let (status, other) = call(&app, "GET", "/v1/models/other", serde_json::Value::Null).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(other["joshua"]["loaded"], false);

    // Loading one turns its entry into the loaded model's; the other file
    // stays listed.  The load's answer is the same card.
    let (status, loaded) = call(
        &app,
        "POST",
        "/v1/models/load",
        serde_json::json!({"model": "tiny.gguf"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{loaded}");
    assert_eq!(loaded["context_length"], 64);
    assert_eq!(loaded["joshua"]["loaded"], true);
    assert_eq!(loaded["workers"], serde_json::json!([]));
    let (_, body) = call(&app, "GET", "/v1/models", serde_json::Value::Null).await;
    let cards = body["data"].as_array().unwrap();
    assert_eq!(cards[0]["id"], "tiny");
    assert_eq!(cards[0]["joshua"]["loaded"], true);
    assert_eq!(cards[0]["joshua"]["backend"], "cpu");
    assert_eq!(cards[0]["max_output_tokens"], 16);
    assert_eq!(cards[1]["id"], "other");
    assert_eq!(cards[1]["joshua"]["loaded"], false);

    // A listed file's id belongs to that file: a request that names it loads
    // that file — which cannot load yet, its tokenizer is missing — rather than
    // being answered by the model that is loaded.
    let messages = serde_json::json!([{"role": "user", "content": "hi"}]);
    for (uri, request) in [
        ("/v1/completions", completion("other")),
        (
            "/v1/chat/completions",
            serde_json::json!({"model": "other", "messages": messages, "max_tokens": 2}),
        ),
        (
            "/v1/messages",
            serde_json::json!({"model": "other", "messages": messages, "max_tokens": 2}),
        ),
        (
            "/api/chat",
            serde_json::json!({"model": "other", "messages": messages, "stream": false}),
        ),
    ] {
        let (status, body) = call(&app, "POST", uri, request).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{uri}: {body}");
        if uri.starts_with("/api/") {
            // Ollama's error shape: the message alone.
            assert!(body["error"].as_str().is_some_and(|m| !m.is_empty()), "{uri}: {body}");
        } else {
            assert_eq!(body["error"]["type"], "model_load_failed", "{uri}: {body}");
        }
    }

    // Ollama's /api/tags lists the same inventory, now with each file's size
    // and family; /api/ps only what is loaded.
    let (_, tags) = call(&app, "GET", "/api/tags", serde_json::Value::Null).await;
    let names: Vec<_> = tags["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["name"].clone())
        .collect();
    assert_eq!(names, ["tiny:latest", "other:latest"]);
    assert_eq!(tags["models"][0]["details"]["family"], "llama");
    assert_eq!(tags["models"][0]["size"], cards[0]["joshua"]["file_size"]);
    assert_eq!(tags["models"][1]["details"]["family"], "qwen3");
    let (_, ps) = call(&app, "GET", "/api/ps", serde_json::Value::Null).await;
    assert_eq!(ps["models"].as_array().unwrap().len(), 1);
    assert_eq!(ps["models"][0]["name"], "tiny:latest");

    // The id the list gives the file in the subdirectory loads it.
    std::fs::copy(dir.join("tokenizer.json"), dir.join("sub/tokenizer.json")).unwrap();
    let (status, loaded) = call(
        &app,
        "POST",
        "/v1/models/load",
        serde_json::json!({"model": "other"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{loaded}");
    assert_eq!(loaded["id"], "other");
    assert!(loaded["root"].as_str().unwrap().ends_with("other.gguf"));
    let (_, body) = call(&app, "GET", "/v1/models", serde_json::Value::Null).await;
    let loaded: Vec<_> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| serde_json::json!([m["id"], m["joshua"]["loaded"]]))
        .collect();
    assert_eq!(
        loaded,
        [
            serde_json::json!(["tiny", true]),
            serde_json::json!(["other", true])
        ]
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn each_model_file_is_listed_once_under_an_id_a_load_takes() {
    use candle_core::quantized::gguf_file::Value;

    let dir = common::model_dir("controller-inventory");
    for sub in ["a", "b"] {
        std::fs::create_dir_all(dir.join(sub)).unwrap();
        common::write_tiny_llama_gguf(&dir.join(sub).join("model.gguf"));
        std::fs::copy(
            dir.join("tokenizer.json"),
            dir.join(sub).join("tokenizer.json"),
        )
        .unwrap();
    }
    common::write_tiny_llama_gguf(&dir.join("solo.gguf"));
    // Files that are not models on their own.
    for (name, kind) in [("lora.gguf", "adapter"), ("mmproj.gguf", "mmproj")] {
        let kind = [("general.type".to_string(), Value::String(kind.to_string()))];
        common::write_tiny_gguf_with(&dir.join(name), "llama", &kind);
    }
    // A link leading out of the directory, which a load refuses, and a
    // second link to a file in it.
    #[cfg(unix)]
    let outside = {
        let outside = common::model_dir("controller-inventory-outside");
        common::write_tiny_llama_gguf(&outside.join("far.gguf"));
        std::os::unix::fs::symlink(outside.join("far.gguf"), dir.join("far.gguf")).unwrap();
        std::os::unix::fs::symlink(dir.join("a/model.gguf"), dir.join("same.gguf")).unwrap();
        outside
    };
    let app = create_router(state(ModelRegistry::default(), Some(manager(&dir))));
    let ids = |body: &serde_json::Value| -> Vec<serde_json::Value> {
        body["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| serde_json::json!([m["id"], m["joshua"]["loaded"]]))
            .collect()
    };

    // Both `model.gguf`s are listed, each under its path since they share
    // a stem; a file with a stem of its own goes by the stem.
    let (_, body) = call(&app, "GET", "/v1/models", serde_json::Value::Null).await;
    assert_eq!(
        ids(&body),
        [
            serde_json::json!(["a/model.gguf", false]),
            serde_json::json!(["b/model.gguf", false]),
            serde_json::json!(["solo", false]),
        ]
    );
    assert!(body["data"][0]["root"]
        .as_str()
        .unwrap()
        .ends_with("a/model.gguf"));
    let (status, card) = call(
        &app,
        "GET",
        "/v1/models/b/model.gguf",
        serde_json::Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{card}");
    assert!(card["root"].as_str().unwrap().ends_with("b/model.gguf"));

    // A load without an `id` serves the model under the id it was listed
    // by, and the other file keeps its own.
    let (status, body) = call(
        &app,
        "POST",
        "/v1/models/load",
        serde_json::json!({"model": "b/model.gguf"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["id"], "b/model.gguf");
    assert!(body["root"].as_str().unwrap().ends_with("b/model.gguf"));
    let (_, body) = call(&app, "GET", "/v1/models", serde_json::Value::Null).await;
    assert_eq!(
        ids(&body),
        [
            serde_json::json!(["b/model.gguf", true]),
            serde_json::json!(["a/model.gguf", false]),
            serde_json::json!(["solo", false]),
        ]
    );
    let (status, body) = call(&app, "POST", "/v1/completions", completion("b/model.gguf")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["model"], "b/model.gguf");

    // A model loaded under a file's stem as its id: that file is listed by
    // its path, so no two entries share an id.
    let (status, body) = call(
        &app,
        "POST",
        "/v1/models/load",
        serde_json::json!({"model": "a/model.gguf", "id": "solo"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (_, body) = call(&app, "GET", "/v1/models", serde_json::Value::Null).await;
    assert_eq!(
        ids(&body),
        [
            serde_json::json!(["b/model.gguf", true]),
            serde_json::json!(["solo", true]),
            serde_json::json!(["solo.gguf", false]),
        ]
    );
    assert!(body["data"][1]["root"]
        .as_str()
        .unwrap()
        .ends_with("a/model.gguf"));
    let (status, body) = call(
        &app,
        "POST",
        "/v1/models/unload",
        serde_json::json!({"id": "solo"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (_, body) = call(&app, "GET", "/v1/models", serde_json::Value::Null).await;
    assert_eq!(
        ids(&body),
        [
            serde_json::json!(["b/model.gguf", true]),
            serde_json::json!(["a/model.gguf", false]),
            serde_json::json!(["solo", false]),
        ]
    );

    // Paths the list leaves out (an adapter, a link leading out of the
    // directory) are free to take as ids.
    let mut unlisted = vec!["lora.gguf"];
    if cfg!(unix) {
        unlisted.push("far.gguf");
    }
    for id in unlisted {
        let load = serde_json::json!({"model": "solo.gguf", "id": id});
        let (status, body) = call(&app, "POST", "/v1/models/load", load).await;
        assert_eq!(status, StatusCode::OK, "{id}: {body}");
        let unload = serde_json::json!({"id": id});
        let (status, body) = call(&app, "POST", "/v1/models/unload", unload).await;
        assert_eq!(status, StatusCode::OK, "{id}: {body}");
    }
    std::fs::remove_dir_all(dir).unwrap();
    #[cfg(unix)]
    std::fs::remove_dir_all(outside).unwrap();
}

/// A file's path is its id once its stem is shared, so a load cannot take
/// another file's path as its id, and a file whose path a loaded model
/// already has (it appeared after the load) is not listed beside it: every
/// listed id answers for its own entry.
#[tokio::test]
async fn no_two_listed_models_share_an_id() {
    let dir = common::model_dir("controller-path-ids");
    for sub in ["a", "b"] {
        std::fs::create_dir_all(dir.join(sub)).unwrap();
        common::write_tiny_llama_gguf(&dir.join(sub).join("model.gguf"));
        std::fs::copy(
            dir.join("tokenizer.json"),
            dir.join(sub).join("tokenizer.json"),
        )
        .unwrap();
    }
    let app = create_router(state(ModelRegistry::default(), Some(manager(&dir))));
    let ids = |body: &serde_json::Value| -> Vec<serde_json::Value> {
        body["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| serde_json::json!([m["id"], m["joshua"]["loaded"]]))
            .collect()
    };
    let load = |model: &str, id: &str| serde_json::json!({"model": model, "id": id});

    // `a/model.gguf` is the other file's id: refused, nothing loaded.
    let (status, body) = call(
        &app,
        "POST",
        "/v1/models/load",
        load("b/model.gguf", "a/model.gguf"),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["type"], "model_exists");
    let (_, body) = call(&app, "GET", "/v1/models", serde_json::Value::Null).await;
    assert_eq!(
        ids(&body),
        [
            serde_json::json!(["a/model.gguf", false]),
            serde_json::json!(["b/model.gguf", false]),
        ]
    );

    // A path no file has yet is free; a file copied there afterwards is not
    // listed under the loaded model's id until that model is unloaded.
    let (status, body) = call(
        &app,
        "POST",
        "/v1/models/load",
        load("b/model.gguf", "c/model.gguf"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    std::fs::create_dir_all(dir.join("c")).unwrap();
    common::write_tiny_llama_gguf(&dir.join("c").join("model.gguf"));
    let (_, body) = call(&app, "GET", "/v1/models", serde_json::Value::Null).await;
    assert_eq!(
        ids(&body),
        [
            serde_json::json!(["c/model.gguf", true]),
            serde_json::json!(["a/model.gguf", false]),
        ]
    );
    assert!(body["data"][0]["root"]
        .as_str()
        .unwrap()
        .ends_with("b/model.gguf"));
    let (status, body) = call(&app, "POST", "/v1/completions", completion("c/model.gguf")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["model"], "c/model.gguf");
    let (status, body) = call(
        &app,
        "POST",
        "/v1/models/unload",
        serde_json::json!({"id": "c/model.gguf"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (_, body) = call(&app, "GET", "/v1/models", serde_json::Value::Null).await;
    assert_eq!(
        ids(&body),
        [
            serde_json::json!(["a/model.gguf", false]),
            serde_json::json!(["b/model.gguf", false]),
            serde_json::json!(["c/model.gguf", false]),
        ]
    );

    // A loaded file's path is refused as another load's id too, so the
    // file is listed under it again once it is unloaded.
    let (status, body) = call(&app, "POST", "/v1/models/load", load("b/model.gguf", "bee")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, body) = call(
        &app,
        "POST",
        "/v1/models/load",
        load("a/model.gguf", "b/model.gguf"),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    std::fs::remove_dir_all(dir).unwrap();
}

/// A request's model name is checked against the model directory's ids
/// without reading a file there: a FIFO named like a GGUF, whose header
/// read would block until someone writes to it, holds nothing up.
#[cfg(unix)]
#[test]
fn a_model_name_is_checked_without_reading_a_header() {
    use axum::response::IntoResponse;

    let dir = common::model_dir("controller-resolve");
    let path = dir.join("tiny.gguf");
    common::write_tiny_llama_gguf(&path);
    let fifo = std::process::Command::new("mkfifo")
        .arg(dir.join("pending.gguf"))
        .status()
        .unwrap();
    assert!(fifo.success());
    let engine = joshua::Engine::with_options(&path, EngineOptions::with_n_ctx(64)).unwrap();
    let state = state(
        ModelRegistry::with_model(Arc::new(engine)),
        Some(manager(&dir)),
    );

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn({
        let state = Arc::clone(&state);
        move || {
            let resolve = |name: &str| {
                state
                    .resolve(name)
                    .map(|(id, _)| id)
                    .map_err(|e| e.into_response().status())
            };
            let _ = tx.send((resolve("gpt-4o"), resolve("pending"), resolve("tiny")));
        }
    });
    let (other, pending, tiny) = rx
        .recv_timeout(std::time::Duration::from_secs(60))
        .expect("resolving a model name opened a file in the model directory");
    // The one model loaded answers a name no file has; a file's id is a
    // 404 even though its header was never read.
    assert_eq!(other, Ok("tiny".to_string()));
    assert_eq!(pending, Err(StatusCode::NOT_FOUND));
    assert_eq!(tiny, Ok("tiny".to_string()));
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
    // A directory inside the root whose model file links outside it.
    #[cfg(unix)]
    {
        let linked = inner.join("linked");
        std::fs::create_dir_all(&linked).unwrap();
        std::os::unix::fs::symlink(dir.join("tiny.gguf"), linked.join("tiny.gguf")).unwrap();
        let (status, body) = call(
            &app,
            "POST",
            "/v1/models/load",
            serde_json::json!({"model": "linked"}),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
        // A link to a model inside the root loads, with the tokenizer
        // found beside the link.
        let alias = inner.join("alias");
        std::fs::create_dir_all(&alias).unwrap();
        std::fs::create_dir_all(inner.join("weights")).unwrap();
        std::fs::copy(dir.join("tiny.gguf"), inner.join("weights/tiny.gguf")).unwrap();
        std::os::unix::fs::symlink(inner.join("weights/tiny.gguf"), alias.join("tiny.gguf"))
            .unwrap();
        std::fs::copy(dir.join("tokenizer.json"), alias.join("tokenizer.json")).unwrap();
        let (status, body) = call(
            &app,
            "POST",
            "/v1/models/load",
            serde_json::json!({"model": "alias"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
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
    // The controller samples, so the server's sampling flags reach it.  Its
    // --n-ctx is past the context the model was trained with (512).
    manager.engine_options = Arc::new(|_: &std::path::Path| {
        EngineOptions::with_n_ctx(1024).sampling(joshua::SamplingOverrides {
            temperature: Some(0.25),
            ..Default::default()
        })
    });
    let app = create_router(state(ModelRegistry::default(), Some(manager)));

    // Not loaded yet, the file is listed with the context a load onto the
    // workers gives it: the trained context, not the larger --n-ctx.
    let (status, body) = call(&app, "GET", "/v1/models", serde_json::Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let listed = &body["data"][0];
    assert_eq!(listed["id"], "qwen", "{body}");
    assert_eq!(listed["joshua"]["loaded"], false, "{body}");
    assert_eq!(listed["context_length"], 512, "{body}");
    assert_eq!(listed["max_model_len"], 512, "{body}");

    for round in 0..2 {
        // The second round names no context and gets the one listed.
        let (request, context) = if round == 0 {
            let request = serde_json::json!({
                "model": "qwen.gguf", "n_ctx": 64, "chunk": 16, "sessions": 4,
            });
            (request, serde_json::json!(64))
        } else {
            let request = serde_json::json!({"model": "qwen.gguf", "ends": [2, 3]});
            (request, listed["context_length"].clone())
        };
        let (status, body) = call(&app, "POST", "/v1/models/load", request).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["workers"].as_array().unwrap().len(), 2);
        if round == 1 {
            assert_eq!(body["stages"], serde_json::json!([[0, 2], [2, 3]]));
        }
        // The answer is the model's /v1/models entry, which says it runs on
        // the workers.
        assert_eq!(body["context_length"], context);
        assert_eq!(body["joshua"]["remote"], true);
        assert_eq!(body["joshua"]["workers"], body["workers"]);
        assert_eq!(body["joshua"]["stages"], body["stages"]);
        let (status, props) = call(&app, "GET", "/props", serde_json::Value::Null).await;
        assert_eq!(status, StatusCode::OK, "{props}");
        assert_eq!(props["default_generation_settings"]["temperature"], 0.25);
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

#[tokio::test(flavor = "multi_thread")]
async fn a_request_router_reaches_every_loaded_model() {
    use joshua::coordinator::{Coordinator, CoordinatorConfig};

    let dir = common::model_dir("controller-routed");
    common::write_tiny_llama_gguf(&dir.join("tiny.gguf"));
    let app = create_router(state(ModelRegistry::default(), Some(manager(&dir))));
    for id in ["tiny", "second"] {
        let (status, body) = call(
            &app,
            "POST",
            "/v1/models/load",
            serde_json::json!({"model": "tiny.gguf", "id": id}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }
    let (status, info) = call(&app, "GET", "/v1/worker/info", serde_json::Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{info}");
    let ids: Vec<_> = info["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].clone())
        .collect();
    assert_eq!(ids, ["tiny", "second"]);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let coordinator = Coordinator::new(CoordinatorConfig::new(vec![addr])).unwrap();
    coordinator.refresh().await;
    let router = joshua::coordinator::create_router(coordinator);
    for model in ["tiny", "second"] {
        let (status, body) = call(&router, "POST", "/v1/completions", completion(model)).await;
        assert_eq!(status, StatusCode::OK, "{model}: {body}");
        assert_eq!(body["model"], model);
    }
    std::fs::remove_dir_all(dir).unwrap();
}

/// A `--model-dir` server serves its directory: a request that names a listed
/// model loads it, so nothing has to be loaded through `POST /v1/models/load`
/// before it is used.
#[tokio::test]
async fn a_request_loads_the_model_it_names() {
    let dir = common::model_dir("controller-select");
    for sub in ["alpha", "beta"] {
        std::fs::create_dir_all(dir.join(sub)).unwrap();
        common::write_tiny_llama_gguf(&dir.join(sub).join("model.gguf"));
        std::fs::copy(
            dir.join("tokenizer.json"),
            dir.join(sub).join("tokenizer.json"),
        )
        .unwrap();
    }
    let app = create_router(state(ModelRegistry::default(), Some(manager(&dir))));
    let listed = |body: &serde_json::Value| -> Vec<serde_json::Value> {
        body["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| serde_json::json!([m["id"], m["joshua"]["loaded"]]))
            .collect()
    };

    // Both files share a stem, so each is listed by its path, unloaded.
    let (_, body) = call(&app, "GET", "/v1/models", serde_json::Value::Null).await;
    assert_eq!(
        listed(&body),
        [
            serde_json::json!(["alpha/model.gguf", false]),
            serde_json::json!(["beta/model.gguf", false]),
        ]
    );

    // The request names one by its listed id: it loads and answers.
    let (status, body) = call(
        &app,
        "POST",
        "/v1/completions",
        completion("beta/model.gguf"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["model"], "beta/model.gguf");
    let (_, body) = call(&app, "GET", "/v1/models", serde_json::Value::Null).await;
    assert_eq!(
        listed(&body),
        [
            serde_json::json!(["beta/model.gguf", true]),
            serde_json::json!(["alpha/model.gguf", false]),
        ]
    );

    // The next request reuses the loaded model, and the other file is still
    // available (and untouched) under its own id.
    let (status, body) = call(
        &app,
        "POST",
        "/v1/completions",
        completion("beta/model.gguf"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, card) = call(
        &app,
        "GET",
        "/v1/models/alpha/model.gguf",
        serde_json::Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{card}");
    assert_eq!(card["joshua"]["loaded"], false, "{card}");

    // Unloading hands the file back to the inventory; naming it loads it again.
    let (status, _) = call(
        &app,
        "POST",
        "/v1/models/unload",
        serde_json::json!({"id": "beta/model.gguf"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (_, body) = call(&app, "GET", "/v1/models", serde_json::Value::Null).await;
    assert!(
        listed(&body).iter().all(|entry| entry[1] == false),
        "{body}"
    );
    let (status, body) = call(
        &app,
        "POST",
        "/v1/completions",
        completion("beta/model.gguf"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["model"], "beta/model.gguf");
    std::fs::remove_dir_all(dir).unwrap();
}

/// A burst of requests for one cold model loads it once: they share the load
/// rather than one of them failing as a duplicate.
#[tokio::test]
async fn concurrent_requests_share_one_load_of_a_cold_model() {
    let dir = common::model_dir("controller-select-race");
    common::write_tiny_llama_gguf(&dir.join("tiny.gguf"));
    let app = create_router(state(ModelRegistry::default(), Some(manager(&dir))));

    // Two different routes name the same unloaded model at once.
    let (completion_res, chat_res) = tokio::join!(
        call(&app, "POST", "/v1/completions", completion("tiny")),
        call(
            &app,
            "POST",
            "/api/chat",
            serde_json::json!({
                "model": "tiny",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": false,
            }),
        ),
    );
    assert_eq!(completion_res.0, StatusCode::OK, "{}", completion_res.1);
    assert_eq!(chat_res.0, StatusCode::OK, "{}", chat_res.1);

    // One entry, loaded once.
    let (_, body) = call(&app, "GET", "/v1/models", serde_json::Value::Null).await;
    assert_eq!(body["data"].as_array().unwrap().len(), 1, "{body}");
    assert_eq!(body["data"][0]["id"], "tiny", "{body}");
    assert_eq!(body["data"][0]["joshua"]["loaded"], true, "{body}");
    std::fs::remove_dir_all(dir).unwrap();
}

/// A GGUF that is not a model on its own is not loaded by naming it: the
/// request keeps the answer `/v1/models` implies (that id is no model there),
/// rather than starting an engine on an adapter.
#[tokio::test]
async fn a_gguf_that_is_not_a_model_is_not_loaded_by_naming_it() {
    use candle_core::quantized::gguf_file::Value;
    let dir = common::model_dir("controller-select-adapter");
    common::write_tiny_llama_gguf(&dir.join("base.gguf"));
    let adapter = [(
        "general.type".to_string(),
        Value::String("adapter".to_string()),
    )];
    common::write_tiny_gguf_with(&dir.join("lora.gguf"), "llama", &adapter);
    let app = create_router(state(ModelRegistry::default(), Some(manager(&dir))));

    // The adapter is not listed as a model, so it is not a name to select.
    let (_, body) = call(&app, "GET", "/v1/models", serde_json::Value::Null).await;
    let ids: Vec<_> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].clone())
        .collect();
    assert_eq!(ids, ["base"], "{body}");

    // Load the model the directory does list, then name the adapter: the
    // answer stays the unknown-model one and no second engine starts.
    let (status, body) = call(&app, "POST", "/v1/completions", completion("base")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, body) = call(&app, "POST", "/v1/completions", completion("lora")).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"]["code"], "model_not_found", "{body}");
    let (_, body) = call(&app, "GET", "/v1/models", serde_json::Value::Null).await;
    let loaded: Vec<_> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["joshua"]["loaded"] == true)
        .map(|m| m["id"].clone())
        .collect();
    assert_eq!(loaded, ["base"], "{body}");
    std::fs::remove_dir_all(dir).unwrap();
}

/// The model `--model DIR` starts on comes from the inventory's own discovery:
/// a directory whose models are all in subdirectories starts, and one with no
/// model at all is reported.
#[test]
fn the_first_model_in_a_directory_is_a_loadable_one() {
    use candle_core::quantized::gguf_file::Value;
    let dir = common::model_dir("controller-first-model");
    let kind = |k: &str| {
        [(
            "general.type".to_string(),
            Value::String(k.to_string()),
        )]
    };
    // Neither a model on its own, both before the model in path order.
    common::write_tiny_gguf_with(&dir.join("a-adapter.gguf"), "llama", &kind("adapter"));
    std::fs::write(dir.join("b-broken.gguf"), b"not a gguf").unwrap();
    std::fs::create_dir_all(dir.join("nested")).unwrap();
    common::write_tiny_llama_gguf(&dir.join("nested").join("model.gguf"));

    assert_eq!(
        joshua::model_info::first_model(&dir),
        Some(dir.join("nested").join("model.gguf"))
    );

    // A directory of anything-but-models has nothing to start on.
    let none = common::model_dir("controller-first-model-none");
    common::write_tiny_gguf_with(&none.join("a.gguf"), "llama", &kind("adapter"));
    common::write_tiny_gguf_with(&none.join("b.gguf"), "llama", &kind("imatrix"));
    assert_eq!(joshua::model_info::first_model(&none), None);
    std::fs::remove_dir_all(dir).unwrap();
    std::fs::remove_dir_all(none).unwrap();
}

/// Ollama's own clients: the `HEAD /` heartbeat, `name:tag` names, RFC 3339
/// times and digests in the listings, the done object's counts and
/// durations, load/unload through `keep_alive`, `format`, the legacy
/// embeddings shape, and `{"error": "..."}` errors.
#[tokio::test]
async fn ollama_clients_get_ollama_shaped_answers() {
    let dir = common::model_dir("controller-ollama");
    common::write_tiny_llama_gguf(&dir.join("tiny.gguf"));
    common::write_tiny_llama_gguf(&dir.join("other.gguf"));
    let app = create_router(state(ModelRegistry::default(), Some(manager(&dir))));

    for method in ["HEAD", "GET"] {
        let req = Request::builder().method(method).uri("/").body(Body::empty()).unwrap();
        let res = app.clone().oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK, "{method} /");
    }

    let rfc3339 = |v: &serde_json::Value| {
        let s = v.as_str().unwrap_or_default();
        // 2026-11-06T07:48:02.123456789Z
        s.len() >= 20 && s.as_bytes()[4] == b'-' && s.as_bytes()[10] == b'T' && s.ends_with('Z')
    };
    let (_, tags) = call(&app, "GET", "/api/tags", serde_json::Value::Null).await;
    for model in tags["models"].as_array().unwrap() {
        let digest = model["digest"].as_str().unwrap();
        assert_eq!(digest.len(), 64, "{model}");
        assert!(digest.bytes().all(|b| b.is_ascii_hexdigit()), "{model}");
        assert!(rfc3339(&model["modified_at"]), "{model}");
        assert!(!model["details"]["parameter_size"].as_str().unwrap().is_empty(), "{model}");
    }

    // A request with no messages only loads the model, by its tagged name.
    let (status, body) = call(
        &app,
        "POST",
        "/api/chat",
        serde_json::json!({"model": "tiny:latest", "messages": []}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["done_reason"], "load", "{body}");
    assert_eq!(body["model"], "tiny", "{body}");
    let (_, ps) = call(&app, "GET", "/api/ps", serde_json::Value::Null).await;
    assert_eq!(ps["models"][0]["name"], "tiny:latest", "{ps}");
    assert_eq!(ps["models"][0]["context_length"], 64, "{ps}");
    assert_eq!(ps["models"][0]["size_vram"], 0, "{ps}");
    assert!(rfc3339(&ps["models"][0]["expires_at"]), "{ps}");

    // A whole answer carries the counts and durations.
    let messages = serde_json::json!([{"role": "user", "content": "hi"}]);
    let (status, body) = call(
        &app,
        "POST",
        "/api/chat",
        serde_json::json!({
            "model": "tiny:latest", "messages": messages, "stream": false,
            "options": {"num_predict": 3, "temperature": 0},
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["done"], true, "{body}");
    assert!(rfc3339(&body["created_at"]), "{body}");
    assert!(body["eval_count"].as_u64().unwrap() <= 3, "{body}");
    assert!(body["prompt_eval_count"].as_u64().unwrap() > 0, "{body}");
    for key in ["total_duration", "load_duration", "prompt_eval_duration", "eval_duration"] {
        assert!(body[key].is_u64(), "{key}: {body}");
    }
    assert!(["stop", "length"].contains(&body["done_reason"].as_str().unwrap()), "{body}");

    // Streaming is newline-delimited JSON ending in the done object.
    let req = Request::builder()
        .method("POST")
        .uri("/api/generate")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::json!({
                "model": "tiny", "prompt": "hello", "options": {"num_predict": 3},
            })
            .to_string(),
        ))
        .unwrap();
    let res = app.clone().oneshot(req).await.unwrap();
    assert_eq!(res.headers()[header::CONTENT_TYPE], "application/x-ndjson");
    let bytes = axum::body::to_bytes(res.into_body(), 1 << 20).await.unwrap();
    let lines: Vec<serde_json::Value> = std::str::from_utf8(&bytes)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let last = lines.last().unwrap();
    assert_eq!(last["done"], true, "{last}");
    assert!(last["eval_duration"].is_u64(), "{last}");
    assert!(lines[..lines.len() - 1].iter().all(|l| l["done"] == false && l["response"].is_string()));

    // `format` takes "json" or a schema, and nothing else.
    let (status, body) = call(
        &app,
        "POST",
        "/api/chat",
        serde_json::json!({
            "model": "tiny", "messages": messages, "stream": false, "format": "json",
            "options": {"num_predict": 2},
        }),
    )
    .await;
    // The tiny fixture's vocabulary cannot spell JSON, so the grammar
    // refuses it; reaching the grammar is what this checks.
    assert!(
        status == StatusCode::OK || body["error"].as_str().unwrap().contains("JSON schema"),
        "{body}"
    );
    let (status, body) = call(
        &app,
        "POST",
        "/api/chat",
        serde_json::json!({"model": "tiny", "messages": messages, "stream": false, "format": 7}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["error"].as_str().unwrap().contains("format"), "{body}");

    // `/api/show` describes the loaded model from its header and engine.
    let (status, show) = call(
        &app,
        "POST",
        "/api/show",
        serde_json::json!({"model": "tiny:latest"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{show}");
    assert_eq!(show["details"]["family"], "llama", "{show}");
    assert_eq!(show["model_info"]["general.architecture"], "llama", "{show}");
    assert!(show["model_info"]["general.parameter_count"].as_u64().unwrap() > 0, "{show}");
    assert!(show["parameters"].as_str().unwrap().contains("temperature"), "{show}");
    let caps = show["capabilities"].as_array().unwrap();
    assert!(caps.contains(&serde_json::json!("completion")), "{show}");
    assert!(caps.contains(&serde_json::json!("embedding")), "{show}");
    // An unloaded one, from its header alone.
    let (status, show) = call(
        &app,
        "POST",
        "/api/show",
        serde_json::json!({"name": "other"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{show}");
    assert_eq!(show["details"]["family"], "llama", "{show}");

    // The legacy embeddings route answers with one `embedding`.
    let (status, body) = call(
        &app,
        "POST",
        "/api/embeddings",
        serde_json::json!({"model": "tiny", "prompt": "hello"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(!body["embedding"].as_array().unwrap().is_empty(), "{body}");
    let (status, body) = call(
        &app,
        "POST",
        "/api/embed",
        serde_json::json!({"model": "tiny", "input": ["a", "b"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["embeddings"].as_array().unwrap().len(), 2, "{body}");

    // `keep_alive: 0` with nothing to say unloads it.
    let (status, body) = call(
        &app,
        "POST",
        "/api/generate",
        serde_json::json!({"model": "tiny:latest", "keep_alive": 0}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["done_reason"], "unload", "{body}");
    let (_, ps) = call(&app, "GET", "/api/ps", serde_json::Value::Null).await;
    assert!(ps["models"].as_array().unwrap().is_empty(), "{ps}");

    // ... and after an answer, once it is sent.
    let (status, body) = call(
        &app,
        "POST",
        "/api/chat",
        serde_json::json!({
            "model": "tiny", "messages": messages, "stream": false, "keep_alive": "0s",
            "options": {"num_predict": 1},
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (_, ps) = call(&app, "GET", "/api/ps", serde_json::Value::Null).await;
    assert!(ps["models"].as_array().unwrap().is_empty(), "{ps}");

    // ... and when the request fails after loading it.
    let (status, _) = call(
        &app,
        "POST",
        "/api/chat",
        serde_json::json!({
            "model": "tiny", "messages": messages, "stream": false, "keep_alive": 0,
            "format": "xml",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (_, ps) = call(&app, "GET", "/api/ps", serde_json::Value::Null).await;
    assert!(ps["models"].as_array().unwrap().is_empty(), "{ps}");

    // A name generation would resolve to the one loaded model unloads that
    // model, not nothing.
    let (status, _) = call(
        &app,
        "POST",
        "/api/generate",
        serde_json::json!({"model": "tiny"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, body) = call(
        &app,
        "POST",
        "/api/generate",
        serde_json::json!({"model": "an-alias", "keep_alive": 0}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["model"], "tiny", "{body}");
    let (_, ps) = call(&app, "GET", "/api/ps", serde_json::Value::Null).await;
    assert!(ps["models"].as_array().unwrap().is_empty(), "{ps}");

    // Errors are `{"error": "<message>"}`: an unknown model, and a body that
    // does not parse.
    let (status, body) = call(
        &app,
        "POST",
        "/api/chat",
        serde_json::json!({"model": "missing", "messages": messages, "stream": false}),
    )
    .await;
    assert!(status.is_client_error() || status.is_server_error(), "{status}");
    assert!(body["error"].is_string(), "{body}");
    let (status, body) = call(&app, "POST", "/api/show", serde_json::json!({"model": 7})).await;
    assert!(status.is_client_error(), "{status}");
    assert!(body["error"].is_string(), "{body}");
    // The OpenAI routes keep their envelope.
    let (_, body) = call(
        &app,
        "POST",
        "/v1/chat/completions",
        serde_json::json!({"model": "missing", "messages": messages}),
    )
    .await;
    assert!(body["error"]["message"].is_string(), "{body}");
    std::fs::remove_dir_all(dir).unwrap();
}
