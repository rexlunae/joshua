//! llama-server's LoRA routes: `GET /lora-adapters` lists the adapters a
//! model has loaded (see [`crate::lora_registry`]) with their server-wide
//! scales, and `POST /lora-adapters` sets those scales.  Both act on the
//! model `?model=` names, or the default one.
//!
//! vLLM's `POST /v1/load_lora_adapter` and `POST /v1/unload_lora_adapter`
//! load and unload adapters at runtime (with `--allow-runtime-lora`); a
//! loaded adapter is listed by `/v1/models` and selected by naming it as a
//! request's `model` (see [`crate::server::select_generation`]).

use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    extract::{Query, State},
    http::StatusCode,
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::server::{select_model, ApiError, AppState};
use crate::types::LoraScale;

#[derive(Debug, Default, Deserialize)]
pub struct Target {
    #[serde(default)]
    model: Option<String>,
}

/// `GET /lora-adapters`: `[{"id", "path", "scale", ...}]`, by id.
pub async fn list(
    State(state): State<AppState>,
    Query(target): Query<Target>,
) -> Result<Json<Value>, ApiError> {
    let (_, engine) = select_model(&state, target.model.as_deref().unwrap_or_default()).await?;
    Ok(Json(json!(engine.lora_adapters())))
}

/// `POST /lora-adapters` with `[{"id": 0, "scale": 0.5}]`: set the
/// server-wide scales, every adapter left out to 0.
pub async fn set(
    State(state): State<AppState>,
    Query(target): Query<Target>,
    Json(scales): Json<Vec<LoraScale>>,
) -> Result<Json<Value>, ApiError> {
    let (_, engine) = select_model(&state, target.model.as_deref().unwrap_or_default()).await?;
    engine.set_lora_scales(&scales).map_err(ApiError::from)?;
    Ok(Json(json!({ "success": true })))
}

/// Serialises runtime adapter loads (see [`vllm_load`]).
pub(crate) static RUNTIME_LOADS: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// vLLM's `POST /v1/load_lora_adapter` body.
#[derive(Debug, Deserialize)]
pub struct VllmLoad {
    #[serde(default)]
    lora_name: String,
    #[serde(default)]
    lora_path: String,
    /// Replace an adapter already loaded under this name.
    #[serde(default)]
    load_inplace: bool,
    /// The base model to load it for (joshua's: vLLM serves one).
    #[serde(default)]
    model: Option<String>,
}

/// vLLM's `POST /v1/unload_lora_adapter` body.
#[derive(Debug, Deserialize)]
pub struct VllmUnload {
    #[serde(default)]
    lora_name: String,
    #[serde(default)]
    model: Option<String>,
}

/// The routes are refused unless the server was started with
/// `--allow-runtime-lora`; vLLM leaves them out unless
/// `VLLM_ALLOW_RUNTIME_LORA_UPDATING` is set.
fn runtime_updates(state: &AppState) -> Result<(), ApiError> {
    if state.allow_runtime_lora {
        return Ok(());
    }
    Err(ApiError::new(
        StatusCode::NOT_FOUND,
        "runtime LoRA updates are off; start the server with --allow-runtime-lora",
        "NotFoundError",
    ))
}

/// `POST /v1/load_lora_adapter` with `{"lora_name", "lora_path"}`: load a
/// GGUF adapter, which requests then select by naming it as their `model`
/// (vLLM's dynamic LoRA serving).  It loads at server-wide scale 0, so a
/// request naming the base model is not changed by it.
pub async fn vllm_load(
    State(state): State<AppState>,
    Json(req): Json<VllmLoad>,
) -> Result<String, ApiError> {
    runtime_updates(&state)?;
    let name = req.lora_name.trim().to_string();
    if name.is_empty() || req.lora_path.trim().is_empty() {
        return Err(ApiError::bad_request(
            "Both 'lora_name' and 'lora_path' must be provided.",
        ));
    }
    // Adapter names are model names: unique across the loaded models'
    // adapters and never a model's id.  Runtime loads are serialised so
    // two of one name cannot both pass the check.
    let _serial = RUNTIME_LOADS.lock().await;
    let (_, engine) = select_model(&state, req.model.as_deref().unwrap_or_default()).await?;
    if crate::server::names_a_model(&state, &name).await? {
        return Err(ApiError::bad_request(format!(
            "'{name}' is a model's id; an adapter needs a name of its own"
        )));
    }
    // An adapter model Ollama's create made owns its name too, loaded or
    // not: a runtime adapter of that name would run in its place.
    if state.adapter_model(&name).is_some() {
        return Err(ApiError::bad_request(format!(
            "'{name}' is an adapter model; an adapter needs a name of its own"
        )));
    }
    let replace = match state.models.adapter(&name) {
        Some((owner, _)) if !Arc::ptr_eq(&owner, &engine) => {
            return Err(ApiError::bad_request(format!(
                "The lora adapter '{name}' has already been loaded for another model."
            )))
        }
        Some(_) if !req.load_inplace => {
            return Err(ApiError::bad_request(format!(
                "The lora adapter '{name}' has already been loaded."
            )))
        }
        Some(_) => true,
        None => false,
    };
    let path = PathBuf::from(req.lora_path.trim());
    if let Err(e) = std::fs::metadata(&path) {
        return Err(ApiError::bad_request(format!(
            "cannot read the LoRA adapter at {}: {e}",
            path.display()
        )));
    }
    let loaded = tokio::task::spawn_blocking({
        let name = name.clone();
        move || {
            // A replacement is read and validated before the adapter it
            // replaces is let go, so a bad file leaves that one in place.
            if replace {
                engine.replace_lora(&name, &path)
            } else {
                engine.load_lora(&path, Some(&name), 0.0)
            }
        }
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?;
    // A file that is not a usable adapter is the request's fault (vLLM's
    // 400), not the server's.
    loaded.map_err(|e| match e {
        crate::JoshuaError::ModelLoad(msg) => ApiError::bad_request(msg),
        e => ApiError::from(e),
    })?;
    Ok(format!("Success: LoRA adapter '{name}' added successfully."))
}

/// `POST /v1/unload_lora_adapter` with `{"lora_name"}`.
pub async fn vllm_unload(
    State(state): State<AppState>,
    Json(req): Json<VllmUnload>,
) -> Result<String, ApiError> {
    runtime_updates(&state)?;
    let name = req.lora_name.trim();
    let (_, engine) = select_model(&state, req.model.as_deref().unwrap_or_default()).await?;
    if engine.lora_id(name).is_none() {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            format!("The lora adapter '{name}' cannot be found."),
            "NotFoundError",
        ));
    }
    engine.unload_lora(name).map_err(ApiError::from)?;
    Ok(format!("Success: LoRA adapter '{name}' removed successfully."))
}
