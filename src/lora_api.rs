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
    let (_, engine) = select_model(&state, req.model.as_deref().unwrap_or_default()).await?;
    if state.models.list().iter().any(|m| m.id == name) {
        return Err(ApiError::bad_request(format!(
            "'{name}' is a model's id; an adapter needs a name of its own"
        )));
    }
    if engine.lora_id(&name).is_some() && !req.load_inplace {
        return Err(ApiError::bad_request(format!(
            "The lora adapter '{name}' has already been loaded."
        )));
    }
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
            if req.load_inplace && engine.lora_id(&name).is_some() {
                engine.unload_lora(&name)?;
            }
            engine.load_lora(&path, Some(&name), 0.0)
        }
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?;
    loaded.map_err(ApiError::from)?;
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
