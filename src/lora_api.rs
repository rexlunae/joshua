//! llama-server's LoRA routes: `GET /lora-adapters` lists the adapters a
//! model has loaded (see [`crate::lora_registry`]) with their server-wide
//! scales, and `POST /lora-adapters` sets those scales.  Both act on the
//! model `?model=` names, or the default one.

use axum::{
    extract::{Query, State},
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
