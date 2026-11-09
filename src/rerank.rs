//! Reranking with a generative reranker: `POST /v1/rerank` (also `/rerank`,
//! `/v2/rerank`, `/reranking` and `/v1/reranking`, the vLLM, llama-server,
//! Cohere and Jina spellings) and vLLM's `POST /v1/score` (also `/score`).
//!
//! Each query/document pair is judged by [`Engine::rerank`]: the model is
//! asked, through its chat template, whether the document meets the query,
//! and the score is the chance it says `yes` rather than `no`.  That is how
//! Qwen3-Reranker and its fine-tunes are meant to be run; a model without
//! `yes` and `no` tokens is refused.
//!
//! [`Engine::rerank`]: crate::engine::Engine::rerank

use std::sync::Arc;

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use uuid::Uuid;

use crate::engine::Engine;
use crate::server::{check_embedding_inputs, select_model, ApiError, AppState, CancelOnDrop};
use crate::types::{unix_time, GenerationOptions, UsageInfo};

/// A document to rerank: its text, or an object carrying it (Cohere's and
/// Jina's `{"text": ...}`).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Document {
    Text(String),
    Object { text: String },
}

impl Document {
    fn text(&self) -> &str {
        match self {
            Document::Text(text) | Document::Object { text } => text,
        }
    }
}

/// A rerank request, in the fields vLLM, llama-server, Cohere and Jina share.
#[derive(Debug, Deserialize)]
pub struct RerankRequest {
    #[serde(default)]
    pub model: String,
    pub query: String,
    /// The documents to rank (`texts` is llama-server's other name).
    #[serde(alias = "texts")]
    pub documents: Vec<Document>,
    /// How many of the best documents to return; all of them without it.
    #[serde(default)]
    pub top_n: Option<usize>,
    /// Whether each result repeats its document's text (default on).
    #[serde(default)]
    pub return_documents: Option<bool>,
    /// What counts as relevant, in the reranker's `<Instruct>` slot.
    #[serde(default)]
    pub instruction: Option<String>,
}

/// One side of a score request: a text, or a list of them.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Texts {
    One(String),
    Many(Vec<String>),
}

impl Texts {
    fn into_vec(self) -> Vec<String> {
        match self {
            Texts::One(text) => vec![text],
            Texts::Many(texts) => texts,
        }
    }
}

/// vLLM's score request: `text_1` against each `text_2`, or pair by pair
/// when both are lists of the same length.
#[derive(Debug, Deserialize)]
pub struct ScoreRequest {
    #[serde(default)]
    pub model: String,
    pub text_1: Texts,
    pub text_2: Texts,
    #[serde(default)]
    pub instruction: Option<String>,
}

/// `POST /v1/rerank` and its other spellings.
pub async fn rerank(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<RerankRequest>,
) -> Result<Json<Value>, ApiError> {
    if req.documents.is_empty() {
        return Err(ApiError::bad_request("`documents` must not be empty"));
    }
    if req.top_n == Some(0) {
        return Err(ApiError::bad_request("`top_n` must be at least 1"));
    }
    check_embedding_inputs(req.documents.len())?;
    let (model, engine) = select_model(&state, &req.model).await?;
    let documents: Arc<Vec<String>> = Arc::new(
        req.documents
            .iter()
            .map(|document| document.text().to_string())
            .collect(),
    );
    let (scores, tokens) = score_pairs(
        &state,
        &engine,
        &headers,
        vec![req.query.clone()],
        Arc::clone(&documents),
        req.instruction.clone(),
    )
    .await?;
    let mut ranked: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    ranked.truncate(req.top_n.unwrap_or(ranked.len()));
    let with_documents = req.return_documents.unwrap_or(true);
    let results: Vec<Value> = ranked
        .into_iter()
        .map(|(index, score)| {
            let mut result = json!({ "index": index, "relevance_score": score });
            if with_documents {
                result["document"] = json!({ "text": documents[index] });
            }
            result
        })
        .collect();
    Ok(Json(json!({
        "id": format!("rerank-{}", Uuid::new_v4().simple()),
        "model": model,
        "object": "list",
        "usage": { "prompt_tokens": tokens, "total_tokens": tokens },
        "results": results,
    })))
}

/// `POST /v1/score` and `POST /score`.
pub async fn score(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ScoreRequest>,
) -> Result<Json<Value>, ApiError> {
    let first = req.text_1.into_vec();
    let second = req.text_2.into_vec();
    if first.is_empty() || second.is_empty() {
        return Err(ApiError::bad_request(
            "`text_1` and `text_2` must not be empty",
        ));
    }
    if first.len() != 1 && first.len() != second.len() {
        return Err(ApiError::bad_request(format!(
            "`text_1` has {} texts and `text_2` {}: give one `text_1`, or as many as `text_2`",
            first.len(),
            second.len()
        )));
    }
    check_embedding_inputs(second.len())?;
    let (model, engine) = select_model(&state, &req.model).await?;
    let (scores, tokens) = score_pairs(
        &state,
        &engine,
        &headers,
        first,
        Arc::new(second),
        req.instruction,
    )
    .await?;
    let data: Vec<Value> = scores
        .into_iter()
        .enumerate()
        .map(|(index, score)| json!({ "index": index, "object": "score", "score": score }))
        .collect();
    Ok(Json(json!({
        "id": format!("score-{}", Uuid::new_v4().simple()),
        "object": "list",
        "created": unix_time(),
        "model": model,
        "data": data,
        "usage": { "prompt_tokens": tokens, "completion_tokens": 0, "total_tokens": tokens },
    })))
}

/// Run [`Engine::rerank`] off the async runtime on each document paired
/// with its query (the one query, or the query at the document's index),
/// stopping if the client goes away, and count the prompt tokens it read.
/// The pairs borrow the texts, so a long query shared by many documents is
/// held once.
async fn score_pairs(
    state: &AppState,
    engine: &Arc<Engine>,
    headers: &HeaderMap,
    queries: Vec<String>,
    documents: Arc<Vec<String>>,
    instruction: Option<String>,
) -> Result<(Vec<f32>, u32), ApiError> {
    let options = GenerationOptions {
        cache_scope: Some(crate::affinity::cache_scope(headers, None, None)),
        ..Default::default()
    };
    let cancel = CancelOnDrop::new();
    let (scores, tokens) = tokio::task::spawn_blocking({
        let engine = Arc::clone(engine);
        let flag = cancel.flag();
        move || {
            let pairs: Vec<(&str, &str)> = documents
                .iter()
                .enumerate()
                .map(|(i, document)| {
                    let query = if queries.len() == 1 {
                        &queries[0]
                    } else {
                        &queries[i]
                    };
                    (query.as_str(), document.as_str())
                })
                .collect();
            engine.rerank(&pairs, instruction.as_deref(), &options, Some(&flag))
        }
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?
    .map_err(ApiError::from)?;
    if scores.iter().any(|score| !score.is_finite()) {
        return Err(ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "the model gave a non-finite relevance score",
            "server_error",
        ));
    }
    state.metrics.record_generation(&UsageInfo {
        prompt_tokens: tokens,
        ..Default::default()
    });
    Ok((scores, tokens))
}
