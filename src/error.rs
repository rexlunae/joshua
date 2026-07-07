//! Error types for the Joshua LLM inference engine.

use thiserror::Error;

/// Primary error type for Joshua operations.
#[derive(Error, Debug)]
pub enum JoshuaError {
    /// The model file could not be loaded.
    #[error("Failed to load model: {0}")]
    ModelLoad(String),

    /// The inference context could not be created.
    #[error("Failed to create inference context: {0}")]
    ContextCreation(String),

    /// An error occurred during tokenization.
    #[error("Tokenization failed: {0}")]
    Tokenization(String),

    /// An error occurred during inference decoding.
    #[error("Inference failed: {0}")]
    Inference(String),

    /// The input prompt was too long for the context window.
    #[error("Prompt too long: {0} tokens (context window: {1})")]
    PromptTooLong(usize, usize),

    /// The request was malformed or missing required fields.
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// The engine is at its concurrency limit and cannot start more work.
    #[error("Server overloaded: {0}")]
    Overloaded(String),

    /// An I/O error occurred.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// A JSON serialisation/deserialisation error occurred.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// An unexpected internal error.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Convenience alias for `Result<T, JoshuaError>`.
pub type Result<T, E = JoshuaError> = std::result::Result<T, E>;
