//! Joshua — a pure-Rust LLM inference engine.
//!
//! Joshua provides a pure-Rust inference layer built on
//! [candle](https://github.com/huggingface/candle) (HuggingFace's native Rust
//! ML framework) plus an OpenAI-compatible HTTP server built on [`axum`].
//! No C or C++ runtime dependencies are required for CPU inference.
//!
//! # Quick start
//!
//! ```no_run
//! use joshua::{Engine, GenerationOptions};
//!
//! let engine = Engine::new("path/to/model.gguf").unwrap();
//!
//! let messages = vec![joshua::ChatMessage {
//!     role: "user".to_string(),
//!     content: "Hello!".to_string(),
//!     images: None,
//!     name: None,
//! }];
//!
//! let (text, usage, _, _) = engine.complete(&messages, &GenerationOptions::default()).unwrap();
//! println!("{text}");
//! ```
//!
//! A `tokenizer.json` from the model's HuggingFace repository must be placed
//! alongside the `.gguf` file so the engine can tokenise prompts.

pub mod engine;
pub mod error;
pub mod model;
pub mod server;
pub mod types;

pub use engine::Engine;
pub use error::{JoshuaError, Result};
pub use types::{
    ChatMessage, EmbeddingRequest, EmbeddingResponse, GenerationOptions, UsageInfo,
};
