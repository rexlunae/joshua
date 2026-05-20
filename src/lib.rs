//! Joshua — an mmap-based LLM inference engine for the Rust ecosystem.
//!
//! Joshua provides a pure-Rust API layer over
//! [llama.cpp](https://github.com/ggerganov/llama.cpp) (via
//! [`llama_cpp_2`]) plus an OpenAI-compatible HTTP server built on
//! [`axum`].
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

pub mod engine;
pub mod error;
pub mod server;
pub mod types;

pub use engine::Engine;
pub use error::{JoshuaError, Result};
pub use types::{
    ChatMessage, EmbeddingRequest, EmbeddingResponse, GenerationOptions, UsageInfo,
};
