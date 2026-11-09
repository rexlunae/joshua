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
//! let messages = vec![joshua::ChatMessage::text("user", "Hello!")];
//!
//! let (text, usage, _, _) = engine.complete(&messages, &GenerationOptions::default()).unwrap();
//! println!("{text}");
//! ```
//!
//! A `tokenizer.json` from the model's HuggingFace repository must be placed
//! alongside the `.gguf` file so the engine can tokenise prompts.

mod activation_trace;
mod precise;
pub mod anthropic;
pub mod attention;
pub mod auto_placement;
pub mod compression;
pub mod completions;
pub mod llama_server;
pub(crate) mod logprobs;
pub mod coordinator;
pub mod cpu_pinning;
pub mod profile;
pub mod draft_model;
pub mod distributed;
pub mod embedding;
pub mod fim;
pub mod constraints;
pub mod conv1d;
pub mod engine;
pub mod error;
pub mod gguf_ext;
pub mod gguf_meta;
pub mod hot_experts;
pub mod iq2xxs;
pub mod kquant_dot;
pub mod kv_int8;
pub mod kimi_k3;
pub mod low_bit;
pub mod lora;
pub mod lora_peft;
pub mod lora_registry;
pub mod lora_api;
pub mod mapping;
pub mod megakernel;
pub mod mmap_tensor;
pub mod mhc;
pub mod moe;
pub mod mxfp4;
pub mod nvfp4_gemv;
pub mod iquants;
pub mod iq_grids;
pub mod native_session;
pub mod ngram;
pub mod model;
pub mod model_info;
pub mod ollama_pull;
pub mod quantized_deepseek2;
pub mod quantized_deepseek4;
pub mod quantized_nemotron;
pub mod quantized_qwen;
pub mod quant_matmul;
pub mod raw_block;
pub mod reasoning;
pub mod rerank;
pub mod responses;
pub mod cuda_io;
pub mod residency;
pub mod route_trace;
pub mod affinity;
pub mod sampling;
pub mod scan;
pub mod cache_sim;
pub mod npu;
pub mod ollama;
pub mod paged_weights;
pub mod placement;
pub mod ppl;
pub mod prefill_cost;
pub mod prefix_cache;
pub mod prefix_key;
pub mod prompt_stability;
pub mod stream_prefill;
pub mod server;
mod job_admission;
pub mod simd;
pub mod speculative;
pub mod template;
pub mod tensor_source;
pub mod token_embedding;
pub mod whisper;
pub mod yarn;
pub mod zero_copy_metal;
pub mod tools;
pub mod types;

/// Re-exported so consumers of the `tls` feature can configure rustls
/// (e.g. install a different [`rustls::crypto::CryptoProvider`]) without
/// adding their own dependency on a matching rustls version.
#[cfg(feature = "tls")]
pub use rustls;

pub use engine::{
    find_gguf_in_dir, Completion, ComputeBackend, Engine, EngineOptions, DensePlacement,
    EngramTier, ExpertPlacement, HugePages, MlockMode, MmapMode, PageSize, PromptPart, RawPrompt,
    SsdExpertCache, StopType, VramExpertCache,
};
pub use error::{JoshuaError, Result};
pub use sampling::{SamplingDefaults, SamplingOverrides, SamplingProfile};
pub use speculative::{SpeculativeConfig, SpeculativeStats};
pub use types::{
    ChatMessage, EmbeddingRequest, EmbeddingResponse, GenerationOptions, ReasoningConfig,
    ReasoningEffort, UsageInfo,
};
