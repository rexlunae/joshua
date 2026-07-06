//! NPU backend infrastructure.
//!
//! Vendor NPU runtimes (Qualcomm QNN/Hexagon, Huawei CANN, Core ML bridges,
//! …) are proprietary C/C++ stacks that cannot be part of Joshua's pure-Rust
//! core.  This module contains them behind three safety layers:
//!
//! 1. **Trait boundary** ([`NpuBackend`] / [`NpuSession`]): the engine only
//!    knows an object-safe token-in / logits-out interface.  The default
//!    build contains no vendor code, and generation transparently falls back
//!    to the candle CPU/GPU path when a backend is missing, fails to load,
//!    or starts erroring (a circuit breaker disables it after repeated
//!    failures).
//!
//! 2. **Plugin ABI, loaded at runtime** ([`vendor`]): vendor support ships as
//!    a shared library exporting the tiny `joshua_npu_*` C ABI below.  Joshua
//!    `dlopen`s it with `libloading` — nothing is linked at build time, the
//!    `unsafe` surface is confined to [`vendor`], and a missing library is a
//!    clean fallback, not a build or startup failure.
//!    [`InProcessBackend`] runs a plugin this way for minimum overhead —
//!    accepting that a buggy vendor runtime shares the process.
//!
//! 3. **Process isolation** ([`ShimBackend`]): the same plugin is loaded by
//!    the small `joshua-npu-shim` subprocess instead.  Control messages run
//!    over pipes (NDJSON), tensors over a shared memory-mapped file, and the
//!    host enforces timeouts and kills the child on any protocol violation.
//!    If the vendor runtime crashes or hangs, one request fails and the
//!    server keeps running; the next request starts a fresh shim.  This is
//!    the recommended mode for production.
//!
//! # The plugin ABI
//!
//! A vendor plugin is any `cdylib` exporting:
//!
//! ```c
//! int32_t joshua_npu_init(const char *model_path, uint32_t n_ctx,
//!                         uint32_t *out_vocab, void **out_handle);
//! int32_t joshua_npu_forward(void *handle, const uint32_t *tokens,
//!                            uint32_t n_tokens, uint32_t pos,
//!                            float *out_logits);
//! int32_t joshua_npu_reset(void *handle);
//! void    joshua_npu_free(void *handle);
//! ```
//!
//! Semantics: `init` loads/compiles whatever artifact the vendor needs for
//! `model_path` and reports the vocabulary size.  `forward` appends
//! `n_tokens` tokens starting at absolute position `pos` (which always
//! equals the number of tokens fed so far — a KV-cache contract) and writes
//! `out_vocab` logits for the last fed token.  `reset` clears the internal
//! state for reuse with an unrelated prompt.  All functions return `0` on
//! success.
//!
//! `crates/joshua-mock-npu` is a pure-Rust reference plugin used by the test
//! suite; real vendor plugins wrap their SDK behind the same four symbols.

mod proto;
mod shim;
mod vendor;

pub use shim::ShimBackend;
pub use vendor::InProcessBackend;

// Internals shared with the `joshua-npu-shim` binary.
#[doc(hidden)]
pub mod internal {
    pub use super::proto::{Request, Response, SHM_LOGITS_CAPACITY};
    pub use super::vendor::VendorLibrary;
}

use std::path::Path;

/// A source of NPU-backed generation sessions.
///
/// Implementations must be cheap to share (`Send + Sync`); session creation
/// may be expensive (model compilation, subprocess start).
pub trait NpuBackend: Send + Sync {
    /// Human-readable backend name for logs.
    fn name(&self) -> String;

    /// Create a session prepared to generate for the model at `model_path`.
    ///
    /// `model_path` is the GGUF the engine was opened with; a vendor plugin
    /// may derive its own artifact path from it (e.g. a sibling compiled
    /// context binary).
    fn create_session(
        &self,
        model_path: &Path,
        n_ctx: u32,
    ) -> std::result::Result<Box<dyn NpuSession>, String>;
}

/// A stateful generation session on an NPU.
///
/// Mirrors the KV-cache contract of the candle models: `forward(tokens, pos)`
/// appends `tokens` at absolute position `pos` (== number of tokens fed so
/// far) and returns the logits for the last token.
pub trait NpuSession: Send {
    /// Vocabulary size (length of the logit vectors).
    fn vocab_size(&self) -> usize;

    /// Feed `tokens` at position `pos`, returning last-token logits.
    fn forward(&mut self, tokens: &[u32], pos: usize) -> std::result::Result<Vec<f32>, String>;

    /// Clear internal state so the session can serve an unrelated prompt.
    ///
    /// Returns `false` if the session cannot be safely reset (e.g. its shim
    /// process died) — the caller must discard it.
    fn reset(&mut self) -> bool;
}
