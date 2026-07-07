//! Wire protocol between the engine and the `joshua-npu-shim` subprocess.
//!
//! Control messages are newline-delimited JSON on the child's stdin/stdout —
//! one request line in, one response line out, strictly serialised (no
//! pipelining).  Bulk data never rides the pipe: token IDs and logits live
//! in a shared memory-mapped file that both processes map read-write.
//!
//! # Shared-memory layout
//!
//! The host creates a temp file of `n_ctx * 4 + SHM_LOGITS_CAPACITY` bytes:
//!
//! ```text
//! [0 .. n_ctx*4)                 u32-LE token IDs for the current request
//! [n_ctx*4 .. n_ctx*4 + cap)    f32-LE logits written by the shim
//! ```
//!
//! Access is strictly request/response: the host writes tokens before
//! sending `forward`, the shim writes logits before replying, so the two
//! processes never touch the region concurrently.

use std::path::PathBuf;

use base64::Engine as _;
use serde::{Deserialize, Serialize};

/// Fixed capacity reserved for the logit region (supports vocabularies up to
/// one million entries — comfortably above any current model).
pub const SHM_LOGITS_CAPACITY: usize = 4 * 1024 * 1024;

/// Requests sent host → shim.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Request {
    /// First message: load the vendor plugin and initialise a session.
    Init {
        /// Path of the vendor plugin cdylib to `dlopen`.
        library: PathBuf,
        /// Path of the model (GGUF) to initialise for.
        model: PathBuf,
        /// Context window in tokens (also sizes the token region).
        n_ctx: u32,
        /// Path of the shared-memory file to map.
        shm: PathBuf,
    },
    /// Feed `n_tokens` tokens (already written to the token region) at
    /// absolute position `pos`; write last-token logits to the logit region.
    Forward { pos: u32, n_tokens: u32 },
    /// Tokenise-and-prefill a multimodal prompt (optional plugin capability).
    ///
    /// `images` are base64-encoded media items in `<__media__>` marker
    /// order.  They ride the control pipe rather than the shared region
    /// because they are one-shot per request and unbounded in size.  Logits
    /// land in the logit region as usual.
    MediaPrefill { prompt: String, images: Vec<String> },
    /// Clear the vendor session state.
    Reset,
    /// Clean shutdown.
    Shutdown,
}

/// Responses sent shim → host.
#[derive(Debug, Serialize, Deserialize)]
pub struct Response {
    /// Whether the request succeeded.
    pub ok: bool,
    /// Error description when `ok` is false.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Vocabulary size, present on a successful `init` reply.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocab: Option<u32>,
    /// Whether the plugin supports multimodal prefill (`init` reply).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media: Option<bool>,
    /// Positions consumed, present on a successful `media_prefill` reply.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_past: Option<u32>,
}

impl Response {
    /// A success response.
    pub fn ok() -> Self {
        Self {
            ok: true,
            error: None,
            vocab: None,
            media: None,
            n_past: None,
        }
    }

    /// An error response.
    pub fn err(message: impl Into<String>) -> Self {
        Self {
            ok: false,
            error: Some(message.into()),
            vocab: None,
            media: None,
            n_past: None,
        }
    }
}

/// Base64-encode media bytes for the control pipe.
pub fn b64_encode(data: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(data)
}

/// Decode base64 media bytes from the control pipe.
pub fn b64_decode(s: &str) -> std::result::Result<Vec<u8>, String> {
    base64::engine::general_purpose::STANDARD
        .decode(s)
        .map_err(|e| e.to_string())
}
