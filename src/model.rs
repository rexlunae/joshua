//! Architecture-aware model dispatch for Joshua.
//!
//! Reads `general.architecture` from the GGUF metadata and routes to the
//! correct candle quantized model type.  This allows the same engine to load
//! Llama, Phi, Phi-3, Qwen2, and any other architecture that candle supports,
//! while giving clear errors for unsupported models like LFM2.

use std::io::{Read, Seek};

use candle_core::quantized::gguf_file;
use candle_core::{Device, Result, Tensor};
use candle_transformers::models::quantized_llama;
use candle_transformers::models::quantized_phi;
use candle_transformers::models::quantized_phi3;
use candle_transformers::models::quantized_qwen2;

// ─── Architecture enum ──────────────────────────────────────────────────────

/// Known GGUF model architectures that candle has quantized support for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    Llama,
    Phi,
    Phi3,
    Qwen2,
}

impl Architecture {
    /// Parse an architecture name from the GGUF `general.architecture` metadata.
    ///
    /// Returns `None` if the architecture is not recognised by candle's
    /// quantized loaders.
    pub fn from_gguf_metadata(metadata: &std::collections::HashMap<String, gguf_file::Value>) -> Option<Self> {
        let arch = metadata.get("general.architecture")?.to_string().ok()?;
        Some(match arch.as_str() {
            "llama" | "mistral" | "gemma" | "gemma2" | "mixtral" | "starcoder2" | "yi" => Self::Llama,
            "phi" | "phi2" => Self::Phi,
            "phi3" => Self::Phi3,
            "qwen2" => Self::Qwen2,
            other => {
                // Log the unknown architecture so users can see what was requested.
                tracing::warn!("Unsupported GGUF architecture '{other}' — no quantized candle loader available");
                return None;
            }
        })
    }

    /// Human-readable name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Llama => "Llama (also Mistral, Gemma, Mixtral, StarCoder2, Yi)",
            Self::Phi => "Phi / Phi-2",
            Self::Phi3 => "Phi-3",
            Self::Qwen2 => "Qwen2",
        }
    }
}

// ─── Dispatched model ───────────────────────────────────────────────────────

/// A quantized model loaded from a GGUF file, wrapping the correct candle
/// model type for the detected architecture.
///
/// This hides the concrete model type behind a uniform `forward()` API so the
/// engine can handle multiple architectures without code duplication.
pub enum QuantizedModel {
    Llama(quantized_llama::ModelWeights),
    Phi(quantized_phi::ModelWeights),
    Phi3(quantized_phi3::ModelWeights),
    Qwen2(quantized_qwen2::ModelWeights),
}

impl QuantizedModel {
    /// Load a model from a GGUF file, dispatching to the correct quantized
    /// loader based on `general.architecture`.
    pub fn from_gguf<R: Read + Seek>(
        gguf: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let arch = Architecture::from_gguf_metadata(&gguf.metadata).ok_or_else(|| {
            let known = Architecture::list_known();
            candle_core::Error::Msg(format!(
                "Unsupported GGUF architecture. \
                 Joshua supports: {known}. \
                 Read `general.architecture` from the GGUF metadata to check yours.",
            ))
        })?;

        tracing::info!("Detected model architecture: {}", arch.display_name());

        match arch {
            Architecture::Llama => {
                quantized_llama::ModelWeights::from_gguf(gguf, reader, device)
                    .map(Self::Llama)
            }
            Architecture::Phi => {
                quantized_phi::ModelWeights::from_gguf(gguf, reader, device)
                    .map(Self::Phi)
            }
            Architecture::Phi3 => {
                // CPU-only: flash attention not available.
                quantized_phi3::ModelWeights::from_gguf(false, gguf, reader, device)
                    .map(Self::Phi3)
            }
            Architecture::Qwen2 => {
                quantized_qwen2::ModelWeights::from_gguf(gguf, reader, device)
                    .map(Self::Qwen2)
            }
        }
    }

    /// Unified forward pass.
    ///
    /// `input` has shape `[1, seq_len]` for the initial prefill, or `[1, 1]`
    /// for single-token decode steps.  `index_pos` is the absolute position in
    /// the KV cache of the first token in `input`.
    pub fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
        match self {
            Self::Llama(m) => m.forward(input, index_pos),
            Self::Phi(m) => m.forward(input, index_pos),
            Self::Phi3(m) => m.forward(input, index_pos),
            Self::Qwen2(m) => m.forward(input, index_pos),
        }
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

impl Architecture {
    /// Return a comma-separated list of known architecture names for error
    /// messages.
    fn list_known() -> &'static str {
        "llama, mistral, gemma, gemma2, mixtral, starcoder2, yi, phi, phi2, phi3, qwen2"
    }
}
