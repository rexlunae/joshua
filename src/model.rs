//! Architecture-aware model dispatch for Joshua.
//!
//! Reads `general.architecture` from the GGUF metadata and routes to the
//! correct candle quantized model type.  This covers every GGUF architecture
//! that candle ships a pure-Rust quantized loader for:
//!
//! | `general.architecture`                          | candle loader
//! |-------------------------------------------------|---------------------------
//! | `llama` (Llama 1-3, Mistral, Mixtral, TinyLlama, SmolLM, Vicuna, Zephyr, Yi, …) | `quantized_llama`
//! | `gemma` / `gemma2` / `gemma3` / `gemma-embedding` | `quantized_gemma3`
//! | `glm4`                                          | `quantized_glm4`
//! | `lfm2`                                          | `quantized_lfm2`
//! | `phi2`                                          | `quantized_phi`
//! | `phi3`                                          | `quantized_phi3`
//! | `qwen2`                                         | `quantized_qwen2`
//! | `qwen3`                                         | `quantized_qwen3`
//! | `qwen3moe`                                      | `quantized_qwen3_moe`
//!
//! Every other architecture name in llama.cpp's registry is recognised and
//! reported with a clear "known but not yet loadable in pure Rust" error, so
//! users can tell the difference between an unsupported model and a corrupt
//! or mislabelled file.

use std::collections::HashMap;
use std::io::{Read, Seek};

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Result, Tensor};
use candle_transformers::models::{
    quantized_gemma3, quantized_glm4, quantized_lfm2, quantized_llama, quantized_phi,
    quantized_phi3, quantized_qwen2, quantized_qwen3, quantized_qwen3_moe,
};

// ─── Architecture enum ──────────────────────────────────────────────────────

/// GGUF model architectures with a pure-Rust quantized candle loader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// `llama` — Llama 1/2/3, Mistral, Mixtral, TinyLlama, SmolLM, Yi, and
    /// every other model that llama.cpp's converters emit as `llama`.
    Llama,
    /// `gemma`, `gemma2`, `gemma3`, `gemma-embedding`.
    Gemma,
    /// `glm4` — GLM-4 dense models.
    Glm4,
    /// `lfm2` — Liquid LFM2 hybrid (attention + short-conv) models.
    Lfm2,
    /// `phi2` — Phi-1, Phi-1.5, Phi-2.
    Phi2,
    /// `phi3` — Phi-3 / Phi-3.5.
    Phi3,
    /// `qwen2` — Qwen1.5 / Qwen2 / Qwen2.5 dense models.
    Qwen2,
    /// `qwen3` — Qwen3 dense models.
    Qwen3,
    /// `qwen3moe` — Qwen3 mixture-of-experts models.
    Qwen3Moe,
}

/// Architecture names understood by llama.cpp but without a pure-Rust
/// quantized loader in candle yet.  Kept in sync with llama.cpp's
/// `llama-arch.cpp` registry so we can give a precise error instead of a
/// generic "unknown architecture".
const KNOWN_UNSUPPORTED_ARCHS: &[&str] = &[
    "afmoe",
    "apertus",
    "arcee",
    "arctic",
    "arwkv7",
    "baichuan",
    "bailingmoe",
    "bailingmoe2",
    "bert",
    "bitnet",
    "bloom",
    "chameleon",
    "chatglm",
    "codeshell",
    "cogvlm",
    "cohere2",
    "command-r",
    "dbrx",
    "deci",
    "deepseek",
    "deepseek2",
    "dots1",
    "dream",
    "ernie4_5",
    "ernie4_5-moe",
    "exaone",
    "exaone4",
    "falcon",
    "falcon-h1",
    "gemma3n",
    "glm4moe",
    "gpt2",
    "gpt-oss",
    "gptj",
    "gptneox",
    "granite",
    "granitehybrid",
    "granitemoe",
    "grok",
    "grovemoe",
    "hunyuan-dense",
    "hunyuan-moe",
    "jais",
    "jamba",
    "jina-bert-v2",
    "llada",
    "llada-moe",
    "llama4",
    "lfm2moe",
    "mamba",
    "mamba2",
    "minicpm",
    "minicpm3",
    "minimax-m2",
    "mpt",
    "nemotron",
    "nemotron-h",
    "neo-bert",
    "nomic-bert",
    "nomic-bert-moe",
    "olmo",
    "olmo2",
    "olmoe",
    "openelm",
    "orion",
    "phimoe",
    "plamo",
    "plamo2",
    "plm",
    "qwen",
    "qwen2moe",
    "qwen2vl",
    "refact",
    "rwkv6",
    "rwkv6qwen2",
    "rwkv7",
    "seed-oss",
    "smallthinker",
    "smollm3",
    "stablelm",
    "starcoder",
    "starcoder2",
    "t5",
    "t5encoder",
    "wavtokenizer-dec",
    "xverse",
    "internlm2",
];

impl Architecture {
    /// Parse an architecture from its GGUF `general.architecture` name.
    ///
    /// Returns `None` if candle has no quantized loader for it — use
    /// [`Architecture::is_known_llama_cpp_arch`] to distinguish "known to
    /// llama.cpp but unimplemented" from "never heard of it".
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "llama" => Self::Llama,
            "gemma" | "gemma2" | "gemma3" | "gemma-embedding" => Self::Gemma,
            "glm4" => Self::Glm4,
            "lfm2" => Self::Lfm2,
            "phi2" => Self::Phi2,
            "phi3" => Self::Phi3,
            "qwen2" => Self::Qwen2,
            "qwen3" => Self::Qwen3,
            "qwen3moe" => Self::Qwen3Moe,
            _ => return None,
        })
    }

    /// Parse an architecture from the GGUF `general.architecture` metadata.
    pub fn from_gguf_metadata(metadata: &HashMap<String, gguf_file::Value>) -> Option<Self> {
        Self::from_name(Self::arch_name(metadata)?.as_str())
    }

    /// Extract the raw `general.architecture` string from GGUF metadata.
    pub fn arch_name(metadata: &HashMap<String, gguf_file::Value>) -> Option<String> {
        metadata
            .get("general.architecture")?
            .to_string()
            .ok()
            .cloned()
    }

    /// Detect the architecture, or return a human-readable explanation of why
    /// the model cannot be loaded.
    pub fn detect(
        metadata: &HashMap<String, gguf_file::Value>,
    ) -> std::result::Result<Self, String> {
        let Some(name) = Self::arch_name(metadata) else {
            return Err(
                "GGUF metadata has no `general.architecture` key — the file is corrupt \
                 or is not a model file"
                    .to_string(),
            );
        };
        if let Some(arch) = Self::from_name(&name) {
            return Ok(arch);
        }
        if Self::is_known_llama_cpp_arch(&name) {
            Err(format!(
                "Model architecture '{name}' is a known llama.cpp architecture, but no \
                 pure-Rust quantized loader exists for it in candle yet. \
                 Joshua currently supports: {}.",
                Self::list_known()
            ))
        } else {
            Err(format!(
                "Unrecognised GGUF architecture '{name}'. \
                 Joshua currently supports: {}.",
                Self::list_known()
            ))
        }
    }

    /// Whether `name` appears in llama.cpp's architecture registry (either
    /// supported here or known-but-unimplemented).
    pub fn is_known_llama_cpp_arch(name: &str) -> bool {
        Self::from_name(name).is_some() || KNOWN_UNSUPPORTED_ARCHS.contains(&name)
    }

    /// Human-readable name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Llama => "Llama (also Mistral, Mixtral, TinyLlama, SmolLM, Yi, …)",
            Self::Gemma => "Gemma / Gemma 2 / Gemma 3",
            Self::Glm4 => "GLM-4",
            Self::Lfm2 => "LFM2",
            Self::Phi2 => "Phi-1 / Phi-1.5 / Phi-2",
            Self::Phi3 => "Phi-3",
            Self::Qwen2 => "Qwen2 / Qwen2.5",
            Self::Qwen3 => "Qwen3",
            Self::Qwen3Moe => "Qwen3-MoE",
        }
    }

    /// Comma-separated list of supported architecture names for error messages.
    pub fn list_known() -> &'static str {
        "llama (incl. Mistral/Mixtral), gemma, gemma2, gemma3, gemma-embedding, \
         glm4, lfm2, phi2, phi3, qwen2, qwen3, qwen3moe"
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
    Gemma(quantized_gemma3::ModelWeights),
    Glm4(quantized_glm4::ModelWeights),
    Lfm2(quantized_lfm2::ModelWeights),
    Phi2(quantized_phi::ModelWeights),
    Phi3(quantized_phi3::ModelWeights),
    Qwen2(quantized_qwen2::ModelWeights),
    Qwen3(quantized_qwen3::ModelWeights),
    Qwen3Moe(quantized_qwen3_moe::GGUFQWenMoE),
}

impl QuantizedModel {
    /// Load a model from a GGUF file, dispatching to the correct quantized
    /// loader based on `general.architecture`.
    pub fn from_gguf<R: Read + Seek>(
        gguf: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let arch = Architecture::detect(&gguf.metadata).map_err(candle_core::Error::Msg)?;

        tracing::info!("Detected model architecture: {}", arch.display_name());

        match arch {
            Architecture::Llama => {
                quantized_llama::ModelWeights::from_gguf(gguf, reader, device).map(Self::Llama)
            }
            Architecture::Gemma => {
                quantized_gemma3::ModelWeights::from_gguf(gguf, reader, device).map(Self::Gemma)
            }
            Architecture::Glm4 => {
                // F32 activations: fastest/most accurate compute dtype on CPU.
                quantized_glm4::ModelWeights::from_gguf(gguf, reader, device, DType::F32)
                    .map(Self::Glm4)
            }
            Architecture::Lfm2 => {
                quantized_lfm2::ModelWeights::from_gguf(gguf, reader, device).map(Self::Lfm2)
            }
            Architecture::Phi2 => {
                quantized_phi::ModelWeights::from_gguf(gguf, reader, device).map(Self::Phi2)
            }
            Architecture::Phi3 => {
                // CPU-only: flash attention not available.
                quantized_phi3::ModelWeights::from_gguf(false, gguf, reader, device)
                    .map(Self::Phi3)
            }
            Architecture::Qwen2 => {
                quantized_qwen2::ModelWeights::from_gguf(gguf, reader, device).map(Self::Qwen2)
            }
            Architecture::Qwen3 => {
                quantized_qwen3::ModelWeights::from_gguf(gguf, reader, device).map(Self::Qwen3)
            }
            Architecture::Qwen3Moe => {
                quantized_qwen3_moe::GGUFQWenMoE::from_gguf(gguf, reader, device, DType::F32)
                    .map(Self::Qwen3Moe)
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
            Self::Gemma(m) => m.forward(input, index_pos),
            Self::Glm4(m) => m.forward(input, index_pos),
            Self::Lfm2(m) => m.forward(input, index_pos),
            Self::Phi2(m) => m.forward(input, index_pos),
            Self::Phi3(m) => m.forward(input, index_pos),
            Self::Qwen2(m) => m.forward(input, index_pos),
            Self::Qwen3(m) => m.forward(input, index_pos),
            Self::Qwen3Moe(m) => m.forward(input, index_pos),
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata_with_arch(arch: &str) -> HashMap<String, gguf_file::Value> {
        let mut m = HashMap::new();
        m.insert(
            "general.architecture".to_string(),
            gguf_file::Value::String(arch.to_string()),
        );
        m
    }

    #[test]
    fn supported_architectures_resolve() {
        for (name, expected) in [
            ("llama", Architecture::Llama),
            ("gemma", Architecture::Gemma),
            ("gemma2", Architecture::Gemma),
            ("gemma3", Architecture::Gemma),
            ("gemma-embedding", Architecture::Gemma),
            ("glm4", Architecture::Glm4),
            ("lfm2", Architecture::Lfm2),
            ("phi2", Architecture::Phi2),
            ("phi3", Architecture::Phi3),
            ("qwen2", Architecture::Qwen2),
            ("qwen3", Architecture::Qwen3),
            ("qwen3moe", Architecture::Qwen3Moe),
        ] {
            assert_eq!(Architecture::from_name(name), Some(expected), "arch {name}");
            assert_eq!(
                Architecture::from_gguf_metadata(&metadata_with_arch(name)),
                Some(expected),
                "metadata arch {name}"
            );
            assert!(Architecture::is_known_llama_cpp_arch(name));
        }
    }

    #[test]
    fn known_unsupported_architectures_give_specific_error() {
        for name in ["mamba", "gpt2", "deepseek2", "rwkv7", "starcoder2"] {
            assert_eq!(Architecture::from_name(name), None);
            assert!(Architecture::is_known_llama_cpp_arch(name), "arch {name}");
            let err = Architecture::detect(&metadata_with_arch(name)).unwrap_err();
            assert!(
                err.contains("known llama.cpp architecture"),
                "error for {name}: {err}"
            );
        }
    }

    #[test]
    fn unknown_architecture_gives_generic_error() {
        assert!(!Architecture::is_known_llama_cpp_arch("not-a-real-arch"));
        let err = Architecture::detect(&metadata_with_arch("not-a-real-arch")).unwrap_err();
        assert!(err.contains("Unrecognised"), "error: {err}");
    }

    #[test]
    fn missing_architecture_key_gives_corrupt_file_error() {
        let err = Architecture::detect(&HashMap::new()).unwrap_err();
        assert!(err.contains("general.architecture"), "error: {err}");
    }
}
