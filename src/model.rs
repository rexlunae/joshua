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
//! | `deepseek2` (DeepSeek-V2/V3, Kimi-K2)            | `quantized_deepseek2` (Joshua)
//! | `deepseek4` (DeepSeek-V4)                        | `quantized_deepseek4` (Joshua)
//!
//! Every other architecture name in llama.cpp's registry is recognised and
//! reported with a clear "known but not yet loadable in pure Rust" error, so
//! users can tell the difference between an unsupported model and a corrupt
//! or mislabelled file.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Result, Tensor};
use candle_transformers::models::{
    quantized_gemma3, quantized_glm4, quantized_lfm2, quantized_llama, quantized_phi,
    quantized_phi3, quantized_qwen2, quantized_qwen3,
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
    /// `deepseek2` — DeepSeek-V2/V3 and Kimi-K2 (MLA + fine-grained MoE).
    DeepSeek2,
    /// `deepseek4` — DeepSeek-V4 (sliding-window MLA + KV compression +
    /// indexer-selected sparse attention + Hyper-Connections).
    DeepSeek4,
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
    // Kimi K3. The architecture primitives (KDA, attention residuals, the
    // latent MoE) live in `crate::kimi_k3`, but the end-to-end loader is not
    // finished, so the model is still reported as unsupported rather than
    // half-loading.
    "kimi-k3",
    "kimi-linear",
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
            "deepseek2" => Self::DeepSeek2,
            "deepseek4" => Self::DeepSeek4,
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
            Self::DeepSeek2 => "DeepSeek-V2 / DeepSeek-V3 / Kimi-K2",
            Self::DeepSeek4 => "DeepSeek-V4",
        }
    }

    /// Comma-separated list of supported architecture names for error messages.
    pub fn list_known() -> &'static str {
        "llama (incl. Mistral/Mixtral), gemma, gemma2, gemma3, gemma-embedding, \
         glm4, lfm2, phi2, phi3, qwen2, qwen3, qwen3moe, deepseek2, deepseek4"
    }
}

// ─── Dispatched model ───────────────────────────────────────────────────────

/// A quantized model loaded from a GGUF file, wrapping the correct candle
/// model type for the detected architecture.
///
/// This hides the concrete model type behind a uniform `forward()` API so the
/// engine can handle multiple architectures without code duplication.
// Model structs are built once per model load and matched by reference; the
// size difference between variants is irrelevant.
#[allow(clippy::large_enum_variant)]
pub enum QuantizedModel {
    Llama(quantized_llama::ModelWeights),
    Gemma(quantized_gemma3::ModelWeights),
    Glm4(quantized_glm4::ModelWeights),
    Lfm2(quantized_lfm2::ModelWeights),
    Phi2(quantized_phi::ModelWeights),
    Phi3(quantized_phi3::ModelWeights),
    Qwen2(quantized_qwen2::ModelWeights),
    Qwen3(quantized_qwen3::ModelWeights),
    Qwen3Moe(crate::quantized_qwen3_moe::GGUFQWenMoE),
    DeepSeek2(crate::quantized_deepseek2::ModelWeights),
    DeepSeek4(crate::quantized_deepseek4::ModelWeights),
}

impl QuantizedModel {
    /// Load a model from a GGUF file, dispatching to the correct quantized
    /// loader based on `general.architecture`.
    pub fn from_gguf<R: Read + Seek>(
        gguf: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        Self::from_gguf_mmap(gguf, reader, device, None, None)
    }

    /// Load, borrowing weights in place from `mmap` for the architectures
    /// whose loaders Joshua owns.
    ///
    /// candle's own quantized loaders read through the `reader` and copy, so
    /// for those architectures the mapping is ignored — they are small enough
    /// that it does not matter.  Joshua's own loaders (currently `deepseek2`)
    /// borrow, which is what makes the very large MoE models tractable.
    pub fn from_gguf_mmap<R: Read + Seek>(
        gguf: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        mmap: Option<std::sync::Arc<memmap2::Mmap>>,
        file: Option<std::sync::Arc<std::fs::File>>,
    ) -> Result<Self> {
        let arch = Architecture::detect(&gguf.metadata).map_err(candle_core::Error::Msg)?;

        tracing::info!("Detected model architecture: {}", arch.display_name());

        // Re-read the header the way `gguf_ext` does — with each tensor's
        // dtype kept as its raw GGUF id — but only for deepseek4, the one
        // loader that consumes the raw table (candle's `Content` cannot name
        // IQ2_XXS or I32, so the deepseek4 loader needs this to decode those
        // tensors).  The reader is rewound to where candle's header parse
        // left it (the aligned data start) afterwards, so nothing downstream
        // shifts.
        //
        // The re-read must not run for other architectures: none of them
        // consume the raw table, so the second parse would be pure cost (a
        // full re-parse and clone of the metadata + tensor maps for every
        // warm-pool instance) and any difference between the two parsers
        // could reject a file candle accepts.  `read_header` is a superset of
        // candle's parser for every file that can actually load (nested
        // arrays, depth cap, lenient strings; its per-string/array resource
        // caps are deliberately tighter than candle's theoretical 1 GiB, on
        // which candle itself eagerly allocates and fails), but other
        // architectures should stay on exactly the behaviour candle gave
        // them.
        //
        // A failed re-read is a real header problem (truncated/corrupt file,
        // implausible counts, non-UTF-8 key) — surface it as the load error
        // rather than silently treating the file as having no raw table, which
        // would surface much later as a misleading "cannot find tensor".
        let raw = if arch == Architecture::DeepSeek4 {
            let data_pos = reader.stream_position()?;
            let h = (|| -> std::result::Result<_, crate::JoshuaError> {
                reader.seek(SeekFrom::Start(0))?;
                let h = crate::gguf_ext::read_header(reader)?;
                reader.seek(SeekFrom::Start(data_pos))?;
                Ok(h)
            })()
            .map_err(|e| {
                // Restore the reader even on failure so callers see a
                // deterministic position, then report the actual problem.
                let _ = reader.seek(SeekFrom::Start(data_pos));
                candle_core::Error::Msg(format!("GGUF header re-read failed: {e}"))
            })?;
            Some(h)
        } else {
            None
        };
        let raw = raw.as_ref();

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
                quantized_phi3::ModelWeights::from_gguf(false, gguf, reader, device).map(Self::Phi3)
            }
            Architecture::Qwen2 => {
                quantized_qwen2::ModelWeights::from_gguf(gguf, reader, device).map(Self::Qwen2)
            }
            Architecture::Qwen3 => {
                quantized_qwen3::ModelWeights::from_gguf(gguf, reader, device).map(Self::Qwen3)
            }
            Architecture::Qwen3Moe => {
                crate::quantized_qwen3_moe::GGUFQWenMoE::from_gguf_mmap(gguf, reader, device, mmap)
                    .map(Self::Qwen3Moe)
            }
            Architecture::DeepSeek2 => {
                crate::quantized_deepseek2::ModelWeights::from_gguf_mmap(gguf, reader, device, mmap)
                    .map(Self::DeepSeek2)
            }
            Architecture::DeepSeek4 => crate::quantized_deepseek4::ModelWeights::from_gguf_mmap(
                gguf, raw, reader, device, mmap, file,
            )
            .map(Self::DeepSeek4),
        }
    }

    /// Clear the KV cache so the instance can serve an unrelated prompt,
    /// where the underlying candle model supports it.
    ///
    /// Returns `false` when the architecture has no reset hook — the caller
    /// must build a fresh instance instead.
    pub fn clear_kv_cache(&mut self) -> bool {
        match self {
            Self::Llama(m) => {
                m.clear_kv_cache();
                true
            }
            Self::Qwen2(m) => {
                m.clear_kv_cache();
                true
            }
            Self::Qwen3(m) => {
                m.clear_kv_cache();
                true
            }
            Self::Qwen3Moe(m) => {
                m.clear_kv_cache();
                true
            }
            Self::DeepSeek2(m) => {
                m.clear_kv_cache();
                true
            }
            Self::DeepSeek4(m) => {
                m.clear_kv_cache();
                true
            }
            _ => false,
        }
    }

    /// Whether [`QuantizedModel::clear_kv_cache`] can reset this instance.
    pub fn supports_kv_clear(&self) -> bool {
        matches!(
            self,
            Self::Llama(_)
                | Self::Qwen2(_)
                | Self::Qwen3(_)
                | Self::Qwen3Moe(_)
                | Self::DeepSeek2(_)
                | Self::DeepSeek4(_)
        )
    }

    /// Whether [`QuantizedModel::truncate_kv_cache`] can shorten this
    /// instance's KV cache to an arbitrary prefix length.
    ///
    /// Only the Joshua-owned loaders with plain append-only caches qualify:
    /// candle's stock loaders keep their caches private, and `deepseek4`'s
    /// hybrid attention keeps running compressor states that cannot be
    /// rewound to an arbitrary past position (only cleared).
    pub fn supports_kv_truncate(&self) -> bool {
        matches!(self, Self::Qwen3Moe(_) | Self::DeepSeek2(_))
    }

    /// Keep only the first `keep_len` fed tokens of every layer's KV cache.
    ///
    /// Positions `[0..keep_len)` stay exactly as they were produced by the
    /// tokens both conversations share; everything after is recomputed by
    /// the next prefill, which continues at absolute position `keep_len`.
    /// Used for edited-context prefix reuse: agent harnesses truncate or
    /// replace middle blocks of a conversation (old tool outputs, thinking
    /// segments), so a follow-up prompt often shares a *prefix* with the
    /// cached history without extending it.
    ///
    /// Returns `Ok(false)` when the architecture has no truncation hook —
    /// check [`Self::supports_kv_truncate`] to distinguish that from a real
    /// error.
    pub fn truncate_kv_cache(&mut self, keep_len: usize) -> Result<bool> {
        match self {
            Self::Qwen3Moe(m) => m.truncate_kv_cache(keep_len).map(|_| true),
            Self::DeepSeek2(m) => m.truncate_kv_cache(keep_len).map(|_| true),
            _ => Ok(false),
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
            Self::DeepSeek2(m) => m.forward(input, index_pos),
            Self::DeepSeek4(m) => m.forward(input, index_pos),
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
            ("deepseek2", Architecture::DeepSeek2),
            ("deepseek4", Architecture::DeepSeek4),
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
        for name in ["mamba", "gpt2", "deepseek", "rwkv7", "starcoder2"] {
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
