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
//! | `lfm2`                                          | `quantized_lfm2`
//! | `phi2`                                          | `quantized_phi`
//! | `phi3`                                          | `quantized_phi3`
//! | `qwen2`                                         | `quantized_qwen2`
//! | `qwen`, `qwen2moe`, `qwen2vl`, `qwen3`, `qwen3moe`, `qwen3vl`, `qwen3vlmoe`, `qwen3next`, `qwen35`, `qwen35moe`, `qwen4exp` | `quantized_qwen` (Joshua)
//! | `chatglm` (ChatGLM2/3, GLM-4-9B-Chat), `glm4` (GLM-4-0414, GLM-4.1V, GLM-OCR), `glm4moe` (GLM-4.5/4.6/4.7, Solar-Open) | `quantized_qwen` (Joshua)
//! | `deepseek` (DeepSeek-MoE)                        | `quantized_deepseek2` (Joshua)
//! | `deepseek2` (DeepSeek-V2/V2.5/V3/V3.1/R1, Kimi-K2, GLM-4.7-Flash) | `quantized_deepseek2` (Joshua)
//! | `glm-dsa` (GLM-5 / 5.1 / 5.2)                    | `quantized_deepseek2` (Joshua)
//! | `glm5next` / `glm5-next` (GLM-5.3-Flash)         | `quantized_deepseek2` (Joshua)
//! | `deepseek4` (DeepSeek-V4), `deepseek41` (DeepSeek-V4.1) | `quantized_deepseek4` (Joshua)
//!
//! The dense DeepSeek releases — DeepSeek-LLM, DeepSeek-Coder (V1) and the
//! DeepSeek-R1 distills — are converted by llama.cpp as `llama` or `qwen2`
//! and load through those rows.
//!
//! Every other architecture name in llama.cpp's registry is recognised and
//! reported with a clear "known but not yet loadable in pure Rust" error, so
//! users can tell the difference between an unsupported model and a corrupt
//! or mislabelled file.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

use candle_core::quantized::gguf_file;
use candle_core::{Device, Result, Tensor};
use candle_transformers::models::{
    quantized_gemma3, quantized_lfm2, quantized_llama, quantized_phi, quantized_phi3,
    quantized_qwen2,
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
    /// `chatglm` — ChatGLM2 / ChatGLM3 / GLM-4-9B-Chat (fused QKV, fused
    /// SwiGLU).
    ChatGlm,
    /// `glm4` — GLM-4-0414 dense models, GLM-4.1V (text) and GLM-OCR
    /// (sandwich norms).
    Glm4,
    /// `glm4moe` — GLM-4.5 / 4.5-Air / 4.6 / 4.7 and Solar-Open (sigmoid-routed
    /// MoE with a shared expert).
    Glm4Moe,
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
    /// `qwen` — Qwen (1).
    Qwen,
    /// `qwen2moe` — Qwen1.5-MoE / Qwen2-MoE.
    Qwen2Moe,
    /// `qwen2vl` — Qwen2-VL / Qwen2.5-VL (text decoder).
    Qwen2Vl,
    /// `qwen3moe` — Qwen3 mixture-of-experts models.
    Qwen3Moe,
    /// `qwen3vl` — Qwen3-VL (text decoder).
    Qwen3Vl,
    /// `qwen3vlmoe` — Qwen3-VL-MoE (text decoder).
    Qwen3VlMoe,
    /// `qwen3next` — Qwen3-Next (Gated DeltaNet + gated attention hybrid MoE).
    Qwen3Next,
    /// `qwen35` — Qwen3.5 dense (Gated DeltaNet hybrid).
    Qwen35,
    /// `qwen35moe` — Qwen3.5-MoE (Gated DeltaNet hybrid).
    Qwen35Moe,
    /// `qwen4exp` — Qwen3.8-Flash-Next (Qwen3.5-MoE plus hyper-connections,
    /// QSA block-sparse attention and PLE n-gram hash embeddings).
    Qwen4Exp,
    /// `deepseek` — DeepSeek-MoE (GQA + fine-grained MoE with shared
    /// experts).  Loaded by the `deepseek2` loader into
    /// [`QuantizedModel::DeepSeek2`].
    DeepSeek,
    /// `deepseek2` — DeepSeek-V2/V2.5/V3/V3.1/R1, Kimi-K2 and GLM-4.7-Flash
    /// (MLA + fine-grained MoE).
    DeepSeek2,
    /// `glm-dsa` — GLM-5 / 5.1 / 5.2 (the `deepseek2` stack plus DeepSeek
    /// Sparse Attention with IndexShare).  Served by the `deepseek2` loader.
    GlmDsa,
    /// `glm5next` (also `glm5-next`, per llama.cpp's two open pull
    /// requests) — GLM-5.3-Flash: KDA / sparse-MLA hybrid with
    /// hyper-connections.  Served by the `deepseek2` loader.
    Glm5Next,
    /// `deepseek4` — DeepSeek-V4 (sliding-window MLA + KV compression +
    /// indexer-selected sparse attention + Hyper-Connections).
    DeepSeek4,
    /// `deepseek41` — DeepSeek-V4.1 (V4 with a lagged hyper-connection mix,
    /// KV compressed on a few source layers and shared, engram n-gram
    /// tables).  Served by the `deepseek4` loader.
    DeepSeek41,
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
    "codeshell",
    "cogvlm",
    "cohere2",
    "command-r",
    "dbrx",
    "deci",
    "dots1",
    "dream",
    "ernie4_5",
    "ernie4_5-moe",
    "exaone",
    "exaone4",
    "falcon",
    "falcon-h1",
    "gemma3n",
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
    // Qwen3-TTS emits audio codec tokens, not text.
    "qwen3tts",
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

/// Every supported `general.architecture` name and the variant it loads as.
const NAMES: &[(&str, Architecture)] = &[
    ("llama", Architecture::Llama),
    ("gemma", Architecture::Gemma),
    ("gemma2", Architecture::Gemma),
    ("gemma3", Architecture::Gemma),
    ("gemma-embedding", Architecture::Gemma),
    ("chatglm", Architecture::ChatGlm),
    ("glm4", Architecture::Glm4),
    ("glm4moe", Architecture::Glm4Moe),
    ("lfm2", Architecture::Lfm2),
    ("phi2", Architecture::Phi2),
    ("phi3", Architecture::Phi3),
    ("qwen", Architecture::Qwen),
    ("qwen2", Architecture::Qwen2),
    ("qwen2moe", Architecture::Qwen2Moe),
    ("qwen2vl", Architecture::Qwen2Vl),
    ("qwen3", Architecture::Qwen3),
    ("qwen3moe", Architecture::Qwen3Moe),
    ("qwen3vl", Architecture::Qwen3Vl),
    ("qwen3vlmoe", Architecture::Qwen3VlMoe),
    ("qwen3next", Architecture::Qwen3Next),
    ("qwen35", Architecture::Qwen35),
    ("qwen35moe", Architecture::Qwen35Moe),
    ("qwen4exp", Architecture::Qwen4Exp),
    ("deepseek", Architecture::DeepSeek),
    ("deepseek2", Architecture::DeepSeek2),
    ("glm-dsa", Architecture::GlmDsa),
    ("glm5next", Architecture::Glm5Next),
    ("glm5-next", Architecture::Glm5Next),
    ("deepseek4", Architecture::DeepSeek4),
    ("deepseek41", Architecture::DeepSeek41),
];

impl Architecture {
    /// Parse an architecture from its GGUF `general.architecture` name.
    ///
    /// Returns `None` if candle has no quantized loader for it — use
    /// [`Architecture::is_known_llama_cpp_arch`] to distinguish "known to
    /// llama.cpp but unimplemented" from "never heard of it".
    pub fn from_name(name: &str) -> Option<Self> {
        NAMES.iter().find(|(n, _)| *n == name).map(|&(_, a)| a)
    }

    /// Every supported `general.architecture` name.
    pub fn supported_names() -> impl Iterator<Item = &'static str> {
        NAMES.iter().map(|&(n, _)| n)
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
    /// Whether this architecture's loader keeps its weights behind a shared
    /// handle so sessions can be derived without copying them (see
    /// [`QuantizedModel::new_session`]).
    pub fn shares_weights(&self) -> bool {
        self.is_native()
    }

    /// Whether this architecture is served by one of Joshua's own loaders
    /// (`quantized_qwen`, `quantized_deepseek2`, `quantized_deepseek4`)
    /// rather than a stock candle loader.  Those keep the token embedding
    /// and output head quantized, hold MoE routers in f32, and share weights
    /// across sessions.
    pub fn is_native(&self) -> bool {
        !matches!(
            self,
            Self::Llama
                | Self::Gemma
                | Self::Lfm2
                | Self::Phi2
                | Self::Phi3
                | Self::Qwen2
        )
    }

    /// Whether this architecture's loader keeps the routed experts in host
    /// RAM regardless of the requested placement.  `deepseek4`'s IQ2_XXS
    /// experts have no accelerator kernel, so its loader always borrows
    /// them from the mapping on the CPU and runs only the dense set on the
    /// device; an `ExpertPlacement::Device` request cannot change that and
    /// must not be accounted as if it had.
    ///
    /// Exception: on an OpenCL device (e.g. an Arc discrete GPU) there IS an
    /// IQ2_XXS kernel (`crate::iq2xxs` OpenCL fused GEMV), so experts may
    /// target the device there.  See [`Self::experts_always_on_host_for`].
    pub fn experts_always_on_host(&self) -> bool {
        self.is_deepseek4()
    }

    /// DeepSeek-V4 or V4.1 — the architectures the `deepseek4` loader serves.
    pub fn is_deepseek4(&self) -> bool {
        matches!(self, Self::DeepSeek4 | Self::DeepSeek41)
    }

    /// Whether the routed experts must stay in host RAM *given the active
    /// device*.  `DeepSeek4` experts may run on an OpenCL or SYCL device
    /// (both have the IQ2_XXS kernel); on every other device (CPU, Vulkan,
    /// Metal, CUDA) they are host-only.  Non-MoE architectures never force
    /// host experts.
    pub fn experts_always_on_host_for(&self, device: &Device) -> bool {
        self.is_deepseek4() && !(device.is_opencl() || device.is_sycl())
    }

    /// Whether this architecture's "experts on the device" form is a bounded
    /// VRAM cache over the host pool rather than an upload of the whole pool:
    /// `deepseek4` on OpenCL or SYCL keeps every routed expert borrowed from the
    /// mapping and caches a budgeted subset on the card (the ~72 GiB pool of
    /// V4-Flash never fits), so an `ExpertPlacement::Device` request means
    /// "size that cache" (see `--vram-expert-cache`).
    pub fn device_experts_are_a_cache(&self, device: &Device) -> bool {
        self.is_deepseek4() && (device.is_opencl() || device.is_sycl())
    }

    pub fn is_known_llama_cpp_arch(name: &str) -> bool {
        Self::from_name(name).is_some() || KNOWN_UNSUPPORTED_ARCHS.contains(&name)
    }

    /// Human-readable name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Llama => "Llama (also Mistral, Mixtral, TinyLlama, SmolLM, Yi, …)",
            Self::Gemma => "Gemma / Gemma 2 / Gemma 3",
            Self::ChatGlm => "ChatGLM2 / ChatGLM3 / GLM-4-9B-Chat",
            Self::Glm4 => "GLM-4 / GLM-4.1V (text) / GLM-OCR",
            Self::Glm4Moe => "GLM-4.5 / GLM-4.6 / GLM-4.7 / Solar-Open",
            Self::Lfm2 => "LFM2",
            Self::Phi2 => "Phi-1 / Phi-1.5 / Phi-2",
            Self::Phi3 => "Phi-3",
            Self::Qwen => "Qwen",
            Self::Qwen2 => "Qwen2 / Qwen2.5",
            Self::Qwen2Moe => "Qwen1.5-MoE / Qwen2-MoE",
            Self::Qwen2Vl => "Qwen2-VL / Qwen2.5-VL (text)",
            Self::Qwen3 => "Qwen3",
            Self::Qwen3Moe => "Qwen3-MoE",
            Self::Qwen3Vl => "Qwen3-VL (text)",
            Self::Qwen3VlMoe => "Qwen3-VL-MoE (text)",
            Self::Qwen3Next => "Qwen3-Next",
            Self::Qwen35 => "Qwen3.5",
            Self::Qwen35Moe => "Qwen3.5-MoE",
            Self::Qwen4Exp => "Qwen3.8-Flash-Next (qwen4exp)",
            Self::DeepSeek => "DeepSeek-MoE",
            Self::DeepSeek2 => "DeepSeek-V2 / DeepSeek-V3 / DeepSeek-R1 / Kimi-K2 / GLM-4.7-Flash",
            Self::GlmDsa => "GLM-5 / GLM-5.1 / GLM-5.2",
            Self::Glm5Next => "GLM-5.3-Flash",
            Self::DeepSeek4 => "DeepSeek-V4",
            Self::DeepSeek41 => "DeepSeek-V4.1",
        }
    }

    /// Comma-separated list of supported architecture names for error messages.
    pub fn list_known() -> &'static str {
        static LIST: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        LIST.get_or_init(|| Self::supported_names().collect::<Vec<_>>().join(", "))
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
    Lfm2(quantized_lfm2::ModelWeights),
    Phi2(quantized_phi::ModelWeights),
    Phi3(quantized_phi3::ModelWeights),
    Qwen2(quantized_qwen2::ModelWeights),
    /// Every architecture of the Qwen family loader (see
    /// [`crate::quantized_qwen::ARCHES`]).
    Qwen(crate::quantized_qwen::ModelWeights),
    DeepSeek2(crate::quantized_deepseek2::ModelWeights),
    DeepSeek4(crate::quantized_deepseek4::ModelWeights),
}

/// Evaluate `$body` with `$m` bound to the model of any Joshua-native
/// variant (`Qwen`, `DeepSeek2`, `DeepSeek4`) and `$wrap` to that variant's
/// constructor; `$other` for candle's stock loaders.
macro_rules! native {
    ($self:expr, ($m:ident, $wrap:ident) => $body:expr, _ => $other:expr) => {
        match $self {
            QuantizedModel::Qwen($m) => {
                #[allow(unused_variables)]
                let $wrap = QuantizedModel::Qwen;
                $body
            }
            QuantizedModel::DeepSeek2($m) => {
                #[allow(unused_variables)]
                let $wrap = QuantizedModel::DeepSeek2;
                $body
            }
            QuantizedModel::DeepSeek4($m) => {
                #[allow(unused_variables)]
                let $wrap = QuantizedModel::DeepSeek4;
                $body
            }
            _ => $other,
        }
    };
    ($self:expr, $m:ident => $body:expr, _ => $other:expr) => {
        native!($self, ($m, _wrap) => $body, _ => $other)
    };
}

/// [`native!`] over the loaders built on
/// [`crate::native_session::Session`] (`Qwen`, `DeepSeek2`), which add
/// all-position logits and prefix truncation.
macro_rules! session {
    ($self:expr, $m:ident => $body:expr, _ => $other:expr) => {
        match $self {
            QuantizedModel::Qwen($m) => $body,
            QuantizedModel::DeepSeek2($m) => $body,
            _ => $other,
        }
    };
}

impl QuantizedModel {
    /// Load a model from a GGUF file, dispatching to the correct quantized
    /// loader based on `general.architecture`.
    pub fn from_gguf<R: Read + Seek>(
        gguf: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        Self::from_gguf_mmap(gguf, reader, device, None, None, 0)
    }

    /// Load, borrowing weights in place from `mmap` for the architectures
    /// whose loaders Joshua owns.
    ///
    /// candle's own quantized loaders read through the `reader` and copy, so
    /// for those architectures the mapping is ignored — they are small enough
    /// that it does not matter.  Joshua's own loaders (Qwen family, `deepseek2`)
    /// borrow, which is what makes the very large MoE models tractable.
    pub fn from_gguf_mmap<R: Read + Seek>(
        gguf: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        mmap: Option<std::sync::Arc<memmap2::Mmap>>,
        file: Option<std::sync::Arc<std::fs::File>>,
        // The engine's configured context length (0 = no engine limit).
        // Only `deepseek4` consumes it, to size its KV caches to what the
        // engine will actually serve instead of the model's context length.
        n_ctx: usize,
    ) -> Result<Self> {
        Self::from_gguf_mmap_placed(gguf, reader, device, device, mmap, file, n_ctx, None)
    }

    /// [`QuantizedModel::from_gguf_mmap`] with an explicit device for the
    /// routed-expert pool of a sparse MoE model.
    ///
    /// The dense set always goes to `device`.  With `expert_device` the CPU
    /// on an accelerator model, the MoE loaders (`qwen3moe`, `deepseek2`)
    /// keep the routed experts in host RAM — borrowed from the mapping and
    /// run through the CPU expert kernels, with each layer's MoE activations
    /// hopping across the bus — which is what runs a model larger than the
    /// device's memory (see [`crate::placement::ExpertPlacement`]).
    /// `deepseek4` always keeps a mapped model's experts borrowed on the
    /// CPU; with a `device_expert_cache_bytes` budget on an OpenCL or SYCL
    /// device it
    /// also runs a bounded cache of them on the card (see
    /// [`crate::quantized_deepseek4::ModelWeights::from_gguf_mmap_placed`]).
    /// Dense architectures ignore the parameter.
    pub fn from_gguf_mmap_placed<R: Read + Seek>(
        gguf: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        expert_device: &Device,
        mmap: Option<std::sync::Arc<memmap2::Mmap>>,
        file: Option<std::sync::Arc<std::fs::File>>,
        n_ctx: usize,
        device_expert_cache_bytes: Option<u64>,
    ) -> Result<Self> {
        let arch = Architecture::detect(&gguf.metadata).map_err(candle_core::Error::Msg)?;

        tracing::info!("Detected model architecture: {}", arch.display_name());

        // Re-read the header the way `gguf_ext` does — with each tensor's
        // dtype kept as its raw GGUF id — for Joshua's own loaders, which
        // decode the formats candle's `Content` cannot name (IQ2_XXS, MXFP4,
        // Bonsai's Q1_0 / Q2_0, I32).  The reader is rewound to where
        // candle's header parse left it (the aligned data start) afterwards,
        // so nothing downstream shifts.
        //
        // Candle's stock loaders never consume the raw table, so they skip
        // the second parse and stay on exactly the behaviour candle gave
        // them.  For the native loaders other than deepseek4 (which also
        // reads I32 tables through it) the table is kept only when some
        // tensor actually needs it, so a warm pool of ordinary instances
        // does not hold a second copy of the metadata.
        //
        // For deepseek4 a failed re-read is a real header problem
        // (truncated/corrupt file, implausible counts, non-UTF-8 key) —
        // surface it as the load error rather than silently treating the
        // file as having no raw table, which would surface much later as a
        // misleading "cannot find tensor".  The other native loaders read
        // every tensor candle accepted through `gguf` anyway, so there the
        // re-read must never reject a file candle parsed: a failure just
        // means no raw table.
        let raw = if arch.is_native() {
            let data_pos = reader.stream_position()?;
            let h = (|| -> std::result::Result<_, crate::JoshuaError> {
                reader.seek(SeekFrom::Start(0))?;
                let h = crate::gguf_ext::read_header(reader)?;
                reader.seek(SeekFrom::Start(data_pos))?;
                Ok(h)
            })();
            // Restore the reader even on failure so callers see a
            // deterministic position.
            let _ = reader.seek(SeekFrom::Start(data_pos));
            match h {
                Ok(h) if arch.is_deepseek4() || h.has_unsupported_tensors() => Some(h),
                Ok(_) => None,
                Err(e) if arch.is_deepseek4() => {
                    return Err(candle_core::Error::Msg(format!("GGUF header re-read failed: {e}")))
                }
                Err(e) => {
                    tracing::debug!("GGUF header re-read failed ({e}); loading without a raw table");
                    None
                }
            }
        } else {
            None
        };

        match arch {
            Architecture::Llama => {
                quantized_llama::ModelWeights::from_gguf(gguf, reader, device).map(Self::Llama)
            }
            Architecture::Gemma => {
                quantized_gemma3::ModelWeights::from_gguf(gguf, reader, device).map(Self::Gemma)
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
            Architecture::Qwen
            | Architecture::Qwen3
            | Architecture::Qwen2Moe
            | Architecture::Qwen2Vl
            | Architecture::Qwen3Moe
            | Architecture::Qwen3Vl
            | Architecture::Qwen3VlMoe
            | Architecture::Qwen3Next
            | Architecture::Qwen35
            | Architecture::Qwen35Moe
            | Architecture::Qwen4Exp
            | Architecture::ChatGlm
            | Architecture::Glm4
            | Architecture::Glm4Moe => {
                crate::quantized_qwen::ModelWeights::from_gguf_mmap_placed(
                    gguf,
                    raw,
                    reader,
                    device,
                    expert_device,
                    mmap,
                    device_expert_cache_bytes,
                )
                .map(Self::Qwen)
            }
            // One loader serves them all: DeepSeek-MoE differs from V2+ only
            // in its (GQA) attention, GLM-5 in its sparse one, GLM-5.3 in
            // that plus KDA layers and hyper-connections.
            Architecture::DeepSeek
            | Architecture::DeepSeek2
            | Architecture::GlmDsa
            | Architecture::Glm5Next => {
                crate::quantized_deepseek2::ModelWeights::from_gguf_mmap_placed(
                    gguf,
                    raw,
                    reader,
                    device,
                    expert_device,
                    mmap,
                    device_expert_cache_bytes,
                )
                .map(Self::DeepSeek2)
            }
            Architecture::DeepSeek4 | Architecture::DeepSeek41 => {
                crate::quantized_deepseek4::ModelWeights::from_gguf_mmap_placed(
                    gguf,
                    raw.as_ref(),
                    reader,
                    device,
                    expert_device,
                    mmap,
                    file,
                    n_ctx,
                    device_expert_cache_bytes,
                )
                .map(Self::DeepSeek4)
            }
        }
    }

    /// A fresh session over this instance's weights, sharing every weight
    /// tensor and starting with an empty cache — for Joshua's own loaders,
    /// which keep their weights behind a shared handle.
    ///
    /// `None` for candle's stock loaders, whose instances own their weights;
    /// those need a full load per session.  The engine keeps one loaded
    /// instance as the template and derives every concurrent session from
    /// it, so extra sessions cost their cache rather than a second copy (on
    /// an accelerator, a second upload) of the weights.
    pub fn new_session(&self) -> Option<Self> {
        native!(self, (m, wrap) => Some(wrap(m.new_session())), _ => None)
    }

    /// Whether [`QuantizedModel::new_session`] can derive sessions from this
    /// instance without copying its weights.
    pub fn supports_shared_weights(&self) -> bool {
        native!(self, _m => true, _ => false)
    }

    /// Number of live sessions sharing this instance's weights (Joshua's own
    /// loaders; 1 otherwise).
    pub fn shared_session_count(&self) -> usize {
        native!(self, m => m.shared_session_count(), _ => 1)
    }

    /// Whether the loaded token-embedding table is held quantized rather
    /// than dequantized to f32 (joshua-native loaders; diagnostics).
    pub fn embeddings_quantized(&self) -> Option<bool> {
        native!(self, m => Some(m.embeddings_quantized()), _ => None)
    }

    /// Pass the routing-frequency hot-expert cache budget to Joshua's own
    /// loaders; candle's stock loaders — including Mixtral through the
    /// vendored `llama` loader — ignore the request.
    ///
    /// The budget is applied after load so the loader's signature stays
    /// stable for every other architecture and for tests.
    pub fn set_pin_hot_experts(&mut self, n: usize) {
        if n > 0 {
            native!(self, m => m.set_pin_hot_experts(n), _ => {})
        }
    }

    /// Number of experts the residency backend can hold resident (a phase-5
    /// auto-sizing input); 0 for architectures without per-expert handles.
    pub fn expert_residency_capacity(&self) -> usize {
        native!(self, m => m.expert_residency_capacity(), _ => 0)
    }

    /// The model-wide device expert pool's budget, occupancy and counters,
    /// for the loaders that build one from a VRAM cache budget (`deepseek4`).
    pub fn device_expert_cache(&self) -> Option<crate::residency::DeviceCacheReport> {
        match self {
            Self::DeepSeek4(m) => m.device_expert_cache(),
            _ => None,
        }
    }

    /// The device expert pool's decode time split and host-miss page
    /// residency (`deepseek4`); `None` without a pool.
    pub fn expert_phase_timing(&self) -> Option<crate::quantized_deepseek4::ExpertPhaseTiming> {
        match self {
            Self::DeepSeek4(m) => m.expert_phase_timing(),
            _ => None,
        }
    }

    /// Block until the device expert pool's background uploads have drained
    /// (tests, diagnostics); a no-op without a pool.
    pub fn wait_for_expert_uploads(&self) {
        if let Self::DeepSeek4(m) = self {
            m.wait_for_expert_uploads();
        }
    }

    /// Clear the KV cache so the instance can serve an unrelated prompt,
    /// where the underlying model supports it.
    ///
    /// Returns `false` when the architecture has no reset hook — the caller
    /// must build a fresh instance instead.
    pub fn clear_kv_cache(&mut self) -> bool {
        match self {
            Self::Llama(m) => m.clear_kv_cache(),
            Self::Qwen2(m) => m.clear_kv_cache(),
            other => return native!(other, m => { m.clear_kv_cache(); true }, _ => false),
        }
        true
    }

    /// Whether [`QuantizedModel::clear_kv_cache`] can reset this instance.
    pub fn supports_kv_clear(&self) -> bool {
        matches!(self, Self::Llama(_) | Self::Qwen2(_)) || self.supports_shared_weights()
    }

    /// Whether [`QuantizedModel::truncate_kv_cache`] can shorten this
    /// instance's KV cache to an arbitrary prefix length.
    ///
    /// Only the Joshua-owned loaders with plain append-only caches qualify:
    /// candle's stock loaders keep their caches private, and recurrent state
    /// — `deepseek4`'s running compressors, the Gated DeltaNet layers of
    /// Qwen3-Next / Qwen3.5 — cannot be rewound to an arbitrary past
    /// position (only cleared).
    pub fn supports_kv_truncate(&self) -> bool {
        session!(self, m => m.supports_truncate(), _ => false)
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
        if !self.supports_kv_truncate() {
            return Ok(false);
        }
        session!(self, m => m.truncate_kv_cache(keep_len).map(|_| true), _ => Ok(false))
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
            Self::Lfm2(m) => m.forward(input, index_pos),
            Self::Phi2(m) => m.forward(input, index_pos),
            Self::Phi3(m) => m.forward(input, index_pos),
            Self::Qwen2(m) => m.forward(input, index_pos),
            Self::Qwen(m) => m.forward(input, index_pos),
            Self::DeepSeek2(m) => m.forward(input, index_pos),
            Self::DeepSeek4(m) => m.forward(input, index_pos),
        }
    }

    /// Whether this instance can run speculative decoding: score every
    /// position of a multi-token input ([`Self::forward_all_logits`]) *and*
    /// roll the KV cache back past rejected draft tokens
    /// ([`Self::truncate_kv_cache`]).  See [`crate::speculative`].
    pub fn supports_speculative(&self) -> bool {
        self.supports_kv_truncate()
    }

    /// Forward pass returning the logits of every input position,
    /// `[1, seq_len, vocab]` — the speculative-decoding verification pass.
    /// Only the [`crate::native_session::Session`] loaders implement it; the
    /// rest return an unsupported error.
    pub fn forward_all_logits(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
        session!(self, m => m.forward_all_logits(input, index_pos), _ => Err(candle_core::Error::Msg(
            "all-position logits are not implemented for this architecture".into(),
        )))
    }

    /// Whether this architecture has a native layer-streaming prefill path.
    /// See [`crate::stream_prefill`].  Joshua's own loaders implement it;
    /// other architectures fall back to the standard chunked prefill.
    pub fn supports_streaming(&self) -> bool {
        self.supports_shared_weights()
    }

    /// Layer-streaming prefill over a set of bounded chunks.  See
    /// [`crate::stream_prefill`].
    pub fn prefill_streamed(
        &mut self,
        chunks: &[crate::stream_prefill::Chunk],
        device: &Device,
    ) -> Result<Tensor> {
        native!(self, m => crate::stream_prefill::stream_prefill(m, chunks, device), _ => Err(candle_core::Error::Msg(
            "layer-streaming prefill is not implemented for this architecture".into(),
        )))
    }

    /// Batch one decode step across `seqs` independent sequences, amortizing
    /// the routed-expert fetch across the MoE dispatch.  See
    /// `quantized_deepseek4::ModelWeights::forward_sequences`.  Only supported
    /// on architectures with a native batched path (deepseek4); other
    /// architectures report an unsupported error so a caller can fall back
    /// to iterating sequences individually.  Part of the library API; the
    /// engine's own request paths do not batch yet.
    pub fn forward_sequences(&mut self, seqs: &[(&Tensor, usize)]) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::DeepSeek4(m) => m.forward_sequences(seqs),
            _ => Err(candle_core::Error::Msg(
                "batched forward_sequences is not implemented for this architecture".into(),
            )),
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
        for name in Architecture::supported_names() {
            let expected = Architecture::from_name(name).expect("listed name resolves");
            assert_eq!(
                Architecture::from_gguf_metadata(&metadata_with_arch(name)),
                Some(expected),
                "metadata arch {name}"
            );
            assert!(Architecture::is_known_llama_cpp_arch(name));
            assert!(
                !KNOWN_UNSUPPORTED_ARCHS.contains(&name),
                "{name} is both supported and listed as unsupported"
            );
            assert!(Architecture::list_known().contains(name));
        }
        // Every Qwen-family loader name is routed to the Qwen loader.
        for name in crate::quantized_qwen::ARCHES {
            assert!(Architecture::from_name(name).is_some_and(|a| a.is_native()), "{name}");
        }
    }

    #[test]
    fn known_unsupported_architectures_give_specific_error() {
        for name in ["mamba", "gpt2", "dots1", "rwkv7", "starcoder2"] {
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
