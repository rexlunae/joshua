//! One description of each model joshua serves or can load.
//!
//! Clients size requests from a model's limits (#260): the context one
//! request may use, the context the model was trained with, and the cap on
//! its output.  llama-server's and vLLM's model lists also carry what the
//! GGUF says (vocabulary, width, parameter count, size, file type) and the
//! model's path.  [`ModelInfo`] gathers all of it in one place, from a
//! loaded [`Engine`] ([`ModelInfo::of_engine`]) or, for a file in the model
//! directory that is not loaded, from its GGUF header alone without
//! touching a weight ([`ModelInfo::of_file`]).  [`ModelInfo::card`] renders
//! it as the [`ModelCard`] that `GET /v1/models`, `GET /models` and
//! `GET /v1/models/{id}` return, on a server and on a `joshua route`
//! coordinator, which passes its workers' cards through.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use candle_core::quantized::gguf_file::Value;
use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::gguf_ext::GgufHeader;
use crate::types::{unix_time, JoshuaModelExtras, LlamaCppModelMeta, ModelCard, ModelPermission};

/// What a GGUF header says about a model, read without touching a weight.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GgufFacts {
    /// `general.architecture`.
    pub architecture: Option<String>,
    /// `general.type`: `model`, or what a file that is not a model on its
    /// own holds (`adapter`, `mmproj`, `imatrix`).
    pub kind: Option<String>,
    /// The context the model was trained with (`<arch>.context_length`).
    pub n_ctx_train: Option<u32>,
    /// Embedding width (`<arch>.embedding_length`).
    pub n_embd: Option<u32>,
    /// Tokens in the GGUF vocabulary (`tokenizer.ggml.tokens`, else
    /// `<arch>.vocab_size`).
    pub n_vocab: Option<u32>,
    /// llama.cpp's `llama_vocab_type` for `tokenizer.ggml.model`.
    pub vocab_type: Option<u32>,
    /// Routed experts per layer (`<arch>.expert_count`); `None` for a dense
    /// model.
    pub expert_count: Option<u32>,
    /// `general.file_type` (llama.cpp's `llama_ftype`).
    pub file_type: Option<u32>,
    /// Elements across every tensor.
    pub n_params: u64,
    /// Bytes of tensor data across every tensor.
    pub tensor_bytes: u64,
    /// Bytes of tensor data in the routed experts (`*_exps` tensors); 0 for
    /// a dense model.
    pub expert_bytes: u64,
    /// The file's size in bytes.
    pub file_size: u64,
}

impl GgufFacts {
    /// The facts in `header`, read from a file of `file_size` bytes.
    pub fn from_header(header: &GgufHeader, file_size: u64) -> Self {
        let md = &header.metadata;
        let architecture = header.architecture();
        let arch = architecture
            .as_deref()
            .map(|arch| crate::gguf_meta::Meta::new(md, arch));
        let arch_u32 = |key: &str| arch.as_ref().and_then(|m| m.u32_opt(key));
        let n_vocab = match md.get("tokenizer.ggml.tokens") {
            Some(Value::Array(tokens)) => u32::try_from(tokens.len()).ok(),
            _ => arch_u32("vocab_size"),
        };
        let vocab_type = md
            .get("tokenizer.ggml.model")
            .and_then(|v| v.to_string().ok())
            .and_then(|model| vocab_type(model));
        let file_type = md
            .get("general.file_type")
            .and_then(crate::gguf_meta::as_u64)
            .and_then(|ft| u32::try_from(ft).ok());
        let kind = md
            .get("general.type")
            .and_then(|v| v.to_string().ok())
            .cloned();
        let (mut n_params, mut tensor_bytes, mut expert_bytes) = (0u64, 0u64, 0u64);
        for (name, tensor) in &header.tensors {
            let elems = tensor.elem_count();
            let size = crate::gguf_ext::type_size_bytes(tensor.dtype, elems).unwrap_or(0) as u64;
            n_params += elems as u64;
            tensor_bytes += size;
            // Routed experts: `blk.N.ffn_{gate,up,down}_exps.weight`.
            if name.contains("_exps") {
                expert_bytes += size;
            }
        }
        Self {
            n_ctx_train: arch_u32("context_length"),
            n_embd: arch_u32("embedding_length"),
            n_vocab,
            vocab_type,
            expert_count: arch_u32("expert_count").filter(|&n| n > 0),
            file_type,
            n_params,
            tensor_bytes,
            expert_bytes,
            file_size,
            architecture,
            kind,
        }
    }

    /// The facts of the GGUF at `path`.  Only the header is read, once per
    /// version of the file: the result is kept until the file's size or
    /// modification time changes.
    pub fn read(path: &Path) -> crate::Result<Self> {
        type Cache = HashMap<PathBuf, ((u64, Option<SystemTime>), GgufFacts)>;
        static CACHE: Mutex<Option<Cache>> = Mutex::new(None);
        /// Files remembered at once; a larger directory starts over.
        const CACHE_FILES: usize = 1024;

        let stat = std::fs::metadata(path)?;
        let version = (stat.len(), stat.modified().ok());
        let cached = CACHE
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .and_then(|cache| cache.get(path))
            .filter(|(seen, _)| *seen == version)
            .map(|(_, facts)| facts.clone());
        if let Some(facts) = cached {
            return Ok(facts);
        }
        let file = std::fs::File::open(path)?;
        let header = crate::gguf_ext::read_header(&mut std::io::BufReader::new(file))?;
        let facts = Self::from_header(&header, stat.len());
        let mut cache = CACHE.lock().unwrap_or_else(|p| p.into_inner());
        let cache = cache.get_or_insert_with(HashMap::new);
        if cache.len() >= CACHE_FILES {
            cache.clear();
        }
        cache.insert(path.to_path_buf(), (version, facts.clone()));
        Ok(facts)
    }

    /// Whether the file is a model on its own, rather than one a model
    /// uses (a LoRA adapter, a multimodal projector, an importance matrix):
    /// its `general.type` is `model` or absent, and its architecture is not
    /// `clip` (projectors written before `general.type`).
    pub fn is_model(&self) -> bool {
        self.kind.as_deref().is_none_or(|kind| kind == "model")
            && self.architecture.as_deref() != Some("clip")
    }

    /// The quantization as Ollama names it (`"Q4_K_M"`).
    pub fn quantization(&self) -> Option<&'static str> {
        file_type_names(self.file_type?).map(|(ollama, _)| ollama)
    }

    /// The file type as llama.cpp names it (`"Q4_K - Medium"`).
    pub fn ftype_name(&self) -> Option<&'static str> {
        file_type_names(self.file_type?).map(|(_, llama_cpp)| llama_cpp)
    }
}

/// llama.cpp's `llama_vocab_type` for a GGUF `tokenizer.ggml.model`
/// (`llama_vocab::impl::load`).
fn vocab_type(model: &str) -> Option<u32> {
    Some(match model {
        "no_vocab" | "none" => 0,
        "llama" => 1,
        "gpt2" | "hybriddna" | "whitespace" | "gemma4" => 2,
        "bert" => 3,
        "t5" => 4,
        "rwkv" => 5,
        "plamo2" => 6,
        "test" => 7,
        "plamo3" => 8,
        _ => return None,
    })
}

/// `general.file_type` (llama.cpp's `llama_ftype`) as Ollama
/// (`fs/gguf/file_type.go`) and llama.cpp (`llama_ftype_name`) name it.
fn file_type_names(file_type: u32) -> Option<(&'static str, &'static str)> {
    Some(match file_type {
        0 => ("F32", "all F32"),
        1 => ("F16", "F16"),
        2 => ("Q4_0", "Q4_0"),
        3 => ("Q4_1", "Q4_1"),
        7 => ("Q8_0", "Q8_0"),
        8 => ("Q5_0", "Q5_0"),
        9 => ("Q5_1", "Q5_1"),
        10 => ("Q2_K", "Q2_K - Medium"),
        11 => ("Q3_K_S", "Q3_K - Small"),
        12 => ("Q3_K_M", "Q3_K - Medium"),
        13 => ("Q3_K_L", "Q3_K - Large"),
        14 => ("Q4_K_S", "Q4_K - Small"),
        15 => ("Q4_K_M", "Q4_K - Medium"),
        16 => ("Q5_K_S", "Q5_K - Small"),
        17 => ("Q5_K_M", "Q5_K - Medium"),
        18 => ("Q6_K", "Q6_K"),
        19 => ("IQ2_XXS", "IQ2_XXS - 2.0625 bpw"),
        20 => ("IQ2_XS", "IQ2_XS - 2.3125 bpw"),
        21 => ("Q2_K_S", "Q2_K - Small"),
        22 => ("IQ3_XS", "IQ3_XS - 3.3 bpw"),
        23 => ("IQ3_XXS", "IQ3_XXS - 3.0625 bpw"),
        24 => ("IQ1_S", "IQ1_S - 1.5625 bpw"),
        25 => ("IQ4_NL", "IQ4_NL - 4.5 bpw"),
        26 => ("IQ3_S", "IQ3_S - 3.4375 bpw"),
        27 => ("IQ3_M", "IQ3_S mix - 3.66 bpw"),
        28 => ("IQ2_S", "IQ2_S - 2.5 bpw"),
        29 => ("IQ2_M", "IQ2_M - 2.7 bpw"),
        30 => ("IQ4_XS", "IQ4_XS - 4.25 bpw"),
        31 => ("IQ1_M", "IQ1_M - 1.75 bpw"),
        32 => ("BF16", "BF16"),
        36 => ("TQ1_0", "TQ1_0 - 1.69 bpw ternary"),
        37 => ("TQ2_0", "TQ2_0 - 2.06 bpw ternary"),
        38 => ("MXFP4_MOE", "MXFP4 MoE"),
        39 => ("NVFP4", "NVFP4"),
        40 => ("Q1_0", "Q1_0"),
        41 => ("Q2_0", "Q2_0"),
        _ => return None,
    })
}

/// A model's limits, as a client sizing a request needs them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelLimits {
    /// Most tokens, prompt and output together, one request may use: the
    /// context window the model is loaded with (`--n-ctx`), which caps a
    /// request's `context_window` (`n_ctx`, `num_ctx`).
    pub context_length: u32,
    /// The context the model was trained with (`<arch>.context_length`).
    pub max_context_length: Option<u32>,
    /// The ceiling on one request's output (`--max-output-tokens`).
    pub max_output_tokens: u32,
}

impl ModelLimits {
    /// The limits of a model loaded with a `context_length` window, trained
    /// with `max_context_length`, under an `output_cap` ceiling per request.
    pub fn new(context_length: u32, max_context_length: Option<u32>, output_cap: u32) -> Self {
        Self {
            context_length,
            max_context_length,
            max_output_tokens: output_cap,
        }
    }
}

/// Where a loaded model runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Placement {
    /// Compute backend: `cpu`, `cuda`, `metal`, `opencl`, `sycl` or `vulkan`.
    pub backend: String,
    /// Where the dense weights (attention, norms, dense FFN) run.
    pub dense: String,
    /// Where the routed experts of a mixture-of-experts model live.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experts: Option<String>,
    /// Bytes of the bounded VRAM cache holding a hot subset of host-resident
    /// experts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_cache_bytes: Option<u64>,
    /// Requests run on an NPU backend (`--npu-plugin`) when it can take them.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub npu: bool,
    /// Every forward pass runs on pipeline workers: this host only
    /// tokenizes and samples, on `backend`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub remote: bool,
    /// The pipeline workers holding the model's layers; empty for a model
    /// run here.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub workers: Vec<String>,
    /// Each worker's layer range, in worker order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stages: Vec<(usize, usize)>,
}

impl Placement {
    /// Where `engine` runs (its workers, if any, are the caller's to add).
    pub fn of_engine(engine: &Engine) -> Self {
        Self {
            dense: device_label(engine.dense_device()).to_string(),
            experts: engine
                .gguf_facts()
                .expert_count
                .map(|_| device_label(engine.expert_device()).to_string()),
            expert_cache_bytes: engine.device_expert_cache().filter(|&bytes| bytes > 0),
            npu: engine.npu_active() && !engine.remote_only(),
            remote: engine.remote_only(),
            ..Self::on(engine.device())
        }
    }

    /// A model that runs wholly on `device`.
    pub fn on(device: &candle_core::Device) -> Self {
        let label = device_label(device).to_string();
        Self {
            backend: label.clone(),
            dense: label,
            experts: None,
            expert_cache_bytes: None,
            npu: false,
            remote: false,
            workers: Vec::new(),
            stages: Vec::new(),
        }
    }
}

/// Short backend name for a candle device.
pub fn device_label(device: &candle_core::Device) -> &'static str {
    use candle_core::DeviceLocation as L;
    match device.location() {
        L::Cpu => "cpu",
        L::Cuda { .. } => "cuda",
        L::Metal { .. } => "metal",
        L::OpenCl { .. } => "opencl",
        L::Sycl { .. } => "sycl",
        L::Vulkan { .. } => "vulkan",
    }
}

/// Everything joshua reports about one model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    /// The id requests name the model by.
    pub id: String,
    /// Unix time the model was loaded (for a file that is not loaded, when
    /// it was last modified).
    pub created: u64,
    /// Whether the model is loaded.
    pub loaded: bool,
    /// The model's file.
    pub path: Option<PathBuf>,
    /// The base model of an adapter; `None` for a base model.
    pub parent: Option<String>,
    /// What its GGUF header says.
    pub facts: Option<GgufFacts>,
    /// Its limits: as loaded, or, for a file that is not loaded, as a load
    /// with the server's flags would set them.
    pub limits: Option<ModelLimits>,
    /// Where it runs, once loaded.
    pub placement: Option<Placement>,
}

impl ModelInfo {
    /// A loaded model known only by its id.
    pub fn new(id: impl Into<String>, created: u64) -> Self {
        Self {
            id: id.into(),
            created,
            loaded: true,
            path: None,
            parent: None,
            facts: None,
            limits: None,
            placement: None,
        }
    }

    /// The model `engine` serves under `id`, loaded at `created`.
    pub fn of_engine(id: &str, created: u64, engine: &Engine) -> Self {
        let mut facts = engine.gguf_facts().clone();
        // A GGUF without an embedded vocabulary: the tokenizer joshua loads.
        if facts.n_vocab.is_none() {
            facts.n_vocab = u32::try_from(engine.vocab_size()).ok();
        }
        Self {
            path: Some(engine.model_path().to_path_buf()),
            facts: Some(facts),
            limits: Some(engine.limits()),
            placement: Some(Placement::of_engine(engine)),
            ..Self::new(id, created)
        }
    }

    /// The GGUF at `path` in `manager`'s directory, not loaded, listed as
    /// `id` (see [`crate::server::ModelManager::unloaded`]).  Its limits are
    /// the ones a load with the server's flags gives it.  Reads only the
    /// file's header.
    pub fn of_file(id: &str, path: &Path, manager: &crate::server::ModelManager) -> Self {
        let facts = GgufFacts::read(path).ok();
        let created = std::fs::metadata(path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map_or_else(unix_time, |d| d.as_secs());
        Self {
            loaded: false,
            path: Some(path.to_path_buf()),
            limits: Some(manager.default_limits(path, facts.as_ref())),
            facts,
            ..Self::new(id, created)
        }
    }

    /// The model's `/v1/models` entry.
    pub fn card(&self) -> ModelCard {
        let facts = self.facts.as_ref();
        let context = self.limits.map(|l| l.context_length);
        let n_ctx_train = self
            .limits
            .and_then(|l| l.max_context_length)
            .or_else(|| facts.and_then(|f| f.n_ctx_train));
        ModelCard {
            id: self.id.clone(),
            object: "model".to_string(),
            created: self.created,
            owned_by: "joshua".to_string(),
            context_length: context,
            max_context_length: n_ctx_train,
            max_output_tokens: self.limits.map(|l| l.max_output_tokens),
            root: Some(
                self.path
                    .as_deref()
                    .map_or_else(|| self.id.clone(), |p| p.display().to_string()),
            ),
            parent: self.parent.clone(),
            max_model_len: context,
            permission: vec![ModelPermission::new(unix_time())],
            aliases: Vec::new(),
            tags: Vec::new(),
            meta: facts.map(|f| LlamaCppModelMeta {
                vocab_type: f.vocab_type,
                n_vocab: f.n_vocab,
                n_ctx: context,
                n_ctx_train,
                n_embd: f.n_embd,
                n_params: f.n_params,
                size: f.tensor_bytes,
                ftype: f.ftype_name().map(str::to_string),
            }),
            joshua: Some(JoshuaModelExtras {
                loaded: self.loaded,
                architecture: facts.and_then(|f| f.architecture.clone()),
                quantization: facts.and_then(|f| f.quantization()).map(str::to_string),
                file_size: facts.map(|f| f.file_size),
                placement: self.placement.clone(),
            }),
        }
    }
}

/// Every `.gguf` under `dir` (two directory levels deep), sorted.
pub fn discover_gguf(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect_gguf(dir, 0, &mut out);
    out.sort();
    out
}

/// The first file under `dir` that the inventory lists as a model: a GGUF that
/// is a model on its own, in [`discover_gguf`] order (two levels deep, by
/// path).
///
/// A model directory can hold more than models — adapters, projectors,
/// importance matrices — and only [`GgufFacts::is_model`] tells them apart, so
/// picking a model to start on reads the candidate headers instead of trusting
/// the extension.  `None` means the directory holds no loadable model.
pub fn first_model(dir: &Path) -> Option<PathBuf> {
    discover_gguf(dir)
        .into_iter()
        .find(|file| GgufFacts::read(file).is_ok_and(|facts| facts.is_model()))
}

fn collect_gguf(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > 2 {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_gguf(&path, depth + 1, out);
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            out.push(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_cap_is_reported_as_configured() {
        // As #261 settled: the cap is reported even when the context would
        // run out first, so a client always has both numbers.
        let capped = ModelLimits::new(16384, Some(40960), 4096);
        assert_eq!(capped.max_output_tokens, 4096);
        assert_eq!(ModelLimits::new(2048, None, 4096).max_output_tokens, 4096);
    }

    #[test]
    fn file_types_use_llama_cpp_numbering() {
        let names = |ft| file_type_names(ft).unwrap();
        assert_eq!(names(15), ("Q4_K_M", "Q4_K - Medium"));
        assert_eq!(names(7), ("Q8_0", "Q8_0"));
        // Past IQ3_S the ftype enum diverges from the ggml type ids.
        assert_eq!(names(27), ("IQ3_M", "IQ3_S mix - 3.66 bpw"));
        assert_eq!(names(32), ("BF16", "BF16"));
        assert_eq!(names(38), ("MXFP4_MOE", "MXFP4 MoE"));
        assert_eq!(file_type_names(4), None);
    }

    #[test]
    fn facts_come_from_the_header_and_its_tensor_table() {
        use crate::gguf_ext::RawTensorInfo;
        let string = |s: &str| Value::String(s.to_string());
        let header = GgufHeader {
            version: 3,
            metadata: [
                ("general.architecture", string("qwen3moe")),
                ("general.type", string("model")),
                ("general.file_type", Value::U32(15)),
                ("qwen3moe.context_length", Value::U32(40960)),
                ("qwen3moe.embedding_length", Value::U32(1024)),
                ("qwen3moe.expert_count", Value::U32(128)),
                ("tokenizer.ggml.model", string("gpt2")),
                (
                    "tokenizer.ggml.tokens",
                    Value::Array(vec![string("a"), string("b"), string("c")]),
                ),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
            tensors: [
                // 32 F32 elements, then 512 Q4_K elements: two 144-byte blocks.
                (
                    "norm",
                    RawTensorInfo {
                        dtype: 0,
                        dims: vec![4, 8],
                        offset: 0,
                    },
                ),
                (
                    "blk.0.ffn_up_exps.weight",
                    RawTensorInfo {
                        dtype: 12,
                        dims: vec![2, 256],
                        offset: 128,
                    },
                ),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
            tensor_data_offset: 4096,
        };
        let facts = GgufFacts::from_header(&header, 5000);
        assert_eq!(
            facts,
            GgufFacts {
                architecture: Some("qwen3moe".to_string()),
                kind: Some("model".to_string()),
                n_ctx_train: Some(40960),
                n_embd: Some(1024),
                n_vocab: Some(3),
                vocab_type: Some(2),
                expert_count: Some(128),
                file_type: Some(15),
                n_params: 32 + 512,
                tensor_bytes: 32 * 4 + 2 * 144,
                expert_bytes: 2 * 144,
                file_size: 5000,
            }
        );
        assert_eq!(facts.quantization(), Some("Q4_K_M"));
        assert_eq!(facts.ftype_name(), Some("Q4_K - Medium"));
        assert!(facts.is_model());
    }

    #[test]
    fn companion_files_are_not_models() {
        let facts = |architecture: &str, kind: Option<&str>| GgufFacts {
            architecture: Some(architecture.to_string()),
            kind: kind.map(str::to_string),
            ..GgufFacts::default()
        };
        // Older converters write no `general.type`.
        assert!(facts("qwen3", None).is_model());
        assert!(facts("qwen3", Some("model")).is_model());
        // llama.cpp's LoRA, projector and imatrix files.
        assert!(!facts("qwen3", Some("adapter")).is_model());
        assert!(!facts("clip", Some("mmproj")).is_model());
        assert!(!facts("qwen3", Some("imatrix")).is_model());
        // A projector written before `general.type`.
        assert!(!facts("clip", None).is_model());
    }

    #[test]
    fn vocab_types_follow_tokenizer_ggml_model() {
        assert_eq!(vocab_type("llama"), Some(1));
        assert_eq!(vocab_type("gpt2"), Some(2));
        assert_eq!(vocab_type("bert"), Some(3));
        assert_eq!(vocab_type("t5"), Some(4));
        assert_eq!(vocab_type("mystery"), None);
    }

    #[test]
    fn a_card_without_facts_keeps_upstream_shapes() {
        let card = serde_json::to_value(ModelInfo::new("m", 7).card()).unwrap();
        assert_eq!(card["id"], "m");
        assert_eq!(card["object"], "model");
        assert_eq!(card["owned_by"], "joshua");
        assert_eq!(card["root"], "m");
        assert!(card["parent"].is_null() && card.get("parent").is_some());
        assert!(card["max_model_len"].is_null());
        assert!(card.get("context_length").is_none());
        assert!(card.get("meta").is_none());
        assert_eq!(card["aliases"], serde_json::json!([]));
        assert_eq!(card["joshua"], serde_json::json!({"loaded": true}));
        let permission = &card["permission"][0];
        assert_eq!(permission["object"], "model_permission");
        assert!(permission["id"].as_str().unwrap().starts_with("modelperm-"));
        // A worker's card survives the coordinator's round trip.
        let back: ModelCard = serde_json::from_value(card.clone()).unwrap();
        assert_eq!(serde_json::to_value(back).unwrap(), card);
    }
}
