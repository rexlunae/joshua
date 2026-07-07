//! Core LLM inference engine for Joshua.
//!
//! The engine loads a GGUF model file and tokenises input using a
//! `tokenizer.json` file placed alongside the model.  Both the GGUF weights
//! and the tokenizer are loaded entirely in pure Rust — no C or C++ runtime
//! is required.
//!
//! # Memory mapping
//!
//! The GGUF file is memory-mapped (`mmap`) once when the engine is created,
//! exactly like llama.cpp's default loading strategy.  Weight data is paged
//! in lazily by the OS on first touch and stays resident in the page cache,
//! so it is shared between engine clones and across requests, and never
//! copied through a `read()` syscall path.
//!
//! The page size is selectable via [`EngineOptions::huge_pages`]: the default
//! keeps this file-backed mapping on normal pages; [`HugePages::Transparent`]
//! adds a `MADV_HUGEPAGE` hint while preserving the shared page cache; and
//! [`HugePages::Explicit`] copies the weights into an anonymous `MAP_HUGETLB`
//! mapping of a chosen size (2 MiB / 1 GiB) for guaranteed huge pages at the
//! cost of private RAM.
//!
//! # KV-cache sharing
//!
//! Finished requests park their model instance — including its populated KV
//! cache — in a small pool.  A follow-up request whose prompt extends a
//! parked instance's token history (the normal multi-turn chat pattern)
//! reuses it and prefills only the new suffix, skipping recomputation of the
//! shared prefix entirely.  Unrelated prompts reuse a pooled instance with a
//! cleared cache where the architecture supports it, or build a fresh
//! instance from the mapping (no disk I/O after first load).  Requests never
//! observe each other's cache contents: an instance is owned by exactly one
//! request at a time, and reuse requires an exact token-prefix match.
//!
//! The engine auto-detects the model architecture from the GGUF
//! `general.architecture` metadata and dispatches to the correct candle
//! quantized loader.  Supported architectures:
//!
//! | `general.architecture` | Model family
//! |------------------------|-------------
//! | `llama`                | Llama 1/2/3, Mistral, Mixtral, TinyLlama, SmolLM, Yi, …
//! | `gemma` / `gemma2` / `gemma3` / `gemma-embedding` | Gemma 1/2/3
//! | `glm4`                 | GLM-4
//! | `lfm2`                 | LFM2
//! | `phi2`                 | Phi-1, Phi-1.5, Phi-2
//! | `phi3`                 | Phi-3 / Phi-3.5
//! | `qwen2`                | Qwen1.5 / Qwen2 / Qwen2.5
//! | `qwen3`                | Qwen3
//! | `qwen3moe`             | Qwen3 MoE
//!
//! Any other architecture in llama.cpp's registry is recognised by name and
//! rejected with an error explaining that no pure-Rust loader exists yet.
//!
//! # Model directory layout
//!
//! ```text
//! my-model/
//! ├── model.gguf          ← quantised weights (any GGUF-compatible architecture)
//! └── tokenizer.json      ← HuggingFace tokenizer (download from the model card)
//! ```
//!
//! You can also point directly at a `.gguf` file; `tokenizer.json` is then
//! looked up in the same directory.

use std::fs::File;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use memmap2::Mmap;
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;
use tokenizers::Tokenizer;

use crate::embedding::EmbeddingModel;
use crate::model::{Architecture, QuantizedModel};
use crate::npu::{NpuBackend, NpuSession};
use crate::template::ChatTemplate;

use crate::error::{JoshuaError, Result};
use crate::types::{ChatMessage, GenerationOptions, Tool, UsageInfo};

// ─── Mmap configuration ─────────────────────────────────────────────────────

/// Explicit huge-page size for [`HugePages::Explicit`].
///
/// The page-bits values match `MAP_HUGE_*` in `mmap(2)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PageSize {
    /// The system's default huge-page size (from `/proc/meminfo`, usually
    /// 2 MiB).  Corresponds to `MAP_HUGETLB` without a size selector.
    #[default]
    Default,
    /// 2 MiB "large" pages (`MAP_HUGE_2MB`).
    TwoMiB,
    /// 1 GiB "huge" pages (`MAP_HUGE_1GB`); needs 1 GiB pages preallocated.
    OneGiB,
}

impl PageSize {
    /// `(page-bits for MmapOptions::huge, page size in bytes)`.
    fn params(self) -> (Option<u8>, usize) {
        match self {
            Self::Default => (None, default_hugepage_bytes()),
            Self::TwoMiB => (Some(21), 2 * 1024 * 1024),
            Self::OneGiB => (Some(30), 1024 * 1024 * 1024),
        }
    }
}

/// How the model file is backed by physical memory.
///
/// The default keeps the file-backed mmap Joshua has always used; the other
/// variants trade that for huge pages, which cut TLB misses on large models.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum HugePages {
    /// Normal page size; the model stays file-backed via `mmap` and is
    /// shared through the OS page cache (default).
    #[default]
    Off,
    /// Keep the file-backed mmap but ask the kernel to promote it to
    /// transparent huge pages (`MADV_HUGEPAGE`).
    ///
    /// Best-effort and portable — it preserves the shared-page-cache model
    /// and silently does nothing if the kernel can't honour it (no size
    /// control; the kernel picks the THP size, normally 2 MiB).  Linux only.
    Transparent,
    /// Load the model into an **anonymous** mapping backed by explicit
    /// huge pages of the given size (`MAP_HUGETLB`).
    ///
    /// This guarantees the page size but copies the weights into private
    /// RAM: the shared page cache is given up, load touches the whole file
    /// once, and the hugepage pool must be preallocated (e.g.
    /// `sysctl vm.nr_hugepages=…` or `hugeadm`).  Linux only; on other
    /// platforms it falls back to a normal file mapping with a warning.
    Explicit(PageSize),
}

/// Construction options for [`Engine`].
///
/// Use [`Engine::with_options`] for full control; [`Engine::new`] and
/// [`Engine::with_n_ctx`] are convenience wrappers over the defaults.
#[derive(Debug, Clone, Default)]
pub struct EngineOptions {
    /// Context-window size in tokens (0 selects the 4096 default).
    pub n_ctx: u32,
    /// Physical-memory backing strategy for the model mapping.
    pub huge_pages: HugePages,
}

impl EngineOptions {
    /// Default options with an explicit context-window size.
    pub fn with_n_ctx(n_ctx: u32) -> Self {
        Self {
            n_ctx,
            ..Self::default()
        }
    }

    /// Select the huge-page strategy.
    pub fn huge_pages(mut self, huge_pages: HugePages) -> Self {
        self.huge_pages = huge_pages;
        self
    }
}

// ─── Engine ───────────────────────────────────────────────────────────────────

/// The Joshua inference engine.
///
/// Instances are cheaply clonable (the tokenizer is `Arc`-wrapped) and are
/// `Send + Sync`, so a single `Arc<Engine>` can be shared across threads.
pub struct Engine {
    /// Path to the `.gguf` file.
    model_path: PathBuf,
    /// The GGUF file memory-mapped into the process address space.
    ///
    /// All model loads read weights directly out of this mapping, so the OS
    /// page cache backs every request and engine clones share the same
    /// physical pages.
    mmap: Arc<Mmap>,
    /// Stateless tokenizer, shared across all inference calls.
    tokenizer: Arc<Tokenizer>,
    /// EOS token IDs derived from the GGUF metadata and common special tokens.
    eos_token_ids: Vec<u32>,
    /// The model's chat template from GGUF metadata, if it ships one.
    chat_template: Option<ChatTemplate>,
    /// Lazily built embedding model (stateless, shared by all embed calls).
    embed_model: Mutex<Option<Arc<EmbeddingModel>>>,
    /// Pool of loaded model instances with warm KV caches.
    ///
    /// A finished request parks its model here together with the exact token
    /// sequence its KV cache holds.  A later request whose prompt extends
    /// that sequence (the normal multi-turn chat pattern) picks the instance
    /// up and prefills only the new suffix.
    model_cache: Mutex<Vec<CachedModel>>,
    /// Number of requests that continued from a cached KV prefix.
    kv_reuses: AtomicU64,
    /// Optional NPU backend with its circuit breaker.
    npu: Option<NpuState>,
    /// Number of generations/embeddings currently executing.
    ///
    /// Each in-flight request holds a full model instance (weights + KV
    /// cache), so this is capped at `max_concurrency` to bound peak memory;
    /// requests over the cap are rejected rather than piling up unbounded
    /// heavyweight model loads.
    in_flight: AtomicUsize,
    /// Maximum concurrent generations/embeddings.
    max_concurrency: usize,
    /// Upper bound on tokens generated per request, regardless of the
    /// client-supplied `max_tokens`.
    max_output_tokens: u32,
    /// Stem of the model file (used as the model identifier in API responses).
    model_name: String,
    /// Context-window size in tokens.
    n_ctx: u32,
    /// Compute device: CUDA or Metal when built with the matching feature
    /// (falling back to CPU if unavailable at runtime), CPU otherwise.
    device: Device,
}

// `PathBuf`, `Arc<Mmap>`, `Arc<Tokenizer>`, `Vec<u32>`, `String`, `u32`,
// `Mutex<…>`, and `AtomicU64` are all `Send + Sync`, so Engine is
// automatically `Send + Sync`.

/// Maximum number of idle model instances kept warm in the pool.
///
/// Each instance holds the (quantized) weights plus its KV cache, so this
/// bounds memory: two instances cover the common "one active conversation
/// plus one concurrent request" pattern without tripling residency.
const MAX_CACHED_MODELS: usize = 2;

/// Consecutive NPU failures before the backend is disabled for the rest of
/// the engine's lifetime (all requests then run on the candle path).
const NPU_MAX_FAILURES: u32 = 3;

/// Default ceiling on tokens generated per request (independent of the
/// client-supplied `max_tokens`), bounding single-request CPU/time cost.
const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 4096;

/// RAII permit for one in-flight generation/embedding.
///
/// Increments the engine's in-flight counter on acquisition (rejecting once
/// `max_concurrency` is reached) and decrements it on drop, so the count is
/// released even if generation errors or panics.
struct InFlightGuard<'a> {
    counter: &'a AtomicUsize,
}

impl<'a> InFlightGuard<'a> {
    fn acquire(counter: &'a AtomicUsize, max: usize) -> Result<Self> {
        // Reserve a slot optimistically, then bail out if we blew the cap.
        let prev = counter.fetch_add(1, Ordering::AcqRel);
        if prev >= max {
            counter.fetch_sub(1, Ordering::AcqRel);
            return Err(JoshuaError::Overloaded(format!(
                "at capacity ({max} concurrent requests); retry shortly"
            )));
        }
        Ok(Self { counter })
    }
}

impl Drop for InFlightGuard<'_> {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::AcqRel);
    }
}

/// A parked generation session whose state holds exactly `tokens`.
struct CachedModel {
    session: GenSession,
    tokens: Vec<u32>,
}

/// A generation session: either a candle model on CPU/GPU or a vendor NPU
/// session behind the [`crate::npu`] plugin interface.  Both follow the same
/// contract: feed tokens at an absolute position, get last-token logits.
enum GenSession {
    Candle(Box<QuantizedModel>),
    Npu(Box<dyn NpuSession>),
}

impl GenSession {
    /// Feed `tokens` at absolute position `pos`, returning last-token logits.
    fn forward_tokens(&mut self, tokens: &[u32], pos: usize, device: &Device) -> Result<Vec<f32>> {
        match self {
            Self::Candle(model) => {
                let input = Tensor::new(tokens, device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| JoshuaError::Inference(e.to_string()))?;
                let logits = model
                    .forward(&input, pos)
                    .map_err(|e| JoshuaError::Inference(e.to_string()))?;
                squeeze_batch_logits(&logits)
            }
            Self::Npu(session) => session
                .forward(tokens, pos)
                .map_err(JoshuaError::Inference),
        }
    }

    /// Clear internal state for reuse with an unrelated prompt.
    ///
    /// Returns `false` when the session cannot be reset and must be dropped.
    fn clear_state(&mut self) -> bool {
        match self {
            Self::Candle(model) => model.clear_kv_cache(),
            Self::Npu(session) => session.reset(),
        }
    }

    fn is_npu(&self) -> bool {
        matches!(self, Self::Npu(_))
    }

    /// Whether this session can prefill multimodal prompts.
    fn supports_media(&self) -> bool {
        match self {
            Self::Candle(_) => false,
            Self::Npu(session) => session.supports_media(),
        }
    }

    /// Tokenise-and-prefill a multimodal prompt (NPU sessions only).
    fn media_prefill(&mut self, prompt: &str, images: &[Vec<u8>]) -> Result<(usize, Vec<f32>)> {
        match self {
            Self::Candle(_) => Err(JoshuaError::InvalidRequest(
                "the candle path does not support multimodal input".to_string(),
            )),
            Self::Npu(session) => session
                .media_prefill(prompt, images)
                .map_err(JoshuaError::Inference),
        }
    }
}

/// Result of a decode loop.
struct DecodeOutcome {
    response: String,
    n_decoded: u32,
    /// Tokens actually fed to the model during decode (KV-state delta).
    fed_tokens: Vec<u32>,
    decode_tps: f64,
}

/// NPU backend state: the backend plus its circuit breaker.
struct NpuState {
    backend: Arc<dyn NpuBackend>,
    failures: AtomicU32,
    disabled: AtomicBool,
}

impl NpuState {
    /// Record a failure; disable the backend once the limit is reached.
    fn record_failure(&self, backend_name: &str, error: &str) {
        let failures = self.failures.fetch_add(1, Ordering::Relaxed) + 1;
        tracing::warn!("NPU backend {backend_name} failure {failures}/{NPU_MAX_FAILURES}: {error}");
        if failures >= NPU_MAX_FAILURES && !self.disabled.swap(true, Ordering::Relaxed) {
            tracing::error!(
                "NPU backend {backend_name} disabled after {failures} failures; \
                 all requests will use the candle CPU/GPU path"
            );
        }
    }

    fn usable(&self) -> bool {
        !self.disabled.load(Ordering::Relaxed)
    }
}

impl Engine {
    /// Load a GGUF model using a 4 096-token context window.
    ///
    /// `model_path` can be either the path to a `.gguf` file or a directory
    /// that contains one.  A `tokenizer.json` must exist in the same directory
    /// as the `.gguf` file.
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        Self::with_options(model_path, EngineOptions::default())
    }

    /// Load a GGUF model with a custom context-window size.
    pub fn with_n_ctx(model_path: impl AsRef<Path>, n_ctx: u32) -> Result<Self> {
        Self::with_options(model_path, EngineOptions::with_n_ctx(n_ctx))
    }

    /// Load a GGUF model with full [`EngineOptions`] (context size and the
    /// huge-page backing strategy).
    pub fn with_options(model_path: impl AsRef<Path>, options: EngineOptions) -> Result<Self> {
        let n_ctx = if options.n_ctx == 0 { 4096 } else { options.n_ctx };
        let raw_path = model_path.as_ref().to_path_buf();

        // Resolve the actual .gguf file path.
        let gguf_path = if raw_path.is_dir() {
            find_gguf_in_dir(&raw_path)?
        } else {
            raw_path
        };

        let model_name = gguf_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        tracing::info!("Loading model from {:?}", gguf_path);

        // Locate tokenizer.json in the same directory as the GGUF file.
        let model_dir = gguf_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(JoshuaError::ModelLoad(format!(
                "tokenizer.json not found at {:?}.\n\
                 Place it alongside the .gguf file \
                 (download from the model's HuggingFace repository).",
                tokenizer_path
            )));
        }

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| JoshuaError::ModelLoad(format!("tokenizer load failed: {e}")))?;

        // Map the GGUF file into memory using the configured backing.
        let mmap = map_model(&gguf_path, options.huge_pages)?;

        // Read GGUF metadata once to validate the architecture up front and
        // extract EOS token IDs.
        let gguf = gguf_file::Content::read(&mut Cursor::new(&mmap[..]))
            .map_err(|e| JoshuaError::ModelLoad(format!("GGUF read failed: {e}")))?;

        let arch = Architecture::detect(&gguf.metadata).map_err(JoshuaError::ModelLoad)?;

        let eos_token_ids = extract_eos_ids(&gguf, &tokenizer);
        let chat_template = extract_chat_template(&gguf, &tokenizer);
        let device = Self::default_device();

        tracing::info!(
            "Model '{}' ready (arch={}, ctx={}, eos_ids={:?}, chat_template={}, device={:?})",
            model_name,
            arch.display_name(),
            n_ctx,
            eos_token_ids,
            if chat_template.is_some() {
                "from GGUF"
            } else {
                "ChatML fallback"
            },
            device
        );

        // Default the concurrency cap to the machine's parallelism: running
        // more heavyweight generations at once than the CPU can serve gains
        // no throughput and only multiplies peak memory.  Operators tune it
        // with `with_max_concurrency`.
        let max_concurrency = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Ok(Self {
            model_path: gguf_path,
            mmap: Arc::new(mmap),
            tokenizer: Arc::new(tokenizer),
            eos_token_ids,
            chat_template,
            embed_model: Mutex::new(None),
            model_cache: Mutex::new(Vec::new()),
            kv_reuses: AtomicU64::new(0),
            npu: None,
            in_flight: AtomicUsize::new(0),
            max_concurrency,
            max_output_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
            model_name,
            n_ctx,
            device,
        })
    }

    /// Set the maximum number of concurrent generations/embeddings.
    ///
    /// Requests beyond this cap are rejected with [`JoshuaError::Overloaded`]
    /// (HTTP 503) rather than queued, bounding peak memory from concurrent
    /// model instances.  Values below 1 are treated as 1.
    pub fn with_max_concurrency(mut self, max: usize) -> Self {
        self.max_concurrency = max.max(1);
        self
    }

    /// Set the hard ceiling on tokens generated per request, applied on top
    /// of the client-supplied `max_tokens`.  Values below 1 are treated as 1.
    pub fn with_max_output_tokens(mut self, max: u32) -> Self {
        self.max_output_tokens = max.max(1);
        self
    }

    /// Pick the compute device.
    ///
    /// With the `cuda` or `metal` cargo feature enabled this tries the GPU
    /// first and falls back to CPU (with a warning) when no usable device is
    /// present at runtime.  Without those features it is always CPU.
    fn default_device() -> Device {
        #[cfg(feature = "cuda")]
        {
            match Device::new_cuda(0) {
                Ok(device) => return device,
                Err(e) => tracing::warn!("CUDA unavailable, falling back to CPU: {e}"),
            }
        }
        #[cfg(feature = "metal")]
        {
            match Device::new_metal(0) {
                Ok(device) => return device,
                Err(e) => tracing::warn!("Metal unavailable, falling back to CPU: {e}"),
            }
        }
        Device::Cpu
    }

    /// The stem of the loaded model file name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Absolute path of the loaded `.gguf` file.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Context-window size in tokens.
    pub fn n_ctx(&self) -> u32 {
        self.n_ctx
    }

    // ─── Prompt formatting ────────────────────────────────────────────────────

    /// Whether the loaded GGUF ships its own chat template.
    pub fn has_chat_template(&self) -> bool {
        self.chat_template.is_some()
    }

    /// Format messages as a ChatML prompt and append the assistant turn header.
    ///
    /// When tools are supplied, a Hermes-style system block advertising them
    /// is prepended — the same convention our tool-call parser understands.
    fn format_chatml_prompt(messages: &[ChatMessage], tools: Option<&[Tool]>) -> String {
        let mut prompt = String::new();
        if let Some(tools) = tools.filter(|t| !t.is_empty()) {
            prompt.push_str(
                "<|im_start|>system\n\
                 # Tools\n\n\
                 You may call one or more functions to assist with the user query.\n\n\
                 You are provided with function signatures within <tools></tools> XML tags:\n\
                 <tools>\n",
            );
            for tool in tools {
                if let Ok(json) = serde_json::to_string(tool) {
                    prompt.push_str(&json);
                    prompt.push('\n');
                }
            }
            prompt.push_str(
                "</tools>\n\n\
                 For each function call, return a json object with function name and arguments \
                 within <tool_call></tool_call> XML tags:\n\
                 <tool_call>\n\
                 {\"name\": <function-name>, \"arguments\": <args-json-object>}\n\
                 </tool_call><|im_end|>\n",
            );
        }
        for msg in messages {
            prompt.push_str("<|im_start|>");
            prompt.push_str(&msg.role);
            prompt.push('\n');
            prompt.push_str(&msg.content);
            prompt.push_str("<|im_end|>\n");
        }
        prompt.push_str("<|im_start|>assistant\n");
        prompt
    }

    /// Format messages into the prompt the model was trained on.
    ///
    /// Uses the GGUF-embedded chat template when present; otherwise (or if the
    /// template fails to render) falls back to ChatML.  Returns the prompt and
    /// whether the tokenizer should still add special tokens: a rendered chat
    /// template already contains every special token (including BOS), so
    /// adding them again would duplicate BOS.
    fn format_prompt(&self, messages: &[ChatMessage], tools: Option<&[Tool]>) -> (String, bool) {
        if let Some(template) = &self.chat_template {
            match template.render(messages, tools) {
                Ok(prompt) => return (prompt, false),
                Err(e) => {
                    tracing::warn!("GGUF chat template unusable, falling back to ChatML: {e}");
                }
            }
        }
        (Self::format_chatml_prompt(messages, tools), true)
    }

    // ─── Completion ───────────────────────────────────────────────────────────

    /// Run a chat completion.
    ///
    /// Messages are formatted with the model's own chat template when the
    /// GGUF provides one (`tokenizer.chat_template` metadata), with ChatML as
    /// the fallback.  Returns `(generated_text, usage, prefill_tps, decode_tps)`.
    pub fn complete(
        &self,
        messages: &[ChatMessage],
        options: &GenerationOptions,
    ) -> Result<(String, UsageInfo, f64, f64)> {
        self.complete_chat(messages, None, options)
    }

    /// Run a chat completion with optional tool definitions.
    ///
    /// Tools are exposed to the chat template as the standard `tools`
    /// variable so the model is instructed how to emit calls; parse the
    /// generated text with [`crate::tools::parse_tool_calls`] to extract
    /// them.
    pub fn complete_chat(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        options: &GenerationOptions,
    ) -> Result<(String, UsageInfo, f64, f64)> {
        // Multimodal branch: messages carrying images go through a
        // media-capable NPU/llama.cpp plugin session.
        if messages.iter().any(|m| m.images.as_ref().is_some_and(|i| !i.is_empty())) {
            let (marked_messages, images) = resolve_message_media(messages)?;
            let (prompt, _) = self.format_prompt(&marked_messages, tools);
            return self.complete_media(&prompt, &images, options);
        }
        let (prompt, add_special_tokens) = self.format_prompt(messages, tools);
        self.complete_with(&prompt, add_special_tokens, options)
    }

    /// Run a multimodal completion: the plugin tokenises and prefills the
    /// marked prompt together with the media, then decode proceeds normally.
    fn complete_media(
        &self,
        prompt: &str,
        images: &[Vec<u8>],
        options: &GenerationOptions,
    ) -> Result<(String, UsageInfo, f64, f64)> {
        // Bound concurrent heavyweight generations before doing any work.
        let _permit = InFlightGuard::acquire(&self.in_flight, self.max_concurrency)?;
        // The plugin owns tokenisation, so the prompt length is unknown here;
        // clamp only to the server ceiling. The decode loop's in-context
        // guard bounds the total length.
        let options = &self.clamp_options(options, None);

        // Acquire an NPU session (never by token prefix — the plugin owns
        // tokenisation here, so no history to match).
        let (mut session, _) = self.acquire_session(&[], true)?;
        if !session.supports_media() {
            // Repool the (text-capable) session before failing.
            if session.clear_state() {
                self.release_model(session, Vec::new());
            }
            return Err(JoshuaError::InvalidRequest(
                "this request contains images, which require a multimodal NPU plugin — \
                 run with --npu-plugin pointing at the llama.cpp adapter built with an \
                 mmproj (JOSHUA_LLAMA_MMPROJ) or another media-capable plugin"
                    .to_string(),
            ));
        }

        let prefill_start = Instant::now();
        let result = session
            .media_prefill(prompt, images)
            .and_then(|(n_past, logits)| {
                let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
                let outcome = self.decode_loop(&mut session, logits, n_past, &[], options)?;
                Ok((n_past, prefill_ms, outcome))
            });

        match result {
            Ok((n_past, prefill_ms, outcome)) => {
                // The session's token history is plugin-internal; repool only
                // after a clean reset.
                if session.clear_state() {
                    self.release_model(session, Vec::new());
                }
                let prefill_tps = if prefill_ms > 0.0 {
                    n_past as f64 / (prefill_ms / 1000.0)
                } else {
                    0.0
                };
                let usage = UsageInfo {
                    // Positions consumed by text tokens + media embeddings.
                    prompt_tokens: n_past as u32,
                    completion_tokens: outcome.n_decoded,
                    total_tokens: n_past as u32 + outcome.n_decoded,
                };
                Ok((outcome.response, usage, prefill_tps, outcome.decode_tps))
            }
            Err(e) => {
                if let Some(npu) = &self.npu {
                    npu.record_failure(&npu.backend.name(), &e.to_string());
                }
                // No candle fallback exists for vision — propagate.
                Err(e)
            }
        }
    }

    /// Run completion from an arbitrary raw prompt string.
    pub fn complete_raw(
        &self,
        prompt: &str,
        options: &GenerationOptions,
    ) -> Result<(String, UsageInfo, f64, f64)> {
        self.complete_with(prompt, true, options)
    }

    /// Shared completion path.  `add_special_tokens` controls whether the
    /// tokenizer wraps the prompt with its special tokens (disabled for
    /// template-rendered prompts, which already include them).
    /// Clamp a request's generation length to the server's `max_output_tokens`
    /// ceiling and, when the prompt length is known, the remaining context
    /// window — so a client-supplied `max_tokens` can't force unbounded work.
    fn clamp_options(&self, options: &GenerationOptions, prompt_len: Option<usize>) -> GenerationOptions {
        let mut clamped = options.clone();
        let mut cap = clamped.max_tokens.min(self.max_output_tokens);
        if let Some(n_prompt) = prompt_len {
            let remaining = (self.n_ctx as usize).saturating_sub(n_prompt).max(1) as u32;
            cap = cap.min(remaining);
        }
        clamped.max_tokens = cap;
        clamped
    }

    fn complete_with(
        &self,
        prompt: &str,
        add_special_tokens: bool,
        options: &GenerationOptions,
    ) -> Result<(String, UsageInfo, f64, f64)> {
        // Bound concurrent heavyweight generations before doing any work.
        let _permit = InFlightGuard::acquire(&self.in_flight, self.max_concurrency)?;

        // ── Tokenise ─────────────────────────────────────────────────────────
        let encoding = self
            .tokenizer
            .encode(prompt, add_special_tokens)
            .map_err(|e| JoshuaError::Tokenization(e.to_string()))?;
        let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
        let n_prompt = prompt_tokens.len();

        if n_prompt >= self.n_ctx as usize {
            return Err(JoshuaError::PromptTooLong(n_prompt, self.n_ctx as usize));
        }

        // Clamp the client-supplied generation length to the server ceiling
        // and the remaining context window.
        let options = &self.clamp_options(options, Some(n_prompt));

        // ── Acquire a session, generate, retry on CPU if the NPU fails ──────
        // Prefer a pooled instance whose state already covers a prefix of
        // this prompt; fall back to a reset instance or a fresh load.
        let (mut session, n_reused) = self.acquire_session(&prompt_tokens, true)?;
        let was_npu = session.is_npu();

        let result = self.run_generation(&mut session, &prompt_tokens, n_reused, options);
        match result {
            Ok((response, usage, prefill_tps, decode_tps, kv_tokens)) => {
                // Park the instance for reuse by a follow-up request.
                self.release_model(session, kv_tokens);
                Ok((response, usage, prefill_tps, decode_tps))
            }
            Err(e) if was_npu => {
                // Count the failure (possibly disabling the backend), drop
                // the session unless it can prove a clean reset, and retry
                // the whole request once on the candle path.
                if let Some(npu) = &self.npu {
                    npu.record_failure(&npu.backend.name(), &e.to_string());
                }
                if session.clear_state() {
                    self.release_model(session, Vec::new());
                }
                tracing::warn!("Retrying request on the candle path after NPU failure: {e}");
                let (mut session, n_reused) = self.acquire_session(&prompt_tokens, false)?;
                match self.run_generation(&mut session, &prompt_tokens, n_reused, options) {
                    Ok((response, usage, prefill_tps, decode_tps, kv_tokens)) => {
                        self.release_model(session, kv_tokens);
                        Ok((response, usage, prefill_tps, decode_tps))
                    }
                    Err(e) => {
                        if session.clear_state() {
                            self.release_model(session, Vec::new());
                        }
                        Err(e)
                    }
                }
            }
            Err(e) => {
                // The KV cache may be partially updated at the failure point;
                // a cleared cache is fully consistent, so keep the (expensive
                // to reload) weights warm where the architecture allows it.
                if session.clear_state() {
                    self.release_model(session, Vec::new());
                }
                Err(e)
            }
        }
    }

    /// Prefill + decode on an acquired session.
    ///
    /// Returns the generated text, usage, throughput figures, and the exact
    /// token sequence now held in the session's state.
    fn run_generation(
        &self,
        model: &mut GenSession,
        prompt_tokens: &[u32],
        n_reused: usize,
        options: &GenerationOptions,
    ) -> Result<(String, UsageInfo, f64, f64, Vec<u32>)> {
        let n_prompt = prompt_tokens.len();
        let new_tokens = &prompt_tokens[n_reused..];

        // Every token fed to the model so far — i.e. the exact contents of
        // its KV cache.  Returned to the pool with the model afterwards.
        let mut kv_tokens = prompt_tokens.to_vec();

        // ── Prefill ───────────────────────────────────────────────────────────
        // Process the not-yet-cached prompt tokens in a single forward pass,
        // starting right after the reused KV prefix.
        let prefill_start = Instant::now();
        let logits_vec = model.forward_tokens(new_tokens, n_reused, &self.device)?;
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        // ── Repetition-penalty history ────────────────────────────────────────
        let outcome = self.decode_loop(model, logits_vec, n_prompt, prompt_tokens, options)?;
        kv_tokens.extend_from_slice(&outcome.fed_tokens);

        // Throughput reflects the tokens actually processed in the prefill
        // window: with a reused KV prefix that is only the new suffix.
        let n_prefilled = new_tokens.len();
        let prefill_tps = if prefill_ms > 0.0 {
            n_prefilled as f64 / (prefill_ms / 1000.0)
        } else {
            0.0
        };

        tracing::debug!(
            prompt_tokens = n_prompt,
            prefill_tokens = n_prefilled,
            reused_tokens = n_reused,
            prefill_tps,
            decode_tokens = outcome.n_decoded,
            decode_tps = outcome.decode_tps,
            "Completion finished"
        );

        let usage = UsageInfo {
            prompt_tokens: n_prompt as u32,
            completion_tokens: outcome.n_decoded,
            total_tokens: n_prompt as u32 + outcome.n_decoded,
        };

        Ok((
            outcome.response,
            usage,
            prefill_tps,
            outcome.decode_tps,
            kv_tokens,
        ))
    }

    /// Greedy/sampled token generation from an initial logit vector.
    ///
    /// `start_pos` is the absolute position of the next token to feed
    /// (`prompt length` for text prompts, `n_past` after a multimodal
    /// prefill).  `penalty_seed` primes the repetition-penalty window
    /// (empty when prompt tokens are unknown, e.g. multimodal prefill).
    fn decode_loop(
        &self,
        model: &mut GenSession,
        mut logits_vec: Vec<f32>,
        start_pos: usize,
        penalty_seed: &[u32],
        options: &GenerationOptions,
    ) -> Result<DecodeOutcome> {
        // Seed the recent-token window with the tail of the prompt (up to 64 tokens).
        const REP_WINDOW: usize = 64;
        let mut recent_tokens: Vec<u32> = penalty_seed
            .iter()
            .rev()
            .take(REP_WINDOW)
            .copied()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        let mut rng = thread_rng();
        let mut response = String::new();
        let mut fed_tokens: Vec<u32> = Vec::new();
        let mut n_decoded: u32 = 0;
        let mut n_cur = start_pos;
        let decode_start = Instant::now();

        loop {
            if n_decoded >= options.max_tokens {
                break;
            }
            // Never generate past the context window, regardless of
            // max_tokens — bounds KV-cache growth and matches the RoPE tables.
            if n_cur >= self.n_ctx as usize {
                break;
            }

            let next_token = sample_token(&logits_vec, options, &mut rng, &recent_tokens)?;

            if self.eos_token_ids.contains(&next_token) {
                break;
            }

            // Decode the new token to text.
            let piece = self
                .tokenizer
                .decode(&[next_token], false)
                .map_err(|e| JoshuaError::Inference(e.to_string()))?;
            response.push_str(&piece);
            n_decoded += 1;

            // Maintain sliding-window token history for repetition penalty.
            if recent_tokens.len() >= REP_WINDOW {
                recent_tokens.remove(0);
            }
            recent_tokens.push(next_token);

            if Self::check_stop_sequences(&mut response, &options.stop_sequences) {
                break;
            }

            // Single-token decode step.
            logits_vec = model.forward_tokens(&[next_token], n_cur, &self.device)?;
            fed_tokens.push(next_token);
            n_cur += 1;
        }

        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        let decode_tps = if decode_ms > 0.0 && n_decoded > 0 {
            n_decoded as f64 / (decode_ms / 1000.0)
        } else {
            0.0
        };

        Ok(DecodeOutcome {
            response,
            n_decoded,
            fed_tokens,
            decode_tps,
        })
    }

    // ─── Embeddings ───────────────────────────────────────────────────────────

    /// Compute dense embeddings for one or more texts.
    ///
    /// Runs a single hidden-state forward pass per text and pools according
    /// to the model's GGUF `pooling_type` metadata (mean by default, or
    /// CLS / last-token for models converted with an explicit pooling head,
    /// e.g. Qwen3-Embedding).  Vectors are L2-normalised.
    ///
    /// Supported architectures: llama (e5-mistral, SFR-Embedding, …), qwen2
    /// (gte-Qwen2), and qwen3 (Qwen3-Embedding).
    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(self.embed_with_usage(texts)?.0)
    }

    /// Like [`Engine::embed`], additionally returning the total number of
    /// input tokens processed.
    pub fn embed_with_usage(&self, texts: &[String]) -> Result<(Vec<Vec<f32>>, u32)> {
        // Embeddings also load/hold a model instance — bound concurrency.
        let _permit = InFlightGuard::acquire(&self.in_flight, self.max_concurrency)?;
        let model = self.embedding_model()?;
        let mut vectors = Vec::with_capacity(texts.len());
        let mut total_tokens: u32 = 0;
        for text in texts {
            let encoding = self
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| JoshuaError::Tokenization(e.to_string()))?;
            let tokens = encoding.get_ids();
            if tokens.len() >= self.n_ctx as usize {
                return Err(JoshuaError::PromptTooLong(tokens.len(), self.n_ctx as usize));
            }
            total_tokens += tokens.len() as u32;
            let vector = model
                .embed_tokens(tokens)
                .map_err(|e| JoshuaError::Inference(e.to_string()))?;
            vectors.push(vector);
        }
        Ok((vectors, total_tokens))
    }

    /// Get (building on first use) the shared embedding model.
    fn embedding_model(&self) -> Result<Arc<EmbeddingModel>> {
        // Recover from poisoning: the slot holds an `Arc<EmbeddingModel>`
        // (immutable once built), so a prior panic can't have left it
        // inconsistent, and failing permanently would break all embeddings.
        let mut slot = self
            .embed_model
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(model) = slot.as_ref() {
            return Ok(Arc::clone(model));
        }
        let mut cursor = Cursor::new(&self.mmap[..]);
        let gguf = gguf_file::Content::read(&mut cursor)
            .map_err(|e| JoshuaError::ModelLoad(format!("GGUF read failed: {e}")))?;
        let model = EmbeddingModel::from_gguf(gguf, &mut cursor, &self.device)
            .map_err(|e| JoshuaError::InvalidRequest(e.to_string()))?;
        let model = Arc::new(model);
        *slot = Some(Arc::clone(&model));
        Ok(model)
    }

    // ─── Private helpers ─────────────────────────────────────────────────────

    /// Number of requests so far that continued from a cached KV prefix.
    pub fn kv_reuse_count(&self) -> u64 {
        self.kv_reuses.load(Ordering::Relaxed)
    }

    /// Route generation through an NPU backend (see [`crate::npu`]).
    ///
    /// Generation requests try the backend first and transparently fall back
    /// to the candle CPU/GPU path when session creation or a forward pass
    /// fails; after [`NPU_MAX_FAILURES`] failures the backend is disabled
    /// for the engine's lifetime.  Embeddings always run on candle.
    pub fn with_npu_backend(mut self, backend: Arc<dyn NpuBackend>) -> Self {
        tracing::info!("NPU backend configured: {}", backend.name());
        self.npu = Some(NpuState {
            backend,
            failures: AtomicU32::new(0),
            disabled: AtomicBool::new(false),
        });
        self
    }

    /// Whether an NPU backend is configured and not (yet) disabled by the
    /// circuit breaker.
    pub fn npu_active(&self) -> bool {
        self.npu.as_ref().is_some_and(|n| n.usable())
    }

    /// Get a generation session ready to prefill `prompt_tokens`.
    ///
    /// When an NPU backend is configured, usable, and `allow_npu` is set,
    /// the session runs there; otherwise on the candle CPU/GPU path.  A
    /// failed NPU session creation counts against the circuit breaker and
    /// falls back to candle.
    ///
    /// Returns the session and how many leading prompt tokens its state
    /// already covers.  Preference order within the chosen kind:
    ///
    /// 1. a pooled instance whose fed-token history is a strict prefix of
    ///    the prompt (longest match wins) — only the suffix needs prefill;
    /// 2. a pooled instance whose state can be cleared — skips re-creating
    ///    the session;
    /// 3. a fresh session (NPU) or a fresh instance from the mmap (candle).
    fn acquire_session(
        &self,
        prompt_tokens: &[u32],
        allow_npu: bool,
    ) -> Result<(GenSession, usize)> {
        let want_npu = allow_npu && self.npu.as_ref().is_some_and(|n| n.usable());

        {
            let mut pool = self.model_pool();
            // Only reuse sessions of the kind this request will run on —
            // mixing kinds mid-conversation would splice numerically
            // different logits into one generation.
            let best = pool
                .iter()
                .enumerate()
                .filter(|(_, c)| {
                    c.session.is_npu() == want_npu
                        && c.tokens.len() < prompt_tokens.len()
                        && prompt_tokens.starts_with(&c.tokens)
                })
                .max_by_key(|(_, c)| c.tokens.len())
                .map(|(i, _)| i);
            if let Some(i) = best {
                let cached = pool.swap_remove(i);
                self.kv_reuses.fetch_add(1, Ordering::Relaxed);
                tracing::debug!(
                    reused_tokens = cached.tokens.len(),
                    prompt_tokens = prompt_tokens.len(),
                    npu = want_npu,
                    "Continuing from cached KV prefix"
                );
                return Ok((cached.session, cached.tokens.len()));
            }
            let resettable = pool.iter().position(|c| c.session.is_npu() == want_npu);
            if let Some(i) = resettable {
                let mut cached = pool.swap_remove(i);
                if cached.session.clear_state() {
                    tracing::debug!(npu = want_npu, "Reusing pooled session with cleared state");
                    return Ok((cached.session, 0));
                }
                // Reset failed (e.g. dead shim): drop it and fall through.
            }
        }

        if want_npu {
            let npu = self.npu.as_ref().expect("checked above");
            match npu.backend.create_session(&self.model_path, self.n_ctx) {
                Ok(session) => return Ok((GenSession::Npu(session), 0)),
                Err(e) => {
                    npu.record_failure(&npu.backend.name(), &e);
                    tracing::warn!("NPU session creation failed, using candle path: {e}");
                }
            }
        }

        Ok((GenSession::Candle(Box::new(self.load_model()?)), 0))
    }

    /// Lock the warm-model pool, recovering the guard if a previous holder
    /// panicked.  A poisoned lock must not permanently disable reuse: the
    /// cached instances are plain data, and silently treating poison as
    /// "no pool" would force a fresh full model load on every subsequent
    /// request (a memory-amplifying, silent degradation).
    fn model_pool(&self) -> std::sync::MutexGuard<'_, Vec<CachedModel>> {
        self.model_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Return a finished session (state = `tokens`) to the pool.
    fn release_model(&self, session: GenSession, tokens: Vec<u32>) {
        {
            let mut pool = self.model_pool();
            pool.push(CachedModel { session, tokens });
            // Evict oldest beyond the cap.
            while pool.len() > MAX_CACHED_MODELS {
                pool.remove(0);
            }
        }
    }

    /// Load a [`QuantizedModel`] from the memory-mapped GGUF file —
    /// architecture is auto-detected from the GGUF metadata.
    ///
    /// The instance starts with an empty KV cache.  Weights are read straight
    /// out of the shared mmap, so reloads involve no disk I/O.
    fn load_model(&self) -> Result<QuantizedModel> {
        let mut cursor = Cursor::new(&self.mmap[..]);
        let gguf = gguf_file::Content::read(&mut cursor)
            .map_err(|e| JoshuaError::ModelLoad(format!("GGUF read failed: {e}")))?;
        QuantizedModel::from_gguf(gguf, &mut cursor, &self.device)
            .map_err(|e| JoshuaError::ModelLoad(format!("model init failed: {e}")))
    }

    /// Scan `response` for any configured stop sequence and truncate it.
    fn check_stop_sequences(response: &mut String, stops: &[String]) -> bool {
        for stop in stops {
            if stop.is_empty() {
                continue;
            }
            if response.ends_with(stop.as_str()) {
                response.truncate(response.len() - stop.len());
                return true;
            }
        }
        false
    }
}

// ─── Media helpers ────────────────────────────────────────────────────────────

/// Resolve message-attached images to raw bytes and inject one media marker
/// per image into the owning message's content (marker order == byte order),
/// following llama.cpp's `mtmd` prompt convention.
fn resolve_message_media(messages: &[ChatMessage]) -> Result<(Vec<ChatMessage>, Vec<Vec<u8>>)> {
    let mut marked = messages.to_vec();
    let mut images = Vec::new();
    for msg in &mut marked {
        let Some(attached) = msg.images.take() else {
            continue;
        };
        let mut markers = String::new();
        for source in &attached {
            images.push(load_image_bytes(source)?);
            markers.push_str(crate::npu::MEDIA_MARKER);
            markers.push('\n');
        }
        msg.content = format!("{markers}{}", msg.content);
    }
    Ok((marked, images))
}

/// Maximum size of a decoded inline image, as a defence-in-depth cap on
/// top of the HTTP body limit.  16 MiB comfortably covers any real photo.
const MAX_IMAGE_BYTES: usize = 16 * 1024 * 1024;

/// Decode image bytes from a base64 `data:` URL.
///
/// Only `data:` URLs are accepted.  Filesystem paths are deliberately **not**
/// read: the image field of a chat message is attacker-controlled over the
/// HTTP API, so honouring a path there would let an unauthenticated client
/// make the server open arbitrary local files (information disclosure, plus
/// denial of service via `/dev/zero`, FIFOs, or huge files).  Remote URLs are
/// not fetched either (SSRF); callers must inline the image as a data URL,
/// which is exactly what OpenAI-compatible vision clients already send.
fn load_image_bytes(source: &str) -> Result<Vec<u8>> {
    let Some(rest) = source.strip_prefix("data:") else {
        return Err(JoshuaError::InvalidRequest(
            "image sources must be inline base64 `data:` URLs; \
             filesystem paths and remote URLs are not accepted"
                .to_string(),
        ));
    };
    let b64 = rest.split_once("base64,").map(|(_, b)| b).ok_or_else(|| {
        JoshuaError::InvalidRequest("only base64 data: URLs are supported".to_string())
    })?;
    use base64::Engine as _;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .map_err(|e| JoshuaError::InvalidRequest(format!("invalid base64 image data: {e}")))?;
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(JoshuaError::InvalidRequest(format!(
            "image is {} bytes, exceeding the {MAX_IMAGE_BYTES}-byte limit",
            bytes.len()
        )));
    }
    Ok(bytes)
}

// ─── GGUF / tokenizer helpers ─────────────────────────────────────────────────

// ─── Model mapping ────────────────────────────────────────────────────────────

/// Map the model file into memory according to the huge-page strategy.
///
/// SAFETY (file-backed variants): the mapping is only undefined behaviour if
/// the file is truncated or rewritten while mapped.  Model files are treated
/// as immutable once downloaded, matching llama.cpp's own mmap usage.
fn map_model(path: &Path, huge: HugePages) -> Result<Mmap> {
    let file = File::open(path)?;

    // Explicit huge pages use an anonymous copy; handle separately.
    if let HugePages::Explicit(size) = huge {
        return map_model_hugetlb(path, &file, size);
    }

    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| JoshuaError::ModelLoad(format!("mmap of GGUF file failed: {e}")))?;

    // Weight tensors are consumed in file order during a load, so tell the
    // kernel to read ahead aggressively.  Best effort only.
    #[cfg(unix)]
    let _ = mmap.advise(memmap2::Advice::Sequential);

    if huge == HugePages::Transparent {
        #[cfg(target_os = "linux")]
        match mmap.advise(memmap2::Advice::HugePage) {
            Ok(()) => tracing::info!("requested transparent huge pages for the model mapping"),
            Err(e) => {
                tracing::warn!("transparent huge pages unavailable; using normal pages: {e}")
            }
        }
        #[cfg(not(target_os = "linux"))]
        tracing::warn!("transparent huge pages are Linux-only; using normal pages");
    }

    Ok(mmap)
}

/// Load the model into an anonymous mapping backed by explicit huge pages.
#[cfg(target_os = "linux")]
fn map_model_hugetlb(path: &Path, file: &File, size: PageSize) -> Result<Mmap> {
    let len = file.metadata()?.len() as usize;
    let (page_bits, page_len) = size.params();
    // MAP_HUGETLB requires the mapping length to be a multiple of the page
    // size; round up (the tail is zero-filled and unused).
    let mapped_len = len
        .div_ceil(page_len)
        .checked_mul(page_len)
        .ok_or_else(|| JoshuaError::ModelLoad("model too large for huge-page mapping".into()))?;

    let mut anon = memmap2::MmapOptions::new()
        .len(mapped_len)
        .huge(page_bits)
        .map_anon()
        .map_err(|e| {
            JoshuaError::ModelLoad(format!(
                "could not allocate {} MiB of {}-byte huge pages — is the pool configured \
                 (e.g. `sysctl vm.nr_hugepages`)?: {e}",
                mapped_len / (1024 * 1024),
                page_len
            ))
        })?;

    // Copy the model bytes into the huge-page-backed region.
    File::open(path)?
        .read_exact(&mut anon[..len])
        .map_err(|e| JoshuaError::ModelLoad(format!("reading model into huge pages failed: {e}")))?;

    let mmap = anon
        .make_read_only()
        .map_err(|e| JoshuaError::ModelLoad(format!("freezing huge-page mapping failed: {e}")))?;
    tracing::info!(
        "loaded model into {} MiB of explicit huge pages ({}-byte pages, anonymous — \
         not shared through the page cache)",
        mapped_len / (1024 * 1024),
        page_len
    );
    Ok(mmap)
}

/// Non-Linux fallback: explicit huge pages are unsupported, so map the file
/// normally with a warning.
#[cfg(not(target_os = "linux"))]
fn map_model_hugetlb(_path: &Path, file: &File, _size: PageSize) -> Result<Mmap> {
    tracing::warn!("explicit huge pages are Linux-only; using a normal file mapping");
    unsafe { Mmap::map(file) }
        .map_err(|e| JoshuaError::ModelLoad(format!("mmap of GGUF file failed: {e}")))
}

/// The system's default huge-page size in bytes, read from `/proc/meminfo`
/// (`Hugepagesize:`), falling back to 2 MiB.
fn default_hugepage_bytes() -> usize {
    const FALLBACK: usize = 2 * 1024 * 1024;
    let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") else {
        return FALLBACK;
    };
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("Hugepagesize:") {
            // Format: "Hugepagesize:    2048 kB".
            let mut it = rest.split_whitespace();
            if let (Some(kb), Some(_unit)) = (it.next(), it.next()) {
                if let Ok(kb) = kb.parse::<usize>() {
                    return kb * 1024;
                }
            }
        }
    }
    FALLBACK
}

/// Walk `dir` and return the first `.gguf` file found.
fn find_gguf_in_dir(dir: &Path) -> Result<PathBuf> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
            return Ok(path);
        }
    }
    Err(JoshuaError::ModelLoad(format!(
        "No .gguf file found in {:?}",
        dir
    )))
}

/// Read a token ID from GGUF metadata and decode it to its string form.
fn token_str_from_metadata(
    gguf: &gguf_file::Content,
    key: &str,
    tokenizer: &Tokenizer,
) -> Option<String> {
    let id = match gguf.metadata.get(key)? {
        gguf_file::Value::U32(id) => *id,
        gguf_file::Value::I32(id) => *id as u32,
        gguf_file::Value::U64(id) => *id as u32,
        _ => return None,
    };
    tokenizer.id_to_token(id)
}

/// Extract the model's chat template from GGUF metadata, if present.
///
/// llama.cpp's converters store the HuggingFace chat template verbatim under
/// `tokenizer.chat_template`.  BOS/EOS strings are resolved from their token
/// IDs so the template can interpolate them.
fn extract_chat_template(gguf: &gguf_file::Content, tokenizer: &Tokenizer) -> Option<ChatTemplate> {
    let source = gguf
        .metadata
        .get("tokenizer.chat_template")?
        .to_string()
        .ok()?
        .clone();
    if source.trim().is_empty() {
        return None;
    }
    let bos = token_str_from_metadata(gguf, "tokenizer.ggml.bos_token_id", tokenizer)
        .unwrap_or_default();
    let eos = token_str_from_metadata(gguf, "tokenizer.ggml.eos_token_id", tokenizer)
        .unwrap_or_default();
    Some(ChatTemplate::new(source, bos, eos))
}

/// Derive EOS token IDs from GGUF metadata and well-known special token strings.
fn extract_eos_ids(gguf: &gguf_file::Content, tokenizer: &Tokenizer) -> Vec<u32> {
    let mut ids: Vec<u32> = Vec::new();

    // Primary: explicit EOS from GGUF metadata.
    let eos_key = "tokenizer.ggml.eos_token_id";
    match gguf.metadata.get(eos_key) {
        Some(gguf_file::Value::U32(id)) => ids.push(*id),
        Some(gguf_file::Value::I32(id)) => ids.push(*id as u32),
        Some(gguf_file::Value::U64(id)) => ids.push(*id as u32),
        _ => {}
    }

    // Fallback: common EOS token strings for popular model families.
    for token_str in &[
        "</s>",
        "<|endoftext|>",
        "<|im_end|>",
        "<end_of_turn>",
        "<eos>",
        "<|eot_id|>",
        "<|end|>",
    ] {
        if let Some(id) = tokenizer.token_to_id(token_str) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }

    ids
}

// ─── Tensor helpers ───────────────────────────────────────────────────────────

/// Convert a `[1, vocab_size]` logits tensor to a flat `Vec<f32>`.
///
/// `ModelWeights::forward()` always returns shape `[batch, vocab_size]`
/// because it selects the last sequence position internally.
fn squeeze_batch_logits(logits: &Tensor) -> Result<Vec<f32>> {
    // Remove the batch dimension (index 0) to get [vocab_size].
    logits
        .squeeze(0)
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| JoshuaError::Inference(e.to_string()))
}

// ─── Sampling ────────────────────────────────────────────────────────────────

/// Sample the next token from a raw logit vector.
///
/// Implements repetition penalty, temperature scaling, top-k filtering,
/// min-p filtering, top-p (nucleus) filtering, and weighted random sampling,
/// all in pure Rust.
fn sample_token(
    logits: &[f32],
    opts: &GenerationOptions,
    rng: &mut impl rand::Rng,
    recent_tokens: &[u32],
) -> Result<u32> {
    if logits.is_empty() {
        return Ok(0);
    }

    // ── Repetition penalty ────────────────────────────────────────────────────
    // For tokens present in the recent window, divide positive logits and
    // multiply negative logits by `repetition_penalty` (> 1.0 discourages
    // repetition; 1.0 is a no-op).  Applied before temperature so the penalty
    // is independent of the temperature scale.
    let logits: Vec<f32> = if opts.repetition_penalty != 1.0 {
        let mut v = logits.to_vec();
        for &token in recent_tokens {
            if let Some(l) = v.get_mut(token as usize) {
                if *l > 0.0 {
                    *l /= opts.repetition_penalty;
                } else {
                    *l *= opts.repetition_penalty;
                }
            }
        }
        v
    } else {
        logits.to_vec()
    };

    // ── Greedy ────────────────────────────────────────────────────────────────
    if opts.temperature <= 0.0 {
        return Ok(logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0));
    }

    // ── Temperature scaling ───────────────────────────────────────────────────
    let inv_temp = 1.0_f32 / opts.temperature;
    // Subtract max for numerical stability before exp.
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits
        .iter()
        .map(|&l| ((l - max_logit) * inv_temp).exp())
        .collect();

    // ── Top-k ─────────────────────────────────────────────────────────────────
    let k = opts.top_k as usize;
    if k > 0 && k < probs.len() {
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|(_, a), (_, b)| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });
        for &(idx, _) in indexed.iter().skip(k) {
            probs[idx] = 0.0;
        }
    }

    // ── Min-p ─────────────────────────────────────────────────────────────────
    if opts.min_p > 0.0 {
        let max_p = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let threshold = max_p * opts.min_p;
        for p in &mut probs {
            if *p < threshold {
                *p = 0.0;
            }
        }
    }

    // ── Top-p (nucleus) ───────────────────────────────────────────────────────
    if opts.top_p < 1.0 && opts.top_p > 0.0 {
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            let mut sorted_idx: Vec<usize> = (0..probs.len()).collect();
            sorted_idx.sort_unstable_by(|&a, &b| {
                probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut cumsum = 0.0_f32;
            let mut cut_from = probs.len();
            for (rank, &idx) in sorted_idx.iter().enumerate() {
                cumsum += probs[idx] / sum;
                if cumsum > opts.top_p {
                    cut_from = rank + 1;
                    break;
                }
            }
            for &idx in sorted_idx.iter().skip(cut_from) {
                probs[idx] = 0.0;
            }
        }
    }

    // ── Normalise & sample ────────────────────────────────────────────────────
    let total: f32 = probs.iter().sum();
    if total <= 0.0 {
        // Fallback: greedy from original (penalty-adjusted) logits.
        return Ok(logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0));
    }

    for p in &mut probs {
        *p /= total;
    }

    let dist =
        WeightedIndex::new(&probs).map_err(|e| JoshuaError::Inference(e.to_string()))?;
    Ok(dist.sample(rng) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_size_params_are_correct() {
        assert_eq!(PageSize::TwoMiB.params(), (Some(21), 2 * 1024 * 1024));
        assert_eq!(PageSize::OneGiB.params(), (Some(30), 1024 * 1024 * 1024));
        // System default: no size selector, and a sane power-of-two byte size.
        let (bits, bytes) = PageSize::Default.params();
        assert_eq!(bits, None);
        assert!(bytes >= 4096 && bytes.is_power_of_two(), "got {bytes}");
    }

    #[test]
    fn default_hugepage_bytes_is_sane() {
        let bytes = default_hugepage_bytes();
        assert!(bytes >= 2 * 1024 * 1024 && bytes.is_power_of_two(), "got {bytes}");
    }

    #[test]
    fn engine_options_builder() {
        let o = EngineOptions::with_n_ctx(2048).huge_pages(HugePages::Transparent);
        assert_eq!(o.n_ctx, 2048);
        assert_eq!(o.huge_pages, HugePages::Transparent);
        assert_eq!(EngineOptions::default().huge_pages, HugePages::Off);
    }

    #[test]
    fn in_flight_guard_caps_and_releases() {
        let counter = AtomicUsize::new(0);
        // Fill to the cap of 2.
        let a = InFlightGuard::acquire(&counter, 2).expect("first permit");
        let b = InFlightGuard::acquire(&counter, 2).expect("second permit");
        // Third is rejected as Overloaded, and the counter is not left inflated.
        match InFlightGuard::acquire(&counter, 2) {
            Err(JoshuaError::Overloaded(_)) => {}
            Err(e) => panic!("expected Overloaded, got {e:?}"),
            Ok(_) => panic!("expected Overloaded, got a permit"),
        }
        assert_eq!(counter.load(Ordering::Acquire), 2);
        // Dropping a permit frees a slot for the next request.
        drop(a);
        let _c = InFlightGuard::acquire(&counter, 2).expect("permit after release");
        drop(b);
        drop(_c);
        assert_eq!(counter.load(Ordering::Acquire), 0);
    }

    #[test]
    fn image_sources_must_be_data_urls() {
        // A valid base64 data URL decodes.
        let ok = load_image_bytes("data:image/png;base64,AQID").expect("data url");
        assert_eq!(ok, vec![1, 2, 3]);

        // Filesystem paths are refused without any read attempt — a real
        // local file (this source tree) must not be opened.
        let err = load_image_bytes("/etc/passwd").unwrap_err();
        assert!(matches!(err, JoshuaError::InvalidRequest(_)));
        assert!(err.to_string().contains("data:"), "got: {err}");
        assert!(load_image_bytes("src/engine.rs").is_err());

        // Remote URLs are not fetched (no SSRF).
        assert!(load_image_bytes("http://169.254.169.254/").is_err());
        assert!(load_image_bytes("https://example.com/x.png").is_err());
    }
}

