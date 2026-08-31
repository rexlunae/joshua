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
//! exactly like llama.cpp's default loading strategy.  Weight data lives in
//! the OS page cache, so it is shared between engine clones and across
//! requests and never copied through a `read()` syscall path.
//!
//! How much of the model is resident up front is a choice:
//!
//! * [`EngineOptions::prefetch_whole_model`] issues a best-effort
//!   `MADV_WILLNEED` over the whole mapping at load, so a model that fits in
//!   RAM is fully resident in the page cache before the first request and
//!   inference never re-reads weights from disk.  This is the CLI default
//!   whenever the model file fits in RAM.
//! * Without it, nothing is read until inference touches a weight: pages
//!   fault in on first use and the kernel evicts them under memory pressure
//!   like any clean file-backed page.  The mapping is never hinted
//!   `MADV_SEQUENTIAL` — that would mean "free after use" and evict weight
//!   pages right after each token, which is the opposite of what an engine
//!   that re-reads every weight on every token wants.
//!
//! The page size is selectable via [`EngineOptions::huge_pages`]: the default
//! keeps this file-backed mapping on normal pages; [`HugePages::Transparent`]
//! adds a `MADV_HUGEPAGE` hint (2 MiB pages where the kernel can manage it —
//! the CLI default on Linux) while preserving the shared page cache; and
//! [`HugePages::Explicit`] copies the weights into an anonymous `MAP_HUGETLB`
//! mapping of a chosen size (2 MiB / 1 GiB) for guaranteed huge pages at the
//! cost of private RAM.  File-backed huge pages are Linux-only: macOS maps
//! files on its 16 KiB base pages and has no file-backed superpage API.
//!
//! For a sparse mixture-of-experts model far larger than RAM, the access
//! pattern is bimodal and neither of the two whole-mapping strategies above is
//! right: the dense weights are touched on every token, while routed experts
//! are touched sparsely.  [`EngineOptions::pin_hot_weights`] prefetches
//! the dense ranges (`MADV_WILLNEED`) at load and advises `MADV_RANDOM` on the
//! expert ranges so they page in on demand without evicting the hot set;
//! [`EngineOptions::mlock_hot_weights`] additionally locks the dense ranges
//! into RAM for a hard residency guarantee.  The CLI turns this on by default
//! when the model file is larger than RAM.
//!
//! Because that mapping is the whole loading strategy, the file is checked
//! before it is mapped: a model that is really a gzip/zstd/… stream, or one
//! the filesystem stores compressed, cannot be paged in usefully (see
//! [`crate::compression`]).  It is reported as a warning by default and as a
//! load error when the caller asked for mapping explicitly via
//! [`EngineOptions::mmap`].
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

/// How firmly the caller is asking for the model to be memory-mapped.
///
/// Joshua always maps the model file; this only decides how loudly it
/// complains when the file cannot usefully be mapped — because it is a
/// compression container rather than raw GGUF, or because the filesystem
/// stores it compressed (see [`crate::compression`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MmapMode {
    /// Mapping is the implicit default, so a file that maps badly is reported
    /// as a warning and the load continues (default).
    #[default]
    Auto,
    /// The caller explicitly asked for `mmap`, so a file that cannot be mapped
    /// usefully is an error: silently giving them a mapping that decompresses
    /// on every page fault would defeat the point of asking.
    Required,
}

/// How strictly [`EngineOptions::mlock_hot_weights`] must pin the hot set.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MlockMode {
    /// No `mlock(2)`; the hot set is only prefetched (advisory).
    #[default]
    Off,
    /// Lock the hot set; if `RLIMIT_MEMLOCK` is too small, warn once up front
    /// and degrade to advisory pinning (the load still succeeds).
    On,
    /// Lock the hot set; if `RLIMIT_MEMLOCK` is too small, fail the load so a
    /// deployment that must guarantee residency cannot silently lose it.
    Required,
}

/// Compute backend for the engine.
///
/// [`ComputeBackend::Auto`] (the default) picks the best backend this build
/// was compiled with — CUDA first when the `cuda` feature is on, then Metal
/// when the `metal` feature is on, else CPU — and falls back to CPU with a
/// warning when the compiled-in backend is unavailable at runtime.  An
/// explicit [`ComputeBackend::Metal`] or [`ComputeBackend::Cuda`] request
/// instead fails the load when the backend is not compiled in or cannot be
/// initialised, so a wrong `--device` flag is never silently ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComputeBackend {
    /// Best backend compiled in (CUDA > Metal > CPU), falling back to CPU.
    #[default]
    Auto,
    /// CPU inference (always available).
    Cpu,
    /// Apple Metal.  Requires the `metal` cargo feature and a Metal-capable
    /// macOS machine.
    Metal,
    /// NVIDIA CUDA.  Requires the `cuda` cargo feature and a CUDA toolkit.
    Cuda,
}

/// Construction options for [`Engine`].
///
/// Use [`Engine::with_options`] for full control; [`Engine::new`] and
/// [`Engine::with_n_ctx`] are convenience wrappers over the defaults.
#[derive(Debug, Clone, Default)]
pub struct EngineOptions {
    /// Context-window size in tokens (0 selects the 4096 default).
    pub n_ctx: u32,
    /// Compute backend to run inference on.  See [`ComputeBackend`].
    pub backend: ComputeBackend,
    /// Physical-memory backing strategy for the model mapping.
    pub huge_pages: HugePages,
    /// Optimise the mapping for models far larger than RAM.
    ///
    /// For a sparse mixture-of-experts model that does not fit in RAM, a
    /// given token touches a handful of experts and whole-file readahead
    /// drags in hundreds of megabytes that are immediately evicted.  Setting
    /// this advises `MADV_RANDOM` on the whole mapping, so pages fault in
    /// only as weights are genuinely touched and clean pages stay evictable.
    ///
    /// Without it the mapping gets no blanket hint at all (the kernel's
    /// default readahead and normal page-cache retention apply), which is
    /// right for models that fit in RAM — see
    /// [`EngineOptions::prefetch_whole_model`] for warming those in at load.
    /// This does not conflict with `prefetch_whole_model`: the explicit
    /// whole-file prefetch wins and the random hint is skipped.
    ///
    /// Only meaningful for architectures whose loaders borrow from the mapping
    /// (see [`crate::mmap_tensor`]); the readahead hint never affects
    /// correctness.
    ///
    /// This does conflict with [`HugePages::Explicit`], which copies the whole
    /// model into an anonymous huge-page region up front — the opposite of
    /// paging in on demand, and impossible for a model larger than RAM.  When
    /// both are requested this one wins and the huge-page request is ignored
    /// with a warning; [`HugePages::Transparent`] has no such conflict, since
    /// it keeps the mapping file-backed.
    pub lazy_weights: bool,
    /// Whether memory mapping was explicitly requested.
    ///
    /// The mapping itself is unconditional either way; setting this to
    /// [`MmapMode::Required`] turns "this file cannot usefully be mapped" from
    /// a warning into a load error.
    pub mmap: MmapMode,
    /// Prefetch the whole model file into the OS page cache at load.
    ///
    /// Issues a best-effort `MADV_WILLNEED` over the entire mapping right
    /// after it is created, so a model that fits in RAM is fully resident
    /// before the first request: inference never re-reads weights from disk,
    /// and engine clones all share the same cached pages.  The prefetch is
    /// advisory — under memory pressure the kernel evicts clean pages and the
    /// next access re-faults them, exactly as if nothing had been prefetched.
    ///
    /// Right for models whose file fits comfortably in RAM; wasteful for
    /// models far larger than RAM, where the whole file could never be
    /// resident and the eager read only thrashes the cache (use
    /// [`EngineOptions::pin_hot_weights`] there instead — the CLI auto-picks
    /// between the two from the model size vs RAM).
    ///
    /// Redundant for [`HugePages::Explicit`], which copies the whole model
    /// into anonymous RAM up front; the request is then ignored with a
    /// warning.  On non-Unix platforms there is no `madvise` and the request
    /// is a no-op.
    pub prefetch_whole_model: bool,
    /// Prefetch the always-touched weights at load and advise random access
    /// on routed-expert weights.
    ///
    /// A mixture-of-experts model far larger than RAM has two very different
    /// access patterns.  A small "dense" set — embeddings, norms, attention,
    /// routers, shared experts, indexer/compressor, output — is touched on
    /// every token, while the routed experts are touched sparsely (a token
    /// routes through a handful of the 256 per layer).  Plain mmap leaves both
    /// to the page cache: the dense set faults in on first use, and the
    /// experts fault in per touch (the per-range `MADV_RANDOM` advice here
    /// stops whole-file readahead from dragging them in wholesale).
    ///
    /// With this set, the dense ranges are prefetched (`MADV_WILLNEED`) so the
    /// per-token working set is resident before the first request, and the
    /// expert ranges get `MADV_RANDOM` so sparse access does not evict it.  The
    /// blanket hint is skipped in favour of these per-range hints.  The
    /// prefetch is best-effort (the kernel decides); combine with
    /// [`EngineOptions::mlock_hot_weights`] for a hard guarantee.
    ///
    /// Meaningless for [`HugePages::Explicit`], which copies the whole model
    /// into anonymous RAM up front; the request is then ignored with a warning.
    pub pin_hot_weights: bool,
    /// Lock the always-touched weight ranges into RAM with `mlock(2)`.
    ///
    /// Same dense/expert split as [`EngineOptions::pin_hot_weights`]: the dense
    /// ranges are locked (which both faults them in and keeps them resident no
    /// matter what the page cache does), and expert ranges still get
    /// `MADV_RANDOM`.  Implies the pinning advice split even when
    /// `pin_hot_weights` is unset.
    ///
    /// Requires the process memlock limit to cover the dense set — a few GiB on
    /// typical MoE models.  The limit is checked against the hot-set size
    /// before any `mlock` call: [`MlockMode::On`] warns once and degrades to
    /// advisory pinning when it is too low, [`MlockMode::Required`] fails the
    /// load.  Raise it with `LimitMEMLOCK=infinity` (systemd) or
    /// `ulimit -l unlimited`; on a systemd **user** session the hard limit is
    /// inherited from the login session, so a unit's `LimitMEMLOCK` alone is
    /// silently capped — apply `sudo prlimit --pid <user manager> --memlock=-1:-1`
    /// for a live fix, or re-login.
    pub mlock_hot_weights: MlockMode,
    /// Keep the `N` most frequently routed experts resident (a
    /// routing-frequency LRU cache).
    ///
    /// Sparse MoE models far larger than RAM (DeepSeek-V2/V3/V4,
    /// Qwen3-MoE, …) demand-fault their routed-expert weights from the
    /// mapping on every token; routing is temporally local, so a small
    /// fraction of experts carries most of the traffic.  When this is
    /// non-zero, the engine records which experts each token routes through
    /// and, every [`crate::hot_experts::REFRESH_STEPS`] decode steps,
    /// re-selects the `N` most-used experts (recency as the tie-break)
    /// and issues a best-effort `MADV_WILLNEED` on their weight pages, so the
    /// hot set stays resident in the page cache instead of faulting from disk
    /// on every step.  Advisory: under memory pressure the kernel may still
    /// evict clean pages, so size the budget below the free RAM.
    ///
    /// `0` disables the cache.  Wired for the joshua-native MoE loaders
    /// (`deepseek2`, `deepseek4`, `qwen3moe`), which carry per-expert
    /// prefetch handles; other architectures — including Mixtral through the
    /// vendored candle `llama` loader — ignore the setting.
    pub pin_hot_experts: usize,
}

impl EngineOptions {
    /// Default options with an explicit context-window size.
    pub fn with_n_ctx(n_ctx: u32) -> Self {
        Self {
            n_ctx,
            ..Self::default()
        }
    }

    /// Select the compute backend.  See [`ComputeBackend`].
    pub fn backend(mut self, backend: ComputeBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Select the huge-page strategy.
    pub fn huge_pages(mut self, huge_pages: HugePages) -> Self {
        self.huge_pages = huge_pages;
        self
    }

    /// Optimise the mapping for a model much larger than RAM.
    pub fn lazy_weights(mut self, lazy: bool) -> Self {
        self.lazy_weights = lazy;
        self
    }

    /// Ask for memory mapping explicitly, making an unmappable model file an
    /// error instead of a warning.
    pub fn mmap(mut self, mmap: MmapMode) -> Self {
        self.mmap = mmap;
        self
    }

    /// Prefetch the whole model into the OS page cache at load.  See
    /// [`EngineOptions::prefetch_whole_model`].
    pub fn prefetch_whole_model(mut self, prefetch: bool) -> Self {
        self.prefetch_whole_model = prefetch;
        self
    }

    /// Prefetch the always-touched weights at load and advise random access on
    /// routed experts.  See [`EngineOptions::pin_hot_weights`].
    pub fn pin_hot_weights(mut self, pin: bool) -> Self {
        self.pin_hot_weights = pin;
        self
    }

    /// Lock the always-touched weight ranges into RAM.  See
    /// [`EngineOptions::mlock_hot_weights`] and [`MlockMode`].
    pub fn mlock_hot_weights(mut self, mode: MlockMode) -> Self {
        self.mlock_hot_weights = mode;
        self
    }

    /// Keep the `n` most frequently routed experts resident.  See
    /// [`EngineOptions::pin_hot_experts`].
    pub fn pin_hot_experts(mut self, n: usize) -> Self {
        self.pin_hot_experts = n;
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
    /// The model file itself, kept so the deepseek4 prefill can run a
    /// layer-ahead `pread` prefetch thread against it.  `None` when the model
    /// was loaded into anonymous huge pages (nothing to prefetch).
    model_file: Option<Arc<File>>,
    /// Stateless tokenizer, shared across all inference calls.
    tokenizer: Arc<Tokenizer>,
    /// Whether the tokenizer uses a byte-level decoder (`ByteLevel`).
    ///
    /// Byte-level BPE tokenizers (DeepSeek-V2/V3, Kimi-K2, Qwen, ...) store
    /// raw bytes in their vocab and can split a multi-byte UTF-8 character
    /// across two tokens — a lone byte such as `0xE4` is invalid UTF-8 on its
    /// own.  Those tokenizers need whole-buffer decoding so the byte state
    /// carries across tokens; decoder-less/word-level tokenizers must instead
    /// be decoded token-by-token (batch decoding them would insert spaces).
    byte_level_decode: bool,
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
    /// Number of requests that continued from a cached KV prefix recovered
    /// across an *edit* of the conversation — the prompt shares only a
    /// prefix with the cached history (agent harnesses truncate or replace
    /// middle blocks), so the session state was rewound to that prefix
    /// instead of being cleared.  A subset of [`Engine::kv_reuse_count`].
    kv_edit_reuses: AtomicU64,
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
    /// Routing-frequency hot-expert cache budget (see
    /// [`EngineOptions::pin_hot_experts`]).
    pin_hot_experts: usize,
    /// Compute device: CUDA or Metal when built with the matching feature
    /// (falling back to CPU if unavailable at runtime), CPU otherwise.
    device: Device,
    /// Why the candle path cannot load this model, if it cannot.
    ///
    /// `Engine` construction succeeds even for architectures candle has no
    /// loader for (e.g. `deepseek4`) so an NPU backend configured afterwards
    /// via [`Engine::with_npu_backend`] can still serve the model.  The error
    /// is surfaced only when the candle path is actually needed — see
    /// [`Engine::load_model`].
    arch_error: Option<String>,
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
            Self::Npu(session) => session.forward(tokens, pos).map_err(JoshuaError::Inference),
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

    /// Whether this session's KV state can be truncated to a prefix length
    /// at all (see [`GenSession::truncate_to`]).
    fn supports_truncate(&self) -> bool {
        match self {
            Self::Candle(model) => model.supports_kv_truncate(),
            Self::Npu(_) => false,
        }
    }

    /// Keep only the first `keep` fed tokens of the session's KV state.
    ///
    /// This is what makes *edited-context* prefix reuse sound: positions
    /// `[0..keep)` were produced by exactly the tokens both conversations
    /// share, so they stay valid; everything after is recomputed when the
    /// caller prefills from absolute position `keep`.  Returns `false` when
    /// the architecture cannot truncate or the operation failed — callers
    /// fall back to the plain clear/fresh-load paths.
    fn truncate_to(&mut self, keep: usize) -> bool {
        match self {
            Self::Candle(model) => match model.truncate_kv_cache(keep) {
                Ok(supported) => supported,
                Err(e) => {
                    tracing::debug!(error = %e, keep, "KV truncation failed");
                    false
                }
            },
            Self::Npu(_) => false,
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

/// Whether the tokenizer at `path` uses a byte-level decoder.
///
/// Byte-level BPE tokenizers (DeepSeek-V2/V3, Kimi-K2, Qwen, ...) map raw
/// bytes through their own table and can split a multi-byte UTF-8 sequence
/// across two tokens, so generated text must be decoded from the whole
/// accumulated token buffer rather than one token at a time.  Detected from
/// the `decoder.type` in `tokenizer.json`; everything else (word-level,
/// decoder-less, BPE with explicit decoders) decodes token-by-token.
fn tokenizer_is_byte_level(path: &std::path::Path) -> Result<bool, JoshuaError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| JoshuaError::ModelLoad(format!("tokenizer read failed: {e}")))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| JoshuaError::ModelLoad(format!("tokenizer parse failed: {e}")))?;
    Ok(matches!(
        value.pointer("/decoder/type").and_then(|v| v.as_str()),
        Some("ByteLevel")
    ))
}

/// Incremental decoder for byte-level tokenizers (GPT-2 style, where a
/// token's text is a byte-unicode string and may be a *lone* byte).
///
/// Byte-level BPE splits multi-byte UTF-8 across token boundaries — a single
/// generated token can carry one raw byte (e.g. DeepSeek's vocab entries
/// `¡`..`ÿ`), so byte state must survive across tokens.  Re-decoding the
/// whole accumulated buffer every step did that at O(n²) in output length;
/// instead keep only a small trailing window:
///
/// * within a window whose boundary does not split a codepoint, decoding is
///   per-character, so a prefix-stable extension adds exactly the suffix a
///   whole-buffer decode would add;
/// * whenever stability breaks, fall back to one whole-buffer decode of all
///   ids so far — reproducing the previous behaviour verbatim.
///
/// A codepoint spans at most 4 bytes, so an 8-token window always contains
/// every not-yet-final byte.
#[derive(Default)]
struct ByteWindowDecoder {
    tail: Vec<u32>,
    tail_text: String,
}

impl ByteWindowDecoder {
    const WINDOW: usize = 8;

    /// Feed one token, returning the updated output text (`response`).
    fn push(
        &mut self,
        tokenizer: &Tokenizer,
        token: u32,
        response: String,
        all_ids: &[u32],
    ) -> Result<String> {
        let decode = |ids: &[u32]| -> Result<String> {
            tokenizer
                .decode(ids, false)
                .map_err(|e| JoshuaError::Inference(e.to_string()))
        };

        self.tail.push(token);
        let new_text = decode(&self.tail)?;
        let mut response = if new_text.starts_with(&self.tail_text) {
            // Stable extension: append just the new suffix.
            let mut response = response;
            response.push_str(&new_text[self.tail_text.len()..]);
            response
        } else {
            // A codepoint completed across the boundary: take the oracle.
            decode(all_ids)?
        };
        self.tail_text = new_text;

        if self.tail.len() > Self::WINDOW {
            // Slide.  `response` already holds every decoded byte — the head
            // token's text was appended when it was pushed — so sliding only
            // shrinks the window to keep future steps cheap; appending here
            // would duplicate output (and slicing `tail_text` at a byte
            // offset could split a codepoint).  Just refresh the window text
            // to match the shrunken window.
            let shrunk = decode(&self.tail[1..])?;
            self.tail.remove(0);
            self.tail_text = shrunk;
        }
        Ok(response)
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
        let n_ctx = if options.n_ctx == 0 {
            4096
        } else {
            options.n_ctx
        };
        let pin_hot_experts = options.pin_hot_experts;
        let raw_path = model_path.as_ref().to_path_buf();

        // Resolve the actual .gguf file path.
        let gguf_path = if raw_path.is_dir() {
            find_gguf_in_dir(&raw_path)?
        } else {
            raw_path
        };

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
        let byte_level_decode = tokenizer_is_byte_level(&tokenizer_path)?;
        // Map the GGUF file into memory using the configured backing.
        //
        // Hot-weight pinning applies per-range advice after the header is
        // parsed below, and whole-model prefetch replaces the blanket hint
        // entirely, so both ask map_model to skip the blanket random/sequential
        // hint (which would otherwise keep "free after use" semantics on the
        // very ranges we want to keep resident).
        let hot_pinning = options.pin_hot_weights
            || options.mlock_hot_weights != MlockMode::Off
            || options.prefetch_whole_model;
        let (mmap, model_file) = map_model(
            &gguf_path,
            options.huge_pages,
            options.lazy_weights,
            options.mmap,
            hot_pinning,
            options.prefetch_whole_model,
        )?;
        // Explicit huge pages copy the model into anonymous RAM — there is no
        // file to prefetch from, so drop the handle.  With `lazy_weights` the
        // copy is skipped and the mapping stays file-backed, so keep it.
        let model_file =
            if matches!(options.huge_pages, HugePages::Explicit(_)) && !options.lazy_weights {
                None
            } else {
                Some(Arc::new(model_file))
            };

        // Read GGUF metadata once to validate the architecture up front and
        // extract EOS token IDs.  The tolerant reader keeps dtypes candle
        // cannot represent (IQ2_XXS, I32, MXFP4), so files using them reach
        // arch detection instead of dying on the first unknown dtype.
        let gguf = read_gguf_header(&mmap)
            .map_err(|e| JoshuaError::ModelLoad(format!("GGUF read failed: {e}")))?;

        // Prefetch and/or lock the always-touched weights, and advise random
        // access on routed experts, when requested.  The raw header is used so
        // tensors in dtypes candle cannot name (IQ2_XXS, I32, MXFP4) are still
        // classified; the projected `Content` above drops them.
        if hot_pinning {
            if matches!(options.huge_pages, HugePages::Explicit(_)) {
                tracing::warn!(
                    "ignoring hot-weight pinning and whole-model prefetch: explicit huge pages \
                     already copy the whole model into anonymous RAM, so every range is \
                     resident. Use --huge-pages transparent (or none) for either to matter."
                );
            } else {
                let raw =
                    crate::gguf_ext::read_header(&mut Cursor::new(&mmap[..])).map_err(|e| {
                        JoshuaError::ModelLoad(format!("GGUF header re-read failed: {e}"))
                    })?;
                // A whole-file prefetch already covers the dense ranges, so the
                // dense WILLNEED is skipped; the expert RANDOM advice (and any
                // mlock) still apply.
                apply_hot_weight_pinning(
                    &mmap,
                    &raw,
                    options.pin_hot_weights && !options.prefetch_whole_model,
                    options.mlock_hot_weights,
                )?;
            }
        }

        // The API identifier is the file stem — the documented contract on
        // `model_name` (see the field and accessor docs) — so clients keyed
        // on it see no change.  The model's own `general.name` is sanitized
        // (model-supplied metadata is attacker-controlled) and used only for
        // the operator log line below.
        let model_name = gguf_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        let display_name = gguf
            .metadata
            .get("general.name")
            .and_then(|v| v.to_string().ok().cloned())
            .and_then(|s| sanitize_model_name(&s))
            .unwrap_or_else(|| model_name.clone());

        // Defer architectures candle cannot load (e.g. `deepseek4`) instead of
        // failing construction: an NPU backend may still serve them.  The
        // reason is remembered and returned by `load_model` when the candle
        // path is actually needed.
        let (arch, arch_error) = match Architecture::detect(&gguf.metadata) {
            Ok(arch) => (Some(arch), None),
            Err(e) => {
                tracing::warn!("{e} (continuing; an NPU backend may still serve this model)");
                (None, Some(e))
            }
        };

        let eos_token_ids = extract_eos_ids(&gguf, &tokenizer);
        let chat_template = extract_chat_template(&gguf, &tokenizer);
        let device = Self::resolve_device(options.backend)?;

        tracing::info!(
            "Model '{}' ready (arch={}, ctx={}, eos_ids={:?}, chat_template={}, device={:?})",
            display_name,
            arch.as_ref()
                .map(|a| a.display_name())
                .unwrap_or("unknown (NPU-only)"),
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
            model_file,
            tokenizer: Arc::new(tokenizer),
            byte_level_decode,
            eos_token_ids,
            chat_template,
            embed_model: Mutex::new(None),
            model_cache: Mutex::new(Vec::new()),
            kv_reuses: AtomicU64::new(0),
            kv_edit_reuses: AtomicU64::new(0),
            npu: None,
            in_flight: AtomicUsize::new(0),
            max_concurrency,
            max_output_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
            model_name,
            n_ctx,
            pin_hot_experts,
            device,
            arch_error,
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

    /// Pick the compute device for [`ComputeBackend::Auto`].
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

    /// Resolve the [`ComputeBackend`] requested in [`EngineOptions`] to a
    /// concrete candle [`Device`].
    ///
    /// [`ComputeBackend::Auto`] follows [`Self::default_device`], degrading to
    /// CPU with a warning.  An explicit GPU request is strict: failing to
    /// initialise the device — or a build without the matching cargo feature —
    /// is a load error, never a silent CPU fallback.
    fn resolve_device(backend: ComputeBackend) -> Result<Device> {
        match backend {
            ComputeBackend::Auto => Ok(Self::default_device()),
            ComputeBackend::Cpu => Ok(Device::Cpu),
            ComputeBackend::Metal => {
                #[cfg(feature = "metal")]
                {
                    Device::new_metal(0).map_err(|e| {
                        JoshuaError::ModelLoad(format!(
                            "Metal device requested but unavailable: {e}. \
                             Build with `--features metal` and run on a Metal-capable Mac, \
                             or pass --device cpu / auto."
                        ))
                    })
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err(JoshuaError::ModelLoad(
                        "Metal device requested but this build has no `metal` feature. \
                         Rebuild with `cargo build --features metal`, or pass --device cpu / auto."
                            .to_string(),
                    ))
                }
            }
            ComputeBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    Device::new_cuda(0).map_err(|e| {
                        JoshuaError::ModelLoad(format!(
                            "CUDA device requested but unavailable: {e}. \
                             Build with `--features cuda` and a CUDA toolkit, \
                             or pass --device cpu / auto."
                        ))
                    })
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(JoshuaError::ModelLoad(
                        "CUDA device requested but this build has no `cuda` feature. \
                         Rebuild with `cargo build --features cuda`, or pass --device cpu / auto."
                            .to_string(),
                    ))
                }
            }
        }
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

    /// The compute device inference runs on (CPU, or Metal/CUDA when the
    /// matching feature is enabled and a device is available).
    pub fn device(&self) -> &Device {
        &self.device
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
        if messages
            .iter()
            .any(|m| m.images.as_ref().is_some_and(|i| !i.is_empty()))
        {
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
    fn clamp_options(
        &self,
        options: &GenerationOptions,
        prompt_len: Option<usize>,
    ) -> GenerationOptions {
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
                if self.arch_error.is_some() {
                    // No candle loader exists for this architecture, so the
                    // retry could only report "unsupported model" and hide
                    // the accelerator failure the caller needs to see.
                    return Err(e);
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
        let mut decoded_ids: Vec<u32> = Vec::new();
        // Incremental byte-level decode state (see [`ByteWindowDecoder`]).
        let mut byte_window = ByteWindowDecoder::default();
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

            if std::env::var_os("JOSHUA_DEBUG_TOKENS").is_some() {
                let mut order: Vec<usize> = (0..logits_vec.len()).collect();
                order.sort_by(|&a, &b| logits_vec[b].total_cmp(&logits_vec[a]));
                let top5: Vec<(usize, f32)> = order[..5.min(order.len())]
                    .iter()
                    .map(|&i| (i, logits_vec[i]))
                    .collect();
                eprintln!(
                    "[eng] n_cur={n_cur} tok={next_token} temp={} topk={} topp={} minp={} reppen={} top5={top5:?}",
                    options.temperature, options.top_k, options.top_p, options.min_p, options.repetition_penalty
                );
            }

            if self.eos_token_ids.contains(&next_token) {
                break;
            }

            if self.byte_level_decode {
                // Byte-level BPE splits multi-byte UTF-8 across token
                // boundaries — a single token can be a lone byte (e.g.
                // DeepSeek's raw-byte vocab entries `¡`..`ÿ`), which is
                // invalid UTF-8 on its own and would decode to U+FFFD.
                // [`ByteWindowDecoder`] keeps that byte state across tokens
                // without re-decoding the whole output every step (the old
                // whole-buffer decode made generation O(n²) in length).
                decoded_ids.push(next_token);
                response =
                    byte_window.push(&self.tokenizer, next_token, response, &decoded_ids)?;
            } else {
                // Decoder-less / word-level tokenizers: batch decoding would
                // join pieces with spaces, so decode each token and append.
                let piece = self
                    .tokenizer
                    .decode(&[next_token], false)
                    .map_err(|e| JoshuaError::Inference(e.to_string()))?;
                response.push_str(&piece);
            }
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
                return Err(JoshuaError::PromptTooLong(
                    tokens.len(),
                    self.n_ctx as usize,
                ));
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

    /// Number of requests so far that continued from a cached KV prefix
    /// recovered across an edit of the conversation (the prompt shares only
    /// a prefix with the cached history, so the session state was rewound
    /// to that common prefix).  Included in [`Engine::kv_reuse_count`].
    pub fn kv_edit_reuse_count(&self) -> u64 {
        self.kv_edit_reuses.load(Ordering::Relaxed)
    }

    /// Route generation through an NPU backend (see [`crate::npu`]).
    ///
    /// Generation requests try the backend first and transparently fall back
    /// to the candle CPU/GPU path when session creation or a forward pass
    /// fails; after `NPU_MAX_FAILURES` failures the backend is disabled
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
    /// 2. a pooled instance whose history shares only a *prefix* with the
    ///    prompt (an edited conversation — agent harnesses truncate or
    ///    replace middle blocks such as old tool outputs): where the
    ///    architecture supports it, its KV state is rewound to the longest
    ///    common prefix and only the diverging remainder is prefilled;
    /// 3. a pooled instance whose state can be cleared — skips re-creating
    ///    the session;
    /// 4. a fresh session (NPU) or a fresh instance from the mmap (candle).
    fn acquire_session(
        &self,
        prompt_tokens: &[u32],
        allow_npu: bool,
    ) -> Result<(GenSession, usize)> {
        let want_npu = allow_npu && self.npu.as_ref().is_some_and(|n| n.usable());

        // Edited-context candidate, if any: picked under the pool lock, but
        // rewound (a potentially large KV copy) outside the critical section
        // below so concurrent acquire/release is never stalled by it.
        let mut edit_pick: Option<(CachedModel, usize)> = None;

        {
            let mut pool = self.model_pool();
            // Only reuse sessions of the kind this request will run on —
            // mixing kinds mid-conversation would splice numerically
            // different logits into one generation.
            let best = pool
                .iter()
                .enumerate()
                .filter(|(_, c)| {
                    // An empty history is a prefix of every prompt but
                    // carries no reusable KV — skip it so entries parked
                    // empty (or poisoned, below) reach the clearing path
                    // instead of being served as-is.
                    !c.tokens.is_empty()
                        && c.session.is_npu() == want_npu
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
            // Edited-context reuse: no pooled history extends this prompt,
            // but one may share a prefix with it.  Rewind that instance's
            // KV state to the common prefix instead of throwing the whole
            // prefill away.  Candle path only — NPU plugins own their token
            // history internally, so their sessions can never be rewound.
            //
            // Two shapes of edit reach this path:
            // * a *truncating* edit (harness dropped/replaced middle blocks)
            //   leaves `lcp < prompt.len()`: rewind to `lcp` and prefill the
            //   diverging remainder;
            // * a *rollback* (regenerate/retry: prompt is a strict prefix of
            //   the history) has `lcp == prompt.len()` — rewinding to `lcp`
            //   would leave nothing to prefill, and the logits for the token
            //   after the prompt are not recoverable from a cache alone, so
            //   rewind to `lcp - 1` and re-prefill just the final token.
            //   (`keep == 0` on single-token prompts degenerates to a clear.)
            if !want_npu {
                let best_edit = pool
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| c.session.supports_truncate())
                    .map(|(i, c)| (i, longest_common_prefix(&c.tokens, prompt_tokens)))
                    .filter(|(_, lcp)| *lcp > 0 && *lcp <= prompt_tokens.len())
                    // Longest common prefix first; pass 1's exact-prefix
                    // matches are intentionally preferred over longer edit
                    // candidates because they need no KV copy at all.
                    .max_by_key(|(_, lcp)| *lcp);
                if let Some((i, lcp)) = best_edit {
                    // Rollback case: keep one token behind so the suffix
                    // prefill is never empty.
                    let keep = lcp.min(prompt_tokens.len() - 1);
                    edit_pick = Some((pool.swap_remove(i), keep));
                }
            }
            // The clear path runs only when no rewind was picked — rewinding
            // preserves strictly more work than clearing.
            if edit_pick.is_none() {
                let resettable = pool.iter().position(|c| c.session.is_npu() == want_npu);
                if let Some(i) = resettable {
                    let mut cached = pool.swap_remove(i);
                    if cached.session.clear_state() {
                        tracing::debug!(
                            npu = want_npu,
                            "Reusing pooled session with cleared state"
                        );
                        return Ok((cached.session, 0));
                    }
                    // Reset failed (e.g. dead shim): drop it and fall through.
                }
            }
        }

        // Attempt the rewind outside the pool lock.
        if let Some((mut cached, lcp)) = edit_pick {
            if cached.session.truncate_to(lcp) {
                self.kv_reuses.fetch_add(1, Ordering::Relaxed);
                self.kv_edit_reuses.fetch_add(1, Ordering::Relaxed);
                tracing::debug!(
                    reused_tokens = lcp,
                    dropped_tokens = cached.tokens.len().saturating_sub(lcp),
                    prompt_tokens = prompt_tokens.len(),
                    "Continuing from cached KV prefix after context edit"
                );
                return Ok((cached.session, lcp));
            }
            // Truncation failed unexpectedly.  The failure may have hit
            // part-way through the per-layer loop, leaving earlier layers
            // shortened and later ones not — a cache that no longer matches
            // any single prefix of anything.  Poison the entry twice over:
            // clear the state outright when possible, and park it with empty
            // tokens so every reuse pass skips it (an empty history is a
            // prefix of every prompt, so the strict-prefix pass alone would
            // still match it).  If the clear also fails, the entry is dropped
            // by falling out of the block with `clear_ok == false`.
            {
                let clear_ok = cached.session.clear_state();
                if clear_ok {
                    cached.tokens.clear();
                    let mut pool = self.model_pool();
                    pool.push(cached);
                    while pool.len() > MAX_CACHED_MODELS {
                        pool.remove(0);
                    }
                }
                // !clear_ok: drop the instance entirely — a session whose
                // state cannot even be reset is not worth parking.
            }
        }

        if want_npu {
            let npu = self.npu.as_ref().expect("checked above");
            match npu.backend.create_session(&self.model_path, self.n_ctx) {
                Ok(session) => return Ok((GenSession::Npu(session), 0)),
                Err(e) => {
                    npu.record_failure(&npu.backend.name(), &e);
                    if self.arch_error.is_some() {
                        // Without a candle loader for this architecture the
                        // fallback can only report "unsupported model", which
                        // would mask the real accelerator failure.
                        return Err(JoshuaError::ModelLoad(format!(
                            "{} session creation failed: {e}",
                            npu.backend.name()
                        )));
                    }
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
        // Unknown-architecture models (e.g. `deepseek4`) can only be served
        // by an NPU backend; surface the stored detection error when the
        // candle path is needed instead of half-loading.
        if let Some(err) = &self.arch_error {
            let msg = match &self.npu {
                Some(npu) if !npu.usable() => format!(
                    "{err}; the {} backend serving this model was disabled after \
                     {NPU_MAX_FAILURES} failures",
                    npu.backend.name()
                ),
                _ => err.clone(),
            };
            return Err(JoshuaError::ModelLoad(msg));
        }
        let mut cursor = Cursor::new(&self.mmap[..]);
        let gguf = read_gguf_header(&self.mmap)
            .map_err(|e| JoshuaError::ModelLoad(format!("GGUF read failed: {e}")))?;
        // Hand the loader the mapping so architectures Joshua implements
        // itself can borrow weights in place rather than copying them.
        let mut model = QuantizedModel::from_gguf_mmap(
            gguf,
            &mut cursor,
            &self.device,
            Some(Arc::clone(&self.mmap)),
            self.model_file.clone(),
        )
        .map_err(|e| JoshuaError::ModelLoad(format!("model init failed: {e}")))?;
        model.set_pin_hot_experts(self.pin_hot_experts);
        Ok(model)
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
///
/// `per_range_advice` skips the blanket hint: the caller (hot-weight pinning)
/// applies per-range advice after parsing the header.  `prefetch_whole` asks
/// the kernel to pull the whole file into the page cache right after mapping
/// (a single `MADV_WILLNEED`); it implies `per_range_advice` (the blanket
/// hint is pointless once every page is being read in anyway).
fn map_model(
    path: &Path,
    huge: HugePages,
    lazy: bool,
    mmap_mode: MmapMode,
    per_range_advice: bool,
    prefetch_whole: bool,
) -> Result<(Mmap, File)> {
    let file = File::open(path)?;

    // Explicit huge pages use an anonymous copy; handle separately.
    //
    // That copy is read in full up front, which is exactly what `lazy_weights`
    // exists to avoid — and a caller sets it precisely because the model does
    // not fit in memory, where an eager copy cannot succeed at all.  Laziness
    // therefore wins, loudly: silently honouring the huge-page request would
    // turn "page this in on demand" into an out-of-memory failure.
    if let HugePages::Explicit(size) = huge {
        if lazy {
            tracing::warn!(
                "ignoring the explicit huge-page request: it copies the whole model into \
                 anonymous memory, which cannot work for a model flagged larger than RAM. \
                 Using the file-backed mapping so weights page in on demand. Use \
                 --huge-pages transparent for huge pages that keep the mapping file-backed."
            );
        } else {
            check_mappable(path, &file, mmap_mode, false)?;
            return map_model_hugetlb(path, &file, size).map(|m| (m, file));
        }
    }

    check_mappable(path, &file, mmap_mode, true)?;

    let mmap = map_model_file_padded(&file)?;

    // Blanket access hint.  The engine *re-reads* every weight on every token,
    // so `MADV_SEQUENTIAL` ("pages may be freed soon after they are accessed")
    // would evict the model from the page cache right after each use — the
    // default path keeps no blanket hint instead, letting the kernel's normal
    // readahead and page-cache retention apply.  Only a model explicitly
    // flagged as far larger than RAM gets `MADV_RANDOM`, since sparse expert
    // access makes readahead evict more than it saves.  Hot-weight pinning and
    // whole-model prefetch skip the blanket hint entirely: they apply targeted
    // advice per range (or none) after the header is parsed.  Best effort
    // either way.
    #[cfg(unix)]
    if !per_range_advice && !prefetch_whole {
        let _ = mmap.advise(if lazy {
            memmap2::Advice::Random
        } else {
            memmap2::Advice::Normal
        });
    }

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

    // Prefetch the whole model into the page cache so it is resident before
    // the first request.  A single whole-mapping `MADV_WILLNEED`: the kernel
    // streams the file in behind the load, and clean pages stay evictable
    // under memory pressure, so this is advisory and never blocks for long.
    #[cfg(unix)]
    if prefetch_whole {
        match mmap.advise(memmap2::Advice::WillNeed) {
            Ok(()) => {
                tracing::info!("prefetching the whole model into the page cache (MADV_WILLNEED)")
            }
            Err(e) => tracing::warn!("could not prefetch the model into the page cache: {e}"),
        }
    }
    #[cfg(not(unix))]
    if prefetch_whole {
        tracing::warn!("whole-model prefetch is unix-only (madvise); ignoring the request");
    }

    Ok((mmap, file))
}

// ─── Hot-weight pinning ────────────────────────────────────────────────────────

/// A byte range in the model mapping: `(offset, len)`.
type ByteRange = (usize, usize);

/// Length of the longest common token prefix of two sequences.
///
/// The reuse predicate for *edited* conversations: an agent harness that
/// truncates or replaces middle blocks (old tool outputs, thinking segments)
/// leaves a follow-up prompt that shares a prefix with the cached history
/// without extending it.  O(min(len)) with no allocation.
fn longest_common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Whether a tensor name belongs to the routed-expert set.
///
/// These are the only weights touched sparsely: a token routes through a
/// handful of the model's experts per layer, so readahead/prefetch drags in
/// far more than a token will use.  Everything else — embeddings, norms,
/// attention, routers, shared experts, indexer/compressor, output — is dense
/// and touched on every token.
fn is_routed_expert(name: &str) -> bool {
    name.contains(".ffn_gate_exps")
        || name.contains(".ffn_down_exps")
        || name.contains(".ffn_up_exps")
}

/// Byte ranges of the model file holding dense vs routed-expert weights.
///
/// Tensor data is laid out back-to-back (aligned to `general.alignment`,
/// normally 32 bytes), so each tensor's size is the gap to the next tensor's
/// offset and the last tensor ends at the file size.  Offsets are absolute
/// (the header's `tensor_data_offset` is added).  Ranges separated by at most
/// one *system base page* are merged — `madvise`/`mlock` round in base pages,
/// so a sub-page gap costs nothing and merging cuts the syscall count.  (The
/// merge gap stays at base-page granularity on purpose: a huge-page-sized gap
/// would merge dense ranges across the expert tensors between them, turning
/// the dense/expert split into "prefetch everything".)
///
/// Returns `(dense, routed_experts)`, each a sorted list of [`ByteRange`]s.
fn weight_ranges(
    header: &crate::gguf_ext::GgufHeader,
    file_size: u64,
) -> (Vec<ByteRange>, Vec<ByteRange>) {
    weight_ranges_with_page(header, file_size, base_page_size())
}

/// [`weight_ranges`] with an explicit page size, so tests can pin the merge
/// granularity without touching the real system.
fn weight_ranges_with_page(
    header: &crate::gguf_ext::GgufHeader,
    file_size: u64,
    page: usize,
) -> (Vec<ByteRange>, Vec<ByteRange>) {
    let mut all: Vec<(u64, &str)> = header
        .tensors
        .iter()
        .map(|(n, t)| (header.tensor_data_offset + t.offset, n.as_str()))
        .collect();
    all.sort_by_key(|(o, _)| *o);

    let mut dense: Vec<ByteRange> = Vec::new();
    let mut experts: Vec<ByteRange> = Vec::new();
    for i in 0..all.len() {
        let (off, name) = all[i];
        let end = if i + 1 < all.len() {
            all[i + 1].0
        } else {
            file_size
        };
        if end <= off {
            continue; // zero-size tensor (or a malformed trailing one)
        }
        let range = (off as usize, (end - off) as usize);
        if is_routed_expert(name) {
            experts.push(range);
        } else {
            dense.push(range);
        }
    }

    fn merge(mut ranges: Vec<ByteRange>, page: usize) -> Vec<ByteRange> {
        ranges.sort_by_key(|(o, _)| *o);
        let mut out: Vec<ByteRange> = Vec::new();
        for (off, len) in ranges {
            if let Some(last) = out.last_mut() {
                if off <= last.0 + last.1 + page {
                    let end = (off + len).max(last.0 + last.1);
                    last.1 = end - last.0;
                    continue;
                }
            }
            out.push((off, len));
        }
        out
    }

    (merge(dense, page), merge(experts, page))
}

/// Base memory page size in bytes — the granularity `madvise`/`mlock` round
/// to.  4 KiB on most Linux, 16 KiB on Apple Silicon macOS; falls back to
/// 4 KiB when the system value is unavailable or implausible.
fn base_page_size() -> usize {
    #[cfg(unix)]
    {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        if page >= 4096 && page.is_power_of_two() {
            return page;
        }
    }
    4096
}

/// Apply hot-weight pinning to the model mapping: prefetch (`MADV_WILLNEED`)
/// and/or `mlock(2)` the dense ranges, and advise `MADV_RANDOM` on the routed
/// experts so sparse access does not evict the resident hot set.
///
/// Best effort: advice failures are warnings (the kernel may ignore them
/// anyway).  The memlock limit is checked against the hot-set size before any
/// `mlock` call: [`MlockMode::On`] warns once and degrades to advisory when
/// the limit is too low, [`MlockMode::Required`] fails the load.  Never
/// affects correctness.
#[cfg(unix)]
fn apply_hot_weight_pinning(
    mmap: &Mmap,
    header: &crate::gguf_ext::GgufHeader,
    prefetch: bool,
    mlock: MlockMode,
) -> Result<()> {
    let (dense, experts) = weight_ranges(header, mmap.len() as u64);
    let gib =
        |r: &[ByteRange]| r.iter().map(|(_, l)| *l as u64).sum::<u64>() as f64 / 2f64.powi(30);
    tracing::info!(
        "hot-weight pinning: {} dense range(s) ({:.2} GiB, WILLNEED {}) and {} expert range(s) ({:.2} GiB, RANDOM)",
        dense.len(),
        gib(&dense),
        if prefetch { "on" } else { "off" },
        experts.len(),
        gib(&experts),
    );

    if prefetch {
        let mut failed = 0usize;
        for &(off, len) in &dense {
            if mmap
                .advise_range(memmap2::Advice::WillNeed, off, len)
                .is_err()
            {
                failed += 1;
            }
        }
        if failed > 0 {
            tracing::warn!(
                "{failed}/{} dense ranges ignored the WILLNEED prefetch hint",
                dense.len()
            );
        }
    }

    {
        let mut failed = 0usize;
        for &(off, len) in &experts {
            if mmap
                .advise_range(memmap2::Advice::Random, off, len)
                .is_err()
            {
                failed += 1;
            }
        }
        if failed > 0 {
            tracing::warn!(
                "{failed}/{} expert ranges ignored the RANDOM access hint",
                experts.len()
            );
        }
    }

    if mlock != MlockMode::Off {
        let page = base_page_size();
        let required = aligned_range_bytes(&dense, page, mmap.len());
        let limit = memlock_limit_bytes();
        match mlock_decision(mlock, limit, required) {
            MlockDecision::Proceed => {
                let failed = mlock_ranges(mmap, &dense, page);
                if failed > 0 && mlock == MlockMode::Required {
                    return Err(JoshuaError::ModelLoad(format!(
                        "mlock of the hot weight set failed ({failed}/{} ranges) despite \
                         RLIMIT_MEMLOCK appearing sufficient — see the warnings above",
                        dense.len()
                    )));
                }
            }
            MlockDecision::Degrade => tracing::warn!(
                "RLIMIT_MEMLOCK is {limit}, below the {:.2} GiB hot set — skipping the lock \
                 and degrading to advisory pinning. Raise it with /etc/security/limits.conf \
                 (`{user} - memlock unlimited`), `LimitMEMLOCK=infinity` (systemd), or \
                 `ulimit -l unlimited`; for a live systemd user session apply it without \
                 re-login: `sudo prlimit --pid <user manager pid> --memlock=-1:-1`.",
                required as f64 / 2f64.powi(30),
                limit = display_memlock_limit(limit),
                user = whoami(),
            ),
            MlockDecision::Fail => {
                return Err(JoshuaError::ModelLoad(format!(
                    "RLIMIT_MEMLOCK is {limit}, below the {:.2} GiB hot set that \
                     --mlock-hot-weights=required demands be locked. Raise it with \
                     /etc/security/limits.conf (`{user} - memlock unlimited`), \
                     `LimitMEMLOCK=infinity` (systemd), or `ulimit -l unlimited`; for a \
                     live systemd user session: `sudo prlimit --pid <user manager pid> \
                     --memlock=-1:-1`.",
                    required as f64 / 2f64.powi(30),
                    limit = display_memlock_limit(limit),
                    user = whoami(),
                )))
            }
        }
    }
    Ok(())
}

#[cfg(not(unix))]
fn apply_hot_weight_pinning(
    _mmap: &Mmap,
    _header: &crate::gguf_ext::GgufHeader,
    _prefetch: bool,
    _mlock: MlockMode,
) -> Result<()> {
    tracing::warn!("hot-weight pinning is unix-only; ignoring the request");
    Ok(())
}

/// What to do about `mlock` given the limit and what the hot set needs.
#[cfg(unix)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MlockDecision {
    /// The limit is unknown/unlimited or large enough — attempt the lock.
    Proceed,
    /// [`MlockMode::On`] with a too-small limit: warn once, skip the lock.
    Degrade,
    /// [`MlockMode::Required`] with a too-small limit: fail the load.
    Fail,
}

#[cfg(unix)]
fn mlock_decision(mode: MlockMode, limit: Option<u64>, required: u64) -> MlockDecision {
    match mode {
        MlockMode::Off => MlockDecision::Proceed, // caller never invokes with Off
        MlockMode::On => match limit {
            Some(l) if l < required => MlockDecision::Degrade,
            _ => MlockDecision::Proceed,
        },
        MlockMode::Required => match limit {
            Some(l) if l < required => MlockDecision::Fail,
            _ => MlockDecision::Proceed,
        },
    }
}

/// Current `RLIMIT_MEMLOCK` in bytes; `None` when unlimited or unreadable.
#[cfg(unix)]
fn memlock_limit_bytes() -> Option<u64> {
    let mut lim = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `lim` is a valid out-pointer for getrlimit; no other memory is
    // touched.
    if unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut lim) } != 0 {
        return None;
    }
    if lim.rlim_cur == libc::RLIM_INFINITY {
        return None; // unlimited
    }
    Some(lim.rlim_cur)
}

/// Human-readable form of a memlock limit for warnings.
#[cfg(unix)]
fn display_memlock_limit(limit: Option<u64>) -> String {
    match limit {
        None => "unlimited".to_string(),
        Some(b) if b >= 1024 * 1024 && b % (1024 * 1024) == 0 => {
            format!("{} MiB", b / (1024 * 1024))
        }
        Some(b) => format!("{b} bytes"),
    }
}

/// Best-effort current login name, for the limits.conf remediation hint.
#[cfg(unix)]
fn whoami() -> String {
    std::env::var("USER").unwrap_or_else(|_| "USER".to_string())
}

/// Total bytes the ranges occupy once page-aligned and clamped to the
/// mapping — exactly what `mlock` will lock.  Ranges are expected sorted by
/// offset; page-space overlaps between neighbouring ranges are coalesced so
/// a page shared by two ranges is counted once (the kernel counts locked
/// pages once too).
#[cfg(unix)]
fn aligned_range_bytes(ranges: &[ByteRange], page: usize, map_len: usize) -> u64 {
    // The kernel locks every page intersecting the requested range, so the
    // clamp is to the mapping's last *page* (a tail that ends mid-page still
    // locks that page).
    let map_end = map_len.saturating_add(page - 1) & !(page - 1);
    let mut total: u64 = 0;
    let mut covered_end: usize = 0; // exclusive page-space end of previous extent
    for &(off, len) in ranges {
        let start = off & !(page - 1);
        let end = ((off + len).saturating_add(page - 1)) & !(page - 1);
        let end = end.min(map_end);
        if end <= start {
            continue;
        }
        if start >= covered_end {
            total += (end - start) as u64;
        } else if end > covered_end {
            total += (end - covered_end) as u64;
        }
        covered_end = covered_end.max(end);
    }
    total
}

/// `mlock(2)` the given ranges, page-aligned and clamped to the mapping.
///
/// Returns the number of ranges that failed to lock.  Failures are reported
/// with the remediation hint; the caller decides whether they are fatal
/// ([`MlockMode::Required`] fails the load, [`MlockMode::On`] degrades to
/// advisory pinning).
#[cfg(unix)]
fn mlock_ranges(mmap: &Mmap, ranges: &[ByteRange], page: usize) -> usize {
    let total: u64 = ranges.iter().map(|(_, l)| *l as u64).sum();
    let mut locked: u64 = 0;
    let mut failed = 0usize;
    for &(off, len) in ranges {
        let start = off & !(page - 1);
        let end = ((off + len).saturating_add(page - 1)) & !(page - 1);
        // The kernel locks every page intersecting the range, so clamp to the
        // mapping's last page, not its raw length (a tail ending mid-page
        // still locks that page).
        let end = end.min(mmap.len().saturating_add(page - 1) & !(page - 1));
        if end <= start {
            continue;
        }
        // SAFETY: `start..end` lies within the mapping (clamped above) and
        // mlock does not modify memory.  The mapping lives until the engine
        // is dropped; mlock is released automatically on munmap.
        let rc = unsafe {
            libc::mlock(
                mmap.as_ptr().add(start).cast::<libc::c_void>(),
                end - start,
            )
        };
        if rc == 0 {
            locked += (end - start) as u64;
        } else {
            failed += 1;
            if failed <= 3 {
                tracing::warn!(
                    "mlock of hot weight range at offset {off:#x} failed: {}",
                    std::io::Error::last_os_error()
                );
            }
        }
    }
    if failed > 0 {
        tracing::warn!(
            "mlock failed for {failed}/{} hot ranges ({:.2} of {:.2} GiB locked). \
             Raise the process memlock limit to pin them: /etc/security/limits.conf \
             (`{user} - memlock unlimited`), `LimitMEMLOCK=infinity` (systemd), or \
             `ulimit -l unlimited`; for a live systemd user session: \
             `sudo prlimit --pid <user manager pid> --memlock=-1:-1`.",
            ranges.len(),
            locked as f64 / 2f64.powi(30),
            total as f64 / 2f64.powi(30),
            user = whoami(),
        );
    }
    failed
}

/// Refuse — or at least complain about — a model file that `mmap` cannot serve
/// usefully.
///
/// Compressed model files are the recurring trap: a `.gguf` that is really a
/// gzip stream maps to compressed bytes and fails to parse at all, and a
/// `.gguf` stored on a transparently compressing filesystem maps fine but
/// decompresses a block on every page fault, which quietly costs most of what
/// mmap-based loading buys.  Neither is visible from the filename.
///
/// `file_backed` says whether the mapping about to be created actually reads
/// through the file: the explicit huge-page path copies the model into
/// anonymous memory with one sequential pass, so filesystem compression costs
/// it a single decompression rather than a per-fault one and is not worth
/// reporting — a compression *container* is still fatal there, since the bytes
/// copied in are not the model.
///
/// Under [`MmapMode::Required`] the caller asked for mapping explicitly, so
/// this is an error; otherwise mapping is just the implicit default and the
/// load continues with a warning.
fn check_mappable(path: &Path, file: &File, mode: MmapMode, file_backed: bool) -> Result<()> {
    let Some(found) = crate::compression::detect_gguf(file) else {
        return Ok(());
    };
    if !file_backed && matches!(found, crate::compression::Compression::Filesystem { .. }) {
        return Ok(());
    }

    let what = format!("{path:?} cannot be memory-mapped usefully: {found}");
    match mode {
        MmapMode::Required => Err(JoshuaError::ModelLoad(format!(
            "{what}. Memory mapping was requested explicitly, so this is an error; \
             drop the explicit mmap request to load the model anyway."
        ))),
        MmapMode::Auto => {
            tracing::warn!("{what}.");
            Ok(())
        }
    }
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
        .map_err(|e| {
            JoshuaError::ModelLoad(format!("reading model into huge pages failed: {e}"))
        })?;

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
    map_model_file_padded(file)
}

/// Map `file` with its length rounded up to a whole page.
///
/// `mmap(2)` maps whole pages anyway, so the rounded-up region is valid
/// address space (the tail past EOF is zero-filled).  The page-multiple
/// length matters to the zero-copy Metal weight path:
/// `newBufferWithBytesNoCopy` requires the wrapped region's length to be a
/// multiple of the page size.
fn map_model_file_padded(file: &File) -> Result<Mmap> {
    let len = file
        .metadata()
        .map_err(|e| JoshuaError::ModelLoad(format!("stat of GGUF file failed: {e}")))?
        .len() as usize;
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let map_len = if page > 0 && page.is_power_of_two() {
        len.next_multiple_of(page)
    } else {
        len
    };
    unsafe { memmap2::MmapOptions::new().len(map_len).map(file) }
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
/// Find the single `.gguf` file in a model directory.
pub fn find_gguf_in_dir(dir: &Path) -> Result<PathBuf> {
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

/// Parse the GGUF header tolerantly (raw dtype ids) and project it onto
/// candle's `Content`, dropping tensors whose dtype candle cannot name.
/// Those tensors are decoded by Joshua's own loaders via the raw header.
fn read_gguf_header(mmap: &[u8]) -> Result<gguf_file::Content> {
    let header = crate::gguf_ext::read_header(&mut Cursor::new(mmap))?;
    // The deepseek4 loader is the only one that reads tensors by their raw
    // GGUF dtype id (IQ2_XXS, I32, …).  For every other architecture the
    // candle loaders only ever see the projected `Content`, so an
    // unsupported-dtype tensor would be dropped here and surface much later
    // as a misleading "cannot find tensor" for a required weight — or, worse,
    // silently treated as absent by a loader probing an optional tensor,
    // changing the model without an error.  Refuse the load with the precise
    // cause instead.  (An undetectable architecture is left alone: the load
    // already fails with an accurate architecture error downstream.)
    if let Ok(arch) = Architecture::detect(&header.metadata) {
        if arch != Architecture::DeepSeek4 {
            let unsupported = header.unsupported_tensors();
            if !unsupported.is_empty() {
                let names = unsupported
                    .iter()
                    .map(|(n, d)| format!("`{n}` (GGUF dtype id {d})"))
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(crate::JoshuaError::ModelLoad(format!(
                    "GGUF header: model architecture '{}' contains {} tensor(s) in a GGUF \
                     dtype the engine cannot decode: {}. Only the deepseek4 loader reads \
                     tensors by raw dtype id; re-quantize this model to a candle-supported \
                     format (F32/F16/Q8_0/Q4_K, …) or use a loader that decodes these dtypes.",
                    arch.display_name(),
                    unsupported.len(),
                    names
                )));
            }
        }
    }
    header.to_candle_content()
}

/// Sanitize a model-supplied `general.name` into a safe display identifier.
///
/// The GGUF metadata is fully controlled by whoever ships the model file, and
/// `Engine::model_name` is echoed into operator logs and OpenAI-compatible API
/// responses (`model` field).  This strips control characters (newlines, ANSI
/// escapes, NUL, ...), collapses and trims whitespace, and caps the length so
/// the value cannot inject log lines, escape sequences, or unbounded strings.
/// Returns `None` when nothing usable remains — the caller falls back to the
/// file stem.
fn sanitize_model_name(raw: &str) -> Option<String> {
    const MAX_BYTES: usize = 128;
    let mut out = String::with_capacity(raw.len().min(MAX_BYTES));
    let mut seen_non_space = false;
    let mut pending_space = false;
    for c in raw.chars() {
        if out.len() >= MAX_BYTES {
            break;
        }
        if c.is_control() {
            continue;
        }
        if c.is_whitespace() {
            pending_space = seen_non_space;
            continue;
        }
        if pending_space {
            out.push(' ');
            pending_space = false;
        }
        out.push(c);
        seen_non_space = true;
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
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
    let bos =
        token_str_from_metadata(gguf, "tokenizer.ggml.bos_token_id", tokenizer).unwrap_or_default();
    let eos =
        token_str_from_metadata(gguf, "tokenizer.ggml.eos_token_id", tokenizer).unwrap_or_default();
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
    // NOTE: at temperature 0 the penalty is skipped entirely (pure greedy),
    // unless JOSHUA_LEGACY_REPPEN is set — matching the semantics of every
    // other sampler where a "temperature" of 0 means "take the argmax of the
    // raw logits".
    let reppen = if opts.temperature <= 0.0 && std::env::var_os("JOSHUA_LEGACY_REPPEN").is_none() {
        1.0
    } else {
        opts.repetition_penalty
    };
    let logits: Vec<f32> = if reppen != 1.0 {
        let mut v = logits.to_vec();
        for &token in recent_tokens {
            if let Some(l) = v.get_mut(token as usize) {
                if *l > 0.0 {
                    *l /= reppen;
                } else {
                    *l *= reppen;
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
                probs[b]
                    .partial_cmp(&probs[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
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

    let dist = WeightedIndex::new(&probs).map_err(|e| JoshuaError::Inference(e.to_string()))?;
    Ok(dist.sample(rng) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_model_name_keeps_plain_names() {
        assert_eq!(
            sanitize_model_name("DeepSeek-V4-Flash"),
            Some("DeepSeek-V4-Flash".into())
        );
        assert_eq!(
            sanitize_model_name("  Llama 3.1  8B  "),
            Some("Llama 3.1 8B".into())
        );
    }

    #[test]
    fn sanitize_model_name_strips_control_chars_and_newlines() {
        // Newline + ANSI ESC injection must not survive into logs/API; the
        // ESC byte is stripped (the trailing `[31m` is inert text without it).
        assert_eq!(
            sanitize_model_name("evil\n[2Jname"),
            Some("evil[2Jname".into())
        );
        assert_eq!(sanitize_model_name("a\u{1b}[31mb"), Some("a[31mb".into()));
        assert_eq!(sanitize_model_name("\u{0}"), None);
        assert_eq!(sanitize_model_name("   "), None);
    }

    #[test]
    fn sanitize_model_name_caps_length() {
        let long = "x".repeat(1000);
        assert_eq!(sanitize_model_name(&long), Some("x".repeat(128)));
    }

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
        assert!(
            bytes >= 2 * 1024 * 1024 && bytes.is_power_of_two(),
            "got {bytes}"
        );
    }

    #[test]
    fn engine_options_builder() {
        let o = EngineOptions::with_n_ctx(2048).huge_pages(HugePages::Transparent);
        assert_eq!(o.n_ctx, 2048);
        assert_eq!(o.huge_pages, HugePages::Transparent);
        assert_eq!(EngineOptions::default().huge_pages, HugePages::Off);
        // Mapping is implicit unless the caller asks for it by name.
        assert_eq!(EngineOptions::default().mmap, MmapMode::Auto);
        assert_eq!(
            EngineOptions::default().mmap(MmapMode::Required).mmap,
            MmapMode::Required
        );
        // Pinning is off by default and settable through the builders.
        assert!(!EngineOptions::default().pin_hot_weights);
        assert_eq!(EngineOptions::default().mlock_hot_weights, MlockMode::Off);
        // Whole-model prefetch is off by default and settable through the
        // builder.
        assert!(!EngineOptions::default().prefetch_whole_model);
        assert!(
            EngineOptions::default()
                .prefetch_whole_model(true)
                .prefetch_whole_model
        );
        let o = EngineOptions::default()
            .pin_hot_weights(true)
            .mlock_hot_weights(MlockMode::On);
        assert!(o.pin_hot_weights);
        assert_eq!(o.mlock_hot_weights, MlockMode::On);
        assert_eq!(
            EngineOptions::default()
                .mlock_hot_weights(MlockMode::Required)
                .mlock_hot_weights,
            MlockMode::Required
        );
    }

    // ─── Hot-weight pinning ────────────────────────────────────────────────

    fn header_with_tensors(
        tensor_data_offset: u64,
        tensors: &[(&str, u64)],
    ) -> crate::gguf_ext::GgufHeader {
        use std::collections::HashMap;
        crate::gguf_ext::GgufHeader {
            version: 3,
            metadata: HashMap::new(),
            tensors: tensors
                .iter()
                .map(|(n, off)| {
                    (
                        (*n).to_string(),
                        crate::gguf_ext::RawTensorInfo {
                            dtype: 0,
                            dims: vec![1],
                            offset: *off,
                        },
                    )
                })
                .collect(),
            tensor_data_offset,
        }
    }

    /// A minimal GPT-2-style byte-level tokenizer: the ByteLevel decoder
    /// maps each token's byte-unicode characters back to raw bytes, so a
    /// token can carry a single UTF-8 byte — including lone *continuation*
    /// bytes that are invalid UTF-8 on their own.  `\u{c3}` is byte 0xC3,
    /// `\u{a9}` is 0xA9 (together they form `é`), `\u{bc}` is 0xBC
    /// (with 0xC3: `ü`); ASCII letters map to themselves.  Every vocabulary
    /// character must be a genuine GPT-2 byte-unicode table entry — a
    /// character outside the table passes through decoding untouched and
    /// would make the fixture lie about what decoders produce.
    fn byte_level_tokenizer() -> Tokenizer {
        use std::str::FromStr;
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": {"type": "ByteLevel", "add_prefix_space": false,
                        "trim_offsets": true, "use_regex": false},
            "model": {
                "type": "WordLevel",
                "vocab": {"<unk>": 0, "h": 1, "i": 2, "Ã": 3, "©": 4,
                          "¼": 5, "x": 6},
                "unk_token": "<unk>"
            }
        }"#;
        Tokenizer::from_str(json).expect("byte-level fixture tokenizer must parse")
    }

    #[test]
    fn byte_window_decoder_completes_split_codepoints() {
        let tk = byte_level_tokenizer();
        let mut w = ByteWindowDecoder::default();
        let mut response = String::new();
        let mut all = Vec::new();

        // é arrives as two lone bytes (0xC3, 0xA9).
        for t in [3u32, 4] {
            all.push(t);
            response = w.push(&tk, t, response, &all).unwrap();
        }
        assert_eq!(response, "é", "split codepoint must complete: {response:?}");

        // ü arrives right behind it, again as two lone bytes.
        for t in [3u32, 5] {
            all.push(t);
            response = w.push(&tk, t, response, &all).unwrap();
        }
        assert_eq!(response, "éü");
    }

    #[test]
    fn byte_window_decoder_handles_long_identical_token_runs() {
        let tk = byte_level_tokenizer();
        let mut w = ByteWindowDecoder::default();
        let mut response = String::new();
        let mut all = Vec::new();

        // Twelve identical ASCII tokens: slides start at step 9 while every
        // window boundary stays clean, so any double-commit on slide shows
        // up immediately as duplicated characters.
        for _ in 0..12 {
            all.push(1u32); // "h"
            response = w.push(&tk, 1, response, &all).unwrap();
        }
        assert_eq!(response, "hhhhhhhhhhhh");

        // And a completing multi-byte pair right behind the run.
        for t in [3u32, 4] {
            all.push(t);
            response = w.push(&tk, t, response, &all).unwrap();
        }
        assert_eq!(response, "hhhhhhhhhhhhé");
    }

    #[test]
    fn byte_window_decoder_matches_whole_buffer_oracle() {
        let tk = byte_level_tokenizer();
        let mut w = ByteWindowDecoder::default();
        let mut response = String::new();
        let mut all = Vec::new();

        // Deterministic pseudo-random stream over the whole vocabulary:
        // heavy on lone continuation bytes so codepoints repeatedly complete
        // across boundaries, and long enough to slide the 8-token window
        // many times.  The incremental result must equal the old
        // whole-buffer decode after every single step.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for step in 0..400 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Every third step repeats the previous token so the stream
            // contains long identical runs — the shape that exercises
            // clean-boundary window slides.
            let t = if step % 3 == 2 && !all.is_empty() {
                *all.last().unwrap()
            } else {
                (state >> 33) as u32 % 7
            };
            let _ = step;
            all.push(t);
            response = w.push(&tk, t, response, &all).unwrap();
            let oracle = tk.decode(all.as_slice(), false).unwrap();
            assert_eq!(
                response, oracle,
                "incremental text diverged from whole-buffer decode at step {}",
                all.len()
            );
        }
    }

    #[test]
    fn longest_common_prefix_basics() {
        assert_eq!(longest_common_prefix(&[], &[]), 0);
        assert_eq!(longest_common_prefix(&[1], &[]), 0);
        assert_eq!(longest_common_prefix(&[], &[1]), 0);
        // Identical sequences share their full length.
        assert_eq!(longest_common_prefix(&[1, 2, 3], &[1, 2, 3]), 3);
        // Proper common prefix.
        assert_eq!(longest_common_prefix(&[1, 2, 9], &[1, 2, 3, 4]), 2);
        assert_eq!(longest_common_prefix(&[7], &[1, 2, 3]), 0);
    }

    #[test]
    fn is_routed_expert_matches_moe_names() {
        for dense in [
            "token_embd.weight",
            "output.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.ffn_gate_inp.weight",
            "blk.0.ffn_gate_shexp.weight",
            "blk.0.ffn_up_shexp.weight",
            "blk.0.indexer.proj.weight",
            "blk.0.indexer_compressor_gate.weight",
            "blk.0.hc_attn_fn.weight",
        ] {
            assert!(!is_routed_expert(dense), "{dense} should be dense");
        }
        for expert in [
            "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_down_exps.weight",
            "blk.0.ffn_up_exps.weight",
            "blk.42.ffn_gate_exps.weight",
        ] {
            assert!(is_routed_expert(expert), "{expert} should be an expert");
        }
    }

    #[test]
    fn weight_ranges_classifies_and_sizes_interleaved_tensors() {
        // Tensor data offset 4096; the file interleaves dense and expert
        // tensors per layer, so sizes come from offsets of the *next* tensor.
        // The expert is realistically large (5 MiB) so the dense ranges on
        // either side stay separate.
        let header = header_with_tensors(
            4096,
            &[
                ("blk.0.attn_q.weight", 0),
                ("blk.0.ffn_gate_exps.weight", 100),
                ("blk.0.ffn_norm.weight", 5 * 1024 * 1024 + 100),
                ("blk.0.ffn_up_exps.weight", 5 * 1024 * 1024 + 200),
            ],
        );
        let (dense, experts) = weight_ranges(&header, 4096 + 5 * 1024 * 1024 + 300);
        // attn_q 0..100; ffn_norm (5MiB+100)..(5MiB+200)
        assert_eq!(dense, vec![(4096, 100), (4096 + 5 * 1024 * 1024 + 100, 100)]);
        // gate_exps 100..(5MiB+100) and up_exps (5MiB+200)..(5MiB+300) are
        // separated by only the 100-byte norm tensor — a sub-page gap, so the
        // two expert ranges merge into one.
        assert_eq!(experts, vec![(4196, 5243080)]);
    }

    #[test]
    fn weight_ranges_merges_dense_ranges_across_small_expert_gaps() {
        // A tiny expert between two dense tensors leaves a sub-page gap, which
        // madvise/mlock would round over anyway — merge into one range.
        let header = header_with_tensors(
            0,
            &[
                ("blk.0.attn_q.weight", 0),
                ("blk.0.ffn_gate_exps.weight", 100), // 1000-byte expert
                ("blk.0.ffn_norm.weight", 1100),
            ],
        );
        let (dense, experts) = weight_ranges(&header, 1200);
        assert_eq!(dense, vec![(0, 1200)]);
        assert_eq!(experts, vec![(100, 1000)]);

        // A big expert (5 MiB) keeps the dense ranges apart.
        let header = header_with_tensors(
            0,
            &[
                ("blk.0.attn_q.weight", 0),
                ("blk.0.ffn_gate_exps.weight", 100),
                ("blk.0.ffn_norm.weight", 5 * 1024 * 1024 + 100),
            ],
        );
        let (dense, _) = weight_ranges(&header, 5 * 1024 * 1024 + 200);
        assert_eq!(dense, vec![(0, 100), (5 * 1024 * 1024 + 100, 100)]);
    }

    #[test]
    fn weight_ranges_adds_tensor_data_offset() {
        let header = header_with_tensors(
            0x1_0000,
            &[("token_embd.weight", 0), ("blk.0.ffn_down_exps.weight", 64)],
        );
        let (dense, experts) = weight_ranges(&header, 0x1_0000 + 128);
        assert_eq!(dense, vec![(0x1_0000, 64)]);
        assert_eq!(experts, vec![(0x1_0000 + 64, 64)]);
    }

    #[test]
    fn weight_ranges_merge_gap_follows_the_base_page_size() {
        // An 8 KiB expert between two dense tensors is sub-page on a 16 KiB
        // system (Apple Silicon macOS) but spans two 4 KiB pages — so the
        // same file merges into one dense range there and stays split on
        // classic 4 KiB Linux.  Either way the expert range itself is
        // classified exactly once.
        let header = header_with_tensors(
            0,
            &[
                ("blk.0.attn_q.weight", 0),
                ("blk.0.ffn_gate_exps.weight", 100), // 8 KiB expert
                ("blk.0.ffn_norm.weight", 100 + 8192),
            ],
        );
        let size = 100usize + 8192 + 100;
        let (dense_16k, experts) = weight_ranges_with_page(&header, size as u64, 16 * 1024);
        assert_eq!(
            dense_16k,
            vec![(0, size)],
            "8 KiB gap is sub-page at 16 KiB"
        );
        assert_eq!(experts, vec![(100, 8192)]);
        let (dense_4k, _) = weight_ranges_with_page(&header, size as u64, 4096);
        assert_eq!(
            dense_4k,
            vec![(0, 100), (100 + 8192, 100)],
            "an 8 KiB gap spans pages at 4 KiB"
        );
    }

    /// The production split must merge at *base*-page granularity: that is
    /// what `madvise`/`mlock` round to, and it keeps the dense/expert
    /// classification intact even when transparent huge pages are in play.
    #[test]
    fn base_page_size_matches_the_system() {
        let page = base_page_size();
        assert!(page >= 4096 && page.is_power_of_two(), "got {page}");
        #[cfg(unix)]
        {
            let sys = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
            let expected = if sys >= 4096 && sys.is_power_of_two() {
                sys
            } else {
                4096
            };
            assert_eq!(page, expected);
        }
    }

    /// The advice/mlock path must never fail the load, even when the mapping
    /// or the memlock limit is small (warnings only).
    #[cfg(unix)]
    #[test]
    fn apply_hot_weight_pinning_is_best_effort_on_a_real_mapping() {
        let (path, _) = model_fixture("pin.gguf", &vec![0u8; 16 * 4096]);
        let file = File::open(&path).expect("reopen fixture");
        let mmap = unsafe { Mmap::map(&file) }.expect("mmap fixture");

        // Empty tensor table: nothing to do, still Ok.
        let empty = crate::gguf_ext::GgufHeader {
            version: 3,
            metadata: std::collections::HashMap::new(),
            tensors: std::collections::HashMap::new(),
            tensor_data_offset: 0,
        };
        apply_hot_weight_pinning(&mmap, &empty, true, MlockMode::On).expect("no tensors is fine");

        // Real split: prefetch + lock, plus the random hint on experts.
        let header = header_with_tensors(
            0,
            &[
                ("token_embd.weight", 0),
                ("blk.0.ffn_gate_exps.weight", 4096),
            ],
        );
        apply_hot_weight_pinning(&mmap, &header, true, MlockMode::On)
            .expect("advice/mlock is best effort");

        let _ = std::fs::remove_file(&path);
    }

    /// Whole-model prefetch must never fail the load: `MADV_WILLNEED` over
    /// the whole mapping is advisory, and a mapping that supports advice at
    /// all accepts it.  The observable contract is a usable mapping of the
    /// full file length whose contents are readable.
    #[cfg(unix)]
    #[test]
    fn map_model_prefetch_whole_is_best_effort_and_readable() {
        let payload = vec![7u8; 64 * 1024];
        let (path, _) = model_fixture("prefetch.gguf", &payload);

        // With prefetch (and with pinning, which skips the blanket hint too):
        let (mmap, _) = map_model(&path, HugePages::Off, false, MmapMode::Auto, true, true)
            .expect("prefetching map must succeed");
        assert_eq!(mmap.len(), payload.len());
        assert!(mmap[..].iter().all(|&b| b == 7), "contents readable");

        // And without any of it — the plain default path.
        let (mmap, _) = map_model(&path, HugePages::Off, false, MmapMode::Auto, false, false)
            .expect("plain map must still succeed");
        assert_eq!(mmap.len(), payload.len());

        let _ = std::fs::remove_file(&path);
    }

    #[cfg(unix)]
    #[test]
    fn mlock_decision_degrades_or_fails_on_short_limit() {
        let required = 8 * 1024 * 1024 * 1024; // 8 GiB hot set
                                               // On: too-small limit degrades; sufficient or unknown limit proceeds.
        assert_eq!(
            mlock_decision(MlockMode::On, Some(8 * 1024 * 1024), required),
            MlockDecision::Degrade
        );
        assert_eq!(
            mlock_decision(MlockMode::On, Some(required), required),
            MlockDecision::Proceed
        );
        assert_eq!(
            mlock_decision(MlockMode::On, Some(required + 1), required),
            MlockDecision::Proceed
        );
        assert_eq!(
            mlock_decision(MlockMode::On, None, required),
            MlockDecision::Proceed
        );
        // Required: too-small limit fails the load; sufficient or unknown
        // limit proceeds (an unreadable limit is not fatal — runtime mlock
        // failures still get reported).
        assert_eq!(
            mlock_decision(MlockMode::Required, Some(8 * 1024 * 1024), required),
            MlockDecision::Fail
        );
        assert_eq!(
            mlock_decision(MlockMode::Required, Some(required), required),
            MlockDecision::Proceed
        );
        assert_eq!(
            mlock_decision(MlockMode::Required, None, required),
            MlockDecision::Proceed
        );
    }

    #[cfg(unix)]
    #[test]
    fn memlock_limit_bytes_reads_without_panicking() {
        // The value is environment-dependent; this only pins down that the
        // syscall path works (returns Some for a finite limit, None for
        // unlimited — never a panic).
        if let Some(bytes) = memlock_limit_bytes() {
            assert!(bytes > 0, "a finite memlock limit must be positive");
        }
    }

    #[cfg(unix)]
    #[test]
    fn aligned_range_bytes_rounds_to_pages_and_clamps() {
        let page = 4096;
        // Exact page already aligned.
        assert_eq!(aligned_range_bytes(&[(0, 4096)], page, 1 << 20), 4096);
        // Sub-page range still locks the whole page.
        assert_eq!(aligned_range_bytes(&[(100, 100)], page, 1 << 20), 4096);
        // Two disjoint ranges lock two pages each... on distinct pages.
        assert_eq!(
            aligned_range_bytes(&[(0, 100), (8192, 100)], page, 1 << 20),
            2 * 4096
        );
        // Ranges on the same page merge into one page of locked memory.
        assert_eq!(
            aligned_range_bytes(&[(0, 100), (100, 100)], page, 1 << 20),
            4096
        );
        // Clamped to the mapping: a range running past the end locks only
        // what exists (and the partial tail page still counts).
        assert_eq!(aligned_range_bytes(&[(0, 1 << 20)], page, 4096 + 10), 8192);
    }

    /// Write `bytes` to a temp file, returning its path and an open handle.
    fn model_fixture(name: &str, bytes: &[u8]) -> (PathBuf, File) {
        use std::io::Write;
        let path =
            std::env::temp_dir().join(format!("joshua-mapcheck-{}-{name}", std::process::id()));
        let mut f = File::create(&path).expect("create fixture");
        f.write_all(bytes).expect("write fixture");
        drop(f);
        let opened = File::open(&path).expect("open fixture");
        (path, opened)
    }

    #[test]
    fn compressed_model_warns_by_default_and_errors_when_mmap_is_explicit() {
        // A gzip stream that happens to be named `.gguf`.
        let (path, file) = model_fixture("gz.gguf", b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03");

        // Implicit mapping: complain in the log, but let the load proceed and
        // fail (or not) on its own terms.
        check_mappable(&path, &file, MmapMode::Auto, true).expect("warn, not fail");

        // Explicit request: refuse, naming the format and the way out.
        let err = check_mappable(&path, &file, MmapMode::Required, true).unwrap_err();
        assert!(matches!(err, JoshuaError::ModelLoad(_)));
        let msg = err.to_string();
        assert!(msg.contains("gzip"), "got: {msg}");
        assert!(msg.contains("gunzip"), "got: {msg}");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn plain_model_file_passes_the_mappability_check() {
        let (path, file) = model_fixture("plain.gguf", b"GGUF\x03\x00\x00\x00");
        check_mappable(&path, &file, MmapMode::Auto, true).expect("plain GGUF is fine");
        check_mappable(&path, &file, MmapMode::Required, true).expect("plain GGUF is fine");
        let _ = std::fs::remove_file(&path);
    }

    /// Filesystem-level compression only matters for a mapping that faults
    /// through the file; the huge-page path copies the model in one pass.
    #[cfg(unix)]
    #[test]
    fn filesystem_compression_is_ignored_for_the_anonymous_copy() {
        let path = std::env::temp_dir().join(format!(
            "joshua-mapcheck-{}-sparse.gguf",
            std::process::id()
        ));
        let mut f = File::create(&path).expect("create sparse fixture");
        // A written prefix plus a hole for the rest: allocation-poor in the
        // same way a transparently compressed file is, and creatable anywhere.
        std::io::Write::write_all(&mut f, &[0x5Au8; 64 * 1024]).expect("write prefix");
        f.set_len(64 * 1024 * 1024).expect("set_len");
        drop(f);
        let file = File::open(&path).expect("open sparse fixture");

        // Not every filesystem reports a sparse allocation; skip where the
        // condition under test cannot be created.
        if !matches!(
            crate::compression::detect_gguf(&file),
            Some(crate::compression::Compression::Filesystem { .. })
        ) {
            let _ = std::fs::remove_file(&path);
            return;
        }

        // Anonymous copy: not this check's problem, even when mmap is required.
        check_mappable(&path, &file, MmapMode::Required, false).expect("copy path is unaffected");
        // File-backed mapping: warn by default, refuse when asked explicitly.
        check_mappable(&path, &file, MmapMode::Auto, true).expect("warn, not fail");
        assert!(check_mappable(&path, &file, MmapMode::Required, true).is_err());

        let _ = std::fs::remove_file(&path);
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
