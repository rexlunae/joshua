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
//! | `chatglm` / `glm4` / `glm4moe` | ChatGLM2/3, GLM-4, GLM-4.5 – 4.7
//! | `glm-dsa` / `glm5next` | GLM-5 / 5.1 / 5.2, GLM-5.3-Flash
//! | `kimi-linear` / `kimi-k3` | Kimi-Linear, Kimi K3
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
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use llguidance::Matcher;
use memmap2::Mmap;
use rand::rngs::StdRng;
use rand::{thread_rng, Rng, SeedableRng};
use tokenizers::Tokenizer;

use crate::embedding::EmbeddingModel;
use crate::model::{Architecture, ModelSnapshot, QuantizedModel};
use crate::npu::{NpuBackend, NpuSession};
use crate::constraints::{ConstraintEngine, SimpleVob};
use crate::speculative::{
    next_draft_len, NgramDrafter, SpeculationGate, SpeculativeConfig, SpeculativeStats,
    TokenDistRef, Verdict,
};
pub use crate::placement::{DensePlacement, EngramTier, ExpertPlacement, SsdExpertCache, VramExpertCache};
use crate::template::ChatTemplate;

use crate::error::{JoshuaError, Result};
use crate::types::{ChatMessage, GenerationOptions, Tool, UsageInfo};

pub mod external_draft;

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
    /// OpenCL (any vendor ICD: Intel iGPU/Arc, NVIDIA via the OpenCL ICD,
    /// ...).  Requires the `opencl` cargo feature and a system libOpenCL.
    OpenCl,
    /// Vulkan (any ICD: AMD RADV, Intel ANV, ...).  Requires the `vulkan`
    /// cargo feature and a system libvulkan.so.  Bring-up target is the AMD
    /// Renoir iGPU (unified memory, host-visible+coherent round-trip fast path).
    Vulkan,
    /// SYCL 2020 (Intel oneAPI/LLVM `--features sycl` bridge looped through a
    /// dynamically loaded C++ library — no SYCL toolchain needed to build the
    /// Rust crate).  Requires the `sycl` cargo feature and a `libjoshua_sycl`
    /// bridge built with a SYCL compiler.  Use `--device sycl`, or set
    /// `JOSHUA_SYCL_LIBRARY` to point at a non-standard bridge path.
    Sycl,
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
    /// Optional byte cap on the mapped weight pages locked by
    /// `mlock_hot_weights`. This permits a resident subset when even the
    /// dense weights exceed RAM. Whole pages are selected in file order;
    /// routed experts and an untied input embedding are excluded. Leave
    /// room outside this budget for activations, KV state and disk staging.
    /// The cap is reduced when necessary to leave at least 3 GiB of detected
    /// available RAM at load; longer contexts can require more headroom.
    /// `None` preserves locking the full dense set; `Some(0)` locks nothing.
    pub mlock_weight_budget_bytes: Option<u64>,
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
    /// Size the hot-expert cache automatically at load (phase 5 of the
    /// device expert cache design): the budget is computed from available
    /// memory, the model's per-expert bytes and the residency backend's
    /// capacity via [`crate::placement::expert_budget_for_memory`], instead
    /// of taking [`EngineOptions::pin_hot_experts`] verbatim.  `auto` sizing
    /// wins when set; `pin_hot_experts` remains the explicit override.
    pub expert_cache_auto: bool,
    /// Bounded VRAM expert cache over the host expert pool (#62): upload a
    /// subset of the routed experts onto the device and run them there,
    /// falling back to the host expert kernels for the rest.
    ///
    /// [`VramExpertCache::Unset`] (the default) leaves the decision to the
    /// engine: an `--expert-placement device` request on an architecture
    /// whose device expert form *is* this cache (deepseek4 on OpenCL/SYCL)
    /// implies it, and (#110) an `auto` placement that fell back to the host
    /// engages it when the leftover device memory is worth a cache and the
    /// device is not slower than CPU-BLAS on decode.
    /// [`VramExpertCache::Auto`] (`--vram-expert-cache auto`) forces it sized
    /// from the device's free memory, [`VramExpertCache::Bytes(n)`] fixes the
    /// budget, and [`VramExpertCache::Disabled`] (`--vram-expert-cache 0`) is
    /// the explicit opt-out (all routed experts stay on the host).
    pub device_expert_cache: VramExpertCache,
    /// Tier-3 NVMe/SSD staging slab over the routed experts (`--ssd-expert-cache`):
    /// a bounded host-RAM slab that `pread(2)`-streams a hot subset of the
    /// routed experts off the SSD (see [`crate::residency::NvmeResidency`]).
    ///
    /// [`SsdExpertCache::Unset`] (the default) leaves the decision to the
    /// engine; [`SsdExpertCache::Auto`] (`--ssd-expert-cache auto`) sizes the
    /// slab from the host's available RAM at load, [`SsdExpertCache::Bytes(n)`]
    /// fixes the budget, and [`SsdExpertCache::Disabled`] (`--ssd-expert-cache 0`)
    /// is the explicit opt-out.
    pub ssd_expert_cache: SsdExpertCache,
    /// Tokens per prefill chunk; `0` selects [`DEFAULT_PREFILL_CHUNK`].
    ///
    /// A prompt is prefilled in chunks of this many tokens.  The chunk bounds
    /// the per-layer transient workspace (attention scores, MoE dispatch
    /// buffers) and, for the MoE loaders, sets how many prompt rows each
    /// routed expert sees per matmul: a larger chunk batches more rows per
    /// expert (fewer, wider kernel launches on a device expert cache) at the
    /// cost of a larger per-layer workspace.  The joshua-native MoE loaders
    /// stream the prefill layer by layer, so each layer's weights are read
    /// once per prefill whatever the chunk; on the other architectures every
    /// chunk re-reads the whole model, so a larger chunk also means fewer
    /// passes over the weights.
    pub prefill_chunk: usize,
    /// Target wall time for one prefill chunk, in milliseconds; `0` (the
    /// default) keeps the fixed [`EngineOptions::prefill_chunk`] size.
    ///
    /// When set, architectures that prefill chunk by chunk (the ones without
    /// a layer-streaming path) measure each completed chunk and pick the next
    /// size from an online cost model, aiming for this budget.  Chunk sizes
    /// chosen from wall time can change floating-point results between runs
    /// for the same prompt and seed, so this is opt-in.  The configured
    /// `prefill_chunk` is the ceiling and the size used for the first probe.
    pub prefill_budget_ms: u64,
    /// How long a request may wait for a concurrency permit before returning
    /// the overloaded error, in milliseconds; `0` (the default) rejects
    /// immediately.
    ///
    /// A short wait smooths bursts instead of failing them, but a queue only
    /// delays the error when the engine is genuinely saturated: the wait is
    /// bounded per request and the concurrency cap still bounds memory.
    pub queue_timeout_ms: u64,
    /// How many requests may wait for a concurrency permit when
    /// [`EngineOptions::queue_timeout_ms`] is non-zero; `0` selects
    /// [`DEFAULT_ADMISSION_QUEUE_DEPTH`].
    ///
    /// The concurrency cap bounds inference memory, but every parked request
    /// occupies a server thread; a finite queue keeps a burst from exhausting
    /// the blocking pool.  A request that finds the queue full gets the same
    /// overloaded error as one that times out.
    pub admission_queue_depth: usize,
    /// Prefix checkpoints each session of a model with recurrent layers
    /// keeps (Qwen3-Next, Qwen3.5, `qwen4exp`, the KDA hybrids, and
    /// DeepSeek-V4/V4.1); `None` selects
    /// [`crate::native_session::DEFAULT_PREFIX_CHECKPOINTS`] and `Some(0)`
    /// turns them off.
    ///
    /// Such a model's running state cannot be cut back to an arbitrary
    /// prefix, so an edited conversation (an agent replacing its last
    /// message or an old tool result) used to re-read the whole prompt.
    /// Each checkpoint is a copy of every recurrent layer's state at the end
    /// of a prefill — a turn boundary — that the edit resumes from instead.
    /// Each costs one copy of that state per pooled session.
    pub prefix_checkpoints: Option<usize>,
    /// Host RAM the prefix cache may hold, in bytes; `None` sizes it from free
    /// RAM at start ([`auto_prefix_cache_bytes`]) and `Some(0)` turns it off.
    ///
    /// The cache keeps detached copies of conversation state between
    /// requests (see [`crate::prefix_cache`]): the state at the end of each
    /// client's system prompt, so a new chat skips re-reading it, and the
    /// state of conversations evicted from the warm session pool, so they
    /// resume later.  Qwen-family and `deepseek2` models only.
    pub prefix_cache_bytes: Option<u64>,
    /// Shortest shared prompt head, in tokens, worth keeping as a prefix-cache
    /// root; `None` selects [`DEFAULT_PREFIX_CACHE_ROOT_MIN`].  Below it the
    /// head is cheaper to read again than its copy costs.
    pub prefix_cache_root_min: Option<usize>,
    /// Move an `<env>…</env>` block (working directory, today's date — what
    /// opencode writes near the start of its system prompt) out of the system
    /// message into the front of the first user message.  Off by default: it
    /// changes the prompt.  With it, a new folder or a new day leaves the
    /// system prompt, and so the prefix cache's root, unchanged; Strata
    /// measured opencode's first request going from 11.9 s to 1.1 s.
    pub env_to_user: bool,
    /// Where a sparse MoE model's routed experts live when inference runs on
    /// an accelerator: uploaded to the device, or kept in host RAM (borrowed
    /// from the mapping, run on the CPU expert kernels, activations hopping
    /// across per layer).  [`ExpertPlacement::Auto`] (the default) uploads
    /// them only when the whole model fits the device's free memory with
    /// headroom — probed with `cudaMemGetInfo` on CUDA, or taken from
    /// [`EngineOptions::device_memory_budget`] — and otherwise keeps them on
    /// the host, which is what runs a model larger than VRAM.  Ignored on the
    /// CPU device.  Wired for `qwen3moe` and `deepseek2`; `deepseek4` always
    /// keeps a mapped model's experts on the host.
    pub expert_placement: ExpertPlacement,
    /// Where the dense set (embeddings, norms, attention, routers, shared
    /// experts, output) lives on an accelerator: the device itself, or the CPU.
    /// [`DensePlacement::Auto`] (the default) keeps dense on the device —
    /// right for a real accelerator whose GEMM beats CPU-BLAS.  On a weak
    /// iGPU / software OpenCL backend the device GEMM can be ~50x *slower*
    /// than CPU-BLAS, so [`DensePlacement::Cpu`] keeps the dense set on the
    /// host (the routed-expert placement is unaffected).  Ignored on the CPU
    /// device.
    pub dense_placement: DensePlacement,
    /// Where the DeepSeek-V4.1 Engram n-gram tables are resident.  A separate
    /// placement policy from the routed experts (see
    /// [`crate::placement::EngramTier`]): the tables are always resolved
    /// straight from the mmap mapping — never a Tier-1 VRAM slot or a Tier-2
    /// host hot-expert slot — and this only controls whether a working subset
    /// is additionally promoted to RAM (`--engram-tier ram` / ample RAM on
    /// `auto`) or pinned to disk (the default, `--engram-tier disk`).
    pub engram_tier: EngramTier,
    /// Bytes of accelerator memory the model may use, overriding the probe
    /// for the [`ExpertPlacement::Auto`] decision and the per-device session
    /// cap.  `None` uses the probe where one exists.  This is the knob for
    /// backends without a probe (Metal's unified memory, OpenCL) and for
    /// sharing a card with other processes.
    pub device_memory_budget: Option<u64>,
    /// Speculative token generation (draft-and-verify decoding).  `None`
    /// (the default) decodes one token per forward pass.
    ///
    /// With `Some(config)`, each decode step drafts up to
    /// [`SpeculativeConfig::max_draft`] tokens by prompt lookup (the
    /// continuation of the latest earlier occurrence of the context's
    /// trailing n-gram), scores them all in one forward pass and keeps the
    /// prefix the model agrees with, rolling the KV cache back past the
    /// first rejection.  Output is unchanged — token-identical under greedy
    /// decoding, identically distributed under sampling — while repetitive
    /// output (code edits, quoted context, tool-call arguments) needs far
    /// fewer weight sweeps.  See [`crate::speculative`].
    ///
    /// Applies to the architectures whose loaders can score every position
    /// and roll their KV cache back (`qwen3moe`, `deepseek2`, `deepseek4` —
    /// see [`crate::model::Architecture::supports_speculative_decoding`]);
    /// on the others — candle's stock loaders and the recurrent DeltaNet /
    /// KDA hybrids — the request warns at load and is ignored.
    pub speculative: Option<SpeculativeConfig>,
    /// MTP drafting: up to this many draft tokens per decode step from the
    /// model's own NextN / MTP block (the `blk.<n_layer>.nextn.*` tensors
    /// GLM-4.5+ and Qwen3.5/3.6 GGUFs carry), verified like prompt-lookup
    /// drafts, so output is unchanged.  `0` (the default) turns it off.  On a
    /// model with the block it replaces prompt lookup; elsewhere it is
    /// ignored with a warning.  Each forward pass also runs its rows through
    /// the block (one more layer) to keep the block's KV current.  The cost
    /// gate of [`Self::speculative`] (on unless that config turns it off)
    /// holds drafting off while it measures slower than plain decoding.
    pub mtp: usize,
    /// A separate draft model for speculative decoding: a small GGUF that
    /// shares this model's tokenizer proposes up to [`Self::draft_max`]
    /// tokens per step, which this model verifies in one forward pass.
    /// Output is token-identical to plain decoding: a sampled draft is kept
    /// only when it equals the target's own draw at that position, and a
    /// rejection emits that draw, so the draft model can change how many
    /// tokens are accepted but never which tokens are produced.  The draft
    /// must be an attention-only model one of Joshua's own loaders serves
    /// (qwen3,
    /// qwen3moe, glm4, deepseek2, …).  Takes the place of prompt lookup and
    /// MTP; the cost gate of [`Self::speculative`] (on unless that config
    /// turns it off) applies.  See [`crate::draft_model`].
    pub draft_model: Option<PathBuf>,
    /// Most draft-model tokens per step (`0`: the default,
    /// [`crate::draft_model::DEFAULT_DRAFT_MODEL_MAX`]).  The live length
    /// adapts between 1 and this bound from the acceptance observed.
    pub draft_max: usize,
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

    /// Cap the resident mapped weight subset in bytes. See
    /// [`EngineOptions::mlock_weight_budget_bytes`].
    pub fn mlock_weight_budget(mut self, bytes: Option<u64>) -> Self {
        self.mlock_weight_budget_bytes = bytes;
        self
    }

    /// Keep the `n` most frequently routed experts resident.  See
    /// [`EngineOptions::pin_hot_experts`].
    pub fn pin_hot_experts(mut self, n: usize) -> Self {
        self.pin_hot_experts = n;
        self
    }

    /// Size the hot-expert cache automatically at load.  See
    /// [`EngineOptions::expert_cache_auto`].
    pub fn expert_cache_auto(mut self, auto: bool) -> Self {
        self.expert_cache_auto = auto;
        self
    }

    /// Set the bounded VRAM expert-cache setting.  See
    /// [`EngineOptions::device_expert_cache`].
    pub fn vram_expert_cache(mut self, setting: VramExpertCache) -> Self {
        self.device_expert_cache = setting;
        self
    }

    /// Set the Tier-3 NVMe/SSD staging-slab setting.  See
    /// [`EngineOptions::ssd_expert_cache`].
    pub fn ssd_expert_cache(mut self, setting: SsdExpertCache) -> Self {
        self.ssd_expert_cache = setting;
        self
    }

    /// Set the tokens per prefill chunk (`0` = the default).  See
    /// [`EngineOptions::prefill_chunk`].
    pub fn prefill_chunk(mut self, tokens: usize) -> Self {
        self.prefill_chunk = tokens;
        self
    }
    /// Set the target wall time for one prefill chunk (`0` = fixed size).
    /// See [`EngineOptions::prefill_budget_ms`].
    pub fn prefill_budget_ms(mut self, millis: u64) -> Self {
        self.prefill_budget_ms = millis;
        self
    }

    /// Set how long a request may wait for a concurrency permit (`0` =
    /// reject immediately).  See [`EngineOptions::queue_timeout_ms`].
    pub fn queue_timeout_ms(mut self, millis: u64) -> Self {
        self.queue_timeout_ms = millis;
        self
    }

    /// Set the admission queue depth (`0` = the default).  See
    /// [`EngineOptions::admission_queue_depth`].
    pub fn admission_queue_depth(mut self, depth: usize) -> Self {
        self.admission_queue_depth = depth;
        self
    }

    /// Set how many prefix checkpoints a recurrent model's session keeps
    /// (`0` = off).  See [`EngineOptions::prefix_checkpoints`].
    pub fn prefix_checkpoints(mut self, n: usize) -> Self {
        self.prefix_checkpoints = Some(n);
        self
    }

    /// [`Self::prefix_checkpoints`] when `n` is set; `None` keeps the default.
    pub fn prefix_checkpoints_opt(mut self, n: Option<usize>) -> Self {
        self.prefix_checkpoints = n.or(self.prefix_checkpoints);
        self
    }

    /// Set the prefix cache's byte budget (`0` = off).  See
    /// [`EngineOptions::prefix_cache_bytes`].
    pub fn prefix_cache_bytes(mut self, bytes: u64) -> Self {
        self.prefix_cache_bytes = Some(bytes);
        self
    }

    /// Set the shortest prompt head kept as a prefix-cache root.  See
    /// [`EngineOptions::prefix_cache_root_min`].
    pub fn prefix_cache_root_min(mut self, tokens: usize) -> Self {
        self.prefix_cache_root_min = Some(tokens);
        self
    }

    /// [`Self::prefix_cache_bytes`] in MiB when set; `None` keeps the default.
    pub fn prefix_cache_mib_opt(mut self, mib: Option<u64>) -> Self {
        if let Some(mib) = mib {
            self.prefix_cache_bytes = Some(mib.saturating_mul(1024 * 1024));
        }
        self
    }

    /// [`Self::prefix_cache_root_min`] when set; `None` keeps the default.
    pub fn prefix_cache_root_min_opt(mut self, tokens: Option<usize>) -> Self {
        self.prefix_cache_root_min = tokens.or(self.prefix_cache_root_min);
        self
    }

    /// Move an `<env>` block out of the system prompt.  See
    /// [`EngineOptions::env_to_user`].
    pub fn env_to_user(mut self, on: bool) -> Self {
        self.env_to_user = on;
        self
    }

    /// Choose where the routed experts of a MoE model live on an
    /// accelerator.  See [`EngineOptions::expert_placement`].
    pub fn expert_placement(mut self, placement: ExpertPlacement) -> Self {
        self.expert_placement = placement;
        self
    }

    /// Choose where the dense set lives on an accelerator.  See
    /// [`EngineOptions::dense_placement`].
    pub fn dense_placement(mut self, placement: DensePlacement) -> Self {
        self.dense_placement = placement;
        self
    }

    /// Choose where the DeepSeek-V4.1 Engram n-gram tables are resident.  See
    /// [`EngineOptions::engram_tier`].
    pub fn engram_tier(mut self, tier: EngramTier) -> Self {
        self.engram_tier = tier;
        self
    }

    /// Cap the accelerator memory the model may use (bytes).  See
    /// [`EngineOptions::device_memory_budget`].
    pub fn device_memory_budget(mut self, bytes: Option<u64>) -> Self {
        self.device_memory_budget = bytes;
        self
    }

    /// Enable (or, with `None`, disable) speculative token generation.  See
    /// [`EngineOptions::speculative`].
    pub fn speculative(mut self, config: Option<SpeculativeConfig>) -> Self {
        self.speculative = config;
        self
    }

    /// Draft up to `n` tokens per step with the model's MTP block (`0`: off).
    /// See [`EngineOptions::mtp`].
    pub fn mtp(mut self, n: usize) -> Self {
        self.mtp = n;
        self
    }

    /// Draft with a separate small model (`None`: off).  See
    /// [`EngineOptions::draft_model`].
    pub fn draft_model(mut self, path: Option<PathBuf>) -> Self {
        self.draft_model = path;
        self
    }

    /// Most draft-model tokens per step (`0`: the default).  See
    /// [`EngineOptions::draft_max`].
    pub fn draft_max(mut self, n: usize) -> Self {
        self.draft_max = n;
        self
    }
}

// ─── Env flags ───────────────────────────────────────────────────────────────

/// Debug logging of every sampled token.  Read once: this sits on the
/// per-token decode path, and an `env::var_os` lookup there would lock the
/// environment and allocate on every sample.
fn debug_tokens() -> bool {
    static D: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *D.get_or_init(|| std::env::var_os("JOSHUA_DEBUG_TOKENS").is_some())
}

/// Whether temperature-0 sampling applies the repetition penalty (the
/// pre-legacy behaviour).  Read once: this sits on the per-sample path.
fn legacy_reppen() -> bool {
    static L: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *L.get_or_init(|| std::env::var_os("JOSHUA_LEGACY_REPPEN").is_some())
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
    /// The `tokenizer.json` this engine was loaded from; used to build the
    /// structured-output grammar environment lazily.
    tokenizer_path: PathBuf,
    /// Compiled JSON grammars, built on first constrained request.
    constraints: std::sync::OnceLock<Result<ConstraintEngine, String>>,
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
    /// The tokenizer's control-token strings, longest first.
    ///
    /// Message text is escaped against these before rendering so a user or
    /// tool message cannot inject a turn boundary (see
    /// [`crate::template::sanitize_messages`]).
    special_tokens: Vec<String>,
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
    /// cache), so this is capped at `max_concurrency` to bound peak memory.
    /// A request over the cap may wait up to `queue_timeout` for a permit
    /// and is then rejected.
    in_flight: AdmissionGate,
    /// Maximum concurrent generations/embeddings.
    max_concurrency: usize,
    /// How long a request waits for a permit (zero = reject immediately).
    queue_timeout: Duration,
    /// How many requests may wait for a permit at once.
    admission_queue_depth: usize,
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
    /// Whether the budget is sized automatically at load (see
    /// [`EngineOptions::expert_cache_auto`]).
    expert_cache_auto: bool,
    /// Resolved bounded VRAM expert-cache byte budget the loaders get (see
    /// [`EngineOptions::device_expert_cache`]); `None` disables.
    device_expert_cache: Option<u64>,
    /// The Tier-3 `--ssd-expert-cache` policy the loader resolves into an NVMe
    /// staging-slab budget at load (see [`EngineOptions::ssd_expert_cache`]).
    ssd_expert_cache: crate::placement::SsdExpertCache,
    /// Where a DeepSeek-V4.1 model's Engram n-gram tables are resident (a
    /// separate placement policy from the routed experts; see
    /// [`EngineOptions::engram_tier`]).
    engram_tier: EngramTier,
    /// Tokens per prefill chunk (see [`EngineOptions::prefill_chunk`]),
    /// resolved to at least one token.
    prefill_chunk: usize,
    /// Target prefill-chunk time (see [`EngineOptions::prefill_budget_ms`]);
    /// zero keeps the fixed chunk size.
    prefill_budget: Duration,
    /// Online fit used to choose adaptive prefill chunks.
    prefill_cost: Mutex<crate::prefill_cost::PrefillCostModel>,
    /// Prefix checkpoints per recurrent-model session (see
    /// [`EngineOptions::prefix_checkpoints`]).
    prefix_checkpoints: usize,
    /// Parked conversation state (see [`EngineOptions::prefix_cache_bytes`]).
    prefix_cache: Mutex<crate::prefix_cache::PrefixCache<ModelSnapshot>>,
    /// Shortest prompt head kept as a prefix-cache root (see
    /// [`EngineOptions::prefix_cache_root_min`]).
    prefix_cache_root_min: usize,
    /// The order each recently seen tool set first came in (see
    /// [`crate::prompt_stability::ToolOrders`]).
    tool_orders: crate::prompt_stability::ToolOrders,
    /// See [`EngineOptions::env_to_user`].
    env_to_user: bool,
    /// Whether the routed experts page in on demand (the model was loaded as
    /// larger than RAM: [`EngineOptions::pin_hot_weights`] or
    /// [`EngineOptions::lazy_weights`]), so decode prefetches each layer's
    /// predicted experts.
    predict_experts: bool,
    /// Compute device: CUDA or Metal when built with the matching feature
    /// (falling back to CPU if unavailable at runtime), CPU otherwise.
    device: Device,
    /// Device the dense set runs on: `device`, or the CPU when the operator
    /// forced [`EngineOptions::dense_placement`] to `Cpu` (e.g. a weak iGPU
    /// whose accelerator GEMM is slower than CPU-BLAS).  The dense weights are
    /// laid out on this device and the accelerator is unused for them.
    dense_device: Device,
    /// Device the routed-expert pool of a MoE model is built on: `device`,
    /// or the CPU when the experts stay in host RAM (see
    /// [`EngineOptions::expert_placement`]).
    expert_device: Device,
    /// The one fully loaded instance of a weight-sharing architecture
    /// (`qwen3moe`, `deepseek2`, `deepseek4`), never used for inference
    /// itself: every session is derived from it with
    /// [`QuantizedModel::new_session`] and shares its weights, so concurrent
    /// requests cost one KV cache each rather than one copy — on an
    /// accelerator, one upload — of the model.  `None` until the first load,
    /// and always for architectures whose instances own their weights.
    weights_template: Mutex<Option<QuantizedModel>>,
    /// Cap on idle sessions kept warm for architectures whose sessions each
    /// carry a full copy of the weights on an accelerator, sized from the
    /// device's memory at load (see [`crate::placement::instances_for_memory`]).
    /// `None` leaves the pool on the RAM-adaptive rule.
    device_session_cap: Option<usize>,
    /// Pooled-session cap for a remote backend, whose sessions occupy
    /// bounded worker capacity (see [`Engine::with_remote_backend`]).
    remote_pool_cap: Option<usize>,
    /// Sessions a remote backend holds at once.
    remote_capacity: Option<usize>,
    /// The model's architecture, when candle has a loader for it (see
    /// [`Self::arch_error`]).  Gates per-architecture capabilities at
    /// construction time.
    arch: Option<crate::model::Architecture>,
    /// Speculative decoding settings (see [`EngineOptions::speculative`]).
    speculative: Option<SpeculativeConfig>,
    /// MTP drafts per step (see [`EngineOptions::mtp`]).
    mtp: usize,
    /// The separate draft model, when one is loaded (see
    /// [`EngineOptions::draft_model`]).
    draft: Option<crate::draft_model::DraftModel>,
    /// Most draft-model tokens per step.
    draft_max: usize,
    /// Draft tokens verified by speculative decoding.
    spec_drafted: AtomicU64,
    /// Draft tokens the model accepted.
    spec_accepted: AtomicU64,
    /// Multi-token verification passes run.
    spec_verify_steps: AtomicU64,
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

/// How many requests may wait for a concurrency permit when the admission
/// queue is enabled ([`EngineOptions::queue_timeout_ms`] > 0).  Waiting
/// requests occupy a server thread, so the queue must be finite even though
/// the concurrency cap alone bounds inference memory.
pub const DEFAULT_ADMISSION_QUEUE_DEPTH: usize = 64;

/// Minimum free system RAM required to admit a new request (bytes).  Below
/// this the engine refuses new generations (memory-aware admission, #2) rather
/// than letting a heavy load push the working set into swap.  Kept well above
/// a single decode's transient allocations so it only fires under real
/// pressure.
const LOW_MEM_FLOOR: u64 = 1536 * 1024 * 1024; // 1.5 GiB

/// Default tokens per prefill chunk ([`EngineOptions::prefill_chunk`]),
/// bounding the transient activation tensor for a long prompt (finding #4).
/// 512 keeps memory flat for a huge prompt without re-reading expert weights
/// so often that it dominates on the architectures without a layer-streaming
/// prefill.
pub const DEFAULT_PREFILL_CHUNK: usize = 512;

/// Default [`EngineOptions::prefix_cache_root_min`]: a shared prompt head of
/// fewer tokens is not kept as a prefix-cache root.  Strata uses 2,048 on a
/// GPU; a weak CPU box reads a prompt an order of magnitude slower, so a
/// shorter head already pays for its copy.
pub const DEFAULT_PREFIX_CACHE_ROOT_MIN: usize = 1024;

/// Shortest conversation, in tokens, the prefix cache parks when the warm
/// pool lets it go: a shorter one is cheaper to read again.  A lower
/// [`EngineOptions::prefix_cache_root_min`] lowers it too.
const PREFIX_CACHE_PARK_MIN: usize = 256;

/// Most host RAM [`auto_prefix_cache_bytes`] gives the prefix cache.
const PREFIX_CACHE_AUTO_MAX: u64 = 4 * 1024 * 1024 * 1024; // 4 GiB

/// The prefix cache's budget when the operator sets none: an eighth of the
/// RAM free at start, at most [`PREFIX_CACHE_AUTO_MAX`], and nothing when the
/// box is already short of memory or cannot say.  Each store also checks the
/// free RAM at that moment (see [`Engine::park_snapshot`]).
fn auto_prefix_cache_bytes(free: Option<u64>) -> u64 {
    match free {
        Some(free) if free > LOW_MEM_FLOOR * 4 => (free / 8).min(PREFIX_CACHE_AUTO_MAX),
        _ => 0,
    }
}

/// Maximum number of idle model instances kept warm in the pool.
///
/// Each instance holds the (quantized) weights plus its KV cache, so this
/// bounds memory: two instances cover the common "one active conversation
/// plus one concurrent request" pattern without tripling residency.
const MAX_CACHED_MODELS: usize = 2;

/// Upper pool cap used only when free RAM is ample, so several concurrent
/// requests reuse one loaded model (and one device upload) instead of each
/// loading its own (#5).  `release_model` picks between 0 / 1 / MAX_CACHED_MODELS
/// / MAX_CACHED_MODELS_MEM by free RAM.
const MAX_CACHED_MODELS_MEM: usize = 4;

/// Free RAM (bytes) above which the pool keeps `MAX_CACHED_MODELS_MEM` warm
/// sessions; below it degrades toward the baseline and then to 0.
const MAX_CACHED_MODELS_MEM_HEADROOM: u64 = 16 * 1024 * 1024 * 1024; // 16 GiB

/// Consecutive NPU failures before the backend is disabled for the rest of
/// the engine's lifetime (all requests then run on the candle path).
const NPU_MAX_FAILURES: u32 = 3;

/// Default ceiling on tokens generated per request (independent of the
/// client-supplied `max_tokens`), bounding single-request CPU/time cost.
const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 4096;

/// Admission state for in-flight generations/embeddings.
///
/// A request takes a permit before doing any work.  When every permit is in
/// use it may wait on [`Self::released`] for a bounded time
/// ([`EngineOptions::queue_timeout_ms`]); with a zero timeout it fails at
/// once.  Waiting is only about smoothing bursts — the cap still bounds peak
/// memory, and a request that waits too long returns the same overloaded
/// error the immediate path returns.
#[derive(Debug)]
struct AdmissionGate {
    state: Mutex<AdmissionState>,
    released: Condvar,
}

/// Active permits and parked waiters, updated under one lock so the queue
/// bound cannot race with a permit release.
#[derive(Debug, Default)]
struct AdmissionState {
    active: usize,
    waiting: usize,
}

impl AdmissionGate {
    fn new() -> Self {
        Self {
            state: Mutex::new(AdmissionState::default()),
            released: Condvar::new(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, AdmissionState> {
        // A panic while holding the count cannot leave it inconsistent (the
        // only mutation is +=/-= around a permit), so recover the guard.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn active(&self) -> usize {
        self.lock().active
    }

    #[cfg(test)]
    fn waiting(&self) -> usize {
        self.lock().waiting
    }
}

/// Reject before admitting when the machine is critically low on RAM, so a
/// new request cannot push a heavy model load/working set into swap.  Run it
/// again after waiting: memory can fall while a request sits in the queue.
fn check_admission_memory() -> Result<()> {
    if let Some(free) = crate::placement::available_ram_bytes() {
        if free < LOW_MEM_FLOOR {
            return Err(JoshuaError::Overloaded(format!(
                "low memory ({} MiB free < {} MiB floor); retry shortly",
                free / (1024 * 1024),
                LOW_MEM_FLOOR / (1024 * 1024),
            )));
        }
    }
    Ok(())
}

fn at_capacity(max: usize, waited: Option<Duration>) -> JoshuaError {
    let suffix = waited.map_or_else(String::new, |w| format!(" for {} ms", w.as_millis()));
    JoshuaError::Overloaded(format!(
        "at capacity ({max} concurrent requests){suffix}; retry shortly"
    ))
}

/// RAII permit for one in-flight generation/embedding.
///
/// Takes a slot from the gate on acquisition and returns it on drop, so the
/// count is released even if generation errors or panics.
struct InFlightGuard<'a> {
    gate: &'a AdmissionGate,
}

impl<'a> InFlightGuard<'a> {
    fn acquire(gate: &'a AdmissionGate, max: usize, wait: Duration) -> Result<Self> {
        Self::acquire_with(gate, max, DEFAULT_ADMISSION_QUEUE_DEPTH, wait, None)
    }

    /// Take a permit, waiting at most `wait` and at most `queue_depth`
    /// requests deep.  `cancel` is polled while parked so a disconnected
    /// client stops occupying a thread before a slot frees.
    fn acquire_with(
        gate: &'a AdmissionGate,
        max: usize,
        queue_depth: usize,
        wait: Duration,
        cancel: Option<&AtomicBool>,
    ) -> Result<Self> {
        check_admission_memory()?;
        let max = max.max(1);
        let queue_depth = queue_depth.max(1);
        let mut state = gate.lock();
        // Let parked requests go first: a new caller must not barge past a
        // waiter that was woken by the permit it is about to take.
        if state.active < max && state.waiting == 0 {
            state.active += 1;
            return Ok(Self { gate });
        }
        if wait.is_zero() {
            return Err(at_capacity(max, None));
        }
        if state.waiting >= queue_depth {
            return Err(JoshuaError::Overloaded(format!(
                "admission queue is full ({queue_depth} waiting); retry shortly"
            )));
        }
        state.waiting += 1;
        let deadline = Instant::now() + wait;
        let outcome = loop {
            if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                break Err(JoshuaError::Cancelled);
            }
            let now = Instant::now();
            let Some(left) = deadline.checked_duration_since(now) else {
                break Err(at_capacity(max, Some(wait)));
            };
            // Wake periodically even without a release so cancellation is
            // observed promptly.
            let slice = left.min(Duration::from_millis(20));
            let (guard, _) = gate
                .released
                .wait_timeout(state, slice)
                .unwrap_or_else(|e| e.into_inner());
            state = guard;
            if state.active < max {
                if Instant::now() >= deadline {
                    // The slot freed after the deadline; do not admit.
                    break Err(at_capacity(max, Some(wait)));
                }
                state.active += 1;
                break Ok(());
            }
        };
        state.waiting -= 1;
        drop(state);
        // A timed-out or cancelled waiter may have left a slot free for
        // someone else, and a new waiter may have joined; wake them all.
        // (The queue is bounded, so this is not a storm.)
        gate.released.notify_all();
        outcome?;
        if let Err(e) = check_admission_memory() {
            let mut state = gate.lock();
            state.active = state.active.saturating_sub(1);
            drop(state);
            gate.released.notify_all();
            return Err(e);
        }
        Ok(Self { gate })
    }
}

impl Drop for InFlightGuard<'_> {
    fn drop(&mut self) {
        let mut state = self.gate.lock();
        state.active = state.active.saturating_sub(1);
        drop(state);
        self.gate.released.notify_all();
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

    /// Feed `tokens` at absolute position `pos`, returning the logits of
    /// every fed position (row `i` predicts the token after `tokens[i]`).
    /// Only valid when [`GenSession::supports_speculative`] holds.
    fn forward_tokens_all(
        &mut self,
        tokens: &[u32],
        pos: usize,
        device: &Device,
    ) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Candle(model) => {
                let input = Tensor::new(tokens, device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| JoshuaError::Inference(e.to_string()))?;
                model
                    .forward_all_logits(&input, pos)
                    .and_then(|l| l.squeeze(0))
                    .and_then(|l| l.to_vec2::<f32>())
                    .map_err(|e| JoshuaError::Inference(e.to_string()))
            }
            Self::Npu(_) => Err(JoshuaError::Inference(
                "speculative verification is not supported on the NPU session".into(),
            )),
        }
    }

    /// Whether MTP drafting is on for this session.
    fn mtp_enabled(&self) -> bool {
        match self {
            Self::Candle(model) => model.mtp_enabled(),
            Self::Npu(_) => false,
        }
    }

    /// Up to `n` MTP drafts after `next`, the token about to be fed at `pos`.
    fn mtp_draft(&mut self, next: u32, pos: usize, n: usize) -> Result<Vec<u32>> {
        match self {
            Self::Candle(model) => model
                .mtp_draft(next, pos, n)
                .map_err(|e| JoshuaError::Inference(e.to_string())),
            Self::Npu(_) => Ok(Vec::new()),
        }
    }

    /// Whether this session can verify drafts (all-position logits plus KV
    /// truncation) — see [`crate::speculative`].
    fn supports_speculative(&self) -> bool {
        match self {
            Self::Candle(model) => model.supports_speculative(),
            Self::Npu(_) => false,
        }
    }

    /// Layer-streaming prefill over bounded chunks (`None` on architectures
    /// without a native streaming path, so callers fall back to the standard
    /// chunked loop).  Returns the prefill logits vector.
    fn prefill_streamed(
        &mut self,
        chunks: &[crate::stream_prefill::Chunk],
        device: &Device,
    ) -> std::result::Result<Option<Vec<f32>>, JoshuaError> {
        let model = match self {
            Self::Candle(m) => m,
            Self::Npu(_) => return Ok(None),
        };
        if !model.supports_streaming() {
            return Ok(None);
        }
        let logits = model
            .prefill_streamed(chunks, device)
            .map_err(|e| JoshuaError::Inference(e.to_string()))?;
        Ok(Some(
            squeeze_batch_logits(&logits).map_err(|e| JoshuaError::Inference(e.to_string()))?,
        ))
    }

    /// Feed one decode step across `seqs` **independent** sequences at their
    /// own positions, returning one logits vector per sequence.  The MoE
    /// dispatch for the batch is shared, so the routed-expert fetch is
    /// amortized across the sequences (see
    /// `quantized_deepseek4::ModelWeights::forward_sequences`).  Only the
    /// candle deepseek4 model implements this; anything else reports an
    /// unsupported error so callers can fall back to per-sequence steps.
    fn forward_tokens_batched(
        &mut self,
        seqs: &[(&Tensor, usize)],
    ) -> Result<Vec<Vec<f32>>, JoshuaError> {
        match self {
            Self::Candle(model) => model
                .forward_sequences(seqs)
                .map_err(|e| JoshuaError::Inference(e.to_string())),
            Self::Npu(_) => Err(JoshuaError::Inference(
                "batched forward is not supported on the NPU session".into(),
            )),
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

    /// The longest prefix of at most `keep` tokens [`Self::truncate_to`] can
    /// rewind this session to (`0`: none).  `keep` itself for an
    /// attention-only model; the latest prefix checkpoint at or before it
    /// for one with recurrent layers.
    fn rewind_target(&self, keep: usize) -> usize {
        match self {
            Self::Candle(model) => model.kv_rewind_target(keep),
            Self::Npu(_) => 0,
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

    /// A detached host copy of the state after `keep` of the `len` tokens fed
    /// (see [`QuantizedModel::snapshot`]); `None` where none can be had.
    fn snapshot(&self, keep: usize, len: usize) -> Option<ModelSnapshot> {
        match self {
            Self::Candle(model) => match model.snapshot(keep, len, &Device::Cpu) {
                Ok(snap) => snap,
                Err(e) => {
                    tracing::debug!(error = %e, keep, "prefix cache: snapshot failed");
                    None
                }
            },
            Self::Npu(_) => None,
        }
    }

    /// Whether [`Self::snapshot`] can detach this session's state.
    fn supports_snapshot(&self) -> bool {
        match self {
            Self::Candle(model) => model.supports_snapshot(),
            Self::Npu(_) => false,
        }
    }

    /// Bytes the session's state holds: what a snapshot copies at most.
    fn state_bytes(&self) -> u64 {
        match self {
            Self::Candle(model) => model.state_bytes(),
            Self::Npu(_) => 0,
        }
    }

    /// Continue from `snap`; `false` when the session cannot take it.  A
    /// failure part-way leaves the state unknown, so the caller clears it.
    fn restore(&mut self, snap: ModelSnapshot) -> bool {
        match self {
            Self::Candle(model) => match model.restore_snapshot(snap) {
                Ok(ok) => ok,
                Err(e) => {
                    tracing::debug!(error = %e, "prefix cache: restore failed");
                    false
                }
            },
            Self::Npu(_) => false,
        }
    }

    /// Take a recurrent model's prefix checkpoint at `pos` (see
    /// [`QuantizedModel::checkpoint_prefix`]).
    fn checkpoint_prefix(&mut self, pos: usize) {
        if let Self::Candle(model) = self {
            model.checkpoint_prefix(pos);
        }
    }

    /// Whether every layer's state can be cut back to an arbitrary prefix
    /// (an attention-only model), so a snapshot can be taken behind the
    /// current position without a checkpoint there.
    fn supports_truncate(&self) -> bool {
        match self {
            Self::Candle(model) => model.supports_kv_truncate(),
            Self::Npu(_) => false,
        }
    }

    /// Release the pending verification checkpoint once the decode loop is done
    /// with the step, so a pooled session does not hold a full copy of every
    /// recurrent layer's state (see [`crate::native_session::Session::discard_verify`]).
    fn discard_verify(&mut self) {
        match self {
            Self::Candle(model) => model.discard_verify(),
            Self::Npu(_) => {}
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
    /// The configured stop sequence that ended generation, if any.
    stop_sequence: Option<String>,
}

/// A completed generation, including which stop sequence ended it.
///
/// The tuple-returning completion helpers on [`Engine`] are thin wrappers
/// around this type; callers that need to report Anthropic's
/// `stop_reason`/`stop_sequence` fields use the `*_detailed` methods.
#[derive(Debug, Clone)]
pub struct Completion {
    /// Generated text, with any matched stop sequence removed.
    pub text: String,
    /// Token accounting for the request.
    pub usage: UsageInfo,
    /// Prefill throughput in tokens per second.
    pub prefill_tps: f64,
    /// Decode throughput in tokens per second.
    pub decode_tps: f64,
    /// The configured stop sequence that ended generation, if any.
    pub stop_sequence: Option<String>,
}

impl Completion {
    /// Split into the historical `(text, usage, prefill_tps, decode_tps)`
    /// tuple shape.
    fn into_tuple(self) -> (String, UsageInfo, f64, f64) {
        (
            self.text,
            self.usage,
            self.prefill_tps,
            self.decode_tps,
        )
    }
}

/// Repetition-penalty window: the last this-many tokens are penalised.
const REP_WINDOW: usize = 64;

/// A request's use of the separate draft model (see
/// [`crate::draft_model`]): the session, the decode context it drafts from,
/// and the penalty window extended with its drafts.
struct DraftModelState {
    session: crate::draft_model::DraftSession,
    /// The prompt followed by every emitted token.
    context: Vec<u32>,
    sampler: SamplingWorkspace,
    /// The penalty window extended with the tokens drafted so far.
    window: Vec<u32>,
}

impl DraftModelState {
    fn new(session: crate::draft_model::DraftSession, context: &[u32]) -> Self {
        Self {
            session,
            context: context.to_vec(),
            sampler: SamplingWorkspace::default(),
            window: Vec::with_capacity(REP_WINDOW),
        }
    }
}

/// Concrete stream decoder for the engine's tokenizer.
type TextDecodeStream<'a> = tokenizers::tokenizer::DecodeStream<
    'a,
    tokenizers::models::ModelWrapper,
    tokenizers::normalizers::NormalizerWrapper,
    tokenizers::pre_tokenizers::PreTokenizerWrapper,
    tokenizers::processors::PostProcessorWrapper,
    tokenizers::decoders::DecoderWrapper,
>;

/// Output text and repetition history for one decode loop.
struct DecodeText<'a> {
    response: String,
    /// Generated ids, for byte-level whole-buffer fallback decoding.
    decoded_ids: Vec<u32>,
    /// Incremental byte-level decode state (see [`ByteWindowDecoder`]).
    byte_window: ByteWindowDecoder,
    /// Decoder state for ByteFallback, Metaspace and other explicit decoders.
    stream: TextDecodeStream<'a>,
    /// Sliding repetition-penalty window.
    recent_tokens: Vec<u32>,
    n_decoded: u32,
}

impl<'a> DecodeText<'a> {
    /// Fresh state whose penalty window is seeded with the tail of `seed`.
    fn new(seed: &[u32], tokenizer: &'a tokenizers::Tokenizer) -> Self {
        Self {
            response: String::new(),
            decoded_ids: Vec::new(),
            byte_window: ByteWindowDecoder::default(),
            stream: tokenizer.decode_stream(false),
            recent_tokens: seed[seed.len().saturating_sub(REP_WINDOW)..].to_vec(),
            n_decoded: 0,
        }
    }
}

/// Reports decoded text to a streaming caller without leaking a stop
/// sequence.  A tail as long as the longest stop sequence is withheld until
/// the sequence either completes (and is trimmed from `response`) or
/// decoding ends.
struct StreamGate {
    sent: usize,
    holdback: usize,
}

impl StreamGate {
    fn new(stop_sequences: &[String]) -> Self {
        Self {
            sent: 0,
            holdback: stop_sequences
                .iter()
                .map(String::len)
                .max()
                .unwrap_or(0),
        }
    }

    fn push(
        &mut self,
        response: &str,
        stopped: bool,
        on_text: Option<&mut (dyn FnMut(&str) + '_)>,
    ) {
        let end = if stopped {
            response.len()
        } else {
            floor_char_boundary(response, response.len().saturating_sub(self.holdback))
        };
        self.emit_to(response, end, on_text);
    }

    fn finish(
        &mut self,
        response: &str,
        on_text: Option<&mut (dyn FnMut(&str) + '_)>,
    ) {
        self.emit_to(response, response.len(), on_text);
    }

    fn emit_to(
        &mut self,
        response: &str,
        end: usize,
        on_text: Option<&mut (dyn FnMut(&str) + '_)>,
    ) {
        let end = end.min(response.len());
        if end <= self.sent {
            // A stop sequence can trim the response below text already sent
            // only if it is longer than the holdback, which `new` prevents.
            return;
        }
        if let Some(on_text) = on_text {
            on_text(&response[self.sent..end]);
        }
        self.sent = end;
    }
}

/// Largest char boundary at or below `index` (the standard-library
/// `floor_char_boundary` is still unstable).
fn floor_char_boundary(s: &str, mut index: usize) -> usize {
    while index > 0 && !s.is_char_boundary(index) {
        index -= 1;
    }
    index
}

/// NPU backend state: the backend plus its circuit breaker.
struct NpuState {
    backend: Arc<dyn NpuBackend>,
    failures: AtomicU32,
    disabled: AtomicBool,
    /// The only place this engine runs generations (a remote pipeline):
    /// failures surface to the caller and never fall back to loading the
    /// model locally.
    exclusive: bool,
}

impl NpuState {
    /// Record a failure; disable the backend once the limit is reached.
    fn record_failure(&self, backend_name: &str, error: &str) {
        if self.exclusive {
            tracing::warn!("{backend_name} failure: {error}");
            return;
        }
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
        self.exclusive || !self.disabled.load(Ordering::Relaxed)
    }
}

/// Whether the tokenizer at `path` uses a byte-level decoder.
///
/// Byte-level BPE tokenizers (DeepSeek-V2/V3, Kimi-K2, Qwen, ...) map raw
/// bytes through their own table and can split a multi-byte UTF-8 sequence
/// across two tokens, so generated text must be decoded from the whole
/// accumulated token buffer rather than one token at a time.  Detected from
/// the `decoder.type` in `tokenizer.json`; everything else (word-level,
/// decoder-less, BPE with explicit decoders) uses the fallback paths in `emit_token`.
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
        let response = if new_text.starts_with(&self.tail_text) {
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
        let expert_cache_auto = options.expert_cache_auto;
        let prefill_chunk = match options.prefill_chunk {
            0 => DEFAULT_PREFILL_CHUNK,
            n => n,
        };
        let prefill_budget = Duration::from_millis(options.prefill_budget_ms);
        let prefill_cost = Mutex::new(crate::prefill_cost::PrefillCostModel::new(
            crate::prefill_cost::MIN_CHUNK,
            prefill_chunk,
        ));
        let prefix_checkpoints = options
            .prefix_checkpoints
            .unwrap_or(crate::native_session::DEFAULT_PREFIX_CHECKPOINTS);
        let prefix_cache_bytes = options
            .prefix_cache_bytes
            .unwrap_or_else(|| auto_prefix_cache_bytes(crate::placement::available_ram_bytes()));
        let prefix_cache_root_min = options
            .prefix_cache_root_min
            .unwrap_or(DEFAULT_PREFIX_CACHE_ROOT_MIN)
            .max(1);
        // A model loaded as larger than RAM faults its routed experts in on
        // demand; only then is a speculative expert prefetch worth its advice
        // calls (on a resident model every hint would be a no-op syscall).
        let predict_experts = options.pin_hot_weights || options.lazy_weights;
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

        // The raw header keeps every tensor's dtype id, so tensors in dtypes
        // candle cannot name (IQ2_XXS, I32, MXFP4) are still classified into
        // the dense / routed-expert split that hot-weight pinning and the
        // expert placement below both need; the projected `Content` above
        // drops them.
        let raw = crate::gguf_ext::read_header(&mut Cursor::new(&mmap[..]))
            .map_err(|e| JoshuaError::ModelLoad(format!("GGUF header re-read failed: {e}")))?;
        // Check the layout against the file's own length, not the mapping's:
        // the mapping is rounded up to a page, and a tensor cut short inside
        // that last page would otherwise read its missing bytes as zeros.
        let file_len = std::fs::metadata(&gguf_path)
            .map_err(|e| JoshuaError::ModelLoad(format!("stat of GGUF file failed: {e}")))?
            .len();
        raw.validate_layout(file_len)
            .map_err(|e| JoshuaError::ModelLoad(format!("GGUF layout check failed: {e}")))?;

        // Prefetch and/or lock the always-touched weights, and advise random
        // access on routed experts, when requested.
        if hot_pinning {
            if matches!(options.huge_pages, HugePages::Explicit(_)) {
                tracing::warn!(
                    "ignoring hot-weight pinning and whole-model prefetch: explicit huge pages \
                     already copy the whole model into anonymous RAM, so every range is \
                     resident. Use --huge-pages transparent (or none) for either to matter."
                );
            } else {
                // A whole-file prefetch already covers the dense ranges, so the
                // dense WILLNEED is skipped; the expert RANDOM advice (and any
                // mlock) still apply.
                apply_hot_weight_pinning(
                    &mmap,
                    &raw,
                    options.pin_hot_weights && !options.prefetch_whole_model,
                    options.mlock_hot_weights,
                    options.mlock_weight_budget_bytes,
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

        // Speculative decoding is per-architecture: it needs a verification
        // pass (all-position logits) plus a KV rollback past the rejected
        // draft tokens.  The loaded instance is re-checked at decode time;
        // this static check is what keeps the flag from being silently
        // ignored — an unsupported request warns here, decoding proceeds
        // plainly, and the effective configuration (`speculative_config`)
        // reads back `None`.
        let speculative = match (options.speculative, arch) {
            (Some(config), Some(arch)) => match arch.unsupported_speculative_reason() {
                Some(reason) => {
                    tracing::warn!(
                        "--speculative requested, but speculative decoding is not available on \
                         {}: {reason}; the flag is ignored for this model",
                        arch.display_name(),
                    );
                    None
                }
                None => Some(SpeculativeConfig::normalized(config)),
            },
            (Some(_), None) => {
                tracing::warn!(
                    "--speculative requested, but no candle loader was detected for this model \
                     (unsupported architecture or NPU-only); the flag is ignored"
                );
                None
            }
            (None, _) => None,
        };

        let eos_token_ids = extract_eos_ids(&gguf, &tokenizer);
        let special_tokens = crate::template::special_token_strings(&tokenizer);
        let chat_template = extract_chat_template(&gguf, &tokenizer);
        let target_vocab = crate::draft_model::gguf_tokens(&gguf.metadata);
        let device = Self::resolve_device(options.backend)?;
        #[cfg(feature = "sycl")]
        let qwen_b50_shape = crate::auto_placement::validated_b50_qwen_shape(&raw, n_ctx as usize, prefill_chunk);
        #[cfg(feature = "sycl")]
        let profile_budget_fits = options.device_memory_budget.is_none_or(|budget| {
            crate::placement::dense_set_fits(device_weight_bytes(&raw, arch).dense_lower,
                crate::placement::DEVICE_PLACEMENT_HEADROOM, budget)
        });
        #[cfg(feature = "sycl")]
        let device = if qwen_b50_shape && matches!(options.backend, ComputeBackend::Auto)
            && profile_budget_fits
            && options.dense_placement != crate::placement::DensePlacement::Cpu
            && (options.dense_placement != crate::placement::DensePlacement::Auto
                || crate::auto_placement::probe_enabled()) {
            Self::prefer_validated_qwen_sycl(device)
        } else { device };
        #[cfg(feature = "sycl")]
        let validated_sycl_qwen = qwen_b50_shape && Self::validated_b50_sycl(&device);
        #[cfg(not(feature = "sycl"))]
        let validated_sycl_qwen = false;

        // Memory placement on an accelerator: where the routed experts live,
        // and how many full weight copies the device can hold for the
        // architectures whose sessions cannot share one.
        let device_budget = options
            .device_memory_budget
            .or_else(|| crate::placement::device_memory_info(&device).map(|(free, _)| free));
        // Device footprint of the dense and routed-expert sets, as the
        // loaders leave them resident on the *resolved* device (see
        // `device_weight_bytes`).  The dense set comes as two bounds: the
        // lower one assumes every tensor whose residency is not known for
        // certain stays quantized, the upper one assumes it is dequantized
        // to f32.  A hard refusal to load needs certainty, so it uses the
        // lower bound; the soft decisions (expert placement, per-session
        // caps) err on the safe side with the upper bound.
        let footprint = device_weight_bytes(&raw, arch);
        // Whether the dense set can live on the device at all.  It is the
        // same test the hard refusal below applies, so an `Auto` placement
        // can never pick a device that would then reject the model: expert
        // placement cannot shrink the dense set, and `--expert-placement
        // cpu` is the documented remedy for a model too large for the card
        // (#176).  Without a budget or a candle loader nothing is known, and
        // the gate stays open.
        let dense_fits = match (device_budget, arch) {
            (Some(budget), Some(_)) => crate::placement::dense_set_fits(
                footprint.dense_lower,
                crate::placement::DEVICE_PLACEMENT_HEADROOM,
                budget,
            ),
            _ => true,
        };
        let dense_device_bytes = footprint.dense_upper;
        let expert_device_bytes = footprint.experts;
        let expert_bytes = expert_device_bytes;
        // Architectures whose loader keeps the experts on the CPU no matter
        // what (deepseek4 off OpenCL: IQ2_XXS has a kernel only there) are
        // resolved as host placement so the device accounting below matches
        // what the loader actually uploads.
        let experts_host_only = arch.is_some_and(|a| a.experts_always_on_host_for(&device));
        let requested_placement = if experts_host_only {
            if !device.is_cpu() && options.expert_placement == ExpertPlacement::Device {
                tracing::info!(
                    "expert placement `device` requested, but this architecture always keeps its \
                     routed experts in host RAM; only the dense set goes to the device"
                );
            }
            ExpertPlacement::Host
        } else {
            options.expert_placement
        };
        let placement = crate::placement::resolve_expert_placement(
            requested_placement,
            crate::placement::DeviceProfile {
                is_cpu: device.is_cpu(),
                dense_only: device.is_opencl() || device.is_vulkan(),
                free_bytes: device_budget,
            },
            dense_device_bytes,
            expert_device_bytes,
            crate::placement::DEVICE_PLACEMENT_HEADROOM,
        );
        let mut expert_device = match placement {
            crate::placement::ResolvedPlacement::Host => Device::Cpu,
            crate::placement::ResolvedPlacement::Device => device.clone(),
        };
        // Where the dense set runs.  A real accelerator (fast GEMM) wants dense
        // on-device; a weak iGPU / software OpenCL-Vulkan path is far slower
        // than CPU-BLAS and wants dense on the CPU.  `Auto` (the default)
        // runs a quick probe and picks by measured throughput; an explicit
        // `Device` / `Cpu` request is honoured as-is.  The same probe gates
        // the auto-engaged VRAM expert cache (#110) on the device's decode
        // throughput, so the bench runs once when either decision wants it.
        let cache_capable = arch.is_some_and(|a| a.device_experts_are_a_cache(&device));
        let cache_candidate = !device.is_cpu()
            && matches!(
                options.device_expert_cache,
                crate::placement::VramExpertCache::Unset
            )
            && requested_placement == ExpertPlacement::Auto
            && placement == crate::placement::ResolvedPlacement::Host
            && cache_capable;
        let validated_dense = validated_sycl_qwen && dense_fits
            && options.dense_placement == crate::placement::DensePlacement::Auto
            && crate::auto_placement::probe_enabled();
        let dense_probe = !validated_dense && !device.is_cpu()
            && options.dense_placement == crate::placement::DensePlacement::Auto
            && crate::auto_placement::probe_enabled();
        let bench = if dense_probe || (cache_candidate && crate::auto_placement::probe_enabled())
        {
            let bench = crate::auto_placement::benchmark(&device);
            tracing::info!(
                "dense placement probe: device decode {:.1} GFLOPS / prefill {:.1}; \
                 cpu decode {:.1} GFLOPS / prefill {:.1}; decode speedup {:.2}x / prefill {:.2}x",
                bench.dev_decode_gflops,
                bench.dev_prefill_gflops,
                bench.cpu_decode_gflops,
                bench.cpu_prefill_gflops,
                bench.decode_speedup(),
                bench.prefill_speedup(),
            );
            Some(bench)
        } else {
            None
        };
        let resolved_dense = if validated_dense {
            tracing::info!("dense placement: validated B50 Qwen3-8B Q4_K_M SYCL profile");
            crate::placement::ResolvedDense::Device
        } else if dense_probe {
            crate::auto_placement::recommend_dense(
                options.dense_placement,
                bench.as_ref().expect("the dense probe ran the bench"),
                false,
                dense_fits,
            )
        } else {
            crate::auto_placement::resolve_dense_without_probe(
                options.dense_placement,
                device.is_cpu(),
                dense_fits,
            )
        };
        let mut dense_device = match resolved_dense {
            crate::placement::ResolvedDense::Device => device.clone(),
            crate::placement::ResolvedDense::Cpu => Device::Cpu,
        };
        // Dense-only loaders still validate the expert-device argument. Keep
        // it with dense when there is no routed pool to place separately.
        if expert_bytes == 0 {
            expert_device = dense_device.clone();
        }
        if !device.is_cpu() && dense_device.is_cpu() {
            if !dense_fits && options.dense_placement == crate::placement::DensePlacement::Auto {
                tracing::info!(
                    "dense placement: cpu — the dense set (at least {:.2} GiB plus {:.1} GiB \
                     headroom) does not fit the device memory budget of {:.2} GiB, so the model \
                     cannot be held on the {:?} accelerator; expert placement does not change \
                     this. Running with the dense set on CPU-BLAS.",
                    footprint.dense_lower as f64 / 2f64.powi(30),
                    crate::placement::DEVICE_PLACEMENT_HEADROOM as f64 / 2f64.powi(30),
                    device_budget.unwrap_or(0) as f64 / 2f64.powi(30),
                    device,
                );
            } else {
                tracing::info!(
                    "dense placement: cpu — keeping the dense set on CPU-BLAS (requested {:?}); \
                     the {:?} accelerator is reserved but unused for weights",
                    options.dense_placement,
                    device,
                );
            }
        } else {
            tracing::info!("dense placement: device ({device:?})",);
        }
        // A dense set placed on the device cannot be shrunk by any expert
        // placement, so a budget it does not fit is refused here, with the
        // numbers, rather than deferred to an out-of-memory upload on the
        // first request.  Only for architectures the candle path loads: a
        // model candle cannot load (`arch == None`) allocates nothing on the
        // device and may still be served by an NPU backend attached after
        // construction, so its deferred `arch_error` path is left alone.
        if !dense_device.is_cpu() && arch.is_some() {
            if let Some(budget) = device_budget {
                if !crate::placement::dense_set_fits(
                    footprint.dense_lower,
                    crate::placement::DEVICE_PLACEMENT_HEADROOM,
                    budget,
                ) {
                    return Err(JoshuaError::ModelLoad(format!(
                        "the model's dense set (at least {:.2} GiB on {:?}, plus {:.1} GiB headroom) does not fit \
                         the device memory budget of {:.2} GiB ({}). Expert placement can only move \
                         the routed experts off the device; run with --device cpu or --dense-placement cpu, \
                         free device memory, or raise --vram-budget if the budget was set by hand.",
                        footprint.dense_lower as f64 / 2f64.powi(30),
                        device,
                        crate::placement::DEVICE_PLACEMENT_HEADROOM as f64 / 2f64.powi(30),
                        budget as f64 / 2f64.powi(30),
                        if options.device_memory_budget.is_some() {
                            "--vram-budget"
                        } else {
                            "probed free memory"
                        },
                    )));
                }
            }
        }
        // Resolve the VRAM expert cache (#62, #110) to the concrete byte
        // budget the loaders get.  An explicit setting wins: `Auto`
        // (`--vram-expert-cache auto`, the `Some(0)` sentinel) sizes from the
        // device's memory budget below, a fixed MiB budget passes through,
        // and `0` is the opt-out.  With no setting, an `--expert-placement
        // device` request on an architecture whose device expert form *is*
        // this cache (deepseek4 on OpenCL/SYCL) implies `auto`, so the flag
        // alone puts experts on the card; and an `auto` placement that fell
        // back to the host engages the cache over the host pool when the
        // leftover device memory is worth a cache and the device is not
        // slower than CPU-BLAS on decode.
        //
        // The leftover is the same terms the `--vram-expert-cache auto`
        // sizing below applies: device free − dense (only when the dense set
        // is on the device) − placement headroom − scratch − KV reserve (only
        // with the dense set on the device).
        let (mut dense_term, mut kv_reserve, mut scratch) = if dense_device.is_cpu() {
            (0, 0, DEVICE_SCRATCH_RESERVE_DENSE_ON_CPU)
        } else {
            (
                footprint.dense_upper,
                DEVICE_KV_RESERVE_BYTES,
                DEVICE_SCRATCH_RESERVE_DENSE_ON_DEVICE,
            )
        };
        let mut vram_cache_leftover = device_budget.map(|free| {
            crate::placement::device_expert_cache_bytes(
                free,
                dense_term,
                crate::placement::DEVICE_PLACEMENT_HEADROOM + scratch,
                kv_reserve,
            )
        });
        let expert_cache_implied = cache_capable
            && placement == crate::placement::ResolvedPlacement::Device
            && matches!(
                options.device_expert_cache,
                crate::placement::VramExpertCache::Unset
            );
        // #114: the auto engagement may move the dense set to CPU-BLAS to
        // grow the cache over the residency floor (see
        // `dense_to_cpu_for_cache`); applied to `dense_device` below.
        let mut dense_for_cache = false;
        let mut device_expert_cache: Option<u64> = match options.device_expert_cache {
            crate::placement::VramExpertCache::Disabled => {
                tracing::info!("vram expert cache: disabled (--vram-expert-cache 0)");
                None
            }
            crate::placement::VramExpertCache::Bytes(0) => {
                tracing::info!("vram expert cache: disabled (a zero byte budget)");
                None
            }
            crate::placement::VramExpertCache::Bytes(n) => Some(n),
            crate::placement::VramExpertCache::Auto => Some(0),
            crate::placement::VramExpertCache::Unset if expert_cache_implied => {
                tracing::info!(
                    "expert placement `device` on {device:?}: the routed experts run from a bounded VRAM cache; sizing it as --vram-expert-cache auto"
                );
                Some(0)
            }
            crate::placement::VramExpertCache::Unset => {
                // The probe figure is the dense set's Q4_K GEMM — a proxy for
                // the expert kernels (the cached deepseek4 experts run
                // IQ2_XXS gate/up and Q2_K down on the device), so this gate
                // is a coarse filter; confirm on hardware.  See
                // `auto_vram_expert_cache`.
                match crate::placement::auto_vram_expert_cache(
                    requested_placement,
                    placement,
                    // V4.1's CPU software cache changes the mixed expert path
                    // without accelerating it. Auto requires an actual device
                    // consumer; explicit cache requests remain explicit.
                    cache_capable && (arch != Some(Architecture::DeepSeek41)
                        || !dense_device.is_cpu() || !expert_device.is_cpu()),
                    vram_cache_leftover,
                    expert_device_bytes,
                    bench.as_ref().map(|b| b.decode_speedup()),
                ) {
                    Ok(()) => {
                        tracing::info!(
                            "expert placement `auto` fell back to the host with device memory to spare on {device:?}: engaging a bounded VRAM expert cache over the host pool, sized as --vram-expert-cache auto (#110)"
                        );
                        Some(0)
                    }
                    Err(decline) => {
                        // #114: when the only decline is residency and the
                        // dense set on the card is what squeezes the
                        // leftover, moving dense to CPU-BLAS can grow the
                        // cache over the floor — worth it when the device's
                        // dense GEMM is not decisively faster (the probe).
                        if let (Some(leftover), Some(bench)) = (vram_cache_leftover, bench.as_ref())
                        {
                            if let Some(freed) = crate::placement::dense_to_cpu_for_cache(
                                &decline,
                                // Moving dense must leave an accelerator for
                                // the V4.1 cache, not silently build it on CPU.
                                cache_capable && (arch != Some(Architecture::DeepSeek41)
                                    || !expert_device.is_cpu()),
                                dense_term,
                                leftover,
                                expert_device_bytes,
                                Some(bench.decode_speedup()),
                            ) {
                                dense_for_cache = true;
                                tracing::info!(
                                    "vram expert cache: moving the dense set to CPU-BLAS frees \
                                     {:.2} GiB on {device:?} and grows the auto cache over the \
                                     residency floor (#114); the probe put the device's dense \
                                     decode at {:.2}x CPU-BLAS, so the expert residency is the \
                                     better use of the card",
                                    dense_term as f64 / 2f64.powi(30),
                                    bench.decode_speedup(),
                                );
                                Some(freed)
                            } else {
                                if decline.applicable() {
                                    tracing::info!(
                                        "vram expert cache not engaged: {}; --vram-expert-cache auto forces it",
                                        decline.describe()
                                    );
                                }
                                None
                            }
                        } else {
                            if decline.applicable() {
                                tracing::info!(
                                    "vram expert cache not engaged: {}; --vram-expert-cache auto forces it",
                                    decline.describe()
                                );
                            }
                            None
                        }
                    }
                }
            }
        };
        if dense_for_cache {
            // The engagement moved dense to CPU-BLAS (#114): recompute the
            // leftover with the dense set freed and re-derive the resolved
            // placement, then size the cache from the new leftover below.
            dense_device = Device::Cpu;
            dense_term = 0;
            kv_reserve = 0;
            scratch = DEVICE_SCRATCH_RESERVE_DENSE_ON_CPU;
            if let Some(free) = device_budget {
                vram_cache_leftover = Some(crate::placement::device_expert_cache_bytes(
                    free,
                    0,
                    crate::placement::DEVICE_PLACEMENT_HEADROOM + scratch,
                    0,
                ));
            }
            expert_device = Device::Cpu;
        }
        if device_expert_cache == Some(0) {
            // `--vram-expert-cache auto` (an explicit request, implied by an
            // expert-placement `device` on a cache-capable architecture, or
            // engaged by #110): size from the leftover device memory computed
            // above.
            match vram_cache_leftover {
                Some(budget) => {
                    tracing::info!(
                        "vram-expert-cache auto: {:.2} GiB on {:?} ({:.2} GiB device memory − {:.2} GiB dense − {:.2} GiB headroom+scratch − {:.2} GiB KV reserve)",
                        budget as f64 / 2f64.powi(30),
                        device,
                        device_budget.expect("a leftover implies a budget") as f64 / 2f64.powi(30),
                        dense_term as f64 / 2f64.powi(30),
                        (crate::placement::DEVICE_PLACEMENT_HEADROOM + scratch) as f64 / 2f64.powi(30),
                        kv_reserve as f64 / 2f64.powi(30),
                    );
                    device_expert_cache = Some(budget);
                }
                None => {
                    tracing::info!(
                        "vram-expert-cache auto: no device memory probe and no --vram-budget; disabled"
                    );
                    device_expert_cache = None;
                }
            }
        }
        let shares_weights = arch.is_some_and(|a| a.shares_weights());
        // Device memory one session's weights occupy (0 when sessions share
        // the template's weights and only add a KV cache).
        let instance_device_bytes = if device.is_cpu() || shares_weights {
            0
        } else if expert_device.is_cpu() {
            dense_device_bytes
        } else {
            dense_device_bytes.saturating_add(expert_device_bytes)
        };
        let device_session_cap = match (device_budget, instance_device_bytes) {
            (Some(free), bytes) if bytes > 0 => Some(crate::placement::instances_for_memory(
                free,
                bytes,
                crate::placement::DEVICE_PLACEMENT_HEADROOM,
                MAX_CACHED_MODELS,
            )),
            _ => None,
        };
        if !device.is_cpu() && expert_bytes > 0 {
            tracing::info!(
                "expert placement: {} (dense {:.1}–{:.1} GiB, experts {:.1} GiB, device budget {}, requested {:?}{})",
                match placement {
                    crate::placement::ResolvedPlacement::Host =>
                        "host RAM — experts borrowed from the mapping, dense set on the device",
                    crate::placement::ResolvedPlacement::Device => "device",
                },
                footprint.dense_lower as f64 / 2f64.powi(30),
                footprint.dense_upper as f64 / 2f64.powi(30),
                expert_device_bytes as f64 / 2f64.powi(30),
                match device_budget {
                    Some(b) => format!("{:.1} GiB", b as f64 / 2f64.powi(30)),
                    None => "unknown".to_string(),
                },
                requested_placement,
                match device_expert_cache {
                    Some(b) if b > 0 => format!("; VRAM expert cache {:.2} GiB", b as f64 / 2f64.powi(30)),
                    _ => String::new(),
                },
            );
        }

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
        // with `with_max_concurrency`.  On an accelerator where every session
        // carries its own copy of the weights, the device's memory is the
        // real bound: a CPU-count of concurrent uploads would exhaust it.
        let cpu_parallelism = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let max_concurrency = match (device_budget, instance_device_bytes) {
            (Some(free), bytes) if bytes > 0 => {
                let n = crate::placement::instances_for_memory(
                    free,
                    bytes,
                    crate::placement::DEVICE_PLACEMENT_HEADROOM,
                    cpu_parallelism,
                );
                if n < cpu_parallelism {
                    tracing::info!(
                        "device memory fits {n} model instance(s) of {:.1} GiB — capping \
                         concurrency at {n} (override with --max-concurrency)",
                        bytes as f64 / 2f64.powi(30)
                    );
                }
                n
            }
            _ => cpu_parallelism,
        };

        // A separate draft model.  It needs a target that can verify and
        // roll back; it replaces MTP drafting, so the NextN block is not
        // loaded alongside it.
        let mut mtp = options.mtp.min(crate::speculative::MAX_MTP_DRAFT);
        let draft = match (&options.draft_model, arch) {
            (Some(path), Some(a)) if a.supports_speculative_decoding() => {
                // Both models share the dense device, so their weights must
                // fit together — the target's dense set and any routed
                // experts or VRAM expert cache it keeps on the card; the
                // target's own check ran before the draft existed.  A pairing
                // that cannot fit falls back to plain decoding instead of
                // leaving the target unable to upload.
                let draft_bytes = crate::draft_model::resident_bytes(path).unwrap_or(0);
                // Native bounded-cache loaders keep their source experts on
                // the host, with the cache on the expert or dense device.
                let accel = |d: &Device| !d.is_cpu();
                let ds4 = a.is_deepseek4();
                let cache = if ds4 {
                    device_expert_cache.filter(|&b| b > 0)
                } else {
                    let slot = crate::moe::compact_expert_slot(raw.tensors.iter().map(|(name, info)| {
                        (name.as_str(), info.dims.as_slice(), info.dtype)
                    }));
                    device_expert_cache.filter(|&bytes| slot.is_some_and(|slot| slot > 0 && bytes >= slot))
                };
                let bounded = ds4
                    || (a.is_native()
                        && a != Architecture::NemotronH
                        && cache.is_some());
                let pool_on_device = !bounded && !expert_device.is_cpu();
                let cache_on_device = if bounded {
                    cache.filter(|_| accel(&expert_device) || accel(&dense_device))
                } else {
                    cache.filter(|_| !expert_device.is_cpu())
                };
                let target_bytes = crate::placement::target_device_weight_bytes(
                    footprint.dense_upper,
                    expert_device_bytes,
                    pool_on_device,
                    cache_on_device,
                );
                let fits = dense_device.is_cpu()
                    || device_budget.is_none_or(|budget| {
                        crate::placement::dense_set_fits(
                            target_bytes.saturating_add(draft_bytes),
                            crate::placement::DEVICE_PLACEMENT_HEADROOM,
                            budget,
                        )
                    });
                if !fits {
                    tracing::warn!(
                        "--draft-model {} ignored: the target's {:.2} GiB of device weights plus \
                         the draft's {:.2} GiB do not fit the device memory budget; decoding plainly",
                        path.display(),
                        target_bytes as f64 / 2f64.powi(30),
                        draft_bytes as f64 / 2f64.powi(30),
                    );
                    None
                } else {
                    let draft = crate::draft_model::DraftModel::load(
                        path,
                        &dense_device,
                        n_ctx as usize,
                        target_vocab.as_deref(),
                    )?;
                    if mtp > 0 {
                        tracing::warn!("--mtp is ignored while --draft-model drafts");
                        mtp = 0;
                    }
                    Some(draft)
                }
            }
            (Some(_), arch) => {
                tracing::warn!(
                    "--draft-model requested, but {} cannot verify drafts; the flag is ignored",
                    arch.map_or("this model", |a| a.display_name()),
                );
                None
            }
            (None, _) => None,
        };
        let draft_max = match options.draft_max {
            0 => crate::draft_model::DEFAULT_DRAFT_MODEL_MAX,
            n => n,
        };

        Ok(Self {
            model_path: gguf_path,
            mmap: Arc::new(mmap),
            model_file,
            tokenizer: Arc::new(tokenizer),
            tokenizer_path,
            constraints: std::sync::OnceLock::new(),
            byte_level_decode,
            eos_token_ids,
            chat_template,
            special_tokens,
            embed_model: Mutex::new(None),
            model_cache: Mutex::new(Vec::new()),
            kv_reuses: AtomicU64::new(0),
            kv_edit_reuses: AtomicU64::new(0),
            npu: None,
            in_flight: AdmissionGate::new(),
            max_concurrency,
            queue_timeout: Duration::from_millis(options.queue_timeout_ms),
            admission_queue_depth: if options.admission_queue_depth == 0 {
                DEFAULT_ADMISSION_QUEUE_DEPTH
            } else {
                options.admission_queue_depth
            },
            max_output_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
            model_name,
            n_ctx,
            pin_hot_experts,
            expert_cache_auto,
            device_expert_cache,
            ssd_expert_cache: options.ssd_expert_cache,
            engram_tier: options.engram_tier,
            prefill_chunk,
            prefill_budget,
            prefill_cost,
            prefix_checkpoints,
            prefix_cache: Mutex::new(crate::prefix_cache::PrefixCache::new(prefix_cache_bytes)),
            prefix_cache_root_min,
            tool_orders: Default::default(),
            env_to_user: options.env_to_user,
            predict_experts,
            device,
            dense_device,
            expert_device,
            weights_template: Mutex::new(None),
            device_session_cap,
            remote_pool_cap: None,
            remote_capacity: None,
            speculative,
            mtp,
            draft,
            draft_max,
            spec_drafted: AtomicU64::new(0),
            spec_accepted: AtomicU64::new(0),
            spec_verify_steps: AtomicU64::new(0),
            arch,
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
        self.fit_remote_capacity();
        self
    }

    /// Set how long a request may wait for a concurrency permit before it
    /// returns [`JoshuaError::Overloaded`].  `Duration::ZERO` (the default)
    /// rejects immediately.  See [`EngineOptions::queue_timeout_ms`].
    pub fn with_queue_timeout(mut self, timeout: Duration) -> Self {
        self.queue_timeout = timeout;
        self
    }

    /// Set how many requests may wait for a permit (`0` = the default).  See
    /// [`EngineOptions::admission_queue_depth`].
    pub fn with_admission_queue_depth(mut self, depth: usize) -> Self {
        self.admission_queue_depth = if depth == 0 {
            DEFAULT_ADMISSION_QUEUE_DEPTH
        } else {
            depth
        };
        self
    }

    /// Set the hard ceiling on tokens generated per request, applied on top
    /// of the client-supplied `max_tokens`.  Values below 1 are treated as 1.
    pub fn with_max_output_tokens(mut self, max: u32) -> Self {
        self.max_output_tokens = max.max(1);
        self
    }

    /// Baseline availability order for [`ComputeBackend::Auto`].
    ///
    /// Tries compiled backends in CUDA, Metal, OpenCL, Vulkan, SYCL order,
    /// then CPU. This selects an available backend; dense placement is
    /// measured separately after opening it. Qualified model preferences
    /// are applied by the constructor within their benchmarked bounds.
    fn default_device() -> Device {
        #[cfg(feature = "cuda")]
        {
            match Device::new_cuda_with_stream(0) {
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
        #[cfg(feature = "opencl")]
        {
            match Device::new_opencl(0) {
                Ok(device) => {
                    Self::log_opencl_device(&device);
                    return device;
                }
                Err(e) => tracing::warn!("OpenCL unavailable, falling back to CPU: {e}"),
            }
        }
        #[cfg(feature = "vulkan")]
        {
            match Device::new_vulkan(0) {
                Ok(device) => {
                    Self::log_vulkan_device(&device);
                    return device;
                }
                Err(e) => tracing::warn!("Vulkan unavailable, falling back to CPU: {e}"),
            }
        }
        #[cfg(feature = "sycl")]
        {
            match Device::new_sycl(0) {
                Ok(device) => {
                    Self::log_sycl_device(&device);
                    return device;
                }
                Err(e) => tracing::warn!("SYCL unavailable, falling back to CPU: {e}"),
            }
        }
        Device::Cpu
    }

    #[cfg(feature = "sycl")]
    fn validated_b50_sycl(device: &Device) -> bool {
        device.as_sycl_device().is_ok_and(|d| d.ctx().canonical_quantized)
            && candle_core::sycl_backend::kernels::native_enabled()
    }

    #[cfg(feature = "sycl")]
    fn prefer_validated_qwen_sycl(current: Device) -> Device {
        // Only replace a backend already targeting B50. CUDA, Metal, CPU,
        // other adapters, and explicit backend requests keep their policy.
        let current_b50 = match &current {
            #[cfg(feature = "vulkan")]
            Device::Vulkan(d) => d.name().to_ascii_lowercase().contains("pro b50"),
            #[cfg(feature = "opencl")]
            Device::OpenCl(d) => d.name().to_ascii_lowercase().contains("pro b50"),
            _ => false,
        };
        if current_b50 {
            if let Ok(device) = Device::new_sycl(0) {
                if Self::validated_b50_sycl(&device) {
                    Self::log_sycl_device(&device);
                    return device;
                }
            }
        }
        current
    }

    /// Log which OpenCL device was opened and how it holds weights, so a
    /// slow run can be read off the log: a device without host-unified
    /// memory (a discrete card) copies the weights, one with it (an iGPU or
    /// a CPU runtime) aliases the memory-mapped file in place.
    #[cfg(feature = "opencl")]
    fn log_opencl_device(device: &Device) {
        if let Ok(ocl) = device.as_opencl_device() {
            let unified = ocl.host_unified_memory();
            tracing::info!(
                "OpenCL device: {} ({} MiB global memory, {}; native kernels {}, zero-copy weights {})",
                ocl.name(),
                ocl.global_mem_size().unwrap_or(0) >> 20,
                if unified { "host-unified memory" } else { "discrete memory" },
                if candle_core::opencl_backend::native_enabled() { "on" } else { "off" },
                if unified && candle_core::opencl_backend::zero_copy_enabled() { "on" } else { "off" },
            );
        }
    }

    /// Log which Vulkan device was opened and how it holds weights.
    #[cfg(feature = "vulkan")]
    fn log_vulkan_device(device: &Device) {
        if let Ok(vk) = device.as_vulkan_device() {
            let lim = vk.limits();
            tracing::info!(
                "Vulkan device: {} ({}; native kernels {}; quantized GEMV {:?}, int8 dot product {})",
                vk.name(),
                if vk.host_unified_memory() {
                    "host-unified memory"
                } else {
                    "discrete memory, host-visible staging"
                },
                if candle_core::vulkan_backend::native_enabled() {
                    "on"
                } else {
                    "off"
                },
                lim.qgemv,
                if lim.int_dot { "on" } else { "off" },
            );
        }
    }

    /// Log which SYCL device was opened and whether its native kernels are on.
    #[cfg(feature = "sycl")]
    fn log_sycl_device(device: &Device) {
        if let Ok(sycl) = device.as_sycl_device() {
            tracing::info!(
                "SYCL device: {} ({}; native kernels {})",
                sycl.name(),
                if sycl.host_unified_memory() {
                    "host-unified memory"
                } else {
                    "discrete memory"
                },
                if candle_core::sycl_backend::kernels::native_enabled() {
                    "on"
                } else {
                    "off"
                },
            );
        }
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
                    Device::new_cuda_with_stream(0).map_err(|e| {
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
            ComputeBackend::OpenCl => {
                #[cfg(feature = "opencl")]
                {
                    let device = Device::new_opencl(0).map_err(|e| {
                        JoshuaError::ModelLoad(format!(
                            "OpenCL device requested but unavailable: {e}. \
                             Build with `--features opencl` on a host with libOpenCL \
                             (Intel iGPU/Arc, NVIDIA ICD, ...), or pass --device cpu / auto."
                        ))
                    })?;
                    Self::log_opencl_device(&device);
                    Ok(device)
                }
                #[cfg(not(feature = "opencl"))]
                {
                    Err(JoshuaError::ModelLoad(
                        "OpenCL device requested but this build has no `opencl` feature. \
                         Rebuild with `cargo build --features opencl`, or pass --device cpu / auto."
                            .to_string(),
                    ))
                }
            }
            ComputeBackend::Vulkan => {
                #[cfg(feature = "vulkan")]
                {
                    let device = Device::new_vulkan(0).map_err(|e| {
                        JoshuaError::ModelLoad(format!(
                            "Vulkan device requested but unavailable: {e}. \
                             Build with `--features vulkan` on a host with libvulkan.so \
                             (AMD RADV, Intel ANV, ...), or pass --device cpu / auto."
                        ))
                    })?;
                    Self::log_vulkan_device(&device);
                    Ok(device)
                }
                #[cfg(not(feature = "vulkan"))]
                {
                    Err(JoshuaError::ModelLoad(
                        "Vulkan device requested but this build has no `vulkan` feature. \
                         Rebuild with `cargo build --features vulkan`, or pass --device cpu / auto."
                            .to_string(),
                    ))
                }
            }
            ComputeBackend::Sycl => {
                #[cfg(feature = "sycl")]
                {
                    let device = Device::new_sycl(0).map_err(|e| {
                        JoshuaError::ModelLoad(format!(
                            "SYCL device requested but unavailable: {e}. \
                             Build `--features sycl` and point JOSHUA_SYCL_LIBRARY at the built \
                             libjoshua_sycl (SYCL 2020 oneAPI/LLVM bridge), or pass --device cpu / auto."
                        ))
                    })?;
                    Self::log_sycl_device(&device);
                    Ok(device)
                }
                #[cfg(not(feature = "sycl"))]
                {
                    Err(JoshuaError::ModelLoad(
                        "SYCL device requested but this build has no `sycl` feature. \
                         Rebuild with `cargo build --features sycl`, or pass --device cpu / auto."
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

    /// Tokenize `text` with the model's tokenizer.  `add_special_tokens`
    /// controls the tokenizer's BOS/EOS wrapper; chat prompts formatted by
    /// the engine already contain their special tokens.
    pub fn tokenize(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text, add_special_tokens)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|e| JoshuaError::Tokenization(e.to_string()))
    }

    /// Detokenize `tokens` back to text — the inverse of
    /// [`Engine::tokenize`], and what a vLLM / llama.cpp client expects from
    /// `POST /v1/detokenize`.
    ///
    /// `skip_special_tokens` matches the convention of the same name on the
    /// tokenizer: `true` drops BOS/EOS/padding, `false` keeps whatever the
    /// ids decode to.  Ids outside the vocabulary are rejected by the
    /// tokenizer rather than silently skipped, so a caller that passed the
    /// wrong model gets an error instead of plausible-looking text.
    pub fn detokenize(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(tokens, skip_special_tokens)
            .map_err(|e| JoshuaError::Tokenization(e.to_string()))
    }

    /// The vocabulary size, for the tokenizer endpoints that report it.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Count the prompt tokens a chat request will use, without running the
    /// model.  Useful for API usage figures and admission checks; media
    /// messages count only their text portion.
    pub fn count_chat_tokens(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        options: &GenerationOptions,
    ) -> Result<u32> {
        let reordered = tools.and_then(|t| self.tool_orders.stable(t));
        let tools = reordered.as_deref().or(tools);
        let moved = if self.env_to_user {
            crate::prompt_stability::env_to_user(messages)
        } else {
            None
        };
        let messages = moved.as_deref().unwrap_or(messages);
        let (prompt, add_special_tokens) =
            self.format_prompt(messages, tools, &options.chat_template_kwargs);
        self.tokenize(&prompt, add_special_tokens)
            .map(|tokens| tokens.len() as u32)
    }

    /// Device the routed-expert pool of a MoE model lives on: the compute
    /// device, or the CPU when the experts stay in host RAM (see
    /// [`EngineOptions::expert_placement`]).
    pub fn expert_device(&self) -> &Device {
        &self.expert_device
    }

    /// Device holding the dense projections after automatic placement.
    pub fn dense_device(&self) -> &Device {
        &self.dense_device
    }

    /// The resolved bounded VRAM expert-cache byte budget the loaders get:
    /// what [`EngineOptions::device_expert_cache`] asked for, after the
    /// `auto` sizing and the #110 engagement decision.  `None` disables.
    pub fn device_expert_cache(&self) -> Option<u64> {
        self.device_expert_cache
    }

    /// Number of sessions currently sharing the loaded weights of a
    /// weight-sharing architecture (the template counts as one), or 0 when
    /// no such template has been loaded.  Diagnostics.
    pub fn shared_weight_sessions(&self) -> usize {
        self.weights_template
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(QuantizedModel::shared_session_count)
            .unwrap_or(0)
    }

    /// The compute device inference runs on (CPU, or Metal/CUDA when the
    /// matching feature is enabled and a device is available).
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Generations/embeddings executing right now (admitted and holding a
    /// concurrency permit).  Reported to request routers as live load.
    pub fn in_flight(&self) -> usize {
        self.in_flight.active()
    }

    /// The admission cap: requests beyond this many in flight are rejected
    /// with [`JoshuaError::Overloaded`] (see [`Engine::with_max_concurrency`]).
    pub fn max_concurrency(&self) -> usize {
        self.max_concurrency
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
    ///
    /// `vars` are extra template variables
    /// ([`GenerationOptions::chat_template_kwargs`]); the ChatML fallback has
    /// no switches and ignores them.
    fn format_prompt(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        vars: &serde_json::Map<String, serde_json::Value>,
    ) -> (String, bool) {
        if let Some(template) = &self.chat_template {
            match template.render_with_vars(messages, tools, vars) {
                Ok(prompt) => return (prompt, false),
                Err(e) => {
                    tracing::warn!("GGUF chat template unusable, falling back to ChatML: {e}");
                }
            }
        }
        let sanitized = crate::template::sanitize_messages(messages, &self.special_tokens);
        (Self::format_chatml_prompt(&sanitized, tools), true)
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
        self.complete_chat_detailed(messages, tools, options)
            .map(Completion::into_tuple)
    }

    /// [`Engine::complete_chat`] with the matched stop sequence included.
    pub fn complete_chat_detailed(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        options: &GenerationOptions,
    ) -> Result<Completion> {
        self.complete_chat_inner(messages, tools, options, None, None)
    }

    /// [`Engine::complete_chat`] that reports text as it is decoded.
    ///
    /// `on_text` receives newly decoded text, not raw token ids: byte-level
    /// tokenizers may hold back a partial UTF-8 character until the next
    /// token completes it.  When `stop_sequences` is non-empty the engine
    /// withholds a tail as long as the longest stop sequence, so a sequence
    /// that spans token boundaries is never sent to the caller; the tail is
    /// flushed when decoding ends normally.
    ///
    /// The callback runs on the decoding thread.  It should hand the text to
    /// another thread or channel rather than block, because every call sits
    /// on the per-token path.
    pub fn complete_chat_streaming(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        options: &GenerationOptions,
        cancel: &AtomicBool,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<(String, UsageInfo, f64, f64)> {
        self.complete_chat_streaming_detailed(messages, tools, options, cancel, on_text)
            .map(Completion::into_tuple)
    }

    /// [`Engine::complete_chat_streaming`] with the matched stop sequence.
    pub fn complete_chat_streaming_detailed(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        options: &GenerationOptions,
        cancel: &AtomicBool,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Completion> {
        self.complete_chat_inner(
            messages,
            tools,
            options,
            Some(cancel),
            Some(on_text),
        )
    }

    /// [`Engine::complete_chat`] that stops decoding once `cancel` is set.
    ///
    /// The flag is checked before every decoded token, so a caller whose
    /// client has gone away (the HTTP server sets it when the request is
    /// dropped) frees its permit and session after at most one more step
    /// instead of generating up to `max_tokens` for nobody.  A cancelled
    /// generation returns the text decoded so far; the pooled session is
    /// left exactly as after a `max_tokens` stop, so it stays reusable.
    pub fn complete_chat_cancellable(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        options: &GenerationOptions,
        cancel: &AtomicBool,
    ) -> Result<(String, UsageInfo, f64, f64)> {
        self.complete_chat_cancellable_detailed(messages, tools, options, cancel)
            .map(Completion::into_tuple)
    }

    /// [`Engine::complete_chat_cancellable`] with the matched stop sequence.
    pub fn complete_chat_cancellable_detailed(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        options: &GenerationOptions,
        cancel: &AtomicBool,
    ) -> Result<Completion> {
        self.complete_chat_inner(messages, tools, options, Some(cancel), None)
    }

    fn complete_chat_inner(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        options: &GenerationOptions,
        cancel: Option<&AtomicBool>,
        mut on_text: Option<&mut (dyn FnMut(&str) + '_)>,
    ) -> Result<Completion> {
        // Keep the prompt's head byte-identical across requests where the
        // client varies it for no semantic reason, so cached prompt state
        // still matches (see `crate::prompt_stability`).
        let reordered = tools.and_then(|t| self.tool_orders.stable(t));
        let tools = reordered.as_deref().or(tools);
        let moved = if self.env_to_user {
            crate::prompt_stability::env_to_user(messages)
        } else {
            None
        };
        let messages = moved.as_deref().unwrap_or(messages);
        // Multimodal branch: messages carrying images go through a
        // media-capable NPU/llama.cpp plugin session.
        if messages
            .iter()
            .any(|m| m.images.as_ref().is_some_and(|i| !i.is_empty()))
        {
            let (marked_messages, images) = resolve_message_media(messages)?;
            let (prompt, _) =
                self.format_prompt(&marked_messages, tools, &options.chat_template_kwargs);
            return self.complete_media(
                &prompt,
                &images,
                options,
                cancel,
                on_text.as_deref_mut(),
            );
        }
        let (prompt, add_special_tokens) =
            self.format_prompt(messages, tools, &options.chat_template_kwargs);
        let root = self.prompt_root_chars(messages, tools, options, &prompt);
        self.complete_with(
            &prompt,
            add_special_tokens,
            options,
            cancel,
            root,
            on_text.as_deref_mut(),
        )
    }

    /// Run a multimodal completion: the plugin tokenises and prefills the
    /// marked prompt together with the media, then decode proceeds normally.
    fn complete_media(
        &self,
        prompt: &str,
        images: &[Vec<u8>],
        options: &GenerationOptions,
        cancel: Option<&AtomicBool>,
        on_text: Option<&mut (dyn FnMut(&str) + '_)>,
    ) -> Result<Completion> {
        // Bound concurrent heavyweight generations before doing any work.
        let _permit = InFlightGuard::acquire_with(
            &self.in_flight,
            self.max_concurrency,
            self.admission_queue_depth,
            self.queue_timeout,
            cancel,
        )?;
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            // A generation cancelled before any work is an empty completion,
            // matching the decode loop's stop path (and the test contract).
            return Ok(Completion {
                text: String::new(),
                usage: UsageInfo::default(),
                prefill_tps: 0.0,
                decode_tps: 0.0,
                stop_sequence: None,
            });
        }
        let n_ctx = self.request_n_ctx(options)?;
        // The plugin owns tokenisation, so the prompt length is unknown here;
        // clamp only to the server ceiling. The prompt is checked against the
        // request's window once prefill reports its length, and the decode
        // loop's in-context guard bounds the total length.
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
                // Only now is the prompt's length (text plus media positions)
                // known: reject it as the text path does rather than
                // returning an empty completion from a full window.
                if n_past >= n_ctx {
                    return Err(JoshuaError::PromptTooLong(n_past, n_ctx));
                }
                let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
                let outcome = self.decode_loop(
                    &mut session,
                    logits,
                    n_past,
                    &[],
                    options,
                    cancel,
                    on_text,
                )?;
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
                Ok(Completion {
                    text: outcome.response,
                    usage,
                    prefill_tps,
                    decode_tps: outcome.decode_tps,
                    stop_sequence: outcome.stop_sequence,
                })
            }
            Err(e @ JoshuaError::PromptTooLong(..)) => {
                // A request error, not a backend fault: keep the session and
                // leave the circuit breaker alone.
                if session.clear_state() {
                    self.release_model(session, Vec::new());
                }
                Err(e)
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
        self.complete_with(prompt, true, options, None, None, None)
            .map(Completion::into_tuple)
    }

    /// The context window a request may use: its
    /// [`GenerationOptions::context_window`], which must lie within the
    /// engine's `n_ctx` (the KV cache and RoPE tables are sized at load, so a
    /// request cannot widen it), or the whole window when unset.
    fn request_n_ctx(&self, options: &GenerationOptions) -> Result<usize> {
        match options.context_window {
            None => Ok(self.n_ctx as usize),
            Some(0) => Err(JoshuaError::InvalidRequest(
                "context_window must be at least 1".to_string(),
            )),
            Some(w) if w > self.n_ctx => Err(JoshuaError::InvalidRequest(format!(
                "context_window {w} exceeds the loaded context window of {} tokens \
                 (raise the server's --n-ctx)",
                self.n_ctx
            ))),
            Some(w) => Ok(w as usize),
        }
    }

    /// [`Engine::request_n_ctx`] for an already-validated request: clamps
    /// instead of failing.
    fn effective_n_ctx(&self, options: &GenerationOptions) -> usize {
        options
            .context_window
            .map_or(self.n_ctx, |w| w.clamp(1, self.n_ctx)) as usize
    }

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
            let n_ctx = self.effective_n_ctx(options);
            let remaining = n_ctx.saturating_sub(n_prompt).max(1) as u32;
            cap = cap.min(remaining);
        }
        clamped.max_tokens = cap;
        clamped
    }

    /// Shared completion path.  `add_special_tokens` controls whether the
    /// tokenizer wraps the prompt with its special tokens (disabled for
    /// template-rendered prompts, which already include them).
    ///
    /// `root_chars` is where the prompt's shared head (system prompt and
    /// tools) ends in `prompt`, for the prefix cache's root entry; see
    /// [`Self::prompt_root_chars`].
    fn complete_with(
        &self,
        prompt: &str,
        add_special_tokens: bool,
        options: &GenerationOptions,
        cancel: Option<&AtomicBool>,
        root_chars: Option<usize>,
        mut on_text: Option<&mut (dyn FnMut(&str) + '_)>,
    ) -> Result<Completion> {
        // Bound concurrent heavyweight generations before doing any work.
        let _permit = InFlightGuard::acquire_with(
            &self.in_flight,
            self.max_concurrency,
            self.admission_queue_depth,
            self.queue_timeout,
            cancel,
        )?;
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return Ok(Completion {
                text: String::new(),
                usage: UsageInfo::default(),
                prefill_tps: 0.0,
                decode_tps: 0.0,
                stop_sequence: None,
            });
        }

        // ── Tokenise ─────────────────────────────────────────────────────────
        let encoding = self
            .tokenizer
            .encode(prompt, add_special_tokens)
            .map_err(|e| JoshuaError::Tokenization(e.to_string()))?;
        let prompt_tokens = encoding.get_ids();
        let n_prompt = prompt_tokens.len();

        let n_ctx = self.request_n_ctx(options)?;
        if n_prompt >= n_ctx {
            return Err(JoshuaError::PromptTooLong(n_prompt, n_ctx));
        }

        // Clamp the client-supplied generation length to the server ceiling
        // and the remaining context window.
        let options = &self.clamp_options(options, Some(n_prompt));
        let root = root_chars.and_then(|c| self.root_tokens(prompt, c, add_special_tokens, prompt_tokens));

        // ── Validate the structured-output constraint before any backend work ─
        // A malformed `response_format` (or a tokenizer that cannot support
        // one) is the client's error, not the backend's.  Compiling the
        // grammar here keeps such a request away from the NPU health counter
        // altogether; the decode loop then hits the cached matcher.
        if options.constraint.is_some() {
            self.constraint_matcher(options)?;
        }

        // ── Acquire a session, generate, retry on CPU if the NPU fails ──────
        // Prefer a pooled instance whose state already covers a prefix of
        // this prompt; fall back to a reset instance or a fresh load.
        let (mut session, n_reused) = self.acquire_session(prompt_tokens, true)?;
        let was_npu = session.is_npu();

        let streaming = on_text.is_some();
        let result = self.run_generation(
            &mut session,
            prompt_tokens,
            n_reused,
            options,
            cancel,
            root,
            on_text.as_deref_mut(),
        );
        match result {
            Ok((completion, kv_tokens)) => {
                // Park the instance for reuse by a follow-up request.
                self.release_model(session, kv_tokens);
                Ok(completion)
            }
            // Client errors (`InvalidRequest`) are not backend failures: they
            // must not count against the NPU, and retrying them on the candle
            // path would only reproduce the same error.  They fall through to
            // the plain error arm below.  A streaming request keeps its SSE
            // channel open for the error rather than restarting on CPU.
            Err(e)
                if was_npu
                    && !self.remote_only()
                    && !streaming
                    && !matches!(e, JoshuaError::InvalidRequest(_)) =>
            {
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
                let (mut session, n_reused) = self.acquire_session(prompt_tokens, false)?;
                match self.run_generation(
                    &mut session,
                    prompt_tokens,
                    n_reused,
                    options,
                    cancel,
                    root,
                    None,
                ) {
                    Ok((completion, kv_tokens)) => {
                        self.release_model(session, kv_tokens);
                        Ok(completion)
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

    /// Feed `new_tokens` (the prompt from absolute position `base` on) and
    /// return the last position's logits.
    ///
    /// Process the not-yet-cached prompt tokens in bounded chunks, starting
    /// right after the reused KV prefix.  Feeding an arbitrarily long prompt
    /// in one pass materializes a `[1, n_prompt, hidden]` activation tensor;
    /// chunking caps that transient allocation (finding #4).  Positions
    /// advance so the KV cache accumulates across pieces, and only the final
    /// piece's last-token logits are kept (the prefill prediction).  Smaller
    /// chunks bound memory but can re-read expert weights more often.
    /// Number of prompt tokens to feed in the next prefill chunk.
    ///
    /// With no time budget this is the configured chunk; with one, the shared
    /// cost model picks a size from recently measured chunks.  Always at
    /// least one token and never more than what is left.
    fn next_prefill_chunk(&self, remaining: usize) -> usize {
        if self.prefill_budget.is_zero() {
            return self.prefill_chunk.min(remaining).max(1);
        }
        let cost = self
            .prefill_cost
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        cost.next_chunk(remaining, self.prefill_budget, self.prefill_chunk)
            .min(remaining)
            .max(1)
    }

    fn prefill(
        &self,
        model: &mut GenSession,
        new_tokens: &[u32],
        base: usize,
    ) -> Result<Vec<f32>> {
        let mut last: Vec<f32> = Vec::new();

        // Layer-streaming prefill: sweep each layer once over all chunks,
        // so every layer's weights are read once per prefill instead of
        // once per chunk.  Falls back to the standard chunked loop for
        // architectures without a native streaming path (or when a piece
        // would exceed the KV cap mid-chunk; see below).
        let chunk = self.prefill_chunk;
        let chunks: Vec<crate::stream_prefill::Chunk> = (0..new_tokens.len())
            .step_by(chunk)
            .map(|piece_start| {
                let end = (piece_start + chunk).min(new_tokens.len());
                let piece = &new_tokens[piece_start..end];
                crate::stream_prefill::Chunk {
                    tokens: piece,
                    pos: base + piece_start,
                }
            })
            .collect();

        // Streaming requires every layer's KV to hold the whole prompt;
        // enforce the engine's KV cap up front (per-layer KV is written
        // incrementally and would otherwise overrun mid-stream).  The
        // standard forward enforces the cap inside attention, so mirror
        // that guard here and only stream when the prompt fits.
        let streamable = model.prefill_streamed(&chunks, &self.dense_device);
        Ok(match streamable {
            Ok(Some(v)) => v,
            _ => {
                // Streamed prefill unsupported on this architecture, or a
                // KV-cap guard tripped — fall back to the per-chunk loop.
                // With a time budget, the next chunk comes from the online
                // cost model; otherwise it is the fixed configured size.
                let mut piece_start = 0;
                while piece_start < new_tokens.len() {
                    let remaining = new_tokens.len() - piece_start;
                    let rows = self.next_prefill_chunk(remaining);
                    let end = piece_start + rows;
                    let started = Instant::now();
                    last = model.forward_tokens(
                        &new_tokens[piece_start..end],
                        base + piece_start,
                        &self.dense_device,
                    )?;
                    if !self.prefill_budget.is_zero() {
                        let elapsed = started.elapsed().as_secs_f64();
                        let mut cost = self
                            .prefill_cost
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        cost.observe(end - piece_start, elapsed);
                    }
                    piece_start = end;
                }
                last
            }
        })
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
        cancel: Option<&AtomicBool>,
        root: Option<usize>,
        on_text: Option<&mut (dyn FnMut(&str) + '_)>,
    ) -> Result<(Completion, Vec<u32>)> {
        let n_prompt = prompt_tokens.len();
        let new_tokens = &prompt_tokens[n_reused..];

        // Every token fed to the model so far — i.e. the exact contents of
        // its KV cache.  Returned to the pool with the model afterwards.
        let mut kv_tokens = prompt_tokens.to_vec();

        // ── Prefill ───────────────────────────────────────────────────────────
        // A follow-up turn starts where the last one left off: checkpoint that
        // boundary (a recurrent model's only way back to it), so an agent that
        // replaces the message it is about to read resumes from here.
        if n_reused > 0 {
            model.checkpoint_prefix(n_reused);
        }
        // The prompt's shared head becomes a prefix-cache root unless one is
        // already kept for it.  A model with recurrent layers has its state at
        // the head's end only if the prefill stops there, so it reads the
        // prompt in two parts; an attention-only model reads it in one and
        // the snapshot is cut back to the head.
        let root = root.filter(|&r| {
            r > n_reused && r < n_prompt && model.supports_snapshot() && self.root_wanted(&prompt_tokens[..r])
        });
        let prefill_start = Instant::now();
        let logits_vec = match root {
            Some(r) if !model.supports_truncate() => {
                self.prefill(model, &prompt_tokens[n_reused..r], n_reused)?;
                self.store_root(model, &prompt_tokens[..r], r);
                self.prefill(model, &prompt_tokens[r..], r)?
            }
            Some(r) => {
                let logits = self.prefill(model, new_tokens, n_reused)?;
                self.store_root(model, &prompt_tokens[..r], n_prompt);
                logits
            }
            None => self.prefill(model, new_tokens, n_reused)?,
        };
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        // ── Repetition-penalty history ────────────────────────────────────────
        let outcome = self.decode_loop(
            model,
            logits_vec,
            n_prompt,
            prompt_tokens,
            options,
            cancel,
            on_text,
        )?;
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
            Completion {
                text: outcome.response,
                usage,
                prefill_tps,
                decode_tps: outcome.decode_tps,
                stop_sequence: outcome.stop_sequence,
            },
            kv_tokens,
        ))
    }

    /// Greedy/sampled token generation from an initial logit vector.
    ///
    /// `start_pos` is the absolute position of the next token to feed
    /// (`prompt length` for text prompts, `n_past` after a multimodal
    /// prefill).  `penalty_seed` primes the repetition-penalty window
    /// (empty when prompt tokens are unknown, e.g. multimodal prefill) and
    /// is the context the speculative drafter matches against.
    ///
    /// With speculative decoding enabled on a session that supports it, each
    /// step feeds the chosen token together with a prompt-lookup draft and
    /// verifies the draft against the per-position logits (see
    /// [`crate::speculative`]); otherwise — or whenever the drafter has
    /// nothing to propose — it is the plain one-token loop.
    /// Build (or fetch from the cache) the grammar matcher for this request.
    ///
    /// The tokenizer trie is compiled lazily, once per engine.  A schema that
    /// cannot compile is an invalid request: the client gets a 400 instead of
    /// a prefilled prompt and a garbage completion.
    fn constraint_matcher(&self, options: &GenerationOptions) -> Result<Option<Matcher>> {
        let Some(spec) = options.constraint.as_ref() else {
            return Ok(None);
        };
        let engine = match self.constraints.get_or_init(|| {
            ConstraintEngine::from_path(&self.tokenizer_path, &self.eos_token_ids)
        }) {
            Ok(engine) => engine,
            Err(e) => {
                return Err(JoshuaError::InvalidRequest(format!(
                    "response_format is not available for this tokenizer: {e}"
                )));
            }
        };
        engine
            .matcher(spec)
            .map(Some)
            .map_err(|e| JoshuaError::InvalidRequest(format!("response_format: {e}")))
    }

    fn decode_loop(
        &self,
        model: &mut GenSession,
        mut logits_vec: Vec<f32>,
        start_pos: usize,
        penalty_seed: &[u32],
        options: &GenerationOptions,
        cancel: Option<&AtomicBool>,
        mut on_text: Option<&mut (dyn FnMut(&str) + '_)>,
    ) -> Result<DecodeOutcome> {
        let mut text = DecodeText::new(penalty_seed, &self.tokenizer);
        let mut stream = StreamGate::new(&options.stop_sequences);
        // A fixed seed makes every draw a pure function of (seed, position,
        // stream); without one, pick a request seed once and keep the stream
        // reproducible for the rest of this generation.
        let seed = options.seed.unwrap_or_else(|| thread_rng().gen::<u64>());
        // Draft proposals may be sampled, but verification keeps only the ones
        // that match the target's own draw at that position, so this stream
        // affects how many drafts are accepted, never which tokens are emitted.
        let mut draft_rng = StdRng::seed_from_u64(seed);
        let n_ctx = self.effective_n_ctx(options);
        let mut sampler = SamplingWorkspace::default();
        let mut constraint = self.constraint_matcher(options)?;
        // A user stop string could truncate the document before the grammar
        // completes, producing invalid JSON.  The grammar's own EOS is the
        // only legal stop in constrained mode.
        let constrained_options = constraint.as_ref().map(|_| {
            let mut options = options.clone();
            options.stop_sequences.clear();
            options
        });
        let options = constrained_options.as_ref().unwrap_or(options);
        let mut fed_tokens: Vec<u32> = Vec::new();
        let mut n_cur = start_pos;
        let decode_start = Instant::now();
        // Which configured stop sequence ended generation, if any.
        let mut stop_sequence: Option<String> = None;

        // Speculative decoding needs a text context to draft from and a
        // session that can verify and roll back.
        // A separate draft model comes first; MTP drafting (the model's own
        // NextN block) takes the place of prompt lookup where the session
        // has it.
        let mut draft_model = match &self.draft {
            Some(d)
                if constraint.is_none() && !penalty_seed.is_empty() && model.supports_speculative() =>
            {
                match self.start_draft(d, penalty_seed) {
                    Ok(s) => Some(DraftModelState::new(s, penalty_seed)),
                    Err(e) => {
                        tracing::warn!(error = %e, "draft model failed; decoding plainly");
                        None
                    }
                }
            }
            _ => None,
        };
        let mtp = if constraint.is_none()
            && draft_model.is_none()
            && self.mtp > 0
            && model.mtp_enabled()
            && model.supports_speculative()
        {
            self.mtp
        } else {
            0
        };
        let spec = self.speculative.filter(|_| {
            constraint.is_none()
                && draft_model.is_none()
                && mtp == 0
                && !penalty_seed.is_empty()
                && model.supports_speculative()
        });
        let mut drafter = spec.map(|c| NgramDrafter::new(c, penalty_seed));
        let max_draft = if draft_model.is_some() {
            self.draft_max
        } else if mtp > 0 {
            mtp
        } else {
            spec.map_or(1, |c| c.max_draft)
        };
        let mut draft_len = if draft_model.is_some() || mtp > 0 {
            max_draft
        } else {
            spec.map_or(0, |c| c.max_draft)
        };
        // Holds drafting off while it measures slower than plain decoding.
        let gated = if draft_model.is_some() || mtp > 0 {
            self.speculative.is_none_or(|c| c.cost_gate)
        } else {
            spec.is_some_and(|c| c.cost_gate)
        };
        let mut gate = gated.then(SpeculationGate::default);
        let (mut n_drafted, mut n_accepted, mut n_verify) = (0u64, 0u64, 0u64);
        // A token already chosen by a rejected verification (the
        // correction), to emit before sampling anything new.
        let mut pending: Option<u32> = None;
        let mut drafts = Vec::new();
        let mut block = Vec::new();

        loop {
            if text.n_decoded >= options.max_tokens {
                break;
            }
            // The requester went away: stop as a `max_tokens` stop would, so
            // the session's state stays consistent and reusable.
            if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                break;
            }
            // Never generate past the context window, regardless of
            // max_tokens — bounds KV-cache growth and matches the RoPE tables.
            if n_cur >= n_ctx {
                break;
            }

            let mask = match constraint.as_mut() {
                Some(matcher) => {
                    let mask = matcher.compute_mask_or_eos().map_err(|e| {
                        JoshuaError::Inference(format!("constrained decoding: {e}"))
                    })?;
                    if mask.num_set() == 0 {
                        return Err(JoshuaError::Inference(
                            "constrained decoding reached a dead end".to_string(),
                        ));
                    }
                    Some(mask)
                }
                None => None,
            };
            let next_token = match pending.take() {
                Some(t) => t,
                None => {
                    let t = sampler.sample_at(
                        &logits_vec,
                        options,
                        &text.recent_tokens,
                        mask.as_ref(),
                        seed,
                        n_cur as u64,
                    )?;
                    if debug_tokens() {
                        let mut order: Vec<usize> = (0..logits_vec.len()).collect();
                        order.sort_by(|&a, &b| logits_vec[b].total_cmp(&logits_vec[a]));
                        let top5: Vec<(usize, f32)> = order[..5.min(order.len())]
                            .iter()
                            .map(|&i| (i, logits_vec[i]))
                            .collect();
                        eprintln!(
                            "[eng] n_cur={n_cur} tok={t} temp={} topk={} topp={} minp={} reppen={} top5={top5:?}",
                            options.temperature, options.top_k, options.top_p, options.min_p, options.repetition_penalty
                        );
                    }
                    t
                }
            };

            if self.eos_token_ids.contains(&next_token) {
                break;
            }
            if let Some(matcher) = constraint.as_mut() {
                // The sampler chose a token the mask allowed; advance the
                // grammar before requesting the next mask.
                matcher.consume_token(next_token).map_err(|e| {
                    JoshuaError::Inference(format!("constrained decoding: {e}"))
                })?;
            }
            let stopped = self.emit_token(&mut text, next_token, options)?;
            stream.push(&text.response, stopped.is_some(), on_text.as_deref_mut());
            if let Some(stop) = stopped {
                stop_sequence = Some(stop);
                break;
            }
            if let Some(d) = drafter.as_mut() {
                d.push(next_token);
            }
            if let Some(d) = draft_model.as_mut() {
                d.context.push(next_token);
            }

            // Draft continuation tokens, bounded so that every accepted one
            // can still be emitted (max_tokens) and fed (context window).
            // The gate is asked only when there is a draft, so steps with no
            // match don't use up its probes.
            drafts.clear();
            // An MTP step's cost includes drafting, so its clock starts here.
            let step_start = Instant::now();
            let room = (options.max_tokens - text.n_decoded) as usize;
            let ctx_room = n_ctx.saturating_sub(n_cur + 1);
            if let Some(d) = draft_model.as_mut() {
                // Like MTP, a draft model always has a draft: ask the gate
                // first so a held step costs no draft passes.
                let n = draft_len.min(room).min(ctx_room);
                if n > 0 && gate.as_mut().is_none_or(SpeculationGate::allow) {
                    if let Err(e) =
                        self.draft_with_model(
                            d,
                            n,
                            logits_vec.len(),
                            options,
                            &text.recent_tokens,
                            &mut draft_rng,
                            &mut drafts,
                        )
                    {
                        // A failing draft model must not fail the request:
                        // carry on plainly without it.
                        tracing::warn!(error = %e, "draft model failed; decoding plainly");
                        drafts.clear();
                        draft_model = None;
                    }
                }
            } else if mtp > 0 {
                // The block always has a draft, so the gate is asked first
                // and a held step costs no draft pass.
                let n = draft_len.min(room).min(ctx_room);
                if n > 0 && gate.as_mut().is_none_or(SpeculationGate::allow) {
                    drafts = model.mtp_draft(next_token, n_cur, n)?;
                }
            } else if let Some(d) = drafter.as_mut() {
                d.draft_into(draft_len.min(room).min(ctx_room), &mut drafts);
                if !drafts.is_empty() && !gate.as_mut().is_none_or(SpeculationGate::allow) {
                    drafts.clear();
                }
            }

            if drafts.is_empty() {
                // Single-token decode step.
                logits_vec = model.forward_tokens(&[next_token], n_cur, &self.dense_device)?;
                fed_tokens.push(next_token);
                n_cur += 1;
                if let Some(g) = gate.as_mut() {
                    g.record_plain(step_start.elapsed().as_secs_f64());
                }
                continue;
            }

            // ── Verify the draft in one forward pass ─────────────────────────
            // Row i of `rows` predicts the token after block[i], so row 0
            // checks drafts[0], and the last row is the next step's logits if
            // the whole draft is accepted.
            block.clear();
            block.reserve(drafts.len() + 1);
            block.push(next_token);
            block.extend_from_slice(&drafts);
            let mut rows = model.forward_tokens_all(&block, n_cur, &self.dense_device)?;
            fed_tokens.push(next_token);
            n_cur += 1;
            // `drafts[i]` is decided by row i at this absolute position.  An
            // accepted draft advances `n_cur`, so derive every verification
            // position from here rather than from the moving `n_cur`.
            let draft_pos = n_cur;

            let mut accepted = 0usize;
            let mut kept = 0usize;
            let mut stop = false;
            for (i, &draft) in drafts.iter().enumerate() {
                // Speculation is only enabled when no grammar mask is active
                // (`constraint.is_none()` gates the n-gram drafter, MTP and
                // the draft model), so verification never needs a mask.
                let dist = sampler.distribution(
                    &rows[i], options, &text.recent_tokens, None, (draft_pos + i) as u64,
                )?;
                // `rows[i]` decides the draft at `draft_pos + i`; the draw
                // depends only on that position, so the outcome does not
                // depend on how many earlier drafts this pass accepted.
                match dist.verify_at(draft, seed, (draft_pos + i) as u64)? {
                    Verdict::Accept => {
                        accepted += 1;
                        if self.eos_token_ids.contains(&draft) {
                            stop = true;
                            break;
                        }
                        // The token is in the KV cache already; count it as
                        // fed before the stop check so the cache and
                        // `fed_tokens` agree however the loop exits.
                        kept += 1;
                        fed_tokens.push(draft);
                        n_cur += 1;
                        if let Some(d) = drafter.as_mut() {
                            d.push(draft);
                        }
                        if let Some(d) = draft_model.as_mut() {
                            d.context.push(draft);
                        }
                        let stopped = self.emit_token(&mut text, draft, options)?;
                        stream.push(&text.response, stopped.is_some(), on_text.as_deref_mut());
                        if let Some(matched) = stopped {
                            stop_sequence = Some(matched);
                            stop = true;
                            break;
                        }
                    }
                    Verdict::Reject(correction) => {
                        pending = Some(correction);
                        break;
                    }
                }
            }

            n_drafted += drafts.len() as u64;
            n_accepted += accepted as u64;
            n_verify += 1;
            draft_len = next_draft_len(draft_len, drafts.len(), accepted, max_draft);

            // Roll the KV cache back past every draft token not kept.
            if kept < drafts.len() && !model.truncate_to(n_cur) {
                return Err(JoshuaError::Inference(format!(
                    "speculative decoding could not roll the KV cache back to {n_cur} tokens"
                )));
            }
            // The pass's cost includes its rollback.
            if let Some(g) = gate.as_mut() {
                g.record_verify(accepted + 1, step_start.elapsed().as_secs_f64());
            }
            if stop {
                break;
            }
            if pending.is_none() {
                // Whole draft accepted: the last row predicts the next token.
                logits_vec = rows.swap_remove(drafts.len());
            }
        }

        if n_drafted > 0 {
            self.spec_drafted.fetch_add(n_drafted, Ordering::Relaxed);
            self.spec_accepted.fetch_add(n_accepted, Ordering::Relaxed);
            self.spec_verify_steps.fetch_add(n_verify, Ordering::Relaxed);
            tracing::debug!(
                drafted = n_drafted,
                accepted = n_accepted,
                acceptance = n_accepted as f64 / n_drafted as f64,
                "Speculative decoding"
            );
            // The step is over: the last verification pass is either already
            // rolled back or needs no rollback, so its snapshots can go rather
            // than linger until the next forward pass replaced them.
            model.discard_verify();
        }
        if let (Some(pool), Some(d)) = (self.draft.as_ref(), draft_model) {
            pool.release(d.session);
        }

        stream.finish(&text.response, on_text.as_deref_mut());

        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        let n_decoded = text.n_decoded;
        let decode_tps = if decode_ms > 0.0 && n_decoded > 0 {
            n_decoded as f64 / (decode_ms / 1000.0)
        } else {
            0.0
        };

        Ok(DecodeOutcome {
            response: text.response,
            n_decoded,
            fed_tokens,
            decode_tps,
            stop_sequence,
        })
    }

    /// A draft-model session for a request whose context is `prompt`, its
    /// KV cache brought up to the whole prompt (outside the decode loop's
    /// cost gate, which measures the per-step cost only).
    fn start_draft(
        &self,
        draft: &crate::draft_model::DraftModel,
        prompt: &[u32],
    ) -> Result<crate::draft_model::DraftSession> {
        let mut session = draft.acquire(prompt)?;
        session.sync(prompt, draft.device())?;
        Ok(session)
    }

    /// Draft up to `n` tokens continuing `state.context` with the draft
    /// model, sampling each from the draft's own distribution under the
    /// request's sampling settings (its argmax at temperature 0).  The target
    /// later keeps a draft only when it matches its own draw at that position,
    /// so the draft distribution does not need to be recorded.  Drafting stops
    /// after an end-of-sequence token.
    fn draft_with_model(
        &self,
        state: &mut DraftModelState,
        n: usize,
        target_vocab: usize,
        options: &GenerationOptions,
        recent_tokens: &[u32],
        rng: &mut StdRng,
        drafts: &mut Vec<u32>,
    ) -> Result<()> {
        let Some(draft) = self.draft.as_ref() else {
            return Ok(());
        };
        let device = draft.device();
        let DraftModelState {
            session,
            context,
            sampler,
            window,
        } = state;
        drafts.clear();
        let mut logits = session.sync(context, device)?;
        window.clear();
        window.extend_from_slice(recent_tokens);
        for j in 0..n {
            let dist = sampler.distribution(
                &logits, options, window, None, (context.len() + j) as u64,
            )?;
            let token = dist.sample(rng)?;
            if token as usize >= target_vocab {
                // The draft vocabulary may be longer than the target's; such an
                // id has no target embedding, so end the draft here instead of
                // feeding a block the target cannot score.
                break;
            }
            drafts.push(token);
            if j + 1 == n || self.eos_token_ids.contains(&token) {
                break;
            }
            if window.len() >= REP_WINDOW {
                window.remove(0);
            }
            window.push(token);
            logits = session.push(token, device)?;
        }
        Ok(())
    }

    /// Append one generated token to the response text and the
    /// repetition-penalty window.  Returns the matched stop sequence, if any
    /// (the response is trimmed to before it).
    fn emit_token(
        &self,
        text: &mut DecodeText<'_>,
        token: u32,
        options: &GenerationOptions,
    ) -> Result<Option<String>> {
        if self.byte_level_decode {
            // Byte-level BPE splits multi-byte UTF-8 across token
            // boundaries — a single token can be a lone byte (e.g.
            // DeepSeek's raw-byte vocab entries `¡`..`ÿ`), which is
            // invalid UTF-8 on its own and would decode to U+FFFD.
            // [`ByteWindowDecoder`] keeps that byte state across tokens
            // without re-decoding the whole output every step (the old
            // whole-buffer decode made generation O(n²) in length).
            text.decoded_ids.push(token);
            let response = std::mem::take(&mut text.response);
            text.response =
                text.byte_window
                    .push(&self.tokenizer, token, response, &text.decoded_ids)?;
        } else if self.tokenizer.get_decoder().is_some() {
            // Explicit decoders may need multiple tokens to complete a UTF-8
            // character or determine spacing. Only emitted (accepted) tokens
            // reach this state, including during speculative verification.
            if let Some(piece) = text
                .stream
                .step(token)
                .map_err(|e| JoshuaError::Inference(e.to_string()))?
            {
                text.response.push_str(&piece);
            }
        } else {
            // Decoder-less / word-level tokenizers: batch decoding would
            // join pieces with spaces, so decode each token and append.
            let piece = self
                .tokenizer
                .decode(&[token], false)
                .map_err(|e| JoshuaError::Inference(e.to_string()))?;
            text.response.push_str(&piece);
        }
        text.n_decoded += 1;

        // Maintain sliding-window token history for repetition penalty.
        if text.recent_tokens.len() >= REP_WINDOW {
            text.recent_tokens.remove(0);
        }
        text.recent_tokens.push(token);

        Ok(Self::check_stop_sequences(
            &mut text.response,
            &options.stop_sequences,
        ))
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
        self.embed_inner(texts, None)
    }

    /// [`Engine::embed_with_usage`] that gives up once `cancel` is set.
    ///
    /// The flag is checked before each input text, so a batch whose
    /// requester went away (an HTTP client, or a `joshua route` coordinator
    /// that closed its connection) releases the engine's concurrency permit
    /// after at most one more text instead of embedding the whole batch for
    /// nobody.  A cancelled call returns [`JoshuaError::Cancelled`].
    pub fn embed_with_usage_cancellable(
        &self,
        texts: &[String],
        cancel: &AtomicBool,
    ) -> Result<(Vec<Vec<f32>>, u32)> {
        self.embed_inner(texts, Some(cancel))
    }

    fn embed_inner(
        &self,
        texts: &[String],
        cancel: Option<&AtomicBool>,
    ) -> Result<(Vec<Vec<f32>>, u32)> {
        if self.remote_only() {
            return Err(JoshuaError::InvalidRequest(
                "embeddings are not available for a model served by remote workers".into(),
            ));
        }
        // Embeddings also load/hold a model instance — bound concurrency.
        let _permit = InFlightGuard::acquire_with(
            &self.in_flight,
            self.max_concurrency,
            self.admission_queue_depth,
            self.queue_timeout,
            cancel,
        )?;
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return Err(JoshuaError::Cancelled);
        }
        let model = self.embedding_model()?;
        let mut vectors = Vec::with_capacity(texts.len());
        let mut total_tokens: u32 = 0;
        for text in texts {
            if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                return Err(JoshuaError::Cancelled);
            }
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

    /// Speculative decoding settings in effect, if enabled — `None` also
    /// when a configuration was requested but the model's architecture
    /// cannot run speculative decoding (see
    /// [`crate::model::Architecture::unsupported_speculative_reason`]),
    /// rather than a setting decode time would silently skip.
    pub fn speculative_config(&self) -> Option<SpeculativeConfig> {
        self.speculative
    }

    /// Enable (or disable) speculative token generation after construction.
    /// See [`EngineOptions::speculative`].
    ///
    /// Gated on the architecture like the constructor: on a model that
    /// cannot run speculative decoding this warns and stores `None` rather
    /// than a configuration decode time would silently skip.
    pub fn with_speculative(mut self, config: Option<SpeculativeConfig>) -> Self {
        self.speculative = config.and_then(|config| match self.arch {
            Some(arch) => match arch.unsupported_speculative_reason() {
                Some(reason) => {
                    tracing::warn!(
                        "speculative decoding is not available on {}: {reason}; \
                         the request is ignored for this model",
                        arch.display_name(),
                    );
                    None
                }
                None => Some(SpeculativeConfig::normalized(config)),
            },
            None => {
                tracing::warn!(
                    "speculative decoding requested, but no candle loader was detected for this \
                     model (unsupported architecture or NPU-only); the request is ignored"
                );
                None
            }
        });
        self
    }

    /// Speculative-decoding counters accumulated over every request so far.
    pub fn speculative_stats(&self) -> SpeculativeStats {
        SpeculativeStats {
            drafted: self.spec_drafted.load(Ordering::Relaxed),
            accepted: self.spec_accepted.load(Ordering::Relaxed),
            verify_steps: self.spec_verify_steps.load(Ordering::Relaxed),
        }
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
            exclusive: false,
        });
        self
    }

    /// Run every generation on `backend` — a model whose weights live on
    /// other machines — and never load them locally: backend failures are
    /// returned to the caller instead of retried on the candle path, and
    /// embeddings are refused. `capacity` is how many sessions the backend
    /// holds at once; in-flight requests plus warm pooled sessions stay
    /// within it.
    pub fn with_remote_backend(mut self, backend: Arc<dyn NpuBackend>, capacity: usize) -> Self {
        tracing::info!("remote backend configured: {}", backend.name());
        self.remote_capacity = Some(capacity.max(1));
        self.fit_remote_capacity();
        self.npu = Some(NpuState {
            backend,
            failures: AtomicU32::new(0),
            disabled: AtomicBool::new(false),
            exclusive: true,
        });
        self
    }

    /// Split a remote backend's sessions between in-flight requests and the
    /// warm pool, whichever builder ran last.
    fn fit_remote_capacity(&mut self) {
        if let Some(capacity) = self.remote_capacity {
            // Keep about half the sessions warm for multi-turn KV reuse.
            self.max_concurrency = self.max_concurrency.min(capacity.div_ceil(2));
            self.remote_pool_cap = Some(capacity - self.max_concurrency);
        }
    }

    /// Whether every forward pass runs on a remote pipeline (this host has
    /// no weights to compute with, so embeddings are unavailable).
    pub fn remote_only(&self) -> bool {
        self.npu.as_ref().is_some_and(|n| n.exclusive)
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
        // Whether the prefix cache holds a longer reusable state than any
        // pooled session can be rewound to.
        let mut from_cache = false;
        // A pooled session to clear for this unrelated prompt.
        let mut clear_pick: Option<CachedModel> = None;

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
            // A longer prefix-cache entry (a parked conversation that went
            // further) beats continuing a shorter pooled one, which then
            // stays pooled.
            let cached = if want_npu { 0 } else { self.prefix_cache().best_len(prompt_tokens) };
            from_cache = best.is_some_and(|i| cached > pool[i].tokens.len());
            if let Some(i) = best.filter(|_| !from_cache) {
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
            //
            // A model with recurrent layers rewinds only to a prefix
            // checkpoint (a copy of its running state taken at the end of an
            // earlier prefill), so its candidate keeps the latest checkpoint
            // at or before that point rather than the whole common prefix.
            if !want_npu && !from_cache {
                let best_edit = pool
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| !c.session.is_npu())
                    .map(|(i, c)| {
                        let lcp = longest_common_prefix(&c.tokens, prompt_tokens);
                        // Rollback case: keep one token behind so the suffix
                        // prefill is never empty.
                        let keep = lcp.min(prompt_tokens.len().saturating_sub(1));
                        (i, c.session.rewind_target(keep))
                    })
                    .filter(|(_, keep)| *keep > 0)
                    // Longest reusable prefix first; pass 1's exact-prefix
                    // matches are intentionally preferred over longer edit
                    // candidates because they need no KV copy at all.
                    .max_by_key(|(_, keep)| *keep);
                // A prefix-cache entry at least as long as the best rewind
                // wins: starting from it leaves the pooled conversation whole
                // (a new chat sharing only the system prompt would otherwise
                // rewind, and so destroy, another chat's session).
                match best_edit {
                    Some((_, keep)) if cached >= keep => from_cache = true,
                    Some((i, keep)) => edit_pick = Some((pool.swap_remove(i), keep)),
                    None => from_cache = cached > 0,
                }
            }
            // The clear path runs only when no rewind was picked — rewinding
            // preserves strictly more work than clearing.
            if edit_pick.is_none() && !from_cache {
                let resettable = pool.iter().position(|c| c.session.is_npu() == want_npu);
                clear_pick = resettable.map(|i| pool.swap_remove(i));
            }
        }

        // Clearing a pooled session ends its conversation; park it first
        // (outside the pool lock) so it can resume when its client returns —
        // two alternating conversations would otherwise each re-read the
        // other's history away.
        if let Some(mut cached) = clear_pick {
            self.park_session(&cached.session, &cached.tokens);
            if cached.session.clear_state() {
                tracing::debug!(npu = want_npu, "Reusing pooled session with cleared state");
                return Ok((cached.session, 0));
            }
            // Reset failed (e.g. dead shim): drop it and fall through.
        }

        if from_cache {
            if let Some(resumed) = self.resume_from_prefix_cache(prompt_tokens)? {
                return Ok(resumed);
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
                    if self.arch_error.is_some() || npu.exclusive {
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

    // ─── Prefix cache ─────────────────────────────────────────────────────────

    /// Lock the prefix cache (poison is recovered for the same reason as the
    /// pool's: its entries are plain data).
    fn prefix_cache(&self) -> std::sync::MutexGuard<'_, crate::prefix_cache::PrefixCache<ModelSnapshot>> {
        self.prefix_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The prefix cache's counters and occupancy.
    pub fn prefix_cache_stats(&self) -> crate::prefix_cache::PrefixCacheStats {
        self.prefix_cache().stats()
    }

    /// Whether free RAM leaves room to store another snapshot: the cache's
    /// budget is a ceiling, not a reservation, so a box that has since filled
    /// up keeps its roots but parks nothing more.
    fn ram_allows_parking() -> bool {
        Self::ram_allows_copy(0)
    }

    /// Whether host RAM can take a `bytes` snapshot and still keep
    /// [`Self::ram_allows_parking`]'s margin: the copy exists alongside the
    /// live session it is taken from.
    fn ram_allows_copy(bytes: u64) -> bool {
        crate::placement::available_ram_bytes()
            .is_none_or(|free| free >= LOW_MEM_FLOOR * 3 + bytes)
    }

    /// Whether a snapshot of `session` after `keep` of its `len` tokens fits
    /// the cache and host RAM, checked before copying anything.  The size is
    /// an upper bound: the whole state, scaled to `keep` for an attention-only
    /// model, whose state is all KV rows.
    fn snapshot_fits(&self, session: &GenSession, keep: usize, len: usize) -> bool {
        let mut bytes = session.state_bytes();
        if session.supports_truncate() && keep < len && len > 0 {
            bytes = (bytes as u128 * keep as u128 / len as u128) as u64;
        }
        self.prefix_cache().fits(bytes) && Self::ram_allows_copy(bytes)
    }

    /// Where the shared head of a chat prompt ends, as a byte offset into the
    /// rendered `prompt`: the start of the first non-system message, i.e. the
    /// system prompt and the tool list every new chat of a client repeats.
    ///
    /// Found template-agnostically: the conversation is rendered again with
    /// that message's text replaced, and the two renders agree exactly up to
    /// it.  `None` with the prefix cache off or no such message.
    fn prompt_root_chars(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        options: &GenerationOptions,
        prompt: &str,
    ) -> Option<usize> {
        if !self.prefix_cache().enabled() {
            return None;
        }
        let i = messages.iter().position(|m| m.role != "system")?;
        let mut variant = messages.to_vec();
        variant[i].content = if messages[i].content.starts_with('\u{1}') { "\u{2}" } else { "\u{1}" }.into();
        let (other, _) = self.format_prompt(&variant, tools, &options.chat_template_kwargs);
        let n = crate::prefix_cache::common_text_prefix(prompt, &other);
        (n > 0).then_some(n)
    }

    /// The prompt's shared head in tokens: the longest run of `prompt_tokens`
    /// that the text before `root_chars` tokenizes to (cutting the text can
    /// change how its last token splits, so the common prefix is what both
    /// agree on).  `None` below the root minimum.
    fn root_tokens(
        &self,
        prompt: &str,
        root_chars: usize,
        add_special_tokens: bool,
        prompt_tokens: &[u32],
    ) -> Option<usize> {
        let head = self
            .tokenizer
            .encode(prompt.get(..root_chars)?, add_special_tokens)
            .ok()?;
        let n = crate::prefix_cache::common_token_prefix(head.get_ids(), prompt_tokens);
        (n >= self.prefix_cache_root_min).then_some(n)
    }

    /// Whether `head` should become a root entry: the cache is on, keeps none
    /// for it yet, and the box has room.
    fn root_wanted(&self, head: &[u32]) -> bool {
        let cache = self.prefix_cache();
        cache.enabled() && !cache.has_root(head) && Self::ram_allows_parking()
    }

    /// Keep the state after `head` (of the `len` tokens the session has been
    /// fed) as a root entry.  Best-effort: a model outside the shared session,
    /// or a snapshot over budget, stores nothing.
    fn store_root(&self, model: &GenSession, head: &[u32], len: usize) {
        if !self.snapshot_fits(model, head.len(), len) {
            return;
        }
        if let Some(snap) = model.snapshot(head.len(), len) {
            let stored = self.prefix_cache().insert(head.to_vec(), snap, true);
            tracing::debug!(tokens = head.len(), stored, "prefix cache: system-prompt root");
        }
    }

    /// Park a session the warm pool is letting go of, so its conversation can
    /// resume later without re-reading its history.
    fn park_session(&self, session: &GenSession, tokens: &[u32]) {
        let min = PREFIX_CACHE_PARK_MIN.min(self.prefix_cache_root_min);
        if tokens.len() < min || !self.prefix_cache().enabled() {
            return;
        }
        if !Self::ram_allows_parking() {
            self.prefix_cache().shed();
            return;
        }
        if !self.snapshot_fits(session, tokens.len(), tokens.len()) {
            return;
        }
        if let Some(snap) = session.snapshot(tokens.len(), tokens.len()) {
            let stored = self.prefix_cache().insert(tokens.to_vec(), snap, false);
            tracing::debug!(tokens = tokens.len(), stored, "prefix cache: parked a conversation");
        }
    }

    /// Start a session from the longest prefix-cache entry `prompt_tokens`
    /// extends: a fresh session over the shared weights takes the entry's
    /// state, so no pooled conversation is disturbed.  `None` when nothing
    /// matches or the state cannot be restored.
    fn resume_from_prefix_cache(&self, prompt_tokens: &[u32]) -> Result<Option<(GenSession, usize)>> {
        let hit = {
            let mut cache = self.prefix_cache();
            match cache.take(prompt_tokens, |s| s.duplicate(&Device::Cpu)) {
                Ok(hit) => hit,
                Err(e) => {
                    tracing::debug!(error = %e, "prefix cache: copying a root failed");
                    None
                }
            }
        };
        let Some(hit) = hit else {
            return Ok(None);
        };
        let mut session = GenSession::Candle(Box::new(self.load_model()?));
        if !session.restore(hit.snap) {
            return Ok(None);
        }
        self.kv_reuses.fetch_add(1, Ordering::Relaxed);
        tracing::debug!(
            reused_tokens = hit.len,
            prompt_tokens = prompt_tokens.len(),
            root = hit.root,
            "Continuing from the prefix cache"
        );
        Ok(Some((session, hit.len)))
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
        let mut pool = self.model_pool();
        pool.push(CachedModel { session, tokens });
        // Evict oldest beyond the cap.  Under memory pressure the pool also
        // holds fewer (or no) idle sessions, so a heavy model's working set
        // isn't kept resident twice/three times just to warm a next request
        // (finding #2).
        // RAM-adaptive pool cap (#2 + #5).  Under pressure keep few/no idle
        // sessions (#2).  With ample free RAM keep MORE than the baseline so
        // concurrent requests REUSE one loaded model (and one GPU upload) rather
        // than each load+upload its own working set — that is the "share
        // immutable weights across sessions" win (#5), bounded by memory.
        let mut pool_cap: usize = match crate::placement::available_ram_bytes() {
            Some(free) if free < LOW_MEM_FLOOR => 0,
            Some(free) if free < LOW_MEM_FLOOR * 2 => 1,
            Some(free) if free >= MAX_CACHED_MODELS_MEM_HEADROOM => MAX_CACHED_MODELS_MEM,
            _ => MAX_CACHED_MODELS,
        };
        // On an accelerator, a session of an architecture that cannot share
        // weights is a whole copy of the model in device memory; the device
        // budget, not free RAM, bounds how many can stay warm.  Sessions
        // derived from the shared template are only a KV cache and keep the
        // RAM rule.
        if let Some(cap) = self.device_session_cap {
            let owns_weights = pool.last().is_some_and(
                |c| matches!(&c.session, GenSession::Candle(m) if !m.supports_shared_weights()),
            );
            if owns_weights {
                pool_cap = pool_cap.min(cap);
            }
        }
        if let Some(cap) = self.remote_pool_cap {
            pool_cap = cap;
        }
        let excess = pool.len().saturating_sub(pool_cap);
        let evicted: Vec<CachedModel> = pool.drain(..excess).collect();
        drop(pool);
        // Copy each evicted conversation into the prefix cache outside the
        // pool lock (a large KV copy must not stall other requests).
        for c in evicted {
            self.park_session(&c.session, &c.tokens);
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
        // Weight-sharing architectures load once: the first load becomes the
        // template and every session — this one included — is derived from
        // it, sharing the weights and adding only a KV cache.  The template
        // lock is held across the load so two first requests cannot race
        // into two uploads; later callers find the template and derive a
        // session in microseconds.
        let mut template = self
            .weights_template
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(session) = template.as_ref().and_then(|t| t.new_session()) {
            return Ok(session);
        }
        let mut cursor = Cursor::new(&self.mmap[..]);
        let gguf = read_gguf_header(&self.mmap)
            .map_err(|e| JoshuaError::ModelLoad(format!("GGUF read failed: {e}")))?;
        // Hand the loader the mapping so architectures Joshua implements
        // itself can borrow weights in place rather than copying them.
        let load = || {
            QuantizedModel::from_gguf_mmap_placed_with_ssd(
                gguf,
                &mut cursor,
                &self.dense_device,
                &self.expert_device,
                Some(Arc::clone(&self.mmap)),
                self.model_file.clone(),
                self.n_ctx as usize,
                self.device_expert_cache,
                self.engram_tier,
                self.ssd_expert_cache,
            )
        };
        // Read the NextN block only when it will draft.
        let mut model = if self.mtp > 0 {
            crate::quantized_qwen::with_mtp_block(load)
        } else {
            load()
        }
        .map_err(|e| JoshuaError::ModelLoad(format!("model init failed: {e}")))?;
        let mut budget = if self.expert_cache_auto {
            self.auto_expert_budget(&model)
        } else {
            self.pin_hot_experts
        };
        if let Some(cache) = model.device_expert_cache() {
            tracing::info!(
                "vram expert cache: {} slots ({:.2} GiB); {}",
                cache.slots,
                cache.budget_bytes as f64 / 2f64.powi(30),
                if budget == 0 {
                    "protecting the routing-frequency hot set (up to 3/4 of the slots) from eviction"
                } else {
                    "the --pin-hot-experts / --expert-cache hot set is protected from eviction"
                }
            );
            if budget == 0 {
                // With a device pool the hot set is what stays resident
                // under churn; size it to the pool's protected share so the
                // frequency policy runs without an explicit host budget.
                budget = cache.automatic_hot_budget();
            }
        }
        // Tier 3: the SSD staging slab's engaged budget, alongside the VRAM
        // cache line above.  Only a deepseek4 load can have one.
        if let Some(cache) = model.ssd_expert_cache() {
            tracing::info!(
                "ssd expert cache (Tier 3): {} experts ({:.2} GiB staged-RAM budget); \
                 hot/staged experts are pread(2)-streamed from the model file",
                cache.slots,
                cache.budget_bytes as f64 / 2f64.powi(30),
            );
        }
        model.set_pin_hot_experts(budget);
        model.set_prefix_checkpoints(self.prefix_checkpoints);
        model.set_speculative_expert_prefetch(self.predict_experts);

        // Engram residency is reported separately from the routed-expert cache:
        // the engram tables are an independent placement policy, always
        // resolved straight from the mapping, never a Tier-1/2 slot.
        if let Some(eg) = model.engram_residency() {
            use crate::placement::EngramResidency;
            let tier = match eg.tier {
                EngramResidency::PromoteToRam => "ram",
                EngramResidency::ResolveFromMmap => "disk",
            };
            match eg.resident_bytes {
                Some(resident) => tracing::info!(
                    "engram residency [separate from expert cache]: tier={tier} layers={} mapped={:.2} GiB resident={:.2} GiB",
                    eg.layers,
                    eg.bytes_total as f64 / 2f64.powi(30),
                    resident as f64 / 2f64.powi(30),
                ),
                None => tracing::info!(
                    "engram residency [separate from expert cache]: tier={tier} layers={} mapped={:.2} GiB (resident unmeasurable)",
                    eg.layers,
                    eg.bytes_total as f64 / 2f64.powi(30),
                ),
            }
        }

        if self.mtp > 0 && !model.set_mtp(true) {
            tracing::warn!(
                "--mtp requested, but this model has no NextN / MTP block the loader reads; \
                 the flag is ignored"
            );
        }
        match model.new_session() {
            Some(session) => {
                tracing::info!(
                    "weights loaded once and shared: every session adds only its KV cache"
                );
                *template = Some(model);
                Ok(session)
            }
            None => Ok(model),
        }
    }

    /// Size the protected hot set from the resolved device pool when one
    /// exists; otherwise use available RAM minus KV/OS headroom.
    fn auto_expert_budget(&self, model: &QuantizedModel) -> usize {
        if let Some(cache) = model.device_expert_cache() {
            return cache.automatic_hot_budget();
        }
        let Some(free) = crate::placement::available_ram_bytes() else {
            tracing::info!("expert cache auto: no memory probe — leaving the budget 0");
            return 0;
        };
        let capacity = model.expert_residency_capacity();
        if capacity == 0 {
            tracing::info!("expert cache auto: no mmap-backed experts — leaving the budget 0");
            return 0;
        }
        let expert_bytes = (self.mmap.len() / capacity) as u64;
        let budget = crate::placement::expert_budget_for_memory(
            free,
            expert_bytes,
            capacity,
            AUTO_EXPERT_HEADROOM,
        );
        tracing::info!(
            "expert cache auto: {budget} experts from {:.1} GiB free (expert ~{:.1} MiB, capacity {capacity})",
            free as f64 / 2f64.powi(30),
            expert_bytes as f64 / 2f64.powi(20),
        );
        budget
    }

    /// Scan `response` for any configured stop sequence and truncate it.
    /// Returns the matched sequence so the caller can report why decoding
    /// stopped.
    fn check_stop_sequences(response: &mut String, stops: &[String]) -> Option<String> {
        for stop in stops {
            if stop.is_empty() {
                continue;
            }
            if response.ends_with(stop.as_str()) {
                response.truncate(response.len() - stop.len());
                return Some(stop.clone());
            }
        }
        None
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

/// Memory reserved for the KV cache and the OS when auto-sizing the expert
/// cache (see [`Engine::auto_expert_budget`]).
const AUTO_EXPERT_HEADROOM: u64 = 2 * 1024 * 1024 * 1024;
/// Reserved KV-cache bytes kept clear of the VRAM expert cache (#62 §5):
/// the cache must never push a session's KV out of memory.
const DEVICE_KV_RESERVE_BYTES: u64 = 1 << 30;

/// Device memory left for transient buffers (prefill dequantize scratch,
/// packed index/weight uploads, outputs in flight) when the VRAM expert
/// cache is sized automatically: the dense set on the CPU leaves only the
/// expert kernels' scratch on the device; with it on the device the dense
/// prefill dequantize joins in.
const DEVICE_SCRATCH_RESERVE_DENSE_ON_CPU: u64 = 512 << 20;
const DEVICE_SCRATCH_RESERVE_DENSE_ON_DEVICE: u64 = 1 << 30;

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

/// What the kernel is willing to do about transparent huge pages for the VMA
/// at `addr`, read back from `/proc/self/smaps`.
///
/// `madvise(MADV_HUGEPAGE)` returns success as long as the *hint* was accepted;
/// it says nothing about whether the VMA can ever be promoted.  For a shared
/// file-backed mapping on a filesystem without file THP support (ext4 among
/// them) the hint succeeds and the pages stay at base size — a silent no-op.
/// `THPeligible` is the kernel's own answer to "can this mapping get huge
/// pages", so it is what the load path reports instead of the return code.
///
/// `None` means the VMA could not be inspected (no `/proc`, the mapping moved,
/// or an unexpected smaps layout); callers must not claim success in that case.
#[cfg(target_os = "linux")]
pub fn thp_eligible_for(addr: usize) -> Option<bool> {
    use std::io::BufRead;
    let file = std::fs::File::open("/proc/self/smaps").ok()?;
    let reader = std::io::BufReader::new(file);
    // Streaming, not a whole-file read: a process with many mappings has a large
    // smaps, and this runs on the load path.  Lines before the region of interest
    // are skipped, and iteration stops as soon as the region ends.
    let mut in_region = false;
    for line in reader.lines() {
        let Ok(line) = line else { break };
        // Header: "7f8a1c000000-7f8a1c001000 rw-p 00000000 00:00 0". Every other
        // line belongs to the region most recently seen. The mapping pointer is
        // only guaranteed to fall *inside* its VMA, so match by containment.
        if let Some((lo, hi)) = line
            .split(' ')
            .next()
            .and_then(|first| first.split_once('-'))
        {
            let parsed = (
                usize::from_str_radix(lo, 16).ok(),
                usize::from_str_radix(hi, 16).ok(),
            );
            match parsed {
                (Some(start), Some(end)) => {
                    if in_region {
                        // Passed the region without reading its answer.
                        break;
                    }
                    in_region = start <= addr && addr < end;
                }
                // Not a header (a field line never parses as two hex numbers);
                // fall through to the field check below.
                _ => {}
            }
            continue;
        }
        if in_region {
            if let Some(v) = line.strip_prefix("THPeligible:") {
                return Some(v.trim() == "1");
            }
        }
    }
    // `None`, not `Some(false)`: a kernel that does not report the field has not
    // said the mapping is ineligible, and the caller must not claim it did.
    None
}

/// Non-Linux has no `/proc/self/smaps` and no file-backed THP to verify.
#[cfg(not(target_os = "linux"))]
pub fn thp_eligible_for(_addr: usize) -> Option<bool> {
    None
}

/// Ask for transparent huge pages on the model mapping and report what the
/// kernel will actually do, not what the hint returned.
fn advise_transparent_huge_pages(mmap: &Mmap) {
    #[cfg(target_os = "linux")]
    {
        match mmap.advise(memmap2::Advice::HugePage) {
            Ok(()) => match thp_eligible_for(mmap.as_ptr() as usize) {
                Some(true) => tracing::info!(
                    "transparent huge pages are eligible for the model mapping; the kernel will \
                     promote pages as they are touched"
                ),
                Some(false) => tracing::warn!(
                    "transparent huge pages were requested but the kernel will not promote this \
                     mapping: THPeligible=0 on a shared file-backed region. Weights stay on base \
                     (4 KiB) pages. Use --huge-pages 2mb or 1gb for an anonymous huge-page \
                     mapping, which does take effect."
                ),
                None => tracing::warn!(
                    "transparent huge pages were requested but eligibility could not be read from \
                     /proc/self/smaps; the mapping may still be on base pages"
                ),
            },
            Err(e) => tracing::warn!("transparent huge pages unavailable; using normal pages: {e}"),
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = mmap;
        tracing::warn!("transparent huge pages are Linux-only; using normal pages");
    }
}

    if huge == HugePages::Transparent {
        advise_transparent_huge_pages(&mmap);
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

use crate::moe::is_routed_expert;

/// Names a loader may read as the token-embedding table, in precedence:
/// `token_embd.weight` everywhere, plus two aliases candle's `lfm2` loader
/// probes.
const EMBEDDING_NAMES: [&str; 3] = [
    "token_embd.weight",
    "tok_embeddings.weight",
    "model.embed_tokens.weight",
];

/// The names `arch`'s loader probes for a tensor, in its precedence order:
/// candle's `lfm2` loader tries every alias in `all`; every other loader
/// reads only the canonical first name.
fn probed_names(
    arch: Option<Architecture>,
    all: &'static [&'static str],
) -> &'static [&'static str] {
    if arch == Some(Architecture::Lfm2) {
        all
    } else {
        &all[..1]
    }
}

/// The embedding-table names `arch`'s loader probes, in its precedence order.
fn probed_embedding_names(arch: Option<Architecture>) -> &'static [&'static str] {
    probed_names(arch, &EMBEDDING_NAMES)
}

/// Tensor-name stems (see [`tensor_stem`]) that a loader keeps as quantized
/// `QMatMul` weights on an accelerator: the attention and feed-forward
/// projections of the joshua-native loaders (`quantized_qwen` incl. its
/// Gated DeltaNet projections, `deepseek2`, `deepseek4`) and the stock
/// candle loaders (`llama` incl. Mixtral's
/// per-expert `ffn_gate.<i>` matrices, `gemma*`, `lfm2` with its
/// `feed_forward.*` / `mlp.*_proj` / `shortconv.*_proj` aliases, `phi*`,
/// `qwen2`, `qwen3`), deepseek4's hyper-connection, indexer and compressor
/// projections, GLM-5.x's indexer key and KDA low-rank gate projections,
/// and Kimi K3's full-rank KDA gate and latent-MoE projections.
const QUANTIZED_MATRIX_STEMS: [&str; 68] = [
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "attn_qkv",
    "attn_q_a",
    "attn_q_b",
    "attn_kv_a_mqa",
    "attn_kv_b",
    "attn_kv",
    "attn_output_a",
    "attn_output_b",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
    "ffn_gate_shexp",
    "ffn_up_shexp",
    "ffn_down_shexp",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "hc_attn_fn",
    "hc_ffn_fn",
    "output_hc_fn",
    "indexer.proj",
    "indexer.attn_q_b",
    "indexer.attn_k",
    "ssm_f_a",
    "ssm_f_b",
    "ssm_g_a",
    "ssm_g_b",
    "ssm_g",
    "ffn_routed_down",
    "ffn_routed_up",
    "attn_compressor_kv",
    "attn_compressor_gate",
    "indexer_compressor_kv",
    "indexer_compressor_gate",
    "shortconv.in_proj",
    "shortconv.out_proj",
    "conv.in_proj",
    "conv.out_proj",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.out_proj",
    "attn_gate",
    "ssm_in",
    "ssm_ba",
    "ssm_beta",
    "ssm_alpha",
    "ssm_out",
    "hc_attn_down",
    "hc_attn_up",
    "hc_attn_inject",
    "hc_ffn_down",
    "hc_ffn_up",
    "hc_ffn_inject",
    "output_hc_down",
    "output_hc_up",
    "ple_key",
    "ple_value",
    "indexer.q_proj",
    "indexer.k_proj",
    "per_layer_token_embd",
];

/// Tensor-name stems a loader is known to hold as dequantized f32 on an
/// accelerator (besides norms and biases, matched by name): deepseek2's
/// split-KV halves (folded into a dense up-projection), deepseek4's
/// hyper-connection and compressor position vectors, lfm2's conv kernels,
/// the Qwen loader's DeltaNet conv kernels / decay rates and shared-expert
/// gate vectors, the per-projection KDA conv kernels (GLM-5.3-Flash, Kimi)
/// and Kimi K3's attention-residual scores.
/// The MoE router (`ffn_gate_inp`) is handled per architecture in
/// [`residency`]: the joshua-native loaders read it with `f32_tensor`, the
/// stock `llama` loader (Mixtral) keeps it a quantized `QMatMul`.
const F32_STEMS: [&str; 21] = [
    "attn_k_b",
    "attn_v_b",
    "hc_attn_base",
    "hc_attn_scale",
    "hc_ffn_base",
    "hc_ffn_scale",
    "output_hc_base",
    "output_hc_scale",
    "attn_compressor_ape",
    "indexer_compressor_ape",
    "shortconv.conv",
    "ssm_conv1d",
    "ssm_conv1d_q",
    "ssm_conv1d_k",
    "ssm_conv1d_v",
    "ssm_a",
    "ffn_gate_inp_shexp",
    "ple_conv1d",
    "attn_res_score",
    "ffn_res_score",
    "output_res_score",
];

/// `name` without its `blk.<n>.` prefix, `.weight` suffix and any trailing
/// `.<index>` (Mixtral's per-expert `ffn_gate.3.weight`) — the part the
/// residency rules key on.  `blk.3.attn_q.weight` → `attn_q`,
/// `output.weight` → `output`, `blk.0.attn_q.bias` → `attn_q.bias`,
/// `blk.0.ffn_up.7.weight` → `ffn_up`.
fn tensor_stem(name: &str) -> &str {
    let rest = match name.strip_prefix("blk.") {
        Some(r) => r.split_once('.').map(|(_, tail)| tail).unwrap_or(r),
        None => name,
    };
    let rest = rest.strip_suffix(".weight").unwrap_or(rest);
    match rest.rsplit_once('.') {
        Some((head, idx)) if !idx.is_empty() && idx.bytes().all(|b| b.is_ascii_digit()) => head,
        _ => rest,
    }
}

/// How a loader keeps a tensor resident on an accelerator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Residency {
    /// Borrowed or loaded on CPU even when dense projections use a GPU.
    Host,
    /// In its quantized blocks (a `QMatMul` weight, a routed expert).
    Quantized,
    /// Dequantized to f32 (norms, biases, dequantized tables, …).
    F32,
    /// No rule names the tensor: somewhere between the two.
    Unknown,
}

/// The residency of tensor `name` under `arch`'s loader.
///
/// * DeepSeek-V4.1 Engram tables stay on CPU; only gathered rows transfer.
/// * Routed experts are never dequantized.
/// * Every loader decodes each dense tensor in a dtype candle cannot
///   represent (the i-quants, MXFP4, Q1_0, I32, …) to f32 on the device.
/// * The embedding table is f32 for the stock candle loaders, and for float
///   dtypes on an accelerator with the joshua-native loaders
///   ([`crate::token_embedding::TokenEmbedding`]); quantized otherwise.
/// * Output heads and the [`QUANTIZED_MATRIX_STEMS`] are `QMatMul`s.
/// * Norms (`RmsNorm::from_qtensor` dequantizes), biases and the
///   [`F32_STEMS`] are f32.
/// * Anything else — or anything at all for a model candle cannot load —
///   is [`Residency::Unknown`].
fn residency(
    arch: Option<Architecture>,
    name: &str,
    info: &crate::gguf_ext::RawTensorInfo,
) -> Residency {
    if is_routed_expert(name) {
        return Residency::Quantized;
    }
    let Some(arch) = arch else {
        return Residency::Unknown;
    };
    let stem = tensor_stem(name);
    if arch == Architecture::DeepSeek41 {
        match stem {
            // The DeepSeek loader keeps this table on CPU; only selected
            // rows cross to the accelerator before the wkv projection.
            "engram_embd" => return Residency::Host,
            "engram_q" | "engram_k" => return Residency::F32,
            _ => {}
        }
    }
    if !crate::gguf_ext::is_candle_supported(info.dtype) {
        return Residency::F32;
    }
    if probed_embedding_names(Some(arch)).contains(&name) {
        // GGUF dtype ids: 0 = F32, 1 = F16, 30 = BF16.
        let float_dtype = matches!(info.dtype, 0 | 1 | 30);
        return if arch.is_native() && !float_dtype {
            Residency::Quantized
        } else {
            Residency::F32
        };
    }
    if OUTPUT_HEAD_NAMES.contains(&name) {
        return Residency::Quantized;
    }
    if name.ends_with(".bias") || name.contains("norm") {
        return Residency::F32;
    }
    if stem == "ffn_gate_inp" {
        return match arch {
            Architecture::Llama => Residency::Quantized,
            a if a.is_native() => Residency::F32,
            _ => Residency::Unknown,
        };
    }
    if QUANTIZED_MATRIX_STEMS.contains(&stem)
        || (arch == Architecture::DeepSeek41 && stem == "engram_wkv")
    {
        Residency::Quantized
    } else if F32_STEMS.contains(&stem) {
        Residency::F32
    } else {
        Residency::Unknown
    }
}

/// Lower and upper bounds on the bytes tensor `info` occupies resident on
/// the device: its quantized size (f32 for a dtype of unknown size, which
/// no quantized kernel could hold anyway) and its f32 size.  A tensor of
/// known residency has equal bounds; an [`Residency::Unknown`] one spans
/// both.
fn resident_tensor_bounds(
    arch: Option<Architecture>,
    name: &str,
    info: &crate::gguf_ext::RawTensorInfo,
) -> (u64, u64) {
    let f32_bytes = (info.elem_count() as u64).saturating_mul(4);
    let quantized = crate::gguf_ext::type_size_bytes(info.dtype, info.elem_count())
        .map(|b| b as u64)
        .unwrap_or(f32_bytes);
    match residency(arch, name, info) {
        Residency::Host => (0, 0),
        Residency::Quantized => (quantized, quantized),
        Residency::F32 => (f32_bytes, f32_bytes),
        Residency::Unknown => (quantized, f32_bytes),
    }
}

/// Device footprint of a model's weights as the loader leaves them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeviceFootprint {
    /// Dense set, assuming every tensor of unknown residency stays
    /// quantized — the least the loader can allocate.  A load is refused
    /// only when this does not fit.
    pub(crate) dense_lower: u64,
    /// Dense set, assuming every tensor of unknown residency is f32 — the
    /// most the loader can allocate.  Drives the placement decision and the
    /// per-session caps.
    pub(crate) dense_upper: u64,
    /// Routed experts (always quantized).
    pub(crate) experts: u64,
}

/// Bytes the dense and routed-expert weights occupy on the device once the
/// loader has placed them, split like [`weight_ranges`].
///
/// Each tensor counts as [`resident_tensor_bounds`]; then every output-head
/// tensor the scan counted is replaced by the one head the loader ends up
/// resident with, which for a tied head is a second, quantized copy of the
/// embedding table (see [`resident_head_bytes`]).
pub(crate) fn device_weight_bytes(
    header: &crate::gguf_ext::GgufHeader,
    arch: Option<Architecture>,
) -> DeviceFootprint {
    let mut lower = 0u64;
    let mut upper = 0u64;
    let mut experts = 0u64;
    for (name, info) in &header.tensors {
        let (lo, hi) = resident_tensor_bounds(arch, name, info);
        if is_routed_expert(name) {
            experts = experts.saturating_add(lo);
        } else {
            lower = lower.saturating_add(lo);
            upper = upper.saturating_add(hi);
        }
    }
    let scanned = scanned_head_bytes(header);
    let resident = resident_head_bytes(header, arch);
    // Likewise the embedding table: a loader that probes several names
    // keeps the first it can read, so the tables the scan counted are
    // replaced by one — the smallest candidate in the lower bound, the
    // largest in the upper.
    let (emb_scanned_lo, emb_scanned_hi) = scanned_embedding_bounds(header, arch);
    let (emb_lo, emb_hi) = resident_embedding_bounds(header, arch);
    DeviceFootprint {
        dense_lower: lower
            .saturating_sub(scanned)
            .saturating_add(resident)
            .saturating_sub(emb_scanned_lo)
            .saturating_add(emb_lo),
        dense_upper: upper
            .saturating_sub(scanned)
            .saturating_add(resident)
            .saturating_sub(emb_scanned_hi)
            .saturating_add(emb_hi),
        experts,
    }
}

/// Summed [`resident_tensor_bounds`] of every embedding-table name `arch`'s
/// loader probes that is present — what the whole-header scan counted for
/// the table.
fn scanned_embedding_bounds(
    header: &crate::gguf_ext::GgufHeader,
    arch: Option<Architecture>,
) -> (u64, u64) {
    probed_embedding_names(arch)
        .iter()
        .filter_map(|n| header.tensors.get(*n).map(|info| (*n, info)))
        .fold((0u64, 0u64), |(lo, hi), (n, info)| {
            let (l, h) = resident_tensor_bounds(arch, n, info);
            (lo.saturating_add(l), hi.saturating_add(h))
        })
}

/// Bounds on the one embedding table the loader keeps resident: it takes
/// the first name it can read, which sizing cannot know, so the lower bound
/// is the smallest candidate present and the upper the largest.  Zero when
/// none is present.
fn resident_embedding_bounds(
    header: &crate::gguf_ext::GgufHeader,
    arch: Option<Architecture>,
) -> (u64, u64) {
    let mut lo: Option<u64> = None;
    let mut hi = 0u64;
    for n in probed_embedding_names(arch) {
        if let Some(info) = header.tensors.get(*n) {
            let (l, h) = resident_tensor_bounds(arch, n, info);
            lo = Some(lo.map_or(l, |x| x.min(l)));
            hi = hi.max(h);
        }
    }
    (lo.unwrap_or(0), hi)
}

/// Every tensor name any loader may read as the output head.  `output.weight`
/// is universal; candle's `lfm2` loader additionally probes the three
/// aliases, in this order, and stops at the first one it can read.
const OUTPUT_HEAD_NAMES: [&str; 4] = [
    "output.weight",
    "lm_head.weight",
    "model.output.weight",
    "model.lm_head.weight",
];

/// The head names `arch`'s loader probes, in its precedence order.
fn probed_head_names(arch: Option<Architecture>) -> &'static [&'static str] {
    probed_names(arch, &OUTPUT_HEAD_NAMES)
}

/// Bytes of the tensor `name` in the header as the output head holds it:
/// dequantized f32 when `f32` (OpenCL), otherwise its on-disk quantized
/// size (every loader keeps the head a quantized `QMatMul`).  Zero when
/// absent or of a dtype with no known size.
fn tensor_bytes(header: &crate::gguf_ext::GgufHeader, name: &str) -> u64 {
    let Some(info) = header.tensors.get(name) else {
        return 0;
    };
    crate::gguf_ext::type_size_bytes(info.dtype, info.elem_count())
        .map(|b| b as u64)
        .unwrap_or(0)
}

/// Bytes of every output-head-named tensor present in the file — what the
/// whole-file scans counted for the head, whether or not a loader reads it.
fn scanned_head_bytes(header: &crate::gguf_ext::GgufHeader) -> u64 {
    OUTPUT_HEAD_NAMES
        .iter()
        .fold(0u64, |acc, n| acc.saturating_add(tensor_bytes(header, n)))
}

/// Bytes of the output head the loader ends up with on the device, sized
/// conservatively.  A loader probes its head names in order and takes the
/// first it can *read* — a present-but-unreadable tensor (bad block count,
/// truncated data) is skipped, not fatal — and falls back to reloading the
/// embedding table (whichever of its embedding names it found) as a second,
/// quantized copy when none reads.  Sizing cannot know which read succeeds
/// without loading, so it takes the largest of the candidates the loader
/// could end up with: every probed head present, and every probed
/// embedding table present.  That never undercounts; at worst it overcounts
/// by the size difference between a head and the embedding (same shape,
/// possibly a different dtype).  Head tensors the loader never probes are
/// not candidates.
fn resident_head_bytes(header: &crate::gguf_ext::GgufHeader, arch: Option<Architecture>) -> u64 {
    probed_head_names(arch)
        .iter()
        .chain(probed_embedding_names(arch).iter())
        .map(|n| tensor_bytes(header, n))
        .max()
        .unwrap_or(0)
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

/// Select a byte-bounded resident subset without copying quantized weights.
/// Skip an untied input embedding: decode reads a few rows there, whereas
/// output projection and the other dense matrices are swept every token.
/// Merge page overlap so shared tensor boundary pages are charged once.
#[cfg(unix)]
fn bounded_weight_lock_ranges(
    header: &crate::gguf_ext::GgufHeader,
    map_len: usize,
    page: usize,
    budget: u64,
) -> Vec<ByteRange> {
    let mut tensors: Vec<_> = header.tensors.iter().filter_map(|(name, info)| {
        let off = header.tensor_data_offset.checked_add(info.offset)?;
        let off = usize::try_from(off).ok()?;
        (off < map_len).then_some((off, name.as_str()))
    }).collect();
    tensors.sort_by_key(|(off, _)| *off);
    let map_end = map_len.saturating_add(page - 1) / page * page;
    let untied = header.tensors.contains_key("output.weight");
    let mut remaining = budget / page as u64;
    let mut selected: Vec<ByteRange> = Vec::new();
    let mut covered_end = 0;
    for (i, &(off, name)) in tensors.iter().enumerate() {
        if remaining == 0 { break; }
        if is_routed_expert(name) || (untied && name == "token_embd.weight") {
            continue;
        }
        let end = tensors.get(i + 1).map_or(map_len, |t| t.0);
        if end <= off { continue; }
        let start = (off / page * page).max(covered_end);
        let end = (end.saturating_add(page - 1) / page * page).min(map_end);
        if end <= start { continue; }
        let pages = ((end - start) / page) as u64;
        let pages = pages.min(remaining) as usize;
        let len = pages * page;
        if let Some(last) = selected.last_mut().filter(|r| r.0 + r.1 == start) {
            last.1 += len;
        } else {
            selected.push((start, len));
        }
        remaining -= pages as u64;
        covered_end = start + len;
    }
    selected
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
    lock_budget: Option<u64>,
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

    let page = base_page_size();
    let locked_ranges = (mlock != MlockMode::Off).then_some(lock_budget).flatten().map(|budget| {
        let effective = crate::placement::available_ram_bytes()
            .map_or(budget, |free| budget.min(free.saturating_sub(LOW_MEM_FLOOR * 2)));
        if effective < budget {
            tracing::warn!("weight lock budget reduced from {} to {} MiB to reserve 3 GiB of available RAM", budget / (1 << 20), effective / (1 << 20));
        }
        bounded_weight_lock_ranges(header, mmap.len(), page, effective)
    });
    // A lock budget also bounds the load-time prefetch to the selected pages.
    let resident = locked_ranges.as_deref().unwrap_or(&dense);

    if prefetch {
        let mut failed = 0usize;
        for &(off, len) in resident {
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
                resident.len()
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
        let dense = resident;
        let required = aligned_range_bytes(dense, page, mmap.len());
        if required == 0 && lock_budget.is_some_and(|budget| budget > 0)
            && mlock == MlockMode::Required
        {
            return Err(JoshuaError::ModelLoad(
                "no weight pages fit the requested lock budget and available RAM headroom".into(),
            ));
        }
        if let Some(budget) = lock_budget {
            tracing::info!("bounded weight residency: {:.2} GiB selected within {:.2} GiB budget", required as f64 / 2f64.powi(30), budget as f64 / 2f64.powi(30));
        }
        let limit = memlock_limit_bytes();
        match mlock_decision(mlock, limit, required) {
            MlockDecision::Proceed => {
                let failed = mlock_ranges(mmap, dense, page);
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
    _lock_budget: Option<u64>,
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
        let rc =
            unsafe { libc::mlock(mmap.as_ptr().add(start).cast::<libc::c_void>(), end - start) };
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

/// A huge-page size the kernel offers, and how much of it can be obtained.
///
/// Plain numbers so [`select_huge_page`] is unit-testable anywhere; the values
/// come from [`read_huge_pools`] (Linux sysfs) at load time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HugePool {
    /// Page size in bytes (2 MiB, 64 MiB, 1 GiB, ...).
    pub page_bytes: usize,
    /// Pages in the persistent pool not currently allocated.
    pub free_bytes: u64,
    /// Spare bytes from the kernel's surplus (overcommit) allowance — headroom
    /// beyond `nr_hugepages` that the kernel will still hand out.  Zero when
    /// overcommit is off or already fully consumed.
    pub surplus_bytes: u64,
}

impl HugePool {
    /// Bytes this pool could plausibly back a mapping with: persistent free
    /// pages plus whatever surplus allowance is still unspent.  Surplus pages
    /// are not guaranteed — the kernel may decline under fragmentation — so this
    /// deliberately keeps the preflight permissive and leaves the final word to
    /// `mmap`, which still reports honestly if it fails.
    pub fn available_bytes(&self) -> u64 {
        self.free_bytes.saturating_add(self.surplus_bytes)
    }
}

/// `MmapOptions::huge` selector bits for a page size, or `None` when the size
/// cannot be asked for explicitly.
///
/// Linux encodes the requested size as `MAP_HUGE_SHIFT`: the page size's own
/// power of two (2 MiB -> 21, 64 MiB -> 26, 1 GiB -> 30).  Any power-of-two size
/// at least the 2 MiB minimum is therefore encodable, not just those two.  Sizes
/// that cannot be encoded must never be selected, because `None` means "kernel
/// default size" rather than "this size", and would round and preflight against
/// one size while mapping another.
fn huge_page_bits(page_bytes: usize) -> Option<u8> {
    if page_bytes < 2 * 1024 * 1024 || !page_bytes.is_power_of_two() {
        return None;
    }
    u8::try_from(page_bytes.trailing_zeros()).ok()
}

/// Bytes needed to cover `model_len` with whole pages of `page_bytes`.
///
/// Each candidate size rounds to **its own** page size.  Rounding the model once
/// up front at the *requested* size would reject a 2 MiB pool that is perfectly
/// sufficient: a file of 2 MiB + 1 byte needs two 2 MiB pages, not one gibibyte.
/// Returns `None` only on overflow.
fn huge_mapping_len(model_len: u64, page_bytes: usize) -> Option<u64> {
    let page = u64::try_from(page_bytes).ok()?;
    if page == 0 {
        return None;
    }
    model_len.div_ceil(page).checked_mul(page)
}

/// Choose which huge-page size should back a `model_len` mapping.
///
/// The requested size is honoured whenever its pool can actually hold the
/// mapping.  Otherwise the **largest encodable size that fits** is chosen, so a
/// `1gb` request degrades to 2 MiB pages rather than failing the load outright —
/// the weights are still huge-backed, just with more (and smaller) entries.
/// `None` means no offered size fits, and the caller must not attempt the
/// allocation.
///
/// Returns the chosen pool, whether it differs from the request, and the byte
/// length to map — always rounded to the chosen size, so the preflight, the
/// length and the `mmap` request can never disagree.
///
/// Pure: no I/O, no device, no `/proc`.
pub fn select_huge_page(
    pools: &[HugePool],
    requested_page_bytes: usize,
    model_len: u64,
) -> Option<(HugePool, bool, u64)> {
    let fits = |p: &HugePool| -> Option<u64> {
        huge_page_bits(p.page_bytes)?;
        let need = huge_mapping_len(model_len, p.page_bytes)?;
        (p.available_bytes() >= need).then_some(need)
    };
    // Exact request first: an available 1 GiB pool must never be silently
    // downgraded to something smaller.
    if let Some(exact) = pools.iter().find(|p| p.page_bytes == requested_page_bytes) {
        if let Some(need) = fits(exact) {
            return Some((*exact, false, need));
        }
    }
    let mut best: Option<(HugePool, u64)> = None;
    for p in pools.iter().filter(|p| p.page_bytes != requested_page_bytes) {
        if let Some(need) = fits(p) {
            if best.is_none() || p.page_bytes > best.unwrap().0.page_bytes {
                best = Some((*p, need));
            }
        }
    }
    best.map(|(p, need)| (p, p.page_bytes != requested_page_bytes, need))
}


/// Huge-page sizes the kernel offers, with the bytes free at each.
///
/// Empty means "cannot tell" (non-Linux, an unexpected sysfs layout, or no
/// huge-page support), in which case the caller attempts the requested size
/// unmodified rather than guessing.
#[cfg(target_os = "linux")]
pub fn read_huge_pools() -> Vec<HugePool> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir("/sys/kernel/mm/hugepages") else {
        return out;
    };
    for entry in entries.flatten() {
        // "hugepages-1048576kB" -> 1048576 KiB -> bytes.
        let name = entry.file_name().to_string_lossy().to_string();
        let Some(kb) = name
            .strip_prefix("hugepages-")
            .and_then(|s| s.strip_suffix("kB"))
            .and_then(|s| s.parse::<u64>().ok())
        else {
            continue;
        };
        let count = |f: &str| -> u64 {
            std::fs::read_to_string(entry.path().join(f))
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(0)
        };
        let page_bytes = kb * 1024;
        // Surplus capacity is the overcommit allowance minus the surplus already
        // handed out elsewhere; pages reserved for live mappings are not ours to
        // take.  Counting only free_hugepages would reject a mapping the kernel
        // could still satisfy from `nr_overcommit_hugepages`.
        let surplus = count("nr_overcommit_hugepages")
            .saturating_sub(count("surplus_hugepages"))
            .saturating_mul(page_bytes);
        out.push(HugePool {
            page_bytes: page_bytes as usize,
            free_bytes: count("free_hugepages").saturating_mul(page_bytes),
            surplus_bytes: surplus,
        });
    }
    out.sort_by_key(|p| std::cmp::Reverse(p.page_bytes));
    out
}

/// Non-Linux: no sysfs huge-page inventory.
#[cfg(not(target_os = "linux"))]
pub fn read_huge_pools() -> Vec<HugePool> {
    Vec::new()
}

/// Load the model into an anonymous mapping backed by explicit huge pages.
#[cfg(target_os = "linux")]
fn map_model_hugetlb(path: &Path, file: &File, size: PageSize) -> Result<Mmap> {
    let len = file.metadata()?.len() as usize;
    let (req_bits, req_len) = size.params();
    // The requirement for the *requested* size, for diagnostics only: each
    // candidate in select_huge_page is measured against its own page size.
    let need_at_request = (len as u64)
        .div_ceil(req_len as u64)
        .checked_mul(req_len as u64)
        .unwrap_or(u64::MAX);

    // Inspect the pool before attempting the mapping, the same way the memlock
    // path checks the limit before locking: an impossible request degrades with
    // one warning naming requested-vs-available instead of failing the load, and
    // a request that is available is never silently downgraded.  An empty
    // inventory means "cannot tell", so the request is honoured untouched.
    let pools = read_huge_pools();
    let req_avail = pools
        .iter()
        .find(|p| p.page_bytes == req_len)
        .map(HugePool::available_bytes)
        .unwrap_or(0);
    let gib = |b: u64| b as f64 / 2f64.powi(30);
    // `mapped_len` is always the chosen size's own rounding, and `page_bits`
    // always encodes that same size, so the three can never disagree.
    let (page_bits, page_len, mapped_len) = match select_huge_page(&pools, req_len, len as u64) {
        Some((pool, degraded, mapped_len)) => {
            if degraded {
                tracing::warn!(
                    "huge pages: cannot back a {:.2} GiB mapping with {}-byte pages ({:.2} GiB \
                     available); using {}-byte pages instead ({:.0} entries). To keep the \
                     requested size, enlarge the pool — 1 GiB pages must be reserved at boot \
                     (`default_hugepagesz=1G hugepagesz=1G hugepages=N`).",
                    gib(need_at_request),
                    req_len,
                    gib(req_avail),
                    pool.page_bytes,
                    mapped_len as f64 / pool.page_bytes as f64,
                );
            }
            (
                huge_page_bits(pool.page_bytes),
                pool.page_bytes,
                usize::try_from(mapped_len)
                    .map_err(|_| JoshuaError::ModelLoad("model too large for this platform".into()))?,
            )
        }
        None if pools.is_empty() => {
            let mapped = (len as u64)
                .div_ceil(req_len as u64)
                .checked_mul(req_len as u64)
                .ok_or_else(|| JoshuaError::ModelLoad("model too large for huge-page mapping".into()))?;
            (req_bits, req_len, mapped as usize)
        }
        None => {
            return Err(JoshuaError::ModelLoad(format!(
                "no huge-page pool can back a {:.2} GiB model mapping: requested {}-byte pages \
                 with {:.2} GiB available (free pool plus surplus allowance). Raise the pool \
                 (`sysctl vm.nr_hugepages=N`, sized for {:.0} pages), allow overcommit \
                 (`/sys/kernel/mm/hugepages/hugepages-{}kB/nr_overcommit_hugepages`), or load \
                 without --huge-pages.",
                gib(need_at_request),
                req_len,
                gib(req_avail),
                need_at_request as f64 / 2097152.0,
                req_len / 1024,
            )))
        }
    };

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
pub(crate) fn read_gguf_header(mmap: &[u8]) -> Result<gguf_file::Content> {
    let header = crate::gguf_ext::read_header(&mut Cursor::new(mmap))?;
    // Joshua's own loaders read tensors by their raw GGUF dtype id (the
    // i-quants, MXFP4, Q1_0, I32, …).  Candle's stock loaders only ever see
    // the projected `Content`, which serves the `raw_block` formats through
    // its external-tensor hook but nothing else, so any other
    // unsupported-dtype tensor would surface much later as a misleading
    // "cannot find tensor" for a required weight — or, worse, be silently
    // treated as absent by a loader probing an optional tensor, changing the
    // model without an error.  Refuse the load with the precise cause
    // instead.  (An undetectable architecture is left alone: the load
    // already fails with an accurate architecture error downstream.)
    if let Ok(arch) = Architecture::detect(&header.metadata) {
        if !arch.is_native() {
            let mut unsupported = header.unsupported_tensors();
            unsupported.retain(|(_, d)| !crate::raw_block::is_raw_block(*d));
            if !unsupported.is_empty() {
                let names = unsupported
                    .iter()
                    .map(|(n, d)| format!("`{n}` (GGUF dtype id {d})"))
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(crate::JoshuaError::ModelLoad(format!(
                    "GGUF header: model architecture '{}' contains {} tensor(s) in a GGUF \
                     dtype the engine cannot decode: {}. Only Joshua's native loaders read \
                     these dtypes; re-quantize this model to a supported format (F16, Q8_0, \
                     the K-quants, the i-quants, …) or use a loader that decodes them.",
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
    Some(
        ChatTemplate::new(source, bos, eos)
            .with_special_tokens(crate::template::special_token_strings(tokenizer)),
    )
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

    /// Sampling buffers retained across decode steps to avoid allocating vocabulary-sized
    /// vectors and sort workspaces for every generated token.
    #[derive(Default)]
    struct SamplingWorkspace {
        adjusted: Vec<f32>,
        probs: Vec<f32>,
        indexed: Vec<(usize, f32)>,
        sorted_idx: Vec<usize>,
        /// Sorted copy of the penalty window, for occurrence counts.
        window: Vec<u32>,
    }

    enum SamplingKind {
        Greedy(u32),
        Probs,
    }

    impl SamplingWorkspace {
        /// Draw the token at absolute position `pos` from `seed`.
        fn sample_at(
            &mut self,
            logits: &[f32],
            opts: &GenerationOptions,
            recent_tokens: &[u32],
            mask: Option<&SimpleVob>,
            seed: u64,
            pos: u64,
        ) -> Result<u32> {
            self.distribution(logits, opts, recent_tokens, mask, pos)?
                .sample_at(seed, pos)
        }

        fn distribution(
            &mut self,
            logits: &[f32],
            opts: &GenerationOptions,
            recent_tokens: &[u32],
            mask: Option<&SimpleVob>,
            pos: u64,
        ) -> Result<TokenDistRef<'_>> {
            if let Some(id) = logits.iter().position(|logit| !logit.is_finite()) {
                return Err(JoshuaError::Inference(format!(
                    "non-finite logits at position {pos}, first at id {id}"
                )));
            }
            Ok(match self.fill(logits, opts, recent_tokens, mask) {
                SamplingKind::Greedy(token) => TokenDistRef::Greedy(token),
                SamplingKind::Probs => TokenDistRef::Probs(&self.probs),
            })
        }

        fn fill(
            &mut self,
            logits: &[f32],
            opts: &GenerationOptions,
            recent_tokens: &[u32],
            mask: Option<&SimpleVob>,
        ) -> SamplingKind {
            if logits.is_empty() {
                return SamplingKind::Greedy(0);
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
            let reppen =
                if opts.temperature <= 0.0 && !legacy_reppen() {
                    1.0
                } else {
                    opts.repetition_penalty
                };
            // OpenAI presence/frequency penalties over the same window,
            // applied after the multiplicative penalty (llama.cpp's order).
            // They default to 0 and are honoured at every temperature.
            let (presence, frequency) = (opts.presence_penalty, opts.frequency_penalty);
            let additive = presence != 0.0 || frequency != 0.0;
            let logits = if (reppen != 1.0 || additive) && !recent_tokens.is_empty() {
                self.adjusted.clear();
                self.adjusted.extend_from_slice(logits);
                let v = &mut self.adjusted;
                if reppen != 1.0 {
                    for &token in recent_tokens {
                        if let Some(l) = v.get_mut(token as usize) {
                            if *l > 0.0 {
                                *l /= reppen;
                            } else {
                                *l *= reppen;
                            }
                        }
                    }
                }
                if additive {
                    // Count occurrences: sort a copy of the (≤ 64-token)
                    // window and penalise each distinct token once.
                    let window = &mut self.window;
                    window.clear();
                    window.extend_from_slice(recent_tokens);
                    window.sort_unstable();
                    for run in window.chunk_by(|a, b| a == b) {
                        if let Some(l) = v.get_mut(run[0] as usize) {
                            *l -= presence + frequency * run.len() as f32;
                        }
                    }
                }
                v.as_slice()
            } else {
                logits
            };

            // ── Greedy ────────────────────────────────────────────────────────────────
            if opts.temperature <= 0.0 {
                return SamplingKind::Greedy(argmax_allowed(logits, mask));
            }

            // ── Temperature scaling ───────────────────────────────────────────────────
            let inv_temp = 1.0_f32 / opts.temperature;
            // Subtract the largest legal logit for numerical stability before
            // exp.  The vocabulary-wide maximum is wrong under a grammar mask:
            // one dominant forbidden token would push every permitted logit
            // far below zero, their exponentials would underflow to zero, and
            // a sampled request would silently fall back to greedy decoding.
            let max_logit = match mask {
                Some(mask) => logits
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| mask.is_allowed(*i as u32))
                    .map(|(_, &l)| l)
                    .fold(f32::NEG_INFINITY, f32::max),
                None => logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            };
            let probs = &mut self.probs;
            probs.clear();
            probs.extend(logits.iter().map(|&l| ((l - max_logit) * inv_temp).exp()));

            // ── Grammar mask ──────────────────────────────────────────────────────────
            // Applied before top-k/min-p/top-p so those filters only ever
            // consider legal tokens (and never re-normalize an illegal one
            // back into the distribution).
            if let Some(mask) = mask {
                for (i, p) in probs.iter_mut().enumerate() {
                    if !mask.is_allowed(i as u32) {
                        *p = 0.0;
                    }
                }
            }

            // ── Top-k ─────────────────────────────────────────────────────────────────
            retain_top_k(probs, opts.top_k as usize, &mut self.indexed);

            // ── Min-p ─────────────────────────────────────────────────────────────────
            if opts.min_p > 0.0 {
                let max_p = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let threshold = max_p * opts.min_p;
                for p in probs.iter_mut() {
                    if *p < threshold {
                        *p = 0.0;
                    }
                }
            }

            // ── Top-p (nucleus) ───────────────────────────────────────────────────────
            if opts.top_p < 1.0 && opts.top_p > 0.0 {
                let sum: f32 = probs.iter().sum();
                if sum > 0.0 {
                    // Top-k/min-p already zeroed the tail.  Sort only the
                    // survivors: with the default top-k this is a few dozen
                    // entries instead of the whole vocabulary.  Token id
                    // breaks equal-probability ties so the cut is stable.
                    let sorted_idx = &mut self.sorted_idx;
                    sorted_idx.clear();
                    sorted_idx.extend((0..probs.len()).filter(|&i| probs[i] > 0.0));
                    sorted_idx.sort_unstable_by(|&a, &b| {
                        probs[b]
                            .partial_cmp(&probs[a])
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| a.cmp(&b))
                    });
                    let mut cumsum = 0.0_f32;
                    let mut cut_from = sorted_idx.len();
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
                return SamplingKind::Greedy(argmax_allowed(logits, mask));
            }

            for p in probs.iter_mut() {
                *p /= total;
            }
            SamplingKind::Probs
        }
    }

/// Keep the top-k probabilities without fully ordering the vocabulary.
/// A tied cutoff uses the original sort and input order to preserve its
/// exact choice of token IDs. Non-finite inputs also keep the old behavior.
fn retain_top_k(probs: &mut [f32], k: usize, indexed: &mut Vec<(usize, f32)>) {
    if k == 0 || k >= probs.len() {
        return;
    }
    indexed.clear();
    indexed.extend(probs.iter().copied().enumerate());
    let compare = |a: &(usize, f32), b: &(usize, f32)| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    };
    let needs_sort = if probs.iter().all(|p| p.is_finite()) {
        let (kept, boundary, _) = indexed.select_nth_unstable_by(k, compare);
        let tied = kept.iter().any(|(_, p)| *p == boundary.1);
        if tied {
            // Selection permutes the entries. Restore the original order
            // before the fallback: unstable sorting uses it to break ties.
            indexed.clear();
            indexed.extend(probs.iter().copied().enumerate());
        }
        tied
    } else {
        true
    };
    if needs_sort {
        indexed.sort_unstable_by(compare);
    }
    for &(idx, _) in indexed.iter().skip(k) {
        probs[idx] = 0.0;
    }
}

/// Index of the largest logit (0 for an empty vector).
/// `argmax`, restricted to the grammar-legal tokens when a mask is active.
fn argmax_allowed(logits: &[f32], mask: Option<&SimpleVob>) -> u32 {
    let Some(mask) = mask else {
        return argmax(logits);
    };
    let mut best: Option<(usize, f32)> = None;
    for (i, &logit) in logits.iter().enumerate() {
        if !mask.is_allowed(i as u32) {
            continue;
        }
        if best.is_none_or(|(_, b)| logit > b) {
            best = Some((i, logit));
        }
    }
    best.map(|(i, _)| i as u32).unwrap_or_else(|| argmax(logits))
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_finite_logits_are_an_error_not_last_token() {
        let mask = SimpleVob::from_slice(&[false, true, true]);
        let mut workspace = SamplingWorkspace::default();
        for temperature in [0.0, 0.8] {
            let options = GenerationOptions { temperature, ..Default::default() };
            for logits in [
                [f32::NAN; 3],
                [0.0, f32::NAN, 1.0],
                [0.0, 1.0, f32::NAN],
                [f32::INFINITY, 0.0, 1.0],
                [0.0, f32::NEG_INFINITY, 1.0],
            ] {
                let id = logits.iter().position(|value| !value.is_finite()).unwrap();
                for mask in [None, Some(&mask)] {
                    let error = workspace.sample_at(&logits, &options, &[], mask, 7, 42).unwrap_err();
                    assert!(matches!(error, JoshuaError::Inference(_)));
                    assert_eq!(error.to_string(), format!("Inference failed: non-finite logits at position 42, first at id {id}"));
                    assert!(workspace.distribution(&logits, &options, &[], mask, 42).is_err());
                }
            }
            assert!(workspace.sample_at(&[0.0, 3.0, 1.0], &options, &[], None, 7, 43).is_ok());
        }
        assert_eq!(argmax(&[1.0, 3.0, 3.0]), 2);
    }

    fn sorted_top_k(probs: &mut [f32], k: usize, indexed: &mut Vec<(usize, f32)>) {
        if k == 0 || k >= probs.len() {
            return;
        }
        indexed.clear();
        indexed.extend(probs.iter().copied().enumerate());
        indexed.sort_unstable_by(|(_, a), (_, b)| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });
        for &(i, _) in indexed.iter().skip(k) {
            probs[i] = 0.0;
        }
    }

    #[test]
    fn partial_top_k_matches_full_sort_including_tied_cutoffs() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(93);
        let mut fixtures = vec![
            vec![],
            vec![1.0],
            vec![0.0, -0.0, 0.0],
            vec![1.0, 0.5, 0.5, 0.5, 0.1],
            vec![f32::NAN, 1.0, 0.0],
            vec![f32::INFINITY, 1.0, f32::NEG_INFINITY],
        ];
        for len in [2, 7, 17, 64, 257] {
            fixtures.push((0..len).map(|_| rng.gen_range(0.0..1.0)).collect());
            fixtures.push((0..len).map(|_| rng.gen_range(0..4) as f32).collect());
            fixtures.push(vec![0.25; len]);
        }
        let mut selected = Vec::new();
        let mut sorted = Vec::new();
        for input in fixtures {
            for k in 0..=input.len() + 1 {
                let mut want = input.clone();
                let mut got = input.clone();
                sorted_top_k(&mut want, k, &mut sorted);
                retain_top_k(&mut got, k, &mut selected);
                assert_eq!(
                    got.iter().map(|p| p.to_bits()).collect::<Vec<_>>(),
                    want.iter().map(|p| p.to_bits()).collect::<Vec<_>>(),
                    "k={k} input={input:?}"
                );
            }
        }
    }

    #[test]
    #[ignore = "manual top-k benchmark; run release mode with --nocapture"]
    fn bench_top_k_selection() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        use std::{hint::black_box, time::Instant};
        let mut rng = StdRng::seed_from_u64(31);
        let input: Vec<f32> = (0..131_072).map(|_| rng.gen_range(0.0..1.0)).collect();
        let mut probs = input.clone();
        let mut indexed = Vec::new();
        let mut timings = [0.0; 2];
        for selection in [false, true] {
            for warmup in [true, false] {
                let repeats = if warmup { 2 } else { 100 };
                let start = Instant::now();
                for _ in 0..repeats {
                    probs.copy_from_slice(&input);
                    if selection {
                        retain_top_k(black_box(&mut probs), 40, &mut indexed);
                    } else {
                        sorted_top_k(black_box(&mut probs), 40, &mut indexed);
                    }
                    black_box(&probs);
                }
                if !warmup {
                    timings[selection as usize] = start.elapsed().as_secs_f64();
                }
            }
        }
        eprintln!(
            "top-k=40 vocab=131072: full sort {:.3}ms, selection {:.3}ms ({:.2}x)",
            timings[0] * 10.0,
            timings[1] * 10.0,
            timings[0] / timings[1]
        );
    }

    /// Former speculative path: one fresh workspace and an owned distribution
    /// per verified token. Kept here as a reference and benchmark baseline.
    fn fresh_distribution(
        logits: &[f32],
        opts: &GenerationOptions,
        recent: &[u32],
    ) -> crate::speculative::TokenDist {
        use crate::speculative::TokenDist;
        let mut workspace = SamplingWorkspace::default();
        match workspace.fill(logits, opts, recent, None) {
            SamplingKind::Greedy(t) => TokenDist::Greedy(t),
            SamplingKind::Probs => TokenDist::Probs(workspace.probs),
        }
    }

    /// A token's draw depends only on `(seed, position)`, so verifying drafts
    /// at earlier positions cannot shift the token emitted later.  That is
    /// what makes `--speculative` output-preserving for a fixed seed.
    #[test]
    fn position_keyed_sampling_ignores_verification_history() {
        let logits: Vec<f32> = (0..128)
            .map(|i| ((i * 37) % 101) as f32 / 20.0)
            .collect();
        let opts = GenerationOptions {
            temperature: 1.0,
            top_k: 20,
            top_p: 0.9,
            ..Default::default()
        };
        let recent = [3u32, 7];

        let mut ws = SamplingWorkspace::default();
        let plain: Vec<u32> = (0..8)
            .map(|pos| {
                ws.sample_at(&logits, &opts, &recent, None, 1234, pos)
                    .unwrap()
            })
            .collect();

        let mut ws = SamplingWorkspace::default();
        let mut with_verification = Vec::new();
        for pos in 0..8 {
            {
                let dist = ws.distribution(&logits, &opts, &recent, None, pos).unwrap();
                for draft in 0..3 {
                    let _ = dist.verify_at(draft, 1234, pos).unwrap();
                }
            }
            with_verification.push(
                ws.sample_at(&logits, &opts, &recent, None, 1234, pos)
                    .unwrap(),
            );
        }
        assert_eq!(plain, with_verification);
    }

    #[test]
    fn reused_distributions_match_fresh_after_size_and_option_changes() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut workspace = SamplingWorkspace::default();
        let mut rng = StdRng::seed_from_u64(42);
        for len in [257, 7, 1, 0, 19, 513, 7] {
            let logits: Vec<f32> = (0..len).map(|_| rng.gen_range(-4.0..4.0)).collect();
            for (temperature, top_k, top_p, min_p) in [
                (0.0, 0, 1.0, 0.0),
                (0.7, 5, 0.8, 0.05),
                (1.0, 0, 1.0, 0.0),
                (0.5, 0, 0.9, 0.2),
            ] {
                let opts = GenerationOptions {
                    temperature,
                    top_k,
                    top_p,
                    min_p,
                    repetition_penalty: 1.2,
                    ..Default::default()
                };
                let recent = [0, 3, 3, 999];
                let expected = fresh_distribution(&logits, &opts, &recent);
                let actual = workspace.distribution(&logits, &opts, &recent, None, 0).unwrap();
                match (actual, expected.as_ref()) {
                    (TokenDistRef::Greedy(a), TokenDistRef::Greedy(b)) => assert_eq!(a, b),
                    (TokenDistRef::Probs(a), TokenDistRef::Probs(b)) => assert_eq!(a, b),
                    _ => panic!("distribution kind changed"),
                }
                let mut a = StdRng::seed_from_u64(7);
                let mut b = a.clone();
                for draft in [0, 3, 999] {
                    assert_eq!(
                        actual.verify(draft, &mut a).unwrap(),
                        crate::speculative::verify_token(&expected, draft, &mut b).unwrap()
                    );
                }
                assert_eq!(
                    actual.sample(&mut a).unwrap(),
                    expected.sample(&mut b).unwrap()
                );
            }
        }
    }

    #[test]
    fn grammar_mask_filters_before_selection() {
        use crate::constraints::SimpleVob;

        let logits = [1.0_f32, 5.0, 3.0, 4.0];
        let opts = GenerationOptions {
            temperature: 0.0,
            ..Default::default()
        };
        // The raw argmax is token 1; the grammar allows 1 and 3, so greedy
        // must still agree with the raw argmax here.
        let mask = SimpleVob::from_slice(&[false, true, false, true]);
        let got = SamplingWorkspace::default()
            .sample_at(&logits, &opts, &[], Some(&mask), 0, 0)
            .unwrap();
        assert_eq!(got, 1);

        // Excluding token 1 leaves token 3, even though it is not the raw
        // argmax.
        let mask = SimpleVob::from_slice(&[false, false, false, true, false]);
        let got = SamplingWorkspace::default()
            .sample_at(&logits, &opts, &[], Some(&mask), 0, 0)
            .unwrap();
        assert_eq!(got, 3);

        // With probabilities, illegal tokens are zeroed before top-k/top-p.
        let opts = GenerationOptions {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            ..Default::default()
        };
        let mask = SimpleVob::from_slice(&[false, false, true, false]);
        let mut workspace = SamplingWorkspace::default();
        match workspace.distribution(&logits, &opts, &[], Some(&mask), 0).unwrap() {
            TokenDistRef::Probs(probs) => {
                assert_eq!(probs[2], 1.0);
                assert_eq!(probs[0], 0.0);
                assert_eq!(probs[1], 0.0);
                assert_eq!(probs[3], 0.0);
            }
            TokenDistRef::Greedy(_) => panic!("temperature 1 is not greedy"),
        }
    }

    #[test]
    fn a_dominant_forbidden_logit_does_not_starve_permitted_tokens() {
        use crate::constraints::SimpleVob;

        // Token 0 is a forbidden argmax far above the legal ones.  Scaling by
        // the vocabulary maximum would underflow 0.0 and 1.0 to zero, leaving
        // an empty distribution and a silent greedy fallback.
        let logits = [1000.0_f32, 0.0, 1.0];
        let mask = SimpleVob::from_slice(&[false, true, true]);
        let opts = GenerationOptions {
            temperature: 0.01,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            ..Default::default()
        };
        let mut workspace = SamplingWorkspace::default();
        match workspace.distribution(&logits, &opts, &[], Some(&mask), 0).unwrap() {
            TokenDistRef::Probs(probs) => {
                assert_eq!(probs[0], 0.0, "the forbidden token stays masked");
                assert!(probs[1] > 0.0, "a permitted token must survive: {probs:?}");
                assert_eq!(probs[2], 1.0, "the legal argmax is the mode");
            }
            TokenDistRef::Greedy(_) => panic!("a sampled request must not become greedy"),
        }
    }

    #[test]
    fn presence_and_frequency_penalties_shift_logits_per_occurrence() {
        let logits = [2.0_f32, 1.9, 1.0, -1.0];
        // Token 0 seen twice, token 3 once; token 1 never.
        let recent = [0, 3, 0];
        let greedy = |presence, frequency| {
            let opts = GenerationOptions {
                temperature: 0.0,
                presence_penalty: presence,
                frequency_penalty: frequency,
                ..Default::default()
            };
            match SamplingWorkspace::default().fill(&logits, &opts, &recent, None) {
                SamplingKind::Greedy(t) => t,
                SamplingKind::Probs => panic!("temperature 0 is greedy"),
            }
        };
        assert_eq!(greedy(0.0, 0.0), 0, "no penalty: raw argmax");
        // Presence: 2.0 - 0.2 = 1.8 < 1.9.
        assert_eq!(greedy(0.2, 0.0), 1);
        // Frequency counts occurrences: 2.0 - 2 * 0.06 = 1.88 < 1.9, while
        // a single occurrence (2.0 - 0.06) would not flip it.
        assert_eq!(greedy(0.0, 0.06), 1);
        assert_eq!(greedy(0.0, 0.04), 0);
        // Negative penalties encourage repetition.
        assert_eq!(greedy(-1.0, 0.0), 0);

        // Exact adjusted values under sampling (reppen off).
        let opts = GenerationOptions {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.5,
            frequency_penalty: 0.25,
            ..Default::default()
        };
        let mut ws = SamplingWorkspace::default();
        ws.fill(&logits, &opts, &recent, None);
        assert_eq!(
            ws.adjusted,
            vec![2.0 - 0.5 - 0.5, 1.9, 1.0, -1.0 - 0.5 - 0.25]
        );
    }

    #[test]
    fn zero_additive_penalties_leave_sampling_unchanged() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(5);
        let logits: Vec<f32> = (0..300).map(|_| rng.gen_range(-4.0..4.0)).collect();
        let recent = [1, 2, 2, 250];
        let base = GenerationOptions::default();
        let explicit = GenerationOptions {
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            ..GenerationOptions::default()
        };
        let a = fresh_distribution(&logits, &base, &recent);
        let b = fresh_distribution(&logits, &explicit, &recent);
        match (a.as_ref(), b.as_ref()) {
            (TokenDistRef::Probs(a), TokenDistRef::Probs(b)) => assert_eq!(a, b),
            _ => panic!("expected sampled distributions"),
        }
    }

    #[test]
    #[ignore = "manual workspace benchmark; run release mode with --nocapture"]
    fn bench_speculative_workspace_reuse() {
        use rand::{rngs::StdRng, SeedableRng};
        use std::{hint::black_box, time::Instant};
        let logits: Vec<f32> = (0..131_072)
            .map(|i| ((i * 7919) % 1000) as f32 / 100.0)
            .collect();
        for filtered in [false, true] {
            let opts = if filtered {
                GenerationOptions::default()
            } else {
                GenerationOptions {
                    top_k: 0,
                    top_p: 1.0,
                    min_p: 0.0,
                    ..Default::default()
                }
            };
            let mut workspace = SamplingWorkspace::default();
            let recent = [1, 17, 59];
            let mut timings = [0.0; 2];
            for reuse in [false, true] {
                let mut rng = StdRng::seed_from_u64(7);
                for warmup in [true, false] {
                    let repeats = if warmup { 2 } else { 100 };
                    let start = Instant::now();
                    for _ in 0..repeats {
                        let verdict = if reuse {
                            workspace
                                .distribution(black_box(&logits), &opts, &recent, None, 0)
                                .unwrap()
                                .verify(17, &mut rng)
                                .unwrap()
                        } else {
                            crate::speculative::verify_token(
                                &fresh_distribution(black_box(&logits), &opts, &recent),
                                17,
                                &mut rng,
                            )
                            .unwrap()
                        };
                        black_box(verdict);
                    }
                    if !warmup {
                        timings[reuse as usize] = start.elapsed().as_secs_f64();
                    }
                }
            }
            eprintln!(
                "filtered={filtered}: fresh {:.3}ms, reused {:.3}ms per token ({:.2}x)",
                timings[0] * 10.0,
                timings[1] * 10.0,
                timings[0] / timings[1]
            );
        }
    }

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

    #[test]
    fn deepseek41_engram_table_stays_out_of_device_weight_budget() {
        use crate::gguf_ext::{GgufHeader, RawTensorInfo};
        use std::collections::HashMap;
        let table = RawTensorInfo {
            dtype: 10,
            dims: vec![256, 1_000_000],
            offset: 0,
        };
        let projection = RawTensorInfo {
            dtype: 12,
            dims: vec![256, 1],
            offset: 0,
        };
        let arch = Some(Architecture::DeepSeek41);
        let host_name = "blk.7.engram_embd.weight";
        let header = GgufHeader {
            version: 3,
            metadata: HashMap::new(),
            tensors: HashMap::from([
                (host_name.into(), table.clone()),
                ("blk.7.engram_wkv.weight".into(), projection.clone()),
                ("blk.7.engram_q.weight".into(), projection.clone()),
                ("blk.7.engram_k.weight".into(), projection.clone()),
            ]),
            tensor_data_offset: 0,
        };
        let footprint = device_weight_bytes(&header, arch);
        // Packed wkv plus the two f32 norm vectors; no device table copy.
        assert_eq!(footprint.dense_lower, 144 + 2 * 256 * 4);
        assert_eq!(footprint.dense_upper, footprint.dense_lower);
        assert_eq!(footprint.experts, 0);
        assert_eq!(resident_tensor_bounds(arch, host_name, &table), (0, 0));
        // Restrict the exception to the architecture whose loader guarantees it.
        assert!(resident_tensor_bounds(Some(Architecture::DeepSeek4), host_name, &table).0 > 0);
        assert!(resident_tensor_bounds(None, host_name, &table).0 > 0);
        assert_eq!(
            resident_tensor_bounds(arch, "blk.7.engram_future.weight", &projection),
            (144, 1024)
        );
    }

    // ─── Hot-weight pinning ────────────────────────────────────────────────

    /// Device sizing follows what each loader leaves resident — quantized
    /// `QMatMul` matrices and routed experts, f32 for norms, biases, routers,
    /// dequantized tables and deepseek4's raw-dtype tensors, both bounds for
    /// a tensor no rule names — with the output head counted once as the
    /// loader keeps it.
    #[test]
    fn device_weight_bytes_follows_loader_residency() {
        use std::collections::HashMap;
        // Q4_K: 256 elements per 144-byte block.
        let q4k = |dims: Vec<usize>| crate::gguf_ext::RawTensorInfo {
            dtype: 12,
            dims,
            offset: 0,
        };
        let base = crate::gguf_ext::GgufHeader {
            version: 3,
            metadata: HashMap::new(),
            tensors: HashMap::from([
                ("token_embd.weight".to_string(), q4k(vec![256, 4])), // 1024 elems = 576 B
                ("output.weight".to_string(), q4k(vec![256, 4])),     // 576 B
                ("blk.0.attn_q.weight".to_string(), q4k(vec![256, 1])), // 256 elems = 144 B
                (
                    "blk.0.ffn_gate_exps.weight".to_string(),
                    q4k(vec![4, 64, 1]),
                ), // 144 B
                ("blk.0.ffn_up_exps.weight".to_string(), q4k(vec![4, 64, 1])),
            ]),
            tensor_data_offset: 0,
        };
        let q = Some(Architecture::Qwen3Moe);
        let llama = Some(Architecture::Llama);
        let lfm2 = Some(Architecture::Lfm2);
        let ds4 = Some(Architecture::DeepSeek4);
        const EMBD_F32: u64 = 1024 * 4;
        let fp = |h: &crate::gguf_ext::GgufHeader, a| device_weight_bytes(h, a);
        let known = |lower: u64, experts: u64| DeviceFootprint {
            dense_lower: lower,
            dense_upper: lower,
            experts,
        };

        // joshua-native loader, CUDA/Metal: everything known and quantized.
        assert_eq!(fp(&base, q), known(576 + 576 + 144, 2 * 144));
        // Stock loader: the embedding table is f32, the head a quantized QMatMul.
        assert_eq!(fp(&base, llama), known(EMBD_F32 + 576 + 144, 2 * 144));
        assert_eq!(scanned_head_bytes(&base), 576);
        assert_eq!(resident_head_bytes(&base, q), 576);

        // Tied head (no head tensor): the embedding is resident twice — as
        // the table (quantized for joshua-native, f32 for stock) and as a
        // quantized head copy.
        let mut tied = base.clone();
        tied.tensors.remove("output.weight");
        assert_eq!(scanned_head_bytes(&tied), 0);
        assert_eq!(
            resident_head_bytes(&tied, q),
            576,
            "tied head is the quantized table"
        );
        assert_eq!(fp(&tied, q).dense_lower, 576 + 144 + 576);
        assert_eq!(fp(&tied, llama), known(EMBD_F32 + 144 + 576, 2 * 144));
        // A model candle cannot load: nothing is known, the bounds span
        // quantized..f32 for every tensor (and the check never runs).
        let unknown = fp(&tied, None);
        assert_eq!(unknown.dense_lower, 576 + 144 + 576);
        assert_eq!(unknown.dense_upper, EMBD_F32 + 1024 + 576);

        // A float-dtype embedding is f32 on an accelerator for every loader.
        let mut f16_embd = tied.clone();
        f16_embd.tensors.insert(
            "token_embd.weight".to_string(),
            crate::gguf_ext::RawTensorInfo {
                dtype: 1,
                dims: vec![256, 4],
                offset: 0,
            },
        );
        // Table f32 + head copy at its on-disk f16 size (2 B/elem).
        assert_eq!(fp(&f16_embd, q), known(EMBD_F32 + 144 + 1024 * 2, 2 * 144));

        // deepseek4 decodes a raw-dtype (IQ2_XXS = 16) dense tensor to f32 on
        // the device; its compressor projections stay quantized, the
        // position vector is f32.
        let mut ds4h = base.clone();
        ds4h.tensors.insert(
            "blk.0.attn_output.weight".to_string(),
            crate::gguf_ext::RawTensorInfo {
                dtype: 16,
                dims: vec![256, 1],
                offset: 0,
            },
        );
        ds4h.tensors.insert(
            "blk.0.attn_compressor_kv.weight".to_string(),
            q4k(vec![256, 1]),
        );
        ds4h.tensors.insert(
            "blk.0.attn_compressor_ape.weight".to_string(),
            q4k(vec![256, 1]),
        );
        ds4h.tensors
            .insert("output_hc_fn.weight".to_string(), q4k(vec![256, 1]));
        assert_eq!(
            fp(&ds4h, ds4),
            known(576 + 576 + 144 + 1024 + 144 + 1024 + 144, 2 * 144),
            "IQ2_XXS dense tensor and ape vector as f32, projections quantized"
        );

        // Everything known to be dequantized is f32 in both bounds: norms,
        // biases, routers, deepseek2's split-KV halves, deepseek4's
        // hyper-connection vectors.  A tensor no rule names spans
        // quantized..f32.  Projections stay quantized.
        let mut misc = base.clone();
        for n in [
            "blk.0.attn_norm.weight",
            "blk.0.attn_q.bias",
            "blk.0.ffn_gate_inp.weight",
            "blk.0.attn_k_b.weight",
            "blk.0.hc_attn_base.weight",
        ] {
            misc.tensors.insert(n.to_string(), q4k(vec![256, 1])); // 144 B quantized, 1024 B f32
        }
        misc.tensors.insert(
            "blk.0.some_future_tensor.weight".to_string(),
            q4k(vec![256, 1]),
        );
        misc.tensors
            .insert("blk.0.attn_kv_b.weight".to_string(), q4k(vec![256, 1]));
        misc.tensors
            .insert("blk.0.ffn_gate_shexp.weight".to_string(), q4k(vec![256, 1]));
        let m = fp(&misc, Some(Architecture::DeepSeek2));
        assert_eq!(
            m.dense_lower,
            576 + 576 + 144 + 5 * 1024 + 144 + 2 * 144,
            "unknown tensor quantized in the lower bound"
        );
        assert_eq!(
            m.dense_upper,
            576 + 576 + 144 + 5 * 1024 + 1024 + 2 * 144,
            "unknown tensor f32 in the upper bound"
        );
        assert_eq!(tensor_stem("blk.12.attn_q.weight"), "attn_q");
        assert_eq!(tensor_stem("output.weight"), "output");
        assert_eq!(tensor_stem("blk.0.attn_q.bias"), "attn_q.bias");
        assert_eq!(tensor_stem("blk.0.ffn_up.7.weight"), "ffn_up");
        assert_eq!(
            tensor_stem("blk.0.feed_forward.w1.weight"),
            "feed_forward.w1"
        );

        // Mixtral's per-expert matrices (`ffn_gate.<i>`) and lfm2's aliases
        // are QMatMuls: quantized in both bounds, so a fitting model is never
        // refused for them.
        let mut mixtral = base.clone();
        for n in [
            "blk.0.ffn_gate.3.weight",
            "blk.0.ffn_up.3.weight",
            "blk.0.ffn_down.3.weight",
        ] {
            mixtral.tensors.insert(n.to_string(), q4k(vec![256, 1]));
        }
        assert_eq!(
            fp(&mixtral, llama),
            known(EMBD_F32 + 576 + 144 + 3 * 144, 2 * 144)
        );
        let mut lfm = base.clone();
        for n in [
            "blk.0.feed_forward.w1.weight",
            "blk.0.mlp.gate_proj.weight",
            "blk.0.shortconv.in_proj.weight",
            "blk.0.self_attn.out_proj.weight",
        ] {
            lfm.tensors.insert(n.to_string(), q4k(vec![256, 1]));
        }
        lfm.tensors
            .insert("blk.0.shortconv.conv.weight".to_string(), q4k(vec![256, 1])); // dequantized kernel
        assert_eq!(
            fp(&lfm, lfm2),
            known(EMBD_F32 + 576 + 144 + 4 * 144 + 1024, 2 * 144)
        );

        // LFM2: an alias head is a head (counted once, no tied copy); for
        // any other loader it is dead weight and the embedding is tied.
        for alias in [
            "lm_head.weight",
            "model.output.weight",
            "model.lm_head.weight",
        ] {
            let mut aliased = tied.clone();
            aliased.tensors.insert(alias.to_string(), q4k(vec![256, 4]));
            assert_eq!(scanned_head_bytes(&aliased), 576, "{alias}: scanned");
            assert_eq!(
                fp(&aliased, lfm2).dense_lower,
                EMBD_F32 + 144 + 576,
                "{alias}: LFM2 head"
            );
            assert_eq!(
                fp(&aliased, q).dense_lower,
                576 + 144 + 576,
                "{alias}: alias dropped, tied"
            );
        }

        // LFM2 embedding aliases: the table under `model.embed_tokens.weight`
        // is dequantized like any stock embedding and, with no head, also
        // the tied head's fallback.
        let mut lfm2_embd = tied.clone();
        lfm2_embd.tensors.remove("token_embd.weight");
        lfm2_embd
            .tensors
            .insert("model.embed_tokens.weight".to_string(), q4k(vec![256, 4]));
        assert_eq!(
            resident_head_bytes(&lfm2_embd, lfm2),
            576,
            "alias table is the tied head"
        );
        assert_eq!(fp(&lfm2_embd, lfm2).dense_lower, EMBD_F32 + 144 + 576);
        // Another loader never probes that name: a tensor no rule names
        // (quantized..f32), and no head.
        assert_eq!(resident_head_bytes(&lfm2_embd, q), 0);
        assert_eq!(fp(&lfm2_embd, q).dense_lower, 576 + 144);
        assert_eq!(fp(&lfm2_embd, q).dense_upper, EMBD_F32 + 144);

        // Two embedding names present: the lfm2 loader keeps the first it
        // can read, so exactly one table is resident — the smaller in the
        // lower bound, the larger in the upper — plus the tied head copy.
        let mut two_tables = tied.clone();
        two_tables
            .tensors
            .insert("model.embed_tokens.weight".to_string(), q4k(vec![64, 4])); // 256 elems
        let t = fp(&two_tables, lfm2);
        assert_eq!(
            t.dense_lower,
            256 * 4 + 144 + 576,
            "smaller table, f32, plus the head copy"
        );
        assert_eq!(
            t.dense_upper,
            EMBD_F32 + 144 + 576,
            "larger table, f32, plus the head copy"
        );
        // Another loader probes only token_embd.weight: the alias is an
        // unknown tensor, the table is counted once as before.
        let t = fp(&two_tables, q);
        assert_eq!(t.dense_lower, 576 + 144 + 576 + 144);
        assert_eq!(t.dense_upper, 576 + 144 + 576 + 256 * 4);

        // The MoE router is per loader: Mixtral (llama) keeps it a quantized
        // QMatMul, the joshua-native loaders read it as f32.
        let mut router = base.clone();
        router
            .tensors
            .insert("blk.0.ffn_gate_inp.weight".to_string(), q4k(vec![256, 1]));
        assert_eq!(
            fp(&router, llama),
            known(EMBD_F32 + 576 + 144 + 144, 2 * 144)
        );
        assert_eq!(fp(&router, q), known(576 + 576 + 144 + 1024, 2 * 144));

        // Several head names present: only one ends up resident.  Sizing
        // cannot know which read succeeds, so it keeps the largest candidate
        // — the big alias over a small (possibly unreadable) output.weight.
        let mut two = tied.clone();
        two.tensors
            .insert("output.weight".to_string(), q4k(vec![64, 4])); // 256 elems = 144 B
        two.tensors
            .insert("lm_head.weight".to_string(), q4k(vec![256, 4]));
        assert_eq!(scanned_head_bytes(&two), 144 + 576);
        assert_eq!(
            resident_head_bytes(&two, lfm2),
            576,
            "largest LFM2 candidate"
        );
        assert_eq!(fp(&two, lfm2).dense_lower, EMBD_F32 + 144 + 576);
        // Another loader: candidates are output.weight and the table fallback.
        assert_eq!(resident_head_bytes(&two, q), 576);
        assert_eq!(fp(&two, q).dense_lower, 576 + 144 + 576);
    }

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
                response,
                oracle,
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
        assert_eq!(
            dense,
            vec![(4096, 100), (4096 + 5 * 1024 * 1024 + 100, 100)]
        );
        // gate_exps 100..(5MiB+100) and up_exps (5MiB+200)..(5MiB+300) are
        // separated by only the 100-byte norm tensor — a sub-page gap, so the
        // two expert ranges merge into one.
        assert_eq!(experts, vec![(4196, 5243080)]);
    }

    /// A huge-page pool: page size in MiB, free megabytes, no surplus.
    fn pool(page_mib: usize, free_mib: u64) -> HugePool {
        HugePool {
            page_bytes: page_mib * 1024 * 1024,
            free_bytes: free_mib * 1024 * 1024,
            surplus_bytes: 0,
        }
    }

    /// The 1 GiB and 2 MiB sizes, in bytes, for readable call sites.
    const GIB_BYTES: usize = 1024 * 1024 * 1024;
    const MIB2_BYTES: usize = 2 * 1024 * 1024;
    fn gib(n: u64) -> u64 { n * 1024 * 1024 * 1024 }

    /// The pool preflight must honour an available 1 GiB request exactly — a
    /// real 1 GiB mapping is the point of asking for it.
    #[test]
    fn select_huge_page_honours_an_available_request() {
        let pools = [pool(1024, 60 * 1024), pool(2, 60 * 1024)];
        let (chosen, degraded, mapped_len) =
            select_huge_page(&pools, GIB_BYTES, gib(58) + 1).unwrap();
        assert_eq!(chosen.page_bytes, GIB_BYTES);
        assert!(!degraded, "an available 1 GiB pool must never be silently downgraded");
        // 58 GiB + 1 byte needs 59 whole 1 GiB pages, not 58.
        assert_eq!(mapped_len, gib(59), "length rounded to the chosen size");
    }

    /// 1 GiB pages exist but are fragmentation-capped below the model: the load
    /// must fall back to 2 MiB rather than failing outright.
    #[test]
    fn select_huge_page_degrades_when_the_requested_size_cannot_hold_the_model() {
        let pools = [pool(1024, 31 * 1024), pool(2, 70 * 1024)];
        let model = 62_394_667_168u64; // the reap-150b GGUF, as measured
        let (chosen, degraded, mapped_len) = select_huge_page(&pools, GIB_BYTES, model).unwrap();
        assert_eq!(chosen.page_bytes, MIB2_BYTES, "must fall back to the size that fits");
        assert!(degraded, "the caller has to warn that the request changed");
        assert_eq!(mapped_len % MIB2_BYTES as u64, 0, "length is a whole number of chosen pages");
        assert!(mapped_len >= model);
    }

    /// Regression: each candidate must be measured against **its own** page
    /// size.  Rounding once at the requested 1 GiB size rejected a 2 MiB pool
    /// that could comfortably hold the file (2 MiB + 1 byte needs two 2 MiB
    /// pages, not one gibibyte).
    #[test]
    fn select_huge_page_rounds_each_candidate_to_its_own_page_size() {
        let pools = [pool(1024, 0), pool(2, 4)];
        let model = 2 * 1024 * 1024 + 1;
        let (chosen, degraded, mapped_len) =
            select_huge_page(&pools, GIB_BYTES, model).unwrap();
        assert_eq!(chosen.page_bytes, MIB2_BYTES);
        assert!(degraded);
        assert_eq!(mapped_len, 4 * 1024 * 1024, "two 2 MiB pages, not one 1 GiB page");
    }

    /// Regression: Linux hands out surplus pages beyond `nr_hugepages` when
    /// overcommit allows it.  Counting only the persistent free pool rejected
    /// mappings the kernel would have satisfied.
    #[test]
    fn select_huge_page_counts_surplus_allowance_as_available() {
        let empty_pool_with_headroom = HugePool {
            page_bytes: MIB2_BYTES,
            free_bytes: 0,
            surplus_bytes: 4 * 1024 * 1024,
        };
        let (chosen, _, mapped_len) =
            select_huge_page(&[empty_pool_with_headroom], MIB2_BYTES, 2 * 1024 * 1024 + 1)
                .expect("surplus allowance must be usable");
        assert_eq!(chosen.page_bytes, MIB2_BYTES);
        assert_eq!(mapped_len, 4 * 1024 * 1024);
    }

    /// Surplus already consumed by other mappings is not ours to count.
    #[test]
    fn select_huge_page_does_not_double_count_consumed_surplus() {
        let p = HugePool {
            page_bytes: MIB2_BYTES,
            free_bytes: 0,
            surplus_bytes: 0, // allowance fully spent already
        };
        assert!(select_huge_page(&[p], MIB2_BYTES, gib(1)).is_none());
    }

    /// Regression: a size whose selector bits cannot be computed must never be
    /// chosen — `None` means "kernel default size", so preflight and mapping
    /// would disagree about which pages are actually being asked for.
    #[test]
    fn select_huge_page_refuses_sizes_it_cannot_encode() {
        // 3 MiB is not a power of two: unencodable.  1 MiB is below the minimum.
        let bad = [pool(3, 70 * 1024), pool(1, 70 * 1024)];
        assert!(select_huge_page(&bad, GIB_BYTES, gib(8)).is_none());
        // 64 MiB *is* encodable and must be preferred when it is the largest fit.
        let ok = [pool(64, 70 * 1024), pool(2, 70 * 1024)];
        let (chosen, degraded, _) = select_huge_page(&ok, GIB_BYTES, gib(8)).unwrap();
        assert_eq!(chosen.page_bytes, 64 * 1024 * 1024);
        assert!(degraded);
    }

    /// Nothing fits: report it instead of attempting a mapping that must fail.
    #[test]
    fn select_huge_page_reports_when_no_size_fits() {
        let pools = [pool(1024, 4 * 1024), pool(2, 8 * 1024)];
        assert!(select_huge_page(&pools, GIB_BYTES, gib(59)).is_none());
    }

    /// 4 KiB base pages are not huge pages and must never be chosen; an empty
    /// inventory means "cannot tell", not "no way".
    #[test]
    fn select_huge_page_ignores_base_pages_and_empty_inventories() {
        let base_only = [HugePool {
            page_bytes: 4 * 1024,
            free_bytes: gib(200),
            surplus_bytes: 0,
        }];
        assert!(select_huge_page(&base_only, MIB2_BYTES, gib(1)).is_none());
        assert!(
            select_huge_page(&[], MIB2_BYTES, gib(10)).is_none(),
            "no inventory must read as unknown, not as impossible"
        );
    }

    #[test]
    fn huge_page_bits_encodes_every_power_of_two_size_at_least_2mib() {
        assert_eq!(huge_page_bits(2 * 1024 * 1024), Some(21));
        assert_eq!(huge_page_bits(64 * 1024 * 1024), Some(26));
        assert_eq!(huge_page_bits(1024 * 1024 * 1024), Some(30));
        assert_eq!(huge_page_bits(4 * 1024), None, "below the huge-page minimum");
        assert_eq!(huge_page_bits(3 * 1024 * 1024), None, "not a power of two");
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
        apply_hot_weight_pinning(&mmap, &empty, true, MlockMode::On, None).expect("no tensors is fine");

        // Real split: prefetch + lock, plus the random hint on experts.
        let header = header_with_tensors(
            0,
            &[
                ("token_embd.weight", 0),
                ("blk.0.ffn_gate_exps.weight", 4096),
            ],
        );
        apply_hot_weight_pinning(&mmap, &header, true, MlockMode::On, None)
            .expect("advice/mlock is best effort");

        let _ = std::fs::remove_file(&path);
    }

    #[cfg(unix)]
    #[test]
    fn bounded_weight_residency_skips_sparse_weights_and_charges_whole_pages() {
        let page = 4096;
        let header = header_with_tensors(0, &[
            ("token_embd.weight", 0),
            ("output.weight", page as u64),
            ("blk.0.attn_q.weight", (2 * page + 16) as u64),
            ("blk.0.ffn_gate_inp.weight", (3 * page + 16) as u64),
            ("blk.0.ffn_gate_exps.weight", (4 * page + 16) as u64),
        ]);
        let map_len = 6 * page + 7;
        for budget in [0, 1, 4095, 4096, 8191, 8192, 8193, 16384, u64::MAX] {
            let ranges = bounded_weight_lock_ranges(&header, map_len, page, budget);
            let bytes = aligned_range_bytes(&ranges, page, map_len);
            assert!(bytes <= budget);
            assert!(ranges.iter().all(|&(off, len)| {
                off >= page && off + len <= 5 * page && off % page == 0 && len % page == 0
            }));
        }
        assert_eq!(bounded_weight_lock_ranges(&header, map_len, page, 8193), vec![(page, 2 * page)]);
        assert_eq!(bounded_weight_lock_ranges(&header, map_len, page, u64::MAX), vec![(page, 4 * page)]);
    }

    #[cfg(unix)]
    #[test]
    fn bounded_weight_residency_keeps_a_tied_output_embedding() {
        let header = header_with_tensors(0, &[
            ("token_embd.weight", 0), ("blk.0.attn_q.weight", 8192),
        ]);
        assert_eq!(bounded_weight_lock_ranges(&header, 16384, 4096, 4096), vec![(0, 4096)]);
    }

    #[cfg(unix)]
    #[test]
    fn bounded_weight_residency_accounts_for_the_final_partial_page() {
        let header = header_with_tensors(0, &[("output.weight", 4097)]);
        let ranges = bounded_weight_lock_ranges(&header, 8193, 4096, 8192);
        assert_eq!(ranges, vec![(4096, 8192)]);
        assert_eq!(aligned_range_bytes(&ranges, 4096, 8193), 8192);
        assert_eq!(bounded_weight_lock_ranges(&header, 8193, 4096, 4096), vec![(4096, 4096)]);
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
        let gate = AdmissionGate::new();
        // Fill to the cap of 2.
        let a = InFlightGuard::acquire(&gate, 2, Duration::ZERO).expect("first permit");
        let b = InFlightGuard::acquire(&gate, 2, Duration::ZERO).expect("second permit");
        // Third is rejected as Overloaded, and the count is not left inflated.
        match InFlightGuard::acquire(&gate, 2, Duration::ZERO) {
            Err(JoshuaError::Overloaded(_)) => {}
            Err(e) => panic!("expected Overloaded, got {e:?}"),
            Ok(_) => panic!("expected Overloaded, got a permit"),
        }
        assert_eq!(gate.active(), 2);
        // Dropping a permit frees a slot for the next request.
        drop(a);
        let _c =
            InFlightGuard::acquire(&gate, 2, Duration::ZERO).expect("permit after release");
        drop(b);
        drop(_c);
        assert_eq!(gate.active(), 0);
    }

    #[test]
    fn queued_request_waits_for_a_permit() {
        let gate = Arc::new(AdmissionGate::new());
        let held = InFlightGuard::acquire(&gate, 1, Duration::ZERO).expect("first permit");
        let waiter = {
            let gate = Arc::clone(&gate);
            std::thread::spawn(move || {
                let started = Instant::now();
                let permit = InFlightGuard::acquire(&gate, 1, Duration::from_secs(2))
                    .expect("queued permit");
                assert_eq!(gate.active(), 1);
                drop(permit);
                started.elapsed()
            })
        };
        std::thread::sleep(Duration::from_millis(40));
        drop(held);
        let waited = waiter.join().expect("waiter thread");
        assert!(waited >= Duration::from_millis(30), "waited only {waited:?}");
        assert_eq!(gate.active(), 0);
    }

    #[test]
    fn admission_queue_is_bounded() {
        let gate = Arc::new(AdmissionGate::new());
        let held = InFlightGuard::acquire(&gate, 1, Duration::ZERO).expect("first permit");
        let waiter = {
            let gate = Arc::clone(&gate);
            std::thread::spawn(move || {
                let permit = InFlightGuard::acquire_with(
                    &gate,
                    1,
                    1,
                    Duration::from_secs(2),
                    None,
                )
                .expect("queued permit");
                drop(permit);
            })
        };
        // Wait until the one queue slot is occupied.
        for _ in 0..200 {
            if gate.waiting() == 1 {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(gate.waiting(), 1);
        match InFlightGuard::acquire_with(&gate, 1, 1, Duration::from_millis(10), None) {
            Err(JoshuaError::Overloaded(msg)) => {
                assert!(msg.contains("queue is full"), "got: {msg}");
            }
            Err(e) => panic!("expected Overloaded, got {e:?}"),
            Ok(_) => panic!("expected the full queue to reject"),
        }
        assert_eq!(gate.active(), 1, "rejected waiter must not take a slot");
        drop(held);
        waiter.join().expect("waiter thread");
        assert_eq!(gate.active(), 0);
        assert_eq!(gate.waiting(), 0);
    }

    #[test]
    fn cancelled_waiter_leaves_the_queue() {
        let gate = Arc::new(AdmissionGate::new());
        let held = InFlightGuard::acquire(&gate, 1, Duration::ZERO).expect("first permit");
        let cancel = Arc::new(AtomicBool::new(false));
        let waiter = {
            let gate = Arc::clone(&gate);
            let cancel = Arc::clone(&cancel);
            std::thread::spawn(move || {
                InFlightGuard::acquire_with(
                    &gate,
                    1,
                    4,
                    Duration::from_secs(5),
                    Some(&cancel),
                )
                .map(|_| ())
            })
        };
        for _ in 0..200 {
            if gate.waiting() == 1 {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(gate.waiting(), 1);
        cancel.store(true, Ordering::Relaxed);
        match waiter.join().expect("waiter thread") {
            Err(JoshuaError::Cancelled) => {}
            other => panic!("expected Cancelled, got {other:?}"),
        }
        assert_eq!(gate.waiting(), 0);
        assert_eq!(gate.active(), 1);
        drop(held);
        assert_eq!(gate.active(), 0);
    }

    #[test]
    fn queued_request_times_out() {
        let gate = AdmissionGate::new();
        let _held = InFlightGuard::acquire(&gate, 1, Duration::ZERO).expect("first permit");
        let started = Instant::now();
        match InFlightGuard::acquire(&gate, 1, Duration::from_millis(30)) {
            Err(JoshuaError::Overloaded(_)) => {}
            Err(e) => panic!("expected Overloaded, got {e:?}"),
            Ok(_) => panic!("expected Overloaded, got a permit"),
        }
        assert!(started.elapsed() >= Duration::from_millis(20));
        // The rejected waiter left the count alone.
        assert_eq!(gate.active(), 1);
    }

    #[test]
    fn stream_gate_withholds_a_stop_sequence_until_it_is_resolved() {
        let seen = std::cell::RefCell::new(String::new());
        let mut sink = |text: &str| seen.borrow_mut().push_str(text);
        let stops = vec!["END".to_string()];
        let mut gate = StreamGate::new(&stops);

        // "EN" is a prefix of the stop sequence, so it stays withheld.
        gate.push("hello EN", false, Some(&mut sink));
        assert_eq!(&*seen.borrow(), "hello");
        // The complete sequence trims the response to "hello "; nothing
        // already sent is retracted and no stop text leaks.
        gate.push("hello ", true, Some(&mut sink));
        gate.finish("hello ", Some(&mut sink));
        assert_eq!(&*seen.borrow(), "hello ");
    }

    #[test]
    fn stream_gate_flushes_without_stop_sequences() {
        let seen = std::cell::RefCell::new(String::new());
        let mut sink = |text: &str| seen.borrow_mut().push_str(text);
        let mut gate = StreamGate::new(&[]);
        gate.push("hel", false, Some(&mut sink));
        gate.push("lo", false, Some(&mut sink));
        gate.finish("hello", Some(&mut sink));
        assert_eq!(&*seen.borrow(), "hello");
    }

    #[test]
    fn check_stop_sequences_reports_the_matched_sequence() {
        let mut response = "hello END".to_string();
        let stops = vec!["STOP".to_string(), "END".to_string()];
        assert_eq!(
            Engine::check_stop_sequences(&mut response, &stops).as_deref(),
            Some("END")
        );
        assert_eq!(response, "hello ");

        let mut response = "nothing here".to_string();
        assert!(Engine::check_stop_sequences(&mut response, &stops).is_none());
        assert_eq!(response, "nothing here");
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
