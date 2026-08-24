# FreeToken — how it works, and what joshua can borrow

Paper: *FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution*
(arXiv 2608.16157, Yang/Fan/Pan/Xi et al. — Berkeley/UT Austin; Stoica, Zaharia, Song Han).
Open source: https://github.com/FlashML-org/FreeToken

Headline results: 1.3–2.3× decode throughput vs llama.cpp / KTransformers / Ollama /
MoE-Infinity on real agentic workloads; 22–25 tok/s on DeepSeek-V4-Flash (284B, MXFP4)
from a single RTX 5090; 35B-A3B at 39 tok/s on an 8 GB laptop GPU; tail TTFT < 44 s
where every baseline exceeds 150 s somewhere.

## 1. What FreeToken is

A **GPU-centric** edge serving system (vLLM/SGLang-style substrate, CUDA graphs) built
around a two-level expert-memory hierarchy:

- **Host RAM holds the complete routed-expert pool** (the source of truth).
- **The GPU keeps non-expert weights** (attention, norms, routers, embeddings, shared
  experts) plus **one elastic expert cache**: slots each hold *every* tensor needed to
  evaluate one `(layer, expert)` pair. Residency, lookup, and execution all operate on
  logical `(layer, expert)` ids, never tensor shards.

Everything else in the paper is policy over this hierarchy.

## 2. The techniques

### 2.1 Decode: LRU expert cache driven by routing (semantic-aware caching)

Routing has strong *temporal locality*: consecutive tokens reuse overlapping expert sets
(cited across model families). Instead of a static placement frozen at load/prefill time
(llama.cpp device assignment, KTransformers pinned hot set), FreeToken keeps a shared LRU
whose contents follow the live working set: hit refreshes recency, fill admits, eviction
removes the least-recently-*routed* expert.

Measured effect (replayed traces, equal capacity): global LRU misses **16 %** /
**39 %** of decode reads (Qwen3.6 / DSV4-Flash at 37 % / 11 % of pool cached) vs
41 % / 59 % for KTransformers' prefill-updated placement and 62 % / 89 % for
llama.cpp's routing-blind static split.

### 2.2 Decode: the q⋆ bandwidth-adaptive miss policy

Caching can't eliminate misses; FreeToken splits the `m` unique missing experts of a step
into a **PCIe cache-fill set F** (`q` experts) and a **direct-CPU-execution set C**
(`m−q` experts), run **concurrently**:

```
T_fill(q)  ≈ q·S / B_P            S = bytes per complete expert
T_cpu(m−q) ≈ (m−q)·S / B_R        B_R = max(B_H − B_P, 0)
q⋆ ≈ m · B_P / B_H                (balance the two concurrent branches)
```

- `B_P` = measured pinned host→device transfer bandwidth, `B_H` = measured effective
  bandwidth of the CPU-side expert kernel — **profiled on the deployed machine on real
  tensor shapes**, not spec-sheet numbers.
- Always keep ≥ 1 fill so the cache keeps warming even when the CPU absorbs most misses.
- Outputs are gate-weighted partial sums merged GPU+CPU — **exact**, no approximation.
- As `B_H → B_P`, `q⋆ → m`: degenerates gracefully into pure on-demand cache fill.

This is the core novelty: prior systems (Fiddler, KTransformers, HybriMoE, SMoE) either
fix the CPU/GPU division at startup or recompute it with host-side heuristics; FreeToken
derives it as a closed-form ratio cheap enough to evaluate *inside a captured CUDA graph*.

### 2.3 Prefill: full-layer double buffering

Prefill routes thousands of tokens through each layer, so nearly **all** experts of every
layer activate — per-miss fetching is hopeless. FreeToken allocates **two full-layer
buffers** from the same slot pool used by decode: while the GPU computes layer `l`, a
dedicated transfer stream loads the *complete* expert set of layer `l+1` (routing for it
not even known yet — doesn't matter). Buffers swap; survivors seed the decode cache.
Fallback to on-demand loading when the pool can't spare two full layers.

Ablation: overlap is worth 19 % (4 k tokens) → 26 % (16 k) of prefill throughput;
with it, prefill becomes purely transfer-bound at the PCIe ceiling (8 192-token chunk in
1.19–1.22 s ≈ streaming the 64 GB pool once at 52.7 GB/s).

### 2.4 Prefill: semantic-aware state caching (the agentic-workload killer feature)

Agent harnesses edit context at **special-token boundaries** — OpenClaw strips thinking
blocks except the latest, OpenCode replaces old tool outputs with placeholders,
SWE-agent elides old observations. Any KV/state checkpoint taken *after* the edited
position is invalid, so engines fall back to stale checkpoints and re-prefill thousands
of tokens.

FreeToken anchors **recurrent-state checkpoints at exactly those boundaries** (thinking
segments, tool calls, tool outputs, turns) in its radix prefix tree. After an edit, it
restores from the deepest checkpoint whose position survives, recomputing only the new
suffix. Full-attention layers reuse KV normally; hybrid recurrent layers (gated DeltaNet,
Kimi Delta Attention) resume from the anchored state snapshot. This is why tail TTFT stays
< 44 s while baselines blow past 150 s (client timeouts).

### 2.5 Elastic memory + fast bootstrap

- Expert cache rebuilt at runtime for a revised VRAM budget at scheduler safe points —
  no restart, no reload (VRAM affects performance, never correctness).
- Startup: expert weights read from NVMe **directly into their final host layout**, pinned
  only after filling (pinning empty buffers first faults in and zeroes gigabytes just to
  overwrite them); no warmup pass — cold cache is served by the ordinary decode path.

### 2.6 Implementation machinery

- **Device-side cache control**: one GPU kernel dedupes routed ids, classifies vs the
  residency table, computes `q`, picks victims, rewrites logical→physical slot ids — all
  inside a statically captured CUDA graph with fixed-shape work buffers + device-resident
  valid counts. Victim selection finds K LRU candidates in **one pass** regardless of miss
  count; one fused copy launch serves all banks via a shared logical mapping.
- **FTW weight format**: experts repacked into banks whose leading dimension is the
  flattened `l·E + e` id; parallel direct I/O reads aligned chunks straight into
  exact-size host banks. Skips tensor discovery/repack at every startup.
- **Persistent CPU worker pool** pinned to physical cores; SIMD kernels with in-kernel
  dequantization returning gate-weighted partials.
- **Pure-CPU fallback backend** when the pool can't be pinned/DMA-registered.

## 3. Where joshua stands today (the relevant parts)

| Concern | joshua today |
|---|---|
| Expert storage | mmap'd GGUF; OS page cache is the implicit expert pool |
| Dense weights | `--pin-hot-weights` / `--mlock-hot-weights`: WILLNEED+mlock on the dense set, `MADV_RANDOM` on expert ranges (`engine.rs`) |
| Prefill | `MADV_SEQUENTIAL` over the expert span + tensor-major dispatch + 2-layer-ahead WILLNEED (`quantized_deepseek4.rs:1986-2098`) |
| Decode | per-selected-expert `MADV_WILLNEED` fired **inside dispatch, right before the matmuls** — "no head start", matmuls stall on faults (`quantized_deepseek4.rs:1423-1433`) |
| CPU compute | rayon row-parallel quantized matmuls (SIMD AVX2) — FreeToken's persistent worker pool is roughly covered |
| KV reuse | pool of ≤ 2 warm sessions, **exact token-prefix match only**, prefill suffix only (`engine.rs` `acquire_session`, `MAX_CACHED_MODELS = 2`) |
| GPU | Metal/CUDA run Qwen3-MoE & DSV2/V3/K2 quantized, but weights are copied into GPU buffers wholesale (model must fit unified memory); deepseek4 refuses GPU (IQ2_XXS) |

So joshua is the **CPU twin** of FreeToken's architecture: page cache ↔ host expert pool,
dense pinning ↔ GPU-resident non-expert weights. Several FreeToken ideas transfer almost
directly; a few become *better* ideas here because MLA makes state snapshots tiny.

## 4. Applicable techniques, ranked

### A. Semantic-boundary prefix checkpoints (highest value, lowest risk)

Our KV reuse breaks exactly where FreeToken's does not: `acquire_session` requires the
cached history to be a **strict prefix** of the new prompt. The moment an agent harness
truncates thinking blocks or old tool outputs *in the middle*, the match fails and we
re-prefill the entire session. FreeToken's measurements say this dominates agentic TTFT
(>150 s tails → <44 s).

Concrete plan:
1. Extend `CachedModel { session, tokens }` to carry a small ring of checkpoints
   `(prefix_len, state_snapshot)` recorded at special-token boundaries surfaced by the
   chat template / tool-call parser (`template.rs`, `tools.rs`).
2. Lookup becomes: longest checkpoint whose `[0..prefix_len]` matches the prompt head —
   survives mid-context truncation as long as the cut lands on (or after) a boundary.
3. Snapshot cost is the reason this is *easier* for us than for most engines: DeepSeek-V2/
   V3 MLA caches the compressed latent (`c_kv` + `k_pe`, ~70× smaller than per-head KV per
   the README). Copying one latent per layer per checkpoint is trivially cheap. Standard-KV
   architectures can start with turn-boundary-only checkpoints (copy-on-write pages later).

### B. Give decode's expert residency an explicit controller (user-space expert cache)

Today residency is whatever the kernel page cache feels like, globally competing with
everything else; we can neither size it nor observe it. A bounded, hugepage-backed
user-space arena holding recently-routed `(layer, expert)` weight copies, LRU by last
routing touch (FreeToken §3.2), populated lazily on decode misses:

- explicit capacity knob (like their slot pool / WiSP-style VRAM split),
- measurable hit rate — the first MoE metric joshua doesn't have,
- portable: macOS madvise is far weaker than Linux's, and this machine is a Mac,
- complements (does not replace) `mlock` pinning of the dense set.

Their replay methodology (log routing ids, sweep cache sizes offline) is directly reusable
as a joshua benchmark mode.

### C. Speculative cross-step prefetch at decode (their locality insight, page-cache flavor)

We know each layer's routed experts for token `t−1`; routing consistency says token `t`
largely repeats them. Fire the per-expert `WILLNEED` for layer `i`'s *predicted* ids when
layer `i−1` finishes (or at latest when layer `i`'s attention starts), instead of firing
synchronously inside `dispatch` with zero head start. Wrong predictions cost one redundant
advice call; correct ones convert fault stalls into background streaming. This is a ~20-line
change to `Moe::dispatch` + a small per-layer ring of last-seen ids.

### D. Per-expert-contiguous weight layout (FTW analog)

GGUF stores `ffn_gate_exps` / `ffn_down_exps` / `ffn_up_exps` as three separate tensors per
layer, so **one expert spans three disjoint file extents**; under `MADV_RANDOM` each decode
miss fragments into scattered 4 KiB faults across all three. An optional repack tool
(`joshua repack model.gguf` → sidecar banked file, leading dim = `l·E + e`, gate/up/down
interleaved per expert) makes each expert **one contiguous extent**: larger reads per fault,
trivially prefetchable ranges, one prefetch handle per expert instead of three. This is
exactly why FreeToken built FTW — and our `gguf_ext::layer_expert_ranges` +
`mmap_tensor::prefetch_handle` plumbing already speaks "byte range per expert".

### E. Deterministic prefill overlap (second-order)

We already do their §3.1 trick (SEQUENTIAL advice + tensor-major + layer-ahead hints);
their ablation attributes 19–26 % to overlap *quality*. Where ours depends on kernel
readahead mood, a dedicated reader thread doing positioned `read()` of layer `l+1`'s expert
span into a staging buffer would make the overlap structural. Moderate effort, modest gain —
do after A–D.

### F. Not applicable / deferred

- **CUDA-graph device-side cache control**: no CUDA graphs here. (Conceptual echo for the
  NPU shim: batch descriptors, avoid per-step host round-trips.)
- **GPU expert cache + q⋆ CPU co-execution**: joshua is single-device today. But if we ever
  build a Metal path for deepseek-class MoE (RAM-resident experts + GPU dense path),
  FreeToken *is* the blueprint: LRU slots, q⋆ = m·B_P/B_H with measured bandwidths, exact
  partial-sum merge. On Apple silicon the "two branches" become unified-memory copy vs
  direct execution — the same residual-bandwidth argument applies with ANE/GPU numbers.
- **Runtime elastic VRAM resplit**: meaningless without a managed GPU pool; revisit with F.

## 5. Suggested order of attack

1. **C** — speculative decode prefetch (tiny, immediate latency win on RAM-tight hosts).
2. **A** — boundary checkpoints in `acquire_session` (biggest agentic-TTFT win; MLA makes it cheap).
3. **D** — `repack` sidecar format (unlocks B and helps prefill too).
4. **B** — bounded LRU expert cache with hit-rate metrics (uses D's contiguity).
5. **E** — threaded prefill staging (polish).

Measurement first: add routing-id logging (per layer, per decode step) so B/D wins can be
replayed offline exactly as FreeToken's Figure 4b does.

## 6. Implementation status

- **A — shipped, in a simpler and strictly safer form than snapshots.** Instead of
  checkpoint rings, `acquire_session` now matches pooled sessions by *longest common
  token prefix* and rewinds the KV state in place to that prefix
  (`truncate_kv_cache` on the Qwen3-MoE and DeepSeek-V2/V3/K2 loaders; append-only
  caches make an in-place rewind sound, and the prefix is materialised so the old buffer
  actually frees). The LCP of an edited conversation ends exactly at the harness's edit
  point — i.e. on FreeToken's semantic anchor by construction. Counted separately as
  `Engine::kv_edit_reuse_count`. `deepseek4` is deliberately excluded: its hybrid
  attention keeps running compressor states that cannot be rewound to an arbitrary past
  position — that architecture still needs true state checkpoints (FreeToken §3.1).
- **C — shipped** for `deepseek4`: each MoE layer's routed ids are recorded per forward
  (`ModelWeights::last_routed_experts`, seeded from prefill's final row), and decode
  fires their prefetch before any layer runs, giving the pages a full step of compute to
  stream behind. Mispredictions fall through to the existing dispatch-level prefetch.
- B, D, E remain open, in that order.
