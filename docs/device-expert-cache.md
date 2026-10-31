# Joshua device expert cache — design sketch

**Status:** design + phased implementation. Phases 1 and 5-core, device-pool
primitives, and CUDA/Vulkan IQ2_XXS dispatch are merged; the device-validation and remaining runtime
items are tracked in [issue #52](https://github.com/rexlunae/joshua/issues/52).  The
CPU-testable Tier-3 (SSD/Engram) additions live in
[`docs/multilevel-weight-cache.md`](multilevel-weight-cache.md) §H.

| Phase | What | Status |
|---|---|---|
| 1 | `ExpertResidency` seam + `CpuResidency` (dedupe `MlpPrefetch`) | ✅ merged (#49) |
| 5 core | placement math (`--expert-cache auto`) + `adaptive_budget` | ✅ merged (#50) |
| 2 | VRAM slot pool + background upload | Partial — `DeviceResidency` + `ExpertUploader`; CUDA transfer-stream completion before publication, not a pending-slot/event handoff; see the audit below |
| 3 | device-backed expert storage + matmul dispatch | ✅ CUDA/Vulkan IQ2_XXS experts (`cf0c938`, `0e953af`); `Ds4DeviceExpert` |
| 4 | layer offload (dense models) + KV placement | ⚠️ partial — `--expert-placement host` keeps experts CPU-side; no explicit KV-tier placement |
| 5 runtime | adaptive loop feeding measured hit-ratio/bus bytes into `adaptive_budget` | ⚠️ partial — budget sizing is static/`auto`, not a live feedback loop |
| 6 | IQ2_XXS / MXFP4 deepseek4 GPU kernels (V4-Flash on GPU) | Partial — IQ2_XXS merged (`cf0c938` CUDA, `0e953af` Vulkan); not evidence of MXFP4 or full-model GPU coverage |
| 7 | explicit SSD (NVMe/SATA) residency tier + Engram-aware tier | ✅ `NvmeResidency` (`--ssd-expert-cache`), `--engram-tier` — see `docs/multilevel-weight-cache.md` §H |

**#114 (partial-cache usability on small cards).** The OpenCL/SYCL slot pool
is live, and the measured small-card thrash was fixed at its roots, not by
policy: (1) the SYCL bridge gained a second in-order **transfer queue**
(`joshua_sycl_transfer_write`) — uploads previously shared the compute queue,
so a large memcpy delayed every enqueued kernel; (2) the host-page release
hook keeps the pages of **non-hot** (churn) experts, so a host miss after an
eviction re-runs from the warm page cache instead of faulting the model file;
(3) the hot-set refresh now also **requests uploads** for its members, so the
cache converges on the stable working set ahead of the per-step churn; and
(4) `--expert-placement auto`'s engagement floor dropped from 10% to 5%
residency (a hot-protected cache mostly hits even at low residency), and a
decision that declines *only* on residency moves the dense set to CPU-BLAS
when the freed memory clears the floor and the placement probe does not show
a decisive device GEMM win (`dense_to_cpu_for_cache`).  The CUDA async-copy
pending-slot/event handoff below remains future work.

### CUDA audit (2026-10-31, issues #62 / #52)

DeepSeek4 has a model-wide bounded pool, a background `ExpertUploader`, and
lookup-only dispatch with host fallback. Its mmap-backed gate/up/down blocks
reach `QStorage::from_data_transfer`. CUDA previously routed that call through
ordinary `from_data`, allocating and copying on the compute stream. It now uses
one lazily created non-blocking transfer stream per `CudaDevice` (shared by its
clones). The worker completes that stream before returning the storage; only
then may `DeviceResidency` publish it. Compact blocks and existing kernel padding
are preserved. No pinned staging allocation, raw FFI, pending-slot readiness
state, or change to cache eviction/hot-set policy is introduced.
Cudarc can synchronize the context once when the first owned stream is created
to enter multi-stream mode; the per-upload completion wait is stream-local.

`tests/cuda_expert_cache_tests.rs` requires a working CUDA device and covers
Q2_K/Q4_K/Q6_K/Q8_0 byte preservation and bit-identical matmul versus the normal
CUDA upload, plus a tiny mmap-backed DeepSeek4 IQ2_XXS/Q8_0 cache. Two sessions
share a three-expert, 155,136-byte logical payload budget, with concurrent prefill,
70 decode steps each (crossing the hot-set refresh interval), all-row logits,
hits/uploads, bounded occupancy, and exact budget-zero/too-small-budget host
parity. The cache logit tolerance is the existing accelerator-cache tolerance (5% of the reference
logit spread), not a claim of CPU/CUDA bit parity. Kernel padding is additional
to the pool's existing logical-payload accounting.

This change wires qwen3moe/deepseek2 to one model-wide `CompositeResidency` and
background uploader rather than unreachable per-layer pools behind a CPU-only
session residency. With an explicit cache option, compact source experts remain
on the host (mmap or streamed loading); the destination is the requested expert
device, or the dense device when experts are host-placed. Uploads use the shared
transfer path; dispatch looks up the device payload, moves activations to its
actual device, and retires the payload after enqueueing its work. Misses retain
host execution. No full expert-pool f32 materialization or per-session weight
copy is introduced.

Slots reserve the largest compact gate/up/down payload, with the total budget
capped at that reservation times the actual expert count. Auto hot-set sizing
uses three quarters of resolved slots instead of host RAM (at least one tracked
expert for a nonempty single-slot pool, so uploads are not disabled); refresh replaces the
capped protected set instead of accumulating stale protected members. Sessions
share the pool and uploader, but their frequency histories remain independent:
one session can still replace another's protected set. **Disjoint-session
no-thrash policy is not solved.** Logical payload accounting excludes backend
padding, transient activations, and retired in-flight payloads; it is not a hard
bound on all CUDA allocations.

On kleya's RTX 5060 Ti (16,311 MiB), existing services were preserved; the initial
probe showed 3,371 MiB used and 12,477 MiB free. Only tiny fixtures were loaded.
Release CUDA tests pass twice on the final dispatch-lifetime implementation
(3 passed, 1 explicitly ignored timing probe): compact transfers, DeepSeek4,
and Qwen/DeepSeek2 shared-cache prefill, 70 decode steps per concurrent session,
all-row continuation, streamed-source fallback, and exact budget-zero host-expert
parity. The older-loader destination is explicitly CUDA (dense CUDA, experts
CPU), not an OpenCL request whose cache actually runs on CPU. Each older pool
has three slots / 331,776 logical bytes, positive upload/hit counts and no upload
failures. Qwen maximum absolute logit error was 0.020814; DeepSeek2 0.027171;
all tested greedy winners match. These use the existing 5%-of-spread accelerator
tolerance, not bitwise CPU/CUDA parity. An earlier release attempt failed parity
before explicit dispatched-payload retirement was added; unchanged-main's
existing two CUDA tests passed in a separate source archive.

The explicit release timing probe uses a 3-token prefill, 70 warm-up decode
tokens and three trials of 64 timed decode tokens, synchronizing through logits
readback. Dense weights stay CUDA in every mode. Median tokens/s:

| Tiny fixture | Host experts, cache disabled | Three-slot CUDA cache | All experts CUDA |
|---|---:|---:|---:|
| Qwen3MoE Q4_K | 329.659 | 258.441 | 790.479 |
| DeepSeek2 Q4_K | 627.408 | 794.826 | 939.931 |

The Qwen partial cache **regresses** in this bounded workload. The host-expert
baseline is not all-host inference; services were running and samples varied.
These figures do not establish production throughput, a real-model hit-rate
target, or no-regression acceptance. Reproduce with:

```sh
CARGO_BUILD_JOBS=2 cargo test --release --features cuda --test cuda_expert_cache_tests -- --nocapture --test-threads=1
CARGO_BUILD_JOBS=2 cargo test --release --features cuda --test cuda_expert_cache_tests cuda_older_loaders_decode_measurement -- --ignored --nocapture --test-threads=1
```

Broader CUDA-feature validation passed 490 library tests (10 ignored), 12
DeepSeek2, 13 memory-placement and 8 Qwen3MoE tests. The added single-slot
automatic-budget unit test also passes. With `CUDA_VISIBLE_DEVICES=''`, the
CUDA-feature speculative suite passes all 22 tests on the CPU fallback. Final
`cargo check --features cuda --all-targets` passes. The CUDA-visible speculative
suite failed 12 of 22 tests, reproduced with the same failure set on unchanged
`99d2ddb` in a separate source archive: ten model-load errors rejecting a foreign
CUDA expert device and two DeepSeek4 `unsupported dtype for quantized matmul
Iq2Xxs` errors. These broader engine-placement/dtype failures are not silently
counted as passes or fixed by this cache change.

Remaining acceptance: real-model all-host/all-device no-regression comparisons,
measured PCIe traffic, pinned-memory/pending-event overlap, aggregate hot-set
policy for disjoint concurrent sessions, physical-allocation-aware sizing and
live adaptive placement. Existing older-loader foreign-device validation also
still restricts CPU-dense/accelerator-expert placement. Keep #62 and umbrella
#52 open. OpenCL-with-CPU-cache evidence (including #244) is CPU/fallback evidence,
not a replacement for the actual CUDA tests above.

**Backend note:** the device backend must not be CUDA-only — ROCm stays an
option.  HIP's API mirrors CUDA's (`hipMemcpyAsync` ≈ `cudaMemcpyAsync`,
`hipStreamSynchronize` ≈ `cudaStreamSynchronize`), and candle exposes both via
feature flags, so the slot-cache logic should be written behind a thin
device-copy trait and instantiated for whichever backend the build enables.
## 1. Goal and target regime

Run a sparse MoE model whose **routed-expert weights do not fit in VRAM**, but do fit
in VRAM + system RAM together, using the machine's total memory efficiently:

- **GPU** holds the dense set (embeddings, norms, attention, routers, shared
  experts, output) — small, touched every token, latency-critical.
- **Host RAM** holds the complete routed-expert pool (the source of truth).
- **VRAM** holds a **bounded cache of hot experts**, evicted LRU-by-routing-frequency.

This is the "edge MoE serving" regime, and the shape of FreeToken (22–25 tok/s on
DeepSeek-V4-Flash 284B/MXFP4 from one RTX 5090), which the repo's
`FREETOKEN_NOTES.md` already summarizes.

What "efficient" means quantitatively (PCIe 5.0 x16 ≈ 63 GB/s, 4.0 x16 ≈ 31.5 GB/s):

| Per-token expert traffic | Worst case (all cold, 2 GB) | With a hot cache (~85% hit, ~0.3 GB) |
|---|---|---|
| Bus time at 63 GB/s | ~32 ms (~30 tok/s ceiling) | ~5 ms (~200 tok/s bus ceiling) |

The cache exists to turn most of the per-token weight traffic into VRAM hits, so the
limiter becomes compute, not the bus.

## 2. Reused building blocks (already in Joshua)

- `hot_experts::HotExpertCache` — routing-frequency LRU policy. **Reuse unchanged.**
  It already yields the ordered hot set `(layer, expert)` each `REFRESH_STEPS` (64)
  decode steps; on CPU the loader feeds it to `madvise`. On GPU the *same* set feeds a
  device-cache executor. This is the key point: the policy is device-agnostic.
- Per-expert weight ranges + prefetch handles — `deepseek2`/`deepseek4`/`qwen3moe`
  already slice each expert tensor into per-expert `Mlp`/`ExpertTensor` with exact
  byte offsets and 3 handles (gate/up/down). That granularity is exactly what a
  per-expert device upload needs.
- `ComputeBackend` / `--device auto|cpu|cuda|metal` and the `cuda`/`metal` feature
  gates.

## 3. Design

### 3.1 Residency backend (the executor)

Introduce a trait that decouples the *policy* from *where the bytes live*:

```rust
/// Makes a hot expert's weights resident on the active compute device, and
/// releases that residency. The policy (HotExpertCache) only names experts.
pub trait ExpertResidency: Send + Sync + 'static {
    /// Ensure `(layer, expert)` is resident and return the read handle the
    /// matmul path uses. Idempotent (a re-acquire of a resident expert is a
    /// hit and cheap).
    fn acquire(&self, layer: u32, expert: u32) -> Result<ExpertHandle>;
    /// Drop the residency (decrement refcount / free a slot).
    fn release(&self, layer: u32, expert: u32);
    /// Expert capacity the device can hold, driving the HotExpertCache budget.
    fn capacity(&self) -> usize;
}
```

Two implementations (the device one is backend-agnostic — CUDA or ROCm,
whichever the build enables; HIP mirrors the CUDA API, so one slot-cache
implementation covers both behind a thin copy-trait):

- **`Cpu` (today's `madvise` path)** — `acquire` = `Mlp::prefetch()`
  (`MADV_WILLNEED` over the 3 mmap ranges); `release` = no-op; `capacity` = derived
  from free RAM / per-expert bytes (what `--pin-hot-experts` is today).
- **`Cuda`** — `acquire` = find-or-fill a VRAM slot with `cudaMemcpyAsync`; `release`
  = free the slot; `capacity` = `floor(VRAM_budget / expert_bytes)`.

The loaders stop calling `Mlp::prefetch()` directly and call
`residency.acquire(layer, expert)` for the set `HotExpertCache::refresh()` returns.

### 3.2 CUDA device expert cache

A fixed pool of **slots**, one expert per slot:

```
slot := { 3 device buffers (gate/up/down), size = max expert bytes for that layer }
state := free_list: Vec<SlotId>
         table: HashMap<(u32,u32), SlotId>      // resident experts
         last_use: Vec<u64>                     // slot-level LRU clock
         stream: CUstream                        // copy stream, separate from compute
```

- **Acquire:** hit → mark slot used, return buffers. Miss → pop a free slot (or evict
  the least-recently-used slot not in the current hot set), enqueue 3 `cudaMemcpyAsync`
  from the mmap host ranges into the slot on the copy stream, record `(l,e) → slot`.
- **Overlap:** the copy is issued from the *previous step's* routed ids (speculative,
  exactly like today's `prefetch_speculative`) so transfers overlap compute. Before
  the MoE matmul, `cudaStreamSynchronize` (or an event) on the copy stream for this
  token's experts. Misprediction is cheap — a wasted copy, not a stall.
- **Eviction policy:** slot-level LRU, but biased by `HotExpertCache` — never evict a
  slot that is in the current hot set; prefer evicting slots that have fallen out.
  (This is the FreeToken "semantic-aware LRU" — the routing-frequency set protects
  the working set from a plain-LRU thrash.)

### 3.3 Matmul integration (the hard, load-bearing part)

Today on CPU the borrowed weights are `QStorage::Cpu` read in-place from the mmap. On
a GPU, the matmul must read the expert from a **device buffer** instead. This is the
one piece that is not a "policy" or "copy" change:

- The loader needs a `QStorage::Cuda` (or Metal) variant for the expert tensors, so
  `QMatMul::forward` dispatches to a device kernel reading the slot buffer.
- **DeepSeek4:** compact CUDA/Vulkan IQ2_XXS expert dispatch exists. MXFP4 and
  full attention/indexer/compressor coverage are separate claims requiring their
  own evidence, not consequences of the expert-cache wiring.
- **Older loaders:** `deepseek2` and `qwen3moe` use standard k-quants
  (Q4_K/Q6_K/…) covered by candle's GPU kernels. The audit above records the
  bounded-cache integration and the remaining performance acceptance.

### 3.4 Layer offload (second strategy — for dense models and KV)

A *separate* placement strategy, chosen when the model is dense (or when VRAM is so
small the dense set doesn't fit):

- Assign each tensor a fixed home (GPU or CPU). Weights **never** move; only the
  activation `[batch, seq, hidden]` crosses the boundary each step — a few hundred KB
  per token, negligible on PCIe.
- CPU runs layers `0..k`, GPU runs `k..n`. The cost is the CPU compute for its layers
  (CPU decode ~0.5 tok/s for dense work), not the bus.
- KV cache lives with its attention device: offloading attention to CPU keeps KV in
  cheap RAM but pays activation + KV round-trips; keeping attention on GPU eats VRAM.
  The placement model must price both.

## 4. Automatic placement

Three tiers, as in the leading systems (llama.cpp `-ngl`, Transformers `device_map`,
ZeRO-Offload/Infinity, FlexServe, FreeToken):

1. **Static sizing** — from `cudaMemGetInfo` (free VRAM), per-expert byte sizes, and
   per-layer dense sizes: pick the device-expert-cache budget
   `min(VRAM_free / expert_bytes, n_experts)` and, if used, the layer-offload split
   point `k`.
2. **Cost model** — estimate tok/s for each candidate configuration:
   `max(compute_time, PCIe_bytes / bus_bw)` per token, using a once-measured device
   compute bandwidth and a probe of PCIe bandwidth (or the user's bus spec). Choose
   the argmax. This is deterministic and testable.
3. **Online adaptation** (FreeToken's "bandwidth-adaptive execution") — watch the
   router's cache hit rate and the measured copy throughput, then grow/shrink the
   cache: if the miss rate is low, shrink to free VRAM for KV/attention; if misses
   are saturating the bus, grow (until VRAM). Expose the chosen budget via a log line
   and a metrics event, and let `--pin-hot-experts` override it for reproducibility.

The automatic knob should land as: `--device auto` already exists; add
`--expert-cache auto|<n>` (default `auto`) where `auto` = tiers 1–3, `n` = fixed
budget (what we have today on CPU).

## 5. Phased roadmap

1. **Refactor (no behavior change):** introduce `ExpertResidency` with a `Cpu` impl
   wrapping today's `madvise`; move the 3 loaders' `prefetch_expert` onto it. Keeps
   the CPU tests green and proves the seam.
2. **CUDA residency for k-quant experts (unblocked):** slot allocator + async copy +
   event sync for `deepseek2`/`qwen3moe`; wire `HotExpertCache` budget to
   `residency.capacity()`. Validate on a GPU pod with a Q4_K MoE model.
3. **Device-backed QStorage + matmul dispatch** so the copied experts are actually
   used (needs the loader to build a device-storage variant of `Mlp`).
4. **Layer offload** for dense models + KV placement, behind the same cost model.
5. **Automatic placement** (static + adaptive) and the `--expert-cache auto` surface.
6. **MXFP4 deepseek4 GPU kernels** — the long pole to bring V4-Flash onto this path.

## 6. Risks / open questions

- **The copy/compute overlap and the sync point** dominate latency; getting the
  speculative prefetch (issue from last step's routing) + event-based sync right is
  the difference between a win and a stall. Needs microbenchmarks, not just tests.
- **Slot sizing is per-layer** (expert dims can differ across layers; deepseek2 has
  hash/dense layer variants). Either per-layer pools or one max-size pool — measure.
- **Host-side pinned memory** for `cudaMemcpyAsync`: the mmap pages are not pinned;
  copy throughput to/from pageable memory is lower and may need `cudaMemcpyAsync`
  on non-default streams or staging buffers. This affects the PCIe numbers above.
- **Full-model GPU coverage remains distinct from expert kernels** — CUDA/Vulkan
  IQ2_XXS expert kernels exist; that does not establish MXFP4 or every V4-Flash
  attention/indexer/compressor path.
- **Concurrent sessions** share one expert cache; the `HotExpertCache` counters are
  per model instance today, so a multi-session warm pool needs the policy lifted to
  be engine-shared (a known follow-up from #48).

## 7. Test plan

- **Unit:** `ExpertResidency::Cpu` == today's madvise (parity test); a fake `Cuda`
  backend that counts acquire/release/evict against `HotExpertCache` output; the
  placement cost model (choose cache size) on synthetic bandwidth numbers.
- **Integration (GPU CI):** a small Qwen3-MoE / DeepSeek-V2-Lite Q4_K; assert
  (a) the expert cache fills to `capacity`, (b) hit-rate > threshold on a repeated
  prompt, (c) host→device bytes over PCIe ≈ (misses × expert_bytes), (d) throughput
  vs. the whole-in-VRAM baseline.
- **Adaptation:** a bandwidth probe hook returns a synthetic slow bus; assert the
  adaptive budget shrinks accordingly.
