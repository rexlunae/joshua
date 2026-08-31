# Joshua device expert cache — design sketch

**Status:** draft (not implemented). Builds on `--pin-hot-experts` / `HotExpertCache`
(merged in #48).

**Status:** design + phased implementation.  Phase 1 (residency seam) and
phase 5 core (placement math) are merged; the CUDA/ROCm-dependent phases are
tracked future work in [issue #52](https://github.com/rexlunae/joshua/issues/52).

| Phase | What | Status |
|---|---|---|
| 1 | `ExpertResidency` seam + `CpuResidency` (dedupe `MlpPrefetch`) | ✅ merged (#49) |
| 5 core | placement math (`--expert-cache auto`) + `adaptive_budget` | ✅ merged (#50) |
| 2 | CUDA/ROCm residency: VRAM slot pool + async copy + event sync | ⏳ future — needs a GPU to build/test |
| 3 | device-backed expert `QStorage` + matmul dispatch | ⏳ future |
| 4 | layer offload (dense models) + KV placement | ⏳ future |
| 5 runtime | adaptive loop feeding measured hit-ratio/bus bytes into `adaptive_budget` | ⏳ future (with the device backend) |
| 6 | MXFP4 deepseek4 GPU kernels (V4-Flash on GPU) | ⏳ future (long pole) |

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
- **Blocker for V4-Flash specifically:** `deepseek4`'s IQ2_XXS experts have no
  CUDA/Metal kernel (the loader today refuses a GPU — "would materialise ~16× f32").
  The proven path is **MXFP4** weights + a deepseek4 GPU matmul (FreeToken's choice),
  plus GPU kernels for the deepseek4 attention/indexer/compressor. That is the real
  work, not the cache.
- **Unblocked today:** `deepseek2`, `qwen3moe`, and the dense/candle architectures use
  standard k-quants (Q4_K/Q6_K/…) that candle's GPU kernels already cover. So the
  first shippable version is a GPU box serving e.g. DeepSeek-V2/V3 or Qwen3-MoE in
  Q4_K with a device expert cache — not V4-Flash.

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
- **IQ2_XXS has no GPU kernel** — the V4-Flash path is blocked on kernels, not on the
  cache. MXFP4 is the known-good alternative (FreeToken).
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
