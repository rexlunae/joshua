# Memory analysis: VRAM, RAM and disk in the joshua engine

**Scope.** Where each byte of a model lives at load and at inference time,
per session and per request, on the CPU path and on an accelerator
(CUDA / Metal / OpenCL), and what was changed to make a machine whose
accelerator cannot hold the whole model serve it well.  Numbers use
representative GGUFs: Qwen3-30B-A3B Q4_K_M (18.6 GB, 128 experts/layer × 48
layers, vocab 151 936 × hidden 2 048), DeepSeek-V3 / Kimi-K2 Q4_K_M (~400 GB /
~600 GB, 256 experts × 61 layers, vocab 129 280 / 163 840 × hidden 7 168), and
DeepSeek-V4-Flash IQ2_XXS (80.8 GB, dense set ~8 GiB).

## 1. Disk

| Concern | State |
|---|---|
| Model bytes | The GGUF is `mmap`ed read-only; the OS page cache is the only copy of the weights on the CPU path (`engine::map_model`, `mmap_tensor`).  Nothing is duplicated on disk. |
| Compressed / sparse files | Detected before mapping and reported (`compression.rs`): a mapping over a compressed stream would page-fault-decompress every read. |
| Prefill streaming | The MoE loaders advise `MADV_SEQUENTIAL` over the expert span, dispatch experts in file order, and (deepseek4) run a layer-ahead `pread` thread, so a cold prefill reads each expert tensor as one sequential pass (~1.9 GB/s measured vs ~175 MB/s of demand faults). |
| Decode | Speculative `MADV_WILLNEED` on the previous step's routed experts (deepseek4) and the routing-frequency hot-expert cache (`hot_experts.rs`) keep the working set in the page cache; misses are 4 KiB random reads. |

Disk was already well handled; nothing changed here.

## 2. RAM (host)

### What was already right

* Zero-copy weights: every joshua-native loader borrows quantized tensors
  in place from the mapping (`mmap_tensor::borrowed_range`), so a 600 GB
  Kimi-K2 costs pointers until tokens route to an expert.
* Dense/expert split: `--pin-hot-weights` prefetches the ~8 GiB per-token
  working set and advises `MADV_RANDOM` on the experts; `--mlock-hot-weights`
  pins it.  `--prefetch-model` auto-warms a model that fits.
* Admission and pool sizing are RAM-adaptive (`LOW_MEM_FLOOR`,
  `release_model`), prefill runs in 512-token chunks, and the deepseek4
  layer-ahead prefetch depth follows free RAM.

### What was wrong

**R1. The token-embedding table was dequantized to f32 per session.**
Every joshua-native loader did `token_embd.weight → dequantize → f32
Tensor` at load.  That table is `vocab × hidden × 4` bytes of *anonymous*
memory (not page cache, not shareable, not evictable):

| Model | f32 table per session |
|---|---|
| Qwen3-30B-A3B | 1.24 GB |
| DeepSeek-V3 | 3.7 GB |
| Kimi-K2 | 4.7 GB |
| DeepSeek-V4-Flash | ~2 GB |

With the warm pool holding up to 4 sessions, a Kimi-K2 server paid up to
19 GB of RAM for a table whose quantized form is 5–10× smaller and already
resident in the page cache.

*Fix:* `token_embedding::TokenEmbedding` keeps the table quantized and
gathers rows through candle's quantized embedding kernel (borrowed from the
mapping on the CPU, uploaded compressed on CUDA/Metal).  Per-session cost is
now zero; the numbers are bit-identical (same blocks, same `to_float`, per
row instead of whole).  Dense f32 is kept only for OpenCL (its storage is
f32 already) and float dtypes on accelerators.

**R2. Every session was a full model instance.**  `Engine::load_model`
built a complete `QuantizedModel` per session.  On the CPU that is cheap
for the borrowed tensors but still repeats every per-instance allocation
(embedding table, router weights, RoPE tables, residency tables).

*Fix:* the `qwen3moe` and `deepseek2` loaders split into an immutable
`Arc<Shared>` (all weights, residency backend, device) and per-session state
(the per-layer KV cache, the hot-expert routing record).  `new_session()` is
one `Arc` clone.  The engine keeps the first load as a template and derives
every session from it; a concurrent request now costs its KV cache.

## 3. VRAM (accelerator memory)

This was the weak part of the design, and the subject of the request.

**V1. The whole model had to fit in device memory.**  On CUDA/Metal the
`qwen3moe` and `deepseek2` loaders skipped the borrow path entirely and
copied every expert to the device.  Qwen3-30B-A3B Q4_K_M (18.6 GB) would
not load on a 12 GB or 16 GB card at all; only `deepseek4` had the split
layout (dense set on the device, experts on the CPU).

*Fix:* `ExpertPlacement` (`--expert-placement auto|device|host`,
`EngineOptions::expert_placement`).  `host` keeps the routed experts in
host RAM — borrowed from the mapping, run on the CPU SIMD expert kernels,
with the whole hot-expert / prefetch machinery active — and puts only the
dense set (attention, norms, routers, shared experts, embeddings, output)
on the device.  Each MoE layer moves its input across once and its output
back once (`Moe::dispatch`), exactly as deepseek4 does; for decode that is
two `hidden × 4`-byte transfers per layer, negligible on PCIe.  `auto` picks
`device` only when `dense + experts + 1 GiB` fits the device's free memory
(`cudaMemGetInfo` on CUDA, or `--vram-budget`), and `host` otherwise; on
OpenCL, whose storage is dense f32 (an uploaded expert would be 8–16× its
on-disk size), `auto` is always `host`.  With no probe and no budget the
historical layout is kept, so nothing changes on machines that fit.

Per-device memory with `host` placement:

| Model | Device-resident | Host-resident (page cache) |
|---|---|---|
| Qwen3-30B-A3B Q4_K_M | ~1.1 GB dense + KV | ~17.5 GB experts |
| DeepSeek-V2-Lite Q4_K_M | ~1.5 GB dense + KV | ~8 GB experts |
| Kimi-K2 Q4_K_M | ~6 GB dense + KV | ~590 GB experts |

**V2. Loading an expert tensor to the device copied it three times.**
The device path did `read whole tensor → host Vec` → `upload as one device
tensor` → `QTensor::data()` (download the whole tensor again) → per-expert
`QStorage::from_data` (upload again).  Three bus crossings per tensor, a
transient host `Vec` and a transient whole-tensor device buffer on top of
the per-expert buffers — 2× the tensor's size in VRAM at the peak of every
layer (a Kimi-K2 expert tensor is ~2.5 GB).

*Fix:* `mmap_tensor::expert_slices` hands the loader each expert's bytes
straight out of the mapping; each expert is one `from_data` upload from
the page cache with no staging.  The streamed (no-mmap) fallback now reads
onto the host once and never allocates a whole-tensor device buffer.

**V3. Concurrent sessions multiplied the model in VRAM.**  `max_concurrency`
defaulted to the CPU count and the warm pool kept up to 4 sessions by *RAM*
availability, so on a GPU box with 16 cores and 32 GB of RAM a burst of
requests uploaded the model 16 times and kept 4 copies warm.

*Fix:* weight sharing (R2) removes the multiplier for `qwen3moe` /
`deepseek2`: every session shares the one uploaded copy.  For architectures
whose sessions still own their weights (candle's stock loaders, deepseek4),
the engine now sizes the default concurrency and the warm-pool cap from the
device's free memory and the per-session device footprint
(`placement::instances_for_memory`) when a probe or `--vram-budget` is
available.

**V4. The embedding table (R1) was also a per-session VRAM cost** — the
same 1.2–4.7 GB per session, on the device.  Same fix.

## 4. What the change means on a memory-constrained GPU box

Qwen3-30B-A3B Q4_K_M on a 12 GB card with 32 GB of RAM, before and after:

| | Before | After |
|---|---|---|
| Load | fails (18.6 GB upload) | dense set ~1.1 GB on the GPU, experts in the page cache |
| Second concurrent request | +18.6 GB VRAM (fails) | +KV cache only |
| Embedding table | 1.24 GB f32 per session | 0 (quantized, shared, in page cache) |
| Expert compute | GPU (if it fit) | CPU SIMD kernels, page-cache resident, hot-expert cache + prefetch |
| Attention / dense compute | GPU | GPU |

The decode cost of `host` placement is the CPU expert matmul: ~3 B active
parameters per token for 30B-A3B, which the fused AVX2 kernels serve at a
few tens of ms per token — the same regime the CPU-only path has always
been in, now with attention and the output head on the GPU.

## 5. Not changed (follow-ups)

* **A bounded VRAM expert cache** (hot experts uploaded, cold ones on the
  CPU — FreeToken's split) is the next step past `host` placement; the
  `WeightCache` in `paged_weights.rs` and the `ExpertResidency` seam are the
  pieces.  Requires a GPU to validate.
* **deepseek4 weight sharing.**  Its KV is already separate from the
  weights, so the same `Arc<Shared>` split applies, but the loader is 3 000
  lines and was not refactored here; on an accelerator its ~8 GiB dense set
  is still per session, bounded by the new device-aware caps.
* **candle's stock loaders** (llama, gemma, …) copy the whole model to the
  heap/device and cannot share weights; they are the vendored crate's
  code.
* **Metal memory probe.**  Unified memory: use `--vram-budget` to state
  the budget; `auto` without it keeps today's zero-copy Metal layout.
