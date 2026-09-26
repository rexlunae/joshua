# OpenCL and Vulkan backends: running models on an iGPU

joshua's OpenCL and Vulkan backends are vendored into `candle-core`
(`vendor/candle-core/src/opencl_backend`, `vendor/candle-core/src/vulkan_backend`)
and selected with `--device opencl` / `--device vulkan` (cargo features
`opencl` / `vulkan`).  This note describes how they execute a model and what
to expect from them on an integrated GPU, which is the hardware they were
written for.

## Why an iGPU used to lose to the CPU

Both backends began as "storage on the device, compute on the host": every
operator copied its inputs back to the CPU, ran candle's CPU kernel and
copied the result up again, and every weight was held as dense f32.  On an
iGPU that is strictly worse than the CPU path:

* an iGPU shares DRAM with the CPU, so it has no bandwidth advantage to
  begin with, and holding weights as f32 makes it stream 4–8× the bytes of a
  Q4/Q8 model per token;
* each operator paid a host round trip and a queue drain, so a decode step
  was a few hundred synchronous copies rather than a stream of kernels;
* the one native path (a naive f32 GEMM) was gated off by default.

## What runs on the device now

Both backends implement the same design:

* **Every `BackendStorage` operator is a device kernel** — strided and
  broadcast elementwise ops, comparisons, `where`, casts, strided copies
  (what `contiguous`, `cat`, `narrow` and transposes materialise through),
  row and generic reductions, arg-reductions, `index_select`, `gather`,
  `scatter`, `index_add`, a tiled f32 GEMM with explicit strides (so
  transposed and broadcast operands need no copy) and GEMV kernels for the
  single-token case.  Convolutions and pooling still take the CPU round-trip.
* **Fused attention-path ops.**  candle-nn's `softmax_last_dim`, `rms_norm`,
  `rope` and `rope_i` have device implementations (one kernel per row / pair)
  instead of the host copy the custom-op default takes.
* **Block-quantized weights stay quantized on the device.**  `QOpenClStorage`
  / `QVulkanStorage` hold the GGUF blocks as on disk for every dtype
  (Q4_0 … Q8_1, Q2_K … Q8_K, F16, BF16, F32).  Decode (≤ 16 rows) runs a
  dequantize-in-kernel GEMV: each work-group owns one output column and
  streams the row's blocks once, so a 4-bit model moves 4-bit bytes.  Prefill
  dequantizes the weight to an f32 scratch buffer once and runs the tiled
  GEMM.  Token embeddings are gathered by a dequantizing row kernel, so the
  table is never expanded to f32 either.
* **Arm Mali / Immortalis (Vulkan).**  On an Arm GPU (Vulkan vendor id
  `0x13B5` — e.g. the Immortalis-G720 of the CIX P1 in the Orange Pi 6)
  the decode GEMV switches to `k_qgemv_mali`: it dequantizes each weight
  sub-block once for up to four activation rows (the generic kernel
  decodes it once per row), gives each output column a team of four lanes
  inside a 64-invocation work-group (four 16-wide Mali warps, 16 columns)
  and meets the partial sums in one shared-memory step instead of a
  256-wide tree with eight barriers — Mali's shared memory is ordinary
  cached memory, so that tree is the generic kernel's costliest part there.
  It keeps the generic kernel's decoders, so every quantisation is covered;
  it is checked against the CPU on every block format (forced on the test
  device), but its speed has not yet been measured on Mali hardware.
  `JOSHUA_VULKAN_QGEMV=mali|generic` forces either kernel on any device.
* **Asynchronous execution.**  OpenCL launches on an in-order queue and only
  blocks on a host read-back.  Vulkan records into one command buffer per
  device and submits lazily — on a read-back, an explicit `synchronize`, or
  every 1024 launches — with a compute→compute barrier after each dispatch;
  buffers dropped while a batch is open are released after it completes.
* **Zero-copy weights (OpenCL).**  On a device with host-unified memory the
  loader wraps page-aligned ranges of the memory-mapped GGUF with
  `CL_MEM_USE_HOST_PTR`, so the model file's page cache *is* the weight
  memory: no second copy, and the kernel can evict clean pages under
  pressure exactly as on the CPU.  Vulkan uploads the blocks (still the
  on-disk size, not f32).
* **Memory.**  The Vulkan backend sub-allocates every tensor from 128 MiB
  blocks of one host-visible, host-coherent memory type (device-local when
  the hardware offers a unified type), because drivers cap the number of
  `VkDeviceMemory` allocations, often at 4096; pipelines are compiled once
  per device (GLSL → SPIR-V through `naga`, no external compiler).

The routed experts of an MoE model stay on the CPU SIMD expert kernels on
both backends by default (see `--expert-placement`): the dense set is what
an iGPU speeds up, and on unified memory the two share DRAM either way.  A
discrete OpenCL card can additionally hold a bounded cache of `deepseek4`'s
routed experts — see the next section.  The dense set itself is placed by
the startup probe (`--dense-placement auto`), which times the *quantized*
matmul the loaders actually run — a Q4_K weight against f32 activations on
the CPU and on the device — rather than an f32 GEMM, so it measures the
path a model takes.

## Discrete GPUs: the VRAM expert cache (DeepSeek-V4 on an Arc B50)

DeepSeek-V4-Flash is 43 MoE layers × 256 routed experts (6 used per token);
each expert is IQ2_XXS gate/up plus a Q2_K down projection, 6.75 MiB, and
the pool is ~72 GiB — more than any card holds, while the dense set is
~8 GiB.  On a discrete OpenCL device (an Intel Arc B50, 16 GB) the loader
therefore keeps *every* routed expert borrowed from the memory-mapped file
on the host — the CPU path, the prefetch source and the source of truth —
and runs a **bounded device cache** over them:

* One model-wide slot pool (`residency::DeviceResidency`), a byte-budgeted
  LRU keyed by `(layer, expert)`.  A slot is the expert's three quantized
  tensors uploaded as-is (IQ2_XXS is a first-class candle dtype on the
  device; `k_qgemv` decodes it in the kernel) — 7,077,888 bytes each on the
  real model, counted exactly against the budget, never over-committed.
* A background uploader thread fills it on the device's *transfer queue*
  (a second in-order OpenCL queue), so an upload never waits behind, or
  holds back, the compute queue's kernels.  Uploads are blocking writes on
  that queue; a slot becomes visible only after its write returned, which
  makes it safe for any later launch on any queue of the context.
* Per MoE layer, dispatch partitions the routed experts: the resident ones
  are enqueued on the device first (asynchronous launches), the rest run on
  the host expert kernels while the device works, and the two partial sums
  are added on the block's output device.  Decode misses are queued for
  upload so the next step that routes there finds them resident; a prefill
  only *reads* the pool (a prompt routes through nearly every expert, and
  uploading them all would stream the pool over the bus and evict the
  decode working set) but queues the last prompt row's experts for the
  decode that follows.  The routing-frequency hot set
  (`--pin-hot-experts` / `--expert-cache`, defaulting to ¾ of the slots
  when a pool exists) is protected from eviction.
* The dense set can live on the card too, or stay on the CPU
  (`--dense-placement cpu`) with only the expert cache on the device; the
  activations then hop across once per layer in each direction.
* **The host page cache and the device pool hold different experts.**  On
  a host whose RAM cannot hold the whole pool (64 GB against ~72 GiB), every
  expert the card carries is one the page cache no longer needs: the
  hot-set refresh and the speculative next-step prefetch skip the host
  pages of device-resident experts, and a second after an upload the
  expert's host pages are dropped from the mapping and the page cache
  (`MADV_DONTNEED` + `posix_fadvise(DONTNEED)`; the page-cache folios that
  lie wholly inside the expert's ranges go at once, the ones straddling
  an edge are unmapped and deactivated, so the kernel reclaims them
  first), so RAM fills with the experts the host still has to run.  The
  release skips an expert the host kernels are running at that moment and
  retries later, and only ever targets the upload it was scheduled for.
  Before this the refresh pulled the
  card's ~14 GiB back into the page cache every 64 steps, and each decode
  step re-advised the previous step's experts whether or not they were on
  the device.  `JOSHUA_EXPERT_HOST_PAGES=keep` restores the old, inclusive
  behaviour in full — pages kept, device-resident experts advised again
  (for a host with RAM to spare, or to bisect).

Flags (all also environment variables, `JOSHUA_…`):

| Flag | Effect |
|---|---|
| `--expert-placement device` | On OpenCL for `deepseek4`: run the routed experts from the VRAM cache, sized as `--vram-expert-cache auto` unless a budget is given. |
| `--vram-expert-cache auto` / `<MiB>` | The cache budget.  `auto` is device memory − dense set (only when the dense set is on the device) − 1 GiB placement headroom − scratch (512 MiB with the dense set on the CPU, 1 GiB with it on the device) − 1 GiB KV reserve (only with the dense set on the device).  `0` / unset disables the cache. |
| `--dense-placement cpu` | Keep the dense set on the CPU and use the card for the expert cache alone (the largest budget: ~14.5 GiB → ~2,200 slots on a 16 GB card). |
| `--prefill-chunk <tokens>` | Tokens per prefill chunk (default 512).  The MoE loaders stream a prefill layer by layer, so each layer's weights are read once per prefill whatever the chunk; the chunk sets how many prompt rows each resident expert batches per launch and the per-layer workspace. |

The load log prints the budget and slot count; every 64 decode steps a
`debug` line reports hits, misses, uploads, evictions, resident bytes and
the host pages released (`RUST_LOG=joshua=debug`), followed by the
**decode time split**: per step, how long the resident experts' launches
took to enqueue, how long the host misses ran, how long the step then
waited for the device, and the rest (attention, dense set, routing); and
for the host misses, what share of their pages were resident in RAM before
they ran — the difference between a miss served from the page cache and one
read from the disk.  A prefill logs the same split once at its end.  The
page probe (`mincore` per host miss) is on whenever the `joshua` `debug`
filter is active at load, or with `JOSHUA_EXPERT_STATS=1`.
`JOSHUA_EXPERT_MISS=upload` switches decode misses to a synchronous upload
on the calling thread followed by a device run — a measurement mode that
warms the pool fastest at the cost of stalling the token for the transfer.

### Reading the time split

* **host experts** dominates and the pages were mostly *not* resident:
  the disk is the clock.  More RAM, a faster disk, a smaller expert
  quantization, or a larger device budget (a higher hit rate) are the
  levers; nothing in the kernels is.
* **host experts** dominates with the pages resident: the CPU expert
  kernels are the clock, and a higher hit rate is the lever.
* **device wait** dominates: the card is the clock (few misses, the device
  kernels slower than the host would have been).
* **rest** dominates: attention and the dense set — the dense placement
  question, not the expert cache.

### Routing trace and the offline cache simulator

`JOSHUA_ROUTE_TRACE=trace.csv` writes every `(call, phase, chunk, row,
layer, expert)` visit of a run to a CSV (one file per process: trace one request
at a time, since concurrent sessions interleave their calls).  `cargo run --release --example cache_sim
-- trace.csv --slots 2067 --host-slots 7400` replays it against plain LRU,
the loader's LRU with the protected hot set, a static most-frequent
placement and Belady's optimal policy at that slot count — the hit-rate
ceiling for the observed routing, so a pinning-budget tweak (`--hot-share
n/d`) is judged offline before a run.  With `--host-slots` (RAM available
for experts ÷ expert size) it also replays the two tiers together and
reports the visits per decode step served by the device, by RAM and by the
disk, for exclusive and for inclusive tiers.

Numbers to check on the card rather than assume: the host-to-device
bandwidth from mapped pages (the per-upload time is in the debug line;
6.75 MiB should take well under a millisecond on PCIe 5.0 x8), whether the
runtime over-subscribes silently past 16 GB (the budget is byte-counted for
that reason; keep `auto`'s headroom), and the hit rate the counters show
for a real prompt — that is what decides how much of the expert block the
card carries.

### Finding a wrong result on one driver

A result that is NaN (or plainly wrong) only on the hardware and correct on
pocl is bisected without recompiling:

1. `JOSHUA_OPENCL_NATIVE=0` — every operator through the CPU round-trip.
   Correct now?  A kernel is at fault; otherwise look elsewhere.
2. `JOSHUA_OPENCL_NATIVE_DENY=k_gemm,k_qgemv` — a comma-separated list of
   *kernel* names (as in `kernels.cl`) that are refused, so only the ops
   that wanted them take the CPU path.  Halve the list until one kernel
   remains.  `JOSHUA_OPENCL_TRACE=1` shows which ops fell back.
3. `JOSHUA_OPENCL_CHECK_NAN=1` — after every native launch that produces an
   f32 tensor, count the NaNs on the device and report the first op whose
   output has any (a sync per op; infinities are not reported, mask fills
   and reduction identities are legitimate).
4. `JOSHUA_OPENCL_BUILD_OPTS="-cl-opt-disable"` — extra options for the
   kernel compiler.  A result that changes with the optimiser is a compiler
   issue, not a kernel bug.
5. `JOSHUA_OPENCL_QGEMV=v1` — the expert formats (IQ2_XXS, Q2_K) normally
   run `k_qgemv_mr`, which decodes each weight once for up to 16 rows with
   every lane owning eight consecutive elements; this switches them to the
   one-row `k_qgemv` every other block format uses.

The per-visit device cost of an expert is what the cache's hit rate buys,
so the kernel matters as much as the residency: with `k_qgemv_mr` a
resident IQ2_XXS `[2048, 4096]` matmul agrees with the fused CPU kernel to
~2e-6 (`iq2_opencl_tests` prints the timings next to the CPU's; run it on
the card for the real numbers).

## Environment variables

| Variable | Effect |
|---|---|
| `JOSHUA_OPENCL_NATIVE=0` / `JOSHUA_VULKAN_NATIVE=0` | Run every operator through the CPU round-trip (a correctness reference). Native kernels are on by default. |
| `JOSHUA_OPENCL_TRACE=1` / `JOSHUA_VULKAN_TRACE=1` | Log each operator that still falls back to the CPU and why. |
| `JOSHUA_OPENCL_ZERO_COPY=0` | Upload weights instead of aliasing the mapped file (for a driver that copies anyway). |
| `JOSHUA_OPENCL_NATIVE_DENY=k_a,k_b` | Refuse the named kernels so their ops take the CPU path (bisecting a bad result on one driver). |
| `JOSHUA_OPENCL_CHECK_NAN=1` | Count NaNs on the device after every native f32 launch and name the first op that produced one. |
| `JOSHUA_OPENCL_BUILD_OPTS="…"` | Extra options for the OpenCL kernel compiler (e.g. `-cl-opt-disable`). |
| `JOSHUA_EXPERT_MISS=upload` | Decode misses of the VRAM expert cache upload synchronously and run on the device (measurement mode). |
| `JOSHUA_EXPERT_HOST_PAGES=keep` | Keep an uploaded expert's host pages instead of releasing them (inclusive tiers; the default releases them). |
| `JOSHUA_EXPERT_STATS=1` | Probe each host miss's page residency for the decode time split even without a `debug` log filter. |
| `JOSHUA_ROUTE_TRACE=<path>` | Write the routing trace CSV for the offline cache simulator (`examples/cache_sim.rs`).  One file per process: run a single request at a time while tracing, or concurrent requests interleave their calls. |
| `JOSHUA_PREFILL_CHUNK=<n>` | Tokens per prefill chunk (also `--prefill-chunk`). |
| `JOSHUA_OPENCL_QGEMV=v1` | Run the expert formats through the one-row quantized GEMV instead of the multi-row kernel (bisecting). |
| `JOSHUA_VULKAN_QGEMV=mali` / `generic` | Force the Mali-shaped quantized GEMV (the default on Arm GPUs) or the generic one on any Vulkan device. |

The engine logs the device it opened, its memory model and the active paths
at startup:

```text
INFO joshua: OpenCL device: Intel(R) UHD Graphics 730 (58000 MiB global memory, host-unified memory; native kernels on, zero-copy weights on)
INFO joshua: Vulkan device: AMD Radeon Graphics (RADV RENOIR) (host-unified memory; native kernels on)
```

## Limits and fallbacks

* Tensors stay below 2³¹ elements (kernel indices are 32-bit).
* A Vulkan kernel cannot bind a buffer larger than the device's
  `maxStorageBufferRange`; the op, quantized matmuls and embedding gathers
  included, then falls back to the CPU path.  Real drivers report 4 GiB;
  llvmpipe reports 128 MiB.
* The quantized ops (matmul, embedding gather) follow the same contract as
  every other operator: with `JOSHUA_*_NATIVE=0`, or when the device
  rejects a launch, the block bytes and the activation come back to the
  host, candle's CPU kernel runs and the result goes up.
* An out-of-range id in `index_select`, `gather`, `scatter`, `index_add` or
  an embedding gather cannot raise an error inside a kernel.  The kernel
  skips the element and sets a fault word instead, which the host checks
  and clears at the next read-back or `synchronize` and reports as an error
  there.  Nothing is read or written out of bounds and a bad id never
  becomes a plausible result; the error surfaces at the point where the CPU
  backend's error for the same input would have been observed.  The fault
  buffer holds one word per thread (a request's launches and read-backs
  run on the same thread), so concurrent requests sharing the device never
  see each other's faults; a thread's word is drained (pending work
  completed, word cleared) on every device before it is recycled.
* F16 / BF16 embedding tables are gathered row by row by a half-precision
  kernel, like the block-quantized ones; no table is ever expanded to f32.
* naga's GLSL front end has no atomics, so the Vulkan scatter kernels walk
  the scattered dimension sequentially per output position (deterministic,
  same result as the CPU loop), and 1- and 2-byte outputs are written a
  whole word per invocation.
* The OpenCL device search accepts GPU, then ACCELERATOR, then CPU device
  types, so a CPU runtime such as pocl serves for testing.

## Testing

Both backends have the same parity tests, which skip cleanly without a
device:

```sh
cargo test -p candle-core --features opencl --lib -- opencl     # every op, every quantized dtype, zero-copy
cargo test -p candle-core --features vulkan --lib -- vulkan     # every op, every quantized dtype, allocator churn
cargo test --features opencl,vulkan --test opencl_model_tests --test vulkan_model_tests
cargo test --features opencl --test iq2_opencl_tests --test opencl_ds4_cache_tests
```

The model tests run the tiny llama / qwen3moe / deepseek2 / deepseek4 GGUFs
on the device and compare prefill and decode logits with the CPU; they also
report the number of native launches and fallbacks (zero for all four).
`iq2_opencl_tests` checks the IQ2_XXS device matmul against the fused CPU
kernel at the real expert shapes and prints resident-weight timings;
`opencl_ds4_cache_tests` runs the tiny deepseek4 through the VRAM expert
cache under a three-slot budget (dense set on the device and on the CPU,
background and synchronous uploads) and asserts parity, the pool's
counters and that no expert op fell back to the CPU.  In CI they run on
pocl (OpenCL) and llvmpipe (Vulkan).

`cargo run --release --features opencl,vulkan --example bench_backends --
--device opencl` times the quantized decode / prefill matmuls and the f32
GEMMs on a device.
