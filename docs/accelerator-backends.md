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
both backends (see `--expert-placement`): the dense set is what the iGPU
speeds up, and on unified memory the two share DRAM either way.  The dense
set itself is placed by the startup probe (`--dense-placement auto`), which
now times the *quantized* matmul the loaders actually run — a Q4_K weight
against f32 activations on the CPU and on the device — rather than an f32
GEMM, so it measures the path a model takes.

## Environment variables

| Variable | Effect |
|---|---|
| `JOSHUA_OPENCL_NATIVE=0` / `JOSHUA_VULKAN_NATIVE=0` | Run every operator through the CPU round-trip (a correctness reference). Native kernels are on by default. |
| `JOSHUA_OPENCL_TRACE=1` / `JOSHUA_VULKAN_TRACE=1` | Log each operator that still falls back to the CPU and why. |
| `JOSHUA_OPENCL_ZERO_COPY=0` | Upload weights instead of aliasing the mapped file (for a driver that copies anyway). |

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
```

The model tests run the tiny llama / qwen3moe / deepseek2 / deepseek4 GGUFs
on the device and compare prefill and decode logits with the CPU; they also
report the number of native launches and fallbacks (zero for all four).
In CI they run on pocl (OpenCL) and llvmpipe (Vulkan).

`cargo run --release --features opencl,vulkan --example bench_backends --
--device opencl` times the quantized decode / prefill matmuls and the f32
GEMMs on a device.
