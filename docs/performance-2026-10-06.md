# CPU inference performance, 2026-10-06

## Environment and scope

Measurements and tests ran over SSH on Kleya: AMD Ryzen 9 9950X,
16 physical cores / 32 logical CPUs, 73 GiB RAM, Linux x86-64,
Rust 1.98.1. CPU inference used the default release build and runtime
AVX-512 dispatch, with f32 activations and quantized mmap weights.
The diagnostic configured Candle's global Rayon pool with 16 threads;
Joshua's separate quantized-matmul pool used the 32 available CPUs.
No native CPU build flags or approximation changes were used.

Baseline: commit `97a4da0eb3d489295948ef0714942486d4a2893b`.
The remote user's existing checkout and its uncommitted changes were left
intact. Experiments ran in `/home/tserica/joshua-perf-20261006`; an unchanged
source copy used for failure reproduction lives in
`/home/tserica/joshua-perf-baseline-20261006`.

Models were existing substantial models on Kleya:

| Model file | Bytes |
| --- | ---: |
| Qwen3.8-27B-Q4_K_M.gguf | 17,106,775,008 |
| DeepSeek-V4-Flash-0731-reap-150b-Q2_K.gguf | 62,394,667,168 |

These are direct model-forward measurements, not HTTP request benchmarks.
Load, tokenization, and logit file I/O are excluded from inference timings.
Every repeat starts with an empty KV cache. Prompt tokens come from real
text and code, cycled and truncated to the requested length. Decode follows
greedy model outputs, rather than feeding a constant token or fixing routes.

## Hot paths and changes

The traced path is mmap load → token embedding → per-layer attention and
feed-forward/MoE → output projection → greedy sampling → cache continuation.

* Fused quantized matmuls previously scanned every prompt activation row
  for each weight row. Visiting a small activation tile across each worker's
  contiguous weight rows improves private-cache reuse without changing any
  output's block, FMA, or reduction order.
* AVX-512 row kernels now have constant one-row and full-register-tile cases.
  Decode avoids checks for unused activation rows; full tiles avoid repeated
  live-row checks.
* CPU MoE prefill previously multiplied into temporary tensors and copied
  the entire accumulated output for each active expert via `index_add`.
  The shared CPU helper scatters weighted rows into one owned output buffer,
  retaining ascending expert order, route order, duplicate routes, and
  separate multiplication/addition. DeepSeek-V4 retains its tensor-major
  gate/up/down traversal. Accelerator tensor accumulation is unchanged.
* Q6_K previously dequantized each complete weight row into an f32 scratch
  buffer before the SIMD dot. Its new AVX-512 kernel decodes blocks in
  registers, with the same `(d * signed_scale) * signed_value` arithmetic
  and 16-lane FMA/reduction order. AVX2, NEON, and scalar Q6_K retain their
  original fallback. Weights remain compact and borrowed.

No persistent f32 weight copies, per-session model copies, new retained
scratch buffers, or changes to sampling/model configuration were introduced.

## Results

Each row reports the second of two consecutive repeats (warm weights,
fresh KV cache). These are development measurements, not confidence intervals.

| CPU workload | Baseline prefill | Updated prefill | Throughput gain | Baseline decode | Updated decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| 27B dense, 128 prompt + 16 decode tokens | 10.218 s | 8.257 s | 1.24× | 2.140 tok/s | 2.264 tok/s |
| 27B dense, 512 prompt + 8 decode tokens | 46.381 s | 33.404 s | 1.39× | 2.137 tok/s | 2.245 tok/s |
| 150B MoE, 128 prompt + 16 decode tokens | 16.056 s | 12.598 s | 1.27× | 1.944 tok/s | 1.916 tok/s |

There is no measured DeepSeek decode gain. Dense-model decode improves
approximately 5–6%. Peak RSS for the 512-token dense run was 15.747 GiB
before and 15.757 GiB after (a 0.07% difference); RSS includes resident mmap pages.

The initial traversal-only change gave 1.14× warm dense prefill at 128
tokens and 1.16× MoE prefill. CPU accumulation and fused Q6_K provided the
additional gains above. Direct hardware half-scale conversion did not
provide a consistent gain and was removed. Restricting the unchanged code
to 16 physical cores regressed dense decode to about 1.79 tok/s and was
discarded.

The cache tile sweep used the same 27B model, 128-token prompt, greedy
decode, and full-logit parity check for every variant. Warm prefill times
were 9.854 s for two rows, 8.257 s for four, 8.634 s for eight, and 8.986 s
for sixteen. Four rows was retained and also improved the 512-token dense
and 128-token MoE workloads above. The register tiles inside kernels remain
architecture specific; this bounds the outer activation-cache tile.

## Correctness and checks

Every captured f32 logit after prefill and every decode step matches the
unchanged baseline byte for byte, including after cache resets. Greedy
token IDs match as well. Logit capture SHA-256 values for the 128-token runs:

* Dense: `d85c55055f79071c583c0f44f95387e3db2d0c90db1d6e3dfcd055bffce5a85c`.
* MoE: `46cbc57202f7bc89b60684c5d03cadc7ad82b7d4d67a170db96e0569402e5ad4`.

The release suite passed 476 tests, with 10 existing ignored tests. It includes
tiny GGUF reference parity, prefill/decode, cache continuation, batched
sequence inference, and speculative all-row logits. New tests compare
distinct activation rows across partial/full tiles, Q6_K against the previous
dequantized-row dot bit for bit, and CPU MoE scatter against tensor
accumulation with duplicate routes and offset/strided output views.
The forced AVX2 library and allocation suites passed 307 tests. The Q6_K
allocation test requires zero dequant scratch allocations on AVX-512 and
retains the exact per-chunk scratch count on fallback paths.

Five OpenCL model-parity tests passed on the Intel Arc Pro B50, including
DeepSeek-V4 long-prompt parity. The separate
`deepseek4_vram_expert_cache_matches_cpu` test fails at
`tests/opencl_ds4_cache_tests.rs:202`: uploaded experts report
`host_releases: 0`. The same failure was reproduced with freshly compiled,
unchanged baseline source. This cache-accounting failure remains unresolved;
it was not skipped or reclassified as a passing test.

One initial library run failed because a test creates a fixture under the
relative `target/` directory while the experiment used an external Cargo
target directory. This was reproduced on unchanged source. Creating the
expected directory fixed the test environment, after which the suite passed.

ARM, CUDA, Metal, SYCL, other CPU hardware, HTTP concurrency, much longer
contexts, and accelerator throughput remain unmeasured. OpenCL checks above
establish parity, not a GPU performance gain. These results establish the
best measured changes in this experiment, not universal optimality.

## Reproduction

Build the diagnostic on the machine that will run inference:

```sh
mkdir -p target
cargo build --release --example perf_inference
target/release/examples/perf_inference \
  /path/to/model.gguf /path/to/tokenizer.json /path/to/prompt.txt \
  512 8 2 updated-logits.bin 16
cmp baseline-logits.bin updated-logits.bin
cargo test --release
JOSHUA_SIMD=avx2 cargo test --release -p joshua --lib --test simd_chunk_alloc
cargo test --release --features opencl --test opencl_model_tests -- --nocapture
```

The exact experiment prompt and raw timing/test logs are preserved in the
remote experiment directory. The prompt asks about Rust inference memory
traffic, quantized MoE weights, and correctness of a weighted-sum function,
then requests discussion of duplicate routes, cache continuation, and
speculative verification. No original benchmark workload was changed.
