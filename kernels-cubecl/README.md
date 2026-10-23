# Rust-authored packed GEMVs for native Vulkan

The kernel is authored in `src/lib.rs` using CubeCL 0.11.0. Its generated WGSL
is checked in under `vendor/candle-core/src/vulkan_backend/` as
`q4k_cubecl.wgsl`, `q8_0_cubecl.wgsl` and `q2k_cubecl.wgsl`.
The additional kernels also provide `q6k_cubecl.wgsl` and
`grouped_q8_cubecl.wgsl`.
Candle compiles that artifact with its existing Naga dependency and runs it
through its existing Vulkan runtime. **Production does not use wgpu, link
CubeCL, require its toolchain, import buffers, or create a second queue.**
This standalone development crate remains outside Joshua's workspace.

## Full-model B50 acceptance (2026-10-22)

Q6_K and grouped Q8_0 passed the full-model gates and are now included in the
**B50-only** default, `CubeClNext`. The absolute tolerance remains **1e-5**.
[The saved report](benchmarks/q6-grouped-q8-b50-20261022.json) includes commands,
source/binary hashes, full-logit hashes, timings, memory bounds and failures.

Both paths used the complete realistic chat prompt and the same 128
teacher-forced decode inputs, with two serial sessions per path and host
logits readback included. Baseline `CubeClExpanded` retains generic Vulkan
for the two tested operations; the earlier Q4_K/Q8_0/Q2_K kernels are identical
between paths. Qwen's only newly selected operation is Q6_K; DeepSeek has no
Q6_K tensors, so its comparison isolates grouped Q8. No other benchmark or
build ran during timed forwards.

| Model / new kernel | Prompt / prefill chunk | Baseline decode | New decode | Decode speedup | Prefill + decode speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-8B Q4_K_M / Q6_K | 126 tokens / 1 | 490.617 ms/token | 192.998 ms/token | **2.542x** | **2.558x** |
| DeepSeek-V4.1 Flash REAP-272E / grouped Q8 | 121 tokens / 32 | 3739.567 ms/token | 803.847 ms/token | **4.652x** | **3.690x** |

All **19,599,744 Qwen logits** and **16,677,120 DeepSeek logits** are
bit-identical between paths (maximum difference **0**); repeated-session
logits are also identical. Every candidate decode/session timing sample beat
every baseline sample. There were zero CPU fallbacks. Engram stayed enabled
and DeepSeek experts remained compact CPU-mmap weights, shared across sessions.

Allocator trajectories and loaded/warm/peak/after-session counts match exactly.
Driver-resident VRAM endpoints are flat at **5,268,217,856 bytes** for Qwen
and **9,538,830,336 bytes** for DeepSeek, for both paths and both sessions.
Qwen private-memory growth is 0 KiB in each path. DeepSeek private growth is
24 KiB baseline and 20 KiB candidate, including swapped anonymous pages.
The candidate's private-memory footprint stays within the existing 1 MiB
bound against baseline. DeepSeek physical reads were 194.44/135.26 GB
for the baseline sessions and 133.72/125.88 GB for candidate sessions; these
are workload timings including I/O, not isolated GPU compute measurements. File-backed RSS is reported separately; this test ran
on a roughly 74 GiB host and includes physical mmap I/O, not a 64 GiB host
acceptance claim. The paths are serial separate processes, not interleaved
passes, and HTTP serving/concurrent requests were not measured.

Qwen's baseline 32-token prefill attempt failed with `ENOMEM` followed by
`ERROR_DEVICE_LOST` (repeated while resuming the comparison tool). The
successful Qwen comparison uses single-token prefill for the full prompt;
this does not validate batched Qwen prefill. DeepSeek completed batches
32/32/32/25. The failed baseline is retained in the report. No prefill policy
or backend priority changed.

Q6_K reads the original 210-byte blocks and preserves generic Vulkan's
256-lane reduction and multiplication order. Grouped Q8_0 computes only the
diagonal output blocks: output row `r` reads activation group
`r / rows_per_group`, and writes the final projection directly. It avoids
off-diagonal results and their concatenation. Single-token projection with
at most 16 groups is eligible; other paths retain the existing arithmetic.
Buffers, queue, parameter ring and deferred release stay in the native runtime.

`--kernel cubecl` retains the previous three-format baseline;
`--kernel cubecl-next` explicitly selects the newly validated set, and
`--kernel production` verifies the actual adapter default. Other adapters
retain their defaults until separately validated. For actual-weight grouped
microbenchmarks, use `--projection-tensor blk.0.attn_output_a.weight
--projection-rows 8 --projection-grouped`. Full outputs and raw memory samples
are retained outside `/tmp` at
`/home/tserica/benchmark-results/cubecl-b50-20261022-pr198`.

After enabling B50 dispatch, another full-prompt/128-token session per model
through `--kernel production` reproduced the candidate's full-logit hash and
exact allocator trajectory, with zero CPU fallbacks. The seven native Vulkan
tests (two timing tests ignored), eleven focused Candle Vulkan tests and
release benchmark build passed after the dispatch change. The production
replays verify routing; repeated-memory acceptance comes from the two-session
matched comparisons above.

`JOSHUA_NO_ENGRAM` is a diagnostic opt-out that changes model behavior; it
was unset for these comparisons. `examples/dump_compressors.rs` inspects
conversion metadata and filtered tensor names without loading weights.

## Dispatch and interoperability

On the validated Intel Arc Pro B50 (PCI vendor/device 8086:E212), the default
Vulkan QGEMV selection is now **`CubeClNext`**: native Q4_K/Q6_K/Q8_0/Q2_K
and grouped Q8, all with generic's 256-lane accumulation order. The full-model
checks above passed parity, speed and memory gates.
`JOSHUA_VULKAN_QGEMV=generic` forces the baseline; `cubecl` keeps a Q4_K-only
compatibility selection; `cubecl-expanded` selects the previous three-format
baseline, and `cubecl-next` explicitly selects the current production set.
Small Q2_K projections (fewer than 64 output rows) retain generic dispatch
after a measured regression at DeepSeek’s 24-row shape.
Other formats, batched rows, unsupported widths, excessive dispatch
counts, and devices with insufficient workgroup limits retain the generic
kernel. Other adapters retain their previous defaults. For comparison or
explicit experimentation, `JOSHUA_VULKAN_QGEMV=generic|cubecl|cubecl-expanded|cubecl-next` selects a
path; existing `mali` and `mali-int8` selectors remain available.

The native launcher binds the **same** packed weight buffer, activation
buffer, and output buffer as generic QGEMV. Its fourth binding is a 16-byte
parameter record in Candle's existing parameter ring, not a new allocation.
It records into the same command buffer, uses the same compute barriers and
batch limit, and inherits deferred buffer release. One cached pipeline per format
handles every supported N/K; there is no per-shape or per-session pipeline
cache growth. Weight blocks retain their original GGUF layouts: Q4_K 144 bytes/256 values,
Q8_0 34 bytes/32 values, Q6_K 210 bytes/256 values and Q2_K 84 bytes/256 values.
Scratch is 1,024 bytes of workgroup memory per workgroup; there is no f32 weight matrix,
weight repacking, tensor copy, readback, or additional fence in the launcher.

The baseline already fuses dequantization and GEMV. The improvement is in the
Rust-authored kernel's generated code/workgroup configuration, not elimination
of a previously separate dequantization pass. The experimental CubeCL runtime
microbenchmark remains in `tests/gemv.rs` for comparison; its separate-runtime
results must not be confused with the native integration below.

## Acceptance results (2026-10-17)

Intel Arc Pro B50 (BMG G21), Mesa 26.0.8, Vulkan 1.4.335, Rust 1.98.1, release
build. Full model forward and host-logit retrieval on a **synthetic Qwen3**
GGUF: four layers, hidden width 512, FFN width 1536, four-token prefill followed
by 32 fixed decode tokens. Seven passes alternate the timing order. Load,
prefill, session construction and initial shader compilation are outside the
decode timer. Tokenization, sampling and HTTP serving are not timed.

| Native dispatch | Median ms/token | Tokens/sec |
| --- | ---: | ---: |
| Generic GLSL | 12.610 | 79.3 |
| CubeCL-generated WGSL | 1.860 | 537.7 |

This run measured **6.78x** faster decode. It is not a real-model throughput
claim, and absolute timings vary across runs.

Both paths used exactly **134,217,728 reserved / 5,520,640 live allocator
bytes** after load/warmup and after every repeated session. Process RSS stayed
at **76,760 KiB** after warmup and after seven sessions. The RSS test allows
1 MiB of host allocator noise; the device allocator comparison is exact.
Driver-private pipeline memory is not included in allocator snapshots. The
shader pipeline and existing parameter ring are fixed, cached resources.

Acceptance checks cover:

- Direct native-buffer parity, nonzero activation offsets, queued native
  producers/consumers, dropped buffers, and the 1,024-dispatch batch boundary.
- Qwen3 and Qwen3-MoE with heap and mmap loads; DeepSeek4 with its supported
  mmap-backed CPU experts and Vulkan dense weights. Prefill, decode, cache
  continuation, and speculative all-position logits match generic Vulkan
  within absolute `1e-5`, with matching argmax and no additional CPU fallbacks.
  DeepSeek4's heap raw-expert upload is intentionally unsupported by its loader.
- The Candle Vulkan kernel suite, including QGEMV variants over all supported
  quantized formats and batched row counts (CubeCL falls back for unsupported formats and batched rows).
- Naga 27 parsing, validation, SPIR-V emission and reflection: only bindings
  0..=3 are used; weight, activation and parameter buffers are never written.
- Repeated shared-weight sessions: exact device allocator equality and bounded
  host RSS, with all benchmark token logits compared against the baseline.

CPU, CUDA, Metal, OpenCL and SYCL execution are unchanged. Further hardware
needs parity, memory and decode measurements before expanding default dispatch.

## Reproduce

From the repository root on the Vulkan host:

```sh
cargo test --release --features vulkan --test cubecl_vulkan_tests -- --nocapture --test-threads=1
cargo test --release --features vulkan --test cubecl_vulkan_tests benchmark_decode -- --ignored --nocapture --test-threads=1
cargo test --release --features vulkan --test vulkan_model_tests -- --test-threads=1
cargo test --release -p candle-core --features vulkan --lib vulkan_backend -- --test-threads=1
cargo test --manifest-path kernels-cubecl/Cargo.toml
```

Hardware acceptance tests require Vulkan and fail if it is unavailable. The
benchmark explicitly selects generic and candidate kernels on the same GPU.
Run GPU timing serially without competing GPU tests.

To regenerate/verify the artifact (development only; currently initializes a
CubeCL client on an available GPU to obtain compiler properties):

```sh
cargo run --release --manifest-path kernels-cubecl/Cargo.toml --bin generate -- --output vendor/candle-core/src/vulkan_backend/q4k_cubecl.wgsl
cargo run --release --manifest-path kernels-cubecl/Cargo.toml --bin generate -- --check vendor/candle-core/src/vulkan_backend/q4k_cubecl.wgsl
cargo clippy --manifest-path kernels-cubecl/Cargo.toml --all-targets -- -D warnings
```

The generator normalizes the entry point to `main` and spells function-local
variables with their implicit address space for compatibility with Naga 27.
No arithmetic is edited in generated WGSL. The reduction tree is explicitly
unrolled: mutating an inferred compile-time stride in a runtime while loop
produced invalid CubeCL 0.11 IR in the original prototype. Smaller workgroups
change accumulation order relative to generic QGEMV, hence the model parity
gate. The development toolchain was Rust 1.98.1; compatibility of CubeCL itself
with Joshua's Rust 1.89 minimum has not been established or made a build
requirement for Joshua.

## Suite status

The native interoperability/model integration tests passed (3 tests; timing is
an explicit ignored test), as did the existing Vulkan model suite (5), all
Candle core library tests with Vulkan enabled (31), the standalone CubeCL
suite, generated-artifact check, Clippy for the development crate, and the
default CPU library build check.

The broader Joshua library run reported 362 passed, 5 ignored and one failure:
`template::tests::engine_variables_win_over_template_kwargs` at
`src/template.rs:390` renders `<s>|true|1|false|ok` while expecting
`<s>|True|1|False|ok`. The same exact failure was reproduced on unchanged base
commit `661d808` in an isolated worktree with the same dependency lockfile:

```sh
cargo test --release --features vulkan --lib template::tests::engine_variables_win_over_template_kwargs -- --exact --nocapture
```

That unrelated template failure was not modified or omitted from the report.

## Real Qwen3-8B Q4_K_M benchmark (2026-10-18)

The local **Qwen3-8B-Q4_K_M.gguf** (5,027,783,488 bytes, SHA-256
`d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`)
was benchmarked on the B50 with one resident model per process. Its tensors
contain 6,586,630,144 Q4_K parameters, 1,603,796,992 Q6_K parameters, and
308,224 F32 parameters. Q6_K, attention, normalization and other operations
remain on their existing kernels; parameter proportions do not directly
measure their share of execution time.

A 16-token natural-language prompt was prefilled in **one-token chunks**,
followed by 32 decode tokens, over three warmed passes per kernel. Generic ran
first; CubeCL used its saved generated token sequence, so every timed input
and cache position matched. Timing includes model forward, host-logit
retrieval, finite checks and greedy argmax. It excludes loading, prefill,
statistics, repeat-output comparison and dump-file I/O. The kernels were not
alternated between passes, and no other GPU benchmark ran concurrently.

| Native dispatch | Median ms/token | Tokens/sec | Per-pass ms/token |
| --- | ---: | ---: | --- |
| Generic | 1730.162 | 0.578 | 1730.162, 1730.063, 1730.325 |
| CubeCL | 437.815 | 2.284 | 432.304, 437.815, 437.825 |

The real-model speedup was **3.95x**, compared with **6.78x** on the earlier
synthetic fixture. This mixed-quantization model does not accelerate every
projection with the Q4_K kernel; detailed per-kernel attribution was not
measured.

All **33/33 greedy choices** (prefill plus 32 decode steps) and decoded text
matched. Across **5,013,888 full-vocabulary logits**, max absolute difference
was **1.90735e-5**, RMS difference **1.82918e-6**, and maximum softmax
KL(generic || CubeCL) **3.68043e-12**. Repeated passes were bit-identical
within each kernel. Between kernels, the real-model maximum **does exceed**
the tiny-fixture absolute `1e-5` threshold; these results establish matching
greedy output and very small rounding differences, not bitwise equality or a
pass at that strict absolute threshold. No test tolerance was changed.

Both processes had exactly the same device allocator figures:

- Loaded/after every session: **5,227,622,400 reserved / 5,042,798,592 live
  bytes** (4.696 GiB live).
- Peak sampled at completed decode-token boundaries: **5,227,622,400 reserved /
  5,061,672,960 live bytes**. This is not a continuously sampled high-water
  mark and excludes driver-private pipeline allocations.
- Host RSS after each session: generic **174,492 / 175,676 / 175,676 KiB**;
  CubeCL **172,276 / 173,464 / 173,464 KiB**. The harness retains 20,055,552
  bytes of reference-logit float data for comparison. RSS plateaued after
  the second session; device allocation did not grow after any session.
- **Zero CPU fallbacks** in both prefill and decode for all completed passes.

Larger-batch prefill diagnostics are a separate limitation: an initial
64-token batch produced non-finite output; a traced retry hit
`ERROR_OUT_OF_DEVICE_MEMORY` in the full-weight dequantization/GEMM prefill
path and fell back to CPU. A 16-token batched attempt returned
`ERROR_DEVICE_LOST`. Those runs were excluded. The successful comparison uses
one-token chunks for prefill, so it does **not** validate default large-batch
prefill or claim its memory behavior is safe. No production kernel changes
were made during this real-model benchmark.

### Reproduce the real-model run

The reusable harness is `examples/bench_cubecl_real.rs`; it requires the
`vulkan` feature. Replace the local model/tokenizer paths as needed:

```sh
cargo build --release --features vulkan --example bench_cubecl_real
target/release/examples/bench_cubecl_real --model /home/tserica/models/qwen3-dense/Qwen3-8B-Q4_K_M.gguf --tokenizer /home/tserica/models/qwen3-dense/tokenizer.json --kernel generic --output /tmp/joshua-cubecl-real/generic --prompt-tokens 16 --prefill-chunk 1 --decode 32 --passes 3
target/release/examples/bench_cubecl_real --model /home/tserica/models/qwen3-dense/Qwen3-8B-Q4_K_M.gguf --tokenizer /home/tserica/models/qwen3-dense/tokenizer.json --kernel cubecl --output /tmp/joshua-cubecl-real/cubecl --reference /tmp/joshua-cubecl-real/generic/trace.json --prompt-tokens 16 --prefill-chunk 1 --decode 32 --passes 3
python3 kernels-cubecl/benchmarks/compare_real.py /tmp/joshua-cubecl-real/generic /tmp/joshua-cubecl-real/cubecl /tmp/joshua-cubecl-real/comparison.json
```

Full summary metrics, inputs, greedy outputs, environment and limitations are
saved in `benchmarks/real-qwen3-8b-b50-20261018.json`. The comparison script uses
NumPy. Raw float dumps remain under `/tmp/joshua-cubecl-real-20261018/` rather
than being added to the repository. Measurements used Linux 7.0.0-38-generic,
Mesa 26.0.8 and Rust 1.98.1.

## Initial DeepSeek Flash expansion (before the exact-reduction fix)

The local `DSV41-REAP272E-balanced.gguf` identifies itself as **DeepSeek-V4.1
Flash**, 40 layers and 272 routed experts, 290,070,476,416 file bytes. This is
REAP-272E, not the original unpruned V4 Flash checkpoint. Header inspection
shows Q8_0 dense attention/output projections, Q4_K shared/routed experts,
Q2_K hyper-connection projections and engram tables, and F16 embeddings/router
weights. The huge engram tables stay CPU-mapped; they are gathers, not GEMVs.
Routed experts also stay compact CPU mmap views. This benchmark does not add
a Vulkan expert cache or upload the entire expert pool.

New Rust kernels specialize Q8_0 and Q2_K dequantization/dot products without
repacking weights. Q8_0 uses 256 lanes to match generic’s reduction order and handles unaligned
34-byte blocks; a guarded two-dimensional
launch supports the 129,280-row output head. All use the existing native queue,
parameter ring, descriptor bindings and deferred releases. Batched prefill
continues through existing dispatch. Small Q2_K projections retain generic
GEMV because the candidate regressed at the real 24-row hyper-connection shape.

Final projection microbenchmark, B50, release, seven alternating passes, ten queued
launches per timing with a final fence (initial shader compilation excluded):

| Format / shape (N × K) | Generic ms | CubeCL ms | Speedup |
| --- | ---: | ---: | ---: |
| Q8_0 query-A, 1,280 × 5,120 | 1.2835 | 0.1121 | 11.45× |
| Q8_0 query-B, 32,768 × 1,280 | 28.9049 | 2.2386 | 12.91× |
| Q8_0 output, 129,280 × 5,120 | 144.2129 | 11.6316 | 12.40× |
| Q2_K hyper-connection, 24 × 20,480 (generic fallback) | 0.0462 | 0.0462 | 1.00× |
| Q2_K larger projection, 512 × 4,096 | 0.4209 | 0.0541 | 7.78× |

Every timed projection matched generic within absolute `1e-5`; the allocator
snapshot was exactly unchanged by either kernel. These are synthetic weights
at real model shapes, not end-to-end throughput. The initial 64-lane Q2_K
candidate was 0.0545 ms versus generic 0.0492 ms at 24 × 20,480, which motivated
the small-projection generic fallback. Final Q8_0 uses 256 lanes; Q4_K/Q2_K use 64.

Run a complete real-model A/B benchmark with:

```sh
kernels-cubecl/benchmarks/deepseek_b50.sh \
  /home/tserica/models/DSV41-REAP272E-balanced.gguf \
  /home/tserica/models/DeepSeek-V4.1-Flash-REAP-272E/tokenizer.json \
  /tmp/deepseek-flash-b50
```

The script builds the release example, runs generic then expanded CubeCL in separate
processes, and compares all saved logits and memory snapshots. Defaults are
four prompt tokens, eight decode tokens, three timed sessions and two warmup
decode tokens. Append `--prompt-tokens`, `--decode`, `--passes`, `--warm-decode`
and `--prefill-chunk` options to change the workload. A populated result folder
is refused to avoid mixing runs. NumPy is required for the comparison script.

The example detects `deepseek4`/`deepseek41`, explicitly selects CPU experts,
validates the GGUF layout, and sizes the context to prompt plus decode/warmup
rather than the checkpoint's million-token limit. `--context` can increase it.
Engram stays enabled unless the caller explicitly sets the existing
`JOSHUA_NO_ENGRAM` override; its state is recorded in the result. Full dense
Vulkan forward plus CPU experts and host logits are timed. Repeated sessions
share model weights; load, warmup and prefill are reported separately.

Regenerate additional artifacts with:

```sh
GGML_DTYPE=8 cargo run --release --manifest-path kernels-cubecl/Cargo.toml --bin generate -- --output vendor/candle-core/src/vulkan_backend/q8_0_cubecl.wgsl
GGML_DTYPE=10 cargo run --release --manifest-path kernels-cubecl/Cargo.toml --bin generate -- --output vendor/candle-core/src/vulkan_backend/q2k_cubecl.wgsl
```

Expanded verification passed: native interoperability/offset/deferred-release
and tiled-output tests; tiny Q8_0/Q4_K Qwen3, Qwen3-MoE, DeepSeek4 and DeepSeek4.1
prefill/decode/cache-continuation/all-row parity; DeepSeek4.1 integration (7),
DeepSeek4 integration (21, two opt-in tests ignored), Vulkan model integration
(5), and Candle core library (31). No parity tolerance was loosened.

The completed [real DeepSeek result](benchmarks/real-deepseek41-flash-b50-20261018.json)
used the script defaults above, with engram enabled. Generic measured
**5,436.972 ms/token (0.184 tokens/sec)** versus expanded CubeCL
**3,449.972 ms/token (0.290 tokens/sec)**, a **1.576×** end-to-end speedup.
All nine greedy choices and decoded text matched; all 1,163,520 logits were
finite and repeat sessions were bit-identical within each mode. Maximum
cross-mode logit difference was **6.1035e-5** (RMS 7.0129e-6), exceeding the
existing absolute `1e-5` gate. Matching Q8_0's reduction order did not eliminate
the full-model difference. **At this stage Q8_0/Q2_K remained opt-in through
`CubeClExpanded` / `JOSHUA_VULKAN_QGEMV=cubecl-expanded`; the B50 default stays
Q4_K-only.** This status is superseded by the exact-reduction fix below. The benchmark's `--kernel cubecl` selects this expanded candidate.

Device allocator snapshots matched exactly: **9,365,831,680 reserved /
9,223,318,528 live bytes** after load and after every session, with identical
sampled live peaks of 9,233,920,000 bytes. Both modes reported zero Vulkan CPU
fallbacks during prefill and decode (CPU expert execution is intentional and
is included in the timing). Host RSS after sessions was generic
21,410,428 / 22,014,916 / 22,063,576 KiB, versus candidate
22,062,932 / 22,062,960 / 22,062,972 KiB. Generic paged more mmap-backed weights
in during its first sessions; sampled anonymous memory was roughly 180 MiB
in both modes. The candidate's after-session RSS varied by 40 KiB. This is
bounded shared-weight execution, not a private copy of the 270 GiB checkpoint.

The raw four-token prompt produced a repetitive short continuation in both
modes. These results establish throughput and cross-kernel parity limits,
not checkpoint quality or chat-template correctness. Longer contexts, larger
batched prefill, and a Vulkan routed-expert cache remain unmeasured here.
The initial 64-lane Q8_0 [smoke result](benchmarks/real-deepseek41-flash-b50-smoke-20261018.json)
is retained separately and does not describe the final 256-lane shader.

## Drift isolation and exact reduction fix (2026-10-18)

[Per-format ablations and projection stress results](benchmarks/deepseek-drift-isolation-20261018.json)
isolate the original DeepSeek drift to **Q4_K GEMV in the shared-expert gate/up
projections**. Using the same four-token prompt and first two saved decode
inputs (context 12), the original Q4_K-only candidate reproduced every logit
of the original expanded run, including the maximum error of 6.1035e-5.
Q8_0-only and Q2_K-only controls were bit-identical to generic. The checkpoint's
small Q2_K hyper-connection projections already use generic dispatch; its
routed experts stay CPU-mapped and never enter these Vulkan kernels.

The cause was arithmetic order: 64 lanes assigned multiple 32-weight
sub-blocks to each lane before the reduction, whereas generic uses 256 lanes.
This changes both the lane partial sums and the reduction tree for widths
above 2,048. Q2_K had the same latent issue at larger widths. Q8_0 was already
using generic's 256-lane order and did not cause the observed drift.

All native CubeCL artifacts now use **256 lanes**, matching generic's lane
assignment, sequential dot accumulation, partial accumulation and reduction
order. Devices whose generic workgroup is not 256 retain generic dispatch.
Packed layouts, buffers, parameter ring, queue, device placement and the
absolute **1e-5** model tolerance are unchanged. Shared scratch is now 1 KiB
per workgroup for each format. The development runtime API still allows other
workgroup sizes for experiments; those are not production artifacts.

The new `packed_gemv_preserves_generic_accumulation_order` regression requires
bit-identical output for Q4_K, Q8_0 and Q2_K at widths 1,280 / 5,120 / 8,192 /
20,480 with larger-magnitude weights and inputs. Before the fix, Q4_K differed
by up to 6.1035e-5 and Q2_K by up to 4.5776e-5; the fixed kernels pass exactly.
The real-model benchmark supports `--kernel cubecl-q4|cubecl-q8|cubecl-q2` for
future per-format ablations, alongside `generic` and expanded `cubecl`.

The runner now records anonymous versus file-backed host RSS, and requires
parity, faster decode, equal/stable device allocations, bounded anonymous
memory and no extra CPU fallbacks. Anonymous memory allows 1 MiB of allocator
noise (the existing host-memory budget); logit tolerance remains 1e-5.
`compare_real.py --require-pass` exits unsuccessfully when a gate fails.

Fixed projection timings on B50 (same seven alternating passes / ten launches
per timing, synthetic weights at real shapes): Q4_K shared gate/up 2,304 ×
5,120 measured 2.4960 ms generic versus 0.1619 ms CubeCL (**15.42×**); Q8_0
shapes measured **12.18–13.45×**; Q2_K 512 × 4,096 measured **6.68×**. The
small Q2_K hyper-connection shape keeps its generic fallback.
Historical 64-lane measurements above describe the earlier implementation,
not the corrected production artifacts.


The [fixed real DeepSeek acceptance run](benchmarks/real-deepseek41-flash-b50-fixed-20261018.json)
passed every gate: **all 1,163,520 logits are bit-identical**
(maximum absolute difference **0**, unchanged 1e-5 tolerance), all nine greedy
choices match, and all repeat sessions are bit-identical. Generic median
was **5435.129 ms/token** versus **3454.940 ms/token**
for corrected CubeCL, a **1.573×** decode speedup. This is the same
four-prompt / eight-decode / three-session workload, with engram enabled,
dense Vulkan weights and compact CPU mmap experts.

Both modes used exactly **9,365,831,680 reserved / 9,223,318,528 live device
bytes** after load and every session; sampled live peaks were also identical
at 9,233,920,000 bytes. Anonymous host memory grew only **28 KiB generic /
36 KiB CubeCL** across sessions; the candidate's peak anonymous memory was
336 KiB below generic. File-backed RSS stayed fixed within each mode. Both
reported zero Vulkan CPU fallbacks during prefill/decode.

**Production is enabled by default on B50 Vulkan (PCI 8086:E212)** through
`CubeClExpanded`. Other adapters, explicit environment overrides, unsupported
shapes and batched paths preserve their fallback behavior. The production
adapter-selection test now requires the expanded selection. No model-loader,
CPU expert-placement or engram override changes were needed.

Post-enable verification passed: B50 default-selector check; packed-buffer,
queue/deferred-release, tiled-output and exact-accumulation regressions;
Qwen3/Qwen3-MoE/DeepSeek4/DeepSeek4.1 prefill/decode/cache-continuation/all-row
parity (5 native integration tests); DeepSeek4.1 integration (7), DeepSeek4
integration (21, two opt-in tests ignored), Vulkan model integration (5),
and Candle core library tests (31). The standalone CubeCL tests and Clippy,
CPU library check and Vulkan benchmark-example check passed. Full logit-file
SHA-256 values are equal and recorded in the fixed-result JSON.

The broader Joshua library rerun again reported 362 passed / 5 ignored / one
failure: `template::tests::engine_variables_win_over_template_kwargs` at
`src/template.rs:390`, actual `<s>|true|1|false|ok` versus expected
`<s>|True|1|False|ok`. This is the exact same unrelated failure previously
reproduced on unchanged base `661d808` with the same lockfile (see Suite status
above); the template code was not changed. Production parity/speed/memory
gates all pass independently of that existing assertion failure.

## Long batched-prefill validation workload

The longer workload uses the complete
[`deepseek-realistic-prompt.txt`](benchmarks/deepseek-realistic-prompt.txt),
rendered through the checkpoint's GGUF chat template: **121 prompt tokens**,
8-token prefill batches, **128 decode steps**, two measured sessions per mode,
and context 249. It crosses the 128-token sliding attention window. Warmup
uses an 8-token prefix and two decode steps; the timed sessions always use
the full prompt. The candidate selects the actual B50 production default.

```sh
JOSHUA_BENCH_CANDIDATE=production kernels-cubecl/benchmarks/deepseek_b50.sh \
  /home/tserica/models/DSV41-REAP272E-balanced.gguf \
  /home/tserica/models/DeepSeek-V4.1-Flash-REAP-272E/tokenizer.json \
  /tmp/deepseek-flash-b50-long \
  --prompt-file kernels-cubecl/benchmarks/deepseek-realistic-prompt.txt --chat \
  --prefill-chunk 8 --decode 128 --passes 2 \
  --warm-prompt-tokens 8 --warm-decode 2
```

The harness saves memory samples at every prefill batch and decode token,
including device reserved/live bytes, resident anonymous memory, swapped
anonymous memory and file-backed RSS. Heavy mmap paging can swap cold heap
pages out, so anonymous-memory stability counts **RssAnon + VmSwap**, with
the same 1 MiB allocator-noise budget. File-backed weight residency is reported
separately. Reference-logit bytes are recorded: 129 full-vocabulary rows retain
66,708,480 bytes in the harness after the first pass, shared by both modes.
The samples identify that deliberate storage separately from model memory.
Repeated-session parity now includes the final prefill logits as well as every
decode row, still at absolute 1e-5. Two-pass medians use the mean of the two
middle values; prefill and total-session latency are reported separately.

The completed [long-run result](benchmarks/real-deepseek41-flash-b50-long-20261018.json)
passed all acceptance checks. The 16,677,120 compared logits (final prefill
row plus all 128 decode rows) are **bit-identical**, with equal SHA-256 files
and maximum absolute difference **0** under the unchanged **1e-5** gate.
All 129 greedy choices match, and both repeated sessions have maximum logit
difference 0, including the repeated prefill result. Both modes report zero
Vulkan CPU fallbacks during prefill and decode.

| Two-session median / mean | Generic | Production CubeCL | Speedup |
| --- | ---: | ---: | ---: |
| Full 121-token prefill | 285.56 s | 276.96 s | 1.03× |
| Decode | 5.470 s/token | 3.459 s/token | **1.58×** |
| Prefill + 128 decode steps | 985.67 s | 719.70 s | **1.37×** |

The complete device-memory trajectories match at all 288 batch/token samples
per mode. Live device memory is **9,236,945,920 bytes** and remains constant
throughout all four 128-token decode sessions. After each session it returns
to exactly **9,224,831,488 live / 9,365,831,680 reserved bytes**, matching the
post-load allocation. Reserved memory never grows across the measured run.

Anonymous memory, counting resident plus swapped pages, grows only **24 KiB
generic / 8 KiB production** between the two session endpoints. Production's
peak after-session anonymous footprint is 340 KiB below generic. This passes
the same 1 MiB allocator-noise budget; the logit gate is unchanged.
File-backed RSS is different: mapped weight pages warm from 61.65 to 66.44 GiB
in generic and from 65.15 to 66.84 GiB in production. That is OS-managed mmap
weight residency, not a private model copy or growth in model-owned buffers.
The host must still accommodate the substantial weight working set.

The full prompt is realistic and uses the actual GGUF chat template, but the
baseline continuation repeats `1`, and production reproduces it exactly.
Thus this run validates long cache/window behavior and the measured parity,
speed and memory gates, while providing limited diversity in generated expert
routing. It does not establish checkpoint language quality. The raw token
trace and full logits remain outside the repository in
`/tmp/joshua-cubecl-deepseek41-long-20261018/`; the report retains the prompt,
per-token memory samples, timings, hashes and acceptance flags.

Only the benchmark harness/reporting changed for this validation: full prompt
input and rendering, actual production selection, bounded warmup prefix,
batch/token memory sampling, swapped-anonymous accounting, two-pass median,
and repeat-prefill parity. The benchmark release build and Vulkan example
check passed, the runner passed shell syntax validation, and `--require-pass`
still rejects the earlier real 6.1035e-5 drift result. No inference-kernel or
model-loader code changed in this validation.

## Cross-backend evaluation on the B50

`bench_cubecl_real` also accepts `--backend cpu|opencl|sycl|vulkan` and
`--auto-placement`. Fixed placement keeps DeepSeek routed experts borrowed
on CPU, dense weights on the selected backend, and disables the expert cache.
Automatic placement calls `Engine::with_options` and uses its resolved dense,
expert and cache settings; it does not reimplement the placement rules.
`--kernel production` is required for automatic placement.

Build with all three accelerator features, then run the serial evaluator:

```sh
cargo build --release --features vulkan,opencl,sycl --example bench_cubecl_real
# Initialize the installed SYCL environment and select its bridge as needed.
export JOSHUA_SYCL_LIBRARY="$PWD/sycl-build/libjoshua_sycl.so"
python3 kernels-cubecl/benchmarks/evaluate_backends.py \
  --model /absolute/path/model.gguf --tokenizer /absolute/path/tokenizer.json \
  --output /tmp/backend-evaluation \
  --prompt-file kernels-cubecl/benchmarks/deepseek-realistic-prompt.txt \
  --decode-text-file kernels-cubecl/benchmarks/deepseek-diverse-continuation.txt \
  --decode 128 --passes 2 --prefill-chunk 8
```

The varied continuation is teacher-forced so backend comparisons exercise the
same token sequence even when their logits differ. This measures conditional
inference performance and full-logit parity, rather than free-running language
quality. Each process warms eight prompt tokens and two decode steps first;
all timed passes use the full prompt. Kernels run serially across APIs.
The evaluator records failed or unavailable configurations and preserves
existing complete results. A folder left without `summary.json` is quarantined
with `--retry-incomplete` and rerun instead of requiring manual cleanup.
`compare_backends.py` retains the absolute **1e-5** logit and repeatability
gates, and a candidate is eligible for automatic preference only when the
reference baseline's own allocator and driver memory are stable. A fast
configuration that fails parity does not qualify for a new automatic selection
preference.

Device statistics have different meanings across APIs. Vulkan allocator
reserved/live bytes remain available. The harness additionally samples Linux
XE DRM client counters for B50 resident VRAM and other memory regions,
deduplicating descriptors by client ID. OpenCL/SYCL allocator counters are
unavailable, so their legacy reserved/live placeholders must not be interpreted
as zero device memory; comparison uses the separately recorded driver samples.
Private host memory includes `RssAnon + VmSwap`; file-backed residency remains
separate. New runs also record process I/O counters and expert-cache reports
at session boundaries. Cache warming and mmap page faults can affect timing,
so compare repeated sessions and read bytes before attributing differences to
kernel execution alone.

### Measured backend results, 2026-10-18

The [complete evaluation report](benchmarks/backend-evaluation-b50-20261018.json)
contains timings, full-logit comparisons, memory bounds, I/O counters, source
hashes and baseline reproductions. Hardware was an Intel Arc Pro B50 and
Ryzen 9 9950X with approximately 74 GiB host RAM. Runs were serial, with
shared mmap-backed weights, host logits included, and three measured sessions
per configuration unless noted. These are single-request measurements.

Fixed-placement median decode milliseconds per token:

| Model and workload | CPU | OpenCL | SYCL | Vulkan CubeCL |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-1.7B Q8_0: 126 chat tokens, batches of 8, 128 varied decode inputs | 85.1 | 67.6 | **44.8** | 86.0 |
| Qwen3-8B Q4_K_M: 16 prompt tokens, one-token prefill, 32 varied decode inputs | 171.6 | 161.9 | **107.5** | 448.4 |
| DeepSeek-V4.1 Flash REAP-272E: 8 prompt tokens, batch 8, 8 decode inputs | **639.7** | 836.6 | 720.2 | 3,498.4 |

Thus the existing CubeCL-versus-generic Vulkan gain does **not** establish
Vulkan as the fastest backend. SYCL wins the tested Qwen decode workloads;
CPU wins this short DeepSeek workload. The Q4_K_M file includes Q6_K weights,
which the current CubeCL kernels do not cover.

For the Q8 long workload, prefill plus 128 decode tokens takes 79.56 seconds
on Vulkan with batches of 8, 12.49 seconds on CPU, 14.50 seconds on OpenCL,
and 9.37 seconds on SYCL. Changing Vulkan prefill to one-token chunks reduces
prefill from 68.55 to 10.25 seconds (6.69x) and total session time to 21.23
seconds, with **bit-identical** full logits and stable allocator memory.
Batches of 32 reduce prefill further to 2.54 seconds, but drift by
**7.37e-5**, failing the unchanged **1e-5** gate. Their sampled allocator
reservation peaks at 3.48 GB versus 2.27 GB for batches of 8. No global
prefill default was changed from these model-specific measurements.

The 8-token-batch Qwen3-8B Vulkan run loses its device during the second
prefill chunk. The same failure occurs with generic Vulkan and on unchanged
commit `661d808648e72b2b6d2e4ccfe8695f264d96d50f`, using the same lockfile,
model and prompt: `vulkan queue_submit failed: ERROR_DEVICE_LOST`. The
single-token-prefill matrix above avoids that failure. The batched matrix
uses CPU as its reference so the other backends can still be evaluated.
The report preserves the failed run and the unchanged-base reproducer.

### Long DeepSeek CPU replay

A new CPU run replayed the **exact earlier Vulkan trace**: the 121-token
realistic chat prompt, batches of 8, 128 decode steps, and two sessions.
CPU decode averaged **641.6 ms/token** versus the earlier production Vulkan
**3,458.9 ms/token**, a **5.39x** speed difference. CPU prefill averaged
52.05 seconds, and the full session averaged 134.18 seconds versus Vulkan's
719.70 seconds (5.36x). The Vulkan reference is historical, not interleaved
with the new CPU run, and its older harness lacks process I/O counters.

All 129 greedy choices match, but the full-logit maximum difference is
**0.057971**, far beyond **1e-5**. CPU repeats are exact. For block-quantized weights, CPU
matmul quantizes activations into `VecDotType` before its dot products;
native Vulkan follows a different floating-point path. This evaluation does
not isolate every contributor to the cross-backend drift. The earlier
Vulkan generic-versus-CubeCL comparison remains bit-identical; this is a
separate cross-backend comparison.

CPU private memory (`RssAnon + VmSwap`) grows **304 KiB** between sessions.
File-backed residency grows from 59.12 to 64.18 GiB. The process reads about
116.0 and 74.9 GiB physically during the two sessions, so storage and page
residency are part of the measured workload. This does not establish the
same performance under a smaller RAM budget or concurrent serving.

### Automatic placement corrections and gates

Backend priority remains **CUDA, Metal, OpenCL, Vulkan, SYCL, CPU** among
compiled, available backends. Dense placement still probes the selected
backend against CPU; it does not rank all APIs. Its Q4_K-only probe can miss
Q8 and grouped-projection costs, so hardware alone is insufficient for a
new preference.

The evaluation exposed and corrected these planner issues:

- A dense-only model assigned to CPU could retain a SYCL expert-device
  argument and fail loader validation. With no routed expert pool, that
  argument now follows the resolved dense device.
- V4.1's CPU-only Engram tables counted toward VRAM, falsely rejecting a
  usable B50 with a 70.35 GiB dense estimate. Tables now count as host-only;
  Engram wkv remains packed and q/k vectors count as f32. The conservative
  real-model dense estimate becomes 10.27 GiB. Table loading is unchanged.
- V4.1 automatic cache sizing could move dense to CPU and silently build a
  CPU software cache. That path changed repeat logits without an accelerator
  consumer. Automatic caching now requires one; explicit requests remain
  available. The repaired GPU auto path is bit-identical to explicit SYCL,
  and the CPU-dense auto path is bit-identical to plain CPU, with no automatic
  cache in either tested configuration.

Explicit CPU-dense / 12 GiB GPU-expert-cache experiments are **676.7 ms/token
OpenCL / 735.1 ms/token SYCL**, slower than plain CPU in the short test.
They also fail repeat parity at approximately **2.9e-5**. They were not used
to promote an automatic cache policy.

Matching greedy choices is insufficient for the **1e-5** logit gate. All
alternative fixed backends fail that gate against Vulkan on these traces.
Backend priority and the existing CubeCL dispatch were therefore left
unchanged. CPU private memory also fluctuated by several MiB in some
repeated Qwen runs, so those strict memory gates remain failed rather than
being relabeled stable. OpenCL's initial driver-memory growth settles after
full-workload warmup: five fully warmed sessions have identical private and
VRAM endpoints. Vulkan reservations remain stable in the accepted runs.

### Next measured kernel targets

Projection benchmarks use the actual model weights, deterministic synthetic
activations, 32 repetitions per pass, and three passes. Their timing includes
host output readback; they are cost measurements rather than model-parity
acceptance tests. Reproduce one with `--projection-tensor output.weight`,
or add `--projection-rows 8` for the grouped shape.

| Actual projection | CPU | OpenCL | SYCL | Vulkan |
| --- | ---: | ---: | ---: | ---: |
| Qwen8 Q6_K head, `(1,4096) x (151936,4096)` | 11.59 ms | 26.94 ms | 23.98 ms | **194.04 ms** |
| DeepSeek Q8 output A, `(8,4096) x (8192,4096)` | 6.39 ms | 7.03 ms | 6.65 ms | **72.96 ms** |

The first target is packed **Q6_K**, including the large output head. The
second is the **grouped Q8 attention output projection**. The current
DeepSeek path computes all eight output blocks for each of eight input
groups, then retains only the diagonal blocks. That does eight times the
required products and presents eight rows to dispatch, bypassing CubeCL's
single-row eligibility. There are 40 such projections in this model;
40 times the isolated Vulkan measurement is roughly 2.9 seconds per step,
a cost estimate consistent with much of the observed 3.5-second decode.
A one-row CubeCL measurement on the same weight is 1.67 ms, but it uses
different mathematical inputs and is only a proxy for a future fused
implementation. Neither candidate had been implemented at the time of
that evaluation; see the later checkpoint above.

Validation: **44 engine unit tests passed (2 ignored), 12 placement tests,
7 DeepSeek41 integration tests, and 1 supported SYCL placement regression
passed**, plus **5 native Vulkan CubeCL tests (2 benchmark tests ignored)**
and the all-accelerator benchmark build/check. Compiling all
accelerators exposed a CPU-only test's implicit `Auto == CPU` assumption;
that failure was reproduced on unchanged base, then the CPU fixtures were
made explicit. Raw logits and full memory timelines remain in
`/tmp/joshua-backends-20261018`; the saved report retains hashes, bounds,
endpoints, timings and the failed configurations.
