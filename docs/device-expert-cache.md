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
are preserved. No pinned staging allocation, raw FFI, or pending-slot readiness
state is introduced. Completion ownership and the one-slot hot policy were
subsequently corrected during review, as described below.
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
session residency. With a usable explicit cache budget, compact source experts remain
on the host (mmap or streamed loading); the destination is the requested expert
device, or the dense device when experts are host-placed. Uploads use the shared
transfer path; dispatch looks up the device payload, moves activations to its
actual device, and holds a launch-owned lease through compute-queue completion. Misses retain
host execution. No full expert-pool f32 materialization or per-session weight
copy is introduced.

Slots reserve the largest compact gate/up/down payload, with the total budget
capped at that reservation times the actual expert count. Auto hot-set sizing
uses three quarters of resolved slots instead of host RAM (at least one tracked
expert for a nonempty single-slot pool, which protects that slot and refuses cold
uploads until the next hot-set replacement); refresh replaces the
capped protected set instead of accumulating stale protected members. Sessions
share the pool and uploader, but their frequency histories remain independent:
one session can still replace another's protected set. **Disjoint-session
no-thrash policy is not solved.** Logical payload accounting excludes backend
padding and transient activations/upload or destruction workspace; it is not a hard
bound on all CUDA allocations.

### PR #248 review follow-up (2026-10-31)

The model-wide retirement list and its age-based overflow deletion are removed.
Every successful lookup now returns a launch-owned `DeviceExpertLease`. The
pool's mutex serializes lookup with victim selection: a payload with another
strong reference cannot be evicted or explicitly released, and remains charged
to `resident_bytes`. Before releasing its reference, the lease synchronizes the
payload's actual compute device/queue. This also runs on dispatch `?`, unwind,
and session/model teardown; destroying the pool does not destroy a live lease.
A failed backend completion fence aborts rather than releasing potentially live
weights or retaining an unbounded error list. A routing readback in a different
session has no reclamation authority. Qwen, DeepSeek2, and the shared DeepSeek4
pool use this same contract. **This is bounded synchronous cached dispatch,
not evidence of asynchronous compute/upload overlap.**

For Metal the lease uses `flush_and_wait_current`, retaining the specific
committed command buffer while waiting. The general `Device::synchronize`
drains a shared in-flight list and can return early if another thread already
took that list; it is not the lease completion primitive. CUDA waits on the
payload's compute stream; OpenCL/SYCL/Vulkan use their queue completion fences.
Only CUDA was executed in this follow-up; Metal and other accelerator behavior
was inspected, not rerun or locally rebuilt.

Before overriding an explicitly requested expert device, Qwen and DeepSeek2
resolve the largest complete gate/up/down compact payload from the GGUF tensor
shapes and block types. A zero, sub-slot, incomplete, or noncompact configuration
does not force CPU sources and builds no cache. The exact Q4_K fixture slot is
`3 * (256 * 256 / 256 * 144) = 110592` bytes. Draft admission uses the same
compact-slot feasibility policy and includes Metal alongside the other device
backends. Both mmap and streamed placement are tested at zero, slot-minus-one,
and exactly one slot, inspecting the actual loaded expert device.

When that explicit cache is usable, Qwen suppresses `JOSHUA_GPU_WEIGHT_CACHE`.
Inspection of `WeightCache::new` confirms that it previously only created an
empty host-side map and retained the mmap/device handles: its numeric capacity
was **not a VRAM reservation**. Device allocation occurs lazily in `forward`,
which host source experts did not call. Suppression makes the precedence
explicit and avoids constructing the unused cache at all.

Physical accounting remains distinct: the logical slot budget includes active
leases and upload reservations, with no model-wide retired-payload allowance.
Backend quantized-buffer padding, dense weights, KV, activations, driver/allocator
retention, and temporary upload/deletion workspace are additional. Payloads
selected for deletion are dropped outside the pool lock; concurrent acquisition
may overlap that destructor with the next reservation. Thus logical accounting
is not a hard bound on process-wide VRAM or transient allocation peaks.

### CUDA shared-session ownership investigation (2026-10-31)

**Do not merge.** Devin thread `PRRT_kwDOSigPUM6yWM8u` remains open. A verified
backend ownership defect explains cache-independent concurrent corruption, but
fixing it does not clear strict cached/host greedy parity or historical Qwen
performance acceptance.

`CudaDevice::new` selects cudarc's `per_thread_stream()`, whose raw
handle is the CUDA per-thread sentinel `0x2`, not one physical queue shared by
callers. Cloned Candle devices nevertheless share one cuBLAS handle and
per-DeviceId MMVQ/MMQ scratch. Scratch locks cover submission, not completion;
different host threads could therefore reuse scratch on different queues.
`Device::synchronize` also targeted only the calling thread's default queue.
The initial fix in `ff746d7` globally delegated that constructor to
`new_with_stream`. The follow-up for discussion `4434019973` restores the
vendored public constructor's original per-thread semantics. Only Joshua's
Engine CUDA selection paths (automatic and explicit) now call the existing
`Device::new_cuda_with_stream`: one owned, nonblocking compute stream, shared
by device clones, cuBLAS, and scratch. Library callers supplying their own
device must also select that explicit constructor before concurrently sharing
sessions; merely cloning a per-thread default device is not race-safe. This
is not a universal fix for external Candle applications sharing default-stream
devices across host threads.
Submission locks then order scratch users on the same actual queue, and lease
fences target that queue. **Sessions share serialized compute, not independent
per-session GPU overlap.** Background uploads retain their separate completed
transfer stream. No retirement lists or per-session weight copies are added.

The call-site audit found only two production CUDA constructors, both in
`src/engine.rs`: `default_device` and explicit `ComputeBackend::Cuda`
resolution. Other constructors under `src` are test-only. Engine template,
draft, native/stock model, and session loads receive the selected device rather
than construct another CUDA device. Quantized MMVQ/MMQ and cuBLAS submit on
the exposed device stream; compact uploads use `cuda_transfer_stream`, complete
the copy before publication, and keep cudarc event tracking enabled. No Joshua
production CUDA graph-capture entry point was found. Exposing `cuda_stream`
does not promise implicit ordering with external default streams: callers
must use events or explicit synchronization.

New bounded regressions retain the public constructor's raw per-thread sentinel
`0x2`, exercise Engine automatic and explicit CUDA construction plus repeated
tiny Llama generation, and test upload on legacy-default, per-thread-default,
and transfer streams followed by Candle compute, an explicit completion-event
handoff, and external-stream readback. The separate graph replay regression
uses one private owned stream, disables event tracking **before allocations**,
warms its metadata/host-copy cache, captures affine GPU work, and checks three
replays. Its unsafe precondition is confined to that test: every allocation,
use, replay and completion stays on that stream, with no external consumers.
This does not establish general graph capture with normal event tracking:
the exploratory tracked attempt failed with `CUDA_ERROR_STREAM_CAPTURE_ISOLATION`
(then capture invalidation), retained in `interop-capture.log` and
`interop-workspace-clean.log`. That limitation is not silently fixed or hidden.
`interop-engine-concurrent-final.log` additionally passes two concurrent Engine
completions against its serial result for both automatic and explicit CUDA
selection (five tests passed, one diagnostic ignored).

`cuda_shared_workspace_tests` uses fixed weights/inputs, three host workers,
F32/F16/BF16, and 1/8/32 input rows. Each worker queues 96 quantized and 96 dense
operations, then compares every output bit-for-bit with serial GPU references.
Pending outputs are bounded; the compact Q4_K matrix is only 512 by 1024.
This tests scratch and dense-handle sharing without CPU/GPU rounding in its
reference. Inputs are deterministic, OS/GPU scheduling is not; repeated passes
do not prove every CUDA kernel race-free.

Sources/logs remain in `/tmp/joshua-248-devin-20261031-171830`. Controls restore
only `cuda_backend/device.rs` from the exact saved
`source-e48b61804abd2f53f4453bf7d0d2b983e0fbffee.tar`. Candidate rebuilds touch
that file afterward; its owned-stream SHA256 is
`89ee49b61803f6996fb84d5d2d6d36da38325011d419c4a71b5d369a14e4cfee`.
These focused runs use the **debug** profile, the existing shared target,
`CARGO_BUILD_JOBS=2`, and `nice -n 19`, not release/real-model acceptance.

- `stream-control-before.log`: the initial F32-only default-stream experiment
  failed (`-0.09726913` versus `-0.079484716`); the owned-stream control passed.
  Two further processes happened to pass both.
- `stream-before-expanded-{1..3}.log`: the expanded default-stream experiment
  failed twice, including F16/BF16 quantized and dense F32 results; the explicit
  owned-stream control passed all three processes.
- `stream-before-no-pool{,-2,-3,-4,-5}.log`: bit-exact concurrent Qwen and
  DeepSeek2 no-pool model testing failed once in five processes before the fix.
  `stream-owned-no-pool-{1..5}.log`: five passes after it, covering prefill,
  70 decode rows, and three all-logit continuation rows.
- `stream-owned-workspace-{1..5}.log`: both constructors passed every dense and
  quantized case in all five processes, bit-exact for all three dtypes.
- `stream-fix-strict.log` and `stream-owned-strict-churn-{2,3}.log`: strict
  interleaved cache/host greedy parity still fails, with 5, 5, and 7 DeepSeek2
  disagreements. The test now **asserts** the total disagreement count after
  both architectures complete instead of merely logging it. Tolerances are
  unchanged. Qwen had zero greedy disagreements in these three runs.

The final rebuilt candidate (`stream-final-validation.log`, using
`--no-fail-fast`) reports: CPU numerical reference 1 passed; expert-cache tests
6 passed, **1 failed**, 2 ignored; shared-workspace tests 2 passed. The failure
is strict DeepSeek2 churn with five greedy disagreements, not a passing gate.
The actual interleaved readback/launch/eviction/error-drop lifetime test passes.
Both actual CUDA zero/sub-slot placement tests also pass with
`JOSHUA_GPU_WEIGHT_CACHE=1` (`stream-final-placement.log`).

The separate `cuda_serial_expert_precision_probe` loads one all-CUDA-expert
model and one CUDA-dense/CPU-expert model, **neither with a pool**, and runs them
sequentially on one host thread. `stream-serial-precision{,-2}.log` repeats the
same four DeepSeek2 winner flips for prompt `[1,4,2]` at offsets 30, 38, 39, 40,
and one for `[5,7,1]` at offset 26. Maximum absolute differences are 0.030333817
and 0.030445576. At offset 26 in the interleaved run, cache scores choose token
3 (1.9908428) over 6 (1.9860317), but host scores choose 6 (1.9931693) over
3 (1.9886569). This isolates a repeatable expert-backend numerical-path
disagreement from concurrent shared-stream corruption. Whether activation
requantization, reduction ordering, or another kernel precision difference
fully explains the end-to-end winner flips is **not established**. The ignored diagnostic's successful
exit checks numerical bounds only, not greedy acceptance. Changing arithmetic
or tolerances to hide these flips is not part of this fix. They remain a real
PR blocker even though they also occur without an expert pool.

The bounded ignored `cuda_q4k_expert_arithmetic_probe` now isolates a concrete
arithmetic difference using four fixed seeds, 256-by-256 compact Q4_K weights,
and one 256-element input row. The CPU Q4_K dot product quantizes activations
as Q8_K (256-element blocks, signed-max scale stored as f32); the fast CUDA
path uses Q8_1 (32-element blocks with half scales/sums). On the same input,
fast CUDA differs from CPU by up to 0.0009132363. Bypassing that fast path
with direct-float DMMV still differs by up to 0.00071775913: merely disabling
MMVQ is not a parity fix. Quantizing/dequantizing the input to CPU Q8_K first
and then using CUDA DMMV reduces the observed maximum to 0.000000044703484
across all four seeds (asserted below 0.000001), with bit-exact repeated GPU
results. `arithmetic-q4k-control.log` records this initial isolation experiment.
The scoped-constructor rerun `interop-arithmetic.log` repeats every value.
This demonstrates different activation arithmetic, not just reduction order,
for the isolated Q4_K operator. It does **not** prove every model-level flip is
accounted for or implement a safe production fix. The probe's host round trip
and process-global DMMV switch are diagnostic only, never a production dispatch
policy. Matching backend arithmetic on device, all involved quantized formats,
prefill/decode paths and greedy continuation still needs a verified design;
neither per-expert host transfers nor tolerance changes are accepted shortcuts.

One DeepSeek4 shared-cache run also failed its unchanged numerical bound
(`0.98116964 != 0.79818434`, `stream-fix-validation.log`); two immediate focused
reruns and the next combined run passed. Three constructor-reverted controls
(`stream-before-ds4-{1..3}.log`) passed, so the failure is **not labelled
pre-existing** or dismissed as baseline. It remains an unresolved parity
observation.

After restoring the public constructor and opting the Engine/tests into owned
streams, `interop-engine-capture.log` passes all five nonignored stream/Engine
tests (one arithmetic diagnostic ignored). `interop-model-validation.log`
passes the CPU numerical-reference test and six cache tests, but **fails**
`cuda_older_loaders_interleaved_eviction_numerical_parity`: Qwen has zero greedy
flips; DeepSeek2 has seven, split five for `[1,4,2]` (maximum absolute error
0.030209064) and two for `[5,7,1]` (0.026428163). Upload timing changes which
experts run on CPU versus CUDA, so mixed-path churn need not match the five
all-CUDA/host flips. Final DeepSeek2 accounting is three slots / 331776 logical
bytes, 149 hits, 79 uploads, 76 evictions, zero upload failures. The no-pool
concurrent gate and DeepSeek4 cache test pass in this run, without clearing
the earlier unexplained DeepSeek4 observation. `interop-serial-precision.log`
again reproduces the exact same five all-CUDA/host flips on one host thread.
Diagnostic success means the experiment completed, not that greedy parity
passed. Both the strict parity thread and no-regression acceptance stay open.

Three owned-stream timing processes repeat three-token prefill, 70 warmup
tokens, and three trials of 64 timed tokens, with blocking readbacks and a
331776-byte logical cache. `stream-owned-timing-{1..3}.log` records:

| Run / architecture | CPU experts, CUDA dense (tok/s) | 3-slot cache (tok/s) | All CUDA (tok/s) |
|---|---:|---:|---:|
| 1 / Qwen | 374.835 | 462.578 | 1288.685 |
| 2 / Qwen | 378.597 | 449.810 | 1331.570 |
| 3 / Qwen | 375.289 | 441.142 | 1361.540 |
| 1 / DeepSeek2 | 709.550 | 1278.448 | 1334.122 |
| 2 / DeepSeek2 | 684.341 | 1258.509 | 1330.642 |
| 3 / DeepSeek2 | 747.854 | 1266.408 | 1342.061 |
| scoped constructor / Qwen | 374.998 | 461.598 | 1304.135 |
| scoped constructor / DeepSeek2 | 729.936 | 1254.571 | 1304.585 |

The scoped-constructor pair is `interop-timing.log`, with the same bounded
prefill/warmup and three-by-64-token debug protocol, not a release or controlled
no-regression measurement. The earlier roughly 25% Qwen regression remains a
blocker; these samples do not close it.

The constructor-only baseline (`stream-before-timing.log`) gives Qwen
373.898 / 453.212 / 1280.390 tok/s and DeepSeek2
712.256 / 1214.650 / 1244.758. These live-service debug samples do not prove a
controlled throughput improvement or clear the historical Qwen regression
below. Physical padding, activations, cuBLAS/driver workspace, stream/event
metadata, allocator retention, and destruction/upload transients remain extra
to logical payload accounting. No new physical VRAM cap or async overlap is
claimed. Existing CPU :8080 and CUDA :8081 PIDs remained 4061392 and 4063925,
with approximately 12517 MiB free before/after focused runs. No local builds
or new targets were used. #62 and #52 remain open.

Reproduce focused tests with `cargo test --features cuda --test
cuda_shared_workspace_tests --test cuda_expert_cache_tests --no-fail-fast --
--test-threads=1 --nocapture`. Run the serial diagnostic or timing explicitly
with their exact test name and `-- --ignored --exact --nocapture`.

### Historical review validation at `e48b618` and remaining blockers

All follow-up accelerator runs used kleya's RTX 5060 Ti with the existing
CPU :8080 and CUDA :8081 services intact (PIDs 4061392 and 4063925). Before tiny
GPU tests, free VRAM was about 12,500 MiB. No large models or local builds were
started. Sources live in `/tmp/joshua-248-devin-20261031-171830`; commands used
`CARGO_BUILD_JOBS=2`, `nice -n 19`, and the existing shared target
`/home/tserica/joshua-cache-62-52-ff7528ad/target`. Changed source files were
touched after baseline builds before rebuilding the candidate; passing an old
shared-target binary is not counted as candidate validation.

`final-validation.log` records:

- 497 library tests passed, 10 ignored, including exact block sizing, one-slot
  protected-hot churn, active leases remaining charged through hundreds of
  acquisitions, and a blocking completion fence preventing payload destruction.
- Both CUDA placement tests passed, inspecting actual expert devices for
  `None`, zero, slot-minus-one, and exact-slot budgets, streamed and mmap loads.
  `final-overlap-placement.log` repeats these with `JOSHUA_GPU_WEIGHT_CACHE=1`:
  2 passed. `final-check.log`: `cargo check --features cuda --all-targets` passed.
- The deterministic actual-GPU lifetime regression passed: A completes a
  readback, B holds a lease and queues 33 expert matmuls, A attempts 300
  acquisitions/releases, then B queues another launch and returns an injected
  error. Sixteen cycles verify the payload stays charged and cannot be freed;
  pool destruction with a live lease is also covered.
- The 72-token greedy-continuation test passed for Qwen and DeepSeek2 against
  CUDA-dense/CPU-expert execution, plus all-CPU numerical reference and three
  all-row verification logits. Maximum cache-vs-host absolute errors were
  0.018511906 and 0.03250295, respectively. Dropping a queued session and
  continuing another session is covered. Existing tolerances are unchanged.
- The separate all-CPU numerical-reference test passed with disabled and
  three-slot caches. It does **not** establish CPU/CUDA greedy identity:
  DeepSeek2's offset-26 winner is 15 on CUDA versus 3 on CPU, with maximum
  absolute error 0.07400927, in both modes.
- CUDA integration finished **5 passed, 1 failed, 1 ignored**. The failing new
  test was `cuda_older_loaders_interleaved_eviction_numerical_parity`, with
  `0.73215044 != 2.1182497` in concurrent Qwen execution. An earlier run passed
  the same churn test, but recorded 4 and 1 DeepSeek2 greedy disagreements for
  its two sessions. The original strict shared-cache test passed in the final
  run but failed intermittently in repetition (4/5 repeated runs passed;
  failing example `-0.39236903 != 1.2431526`). These are not green parity gates.

The final churn harness preloads both reference models before starting all
three workers, so no failure can strand the uploader at its start barrier.
`final-churn-{1..3}.log` records this version: two failures
(`1.5724974 != 1.2495484`, `-0.566552 != -0.7212319`) and one numerical pass.
That pass exercised 401 Qwen and 70 DeepSeek2 evictions, with three resident
slots and 331776 logical bytes each, but still reported 5 and 2 DeepSeek2
greedy disagreements. It does not replace the original strict greedy test or
clear concurrent-output acceptance.

The intermittent concurrent numerical failure is independently reproducible
with **no expert cache at all** on unchanged pre-review source. The original
artifact `/tmp/joshua-62-20261031.YI3I9oOi` has Qwen source SHA256
`fa661ca1a9b4c4fa7e8c26fc1b1e2d2bb6b6f0abb7c6859ae72583851d64bbfb`
and residency SHA256
`494406ece0dba15fc1ba85b28bc784e4b881110a69c936b7f94b57d977ad6754`,
matching `9736eb6`. `baseline-concurrent-repeat-{1..6}.log` in the follow-up
directory records the unchanged strict test: 2 passed, 4 failed; examples
`1.2475119 != 0.8166167` and `-0.11178517 != 1.1484298`.
The standalone diagnostic
`/tmp/joshua-62-20261031.YI3I9oOi/tests/cuda_dense_concurrency_probe.rs`
loads all models with `Some(0)` and asserts no pool exists. Running
`cargo test --release --features cuda --test cuda_dense_concurrency_probe -- --nocapture --test-threads=1`
on that artifact failed twice in three runs: offset 7
`1.2651038 != 1.072143` and offset 42 `0.17209363 != 0.16552365`
(`baseline-no-cache-{1..3}.log`). Its stricter diagnostic threshold is not a
change to any existing test. At that point the concurrency failure's source
was not established or fixed. The later shared-stream investigation above
identifies a backend ownership defect, but retains separate strict output
parity blockers; these historical failures are not passing evidence.

Other exact baseline reproductions are retained: `baseline-9736-cpu-reference-2.log`
records the same disabled-cache DeepSeek2 CPU/CUDA winner divergence, and
`baseline-9736-churn-2.log` records the strict adversarial-churn 3-versus-6
DeepSeek2 greedy mismatch on the pre-review source. The diagnostic test for the
latter remains at
`/tmp/joshua-62-20261031.YI3I9oOi/tests/cuda_churn_baseline_probe.rs`.
No tolerance was loosened to call these passes.

`integration-review.log` records CUDA-feature builds with CUDA explicitly hidden
for CPU/fallback validation: DeepSeek2 12 passed, external draft 7 passed,
memory placement 13 passed, Qwen3MoE 8 passed, speculative 22 passed. These are
not GPU passes. The original CUDA-visible speculative baseline failure log
`/tmp/joshua-62-base-SnqkN1/baseline-speculative.log` is preserved: ten foreign
CUDA expert-device load failures and two `unsupported dtype for quantized matmul
Iq2Xxs` failures on unchanged `99d2ddb`. Exact failing names:

```text
deepseek2_speculative_greedy_matches_plain_decoding
deepseek4_speculative_greedy_matches_plain_decoding
deepseek4_speculative_speeds_up_repetitive_output
draft_model_greedy_matches_plain_decoding_deepseek2
draft_model_greedy_matches_plain_decoding_qwen3moe
draft_model_sampling_matches_plain_sampling_distribution
kda_hybrid_speculates_without_changing_output
qwen3moe_speculative_greedy_matches_plain_decoding
recurrent_hybrid_speculates_without_changing_output
sampled_self_draft_matches_plain_token_for_token
self_draft_is_always_accepted
speculative_sampling_matches_plain_for_a_fixed_seed
```

Three independent bounded timing processes used three-token prefill, 70 warmup
tokens, then three trials of 64 timed decode tokens per mode with blocking
logit readbacks. The three-slot logical budget was 331776 bytes, with positive
hits/uploads and zero upload failures. Services remained live, so these tiny
measurements are noisy, not an idle-device performance acceptance:

| Run / architecture | CPU experts, CUDA dense (tok/s) | 3-slot cache (tok/s) | All CUDA (tok/s) |
|---|---:|---:|---:|
| 1 / Qwen | 914.092 | 927.198 | 2082.637 |
| 2 / Qwen | 342.281 | 356.889 | 1159.392 |
| 3 / Qwen | 576.655 | 432.335 | 1132.394 |
| 1 / DeepSeek2 | 1262.852 | 1950.120 | 2023.647 |
| 2 / DeepSeek2 | 576.690 | 1043.116 | 1054.181 |
| 3 / DeepSeek2 | 480.406 | 1384.761 | 1489.317 |

Logs are `timing-review.log`, `timing-review-2.log`, and `timing-review-3.log`
in the follow-up directory. Qwen's third run still regresses by about 25% versus
host experts. **The known Qwen no-regression blocker remains open.** The first
two paired runs do not clear it. No real-model, PCIe-traffic, pinned-transfer,
async-overlap, or new Metal performance acceptance is claimed. Keep #62 and #52
open; neither the review fixes nor the pre-review green CI establish completion
of those issues.

### Pre-review validation (`9736eb6`)

On kleya's RTX 5060 Ti (16,311 MiB), existing services were preserved; the initial
probe showed 3,371 MiB used and 12,477 MiB free. Only tiny fixtures were loaded.
Release CUDA tests passed twice on the pre-review retirement-list implementation
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
