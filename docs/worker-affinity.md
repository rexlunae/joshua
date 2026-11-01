# Model-scoped worker affinity

This implements the production routing/cache-ownership part of #156, separately
from the client-only experiment in `prefix-locality-measurement.md` and the #201
prefix chunk directory. It does not migrate KV state between workers.

## Protocol and compatibility

For an explicit `model` on `/v1/chat/completions` or `/v1/completions`, send
`x-joshua-affinity: new` to the coordinator. Retain the returned UUID header and
send it on subsequent requests, **along with the full current prompt/history**.
An existing canonical UUID is also accepted. Invalid handles are rejected with
400 before forwarding. The handle is a routing hint, not a token, authorization
grant, durable conversation record, or promise that a cache exists. Requests
without this header retain the original work-aware worker selection.

The coordinator directory key hashes length-delimited authenticated credential,
handle, explicit model ID and requested context window. Bearer and `x-api-key`
forms of the same credential share an owner. Different credentials do not.
The coordinator's existing auth accepts Bearer credentials only; this does not
extend its authentication API to `x-api-key` (worker auth already accepts both).
Credential-free requests share an anonymous owner; clients sharing one API key
also share an owner. When auth is disabled, supplied owner headers are not
authenticated identities. Deploy distinct credentials/trusted authentication upstream
when these must be different tenants. Existing authentication middleware remains
authoritative, and a UUID never bypasses it.

The coordinator overwrites the internal `x-joshua-cache-scope` on forwarding.
Workers accept this internal scope only after successful configured worker
authentication. With no worker API key configured, the middleware ignores the
internal header, even if a client supplies an arbitrary Authorization/x-api-key
value. Public affinity hints and ordinary requests remain accepted; disabling
authentication does not create trusted identities or tenant isolation. Configure
the worker's `--api-key` and the coordinator's `--worker-api-key` consistently to
enable trusted routed cache ownership/residency. Every accepted worker key is a
trusted caller for its own credential namespace, including during key rotation;
do not distribute coordinator worker credentials to untrusted clients.
Workers hash the accepted scope with their authenticated credential and context; raw
credentials and prompts are not kept in the directory or inventory. Direct
worker HTTP calls are owner/context isolated too, but `new` is a coordinator
protocol operation, not a direct-worker handle-allocation API. Library callers
may set the trusted, serde-skipped `GenerationOptions.cache_scope`; omitted
scopes keep the original unscoped library cache behavior across compatible
default/explicit context windows. Explicit trusted scopes remain context-isolated.
Exact token-prefix checks, cache compatibility, and rollback rules remain the
authority for every reuse; neither a scope nor a routing hint supplies history.

Each loaded Engine has a fresh incarnation UUID, covering the actual loaded
model/tokenizer revision, backend and immutable execution configuration. No
cache crosses Engine instances, even if API model IDs are reused. Reload or
restart invalidates the directory hint at the next probe/lookup. Context changes
get separate namespaces. Template/tool/reasoning changes are checked through the
actual rendered token sequence; sampling settings do not grant cache reuse.
Workers still validate exact token prefixes and architecture-specific rewind
capabilities. Same-handle unrelated prompts are recomputed/reset, not trusted.

## Scheduling, feedback and bounds

The original capability, health, context, model, slot and admission checks run
first. Locality subtracts at most `min(cached_tokens * 4, request_work / 5)` from
the preferred worker's **incoming** estimated work, divided by slots. It never
discounts outstanding queued work. Four work units per token and the 20% cap are
conservative heuristics, not calibrated prefill-time/queue-time predictions.
The initial length is a work-derived estimate; usable residency feedback updates
it with actual cached lengths. An overloaded owner can lose to a cheaper eligible
worker, or immediately spill when full. The alternate worker reconstructs from
the supplied prompt. There is no cross-worker cache copy or hidden replay.

Directory ownership changes at dispatch under the existing selection lock.
Completions never rewrite it, so an older request completing after a spill
cannot restore an obsolete owner. Each entry includes a worker incarnation and
dispatch lease. A worker-info probe records its dispatch generation **before**
reading inventory. Complete absence invalidates a hint only if that observation
covers its lease, this coordinator has no local dispatch there, and the worker
reports itself idle. Older probes cannot erase newer leases. Missing inventory
(old workers) or a truncated/incomplete inventory is unknown, never eviction.
Positive reports update approximate length; negative reports cause cold routing.
Feedback is reconciled lazily at lookup; status is advisory, never authorization.

Bounds are global, not multiplied by conversation count:

- Coordinator: 4,096 entries with least-recently-dispatched eviction and 300-second
  idle TTL; O(capacity) expiry/eviction scans. Counters are constant-size.
- Worker-info: at most 128 scope/length records total, divided among loaded
  models; overflow sets `complete: false`. No tokens, prompt text or KV payloads
  travel in these records. Existing model metadata is unchanged.
- Scoped warm sessions and snapshots: 300-second idle TTL, reclaimed lazily on
  acquire/insertion/inventory. Unscoped library sessions keep existing behavior.
- Existing RAM-adaptive warm pool (0/1/2/4 sessions), accelerator session/VRAM
  caps, request concurrency, detached snapshot byte budget and four-root limit
  remain in force. Scopes do not allocate independent pools or weight copies.
- Session extraction remains exclusive under the pool mutex; rewind/reset and
  snapshot work retain the existing ownership, cancellation and failure paths.

`GET /v1/affinity` uses coordinator authentication and exposes directory
preferences, misses, spillovers, capacity evictions, expirations, stale epochs,
residency misses and aggregate estimated tokens. A preference is **not** a
confirmed token-cache hit. Worker `/metrics` JSON adds per-model warm reuse,
rewind, successfully prefilled saved tokens, snapshot occupancy/budget/eviction/
expiry and warm-state bytes. Warm bytes include reported state plus token-vector
capacity, not weights, allocator overhead, RSS, full VRAM or opaque remote state.
`x-joshua-queue-ms` is cumulative coordinator admission time in integer ms,
including earlier attempts if any; it is not worker-internal queue telemetry.

## Architecture limits

Attention-only native models use their existing token-prefix truncation paths.
Recurrent/hybrid models can rewind only to an existing compatible checkpoint;
they recompute the remainder, or reset if no supported checkpoint exists.
DeepSeek compressed/windowed state retains the native checkpoint/window rules.
NPU/plugin sessions can continue exact prefixes but do not export snapshots or
support arbitrary rewind; edits require a supported clear or a fresh session.
Remote-capacity splitting and device session caps are unchanged. Affinity neither
enables unsupported operations nor raises accelerator concurrency limits.

## Reproducible tiny-GGUF measurement

Run `RAYON_NUM_THREADS=1 CARGO_BUILD_JOBS=2 cargo test --test route_tests
benchmark_tiny_affinity_policies -- --ignored --nocapture --test-threads=1`.
The test writes `joshua-156-tiny-affinity.json` in the OS temporary directory.

Two independent loopback CPU Engines use a synthetic one-layer F32 Qwen3 GGUF,
256-token context, one slot each, 1 MiB snapshot budget and 16-token root minimum.
Six real chat turns retain generated assistant responses. Prefixes grow from
8 to 32 to 64 words; concurrent conversations are 1/2/4. Three repeats reverse
policy order on the middle repeat: direct round-robin, direct pinned session,
normal work-aware routing, and production load-aware affinity. Direct policies
use per-worker client admission; routed policies use coordinator admission.
For even conversation counts, round-robin naturally keeps each conversation on
the same worker; this is a control, not an expected locality penalty.

Each measured wave finishes before an independent third Engine checks every
stream against a fresh-namespace cold prefill. Across policies the test asserts
identical request and output hashes, including finish reasons and usage. There
are 1,512 measured requests and 1,512 cold parity requests in 108 cells. Worker
inventory refreshes precede each wave. Loading, probing and baseline computation
are outside measured request latency; first-use initialization is not removed.

TTFT includes client admission and ends at the first content/reasoning delta;
EOS-only replies use the terminal event. One-token decode is deliberately short:
this measures prefill/locality/control overhead, not real-model decode throughput.
Report p50/max TTFT, completion latency, queue delay, saved-prefill-token counters,
reuse rate, worker distribution and resident cache bytes. Routed queue delay has
integer-ms resolution; sub-ms values are not precise scheduler measurements.
Unscoped normal routing may share roots between same-owner conversations, whereas
explicit handles isolate those conversations. Direct policies omit a router hop.
These are intentional policy differences, not a pure isolated scheduler A/B.

The saved reports include GGUF/request/output hashes. The final Linux report
also hashes all changed sources/tests, the fixture builder and Cargo.lock;
interpret those against base `99d2ddbbb71239d810b4f00c1699ea89abe2cfa7`.
The earlier macOS exploratory report predates final scoring/tests and records
only its tracked diff hash, omitting then-untracked affinity files; it is retained
as historical evidence, not final-source provenance or an independent speedup claim.

### Historical CPU results (October 31, 2026)

The saved JSON files and their source manifests are immutable historical evidence
for the pre-review implementation at `5ca32af`, not measurements of the current
revision. Review fixes preserve unscoped library context reuse, ignore internal
scopes on unauthenticated workers, and authenticate the real-worker/benchmark
fixtures. Those semantic and fixture source changes invalidate any current-source
timing attribution to these reports. No updated performance result or universal
latency improvement is claimed; rerun the authenticated fixture for new timings.

Final report: `measurements/affinity-156-erskin-cpu.json`, Linux x86_64,
AMD Ryzen 9 5900HX (8 cores/16 threads), Rust 1.99 toolchain, debug profile,
`RAYON_NUM_THREADS=1`, `CARGO_BUILD_JOBS=2`, `nice -n19`. Other delegates' CPU
timing jobs had finished before this benchmark; the pre-existing inference
server was left running and untouched. No claim of an isolated physical host
or enforced single-threaded BLAS is made. The 108-cell test completed in 10.89s.

Each timing is the median of **nine cell p50s** (three prefix sizes × three
repeats), not a pooled request median. Reuse is requests with any warm-session
reuse / requests; saved tokens sum actual successful-prefill reuse counters.
Memory is maximum end-of-cell summed warm-state + snapshot bytes, **not peak RSS**.

| Concurrent chats | Policy | TTFT ms | Reuse | Saved tokens | Queue ms | Resident bytes |
|---|---|---:|---:|---:|---:|---:|
| 1 | Round-robin | 2.390 | 36/54 | 1,734 | 0.002 | 23,304 |
| 1 | Work-aware | 2.854 | 36/54 | 1,734 | 0.000 | 23,304 |
| 1 | Pinned session | 1.859 | 45/54 | 2,325 | 0.002 | 11,976 |
| 1 | Load-aware affinity | 2.374 | 45/54 | 2,325 | 0.000 | 11,976 |
| 2 | Round-robin | 2.058 | 90/108 | 4,695 | 0.001 | 24,088 |
| 2 | Work-aware | 2.736 | 90/108 | 3,800 | 0.000 | 46,880 |
| 2 | Pinned session | 1.921 | 90/108 | 4,695 | 0.001 | 24,088 |
| 2 | Load-aware affinity | 2.483 | 90/108 | 4,695 | 0.000 | 24,088 |
| 4 | Round-robin | 4.088 | 120/216 | 7,968 | 2.297 | 57,872 |
| 4 | Work-aware | 5.401 | 198/216 | 6,394 | 2.000 | 91,824 |
| 4 | Pinned session | 4.100 | 120/216 | 7,968 | 2.147 | 57,872 |
| 4 | Load-aware affinity | 5.309 | 116/216 | 6,954 | 2.000 | 110,120 |

At one conversation, affinity saved 2,325 rather than 1,734 tokens across the
nine cells; it avoided duplicate worker residency and reduced this summary
TTFT versus ordinary routed work-aware selection. With four conversations it
spilled 56 times, trading locality and more resident snapshots for available
slots. The tiny latency difference against work-aware selection at that load
does **not** establish a statistically reliable win, and direct pinned routing
was faster. Same-owner unscoped root sharing also explains why work-aware
selection can have more reuse events but fewer saved tokens. These results
validate the mechanism and its trade-offs, not a blanket speedup or calibrated
scheduler cost model. The raw reports retain per-cell maxima and completion
latencies for inspection rather than hiding those costs.

Historical macOS report: `measurements/affinity-156-macos-exploratory.json`.
It completed in 6.15s before the final residency-bounded scoring change and
additional boundary/eviction/NPU tests. Do not combine its timing with the
final Linux result or treat its incomplete source fingerprint as final evidence.

## Acceptance evidence and outstanding gates

CPU correctness coverage includes exact continuation, edits, unrelated prompts,
owner/model/context isolation, concurrent follow-ups, bounded snapshot/directory
growth, TTL, truncated/complete inventory, real worker eviction/recomputation,
stale probe leases, incarnation changes, transport failure and admission failover.
Native Qwen recurrent checkpoint and DeepSeek compressed window/chunk suites
compare against fresh logits; scoped tests compare fresh greedy outputs across
7/8/9, 15/16/17 and 23-word compressed-cache boundaries. Routing/server suites
also retain auth, cancellation and stream/output checks.

Historical pre-review commands at `5ca32af` used a unique source archive (no checkout), shared
`CARGO_TARGET_DIR=/tmp/joshua-prefix-201-NXjVJ2AM/target`, jobs=2 and nice=19.
Changed source/test files were touched when switching roots. Results:

- `cargo test --lib --test affinity_tests --test route_tests --test integration_tests
  --test server_tests --test qwen_tests --test deepseek4_tests --test prefix_locality_tests
  --test request_settings_tests -- --test-threads=1`: 496 library, 5 scoped affinity,
  36 routing, 41 integration, 9 server, 10 Qwen, 22 DeepSeek4, 1 locality and 3
  request-settings tests passed; 13 existing manual benchmarks ignored.
- `cargo build --bins` and `cargo check --tests`: passed (existing warnings).
- `cargo test --test npu_tests -- --test-threads=1`: 13 passed, including scoped
  mock-plugin continuation, unsupported-rewind reset, owner separation and fresh
  output parity. This tests the real shim/process path, not accelerator hardware.
- Final ignored benchmark: passed; all 1,512 measured replies matched cold
  baselines and cross-policy request/output hashes agreed. The subsequent
  manifest verification matched every recorded source hash to that pre-review source.

An initial parallel library run on erskin failed only
`auto_placement::tests::cpu_probe_measures_both_shapes_within_the_deadline`,
`src/auto_placement.rs:457` (`start.elapsed() < BENCH_DEADLINE`): 494 passed,
1 failed, 10 ignored, 47.84s. An unchanged `99d2ddb` source archive reproduced
the **same assertion** with the same target/jobs/nice/Rayon settings and parallel
test harness (488 passed, 1 failed, 10 ignored, 42.94s). The initial changed run
also overlapped #157 validation; the unchanged reproduction did not need that
additional load. Both serialized implementation runs passed, including the
deadline test. No deadline/tolerance was relaxed and no failure was skipped.

The pre-review validation did not run accelerator hardware because local disk
was unsafe. The review validation below supersedes that hardware limitation;
CUDA and NPU hardware services/workloads were preserved. CPU synthetic timings do not establish a
production quantized-model speedup, full-device memory bound, or calibrated
saved-prefill-versus-queue cost model. Before closing #156, review the final
CPU evidence and the scoped physical-device correctness evidence; validate
realistic quantized-model multi-turn/load performance
if claiming a deployment latency improvement. This PR intentionally does not
auto-close #156 or claim universal latency improvement.

### Review validation (October 31, 2026)

The merged-main review revision preserves the external-draft integration fix:
draft session acquisition receives `options.cache_namespace()`. A new regression
compares scoped warm acquisition with cold draft verification and rejects reuse
across different scopes or from the unscoped pool. The full external-draft suite
also exercises all-position verification, rejection continuation and replay.

On erskin (Ryzen 9 5900HX, Linux x86_64, Rust 1.99.0), a source-only snapshot
and the local Cargo.lock were validated with jobs=2, incremental=0, Rayon=1,
nice=19 and serialized test execution:

- `cargo test --locked --lib --test affinity_tests --test server_tests
  --test route_tests --test external_draft_tests -- --test-threads=1`:
  498 library tests passed (10 manual tests ignored). After correcting the
  remaining unauthenticated eviction fixture and new test assertions to respect
  bounded pool replacement and exact-prefix rules, the final integration rerun
  passed all 6 affinity, 11 server, 36 routing and 8 external-draft tests
  (1 manual routing benchmark ignored).
- `cargo check --locked --features distributed,tls --lib --bins --tests`:
  passed. The same four integration suites with `--features distributed,tls`
  passed 6 affinity, 12 server, 37 routing and 8 external-draft tests, including
  both worker and coordinator HTTPS round trips (1 manual benchmark ignored).
- Default tests reused `/tmp/joshua-prefix-201-NXjVJ2AM/target`. Feature checks
  and tests reused the existing `/home/tserica/joshua/target` to avoid exhausting
  tmpfs (2.2 GiB remaining). No target directory or remote checkout was created.

Physical device correctness was executed on this Mac: Apple M5 Pro, 16 GPU cores,
Metal 4, macOS 27.0.1, Rust 1.99.0-nightly. With the existing shared local target,
jobs=2, incremental=0 and Rayon=1:

`cargo test --locked --features metal --test affinity_tests
scoped_metal_sessions_preserve_cpu_output_and_owner_isolation -- --exact --nocapture`

passed (1 test, 0 ignored, final test duration 0.89s). The test now fails if no
physical Metal device initializes, explicitly selects and asserts Metal for the
tested Engine and CPU for the reference, and checks greedy output parity on the
tiny llama head-dimension-32 fixture: scoped continuation, owner change, edited
and unrelated prompts, and context change. Thus CPU fallback cannot satisfy this
gate. Local free disk remained 7.8 GiB; no other artifacts/services were removed.

`measurements/affinity-247-review-validation.json` records the validated source
hashes (also matched against the remote snapshot), commands and limitations.
This is correctness acceptance evidence, not a performance benchmark. The older
timing JSON/manifests were not rewritten or rerun; realistic quantized-model
latency, CUDA/NPU hardware parity, peak device memory and universal speedup remain
unmeasured. Issue closure still requires maintainer acceptance of this evidence.

### Authenticated post-integration measurement (October 31, 2026)

After merging #247 into the #246 job-routing branch, the authenticated tiny-GGUF
benchmark was rerun on the Apple M5 Pro host above using CPU execution, the debug
profile, Rust 1.99.0-nightly, jobs=2, incremental=0, debug symbols disabled and
`RAYON_NUM_THREADS=1`. The 108 cells completed in 5.00s: all 1,512 measured replies
matched independently computed cold baselines, and request/output hashes matched
across all four policies. The raw report is
`measurements/affinity-246-post-integration-macos.json`; every source hash listed
in it was verified against the tested checkout. This is a new report, not a
replacement for either historical measurement.

At concurrency one, affinity saved 2,325 prefill tokens versus work-aware routing's
1,734, with median-cell-p50 TTFT 1.088ms versus 1.302ms. At concurrency two and
four, affinity's corresponding TTFT was 1.236ms and 2.119ms versus 1.161ms and
2.079ms for work-aware routing. Thus the authenticated run confirms locality and
correctness, not a universal latency improvement. Memory remains end-of-cell
warm/snapshot accounting, not peak RSS or device allocation. Host isolation and
production quantized-model performance are not established.

The combined branch passed 497 default library tests, 6 affinity, 8 external-draft,
52 routing and 11 server tests. With `distributed,tls`, 14 focused coordinator,
6 affinity, 8 external-draft, 53 routing and 12 server tests passed; the all-target
feature check passed. Manual benchmarks were run separately rather than counted
as ordinary tests. Both audio-admission regressions remain enabled: competing
zero-queue uploads coalesce refresh, and a freed generation slot is usable while
an unrelated audio probe is pending. The merge retains authenticated internal
cache scopes, stale-residency lease guards and conservative audio occupancy.

The report's source list predates the job-routing module split. Supplemental
SHA-256 values for this combined validation are:

- `src/coordinator/jobs.rs`: `723d0ba19c94fbe5540cf834a76abf9d9c015c628e805f22d775dcc186d53be7`
- `src/job_admission.rs`: `7eea0c91ecf33fa6cd2235e6a7e388f17f59e27dc083ad8d4e3d3dda97dc2b25`
