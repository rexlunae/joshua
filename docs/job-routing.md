# Bounded embedding and whole-audio routing

This implements the job scheduler portion of #157, not its complete production
acceptance matrix. Keep #157 open until the gates below have evidence.

## Registry and compatibility

`/v1/worker/info` optionally advertises `embedding_jobs`, `transcription_job`,
and separate `transcription_in_flight`. Old workers remain readable and use the
unsplit embedding fallback. Embedding capability is advertised only for the
local Llama/Qwen2/Qwen3 loaders that implement it. Whisper-only servers can
advertise transcription without a chat model.

Each resident model has a random owner identity. Embedding revision hashes cover
the actual mapped weights, serialized tokenizer, context length, and an embedding
implementation marker. Hashing happens at registry load, not per job; it touches
the whole mmap and its large-model startup/RSS cost remains unmeasured.
Fan-out requires the same model ID, nonmissing matching revision, backend and
input format. A missing revision pins all subbatches to one resident owner.
Only healthy, embedding-eligible compatible owners contribute subbatch limits;
an offline owner's cached limits cannot reduce a healthy owner's batches.
Owner snapshots and `x-joshua-job-owner` reject unload/reload races. A different
backend is deliberately not considered interchangeable even with identical weights.

Every subbatch preserves the original JSON options, including unknown fields;
only `input` is replaced. Requests are not coalesced across clients or option
sets. Existing accepted-but-ignored options are still ignored, not newly
implemented: notably embedding encoding options and Whisper prompt, temperature
and timestamp controls. Audio multipart bytes/boundaries are forwarded intact;
one whole WAV job executes on one Whisper worker, with no segmentation.

## Bounds and lifecycle

Limits apply independently at each coordinator and worker router, after auth but
before request-body extraction. Exhausted budgets return 503 `job_budget_full`
with `Retry-After: 1` without polling the upload.

| Resource | Bound |
| --- | --- |
| Resident embedding request leases | 32 |
| Resident whole-audio request leases | 2, independent of embeddings |
| Embedding request body | 2 MiB |
| Whole multipart audio body | 25 MiB, including fields/boundaries |
| Embedding inputs | 2,048 per request |
| Embedding subbatch | 32 inputs / 256 KiB JSON, or smaller worker limits |
| Concurrent subbatches per parent | At most 4, also at most queue depth + 1 |
| Subbatch response | 2 MiB, or smaller configured/advertised worker limit |
| Aggregate/direct embedding response | 8 MiB, or smaller coordinator limit |
| Transcription response | 2 MiB |
| Locally dispatched audio per worker | 1, separate from model embedding slots |

The request leases bound retained raw queued uploads to at most 64 MiB of
embedding bodies plus 50 MiB of multipart audio per router, not total process
RSS. Parsing, tensors, output objects and serialization add bounded copies or
model-dependent working memory. Response leases remain held through body drain
or drop; worker blocking closures retain their leases until actual exit.
The coordinator uses existing configured queue depth/timeout (defaults 64/30s)
and response timeout (default 600s). Queue timeout is per acquisition, not an
absolute whole-parent deadline. Requests may be rejected rather than queued.

Subbatch encoding is lazy, bounded by active futures, and serialized `Bytes`
are shared across attempts. Aggregation verifies local indexes, row count,
model, vector shape/encoding and usage overflow, then restores original order.
Malformed output fails the entire request without returning partial results.

Disconnect/error drops pending subbatches and outbound connections. Workers
retain the merged #240 cooperative cancellation and asynchronous Whisper
admission; individual Candle calls and whole-audio mel remain non-preemptible.
No temporary payload files are created. Audio slots account for observed external
Whisper work and explicit worker-wide caps without changing affinity scoring.
An unsent reservation is not proof that this coordinator owns the worker's
Whisper permit. Refresh therefore retains the entire observed audio count when
ownership is ambiguous, separately from generation occupancy; it never subtracts
local audio reservations from that count. Admission uses current local audio
reservations plus this conservative observed snapshot. A reservation-completion
epoch invalidates occupied snapshots taken before local release (including a
probe still in flight at release). New and queued acquisitions re-probe these
snapshots on demand before selecting, so completed local jobs do not wait for the
periodic health interval. A fresh probe that still observes external audio keeps
admission blocked, including after a local 503 refusal or unsent cancellation.
On-demand refreshes are coalesced to one per coordinator, with queued waits still
bounded by the existing deadline; refresher cancellation releases the refresh
lock and wakes waiters rather than stranding them until a periodic probe.
Unknown ownership can conservatively double-count local audio against an explicit
worker-wide cap during execution, sacrificing utilization rather than hiding
external work. No extra probe is needed when the cached audio count is zero.
Sequential uploads rotate their
starting worker under the selection lock; the independent audio permit and
configured worker-wide cap still apply. External occupancy remains a probe
snapshot, not a globally synchronized reservation across coordinators.

## Retry and security contract

Subjob IDs hash an independent random per-request identity plus ordinal. They
are stable for internal attempts within that logical request, but independent
of the client-visible `x-request-id`, even when a client reuses that header.
This is not a durable deduplication store or a client replay guarantee.
Retries use each eligible worker at most once and only after a pre-send failure
or a worker 503 admission refusal. Transport/result loss after sending is
ambiguous and is never automatically re-executed for embeddings or audio.
Workers must obey the 503-before-execution admission contract. A process crash
loses all in-memory tracking. Audio has no revision hash: an admission retry can
select a different Whisper model; cross-worker deterministic audio retry
compatibility is not guaranteed.

Client authentication and configured `worker_api_key` bearer authentication
remain in place. All payload dispatch, including chat/completions, fails closed
with 503 `secure_transport_required` for non-loopback worker addresses: the current
coordinator transport is HTTP, even when built with the server `tls` feature.
The central send path rejects these endpoints before connecting, so refresh
probes also cannot transmit bearer credentials. Non-loopback workers remain
unhealthy rather than entering the routing pool. Direct private-network worker
addresses previously used for chat/completions and probes must be migrated.
For multiple hosts, expose workers through authenticated encrypted tunnels
bound on loopback and configure worker bearer keys. This code cannot verify the
security of a user-managed tunnel; see `request-routing.md` for an SSH forwarding
example and the tightened compatibility contract. Native authenticated encrypted
remote transport remains a gate, not something supplied by a flag.

## Evidence and reproducibility

Focused CPU library, route, server, controller and Whisper tests cover reordered
completion, local index restoration, options/usage, compatibility, owner reload,
input/output budgets, pre-upload overload, independent audio admission, actual
worker auth, deterministic synthetic model parity, disconnect cancellation,
malformed/oversized audio, pre-execution retries and ambiguous result loss.

Final Linux CPU validation on `erskin`, using the existing shared target and
`nice -n 19`, passed 492 library tests, 40 route tests, six Whisper tests,
nine server tests and three controller tests (11 ignored across these suites).
The `distributed,tls` library feature check passed. A preceding concurrent
Linux library run failed the CPU-probe wall-clock deadline assertion in
`auto_placement::tests::cpu_probe_measures_both_shapes_within_the_deadline`;
the complete final run with `--test-threads=1` passed. This is recorded as a
contention-sensitive validation failure, not claimed to be a reproduced
unchanged-base failure. Earlier macOS focused validation also passed.

Run with `CARGO_BUILD_JOBS=2` and the environment's shared `CARGO_TARGET_DIR`:

```sh
cargo test --lib --test route_tests --test whisper_tests --test server_tests --test controller_tests --no-default-features -- --test-threads=1
cargo check --lib --no-default-features --features distributed,tls
cargo test --test route_tests benchmark_embedding_jobs_loopback_acceptance_matrix -- --ignored --nocapture
```

### PR #246 review follow-up, October 31, 2026

The six review findings are covered by focused regressions: a probe during a
locally dispatched audio job followed by completion and immediate readmission
without waiting for a periodic probe, external audio blocking, sequential audio rotation and
the explicit worker cap, unhealthy/ineligible cached embedding owners with
one-byte limits, repeated client IDs with independent internal IDs, and stable
IDs across an admission retry. Transport tests exercise chat, completion,
embedding and audio sends plus repeated refresh probes with a configured secret
and sensitive payload; no TCP connection reaches the non-loopback listener.
Existing loopback auth and real synthetic Llama/Whisper parity tests still pass.

Initial review validation at `9044cbc` ran on Erskin Linux x86_64 CPU, from the unique source copy
`/tmp/joshua-246-review-9EDvdW5C`, using the existing
`/tmp/joshua-prefix-201-NXjVJ2AM/target`, `CARGO_BUILD_JOBS=2`, `nice -n 19`,
and serial test execution after #247 released the lane. Source hashes for both
coordinator files and route tests matched this checkout. No local build or new
target directory was used; existing services and other worktrees were untouched.

- Default-feature library/controller/route/server/Whisper suites: respectively
  494 / 3 / 44 / 9 / 6 passed, zero failures; 10 library tests and the exploratory
  route benchmark remain ignored. The mock-NPU library has zero unit tests.
- `cargo check --all-targets --features distributed,tls`: passed.
- `cargo build --lib --bin joshua --features distributed,tls`: passed.
- `cargo test --features distributed,tls --test route_tests -- --test-threads=1`:
  45 passed, including HTTPS round-trip, one exploratory benchmark ignored.
- `cargo test --lib --features distributed,tls coordinator:: -- --test-threads=1`:
  11 passed, including transport rejection and bounded partition tests.
- `git diff --check`: passed. Existing compiler warnings are not changed here.

This follow-up measures correctness/admission, not throughput or latency; tiny
synthetic GGUF fixtures and small fake-worker payloads do not substitute for
pretrained real-audio, real-model or multi-host acceptance. No accelerator parity,
peak RSS, network crossover, or independently verified remote SSH/tunnel run was
performed. The gates below remain open, including #157; #247's supported-device
parity also remains open.

The subsequent ownership-race review adds
`unsent_audio_reservation_never_erases_external_occupancy`: it reserves locally
without sending, starts simulated external worker occupancy, probes, then either
drops the unsent reservation or sends and receives an admission refusal. In both
cases occupancy remains one, the next acquisition re-probes and stays blocked,
and no second upload reaches the busy worker. External release plus a probe
restores admission. `completed_audio_wakes_queued_upload_without_periodic_probe`
checks that a waiting upload resumes after local completion using on-demand
revalidation, while the earlier zero-queue-depth completion regression still
passes. `audio_revalidation_is_coalesced_and_cancellation_wakes_waiters` verifies
that competing acquisitions do not start duplicate refresh connections and an
aborted refresher unlocks and notifies queued work.

Final ownership-race validation used the unique source directory
`/tmp/joshua-246-audio-race-sVbg5CBo`, the same existing target, jobs=2,
nice=19 and serial tests. Other observed CPU-heavy work was allowed to finish
before these runs started; no processes or services were stopped. Source hashes
again matched the local coordinator files and route tests. Results:

- Default-feature library/controller/route/server/Whisper: 496 / 3 / 45 / 9 / 6
  passed (559 total), zero failures; the same 11 tests remain ignored.
- Combined `distributed,tls` all-target check and explicit library/binary build:
  passed, using the exact check/build commands above.
- Combined-feature route suite: 46 passed, one exploratory benchmark ignored;
  focused coordinator suite: 13 passed, including both new unit regressions.
- `git diff --check`: passed. All prior device, throughput/RSS, real-audio and
  independently verified remote-tunnel acceptance gaps remain unchanged.

The ignored benchmark emits JSONL for 1/2/4 workers, 1/32/128 inputs,
1/8 words and concurrency 1/4, direct versus routed, with warmup then eight
requests per cell. All workers share one process; direct targets one worker.
It measures documents/s, observed p50/p95 and client upload bytes, not wire
bytes, peak RSS or an inter-host network crossover. With eight observations
p95 is the maximum sample, not a statistically reliable production percentile.

October 31, 2026 exploratory run: Apple M5 Pro, arm64, CPU, debug build,
tiny synthetic Llama GGUF, loopback TCP. Single input/word/concurrency with one
worker: direct 1,850 docs/s (p50 0.521ms, p95 0.619ms), routed 1,168 docs/s
(0.854ms, 0.938ms). At four workers, 128 inputs, eight words, concurrency four:
direct 8,016 docs/s (63.45ms, 64.05ms), routed 19,969 docs/s (25.06ms, 26.12ms).
These illustrate dispatch overhead and fan-out, not real-model acceptance.

## Gates before issue closure

- Native encrypted/authenticated remote transport, including registry probes,
  or a separately verified secure deployment contract.
- Durable idempotency if automatic ambiguous post-send retries are required;
  defined compatible Whisper identity for retries across different owners.
- Pretrained deterministic whole-audio text/timestamp accuracy and supported
  decoding-option parity, not only synthetic Whisper pipeline parity.
- Release-build, real-model 1/2/4-host benchmarks: documents/s, audio-seconds/s,
  queue wait, p50/p95, actual upload/retry bytes and peak per-process RSS.
- Measured small-job network crossover and routing policy; direct worker APIs
  remain available, but automatic local/direct crossover selection is absent.
- Separate CPU and available accelerator evidence, plus large-model revision
  hashing startup/RSS measurement. No accelerator performance claim is made.

Overlap with #156 / PR #247 (`5ca32afa361ce0b1ea1a35c74fb0155310367975`):
`src/coordinator.rs`, `src/server.rs`, and `tests/route_tests.rs` are shared files
across the PRs. This review follow-up changes coordinator probe accounting and
transport plus route regressions; it does not change `src/server.rs` or affinity
scoring. Expect coordinator/test merge conflicts; integration follows green
validation. Neither #247's worktree nor the root checkout is modified. #156's
supported-device parity gate and #157's acceptance gates remain open.
