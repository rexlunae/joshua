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

## Retry and security contract

Subjob IDs hash the parent request ID plus ordinal. They are stable for internal
attempts, not a durable deduplication store or a client replay guarantee.
Retries use each eligible worker at most once and only after a pre-send failure
or a worker 503 admission refusal. Transport/result loss after sending is
ambiguous and is never automatically re-executed for embeddings or audio.
Workers must obey the 503-before-execution admission contract. A process crash
loses all in-memory tracking. Audio has no revision hash: an admission retry can
select a different Whisper model; cross-worker deterministic audio retry
compatibility is not guaranteed.

Client authentication and configured `worker_api_key` bearer authentication
remain in place. Text/audio payload dispatch fails closed with 503
`secure_transport_required` for non-loopback worker addresses: the current
coordinator transport is HTTP, even when built with the server `tls` feature.
For multiple hosts, expose workers through authenticated encrypted tunnels
bound on loopback and configure worker bearer keys. This code cannot verify the
security of a user-managed tunnel. Health/registry probes still use the existing
HTTP transport; native authenticated encrypted remote transport remains a gate.

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
cargo test --lib --test route_tests --test whisper_tests --test server_tests --test controller_tests --no-default-features
cargo check --lib --no-default-features --features distributed,tls
cargo test --test route_tests benchmark_embedding_jobs_loopback_acceptance_matrix -- --ignored --nocapture
```

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

Overlap with #156: `src/coordinator.rs` and `tests/route_tests.rs` are shared
files. This patch adds audio admission and job dispatch but does not alter the
affinity scoring algorithm.
