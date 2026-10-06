# Qwen3 pipeline prototype validation

This implements the first CPU-only, static-stage prototype for #158. It is a
library/example path; it does not replace `Engine` or `DeepSeekCluster`.

## Environment and scope

- Host: Apple M5 Pro, arm64, macOS.
- Rust: `1.99.0-nightly (d453bdd8f 2026-08-14)`.
- Unchanged base: `f7b1dffd4e0ec04b03b39fbc3cdb932d34d976c0`.
- Network: TCP loopback; no external LAN or accelerator measurements.
- Parity: F32 and Q8_0 tiny Qwen3 fixtures from `tests/common`, three layers,
  two/three stages, mmap and streamed/copying loads. Bit-identical local logits
  across prefill chunks, interleaved sessions, decode and KV buffer growth.
- Standalone example: three separate worker processes, Q8_0 weights, two
  requests, five prompt tokens in chunks of three, eight greedy output tokens.
  Generated IDs match the local model.

## Checks

```bash
cargo check -p joshua --lib
cargo check -p joshua --features distributed --all-targets
cargo test -p joshua --features distributed --test pipeline_tests \
  --test qwen_tests --test qwen3moe_tests
cargo test -p joshua --features distributed --lib
cargo test -p joshua --features distributed --test deepseek4_tests
```

The focused pipeline/Qwen suites and default/distributed build checks pass.
Explicit diagnostics require `--ignored`: the warm release loopback benchmark
and the standalone example process test. See README for commands and timings.

The wider suites have these failures, reproduced on the unchanged base in a
separate temporary worktree (without altering the working checkout):

- `deepseek4_forward_sequences_matches_single_sequence` at
  `tests/deepseek4_tests.rs:839`: sequence 1 max relative logit delta
  `0.00019835481`, absolute `1.535e-6`. The same numerical values appear on the
  unchanged base. The remaining 26 DeepSeek4 tests pass; two diagnostics are
  ignored by default.
- Five UDP collective tests fail while constructing sockets with
  `Address already in use (os error 48)` on this macOS host:
  `completion_ack_prevents_leaving_a_peer_without_data` (line 738),
  `loopback_consecutive_nonuniform_partial_vectors` (line 627),
  `loopback_retries_deterministically_dropped_data_and_acks` (line 627),
  `mismatched_lengths_cannot_succeed` (line 715), and
  `rank_order_is_deterministic_despite_reordered_arrival` (line 762), all in
  `src/distributed/collective.rs`. The full library run has 321 passing,
  five failing and seven ignored tests. A base run filtered to
  `distributed::collective` reproduces all five failures, with four passing
  and one ignored test.

Base reproduction commands:

```bash
cargo test -p joshua --features distributed --test deepseek4_tests \
  deepseek4_forward_sequences_matches_single_sequence -- --exact --nocapture
cargo test -p joshua --features distributed --lib distributed::collective
```

An initial all-target check exhausted local disk space. After removing only this
task's disposable release build products, the check was rerun successfully.
The saved benchmark numbers were collected before removing those products;
rebuild the example/release tests to repeat the diagnostics.

## Practical limits and follow-up

- The scheduler creates dispatch threads per `forward_batch`, while worker TCP
  connections and model weights remain persistent. Its overhead appears in
  end-to-end timings. A production scheduler should profile persistent dispatch
  threads and transport-buffer reuse.
- The full-file startup hash proves model identity but scans every weight byte;
  stage-selective construction does not imply stage-selective startup I/O.
- Activation traffic takes two network hops per stage boundary through the
  coordinator. Direct worker-to-worker transport is a potential next experiment.
- Stage memory reservations are tensor estimates, not hard RSS enforcement.
  File-page residency, allocator/thread overhead and model-sized peak memory
  have not been measured. No aggregate capacity gain is established yet.
- Qwen3 here uses full attention. Sliding-window wraparound, other residual
  schemas, accelerator stages, speculative all-row logits and server integration
  are outside the implemented path.
- TCP frame HMAC authenticates traffic without encryption. Use a private network
  or encrypted tunnel. Worker failure or active abort invalidates the whole job;
  there is no recovery, partial-forward replay or dynamic membership.
- The local benchmark baseline processes independent requests serially; it is
  not a comparison against an optimized local concurrent scheduler. Larger
  models, real network RTT/bandwidth, latency percentiles and sustained throughput
  need separate measurements before claiming a production performance gain.
