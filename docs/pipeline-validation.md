# Qwen3 pipeline prototype validation

This implements the first CPU-only, static-stage prototype for #158. It is a
library/example path; it does not replace `Engine` or `DeepSeekCluster`.

## Initial validation environment and scope

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

## Follow-up: mixed chunk schedules on two physical hosts (#158)

This acceptance slice adds tests only, not another execution architecture.
`mixed_chunk_schedules_match_full_prefix_across_stage_boundaries` compares
three interleaved sessions with 9-, 3-, and 1-token chunks, rotating result order,
against both the unchanged local same-chunk path and independently recomputed
whole-prefix logits. Distinct prompts exercise session isolation; equal prompts
with different chunk sizes exercise causal masking independently of chunking.
All sessions then decode through position 32, crossing KV allocation growth.
The matrix covers F32/Q8_0, heap/mmap loading and layer ends `[1,3]`, `[2,3]`,
`[1,2,3]`, with queue depth one. Same-host same-chunk logits remain bit-exact;
the full-prefix comparison uses the existing Qwen family's absolute `1e-4`
tolerance for different prefill shapes, not a newly loosened tolerance.

The ignored `two_host_deployment_matches_full_prefix_and_cache_continuation`
repeats the schedule via `Pipeline::deploy`: the controller creates tiny fixtures,
streams only assigned stage tensors, and compares every returned logit and greedy
choice to its local reference. It requires three dedicated workers, with the
first two on different physical hosts. Different IP addresses are a configuration
sanity check, not proof of physical placement (SSH forwarding may use aliases).
It reuses persistent workers for six deployments, including both two-stage
partitions. It does not use the production router or source-parity tool.

### Repeatable two-host procedure

Inspect host workloads first. Use fresh scratch directories, not existing model,
checkout, cache or build directories. On a compatible Linux pair, build once on
the idle controller host; transfer that executable to the other host:

```bash
E=$(ssh erskin 'mktemp -d /tmp/joshua-158-XXXXXXXX')
K=$(ssh kleya 'mktemp -d /tmp/joshua-158-XXXXXXXX')
git archive HEAD | ssh erskin "tar -x -C '$E'"
ssh erskin "cd '$E' && CARGO_TARGET_DIR='$E/target' CARGO_BUILD_JOBS=2 \
  CARGO_PROFILE_DEV_DEBUG=0 nice -n 15 cargo build -p joshua \
  --features distributed --bin joshua"
scp -3 "erskin:$E/target/debug/joshua" "kleya:$K/joshua"
```

Start each of these foreground commands in its own terminal, carrying over `E`
and `K`. Replace the LAN addresses for another host pair. Record each printed
ephemeral port. The public fixture key below is **test-only**: use disposable
workers on a trusted network, never a worker serving private models or requests.

```bash
KEY=1515151515151515151515151515151515151515151515151515151515151515
ssh kleya "env JOSHUA_CLUSTER_KEY='$KEY' RUST_LOG=info \
  nice -n 19 '$K/joshua' worker --listen 10.0.0.83:0"
ssh erskin "env JOSHUA_CLUSTER_KEY='$KEY' RUST_LOG=info \
  nice -n 19 '$E/target/debug/joshua' worker --listen 10.0.0.223:0"
ssh erskin "env JOSHUA_CLUSTER_KEY='$KEY' RUST_LOG=info \
  nice -n 19 '$E/target/debug/joshua' worker --listen 10.0.0.223:0 --no-mmap"
```

If the firewall blocks the remote worker port, do not change firewall rules.
In another terminal **on erskin**, forward to the printed kleya port:

```bash
ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -L 127.0.0.2:59158:10.0.0.83:KLEYA_PORT kleya
```

Use an unused local forwarding port. Set the first peer to `127.0.0.2:59158`
for that tunnel, or to `10.0.0.83:KLEYA_PORT` for direct TCP. Run on erskin,
substituting the actual ports and retained `E`:

```bash
cd "$E"
JOSHUA_PIPELINE_PEERS=127.0.0.2:59158,10.0.0.223:ERSKIN_PORT1,10.0.0.223:ERSKIN_PORT2 \
  CARGO_TARGET_DIR="$E/target" CARGO_PROFILE_DEV_DEBUG=0 CARGO_BUILD_JOBS=2 \
  nice -n 15 cargo test -p joshua --features distributed --test pipeline_tests \
  two_host_deployment_matches_full_prefix_and_cache_continuation \
  -- --exact --ignored --nocapture
```

Repeat with the first two peer addresses exchanged to move the remote stage
from embedding to output-head/middle-layer ownership. Stop only these dedicated
worker/tunnel commands after the test. Preserve their
logs and remove only the scratch directories allocated for this run when finished.
The test prints maximum absolute logit deltas and existing per-stage counters;
these debug-build counters are diagnostics, not a crossover benchmark.

### Measured scope and remaining acceptance

- Base: `bd718f2d9d921227cbb63d8312cb47e35a00da72` (fresh `origin/main`).
- Run date: October 30, 2026 (America/Los_Angeles; October 31 UTC).
- Controller: erskin, Ryzen 9 5900HX, Linux x86_64; remote stage host: kleya,
  Ryzen 9 9950X, Linux x86_64. CPU only; Rust `1.99.0`
  (`b940084d7`, 2026-09-28), debug build, two build jobs, low-priority workers.
  Kleya's ongoing `ds4-convert` regeneration and serving process were not stopped
  or reused. Scratch was `/tmp/joshua-158-1XBd2pCh` on erskin and
  `/tmp/joshua-158-EVYBBPnM` on kleya.
- Direct LAN connection timed out at kleya's firewall; validation uses an
  erskin-to-kleya SSH local forward, plus two controller-local worker processes
  (one mmap, one heap). This is real two-host computation, not three physical
  hosts and not an unencrypted direct-LAN throughput measurement.
- The first strict cross-host run passed F32 but found Q8_0 rounding differences.
  A separate temporary diagnostic using the **same executable on both CPUs**,
  the unchanged local loader/forward path, the nine-token prompt and 24 decode
  steps reproduced cross-host differences without pipeline execution: F32 zero,
  Q8_0 maximum absolute delta `2.384185791015625e-7` over 25 rows of 16 logits.
  Accordingly only the new cross-host same-chunk check uses absolute `1e-6`
  (tighter than the existing cross-shape `1e-4`), plus exact greedy-choice checks.
  Existing local bit-equality assertions are unchanged. No tensor/kernel changes
  are made to force different CPUs to share an accumulation implementation.
- ICMP erskin to kleya LAN: ten packets, no loss, min/mean/max RTT
  `0.114/0.158/0.250 ms`. This is not TCP/SSH RTT or application bandwidth.
- Both peer placements passed all six deployments: 12 total, F32/Q8_0,
  two/three stages, three sessions, 9-token prompts and 24 continuation steps.
  Across those runs, maximum same-chunk absolute delta was `3.427267e-7`;
  maximum whole-prefix delta was `2.3841858e-7`. Greedy choices matched exactly.
  F32 same-chunk comparisons had zero delta. Each stage logged 91 calls per
  deployment (including session open/close), serialized bytes, codec/compute/RPC
  time and input/downstream wait. These counters do not measure pipeline bubbles
  or peak resident memory.
- On erskin: pipeline tests 15 passed / 3 ignored by default; Qwen tests 10
  passed; Qwen3MoE tests 8 passed; distributed library tests 527 passed /
  12 ignored. The explicit three-worker-process greedy generation smoke test
  also passed. Default library and distributed all-target checks passed.
  The loopback release benchmark was not rerun. Existing compiler warnings
  remain; no failing suite is being labelled pre-existing or silently skipped.
- This does **not** complete #158: fixed-length/concurrency benchmark comparisons
  against local and tensor-sharded execution, isolated decode latency, TTFT,
  sustained tokens/s, pipeline bubbles, peak RSS/device memory, bandwidth,
  measured crossover and actual aggregate capacity gains remain unestablished.
  Sliding-window wrap is not supported by this full-attention Qwen3 path;
  accelerator parity is still required if device stages are implemented.
  Engine/HTTP integration and other activation schemas remain outside this
  prototype. Keep #158 open; make no speedup or capacity claim from this test.
