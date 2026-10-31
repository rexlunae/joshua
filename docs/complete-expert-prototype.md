# Complete-expert layer prototype

This is a bounded CPU library experiment for #159, not an Engine backend,
distributed deployment command, or a performance result. It executes one
Qwen3-MoE SwiGLU layer using workers that own complete gate/up/down experts.
The coordinator supplies authoritative expert IDs and weights; workers return
unweighted rows. Attention, routing, KV, and shared MLP execution are not moved.
Models containing shared-expert tensors and architectures other than Qwen3-MoE
are rejected rather than silently losing their shared contribution.

## API and execution contract

Build with `--features distributed`. The types are exported under
`joshua::distributed::expert`:

1. Create `Limits` with row, route, distinct-session, frame-byte, scratch-byte,
   per-owner resident-byte, and I/O timeout bounds.
2. Use `Placement::from_gguf(path, layer, owners, limits)`. `owners` contains one
   rank per expert; ranks must be dense and nonempty. The placement includes
   the complete file's SHA-256, layer, architecture, projection formats/shapes,
   ownership, and limits. Placement fields cannot be supplied unchecked through
   deserialization. Each process derives the same placement from its model file.
3. Each owner calls `ExpertSet::load(path, placement, owner, mapping)` and serves
   an explicitly accepted TCP connection with `ExpertSet::serve`. Only its
   assigned experts are loaded. `None` selects streamed reads; a provided
   immutable mapping is identity-checked. Keep the mapped file immutable for
   the mapping's entire lifetime.
4. Supply one address per owner in rank order to `Coordinator::connect`, using
   a fresh v4 job UUID and shared authentication key. The application owns
   listener binding, address/key distribution, and worker lifecycle.
5. Call `Coordinator::forward(rows, input, ids, weights)` with unique
   `RowId { session, token }` positions, CPU F32 activations `[rows, hidden]`,
   U32 expert IDs `[rows, selected]`, and F32 weights of the same shape.
   `token` is a sequence position, not a vocabulary ID. Distinct sessions are
   bounded per call; workers do not hold session KV.

Exactly one forward is in flight per coordinator. Rows are bucketed by owner,
then expert and original route ID. Duplicate expert selections remain separate
routes. Single-row decode evaluates each selection separately and reconstructs
local slot-order weighting/reduction; multi-row prefill retains local
expert-major accumulation. Owner reply order never determines summation order.
All decode selections on one owner borrow the same input tensor; inconsistent
copies of a single position's activation are rejected before computation.

Malformed frames, nonfinite activations/results/weights, identity or sequence
mismatches, invalid routes, socket failures, and computation errors fail the
whole forward. The coordinator then closes all channels and refuses reuse;
create a fresh job, not a partial-output retry. Dropping a coordinator closes its
channels. A worker exits when its connection closes. Connections and framed I/O
have configured timeouts; tensor computation is not preemptible. Idle workers
wait for the first byte without an idle timeout, so healthy jobs survive pauses
between forwards. Once a frame starts, its entire receive remains deadline-bound;
coordinators also retain their bounded response wait.

## Memory, formats, and security

Quantized projections remain compact QTensor data; mmap is borrowed where the
existing backend supports it, otherwise only the selected expert is read.
F32 projections use the existing dense Tensor execution path. Forcing F32
through the quantized matmul path changes rounding and is deliberately avoided.
Accepted formats are F32, Q2_K, Q4_K, Q6_K, and Q8_0; only F32 and Q4_K have
fixture parity coverage in this slice. No entire quantized expert pool is
expanded to F32 and no weights are copied per session.

Selective expert residency does **not** imply selective startup I/O.
`Placement::from_gguf` scans the entire file to establish its SHA-256 identity.
Each `ExpertSet::load` verifies that same full-file digest once through the
handle used for loading, then parses its header once and reads only owned
projections. A separately supplied mapping requires an additional digest scan
to verify its bytes too. File and mapping inputs must remain immutable. Skipping
verification based only on file path, size, or timestamps would weaken the
exact-revision guarantee; a reusable verified-model handle is future work.

Admission validates owned projection bytes and reserves conservative frame,
activation/output, metadata, and dense CPU temporary workspace. These are
algorithmic admission bounds, not an OS RSS quota or a measured memory peak.
Placement also rejects frame budgets that cannot hold the worst-case admitted
route payload and JSON metadata (including maximum-width identifiers). The same
bound covers requests and unweighted responses, including fully skewed routing.
There is no unbounded application queue, timed cross-request batcher, expert
replication, migration, or fallback. A second forward waits at its caller.

The existing HMAC-authenticated pipeline framing is reused with a distinct
`joshua-complete-expert-v1` domain and owner/job/sequence binding. Authentication
does **not** encrypt activations or hide traffic metadata. Use a trusted network
or an encrypted tunnel; all holders of the job key are trusted. This is not an
Internet-facing service, a key-management system, or a TLS deployment example.

## Validation and remaining gates

Focused commands:

```sh
cargo test --features distributed --lib distributed::expert::tests
cargo test --features distributed --test expert_ownership_tests
```

The integration fixture compares actual local GGUF expert execution with two
loopback owners, exactly (no tolerance relaxation), for F32 and Q4_K, streamed
and mapped loading, 1/3/5 activation rows, repeated calls with independent
session/position IDs, duplicate routes, and tied router scores. Unit tests
exercise out-of-order owner replies, cancellation-sensitive reduction order,
malformed/oversized/nonfinite frames, route and session bounds, identity and
sequence mismatches, wrong authentication, disconnects, and fail-stop reuse.
Admission tests also reject insufficient budgets and shared-expert models.

Each reference uses the corresponding established local loader path:
`ExpertTensor::borrow_host` for mapped experts and `read_and_split` for streamed
experts, not the new owner's single-expert reader. On x86, the borrowed k-quant
kernel uses F32 activations whereas Candle's streamed kernel can quantize
activations. Cross-path outputs are therefore not assumed bit-identical. The
original test incorrectly used a streamed reference for both modes and failed
the mapped Q4_K duplicate-route case in Linux release CI; this is corrected
without changing production kernels, remote execution, or exact assertions.
On erskin's x86 Linux CPU, the pre-fix release test reproduces the CI failure;
after correction all four expert tests, 12 memory-placement tests, and eight
Qwen3-MoE tests pass with `--release --features distributed`. All four expert
tests also pass on the local macOS CPU. These are backend-matched correctness
checks, not cross-backend bit-equivalence or throughput measurements.

Local validation on macOS CPU: seven protocol unit tests and four complete-expert
integration tests pass. The default library suite passes (480 tests, 10 ignored),
as do the Qwen (10), Qwen3-MoE (8), and memory-placement (12) suites. The
`cargo check --all-targets --features tls,distributed` build passes with existing
warnings. These are tiny-fixture correctness checks, not timing measurements.

On macOS, the broader distributed library suite has five multicast socket
failures (`Address already in use`, OS error 48), also reproduced on unchanged
base `22e02672a5655806e2eff488a0be6a79cdb61877`: `completion_ack_prevents_leaving_a_peer_without_data`,
`loopback_consecutive_nonuniform_partial_vectors`,
`loopback_retries_deterministically_dropped_data_and_acks`,
`mismatched_lengths_cannot_succeed`, and
`rank_order_is_deterministic_despite_reordered_arrival`. Serial execution does
not resolve the host's multicast bind behavior. No multicast code is changed.
After the review follow-up, the distributed library result is 530 passed,
five failed, and 12 ignored (the same five multicast bind failures).
The pipeline suite also reproduces two unchanged-base exact-logit failures:
`model_less_workers_load_their_stage_from_the_controller_bit_exact` and
`mixed_chunk_schedules_match_full_prefix_across_stage_boundaries` (13 passed,
two failed, three ignored). Assertions and tolerances are unchanged; this
prototype does not modify pipeline tensor execution.

The review regressions reproduce all three reported bugs when the fixes are
removed: healthy forwards fail after a 600 ms idle gap with a 250 ms frame
timeout, 1-byte/1 KiB budgets admit incompatible routes, and decode route inputs
have different Tensor identities. They pass with the fixes applied. A partial
frame still times out, worst-case metadata/payload reaches the computed bound
exactly, and modified source or mapping bytes still fail identity verification.

Still required for #159: architecture/Engine integration, shared experts,
cross-host validation, real-model all-row logits and generation parity,
explicitly bounded waiting-time batching across Engine requests, measured RSS,
local-versus-tensor-sharded-versus-remote benchmarks, queue/compute/network
breakdowns, routing-skew and concurrency sweeps, and measured network/batch
crossover. No accelerator execution path changes here, and no capacity or
latency improvement is claimed. #159 remains open.
