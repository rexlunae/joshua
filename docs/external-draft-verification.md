# Caller-supplied draft verification: local foundation for #160

## Status and scope

This is an experimental in-process API, **not a deployable remote speculative
worker**. Issue #160 remains open. No network routes, wire serialization,
authentication or encryption are added. A generation UUID is a local handle,
not authorization or a model/tokenizer fingerprint. Do not expose these types
directly as an RPC protocol.

The engine owns the target weights, KV and sampling. A caller supplies a
deterministic candidate prefix, such as prompt lookup. A learned stochastic
drafter's proposal probabilities are not supported; implementing that protocol
requires its own justified acceptance/correction algorithm.

## Local API contract

`Engine::begin_draft_session(prompt_ids, options, max_draft, lifetime)`:

- Requires this engine's tokenizer IDs, a nonempty prompt shorter than its
  effective context, draft bound 1–32 and a fixed monotonic lifetime greater
  than zero and at most 300 seconds. Generation options and seed are fixed
  for the handle's lifetime. The engine's output/context ceilings still apply.
- Acquires the engine's existing concurrency permit immediately; it does not
  queue. Configure a nonzero `Engine::with_max_concurrency` to bound admission.
  The engine's default unlimited concurrency is **not** a remote-worker limit.
- Refuses constrained decoding, NPU/remote backends and sessions without both
  all-position logits and safe rollback. A refusal is an error, not automatic
  remote-decode fallback. The caller may explicitly choose ordinary completion.
- Borrows its engine and exclusively owns a native session. It reuses existing
  loader/prefill paths and borrowed mmap weights; no private whole-model f32
  copy or new weight cache is introduced. Completed handles do not return KV to
  the warm pool.

`DraftSession::verify(&VerifyRequest)` requires its generation UUID, the next
sequence number (starting at zero), exact current KV offset (initially prompt
length) and at most the negotiated number of valid model-vocabulary IDs.
Malformed requests are rejected without advancing KV or sampling. Validation
includes all proposed IDs, even when generation limits will shorten the block.

An empty proposal performs ordinary one-token decode. For a nonempty block,
the saved prefill/previous-step logits score its first token; row `i - 1`
scores token `i`. Only the accepted prefix is retained, followed by a
correction at the first rejection or a bonus after full acceptance. Rejected
suffixes are rolled back, including recurrent state. Sampling stays on the
worker and uses the existing position-keyed sampler; there are no logits in
the response. This preserves the local engine's seeded output contract as
well as its sampled distribution, rather than consuming new RNG on retries.

Response fields have deliberately distinct meanings:

- `accepted`: matching draft tokens, including a matching EOS.
- `tokens`: emitted IDs, including a triggering stop-string token, excluding
  EOS. This is not necessarily visible text: the existing text decoder removes
  a matched stop string and may buffer partial characters.
- `offset`: tokens actually fed into KV, including prompt tokens. An accepted
  stop token was already fed; a correction/bonus stop token and EOS are not
  fed. Thus `offset` is not always prompt length plus `completion_tokens`.
- `completion_tokens`: emitted non-EOS token count, including stop-string
  tokens, matching local usage accounting.
- `finish`: EOS, matched stop, context exhaustion or output-token ceiling.
  Context takes precedence when context and token limits coincide.
- `stop_sequence`: the matched string, when applicable. `session.text()` is
  the authoritative decoded text, with the stop string removed. A future
  network streamer must withhold possible stop prefixes across RPC boundaries;
  simply detokenizing each response separately is incorrect.

Exactly the latest successful request/response pair is retained. An exact
retry returns an equal response without another forward pass, including the
terminal response. Conflicting duplicates and older/out-of-order requests are
rejected; no unbounded replay map is allocated. Terminal success frees KV and
the concurrency permit immediately while retaining the final replay pair.

`cancel` and `end` are idempotent and destroy replay, KV and admission. Dropping
the handle also releases its resources. An inference error fails closed and
cannot be retried against that handle. Invalid requests do not extend the
deadline. `verify` checks expiry; an idle owner must call `expire` or drop the
handle. There is no autonomous idle reaper. Text/history retained for inspection
is bounded by the engine context/output ceilings, but callers remain responsible
for limiting live handles and input option/string sizes.

## Validation

The tiny-fixture integration suite covers Qwen3 dense/MoE, Qwen3Next,
Qwen3.5 dense/MoE, DeepSeek-MoE/V2, GLM-DSA, DeepSeek-V4/V4.1, Kimi-Linear/K3,
GLM5Next and Nemotron-H. It exercises:

- Ordinary and local-speculative output/usage parity; seeded sampling with
  repetition, presence and frequency penalties; empty proposals, full acceptance,
  rejection at each position and cache continuation.
- Accepted, correction and bonus EOS, including first/later-position EOS;
  stop strings within/across blocks; zero/partial/full-block output limits
  and context exhaustion, with terminal response replay and released admission.
- Lost-response retries, conflicting/evicted replay, wrong generation, stale
  sequence/offset, invalid IDs and bounds, cancellation, end, expiry and drop.
- Non-finite target inference failures after admission and a verification
  forward pass, ensuring closed handles release KV/admission and reject retries.
- A Pearson goodness-of-fit test of the actual position-keyed verification
  sampler: 50,000 seeds, three nonzero probabilities plus a zero-probability
  draft, chi-square threshold 27.631 for two degrees of freedom (alpha 1e-6).

These are synthetic correctness tests, not remote performance measurements.
No remote TTFT, inter-token latency, accepted tokens/block, tokens/round trip,
bytes or target-compute speedup is claimed.

### Local validation record (2026-10-31)

Host: Apple M5 Pro, macOS arm64, rustc 1.99.0-nightly. Cargo used two build
jobs and the requested shared target directory. CPU fixtures used a six-token
prompt, 64-token engine context, up to 24 generated tokens and drafts of length
0–4; separate cases exercised the hard 32-token proposal bound, smaller request
contexts and the server output ceiling. Chunked prefill used two-token chunks.

| Check | Result |
| --- | --- |
| `cargo test --test external_draft_tests` | 7 passed; 14 architecture fixtures |
| `cargo test --lib --tests` | All 59 suite executions completed without failures |
| `cargo test --doc` | 1 passed, 1 ignored |
| `cargo check --all-targets --features distributed,tls` | Passed |
| `cargo test --features metal --test external_draft_tests metal_draft_verification_matches_local_decode_and_replays` | 1 passed; Qwen3 with explicitly requested Metal dense placement, no CPU fallback |
| `rustfmt --check` on the new Rust files; `git diff --check` | Passed |

The shared target was concurrently used by other checkouts. Differing sources
were touched to invalidate stale artifacts; an unrelated routing test executable
was still replaced during the broad run, so it was freshly rebuilt and rerun
separately (`cargo test --test route_tests`: 29 passed). No combined test-count
total is claimed for that shared-directory run. The external-draft suite and
forced-Metal test were also rebuilt and run directly. CUDA, OpenCL, SYCL and
Vulkan execution and remote performance remain unmeasured.

## Gates before #160 can close

1. An owned worker lifecycle/admission protocol, with per-principal and global
   session/KV-byte limits, bounded request bodies and option strings, disconnect
   ownership, cancellation races and autonomous expiry. A borrowed local handle
   alone is not a server registry.
2. Authenticated **and encrypted** transport with actual authorization binding
   each session to its creator. UUID secrecy is not authentication. Existing
   unrelated server TLS/bearer support does not secure nonexistent draft routes.
3. Stable protocol version, model/tokenizer artifact identities, capability
   negotiation and an ordinary remote-decode fallback for unsupported sessions.
4. End-to-end duplicate/dropped-response, disconnect/reconnect, expiry and
   failure tests over that secured transport, including concurrent requests
   and deterministic sampled replay. Existing local tests do not prove RPC
   behavior or disconnect reclamation.
5. Remote correctness/statistical validation, plus ordinary-remote versus
   block-remote benchmarks across draft lengths, acceptance rates, prompt
   patterns and measured/injected RTT. Report all metrics named above and
   separate prefill, target compute and network time. Low acceptance adds a
   wasted verification/rollback pass; cheap local prompt lookup by itself does
   not justify assumed compute savings or a claimed network speedup.

Shipping this isolated verifier provides a tested execution primitive without
implying that these security, lifecycle and performance gates have been met.
