# Phase-0 prefix-locality measurement

Part of #201 and #156; also exercises the existing #155/#238 routing path.
This is an **experiment**, not production affinity. No router index, affinity
header, tokenizer, cache advertisement, or KV transfer is introduced. The RFC's
evidence gate for phases 3–5 remains open.

## Repeat a run

Requires Python 3.11+, `ps`, a default CPU `joshua` binary, and a GGUF with its
matching `tokenizer.json` alongside it. No Python dependencies or downloads.

```sh
cargo build --release --bin joshua
python3 scripts/prefix_locality.py \
  --binary target/release/joshua \
  --model /path/to/Qwen3-0.6B-Q4_K_M.gguf \
  --output /tmp/prefix-locality-unique-run \
  --workers 2 --prefix-counts 2,6,12 --concurrency 1,2 \
  --rounds 3 --repeats 2 --prefix-words 64 --max-tokens 4 \
  --n-ctx 512 --cache-mib 128 --root-min 16 --threads 2
```

`--output` must not exist. Keep `report.json` **and all worker/router logs**.
Each policy/cell uses newly loaded independent workers, unique loopback ports,
and separate logs; existing servers are never reused or stopped. Processes
started by the harness are reaped on success, failure, or interruption. Port
selection briefly releases an ephemeral reservation before spawning; a competing
bind causes startup failure rather than deliberate reuse of an existing service.
Record the source revision/build flags in your experiment notes; binary, model,
tokenizer, request and trace SHA-256 digests are in the report.

Inspect host workloads before running. `--avoid-process-substring joshua-117`
conservatively aborts if a matching executable, cargo/rustc argument, or Linux
process working directory is observed before/after a wave. Discard the interrupted
cell; the report records the error. This is a sampled guard, not resource
reservation or proof that no brief overlap occurred. It cannot detect a renamed
job elsewhere. Never kill another experiment to obtain a clean host. On Linux,
use `taskset`/`nice` externally if appropriate; `--threads` sets
`RAYON_NUM_THREADS`, **not** a hard CPU quota for every engine kernel.

For a one-worker hop/control run use `--workers 1`; for four colocated workers
use `--workers 4`. These are process-count tests, not physical-host scaling.
Unless `--skip-calibration` is set, the separate warm/cold control always starts
two workers, even in a one-worker matrix. Sweep prefix length with
`--prefix-words` in separate output directories. The words are synthetic,
not a real conversation or production prefix-overlap distribution.

## What is compared

- **load:** the ordinary `joshua route` binary, with #238 model/context and
  outstanding-work admission. All requests in a cell have the same prompt length
  and output budget; no locality hints are sent.
- **oracle:** the client already knows each synthetic shared system head's ID,
  `SHA256(canonical JSON [model_id, system_text])`. Highest-score rendezvous
  hashing of `[prefix_id, worker_ordinal]` fixes its worker. This is *not* the
  salted post-template token chain from #234, not a residency oracle, and not an
  implementation of #156's authenticated conversation handle. Workers still
  validate reuse themselves. Imbalance, eviction, and a busy owner can hurt it.
- Oracle requests go **directly** to that worker, avoiding the router hop.
  Per-worker client semaphores respect `--slots`; otherwise the worker rejects
  simultaneous pinned requests instead of queueing. This bounded client queue
  does not spill to another worker. Its wait is included in TTFT/latency and
  reported separately. Admission waits at most `--timeout` seconds, then fails
  without sending; the HTTP socket timeout applies separately after admission,
  as the router also has separate queue/response timeouts. This is deliberately
  not a production load-aware policy.
  A one-worker run helps bound the hop/harness difference; the policy delta alone
  cannot be attributed entirely to locality.
- Every round shuffles all prefix IDs with the same seeded PRNG. Each request
  opens a new conversation sharing only its system head, with a round-specific
  user suffix. `a`/`b` prefix bits and `c`/`d` suffix bits survive the tiny
  WordLevel tokenizer too. `--prefix-words` is the number of alternating
  `hello`/`world` words **after** a 32-word identity. Token counts come from worker
  usage, not a character-count approximation.
- Both policies receive byte-identical request bodies, greedy temperature 0,
  the same seed, `max_tokens`, context and worker configuration. A fixed wave
  contains at most `concurrency` requests; a barrier separates waves. Within a
  wave, OS/executor scheduling is not deterministic. This is closed-loop fixed
  concurrency, **not fixed offered QPS**. No HTTP retries mask failures.
- Cold-start state is reset **between policies**, not between rounds. Repeats
  alternate load/oracle order. SHA-256 compares the concatenated content,
  reasoning content, finish reason, and full usage for each request index;
  transient IDs/timestamps are intentionally excluded. HTTP errors, malformed or
  truncated streams, missing terminal usage/finish/`[DONE]`, changed request
  bodies, or changed outputs fail the run. No tolerance is relaxed. A failed
  cell still has its errors in `report.json`; failures must not be dropped from
  a headline latency comparison.
- The separate calibration warms worker 0 with a request, then sends the **same
  request** to that warm worker and untouched worker 1. It alternates cold/warm
  measurement order across repeats, checks output equality, and retains the
  exact sequential engine completion records. This isolates deliberate cold
  recomputation without a router hop. Repeating an entire prompt can reuse more
  than a shared-root-only new conversation, so it is a control, not the matrix's
  expected savings. A single pair is not a statistically established penalty.

## Metrics and bounds

| Report field | Meaning and limitations |
|---|---|
| `ttft_ms` | Nearest-rank p50/p95 from dispatch to first nonempty content **or reasoning** delta; headers/role events are not tokens. Empty-output successes have null TTFT, not zero. Oracle includes client admission wait. |
| `latency_ms`, `elapsed_seconds` | Full stream duration and closed-loop cell time, excluding loading/probing/calibration. First-round and later-round TTFT are also separate; neither is automatically a cache-miss/hit classification. |
| `completion_tokens_per_second` | Successful completion tokens / full cell elapsed time, including errors and wave barriers. Also retains total prompt/completion tokens and every request status. |
| `request_body_bytes`, `response_body_bytes` | Client-hop application body bytes, SSE framing included in responses. **Not NIC bytes**: excludes HTTP/TCP/TLS headers, probes and router–worker duplication/retries; error bodies are read only up to 4 KiB. No directory exists, so directory/KV wire fractions are unmeasured, not claimed zero-cost. |
| `queue_ms` | Router `routed stream finished` admission log p50/p95, integer milliseconds and record count. Null for direct oracle. Excludes engine-internal scheduling and is not a precise prefill/decode split. |
| `client_queue_ms` | Oracle semaphore wait p50/p95, includes pinned-owner imbalance; null for load. Not interchangeable with worker queue telemetry. |
| `reuse` | Aggregate `Completion finished` log counters: reused and prefilled tokens, requests with any reuse, root-store events and completion-record coverage. Separate log-event counts distinguish root/parked-cache resumes and pool rewinds. Reuse is **not** root-cache hit rate or live occupancy. No `had=` header is implemented today. Missing completion logs mean null reuse tokens, not zero. |
| `prefill_seconds_per_token_from_logs` | Sum(prefill_tokens / logged prefill_tps) / sum(timed prefill_tokens); timer-rounded zero-rate records are excluded and coverage retained. Engine-only timing, including current prefill/snapshot behavior; not end-to-end TTFT per prompt token. |
| `rss_sampled_process_sum_bytes` | First/last/max sum of live worker (+ router for load) RSS every 100 ms during requests. Does not include loading peaks or the Python driver. Shared mmap pages are counted in each process; **not private/unique physical memory**, not an allocator high-water bound, and short cells may have few samples. |

The source bounds root entries at `prefix_cache::MAX_ROOTS = 4` **per worker**.
The matrix's 2/6/12 prefixes spans below/near/above the nominal eight roots of
two workers, but actual occupancy can be lower: byte budget, parked sessions,
warm-pool entries, architecture capability and rendezvous skew all matter.
`oracle_prefixes_per_worker` exposes the actual bucket sizes; compare those to
four rather than treating aggregate capacity as perfectly balanced. Stored-root
event counts alone do not demonstrate eviction or four live roots. Varying
concurrency measures pinned-owner delay and, when a wave spans repeated rounds,
duplicate-prefix competition.

Memory is bounded by configuration, **not proven solely by flat RSS**:
`--cache-mib` per worker limits parked snapshots, `n_ctx` and `slots` limit live
sessions, and the ordinary router's queue is capped by `--queue-depth`. Live
sessions, warm-pool state, weights, workspaces, and process overhead are outside
the parked-cache byte budget. The harness limits total requests to 20,000,
concurrency to 256, workers to 16, system filler to 4,096 words, and one trace to
32 MiB. At most one wave is submitted, SSE lines are capped at 1 MiB, each
response at 16 MiB, and RSS summaries are constant-size; timing/output hashes
remain O(total requests). These are safety ceilings, not recommended run sizes.

Worker readiness precedes the first generation. Any lazy first-use weight load,
page faults or kernel initialization is therefore included in first-round and
cold-control client timings, not silently subtracted. The warm/cold difference
is not guaranteed to be *only* prefill; use the accompanying engine prefill
counters/rates to separate that mechanism. No OS page-cache flush is performed.

## CI and validation

```sh
python3 -m unittest discover -s scripts -p test_prefix_locality.py -v
cargo test --test prefix_locality_tests -- --nocapture
cargo test --test route_tests --test integration_tests
cargo test --lib prefix_cache
cargo test --lib prefix_key
```

The Rust test writes a tiny native Qwen3 GGUF and matching tokenizer using
`tests/common`, then drives real worker/router **processes** through the Python
harness: two workers, 2/6/12 prefixes, concurrency 1/2/4, two rounds, four output
tokens, two repeats with reversed policy order, and warm/cold calibration. It
asserts identical outputs, observed reuse,
completion-log coverage and router queue-log coverage. Python tests cover stable
trace/placement, fail-closed output checks, nearest-rank percentiles, semaphore
wait accounting, engine/router log parsing, first-content timing, and truncated
streams, oracle admission timeout, and slot release on exceptions. CI's default
Rust integration suite includes the fixture; the
benchmark-tools job also runs the stdlib unit tests.

## Evidence

### Saved run, audited October 31, 2026

No real-model measurement was rerun to finish this patch. The existing artifacts
on **erskin**, `/tmp/joshua-prefix-201-NXjVJ2AM`, were inspected first. They contain
the release binary, source snapshot, model/tokenizer, reports and per-process
logs. These are host-local scratch artifacts, not a permanent public archive.
The 619 tracked files under the build/source directories plus `Cargo.toml` and
`Cargo.lock` match base `33d9d6e4b8927752cdde23ad08e72e797f375ae3`
byte-for-byte. `release-build.log` records an optimized release build; the exact
compiler version and original build environment/`RUSTFLAGS` were not captured.

Host/backend: Linux x86_64, AMD Ryzen 9 5900HX, 16 logical CPUs, 62.19 GiB RAM;
CPU engine, default features, mmap-backed Qwen3-0.6B Q4_K_M (396,705,472 bytes).
The driver and its children inherited CPU affinity **4–7**; each worker set
`RAYON_NUM_THREADS=2`. Joshua's private matmul pool uses available parallelism,
not that environment variable; this is not evidence of a two-thread matmul pool.
This is four logical CPUs shared by colocated processes,
not two independent machines, not accelerator inference, and not a hard quota
on every library thread. Host load-average vectors were retained; one-minute
load ranged from 0.39 before the first run to 7.59 between runs. These samples
do not prove exclusive access or steady-state performance. At the finishing
inspection no harness job was running on erskin; the separate converter on
kleya and existing services were not signalled or stopped.

All runs used seed 201156, 3 shuffled rounds, 2 alternating-order repeats,
64 filler words after the 32-word prefix identity, `n_ctx=512`, 4 output tokens,
one slot/worker, root minimum 16, queue depth 16 and 120-second timeouts.
Every measured generation reported **151 prompt tokens and 4 completion tokens**;
logs identify a 118-token system root. Prefix counts 2/6/12 rendezvous into
**[0,2] / [2,4] / [5,7]** for this model ID and seed. Thus two heads already
concentrate on one worker, six fit the per-worker count bound only with this
placement, and twelve exceed `MAX_ROOTS=4` on both workers. Count capacity alone
does not establish byte capacity or actual occupancy.

| Artifact directory | Start (UTC, October 31) | Workers | Cache MiB/worker | Prefixes | Concurrency | Paired cells / policy requests |
|---|---|---:|---:|---|---|---:|
| `real-qwen-2w` | 06:58:44 | 2 | 32 | 2,6,12 | 1,2 | 12 / 480 |
| `real-qwen-128mib` | 07:15:15 | 2 | 128 | 2,6,12 | 2 | 6 / 240 |
| `real-qwen-128mib-c4` | 07:21:08 | 2 | 128 | 6 | 4 | 2 / 72 |
| `real-qwen-1w` | 07:22:44 | 1 | 128 | 2 | 1,2 | 4 / 48 |

The first report predates the final `completed` field and root/pool event
breakdown. It has all 12 expected cells, but its missing completion marker is
**not** silently treated as `completed=true`. During this audit every saved cell
was rechecked against its raw logs: 840/840 policy requests succeeded, 840
completion records were present, every load request had a router queue record,
and every pair had byte-identical request hashes and matching semantic output
hashes (content, reasoning, finish and usage). Root/pool counts below were
reparsed from raw logs, not inferred from token overlap. The other three reports
explicitly record `completed=true`. All four record `all_outputs_match=true`.

The first run also contains two warm/cold calibrations (six additional requests,
including conditioning). Warm TTFT was **81.90–82.39 ms**, versus
**1627.85–1632.44 ms** on a fresh second worker. Warm reused 150 tokens and
prefilled 1; cold reused 0 and prefilled 151. Cold logged prefill rates correspond
to **0.009971–0.010028 seconds/token**. This is a direct repeated-prompt control,
not the later-round changing-suffix workload, and not a pure prefill-only penalty:
first-use initialization/page faults, session creation and snapshot work can
also contribute. Warm/cold semantic outputs match in both orderings.

### Locality versus load and capacity

The following are **min–max ranges of the two individual repeats**, not pooled
percentiles, confidence intervals, or means. TTFT is first content/reasoning,
including client semaphore wait for oracle. Throughput is completion tokens/s
over full cell elapsed time, including all three rounds. Reuse is the aggregate
engine token counter, **not a whole-entry hit count**.

Two workers, **128 MiB/worker**:

| Prefixes / concurrency | Load p50 TTFT ms | Oracle p50 TTFT ms | Load p95 TTFT ms | Oracle p95 TTFT ms | Load / oracle tokens/s | Load / oracle reused tokens |
|---|---:|---:|---:|---:|---|---|
| 2 / 2 | 725–2769 | 1060–1081 | 3145–3205 | 3427–3456 | 2.736–3.670 / 3.573–3.606 | 254–476 / 483 |
| 6 / 2 | 935–2132 | 795–800 | 3212–3266 | 3296–3368 | 2.988–3.302 / 3.616–3.630 | 1020–1238 / 1456–1460 |
| 12 / 2 | 2910–2923 | 2180–2181 | 3174–3196 | 3338–3361 | 2.254–2.353 / 2.664–2.770 | 419–748 / 1513–1732 |
| 6 / 4 | 2992–3303 | 1785–2062 | 6917–7022 | 7261–7266 | 2.709–2.724 / 3.621–3.625 | 798–801 / 1460 |

Oracle improves throughput in the 6- and 12-prefix cells, but **does not improve
p95 TTFT in any of these paired cells**. At concurrency 4, load admission p95
was 3964–4039 ms; oracle pinned-owner client wait p95 was 5761–5771 ms. Those
are different queue scopes, but their growth is consistent with the observed
tail-latency disadvantage. This supports measuring a future saved-prefill versus
queue-delay spillover policy, not deploying static pinning as a general win.
At 12 prefixes, oracle restored roots 6–8 times and rewound pooled sessions
26–28 times per repeat; load restored roots 1–2 times and rewound 32–33 times.
Partial pool rewinds can share only a short template/identity prefix and must
not be reported as reusable whole system roots or advertised cache entries.

The **32 MiB/worker** run shows why root count is insufficient:

| Prefixes / concurrency | Load tokens/s | Oracle tokens/s | Load reused tokens | Oracle reused tokens |
|---|---:|---:|---:|---:|
| 2 / 1 | 2.615–2.649 | 2.408–2.417 | 254 | 151 |
| 2 / 2 | 2.699–3.605 | 2.398–2.680 | 254–476 | 151–262 |
| 6 / 1 | 2.152–2.153 | 2.220–2.233 | 135 | 249 |
| 6 / 2 | 2.192–2.205 | 2.255–2.262 | 138–142 | 245–249 |
| 12 / 1 | 2.184–2.199 | 2.194–2.198 | 409 | 400 |
| 12 / 2 | 2.228–2.238 | 2.180–2.181 | 399–409 | 292–296 |

There were **zero root-resume events** in that run despite some successful root
store events. Reuse instead included pool rewinds. No live occupancy/eviction
timeline was recorded, so do not infer a hit probability or a particular
eviction cause from these totals. Increasing the byte budget changes that
regime; comparing the two runs is descriptive, not a controlled causal estimate
with noise bounds.

The one-worker 128 MiB control yielded load/oracle throughput of
3.445–3.460 / 3.467–3.472 tokens/s at concurrency 1, and
3.477–3.483 / 3.486–3.499 at concurrency 2. Both policies reused 483 tokens,
restored one root and performed four pool rewinds per cell. Their p95 TTFT
ranges overlapped (1715–1759 / 1716–1737 ms at concurrency 1;
3581–3594 / 3556–3595 ms at concurrency 2). This bounds the observed hop/harness
difference only for this small control; it is not a portable router-overhead
subtraction for the two-worker results.

### Memory, bytes and remaining gates

Maximum sampled summed RSS across the 128 MiB two-worker cells was
**1,880,678,400 bytes (1.75 GiB)**; the 32 MiB run reached 1,366,110,208 bytes.
These include shared mmap pages counted per process and, for load, the router.
They exclude the driver, loading peaks and unobserved sampling peaks. They do
not demonstrate an RSS plateau over long sessions, exact cache occupancy, or a
bound on unique physical memory. Configured snapshot/context/session/queue
caps are the boundedness mechanism; directory growth is not tested because
there is no directory.

Across 840 policy requests the saved reports count **580,440 request-body bytes**
and **973,092 response-body bytes**, not NIC traffic. Those historical response
counts omit the final SSE separator after `[DONE]`; they are undercounts, not
exact complete-body totals. The corrected harness drains the bounded response
through EOF and counts the separator and any trailing body bytes. The saved
reports have not been rewritten or relabelled as corrected reruns. No directory or KV-transfer
bytes were measured; their fraction cannot be asserted zero or below the RFC's
0.1% target. All logged prefilled tokens in these saved cells had positive
logged prefill rates, but rates vary with concurrency and remaining suffix size;
the cold-control rate above is not a universal cost model.

The evidence is a synthetic, single-model/single-host CPU experiment. It does
not provide a realistic cluster-wide overlap distribution, authenticated
conversation affinity, exact-continuation/edited-context logits parity for a
future index, stale-directory/restart behavior, directory wire cost, multi-host
scaling, accelerator parity, or an isolated misroute penalty proven to be
*exactly* recomputation time. Chunk hash overlap and any nonzero reuse counter
are **not whole-entry cache hits**. The RFC gate for phases 3–5 remains open;
this slice does not close #201, #156, or #155.

### Artifact identity and validation

SHA-256 identities (paths relative to the saved artifact root):

```text
target/release/joshua   6dfa207f4106eb3a24bfc932cdd082325c0b5447b20a51a863aefdd563e2c1d0
model/Qwen3-0.6B-Q4_K_M.gguf
                       ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a
model/tokenizer.json   c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b
measured-script-32mib.py
                       d68d1d1f169a96cc10bcd0540abe57678d763325194024725ca386fb5ac9dbee
scripts/prefix_locality.py
                       1c6feb93ba5599797bf389014f0d4873bd6c2121781df496901227aaf3232e2b
real-qwen-2w/report.json
                       60e272a40d1e7da862caefcbbdb61fe80138ab167652696da53907f9489b63c3
real-qwen-128mib/report.json
                       d2430336e8c790793d17f704bc8b3152371218a34403332369d62663e2758664
real-qwen-128mib-c4/report.json
                       6c2f6403279cc8a2d36a99530a926984fc994b470d27ad0d85094536aff70813
real-qwen-1w/report.json
                       6b3511db58f3e2fa465a7433248567f81e138e11d18b41e92b3a30565bb973bf
```

The final script differs from the saved later script by bounding oracle
admission wait, releasing its slot even on an unexpected request exception,
rejecting duplicate sweep values before starting, and counting response-body
bytes through EOF rather than stopping at the `[DONE]` data line.
Saved runs had no admission timeout; their timing/output evidence is retained
as measured, not relabelled as a rerun of the final script. The earlier script
also predates completion markers and richer root/pool event accounting.

Final local validation on Darwin arm64, default CPU backend, Rust
`1.99.0-nightly (d453bdd8f 2026-08-14)` and Python 3.15.0:

- Python accounting/transport tests: 11 passed after review fixes, including
  exact LF/CRLF response lengths with trailing bytes and duplicate sweep values.
- Expanded tiny Qwen process fixture: 18 paired cells, both policy orders,
  2/6/12 prefixes × concurrency 1/2/4, and two warm/cold calibrations; passed.
- `route_tests`: 29 passed; `integration_tests`: 41 passed (including existing
  architecture-specific continuation/rewind checks).
- Full `cargo test -j 2 --lib`: 472 passed, 10 ignored by the suite, 0 failed.
  Focused `prefix_cache` (6) and `prefix_key` (12) tests also passed.
- `cargo check -j 2 --all-targets` and
  `cargo check -j 2 --features distributed --all-targets`: passed with warnings.

No accelerator code changed or accelerator validation was claimed. CI still
needs to validate its pinned Rust 1.89/Linux environment; the local nightly
results do not stand in for that check.
