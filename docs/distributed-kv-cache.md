# Chunk-addressed cross-node prefix cache

**Status:** design only, nothing implemented. Tracked in
[issue #201](https://github.com/rexlunae/joshua/issues/201). Phase 0
(measurement) gates phases 3–5; phase 1 is useful on its own.

| Phase | What | Status |
|---|---|---|
| 0 | Measurement: prefill s/token, cluster-wide prefix overlap, misroute cost | ⏳ not started |
| 1 | `src/prefix_key.rs`: chunk hash chain, salt, coarsening | ⏳ not started |
| 2 | Worker advertises cache residency in `WorkerInfo` | ⏳ not started |
| 3 | Coordinator advisory index + overlap term in `try_pick` | ⏳ depends on 0–2 |
| 4 | Deltas, tombstones, TTL, anti-entropy sketch | ⏳ depends on 3 |
| 5 | Deterministic affinity fallback (shared with #156) | ⏳ depends on 1 |

Related: #155 (whole-request routing, merged), #156 (conversation
affinity), #86 (distributed cluster RFC), #186 (deepseek4 boundary
checkpoints).

## 1. Where the cache stands

`PrefixCache` (`src/prefix_cache.rs`) parks detached session state in host
RAM under a byte budget, LRU first out. Entries are keyed by the tokens they
cover, and a lookup matches an entry whose tokens are a strict prefix of the
prompt (`PrefixCache::best_len` scans for the longest one). Two kinds of
entry: **root** entries at the end of a prompt's shared head (a system
prompt and tool list), kept until everything else has been evicted, and
**parked conversations**. A finished conversation reaches the cache only
after the warm session pool lets it go (`Engine::release_model`). Until
then it is reused straight from the pool by longest common prefix. Both
tiers are per process, so on a routed cluster they are per worker, and
nothing outside the worker can see them.

`docs/request-routing.md` states the routing limit plainly: routing uses
model id, slots and reported in-flight count, and "does not … prefer a
worker holding a reusable KV prefix". `try_pick` (`src/coordinator.rs`)
walks workers from a rotating start and keeps the lowest load ratio.

#156 adds *conversation* affinity: an opaque handle for a conversation keeps
its follow-ups on one worker. That solves the multi-turn case and nothing
else. A new conversation that opens with the same 4k-token system prompt,
tool schema or attached document has no handle, so it is routed by load and
prefilled from zero. That is the gap this design closes: **identity for
prefixes that no client named, and a directory cheap enough to consult on
every request.**

## 2. The asymmetry that shapes the design

Per-token KV is `2 × n_kv_heads × head_dim × dtype_bytes × layers`. For
Qwen3-4B (36 layers, 8 KV heads, head_dim 128) at f32 that is
`2 × 8 × 128 × 4 × 36` = **288 KiB per token**:

| Quantity | Bytes |
|---|---|
| One token of KV, Qwen3-4B f32 | 288 KiB |
| One 16-token hash chunk of KV | 4.5 MiB |
| One 32k-token prefix of KV | 9 GiB (2.3 GiB int8) |
| One directory entry (one 64-token chunk) | ~48 B |
| 10k entries, 8 workers, replicated on the router | 3.8 MB |

(KV-per-token is model-specific — MLA stacks such as deepseek2/deepseek4
compress the latent far below this — but the ratio between an entry and the
state it names stays around 10⁵.) Two rules follow:

1. **The hit check adds no packet and no lock to the request path.** A
   fan-out probe to `n` workers costs a round trip per request and stalls
   TTFT on the slowest one; that is the design to avoid.
2. **KV transfer is the exception; routing to the KV is the mechanism.**
   Transfer beats recompute only when
   `NIC bytes/s > KV bytes/token ÷ prefill seconds/token`. At 288 KiB/token:
   2 ms/token (144 MB/s) is within 10GbE; 0.1 ms/token (2.9 GB/s) needs a
   fast fabric. Measure, do not assume.

## 3. Chunk identity

A chained hash over prompt tokens. Any component can name a prefix without
talking to its owner, and two components agree without negotiating:

```rust
// h_0 is a fixed constant so independent processes agree (as in vLLM APC).
let mut h = H(salt, NONE_HASH);
// Only complete chunks are keyed: a short trailing chunk would let one key
// stand for fewer than HASH_CHUNK tokens.
for chunk in tokens.chunks_exact(HASH_CHUNK) {
    h = H(&h, chunk);          // 32-byte digest
    keys.push(h);
}
```

* `HASH_CHUNK = 16` tokens, `STORAGE_CHUNK = 64` tokens (four hashes per
  stored chunk). vLLM splits the same way (`hash_block_size` defaults to
  16); DeepSeek and SGLang HiCache use 64-token storage units. Finer
  hashing lets a node advertise a coarser view later without recomputing
  keys.
* `salt = H(model_fingerprint, tokenizer_digest, template_id, tenant_id)`.
  Not optional: two workers with different chat templates or tokenizer
  revisions must never agree on a key. `model_fingerprint` is the GGUF
  tensor/metadata identity, not the file bytes, so a re-download does not
  invalidate the cache.
* A shared *key prefix* of length `c` means the first `16c` tokens are
  identical. The tokens after the last complete chunk are never keyed, so a
  chain says nothing about them.
* The local check is stricter than a shared prefix. `PrefixCache::best_len`
  takes an entry only when it is a **strict** prefix of the prompt
  (`entry.len() < prompt.len()`), so at least one prompt token is left to
  produce logits. An entry the length of the whole prompt never matches. The
  phase-1 property test therefore states the equivalence the cache uses:
  for an entry `E` ending on a chunk boundary and a prompt `P`,
  `keys(E)` is a prefix of `keys(P)` and `E.len() < P.len()` ⟺
  `P.starts_with(E)` and `E.len() < P.len()`. The usable overlap for a
  prompt of `n` tokens is capped at `⌊(n − 1) / 16⌋` chunks.
* An advertised key promises a servable entry that ends exactly at that
  chunk boundary. `chunk`-granularity workers can cut one there (§4).
  A `snapshot`-granularity entry ends wherever its conversation ended, so
  it is reusable only by prompts that contain its whole token sequence. Its
  last complete chunk key alone would over-promise. See open question 2.
* Keys are over **post-template** prompt tokens (what `Engine::tokenize`
  yields), not raw request text.

Why a chain rather than a hash of the whole prompt: prefixes grow one chunk
at a time during a conversation, truncation needs the key at a cut point,
and a per-chunk key makes the directory a trie by construction.

## 4. What a worker advertises

`WorkerInfo` (`src/coordinator.rs`) gains one optional, serde-defaulted
field. The protocol version stays 1: `WORKER_PROTOCOL_VERSION` makes a
router refuse workers that report a different version, so *additive* fields
with `#[serde(default, skip_serializing_if = "Option::is_none")]` are the
only backward-compatible change. An older router ignores the field; an older
worker simply omits it and is routed as today.

```json
"cache": {
  "epoch": 7,
  "chunks": 1240,
  "tokens": 79360,
  "root": "9f1c2ab3de440157",
  "watermark": 918233,
  "granularity": "chunk"
}
```

* `epoch` — bumped on restart, model reload, or cache clear. Fences every
  directory entry for that worker: a restarted worker can never inherit its
  own stale entries.
* `chunks` / `tokens` — what the worker can serve **now**, after its own
  LRU has run, not a high-water mark. This is what makes a replica
  self-correcting. The servable set has two tiers, and both count:
  * the **warm session pool**. `Engine::release_model` returns a finished
    session there. A new prompt reuses it by longest common prefix, and
    rewinds through `rewind_target` where the architecture allows it.
  * the **`PrefixCache`**. A pooled session is parked here only when the
    pool evicts it, and only if it is at least
    `min(PREFIX_CACHE_PARK_MIN, prefix_cache_root_min)` tokens (256 by
    default; lower when the operator lowers the root minimum, see
    `Engine::park_session`). Shorter sessions are dropped. A parked entry
    shorter than one 64-token storage chunk is held but never advertised,
    because it has no complete storage chunk to key (open question 2).

  A key is held when either tier can serve it, and the two tiers serve
  different keys. A warm attention-only session can rewind to any chunk
  boundary of its chain, so it serves every key along it. A parked snapshot
  covers its whole conversation, and `best_len` takes it only as a strict
  prefix of a new prompt, so it serves no intermediate key.

  The advertised set is therefore defined by the worker's **state**, not by
  per-session events: it is the union, over every holder (each pooled
  session and each cache entry), of the keys that holder can serve. Many
  holders can serve the same key, because two conversations that open with
  the same system prompt share its chain. So the worker keeps a holder count
  per key. Any change to a holder recomputes that holder's servable keys and
  adjusts the counts: a new session, a rewind, a pool-to-cache move, a drop,
  or a cache eviction. A key whose count goes from 0 to 1 emits an insert. A
  key whose count drops to 0 emits a tombstone. Every other change is silent.
  Parking one of two sessions that share a head therefore tombstones nothing
  the other session still serves.

  A `chunk`-granularity worker can shrink what parking loses by parking at
  the last storage boundary instead of the whole conversation
  (`snapshot(64·⌊len/64⌋, len)`), so the parked entry still ends on a key.
* `root` — 8-byte XOR (or Merkle) sketch over the sorted chunk keys, for
  anti-entropy (§6).
* `watermark` — the last delta sequence number the worker has emitted.
* `granularity` — `chunk` when the worker can park a snapshot at any chunk
  boundary, `snapshot` when it can only park whole sessions.

Granularity is a property of the architecture, not a policy flag.
`Session::snapshot(keep, len)` (`src/native_session.rs:912`) only produces a
snapshot at an arbitrary `keep` when `weights.can_truncate()` holds or a
prefix checkpoint exists there (`take_prefix_checkpoint`, `:882`).
Attention-only stacks qualify at any boundary. Recurrent/SSA stacks (lfm2,
Qwen delta-net, deepseek4's SWA ring and streaming compressors, checkpointed
for #186) need a checkpoint per boundary and stay `snapshot` until their
loaders provide one. The directory format is identical either way; only the
promise differs.

## 5. The coordinator's index

The router already sees every request, so the index is fed by the data
plane rather than by polling:

* **Request**: the router computes `h_1..h_k` (§8) and forwards only the
  count and the tip: `x-joshua-prefix: k=32 tip=<base64 h_k>`. The chain
  cannot be run backwards, so the earlier keys are not recoverable from the
  tip. The worker does not need them: it tokenizes the prompt itself (§8)
  and derives its own `h_1..h_k` under its own salt. The tip is a drift
  check. If the worker's `h_k` differs, the two sides tokenized or templated
  differently. The worker then reports `had=0` and counts a drift, rather
  than reporting a count against a chain the router does not share.
* **Response header**: `x-joshua-cache: had=17`, the number of chunks of
  this chain the worker reused. That count is known at admission, when the
  worker's `best_len` lookup runs, so it is available before the first byte
  even on a streamed response. **Zero extra round trips.**
* **After the response**: what the worker holds *after* the request is not
  known when headers go out. A stream's headers precede decode, and parking
  happens when the session ends. So it does not ride the header. When a
  request completes successfully, the router records a **provisional**
  positive for the prompt's usable chunks on that worker, because a finished
  session normally stays in its warm pool (§4). Provisional entries are kept
  apart from the replica (§6). A worker can never tombstone something it
  never held, such as a session the pool dropped under memory pressure. So
  a provisional entry is confirmed by an insert delta (or a full snapshot)
  that names it, or it expires on its own after two probe intervals. Until
  then it is a possible stale positive: it costs one recompute, never a
  wrong output (see below).
* The router maps the counts onto its own key list. `had=17` means
  `h_1..h_17` are held by that worker. The router folds this into
  `HashMap<Key, Entry { worker, last_seen }>` under one mutex or a sharded
  set of them, read on the pick path. No RPC, no `.await`, no Redis.

Sizing, at 10k chunks per worker: 480 KB of index per worker, 3.8 MB for
eight — less than one 16-token block of Qwen3-4B's f32 KV. A 2k-token prompt
is 32 entries (~1.5 KB) naming 576 MB of KV.

The index is **advisory**. A stale positive sends a request to a worker that
no longer has the state; the worker's own `best_len` misses and it
prefills — slower, never wrong. That asymmetry is what lets the directory
skip consistency machinery: bias every ambiguous case toward keeping a
positive, because a stale negative costs a guaranteed recompute.

## 6. Anti-entropy, not directory polling

The probe loop already exists (`--health-interval-ms`, default 2000 ms).
Extend the probe body instead of adding a channel:

| Event | Cost |
|---|---|
| Probe, replica agrees (`epoch`, `chunks`, `root` match) | ~64 B per probe |
| Probe, drift (`root` or `chunks` differ) | + `GET /v1/worker/cache?since=watermark` |
| Epoch change (restart, reload) | full snapshot, then normal deltas |
| Eviction or park | tombstones only for keys whose holder count reaches 0 (§4) |

Eight workers at 2 s is 4 probes/s, so steady-state directory traffic is a
few hundred bytes per second — independent of cache size, because the sketch
compresses it. This is Merkle/XOR anti-entropy as used in Cassandra and
Dynamo: cheap comparison in the common case, real data only on divergence.

The replica holds only what the worker reported, through a snapshot or a
delta. The router's provisional entries (§5) live in a separate map with a
TTL of two probe intervals. They are consulted on the pick path, but they are
never folded into the replica or the sketch comparison, so they cannot cause
a resync. An insert delta for the same key promotes one into the replica.
Expiry removes one without any message from the worker. Reconciliation
therefore never needs a tombstone for state the worker never held.

Workers keep a bounded ring of recent deltas (say, 10 minutes or 100k
entries); a router that falls further behind than that takes a full
snapshot. Deltas carry the `epoch` and a monotonic sequence number so a
dropped response is retried, not applied out of order. Absolute per-key
updates rather than diffs keep re-application idempotent.

## 7. Routing on the overlap

Longest matching chain wins, but only when it pays:

```text
k          = ⌊(prompt_tokens − 1) / HASH_CHUNK⌋     // strict-prefix cap, §3
gain(node) = min(overlap_chunks(node), k) × HASH_CHUNK × prefill_s_per_token
           − queue_wait_delta(node)
           − misroute_penalty
```

`overlap_chunks` counts hash chunks. A worker advertises at storage-chunk
granularity, so the overlap moves in steps of four.

`try_pick` keeps its existing eligibility filter (model, capability,
health, slots) and adds this term to its load comparison: a worker with a
long overlap may accept a longer queue and still win. Two guards:

* **Hysteresis.** A conversation already on a worker stays there unless the
  gain elsewhere exceeds a margin, so marginal overlaps do not bounce
  requests between nodes and destroy locality on both.
* **Singleflight.** Concurrent requests sharing a long prefix are pinned to
  one worker even when it is busier, because `n` of them computing the same
  missing prefix is strictly worse than one computing it and the rest
  continuing. (This is the stampede a fan-out hit check would create and a
  directory avoids.)

Fallback ladder when there is no exact information:

1. exact chunk index (§5);
2. deterministic affinity, `hash(affinity_key) % eligible_workers`. This
   is the CARP pattern: no state and no traffic, shared with #156's handle.
   Use rendezvous hashing rather than a plain modulus, so a worker joining
   or leaving moves only its own share. The key depends on what the router
   has:
   * **with a tokenizer**: `H(salt, h_c)`, where `h_c` is the chain key at
     the last complete storage chunk of the shared head (system prompt and
     tools). It is the same identity as step 1, just coarser.
   * **without a tokenizer**: there are no post-template tokens to hash,
     so the key is taken from the request body instead:
     `H(model, tenant_id, canon(tools), canon(leading system/developer
     messages))` for chat requests, and `H(model, tenant_id, first 4 KiB
     of prompt)` for completions. `canon` is a canonical JSON
     serialisation (sorted keys, no insignificant whitespace). It is
     cruder than tokens: a whitespace edit inside a system prompt changes
     the key, and two templates that render the same text get different
     keys. That only costs locality. The worker's own `best_len` still
     authorizes every reuse.
   * an explicit #156 conversation handle, when the client sends one, takes
     precedence over both.
3. least-loaded (`try_pick` today).


## 8. Keys on the router mean a tokenizer on the router

The router forwards bodies byte for byte and reads only `model`
(`docs/request-routing.md`). Computing keys needs the prompt tokens, so it
needs a tokenizer — but only a tokenizer: `tokenizer.json` (a few MB, pure
`tokenizers` crate, already a dependency) plus the chat template
(`src/template.rs`). No weights, no mmap, no backend.

* `joshua route --tokenizer-dir DIR` (or the same model-path resolution the
  workers use) loads it once at startup.
* The **worker stays authoritative**: it tokenizes the prompt itself and
  authorizes reuse only through `PrefixCache::best_len`. A router whose
  tokenization or template differs therefore produces a miss, never a false
  hit. The salt in §3 is the second guard, catching the case where both
  sides mis-tokenize identically.
* Without a configured tokenizer, the router runs the affinity ladder
  (§7, steps 2–3) on the body-derived key defined in §7. That needs no
  tokenization at all. This is also why phase 5 is worth landing on its own:
  it requires no chunk keys, no index, and no new traffic.

## 9. What this deliberately does not do

**It does not move KV.** `Session::restore` refuses a snapshot unless the
weights are the same `Arc` (`src/native_session.rs:975`), and
`SessionSnapshot` has no wire form. The state is not just attention KV:
per-layer state includes the DSA indexer for deepseek2/deepseek4, recurrent
`delta` state for the hybrid stacks, and MTP draft state where configured.
Cross-node restore is a separate project needing (a) a weights fingerprint
instead of `Arc::ptr_eq`, (b) `to_bytes`/`from_bytes` for every layer's
state — `compact_copy` is the natural starting point — and (c) a transfer
path (HTTP streaming first, RDMA only if measured). It is worth doing only
where `KV bytes/token ÷ prefill s/token` is below the fabric's bandwidth,
which is exactly what phase 0 measures. Landing the keys and the directory
first costs nothing and is a prerequisite either way, because a transfer
scheme needs the same chunk identity and the same ownership answers.

**It does not add a shared store.** A Redis/3FS/NIXL directory puts a
network hop on the request path — the one thing this design exists to avoid.
If one is added later it must use client-side caching (Valkey RESP tracking)
and stay off the pick path.

**It does not share across tenants.** A prefix cache is a prompt-prefix
oracle: a hit tells one client that another used the same text. DeepSeek
isolates user caches for this reason. `tenant_id` in the salt is the
boundary, and it must come from authentication, not the request body.


## 10. Metrics

Counters on the router (surfaced in `/v1/workers` and the log line per
attempt, which already carries queue time and outcome):

* `prefix_overlap_tokens` — longest chain matched per request;
* `prefix_saved_prefill_tokens` — overlap actually realized on the serving
  worker (`had=` in the response);
* `prefix_index_hits`, `prefix_index_misses`, `prefix_index_stale` — where
  the winner's `had=` was 0 despite a positive directory entry;
* `prefix_drift` — requests whose `tip` the worker did not reproduce
  (router and worker tokenization or template disagree);
* `prefix_affinity_fallbacks` — picks made by ladder step 2 or 3;
* `prefix_delta_bytes`, `prefix_probe_bytes` — directory traffic;
* per-worker `cache.chunks`, `cache.bytes`, `cache.epoch`.

On the worker: reuse already reports as `PrefixCacheStats { hits,
tokens_reused, stored, evicted, ... }`; add `evictions_since_probe` so the
delta stream and the sketch can be cross-checked.

## 11. Phase 0: what to measure before building phases 3–5

1. **Prefill seconds per token**, per model and backend, on the target host.
   Needed twice: to convert an overlap into a time saving, and to evaluate
   the transfer break-even in §2.
2. **Cluster-wide prefix overlap distribution** in realistic traffic
   (multi-turn sessions, shared system prompts and tool schemas, repeated
   documents): what fraction of prompt tokens a cluster-wide cache could
   have served, and how often the sharing is *across* conversations rather
   than within one. If little sharing crosses conversations, #156's
   affinity already captures most of it and this issue stalls — which is a
   fine outcome.
3. **Misroute cost.** Route a request to a worker that lacks the prefix and
   confirm the penalty is exactly the prefill it must redo (no error, no
   partial state).
4. **Directory bytes on wire** for a fixed run, as a fraction of KV bytes
   and of response bytes. Expected to be < 0.1%.

Items 1 and 2 can be answered on one host with several `joshua serve`
processes behind `joshua route`, with the local cache sized large enough
that eviction is not the confounder; 3 and 4 need the real deployment.


## 12. Open questions

1. **Ownership of the affinity handle.** #156 defines a conversation handle
   scoped to authenticated ownership. Phase 5 here wants a *content*-derived
   affinity. They should share one implementation, so one of the two issues
   should own it — probably #156, with this issue consuming it.
2. **Small prompts.** A session shorter than one 64-token storage chunk
   has no key to advertise. With the defaults it is not parked either: the
   park threshold is 256 tokens. An operator who lowers
   `prefix_cache_root_min` below 64 makes such sessions parkable, though
   still not advertisable, so they serve local follow-ups only. Likewise a
   pooled session's trailing tokens past its last complete chunk are never
   keyed. Options: key a trailing partial chunk with its true length, so
   short entries become routable, or let anything under 64 tokens fall
   through to affinity (§7). Leaning to the latter until measured.
3. **Reporting what was produced.** §5 takes `produced` off the request path:
   `had=` rides a header, and occupancy after the request comes from an
   optimistic insert corrected by deltas. If the optimistic entries prove
   noisy, the alternatives are an HTTP trailer, which streaming clients and
   proxies often drop, or a final SSE comment line (`: joshua-cache
   produced=32`). Clients ignore SSE comments, but the router would have to
   parse the stream it now forwards byte for byte. Decide after phase 3
   measures how often an optimistic positive turns out stale.
4. **Detecting template drift.** A digest over `tokenizer.json` plus the
   resolved template string should be enough, but a worker could also report
   the hash of its first rendered chunk so a mismatch shows up as a
   first-chunk divergence rather than a mysterious miss.
5. **Should `granularity: snapshot` workers advertise at all?** A
   whole-session entry can only serve the exact same prompt, which is what
   #156's handle already covers. Advertising it may add noise for little
   gain; measure before implementing.

## References

* vLLM automatic prefix caching — chained SHA-256 block hashes,
  `cache_salt`, `hash_block_size` vs storage block size.
* SGLang RadixAttention and HiCache — `HiRadixTree` as a cross-tier page
  table, 64-token pages, `get`/`exist`/`set` backends.
* NVIDIA Dynamo KV-aware routing — worker radix events feeding a router
  index; multi-DC relay with exact local ownership and a lossy Cuckoo-filter
  projection across sites, plus sequenced idempotent bucket images.
* Mooncake, MemServe/MemCache, LMCache — pooled KV and transfer engines,
  the design family this document defers (§9).
* DeepSeek context caching — 64-token units, per-user isolation.
* Squid ICP → Summary Cache (Fan et al., SIGCOMM '00) → CARP — the same
  ladder from fan-out probes to gossiped digests to computable placement.

