# Multilevel 4-tier weight cache for DeepSeek-V4.1-Flash (REAP-272E)

**Status:** design proposal for an idealized reference build to guide a hardware
buy. Extends the two-tier model in [`docs/device-expert-cache.md`](device-expert-cache.md)
(VRAM hot-expert cache + host-RAM backing) into a **four-tier** hierarchy whose bottom
tiers are two SSDs. Everything here is a target *capacity/bandwidth* budget for a
hardware-purchase decision; it deliberately does not assume a specific machine's
inventory, so each tier is stated as a **minimum spec to buy** rather than a count of
existing parts.

---

## 0. Why four tiers, and why this model in particular

Sparse MoE inference on a **bandwidth-bound** workload has a clean budget structure:
the only thing that scales throughput is the bandwidth of the tier the hot weights
actually sit in, plus the efficiency of the managed cache that keeps the working set
there. DeepSeek-V4.1-Flash is an unusually good subject for a multi-tier cache because
its architecture already *separates* its state into classes with very different
locality, so the cache tiers map onto natural model boundaries rather than arbitrary
byte ranges:

| Weight / state class | Size (REAP-272E, native 4-bit) | Touched per token | Bandwidth sensitivity |
|---|---|---|---|
| Dense "spine" (embeddings, norms, attention projections, routers, shared expert, output, MTP) | ~8–14 GiB | **every** token | **HD** — decode is capped by this |
| Routed-expert pool, 272 experts × 40 layers (~87% of FFN) | ~145–180 GiB | only **6 of 272** per layer (~2.2%) | **HD** for the hot subset, low for the cold tail |
| Engram tables (hashed n-gram gather, ~12.4 KB/token read) | ~189 GiB (or ~98 GiB NVFP4) | sparse, direct lookup | **low** — tiny random reads, no matmul |
| Global KV cache (CSA2 + FP4, always HBM) | 890 B/token | every decoded token | must be on-GPU |
| Persistent KV cache (SWA Bounded Replay, always SSD/host-RAM) | small; explicitly disk-resident | as replayed | designed for SSD |

The full REAP-272E model is **396.8 GiB** (207.7 GiB backbone + 189.1 GiB Engram).
No single-node configuration puts all of that in VRAM + RAM. The design must therefore
make the **two SSD tiers carry the coldest classes** (Engram tables + the cold expert
tail), keep the **hot experts + dense spine** at the fastest available bandwidth, and
price the whole thing so the machine is bottleneck-limited by GPU compute during
decode, not by a per-token disk read.

---

## 1. The four tiers and their roles

Ordered from hottest (fastest, smallest, most expensive per byte) to coldest.

### Tier 1 — VRAM / accelerator HBM
**Role:** the dense spine (touched every token, latency + bandwidth critical) **plus**
the resident hot-expert set. Everything in Tier 1 is read on (almost) every decode step,
so its effective bandwidth must not be the bottleneck. On modern consumer/workstation
GPU memory (RTX 5060 Ti-class → 48GiB+ budget for buy recommendations), this holds:

```
Must-reside (never evicted):
  dense spine ................................ ~12 GiB  (see §3)
  global KV cache (890 B/token × min context) .. budgeted separately
Often-reside (cache-managed):
  hot-expert slots (the "front" of the priority queue)
```
The rest of VRAM above the must-reside floor is the **expert slot pool**. Tier-1 slot
policy = the existing `HotExpertCache` (routing-frequency LRU, `REFRESH_STEPS = 64`)
already merged in Joshua; it names the ordered hot set, and the residency executor moves
those names into slots.

### Tier 2 — system RAM
**Role:** the *resident cache working set* for experts that cannot fit in VRAM — the
medium-hot expert population and, critically, the memory that absorbs a Tier-1 miss
without touching a disk. Tier 2 holds:

```
Must-reside:  the mmap'd page-cache cold path + persistent-KV replay staging  (small)
Cache-managed: medium-hot experts (LRU beyond Tier 1's protected set)
               + the "write side" of the two-level eviction (Tier1 → Tier2 → disk)
```
Tier 2 is the *source of truth* for the expert pool that fits. A VE2 (Tier-1→Tier-2)
miss goes to disk only when the expert is genuinely cold enough to have been evicted
to Tier 3/4.

### Tier 3 — fast SSD (NVMe, gen4/5)
**Role:** persistent backing store for the expert pool that does **not** fit in RAM,
plus the Engram tables when hot. Asynchronous, high-bandwidth, moderate-queue-depth
reads. This is the tier that turns a machine with modest RAM into a viable 397 GiB
model host, because ~90% of the optional weight reads land here with tens of GB/s of
sequential bandwidth.

### Tier 4 — slow SSD (SATA / bulk ZFS)
**Role:** the cold, rarely-replayed reservoir. Holds the **Engram tables** (the largest
single component, but accessed as sparse ~12.4 KB gathers that are latency-tolerant),
the coldest expert tail, and any checkpoint/archive copies. Cheapest bytes-per-GB; the
tier that lets the *whole* model be present without spending NVMe/RAM dollars on the
long tail. For the idealized buy, Tier 4 is a mirrored/bulk-redundant set of slow SSDs
(Tier 3 is the fast single high-endurance pair).

**Tier hierarchy invariants:**
1. *Inclusive by preference, exclusive in practice* — hot data is preferenced upward; a
   resident copy + a backing copy is allowed, but the backing copy is what is evicted.
2. *Write-through on promote within RAM* — an expert promoted to VRAM keeps its RAM
   page so a demotion is instant; demotions stopped being "promotions in reverse."
3. *Read-mostly* — weights are immutable. The cache is a read cache; there is no dirty
   writeback, which removes most of the coherency burden of a normal 4-tier store.

---

## 2. What gets cached, ranked hot → cold (the priority ordering)

This is the core of the request: order model components by how much it matters that
they live in fast storage. The ranking is driven by **(touch-count) × (bytes read per
touch) / (tier bandwidth gap)**, i.e. the contribution to end-to-end decode time if the
component were one tier slower.

### Rank 1 — Dense spine (must always be Tier 1)
- Input/output embeddings, RMSNorm weights, attention `q/k/v/o` projections, the
  router/gate weights, the shared expert, the output layer, MTP layers.
- *Why highest:* read on **every** token. 16B-active decode means ~16B params per token;
  a large fraction of that is this fixed spine. If any piece of it sits one tier down,
  it is re-read on **every** token from a slower tier — a direct throughput tax, not a
  cache miss.
- *Reservation:* pin permanently in VRAM; never a slot.

### Rank 2 — Hot routed experts (cache-managed Tier 1)
- The `HotExpertCache`-selected experts per layer. The router is extremely skewed
  (top-6 of 272 → in practice a few dozen experts across layers dominate decode tokens).
- *Why high:* on a bandwidth-bound MoE, the expert GEMMs are most of the per-token bytes
  (6 active experts × ~19 MB each ≈ ~110 MB/layer; the full ~180 GiB pool dwarfs RAM, so
  which experts are resident decides whether decode reads VRAM/RAM or a disk). Putting
  the *frequently used* subset in VRAM removes the largest per-token PCIe/NVMe transfer.
  This is the existing `HotExpertCache` target — the design just gives it a bigger,
  hotter budget.

### Rank 3 — Medium-hot routed experts (cache-managed Tier 2, LRU)
- Body of the expert pool that fits in RAM but not VRAM. Serves as the hit tier for a
  Tier-1 miss without touching disk; also the resident copy that makes a Tier-3→1 load
  fast (copy, not fault).
- *Why above cold experts:* still frequently enough routed that a RAM hit meaningfully
  raises decode throughput versus a disk fault, and RAM bandwidth (hundreds of GB/s)
  is ~an order of magnitude higher than even gen4 NVMe.

### Rank 4 — Engram tables (Tier 3/4, and NVFP4 optioned down to fit)
- 189.1 GiB of hashed n-gram gather tables (48 rows/token, ~12.4 KB). *Read is tiny and
  sparse; latency-tolerant.* The model's own docs say it "never needs to be resident,
  which is why it can be served from disk or host RAM."
- *Why lower than experts:* even though it is the biggest single component, its
  bandwidth demand per token is microscopic (~12.4 KB reads) compared to an expert GEMM
  (~1 GiB). It is the archetypal *capacity-tier* resident. If RAM is ample it can ride
  in Tier 2 to cut random-read latency; otherwise it is the natural bulk occupant of
  Tier 4. The NVFP4 subset (97.6 GiB, cosine 0.9934) is the size-reduction lever to keep
  it in RAM/NVMe when capacity is tight.

### Rank 5 — Cold routed experts (Tier 3/4)
- Expert tail routed so rarely that a resident slot is wasted. Lives on SSD; promoted to
  VRAM only on a routing spike, then demoted on the next refresh if the spike was a
  one-off.

### Rank 6 — Persistent KV cache (Tier 3/4)
- SWA Bounded Replay KV that is *designed* for SSD/host-RAM. It follows the access,
  not a weight cache policy; treated as a separate, self-managing stream (see §5).

**Order-of-magnitude takeaway:** the dense spine ~must~ be VRAM, hot experts ~must~ be
VRAM-or-RAM, medium experts ~should~ be RAM, Engram and cold experts ~may~ be SSD. Every
position of the boundary up-moves cost a direct throughput hit on a bandwidth-bound
decode; every position down-moves wastes money on storage you do not need.

---

## 3. Reservation math (how much to hold at each tier)

These are idealized budgets for a hardware buy targeting *fair* (not best-case) decode
performance. They assume the native-4-bit REAP-272E model and a consumer/workstation
single-node buy. All numbers are design targets, stated with the floor you should not
buy below.

**Model working set (396.8 GiB total):**
- Dense spine: fixed ≈ **12 GiB** (computed from 40-layer attention/norm/router/shared-expert
  + embeddings + output; FP8/MXFP4 weighted — see §B for the basis).
- Routed experts (272/layer × 40): ≈ **180 GiB**.
- Engram tables: **189.1 GiB** (or 97.6 GiB NVFP4).

> The 397 GiB figure is the base REAP-272E. The active set per token is only ~16B
> params (~17 GB read/token); the cache exists to make those 17 GB come from VRAM/RAM
> rather than disk on the common (hot) path.

### Tier 1 — VRAM reservations
```
  allocate_total      = VRAM_total - safety_headroom (≈ 8–10% for driver/other/scratch)
  dense_spine         = ~12 GiB                      (pinned, rank 1)
  global_KV_base      = 890 B/token × min_ctx  (e.g. 0.9 GiB per 1M tokens housekeeping)
  expert_slot_pool    = allocate_total - dense_spine - global_KV_base
  hot_expert_budget   = min(expert_slot_pool, max_tenable_slots × per_expert_bytes)
```
**Minimum spec to buy (Tier 1):** a GPU with **≥ 24 GiB** of VRAM and ≥ ~560 GB/s of
bandwidth *just to hold the dense spine + a usable hot pool + a minimal live context*.
**Recommended:** **48–64 GiB** if agentic long-context decode throughput matters, so the
hot-expert budget is large enough that the ~17 GB/token decode working set has a high
VRAM hit rate. Buy bandwidth, then capacity: a 16 GiB card (e.g. RTX 5060 Ti) is *not*
enough as the primary inference tier for this 397 GiB model — you would be reselling
the whole decode from RAM/disk.

### Tier 2 — System RAM reservations
```
  RAM_min             = medium_hot_experts_working_set
                     + engram_resident_working_set   (optional)
                     + persistent_KV_replay_budget
                     + mmap_page_cache_baseline      (16–32 GiB for the OS + loader)
  RAM_recommended     = enough so that the *entire* routed-expert pool (≈180 GiB)
                        can be resident if it fits budget, OR at least the
                        medium-hot half + Engram
```
**Minimum spec to buy (Tier 2):** **≥ 128 GiB** of DDR5 (dual channel preferred;
bandwidth ≥ ~300 GB/s). This gives you a large medium-hot expert tier in RAM (roughly
half the pool page-resident) plus the Engram tables or KV-replay headroom — a *fair*
floor, with the remainder of the experts spilling to NVMe.
**Recommended:** **192–256 GiB** — the point where the *entire routed-expert pool*
(~180 GiB) is RAM-resident, which is the highest-value outcome: steady-state decode
never touches a disk for weight reads, only the Engram tables (189 GiB) push into
NVMe/SATA. This is the single highest-leverage RAM buy because it converts most disk
work into memory traffic.

### Tier 3 — Fast SSD reservations
```
  NVMe_reserved       = (model_not_in_RAM) + persistent_KV + checkpoint_margin
                      ≈ enough to back the expert + Engram overflow
  floor_to_buy        = 2 TB (gen4/5, high-endurance TLC, ≥ 7000 MB/s read)
```
**Minimum spec to buy (Tier 3):** one **2 TB PCIe-gen5 NVMe (~7+ GB/s)**; buy in a 1+1
or 2+1 mirrored set only if you also use Tier 3 for checked archiving. If RAM is 128 GiB
(the floor), Tier 3 holds most of the expert overflow that does not fit in RAM, and
decode that misses RAM reads from gen5 NVMe at multi-GB/s — fair for a cache, not great
for a hot path (see §4 bandwidth math). Engram is deliberately *not* the primary Tier-3
resident; it is latency-tolerant (sparse 12.4 KB gathers) so it is the natural Tier-4
resident, keeping NVMe bandwidth for expert GEMM misses. On generous budgets you may
promote Engram to Tier 3 or RAM instead.

### Tier 4 — Slow SSD reservations
```
  Tier4_must_hold     = Engram_full (189 GiB) or Engram_NVFP4 (98 GiB)
                      + cold_expert_tail (a few tens of GiB)
                      + swap/scratch + headroom
  floor_to_buy        = 4 TB SATA/SAS or bulk ZFS (mirrored/RAID for safety)
```
**Minimum spec to buy (Tier 4):** **4 TB** of lower-cost SSD (SATA-class) with modest
bandwidth (~500 MB/s–1.5 GB/s), ideally 2 drives in a mirror or as a warm ZFS pool. This
is the "everything present, nothing missing" tier — it guarantees the full 397 GiB model
is on-box, even if the hot path rarely reads it.

**Combined minimum reference buy** (one node, fair decode for a 397 GiB sparse MoE):
- GPU: **24 GiB** VRAM, high bandwidth (see §4).
- RAM: **128 GiB** DDR5.
- Fast SSD: **1× 2 TB gen4/5 NVMe**.
- Slow SSD: **2× 4 TB SATA-class (mirror)** as a ZFS/bulk tier.

---

## 4. Bandwidth budget — why Tier 1/2 sizing dominates decode

Decode reads ~16B active params/token ≈ **~17 GB/token** (MXFP4/FP8 ~1.06 B/weight).
Compare where that read would land on a miss:

| Tier | Typical sequential bandwidth | Time to read ~17 GB |
|---|---:|---:|
| VRAM (Tier 1) | ~600–900 GB/s | ~20–30 µs |
| DDR5 RAM (Tier 2) | ~300–500 GB/s | ~40–60 µs |
| Gen5 NVMe (Tier 3) | ~7–13 GB/s | ~1.3–2.4 ms |
| SATA SSD (Tier 4) | ~0.5–1.5 GB/s | >11 ms |

At 16B-active decode, a single **Tier-4 miss** costs ~10+ ms → ~≤100 tok/s for that
token. A steady Tier-3-fed decode of the whole per-token working set caps at
~0.5–1 tok/s. **Therefore:** the *persistent* per-token working set (dense spine + the
hot experts the router actually picks) must be resident in Tier 1+2. The SSDs exist to
cover (a) Engram's tiny sparse reads and (b) the cold-expert tail that a long dialogue
occasionally touches — not the steady hot path. This is the single most important
conclusion for the hardware buy: **RAM is the tier that buys decode speed; NVMe buys
capacity/presence; SATA buys the tail.**

For the cache to be "fair," the **effective hit rate** of Tier 1+2 on the ~17 GB/token
working set should be ≥ ~95 %. With a good `HotExpertCache`, the hot-experts working set
is a few hundred GiB of round-robin favorites; 128 GiB RAM holds it; 24 GiB VRAM holds
the spine + a meaningful fraction of the hot set for the fastest tier.

---

## 5. The four-tier promotion/eviction flow (and where Joshua's pieces fit)

Leverages existing merged parts — `HotExpertCache` (policy), `ExpertResidency` seam
with `CpuResidency`, per-expert `Mlp`/`ExpertTensor` byte ranges, `--expert-cache auto`
placement math, `adaptive_budget`. New pieces are the two disk residency executors
(NVMe + SATA) and the queue that links all four tiers.

Flow per decode step (~`REFRESH_STEPS = 64` granularity for policy decisions):

1. **Router emits top-6 experts/layer.** Read the `HotExpertCache` refresh.
2. **Ordered priority set** is produced: `[dense spine (fixed)] + [hot experts] +
   [medium experts] + [Engram gather] + [cold experts]`.
3. **Tier cascade on access:**
   - Tier 1 hit → matmul reads VRAM slots (async copy stream overlaps compute, as in
     `device-expert-cache.md`).
   - Tier 1 miss, Tier 2 hit → `cudaMemcpyAsync` / SYCL transfer from RAM with event
     sync (existing `CpuResidency` → device flow). RAM page stays.
   - Tier 2 miss, Tier 3 hit → pread from NVMe into a RAM slot (async, one expert at a
     time), then promote. NVMe keeps the file backing (write-through).
   - Tier 3 miss → Tier 4 (SATA/ZFS) read as last resort — rare, and only for Engram or
     a genuinely cold expert.
4. **Eviction is LRU-by-routing-frequency at each level**, but *never demotes a member
   still in the current hot set* (the "semantic-aware LRU" from FreeToken / existing
   design). A Tier-1 eviction demotes to Tier 2 (keeps its RAM page); a Tier-2 eviction
   demotes to Tier 3 (file already there); Tier-4 is terminal.
5. **Engram handled out-of-band:** it is a gather (48 rows / 12.4 KB / token), *not* a
   weight matmul, so it does not share the expert slot pool. It stays pinned at Tier 3/4
   and is read directly; only if RAM is ample is it promoted to Tier 2 to cut latency.
6. **Persistent KV (SWA Bounded Replay)** is a separate stream, designed for SSD/host
   RAM; it gets its own reserved budget and does not evict from the weight pool.

### Adaptive budget integration
The existing placement cost model becomes 4-tier:
```
cost ≈ max( compute_time,
            Σ_tiers read_bytes(tier) / bandwidth(tier) ) per token
```
`adaptive_budget` measures hit-rate and per-tier bus bytes and grows the hotter tiers
(up to reservation floor) when misses saturate a slower tier, shrinking a hotter tier
when its hit rate is already maximal — preserving the `--expert-cache auto` surface and
the deterministic tier-1/2 sizing from §3.

---

## 6. New code surface (vs. `device-expert-cache.md`)

`ExpertResidency` gains two disk-backed implementations; the VRAM/RAM executors already
exist conceptually.

```
trait ExpertResidency { acquire(layer, expert)->handle; release(...); capacity(); }
impl Cpu      ...  existing (madvise / CpuResidency)              // Tier 2
impl Device   ...  CUDA/ROCm/Metal slot pool + async copy          // Tier 1
impl Nvme     ...  pread-backed, async, per-expert file slices     // Tier 3
impl SataZfs  ...  sparse-gather / bulk-backing, latency-tolerant  // Tier 4

4Q::priority_refresh()  -> ordered [Tier1 candidates..Tier2..Tier3..Tier4]
4Q::promote(miss_at_tier)  paged cascade with write-through invariants
```
The Engram gather path and the KV persist stream get dedicated, separate executors so
they never share (and never thrash) the expert slot pool.

---

## 7. Minimums table (design targets for the hardware buy)

| Tier | Class | Recom. min | Fair-performance rationale |
|---|---:|---|---|
| 1 VRAM | dense spine + hot experts | **24 GiB** (48–64 GiB recom.) | holds ~17 GB/token working set hot; ≥95 % hit on decode |
| 2 RAM | medium experts + Engram (+ KV replay) | **128 GiB** (192–256 recom.) | makes the hot path RAM-resident; turns disk work into memory work |
| 3 fast SSD | expert overflow, KV persist (Engram optional) | **2 TB gen4/5 NVMe** | multi-GB/s backing so non-RAM decode is *fair*, not fatal |
| 4 slow SSD | Engram tables + cold tail (whole-model presence) | **4 TB SATA mirror** | guarantees the full 397 GiB is on-box; latency-tolerant sparse reads |

**Buy order matters:** if a single node, spend first on **RAM** (raises decode ceiling
most), then **VRAM bandwidth/capacity**, then **NVMe**, then **SATA** for presence only.

---

## A. Risks / notes

- **The ~17 GB/token figure is active-params × byte/weight;**
  MXFP4 ≈ 4.25 bits/weight + scales, FP8 dense ≈ 1.06 B/weight → use ~1.06 B/weight.
  The dense-spine estimate (~12 GiB) is derived from the config (hidden 5120, 40 layers,
  272+1 experts/layer FFN non-expert parts, FP8/MXFP4 math); measure per-expert/mmap
  sizes at load for ground truth before finalizing a specific box.
- **Tier-3 SSD endurance** on a hot-weight cache is low because reads dominate; prefer
  TLC and enough overprovisioning; a write-through is only on promote, so TBW is small.
- **ZFS ARC interaction (if Tier 4 is ZFS):** pre-allocate / pin the Tier-4 mappings so
  the ARC does not double-buffered-thrash; a `zfs set primarycache=metadata` for the
  model dataset avoids wasting RAM duplicating the cold tier.
- **Multi-session sharing** of the four-tier pool needs the `HotExpertCache` counters
  lifted to be engine-shared (already a known follow-up from #48).

## B. Basis for the quantity estimates

Config facts (`/tank/share/models/DeepSeek-V4.1-Flash-REAP-272E/config.json`, base
`deepseek-ai/DeepSeek-V4.1-Flash`): hidden_size 5120, 40 layers, num_experts_per_tok 6,
n_shared_experts 1, n_routed_experts 272 (pruned from 384 by REAP), moe_intermediate_size
2304, q_lora_rank 1280, o_lora_rank 1024, MLP+attention, MTP retained, Engram gather.
REAP card and base card: 552B backbone + 196B Engram; REAP-272E = ~393B backbone
parameter-pruned, file **396.8 GiB** (= 207.7 GiB backbone here + 189.1 GiB Engram).
Active per token: **8B prefill / 16B decode**; decode is bandwidth-bound. KV cache:
global 890 B/token (HBM), persistent KV on SSD/host-RAM (SWA Bounded Replay).

---

## C. Distributing the KV cache across a large system (and the tool to use)

The four-tier cache above is a *single-node* weight plan. For a large distributed
deployment the weights barely change — only the **KV cache**(and the shared Engram /
persistent-KV spill) genuinely scale across many machines. The right mental model and
the right tool both depend on splitting KV into its two architectural classes:

- **Global KV cache (890 B/token, CSA2 + FP4, on HBM).** This is per-request, per-GPU,
  latency-critical, and must stay on the accelerator. It is **not** the thing you
  distribute; it rides in Tier 1's reservation and stays local to the GPU decoding a
  given request.
- **Persistent KV cache (SWA Bounded Replay, always SSD / host-memory).** This is the
  cache that *is* built to live off-GPU: it is explicitly designed to spill to storage and
  to be **shared across requests and serving instances**. This is the distributed object.

### The model treats prefix caching as a storage policy
V4.1-Flash vLLM/SGLang work and the model docs treat long-context KV as *cached blocks*
keyed by prefix hash, not as unmanaged GPU tensors. That means a system of record that
stores hash-addressed KV blocks on shared/networked storage, publishes them atomically,
and lets any serving worker load them is the natural fit. This is exactly what the
external-KV-cache systems below do, and it is also how DeepSeek's own deployment
middleware persists "KV cache and expert routing at token granularity" for seamless
resumption.

### Appropriate tools (external / distributed KV cache systems)
| System | Fit | How it distributes |
|---|---|---|
| **LMCache** | General-purpose, multi-engine, multi-backend, distributed | KV block store shared across serving engines + nodes; supports local RAM/NVMe, shared filesystem, and remote backends; `KV Transfer API` for cross-instance reuse |
| **llm-d** | Native, engine-agnostic | Bidirectional `KV Transfer`/`KV Offload` connectors; hierarchical hash-addressed files on any filesystem, shared across vLLM/SGLang instances |
| **py-kvcache** (or native **vLLM KV Offload API**) | vLLM-backed shared filesystem | Asynchronous direct-I/O, hash-addressed shared files, scheduler-aware preload; shares cached prefixes across vLLM instances |
| **Mooncake** / **MemServe** / **DualPath** | KV-centric *disaggregated* serving | Separate prefill/decode pools coordinated around a KV-centric data path; KV stored in an elastic/external memory pool spanning the cluster |
| **Tutti** | SSD-backed, GPU-centric | NVMe-direct KV cache, removes CPU staging from the control path |

For a large distributed deployment of V4.1-Flash the honest recommendation is: keep the
**global** KV on-GPU, and push the **persistent** KV (and the unrelated Engram tables,
which are also hash-gather and disk-tolerant) onto a **shared external KV tier** via
**LMCache** (broadest) or **llm-d / py-kvcache** (filesystem-native, if your SSDs/ZFS
pool is the target). Mooncake-style disaggregation buys the most at datacenter scale but
is the most operationally heavy.

### The critical caveat — external caching is a *break-even* decision, not free
Empirical characterization (the arXiv external-KV-caching study, §5.2) shows that
loading a prefix from an external tier only beats recomputing it above a **break-even
prefix-reuse fraction** — ~78% reuse at 8k tokens, dropping to <8% at 80k tokens on a
mid-range node. Cache *admission* must be gated on (prefix length, expected reuse, GPU
compute speed, tier bandwidth), and recomputation must remain a valid fallback — never
treat a missed external-cache load as a hard failure. This is the same
capacity-vs-compute break-even logic already in the four-tier weight plan: the
persistent-KV tier is worth its money precisely for the **1M-token, long-horizon,
agentic** workloads this model targets, and is not worth it for short, low-reuse
traffic.

---

## D. Multi-GPU "VRAM instead of RAM" (slow inter-GPU bus, large total memory)

The design can accommodate treating aggregate GPU VRAM as the memory tier that replaces
system RAM — but **only under the right interconnect**, and with the tier semantics
changed. The trap is to assume that because N GPUs sum to a large VRAM total, that memory
is a drop-in Tier-2 for host RAM. It is not, unless the GPUs are NVLink-connected.

### Why the bus, not the VRAM size, decides the tier

In §4 the whole point of Tier 2 (RAM) is that it sits on the fast on-board host path and
makes a Tier-1 miss cheap (~40–60 µs for the ~17 GB/token working set). When "the second
tier" is *another GPU's* VRAM over an inter-GPU bus, every Tier-1 miss pulls the expert
across that bus. The *per-link* bandwidth — not aggregate capacity — is what bounds the
miss path:

```
Tier 2 candidate bus      Bandwidth      ~17 GB/token miss latency
  host RAM (on-board)      ~300-500 GB/s  ~40-60 us          (the true Tier 2)
  PCIe 4.0 x16 (peer GPU)  ~31.5 GB/s     ~540 us            (~5-15x RAM penalty)
  PCIe 5.0 x16 (peer GPU)  ~63 GB/s       ~270 us
  NVLink (per link)        ~50-150 GB/s   ~110-350 us  →  closer to RAM only with
                                       several links / full NVLink bisection
  NVMe gen5                ~7-13 GB/s      ~1.3-2.4 ms
```

Two consequences:

1. **On a PCIe-only "slow bus", peer-GPU VRAM is a fast-SSD replacement, not a RAM
   replacement.** PCIe 4/5 peer bandwidth lands in the same order of magnitude as NVMe
   (30-60 vs ~10 GB/s), not the same order as RAM (300-500 GB/s). So in the four-tier
   model the correct relabeling with a slow bus is:
   ```
   Tier 1  VRAM (the decoding GPU's own)         — unchanged
   Tier 2  host RAM (if any) OR NVLink peer VRAM  — genuine RAM-class
   Tier 3  peer-GPU VRAM over PCIe  == effectively the "fast SSD" slot
   Tier 4  bulk SSD (Engram tail, cold experts)
   ```
   Aggregate VRAM buying you *capacity* (more experts resident) is still valuable — it
   shrinks how often you fall all the way to disk — but it does **not** buy you *decode
   speed* the way RAM did. You have traded RAM bandwidth for NVMe-class bandwidth
   distributed across cards.

2. **Per-GPU bandwidth, not aggregate, governs each GPU's miss ceiling.** With a slow bus,
   even 8 GPUs × 32 GiB = 256 GiB total cannot make one requesting GPU's misses fast; it
   can only stop them from falling to disk. The design must therefore model the cost as
   `max over targets of (peer latency + bytes/link_bandwidth)`, never a single pooled
   `total_bytes / total_bandwidth`.

### When it DOES reach Tier-2 class: NVLink / peer-to-peer

If the GPUs share a high-bandwidth interconnect — NVLink-connected (NVLink/peer-to-peer),
or a CXL-attached pool — peer-VRAM bandwidth enters the RAM order of magnitude and the
"VRAM instead of RAM" plan is legitimate. Then aggregate VRAM genuinely replaces host RAM
as Tier 2, and the reservation math of §3 carries over with `Tier2 = sum(peer VRAM)`.
The distinguishing test for the hardware buy:
> measure a peer-peer copy bandwidth (`cudaMemcpyPeer`/`nvbandwidth` P2P). If it is
> **≥ ~150-300 GB/s** treat peer VRAM as Tier 2; if it is **≤ ~60 GB/s** treat it as
> Tier 3 (fast-SSD-class).

### Placement / reservation consequences for the design

- **Expert→GPU affinity must be interconnect-aware.** A medium-hot expert should live on
  the GPU most likely to decode a token that selects it (router-locality aware), and
  replica it on a second GPU only if the peer bandwidth justifies the duplication.
- **Avoid cross-link round-trips in the cascade.** Promote along a Topology, not at random:
  a Tier-1...Tier-3 (peer GPU) eviction should demote to a *peer* that is a single
  high-bandwidth hop from the original, never a far card over a cheap spine.
- **Aggregate VRAM still helps the hit-rate math.** The "fair" ≥95 % Tier1+2 hit target
  of §4 becomes "Tier1 + reachable-at-RAM-bw" here; the slow-bus portion instead
  contributes to keeping decode off disk, which is the *capacity* benefit, priced
  accordingly (cheap in dollars, bounded in speed).
- **Attenuate the §4 conclusion.** "RAM buys decode speed" becomes tweaked under the
  slow-bus plan: with a PCIe-only multi-GPU rig the fastest thing you can reach for a
  miss is still peer-VRAM *over PCIe*, so the buy guidance shifts to either (a) enough
  NVLink to make peer VRAM a true Tier 2, or (b) accept peer VRAM as an NVMe-class Tier 3
  and put the *decode-speed* dollars into fewer, larger, faster per-GPU VRAM within each
  card's own Tier 1.

### Bottom line
Yes, this configuration is accommodated — with a slow bus you get a **capacity** win
(more experts resident, fewer disk faults), not a **speed** win, and peer GPU VRAM
should be slotted as the fast-SSD tier, not as RAM. You buy genuine Tier-2 behavior only
with a high-bandwidth (NVLink-class) interconnect linking the VRAM you intend to use as
memory. For a sparse, bandwidth-bound 16B-active model like V4.1-Flash, that distinction
is decisive: it is better to buy one big fast GPU (bigger Tier 1) than many small GPUs
whose combined VRAM you can only reach at PCIe speed.

---

## E. Model-level levers to cut the bandwidth requirement (honest inventory)

A 16B-active, bandwidth-bound decode reads `active_params × bytes_per_active_weight`
per token (~17 GB/token). The hardware cache moves *where* those bytes come from; a
*model* restructure attacks the bytes themselves. There are exactly three-and-a-half
levers, and they partition cleanly into ones that preserve intelligence and ones that
trade it.

### Levers that preserve intelligence (win the read without losing quality)
- **True active-parameter sparsity (the genuine lever).** MoE intelligence is stored in
  the *total* pool (552B backbone + 196B Engram) but only 6/272 routed + 1 shared fire
  per token. Cutting the **active** count — e.g. deeper router skew so most tokens touch
  fewer, larger experts; or layer-level sparsity that skips a layer for easy tokens — is
  the honest restructure. It does not remove knowledge, it changes which weights per
  token are *paying*. This is what REAP was aiming at (384→272, +4.13% perplexity, active
  compute unchanged); a *routing-aware* active reduction is the untapped next step that
  would actually move bandwidth.
- **Speculative / multi-token reuse (DSpark, MTP).** Verify several candidate next
  tokens against one weight read instead of one-token-per-read. This cuts *effective*
  bytes/token without touching weights. NVIDIA's Dynamo recipe for V4.1-Flash already
  pairs it with DSpark speculative decoding, and MTP (retained here) is the model-native
  form of the same idea.
- **Recompute-vs-read for the dense spine.** The dense spine is read on *every* token but
  is largely static. Fusing/folding tiny dense matrices, or opportunistically re-deriving
  a cheap low-rank projection instead of re-fetching the full one, keeps the always-touched
  working set small. Latent/low-rank attention (the config's `q_lora_rank`/`o_lora_rank`,
  MLA) is already this; pushing more per-token work into already-cost-amortized experts
  and *shrinking the always-resident spine* is the lever that reduces the real Tier-1
  floor.

### Levers that look free but are a quality-for-bytes trade
- **Deeper quantization (FP4/2-bit dense, NVFP4 engines).** Would cut
  `bytes_per_active_weight`, but at a real accuracy cost; the model already sits at the
  practical MXFP4/FP8 frontier. This is *reducing* the bytes by reducing the precision of
  the knowledge — the opposite of "without reducing intelligence."
- **Blind expert eviction / aggressive pruning** past REAP's keep-set. Works only if the
  router truly never needs the pruned experts; beyond that it is losing the tail
  knowledge.

### The conservation-law reality
"Without reducing intelligence" forces you into **sparsity (fewer active bytes) and
reuse (fewer reads per token)** — the two levers that cut *bandwidth* while leaving *total
stored knowledge* intact. Everything else (deeper quant, hard pruning, distillation to a
smaller model) reduces bandwidth by reducing what the model holds. For the four-tier plan,
the model-side win doesn't change the cache — it raises the threshold where Tier 1 stops
being the bottleneck, so the "fair" VRAM minimum of §7 can come *down* by roughly the same
factor as the active-params reduction.

---

## F. Worked scenario: 2×16 GiB GPUs over PCIe 4/5 (with the two model levers)

Concrete numbers for the specific hardware: two 16 GiB cards connected by PCIe 4.0 x16
(~31.5 GB/s) or PCIe 5.0 x16 (~63 GB/s), no NVLink. This is the "peer VRAM as Tier 3"
regime of §D. Baseline constants: decode active 16B → ~17 GB/token; HBM ~700 GB/s;
per-expert ~19 MB; 272×40 = 10,880 routed experts.

### Capacity reality
```
Usable VRAM (2 cards)            ~30 GiB
Dense spine (resident, decode GPU) ~12 GiB   (must stay on-card)
Hot-expert pool budget            ~18 GiB   (left over)
Hot experts resident              ~947 of 10,880 (≈9%)
Router top-5% hot set (544 exp)   ~10 GiB   -> fits on-card
Router top-8% hot set (870 exp)   ~16.5 GiB  -> fits in aggregate
```
So a good `HotExpertCache` keeps the *actually-used* experts resident across the two
cards. But only ~15 GiB is on the decoding card itself; experts living on the peer must
cross PCIe at 31.5 / 63 GB/s, which is NVMe-class, not RAM-class (§D).

### Achievable decode rate (bytes split across tiers)
Per-token time = `fr*BB/HBM + fp*BB/peerBW + fd*BB/NVMe(10GB/s)`, `BB`=bytes per token.

| On-card | peer | disk | tok/s (PCIe4) | tok/s (PCIe5) |
|---|---:|---:|---:|---:|
| 90 % | 8 % | 2 % | **10** | **13** |
| 95 % | 4 % | 1 % | **16** | **20** |
| 80 % | 15 % | 5 % | 5 | 7 |
| 70 % | 20 % | 10 % | 3 | 4 |

With the spine resident and a hot cache, expect **~8–20 tok/s**, peer-bandwidth-bound.
These bracket the established ~22–25 tok/s single-big-card reference, confirming PCIe
peer hits cap the rig.

### Effect of the two model-level levers (§E)
Deeper active sparsity (17 → ~6–8 GB/token) and speculative reuse (2–3 tok per read,
effective 8.5–5.7 GB/token) overlap; taken together on the same 90 % on-card residency:

| Active sparsity | bytes/token | tok/s (PCIe4, 90 %/8 %/2 %) | tok/s (PCIe5) |
|---|---:|---:|---:|
| 16B (current) | 17.0 GB | 10 | 13 |
| ~10B (top-4+shared) | 10.6 GB | 16 | 21 |
| ~8B (encoder-like / reuse) | 8.5 GB | 20 | 26 |
| ~6B (topk-4 + layer-skip) | 6.4 GB | **27** | **35** |

**Headline:** combining top-k reduction with speculative reuse roughly **doubles** the
throughput this 2×16 GiB rig can actually deliver — from ~10 to ~27 (PCIe4) / ~35 (PCIe5)
tok/s — because both levers subtract from `bytes/token` without touching the resident
weights.

### Buy guidance for this rig
1. **Do the model-side change first** — it is the cheapest way to make 2×16 GiB viable and
   is the intelligence-preserving path (§E: sparsity + reuse).
2. **Spend nothing on fast NVMe for hot-path decode** — with the spine + hot set resident,
   SSD is only a cold-tail safety net here.
3. **Capacity ≠ speed on this box.** Peer PCIe is NVMe-class; the upgrade that actually
   helps is a single **32–48 GiB** card (bigger Tier 1, HBM-speed hits) rather than adding
   more small PCIe cards.

---

## G. Gap analysis — how close Joshua is to this design

Verified against source (updated 2026-10-28), not just the design docs. Overall: the **two
hot tiers (VRAM + RAM), the Tier-3 SSD slab, the Engram placement policy, the policy, and the
hierarchy simulator are built and merged**; the **multi-GPU peer pool and the distributed/
external KV tier remain the gaps**.

### Implemented (merged, real code)

| Element of this design | Joshua evidence | status |
|---|---|---|
| Tier 2 — RAM residency over mmap | `residency.rs` `ExpertResidency` + `CpuResidency` (`MADV_WILLNEED` over per-expert mmap ranges); model `mmap`ed read-only (`engine.rs`), OS page cache is the implicit `disk→RAM` tier; `MADV_SEQUENTIAL` prefill + layer-ahead `pread`, `MADV_RANDOM` cold | ✅ |
| Routing-frequency LRU policy | `hot_experts.rs` `HotExpertCache`, `REFRESH_STEPS=64`, freq-then-recency sort | ✅ |
| Tier 1 — VRAM slot cache + async upload | `residency.rs` `DeviceResidency<T>` (LRU slots, hot-set protection `mark_hot`/`replace_hot_set`) + `ExpertUploader` (async, non-blocking, `HOST_RELEASE_DELAY`); `quantized_deepseek4.rs` `Ds4DeviceExpert` | ✅ |
| V4.1 experts on accelerator | CUDA IQ2_XXS kernels (`cf0c938`), Vulkan (`0e953af`); `--vram-expert-cache auto` sizes `free − dense − headroom − KV` (engine.rs:1683) — §3 Tier-1 formula | ✅ |
| Auto placement / sizing | `--expert-placement auto\|device\|host`, `--expert-cache auto`, `--vram-expert-cache auto`, RAM-adaptive pool caps, weight sharing across sessions | ✅ |
| Full-hierarchy *simulator* | `cache_sim.rs` models device slots + host page cache in front of disk, `--slots`/`--host-slots`, inclusive vs. exclusive residency | ✅ |
| Model reuse lever (§E) | `src/speculative.rs` `NgramDrafter`/`SpeculationGate`, prompt checkpoints + expert look-ahead (`af41e92` Strata/DwarfStar); expert next-step prefetch | ✅ (partial) |
| **Tier 3 — explicit NVMe/SSD slab** | `residency.rs` `NvmeResidency<T>` + `NvmeStaged` (hard-cap host-RAM slab, `pread(2)` staging, non-blocking `ExpertUploader`, hot-set protection, host-page release hook); `--ssd-expert-cache` (`SsdExpertCache`); merged **#207** | ✅ (policy/executor) |
| **Engram-aware placement** | `placement.rs` `EngramTier`/`resolve_engram_residency` (~192 GiB promote floor); `--engram-tier`; engam resolved from mmap, disk-pinned by default, reported separately (`engram_residency()`); merged **#205** | ✅ |

### Not implemented (the real gaps)

| Element of this design | Gap | effort |
|---|---|---|
| Multi-GPU peer level (§D/F) | Single-pool device model; routing each expert to RTX *or* Arc explicitly listed as a large refactor ("two-device expert pool") in `noram-perf-progress.md` | very high |
| Distributed / external KV (§C) | No LMCache/llm-d-style prefix-cache tier; no external or persistent-KV offload | high |
| Tier 4 — explicit SATA/ZFS residency | No distinct lowest storage tier; the raw file (below the NVMe slab) stands in for it | medium |
| Adaptive runtime feedback loop (phase 5 runtime) | Budget sizing is `auto` (once at load), not a live hit-ratio/bus-bytes tuning loop | medium |

> Note: `docs/device-expert-cache.md` still listed the VRAM slot-cache and deepseek4 GPU
> phases as "future" — that is **stale**; the commits above merged them. That table was
> corrected alongside this section (§G supersedes it).

### Bottom line
Joshua implements the **hardest, highest-value foundation** — Tier 1 VRAM slot cache (async,
hot-protected, auto-sized), Tier 2 RAM/mmap residency, the routing-frequency-LRU policy,
and a hierarchical hit-rate *simulator* that already thinks in terms of device→host→disk.
What remains is **(1) the multi-GPU peer pool (§D/F)**, **(2) the distributed/external
KV tier (§C)**, and the Tier-3 *dispatch* wiring (the slab and flag exist and are tested;
the deepseek4 loader engagement is tracked separately). The model-level levers (§E) are
partly shipped via speculative decoding and expert look-ahead.

---

## H. Explicit NVMe residency (Tier 3) + Engram-aware placement

> **Implementation status (2026-10-28):** the policy and executor landed as **#207**
> (`NvmeResidency` + `--ssd-expert-cache`) and the Engram placement as **#205**
> (`EngramTier` + `--engram-tier`). Both are merged on `main` and CPU-testable. Remaining:
> the deepseek4 loader *dispatch* wiring that constructs an `NvmeResidency` at load
> (tracked separately) — until that lands, `--ssd-expert-cache` sizes a slab that no
> loader engages.

Highest-value gap to close first, per §G. The seam already exists; what is missing is
making the disk a *managed* tier rather than the implicit mmap page-cache.

### Why this one first
- It is the actual wall on the hardware: `noram-perf-progress.md` names **NVMe streaming
  (~2 GB/s) of the 201 GB expert pool** as the bottleneck — the Tier-3 path in §4.
- The primitive it needs is **already present**: `mmap_tensor::MappedRange` carries
  `offset` (= file offset), `len`, `resident_pages()`, `drop_pages()`, `evict_from_cache()`;
  `mmap_tensor::LayerPrefetcher` is a working layer-ahead `pread(2)` streaming thread;
  `residency::ExpertResidency` is the dispatch seam; `hot_experts::HotExpertCache` names the
  working set. This is a generalization, not a rewrite.
- It slots into the hierarchy cleanly: `DeviceResidency` (Tier 1) → `CpuResidency`
  (Tier 2) → **new `NvmeResidency` (Tier 3)** → the raw file (Tier 3.5/4).

### Design

A new `impl ExpertResidency for NvmeResidency` that, unlike `CpuResidency` (`madvise`, which
is bounded by how much the OS keeps in page cache under pressure), **explicitly streams an
expert's `MappedRange`s into a caller-owned RAM staging slab via `pread(2)`, evicting the
least-recently-used staging entries and release-hooking the now-cold page-cache pages**. Key
properties mirroring `MmapPrefetch`/`ExpertHandles`:

```rust
pub struct NvmeResidency {
    // per-layer expert -> the three mapped byte ranges (reuse ExpertHandles)
    experts: Vec<Vec<Option<ExpertHandles>>>,
    // bounded staging ring in host RAM: [layer][expert] -> byte tensor or file offset
    stage: MruStage<Key, staged::Bytes>,
    // its own file fd for independent pread readahead (mirror LayerPrefetcher)
    file: Arc<File>,
    budget_bytes: u64,        // caller-set, default = RAM_floor fraction (@see §3)
    depth: usize,             // streaming depth, reuse prefetch_ahead_depth()
    hot: Mutex<HashSet<Key>>, // never evict the routing-frequency hot set
}

impl ExpertResidency for NvmeResidency {
    fn acquire(&self, layer, expert) { /* pread missing staged bytes -> request device upload */ }
    fn release(&self, layer, expert) { /* free a staging slot; drop host pages */ }
    fn capacity(&self) -> usize { /* budget_bytes / per_expert_bytes */ }
    // mark_hot/replace_hot_set: protect the @HotExpertCache set from LRU eviction
}
```

Concretely the executor reuses the `LayerPrefetcher::run_prefetch` `pread(2)` loop under a
new name (a slot-staged variant fed by routing, not layer order), the `MappedRange` structs
for exact offsets, and mirrors `ExpertUploader`'s async/non-blocking discipline so a Tier-3
fault never stalls decode.

### Engram-aware placement (the same commit, smaller)

`quantized_deepseek4.rs` already parses `{arch}.engram.*` (`EngramConfig`, `engram.layer_ids`).
Because the Engram tables are ~189 GiB of latency-tolerant hash gathers (§2 Rank 4, §C), they
**should not** compete with routing experts for Tier-1/2 budget. Add one placement policy:

```
resolve engram fragment straight from the mapping (never a Tier-1/2 slot);
pin to Tier 3 (or Tier 4 on ZFS) by default; promote a working subset to Tier 2 only
when RAM is ample (§3 floor 128→192+ GiB); report its residency separately in metrics.
```

This is orthogonal to the expert pool and needs no matmul-path change (Engram is a gather).

### Phased plan (mirrors `device-expert-cache.md` style)

| Phase | Deliverable | Params / seam |
|---|---|---|
| 1 — seam (no behavior change) | `NvmeResidency` strawman that forwards to `CpuResidency`; keep the CPU tests green. Proves the seam and the budget plumbing. | `--ssd-expert-cache auto\|<n>` |
| 2 — staging slab | Bounded host-RAM staging ring + `pread` streaming for Tier-3 experts, reusing `LayerPrefetcher` fd/readahead; release-hook cold pages like `ExpertUploader::HOST_RELEASE_DELAY`. | budget = §3 Tier-3 reservation |
| 3 — Engram tier | `EngramConfig` residency/pinning policy (§ above) + separate metrics; no expert-pool interaction. | `--engram-tier auto\|ram\|disk` |
| 4 — adaptive | Feed measured Tier-3 hit-rate and staged bytes (reuse `DeviceResidencyStats`) into budget; grow/shrink like the `--vram-expert-cache auto` loop. | extend `adaptive_budget` |
| 5 — latency path | Keep Tier-3 reads off the decode critical path: issue `pread` from the previous step's routing (mirror `prefetch_hint`), bound queue depth. | `prefetch_hint` → stream |
| 6 — Tier 4 | SATA/ZFS (bulk) as the last resort + the `NvmeResidency`/`SataZfs` split. | map §D/F hybrid |

### Verification
- **Unit:** `NvmeResidency` byte-identical to `CpuResidency` when the slab is large enough
  (parity test on tiny GGUF fixtures in `tests/common`); staging accounts for the budget;
  hot-set protection keeps `@HotExpertCache` members resident; release-hook drops cold pages.
- **Integration (GPU CI / kleya):** run the existing `DSV41-REAP272E-balanced.gguf` trace;
  assert (a) Tier-3 staged bytes ≈ miss bytes, (b) hit-rate on a repeated prompt exceeds the
  page-cache-only baseline from `noram-perf-progress.md`, (c) throughput vs. `CpuResidency` —
  the NVMe wall is the headline number to move.
- **Report** measured tok/s and per-tier bytes on the §F 2×16 GiB rig specification and on
  kleya, with the same caveat about OS page-cache drift noted in `noram-perf-progress.md`.
