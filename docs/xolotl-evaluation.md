# Xolotl: evaluation against the multilevel cache design

**Source:** [Indras-Mirror/Xolotl](https://github.com/Indras-Mirror/Xolotl) — README,
`docs/xolotl/HOW_IT_WORKS.md`, `docs/DEEPSEEK_V41.md` (read 2026-10-28).
Evaluated against `docs/multilevel-weight-cache.md` (§0–H) and the code on
`origin/main` @ `7e8957f`.

**Verdict up front:** unlike axolotl, this is the closest external artifact to
our problem that has come up — same model, same 3-tier problem, with **measured
numbers on real hardware**. It confirms the tier model, corrects one of our
assumptions, and contributes four concrete techniques we have no equivalent of.
It is also **experimental single-developer research code**, so its numbers are
one machine and its practices are not automatically safe to copy.

---

## 1. What it is

A **Strata fork** (`Niko1221/Strata`, merged at 0.1.44.1) — a C++/CUDA/ggml
engine, not Python. ~71k lines added: DeepSeek-V4.1-Flash, V4-Flash,
GLM-5.3-Flash and Qwen3.8 engines. Serves OpenAI + Anthropic APIs with tool
calls, one sequence at a time.

**Reference measurement point:** RTX 4090 (24 GB) + Ryzen 7 5700X + 90 GB DDR4 +
one NVMe → **DeepSeek-V4.1-Flash 552B at 25.7–26.8 tok/s decode**, 289–409 tok/s
prefill, 512K context.

This is directly comparable to the §F worked scenario in our design, where I
predicted for a 2×16 GiB PCIe rig: **10–20 tok/s** with the hot cache, and
~25–35 tok/s only *after* model-level active-param reduction. Xolotl gets
26.8 tok/s on one 4090 **without** changing the model — which means our §F
estimate was **pessimistic** for a well-engineered single-card setup, and the gap
is engine work, not architecture.

---

## 2. Their tiering vs ours

| Our design (§1) | Xolotl | Match? |
|---|---|---|
| T1 VRAM: hot-expert LRU slot pool, sized from free VRAM | "hottest ~1,300–1,600 experts (sized from measured free VRAM after the prompt)", `--vram-lru` | ✅ **Same design, independently arrived at** |
| T2 RAM: experts resident in host memory | "Pinned RAM arena, ~6,700 experts (60 GiB, `cudaHostRegister`), DMA straight to the GPU" | ⚠️ Same *role*, but **pinned** — see §3.1 |
| T3 fast SSD: `pread` staging slab | "NVMe file tier, O_DIRECT reads at ~4.5–5.3 GB/s, prefetched a layer ahead" | ✅ Same role; **O_DIRECT + 32 concurrent reads** is better than our plain `pread` |
| T4 slow SSD: cold/Engram, latency-tolerant | Engram table (~200–300 GB) read from disk at layers 1 and 14 | ✅ **Independently confirms our §2 Rank-4 placement** |
| `HotExpertCache` routing-frequency LRU, refresh 64 | "census-ranked" LRU + **route bias** | ⚠️ Ours is more principled; theirs is faster — §3.2 |

So the tier *decomposition* is confirmed by an independent implementation. That
is the single most valuable finding here: our §0–§3 is not idiosyncratic.

---

## 3. Four things we do not have

### 3.1 Pinned (page-locked) RAM for the host tier — we use `madvise` instead

They `cudaHostRegister` ~60 GiB so the GPU DMAs from it directly. We use
`MADV_WILLNEED` over mmap and let the kernel manage residency. Pinning buys
deterministic DMA bandwidth (no page-fault stalls mid-transfer) at the cost of
RAM the OS can no longer reclaim. **This is a real gap in our Tier-2 design**,
and it is the mechanism that makes their "served from RAM" faster than a
page-cache hit.

Caveat: 60 GiB pinned out of 90 GB is aggressive; if their process is killed, or
another workload needs RAM, that memory is unavailable.

Verified against our tree: there is **no `cudaHostRegister` call anywhere** in
joshua, and `mlock` is not applied in the residency or staging paths at all (the
`--mlock-hot-weights` flag, `MlockMode::{On, Required, Off}`, covers the hot
weight set elsewhere). So the DMA-from-pinned-RAM mechanism is genuinely absent,
not merely un-defaulted.

### 3.2 "Route bias" — deliberately trade exact routing for speed

This is their most interesting and **most dangerous** idea. They bias the router
toward experts that are already resident, so tokens hit the fast tiers:

| route bias | HumanEval (21) | DSpark tok/s |
|---|---:|---:|
| x0.25 | 20/21 | 13.7–16.1 |
| **x0.5 (default)** | **20/21** | **22.3–30.3** |
| x0.75 | 19/21 | 28.3–35.2 |
| 1.0 (old default) | 16/21 | 37.7–39.1 |

They measured that the default bias **costs 14 points of router agreement with
the DeepSeek API** (86.2% → 72.1%) and "causes repetition loops on some prompts".
They also ship `STRATA_RB_PROTECT` / `STRATA_RB_POOL=M` to keep picks inside the
true top-M, and per-request overrides so one server can answer exact-routing
requests.

**Our position should not change:** `AGENTS.md` requires preserving model
outputs, and our `hot_or_staged` fix in #212 was *specifically* to keep logits
bit-identical. Route bias is the opposite trade. But it is worth recording as a
**documented, opt-in** speed/quality knob — the "0.25 is near-exact at 26%
swapped" data point is genuinely useful.

### 3.3 Conversation-adaptive arena — learn the session's experts between turns

On by default (`STRATA_ARENA_REBAL`). Per-expert routed-token counts, decayed per
prompt; after each prompt a **swap plan** moves the conversation's frequently
used file-tier experts into the RAM arena, evicting rarely-used ones, and **a
reader thread performs the swaps in the NVMe's idle time** — i.e. while the agent
is running its tools, not during decode.

Measured: file-tier reads per small turn **−21..−30%**, small-turn time 8.48 →
7.60 s, verified byte-exact (0 bad of 2,500 slots), quality unchanged.

**This is a real gap for us.** Our `HotExpertCache` counts routing frequency but
does **not**:
- decay per prompt,
- plan *swaps* (it just re-selects a hot set),
- schedule the work into idle time,
- adapt to the *conversation* rather than the aggregate.

For an agentic workload — which is exactly what V4.1-Flash is for — this is the
single most transferable idea in the repo.

### 3.4 O_DIRECT + parallel reads + slab engineering

- `O_DIRECT` to bypass the page cache (they are explicit that the page cache
  fights them at this size; we rely on it).
- 32 concurrent reads per call, up from 8: **−17% per expert read, +4–5% decode**.
- **Not exFAT** — "Linux's exFAT driver serialises parallel reads; expert reads
  were 5x slower here". A concrete, surprising, actionable warning.
- Reclaim VRAM opportunistically: the prompt batch's 8192-column state is
  allocated only while a batch runs → **+2.4 GiB of expert slots while serving**.
- Their NVMe tier hits ~5.3 GB/s ("the NVMe's ceiling") — matching the
  `noram-perf-progress.md` note that Kleya's wall is ~2 GB/s streaming.

---

## 4. Where our design was wrong or naive

1. **§F was pessimistic about what engine work alone can reach** — with a
   caveat. Their box is 24 GB VRAM + **90 GB** host RAM, whereas §F was 2×16 GiB
   with peer VRAM as the *slow* tier and no large RAM arena. So this is not a
   like-for-like correction: a 90 GB RAM arena is doing a lot of the work, and
   that is precisely Tier-2 capacity §F did not have. What it does establish is
   that **engine engineering reaches ~27 tok/s without touching the
   architecture**, so §E's active-param reduction should be presented as *one*
   lever, not the only route to that number — and that §F's VRAM minimums assume
   a Tier-2 arena we had not made room for.
2. **"Bigger RAM arena always helps" is false at their scale.** 62 / 65 GiB cut
   NVMe reads 10% but decode stayed within noise (24.4–25.2 tok/s) "because the
   file penalty already keeps decode routing off the NVMe tier". Our §3 Tier-2
   recommendation (192–256 GiB "so the whole pool is RAM-resident") should be
   qualified: **beyond the point where decode routing stops touching the file
   tier, more RAM buys nothing.**
3. **Plain `pread` may be leaving bandwidth on the table.** O_DIRECT + higher
   queue depth is measurably better; ours uses the file's own fd with default
   buffering.
4. **Page-cache reliance is a liability, not just a convenience.** We treat the
   OS page cache as our implicit disk tier; they explicitly bypass it because it
   makes latency unpredictable at this working-set size. Our `NvmeResidency`
   already takes the first step (an owned slab) — going `O_DIRECT` would complete
   it.

---

## 5. What I would actually borrow, in order

| # | Idea | Effort | Expected |
|---|---|---|---|
| 1 | **Conversation-adaptive arena**: decayed per-conversation routing counts + swap plan executed in idle time | medium | −21..−30% file-tier reads per agent turn |
| 2 | **`O_DIRECT` + higher queue depth** in `NvmeResidency`'s stage path | low | their +4–5% decode, −17%/read |
| 3 | **Optional `cudaHostRegister` pinning** for the Tier-2 arena (`--pin-arena <n>`) | medium | deterministic DMA, fewer stalls |
| 4 | Document the **route-bias trade** as an opt-in knob, *not* a default | low | user choice; see the 14-point cost |
| 5 | Propagate the **filesystem caveats** (not exFAT, not HDD) into our docs | trivial | avoids a 5× regression |

## 6. What is *not* safe to copy

- **Route bias as a default.** It costs 14 points of router agreement and causes
  repetition loops. Our `AGENTS.md` output-preservation rule forbids it by
  default.
- **Their numbers as ours.** "Measured on one PC", single developer, explicitly
  "experimental research code". The *techniques* transfer; the tok/s figures are
  a single-machine data point.
- **60 GiB of pinned RAM** as a recommended default — that is 2/3 of their
  machine held unreclaimable.

## 7. Reproducibility caveat

I read their documentation, not their source. Claims about *their* internals are
as-reported; the tier comparison and the two corrections to our design (§4.1,
§4.2) are the parts I would act on, and both are checkable against our own code
and hardware.
