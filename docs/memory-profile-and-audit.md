# Memory configuration: peak-release audit and a unified profile

**Status:** audit complete + design proposal. Follows the axolotl comparison —
what actually transfers from a training framework to this inference engine, and
what does not.

Two questions, both asked against `origin/main` @ `7e8957f`:

1. **Peak-memory release audit** — does each loader free the transient form of a
   tensor at the right moment? (Axolotl's `quantize_moe_experts` insight: peak is
   a *different budget* from steady state.)
2. **Unified config** — 107 CLI flags, most cache knobs declared twice (once for
   `serve`, once for `run`). Can one profile name the memory plan?

---

## 1. Peak-memory release audit — finding: no significant defect

### What was checked

| Question | Result |
|---|---|
| Is the token-embedding table still quantized (not f32 per session)? | **Yes** — all three native loaders use `TokenEmbedding::load(...)`; the R1 fix in `memory-analysis.md` holds (`quantized_deepseek2.rs:1409`, `quantized_deepseek4.rs:3494`, `quantized_qwen.rs`). |
| Does any loader hold a whole **expert** tensor in float? | **No** — expert path slices `mmap` ranges per expert (`ExpertHandles`); `moe.rs` `expert_tensor` refuses raw-dtype expert stacks instead of materialising them. |
| Are block-quant dequantizers per-block or whole-tensor? | **Per-block** (`iq2xxs.rs:43/351/596`, `mxfp4.rs:141` dequantize one block into a caller buffer). Correct — no whole-tensor float copy. |
| Does `f32_tensor` hold two allocations at once? | **Yes, but bounded** — `qtensor(name)?.dequantize()?.to_dtype(F32)` is two live tensors. Every caller is a *small* tensor (see below). |
| Is residency built from cheap clones, not copies? | **Yes** — `CpuResidency::new(layers.iter().map(.. m.prefetch.clone() ..))` clones `Arc<dyn MmapPrefetch>` handles. |

### The one real (but minor) inefficiency

`src/tensor_source.rs:67`:

```rust
pub fn f32_tensor(&mut self, name: &str) -> Result<Tensor> {
    self.qtensor(name)?.dequantize(&self.device)?.to_dtype(DType::F32)
}
```

Two full-tensor buffers are alive at the same moment (dequantized, then the f32
copy), plus the source `QTensor`. Callers are:

- routers `ffn_gate_inp` `[n_expert, n_embd]` (272 × 5120 → ~5.6 MB fp32)
- Hyper-Connection `hc_base` / `hc_scale` vectors
- `attn_sinks`, `engram_q` / `engram_k`, `attn_res_score` / `ffn_res_score`
- `ape` positional table (the largest of these, still one layer's worth)

None is a routed-expert or embedding-sized tensor, so the transient peak is a few
tens of MB at most. **Verdict: not worth changing for memory reasons** — and
changing it risks the exact dtype/rounding parity that `AGENTS.md` asks us to
preserve on this path. Recorded so the next audit does not re-derive it.

### Where axolotl's insight *does* apply

The principle (free the source the moment the compact form exists) is already
honoured at the two places it matters:

- **device expert upload** — `memory-analysis.md` V2 removed the triple-copy
  (whole tensor → host `Vec` → device → download → per-expert upload) in favour of
  `mmap_tensor::expert_slices` + one `from_data` per expert.
- **Tier-3 stage** — `NvmeResidency` streams into a *bounded* slab and calls the
  host-page release hook once the staged copy settles, so the page cache does not
  hold a second copy (`HOST_RELEASE_DELAY`).

So the borrowable idea is *already implemented*. **No code change recommended
from the audit.**

---

## 2. Unified memory profile — design

### Problem

The memory knobs are spread across flags whose interaction is non-obvious and
whose *combination* is what actually matters:

```
--pin-hot-weights, --pin-hot-experts, --expert-cache, --vram-expert-cache,
--ssd-expert-cache, --engram-tier, --expert-placement, --prefetch-model,
--lazy-weights, --mlock-hot-weights, --prefill-chunk, --prefill-budget-ms
```

They are declared **twice** (serve + run), and each has its own `auto|<n>`
grammar with different units (experts vs MiB vs bytes). Nothing states the
*coherent plan* ("Tier-2 holds the pool, Tier-3 backs the overflow, Engram on
disk").

### Proposal: `--profile <name>` naming a memory plan

One `--profile` selects a **coherent set** of tier decisions; individual flags
still override. This is the axolotl "one config across the lifecycle" idea
narrowed to what an inference engine actually needs.

```toml
# joshua.toml (optional; flags still win)
[profile]                      # default profile
name = "ssd-backed"

[memory]
host_pool        = "auto"      # Tier-2: experts resident in RAM (auto = fit available)
hot_experts      = "auto"      # Tier-1: VRAM slot pool
ssd_slab         = "auto"      # Tier-3: NvmeResidency budget (auto = 1/2 free RAM)
engram           = "disk"      # Tier-3/4 by default; "ram" only when ample
placement        = "auto"

[load]
prefetch_model   = "auto"
lazy_weights     = "auto"
mlock            = "off"       # MlockMode: on | required | off
```

### The three built-in profiles

| Profile | Tier-1 VRAM | Tier-2 RAM | Tier-3 SSD | Engram | Intended box |
|---|---|---|---|---|---|
| `fit` | all experts if they fit | — | off | ram | model ≤ VRAM |
| `host` | dense spine only (`--expert-placement host`) | hot experts (auto) | off | ram if ample | model ≤ RAM |
| `ssd-backed` | dense + hot subset | hot subset (auto) | **on** (auto) | disk | model > RAM |

`ssd-backed` is the §F/large-model default and is exactly the plan the
multilevel design describes.

### Semantics

1. **Profile is a set of defaults, not a mode switch.** Every existing flag keeps
   working; an explicit flag beats the profile. `--profile` is therefore
   backwards-compatible and cannot regress a tuned invocation.
2. **`auto` is resolved once at load and logged**, as the VRAM cache already does
   (`vram-expert-cache auto: X GiB ...`), so the effective plan is auditable.
3. **Invalid combinations are rejected at load, naming both values** — the
   pattern `--expert-placement` already uses when the dense set does not fit.
   E.g. `ssd_slab` on with `host_pool = "all"` and a pool larger than RAM is fine;
   `engram = "ram"` with a <192 GiB machine is a warning, not a silent promotion.
4. **No new flags for the tiers themselves** — the profile only *coalesces*
   existing knobs, which keeps the 107-flag surface from growing.

### Why not just document the flags?

Because the failure mode this prevents is real and expensive: a user sets
`--pin-hot-experts 2800` on a machine where that exceeds RAM and gets an OOM, or
enables `--ssd-expert-cache` without realising it competes with the host pool for
RAM. A profile states the invariant ("the pool fits RAM") that the individual
flags do not.

### Implementation sketch (not yet done)

- `src/profile.rs`: `Profile { Fit, Host, SsdBacked }` + `resolve(&self,
  machine_mem, model_bytes) -> MemoryPlan`, pure and unit-testable exactly like
  `placement::resolve_engram_residency`.
- `EngineOptions` gains `profile: Option<Profile>`; the existing setters keep
  precedence.
- CLI: one `--profile` on both `serve` and `run` (the duplication stays, but for
  one flag instead of twelve).
- Load-time log line: `profile `ssd-backed`: host pool 118 GiB, vram slots 947,
  ssd slab 8 GiB, engram disk`.

### Open questions

- Should a profile be able to *disable* a flag the user set explicitly, or only
  fill unset ones? Proposed: fill-unset only (never surprise a tuned user).
- Does `fit` need to imply `--mlock-hot-weights on`? No — mlock is an operator
  choice about competing workload (and about the `memlock` rlimit), not a
  model-fitness decision.
