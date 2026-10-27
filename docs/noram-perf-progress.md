# Free (no-RAM-purchase) performance work for DeepSeek-V4.1 on kleya

Branch: `dsh/cuda-ig2xxs-kernels`. All builds with `--features cuda` (and the opencl arc path).

## Goal
Speed up DeepSeek-V4.1 Flash decode on kleya (RTX 5060 Ti + Intel Arc B50, 74 GB RAM)
without buying RAM. The wall is NVMe streaming (~2 GB/s) of the 201 GB IQ2_XXS routed
expert pool, not GPU compute or VRAM capacity.

## Levers implemented (each committed + unit-tested)

### 1. a560b40 — CUDA IQ2_XXS kernels, experts run on CUDA
Vendors candle-kernels (vendor/candle-kernels), adds IQ2_XXS tables + dequantize_block
f32/f16 + get_rows + DMMV + MMVQ q8_1 kernels in quantized.cu; wires Iq2Xxs into
candle-core CUDA dispatch; extends deepseek4 placement/loader so routed experts target CUDA.
Verified by candle-core unit test (DMMV = exact CPU dequant; MMVQ tracks it within Q8_1).
DeepSeek-V4.1 runs with 53/10880 IQ2 experts resident on the RTX via CUDA (GPU util ~47%).

### 2. 3683dcb — Lower auto VRAM expert cache residency floor (5% -> 1.56%)
AUTO_VRAM_CACHE_MIN_RESIDENCY (1,20)->(1,64). The old 5%-of-201GB floor demanded a ~10 GB
cache before --expert-placement auto engaged one, leaving the idle RTX VRAM unused.

### 3. 8c551df — Raise CACHE_MEMORY_FRACTION 0.5 -> 0.75 for RAM hot-expert pinning
Auto hot-expert RAM budget (auto_expert_budget) now pins ~2346 of 10880 experts
(MADV_WILLNEED) up from ~1600. Measured on kleya: pinning toward 2800 experts cut decode
page-in from 68.6 GB to 11.7 GB (prefill 2->4-5 t/s, decode >=3.2 t/s). Caveat: full-model
runtime A/B is confounded by OS page-cache drift; figures are directional, mechanism + budget
math are unit-verified.

## Verified
- placement 38, residency 27, hot_experts 9, iq2xxs 11, candle-core CUDA IQ2 1  -> all pass.
- OpenCL build + DeepSeek-V4.1 on the Intel Arc B50 via OpenCL runs (rc=0) — the second GPU is viable.

## Not implemented (identified)
- Two-device expert pool (dispatch each routed expert to RTX via CUDA OR Arc via OpenCL): a
  large refactor of the single-pool_device model; per measurements the marginal VRAM gain beyond
  a few GB is smaller than the RAM-pin lever already done.
- Weight-set shrinkage (coarser/compressed experts) to shrink the 201 GB working set.
