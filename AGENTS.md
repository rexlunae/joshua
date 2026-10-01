# Agent guidance for Joshua

This repository is a Rust inference engine. Performance work must preserve
model outputs, bounded memory use, and the distinction between CPU, mmap, and
accelerator execution. Follow the user's requested scope and keep unrelated
changes out of the patch.

## Find the cost before changing the code

- Trace the full hot path for the workload in question: load, prefill, decode,
  batched decode, or speculative verification. A faster helper may have no
  measurable effect on end-to-end latency.
- Look for work repeated per token, layer, expert, or request that depends only
  on model weights or configuration. Compute immutable values at load time;
  reuse scratch buffers when their lifetime and concurrency rules allow it.

## Avoid unnecessary copies and synchronization

- Treat `Tensor::to_vec*`, `to_device`, `contiguous`, `Tensor::cat`, and repeated
  `index_select` as possible allocation or transfer points in hot paths. Check
  whether a borrowed tensor, view, direct gather, or reusable buffer can do
  the same job. Do not remove a `contiguous` call until the consuming kernel's
  layout requirements are understood.
- Keep IDs, activations, and routing intermediates on the device when the next
  operation runs there. A device-to-host read can synchronize queued work;
  copying the result back immediately is especially costly. Use
  `moe::on_device` when a transfer is genuinely needed so same-device tensors
  remain borrowed.
- Keep quantized and mmap-backed weights in their compact, borrowed form when
  possible. Avoid materializing a whole expert pool as f32 or making a private
  model copy per session. Preserve the streamed-load fallback and backend
  capability checks.
- Avoid gathering or reordering an entire cache when the next operation needs
  only selected rows. Preserve token order, masks, and the behavior at window
  wraparound and chunk boundaries.
- Traverse hot state and weight buffers in contiguous storage order. When
  changing loop order for cache locality or vectorization, preserve each
  output's accumulation order and verify the state carried across tokens.
- Reserve vectors when the final size is known. Avoid per-expert temporary
  tensors and host allocations for a single-row decode when direct use of the
  input row is valid. Keep duplicate routes and batched paths correct.

## Share code without obscuring behavior

- Check existing shared modules before adding a loader-specific copy:
  `moe`, `mhc`, `attention`, `token_embedding`, `raw_block`, and
  `stream_prefill`. Put genuinely common behavior there; keep architecture
  differences explicit at the call site.
- Preserve tensor shapes, dtypes, device placement, expert order, and
  floating-point accumulation order where possible. Small rounding changes
  need parity checks; they are not a reason to loosen tolerances without
  evidence.
- Prefer a focused change with a clear performance mechanism over a broad
  refactor. Avoid whole-file formatting churn in files that are not already
  consistently formatted.

## Verify and report

- Run focused tests for each affected architecture and path, including prefill
  versus decode, cache continuation, batched routing, and speculative all-row
  logits where relevant. Use the tiny GGUF fixtures in `tests/common` for
  output parity and a supported accelerator test when changing device code.
- Run the relevant library/integration suite and build checks for affected
  features or targets. A failing test should be reproduced on the unchanged
  base before calling it pre-existing; document the exact failure rather than
  silently skipping it.
- Report what was measured, on which backend and input size, and what remains
  unmeasured. Keep the original checkout and user changes intact when working
  in a separate branch or worktree.
