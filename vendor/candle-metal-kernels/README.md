# vendored `candle-metal-kernels` (0.11.0 + zero-copy additions)

Fork of `candle-metal-kernels` 0.11.0 (MIT OR Apache-2.0, © Hugging Face),
vendored so Joshua can add offset-aware quantized-kernel entry points without
waiting on upstream.

## The delta

The three quantized entry points bind the weight buffer at index 0 with a zero
offset. To back a whole memory-mapped GGUF with a *single* no-copy Metal
buffer, each tensor must be bound at its own byte offset into that buffer —
Metal's `setBuffer:offset:atIndex:` handles this natively (the shader sees the
buffer base moved by the offset), so **no shader changes** are required.

Added (all backwards-compatible; the original functions are unchanged and
delegate to a private `_impl`):

| function | extra param |
|---|---|
| `call_quantized_matmul_mv_t_zc` | `rhs_offset: usize` (weights) |
| `call_quantized_matmul_mm_t_zc` | `src0_offset: usize` (weights) |
| `call_quantized_get_rows_zc` | `src_offset: usize` (weights) |

Offset alignment: Metal requires `setBuffer:offset:` to be a multiple of the
buffer's alignment (≥ 16 bytes on Apple Silicon compute). GGUF tensor offsets
are 32-byte aligned, so file offsets are always valid.

## Rebasing onto a new upstream version

1. Re-copy the crate from `~/.cargo/registry/src/…/candle-metal-kernels-<ver>/`
   (drop `Cargo.toml.orig`, `Cargo.lock`, `examples/`, `tests/`).
2. Re-apply the three `_zc` functions + `_impl` refactor in
   `src/kernels/quantized.rs` (grep for `_zc` to find every hunk).
3. Drop the `[[example]]` section from `Cargo.toml` (we don't vendor examples).
