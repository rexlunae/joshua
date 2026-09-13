# kernels-rust — joshua's Rust-native CUDA kernels (cuda-oxide)

## Goal

Author joshua's custom CUDA kernels in **pure Rust** (no `*.cu`/C) and compile
them to PTX with [NVlabs/cuda-oxide](https://github.com/NVlabs/cuda-oxide)
(`#[kernel]` → rustc → LLVM IR → PTX). This is the "write our own GPU kernels
in Rust where possible" track for joshua.

## Scope boundary (why this is a separate crate)

- cuda-oxide needs **CUDA Toolkit 13.0+ / driver R580+**.
- joshua's current acceleration target — a Tesla M40 (Maxwell, **sm_52**) — is
  capped at **CUDA 12.4** (CUDA 13 dropped Maxwell).

So this crate is **never** compiled by the main `cargo build`. It lives here so
the kernel *source* is versioned with joshua, ready to build on a CUDA-13 host.

## What's here

- `src/lib.rs` — a `#[cuda_module]` with an `#[kernel]` for the **IQ2_XXS-style
  dequant+add**, plus a host driver. The numerics mirror joshua's CPU `iq2xxs`
  path so that on capable hardware the same results are computed on-device with
  the dense tensor resident, only activations hopping the PCIe bus.
- This is a **scaffold/spec**: the exact cuda-oxide intrinsic names should be
  verified against `cuda-oxide-book` on the build host.

## Build (on a CUDA-13 host)

```bash
rustup toolchain install nightly-2026-08-28      # pinned in rust-toolchain.toml
rustup component add rustc-dev rust-src llvm-tools --toolchain nightly-2026-08-28
sudo apt install clang-21                        # libclang dev for cuda-bindings

cargo +nightly oxide build                       # compile kernel to PTX
cargo +nightly oxide run dequant_demo            # load + launch + print
cargo +nightly oxide inspect dequant_demo        # dump generated PTX
cargo +nightly oxide sanitize dequant_demo --tool memcheck
```

Requires a CUDA 13.x toolkit (`nvcc --version` reports V13+). `cargo oxide
doctor` validates the toolchain up front.

## Roadmap / natural next steps

1. Fuse the IQ2_XXS **dequant + 8832-wide matmul** into a single `#[kernel]`
   (tile over the up/down dimensions, stream the packed weights once).
2. Add a `#[launch_contract(...)]` for a checked safe launch.
3. Add a Q2_K-down / fused gate-up-down expert kernel, mirroring
   `quantized_deepseek4`'s `Moe::dispatch` phases so the whole expert block
   can run on-device on Hopper/Blackwell machines.
4. Gate the joshua wiring behind a CUDA-13 compile-time cfg so mainline stays
   CUDA-12.4-clean.

## Related

- Design doc for the CPU/GPU split: `docs/device-expert-cache-design.md`
- The CUDA-12.4 path this does NOT replace: joshua's `--device cuda` hybrid
  (dense on device, experts on CPU via vendored candle-kernels patched through
  `rexlunae/candle-kernels`).