#!/usr/bin/env bash
# Validate the opt-in SYCL decode megakernel on a real device.
#
# The host emulation in vendor/candle-core/src/sycl_backend/emu uses stronger
# memory ordering than a GPU, so it cannot prove that the persistent kernel's
# ticket / completion-counter synchronization is visible across work-groups.
# This script builds the bridge with a SYCL compiler and runs the parity suite
# (prefill, KV-cache growth, a multi-token chunk and a rollback) at several
# work-group counts, then optionally against a real GGUF.
#
# Usage:
#   scripts/sycl-megakernel-hw-check.sh
#   JOSHUA_MEGAKERNEL_GGUF=/models/qwen3-0.6b.gguf scripts/sycl-megakernel-hw-check.sh
#
# Environment:
#   CXX                     SYCL compiler for the bridge (default: icpx)
#   JOSHUA_SYCL_LIBRARY     use an existing bridge instead of building one
#   JOSHUA_MEGAKERNEL_GGUF  real model for the real-shape parity check
#   GROUPS                  work-group counts to try (default: "64 256 0")
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if [[ -z "${JOSHUA_SYCL_LIBRARY:-}" ]]; then
    cxx="${CXX:-icpx}"
    if ! command -v "$cxx" >/dev/null; then
        echo "error: $cxx not found (set CXX or JOSHUA_SYCL_LIBRARY)" >&2
        exit 1
    fi
    build="$root/target/sycl-bridge"
    cmake -S vendor/candle-core/src/sycl_backend -B "$build" \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$cxx" >/dev/null
    cmake --build "$build" -j >/dev/null
    export JOSHUA_SYCL_LIBRARY="$build/libjoshua_sycl.so"
fi
echo "bridge: $JOSHUA_SYCL_LIBRARY"

if command -v sycl-ls >/dev/null; then
    sycl-ls
fi

for groups in ${GROUPS:-64 256 0}; do
    echo "== JOSHUA_MEGAKERNEL_GROUPS=$groups =="
    JOSHUA_MEGAKERNEL=1 JOSHUA_MEGAKERNEL_GROUPS="$groups" \
        cargo test --release --features sycl --test sycl_megakernel_tests -- \
        --nocapture --test-threads=1
done

if [[ -n "${JOSHUA_MEGAKERNEL_GGUF:-}" ]]; then
    echo "== real model: $JOSHUA_MEGAKERNEL_GGUF =="
    JOSHUA_MEGAKERNEL=1 cargo test --release --features sycl \
        --test sycl_megakernel_tests -- --nocapture --test-threads=1
fi

echo "all megakernel checks passed"
