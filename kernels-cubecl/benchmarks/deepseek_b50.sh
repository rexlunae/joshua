#!/usr/bin/env bash
# Native Vulkan A/B decode, identical tokens, compact mmap-backed CPU experts.
set -euo pipefail
if (( $# < 3 )); then
    echo "usage: $0 MODEL.gguf TOKENIZER.json OUTPUT_DIR [benchmark options]" >&2
    exit 2
fi
model=$(realpath "$1")
tokenizer=$(realpath "$2")
output=$(realpath -m "$3")
shift 3
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
for kernel in generic cubecl; do
    if [[ -e "$output/$kernel/summary.json" ]]; then
        echo "refusing to mix results with existing $output/$kernel/summary.json" >&2
        exit 2
    fi
done
candidate=${JOSHUA_BENCH_CANDIDATE:-cubecl}
case "$candidate" in
    cubecl|cubecl-next|production) ;;
    *) echo "JOSHUA_BENCH_CANDIDATE must be cubecl, cubecl-next or production" >&2; exit 2 ;;
esac
mkdir -p "$output"
cargo build --release --features vulkan --example bench_cubecl_real
for kernel in generic cubecl; do
    selection=$kernel
    reference=()
    if [[ "$kernel" == cubecl ]]; then
        selection=$candidate
        reference=(--reference "$output/generic/trace.json")
    fi
    target/release/examples/bench_cubecl_real \
        --model "$model" --tokenizer "$tokenizer" --kernel "$selection" \
        --output "$output/$kernel" --prompt-tokens 4 --decode 8 --passes 3 \
        --warm-decode 2 "${reference[@]}" "$@" 2>&1 | tee "$output/$kernel.log"
done
python3 kernels-cubecl/benchmarks/compare_real.py \
    "$output/generic" "$output/cubecl" "$output/comparison.json" --require-pass --allow-reduced-memory
