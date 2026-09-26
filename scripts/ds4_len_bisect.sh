#!/usr/bin/env bash
# Bisect DeepSeek-V4 long-prompt garbling by prompt length.
#
# Cuts one long prompt down to roughly N tokens for each requested length,
# runs `joshua run` greedily on the CPU, and prints the first generated
# tokens of each run next to the real prompt token count joshua reports.
#
#   scripts/ds4_len_bisect.sh MODEL.gguf PROMPT.txt [N ...]
#
# Lengths default to 1500 2100 2600.  The lightning indexer keeps the top
# 512 compressed blocks of 4 tokens, so it starts dropping blocks past 2048
# tokens: a clean 1500 with garbled 2100/2600 points at the indexer's
# selection, garbling already at 1500 points elsewhere.  The IQ2_XXS reap
# artifact is known to garble past ~85 prompt tokens in llama.cpp too (see
# tests/long_context_acceptance.rs), so add short lengths such as 60 and 120
# to tell that cliff apart from an engine fault, and compare a Q2_K model.
#
# Environment:
#   JOSHUA            joshua binary (default: target/release/joshua, else PATH)
#   MAX_TOKENS        tokens to generate per run (default 32)
#   N_CTX             context size (default 4096)
#   KEEP              head | tail: which end of the prompt to keep when
#                     cutting (default tail, which keeps a trailing question)
#   CHARS_PER_TOKEN   cut estimate (default 3.8); the reported count is exact
#   CHUNKS            prefill chunk sizes to try per length, "default" for
#                     the engine default (default "default"); e.g.
#                     CHUNKS="default 64" separates a batch-size effect
#   DEVICE            joshua --device (default cpu)
#   EXTRA_ARGS        extra joshua run arguments
#   LLAMA_CMD         optional reference command; the cut prompt file path is
#                     appended, e.g.
#                     LLAMA_CMD='llama-cli -m MODEL.gguf --temp 0 -n 32 -st -f'
#   OUT               output directory (default ./ds4-bisect-<timestamp>)
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,/^set /p' "$0" | sed '$d; s/^# \{0,1\}//'
    exit 2
fi

MODEL=$1
PROMPT=$2
shift 2
LENGTHS=("$@")
[ ${#LENGTHS[@]} -eq 0 ] && LENGTHS=(1500 2100 2600)

if [ -z "${JOSHUA:-}" ]; then
    if [ -x target/release/joshua ]; then JOSHUA=target/release/joshua; else JOSHUA=joshua; fi
fi
MAX_TOKENS=${MAX_TOKENS:-32}
N_CTX=${N_CTX:-4096}
KEEP=${KEEP:-tail}
CHARS_PER_TOKEN=${CHARS_PER_TOKEN:-3.8}
CHUNKS=${CHUNKS:-default}
DEVICE=${DEVICE:-cpu}
OUT=${OUT:-./ds4-bisect-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"

echo "joshua:  $($JOSHUA --version 2>/dev/null || echo "$JOSHUA")"
echo "commit:  $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "model:   $MODEL"
echo "prompt:  $PROMPT ($(wc -c <"$PROMPT") bytes)"
echo "output:  $OUT"
echo

# Cut the prompt to about N tokens at a whitespace boundary.
cut_prompt() {
    python3 - "$PROMPT" "$1" "$KEEP" "$CHARS_PER_TOKEN" <<'PY'
import sys
path, n, keep, cpt = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4])
text = open(path, encoding="utf-8", errors="replace").read()
chars = int(n * cpt)
if len(text) > chars:
    if keep == "head":
        text = text[:chars]
        cut = text.rfind(" ")
        text = text[:cut] if cut > chars // 2 else text
    else:
        text = text[-chars:]
        cut = text.find(" ")
        text = text[cut + 1:] if 0 <= cut < chars // 2 else text
sys.stdout.write(text)
PY
}

summary="$OUT/summary.txt"
: >"$summary"
for n in "${LENGTHS[@]}"; do
    cut_file="$OUT/prompt-$n.txt"
    cut_prompt "$n" >"$cut_file"
    for chunk in $CHUNKS; do
        tag="n$n-chunk$chunk"
        chunk_args=()
        [ "$chunk" != "default" ] && chunk_args=(--prefill-chunk "$chunk")
        echo "=== ~$n tokens, prefill chunk $chunk ==="
        start=$(date +%s)
        set +e
        # shellcheck disable=SC2086
        "$JOSHUA" run -m "$MODEL" --device "$DEVICE" --temperature 0 \
            --max-tokens "$MAX_TOKENS" --n-ctx "$N_CTX" ${chunk_args[@]+"${chunk_args[@]}"} \
            ${EXTRA_ARGS:-} "$(cat "$cut_file")" \
            >"$OUT/$tag.out" 2>"$OUT/$tag.err"
        rc=$?
        set -e
        secs=$(( $(date +%s) - start ))
        counts=$(grep -o 'prompt=[0-9]* completion=[0-9]*' "$OUT/$tag.err" | tail -1 || true)
        text=$(tr '\n' ' ' <"$OUT/$tag.out" | cut -c1-160)
        line="$tag rc=$rc ${secs}s ${counts:-no-token-line} | $text"
        echo "$line"
        if [ $rc -ne 0 ]; then
            grep -m3 -E "^Error|[Ee]rror:" "$OUT/$tag.err" || tail -5 "$OUT/$tag.err"
        fi
        echo "$line" >>"$summary"
        echo
    done
    if [ -n "${LLAMA_CMD:-}" ]; then
        echo "=== ~$n tokens, reference ($LLAMA_CMD) ==="
        set +e
        # shellcheck disable=SC2086
        $LLAMA_CMD "$cut_file" >"$OUT/n$n-ref.out" 2>"$OUT/n$n-ref.err" </dev/null
        set -e
        text=$(tr '\n' ' ' <"$OUT/n$n-ref.out" | tail -c 400)
        echo "n$n-ref | $text" | tee -a "$summary"
        echo
    fi
done

echo "=== summary ($summary) ==="
cat "$summary"
