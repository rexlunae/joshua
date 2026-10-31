# GGUF versus HF source parity

`scripts/gguf_source_parity.py` is a read-only CPU diagnostic for selected
GGUF tensors versus an indexed, local Hugging Face safetensors source. It is
intentionally separate from Joshua's inference engine: no model construction,
accelerator, PyTorch, tokenizer, conversion, or model download is involved.
GGUF dequantization uses `gguf-py`; HF decoding is implemented independently in
NumPy so the converter's FP4 unpacking is not reused as the reference.

## Setup and tests

Python 3.10+ and the explicit dependencies below are required. Run from the
repository root; use a virtual environment rather than changing system Python.

```bash
python3 -m venv /tmp/joshua-parity-venv
/tmp/joshua-parity-venv/bin/python -m pip install \
  -r scripts/gguf_source_parity_requirements.txt
/tmp/joshua-parity-venv/bin/python -m unittest discover \
  -s scripts -p 'test_gguf_source_parity.py' -v
```

Tests write independent tiny GGUF/safetensors fixtures for F32/F16/BF16,
FP8 E4M3FN/E8M0 (including tail blocks and per-row scales), and low-first FP4
E2M1/E8M0. They cover Q8_0, a known Q4_K block, expert selection, swapped-column
detection, invalid shapes/scales, truncated files, non-finite values, chunk-size
invariance, bounded read sizes, and a sparse multi-GB table at row 12,345,678.
The `GGUF source parity diagnostic` CI job runs these tests without real models.

## Explicit selections

Names are explicit, not guessed from architecture. Repeat `--tensor GGUF=HF`
for tensors sharing the same selection options. GGUF dimensions are reversed
into normal row/column order; a vector is one row. For an expert pool stored as
GGUF `[input, output, expert]`, `--expert` is required and selects just that slab.
Use a separate invocation for another expert, row interval, or scale layout.
Selections must fit both tensors; rows are never silently clipped.

```bash
PYTHON=/tmp/joshua-parity-venv/bin/python
GGUF=/tank/share/models/DSV41-REAP272E-balanced.gguf
SOURCE=/tank/share/models/DeepSeek-V4.1-Flash-REAP-272E

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 19 ionice -c3 \
  "$PYTHON" scripts/gguf_source_parity.py --gguf "$GGUF" --source "$SOURCE" \
  --tensor blk.0.ffn_gate_exps.weight=layers.0.ffn.experts.0.w1.weight \
  --tensor blk.0.ffn_up_exps.weight=layers.0.ffn.experts.0.w3.weight \
  --tensor blk.0.ffn_down_exps.weight=layers.0.ffn.experts.0.w2.weight \
  --expert 0 --fp4 --rows 64 --chunk-rows 8 > experts-l0.jsonl

"$PYTHON" scripts/gguf_source_parity.py --gguf "$GGUF" --source "$SOURCE" \
  --tensor blk.0.ffn_gate_inp.weight=layers.0.ffn.gate.weight \
  --tensor blk.0.attn_q_a.weight=layers.0.attn.wq_a.weight \
  --tensor blk.0.hc_attn_fn.weight=layers.0.hc_attn_fn \
  --tensor blk.1.engram_wkv.weight=layers.1.engram.wkv.weight \
  --rows 24 --chunk-rows 8 > controls.jsonl

"$PYTHON" scripts/gguf_source_parity.py --gguf "$GGUF" --source "$SOURCE" \
  --tensor blk.1.engram_embd.weight=layers.1.engram.embed.weight \
  --fp8-block-rows 1 --row-start 12345678 --rows 4 --chunk-rows 2 > engram.jsonl
```

`ionice` is Linux-only; omit it on macOS. On shared machines, inspect model paths
first and use a unique scratch directory (`mktemp -d`) for the environment and
reports. Never read an actively regenerating GGUF as a completed model or
overwrite the original to test a converter fix.

Supported HF weights are F32, F16, BF16, `F8_E4M3` (E4M3FN), and explicitly
opted-in `I8` packed FP4. `--fp4` means the **low nibble is element 2i**, the
high nibble is element 2i+1, and each group of 32 decoded columns has an E8M0
scale. Arbitrary I8 tensors are rejected without this opt-in. FP8 defaults to
32x32 E8M0 scale blocks; override `--fp8-block-rows` and `--fp8-block-cols`
explicitly for another layout (the V4.1 engram embedding uses **1x32**, not
32x32). Scale shapes must exactly match the ceiling-divided weight shape;
block sizes are not inferred from tail dimensions. `--scale NAME` overrides
the default replacement of the final `weight` suffix with `scale`.
Other scale encodings, transposes, fused tensors, source tensors of rank >2,
and GGUF tensors of rank >3 are deliberately unsupported. GGUF quantization
support follows `gguf-py`'s CPU dequantizers; unsupported types fail explicitly.

## Output and memory bounds

Each mapping emits one JSON object per line, including names, dtypes, selected
expert, full decoded shape, and the exact sampled row interval. `direct` and
`adjacent_swapped` contain:

- `relative_l2`: `norm(GGUF - HF) / norm(HF)`;
- `cosine`: uncentered normalized dot product;
- `pearson`: centered correlation across the selected elements;
- `rmse`, `max_abs`, `finite_pairs`, and `nonfinite_pairs`.

`adjacent_swapped` compares the same GGUF rows to HF columns reordered
`[1, 0, 3, 2, ...]`, independently within each row. For packed FP4 this is
equivalent to reversing nibble order, since adjacent pairs share a scale.
For odd column counts it is `null`. Undefined metrics (zero norm/variance or
no finite pairs) are `null`, never JSON NaN. Non-finite pairs are counted and
excluded from numeric metrics; **any such pairs make the command fail**.
Exit status 1 also reports missing tensors, malformed/unsupported input or
invalid selections; later mappings still run when one mapping fails. Errors
opening model headers/indexes go to stderr. Status 0 means all requested
comparisons were computed on finite pairs, **not** that a model passed parity.
No universal quantization-error threshold or automatic root-cause verdict is
imposed. Sampled agreement alone cannot certify all rows or coherent decoding.

Default selection is 64 rows from row 0, chunked in batches of at most 16.
`--max-elements` (default 1,048,576) further caps decoded elements per batch;
a single row exceeding the cap is rejected before reading weight data.
Only the selected GGUF rows, source rows and their scale blocks are read via
seek/read, with no whole-model mmap or deserialization. Statistics accumulate
across chunks in float64; temporary NumPy/dequantizer buffers use a constant
multiple of the batch size. This is not a hard process RSS limit.
Header/index inputs are capped at 64 MiB each; the GGUF tensor directory, HF
index and at most two shard headers are retained (Python object overhead is
additional). Memory does not grow with an engram table's row count or with the
number of rows requested. Input files must remain immutable during a run.

## Issue #175: sampled evidence, not end-to-end validation

On 2026-10-30 (America/Los_Angeles), read-only sampling on **kleya CPU** used
the original `/tank/share/models/DSV41-REAP272E-balanced.gguf` and the HF source
directory above, Python 3.14.4, NumPy 2.3.5 and gguf 0.19.0. Data batches were
at most eight rows (two for the engram table), with single-thread environment
settings and `nice -n 19 ionice -c3`. All sampled pairs were finite.

| GGUF tensor | Expert; row interval | Direct relL2 | Direct cosine | Swapped relL2 | Swapped cosine |
|---|---|---:|---:|---:|---:|
| `blk.0.ffn_gate_exps.weight` | 0; [0,64) | 1.40756 | 0.00210 | 0.07438 | 0.99724 |
| `blk.0.ffn_up_exps.weight` | 0; [0,64) | 1.41059 | -0.00188 | 0.07460 | 0.99722 |
| `blk.0.ffn_down_exps.weight` | 0; [0,64) | 1.40617 | 0.00464 | 0.07423 | 0.99725 |
| `blk.20.ffn_gate_exps.weight` | 271; [100,132) | 1.40430 | 0.00682 | 0.07459 | 0.99722 |
| `blk.20.ffn_down_exps.weight` | 271; [100,132) | 1.41160 | -0.00352 | 0.07474 | 0.99721 |
| `blk.0.ffn_gate_inp.weight` | none; [0,24) | 6.10e-9 | 1.00000 | 1.41628 | -0.00293 |
| `blk.0.attn_q_a.weight` | none; [0,24) | 0.00541 | 0.99999 | 1.41332 | 0.00156 |
| `blk.0.hc_attn_fn.weight` | none; [0,24) | 0.30674 | 0.95361 | 1.40548 | 0.02458 |
| `blk.1.engram_wkv.weight` | none; [0,24) | 0.00545 | 0.99999 | 1.40358 | 0.01542 |
| `blk.1.engram_embd.weight` | none; [12345678,12345682) | 0.28625 | 0.95946 | 1.40724 | 0.01924 |

To reproduce the layer-20 selection, use the expert command with layer 20,
expert 271, HF `experts.271`, `--row-start 100 --rows 32`, and gate/down
projections. The other selections are shown above. These samples reproduce
the adjacent-column inversion in the original routed experts and separately
show quantization loss in Q2_K mixers/engram. They do not establish the cause
of every possible inference failure.

Converter PR `rexlunae/ds4-convert#1` has merged with the FP4 nibble fix.
The corrected GGUF was still actively regenerating at
`/home/tserica/models/DSV41-REAP272E-balanced-fp4fix.gguf` during this sampling;
it was **not** used, interrupted, or replaced. Original `/tank` files were
untouched. After regeneration completes, rerun identical selections against
that new path, then separately validate Joshua's raw/chat prompt decoding and
relevant backends end to end. No performance measurements, corrected-model
parity, full-model coverage, or issue-closure claim is made here. #175 remains
open pending that validation; this tool changes no loader or inference path.
