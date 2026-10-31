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
Quantized vectors use a single scale row. Pearson accumulation uses merged
centered moments relative to fixed origins, keeping constant inputs undefined
and avoiding raw-moment cancellation for large offsets across chunk sizes.
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
At the time of the October 30 sampling, the corrected GGUF was still regenerating at
`/home/tserica/models/DSV41-REAP272E-balanced-fp4fix.gguf` during this sampling;
it was **not** used, interrupted, or replaced. Original `/tank` files were
untouched. After regeneration completes, rerun identical selections against
that new path, then separately validate Joshua's raw/chat prompt decoding and
relevant backends end to end. No performance measurements, corrected-model
parity, full-model coverage, or issue-closure claim is made here. #175 remains
open pending that validation; this tool changes no loader or inference path.

### October 31: corrected artifact and CPU generation

The separate regeneration completed on kleya with converter exit status 0.
`/home/tserica/ds4-fp4fix-run.log` reports all 1,046 tensors written. A read-only
header/directory audit of the corrected path found 1,046 tensors and checked
every tensor's encoded extent against the 290,070,476,416-byte file. The final
tensor ends exactly at EOF. This checks structural completeness, not a full-file
checksum or every tensor's numerical fidelity. Original `/tank` files remain
untouched.

Rerunning the exact selections above against the corrected path on **kleya CPU**
produced the following finite results. The Python environment and row/chunk
bounds were unchanged from the October 30 diagnostic. The final rerun used the
tool from `99d2ddb` (SHA-256
`2345d574ebc4cbfe782eb679ab3c425d765733b75c0e3d57a97592c283095ff3`);
all 14 diagnostic unit tests passed in that environment. The table's five rows
were checked against its JSONL output, including the exact selections and
rounded metrics.

| GGUF tensor | Expert; row interval | Direct relL2 | Direct cosine | Swapped cosine |
|---|---|---:|---:|---:|
| `blk.0.ffn_gate_exps.weight` | 0; [0,64) | 0.07439 | 0.99724 | 0.00210 |
| `blk.0.ffn_up_exps.weight` | 0; [0,64) | 0.07460 | 0.99722 | -0.00188 |
| `blk.0.ffn_down_exps.weight` | 0; [0,64) | 0.07423 | 0.99725 | 0.00464 |
| `blk.20.ffn_gate_exps.weight` | 271; [100,132) | 0.07460 | 0.99722 | 0.00684 |
| `blk.20.ffn_down_exps.weight` | 271; [100,132) | 0.07474 | 0.99721 | -0.00352 |

The expert samples now favor direct rather than adjacent-swapped columns.
Control/engram samples remained finite: direct cosines were 1.00000 for the
router, 0.99999 for query A and engram WKV, 0.95361 for the Q2_K hyperconnection
mixer and 0.95946 for the sampled engram embedding. The latter quantization
losses did not disappear with the expert nibble fix.

A default-feature CPU release binary built from Joshua
`99d2ddbbb71239d810b4f00c1699ea89abe2cfa7` on erskin was copied to a separate
temporary path on kleya. The existing CPU and CUDA servers were left running.
The following greedy chat run used the corrected file and its embedded template:

```sh
systemd-run --user --scope --quiet -p MemoryMax=16G -p MemorySwapMax=0 \
  timeout --signal=TERM --kill-after=10s 900s env RUST_LOG=info \
  /usr/bin/time -v nice -n19 ionice -c3 /tmp/joshua-175-main-99d2ddb run \
  --model /home/tserica/models/DSV41-REAP272E-balanced-fp4fix.gguf \
  --device cpu --temperature 0 --max-tokens 80 --n-ctx 128 --lazy-weights \
  --prefetch-model=false --pin-hot-weights=false --expert-cache 0 \
  --ssd-expert-cache 0 --engram-tier disk \
  'Explain why the sky looks blue in three short sentences.'
```

It exited normally with **15 prompt tokens and 77 completion tokens**, beyond
the historical step-64 failure boundary, and produced three coherent numbered
sentences about shorter-wavelength scattering and sunset colors. No non-finite
logit error or repetitive whitespace-only output occurred in this run. A prior
eight-token exploratory run with the existing October 29 binary answered
`What is 2 + 2?` with `The answer is **4**.`; that shorter run is not evidence
for current-main long-decode behavior.

The current-main run took 4:37.88 wall time, with maximum process RSS
19,940,456 KiB, 438,591 major page faults and zero swaps reported by `time -v`.
**The 16-GiB cgroup setting limits charged memory, not process RSS including
shared file-backed pages.** Do not report it as a 16-GiB RSS bound. This
low-priority, I/O-heavy run on a shared machine is correctness evidence, not a
throughput comparison or speedup claim.

A CPU raw-prompt check used the existing `debug_dsv4` example, built from the
same `99d2ddb` source, and the source directory's `tokenizer.json`:

```sh
debug_dsv4 /home/tserica/models/DSV41-REAP272E-balanced-fp4fix.gguf \
  /tank/share/models/DeepSeek-V4.1-Flash-REAP-272E/tokenizer.json \
  'The capital of France is' 8
```

The five-token prompt produced ` Paris`, `.`, ` \n`, then EOS (token 1 at
absolute position 8). All eight displayed decode steps had finite top-five
logits and log-sum-exp summaries, and the process exited 0. This diagnostic
example deliberately keeps decoding after EOS until its requested step count;
the four later tokens are not part of an ordinary EOS-terminated completion.
The run used a separate 12-GiB charged-memory/no-swap scope and 600-second
deadline; peak process RSS was 15,083,908 KiB and wall time 1:23.29. It overlapped
another validation run, so its timing must not be compared to the chat run as
a performance result.

An OpenCL-enabled build of the same Joshua source also completed the 15-token
prompt with 77 generated tokens and exit 0, using the chat command above with
`--device opencl --dense-placement cpu --vram-budget 2048
--vram-expert-cache 512`. Its completion text matched the CPU run byte for byte.
However, the logs explicitly identify **dense placement on CPU and the expert
cache on `Cpu`**, even though an NVIDIA RTX 5060 Ti OpenCL device was opened.
This is a successful CPU-placement/fallback configuration check, **not evidence
that OpenCL kernels or GPU expert-cache dispatch generated the tokens**. It
cannot satisfy accelerator acceptance for #175 or #62. No GPU kernel activity,
cache hit count or device-memory peak was measured. Its 6:39.13 wall time and
12,455,656-KiB maximum process RSS must not be used for a CPU-versus-GPU speed
comparison; it overlapped the raw-prompt check and used a different placement
and cache configuration.

Raw JSONL samples, full generated text, command/timing output and exit statuses
are retained on kleya at `/tmp/joshua-175-corrected.7YpNYzeQ` (final diagnostic
reruns and structural audit are in `current-tool/`); temporary paths
are not durable CI fixtures. This validates selected corrected REAP weights and
CPU chat generation, one short raw-prompt diagnostic and an OpenCL-selected
CPU-placement configuration only. Longer/context-window coverage, actual
accelerator generation, and the separate REF41 artifact remain
unvalidated here. Keep #175 open for those remaining questions.
