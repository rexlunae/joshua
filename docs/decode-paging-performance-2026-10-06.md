# Decode under an 8 GiB memory limit

Measurements ran on Kleya: Ryzen 9 9950X, Samsung 9100 PRO NVMe,
Linux, CPU execution, cgroup v2 MemoryMax=8G and MemorySwapMax=0.
The model was DeepSeek-V4-Flash-0731-reap-150b-Q2_K (62.4 GB), not a
small substitute. Generation used a 15-token prompt, 16 greedy output
tokens, context 128, and two consecutive completions through the public
Engine. Separate model copies allowed cold file-cache preparation without
dropping the machine's global cache. Original model files were unchanged.

The unchanged baseline decoded at 0.418 and 0.408 tokens/s. With the
candidate's selected-range readahead and a 4 GiB resident weight budget,
it decoded at 0.556 and 0.553 tokens/s: 1.33–1.35 times the throughput.
Completion disk reads fell from 138.85/134.41 GB to 101.85/100.71 GB.
Those disk counters include prefill and, on the first completion, lazy
model loading. Generated text was identical for both completions. The
engine's ordinary session pool remains enabled; these are not fresh-KV
microbenchmarks. VmLck confirmed the actual 4 GiB resident subset.

The mechanism is to protect a bounded subset of non-routed weights that
are revisited every token, while preserving compact mmap-backed experts.
Whole pages are selected in physical file order, overlapping pages count
once, and untied input embeddings are excluded. Selected tensor ranges
switch from random-access advice to normal readahead before WillNeed;
unselected experts retain random-access advice. No matmul arithmetic or
routing is changed.

A 6 GiB requested budget initially accelerated one completion but failed
admission on the second. The final implementation therefore caps bounded
locking to leave 3 GiB of detected available RAM: the existing 1.5 GiB
admission floor plus another 1.5 GiB for lazy model allocation and working
state. This is conservative headroom, not a guarantee for every model or
context. Larger contexts may need smaller explicit budgets. The admission
floor is unchanged. A final 6 GiB request was capped to 5,278,937,088
locked bytes; both consecutive 16-token completions succeeded and matched
the baseline text. That check overlapped test execution and is used only
for reserve and output validation, not as a throughput measurement. Legacy unbounded locking remains unchanged when no
budget is supplied. Linux memory detection now respects cgroup v2 limits
and their ancestors instead of treating host RAM as process capacity.

Usage: pass `--mlock-hot-weights required --mlock-weight-budget 4096`
(the budget is in MiB). The OS memlock limit must permit the actual selected
pages. The option is explicit; it does not automatically change placement.

Rejected experiments included whole-model population, serial page
population, and speculative expert prefetch depth changes: they did not
produce a reliable end-to-end improvement. Parallel page population added
complexity for a small gain and is not included.

Validation includes page-budget planning, tied embedding handling, mapping
boundaries, cgroup ancestor accounting, CLI argument validation, CPU model
parity suites, and five supported OpenCL model parity tests on Kleya.
Earlier raw-forward experiments also compared full float outputs byte for
byte. The reported throughput comparison used an unchanged production
baseline and otherwise matching engine options. Accelerator throughput,
other RAM limits, long contexts, NUMA placement, and general optimality
remain unmeasured; the measured result is specific to this CPU workload.
