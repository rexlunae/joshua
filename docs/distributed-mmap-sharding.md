# Lazy mmap for sharded weights (#90)

## Corrected design

Issue #90 first proposed `Mmap::new_offset`: map only a node's contiguous
fraction of the GGUF file. That does not match how tensor-parallel shards are
laid out. A GGUF matrix is stored row by row (`[output, input]` after
`read_header` reverses the dimensions), and an input-column shard owns a slice
of **every** output row, not one contiguous extent of the tensor. Mapping
"the node's range" would either miss rows or include the other ranks' columns.

The shipped design (`src/distributed/shard.rs`) is:

1. Every node keeps a full local copy of the GGUF file (no shared NFS page
   cache; see RFC #86).
2. `ShardedTensor::map_file` maps the whole file read-only, without
   `MAP_POPULATE` or a whole-file `MADV_WILLNEED`. Virtual address space is
   cheap; physical memory is what matters.
3. A shard computes one absolute byte range per output row
   (`row_ranges()`): `tensor_start + row * row_bytes + local_offset`, of
   `local_row_bytes` bytes, block aligned.
4. `forward` dereferences only those ranges; `prefetch` issues
   `MADV_WILLNEED` only on those ranges. Off-shard columns are never touched,
   so the kernel never faults them in (beyond readahead, below).

`Mmap::new_offset` is therefore deliberately not implemented. DeepSeek V4's
distributed experts use the same type through `with_input_range`.

## Criteria and how they are verified

| Criterion | Evidence |
| --- | --- |
| Mapping a shard reads no tensor bytes | `tests/shard_residency_tests.rs`: after `map_file`, `mincore` shows 0 resident tensor pages and smaps `Rss` is 0 |
| Running a shard makes its own bytes resident, not the whole tensor | same test, after `forward` on rank 1 of 4 |
| `prefetch` reads only the shard's ranges | same test, after `prefetch` |
| Without readahead, off-shard residency is ~0 | same test with `MADV_RANDOM` |
| Split (multi-file) models: shards are correct and read only their part | `split_gguf_shards_match_reference_and_touch_only_their_part` |
| Shard sums equal the unsharded product | both tests, plus unit tests in `shard.rs` |

The residency test writes a 33 MB Q8_0 GGUF (8 rows of 3,932,160 columns,
~4 MiB per row) to `CARGO_TARGET_TMPDIR` (disk backed), `fsync`s it, evicts it
with `posix_fadvise(DONTNEED)` before each phase, maps it through the real
`map_file` path and classifies every page of the tensor as shard (overlaps one
of the rank's row ranges) or off-shard. Measured on Linux 6.18, ext4,
`read_ahead_kb = 128`:

| Phase | Shard resident | Off-shard resident | smaps `Rss` |
| --- | --- | --- | --- |
| after `map_file` | 0 / 8.00 MiB | 0 / 23.87 MiB | 0 |
| after `forward` | 8.00 / 8.00 MiB | 1.97 / 23.87 MiB (8.3%) | 8.75 MiB |
| after `prefetch` | 8.00 / 8.00 MiB | 0 / 23.87 MiB | 0 (cached, not yet mapped) |
| `MADV_RANDOM` + `forward` | 8.00 / 8.00 MiB | 0 / 23.87 MiB | 8.00 MiB |

The off-shard pages after a plain `forward` are kernel readahead past the end
of each row's shard (about two 128 KiB windows per row) plus fault-around.
They are page cache, mostly not mapped into the process: `Rss` exceeds the
shard by 0.75 MiB. The test bounds them by the backing device's
`read_ahead_kb`, so it stays meaningful if readahead is tuned, and checks the
strict zero-off-shard case with `MADV_RANDOM`. A store whose page cache cannot
be evicted (tmpfs) makes the measurement meaningless; the test detects that and
skips with a message instead of passing vacuously.

## Split GGUF files

Joshua has no automatic loader for llama.cpp `gguf-split` sets
(`*-00001-of-0000N.gguf`); a model is one file. The split test writes two
self-contained parts carrying `split.no`, `split.count` and
`split.tensors.count`, resolves each tensor to its part from the per-file
headers (what a split-aware loader would do), shards it across three ranks and
checks the sum against the dequantized reference. After evicting both parts it
verifies with `mincore` that sharding a tensor from one part leaves the other
part with zero resident pages. Automatic split-set discovery remains out of
scope.

## Not covered

- Memory on a real multi-node cluster or on a production-sized model; the
  measurements above are single-process on a 33 MB fixture.
- `MADV_DONTNEED` on off-node ranges (item 4 of the corrected issue text) is
  not needed: off-node ranges are never faulted in to begin with.
- Remapping on membership change (node join/leave) is a runtime
  reconfiguration problem tracked with repartitioning (#91), not an mmap one.
