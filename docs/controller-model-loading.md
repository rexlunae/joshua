# Loading models from a controller

Workers started with `joshua worker` hold no model. A controller
(`joshua serve --model-dir DIR --worker ADDR,...`) places a model on them when
`POST /v1/models/load` asks for it, and releases them on
`POST /v1/models/unload`. This builds on the static Qwen3 layer pipeline
(#158, [pipeline-validation.md](pipeline-validation.md)).

## Protocol

Each controller connection is authenticated with HMAC-SHA256 under
`JOSHUA_CLUSTER_KEY` and bound to random values from both ends, so a recorded
connection cannot be replayed. Authentication does not encrypt weights,
prompts or activations; use a private network or a tunnel.

1. The controller asks each worker for its free memory (`Probe`/`Capacity`).
2. It plans contiguous layer ranges: explicit `ends`, or weight bytes split in
   proportion to free memory with at least one layer per worker. Each stage's
   planning reservation (weights, KV, workspace) must fit that worker's free
   memory, so an impossible placement fails before any transfer.
3. It sends each worker its plan and stage (`Assign`). The worker answers
   whether it already has that stage cached (`Ready`).
4. Otherwise the controller streams a self-contained GGUF holding all
   metadata and only the stage's tensors (first stage: embedding; last stage:
   output norm and head), in 1 MiB frames, from all workers in parallel, then
   its SHA-256 (`Commit`). Neither side buffers the slice in memory.
5. The worker checks length and checksum, writes the slice to disk, maps it
   (or copies with `--no-mmap`), checks it against the plan, and replies.

Afterwards the connection carries the ordinary pipeline protocol. The
controller's `Engine` serves the model through a remote backend: one
pipeline session per generation session, so chat templates, sampling,
streaming, stop handling and multi-turn KV reuse are unchanged. Backend
failures return errors; the controller never falls back to loading the
weights locally. Embeddings are refused for such models.

A worker serves one controller at a time. When the controller stops (unload)
or disconnects, the worker drops the stage and waits for the next controller.

## Cache

With `--cache-dir`, a slice is kept as `<slice id>.gguf` plus its checksum.
The slice id hashes the full model's SHA-256 and the stage range, and the
slice writer is deterministic, so a later load of the same stage of the same
model reuses the file after re-hashing it; a damaged file is fetched again.
Without `--cache-dir` the slice lives in the temporary directory and is
deleted on unload.

## Limits

- CPU Qwen3 dense models only (the pipeline's current scope).
- Concurrent sessions share one pipeline and run one batch at a time;
  requests are not yet batched across sessions.
- A lost worker fails the model; unload and load it again.
- Validated with in-process and separate-process loopback workers only; no
  multi-host or bandwidth measurements yet.
