# joshua

A pure-Rust LLM inference engine — a Rust clone of [Cactus](https://github.com/cactus-compute/cactus).

No C or C++ dependencies.  CPU inference runs entirely in safe Rust via
[candle](https://github.com/huggingface/candle) (HuggingFace's native Rust ML
framework) and [tokenizers](https://github.com/huggingface/tokenizers).

---

## Features

| Feature | Details |
|---|---|
| **Pure Rust** | Zero C/C++ dependencies — `cargo build` requires only a Rust toolchain |
| **mmap loading** | The GGUF file is memory-mapped like llama.cpp: weights live in the OS page cache, shared across engine clones — and a model that fits in RAM is prefetched whole at load (`MADV_WILLNEED`), so inference never re-reads weights from disk |
| **Huge pages** | Transparent 2 MiB pages (`MADV_HUGEPAGE`, default on Linux) or explicit 2 MiB / 1 GiB (`MAP_HUGETLB`) backing to cut TLB misses on large models |
| **OpenAI-compatible** | Drop-in replacement for `/v1/chat/completions`, `/v1/embeddings`, `/v1/models` |
| **Streaming** | Server-Sent Events (SSE) for token-by-token streaming |
| **GGUF support** | Llama/Mistral/Mixtral, Gemma 1–3, every GLM generation (ChatGLM2/3, GLM-4, GLM-4.1V text, GLM-OCR, GLM-4.5/4.6/4.7 incl. Air and Flash, GLM-5/5.1/5.2, GLM-5.3-Flash), LFM2, Phi-2, Phi-3, every Qwen generation (Qwen 1, Qwen1.5/2/2.5 incl. MoE, Qwen2/2.5-VL and Qwen3-VL text, Qwen3, Qwen3-MoE, Qwen3-Next, Qwen3.5 dense/MoE, Qwen3.8-Flash-Next / `qwen4exp`, Bonsai 1-bit / 2-bit), DeepSeek-MoE, DeepSeek-V2/V2.5/V3/R1, DeepSeek-V4 / V4-Flash, DeepSeek-V4.1-Flash, every Kimi generation (Kimi-K2 / K2.5, Moonlight, Kimi-VL text, Kimi-Linear, Kimi K3) (dense DeepSeek-LLM / Coder / R1-Distill load as `llama` / `qwen2`) |
| **Exotic quant dtypes** | In-mapping decoders for IQ2_XXS (DeepSeek-V4's 2.0625-bit expert weights), MXFP4 (Kimi K3's routed experts) and Q1_0 / Q2_0 (Bonsai's 1-bit and 2-bit weights), with matmuls that keep the blocks in the mmap instead of materialising f32 — one generic `RawBlock` layer, so any native loader reads any of them |
| **Fused SIMD kernels** | AVX-512 and AVX2 (x86-64) and NEON (aarch64) dequant+dot fusion — the weights decode inside the dot in registers — for Q8_0/Q2_K/Q4_K, IQ2_XXS, MXFP4 and Bonsai's Q1_0/Q2_0, plus parallel SIMD matmuls for the other k-quants; the widest ISA the CPU has is picked at startup (`JOSHUA_SIMD` overrides) |
| **Chat templates** | Renders the model's own `tokenizer.chat_template` from the GGUF (Jinja via pure-Rust minijinja); ChatML fallback |
| **Tool calling** | OpenAI-compatible `tools` / `tool_calls`, parsing Hermes/Qwen, Mistral, and Llama-3 call formats |
| **Embeddings** | Dense sentence embeddings for llama / qwen2 / qwen3 embedding models, with GGUF pooling metadata |
| **KV-cache reuse** | Multi-turn requests continue from a warm model pool and prefill only the new suffix — including across *context edits*: when an agent harness truncates or replaces middle blocks, the pooled session's KV state is rewound to the longest common token prefix instead of being cleared (Qwen and DeepSeek-V2/V3/K2 loaders; models with recurrent layers — Qwen3-Next, Qwen3.5 — reuse extensions and re-prefill on edits). DeepSeek MLA caches the compressed latent (`c_kv` + `k_pe`) instead of the reconstructed per-head K/V, cutting KV memory ~70× |
| **Speculative decoding** | `--speculative N` drafts up to N tokens per step by prompt lookup and verifies them in one forward pass, rolling the KV cache back past the first rejection — output unchanged (token-identical when greedy, same distribution when sampling), fewer weight sweeps on repetitive output (Qwen and `deepseek2` loaders, except recurrent models) |
| **Speculative expert prefetch** | Decode fires `MADV_WILLNEED` for each MoE layer's *predicted* experts (the ids its router chose last step — routing is temporally local) before any layer runs, so expert pages stream in behind compute instead of faulting on demand (`deepseek4` loader) |
| **GPU (optional)** | `--features cuda`, `metal`, `opencl`, `vulkan` or `sycl` route inference through candle's GPU backends |
| **NPU / llama.cpp interop (optional)** | Vendor plugins run in a crash-isolated shim process; a llama.cpp adapter brings every ggml backend (Hexagon NPU, CANN, CUDA, Vulkan, …) |
| **Vision (optional)** | OpenAI-style image messages routed through llama.cpp's `mtmd` (Qwen2.5-VL, Gemma 3, LLaVA, …) via the same isolated plugin |
| **Speech-to-text** | Whisper transcription in pure Rust: `/v1/audio/transcriptions` + `joshua transcribe` |
| **Sampling** | Temperature, top-k, min-p, top-p (nucleus), greedy — all in Rust |
| **HTTPS (optional)** | `--features tls` terminates TLS in-process via rustls — no reverse proxy needed |
| **API-key auth** | Optional `--api-key` guards the `/v1` routes with OpenAI-style bearer authentication |

---

## Architecture

```
┌──────────────────────────┐
│  Joshua  (Rust crate)    │  ← OpenAI-compatible REST API (axum)
└──────────────────────────┘    Chat completions, embeddings, streaming
           │
┌──────────────────────────┐
│  candle  (pure Rust)     │  ← Tensor operations + quantized GGUF inference
└──────────────────────────┘    Llama / Gemma / LFM2 / Phi / Qwen2 loaders
           │
┌──────────────────────────┐
│  tokenizers (pure Rust)  │  ← BPE tokenisation from tokenizer.json
└──────────────────────────┘    HuggingFace tokenizers library
```

---

## Requirements

| Tool | Minimum version |
|---|---|
| Rust toolchain | 1.89 |

The minimum is set by the AVX-512 kernels, whose `core::arch` intrinsics
were stabilized in Rust 1.89 (the dependency tree alone needs 1.88, for
`zip 8.6.0`).

No CMake, no C++ compiler, no CUDA toolkit required.

---

## Quick start

### 1 — Add to `Cargo.toml`

```toml
[dependencies]
joshua = { git = "https://github.com/rexlunae/joshua" }
```

### 2 — Download a model

Any GGUF model with a supported architecture works (see
[Supported models](#supported-models) below).  You also need the
`tokenizer.json` from the same HuggingFace repository — place it alongside the
`.gguf` file.

```bash
# Using the Hugging Face CLI
pip install huggingface-hub

# Download GGUF weights + tokenizer into ./weights/
huggingface-cli download \
    bartowski/google_gemma-3-1b-it-GGUF \
    gemma-3-1b-it-Q4_K_M.gguf \
    --local-dir ./weights

huggingface-cli download \
    google/gemma-3-1b-it \
    tokenizer.json \
    --local-dir ./weights
```

The layout Joshua expects:

```
weights/
├── gemma-3-1b-it-Q4_K_M.gguf   ← quantised weights
└── tokenizer.json               ← HuggingFace tokenizer
```

### 3 — Library usage

```rust
use joshua::{Engine, GenerationOptions, ChatMessage};

fn main() -> anyhow::Result<()> {
    let engine = Engine::new("./weights/gemma-3-1b-it-Q4_K_M.gguf")?;

    let messages = vec![ChatMessage::text("user", "What is Rust?")];

    let opts = GenerationOptions {
        max_tokens:  128,
        temperature: 0.7,
        ..Default::default()
    };

    let (text, usage, prefill_tps, decode_tps) = engine.complete(&messages, &opts)?;
    println!("{text}");
    eprintln!("tokens: {}/{} | prefill {prefill_tps:.0}t/s decode {decode_tps:.0}t/s",
        usage.prompt_tokens, usage.completion_tokens);
    Ok(())
}
```

### 4 — CLI

```bash
# Build (no C++ compiler needed)
cargo build --release

# One-shot completion
./target/release/joshua run \
    --model ./weights/gemma-3-1b-it-Q4_K_M.gguf \
    "Explain memory-mapped I/O in one paragraph"

# Embed texts (dense vectors, llama/qwen2/qwen3 embedding models)
./target/release/joshua embed \
    --model ./weights/nomic-embed-text-v1.5.Q8_0.gguf \
    "first text" "second text"

# Transcribe speech (Whisper model directory, pure Rust)
./target/release/joshua transcribe \
    --model ./weights/whisper-tiny \
    --language en speech.wav

# Start the API server
./target/release/joshua serve \
    --model ./weights/gemma-3-1b-it-Q4_K_M.gguf \
    --addr 0.0.0.0:8080
```

### 5 — GPU acceleration (Metal / CUDA / OpenCL / Vulkan / SYCL)

Joshua runs inference on the CPU by default, but a single build can also run
on a GPU, chosen per invocation.  Add the backend feature **at build time**
(candle compiles the kernels in):

```bash
# Apple Silicon Mac: Metal
cargo build --release --features metal

# NVIDIA GPU: CUDA (needs a CUDA toolkit; Linux/Windows)
cargo build --release --features cuda

# Any OpenCL ICD (Intel iGPU/Arc, pocl, NVIDIA): needs libOpenCL at link time
cargo build --release --features opencl

# Any Vulkan ICD (AMD RADV, Intel ANV, llvmpipe): loads libvulkan.so at runtime
cargo build --release --features vulkan

# SYCL 2020 (Intel oneAPI/LLVM): the Rust crate compiles without a SYCL
# toolchain; build the bridge with a SYCL compiler, then run with JOSHUA_SYCL_LIBRARY set
cargo build --release --features sycl
```

Then pick the device at runtime — `auto` (the default) uses the best backend
this build was compiled with and degrades to CPU with a warning; an explicit
request is strict and fails the load if the device is missing:

```bash
./target/release/joshua serve --model m.gguf --device metal      # force Metal
./target/release/joshua serve --model m.gguf --device cpu        # force CPU
./target/release/joshua serve --model m.gguf --device auto       # default
```

`--device` also accepts `cuda`, `opencl`, `vulkan` and `sycl`, and the same
selection is available as `JOSHUA_DEVICE` for the server.  The resolved device is logged at startup;
library callers use [`EngineOptions::backend`]:

```rust
use joshua::{Engine, EngineOptions, ComputeBackend};

let engine = Engine::with_options("m.gguf",
    EngineOptions::with_n_ctx(4096).backend(ComputeBackend::Metal))?;
```

**What runs on Metal.** Candle's Metal kernels cover every standard GGUF
quantisation (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1 and the k-quants Q2_K…Q8_K), so
the dense architectures, **Qwen3-MoE**, and Joshua's own **DeepSeek-V2/V3 /
Kimi-K2** loader all run quantized on the GPU — experts stay in their
on-disk quantisation rather than being dequantised to f32.  A model that
fits in unified memory (e.g. DeepSeek-V2-Lite Q4_K_M or Qwen3-30B-A3B on a
16 GB Mac) works end to end; weights are copied into Metal buffers at load,
so the file is still mmap'd but the zero-copy borrowing and huge-page tricks
are CPU-only by design.

One kernel constraint: candle's Metal backend routes single-token *decode*
through its fused attention kernel, which supports head dims ≥ 32 — every
real model (llama 64/128, Qwen 128, Gemma 256, DeepSeek MLA 64/128)
qualifies; only toy embeddings with head dim 4 need the CPU.

**What stays on CPU.** `deepseek4` (DeepSeek-V4-Flash) keeps its IQ2_XXS
routed experts on the CPU (no Metal/CUDA kernel exists for them) and runs
only the dense set on the device.  Whisper transcription is CPU-only.

**OpenCL and Vulkan (integrated GPUs).** Both backends run every operator
as a device kernel, keep block-quantized weights in their GGUF format on
the device (dequantized inside the matmul, for every quantisation) and
execute asynchronously; on a device with host-unified memory the OpenCL
backend aliases the memory-mapped file instead of copying it.  The routed
experts of an MoE model stay on the CPU expert kernels, and
`--dense-placement auto` measures whether the device beats the CPU on the
quantized matmul before moving the dense set there.  See
[`docs/accelerator-backends.md`](docs/accelerator-backends.md) for the
design, the environment variables and the limits.

**SYCL (Intel oneAPI / LLVM).** The Rust crate always compiles without a
SYCL toolchain; the kernels live in a separate `libjoshua_sycl` shared
library (built from `vendor/candle-core/src/sycl_backend/bridge.cpp` with
a SYCL compiler via `cmake`), which joshua `dlopen`s at first use.  Point
`JOSHUA_SYCL_LIBRARY` at a non-standard path, `JOSHUA_SYCL_NATIVE=0` to
force the CPU round-trip, and `JOSHUA_SYCL_TRACE=1` to log every native
kernel and fallback.  Supported operations run as native SYCL kernels
(f32 GEMM, RMSNorm, RoPE, softmax, reductions, index/gather/scatter), and
block-quantized weights stay in their GGUF format on the device.
On Linux, `JOSHUA_SYCL_RUNTIME` can point to `libsycl.so.9` to preload
the oneAPI runtime when it is not on the system library path.

### Models larger than VRAM

A sparse MoE model is mostly routed experts, and a GPU that cannot hold all
of them can still run the model well: the *dense* set (embeddings, norms,
attention, routers, shared experts, output) is what every token touches and
what benefits from the device, while each token touches only a handful of
experts.  `--expert-placement` chooses where the routed experts of a
Qwen-family MoE or `deepseek2` model live:

| Value | Effect |
|---|---|
| `auto` (default) | `device` when `dense + experts + 1 GiB` fits the GPU's free memory (`cudaMemGetInfo` on CUDA, or `--vram-budget`), else `host`.  On OpenCL and Vulkan always `host` (the dense set is what an iGPU speeds up; the experts run on the CPU expert kernels).  With neither a probe nor a budget, `device`. |
| `device` | Upload the experts too — the whole model must fit.  For `deepseek4` on OpenCL: run them from a bounded VRAM cache instead (sized as `--vram-expert-cache auto` unless a budget is given; see [`docs/accelerator-backends.md`](docs/accelerator-backends.md)). |
| `host` | Keep the experts in host RAM, borrowed in place from the mapping and run on the CPU SIMD expert kernels (with the hot-expert cache and prefetch machinery active); only the dense set goes to the GPU.  Each MoE layer moves its activations across once in each direction. |

`--vram-budget <MiB>` (or `JOSHUA_VRAM_BUDGET`) states the memory the model
may use when there is no probe (Metal's unified memory, OpenCL, Vulkan) or
when the card is shared.  Placement can only move the routed experts: the
dense set always goes to the device, so a budget (or probed free memory) it
does not fit with 1 GiB of headroom fails the load with the two numbers
instead of running out of device memory on the first request.  The figures
are the on-disk (quantized) sizes on every backend.  `deepseek4` always
keeps its IQ2_XXS experts on the host whatever is requested.  The decision is logged at startup:

```text
INFO joshua: expert placement: host RAM — experts borrowed from the mapping,
     dense set on the device (dense 1.1 GiB, experts 17.5 GiB, device budget
     11.6 GiB, requested Auto)
```

Two things make a second concurrent conversation cheap on the device.
Weights of a Qwen-family / `deepseek2` / `deepseek4` model are loaded (and
uploaded) once and shared by every session: a new session is one
reference-count bump plus an empty KV cache, never a second copy.  And the token-embedding table is
kept quantized and gathered per row instead of being dequantized to a
private f32 copy per session (1.2 GB on Qwen3-30B-A3B, 4.7 GB on Kimi-K2).
For architectures whose sessions still own their weights, the default
concurrency and the warm-session pool are sized from the device's free
memory.  See `docs/memory-analysis.md` for the full accounting.

---

## HTTP API

All endpoints are OpenAI-compatible.

### `POST /v1/chat/completions`

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-1b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64,
    "temperature": 0.7
  }'
```

**Streaming** — add `"stream": true` and consume SSE events:

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma","messages":[{"role":"user","content":"Count to 5"}],"stream":true}'
```

### `POST /v1/embeddings`

```bash
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"nomic-embed","input":["Hello","World"]}'
```

> **Note:** Embeddings run a hidden-state forward pass with the pooling
> strategy from the GGUF metadata (mean / CLS / last-token).  Supported
> architectures: `llama` (e5-mistral, SFR-Embedding), `qwen2` (gte-Qwen2),
> and `qwen3` (Qwen3-Embedding).

### `POST /v1/audio/transcriptions`

Mounted when the server is started with `--whisper-model <dir>` (a directory
holding `model.safetensors` + `config.json` + `tokenizer.json`, e.g. from
`openai/whisper-tiny`).  Pure-Rust pipeline: WAV in any sample rate/channel
count → mel spectrogram → greedy decode.

```bash
curl http://localhost:8080/v1/audio/transcriptions \
  -F file=@speech.wav -F language=en
```

### `GET /v1/models`

```bash
curl http://localhost:8080/v1/models
```

### `GET /health`

```bash
curl http://localhost:8080/health
# {"status":"ok"}
```

---

## Securing the server

The server binds `127.0.0.1` by default and speaks plaintext HTTP with no
authentication — fine for local use or behind a reverse proxy.  Before
binding a public interface (`--addr 0.0.0.0:8080`), enable authentication
and/or TLS below.

### Request limits

Two limits bound how much work a single client can demand, both tunable:

- `--max-concurrency` (default: CPU count) caps simultaneous
  generations/embeddings; requests over the cap get `503 Service Unavailable`
  instead of piling up model instances and exhausting memory.  Lower it for
  large models on small boxes.
- `--max-output-tokens` (default: 4096) caps generated tokens per request
  regardless of the client's `max_tokens`, bounding single-request CPU time.

Uploaded audio is limited to ~30 minutes of 16 kHz-equivalent samples, and
inline image data must be a base64 `data:` URL (filesystem paths and remote
URLs in image fields are rejected).

### API-key authentication

Pass `--api-key` (or set `JOSHUA_API_KEY`) and every `/v1` route requires the
key as an OpenAI-style bearer token; `GET /health` stays open for liveness
probes.  Requests without the key get a `401` with an OpenAI-format error
body, so standard clients report it cleanly.

```bash
joshua serve --model m.gguf --api-key sk-my-secret

curl http://localhost:8080/v1/models \
  -H "Authorization: Bearer sk-my-secret"
```

### TLS (HTTPS)

TLS termination is built in via [rustls](https://github.com/rustls/rustls),
behind an opt-in cargo feature — rustls' ring crypto provider compiles
C/assembly, so it is excluded from the default build to keep the
"only a Rust toolchain required" guarantee (same policy as the `cuda`/`metal`
features).  Building with `--features tls` requires a C compiler.

```bash
cargo build --release --features tls

joshua serve --model m.gguf \
    --tls-cert ./cert.pem \
    --tls-key  ./key.pem

curl https://localhost:8080/v1/models
```

`--tls-cert` takes a PEM certificate chain and `--tls-key` the matching
PKCS#8/RSA/SEC1 private key (both flags required together).  Library users
can call `server::serve_with_state_tls` directly.

---

## Huge pages

Large models thrash the TLB: a 7B Q4 model is ~4 GB, which is two million
4 KiB pages.  Backing the mapping with huge pages cuts TLB misses — and cuts
page-fault count by the same factor while the model warms up.  Select a
strategy with `--huge-pages` (or `EngineOptions::huge_pages` in the library);
the huge-page modes are Linux-only and fall back to normal pages elsewhere
(macOS maps files on its 16 KiB base pages and has no file-backed superpage
API to ask for).

| Mode | Mechanism | Trade-off |
|---|---|---|
| `transparent` (**default on Linux**) | file-backed `mmap` + `MADV_HUGEPAGE` | keeps the shared page cache; best-effort, kernel picks the size (usually 2 MiB); no setup |
| `off` (default elsewhere) | file-backed `mmap`, normal pages | shared via the page cache; no setup |
| `2mb` / `1gb` / `huge` | model copied into an anonymous `MAP_HUGETLB` mapping | guarantees the page size, but uses **private** RAM (no shared page cache) and needs a preallocated pool |

```bash
# Best-effort transparent huge pages — safe to enable anywhere (and the
# Linux default already does this):
joshua serve --model m.gguf --huge-pages transparent

# Explicit 1 GiB pages (reserve the pool first):
sudo sysctl vm.nr_hugepages=$(( 5 * 1024 / 2 ))   # ~5 GiB of 2 MiB pages
joshua serve --model m.gguf --huge-pages 2mb
```

`transparent` is default-on for the CLI on Linux because it is free when it
works and free when it doesn't: the kernel simply keeps normal pages if THP is
unavailable, and the mapping stays file-backed either way.  The explicit modes
give you a guaranteed page size at the cost of a private in-RAM copy and a
preconfigured hugepage pool (`vm.nr_hugepages`, or `hugeadm` for 1 GiB pages).

---

## Compressed model files

Mapping only works when the bytes on disk *are* the model, and there are two
common ways for that to stop being true — neither of them visible from the
filename:

* the `.gguf` is really a **gzip/zstd/xz/… stream** (a download that was never
  unpacked).  Mapping it maps the compressed bytes, so nothing in the mapping
  is a tensor and the load fails — historically with a baffling magic-mismatch
  error from the header parser;
* the `.gguf` sits on a **transparently compressing filesystem** (btrfs/ZFS
  `compress=…`, NTFS compression).  The mapping works, which is why this one
  goes unnoticed, but every page fault has to decompress a block instead of
  handing back a shared page-cache page, and loading and inference get
  dramatically slower.

Joshua checks for both before mapping and reports what it found, naming the
format and the way out:

```text
WARN "./weights/model.gguf" cannot be memory-mapped usefully: the file is a
     gzip stream, not raw GGUF — mapping it maps the compressed bytes, so no
     tensor can be read in place. Decompress it first (gunzip).
```

By default that is a warning and the load continues.  Pass `--mmap` (or
`EngineOptions::mmap(MmapMode::Required)`) to say that mapping is the point of
the run: the same finding then fails the load instead, rather than silently
handing you a mapping that decompresses on every page fault.

```bash
joshua serve --model ./weights/model.gguf --mmap
```

Filesystem compression is detected from the file's on-disk allocation, so a
model stored sparsely reports the same way — equally bad news for a mapping.
It is not reported for `--huge-pages 2mb/1gb/huge`, which copy the model into
anonymous memory in one pass and pay for the decompression only once.

---

## Models larger than RAM

A sparse mixture-of-experts model (DeepSeek-V4-Flash, Qwen3-MoE, …) far larger
than RAM has a bimodal access pattern that plain mmap serves poorly.  A small
*dense* set — embeddings, norms, attention, routers, shared experts,
indexer/compressor, output — is touched on **every** token, while the routed
experts are touched sparsely (a token routes through a handful of the 256 per
layer).  A whole-mapping hint is wrong for one of the two halves: sequential
readahead sets "free after use" and drags expert pages in wholesale; a blanket
random hint kills readahead for everything.

Joshua therefore never hints the model mapping sequentially at all, and splits
the model into dense and expert ranges that are treated differently:

| Flag | Effect |
|---|---|
| `--prefetch-model` | Prefetches the **whole file** into the page cache at load (`MADV_WILLNEED`), so a model that fits in RAM is fully resident before the first request and inference never re-reads weights from disk.  **Auto-on when the model file fits in RAM**; `--prefetch-model=false` forces it off. |
| `--pin-hot-weights` | For models larger than RAM: prefetches only the dense ranges (`MADV_WILLNEED`) at load and advises `MADV_RANDOM` on expert ranges, so the per-token working set is resident before the first request while experts page in on demand.  **Auto-on when the model file is larger than RAM**; `--pin-hot-weights=false` forces it off. |
| `--mlock-hot-weights` | Additionally `mlock(2)`s the dense ranges for a hard residency guarantee.  `=required` fails the load when the memlock limit is too low; the default (`on`) warns once and degrades to advisory pinning. |
| `--lazy-weights` | The blanket random-access hint for the whole mapping — no readahead at all.  Mostly superseded by `--pin-hot-weights`, kept for the truly-RAM-starved case. |

The auto choice between the first two is made from the model size vs total RAM
(`/proc/meminfo` on Linux, `hw.memsize` on macOS) and logged at startup:

```text
INFO joshua: page-cache auto: model 0.4 GiB vs RAM 24.0 GiB — prefetching the
     whole model into the page cache
```

The memlock limit is checked against the hot-set size **before** any `mlock`
call, with one warning naming limit vs required size (on DeepSeek-V4-Flash Q2_K
the dense set is ~8.2 GiB).  Raise it with:

```bash
# systemd services
LimitMEMLOCK=infinity

# login session / PAM
# /etc/security/limits.conf:
#   tserica - memlock unlimited

# live, without re-login (systemd user session)
sudo prlimit --pid <user manager pid> --memlock=-1:-1
```

Prefill on the sparse MoE loaders streams the experts instead of faulting them
in one 4 KiB page at a time: the layer loop advises `MADV_SEQUENTIAL` over the
whole expert span for the duration of the pass and dispatches the routed
experts in tensor-major (file) order, so each expert tensor is read as one
clean sequential pass while the layer computes (measured ~1.9 GB/s vs
~175 MB/s for per-page demand faults).

> **systemd *user* session trap:** the unit's `LimitMEMLOCK=infinity` is capped
> by the user manager's own hard limit, which is inherited from the login
> session — so on a headless box the unit setting alone does nothing.  Add the
> PAM line and apply the `prlimit` live fix (or re-login) for it to take effect.

On DeepSeek-V4-Flash Q2_K the whole file collapses to 2 dense + 1 expert
ranges, so pinning costs two `madvise` calls and one `mlock`.  `examples/tensor_sizes.rs`
prints the dense/expert split of any GGUF to sanity-check a new model.

---

## Environment variables

| Variable | Description |
|---|---|
| `JOSHUA_MODEL_PATH` | Default model path (overrides `--model` flag) |
| `JOSHUA_API_KEY` | API key required on `/v1` routes (same as `--api-key`) |
| `JOSHUA_TLS_CERT` | PEM certificate chain for HTTPS (same as `--tls-cert`; needs `--features tls`) |
| `JOSHUA_TLS_KEY` | PEM private key for HTTPS (same as `--tls-key`) |
| `JOSHUA_LAZY_WEIGHTS` | Same as `--lazy-weights` |
| `JOSHUA_PREFETCH_MODEL` | Same as `--prefetch-model` (`true`/`false`, or the flag with no value) |
| `JOSHUA_PIN_HOT_WEIGHTS` | Same as `--pin-hot-weights` (`true`/`false`, or the flag with no value) |
| `JOSHUA_MLOCK_HOT_WEIGHTS` | Same as `--mlock-hot-weights` (`on`, `required`, or `off`) |
| `JOSHUA_EXPERT_PLACEMENT` | Same as `--expert-placement` (`auto`, `device`, or `host`) |
| `JOSHUA_VRAM_BUDGET` | Same as `--vram-budget` (MiB of accelerator memory the model may use) |
| `JOSHUA_DENSE_PLACEMENT` | Same as `--dense-placement` (`auto`, `device`, or `cpu`) |
| `JOSHUA_VRAM_EXPERT_CACHE` | Same as `--vram-expert-cache` (`auto` or MiB of device memory for the bounded expert cache) |
| `JOSHUA_EXPERT_MISS` | `upload` makes the VRAM expert cache upload decode misses synchronously (measurement mode; default: host run + background upload) |
| `JOSHUA_EXPERT_HOST_PAGES` | `keep` leaves an uploaded expert's host pages in the page cache; the default releases them so RAM and VRAM hold different experts |
| `JOSHUA_EXPERT_STATS` | `1` probes host-miss page residency for the decode time split logged with the VRAM expert cache |
| `JOSHUA_ROUTE_TRACE` | Path of a routing-trace CSV to write, for `cargo run --example cache_sim` |
| `JOSHUA_PREFILL_CHUNK` | Tokens per prefill chunk (default 512; also `--prefill-chunk`) |
| `JOSHUA_SPECULATIVE` | Max draft tokens per speculative decode step (default 0 = off; also `--speculative`) |
| `JOSHUA_SIMD` | Force a lower CPU kernel family: `avx512`, `avx2`, `neon` or `scalar` (default: the widest the CPU supports).  Every level computes the same f32-activation math, so this bisects a numerical difference or a slowdown to one family |
| `JOSHUA_SKIP_PLACEMENT_BENCH` | Skip the startup quantized-matmul probe that `auto` dense placement uses |
| `JOSHUA_OPENCL_NATIVE` / `JOSHUA_VULKAN_NATIVE` | `0` runs every operator through the CPU round-trip instead of the device kernels (default on) |
| `JOSHUA_OPENCL_TRACE` / `JOSHUA_VULKAN_TRACE` | `1` logs each operator that falls back to the CPU and why |
| `JOSHUA_OPENCL_ZERO_COPY` | `0` uploads weights instead of aliasing the memory-mapped file on unified-memory OpenCL devices |
| `JOSHUA_OPENCL_NATIVE_DENY` | Comma-separated OpenCL kernel names to refuse (their ops take the CPU path) — bisecting a bad result on one driver |
| `JOSHUA_OPENCL_CHECK_NAN` | `1` counts NaNs on the device after every native f32 launch and names the first op that produced one |
| `JOSHUA_OPENCL_BUILD_OPTS` | Extra options for the OpenCL kernel compiler (e.g. `-cl-opt-disable`) |
| `JOSHUA_OPENCL_QGEMV` | `v1` runs the IQ2_XXS / Q2_K matmuls through the one-row quantized GEMV instead of the multi-row kernel (bisecting) |
| `JOSHUA_MAX_CONCURRENCY` | Cap on simultaneous generations/embeddings (same as `--max-concurrency`) |
| `JOSHUA_MAX_OUTPUT_TOKENS` | Hard ceiling on generated tokens per request (same as `--max-output-tokens`) |
| `JOSHUA_WHISPER_MODEL` | Whisper model directory mounted at `/v1/audio/transcriptions` (same as `--whisper-model`) |
| `JOSHUA_NPU_PLUGIN` | NPU vendor plugin path (same as `--npu-plugin`) |
| `JOSHUA_LLAMA_N_GPU_LAYERS` | llama.cpp adapter layer offload count (default: all) |
| `JOSHUA_LLAMA_MMPROJ` | Multimodal projector GGUF for vision via llama.cpp's `mtmd` |
| `JOSHUA_LLAMA_BACKENDS_DIR` | Directory of `libggml-<name>` modules to register at adapter startup (llama.cpp `dynamic-backends` builds) |
| `RUST_LOG` | Log filter (e.g. `info`, `joshua=debug`) |

---

## Speculative decoding

A decode step is bound by reading the weights — for a sparse MoE model, by
faulting in the routed experts — not by arithmetic, so scoring several tokens
in one pass costs little more than scoring one.  `--speculative N` (or
[`EngineOptions::speculative`]) uses that:

1. After each token, a **prompt-lookup drafter** finds the latest earlier
   occurrence of the context's trailing n-gram (4 down to 2 tokens) and
   proposes the up-to-N tokens that followed it.  No draft model is needed;
   drafting is a hash lookup.
2. The model scores the token plus the draft in **one forward pass**,
   returning logits for every position.
3. Draft tokens are checked in order against the request's own sampler
   (repetition penalty, temperature, top-k/min-p/top-p).  Greedy decoding
   accepts a token iff it is the argmax; sampled decoding uses speculative
   sampling (accept with probability `p(x)`, otherwise sample the
   correction from `p` without `x`), so the output distribution is exactly
   that of plain decoding.
4. The KV cache is truncated back past the first rejected token, and the
   live draft length adapts (doubling after a fully accepted draft, halving
   after a fully rejected one).

It pays off on output that repeats its context — code edits, quoted
passages, tool-call arguments, structured data — and costs roughly nothing
when the drafter finds no match (it then falls back to the one-token step).

```bash
joshua run model.gguf "Rewrite this function with better names: ..." --speculative 8
# stderr ends with: [speculative: accepted <a>/<d> drafted (<rate>%) over <n> verify steps]
```

Library users read the same counters from `Engine::speculative_stats()`.
Supported by the architectures whose KV cache can be rolled back — the Qwen
loader's attention-only models (`qwen`, `qwen2moe`, `qwen2vl`, `qwen3moe`,
`qwen3vl`, `qwen3vlmoe`) and `deepseek`/`deepseek2` (DeepSeek-MoE/V2/V3,
Kimi-K2); other models — including the recurrent Qwen3-Next / Qwen3.5 —
ignore the setting and decode one token at a time.

---

## Generation options

| Field | Type | Default | Description |
|---|---|---|---|
| `max_tokens` | `u32` | `256` | Maximum tokens to generate |
| `temperature` | `f32` | `0.7` | Sampling temperature (0 = greedy) |
| `top_p` | `f32` | `0.9` | Nucleus sampling threshold |
| `top_k` | `i32` | `40` | Top-k sampling (0 = disabled) |
| `min_p` | `f32` | `0.05` | Min-p filter relative to top token |
| `repetition_penalty` | `f32` | `1.1` | Penalise tokens seen in the last 64-token window (1.0 = disabled) |
| `stop_sequences` | `Vec<String>` | `[]` | Stop on these strings |

---

## Supported models

Joshua reads `general.architecture` from the GGUF metadata and dispatches to
the matching pure-Rust candle loader.  Currently supported architectures:

| `general.architecture` | Model families |
|---|---|
| `llama` | Llama 1/2/3, Mistral, Mixtral, TinyLlama, SmolLM, Vicuna, Zephyr, Yi, and anything else llama.cpp's converters emit as `llama` |
| `gemma` / `gemma2` / `gemma3` / `gemma-embedding` | Gemma 1, Gemma 2, Gemma 3 |
| `chatglm` | ChatGLM2, ChatGLM3, GLM-4-9B-Chat (fused QKV, fused SwiGLU) |
| `glm4` | GLM-4-0414 (dense, sandwich norms), GLM-4.1V (text decoder, M-RoPE), GLM-OCR (NextN block skipped) |
| `glm4moe` | GLM-4.5, GLM-4.5-Air, GLM-4.6, GLM-4.7, Solar-Open (sigmoid-routed MoE with a shared expert; NextN blocks skipped) |
| `lfm2` | Liquid LFM2 |
| `phi2` | Phi-1, Phi-1.5, Phi-2 |
| `phi3` | Phi-3, Phi-3.5 |
| `qwen` | Qwen (1) — fused QKV with bias |
| `qwen2` | Qwen1.5, Qwen2, Qwen2.5 |
| `qwen2moe` | Qwen1.5-MoE, Qwen2-57B-A14B (gated shared expert) |
| `qwen2vl` | Qwen2-VL, Qwen2.5-VL (text decoder, M-RoPE) |
| `qwen3` | Qwen3 (dense), Bonsai (Q1_0 / Q2_0 weights) |
| `qwen3moe` | Qwen3 mixture-of-experts, Qwen3-Coder |
| `qwen3vl` / `qwen3vlmoe` | Qwen3-VL dense / MoE (text decoder, interleaved M-RoPE) |
| `qwen3next` | Qwen3-Next (Gated DeltaNet + gated attention hybrid, MoE) |
| `qwen35` / `qwen35moe` | Qwen3.5 dense / MoE (Gated DeltaNet hybrid) |
| `qwen4exp` | Qwen3.8-Flash-Next (Qwen3.5-MoE plus hyper-connections, QSA block-sparse attention, PLE n-gram hash embeddings) |
| `deepseek` | DeepSeek-MoE 16B (GQA attention + fine-grained MoE with shared experts) |
| `deepseek2` | DeepSeek-V2, DeepSeek-V2-Lite, DeepSeek-V2.5, DeepSeek-V3 / V3.1, DeepSeek-R1, **Kimi-K2** / K2.5, Moonlight, Kimi-VL (text), GLM-4.7-Flash (MLA attention + fine-grained MoE) |
| `glm-dsa` | GLM-5, GLM-5.1, GLM-5.2 (the `deepseek2` stack plus DeepSeek Sparse Attention: a lightning indexer picks each query's top-k keys, and GLM-5.2's IndexShare layers reuse an earlier layer's pick) |
| `glm5next` / `glm5-next` | GLM-5.3-Flash (three in four layers Kimi Delta Attention, the rest NoPE MLA over a k-pool sparse indexer; manifold-constrained hyper-connections; clamped-SwiGLU MoE) — both names written by llama.cpp's open pull requests load |
| `kimi-linear` | Kimi-Linear (three in four layers Kimi Delta Attention, the rest NoPE MLA; sigmoid-routed MoE with shared experts) |
| `kimi-k3` | Kimi K3 (Kimi-Linear's hybrid plus attention residuals, a latent MoE with MXFP4 experts, `situ` activations and gated MLA) |
| `deepseek4` | DeepSeek-V4 (Hyper-Connections residual mixing, alternating sliding-window / learned KV-compressor attention, Lightning-Indexer sparse attention, fine-grained MoE with IQ2_XXS experts) |
| `deepseek41` | DeepSeek-V4.1-Flash (V4 with a one-sublayer-lagged hyper-connection mix and no learned HC head, KV compressed on a few source layers and shared by the layers after them, engram n-gram hash tables) |

The `deepseek2` loader is Joshua's own (candle has no quantized DeepSeek
path). It implements Multi-head Latent Attention with Q/KV LoRA, DeepSeek-V3 /
Kimi-K2 sigmoid-with-bias group-limited expert routing, shared experts, and
YaRN RoPE — and keeps the experts **quantized** (a 1 T-parameter MoE keeps its
on-disk footprint instead of exploding to f32 in RAM). Since PR #30, MLA
attention caches the *compressed latent* (`c_kv` + `k_pe`) rather than the
reconstructed per-head K/V — numerically identical to llama.cpp and ~70× less
KV memory for V3-class models; the full K/V is rebuilt once per forward from
the latent. Both the legacy combined (`attn_kv_b`) and modern MLA-split
(`attn_k_b`/`attn_v_b`) GGUF encodings load, and its logits are cross-checked
against llama.cpp. The same loader serves `deepseek` (DeepSeek-MoE, the V1
generation), which shares the whole MoE stack and swaps MLA for plain GQA
attention over the same KV-cache layout, and the GLM-5 generation:
`glm-dsa` adds DeepSeek Sparse Attention (a lightning indexer restricting each
query to its top-k keys, with GLM-5.2's IndexShare), and `glm5next`
(GLM-5.3-Flash) further interleaves Kimi Delta Attention layers, drops RoPE,
pools the indexer's keys and carries the residual as manifold-constrained
hyper-connections (shared with the `deepseek4` loader). Both are checked
against independent float64 transcriptions of llama.cpp's graph (`glm-dsa`)
and of HF transformers' model code under llama.cpp's GGUF conventions
(`glm5next`). The other GLMs (`chatglm`, `glm4`, `glm4moe`) run on the Qwen
family loader, cross-checked the same way; appended NextN (MTP) blocks are
skipped by every GLM loader. The Kimi hybrids run on the same loader too:
`kimi-linear` interleaves Kimi Delta Attention with MLA whose rope slice is
never rotated, and `kimi-k3` adds a full-rank KDA output gate, a sigmoid
output gate on MLA, `situ` activations, a latent MoE (routed experts in a
narrower space, MXFP4 blocks borrowed from the mapping) and attention
residuals (every few layers the residual stream is banked and restarted, and
each sublayer reads a softmax mix of the bank). Both are checked against an
independent float64 transcription of llama.cpp's `kimi-linear.cpp` /
`kimi-k3.cpp`.

The dense DeepSeek releases — DeepSeek-LLM, DeepSeek-Coder V1 and the
DeepSeek-R1 distills — are converted by llama.cpp as `llama` or `qwen2` and
load through those rows.

The `deepseek4` loader handles the architecture's three additions over V3:
Hyper-Connections mix the residual stream to `hc_mult` parallel copies with
Sinkhorn-normalised per-token weights, CSA/HCA compressor layers pool blocks of
4 / 128 tokens into a compressed KV, and the Lightning Indexer picks the
`index_topk` compressed positions each query attends to. The routed experts
stay quantized as IQ2_XXS trellis blocks decoded in-mapping during the matmul,
so a 162 B model keeps its on-disk footprint. Activations run in f32 on CPU.

DeepSeek-V4.1 (`deepseek41`) is a separate GGUF architecture served by the
same loader.  Each sublayer collapses the hyper-connection copies with the
mix the *previous* sublayer computed, and the last one's mix replaces V4's
learned head.  Only the layers that carry a compressor pool KV (at ratio 1
or 2, no overlap); the layers after each source read its rows, and index
keys come from that shared latent rather than from a second compressor.  A
few layers first add engram rows: each token's preceding n-grams are
hashed (after case/accent folding) into buckets of a huge table, which stays
paged in the mapping, and gated into every copy.  llama.cpp's own support is
still an unmerged runtime branch (behind ggml-org/llama.cpp#28696), which
leaves its two-level candidate mask unimplemented; that mask selects every
block below ~16K tokens of compressed context.  Joshua follows the same
graph, and its logits are pinned to an independent NumPy transcription of it.

Every Qwen architecture except the dense `qwen2` (candle's loader) goes
through Joshua's own `quantized_qwen` loader: one decoder layer
whose parts are optional — fused or split QKV with biases, per-head Q/K norm,
the Qwen3-Next generation's sigmoid output gate and partial RoPE, dense or
MoE FFN with a sigmoid-gated shared expert, and Gated DeltaNet linear
attention (causal conv + gated delta rule) for the hybrid models.  The VL
models' multi-section RoPE reduces, for text positions, to 1-D RoPE with the
"extra"-section frequencies frozen, exactly as llama.cpp computes it; image
input goes through the llama.cpp `mtmd` plugin below.  `qwen4exp` adds three
pieces: hyper-connections (the residual is several parallel streams mixed
by low-rank sigmoid gates, and the final mixer is the output norm), QSA
block-sparse attention (an indexer scores mean-pooled blocks of cached keys
and each query attends only to its best `indexer.top_k` cells plus its
incomplete tail block — dense until the context outgrows that budget), and
PLE n-gram hash embeddings (each token hashes its preceding n-grams into
rows of a shared table, gated into every stream through a dilated causal
conv).  Bonsai models are `qwen3` GGUFs in the 1-bit Q1_0 (`±d` per
element, 128-element blocks) and 2-bit Q2_0 (`{-1, 0, 1, 2}·d`, 64-element
blocks) formats candle cannot parse; the native loaders read such tensors
from the raw GGUF header, borrowing the blocks from the mapping and decoding
them inside an f32-activation matmul (on an accelerator, or without a
mapping, they are decoded to f32 at load).  Each architecture's logits are
pinned to an independent NumPy transcription of llama.cpp's graphs.  Recurrent layers cannot rewind to an arbitrary prefix, so Qwen3-Next
and Qwen3.5 re-prefill on edited contexts instead of truncating.  Not
loadable: `qwen3tts` (speech codec output) and `rwkv6qwen2` (an RWKV-6
distillation, not a Qwen decoder).

Example models:

- `google/gemma-3-270m-it` / `1b-it` / `4b-it`
- `Qwen/Qwen3-0.6B` / `1.7B`, `Qwen/Qwen3-30B-A3B`, `Qwen/Qwen3-Next-80B-A3B-Instruct`, `Qwen/Qwen3.5-9B` (as GGUF)
- `LiquidAI/LFM2-1.2B`
- `microsoft/Phi-3-mini-4k-instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `THUDM/GLM-4-9B-0414`, `zai-org/GLM-4.5-Air`, `zai-org/GLM-4.7-Flash`, `zai-org/GLM-5.2`, `zai-org/GLM-5.3-Flash` (as GGUF)
- `deepseek-ai/deepseek-moe-16b-chat`, `deepseek-ai/DeepSeek-V2-Lite`, `moonshotai/Kimi-K2-Instruct` (as GGUF)
- `deepseek-ai/DeepSeek-V4-Flash-162B` (as GGUF)

Every other architecture name in llama.cpp's registry (Mamba, RWKV, GPT-2,
Granite, OLMo, StarCoder2, and ~70 more) is recognised at load time
and rejected with an error that names the architecture and lists what is
supported — so an unsupported model fails fast with a clear message instead
of a cryptic missing-tensor error.  Coverage grows as candle gains loaders;
adding one is a small patch to `src/model.rs`.

---

## NPU & llama.cpp backend interop (experimental)

Vendor NPU runtimes are proprietary C/C++ stacks, so Joshua contains them
behind three safety layers instead of linking them into the pure-Rust core:

1. **Trait boundary** — generation transparently falls back to the candle
   CPU/GPU path when a backend is missing or failing; a circuit breaker
   disables a backend after repeated failures.
2. **Plugin ABI, loaded at runtime** — a backend is any shared library
   exporting the four-function `joshua_npu_*` C ABI (`init` / `forward` /
   `reset` / `free`, documented in `joshua::npu`).  Nothing is linked at
   build time; the default build stays pure Rust.
3. **Process isolation** — by default the plugin runs inside the small
   `joshua-npu-shim` subprocess: control over pipes, tensors over shared
   memory, timeouts enforced, child killed on any violation.  A crashing or
   hanging vendor runtime costs one request, never the server.

```bash
# Isolated by default:
joshua serve --model m.gguf --npu-plugin /path/to/libvendor.so
# Opt into in-process loading (faster, but a plugin crash is fatal):
joshua serve --model m.gguf --npu-plugin /path/to/libvendor.so --npu-in-process
```

### Vision / multimodal

Vision rides the same plugin mechanism: an optional fifth ABI symbol,
`joshua_npu_media_prefill`, lets a plugin tokenise-and-prefill a prompt whose
`<__media__>` markers correspond to attached images.  The llama.cpp adapter
implements it with llama.cpp's `mtmd` — covering Qwen2.5-VL, Gemma 3 vision,
LLaVA, and the rest of its multimodal zoo:

```bash
# Point the adapter at the model's multimodal projector:
JOSHUA_LLAMA_MMPROJ=./weights/mmproj.gguf \
joshua serve --model ./weights/qwen2.5-vl.gguf \
    --npu-plugin target/release/libjoshua_llamacpp_npu.so
```

Clients send standard OpenAI vision messages (content parts with
`image_url` data URLs) or `ChatMessage.images` paths; decoding, sampling,
streaming, and tool calling work unchanged after the multimodal prefill.
Requests with images and no media-capable plugin fail fast with a clear
error.

### The llama.cpp adapter

No vendor ships Joshua plugins — they ship **llama.cpp/ggml backends**.  The
`joshua-llamacpp-npu` crate bridges that: it implements the plugin ABI by
driving llama.cpp itself, so every backend llama.cpp supports (Qualcomm
Hexagon NPU, Huawei CANN, CUDA, Vulkan, OpenCL, Metal, …) works through the
same isolated shim, against the same GGUF file, with no model conversion:

```bash
# Compiles llama.cpp — needs CMake + a C++ toolchain, which is exactly why
# it is NOT part of the default build; the C++ only ever runs in the shim.
cargo build --release -p joshua-llamacpp-npu

joshua serve --model m.gguf \
    --npu-plugin target/release/libjoshua_llamacpp_npu.so
```

Layer offload is controlled with `JOSHUA_LLAMA_N_GPU_LAYERS` (default: all).
NPU backends are enabled the same way as in llama.cpp itself — build it with
the vendor SDK (see llama.cpp's Snapdragon/CANN backend docs).  With
llama.cpp's `dynamic-backends` feature, ready-built `libggml-<name>` modules
are picked up at startup from `JOSHUA_LLAMA_BACKENDS_DIR` (when unset, the
compile-time default directory is scanned), so a cross-compiled Hexagon/CANN
backend drops in with no code change.

The test suite proves the stack end to end without real hardware: a mock
vendor plugin exercises determinism, crash containment (the plugin aborts —
the server survives), hang timeouts, and engine fallback; the llama.cpp
adapter is verified to produce byte-identical greedy output to Joshua's own
candle path on the same weights.

---

## Experimental DeepSeek V4 clustering (#86–91)

The `distributed` Cargo feature adds **CPU-only, static-cluster DeepSeek V4
generation** through `joshua cluster-run`, plus the building blocks below.
DeepSeek V4 Flash is the first full-model target. This is not a distributed
`Engine::complete` or OpenAI server mode; existing single-node inference is
unchanged. The library exposes:

- `distributed::deepseek4::DeepSeekCluster`: one ordered DeepSeek V4 request
  with routed-expert tensor parallelism, replicated attention and KV caches.
- `distributed::session::ClusterSession`: fixed authenticated membership,
  input/configuration agreement, chunked reductions and rank-zero broadcasts.
- `distributed::shard`: checked, block-aligned input-column slices of GGUF
  matrices, local mmap ownership, shard-only prefetch, and partial matvecs.
- `distributed::collective` (feature-gated): authenticated UDP multicast
  summation for a fixed set of ranks, with bounded retries/timeouts.
- `distributed::tcp` (feature-gated): authenticated TCP ring summation for
  explicitly configured ranks on networks without multicast.
- `distributed::discovery` (feature-gated): advisory `_joshua._tcp.local.`
  discovery and coordinator selection. Discovered nodes are **not** automatically
  trusted or admitted to a collective.
- `distributed::partition`: deterministic, memory-constrained block allocation
  using caller-supplied capacity and topology observations.

### Correctness and memory model

Joshua stores a linear weight as `[output, input]` and computes `y = W x`.
For an all-reduce, split the **input/reduction dimension**: each rank computes
`W[:, start..end] x[start..end]`, then sums full-length output vectors. These
columns form a separate byte range in **every output row**, not one contiguous
slice of the tensor. Splitting output rows instead requires an all-gather,
not an all-reduce.

Boundaries are whole quantization blocks. Q8_0, Q4_K, Q2_K, IQ2_XXS and MXFP4
are covered by the sharding tests; invalid and empty shards are rejected.
Byte reconstruction is exact, but floating-point sums can differ with reduction
grouping, so numerical comparisons use tolerances rather than promising
bit-identical logits or generated tokens.

Pre-sync the **same immutable GGUF** to each node's local SSD, for example with
`rsync --partial model.gguf node:/srv/models/`, and compare `sha256sum` on each
copy before launch. Do not overwrite or truncate a file while mapped.
The shard path maps the local file lazily and only accesses/prefetches local
row slices; it does not call the engine's full-file prefetch. Mapping the whole
file reserves virtual addresses, not that amount of physical RAM. OS page
granularity, readahead, metadata, activations and runtime allocations mean RSS
is **not exactly** the encoded shard size; reserve additional RAM accordingly.
NFS does not share a page cache across machines and is not required.

### Run DeepSeek V4 Flash on a static cluster

Build the **same revision** on every node:

```bash
cargo build --release -p joshua --features distributed
```

Pre-sync the GGUF and `tokenizer.json` to each node's local SSD and verify
their checksums before launching. Startup checks GGUF headers, file lengths,
context limits, input tokens and generation settings; it deliberately **does
not hash all weight bytes**, which would read off-rank weights. Equal headers
do not prove equal weights.

Create a fresh job UUID and a random 32-byte key once, and securely distribute
the same values to every rank. For example:

```bash
export SESSION=$(python3 -c 'import uuid; print(uuid.uuid4())')
export JOSHUA_CLUSTER_KEY=$(openssl rand -hex 32)
```

On node 0 (`10.0.0.1`), with the model under `/srv/models/`:

```bash
./target/release/joshua cluster-run \
  --model /srv/models/deepseek-v4-flash.gguf \
  --rank 0 --world-size 2 --transport tcp \
  --peers 10.0.0.1:48888,10.0.0.2:48888 --session "$SESSION" \
  --n-ctx 4096 --prefill-chunk 32 --max-tokens 64 \
  "Explain how a hash table works."
```

Run the identical command on node 1, changing only `--rank 0` to `--rank 1`
and, if necessary, the local model/tokenizer paths. Launch all ranks together.
Only rank zero prints the completion; all ranks must participate until the job
finishes. The GGUF chat template is used by default. Use `--raw-prompt` for
already-formatted model input, or `--tokens 1,4,2` instead of a text prompt to
test pre-tokenized input and receive generated token IDs as JSON.

TCP is the default. To use multicast, omit `--peers`, select `--transport udp`,
and set the same `--group`/`--port` on every node with a per-node `--interface`
LAN address. For local testing, TCP peers can be
`127.0.0.1:48888,127.0.0.1:48889`. Open only the selected ports on a trusted
network: HMAC authenticates traffic but does **not encrypt it**. mDNS does not
automatically admit machines to the job.

Each routed MLP splits its intermediate dimension: gate/up **output rows**
and matching down-projection **input columns**. The elementwise SwiGLU stays
local; the weighted routed outputs are reduced before adding the replicated
shared expert. Boundaries respect down-projection quantization blocks. Too
many ranks for a nonempty aligned shard are rejected. Router decisions and
generated tokens are coordinated across ranks; sampling is greedy.

The encoded routed-expert byte totals printed at startup are **not RSS**.
Dense weights, embeddings, shared experts, attention/compression state and KV
are replicated on every node. Only local routed-expert ranges are read or
prefetched; whole-layer expert prefetch and the device expert cache are
disabled on this path. Short down-projection rows can put local and off-rank
columns on **every same OS page**, so that projection's residency can remain
near its unsharded size even though only local columns are decoded. Do not
budget physical RAM from the encoded shard ratio alone.
This first implementation uses CPU reference shard
matvecs, not the optimized accelerator or fused quantized kernels. It is a
correctness bring-up path, **not a throughput promise**.

The cluster handles one request at a time, with fixed membership and equal
block-balanced expert shards. It does not yet consume the heterogeneous
partition planner. A mismatch, timeout or failed rank invalidates the job;
restart **every** rank with a fresh UUID. There is no partial-result
degradation, mid-request transport switch, live repartitioning or automatic
failover. `--timeout-seconds` (default 120, maximum 120) applies to each
transport exchange, not the total inference time.

### Offline acceptance checks and issue status

Local regression tests compare tiny DeepSeek V4 IQ2_XXS/I32 hash and regular
MoE inference against a single-node reference, including prefill and decode.
Run the clustering and model tests with:

```bash
cargo test -p joshua --features distributed --lib distributed
cargo test -p joshua --features distributed --test deepseek4_tests
cargo test -p joshua --features distributed --bin joshua cluster_cli
```

Before treating the RFC as complete, validate the real Flash GGUF offline:

1. Compare a one-rank `cluster-run` baseline with two ranks using identical
   token input and greedy decoding. Record revision, GGUF checksum, quantization,
   context/chunk size, generated IDs and per-rank encoded shard bytes.
2. Inspect each process's `/proc/<pid>/smaps_rollup` after prefill and decode.
   Include replicated weights/KV and page overhead when interpreting residency.
3. Repeat on the LAN with both TCP and UDP; record end-to-end throughput
   (including prefill), interconnect speed and packet-loss behavior. Kill a rank
   and confirm peers fail within their configured exchange deadlines rather
   than emitting a successful incomplete completion.
4. Separately validate discovery join/leave timing and a 100-node discovery
   load. A tiny loopback test is not evidence for the LAN scalability targets.

Issues #86–91 are **not all complete**: #87 still needs LAN/scalability
verification; #88 needs real-network loss/throughput validation and its remaining
protocol optimizations; #89 has local shard/model coverage but does not promise
bit-exact floating-point results; #90 needs real-model residency measurements;
#91 still needs live topology measurement, runtime scheduling/repartitioning and
wall-clock balance validation. These gaps must not be marked complete merely
because the local inference path works.

### Run the linear-layer proof

The example generates deterministic activations and accepts any supported 2-D
GGUF weight via `--model` / `--tensor`. A small Q8_0 fixture avoids downloading
a model:

```bash
cargo build --features distributed --example cluster_linear
./target/debug/examples/cluster_linear fixture --output /tmp/joshua-linear.gguf
./target/debug/examples/cluster_linear local --model /tmp/joshua-linear.gguf --shards 3

# Generate once per job; share these out of band with the other ranks.
export JOSHUA_CLUSTER_KEY="$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
session="$(cat /proc/sys/kernel/random/uuid)"

# Two independent processes, three consecutive collectives, on Linux loopback.
./target/debug/examples/cluster_linear rank --model /tmp/joshua-linear.gguf \
  --rank 0 --world-size 2 --session "$session" --steps 3 --verify &
rank0=$!
./target/debug/examples/cluster_linear rank --model /tmp/joshua-linear.gguf \
  --rank 1 --world-size 2 --session "$session" --steps 3 --verify &
rank1=$!
wait "$rank0"
wait "$rank1"
unset JOSHUA_CLUSTER_KEY
```

The fixture command refuses to overwrite existing files. `--verify` deliberately
reads the full weight on each rank; omit it when evaluating shard-only residency.
For two machines, use the same key, fresh session, tensor and GGUF on both,
distinct ranks, and `--interface <local-LAN-IPv4>` instead of loopback. Permit the
chosen multicast group/UDP port through the firewall. Launch all ranks within
the configured deadline (`--timeout-seconds`, default 10); rank output is one
JSON vector per step. This proves a **linear layer**, not token generation.

### TCP fallback without multicast (#88)

Select `--transport tcp` explicitly before starting the job. Supply the same
`--peers` list on every rank, ordered by rank number, with exactly `--world-size`
distinct IP:port addresses. Each rank listens on its own listed address and
connects to its successor in the ring; the last rank connects to rank zero.
Use reachable local-interface addresses, not wildcard addresses. For a
same-machine test, assign a different loopback port to each process:

```bash
export JOSHUA_CLUSTER_KEY="$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
session="$(cat /proc/sys/kernel/random/uuid)"
peers="127.0.0.1:48888,127.0.0.1:48889"

./target/debug/examples/cluster_linear rank --model /tmp/joshua-linear.gguf \
  --transport tcp --peers "$peers" --rank 0 --world-size 2 \
  --session "$session" --steps 3 --verify &
rank0=$!
./target/debug/examples/cluster_linear rank --model /tmp/joshua-linear.gguf \
  --transport tcp --peers "$peers" --rank 1 --world-size 2 \
  --session "$session" --steps 3 --verify &
rank1=$!
wait "$rank0"
wait "$rank1"
unset JOSHUA_CLUSTER_KEY
```

On separate machines, replace loopback with each machine's LAN address and
permit inbound TCP on the listed ports. The UDP `--group`, `--port` and
`--interface` settings do not configure TCP. All ranks still require the same
immutable local GGUF, tensor, fresh session and key. TCP does not perform
discovery, model synchronization or automatic cluster admission.

The library's `TcpAllReduceGroup` binds its listener on construction and
establishes ring connections during the first collective. Connections persist
across steps. Input vectors circulate around the ring and are summed in rank
order, with the same tensor/aggregate storage bounds as UDP. This is a
correctness-first ring exchange, not an optimized reduce-scatter implementation.
Authentication binds messages to the session and ordered membership; it does
**not encrypt** activations. Connection setup and I/O share a bounded deadline.
An exchange failure leaves the caller's input unchanged and poisons the group.
Restart **all ranks** with a new session after failure; never switch transports
mid-job. TCP does not guarantee globally atomic success during a partition.

### Discovery and planning

For advisory discovery, run `cluster_linear discover --node-id <persistent-UUID>`
on each machine. Save each node's UUID in its configuration and reuse it on
restart. The advertised port is reserved for future control-plane integration;
the example does not implement a TCP inference service.

`cluster_linear plan --model <GGUF> --tensor <name> --nodes <nodes.json>`
prints a static allocation. `nodes.json` is an array of `NodeCapacity` records
(`id`, `available_bytes`, `reserved_bytes`, `compute_weight`);
`--links <links.json>` accepts measured `LinkObservation` records (`from`, `to`,
`latency_seconds`, `bandwidth_bytes_per_second`). These records are operator
inputs, not automatically collected probes. Library callers can apply the
returned column ranges with `ShardedTensor::with_input_range`; the `rank`
example intentionally uses equal block partitions rather than consuming a plan.

### Operational limits and remaining work

Use an isolated, trusted LAN. Collective participants must share a fresh
session ID and a strong pre-shared key; authentication does not encrypt
activations or protect against a malicious key holder. Never reuse a session
ID after restarting ranks. mDNS advertisements are untrusted hints, not an
authorization or resource-verification mechanism.

A missing rank causes an error, never a silently incomplete activation sum.
After a failed collective, stop the entire job and restart all ranks with a new
session; dynamic membership/remapping is not implemented. A network partition
can cause participants to observe different success/failure outcomes.

These issues remain **partially addressed**: full-model forward/KV-cache
integration, authenticated cluster admission and live topology probes,
automatic model sync, FEC, draining/repartitioning, and hardware
acceptance tests are future work. Neither two-machine/100-node discovery,
10% real-network packet-loss tolerance, `/proc/self/smaps` scaling, nor the
80%-throughput and <10%-imbalance targets have been established. The planner
is a heuristic, not an optimal P99 scheduler.

Validation includes quantized shard/reference comparisons, deterministic
membership/election and memory-planning tests, real loopback UDP retry
tests with injected packet loss, and TCP ring tests. The example tests also
compare three TCP ranks' quantized partial matvecs with an unsharded reference:

```bash
cargo test -p joshua --features distributed --lib distributed
cargo test -p joshua --features distributed --example cluster_linear
```

The multicast integration test is explicitly
ignored by default and can be run on a multicast-enabled host:

```bash
cargo test -p joshua --features distributed --lib distributed::collective -- --ignored
```

The development sandbox rejected multicast sends with `EPERM`, including the
two-process example above; unicast loopback protocol tests do not establish
multicast interoperability or LAN throughput.

---

## Roadmap

- [x] Chat completions (non-streaming)
- [x] Chat completions (SSE streaming)
- [x] Legacy text completions
- [x] OpenAI-compatible model list
- [x] mmap-based model loading
- [x] Multi-architecture GGUF dispatch (all candle quantized loaders)
- [x] Per-model chat templates from GGUF metadata
- [x] Dense embeddings (llama / qwen2 / qwen3 embedding models, GGUF pooling metadata)
- [x] Tool / function calling (OpenAI-compatible, Hermes/Mistral/Llama-3 formats)
- [x] GPU acceleration (`cuda` / `metal` cargo features)
- [x] KV-cache sharing across requests (warm model pool with prefix reuse)
- [x] Speculative decoding (prompt-lookup drafts, lossless verification with KV rollback)
- [x] DeepSeek-V4 sparse-attention MoE loader (Hyper-Connections, CSA/HCA KV compression, Lightning Indexer, IQ2_XXS experts)
- [x] DeepSeek-V2/V3 MLA latent cache (~70× smaller KV cache, prefill == incremental)
- [x] Fused AVX2 k-quant kernels and SIMD quantized matmuls (CPU prefill/decode speed-ups)
- [x] AVX-512 backend: 16-lane fused kernels for every CPU quant format (k-quants, IQ2_XXS, MXFP4, Q1_0/Q2_0)
- [x] Sparse-MoE weight management (hot-weight pinning, mlock with memlock-limit check, prefill streaming)
- [x] Models larger than VRAM (host-resident experts with the dense set on the GPU, weights shared across sessions, quantized embedding table)
- [x] Vision / multimodal support (OpenAI image messages via llama.cpp `mtmd` through the plugin shim)
- [x] Speech-to-text (Whisper — pure-Rust pipeline, `/v1/audio/transcriptions`)
- [x] NPU backend architecture (isolated vendor-plugin shim + llama.cpp adapter for Hexagon/CANN/…)
- [x] Kimi-Linear and Kimi K3 (Kimi Delta Attention, attention residuals, latent MoE, `situ`, MXFP4 experts)

---

## License

MIT
