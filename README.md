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
| **GGUF support** | Llama/Mistral/Mixtral, Gemma 1–3, GLM-4, LFM2, Phi-2, Phi-3, Qwen2, Qwen3, Qwen3-MoE, DeepSeek-V2/V3, DeepSeek-V4, Kimi-K2 |
| **Exotic quant dtypes** | In-mapping decoders for IQ2_XXS (DeepSeek-V4's 2.0625-bit expert weights) and MXFP4 (Kimi-K3-class), with matmuls that keep the blocks in the mmap instead of materialising f32 |
| **Fused SIMD kernels** | AVX2 dequant+dot fusion for Q8_0/Q2_K/Q4_K and parallel SIMD quantized matmuls on x86-64 |
| **Chat templates** | Renders the model's own `tokenizer.chat_template` from the GGUF (Jinja via pure-Rust minijinja); ChatML fallback |
| **Tool calling** | OpenAI-compatible `tools` / `tool_calls`, parsing Hermes/Qwen, Mistral, and Llama-3 call formats |
| **Embeddings** | Dense sentence embeddings for llama / qwen2 / qwen3 embedding models, with GGUF pooling metadata |
| **KV-cache reuse** | Multi-turn requests continue from a warm model pool and prefill only the new suffix; DeepSeek-V2/V3 MLA caches the compressed latent (`c_kv` + `k_pe`) instead of the reconstructed per-head K/V, cutting KV memory ~70× |
| **GPU (optional)** | `--features cuda` or `metal` route inference through candle's GPU backends |
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
└──────────────────────────┘    Llama / Gemma / GLM-4 / LFM2 / Phi / Qwen loaders
           │
┌──────────────────────────┐
│  tokenizers (pure Rust)  │  ← BPE tokenisation from tokenizer.json
└──────────────────────────┘    HuggingFace tokenizers library
```

---

## Requirements

| Tool | Minimum version |
|---|---|
| Rust toolchain | 1.88 |

The minimum is set by the dependency tree — currently `zip 8.6.0`, which
requires Rust 1.88 (the table previously claimed 1.87 via `candle-core`'s use
of `{integer}::is_multiple_of`, but the tree has moved on).

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

### 5 — GPU acceleration (Metal / CUDA)

Joshua runs inference on the CPU by default, but a single build can also run
on a GPU, chosen per invocation.  Add the backend feature **at build time**
(candle compiles the kernels in):

```bash
# Apple Silicon Mac: Metal
cargo build --release --features metal

# NVIDIA GPU: CUDA (needs a CUDA toolkit; Linux/Windows)
cargo build --release --features cuda
```

Then pick the device at runtime — `auto` (the default) uses the best backend
this build was compiled with and degrades to CPU with a warning; an explicit
request is strict and fails the load if the device is missing:

```bash
./target/release/joshua serve --model m.gguf --device metal      # force Metal
./target/release/joshua serve --model m.gguf --device cpu        # force CPU
./target/release/joshua serve --model m.gguf --device auto       # default
```

`--device` also accepts `cuda`, and the same selection is available as
`JOSHUA_DEVICE` for the server.  The resolved device is logged at startup;
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

**What stays on CPU.** `deepseek4` (DeepSeek-V4-Flash) refuses a GPU device
at load: its IQ2_XXS expert weights have no Metal/CUDA kernel and the
loader degrades to a clear error rather than materialising ~16× f32.
Whisper transcription is CPU-only.

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
| `JOSHUA_MAX_CONCURRENCY` | Cap on simultaneous generations/embeddings (same as `--max-concurrency`) |
| `JOSHUA_MAX_OUTPUT_TOKENS` | Hard ceiling on generated tokens per request (same as `--max-output-tokens`) |
| `JOSHUA_WHISPER_MODEL` | Whisper model directory mounted at `/v1/audio/transcriptions` (same as `--whisper-model`) |
| `JOSHUA_NPU_PLUGIN` | NPU vendor plugin path (same as `--npu-plugin`) |
| `JOSHUA_LLAMA_N_GPU_LAYERS` | llama.cpp adapter layer offload count (default: all) |
| `JOSHUA_LLAMA_MMPROJ` | Multimodal projector GGUF for vision via llama.cpp's `mtmd` |
| `JOSHUA_LLAMA_BACKENDS_DIR` | Directory of `libggml-<name>` modules to register at adapter startup (llama.cpp `dynamic-backends` builds) |
| `RUST_LOG` | Log filter (e.g. `info`, `joshua=debug`) |

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
| `glm4` | GLM-4 (dense) |
| `lfm2` | Liquid LFM2 |
| `phi2` | Phi-1, Phi-1.5, Phi-2 |
| `phi3` | Phi-3, Phi-3.5 |
| `qwen2` | Qwen1.5, Qwen2, Qwen2.5 |
| `qwen3` | Qwen3 (dense) |
| `qwen3moe` | Qwen3 mixture-of-experts |
| `deepseek2` | DeepSeek-V2, DeepSeek-V3, **Kimi-K2** (MLA attention + fine-grained MoE) |
| `deepseek4` | DeepSeek-V4 (Hyper-Connections residual mixing, alternating sliding-window / learned KV-compressor attention, Lightning-Indexer sparse attention, fine-grained MoE with IQ2_XXS experts) |

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
against llama.cpp.

The `deepseek4` loader handles the architecture's three additions over V3:
Hyper-Connections mix the residual stream to `hc_mult` parallel copies with
Sinkhorn-normalised per-token weights, CSA/HCA compressor layers pool blocks of
4 / 128 tokens into a compressed KV, and the Lightning Indexer picks the
`index_topk` compressed positions each query attends to. The routed experts
stay quantized as IQ2_XXS trellis blocks decoded in-mapping during the matmul,
so a 162 B model keeps its on-disk footprint. Activations run in f32 on CPU.

**Kimi-K3 (in progress).** The correctness-critical primitives — Kimi Delta
Attention (per-channel decay gates), attention residuals, the `situ`
activation, and an MXFP4 decoder — are implemented in `kimi_k3.rs` and
unit-tested against the reference formulations, but a full forward pass is not
wired up yet: no released llama.cpp runs K3, and no weights were available to
check end-to-end logits against. It is not yet dispatchable from GGUF
metadata.

Example models:

- `google/gemma-3-270m-it` / `1b-it` / `4b-it`
- `Qwen/Qwen3-0.6B` / `1.7B`
- `LiquidAI/LFM2-1.2B`
- `microsoft/Phi-3-mini-4k-instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `THUDM/GLM-4-9B-0414`
- `deepseek-ai/DeepSeek-V2-Lite`, `moonshotai/Kimi-K2-Instruct` (as GGUF)
- `deepseek-ai/DeepSeek-V4-Flash-162B` (as GGUF)

Every other architecture name in llama.cpp's registry (Mamba, RWKV, GPT-2,
DeepSeek-V1, Granite, OLMo, StarCoder2, and ~70 more) is recognised at load time
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
- [x] DeepSeek-V4 sparse-attention MoE loader (Hyper-Connections, CSA/HCA KV compression, Lightning Indexer, IQ2_XXS experts)
- [x] DeepSeek-V2/V3 MLA latent cache (~70× smaller KV cache, prefill == incremental)
- [x] Fused AVX2 k-quant kernels and SIMD quantized matmuls (CPU prefill/decode speed-ups)
- [x] Sparse-MoE weight management (hot-weight pinning, mlock with memlock-limit check, prefill streaming)
- [x] Vision / multimodal support (OpenAI image messages via llama.cpp `mtmd` through the plugin shim)
- [x] Speech-to-text (Whisper — pure-Rust pipeline, `/v1/audio/transcriptions`)
- [x] NPU backend architecture (isolated vendor-plugin shim + llama.cpp adapter for Hexagon/CANN/…)
- [ ] Kimi-K3 full forward pass (primitives done: Kimi Delta Attention, attention residuals, `situ`, MXFP4 — see Supported models)

---

## License

MIT
