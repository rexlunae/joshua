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
| **mmap loading** | The GGUF file is memory-mapped like llama.cpp: weights page in lazily and stay in the OS page cache |
| **OpenAI-compatible** | Drop-in replacement for `/v1/chat/completions`, `/v1/embeddings`, `/v1/models` |
| **Streaming** | Server-Sent Events (SSE) for token-by-token streaming |
| **GGUF support** | Llama/Mistral/Mixtral, Gemma 1–3, GLM-4, LFM2, Phi-2, Phi-3, Qwen2, Qwen3, Qwen3-MoE |
| **Chat templates** | Renders the model's own `tokenizer.chat_template` from the GGUF (Jinja via pure-Rust minijinja); ChatML fallback |
| **Tool calling** | OpenAI-compatible `tools` / `tool_calls`, parsing Hermes/Qwen, Mistral, and Llama-3 call formats |
| **Embeddings** | Dense sentence embeddings for llama / qwen2 / qwen3 embedding models, with GGUF pooling metadata |
| **KV-cache reuse** | Multi-turn requests continue from a warm model pool and prefill only the new suffix |
| **GPU (optional)** | `--features cuda` or `metal` route inference through candle's GPU backends |
| **NPU / llama.cpp interop (optional)** | Vendor plugins run in a crash-isolated shim process; a llama.cpp adapter brings every ggml backend (Hexagon NPU, CANN, CUDA, Vulkan, …) |
| **Sampling** | Temperature, top-k, min-p, top-p (nucleus), greedy — all in Rust |

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
| Rust toolchain | 1.75 |

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

# Start the API server
./target/release/joshua serve \
    --model ./weights/gemma-3-1b-it-Q4_K_M.gguf \
    --addr 0.0.0.0:8080
```

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

## Environment variables

| Variable | Description |
|---|---|
| `JOSHUA_MODEL_PATH` | Default model path (overrides `--model` flag) |
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

Example models:

- `google/gemma-3-270m-it` / `1b-it` / `4b-it`
- `Qwen/Qwen3-0.6B` / `1.7B`
- `LiquidAI/LFM2-1.2B`
- `microsoft/Phi-3-mini-4k-instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `THUDM/GLM-4-9B-0414`

Every other architecture name in llama.cpp's registry (Mamba, RWKV, GPT-2,
DeepSeek, Granite, OLMo, StarCoder2, and ~70 more) is recognised at load time
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
the vendor SDK (see llama.cpp's Snapdragon/CANN backend docs).

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
- [ ] Vision / multimodal support (needs a quantized vision encoder + projector pipeline)
- [ ] Speech-to-text (Whisper — candle has the model; needs an audio ingest + mel pipeline)
- [x] NPU backend architecture (isolated vendor-plugin shim + llama.cpp adapter for Hexagon/CANN/…)

---

## License

MIT
