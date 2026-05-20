# joshua

An mmap-based high-speed LLM executor for the Rust ecosystem — a Rust clone of [Cactus](https://github.com/cactus-compute/cactus).

---

## Features

| Feature | Details |
|---|---|
| **Fast** | Delegates to llama.cpp — the fastest CPU inference engine on the market |
| **Low RAM** | Model weights are memory-mapped (mmap) so only the pages you touch live in RAM |
| **OpenAI-compatible** | Drop-in replacement for `/v1/chat/completions`, `/v1/embeddings`, `/v1/models` |
| **Streaming** | Server-Sent Events (SSE) for token-by-token streaming |
| **Pure Rust API** | Idiomatic `Engine` type — no C/C++ in your call stack |
| **GGUF support** | Any GGUF model that llama.cpp supports (Llama, Gemma, Qwen, Mistral, …) |

---

## Architecture

```
┌──────────────────────────┐
│  Joshua  (Rust crate)    │  ← OpenAI-compatible REST API
└──────────────────────────┘    Chat, embeddings, streaming, tool stubs
           │
┌──────────────────────────┐
│  llama-cpp-2  (FFI)      │  ← Safe Rust wrapper over llama.cpp
└──────────────────────────┘    GGUF model loading, KV-cache, batching
           │
┌──────────────────────────┐
│  llama.cpp  (C++)        │  ← Compiled natively during `cargo build`
└──────────────────────────┘    ARM NEON / AVX2 / Metal / CUDA kernels
```

---

## Requirements

| Tool | Minimum version |
|---|---|
| Rust toolchain | 1.75 |
| CMake | 3.14 |
| C++17 compiler | GCC 10 / Clang 12 / MSVC 2019 |

---

## Quick start

### 1 — Add to `Cargo.toml`

```toml
[dependencies]
joshua = { git = "https://github.com/rexlunae/joshua" }
```

### 2 — Download a model

Any GGUF file works.  A tiny starting point:

```bash
# Using the Hugging Face CLI
pip install huggingface-hub
huggingface-cli download \
    bartowski/google_gemma-3-1b-it-GGUF \
    gemma-3-1b-it-Q4_K_M.gguf \
    --local-dir ./weights
```

### 3 — Library usage

```rust
use joshua::{Engine, GenerationOptions, ChatMessage};

fn main() -> anyhow::Result<()> {
    let engine = Engine::new("./weights/gemma-3-1b-it-Q4_K_M.gguf")?;

    let messages = vec![ChatMessage {
        role:    "user".to_string(),
        content: "What is Rust?".to_string(),
        images:  None,
        name:    None,
    }];

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
# Build
cargo build --release

# One-shot completion
./target/release/joshua run \
    --model ./weights/gemma-3-1b-it-Q4_K_M.gguf \
    "Explain memory-mapped I/O in one paragraph"

# Start the API server
./target/release/joshua serve \
    --model ./weights/gemma-3-1b-it-Q4_K_M.gguf \
    --addr 0.0.0.0:8080

# Embed text
./target/release/joshua embed \
    --model ./weights/nomic-embed-text-v2.gguf \
    "Hello, world"
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
| `repetition_penalty` | `f32` | `1.1` | Penalise recent tokens (1.0 = off) |
| `stop_sequences` | `Vec<String>` | `[]` | Stop on these strings |

---

## Supported models

Any GGUF model supported by llama.cpp works.  Tested models include:

- `google/gemma-3-270m-it` / `1b-it` / `4b-it`
- `Qwen/Qwen3-0.6B` / `1.7B`
- `LiquidAI/LFM2.5-1.2B-Instruct`
- `microsoft/Phi-3-mini-4k-instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- Embedding models: `nomic-ai/nomic-embed-text-v2-moe`

---

## Roadmap

- [x] Chat completions (non-streaming)
- [x] Chat completions (SSE streaming)
- [x] Legacy text completions
- [x] Dense embeddings
- [x] OpenAI-compatible model list
- [ ] Vision / multimodal support
- [ ] Speech-to-text (Whisper)
- [ ] Tool / function calling
- [ ] Cloud fallback (auto-handoff when confidence is low)
- [ ] KV-cache quantisation
- [ ] NPU support (Apple Neural Engine, Snapdragon HTP)

---

## License

MIT
