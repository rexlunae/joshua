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
| **OpenAI-compatible** | Drop-in replacement for `/v1/chat/completions`, `/v1/embeddings`, `/v1/models` |
| **Streaming** | Server-Sent Events (SSE) for token-by-token streaming |
| **GGUF support** | Llama, Gemma, Qwen, Mistral, and any other Llama-architecture GGUF model |
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
└──────────────────────────┘    quantized_llama: Llama / Mistral / Gemma / Qwen
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

Any GGUF model that follows the Llama architecture works.  You also need the
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

> **Note:** Embeddings require a dedicated pooling model.  Standard language
> models return a descriptive error explaining what is needed.

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

Any GGUF model with a Llama-compatible architecture.  Tested models include:

- `google/gemma-3-270m-it` / `1b-it` / `4b-it`
- `Qwen/Qwen3-0.6B` / `1.7B`
- `LiquidAI/LFM2.5-1.2B-Instruct`
- `microsoft/Phi-3-mini-4k-instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

---

## Roadmap

- [x] Chat completions (non-streaming)
- [x] Chat completions (SSE streaming)
- [x] Legacy text completions
- [x] OpenAI-compatible model list
- [ ] Dense embeddings (requires pooling model)
- [ ] Vision / multimodal support
- [ ] Speech-to-text (Whisper)
- [ ] Tool / function calling
- [ ] GPU acceleration (via candle CUDA/Metal features)
- [ ] KV-cache sharing across requests
- [ ] NPU support (Apple Neural Engine, Snapdragon HTP)

---

## License

MIT
