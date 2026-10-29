# Client compatibility matrix

What joshua's HTTP server supports for each client ecosystem, verified against
the routes registered in `src/server.rs` and `src/ollama.rs` on `main`.

In every row the answer is **"verified in code"** (the route exists and is
wired to the engine) or a stated reason it does not.  Behavioural parity with a
real client binary is *not* covered by the test suite today — see
[Verification status](#verification-status).

---

## OpenAI

| Endpoint | Status |
|---|---|
| `POST /v1/chat/completions` | ✅ streaming (SSE) + non-stream, tools, `response_format` |
| `POST /v1/completions` | ✅ legacy text completion |
| `POST /v1/embeddings` | ✅ |
| `GET /v1/models` | ✅ |
| `POST /v1/models/load` / `unload` | ✅ joshua extension (when a model manager is configured) |

This is the primary target and the README's "drop-in replacement" claim is
about these routes.

## Anthropic

| Endpoint | Status |
|---|---|
| `POST /v1/messages` | ✅ text, streaming, tool use |
| `POST /v1/messages/count_tokens` | ✅ engine's own chat-token count, no model run |

## llama.cpp (`llama-server`)

| Endpoint | Status |
|---|---|
| `GET /health` | ✅ |
| `GET /props` | ✅ properties + `default_generation_settings` |
| `GET /slots` | ✅ one record per loaded model |
| `POST /tokenize` | ✅ (`/api/tokens` too) |
| `POST /detokenize` | ✅ via `POST /v1/detokenize` |

`/props` returns the fields clients read for defaults (`n_ctx`,
`default_generation_settings`, `total_slots`).  Field-for-field parity with a
specific `llama-server` build has not been diffed.

## vLLM

vLLM *serves* the OpenAI API, so OpenAI compatibility (above) already covers the
bulk of it.  Of vLLM's own additions:

| Endpoint | Status |
|---|---|
| `GET /health` | ✅ |
| `GET /load` | ✅ running/waiting/capacity/free |
| `POST /v1/tokenize` | ✅ |
| `POST /v1/detokenize` | ✅ |
| `GET /v1/models` | ✅ |
| `POST /v1/score`, `/score` | ❌ **by design** — vLLM's scoring API is for *score models* (cross-encoder, bi-encoder, late-interaction). joshua runs generative models; there is no scoring head to call. |
| `/v1/rerank`, `/rerank`, `/v2/rerank` | ❌ same reason (pooling/rerank models) |
| `/classify`, `/pooling` | ❌ same reason |
| `/v1/load_lora_adapter`, `/v1/unload_lora_adapter` | ❌ no LoRA adapter runtime |
| `/v1/responses`, `/v1/chat/completions/batch` | ❌ not implemented |
| `/v1/audio/translations` | ❌ (transcriptions are supported) |

**Honest summary:** vLLM *generative* serving is compatible; vLLM *pooling /
scoring / rerank* serving is not, and adding it means running a different class
of model, not adding a route.

## Ollama

| Endpoint | Status |
|---|---|
| `GET /api/version` | ✅ |
| `GET /api/tags` | ✅ |
| `GET /api/ps` | ✅ |
| `POST /api/show` | ✅ |
| `POST /api/chat` | ✅ newline-delimited JSON streaming |
| `POST /api/generate` | ✅ |
| `POST /api/embed`, `/api/embeddings` | ✅ |
| `POST /api/tokens` | ✅ (joshua extension, mirrors `/tokenize`) |
| `DELETE /api/delete` | ✅ removes `<name>.gguf` from the model directory |
| `POST /api/copy` | ✅ |
| `POST /api/create` (`from:`) | ✅ derives from an existing model |
| `POST /api/create` (`files:`) | ❌ rejected explicitly — no blob store |
| `POST /api/pull` | ❌ not implemented — see below |
| `POST /api/push` | ❌ no registry |

### Why `pull` is not simply "missing"

Ollama's `pull` is not a file download: it fetches a **manifest** from
`registry.ollama.ai`, then layers addressed by SHA256 digest, resuming partial
downloads across concurrent callers. Supporting `ollama pull llama3` means
implementing that registry protocol, not adding a route. The endpoint returns
guidance rather than hanging:

> cannot pull `llama3`: joshua does not implement the Ollama registry protocol.
> Pass an explicit https:// URL to a .gguf file, or place the file in the model
> directory and load it with POST /v1/models/load.

Explicit-URL downloading is itself still unimplemented (returns 501); adding an
HTTP client is a separate decision because it introduces outbound network
access into the server.

### Management-endpoint safety

`delete`, `copy` and `create` **mutate the filesystem**, so names are confined
to the model directory: `..`, absolute paths and bare path separators are
rejected, and a name that canonicalises outside the directory (via a symlink)
is refused.  `llama3:13b` and `llama3:latest` both address `llama3.gguf` —
Ollama's `:tag` names a manifest, which has no joshua equivalent.

## Verification status

| What | Covered? |
|---|---|
| Route exists and is wired | ✅ by inspection + the inline unit tests |
| Translation layer (Ollama option mapping, name tagging, path confinement) | ✅ 7 unit tests in `src/ollama.rs` |
| **A real `ollama` / `llama-server` / vLLM client against the server** | ❌ **not tested** |

The last row is the real gap. The endpoints look right, but no test drives them
with an actual client binary, so field-level differences (a required key a
client reads, an error shape, a streaming-term detail) are unverified. That is
the next piece of work if "drop-in" is to mean more than "the route exists".
