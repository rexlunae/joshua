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
| `POST /v1/chat/completions` | ✅ streaming (SSE) + non-stream, tools (`tool_choice` none / auto / required / a named function), `response_format`, `prompt_cache_key`, reasoning as `reasoning_content` / `reasoning` (server default `--reasoning-format`) |
| `POST /v1/completions` | ✅ raw prompt (no chat template): a string, token ids, or a batch of either (one choice each); SSE streaming, `echo`, `suffix` (fill-in-the-middle models), `response_format`, `add_special_tokens`. `logprobs`, `n > 1` and `best_of > 1` are refused |
| `POST /v1/embeddings` | ✅ `input` as text, token ids or an array of either; `encoding_format` `float` or `base64` (the SDKs' default); `dimensions` (a prefix, L2-normalized again: meaningful for Matryoshka models); vLLM's `add_special_tokens`. vLLM's `normalize: false` is refused |
| `GET /v1/models` (also `/models`), `GET /v1/models/{id}` | ✅ loaded models, plus the files the model directory holds (each with its limits; a request naming one loads it on first use) |
| `POST /v1/audio/transcriptions`, `/v1/audio/translations` | ✅ when the server has a `--whisper-model`; translations give English |
| `POST /v1/models/load` / `unload` | ✅ joshua extension (when a model manager is configured) |

Every SSE stream sends a keep-alive comment every 15 s while nothing else is
sent (a long prefill), so clients and proxies with read timeouts keep the
connection.  Behind `joshua route` the first comment opens the client's
stream, so a request is retried on another worker only if its worker fails
in its first 15 s.

This is the primary target and the README's "drop-in replacement" claim is
about these routes.

## Anthropic

| Endpoint | Status |
|---|---|
| `POST /v1/messages` | ✅ text, streaming, tool use |
| `x-api-key` authentication | ✅ on the server and on `joshua route` |
| `POST /v1/messages/count_tokens` | ✅ engine's own chat-token count, no model run |

## llama.cpp (`llama-server`)

| Endpoint | Status |
|---|---|
| `GET /health` (`/v1/health`) | ✅ |
| `POST /chat/completions` | ✅ the OpenAI chat route without `/v1` |
| `GET /metrics` | ✅ Prometheus text (joshua's own counter names; JSON at `/metrics/json`) |
| `POST /completion` (`/completions`) | ✅ raw prompt: a string, token ids, a mix of both, or an array of those (an array of results); SSE streaming without `[DONE]`; `json_schema` / `response_format`; `return_tokens`; llama-server's result fields (`stop_type`, `stopping_word`, `tokens_cached`, `timings`, `generation_settings`). `grammar` (GBNF), `n_probs`, per-request `lora` and multimodal prompts are refused |
| `POST /infill` | ✅ `input_prefix`, `input_suffix`, `input_extra`, `prompt`, in the model's FIM format (refused for a model without FIM tokens) |
| `POST /embedding` (`/embeddings`) | ✅ `content` or `input`; `[{"index", "embedding": [[…]]}]`, one pooled L2-normalized vector per input (`embd_normalize` other than 2 is refused) |
| `POST /apply-template` | ✅ the rendered chat prompt, `{"prompt": …}` |
| `GET /props` | ✅ the model's real chat template, `default_generation_settings.params` under llama-server's names, `total_slots`, `build_info` |
| `GET /slots` | ✅ one record per loaded model |
| `POST /tokenize` | ✅ `with_pieces` (a piece is text, or its bytes when not whole UTF-8); `add_special` / `add_special_tokens`; `/api/tokens` too |
| `POST /detokenize` | ✅ answers with `content` (llama.cpp's key) and `prompt` (vLLM's) |

Where joshua's default differs from llama-server's: `/tokenize` adds the
tokenizer's special tokens unless told not to.  The routes take an optional
`model` to pick one of several loaded models.  Field-for-field parity with a
specific `llama-server` build has not been diffed.

## vLLM

vLLM *serves* the OpenAI API, so OpenAI compatibility (above) already covers the
bulk of it.  Of vLLM's own additions:

| Endpoint | Status |
|---|---|
| `GET /health` | ✅ |
| `GET`/`POST /ping` | ✅ 200, no body |
| `GET /version` | ✅ `{"version": …}` |
| `GET /metrics` | ✅ Prometheus text (joshua's own counter names, not `vllm:`) |
| `GET /load` | ✅ `server_load`, plus running/waiting/capacity/free |
| `POST /v1/tokenize` | ✅ `prompt` or the chat form (`messages`, `tools`, `add_generation_prompt`); `return_token_strs`; `max_model_len` |
| `POST /v1/detokenize` | ✅ |
| `GET /v1/models` | ✅ |
| `POST /v1/score`, `/score` | ❌ **by design** — vLLM's scoring API is for *score models* (cross-encoder, bi-encoder, late-interaction). joshua runs generative models; there is no scoring head to call. |
| `/v1/rerank`, `/rerank`, `/v2/rerank` | ❌ same reason (pooling/rerank models) |
| `/classify`, `/pooling` | ❌ same reason |
| `/v1/load_lora_adapter`, `/v1/unload_lora_adapter` | ❌ no LoRA adapter runtime |
| `/v1/responses`, `/v1/chat/completions/batch` | ❌ not implemented |

**Honest summary:** vLLM *generative* serving is compatible; vLLM *pooling /
scoring / rerank* serving is not, and adding it means running a different class
of model, not adding a route.

## Ollama

| Endpoint | Status |
|---|---|
| `HEAD /`, `GET /` | ✅ the heartbeat the `ollama` CLI sends before every command |
| `GET /api/version` | ✅ |
| `GET /api/tags` | ✅ |
| `GET /api/ps` | ✅ |
| `POST /api/show` | ✅ |
| `POST /api/chat` | ✅ newline-delimited JSON streaming, tool calls, `format`, images, `keep_alive` |
| `POST /api/generate` | ✅ `format`, images, `keep_alive`; `system`, `raw`, `context` are not yet honoured |
| `POST /api/embed` | ✅ |
| `POST /api/embeddings` | ✅ legacy shape: one `prompt` in, one `embedding` out |
| `POST /api/tokens` | ✅ (joshua extension, mirrors `/tokenize`) |
| `DELETE /api/delete` | ✅ removes `<name>.gguf` from the model directory |
| `POST /api/copy` | ✅ |
| `POST /api/create` (`from:`) | ✅ derives from an existing model |
| `POST /api/create` (`files:`) | ❌ rejected explicitly — no blob store |
| `POST /api/pull` | ❌ not implemented — see below |
| `POST /api/push` | ❌ no registry |

Generation `options` honoured on `/api/chat` and `/api/generate`:
`num_predict`, `temperature`, `top_k`, `top_p`, `min_p`, `typical_p`,
`repeat_penalty`, `repeat_last_n`, `presence_penalty`, `frequency_penalty`,
`mirostat`, `mirostat_tau`, `mirostat_eta`, `seed` and `stop`.  `num_ctx`
narrows the context window, clamped to the server's `--n-ctx` (Ollama treats
it as a load setting and clients send their own default, so a larger value is
not an error); other options are ignored.

What the rest of a request and its answer carry:

* **Names**: `name:tag` resolves to the model whose id is the name, else to
  the one whose id is the name without its tag, so the `tiny:latest` that
  `/api/tags` lists addresses `tiny.gguf`.  A model in the model directory is
  loaded by the first request that names it.
* **Listings**: `modified_at` and `created_at` are RFC 3339 times; `digest` is
  a 64-hex-digit SHA-256 of the model's id, path, size and modification time
  (not of its contents, so listing a directory reads no weights); `details`
  include `parameter_size`.  `/api/ps` adds `context_length` and `size_vram`
  (the whole file when the model runs on an accelerator; its dense weights plus
  its expert cache when the routed experts stay in host memory; 0 on the CPU), and reports no expiry.
* **`/api/show`** returns the GGUF header facts under their metadata keys
  (`model_info`), and for a loaded model its chat template, its default
  sampling parameters and its `capabilities` (`completion`, plus `tools` when
  its template reads `tools` or it has none and uses the ChatML fallback,
  `insert`, `thinking` and `embedding` where they apply).
* **Done object**: `done_reason` (`stop` or `length`), `prompt_eval_count`,
  `eval_count`, and `total_duration`, `load_duration`, `prompt_eval_duration`
  and `eval_duration` in nanoseconds.
* **Tool calls** are read by the same parsers as the OpenAI routes and sent as
  `message.tool_calls`, each with its arguments as an object, in one chunk
  before the done object.  Earlier calls and `tool` results (`tool_name`) are
  accepted in the history.
* **`format`**: `"json"` constrains the output to a JSON object, a schema to
  documents valid under it.
* **`images`**: base64 strings, passed to a media-capable backend as the
  OpenAI routes' `data:` URLs are.
* **`keep_alive`**: `0` (or `"0s"`, `"0m"`) unloads the model once the answer
  is sent (or the request fails), when the server manages a model directory.  A request with no
  messages or prompt only loads the model (`done_reason: "load"`), or with
  `keep_alive: 0` unloads it (`"unload"`).  Other values are accepted and
  change nothing: a model stays loaded until it is unloaded.
* **Errors** are `{"error": "<message>"}`, before a stream starts where the
  request can be checked up front, and as a final `{"error": ...}` line when
  generation fails mid-stream.

### Why `pull` is not simply "missing"

Ollama's `pull` is not a file download: it fetches a **manifest** from
`registry.ollama.ai`, then layers addressed by SHA256 digest, resuming partial
downloads across concurrent callers. Supporting `ollama pull llama3` means
implementing that registry protocol, not adding a route. The endpoint returns
guidance rather than hanging:

> cannot pull `llama3`: joshua does not implement the Ollama registry protocol.
> Place the .gguf file in the model directory and load it with
> POST /v1/models/load.

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
| Translation layer (Ollama option mapping, name tagging, path confinement) | ✅ unit tests in `src/ollama.rs` |
| **A real `ollama` / `llama-server` / vLLM client against the server** | ❌ **not tested** |

The last row is the real gap. The endpoints look right, but no test drives them
with an actual client binary, so field-level differences (a required key a
client reads, an error shape, a streaming-term detail) are unverified. That is
the next piece of work if "drop-in" is to mean more than "the route exists".
