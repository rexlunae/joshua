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
| `POST /v1/chat/completions` | ✅ streaming (SSE) + non-stream, tools (`tool_choice` none / auto / required / a named function), `response_format`, `logprobs` / `top_logprobs`, vLLM's `prompt_logprobs` (not streamed, as vLLM), `prompt_cache_key`, reasoning as `reasoning_content` / `reasoning` (server default `--reasoning-format`) |
| `POST /v1/completions` | ✅ raw prompt (no chat template): a string, token ids, or a batch of either (one choice each); SSE streaming, `echo`, `suffix` (fill-in-the-middle models), `response_format`, `add_special_tokens`, `logprobs` (the generated tokens', and with `echo` the prompt's too), vLLM's `prompt_logprobs` (not streamed, as vLLM). `n > 1` and `best_of > 1` are refused |
| `POST /v1/responses` (also `/responses`) | ✅ `input` as a string or items (messages with text and images, `function_call`, `function_call_output`, `reasoning`), `instructions`, function `tools` and `tool_choice`, `text.format`, `reasoning.effort`, `max_output_tokens`, `previous_response_id`, `metadata`, `parallel_tool_calls: false` (the first call only); output as `reasoning`, `message` and `function_call` items; SSE streaming with the typed `response.*` events. Responses are kept (unless `store: false`) in a bounded in-memory store: the newest 1024, up to 64 MiB, until restart (one response past 64 MiB on its own is not kept). `background`, `conversation`, hosted tools, `item_reference`, file inputs, `truncation: "auto"`, `top_logprobs` and `max_tool_calls` are refused |
| `GET`/`DELETE /v1/responses/{id}`, `GET /v1/responses/{id}/input_items` | ✅ from the store (`order`, `limit`, `after`); `POST /v1/responses/{id}/cancel` answers that only background responses can be cancelled |
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
| `POST /completion` (`/completions`) | ✅ raw prompt: a string, token ids, a mix of both, or an array of those (an array of results); SSE streaming without `[DONE]`; `json_schema` / `response_format`; `return_tokens`; `n_probs` (`completion_probabilities` with `logprob` and `top_logprobs`); llama-server's result fields (`stop_type`, `stopping_word`, `tokens_cached`, `timings`, `generation_settings`). per-request `lora` (`[{"id", "scale"}]`, unlisted adapters at 0). `grammar` (GBNF), `post_sampling_probs` and multimodal prompts are refused |
| `POST /infill` | ✅ `input_prefix`, `input_suffix`, `input_extra`, `prompt`, in the model's FIM format (refused for a model without FIM tokens) |
| `POST /embedding` (`/embeddings`) | ✅ `content` or `input`; `[{"index", "embedding": [[…]]}]`, one pooled L2-normalized vector per input (`embd_normalize` other than 2 is refused) |
| `POST /rerank` (`/reranking`, `/v1/rerank`, `/v1/reranking`) | ✅ `query`, `documents` (or `texts`), `top_n`, with a generative reranker such as Qwen3-Reranker (see vLLM below); a model without `yes` and `no` tokens is refused |
| `POST /apply-template` | ✅ the rendered chat prompt, `{"prompt": …}` |
| `GET /props` | ✅ the model's real chat template, `default_generation_settings.params` under llama-server's names, `total_slots`, `build_info` |
| `GET /slots` | ✅ one record per loaded model |
| `GET /lora-adapters` | ✅ `[{"id", "path", "scale", "task_name", "prompt_prefix"}]` for the adapters loaded with `--lora` / `--lora-scaled` (`?model=` picks a model other than the default) |
| `POST /lora-adapters` | ✅ `[{"id", "scale"}]` sets the server-wide scales; adapters left out go to 0 |
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
| `POST /v1/score`, `/score` | ✅ with a generative reranker (Qwen3-Reranker and its fine-tunes): `text_1` against each `text_2`, or pair by pair; `instruction`. Cross-encoder and late-interaction score models are not supported |
| `POST /v1/rerank`, `/rerank`, `/v2/rerank` | ✅ with a generative reranker: `query`, `documents` (strings or `{"text"}`), `top_n`, `return_documents`, `instruction`; results sorted by `relevance_score` |
| `/classify`, `/pooling` | ❌ **by design** — they are for pooling models with a classification or pooling head, which joshua does not run |
| `/v1/load_lora_adapter`, `/v1/unload_lora_adapter` | ✅ with `--allow-runtime-lora` (vLLM's `VLLM_ALLOW_RUNTIME_LORA_UPDATING`): `{"lora_name", "lora_path", "load_inplace"}` and `{"lora_name"}`, answered with vLLM's `Success: ...` text; GGUF adapters only. Without the flag both are `404` |
| adapter-named models | ✅ `/v1/models` lists each loaded adapter with `parent` set to its base; naming one as `model` on `/v1/chat/completions`, `/v1/completions` or `/v1/responses` runs it at scale 1 |
| `/v1/responses` | ✅ see OpenAI above |
| `/v1/chat/completions/batch` | ❌ not implemented |

**Honest summary:** vLLM *generative* serving is compatible.  Scoring and
reranking work with a *generative* reranker, which answers a yes/no
relevance question (the score is the chance it says `yes`); cross-encoder
rerankers with a classification head, and the pooling routes, are not
supported.

## Ollama

| Endpoint | Status |
|---|---|
| `HEAD /`, `GET /` | ✅ the heartbeat the `ollama` CLI sends before every command |
| `GET /api/version` | ✅ |
| `GET /api/tags` | ✅ |
| `GET /api/ps` | ✅ |
| `POST /api/show` | ✅ |
| `POST /api/chat` | ✅ newline-delimited JSON streaming, tool calls, `format`, images, `keep_alive`; `think` (`true`, `false` or `"low"`/`"medium"`/`"high"`) with the reasoning under `message.thinking`; `logprobs` and `top_logprobs` |
| `POST /api/generate` | ✅ `format`, images, `keep_alive`, `system`, `raw`, `suffix` (fill-in-the-middle, for a model with FIM tokens), `context` (given back on the done object unless the prompt was raw, an infill or had images), `think` (reasoning under `thinking`), `logprobs` and `top_logprobs`. `template` is refused: joshua cannot run Ollama's Go templates |
| `POST /api/embed` | ✅ |
| `POST /api/embeddings` | ✅ legacy shape: one `prompt` in, one `embedding` out |
| `POST /api/tokens` | ✅ (joshua extension, mirrors `/tokenize`) |
| `DELETE /api/delete` | ✅ removes `<name>.gguf` from the model directory |
| `POST /api/copy` | ✅ |
| `POST /api/create` (`from:`) | ✅ derives from an existing model |
| `POST /api/create` (`files:`) | ✅ places one uploaded GGUF blob as `<model>.gguf`; Modelfile settings (`template`, `system`, `parameters`, `messages`, `adapters`, `quantize`) are refused |
| `HEAD`/`POST /api/blobs/{digest}` | ✅ uploads checked against their SHA-256, kept in `<model dir>/.blobs` |
| `POST /api/pull` | ✅ from registry.ollama.ai, another registry speaking its protocol (`hf.co/...`), or an `https://` `.gguf` URL — see below |
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

### `pull`

`pull` reads a name the way Ollama does, `[host/][namespace/]model[:tag]`
(`registry.ollama.ai`, `library` and `latest` by default), fetches the
manifest and downloads its model layer, checking it against the layer's
SHA-256 digest before it is placed as `<model>.gguf` (the tag and namespace
are dropped, as everywhere on the Ollama routes).  A blob already in the blob
store is reused.  The manifest's template, parameters and system layers are
not used, as joshua renders the GGUF's own template; projector and adapter
layers are skipped with a status line saying so.  An `https://` URL ending in
`.gguf` is downloaded as it is, and must start with GGUF's magic.  Plain
`http://` needs `insecure: true`.  Progress streams as Ollama's
newline-delimited status lines (`stream: false` answers once, at the end).
Interrupted downloads start again rather than resuming.

A pull connects to public addresses only: a name that resolves to one of this host's own addresses (public ones included),
a private network or a link-local address (checked at every connection,
redirects included) is refused, so the route cannot be used to reach
services behind the server.  Start the server with `--allow-private-pull`
(`JOSHUA_ALLOW_PRIVATE_PULL`) to pull from a registry of your own on the
LAN.  `insecure` never takes registry.ollama.ai or hf.co off https.

The download half sits behind the `pull` cargo feature, on by default; a
server built without it answers `pull` with a 501 and the advice to place
the file in the model directory.  Like the other management routes, `pull`,
`create` and the blob routes need a server that manages a model directory.

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

## LoRA adapters

`--lora FNAME` (scale 1) and `--lora-scaled FNAME SCALE` load GGUF LoRA
adapters, as llama.cpp's `convert_lora_to_gguf.py` writes them, for the
startup model; `--lora-init-without-apply` loads them at scale 0.  Each is
validated against the model's projections when it loads, and refused when it
targets one the loader cannot add a delta to.  Adapters run on the native
loaders (Qwen and its relatives) and on the `llama` (Llama, Mistral and the
other architectures it serves), `qwen2`, `gemma3` and `phi3` loaders, whose
attention and dense MLP projections take a delta (Phi-3's fused `attn_qkv`
and `[gate ‖ up]` `ffn_up` as they are; Mixtral's experts take none).  Other
architectures refuse them.

A Hugging Face PEFT adapter loads the same way, unconverted: name its
directory (holding `adapter_config.json` and `adapter_model.safetensors`) or
the `.safetensors` file.  It is read as `convert_lora_to_gguf.py` would
convert it: `q_proj` / `k_proj` / `v_proj` / `o_proj`, `gate_proj` /
`up_proj` / `down_proj`, Phi-3's `qkv_proj` / `gate_up_proj` and Qwen MoE's
`shared_expert` map to their GGUF projections; a llama-arch model's Q/K
`B` rows are permuted for its interleaved RoPE; and PEFT's scale
(`lora_alpha / r`, `/ √r` with `use_rslora`, per module with `rank_pattern`
/ `alpha_pattern`) is applied, so scale 1 runs the adapter as PEFT does.
DoRA, trained biases, `modules_to_save`, embedding or output-head pairs and
routed-expert pairs are refused.  Ids
follow the order given and are never reused.

A request runs the server-wide scales unless it sends its own `lora` list
(on `/completion`, `/infill`, `/v1/completions` and `/v1/chat/completions`).
The selection is part of the request's cache namespace: pooled sessions,
prefix-cache entries and snapshots computed under one set of adapters and
scales are only continued under the same set.  Requests with images (which
run on the NPU plugin) refuse adapters.

As in vLLM, each loaded adapter is also a model: `/v1/models` lists it under
its name with its base as `parent`, and a request on `/v1/chat/completions`,
`/v1/completions` or `/v1/responses` that names it as its `model` runs the
base model with that adapter at scale 1 (its own `lora` list, if it sends
one, still wins).  A model's own id is never taken as an adapter's name.
With `--allow-runtime-lora`, vLLM's `POST /v1/load_lora_adapter`
(`{"lora_name", "lora_path"}`, `load_inplace` to replace one) and
`POST /v1/unload_lora_adapter` (`{"lora_name"}`) load and unload adapters
while the server runs; a body's `model` picks the base when several are
loaded.  Adapters loaded this way start at server-wide scale 0, so requests
that do not name them are unchanged.  The paths are read as given, so the
flag is off by default; these routes, like `/lora-adapters` and model
management, take the API key and send no CORS headers.  An unloaded adapter's
weights are freed once the requests running it finish.
