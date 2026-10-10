# Request-level routing (`joshua route`, #155)

`joshua route` runs a coordinator in front of several independent
`joshua serve` workers.  Each chat, completion or embedding request is sent,
whole, to one worker and runs entirely there: prompt formatting,
tokenization, prefill, decode, KV cache reuse and sampling never leave that
worker.  This is a separate serving mode from the experimental DeepSeek V4
cluster (`--features distributed`), whose contract it does not touch; it
needs no cargo feature.

`--worker` takes the same form as the model-loading controller's worker list
(`joshua serve --model-dir DIR --worker ADDR,...`, `Vec<SocketAddr>`), but the
two name different processes: there, model-less `joshua worker` pipeline
agents that each hold a slice of one model; here, complete `joshua serve`
HTTP servers that each run whole requests.

## Running it

```bash
joshua serve --model m.gguf --addr 0.0.0.0:8081 --api-key w-key --max-concurrency 2
joshua serve --model m.gguf --addr 0.0.0.0:8082 --api-key w-key --max-concurrency 2
joshua route --worker 10.0.0.2:8081,10.0.0.3:8082 \
  --worker-api-key w-key --api-key client-key --addr 0.0.0.0:8090
```

| Flag | Default | Meaning |
|---|---|---|
| `--worker` (`JOSHUA_ROUTE_WORKERS`) | required | `ip:port` HTTP addresses of `joshua serve` workers, comma-separated or repeated. Plain HTTP. |
| `--addr` | `127.0.0.1:8090` | Listen address. |
| `--api-key` (`JOSHUA_API_KEY`) | none | Bearer key clients must send on `/v1` routes. |
| `--worker-api-key` (`JOSHUA_WORKER_API_KEY`) | none | Bearer key sent to workers (their `--api-key`). |
| `--worker-slots` | worker's `max_concurrency` | Requests in flight per worker. |
| `--queue-depth` | 64 | Requests that may wait for a slot when all are busy. |
| `--queue-timeout-ms` | 30000 | Longest wait in the queue. |
| `--health-interval-ms` | 2000 | Worker probe interval. |
| `--connect-timeout-ms` | 2000 | Connect timeout, and the whole-probe timeout. |
| `--response-timeout-ms` | 600000 | Longest a dispatched request waits for the worker's first output (the whole JSON response, or the first stream event); `0` waits indefinitely. A joshua worker computes a completion before sending any of it, so this also bounds generation time: raise it for long CPU generations. |
| `--max-response-bytes` | 67108864 (64 MiB) | Largest non-streaming worker response the router buffers. |

## Endpoints

On the router:

* `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/embeddings`
  — routed.  The body is forwarded byte for byte, so every parameter
  (sampling, `stop`, `max_tokens`, tools, `stream`) reaches the worker
  unchanged, and the worker's response, usage and errors come back unchanged.
  The router reads only `model` from the body.  A response is streamed when
  the worker answers `text/event-stream` (a joshua worker streams only chat
  completions; `stream: true` on the other routes still gets JSON), and
  buffered otherwise.
* `GET /v1/models` — union of the healthy workers' models.
* `GET /v1/workers` — registry: health, last reported info, slots, requests
  in flight from this router and from other clients, last error.
* `GET /health` — `200` while at least one worker is healthy, else `503`.

`/v1/audio/transcriptions` is not routed.

On each worker, `GET /v1/worker/info` (protocol version 1):

```json
{"protocol_version":1,"model":"m","n_ctx":4096,"backend":"cpu",
 "max_concurrency":2,"in_flight":0,
 "capabilities":{"chat":true,"completions":true,"embeddings":true,
                 "transcriptions":false,"streaming":true,"tools":true}}
```

## Registry and admission

* Membership is the configured list; discovery is not used for
  authorization.  A worker is routable only while its last probe succeeded
  with protocol version 1; a transport failure during a request also takes it
  out of rotation until the next good probe.
* A request names its model in `model`.  It goes to a healthy worker
  reporting exactly that id; with no `model`, any worker serves it.  An
  unknown model is a `404` with `code: "model_not_found"` listing the served
  models; a known model with no healthy worker is a `503` (`no_worker`).
* Each worker gets at most `slots` requests from the router, minus requests
  it reported running for other clients at the last probe.  The router picks
  the matching worker with the lowest load ratio; ties rotate.
* When every slot is busy the request waits in the queue.  A full queue is a
  `429` (`queue_full`), a wait past `--queue-timeout-ms` a `503`
  (`queue_timeout`); both carry `Retry-After: 1`.

## Failure and retry rules

Admission and execution are kept apart, so a retry cannot duplicate output:

* **Retried on another worker** (each worker at most once per request), only
  while nothing has been sent to the client: a refused or failed connection,
  a connection lost before the first stream event or before the complete
  non-streaming body, or the worker's own `503` (its engine refuses admission
  before any generation starts).
* **Passed through, not retried**: any other worker response, such as a
  `400` for an invalid request or a `500` from a failed generation.  A worker
  rejecting the router's key becomes a `502`.
* **Never restarted**: once a stream event has reached the client, losing
  the worker — or the worker closing the stream before its `[DONE]` event —
  ends the stream with an error event
  (`{"error":{"code":"worker_lost",...}}`) and no `[DONE]`.
* **Deadline**: a worker that produces no first output (whole JSON response
  or first stream event) within `--response-timeout-ms` gets a `504`
  (`worker_timeout`), not retried: the router closes the connection, which
  cancels the work on the worker, frees the slot, and takes the worker out
  of rotation until its next good probe.  There is no deadline once a stream
  has started.
* **Too large**: a non-streaming response over `--max-response-bytes` is a
  `502` (`response_too_large`).  The worker did answer, so it stays healthy
  and the request is not retried.
* If every eligible worker refused admission the client gets a `503`
  (`workers_busy`); if any attempt failed in transit and no untried worker
  remains, a `502` (`workers_failed`) — also when the failed workers have
  since been marked unhealthy.  `503` (`no_worker`) means no attempt could
  be made at all.

Streams are forwarded as whole SSE events in the worker's order; an
unfinished event (one cut off by the end of the stream) is never forwarded.

## Cancellation

Each forwarded request owns its own connection to the worker.  When the
client disconnects, the router drops the request, closing that connection
and freeing the slot.  The worker's handler is then dropped, which sets a
cancel flag the engine checks before every decoded token
(`Engine::complete_chat_cancellable`) and, for embeddings, before every
input text (`Engine::embed_with_usage_cancellable`), so the work stops and
the engine's concurrency permit and session are released.  (Prefill, or the
embedding of the current text, already in progress runs to its end, so the
worker may briefly refuse a new request with a `503`; the router then tries
another worker.)

## Records

Every request has an id: the client's `x-request-id` when it is 1–128
characters of `[A-Za-z0-9._-]`, else a generated `req-…`.  It is returned on
every response and forwarded to the worker.  Responses from a dispatched
attempt also carry `x-joshua-worker` (the serving worker's URL); requests
rejected before dispatch (`404`, `429`, `503`, `502 workers_failed`) carry
no worker header.  The router logs, per attempt, the request id, worker,
path, attempt number, queue time, duration, status and outcome (`complete`,
`worker_error`, `client_disconnected`, `worker_lost`, `worker_timeout`,
`response_too_large`).  Prompt and output text are never logged.

## Cross-origin access

The router sends CORS headers (permissive: any origin) only when `--api-key`
is set, so browser pages on other origins can call it but must present the
key.  Without a key the browser's same-origin policy applies, so an arbitrary
website opened on a machine that can reach the router cannot read its model
output or worker status.  (`joshua serve` currently sends permissive CORS
headers with or without a key.)

## Limits of this first version

* Plain HTTP to workers (no TLS client); run workers on a trusted network or
  behind a TLS-terminating tunnel.  The router itself serves plain HTTP.
* Routing uses model id, slots and reported in-flight count.  It does not
  estimate per-request work (prompt length, `max_tokens`), check a
  request's length against a worker's `n_ctx`, compare tokenizers beyond the
  model id, or prefer a worker holding a reusable KV prefix.
* A joshua worker computes the whole completion before streaming it, so a
  worker lost mid-generation is usually lost before output (and retried).
* Throughput, tail latency and per-request overhead have not been measured;
  that needs real multi-host hardware.
