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
joshua serve --model m.gguf --addr 127.0.0.1:8081 --api-key w-key --max-concurrency 2
joshua serve --model m.gguf --addr 127.0.0.1:8082 --api-key w-key --max-concurrency 2
joshua route --worker 127.0.0.1:8081,127.0.0.1:8082 \
  --worker-api-key w-key --api-key client-key --addr 0.0.0.0:8090
```

| Flag | Default | Meaning |
|---|---|---|
| `--worker` (`JOSHUA_ROUTE_WORKERS`) | required | Loopback `ip:port` HTTP addresses of `joshua serve` workers or local encrypted-tunnel listeners, comma-separated or repeated. Non-loopback sends/probes fail closed. |
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
| `--tls-cert` (`JOSHUA_TLS_CERT`) | none | Serve HTTPS with this PEM certificate chain; needs `--tls-key` and a build with `--features tls`. |
| `--tls-key` (`JOSHUA_TLS_KEY`) | none | PEM private key for `--tls-cert`. |

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
  in flight from this router and a conservative other-client snapshot, last
  error. Worker audio counts have no ownership identity, so the snapshot can
  include locally reserved audio; completion triggers on-demand revalidation.
* `GET /health` — `200` while at least one worker is healthy, else `503`.

* `POST /v1/audio/transcriptions` — routed to a worker reporting the
  `transcriptions` capability.  The upload is forwarded byte for byte
  together with the client's `Content-Type` (the multipart boundary the
  worker needs); the body is never parsed as JSON and no `model` field is
  read, so the whole audio job runs on one worker.  A worker without a
  loaded Whisper model does not advertise the capability and never receives
  the request; with no such worker the router answers `503`
  (`code: "no_worker"`).

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
  reporting exactly that id; with no `model`, only single-model workers are
  eligible (the server registry requires an explicit id with multiple models).
  The router never rewrites the request to insert a model. An
  unknown model is a `404` with `code: "model_not_found"` listing the served
  models; a known model with no healthy worker is a `503` (`no_worker`).
* Each model gets at most its advertised `max_concurrency` requests, minus
  requests reported running for other clients at the last probe. Legacy
  single-model workers use the worker-wide figures. `--worker-slots` remains
  an additional worker-wide cap when per-model figures are present.
* Chat `context_window` (including `num_ctx` / `n_ctx` aliases) is matched
  against the selected model's loaded window, not another model's window.
  Workers can add optional `n_ctx` to each entry in `models` without changing
  protocol version 1. Missing secondary-model context is **unknown** on old
  workers; the top-level `n_ctx` describes only the primary model. Unknown
  context remains eligible as a legacy fallback, with known-fitting free
  slots preferred. Missing or zero context metadata is not a fit guarantee.
  When all matching workers have reported known, insufficient windows,
  admission returns `400 context_window_exceeded` without dispatch or queuing.
  An unprobed or potentially fitting unhealthy worker prevents this definitive
  rejection (`503 no_worker` instead). Busy fitting workers use the normal
  queue even when smaller incompatible workers have free slots.
* Among available, equally context-qualified candidates, the router ranks
  outstanding estimated work plus the incoming request, divided by that
  model's slots; ties use request count, then rotate. Work hints are serialized
  bytes of `messages`, `prompt`, or embedding `input`, plus four work units per
  requested output token (chat `max_completion_tokens` wins over `max_tokens`).
  The fallback output hint is 256 for generation and zero for embeddings.
  Each request's hint is capped at 16,777,216 units. Hints stay reserved until
  its response/stream ends or is dropped; external requests, whose sizes are
  unknown, are assigned the incoming request's hint. This is a scheduling
  heuristic, **not exact tokens, remaining work, or predicted latency**.
  Prompt JSON is borrowed from the bounded HTTP body without constructing a
  JSON tree or decoding/copying its strings. Accounting has one entry per
  active model and is released on all existing guard/drop paths.
* When every slot is busy the request waits in the queue.  A full queue is a
  `429` (`queue_full`), a wait past `--queue-timeout-ms` a `503`
  (`queue_timeout`); both carry `Retry-After: 1`.

### Context-fit limits

The only hard context constraint added here is the explicit chat window the
engine already validates. Completion and embedding routes do not interpret
that option, so the router does not enforce it there. The router does not
load weights or tokenizers, create private model copies, count rendered chat
template/media tokens, or reject a prompt based on byte size. Exact prompt
fit, invalid generation options, and output-token clamping remain worker
decisions. In particular, `max_tokens` is a ceiling the engine may clamp,
not a minimum context reservation. Omitted windows still require worker-side
tokenization to validate prompt fit. This slice supplies neither affinity nor
real-host 1/2/4-worker throughput/latency benchmarks; issue #155 remains open.

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
output or worker status.  (`joshua serve` sends permissive CORS headers by
default; set `--cors-origin` to restrict it to explicit browser origins.)

## HTTPS

The client-facing hop can be encrypted in-process, the same way as
`joshua serve`:

```bash
joshua route --worker 127.0.0.1:8081,127.0.0.1:8082 --api-key client-key \
  --tls-cert /etc/joshua/fullchain.pem --tls-key /etc/joshua/key.pem
```

`--tls-cert` takes a PEM certificate chain and `--tls-key` a PKCS#8/RSA/SEC1
private key; both flags (or `JOSHUA_TLS_CERT`/`JOSHUA_TLS_KEY`) are required
together, and the binary must be built with `--features tls`, which pulls in
`rustls` + `axum-server` (ring provider).  Without the feature the router
refuses to start rather than silently serving plain HTTP.

TLS is terminated at the router, not on its worker connections. The worker
client supports HTTP only; `--features tls` and the router's TLS flags do not
encrypt worker traffic. Every worker send, including registry probes, refuses
non-loopback endpoints before connecting or attaching credentials. Such workers
remain unhealthy and receive no payload or bearer key. This intentionally
tightens compatibility: direct LAN, private-network and tailnet worker addresses
are no longer accepted for HTTP traffic, even for chat/completions or probes.

For remote workers, retain an actual authenticated encrypted tunnel with a
loopback listener on the router host and a loopback worker endpoint on the remote
host. For example, run `joshua serve --addr 127.0.0.1:8081 ...` remotely, then
`ssh -N -L 127.0.0.1:18081:127.0.0.1:8081 worker-host` on the router host,
and configure `joshua route --worker 127.0.0.1:18081 ...`. Verify SSH host keys,
authentication, and tunnel deployment separately. The router permits the local
HTTP hop; it cannot attest what a local forwarder does beyond that hop. A flag,
private IP, or local plaintext proxy is not encryption. Native worker TLS is
not implemented.

## Limits of this first version

* Loopback HTTP to workers (no TLS client); remote workers require independently
  verified authenticated encrypted tunnels. The router itself serves plain HTTP
  unless `--tls-cert`/`--tls-key` are given (see above).
* Routing uses model id, slots, reported in-flight count and bounded work
  hints from input bytes and output budgets. Explicit chat context windows
  are checked against known per-model limits, with legacy fallback when
  limits are unknown. It does not count rendered prompt tokens or guarantee
  exact prompt fit, compare tokenizers beyond the model id, or prefer a
  worker holding a reusable KV prefix. The last of
  those is designed in `docs/distributed-kv-cache.md` (#201): a chunk-hashed
  directory of which worker holds which prefix, consulted without an extra
  round trip.
* A worker streams chat completions token by token, so a worker lost after
  the first event ends the client stream rather than retrying it.
* Throughput, tail latency and per-request overhead have not been measured;
  that needs real multi-host hardware.
