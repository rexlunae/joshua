# joshua-ui

A browser frontend for [Joshua](../../README.md), written in Rust with
[Dioxus 0.7](https://dioxuslabs.com/) and
[dioxus-bulma](https://crates.io/crates/dioxus-bulma) 0.7.3. It compiles to
WebAssembly and talks to a running `joshua serve` over its HTTP API, so it can
point at a local instance or at any node of a cluster.

The crate is a workspace member but not a default member: a plain
`cargo build` / `cargo test` of Joshua does not build it.

## What it does

| View       | Shows / does                                                    | Server endpoint(s)            |
|------------|-----------------------------------------------------------------|-------------------------------|
| Monitor    | Health and round-trip latency, loaded models, list of server routes, auto-refresh every 5 s | `GET /health`, `GET /v1/models` |
| Monitor    | Request count, token usage, wall time and tok/s of generations sent from this browser (measured client-side) | — |
| Playground | Chat with the loaded model: system prompt, `max_tokens`, `temperature`, `top_p`, `top_k`, stop sequences, SSE streaming on/off, cancel | `POST /v1/chat/completions` |
| Cluster    | Add several `joshua serve` addresses and check each one's health, latency and model | `GET /health`, `GET /v1/models` per node |

An API key entered in the connection bar is sent as `Authorization: Bearer
<key>` to the `/v1` routes (matching `joshua serve --api-key`); `/health`
never needs one. The server already enables permissive CORS, so the UI can be
served from a different origin than the server.

### Placeholders

These are shown as labelled placeholders because the server has no API for
them yet; the UI does not invent endpoints:

- **Cluster topology** — peers, ranks, shard placement and collective health.
  Clusters currently run through the one-shot `joshua cluster-run` CLI (`--features distributed`), which has
  no HTTP status endpoint.
- **Server-side metrics** — queue depth, KV-cache use, backend placement,
  server-measured decode speed.
- **Runtime controls** — loading/unloading a model or changing placement.
  `joshua serve` loads one model at start-up and has no admin API.

Note that the server finishes generating before it starts sending SSE
chunks, so "streaming" shows the text arriving all at once after the full
generation time, and *Cancel* stops the browser waiting but not the server's
work.

## Build and run

Install the WebAssembly target and the Dioxus CLI (once):

```sh
rustup target add wasm32-unknown-unknown
cargo install dioxus-cli --version 0.7.10 --locked   # provides `dx`
```

Start a Joshua server, then serve the UI from this directory:

```sh
joshua serve --model path/to/model.gguf            # listens on 127.0.0.1:8080
cd crates/joshua-ui
dx serve --platform web --port 8081               # dx defaults to 8080, which joshua uses
dx bundle --platform web --release                 # static files for any web server
```

Open the address `dx` prints, enter the server's base URL
(`http://127.0.0.1:8080` by default) and press **Connect**.

## Development

The HTTP-free client logic lives in `src/api.rs` (URL handling, request
bodies, response and error parsing, an incremental SSE decoder) and is
unit-tested on the host; `src/client.rs` is the `reqwest` transport (browser
`fetch` on wasm) and `src/ui.rs` holds the components.

```sh
cargo check  -p joshua-ui                                  # host check
cargo check  -p joshua-ui --target wasm32-unknown-unknown  # the real target
cargo test   -p joshua-ui                                  # api.rs unit tests
cargo clippy -p joshua-ui --all-targets -- -D warnings
```

The host build exists for checking and tests only; the UI itself runs in the
browser.
