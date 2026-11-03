# Joshua in T3 Code

Two small pieces that let [T3 Code](https://t3.codes) use Joshua as a local model on this machine or any
ssh host it knows about:

| File | Role |
| - | - |
| `joshua_acp.py` | An ACP agent: exposes a Joshua OpenAI endpoint to T3 as a provider, runs the tool loop (list/read/write/edit/run) |
| `joshua-control` | Control plane: list hosts, start/stop servers, load/unload models, bring up a router + workers, register the provider |

T3 speaks the Agent Client Protocol, so a provider is just a program on the host that answers
`initialize` / `session/new` / `session/prompt`. Because each host runs its own T3 server, the same
provider name gives that host's Joshua when the user picks **Run on → that machine**.

## Install

```bash
mkdir -p ~/.t3/joshua
cp joshua_acp.py joshua-control ~/.t3/joshua/
chmod +x ~/.t3/joshua/joshua_acp.py ~/.t3/joshua/joshua-control
joshua-control hosts                 # this machine + the ssh device hosts in T3's settings
joshua-control start --host <host> --model <path.gguf> --model-dir ~/models --port 18080
joshua-control install --host <host> --port 18080     # agent + provider instance in T3's settings
```

Reconnect the device in T3 (or restart its server) after `install`.

## Model picker

The agent advertises every loadable GGUF on the host, not only what is loaded:

- scans `~/models`, `~/llama.cpp/models`, `~/.lmstudio/models`, `~/nemo` and `~/joshua-decode-*`
  (override with `--catalog DIR`, repeatable)
- offers a file when it has a sibling `tokenizer.json` and a real `GGUF` magic (truncated downloads
  and HTML error pages are skipped)
- sharded models (`…-00001-of-00003.gguf`) appear once, pointing at the first shard
- labels lead with the distinguishing part (`UD-IQ1_S Qwen3.8 Flash Next`, `[qwen3] Q4_K_M Qwen3 14B`),
  because the client elides the tail of a long name; `modelId` is unchanged
- selecting a model that is not loaded issues `POST /v1/models/load` for it, then uses the id Joshua
  reports

## GPU and CPU servers

Placement is fixed per server, so a host that should serve both a model that fits the card and a
larger one runs two:

```bash
joshua-control start --host kleya --model ~/models/Qwen3-14B-Q4_K_M.gguf --model-dir ~ \
  --port 18080 --ctx 16384 --device cuda --dense-placement device
joshua-control start --host kleya --model ~/models/Qwen3.8-27B-Q4_K_M.gguf --model-dir ~ \
  --port 18090 --ctx 32768 --device cpu
joshua-control install --host kleya --port 18080 --cpu-port 18090
```

The agent tries the GPU server first and falls back to the CPU one when a load is refused for device
memory, so the picker can offer everything on the host.

## Clustering

Joshua's router is `serve --worker URL …` with a shared `JOSHUA_CLUSTER_KEY`; every worker is a
normal `serve` that also answers `/v1/worker/info`. It needs a build with the `distributed` feature and
the **same model file** on every worker.

```bash
joshua-control cluster up --router kleya --workers kleya,erskin \
  --model /home/tserica/models/Qwen3-8B-Q4_K_M.gguf --port 18080 --ctx 8192
joshua-control cluster status --workers kleya,erskin --port 18092
```

Workers must bind an address the router can reach (`--addr 0.0.0.0:PORT`, tailnet IPs across
machines) and share the key the command prints; `--worker` takes `host:port`, not a URL. Point a
provider at the router's port with `joshua-control install --host <router-host> --port <router-port>`.

Membership is only visible on a coordinator (`GET /v1/workers`); `serve --worker` does not mount that
route, so `cluster status` reports each worker's own `/v1/worker/info` (model, backend, in-flight)
instead.

## Controls

Each session exposes Joshua's request-level knobs as ACP `select`s: context window, reasoning effort,
max output tokens, temperature, top-p, top-k, min-p. `default` omits the field. Everything else
(device, placement, VRAM budget, speculative decoding, expert cache, prefill chunk) is a server flag
— set it through `joshua-control start … --device … --extra "…"`.

## Behaviour worth knowing

- Completions stream (`stream: true`); the agent forwards deltas as they arrive, so a long prefill
  shows progress and no client-side timeout fires. Read timeout defaults to one hour
  (`JOSHUA_READ_TIMEOUT`).
- Models wrap tool calls in their own markers and sometimes emit broken JSON
  (`<tool_call>{"name": …}}}</tool_call>`). The agent parses `<tool_call>`, `<tool_code>` and
  `<|call|>` blocks tolerantly, repairs runaway braces and code fences, executes the call and keeps
  the marker out of the transcript.
- Thinking spans (` thinking…</think>`) are stripped from the answer; upstream sends them as
  `reasoning_content` instead.
- The agent auto-approves its tools, matching a T3 "Full access" thread.
