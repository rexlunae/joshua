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
joshua-control install --host <host> --port 18080 --model-dir ~/models
```

Pass the same `--model-dir` used by `joshua-control start`; catalog discovery is restricted to that
directory, matching the server's loadable paths. Reconnect the device in T3 (or restart its server)
after `install`.

Use `--api-key` with `start` or `cluster up` to protect the controller API. The key is stored in a
mode-600 environment file rather than the server's process arguments; pass it to `status`, `models`,
`load`, `unload`, or `cluster status` when querying that server. `install` registers a launcher
that reads the matching environment file, so the T3 provider can authenticate without putting the
key in provider arguments or the process command line.

## Model picker

The agent advertises every loadable GGUF on the host, not only what is loaded:

- scans the default model directories (including current `~/joshua-decode-*` directories), but only
  paths inside the configured server `--model-dir`; `--catalog DIR` can add a narrower root inside it
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

Joshua's distributed build splits a model into stages: **workers are model-less** and a controller
streams each stage's weights to them.

```bash
# Copy this mode-600 file to each worker and controller; do not put the key in command arguments.
mkdir -p ~/.config/joshua
(umask 077; openssl rand -hex 32 | sed 's/^/export JOSHUA_CLUSTER_KEY=/' > ~/.config/joshua/cluster.env)
chmod 600 ~/.config/joshua/cluster.env

# one worker node per host (raw stage protocol on 7070, no HTTP, no model)
. ~/.config/joshua/cluster.env && joshua worker --listen 0.0.0.0:7070

# controller: an ordinary serve that places the model on those nodes
. ~/.config/joshua/cluster.env && joshua serve --model ~/models/Qwen3-8B-Q4_K_M.gguf \
  --model-dir ~/models --worker 100.97.218.93:7070,100.75.187.112:7070 \
  --addr 127.0.0.1:18085 --n-ctx 8192
```

Both sides need the same `JOSHUA_CLUSTER_KEY`, the build must include `distributed`, and `--worker`
takes `host:port` (not a URL).

`joshua-control` wraps the same thing:

```bash
joshua-control cluster up --router kleya --workers kleya,erskin \
  --model /home/tserica/models/Qwen3-8B-Q4_K_M.gguf --port 18085 --ctx 8192 --api-key "$JOSHUA_API_KEY"
joshua-control cluster status --workers kleya,erskin --router kleya --port 18085 --api-key "$JOSHUA_API_KEY"
```

`cluster status` probes the worker sockets (a `joshua worker` node does not answer HTTP) and asks the
controller for its models.

**Known limitation:** the stage planner currently over-reserves and refuses to place even a small
model — a 5 GB Qwen3-8B across two idle workers reports
`stage 0 planning reservation 40910570496 exceeds budget 34992283648` — tracked in
[rexlunae/joshua#267](https://github.com/rexlunae/joshua/issues/267). Until that lands, point one
provider at each host's own server (below) rather than at a controller.

## Running under systemd

`systemd/` holds two user units that survived a reboot test here:

| Unit | Runs |
| - | - |
| `joshua-worker.service` | `joshua worker --listen 0.0.0.0:7070`, model-less |
| `joshua-serve.service` | one `joshua serve` for this host (edit model path, port, `--device`) |

```bash
mkdir -p ~/.config/systemd/user ~/.config/joshua
printf 'JOSHUA_CLUSTER_KEY=%s\n' "$(openssl rand -hex 32)" > ~/.config/joshua/cluster.env   # chmod 600
cp systemd/*.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now joshua-worker joshua-serve
loginctl enable-linger "$USER"      # keep them up without a login session
```

The cluster key goes in `~/.config/joshua/cluster.env` (mode 600), never in the unit file. The units
use `UnsetEnvironment=JOSHUA_API_KEY`: an **empty** `JOSHUA_API_KEY` is itself a misconfiguration
(`--api-key contains no non-empty keys`), so the variable has to be removed rather than set blank.

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
- Explicit `<think>…</think>` spans are stripped; separate upstream `reasoning_content` is ignored.
  Ordinary prose containing words such as “thinking” is preserved.
- `session/cancel` is handled while a completion or command is running; it closes the active request
  and terminates the command process group.
- The agent auto-approves its tools, matching a T3 "Full access" thread.
