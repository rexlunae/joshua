#!/usr/bin/env bash
# Run the real-client suites under clients/ against a live `joshua serve` on
# the test suite's tiny model: builds joshua, writes the model, starts the
# server, waits for /health, runs the Python (openai, ollama) and Node
# (openai, ollama) suites, and stops the server.  Needs python3 and node on
# PATH; the client libraries are installed into clients/python/.venv and
# clients/node/node_modules unless SKIP_INSTALL=1.
#
#   scripts/client-tests.sh            # debug build
#   PROFILE=release scripts/client-tests.sh
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
profile="${PROFILE:-debug}"
port="${JOSHUA_PORT:-18080}"
work="$(mktemp -d)"
server_pid=""
cleanup() {
    if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || true; fi
    rm -rf "$work"
}
trap cleanup EXIT

flag=()
[[ "$profile" == release ]] && flag=(--release)
cargo build "${flag[@]}" --bin joshua --example tiny_model
"target/$profile/examples/tiny_model" "$work/tiny" >/dev/null

if [[ "${SKIP_INSTALL:-0}" != 1 ]]; then
    python3 -m venv clients/python/.venv
    clients/python/.venv/bin/pip install -q -r clients/python/requirements.txt
    (cd clients/node && npm ci --no-audit --no-fund)
fi
python="${PYTHON:-clients/python/.venv/bin/python}"

# A server already on the port would answer /health for ours and take the
# tests: refuse to start rather than test the wrong binary.
if curl -s -o /dev/null "http://127.0.0.1:$port/"; then
    echo "port $port is already in use; set JOSHUA_PORT to a free one" >&2
    exit 1
fi
"target/$profile/joshua" serve --model "$work/tiny" --addr "127.0.0.1:$port" --n-ctx 256 \
    >"$work/server.log" 2>&1 &
server_pid=$!
for _ in $(seq 120); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null && break
    if ! kill -0 "$server_pid" 2>/dev/null; then
        cat "$work/server.log"
        echo "joshua serve exited before it was ready" >&2
        exit 1
    fi
    sleep 1
done
curl -sf "http://127.0.0.1:$port/health" >/dev/null || { cat "$work/server.log"; exit 1; }
kill -0 "$server_pid" 2>/dev/null || { cat "$work/server.log"; echo "joshua serve exited" >&2; exit 1; }

export JOSHUA_URL="http://127.0.0.1:$port" JOSHUA_MODEL=tiny
status=0
"$python" -m pytest -q clients/python || status=1
(cd clients/node && node --test test.mjs) || status=1
if [[ $status != 0 ]]; then
    echo "--- server log ---"
    tail -n 100 "$work/server.log"
fi
exit $status
