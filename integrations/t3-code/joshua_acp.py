#!/usr/bin/env python3
"""Joshua ACP agent: exposes a Joshua OpenAI-compatible server as a T3 Code provider.

Speaks Agent Client Protocol (JSON-RPC over stdio) and runs a small tool-using agent loop
against Joshua's /v1/chat/completions (tools + tool_calls are supported by Joshua).

    joshua_acp.py --base-url http://127.0.0.1:8080 [--api-key K] [--model ID]

Models come from Joshua's /v1/models; the model is switchable per session through the ACP
`model` config option, so no restart is needed when Joshua serves several models
(`joshua serve --model-dir DIR`).
"""
import argparse, glob, json, os, re, signal, socket, subprocess, sys, threading, time, uuid
from pathlib import Path

MAX_TURNS = 24
TOOL_OUTPUT_LIMIT = 20000

TOOLS = [
    {"type": "function", "function": {
        "name": "list_dir", "description": "List files in a directory of the project.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory, relative to the project root."}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Create or overwrite a text file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit_file", "description": "Replace an exact text block in a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "old_text": {"type": "string"},
            "new_text": {"type": "string"}},
            "required": ["path", "old_text", "new_text"]}}},
    {"type": "function", "function": {
        "name": "run_command", "description": "Run a shell command in the project directory.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}, "timeout_s": {"type": "integer"}},
            "required": ["command"]}}},
]

SYSTEM = """You are Joshua, a coding agent working inside the user's project.
Use the tools to inspect and change files; do not guess file contents.
Work in small steps, keep edits minimal, and finish with a short summary of what changed.
The project root is {cwd}."""



TOOL_BLOCKS = {"<tool_call>": "</tool_call>", "<tool_code>": "</tool_code>", "<|call|>": "<|end|>"}


def parse_tool_block(text):
    """Turn a model-emitted tool-call block into OpenAI-shaped calls, tolerantly.

    Models wrap the call in their own markers and often emit slightly broken JSON (an extra
    closing brace, a code fence, single quotes).  A strict parse here would hand the raw marker
    straight to the user, which is what we saw with `<tool_call>{"name": ...}}}</tool_call>`.
    """
    body = text.strip()
    for opener, closer in TOOL_BLOCKS.items():
        if body.startswith(opener):
            body = body[len(opener):]
        if body.endswith(closer):
            body = body[: -len(closer)]
    body = body.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[-1] if "\n" in body else body
        body = body.rsplit("```", 1)[0]
    body = body.strip()

    payload = None
    for attempt in range(6):
        try:
            payload = json.loads(body)
            break
        except json.JSONDecodeError:
            trimmed = body.rstrip()
            if trimmed.endswith(("}", "]")):
                body = trimmed[:-1]
                continue
            break
    if payload is None:
        return []
    if isinstance(payload, list):
        entries = payload
    else:
        entries = [payload]

    calls = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function") if isinstance(entry.get("function"), dict) else entry
        name = fn.get("name") or fn.get("tool") or entry.get("name")
        args = fn.get("arguments", fn.get("parameters", entry.get("arguments", {})))
        if not name:
            continue
        calls.append({"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                      "function": {"name": name,
                                   "arguments": args if isinstance(args, str) else json.dumps(args)}})
    return calls


class StreamSplitter:
    """Strips thinking spans and tool-call blocks out of the answer as it streams.

    Text is held back only while it could still be the start of a marker, so nothing emitted has
    to be retracted; a completed tool block is parsed into calls instead of being shown.
    """

    def __init__(self):
        self.pending = ""
        self.calls = []
        self.in_think = False

    def _emit_safe(self, text):
        keep = 0
        for marker in list(TOOL_BLOCKS) + ["<think>", "</think>"]:
            for size in range(min(len(marker) - 1, len(text)), 0, -1):
                if text.endswith(marker[:size]):
                    keep = max(keep, size)
                    break
        cut = len(text) - keep
        return text[:cut], text[cut:]

    def feed(self, delta, final=False):
        self.pending += delta
        out = []
        while True:
            if self.in_think:
                end = self.pending.find("</think>")
                if end < 0:
                    self.pending = "" if final else self.pending
                    break
                self.pending = self.pending[end + len("</think>"):]
                self.in_think = False
                continue

            think = self.pending.find("<think>")
            tool_at, tool_marker = None, None
            for marker in TOOL_BLOCKS:
                at = self.pending.find(marker)
                if at >= 0 and (tool_at is None or at < tool_at):
                    tool_at, tool_marker = at, marker

            # A close marker with no open span (thinking disabled) is dropped, not shown.
            close_at = self.pending.find("</think>")
            if close_at >= 0 and (tool_at is None or close_at < tool_at) and (think < 0 or close_at < think):
                out.append(self.pending[:close_at])
                self.pending = self.pending[close_at + len("</think>"):]
                continue

            if tool_at is None and think < 0:
                safe, rest = self._emit_safe(self.pending)
                out.append(safe)
                self.pending = "" if final else rest
                break

            if think >= 0 and (tool_at is None or think < tool_at):
                out.append(self.pending[:think])
                self.pending = self.pending[think + len("<think>"):]
                self.in_think = True
                continue

            out.append(self.pending[:tool_at])
            self.pending = self.pending[tool_at:]
            closer = TOOL_BLOCKS[tool_marker]
            end = self.pending.find(closer)
            if end < 0:
                if not final:
                    break
                block, self.pending = self.pending, ""
            else:
                block = self.pending[: end + len(closer)]
                self.pending = self.pending[end + len(closer):]
            self.calls.extend(parse_tool_block(block))
        return "".join(out)

    def flush(self):
        rest = self.feed("", final=True)
        leftover = self.pending
        self.pending = ""
        return rest + leftover


QUANT_RE = re.compile(r"(?:^|[-_\s])(UD-)?(IQ\d[_A-Z0-9]*|Q\d_K(?:_[A-Z0-9]+)?|Q\d_\d|Q8_0|Q6_K|Q5_K|Q4_0|MXFP4|TQ1_0|TQ2_0|BF16|F16)"
                      r"(?:$|[-_\s])")


def short_name(model_id, gguf_name="", limit=34):
    """A picker label whose distinguishing part survives right-hand truncation.

    T3 elides the tail of a long model name, and the tail is exactly where the difference lives
    (`…-UD-IQ1_S`, `Qwen3-14B-Q4_K_M [qwen3]`).  Put the disambiguator first and keep the whole
    label short enough that nothing important is the part that gets cut.
    """
    identifier = model_id
    bracket = ""
    m = re.search(r"\[([^\]]+)\]\s*$", identifier)
    if m:
        bracket = m.group(1).strip()
        identifier = identifier[: m.start()].strip()

    quant = ""
    q = QUANT_RE.search(identifier)
    if q:
        quant = (q.group(1) or "") + q.group(2)

    base = gguf_name or identifier
    if base.lower().startswith("qwen_"):
        base = base[5:]            # Qwen_Qwen3.8 27B -> Qwen3.8 27B
    base = QUANT_RE.sub(" ", base)  # the quant is shown as a prefix, not twice
    base = base.replace("_", " ")
    base = re.sub(r"\s+", " ", base).strip(" -_")
    prefix = " ".join(part for part in ((f"[{bracket}]" if bracket else ""), quant) if part)
    if not prefix:
        return base if len(base) <= limit else base[: limit - 1].rstrip() + "…"

    room = max(8, limit - len(prefix) - 1)
    if len(base) > room:
        base = base[: room - 1].rstrip() + "…"
    return f"{prefix} {base}"

def gguf_names(path):
    """Read `general.basename` / `general.name` from a GGUF header, without loading weights.

    Joshua reports a loaded model under its `general.basename` (moe.gguf loads as
    "DeepSeek-V4-Flash-0731-reap"), so the catalog must use that id or the picker would offer a
    name the server never matches.
    """
    import struct
    names = {}
    fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    try:
        with open(path, "rb") as fh:
            if fh.read(4) != b"GGUF":
                return names
            fh.read(4)                      # version
            fh.read(8)                      # tensor count
            n_kv = struct.unpack("<Q", fh.read(8))[0]
            for _ in range(min(n_kv, 512)):
                key = fh.read(struct.unpack("<Q", fh.read(8))[0]).decode("utf-8", "replace")
                vtype = struct.unpack("<I", fh.read(4))[0]
                if vtype == 8:
                    value = fh.read(struct.unpack("<Q", fh.read(8))[0]).decode("utf-8", "replace")
                    if key in ("general.basename", "general.name"):
                        names[key] = value
                elif vtype == 9:
                    atype = struct.unpack("<I", fh.read(4))[0]
                    alen = struct.unpack("<Q", fh.read(8))[0]
                    if atype == 8:
                        for _ in range(alen):
                            fh.read(struct.unpack("<Q", fh.read(8))[0])
                    else:
                        fh.read(fixed.get(atype, 4) * alen)
                elif vtype in fixed:
                    fh.read(fixed[vtype])
                else:
                    break
                if "general.basename" in names and "general.name" in names:
                    break
    except Exception:
        return names
    return names


class PromptCancelled(Exception):
    pass


def http_json(url, payload=None, api_key=None, timeout=300, cancel_event=None,
              response_callback=None):
    import urllib.request
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if response_callback:
                response_callback(r)
            if cancel_event and cancel_event.is_set():
                r.close()
                raise PromptCancelled()
            try:
                return json.loads(r.read().decode())
            finally:
                if response_callback:
                    response_callback(None)
    except Exception:
        if response_callback:
            response_callback(None)
        if cancel_event and cancel_event.is_set():
            raise PromptCancelled() from None
        raise


class Session:
    def __init__(self, sid, cwd, base=None):
        self.base = base
        self.id = sid
        self.cwd = Path(cwd or os.getcwd()).expanduser().resolve()
        self.messages = []
        self.model = None
        self.cancel_event = threading.Event()
        self.prompt_lock = threading.Lock()
        self.response_lock = threading.Lock()
        self.active_response = None
        self.active_process = None
        self.opts = {"context_window": None, "reasoning_effort": None, "max_tokens": None,
                     "temperature": None, "top_p": None, "top_k": None, "min_p": None}


DEFAULT_ROOTS = ["~/models", "~/llama.cpp/models", "~/.lmstudio/models", "~/nemo"]


def default_roots():
    roots = [os.path.expanduser(root) for root in DEFAULT_ROOTS]
    roots.extend(glob.glob(os.path.expanduser("~/joshua-decode-*")))
    return roots


class Agent:
    def __init__(self, base_url, api_key, default_model, catalog_roots=None, cpu_url=None,
                 model_dir=None):
        self.base = base_url.rstrip("/")
        self.cpu_base = cpu_url.rstrip("/") if cpu_url else None
        self.key = api_key
        self.default_model = default_model
        self.sessions = {}
        self.aliases = {}
        self._send_lock = threading.Lock()
        self.read_timeout = float(os.environ.get("JOSHUA_READ_TIMEOUT", 3600))
        self.model_dir = Path(os.path.expanduser(model_dir)).resolve() if model_dir else None
        self.roots = []
        if self.model_dir:
            for root in (catalog_roots or default_roots()):
                resolved = Path(os.path.expanduser(root)).resolve()
                if resolved.is_relative_to(self.model_dir):
                    self.roots.append(resolved)
            if not self.roots:
                self.roots.append(self.model_dir)

    @staticmethod
    def _active_response(session, response):
        with session.response_lock:
            session.active_response = response
            if response is not None and session.cancel_event.is_set():
                Agent._close_response(response)

    @staticmethod
    def _close_response(response):
        try:
            sock = response.fp.raw._sock
        except AttributeError:
            response.close()
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        response.close()

    @classmethod
    def _cancel_session(cls, session):
        session.cancel_event.set()
        with session.response_lock:
            response = session.active_response
            if response is not None:
                cls._close_response(response)
            process = session.active_process
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except OSError:
                    process.terminate()

    @staticmethod
    def _stop_process(process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            process.terminate()
        try:
            process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                process.kill()
            process.communicate()

    # ---- Joshua ----
    def servers(self):
        return [u for u in (self.base, self.cpu_base) if u]

    def loaded(self, base=None, session=None):
        """Model ids a Joshua server already has in memory (all of them by default)."""
        bases = [base] if base else self.servers()
        ids = []
        for url in bases:
            try:
                data = http_json(
                    f"{url}/v1/models", api_key=self.key, timeout=30,
                    cancel_event=session.cancel_event if session else None,
                    response_callback=(lambda response: self._active_response(session, response))
                    if session else None)
            except Exception:
                if session and session.cancel_event.is_set():
                    raise PromptCancelled()
                continue
            for m in (data.get("data", []) if isinstance(data, dict) else []):
                if m.get("id") and m["id"] not in ids:
                    ids.append(m["id"])
        return ids

    def catalog(self):
        """Loadable GGUFs on this host: id -> {path, name}. Needs a sibling tokenizer.json.

        The id is the file stem (what Joshua reports for most models); the display name comes from
        the GGUF metadata, so `moe.gguf` still reads as "DeepSeek V4 Flash 0731 Reap 150b". When a
        catalog id has already been loaded, the alias map hides it in favour of the server's own id.
        """
        found = {}
        if not self.model_dir:
            return found
        for root in self.roots:
            if not os.path.isdir(root):
                continue
            for path in sorted(Path(root).rglob("*.gguf")):
                if path.name.startswith(("mmproj-", "ggml-vocab")):
                    continue
                try:
                    path = path.resolve(strict=True)
                    if not path.is_relative_to(self.model_dir):
                        continue
                except OSError:
                    continue
                if not (path.parent / "tokenizer.json").is_file():
                    continue
                try:
                    with open(path, "rb") as fh:
                        if fh.read(4) != b"GGUF":     # truncated download / HTML error page
                            continue
                except OSError:
                    continue
                stem = path.stem
                shard = re.match(r"^(.*)-(\d{5})-of-(\d{5})$", stem)
                if shard:
                    if shard.group(2) != "00001":      # only the first part is a model entry
                        continue
                    stem = shard.group(1)
                model_id = stem
                if model_id in found:
                    model_id = f"{stem} [{path.parent.name}]"
                meta = gguf_names(path)
                label = meta.get("general.name") or meta.get("general.basename") or path.stem
                found[model_id] = {"path": str(path), "name": label}
        return found

    def models(self):
        """Everything the user may pick: what Joshua has loaded, plus the rest of the catalog.

        Each entry's `name` is a short label with the distinguishing part first, because the
        client elides the tail of a long one (`…-UD-IQ1_S` hid which quant it was).
        """
        catalog = self.catalog()
        loaded = self.loaded()
        out = [{"modelId": mid,
                "name": short_name(mid, catalog.get(mid, {}).get("name", ""))} for mid in loaded]
        for model_id, info in catalog.items():
            if model_id in loaded or self.aliases.get(model_id) in loaded:
                continue
            out.append({"modelId": model_id, "name": short_name(model_id, info["name"])})
        return out

    def ensure_loaded(self, model_id, session=None):
        """Resolve a picker id to a model the server has actually loaded.

        A loaded id is returned as-is; a catalog id is loaded through POST /v1/models/load and the
        server's own id for it is remembered, so later turns and the picker agree.
        """
        for candidate in self.servers():
            in_this = self.loaded(candidate, session=session)
            if model_id in in_this:
                return model_id, candidate
            if self.aliases.get(model_id) in in_this:
                return self.aliases[model_id], candidate
        # A catalog id may have been renamed after the server loaded it. The
        # persistent alias is a hint, not an identity guarantee: discard a
        # stale alias and continue with the catalog path instead of returning
        # a model that no longer exists.
        stale_alias = self.aliases.pop(model_id, None)
        if stale_alias:
            for candidate in self.servers():
                loaded = self.loaded(candidate, session=session)
                if model_id in loaded:
                    return model_id, candidate
                if stale_alias in loaded:
                    return stale_alias, candidate
        info = self.catalog().get(model_id)
        if info is None:
            raise ValueError(f"model {model_id!r} is not loaded and not in the catalog on this host")
        res, base = None, self.base
        errors = []
        for candidate in self.servers():
            if model_id in self.loaded(candidate, session=session):
                return model_id, candidate
            try:
                attempt = http_json(f"{candidate}/v1/models/load", {"model": info["path"]},
                                    self.key, timeout=3600,
                                    cancel_event=session.cancel_event if session else None,
                                    response_callback=(lambda response: self._active_response(session, response))
                                    if session else None)
            except PromptCancelled:
                raise
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
                continue
            if isinstance(attempt, dict) and attempt.get("error"):
                error = attempt["error"]
                message = error.get("message", error) if isinstance(error, dict) else error
                errors.append(f"{candidate}: {message}")
                continue
            res, base = attempt, candidate
            break
        if res is None:
            raise ValueError("; ".join(errors) or f"could not load {model_id!r}")
        server_id = res.get("id") if isinstance(res, dict) else None
        for _ in range(900):
            if session and session.cancel_event.is_set():
                raise PromptCancelled()
            loaded = self.loaded(base, session=session)
            if server_id and server_id in loaded:
                self.aliases[model_id] = server_id
                return server_id, base
            if model_id in loaded:
                self.aliases[model_id] = model_id
                return model_id, base
            if session:
                session.cancel_event.wait(1)
            else:
                time.sleep(1)
        raise ValueError(f"Joshua accepted {model_id!r} but never reported it as loaded")

    def chat(self, session, tools=None, emit=None):
        """Stream a completion from Joshua and return the assembled message.

        Streaming is what keeps a long prefill alive: tokens keep arriving on the socket, so there
        is no single read that can hit a client timeout, and the caller sees progress instead of a
        request that looks dead for minutes. `emit` receives text deltas as they arrive.
        """
        import urllib.request

        url = f"{session.base or self.base}/v1/chat/completions"
        payload = {"model": session.model, "messages": session.messages, "stream": True}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        for key, value in session.opts.items():
            if value is not None:
                payload[key] = value

        req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
        req.add_header("Content-Type", "application/json")
        if self.key:
            req.add_header("Authorization", f"Bearer {self.key}")

        content, calls, usage, saw_sse, raw_body = [], {}, None, False, []
        splitter = StreamSplitter()
        try:
            with urllib.request.urlopen(req, timeout=self.read_timeout) as resp:
                self._active_response(session, resp)
                try:
                    for raw in resp:
                        if session.cancel_event.is_set():
                            break
                        line = raw.decode("utf-8", "replace").strip()
                        if not line:
                            continue
                        if not line.startswith("data:"):
                            raw_body.append(line)
                            continue
                        saw_sse = True
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(chunk, dict) and chunk.get("error"):
                            raise ValueError(str(chunk["error"].get("message") or chunk["error"]))
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                        for choice in chunk.get("choices") or []:
                            delta = choice.get("delta") or {}
                            text = delta.get("content")
                            if text:
                                visible = splitter.feed(text)
                                if visible:
                                    content.append(visible)
                                    if emit:
                                        emit(visible)
                            for piece in delta.get("tool_calls") or []:
                                slot = calls.setdefault(piece.get("index", 0),
                                                        {"id": None, "type": "function",
                                                         "function": {"name": "", "arguments": ""}})
                                if piece.get("id"):
                                    slot["id"] = piece["id"]
                                fn = piece.get("function") or {}
                                if fn.get("name"):
                                    slot["function"]["name"] = fn["name"]
                                if fn.get("arguments"):
                                    slot["function"]["arguments"] += fn["arguments"]
                finally:
                    self._active_response(session, None)
        except Exception:
            if session.cancel_event.is_set():
                return {"role": "assistant", "content": "".join(content)}
            raise
        if session.cancel_event.is_set():
            return {"role": "assistant", "content": "".join(content)}

        if not saw_sse and raw_body:
            try:
                body = json.loads(chr(10).join(raw_body))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Joshua returned an unreadable body: {exc}") from exc
            if isinstance(body, dict) and body.get("error"):
                raise ValueError(str(body["error"].get("message") or body["error"]))
            msg = ((body.get("choices") or [{}])[0].get("message") or {}) if isinstance(body, dict) else {}
            if msg.get("content"):
                content.append(msg["content"])
                if emit:
                    emit(msg["content"])
            for i, call in enumerate(msg.get("tool_calls") or []):
                calls[i] = {"id": call.get("id"), "type": "function",
                            "function": {"name": call.get("function", {}).get("name", ""),
                                         "arguments": call.get("function", {}).get("arguments", "")}}
        elif not saw_sse:
            raise ValueError("Joshua returned an empty response")

        tail = splitter.flush()
        if tail:
            content.append(tail)
            if emit:
                emit(tail)
        for index, call in enumerate(splitter.calls):
            calls.setdefault(len(calls) + index, call)

        message = {"role": "assistant"}
        if content:
            message["content"] = "".join(content)
        if calls:
            message["tool_calls"] = [calls[i] for i in sorted(calls)]
            for i, call in enumerate(message["tool_calls"]):
                call["id"] = call.get("id") or f"call_{i}"
        if usage:
            message["usage"] = usage
        return message

    # ---- tools ----
    def _path(self, session, raw):
        p = Path(os.path.expanduser(str(raw)))
        return p if p.is_absolute() else (session.cwd / p)

    def run_tool(self, session, name, args):
        try:
            if name == "list_dir":
                p = self._path(session, args.get("path", "."))
                entries = sorted(os.listdir(p))[:400]
                return "\n".join(entries) or "(empty)"
            if name == "read_file":
                p = self._path(session, args["path"])
                text = p.read_text(errors="replace").splitlines()
                s = max(1, int(args.get("start_line") or 1))
                e = min(len(text), int(args.get("end_line") or len(text)))
                body = "\n".join(f"{i}: {text[i-1]}" for i in range(s, e + 1))
                return body[:TOOL_OUTPUT_LIMIT] or "(empty file)"
            if name == "write_file":
                p = self._path(session, args["path"])
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(args.get("content", ""))
                return f"wrote {p} ({len(args.get('content',''))} bytes)"
            if name == "edit_file":
                p = self._path(session, args["path"])
                text = p.read_text()
                old, new = args["old_text"], args.get("new_text", "")
                if old not in text:
                    return "ERROR: old_text not found; read the file and retry with an exact block"
                p.write_text(text.replace(old, new, 1))
                return f"edited {p}"
            if name == "run_command":
                process = subprocess.Popen(
                    ["/bin/bash", "-lc", args["command"]], cwd=session.cwd,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    start_new_session=True)
                with session.response_lock:
                    session.active_process = process
                    cancelled = session.cancel_event.is_set()
                try:
                    if cancelled:
                        self._stop_process(process)
                        return "ERROR: command cancelled"
                    deadline = time.monotonic() + float(args.get("timeout_s") or 120)
                    while True:
                        if session.cancel_event.is_set():
                            self._stop_process(process)
                            return "ERROR: command cancelled"
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            self._stop_process(process)
                            return "ERROR: command timed out"
                        try:
                            stdout, stderr = process.communicate(timeout=min(0.25, remaining))
                            if session.cancel_event.is_set():
                                return "ERROR: command cancelled"
                            break
                        except subprocess.TimeoutExpired:
                            continue
                    body = f"exit {process.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
                    return body[:TOOL_OUTPUT_LIMIT]
                finally:
                    with session.response_lock:
                        if session.active_process is process:
                            session.active_process = None
            return f"ERROR: unknown tool {name}"
        except subprocess.TimeoutExpired:
            return "ERROR: command timed out"
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"

    # ---- ACP ----
    def send(self, msg):
        with self._send_lock:
            sys.stdout.write(json.dumps(msg) + "\n")
            sys.stdout.flush()

    def note(self, session, text):
        self.send({"jsonrpc": "2.0", "method": "session/update", "params": {
            "sessionId": session.id,
            "update": {"sessionUpdate": "agent_message_chunk",
                       "content": {"type": "text", "text": text}}}})

    CONTROLS = [
        ("context_window", "Context window", ["default", "4096", "8192", "16384", "32768", "65536"]),
        ("reasoning_effort", "Reasoning effort",
         ["default", "none", "minimal", "low", "medium", "high", "xhigh"]),
        ("max_tokens", "Max output tokens", ["default", "512", "1024", "2048", "4096", "8192"]),
        ("temperature", "Temperature", ["default", "0.0", "0.2", "0.5", "0.7", "1.0"]),
        ("top_p", "Top-p", ["default", "0.9", "0.95", "1.0"]),
        ("top_k", "Top-k", ["default", "0", "20", "40", "100"]),
        ("min_p", "Min-p", ["default", "0.0", "0.02", "0.05", "0.1"]),
    ]

    def config_options(self, session):
        models = self.models() or [{"modelId": session.model, "name": session.model}]
        options = [{"type": "select", "id": "model", "name": "Model", "category": "model",
                    "currentValue": session.model,
                    "options": [{"value": m["modelId"], "name": m["name"]} for m in models]}]
        for key, label, values in self.CONTROLS:
            current = session.opts.get(key)
            current = "default" if current is None else str(current)
            options.append({"type": "select", "id": key, "name": label, "category": "model",
                            "currentValue": current,
                            "options": [{"value": v, "name": v} for v in values]})
        return options

    def apply_option(self, session, key, value):
        """Store a UI control; `default` clears it so Joshua's own default applies."""
        if key == "model":
            session.model, session.base = self.ensure_loaded(value or session.model)
            return
        if key not in session.opts:
            raise ValueError(f"unknown option {key!r}")
        if value in (None, "", "default"):
            session.opts[key] = None
            return
        if key in ("temperature", "top_p", "min_p"):
            session.opts[key] = float(value)
        elif key in ("top_k", "max_tokens", "context_window"):
            session.opts[key] = int(value)
        else:
            session.opts[key] = value

    def handle(self, req, prompt_reserved=False):
        method = req.get("method")
        params = req.get("params") or {}
        rid = req.get("id")

        if method == "initialize":
            return {"protocolVersion": 1,
                    "agentCapabilities": {"loadSession": False,
                                          "promptCapabilities": {"image": False, "audio": False,
                                                                 "embeddedContext": False}},
                    "agentInfo": {"name": "joshua", "version": "0.1.0"},
                    "authMethods": []}

        if method == "session/new":
            sid = f"joshua-{uuid.uuid4().hex[:12]}"
            s = Session(sid, params.get("cwd"), self.base)
            available = self.models()
            s.model = self.default_model or (available[0]["modelId"] if available else "default")
            self.sessions[sid] = s
            return {"sessionId": sid, "models": {"availableModels": available,
                                                 "currentModelId": s.model},
                    "configOptions": self.config_options(s)}

        if method == "session/set_config_option":
            s = self.sessions.get(params.get("sessionId"))
            if not s:
                raise ValueError("unknown session")
            if not s.prompt_lock.acquire(blocking=False):
                raise ValueError("cannot change configuration during an active prompt")
            try:
                self.apply_option(s, params.get("configId"), params.get("value"))
            finally:
                s.prompt_lock.release()
            return {"configOptions": self.config_options(s)}

        if method == "session/cancel":
            s = self.sessions.get(params.get("sessionId"))
            if s:
                self._cancel_session(s)
            return {}

        if method == "session/prompt":
            s = self.sessions.get(params.get("sessionId"))
            if not s:
                raise ValueError("unknown session")
            if not prompt_reserved:
                if not s.prompt_lock.acquire(blocking=False):
                    raise ValueError("a prompt is already active for this session")
                s.cancel_event.clear()
            try:
                text = "\n".join(p.get("text", "") for p in params.get("prompt", []) if p.get("type") == "text")
                if not s.messages:
                    s.messages.append({"role": "system", "content": SYSTEM.format(cwd=s.cwd)})
                s.messages.append({"role": "user", "content": text})
                try:
                    s.model, s.base = self.ensure_loaded(s.model, session=s)
                except PromptCancelled:
                    return {"stopReason": "cancelled"}
                for _ in range(MAX_TURNS):
                    if s.cancel_event.is_set():
                        return {"stopReason": "cancelled"}
                    msg = self.chat(s, TOOLS, emit=lambda t: self.note(s, t))
                    if s.cancel_event.is_set():
                        if msg.get("content"):
                            s.messages.append({"role": "assistant", "content": msg["content"]})
                        return {"stopReason": "cancelled"}
                    s.messages.append({"role": "assistant", **{k: msg[k] for k in ("content", "tool_calls") if msg.get(k) is not None}})
                    calls = msg.get("tool_calls") or []
                    if not calls:
                        break
                    for call in calls:
                        if s.cancel_event.is_set():
                            return {"stopReason": "cancelled"}
                        fn = call.get("function") or {}
                        try:
                            args = json.loads(fn.get("arguments") or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        result = self.run_tool(s, fn.get("name", ""), args)
                        s.messages.append({"role": "tool", "tool_call_id": call.get("id"),
                                           "content": result})
                return {"stopReason": "end_turn"}
            finally:
                s.prompt_lock.release()

        raise ValueError(f"unsupported method {method}")

    def run(self):
        def respond(req, prompt_reserved=False):
            try:
                result = self.handle(req, prompt_reserved=prompt_reserved)
                self.send({"jsonrpc": "2.0", "id": req["id"], "result": result})
            except Exception as exc:
                self.send({"jsonrpc": "2.0", "id": req["id"],
                           "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"}})

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue
            if req.get("method") == "session/cancel" and "id" not in req:
                self.handle(req)
                continue
            if "id" not in req:
                continue
            if req.get("method") == "session/prompt":
                session = self.sessions.get((req.get("params") or {}).get("sessionId"))
                if session and session.prompt_lock.acquire(blocking=False):
                    session.cancel_event.clear()
                    threading.Thread(target=respond, args=(req, True), daemon=True).start()
                else:
                    threading.Thread(target=respond, args=(req,), daemon=True).start()
            else:
                respond(req)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("JOSHUA_BASE_URL", "http://127.0.0.1:8080"))
    ap.add_argument("--api-key", default=os.environ.get("JOSHUA_API_KEY"))
    ap.add_argument("--model", default=os.environ.get("JOSHUA_MODEL"))
    ap.add_argument("--cpu-url", default=os.environ.get("JOSHUA_CPU_URL"),
                    help="second Joshua server for models that cannot fit the GPU")
    ap.add_argument("--catalog", action="append", default=None,
                    help="additional directory inside --model-dir to scan (repeatable)")
    ap.add_argument("--model-dir", default=os.environ.get("JOSHUA_MODEL_DIR"),
                    help="Joshua server's --model-dir; catalog results are restricted to this directory")
    args = ap.parse_args()
    Agent(args.base_url, args.api_key, args.model, args.catalog, args.cpu_url, args.model_dir).run()


if __name__ == "__main__":
    main()
