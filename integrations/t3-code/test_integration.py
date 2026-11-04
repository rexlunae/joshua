import importlib.util
import io
import json
import os
import shlex
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from importlib.machinery import SourceFileLoader
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch


ROOT = Path(__file__).parent


def load_module(name, path):
    if path.suffix:
        spec = importlib.util.spec_from_file_location(name, path)
    else:
        loader = SourceFileLoader(name, str(path))
        spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


acp = load_module("joshua_acp", ROOT / "joshua_acp.py")
control = load_module("joshua_control", ROOT / "joshua-control")


class StreamSplitterTests(unittest.TestCase):
    def test_only_explicit_thinking_tags_are_stripped(self):
        splitter = acp.StreamSplitter()
        output = [
            splitter.feed("I was thinking about the fix. <thi"),
            splitter.feed("nk>private</think> Done."),
            splitter.flush(),
        ]
        self.assertEqual("".join(output), "I was thinking about the fix.  Done.")


class CatalogTests(unittest.TestCase):
    def test_catalog_is_confined_to_server_model_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "models"
            outside = Path(temp) / "outside"
            root.mkdir()
            outside.mkdir()
            (root / "inside.gguf").write_bytes(b"GGUF" + b"\0" * 24)
            (root / "tokenizer.json").write_text("{}")
            (outside / "outside.gguf").write_bytes(b"GGUF" + b"\0" * 24)
            (outside / "tokenizer.json").write_text("{}")
            (root / "linked.gguf").symlink_to(outside / "outside.gguf")

            agent = acp.Agent("http://127.0.0.1:8080", None, None,
                              catalog_roots=[str(root)], model_dir=str(root))

            self.assertEqual(set(agent.catalog()), {"inside"})
            self.assertEqual(acp.Agent("http://127.0.0.1:8080", None, None).catalog(), {})

    def test_nested_server_model_directory_is_scanned(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp) / "models"
            root = parent / "project"
            root.mkdir(parents=True)
            (root / "inside.gguf").write_bytes(b"GGUF" + b"\0" * 24)
            (root / "tokenizer.json").write_text("{}")

            agent = acp.Agent(
                "http://127.0.0.1:8080", None, None,
                catalog_roots=[str(parent)], model_dir=str(root),
            )

            self.assertEqual(set(agent.catalog()), {"inside"})


class CancellationTests(unittest.TestCase):
    def test_cancel_closes_active_response(self):
        class FakeResponse:
            closed = False

            def close(self):
                self.closed = True

        session = acp.Session("session", ".")
        response = FakeResponse()
        session.active_response = response

        acp.Agent._cancel_session(session)

        self.assertTrue(session.cancel_event.is_set())
        self.assertTrue(response.closed)

    def test_cancel_interrupts_a_blocked_stream_read(self):
        started = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunk = {"choices": [{"delta": {"content": "partial"}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                started.set()
                time.sleep(10)

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        agent = acp.Agent(f"http://127.0.0.1:{server.server_port}", None, "model")
        session = acp.Session("session", ".")
        session.model = "model"
        received = threading.Event()
        result = {}

        def chat():
            result["message"] = agent.chat(session, emit=lambda _: received.set())

        thread = threading.Thread(target=chat)
        thread.start()
        try:
            self.assertTrue(started.wait(2))
            self.assertTrue(received.wait(2))
            acp.Agent._cancel_session(session)
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(result["message"]["content"], "partial")
        finally:
            server.shutdown()
            server.server_close()

    def test_cancel_is_processed_while_prompt_is_blocked(self):
        agent = acp.Agent("http://127.0.0.1:8080", None, "model")
        session = acp.Session("session", ".")
        session.model = "model"
        agent.sessions[session.id] = session
        prompt_started = threading.Event()
        prompt_replied = threading.Event()
        responses = []

        def ensure_loaded(model, session=None):
            return model, agent.base

        def chat(session, *args, **kwargs):
            prompt_started.set()
            session.cancel_event.wait(2)
            return {"role": "assistant", "content": ""}

        def send(message):
            responses.append(message)
            if message.get("id") == "prompt":
                prompt_replied.set()

        class Requests:
            def __iter__(self):
                yield json.dumps({"id": "prompt", "method": "session/prompt", "params": {
                    "sessionId": session.id, "prompt": [{"type": "text", "text": "hello"}]}})
                if not prompt_started.wait(2):
                    raise AssertionError("prompt handler did not start before the next request")
                yield json.dumps({"method": "session/cancel", "params": {
                    "sessionId": session.id}})

        agent.ensure_loaded = ensure_loaded
        agent.chat = chat
        agent.send = send
        with patch.object(acp.sys, "stdin", Requests()):
            agent.run()

        self.assertTrue(prompt_replied.wait(2))
        prompt_response = next(message for message in responses if message.get("id") == "prompt")
        self.assertEqual(prompt_response["result"]["stopReason"], "cancelled")
        self.assertEqual([message.get("id") for message in responses], ["prompt"])

    def test_cancel_terminates_a_running_command(self):
        agent = acp.Agent("http://127.0.0.1:8080", None, "model")
        session = acp.Session("session", ".")
        started = threading.Event()
        result = {}
        real_popen = acp.subprocess.Popen

        def tracked_popen(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            started.set()
            return process

        def run_command():
            result["body"] = agent.run_tool(
                session, "run_command", {"command": "sleep 10", "timeout_s": 30})

        with patch.object(acp.subprocess, "Popen", tracked_popen):
            thread = threading.Thread(target=run_command)
            thread.start()
            self.assertTrue(started.wait(2))
            acp.Agent._cancel_session(session)
            thread.join(4)

        self.assertFalse(thread.is_alive())
        self.assertEqual(result["body"], "ERROR: command cancelled")


class ControlTests(unittest.TestCase):
    def test_api_key_is_sent_through_stdin_not_curl_arguments(self):
        secret = 'secret "with quotes"'
        with patch.object(control, "run", return_value=SimpleNamespace(
                returncode=0, stdout='{"data":[]}')) as run:
            response = control.api("host", 8080, "/v1/models", api_key=secret)

        command = run.call_args.args[1]
        self.assertNotIn(secret, command)
        self.assertIn("--config -", command)
        self.assertEqual(run.call_args.kwargs["input"],
                         'header = "Authorization: Bearer secret \\"with quotes\\""\n')
        self.assertEqual(response, {"data": []})

    def test_start_accepts_model_id_from_gguf_metadata(self):
        args = SimpleNamespace(host="local", port=8080, model="/models/alias.gguf",
                               model_dir="/models", api_key="key", addr=None, ctx=8192,
                               device="cpu", dense_placement=None, expert_placement=None, extra="")
        with patch.object(control, "run", return_value=SimpleNamespace(returncode=0, stdout="")):
            with patch.object(control, "api", return_value={"data": [{"id": "metadata-name"}]}) as api:
                with patch.object(control.time, "sleep"):
                    with patch("sys.stdout", new_callable=io.StringIO) as output:
                        control.cmd_start(args)

        self.assertEqual(api.call_args.kwargs["api_key"], "key")
        self.assertIn("metadata-name", output.getvalue())

    def test_worker_ip_ignores_invalid_and_non_ipv4_addresses(self):
        output = "192.0.2.1; touch /tmp/pwned fe80::1 198.51.100.4"
        with patch.object(control, "run", return_value=SimpleNamespace(stdout=output)):
            self.assertEqual(control.worker_ipv4("host"), "198.51.100.4")
        with patch.object(control, "run", return_value=SimpleNamespace(
                stdout="192.0.2.1; touch /tmp/pwned fe80::1")):
            self.assertIsNone(control.worker_ipv4("host"))

    def test_remote_home_paths_quote_shell_metacharacters(self):
        malicious = str(Path.home() / "models/$(touch should-not-run)")
        with patch.object(control, "resolve", return_value="remote"):
            result = control.shell_path("remote", malicious)

        self.assertEqual(result, '"$HOME"/' + shlex.quote("models/$(touch should-not-run)"))

    def test_cluster_worker_probe_runs_from_router_host(self):
        with patch.object(control, "run", return_value=SimpleNamespace(returncode=0)) as run:
            self.assertTrue(control.tcp_open_from("router", "192.0.2.1", 7070))

        self.assertEqual(run.call_args.args[0], "router")
        command = run.call_args.args[1]
        self.assertIn("192.0.2.1", command)
        self.assertIn("7070", command)

    def test_secret_is_written_to_stdin_not_command_arguments(self):
        secret = "sensitive value; $(not-a-command)"
        with patch.object(control, "run") as run:
            control.store_secret("host", '"$HOME/.config/joshua/cluster.env"',
                                 "JOSHUA_CLUSTER_KEY", secret)

        command = run.call_args.args[1]
        self.assertNotIn(secret, command)
        self.assertIn("chmod 600", command)
        self.assertEqual(run.call_args.kwargs["input"],
                         f"export JOSHUA_CLUSTER_KEY={shlex.quote(secret)}\n")

    def test_unload_uses_server_model_id_field(self):
        args = SimpleNamespace(host="local", port=8080, model="model-id")
        with patch.object(control, "api", return_value={"ok": True}) as api:
            with patch.object(control, "print"):
                control.cmd_unload(args)

        self.assertEqual(api.call_args.args[4], {"id": "model-id"})

    def test_install_restricts_settings_permissions_and_passes_model_dir(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            agent = home / ".t3/joshua/joshua_acp.py"
            agent.parent.mkdir(parents=True)
            agent.write_text("#!/bin/sh\nprintf '%s\\n' \"$JOSHUA_API_KEY\" \"$@\"\n")
            agent.chmod(0o700)
            commands = []

            def fake_run(host, command, **kwargs):
                commands.append((command, kwargs.get("input")))
                return SimpleNamespace(stdout="registered provider instance joshua_local",
                                       returncode=0)

            args = SimpleNamespace(host="local", port=8080, cpu_port=None, model_dir="/models")
            with patch.object(control, "HOME", home), patch.dict(os.environ, {"HOME": temp}):
                with patch.object(control, "run", side_effect=fake_run):
                    with patch("sys.stdout", new_callable=io.StringIO):
                        control.cmd_install(args)

            command, payload = commands[-1]
            script = shlex.split(command)[2]
            subprocess.run(["python3", "-c", script], input=payload, text=True,
                           capture_output=True, check=True,
                           env={**os.environ, "HOME": temp})
            settings = json.loads((home / ".t3/userdata/settings.json").read_text())
            provider_config = settings["providerInstances"]["joshua_local"]["config"]
            config_args = provider_config["commandArgs"]
            self.assertEqual(config_args[-2:], ["--model-dir", "/models"])
            self.assertEqual((home / ".t3/userdata/settings.json").stat().st_mode & 0o777, 0o600)
            launcher = Path(provider_config["commandPath"])
            self.assertEqual(launcher.stat().st_mode & 0o777, 0o700)
            self.assertIn("api-8080.env", launcher.read_text())
            self.assertNotIn("--api-key", launcher.read_text())
            api_env = home / ".config/joshua/api-8080.env"
            api_env.parent.mkdir(parents=True, exist_ok=True)
            api_env.write_text("export JOSHUA_API_KEY='integration-test-secret'\n")
            api_env.chmod(0o600)
            launched = subprocess.run(
                [str(launcher), "--base-url", "http://127.0.0.1:8080"],
                capture_output=True, text=True, check=True,
                env={**os.environ, "HOME": temp},
            )
            self.assertEqual(
                launched.stdout.splitlines(),
                ["integration-test-secret", "--base-url", "http://127.0.0.1:8080"],
            )

    def test_status_does_not_treat_error_envelope_as_healthy(self):
        args = SimpleNamespace(action="status", workers="", worker_port=7070,
                               port=8080, router="router", api_key="key")
        with patch.object(control, "api", return_value={"error": "worker unavailable"}):
            with patch.object(control, "tcp_open", return_value=False):
                with patch.object(control, "print") as output:
                    control.cmd_cluster(args)

        self.assertIn("controller not answering", output.call_args.args[0])

    def test_cluster_cli_accepts_api_key_for_status(self):
        result = subprocess.run(
            ["python3", str(ROOT / "joshua-control"), "cluster", "status",
             "--workers", "", "--api-key", "key", "--help"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--api-key", result.stdout)


if __name__ == "__main__":
    unittest.main()
