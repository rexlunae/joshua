import argparse
import http.server
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import prefix_locality as locality


class LocalityTests(unittest.TestCase):
    def test_sweeps_reject_duplicate_values_before_starting(self):
        self.assertEqual(locality.numbers("2,6,12"), [2, 6, 12])
        for value in ["1,1", "2,2", "2,6,2", "1,01"]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(argparse.ArgumentTypeError, "unique"):
                    locality.numbers(value)

    def test_trace_identity_and_prefix_placement(self):
        args = argparse.Namespace(seed=201156, rounds=3, prefix_words=8,
                                  model_id="tiny", max_tokens=4)
        first = locality.trace(args, 12)
        self.assertEqual(first, locality.trace(args, 12))
        self.assertEqual(len({key for key, _ in first}), 12)
        self.assertEqual(len({body for _, body in first}), 36)
        for prefix_id, body in first:
            request = json.loads(body)
            self.assertEqual(request["seed"], args.seed)
            self.assertEqual(request["temperature"], 0)
            self.assertEqual(prefix_id, locality.digest(["tiny", request["messages"][0]["content"]]))
            self.assertIn(locality.rendezvous(prefix_id, 2), [0, 1])
        self.assertEqual([locality.rendezvous(str(index), 2) for index in range(8)],
                         [0, 1, 0, 1, 1, 0, 1, 1])
        for prefix_id, _ in first:
            selected = locality.rendezvous(prefix_id, 3)
            if selected != 2:
                self.assertEqual(selected, locality.rendezvous(prefix_id, 2))

    def test_compare_fails_closed(self):
        record = dict(status=200, error=None, request_sha256="same", output_sha256="answer")
        left = {"requests": [record]}
        self.assertTrue(locality.compare(left, left))
        for changed in [dict(error="timeout"), dict(status=503), dict(output_sha256="different"),
                        dict(request_sha256="other")]:
            self.assertFalse(locality.compare(left, {"requests": [{**record, **changed}]}))
        self.assertFalse(locality.compare(left, {"requests": []}))
        self.assertFalse(locality.compare({"requests": []}, {"requests": []}))

    def test_nearest_rank_percentiles(self):
        self.assertIsNone(locality.percentile([], .95))
        self.assertEqual(locality.percentile([3, 1, 2, 4], .5), 2)
        self.assertEqual(locality.percentile([3, 1, 2, 4], .95), 4)

    def test_contention_guard_ignores_own_arguments_but_rejects_build(self):
        args = argparse.Namespace(avoid_process_substring="joshua-117")
        with patch.object(locality.subprocess, "check_output", return_value=
                          f"{os.getpid()} python3 python3 --avoid-process-substring joshua-117\n"):
            locality.check_contention(args)
        with patch.object(locality.subprocess, "check_output", return_value=
                          "999999 rustc rustc --out-dir /tmp/joshua-117/target\n"), \
                patch.object(locality.os, "readlink", return_value="/other"):
            with self.assertRaisesRegex(RuntimeError, "contention guard"):
                locality.check_contention(args)

    def test_oracle_bounds_per_worker_admission_and_includes_wait(self):
        semaphore = threading.Semaphore(1)
        semaphore.acquire()
        outcome = []
        waiting = threading.Event()

        class Queue:
            def acquire(self, timeout):
                waiting.set()
                return semaphore.acquire(timeout=timeout)

            def release(self):
                semaphore.release()

        with patch.object(locality, "stream_request", return_value={"ttft_ms": 2, "latency_ms": 3}):
            thread = threading.Thread(target=lambda: outcome.append(
                locality.oracle_request("unused", b"{}", 1, Queue())))
            thread.start()
            try:
                self.assertTrue(waiting.wait(2))
                time.sleep(.03)
                self.assertFalse(outcome)
            finally:
                semaphore.release()
                thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
        self.assertGreaterEqual(outcome[0]["client_queue_ms"], 25)
        self.assertAlmostEqual(outcome[0]["ttft_ms"], outcome[0]["client_queue_ms"] + 2)

    def test_oracle_admission_timeout_fails_without_sending(self):
        with patch.object(locality, "stream_request") as send:
            result = locality.oracle_request("unused", b"{}", .01, threading.Semaphore(0))
        send.assert_not_called()
        self.assertEqual(result["error"], "oracle admission timed out")
        self.assertIsNone(result["status"])
        self.assertIsNone(result["ttft_ms"])
        self.assertGreaterEqual(result["latency_ms"], 10)
        self.assertEqual(result["latency_ms"], result["client_queue_ms"])

    def test_oracle_releases_slot_on_request_exception(self):
        semaphore = threading.Semaphore(1)
        with patch.object(locality, "stream_request", side_effect=ValueError("bad stream")):
            with self.assertRaisesRegex(ValueError, "bad stream"):
                locality.oracle_request("unused", b"{}", 1, semaphore)
        self.assertTrue(semaphore.acquire(blocking=False))

    def test_log_evidence_is_not_invented(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.assertIsNone(locality.reuse_evidence(directory)["reused_tokens"])
            (directory / "worker-0.log").write_text(
                "prefix cache: system-prompt root tokens=64 stored=true\n"
                "Continuing from the prefix cache reused_tokens=64 root=true\n"
                "Completion finished prompt_tokens=70 prefill_tokens=6 reused_tokens=64 prefill_tps=12 decode_tokens=4\n")
            evidence = locality.reuse_evidence(directory)
            self.assertEqual(evidence["reused_tokens"], 64)
            self.assertEqual(evidence["prefill_seconds_per_token_from_logs"], 1 / 12)
            self.assertEqual(evidence["stored_root_events"], 1)
            self.assertEqual(evidence["root_resume_events"], 1)
            (directory / "router.log").write_text(
                "routed stream finished request_id=a queue_ms=17 outcome=complete\n"
                "routed request finished queue_ms=99\n")
            self.assertEqual(locality.queue_evidence(directory)["p95"], 17)

    def test_stream_timing_requires_content_and_checks_terminal_events(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n')
                self.wfile.flush()
                time.sleep(.03)
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n')
                if not self.server.broken:
                    self.wfile.write(b'data: {"choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1}}\n\n')
                    self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.broken = False
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            result = locality.stream_request(f"127.0.0.1:{server.server_port}", b"{}", 2)
            self.assertIsNone(result["error"])
            self.assertGreaterEqual(result["ttft_ms"], 25)
            self.assertGreater(result["response_body_bytes"], 0)
            server.broken = True
            result = locality.stream_request(f"127.0.0.1:{server.server_port}", b"{}", 2)
            self.assertIn("incomplete stream", result["error"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_stream_counts_entire_body_after_done(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(self.server.payload)))
                self.end_headers()
                self.wfile.write(self.server.payload)
                self.wfile.flush()

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            for newline in [b"\n", b"\r\n"]:
                with self.subTest(newline=newline):
                    server.payload = (
                        b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
                        b'data: {"choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1}}\n\n'
                        b'data: [DONE]\n\n: trailing comment\n\n'
                    ).replace(b"\n", newline)
                    result = locality.stream_request(f"127.0.0.1:{server.server_port}", b"{}", 2)
                    self.assertIsNone(result["error"])
                    self.assertEqual(result["response_body_bytes"], len(server.payload))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
