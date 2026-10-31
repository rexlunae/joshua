#!/usr/bin/env python3
"""Phase-0 synthetic locality experiment; no production affinity or tokenizer.

Stdlib only. Every policy/cell starts fresh worker processes and uses the same
bounded, closed-loop request trace. See docs/prefix-locality-measurement.md.
"""
import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def rendezvous(prefix_id, workers):
    return max(range(workers), key=lambda index: digest([prefix_id, index]))


def trace(args, prefix_count):
    rng = random.Random(args.seed)
    requests, total_bytes = [], 0
    for round_index in range(args.rounds):
        order = list(range(prefix_count))
        rng.shuffle(order)
        for prefix in order:
            bits = " ".join("a" if bit == "0" else "b" for bit in f"{prefix:032b}"[::-1])
            words = ["hello", "world"] * ((args.prefix_words + 1) // 2)
            system = bits + " " + " ".join(words[:args.prefix_words])
            suffix = " ".join("c" if bit == "0" else "d" for bit in f"{round_index:016b}"[::-1])
            prefix_id = digest([args.model_id, system])
            body = encoded({
                "model": args.model_id,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": f"hello {suffix} world"}],
                "temperature": 0, "seed": args.seed, "max_tokens": args.max_tokens,
                "stream": True, "stream_options": {"include_usage": True},
            })
            total_bytes += len(body)
            if total_bytes > 32 * 1024 * 1024:
                raise ValueError("trace exceeds 32 MiB; reduce rounds/prefixes/prefix-words")
            requests.append((prefix_id, body))
    return requests


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def request_result(body):
    return {"status": None, "ttft_ms": None, "latency_ms": None,
            "request_body_bytes": len(body), "response_body_bytes": 0,
            "worker": None, "error": None, "usage": None,
            "output_sha256": digest(["", "", None, None])}


def stream_request(address, body, timeout):
    started = time.perf_counter()
    connection = http.client.HTTPConnection(address, timeout=timeout)
    result = request_result(body)
    text, reasoning, finish, done = [], [], None, False
    try:
        connection.request("POST", "/v1/chat/completions", body,
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        result["status"] = response.status
        result["worker"] = response.getheader("x-joshua-worker")
        if response.status != 200:
            error_body = response.read(4096)
            result["response_body_bytes"] = len(error_body)
            result["error"] = error_body.decode(errors="replace")
        else:
            while True:
                line = response.readline(1024 * 1024 + 1)
                result["response_body_bytes"] += len(line)
                if len(line) > 1024 * 1024:
                    raise ValueError("SSE line exceeds 1 MiB")
                if result["response_body_bytes"] > 16 * 1024 * 1024:
                    raise ValueError("response exceeds 16 MiB")
                if not line:
                    break
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if "error" in event:
                    raise ValueError(f"stream error: {event['error']}")
                if event.get("usage") is not None:
                    result["usage"] = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    content = delta.get("content") or ""
                    thought = delta.get("reasoning_content") or ""
                    if content or thought:
                        if result["ttft_ms"] is None:
                            result["ttft_ms"] = (time.perf_counter() - started) * 1000
                        text.append(content)
                        reasoning.append(thought)
                    if choice.get("finish_reason") is not None:
                        finish = choice["finish_reason"]
            if not done or finish is None or result["usage"] is None:
                raise ValueError("incomplete stream: need finish, usage, and [DONE]")
    except (OSError, ValueError, http.client.HTTPException) as error:
        result["error"] = str(error)
    finally:
        connection.close()
    result["latency_ms"] = (time.perf_counter() - started) * 1000
    result["output_sha256"] = digest(["".join(text), "".join(reasoning), finish, result["usage"]])
    return result


def oracle_request(address, body, timeout, semaphore):
    started = time.perf_counter()
    admitted = semaphore.acquire(timeout=timeout)
    queued_ms = (time.perf_counter() - started) * 1000
    if admitted:
        try:
            result = stream_request(address, body, timeout)
        finally:
            semaphore.release()
    else:
        result = request_result(body)
        result.update(error="oracle admission timed out", latency_ms=0)
    result["client_queue_ms"] = queued_ms
    result["latency_ms"] += queued_ms
    if result["ttft_ms"] is not None:
        result["ttft_ms"] += queued_ms
    return result


def get_json(address, path):
    with urllib.request.urlopen(f"http://{address}{path}", timeout=2) as response:
        return json.load(response)


def check_contention(args):
    if not args.avoid_process_substring:
        return
    output = subprocess.check_output(["ps", "-axo", "pid=,comm=,args="], text=True)
    for line in output.splitlines():
        fields = line.split(None, 2)
        if len(fields) != 3 or int(fields[0]) == os.getpid():
            continue
        pid, command, arguments = fields
        if Path(command).name in {"bash", "fish", "sh", "zsh", "ssh", "ps"}:
            continue
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except OSError:
            cwd = ""
        needle = args.avoid_process_substring
        if (needle in cwd or needle in arguments.split()[0]
                or (Path(command).name in {"rustc", "cargo"} and needle in arguments)):
            raise RuntimeError(f"contention guard matched pid {pid}: {command}; discard interrupted cell")


class Processes:
    def __init__(self, args, directory):
        self.args = args
        self.directory = directory
        self.children = []
        self.files = []
        self.stop = threading.Event()
        self.rss = {"first": None, "last": None, "max": None, "samples": 0}
        self.monitor = None
        self.commands = []

    def start(self, name, command):
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        address = f"127.0.0.1:{port}"
        command = [str(self.args.binary), *command, "--addr", address]
        log = (self.directory / f"{name}.log").open("wb")
        self.files.append(log)
        environment = os.environ.copy()
        environment.update(RUST_LOG="joshua=debug", NO_COLOR="1",
                           RAYON_NUM_THREADS=str(self.args.threads))
        environment = {key: value for key, value in environment.items()
                       if not key.startswith("JOSHUA_")}
        child = subprocess.Popen(command, stdout=log, stderr=log, env=environment)
        self.children.append(child)
        self.commands.append({"name": name, "pid": child.pid, "argv": command})
        deadline = time.monotonic() + self.args.startup_timeout
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError(f"{name} exited ({child.returncode}); see {log.name}")
            try:
                get_json(address, "/health")
                if b"listening on" in Path(log.name).read_bytes():
                    return address
            except (OSError, ValueError, urllib.error.URLError):
                pass
            time.sleep(0.05)
        raise RuntimeError(f"{name} startup timed out; see {log.name}")

    def sample(self):
        while not self.stop.is_set():
            try:
                pids = ",".join(str(child.pid) for child in self.children)
                output = subprocess.check_output(["ps", "-o", "rss=", "-p", pids], text=True)
                rss = sum(int(value) * 1024 for value in output.split())
                if self.rss["first"] is None:
                    self.rss["first"] = rss
                self.rss.update(last=rss, max=max(self.rss["max"] or 0, rss),
                                samples=self.rss["samples"] + 1)
            except (OSError, ValueError, subprocess.CalledProcessError):
                pass
            self.stop.wait(0.1)

    def close(self):
        self.stop.set()
        if self.monitor:
            self.monitor.join()
        for child in reversed(self.children):
            if child.poll() is None:
                child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for log in self.files:
            log.close()


def reuse_evidence(directory, pattern="worker-*.log", include_records=False):
    records, roots = [], 0
    root_resumes, parked_resumes, pool_rewinds = 0, 0, 0
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    for log in sorted(directory.glob(pattern)):
        for line in log.read_text(errors="replace").splitlines():
            line = ansi.sub("", line)
            if "prefix cache: system-prompt root" in line and "stored=true" in line:
                roots += 1
            if "Continuing from the prefix cache" in line:
                if "root=true" in line:
                    root_resumes += 1
                elif "root=false" in line:
                    parked_resumes += 1
            if "Continuing from cached KV prefix after context edit" in line:
                pool_rewinds += 1
            if "Completion finished" not in line:
                continue
            fields = dict(re.findall(r"(prompt_tokens|prefill_tokens|reused_tokens|prefill_tps)=([0-9.eE+-]+)", line))
            if all(key in fields for key in ("prompt_tokens", "prefill_tokens", "reused_tokens", "prefill_tps")):
                records.append({key: float(value) for key, value in fields.items()})
    prefilled = sum(record["prefill_tokens"] for record in records)
    timed_tokens = sum(record["prefill_tokens"] for record in records if record["prefill_tps"] > 0)
    seconds = sum(record["prefill_tokens"] / record["prefill_tps"]
                  for record in records if record["prefill_tps"] > 0)
    result = {"completion_records": len(records), "stored_root_events": roots,
            "root_resume_events": root_resumes, "parked_resume_events": parked_resumes,
            "pool_rewind_events": pool_rewinds,
            "reused_tokens": sum(record["reused_tokens"] for record in records) if records else None,
            "requests_with_reuse": sum(record["reused_tokens"] > 0 for record in records) if records else None,
            "prefilled_tokens": prefilled if records else None,
            "timed_prefill_tokens": timed_tokens,
            "prefill_seconds_per_token_from_logs": seconds / timed_tokens if timed_tokens else None,
            "scope": "aggregate engine log counters, includes warm-pool rewind; not per-request cache hits"}
    if include_records:
        result["records"] = records
    return result


def queue_evidence(directory):
    log = directory / "router.log"
    values = []
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            line = re.sub(r"\x1b\[[0-9;]*m", "", line)
            if "routed stream finished" in line:
                match = re.search(r"queue_ms=(\d+)", line)
                if match:
                    values.append(int(match[1]))
    return {"records": len(values), "p50": percentile(values, .5), "p95": percentile(values, .95),
            "scope": "router admission elapsed ms, integer-rounded; not engine execution wait"}


def run_cell(args, directory, requests, concurrency, policy):
    check_contention(args)
    directory.mkdir(parents=True)
    processes = Processes(args, directory)
    try:
        workers = [processes.start(f"worker-{index}", [
            "serve", "--model", str(args.model), "--device", "cpu",
            "--n-ctx", str(args.n_ctx), "--max-concurrency", str(args.slots),
            "--prefix-cache-mib", str(args.cache_mib),
            "--prefix-cache-root-min", str(args.root_min),
        ]) for index in range(args.workers)]
        infos = [get_json(worker, "/v1/worker/info") for worker in workers]
        if any(info["model"] != args.model_id for info in infos):
            raise ValueError("--model-id does not match worker model identity")
        semaphores = [threading.Semaphore(args.slots) for _ in workers]
        router = processes.start("router", [
            "route", "--worker", ",".join(workers), "--queue-depth", str(args.queue_depth),
            "--queue-timeout-ms", str(int(args.timeout * 1000)),
            "--response-timeout-ms", str(int(args.timeout * 1000)),
        ]) if policy == "load" else None
        processes.monitor = threading.Thread(target=processes.sample, daemon=True)
        processes.monitor.start()
        results = []
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            for offset in range(0, len(requests), concurrency):
                check_contention(args)
                wave = requests[offset:offset + concurrency]
                futures = []
                for prefix_id, body in wave:
                    worker = rendezvous(prefix_id, len(workers))
                    address = router if policy == "load" else workers[worker]
                    if policy == "oracle":
                        futures.append(executor.submit(oracle_request, address, body, args.timeout,
                                                       semaphores[worker]))
                    else:
                        futures.append(executor.submit(stream_request, address, body, args.timeout))
                for index, future in enumerate(futures):
                    result = future.result()
                    result.update(request_index=offset + index, prefix_id=wave[index][0],
                                  request_sha256=hashlib.sha256(wave[index][1]).hexdigest())
                    if policy == "oracle":
                        result["worker"] = workers[rendezvous(wave[index][0], len(workers))]
                    worker_address = (result["worker"] or "").removeprefix("http://").rstrip("/")
                    result["worker_index"] = workers.index(worker_address) if worker_address in workers else None
                    results.append(result)
                check_contention(args)
        elapsed = time.perf_counter() - started
        rss = dict(processes.rss)
        commands = list(processes.commands)
    finally:
        processes.close()
    successes = [result for result in results if result["status"] == 200 and not result["error"]]
    tokens = sum(result["usage"]["completion_tokens"] for result in successes)
    round_size = len(requests) // args.rounds
    phases = {}
    for name, selected in [("first_round", [result for result in successes if result["request_index"] < round_size]),
                           ("later_rounds", [result for result in successes if result["request_index"] >= round_size])]:
        timings = [result["ttft_ms"] for result in selected if result["ttft_ms"] is not None]
        phases[name] = {"requests": len(selected), "ttft_ms_p50": percentile(timings, .5),
                        "ttft_ms_p95": percentile(timings, .95)}
    return {"policy": policy, "elapsed_seconds": elapsed, "requests": results,
            "phases": phases,
            "requests_per_worker": [sum(result["worker_index"] == index for result in results)
                                    for index in range(args.workers)],
            "successful_requests": len(successes), "errors": len(results) - len(successes),
            "ttft_ms": {"p50": percentile([result["ttft_ms"] for result in successes if result["ttft_ms"] is not None], .5),
                        "p95": percentile([result["ttft_ms"] for result in successes if result["ttft_ms"] is not None], .95)},
            "latency_ms": {"p50": percentile([result["latency_ms"] for result in successes], .5),
                           "p95": percentile([result["latency_ms"] for result in successes], .95)},
            "completion_tokens": tokens, "completion_tokens_per_second": tokens / elapsed,
            "prompt_tokens": sum(result["usage"]["prompt_tokens"] for result in successes),
            "request_body_bytes": sum(len(body) for _, body in requests),
            "response_body_bytes": sum(result["response_body_bytes"] for result in results),
            "queue_ms": queue_evidence(directory),
            "client_queue_ms": {"p50": percentile([result["client_queue_ms"] for result in results
                                                    if "client_queue_ms" in result], .5),
                                "p95": percentile([result["client_queue_ms"] for result in results
                                                    if "client_queue_ms" in result], .95)},
            "rss_sampled_process_sum_bytes": rss,
            "worker_info_before": infos, "processes": commands,
            "reuse": reuse_evidence(directory)}


def compare(left, right):
    if not left["requests"] or len(left["requests"]) != len(right["requests"]):
        return False
    return all(not first["error"] and not second["error"] and first["status"] == second["status"] == 200
               and first["request_sha256"] == second["request_sha256"]
               and first["output_sha256"] == second["output_sha256"]
               for first, second in zip(left["requests"], right["requests"]))


def calibrate(args, repeat):
    check_contention(args)
    directory = args.output / f"calibration-r{repeat}"
    directory.mkdir()
    processes = Processes(args, directory)
    _, body = trace(args, 1)[0]
    results = {}
    try:
        workers = [processes.start(f"worker-{index}", [
            "serve", "--model", str(args.model), "--device", "cpu", "--n-ctx", str(args.n_ctx),
            "--max-concurrency", str(args.slots), "--prefix-cache-mib", str(args.cache_mib),
            "--prefix-cache-root-min", str(args.root_min),
        ]) for index in range(2)]
        conditioning = stream_request(workers[0], body, args.timeout)
        if conditioning["error"] or conditioning["status"] != 200:
            raise RuntimeError(f"calibration conditioning failed: {conditioning}")
        order = ["warm", "cold"] if repeat % 2 == 0 else ["cold", "warm"]
        for policy in order:
            check_contention(args)
            result = stream_request(workers[0 if policy == "warm" else 1], body, args.timeout)
            result["request_sha256"] = hashlib.sha256(body).hexdigest()
            results[policy] = result
        check_contention(args)
    finally:
        processes.close()
    evidence = {}
    for policy, worker in [("warm", 0), ("cold", 1)]:
        records = reuse_evidence(directory, f"worker-{worker}.log", include_records=True)["records"]
        evidence[policy] = records[-1] if records else None
    return {"repeat": repeat, "order": order, "conditioning": conditioning,
            "requests": results, "engine_completion": evidence,
            "outputs_match": compare({"requests": [results["warm"]]}, {"requests": [results["cold"]]}),
            "processes": processes.commands}


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def numbers(value):
    try:
        return [positive(item) for item in value.split(",")]
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from error


def interrupted(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def main():
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", help="default: model file stem")
    parser.add_argument("--output", type=Path, required=True, help="new directory; never overwrite a run")
    for name, default in [("workers", 2), ("rounds", 4), ("repeats", 2), ("prefix-words", 64),
                          ("max-tokens", 8), ("n-ctx", 1024), ("cache-mib", 128),
                          ("root-min", 16), ("slots", 1), ("threads", 2), ("queue-depth", 16)]:
        parser.add_argument(f"--{name}", type=positive, default=default)
    parser.add_argument("--prefix-counts", type=numbers, default=[2, 4, 12])
    parser.add_argument("--concurrency", type=numbers, default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=201156)
    parser.add_argument("--timeout", type=positive, default=120)
    parser.add_argument("--startup-timeout", type=positive, default=120)
    parser.add_argument("--skip-calibration", action="store_true", help="omit two-worker warm/cold control")
    parser.add_argument("--avoid-process-substring", help="abort on matching executable/build args or Linux cwd")
    args = parser.parse_args()
    if args.seed < 0 or args.seed >= 2 ** 63:
        parser.error("seed must fit a nonnegative i64")
    total_requests = sum(args.prefix_counts) * args.rounds * len(args.concurrency) * args.repeats * 2 + args.repeats * 3
    if total_requests > 20000 or max(args.concurrency) > 256 or args.workers > 16 or args.prefix_words > 4096:
        parser.error("bounded harness: <=20000 requests, <=256 concurrency, <=16 workers, <=4096 prefix-words")
    args.binary, args.model = args.binary.resolve(), args.model.resolve()
    args.model_id = args.model_id or args.model.stem
    if not args.binary.is_file() or not args.model.is_file():
        parser.error("binary and model must be existing files")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    with args.binary.open("rb") as source:
        binary_hash = hashlib.file_digest(source, "sha256").hexdigest()
    with args.model.open("rb") as source:
        model_hash = hashlib.file_digest(source, "sha256").hexdigest()
    tokenizer_path = args.model.parent / "tokenizer.json"
    with tokenizer_path.open("rb") as source:
        tokenizer_hash = hashlib.file_digest(source, "sha256").hexdigest()
    report = {"schema_version": 1, "parameters": metadata, "host": platform.uname()._asdict(),
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "binary_sha256": binary_hash, "model_sha256": model_hash,
              "tokenizer_sha256": tokenizer_hash, "logical_cpus": os.cpu_count(),
              "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "load_average_before": os.getloadavg(),
              "model_size_bytes": args.model.stat().st_size, "cells": [], "calibration": [],
              "all_outputs_match": True, "completed": False}
    try:
        if not args.skip_calibration:
            for repeat in range(args.repeats):
                calibration = calibrate(args, repeat)
                report["calibration"].append(calibration)
                report["all_outputs_match"] &= calibration["outputs_match"]
        for prefix_count in args.prefix_counts:
            requests = trace(args, prefix_count)
            for concurrency in args.concurrency:
                for repeat in range(args.repeats):
                    order = ["load", "oracle"] if repeat % 2 == 0 else ["oracle", "load"]
                    cell = {"prefix_count": prefix_count, "concurrency": concurrency, "repeat": repeat,
                            "policy_order": order, "trace_sha256": digest([hashlib.sha256(body).hexdigest() for _, body in requests]),
                            "oracle_prefixes_per_worker": [sum(rendezvous(prefix_id, args.workers) == worker
                                for prefix_id in {prefix_id for prefix_id, _ in requests}) for worker in range(args.workers)],
                            "policies": {}}
                    for policy in order:
                        directory = args.output / f"p{prefix_count}-c{concurrency}-r{repeat}-{policy}"
                        cell["policies"][policy] = run_cell(args, directory, requests, concurrency, policy)
                    cell["outputs_match"] = compare(cell["policies"]["load"], cell["policies"]["oracle"])
                    report["all_outputs_match"] &= cell["outputs_match"]
                    report["cells"].append(cell)
                    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps({key: value for key, value in cell.items() if key != "policies"}), flush=True)
        report["completed"] = True
    except BaseException as error:
        report["fatal_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["load_average_after"] = os.getloadavg()
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["all_outputs_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
