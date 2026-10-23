"""Compare full logits and summary metrics from bench_cubecl_real processes.
Usage: python3 compare_real.py GENERIC_DIR CUBECL_DIR OUTPUT_JSON [--require-pass]
Requires NumPy; binary dumps remain outside the source tree.
"""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import hashlib
from pathlib import Path
import sys
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("generic_dir", type=Path)
parser.add_argument("cubecl_dir", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--require-pass", action="store_true")
parser.add_argument("--allow-reduced-memory", action="store_true", help="accept lower candidate allocations, never higher")
args = parser.parse_args()
base_dir, candidate_dir, output = args.generic_dir, args.cubecl_dir, args.output
a = json.loads((base_dir / "summary.json").read_text())
b = json.loads((candidate_dir / "summary.json").read_text())
trace_a = json.loads((base_dir / "trace.json").read_text())
trace_b = json.loads((candidate_dir / "trace.json").read_text())
assert trace_a == trace_b, "different prompt/decode inputs"
for key in ["model", "model_bytes", "gpu", "prompt_tokens", "decode_tokens", "vocab", "logit_rows", "prefill_chunk"]:
    assert a[key] == b[key], (key, a[key], b[key])
for key in ["architecture", "expert_placement", "context", "warm_decode", "warm_prompt_tokens", "chat", "engram_disabled"]:
    assert a.get(key) == b.get(key), (key, a.get(key), b.get(key))
shape = (a["logit_rows"], a["vocab"])
x = np.fromfile(base_dir / "logits.f32", dtype="<f4").reshape(shape).astype(np.float64)
y = np.fromfile(candidate_dir / "logits.f32", dtype="<f4").reshape(shape).astype(np.float64)
assert np.isfinite(x).all() and np.isfinite(y).all()
delta = np.abs(x - y)
logp = x - x.max(axis=1, keepdims=True)
logp -= np.log(np.exp(logp).sum(axis=1, keepdims=True))
logq = y - y.max(axis=1, keepdims=True)
logq -= np.log(np.exp(logq).sum(axis=1, keepdims=True))
kl = (np.exp(logp) * (logp - logq)).sum(axis=1)
def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()

hash_a = sha256_file(base_dir / "logits.f32")
hash_b = sha256_file(candidate_dir / "logits.f32")
result = {
    "date": datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat(),
    "generic": a, "cubecl": b,
    "prompt_ids": trace_a["prompt"], "decode_input_ids": trace_a["inputs"],
    "speedup": a["median_ms_per_token"] / b["median_ms_per_token"],
    "output_comparison": {
        "bit_identical": hash_a == hash_b, "generic_logits_sha256": hash_a, "cubecl_logits_sha256": hash_b,
        "logits_compared": x.size, "max_abs": float(delta.max()),
        "rms_abs": float(np.sqrt(np.mean((x-y)**2))),
        "absolute_difference_p50_p95_p99": np.quantile(delta, [.5,.95,.99]).tolist(),
        "max_abs_by_row": delta.max(axis=1).tolist(),
        "top1_matching_rows": int((x.argmax(axis=1) == y.argmax(axis=1)).sum()),
        "rows": shape[0], "top1_ids_match": a["top1"] == b["top1"],
        "greedy_text_match": a["greedy_text"] == b["greedy_text"],
        "max_softmax_kl_generic_to_cubecl": float(kl.max()),
        "within_absolute_1e_5": bool((delta <= 1e-5).all()),
    },
    "memory_comparison": {
        "loaded_device_bytes_identical": a["loaded_reserved_live"] == b["loaded_reserved_live"],
        "warm_device_bytes_identical": a["warm_reserved_live"] == b["warm_reserved_live"],
        "sampled_peak_device_bytes_identical": a["peak_reserved_live"] == b["peak_reserved_live"],
        "after_session_device_bytes_identical": a["after_session_reserved_live"] == b["after_session_reserved_live"],
        "generic_after_sessions_stable": len(set(map(tuple, a["after_session_reserved_live"]))) == 1,
        "cubecl_after_sessions_stable": len(set(map(tuple, b["after_session_reserved_live"]))) == 1,
    },
}
if "prefill_ms" in a and "prefill_ms" in b:
    total_a = np.asarray(a["prefill_ms"]) + np.asarray(a["ms_per_token"]) * a["decode_tokens"]
    total_b = np.asarray(b["prefill_ms"]) + np.asarray(b["ms_per_token"]) * b["decode_tokens"]
    result["prefill_comparison"] = {"generic_ms": a["prefill_ms"], "cubecl_ms": b["prefill_ms"],
        "speedup": float(np.median(a["prefill_ms"]) / np.median(b["prefill_ms"]))}
    result["end_to_end_comparison"] = {"generic_session_ms": total_a.tolist(), "cubecl_session_ms": total_b.tolist(),
        "speedup": float(np.median(total_a) / np.median(total_b))}
if "after_session_host_memory_kib" in a and "after_session_host_memory_kib" in b:
    anon_a = [v["RssAnon"] + v.get("VmSwap", 0) for v in a["after_session_host_memory_kib"]]
    anon_b = [v["RssAnon"] + v.get("VmSwap", 0) for v in b["after_session_host_memory_kib"]]
    result["memory_comparison"].update({
        "anonymous_accounting": "resident plus swapped anonymous pages" if "VmSwap" in a["after_session_host_memory_kib"][0] else "resident anonymous pages (legacy)",
        "generic_anonymous_growth_kib": max(anon_a) - min(anon_a),
        "cubecl_anonymous_growth_kib": max(anon_b) - min(anon_b),
        "anonymous_memory_bounded_within_1_mib": max(anon_a) - min(anon_a) <= 1024 and max(anon_b) - min(anon_b) <= 1024,
        "candidate_anonymous_peak_within_1_mib_of_generic": max(anon_b) <= max(anon_a) + 1024,
    })
memory = result["memory_comparison"]
def repeated_sessions_measured(summary):
    return summary.get("passes", 0) >= 2 and all(
        len(summary.get(key, [])) == summary["passes"]
        for key in ["ms_per_token", "prefill_ms", "after_session_reserved_live",
                    "after_session_host_memory_kib"])

checks = {
    "same_executable": bool(a.get("executable_sha256")) and a.get("executable_sha256") == b.get("executable_sha256"),
    "repeated_sessions_measured": repeated_sessions_measured(a) and repeated_sessions_measured(b),
    "parity": result["output_comparison"]["within_absolute_1e_5"] and result["output_comparison"]["top1_ids_match"],
    "repeated_session_parity": a["max_repeat_logit_difference"] <= 1e-5 and b["max_repeat_logit_difference"] <= 1e-5,
    "faster_decode": result["speedup"] > 1.0,
    "all_decode_samples_faster": max(b["ms_per_token"]) < min(a["ms_per_token"]),
    "device_memory": all(memory[k] for k in ["loaded_device_bytes_identical", "warm_device_bytes_identical", "sampled_peak_device_bytes_identical", "after_session_device_bytes_identical", "generic_after_sessions_stable", "cubecl_after_sessions_stable"]),
    "host_memory": memory.get("anonymous_memory_bounded_within_1_mib", False) and memory.get("candidate_anonymous_peak_within_1_mib_of_generic", False),
    "no_extra_fallbacks": all(len(a[key]) == len(b[key]) and all(y <= x for x, y in zip(a[key], b[key])) for key in ["decode_cpu_fallbacks", "prefill_cpu_fallbacks"]),
}
if "end_to_end_comparison" in result:
    checks["faster_end_to_end"] = result["end_to_end_comparison"]["speedup"] > 1.0
    checks["all_session_samples_faster"] = bool(max(total_b) < min(total_a))
if "memory_samples" in a and "memory_samples" in b:
    def decode_spans(summary):
        spans = []
        for p in range(1, summary["passes"] + 1):
            samples = [v["live_bytes"] for v in summary["memory_samples"] if v["phase"] == "decode" and v["pass"] == p]
            spans.append(max(samples) - min(samples))
        return spans
    result["memory_comparison"]["generic_decode_live_span_bytes_by_pass"] = decode_spans(a)
    result["memory_comparison"]["cubecl_decode_live_span_bytes_by_pass"] = decode_spans(b)
    fields = ["phase", "pass", "position", "reserved_bytes", "live_bytes"]
    identical_samples = [[v[k] for k in fields] for v in a["memory_samples"]] == [[v[k] for k in fields] for v in b["memory_samples"]]
    result["memory_comparison"]["device_memory_trajectory_identical"] = identical_samples
    checks["device_memory_trajectory"] = identical_samples
if args.allow_reduced_memory:
    def no_larger(a, b):
        return len(a) == len(b) and all(y <= x for x, y in zip(a, b))
    bounds = {
        key: no_larger(a[key], b[key]) if key in ["loaded_reserved_live", "warm_reserved_live", "peak_reserved_live"]
        else len(a[key]) == len(b[key]) and all(no_larger(x, y) for x, y in zip(a[key], b[key]))
        for key in ["loaded_reserved_live", "warm_reserved_live", "peak_reserved_live", "after_session_reserved_live"]
    }
    memory["candidate_allocations_no_larger"] = bounds
    checks["device_memory"] = all(bounds.values()) and memory["generic_after_sessions_stable"] and memory["cubecl_after_sessions_stable"]
    if "memory_samples" in a and "memory_samples" in b:
        xs, ys = a["memory_samples"], b["memory_samples"]
        bounded = len(xs) == len(ys) and all(
            [x[k] for k in ["phase", "pass", "position"]] == [y[k] for k in ["phase", "pass", "position"]]
            and y["reserved_bytes"] <= x["reserved_bytes"] and y["live_bytes"] <= x["live_bytes"]
            for x, y in zip(xs, ys))
        memory["candidate_device_trajectory_no_larger"] = bounded
        checks["device_memory_trajectory"] = bounded
    def driver_endpoints(summary):
        return [sum(v for k, v in row.items() if k.startswith("drm-resident-vram"))
                for row in summary.get("after_session_driver_memory_bytes", [])]
    da, db = driver_endpoints(a), driver_endpoints(b)
    known = len(da) >= 2 and len(db) >= 2 and min(da + db) > 0
    checks["driver_memory"] = bool(known and max(da)-da[0] <= 1024**2
        and max(db)-db[0] <= 1024**2 and max(db) <= max(da) + 1024**2)
    memory["driver_vram_endpoints"] = {"generic": da, "cubecl": db}
result["acceptance"] = {**checks, "passed": all(checks.values())}
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: result[k] for k in ["speedup", "output_comparison", "memory_comparison", "acceptance"]}, indent=2))

if args.require_pass and not result["acceptance"]["passed"]:
    sys.exit("benchmark acceptance failed: " + ", ".join(k for k, v in checks.items() if not v))
