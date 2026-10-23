"""Compare real-model runs without assuming identical allocators across APIs.
Usage: python3 compare_backends.py REFERENCE_DIR OUTPUT_JSON RUN_DIR...
Missing measurements remain unknown. Absolute logit tolerance stays 1e-5.
"""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('reference', type=Path)
p.add_argument('output', type=Path)
p.add_argument('runs', type=Path, nargs='+')
p.add_argument('--allow-prefill-chunk-difference', action='store_true')
a = p.parse_args()
base = json.loads((a.reference / 'summary.json').read_text())
trace = json.loads((a.reference / 'trace.json').read_text())
x = np.fromfile(a.reference / 'logits.f32', dtype='<f4').reshape(base['logit_rows'], base['vocab'])
assert np.isfinite(x).all()
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024*1024):
            h.update(block)
    return h.hexdigest()

base_hash = digest(a.reference / 'logits.f32')
results = []
for folder in a.runs:
    if not (folder / 'summary.json').exists():
        results.append({'run': str(folder), 'status': 'incomplete', 'eligible_for_auto': False})
        continue
    s = json.loads((folder / 'summary.json').read_text())
    for sample in s.get('memory_samples', []):
        sample['driver_memory_bytes'] = {k:v for k,v in sample.get('driver_memory_bytes', {}).items() if not k.startswith('drm-total-cycles-')}
    assert json.loads((folder / 'trace.json').read_text()) == trace, 'different teacher-forced inputs'
    for key in ['model', 'model_bytes', 'context', 'prompt_tokens', 'decode_tokens', 'prefill_chunk', 'warm_decode', 'warm_prompt_tokens', 'engram_disabled']:
        if key == 'prefill_chunk' and a.allow_prefill_chunk_difference:
            continue
        assert s[key] == base[key], (key, s[key], base[key])
    y = np.fromfile(folder / 'logits.f32', dtype='<f4').reshape(x.shape)
    assert np.isfinite(y).all()
    delta = np.abs(x.astype(np.float64) - y.astype(np.float64))
    sessions = s['after_session_host_memory_kib']
    private = [v['RssAnon'] + v.get('VmSwap', 0) for v in sessions]
    private_growth = max(private) - private[0]
    samples = s['memory_samples']
    driver = [v['driver_memory_bytes'] for v in samples]
    vram_keys = sorted({k for v in driver for k in v if k.startswith('drm-resident-vram')})
    vram = [sum(v.get(k, 0) for k in vram_keys) for v in driver]
    endpoints = [v for v in samples if v['phase'] == 'decode' and v['position'] == s['context']]
    endpoint_vram = [sum(v.get(k, 0) for k in vram_keys) for v in s.get('after_session_driver_memory_bytes', [])]
    if not endpoint_vram:
        endpoint_vram = [sum(v['driver_memory_bytes'].get(k, 0) for k in vram_keys) for v in endpoints]
    # Compare equal workload positions after warmup. Cache fills are reported
    # explicitly and cannot silently qualify as stable memory.
    driver_stable = max(endpoint_vram) - endpoint_vram[0] <= 1024**2 if vram_keys and len(endpoint_vram) >= 2 else None
    allocator_stable = len(set(map(tuple, s['after_session_reserved_live']))) == 1 if s['allocator_stats_available'] else None
    known_device_checks = [v for v in [allocator_stable, driver_stable] if v is not None]
    device_stable = True if s['backend'] == 'cpu' else all(known_device_checks) if known_device_checks else None
    totals = np.asarray(s['prefill_ms']) + np.asarray(s['ms_per_token']) * s['decode_tokens']
    base_totals = np.asarray(base['prefill_ms']) + np.asarray(base['ms_per_token']) * base['decode_tokens']
    gates = {'absolute_logits_1e_5': bool((delta <= 1e-5).all()),
             'repeat_logits_1e_5': s['max_repeat_logit_difference'] <= 1e-5,
             'same_greedy_choices': s['top1'] == base['top1'],
             'private_memory_growth_le_1MiB': private_growth <= 1024,
             'device_memory_stable': device_stable,
             'no_cpu_fallbacks': not any(s['decode_cpu_fallbacks'] + s['prefill_cpu_fallbacks'])}
    run_hash = digest(folder / 'logits.f32')
    r = {'run': str(folder), 'logits_sha256': run_hash, 'bit_identical': run_hash == base_hash, 'backend': s['backend'], 'auto_placement': s['auto_placement'],
         'summary': s, 'max_abs_logit_difference': float(delta.max()),
         'rms_logit_difference': float(np.sqrt(np.mean(delta**2))),
         'max_abs_by_row': delta.max(axis=1).tolist(),
         'matching_top1_rows': int((x.argmax(axis=1) == y.argmax(axis=1)).sum()),
         'logits_compared': x.size, 'decode_speedup_vs_reference': base['median_ms_per_token']/s['median_ms_per_token'],
         'prefill_median_ms': float(np.median(s['prefill_ms'])), 'session_median_ms': float(np.median(totals)),
         'session_speedup_vs_reference': float(np.median(base_totals)/np.median(totals)),
         'private_growth_kib': private_growth,
         'private_last_two_session_growth_kib': private[-1] - private[-2] if len(private) >= 2 else None, 'peak_driver_vram_bytes': max(vram) if vram_keys else None,
         'decode_endpoint_driver_vram_bytes': endpoint_vram if vram_keys else None,
         'driver_vram_growth_bytes': max(endpoint_vram)-endpoint_vram[0] if endpoint_vram else None,
         'gates': gates, 'eligible_for_auto': all(v is True for v in gates.values())}
    if 'after_session_io_counters' in s:
        counters = [s['warm_io_counters'], *s['after_session_io_counters']]
        r['physical_read_bytes_per_session'] = [b['read_bytes']-a['read_bytes'] for a,b in zip(counters, counters[1:])]
    r['performance_gates'] = {
        'all_decode_samples_faster': max(s['ms_per_token']) < min(base['ms_per_token']),
        'all_session_samples_faster': float(max(totals)) < float(min(base_totals)),
    }
    r['eligible_for_auto_preference'] = r['eligible_for_auto'] and all(r['performance_gates'].values())
    results.append(r)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps({'reference': str(a.reference), 'reference_logits_sha256': base_hash, 'tolerance': 1e-5, 'runs': results}, indent=2) + '\n')
for r in results:
    if 'summary' in r:
        print(r['backend'], 'auto='+str(r['auto_placement']), 'ms/token='+str(round(r['summary']['median_ms_per_token'], 3)), 'max_abs='+str(r['max_abs_logit_difference']), 'gates='+str(r['gates']))
    else:
        print(r)
