"""Serial B50 real-model evaluation. Build bench_cubecl_real with vulkan,opencl,sycl first.
Each API gets identical teacher-forced tokens. Fixed means device dense/CPU
DeepSeek experts/no expert cache; auto uses the Engine's actual placement.
Existing complete results are never overwritten; --retry-incomplete moves a
folder without summary.json aside and reruns it. A failed backend does not hide
others.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--tokenizer', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--binary', type=Path, default=Path('target/release/examples/bench_cubecl_real'))
p.add_argument('--prompt-file', type=Path)
inputs = p.add_mutually_exclusive_group()
inputs.add_argument('--decode-text-file', type=Path)
inputs.add_argument('--reference', type=Path, help='replay a saved prompt/decode trace for the first backend too')
p.add_argument('--prompt-tokens', type=int, default=16)
p.add_argument('--decode', type=int, default=32)
p.add_argument('--passes', type=int, default=2)
p.add_argument('--prefill-chunk', type=int, default=8)
p.add_argument('--warm-prompt-tokens', type=int, default=8)
p.add_argument('--timeout', type=int, default=3600)
p.add_argument('--policies', nargs='+', choices=['fixed', 'auto'], default=['fixed', 'auto'])
p.add_argument('--backends', nargs='+', choices=['vulkan', 'cpu', 'opencl', 'sycl', 'auto'], default=['vulkan', 'cpu', 'opencl', 'sycl'])
p.add_argument('--retry-incomplete', action='store_true',
               help='quarantine an existing folder without summary.json and rerun it')
a = p.parse_args()
if 'auto' in a.backends and 'fixed' in a.policies:
    p.error('backend auto requires --policies auto')
a.output.mkdir(parents=True, exist_ok=True)
base = a.output / 'vulkan-fixed'
runs = [(b, policy) for policy in a.policies for b in a.backends]
if ('vulkan', 'fixed') not in runs:
    runs.insert(0, ('vulkan', 'fixed'))
def ready(folder):
    return (folder / 'summary.json').exists() and (folder / 'trace.json').exists()

state = []
for backend, policy in runs:
    folder = a.output / f'{backend}-{policy}'
    entry = {'backend': backend, 'policy': policy, 'started': datetime.now(timezone.utc).isoformat()}
    if not ready(base) and folder != base and (ready(folder) or not folder.exists()):
        base = folder
        entry['reference_fallback'] = True
    should_run = False
    if (folder / 'summary.json').exists():
        entry['status'] = 'existing_result'
    elif folder.exists():
        if a.retry_incomplete:
            # Quarantine the partial output instead of overwriting it, then
            # rerun. Without the flag an incomplete folder still needs an
            # explicit decision, so a transient failure is never hidden.
            archived = folder.with_name(
                f"{folder.name}-incomplete-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
            folder.rename(archived)
            # The subprocess log is a sibling, not part of the run folder. Move
            # it too, or the rerun would truncate the failure it explains.
            log_path = folder.with_suffix('.log')
            archived_log = archived.with_suffix('.log')
            if log_path.exists():
                log_path.rename(archived_log)
            entry['status'] = 'retrying_incomplete_result'
            entry['archived_incomplete_folder'] = str(archived)
            if archived_log.exists():
                entry['archived_incomplete_log'] = str(archived_log)
            should_run = True
        else:
            entry['status'] = 'existing_incomplete_result'
    elif folder != base and not ready(base):
        entry['status'] = 'reference_unavailable'
    else:
        should_run = True
    if should_run:
        cmd = [str(a.binary.resolve()), '--model', str(a.model.resolve()), '--tokenizer', str(a.tokenizer.resolve()),
               '--backend', backend, '--kernel', 'production', '--output', str(folder.resolve()),
               '--prompt-tokens', str(a.prompt_tokens), '--prefill-chunk', str(a.prefill_chunk),
               '--decode', str(a.decode), '--passes', str(a.passes),
               '--warm-prompt-tokens', str(a.warm_prompt_tokens), '--warm-decode', '2']
        if a.prompt_file:
            cmd += ['--prompt-file', str(a.prompt_file.resolve()), '--chat']
        if policy == 'auto':
            cmd += ['--auto-placement']
        if folder != base:
            cmd += ['--reference', str((base / 'trace.json').resolve())]
        elif a.reference:
            cmd += ['--reference', str(a.reference.resolve())]
        elif a.decode_text_file:
            cmd += ['--decode-text-file', str(a.decode_text_file.resolve())]
        entry['command'] = cmd
        print('START', backend, policy, entry['started'], flush=True)
        with folder.with_suffix('.log').open('w') as log:
            try:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout)
                entry['exit_code'] = result.returncode
                entry['status'] = 'complete' if result.returncode == 0 else 'failed'
            except subprocess.TimeoutExpired:
                entry['status'] = 'timeout'
    if (folder / 'summary.json').exists():
        summary = json.loads((folder / 'summary.json').read_text())
        print('MEASURED', backend, policy, 'ms/token', round(summary['median_ms_per_token'], 3), 'prefill_ms', [round(v, 1) for v in summary['prefill_ms']], flush=True)
    entry['ended'] = datetime.now(timezone.utc).isoformat()
    state.append(entry)
    (a.output / 'evaluation-state.json').write_text(json.dumps(state, indent=2)+'\n')
    print('END', backend, policy, entry['status'], entry['ended'], flush=True)
if (base / 'summary.json').exists():
    subprocess.run([sys.executable, str(Path(__file__).with_name('compare_backends.py')),
                    str(base), str(a.output / 'comparison.json'),
                    *[str(a.output / f'{b}-{policy}') for b, policy in runs]], check=True)
