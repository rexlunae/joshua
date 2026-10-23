"""Locate the first activation divergence, including differences below 1e-5.

Use captures from the same instrumentation and inputs. This is diagnostic;
it does not certify performance or full-model parity.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('reference', type=Path)
p.add_argument('candidate', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
files = sorted(a.reference.glob('*.json'))
assert files and [v.name for v in files] == sorted(v.name for v in a.candidate.glob('*.json')), 'different checkpoints'
results = []
for path in files:
    metadata = json.loads(path.read_text())
    assert metadata == json.loads((a.candidate / path.name).read_text()), 'different activation shapes/labels'
    x = np.fromfile(path.with_suffix('.f32'), dtype='<f4')
    y = np.fromfile((a.candidate / path.name).with_suffix('.f32'), dtype='<f4')
    assert x.size == y.size == math.prod(metadata['shape']), 'incomplete capture'
    finite = np.isfinite(x) & np.isfinite(y)
    invalid = np.any((~finite) & (x != y))
    delta = np.abs(x[finite].astype(np.float64) - y[finite].astype(np.float64))
    maximum = float(delta.max(initial=0))
    different = int(np.count_nonzero(x.view('<u4') != y.view('<u4')))
    results.append({'checkpoint':path.stem, **metadata, 'elements':int(x.size),
                    'different_bits':different, 'max_abs_finite':maximum,
                    'nonfinite_mismatch':bool(invalid), 'absolute_1e_5':not invalid and maximum <= 1e-5})
report = {'reference':str(a.reference), 'candidate':str(a.candidate),
          'first_different_bits':next((r for r in results if r['different_bits']), None),
          'first_exceeding_1e_5':next((r for r in results if not r['absolute_1e_5']), None),
          'checkpoints':results}
a.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k:v for k,v in report.items() if k != 'checkpoints'}, indent=2))
