"""Regression checks for automatic-preference certification (no model/GPU needed)."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


class CertificationTests(unittest.TestCase):
    def compare(self, base_changes=None, candidate_changes=None, comparator='compare_backends.py'):
        reference = {
            'model': 'fixture', 'model_bytes': 256, 'context': 2,
            'executable_sha256': 'a' * 64,
            'prompt_tokens': 1, 'decode_tokens': 1, 'prefill_chunk': 1,
            'warm_decode': 1, 'warm_prompt_tokens': 1, 'engram_disabled': False,
            'logit_rows': 2, 'vocab': 2, 'backend': 'vulkan',
            'gpu': 'fixture GPU', 'passes': 2, 'greedy_text': 'fixture',
            'loaded_reserved_live': [4096, 1024],
            'warm_reserved_live': [4096, 1024],
            'peak_reserved_live': [4096, 1024],
            'auto_placement': False, 'allocator_stats_available': True,
            'after_session_reserved_live': [[4096, 1024], [4096, 1024]],
            'after_session_host_memory_kib': [{'RssAnon': 128}, {'RssAnon': 128}],
            'after_session_driver_memory_bytes': [{'drm-resident-vram0': 4096}] * 2,
            'memory_samples': [
                {'phase': 'decode', 'position': 2, 'pass': p,
                 'reserved_bytes': 4096, 'live_bytes': 1024,
                 'driver_memory_bytes': {'drm-resident-vram0': 4096}} for p in [1, 2]],
            'max_repeat_logit_difference': 0.0, 'top1': [0, 1],
            'decode_cpu_fallbacks': [0, 0], 'prefill_cpu_fallbacks': [0, 0],
            'prefill_ms': [10.0, 10.0], 'ms_per_token': [10.0, 10.0],
            'median_ms_per_token': 10.0,
        }
        candidate = copy.deepcopy(reference)
        candidate.update(prefill_ms=[5.0, 5.0], ms_per_token=[5.0, 5.0],
                         median_ms_per_token=5.0)
        reference.update(base_changes or {})
        candidate.update(candidate_changes or {})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, summary in [('base', reference), ('candidate', candidate)]:
                folder = root / name
                folder.mkdir()
                (folder / 'summary.json').write_text(json.dumps(summary))
                (folder / 'trace.json').write_text(json.dumps({'prompt': [1], 'inputs': [2]}))
                (folder / 'logits.f32').write_bytes(struct.pack('<4f', 1, 0, 0, 1))
            paths = [root / 'base', root / 'comparison.json', root / 'candidate']
            if comparator == 'compare_real.py':
                paths = [root / 'base', root / 'candidate', root / 'comparison.json']
            subprocess.run([sys.executable, str(Path(__file__).with_name(comparator)),
                            *map(str, paths)], check=True, capture_output=True, text=True)
            result = json.loads((root / 'comparison.json').read_text())
            return result['acceptance'] if comparator == 'compare_real.py' else result['runs'][0]

    def test_stable_matching_runs_can_qualify(self):
        self.assertTrue(self.compare()['eligible_for_auto_preference'])

    def test_different_or_unrecorded_builds_cannot_certify_kernels(self):
        for fingerprint in [None, '', 'b' * 64]:
            with self.subTest(fingerprint=fingerprint):
                result = self.compare(candidate_changes={'executable_sha256': fingerprint})
                self.assertFalse(result['gates']['same_executable'])
                self.assertFalse(result['eligible_for_auto'])
        self.assertFalse(self.compare(
            base_changes={'executable_sha256': None},
            candidate_changes={'executable_sha256': None})['eligible_for_auto'])

    def test_kernel_comparator_requires_same_build_and_repeated_sessions(self):
        self.assertTrue(self.compare(comparator='compare_real.py')['passed'])
        for changes in [{'executable_sha256': 'b' * 64},
                        {'executable_sha256': None}, {'passes': 1},
                        {'ms_per_token': [5.0]}]:
            with self.subTest(changes=changes):
                self.assertFalse(self.compare(candidate_changes=changes,
                                              comparator='compare_real.py')['passed'])

    def test_unstable_reference_cannot_certify_a_faster_candidate(self):
        cases = [
            {'max_repeat_logit_difference': 2e-5},
            {'after_session_host_memory_kib': [
                {'RssAnon': 128, 'VmSwap': 0}, {'RssAnon': 128, 'VmSwap': 2048}]},
            {'after_session_host_memory_kib': [{'RssAnon': 128}, {'RssAnon': 2176}]},
            {'after_session_host_memory_kib': [{'RssAnon': 2176}, {'RssAnon': 128}]},
            {'after_session_reserved_live': [[4096, 1024], [8192, 1024]]},
            {'after_session_driver_memory_bytes': [
                {'drm-resident-vram0': 4096}, {'drm-resident-vram0': 2**22}]},
        ]
        for changes in cases:
            with self.subTest(changes=changes):
                result = self.compare(base_changes=changes)
                self.assertFalse(result['eligible_for_auto'])
                self.assertFalse(result['eligible_for_auto_preference'])

    def test_single_session_is_not_repeatability_evidence(self):
        changes = {'ms_per_token': [10.0], 'prefill_ms': [10.0],
                   'after_session_host_memory_kib': [{'RssAnon': 128}],
                   'after_session_reserved_live': [[4096, 1024]],
                   'after_session_driver_memory_bytes': [{'drm-resident-vram0': 4096}]}
        for side in ['base_changes', 'candidate_changes']:
            with self.subTest(side=side):
                self.assertFalse(self.compare(**{side: changes})['eligible_for_auto'])

    def test_existing_tolerance_and_memory_boundary_still_pass(self):
        changes = {'max_repeat_logit_difference': 1e-5,
                   'after_session_host_memory_kib': [{'RssAnon': 128}, {'RssAnon': 1152}]}
        self.assertTrue(self.compare(base_changes=changes,
                                     candidate_changes=changes)['eligible_for_auto_preference'])

    def test_candidate_private_memory_uses_full_span(self):
        self.assertFalse(self.compare(candidate_changes={
            'after_session_host_memory_kib': [{'RssAnon': 2176}, {'RssAnon': 128}],
        })['eligible_for_auto'])

    def test_cleanup_is_charged_to_session_speed(self):
        common = {'host_heap_cleanup_method': 'fixture-release',
                  'session_cleanup_ms': [0., 0.]}
        result = self.compare(base_changes=common, candidate_changes={
            **common, 'session_cleanup_ms': [20., 20.]})
        self.assertTrue(result['performance_gates']['all_decode_samples_faster'])
        self.assertFalse(result['performance_gates']['all_session_samples_faster'])
        self.assertFalse(result['eligible_for_auto_preference'])

    def test_released_pages_cannot_hide_live_heap_growth(self):
        common = {'host_heap_cleanup_method': 'fixture-release',
                  'session_cleanup_ms': [0., 0.],
                  'after_session_heap_stats': [{'live_bytes': 100_000}] * 2}
        result = self.compare(base_changes=common, candidate_changes={
            **common, 'after_session_heap_stats': [
                {'live_bytes': 100_000}, {'live_bytes': 4_000_000}]})
        self.assertTrue(result['gates']['private_memory_growth_le_1MiB'])
        self.assertFalse(result['gates']['live_heap_stable_le_1MiB'])
        self.assertFalse(result['eligible_for_auto'])

    def test_rejects_mixed_cleanup_protocols(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.compare(base_changes={'host_heap_cleanup_method': 'fixture-release'},
                         candidate_changes={'host_heap_cleanup_method': 'different'})

    def test_incomplete_live_heap_samples_cannot_certify(self):
        common = {'host_heap_cleanup_method': 'fixture-release',
                  'session_cleanup_ms': [0., 0.],
                  'after_session_heap_stats': [{'live_bytes': 100_000}] * 2}
        result = self.compare(base_changes=common, candidate_changes={
            **common, 'after_session_heap_stats': [{'live_bytes': 100_000}]})
        self.assertIsNone(result['gates']['live_heap_stable_le_1MiB'])
        self.assertFalse(result['eligible_for_auto'])


if __name__ == '__main__':
    unittest.main()
