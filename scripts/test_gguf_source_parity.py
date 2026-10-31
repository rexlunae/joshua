"""Tiny independent on-disk fixtures; no model downloads or accelerator needed."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from gguf.constants import GGMLQuantizationType as Quant
from gguf.quants import quantize

import gguf_source_parity as parity


def write_source(directory, tensors, shard='part.safetensors'):
    header, payload = {}, bytearray()
    for name, (dtype, values) in tensors.items():
        data = values.tobytes()
        header[name] = {'dtype': dtype, 'shape': list(values.shape),
                        'data_offsets': [len(payload), len(payload) + len(data)]}
        payload.extend(data)
    encoded = json.dumps(header).encode()
    (directory / shard).write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)
    index_path = directory / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())['weight_map'] if index_path.exists() else {}
    index.update({name: shard for name in tensors})
    index_path.write_text(json.dumps({'weight_map': index}))


def write_gguf(path, tensors):
    def string(text):
        encoded = text.encode()
        return struct.pack('<Q', len(encoded)) + encoded
    metadata = string('general.alignment') + struct.pack('<II', 4, 64)
    metadata += string('ignored.array') + struct.pack('<IIQ', 9, 8, 2) + string('a') + string('b')
    header = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), 2) + metadata
    payload = bytearray()
    for name, dims, kind, data in tensors:
        payload.extend(b'\0' * (-len(payload) % 64))
        header += string(name) + struct.pack('<I', len(dims))
        header += struct.pack('<' + 'Q' * len(dims), *dims)
        header += struct.pack('<IQ', kind, len(payload))
        payload.extend(data)
    path.write_bytes(header + b'\0' * (-len(header) % 64) + payload)


class ParityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.gguf_path = self.directory / 'tiny.gguf'

    def args(self, **changes):
        values = dict(expert=None, fp4=False, scale=None, fp8_block_rows=32, fp8_block_cols=32,
                      row_start=0, rows=2, chunk_rows=1, max_elements=512)
        values.update(changes)
        return argparse.Namespace(**values)

    def compare(self, args=None):
        return parity.compare(parity.GGUF(self.gguf_path), parity.Source(self.directory),
                              'test', 'test.weight', args or self.args())

    def test_float_known_answers(self):
        raw = np.array([0x7e, 0x78, 0x7f, 0x01, 0x80, 0xfe, 0xff], dtype='u1')
        decoded = parity.e4m3fn(raw)
        np.testing.assert_array_equal(decoded[[0, 1, 3, 5]], [448, 256, 2 ** -9, -448])
        self.assertTrue(np.isnan(decoded[[2, 6]]).all())
        self.assertTrue(np.signbit(decoded[4]))
        scales = parity.e8m0(np.array([127, 0, 255], dtype='u1'))
        np.testing.assert_array_equal(scales[:2], [1, 2 ** -127])
        self.assertTrue(np.isnan(scales[2]))

    def test_dense_shards_vectors_and_bf16(self):
        for dtype, numpy_dtype, kind in [('F32', '<f4', Quant.F32), ('F16', '<f2', Quant.F16),
                                          ('BF16', '<u2', Quant.BF16)]:
            with self.subTest(dtype=dtype):
                values = np.arange(512, dtype='<f4').reshape(2, 256) / 8
                raw = (values.view('<u4') >> 16).astype('<u2') if dtype == 'BF16' else values.astype(numpy_dtype)
                write_source(self.directory, {'test.weight': (dtype, raw)}, f'{dtype}.safetensors')
                write_gguf(self.gguf_path, [('test', (256, 2), kind, raw.tobytes())])
                self.assertEqual(self.compare()['direct']['max_abs'], 0)
        vector = np.arange(5, dtype='<f4')
        write_source(self.directory, {'test.weight': ('F32', vector)}, 'vector.safetensors')
        write_gguf(self.gguf_path, [('test', (5,), Quant.F32, vector.tobytes())])
        report = self.compare(self.args(rows=1))
        self.assertIsNone(report['adjacent_swapped'])
        self.assertEqual(report['direct']['finite_pairs'], 5)

    def test_fp8_block_boundary_tail_and_q8(self):
        raw = np.full((35, 96), 0x38, dtype='u1')
        scales = np.array([[127, 128, 129], [130, 131, 132]], dtype='u1')
        write_source(self.directory, {'test.weight': ('F8_E4M3', raw),
                                      'test.scale': ('F8_E8M0', scales)})
        expected = np.repeat(np.repeat(2. ** (scales.astype(int) - 127), 32, 0), 32, 1)[:35]
        write_gguf(self.gguf_path, [('test', (96, 35), Quant.Q8_0,
                                    quantize(expected.astype('<f4'), Quant.Q8_0).tobytes())])
        report = self.compare(self.args(row_start=31, rows=4, chunk_rows=2))
        self.assertLess(report['direct']['relative_l2'], .001)
        self.assertGreater(report['direct']['cosine'], .999)
        np.testing.assert_array_equal(parity.Source(self.directory).rows('test.weight', 31, 4), expected[31:35])

    def test_fp8_per_row_scales(self):
        raw = np.full((3, 64), 0x38, dtype='u1')
        scales = np.array([[127, 128], [129, 130], [131, 132]], dtype='u1')
        write_source(self.directory, {'test.weight': ('F8_E4M3', raw),
                                      'test.scale': ('F8_E8M0', scales)})
        expected = np.repeat(2. ** (scales.astype(int) - 127), 32, 1).astype('<f4')
        write_gguf(self.gguf_path, [('test', (64, 3), Quant.F32, expected.tobytes())])
        report = self.compare(self.args(fp8_block_rows=1, row_start=1))
        self.assertEqual(report['direct']['max_abs'], 0)

    def test_fp4_known_answer_scales_expert_and_swapped_detection(self):
        raw = np.tile(np.array([0x21, 0x98, 0x7f, 0x35], dtype='u1'), (3, 32))
        scales = np.tile(np.array([127, 128, 126, 129, 127, 128, 126, 129], dtype='u1'), (3, 1))
        write_source(self.directory, {'test.weight': ('I8', raw), 'test.scale': ('F8_E8M0', scales)})
        source = parity.Source(self.directory)
        expected = source.rows('test.weight', 0, 3, fp4=True)
        np.testing.assert_array_equal(expected[0, :8], [.5, 1, -0., -.5, -6, 6, 3, 1.5])
        np.testing.assert_array_equal(expected[0, 32:40], 2 * expected[0, :8])
        self.assertTrue(np.signbit(expected[0, 2]))
        swapped = expected.reshape(3, -1, 2)[:, :, ::-1].reshape(3, 256)
        slabs = np.stack((expected, swapped)).astype('<f4')
        write_gguf(self.gguf_path, [('test', (256, 3, 2), Quant.Q8_0,
                                    quantize(slabs, Quant.Q8_0).tobytes())])
        with self.assertRaisesRegex(ValueError, 'explicit --fp4'):
            self.compare(self.args(expert=0))
        good = self.compare(self.args(fp4=True, expert=0))
        bad = self.compare(self.args(fp4=True, expert=1))
        self.assertGreater(good['direct']['cosine'], .999)
        self.assertLess(bad['direct']['cosine'], .2)
        self.assertGreater(bad['adjacent_swapped']['cosine'], .999)
        self.assertLess(bad['adjacent_swapped']['relative_l2'], .01)
        for expert in (None, -1, 2):
            with self.assertRaises(ValueError):
                self.compare(self.args(fp4=True, expert=expert))

    def test_q4_k_known_block(self):
        block = struct.pack('<ee', .5, 0) + bytes([1] * 4 + [0] * 4 + [1] * 4) + bytes([0x21] * 128)
        expected = np.tile(np.repeat(np.array([.5, 1], dtype='<f4'), 32), (2, 4))
        write_source(self.directory, {'test.weight': ('F32', expected)})
        write_gguf(self.gguf_path, [('test', (256, 2), Quant.Q4_K, block * 2)])
        self.assertEqual(self.compare()['direct']['max_abs'], 0)

    def test_chunk_invariance_and_read_bound(self):
        values = np.random.default_rng(175).normal(size=(9, 256)).astype('<f4')
        write_source(self.directory, {'test.weight': ('F32', values)})
        write_gguf(self.gguf_path, [('test', (256, 9), Quant.Q8_0,
                                    quantize(values, Quant.Q8_0).tobytes())])
        original = parity.read_exact
        read_sizes = []
        def tracked(handle, size):
            read_sizes.append(size)
            return original(handle, size)
        with patch.object(parity, 'read_exact', tracked):
            small = self.compare(self.args(rows=9, chunk_rows=20, max_elements=256))
        large = self.compare(self.args(rows=9, chunk_rows=9, max_elements=4096))
        self.assertLessEqual(max(read_sizes), 1024)
        for metric in ('relative_l2', 'cosine', 'pearson', 'rmse', 'max_abs'):
            self.assertAlmostEqual(small['direct'][metric], large['direct'][metric], places=12)

    def test_nonfinite_and_degenerate_metrics(self):
        stats = parity.Metrics()
        stats.add(np.array([0., np.nan, np.inf]), np.zeros(3))
        result = stats.result()
        self.assertEqual(result['nonfinite_pairs'], 2)
        self.assertEqual(result['max_abs'], 0)
        self.assertIsNone(result['cosine'])
        self.assertIsNone(result['pearson'])
        json.dumps(result, allow_nan=False)

    def test_sparse_large_table_reads_only_selected_rows(self):
        start, cols = 12345678, 32
        values = np.arange(64, dtype='<f4').reshape(2, cols)
        shape = [start + 2, cols]
        data_size = (start + 2) * cols * 4
        header = json.dumps({'test.weight': {'dtype': 'F32', 'shape': shape,
                                             'data_offsets': [0, data_size]}}).encode()
        with (self.directory / 'large.safetensors').open('wb') as handle:
            handle.write(struct.pack('<Q', len(header)) + header)
            handle.seek(start * cols * 4, 1)
            handle.write(values.tobytes())
        (self.directory / 'model.safetensors.index.json').write_text(
            json.dumps({'weight_map': {'test.weight': 'large.safetensors'}}))
        write_gguf(self.gguf_path, [('test', (cols, start + 2), Quant.F32, b'')])
        with self.gguf_path.open('r+b') as handle:
            handle.seek(parity.GGUF(self.gguf_path).data_start + start * cols * 4)
            handle.write(values.tobytes())
        with patch.object(np, 'fromfile', side_effect=AssertionError('whole-file reads forbidden')):
            report = self.compare(self.args(row_start=start, rows=2, max_elements=32))
        self.assertEqual(report['direct']['max_abs'], 0)
        self.assertEqual(report['direct']['finite_pairs'], 64)

    def test_invalid_selection_scale_and_truncation(self):
        values = np.ones((2, 32), dtype='<f4')
        write_source(self.directory, {'test.weight': ('F32', values)})
        write_gguf(self.gguf_path, [('test', (32, 2), Quant.F32, values.tobytes())])
        for changes in ({'row_start': -1}, {'rows': 3}, {'max_elements': 31}):
            with self.assertRaises(ValueError):
                self.compare(self.args(**changes))
        write_source(self.directory, {'test.weight': ('F8_E4M3', np.ones((2, 32), dtype='u1')),
                                      'test.scale': ('F8_E8M0', np.ones((2, 1), dtype='u1'))})
        with self.assertRaisesRegex(ValueError, 'scale shape'):
            self.compare()
        self.gguf_path.write_bytes(self.gguf_path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, 'truncated'):
            parity.GGUF(self.gguf_path).rows('test', 1, 1)
        (self.directory / 'part.safetensors').write_bytes(struct.pack('<Q', parity.HEADER_LIMIT + 1))
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            parity.Source(self.directory).meta('test.weight')

    def test_cli_continues_errors_and_nonfinite_exit_status(self):
        values = np.array([[1., np.nan]], dtype='<f4')
        write_source(self.directory, {'test.weight': ('F32', values)})
        write_gguf(self.gguf_path, [('test', (2, 1), Quant.F32, values.tobytes())])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = parity.main(['--gguf', str(self.gguf_path), '--source', str(self.directory),
                                  '--tensor', 'missing=test.weight', '--tensor', 'test=test.weight', '--rows', '1'])
        reports = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(status, 1)
        self.assertIn('error', reports[0])
        self.assertEqual(reports[1]['direct']['nonfinite_pairs'], 1)


if __name__ == '__main__':
    unittest.main()
