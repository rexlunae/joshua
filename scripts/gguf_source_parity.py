"""Read-only, row-streamed GGUF/HF tensor diagnostics. See docs/gguf-source-parity.md."""
import argparse
from collections import OrderedDict
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
from gguf.quants import dequantize


HEADER_LIMIT = 64 * 1024 * 1024
SCALAR_FORMATS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
                  6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
DTYPES = {'F32': '<f4', 'F16': '<f2', 'BF16': '<u2',
          'F8_E4M3': 'u1', 'F8_E8M0': 'u1', 'I8': 'u1'}
FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6,
                -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float64)


def read_exact(handle, size):
    data = handle.read(size)
    if len(data) != size:
        raise ValueError(f'truncated file: wanted {size} bytes, got {len(data)}')
    return data


def read_json(handle, size):
    if not 0 < size <= HEADER_LIMIT:
        raise ValueError(f'JSON header/index exceeds {HEADER_LIMIT} bytes or is empty')
    return json.loads(read_exact(handle, size))


def e8m0(raw):
    return np.where(raw == 255, np.nan, np.exp2(raw.astype(np.float64) - 127))


def e4m3fn(raw):
    sign = np.where(raw & 128, -1., 1.)
    exponent = (raw >> 3) & 15
    mantissa = raw & 7
    values = sign * np.where(exponent == 0, mantissa / 512.,
                             (1 + mantissa / 8.) * np.exp2(exponent.astype(int) - 7))
    return np.where((exponent == 15) & (mantissa == 7), np.nan, values)


class GGUF:
    def __init__(self, path):
        self.path = Path(path)
        self.tensors = {}
        with self.path.open('rb') as handle:
            def unpack(fmt):
                return struct.unpack('<' + fmt, read_exact(handle, struct.calcsize('<' + fmt)))

            def guard(size=0):
                if handle.tell() + size > min(HEADER_LIMIT, self.path.stat().st_size):
                    raise ValueError('GGUF header truncated or exceeds header limit')

            def string():
                size, = unpack('Q')
                guard(size)
                return read_exact(handle, size).decode('utf-8')

            def skip(kind):
                if kind == 8:
                    string()
                elif kind == 9:
                    element, count = unpack('IQ')
                    if element == 8:
                        guard(count * 8)
                        for _ in range(count):
                            string()
                    elif element in SCALAR_FORMATS:
                        size = count * struct.calcsize('<' + SCALAR_FORMATS[element])
                        guard(size)
                        handle.seek(size, 1)
                    else:
                        raise ValueError(f'unsupported GGUF array type {element}')
                elif kind in SCALAR_FORMATS:
                    size = struct.calcsize('<' + SCALAR_FORMATS[kind])
                    guard(size)
                    handle.seek(size, 1)
                else:
                    raise ValueError(f'unsupported GGUF metadata type {kind}')

            if read_exact(handle, 4) != b'GGUF':
                raise ValueError('not a little-endian GGUF')
            version, = unpack('I')
            if version not in (2, 3):
                raise ValueError(f'unsupported GGUF version {version}')
            tensor_count, metadata_count = unpack('QQ')
            guard(tensor_count * 32 + metadata_count * 12)
            alignment = 32
            for _ in range(metadata_count):
                name = string()
                kind, = unpack('I')
                if name == 'general.alignment':
                    if kind != 4:
                        raise ValueError('general.alignment must be UINT32')
                    alignment, = unpack('I')
                else:
                    skip(kind)
            if alignment <= 0 or alignment & (alignment - 1):
                raise ValueError('GGUF alignment must be a positive power of two')
            for _ in range(tensor_count):
                name = string()
                rank, = unpack('I')
                if rank not in (1, 2, 3, 4):
                    raise ValueError(f'invalid GGUF rank {rank}')
                dims = unpack('Q' * rank)
                kind, offset = unpack('IQ')
                if not all(dims) or name in self.tensors:
                    raise ValueError(f'invalid/duplicate tensor {name}')
                self.tensors[name] = (dims, kind, offset)
            guard()
            self.data_start = (handle.tell() + alignment - 1) // alignment * alignment

    def layout(self, name, expert):
        dims, kind, offset = self.tensors[name]
        if len(dims) > 3 or (len(dims) == 3) != (expert is not None):
            raise ValueError('select --expert for a 3-D GGUF tensor; only vectors/matrices/slabs supported')
        if expert is not None and not 0 <= expert < dims[2]:
            raise ValueError(f'expert {expert} out of range')
        quant = GGMLQuantizationType(kind)
        block, size = GGML_QUANT_SIZES[quant]
        if dims[0] % block:
            raise ValueError('GGUF row width is not divisible by quantization block size')
        rows = dims[1] if len(dims) >= 2 else 1
        row_bytes = dims[0] // block * size
        base = self.data_start + offset + (expert or 0) * rows * row_bytes
        return rows, dims[0], quant, row_bytes, base

    def rows(self, name, start, count, expert=None):
        rows, cols, quant, row_bytes, base = self.layout(name, expert)
        if not 0 <= start < start + count <= rows:
            raise ValueError('GGUF row selection out of range')
        with self.path.open('rb') as handle:
            handle.seek(base + start * row_bytes)
            raw = np.frombuffer(read_exact(handle, count * row_bytes), dtype=np.uint8)
        raw = raw.reshape(count, row_bytes)
        if quant == GGMLQuantizationType.F32:
            return raw.view('<f4').reshape(count, cols).astype(np.float64)
        if quant == GGMLQuantizationType.F16:
            return raw.view('<f2').reshape(count, cols).astype(np.float64)
        return dequantize(raw, quant).reshape(count, cols).astype(np.float64)


class Source:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        index_path = self.directory / 'model.safetensors.index.json'
        with index_path.open('rb') as handle:
            self.index = read_json(handle, index_path.stat().st_size)['weight_map']
        self.headers = OrderedDict()

    def meta(self, name):
        path = (self.directory / self.index[name]).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError('source shard must stay within source directory')
        if path not in self.headers:
            with path.open('rb') as handle:
                size, = struct.unpack('<Q', read_exact(handle, 8))
                header = read_json(handle, size)
            self.headers[path] = (8 + size, header)
            if len(self.headers) > 2:
                self.headers.popitem(last=False)
        self.headers.move_to_end(path)
        base, header = self.headers[path]
        meta = header[name]
        shape = meta['shape']
        dtype = meta['dtype']
        if len(shape) not in (1, 2) or not all(isinstance(dim, int) and dim > 0 for dim in shape):
            raise ValueError('source must be a nonempty vector or matrix')
        if dtype not in DTYPES:
            raise ValueError(f'unsupported source dtype {dtype}')
        begin, end = meta['data_offsets']
        if begin < 0 or end - begin != math.prod(shape) * np.dtype(DTYPES[dtype]).itemsize:
            raise ValueError(f'invalid safetensors offsets for {name}')
        if base + end > path.stat().st_size:
            raise ValueError(f'truncated source tensor {name}')
        return path, base + begin, dtype, shape

    def raw_rows(self, name, start, count):
        path, offset, dtype, shape = self.meta(name)
        rows, cols = shape if len(shape) == 2 else (1, shape[0])
        if not 0 <= start < start + count <= rows:
            raise ValueError('source row selection out of range')
        itemsize = np.dtype(DTYPES[dtype]).itemsize
        with path.open('rb') as handle:
            handle.seek(offset + start * cols * itemsize)
            raw = read_exact(handle, count * cols * itemsize)
        return np.frombuffer(raw, dtype=DTYPES[dtype]).reshape(count, cols), dtype

    def rows(self, name, start, count, fp4=False, scale_name=None,
             fp8_block_rows=32, fp8_block_cols=32):
        raw, dtype = self.raw_rows(name, start, count)
        if dtype in ('F32', 'F16'):
            return raw.astype(np.float64)
        if dtype == 'BF16':
            return (raw.astype('<u4') << 16).view('<f4').astype(np.float64)
        if dtype == 'F8_E4M3':
            values = e4m3fn(raw)
            block_rows, block_cols = fp8_block_rows, fp8_block_cols
        elif dtype == 'I8' and fp4:
            pairs = np.stack((raw & 15, raw >> 4), axis=-1)
            values = FP4[pairs.reshape(count, -1)]
            block_rows, block_cols = 1, 32
        else:
            raise ValueError(f'{dtype} is not a weight encoding (packed I8 requires --fp4)')
        if scale_name is None:
            if not name.endswith('weight'):
                raise ValueError('quantized source needs --scale or a name ending in weight')
            scale_name = name[:-6] + 'scale'
        shape = self.meta(name)[3]
        total_rows = shape[0] if len(shape) == 2 else 1
        expected = ((total_rows + block_rows - 1) // block_rows,
                    (values.shape[1] + block_cols - 1) // block_cols)
        if tuple(self.meta(scale_name)[3]) != expected:
            raise ValueError(f'scale shape must be {expected} for blocks {block_rows}x{block_cols}')
        first = start // block_rows
        last = (start + count - 1) // block_rows + 1
        scales, scale_dtype = self.raw_rows(scale_name, first, last - first)
        if scale_dtype != 'F8_E8M0':
            raise ValueError('quantized source scales must be F8_E8M0')
        scales = e8m0(scales)
        row_indices = np.arange(start, start + count) // block_rows - first
        col_indices = np.arange(values.shape[1]) // block_cols
        return values * scales[row_indices[:, None], col_indices[None, :]]


class Metrics:
    def __init__(self):
        self.count = self.nonfinite = 0
        self.error2 = self.reference2 = self.actual2 = self.dot = 0.
        self.max_abs = 0.
        self.actual_origin = self.reference_origin = 0.
        self.actual_mean = self.reference_mean = 0.
        self.actual_variance = self.reference_variance = self.covariance = 0.

    def add(self, actual, reference):
        finite = np.isfinite(actual) & np.isfinite(reference)
        self.nonfinite += int(finite.size - np.count_nonzero(finite))
        actual = actual[finite].astype(np.float64)
        reference = reference[finite].astype(np.float64)
        if not actual.size:
            return
        if not self.count:
            self.actual_origin, self.reference_origin = float(actual[0]), float(reference[0])
        actual_shifted = actual - self.actual_origin
        reference_shifted = reference - self.reference_origin
        actual_mean = float(np.mean(actual_shifted))
        reference_mean = float(np.mean(reference_shifted))
        actual_centered = actual_shifted - actual_mean
        reference_centered = reference_shifted - reference_mean
        actual_delta = actual_mean - self.actual_mean
        reference_delta = reference_mean - self.reference_mean
        total = self.count + actual.size
        merge_weight = self.count * actual.size / total
        self.actual_variance += float(np.sum(actual_centered ** 2)) + actual_delta ** 2 * merge_weight
        self.reference_variance += float(np.sum(reference_centered ** 2)) + reference_delta ** 2 * merge_weight
        self.covariance += float(np.sum(actual_centered * reference_centered)) + actual_delta * reference_delta * merge_weight
        self.actual_mean += actual_delta * actual.size / total
        self.reference_mean += reference_delta * actual.size / total
        self.count = total
        difference = actual - reference
        self.error2 += float(np.sum(difference * difference))
        self.reference2 += float(np.sum(reference * reference))
        self.actual2 += float(np.sum(actual * actual))
        self.dot += float(np.sum(actual * reference))
        self.max_abs = max(self.max_abs, float(np.max(np.abs(difference))))

    def result(self):
        def ratio(numerator, denominator):
            return numerator / denominator if denominator > 0 else None
        return {'finite_pairs': self.count, 'nonfinite_pairs': self.nonfinite,
                'rmse': math.sqrt(self.error2 / self.count) if self.count else None,
                'max_abs': self.max_abs if self.count else None,
                'relative_l2': ratio(math.sqrt(self.error2), math.sqrt(self.reference2)),
                'cosine': ratio(self.dot, math.sqrt(self.actual2 * self.reference2)),
                'pearson': ratio(self.covariance, math.sqrt(self.actual_variance * self.reference_variance))}


def compare(gguf, source, gguf_name, hf_name, args):
    rows, cols, quant, _, _ = gguf.layout(gguf_name, args.expert)
    _, _, dtype, shape = source.meta(hf_name)
    if dtype == 'I8' and not args.fp4:
        raise ValueError('source I8 requires explicit --fp4; arbitrary integer tensors are not FP4')
    source_shape = tuple(shape) if len(shape) == 2 else (1, shape[0])
    if dtype == 'I8' and args.fp4:
        source_shape = (source_shape[0], source_shape[1] * 2)
    if source_shape != (rows, cols):
        raise ValueError(f'shape mismatch: GGUF {(rows, cols)}, decoded HF {source_shape}')
    if not 0 <= args.row_start < args.row_start + args.rows <= rows:
        raise ValueError(f'row selection exceeds {rows} rows')
    chunk = min(args.chunk_rows, args.max_elements // cols)
    if chunk < 1:
        raise ValueError(f'row width {cols} exceeds --max-elements {args.max_elements}')
    direct, swapped = Metrics(), Metrics()
    for start in range(args.row_start, args.row_start + args.rows, chunk):
        count = min(chunk, args.row_start + args.rows - start)
        actual = gguf.rows(gguf_name, start, count, args.expert)
        reference = source.rows(hf_name, start, count, args.fp4, args.scale,
                                args.fp8_block_rows, args.fp8_block_cols)
        direct.add(actual, reference)
        if cols % 2 == 0:
            swapped.add(actual, reference.reshape(count, -1, 2)[:, :, ::-1].reshape(count, cols))
    return {'gguf': gguf_name, 'hf': hf_name, 'expert': args.expert,
            'gguf_dtype': quant.name, 'hf_dtype': dtype, 'shape': [rows, cols],
            'row_start': args.row_start, 'rows': args.rows,
            'direct': direct.result(),
            'adjacent_swapped': swapped.result() if cols % 2 == 0 else None}


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', required=True)
    parser.add_argument('--source', required=True, help='HF directory with model.safetensors.index.json')
    parser.add_argument('--tensor', required=True, action='append', metavar='GGUF_NAME=HF_NAME')
    parser.add_argument('--expert', type=int, help='required slab index for 3-D GGUF tensors')
    parser.add_argument('--fp4', action='store_true', help='interpret source I8 as low-first E2M1 with per-32 E8M0 scales')
    parser.add_argument('--scale', help='explicit HF scale tensor name (otherwise replace weight with scale)')
    parser.add_argument('--fp8-block-rows', type=positive, default=32)
    parser.add_argument('--fp8-block-cols', type=positive, default=32)
    parser.add_argument('--row-start', type=int, default=0)
    parser.add_argument('--rows', type=positive, default=64)
    parser.add_argument('--chunk-rows', type=positive, default=16)
    parser.add_argument('--max-elements', type=positive, default=1048576)
    args = parser.parse_args(argv)
    failed = False
    try:
        gguf, source = GGUF(args.gguf), Source(args.source)
        for mapping in args.tensor:
            try:
                gguf_name, hf_name = mapping.split('=', 1)
                report = compare(gguf, source, gguf_name, hf_name, args)
                failed |= report['direct']['nonfinite_pairs'] > 0
            except (ValueError, KeyError, OSError, NotImplementedError) as error:
                report = {'tensor': mapping, 'error': str(error)}
                failed = True
            print(json.dumps(report, allow_nan=False), flush=True)
    except (ValueError, KeyError, OSError) as error:
        print(json.dumps({'error': str(error)}), file=sys.stderr)
        return 1
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
