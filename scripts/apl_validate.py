#!/usr/bin/env python3
"""Validate 1-bit backend outputs against Python dequantization.

Features:
- Load from NPZ (weights packed fields) or manifest v2 JSON.
- Validate single weight or all weights.
- Skip patterns for embeddings/norms/biases that aren't expected to be validated via matmul.
- Optional per-row diff reporting to help debug mismatches.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantization import unpack_binarized


def load_manifest(manifest_path):
    """Load manifest JSON and return (Path, dict)."""
    mp = Path(manifest_path)
    with mp.open('r', encoding='utf-8') as fh:
        manifest = json.load(fh)
    return mp, manifest


def load_weight_from_npz(npz_obj, name):
    """Load packed weight from NPZ. Returns (packed, shape, scales, pc_axis) or None."""
    packed_key = f"{name}_1bit"
    shape_key = f"{name}_shape"
    scales_key = f"{name}_scales"
    if packed_key not in npz_obj:
        return None
    packed = np.array(npz_obj[packed_key])
    shape = tuple(npz_obj[shape_key]) if shape_key in npz_obj else None
    scales = np.array(npz_obj[scales_key]) if scales_key in npz_obj else None
    return packed, shape, scales, 0


def load_weight_from_manifest(manifest_path, manifest, name):
    """Load packed weight from manifest. Returns (packed, shape, scales, pc_axis) or None."""
    weights = manifest.get('weights', {})
    entry = weights.get(name) or manifest.get(name)
    if not entry:
        return None
    packed_rel = entry.get('packed')
    scales_rel = entry.get('scales') or entry.get('scales_txt')
    if not packed_rel:
        return None
    shape = tuple(entry.get('shape')) if 'shape' in entry else None
    pc_axis = entry.get('scale_axis', {}).get('index', 0)
    packed_path = (manifest_path.parent / packed_rel).as_posix() if not Path(packed_rel).is_absolute() else packed_rel
    scales_path = (manifest_path.parent / scales_rel).as_posix() if not Path(scales_rel).is_absolute() else scales_rel
    arr = np.fromfile(packed_path, dtype=np.uint8)
    if shape is not None:
        bytes_per_row = math.ceil(int(shape[1]) / 8)
        arr = arr.reshape((int(shape[0]), bytes_per_row))
    if scales_path.endswith('.npy'):
        scales = np.load(scales_path)
    else:
        scales = np.loadtxt(scales_path)
    return arr, shape, scales, int(pc_axis)


def call_backend(cb, packed, scales, v, name, manifest_path=None, manifest=None, mode=0, threads=0):
    """Call backend matmul - via manifest helper or direct file write."""
    if manifest_path:
        return cb.call_matmul_from_manifest(str(manifest_path), name, v, mode=mode, threads=threads)
    else:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            tmp_packed = f.name
            packed.tofile(f)
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            tmp_scales = f.name
            np.savetxt(f, scales)
        out, inp = packed.shape[0], packed.shape[1] * 8
        result = cb.matmul_1bit(tmp_packed, tmp_scales, out, inp, v, mode=mode, threads=threads)
        Path(tmp_packed).unlink(missing_ok=True)
        Path(tmp_scales).unlink(missing_ok=True)
        return result


def validate_one(cb, npz_obj, manifest_path, manifest, name, threads=0, skip_patterns=None, report=False):
    """Validate a single weight. Returns max diff or None if skipped."""
    skip_patterns = skip_patterns or []
    for pat in skip_patterns:
        if re.search(pat, name):
            return None

    if manifest:
        res = load_weight_from_manifest(manifest_path, manifest, name)
        if not res:
            print('Not a packed weight in manifest:', name)
            return None
        packed, shape, scales, pc_axis = res
    else:
        res = load_weight_from_npz(npz_obj, name)
        if not res:
            print('Not a packed weight in NPZ for key', name)
            return None
        packed, shape, scales, pc_axis = res

    if shape is None:
        print('Missing shape metadata for', name)
        return None

    out, inp = shape
    bytes_per_row = math.ceil(int(inp) / 8)
    if packed.ndim == 1:
        packed = packed.reshape((out, bytes_per_row))

    mat = unpack_binarized(packed, scales, shape, per_channel_axis=pc_axis)
    np.random.seed(0)
    v = np.random.randn(inp).astype(np.float32)
    ref = mat @ v
    outv = call_backend(cb, packed, scales, v, name, manifest_path, manifest, mode=0, threads=threads)
    diff = float(np.max(np.abs(ref - outv)))
    if report and diff > 1e-6:
        row_diffs = np.abs(ref - outv)
        top5_idx = np.argsort(-row_diffs)[:5]
        print(f"Top diffs for {name}:")
        for idx in top5_idx:
            print(f"  row {idx}: ref={ref[idx]:.6g} out={outv[idx]:.6g} diff={row_diffs[idx]:.6g}")
    return diff


def main():
    parser = argparse.ArgumentParser(description='Validate 1-bit backend matmul against Python dequantization.')
    parser.add_argument('--npz', type=str, default='student_quantized_1bit.npz', help='Path to NPZ with packed weights')
    parser.add_argument('--manifest', type=str, default='', help='Path to manifest v2 JSON (overrides NPZ for paths)')
    parser.add_argument('--name', type=str, default='fc.weight', help='Weight name to validate or "all" for all packed weights')
    parser.add_argument('--skip', type=str, default='embedding,.*norm,.*bias', help='Comma-separated regex patterns to skip')
    parser.add_argument('--report', action='store_true', help='Print per-row diffs for large errors')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads (0 = auto)')
    args = parser.parse_args()

    npz_obj = np.load(args.npz)
    manifest_path = None
    manifest = None
    if args.manifest:
        manifest_path, manifest = load_manifest(args.manifest)

    skip_patterns = [s.strip() for s in args.skip.split(',') if s.strip()]
    import cpp.call_backend as cb

    if args.name == 'all':
        if manifest:
            names = list(manifest.get('weights', {}).keys())
        else:
            names = [k[:-5] for k in npz_obj.files if k.endswith('_1bit')]
        for n in names:
            d = validate_one(cb, npz_obj, manifest_path, manifest, n, threads=args.threads, skip_patterns=skip_patterns, report=args.report)
            if d is not None:
                print(f"{n}: max diff = {d}")
    else:
        d = validate_one(cb, npz_obj, manifest_path, manifest, args.name, threads=args.threads, skip_patterns=skip_patterns, report=args.report)
        if d is not None:
            print(f"{args.name}: max diff = {d}")


if __name__ == '__main__':
    main()
