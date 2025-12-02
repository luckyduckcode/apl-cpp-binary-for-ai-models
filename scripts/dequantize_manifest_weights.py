#!/usr/bin/env python3
"""Dequantize 1-bit packed weights from a manifest into FP32 numpy arrays.

Usage:
    python scripts/dequantize_manifest_weights.py --manifest path/to/manifest.json --out_dir path/to/out
    or to update manifest: python scripts/dequantize_manifest_weights.py --manifest path/to/manifest.json --update-manifest

This tool reads the manifest v2 `weights` section, and for each packed weight it:
  - loads the packed `.bin` data
  - loads the scales (.npy or .txt)
  - calls quantization.unpack_binarized()
  - saves the resulting FP32 weight to `<weightname>_fp32.npy` in manifest directory or out_dir
  - optionally updates the manifest adding `fp32` key for each weight
"""
import argparse
import json
from pathlib import Path
import numpy as np
from quantization import unpack_binarized


def load_scales(scales_path: Path):
    if scales_path.suffix == '.npy':
        return np.load(scales_path)
    else:
        return np.loadtxt(scales_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--update-manifest', action='store_true', help='Update manifest in place with `fp32` paths')
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print('Manifest not found:', manifest_path)
        return 1

    manifest_dir = manifest_path.parent
    out_dir = Path(args.out_dir) if args.out_dir else manifest_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    weights = manifest.get('weights', {})
    touched = False

    for name, entry in weights.items():
        packed = entry.get('packed')
        shape = entry.get('shape')
        scales = entry.get('scales') or entry.get('scales_txt')
        scale_axis = entry.get('scale_axis', {}).get('index', 0)
        if not packed or not shape or not scales:
            # skip fp32-only or other weights
            continue

        # Resolve paths relative to the manifest
        packed_path = (manifest_dir / packed) if not Path(packed).is_absolute() else Path(packed)
        if not packed_path.exists():
            print('Packed file not found for', name, packed_path, 'skipping')
            continue
        scales_path = (manifest_dir / scales) if not Path(scales).is_absolute() else Path(scales)
        if not scales_path.exists():
            print('Scales file not found for', name, scales_path, 'skipping')
            continue

        # Read packed rows as uint8
        packed_raw = np.fromfile(str(packed_path), dtype=np.uint8)
        out_dim, in_dim = int(shape[0]), int(shape[1])
        bytes_per_row = (in_dim + 7) // 8
        if packed_raw.size % bytes_per_row != 0:
            print('Packed size not multiple of bytes_per_row for', name)
            continue
        packed_rows = packed_raw.reshape((-1, bytes_per_row))

        scales_arr = load_scales(scales_path)
        fp32 = unpack_binarized(packed_rows, scales_arr, (out_dim, in_dim), per_channel_axis=int(scale_axis))

        # Write fp32 to out_dir with a safe filename
        safe_name = name.replace('.', '_').replace('/', '_')
        out_fp32 = out_dir / f"{safe_name}_fp32.npy"
        np.save(out_fp32, fp32)
        print('Dequantized', name, '->', out_fp32)

        if args.update_manifest:
            entry['fp32'] = str(out_fp32.relative_to(manifest_dir)).replace('\\', '/')
            touched = True

    if touched and args.update_manifest:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print('Manifest updated:', manifest_path)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
