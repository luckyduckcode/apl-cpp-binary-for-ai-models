import json
import os
import tempfile
import numpy as np
import subprocess
from pathlib import Path

from quantization import binarize_weights


def test_dequantize_and_manifest_update(tmp_path: Path):
    # Build fake packed weight
    W = np.random.randn(6, 10).astype(np.float32)
    packed, scales, meta = binarize_weights(W, per_channel_axis=0)
    tmpdir = tmp_path
    packed_file = tmpdir / 'embedding.weight_1bit.bin'
    packed.tofile(str(packed_file))
    scales_file = tmpdir / 'embedding.weight_scales.npy'
    np.save(str(scales_file), scales)

    manifest = {
        'format_version': 2,
        'model': {'primary_family': 'llama'},
        'architecture': {},
        'weights': {
            'embedding.weight': {
                'packed': os.path.basename(str(packed_file)),
                'shape': [6, 10],
                'scales': os.path.basename(str(scales_file)),
                'scale_axis': {'index': 0, 'name': 'out_features'},
            }
        }
    }
    manifest_path = tmpdir / 'test_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Run dequantize (updates manifest)
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(repo_root)
    runner = subprocess.run(["python", "scripts/dequantize_manifest_weights.py", "--manifest", str(manifest_path), "--update-manifest"], cwd=str(repo_root), capture_output=True, text=True, env=env)
    assert runner.returncode == 0, f"dequantize script failed: {runner.stderr}"

    # Manifest should have been updated with fp32 path
    updated = json.loads(manifest_path.read_text(encoding='utf-8'))
    entry = updated['weights']['embedding.weight']
    assert 'fp32' in entry
    # Check fp32 file exists
    fp32_path = (manifest_path.parent / entry['fp32']).resolve()
    assert fp32_path.exists()
    arr = np.load(str(fp32_path))
    assert arr.shape == (6, 10)
