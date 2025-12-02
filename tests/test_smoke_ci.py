import json
import os
import tempfile
import numpy as np
import subprocess
from pathlib import Path

from quantization import binarize_weights


def create_test_manifest(tmpdir: Path):
    # Create a small weight matrix and write packed/scales
    W = np.random.randn(8, 8).astype(np.float32)
    packed, scales, meta = binarize_weights(W, per_channel_axis=0)
    packed_file = tmpdir / 'fc.weight_1bit.bin'
    packed.tofile(str(packed_file))
    scales_file = tmpdir / 'fc.weight_scales.npy'
    np.save(str(scales_file), scales)

    manifest = {
        'format_version': 2,
        'model': {'primary_family': 'llama', 'families': ['llama'], 'source_npz': ''},
        'architecture': {'num_layers': 1, 'hidden_size': 8, 'intermediate_size': 32},
        'quantization': {'bit_width': 1},
        'weights': {
            'fc.weight': {
                'packed': os.path.basename(str(packed_file)),
                'shape': [8, 8],
                'scales': os.path.basename(str(scales_file)),
                'scale_axis': {'index': 0, 'name': 'out_features'},
                'bit_width': 1,
            }
        }
    }
    manifest_path = tmpdir / 'test_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def test_manifest_to_apl_and_call_backend(tmp_path: Path):
    # Setup test workspace files
    manifest_path = create_test_manifest(tmp_path)

    # Run manifest_to_apl
    runner = subprocess.run(["python", "scripts/manifest_to_apl.py", "--manifest", str(manifest_path), "--out_apl", "apl/generated_manifest_test.apl"], cwd=os.getcwd(), capture_output=True, text=True)
    assert runner.returncode == 0, f"manifest_to_apl failed: {runner.stderr}"
    assert Path('apl/generated_manifest_test.apl').exists()

    # Call backend (python wrapper) using generated manifest
    runner2 = subprocess.run(["python", "cpp/call_backend.py", "--manifest", str(manifest_path), "--weight", "fc.weight", "--out-json"], cwd=os.getcwd(), capture_output=True, text=True)
    assert runner2.returncode == 0, f"call_backend.py failed: {runner2.stderr}"
    # Should have JSON on stdout — parse it
    try:
        out_list = json.loads(runner2.stdout.strip())
    except Exception:
        # If JSON not found, check plain output
        assert 'Output' in runner2.stdout
