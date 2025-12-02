import json
import subprocess
import os
from pathlib import Path
import numpy as np
from quantization import quantize_per_row


def test_call_backend_integer_q(tmp_path):
    # Create simple 3x4 weight matrix and quantize to q2
    W = np.array([[1.0, -0.5, 0.1, 2.0],
                  [-1.0, 0.75, 0.6, -0.2],
                  [0.0, 0.5, -0.5, 1.25]], dtype=np.float32)
    q, scales, zps = quantize_per_row(W, bits=2)
    # Write files
    manifest_dir = Path(tmp_path)
    q_path = manifest_dir / 'myw_q2.npy'
    scales_path = manifest_dir / 'myw_q2_scales.npy'
    zps_path = manifest_dir / 'myw_q2_zero_point.npy'
    np.save(q_path, q)
    np.save(scales_path, scales)
    np.save(zps_path, zps)

    # create manifest
    manifest = {
        'format_version': 2,
        'model': {'primary_family': 'llama'},
        'weights': {
            'myw': {
                'q': str(q_path.name),
                'q_scales': str(scales_path.name),
                'q_zero_point': str(zps_path.name),
                'shape': [3, 4],
                'bit_width': 2
            }
        }
    }
    manifest_path = manifest_dir / 'manifest.json'
    with manifest_path.open('w', encoding='utf-8') as fh:
        json.dump(manifest, fh)

    # Input vector
    vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    input_path = manifest_dir / 'input.txt'
    np.savetxt(input_path, vec)

    # Call the backend CLI
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(repo_root)
    cmd = [
        'python', 'cpp/call_backend.py',
        '--manifest', str(manifest_path),
        '--weight', 'myw',
        '--input', str(input_path),
        '--out-json'
    ]
    out = subprocess.check_output(cmd, cwd=str(repo_root), env=env)
    got = json.loads(out.decode('utf-8').strip().splitlines()[-1])

    # compute expected result
    q_int = q.astype(np.int32)
    zps = zps.astype(np.int32)
    # dequantize
    deq = (q_int - zps.reshape((-1, 1))).astype(np.float32) * scales.reshape((-1, 1))
    expected = deq.dot(vec).tolist()
    # Compare
    assert np.allclose(got, expected, atol=1e-5)
