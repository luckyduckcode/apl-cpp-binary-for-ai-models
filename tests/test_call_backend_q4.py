import json
import subprocess
import os
from pathlib import Path
import numpy as np
from quantization import quantize_per_row


def test_call_backend_q4(tmp_path):
    # Create a small weight matrix and quantize to q4
    W = np.array([[0.5, -0.2, 1.0, -0.3],
                  [1.5, 0.0, -0.5, 0.2]], dtype=np.float32)
    q, scales, zps = quantize_per_row(W, bits=4)
    manifest_dir = Path(tmp_path)
    q_path = manifest_dir / 'myw_q4.npy'
    scales_path = manifest_dir / 'myw_q4_scales.npy'
    zps_path = manifest_dir / 'myw_q4_zero_point.npy'
    np.save(q_path, q)
    np.save(scales_path, scales)
    np.save(zps_path, zps)

    manifest = {
        'format_version': 2,
        'model': {'primary_family': 'llama'},
        'weights': {
            'myw': {
                'q': str(q_path.name),
                'q_scales': str(scales_path.name),
                'q_zero_point': str(zps_path.name),
                'shape': [2, 4],
                'bit_width': 4
            }
        }
    }
    manifest_path = manifest_dir / 'manifest.json'
    with manifest_path.open('w', encoding='utf-8') as fh:
        json.dump(manifest, fh)

    vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    input_path = manifest_dir / 'input.txt'
    np.savetxt(input_path, vec)

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(repo_root)
    cmd = ['python', 'cpp/call_backend.py', '--manifest', str(manifest_path), '--weight', 'myw', '--input', str(input_path), '--out-json']
    out = subprocess.check_output(cmd, cwd=str(repo_root), env=env)
    got = json.loads(out.decode('utf-8').strip().splitlines()[-1])

    q_int = q.astype(np.int32)
    zps = zps.astype(np.int32)
    deq = (q_int - zps.reshape((-1, 1))).astype(np.float32) * scales.reshape((-1, 1))
    expected = deq.dot(vec).tolist()
    assert np.allclose(got, expected, atol=1e-5)
