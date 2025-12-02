import json
import subprocess
import os
from pathlib import Path
import numpy as np
from quantization import quantize_per_row

def test_backend_large_q4(tmp_path):
    # Create a larger weight matrix to exercise AVX2 path (multiple of 8 and non-multiple)
    # 33 rows, 67 columns (prime numbers to test boundary conditions)
    rows = 33
    cols = 67
    np.random.seed(42)
    W = np.random.randn(rows, cols).astype(np.float32)
    
    q, scales, zps = quantize_per_row(W, bits=4)
    manifest_dir = Path(tmp_path)
    q_path = manifest_dir / 'large_q4.npy'
    scales_path = manifest_dir / 'large_q4_scales.npy'
    zps_path = manifest_dir / 'large_q4_zero_point.npy'
    np.save(q_path, q)
    np.save(scales_path, scales)
    np.save(zps_path, zps)

    manifest = {
        'format_version': 2,
        'model': {'primary_family': 'llama'},
        'weights': {
            'large_w': {
                'q': str(q_path.name),
                'q_scales': str(scales_path.name),
                'q_zero_point': str(zps_path.name),
                'shape': [rows, cols],
                'bit_width': 4
            }
        }
    }
    manifest_path = manifest_dir / 'manifest.json'
    with manifest_path.open('w', encoding='utf-8') as fh:
        json.dump(manifest, fh)

    vec = np.random.randn(cols).astype(np.float32)
    input_path = manifest_dir / 'input.txt'
    np.savetxt(input_path, vec)

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(repo_root)
    
    cmd = ['python', 'cpp/call_backend.py', '--manifest', str(manifest_path), '--weight', 'large_w', '--input', str(input_path), '--out-json']
    out = subprocess.check_output(cmd, cwd=str(repo_root), env=env)
    got = json.loads(out.decode('utf-8').strip().splitlines()[-1])

    q_int = q.astype(np.int32)
    zps_a = zps.astype(np.int32)
    # Dequantize: (q - zp) * scale
    deq = (q_int - zps_a.reshape((-1, 1))).astype(np.float32) * scales.reshape((-1, 1))
    expected = deq.dot(vec).tolist()
    
    assert np.allclose(got, expected, atol=1e-4), f"Max diff: {np.max(np.abs(np.array(got) - np.array(expected)))}"
