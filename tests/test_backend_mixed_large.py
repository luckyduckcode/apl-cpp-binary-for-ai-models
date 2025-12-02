import json
import subprocess
import os
from pathlib import Path
import numpy as np
from quantization import quantize_per_row

def test_backend_mixed_large(tmp_path):
    # Create mixed precision weights (q4 and q8)
    rows = 32
    cols = 64
    np.random.seed(44)
    W1 = np.random.randn(rows, cols).astype(np.float32)
    W2 = np.random.randn(rows, cols).astype(np.float32)
    
    q1, s1, z1 = quantize_per_row(W1, bits=4)
    q2, s2, z2 = quantize_per_row(W2, bits=8)
    
    manifest_dir = Path(tmp_path)
    q1_path = manifest_dir / 'w_q4.npy'
    s1_path = manifest_dir / 'w_q4_scales.npy'
    z1_path = manifest_dir / 'w_q4_zp.npy'
    q2_path = manifest_dir / 'w_q8.npy'
    s2_path = manifest_dir / 'w_q8_scales.npy'
    z2_path = manifest_dir / 'w_q8_zp.npy'
    
    np.save(q1_path, q1)
    np.save(s1_path, s1)
    np.save(z1_path, z1)
    np.save(q2_path, q2)
    np.save(s2_path, s2)
    np.save(z2_path, z2)

    manifest = {
        'format_version': 2,
        'model': {'primary_family': 'llama'},
        'weights': {
            'w_q4': {
                'q': str(q1_path.name),
                'q_scales': str(s1_path.name),
                'q_zero_point': str(z1_path.name),
                'shape': [rows, cols],
                'bit_width': 4
            },
            'w_q8': {
                'q': str(q2_path.name),
                'q_scales': str(s2_path.name),
                'q_zero_point': str(z2_path.name),
                'shape': [rows, cols],
                'bit_width': 8
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
    
    # Test q4
    cmd1 = ['python', 'cpp/call_backend.py', '--manifest', str(manifest_path), '--weight', 'w_q4', '--input', str(input_path), '--out-json']
    out1 = subprocess.check_output(cmd1, cwd=str(repo_root), env=env)
    got1 = json.loads(out1.decode('utf-8').strip().splitlines()[-1])
    
    q_int1 = q1.astype(np.int32)
    z1_a = z1.astype(np.int32)
    deq1 = (q_int1 - z1_a.reshape((-1, 1))).astype(np.float32) * s1.reshape((-1, 1))
    expected1 = deq1.dot(vec).tolist()
    assert np.allclose(got1, expected1, atol=1e-4)

    # Test q8
    cmd2 = ['python', 'cpp/call_backend.py', '--manifest', str(manifest_path), '--weight', 'w_q8', '--input', str(input_path), '--out-json']
    out2 = subprocess.check_output(cmd2, cwd=str(repo_root), env=env)
    got2 = json.loads(out2.decode('utf-8').strip().splitlines()[-1])
    
    q_int2 = q2.astype(np.int32)
    z2_a = z2.astype(np.int32)
    deq2 = (q_int2 - z2_a.reshape((-1, 1))).astype(np.float32) * s2.reshape((-1, 1))
    expected2 = deq2.dot(vec).tolist()
    assert np.allclose(got2, expected2, atol=1e-4)
