import json
import subprocess
import os
from pathlib import Path
import numpy as np
from quantization import quantize_per_row


def test_call_backend_mixed(tmp_path):
    # Create two weight matrices for q2 and q4, then test both from the same manifest
    W1 = np.array([[1.0, -0.5, 0.1, 2.0]], dtype=np.float32)  # 1x4
    W2 = np.array([[0.4, 0.3, -0.9, 1.1]], dtype=np.float32)  # 1x4
    q1, s1, z1 = quantize_per_row(W1, bits=2)
    q2, s2, z2 = quantize_per_row(W2, bits=4)
    manifest_dir = Path(tmp_path)
    q1_path = manifest_dir / 'myw1_q2.npy'
    q1_scales = manifest_dir / 'myw1_q2_scales.npy'
    q1_zp = manifest_dir / 'myw1_q2_zero_point.npy'
    q2_path = manifest_dir / 'myw2_q4.npy'
    q2_scales = manifest_dir / 'myw2_q4_scales.npy'
    q2_zp = manifest_dir / 'myw2_q4_zero_point.npy'
    np.save(q1_path, q1)
    np.save(q1_scales, s1)
    np.save(q1_zp, z1)
    np.save(q2_path, q2)
    np.save(q2_scales, s2)
    np.save(q2_zp, z2)

    manifest = {
        'format_version': 2,
        'model': {'primary_family': 'llama'},
        'weights': {
            'myw1': {
                'q': str(q1_path.name),
                'q_scales': str(q1_scales.name),
                'q_zero_point': str(q1_zp.name),
                'shape': [1, 4],
                'bit_width': 2
            },
            'myw2': {
                'q': str(q2_path.name),
                'q_scales': str(q2_scales.name),
                'q_zero_point': str(q2_zp.name),
                'shape': [1, 4],
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

    # Call both weights and check their outputs
    cmd = ['python', 'cpp/call_backend.py', '--manifest', str(manifest_path), '--weight', 'myw1', '--input', str(input_path), '--out-json']
    out1 = subprocess.check_output(cmd, cwd=str(repo_root), env=env)
    got1 = json.loads(out1.decode('utf-8').strip().splitlines()[-1])
    q_int1 = q1.astype(np.int32)
    z1a = z1.astype(np.int32)
    deq1 = (q_int1 - z1a.reshape((-1, 1))).astype(np.float32) * s1.reshape((-1, 1))
    expected1 = deq1.dot(vec).tolist()
    assert np.allclose(got1, expected1, atol=1e-5)

    cmd2 = ['python', 'cpp/call_backend.py', '--manifest', str(manifest_path), '--weight', 'myw2', '--input', str(input_path), '--out-json']
    out2 = subprocess.check_output(cmd2, cwd=str(repo_root), env=env)
    got2 = json.loads(out2.decode('utf-8').strip().splitlines()[-1])
    q_int2 = q2.astype(np.int32)
    z2a = z2.astype(np.int32)
    deq2 = (q_int2 - z2a.reshape((-1, 1))).astype(np.float32) * s2.reshape((-1, 1))
    expected2 = deq2.dot(vec).tolist()
    assert np.allclose(got2, expected2, atol=1e-5)
