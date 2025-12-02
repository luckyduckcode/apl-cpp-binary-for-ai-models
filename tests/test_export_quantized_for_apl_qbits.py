import json
import subprocess
import os
from pathlib import Path
import numpy as np


def test_export_npz_with_q2(tmp_path):
    # Create simple NPZ with q2 quantized weight
    W = np.random.randn(6, 10).astype(np.float32)
    # create a toy integer quantization per-row: use quantization.quantize_per_row
    from quantization import quantize_per_row
    Q, scales, zps = quantize_per_row(W, 2)
    npz_path = tmp_path / 'toy_q2.npz'
    np.savez(npz_path, **{
        'embedding.weight_q2': Q,
        'embedding.weight_shape': np.array(W.shape),
        'embedding.weight_q2_scales': scales,
        'embedding.weight_q2_zero_point': zps
    })

    out_manifest = tmp_path / 'toy_manifest.json'
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(repo_root)
    subprocess.check_call(['python', 'export_quantized_for_apl.py', '--npz', str(npz_path), '--out_manifest', str(out_manifest)], cwd=str(repo_root), env=env)
    assert out_manifest.exists()
    m = json.loads(out_manifest.read_text())
    assert 'weights' in m
    # The exporter should produce one entry for 'embedding.weight'
    embedding = m['weights'].get('embedding.weight')
    assert embedding is not None
    assert embedding.get('bit_width') == 2
    qinfo = embedding.get('quantization')
    assert qinfo.get('type') == 'perrow_int'
    assert qinfo.get('bits') == 2
