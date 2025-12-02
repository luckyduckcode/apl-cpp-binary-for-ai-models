import subprocess
import os
from pathlib import Path
import numpy as np
from transformers import GPT2Config, GPT2LMHeadModel


def test_easy_run_with_bits_2(tmp_path):
    hf_dir = Path(tmp_path) / 'tiny_hf'
    hf_dir.mkdir()
    cfg = GPT2Config(vocab_size=64, n_embd=32, n_layer=1, n_head=4)
    model = GPT2LMHeadModel(cfg)
    model.save_pretrained(hf_dir)

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(repo_root)
    out_dir = Path(tmp_path) / 'out'
    out_dir.mkdir()
    subprocess.check_call(["python", "easy_run.py", "--custom-model", str(hf_dir), "--output-dir", str(out_dir), "--bits", "2", "--run-demo"], cwd=str(repo_root), env=env)

    # Check that an NPZ with q2 keys was created
    files = list(tmp_path.rglob('*_quantized.npz'))
    assert len(files) > 0
    import numpy as np
    npz = np.load(str(files[0]), allow_pickle=True)
    assert any('_q2' in k for k in npz.files)
