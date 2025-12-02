import tempfile
import numpy as np
import os
import sys
import torch
import subprocess
from pathlib import Path
from transformers import GPT2Config, GPT2LMHeadModel

from scripts.export_model_1bit import quantize_model_to_1bit


class DummyLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))


class DummySelfAttn(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q_proj = DummyLinear(d_model, d_model)
        self.k_proj = DummyLinear(d_model, d_model)
        self.v_proj = DummyLinear(d_model, d_model)
        self.o_proj = DummyLinear(d_model, d_model)


class DummyMLP(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = DummyLinear(d_model, d_ff)
        self.up_proj = DummyLinear(d_ff, d_model)
        self.down_proj = DummyLinear(d_model, d_ff)


class DummyLayer(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.self_attn = DummySelfAttn(d_model)
        self.mlp = DummyMLP(d_model, d_ff)
        self.input_layernorm = torch.nn.LayerNorm(d_model)
        self.post_attention_layernorm = torch.nn.LayerNorm(d_model)


class DummyModel:
    def __init__(self, num_layers=1, d_model=32, d_ff=64, vocab_size=100):
        self.model = self
        self.embed_tokens = torch.nn.Embedding(vocab_size, d_model)
        self.layers = torch.nn.ModuleList([DummyLayer(d_model, d_ff) for _ in range(num_layers)])
        self.lm_head = DummyLinear(d_model, vocab_size)


def test_quantize_dummy_model():
    model = DummyModel(num_layers=1)
    data = quantize_model_to_1bit(model, num_layers=1)

    # Check some expected keys exist
    assert 'embedding.weight_1bit' in data
    assert 'transformer.layers.0.self_attn.q_proj.weight_1bit' in data
    assert 'transformer.layers.0.mlp.gate_proj.weight_1bit' in data

    # Shapes saved too
    assert 'embedding.weight_shape' in data
    # shape is saved as an array of dims
    assert int(data['embedding.weight_shape'][0]) == int(model.embed_tokens.weight.shape[0])

    # Ensure scales arrays exist and are non-empty
    assert data['embedding.weight_scales'].ndim == 1


def test_export_cli_on_tiny_hf_model(tmp_path):
    # Build a tiny GPT2 config/model and save locally as a HF dir
    hf_dir = Path(tmp_path) / 'tiny_gpt'
    hf_dir.mkdir()
    cfg = GPT2Config(vocab_size=64, n_embd=32, n_layer=1, n_head=4)
    model = GPT2LMHeadModel(cfg)
    model.save_pretrained(hf_dir)

    out_npz = str(hf_dir / 'out_1bit.npz')
    # Run export bits==1
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(repo_root)
    subprocess.check_call([sys.executable, 'scripts/export_model_1bit.py', '--hf-model', str(hf_dir), '--out', out_npz, '--bits', '1'], cwd=str(repo_root), env=env)
    assert Path(out_npz).exists()

    # Check the NPZ contains either embedding or some typical keys
    npz = np.load(out_npz, allow_pickle=True)
    # Accept either embedding weight or q_proj keys
    found = any(k for k in npz.keys() if 'embedding' in k or 'self_attn' in k or 'wte' in k)
    assert found

    # Bits==2 export
    out_npz_2 = str(hf_dir / 'out_q2.npz')
    subprocess.check_call([sys.executable, 'scripts/export_model_1bit.py', '--hf-model', str(hf_dir), '--out', out_npz_2, '--bits', '2'], cwd=str(repo_root), env=env)
    assert Path(out_npz_2).exists()
    npz2 = np.load(out_npz_2, allow_pickle=True)
    # Now assert we see q2 keys
    found_q2 = any(k for k in npz2.keys() if '_q2' in k or '.weight_q2' in k)
    assert found_q2


def test_gguf_to_apl_uses_export_model_1bit(tmp_path):
    hf_dir = Path(tmp_path) / 'tiny_hf'
    hf_dir.mkdir()
    cfg = GPT2Config(vocab_size=64, n_embd=32, n_layer=1, n_head=4)
    model = GPT2LMHeadModel(cfg)
    model.save_pretrained(hf_dir)

    # Call gguf_to_apl with hf-dir and ensure export_model_1bit runs
    out_dir = Path(tmp_path) / 'out'
    out_dir.mkdir()
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(repo_root)
    subprocess.check_call([sys.executable, 'scripts/gguf_to_apl.py', '--hf-dir', str(hf_dir), '--out-dir', str(out_dir), '--run-export', '--quantizer', 'export_model_1bit', '--bits', '1'], cwd=str(repo_root), env=env)
    # Expect the manifest to be present
    expected_manifest = out_dir / f'{hf_dir.name}_manifest.json'
    assert expected_manifest.exists()
