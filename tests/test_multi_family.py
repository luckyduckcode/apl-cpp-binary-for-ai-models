#!/usr/bin/env python3
"""Tests for multi-model family support in the export pipeline.

Validates that the exporter correctly generates v2 manifests with
architecture metadata for different model families (Llama, Mistral,
DeepSeek-R1, Code Llama, Gemma, Qwen).
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# Model family configurations with expected architecture metadata
MODEL_FAMILIES = {
    'llama': {
        'num_heads': 32,
        'kv_groups': None,
        'context_length': 4096,
        'attention_variant': 'full',
        'activation': 'swiglu',
        'norm_type': 'rmsnorm',
        'rope_base': 10000.0,
    },
    'mistral': {
        'num_heads': 32,
        'kv_groups': 8,
        'context_length': 32768,
        'attention_variant': 'sliding-window',
        'window_size': 4096,
        'activation': 'swiglu',
        'norm_type': 'rmsnorm',
        'rope_base': 10000.0,
    },
    'deepseek-r1': {
        'num_heads': 32,
        'kv_groups': 8,
        'context_length': 65536,
        'attention_variant': 'gqa',
        'activation': 'swiglu',
        'norm_type': 'rmsnorm',
        'rope_base': 10000.0,
    },
    'code-llama': {
        'num_heads': 32,
        'kv_groups': None,
        'context_length': 16384,
        'attention_variant': 'full',
        'activation': 'swiglu',
        'norm_type': 'rmsnorm',
        'rope_base': 1000000.0,
    },
    'gemma': {
        'num_heads': 16,
        'kv_groups': 1,
        'context_length': 8192,
        'attention_variant': 'mqa',
        'activation': 'geglu',
        'norm_type': 'rmsnorm',
        'rope_base': 10000.0,
    },
    'qwen': {
        'num_heads': 32,
        'kv_groups': 32,
        'context_length': 32768,
        'attention_variant': 'full',
        'activation': 'swiglu',
        'norm_type': 'rmsnorm',
        'rope_base': 1000000.0,
    },
}


def create_toy_npz(tmp_path: Path, hidden_size=64, num_layers=2, vocab_size=1000, out_features=1000):
    """Create a toy quantized NPZ for testing the exporter."""
    data = {}
    
    # Embedding (not quantized, just shape)
    emb_shape = (vocab_size, hidden_size)
    data['embedding.weight'] = np.random.randn(*emb_shape).astype(np.float32)
    
    # Transformer layers
    for layer_idx in range(num_layers):
        prefix = f'transformer.layers.{layer_idx}'
        
        # Self-attention in_proj (3 * hidden for Q, K, V)
        in_proj_shape = (3 * hidden_size, hidden_size)
        packed = np.packbits(np.random.randint(0, 2, in_proj_shape, dtype=np.uint8), axis=1)
        scales = np.random.rand(in_proj_shape[0]).astype(np.float32) * 0.1
        data[f'{prefix}.self_attn.in_proj_weight_1bit'] = packed
        data[f'{prefix}.self_attn.in_proj_weight_shape'] = np.array(in_proj_shape)
        data[f'{prefix}.self_attn.in_proj_weight_scales'] = scales
        
        # Self-attention out_proj
        out_proj_shape = (hidden_size, hidden_size)
        packed = np.packbits(np.random.randint(0, 2, out_proj_shape, dtype=np.uint8), axis=1)
        scales = np.random.rand(out_proj_shape[0]).astype(np.float32) * 0.1
        data[f'{prefix}.self_attn.out_proj.weight_1bit'] = packed
        data[f'{prefix}.self_attn.out_proj.weight_shape'] = np.array(out_proj_shape)
        data[f'{prefix}.self_attn.out_proj.weight_scales'] = scales
        
        # FFN linear1
        ffn_shape = (hidden_size * 4, hidden_size)
        packed = np.packbits(np.random.randint(0, 2, ffn_shape, dtype=np.uint8), axis=1)
        scales = np.random.rand(ffn_shape[0]).astype(np.float32) * 0.1
        data[f'{prefix}.linear1.weight_1bit'] = packed
        data[f'{prefix}.linear1.weight_shape'] = np.array(ffn_shape)
        data[f'{prefix}.linear1.weight_scales'] = scales
        
        # FFN linear2
        ffn2_shape = (hidden_size, hidden_size * 4)
        packed = np.packbits(np.random.randint(0, 2, ffn2_shape, dtype=np.uint8), axis=1)
        scales = np.random.rand(ffn2_shape[0]).astype(np.float32) * 0.1
        data[f'{prefix}.linear2.weight_1bit'] = packed
        data[f'{prefix}.linear2.weight_shape'] = np.array(ffn2_shape)
        data[f'{prefix}.linear2.weight_scales'] = scales
    
    # Output FC layer
    fc_shape = (out_features, hidden_size)
    packed = np.packbits(np.random.randint(0, 2, fc_shape, dtype=np.uint8), axis=1)
    scales = np.random.rand(fc_shape[0]).astype(np.float32) * 0.1
    data['fc.weight_1bit'] = packed
    data['fc.weight_shape'] = np.array(fc_shape)
    data['fc.weight_scales'] = scales
    
    npz_path = tmp_path / 'test_quantized.npz'
    np.savez(npz_path, **data)
    return npz_path


@pytest.fixture
def toy_npz(tmp_path):
    """Fixture that creates a toy NPZ for each test."""
    return create_toy_npz(tmp_path)


class TestManifestV2Schema:
    """Tests for v2 manifest schema structure."""
    
    def test_manifest_has_format_version(self, toy_npz, tmp_path):
        """Manifest should have format_version: 2."""
        manifest_path = tmp_path / 'manifest.json'
        subprocess.run([
            sys.executable, str(PROJECT_ROOT / 'export_quantized_for_apl.py'),
            '--npz', str(toy_npz),
            '--out_manifest', str(manifest_path),
        ], check=True, cwd=str(tmp_path))
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        assert manifest.get('format_version') == 2
    
    def test_manifest_has_model_section(self, toy_npz, tmp_path):
        """Manifest should have model section with family info."""
        manifest_path = tmp_path / 'manifest.json'
        subprocess.run([
            sys.executable, str(PROJECT_ROOT / 'export_quantized_for_apl.py'),
            '--npz', str(toy_npz),
            '--out_manifest', str(manifest_path),
            '--model-family', 'mistral',
            '--target-families', 'llama,qwen',
        ], check=True, cwd=str(tmp_path))
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        model = manifest.get('model', {})
        assert model.get('primary_family') == 'mistral'
        # families includes primary + target families
        families = model.get('families', [])
        assert 'llama' in families
        assert 'qwen' in families
    
    def test_manifest_has_architecture_section(self, toy_npz, tmp_path):
        """Manifest should have architecture section with layer info."""
        manifest_path = tmp_path / 'manifest.json'
        subprocess.run([
            sys.executable, str(PROJECT_ROOT / 'export_quantized_for_apl.py'),
            '--npz', str(toy_npz),
            '--out_manifest', str(manifest_path),
            '--num-heads', '32',
            '--hidden-size', '64',
            '--num-layers', '2',
        ], check=True, cwd=str(tmp_path))
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        arch = manifest.get('architecture', {})
        # num_heads is nested under attention
        attn = arch.get('attention', {})
        assert attn.get('num_heads') == 32
        assert arch.get('hidden_size') == 64
        assert arch.get('num_layers') == 2
    
    def test_manifest_has_quantization_section(self, toy_npz, tmp_path):
        """Manifest should have quantization section."""
        manifest_path = tmp_path / 'manifest.json'
        subprocess.run([
            sys.executable, str(PROJECT_ROOT / 'export_quantized_for_apl.py'),
            '--npz', str(toy_npz),
            '--out_manifest', str(manifest_path),
        ], check=True, cwd=str(tmp_path))
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        quant = manifest.get('quantization', {})
        assert quant.get('bit_width') == 1
        assert 'scale' in quant
        assert 'packing' in quant
    
    def test_manifest_has_weights_section(self, toy_npz, tmp_path):
        """Manifest should have weights section with packed binaries."""
        manifest_path = tmp_path / 'manifest.json'
        subprocess.run([
            sys.executable, str(PROJECT_ROOT / 'export_quantized_for_apl.py'),
            '--npz', str(toy_npz),
            '--out_manifest', str(manifest_path),
        ], check=True, cwd=str(tmp_path))
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        weights = manifest.get('weights', {})
        assert 'fc.weight' in weights
        fc = weights['fc.weight']
        assert 'packed' in fc
        assert 'shape' in fc
        assert 'scales' in fc or 'scales_txt' in fc
        assert fc.get('bit_width') == 1


class TestModelFamilyConfigurations:
    """Tests for different model family configurations."""
    
    @pytest.mark.parametrize('family,config', list(MODEL_FAMILIES.items()))
    def test_model_family_export(self, toy_npz, tmp_path, family, config):
        """Export with specific model family configuration."""
        manifest_path = tmp_path / 'manifest.json'
        
        cmd = [
            sys.executable, str(PROJECT_ROOT / 'export_quantized_for_apl.py'),
            '--npz', str(toy_npz),
            '--out_manifest', str(manifest_path),
            '--model-family', family,
            '--num-heads', str(config['num_heads']),
            '--context-length', str(config['context_length']),
            '--attention-variant', config['attention_variant'],
            '--activation', config['activation'],
            '--norm-type', config['norm_type'],
        ]
        
        if config.get('kv_groups'):
            cmd.extend(['--kv-groups', str(config['kv_groups'])])
        if config.get('window_size'):
            cmd.extend(['--window-size', str(config['window_size'])])
        if config.get('rope_base'):
            cmd.extend(['--rope-base', str(config['rope_base'])])
        
        subprocess.run(cmd, check=True, cwd=str(tmp_path))
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        # Verify model family
        model = manifest.get('model', {})
        assert model.get('primary_family') == family
        
        # Verify architecture
        arch = manifest.get('architecture', {})
        attn = arch.get('attention', {})
        assert attn.get('num_heads') == config['num_heads']
        assert arch.get('context_length') == config['context_length']
        
        # Verify attention config
        assert attn.get('variant') == config['attention_variant']
        if config.get('kv_groups'):
            assert attn.get('kv_groups') == config['kv_groups']
        if config.get('window_size'):
            assert attn.get('window_size') == config['window_size']
        
        # Verify activation (top-level in architecture)
        assert arch.get('activation') == config['activation']
        
        # Verify normalization (stored as 'norm')
        assert arch.get('norm') == config['norm_type']
        
        # Verify RoPE if specified
        if config.get('rope_base'):
            rope = arch.get('rope', {})
            assert rope.get('base_theta') == config['rope_base']
    
    def test_cross_family_compatibility(self, toy_npz, tmp_path):
        """Test that target_families allows cross-family loading."""
        manifest_path = tmp_path / 'manifest.json'
        subprocess.run([
            sys.executable, str(PROJECT_ROOT / 'export_quantized_for_apl.py'),
            '--npz', str(toy_npz),
            '--out_manifest', str(manifest_path),
            '--model-family', 'llama',
            '--target-families', 'mistral,code-llama,qwen',
        ], check=True, cwd=str(tmp_path))
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        model = manifest.get('model', {})
        families = model.get('families', [])
        assert 'mistral' in families
        assert 'code-llama' in families
        assert 'qwen' in families


class TestArchitectureInference:
    """Tests for automatic architecture inference from weights."""
    
    def test_infer_num_layers(self, toy_npz, tmp_path):
        """Should infer num_layers from transformer layer indices."""
        manifest_path = tmp_path / 'manifest.json'
        subprocess.run([
            sys.executable, str(PROJECT_ROOT / 'export_quantized_for_apl.py'),
            '--npz', str(toy_npz),
            '--out_manifest', str(manifest_path),
        ], check=True, cwd=str(tmp_path))
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        arch = manifest.get('architecture', {})
        # Toy NPZ has 2 layers
        assert arch.get('num_layers') == 2
    
    def test_infer_hidden_size(self, toy_npz, tmp_path):
        """Should infer hidden_size from weight shapes."""
        manifest_path = tmp_path / 'manifest.json'
        subprocess.run([
            sys.executable, str(PROJECT_ROOT / 'export_quantized_for_apl.py'),
            '--npz', str(toy_npz),
            '--out_manifest', str(manifest_path),
        ], check=True, cwd=str(tmp_path))
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        arch = manifest.get('architecture', {})
        # Toy NPZ uses hidden_size=64
        assert arch.get('hidden_size') == 64
    
    def test_explicit_overrides_inference(self, toy_npz, tmp_path):
        """Explicit CLI args should override inferred values."""
        manifest_path = tmp_path / 'manifest.json'
        subprocess.run([
            sys.executable, str(PROJECT_ROOT / 'export_quantized_for_apl.py'),
            '--npz', str(toy_npz),
            '--out_manifest', str(manifest_path),
            '--num-layers', '32',
            '--hidden-size', '4096',
        ], check=True, cwd=str(tmp_path))
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        arch = manifest.get('architecture', {})
        assert arch.get('num_layers') == 32
        assert arch.get('hidden_size') == 4096


class TestRoPEConfiguration:
    """Tests for RoPE (Rotary Position Embedding) configuration."""
    
    def test_rope_base_theta(self, toy_npz, tmp_path):
        """Should include rope.base_theta in manifest."""
        manifest_path = tmp_path / 'manifest.json'
        subprocess.run([
            sys.executable, str(PROJECT_ROOT / 'export_quantized_for_apl.py'),
            '--npz', str(toy_npz),
            '--out_manifest', str(manifest_path),
            '--rope-base', '500000',
        ], check=True, cwd=str(tmp_path))
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        arch = manifest.get('architecture', {})
        rope = arch.get('rope', {})
        assert rope.get('base_theta') == 500000.0
    
    def test_rope_scale_factor(self, toy_npz, tmp_path):
        """Should include rope.scale for YaRN-style scaling."""
        manifest_path = tmp_path / 'manifest.json'
        subprocess.run([
            sys.executable, str(PROJECT_ROOT / 'export_quantized_for_apl.py'),
            '--npz', str(toy_npz),
            '--out_manifest', str(manifest_path),
            '--rope-base', '10000',
            '--rope-scale', '4.0',
        ], check=True, cwd=str(tmp_path))
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        rope = manifest.get('architecture', {}).get('rope', {})
        assert rope.get('scale') == 4.0


class TestWeightValidation:
    """Tests for weight file validation."""
    
    def test_packed_binaries_created(self, toy_npz, tmp_path):
        """Should create .bin files for packed weights."""
        manifest_path = tmp_path / 'manifest.json'
        subprocess.run([
            sys.executable, str(PROJECT_ROOT / 'export_quantized_for_apl.py'),
            '--npz', str(toy_npz),
            '--out_manifest', str(manifest_path),
        ], check=True, cwd=str(tmp_path))
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        weights = manifest.get('weights', {})
        for name, entry in weights.items():
            packed_path = tmp_path / entry['packed']
            assert packed_path.exists(), f"Missing packed file for {name}"
    
    def test_scales_files_created(self, toy_npz, tmp_path):
        """Should create scales files (.npy and .txt) for each weight."""
        manifest_path = tmp_path / 'manifest.json'
        subprocess.run([
            sys.executable, str(PROJECT_ROOT / 'export_quantized_for_apl.py'),
            '--npz', str(toy_npz),
            '--out_manifest', str(manifest_path),
        ], check=True, cwd=str(tmp_path))
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        weights = manifest.get('weights', {})
        for name, entry in weights.items():
            if 'scales' in entry:
                scales_path = tmp_path / entry['scales']
                assert scales_path.exists(), f"Missing scales .npy for {name}"
            if 'scales_txt' in entry:
                scales_txt_path = tmp_path / entry['scales_txt']
                assert scales_txt_path.exists(), f"Missing scales .txt for {name}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
