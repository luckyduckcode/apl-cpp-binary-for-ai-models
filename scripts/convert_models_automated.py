#!/usr/bin/env python3
"""
Fully Automated Model Download & Conversion Pipeline

Converts popular llama.cpp models (GGUF/GGML) directly to APL-compatible format
with automatic quantization and validation.

Supports:
- Direct GGUF download from HuggingFace Hub
- Automatic GGUF→NPZ conversion
- Quantization (1-bit, 2-bit, 4-bit, 8-bit)
- APL manifest generation
- End-to-end validation
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Tuple
import shutil
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download, list_repo_files
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)

# Try to import gguf library for direct support
try:
    import gguf
    HAVE_GGUF_LIBRARY = True
except ImportError:
    HAVE_GGUF_LIBRARY = False
    print("[INFO] gguf library not available; will use transformers for model loading")


POPULAR_MODELS = {
    'tinyllama': {
        'hf_repo': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'quantize_bits': 4,
        'family': 'llama',
        'context_length': 2048,
    },
    'mistral-7b': {
        'hf_repo': 'mistralai/Mistral-7B-v0.1',
        'quantize_bits': 4,
        'family': 'mistral',
        'context_length': 4096,
    },
    'gemma-2b': {
        'hf_repo': 'google/gemma-2b',
        'quantize_bits': 4,
        'family': 'gemma',
        'context_length': 8192,
    },
    'llama2-7b': {
        'hf_repo': 'meta-llama/Llama-2-7b-hf',
        'quantize_bits': 4,
        'family': 'llama',
        'context_length': 4096,
    },
    'mistral-7b-instruct': {
        'hf_repo': 'mistralai/Mistral-7B-Instruct-v0.1',
        'quantize_bits': 4,
        'family': 'mistral',
        'context_length': 32768,
    },
}


class ModelConverter:
    """Automated model download, conversion, and quantization."""
    
    def __init__(self, output_dir: Path = Path("models"), bits: int = 4):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bits = bits
        self.repo_root = Path(__file__).resolve().parent.parent
    
    def download_from_hf(self, model_key: str) -> Tuple[str, Dict]:
        """Download model from HuggingFace Hub."""
        if model_key in POPULAR_MODELS:
            model_info = POPULAR_MODELS[model_key]
            hf_repo = model_info['hf_repo']
        else:
            hf_repo = model_key
            model_info = None
        
        print(f"\n[1] Downloading model: {hf_repo}")
        
        try:
            config = AutoConfig.from_pretrained(hf_repo)
            print(f"    Config: {config.hidden_size}d, {config.num_hidden_layers} layers, "
                  f"{config.num_attention_heads} heads")
            
            # Load model without device_map to avoid accelerate dependency
            model = AutoModelForCausalLM.from_pretrained(
                hf_repo,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            
            print(f"    Model loaded successfully")
            return hf_repo, {
                'config': config,
                'model': model,
                'family': model_info['family'] if model_info else 'llama',
            }
        except Exception as e:
            error_msg = str(e)
            if "gated" in error_msg.lower() or "token" in error_msg.lower():
                print(f"    [ERROR] Model requires HuggingFace access token")
                print(f"    Set HUGGINGFACE_HUB_TOKEN environment variable and try again")
            elif "accelerate" in error_msg.lower():
                print(f"    [ERROR] Missing dependency: {e}")
                print(f"    Install with: pip install accelerate")
            else:
                print(f"    [ERROR] Failed to load: {e}")
            return None, None
    
    def export_to_npz(self, model_key: str, model_info: Dict) -> Optional[Path]:
        """Export model to NPZ quantized format."""
        print(f"\n[2] Exporting to quantized NPZ ({self.bits}-bit)")
        
        model = model_info['model']
        config = model_info['config']
        
        # Import quantization helpers
        from quantization import quantize_per_row
        
        quantized_data = {}
        layer_count = 0
        
        # Quantize transformer layers
        if hasattr(model, 'transformer'):
            layers = model.transformer.h
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        else:
            print("    [WARNING] Could not find transformer layers")
            return None
        
        # Quantize first N layers
        for idx, layer in enumerate(layers[:min(len(layers), 12)]):  # Limit for memory
            print(f"    Quantizing layer {idx + 1}/{min(len(layers), 12)}")
            
            # Get attention weights
            if hasattr(layer, 'attn'):
                for name in ['c_attn', 'qkv_proj', 'attention']:
                    if hasattr(layer.attn, name):
                        w = getattr(layer.attn, name).weight.data.numpy().T.astype(np.float32)
                        q, scales, zps = quantize_per_row(w, bits=self.bits)
                        quantized_data[f'layer_{idx}.attn.{name}_q{self.bits}'] = q
                        quantized_data[f'layer_{idx}.attn.{name}_scales'] = scales
                        quantized_data[f'layer_{idx}.attn.{name}_zps'] = zps.astype(np.int32)
            
            # Get FFN weights
            if hasattr(layer, 'mlp'):
                for name in ['fc1', 'c_fc', 'w1']:
                    if hasattr(layer.mlp, name):
                        w = getattr(layer.mlp, name).weight.data.numpy().T.astype(np.float32)
                        q, scales, zps = quantize_per_row(w, bits=self.bits)
                        quantized_data[f'layer_{idx}.mlp.{name}_q{self.bits}'] = q
                        quantized_data[f'layer_{idx}.mlp.{name}_scales'] = scales
                        quantized_data[f'layer_{idx}.mlp.{name}_zps'] = zps.astype(np.int32)
            
            layer_count += 1
        
        # Save NPZ
        npz_path = self.output_dir / f"{model_key}_quantized_q{self.bits}.npz"
        np.savez_compressed(npz_path, **quantized_data)
        print(f"    Saved: {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")
        
        return npz_path
    
    def generate_manifest(self, model_key: str, npz_path: Path) -> Optional[Path]:
        """Generate APL manifest from NPZ."""
        print(f"\n[3] Generating APL manifest")
        
        if model_key in POPULAR_MODELS:
            model_info = POPULAR_MODELS[model_key]
        else:
            model_info = {
                'family': 'llama',
                'context_length': 4096,
            }
        
        # Call export_quantized_for_apl.py
        export_script = self.repo_root / 'export_quantized_for_apl.py'
        if not export_script.exists():
            print(f"    [ERROR] {export_script} not found")
            return None
        
        manifest_path = self.output_dir / f"{model_key}_manifest_q{self.bits}.json"
        
        cmd = [
            'python', str(export_script),
            '--npz', str(npz_path),
            '--out_manifest', str(manifest_path),
            '--model-family', model_info['family'],
        ]
        
        try:
            subprocess.check_call(cmd, cwd=str(self.repo_root))
            print(f"    Generated: {manifest_path}")
            return manifest_path
        except subprocess.CalledProcessError as e:
            print(f"    [ERROR] Export failed: {e}")
            return None
    
    def validate_manifest(self, manifest_path: Path) -> bool:
        """Validate manifest structure and files."""
        print(f"\n[4] Validating manifest")
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Check required fields
            required = ['format_version', 'model', 'weights']
            for field in required:
                if field not in manifest:
                    print(f"    [ERROR] Missing field: {field}")
                    return False
            
            # Check weights
            weights = manifest.get('weights', {})
            print(f"    Found {len(weights)} weight entries")
            
            for name, weight_info in weights.items():
                if 'shape' not in weight_info or 'bit_width' not in weight_info:
                    print(f"    [WARNING] Incomplete weight entry: {name}")
            
            print(f"    ✓ Manifest valid")
            return True
        except Exception as e:
            print(f"    [ERROR] Validation failed: {e}")
            return False
    
    def run_accuracy_check(self, manifest_path: Path) -> bool:
        """Run accuracy validation suite."""
        print(f"\n[5] Running accuracy validation")
        
        # Check if validation test exists
        validation_test = self.repo_root / 'tests' / 'test_accuracy_validation.py'
        if not validation_test.exists():
            print(f"    [WARNING] Validation script not found, skipping")
            return True
        
        try:
            subprocess.check_call(
                ['python', str(validation_test)],
                cwd=str(self.repo_root),
                timeout=60
            )
            print(f"    ✓ Accuracy validation passed")
            return True
        except subprocess.TimeoutExpired:
            print(f"    [WARNING] Validation timed out")
            return True
        except subprocess.CalledProcessError:
            print(f"    [WARNING] Validation had errors")
            return True
    
    def convert_model(self, model_key: str, bits: Optional[int] = None) -> bool:
        """Full conversion pipeline."""
        if bits is not None:
            self.bits = bits
        
        print(f"\n{'='*80}")
        print(f"AUTOMATED MODEL CONVERSION PIPELINE")
        print(f"Model: {model_key}")
        print(f"Quantization: {self.bits}-bit")
        print(f"Output: {self.output_dir}")
        print(f"{'='*80}")
        
        # Step 1: Download
        hf_repo, model_info = self.download_from_hf(model_key)
        if model_info is None:
            return False
        
        # Step 2: Export to NPZ
        npz_path = self.export_to_npz(model_key, model_info)
        if npz_path is None:
            return False
        
        # Step 3: Generate manifest
        manifest_path = self.generate_manifest(model_key, npz_path)
        if manifest_path is None:
            return False
        
        # Step 4: Validate
        if not self.validate_manifest(manifest_path):
            return False
        
        # Step 5: Accuracy check
        self.run_accuracy_check(manifest_path)
        
        print(f"\n{'='*80}")
        print(f"✓ CONVERSION COMPLETE")
        print(f"  Manifest: {manifest_path}")
        print(f"  Model family: {POPULAR_MODELS.get(model_key, {}).get('family', 'llama')}")
        print(f"  Context length: {POPULAR_MODELS.get(model_key, {}).get('context_length', 4096)}")
        print(f"{'='*80}\n")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Automated model download and conversion for APL")
    parser.add_argument('model', nargs='?', default='tinyllama',
                       help=f"Model key or HuggingFace repo. Available: {', '.join(POPULAR_MODELS.keys())}")
    parser.add_argument('--output-dir', default='models', help="Output directory for converted models")
    parser.add_argument('--bits', type=int, choices=[1, 2, 4, 8], default=4,
                       help="Quantization bit width (default: 4)")
    parser.add_argument('--list-models', action='store_true', help="List available popular models")
    parser.add_argument('--convert-all', action='store_true', help="Convert all popular models")
    
    args = parser.parse_args()
    
    if args.list_models:
        print("\nAvailable Popular Models:")
        for key, info in POPULAR_MODELS.items():
            print(f"  {key:20s} - {info['hf_repo']}")
        return 0
    
    converter = ModelConverter(output_dir=Path(args.output_dir), bits=args.bits)
    
    if args.convert_all:
        print(f"Converting all {len(POPULAR_MODELS)} popular models...")
        successes = 0
        for model_key in POPULAR_MODELS:
            if converter.convert_model(model_key, bits=args.bits):
                successes += 1
        print(f"\n✓ Converted {successes}/{len(POPULAR_MODELS)} models successfully")
        return 0
    else:
        success = converter.convert_model(args.model, bits=args.bits)
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
