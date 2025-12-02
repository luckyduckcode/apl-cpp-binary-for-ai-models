#!/usr/bin/env python3
"""
Easy Model Runner for APL Binary AI Models

This script allows users to easily run popular AI models using the APL binary quantization framework.
It handles downloading, quantizing, exporting, and running models with a simple interface.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import json
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig
    from huggingface_hub import HfApi
except ImportError:
    print("Required packages not installed. Please run: pip install -r requirements.txt")
    sys.exit(1)

from quantization import binarize_weights

# Popular models configuration
POPULAR_MODELS = {
    'tinyllama': {
        'name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'family': 'llama',
        'context_length': 2048,
        'layers': 22,  # Full model
        'description': 'TinyLlama 1.1B - Small but capable Llama model'
    },
    'llama-7b': {
        'name': 'meta-llama/Llama-2-7b-hf',
        'family': 'llama',
        'context_length': 4096,
        'layers': 32,
        'description': 'Llama 2 7B - Requires access token'
    },
    'mistral-7b': {
        'name': 'mistralai/Mistral-7B-v0.1',
        'family': 'mistral',
        'context_length': 4096,
        'layers': 32,
        'description': 'Mistral 7B - Fast and capable'
    },
    'gemma-2b': {
        'name': 'google/gemma-2b',
        'family': 'gemma',
        'context_length': 8192,
        'layers': 18,
        'description': 'Gemma 2B - Google\'s lightweight model'
    }
}

def download_and_quantize_model(model_key, output_npz, layers=None):
    """Download a model from HuggingFace and quantize it."""
    if model_key not in POPULAR_MODELS:
        print(f"Model {model_key} not in supported list. Available: {list(POPULAR_MODELS.keys())}")
        return False
    
    model_info = POPULAR_MODELS[model_key]
    model_name = model_info['name']
    num_layers = layers or model_info['layers']
    
    print(f"Loading model: {model_name}")
    
    try:
        config = AutoConfig.from_pretrained(model_name)
        print(f"Config: hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
              f"heads={config.num_attention_heads}")
        
        # Load model on CPU explicitly without device map to avoid needing accelerate.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Make sure you have access to the model and necessary tokens configured.")
        return False
    
    print(f"Quantizing first {num_layers} layers...")
    
    data = {}
    
    # Store architecture info
    data['_config_hidden_size'] = np.array([config.hidden_size])
    data['_config_intermediate_size'] = np.array([config.intermediate_size])
    data['_config_num_layers'] = np.array([num_layers])
    data['_config_num_heads'] = np.array([config.num_attention_heads])
    data['_config_vocab_size'] = np.array([config.vocab_size])
    data['_config_rope_theta'] = np.array([getattr(config, 'rope_theta', 10000.0)])
    
    # Include embedding
    emb = model.model.embed_tokens.weight.detach().numpy()
    print(f"Embedding: {emb.shape}")
    packed, scales = quantize_weight(emb)
    data['embedding.weight_1bit'] = packed
    data['embedding.weight_shape'] = np.array(emb.shape)
    data['embedding.weight_scales'] = scales
    
    # Quantize transformer layers
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        prefix = f'transformer.layers.{layer_idx}'
        
        # Self-attention projections
        q_weight = layer.self_attn.q_proj.weight.detach()
        k_weight = layer.self_attn.k_proj.weight.detach()
        v_weight = layer.self_attn.v_proj.weight.detach()
        o_weight = layer.self_attn.o_proj.weight.detach()
        
        print(f"Layer {layer_idx}: Q={tuple(q_weight.shape)}, O={tuple(o_weight.shape)}")
        
        # Quantize Q, K, V, O
        for name, weight in [('q_proj', q_weight), ('k_proj', k_weight), ('v_proj', v_weight), ('o_proj', o_weight)]:
            packed, scales = quantize_weight(weight)
            data[f'{prefix}.self_attn.{name}.weight_1bit'] = packed
            data[f'{prefix}.self_attn.{name}.weight_shape'] = np.array(weight.shape)
            data[f'{prefix}.self_attn.{name}.weight_scales'] = scales
        
        # FFN
        gate_weight = layer.mlp.gate_proj.weight.detach()
        up_weight = layer.mlp.up_proj.weight.detach()
        down_weight = layer.mlp.down_proj.weight.detach()
        
        print(f"Layer {layer_idx}: FFN gate={tuple(gate_weight.shape)}, down={tuple(down_weight.shape)}")
        
        for name, weight in [('gate_proj', gate_weight), ('up_proj', up_weight), ('down_proj', down_weight)]:
            packed, scales = quantize_weight(weight)
            data[f'{prefix}.mlp.{name}.weight_1bit'] = packed
            data[f'{prefix}.mlp.{name}.weight_shape'] = np.array(weight.shape)
            data[f'{prefix}.mlp.{name}.weight_scales'] = scales
        
        # RMSNorm weights (keep as fp32)
        input_norm = layer.input_layernorm.weight.detach().numpy()
        post_norm = layer.post_attention_layernorm.weight.detach().numpy()
        data[f'{prefix}.input_layernorm.weight_fp32'] = input_norm
        data[f'{prefix}.post_attention_layernorm.weight_fp32'] = post_norm
    
    # Include LM head
    lm_head = model.lm_head.weight.detach().numpy()
    print(f"LM head: {lm_head.shape}")
    packed, scales = quantize_weight(lm_head)
    data['lm_head.weight_1bit'] = packed
    data['lm_head.weight_shape'] = np.array(lm_head.shape)
    data['lm_head.weight_scales'] = scales
    
    # Save
    np.savez(output_npz, **data)
    print(f"Saved quantized weights to {output_npz}")
    
    return True

def quantize_weight(weight_tensor):
    """Quantize a weight tensor to 1-bit."""
    import numpy as np
    w = weight_tensor.numpy() if hasattr(weight_tensor, 'numpy') else np.array(weight_tensor)
    packed, scales, _ = binarize_weights(w, per_channel_axis=0)
    return packed, scales

def export_model(npz_path, manifest_path, model_info):
    """Export the quantized model for APL."""
    cmd = [
        sys.executable, 'export_quantized_for_apl.py',
        '--npz', npz_path,
        '--out_manifest', manifest_path,
        '--model-family', model_info['family'],
        '--context-length', str(model_info['context_length']),
        '--activation', 'swiglu',
        '--norm-type', 'rmsnorm'
    ]
    
    if model_info['family'] == 'mistral':
        cmd.extend(['--attention-variant', 'gqa', '--kv-groups', '8'])
    elif model_info['family'] == 'gemma':
        cmd.extend(['--attention-variant', 'mqa', '--kv-groups', '1', '--activation', 'geglu'])
    
    print("Running export command:", ' '.join(cmd))
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0

def run_demo(manifest_path):
    """Run the APL demo with the exported model."""
    print("Running APL demo...")
    # Generate APL manifest
    cmd_gen = [sys.executable, 'scripts/manifest_to_apl.py', '--manifest', manifest_path, '--out_apl', 'apl/generated_manifest.apl']
    subprocess.run(cmd_gen, cwd=Path(__file__).parent)

    # Build backend for platform using cross-platform helper
    build_script = Path(__file__).parent / 'scripts' / 'build_backend.py'
    if build_script.exists():
        print("Running cross-platform build script...")
        subprocess.run([sys.executable, str(build_script)], cwd=Path(__file__).parent)
    else:
        # Backwards compat: try platform-specific scripts
        if os.name == 'nt':
            build_script = Path(__file__).parent / 'scripts' / 'build_backend_windows.ps1'
            if build_script.exists():
                print("Building backend for Windows using PowerShell...")
                subprocess.run(['powershell.exe', '-ExecutionPolicy', 'Bypass', '-File', str(build_script)], cwd=Path(__file__).parent)
            else:
                print("No Windows build script found: scripts/build_backend_windows.ps1. Please build manually or enable WSL.")
        else:
            bash_build = Path(__file__).parent / 'scripts' / 'build_backend.sh'
            if bash_build.exists():
                print("Building backend using bash script...")
                subprocess.run(['bash', str(bash_build)], cwd=Path(__file__).parent)
            else:
                print("No build script found; please build the native backend manually")

    # Run the demo
    demo_cmd = [sys.executable, 'cpp/call_backend.py', '--manifest', manifest_path, '--input', 'test_input.txt']
    print("Running demo command:", ' '.join(demo_cmd))
    subprocess.run(demo_cmd, cwd=Path(__file__).parent)

def main():
    parser = argparse.ArgumentParser(description="Easy Model Runner for APL Binary AI Models")
    parser.add_argument('--model', choices=list(POPULAR_MODELS.keys()), 
                       help='Choose a popular model to run')
    parser.add_argument('--custom-model', help='HuggingFace model name for custom model')
    parser.add_argument('--layers', type=int, default=None, 
                       help='Number of layers to quantize (default: all)')
    parser.add_argument('--output-dir', default='models', 
                       help='Output directory for quantized models')
    parser.add_argument('--skip-quantize', action='store_true', 
                       help='Skip quantization if NPZ already exists')
    parser.add_argument('--run-demo', action='store_true', default=True,
                       help='Run the demo after export')
    
    args = parser.parse_args()
    
    if not args.model and not args.custom_model:
        print("Available popular models:")
        for key, info in POPULAR_MODELS.items():
            print(f"  {key}: {info['description']}")
        print("\nUsage: python easy_run.py --model <model_key>")
        print("Or: python easy_run.py --custom-model <hf_model_name>")
        return
    
    # Determine model info
    if args.model:
        model_key = args.model
        model_info = POPULAR_MODELS[model_key]
        hf_name = model_info['name']
    else:
        model_key = args.custom_model.replace('/', '_')
        hf_name = args.custom_model
        # Default to llama family for custom
        model_info = {
            'name': hf_name,
            'family': 'llama',
            'context_length': 4096,
            'layers': args.layers or 32
        }
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    npz_path = output_dir / f"{model_key}_quantized.npz"
    manifest_path = output_dir / f"{model_key}_manifest.json"
    
    if not args.skip_quantize or not npz_path.exists():
        print(f"Quantizing model {hf_name}...")
        if not download_and_quantize_model(args.model or 'custom', str(npz_path), args.layers):
            return
    else:
        print(f"Using existing quantized model: {npz_path}")
    
    print("Exporting for APL...")
    if not export_model(str(npz_path), str(manifest_path), model_info):
        print("Export failed")
        return
    
    if args.run_demo:
        run_demo(str(manifest_path))
    
    print("Done! Model is ready to use with APL.")

if __name__ == '__main__':
    main()