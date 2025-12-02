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

from quantization import binarize_weights, quantize_per_row

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

def download_and_quantize_model(model_key, output_npz, layers=None, bits=1):
    """Download a model from HuggingFace and quantize it. model_key may be a known popular key or a HF repo path/local dir."""
    if model_key in POPULAR_MODELS:
        model_info = POPULAR_MODELS[model_key]
        model_name = model_info['name']
    else:
        model_name = model_key
        # Default info for custom HF dirs
        model_info = {
            'name': model_name,
            'family': 'llama',
            'context_length': 4096,
            'layers': layers or 32
        }
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
    data['_config_intermediate_size'] = np.array([getattr(config, 'intermediate_size', 0)])
    data['_config_num_layers'] = np.array([num_layers])
    data['_config_num_heads'] = np.array([config.num_attention_heads])
    data['_config_vocab_size'] = np.array([config.vocab_size])
    data['_config_rope_theta'] = np.array([getattr(config, 'rope_theta', 10000.0)])
    
    # Include embedding -- robust across different HF architectures
    def get_embedding_array(m):
        # try get_input_embeddings
        try:
            emb_mod = m.get_input_embeddings()
            if hasattr(emb_mod, 'weight'):
                return emb_mod.weight.detach().numpy()
        except Exception:
            pass
        # common: m.model.embed_tokens or m.wte
        core = getattr(m, 'model', m)
        if hasattr(core, 'embed_tokens') and hasattr(core.embed_tokens, 'weight'):
            return core.embed_tokens.weight.detach().numpy()
        if hasattr(core, 'wte') and hasattr(core.wte, 'weight'):
            return core.wte.weight.detach().numpy()
        # fallback: model has embedding attribute
        if hasattr(m, 'wte') and hasattr(m.wte, 'weight'):
            return m.wte.weight.detach().numpy()
        raise AttributeError('Unable to find embedding weight in model')

    emb = get_embedding_array(model)
    print(f"Embedding: {emb.shape}")
    packed, scales, zps = quantize_weight(emb, bits)
    if bits == 1:
        data['embedding.weight_1bit'] = packed
        data['embedding.weight_shape'] = np.array(emb.shape)
        data['embedding.weight_scales'] = scales
    else:
        data[f'embedding.weight_q{bits}'] = packed
        data['embedding.weight_shape'] = np.array(emb.shape)
        data[f'embedding.weight_q{bits}_scales'] = scales
        if zps is not None:
            data[f'embedding.weight_q{bits}_zero_point'] = zps
    
    # Quantize transformer layers: find layers flexibly (model.model.layers, model.transformer.h, model.h)
    def find_layers(mod):
        if hasattr(mod, 'layers'):
            return getattr(mod, 'layers')
        if hasattr(mod, 'h'):
            return getattr(mod, 'h')
        tr = getattr(mod, 'transformer', None)
        if tr is not None:
            if hasattr(tr, 'h'):
                return tr.h
            if hasattr(tr, 'blocks'):
                return tr.blocks
        return []

    core = getattr(model, 'model', model)
    layers_list = find_layers(core)
    for layer_idx in range(num_layers):
        try:
            layer = layers_list[layer_idx]
        except Exception:
            continue
        prefix = f'transformer.layers.{layer_idx}'
        
        # Self-attention projections - support different naming patterns
        self_attn = getattr(layer, 'self_attn', None) or getattr(layer, 'attn', None)
        q_weight = k_weight = v_weight = o_weight = None
        if self_attn is not None:
            q_mod = getattr(self_attn, 'q_proj', None)
            k_mod = getattr(self_attn, 'k_proj', None)
            v_mod = getattr(self_attn, 'v_proj', None)
            o_mod = getattr(self_attn, 'o_proj', None)
            if q_mod is not None and hasattr(q_mod, 'weight'):
                q_weight = q_mod.weight.detach()
            if k_mod is not None and hasattr(k_mod, 'weight'):
                k_weight = k_mod.weight.detach()
            if v_mod is not None and hasattr(v_mod, 'weight'):
                v_weight = v_mod.weight.detach()
            if o_mod is not None and hasattr(o_mod, 'weight'):
                o_weight = o_mod.weight.detach()
            # detect combined c_attn (GPT2 style) -> combined weights (numpy) will be handled by quantize
            if q_weight is None and hasattr(self_attn, 'c_attn'):
                combined = self_attn.c_attn.weight.detach().cpu().numpy()
                q_weight = combined
        else:
            q_mod = getattr(layer, 'q_proj', None)
            k_mod = getattr(layer, 'k_proj', None)
            v_mod = getattr(layer, 'v_proj', None)
            o_mod = getattr(layer, 'o_proj', None)
            if q_mod is not None and hasattr(q_mod, 'weight'):
                q_weight = q_mod.weight.detach()
            if k_mod is not None and hasattr(k_mod, 'weight'):
                k_weight = k_mod.weight.detach()
            if v_mod is not None and hasattr(v_mod, 'weight'):
                v_weight = v_mod.weight.detach()
            if o_mod is not None and hasattr(o_mod, 'weight'):
                o_weight = o_mod.weight.detach()
        
        # q_weight and o_weight can be numpy arrays or torch tensors
        def tensor_shape(x):
            import numpy as _np
            if hasattr(x, 'shape'):
                return tuple(x.shape)
            try:
                return tuple(_np.asarray(x).shape)
            except Exception:
                return None
        print(f"Layer {layer_idx}: Q={tensor_shape(q_weight)}, O={tensor_shape(o_weight)}")
        
        # Quantize Q, K, V, O
        for name, weight in [('q_proj', q_weight), ('k_proj', k_weight), ('v_proj', v_weight), ('o_proj', o_weight)]:
            if weight is None:
                continue
                packed, scales, zps = quantize_weight(weight, bits)
                if bits == 1:
                    data[f'{prefix}.self_attn.{name}.weight_1bit'] = packed
                    data[f'{prefix}.self_attn.{name}.weight_shape'] = np.array(weight.shape)
                    data[f'{prefix}.self_attn.{name}.weight_scales'] = scales
                else:
                    data[f'{prefix}.self_attn.{name}.weight_q{bits}'] = packed
                    data[f'{prefix}.self_attn.{name}.weight_shape'] = np.array(weight.shape)
                    data[f'{prefix}.self_attn.{name}.weight_q{bits}_scales'] = scales
                    if zps is not None:
                        data[f'{prefix}.self_attn.{name}.weight_q{bits}_zero_point'] = zps
        
        # FFN
        mlp_mod = getattr(layer, 'mlp', None) or getattr(layer, 'feed_forward', None) or getattr(layer, 'ffn', None)
        # Collect candidate MLP projections with robust naming
        def try_get_weight(mod, name):
            if mod is None:
                return None
            attr = getattr(mod, name, None)
            if attr is None:
                return None
            if hasattr(attr, 'weight'):
                return attr.weight.detach()
            return None

        gate_weight = try_get_weight(mlp_mod, 'gate_proj')
        if gate_weight is None:
            gate_weight = try_get_weight(layer, 'gate_proj')
        up_weight = try_get_weight(mlp_mod, 'up_proj')
        if up_weight is None:
            up_weight = try_get_weight(layer, 'up_proj')
        down_weight = try_get_weight(mlp_mod, 'down_proj')
        if down_weight is None:
            down_weight = try_get_weight(layer, 'down_proj')
        # GPT2 naming 'c_fc'/'c_proj'
        if gate_weight is None:
            gate_weight = try_get_weight(mlp_mod, 'c_fc')
            if gate_weight is None:
                gate_weight = try_get_weight(layer, 'c_fc')
        if up_weight is None:
            up_weight = try_get_weight(mlp_mod, 'c_proj')
            if up_weight is None:
                up_weight = try_get_weight(layer, 'c_proj')
        
        def get_shape_for_print(x):
            if x is None:
                return None
            try:
                return tuple(x.shape)
            except Exception:
                import numpy as _np
                return tuple(_np.asarray(x).shape)
        print(f"Layer {layer_idx}: FFN gate={get_shape_for_print(gate_weight)}, down={get_shape_for_print(down_weight)}")
        
        for name, weight in [('gate_proj', gate_weight), ('up_proj', up_weight), ('down_proj', down_weight)]:
            if weight is None:
                continue
            packed, scales, zps = quantize_weight(weight, bits)
            if bits == 1:
                data[f'{prefix}.mlp.{name}.weight_1bit'] = packed
                data[f'{prefix}.mlp.{name}.weight_shape'] = np.array(weight.shape)
                data[f'{prefix}.mlp.{name}.weight_scales'] = scales
            else:
                data[f'{prefix}.mlp.{name}.weight_q{bits}'] = packed
                data[f'{prefix}.mlp.{name}.weight_shape'] = np.array(weight.shape)
                data[f'{prefix}.mlp.{name}.weight_q{bits}_scales'] = scales
                if zps is not None:
                    data[f'{prefix}.mlp.{name}.weight_q{bits}_zero_point'] = zps
        
        # RMSNorm weights (keep as fp32)
        input_norm = None
        post_norm = None
        if hasattr(layer, 'input_layernorm') and getattr(layer.input_layernorm, 'weight', None) is not None:
            input_norm = layer.input_layernorm.weight.detach().numpy()
        if hasattr(layer, 'post_attention_layernorm') and getattr(layer.post_attention_layernorm, 'weight', None) is not None:
            post_norm = layer.post_attention_layernorm.weight.detach().numpy()
        data[f'{prefix}.input_layernorm.weight_fp32'] = input_norm
        data[f'{prefix}.post_attention_layernorm.weight_fp32'] = post_norm
    
    # Include LM head
    lm_head = model.lm_head.weight.detach().numpy()
    print(f"LM head: {lm_head.shape}")
    packed, scales, zps = quantize_weight(lm_head, bits)
    if bits == 1:
        data['lm_head.weight_1bit'] = packed
        data['lm_head.weight_shape'] = np.array(lm_head.shape)
        data['lm_head.weight_scales'] = scales
    else:
        data[f'lm_head.weight_q{bits}'] = packed
        data['lm_head.weight_shape'] = np.array(lm_head.shape)
        data[f'lm_head.weight_q{bits}_scales'] = scales
        if zps is not None:
            data[f'lm_head.weight_q{bits}_zero_point'] = zps
    
    # Save
    np.savez(output_npz, **data)
    print(f"Saved quantized weights to {output_npz}")
    
    return True

def quantize_weight(weight_tensor, bits=1):
    """Quantize a weight tensor to 1-bit."""
    import numpy as np
    w = weight_tensor.numpy() if hasattr(weight_tensor, 'numpy') else np.array(weight_tensor)
    if bits == 1:
        packed, scales, _ = binarize_weights(w, per_channel_axis=0)
        return packed, scales, None
    else:
        # integer per-row quantization returns zero-points too
        packed_int, scales_arr, zps = quantize_per_row(w, bits)
        return packed_int, scales_arr, zps

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
    parser.add_argument('--bits', type=int, choices=[1,2,4,8], default=1,
                       help='Quantize to this bit width (1=FPTQ binarized, 2/4/8 integer)')
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
        if not download_and_quantize_model(hf_name, str(npz_path), args.layers, args.bits):
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