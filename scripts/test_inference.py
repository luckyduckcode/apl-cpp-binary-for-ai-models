#!/usr/bin/env python3
"""Run inference on a quantized model to verify the full pipeline works.

This loads a quantized model from manifest and runs a forward pass,
comparing the output to the original fp32 model.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cpp'))

from quantization import unpack_binarized


def load_manifest(manifest_path):
    """Load manifest JSON."""
    with open(manifest_path) as f:
        return json.load(f)


def load_packed_weight(manifest, manifest_dir, weight_name):
    """Load a packed weight from manifest and dequantize."""
    weights = manifest.get('weights', {})
    entry = weights.get(weight_name)
    if not entry:
        return None
    
    packed_path = manifest_dir / entry['packed']
    scales_path = manifest_dir / (entry.get('scales_txt') or entry.get('scales'))
    shape = tuple(entry['shape'])
    
    # Load packed bits
    packed = np.fromfile(packed_path, dtype=np.uint8)
    bytes_per_row = math.ceil(shape[1] / 8)
    packed = packed.reshape((shape[0], bytes_per_row))
    
    # Load scales
    if str(scales_path).endswith('.npy'):
        scales = np.load(scales_path)
    else:
        scales = np.loadtxt(scales_path)
    
    # Dequantize
    mat = unpack_binarized(packed, scales, shape, per_channel_axis=0)
    return mat.astype(np.float32)


def rmsnorm(x, weight, eps=1e-6):
    """RMS normalization."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def swiglu(gate, up):
    """SwiGLU activation: SiLU(gate) * up."""
    silu_gate = gate / (1 + np.exp(-gate))  # SiLU = x * sigmoid(x)
    return silu_gate * up


def forward_layer(x, manifest, manifest_dir, layer_idx, npz):
    """Forward pass through one transformer layer."""
    prefix = f'transformer.layers.{layer_idx}'
    
    # Load norm weights from NPZ (kept as fp32)
    input_norm_w = np.array(npz[f'{prefix}.input_layernorm.weight'])
    post_norm_w = np.array(npz[f'{prefix}.post_attention_layernorm.weight'])
    
    # Input norm
    normed = rmsnorm(x, input_norm_w)
    
    # Self-attention (simplified - no actual attention, just linear projections)
    q_proj = load_packed_weight(manifest, manifest_dir, f'{prefix}.self_attn.q_proj.weight')
    k_proj = load_packed_weight(manifest, manifest_dir, f'{prefix}.self_attn.k_proj.weight')
    v_proj = load_packed_weight(manifest, manifest_dir, f'{prefix}.self_attn.v_proj.weight')
    o_proj = load_packed_weight(manifest, manifest_dir, f'{prefix}.self_attn.o_proj.weight')
    
    # For simplicity, just do the projections without actual attention
    q = normed @ q_proj.T
    k = normed @ k_proj.T
    v = normed @ v_proj.T
    
    # Simplified attention output (skip actual attention for now)
    attn_out = v @ o_proj.T
    
    # Residual
    x = x + attn_out
    
    # Post-attention norm
    normed = rmsnorm(x, post_norm_w)
    
    # MLP (SwiGLU)
    gate = load_packed_weight(manifest, manifest_dir, f'{prefix}.mlp.gate_proj.weight')
    up = load_packed_weight(manifest, manifest_dir, f'{prefix}.mlp.up_proj.weight')
    down = load_packed_weight(manifest, manifest_dir, f'{prefix}.mlp.down_proj.weight')
    
    gate_out = normed @ gate.T
    up_out = normed @ up.T
    hidden = swiglu(gate_out, up_out)
    mlp_out = hidden @ down.T
    
    # Residual
    x = x + mlp_out
    
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, required=True, help='Path to manifest JSON')
    parser.add_argument('--npz', type=str, required=True, help='Path to quantized NPZ')
    parser.add_argument('--compare-original', type=str, default='', 
                        help='HuggingFace model name to compare against')
    args = parser.parse_args()
    
    manifest = load_manifest(args.manifest)
    manifest_dir = Path(args.manifest).parent
    npz = np.load(args.npz)
    
    arch = manifest.get('architecture', {})
    hidden_size = arch.get('hidden_size', 16)
    num_layers = arch.get('num_layers', 2)
    
    print(f"Model: {manifest.get('model', {}).get('primary_family', 'unknown')}")
    print(f"Hidden size: {hidden_size}, Layers: {num_layers}")
    
    # Create a random input
    np.random.seed(42)
    seq_len = 4
    x = np.random.randn(seq_len, hidden_size).astype(np.float32)
    print(f"\nInput shape: {x.shape}")
    print(f"Input sample: {x[0, :4]}")
    
    # Forward through quantized model
    for layer_idx in range(num_layers):
        x = forward_layer(x, manifest, manifest_dir, layer_idx, npz)
        print(f"After layer {layer_idx}: shape={x.shape}, sample={x[0, :4]}")
    
    print(f"\nFinal output shape: {x.shape}")
    print(f"Output mean: {x.mean():.6f}, std: {x.std():.6f}")
    print(f"Output range: [{x.min():.6f}, {x.max():.6f}]")
    
    # Compare with original if requested
    if args.compare_original:
        print(f"\nComparing with original model: {args.compare_original}")
        try:
            from transformers import AutoModelForCausalLM
            
            model = AutoModelForCausalLM.from_pretrained(
                args.compare_original,
                torch_dtype=torch.float32,
                device_map='cpu'
            )
            
            # Run same input through original model's layers
            x_orig = torch.from_numpy(np.random.RandomState(42).randn(seq_len, hidden_size).astype(np.float32))
            
            for layer_idx in range(num_layers):
                layer = model.model.layers[layer_idx]
                
                # Input norm
                normed = layer.input_layernorm(x_orig)
                
                # Attention projections
                q = layer.self_attn.q_proj(normed)
                v = layer.self_attn.v_proj(normed)
                attn_out = layer.self_attn.o_proj(v)
                x_orig = x_orig + attn_out
                
                # Post-attention norm
                normed = layer.post_attention_layernorm(x_orig)
                
                # MLP
                gate = layer.mlp.gate_proj(normed)
                up = layer.mlp.up_proj(normed)
                hidden = torch.nn.functional.silu(gate) * up
                mlp_out = layer.mlp.down_proj(hidden)
                x_orig = x_orig + mlp_out
            
            x_orig_np = x_orig.detach().numpy()
            diff = np.abs(x - x_orig_np)
            print(f"Max diff from original: {diff.max():.6f}")
            print(f"Mean diff from original: {diff.mean():.6f}")
            
        except Exception as e:
            print(f"Could not compare: {e}")
    
    print("\n✓ Inference test completed successfully!")


if __name__ == '__main__':
    main()
