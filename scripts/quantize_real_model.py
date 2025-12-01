#!/usr/bin/env python3
"""Load a real model from HuggingFace and quantize to 1-bit for testing.

This script:
1. Loads TinyLlama (or another small model) from HuggingFace
2. Extracts weight tensors
3. Applies 1-bit quantization using our pipeline
4. Saves to NPZ format for export

Usage:
    python scripts/quantize_real_model.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --layers 2
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantization import binarize_weights


def quantize_weight(weight_tensor):
    """Quantize a weight tensor to 1-bit. Returns (packed, scales)."""
    w = weight_tensor.numpy() if hasattr(weight_tensor, 'numpy') else np.array(weight_tensor)
    packed, scales, _ = binarize_weights(w, per_channel_axis=0)
    return packed, scales


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                        help='HuggingFace model name')
    parser.add_argument('--layers', type=int, default=2,
                        help='Number of transformer layers to quantize (for testing)')
    parser.add_argument('--output', type=str, default='real_model_quantized_1bit.npz',
                        help='Output NPZ path')
    parser.add_argument('--include-embedding', action='store_true',
                        help='Include embedding layer (large)')
    parser.add_argument('--include-lm-head', action='store_true',
                        help='Include LM head layer (large)')
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    
    from transformers import AutoModelForCausalLM, AutoConfig
    
    config = AutoConfig.from_pretrained(args.model)
    print(f"Config: hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}")
    
    # Load model with float32 weights on CPU
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map='cpu',
        low_cpu_mem_usage=True
    )
    
    print(f"Model loaded. Quantizing first {args.layers} layers...")
    
    data = {}
    
    # Store architecture info
    data['_config_hidden_size'] = np.array([config.hidden_size])
    data['_config_intermediate_size'] = np.array([config.intermediate_size])
    data['_config_num_layers'] = np.array([min(args.layers, config.num_hidden_layers)])
    data['_config_num_heads'] = np.array([config.num_attention_heads])
    data['_config_num_kv_heads'] = np.array([config.num_key_value_heads])
    data['_config_vocab_size'] = np.array([config.vocab_size])
    data['_config_rope_theta'] = np.array([getattr(config, 'rope_theta', 10000.0)])
    
    # Optionally include embedding
    if args.include_embedding:
        emb = model.model.embed_tokens.weight.detach().numpy()
        print(f"Embedding: {emb.shape}")
        packed, scales = quantize_weight(emb)
        data['embedding.weight_1bit'] = packed
        data['embedding.weight_shape'] = np.array(emb.shape)
        data['embedding.weight_scales'] = scales
    
    # Quantize transformer layers
    for layer_idx in range(min(args.layers, config.num_hidden_layers)):
        layer = model.model.layers[layer_idx]
        prefix = f'transformer.layers.{layer_idx}'
        
        # Self-attention Q, K, V projections
        q_weight = layer.self_attn.q_proj.weight.detach()
        k_weight = layer.self_attn.k_proj.weight.detach()
        v_weight = layer.self_attn.v_proj.weight.detach()
        o_weight = layer.self_attn.o_proj.weight.detach()
        
        print(f"Layer {layer_idx}: Q={tuple(q_weight.shape)}, K={tuple(k_weight.shape)}, "
              f"V={tuple(v_weight.shape)}, O={tuple(o_weight.shape)}")
        
        # Quantize Q
        packed, scales = quantize_weight(q_weight)
        data[f'{prefix}.self_attn.q_proj.weight_1bit'] = packed
        data[f'{prefix}.self_attn.q_proj.weight_shape'] = np.array(q_weight.shape)
        data[f'{prefix}.self_attn.q_proj.weight_scales'] = scales
        
        # Quantize K
        packed, scales = quantize_weight(k_weight)
        data[f'{prefix}.self_attn.k_proj.weight_1bit'] = packed
        data[f'{prefix}.self_attn.k_proj.weight_shape'] = np.array(k_weight.shape)
        data[f'{prefix}.self_attn.k_proj.weight_scales'] = scales
        
        # Quantize V
        packed, scales = quantize_weight(v_weight)
        data[f'{prefix}.self_attn.v_proj.weight_1bit'] = packed
        data[f'{prefix}.self_attn.v_proj.weight_shape'] = np.array(v_weight.shape)
        data[f'{prefix}.self_attn.v_proj.weight_scales'] = scales
        
        # Quantize O
        packed, scales = quantize_weight(o_weight)
        data[f'{prefix}.self_attn.o_proj.weight_1bit'] = packed
        data[f'{prefix}.self_attn.o_proj.weight_shape'] = np.array(o_weight.shape)
        data[f'{prefix}.self_attn.o_proj.weight_scales'] = scales
        
        # FFN: gate_proj, up_proj, down_proj (SwiGLU)
        gate_weight = layer.mlp.gate_proj.weight.detach()
        up_weight = layer.mlp.up_proj.weight.detach()
        down_weight = layer.mlp.down_proj.weight.detach()
        
        print(f"Layer {layer_idx}: gate={tuple(gate_weight.shape)}, up={tuple(up_weight.shape)}, "
              f"down={tuple(down_weight.shape)}")
        
        # Quantize gate
        packed, scales = quantize_weight(gate_weight)
        data[f'{prefix}.mlp.gate_proj.weight_1bit'] = packed
        data[f'{prefix}.mlp.gate_proj.weight_shape'] = np.array(gate_weight.shape)
        data[f'{prefix}.mlp.gate_proj.weight_scales'] = scales
        
        # Quantize up
        packed, scales = quantize_weight(up_weight)
        data[f'{prefix}.mlp.up_proj.weight_1bit'] = packed
        data[f'{prefix}.mlp.up_proj.weight_shape'] = np.array(up_weight.shape)
        data[f'{prefix}.mlp.up_proj.weight_scales'] = scales
        
        # Quantize down
        packed, scales = quantize_weight(down_weight)
        data[f'{prefix}.mlp.down_proj.weight_1bit'] = packed
        data[f'{prefix}.mlp.down_proj.weight_shape'] = np.array(down_weight.shape)
        data[f'{prefix}.mlp.down_proj.weight_scales'] = scales
        
        # RMSNorm weights (keep as fp32)
        input_norm = layer.input_layernorm.weight.detach().numpy()
        post_norm = layer.post_attention_layernorm.weight.detach().numpy()
        data[f'{prefix}.input_layernorm.weight'] = input_norm
        data[f'{prefix}.post_attention_layernorm.weight'] = post_norm
    
    # Optionally include LM head
    if args.include_lm_head:
        lm_head = model.lm_head.weight.detach().numpy()
        print(f"LM head: {lm_head.shape}")
        packed, scales = quantize_weight(lm_head)
        data['lm_head.weight_1bit'] = packed
        data['lm_head.weight_shape'] = np.array(lm_head.shape)
        data['lm_head.weight_scales'] = scales
    
    # Save
    np.savez(args.output, **data)
    print(f"Saved quantized weights to {args.output}")
    
    # Print summary
    total_params = sum(v.size for k, v in data.items() if not k.startswith('_config'))
    print(f"Total elements in NPZ: {total_params:,}")


if __name__ == '__main__':
    main()
