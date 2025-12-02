#!/usr/bin/env python3
"""Export a Hugging Face model to a 1-bit FPTQ NPZ and optionally export a manifest for APL.

This script follows the same export format as `easy_run.py` quantization step but it's self-contained
for using directly from the CLI.

Example:
  python scripts/export_model_1bit.py --hf-model tinyllama/TinyLlama-1.1B-Chat-v1.0 --out models/tinyllama_1bit.npz --out-manifest models/tinyllama_manifest.json --model-family llama --num-layers 22

"""
import argparse
from pathlib import Path
import numpy as np
import os
import sys
import subprocess

try:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
except ImportError:
    print('Missing dependencies: pip install -r requirements.txt')
    raise

from quantization import binarize_weights, quantize_weights, quantize_per_row


def write_packed(data: dict, out_npz: Path):
    # Save data as NPZ
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_npz), **data)


def quantize_model_to_1bit(model, num_layers=None):
    """Quantize model to requested bit width.
    This function keeps a `bits` variable (default=1) and produces appropriate keys.
    For bits==1: keys are `<prefix>.weight_1bit` with packed bytes and scales.
    For bits>1: keys are `<prefix>.weight_q{bits}` with integer array and per-row scales and zero_point arrays.
    """
    bits = getattr(model, 'bits', 1) if hasattr(model, 'bits') else 1
    data = {}
    # Try to find the core model with .model or top-level
    core = getattr(model, 'model', model)

    # Embedding weights
    emb = None
    if hasattr(core, 'get_input_embeddings'):
        try:
            emb_mod = core.get_input_embeddings()
            if hasattr(emb_mod, 'weight'):
                emb = emb_mod.weight.detach().cpu().numpy()
        except Exception:
            emb = None
    else:
        emb_mod = getattr(core, 'embed_tokens', None) or getattr(core, 'wte', None)
        if emb_mod is not None and hasattr(emb_mod, 'weight'):
            emb = emb_mod.weight.detach().cpu().numpy()

    if emb is not None:
        if bits == 1:
            packed, scales, _ = binarize_weights(emb, per_channel_axis=0)
            data['embedding.weight_1bit'] = packed
            data['embedding.weight_shape'] = np.array(emb.shape)
            data['embedding.weight_scales'] = scales
        else:
            # per-row quantization to integers
            packed_int, scales_arr, zps = quantize_per_row(emb, bits)
            data[f'embedding.weight_q{bits}'] = packed_int
            data['embedding.weight_shape'] = np.array(emb.shape)
            data[f'embedding.weight_q{bits}_scales'] = scales_arr
            data[f'embedding.weight_q{bits}_zero_point'] = zps
        packed, scales, _ = binarize_weights(emb, per_channel_axis=0)
        data['embedding.weight_1bit'] = packed
        data['embedding.weight_shape'] = np.array(emb.shape)
        data['embedding.weight_scales'] = scales

    # Layers
    def find_layers(mod):
        # Common layer collection names
        if hasattr(mod, 'layers'):
            return getattr(mod, 'layers')
        if hasattr(mod, 'h'):
            return getattr(mod, 'h')
        if hasattr(mod, 'transformer'):
            tr = getattr(mod, 'transformer')
            if hasattr(tr, 'h'):
                return tr.h
            if hasattr(tr, 'blocks'):
                return tr.blocks
        if hasattr(mod, 'encoder') and hasattr(mod.encoder, 'layer'):
            return mod.encoder.layer
        # fallback to empty
        return []

    layers_list = find_layers(core)
    # Fetch number of layers
    if num_layers is None:
        num_layers = len(layers_list)

    for layer_idx in range(num_layers):
        try:
            layer = layers_list[layer_idx]
        except Exception:
            continue

        prefix = f'transformer.layers.{layer_idx}'

        # Self-attn linear weights; handle different attribute names
        # Support different layer naming: self_attn, attn, or attn modules
        self_attn = getattr(layer, 'self_attn', None) or getattr(layer, 'attn', None)
        q = k = v = o = None
        if self_attn is not None:
            q = getattr(self_attn, 'q_proj', None)
            k = getattr(self_attn, 'k_proj', None)
            v = getattr(self_attn, 'v_proj', None)
            o = getattr(self_attn, 'o_proj', None)
            # GPT2-style combined projection `c_attn` (concat qkv)
            if q is None and hasattr(self_attn, 'c_attn'):
                # we'll write combined in_proj as a fallback
                arr = self_attn.c_attn.weight.detach().cpu().numpy()
                if bits == 1:
                    packed, scales, _ = binarize_weights(arr, per_channel_axis=0)
                    key = f'{prefix}.self_attn.in_proj_weight_1bit'
                    data[key] = packed
                    data[f'{prefix}.self_attn.in_proj_weight_shape'] = np.array(arr.shape)
                    data[f'{prefix}.self_attn.in_proj_weight_scales'] = scales
                else:
                    packed_int, scales_arr, zps = quantize_per_row(arr, bits)
                    key = f'{prefix}.self_attn.in_proj_weight_q{bits}'
                    data[key] = packed_int
                    data[f'{prefix}.self_attn.in_proj_weight_shape'] = np.array(arr.shape)
                    data[f'{prefix}.self_attn.in_proj_weight_q{bits}_scales'] = scales_arr
                    data[f'{prefix}.self_attn.in_proj_weight_q{bits}_zero_point'] = zps
        for name, weight in [('q_proj', q), ('k_proj', k), ('v_proj', v), ('o_proj', o)]:
            if weight is None:
                continue
            arr = weight.weight.detach().cpu().numpy()
            if bits == 1:
                packed, scales, _ = binarize_weights(arr, per_channel_axis=0)
                key = f'{prefix}.self_attn.{name}.weight_1bit'
                data[key] = packed
                data[f'{prefix}.self_attn.{name}.weight_shape'] = np.array(arr.shape)
                data[f'{prefix}.self_attn.{name}.weight_scales'] = scales
            else:
                packed_int, scales_arr, zps = quantize_per_row(arr, bits)
                key = f'{prefix}.self_attn.{name}.weight_q{bits}'
                data[key] = packed_int
                data[f'{prefix}.self_attn.{name}.weight_shape'] = np.array(arr.shape)
                data[f'{prefix}.self_attn.{name}.weight_q{bits}_scales'] = scales_arr
                data[f'{prefix}.self_attn.{name}.weight_q{bits}_zero_point'] = zps

        # FFN (sometimes gated names like gate_proj/up_proj/down_proj)
        mlp_mod = getattr(layer, 'mlp', None) or getattr(layer, 'feed_forward', None) or getattr(layer, 'ffn', None)
        for name in ['gate_proj', 'up_proj', 'down_proj', 'linear1', 'linear2', 'c_fc', 'c_proj']:
            candidate = getattr(mlp_mod, name, None) if mlp_mod is not None else None
            if candidate is None:
                candidate = getattr(layer, name, None)
            if candidate is None:
                continue
            arr = getattr(candidate, 'weight', None)
            if arr is None:
                continue
            arr = arr.detach().cpu().numpy()
            if bits == 1:
                packed, scales, _ = binarize_weights(arr, per_channel_axis=0)
                key = f'{prefix}.mlp.{name}.weight_1bit'
                data[key] = packed
                data[f'{prefix}.mlp.{name}.weight_shape'] = np.array(arr.shape)
                data[f'{prefix}.mlp.{name}.weight_scales'] = scales
            else:
                packed_int, scales_arr, zps = quantize_per_row(arr, bits)
                key = f'{prefix}.mlp.{name}.weight_q{bits}'
                data[key] = packed_int
                data[f'{prefix}.mlp.{name}.weight_shape'] = np.array(arr.shape)
                data[f'{prefix}.mlp.{name}.weight_q{bits}_scales'] = scales_arr
                data[f'{prefix}.mlp.{name}.weight_q{bits}_zero_point'] = zps

        # RMSNorm weights (FP32)
        inorm = None
        pnorm = None
        if hasattr(layer, 'input_layernorm') and getattr(layer.input_layernorm, 'weight', None) is not None:
            inorm = layer.input_layernorm.weight.detach().cpu().numpy()
            data[f'{prefix}.input_layernorm.weight_fp32'] = inorm
        if hasattr(layer, 'post_attention_layernorm') and getattr(layer.post_attention_layernorm, 'weight', None) is not None:
            pnorm = layer.post_attention_layernorm.weight.detach().cpu().numpy()
            data[f'{prefix}.post_attention_layernorm.weight_fp32'] = pnorm

    # LM head
    try:
        lm_head = model.lm_head.weight.detach().cpu().numpy()
        if bits == 1:
            packed, scales, _ = binarize_weights(lm_head, per_channel_axis=0)
            data['lm_head.weight_1bit'] = packed
            data['lm_head.weight_shape'] = np.array(lm_head.shape)
            data['lm_head.weight_scales'] = scales
        else:
            packed_int, scales_arr, zps = quantize_per_row(lm_head, bits)
            data[f'lm_head.weight_q{bits}'] = packed_int
            data['lm_head.weight_shape'] = np.array(lm_head.shape)
            data[f'lm_head.weight_q{bits}_scales'] = scales_arr
            data[f'lm_head.weight_q{bits}_zero_point'] = zps
    except Exception:
        pass

    return data


# quantize_per_row is now defined in quantization.py, import above


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf-model', type=str, required=True, help='Hugging Face model repo or local dir')
    parser.add_argument('--out', type=str, required=True, help='Output NPZ path')
    parser.add_argument('--out-manifest', type=str, default=None, help='If set, call export_quantized_for_apl.py after NPZ generation')
    parser.add_argument('--model-family', type=str, default='llama', help='Model family (llama, mistral, gemma, etc)')
    parser.add_argument('--bits', type=int, default=1, choices=[1,2,4,8], help='Quantization bit-width (1=FPTQ binarized, 2/4/8 quantized uint)')
    parser.add_argument('--num-heads', type=int, default=None, help='Number of attention heads (optional; passed to exporter)')
    parser.add_argument('--hidden-size', type=int, default=None, help='Hidden size (optional; passed to exporter)')
    parser.add_argument('--context-length', type=int, default=None, help='Context length; exported in manifest')
    parser.add_argument('--num-layers', type=int, default=None)
    args = parser.parse_args()
    bits = int(getattr(args, 'bits', 1))

    config = AutoConfig.from_pretrained(args.hf_model)
    print('Loading model:', args.hf_model)
    model = AutoModelForCausalLM.from_pretrained(args.hf_model, torch_dtype=torch.float32, low_cpu_mem_usage=True)

    # attach selected bits to model object to propagate into quantize_model_to_1bit
    setattr(model, 'bits', bits)
    data = quantize_model_to_1bit(model, num_layers=args.num_layers)

    out_npz = Path(args.out)
    write_packed(data, out_npz)
    print('Saved quantized NPZ to', out_npz)

    if args.out_manifest:
        print('Exporting manifest...')
        cmd = ['python', 'export_quantized_for_apl.py', '--npz', str(out_npz), '--out_manifest', str(args.out_manifest), '--model-family', args.model_family]
        if args.num_heads:
            cmd.extend(['--num-heads', str(args.num_heads)])
        if args.hidden_size:
            cmd.extend(['--hidden-size', str(args.hidden_size)])
        if args.context_length:
            cmd.extend(['--context-length', str(args.context_length)])
        subprocess.check_call(cmd)
        print('Exported manifest to', args.out_manifest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
