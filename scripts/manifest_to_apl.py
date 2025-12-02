#!/usr/bin/env python3
"""Generate a small APL snippet (`apl/generated_manifest.apl`) from a v2 manifest JSON.

This translates the `model`, `architecture`, and `weights` sections into simple APL
assignments so `llama.apl` or examples can be configured from an exported manifest.

Example output (APL):
  MANIFEST_FILE ← 'student_quantized_manifest.json'
  MODEL_FAMILY ← 'mistral'
  ARCHITECTURE ← 'num_layers 32 hidden_size 4096 ...'  ⍝ human-friendly comment line
  WEIGHT_NAMES ← 'fc.weight' 'embedding.weight' 'transformer.layers.0.linear1.weight'
  fc_weight_packed ← 'fc.weight_1bit.bin'
  fc_weight_scales_txt ← 'fc.weight_scales.txt'
  fc_weight_shape ← 1000 64

This is a generator helpful for demonstrations and APL-based runtime wiring.
"""
import argparse
import json
import re
from pathlib import Path


def safe_apl_name(name: str) -> str:
    # Replace characters (., /, -) -> underscore and remove other non-alphanum
    s = re.sub(r'[^0-9a-zA-Z]', '_', name)
    # APL symbols can't start with digits; prefix if needed
    if re.match(r'^[0-9]', s):
        s = '_' + s
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, default='student_quantized_manifest.json')
    parser.add_argument('--out_apl', type=str, default='apl/generated_manifest.apl')
    args = parser.parse_args()

    mp = Path(args.manifest)
    if not mp.exists():
        print('Manifest not found:', args.manifest)
        return 1
    j = json.loads(mp.read_text(encoding='utf-8'))
    outp = []
    
    # Basic declarations
    # Use POSIX-style path separators for consistency (works well with WSL and many shells)
    outp.append("MANIFEST_FILE ← '{}'".format(mp.as_posix()))
    model = j.get('model', {})
    primary_family = model.get('primary_family', '')
    outp.append("MODEL_FAMILY ← '{}'".format(primary_family))
    
    # Architecture metadata as APL variables
    arch = j.get('architecture', {})
    outp.append("")
    outp.append("⍝ === Architecture Metadata ===")
    
    # Core dimensions
    hidden_size = arch.get('hidden_size', 0)
    intermediate_size = arch.get('intermediate_size', 0)
    num_layers = arch.get('num_layers', 0)
    vocab_size = arch.get('vocab_size', 0)
    context_length = arch.get('context_length', 0)
    
    outp.append(f"HIDDEN_SIZE ← {hidden_size}")
    outp.append(f"INTERMEDIATE_SIZE ← {intermediate_size}")
    outp.append(f"NUM_LAYERS ← {num_layers}")
    outp.append(f"VOCAB_SIZE ← {vocab_size}")
    outp.append(f"CONTEXT_LENGTH ← {context_length}")
    
    # Attention config
    attention = arch.get('attention', {})
    num_heads = attention.get('num_heads', 0)
    kv_groups = attention.get('kv_groups', num_heads)  # default to full attention
    head_dim = attention.get('head_dim', hidden_size // num_heads if num_heads else 0)
    attention_variant = attention.get('variant', 'full')
    window_size = attention.get('window_size', 0)
    
    outp.append(f"NUM_HEADS ← {num_heads}")
    outp.append(f"KV_GROUPS ← {kv_groups}")
    outp.append(f"HEAD_DIM ← {head_dim}")
    outp.append(f"ATTENTION_VARIANT ← '{attention_variant}'")
    if window_size:
        outp.append(f"WINDOW_SIZE ← {window_size}")
    
    # Activation and normalization
    activation = arch.get('activation', 'relu')
    norm_type = arch.get('norm', 'layernorm')
    outp.append(f"ACTIVATION ← '{activation}'")
    outp.append(f"NORM_TYPE ← '{norm_type}'")
    
    # RoPE config
    rope = arch.get('rope', {})
    rope_base = rope.get('base_theta', 10000.0)
    rope_scale = rope.get('scale', 1.0)
    outp.append(f"ROPE_BASE ← {rope_base}")
    outp.append(f"ROPE_SCALE ← {rope_scale}")
    
    outp.append("")
    outp.append("⍝ === Weight Names ===")
    # weight list
    weights = j.get('weights', {})
    names = sorted(weights.keys())
    names_escaped = ' '.join([f"'{n}'" for n in names])
    outp.append("WEIGHT_NAMES ← {}".format(names_escaped))

    outp.append("")
    outp.append("⍝ === Per-Weight Paths ===")
    # per-weight assignments
    for key, entry in weights.items():
        var = safe_apl_name(key)
        packed = entry.get('packed', None)
        scales_txt = entry.get('scales_txt') or entry.get('scales') or None
        shape = entry.get('shape') or None
        if packed:
            # Normalize the path to POSIX so APL demos are more portable on Windows/WSL
            outp.append(f"{var}_packed ← '{Path(packed).as_posix()}'")
        if scales_txt:
            outp.append(f"{var}_scales_txt ← '{Path(scales_txt).as_posix()}'")
        # If an fp32 (dequantized) path exists, include it as well.
        fp32 = entry.get('fp32') or entry.get('fp32_path') or entry.get('fp32_npy')
        if fp32:
            outp.append(f"{var}_fp32 ← '{Path(fp32).as_posix()}'")
        if shape:
            outp.append(f"{var}_shape ← {shape[0]} {shape[1]}")

    # write file
    outp_text = "\n".join(outp) + "\n"
    out_apl = Path(args.out_apl)
    out_apl.parent.mkdir(parents=True, exist_ok=True)
    out_apl.write_text(outp_text, encoding='utf-8')
    print('Wrote', out_apl)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
