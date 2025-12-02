import argparse
import json
import re
from pathlib import Path

import numpy as np


def parse_list_arg(value: str):
    if not value:
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def dedupe(seq):
    seen = set()
    result = []
    for item in seq:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def load_json_file(path: str):
    with open(path, 'r', encoding='utf-8') as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}")
    return data


def infer_num_layers(weight_names):
    pattern = re.compile(r"transformer\.layers\.(\d+)\.")
    indices = set()
    for name in weight_names:
        match = pattern.search(name)
        if match:
            indices.add(int(match.group(1)))
    return (max(indices) + 1) if indices else None


def infer_hidden_size(weights):
    candidates = [
        'transformer.layers.0.self_attn.in_proj_weight',
        'transformer.layers.0.self_attn.out_proj.weight',
        'transformer.layers.0.linear2.weight',
        'fc.weight'
    ]
    for cand in candidates:
        entry = weights.get(cand)
        if entry and entry.get('shape'):
            shape = entry['shape']
            if len(shape) == 2:
                return int(shape[1])
    return None


def infer_intermediate_size(weights):
    entry = weights.get('transformer.layers.0.linear1.weight')
    if entry and entry.get('shape'):
        shape = entry['shape']
        if len(shape) == 2:
            return int(shape[0])
    return None


def infer_vocab_size(weights):
    entry = weights.get('embedding.weight')
    if entry and entry.get('shape'):
        shape = entry['shape']
        if len(shape) == 2:
            return int(shape[0])
    return None


def infer_num_heads(hidden_size, explicit_heads):
    if explicit_heads:
        return int(explicit_heads)
    if hidden_size and hidden_size % 64 == 0 and hidden_size // 64 > 1:
        return hidden_size // 64
    return None


def infer_scale_axis(npz_files, npz_obj, name):
    axis_key = f"{name}_pc_axis"
    if axis_key in npz_files:
        axis_val = npz_obj[axis_key]
        if np.ndim(axis_val) == 0:
            return int(axis_val)
        axis_list = np.asarray(axis_val).tolist()
        if isinstance(axis_list, list) and axis_list:
            return int(axis_list[0])
    return 0


def axis_name(axis_index):
    return 'out_features' if axis_index in (0, '0') else 'in_features'


def drop_none(obj):
    if isinstance(obj, dict):
        filtered = {}
        for key, value in obj.items():
            cleaned = drop_none(value)
            if cleaned is not None:
                filtered[key] = cleaned
        return filtered
    if isinstance(obj, list):
        cleaned_list = [drop_none(item) for item in obj]
        return [item for item in cleaned_list if item is not None]
    return obj


parser = argparse.ArgumentParser()
parser.add_argument('--npz', type=str, default='student_quantized_1bit.npz', help='Path to quantized NPZ archive produced by quantization.py')
parser.add_argument('--out_manifest', type=str, default='student_quantized_manifest.json', help='Manifest output JSON path')
parser.add_argument('--nhead', type=int, default=None, help='(Deprecated) number of attention heads to include in per-layer metadata')
parser.add_argument('--num-heads', type=int, default=None, help='Number of attention heads for the model; overrides --nhead when set')
parser.add_argument('--kv-groups', type=int, default=None, help='KV group count for GQA/MQA style attention')
parser.add_argument('--model-family', type=str, default='llama', help='Primary model family label (e.g., llama, mistral, code-llama, gemma, qwen)')
parser.add_argument('--target-families', type=str, default='', help='Comma-separated additional model families supported by this export (e.g., "mistral,qwen")')
parser.add_argument('--context-length', type=int, default=None, help='Context length supported by this export (e.g., 4096, 32768, 131072)')
parser.add_argument('--num-layers', type=int, default=None, help='Override transformer layer count; inferred when omitted')
parser.add_argument('--hidden-size', type=int, default=None, help='Override hidden size (d_model); inferred when omitted')
parser.add_argument('--intermediate-size', type=int, default=None, help='Override FFN intermediate size; inferred when omitted')
parser.add_argument('--vocab-size', type=int, default=None, help='Vocabulary size; inferred from embedding weights when omitted')
parser.add_argument('--rope-base', type=float, default=None, help='ROPE base theta value (e.g., 10000, 500000)')
parser.add_argument('--rope-scale', type=float, default=None, help='Global RoPE scaling factor (e.g., YaRN factor)')
parser.add_argument('--rope-scaling-json', type=str, default=None, help='Path to JSON file describing complex RoPE scaling (per-head factors, yarn metadata, etc.)')
parser.add_argument('--attention-variant', type=str, default='full', help='Attention style: full, gqa, mqa, sliding-window, moe, alibi, etc.')
parser.add_argument('--window-size', type=int, default=None, help='Sliding window span when using local attention (Mistral-style)')
parser.add_argument('--activation', type=str, default='relu', help='FFN activation (relu, swiglu, geglu, gemma, etc.)')
parser.add_argument('--norm-type', type=str, default='layernorm', help='Normalization type (layernorm, rmsnorm, pre_rmsnorm, etc.)')
parser.add_argument('--architecture-json', type=str, default=None, help='Optional JSON file to seed architecture metadata')
parser.add_argument('--model-notes', type=str, default='', help='Free-form notes stored with manifest metadata')
args = parser.parse_args()

npz_path = Path(args.npz)
npz = np.load(npz_path, allow_pickle=True)
npz_files = set(npz.files)

weights_section = {}
bits_used = set()

global_heads = args.num_heads or args.nhead

for key in npz.files:
    if key.endswith('_1bit'):
        name = key[:-5]
        packed = npz[key]
        packed_path = f"{name}_1bit.bin"
        Path(packed_path).parent.mkdir(parents=True, exist_ok=True)
        packed.tofile(packed_path)

        shape_key = f"{name}_shape"
        if shape_key not in npz_files:
            raise KeyError(f"Missing shape entry {shape_key} in NPZ")
        shape = list(map(int, np.asarray(npz[shape_key]).tolist()))

        scales_key = f"{name}_scales"
        if scales_key not in npz_files:
            raise KeyError(f"Missing scales entry {scales_key} in NPZ")
        scales_array = np.asarray(npz[scales_key])
        scales_npy = f"{name}_scales.npy"
        np.save(scales_npy, scales_array)
        scales_txt = f"{name}_scales.txt"
        np.savetxt(scales_txt, scales_array)

        scale_axis_index = infer_scale_axis(npz_files, npz, name)
        entry = {
            'packed': packed_path,
            'shape': shape,
            'scales': scales_npy,
            'scales_txt': scales_txt,
            'bit_width': 1,
            'scale_axis': {
                'index': int(scale_axis_index),
                'name': axis_name(scale_axis_index)
            },
            'scale_dtype': 'float32',
            'packed_format': 'numpy.packbits(msb_first,row-major)'
        }
        entry['quantization'] = {
            'type': 'binarized',
            'bits': 1,
            'scales': scales_npy
        }
        bits_used.add(1)

        lname = name.lower()
        if any(tag in lname for tag in ['in_proj', 'self_attn', 'wq', 'wk', 'wv', 'out_proj']):
            inferred_heads = infer_num_heads(shape[1] if len(shape) == 2 else None, global_heads)
            if inferred_heads:
                entry['NHEADS'] = int(inferred_heads)
                entry['num_attention_heads'] = int(inferred_heads)

        weights_section[name] = entry

    elif key.endswith('_fp32'):
        name = key[:-5]
        fp_path = f"{name}_fp32.npy"
        np.save(fp_path, npz[key])
        entry = {
            'fp32': fp_path,
            'dtype': 'fp32'
        }
        weights_section[name] = entry
    else:
        # check for keys like <name>_q{bits}
        m = re.match(r'(.+)_q(\d+)$', key)
        if m:
            base = m.group(1)
            bits = int(m.group(2))
            # shape must exist as base_shape
            shape_key = f"{base}_shape"
            if shape_key not in npz_files:
                raise KeyError(f"Missing shape entry {shape_key} in NPZ for {key}")
            shape = list(map(int, np.asarray(npz[shape_key]).tolist()))

            scales_key = f"{base}_q{bits}_scales"
            zp_key = f"{base}_q{bits}_zero_point"
            if scales_key not in npz_files:
                raise KeyError(f"Missing scales entry {scales_key} in NPZ for {key}")
            scales_array = np.asarray(npz[scales_key])
            scales_npy = f"{base}_q{bits}_scales.npy"
            np.save(scales_npy, scales_array)
            scales_txt = f"{base}_q{bits}_scales.txt"
            np.savetxt(scales_txt, scales_array)

            zps_npy = None
            if zp_key in npz_files:
                zps_arr = np.asarray(npz[zp_key])
                zps_npy = f"{base}_q{bits}_zero_point.npy"
                np.save(zps_npy, zps_arr)

            # Save integer quantized weights to .npy
            q_np = f"{base}_q{bits}.npy"
            np.save(q_np, npz[key])

            scale_axis_index = infer_scale_axis(npz_files, npz, base)
            entry = {
                'q': q_np,
                'shape': shape,
                'q_scales': scales_npy,
                'q_scales_txt': scales_txt,
                'q_zero_point': zps_npy,
                'bit_width': bits,
                'scale_axis': {
                    'index': int(scale_axis_index),
                    'name': axis_name(scale_axis_index)
                },
                'quantization': {
                    'type': 'perrow_int',
                    'bits': bits,
                    'scales': scales_npy,
                    'zero_point': zps_npy
                }
            }
            weights_section[base] = entry
            bits_used.add(bits)


def maybe_override(target, key, value):
    if value is not None:
        target[key] = value


families = dedupe([args.model_family] + parse_list_arg(args.target_families))

architecture = {}
if args.architecture_json:
    architecture.update(load_json_file(args.architecture_json))

maybe_override(architecture, 'families', families or None)
maybe_override(architecture, 'context_length', args.context_length)

hidden_size = args.hidden_size or infer_hidden_size(weights_section)
maybe_override(architecture, 'hidden_size', hidden_size)

intermediate_size = args.intermediate_size or infer_intermediate_size(weights_section)
maybe_override(architecture, 'intermediate_size', intermediate_size)

num_layers = args.num_layers or infer_num_layers(weights_section.keys())
maybe_override(architecture, 'num_layers', num_layers)

vocab_size = args.vocab_size or infer_vocab_size(weights_section)
maybe_override(architecture, 'vocab_size', vocab_size)

num_heads = args.num_heads or args.nhead or infer_num_heads(hidden_size, None)
kv_groups = args.kv_groups

attention_block = architecture.get('attention', {})
maybe_override(attention_block, 'variant', args.attention_variant)
maybe_override(attention_block, 'num_heads', num_heads)
maybe_override(attention_block, 'kv_groups', kv_groups)
maybe_override(attention_block, 'head_dim', (hidden_size // num_heads) if (hidden_size and num_heads) else None)
maybe_override(attention_block, 'window_size', args.window_size)
if attention_block:
    architecture['attention'] = attention_block

maybe_override(architecture, 'activation', args.activation)
maybe_override(architecture, 'norm', args.norm_type)

rope_block = architecture.get('rope', {})
maybe_override(rope_block, 'base_theta', args.rope_base)
maybe_override(rope_block, 'scale', args.rope_scale)
if args.rope_scaling_json:
    rope_block.update(load_json_file(args.rope_scaling_json))
if rope_block:
    architecture['rope'] = rope_block

architecture = drop_none(architecture)

model_block = drop_none({
    'primary_family': args.model_family,
    'families': families or None,
    'notes': args.model_notes or None,
    'source_npz': str(npz_path)
})

if len(bits_used) == 0:
    q_bit = None
elif len(bits_used) == 1:
    q_bit = next(iter(bits_used))
else:
    q_bit = 'mixed'

quantization_meta = {
    'bit_width': q_bit,
    'supported_bit_widths': sorted(list(bits_used)) if bits_used else None,
    'zero_point': 0 if (1 in bits_used) else None,
    'scale': {
        'type': 'per-row',
        'dtype': 'float32'
    },
    'activation_dtype': 'float32',
    'packing': {
        '1bit': {
            'layout': 'row-major',
            'endianness': 'msb_first',
            'library': 'numpy.packbits',
            'bits_per_chunk': 8
        },
        'integer': {
            'layout': 'row-major',
            'dtype': 'numpy.npy',
            'bits': 'variable'
        }
    }
}

manifest = {
    'format_version': 2,
    'model': model_block,
    'architecture': architecture,
    'quantization': quantization_meta,
    'weights': weights_section
}

for name, entry in weights_section.items():
    manifest[name] = entry

with open(args.out_manifest, 'w', encoding='utf-8') as handle:
    json.dump(manifest, handle, indent=2)

print(f"Exported packed quantized weights and wrote manifest: {args.out_manifest}")
