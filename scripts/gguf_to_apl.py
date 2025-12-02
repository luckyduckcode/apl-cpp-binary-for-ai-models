#!/usr/bin/env python3
"""Helper to convert a local gguf/ggml (llama.cpp) model into a format the repo can export to APL.

This script tries to convert gguf/ggml to a HF-style PyTorch model directory using known conversion tools
if they are installed (e.g., `transformers` or `convert-llama-to-hf` scripts from community tools). If conversion
tools aren't available, it prints instructions for the user.

After conversion to an HF directory, it calls `easy_run.py --custom-model /path/to/hf` to quantize and export.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_convert_script():
    # Common locations of conversion utilities
    candidates = [
        Path('llama.cpp') / 'convert.py',
        Path('llama.cpp') / 'scripts' / 'convert.py',
        Path('tools') / 'convert-ggml-to-pt.py',
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def call_convert(cmd, src, dst):
    try:
        subprocess.check_call([cmd, str(src), str(dst)])
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gguf', type=str, default=None, help='Path to gguf or ggml model file')
    parser.add_argument('--hf-dir', type=str, default=None, help='If you already have a HF model dir')
    parser.add_argument('--out-dir', type=str, default='models/gguf_converted', help='Output dir for HF model or NPZ')
    parser.add_argument('--run-export', action='store_true', help='Run easy_run.py to quantize/export to APL after conversion')
    parser.add_argument('--bits', type=int, default=1, choices=[1,2,4,8], help='Bit width for quantization when exporting (default 1)')
    parser.add_argument('--model-family', type=str, default='llama', help='Model family for export_quantized_for_apl (llama|mistral|gemma|etc)')
    parser.add_argument('--quantizer', type=str, default='export_model_1bit', help='Which quantizer/export helper to call: export_model_1bit or easy_run')
    args = parser.parse_args()

    hf_dir = None
    if args.hf_dir:
        hf_dir = Path(args.hf_dir)
        if not hf_dir.exists():
            print('Specified HF dir not found:', hf_dir)
            return 1
    elif args.gguf:
        gguf = Path(args.gguf)
        if not gguf.exists():
            print('GGUF file not found:', gguf)
            return 1
        convert_script = find_convert_script()
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if convert_script:
            print('Found convert script:', convert_script)
            print('Attempting to run conversion...')
            # If convert script takes args as <src> <dst>
            ok = call_convert(convert_script, gguf, out_dir)
            if ok:
                print('Conversion successful; HF dir at', out_dir)
                hf_dir = out_dir
            else:
                print('Conversion script found but failed; please run it manually. Script:', convert_script)
                return 1
        else:
            print('No converter script found locally. Please run one of these commands to convert:')
            print('  # Option 1: Use llama.cpp convert tools (repo must be cloned)')
            print('  python3 llama.cpp/scripts/convert.py --input ggml/gguf --output_hf /tmp/hf_dir')
            print('  # Option 2: Use community converters such as convert-ggml-to-hf script')
            print('After conversion, run: python easy_run.py --custom-model /tmp/hf_dir --output-dir models --run-demo')
            return 1

    if hf_dir and args.run_export:
        if args.quantizer == 'export_model_1bit' and Path('scripts/export_model_1bit.py').exists():
            print('Running scripts/export_model_1bit.py to quantize and export HF model to APL')
            # assemble paths
            out_npz = Path(args.out_dir) / f'{hf_dir.name}_q{args.bits}.npz'
            out_manifest = Path(args.out_dir) / f'{hf_dir.name}_manifest.json'
            cmd = [sys.executable, 'scripts/export_model_1bit.py', '--hf-model', str(hf_dir), '--out', str(out_npz), '--out-manifest', str(out_manifest), '--bits', str(args.bits), '--model-family', args.model_family]
            subprocess.check_call(cmd)
        else:
            print('Running easy_run.py to quantize and export HF model to APL')
            cmd = [sys.executable, 'easy_run.py', '--custom-model', str(hf_dir), '--output-dir', args.out_dir, '--run-demo']
            subprocess.check_call(cmd)

    print('Done. HF dir:', hf_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
