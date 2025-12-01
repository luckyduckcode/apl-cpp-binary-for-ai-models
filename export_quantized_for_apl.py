import argparse
import numpy as np
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--npz', type=str, default='student_quantized_1bit.npz', help='Path to quantized npz file')
parser.add_argument('--out_manifest', type=str, default='student_quantized_manifest.json', help='Manifest output JSON')
parser.add_argument('--nhead', type=int, default=None, help='Optional number of attention heads to embed per-layer as NHEADS')
args = parser.parse_args()

npz = np.load(args.npz)
manifest = {}

for k, v in npz.items():
    if k.endswith("_1bit"):
        name = k[:-5]
        packed = v
        fname = f"{name}_1bit.bin"
        packed.tofile(fname)
        manifest[name] = manifest.get(name, {})
        manifest[name]['packed'] = fname
        manifest[name]['shape'] = list(map(int, npz[f"{name}_shape"].tolist()))
        manifest[name]['scales'] = f"{name}_scales.npy"
        np.save(manifest[name]['scales'], npz[f"{name}_scales"]) 
        # Also write human-readable scales for easier C++ input
        txt = f"{name}_scales.txt"
        np.savetxt(txt, npz[f"{name}_scales"]) 
        manifest[name]['scales_txt'] = txt
        if args.nhead is not None:
            manifest[name]['NHEADS'] = int(args.nhead)
    elif k.endswith("_fp32"):
        name = k[:-5]
        fname = f"{name}_fp32.npy"
        np.save(fname, v)
        manifest[name] = manifest.get(name, {})
        manifest[name]['fp32'] = fname

with open(args.out_manifest, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Exported packed quantized weights and wrote manifest: {args.out_manifest}")
