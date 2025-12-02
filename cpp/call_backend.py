import ctypes
import numpy as np
import os
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if os.name == 'posix':
    # prefer shared library in repo root (built by scripts/build_backend.sh)
    # prefer shared library in repo root. On macOS the dynamic lib may be backend_1bit.dylib
    candidates = [repo_root / 'backend_1bit.so', repo_root / 'backend_1bit.dylib', repo_root / 'cpp' / 'backend_1bit.so', repo_root / 'cpp' / 'backend_1bit.dylib']
    found = None
    for c in candidates:
        if c.exists():
            found = c
            break
    if found is not None:
        lib = ctypes.CDLL(str(found))
    else:
        raise RuntimeError('Shared library not found. Run scripts/build_backend.sh or scripts/build_backend.py')
elif os.name == 'nt':
    # Windows: support DLL and .pyd extensions
    dll_path = repo_root / 'backend_1bit.dll'
    alt = repo_root / 'cpp' / 'backend_1bit.dll'
    pyd_path = repo_root / 'backend_1bit.pyd'
    if dll_path.exists():
        lib = ctypes.WinDLL(str(dll_path))
    elif alt.exists():
        lib = ctypes.WinDLL(str(alt))
    elif pyd_path.exists():
        lib = ctypes.WinDLL(str(pyd_path))
    else:
        # If we couldn't find a compiled backend, offer a pure-Python fallback for demo purposes.
        print('[call_backend] Compiled backend not found; falling back to pure-Python unpack/dequantize matmul for demos')
        lib = None
        try:
            # Attempt to import unpack_binarized from the repo's quantization module
            from quantization import unpack_binarized
        except Exception:
            unpack_binarized = None
else:
    raise RuntimeError('Unsupported OS')

if lib is not None:
    # define prototype for native backend
    lib.matmul_1bit.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.matmul_1bit.restype = ctypes.c_int

    def call_matmul(packed_file, scales_file, vec, out, _in, mode=0, threads=0):
        # vec: either float numpy vector (mode=0) or packed uint8 array (mode=1)
        if mode == 0:
            in_ptr = vec.ctypes.data_as(ctypes.c_void_p)
        else:
            in_ptr = vec.ctypes.data_as(ctypes.c_void_p)
        out_arr = np.zeros(out, dtype=np.float32)
        out_ptr = out_arr.ctypes.data_as(ctypes.c_void_p)
        ret = lib.matmul_1bit(packed_file.encode('utf-8'), scales_file.encode('utf-8'), in_ptr, out_ptr, out, _in, mode, threads)
        if ret != 0:
            raise RuntimeError('backend matmul failed: ' + str(ret))
        return out_arr
else:
    # Python fallback (slow): unpack and dequantize to float then matmul using numpy
    def call_matmul(packed_file, scales_file, vec, out, _in, mode=0, threads=0):
        if unpack_binarized is None:
            raise RuntimeError('No compiled backend and quantization.unpack_binarized not available. Install dependencies or compile the backend.')
        # Load scales
        if scales_file.endswith('.npy'):
            scales = np.load(scales_file)
        else:
            scales = np.loadtxt(scales_file)
        # compute bytes per row
        bytes_per_row = (_in + 7) // 8
        packed_raw = np.fromfile(packed_file, dtype=np.uint8)
        if packed_raw.size % bytes_per_row != 0:
            raise ValueError('Packed file size not a multiple of bytes_per_row')
        packed_rows = packed_raw.reshape((-1, bytes_per_row))
        # Dequantize
        mat = unpack_binarized(packed_rows, scales, (out, _in), per_channel_axis=0)
        if mode == 0:
            # vec is float32 vector
            return mat.dot(vec.astype(np.float32))
        else:
            # mode==1: vec is packed bits -> unpack
            # unpack input vector into floats {-1,1}
            in_bits = np.unpackbits(vec, bitorder='big')[:_in]
            in_sign = np.where(in_bits == 1, 1.0, -1.0).astype(np.float32)
            return mat.dot(in_sign)


def call_matmul_from_manifest(manifest_path, weight_name, vec, mode=0, threads=0):
    """Convenience wrapper that extracts `packed` and `scales` paths for `weight_name` from
    the manifest JSON and calls call_matmul.
    Supports v2 manifest under keys: `weights` -> name -> {packed, scales, shape}
    and legacy / top-level weight keys `name`.
    """
    mp = Path(manifest_path)
    if not mp.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with mp.open('r', encoding='utf-8') as fh:
        manifest = json.load(fh)

    weights_section = manifest.get('weights', {})
    if weight_name in weights_section:
        cfg = weights_section[weight_name]
    else:
        # Back-compat: top-level entry
        cfg = manifest.get(weight_name, {})
    if not cfg:
        raise KeyError(f"Weight {weight_name} not found in manifest {manifest_path}")

    packed = cfg.get('packed') or cfg.get('packed_file')
    scales = cfg.get('scales_txt') or cfg.get('scales')
    shape = cfg.get('shape')
    if not packed or not scales:
        raise KeyError(f"Packed/scale file missing for {weight_name} in manifest {manifest_path}")

    # If not absolute, assume same directory as manifest
    packed_path = (mp.parent / packed).as_posix() if not Path(packed).is_absolute() else packed
    scales_path = (mp.parent / scales).as_posix() if not Path(scales).is_absolute() else scales
    # If scales_path is a numpy binary (.npy), convert it to .txt so the backend can parse text
    if scales_path.endswith('.npy'):
        sc = np.load(scales_path)
        txt_path = scales_path[:-4] + '.txt'
        np.savetxt(txt_path, sc)
        scales_path = txt_path

    # Infer shapes for call
    if shape is None and 'shape' in cfg:
        shape = cfg['shape']
    if shape is None:
        # Try reading scales to infer out dim
        sc = np.load(scales_path) if scales_path.endswith('.npy') else np.loadtxt(scales_path)
        out = int(np.asarray(sc).shape[0])
        # assume hidden size 64 for now (not ideal)
        _in = 64
    else:
        if isinstance(shape, (list, tuple)) and len(shape) == 2:
            out = int(shape[0]); _in = int(shape[1])
        else:
            out = int(shape[0]); _in = 64

    print(f"[call_matmul_from_manifest] manifest={manifest_path} weight={weight_name} packed={packed_path} scales={scales_path} out={out} in={_in} mode={mode}")
    return call_matmul(packed_path, scales_path, vec, out, _in, mode=mode, threads=threads)

if __name__ == '__main__':
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description='Call backend matmul for a single weight from a manifest')
    parser.add_argument('--manifest', type=str, default=None, help='Path to manifest JSON (v2)')
    parser.add_argument('--weight', type=str, default='embedding.weight', help='Weight name to multiply (matching manifest keys)')
    parser.add_argument('--input', type=str, default=None, help='Path to input vector file (txt) or omit to use random vector')
    parser.add_argument('--mode', type=int, default=0, choices=[0,1], help='Mode: 0=float input, 1=packed bits input')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads for backend (0 == default)')
    parser.add_argument('--out-json', dest='out_json', action='store_true', help='Print output as JSON list')
    parser.add_argument('--out-file', dest='out_file', type=str, default=None, help='Write output to numpy .npy file')
    args = parser.parse_args()

    if args.manifest is None:
        print('Usage: python cpp/call_backend.py --manifest <manifest.json> [--weight <name>] [--input <file>]')
        sys.exit(1)

    # If manifest doesn't exist, error out
    if not os.path.exists(args.manifest):
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    # Find the weight entry in the manifest to infer shapes
    with open(args.manifest, 'r', encoding='utf-8') as fh:
        import json
        manifest = json.load(fh)

    weights_section = manifest.get('weights', {})
    if args.weight not in weights_section and args.weight not in manifest:
        raise KeyError(f"Weight {args.weight} not found in manifest {args.manifest}")

    # Use call_matmul_from_manifest which handles the v2 manifest and paths
    if args.input is not None and os.path.exists(args.input):
        # load vector
        vec = np.loadtxt(args.input).astype(np.float32)
    else:
        # Create a random input vector with the appropriate length inferred from shape
        # Try to determine input length from the manifest entry
        entry = weights_section.get(args.weight, manifest.get(args.weight, {}))
        shape = entry.get('shape') or entry.get('fp32_shape')
        if shape and isinstance(shape, (list, tuple)) and len(shape) == 2:
            _in = int(shape[1])
        else:
            # default guess
            _in = 64
        vec = np.random.randn(_in).astype(np.float32)

    out = call_matmul_from_manifest(args.manifest, args.weight, vec, mode=args.mode, threads=args.threads)
    # Output options
    if args.out_file:
        np.save(args.out_file, out)
        print(f'Wrote output to {args.out_file}')
    elif args.out_json:
        import json as _json
        print(_json.dumps(out.tolist()))
    else:
        print('Output (first 16):', out[:16])
