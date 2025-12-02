import ctypes
import numpy as np
import os
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(repo_root))
QUIET = ('--out-json' in sys.argv or '--out-file' in sys.argv)
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
        if not QUIET:
            print('[call_backend] Compiled backend not found; falling back to pure-Python unpack/dequantize matmul for demos')
        lib = None
        try:
            # Attempt to import unpack_binarized from the repo's quantization module
            from quantization import unpack_binarized
        except Exception:
            unpack_binarized = None
else:
    raise RuntimeError('Unsupported OS')

HAVE_MATMUL_Q = False
if lib is not None:
    # define prototype for native backend
    lib.matmul_1bit.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.matmul_1bit.restype = ctypes.c_int
    # optional integer quant kernel in-memory
    have_matmul_q = False
    try:
        lib.matmul_q_in_mem.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.matmul_q_in_mem.restype = ctypes.c_int
        have_matmul_q = True
    except Exception:
        have_matmul_q = False
    HAVE_MATMUL_Q = have_matmul_q

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
    def call_matmul_q_in_mem(q_array, scales_array, zero_point_array, vec, out, _in, bits=2, mode=0, threads=0):
        # q_array: numpy uint8/uint16 array shape (out, in)
        if not HAVE_MATMUL_Q:
            raise RuntimeError('Compiled backend does not support matmul_q_in_mem')
        if not isinstance(q_array, np.ndarray):
            raise ValueError('q_array must be numpy array')
        elem_bytes = q_array.dtype.itemsize
        q_ptr = q_array.ctypes.data_as(ctypes.c_void_p)
        scales_ptr = (scales_array.ctypes.data_as(ctypes.c_void_p) if scales_array is not None else ctypes.c_void_p(0))
        zps_ptr = (zero_point_array.ctypes.data_as(ctypes.c_void_p) if zero_point_array is not None else ctypes.c_void_p(0))
        in_ptr = vec.ctypes.data_as(ctypes.c_void_p)
        out_arr = np.zeros(out, dtype=np.float32)
        out_ptr = out_arr.ctypes.data_as(ctypes.c_void_p)
        ret = lib.matmul_q_in_mem(q_ptr, ctypes.c_int(elem_bytes), scales_ptr, zps_ptr, in_ptr, out_ptr, out, _in, bits, threads)
        if ret != 0:
            raise RuntimeError('backend matmul_q_in_mem failed: ' + str(ret))
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

        def call_matmul_q_in_mem(q_array, scales_array, zero_point_array, vec, out, _in, bits=2, mode=0, threads=0):
            # Python fallback for integer quantized weights: dequantize to float and multiply
            if not isinstance(q_array, np.ndarray):
                raise ValueError('q_array must be numpy array')
            # q_array shape: (out, in)
            if zero_point_array is not None:
                zp = zero_point_array
            else:
                zp = np.zeros(q_array.shape[0], dtype=np.int32)
            if scales_array is not None:
                scales = scales_array
            else:
                scales = np.ones(q_array.shape[0], dtype=np.float32)
            # Dequantize
            q_int = q_array.astype(np.int32)
            if q_int.ndim == 1:
                q_int = q_int.reshape((q_array.shape[0], -1))
            # broadcast subtract
            if zp.ndim == 0 or zp.shape[0] == 1:
                q_int = q_int - int(zp)
            else:
                q_int = (q_int.T - zp).T
            # apply scales
            deq = q_int.astype(np.float32) * scales.reshape((-1, 1))
            return deq.dot(vec.astype(np.float32))


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

    # integer quantized arrays: 'q' (np.npy), 'q_scales', 'q_zero_point'
    qfile = cfg.get('q')
    q_scales = cfg.get('q_scales') or cfg.get('q_scales_txt')
    q_zp = cfg.get('q_zero_point')

    packed = cfg.get('packed') or cfg.get('packed_file')
    scales = cfg.get('scales_txt') or cfg.get('scales')
    shape = cfg.get('shape')
    if not packed or not scales:
        raise KeyError(f"Packed/scale file missing for {weight_name} in manifest {manifest_path}")

    # If not absolute, assume same directory as manifest
    qpath = None
    scales_path = None
    # If qfile exists -> use integer kernel path
    if qfile:
        qpath = (mp.parent / qfile).as_posix() if not Path(qfile).is_absolute() else qfile
        scales_path = (mp.parent / q_scales).as_posix() if (q_scales and not Path(q_scales).is_absolute()) else q_scales
        zp_path = (mp.parent / q_zp).as_posix() if (q_zp and not Path(q_zp).is_absolute()) else q_zp
    else:
        packed_path = (mp.parent / packed).as_posix() if not Path(packed).is_absolute() else packed
        scales_path = (mp.parent / scales).as_posix() if not Path(scales).is_absolute() else scales
    # For 1-bit path: If scales_path is a numpy binary (.npy), convert it to .txt so the backend can parse text
    if qpath is None and scales_path and scales_path.endswith('.npy'):
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

    if qpath is not None:
        # integer quantized path: load numpy arrays and call C++ kernel if available else fallback
        qarr = np.load(qpath, allow_pickle=False)
        scarr = np.load(scales_path, allow_pickle=False) if scales_path and Path(scales_path).exists() else None
        zp_arr = np.load(zp_path, allow_pickle=False) if (q_zp and Path(zp_path).exists()) else None
        if not QUIET:
            print(f"[call_matmul_from_manifest] manifest={manifest_path} weight={weight_name} q={qpath} q_scales={scales_path} q_zero_point={zp_path} out={out} in={_in} mode={mode}")
        # Use compiled kernel if available
        try:
            if lib is not None and HAVE_MATMUL_Q:
                return call_matmul_q_in_mem(qarr, scarr, zp_arr, vec, out, _in, bits=cfg.get('bit_width', 2), mode=mode, threads=threads)
        except Exception:
            pass
        # fallback
        return call_matmul_q_in_mem(qarr, scarr, zp_arr, vec, out, _in, bits=cfg.get('bit_width', 2), mode=mode, threads=threads)
    else:
        if not QUIET:
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
