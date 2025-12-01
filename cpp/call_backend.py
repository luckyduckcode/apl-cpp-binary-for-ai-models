import ctypes
import numpy as np
import os
import json
from pathlib import Path

lib = None
if os.name == 'posix':
    lib = ctypes.CDLL('./cpp/backend_1bit.so')
else:
    raise RuntimeError('Unsupported OS')

# define prototype
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
    import sys
    import subprocess
    # demo
    if not os.path.exists('student_quantized_1bit_qat.npz'):
        print('Need quantized NPZ; run distillation.py')
        sys.exit(1)
    import numpy as np
    npz = np.load('student_quantized_1bit_qat.npz')
    name = 'fc.weight'
    packed = npz[f'{name}_1bit']
    shape = tuple(npz[f'{name}_shape'])
    scales = npz[f'{name}_scales']
    packed.tofile('tmp_packed.bin')
    np.savetxt('tmp_scales.txt', scales)
    # random vector
    v = np.random.randn(shape[1]).astype(np.float32)
    vfile = 'tmp_in.txt'
    np.savetxt(vfile, v)
    out = call_matmul('tmp_packed.bin', 'tmp_scales.txt', v, shape[0], shape[1], mode=0, threads=0)
    print(out[:10])
