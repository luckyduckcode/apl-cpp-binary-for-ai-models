import ctypes
import numpy as np
import os

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
