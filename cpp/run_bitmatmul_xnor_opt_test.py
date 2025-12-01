import subprocess
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from quantization import binarize_weights, unpack_binarized

W = np.random.randn(64, 128).astype(np.float32)
packed, scales, info = binarize_weights(W)
packed.tofile('cpp/test_opt_packed.bin')
np.savetxt('cpp/test_opt_scales.txt', scales)

v = np.random.randn(128).astype(np.float32)
np.savetxt('cpp/test_opt_in.txt', v)

sign_bits = np.packbits((v >= 0).astype(np.uint8))
sign_bits.tofile('cpp/test_opt_in.bin')

# run float activation
r1 = subprocess.run(['./cpp/bitmatmul_xnor_opt', 'cpp/test_opt_packed.bin', 'cpp/test_opt_scales.txt', '64', '128', 'cpp/test_opt_in.txt', 'floatact', '4'], capture_output=True, text=True)
print('FLOATACT OUT:', r1.stdout)

r2 = subprocess.run(['./cpp/bitmatmul_xnor_opt', 'cpp/test_opt_packed.bin', 'cpp/test_opt_scales.txt', '64', '128', 'cpp/test_opt_in.bin', 'binact', '4'], capture_output=True, text=True)
print('BINACT OUT:', r2.stdout)

# Compare to Python dequants
mat = unpack_binarized(packed, scales, (64,128), per_channel_axis=0)
float_out = mat.dot(v)
signs = np.unpackbits(sign_bits)[:128]
binact_out = np.zeros(64)
for i in range(64):
    rowbits = np.unpackbits(packed[i])[:128]
    w_signs = np.where(rowbits == 1, 1.0, -1.0)
    a_signs = np.where(signs == 1, 1.0, -1.0)
    binact_out[i] = np.sum(w_signs * a_signs) * scales[i]

cpp_float = list(map(float, r1.stdout.strip().split()))
cpp_bin = list(map(float, r2.stdout.strip().split()))

print('Max float diff:', max(abs(cpp_float[i] - float_out[i]) for i in range(len(cpp_float))))
print('Max bin diff:', max(abs(cpp_bin[i] - binact_out[i]) for i in range(len(cpp_bin))))
