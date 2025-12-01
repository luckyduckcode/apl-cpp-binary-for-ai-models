import subprocess
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from quantization import binarize_weights, unpack_binarized, pack_bits

# Create a small random matrix and input vector
W = np.random.randn(32, 64).astype(np.float32)
packed, scales, info = binarize_weights(W, per_channel_axis=0)
packed.tofile('cpptest_weights.bin')
np.savetxt('cpptest_scales.txt', scales)

v = np.random.randn(64).astype(np.float32)
np.savetxt('cpptest_in.txt', v)

# Also create a packed sign vector for binary activation
signs = (v >= 0).astype(np.uint8)
sign_bits = np.packbits(signs)
sign_bits.tofile('cpptest_in.bin')

# Compile
subprocess.run(['g++', '-O3', '-march=native', '-std=c++17', '-o', 'cpp/bitmatmul_xnor', 'cpp/bitmatmul_xnor.cpp'])

# Run floatact
import time

def run_exec(cmd):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = (time.time() - t0) * 1000.0
    return r, dt

r1, t1 = run_exec(['./cpp/bitmatmul_xnor', 'cpptest_weights.bin', 'cpptest_scales.txt', '32', '64', 'cpptest_in.txt', 'floatact', '4'])
print('FLOATACT (ms):', t1)
print('FLOATACT OUT:', r1.stdout)

# Run binact (uses sign bits), expects packbits as msb-first
r2, t2 = run_exec(['./cpp/bitmatmul_xnor', 'cpptest_weights.bin', 'cpptest_scales.txt', '32', '64', 'cpptest_in.bin', 'binact', '4'])
print('BINACT (ms):', t2)
print('BINACT OUT:', r2.stdout)

# Compare against python
from quantization import unpack_binarized
mat = unpack_binarized(packed, scales, (32,64), per_channel_axis=0)
float_out = mat.dot(v)
# For binact, compute dot using signs only: sum(sign_w * sign_a) * scale
sign_bits_arr = np.unpackbits(sign_bits)[:64]
binact_out = np.zeros(32)
for i in range(32):
    rowbits = np.unpackbits(packed[i])[:64]
    # w_signs
    w_signs = np.where(rowbits == 1, 1.0, -1.0)
    a_signs = np.where(sign_bits_arr == 1, 1.0, -1.0)
    binact_out[i] = np.sum(w_signs * a_signs) * scales[i]

print('Python FLOAT out (first 5):', float_out[:5])
print('Python BINACT out (first 5):', binact_out[:5])

# Print diffs vs C++
cpp_float = list(map(float, r1.stdout.strip().split()))
cpp_bin = list(map(float, r2.stdout.strip().split()))

print('FLOAT max diff:', max(abs(cpp_float[i] - float_out[i]) for i in range(32)))
print('BINACT max diff:', max(abs(cpp_bin[i] - binact_out[i]) for i in range(32)))
