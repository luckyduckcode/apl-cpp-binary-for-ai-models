import subprocess
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from quantization import unpack_binarized, binarize_weights

# Build a small random weight matrix and pack it using our quantization utilities
W = np.random.randn(16, 64).astype(np.float32)  # 16 outputs, 64 inputs
from quantization import binarize_weights
packed, scales, info = binarize_weights(W, per_channel_axis=0)
# write files
packed.tofile('test_weights.bin')
np.savetxt('test_scales.txt', scales)
# input vector
v = np.random.randn(64).astype(np.float32)
np.savetxt('test_input.txt', v)

# Compile C++ helper
subprocess.run(['g++', '-O3', '-march=native', '-std=c++17', '-o', 'cpp/bitmatmul', 'cpp/bitmatmul.cpp'])
# Run C++ executable
r = subprocess.run(['./cpp/bitmatmul', 'test_weights.bin', 'test_scales.txt', '16', '64', 'test_input.txt'], capture_output=True, text=True)
print('C++ output (first values):', r.stdout)

# Compare with unpacked python result
from quantization import unpack_binarized
mat = unpack_binarized(packed, scales, (16,64), per_channel_axis=0)
out = mat.dot(v)
print('Python dequant outputs (first 16):', out[:16])

# Print mismatch
cpp_vals = list(map(float, r.stdout.strip().split()))
print('Max diff:', max(abs(cpp_vals[i]-out[i]) for i in range(len(cpp_vals))))
