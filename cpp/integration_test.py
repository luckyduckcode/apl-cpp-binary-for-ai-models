import subprocess
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from quantization import unpack_binarized

npzfile = 'student_quantized_1bit_qat.npz'
if not os.path.exists(npzfile):
    npzfile = 'student_quantized_1bit.npz'
assert os.path.exists(npzfile), 'No quantized npz found. Run distillation.py first.'

# Export using existing script
subprocess.run(['python3', 'export_quantized_for_apl.py', '--npz', npzfile, '--out_manifest', 'test_manifest.json'])

# Run on fc.weight (fc.weight_1bit.bin and fc.weight_scales.txt)
packed = 'fc.weight_1bit.bin'
scales = 'fc.weight_scales.txt'
shape = (1000, 64)

# Build random input vector
v = np.random.randn(shape[1]).astype(np.float32)
np.savetxt('input_vec.txt', v)

subprocess.run(['g++', '-O3', '-march=native', '-std=c++17', '-o', 'cpp/bitmatmul', 'cpp/bitmatmul.cpp'])
r = subprocess.run(['./cpp/bitmatmul', packed, scales, str(shape[0]), str(shape[1]), 'input_vec.txt'], capture_output=True, text=True)
print('C++ output:', r.stdout)

# Now Python dequantized reference
npz = np.load(npzfile)
packed = npz['fc.weight_1bit']
shape = tuple(npz['fc.weight_shape'])
scales = npz['fc.weight_scales']
mat = unpack_binarized(packed, scales, shape, per_channel_axis=0)
out = mat.dot(v)
print('Python first 16:', out[:16])

cpp_vals = list(map(float, r.stdout.strip().split()))
print('Max diff:', max(abs(cpp_vals[i] - out[i]) for i in range(min(len(cpp_vals), len(out)))))

print('Integration test complete.')
