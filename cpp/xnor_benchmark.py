import subprocess
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from quantization import binarize_weights

def run_benchmark(out_dim=1024, in_dim=512, runs=10):
    W = np.random.randn(out_dim, in_dim).astype(np.float32)
    packed, scales, info = binarize_weights(W, per_channel_axis=0)
    packed.tofile('cpp/bench_weights.bin')
    np.savetxt('cpp/bench_scales.txt', scales)

    v = np.random.randn(in_dim).astype(np.float32)
    np.savetxt('cpp/bench_in.txt', v)
    # sign bits
    signs = (v >= 0).astype(np.uint8)
    sign_bits = np.packbits(signs)
    sign_bits.tofile('cpp/bench_in.bin')

    # compile
    subprocess.run(['g++', '-O3', '-march=native', '-std=c++17', '-fopenmp', '-o', 'cpp/bitmatmul_xnor', 'cpp/bitmatmul_xnor.cpp'])

    # run a few times and measure time
    def run(cmd):
        t0 = subprocess.getoutput('date +%s%N')
        for _ in range(runs):
            r = subprocess.run(cmd, capture_output=True, text=True)
        t1 = subprocess.getoutput('date +%s%N')
        dt = (int(t1) - int(t0)) / 1e6
        return dt / runs

    t_f = run(['./cpp/bitmatmul_xnor', 'cpp/bench_weights.bin', 'cpp/bench_scales.txt', str(out_dim), str(in_dim), 'cpp/bench_in.txt', 'floatact', '0'])
    t_b = run(['./cpp/bitmatmul_xnor', 'cpp/bench_weights.bin', 'cpp/bench_scales.txt', str(out_dim), str(in_dim), 'cpp/bench_in.bin', 'binact', '0'])
    print('floatact avg ms:', t_f)
    print('binact avg ms:', t_b)

if __name__ == '__main__':
    run_benchmark()
