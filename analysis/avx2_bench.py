import subprocess
import numpy as np
import csv
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from quantization import binarize_weights

def run_case(out_dim, in_dim, threads, runs=10):
    W = np.random.randn(out_dim, in_dim).astype(np.float32)
    packed, scales, info = binarize_weights(W, per_channel_axis=0)
    packed.tofile('cpp/bench_weights.bin')
    np.savetxt('cpp/bench_scales.txt', scales)
    v = np.random.randn(in_dim).astype(np.float32)
    np.savetxt('cpp/bench_in.txt', v)
    sign_bits = np.packbits((v>=0).astype(np.uint8))
    sign_bits.tofile('cpp/bench_in.bin')

    cmd_f = ['./cpp/bitmatmul_xnor_simd', 'cpp/bench_weights.bin', 'cpp/bench_scales.txt', str(out_dim), str(in_dim), 'cpp/bench_in.txt', 'floatact', str(threads)]
    cmd_b = ['./cpp/bitmatmul_xnor_simd', 'cpp/bench_weights.bin', 'cpp/bench_scales.txt', str(out_dim), str(in_dim), 'cpp/bench_in.bin', 'binact', str(threads)]

    # warmup
    subprocess.run(cmd_f)
    subprocess.run(cmd_b)

    t_f = 0.0
    t_b = 0.0
    for _ in range(runs):
        import time
        ts = time.time()
        subprocess.run(cmd_f, stdout=subprocess.DEVNULL)
        t_f += (time.time() - ts)
        ts = time.time()
        subprocess.run(cmd_b, stdout=subprocess.DEVNULL)
        t_b += (time.time() - ts)
    return (t_f/runs)*1000.0, (t_b/runs)*1000.0

def main():
    cases = [(512,512),(1024,512),(2048,1024),(4096,2048)]
    threads = [1,2,4,8]
    rows = []
    os.makedirs('analysis', exist_ok=True)
    for out_dim, in_dim in cases:
        for t in threads:
            print('Running', out_dim, in_dim, 'threads', t)
            ms_f, ms_b = run_case(out_dim, in_dim, t, runs=3)
            rows.append([out_dim, in_dim, t, ms_f, ms_b])
    with open('analysis/avx2_bench.csv','w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['out','in','threads','floatact_ms','binact_ms'])
        w.writerows(rows)
    print('Wrote analysis/avx2_bench.csv')

if __name__ == '__main__':
    main()
