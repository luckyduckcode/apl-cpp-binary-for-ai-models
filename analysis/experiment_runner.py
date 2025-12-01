import argparse
import subprocess
import time
import os
import csv
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
from quantization import unpack_binarized, binarize_weights, replace_linears_with_qat
from distillation import StudentModel

def run_distill(qat=False, epochs=3, batch_size=10, lr=1e-3, learnable_scale=False):
    cmd = ['python3', 'distillation.py', '--epochs', str(epochs), '--batch_size', str(batch_size), '--lr', str(lr)]
    if qat:
        cmd.append('--qat')
    if learnable_scale:
        cmd.append('--learnable_scale')
    subprocess.run(cmd, check=True)

def evaluate_npz(npzfile):
    npz = np.load(npzfile)
    # load student fp32
    saved_state = torch.load('student_fp32.pth')
    is_qat_saved = any(k.endswith('.scale') for k in saved_state.keys())
    student_fp = StudentModel(1000, 64, 4, 1)
    if is_qat_saved:
        from quantization import replace_linears_with_qat
        replace_linears_with_qat(student_fp, per_channel_axis=0)
    student_fp.load_state_dict(saved_state)
    student_fp.eval()
    # dummy input
    dummy = torch.randint(0, 1000, (2,10))
    with torch.no_grad():
        fp_out = student_fp(dummy).numpy()

    # build dequant model
    student_q = StudentModel(1000, 64, 4, 1)
    state = student_q.state_dict()
    for k in list(state.keys()):
        base = k
        key_1bit = f"{base}_1bit"
        key_fp32 = f"{base}_fp32"
        if key_1bit in npz:
            packed = npz[key_1bit]
            shape = tuple(npz[f"{base}_shape"])  # (out, in)
            scales = npz[f"{base}_scales"]
            key_pc = f"{base}_pc_axis"
            pc_axis = 0
            if key_pc in npz:
                pc_axis = int(npz[key_pc])
            else:
                if scales.shape[0] == shape[0]:
                    pc_axis = 0
                elif scales.shape[0] == shape[1]:
                    pc_axis = 1
                else:
                    raise ValueError('Cannot infer pc axis')
            mat = unpack_binarized(packed, scales, shape, per_channel_axis=pc_axis)
            state[k] = torch.tensor(mat, dtype=torch.float32)
        elif key_fp32 in npz:
            state[k] = torch.tensor(npz[key_fp32], dtype=torch.float32)
        else:
            state[k] = state[k]
    student_q.load_state_dict(state)
    student_q.eval()
    with torch.no_grad():
        q_out = student_q(dummy).numpy()
    diff = np.abs(fp_out - q_out)
    return diff.max(), diff.mean()

def bench_latencies(npzfile, runs=10, batch_size=8, seq_len=20):
    # Use benchmark_quant harness code embedded
    # FP32
    saved_state = torch.load('student_fp32.pth')
    student_fp = StudentModel(1000, 64, 4, 1)
    if any(k.endswith('.scale') for k in saved_state.keys()):
        replace_linears_with_qat(student_fp)
    student_fp.load_state_dict(saved_state)
    student_fp.eval()
    def bench(model):
        data = torch.randint(0, 1000, (batch_size, seq_len))
        for _ in range(2):
            with torch.no_grad():
                model(data)
        t0 = time.time()
        for _ in range(runs):
            with torch.no_grad():
                model(data)
        t1 = time.time()
        return (t1 - t0) / runs * 1000.0
    fps = bench(student_fp)

    # dequantize npz
    student_d = StudentModel(1000, 64, 4, 1)
    # reuse evaluate_npz logic to populate state
    npz = np.load(npzfile)
    state = student_d.state_dict()
    for k in list(state.keys()):
        base = k
        key_1bit = f"{base}_1bit"
        key_fp32 = f"{base}_fp32"
        if key_1bit in npz:
            packed = npz[key_1bit]
            shape = tuple(npz[f"{base}_shape"])  # (out, in)
            scales = npz[f"{base}_scales"]
            key_pc = f"{base}_pc_axis"
            pc_axis = 0
            if key_pc in npz:
                pc_axis = int(npz[key_pc])
            else:
                if scales.shape[0] == shape[0]:
                    pc_axis = 0
                elif scales.shape[0] == shape[1]:
                    pc_axis = 1
                else:
                    raise ValueError('Cannot infer pc axis')
            mat = unpack_binarized(packed, scales, shape, per_channel_axis=pc_axis)
            state[k] = torch.tensor(mat, dtype=torch.float32)
        elif key_fp32 in npz:
            state[k] = torch.tensor(npz[key_fp32], dtype=torch.float32)
    student_d.load_state_dict(state)
    student_d.eval()
    ptq_ms = bench(student_d)
    return fps, ptq_ms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_csv', type=str, default='analysis/results.csv')
    args = parser.parse_args()
    os.makedirs('analysis', exist_ok=True)
    configs = [
        {'name': 'ptq', 'qat': False, 'epochs': 0, 'learnable': False},
        {'name': 'qat_short', 'qat': True, 'epochs': 3, 'learnable': False},
        {'name': 'qat_scale', 'qat': True, 'epochs': 3, 'learnable': True}
    ]
    rows = []
    for c in configs:
        print('Running config', c)
        if c['qat']:
            run_distill(qat=True, epochs=c['epochs'], batch_size=10, lr=1e-3, learnable_scale=c['learnable'])
            npz = 'student_quantized_1bit_qat.npz'
        else:
            run_distill(qat=False, epochs=1, batch_size=10, lr=1e-3)
            npz = 'student_quantized_1bit.npz'
        maxd, meand = evaluate_npz(npz)
        fps, ptq_ms = bench_latencies(npz)
        rows.append([c['name'], c['qat'], c['epochs'], c['learnable'], maxd, meand, fps, ptq_ms])
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['config','qat','epochs','learnable','max_diff','mean_diff','fp32_ms','dequant_ms'])
        w.writerows(rows)
    print('Wrote', args.out_csv)

if __name__ == '__main__':
    main()
