import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import resource
try:
    import psutil
except Exception:
    psutil = None

from distillation import StudentModel
from quantization import unpack_binarized
from quantization import replace_linears_with_qat


def get_rss_mb():
    if psutil:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    else:
        # fallback to ru_maxrss in MB (POSIX)
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_dequantized_model(npzfile, vocab_size=1000, d_model=64, nhead=4, num_layers=1):
    npz = np.load(npzfile)
    student = StudentModel(vocab_size, d_model, nhead, num_layers)
    state = student.state_dict()
    for k in list(state.keys()):
        base = k
        key_1bit = f"{base}_1bit"
        key_fp32 = f"{base}_fp32"
        if key_1bit in npz:
            packed = npz[key_1bit]
            shape = tuple(npz[f"{base}_shape"])  # (out, in)
            scales = npz[f"{base}_scales"]
            key_pc = f"{base}_pc_axis"
            if key_pc in npz:
                pc_axis = int(npz[key_pc])
            else:
                # infer per-channel axis from length of scales
                if len(scales) == shape[0]:
                    pc_axis = 0
                elif len(scales) == shape[1]:
                    pc_axis = 1
                else:
                    raise ValueError(f"Cannot infer per-channel axis for {base}, scales length {len(scales)} not matching any shape")
            mat = unpack_binarized(packed, scales, shape, per_channel_axis=pc_axis)
            state[k] = torch.tensor(mat, dtype=torch.float32)
        elif key_fp32 in npz:
            state[k] = torch.tensor(npz[key_fp32], dtype=torch.float32)
        else:
            # leave default
            pass
    student.load_state_dict(state)
    student.eval()
    return student


def bench(model, warmup=5, runs=50, input_shape=(16, 20)):
    data = torch.randint(0, 1000, input_shape)
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model(data)
    # Measure
    t0 = time.time()
    for _ in range(runs):
        with torch.no_grad():
            model(data)
    t1 = time.time()
    # Average per run
    avg_ms = (t1 - t0) / runs * 1000.0
    return avg_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp32', type=str, default='student_fp32.pth')
    parser.add_argument('--ptq', type=str, default='student_quantized_1bit.npz')
    parser.add_argument('--qat', type=str, default='student_quantized_1bit_qat.npz')
    parser.add_argument('--vocab_size', type=int, default=1000)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--runs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=20)
    args = parser.parse_args()

    print('Benchmarking with settings:', args)
    # FP32 model
    saved_state = torch.load(args.fp32)
    is_qat_saved = any(k.endswith('.scale') for k in saved_state.keys())
    student_fp = StudentModel(args.vocab_size, args.d_model, args.nhead, args.num_layers)
    if is_qat_saved:
        replace_linears_with_qat(student_fp, per_channel_axis=0)
    student_fp.load_state_dict(saved_state)
    student_fp.eval()

    rss_before = get_rss_mb()
    ms_fp32 = bench(student_fp, warmup=2, runs=args.runs, input_shape=(args.batch_size, args.seq_len))
    rss_after = get_rss_mb()
    mem_fp32 = rss_after - rss_before
    print(f"FP32 avg latency: {ms_fp32:.3f} ms | Mem delta: {mem_fp32:.3f} MB")

    # PTQ (dequantize) - if file exists
    try:
        try:
            student_ptq = load_dequantized_model(args.ptq, args.vocab_size, args.d_model, args.nhead, args.num_layers)
        except Exception as e:
            print('PTQ (dequantized) loader failed:', e)
            # Inspect PTQ keys for debugging
            npz = np.load(args.ptq)
            for k, v in npz.items():
                print('PTQ', k, v.shape)
            raise
        rss_before = get_rss_mb()
        ms_ptq = bench(student_ptq, warmup=2, runs=args.runs, input_shape=(args.batch_size, args.seq_len))
        rss_after = get_rss_mb()
        mem_ptq = rss_after - rss_before
        print(f"PTQ (dequantized) avg latency: {ms_ptq:.3f} ms | Mem delta: {mem_ptq:.3f} MB")
    except Exception as e:
        print('PTQ (dequantized) failed:', e)

    # QAT (dequantize)
    try:
        student_q = load_dequantized_model(args.qat, args.vocab_size, args.d_model, args.nhead, args.num_layers)
        rss_before = get_rss_mb()
        ms_qat = bench(student_q, warmup=2, runs=args.runs, input_shape=(args.batch_size, args.seq_len))
        rss_after = get_rss_mb()
        mem_qat = rss_after - rss_before
        print(f"QAT (dequantized) avg latency: {ms_qat:.3f} ms | Mem delta: {mem_qat:.3f} MB")
    except Exception as e:
        print('QAT (dequantized) failed:', e)


if __name__ == '__main__':
    main()
