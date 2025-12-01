import numpy as np
import torch
import torch.nn as nn
from distillation import StudentModel
from quantization import unpack_binarized


def main():
    # Load FP32 student
    student_fp32 = StudentModel(1000, 64, 4, 1)
    student_fp32.load_state_dict(torch.load('student_fp32.pth'))
    student_fp32.eval()

    # Dummy input
    dummy_data = torch.randint(0, 1000, (2, 10))
    with torch.no_grad():
        fp_out = student_fp32(dummy_data).numpy()

    # Load quantized npz
    npz = np.load('student_quantized_1bit.npz')

    # Build a new student and populate with dequantized weights
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
            if key_pc in npz:
                pc_axis = int(npz[key_pc])
            else:
                if scales.shape[0] == shape[0]:
                    pc_axis = 0
                elif scales.shape[0] == shape[1]:
                    pc_axis = 1
                else:
                    raise ValueError(
                        f"Can't infer per-channel axis for {base}: scales len {scales.shape[0]} vs shape {shape}")
            mat = unpack_binarized(packed, scales, shape, per_channel_axis=pc_axis)
            state[k] = torch.tensor(mat, dtype=torch.float32)
        elif key_fp32 in npz:
            state[k] = torch.tensor(npz[key_fp32], dtype=torch.float32)
        else:
            print(f"Missing quantized key for {k}; falling back to fp32 baseline")

    student_q.load_state_dict(state)
    student_q.eval()

    # forward
    with torch.no_grad():
        q_out = student_q(dummy_data).numpy()

    # Compare
    print('FP32 output shape:', fp_out.shape)
    print('Quantized dequantized output shape:', q_out.shape)

    # Compare statistics
    diff = np.abs(fp_out - q_out)
    print('Max diff:', diff.max())
    print('Mean diff:', diff.mean())

    # If similar, okay
    if diff.max() < 1e-3:
        print('Quantized (1-bit) dequantized model matches FP32 closely')
    else:
        print('Quantized model deviates, consider QAT or per-channel scaling improvements')


if __name__ == '__main__':
    main()
