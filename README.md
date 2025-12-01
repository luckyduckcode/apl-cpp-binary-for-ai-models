# APL Binary Llama

This project implements a Llama-like model in APL with binary optimization and knowledge distillation.

## Phase 1: Creation

### APL Implementation
- `embeddings.apl`: Token and positional embeddings
- `self_attention.apl`: Multi-head self-attention mechanism
- `ffn.apl`: Feed-forward network
- `layer_norm.apl`: Layer normalization (fixed for last axis)
- `llama.apl`: Full Llama model with embeddings, layers, and output

### Binary Optimization
- `quantization.py`: Post-training quantization to low-bit precision
- `backend.cpp`: Optimized C++ backend for array operations

## Phase 2: Cross-Model Training
- `distillation.py`: Knowledge distillation training loop

## Setup
1. Install GNU APL: Built and installed from source (apl-1.8). The binary is at `/usr/local/bin/apl`.
2. For backend: Compile `backend.cpp` with `g++ -shared -o backend.so backend.cpp -std=c++11`.
3. For distillation: Install PyTorch if needed, run `python3 distillation.py`.
4. For quantization: Run `python3 quantization.py`.

## Usage
- Load APL files and run the transformer block.
- Train with distillation, then quantize and export to ONNX.

## C++ Kernel and Integration
- `cpp/bitmatmul.cpp`: simple dequantized fallback kernel for bitpacked weights.
- `cpp/bitmatmul_xnor.cpp`: optimized prototype kernel for binary activations (XNOR+popcount) with OpenMP for multi-threading.

Compile & run examples:

```bash
# Basic dequantized kernel
g++ -O3 -march=native -std=c++17 -o cpp/bitmatmul cpp/bitmatmul.cpp
python3 cpp/integration_test.py

# Optimized XNOR kernel
g++ -O3 -march=native -std=c++17 -fopenmp -o cpp/bitmatmul_xnor cpp/bitmatmul_xnor.cpp
python3 cpp/run_bitmatmul_xnor_test.py
```

Notes:
- The XNOR kernel demonstrates substantial speedups when activations are binarized (sign-only). For weight-only quantization (float activations), the dequantized fallback is still used. Integration into an APL runtime requires exporting the packed weight files and scales as was implemented in `export_quantized_for_apl.py`.
 - To compile the shared library wrapper and call from Python (demo):

```bash
g++ -O3 -march=native -std=c++17 -fopenmp -shared -fPIC -o cpp/backend_1bit.so cpp/backend_1bit.cpp
python3 cpp/call_backend.py
```

 - Example to integrate into the APL runtime:
	 1. Compile `backend_1bit.so`.
	 2. Use `export_quantized_for_apl.py` to create packed files and `student_quantized_manifest.json`.
	 3. Call `matmul_1bit` from your APL runtime through the language-specific FFI (e.g., using `dlopen` in C).
	 4. For safety and performance, prefer calling native `binact` mode for binarized activations and to use multi-threaded kernels.