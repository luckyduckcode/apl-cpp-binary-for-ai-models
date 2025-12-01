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

Compile & run examples:

```bash
# Basic dequantized kernel
g++ -O3 -march=native -std=c++17 -o cpp/bitmatmul cpp/bitmatmul.cpp
python3 cpp/integration_test.py

# Optimized XNOR kernel
g++ -O3 -march=native -std=c++17 -fopenmp -o cpp/bitmatmul_xnor cpp/bitmatmul_xnor.cpp
python3 cpp/run_bitmatmul_xnor_test.py
```
## Quickstart & Priorities

Short-term (days):
- Run `./scripts/build_backend.sh` to build the backend
- Run `./scripts/ci_runner.sh` to build and exercise the demo tests
- Use `apl/loader_demo.apl` to verify the APL demo that calls the Python wrapper for the C++ backend

Mid-term (weeks):
- Finalize multi-head attention in `llama.apl` and add unit tests for end-to-end correctness
- Add a more robust APL FFI (a C extension or a loadable APL native function) to call the backend directly
- Finalize stable QAT training scripts and benchmark grid for accuracy vs. memory/latency trade-offs

Long-term (months):
- Implement AVX2/AVX512 & GPU kernels for larger scale models
- Full release & CI to run reproducible benchmarks and QAT training runs

If you are interested, I can start by finalizing `llama.apl` and adding comprehensive tests for the attention & ffn blocks — or set up CI to make the repo more robust.

Notes:
- The XNOR kernel demonstrates substantial speedups when activations are binarized (sign-only). For weight-only quantization (float activations), the dequantized fallback is still used. Integration into an APL runtime requires exporting the packed weight files and scales as was implemented in `export_quantized_for_apl.py`.
 - To compile the shared library wrapper and call from Python (demo):

```bash
g++ -O3 -march=native -std=c++17 -fopenmp -shared -fPIC -o cpp/backend_1bit.so cpp/backend_1bit.cpp
python3 cpp/call_backend.py
```

 - Example to integrate into the APL runtime:
	 1. Compile `backend_1bit.so`.
	 2. Use `export_quantized_for_apl.py` to create packed files and the v2 `student_quantized_manifest.json`. Pass architecture flags when exporting other families:
		```bash
		python3 export_quantized_for_apl.py \
		  --npz mistral_qat.npz \
		  --out_manifest mistral_manifest.json \
		  --model-family mistral \
		  --target-families "mistral,deepseek-r1,code-llama,gemma,qwen" \
		  --context-length 32768 \
		  --num-layers 32 \
		  --hidden-size 4096 \
		  --intermediate-size 14336 \
		  --num-heads 32 \
		  --kv-groups 8 \
		  --attention-variant gqa \
		  --activation swiglu \
		  --norm-type rmsnorm \
		  --rope-base 1000000 \
		  --rope-scale 8.0
		```
	 3. Call `matmul_1bit` from your APL runtime through the language-specific FFI (e.g., using `dlopen` in C).
	 4. For safety and performance, prefer calling native `binact` mode for binarized activations and to use multi-threaded kernels.

The manifest now exposes `model`, `architecture`, `quantization`, and `weights` sections (while keeping backwards-compatible top-level entries). See `docs/apl_integration.md` for the full schema and guidance on targeting Mistral, DeepSeek-R1, Code Llama, Gemma, and Qwen while staying on the 1-bit pipeline.

Tip: use `scripts/manifest_to_apl.py` to create `apl/generated_manifest.apl` — a small snippet that exposes manifest properties as simple APL variables to `)load` inside an APL session or include in demos.

## Supported Model Families

The exporter and runtime support multiple model architectures through the v2 manifest schema. All use the same 1-bit FPTQ quantization core:

| Family | Attention | Activation | Norm | RoPE Base | Context |
|--------|-----------|------------|------|-----------|---------|
| **Llama** | Full MHA | SwiGLU | RMSNorm | 10,000 | 4K |
| **Mistral** | Sliding-window GQA | SwiGLU | RMSNorm | 10,000 | 32K |
| **DeepSeek-R1** | GQA | SwiGLU | RMSNorm | 10,000 | 64K |
| **Code Llama** | Full MHA | SwiGLU | RMSNorm | 1,000,000 | 16K |
| **Gemma** | MQA | GeGLU | RMSNorm | 10,000 | 8K |
| **Qwen** | Full MHA | SwiGLU | RMSNorm | 1,000,000 | 32K |

### Export Examples

```bash
# Llama 7B
python3 export_quantized_for_apl.py \
  --npz llama7b_qat.npz \
  --model-family llama \
  --num-heads 32 --num-layers 32 --hidden-size 4096 \
  --activation swiglu --norm-type rmsnorm

# Mistral 7B (GQA with sliding window)
python3 export_quantized_for_apl.py \
  --npz mistral7b_qat.npz \
  --model-family mistral \
  --num-heads 32 --kv-groups 8 \
  --attention-variant sliding-window --window-size 4096 \
  --context-length 32768

# DeepSeek-R1 (GQA)
python3 export_quantized_for_apl.py \
  --npz deepseek_r1_qat.npz \
  --model-family deepseek-r1 \
  --num-heads 32 --kv-groups 8 \
  --attention-variant gqa \
  --context-length 65536

# Code Llama (long context)
python3 export_quantized_for_apl.py \
  --npz codellama_qat.npz \
  --model-family code-llama \
  --rope-base 1000000 \
  --context-length 16384

# Gemma (MQA)
python3 export_quantized_for_apl.py \
  --npz gemma_qat.npz \
  --model-family gemma \
  --num-heads 16 --kv-groups 1 \
  --attention-variant mqa \
  --activation geglu

# Qwen (long context)
python3 export_quantized_for_apl.py \
  --npz qwen_qat.npz \
  --model-family qwen \
  --rope-base 1000000 \
  --context-length 32768
```

### Architecture Metadata in APL

After running `manifest_to_apl.py`, the generated APL file contains:

```apl
MODEL_FAMILY ← 'mistral'
HIDDEN_SIZE ← 4096
NUM_LAYERS ← 32
NUM_HEADS ← 32
KV_GROUPS ← 8
ATTENTION_VARIANT ← 'sliding-window'
ACTIVATION ← 'swiglu'
NORM_TYPE ← 'rmsnorm'
ROPE_BASE ← 10000
```

These variables are used by `llama.apl` to configure the transformer at runtime.