# APL Binary Llama
![CI](https://github.com/luckyduckcode/apl-cpp-binary-for-ai-models/actions/workflows/ci.yml/badge.svg)

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
You can also run the cross-platform build helper, which will choose the appropriate build script for your OS (Windows, macOS, Linux/WSL):

```bash
python3 scripts/build_backend.py
```
## Quickstart & Priorities

Short-term (days):
- Run `./scripts/build_backend.sh` to build the backend
 - Or run the cross-platform helper: `python3 scripts/build_backend.py` to let the script detect and choose the right build for your OS.
- Run `./scripts/ci_runner.sh` to build and exercise the demo tests
- Use `apl/loader_demo.apl` to verify the APL demo that calls the Python wrapper for the C++ backend
 - **NEW: Easy Model Runner** - Run popular AI models easily:
  ```bash
  pip install -r requirements.txt
  python easy_run.py --model tinyllama
  ```

## Easy Model Runner

The `easy_run.py` script makes it simple to run popular AI models with binary quantization:

### Supported Models
- `tinyllama`: TinyLlama 1.1B - Small but capable
- `mistral-7b`: Mistral 7B - Fast and capable  
- `gemma-2b`: Gemma 2B - Google's lightweight model
- `llama-7b`: Llama 2 7B (requires access token)

### Usage
```bash
# Run TinyLlama
python easy_run.py --model tinyllama

# Run with limited layers for testing
python easy_run.py --model mistral-7b --layers 4

# Use a custom HuggingFace model
python easy_run.py --custom-model microsoft/DialoGPT-small
```

The script will:
1. Download the model from HuggingFace
2. Quantize weights to 1-bit
3. Export for APL runtime
4. Run a demo inference

Notes on model access and tokens (Windows & cross-platform):

- Models like Llama 2 and Mistral on Hugging Face may require an access token; set HUGGINGFACE_HUB_TOKEN in your environment. For PowerShell:

  ```powershell
  $env:HUGGINGFACE_HUB_TOKEN = "<token>"
  ```

  For bash/WSL:

  ```bash
  export HUGGINGFACE_HUB_TOKEN="<token>"
  ```

Mid-term (weeks):
- Finalize multi-head attention in `llama.apl` and add unit tests for end-to-end correctness
- Add a more robust APL FFI (a C extension or a loadable APL native function) to call the backend directly
- Finalize stable QAT training scripts and benchmark grid for accuracy vs. memory/latency trade-offs

Long-term (months):
- Implement AVX2/AVX512 & GPU kernels for larger scale models
- Full release & CI to run reproducible benchmarks and QAT training runs

Windows notes:

- If you are running on Windows natively, you have three options to build the native backend:
  - Use `scripts/build_backend_windows.ps1` (PowerShell). This will attempt to use MinGW (g++) or MSVC if available.
  - Open a Visual Studio Developer Command Prompt and build manually (MSVC) producing a `backend_1bit.dll`.
  - Use WSL (recommended) and run `bash scripts/build_backend.sh` which builds a `backend_1bit.so` inside WSL.

Examples (PowerShell):

```powershell
# Install requirements
pip install -r requirements.txt

# Build on Windows (PowerShell)
powershell -ExecutionPolicy Bypass -File scripts/build_backend_windows.ps1

# Run the easy runner
python easy_run.py --model tinyllama
```

Windows Convenience (PowerShell):

```powershell
# Quick setup and run wrapper
.\scripts\run_easy_run_windows.ps1 -Model tinyllama
```

Examples (WSL / POSIX):

```bash
pip install -r requirements.txt
bash scripts/build_backend.sh
python easy_run.py --model tinyllama
```

If you are interested, I can start by finalizing `llama.apl` and adding comprehensive tests for the attention & ffn blocks — or set up CI to make the repo more robust.

CI runs
-------
We run a cross-platform GitHub Actions workflow that tries to build the native backend and run a few smoke tests on Linux, macOS, and Windows. If you want a local reproduction, follow the steps above or run `python scripts/build_backend.py` and `pytest -q`.

Notes:
- The XNOR kernel demonstrates substantial speedups when activations are binarized (sign-only). For weight-only quantization (float activations), the dequantized fallback is still used. Integration into an APL runtime requires exporting the packed weight files and scales as was implemented in `export_quantized_for_apl.py`.
 - To compile the shared library wrapper and call from Python (demo):

```bash
g++ -O3 -march=native -std=c++17 -fopenmp -shared -fPIC -o cpp/backend_1bit.so cpp/backend_1bit.cpp
python3 cpp/call_backend.py
```

 - Example to integrate into the APL runtime:
	 1. Compile `backend_1bit.so`.
    1b. (Optional) Compile `cpp/loader_example` to test native loading with `dlopen`/`LoadLibrary`.
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

Native loader example
---------------------
The repo includes `cpp/loader_example` which demonstrates loading `backend_1bit.so` dynamically and calling `matmul_1bit`. Build it using the cross-platform build script and run it as a standalone test:

```bash
python scripts/build_backend.py
./cpp/loader_example student_quantized_manifest.json cpp/backend_1bit.so
```

On Windows, run the EXE:

```powershell
python scripts/build_backend.py
.\cpp\loader_example.exe student_quantized_manifest.json cpp/backend_1bit.dll
```
Converting llama.cpp models (gguf / ggml)
----------------------------------------

If you have a local `gguf` / `ggml` model (`llama.cpp` format) you can convert it and run it with this repo:

1) Convert `gguf`/`ggml` -> HF directory (use `llama.cpp` tools or community converters):

  Example commands (community tools / `llama.cpp`):

  ```bash
  # Using llama.cpp conversion script (if available)
  python3 llama.cpp/scripts/convert.py --input path/to/model.gguf --output_hf /tmp/hf_model
  ```

  or if you use community tools that convert to PyTorch HF format:

  ```bash
  python convert-ggml-to-hf.py --input path/model.gguf --output /tmp/hf_model
  ```

2) Once you have an HF-style model directory, run the exporter and runner:

```bash
python scripts/gguf_to_apl.py --hf-dir /tmp/hf_model --run-export
```

3) Or pre-convert to NPZ/quantized format and then export to APL using the standard pipeline:

```bash
# Convert to NPZ with 1-bit quantization (if the converter supports it)
# Or use `easy_run.py` to download the HF model and quantize
python easy_run.py --custom-model /tmp/hf_model --output-dir models
```

If you prefer, use `scripts/gguf_to_apl.py --gguf path/to/your.gguf --run-export` to attempt an automatic conversion if ldaamma.cpp/convert is available locally.



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

Dequantization helper
---------------------

If you need to run `llama.apl` directly with FP32 arrays (e.g., for correctness testing inside an APL interpreter), dequantize 1-bit packed weights into YAML/Numpy arrays using:

```bash
python scripts/dequantize_manifest_weights.py --manifest student_quantized_manifest.json --out_dir models/fp32
```

You can optionally update the manifest in place so it includes `fp32` keys for each weight:

```bash
python scripts/dequantize_manifest_weights.py --manifest student_quantized_manifest.json --update-manifest
```

When `fp32` fields are present in the manifest, `scripts/manifest_to_apl.py` will expose `{weight}_fp32` APL variables that `llama.apl` can consume directly.


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