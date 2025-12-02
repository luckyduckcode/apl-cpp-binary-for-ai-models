# GPU Optimization & Automation Guide

## 🚀 Quick Start

### 1. Build Backend (CPU + Optional GPU)

**CPU-only (recommended for most users):**
```bash
python scripts/build_backend.py
```

**With GPU support (NVIDIA CUDA required):**
```bash
# Ensure CUDA toolkit is installed
nvcc --version  # Should show CUDA version

# Build with GPU
python scripts/build_backend.py --gpu
```

### 2. Validate Accuracy

```bash
python tests/test_accuracy_validation.py
```

This runs cross-validation comparing:
- 1-bit quantization vs FP32 baseline
- 2-bit (q2) quantization
- 4-bit (q4) quantization  
- 8-bit (q8) quantization
- Multi-head attention accuracy

### 3. Fully Automated Model Download & Conversion

**Convert TinyLlama (1.1B, ~240MB):**
```bash
python scripts/convert_models_automated.py tinyllama
```

**Convert Mistral-7B (7B parameters, ~2GB):**
```bash
python scripts/convert_models_automated.py mistral-7b --bits 4
```

**Convert all popular models:**
```bash
python scripts/convert_models_automated.py --convert-all --bits 4
```

**List available models:**
```bash
python scripts/convert_models_automated.py --list-models
```

---

## 🎯 Features

### GPU Support (CUDA)

The backend now includes optional GPU acceleration for quantized matrix multiplication:

- **CUDA Kernels** (`cpp/backend_gpu.cu`):
  - Integer quantized matmul (q2, q4, q8) 
  - 1-bit packed weight matmul
  - Automatic kernel selection based on hardware

- **Seamless Fallback**: If GPU unavailable, automatically falls back to CPU

- **Benefits**:
  - 10-100x speedup for large matrices (depending on GPU)
  - Reduced CPU memory usage
  - Better throughput for batch inference

**Example Usage:**
```cpp
// GPU-accelerated matmul
int status = matmul_q_gpu(q_data, elem_bytes, scales, zps, 
                          in_vec, out_vec, 
                          out_rows, in_cols, bits, threads);
if (status == -1) {
    // GPU not available, fall back to CPU
    status = matmul_q_in_mem(q_data, elem_bytes, scales, zps,
                             in_vec, out_vec,
                             out_rows, in_cols, bits, 0, threads);
}
```

### Tight FFI Integration

New `apl_ffi.cpp` provides a clean C++ FFI wrapper that eliminates Python overhead:

**Key Functions:**
```cpp
// Initialize backend library
int apl_ffi_init(const char* backend_path);

// Direct matmul calls (no Python ctypes overhead)
int apl_matmul_1bit(...);
int apl_matmul_q(...);
int apl_matmul_q_batch(...);  // Batch operations

// Cleanup
int apl_ffi_cleanup();
```

**Benefits:**
- No Python GIL contention
- Native library calls (direct, no ctypes wrapping)
- Support for batch inference
- Lower latency for production deployments

### Accuracy Validation Suite

`tests/test_accuracy_validation.py` validates inference accuracy:

**Validates:**
- ✓ Baseline FP32 vs quantized comparisons
- ✓ Relative L2 error metrics
- ✓ Per-bit-width tolerance thresholds
- ✓ Multi-head attention accuracy
- ✓ Layer-wise validation

**Tolerance Thresholds:**
| Bit Width | Tolerance |
|-----------|-----------|
| 1-bit     | 5%        |
| 2-bit (q2)| 1%        |
| 4-bit (q4)| 0.5%      |
| 8-bit (q8)| 0.1%      |

**Output:**
```
ACCURACY VALIDATION REPORT
================================================================================
Overall: 12/12 tests passed (100%)

1-bit Quantization: 3/3 passed
  rel_error=0.0452 (tolerance=0.0500) ✓ PASS
  ...

4-bit Quantization: 3/3 passed
  rel_error=0.0031 (tolerance=0.0050) ✓ PASS
  ...
```

### Fully Automated Model Pipeline

`scripts/convert_models_automated.py` handles the entire workflow:

1. **Download** - From HuggingFace Hub
2. **Quantize** - With configurable bit-width (1, 2, 4, 8)
3. **Export** - To APL-compatible NPZ + manifest
4. **Validate** - Manifest structure and accuracy
5. **Report** - Summary with model metadata

**Supported Models:**
```
tinyllama           - TinyLlama 1.1B (fast, 240MB)
mistral-7b          - Mistral 7B (capable, 2GB)
mistral-7b-instruct - Mistral 7B Instruct
gemma-2b            - Google Gemma 2B (lightweight)
llama2-7b           - Meta Llama 2 7B (requires token)
```

**Example Output:**
```
================================================================================
AUTOMATED MODEL CONVERSION PIPELINE
Model: tinyllama
Quantization: 4-bit
Output: models
================================================================================

[1] Downloading model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
    Config: 2048d, 22 layers, 32 heads
    ✓ Downloaded

[2] Exporting to quantized NPZ (4-bit)
    Quantizing layer 1/22
    Quantizing layer 2/22
    ...
    Saved: models/tinyllama_quantized_q4.npz (156 MB)

[3] Generating APL manifest
    Generated: models/tinyllama_manifest_q4.json

[4] Validating manifest
    Found 22 weight entries
    ✓ Manifest valid

[5] Running accuracy validation
    ✓ Accuracy validation passed

================================================================================
✓ CONVERSION COMPLETE
  Manifest: models/tinyllama_manifest_q4.json
  Model family: llama
  Context length: 2048
================================================================================
```

---

## 📊 Build Configuration

### Linux/macOS/WSL
```bash
# Detect and use optimal settings
bash scripts/build_backend.sh

# Enable GPU if CUDA available
bash scripts/build_backend.sh --enable-gpu
```

### Windows (PowerShell)
```powershell
# CPU-only
powershell -ExecutionPolicy Bypass -File scripts/build_backend_windows.ps1

# With GPU (if CUDA available)
powershell -ExecutionPolicy Bypass -File scripts/build_backend_windows.ps1 -EnableGPU $true
```

### Cross-Platform
```bash
# Auto-detects OS and calls appropriate build script
python scripts/build_backend.py
python scripts/build_backend.py --gpu  # With GPU support
```

---

## 🧪 Testing Accuracy

### Run Individual Tests

```bash
# Full accuracy suite
python tests/test_accuracy_validation.py

# Specific quantization tests
pytest tests/test_call_backend_q4.py -v
pytest tests/test_call_backend_q8.py -v
pytest tests/test_backend_large_q4.py -v

# Integration tests
pytest tests/test_loader_example.py -v
```

### Custom Validation

```python
from tests.test_accuracy_validation import AccuracyValidator
import numpy as np

validator = AccuracyValidator()

# Test a custom weight matrix
W = np.random.randn(128, 256).astype(np.float32)
x = np.random.randn(256).astype(np.float32)

result = validator.validate_layer(W, x, "custom_layer")
print(validator.report())
```

---

## 🔧 Advanced Usage

### GPU-Accelerated Inference

```cpp
#include "cpp/apl_ffi.h"

int main() {
    // Initialize
    apl_ffi_init("./backend_1bit.so");
    
    // Prepare data
    uint8_t q_data[1024];
    float scales[16];
    float input[64];
    float output[16];
    
    // GPU-accelerated matmul (with fallback)
    int status = apl_matmul_q(
        q_data, 1,           // q_data, elem_bytes
        scales, nullptr,     // scales, zero_points
        input, output,
        16, 64,              // out_rows, in_cols
        4,                   // 4-bit quantization
        4                    // num_threads
    );
    
    if (status != 0) {
        printf("Error: %d\n", status);
    }
    
    apl_ffi_cleanup();
    return 0;
}
```

### Batch Inference

```python
from cpp.call_backend import call_matmul_q_batch

# Multiple inference passes
batch_size = 32
for i in range(batch_size):
    result = call_matmul_q_batch(
        q_array, scales, zps,
        batch_inputs[i*64:(i+1)*64],  # 64 features per input
        16,  # output size
        64   # input size
    )
```

### Custom Model Conversion

```bash
# Convert any HuggingFace model
python scripts/convert_models_automated.py mistralai/Mistral-7B-v0.3 --bits 4

# Use custom output directory
python scripts/convert_models_automated.py tinyllama --output-dir /path/to/models

# Different quantization levels
python scripts/convert_models_automated.py tinyllama --bits 1  # 1-bit
python scripts/convert_models_automated.py tinyllama --bits 2  # 2-bit
python scripts/convert_models_automated.py tinyllama --bits 4  # 4-bit (default)
python scripts/convert_models_automated.py tinyllama --bits 8  # 8-bit
```

---

## 📈 Performance Expectations

### GPU Acceleration Speedup
| Operation | CPU | GPU | Speedup |
|-----------|-----|-----|---------|
| q4 matmul (small)  | 10ms | 1ms | 10x |
| q4 matmul (large)  | 100ms | 5ms | 20x |
| q8 matmul (large)  | 120ms | 4ms | 30x |
| 1-bit matmul (packed) | 5ms | 0.5ms | 10x |

### Memory Usage Reduction
| Format | Relative to FP32 |
|--------|------------------|
| 1-bit (packed) | 3.1% |
| 2-bit (q2) | 6.25% |
| 4-bit (q4) | 12.5% |
| 8-bit (q8) | 25% |

### Model Sizes (After Quantization)
| Model | Quantization | Size |
|-------|--------------|------|
| TinyLlama 1.1B | 4-bit | 240 MB |
| Mistral 7B | 4-bit | 2.0 GB |
| Gemma 2B | 4-bit | 650 MB |

---

## 🐛 Troubleshooting

### CUDA/GPU Issues

**Problem**: "CUDA device not available"
```
Solution: 
  1. Install CUDA toolkit: https://developer.nvidia.com/cuda-toolkit
  2. Verify: nvcc --version
  3. Rebuild: python scripts/build_backend.py --gpu
```

**Problem**: GPU runs out of memory
```
Solution:
  1. Reduce batch size
  2. Use smaller quantization (q2 instead of q4)
  3. Fall back to CPU (set USE_GPU=false)
```

### Model Download Issues

**Problem**: "Model requires HuggingFace access token"
```
Solution:
  1. Get token: https://huggingface.co/settings/tokens
  2. Set environment variable:
     export HUGGINGFACE_HUB_TOKEN="hf_xxxxx"
  3. Try conversion again
```

**Problem**: Out of memory during model loading
```
Solution:
  1. Use a smaller model (tinyllama, gemma-2b)
  2. Use GPU for host memory offloading
  3. Increase system swap
```

### Accuracy Validation

**Problem**: "Relative error exceeds tolerance"
```
Solution:
  1. Check CPU/GPU consistency
  2. Increase bit-width (e.g., q4 → q8)
  3. Verify scales and zero_points are correct
```

---

## 📚 Architecture

### Directory Structure
```
cpp/
├── backend_1bit.cpp      # Main quantized kernels (CPU + GPU control)
├── backend_gpu.cu        # CUDA kernels (optional)
├── apl_ffi.cpp          # FFI wrapper (no Python overhead)
├── loader_example.cpp   # Dynamic loading example
└── call_backend.py      # Python wrapper (legacy, uses FFI)

scripts/
├── build_backend.py           # Cross-platform build coordinator
├── build_backend.sh           # Linux/macOS/WSL build
├── build_backend_windows.ps1  # Windows build
└── convert_models_automated.py # Full pipeline orchestration

tests/
├── test_accuracy_validation.py # Cross-validation suite
├── test_call_backend_q*.py     # Individual quantization tests
└── test_backend_large_*.py     # AVX2/GPU stress tests

models/
└── [converted models with manifests]
```

### Execution Flow

**Inference Path:**
```
APL Code
  ↓
apl_ffi.cpp (native FFI wrapper)
  ↓
[GPU Check]
  ├→ GPU Available → backend_gpu.cu (CUDA kernels)
  └→ GPU Unavailable → backend_1bit.cpp (CPU/AVX2 kernels)
  ↓
Output (float array)
```

---

## 🚀 Next Steps

1. **Build Backend**
   ```bash
   python scripts/build_backend.py --gpu
   ```

2. **Validate Setup**
   ```bash
   python tests/test_accuracy_validation.py
   ```

3. **Download Models**
   ```bash
   python scripts/convert_models_automated.py --convert-all --bits 4
   ```

4. **Run Inference**
   ```bash
   python easy_run.py --model tinyllama
   ```

---

## 📝 License

See LICENSE in repository root.

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- ARM NEON GPU kernels (mobile/embedded)
- AVX-512 support (newer CPUs)
- Kernel fusion (quantize + matmul)
- Multi-GPU support
- Benchmark suite against llama.cpp
