# APL Binary Llama: 1-bit Weight-only Quantization with QAT and Native Kernels — Research Paper (Draft)

## Abstract

We present an end-to-end methodology for weight-only, 1-bit quantization for transformer-style models targeted at a high-performance APL/C++ runtime. The pipeline combines per-channel 1-bit binarization, quantization-aware training (QAT) with learnable per-channel scale parameters, a robust export format for packed weights, and an initial prototype of a native 1-bit kernel. The paper documents the rationale for the approach, algorithmic details, pseudocode, experiments, and a comparison with existing PTQ and multi-bit quantization approaches.

## 1. Introduction

Large language models pose significant deployment challenges due to memory footprint and compute costs. While recent work has focused on low-bitwidth quantization (4-bit and below), 1-bit weight-only quantization offers dramatic model size and bandwidth reductions, but is challenging in accuracy. We present a practical pipeline: (1) quantization scheme and format, (2) Quantization-Aware Training (QAT) with learnable per-channel scales, (3) export format for bit-packed weights + scales, and (4) a native C++ prototype kernel for inference.

The contributions are: (a) a reproducible pipeline and artifact set for 1-bit weight-only quantization; (b) QAT modules for weights allowing learnable scales; (c) a robust NPZ export format and manifest for APL/C++ runtime; and (d) a functional C++ prototype demonstrating correct inference and a baseline for performance optimizations.

## 2. Background and Related Work

- Post-Training Quantization (PTQ) reduces precision after training, minimizing compute burden but often resulting in accuracy loss at aggressive bit widths.
- Quantization-Aware Training (QAT) inserts fake-quant operations during training (straight-through gradients) to adapt weights for quantized inference.
- Weight-only quantization focuses on quantizing model parameters but not activations. Scheduling more complex for 1-bit, but reduces memory and disk.
- XNOR-Net and binary networks showcased earlier proof-of-concept that binary weights or activations can be used in deep networks; we adapt some of these ideas for transformer matrices.

Comparison: This approach sits at the intersection of QAT with bitpacking and a native kernel targeted at APL + C++ runtimes.

## 3. Problem Formulation

Goal: Replace float32 weight matrices W with binarized, packed weights B and per-channel (per-row) scales s such that:

W_{i,:} ~ s_i * sign(B_{i,:})

where B_{i,:} is a packed bit row (0 -> -1, 1 -> +1). We want a pipeline that:
- trains weights with QAT, producing binarization-friendly values;
- exports a packed representation with scales; and
- runs a native kernel that uses bit operations (XNOR/popcount) with scales to compute matrix-vector and matrix-matrix multiplies.


## 4. Design and Rationale

Key design choices:
- Per-channel (per-row) scale: preserves more information per output neuron versus a single global scale.
- Learnable scale in QAT: allows the training to compensate and adapt scales for minimized loss.
- Weight-only QAT: we keep activations in float32 to maximize compatibility with existing runtime and numerical stability.
- APL-friendly NPZ + JSON manifest export: lightweight loader format for smaller runtimes.
- C++ kernel prototype (MSB-first packbits used by `numpy.packbits`): simple to read and verify; production kernels will use blocked formats optimized for cache and SIMD.

Why per-row and learnable scales? Per-row preserves the amplitude of each output neuron's contributions. Learnable scales help reduce the accuracy gap between FP32 and 1-bit weights because the network learns to adjust magnitude and sign during training.

## 5. Pseudocode

### 5.1 Binarize weights (PTQ)

```
function BINARIZE_PTQ(W, per_channel_axis=0):
    # W: (out, in) matrix
    if per_channel_axis == 0:
        scales = mean(abs(W), axis=1)  # shape (out,)
    else:
        scales = mean(abs(W), axis=0)  # shape (in,)
    scales[scales == 0] = 1.0
    signs = sign(W)                   # +1 or -1 (0 treated as +1)
    bits = (signs > 0).astype(uint8)  # {0,1}
    packed = packbits(bits, axis=1)   # pack each row into bytes
    return packed, scales, {shape=W.shape, per_channel_axis}
```

### 5.2 Unpack & Dequantize (for dequantized fallback)

```
function UNPACK_BP(packed, scales, shape, per_channel_axis=0):
    out, in = shape
    rows = []
    for r in range(out):
        bits = np.unpackbits(packed[r])[:in]
        sign = (bits == 1) ? +1 : -1
        if per_channel_axis == 0:
            row = sign * scales[r]
        else:
            row = sign * scales[:in]
        rows.append(row)
    return np.vstack(rows)
```

### 5.3 QAT weight wrapper (PyTorch-style)

```
class FakeQuant1bitFunction(torch.autograd.Function):
    def forward(ctx, weight):
        scales = mean(abs(weight), dim=1, keepdim=True)
        scales[scales == 0] = 1.0
        sign = sign(weight); sign[sign == 0] = 1
        return sign * scales
    def backward(ctx, grad_out):
        return grad_out  # STE
```

```
class QATLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, learnable_scale=True):
        weight param
        if learnable_scale:
            scale param (out, 1) initialized to mean abs per row
    def forward(x):
        if learnable_scale:
             sign = sign(weight); sign[sign==0]=1
             qweight = sign * scale
        else:
             qweight = FakeQuant1bit(weight)
        return linear(x, qweight, bias)
```

### 5.4 Export pipeline

```
function EXPORT_NPZ(model_state_dict, out_npz_path):
    quantized = {}
    for (name, param) in state_dict:
        nd = numpy(param)
        if nd.ndim >= 2:
            packed, scales, info = BINARIZE_PTQ(nd)
            quantized[f"{name}_1bit"] = packed
            quantized[f"{name}_scales"] = scales
            quantized[f"{name}_shape"] = info.shape
            quantized[f"{name}_pc_axis"] = info.per_channel_axis
        else:
            quantized[f"{name}_fp32"] = nd
    savez_compressed(out_npz_path, **quantized)
```

### 5.5 C++ bitpacked matmul (prototype)

```
// Given: packed matrix file, scales.txt, shape out,in, input vector v
bytes_per_row = ceil(in / 8)
for r in 0..out-1:
    row_bytes = bytes[r * bytes_per_row : (r+1) * bytes_per_row]
    accum = 0
    for b in 0..bytes_per_row-1:
        byte = row_bytes[b]
        for bit=0..7:
            idx = b*8+bit
            if idx >= in: break
            bitval = (byte >> (7 - bit)) & 1  # MSB first
            sign = bitval ? +1 : -1
            s = (scales_length == out) ? scales[r] : scales[idx]
            accum += sign * s * v[idx]
    outvec[r] = accum
```

Optimized native kernel: replace inner `for bit` loop with XNOR + popcount for chunks of 64 bits and apply scaling.

## 6. Experiments & Results

We target a toy student model and benchmark FP32, PTQ (dequantized), and QAT (dequantized) and compare accuracy deltas and latency on a CPU. The pipeline includes an integration test validating the C++ kernel yields the same output as Python dequantization.

### 6.1 Summary Metrics (toy run)

- FP32 -> baseline
- PTQ (1-bit, per-row) vs FP32: max diff ~ 1.34, mean diff ~ 0.26.
- QAT (1-bit) vs FP32: max diff improved to ~ 1.20, mean diff ~ 0.25 (small improvement in toy training).
- Latency (toy benchmark): FP32 ~ 6ms, PTQ (dequantized) ~ 6.3ms, QAT(dequantized) ~ 4.3ms in our simple micro-benchmark; actual numbers depend on your CPU configuration and vectorization.
- C++ prototype matched Python dequant within 5e-5 for test vectors (validating bitpacking & scaling logic).

### 6.3 Figures and Expanded Results

The following figures summarize the experiment runner results that compare PTQ and QAT variants across short toy runs. See `analysis/results.csv` for exact numbers.

![Max diff by config](figures/max_diff_by_config.png)

![Mean diff by config](figures/mean_diff_by_config.png)

![Avg latency by config](figures/latency_by_config.png)

The CSV also contains aggregated metrics including `max_diff`, `mean_diff`, and average latencies for both FP32 and dequantized models.

### Appendix: Experiment Table

See the full table of results in `docs/results_table.md` (excerpt below):

<!-- include results table -->
```markdown
<!-- results_table.md -->
```

### 6.4 XNOR Kernel Benchmarks

We evaluated the optimized XNOR + popcount kernel using several matrix sizes and thread counts; the results are summarized below.

![XNOR bench threads 1](figures/xnor_threads_1.png)
![XNOR bench threads 4](figures/xnor_threads_4.png)
![XNOR bench threads 8](figures/xnor_threads_8.png)

The figures indicate that for larger matrices and multi-threaded execution, the binact (binary activations) kernel provides better throughput in many, but not all, cases; actual speedups may vary with CPU, SIMD support, and memory/sub-system characteristics.

### 6.5 AVX2 SIMD Kernel & Integration

We implemented an AVX2-friendly SIMD prototype (`cpp/bitmatmul_xnor_simd.cpp`) which uses 64-bit chunk processing and makes use of the `__builtin_popcountll` for popcount of 64-bit words with OpenMP for threading. This kernel performs equivalently to the simpler prototype while setting a foundation for further intrinsics optimization (e.g., AVX2/AVX512 vectorized XNOR, POPCNT, and blocked matmul).

The `backend_1bit.so` C API provides `matmul_1bit(...)` which can be called from Python and from C runtimes. The library supports both float activation mode (dequantize-and-multiply) and binary activation mode with XNOR+popcount. The example `cpp/call_backend.py` demonstrates usage.

For APL integration:

1. Compile the library and load it via the APL C-extension mechanism or via an external bridge.
2. Map the `packed` binary to APL arrays and pass pointers, or call the file-based C API to use packed files directly.
3. Prefer the `binact` mode when activations are binarized to leverage the XNOR kernel; otherwise use `floatact` mode.

Future optimizations:
- Replace the `__builtin_popcountll` loops with AVX2/AVX512 intrinsics that compute popcounts over vector lanes in parallel.
- Blocked matmul with cache-friendly tiling and tile-level popcount reduction.
- Use a column-major layout or bit-tiling scheme for faster word-aligned access.

### 6.2 Discussion

The experiments show that QAT with learnable scales can improve the quantized model over naive PTQ, though the improvement in the toy example is limited by small dataset and training time. The prototype C++ kernel confirms correctness and integration feasibility.

## 7. Why this approach is better than alternatives

- Memory & Bandwidth: 1-bit weight-only reduces model weights by 32x from FP32 (or 8x from FP8), enabling deployments on limited memory devices and significantly reducing memory bandwidth during inference.
- Performance: With native bitwise XNOR+popcount kernels and scaling, 1-bit weights offer huge throughput improvements if optimized kernels are used.
- Accuracy: QAT with learnable per-channel scales allows the network to adapt to binarized weights, reducing the gap in accuracy vs PTQ-only methods and making the approach practical (vs naive PTQ for 1-bit which often fails).
- Safety & Compatibility: Keeping activations in FP32 maintains numerical stability and enables a phased migration where weights are quantized first and activations second.

## 8. Limitations & Future Work

- The current C++ prototype is correct but not optimized. We need a high-performance kernel (blocking, SIMD intrinsics, XNOR-popcount), multi-threading, and vectorized dequantization.
- QAT requires a real dataset and careful hyperparameter tuning; our toy experiments are limited.
- Per-group scales or mixed bit widths (e.g., 1-bit for projection weights, 4-bit for sensitive layers) may offer better trade-offs.
- Activation quantization and end-to-end integer inference would enable more throughput but needs careful numerical analysis.

## 9. Reproducibility

- `distillation.py` reproduces QAT runs; run with `--qat` and log outputs for per-epoch differences.
- `benchmarks` and `cpp/integration_test.py` validate correctness and speed measured on CPU.
- `export_quantized_for_apl.py` writes a `student_quantized_manifest.json` mapping packed files to shapes & scales.

## 10. Conclusion

We present a complete prototype showing how 1-bit weight-only quantization can be made practical with QAT and a per-channel learnable scale design, a robust export format, and a native kernel prototype that ensures compatibility with APL and C++ runtimes. While further optimization and larger-scale experiments are required to reach production-ready accuracy, this pipeline demonstrates the feasibility and performance advantages of this approach.

### Acknowledgements

- Approaches inspired by binary networks and existing quantization literature such as XNOR-Net and QAT methods adapted for transformer-style models.

### Appendix: Pseudocode and Algorithms

See the pseudocode snippets in Section 5. Full implementation and tests are in the codebase.

## 11. Multi-Model Support Roadmap (1-bit Retained)

The same 1-bit FTP/Q pipeline can power a broader family of decoder-only transformers (Mistral 7B, DeepSeek-R1, Code Llama 7B/13B/34B/70B, Gemma, Qwen) by layering structural metadata on top of the packed weights. We keep the quantization core untouched and extend manifests, runtimes, and validation harnesses.

### 11.1 Manifest blueprint

- Extend `student_quantized_manifest.json` to carry both `quantization` (bit-width, packing order, per-layer scale blobs, zero-point metadata) and `architecture` (layer count, dims, head shapes, KV groups, RoPE/base parameters, activation type, norm type, attention mask strategy).
- `export_quantized_for_apl.py` now emits `format_version: 2` manifests with `model`, `architecture`, `quantization`, and `weights` sections and exposes CLI flags to label the target families (Mistral, DeepSeek-R1, Code Llama, Gemma, Qwen) while keeping the legacy top-level keys for compatibility.
- Model-specific keys:
    - **Mistral 7B**: `attn.type="gqa"`, `kv_groups=8`, `window_size=4096`, `activation="swiglu"`.
    - **DeepSeek-R1**: `attn.type="moe"` with router weights and expert bank descriptors; fall back to dense for validation.
    - **Code Llama**: `rope_theta`, `context_length`, `vocab_size` overrides per variant.
    - **Gemma**: `norm="rms"`, `kv_heads=1` (MQA), `rope_scale` factors.
    - **Qwen**: `rope_per_head` array, optional `alibi=true`, `kv_groups` per version.
- Exporter (`export_quantized_for_apl.py`) reads checkpoints, runs existing 1-bit packing, then emits manifest sections describing the structural knobs without repacking bytes.

### 11.2 Runtime refactor plan

- `llama.apl` loads the manifest, iterates over `layers`, and dispatches helpers:
    - `ROPE_APPLY` uses manifest-provided theta/scale, supports per-head factors.
    - `ATTENTION_BLOCK` switches among dense, GQA, MQA, sliding-window, or ALiBi-masked paths.
    - `FFN_BLOCK` selects ReLU, SwiGLU, or Gemma/Qwen variants while consuming the same unpacked 1-bit matrices.
    - `NORM_BLOCK` toggles LayerNorm vs RMSNorm via manifest flag.
- Backend (`cpp/backend_1bit.cpp`) leverages the same packed buffers but reshapes outputs according to layer metadata (e.g., broadcasting single KV heads for Gemma, grouped heads for Mistral/Qwen) before matmul.
- Loader (`cpp/call_backend.py`) maps manifest entries to backend calls so APL and Python share behavior gates.

### 11.3 Validation + test strategy

- `scripts/apl_validate.py` gains `--manifest` and `--model-family` to pick the right numpy reference runner and iterate through representative layers, reporting max/mean diffs per family.
- PyTest adds parametrized slices (single layer or block) for each family with small exported NPZ bundles, verifying that backend + APL outputs match PyTorch/NumPy references within tolerances.
- CI matrix (optional heavy stage) sets `MODEL_FAMILY` to run selected validations nightly, ensuring the shared 1-bit kernel remains consistent across all manifests.

This roadmap lets us ship multi-model binaries without sacrificing the 1-bit advantages: quantized tensors never change, only the metadata describing how the runtime assembles them for each architecture.
