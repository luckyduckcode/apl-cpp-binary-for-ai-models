# APL Integration Guide for 1-bit Backend

This guide explains how to integrate the `backend_1bit.so` (shared library) and bitpacked weight files into an APL runtime.

## 1) Build the backend library

```bash
# From the repository root
g++ -O3 -march=native -std=c++17 -fopenmp -shared -fPIC -o cpp/backend_1bit.so cpp/backend_1bit.cpp
```

## 2) Export quantized NPZ for APL loader

```bash
python3 export_quantized_for_apl.py --npz student_quantized_1bit_qat.npz --out_manifest student_quantized_manifest.json
```

This generates packed `*.bin` files and `*.txt` scale files plus `student_quantized_manifest.json`.

## 3) Use a CLI loader from APL (shell fallback)

APL can call shell commands using `)system` or `)sh` depending on your interpreter; a simple approach is:

- Use `)system` to call `cpp/loader_example` which demonstrates calling into `backend_1bit.so`:

```
)system ./cpp/loader_example student_quantized_manifest.json cpp/backend_1bit.so
```

This CLI wrapper prints outputs to stdout, which APL can capture if needed.

## 4) Direct FFI into `backend_1bit.so` from APL runtime

Some APLs allow loading a shared library and calling functions. To do this:

1. Use `dlopen` and `dlsym` in a small C wrapper (example `cpp/loader_example`) to find `matmul_1bit`.
2. The C wrapper provides a thin API mapping APL native types to the `matmul_1bit(... )` signature. Note: APL uses 1-based indexing and boxed arrays — careful conversion is needed.

Example integration steps:

- Prepare a C wrapper that reads the `packed` and `scales` file (or takes pointers) and calls `matmul_1bit`.
- In APL, call the wrapper using `)system` or by compiling the wrapper into a dynamic library and call into it with your APL's FFI mechanisms.

## 5) Memory Safety & Performance Notes

- Choose `binact` mode to get the fastest XNOR+popcount path (requires activations to be quantized to signs and packed). The runtime needs to provide a packed activation vector.
- If only weights are quantized (weight-only) and activations are still FP32, use `floatact` mode which dequantizes each row on-the-fly and multiplies with float activations.
- For production, prefer pre-packing model weights, broadcasting the same per-batch activation vector across threads, and using multi-threading and vectorized kernels.

## Appendix: Example APL usage via Shell

```
)system ./cpp/loader_example student_quantized_manifest.json cpp/backend_1bit.so
```

This approach is robust and avoids deep changes in APL integration while exploring and validating quantized artifacts and runtime performance.
