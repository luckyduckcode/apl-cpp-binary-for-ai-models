# APL Integration Guide for 1-bit Backend

This guide explains how to integrate the `backend_1bit.so` (shared library) and bitpacked weight files into an APL runtime.

## 1) Build the backend library

```bash
# From the repository root
g++ -O3 -march=native -std=c++17 -fopenmp -shared -fPIC -o cpp/backend_1bit.so cpp/backend_1bit.cpp
```

## 2) Export quantized NPZ for APL loader

```bash
python3 export_quantized_for_apl.py \
	--npz student_quantized_1bit_qat.npz \
	--out_manifest student_quantized_manifest.json \
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

- Only override the fields you know — the exporter infers shapes/scales automatically. Smaller models can omit most of the flags and rely on defaults (e.g., `--model-family llama`).
- `--architecture-json` and `--rope-scaling-json` accept drop-in config files if you already have structured metadata per model family.

The command generates packed `*.bin` files, scale blobs (`*.npy` and `*.txt`), and a v2 `student_quantized_manifest.json` that includes both architecture metadata and quantization details while preserving the legacy per-weight keys for existing tooling.

### Convert manifest to APL variables for demos

You can generate a small APL snippet exposing manifest values as simple APL variables, which is helpful for demos or gluing APL code to the manifest without a JSON parser in APL:

```bash
python3 scripts/manifest_to_apl.py --manifest student_quantized_manifest.json --out_apl apl/generated_manifest.apl
```

The generator writes `apl/generated_manifest.apl` with variables like `MODEL_FAMILY`, `WEIGHT_NAMES`, and per-weight variables, e.g., `fc_weight_packed`, `fc_weight_scales_txt`, and `fc_weight_shape`. A demo use case: call the generator, then read the resulting APL file inside your APL session via `)load` or by including it with your existing APL scripts.

### Manifest schema (v2)

The manifest now contains:

```jsonc
{
	"format_version": 2,
	"model": {
		"primary_family": "mistral",
		"families": ["mistral", "deepseek-r1", "code-llama", "gemma", "qwen"],
		"source_npz": "mistral_qat.npz"
	},
	"architecture": {
		"num_layers": 32,
		"hidden_size": 4096,
		"intermediate_size": 14336,
		"context_length": 32768,
		"attention": {
			"variant": "gqa",
			"num_heads": 32,
			"kv_groups": 8,
			"window_size": null
		},
		"activation": "swiglu",
		"norm": "rmsnorm",
		"rope": {"base_theta": 1000000, "scale": 8.0}
	},
	"quantization": {
		"bit_width": 1,
		"scale": {"type": "per-row", "dtype": "float32"},
		"packing": {"layout": "row-major", "endianness": "msb_first"}
	},
	"weights": {
		"transformer.layers.0.self_attn.in_proj_weight": {
			"packed": "..._1bit.bin",
			"shape": [12288, 4096],
			"scales": "..._scales.npy",
			"scales_txt": "..._scales.txt",
			"bit_width": 1,
			"scale_axis": {"index": 0, "name": "out_features"},
			"packed_format": "numpy.packbits(msb_first,row-major)"
		},
		"transformer.layers.0.self_attn.in_proj_bias": {
			"fp32": "..._fp32.npy",
			"dtype": "fp32"
		}
	}
}
```

> Compatibility: each weight also appears at the manifest root (e.g., `manifest["fc.weight"]`) so older loaders keep working. New tooling should prefer `manifest["weights"][name]` to access the richer metadata.

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

### Using `apl/loader_demo.apl`

There is a minimal example at `apl/loader_demo.apl` to demonstrate a simple, robust approach
to call the Python wrapper that invokes the compiled runtime. This is a practical starting point
until a full FFI extension is implemented.

1. Build and export quantized weights:

```bash
./scripts/build_backend.sh
python3 export_quantized_for_apl.py --npz student_quantized_1bit_qat.npz --out_manifest student_quantized_manifest.json
```

2. Run APL (if installed) and execute the loader demo:

```bash
apl -f apl/loader_demo.apl
```


This approach is robust and avoids deep changes in APL integration while exploring and validating quantized artifacts and runtime performance.
