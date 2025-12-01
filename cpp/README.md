# C++ bitpacked matmul prototype

This is a minimal prototype to load bitpacked weights and per-channel scales and run a vector-matrix product:

- `bitmatmul.cpp` reads a packed weight binary (MSB-first packbits per row), a scales `.txt` file (one float per line), shape `out` and `in`, and an input vector as `.txt` or `random` to generate random input.

Usage:

```bash
g++ -O3 -march=native -std=c++17 -o bitmatmul bitmatmul.cpp
./bitmatmul path/to/weights.bin path/to/scales.txt <out> <in> random
```

Notes:
- This prototype unpacks each bit and computes the dot product directly using the scale (either per-row or per-input-column).
- It uses basic `std::ifstream` to read files and prints the first 16 outputs.
- It's not highly optimized; it's intended as a functional prototype to validate bitpacked formats.

Integrating with APL/C++ runtime:
- The packed binary and scales files are created via `export_quantized_for_apl.py`.
- For performance, replace the per-bit loop with a popcount-based algorithm or SIMD operations.
- For production, implement optimized XNOR + popcount kernel and use blocked matmul.
