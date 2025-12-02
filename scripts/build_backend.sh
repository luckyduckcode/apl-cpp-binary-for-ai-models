#!/usr/bin/env bash
# POSIX build script for Linux/WSL/macOS. For Windows native builds, use `scripts/build_backend_windows.ps1`.
set -euo pipefail

echo "Building backend_1bit.so (shared library)"
mkdir -p build
cd build

GCC=$(command -v g++ || true)
if [ -z "${GCC}" ]; then
  echo "g++ not found. Install build-essential or equivalent."
  exit 1
fi

SRC_ROOT=$(git rev-parse --show-toplevel)
cd "$SRC_ROOT"

OUT=backend_1bit.so
echo "Compiling: cpp/backend_1bit.cpp + cpp/bitmatmul_xnor_avx2.cpp"
g++ -O3 -std=c++17 -mavx2 -fopenmp -fPIC -shared \
    cpp/backend_1bit.cpp cpp/bitmatmul_xnor_avx2.cpp -o $OUT || \
  g++ -O3 -std=c++17 -fPIC -shared cpp/backend_1bit.cpp -o $OUT

echo "Build complete: $OUT"
echo "Compiling loader_example..."
g++ -O3 -std=c++17 -fPIC cpp/loader_example.cpp -ldl -o cpp/loader_example || true
if [ -f cpp/loader_example ]; then
  echo "Built cpp/loader_example"
fi
