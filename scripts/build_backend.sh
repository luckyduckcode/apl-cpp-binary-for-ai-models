#!/usr/bin/env bash
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
