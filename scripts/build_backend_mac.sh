#!/usr/bin/env bash
# Build backend on macOS (uses clang++). Creates backend_1bit.dylib and a backend_1bit.so symlink.
set -euo pipefail

echo "Building backend_1bit on macOS"
SRC_ROOT=$(git rev-parse --show-toplevel)
cd "$SRC_ROOT"

OUT_DYLIB=backend_1bit.dylib
OUT_SO=backend_1bit.so

CLANG=$(command -v clang++ || true)
if [ -z "$CLANG" ]; then
  echo "clang++ not found. Install Xcode Command Line Tools or clang via brew."
  exit 1
fi

mkdir -p build
cd build

echo "Compiling with clang++ (no OpenMP by default on macOS)"
SRC="${SRC_ROOT}/cpp/backend_1bit.cpp ${SRC_ROOT}/cpp/bitmatmul_xnor_avx2.cpp"

# Try to use OpenMP if libomp is available
OPENMP_FLAGS=""
if brew --version >/dev/null 2>&1 && brew list libomp >/dev/null 2>&1; then
  echo "libomp found via brew; enabling OpenMP"
  OPENMP_FLAGS="-Xpreprocessor -fopenmp -lomp"
fi

echo "Compiling to $OUT_DYLIB"
clang++ -O3 -std=c++17 -fPIC -dynamiclib $OPENMP_FLAGS $SRC -o $OUT_DYLIB || {
  echo "First attempt failed; trying less aggressive flags (no avx2)"
  clang++ -O3 -std=c++17 -fPIC -dynamiclib $SRC -o $OUT_DYLIB
}

# Create a .so symlink for compatibility with POSIX loaders
if [ -f "$OUT_DYLIB" ]; then
  ln -sf "$OUT_DYLIB" "$OUT_SO"
  echo "Build complete: $OUT_DYLIB; also created $OUT_SO -> $OUT_DYLIB"
else
  echo "Failed to build $OUT_DYLIB"
  exit 1
fi

echo "Compiling loader_example..."
clang++ -O3 -std=c++17 -fPIC $OPENMP_FLAGS ${SRC_ROOT}/cpp/loader_example.cpp -o ${SRC_ROOT}/cpp/loader_example || echo "loader_example build (mac) failed"

exit 0
