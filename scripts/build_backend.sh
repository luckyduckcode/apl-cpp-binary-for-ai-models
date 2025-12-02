#!/usr/bin/env bash
# POSIX build script for Linux/WSL/macOS
# Supports CPU optimization with AVX2 and optional GPU acceleration
# Usage: bash scripts/build_backend.sh [--enable-gpu]

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_GPU=false
ENABLE_OPENMP=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --enable-gpu)
            BUILD_GPU=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "=========================================="
echo "Building Backend"
echo "=========================================="
echo "Repo root: $REPO_ROOT"
echo "GPU support: $BUILD_GPU"
echo ""

# Check compiler
if ! command -v g++ &> /dev/null; then
    echo "[ERROR] g++ not found. Install with: sudo apt-get install build-essential"
    exit 1
fi

cd "$REPO_ROOT"

echo "[1] Compiling C++ backend..."

# Detect CPU features
if grep -q "avx2" /proc/cpuinfo 2>/dev/null || [[ "$OSTYPE" == "darwin"* ]]; then
    AVX2_FLAG="-mavx2"
    echo "  ✓ AVX2 support detected"
else
    AVX2_FLAG=""
    echo "  ⚠ AVX2 not available; using scalar fallback"
fi

# Base compilation flags
CXXFLAGS="-O3 -march=native $AVX2_FLAG -std=c++17 -fPIC -shared"

# Add OpenMP if available
if $ENABLE_OPENMP; then
    CXXFLAGS="$CXXFLAGS -fopenmp"
fi

# GPU support
EXTRA_OBJS=""
if $BUILD_GPU; then
    # Check for CUDA
    if command -v nvcc &> /dev/null; then
        echo "  ✓ CUDA detected; building with GPU support"
        
        # Compile CUDA kernel
        nvcc -c cpp/backend_gpu.cu -o cpp/backend_gpu.o \
            --compiler-options "-fPIC" -O3
        
        CXXFLAGS="$CXXFLAGS -DENABLE_GPU -lcudart -lcublas"
        EXTRA_OBJS="cpp/backend_gpu.o"
    else
        echo "  ⚠ CUDA not found; GPU support disabled"
    fi
fi

# Build FFI wrapper
echo "[2] Building FFI wrapper..."
g++ -O3 -std=c++17 -fPIC -c cpp/apl_ffi.cpp -o cpp/apl_ffi.o

# Compile backend
echo "[3] Compiling main backend library..."
eval "g++ $CXXFLAGS -o backend_1bit.so cpp/backend_1bit.cpp cpp/apl_ffi.o $EXTRA_OBJS" 2>&1 | tee build.log || {
    echo "[ERROR] Compilation failed; trying with AVX2 disabled..."
    g++ -O3 -std=c++17 -fPIC -shared -fopenmp -o backend_1bit.so cpp/backend_1bit.cpp cpp/apl_ffi.o 2>&1 | tee build.log
}

if [ -f backend_1bit.so ]; then
    SIZE=$(du -h backend_1bit.so | cut -f1)
    echo "  ✓ Built: $REPO_ROOT/backend_1bit.so ($SIZE)"
else
    echo "  ✗ Build failed"
    exit 1
fi

# Build loader example
echo "[4] Building loader example..."
if g++ -O3 -std=c++17 -o cpp/loader_example cpp/loader_example.cpp -ldl 2>/dev/null; then
    echo "  ✓ Built: $REPO_ROOT/cpp/loader_example"
fi

# Cleanup temporary objects
rm -f cpp/*.o cpp/*.d

echo ""
echo "=========================================="
echo "✓ Build complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run accuracy validation: python tests/test_accuracy_validation.py"
echo "  2. Download and convert models: python scripts/convert_models_automated.py tinyllama"
echo "  3. Run inference: python easy_run.py --model tinyllama"
echo ""
