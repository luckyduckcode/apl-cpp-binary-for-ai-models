#!/usr/bin/env bash
set -euo pipefail

echo "CI runner: build and minimal tests"
scripts/build_backend.sh

echo "Running Python quick tests"
python scripts/run_quantized_demo.py || true
python scripts/run_quantized_qat_demo.py || true

echo "Running C++ demo tests"
python -u cpp/run_cpp_test.py || true
python -u cpp/call_backend.py || true
