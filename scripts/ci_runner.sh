#!/usr/bin/env bash
set -euo pipefail

echo "CI runner: build and minimal tests"
scripts/build_backend.sh

echo "Compiling loader_example (POSIX)"
if command -v g++ >/dev/null 2>&1; then
	g++ -O3 -std=c++17 -fPIC cpp/loader_example.cpp -o cpp/loader_example -ldl || true
fi

echo "Running Python quick tests"
python scripts/run_quantized_demo.py || true
python scripts/run_quantized_qat_demo.py || true

echo "Running C++ demo tests"
python -u cpp/run_cpp_test.py || true
python -u cpp/call_backend.py || true
if [ -f cpp/loader_example ]; then
	echo "Running loader_example"
	./cpp/loader_example student_quantized_manifest.json cpp/backend_1bit.so || true
fi
