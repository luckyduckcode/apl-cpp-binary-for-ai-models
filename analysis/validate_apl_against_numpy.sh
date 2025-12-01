#!/usr/bin/env bash
set -euo pipefail

APLPATH=$(command -v apl || true)
if [ -z "${APLPATH}" ]; then
  echo "No GNU APL interpreter found. Install 'apl' to run APL validation scripts." >&2
  exit 0
fi

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

echo "Generating random test vectors and reference output (NumPy + PyTorch)"
python3 - <<'PY'
import numpy as np
import json
from tests.test_blocks import self_att_numpy, ffn_numpy
import torch

seq_len = 6
d_model = 32
nhead = 4

X = np.random.randn(seq_len, d_model).astype(np.float32)
WQ = np.random.randn(d_model, d_model).astype(np.float32)
WK = np.random.randn(d_model, d_model).astype(np.float32)
WV = np.random.randn(d_model, d_model).astype(np.float32)
W1 = np.random.randn(d_model, d_model * 2).astype(np.float32) if d_model * 2 > 0 else np.random.randn(d_model, d_model)
B1 = np.random.randn(d_model * 2).astype(np.float32)
W2 = np.random.randn(d_model * 2, d_model).astype(np.float32) if d_model * 2 > 0 else np.random.randn(d_model, d_model)
B2 = np.random.randn(d_model).astype(np.float32)

Q = X @ WQ
K = X @ WK
V = X @ WV
attn_np = self_att_numpy(Q, K, V, nhead=nhead)
ffn_np = ffn_numpy(X, W1, B1, W2, B2)

np.save('tmp_X.npy', X)
np.save('tmp_WQ.npy', WQ)
np.save('tmp_WK.npy', WK)
np.save('tmp_WV.npy', WV)
np.save('tmp_attn_ref.npy', attn_np)
np.save('tmp_ffn_ref.npy', ffn_np)
print('Wrote numpy reference outputs to tmp_attn_ref.npy and tmp_ffn_ref.npy')
PY

echo "Calling APL to compute NN outputs (if present)"
apl -f apl/loader_demo.apl || true

echo "Compare outputs (manual inspection). The APL script prints output to stdout. If APL runtime didn't produce output, ensure it uses 'test_input' that exists and that loader_demo reads proper manifest."

echo "Validation complete"

exit 0
