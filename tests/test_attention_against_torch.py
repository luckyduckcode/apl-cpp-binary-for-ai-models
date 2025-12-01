import numpy as np
import torch
import torch.nn.functional as F
from tests.test_blocks import self_att_numpy, ffn_numpy


def to_torch(x):
    return torch.tensor(x, dtype=torch.float32)


def test_self_att_against_torch():
    torch.manual_seed(0)
    seq_len = 6
    d_model = 32
    nhead = 4
    X = np.random.randn(seq_len, d_model).astype(np.float32)
    # Random weight matrices
    WQ = np.random.randn(d_model, d_model).astype(np.float32)
    WK = np.random.randn(d_model, d_model).astype(np.float32)
    WV = np.random.randn(d_model, d_model).astype(np.float32)

    # NumPy attention implementation
    Q = X @ WQ
    K = X @ WK
    V = X @ WV
    out_np = self_att_numpy(Q, K, V, nhead=nhead)

    # Torch attention via MultiheadAttention expects shape: (seq_len, batch, embed_dim)
    mha = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=False)
    # Build in_proj_weight combo: MultiheadAttention uses concatenated in_proj_weight (Wq;Wk;Wv)
    # We'll set query, key, value linear transformations via a manual weight construct
    q = to_torch(Q).T.unsqueeze(1)  # shape (d_model, batch=1, seq) - but batch_first is False => (seq, batch, embed)
    # We instead do a lower-level computation: run MultiheadAttention with assigned weights
    mha.in_proj_weight.data.copy_(torch.from_numpy(np.concatenate([WQ.T, WK.T, WV.T], axis=0)))
    # zero bias
    mha.in_proj_bias.data.zero_()
    mha.out_proj.weight.data.copy_(torch.eye(d_model))
    mha.out_proj.bias.data.zero_()

    q_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    k_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    v_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        out_t, _ = mha(q_t, k_t, v_t)
    out_t = out_t.squeeze(1).detach().numpy()

    # Compare outputs (relative small tolerance)
    diff = np.max(np.abs(out_np - out_t))
    assert diff < 1e-4, f"Max diff {diff} exceeds tolerance"


def test_ffn_against_torch():
    torch.manual_seed(0)
    seq_len = 8
    d_model = 16
    hidden = 64
    X = np.random.randn(seq_len, d_model).astype(np.float32)
    W1 = np.random.randn(d_model, hidden).astype(np.float32)
    B1 = np.random.randn(hidden).astype(np.float32)
    W2 = np.random.randn(hidden, d_model).astype(np.float32)
    B2 = np.random.randn(d_model).astype(np.float32)

    out_np = ffn_numpy(X, W1, B1, W2, B2)

    # PyTorch equivalent
    lin1 = torch.nn.Linear(d_model, hidden)
    lin2 = torch.nn.Linear(hidden, d_model)
    with torch.no_grad():
        lin1.weight.data.copy_(torch.from_numpy(W1.T))
        lin1.bias.data.copy_(torch.from_numpy(B1))
        lin2.weight.data.copy_(torch.from_numpy(W2.T))
        lin2.bias.data.copy_(torch.from_numpy(B2))

    xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        out_t = lin2(F.relu(lin1(xt))).detach().numpy()

    diff = np.max(np.abs(out_np - out_t))
    assert diff < 1e-4, f"Max diff {diff} exceeds tolerance"
