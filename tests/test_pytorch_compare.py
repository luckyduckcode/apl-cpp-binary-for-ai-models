import numpy as np
import torch
import math
import pytest


def softmax_np(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def self_att_numpy(Q, K, V, nhead=4):
    seq_len, total_d = Q.shape
    d = total_d // nhead
    Qh = Q.reshape(seq_len, nhead, d)
    Kh = K.reshape(seq_len, nhead, d)
    Vh = V.reshape(seq_len, nhead, d)
    out_heads = np.zeros_like(Qh)
    for h in range(nhead):
        scores = Qh[:, h, :] @ Kh[:, h, :].T
        scores = scores / math.sqrt(d)
        att = softmax_np(scores, axis=-1)
        out_heads[:, h, :] = att @ Vh[:, h, :]
    out = out_heads.reshape(seq_len, total_d)
    return out


def ffn_numpy(X, W1, B1, W2, B2):
    H = np.maximum(0, X @ W1 + B1)
    Z = H @ W2 + B2
    return Z


def torch_self_att(Q, K, V, nhead=4):
    # Q,K,V: seq_len x d
    seq_len, total_d = Q.shape
    d = total_d // nhead
    # convert to seq_len x nhead x d
    Qh = Q.reshape(seq_len, nhead, d)
    Kh = K.reshape(seq_len, nhead, d)
    Vh = V.reshape(seq_len, nhead, d)
    out_heads = torch.zeros_like(torch.from_numpy(Qh))
    for h in range(nhead):
        q = torch.from_numpy(Qh[:, h, :].copy())
        k = torch.from_numpy(Kh[:, h, :].copy())
        v = torch.from_numpy(Vh[:, h, :].copy())
        scores = q @ k.T
        scores = scores / math.sqrt(d)
        att = torch.softmax(scores, dim=-1)
        out = att @ v
        out_heads[:, h, :] = out
    out = out_heads.numpy().reshape(seq_len, total_d)
    return out


def torch_ffn(X, W1, B1, W2, B2):
    x = torch.from_numpy(X.copy())
    w1 = torch.from_numpy(W1.copy())
    b1 = torch.from_numpy(B1.copy())
    w2 = torch.from_numpy(W2.copy())
    b2 = torch.from_numpy(B2.copy())
    h = torch.relu(x @ w1 + b1)
    z = h @ w2 + b2
    return z.numpy()


def test_ffn_compare():
    seq_len = 8
    d_model = 32
    hidden = 64
    X = np.random.randn(seq_len, d_model).astype(np.float32)
    W1 = np.random.randn(d_model, hidden).astype(np.float32)
    B1 = np.random.randn(hidden).astype(np.float32)
    W2 = np.random.randn(hidden, d_model).astype(np.float32)
    B2 = np.random.randn(d_model).astype(np.float32)
    out_np = ffn_numpy(X, W1, B1, W2, B2)
    out_torch = torch_ffn(X, W1, B1, W2, B2)
    diff = np.max(np.abs(out_np - out_torch))
    # Small differences can occur due to float32 rounding differences between numpy and torch.
    assert diff < 1e-4, f"FFN mismatch: max diff {diff}"


def test_self_att_compare():
    seq_len = 6
    d_model = 32
    nhead = 4
    X = np.random.randn(seq_len, d_model).astype(np.float32)
    WQ = np.random.randn(d_model, d_model).astype(np.float32)
    WK = np.random.randn(d_model, d_model).astype(np.float32)
    WV = np.random.randn(d_model, d_model).astype(np.float32)
    Q = X @ WQ
    K = X @ WK
    V = X @ WV
    out_np = self_att_numpy(Q, K, V, nhead=nhead)
    out_torch = torch_self_att(Q, K, V, nhead=nhead)
    diff = np.max(np.abs(out_np - out_torch))
    # Small float rounding differences may exist; allow a small tolerance
    assert diff < 1e-4, f"Self-att mismatch: max diff {diff}"
