import numpy as np


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def self_att_numpy(Q, K, V, nhead=4):
    seq_len, total_d = Q.shape
    assert K.shape == (seq_len, total_d)
    assert V.shape == (seq_len, total_d)
    assert total_d % nhead == 0
    d = total_d // nhead
    Qh = Q.reshape(seq_len, nhead, d)
    Kh = K.reshape(seq_len, nhead, d)
    Vh = V.reshape(seq_len, nhead, d)
    out_heads = np.zeros_like(Qh)
    for h in range(nhead):
        scores = (Qh[:, h, :] @ Kh[:, h, :].T) / np.sqrt(d)
        att = softmax(scores, axis=-1)
        out_heads[:, h, :] = att @ Vh[:, h, :]
    out = out_heads.reshape(seq_len, total_d)
    return out


def ffn_numpy(X, W1, B1, W2, B2):
    H = np.maximum(0, X @ W1 + B1)
    Z = H @ W2 + B2
    return Z


def test_self_att_shape():
    seq_len = 10
    d_model = 32
    nhead = 4
    X = np.random.randn(seq_len, d_model)
    WQ = np.random.randn(d_model, d_model)
    WK = np.random.randn(d_model, d_model)
    WV = np.random.randn(d_model, d_model)
    Q = X @ WQ
    K = X @ WK
    V = X @ WV
    out = self_att_numpy(Q, K, V, nhead=nhead)
    assert out.shape == (seq_len, d_model)


def test_ffn_shape():
    seq_len = 7
    d_model = 16
    hidden = 64
    X = np.random.randn(seq_len, d_model)
    W1 = np.random.randn(d_model, hidden)
    B1 = np.random.randn(hidden)
    W2 = np.random.randn(hidden, d_model)
    B2 = np.random.randn(d_model)
    out = ffn_numpy(X, W1, B1, W2, B2)
    assert out.shape == (seq_len, d_model)
