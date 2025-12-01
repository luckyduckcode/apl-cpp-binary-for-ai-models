import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def quantize_weights(weights, bits=4):
    """
    Simple post-training quantization to bits precision.
    """
    min_val = np.min(weights)
    max_val = np.max(weights)
    scale = (max_val - min_val) / (2**bits - 1)
    zero_point = np.round(-min_val / scale)
    
    quantized = np.round(weights / scale + zero_point)
    quantized = np.clip(quantized, 0, 2**bits - 1).astype(np.uint8 if bits <= 8 else np.uint16)
    
    return quantized, scale, zero_point


def pack_bits(bit_arr):
    """Pack a flat array of {0,1} bits into uint8 bytes using numpy.packbits.
       Accepts a 1D or 2D boolean/uint8 array and returns packed uint8 array.
    """
    # ensure uint8 bits 0/1
    arr = np.asarray(bit_arr, dtype=np.uint8)
    if arr.ndim == 2:
        # pack each row separately
        packed_rows = [np.packbits(row) for row in arr]
        return np.vstack(packed_rows)
    else:
        return np.packbits(arr)


def binarize_weights(weights, per_channel_axis=0):
    """
    Binarize weights to sign ±1 per output channel and compute per-channel scale.
    Returns: packed_bytes (np.uint8), scales (float32), shape metadata.
    If weights shape is (out, in), per_channel_axis=0 computes scales per row.
    """
    W = np.array(weights, copy=True)
    if W.ndim != 2:
        raise ValueError("binarize_weights expects a 2D weight matrix")
    # scale per-channel: axis==0 -> per-output row for shape (out, in)
    if per_channel_axis == 0:
        scales = np.mean(np.abs(W), axis=1)
    else:
        scales = np.mean(np.abs(W), axis=0)
    scales = np.where(scales == 0, 1.0, scales).astype(np.float32)
    # sign: +1 -> 1, -1 -> 0 for bit packing
    sign = np.sign(W)
    sign[sign == 0] = 1
    bits = (sign > 0).astype(np.uint8)
    # pack bits per row
    packed = pack_bits(bits)
    metadata = {"shape": W.shape, "per_channel_axis": int(per_channel_axis)}
    return packed, scales, metadata


def unpack_binarized(packed, scales, shape, per_channel_axis=0):
    """Unpack packed bits and dequantize to float matrix using scales.
       packed: 2D array where each row is packbits of the original bit row
       scales: 1D array length = out_features
       shape: (out, in)
    """
    out, _in = shape
    # unpackbits returns shape (bytes*8, ) - need to trim
    rows = []
    for i, prow in enumerate(packed):
        bits = np.unpackbits(prow)
        bits = bits[:_in]
        sign = np.where(bits == 1, 1.0, -1.0)
        if per_channel_axis == 0:
            # scales per output row
            row = sign * float(scales[i])
        else:
            # scales per input column
            # sign has length _in, scales has length _in
            row = sign * np.array(scales, dtype=np.float32)
        rows.append(row)
    return np.vstack(rows)


# ------------------------------
# QAT utilities (PyTorch)
# ------------------------------
class FakeQuant1bitFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, per_channel_axis: int = 0):
        # weight: (out, in) or (out, in) shaped tensor
        W = weight
        if W.ndim != 2:
            raise ValueError("FakeQuant1bitFunction expects 2D weight")
        # per-channel scale: mean absolute value per output channel
        scales = torch.mean(torch.abs(W), dim=1, keepdim=True)
        scales = torch.where(scales == 0, torch.tensor(1.0, device=W.device, dtype=W.dtype), scales)
        sign = torch.sign(W)
        sign[sign == 0] = 1.0
        q = sign * scales
        # store nothing (STE) - gradients are passed-through
        return q

    @staticmethod
    def backward(ctx, grad_output):
        # Straight through estimator: gradients passthrough
        return grad_output, None


class FakeQuant1bit(nn.Module):
    """Module wrapper that applies FakeQuant1bit to a weight tensor.
    This is useful when replacing Linear modules with QAT variants.
    The function returns the dequantized float weight as sign*scale.
    """
    def __init__(self, per_channel_axis: int = 0):
        super().__init__()
        self.per_channel_axis = per_channel_axis

    def forward(self, weight: torch.Tensor):
        return FakeQuant1bitFunction.apply(weight, self.per_channel_axis)


class QATLinear(nn.Module):
    """Replacement for nn.Linear that quantizes the weight with FakeQuant1bit during forward.
    Only weights are quantized; biases remain in FP32.
    """
    def __init__(self, in_features, out_features, bias=True, per_channel_axis=0, learnable_scale=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.fakequant = FakeQuant1bit(per_channel_axis)
        # optional learnable scale per output channel (out_features x 1)
        self.learnable_scale = learnable_scale
        if self.learnable_scale:
            self.scale = nn.Parameter(torch.ones(out_features, 1))
        else:
            self.register_parameter('scale', None)

    def forward(self, input: torch.Tensor):
        if self.learnable_scale and self.scale is not None:
            sign = torch.sign(self.weight)
            sign[sign == 0] = 1.0
            qweight = sign * (self.scale)
        else:
            qweight = self.fakequant(self.weight)
        return F.linear(input, qweight, self.bias)


def replace_linears_with_qat(module: nn.Module, per_channel_axis=0):
    """Recursively replace nn.Linear instances with QATLinear preserving state dicts.
    Returns the number of replacements performed.
    """
    replacements = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            new = QATLinear(child.in_features, child.out_features, bias=(child.bias is not None), per_channel_axis=per_channel_axis)
            # copy weights
            new.weight.data = child.weight.data.clone()
            if child.bias is not None:
                new.bias.data = child.bias.data.clone()
            # init learnable scale to mean abs over each output row
            if hasattr(new, 'scale') and new.scale is not None:
                with torch.no_grad():
                    W = child.weight.data
                    scales = torch.mean(torch.abs(W), dim=1, keepdim=True)
                    scales[scales == 0] = 1.0
                    new.scale.data = scales.clone()
            setattr(module, name, new)
            replacements += 1
        else:
            replacements += replace_linears_with_qat(child, per_channel_axis)
    return replacements


# Example usage
if __name__ == "__main__":
    # Dummy weights
    weights = np.random.randn(100, 100)
    q_weights, scale, zp = quantize_weights(weights, bits=4)
    print("Quantized shape:", q_weights.shape)
    print("Scale:", scale, "Zero point:", zp)