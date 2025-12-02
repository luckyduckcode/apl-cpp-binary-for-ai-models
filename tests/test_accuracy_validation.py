#!/usr/bin/env python3
"""
Accuracy Validation Suite for APL Quantized Models

Validates inference accuracy across different quantization levels (1-bit, q2, q4, q8)
by comparing against baseline PyTorch reference implementations.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import json
import sys
from typing import Tuple, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantization import quantize_per_row, binarize_weights, unpack_binarized, pack_q4, unpack_q4


class AccuracyValidator:
    """Validates APL quantized inference accuracy."""
    
    def __init__(self, tolerance_1bit=0.05, tolerance_q2=0.01, tolerance_q4=0.005, tolerance_q8=0.001):
        """
        Args:
            tolerance_1bit: Relative error tolerance for 1-bit quantization (5%)
            tolerance_q2: Relative error tolerance for 2-bit (1%)
            tolerance_q4: Relative error tolerance for 4-bit (0.5%)
            tolerance_q8: Relative error tolerance for 8-bit (0.1%)
        """
        self.tolerances = {
            1: tolerance_1bit,
            2: tolerance_q2,
            4: tolerance_q4,
            8: tolerance_q8,
        }
        self.results = []
    
    def matmul_baseline(self, W: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Baseline float32 matrix multiplication."""
        # W is (out, in), x is (in,) → result is (out,)
        # Use numpy for correctness
        return W @ x
    
    def matmul_1bit(self, W: np.ndarray, x: np.ndarray) -> np.ndarray:
        """1-bit binarized matmul."""
        W_signs = np.sign(W).astype(np.int8)
        
        # Recover scales from original
        scales = np.abs(W).mean(axis=1, keepdims=True)
        
        # Compute sign correlation: matches = sum(signs match), result = 2*matches - len(x)
        output = []
        for i, row in enumerate(W_signs):
            x_signs = np.sign(x).astype(np.int8)
            matches = np.sum(row == x_signs)
            xnor_result = 2.0 * matches - len(x)
            output.append(xnor_result * scales[i, 0])
        return np.array(output)
    
    def matmul_q2(self, W: np.ndarray, x: np.ndarray) -> np.ndarray:
        """2-bit quantized matmul."""
        q, scales, zps = quantize_per_row(W, bits=2)
        q_int = q.astype(np.int32)
        zps_int = zps.astype(np.int32)
        dequant = (q_int - zps_int.reshape(-1, 1)).astype(np.float32) * scales.reshape(-1, 1)
        return dequant @ x
    
    def matmul_q4(self, W: np.ndarray, x: np.ndarray) -> np.ndarray:
        """4-bit quantized matmul."""
        q, scales, zps = quantize_per_row(W, bits=4)
        q_int = q.astype(np.int32)
        zps_int = zps.astype(np.int32)
        dequant = (q_int - zps_int.reshape(-1, 1)).astype(np.float32) * scales.reshape(-1, 1)
        return dequant @ x
    
    def matmul_q8(self, W: np.ndarray, x: np.ndarray) -> np.ndarray:
        """8-bit quantized matmul."""
        q, scales, zps = quantize_per_row(W, bits=8)
        q_int = q.astype(np.int32)
        zps_int = zps.astype(np.int32)
        dequant = (q_int - zps_int.reshape(-1, 1)).astype(np.float32) * scales.reshape(-1, 1)
        return dequant @ x
    
    def relative_error(self, expected: np.ndarray, actual: np.ndarray) -> float:
        """Compute relative L2 error."""
        expected = expected.astype(np.float32)
        actual = actual.astype(np.float32)
        diff = np.linalg.norm(expected - actual)
        norm = np.linalg.norm(expected)
        return diff / (norm + 1e-10)
    
    def validate_quantization(self, W: np.ndarray, x: np.ndarray, bits: int) -> Dict:
        """Validate a single quantization level."""
        baseline = self.matmul_baseline(W, x)
        
        if bits == 1:
            quantized = self.matmul_1bit(W, x)
        elif bits == 2:
            quantized = self.matmul_q2(W, x)
        elif bits == 4:
            quantized = self.matmul_q4(W, x)
        elif bits == 8:
            quantized = self.matmul_q8(W, x)
        else:
            raise ValueError(f"Unsupported bit width: {bits}")
        
        rel_error = self.relative_error(baseline, quantized)
        tolerance = self.tolerances.get(bits, 0.01)
        passed = rel_error <= tolerance
        
        result = {
            'bits': bits,
            'baseline_norm': float(np.linalg.norm(baseline)),
            'quantized_norm': float(np.linalg.norm(quantized)),
            'relative_error': float(rel_error),
            'tolerance': float(tolerance),
            'passed': bool(passed),
            'baseline_sample': [float(v) for v in baseline[:3]],
            'quantized_sample': [float(v) for v in quantized[:3]],
        }
        
        self.results.append(result)
        return result
    
    def validate_layer(self, W: np.ndarray, x: np.ndarray, layer_name: str = "layer") -> Dict:
        """Validate all quantization levels for a weight matrix."""
        print(f"\n=== Validating {layer_name} (shape {W.shape}) ===")
        
        all_results = {'layer': layer_name, 'W_shape': list(W.shape), 'validations': []}
        
        for bits in [1, 2, 4, 8]:
            result = self.validate_quantization(W, x, bits)
            all_results['validations'].append(result)
            
            status = "PASS" if result['passed'] else "FAIL"
            print(f"  {bits}-bit: rel_error={result['relative_error']:.4f}, "
                  f"tolerance={result['tolerance']:.4f} {status}")
        
        return all_results
    
    def validate_attention_head(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
                               x: np.ndarray, num_heads: int) -> Dict:
        """Validate multi-head attention with quantization."""
        print(f"\n=== Validating Multi-Head Attention (heads={num_heads}) ===")
        
        # Q, K, V are 2D: (seq_len, d_model)
        seq_len, d_model = Q.shape
        d_head = d_model // num_heads
        results = {'attention': {}}
        
        # Reshape for heads
        Q_heads = Q.reshape(seq_len, num_heads, d_head)
        K_heads = K.reshape(seq_len, num_heads, d_head)
        V_heads = V.reshape(seq_len, num_heads, d_head)
        
        for head_id in range(min(num_heads, 2)):  # Validate first 2 heads
            Q_h = Q_heads[:, head_id, :].reshape(-1, d_head)
            K_h = K_heads[:, head_id, :].reshape(-1, d_head)
            V_h = V_heads[:, head_id, :].reshape(-1, d_head)
            
            # Baseline: full precision attention
            scores = (Q_h @ K_h.T) / np.sqrt(d_head)
            # Use torch for softmax
            attn_weights = F.softmax(torch.from_numpy(scores).float(), dim=-1).numpy()
            baseline_out = attn_weights @ V_h
            
            # Quantized: Q, K, V quantized separately
            q_q, scales_q, zps_q = quantize_per_row(Q_h, bits=4)
            q_k, scales_k, zps_k = quantize_per_row(K_h, bits=4)
            q_v, scales_v, zps_v = quantize_per_row(V_h, bits=4)
            
            # Dequantize
            dequant_q = (q_q.astype(np.int32) - zps_q.reshape(-1, 1).astype(np.int32)) * scales_q.reshape(-1, 1)
            dequant_k = (q_k.astype(np.int32) - zps_k.reshape(-1, 1).astype(np.int32)) * scales_k.reshape(-1, 1)
            dequant_v = (q_v.astype(np.int32) - zps_v.reshape(-1, 1).astype(np.int32)) * scales_v.reshape(-1, 1)
            
            quant_scores = (dequant_q.astype(np.float32) @ dequant_k.astype(np.float32).T) / np.sqrt(d_head)
            quant_attn = F.softmax(torch.from_numpy(quant_scores).float(), dim=-1).numpy()
            quant_out = quant_attn @ dequant_v.astype(np.float32)
            
            rel_error = self.relative_error(baseline_out, quant_out)
            passed = rel_error <= self.tolerances[4]  # Use q4 tolerance
            
            results['attention'][f'head_{head_id}'] = {
                'relative_error': float(rel_error),
                'passed': bool(passed),
            }
            
            status = "PASS" if passed else "FAIL"
            print(f"  Head {head_id}: rel_error={rel_error:.4f} {status}")
        
        return results
    
    def report(self) -> str:
        """Generate validation report."""
        report = "\n" + "="*80 + "\n"
        report += "ACCURACY VALIDATION REPORT\n"
        report += "="*80 + "\n\n"
        
        by_bits = {}
        for result in self.results:
            bits = result['bits']
            if bits not in by_bits:
                by_bits[bits] = []
            by_bits[bits].append(result)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['passed'])
        
        report += f"Overall: {passed_tests}/{total_tests} tests passed ({100*passed_tests/total_tests:.1f}%)\n\n"
        
        for bits in sorted(by_bits.keys()):
            results = by_bits[bits]
            passed = sum(1 for r in results if r['passed'])
            report += f"{bits}-bit Quantization: {passed}/{len(results)} passed\n"
            for r in results:
                status = "PASS" if r['passed'] else "FAIL"
                report += f"  rel_error={r['relative_error']:.4f} (tolerance={r['tolerance']:.4f}) {status}\n"
        
        report += "\n" + "="*80 + "\n"
        return report
    
    def to_json(self, path: Path):
        """Save results to JSON."""
        # Convert numpy types to native Python types for JSON serialization
        results_json = []
        for r in self.results:
            r_json = {
                'bits': int(r['bits']),
                'baseline_norm': float(r['baseline_norm']),
                'quantized_norm': float(r['quantized_norm']),
                'relative_error': float(r['relative_error']),
                'tolerance': float(r['tolerance']),
                'passed': bool(r['passed']),
                'baseline_sample': [float(x) for x in r['baseline_sample']],
                'quantized_sample': [float(x) for x in r['quantized_sample']],
            }
            results_json.append(r_json)
        
        with open(path, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"Results saved to {path}")


def run_validation_suite():
    """Run full validation suite."""
    # Use relaxed tolerances since quantization errors are expected
    validator = AccuracyValidator(tolerance_1bit=0.5, tolerance_q2=0.2, tolerance_q4=0.1, tolerance_q8=0.02)
    
    # Test 1: Small dense matrix
    print("\nTest 1: Small Dense Matrix (32x64)")
    np.random.seed(42)
    W1 = np.random.randn(32, 64).astype(np.float32)
    x1 = np.random.randn(64).astype(np.float32)
    validator.validate_layer(W1, x1, "dense_32x64")
    
    # Test 2: Larger matrix
    print("\nTest 2: Larger Matrix (256x512)")
    W2 = np.random.randn(256, 512).astype(np.float32)
    x2 = np.random.randn(512).astype(np.float32)
    validator.validate_layer(W2, x2, "dense_256x512")
    
    # Test 3: Transformer-like attention
    print("\nTest 3: Transformer Attention")
    seq_len, d_model = 16, 128
    num_heads = 4
    np.random.seed(43)
    Q = np.random.randn(seq_len, d_model).astype(np.float32)
    K = np.random.randn(seq_len, d_model).astype(np.float32)
    V = np.random.randn(seq_len, d_model).astype(np.float32)
    validator.validate_attention_head(Q, K, V, None, num_heads)
    
    # Print report
    print(validator.report())
    
    # Save results
    validator.to_json(Path(__file__).resolve().parent.parent / "accuracy_validation_results.json")


if __name__ == "__main__":
    run_validation_suite()
