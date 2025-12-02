#!/usr/bin/env python3
"""
Quick End-to-End Test Runner

Validates the entire pipeline:
1. Build backend
2. Run accuracy validation
3. Download models
4. Run inference
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*80}")
    print(f"[TEST] {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, timeout=300)
        if result.returncode == 0:
            print(f"\n✓ {description} PASSED")
            return True
        else:
            print(f"\n✗ {description} FAILED (exit code: {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        print(f"\n✗ {description} TIMED OUT")
        return False
    except Exception as e:
        print(f"\n✗ {description} ERROR: {e}")
        return False


def main():
    repo_root = Path(__file__).resolve().parent.parent
    results = {}
    
    print(f"APL Quantized Backend - E2E Test Suite")
    print(f"Repo: {repo_root}")
    
    # Test 1: Build backend
    results['build'] = run_command(
        ['python', str(repo_root / 'scripts' / 'build_backend.py')],
        "Build Backend (CPU)"
    )
    
    # Test 2: Accuracy validation
    results['accuracy'] = run_command(
        ['python', str(repo_root / 'tests' / 'test_accuracy_validation.py')],
        "Accuracy Validation Suite"
    )
    
    # Test 3: Download and convert TinyLlama
    results['convert_tinyllama'] = run_command(
        ['python', str(repo_root / 'scripts' / 'convert_models_automated.py'), 
         'tinyllama', '--bits', '4'],
        "Convert TinyLlama (1.1B) to 4-bit"
    )
    
    # Test 4: Run basic inference
    if results['convert_tinyllama']:
        results['inference'] = run_command(
            ['python', str(repo_root / 'easy_run.py'), '--model', 'tinyllama'],
            "Run Inference with TinyLlama"
        )
    else:
        print(f"\n⊘ Skipping inference test (model conversion failed)")
        results['inference'] = False
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:20s} {status}")
    
    total = len(results)
    passed = sum(1 for p in results.values() if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print(f"{'='*80}\n")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
