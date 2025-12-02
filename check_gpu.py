#!/usr/bin/env python3
"""Check NVIDIA GPU availability."""

import torch

print("=" * 60)
print("GPU Detection Report")
print("=" * 60)

print(f"\nCUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Devices: {torch.cuda.device_count()}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {props.total_memory / 1e9:.1f}GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
else:
    print("\n[WARNING] No NVIDIA GPU detected!")
    print("Make sure:")
    print("  1. NVIDIA GPU is installed")
    print("  2. NVIDIA drivers are installed")
    print("  3. CUDA Toolkit is installed")
    print("  4. cuDNN is installed (optional but recommended)")
    
print("\n" + "=" * 60)
