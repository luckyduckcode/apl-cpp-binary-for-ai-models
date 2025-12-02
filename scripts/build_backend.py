#!/usr/bin/env python3
"""Cross-platform backend build helper.

Detects the current OS and runs the correct build script for the platform.
Supports optional GPU (CUDA) acceleration.

Usage: 
  python scripts/build_backend.py              # CPU only
  python scripts/build_backend.py --gpu        # With CUDA support (if CUDA toolkit available)
"""
import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


def check_cuda_available() -> bool:
    """Check if CUDA toolkit is available."""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✓ CUDA toolkit detected: {result.stdout.split(chr(10))[0]}")
            return True
    except FileNotFoundError:
        pass
    return False


def run_powershell_script(script_path: Path, enable_gpu: bool = False) -> int:
    print(f"Running PowerShell script: {script_path}")
    cmd = ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", str(script_path)]
    if enable_gpu:
        cmd.extend(["-EnableGPU", "true"])
    return subprocess.run(cmd).returncode


def run_bash_script(script_path: Path, enable_gpu: bool = False) -> int:
    print(f"Running shell script: {script_path}")
    cmd = ["bash", str(script_path)]
    if enable_gpu:
        cmd.append("--enable-gpu")
    return subprocess.run(cmd).returncode


def main():
    parser = argparse.ArgumentParser(description="Cross-platform backend build helper")
    parser.add_argument('--gpu', action='store_true', help="Enable CUDA GPU support (if available)")
    parser.add_argument('--cuda-path', type=str, default=None, help="Path to CUDA toolkit installation")
    args = parser.parse_args()
    
    repo_root = Path(__file__).resolve().parent.parent
    sys_platform = sys.platform
    print(f"Detected platform: {sys_platform}")
    
    enable_gpu = args.gpu
    if enable_gpu:
        print("GPU support requested. Checking for CUDA...")
        if not check_cuda_available():
            print("  ⚠ CUDA toolkit not found. Building CPU-only version.")
            enable_gpu = False
        else:
            print("  Building with GPU support enabled")

    if sys_platform.startswith("win"):
        script = repo_root / 'scripts' / 'build_backend_windows.ps1'
        if not script.exists():
            print("Windows build script not found:", script)
            return 1
        return run_powershell_script(script, enable_gpu)
    elif sys_platform == 'darwin':
        script = repo_root / 'scripts' / 'build_backend_mac.sh'
        if not script.exists():
            print("macOS build script not found:", script)
            return 1
        return run_bash_script(script, enable_gpu)
    else:
        # assume POSIX/Linux
        script = repo_root / 'scripts' / 'build_backend.sh'
        if not script.exists():
            print("POSIX build script not found:", script)
            return 1
        return run_bash_script(script, enable_gpu)


if __name__ == '__main__':
    raise SystemExit(main())
