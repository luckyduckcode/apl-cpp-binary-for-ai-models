#!/usr/bin/env python3
"""Cross-platform backend build helper.

Detects the current OS and runs the correct build script for the platform.
Usage: python scripts/build_backend.py
"""
import os
import platform
import subprocess
import sys
from pathlib import Path


def run_powershell_script(script_path: Path) -> int:
    print(f"Running PowerShell script: {script_path}")
    return subprocess.run(["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", str(script_path)]).returncode


def run_bash_script(script_path: Path) -> int:
    print(f"Running shell script: {script_path}")
    return subprocess.run(["bash", str(script_path)]).returncode


def main():
    repo_root = Path(__file__).resolve().parent.parent
    sys_platform = sys.platform
    print(f"Detected platform: {sys_platform}")

    if sys_platform.startswith("win"):
        script = repo_root / 'scripts' / 'build_backend_windows.ps1'
        if not script.exists():
            print("Windows build script not found:", script)
            return 1
        return run_powershell_script(script)
    elif sys_platform == 'darwin':
        script = repo_root / 'scripts' / 'build_backend_mac.sh'
        if not script.exists():
            print("macOS build script not found:", script)
            return 1
        return run_bash_script(script)
    else:
        # assume POSIX/Linux
        script = repo_root / 'scripts' / 'build_backend.sh'
        if not script.exists():
            print("POSIX build script not found:", script)
            return 1
        return run_bash_script(script)


if __name__ == '__main__':
    raise SystemExit(main())
