#!/usr/bin/env python3
"""
Simple PyInstaller build script for APL Chat Interface
Builds standalone executable with console for debugging
"""

import subprocess
import sys
from pathlib import Path

def main():
    workspace = Path(__file__).parent
    
    # Simplified PyInstaller command - show console to help debug issues
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--noconfirm',
        '--onedir', # Folder mode (faster startup) instead of --onefile
        '--console',  # Show console window for debugging
        '--name', 'APL-Chat',
        '--add-data', f"{workspace / 'apl_chat.html'};.",
        '--hidden-import=flask',
        '--hidden-import=werkzeug',
        '--hidden-import=jinja2',
        '--hidden-import=click',
        '--hidden-import=itsdangerous',
        '--hidden-import=flask_cors',
        '--hidden-import=torch',
        '--hidden-import=transformers',
        '--hidden-import=accelerate',
        '--hidden-import=bitsandbytes',
        '--hidden-import=numpy',
        '--collect-all=flask',
        '--collect-all=werkzeug',
        '--collect-all=flask_cors',
        '--collect-all=transformers',
        '--collect-all=bitsandbytes',
        # Exclude heavy unused modules to reduce size
        '--exclude-module=torch.test',
        # '--exclude-module=torch.distributed', # Required by torch.utils.data
        # '--exclude-module=torch.distributions',
        '--exclude-module=caffe2',
        # '--exclude-module=tkinter', # Required for splash screen
        '--exclude-module=matplotlib',
        '--exclude-module=scipy',
        '--exclude-module=pandas',
        '--exclude-module=nvidia.cublas.lib.cublasLt', # Sometimes duplicated
        str(workspace / 'launch_chat.py'),
    ]
    
    print("=" * 70)
    print("Building APL Chat Interface (.exe)")
    print("=" * 70)
    print("\nThis may take 2-5 minutes depending on your system...")
    print(f"Output: {workspace / 'dist' / 'APL-Chat.exe'}\n")
    
    try:
        result = subprocess.run(cmd, cwd=str(workspace))
        
        if result.returncode == 0:
            exe_path = workspace / 'dist' / 'APL-Chat.exe'
            if exe_path.exists():
                size_mb = exe_path.stat().st_size / (1024**2)
                print("\n" + "=" * 70)
                print("BUILD SUCCESSFUL!")
                print("=" * 70)
                print(f"\nExecutable: {exe_path}")
                print(f"Size: {size_mb:.1f} MB")
                print("\nTo run: APL-Chat.exe")
                print("Or double-click from File Explorer")
                print("\nNote: Console window shows logs for debugging")
                return 0
        
        print("\nBuild failed or executable not found")
        return 1
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())

