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
        '--onefile',
        '--console',  # Show console window for debugging
        '--name', 'APL-Chat',
        '--add-data', f"{workspace / 'apl_chat.html'};.",
        '--hidden-import=flask',
        '--hidden-import=torch',
        '--hidden-import=transformers',
        '--hidden-import=numpy',
        '--collect-all=flask',
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

