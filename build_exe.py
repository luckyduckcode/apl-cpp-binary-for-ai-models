#!/usr/bin/env python3
"""
Build executable for APL Chat Interface using PyInstaller
Supports both Flask and Gradio backends
"""

import os
import sys
import subprocess
from pathlib import Path

def build_exe():
    """Build the executable using PyInstaller"""
    
    workspace = Path(__file__).parent
    
    # PyInstaller command for standalone executable
    pyinstaller_args = [
        'pyinstaller',
        '--name', 'APL-Chat',
        '--onefile',  # Single executable file
        '--windowed',  # GUI mode (no console window)
        '--add-data', f"{workspace / 'apl_chat.html'};.",  # Include web UI
        '--hidden-import=flask',
        '--hidden-import=torch',
        '--hidden-import=transformers',
        '--hidden-import=numpy',
        '--hidden-import=gradio',
        '--hidden-import=huggingface_hub',
        '--collect-all=gradio',  # Include all Gradio assets
        '--collect-all=transformers',
        '--hidden-import=_socket',
        '--hidden-import=_ssl',
        '--hidden-import=ssl',
        '--distpath', str(workspace / 'dist'),
        '--workpath', str(workspace / 'build'),
        str(workspace / 'launch_chat.py'),
    ]
    
    print("=" * 70)
    print("Building APL Chat Interface Executable")
    print("=" * 70)
    print(f"\nOutput directory: {workspace / 'dist'}")
    print(f"This may take 2-5 minutes...\n")
    
    try:
        result = subprocess.run(
            pyinstaller_args,
            cwd=str(workspace),
            check=False
        )
        
        if result.returncode == 0:
            exe_path = workspace / 'dist' / 'APL-Chat.exe'
            if exe_path.exists():
                print("\n" + "=" * 70)
                print("BUILD SUCCESSFUL!")
                print("=" * 70)
                print(f"\nExecutable created: {exe_path}")
                print(f"File size: {exe_path.stat().st_size / (1024**2):.1f} MB")
                print("\nTo run the chat interface:")
                print(f"  {exe_path}")
                print("\nOr double-click the file from File Explorer")
                return True
            else:
                print("\nBuild completed but executable not found")
                return False
        else:
            print("\nBuild failed with PyInstaller")
            return False
            
    except Exception as e:
        print(f"\nError during build: {e}")
        return False

def build_exe_with_console():
    """Build executable WITH console window (for debugging)"""
    
    workspace = Path(__file__).parent
    
    pyinstaller_args = [
        'pyinstaller',
        '--name', 'APL-Chat-Console',
        '--onefile',
        '--add-data', f"{workspace / 'apl_chat.html'};.",
        '--hidden-import=flask',
        '--hidden-import=torch',
        '--hidden-import=transformers',
        '--hidden-import=numpy',
        '--hidden-import=gradio',
        '--hidden-import=huggingface_hub',
        '--collect-all=gradio',
        '--collect-all=transformers',
        '--hidden-import=_socket',
        '--hidden-import=_ssl',
        '--hidden-import=ssl',
        '--distpath', str(workspace / 'dist'),
        '--workpath', str(workspace / 'build'),
        str(workspace / 'launch_chat.py'),
    ]
    
    print("=" * 70)
    print("Building APL Chat Interface Executable (with console)")
    print("=" * 70)
    
    try:
        result = subprocess.run(
            pyinstaller_args,
            cwd=str(workspace),
            check=False
        )
        
        if result.returncode == 0:
            exe_path = workspace / 'dist' / 'APL-Chat-Console.exe'
            if exe_path.exists():
                print("\nBUILD SUCCESSFUL!")
                print(f"Executable: {exe_path}")
                return True
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--console':
        success = build_exe_with_console()
    else:
        success = build_exe()
    
    sys.exit(0 if success else 1)
