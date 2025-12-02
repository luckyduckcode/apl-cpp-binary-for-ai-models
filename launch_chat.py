#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APL Chat Launcher - Simple startup script

Automatically installs dependencies and launches the chat interface.
"""

import subprocess
import sys
from pathlib import Path
import time
import platform
import socket
import os

# Force UTF-8 output for emojis and special characters
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def install_dependencies_first():
    """Install required packages BEFORE any imports."""
    print("📦 Installing dependencies...")
    
    required_packages = [
        'flask',
        'flask-cors', 
        'torch',
        'transformers',
        'accelerate',
        'bitsandbytes',
        'numpy',
    ]
    
    # Use pip directly to install all at once
    cmd = [sys.executable, '-m', 'pip', 'install', '--quiet'] + required_packages
    subprocess.run(cmd, capture_output=True)
    print("✓ Dependencies ready\n")

def check_python_version():
    """Check Python version."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    print(f"✓ Python {sys.version.split()[0]}")

def is_port_in_use(port=5000):
    """Check if port is already in use."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(('localhost', port))
            return result == 0
    except:
        return False

def kill_existing_server():
    """Kill any existing Flask server process."""
    try:
        if platform.system() == 'Windows':
            os.system('taskkill /f /im python.exe /fi "WINDOWTITLE eq*apl*" 2>nul')
            os.system('taskkill /f /im python.exe /fi "WINDOWTITLE eq*Flask*" 2>nul')
    except:
        pass

def launch_server():
    """Launch the chat server."""
    print("\n" + "=" * 60)
    print("🚀 Starting APL Chat Server...")
    print("=" * 60)
    
    # Kill any existing instances first (Windows only has multi-instance issues)
    if platform.system() == 'Windows' and is_port_in_use(5000):
        print("⚠️  Port 5000 in use. Clearing...")
        kill_existing_server()
        time.sleep(2)
    
    print("\n📱 Flask Server Starting...")
    print("🌐 Open http://localhost:5000 in your browser")
    print("⏸  Press Ctrl+C to stop\n")
    print("=" * 60)
    
    # Try to open browser after a short delay
    def open_browser():
        time.sleep(3)
        try:
            import webbrowser
            webbrowser.open('http://localhost:5000')
        except:
            pass
    
    import threading
    threading.Thread(target=open_browser, daemon=True).start()
    
    try:
        # Import and run server directly (works with PyInstaller bundles)
        try:
            from apl_chat_server import run_server
        except ImportError as import_err:
            print(f"❌ Failed to import server: {import_err}")
            print("Attempting direct execution...")
            raise
        
        run_server()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        time.sleep(5)
    finally:
        if platform.system() == 'Windows':
            kill_existing_server()

def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("🚀 APL Chat Interface - v1.0.0")
    print("=" * 60 + "\n")
    
    # CRITICAL: Install deps FIRST before any other imports
    install_dependencies_first()
    check_python_version()
    
    # Now we can safely launch
    launch_server()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        time.sleep(5)  # Keep window open to see error
        sys.exit(1)

