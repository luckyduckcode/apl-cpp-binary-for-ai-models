#!/usr/bin/env python3
"""
APL Chat Launcher - Simple startup script

Automatically installs dependencies and launches the chat interface.
"""

import subprocess
import sys
from pathlib import Path
import webbrowser
import time
import platform
import socket
import os

def check_python_version():
    """Check Python version."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    print(f"✓ Python {sys.version.split()[0]}")

def is_port_in_use(port=5000):
    """Check if port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        result = s.connect_ex(('localhost', port))
        return result == 0

def kill_existing_server():
    """Kill any existing Flask server process."""
    try:
        if platform.system() == 'Windows':
            os.system('taskkill /f /im python.exe /fi "WINDOWTITLE eq*Flask*" 2>nul')
    except:
        pass

def install_dependencies():
    """Install required packages."""
    print("\n📦 Checking dependencies...")
    
    required_packages = {
        'flask': 'Flask',
        'flask_cors': 'Flask-CORS',
        'torch': 'PyTorch',
        'transformers': 'Transformers',
    }
    
    missing = []
    
    for import_name, display_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ✗ {display_name} - installing...")
            missing.append(display_name)
    
    if missing:
        print(f"\nInstalling missing packages: {', '.join(missing)}")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + 
                      ['flask', 'flask-cors', 'torch', 'transformers'])

def check_models():
    """Check if quantized models exist."""
    print("\n🎯 Checking quantized models...")
    
    models_path = Path("models")
    required_models = [
        "tinyllama_manifest_q4.json",
        "mistral-7b_manifest_q4.json",
        "mistral-7b-instruct_manifest_q4.json",
    ]
    
    all_found = True
    for model in required_models:
        model_path = models_path / model
        if model_path.exists():
            size_kb = model_path.stat().st_size / 1024
            print(f"  ✓ {model} ({size_kb:.0f} KB)")
        else:
            print(f"  ✗ {model} - NOT FOUND (will download on first use)")
            all_found = False
    
    return all_found

def launch_server():
    """Launch the chat server."""
    print("\n🚀 Starting APL Chat Server...")
    print("=" * 60)
    
    # Kill any existing instances first
    kill_existing_server()
    
    # Check if port is in use
    if is_port_in_use(5000):
        print("\n⚠️  Port 5000 already in use. Attempting to clear...")
        kill_existing_server()
        time.sleep(1)
    
    print("\n📱 Launching Flask server...")
    print("Open browser at: http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    
    # Try to open browser after a short delay
    def open_browser():
        time.sleep(2)
        try:
            webbrowser.open('http://localhost:5000')
        except:
            pass
    
    import threading
    threading.Thread(target=open_browser, daemon=True).start()
    
    try:
        # Run in foreground mode (don't auto-restart)
        proc = subprocess.Popen([sys.executable, "apl_chat_server.py"])
        proc.wait()  # Wait for process to finish
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        try:
            proc.terminate()
        except:
            pass

def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("🚀 APL Chat - Quantized Model Interface")
    print("=" * 60)
    
    check_python_version()
    install_dependencies()
    check_models()
    
    # Auto-launch Flask server
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
        sys.exit(1)

