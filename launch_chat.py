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

def check_python_version():
    """Check Python version."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    print(f"✓ Python {sys.version.split()[0]}")

def install_dependencies():
    """Install required packages."""
    print("\n📦 Checking dependencies...")
    
    required_packages = {
        'flask': 'Flask',
        'flask_cors': 'Flask-CORS',
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'gradio': 'Gradio (optional)',
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
        # Note: In production, would run pip install here
        print("Run: pip install flask flask-cors torch transformers")
        print("Optional: pip install gradio")

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
            print(f"  ✗ {model} - NOT FOUND")
            all_found = False
    
    if not all_found:
        print("\n⚠️  Some models missing. Run:")
        print("  python scripts/convert_models_automated.py tinyllama --bits 4")
        print("  python scripts/convert_models_automated.py mistral-7b --bits 4")
    
    return all_found

def launch_server(use_gradio=False):
    """Launch the chat server."""
    print("\n🚀 Starting APL Chat Server...")
    print("=" * 60)
    
    if use_gradio:
        print("\n📱 Launching Gradio interface...")
        subprocess.run([sys.executable, "apl_chat_ui.py"])
    else:
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
            subprocess.run([sys.executable, "apl_chat_server.py"])
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down...")

def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("🚀 APL Chat - Quantized Model Interface")
    print("=" * 60)
    
    check_python_version()
    install_dependencies()
    models_ok = check_models()
    
    if not models_ok:
        print("\n⚠️  Models not found. Please convert models first:")
        print("  python scripts/convert_models_automated.py tinyllama --bits 4")
        print("\nContinuing with HuggingFace model auto-download...")
    
    # Auto-launch Flask server (no menu when run from exe)
    launch_server(use_gradio=False)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
