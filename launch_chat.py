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
import threading
import queue
import multiprocessing

# Force UTF-8 output for emojis and special characters
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def show_splash_screen(status_queue):
    """Show a simple Tkinter splash screen."""
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        return

    root = tk.Tk()
    root.overrideredirect(True)  # No window decorations
    
    # Center the window
    width = 400
    height = 150
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    # Style
    root.configure(bg='#1a1a1a')
    
    # Title
    label_title = tk.Label(root, text="APL Chat", font=("Segoe UI", 24, "bold"), fg="white", bg="#1a1a1a")
    label_title.pack(pady=(20, 10))
    
    # Status
    label_status = tk.Label(root, text="Initializing...", font=("Segoe UI", 10), fg="#aaaaaa", bg="#1a1a1a")
    label_status.pack(pady=5)
    
    # Progress bar (indeterminate)
    progress = ttk.Progressbar(root, mode='indeterminate', length=300)
    progress.pack(pady=10)
    progress.start(10)

    def check_queue():
        try:
            while not status_queue.empty():
                try:
                    msg = status_queue.get_nowait()
                except:
                    break
                    
                if msg == "QUIT":
                    root.destroy()
                    return
                label_status.config(text=msg)
        except:
            pass
        root.after(100, check_queue)

    root.after(100, check_queue)
    root.mainloop()

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
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"⚠️  pip warnings/errors: {result.stderr[:200]}")
        print("✓ Dependencies ready\n")
    except Exception as e:
        print(f"⚠️  Install attempt completed: {e}\n")

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

def launch_server(run_server):
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
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    try:
        # Run server directly (now passed as parameter)
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

def prepare_env(status_queue, server_context, is_frozen):
    """Prepare environment and load server."""
    try:
        print("\n" + "=" * 60)
        print("🚀 APL Chat Interface - v1.1.0")
        print("=" * 60 + "\n")
        
        if not is_frozen:
            status_queue.put("Checking dependencies...")
            # CRITICAL: Install deps FIRST before any other imports
            try:
                install_dependencies_first()
            except Exception as e:
                print(f"❌ Dependency install failed: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
                sys.exit(1)
        else:
            print("❄️  Running in frozen mode (skipping dependency check)")
        
        status_queue.put("Verifying Python environment...")
        try:
            check_python_version()
        except Exception as e:
            print(f"❌ Python check failed: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)
            sys.exit(1)
        
        # Import server after dependencies are installed
        try:
            status_queue.put("Loading AI Engine (this may take a moment)...")
            print("📥 Importing server module...")
            from apl_chat_server import run_server, warmup
            print("✓ Server imported successfully")
            
            # Save run_server for later execution in main thread
            server_context['run_server'] = run_server
            
            # Initialize heavy libraries while splash is showing
            status_queue.put("Initializing PyTorch & CUDA...")
            warmup()
            
        except ImportError as import_err:
            print(f"❌ Failed to import server: {import_err}")
            import traceback
            traceback.print_exc()
            time.sleep(5)
            sys.exit(1)
        except Exception as e:
            print(f"❌ Unexpected error during import: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)
            sys.exit(1)
        
        status_queue.put("Starting Server...")
        time.sleep(1) # Give user a moment to see "Starting Server"
        status_queue.put("QUIT") # Close splash screen
            
    except Exception as e:
        print(f"Critical Error: {e}")
        status_queue.put("QUIT")

def main():
    """Main entry point."""
    
    # Enable multiprocessing support for frozen apps
    if sys.platform == 'win32':
        multiprocessing.freeze_support()
        
    is_frozen = getattr(sys, 'frozen', False)
    
    if is_frozen:
        # Use multiprocessing Queue for inter-process communication
        status_queue = multiprocessing.Queue()
        
        # Start splash in separate process to completely isolate Tkinter
        splash_p = multiprocessing.Process(target=show_splash_screen, args=(status_queue,))
        splash_p.start()
        
        # Run preparation in main process
        server_context = {}
        try:
            prepare_env(status_queue, server_context, is_frozen)
        except Exception as e:
            print(f"Error in prepare_env: {e}")
            status_queue.put("QUIT")
            
        # Wait for splash to close
        splash_p.join()
        
        # Run server in main process
        if 'run_server' in server_context:
            try:
                launch_server(server_context['run_server'])
            except Exception as e:
                print(f"❌ Server launch failed: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
                sys.exit(1)
    else:
        # Run directly in main thread for development
        # Use multiprocessing here too to avoid Tkinter threading issues
        status_queue = multiprocessing.Queue()
        server_context = {}
        
        splash_p = multiprocessing.Process(target=show_splash_screen, args=(status_queue,))
        splash_p.start()
        
        try:
            prepare_env(status_queue, server_context, is_frozen)
        except Exception as e:
            print(f"Error in prepare_env: {e}")
            status_queue.put("QUIT")
            
        splash_p.join()
        
        if 'run_server' in server_context:
            launch_server(server_context['run_server'])

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
