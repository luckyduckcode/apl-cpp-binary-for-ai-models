#!/usr/bin/env python3
"""
APL Quantized Inference - Quick Start Guide

This is an executable summary of what was built and how to use it.
Run: python QUICK_START.py
"""

import subprocess
import sys
from pathlib import Path


def run_section(title, cmd):
    """Run a command and show results."""
    print(f"\n{'='*80}")
    print(f"[{title}]")
    print(f"{'='*80}")
    print(f"\nCommand: {cmd}\n")
    try:
        if isinstance(cmd, str):
            subprocess.run(cmd, shell=True)
        else:
            subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nSkipped.")


def main():
    repo_root = Path(__file__).resolve().parent
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║  APL Quantized Inference - GPU Optimized, Fully Automated Pipeline      ║
║                                                                           ║
║  This quick start will guide you through:                               ║
║  1. Launching the Ollama-like chat GUI                                  ║
║  2. Building the optimized backend (CPU + optional GPU)                 ║
║  3. Validating accuracy across quantization levels                       ║
║  4. Downloading and converting popular LLMs                              ║
║  5. Running inference                                                    ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n📚 Documentation Files:")
    print("  - GPU_AND_AUTOMATION_GUIDE.md  (Complete feature guide)")
    print("  - OPTIMIZATION_SUMMARY.md      (What was implemented)")
    print("  - README.md                    (Original project guide)")
    print("  - accuracy_validation_results.json (Latest validation results)")
    
    print("\n" + "="*80)
    print("QUICK START OPTIONS")
    print("="*80)
    
    options = {
        '1': ("Launch APL Chat (Ollama-like GUI)", 
              ["python", str(repo_root / "launch_chat.py")]),
        '2': ("Build Backend (CPU-only)", 
              ["python", str(repo_root / "scripts" / "build_backend.py")]),
        '3': ("Build Backend with GPU Support", 
              ["python", str(repo_root / "scripts" / "build_backend.py"), "--gpu"]),
        '4': ("Run Accuracy Validation Suite", 
              ["python", str(repo_root / "tests" / "test_accuracy_validation.py")]),
        '5': ("Download & Convert TinyLlama (1.1B, 4-bit)", 
              ["python", str(repo_root / "scripts" / "convert_models_automated.py"), "tinyllama", "--bits", "4"]),
        '6': ("Download & Convert Mistral 7B (7B, 4-bit)", 
              ["python", str(repo_root / "scripts" / "convert_models_automated.py"), "mistral-7b", "--bits", "4"]),
        '7': ("Download & Convert Mistral Instruct (7B, 4-bit)", 
              ["python", str(repo_root / "scripts" / "convert_models_automated.py"), "mistral-7b-instruct", "--bits", "4"]),
        '8': ("Run Inference with TinyLlama", 
              ["python", str(repo_root / "easy_run.py"), "--model", "tinyllama"]),
        '9': ("Convert All Popular Models", 
              ["python", str(repo_root / "scripts" / "convert_models_automated.py"), "--convert-all", "--bits", "4"]),
        'q': ("Quit", None),
    }
    
    for key, (desc, _) in options.items():
        print(f"  [{key}] {desc}")
    
    print("\n" + "="*80)
    
    while True:
        choice = input("\nSelect an option (1-9, q to quit): ").strip().lower()
        
        if choice == 'q':
            print("\nGoodbye!")
            return 0
        
        if choice not in options:
            print("Invalid option. Try again.")
            continue
        
        title, cmd = options[choice]
        
        if cmd is None:
            continue
        
        print(f"\n{'='*80}")
        print(f"Starting: {title}")
        print(f"{'='*80}\n")
        
        try:
            if isinstance(cmd, str):
                subprocess.run(cmd, shell=True)
            else:
                subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n\nSkipped.")
        except Exception as e:
            print(f"\nError: {e}")
        
        input("\nPress Enter to continue...")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
