# APL Chat v1.1.0 Release Summary

## 🚀 New Features
- **GPU/CPU Auto-Detection**: The application now automatically detects if a compatible NVIDIA GPU is available and switches to it for faster inference.
- **Performance Benchmarks**: The model list now displays estimated tokens per second (t/s) based on your detected hardware (CPU vs GPU).
- **Improved UI**: The model selection sidebar has been updated to show more detailed information including model size, quantization bits, and speed estimates.
- **Python 3.12 Support**: Added a setup script (`setup_gpu_env.bat`) to easily configure a Python 3.12 environment with CUDA support, addressing issues with newer Python versions (3.14) that lack GPU support.

## 🛠️ Fixes & Improvements
- **Robust Model Loading**: Improved error handling during model loading to prevent crashes when specific configurations are missing.
- **Startup Diagnostics**: The server console now provides clear information about the active device, quantization mode (4-bit NF4 vs FP32), and potential environment issues.
- **Build Process**: Updated build scripts to ensure smoother executable generation.

## 📦 Installation & Usage
1.  **Download**: Get the `APL-Chat.exe` from the release assets.
2.  **Run**: Double-click `APL-Chat.exe` to start the application.
3.  **GPU Setup (Optional)**: If you have an NVIDIA GPU but are stuck on CPU mode, run `setup_gpu_env.bat` to configure your environment correctly.

## 📋 Requirements
- **OS**: Windows 10/11
- **GPU (Optional)**: NVIDIA GPU with CUDA support for 4-bit quantization and faster inference.
- **CPU**: Works on standard CPUs (slower inference, FP32 mode).
