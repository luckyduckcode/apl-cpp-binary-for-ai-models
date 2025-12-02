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

## 🤖 Supported Models
The application comes with support for the following AI models ("robots"):

### **TinyLlama 1.1B**
- **Best For**: Speed, low memory usage, and casual conversation.
- **Capabilities**: Can handle basic chat, creative writing, and simple questions.
- **Hardware**: Runs smoothly on almost any CPU and requires very little RAM (~600MB).

### **Mistral 7B**
- **Best For**: General purpose reasoning, summarization, and complex tasks.
- **Capabilities**: Strong logic, good knowledge base, and coherent long-form text generation.
- **Hardware**: Requires more RAM (~5GB) and benefits significantly from a GPU.

### **Mistral 7B Instruct**
- **Best For**: Following specific instructions, Q&A, and structured tasks.
- **Capabilities**: Fine-tuned to follow user commands precisely, making it ideal for assistants and productivity.
- **Hardware**: Same requirements as Mistral 7B.

## 📦 Installation & Usage
1.  **Download**: Download both `APL-Chat.exe.part001` and `APL-Chat.exe.part002` from the release assets.
2.  **Combine Files**:
    *   **Windows**: Open a terminal in the download folder and run:
        ```cmd
        copy /b APL-Chat.exe.part* APL-Chat.exe
        ```
    *   **Linux/Mac**: Run:
        ```bash
        cat APL-Chat.exe.part* > APL-Chat.exe
        ```
3.  **Run**: Double-click the resulting `APL-Chat.exe` to start the application.
4.  **GPU Setup (Optional)**: If you have an NVIDIA GPU but are stuck on CPU mode, run `setup_gpu_env.bat` to configure your environment correctly.

## 📋 Requirements
- **OS**: Windows 10/11
- **GPU (Optional)**: NVIDIA GPU with CUDA support for 4-bit quantization and faster inference.
- **CPU**: Works on standard CPUs (slower inference, FP32 mode).
