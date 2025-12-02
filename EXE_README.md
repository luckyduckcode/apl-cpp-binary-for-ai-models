# APL Chat Interface - Standalone Executable

The APL Chat Interface has been built as a standalone `.exe` file that requires **no Python installation** or dependencies. Just download and run!

## Quick Start

### Option 1: Double-click (Easiest)
1. Navigate to the `dist/` folder
2. Double-click `APL-Chat.exe`
3. Your browser will open automatically with the chat interface

### Option 2: Command Line
```cmd
APL-Chat.exe
```

## What's Included

The executable contains:
- ✓ Flask web server (chat backend)
- ✓ Python runtime
- ✓ All dependencies (torch, transformers, numpy, etc.)
- ✓ Web UI (HTML, CSS, JavaScript)
- ✓ Support for 3 quantized models:
  - TinyLlama 1.1B (251 MB)
  - Mistral 7B (1.1 GB)
  - Mistral Instruct (1.1 GB)

## System Requirements

- **OS**: Windows 10/11 (64-bit)
- **RAM**: 8 GB minimum (16 GB recommended for larger models)
- **Disk**: ~3 GB for models (downloaded on first use)
- **GPU**: Optional (CPU mode works, GPU accelerates)

## First Run

When you first launch the executable:
1. It will extract and start the Flask server
2. Your browser will open to `http://localhost:5000`
3. Models will auto-download on first use (~30 seconds)
4. Start chatting!

## Features

### Model Selection
- Switch between TinyLlama, Mistral 7B, and Mistral Instruct
- Real-time model loading with status updates

### Chat Controls
- **Temperature** (0.0-2.0): Controls creativity
- **Top-p** (0.0-1.0): Controls diversity
- **Max Tokens**: Output length
- **System Prompt**: Customize AI behavior

### UI Features
- Dark theme with responsive design
- Real-time streaming responses
- Chat history management
- Model info and compression metrics
- Mobile-friendly layout

## Troubleshooting

### Port 5000 Already in Use
```cmd
# Find and stop the process:
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Models Not Downloading
- Check internet connection
- Ensure 3+ GB free disk space
- Clear cache: Delete `.cache\huggingface` folder

### Slow Performance
- Use smaller model (TinyLlama) first
- Ensure no other heavy programs running
- GPU recommended for Mistral models

### Browser Not Opening
- Manually open: http://localhost:5000
- Port may be customized in server settings

## Advanced Usage

### Running with Console (Debugging)
```cmd
APL-Chat.exe  # Windowed (no console)
# OR modify launch_chat.py for console output
```

### Model Caching
Models are automatically cached in:
```
%USERPROFILE%\.cache\huggingface
```

Delete this folder to re-download fresh models.

### Customization
Edit `launch_chat.py` before building to customize:
- Default model
- Port number
- Temperature ranges
- UI styling

## Building Your Own Executable

To rebuild from source:

```powershell
# 1. Clone and setup
git clone <repo>
cd apl-cpp-binary-for-ai-models
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Build
python build_exe_simple.py

# 3. Find executable
# dist\APL-Chat.exe
```

## Distribution

The executable is self-contained and portable:
- ✓ No Python installation needed
- ✓ Works offline (models must be downloaded first)
- ✓ Can be shared via USB, email, or cloud storage
- ✓ No registry changes or system modifications

## File Size

- **APL-Chat.exe**: ~8 MB (Python + dependencies)
- **Models** (first download):
  - TinyLlama: 251 MB
  - Mistral 7B: 1.1 GB
  - Mistral Instruct: 1.1 GB

Total: ~2.5 GB after full setup

## Performance

Approximate tokens/second on CPU:
- TinyLlama: 50 tokens/sec
- Mistral 7B: 20 tokens/sec
- Mistral Instruct: 20 tokens/sec

With NVIDIA GPU (CUDA):
- TinyLlama: 200+ tokens/sec
- Mistral 7B: 80+ tokens/sec

## License

See LICENSE file in repository

## Support

- Check logs in `build/APL-Chat/warn-APL-Chat.txt`
- Review Flask output in console window
- Test models with Python script directly if needed

---

**Version**: 1.0  
**Built**: December 2025  
**Python**: 3.14.0  
**PyInstaller**: 6.17.0
