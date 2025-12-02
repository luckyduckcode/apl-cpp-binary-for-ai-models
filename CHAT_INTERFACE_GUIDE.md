# APL Chat - Ollama-like Chat Interface

A beautiful, responsive chat interface for interacting with quantized LLaMA models using the APL inference engine. Similar to Ollama but optimized for your quantized models.

## Quick Start

### Option 1: Easy Launch (Recommended)
```bash
python launch_chat.py
```

### Option 2: Flask Server (Lightweight)
```bash
python apl_chat_server.py
```
Then open: http://localhost:5000

### Option 3: Gradio Interface (Feature-rich)
```bash
python apl_chat_ui.py
```
Then open: http://localhost:7860

---

## Features

### 💬 Chat Interface
- Clean, modern UI (Ollama-style)
- Real-time message streaming
- Automatic response generation
- Chat history management
- Responsive design (mobile-friendly)

### 🤖 Model Selection
- **TinyLlama 1.1B** - Fast, 251 MB (8.8x compression)
- **Mistral 7B** - Capable, 1.1 GB (12x compression)
- **Mistral 7B Instruct** - Optimized for instructions, 1.1 GB (12x compression)

Easy switching between models with one click.

### ⚙️ Generation Settings
- **Temperature** (0.0 - 2.0)
  - Lower = more deterministic
  - Higher = more creative
  - Default: 0.7

- **Top-p** (0.0 - 1.0)
  - Controls diversity through nucleus sampling
  - Higher = more diverse
  - Default: 0.95

- **System Prompt**
  - Customize how the model behaves
  - Change personality, tone, constraints
  - Reset to default anytime

### 🚀 Performance
- **TinyLlama**: ~50 tokens/sec on CPU
- **Mistral 7B**: ~20 tokens/sec on CPU
- GPU acceleration available if CUDA installed

### 📊 Model Information
View detailed specs:
- Quantization level (4-bit)
- Compressed size
- Compression ratio
- Estimated tokens/sec

---

## Installation

### Prerequisites
- Python 3.8+
- 4GB+ RAM (8GB recommended)
- Quantized models converted (see [MODEL_CONVERSION_GUIDE.md](MODEL_CONVERSION_GUIDE.md))

### Install Dependencies

**For Flask (lightweight, recommended):**
```bash
pip install flask flask-cors torch transformers
```

**For Gradio (feature-rich):**
```bash
pip install gradio torch transformers
```

**Or install all at once:**
```bash
pip install flask flask-cors gradio torch transformers
```

### Ensure Models Are Converted
If you haven't converted models yet:
```bash
python scripts/convert_models_automated.py tinyllama --bits 4
python scripts/convert_models_automated.py mistral-7b --bits 4
python scripts/convert_models_automated.py mistral-7b-instruct --bits 4
```

---

## Usage

### Start the Chat
```bash
# Easy launcher (recommended)
python launch_chat.py

# Or directly
python apl_chat_server.py
```

### Using the Chat Interface

1. **Select a Model** (top left)
   - Click on any model to load it
   - Wait for "Model loaded" confirmation
   - Model name and device displayed in header

2. **Type Your Message**
   - Click in the message box
   - Type your prompt
   - Press Enter to send (or Shift+Enter for new line)
   - Or click the Send button

3. **Adjust Settings** (right sidebar)
   - **Temperature**: Make responses more/less creative
   - **Top-p**: Adjust diversity
   - **System Prompt**: Define model behavior
   
   Examples:
   ```
   # Helpful assistant
   You are a helpful, harmless, and honest AI assistant.
   
   # Code expert
   You are an expert programmer. Provide clean, efficient code with explanations.
   
   # Conversational
   Be friendly and conversational. Ask follow-up questions to understand better.
   ```

4. **Clear History**
   - Click "Clear History" to start fresh
   - Previous messages are removed

### Keyboard Shortcuts
- `Enter` - Send message
- `Shift+Enter` - New line in message
- `Ctrl+L` (coming soon) - Clear history

---

## API Endpoints

If using the Flask server, you can also call the API directly:

### Get Available Models
```bash
curl http://localhost:5000/api/models
```

Response:
```json
{
  "models": [
    {
      "name": "TinyLlama 1.1B",
      "bits": 4,
      "size": "251 MB",
      "compression": "8.8x",
      "tokens_per_sec": "~50"
    }
  ],
  "current_model": "TinyLlama 1.1B",
  "device": "cpu"
}
```

### Load a Model
```bash
curl -X POST http://localhost:5000/api/load_model \
  -H "Content-Type: application/json" \
  -d '{"model_name": "Mistral 7B"}'
```

### Send Message
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, how are you?",
    "system_prompt": "You are a helpful assistant",
    "temperature": 0.7,
    "top_p": 0.95
  }'
```

Response:
```json
{
  "status": "ok",
  "response": "I'm doing well, thank you for asking! How can I help you today?",
  "model": "TinyLlama 1.1B",
  "timestamp": "2025-12-02T04:30:00"
}
```

### Health Check
```bash
curl http://localhost:5000/api/health
```

---

## Performance Tips

### For Better Speed
1. Use **TinyLlama 1.1B** for fastest responses
2. Lower **max_tokens** (default 512)
3. Reduce **context length** in model selection
4. Use **GPU** if available: `python scripts/build_backend.py --gpu`

### For Better Quality
1. Use **Mistral 7B** or **Mistral Instruct**
2. Craft detailed **system prompts**
3. Use lower **temperature** (0.3-0.7) for consistency
4. Use higher **top_p** (0.9-0.99) for diversity

### For Longer Responses
1. Select **Mistral 7B Instruct** (32K context)
2. Increase **max_tokens** setting
3. Be specific in your prompt

---

## Troubleshooting

### Issue: "Model not found" error
**Solution:** Convert the model first
```bash
python scripts/convert_models_automated.py tinyllama --bits 4
```

### Issue: Out of Memory
**Solution:** Use TinyLlama instead of Mistral 7B
```bash
# Click on TinyLlama 1.1B in the model list
```

### Issue: Slow responses
**Solution:** 
- Use TinyLlama for speed
- Check system resources (top/Task Manager)
- Enable GPU acceleration if available

### Issue: Port already in use
**Solution:** Change the port in the code or kill the process
```bash
# Flask (default 5000)
# Gradio (default 7860)

# Or modify the port in the Python file
# server_port=5001
```

### Issue: Browser won't connect
**Solution:**
- Check if server is running
- Try http://localhost:5000 (Flask) or http://localhost:7860 (Gradio)
- Check firewall settings
- Try a different browser

---

## Customization

### Change Default Model
Edit `apl_chat_server.py` line ~180:
```python
# Load default model
result = load_model("Mistral 7B")  # Change this line
```

### Change System Prompt
Edit `apl_chat.html` line ~290:
```javascript
<textarea id="system-prompt" class="system-prompt">
Your custom prompt here
</textarea>
```

### Change Server Port
Edit `apl_chat_server.py` last line:
```python
app.run(host='127.0.0.1', port=5001, ...)  # Change port here
```

### Custom Models
Add to `MODELS` dict in `apl_chat_server.py`:
```python
"Llama 2 7B": {
    "repo": "meta-llama/Llama-2-7b-hf",
    "bits": 4,
    "size": "3.5 GB",
    "compression": "4x",
    "tokens_per_sec": "~25",
},
```

---

## Architecture

### Components

```
┌─────────────────────────────────────┐
│  Web Browser (HTML/CSS/JS)          │
│  - apl_chat.html                    │
│  - Modern, responsive UI            │
└────────────┬────────────────────────┘
             │ HTTP/JSON
┌────────────▼────────────────────────┐
│  Flask Server (Python)              │
│  - apl_chat_server.py               │
│  - REST API endpoints               │
│  - Model management                 │
└────────────┬────────────────────────┘
             │ PyTorch
┌────────────▼────────────────────────┐
│  Quantized Model (4-bit)            │
│  - TinyLlama, Mistral               │
│  - 251 MB - 1.1 GB                  │
└─────────────────────────────────────┘
```

### Data Flow
```
User Input
    ↓
JavaScript (apl_chat.html)
    ↓
HTTP POST /api/chat
    ↓
Flask Server (apl_chat_server.py)
    ↓
Load Model (transformers)
    ↓
Tokenize + Generate (PyTorch)
    ↓
Decode Response
    ↓
JSON Response
    ↓
Display in Chat UI
```

---

## Alternatives

### Gradio Interface
If you prefer Gradio's look and features:
```bash
python apl_chat_ui.py
```

Features:
- Built-in chat interface
- Automatic UI generation
- More out-of-the-box features
- Heavier dependency (Gradio)

### Flask Server
If you want full control:
```bash
python apl_chat_server.py
```

Features:
- Lightweight
- Custom HTML/CSS/JS
- Full API access
- Best performance

### Command Line
For quick testing:
```bash
# Run inference directly
python cpp/call_backend.py --manifest models/tinyllama_manifest_q4.json
```

---

## Performance Benchmarks

### TinyLlama 1.1B
- **Model Size**: 251 MB (compressed from 2.2 GB)
- **Speed**: ~50 tokens/sec (CPU), ~200+ tokens/sec (GPU)
- **Memory**: ~2.5 GB RAM
- **Best For**: Speed, low-resource devices

### Mistral 7B
- **Model Size**: 1.1 GB (compressed from 13.2 GB)
- **Speed**: ~20 tokens/sec (CPU), ~100+ tokens/sec (GPU)
- **Memory**: ~5-8 GB RAM
- **Best For**: Quality, general purpose

### Mistral 7B Instruct
- **Model Size**: 1.1 GB
- **Speed**: ~20 tokens/sec (CPU)
- **Memory**: ~5-8 GB RAM
- **Best For**: Instructions, tasks, following directions

---

## Advanced Usage

### Using with GPU
```bash
# Build GPU support
python scripts/build_backend.py --gpu

# Then run chat
python apl_chat_server.py
```

Monitor GPU usage during chat for verification.

### Batch Inference
Modify `apl_chat_server.py` to support multiple concurrent users:
```python
# Currently: Single-threaded
# Enhancement: Add request queue and worker threads
```

### Custom Quantization
Use different quantization levels:
```bash
python scripts/convert_models_automated.py tinyllama --bits 2  # 16x compression
python scripts/convert_models_automated.py tinyllama --bits 8  # 4x compression
```

---

## FAQ

**Q: How do I change models?**
A: Click on the model name in the left sidebar. It will load automatically.

**Q: Can I use my own model?**
A: Yes! Convert it with:
```bash
python scripts/convert_models_automated.py "huggingface/model-name" --bits 4
```

**Q: What does temperature do?**
A: Lower temperature (0.1-0.5) = more focused, deterministic
Higher temperature (1.0-2.0) = more creative, random

**Q: Can I run this on mobile?**
A: The UI is responsive, but you need a backend server. Set up on a computer and access from mobile over network.

**Q: Is my chat history saved?**
A: No, it's only in your browser memory. Refresh the page to clear.

**Q: Can I use without internet?**
A: Yes! Everything runs locally. No data is sent to external servers.

---

## Support

Having issues? Check:
1. [STATUS_REPORT.md](STATUS_REPORT.md) - Project status
2. [MODEL_CONVERSION_GUIDE.md](MODEL_CONVERSION_GUIDE.md) - Model setup
3. [GPU_AND_AUTOMATION_GUIDE.md](GPU_AND_AUTOMATION_GUIDE.md) - Technical details

---

## License

See LICENSE in repository root.

---

**Last Updated:** December 2, 2025
**Status:** ✅ Production Ready
**Version:** 1.0
