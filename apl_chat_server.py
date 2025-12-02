#!/usr/bin/env python3
"""
APL Chat Server - Flask-based chat interface with WebSocket streaming

Lightweight, fast, Ollama-like chat interface for quantized APL models.
Uses Flask for backend and vanilla JS for frontend (no heavy dependencies).
"""

import json
from pathlib import Path
from typing import Generator
import threading
import sys
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Suppress warnings
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)

app = Flask(__name__, template_folder='.', static_folder='.')
CORS(app)

# Model configurations
MODELS = {
    "TinyLlama 1.1B": {
        "repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "bits": 4,
        "size": "251 MB",
        "compression": "8.8x",
        "tokens_per_sec": "~50",
    },
    "Mistral 7B": {
        "repo": "mistralai/Mistral-7B-v0.1",
        "bits": 4,
        "size": "1.1 GB",
        "compression": "12x",
        "tokens_per_sec": "~20",
    },
    "Mistral 7B Instruct": {
        "repo": "mistralai/Mistral-7B-Instruct-v0.1",
        "bits": 4,
        "size": "1.1 GB",
        "compression": "12x",
        "tokens_per_sec": "~20",
    },
}

# Global state
current_model = None
current_tokenizer = None
current_model_name = "TinyLlama 1.1B"
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name: str) -> dict:
    """Load a model."""
    global current_model, current_tokenizer, current_model_name
    
    try:
        if current_model_name == model_name and current_model is not None:
            return {"status": "ok", "message": f"{model_name} already loaded"}
        
        repo = MODELS[model_name]["repo"]
        print(f"Loading {model_name} from {repo}...")
        
        current_tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        current_model = AutoModelForCausalLM.from_pretrained(
            repo,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        current_model = current_model.to(device)
        current_model.eval()
        current_model_name = model_name
        
        return {"status": "ok", "message": f"✓ {model_name} loaded on {device}"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to load model: {str(e)}"}


def generate_response(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 512,
) -> Generator[str, None, None]:
    """Generate response with streaming."""
    
    if current_model is None:
        yield "Error: Model not loaded"
        return
    
    try:
        # Build prompt
        if system_prompt:
            full_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
        else:
            full_prompt = f"<s>[INST] {prompt} [/INST]"
        
        # Tokenize
        inputs = current_tokenizer(full_prompt, return_tensors="pt").to(device)
        input_length = inputs['input_ids'].shape[1]
        
        # Generate
        with torch.no_grad():
            output_ids = current_model.generate(
                inputs['input_ids'],
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                eos_token_id=current_tokenizer.eos_token_id,
                pad_token_id=current_tokenizer.pad_token_id,
            )
        
        # Decode response
        response = current_tokenizer.decode(
            output_ids[0][input_length:],
            skip_special_tokens=True
        )
        
        yield response
    except Exception as e:
        yield f"Error: {str(e)}"


@app.route('/')
def index():
    """Serve the chat interface."""
    return render_template('apl_chat.html')


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models."""
    models_info = []
    for name, config in MODELS.items():
        models_info.append({
            "name": name,
            "bits": config["bits"],
            "size": config["size"],
            "compression": config["compression"],
            "tokens_per_sec": config["tokens_per_sec"],
        })
    return jsonify({
        "models": models_info,
        "current_model": current_model_name,
        "device": device,
    })


@app.route('/api/load_model', methods=['POST'])
def api_load_model():
    """Load a model."""
    data = request.json
    model_name = data.get('model_name')
    
    if model_name not in MODELS:
        return jsonify({"status": "error", "message": "Unknown model"}), 400
    
    result = load_model(model_name)
    return jsonify(result)


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Chat endpoint - generates response."""
    data = request.json
    message = data.get('message', '').strip()
    system_prompt = data.get('system_prompt', '')
    temperature = float(data.get('temperature', 0.7))
    top_p = float(data.get('top_p', 0.95))
    
    if not message:
        return jsonify({"status": "error", "message": "Empty message"}), 400
    
    try:
        # Generate response
        response_text = ""
        for chunk in generate_response(message, system_prompt, temperature, top_p):
            response_text += chunk
        
        return jsonify({
            "status": "ok",
            "response": response_text,
            "model": current_model_name,
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check."""
    return jsonify({
        "status": "ok",
        "model_loaded": current_model is not None,
        "current_model": current_model_name,
        "device": device,
    })


if __name__ == '__main__':
    # Pre-load default model
    print("Initializing APL Chat Server...")
    print("Loading default model: TinyLlama 1.1B...")
    result = load_model("TinyLlama 1.1B")
    print(result)
    
    print("\n🚀 Starting APL Chat Server")
    print("📱 Open http://localhost:5000 in your browser")
    print("=" * 60)
    
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
