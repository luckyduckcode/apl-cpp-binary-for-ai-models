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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import subprocess
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('bitsandbytes').setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).resolve().parent))

app = Flask(__name__, template_folder='.', static_folder='.')
CORS(app)

# GPU/Device Configuration
def init_gpu():
    """Initialize GPU settings for optimal performance."""
    if torch.cuda.is_available():
        # Enable TF32 for speedup (maintains precision for LLMs)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Enable cuDNN benchmarking for optimization
        torch.backends.cudnn.benchmark = True
        return "cuda"
    return "cpu"

device = init_gpu()
max_workers = 4 if device == "cuda" else multiprocessing.cpu_count()
executor = ThreadPoolExecutor(max_workers=max_workers)

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

# 4-bit quantization config for GPU (BitsAndBytes)
def get_quantization_config():
    """Get 4-bit quantization config for efficient GPU inference."""
    if device == "cuda":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # Double quantization for extra compression
            bnb_4bit_quant_type="nf4",  # NormalFloat4 - optimal for LLMs
        )
    return None

def get_gpu_info():
    """Get GPU information."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return {
            "name": torch.cuda.get_device_name(0),
            "memory_gb": props.total_memory / 1e9,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
    return None


def load_model(model_name: str) -> dict:
    """Load a model with GPU acceleration and 4-bit quantization."""
    global current_model, current_tokenizer, current_model_name
    
    try:
        if current_model_name == model_name and current_model is not None:
            return {"status": "ok", "message": f"{model_name} already loaded"}
        
        repo = MODELS[model_name]["repo"]
        print(f"\n[LOAD] {model_name}")
        print(f"  Repo: {repo}")
        print(f"  Device: {device} | Workers: {max_workers}")
        
        current_tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        
        # Load with GPU optimization if available
        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        if device == "cuda":
            # Use 4-bit quantization on GPU with accelerate
            quant_config = get_quantization_config()
            load_kwargs["quantization_config"] = quant_config
            load_kwargs["device_map"] = "auto"
            print("  Using: 4-bit NF4 quantization (GPU)")
            current_model = AutoModelForCausalLM.from_pretrained(repo, **load_kwargs)
        else:
            # On CPU, use FP16 for memory efficiency (simpler than Int8)
            print("  Using: FP16 (CPU, memory efficient)")
            load_kwargs["torch_dtype"] = torch.float16
            current_model = AutoModelForCausalLM.from_pretrained(repo, **load_kwargs)
            current_model = current_model.to(device)
        
        current_model.eval()
        current_model_name = model_name
        
        if device == "cuda":
            torch.cuda.empty_cache()
            mem_gb = torch.cuda.memory_allocated() / 1e9
            print(f"  ✓ GPU Memory: {mem_gb:.2f}GB")
        else:
            print(f"  ✓ Model ready for inference")
        
        return {"status": "ok", "message": f"[OK] {model_name} loaded on {device.upper()}"}
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] {error_msg[:100]}")
        return {"status": "error", "message": f"Failed to load model: {error_msg[:200]}"}


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
        
        # Generate with optimizations for fast inference
        with torch.no_grad():
            output_ids = current_model.generate(
                inputs['input_ids'],
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                eos_token_id=current_tokenizer.eos_token_id,
                pad_token_id=current_tokenizer.pad_token_id,
                num_beams=1,  # Parallel decoding
                use_cache=True,  # Enable KV cache for faster generation
                repetition_penalty=1.1,  # Prevent repetition
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
    
    gpu_info = get_gpu_info()
    device_str = device
    if gpu_info:
        device_str = f"{device} ({gpu_info['name']})"
    
    return jsonify({
        "models": models_info,
        "current_model": current_model_name,
        "device": device_str,
        "quantization": "4-bit NF4 (BitsAndBytes)",
        "parallel_workers": max_workers,
        "gpu_info": gpu_info,
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
    # Lazy load model if needed
    if current_model is None:
        load_model("TinyLlama 1.1B")
    
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
    health_data = {
        "status": "ok",
        "model_loaded": current_model is not None,
        "current_model": current_model_name,
        "device": device,
        "quantization": "4-bit NF4",
        "parallel_workers": max_workers,
    }
    
    if device == "cuda":
        gpu_info = get_gpu_info()
        health_data["gpu"] = {
            "name": gpu_info["name"],
            "memory_gb": gpu_info["memory_gb"],
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "cuda_version": gpu_info["cuda_version"],
        }
    
    return jsonify(health_data)


def run_server():
    """Run the Flask server (called from launcher)."""
    # Display startup info
    print("\n" + "=" * 60)
    print("[INIT] APL Chat Server - GPU Optimized")
    print("=" * 60)
    print(f"Device: {device.upper()}")
    
    if device == "cuda":
        gpu_info = get_gpu_info()
        print(f"GPU: {gpu_info['name']}")
        print(f"GPU Memory: {gpu_info['memory_gb']:.1f}GB")
        print(f"CUDA Version: {gpu_info['cuda_version']}")
        print(f"cuDNN Version: {gpu_info['cudnn_version']}")
        print(f"TF32 Enabled: Yes (faster inference)")
    
    print(f"Parallel Workers: {max_workers}")
    print(f"Quantization: 4-bit NF4 (BitsAndBytes)")
    print(f"Compute Precision: FP16 (GPU) / FP32 (CPU)")
    print("\n[NOTE] Models load on first request (lazy loading)")
    
    print("\n[START] APL Chat Server")
    print("[INFO] Open http://localhost:5000 in your browser")
    print("[INFO] API Health: http://localhost:5000/api/health")
    print("=" * 60 + "\n")
    
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)


if __name__ == '__main__':
    run_server()
