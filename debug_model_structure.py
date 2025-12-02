#!/usr/bin/env python3
"""Debug script to check TinyLlama model structure."""

from transformers import AutoModelForCausalLM
import torch

print("Loading TinyLlama model...")
model = AutoModelForCausalLM.from_pretrained(
    'TinyLlama/TinyLlama-1.1B-Chat-v1.0', 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
)

# Check for layers
if hasattr(model, 'model'):
    print('Model has "model" attribute')
    if hasattr(model.model, 'layers'):
        layers = model.model.layers
        print(f'Layers count: {len(layers)}')
        if len(layers) > 0:
            layer0 = layers[0]
            print(f'\nLayer[0] type: {type(layer0).__name__}')
            
            # Check self_attn
            if hasattr(layer0, 'self_attn'):
                print('\n[self_attn] has weight attributes:')
                attn = layer0.self_attn
                for name in dir(attn):
                    if 'weight' in name.lower() and not name.startswith('_'):
                        weight_shape = getattr(attn, name).shape
                        print(f'  {name}: {weight_shape}')
            
            # Check mlp
            if hasattr(layer0, 'mlp'):
                print('\n[mlp] has weight attributes:')
                mlp = layer0.mlp
                for name in dir(mlp):
                    if 'weight' in name.lower() and not name.startswith('_'):
                        try:
                            weight_shape = getattr(mlp, name).shape
                            print(f'  {name}: {weight_shape}')
                        except:
                            pass
            
            # List all named modules
            print('\nAll parameters in layer[0]:')
            for name, param in layer0.named_parameters():
                print(f'  {name}: {param.shape}')
