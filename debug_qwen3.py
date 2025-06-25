#!/usr/bin/env python3

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load models
print("Loading HF model...")
hf_model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', torch_dtype=torch.bfloat16, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')

# Simple test case
text = "Hello world"
input_ids = tokenizer(text, return_tensors="pt").input_ids.to('cuda')
print(f"Input shape: {input_ids.shape}")

# Forward through one attention layer to see shapes
with torch.no_grad():
    # Get embeddings
    inputs_embeds = hf_model.model.embed_tokens(input_ids)
    print(f"Embeddings shape: {inputs_embeds.shape}")
    
    # Get first attention layer
    attn_layer = hf_model.model.layers[0].self_attn
    
    # Check what happens in q_proj
    hidden_states = inputs_embeds
    q_proj_out = attn_layer.q_proj(hidden_states)
    print(f"q_proj output shape: {q_proj_out.shape}")
    
    # Check input_shape calculation
    input_shape = hidden_states.shape[:-1]  # Should be [batch_size, seq_len]
    print(f"input_shape: {input_shape}")
    
    # Check hidden_shape calculation
    hidden_shape = (*input_shape, -1, attn_layer.head_dim)
    print(f"hidden_shape: {hidden_shape}")
    
    # Check view result
    q_view = q_proj_out.view(hidden_shape)
    print(f"q_proj viewed shape: {q_view.shape}")
    
    # Apply normalization
    q_norm_out = attn_layer.q_norm(q_view)
    print(f"q_norm output shape: {q_norm_out.shape}")
    
    # Transpose
    q_transposed = q_norm_out.transpose(1, 2)
    print(f"q_transposed shape: {q_transposed.shape}")