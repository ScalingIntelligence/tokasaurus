#!/usr/bin/env python3
"""
Simplified debug script that mimics the existing test structure more closely
"""
import tempfile
import os
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

from tokasaurus.common_types import ServerConfig
from tokasaurus.entry import server_manager
from tokasaurus.utils import find_free_port


def compare_with_simple_server(model_name: str = "Qwen/Qwen2-0.5B"):
    print(f"Comparing models using {model_name}")
    
    # Load HF model
    print("Loading HuggingFace model...")
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    hf_model.eval()
    hf_model.to("cuda")
    
    # Test prompt (simple, short)
    prompt = "Hello world"
    input_ids = hf_tokenizer.encode(prompt)
    print(f"Input: {prompt}")
    print(f"Input IDs: {input_ids}")
    
    # Run HF model
    print("\nRunning HuggingFace model...")
    with torch.no_grad():
        hf_input_tensor = torch.tensor(input_ids).unsqueeze(0).to("cuda")
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            hf_outputs = hf_model(hf_input_tensor)
        hf_logits = hf_outputs.logits.squeeze(0).to(torch.float32)  # [seq_len, vocab_size]
        hf_logprobs = F.log_softmax(hf_logits, dim=-1)
        hf_last_logprobs = hf_logprobs[-1]  # Last token logprobs
        hf_top_tokens = torch.topk(hf_last_logprobs, k=5)
    
    print(f"HF logits shape: {hf_logits.shape}")
    print(f"HF top 5 tokens: {hf_top_tokens.indices.tolist()}")
    print(f"HF top 5 logprobs: {hf_top_tokens.values.tolist()}")
    
    # Run minimal Tokasaurus server
    print("\nRunning Tokasaurus server...")
    mp.set_start_method("spawn", force=True)
    
    config = ServerConfig()
    config.model = model_name
    config.kv_cache_num_tokens = 4096  # Small cache
    config.max_num_tokens_per_request = 1024
    config.max_seqs_per_forward = 64
    config.port = find_free_port()
    
    try:
        with server_manager(config):
            from openai import OpenAI
            client = OpenAI(
                api_key="test", base_url=f"http://localhost:{config.port}/v1"
            )
            
            # Get single token completion
            response = client.chat.completions.create(
                model="none",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                temperature=0.0,
                logprobs=1,
            )
            
            if response.choices and response.choices[0].logprobs and response.choices[0].logprobs.content:
                api_logprob = response.choices[0].logprobs.content[0].logprob
                api_token_id = response.choices[0].logprobs.content[0].token
                
                print(f"Toka token: {api_token_id}")
                print(f"Toka logprob: {api_logprob}")
                
                # Convert token to ID for comparison
                toka_token_id = hf_tokenizer.convert_tokens_to_ids([api_token_id])[0]
                hf_top_token_id = hf_top_tokens.indices[0].item()
                
                print(f"\nComparison:")
                print(f"HF top token ID: {hf_top_token_id} ({hf_tokenizer.decode([hf_top_token_id])})")
                print(f"Toka token ID: {toka_token_id} ({api_token_id})")
                print(f"Tokens match: {hf_top_token_id == toka_token_id}")
                
                if hf_top_token_id == toka_token_id:
                    hf_logprob_for_token = hf_last_logprobs[toka_token_id].item()
                    logprob_diff = abs(api_logprob - hf_logprob_for_token)
                    print(f"HF logprob for token: {hf_logprob_for_token:.6f}")
                    print(f"Toka logprob: {api_logprob:.6f}")
                    print(f"Logprob difference: {logprob_diff:.6f}")
                    
                    if logprob_diff > 0.1:
                        print("❌ Significant logprob difference!")
                    else:
                        print("✅ Logprobs match!")
                else:
                    print("❌ Top tokens don't match!")
            else:
                print("No logprobs returned from server")
                
    except Exception as e:
        print(f"Error with server: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2-0.5B"
    compare_with_simple_server(model_name)