#!/usr/bin/env python3

import torch
import json
import tempfile
import time
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokasaurus.common_types import ServerConfig
from tokasaurus.entry import server_manager
from tokasaurus.utils import find_free_port

# Test setup
MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "Can you tell me a long story about a cat?"

print("Loading HF model...")
hf_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained(MODEL)
print("HF model loaded.")

# Create tokasaurus config
config = ServerConfig()
config.model = MODEL
config.kv_cache_num_tokens = 16384
config.max_num_tokens_per_request = 16384
config.max_seqs_per_forward = 1024
config.port = find_free_port()

print(f"Starting tokasaurus server on port {config.port}...")
with server_manager(config):
    client = OpenAI(api_key="beepboop", base_url=f"http://localhost:{config.port}/v1")
    
    # Get response from tokasaurus
    print("Getting tokasaurus response...")
    response = client.chat.completions.create(
        model="none",
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=60,  # Small number to test
        temperature=0.0,
        logprobs=1,
    )
    
    api_tokens = [token_logprob.token for token_logprob in response.choices[0].logprobs.content]
    api_logprobs = [token_logprob.logprob for token_logprob in response.choices[0].logprobs.content]
    
    token_ids = json.loads(response.system_fingerprint)["completion_ids"][0]
    seq_ids = tokenizer.convert_tokens_to_ids(api_tokens)
    
    print(f"Got {len(api_tokens)} tokens from tokasaurus")
    
    # Get HF reference
    input_ids = tokenizer.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True) + seq_ids
    
    with torch.inference_mode():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_tensor = torch.tensor(input_ids).unsqueeze(0).to("cuda")
            outputs = hf_model(input_tensor)
    
    logits = outputs.logits.to(torch.float32)
    hf_logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    print("Comparing predictions...")
    for idx, (api_token_id, hf_logprob_dist, api_logprob) in enumerate(zip(seq_ids, hf_logprobs[0, -len(seq_ids) - 1: -1], api_logprobs)):
        hf_logprob = hf_logprob_dist[api_token_id].item()
        hf_token_id = hf_logprob_dist.argmax().item()
        
        api_token = tokenizer.decode(api_token_id)
        hf_token = tokenizer.decode(hf_token_id)
        
        print(f"Token {idx}: API={api_token_id} ('{api_token}') logprob={api_logprob:.6f}, HF={hf_token_id} ('{hf_token}') logprob={hf_logprob:.6f}")
        
        if hf_token_id != api_token_id:
            print(f"*** DIVERGENCE at token {idx} ***")
            # Show top 5 predictions from each
            api_top5 = torch.topk(hf_logprob_dist, 5)
            print("HF top 5 predictions:")
            for i, (prob, token_id) in enumerate(zip(api_top5.values, api_top5.indices)):
                print(f"  {i+1}. {token_id.item()} ('{tokenizer.decode(token_id.item())}') logprob={prob.item():.6f}")
            break