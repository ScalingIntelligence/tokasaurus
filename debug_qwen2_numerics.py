#!/usr/bin/env python3
"""
Debug script to compare Tokasaurus Qwen2 implementation with HuggingFace
without instantiating a full server.
"""

import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokasaurus.model.utils import make_model, make_input_batch_state, add_decoding_ids_to_batch_state, move_batch_state
from tokasaurus.manager.manager import seqs_to_input
from tokasaurus.manager.types import Sequence
from tokasaurus.server.types import SamplingParams
from tokasaurus.common_types import ServerConfig


def compare_models(model_name: str = "Qwen/Qwen2-0.5B"):
    print(f"Comparing models using {model_name}")
    
    device = "cuda:0"
    torch.cuda.set_device(device)
    
    # Load HuggingFace model and tokenizer
    print("Loading HuggingFace model...")
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    hf_model.eval()
    hf_model.to(device)
    
    # Setup Tokasaurus configuration
    print("Loading Tokasaurus model...")
    server_config = ServerConfig()
    server_config.model = model_name
    server_config.kv_cache_num_tokens = 1024 * 32  # 32K tokens cache
    server_config.page_size = 16
    server_config.pp_size = 1
    server_config.tp_size = 1
    
    # Create Tokasaurus model using the same pattern as bench_model.py
    dtype = torch.bfloat16
    toka_model = make_model(
        server_config,
        device=device,
        dtype=dtype,
        pp_rank=0,
        tp_rank=0,
        tp_group=None,
    )
    
    # Test prompt
    prompt = "The quick brown fox jumps over"
    input_ids = hf_tokenizer.encode(prompt)
    print(f"Input: {prompt}")
    print(f"Input IDs: {input_ids}")
    print(f"Input length: {len(input_ids)}")
    
    # Run HuggingFace model
    print("\nRunning HuggingFace model...")
    print(f"HF input IDs: {input_ids}")
    with torch.no_grad():
        hf_input_tensor = torch.tensor(input_ids).unsqueeze(0).to("cuda")
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            hf_outputs = hf_model(hf_input_tensor)
        hf_logits = hf_outputs.logits.squeeze(0).to(torch.float32)  # [seq_len, vocab_size]
        hf_logprobs = F.log_softmax(hf_logits, dim=-1)
        hf_last_logprobs = hf_logprobs[-1]  # Last token logprobs
        hf_top_tokens = torch.topk(hf_last_logprobs, k=10)
    
    print(f"HF logits shape: {hf_logits.shape}")
    print(f"HF top 10 tokens: {hf_top_tokens.indices.tolist()}")
    print(f"HF top 10 logprobs: {hf_top_tokens.values.tolist()}")
    
    # Run Tokasaurus model
    print("\nRunning Tokasaurus model...")
    try:
        vocab_size = toka_model.config.vocab_size
        page_size = server_config.page_size
        
        # Create sequence similar to bench_model.py pattern
        # We'll create a single decode sequence with the prompt as prefill
        sampling_params = SamplingParams(temperature=0.0, top_p=1.0)
        
        print(f"Toka input IDs: {input_ids}")
        
        seq = Sequence(
            id="debug_seq",
            input_ids=input_ids,
            completion_total=1,  # We want 1 token completion
            completion_scheduled=1,
            prompt_scheduled=len(input_ids),
            batch_index=0,
            kv_indices=[i for i in range(math.ceil(len(input_ids) / page_size) + 1)],  # Allocate enough pages
            sampling_params=sampling_params,
        )
        
        # Create input using seqs_to_input like in bench_model.py
        inp = seqs_to_input(
            decoding_seqs=[seq],
            prefill_seqs=[],  # No separate prefill seqs
            schedule_id="debug_schedule",
            hydragen_groups=None,
            page_size=page_size,
            starting_prefill_offset=0,
        )
        
        # Create batch state
        batch_state = make_input_batch_state(
            inp,
            pp_rank=0,
            pp_size=server_config.pp_size,
            tp_rank=0,
            tp_size=server_config.tp_size,
        )
        
        # Add a single decode token (dummy for now)
        decoding_input_ids = torch.tensor([0], dtype=torch.long)  # dummy token
        add_decoding_ids_to_batch_state(
            batch_state, decoding_input_ids, tp_rank=0, tp_size=server_config.tp_size
        )
        
        # Move to device
        move_batch_state(batch_state, device=device)
        
        # Plan attention
        toka_model.plan(batch_state.attention_info, non_blocking=False)
        
        with torch.no_grad():
            # Run the model
            output_batch_state = toka_model(batch_state)
            
            # Extract logits from the output
            if output_batch_state.output_ids is not None and output_batch_state.logprobs is not None:
                # If we have direct logprobs, use them
                toka_logprobs = output_batch_state.logprobs
                print(f"Got logprobs directly: {toka_logprobs.shape}")
            else:
                print("No logprobs in output, trying to extract from hidden states")
                return
        
        # The logprobs returned are just the chosen token logprobs, not the full distribution
        # Let's get the actual token that was sampled
        if output_batch_state.output_ids is not None:
            toka_token_id = output_batch_state.output_ids[0].item()
            toka_logprob = toka_logprobs[0].item()
            
            print(f"Toka sampled token ID: {toka_token_id}")
            print(f"Toka logprob for sampled token: {toka_logprob}")
            
            # For comparison, let's get the HF logprob for the same token
            hf_logprob_for_toka_token = hf_last_logprobs[toka_token_id].item()
            
            print(f"\nComparison:")
            print(f"HF top token: {hf_top_tokens.indices[0].item()} ({hf_tokenizer.decode([hf_top_tokens.indices[0].item()])})")
            print(f"Toka sampled token: {toka_token_id} ({hf_tokenizer.decode([toka_token_id])})")
            print(f"HF logprob for Toka's token: {hf_logprob_for_toka_token:.6f}")
            print(f"Toka logprob: {toka_logprob:.6f}")
            print(f"Logprob difference: {abs(hf_logprob_for_toka_token - toka_logprob):.6f}")
            
            # Check if they match (temperature=0.0 should give greedy)
            hf_top_token = hf_top_tokens.indices[0].item()
            if toka_token_id == hf_top_token:
                print("✅ Tokens match! (Both chose the same greedy token)")
                logprob_diff = abs(hf_top_tokens.values[0].item() - toka_logprob)
                if logprob_diff > 0.1:
                    print("❌ But logprobs differ significantly!")
                else:
                    print("✅ Logprobs also match!")
            else:
                print("❌ Tokens don't match - numerics issue detected!")
                print(f"Token ID difference: {abs(hf_top_token - toka_token_id)}")
        else:
            print("No output token IDs")
            return
        
    except Exception as e:
        print(f"Error running Tokasaurus model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2-0.5B"
    compare_models(model_name)