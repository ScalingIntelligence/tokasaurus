import os
import shlex

import pydra
import pytest
import torch.multiprocessing as mp
from openai import OpenAI

from tokasaurus.common_types import ServerConfig
from tokasaurus.entry import server_manager
from tokasaurus.utils import find_free_port

MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.2-1B-Instruct")
OVERRIDES = os.environ.get("OVERRIDES", None)


def make_config():
    config = ServerConfig()
    config.model = MODEL
    config.kv_cache_num_tokens = 16384
    config.max_num_tokens_per_request = 16384
    config.max_seqs_per_forward = 1024
    config.port = find_free_port()

    if OVERRIDES:
        # split apart like a shell, respecting quotes
        parsed_overrides = shlex.split(OVERRIDES)
        pydra.apply_overrides(config, parsed_overrides)

    # Enable logprobs features for topk testing
    config.enable_chosen_logprobs = True
    config.max_topk_logprobs = 5

    return config


def _client():
    mp.set_start_method("spawn", force=True)

    config = make_config()
    print(f"Launching server with config: {config.to_dict()}")

    with server_manager(config):
        client = OpenAI(
            api_key="beepboop", base_url=f"http://localhost:{config.port}/v1"
        )
        yield client

@pytest.fixture(scope="module")
def client():
    yield from _client()



# Test prompts
abc_prompt = "A B C D E F G H I J A B C D E F G H I J A B C D E F G H I J A B C D E F G H I J A B C D E F G H I J A B C D E F G H I J"


def test_completions_greedy_logprobs_matches_top1(client: OpenAI):
    """Test that greedy sampling matches top-1 logprobs for completions API"""
    response = client.completions.create(
        model="",
        prompt=abc_prompt,
        max_tokens=10,
        temperature=0.0,
        logprobs=5,
    )

    assert len(response.choices) == 1
    choice = response.choices[0]

    # Check that logprobs are present and populated
    assert choice.logprobs is not None
    assert choice.logprobs.token_logprobs is not None
    assert choice.logprobs.tokens is not None
    assert choice.logprobs.top_logprobs is not None

    # Check lengths match
    assert len(choice.logprobs.token_logprobs) == len(choice.logprobs.tokens)
    assert len(choice.logprobs.token_logprobs) == len(choice.logprobs.top_logprobs)

    # Check that we got the expected number of top logprobs for each token
    for i, (greedy_token, top_logprobs) in enumerate(
        zip(choice.logprobs.tokens, choice.logprobs.top_logprobs)
    ):
        assert len(top_logprobs) == 5  # We requested 5 top logprobs

        # Verify logprobs are in descending order
        logprob_values = list(top_logprobs.values())
        assert logprob_values == sorted(logprob_values, reverse=True)

        # The top-1 token should match the greedily selected token
        top1_token = list(top_logprobs.keys())[0]
        assert top1_token == greedy_token

        # The top-1 logprob should match the token logprob
        top1_logprob = list(top_logprobs.values())[0]
        assert abs(top1_logprob - choice.logprobs.token_logprobs[i]) < 1e-6


def test_chat_completions_greedy_logprobs_matches_top1(client: OpenAI):
    """Test that greedy sampling matches top-1 logprobs for chat completions API"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]

    response = client.chat.completions.create(
        model="",
        messages=messages,
        max_tokens=10,
        temperature=0.0,
        logprobs=True,
        top_logprobs=5,
    )

    assert len(response.choices) == 1
    choice = response.choices[0]

    # Check that logprobs are present
    assert choice.logprobs is not None
    assert choice.logprobs.content is not None

    # Check each token logprob
    for token_logprob in choice.logprobs.content:
        assert token_logprob.token is not None
        assert token_logprob.logprob is not None
        assert token_logprob.top_logprobs is not None
        assert len(token_logprob.top_logprobs) == 5  # We requested 5 top logprobs

        # Verify top logprobs are in descending order
        top_logprobs = token_logprob.top_logprobs
        for i in range(len(top_logprobs) - 1):
            assert top_logprobs[i].logprob >= top_logprobs[i + 1].logprob

        # The top-1 token should match the selected token
        assert top_logprobs[0].token == token_logprob.token

        # The top-1 logprob should match the token logprob
        assert abs(top_logprobs[0].logprob - token_logprob.logprob) < 1e-6


def test_logprobs_in_fingerprint_compression():
    """Test the compression/decompression functions directly"""
    import numpy as np
    from tokasaurus.manager.types import SequenceOutput
    from tokasaurus.server.utils import compress_logprobs_data, decompress_logprobs_data
    
    # Create mock sequence output with topk data
    seq_out = SequenceOutput()
    seq_out.topk_ids = [
        np.array([10, 20, 30], dtype=np.int32),
        np.array([40, 50, 60], dtype=np.int32),
    ]
    seq_out.topk_logprobs = [
        np.array([-1.0, -2.0, -3.0], dtype=np.float32),
        np.array([-0.5, -1.5, -2.5], dtype=np.float32),
    ]
    
    # Test compression
    compressed_data = compress_logprobs_data([seq_out])
    assert isinstance(compressed_data, bytes)
    assert len(compressed_data) > 0
    
    # Test decompression
    decompressed = decompress_logprobs_data(compressed_data)
    assert len(decompressed) == 1
    
    topk_ids, topk_logprobs = decompressed[0]
    assert len(topk_ids) == 2
    assert len(topk_logprobs) == 2
    
    # Check first token
    assert topk_ids[0] == [10, 20, 30]
    assert abs(topk_logprobs[0][0] - (-1.0)) < 1e-2
    assert abs(topk_logprobs[0][1] - (-2.0)) < 1e-2
    assert abs(topk_logprobs[0][2] - (-3.0)) < 1e-2
    
    # Check second token  
    assert topk_ids[1] == [40, 50, 60]
    assert abs(topk_logprobs[1][0] - (-0.5)) < 1e-2
    assert abs(topk_logprobs[1][1] - (-1.5)) < 1e-2
    assert abs(topk_logprobs[1][2] - (-2.5)) < 1e-2
    
    
def test_logprobs_compression_multiple_sequences():
    """Test compression with multiple sequences including empty ones"""
    import numpy as np
    from tokasaurus.manager.types import SequenceOutput
    from tokasaurus.server.utils import compress_logprobs_data, decompress_logprobs_data
    
    # Create multiple sequence outputs
    seq1 = SequenceOutput()
    seq1.topk_ids = [np.array([1, 2], dtype=np.int32)]
    seq1.topk_logprobs = [np.array([-0.1, -0.2], dtype=np.float32)]
    
    seq2 = SequenceOutput()  # Empty sequence
    seq2.topk_ids = []
    seq2.topk_logprobs = []
    
    seq3 = SequenceOutput()
    seq3.topk_ids = [
        np.array([100, 110, 120], dtype=np.int32),
        np.array([200, 300, 400], dtype=np.int32)
    ]
    seq3.topk_logprobs = [
        np.array([-5.0, -5.1, -5.2], dtype=np.float32),
        np.array([-6.0, -7.0, -8.0], dtype=np.float32)
    ]
    
    # Test compression and decompression
    compressed = compress_logprobs_data([seq1, seq2, seq3])
    decompressed = decompress_logprobs_data(compressed)
    
    assert len(decompressed) == 3
    
    # Check seq1
    ids1, logprobs1 = decompressed[0]
    assert len(ids1) == 1
    assert ids1[0] == [1, 2]
    assert abs(logprobs1[0][0] - (-0.1)) < 1e-2
    assert abs(logprobs1[0][1] - (-0.2)) < 1e-2
    
    # Check seq2 (empty)
    ids2, logprobs2 = decompressed[1]
    assert len(ids2) == 0
    assert len(logprobs2) == 0
    
    # Check seq3
    ids3, logprobs3 = decompressed[2]
    assert len(ids3) == 2
    assert ids3[0] == [100, 110, 120]
    assert ids3[1] == [200, 300, 400]
    assert abs(logprobs3[0][0] - (-5.0)) < 1e-2
    assert abs(logprobs3[0][1] - (-5.1)) < 1e-2
    assert abs(logprobs3[0][2] - (-5.2)) < 1e-2
    assert abs(logprobs3[1][0] - (-6.0)) < 1e-2
    assert abs(logprobs3[1][1] - (-7.0)) < 1e-2
    assert abs(logprobs3[1][2] - (-8.0)) < 1e-2


def test_logprobs_in_fingerprint_end_to_end(client: OpenAI):
    """Test that logprobs_in_fingerprint=True produces compressed data that matches regular logprobs"""
    import json
    import base64
    from tokasaurus.server.utils import decompress_logprobs_data
    
    # Make a request with logprobs enabled
    response = client.chat.completions.create(
        model="",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=5,
        temperature=0.0,
        logprobs=True,
        top_logprobs=3,
        extra_body=dict(logprobs_in_fingerprint=True),
    )
    
    assert len(response.choices) == 1
    choice = response.choices[0]
    
    # Check that regular logprobs are STILL present (fingerprint mode does disable them)
    assert choice.logprobs is None
    
    # Check that fingerprint contains compressed logprobs
    assert response.system_fingerprint is not None
    fingerprint_data = json.loads(response.system_fingerprint)
    assert "completion_ids" in fingerprint_data
    assert "logprobs_compressed" in fingerprint_data
    
    # Decompress the logprobs data
    compressed_data = fingerprint_data["logprobs_compressed"].encode('ascii')
    decompressed_sequences = decompress_logprobs_data(compressed_data)
    
    # Should have one sequence (n=1)
    assert len(decompressed_sequences) == 1
    topk_ids, topk_logprobs = decompressed_sequences[0]
    
    # Should have the expected number of tokens (up to max_tokens=5)
    assert len(topk_ids) <= 5
    assert len(topk_logprobs) <= 5
    assert len(topk_ids) == len(topk_logprobs)
    
    # Each token should have up to max_topk_logprobs=5 (configured limit)
    for i, (token_ids, token_logprobs) in enumerate(zip(topk_ids, topk_logprobs)):
        assert len(token_ids) <= 5
        assert len(token_logprobs) <= 5
        assert len(token_ids) == len(token_logprobs)
        
        # Verify logprobs are in descending order
        assert token_logprobs == sorted(token_logprobs, reverse=True)
        
        # Verify token IDs are positive integers
        for token_id in token_ids:
            assert isinstance(token_id, int)
            assert token_id >= 0
        
        # Verify logprobs are negative floats (since they're log probabilities)
        for logprob in token_logprobs:
            assert isinstance(logprob, float)
            assert logprob <= 0.0  # log probabilities should be <= 0


def test_fingerprint_vs_regular_logprobs_comparison(client: OpenAI):
    """Compare regular logprobs vs fingerprint logprobs to ensure they match"""
    import json
    from tokasaurus.server.utils import decompress_logprobs_data
    
    prompt = "What is 2+2?"
    
    # Get regular logprobs response
    regular_response = client.chat.completions.create(
        model="",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3,
        temperature=0.0,
        logprobs=True,
        top_logprobs=5,  # Use max configured value
    )
    
    # Get fingerprint logprobs response
    fingerprint_response = client.chat.completions.create(
        model="",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3,
        temperature=0.0,
        logprobs=True,
        top_logprobs=5,  # Use max configured value
        extra_body=dict(logprobs_in_fingerprint=True),
    )
    
    # Extract regular logprobs
    regular_logprobs = regular_response.choices[0].logprobs
    assert regular_logprobs is not None
    assert regular_logprobs.content is not None
    
    # Extract fingerprint logprobs
    fingerprint_data = json.loads(fingerprint_response.system_fingerprint)
    compressed_data = fingerprint_data["logprobs_compressed"].encode('ascii')
    decompressed_sequences = decompress_logprobs_data(compressed_data)
    topk_ids, topk_logprobs = decompressed_sequences[0]
    
    # Compare token count
    assert len(regular_logprobs.content) == len(topk_ids)
    
    # Compare each token's top logprobs
    for i, regular_token in enumerate(regular_logprobs.content):
        fingerprint_token_logprobs = topk_logprobs[i]
        
        # Should have same number of top logprobs
        assert len(regular_token.top_logprobs) == len(fingerprint_token_logprobs)
        
        # Compare logprob values (allowing for small floating point differences)
        for j, regular_top in enumerate(regular_token.top_logprobs):
            breakpoint()
            fingerprint_logprob = fingerprint_token_logprobs[j]
            assert abs(regular_top.logprob - fingerprint_logprob) < 1e-2
