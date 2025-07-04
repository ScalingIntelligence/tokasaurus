import base64
import json
import os
import shlex

import numpy as np
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


def test_packed_vs_normal_logprobs(client: OpenAI):
    """Test that packed format and normal OpenAI format produce identical results for the same request"""

    # Use a simple prompt and greedy decoding for deterministic results
    prompt = "What is the capital of France?"
    k = 3  # Number of top logprobs to request
    max_tokens = 10

    # Make request with normal OpenAI logprobs format
    normal_response = client.chat.completions.create(
        model="",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,  # Greedy decoding
        logprobs=True,
        top_logprobs=k,
    )

    # Make identical request with packed format
    packed_response = client.chat.completions.create(
        model="",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,  # Greedy decoding
        logprobs=True,
        top_logprobs=k,
        extra_body=dict(logprobs_in_fingerprint=True),
    )

    # Extract normal logprobs
    normal_logprobs = normal_response.choices[0].logprobs
    assert normal_logprobs is not None
    assert normal_logprobs.content is not None

    # Extract packed logprobs from fingerprint
    assert packed_response.system_fingerprint is not None
    assert packed_response.choices[0].logprobs is None
    fingerprint_data = json.loads(packed_response.system_fingerprint)

    # Verify fingerprint contains expected fields
    assert "completion_ids" in fingerprint_data
    assert "packed_chosen_logprobs" in fingerprint_data
    assert "packed_topk_indices" in fingerprint_data
    assert "packed_topk_logprobs" in fingerprint_data

    # Decode packed data
    chosen_logprobs = np.frombuffer(
        base64.b64decode(fingerprint_data["packed_chosen_logprobs"][0]),
        dtype=np.float32,
    )
    topk_ids = np.frombuffer(
        base64.b64decode(fingerprint_data["packed_topk_indices"][0]), dtype=np.int32
    ).reshape(-1, k)
    topk_logprobs = np.frombuffer(
        base64.b64decode(fingerprint_data["packed_topk_logprobs"][0]), dtype=np.float32
    ).reshape(-1, k)

    assert topk_ids.shape == topk_logprobs.shape

    # Number of tokens should match
    num_tokens = len(chosen_logprobs)
    assert num_tokens == len(normal_logprobs.content)

    # Compare token by token
    for i, normal_token in enumerate(normal_logprobs.content):
        # Chosen logprob should match
        assert abs(normal_token.logprob - chosen_logprobs[i]) < 1e-6

        # Number of top logprobs should match
        assert len(normal_token.top_logprobs) == k

        # Compare each top logprob
        for j, normal_top in enumerate(normal_token.top_logprobs):
            packed_logprob = topk_logprobs[i][j]

            # Logprob values should be identical (within floating point tolerance)
            assert abs(normal_top.logprob - packed_logprob) < 1e-6, (
                f"Token {i}, top-{j}: normal={normal_top.logprob}, packed={packed_logprob}"
            )

    # Verify completion text is identical
    assert (
        normal_response.choices[0].message.content
        == packed_response.choices[0].message.content
    )
