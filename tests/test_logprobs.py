import ast
import os
import shlex

import pydra
import pytest
import torch
import torch.multiprocessing as mp
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from tokasaurus.common_types import ServerConfig
from tokasaurus.entry import server_manager
from tokasaurus.utils import find_free_port

MODEL = os.environ.get(
    "MODEL",
    "Qwen/Qwen2-0.5B-Instruct",
)
OVERRIDES = os.environ.get("OVERRIDES", None)
MEAN_REL_TOL_LIMIT = ast.literal_eval(os.environ.get("MEAN_REL_TOL_LIMIT", "0.1"))
MAX_ABS_TOL_LIMIT = ast.literal_eval(os.environ.get("MAX_ABS_TOL_LIMIT", "0.2"))
TOKEN_MATCH_LIMIT = ast.literal_eval(os.environ.get("TOKEN_MATCH_LIMIT", "0.95"))


def make_basic_config():
    print(f"Making basic config for {MODEL}...")
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

    return config


@pytest.fixture(scope="module", params=[make_basic_config()])
def client(request):
    mp.set_start_method("spawn", force=True)

    config: ServerConfig = request.param
    print(f"Launching server with config: {config.to_dict()}")

    with server_manager(config):
        client = OpenAI(
            api_key="beepboop", base_url=f"http://localhost:{config.port}/v1"
        )

        yield client


@pytest.fixture(scope="module")
def hf_model_and_tokenizer() -> tuple[torch.nn.Module, PreTrainedTokenizer]:
    print(f"Loading HF model and tokenizer ({MODEL})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()
    print("Loaded HF model and tokenizer.")
    return model, tokenizer


PROMPTS = {
    "abc": "Please repeat the following pattern:"
    + "a b c d e f g h i j k l m n o p q r s a b c d e f g h i j k l m n o p q r s"
    * 10,
    "story": "Please tell me a long story about a cat.",
}


@pytest.mark.parametrize("prompt_name", list(PROMPTS.keys()))
def test_logprobs(
    client: OpenAI,
    hf_model_and_tokenizer: tuple[torch.nn.Module, PreTrainedTokenizer],
    prompt_name: str,
):
    prompt = PROMPTS[prompt_name]
    response = client.chat.completions.create(
        model="none",
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=64,
        temperature=0.0,
        logprobs=True,
    )
    model, tokenizer = hf_model_and_tokenizer

    for idx, choice in enumerate(response.choices):
        api_tokens = [token_logprob.token for token_logprob in choice.logprobs.content]
        logprobs = [token_logprob.logprob for token_logprob in choice.logprobs.content]

        seq_ids = tokenizer.convert_tokens_to_ids(api_tokens)

        input_ids = (
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                ],
                add_generation_prompt=True,
            )
            + seq_ids
        )
        with torch.inference_mode():
            input_tensor = torch.tensor(input_ids).unsqueeze(0).to("cuda:0")
            outputs = model(input_tensor)

        logits = outputs.logits.to(torch.float32)  # shape [1, seq_len, vocab_size]
        hf_logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

        token_matches = []
        logprob_adiffs = []
        logprob_rdiffs = []

        for idx, (api_token_id, hf_logprob_dist, api_logprob) in enumerate(
            zip(seq_ids, hf_logprobs[0, -len(seq_ids) - 1 : -1], logprobs)
        ):
            hf_logprob = hf_logprob_dist[api_token_id].item()
            hf_token_id = hf_logprob_dist.argmax().item()

            token_match = hf_token_id == api_token_id
            token_matches.append(token_match)

            adiff = abs(api_logprob - hf_logprob)
            rdiff = 2 * adiff / (abs(api_logprob) + abs(hf_logprob) + 1e-3)
            logprob_adiffs.append(adiff)
            logprob_rdiffs.append(rdiff)
            print(
                f"Pos {idx}: token match: {token_match}, logprob adiff: {adiff:.4f}, rdiff: {rdiff:.4f} (API: token={api_token_id} logprob={api_logprob:.4f}, HF: token={hf_token_id} logprob={hf_logprob:.4f})"
            )

        token_match_rate = sum(token_matches) / len(token_matches)
        max_adiff = max(logprob_adiffs)
        mean_rdiff = sum(logprob_rdiffs) / len(logprob_rdiffs)

        print(f"Token match rate: {token_match_rate:.4f}")
        print(f"Max logprob absolute diff: {max_adiff:.4f}")
        print(f"Mean logprob relative diff: {mean_rdiff:.4f}")

        if TOKEN_MATCH_LIMIT is not None:
            assert token_match_rate >= TOKEN_MATCH_LIMIT, (
                f"Token match rate: {token_match_rate} < {TOKEN_MATCH_LIMIT}"
            )

        if MEAN_REL_TOL_LIMIT is not None:
            assert mean_rdiff <= MEAN_REL_TOL_LIMIT, (
                f"Mean logprob relative diff: {mean_rdiff} > {MEAN_REL_TOL_LIMIT}"
            )

        if MAX_ABS_TOL_LIMIT is not None:
            assert max_adiff <= MAX_ABS_TOL_LIMIT, (
                f"Max logprob absolute diff: {max_adiff} > {MAX_ABS_TOL_LIMIT}"
            )
