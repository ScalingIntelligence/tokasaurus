import json
import os
import shlex
import tempfile
import time
import math
import pydra
import pytest
import torch
import torch.multiprocessing as mp
from openai import OpenAI
from openai.types.chat import ChatCompletion
from transformers import AutoModelForCausalLM, AutoTokenizer

from tokasaurus.common_types import ServerConfig
from tokasaurus.entry import server_manager
from tokasaurus.utils import find_free_port


MODEL = os.environ.get(
    "MODEL", 
    # "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2-0.5B-Instruct"
)
OVERRIDES = os.environ.get("OVERRIDES", None)
MODE = os.environ.get("MODE", "simple")
PROB_REL_TOL = float(os.environ.get("PROB_REL_TOL", 0.1))
PROB_ABS_TOL = float(os.environ.get("PROB_ABS_TOL", 0.1))


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
        pydra.apply_overrides(config, parsed_overrides, init_annotations=False)

    return config


def simple_configs():
    return [
        make_basic_config(),
    ]


def multi_gpu_configs():
    npgus = torch.cuda.device_count()
    configs = []
    for dp_size in [1, 2]:
        for pp_size in [1, 2]:
            for tp_size in [1, 2]:
                if dp_size * pp_size * tp_size > npgus:
                    continue

                config = make_basic_config()
                config.dp_size = dp_size
                config.pp_size = pp_size
                config.tp_size = tp_size
                configs.append(config)

    return configs


match MODE:
    case "simple":
        configs = simple_configs()
    case "multigpu":
        configs = multi_gpu_configs()
    case _:
        raise ValueError(f"Invalid mode: {MODE}")



@pytest.fixture(scope="module", params=configs)
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
def hf_model_and_tokenizer() -> tuple[torch.nn.Module, AutoTokenizer]:
    print(f"Loading HF model and tokenizer ({MODEL})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model.eval()
    model.to("cuda", dtype=torch.bfloat16)
    print("Loaded HF model and tokenizer.")
    return model, tokenizer


PROMPTS = {
    "abc": "Please repeat the following pattern:" + "a b c d e f g h i j k l m n o p q r s a b c d e f g h i j k l m n o p q r s" * 10,
    "story": "Can you tell me a long story about a cat?",
    
}

@pytest.mark.parametrize("prompt_name", list(PROMPTS.keys()))
def test_logprobs(client: OpenAI, hf_model_and_tokenizer: tuple[torch.nn.Module, AutoTokenizer], prompt_name: str):
    prompt = PROMPTS[prompt_name]
    response = client.chat.completions.create(
        model="none",
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=16,
        temperature=0.0,
        logprobs=1, 
    )
    model, tokenizer = hf_model_and_tokenizer
        
    for idx, choice in enumerate(response.choices):
        api_tokens = [token_logprob.token for token_logprob in choice.logprobs.content]
        logprobs = [token_logprob.logprob for token_logprob in choice.logprobs.content]
        
        token_ids = json.loads(response.system_fingerprint)["completion_ids"][idx]

        seq_ids = tokenizer.convert_tokens_to_ids(api_tokens)

        input_ids = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
            ],
            add_generation_prompt=True,
        ) + seq_ids
        with torch.inference_mode():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                input_tensor = torch.tensor(input_ids).unsqueeze(0).to("cuda")
                outputs = model(input_tensor)
        
        logits = outputs.logits.to(torch.float32)  # shape [1, seq_len, vocab_size]
        hf_logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        for idx, (api_token_id, hf_logprob_dist, api_logprob) in enumerate(zip(seq_ids, hf_logprobs[0, -len(seq_ids) - 1: -1], logprobs)):
            hf_logprob = hf_logprob_dist[api_token_id].item()
            hf_token_id = hf_logprob_dist.argmax().item()
            # print(f"API token id: {api_token_id}, HF token id: {hf_token_id}")
            # print(f"API logprob: {api_logprob}, HF logprob: {hf_logprob}")
            assert hf_token_id == api_token_id, f"Mismatch token id (after {idx} tokens): API {api_token_id} ({tokenizer.decode(api_token_id)}) vs HF {hf_token_id} ({tokenizer.decode(hf_token_id)})"
            assert math.isclose(api_logprob, hf_logprob, abs_tol=PROB_ABS_TOL, rel_tol=PROB_REL_TOL), f"Mismatch probability (after {idx} tokens): API {api_logprob} vs HF {hf_logprob}"