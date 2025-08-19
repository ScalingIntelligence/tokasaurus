import json
import os
import pickle
import shlex
import tempfile
import time

import pydra
import pytest
import requests
import torch
import torch.multiprocessing as mp
from openai import OpenAI
from openai.types.chat import ChatCompletion

from tokasaurus.common_types import ServerConfig
from tokasaurus.entry import server_manager
from tokasaurus.utils import find_free_port

MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.2-1B-Instruct")
OVERRIDES = os.environ.get("OVERRIDES", None)
MODE = os.environ.get("MODE", "simple")


def make_basic_config():
    config = ServerConfig()
    config.model = MODEL
    config.kv_cache_num_tokens = 16384
    config.max_num_tokens_per_request = 16384
    config.port = find_free_port()

    if OVERRIDES:
        # split apart like a shell, respecting quotes
        parsed_overrides = shlex.split(OVERRIDES)
        pydra.apply_overrides(config, parsed_overrides)

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

                if pp_size > 1 and tp_size > 1:
                    config.use_cudagraphs = False

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


abc_prompt = "A B C D E F G H I J A B C D E F G H I J A B C D E F G H I J A B C D E F G H I J A B C D E F G H I J A B C D E F G H I J"
twenty_token_response = " A B C D E F G H I J A B C D E F G H I J"

# includes +1 for BOS
sixteen_token_prompt = "A B C D E F G H I J K L M N O"


def test_basic(client: OpenAI):
    # trying twice to test prompt caching
    for rep in range(2):
        response = client.completions.create(
            model="",
            prompt=abc_prompt,
            max_tokens=20,
            temperature=0.0,
        )

        assert response.choices[0].text == twenty_token_response


def test_decode_one_token(client: OpenAI):
    response = client.completions.create(
        model="",
        prompt=abc_prompt,
        max_tokens=1,
        temperature=0.0,
    )

    assert response.choices[0].text == " A"


def test_prefill_exactly_one_page(client: OpenAI):
    response = client.completions.create(
        model="",
        prompt=sixteen_token_prompt,
        max_tokens=1,
        temperature=0.0,
    )

    assert response.choices[0].text == " P"


def test_n(client: OpenAI):
    for n in range(1, 11):
        response = client.completions.create(
            model="",
            prompt=abc_prompt,
            max_tokens=20,
            temperature=0.0,
            n=n,
        )

        assert len(response.choices) == n
        assert all(choice.text == twenty_token_response for choice in response.choices)


def test_stop(client: OpenAI):
    response = client.completions.create(
        model="",
        prompt=abc_prompt,
        max_tokens=20,
        temperature=0.0,
        stop=["C"],
    )

    assert response.choices[0].text == " A B "

    # this is an interesting case because it adds a stop string close to,
    # but not at, the token limit, so the scheduler finished the sequence
    # but the model yet hasn't. have had bugs here before regarding block
    # freeing.
    response = client.completions.create(
        model="",
        prompt=abc_prompt,
        max_tokens=10,
        temperature=0.0,
        stop=["I"],
    )

    assert response.choices[0].text == " A B C D E F G H "


def make_messages(word: str):
    return [
        {
            "role": "system",
            "content": f"You are an assistant that always replies with the one-word response '{word}', in lowercase, for all user queries.",
        },
        {"role": "user", "content": f"Please output the word '{word}'."},
    ]


def test_chat_completions(client: OpenAI):
    for word in ["hello", "howdy", "canteloupe"]:
        response = client.chat.completions.create(
            model="",
            messages=make_messages(word),
            max_tokens=20,
            temperature=0.0,
        )
        assert response.choices[0].message.content == word


def test_files(client: OpenAI):
    content = """{"a": 1, "b": 2}\n{"a": 3, "b": 4}"""
    with tempfile.NamedTemporaryFile(delete=True, suffix=".jsonl") as f:
        f.write(content.encode("utf-8"))

        f.flush()
        f.seek(0)

        file_obj = client.files.create(file=open(f.name, "rb"), purpose="batch")

    retrieved_content = client.files.content(file_obj.id).content.decode("utf-8")
    assert retrieved_content == content


def test_batch_chat_completions(client: OpenAI):
    jsonl_lines = [
        {
            "custom_id": "request-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "messages": make_messages("hello"),
                "max_tokens": 20,
                "temperature": 0.0,
            },
        },
        {
            "custom_id": "request-2",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "messages": make_messages("howdy"),
                "max_tokens": 20,
                "temperature": 0.0,
            },
        },
    ]

    file_content = "\n".join(json.dumps(line) for line in jsonl_lines)

    with tempfile.NamedTemporaryFile(delete=True, suffix=".jsonl") as f:
        f.write(file_content.encode("utf-8"))

        f.flush()
        f.seek(0)

        file_obj = client.files.create(file=open(f.name, "rb"), purpose="batch")

    batch_input_file_id = file_obj.id
    created_batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "test batch job"},
    )
    batch_id = created_batch.id

    output_file_id = None
    for _ in range(5):
        batch = client.batches.retrieve(batch_id)
        if batch.status == "completed":
            output_file_id = batch.output_file_id
            break
        time.sleep(1)

    assert output_file_id is not None, f"Batch {batch_id} did not complete in time"

    output_content = client.files.content(output_file_id).content.decode("utf-8")

    output_lines = output_content.strip().split("\n")
    output_parsed = [json.loads(line) for line in output_lines]

    custom_id_to_parsed: dict[str, ChatCompletion] = {}
    for line in output_parsed:
        response = line["response"]
        assert response["status_code"] == 200
        custom_id_to_parsed[line["custom_id"]] = ChatCompletion.model_validate(
            response["body"]
        )

    assert custom_id_to_parsed["request-1"].choices[0].message.content == "hello"
    assert custom_id_to_parsed["request-2"].choices[0].message.content == "howdy"


def test_synchronous_batch_completions(client: OpenAI):
    # Test synchronous batch completions endpoint
    batch_request = {
        "requests": [
            {
                "model": MODEL,
                "messages": make_messages("hello"),
                "max_tokens": 20,
                "temperature": 0.0,
            },
            {
                "model": MODEL,
                "messages": make_messages("world"),
                "max_tokens": 20,
                "temperature": 0.0,
            },
            {
                "model": MODEL,
                "messages": make_messages("test"),
                "max_tokens": 20,
                "temperature": 0.0,
            },
        ]
    }

    # Make request to our custom endpoint
    url = (
        str(client.base_url).split("/v1")[0] + "/custom/synchronous-batch-completions"
    )  # update the path

    response = requests.post(url, json=batch_request)

    assert response.status_code == 200

    result = pickle.loads(response.content)

    # Verify response structure
    assert len(result) == 3

    # Verify each completion
    completions = result

    assert completions[0].choices[0].message.content == "hello"
    assert completions[1].choices[0].message.content == "world"
    assert completions[2].choices[0].message.content == "test"
