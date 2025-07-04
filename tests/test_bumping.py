import os

import pytest
import torch.multiprocessing as mp
from openai import OpenAI

from tokasaurus.common_types import ServerConfig
from tokasaurus.entry import server_manager
from tokasaurus.utils import find_free_port

MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.2-1B-Instruct")


@pytest.fixture(scope="module")
def client():
    mp.set_start_method("spawn", force=True)

    port = find_free_port()
    config = ServerConfig()
    config.model = MODEL
    config.kv_cache_num_tokens = 16384
    config.max_num_tokens_per_request = 16384
    config.port = port
    config.page_size = 16
    config.track_early_stopping = True
    config.use_spec_allocation = True

    with server_manager(config):
        client = OpenAI(
            api_key="beepboop", base_url=f"http://localhost:{config.port}/v1"
        )

        yield client


def test_bumping(client: OpenAI):
    a_through_j = " A B C D E F G H I J"

    abc_prompt = (a_through_j * 10).strip()
    hundred_token_response = a_through_j * 10

    # first we make send enough sequences to
    # set the early stopping tracker
    response = client.completions.create(
        model="", prompt=abc_prompt, max_tokens=100, temperature=0.0, n=1024, stop=["C"]
    )

    for c in response.choices:
        assert c.text == " A B "

    print("Done with first request - no bumping should have occurred yet")

    # now we don't use stop strings to cause bumping
    response2 = client.completions.create(
        model="", prompt=abc_prompt, max_tokens=100, temperature=0.0, n=1024
    )

    for c in response2.choices:
        assert c.text == hundred_token_response
