import asyncio
import base64
import functools
import json
from dataclasses import dataclass, field
import time
from uuid import uuid4

import numpy as np
from fastapi import HTTPException, Request
from loguru import logger
from openai.types.batch import Batch, BatchRequestCounts
from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionTokenLogprob,
    Choice,
    ChoiceLogprobs,
)
from openai.types.chat.chat_completion_token_logprob import TopLogprob
from openai.types.completion import (
    Completion,
    CompletionChoice,
)
from openai.types.completion_choice import Logprobs
from openai.types.completion_usage import (
    CompletionUsage,
    PromptTokensDetails,
)
from openai.types.file_object import FileObject
from tokenizers import Tokenizer
from transformers import AutoTokenizer, GenerationConfig

from tokasaurus.common_types import (
    Engine,
    ServerConfig,
)
from tokasaurus.manager.types import SequenceOutput
from tokasaurus.server.types import (
    BatchFileLine,
    CancelledRequest,
    ChatCompletionRequest,
    CompletionsRequest,
    FileEntry,
    RequestOutput,
    SamplingParams,
    SubmittedBatch,
    SubmittedRequest,
    CartridgeCompletionsRequest,
    CartridgeChatCompletionRequest,
    TokasaurusRequest,
    nowstamp,
)
from tokasaurus.utils import get_eos_token_ids


LLAMA_CARTRIDGE_TEMPLATE = """\
{%- for message in messages %}
    {%- if  (message.role == 'assistant') %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}{% generation %}{{- message['content'] | trim + '<|eot_id|>' }}{% endgeneration %}

    {%- else %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
        
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"""


async def listen_for_disconnect(request: Request) -> None:
    """
    Returns if a disconnect message is received.

    Taken from vLLM.
    """
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            break


def with_cancellation(handler_func):
    """
    Decorator that allows a route handler to be cancelled by client
    disconnections.

    This does _not_ use request.is_disconnected, which does not work with
    middleware. Instead this follows the pattern from
    starlette.StreamingResponse, which simultaneously awaits on two tasks- one
    to wait for an http disconnect message, and the other to do the work that we
    want done. When the first task finishes, the other is cancelled.

    A core assumption of this method is that the body of the request has already
    been read. This is a safe assumption to make for fastapi handlers that have
    already parsed the body of the request into a pydantic model for us.
    This decorator is unsafe to use elsewhere, as it will consume and throw away
    all incoming messages for the request while it looks for a disconnect
    message.

    In the case where a `StreamingResponse` is returned by the handler, this
    wrapper will stop listening for disconnects and instead the response object
    will start listening for disconnects.

    Taken from vLLM.
    """

    # Functools.wraps is required for this wrapper to appear to fastapi as a
    # normal route handler, with the correct request type hinting.
    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):
        # The request is either the second positional arg or `raw_request`
        request = args[1] if len(args) > 1 else kwargs["raw_request"]

        handler_task = asyncio.create_task(handler_func(*args, **kwargs))
        cancellation_task = asyncio.create_task(listen_for_disconnect(request))

        done, pending = await asyncio.wait(
            [handler_task, cancellation_task], return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()

        if handler_task in done:
            return handler_task.result()
        return None

    return wrapper


class DefaultTokenDict(dict):
    """
    Models will make the size of the embedding/unembedding matrices
    be a clean number for efficiency reasons. However, the actual
    number of tokens in the vocab may be less than this. If you
    sample one of these missing tokens, you'll get an error.
    This class just replaces missing tokens with
    a string that indicates they're missing.
    """

    def __missing__(self, token_id):
        return f"<UNK_{token_id}>"


@dataclass
class ServerState:
    config: ServerConfig
    engines: list[Engine]
    process_name: str

    rid_to_req: dict[str, SubmittedRequest] = field(default_factory=dict)
    fid_to_file: dict[str, FileEntry] = field(default_factory=dict)
    bid_to_batch: dict[str, SubmittedBatch] = field(default_factory=dict)

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
        self.generation_config = GenerationConfig.from_pretrained(
            self.config.model, trust_remote_code=True
        )
        self.inverse_vocab = DefaultTokenDict(
            {token_id: token for token, token_id in self.tokenizer.vocab.items()}
        )
        self.requests_per_engine = [0] * len(self.engines)

        self.logger = logger.bind(process_name=self.process_name)

    def get_inner_tokenizer(self) -> Tokenizer:
        return self.tokenizer._tokenizer


async def receive_from_manager_loop(state: ServerState):
    while True:
        did_something = False
        for engine in state.engines:
            if not (q := engine.q_manager_to_server).empty():
                output: RequestOutput = q.get()
                state.rid_to_req[output.id].request_output = output
                state.rid_to_req[output.id].event.set()
                did_something = True

        if not did_something:
            await asyncio.sleep(0.01)


@logger.catch
async def handle_batch(state: ServerState, batch_id: str):
    batch = state.bid_to_batch[batch_id]
    event_waits = [item.submitted_req.event.wait() for item in batch.items]

    # wait for all the events to be set
    await asyncio.gather(*event_waits)

    # get the outputs
    results_lines = []
    for item in batch.items:
        assert item.submitted_req.request_output is not None
        match item.user_req:
            case CompletionsRequest():
                out = process_completions_output(
                    state,
                    item.user_req,
                    item.submitted_req.request,
                    item.submitted_req.request_output,
                )
            case ChatCompletionRequest():
                out = process_chat_completions_output(
                    state,
                    item.user_req,
                    item.submitted_req.request,
                    item.submitted_req.request_output,
                )
            case CartridgeCompletionsRequest():
                out = process_cartridge_completions_output(
                    state,
                    item.user_req,
                    item.submitted_req.request,
                    item.submitted_req.request_output,
                )
            case CartridgeChatCompletionRequest():
                out = process_cartridge_chat_completions_output(
                    state,
                    item.user_req,
                    item.submitted_req.request,
                    item.submitted_req.request_output,
                )

        line = {
            "id": "tokasaurus",
            "custom_id": item.line.custom_id,
            "response": {
                "status_code": 200,
                "request_id": item.submitted_req.request.id,
                "body": out.model_dump(),
            },
        }
        results_lines.append(json.dumps(line))

    file_content = "\n".join(results_lines)
    file_bytes = file_content.encode("utf-8")

    file_object = FileObject(
        id=str(uuid4()),
        bytes=len(file_bytes),
        filename="results.jsonl",
        purpose="batch_output",
        created_at=nowstamp(),
        status="uploaded",
        object="file",
    )

    output_file = FileEntry(
        content=file_bytes,
        details=file_object,
    )

    state.fid_to_file[file_object.id] = output_file
    batch.output_file = output_file


def submit_request(state: ServerState, request: TokasaurusRequest):
    # pick engine with least requests
    min_requests = min(state.requests_per_engine)
    engine_index = state.requests_per_engine.index(min_requests)
    engine = state.engines[engine_index]

    submitted = SubmittedRequest(request=request, engine_index=engine_index)
    state.rid_to_req[request.id] = submitted
    engine.q_server_to_manager.put(request)
    state.requests_per_engine[engine_index] += 1

    return submitted


def cancel_request(state: ServerState, submitted_req: SubmittedRequest):
    engine = state.engines[submitted_req.engine_index]
    engine.q_server_to_manager.put(CancelledRequest(submitted_req.request.id))
    state.requests_per_engine[submitted_req.engine_index] -= 1


def validate_length(state: ServerState, request: TokasaurusRequest):
    assert state.config.max_num_tokens_per_request is not None
    if (
        len(request.input_ids) + request.max_num_tokens
        > state.config.max_num_tokens_per_request
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Request has prompt length = {len(request.input_ids)} and max_tokens = {request.max_num_tokens}, "
            f"which is larger than the maximum number of tokens per request = {state.config.max_num_tokens_per_request}",
        )


def is_ids_list(x):
    return isinstance(x, list) and all(isinstance(i, int) for i in x)


def make_usage_info(request: TokasaurusRequest, output: RequestOutput):
    num_completion_tokens = sum(
        [len(o.completion_ids) for o in output.sequence_outputs]
    )

    cached_tokens = [o.num_cached_prompt_tokens for o in output.sequence_outputs]
    uncached_tokens = [len(request.input_ids) - c for c in cached_tokens]

    total_prompt_tokens = sum(cached_tokens) + sum(uncached_tokens)
    total_tokens = num_completion_tokens + total_prompt_tokens

    prompt_tokens_details = PromptTokensDetails(
        cached_tokens=sum(cached_tokens),
    )

    return CompletionUsage(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=num_completion_tokens,
        total_tokens=total_tokens,
        prompt_tokens_details=prompt_tokens_details,
    )


def make_completion_logprobs(
    crequest: CompletionsRequest,
    seq_out: SequenceOutput,
    inverse_vocab: dict[int, str],
):
    detok_list = [inverse_vocab[cid] for cid in seq_out.completion_ids]

    logprobs = crequest.logprobs
    assert logprobs is not None

    if logprobs > 0:
        top_logprobs_list = []
        for i in range(len(seq_out.completion_ids)):
            top_logprobs = {}
            for k in range(logprobs):
                token_id = seq_out.topk_ids[i][k]
                detok = inverse_vocab[token_id]
                top_logprobs[detok] = seq_out.topk_logprobs[i][k]
            top_logprobs_list.append(top_logprobs)
    else:
        top_logprobs_list = []

    logprobs_obj = Logprobs(
        token_logprobs=seq_out.logprobs,
        tokens=detok_list,
        top_logprobs=top_logprobs_list,
    )
    return logprobs_obj


def make_chat_logprobs(
    crequest: ChatCompletionRequest,
    seq_out: SequenceOutput,
    inverse_vocab: dict[int, str],
):
    topk = crequest.top_logprobs

    logprobs_list = []
    for i, cid in enumerate(seq_out.completion_ids):
        detok = inverse_vocab[cid]
        logprob = seq_out.logprobs[i]

        if topk is not None and topk > 0:
            # Build top_logprobs if top-k data is available
            top_logprobs_list = []

            topk_ids = seq_out.topk_ids[i].tolist()
            topk_logprobs = seq_out.topk_logprobs[i].tolist()

            assert len(topk_ids) == len(topk_logprobs)
            assert len(topk_ids) >= topk

            for k in range(topk):
                top_token = topk_ids[k]
                top_logprob = topk_logprobs[k]

                top_detok = inverse_vocab[top_token]
                top_logprobs_list.append(
                    TopLogprob(
                        token=top_detok,
                        bytes=[],
                        logprob=top_logprob,
                    )
                )
        else:
            top_logprobs_list = []

        logprobs_list.append(
            ChatCompletionTokenLogprob(
                token=detok,
                bytes=[],
                logprob=logprob,
                top_logprobs=top_logprobs_list,
            )
        )

    logprobs_obj = ChoiceLogprobs(content=logprobs_list)
    return logprobs_obj


def get_stop_strings(request: CompletionsRequest | ChatCompletionRequest | CartridgeCompletionsRequest | CartridgeChatCompletionRequest) -> list[str]:
    if request.stop is None:
        stop = []
    elif isinstance(request.stop, str):
        stop = [request.stop]
    else:
        stop = request.stop

    return stop


def decode_completion(
    state: ServerState, request: TokasaurusRequest, output: RequestOutput
):
    eos_token_ids = get_eos_token_ids(state.generation_config)

    to_decode_list = []
    for seq_out in output.sequence_outputs:
        completion_ids = seq_out.completion_ids

        trimmed_completion_ids = completion_ids
        if not request.ignore_eos:
            for eos in eos_token_ids:
                try:
                    pos = trimmed_completion_ids.index(eos)
                    trimmed_completion_ids = trimmed_completion_ids[:pos]
                except ValueError:
                    pass

        to_decode_list.append(trimmed_completion_ids)

    # TODO should we be skipping special tokens here?
    tokenizer: Tokenizer = state.get_inner_tokenizer()
    decoded_list: list[str] = tokenizer.decode_batch(
        to_decode_list,
        skip_special_tokens=True,
    )

    trimmed_list = []

    for decoded in decoded_list:
        trimmed_text = decoded

        # strip decoded string up to (but not including) the first stop token,
        # (this is to be consistent with vLLM and sglang)
        earliest_stop_pos = None
        for stop in request.stop:
            if (pos := trimmed_text.find(stop)) != -1:
                if earliest_stop_pos is None or pos < earliest_stop_pos:
                    earliest_stop_pos = pos

        if earliest_stop_pos is not None:
            trimmed_text = trimmed_text[:earliest_stop_pos]

        trimmed_list.append(trimmed_text)

    return trimmed_list


def validate_chat_completion_request(
    config: ServerConfig, request: ChatCompletionRequest | CartridgeChatCompletionRequest
):
    if request.logprobs and not config.enable_chosen_logprobs:
        raise HTTPException(
            status_code=400,
            detail="logprobs is True but engine was configured with enable_chosen_logprobs=False",
        )

    if (top_logprobs := request.top_logprobs) is not None:
        if not request.logprobs:
            raise HTTPException(
                status_code=400,
                detail="logprobs must be True if top_logprobs is set",
            )

        if top_logprobs < 0:
            raise HTTPException(
                status_code=400,
                detail="top_logprobs must be non-negative",
            )

        if top_logprobs > 0 and (
            config.max_topk_logprobs is None or top_logprobs > config.max_topk_logprobs
        ):
            raise HTTPException(
                status_code=400,
                detail=f"top_logprobs={top_logprobs} but engine was configured with max_topk_logprobs={config.max_topk_logprobs}",
            )


def validate_completions_request(config: ServerConfig, request: CompletionsRequest | CartridgeCompletionsRequest):
    if request.echo not in [False, None]:
        raise HTTPException(
            status_code=400,
            detail="echo not supported",
        )

    if request.best_of not in [None, 1]:
        raise HTTPException(
            status_code=400,
            detail="best_of not supported",
        )

    if request.suffix is not None:
        raise HTTPException(
            status_code=400,
            detail="suffix not supported",
        )

    if (logprobs := request.logprobs) is not None:
        if not config.enable_chosen_logprobs:
            raise HTTPException(
                status_code=400,
                detail="logprobs is set but engine was configured with enable_chosen_logprobs=False",
            )

        if logprobs < 0:
            raise HTTPException(
                status_code=400,
                detail="logprobs must be non-negative",
            )

        if logprobs > 0 and (
            config.max_topk_logprobs is None or logprobs > config.max_topk_logprobs
        ):
            raise HTTPException(
                status_code=400,
                detail=f"logprobs={logprobs} but engine was configured with max_topk_logprobs={config.max_topk_logprobs}",
            )


def validate_args(config: ServerConfig, request: ChatCompletionRequest | CompletionsRequest | CartridgeCompletionsRequest | CartridgeChatCompletionRequest):

    if request.top_p not in [None, 1.0]:
        raise HTTPException(
            status_code=400,
            detail="top_p not supported",
        )

    if request.frequency_penalty not in [None, 0.0]:
        raise HTTPException(
            status_code=400,
            detail="frequency_penalty not supported",
        )

    if request.logit_bias is not None:
        raise HTTPException(
            status_code=400,
            detail="logit_bias not supported",
        )

    if request.presence_penalty not in [None, 0.0]:
        raise HTTPException(
            status_code=400,
            detail="presence_penalty not supported",
        )

    match request:
        case ChatCompletionRequest() | CartridgeChatCompletionRequest():
            raw_max_tokens = request.max_tokens
            raw_max_completion_tokens = request.max_completion_tokens

            exactly_one_is_set = (raw_max_tokens is None) ^ (
                raw_max_completion_tokens is None
            )
            if not exactly_one_is_set and config.max_completion_tokens is not None:
                # TODO(SE): This is a hack to allow toka to work with raycast, which 
                # doesn't allow us to add a max_tokens field to the request. 
                request.max_completion_tokens = config.max_completion_tokens
                raw_max_tokens = config.max_completion_tokens   
            elif not exactly_one_is_set:
                raise HTTPException(
                    status_code=400,
                    detail="Exactly one of max_tokens or max_completion_tokens must be set. If you want to use a default max_completion_tokens, set it in the server config.",
                )

            max_tokens = raw_max_tokens or raw_max_completion_tokens
        case CompletionsRequest() | CartridgeCompletionsRequest():
            max_tokens = request.max_tokens

            if max_tokens is None:
                raise HTTPException(
                    status_code=400,
                    detail="max_tokens set is required",
                )

    if max_tokens <= 0:
        raise HTTPException(
            status_code=400,
            detail="max tokens must be greater than 0",
        )

    match request:
        case ChatCompletionRequest():
            validate_chat_completion_request(config, request)
        case CompletionsRequest():
            validate_completions_request(config, request)

def process_request(
    state: ServerState, request: ChatCompletionRequest | CompletionsRequest | CartridgeCompletionsRequest | CartridgeChatCompletionRequest
):
    try:
        validate_args(state.config, request)
    except HTTPException as e:
        print(f"HTTPException: {e}")
        print(f"request: {e.detail}")
        raise e

    if (n := request.n) is None:
        n = 1

    if (temp := request.temperature) is None:
        temp = 0.0

    if (top_p := request.top_p) is None:
        top_p = 1.0

    sampling_params = SamplingParams(
        temperature=temp,
        top_p=top_p,
    )

    match request:
        case ChatCompletionRequest():
            # Convert messages to plain dictionaries to avoid ValidatorIterator issues
            messages = []
            cartridges_in_system_prompt = []
            for msg in request.messages:
                converted_msg = {"role": msg["role"]}
                
                # Handle content field which might be a ValidatorIterator
                content = msg.get("content")
                if hasattr(content, '__iter__') and not isinstance(content, str) and content is not None:
                    # If content is an iterable (like ValidatorIterator), convert to list
                    try:
                        converted_msg["content"] = list(content)
                    except Exception:
                        # Fallback to string representation if conversion fails
                        converted_msg["content"] = str(content)
                else:
                    converted_msg["content"] = content
                    
                # Copy other fields that might be present
                for key in ["name", "tool_calls", "function_call", "refusal"]:
                    if key in msg:
                        converted_msg[key] = msg[key]
                if not isinstance(converted_msg["content"], str):
                    assert isinstance(converted_msg["content"], list)
                    if len(converted_msg["content"]) != 1:
                        print(converted_msg["content"])
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid content {converted_msg['content']} contains multiple parts. Tokasaurus only supports one part per message.",
                        )
                    if converted_msg["content"][0]["type"] != "text":
                        print(converted_msg["content"])
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid content {converted_msg['content']} contains non-text content. Tokasaurus only supports text content.",
                        )
                    converted_msg["content"] = converted_msg["content"][0]["text"]
                
                if converted_msg["role"] == "system":
                    # search for the following pattern:
                    # --- begin pattern ---
                    # <cartridge> 
                    # {"id": "...", "source": "...", "force_redownload": "..."}
                    # </cartridge> in the content
                    # --- end pattern ---
                    # if found, parse it using the Cartridge BaseModel and add it to the cartridges list
                    import re
                    import json
                    from tokasaurus.server.types import Cartridge
                    
                    content = converted_msg["content"]
                    if isinstance(content, str):
                        # Find all cartridge blocks in the system message
                        cartridge_pattern = r'<cartridge>\s*(\{.*?\})\s*</cartridge>'
                        matches = re.findall(cartridge_pattern, content, re.DOTALL)
                        
                        for match in matches:
                            try:
                                # Parse the JSON configuration
                                cartridge_config = json.loads(match.strip())
                                # Validate using the Cartridge model
                                cartridge = Cartridge.model_validate(cartridge_config)
                                cartridges_in_system_prompt.append(cartridge)
                            except (json.JSONDecodeError, ValueError) as e:
                                # Skip invalid cartridge configs but don't fail the request
                                print(f"Warning: Invalid cartridge config in system prompt: {e}")
                                continue
                        
                        # Remove all cartridge blocks from the content
                        if matches:
                            cleaned_content = re.sub(cartridge_pattern, '', content, flags=re.DOTALL)
                            # Clean up any extra whitespace left behind
                            cleaned_content = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_content.strip())
                            converted_msg["content"] = cleaned_content


                messages.append(converted_msg)
            

            ends_with_user = messages[-1]["role"] == "user"
            apply_chat_template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": ends_with_user,
                "continue_final_message": not ends_with_user,
            }

            if (overrides := request.apply_chat_template_overrides) is not None:
                apply_chat_template_kwargs.update(overrides)
            
            if len(cartridges_in_system_prompt) > 0:                # Create a new CartridgeChatCompletionRequest with the same data
                request_dict = request.model_dump()
                request_dict['cartridges'] = cartridges_in_system_prompt
                cartridge_request = CartridgeChatCompletionRequest(**request_dict)
                # Replace the original request with the cartridge request
                request = cartridge_request
            
            if isinstance(request, CartridgeChatCompletionRequest):
                cartridges = request.cartridges
                if "llama" in state.tokenizer.name_or_path:
                    apply_chat_template_kwargs["chat_template"] = LLAMA_CARTRIDGE_TEMPLATE
                    
            else:
                cartridges = None

            prompt = state.tokenizer.apply_chat_template(
                messages, **apply_chat_template_kwargs
            )
            input_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            top_logprobs = request.top_logprobs
            max_tokens = request.max_completion_tokens or request.max_tokens
        case CompletionsRequest():
            if isinstance(request.prompt, str):
                input_ids = state.tokenizer(request.prompt)["input_ids"]
            elif is_ids_list(request.prompt):
                input_ids = request.prompt
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid type for prompt",
                )
            if isinstance(request, CartridgeCompletionsRequest):
                cartridges = request.cartridges
            else:
                cartridges = None
            top_logprobs = request.logprobs
            max_tokens = request.max_tokens

    # we know this from validation
    assert max_tokens is not None

    rid = str(uuid4())

    req = TokasaurusRequest(
        id=rid,
        input_ids=input_ids,
        max_num_tokens=max_tokens,
        sampling_params=sampling_params,
        stop=get_stop_strings(request),
        n=n,
        ignore_eos=request.ignore_eos,
        cartridges=cartridges,
        topk_logprobs=top_logprobs,
    )

    validate_length(state, req)

    return req


def encode_array(array: np.ndarray):
    return base64.b64encode(array.tobytes()).decode("ascii")


def make_completions_fingerprint(
    output: RequestOutput,
    add_logprobs: bool,
    topk: int | None,
):
    """
    Sneaky way to send extra data back to the client while adhering to the API spec.
    """
    obj = {
        "completion_ids": [o.completion_ids for o in output.sequence_outputs],
    }

    # the default openai logprobs format involves many nested small objects which
    # can take a lot of time to serialize/deserialize and inflate the response size.
    # here we send the logprobs as base64-encoded numpy arrays instead.

    packed_chosen_logprobs = None
    packed_topk_indices = None
    packed_topk_logprobs = None

    if add_logprobs:
        packed_chosen_logprobs = [
            encode_array(np.array(o.logprobs, dtype=np.float32))
            for o in output.sequence_outputs
        ]

        if topk is not None and topk > 0:
            packed_topk_indices = []
            packed_topk_logprobs = []

            for seq_out in output.sequence_outputs:
                stack_topk_ids = np.array(seq_out.topk_ids, dtype=np.int32)[:, :topk]
                stack_topk_logprobs = np.array(seq_out.topk_logprobs, dtype=np.float32)[
                    :, :topk
                ]

                packed_topk_indices.append(encode_array(stack_topk_ids))
                packed_topk_logprobs.append(encode_array(stack_topk_logprobs))

    obj["packed_chosen_logprobs"] = packed_chosen_logprobs
    obj["packed_topk_indices"] = packed_topk_indices
    obj["packed_topk_logprobs"] = packed_topk_logprobs

    out = json.dumps(obj)

    return out


def process_chat_completions_output(
    state: ServerState,
    crequest: ChatCompletionRequest,
    request: TokasaurusRequest,
    output: RequestOutput,
):
    completions = decode_completion(state, request, output)

    choices = []
    for i in range(request.n):
        seq_out = output.sequence_outputs[i]
        new_message = ChatCompletionMessage(
            role="assistant",
            content=completions[i],
        )

        if crequest.logprobs and not crequest.logprobs_in_fingerprint:
            logprobs = make_chat_logprobs(
                crequest=crequest,
                seq_out=seq_out,
                inverse_vocab=state.inverse_vocab,
            )
        else:
            # if None or False
            logprobs = None

        choice = Choice(
            index=i,
            message=new_message,
            logprobs=logprobs,
            finish_reason=seq_out.finish_reason,
        )
        choices.append(choice)

    return ChatCompletion(
        id=request.id,
        model=crequest.model,
        usage=make_usage_info(request, output),
        choices=choices,
        created=nowstamp(),
        object="chat.completion",
        system_fingerprint=make_completions_fingerprint(
            output,
            add_logprobs=crequest.logprobs_in_fingerprint,
            topk=crequest.top_logprobs,
        ),
    )


def process_cartridge_chat_completions_output(
    state: ServerState,
    crequest: CartridgeChatCompletionRequest,
    request: TokasaurusRequest,
    output: RequestOutput,
):
    # Check for errors first
    if output.error_message is not None:
        raise HTTPException(
            status_code=400,
            detail=output.error_message,
        )
    
    completions = decode_completion(state, request, output)

    choices = []
    for i in range(request.n):
        seq_out = output.sequence_outputs[i]
        new_message = ChatCompletionMessage(
            role="assistant",
            content=completions[i],
        )

        if crequest.logprobs and not crequest.logprobs_in_fingerprint:
            logprobs = make_chat_logprobs(
                crequest=crequest,
                seq_out=seq_out,
                inverse_vocab=state.inverse_vocab,
            )
        else:
            # if None or False
            logprobs = None

        choice = Choice(
            index=i,
            message=new_message,
            logprobs=logprobs,
            finish_reason=seq_out.finish_reason,
        )
        choices.append(choice)

    return dict(
        id=request.id,
        model=crequest.model,
        usage=make_usage_info(request, output),
        choices=choices,
        created=nowstamp(),
        object="chat.completion",
        system_fingerprint=make_completions_fingerprint(
            output,
            add_logprobs=crequest.logprobs_in_fingerprint,
            topk=crequest.top_logprobs,
        ),
    )


def process_completions_output(
    state: ServerState,
    crequest: CompletionsRequest,
    request: TokasaurusRequest,
    output: RequestOutput,
):
    # Check for errors first
    if output.error_message is not None:
        raise HTTPException(
            status_code=400,
            detail=output.error_message,
        )
    
    completions = decode_completion(state, request, output)

    choices = []
    for i in range(request.n):
        seq_out = output.sequence_outputs[i]

        if crequest.logprobs and not crequest.logprobs_in_fingerprint:
            logprobs = make_completion_logprobs(
                crequest=crequest,
                seq_out=seq_out,
                inverse_vocab=state.inverse_vocab,
            )
        else:
            logprobs = None

        choice = CompletionChoice(
            index=i,
            text=completions[i],
            logprobs=logprobs,
            finish_reason=seq_out.finish_reason,
        )
        choices.append(choice)

    return Completion(
        id=request.id,
        model=crequest.model,
        usage=make_usage_info(request, output),
        choices=choices,
        created=nowstamp(),
        object="text_completion",
        system_fingerprint=make_completions_fingerprint(
            output,
            add_logprobs=crequest.logprobs_in_fingerprint,
            topk=crequest.logprobs,
        ),
    )


def parse_body(self: BatchFileLine):
    match self.url:
        case "/v1/completions":
            return CompletionsRequest.model_validate(self.body)
        case "/v1/chat/completions":
            return ChatCompletionRequest.model_validate(self.body)
        case _:
            raise HTTPException(
                status_code=400, detail=f"Unsupported endpoint: {self.url}"
            )


def make_batch_status(batch: SubmittedBatch):
    num_completed = 0
    for item in batch.items:
        if item.submitted_req.request_output is not None:
            num_completed += 1

    if batch.output_file is not None:
        output_file_id = batch.output_file.details.id
    else:
        output_file_id = None

    # Create the batch object
    status = Batch(
        id=batch.id,
        endpoint=batch.creation_request.endpoint,
        input_file_id=batch.creation_request.input_file_id,
        output_file_id=output_file_id,
        completion_window=batch.creation_request.completion_window,
        status="completed" if batch.output_file is not None else "in_progress",
        created_at=batch.created_at,
        request_counts=BatchRequestCounts(
            completed=num_completed,
            failed=0,
            total=len(batch.items),
        ),
        metadata=batch.creation_request.metadata,
        object="batch",
    )

    return status


async def generate_output(
    state: ServerState, request: CompletionsRequest | ChatCompletionRequest | CartridgeCompletionsRequest | CartridgeChatCompletionRequest
):
    req = process_request(state, request)
    submitted = submit_request(state, req)


    try:
        await submitted.event.wait()
    except asyncio.CancelledError:
        cancel_request(state, submitted)
        raise

    out = submitted.request_output
    assert out is not None

    return req, out
