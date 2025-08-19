import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Optional, Union

from openai.types.chat import ChatCompletionMessageParam
from openai.types.file_object import FileObject
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from tokasaurus.manager.types import SequenceOutput


def nowstamp():
    return int(datetime.now().timestamp())


class StreamOptions(BaseModel):
    include_usage: Optional[bool] = False


class CompletionsRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    model: str
    prompt: Union[list[int], list[list[int]], str, list[str]]
    best_of: Optional[int] = None
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[dict[str, float]] = None
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = 16
    n: int = 1
    presence_penalty: Optional[float] = 0.0
    seed: Optional[int] = None
    stop: Optional[Union[str, list[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    suffix: Optional[str] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    user: Optional[str] = None
    metadata: Optional[dict] = None

    # pack the logprobs into the fingerprint in a more space-efficient way
    logprobs_in_fingerprint: bool = False

    # extra fields to get sglang benchmarking script to work
    ignore_eos: bool = False

    class Config:
        extra = "forbid"


class JsonSchemaResponseFormat(BaseModel):
    name: str
    description: Optional[str] = None
    # use alias to workaround pydantic conflict
    schema_: Optional[dict[str, object]] = Field(alias="schema", default=None)
    strict: Optional[bool] = False


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[JsonSchemaResponseFormat] = None


class ChatCompletionRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: list[ChatCompletionMessageParam]
    model: str
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[dict[str, float]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, list[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    user: Optional[str] = None
    metadata: Optional[dict] = None

    # extra fields ---

    # needed for sglang benchmarking script
    ignore_eos: bool = False

    # pack the logprobs into the fingerprint in a more space-efficient way
    logprobs_in_fingerprint: bool = False

    # extra chat template args, e.g. to pass enable_thinking for Qwen3 models: https://huggingface.co/Qwen/Qwen3-32B
    apply_chat_template_overrides: Optional[dict[str, object]] = None

    class Config:
        extra = "forbid"


class BatchCreationRequest(BaseModel):
    """Request model for creating a batch"""

    input_file_id: str = Field(
        description="The ID of an uploaded file that contains requests for the new batch"
    )
    endpoint: str = Field(
        description="The endpoint to be used for all requests in the batch"
    )
    completion_window: str = Field(
        description="The time frame within which the batch should be processed"
    )
    metadata: Optional[dict[str, str]] = Field(default=None)


class SynchronousBatchCompletionsRequest(BaseModel):
    """Request model for synchronous batch completions"""

    requests: list[ChatCompletionRequest] = Field(
        description="List of chat completion requests to process"
    )


@dataclass
class RequestOutput:
    id: str
    sequence_outputs: list["SequenceOutput"] = field(default_factory=list)


@dataclass
class SamplingParams:
    temperature: float
    top_p: float


@dataclass
class TokasaurusRequest:
    id: str
    input_ids: list[int]
    max_num_tokens: int
    sampling_params: SamplingParams
    stop: list[str]
    n: int
    ignore_eos: bool
    topk_logprobs: int | None = None  # Number of top tokens to return log probs for
    created_timestamp: float = field(default_factory=time.time)


@dataclass
class SubmittedRequest:
    request: TokasaurusRequest
    engine_index: int

    event: asyncio.Event = field(default_factory=asyncio.Event)
    request_output: RequestOutput | None = None


class BatchFileLine(BaseModel):
    custom_id: str
    method: Literal["POST"]
    url: Literal["/v1/completions", "/v1/chat/completions"]
    body: dict


@dataclass
class FileEntry:
    content: bytes
    details: FileObject


@dataclass
class SubmittedBatchItem:
    line: BatchFileLine
    user_req: CompletionsRequest | ChatCompletionRequest
    submitted_req: SubmittedRequest


@dataclass
class SubmittedBatch:
    id: str
    creation_request: BatchCreationRequest
    items: list[SubmittedBatchItem]
    task: asyncio.Task
    created_at: int = field(default_factory=nowstamp)
    output_file: FileEntry | None = None


@dataclass
class RequestError:
    error: str


@dataclass
class CancelledRequest:
    req_id: str


CommandsFromServer = TokasaurusRequest | CancelledRequest
