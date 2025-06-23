import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional, Union

from openai.types.chat import ChatCompletionMessageParam
from openai.types.file_object import FileObject
from pydantic import BaseModel, Field, model_validator


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
    max_tokens: Optional[int] = None  # Default will be handled by validator
    max_completion_tokens: Optional[int] = None  # New field
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

    # extra fields to get sglang benchmarking script to work
    ignore_eos: bool = False

    class Config:
        extra = "forbid"

    @model_validator(mode="before")
    @classmethod
    def _validate_max_tokens(cls, values):
        max_tokens = values.get("max_tokens")
        max_completion_tokens = values.get("max_completion_tokens")

        if max_tokens is not None and max_completion_tokens is not None:
            raise ValueError(
                "Only one of 'max_tokens' or 'max_completion_tokens' can be set."
            )

        if max_completion_tokens is not None:
            values["max_tokens"] = max_completion_tokens
        elif max_tokens is None: # Neither is set, apply default
            values["max_tokens"] = 16 # Default value

        # Ensure max_completion_tokens is not passed to the model constructor after this
        # as it's not a field it would recognize after our aliasing trick.
        # However, Pydantic v2 'before' validator means we modify the dict before
        # the main model validation, so we don't need to pop it if it's not a real field.
        # If max_completion_tokens was a real field, we'd do:
        # if "max_completion_tokens" in values:
        #     del values["max_completion_tokens"]
        # But since it's not in the model fields (we only added it to the input `values`),
        # Pydantic will ignore it if `extra = "ignore"` or forbid it if `extra = "forbid"`
        # unless it's explicitly defined.

        # Let's define max_completion_tokens as a field to make it explicit and then it will be ignored
        # by the model itself if not used further. The primary purpose is to perform this validation
        # and consolidation into max_tokens.

        return values


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
    max_tokens: Optional[int] = None # Default is None, validator will handle logic
    max_completion_tokens: Optional[int] = None # New field
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

    # extra fields to get sglang benchmarking script to work
    ignore_eos: bool = False

    class Config:
        extra = "forbid"

    @model_validator(mode="before")
    @classmethod
    def _validate_chat_max_tokens(cls, values):
        max_tokens = values.get("max_tokens")
        max_completion_tokens = values.get("max_completion_tokens")

        if max_tokens is not None and max_completion_tokens is not None:
            raise ValueError(
                "Only one of 'max_tokens' or 'max_completion_tokens' can be set for ChatCompletionRequest."
            )

        if max_completion_tokens is not None:
            values["max_tokens"] = max_completion_tokens
        elif max_tokens is not None:
            # max_tokens is already set, and max_completion_tokens is None.
            # No change needed to values["max_tokens"].
            pass
        else:
            # Both are None, so max_tokens remains None as per its field definition.
            values["max_tokens"] = None

        # As with CompletionsRequest, Pydantic handles max_completion_tokens
        # not being an actual model field after this 'before' validator
        # if it's defined in the model itself.
        return values


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


@dataclass
class RequestOutput:
    id: str
    completion_ids: list[list[int]] = field(default_factory=list)
    logprobs: list[list[float]] = field(default_factory=list)
    finish_reason: list[str] = field(default_factory=list)
    num_cached_prompt_tokens: list[int] = field(default_factory=list)

    def validate_lengths(self):
        assert (
            len(self.completion_ids)
            == len(self.logprobs)
            == len(self.finish_reason)
            == len(self.num_cached_prompt_tokens)
        )


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
