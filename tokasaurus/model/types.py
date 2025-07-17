import math
from collections import deque
from dataclasses import dataclass, field, fields, is_dataclass, replace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)
from loguru import logger
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh

from tokasaurus.common_types import ServerConfig, TimedBarrier

KV_Cache = tuple[Tensor, Tensor]
DeviceType = torch.device | str


@dataclass
class WrapperCollection:
    prefill_wrapper: BatchPrefillWithPagedKVCacheWrapper | None = None
    hydragen_wrapper: BatchPrefillWithPagedKVCacheWrapper | None = None
    decode_wrapper: BatchDecodeWithPagedKVCacheWrapper | None = None


def make_ragged_tensor(data: list[list[int]]):
    flattened = []
    indptr = [0]

    for seq in data:
        flattened.extend(seq)
        indptr.append(indptr[-1] + len(seq))

    return flattened, indptr


@dataclass
class PageInformation:
    # shape [batch_size + 1], qo_indptr[i]-qo_indptr[i-1] = range of indices in query_states for the ith sequence
    qo_indptr: Tensor | None

    # shape [batch_size + 1], kv_indptr[i]-kv_indptr[i-1] = range of indices in kv_indices for the ith sequence
    kv_indptr: Tensor

    # shape [kv_indptr[-1]], indices in page table
    kv_indices: Tensor

    # shape [batch_size], kv_last_page_len[i] = length of the last page of the ith sequence (including after current step)
    kv_last_page_len: Tensor

    num_seqs: int
    num_tokens: int

    def to(self, device: DeviceType, non_blocking: bool = False):
        return replace(
            self,
            qo_indptr=self.qo_indptr.to(device, non_blocking=non_blocking)
            if self.qo_indptr is not None
            else None,
            kv_indptr=self.kv_indptr.to(device, non_blocking=non_blocking),
            kv_indices=self.kv_indices.to(device, non_blocking=non_blocking),
            kv_last_page_len=self.kv_last_page_len.to(
                device, non_blocking=non_blocking
            ),
        )

    def pad_for_cudagraph(self, num_seqs: int):
        assert self.num_seqs <= num_seqs

        if self.num_seqs == num_seqs:
            return

        assert self.qo_indptr is None, "cudagraphs are for decode only"

        num_to_pad = num_seqs - self.num_seqs

        kv_indices = F.pad(self.kv_indices, (0, num_to_pad), value=0)
        kv_indptr = torch.cat(
            [
                self.kv_indptr,
                self.kv_indptr[-1]
                + torch.arange(
                    1,
                    num_to_pad + 1,
                    device=self.kv_indptr.device,
                    dtype=self.kv_indptr.dtype,
                ),
            ]
        )
        kv_last_page_len = F.pad(self.kv_last_page_len, (0, num_to_pad), value=1)

        self.kv_indptr = kv_indptr
        self.kv_indices = kv_indices
        self.kv_last_page_len = kv_last_page_len

    @classmethod
    def new_empty(cls):
        return cls(
            qo_indptr=torch.zeros([], dtype=torch.int32),
            kv_indptr=torch.zeros([], dtype=torch.int32),
            kv_indices=torch.zeros([], dtype=torch.int32),
            kv_last_page_len=torch.zeros([], dtype=torch.int32),
            num_seqs=0,
            num_tokens=0,
        )


@dataclass
class PageInformationBuilder:
    qo_indptr: list[int] = field(default_factory=lambda: [0])
    kv_indptr: list[int] = field(default_factory=lambda: [0])
    kv_indices: list[int] = field(default_factory=list)
    kv_last_page_len: list[int] = field(default_factory=list)

    num_seqs: int = 0
    num_tokens: int = 0

    def add_sequence(
        self,
        kv_indices: list[int],
        kv_seq_len: int,
        num_qtokens: int,
        page_size: int,
        starting_block: int = 0,
    ):
        self.qo_indptr.append(self.qo_indptr[-1] + num_qtokens)

        end_block = math.ceil(kv_seq_len / page_size)

        sliced_kv_indices = kv_indices[starting_block:end_block]
        assert len(sliced_kv_indices) > 0

        self.kv_indices.extend(sliced_kv_indices)
        self.kv_indptr.append(self.kv_indptr[-1] + len(sliced_kv_indices))

        last_page_len = kv_seq_len % page_size
        if last_page_len == 0:
            last_page_len = page_size

        self.kv_last_page_len.append(last_page_len)

        self.num_seqs += 1
        self.num_tokens += num_qtokens

    def build(self, skip_qo_indptr: bool = False):
        return PageInformation(
            qo_indptr=None
            if skip_qo_indptr
            else torch.tensor(self.qo_indptr, dtype=torch.int32),
            kv_indptr=torch.tensor(self.kv_indptr, dtype=torch.int32),
            kv_indices=torch.tensor(self.kv_indices, dtype=torch.int32),
            kv_last_page_len=torch.tensor(self.kv_last_page_len, dtype=torch.int32),
            num_seqs=self.num_seqs,
            num_tokens=self.num_tokens,
        )


@dataclass
class AttentionInfoBuilder:
    """
    A version of AttentionInfo that stores the data as Python lists
    rather than tensors, making it more efficient for passing over
    multiprocessing queues.
    """

    page_size: int

    append_kv_token_indices: list[int]
    prefill_builder: PageInformationBuilder
    decode_builder: PageInformationBuilder
    hydragen_builder: PageInformationBuilder | None = None

    num_padding: int = 0

    def build(self) -> "AttentionInfo":
        """Convert builders to tensors"""
        return AttentionInfo(
            page_size=self.page_size,
            append_kv_token_indices=torch.tensor(
                self.append_kv_token_indices, dtype=torch.long
            ),
            prefill_info=self.prefill_builder.build(),
            decode_info=self.decode_builder.build(skip_qo_indptr=True),
            hydragen_info=self.hydragen_builder.build()
            if self.hydragen_builder
            else None,
            num_padding=self.num_padding,
        )


@dataclass
class AttentionInfo:
    """
    Convention: tokens are in order of prefill, hydragen, decode.
    Hydragen = the shared prefix part of hydragen (only one layer
    deep for now). Hydragen calls a prefill kernel but with no
    causal mask.
    """

    page_size: int

    append_kv_token_indices: Tensor
    prefill_info: PageInformation
    decode_info: PageInformation
    hydragen_info: PageInformation | None = None

    # to make batches that are a tp-sized multiple for tensor parallelism
    num_padding: int = 0

    def split_q(self, q: Tensor):
        start = 0
        prefill_q = q[start : start + self.prefill_info.num_tokens]
        start += self.prefill_info.num_tokens

        if self.hydragen_info is not None:
            hydragen_q = q[start : start + self.hydragen_info.num_tokens]
            # remember: decode includes hydragen tokens, so don't increment start

        else:
            hydragen_q = q[:0]

        decode_q = q[start:]

        assert len(decode_q) == self.decode_info.num_tokens, (
            len(decode_q),
            self.decode_info.num_tokens,
        )

        return prefill_q, hydragen_q, decode_q

    def to(self, device: DeviceType, non_blocking: bool = False):
        return replace(
            self,
            append_kv_token_indices=self.append_kv_token_indices.to(
                device, non_blocking=non_blocking
            ),
            prefill_info=self.prefill_info.to(device, non_blocking=non_blocking),
            decode_info=self.decode_info.to(device, non_blocking=non_blocking),
            hydragen_info=self.hydragen_info.to(device, non_blocking=non_blocking)
            if self.hydragen_info
            else None,
        )


@dataclass
class PrefillInfo:
    input_ids: list[int]
    position_ids: list[int]
    kv_indices: list[int]
    kv_last_page_len: int

    def sequence_length(self, page_size: int):
        return (len(self.kv_indices) - 1) * page_size + self.kv_last_page_len


@dataclass
class BatchSamplingParams:
    temperature: Tensor | None = None
    top_p: Tensor | None = None
    greedy_mask: Tensor | None = None

    def to(self, device: DeviceType, non_blocking: bool = False):
        if (temperature := self.temperature) is not None:
            temperature = temperature.to(device, non_blocking=non_blocking)

        if (top_p := self.top_p) is not None:
            top_p = top_p.to(device, non_blocking=non_blocking)

        if (greedy_mask := self.greedy_mask) is not None:
            greedy_mask = greedy_mask.to(device, non_blocking=non_blocking)

        return BatchSamplingParams(
            temperature=temperature,
            top_p=top_p,
            greedy_mask=greedy_mask,
        )

    def copy_(self, src: "BatchSamplingParams"):
        if self.temperature is not None:
            assert src.temperature is not None
            self.temperature.copy_(src.temperature)

        if self.top_p is not None:
            assert src.top_p is not None
            self.top_p.copy_(src.top_p)

        if self.greedy_mask is not None:
            assert src.greedy_mask is not None
            self.greedy_mask.copy_(src.greedy_mask)


@dataclass
class BatchSamplingParamsBuilder:
    temperature: list[float] = field(default_factory=list)
    top_p: list[float] = field(default_factory=list)
    greedy_mask: list[bool] = field(default_factory=list)

    def add_sequence(self, temperature: float, top_p: float):
        greedy = temperature == 0.0
        self.top_p.append(top_p)
        self.greedy_mask.append(greedy)

        if greedy:
            # we don't want to scale by a zero temp since it
            # causes runtime errors in our implementation.
            self.temperature.append(1.0)
        else:
            self.temperature.append(temperature)

    def build(self):
        temperature = torch.tensor(self.temperature, dtype=torch.float32)
        greedy_mask = torch.tensor(self.greedy_mask, dtype=torch.bool)

        assert all([top_p == 1.0 for top_p in self.top_p])
        top_p = None

        # if all(temperature == 1.0 for temperature in self.temperature):
        #     temperature = None
        # else:
        #     temperature = torch.tensor(self.temperature, dtype=torch.float32)

        # if all(top_p == 1.0 for top_p in self.top_p):
        #     top_p = None
        # else:
        #     top_p = torch.tensor(self.top_p, dtype=torch.float32)

        # if all(not greedy for greedy in self.greedy_mask):
        #     greedy_mask = None
        # else:
        #     greedy_mask = torch.tensor(self.greedy_mask, dtype=torch.bool)

        return BatchSamplingParams(
            temperature=temperature,
            greedy_mask=greedy_mask,
            top_p=top_p,
        )


@dataclass
class ModelInput:
    # Use attention_info_builder instead of a pre-built attention_info
    attention_info_builder: AttentionInfoBuilder

    prefill_input_ids: list[int]
    sampling_builder: "BatchSamplingParamsBuilder"

    # one batch id per token (for a prefill sequence, it's batch id is repeated for all tokens)
    batch_indices: list[int]
    lm_head_indices: list[int]
    position_ids: list[int]
    schedule_id: str
    microbatch_index: int | None = None
    microbatch_total: int | None = None
    skip_pipeline_communication: bool = False

    def num_prefill_tokens(self):
        return len(self.prefill_input_ids)

    def num_decode_tokens(self):
        return len(self.batch_indices) - self.num_prefill_tokens()

    def num_lm_head_tokens(self):
        return len(self.lm_head_indices)

    def lm_head_batch_indices(self):
        return [self.batch_indices[x] for x in self.lm_head_indices]

    def decoding_batch_indices(self):
        return self.batch_indices[len(self.prefill_input_ids) :]

    def decode_start_pos(self):
        return len(self.prefill_input_ids)

    def build_attention_info(self) -> AttentionInfo:
        """Build the attention info tensors"""
        return self.attention_info_builder.build()

    def build_sampling_params(self) -> BatchSamplingParams:
        """Build the sampling params tensors"""
        return self.sampling_builder.build()


def move_dataclass_tensors(obj, device: torch.device, non_blocking: bool = False):
    for f in fields(obj):
        attr = getattr(obj, f.name)
        if isinstance(attr, Tensor):
            setattr(obj, f.name, attr.to(device, non_blocking=non_blocking))
        elif is_dataclass(attr):
            move_dataclass_tensors(attr, device, non_blocking=non_blocking)


class NoMoreInputs:
    pass


CommandFromManager = ModelInput | NoMoreInputs


@dataclass
class ModelOutputTensors:
    output_ids: Tensor

    # logprobs of the tokens that were chosen
    chosen_logprobs: Tensor | None = None
    topk_indices: Tensor | None = None
    topk_logprobs: Tensor | None = None

    def to(self, device: DeviceType, non_blocking: bool = False):
        return replace(
            self,
            output_ids=self.output_ids.to(device, non_blocking=non_blocking),
            chosen_logprobs=self.chosen_logprobs.to(device, non_blocking=non_blocking)
            if self.chosen_logprobs is not None
            else None,
            topk_indices=self.topk_indices.to(device, non_blocking=non_blocking)
            if self.topk_indices is not None
            else None,
            topk_logprobs=self.topk_logprobs.to(device, non_blocking=non_blocking)
            if self.topk_logprobs is not None
            else None,
        )


@dataclass
class ModelOutput:
    schedule_id: str
    tensors: ModelOutputTensors
    microbatch_index: int | None = None


@dataclass
class BatchState:
    position_ids: Tensor
    attention_info: AttentionInfo
    sampling_params: BatchSamplingParams | None = None
    input_ids: Tensor | None = None
    prefill_input_ids: Tensor | None = None
    lm_head_indices: Tensor | None = None
    hidden_states: Tensor | None = None
    position_embeddings: tuple[Tensor, Tensor] | None = None
    num_total_padding: int = 0
    num_lm_head_padding: int = 0
    # the unpadded/unsliced lm_head_indices, for use in updating
    # the most-recently-generated tokens map.
    raw_lm_head_indices: Tensor | None = None
    outputs: ModelOutputTensors | None = None


@dataclass
class BasicWorkerState:
    config: ServerConfig
    batch_index_to_last_token: Tensor
    input_q: mp.Queue
    q_model_to_manager: mp.Queue
    device: DeviceType
    dtype: torch.dtype
    process_name: str
    tp_rank: int
    barrier: TimedBarrier

    def __post_init__(self):
        self.logger = logger.bind(process_name=self.process_name)


@dataclass
class PipelineWorkerState:
    config: ServerConfig
    input_q: mp.Queue
    q_pipe_end_to_start: mp.Queue
    q_to_manager: mp.Queue
    process_name: str
    pp_rank: int
    tp_rank: int
    dp_rank: int
    barrier: TimedBarrier
    device_mesh: DeviceMesh | None = None
    inflight_microbatches: deque[ModelInput] = field(default_factory=deque)
    finished_outputs: deque[tuple[ModelInput, ModelOutput]] = field(
        default_factory=deque
    )
    batch_id_to_last_token: Tensor | None = None

    def __post_init__(self):
        self.logger = logger.bind(process_name=self.process_name)


@dataclass
class ExtraModelConfig:
    """
    For flags that we define that aren't in the hf config.
    """

    pp_size: int = 1
    tp_size: int = 1
    pp_rank: int = 0
    tp_rank: int = 0
    tp_group: dist.ProcessGroup | None = None

    torch_compile: bool = False

    rope_scaling: dict | None = None

    enable_chosen_logprobs: bool = True
    topk_logprobs: int | None = None
