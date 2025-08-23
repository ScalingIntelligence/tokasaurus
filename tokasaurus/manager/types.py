import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional
from pathlib import Path

if TYPE_CHECKING:
    from tokasaurus.manager.monitoring import WandbLogger

import numpy as np
import torch.multiprocessing as mp
import yaml
from loguru import logger
from tokenizers import Tokenizer
from transformers import AutoTokenizer, GenerationConfig

from tokasaurus.common_types import ServerConfig
from tokasaurus.manager.allocator import BatchIndexAllocator, BlockAllocator
from tokasaurus.manager.monitoring import StatsTracker
from tokasaurus.manager.stopping_predictor import EarlyStoppingTracker
from tokasaurus.server.types import RequestOutput, SamplingParams, TokasaurusRequest
from tokasaurus.utils import sanitize_cartridge_id


@dataclass
class CartridgeConfig:
    """Configuration for a cartridge containing KV cache data, parsed from wandb config.yaml."""
    cartridge_id: str
    num_tokens: int
    max_tokens_config: int  # Original max_tokens from kv_cache_initializer
    pretrained_model_name: str
    
    @classmethod
    def from_config_yaml(cls, cartridge_id: str, config_path: Path) -> "CartridgeConfig":
        """Create CartridgeConfig from wandb config.yaml file."""
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        
        # Extract max_tokens from kv_cache_initializer section
        kv_cache_init = config_data.get("kv_cache_initializer", {})
        max_tokens = kv_cache_init.get("max_tokens")
        if max_tokens is None:
            # try value field (wandb config format)
            max_tokens = kv_cache_init.get("value", {}).get("max_tokens")
        if max_tokens is None:
            raise ValueError(f"max_tokens not found in {config_path}")
        
        # Extract model information
        model_config = config_data.get("model", {})
        pretrained_model_name = model_config.get("pretrained_model_name_or_path", "meta-llama/Llama-3.2-3B-Instruct")
        
        return cls(
            cartridge_id=cartridge_id,
            num_tokens=max_tokens,
            max_tokens_config=max_tokens,
            pretrained_model_name=pretrained_model_name
        )
    
    def num_blocks_needed(self, page_size: int) -> int:
        """Calculate the number of blocks needed for this cartridge."""
        return math.ceil(self.num_tokens / page_size)


@dataclass
class SequenceOutput:
    completion_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    topk_ids: list[np.ndarray] = field(default_factory=list)
    topk_logprobs: list[np.ndarray] = field(default_factory=list)
    finish_reason: str | None = None
    num_cached_prompt_tokens: int | None = None


@dataclass
class Sequence:
    id: str
    completion_total: int
    input_ids: list[int]
    prepended_cartridge_ids: Optional[tuple[str, ...]] = None

    sampling_params: SamplingParams = field(
        default_factory=lambda: SamplingParams(temperature=0.0, top_p=1.0)
    )
    stop: list[str] = field(default_factory=list)

    # tracks how many prefill/decoding tokens have
    # been scheduled (maybe not actually ran by the model yet)
    # note that the finishing the last prompt token also generates
    # the first completion token.
    prompt_scheduled: int = 0
    completion_scheduled: int = 0

    batch_index: int | None = None
    
    # Split block indices: cartridge blocks vs token-specific blocks
    cartridge_indices: list[int] | None = None
    kv_indices: list[int] | None = None  # Now token-only blocks

    # the tokens/logprobs that have actually come back from the model
    # note that these lists will lag completion_scheduled because we
    # are asynchronously sending work to the model
    seq_output: SequenceOutput = field(default_factory=SequenceOutput)

    cancelled: bool = False

    _expected_completion_length: int | None = None
    _expected_completion_length_with_buffer: int | None = None

    request: TokasaurusRequest | None = None
    req_output: RequestOutput | None = None

    def __repr__(self):
        return f"Seq(idx={self.id}, pre={self.prompt_scheduled}/{self.prompt_total()}, dec={self.completion_scheduled}/{self.completion_total})"

    def __hash__(self):
        return hash(self.id)

    def prompt_total(self):
        return len(self.input_ids)

    def total_scheduled(self):
        return self.prompt_scheduled + self.completion_scheduled

    def prompt_to_schedule(self):
        return self.prompt_total() - self.prompt_scheduled

    def expected_completion_length(self, add_buffer: bool = False):
        if add_buffer:
            val = self._expected_completion_length_with_buffer
        else:
            val = self._expected_completion_length

        if val is None:
            return self.completion_total

        assert val <= self.completion_total, (val, self.completion_total)
        return val

    def expected_completion_to_schedule(self, add_buffer: bool = False):
        return self.expected_completion_length(add_buffer) - self.completion_scheduled

    def expected_total_length(self, add_buffer: bool = False):
        return self.prompt_total() + self.expected_completion_length(add_buffer)

    def get_num_cartridge_blocks(self) -> int:
        """Get the number of cartridge blocks for this sequence."""
        return len(self.cartridge_indices) if self.cartridge_indices else 0

    def expected_num_additional_blocks(self, page_size: int, add_buffer: bool = False):
        # NOTE: This now only considers token blocks, not cartridge blocks
        # -1 since the last generated token isn't sent through the model, so we
        # don't need to reserve space for it
        kv_tokens_needed = (
            self.prompt_total() + self.expected_completion_length(add_buffer) - 1
        )

        if self.kv_indices is not None:  # Only token blocks
            kv_tokens_needed -= page_size * len(self.kv_indices)

        return math.ceil(kv_tokens_needed / page_size)

    def expected_last_page_len(self, page_size: int, add_buffer: bool = False):
        kv_tokens_needed = (
            self.prompt_total() + self.expected_completion_length(add_buffer) - 1
        )

        last_page_len = kv_tokens_needed % page_size
        if last_page_len == 0:
            last_page_len = page_size

        return last_page_len

    def max_num_additional_blocks(self, page_size: int):
        # NOTE: This now only considers token blocks, not cartridge blocks
        kv_tokens_needed = self.prompt_total() + self.completion_total - 1
        if self.kv_indices is not None:  # Only token blocks
            kv_tokens_needed -= page_size * len(self.kv_indices)

        return math.ceil(max(0, kv_tokens_needed) / page_size)

    def most_recent_completion_ids(self, num_to_return: int):
        recently_decoded = self.seq_output.completion_ids[-num_to_return:]
        return recently_decoded

    def total_block_indices(self) -> list[int]:
        """Get all block indices (cartridge + token) for model input."""
        result = []
        if self.cartridge_indices:
            result.extend(self.cartridge_indices)
        if self.kv_indices:
            result.extend(self.kv_indices)
        return result
    def num_uncached_prompt_tokens(self):
        num_cached = self.seq_output.num_cached_prompt_tokens
        assert num_cached is not None
        return self.prompt_total() - num_cached

    def num_cached_blocks(self, page_size: int):
        num_cached = self.seq_output.num_cached_prompt_tokens
        assert num_cached is not None
        assert num_cached % page_size == 0
        return num_cached // page_size

    def cached_blocks(self, page_size: int):
        assert self.kv_indices is not None
        num_cached_blocks = self.num_cached_blocks(page_size)
        return self.kv_indices[:num_cached_blocks]

    def uncached_blocks(self, page_size: int):
        assert self.kv_indices is not None
        num_cached_blocks = self.num_cached_blocks(page_size)
        return self.kv_indices[num_cached_blocks:]


@dataclass
class SchedulingQueue:
    """
    NOTE: this keeps track of the state ACCORDING TO THE SCHEDULER - may not
    actually reflect the true state of requests. E.g., a "finished" sequence
    here may still have outstanding tokens to actually compute, and may not
    yet have had its request returned to the user.


    Also, we're using the fact that Python dicts are ordered, so that these dicts
    are insertion-order queues that we can also index by id.
    """

    decoding_seqs: dict[str, Sequence] = field(default_factory=dict)
    prefilling_seqs: dict[str, Sequence] = field(default_factory=dict)
    queued_seqs: dict[str, Sequence] = field(default_factory=dict)

    def get(self, sid: str):
        out = (
            self.decoding_seqs.get(sid)
            or self.prefilling_seqs.get(sid)
            or self.queued_seqs.get(sid)
        )
        if out is None:
            raise ValueError(f"Request {sid} not found")
        return out

    def __getitem__(self, sid: str):
        return self.get(sid)

    def get_decoding(self, sid: str):
        return self.decoding_seqs[sid]

    def get_prefilling(self, sid: str):
        return self.prefilling_seqs[sid]

    def get_queued(self, sid: str):
        return self.queued_seqs[sid]

    def add_decoding(self, seq: Sequence):
        self.decoding_seqs[seq.id] = seq

    def add_prefilling(self, seq: Sequence):
        self.prefilling_seqs[seq.id] = seq

    def add_queued(self, seq: Sequence):
        self.queued_seqs[seq.id] = seq

    def remove_decoding(self, sid: str):
        self.decoding_seqs.pop(sid)

    def remove_prefilling(self, sid: str):
        self.prefilling_seqs.pop(sid)

    def remove_queued(self, sid: str):
        self.queued_seqs.pop(sid)

    def remove(self, sid: str):
        if self.in_decoding(sid):
            self.remove_decoding(sid)
        elif self.in_prefilling(sid):
            self.remove_prefilling(sid)
        elif self.in_queued(sid):
            self.remove_queued(sid)
        else:
            raise ValueError(f"Sequence {sid} not found")

    def in_decoding(self, sid: str) -> bool:
        return sid in self.decoding_seqs

    def in_prefilling(self, sid: str) -> bool:
        return sid in self.prefilling_seqs

    def in_queued(self, sid: str) -> bool:
        return sid in self.queued_seqs

    def insert_at_head_of_queued(self, seqs: list[Sequence]):
        for d in [self.decoding_seqs, self.prefilling_seqs, self.queued_seqs]:
            for seq in seqs:
                assert seq.id not in d

        new_queued = {}
        for seq in seqs:
            new_queued[seq.id] = seq

        new_queued.update(self.queued_seqs)
        self.queued_seqs = new_queued

    def running_seqs(self):
        return list(self.decoding_seqs.values()) + list(self.prefilling_seqs.values())

    def unfinished_seqs(self):
        return self.running_seqs() + list(self.queued_seqs.values())

    def num_unfinished_seqs(self):
        return (
            len(self.decoding_seqs) + len(self.prefilling_seqs) + len(self.queued_seqs)
        )

    def num_running_seqs(self):
        return len(self.decoding_seqs) + len(self.prefilling_seqs)


@dataclass
class ScheduleDecision:
    id: str
    decoding_seqs: list[Sequence]
    prefill_seqs: list[tuple[Sequence, int]]  # (seq, prefill_length)

    def __post_init__(self):
        # construct the list of ids where we expect to get tokens
        # this order is important, it's what the model returns to us:
        # - first any prefill seqs (in order of prefill_seqs)
        # - then any decode seqs
        self.seqs_with_tokens_to_return = list[Sequence]()

        for seq, prefill_len in self.prefill_seqs:
            assert seq.prompt_to_schedule() >= prefill_len, (
                f"Request {seq.id} has {seq.prompt_to_schedule()} prompt tokens to schedule but prefill length is {prefill_len}"
            )

            # if we finish prefilling, the model will sample for us
            if seq.prompt_to_schedule() == prefill_len:
                self.seqs_with_tokens_to_return.append(seq)

        self.seqs_with_tokens_to_return.extend(self.decoding_seqs)

    def num_decoding_tokens(self):
        return len(self.decoding_seqs)

    def num_prefill_tokens(self):
        return sum(prefill_len for _, prefill_len in self.prefill_seqs)

    def batch_size(self):
        return self.num_decoding_tokens() + self.num_prefill_tokens()

    def num_seqs(self):
        return len(self.decoding_seqs) + len(self.prefill_seqs)


@dataclass
class ManagerState:
    config: ServerConfig
    block_allocator: "BlockAllocator"
    batch_index_allocator: "BatchIndexAllocator"
    q_manager_to_model: mp.Queue
    q_model_to_manager: mp.Queue
    q_server_to_manager: mp.Queue
    q_manager_to_server: mp.Queue
    process_name: str
    
    # New fields for async cartridge downloading
    q_download_requests: mp.Queue
    q_download_complete: mp.Queue
    cartridges_downloading: set[str] = field(default_factory=set)  # cartridge_ids being downloaded
    sequences_waiting_for_cartridges: dict[str, set[str]] = field(default_factory=dict)  # cartridge_id -> seq_ids
    
    scheduling_queue: SchedulingQueue = field(default_factory=SchedulingQueue)
    inflight_schedule_decisions: dict[str, ScheduleDecision] = field(
        default_factory=dict
    )
    finished_seq_ids: set[str] = field(default_factory=set)
    bumped_seq_id_to_ttl: dict[str, int] = field(default_factory=dict)
    num_inflight_batches: int = 0
    stats_tracker: StatsTracker = field(default_factory=StatsTracker)
    early_stopping_tracker: EarlyStoppingTracker | None = None
    req_id_to_seq_ids: dict[str, set[str]] = field(default_factory=dict)
    req_ids_to_cancel: set[str] = field(default_factory=set)
    sent_no_more_inputs: bool = False
    last_step_num_prefill: int = 1024 * 1024 * 1024
    
    # Cartridge management
    cartridge_id_to_config: dict[str, CartridgeConfig] = field(default_factory=dict)
    cartridge_loading_state: dict[str, bool] = field(default_factory=dict)
    cartridge_info_storage: dict[str, dict] = field(default_factory=dict)
    
    # Monitoring
    wandb_logger: "WandbLogger | None" = None

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
        self.generation_config = GenerationConfig.from_pretrained(
            self.config.model, trust_remote_code=True
        )
        self.logger = logger.bind(process_name=self.process_name)

    def get_tokenizer(self) -> Tokenizer:
        return self.tokenizer._tokenizer  # type: ignore

    def store_cartridge_info(self, cartridge_info_list: list[dict]):
        """Store cartridge info for later use during downloading."""
        for cartridge_info in cartridge_info_list:
            cartridge_id = cartridge_info["id"]
            self.cartridge_info_storage[cartridge_id] = cartridge_info

    def get_cartridge_info(self, cartridge_id: str) -> dict | None:
        """Get stored cartridge info for downloading."""
        return self.cartridge_info_storage.get(cartridge_id)

    def load_cartridge_config(self, cartridge_id: str) -> CartridgeConfig:
        """
        Load cartridge configuration from config.yaml file in cartridge directory.
        """
        if cartridge_id in self.cartridge_id_to_config:
            return self.cartridge_id_to_config[cartridge_id]
        
        # Load actual config from cartridge directory (use sanitized ID for directory path)
        sanitized_id = sanitize_cartridge_id(cartridge_id)
        cartridge_path = Path(self.config.cartridge_dir) / sanitized_id
        config_file = cartridge_path / "config.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Cartridge config not found: {config_file}")
        
        try:
            # Use the new from_config_yaml class method
            config = CartridgeConfig.from_config_yaml(cartridge_id, config_file)
            
            # Cache the config
            self.cartridge_id_to_config[cartridge_id] = config
            self.logger.info(f"Loaded cartridge config: {cartridge_id} with {config.num_tokens} tokens from {config_file}")
            return config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in cartridge config {config_file}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading cartridge config {config_file}: {e}")

    def deallocate(self, seq: Sequence):
        assert seq.kv_indices is not None or seq.cartridge_indices is not None

        # NOTE: the last decoded token is NOT sent back through the model, so we don't
        # generate a KV cache for it (and thus can't prefix cache with it).
        ids_for_update = seq.input_ids + seq.seq_output.completion_ids[:-1]
        if seq.prompt_scheduled < seq.prompt_total():
            ids_for_update = ids_for_update[: seq.prompt_scheduled]

        cartridge_indices = seq.cartridge_indices or []
        token_indices = seq.kv_indices or []
        
        self.block_allocator.free_and_update(
            seq.id, cartridge_indices, token_indices, ids_for_update, 
            prepended_cartridge_ids=seq.prepended_cartridge_ids
        )
        seq.cartridge_indices = None
        seq.kv_indices = None

        assert seq.batch_index is not None
        self.batch_index_allocator.free(seq.batch_index)
        seq.batch_index = None

    def is_cartridge_loaded(self, cartridge_id: str) -> bool:
        """Check if a cartridge is loaded."""
        return self.cartridge_loading_state.get(cartridge_id, False)
    
    def mark_cartridge_loading(self, cartridge_id: str):
        """Mark a cartridge as loading/loaded."""
        self.cartridge_loading_state[cartridge_id] = True

    def reset_cartridge_loading_state(self, cartridge_id: str):
        """Reset cartridge loading state to force reload after force redownload."""
        self.cartridge_loading_state[cartridge_id] = False
        self.logger.info(f"Reset loading state for cartridge {cartridge_id} - will be reloaded on next use")


@dataclass
class HydragenGroup:
    block_ids: list[int]
    seq_ids: set[str]
