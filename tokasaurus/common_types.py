import time
from dataclasses import dataclass, field
from typing import Callable

import pydra
import torch.multiprocessing as mp
from transformers import AutoConfig, GenerationConfig

from tokasaurus.core import complete_server_startup


class TimedBarrier:
    def __init__(self, num_procs: int, message: str):
        self.barrier = mp.Barrier(num_procs)
        self.message = message
        self.start_time = time.time()

    def wait(self):
        remaining = self.barrier.wait()
        end = time.time()
        if remaining == 0:
            print(f"{self.message}: {end - self.start_time}")
            complete_server_startup()


@dataclass
class ProcessInfo:
    target: Callable
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)

    def make_process(self):
        return mp.Process(target=self.target, args=self.args, kwargs=self.kwargs)


@dataclass
class Engine:
    """
    Wraps the queues to interact with a manager
    and one or more model processes.
    """

    q_server_to_manager: mp.Queue
    q_manager_to_server: mp.Queue

    proc_dict: dict[str, ProcessInfo]

    def num_procs(self):
        return len(self.proc_dict)


class ServerConfig(pydra.Config):
    model: str
    tokenizer: str | None = None

    trust_remote_code: bool = False
    dtype: str = "bfloat16"
    rope_scaling: str | None = None

    use_hydragen: bool = False
    hydragen_min_group_size: int = 32
    hydragen_min_prefix_len: int = 256

    enable_chosen_logprobs: bool = True
    max_topk_logprobs: int | None = None

    port: int = 10210
    local_proc_name: str = "server"

    log_level: str = "INFO"
    log_procs: list[str] | None = None
    uvicorn_log_level: str = "info"

    stats_report_seconds: float = 5.0
    statsd_server_url: None | str = None
    
    # WandB configuration
    wandb_enabled: bool = False
    wandb_entity: str = "hazy-research"
    wandb_project: str = "tokasaurus"
    wandb_run_name: str | None = None

    page_size: int = 16
    kv_cache_num_tokens: int = 1024 * 128

    torch_compile: bool = False

    # the batch size at which we switch to using async TP
    async_tp_threshold: int | None = None

    max_tokens_per_forward: int = 8192
    max_seqs_per_forward: int = 1024
    prefill_round_up_multiple: int = 16

    scheduling_steps_ahead: int = 8
    stop_string_num_token_lookback: int = 5

    dp_size: int = 1
    pp_size: int = 1
    tp_size: int = 1

    # adding extra stages to hide the latency
    # of sending lm-head results from the end of the pipeline to the start,
    # as well as buffer data dependencies from sequences being rearranged
    # across microbatches (e.g. as sequences finish / new sequences start).
    pp_num_buffer_stages: int = 1

    track_early_stopping: bool = True
    early_stopping_buffer_size: int = 2048
    early_stopping_num_prediction_buckets: int = 1024
    early_stopping_initial_wait: int = 16
    early_stopping_init_mean: float | None = None
    early_stopping_init_std: float | None = None
    max_num_tokens_per_request: int | None = None

    enable_precise_onboard: bool = True
    precise_onboard_batch_size: int = 128
    greedy_prefill: bool = True

    use_spec_allocation: bool = True
    spec_allocation_std_buffer_scale: float = 0.25
    spec_allocation_target_kv_cache_utilization: float = 1.0

    use_cudagraphs: bool = True
    cudagraph_max_size: int = 128
    cudagraph_step: int = 16
    cudagraph_max_kv_indices_per_seq: int = 32768

    # for debugging only, will slow things down
    allocator_sanity_checks: bool = False
    bump_city_population_me: bool = False

    # TODO(SE): This needs to be fixed, it's an uber hack. 
    max_completion_tokens: int | None = None

    # Cartridge configuration
    cartridge_dir: str = "./cartridges"

    def uvsh(self):
        self.uvicorn_log_level = "warning"

    def kv_cache_num_blocks(self):
        assert self.kv_cache_num_tokens % self.page_size == 0
        return self.kv_cache_num_tokens // self.page_size

    def max_batch_index(self):
        # fudge factor on the total number of sequences running at any time
        return self.max_tokens_per_forward * 2

    def model_config(self):
        return AutoConfig.from_pretrained(
            self.model, trust_remote_code=self.trust_remote_code
        )

    def generation_config(self):
        return GenerationConfig.from_pretrained(
            self.model, trust_remote_code=self.trust_remote_code
        )

    def finalize(self):
        super().finalize()

        if self.use_spec_allocation:
            assert self.track_early_stopping, (
                "use_spec_allocation requires track_early_stopping"
            )
            assert self.spec_allocation_std_buffer_scale >= 0, (
                "spec_allocation_std_buffer_scale must be non-negative"
            )

        if self.tokenizer is None:
            self.tokenizer = self.model

        if self.max_num_tokens_per_request is None:
            model_config = self.model_config()
            self.max_num_tokens_per_request = min(
                model_config.max_position_embeddings, self.kv_cache_num_tokens
            )
            print(
                f"Setting max_num_tokens_per_request to {self.max_num_tokens_per_request}"
            )

        if self.use_hydragen and self.use_cudagraphs:
            assert self.cudagraph_max_size < self.hydragen_min_group_size, (
                f"For now hydragen_min_group_size ({self.hydragen_min_group_size}) must exceed cudagraph_max_size ({self.cudagraph_max_size})"
            )

    # for debugging different parts of the system
    def dmanager(self):
        self.local_proc_name = "manager"

    def dmodel(self):
        self.local_proc_name = "model_worker"

    def par(self, dp=1, pp=1, tp=1):
        self.dp_size = dp
        self.pp_size = pp
        self.tp_size = tp

    def scheduler_block_target(self):
        target_blocks = self.kv_cache_num_blocks()
        if self.use_spec_allocation:
            target_blocks = round(
                target_blocks * self.spec_allocation_target_kv_cache_utilization
            )
        return target_blocks
