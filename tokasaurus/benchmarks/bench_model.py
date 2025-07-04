import math
import random

import pydra
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokasaurus.common_types import ServerConfig
from tokasaurus.manager.allocator import BlockAllocator
from tokasaurus.manager.hydragen import (
    group_for_hydragen,
    reorder_decoding_seqs_for_hydragen,
)
from tokasaurus.manager.manager import seqs_to_input
from tokasaurus.manager.types import Sequence
from tokasaurus.model.utils import (
    add_decoding_ids_to_batch_state,
    make_input_batch_state,
    make_model,
    move_batch_state,
    set_async_tp_enabled,
)
from tokasaurus.server.types import (
    SamplingParams,
)
from tokasaurus.utils import timed, timed_with_graph


class ScriptConfig(pydra.Config):
    def __init__(self):
        self.server_config = ServerConfig()
        self.sc = pydra.Alias("server_config")
        self.server_config.kv_cache_num_tokens = 1024 * 512

        self.num_dec = 1
        self.dec_len = 1024

        self.num_pre = 0
        self.pre_len = 1024

        self.num_hyd = 0
        self.hyd_shared_len = 1024
        self.hyd_unique_len = 32

        self.dtype = "bfloat16"
        self.pp_rank = 0
        self.num_iters = 10
        self.num_warmup = 5
        self.compile = False
        self.dynamic = True
        self.fullgraph = True

        self.profile = False
        self.profile_name = "bench_model"

        self.non_blocking = True
        self.plan_before = False
        self.only_plan = False

        self.graph = False

    def prof(self):
        self.profile = True
        self.num_warmup = 3
        self.num_iters = 10
        self.num_profile_repeat = 3

    def total_tokens(self):
        return self.num_dec + self.num_pre * self.pre_len + self.num_hyd

    def finalize(self):
        self.server_config.max_tokens_per_forward = self.total_tokens()
        self.server_config.max_seqs_per_forward = self.total_tokens()

        if self.graph:
            assert self.plan_before and not self.only_plan

    def l8(self):
        self.server_config.model = "meta-llama/Llama-3.1-8B-Instruct"
        self.server_config.kv_cache_num_tokens = 1024 * 192

    def l70(self):
        self.server_config.model = "meta-llama/Llama-3.1-70B-Instruct"

    def l1(self):
        self.server_config.model = "meta-llama/Llama-3.2-1B-Instruct"


@pydra.main(ScriptConfig)
def main(config: ScriptConfig):
    random.seed(0)
    torch.manual_seed(0)

    server_config = config.server_config

    if server_config.tp_size > 1:
        mesh = dist.device_mesh.init_device_mesh(
            device_type="cuda",
            mesh_shape=(server_config.tp_size,),
            mesh_dim_names=("tp",),
        )
        pg = mesh["tp"].get_group()
        tp_rank = mesh["tp"].get_rank()
    else:
        pg = None
        tp_rank = 0

    device = f"cuda:{tp_rank}"
    torch.cuda.set_device(device)

    print(f"Initialized rank {tp_rank} on device {device}")

    def lprint(*args, **kwargs):
        if tp_rank == 0:
            print(*args, **kwargs)

    dtype = getattr(torch, config.dtype)
    model = make_model(
        server_config,
        device=device,
        dtype=dtype,
        pp_rank=config.pp_rank,
        tp_rank=tp_rank,
        tp_group=pg,
    )
    if config.compile:
        lprint("Compiling model...")
        model = torch.compile(model, fullgraph=config.fullgraph, dynamic=config.dynamic)

    lprint(model)

    vocab_size = model.config.vocab_size
    page_size = config.server_config.page_size
    num_pages = config.server_config.kv_cache_num_blocks()

    allocator = BlockAllocator(num_pages, page_size)

    # dummy objects
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0)

    prefill_seqs = []
    prefill_num_pages = math.ceil(config.pre_len / page_size)

    for i in range(config.num_pre):
        seq = Sequence(
            id=f"prefill_{i}",
            completion_total=1,
            batch_index=0,
            kv_indices=[
                random.randint(0, num_pages - 1) for _ in range(prefill_num_pages)
            ],
            input_ids=[
                random.randint(0, vocab_size - 1) for _ in range(config.pre_len)
            ],
            sampling_params=sampling_params,
        )
        prefill_seqs.append((seq, config.pre_len))

    decode_num_pages = math.ceil(config.dec_len / page_size)
    decode_prompt_len = config.dec_len - 1
    decoding_seqs = [
        Sequence(
            id=f"decoding_{i}",
            input_ids=[
                random.randint(0, vocab_size - 1) for _ in range(decode_prompt_len)
            ],
            completion_total=2,
            completion_scheduled=1,
            prompt_scheduled=decode_prompt_len,
            batch_index=0,
            kv_indices=[
                random.randint(0, num_pages - 1) for _ in range(decode_num_pages)
            ],
            sampling_params=sampling_params,
        )
        for i in range(config.num_dec)
    ]

    if config.num_hyd > 0:
        shared_ids = [
            random.randint(0, vocab_size - 1) for _ in range(config.hyd_shared_len)
        ]

        hydragen_seqs = [
            Sequence(
                id=f"hydragen_{i}",
                input_ids=shared_ids,
                completion_total=config.hyd_unique_len,
                batch_index=0,
                sampling_params=sampling_params,
            )
            for i in range(config.num_hyd)
        ]

        for seq in hydragen_seqs:
            kv_indices, num_cached_tokens = allocator.allocate_with_prefix_match(
                seq.id, seq.input_ids
            )
            seq.kv_indices = kv_indices
            seq.prompt_scheduled = len(seq.input_ids)
            seq.completion_scheduled = config.hyd_unique_len - 1
            seq.kv_indices.extend(
                allocator.allocate_up_to_length(
                    seq.id, seq.kv_indices, seq.total_scheduled()
                )
            )

        decoding_seqs.extend(hydragen_seqs)

        if server_config.use_hydragen:
            hydragen_groups = group_for_hydragen(
                allocator.prefix_tree,
                [seq.id for seq in hydragen_seqs],
                min_group_size=config.num_hyd,
                min_prefix_len=config.hyd_shared_len - 2 * server_config.page_size,
                page_size=page_size,
            )

            decoding_seqs = reorder_decoding_seqs_for_hydragen(
                decoding_seqs, hydragen_groups
            )
        else:
            hydragen_groups = None
    else:
        hydragen_groups = None

    inp = seqs_to_input(
        decoding_seqs,
        prefill_seqs,
        schedule_id="schedule_id",
        hydragen_groups=hydragen_groups,
        page_size=page_size,
        starting_prefill_offset=0,
    )

    batch_state = make_input_batch_state(
        inp,
        pp_rank=config.pp_rank,
        pp_size=server_config.pp_size,
        tp_rank=tp_rank,
        tp_size=server_config.tp_size,
    )
    decoding_input_ids = torch.randint(
        0,
        vocab_size,
        (config.num_dec + config.num_hyd,),
        dtype=torch.long,
    )
    add_decoding_ids_to_batch_state(
        batch_state, decoding_input_ids, tp_rank=tp_rank, tp_size=server_config.tp_size
    )
    move_batch_state(
        batch_state,
        device=device,
    )

    batch_size = config.total_tokens()

    if config.pp_rank > 0:
        hidden_states = torch.zeros(
            batch_size,
            model.config.hidden_size,
            device=device,
            dtype=dtype,
        )
        batch_state.hidden_states = hidden_states

    use_async_tp = (
        server_config.async_tp_threshold is not None
        and server_config.tp_size > 1
        and batch_size >= server_config.async_tp_threshold
    )
    print(f"use_async_tp: {use_async_tp}")
    set_async_tp_enabled(use_async_tp)

    def go():
        with torch.inference_mode():
            if not config.plan_before:
                model.plan(batch_state.attention_info, non_blocking=config.non_blocking)
            if config.only_plan:
                return
            _ = model(batch_state, async_tp=use_async_tp)

    if config.plan_before:
        model.plan(batch_state.attention_info, non_blocking=config.non_blocking)

    if config.profile:
        lprint("Running profiler...")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=config.num_warmup,
                active=config.num_iters,
                repeat=config.num_profile_repeat,
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in tqdm(
                range(
                    config.num_profile_repeat
                    * (config.num_iters + config.num_warmup + 1),
                ),
                disable=tp_rank != 0,
            ):
                go()
                prof.step()

        if tp_rank == 0:
            prof.export_chrome_trace(f"local/profs/{config.profile_name}.json")
    else:
        lprint(f"Starting timing (graph={config.graph}) ...")
        time_fn = timed_with_graph if config.graph else timed
        timings = time_fn(go, num_iters=config.num_iters, num_warmup=config.num_warmup)
        lprint(timings.fancy_table())

        mean_ms = timings.mean()
        lprint(f"Tokens per second: {batch_size / mean_ms * 1000:.2f}")


if __name__ == "__main__":
    main()
