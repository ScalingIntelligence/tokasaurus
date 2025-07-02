import json
import math
import os
from copy import deepcopy
from dataclasses import dataclass, replace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from flashinfer import cascade
from torch import Tensor
from torch.distributed._symmetric_memory import enable_symm_mem_for_group
from tqdm import tqdm

from tokasaurus.common_types import ServerConfig
from tokasaurus.manager.input_building import make_dummy_batch
from tokasaurus.model.attention_utils import create_wrappers_for_cudagraph
from tokasaurus.model.llama import LlamaForCausalLM
from tokasaurus.model.qwen import Qwen2ForCausalLM
from tokasaurus.model.qwen3 import Qwen3ForCausalLM
from tokasaurus.model.types import (
    BasicWorkerState,
    BatchState,
    DeviceType,
    ExtraModelConfig,
    ModelInput,
    NoMoreInputs,
    PipelineWorkerState,
    WrapperCollection,
)


def get_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
        "fp8": torch.float8_e4m3fn,
    }
    return dtype_map[dtype_str]


def get_global_rank(config: ServerConfig, dp_rank: int, pp_rank: int, tp_rank: int):
    """
    parallelism order from outer to inner: dp -> pp -> tp
    """
    return (
        dp_rank * config.pp_size * config.tp_size + (pp_rank * config.tp_size) + tp_rank
    )


def setup_distributed(
    config: ServerConfig, dp_rank: int, pp_rank: int, tp_rank: int, master_port: int
):
    global_rank = get_global_rank(config, dp_rank, pp_rank, tp_rank)

    device = f"cuda:{global_rank}"
    torch.cuda.set_device(device)

    if config.pp_size > 1 or config.tp_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(global_rank)
        os.environ["WORLD_SIZE"] = str(config.dp_size * config.pp_size * config.tp_size)

        device_mesh = dist.device_mesh.init_device_mesh(
            device_type="cuda",
            mesh_shape=(config.dp_size, config.pp_size, config.tp_size),
            mesh_dim_names=("dp", "pp", "tp"),
        )
    else:
        device_mesh = None

    return device_mesh, device


def last_page_len(length: int, page_size: int):
    last_page_len = length % page_size
    if last_page_len == 0:
        last_page_len = page_size

    return last_page_len


def tp_slice(x: Tensor, tp_rank: int, tp_size: int) -> Tensor:
    if tp_size == 1:
        return x

    bs = x.shape[0]
    assert bs % tp_size == 0, f"bs={bs} must be divisible by tp_size={tp_size}"
    bs_per_rank = bs // tp_size
    tp_start = tp_rank * bs_per_rank
    tp_end = tp_start + bs_per_rank

    # clone is important for compile - compile
    # guards are used that check for if something is a view
    return x[tp_start:tp_end].clone()


def pad_and_slice_tensor(
    x: Tensor, num_padding: int, tp_rank: int, tp_size: int
) -> Tensor:
    if num_padding > 0:
        x = F.pad(x, (0, num_padding))

    return tp_slice(x, tp_rank, tp_size)


def make_input_batch_state(
    inp: ModelInput,
    tp_rank: int = 0,
    tp_size: int = 1,
    pp_rank: int = 0,
    pp_size: int = 1,
    num_total_padding: int = 0,
    num_lm_head_padding: int = 0,
):
    skip_input_ids = pp_size > 1 and pp_rank > 0
    skip_lm_head_indices = pp_size > 1 and pp_rank < pp_size - 1

    if skip_input_ids:
        prefill_input_ids = None
    else:
        prefill_input_ids = torch.tensor(
            inp.prefill_input_ids,
            dtype=torch.long,
        )

    position_ids = torch.tensor(inp.position_ids, dtype=torch.long)

    if skip_lm_head_indices:
        lm_head_indices = None
    else:
        lm_head_indices = torch.tensor(inp.lm_head_indices, dtype=torch.long)

    # Build attention_info from the builder
    attention_info = inp.build_attention_info()

    # Build sampling_params from the builder
    sampling_params = inp.build_sampling_params()

    input_batch_state = BatchState(
        prefill_input_ids=prefill_input_ids,
        attention_info=attention_info,
        position_ids=position_ids,
        sampling_params=sampling_params,
        lm_head_indices=lm_head_indices,
        num_total_padding=num_total_padding,
        num_lm_head_padding=num_lm_head_padding,
    )
    input_batch_state.attention_info.num_padding = num_total_padding

    if num_total_padding > 0:
        input_batch_state.position_ids = F.pad(
            input_batch_state.position_ids,
            (0, num_total_padding),
            value=0,
        )

    if not skip_lm_head_indices:
        assert input_batch_state.lm_head_indices is not None

        if (tp_size > 1 or num_lm_head_padding > 0) and pp_size == 1:
            # in this case, we need the OG lm head indices to update the most-recent-token buffers
            input_batch_state.raw_lm_head_indices = input_batch_state.lm_head_indices

        input_batch_state.lm_head_indices = pad_and_slice_tensor(
            input_batch_state.lm_head_indices,
            num_lm_head_padding,
            tp_rank,
            tp_size,
        )

        if (greedy_mask := input_batch_state.sampling_params.greedy_mask) is not None:
            input_batch_state.sampling_params.greedy_mask = pad_and_slice_tensor(
                greedy_mask,
                num_lm_head_padding,
                tp_rank,
                tp_size,
            )

        if (top_p := input_batch_state.sampling_params.top_p) is not None:
            input_batch_state.sampling_params.top_p = pad_and_slice_tensor(
                top_p,
                num_lm_head_padding,
                tp_rank,
                tp_size,
            )

        if (temperature := input_batch_state.sampling_params.temperature) is not None:
            input_batch_state.sampling_params.temperature = pad_and_slice_tensor(
                temperature,
                num_lm_head_padding,
                tp_rank,
                tp_size,
            )

    return input_batch_state


def add_decoding_ids_to_batch_state(
    input_batch_state: BatchState,
    decoding_input_ids: Tensor,
    tp_rank: int = 0,
    tp_size: int = 1,
):
    assert input_batch_state.prefill_input_ids is not None
    input_ids = torch.cat(
        [input_batch_state.prefill_input_ids, decoding_input_ids], dim=0
    )

    padded_sliced = pad_and_slice_tensor(
        input_ids,
        num_padding=input_batch_state.num_total_padding,
        tp_rank=tp_rank,
        tp_size=tp_size,
    )

    input_batch_state.input_ids = padded_sliced
    input_batch_state.prefill_input_ids = None


def move_batch_state(
    input_batch_state: BatchState,
    device: DeviceType,
    non_blocking: bool = False,
):
    input_batch_state.position_ids = input_batch_state.position_ids.to(
        device, non_blocking=non_blocking
    )

    input_batch_state.attention_info.append_kv_token_indices = (
        input_batch_state.attention_info.append_kv_token_indices.to(
            device, non_blocking=non_blocking
        )
    )

    input_batch_state.sampling_params = input_batch_state.sampling_params.to(
        device, non_blocking=non_blocking
    )

    if input_batch_state.lm_head_indices is not None:
        input_batch_state.lm_head_indices = input_batch_state.lm_head_indices.to(
            device, non_blocking=non_blocking
        )

    if input_batch_state.raw_lm_head_indices is not None:
        input_batch_state.raw_lm_head_indices = (
            input_batch_state.raw_lm_head_indices.to(device, non_blocking=non_blocking)
        )

    if input_batch_state.input_ids is not None:
        input_batch_state.input_ids = input_batch_state.input_ids.to(
            device, non_blocking=non_blocking
        )

    if input_batch_state.prefill_input_ids is not None:
        input_batch_state.prefill_input_ids = input_batch_state.prefill_input_ids.to(
            device, non_blocking=non_blocking
        )


def run_overlapped_loop(
    preprocess,
    run_model,
    synchronize,
    postprocess,
    max_iters: int | None = None,
    prog_bar_name: str | None = None,
):
    preproc_work = None
    run_work = None
    postproc_work = None

    iter_num = 0

    if max_iters is not None and prog_bar_name is not None:
        prog_bar = tqdm(total=max_iters, desc=prog_bar_name)
    else:
        prog_bar = None

    while True:
        if preproc_work is not None:
            run_work = preproc_work
            preproc_work = None
        else:
            # non-overlapped preprocess, our goal is to
            # avoid these whenever possible
            run_work = preprocess()

        assert run_work is not None
        run_model(run_work)

        if postproc_work is not None:
            postprocess(postproc_work)
            postproc_work = None

        preproc_work = preprocess()

        synchronize(run_work)
        postproc_work = run_work

        # if we're going to block waiting for the next input, postprocess now
        if preproc_work is None:
            postprocess(postproc_work)
            postproc_work = None

        iter_num += 1

        if prog_bar is not None:
            prog_bar.update(1)

        if max_iters is not None and iter_num >= max_iters:
            break


model_type = LlamaForCausalLM | Qwen2ForCausalLM | Qwen3ForCausalLM

models: dict[str, type[model_type]] = {
    "llama": LlamaForCausalLM,
    "qwen2": Qwen2ForCausalLM,
    "qwen3": Qwen3ForCausalLM,
}


def make_model(
    config: ServerConfig,
    device: str,
    dtype: torch.dtype,
    pp_rank: int = 0,
    tp_rank: int = 0,
    tp_group: dist.ProcessGroup | None = None,
) -> LlamaForCausalLM:
    model_config = config.model_config()
    model_type = model_config.model_type  # type: ignore
    base_class = models[model_type]

    extra_model_config = ExtraModelConfig(
        pp_size=config.pp_size,
        pp_rank=pp_rank,
        tp_size=config.tp_size,
        tp_rank=tp_rank,
        tp_group=tp_group,
        torch_compile=config.torch_compile,
        enable_chosen_logprobs=config.enable_chosen_logprobs,
        topk_logprobs=config.max_topk_logprobs,
    )

    if config.rope_scaling is not None:
        extra_model_config.rope_scaling = json.loads(config.rope_scaling)

    model = base_class.from_pretrained(
        config.model,
        extra_config=extra_model_config,
        dtype=dtype,
        device=device,
    )

    num_pages = config.kv_cache_num_blocks()

    # extra page to point padding append indices when using cudagraphs
    if config.use_cudagraphs:
        num_pages += 1

    model.setup_caches(num_pages=num_pages, page_size=config.page_size)

    if config.torch_compile:
        use_async_tp = config.tp_size > 1 and config.async_tp_threshold is not None
        if use_async_tp:
            torch._dynamo.config.cache_size_limit = 32  # type: ignore
            assert tp_group is not None
            enable_symm_mem_for_group(tp_group.group_name)

        model = torch.compile(model, fullgraph=True, dynamic=True)

    return model


def set_async_tp_enabled(enabled: bool):
    torch._inductor.config._micro_pipeline_tp = enabled  # type: ignore


def run_warmup_batches(
    config: ServerConfig,
    input_q: mp.Queue,
    process_name: str,
    preprocess,
    run_model,
    synchronize,
    postprocess,
    device: DeviceType,
    dtype: torch.dtype,
):
    """
    Send a max-sized batch to the model to check if it's gonna OOM.

    Can also send more batches to try to trigger recompiles ahead of time.
    """

    max_tokens_per_forward = math.ceil(config.max_tokens_per_forward / config.pp_size)
    max_lm_head_tokens_per_forward = min(
        config.max_seqs_per_forward, max_tokens_per_forward
    )

    configs = []

    decode_sizes = [
        0,
        1,
        1 * config.tp_size,
        2 * config.tp_size,
        max_lm_head_tokens_per_forward,
    ]

    prefill_sizes = [
        0,
        1,
        1 * config.tp_size,
        2 * config.tp_size,
        max_tokens_per_forward - max_lm_head_tokens_per_forward,
        max_tokens_per_forward,
    ]

    if config.async_tp_threshold is not None:
        prefill_sizes.extend(
            range(
                config.async_tp_threshold - 3,
                config.async_tp_threshold + 1,
            )
        )

    for num_decode_tokens in decode_sizes:
        for num_prefill_tokens in prefill_sizes:
            for prefill_uses_lm_head in [True, False]:
                total_tokens = num_prefill_tokens + num_decode_tokens
                total_lm_head_tokens = num_decode_tokens + (
                    1 if prefill_uses_lm_head else 0
                )
                if total_tokens <= 0 or total_tokens > max_tokens_per_forward:
                    continue

                if total_lm_head_tokens > max_lm_head_tokens_per_forward:
                    continue

                if num_prefill_tokens == 0 and prefill_uses_lm_head:
                    continue

                configs.append(
                    (num_prefill_tokens, num_decode_tokens, prefill_uses_lm_head)
                )

    # sort configs by biggest first (and then tie-break prioritizing decode),
    # so we can discover OOMs as soon as possible
    configs.sort(key=lambda x: (x[0] + x[1], x[1]), reverse=True)

    inputs: list[ModelInput] = []

    for num_prefill_tokens, num_decode_tokens, prefill_uses_lm_head in configs:
        inp = make_dummy_batch(
            config=config,
            prefill_tokens=num_prefill_tokens,
            decode_tokens=num_decode_tokens,
            prefill_uses_lm_head=prefill_uses_lm_head,
            skip_pipeline_communication=True,
        )
        inputs.append(inp)

    if config.pp_size > 1:
        for inp in inputs.copy():
            copy_inp = deepcopy(inp)
            copy_inp.skip_pipeline_communication = False
            inputs.append(copy_inp)

    for inp in inputs:
        input_q.put(inp)

    input_q.put(NoMoreInputs())

    run_overlapped_loop(
        preprocess,
        run_model,
        synchronize,
        postprocess,
        max_iters=len(inputs),
        prog_bar_name=f"Warmup loop for {process_name}",
    )

    # Triggering the compilation/loading of the flashinfer merge
    # kernel here during server startup.
    # TODO: actually send hydragen batches in the warmup loop instead of this.
    if config.use_hydragen:
        bs = 8
        num_heads = 4
        hdim = 256
        out1 = torch.randn(bs, num_heads, hdim, device=device, dtype=dtype)
        out2 = torch.randn(bs, num_heads, hdim, device=device, dtype=dtype)
        lse1 = torch.randn(bs, num_heads, device=device, dtype=torch.float32)
        lse2 = torch.randn(bs, num_heads, device=device, dtype=torch.float32)

        cascade.merge_state(
            out1,
            lse1,
            out2,
            lse2,
        )


@dataclass
class CUDAGraphInfo:
    config: ServerConfig
    graph: torch.cuda.CUDAGraph
    input_batch_state: BatchState
    output_batch_state: BatchState
    wrappers: WrapperCollection
    model: LlamaForCausalLM
    num_decode_tokens: int

    def __post_init__(self):
        self.pp_rank = self.model.extra_config.pp_rank
        self.pp_size = self.model.extra_config.pp_size
        self.tp_rank = self.model.extra_config.tp_rank
        self.tp_size = self.model.extra_config.tp_size

    def copy_into_input_batch_state(
        self, new_input_batch_state: BatchState, non_blocking: bool = False
    ):
        pp_rank = self.pp_rank
        pp_size = self.pp_size

        def copy_into(src: Tensor | None, dst: Tensor | None):
            assert src is not None
            assert dst is not None

            assert src.shape[0] <= dst.shape[0], (
                f"src.shape[0]={src.shape[0]} > dst.shape[0]={dst.shape[0]}"
            )
            dst[: src.shape[0]].copy_(src, non_blocking=non_blocking)

        copy_into(
            new_input_batch_state.position_ids,
            self.input_batch_state.position_ids,
        )

        # the append indices is the only tensor that actually affects the state of our model
        # we point unused indices (i.e. padding) to the dummy last page we've added
        # to the graph to avoid overwriting stuff we care about.

        self.input_batch_state.attention_info.append_kv_token_indices.fill_(
            self.config.kv_cache_num_blocks() * self.config.page_size
        )
        copy_into(
            new_input_batch_state.attention_info.append_kv_token_indices,
            self.input_batch_state.attention_info.append_kv_token_indices,
        )

        if pp_rank == 0:
            copy_into(new_input_batch_state.input_ids, self.input_batch_state.input_ids)

        if pp_rank == pp_size - 1:
            assert self.input_batch_state.lm_head_indices is not None
            assert new_input_batch_state.lm_head_indices is not None

            num_lm_head_padding = (
                self.input_batch_state.lm_head_indices.shape[0]
                - new_input_batch_state.lm_head_indices.shape[0]
            )
            assert num_lm_head_padding >= 0
            num_indices_per_device = new_input_batch_state.lm_head_indices.shape[0]
            lm_head_index_devices = (
                new_input_batch_state.lm_head_indices // num_indices_per_device
            )
            lm_head_index_padding_offsets = lm_head_index_devices * num_lm_head_padding
            lm_head_indices_with_padding = (
                new_input_batch_state.lm_head_indices + lm_head_index_padding_offsets
            )

            copy_into(
                lm_head_indices_with_padding,
                self.input_batch_state.lm_head_indices,
            )

            sampling_params = self.input_batch_state.sampling_params
            assert sampling_params is not None
            if (greedy_mask := sampling_params.greedy_mask) is not None:
                copy_into(
                    new_input_batch_state.sampling_params.greedy_mask, greedy_mask
                )

            if (temperature := sampling_params.temperature) is not None:
                copy_into(
                    new_input_batch_state.sampling_params.temperature, temperature
                )

            if (top_p := sampling_params.top_p) is not None:
                copy_into(new_input_batch_state.sampling_params.top_p, top_p)

        if pp_rank > 0:
            copy_into(
                new_input_batch_state.hidden_states,
                self.input_batch_state.hidden_states,
            )

    def run(
        self,
        input_batch_state: BatchState,
        non_blocking: bool = False,
    ):
        assert input_batch_state.attention_info.hydragen_info is None

        self.copy_into_input_batch_state(input_batch_state, non_blocking)
        self.graph.replay()

        # making a copy since later we sometimes slice the tensors in the object
        assert self.output_batch_state.outputs is not None
        input_batch_state.outputs = replace(self.output_batch_state.outputs)
        input_batch_state.hidden_states = self.output_batch_state.hidden_states

        return input_batch_state

    def plan(self, input_batch_state: BatchState, non_blocking: bool = False):
        assert input_batch_state.attention_info.hydragen_info is None
        self.model.set_wrappers(self.wrappers)
        self.model.plan(input_batch_state.attention_info, non_blocking=non_blocking)


@torch.inference_mode()
def create_cudagraph(
    config: ServerConfig,
    model: LlamaForCausalLM,
    num_decode_tokens: int,
    pp_rank: int,
    tp_rank: int,
    workspace_buffer: Tensor | None = None,
):
    tp_size = config.tp_size
    assert num_decode_tokens % tp_size == 0

    device = model.device
    orig_wrappers = model.wrapper_collection

    dummy_inp = make_dummy_batch(
        config=config,
        prefill_tokens=0,
        decode_tokens=num_decode_tokens,
    )

    input_batch_state = make_input_batch_state(
        inp=dummy_inp,
        tp_rank=tp_rank,
        tp_size=config.tp_size,
        pp_rank=pp_rank,
        pp_size=config.pp_size,
    )

    if pp_rank == 0:
        add_decoding_ids_to_batch_state(
            input_batch_state=input_batch_state,
            decoding_input_ids=torch.zeros(
                num_decode_tokens,
                dtype=torch.long,
            ),
            tp_rank=tp_rank,
            tp_size=config.tp_size,
        )

    move_batch_state(
        input_batch_state,
        device=device,
    )

    if pp_rank != 0:
        input_batch_state.hidden_states = torch.zeros(
            input_batch_state.position_ids.shape[0] // tp_size,
            model.config.hidden_size,
            dtype=model.dtype,
            device=device,
        )

    wrappers = create_wrappers_for_cudagraph(
        device=model.device,
        num_attention_heads=model.num_qo_heads(),
        num_key_value_heads=model.num_kv_heads(),
        num_decode_sequences=num_decode_tokens,
        max_kv_indices=config.cudagraph_max_kv_indices_per_seq * num_decode_tokens,
        workspace_buffer=workspace_buffer,
    )

    model.set_wrappers(wrappers)
    model.plan(input_batch_state.attention_info)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):  # type: ignore
        # warmup
        for _ in range(3):
            model(input_batch_state)

    torch.cuda.current_stream().wait_stream(s)

    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_batch_state = model(input_batch_state)

    model.set_wrappers(orig_wrappers)

    return CUDAGraphInfo(
        config=config,
        graph=g,
        input_batch_state=input_batch_state,
        output_batch_state=out_batch_state,
        wrappers=wrappers,
        model=model,
        num_decode_tokens=num_decode_tokens,
    )


class ModelRunner:
    def __init__(
        self,
        config: ServerConfig,
        model: LlamaForCausalLM,
    ):
        self.config = config
        self.model = model
        self.default_wrappers = model.wrapper_collection
        assert self.default_wrappers is not None

        self.recorded_graphs = False

    def should_use_async_tp_model(self, batch_state: BatchState):
        num_tokens = batch_state.position_ids.shape[0]
        return (
            self.config.async_tp_threshold is not None
            and num_tokens >= self.config.async_tp_threshold
        )

    def run_default(
        self,
        input_batch_state: BatchState,
    ):
        use_async_tp = self.should_use_async_tp_model(input_batch_state)

        self.model.set_wrappers(self.default_wrappers)

        set_async_tp_enabled(use_async_tp)

        output_batch_state: BatchState = self.model(
            input_batch_state, async_tp=use_async_tp
        )
        set_async_tp_enabled(False)

        return output_batch_state

    def plan_default(
        self,
        input_batch_state: BatchState,
        non_blocking: bool = False,
    ):
        self.model.set_wrappers(self.default_wrappers)
        self.model.plan(input_batch_state.attention_info, non_blocking=non_blocking)

    def match_to_graph(self, num_prefill_tokens: int, num_decode_tokens: int):
        if (
            num_prefill_tokens > 0
            or num_decode_tokens > self.config.cudagraph_max_size
            or not self.recorded_graphs
        ):
            return None

        assert num_decode_tokens > 0

        for i, graph in enumerate(self.graphs):
            graph_tokens = graph.num_decode_tokens
            if num_decode_tokens <= graph_tokens:
                return i

        raise RuntimeError(f"Shouldn't get here, num_decode_tokens={num_decode_tokens}")

    def match_state_to_graph(self, input_batch_state: BatchState):
        num_prefill_tokens = input_batch_state.attention_info.prefill_info.num_tokens
        num_decode_tokens = input_batch_state.attention_info.decode_info.num_tokens
        return self.match_to_graph(num_prefill_tokens, num_decode_tokens)

    def calc_padding(
        self, num_prefill_tokens: int, num_decode_tokens: int, num_lm_head_tokens: int
    ):
        """
        returns (all_tokens_padding, lm_head_tokens_padding)
        """
        matched_graph_index = self.match_to_graph(num_prefill_tokens, num_decode_tokens)

        if matched_graph_index is not None:
            # cudagraph padding
            assert num_prefill_tokens == 0
            assert num_lm_head_tokens == num_decode_tokens

            graph_size = self.graphs[matched_graph_index].num_decode_tokens
            assert graph_size % self.config.tp_size == 0

            decode_padding = graph_size - num_decode_tokens

            return decode_padding, decode_padding
        else:
            # normal TP padding
            num_total_tokens = num_prefill_tokens + num_decode_tokens

            padded_num_total_tokens = (
                math.ceil(num_total_tokens / self.config.tp_size) * self.config.tp_size
            )
            total_padding = padded_num_total_tokens - num_total_tokens

            padded_num_lm_head_tokens = (
                math.ceil(num_lm_head_tokens / self.config.tp_size)
                * self.config.tp_size
            )
            lm_head_indices_padding = padded_num_lm_head_tokens - num_lm_head_tokens

            return total_padding, lm_head_indices_padding

    @torch.inference_mode()
    def plan(self, input_batch_state: BatchState, non_blocking: bool = False):
        graph_index = self.match_state_to_graph(input_batch_state)
        if graph_index is None:
            self.plan_default(input_batch_state, non_blocking)
            return

        graph = self.graphs[graph_index]
        input_batch_state.attention_info.decode_info.pad_for_cudagraph(
            graph.num_decode_tokens
        )
        graph.plan(input_batch_state, non_blocking)

    @torch.inference_mode()
    def run(
        self,
        input_batch_state: BatchState,
        non_blocking: bool = False,
    ):
        graph_index = self.match_state_to_graph(input_batch_state)
        if graph_index is None:
            return self.run_default(input_batch_state)

        return self.graphs[graph_index].run(input_batch_state, non_blocking)

    def record_graphs(self, process_name: str):
        # reuse the workspace buffer from the default wrappers
        workspace_buffer = (
            self.model.wrapper_collection.decode_wrapper._float_workspace_buffer
        )

        max_bs = self.config.cudagraph_max_size
        step = self.config.cudagraph_step

        assert max_bs % step == 0

        cuda_graph_sizes = list(range(step, max_bs + 1, step))

        graphs = list[CUDAGraphInfo]()
        for num_decode_tokens in tqdm(
            cuda_graph_sizes,
            desc=f"Capturing cudagraphs for {process_name}",
            disable=len(cuda_graph_sizes) == 0,
        ):
            graphs.append(
                create_cudagraph(
                    config=self.config,
                    model=self.model,
                    num_decode_tokens=num_decode_tokens,
                    pp_rank=self.model.extra_config.pp_rank,
                    tp_rank=self.model.extra_config.tp_rank,
                    workspace_buffer=workspace_buffer,
                )
            )

        self.graphs = graphs
        self.recorded_graphs = True


def setup_and_run_loop(
    state: BasicWorkerState | PipelineWorkerState,
    model_runner: ModelRunner,
    preprocess,
    run_model,
    synchronize,
    postprocess,
):
    if state.config.use_cudagraphs:
        model_runner.record_graphs(state.process_name)

    run_warmup_batches(
        config=state.config,
        input_q=state.input_q,
        process_name=state.process_name,
        preprocess=preprocess,
        run_model=run_model,
        synchronize=synchronize,
        postprocess=lambda _: None,
        device=model_runner.model.device,
        dtype=model_runner.model.dtype,
    )

    state.barrier.wait()

    run_overlapped_loop(
        preprocess=preprocess,
        run_model=run_model,
        synchronize=synchronize,
        postprocess=postprocess,
    )


def unpad_output_batch_state(
    output_batch_state: BatchState,
    model_input: ModelInput,
):
    assert output_batch_state.outputs is not None

    num_lm_head_tokens = model_input.num_lm_head_tokens()

    output_batch_state.outputs.output_ids = output_batch_state.outputs.output_ids[
        :num_lm_head_tokens
    ]

    if output_batch_state.outputs.chosen_logprobs is not None:
        output_batch_state.outputs.chosen_logprobs = (
            output_batch_state.outputs.chosen_logprobs[:num_lm_head_tokens]
        )

    if output_batch_state.outputs.topk_indices is not None:
        output_batch_state.outputs.topk_indices = (
            output_batch_state.outputs.topk_indices[:num_lm_head_tokens]
        )

    if output_batch_state.outputs.topk_logprobs is not None:
        output_batch_state.outputs.topk_logprobs = (
            output_batch_state.outputs.topk_logprobs[:num_lm_head_tokens]
        )
