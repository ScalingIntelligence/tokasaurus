from dataclasses import dataclass

import torch
import torch.multiprocessing as mp
from loguru import logger
from torch import Tensor

from tokasaurus.common_types import (
    ServerConfig,
    TimedBarrier,
)
from tokasaurus.model.llama import LlamaForCausalLM
from tokasaurus.model.types import (
    BasicWorkerState,
    BatchState,
    CommandFromManager,
    ModelInput,
    ModelOutput,
    ModelOutputTensors,
    NoMoreInputs,
)
from tokasaurus.model.utils import (
    ModelRunner,
    add_decoding_ids_to_batch_state,
    get_dtype,
    make_input_batch_state,
    make_model,
    move_batch_state,
    setup_and_run_loop,
    setup_distributed,
    unpad_output_batch_state,
)
from tokasaurus.utils import (
    error_propogation_decorator,
    setup_logging,
)


def basic_model_loop(
    state: BasicWorkerState,
    model: LlamaForCausalLM,
):
    state.logger.info("Model loop started!")

    tp_rank = state.tp_rank
    tp_size = state.config.tp_size
    non_blocking = True

    @dataclass
    class Work:
        model_input: ModelInput
        input_batch_state: BatchState
        batch_indices: Tensor
        output_batch_state: BatchState | None = None
        output_tensors_cpu: ModelOutputTensors | None = None

    def preprocess():
        command: CommandFromManager = state.input_q.get()

        match command:
            case ModelInput():
                inp = command
            case NoMoreInputs():
                return None
            case _:
                raise ValueError(f"Unknown command: {type(command)}")

        batch_indices = torch.tensor(
            inp.batch_indices,
            dtype=torch.long,
        )

        num_total_padding, num_lm_head_padding = model_runner.calc_padding(
            num_prefill_tokens=inp.num_prefill_tokens(),
            num_decode_tokens=inp.num_decode_tokens(),
            num_lm_head_tokens=inp.num_lm_head_tokens(),
        )

        input_batch_state = make_input_batch_state(
            inp,
            tp_rank=tp_rank,
            tp_size=tp_size,
            num_total_padding=num_total_padding,
            num_lm_head_padding=num_lm_head_padding,
        )

        model_runner.plan(input_batch_state, non_blocking=non_blocking)

        move_batch_state(
            input_batch_state=input_batch_state,
            device=state.device,
            non_blocking=non_blocking,
        )

        return Work(
            model_input=inp,
            input_batch_state=input_batch_state,
            batch_indices=batch_indices.to(state.device, non_blocking=non_blocking),
        )

    def run_model(work: Work):
        decoding_batch_indices = work.batch_indices[
            work.model_input.decode_start_pos() :
        ]
        decoding_input_ids = state.batch_index_to_last_token[decoding_batch_indices]

        input_batch_state = work.input_batch_state

        add_decoding_ids_to_batch_state(
            input_batch_state=input_batch_state,
            decoding_input_ids=decoding_input_ids,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

        output_batch_state = model_runner.run(
            input_batch_state, non_blocking=non_blocking
        )

        unpad_output_batch_state(
            output_batch_state=output_batch_state,
            model_input=work.model_input,
        )

        if input_batch_state.raw_lm_head_indices is not None:
            lm_head_indices = input_batch_state.raw_lm_head_indices
        else:
            lm_head_indices = input_batch_state.lm_head_indices

        assert lm_head_indices is not None
        batch_indices = work.batch_indices[lm_head_indices]

        if len(batch_indices) > 0:
            assert output_batch_state.outputs is not None
            state.batch_index_to_last_token[batch_indices] = (
                output_batch_state.outputs.output_ids
            )

        work.output_batch_state = output_batch_state

    def synchronize(work: Work):
        # technically, we don't need to sync when tp_rank != 0,
        # but omitting it causes sporadic nccl illegal memory access errors
        torch.cuda.synchronize()

        work.output_tensors_cpu = work.output_batch_state.outputs.to("cpu")

    def postprocess(work: Work):
        if state.tp_rank != 0:
            return

        assert work.output_tensors_cpu is not None

        out = ModelOutput(
            tensors=work.output_tensors_cpu,
            schedule_id=work.model_input.schedule_id,
        )

        state.q_model_to_manager.put(out)

    model_runner = ModelRunner(
        config=state.config,
        model=model,
    )

    setup_and_run_loop(
        state=state,
        model_runner=model_runner,
        preprocess=preprocess,
        run_model=run_model,
        synchronize=synchronize,
        postprocess=postprocess,
    )


@error_propogation_decorator
def start_basic_model_worker(
    config: ServerConfig,
    input_q: mp.Queue,
    q_model_to_manager: mp.Queue,
    dp_rank: int,
    tp_rank: int,
    master_port: int,
    process_name: str,
    barrier: TimedBarrier,
):
    setup_logging(config)

    device_mesh, device = setup_distributed(
        config,
        dp_rank=dp_rank,
        pp_rank=0,
        tp_rank=tp_rank,
        master_port=master_port,
    )
    dtype = get_dtype(config.dtype)

    batch_index_to_last_token = torch.zeros(
        config.max_batch_index(), dtype=torch.long, device=device
    )

    state = BasicWorkerState(
        config=config,
        batch_index_to_last_token=batch_index_to_last_token,
        input_q=input_q,
        q_model_to_manager=q_model_to_manager,
        device=device,
        dtype=dtype,
        process_name=process_name,
        tp_rank=tp_rank,
        barrier=barrier,
    )

    state.logger.info("Model worker started!")
    state.logger.info(f"Creating model on device {device} with dtype {dtype}")

    model = make_model(
        config,
        device,
        dtype,
        tp_rank=tp_rank,
        tp_group=device_mesh["tp"].get_group() if device_mesh is not None else None,
    )

    state.logger.info("Created model")

    basic_model_loop(state, model)


def start_fanout_worker(
    config: ServerConfig,
    input_q: mp.Queue,
    fanout_qs: list[mp.Queue],
    process_name: str,
    barrier: TimedBarrier,
):
    setup_logging(config)
    bound_logger = logger.bind(process_name=process_name)

    bound_logger.info("Fanout worker started!")

    barrier.wait()

    while True:
        inp = input_q.get()
        for q in fanout_qs:
            q.put(inp)
