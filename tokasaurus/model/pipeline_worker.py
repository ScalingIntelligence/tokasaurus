from dataclasses import dataclass
from typing import Iterable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor

from tokasaurus.common_types import ServerConfig, TimedBarrier
from tokasaurus.model.llama import LlamaForCausalLM
from tokasaurus.model.types import (
    BatchState,
    CartridgeManager,
    LoadCartridge,
    ModelInput,
    ModelOutput,
    ModelOutputTensors,
    NoMoreInputs,
    PipelineWorkerState,
)
from tokasaurus.model.utils import (
    ModelRunner,
    add_decoding_ids_to_batch_state,
    get_dtype,
    get_global_rank,
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


def wait_for_data_dependencies(state: PipelineWorkerState, inp: ModelInput):
    while contains_data_dependency(
        inp=inp,
        inputs_in_front=state.inflight_microbatches,
    ):
        output_ids = state.q_pipe_end_to_start.get()
        handle_output_from_pipeline_end(state, output_ids)


def handle_output_from_pipeline_end(state: PipelineWorkerState, output_ids: Tensor):
    assert state.batch_id_to_last_token is not None

    model_inp = state.inflight_microbatches.popleft()

    batch_ids_to_update = torch.tensor(
        model_inp.lm_head_batch_indices(), dtype=torch.long
    )

    assert batch_ids_to_update.shape == output_ids.shape, (
        f"batch_ids_to_update.shape={batch_ids_to_update.shape} != output_ids.shape={output_ids.shape}"
    )

    state.batch_id_to_last_token[batch_ids_to_update] = output_ids


def handle_outputs_to_manager(state: PipelineWorkerState):
    front_inp, front_out = state.finished_outputs[0]

    assert front_inp.schedule_id == front_out.schedule_id
    schedule_id = front_inp.schedule_id

    microbatch_total = front_inp.microbatch_total
    assert microbatch_total is not None

    if len(state.finished_outputs) >= microbatch_total:
        to_finish = [state.finished_outputs.popleft() for _ in range(microbatch_total)]

        cat_output_tensors: list[ModelOutputTensors] = []

        for i, (mb_inp, mb_out) in enumerate(to_finish):
            assert mb_inp.microbatch_index == i
            assert mb_inp.schedule_id == schedule_id
            assert mb_out.schedule_id == schedule_id

            cat_output_tensors.append(mb_out.tensors)

        cat_output_tokens = torch.cat([x.output_ids for x in cat_output_tensors])
        cat_chosen_token_logprobs = torch.cat(
            [x.chosen_logprobs for x in cat_output_tensors]
        )

        if cat_output_tensors[0].topk_indices is not None:
            cat_topk_indices = torch.cat([x.topk_indices for x in cat_output_tensors])  # type: ignore
            cat_topk_logprobs = torch.cat([x.topk_logprobs for x in cat_output_tensors])  # type: ignore
        else:
            cat_topk_indices = None
            cat_topk_logprobs = None

        cat_tensors = ModelOutputTensors(
            output_ids=cat_output_tokens,
            chosen_logprobs=cat_chosen_token_logprobs,
            topk_indices=cat_topk_indices,
            topk_logprobs=cat_topk_logprobs,
        )

        out = ModelOutput(
            tensors=cat_tensors,
            schedule_id=schedule_id,
        )

        state.q_to_manager.put(out)


@error_propogation_decorator
def pipeline_worker_model_loop(
    state: PipelineWorkerState,
    model: LlamaForCausalLM,
):
    assert state.device_mesh is not None

    config = state.config
    world_size = config.pp_size
    device = model.device

    pp_rank = state.pp_rank
    tp_rank = state.tp_rank
    dp_rank = state.dp_rank

    # Initialize cartridge manager
    cartridge_manager = CartridgeManager(model, config.page_size, logger=state.logger)

    pp_group = state.device_mesh["pp"].get_group()

    if pp_rank > 0:
        pp_src_rank = get_global_rank(config, dp_rank, pp_rank - 1, tp_rank)
    else:
        pp_src_rank = None

    if pp_rank < world_size - 1:
        pp_dst_rank = get_global_rank(config, dp_rank, pp_rank + 1, tp_rank)
    else:
        pp_dst_rank = None

    non_blocking = True

    @dataclass
    class Work:
        model_input: ModelInput
        input_batch_state: BatchState
        output_batch_state: BatchState | None = None
        output_tensors_cpu: ModelOutputTensors | None = None

    def preprocess():
        while True:  # Loop until we get a command that requires model processing
            command = state.input_q.get()
            match command:
                case LoadCartridge():
                    # Load cartridge synchronously before continuing
                    cartridge_manager.load_cartridge(
                        cartridge_id=command.cartridge_id,
                        block_indices=command.block_indices,
                        cartridge_dir=command.cartridge_dir
                    )
                    state.logger.info(f"Loaded cartridge: {command.cartridge_id}")
                    # Continue to next command without returning
                    continue
                    
                case NoMoreInputs():
                    return None  # This is expected to return None
                case _:
                    # ModelInput case - break out of loop to process
                    break

        # Continue with ModelInput processing
        inp = command

        num_total_padding, num_lm_head_padding = model_runner.calc_padding(
            num_prefill_tokens=inp.num_prefill_tokens(),
            num_decode_tokens=inp.num_decode_tokens(),
            num_lm_head_tokens=inp.num_lm_head_tokens(),
        )

        input_batch_state = make_input_batch_state(
            inp,
            pp_rank=pp_rank,
            pp_size=config.pp_size,
            tp_rank=tp_rank,
            tp_size=config.tp_size,
            num_total_padding=num_total_padding,
            num_lm_head_padding=num_lm_head_padding,
        )

        if pp_rank == 0:
            wait_for_data_dependencies(state, inp)

            assert state.batch_id_to_last_token is not None
            decoding_input_ids = state.batch_id_to_last_token[
                torch.tensor(inp.decoding_batch_indices(), dtype=torch.long)
            ]

            add_decoding_ids_to_batch_state(
                input_batch_state=input_batch_state,
                decoding_input_ids=decoding_input_ids,
                tp_rank=tp_rank,
                tp_size=config.tp_size,
            )

            state.inflight_microbatches.append(inp)

        model_runner.plan(input_batch_state, non_blocking=non_blocking)

        move_batch_state(
            input_batch_state=input_batch_state,
            device=device,
            non_blocking=non_blocking,
        )

        return Work(
            model_input=inp,
            input_batch_state=input_batch_state,
        )

    def run_model(work: Work):
        if pp_rank > 0:
            full_bs = work.input_batch_state.position_ids.shape[0]
            assert full_bs % config.tp_size == 0
            bs = full_bs // config.tp_size
            recv = torch.empty(
                bs, model.config.hidden_size, device=device, dtype=model.dtype
            )
            if not work.model_input.skip_pipeline_communication:
                dist.recv(recv, src=pp_src_rank, group=pp_group)

            work.input_batch_state.hidden_states = recv

        output_batch_state = model_runner.run(
            work.input_batch_state, non_blocking=non_blocking
        )
        assert output_batch_state.hidden_states is not None

        # if not last pipeline stage, send hidden states to next stage
        if pp_rank < world_size - 1:
            if not work.model_input.skip_pipeline_communication:
                dist.send(
                    output_batch_state.hidden_states,
                    dst=pp_dst_rank,
                    group=pp_group,
                )

        else:
            unpad_output_batch_state(
                output_batch_state=output_batch_state,
                model_input=work.model_input,
            )

        work.output_batch_state = output_batch_state

    def synchronize(work: Work):
        # NOTE: important to do this for all workers - from what I can tell,
        # if too many nccl sends/recvs are launched by one process without getting
        # fulfilled by other ranks, deadlocks and illegal memory access
        # errors can happen.
        torch.cuda.synchronize()

        if pp_rank != world_size - 1:
            return

        assert work.output_batch_state is not None
        assert work.output_batch_state.outputs is not None

        # NOTE: if there are no output tokens and these tensors are empty,
        # calling .cpu() does not actually cause a sync.
        work.output_tensors_cpu = work.output_batch_state.outputs.to("cpu")

        # we have to send this now (and not in postprocess) because the start of the
        # pipeline may be waiting on it to send the next batch. if this end of the
        # pipeline end worker blocks (i.e. because of a cudaMalloc) after a nccl
        # recv is launched, it will deadlock
        state.q_pipe_end_to_start.put(work.output_tensors_cpu.output_ids)

    def postprocess(work: Work):
        if pp_rank != world_size - 1 or tp_rank != 0:
            return

        assert work.output_tensors_cpu is not None

        out = ModelOutput(
            tensors=work.output_tensors_cpu,
            schedule_id=work.model_input.schedule_id,
            microbatch_index=work.model_input.microbatch_index,
        )

        state.finished_outputs.append((work.model_input, out))
        handle_outputs_to_manager(state)

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


def contains_data_dependency(
    inp: ModelInput,
    inputs_in_front: Iterable[ModelInput],
):
    # checking for data dependencies - we can't schedule a
    # new input (microbatch) if it depends on a token that's in flight
    # NOTE: this set can change between while loop iterations
    # because we add new in flight microbatches as we go
    ids_in_front = set()
    for mb in inputs_in_front:
        ids_in_front.update(mb.lm_head_batch_indices())

    # decoding seqs need to wait on any prev decodes or final-token prefills
    # final-token prefills themselves don't need to wait on anything
    for id in inp.decoding_batch_indices():
        if id in ids_in_front:
            return True

    return False


def start_pipeline_worker(
    config: ServerConfig,
    input_q: mp.Queue,
    q_pipe_end_to_start: mp.Queue,
    q_to_manager: mp.Queue,
    dp_rank: int,
    pp_rank: int,
    tp_rank: int,
    master_port: int,
    process_name: str,
    barrier: TimedBarrier,
):
    setup_logging(config)

    state = PipelineWorkerState(
        config=config,
        input_q=input_q,
        q_pipe_end_to_start=q_pipe_end_to_start,
        q_to_manager=q_to_manager,
        process_name=process_name,
        pp_rank=pp_rank,
        tp_rank=tp_rank,
        dp_rank=dp_rank,
        barrier=barrier,
    )

    if pp_rank == 0:
        state.batch_id_to_last_token = torch.zeros(
            config.max_batch_index(), dtype=torch.long
        )

    state.logger.info(f"Pipeline worker {pp_rank} started!")
    dtype = get_dtype(config.dtype)

    device_mesh, device = setup_distributed(
        config=config,
        dp_rank=dp_rank,
        pp_rank=pp_rank,
        tp_rank=tp_rank,
        master_port=master_port,
    )
    assert device_mesh is not None
    state.device_mesh = device_mesh

    state.logger.info(f"Creating model on device {device} with dtype {dtype}")

    model = make_model(
        config,
        device,
        dtype,
        pp_rank=pp_rank,
        tp_rank=tp_rank,
        tp_group=state.device_mesh["tp"].get_group(),
    )

    state.logger.info("Created model")

    pipeline_worker_model_loop(state=state, model=model)
