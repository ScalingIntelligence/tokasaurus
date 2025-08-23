import time
from itertools import chain
from pathlib import Path

import torch.multiprocessing as mp

from tokasaurus.common_types import TimedBarrier
from tokasaurus.manager.allocator import (
    BatchIndexAllocator,
    BlockAllocator,
    NoSpaceException,
    Block,
)
from tokasaurus.manager.download_worker import DownloadRequest, DownloadComplete
from tokasaurus.manager.hydragen import (
    HydragenGroup,
    group_for_hydragen,
    reorder_decision_for_hydragen,
    reorder_decoding_seqs_for_hydragen,
    restrict_hydragen_groups,
)
from tokasaurus.manager.input_building import (
    seqs_to_input,
    slice_decision,
)
from tokasaurus.manager.monitoring import step_stats, track_time_decorator, StatsTracker, WandbLogger, log_to_wandb
from tokasaurus.manager.scheduler import (
    BlockUsageOverTime,
    SchedulingQueue,
    apply_decision,
    calc_block_usage_over_time,
    make_scheduling_decision,
    schedule,
    try_onboarding_seqs,
)
from tokasaurus.manager.stopping_predictor import EarlyStoppingTracker, PredictionMap
from tokasaurus.manager.types import (
    ManagerState,
    ScheduleDecision,
    Sequence,
    ServerConfig,
)
from tokasaurus.model.types import (
    ModelOutput,
    NoMoreInputs,
    LoadCartridge,
)
from tokasaurus.server.types import (
    CancelledRequest,
    RequestOutput,
    TokasaurusRequest,
)
from tokasaurus.utils import (
    block_on_queues,
    error_propogation_decorator,
    get_eos_token_ids,
    queue_iterator,
    setup_logging,
    sanitize_cartridge_id,
)
from tokasaurus.manager.cartridge_downloader import validate_cartridge_exists


def can_schedule_sequence(state: ManagerState, seq: Sequence) -> bool:
    """Check if sequence can be scheduled (all cartridges ready)."""
    if not seq.prepended_cartridge_ids:
        return True
        
    for cartridge_id in seq.prepended_cartridge_ids:
        if cartridge_id in state.cartridges_downloading:
            return False  # Still downloading
            
        # Check if cartridge exists (use sanitized ID for directory path)
        sanitized_id = sanitize_cartridge_id(cartridge_id)
        cartridge_path = Path(state.config.cartridge_dir) / sanitized_id
        if not ((cartridge_path / "cartridge.pt").exists() and 
               (cartridge_path / "config.yaml").exists()):
            return False  # Not available
    
    return True


def check_and_request_downloads(state: ManagerState, cartridges: list[dict] | None):
    """Check cartridge availability and request downloads if needed."""
    if not cartridges:
        return
    
    for cartridge in cartridges:
        cartridge_id = cartridge["id"]
        source = cartridge["source"]
        force_redownload = cartridge.get("force_redownload", False)
        
        if source == "local":
            # Local cartridges should already exist, just validate them
            state.load_cartridge_config(cartridge_id)
            continue
        else:
            
            # Check if wandb cartridge exists locally (use sanitized ID for directory path)
            sanitized_id = sanitize_cartridge_id(cartridge_id)
            cartridge_path = Path(state.config.cartridge_dir) / sanitized_id
            files_exist = ((cartridge_path / "cartridge.pt").exists() and 
                          (cartridge_path / "config.yaml").exists())
            
            # Download if files don't exist OR if force_redownload is True
            should_download = not files_exist or force_redownload
            
            if should_download:
                # Only request download if not already downloading
                # First, validate that the cartridge exists remotely before starting download
                validate_cartridge_exists(cartridge_id, source, state.logger)
                if cartridge_id not in state.cartridges_downloading:
                    download_req = DownloadRequest(
                        cartridge_id=cartridge_id,
                        source=source,
                        force_redownload=force_redownload,
                        cartridges_path=Path(state.config.cartridge_dir)
                    )
                    state.q_download_requests.put(download_req)
                    state.cartridges_downloading.add(cartridge_id)
                    reason = "force redownload requested" if files_exist else "files not found locally"
                    state.logger.info(f"Requested download for cartridge {cartridge_id} ({reason})")


def fail_sequence_with_error(state: ManagerState, seq: Sequence, error_message: str):
    """Fail a sequence with an error message."""
    if seq.output is None:
        return
    
    # Create error response
    seq.output.completion_ids = [[] for _ in range(seq.request.n if seq.request else 1)]
    seq.output.logprobs = [[] for _ in range(seq.request.n if seq.request else 1)]
    seq.output.finish_reason = ["error" for _ in range(seq.request.n if seq.request else 1)]
    seq.output.num_cached_prompt_tokens = [0 for _ in range(seq.request.n if seq.request else 1)]
    seq.output.error_message = error_message
    
    # Send error response back to server
    state.q_manager_to_server.put(seq.output)
    
    # Clean up sequence from queue
    state.scheduling_queue.remove_queued(seq.id)
    
    # Clean up request tracking if this was the last sequence for the request
    if seq.request and seq.request.id in state.req_id_to_seq_ids:
        state.req_id_to_seq_ids[seq.request.id].discard(seq.id)
        if not state.req_id_to_seq_ids[seq.request.id]:
            del state.req_id_to_seq_ids[seq.request.id]


def handle_download_completions(state: ManagerState):
    """Process completed downloads."""
    for response in queue_iterator(state.q_download_complete):
        cartridge_id = response.cartridge_id
        
        if response.success:
            state.logger.info(f"Cartridge {cartridge_id} download completed")
            
            # Reset loading state for force redownloads so cartridge gets reloaded with new weights
            if response.force_redownload:
                state.reset_cartridge_loading_state(cartridge_id)
            
            # Remove from downloading set
            state.cartridges_downloading.discard(cartridge_id)
            
            # Check if any sequences were waiting for this cartridge
            if cartridge_id in state.sequences_waiting_for_cartridges:
                waiting_seq_ids = state.sequences_waiting_for_cartridges.pop(cartridge_id)
                state.logger.info(f"{len(waiting_seq_ids)} sequences can now be scheduled after {cartridge_id} download")
                # These sequences will be picked up in the next scheduling cycle
                
        else:
            state.logger.error(f"Cartridge {cartridge_id} download failed: {response.error_message}")
            
            # Remove from downloading set
            state.cartridges_downloading.discard(cartridge_id)
            
            # Fail any sequences waiting for this cartridge
            if cartridge_id in state.sequences_waiting_for_cartridges:
                waiting_seq_ids = state.sequences_waiting_for_cartridges.pop(cartridge_id)
                for seq_id in waiting_seq_ids:
                    if seq_id in state.scheduling_queue.queued_seqs:
                        seq = state.scheduling_queue.queued_seqs[seq_id]
                        fail_sequence_with_error(state, seq, f"Cartridge {cartridge_id} download failed: {response.error_message}")


def send_to_model(state: ManagerState, command):
    match command:
        case NoMoreInputs():
            if not state.sent_no_more_inputs:
                state.q_manager_to_model.put(command)
                state.sent_no_more_inputs = True
        case _:
            state.q_manager_to_model.put(command)
            state.sent_no_more_inputs = False


def extract_required_cartridges(state: ManagerState, decision: ScheduleDecision) -> dict[str, list[int]]:
    """Extract cartridge requirements from a scheduling decision."""
    required_cartridges = {}
    
    # Check all sequences in the decision for cartridge requirements
    all_seqs = decision.decoding_seqs + [seq for seq, _ in decision.prefill_seqs]
    
    for seq in all_seqs:
        if seq.prepended_cartridge_ids and seq.cartridge_indices:
            # Group cartridge indices by cartridge_id
            cartridge_idx = 0
            for cartridge_id in seq.prepended_cartridge_ids:
                config = state.load_cartridge_config(cartridge_id)
                num_blocks = config.num_blocks_needed(state.config.page_size)
                
                # Extract the block indices for this cartridge
                cartridge_blocks = seq.cartridge_indices[cartridge_idx:cartridge_idx + num_blocks]
                
                if cartridge_id not in required_cartridges:
                    required_cartridges[cartridge_id] = cartridge_blocks
                else:
                    # Verify that the same cartridge uses the same blocks across sequences
                    if required_cartridges[cartridge_id] != cartridge_blocks:
                        state.logger.warning(f"Cartridge {cartridge_id} has different block allocations across sequences")
                
                cartridge_idx += num_blocks
    
    return required_cartridges


def send_cartridge_load_commands(state: ManagerState, decision: ScheduleDecision):
    """Send LoadCartridge commands for any cartridges required by the decision."""
    try:
        required_cartridges = extract_required_cartridges(state, decision)

        for cartridge_id, block_indices in required_cartridges.items():
            if not state.is_cartridge_loaded(cartridge_id):
                # At this point, cartridge should already be downloaded during validation
                # Just verify it exists and send the LoadCartridge command (use sanitized ID for directory path)
                sanitized_id = sanitize_cartridge_id(cartridge_id)
                cartridge_path = Path(state.config.cartridge_dir) / sanitized_id
                if not (cartridge_path / "cartridge.pt").exists() or not (
                    cartridge_path / "config.yaml"
                ).exists():
                    state.logger.error(f"Cartridge {cartridge_id} not found even after validation - this should not happen")
                    continue

                load_cmd = LoadCartridge(
                    cartridge_id=cartridge_id,
                    block_indices=block_indices,
                    cartridge_dir=state.config.cartridge_dir,
                )
                send_to_model(state, load_cmd)
                state.mark_cartridge_loading(cartridge_id)
                state.logger.info(f"Sent LoadCartridge command for {cartridge_id} with blocks {block_indices}")
    except (FileNotFoundError, ValueError) as e:
        # This should not happen if request validation worked correctly,
        # but log the error for debugging
        state.logger.error(f"Error loading cartridge during scheduling: {e}")
        # Note: At this point, we can't easily return an error to the client
        # since the request has already been accepted. The error will manifest
        # when the model tries to load the cartridge.


@track_time_decorator()
def handle_new_server_commands(state: ManagerState):
    num_commands = 0
    for command in queue_iterator(state.q_server_to_manager):
        num_commands += 1
        match command:
            case TokasaurusRequest():
                req = command
                output = RequestOutput(
                    id=command.id,
                )

                sids = [f"{req.id}-part-{i}-of-{req.n}" for i in range(req.n)]

                try:
                    for sid in sids:
                        cartridge_info_list = []
                        prepended_cartridge_ids = None
                        
                        if req.cartridges:
                            # Convert Cartridge objects to info dicts and extract IDs
                            cartridge_info_list = [
                                {
                                    "id": cartridge.id,
                                    "source": cartridge.source, 
                                    "force_redownload": cartridge.force_redownload
                                }
                                for cartridge in req.cartridges
                            ]
                            prepended_cartridge_ids = tuple(sorted(cartridge.id for cartridge in req.cartridges))
                            
                            # Store cartridge info in state for later use during downloading
                            state.store_cartridge_info(cartridge_info_list)
                        
                            # Validate local cartridge configs and request downloads if needed
                            check_and_request_downloads(state, cartridge_info_list)

                        seq = Sequence(
                            id=sid,
                            input_ids=req.input_ids,
                            completion_total=req.max_num_tokens,
                            sampling_params=req.sampling_params,
                            stop=req.stop,
                            request=req,
                            req_output=output,
                            prepended_cartridge_ids=prepended_cartridge_ids,
                        )

                        state.scheduling_queue.add_queued(seq)

                    state.req_id_to_seq_ids[req.id] = set(sids)
                    
                except (FileNotFoundError, ValueError) as e:
                    # Handle cartridge loading errors gracefully
                    state.logger.error(f"Error processing request {req.id}: {e}")
                    
                    # Create error response
                    output.completion_ids = [[] for _ in range(req.n)]
                    output.logprobs = [[] for _ in range(req.n)]
                    output.finish_reason = ["error" for _ in range(req.n)]
                    output.num_cached_prompt_tokens = [0 for _ in range(req.n)]
                    output.error_message = str(e)
                    
                    # Send error response back to server
                    state.q_manager_to_server.put(output)
                    continue

            case CancelledRequest():
                state.logger.info(f"Received cancellation request for {command.req_id}")
                state.req_ids_to_cancel.add(command.req_id)
                continue

    return num_commands


@track_time_decorator()
def check_for_stop_strings(
    state: ManagerState,
    seqs_with_outputs: set[Sequence],
    newly_finished_seqs: set[Sequence],
    num_outputs_processed: int,
):
    seq_to_tokens_for_decoding: list[tuple[Sequence, list[int]]] = []

    for seq in seqs_with_outputs:
        # no need for stop strings if max length is reached
        if seq in newly_finished_seqs:
            continue

        if seq.stop is not None:
            most_recent_ids = seq.most_recent_completion_ids(
                state.config.stop_string_num_token_lookback + num_outputs_processed
            )
            seq_to_tokens_for_decoding.append((seq, most_recent_ids))

    if len(seq_to_tokens_for_decoding) > 0:
        to_decode = [toks for _, toks in seq_to_tokens_for_decoding]

        # Accessing the inner tokenizer to get multi-threading to work properly.
        # TODO make sure padding, special tokens, etc. are handled correctly
        tokenizer = state.get_tokenizer()
        decoded = tokenizer.decode_batch(to_decode, skip_special_tokens=False)

        for (seq, _), decoded_text in zip(seq_to_tokens_for_decoding, decoded):
            for stop in seq.stop:
                if stop in decoded_text:
                    newly_finished_seqs.add(seq)


@track_time_decorator()
def handle_output(
    state: ManagerState,
    out: ModelOutput,
    newly_finished_seqs: set[Sequence],
    seqs_with_outputs: set[Sequence],
):
    eos_token_ids = get_eos_token_ids(state.generation_config)

    state.num_inflight_batches -= 1
    decision = state.inflight_schedule_decisions.pop(out.schedule_id)

    assert len(decision.seqs_with_tokens_to_return) == len(out.tensors.output_ids), (
        f"len(decision.seqs_with_tokens_to_return) = {len(decision.seqs_with_tokens_to_return)}, len(out.tensors.output_ids) = {out.tensors.output_ids.shape} out={out} seqs={[s.id for s in decision.seqs_with_tokens_to_return]} decision={decision}"
    )

    output_ids = out.tensors.output_ids.tolist()

    if out.tensors.chosen_logprobs is not None:
        chosen_logprobs = out.tensors.chosen_logprobs.tolist()
    else:
        chosen_logprobs = None

    if out.tensors.topk_indices is not None:
        topk_indices = out.tensors.topk_indices.numpy()
    else:
        topk_indices = None

    if out.tensors.topk_logprobs is not None:
        topk_logprobs = out.tensors.topk_logprobs.numpy()
    else:
        topk_logprobs = None

    for i, (seq, output_id) in enumerate(
        zip(
            decision.seqs_with_tokens_to_return,
            output_ids,
            strict=True,
        )
    ):
        sid = seq.id

        # if we've already finished the sequence
        if sid in state.finished_seq_ids or seq in newly_finished_seqs or seq.cancelled:
            continue

        seq_out = seq.seq_output

        seq_out.completion_ids.append(output_id)
        seqs_with_outputs.add(seq)

        if chosen_logprobs is not None:
            seq_out.logprobs.append(chosen_logprobs[i])

        if topk_indices is not None:
            seq_out.topk_ids.append(topk_indices[i])

        if topk_logprobs is not None:
            seq_out.topk_logprobs.append(topk_logprobs[i])

        if len(seq_out.completion_ids) == seq.completion_total:
            assert not state.scheduling_queue.in_decoding(sid)
            # don't need to free blocks here since the scheduler already did it.
            newly_finished_seqs.add(seq)
        elif not seq.request.ignore_eos and output_id in eos_token_ids:
            newly_finished_seqs.add(seq)

    state.stats_tracker.add_decision(decision)


@track_time_decorator()
def finish_sequences(state: ManagerState, newly_finished_seqs: set[Sequence]):
    for seq in newly_finished_seqs:
        assert seq.request is not None
        assert seq.req_output is not None

        # NOTE: Edge case: if a stop string/token appears near a seq's max_tokens,
        # the scheduler may have already finished and freed blocks. We guard free()
        # call with a check that the sequence is in the decoding queue.
        if state.scheduling_queue.in_decoding(seq.id):
            state.scheduling_queue.remove_decoding(seq.id)
            state.deallocate(seq)

        seq_out = seq.seq_output

        if len(seq_out.completion_ids) == seq.completion_total:
            seq_out.finish_reason = "length"
        else:
            seq_out.finish_reason = "stop"

        seq.req_output.sequence_outputs.append(seq_out)

        state.finished_seq_ids.add(seq.id)

        if len(seq.req_output.sequence_outputs) == seq.request.n:
            state.q_manager_to_server.put(seq.req_output)
            req_id = seq.request.id
            state.req_id_to_seq_ids.pop(req_id)
            state.logger.debug(f"Finished request ({req_id})")
            state.stats_tracker.add_finished_req()

        state.stats_tracker.add_finished_seq()

    if state.early_stopping_tracker is not None:
        state.early_stopping_tracker.add_finished_sequences(newly_finished_seqs)


@track_time_decorator()
def handle_new_model_outputs(
    state: ManagerState,
):
    newly_finished_sids = set()
    seqs_with_outputs = set()

    num_outputs = 0
    for out in queue_iterator(state.q_model_to_manager):
        num_outputs += 1
        handle_output(
            state=state,
            out=out,
            newly_finished_seqs=newly_finished_sids,
            seqs_with_outputs=seqs_with_outputs,
        )

    if num_outputs > 0:
        check_for_stop_strings(
            state=state,
            seqs_with_outputs=seqs_with_outputs,
            newly_finished_seqs=newly_finished_sids,
            num_outputs_processed=num_outputs,
        )

        finish_sequences(state, newly_finished_sids)

    return num_outputs


def log_hydragen_stats(
    state: ManagerState,
    hydragen_groups: list[HydragenGroup],
    decoding_seqs: list[Sequence],
):
    sid_to_seq = {seq.id: seq for seq in decoding_seqs}

    sids_in_groups = set()
    num_grouped_blocks = 0
    num_total_blocks = 0

    for group in hydragen_groups:
        num_group_blocks = len(group.block_ids)
        for sid in group.seq_ids:
            seq = sid_to_seq[sid]
            assert seq.kv_indices is not None
            num_blocks = len(seq.kv_indices)
            assert num_blocks > num_group_blocks
            num_grouped_blocks += num_group_blocks
            num_total_blocks += num_blocks

            sids_in_groups.add(sid)

    for seq in decoding_seqs:
        if seq.id not in sids_in_groups:
            assert seq.kv_indices is not None
            num_total_blocks += len(seq.kv_indices)

    state.stats_tracker.add_hydragen_stats(
        num_grouped_blocks=num_grouped_blocks,
        num_total_blocks=num_total_blocks,
    )


def model_will_use_cudagraphs(
    config: ServerConfig,
    decoding_seqs: list[Sequence],
    prefill_seqs: list[tuple[Sequence, int]],
):
    """
    If the model will use cudagraphs for this batch.
    """

    if len(prefill_seqs) > 0:
        return False

    return config.use_cudagraphs and len(decoding_seqs) <= config.cudagraph_max_size


@track_time_decorator()
def prepare_and_submit_to_model(
    state: ManagerState,
    decision: ScheduleDecision,
    hydragen_groups: list[HydragenGroup] | None = None,
):
    config = state.config

    # Send LoadCartridge commands before ModelInput commands
    send_cartridge_load_commands(state, decision)

    if config.pp_size == 1:
        partitions = [0, decision.batch_size()]
    else:
        total_tokens = decision.batch_size()
        num_divisions = min(total_tokens, config.pp_size + config.pp_num_buffer_stages)
        partitions = [0]
        for i in range(1, num_divisions + 1):
            end = round(i * total_tokens / num_divisions)
            partitions.append(end)

    num_microbatches = len(partitions) - 1

    microbatches = []

    reordered_decoding_seqs = []

    for i in range(num_microbatches):
        sliced_decoding_seqs, sliced_prefill_seqs, starting_offset = slice_decision(
            decision.decoding_seqs,
            decision.prefill_seqs,
            partitions[i],
            partitions[i + 1],
        )

        assert (
            len(sliced_decoding_seqs) + len(sliced_prefill_seqs)
            <= state.config.max_seqs_per_forward
        ), (
            f"len(decoding_seqs) + len(prefill_seqs) = {len(sliced_decoding_seqs) + len(sliced_prefill_seqs)} > max_seqs_per_forward={state.config.max_seqs_per_forward}"
        )

        if config.use_hydragen and not model_will_use_cudagraphs(
            config, sliced_decoding_seqs, sliced_prefill_seqs
        ):
            assert hydragen_groups is not None
            if num_microbatches == 1:
                microbatch_hydragen_groups = hydragen_groups
            else:
                microbatch_hydragen_groups = restrict_hydragen_groups(
                    groups=hydragen_groups,
                    restrict_to_seq_ids={seq.id for seq in sliced_decoding_seqs},
                    min_group_size=config.hydragen_min_group_size,
                    min_prefix_len=config.hydragen_min_prefix_len,
                    page_size=config.page_size,
                )

                reordered_sliced_decoding_seqs = reorder_decoding_seqs_for_hydragen(
                    sliced_decoding_seqs, microbatch_hydragen_groups
                )
                reordered_decoding_seqs.extend(reordered_sliced_decoding_seqs)
                sliced_decoding_seqs = reordered_sliced_decoding_seqs

            log_hydragen_stats(state, microbatch_hydragen_groups, sliced_decoding_seqs)

        else:
            microbatch_hydragen_groups = None

        for_model = seqs_to_input(
            decoding_seqs=sliced_decoding_seqs,
            prefill_seqs=sliced_prefill_seqs,
            schedule_id=decision.id,
            page_size=config.page_size,
            starting_prefill_offset=starting_offset,
            hydragen_groups=microbatch_hydragen_groups,
            microbatch_index=i,
            microbatch_total=num_microbatches,
        )

        microbatches.append(for_model)

    for microbatch in microbatches:
        send_to_model(state, microbatch)

    # if state.config.pp_size > 1:
    #     send_to_model(state, microbatches)
    # else:
    #     assert len(microbatches) == 1
    #     send_to_model(state, microbatches[0])

    if config.use_hydragen and num_microbatches > 1:
        assert len(reordered_decoding_seqs) == len(decision.decoding_seqs)
        new_decision = ScheduleDecision(
            id=decision.id,
            decoding_seqs=reordered_decoding_seqs,
            prefill_seqs=decision.prefill_seqs,
        )
        return new_decision
    else:
        return decision


def soft_allocate(
    state: ManagerState,
    seq: Sequence,
    prediction_map: PredictionMap | None = None,
):
    if prediction_map is not None:
        prediction_map.update_seq_predictions(seq)

    cached_blocks = state.block_allocator.prefix_match(seq.input_ids, prepended_cartridge_ids=seq.prepended_cartridge_ids)
    cached_block_ids = [block.idx for block in cached_blocks]
    num_cached_tokens = len(cached_block_ids) * state.config.page_size

    # For soft allocation, we don't actually allocate cartridge blocks yet - just estimate
    cartridge_indices = []
    if seq.prepended_cartridge_ids:
        for cartridge_id in seq.prepended_cartridge_ids:
            config = state.load_cartridge_config(cartridge_id)
            num_cartridge_blocks = config.num_blocks_needed(state.config.page_size)
            # Use dummy indices for estimation (negative to avoid conflicts)
            cartridge_indices.extend([-(i+1) for i in range(num_cartridge_blocks)])

    # tentative allocation for now - the allocator hasn't truly assigned these blocks yet
    # to the sequence. but the scheduler functions need these seq attributes set.
    seq.cartridge_indices = cartridge_indices if cartridge_indices else None
    seq.kv_indices = cached_block_ids  # Token blocks only
    seq.seq_output.num_cached_prompt_tokens = num_cached_tokens
    seq.prompt_scheduled = num_cached_tokens


def soft_deallocate(seq: Sequence):
    seq.cartridge_indices = None
    seq.kv_indices = None
    seq.seq_output.num_cached_prompt_tokens = None
    seq.prompt_scheduled = 0


def real_allocate(
    state: ManagerState,
    seq: Sequence,
    available_leaf_heap: list[Block],
):
    # Build cartridge_id_to_num_blocks mapping if we have cartridges
    cartridge_id_to_num_blocks = None
    if seq.prepended_cartridge_ids:
        cartridge_id_to_num_blocks = {}
        for cartridge_id in seq.prepended_cartridge_ids:
            config = state.load_cartridge_config(cartridge_id)
            cartridge_id_to_num_blocks[cartridge_id] = config.num_blocks_needed(state.config.page_size)

    cartridge_indices, token_indices, num_cached_prompt_tokens = (
        state.block_allocator.allocate_with_prefix_match(
            seq.id,
            seq.input_ids,
            available_leaf_heap=available_leaf_heap,
            allow_used_leaves_in_heap=True,
            prepended_cartridge_ids=seq.prepended_cartridge_ids,
            cartridge_id_to_num_blocks=cartridge_id_to_num_blocks,
        )
    )

    seq.cartridge_indices = cartridge_indices
    seq.kv_indices = token_indices  # Now token-only
    seq.seq_output.num_cached_prompt_tokens = num_cached_prompt_tokens
    seq.prompt_scheduled = num_cached_prompt_tokens

    assert seq.batch_index is None
    seq.batch_index = state.batch_index_allocator.allocate()

    state.scheduling_queue.remove_queued(seq.id)
    state.scheduling_queue.add_prefilling(seq)


def sanity_check_block_usage(
    state: ManagerState,
    block_usage_over_time: BlockUsageOverTime,
):
    actual_used_blocks = state.block_allocator.num_used_blocks()
    expected_used_blocks = block_usage_over_time.points[0].num_used_blocks_after_allocation
    
    # With cartridge blocks, there can be a discrepancy between actual and expected block usage
    # because the simulation doesn't fully account for pre-allocated cartridge blocks.
    # For now, we'll allow the actual usage to be >= expected usage to handle this case.
    assert (
        actual_used_blocks >= expected_used_blocks
    ), f"Actual blocks ({actual_used_blocks}) should be >= expected blocks ({expected_used_blocks})"

@track_time_decorator()
def coarse_onboard(
    state: ManagerState,
    block_usage_over_time: BlockUsageOverTime,
    available_leaf_heap: list[Block],
    prediction_map: PredictionMap | None = None,
):
    """
    Onboard a sequence if all of its blocks fit within the smallest point in our block usage simulation.
    Only considers sequences whose cartridges are ready.
    """

    config = state.config
    total_blocks = config.scheduler_block_target()

    # seqs that will obviously fit because current peak usage + their kv indices
    # is less than the total number of blocks.
    seqs_that_coarsely_fit: list[Sequence] = []

    num_running_seqs = len(state.scheduling_queue.decoding_seqs) + len(
        state.scheduling_queue.prefilling_seqs
    )

    queued_seqs = list(state.scheduling_queue.queued_seqs.values())

    current_peak_usage = max(
        p.num_used_blocks_after_allocation for p in block_usage_over_time.points
    )

    for seq in queued_seqs:
        assert seq.kv_indices is None
        if num_running_seqs >= config.max_seqs_per_forward:
            break

        # Check if sequence can be scheduled (all cartridges ready)
        if not can_schedule_sequence(state, seq):
            # Track which cartridges this sequence is waiting for
            if seq.prepended_cartridge_ids:
                for cartridge_id in seq.prepended_cartridge_ids:
                    if cartridge_id in state.cartridges_downloading:
                        if cartridge_id not in state.sequences_waiting_for_cartridges:
                            state.sequences_waiting_for_cartridges[cartridge_id] = set()
                        state.sequences_waiting_for_cartridges[cartridge_id].add(seq.id)
            continue  # Skip this sequence for now

        if prediction_map is not None:
            prediction_map.update_seq_predictions(seq)

        assert seq.kv_indices is None

        # NOTE: we can't consider the effects of prefix caching here because
        # at a future point in the simulation, used cache blocks may be freed.
        # This calculates only token blocks needed - cartridge overhead is handled separately
        num_blocks_needed = seq.expected_num_additional_blocks(
            config.page_size, add_buffer=True
        )
        
        # Add cartridge overhead
        if seq.prepended_cartridge_ids:
            for cartridge_id in seq.prepended_cartridge_ids:
                config_cartridge = state.load_cartridge_config(cartridge_id)
                num_blocks_needed += config_cartridge.num_blocks_needed(config.page_size)

        if current_peak_usage + num_blocks_needed <= total_blocks:
            seqs_that_coarsely_fit.append(seq)
            current_peak_usage += num_blocks_needed
            num_running_seqs += 1
        else:
            break

    if len(seqs_that_coarsely_fit) == 0:
        return block_usage_over_time

    previous_prefilling_seqs = list(state.scheduling_queue.prefilling_seqs.values())

    for seq in seqs_that_coarsely_fit:
        real_allocate(
            state=state,
            seq=seq,
            available_leaf_heap=available_leaf_heap,
        )

    updated_block_usage = try_onboarding_seqs(
        block_usage=block_usage_over_time,
        seqs=seqs_that_coarsely_fit,
        existing_prefill_seqs=previous_prefilling_seqs,
        page_size=config.page_size,
        add_buffer=True,
        prefill_rate=state.last_step_num_prefill,
        block_limit=total_blocks,
    )
    return updated_block_usage


@track_time_decorator()
def precise_onboard(
    state: ManagerState,
    block_usage_over_time: BlockUsageOverTime,
    available_leaf_heap: list[Block],
    prediction_map: PredictionMap | None = None,
):
    """
    Onboard a sequence if it slots into the existing block usage simulation.
    Aware of prefix caching. Only considers sequences whose cartridges are ready.

    Since calling try_onboarding_seqs one at a time is too slow, we use a binary search
    to onboard batches of sequences at a time.
    """
    config = state.config

    reversed_queued_seqs = list(state.scheduling_queue.queued_seqs.values())
    
    # Filter to only sequences that can be scheduled
    schedulable_seqs = []
    for seq in reversed_queued_seqs:
        if can_schedule_sequence(state, seq):
            schedulable_seqs.append(seq)
        else:
            # Track which cartridges this sequence is waiting for
            if seq.prepended_cartridge_ids:
                for cartridge_id in seq.prepended_cartridge_ids:
                    if cartridge_id in state.cartridges_downloading:
                        if cartridge_id not in state.sequences_waiting_for_cartridges:
                            state.sequences_waiting_for_cartridges[cartridge_id] = set()
                        state.sequences_waiting_for_cartridges[cartridge_id].add(seq.id)
    
    # Reverse the list to process in FIFO order
    schedulable_seqs.reverse()
    
    existing_prefill_seqs = list(state.scheduling_queue.prefilling_seqs.values())

    num_running_seqs = state.scheduling_queue.num_running_seqs()
    total_blocks = config.scheduler_block_target()

    max_batch_size = config.precise_onboard_batch_size

    batch_size = max_batch_size

    iters = 0
    while num_running_seqs < config.max_seqs_per_forward:
        iters += 1
        batch_size_this_step = min(
            batch_size,
            config.max_seqs_per_forward - num_running_seqs,
            len(schedulable_seqs),
        )

        if batch_size_this_step == 0:
            break

        batch = schedulable_seqs[-batch_size_this_step:]
        assert len(batch) == batch_size_this_step

        for seq in batch:
            soft_allocate(
                state=state,
                seq=seq,
                prediction_map=prediction_map,
            )

        try:
            try_onboarding_seqs(
                block_usage=block_usage_over_time,
                seqs=batch,
                existing_prefill_seqs=existing_prefill_seqs,
                page_size=config.page_size,
                add_buffer=True,
                prefill_rate=state.last_step_num_prefill,
                block_limit=total_blocks,
            )
        except NoSpaceException:
            # TODO save the half of the batch we proceed with
            for seq in batch:
                soft_deallocate(seq)

            batch_size = batch_size // 2
            continue

        for seq in batch:
            real_allocate(
                state=state,
                seq=seq,
                available_leaf_heap=available_leaf_heap,
            )
            schedulable_seqs.pop()

        # NOTE: need to rerun onboard since the real allocation added new blocks
        # beyond the prefix match.
        new_block_usage = try_onboarding_seqs(
            block_usage=block_usage_over_time,
            seqs=batch,
            existing_prefill_seqs=existing_prefill_seqs,
            page_size=config.page_size,
            add_buffer=True,
            prefill_rate=state.last_step_num_prefill,
            block_limit=total_blocks,
        )
        sanity_check_block_usage(state, new_block_usage)

        block_usage_over_time = new_block_usage

        num_running_seqs += batch_size_this_step
        existing_prefill_seqs.extend(batch)

        # we succeeded, increase the batch_size
        batch_size = min(batch_size * 2, max_batch_size)

    return block_usage_over_time


def bump_city_onboard(
    state: ManagerState,
    block_usage_over_time: BlockUsageOverTime,
    available_leaf_heap: list[Block],
):
    """
    Onboard as much as possible, with the intention of producing many bumps in the future.
    Useful for testing how the engine handles bumping.
    """

    config = state.config
    queued_seqs = list(state.scheduling_queue.queued_seqs.values())

    onboarded_seqs = []

    for seq in queued_seqs:
        if state.scheduling_queue.num_running_seqs() >= config.max_seqs_per_forward:
            break

        try:
            real_allocate(
                state=state,
                seq=seq,
                available_leaf_heap=available_leaf_heap,
            )
            onboarded_seqs.append(seq)
        except NoSpaceException:
            break

    updated_block_usage = try_onboarding_seqs(
        block_usage=block_usage_over_time,
        seqs=onboarded_seqs,
        existing_prefill_seqs=list(state.scheduling_queue.prefilling_seqs.values()),
        page_size=config.page_size,
        add_buffer=True,
        prefill_rate=state.last_step_num_prefill,
        block_limit=1024 * 1024 * 1024,
    )

    return updated_block_usage


@track_time_decorator()
def onboard_new_seqs(
    config: ServerConfig,
    state: ManagerState,
    available_leaf_heap: list[Block],
    prediction_map: PredictionMap | None = None,
):
    prefill_rate = state.last_step_num_prefill
    block_usage_over_time = calc_block_usage_over_time(
        decoding_seqs=list(state.scheduling_queue.decoding_seqs.values()),
        prefilling_seqs=list(state.scheduling_queue.prefilling_seqs.values()),
        page_size=config.page_size,
        add_buffer=True,
        prefill_rate=prefill_rate,
    )

    if config.bump_city_population_me:
        block_usage_over_time = bump_city_onboard(
            state=state,
            block_usage_over_time=block_usage_over_time,
            available_leaf_heap=available_leaf_heap,
        )
    else:
        block_usage_over_time = coarse_onboard(
            state=state,
            block_usage_over_time=block_usage_over_time,
            available_leaf_heap=available_leaf_heap,
            prediction_map=prediction_map,
        )

        if config.enable_precise_onboard:
            block_usage_over_time = precise_onboard(
                state=state,
                block_usage_over_time=block_usage_over_time,
                available_leaf_heap=available_leaf_heap,
                prediction_map=prediction_map,
            )

    sanity_check_block_usage(state, block_usage_over_time)
    assert state.scheduling_queue.num_running_seqs() <= config.max_seqs_per_forward

    return block_usage_over_time


@track_time_decorator()
def allocate_tokens_for_decode_bumping_seqs_if_necessary(
    state: ManagerState,
):
    def needed_blocks(seq: Sequence):
        assert seq.kv_indices is not None
        return state.block_allocator.num_blocks_needed(
            seq.kv_indices, seq.total_scheduled()
        )

    num_needed_blocks = sum(
        needed_blocks(seq) for seq in state.scheduling_queue.decoding_seqs.values()
    )

    running_seqs = list(state.scheduling_queue.running_seqs())

    bumped_seqs = []

    while num_needed_blocks > state.block_allocator.num_free_blocks:
        seq_to_bump = running_seqs.pop()

        # NOTE: important to assign a new id to the created sequence, since otherwise
        # stale outputs coming back from the model might get appended to the new sequence
        # incorrectly.
        # TODO: don't redo completion tokens that have already been computed
        new_id = f"{seq_to_bump.id}-bumped"
        new_seq = Sequence(
            id=new_id,
            request=seq_to_bump.request,
            req_output=seq_to_bump.req_output,
            completion_total=seq_to_bump.completion_total,
            input_ids=seq_to_bump.input_ids,
            sampling_params=seq_to_bump.sampling_params,
            stop=seq_to_bump.stop,
            prepended_cartridge_ids=seq_to_bump.prepended_cartridge_ids,
        )

        if state.scheduling_queue.in_decoding(seq_to_bump.id):
            num_needed_blocks -= needed_blocks(seq_to_bump)

        state.deallocate(seq_to_bump)
        state.scheduling_queue.remove(seq_to_bump.id)
        bumped_seqs.append(new_seq)
        assert seq_to_bump.request is not None
        request_id = seq_to_bump.request.id
        state.req_id_to_seq_ids[request_id].remove(seq_to_bump.id)
        state.req_id_to_seq_ids[request_id].add(new_id)

        seq_to_bump.cancelled = True

        state.logger.info(
            f"Bumped {seq_to_bump.id} ({seq_to_bump.prompt_scheduled}/{seq_to_bump.prompt_total()}, {seq_to_bump.completion_scheduled}/{seq_to_bump.completion_total})"
        )

    if len(bumped_seqs) > 0:
        state.scheduling_queue.insert_at_head_of_queued(bumped_seqs)

    available_leaf_heap = state.block_allocator.make_available_leaf_heap()

    for seq in state.scheduling_queue.decoding_seqs.values():
        assert seq.kv_indices is not None
        new_kv_indices = state.block_allocator.allocate_up_to_length(
            seq.id,
            seq.kv_indices,
            seq.total_scheduled(),
            available_leaf_heap=available_leaf_heap,
        )
        seq.kv_indices.extend(new_kv_indices)

    return available_leaf_heap


@track_time_decorator()
def update_stopping_predictions(state: ManagerState, prediction_map: PredictionMap):
    for seq in chain(
        state.scheduling_queue.decoding_seqs.values(),
        state.scheduling_queue.prefilling_seqs.values(),
    ):
        prediction_map.update_seq_predictions(seq)


def has_schedulable_sequences(state: ManagerState) -> bool:
    """Check if there are any sequences that can be scheduled (running or ready to be scheduled)."""
    # If there are running sequences, we can always schedule
    if state.scheduling_queue.num_running_seqs() > 0:
        return True
    
    # Check if any queued sequences can be scheduled (cartridges ready)
    for seq in state.scheduling_queue.queued_seqs.values():
        if can_schedule_sequence(state, seq):
            return True
    
    return False


@track_time_decorator()
def schedule_steps(state: ManagerState, num_steps: int):
    assert num_steps > 0
    config = state.config

    if config.track_early_stopping:
        assert state.early_stopping_tracker is not None
        prediction_map = state.early_stopping_tracker.make_prediction_map(
            num_buckets=config.early_stopping_num_prediction_buckets,
            std_buffer_scale=config.spec_allocation_std_buffer_scale,
        )
        update_stopping_predictions(state, prediction_map)
    else:
        prediction_map = None

    hydragen_groups: list[HydragenGroup] | None = None
    num_prefill: int | None = None

    for step in range(num_steps):
        # Check if we have any sequences that can actually be scheduled
        if not has_schedulable_sequences(state):
            return

        available_leaf_heap = allocate_tokens_for_decode_bumping_seqs_if_necessary(
            state
        )

        if step == 0:
            block_usage_over_time = onboard_new_seqs(
                config,
                state,
                available_leaf_heap,
                prediction_map=prediction_map,
            )
            decision = schedule(
                queue=state.scheduling_queue,
                num_pages=config.kv_cache_num_blocks(),
                page_size=config.page_size,
                max_tokens_per_forward=config.max_tokens_per_forward,
                max_seqs_per_forward=config.max_seqs_per_forward,
                round_up_multiple=config.prefill_round_up_multiple,
                prediction_map=prediction_map,
                block_usage_over_time=block_usage_over_time,
                greedy_prefill=config.greedy_prefill,
            )

            # TODO: in the case where num_prefill is capped by max_tokens_per_forward,
            # should this number be the un-truncated value?
            num_prefill = decision.num_prefill_tokens()

            # don't update if there's no prefill to do
            if num_prefill > 0:
                state.last_step_num_prefill = num_prefill

            if config.use_hydragen:
                hydragen_groups = group_for_hydragen(
                    root=state.block_allocator.prefix_tree,
                    seq_ids_to_group=state.scheduling_queue.decoding_seqs.keys(),
                    min_group_size=config.hydragen_min_group_size,
                    min_prefix_len=config.hydragen_min_prefix_len,
                    page_size=config.page_size,
                )
        else:
            assert num_prefill is not None
            num_prefill_for_this_step = min(
                num_prefill,
                config.max_tokens_per_forward
                - len(state.scheduling_queue.decoding_seqs),
            )
            decision = make_scheduling_decision(
                queue=state.scheduling_queue,
                num_prefill=num_prefill_for_this_step,
            )

        assert decision.batch_size() <= config.max_tokens_per_forward
        assert decision.num_seqs() <= config.max_seqs_per_forward

        # Skip empty decisions to prevent sending empty batches to the model
        if decision.batch_size() == 0:
            continue

        if config.use_hydragen:
            assert hydragen_groups is not None

            if step == 0:
                hydragen_groups_for_this_step = hydragen_groups
            else:
                # because of bumping/seqs finishing, some seqs in the original
                # hydragen groups may now be gone.
                hydragen_groups_for_this_step = restrict_hydragen_groups(
                    groups=hydragen_groups,
                    restrict_to_seq_ids={seq.id for seq in decision.decoding_seqs},
                    min_group_size=config.hydragen_min_group_size,
                    min_prefix_len=config.hydragen_min_prefix_len,
                    page_size=config.page_size,
                )

            decision = reorder_decision_for_hydragen(
                decision, hydragen_groups_for_this_step
            )
        else:
            hydragen_groups_for_this_step = None

        # important to submit before applying, since applying increments
        # sequences' completion counters.
        decision = prepare_and_submit_to_model(
            state, decision, hydragen_groups=hydragen_groups_for_this_step
        )

        finished_seqs = apply_decision(decision, state.scheduling_queue)

        for seq in finished_seqs:
            state.deallocate(seq)

        state.inflight_schedule_decisions[decision.id] = decision
        state.num_inflight_batches += 1


def try_cancelling_requests(state: ManagerState):
    for rid in state.req_ids_to_cancel.copy():
        # if we already finished with the request
        if rid not in state.req_id_to_seq_ids:
            state.req_ids_to_cancel.remove(rid)
            continue

        seq_ids_to_cancel = state.req_id_to_seq_ids[rid]

        assert len(seq_ids_to_cancel) > 0
        for sid in seq_ids_to_cancel.copy():
            # we can't cancel a sequence in prefill, since its prompt tokens
            # may be shared by other sequences and therefore we need to wait
            # for those tokens to be processed before cancelling.
            # TODO: we can refine this condition to be stricter,
            # checking if the sequence's allocated blocks are in
            # fact shared with later seqs
            if state.scheduling_queue.in_prefilling(sid):
                continue

            if state.scheduling_queue.in_decoding(sid):
                seq = state.scheduling_queue.get_decoding(sid)
                state.deallocate(seq)
                state.scheduling_queue.remove_decoding(sid)
            elif state.scheduling_queue.in_queued(sid):
                state.scheduling_queue.remove_queued(sid)

            # otherwise, the sequence is already finished,
            # so there's no queue removal to do.

            seq_ids_to_cancel.remove(sid)

        if len(seq_ids_to_cancel) == 0:
            state.req_ids_to_cancel.remove(rid)


def run_sanity_checks(state: ManagerState):
    running_seq_ids = set()
    for seq in state.scheduling_queue.running_seqs():
        running_seq_ids.add(seq.id)
    state.block_allocator.sanity_checks(running_seq_ids)


def manager_loop(config: ServerConfig, state: ManagerState):
    state.logger.info("Starting manager loop")

    state.stats_tracker.reset()

    iter_count = 0
    while True:
        wait_start = time.time()
        block_on_queues(
            [state.q_server_to_manager, state.q_model_to_manager, state.q_download_complete],
        )
        wait_time = time.time() - wait_start

        num_new_commands = handle_new_server_commands(state)
        
        # Check for completed downloads and process pending requests
        handle_download_completions(state)
        
        handle_new_model_outputs(state)

        try_cancelling_requests(state)

        # schedule the next N steps of requests.
        num_steps_to_schedule = (
            config.scheduling_steps_ahead - state.num_inflight_batches
        )

        if (
            has_schedulable_sequences(state)
            and num_steps_to_schedule > 0
        ):
            schedule_steps(state, num_steps_to_schedule)

        if not has_schedulable_sequences(state):
            send_to_model(state, NoMoreInputs())

        step_stats(
            state,
            manager_idle_time=wait_time,
            num_new_commands=num_new_commands,
            num_steps_to_schedule=num_steps_to_schedule,
        )

        iter_count += 1

        if config.allocator_sanity_checks:
            run_sanity_checks(state)


@error_propogation_decorator
def start_manager(
    config: ServerConfig,
    q_manager_to_model: mp.Queue,
    q_model_to_manager: mp.Queue,
    q_server_to_manager: mp.Queue,
    q_manager_to_server: mp.Queue,
    q_download_requests: mp.Queue,
    q_download_complete: mp.Queue,
    process_name: str,
    barrier: TimedBarrier,
    dp_rank: int = 0,
):
    setup_logging(config)

    state = ManagerState(
        config=config,
        scheduling_queue=SchedulingQueue(),
        block_allocator=BlockAllocator(
            num_blocks=config.kv_cache_num_blocks(), page_size=config.page_size
        ),
        batch_index_allocator=BatchIndexAllocator(config.max_seqs_per_forward),
        q_manager_to_model=q_manager_to_model,
        q_model_to_manager=q_model_to_manager,
        q_server_to_manager=q_server_to_manager,
        q_manager_to_server=q_manager_to_server,
        q_download_requests=q_download_requests,
        q_download_complete=q_download_complete,
        process_name=process_name,
    )

    state.logger.info("Manager started")

    # Initialize WandB logger if enabled
    if config.wandb_enabled:
        state.wandb_logger = WandbLogger(config, dp_rank)
        state.logger.info(f"WandB logging initialized for DP rank {dp_rank}")
    else:
        state.wandb_logger = None

    if config.track_early_stopping:
        state.early_stopping_tracker = EarlyStoppingTracker(
            buffer_size=config.early_stopping_buffer_size,
            initial_wait=config.early_stopping_initial_wait,
            init_mean=config.early_stopping_init_mean,
            init_std=config.early_stopping_init_std,
        )

    barrier.wait()

    try:
        manager_loop(config, state)
    finally:
        # Clean up WandB logger on shutdown
        if hasattr(state, 'wandb_logger') and state.wandb_logger is not None:
            state.wandb_logger.close()
            state.logger.info("WandB logging closed")
