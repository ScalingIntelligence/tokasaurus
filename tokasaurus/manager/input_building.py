import math

from tokasaurus.common_types import ServerConfig
from tokasaurus.manager.monitoring import track_time_decorator
from tokasaurus.manager.types import HydragenGroup, Sequence
from tokasaurus.model.types import (
    AttentionInfoBuilder,
    BatchSamplingParamsBuilder,
    ModelInput,
    PageInformationBuilder,
)


def make_dummy_batch(
    config: ServerConfig,
    prefill_tokens: int,
    decode_tokens: int,
    prefill_uses_lm_head: bool = False,
    skip_pipeline_communication: bool = False,
):
    total_tokens = prefill_tokens + decode_tokens
    page_size = config.page_size

    append_kv_token_indices = []

    prefill_builder = PageInformationBuilder()
    decode_builder = PageInformationBuilder()
    sampling_builder = BatchSamplingParamsBuilder()

    if prefill_tokens > 0:
        prefill_kv_indices = list(range(math.ceil(prefill_tokens / page_size)))

        prefill_builder.add_sequence(
            kv_indices=prefill_kv_indices,
            kv_seq_len=prefill_tokens,
            num_qtokens=prefill_tokens,
            page_size=page_size,
        )
        append_kv_token_indices.extend(
            calc_kv_token_indices(
                kv_block_indices=prefill_kv_indices,
                page_size=page_size,
                start_idx=0,
                num_tokens=prefill_tokens,
            )
        )

        if prefill_uses_lm_head:
            sampling_builder.add_sequence(
                temperature=0.5,
                top_p=1.0,
            )

    for _ in range(decode_tokens):
        decode_builder.add_sequence(
            kv_indices=[0],
            kv_seq_len=1,
            num_qtokens=1,
            page_size=page_size,
        )
        append_kv_token_indices.extend(
            calc_kv_token_indices(
                kv_block_indices=[0],
                page_size=page_size,
                start_idx=0,
                num_tokens=1,
            )
        )
        sampling_builder.add_sequence(
            temperature=0.5,
            top_p=1.0,
        )

    attention_info_builder = AttentionInfoBuilder(
        page_size=page_size,
        append_kv_token_indices=append_kv_token_indices,
        prefill_builder=prefill_builder,
        decode_builder=decode_builder,
        hydragen_builder=None,
    )

    # includes prefill lm head tokens and decode tokens
    if prefill_uses_lm_head:
        assert prefill_tokens > 0
        lm_head_indices = list(range(prefill_tokens - 1, total_tokens))
    else:
        lm_head_indices = list(range(prefill_tokens, total_tokens))

    inp = ModelInput(
        attention_info_builder=attention_info_builder,
        prefill_input_ids=[0] * prefill_tokens,
        batch_indices=[0] * total_tokens,
        lm_head_indices=lm_head_indices,
        sampling_builder=sampling_builder,
        position_ids=[0] * total_tokens,
        schedule_id="dummy_batch",
        skip_pipeline_communication=skip_pipeline_communication,
    )

    return inp


def slice_decision(
    decoding_seqs: list[Sequence],
    prefill_seqs: list[tuple[Sequence, int]],
    start_idx: int,
    end_idx: int,
):
    sliced_prefill_seqs: list[tuple[Sequence, int]] = []
    cumsum_start = 0
    starting_offset = None
    for seq, prefill_len in prefill_seqs:
        seq_tok_start = cumsum_start
        seq_tok_end = seq_tok_start + prefill_len

        if start_idx < seq_tok_end and end_idx > seq_tok_start:
            min_end = min(seq_tok_end, end_idx)
            max_start = max(seq_tok_start, start_idx)

            sliced_prefill_len = min_end - max_start
            sliced_prefill_seqs.append((seq, sliced_prefill_len))

            if start_idx >= seq_tok_start:
                assert starting_offset is None
                starting_offset = start_idx - seq_tok_start
            else:
                assert max_start == seq_tok_start

        cumsum_start = seq_tok_end

    decoding_start = max(0, start_idx - cumsum_start)
    decoding_end = max(0, end_idx - cumsum_start)
    sliced_decoding_seqs = decoding_seqs[decoding_start:decoding_end]

    return sliced_decoding_seqs, sliced_prefill_seqs, starting_offset


def calc_kv_token_indices(
    kv_block_indices: list[int], page_size: int, start_idx: int, num_tokens: int
):
    kv_token_indices = []
    for pos in range(start_idx, start_idx + num_tokens):
        block_idx = pos // page_size
        kv_token_indices.append(
            kv_block_indices[block_idx] * page_size + pos % page_size
        )

    return kv_token_indices


@track_time_decorator()
def seqs_to_input(
    decoding_seqs: list[Sequence],
    prefill_seqs: list[tuple[Sequence, int]],
    schedule_id: str,
    page_size: int,
    starting_prefill_offset: int | None = None,
    hydragen_groups: list[HydragenGroup] | None = None,
    microbatch_index: int = 0,
    microbatch_total: int = 1,
):
    use_hydragen = hydragen_groups is not None

    position_ids = []
    lm_head_indices = []

    append_kv_token_indices = []

    prefill_builder = PageInformationBuilder()
    decode_builder = PageInformationBuilder()

    sampling_builder = BatchSamplingParamsBuilder()

    prefill_input_ids_list = []

    for i, (seq, slen) in enumerate(prefill_seqs):
        assert seq.completion_scheduled == 0
        assert seq.kv_indices is not None

        start_position = seq.prompt_scheduled
        if i == 0:
            assert starting_prefill_offset is not None
            start_position += starting_prefill_offset
        end_position = start_position + slen

        prefill_ids = seq.input_ids[start_position:end_position]
        assert len(prefill_ids) == slen

        prefill_input_ids_list.extend(prefill_ids)

        seq_pos_ids = list(range(start_position, end_position))
        position_ids.extend(seq_pos_ids)

        prefill_builder.add_sequence(
            kv_indices=seq.kv_indices,
            kv_seq_len=start_position + slen,
            num_qtokens=slen,
            page_size=page_size,
        )

        append_kv_token_indices.extend(
            calc_kv_token_indices(
                kv_block_indices=seq.kv_indices,
                page_size=page_size,
                start_idx=start_position,
                num_tokens=slen,
            )
        )

        if end_position == seq.prompt_total():
            lm_head_indices.append(len(position_ids) - 1)

            sparams = seq.sampling_params
            sampling_builder.add_sequence(
                temperature=sparams.temperature,
                top_p=sparams.top_p,
            )

    if use_hydragen:
        hydragen_builder = PageInformationBuilder()

        sid_to_pos = {seq.id: i for i, seq in enumerate(decoding_seqs)}
        sid_to_group: dict[str, HydragenGroup] = {}

        seqs_processed = 0

        for group in hydragen_groups:
            hydragen_builder.add_sequence(
                kv_indices=group.block_ids,
                kv_seq_len=len(group.block_ids) * page_size,
                num_qtokens=len(group.seq_ids),
                page_size=page_size,
            )
            group_positions = {sid_to_pos[sid] for sid in group.seq_ids}
            assert group_positions == set(
                range(seqs_processed, seqs_processed + len(group.seq_ids))
            ), "decoding seqs must be ordered by hydragen group"

            seqs_processed += len(group.seq_ids)

            for sid in group.seq_ids:
                sid_to_group[sid] = group

    for seq in decoding_seqs:
        # NOTE: minus one since last prefill token produces first
        # decode token.
        current_token_pos_id = seq.total_scheduled() - 1
        position_ids.append(current_token_pos_id)

        if use_hydragen and seq.id in sid_to_group:
            group = sid_to_group[seq.id]
            starting_block = len(group.block_ids)
        else:
            starting_block = 0

        assert seq.kv_indices is not None
        decode_builder.add_sequence(
            kv_indices=seq.kv_indices,
            kv_seq_len=current_token_pos_id + 1,
            num_qtokens=1,
            page_size=page_size,
            starting_block=starting_block,
        )
        # starting block of 0 needed for append
        append_kv_token_indices.extend(
            calc_kv_token_indices(
                kv_block_indices=seq.kv_indices,
                page_size=page_size,
                start_idx=current_token_pos_id,
                num_tokens=1,
            )
        )

        sparams = seq.sampling_params
        sampling_builder.add_sequence(
            temperature=sparams.temperature,
            top_p=sparams.top_p,
        )

    prefill_lengths = [slen for _, slen in prefill_seqs]
    start_of_decode = sum(prefill_lengths)

    lm_head_indices.extend(
        list(range(start_of_decode, start_of_decode + len(decoding_seqs)))
    )

    batch_indices = []

    def register_batch_index(seq: Sequence, num_tokens: int):
        assert seq.batch_index is not None
        batch_indices.extend([seq.batch_index] * num_tokens)

    for seq, num_tokens in prefill_seqs:
        register_batch_index(seq, num_tokens)

    for seq in decoding_seqs:
        register_batch_index(seq, 1)

    # No need to call build() on builders anymore, just pass them directly
    attention_info_builder = AttentionInfoBuilder(
        page_size=page_size,
        append_kv_token_indices=append_kv_token_indices,
        prefill_builder=prefill_builder,
        decode_builder=decode_builder,
        hydragen_builder=hydragen_builder if use_hydragen else None,
    )

    inp = ModelInput(
        attention_info_builder=attention_info_builder,
        sampling_builder=sampling_builder,
        prefill_input_ids=prefill_input_ids_list,
        batch_indices=batch_indices,
        lm_head_indices=lm_head_indices,
        position_ids=position_ids,
        schedule_id=schedule_id,
        microbatch_index=microbatch_index,
        microbatch_total=microbatch_total,
    )

    return inp
