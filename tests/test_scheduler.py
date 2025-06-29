import random

import pytest

from tokasaurus.manager.allocator import BlockAllocator
from tokasaurus.manager.scheduler import (
    BlockUsageOverTime,
    BlockUsagePoint,
    EventCollection,
    calc_block_usage_over_time,
    try_onboarding_seqs,
)
from tokasaurus.manager.types import Sequence


def make_sequences(
    num_shared_decoding_seqs: int = 128,
    num_unique_decoding_seqs: int = 128,
    num_prefilling_seqs: int = 128,
    shared_length: int = 32,
    max_unique_length: int = 256,
    min_completion_length: int = 1,
    max_completion_length: int = 1024,
    vocab_size: int = 2,
    page_size: int = 2,
    num_blocks: int = 128 * 1024,
    allocator: BlockAllocator | None = None,
    name_prefix: str = "",
):
    if allocator is None:
        allocator = BlockAllocator(page_size=page_size, num_blocks=num_blocks)

    def make_random_ids(length: int):
        return [random.randint(0, vocab_size - 1) for _ in range(length)]

    shared_ids = make_random_ids(shared_length)

    # with only 1 completion token, a seq would never move to
    # decode - it would finish prefill and be done.
    min_decoding_completion_length = max(min_completion_length, 2)

    shared_decoding_seqs = [
        Sequence(
            id=f"{name_prefix}shared-dec-seq{i}",
            completion_total=random.randint(
                min_decoding_completion_length, max_completion_length
            ),
            input_ids=shared_ids
            + make_random_ids(random.randint(1, max_unique_length)),
        )
        for i in range(num_shared_decoding_seqs)
    ]

    unique_decoding_seqs = [
        Sequence(
            id=f"{name_prefix}unique-dec-seq{i}",
            completion_total=random.randint(
                min_decoding_completion_length, max_completion_length
            ),
            input_ids=make_random_ids(random.randint(1, max_unique_length)),
        )
        for i in range(num_unique_decoding_seqs)
    ]

    decoding_seqs = shared_decoding_seqs + unique_decoding_seqs

    prefilling_seqs = [
        Sequence(
            id=f"{name_prefix}prefill-seq{i}",
            completion_total=random.randint(
                min_completion_length, max_completion_length
            ),
            input_ids=make_random_ids(random.randint(1, max_unique_length)),
        )
        for i in range(num_prefilling_seqs)
    ]

    for d in decoding_seqs:
        kvs, num_cached = allocator.allocate_with_prefix_match(d.id, d.input_ids)
        completion_scheduled = random.randint(1, d.completion_total - 1)
        d.prompt_scheduled = len(d.input_ids)
        d.completion_scheduled = completion_scheduled
        d.num_cached_prompt_tokens = num_cached
        allocate_up_to = d.total_scheduled() - 1
        kvs.extend(allocator.allocate_up_to_length(d.id, kvs, allocate_up_to))
        d.kv_indices = kvs
        assert 0 <= page_size * len(kvs) - allocate_up_to < page_size

    for p in prefilling_seqs:
        kvs, num_cached = allocator.allocate_with_prefix_match(p.id, p.input_ids)
        p.prompt_scheduled = num_cached
        p.num_cached_prompt_tokens = num_cached
        p.kv_indices = kvs
        assert 0 <= page_size * len(kvs) - len(p.input_ids) < page_size

    return shared_decoding_seqs, unique_decoding_seqs, prefilling_seqs, allocator


@pytest.mark.parametrize("seed", list(range(10)))
def test_calc_block_usage_over_time(seed):
    random.seed(seed)

    page_size = 2
    prefill_rate = 100
    vocab_size = 2

    shared_decoding_seqs, unique_decoding_seqs, prefilling_seqs, allocator = (
        make_sequences(page_size=page_size, vocab_size=vocab_size)
    )
    decoding_seqs = shared_decoding_seqs + unique_decoding_seqs

    block_usage: BlockUsageOverTime = calc_block_usage_over_time(
        decoding_seqs=decoding_seqs,
        prefilling_seqs=prefilling_seqs,
        page_size=page_size,
        prefill_rate=prefill_rate,
        add_buffer=False,
    )

    # for all timesteps, not just ones where an event happens
    gold_points: list[BlockUsagePoint] = []

    active_decoding_seqs = decoding_seqs.copy()
    active_prefilling_seqs = prefilling_seqs.copy()

    def free_seq(seq: Sequence):
        assert seq.kv_indices is not None
        return allocator.free_and_update(
            seq.id,
            seq.kv_indices,
            seq.input_ids
            + [
                random.randint(0, vocab_size - 1)
                for _ in range(random.randint(0, seq.completion_total))
            ],
        )

    first_used_blocks = set()
    for d in decoding_seqs:
        first_used_blocks.update(d.kv_indices)

    for p in prefilling_seqs:
        first_used_blocks.update(p.kv_indices)

    while len(active_decoding_seqs) + len(active_prefilling_seqs) > 0:
        cur_step = len(gold_points)

        prefill_finishes = []
        decode_finishes = []

        last_page_lens_minus_one = [0] * page_size
        for d in active_decoding_seqs:
            assert d.completion_scheduled < d.completion_total
            d.kv_indices.extend(
                allocator.allocate_up_to_length(d.id, d.kv_indices, d.total_scheduled())
            )
            last_page_len = d.total_scheduled() % page_size
            if last_page_len == 0:
                last_page_len = page_size
            last_page_lens_minus_one[last_page_len - 1] += 1

        used_blocks = set()
        for d in active_decoding_seqs:
            used_blocks.update(d.kv_indices)

        for p in active_prefilling_seqs:
            used_blocks.update(p.kv_indices)

        new_active_decoding_seqs = []
        new_active_prefilling_seqs = []

        freed_blocks = set()

        for d in active_decoding_seqs:
            d.completion_scheduled += 1
            assert d.completion_scheduled <= d.completion_total
            if d.completion_scheduled < d.completion_total:
                new_active_decoding_seqs.append(d)
            else:
                decode_finishes.append(d)
                freed_blocks.update(free_seq(d))

        prefill_available_for_step = prefill_rate
        for p in active_prefilling_seqs:
            new_prompt_scheduled = min(
                p.prompt_scheduled + prefill_available_for_step, len(p.input_ids)
            )
            amount_prefilled = new_prompt_scheduled - p.prompt_scheduled
            prefill_available_for_step -= amount_prefilled
            p.prompt_scheduled = new_prompt_scheduled

            assert prefill_available_for_step >= 0

            if p.prompt_scheduled < len(p.input_ids):
                new_active_prefilling_seqs.append(p)
            else:
                assert p.prompt_scheduled == len(p.input_ids)

                p.completion_scheduled += 1
                if p.completion_scheduled < p.completion_total:
                    new_active_decoding_seqs.append(p)
                    prefill_finishes.append(p)
                else:
                    assert p.completion_total == 1
                    decode_finishes.append(p)
                    freed_blocks.update(free_seq(p))

        point = BlockUsagePoint(
            timestep=cur_step,
            num_used_blocks_after_allocation=len(used_blocks),
            last_page_lens_after_allocation=last_page_lens_minus_one,
            event=EventCollection(
                timestep=cur_step,
                decode_finishes=set(decode_finishes),
                prefill_finishes=set(prefill_finishes),
            ),
            freed_blocks_after_deallocation=freed_blocks,
        )
        gold_points.append(point)

        active_decoding_seqs = new_active_decoding_seqs
        active_prefilling_seqs = new_active_prefilling_seqs

    for point in reversed(block_usage.points):
        gold_point = gold_points[point.timestep]

        assert (
            gold_point.num_used_blocks_after_allocation
            == point.num_used_blocks_after_allocation
        )
        assert (
            gold_point.last_page_lens_after_allocation
            == point.last_page_lens_after_allocation
        )
        assert gold_point.event == point.event
        assert gold_point.freed_blocks_after_deallocation.issuperset(
            point.freed_blocks_after_deallocation
        )

    assert block_usage.used_blocks == first_used_blocks


@pytest.mark.parametrize("seed", list(range(20, 30)))
def test_try_onboarding_seq(seed):
    random.seed(seed)

    page_size = 2
    prefill_rate = 100

    shared_decoding_seqs, unique_decoding_seqs, prefilling_seqs, allocator = (
        make_sequences(page_size=page_size)
    )
    decoding_seqs = shared_decoding_seqs + unique_decoding_seqs

    all_used_blocks = {
        block.idx for block in allocator.all_blocks if len(block.seq_ids) > 0
    }

    (
        _,
        _,
        additional_prefilling_seqs,
        _,
    ) = make_sequences(
        num_shared_decoding_seqs=0,
        num_unique_decoding_seqs=0,
        num_prefilling_seqs=512,
        allocator=allocator,
        name_prefix="more-",
    )

    latest_block_usage: BlockUsageOverTime = calc_block_usage_over_time(
        decoding_seqs=decoding_seqs,
        prefilling_seqs=prefilling_seqs,
        page_size=page_size,
        prefill_rate=prefill_rate,
        add_buffer=False,
    )

    to_onboard = additional_prefilling_seqs.copy()
    cur_used_blocks = all_used_blocks.copy()
    cur_prefilling_seqs = prefilling_seqs.copy()

    iters = 0
    while len(to_onboard) > 0:
        iters += 1
        num_to_onboard = random.randint(1, max(1, len(to_onboard) // 4))
        seqs = to_onboard[:num_to_onboard]
        to_onboard = to_onboard[num_to_onboard:]

        used_by_seqs = set()
        for seq in seqs:
            used_by_seqs.update(seq.kv_indices)

        modified_block_usage = try_onboarding_seqs(
            block_usage=latest_block_usage,
            seqs=seqs,
            existing_prefill_seqs=cur_prefilling_seqs,
            page_size=page_size,
            add_buffer=False,
            prefill_rate=prefill_rate,
            block_limit=float("inf"),
        )

        fresh_block_usage = calc_block_usage_over_time(
            decoding_seqs=decoding_seqs,
            prefilling_seqs=cur_prefilling_seqs + seqs,
            page_size=page_size,
            prefill_rate=prefill_rate,
            add_buffer=False,
        )

        assert fresh_block_usage == modified_block_usage

        cur_used_blocks.update(used_by_seqs)
        cur_prefilling_seqs.extend(seqs)
        latest_block_usage = modified_block_usage
