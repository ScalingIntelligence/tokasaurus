import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from uuid import uuid4

from tokasaurus.manager.allocator import NoSpaceException
from tokasaurus.manager.monitoring import track_time_decorator
from tokasaurus.manager.stopping_predictor import PredictionMap
from tokasaurus.manager.types import (
    ScheduleDecision,
    SchedulingQueue,
    Sequence,
)


@dataclass
class EventCollection:
    timestep: int
    decode_finishes: set[Sequence] = field(default_factory=set)
    prefill_finishes: set[Sequence] = field(default_factory=set)

    def merge(self, other: "EventCollection"):
        assert self.timestep == other.timestep
        return EventCollection(
            timestep=self.timestep,
            decode_finishes=self.decode_finishes | other.decode_finishes,
            prefill_finishes=self.prefill_finishes | other.prefill_finishes,
        )


@dataclass
class BlockUsagePoint:
    timestep: int
    num_used_blocks_after_allocation: int
    last_page_lens_after_allocation: list[int]
    freed_blocks_after_deallocation: set[int]
    event: EventCollection


@dataclass
class BlockUsageOverTime:
    points: list[BlockUsagePoint]
    used_blocks: set[int]


@track_time_decorator()
def calc_block_usage_over_time(
    decoding_seqs: list[Sequence],
    prefilling_seqs: list[Sequence],
    page_size: int,
    add_buffer: bool,
    prefill_rate: int,
    init_num_prefill: int = 0,
):
    """
    For each sequence, calculate how many kv blocks will be used right before
    the seq is deallocated and after all earlier-finishing sequences have been deallocated.

    Assumes that no new seqs are allocated and doesn't consider the deallocation of
    any current prefill seqs.

    Returns results in order of earliest-finishing sequences to latest-finishing sequences
    (this list of sequences is returned too).
    
    Cartridge-aware: tracks which cartridge_ids are active to avoid double-counting
    shared cartridge blocks.
    """

    # events[timestep] = (decode_finishes, prefill_finishes)
    events_builder = defaultdict(lambda: ([], []))

    # create a dummy event for the first step
    events_builder[0]

    # Initialize with all blocks that are already allocated
    # These blocks are already "used" from the beginning
    used_blocks = set()
    for seq in decoding_seqs + prefilling_seqs:
        if seq.kv_indices is not None:
            # Add all allocated blocks to the initial used_blocks set
            used_blocks.update(seq.kv_indices)

    total_prefill = init_num_prefill
    for seq in prefilling_seqs:
        to_schedule = seq.prompt_to_schedule()
        assert to_schedule > 0
        total_prefill += to_schedule

        # -1 since if it fits in the current step, we want that to be step 0
        prefill_finish_step = math.ceil(total_prefill / prefill_rate) - 1

        # -1 since the last prefill batch creates the first decode token
        decode_finish_step = (
            prefill_finish_step
            + seq.expected_completion_length(add_buffer=add_buffer)
            - 1
        )

        events_builder[decode_finish_step][0].append(seq)

        if decode_finish_step == prefill_finish_step:
            assert seq.expected_completion_length() == 1
        else:
            events_builder[prefill_finish_step][1].append(seq)

    for seq in decoding_seqs:
        decode_finish_step = (
            seq.expected_completion_to_schedule(add_buffer=add_buffer) - 1
        )
        events_builder[decode_finish_step][0].append(seq)

    event_timesteps_earliest_to_latest = sorted(events_builder.keys(), reverse=True)
    sorted_event_builder = [
        (timestep, events_builder[timestep])
        for timestep in event_timesteps_earliest_to_latest
    ]
    events_latest_to_earliest = [
        EventCollection(
            timestep=timestep,
            decode_finishes=set(decode_finishes),
            prefill_finishes=set(prefill_finishes),
        )
        for timestep, (decode_finishes, prefill_finishes) in sorted_event_builder
    ]

    # used_blocks is already initialized above with all allocated blocks
    num_used_decode_blocks = 0

    points: list[BlockUsagePoint] = []

    last_page_lens_minus_one = [0] * page_size

    def rollback(num_steps: int):
        nonlocal num_used_decode_blocks
        nonlocal last_page_lens_minus_one

        assert num_steps >= 0

        block_change, last_page_lens_minus_one = simulate_blocks(
            last_page_lens_minus_one, -num_steps
        )
        assert block_change <= 0
        num_used_decode_blocks += block_change

    for event in events_latest_to_earliest:
        timestep = event.timestep

        if len(points) > 0:
            delta = points[-1].timestep - timestep
            assert delta > 0
            rollback(delta)
        else:
            delta = None

        for seq in event.prefill_finishes:
            last_page_len_at_first_decode = (len(seq.input_ids)) % page_size
            if last_page_len_at_first_decode == 0:
                last_page_len_at_first_decode = page_size

            assert last_page_lens_minus_one[last_page_len_at_first_decode - 1] > 0, (
                f"{last_page_len_at_first_decode} {last_page_lens_minus_one} {last_page_lens_minus_one[last_page_len_at_first_decode - 1]}"
            )
            last_page_lens_minus_one[last_page_len_at_first_decode - 1] -= 1

        freed_blocks = set()

        for seq in event.decode_finishes:
            if seq.cartridge_indices:
                # Add cartridge blocks to used_blocks
                freed_blocks.update(seq.cartridge_indices)
                used_blocks.update(seq.cartridge_indices)

            # Handle token blocks (only kv_indices, not cartridge_indices)
            if seq.kv_indices:
                token_indices_set = set(seq.kv_indices)
                delta_token_indices = token_indices_set - used_blocks

                freed_blocks.update(delta_token_indices)
                used_blocks.update(delta_token_indices)

            additional_blocks_needed = seq.expected_num_additional_blocks(
                page_size, add_buffer=add_buffer
            )
            assert additional_blocks_needed >= 0
            num_used_decode_blocks += additional_blocks_needed

            expected_completion_length = seq.expected_completion_length()
            assert expected_completion_length >= 1

            if expected_completion_length > 1:
                last_page_lens_minus_one[
                    seq.expected_last_page_len(page_size, add_buffer=add_buffer) - 1
                ] += 1
            else:
                # If sequence is expected to complete in 1 step, it shouldn't need additional blocks
                # (since cartridge blocks + prompt blocks should be sufficient)
                assert additional_blocks_needed == 0

        points.append(
            BlockUsagePoint(
                timestep=timestep,
                num_used_blocks_after_allocation=len(used_blocks)
                + num_used_decode_blocks,
                last_page_lens_after_allocation=last_page_lens_minus_one,
                freed_blocks_after_deallocation=freed_blocks,
                event=event,
            )
        )

    # return earliest to latest
    return BlockUsageOverTime(
        points=list(reversed(points)),
        used_blocks=used_blocks,
    )


def simulate_blocks(counts_per_last_page_len: list[int], num_steps: int):
    page_size = len(counts_per_last_page_len)
    full_pages = num_steps // page_size

    partial_page = num_steps % page_size

    total_seqs = sum(counts_per_last_page_len)

    split_point = page_size - partial_page
    first_half = counts_per_last_page_len[:split_point]
    second_half = counts_per_last_page_len[split_point:]

    num_blocks = total_seqs * full_pages + sum(second_half)

    rolled_counts = second_half + first_half

    return num_blocks, rolled_counts


def merge_sorted_lists(
    list_a: list[int],
    list_b: list[int],
):
    """
    Merge sorted lists, removing duplicates.
    """
    merged_list = []
    a_index = 0
    b_index = 0

    while a_index < len(list_a) and b_index < len(list_b):
        if list_a[a_index] < list_b[b_index]:
            merged_list.append(list_a[a_index])
            a_index += 1
        elif list_a[a_index] > list_b[b_index]:
            merged_list.append(list_b[b_index])
            b_index += 1
        else:
            merged_list.append(list_a[a_index])
            a_index += 1
            b_index += 1

    merged_list.extend(list_a[a_index:])
    merged_list.extend(list_b[b_index:])

    return merged_list


@track_time_decorator()
def try_merging_usages(
    existing_usage: BlockUsageOverTime,
    new_usage: BlockUsageOverTime,
    block_limit: int,
    page_size: int,
):
    existing_index = 0
    new_index = 0

    existing_timesteps = [p.timestep for p in existing_usage.points]
    new_timesteps = [p.timestep for p in new_usage.points]

    timesteps_to_visit = merge_sorted_lists(existing_timesteps, new_timesteps)

    final_timestep = timesteps_to_visit[-1]
    sentinel_point = BlockUsagePoint(
        timestep=final_timestep + 1,
        num_used_blocks_after_allocation=0,
        last_page_lens_after_allocation=[0] * page_size,
        freed_blocks_after_deallocation=set(),
        event=EventCollection(
            timestep=final_timestep + 1,
        ),
    )

    combined_points = []

    double_counted_blocks = existing_usage.used_blocks & new_usage.used_blocks

    remaining_double_counted_blocks_in_existing = double_counted_blocks.copy()
    remaining_double_counted_blocks_in_new = double_counted_blocks.copy()

    for timestep in timesteps_to_visit:
        # increment to the earliest point that is >= timestep
        while (
            existing_index < len(existing_usage.points)
            and existing_usage.points[existing_index].timestep < timestep
        ):
            remaining_double_counted_blocks_in_existing.difference_update(
                existing_usage.points[existing_index].freed_blocks_after_deallocation
            )

            existing_index += 1

        while (
            new_index < len(new_usage.points)
            and new_usage.points[new_index].timestep < timestep
        ):
            remaining_double_counted_blocks_in_new.difference_update(
                new_usage.points[new_index].freed_blocks_after_deallocation
            )

            new_index += 1

        existing_point = (
            existing_usage.points[existing_index]
            if existing_index < len(existing_usage.points)
            else sentinel_point
        )
        new_point = (
            new_usage.points[new_index]
            if new_index < len(new_usage.points)
            else sentinel_point
        )

        earlier_point, later_point = (
            (
                existing_point,
                new_point,
            )
            if existing_point.timestep < new_point.timestep
            else (new_point, existing_point)
        )

        assert earlier_point.timestep == timestep

        time_delta = later_point.timestep - earlier_point.timestep
        assert time_delta >= 0

        later_last_page_lens = later_point.last_page_lens_after_allocation

        later_block_delta, updated_later_last_page_lens = simulate_blocks(
            counts_per_last_page_len=later_last_page_lens,
            num_steps=-time_delta,
        )
        assert later_block_delta <= 0

        later_num_used_blocks = (
            later_point.num_used_blocks_after_allocation + later_block_delta
        )

        num_double_counted_blocks = len(
            remaining_double_counted_blocks_in_new
            & remaining_double_counted_blocks_in_existing
        )

        combined_num_used_blocks = (
            later_num_used_blocks
            + earlier_point.num_used_blocks_after_allocation
            - num_double_counted_blocks
        )

        if combined_num_used_blocks > block_limit:
            raise NoSpaceException()

        combined_last_page_lens = [
            x + y
            for x, y in zip(
                earlier_point.last_page_lens_after_allocation,
                updated_later_last_page_lens,
                strict=True,
            )
        ]

        if existing_point.timestep == timestep:
            remaining_double_counted_blocks_in_existing.difference_update(
                existing_point.freed_blocks_after_deallocation
            )
            existing_index += 1

        if new_point.timestep == timestep:
            remaining_double_counted_blocks_in_new.difference_update(
                new_point.freed_blocks_after_deallocation
            )
            new_index += 1

        combined_freed_blocks = set()
        if existing_point.timestep == timestep:
            combined_freed_blocks.update(
                existing_point.freed_blocks_after_deallocation
                - remaining_double_counted_blocks_in_new
            )
        if new_point.timestep == timestep:
            combined_freed_blocks.update(
                new_point.freed_blocks_after_deallocation
                - remaining_double_counted_blocks_in_existing
            )

        combined_event = earlier_point.event
        if later_point.timestep == timestep:
            combined_event = combined_event.merge(later_point.event)

        combined_point = BlockUsagePoint(
            timestep=timestep,
            num_used_blocks_after_allocation=combined_num_used_blocks,
            last_page_lens_after_allocation=combined_last_page_lens,
            freed_blocks_after_deallocation=combined_freed_blocks,
            event=combined_event,
        )

        combined_points.append(combined_point)

    return BlockUsageOverTime(
        points=combined_points,
        used_blocks=existing_usage.used_blocks | new_usage.used_blocks,
    )


@track_time_decorator()
def try_onboarding_seqs(
    block_usage: BlockUsageOverTime,
    seqs: list[Sequence],
    existing_prefill_seqs: list[Sequence],
    page_size: int,
    add_buffer: bool,
    prefill_rate: int,
    block_limit: int,
):
    new_block_usage = calc_block_usage_over_time(
        decoding_seqs=[],
        prefilling_seqs=seqs,
        page_size=page_size,
        add_buffer=add_buffer,
        prefill_rate=prefill_rate,
        init_num_prefill=sum([p.prompt_to_schedule() for p in existing_prefill_seqs]),
    )

    return try_merging_usages(
        existing_usage=block_usage,
        new_usage=new_block_usage,
        block_limit=block_limit,
        page_size=page_size,
    )


def calc_prefill_per_forward(
    queue: SchedulingQueue,
    block_usage_over_time: BlockUsageOverTime,
    total_blocks: int,
    page_size: int,
    max_seqs_per_forward: int,
    prediction_map: PredictionMap | None = None,
):
    """
    Returning inf = we can fit all sequences (including queued ones) in the KV cache,
    so prefill as fast as possible so we can run them all immediately.
    """
    prefilling_seqs = list(queue.prefilling_seqs.values())
    decoding_seqs = list(queue.decoding_seqs.values())
    queued_seqs = deque(queue.queued_seqs.values())

    num_running_seqs = len(prefilling_seqs) + len(decoding_seqs)
    assert num_running_seqs <= max_seqs_per_forward, (
        f"num_running_seqs={num_running_seqs} > max_seqs_per_forward={max_seqs_per_forward}"
    )

    assert (
        len(prefilling_seqs) > 0 and len(decoding_seqs) > 0 and len(queued_seqs) > 0
    ), (
        "calculation is meaningful only if there are prefilling, decoding, and queued seqs"
    )

    # in our algorithm, we loop through each decoding sequence and simulate what happens if all sequences
    # "before" that one have finished decoding
    # - we mimic "onboarding" as many seqs as we can
    # - the tokens we want to prefill per forward = (total prefill tokens avail) / num decode steps
    # - this isn't optimal, but seems to work

    # for efficiency, counters storing how prefill tokens are available and how much memory is used
    # from simulating onboarding
    simulated_prefill_tokens: int = 0
    simulated_added_blocks: int = 0

    simulated_num_running_seqs = num_running_seqs

    total_current_tokens_of_prefill = sum(
        seq.prompt_to_schedule() for seq in prefilling_seqs
    )

    min_ppf = float("inf")

    predicted_queued_seqs = set[Sequence]()

    for point in block_usage_over_time.points:
        if len(point.event.decode_finishes) == 0:
            continue

        timestep = point.timestep
        num_used_blocks = point.num_used_blocks_after_allocation

        # simulate that all seqs before this timestep have finished and been freed
        while (
            len(queued_seqs) > 0 and simulated_num_running_seqs < max_seqs_per_forward
        ):
            # checking if the head of queued_seqs could be onboarded
            # there should be this much memory
            expected_free_blocks = (
                total_blocks - num_used_blocks - simulated_added_blocks
            )

            incoming_seq = queued_seqs[0]

            # we lazily predict completion length here to avoid having to do it for all queued seqs
            # every iteration (we may have a really deep stack of queued seqs)
            if prediction_map is not None and incoming_seq not in predicted_queued_seqs:
                prediction_map.update_seq_predictions(incoming_seq)
                predicted_queued_seqs.add(incoming_seq)

            # potential overestimate of number of blocks: no simulation of prefix caching
            # This calculates only token blocks - add cartridge overhead
            incoming_blocks_needed = incoming_seq.expected_num_additional_blocks(
                page_size
            )
            
            # Add cartridge overhead (these blocks will be shared, so this is a conservative estimate)
            if incoming_seq.prepended_cartridge_ids:
                # NOTE: This is a rough estimate - actual implementation would track which cartridges
                # are already allocated. For now, we assume worst-case where we need all cartridge blocks.
                incoming_blocks_needed += incoming_seq.get_num_cartridge_blocks()

            if incoming_blocks_needed <= expected_free_blocks:
                queued_seqs.popleft()

                simulated_prefill_tokens += incoming_seq.prompt_total()
                simulated_added_blocks += incoming_blocks_needed
                simulated_num_running_seqs += 1
            else:
                # onboarded sequences FIFO, so if the head can't be onboarded, stop
                break

        if len(queued_seqs) == 0:
            # if we run out of queued seqs, this means that by this point in the future,
            # we can fit every outstanding sequence in memory, so prefill per forward
            # should be infinite here (to onboard all the stuff we can now fit asap).
            # since this will also be true for all future iterations, we can break out
            # of the loop, since none of these infs will affect min_ppf
            break
        else:
            total_prefill_available = (
                total_current_tokens_of_prefill + simulated_prefill_tokens
            )
            prefill_per_forward = total_prefill_available / (timestep + 1)

        min_ppf = min(min_ppf, prefill_per_forward)
        simulated_num_running_seqs -= 1

    return min_ppf


def make_scheduling_decision(
    queue: SchedulingQueue,
    num_prefill: float,
):
    schedule_id = str(uuid4())

    # allocate the prefill tokens
    prefill_seqs: list[tuple[Sequence, int]] = []
    amount_remaining_to_prefill = num_prefill
    for s in queue.prefilling_seqs.values():
        prompt_to_schedule = s.prompt_to_schedule()
        assert prompt_to_schedule > 0

        if amount_remaining_to_prefill == 0:
            break

        amount_to_prefill_for_this_seq = min(
            prompt_to_schedule, amount_remaining_to_prefill
        )

        assert amount_to_prefill_for_this_seq > 0, amount_to_prefill_for_this_seq
        assert amount_to_prefill_for_this_seq % 1 == 0

        amount_to_prefill_for_this_seq = int(amount_to_prefill_for_this_seq)

        prefill_seqs.append((s, amount_to_prefill_for_this_seq))
        amount_remaining_to_prefill -= amount_to_prefill_for_this_seq

    return ScheduleDecision(
        id=schedule_id,
        decoding_seqs=list(queue.decoding_seqs.values()),
        prefill_seqs=prefill_seqs,
    )


def schedule(
    queue: SchedulingQueue,
    block_usage_over_time: BlockUsageOverTime,
    num_pages: int,
    page_size: int,
    max_seqs_per_forward: int,
    max_tokens_per_forward: int,
    round_up_multiple: int = 1,
    prediction_map: PredictionMap | None = None,
    greedy_prefill: bool = False,
) -> ScheduleDecision:
    """
    Bro trust me, it's a good scheduler. I sketched it out on my iPad bro.
    Graphs and everything. This scheduler slaps.

    Assumptions made here:
    - Once a request is in decode mode, it stays in decode mode until completion.
    - Queued requests are served FIFO.
    - We start processing when have enough tokens for the request's prefill and decode (in expectation)
    """
    assert queue.num_unfinished_seqs() > 0, "Nothing to schedule"

    max_prefill = max_tokens_per_forward - len(queue.decoding_seqs)
    assert max_prefill >= 0

    def make_decision(num_prefill: float):
        return make_scheduling_decision(queue, min(num_prefill, max_prefill))

    for s in queue.decoding_seqs.values():
        assert s.completion_scheduled < s.completion_total

    # if there's nothing to prefill (all requests we can fit are in decode mode), nothing fancy to do
    if len(queue.prefilling_seqs) == 0:
        return make_decision(0)

    # if queue is empty (i.e. everything fits) or we're in greedy mode, max prefill
    if len(queue.queued_seqs) == 0 or greedy_prefill:
        # finish prefill of all mid-prefill + queued requests, move to decoding ASAP
        return make_decision(float("inf"))

    # if there is nothing to decode, prefill the entire first request ASAP
    if len(queue.decoding_seqs) == 0:
        seq_to_prefill = next(iter(queue.prefilling_seqs.values()))
        return make_decision(seq_to_prefill.prompt_to_schedule())

    chosen_ppf = calc_prefill_per_forward(
        queue=queue,
        block_usage_over_time=block_usage_over_time,
        total_blocks=num_pages,
        page_size=page_size,
        max_seqs_per_forward=max_seqs_per_forward,
        prediction_map=prediction_map,
    )

    if chosen_ppf < float("inf"):
        rounded_chosen_ppf = int(
            chosen_ppf + (round_up_multiple - (chosen_ppf % round_up_multiple))
        )
    else:
        rounded_chosen_ppf = float("inf")

    return make_decision(rounded_chosen_ppf)


def apply_decision(
    schedule_decision: ScheduleDecision,
    sched_queue: SchedulingQueue,
):
    finished_seqs = []

    # NOTE: important to do prefill first, since when
    # a sequence has only one token to decode, we may move it
    # from prefill to decode, and then immediately move it
    # to finished

    for seq in schedule_decision.decoding_seqs:
        sid = seq.id
        seq.completion_scheduled += 1
        assert seq.completion_scheduled <= seq.completion_total

        if seq.completion_scheduled == seq.completion_total:
            sched_queue.remove_decoding(sid)
            finished_seqs.append(seq)

    for seq, slen in schedule_decision.prefill_seqs:
        sid = seq.id

        if sched_queue.in_queued(sid):
            sched_queue.remove_queued(sid)
            sched_queue.add_prefilling(seq)
        else:
            assert sched_queue.in_prefilling(sid), (
                f"Seq {sid} not in prefilling, {sched_queue}"
            )

        seq.prompt_scheduled += slen
        assert seq.prompt_scheduled <= seq.prompt_total()

        if seq.prompt_scheduled == seq.prompt_total():
            seq.completion_scheduled += 1
            sched_queue.remove_prefilling(sid)

            assert seq.completion_scheduled <= seq.completion_total
            if seq.completion_scheduled == seq.completion_total:
                finished_seqs.append(seq)
            else:
                sched_queue.add_decoding(seq)

    return finished_seqs
