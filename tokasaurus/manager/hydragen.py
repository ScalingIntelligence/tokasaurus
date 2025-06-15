from dataclasses import dataclass
from typing import Iterable

from tokasaurus.manager.allocator import Block
from tokasaurus.manager.monitoring import track_time_decorator
from tokasaurus.manager.types import HydragenGroup, ScheduleDecision, Sequence


@track_time_decorator()
def reorder_decoding_seqs_for_hydragen(
    decoding_seqs: list[Sequence], hydragen_groups: list[HydragenGroup]
):
    """
    Our Hydragen implementation requires us to reorder the decoding sequences so that
    seqs in the same shared-prefix group are adjacent to each other in the batch.
    """

    sid_to_decoding_seq = {seq.id: seq for seq in decoding_seqs}

    # we re-order the decoding sequences so that seqs in the same hydragen group
    # are adjacent to each other, with ungrouped seqs at the end.
    reordered_decoding_seqs = []

    grouped_sids = set[str]()
    sid_to_group = dict[str, HydragenGroup]()
    for group in hydragen_groups:
        for sid in group.seq_ids:
            sid_to_group[sid] = group
            grouped_sids.add(sid)
            reordered_decoding_seqs.append(sid_to_decoding_seq[sid])

    # ungrouped seqs at the end
    for seq in decoding_seqs:
        if seq.id not in grouped_sids:
            reordered_decoding_seqs.append(seq)

    assert len(reordered_decoding_seqs) == len(decoding_seqs)

    return reordered_decoding_seqs


def reorder_decision_for_hydragen(
    decision: ScheduleDecision, groups: list[HydragenGroup]
):
    return ScheduleDecision(
        id=decision.id,
        decoding_seqs=reorder_decoding_seqs_for_hydragen(
            decision.decoding_seqs, groups
        ),
        prefill_seqs=decision.prefill_seqs,
    )


def node_to_block_ids(node: Block) -> list[int]:
    block_ids_last_to_first = []
    cur = node
    while not cur.is_root():
        block_ids_last_to_first.append(cur.idx)
        cur = cur.parent
        assert cur is not None

    return list(reversed(block_ids_last_to_first))


@track_time_decorator()
def group_for_hydragen(
    root: Block,
    seq_ids_to_group: Iterable[str],
    min_group_size: int,
    min_prefix_len: int,
    page_size: int,
) -> list[HydragenGroup]:
    """
    Iterative version of depth-first search - we make a group for a given prefix if
    it meets the minimum group size/prefix length requirements,
    after checking if any children have met these requirements.
    """
    groups = list[HydragenGroup]()
    all_sids = set(seq_ids_to_group)
    grouped_sids = set[str]()

    @dataclass
    class StackItem:
        node: Block
        depth: int
        visited_children: bool
        potential_sids: set[str]

    # Stack will contain tuples of (node, block_ids_before_this_node, visited_children)
    # visited_children is a boolean indicating whether we've already processed the children
    stack: list[StackItem] = []

    assert min_prefix_len % page_size == 0
    min_depth = min_prefix_len // page_size

    # Initialize the stack with the root's children
    for child in root.children.values():
        stack.append(StackItem(child, 1, False, all_sids & child.seq_ids))

    while stack:
        item = stack.pop()

        if not item.visited_children:
            # Skip this node if it doesn't have enough sequence IDs
            if len(item.potential_sids) < min_group_size:
                continue

            if item.depth >= min_depth:
                # If there's a chance this node is the last block in a group,
                # push it back on the stack to re-consider it after we process its children.
                stack.append(
                    StackItem(item.node, item.depth, True, item.potential_sids)
                )

            # Process children
            for child in item.node.children.values():
                stack.append(
                    StackItem(
                        child,
                        item.depth + 1,
                        False,
                        item.potential_sids & child.seq_ids,
                    )
                )
        else:
            # We need to compute the available ids after considering the children
            # since the children may have created groups.
            available_sids_to_group = item.potential_sids - grouped_sids
            if len(available_sids_to_group) >= min_group_size:
                groups.append(
                    HydragenGroup(
                        block_ids=node_to_block_ids(item.node),
                        seq_ids=available_sids_to_group,
                    )
                )
                grouped_sids.update(available_sids_to_group)

    sorted_groups = list(sorted(groups, key=lambda x: x.block_ids))
    return sorted_groups


def restrict_hydragen_groups(
    groups: list[HydragenGroup],
    restrict_to_seq_ids: set[str],
    min_group_size: int,
    min_prefix_len: int,
    page_size: int,
) -> list[HydragenGroup]:
    """
    Restricts the groups to only include seqs in restrict_to_seq_ids, and ensures
    that the groups meet the minimum group size/prefix length requirements.
    """

    assert min_prefix_len % page_size == 0
    min_prefix_len_in_pages = min_prefix_len // page_size

    now_too_small_groups = list[HydragenGroup]()
    good_groups = list[HydragenGroup]()

    for group in groups:
        restricted_group_seq_ids = group.seq_ids & restrict_to_seq_ids
        if len(restricted_group_seq_ids) >= min_group_size:
            good_groups.append(
                HydragenGroup(
                    block_ids=group.block_ids,
                    seq_ids=restricted_group_seq_ids,
                )
            )
        else:
            now_too_small_groups.append(group)

    if len(now_too_small_groups) > 0:
        # best-effort approach to merge groups that are now too small
        too_small_prefixes = [group.block_ids for group in now_too_small_groups]

        # find longest common prefix of all too_small_prefixes
        common_prefix = []
        min_of_too_small_prefix_lens = min(len(p) for p in too_small_prefixes)
        for i in range(min_of_too_small_prefix_lens):
            potential_block_id = too_small_prefixes[0][i]
            if all(p[i] == potential_block_id for p in too_small_prefixes):
                common_prefix.append(potential_block_id)
            else:
                break

        if len(common_prefix) >= min_prefix_len_in_pages:
            unrestricted_merged_group_seq_ids = set[str]()
            for group in now_too_small_groups:
                unrestricted_merged_group_seq_ids.update(group.seq_ids)

            merged_group_seq_ids = (
                unrestricted_merged_group_seq_ids & restrict_to_seq_ids
            )

            if len(merged_group_seq_ids) >= min_group_size:
                # merge the groups
                merged_group = HydragenGroup(
                    block_ids=common_prefix,
                    seq_ids=merged_group_seq_ids,
                )
                good_groups.append(merged_group)

    return good_groups
