import heapq
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PrefixTreeBlock:
    idx: int
    contents: tuple[int, ...] = field(default_factory=tuple)
    last_used_at: float = 0.0
    seq_ids: set[str] = field(default_factory=set)
    children: dict[tuple[int, ...], "PrefixTreeBlock"] = field(default_factory=dict)
    parent: Optional["PrefixTreeBlock"] = None
    is_cpu: bool = False

    def detach_from_parent(self):
        if self.parent is not None:
            assert self.parent.children[self.contents] == self
            self.parent.children.pop(self.contents)
        self.parent = None

    def attach_to_parent(self, parent: "PrefixTreeBlock"):
        assert self.parent is None
        assert self.contents not in parent.children
        self.parent = parent
        self.parent.children[self.contents] = self

    def wipe(self):
        self.detach_from_parent()
        self.children.clear()
        self.seq_ids.clear()
        self.contents = tuple()
        self.last_used_at = 0.0

    def is_wiped(self):
        return (
            len(self.seq_ids) == 0
            and len(self.children) == 0
            and self.parent is None
            and self.contents == tuple()
        )

    def __repr__(self):
        return f"Block(idx={self.idx}, is_cpu={self.is_cpu}, contents={self.contents}, children={len(self.children)} used_by={len(self.seq_ids)})"

    def __hash__(self):
        return self.idx

    def __lt__(self, other: "PrefixTreeBlock"):
        return self.last_used_at < other.last_used_at

    def tree_repr(self):
        def indent(s: str, spacing: int):
            lines = s.split("\n")
            with_index = [" " * spacing + line for line in lines]
            return "\n".join(with_index)

        out_lines = [repr(self)]

        for child in self.children.values():
            out_lines.append(indent(child.tree_repr(), 2))

        return "\n".join(out_lines)


@dataclass
class SwapIndices:
    gpu_ids: list[int] = field(default_factory=list)
    cpu_ids: list[int] = field(default_factory=list)

    def __post_init__(self):
        assert len(self.gpu_ids) == len(self.cpu_ids)


@dataclass
class SwapInfo:
    cpu_to_gpu: SwapIndices | None = None
    gpu_to_cpu: SwapIndices | None = None


@dataclass
class AllocationResult:
    new_kv_indices: list[int]
    swap_info: SwapInfo = field(default_factory=SwapInfo)


@dataclass
class PrefixCacheAllocationResult:
    new_kv_indices: list[int]
    num_cached_prompt_tokens: int
    swap_info: SwapInfo = field(default_factory=SwapInfo)


class NoSpaceException(ValueError):
    """Exception raised when there is insufficient space for allocating a block."""


def truncate_to_multiple(x: list, multiple: int):
    return x[: len(x) - (len(x) % multiple)]


def pick_blocks_for_allocation(
    num_blocks: int,
    available_floating: set[PrefixTreeBlock],
    available_leaves: set[PrefixTreeBlock],
    is_cpu: bool,
):
    chosen_floating_blocks: list[PrefixTreeBlock] = []
    chosen_tree_blocks: list[PrefixTreeBlock] = []

    for block in available_floating:
        if len(chosen_floating_blocks) == num_blocks:
            return chosen_floating_blocks, chosen_tree_blocks

        chosen_floating_blocks.append(block)

    leaf_heap = list(available_leaves)
    heapq.heapify(leaf_heap)

    num_remaining = num_blocks - len(chosen_floating_blocks)
    tree_block_ids = set()

    for _ in range(num_remaining):
        leaf = heapq.heappop(leaf_heap)
        assert len(leaf.seq_ids) == 0

        parent = leaf.parent
        assert parent is not None

        chosen_tree_blocks.append(leaf)
        tree_block_ids.add(leaf.idx)

        children_of_parent_ids = set(
            v.idx for v in parent.children.values() if v.is_cpu == is_cpu
        )

        if (
            len(children_of_parent_ids - tree_block_ids) == 0
            and len(parent.seq_ids) == 0
        ):
            heapq.heappush(leaf_heap, parent)

    return chosen_floating_blocks, chosen_tree_blocks


class BlockAllocator:
    def __init__(self, num_gpu_blocks: int, num_cpu_blocks: int, page_size: int):
        self.num_blocks = num_gpu_blocks
        self.page_size = page_size

        self.all_gpu_blocks = [PrefixTreeBlock(idx=i) for i in range(num_gpu_blocks)]
        self.all_cpu_blocks = [
            PrefixTreeBlock(idx=i, is_cpu=True) for i in range(num_cpu_blocks)
        ]
        self.floating_gpu_blocks = set(self.all_gpu_blocks)
        self.floating_cpu_blocks = set(self.all_cpu_blocks)
        self.prefix_tree = PrefixTreeBlock(
            idx=-1
        )  # Root of the prefix tree, dummy block

        self.num_free_blocks = len(self.all_gpu_blocks)
        self.available_gpu_leaves: set[PrefixTreeBlock] = set()
        self.available_cpu_leaves: set[PrefixTreeBlock] = set()

    def add_floating(self, block: PrefixTreeBlock):
        assert block.parent is None
        assert len(block.children) == 0
        assert len(block.seq_ids) == 0
        self.floating_gpu_blocks.add(block)

    def pop_gpu_blocks_with_cpu_swaps(self, num_blocks: int):
        assert num_blocks <= self.num_free_blocks, (
            f"Not enough free blocks: {num_blocks} > {self.num_free_blocks}"
        )

        floating_gpu_blocks, tree_gpu_blocks = pick_blocks_for_allocation(
            num_blocks=num_blocks,
            available_floating=self.floating_gpu_blocks,
            available_leaves=self.available_gpu_leaves,
            is_cpu=False,
        )

        floating_cpu_blocks, tree_cpu_blocks = pick_blocks_for_allocation(
            num_blocks=len(tree_gpu_blocks),
            available_floating=self.floating_cpu_blocks,
            available_leaves=self.available_cpu_leaves,
            is_cpu=True,
        )

        all_gpu_blocks = floating_gpu_blocks + tree_gpu_blocks
        all_cpu_blocks = floating_cpu_blocks + tree_cpu_blocks

        for block in floating_cpu_blocks:
            assert len(block.seq_ids) == 0
            assert len(block.children) == 0
            self.floating_cpu_blocks.remove(block)
            block.wipe()

        for block in tree_cpu_blocks:
            assert len(block.seq_ids) == 0
            assert len(block.children) == 0
            block.wipe()

        for block in floating_gpu_blocks:
            assert len(block.seq_ids) == 0
            assert len(block.children) == 0
            self.floating_gpu_blocks.remove(block)
            block.wipe()

        for gpu_block, cpu_block in zip(tree_gpu_blocks, all_cpu_blocks, strict=True):
            assert cpu_block.is_wiped()
            assert len(gpu_block.seq_ids) == 0

            gpu_parent = gpu_block.parent
            assert gpu_parent is not None

            gpu_block.detach_from_parent()

            cpu_block.contents = gpu_block.contents
            cpu_block.attach_to_parent(gpu_parent)
            cpu_block.children = gpu_block.children.copy()

            gpu_block.wipe()

        swap_info = SwapIndices(
            gpu_ids=[block.idx for block in tree_gpu_blocks],
            cpu_ids=[block.idx for block in all_cpu_blocks],
        )

        return all_gpu_blocks, swap_info

    def fraction_used(self):
        return 1 - (self.num_free_blocks / len(self.all_gpu_blocks))

    def fraction_floating(self):
        return len(self.floating_gpu_blocks) / len(self.all_gpu_blocks)

    def sanity_checks(self):
        for block in self.all_gpu_blocks:
            assert not (
                len(block.seq_ids) == 0
                and block.idx not in self.floating_gpu_blocks
                and block.parent is None
            )

            if block.parent is not None:
                assert block.parent.children[block.contents] == block

            for child in block.children.values():
                assert child.parent == block

        for block in self.floating_gpu_blocks:
            assert block.parent is None
            assert len(block.children) == 0

        num_free_blocks = sum(
            1 for block in self.all_gpu_blocks if len(block.seq_ids) == 0
        )
        assert num_free_blocks == self.num_free_blocks

        available_leaves = {
            block
            for block in self.all_gpu_blocks
            if len(block.seq_ids) == 0
            and block.parent is not None
            and len(block.children) == 0
        }
        assert available_leaves == self.available_gpu_leaves

    def prefix_match(self, ids: list[int]) -> list[PrefixTreeBlock]:
        cached_blocks = []
        cur_block_in_tree = self.prefix_tree

        for start in range(0, len(ids), self.page_size):
            # can only cache full pages (for now)
            if start + self.page_size > len(ids):
                break

            sliced_ids = tuple(ids[start : start + self.page_size])

            if (existing_child := cur_block_in_tree.children.get(sliced_ids)) is None:
                break
            else:
                cached_blocks.append(existing_child)
                cur_block_in_tree = existing_child

        return cached_blocks

    def update_prefix_tree(self, blocks: list[PrefixTreeBlock], ids: list[int]):
        assert len(blocks) * self.page_size == len(ids)

        cur_block_in_tree = self.prefix_tree
        for i, block in enumerate(blocks):
            start = i * self.page_size
            page_ids = tuple(ids[start : start + self.page_size])

            if (existing_child := cur_block_in_tree.children.get(page_ids)) is not None:
                cur_block_in_tree = existing_child
            else:
                block.contents = page_ids
                block.parent = cur_block_in_tree
                cur_block_in_tree.children[page_ids] = block

                if cur_block_in_tree in self.available_gpu_leaves:
                    self.available_gpu_leaves.remove(cur_block_in_tree)

                if len(block.seq_ids) == 0:
                    self.available_gpu_leaves.add(block)

                cur_block_in_tree = block

    def assign_block_to_seq(self, block: PrefixTreeBlock, seq_id: str):
        assert seq_id not in block.seq_ids
        assert not block.is_cpu

        if len(block.seq_ids) == 0:
            self.num_free_blocks -= 1

        block.seq_ids.add(seq_id)
        block.last_used_at = time.time()

        if block in self.floating_gpu_blocks:
            self.floating_gpu_blocks.remove(block)

        if block in self.available_gpu_leaves:
            self.available_gpu_leaves.remove(block)

    def enough_free_blocks_for_allocation(
        self,
        num_existing_blocks: int,
        target_token_length: int,
        num_reserved_blocks: int,
    ):
        num_existing_tokens = num_existing_blocks * self.page_size

        num_blocks_needed = math.ceil(
            (target_token_length - num_existing_tokens) / self.page_size
        )

        return num_blocks_needed + num_reserved_blocks <= self.num_free_blocks

    def allocate_up_to_length(self, seq_id: str, kv_indices: list[int], length: int):
        num_existing_blocks = len(kv_indices)
        num_existing_tokens = num_existing_blocks * self.page_size

        new_kv_indices: list[int] = []

        if num_existing_tokens >= length:
            return AllocationResult(
                new_kv_indices=new_kv_indices,
            )

        num_blocks_needed = math.ceil((length - num_existing_tokens) / self.page_size)

        if num_blocks_needed > self.num_free_blocks:
            raise NoSpaceException()

        blocks, gpu_to_cpu_swap_indices = self.pop_gpu_blocks_with_cpu_swaps(
            num_blocks_needed
        )
        for block in blocks:
            self.assign_block_to_seq(block, seq_id)
            new_kv_indices.append(block.idx)

        return AllocationResult(
            new_kv_indices=new_kv_indices,
            swap_info=SwapInfo(
                gpu_to_cpu=gpu_to_cpu_swap_indices,
                cpu_to_gpu=None,
            ),
        )

    def allocate_with_prefix_match(
        self,
        seq_id: str,
        prompt_ids: list[int],
        num_reserved_blocks: int = 0,
    ):
        # we must re-process (i.e. can't cache hit on) the last token in the prompt,
        # since we need the logits from this token for sampling the first completion token
        cached_blocks = self.prefix_match(prompt_ids[:-1])

        gpu_cached_blocks: list[PrefixTreeBlock] = []
        cpu_cached_blocks: list[PrefixTreeBlock] = []
        for block in cached_blocks:
            if block.is_cpu:
                cpu_cached_blocks.append(block)
            else:
                assert len(cpu_cached_blocks) == 0
                gpu_cached_blocks.append(block)

        num_cached_prompt_ids = len(cached_blocks) * self.page_size

        if not self.enough_free_blocks_for_allocation(
            num_existing_blocks=len(gpu_cached_blocks),
            target_token_length=len(prompt_ids),
            num_reserved_blocks=num_reserved_blocks,
        ):
            raise NoSpaceException()

        cached_kv_indices = [block.idx for block in gpu_cached_blocks]

        # important to do assignment before requesting remaining blocks
        # otherwise some of these cached blocks might be reassigned
        for block in gpu_cached_blocks:
            self.assign_block_to_seq(block, seq_id)

        alloc_result = self.allocate_up_to_length(
            seq_id, cached_kv_indices, len(prompt_ids)
        )
        allocated_kv_indices = alloc_result.new_kv_indices
        gpu_to_cpu_swap_indices = alloc_result.swap_info.gpu_to_cpu

        cpu_to_gpu_swap_indices = SwapIndices(
            gpu_ids=allocated_kv_indices[: len(cpu_cached_blocks)],
            cpu_ids=[block.idx for block in cpu_cached_blocks],
        )

        full_kv_ids = cached_kv_indices + allocated_kv_indices
        full_kv_blocks = [self.all_gpu_blocks[idx] for idx in full_kv_ids]

        cachable_prompt_ids = truncate_to_multiple(prompt_ids, self.page_size)
        num_cacheable_blocks = len(cachable_prompt_ids) // self.page_size

        self.update_prefix_tree(
            full_kv_blocks[:num_cacheable_blocks], cachable_prompt_ids
        )

        return PrefixCacheAllocationResult(
            new_kv_indices=full_kv_ids,
            swap_info=SwapInfo(
                gpu_to_cpu=gpu_to_cpu_swap_indices,
                cpu_to_gpu=cpu_to_gpu_swap_indices,
            ),
            num_cached_prompt_tokens=num_cached_prompt_ids,
        )

    def free_and_update(self, seq_id: str, kv_indices: list[int], token_ids: list[int]):
        allocated_blocks = [self.all_gpu_blocks[idx] for idx in kv_indices]

        # free the blocks
        for block in reversed(allocated_blocks):
            block.seq_ids.remove(seq_id)
            if len(block.seq_ids) == 0:
                self.num_free_blocks += 1
            if len(block.seq_ids) == 0:
                # moving the "tail" in the prefix tree that was only used
                # by this sequence, as well as blocks that aren't in the
                # prefix tree (i.e. completion blocks), to floating
                if len(block.children) == 0:
                    block.wipe()
                    self.add_floating(block)

        cachable_full_ids = truncate_to_multiple(token_ids, self.page_size)

        num_cacheable_blocks = len(cachable_full_ids) // self.page_size
        cacheable_blocks = allocated_blocks[:num_cacheable_blocks]

        self.update_prefix_tree(cacheable_blocks, cachable_full_ids)

        # remove any block from floating that's now in the prefix tree
        for block in allocated_blocks:
            if block in self.floating_gpu_blocks and block.parent is not None:
                self.floating_gpu_blocks.remove(block)


class BatchIndexAllocator:
    def __init__(self, max_indices: int):
        self.max_indices = max_indices
        self.available_indices = deque(range(max_indices))

    def allocate(self):
        out = self.available_indices.popleft()
        return out

    def free(self, idx: int):
        self.available_indices.append(idx)
