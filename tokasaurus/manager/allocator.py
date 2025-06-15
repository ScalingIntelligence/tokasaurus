import heapq
import math
import time
from tqdm import tqdm
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from tokasaurus.manager.monitoring import track_time_decorator


@dataclass
class Block:
    idx: int
    prepended_cartridge_ids: Optional[tuple[str, ...]] = None
    contents: tuple[int, ...] = field(default_factory=tuple)
    last_used_at: float = 0.0
    seq_ids: set[str] = field(default_factory=set)
    children: dict[tuple[Optional[tuple[str, ...]], tuple[int, ...]], "Block"] = field(default_factory=dict)
    parent: Optional["Block"] = None

    # Cartridge block related fields
    cartridge_id: Optional[str] = None
    next_cartridge_block_ref: Optional["Block"] = None

    @property
    def key(self):
        return (self.prepended_cartridge_ids, self.contents)

    def is_root(self):
        return self.idx == -1

    def is_cartridge_block(self):
        return self.cartridge_id is not None

    def detach_from_parent(self):
        assert self.parent is not None
        assert self.parent.children[self.key] == self
        self.parent.children.pop(self.key)
        self.parent = None

    def wipe(self):
        if self.parent is not None:
            self.detach_from_parent()
        self.children.clear()
        self.seq_ids.clear()
        self.last_used_at = 0.0
        self.prepended_cartridge_ids = None
        self.contents = tuple()
        self.cartridge_id = None
        self.next_cartridge_block_ref = None

    def __repr__(self):
        if self.is_cartridge_block():
            return f"""Block(
                idx={self.idx},
                cartridge_id={self.cartridge_id},
                used_by={len(self.seq_ids)},
            )"""
        else:
            return f"""Block(
                idx={self.idx},
                contents={self.contents},
                children={len(self.children)}
                used_by={len(self.seq_ids)},
                prepended_cartridge_ids={self.prepended_cartridge_ids}
            )"""

    def __hash__(self):
        return self.idx

    def __lt__(self, other: "Block"):
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


class NoSpaceException(ValueError):
    """Exception raised when there is insufficient space for allocating a block."""


def truncate_to_multiple(x: list, multiple: int):
    return x[: len(x) - (len(x) % multiple)]


def pick_blocks_for_allocation(
    num_blocks: int,
    available_floating: list[Block],
    available_leaves: set[Block],
    cartridge_id_to_head: dict[str, Block],
    available_leaf_heap: list[Block] | None = None,
    allow_used_leaves_in_heap: bool = False,
):
    assert num_blocks > 0

    chosen_floating_blocks: list[Block] = []
    chosen_tree_blocks: list[Block] = []

    for _ in range(min(num_blocks, len(available_floating))):
        chosen_floating_blocks.append(available_floating.pop())

    num_still_needed = num_blocks - len(chosen_floating_blocks)
    if num_still_needed == 0:
        return chosen_floating_blocks, chosen_tree_blocks

    if available_leaf_heap is None:
        available_leaf_heap = list(available_leaves)
        heapq.heapify(available_leaf_heap)

    def get_next_leaf():
        if not allow_used_leaves_in_heap:
            leaf = heapq.heappop(available_leaf_heap)
            available_leaves.remove(leaf)
            assert len(leaf.seq_ids) == 0
            return leaf
        else:
            while True:
                candidate_leaf = heapq.heappop(available_leaf_heap)
                is_free = len(candidate_leaf.seq_ids) == 0
                assert is_free == (candidate_leaf in available_leaves)
                if is_free:
                    available_leaves.remove(candidate_leaf)
                    return candidate_leaf

    for _ in range(num_still_needed):
        try:
            leaf = get_next_leaf()
        except (IndexError, ValueError):
            # Could not get a leaf, so break from this loop
            # The next step is to try and free cartridges
            break

        assert len(leaf.children) == 0 # It's a leaf
        chosen_tree_blocks.append(leaf)

        parent = leaf.parent
        assert parent is not None and not leaf.is_root() # Leaves in tree must have a parent

        leaf.wipe()

        if (
            len(parent.children) == 0
            and len(parent.seq_ids) == 0
            and not parent.is_root()
        ):
            heapq.heappush(available_leaf_heap, parent)
            available_leaves.add(parent)

    num_still_needed -= len(chosen_tree_blocks)
    if num_still_needed == 0:
        return chosen_floating_blocks, chosen_tree_blocks

    # Get empty seq_ids cartridges and sort by LRU
    eligible_cartridges_to_free = []
    for cartridge_id, head_block in cartridge_id_to_head.items():
        if len(head_block.seq_ids) == 0:
            eligible_cartridges_to_free.append((head_block.last_used_at, cartridge_id, head_block))
    eligible_cartridges_to_free.sort()

    while num_still_needed > 0 and eligible_cartridges_to_free:
        _, lru_cartridge_id, lru_head_block = eligible_cartridges_to_free.pop(0)

        del cartridge_id_to_head[lru_cartridge_id]

        current_block = lru_head_block
        while current_block is not None:
            # Always wipe the block because we need to clear the entire cartridge,
            # but only add to chosen_floating_blocks if we still need more blocks
            current_block.wipe()
            if num_still_needed > 0:
                chosen_floating_blocks.append(current_block)
                num_still_needed -= 1
            current_block = current_block.next_cartridge_block_ref

    # Final check: if we still haven't gathered enough blocks, raise an exception
    if num_still_needed > 0:
        raise NoSpaceException(
            f"Could not pick {num_blocks} blocks. Still needed {num_still_needed} "
            "after trying floating blocks, tree leaves, and freeing cartridges."
        )

    return chosen_floating_blocks, chosen_tree_blocks


class BlockAllocator:
    def __init__(self, num_blocks: int, page_size: int):
        self.num_blocks = num_blocks
        self.page_size = page_size

        self.all_blocks = [Block(idx=i) for i in range(num_blocks)]
        self.floating_blocks = list(self.all_blocks)
        self.prefix_tree = Block(
            idx=-1
        )  # Root of the prefix tree, dummy block

        self.num_free_blocks = len(self.all_blocks)
        self.available_leaves: set[Block] = set()
        self.cartridge_id_to_head: dict[str, Block] = {}

    def add_floating(self, block: Block):
        assert block.parent is None
        assert len(block.children) == 0
        assert len(block.seq_ids) == 0
        self.floating_blocks.append(block)

    def num_used_blocks(self):
        return len(self.all_blocks) - self.num_free_blocks

    def fraction_used(self):
        return 1 - (self.num_free_blocks / len(self.all_blocks))

    def fraction_floating(self):
        return len(self.floating_blocks) / len(self.all_blocks)

    def sanity_checks(self, seq_ids: set[str] | None = None):
        set_floating = set(self.floating_blocks)
        for block in self.all_blocks:
            if block not in set_floating:
                # Cartridge blocks are lazily evicted, so may not be in the floating set despite
                # having no seq_ids.
                assert block.parent is not None or len(block.seq_ids) > 0 or block.is_cartridge_block()

            if block.parent is not None:
                assert block.parent.children[block.key] == block

            for child in block.children.values():
                assert child.parent == block

        for block in self.floating_blocks:
            assert block.parent is None
            assert len(block.children) == 0

        num_free_blocks = sum(1 for block in self.all_blocks if len(block.seq_ids) == 0)
        assert num_free_blocks == self.num_free_blocks

        # available leaf is 1) unused by a seq, 2) in the prefix tree, 3) has no children
        available_leaves = {
            block
            for block in self.all_blocks
            if len(block.seq_ids) == 0
            and block.parent is not None
            and len(block.children) == 0
        }
        assert available_leaves == self.available_leaves, (
            f"{len(available_leaves)} {len(self.available_leaves)}"
        )

        if seq_ids is not None:
            all_used_seq_ids = set()
            for block in self.all_blocks:
                all_used_seq_ids.update(block.seq_ids)

            assert all_used_seq_ids == seq_ids

    def prefix_match(self, input_ids: list[int], prepended_cartridge_ids: Optional[tuple[str, ...]] = None) -> list[Block]:
        cached_blocks = []
        cur_block_in_tree = self.prefix_tree

        # we must re-process (i.e. can't cache hit on) the last token in the prompt,
        # since we need the logits from this token for sampling the first completion token
        cacheable_input_ids = input_ids[:-1]

        for start in range(0, len(cacheable_input_ids), self.page_size):
            # can only cache full pages (for now)
            if start + self.page_size > len(cacheable_input_ids):
                break

            sliced_ids = tuple(cacheable_input_ids[start : start + self.page_size])
            key = (prepended_cartridge_ids, sliced_ids)

            if (existing_child := cur_block_in_tree.children.get(key)) is None:
                break
            else:
                cached_blocks.append(existing_child)
                cur_block_in_tree = existing_child

        return cached_blocks

    def update_prefix_tree(self, blocks: list[Block], ids: list[int], prepended_cartridge_ids: Optional[tuple[str, ...]] = None):
        assert len(blocks) * self.page_size == len(ids)

        cur_block_in_tree = self.prefix_tree
        for i, block in enumerate(blocks):
            start = i * self.page_size
            page_ids = tuple(ids[start : start + self.page_size])
            key = (prepended_cartridge_ids, page_ids)

            if (existing_child := cur_block_in_tree.children.get(key)) is not None:
                cur_block_in_tree = existing_child
            else:
                block.contents = page_ids
                block.prepended_cartridge_ids = prepended_cartridge_ids
                block.parent = cur_block_in_tree
                cur_block_in_tree.children[key] = block

                if cur_block_in_tree in self.available_leaves:
                    self.available_leaves.remove(cur_block_in_tree)

                if len(block.seq_ids) == 0:
                    self.available_leaves.add(block)

                cur_block_in_tree = block

    def assign_block_to_seq(self, block: Block, seq_id: str):
        assert seq_id not in block.seq_ids, f"{seq_id} {block}"

        if len(block.seq_ids) == 0:
            self.num_free_blocks -= 1

        block.seq_ids.add(seq_id)
        block.last_used_at = time.time()

        if block in self.available_leaves:
            self.available_leaves.remove(block)

    def num_blocks_needed(self, kv_indices: list[int], length: int):
        num_existing_blocks = len(kv_indices)
        num_existing_tokens = num_existing_blocks * self.page_size

        num_blocks_needed = max(
            0, math.ceil((length - num_existing_tokens) / self.page_size)
        )

        return num_blocks_needed

    @track_time_decorator()
    def allocate_up_to_length(
        self,
        seq_id: str,
        kv_indices: list[int],
        length: int,
        available_leaf_heap: list[Block] | None = None,
        allow_used_leaves_in_heap: bool = False,
    ):
        num_blocks_needed = self.num_blocks_needed(kv_indices, length)

        new_kv_indices: list[int] = []

        if num_blocks_needed == 0:
            return new_kv_indices

        if num_blocks_needed > self.num_free_blocks:
            raise NoSpaceException()

        floating_blocks, tree_blocks = pick_blocks_for_allocation(
            num_blocks=num_blocks_needed,
            available_floating=self.floating_blocks,
            available_leaves=self.available_leaves,
            cartridge_id_to_head=self.cartridge_id_to_head,
            available_leaf_heap=available_leaf_heap,
            allow_used_leaves_in_heap=allow_used_leaves_in_heap,
        )
        all_blocks = floating_blocks + tree_blocks

        for block in all_blocks:
            self.assign_block_to_seq(block, seq_id)
            new_kv_indices.append(block.idx)

        return new_kv_indices

    @track_time_decorator()
    def make_available_leaf_heap(self):
        """
        Valid to reuse across multiple consecutive calls to allocate_up_to_length.

        INVALIDATED by calls to allocate_with_prefix_match or free_and_update.
        """
        available_leaf_heap = list(self.available_leaves)
        heapq.heapify(available_leaf_heap)
        return available_leaf_heap

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

    @track_time_decorator()
    def allocate_with_prefix_match(
        self,
        seq_id: str,
        prompt_ids: list[int],
        num_reserved_blocks: int = 0,
        available_leaf_heap: list[Block] | None = None,
        allow_used_leaves_in_heap: bool = False,
        prepended_cartridge_ids: Optional[tuple[str, ...]] = None,
        cartridge_id_to_num_blocks: Optional[dict[str, int]] = None,
    ) -> tuple[list[int], list[int], int]:
        # First, calculate how many cartridge blocks we'll need
        total_cartridge_blocks_needed = 0
        if prepended_cartridge_ids:
            for c_id in prepended_cartridge_ids:
                total_cartridge_blocks_needed += cartridge_id_to_num_blocks[c_id]

        # prefix_match already uses prepended_cartridge_ids for lookups
        token_cached_blocks = self.prefix_match(prompt_ids, prepended_cartridge_ids=prepended_cartridge_ids)

        # Calculate space needed for prompt tokens
        num_cached_prompt_ids = len(token_cached_blocks) * self.page_size
        num_prompt_blocks_needed = math.ceil(len(prompt_ids) / self.page_size)
        
        # Do space check BEFORE any allocation
        # We need: cartridge blocks + prompt blocks (minus what we already have cached)
        total_blocks_needed = total_cartridge_blocks_needed + num_prompt_blocks_needed
        blocks_we_already_have = len(token_cached_blocks)
        
        if total_blocks_needed - blocks_we_already_have + num_reserved_blocks > self.num_free_blocks:
            raise NoSpaceException()

        # NOW it's safe to allocate cartridges (space check passed)
        cartridge_blocks_for_seq: list[Block] = []
        if prepended_cartridge_ids:
            for c_id in prepended_cartridge_ids:
                cartridge_blocks = self.allocate_cartridge(seq_id, c_id, cartridge_id_to_num_blocks[c_id])
                cartridge_blocks_for_seq.extend(cartridge_blocks)

        # Assign token-cached blocks to the sequence
        for block in token_cached_blocks:
            self.assign_block_to_seq(block, seq_id)

        cartridge_kv_indices = [block.idx for block in cartridge_blocks_for_seq]
        token_cached_kv_indices = [block.idx for block in token_cached_blocks]

        try:
            # Allocate further blocks needed only for prompt_ids, based on token_cached_blocks
            newly_allocated_for_prompt_kv_indices = self.allocate_up_to_length(
                seq_id,
                token_cached_kv_indices,
                len(prompt_ids),
                available_leaf_heap=available_leaf_heap,
                allow_used_leaves_in_heap=allow_used_leaves_in_heap,
            )
        except NoSpaceException:
            # This should ideally not happen if our space check above is correct
            # and no other process is concurrently modifying block allocations.
            raise RuntimeError("Shouldn't happen: space check should have caught this.")

        # Calculate the total number of blocks for validation
        expected_prompt_blocks = math.ceil(len(prompt_ids) / self.page_size)
        assert len(cartridge_blocks_for_seq) + len(token_cached_blocks) + len(newly_allocated_for_prompt_kv_indices) == len(cartridge_blocks_for_seq) + expected_prompt_blocks

        full_kv_blocks = cartridge_blocks_for_seq + token_cached_blocks + [self.all_blocks[idx] for idx in newly_allocated_for_prompt_kv_indices]

        # update_prefix_tree should only be concerned with blocks corresponding to prompt_ids
        # and their associated prepended_cartridge_ids for keying.
        # Cartridge blocks themselves are not added to the prefix tree via this mechanism.
        cachable_prompt_ids = truncate_to_multiple(prompt_ids, self.page_size)
        num_cacheable_prompt_blocks = len(cachable_prompt_ids) // self.page_size

        # We need to pass only the blocks that correspond to prompt_ids to update_prefix_tree
        # These are the blocks from token_cached_blocks and the newly allocated ones for prompt_ids.
        prompt_related_blocks = [
            block for block in full_kv_blocks if not block.is_cartridge_block()
        ]

        self.update_prefix_tree(
            prompt_related_blocks[:num_cacheable_prompt_blocks], cachable_prompt_ids, prepended_cartridge_ids=prepended_cartridge_ids
        )

        # Return separate cartridge and token indices
        token_kv_indices = token_cached_kv_indices + newly_allocated_for_prompt_kv_indices
        return cartridge_kv_indices, token_kv_indices, num_cached_prompt_ids

    @track_time_decorator()
    def free_and_update(self, seq_id: str, cartridge_indices: list[int], token_indices: list[int], token_ids: list[int], prepended_cartridge_ids: Optional[tuple[str, ...]] = None):
        # Only deal with token blocks during free_and_update - cartridge blocks are lazily evicted
        allocated_blocks = [self.all_blocks[idx] for idx in token_indices]

        newly_free_block_indices: list[int] = []

        # Free cartridge blocks from sequence assignment but don't wipe them
        for idx in cartridge_indices:
            cartridge_block = self.all_blocks[idx]
            if seq_id in cartridge_block.seq_ids:
                cartridge_block.seq_ids.remove(seq_id)
                cartridge_block.last_used_at = time.time()  # Update LRU timestamp
                # Increment num_free_blocks when cartridge block becomes empty
                if len(cartridge_block.seq_ids) == 0:
                    self.num_free_blocks += 1

        # free the token blocks - reversed order matters so we
        # can move the tail over, described below
        for block in reversed(allocated_blocks):
            block.seq_ids.remove(seq_id)
            if len(block.seq_ids) == 0:
                self.num_free_blocks += 1
                newly_free_block_indices.append(block.idx)

                # For non-cartridge blocks:
                # moving the "tail" in the prefix tree that was only used
                # by this sequence, as well as blocks that aren't in the
                # prefix tree (i.e. completion blocks), out of the tree.
                # Cartridge blocks are not wiped here; they are handled by cartridge eviction.
                if not block.is_cartridge_block() and len(block.children) == 0:
                    block.wipe()

        # Count cartridge blocks
        num_cartridge_blocks = sum(1 for block in allocated_blocks if block.is_cartridge_block())
        
        # Only pass prompt-related blocks to update_prefix_tree, not cartridge blocks
        prompt_related_blocks = [block for block in allocated_blocks if not block.is_cartridge_block()]
        
        # Cartridge blocks provide KV cache space but don't correspond to actual input tokens.
        # The token_ids represent actual input/completion tokens that start after the cartridge space.
        # So we use all the token_ids but only the prompt-related blocks.
        cachable_prompt_ids = truncate_to_multiple(token_ids, self.page_size)
        num_cacheable_prompt_blocks = len(cachable_prompt_ids) // self.page_size
        
        # Ensure we don't try to take more blocks than we have
        num_cacheable_prompt_blocks = min(num_cacheable_prompt_blocks, len(prompt_related_blocks))
        
        # Take only the number of prompt-related blocks that correspond to cacheable tokens
        cacheable_blocks = prompt_related_blocks[:num_cacheable_prompt_blocks]
        
        # Also truncate the token IDs to match the blocks we're actually using
        effective_cachable_ids = cachable_prompt_ids[:num_cacheable_prompt_blocks * self.page_size]
        
        self.update_prefix_tree(cacheable_blocks, effective_cachable_ids, prepended_cartridge_ids=prepended_cartridge_ids)

        # any block we freed (and potentially wiped) that's not in the prefix tree (parent is None)
        # and not a cartridge block, is now floating.
        # Cartridge blocks remain part of their linked list structure even if their seq_ids are empty.
        for block in allocated_blocks:
            if not block.is_cartridge_block() and block.parent is None and len(block.seq_ids) == 0:
                # A wiped block has no parent and no children.
                # A completion block (never in tree) has no parent and no children.
                # Ensure it's truly detached and empty of children if it was a tree block that got wiped or a completion block.
                assert len(block.children) == 0
                self.add_floating(block)

        return newly_free_block_indices

    @track_time_decorator()
    def allocate_cartridge(self, seq_id: str, cartridge_id: str, num_blocks: int) -> list[Block]:
        """
        Allocate blocks for a cartridge given a sequence ID and set up the cartridge structure.
        
        Args:
            seq_id: Sequence ID to assign the blocks to
            cartridge_id: Unique identifier for the cartridge
            num_blocks: Number of blocks needed for this cartridge
        Returns:
            List of allocated blocks forming the cartridge
        """
        if cartridge_id in self.cartridge_id_to_head:
            # Cartridge already exists, return existing blocks
            blocks = []
            current = self.cartridge_id_to_head[cartridge_id]
            while current:
                self.assign_block_to_seq(current, seq_id)
                blocks.append(current)
                current = current.next_cartridge_block_ref
            return blocks

        if num_blocks <= 0:
            raise ValueError(f"Cannot allocate {num_blocks} blocks for cartridge {cartridge_id}. Must be positive.")
        
        if num_blocks > self.num_free_blocks:
            raise NoSpaceException(f"Cannot allocate {num_blocks} blocks for cartridge {cartridge_id}")
        
        # Allocate blocks using the same logic as allocate_up_to_length
        floating_blocks, tree_blocks = pick_blocks_for_allocation(
            num_blocks=num_blocks,
            available_floating=self.floating_blocks,
            available_leaves=self.available_leaves,
            cartridge_id_to_head=self.cartridge_id_to_head,
        )
        all_blocks = floating_blocks + tree_blocks
        
        # Set up cartridge structure and assign to sequence
        for i, block in enumerate(all_blocks):
            block.cartridge_id = cartridge_id
            if i < len(all_blocks) - 1:
                block.next_cartridge_block_ref = all_blocks[i + 1]
            # Assign block to sequence immediately
            self.assign_block_to_seq(block, seq_id)
        
        # Store head of cartridge
        self.cartridge_id_to_head[cartridge_id] = all_blocks[0]    
        return all_blocks


class BatchIndexAllocator:
    def __init__(self, max_indices: int):
        self.max_indices = max_indices
        self.available_indices = deque(range(max_indices))

    def allocate(self):
        out = self.available_indices.popleft()
        return out

    def free(self, idx: int):
        self.available_indices.append(idx)
