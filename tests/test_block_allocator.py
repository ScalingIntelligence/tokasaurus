import math

import pytest

from tokasaurus.manager.allocator import BlockAllocator, NoSpaceException

PAGE_SIZE = 4
NUM_BLOCKS = 8


@pytest.fixture
def allocator():
    return BlockAllocator(page_size=PAGE_SIZE, num_blocks=NUM_BLOCKS)


def test_basic_allocation(allocator: BlockAllocator):
    allocator.sanity_checks()

    input_ids1 = [1] * 16
    input_ids2 = [2] * 9
    input_ids3 = [3] * 8

    allocator.sanity_checks()

    kvs1, num_cached1 = allocator.allocate_with_prefix_match("seq1", input_ids1)
    allocator.sanity_checks()

    assert num_cached1 == 0
    assert len(kvs1) == math.ceil(len(input_ids1) / PAGE_SIZE)

    kvs2, num_cached2 = allocator.allocate_with_prefix_match("seq2", input_ids2)
    allocator.sanity_checks()

    assert num_cached2 == 0
    assert len(kvs2) == math.ceil(len(input_ids2) / PAGE_SIZE)

    with pytest.raises(NoSpaceException):
        allocator.allocate_with_prefix_match("seq3", input_ids3)


def test_basic_caching(allocator: BlockAllocator):
    allocator.sanity_checks()

    input_ids1 = [1] * 12
    input_ids2 = [1] * 8
    input_ids3 = [2]
    input_ids4 = [3]

    completion_ids1 = input_ids1 + [9] * 2
    completion_ids2 = input_ids2 + [9] * 3
    completion_ids3 = input_ids3 + [9] * 4

    kvs1, num_cached1 = allocator.allocate_with_prefix_match("seq1", input_ids1)
    allocator.sanity_checks()

    assert num_cached1 == 0

    kvs1.extend(allocator.allocate_up_to_length("seq1", kvs1, len(completion_ids1)))
    allocator.sanity_checks()

    kvs2, num_cached2 = allocator.allocate_with_prefix_match("seq2", input_ids2)
    allocator.sanity_checks()

    # 4, not 8, since last block isn't cached
    assert num_cached2 == 4
    assert kvs2[0] == kvs1[0]
    assert kvs2[1] != kvs1[1]

    kvs2.extend(allocator.allocate_up_to_length("seq2", kvs2, len(completion_ids2)))
    allocator.sanity_checks()

    kvs3, num_cached3 = allocator.allocate_with_prefix_match("seq3", input_ids3)
    allocator.sanity_checks()

    assert num_cached3 == 0

    kvs3.extend(allocator.allocate_up_to_length("seq3", kvs3, len(completion_ids3)))
    allocator.sanity_checks()

    with pytest.raises(NoSpaceException):
        allocator.allocate_with_prefix_match("seq4", input_ids4)


def test_basic_free(allocator: BlockAllocator):
    allocator.sanity_checks()

    input_ids1 = [1] * 31
    input_ids2 = [2] * 31
    input_ids3 = [3] * 31

    completion_ids1 = input_ids1 + [1]
    completion_ids2 = input_ids2 + [2]

    kvs1, num_cached1 = allocator.allocate_with_prefix_match("seq1", input_ids1)
    allocator.sanity_checks()

    with pytest.raises(ValueError):
        allocator.allocate_with_prefix_match("seq2", input_ids2)

    kvs1.extend(allocator.allocate_up_to_length("seq1", kvs1, len(completion_ids1)))
    allocator.sanity_checks()
    allocator.free_and_update("seq1", kvs1, completion_ids1)
    allocator.sanity_checks()

    kvs2, num_cached2 = allocator.allocate_with_prefix_match("seq2", input_ids2)
    allocator.sanity_checks()

    assert num_cached2 == 0

    kvs2.extend(allocator.allocate_up_to_length("seq2", kvs2, len(completion_ids2)))
    allocator.sanity_checks()
    allocator.free_and_update("seq2", kvs2, completion_ids2)
    allocator.sanity_checks()

    kvs3, num_cached3 = allocator.allocate_with_prefix_match("seq3", input_ids3)
    allocator.sanity_checks()

    assert num_cached3 == 0


def test_multi_branch_free(allocator: BlockAllocator):
    allocator.sanity_checks()

    input_ids1 = [1] * 15
    input_ids2 = [2] * 15
    input_ids3 = [1] * 8 + [3] * 4

    completion_ids1 = input_ids1 + [9]
    completion_ids2 = input_ids2 + [9]
    completion_ids3 = input_ids3 + [9] * 20

    kvs1, num_cached1 = allocator.allocate_with_prefix_match("seq1", input_ids1)
    allocator.sanity_checks()

    kvs1.extend(allocator.allocate_up_to_length("seq1", kvs1, len(completion_ids1)))
    allocator.free_and_update("seq1", kvs1, completion_ids1)

    kvs2, num_cached2 = allocator.allocate_with_prefix_match("seq2", input_ids2)
    allocator.sanity_checks()

    kvs2.extend(allocator.allocate_up_to_length("seq2", kvs2, len(completion_ids2)))
    allocator.sanity_checks()

    allocator.free_and_update("seq2", kvs2, completion_ids2)
    allocator.sanity_checks()

    kvs3, num_cached3 = allocator.allocate_with_prefix_match("seq3", input_ids3)
    allocator.sanity_checks()

    assert num_cached3 == 8

    kvs3.extend(allocator.allocate_up_to_length("seq3", kvs3, len(completion_ids3)))
    allocator.sanity_checks()

    allocator.free_and_update("seq3", kvs3, completion_ids3)
    allocator.sanity_checks()


if __name__ == "__main__":
    pytest.main([__file__])
