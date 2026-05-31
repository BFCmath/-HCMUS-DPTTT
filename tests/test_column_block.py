from __future__ import annotations

import pytest
from dpt_xgboost_baseline import (
    ExactGreedyConfig,
    exact_greedy_split,
    sparsity_aware_split,
    ColumnBlock,
    column_block_split,
)


def test_column_block_matches_exact_greedy() -> None:
    features = [
        [0.0, 2.0],
        [1.0, 1.5],
        [2.0, 0.2],
        [3.0, 0.1],
    ]
    gradients = [-2.0, -1.0, 1.0, 2.0]
    hessians = [1.0, 1.0, 1.0, 1.0]

    exact = exact_greedy_split(features, gradients, hessians)
    assert exact is not None

    block = ColumnBlock(features)
    block_res = column_block_split(block, gradients, hessians)

    assert block_res is not None
    assert block_res.feature_index == exact.feature_index
    assert block_res.threshold == pytest.approx(exact.threshold)
    assert block_res.gain == pytest.approx(exact.gain)
    assert set(block_res.left_indices) == set(exact.left_indices)
    assert set(block_res.right_indices) == set(exact.right_indices)


def test_column_block_matches_sparsity_aware() -> None:
    # Dataset with missing values
    features = [
        [1.0],
        [2.0],
        [None],
        [3.0],
    ]
    gradients = [2.0, -1.0, -5.0, 4.0]
    hessians = [1.0, 1.0, 1.0, 1.0]

    config = ExactGreedyConfig(l2_regularization=1.0, gamma=0.0)
    sparse_res = sparsity_aware_split(features, gradients, hessians, config=config)
    assert sparse_res is not None

    block = ColumnBlock(features)
    block_res = column_block_split(block, gradients, hessians, config=config)

    assert block_res is not None
    assert block_res.feature_index == sparse_res.feature_index
    assert block_res.threshold == pytest.approx(sparse_res.threshold)
    assert block_res.gain == pytest.approx(sparse_res.gain)
    assert block_res.default_direction == sparse_res.default_direction
    assert set(block_res.left_indices) == set(sparse_res.left_indices)
    assert set(block_res.right_indices) == set(sparse_res.right_indices)


def test_column_block_split_with_instance_indices() -> None:
    features = [
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ]
    gradients = [-100.0, -2.0, 2.0, 100.0]
    hessians = [1.0, 1.0, 1.0, 1.0]

    block = ColumnBlock(features)
    # only consider instance indices 1 and 2
    block_res = column_block_split(block, gradients, hessians, instance_indices=[1, 2])

    assert block_res is not None
    assert block_res.threshold == 1.5
    assert set(block_res.left_indices) == {1}
    assert set(block_res.right_indices) == {2}
