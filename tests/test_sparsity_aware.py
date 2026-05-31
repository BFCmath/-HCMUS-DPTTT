from __future__ import annotations

import pytest
from dpt_xgboost_baseline import (
    ExactGreedyConfig,
    exact_greedy_split,
    sparsity_aware_split,
    leaf_weight,
    split_gain,
)


def test_sparsity_aware_matches_exact_greedy_when_no_missing() -> None:
    features = [
        [0.0, 2.0],
        [1.0, 1.5],
        [2.0, 0.2],
        [3.0, 0.1],
    ]
    gradients = [-2.0, -1.0, 1.0, 2.0]
    hessians = [1.0, 1.0, 1.0, 1.0]

    exact_res = exact_greedy_split(features, gradients, hessians)
    sparse_res = sparsity_aware_split(features, gradients, hessians)

    assert exact_res is not None
    assert sparse_res is not None
    assert exact_res.feature_index == sparse_res.feature_index
    assert exact_res.threshold == sparse_res.threshold
    assert exact_res.gain == pytest.approx(sparse_res.gain)
    assert exact_res.left_indices == sparse_res.left_indices
    assert exact_res.right_indices == sparse_res.right_indices


def test_sparsity_aware_default_direction_right() -> None:
    # 0: value = 1.0, grad = -1.0, hess = 1.0
    # 1: value = 2.0, grad = 2.0, hess = 1.0
    # 2: value = None (missing), grad = -3.0, hess = 1.0
    #
    # If missing goes to Right:
    # Split at 1.5. Left: {0} (grad -1.0, hess 1.0). Right: {1, 2} (grad -1.0, hess 2.0).
    # If missing goes to Left:
    # Split at 1.5. Left: {0, 2} (grad -4.0, hess 2.0). Right: {1} (grad 2.0, hess 1.0).
    
    features = [[1.0], [2.0], [None]]
    gradients = [-1.0, 2.0, -3.0]
    hessians = [1.0, 1.0, 1.0]

    # Let's verify both splits manually:
    # Total G = -2.0, H = 3.0. Score = G^2/(H+lambda) = 4 / 4 = 1.0
    # If Right: Left G_L=-1, H_L=1 (Score=1/2 = 0.5). Right G_R=-1, H_R=2 (Score=1/3 = 0.333). Total Score sum = 0.833.
    # If Left: Left G_L=-4, H_L=2 (Score=16/3 = 5.333). Right G_R=2, H_R=1 (Score=4/2 = 2.0). Total Score sum = 7.333.
    # Clearly, Left direction should be significantly better!
    # Let's test with L2 = 1.0, gamma = 0.0

    config = ExactGreedyConfig(l2_regularization=1.0, gamma=0.0)
    split = sparsity_aware_split(features, gradients, hessians, config=config)

    assert split is not None
    assert split.feature_index == 0
    assert split.threshold == 1.5
    assert split.default_direction == "left"
    assert set(split.left_indices) == {0, 2}
    assert set(split.right_indices) == {1}


def test_sparsity_aware_default_direction_left() -> None:
    # Reverse of previous test to force "right" direction:
    # 0: value = 1.0, grad = 2.0, hess = 1.0
    # 1: value = 2.0, grad = -1.0, hess = 1.0
    # 2: value = None (missing), grad = -5.0, hess = 1.0
    
    features = [[1.0], [2.0], [None]]
    gradients = [2.0, -1.0, -5.0]
    hessians = [1.0, 1.0, 1.0]

    config = ExactGreedyConfig(l2_regularization=1.0, gamma=0.0)
    split = sparsity_aware_split(features, gradients, hessians, config=config)

    assert split is not None
    assert split.feature_index == 0
    assert split.threshold == 1.5
    assert split.default_direction == "right"
    assert set(split.left_indices) == {0}
    assert set(split.right_indices) == {1, 2}
