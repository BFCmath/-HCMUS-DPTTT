from __future__ import annotations

import pytest
from dpt_xgboost_baseline import (
    ExactGreedyConfig,
    exact_greedy_split,
    approximate_split,
    propose_candidates,
)


def test_propose_candidates_quantiles() -> None:
    # 10 values from 0.0 to 9.0, all hessians equal to 1.0
    values = [float(i) for i in range(10)]
    hessians = [1.0] * 10
    indices = list(range(10))

    # eps = 0.2 means candidates should be spaced by at least 2 elements (since total_weight = 10.0)
    candidates = propose_candidates(values, hessians, indices, eps=0.2)

    # 10 * 0.2 = 2.0. So candidates should be selected every 2 elements.
    # The unique values are 0, 1, 2, 3, 4, 5, 6, 7, 8, 9.
    # cum_weights: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    # r = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    # candidates should include 0.0, and then whenever difference from last selected is >= 0.2
    # So 0.0 (r=0.1) -> 1 (r=0.2) -> 3 (r=0.4) -> 5 (r=0.6) -> 7 (r=0.8) -> 9 (r=1.0)
    # Let's check exactly what the candidate list contains.
    assert len(candidates) >= 3
    assert candidates[0] == 0.0
    assert candidates[-1] == 9.0


def test_approximate_split_local_vs_exact_greedy() -> None:
    features = [
        [0.0, 2.0],
        [1.0, 1.5],
        [2.0, 0.2],
        [3.0, 0.1],
    ]
    gradients = [-2.0, -1.0, 1.0, 2.0]
    hessians = [1.0, 1.0, 1.0, 1.0]

    # Exact greedy split is at feature_index=0, threshold=1.5
    exact = exact_greedy_split(features, gradients, hessians)
    assert exact is not None

    # Since there are only 4 distinct values per feature, setting eps=0.25 (or smaller) should
    # propose enough candidates to find the exact same split!
    approx = approximate_split(features, gradients, hessians, eps=0.1)

    assert approx is not None
    assert approx.feature_index == exact.feature_index
    assert approx.threshold == pytest.approx(exact.threshold)
    assert approx.gain == pytest.approx(exact.gain)
    assert approx.left_indices == exact.left_indices
    assert approx.right_indices == exact.right_indices


def test_approximate_split_global_proposal() -> None:
    features = [
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ]
    gradients = [-2.0, -1.0, 1.0, 2.0]
    hessians = [1.0, 1.0, 1.0, 1.0]

    # Pre-computed global candidates for feature 0
    # Let's force split to be at candidate 2.0
    global_candidates = {0: [0.0, 2.0, 3.0]}

    approx = approximate_split(
        features,
        gradients,
        hessians,
        global_candidates=global_candidates,
    )

    assert approx is not None
    assert approx.feature_index == 0
    # threshold should be midpoint between 2.0 and 3.0, i.e. 2.5
    # or midpoint between 0.0 and 2.0, i.e. 1.0
    assert approx.threshold in {1.0, 2.5}
