from __future__ import annotations

from math import nan

import pytest

from dpt_xgboost_baseline import (
    ExactGreedyConfig,
    exact_greedy_split,
    leaf_weight,
    split_gain,
)


def test_leaf_weight_matches_paper_formula() -> None:
    assert leaf_weight(gradient_sum=6.0, hessian_sum=2.0, l2_regularization=1.0) == -2.0


def test_split_gain_matches_regularized_loss_reduction() -> None:
    gain = split_gain(
        left_gradient_sum=-3.0,
        left_hessian_sum=2.0,
        right_gradient_sum=1.0,
        right_hessian_sum=2.0,
        l2_regularization=1.0,
        gamma=0.1,
    )
    expected = 0.5 * ((9.0 / 3.0) + (1.0 / 3.0) - (4.0 / 5.0)) - 0.1
    assert gain == pytest.approx(expected)


def test_exact_greedy_finds_best_split_for_hand_checked_node() -> None:
    features = [
        [0.0, 2.0],
        [1.0, 1.5],
        [2.0, 0.2],
        [3.0, 0.1],
    ]
    gradients = [-2.0, -1.0, 1.0, 2.0]
    hessians = [1.0, 1.0, 1.0, 1.0]

    split = exact_greedy_split(
        features,
        gradients,
        hessians,
        config=ExactGreedyConfig(l2_regularization=1.0),
    )

    assert split is not None
    assert split.feature_index == 0
    assert split.threshold == 1.5
    assert split.left_indices == (0, 1)
    assert split.right_indices == (2, 3)
    assert split.left_gradient_sum == -3.0
    assert split.left_hessian_sum == 2.0
    assert split.right_gradient_sum == 3.0
    assert split.right_hessian_sum == 2.0
    assert split.left_weight == 1.0
    assert split.right_weight == -1.0
    assert split.gain == pytest.approx(3.0)


def test_duplicate_feature_values_are_grouped_into_valid_thresholds() -> None:
    features = [
        [0.0],
        [0.0],
        [1.0],
        [2.0],
    ]
    gradients = [-3.0, 3.0, -1.0, 1.0]
    hessians = [1.0, 1.0, 1.0, 1.0]

    split = exact_greedy_split(
        features,
        gradients,
        hessians,
        config=ExactGreedyConfig(l2_regularization=1.0),
    )

    assert split is not None
    assert split.threshold in {0.5, 1.5}
    assert set(split.left_indices).issubset({0, 1, 2, 3})
    assert not ({0} == set(split.left_indices) or {1} == set(split.left_indices))


def test_min_child_weight_filters_underweight_children() -> None:
    features = [[0.0], [1.0], [2.0]]
    gradients = [-1.0, 0.0, 1.0]
    hessians = [0.1, 0.1, 0.1]

    split = exact_greedy_split(
        features,
        gradients,
        hessians,
        config=ExactGreedyConfig(min_child_weight=0.2),
    )

    assert split is None


def test_instance_indices_limit_search_to_current_node() -> None:
    features = [
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ]
    gradients = [-100.0, -2.0, 2.0, 100.0]
    hessians = [1.0, 1.0, 1.0, 1.0]

    split = exact_greedy_split(features, gradients, hessians, instance_indices=[1, 2])

    assert split is not None
    assert split.threshold == 1.5
    assert split.left_indices == (1,)
    assert split.right_indices == (2,)


def test_returns_none_when_all_valid_splits_have_non_positive_gain() -> None:
    features = [[0.0], [1.0], [2.0]]
    gradients = [0.0, 0.0, 0.0]
    hessians = [1.0, 1.0, 1.0]

    assert exact_greedy_split(features, gradients, hessians) is None


def test_rejects_sparse_or_missing_values_for_dense_exact_baseline() -> None:
    with pytest.raises(ValueError, match=r"features\[0\]\[0\] must be finite"):
        exact_greedy_split([[nan], [1.0]], [0.0, 0.0], [1.0, 1.0])


def test_exact_scan_matches_brute_force_threshold_enumeration() -> None:
    features = [
        [1.0, 5.0],
        [4.0, 0.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [3.0, 4.0],
    ]
    gradients = [-1.5, 1.0, -0.5, 0.2, 0.8]
    hessians = [1.0, 2.0, 1.5, 0.7, 1.2]
    config = ExactGreedyConfig(l2_regularization=0.7, gamma=0.05)

    split = exact_greedy_split(features, gradients, hessians, config=config)
    brute = _brute_force_best_split(features, gradients, hessians, config)

    assert split is not None
    assert brute is not None
    assert split.feature_index == brute["feature_index"]
    assert split.threshold == pytest.approx(brute["threshold"])
    assert split.gain == pytest.approx(brute["gain"])
    assert set(split.left_indices) == brute["left_indices"]
    assert set(split.right_indices) == brute["right_indices"]


def _brute_force_best_split(
    features: list[list[float]],
    gradients: list[float],
    hessians: list[float],
    config: ExactGreedyConfig,
) -> dict[str, object] | None:
    best: dict[str, object] | None = None
    best_gain = 0.0
    row_count = len(features)
    feature_count = len(features[0])

    for feature_index in range(feature_count):
        values = sorted({row[feature_index] for row in features})
        for left_value, right_value in zip(values, values[1:]):
            threshold = (left_value + right_value) / 2.0
            left_indices = {
                row_index
                for row_index in range(row_count)
                if features[row_index][feature_index] <= threshold
            }
            right_indices = set(range(row_count)) - left_indices
            left_gradient = sum(gradients[i] for i in left_indices)
            left_hessian = sum(hessians[i] for i in left_indices)
            right_gradient = sum(gradients[i] for i in right_indices)
            right_hessian = sum(hessians[i] for i in right_indices)
            if (
                left_hessian < config.min_child_weight
                or right_hessian < config.min_child_weight
            ):
                continue
            gain = split_gain(
                left_gradient,
                left_hessian,
                right_gradient,
                right_hessian,
                l2_regularization=config.l2_regularization,
                gamma=config.gamma,
            )
            if gain > best_gain:
                best_gain = gain
                best = {
                    "feature_index": feature_index,
                    "threshold": threshold,
                    "gain": gain,
                    "left_indices": left_indices,
                    "right_indices": right_indices,
                }

    return best
