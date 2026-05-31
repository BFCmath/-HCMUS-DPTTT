"""Exact greedy split finding for second-order tree boosting.

This module implements Algorithm 1, "Exact Greedy Algorithm for Split
Finding", from Chen and Guestrin, "XGBoost: A Scalable Tree Boosting System"
(arXiv:1603.02754).  The algorithm receives the gradients and Hessians for the
instances in one current tree node and returns the best single split.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence


FeatureMatrix = Sequence[Sequence[float]]


@dataclass(frozen=True)
class ExactGreedyConfig:
    """Regularization and structural constraints used by the split finder."""

    l2_regularization: float = 1.0
    gamma: float = 0.0
    min_child_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.l2_regularization < 0:
            raise ValueError("l2_regularization must be non-negative")
        if self.gamma < 0:
            raise ValueError("gamma must be non-negative")
        if self.min_child_weight < 0:
            raise ValueError("min_child_weight must be non-negative")


@dataclass(frozen=True)
class SplitCandidate:
    """Best split for one current node.

    The threshold follows the usual CART interpretation: examples with
    ``x[feature_index] <= threshold`` go left, and the rest go right.
    """

    feature_index: int
    threshold: float
    gain: float
    left_weight: float
    right_weight: float
    left_gradient_sum: float
    left_hessian_sum: float
    right_gradient_sum: float
    right_hessian_sum: float
    left_indices: tuple[int, ...]
    right_indices: tuple[int, ...]


def leaf_weight(gradient_sum: float, hessian_sum: float, l2_regularization: float = 1.0) -> float:
    """Return the optimal leaf score from Eq. (5) in the paper."""

    _validate_finite("gradient_sum", gradient_sum)
    _validate_finite("hessian_sum", hessian_sum)
    if l2_regularization < 0:
        raise ValueError("l2_regularization must be non-negative")
    denominator = hessian_sum + l2_regularization
    if denominator <= 0:
        raise ValueError("hessian_sum + l2_regularization must be positive")
    return -gradient_sum / denominator


def split_gain(
    left_gradient_sum: float,
    left_hessian_sum: float,
    right_gradient_sum: float,
    right_hessian_sum: float,
    *,
    l2_regularization: float = 1.0,
    gamma: float = 0.0,
) -> float:
    """Return the regularized loss reduction from Eq. (7) in the paper."""

    if l2_regularization < 0:
        raise ValueError("l2_regularization must be non-negative")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")

    total_gradient_sum = left_gradient_sum + right_gradient_sum
    total_hessian_sum = left_hessian_sum + right_hessian_sum
    return 0.5 * (
        _structure_score(left_gradient_sum, left_hessian_sum, l2_regularization)
        + _structure_score(right_gradient_sum, right_hessian_sum, l2_regularization)
        - _structure_score(total_gradient_sum, total_hessian_sum, l2_regularization)
    ) - gamma


def exact_greedy_split(
    features: FeatureMatrix,
    gradients: Sequence[float],
    hessians: Sequence[float],
    *,
    instance_indices: Sequence[int] | None = None,
    config: ExactGreedyConfig | None = None,
) -> SplitCandidate | None:
    """Find the best split for one node using exact greedy enumeration.

    ``features`` is row-major: ``features[i][k]`` is the value of feature
    ``k`` for instance ``i``.  If ``instance_indices`` is provided, only those
    rows are considered part of the current node.

    The paper pseudocode visits examples in sorted feature order and updates
    ``G_L``/``H_L`` one instance at a time.  This implementation evaluates
    candidate thresholds only after all equal feature values have moved left,
    because a threshold cannot separate identical continuous feature values.
    """

    config = config or ExactGreedyConfig()
    indices = _validate_inputs(features, gradients, hessians, instance_indices)
    if len(indices) < 2:
        return None

    feature_count = len(features[0])
    total_gradient = sum(gradients[i] for i in indices)
    total_hessian = sum(hessians[i] for i in indices)
    if total_hessian < 2 * config.min_child_weight:
        return None

    best: SplitCandidate | None = None
    best_gain = 0.0

    for feature_index in range(feature_count):
        ordered = sorted(indices, key=lambda i: (features[i][feature_index], i))
        left_gradient = 0.0
        left_hessian = 0.0
        cursor = 0

        while cursor < len(ordered):
            feature_value = features[ordered[cursor]][feature_index]
            group_end = cursor
            while (
                group_end < len(ordered)
                and features[ordered[group_end]][feature_index] == feature_value
            ):
                row_index = ordered[group_end]
                left_gradient += gradients[row_index]
                left_hessian += hessians[row_index]
                group_end += 1

            if group_end == len(ordered):
                break

            right_gradient = total_gradient - left_gradient
            right_hessian = total_hessian - left_hessian
            if (
                left_hessian >= config.min_child_weight
                and right_hessian >= config.min_child_weight
            ):
                gain = split_gain(
                    left_gradient,
                    left_hessian,
                    right_gradient,
                    right_hessian,
                    l2_regularization=config.l2_regularization,
                    gamma=config.gamma,
                )
                next_value = features[ordered[group_end]][feature_index]
                threshold = (feature_value + next_value) / 2.0
                if gain > best_gain:
                    left = tuple(ordered[:group_end])
                    right = tuple(ordered[group_end:])
                    best_gain = gain
                    best = SplitCandidate(
                        feature_index=feature_index,
                        threshold=threshold,
                        gain=gain,
                        left_weight=leaf_weight(
                            left_gradient,
                            left_hessian,
                            config.l2_regularization,
                        ),
                        right_weight=leaf_weight(
                            right_gradient,
                            right_hessian,
                            config.l2_regularization,
                        ),
                        left_gradient_sum=left_gradient,
                        left_hessian_sum=left_hessian,
                        right_gradient_sum=right_gradient,
                        right_hessian_sum=right_hessian,
                        left_indices=left,
                        right_indices=right,
                    )

            cursor = group_end

    return best


def _structure_score(gradient_sum: float, hessian_sum: float, l2_regularization: float) -> float:
    _validate_finite("gradient_sum", gradient_sum)
    _validate_finite("hessian_sum", hessian_sum)
    denominator = hessian_sum + l2_regularization
    if denominator <= 0:
        raise ValueError("hessian_sum + l2_regularization must be positive")
    return gradient_sum * gradient_sum / denominator


def _validate_inputs(
    features: FeatureMatrix,
    gradients: Sequence[float],
    hessians: Sequence[float],
    instance_indices: Sequence[int] | None,
) -> tuple[int, ...]:
    if not features:
        raise ValueError("features must contain at least one row")
    row_count = len(features)
    feature_count = len(features[0])
    if feature_count == 0:
        raise ValueError("features must contain at least one column")
    if len(gradients) != row_count:
        raise ValueError("gradients length must match number of feature rows")
    if len(hessians) != row_count:
        raise ValueError("hessians length must match number of feature rows")

    for row_index, row in enumerate(features):
        if len(row) != feature_count:
            raise ValueError("all feature rows must have the same length")
        for feature_index, value in enumerate(row):
            _validate_finite(f"features[{row_index}][{feature_index}]", value)

    for row_index, value in enumerate(gradients):
        _validate_finite(f"gradients[{row_index}]", value)
    for row_index, value in enumerate(hessians):
        _validate_finite(f"hessians[{row_index}]", value)
        if value < 0:
            raise ValueError("hessians must be non-negative")

    if instance_indices is None:
        return tuple(range(row_count))

    seen: set[int] = set()
    normalized: list[int] = []
    for raw_index in instance_indices:
        if raw_index in seen:
            raise ValueError("instance_indices must not contain duplicates")
        if raw_index < 0 or raw_index >= row_count:
            raise IndexError("instance_indices contains an out-of-range row index")
        seen.add(raw_index)
        normalized.append(raw_index)
    return tuple(normalized)


def _validate_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
