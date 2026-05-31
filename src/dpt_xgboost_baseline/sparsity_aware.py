from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from .exact_greedy import ExactGreedyConfig, leaf_weight, split_gain, _validate_finite


@dataclass(frozen=True)
class SparsitySplitCandidate:
    """Best split for one node when sparsity is present."""

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
    default_direction: str  # "left" or "right"


def _validate_sparse_inputs(
    features: Sequence[Sequence[float | None]],
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
            if value is not None:
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


def sparsity_aware_split(
    features: Sequence[Sequence[float | None]],
    gradients: Sequence[float],
    hessians: Sequence[float],
    *,
    instance_indices: Sequence[int] | None = None,
    config: ExactGreedyConfig | None = None,
) -> SparsitySplitCandidate | None:
    """Find the best split for one node using Sparsity-aware Split Finding (Algorithm 2)."""

    config = config or ExactGreedyConfig()
    indices = _validate_sparse_inputs(features, gradients, hessians, instance_indices)
    if len(indices) < 2:
        return None

    feature_count = len(features[0])
    total_gradient = sum(gradients[i] for i in indices)
    total_hessian = sum(hessians[i] for i in indices)
    if total_hessian < 2 * config.min_child_weight:
        return None

    best: SparsitySplitCandidate | None = None
    best_gain = 0.0

    for feature_index in range(feature_count):
        # Identify instances that have non-missing values
        non_missing = [
            i for i in indices
            if features[i][feature_index] is not None
        ]

        # 1. Enumerate missing value goes to Right
        ordered_asc = sorted(non_missing, key=lambda i: (features[i][feature_index], i))
        left_gradient = 0.0
        left_hessian = 0.0
        cursor = 0

        while cursor < len(ordered_asc):
            feature_value = features[ordered_asc[cursor]][feature_index]
            group_end = cursor
            while (
                group_end < len(ordered_asc)
                and features[ordered_asc[group_end]][feature_index] == feature_value
            ):
                row_index = ordered_asc[group_end]
                left_gradient += gradients[row_index]
                left_hessian += hessians[row_index]
                group_end += 1

            if group_end == len(ordered_asc):
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
                next_value = features[ordered_asc[group_end]][feature_index]
                threshold = (feature_value + next_value) / 2.0
                if gain > best_gain:
                    left_indices = tuple(ordered_asc[:group_end])
                    left_set = set(left_indices)
                    right_indices = tuple(i for i in indices if i not in left_set)
                    best_gain = gain
                    best = SparsitySplitCandidate(
                        feature_index=feature_index,
                        threshold=threshold,
                        gain=gain,
                        left_weight=leaf_weight(left_gradient, left_hessian, config.l2_regularization),
                        right_weight=leaf_weight(right_gradient, right_hessian, config.l2_regularization),
                        left_gradient_sum=left_gradient,
                        left_hessian_sum=left_hessian,
                        right_gradient_sum=right_gradient,
                        right_hessian_sum=right_hessian,
                        left_indices=left_indices,
                        right_indices=right_indices,
                        default_direction="right",
                    )
            cursor = group_end

        # 2. Enumerate missing value goes to Left
        ordered_desc = sorted(non_missing, key=lambda i: (-features[i][feature_index], i))
        right_gradient = 0.0
        right_hessian = 0.0
        cursor = 0

        while cursor < len(ordered_desc):
            feature_value = features[ordered_desc[cursor]][feature_index]
            group_end = cursor
            while (
                group_end < len(ordered_desc)
                and features[ordered_desc[group_end]][feature_index] == feature_value
            ):
                row_index = ordered_desc[group_end]
                right_gradient += gradients[row_index]
                right_hessian += hessians[row_index]
                group_end += 1

            if group_end == len(ordered_desc):
                break

            left_gradient = total_gradient - right_gradient
            left_hessian = total_hessian - right_hessian

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
                next_value = features[ordered_desc[group_end]][feature_index]
                threshold = (feature_value + next_value) / 2.0
                if gain > best_gain:
                    right_indices = tuple(ordered_desc[:group_end])
                    right_set = set(right_indices)
                    left_indices = tuple(i for i in indices if i not in right_set)
                    best_gain = gain
                    best = SparsitySplitCandidate(
                        feature_index=feature_index,
                        threshold=threshold,
                        gain=gain,
                        left_weight=leaf_weight(left_gradient, left_hessian, config.l2_regularization),
                        right_weight=leaf_weight(right_gradient, right_hessian, config.l2_regularization),
                        left_gradient_sum=left_gradient,
                        left_hessian_sum=left_hessian,
                        right_gradient_sum=right_gradient,
                        right_hessian_sum=right_hessian,
                        left_indices=left_indices,
                        right_indices=right_indices,
                        default_direction="left",
                    )
            cursor = group_end

    return best
