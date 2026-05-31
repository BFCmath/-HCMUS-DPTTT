from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .exact_greedy import ExactGreedyConfig, leaf_weight, split_gain
from .sparsity_aware import SparsitySplitCandidate


class ColumnBlock:
    """Pre-sorted in-memory Block structure for XGBoost (Section 4)."""

    def __init__(self, features: Sequence[Sequence[float | None]]):
        if not features:
            raise ValueError("features must contain at least one row")
        self.row_count = len(features)
        self.feature_count = len(features[0])
        if self.feature_count == 0:
            raise ValueError("features must contain at least one column")

        # Pre-sort each column once in the constructor
        self.sorted_values = []
        self.sorted_instance_indices = []

        for feature_index in range(self.feature_count):
            col_data = []
            for row_index in range(self.row_count):
                val = features[row_index][feature_index]
                if val is not None:
                    col_data.append((val, row_index))
            # Sort by value, then row_index (for tie-breaking stability)
            col_data.sort(key=lambda x: (x[0], x[1]))

            vals = tuple(x[0] for x in col_data)
            indices = tuple(x[1] for x in col_data)

            self.sorted_values.append(vals)
            self.sorted_instance_indices.append(indices)


def column_block_split(
    block: ColumnBlock,
    gradients: Sequence[float],
    hessians: Sequence[float],
    *,
    instance_indices: Sequence[int] | None = None,
    config: ExactGreedyConfig | None = None,
) -> SparsitySplitCandidate | None:
    """Find the best split without sorting at runtime using the Column Block layout (Section 4)."""

    config = config or ExactGreedyConfig()

    if len(gradients) != block.row_count:
        raise ValueError("gradients length must match number of feature rows")
    if len(hessians) != block.row_count:
        raise ValueError("hessians length must match number of feature rows")

    if instance_indices is None:
        indices = tuple(range(block.row_count))
    else:
        seen = set()
        normalized = []
        for i in instance_indices:
            if i in seen:
                raise ValueError("instance_indices must not contain duplicates")
            if i < 0 or i >= block.row_count:
                raise IndexError("instance_indices contains an out-of-range row index")
            seen.add(i)
            normalized.append(i)
        indices = tuple(normalized)

    if len(indices) < 2:
        return None

    total_gradient = sum(gradients[i] for i in indices)
    total_hessian = sum(hessians[i] for i in indices)
    if total_hessian < 2 * config.min_child_weight:
        return None

    # Fast active-instance check mask
    mask = [False] * block.row_count
    for i in indices:
        mask[i] = True

    best: SparsitySplitCandidate | None = None
    best_gain = 0.0

    for feature_index in range(block.feature_count):
        sorted_vals = block.sorted_values[feature_index]
        sorted_indices = block.sorted_instance_indices[feature_index]

        # Filter out instances not in the active set of indices
        active_sorted = [
            (val, idx)
            for val, idx in zip(sorted_vals, sorted_indices)
            if mask[idx]
        ]

        # 1. Missing goes to Right (Ascending scan)
        left_gradient = 0.0
        left_hessian = 0.0
        cursor = 0

        while cursor < len(active_sorted):
            feature_value = active_sorted[cursor][0]
            group_end = cursor
            while (
                group_end < len(active_sorted)
                and active_sorted[group_end][0] == feature_value
            ):
                row_index = active_sorted[group_end][1]
                left_gradient += gradients[row_index]
                left_hessian += hessians[row_index]
                group_end += 1

            if group_end == len(active_sorted):
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
                next_value = active_sorted[group_end][0]
                threshold = (feature_value + next_value) / 2.0
                if gain > best_gain:
                    left_indices = tuple(idx for _, idx in active_sorted[:group_end])
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

        # 2. Missing goes to Left (Descending scan)
        active_sorted_desc = sorted(
            active_sorted,
            key=lambda x: (-x[0], x[1])
        )

        right_gradient = 0.0
        right_hessian = 0.0
        cursor = 0

        while cursor < len(active_sorted_desc):
            feature_value = active_sorted_desc[cursor][0]
            group_end = cursor
            while (
                group_end < len(active_sorted_desc)
                and active_sorted_desc[group_end][0] == feature_value
            ):
                row_index = active_sorted_desc[group_end][1]
                right_gradient += gradients[row_index]
                right_hessian += hessians[row_index]
                group_end += 1

            if group_end == len(active_sorted_desc):
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
                next_value = active_sorted_desc[group_end][0]
                threshold = (feature_value + next_value) / 2.0
                if gain > best_gain:
                    right_indices = tuple(idx for _, idx in active_sorted_desc[:group_end])
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
