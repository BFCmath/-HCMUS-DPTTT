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
        self.has_missing = False

        for feature_index in range(self.feature_count):
            col_data = []
            for row_index in range(self.row_count):
                val = features[row_index][feature_index]
                if val is not None:
                    col_data.append((val, row_index))
                else:
                    self.has_missing = True
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

    # Fast path for dense data. In this case Column Block is just a
    # pre-sorted layout for the exact greedy scan, so there is no need to
    # enumerate both missing-value default directions. This is the path that
    # corresponds to the paper's exact-greedy block discussion in Section 4.1.
    if not block.has_missing:
        return _column_block_dense_split(block, gradients, hessians, indices, config)

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
        # The block is already sorted ascending. For the reverse sparsity scan,
        # reverse the filtered column instead of sorting again at runtime.
        active_sorted_desc = list(reversed(active_sorted))

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


def _column_block_dense_split(
    block: ColumnBlock,
    gradients: Sequence[float],
    hessians: Sequence[float],
    indices: tuple[int, ...],
    config: ExactGreedyConfig,
) -> SparsitySplitCandidate | None:
    """Scan a dense pre-sorted block exactly once per feature."""

    best: SparsitySplitCandidate | None = None
    best_gain = 0.0

    total_gradient = sum(gradients[i] for i in indices)
    total_hessian = sum(hessians[i] for i in indices)

    if len(indices) == block.row_count:
        mask: list[bool] | None = None
    else:
        mask = [False] * block.row_count
        for i in indices:
            mask[i] = True

    for feature_index in range(block.feature_count):
        sorted_vals = block.sorted_values[feature_index]
        sorted_indices = block.sorted_instance_indices[feature_index]

        if mask is None:
            active_values = sorted_vals
            active_indices = sorted_indices
        else:
            active_pairs = [
                (val, idx)
                for val, idx in zip(sorted_vals, sorted_indices)
                if mask[idx]
            ]
            if len(active_pairs) < 2:
                continue
            active_values = tuple(val for val, _ in active_pairs)
            active_indices = tuple(idx for _, idx in active_pairs)

        left_gradient = 0.0
        left_hessian = 0.0
        cursor = 0
        active_count = len(active_indices)

        while cursor < active_count:
            feature_value = active_values[cursor]
            group_end = cursor
            while group_end < active_count and active_values[group_end] == feature_value:
                row_index = active_indices[group_end]
                left_gradient += gradients[row_index]
                left_hessian += hessians[row_index]
                group_end += 1

            if group_end == active_count:
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
                if gain > best_gain:
                    threshold = (feature_value + active_values[group_end]) / 2.0
                    left_indices = tuple(active_indices[:group_end])
                    right_indices = tuple(active_indices[group_end:])
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

    return best
