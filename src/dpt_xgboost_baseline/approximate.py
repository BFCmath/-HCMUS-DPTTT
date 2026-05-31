from __future__ import annotations

from math import isfinite
from typing import Sequence

from .exact_greedy import ExactGreedyConfig, leaf_weight, split_gain
from .sparsity_aware import SparsitySplitCandidate, _validate_sparse_inputs


def propose_candidates(
    values: Sequence[float | None],
    hessians: Sequence[float],
    indices: Sequence[int],
    eps: float,
) -> list[float]:
    """Propose candidate split points according to percentiles of feature distribution (Weighted Quantile Sketch)."""

    # Filter non-missing indices
    non_missing = [i for i in indices if values[i] is not None]
    if not non_missing:
        return []

    # Aggregate hessians by feature value
    val_to_hessian = {}
    for i in non_missing:
        v = values[i]
        val_to_hessian[v] = val_to_hessian.get(v, 0.0) + hessians[i]

    # Sort unique values
    unique_vals = sorted(val_to_hessian.keys())
    if len(unique_vals) <= 2:
        return unique_vals

    # Calculate cumulative weights
    cum_weight = 0.0
    cum_weights = []
    for val in unique_vals:
        cum_weight += val_to_hessian[val]
        cum_weights.append(cum_weight)

    total_weight = cum_weight
    if total_weight <= 0.0:
        # If all hessians are zero, divide unique values evenly based on eps
        step = max(1, int(len(unique_vals) * eps))
        candidates = [unique_vals[i] for i in range(0, len(unique_vals), step)]
        if unique_vals[-1] not in candidates:
            candidates.append(unique_vals[-1])
        return candidates

    # Select candidates
    candidates = [unique_vals[0]]
    last_r = 0.0
    for idx, val in enumerate(unique_vals):
        r = cum_weights[idx] / total_weight
        if r - last_r >= eps:
            candidates.append(val)
            last_r = r

    if unique_vals[-1] not in candidates:
        candidates.append(unique_vals[-1])

    return candidates


def approximate_split(
    features: Sequence[Sequence[float | None]],
    gradients: Sequence[float],
    hessians: Sequence[float],
    *,
    instance_indices: Sequence[int] | None = None,
    config: ExactGreedyConfig | None = None,
    eps: float = 0.1,
    global_candidates: dict[int, list[float]] | None = None,
) -> SparsitySplitCandidate | None:
    """Find the best split using the Approximate Algorithm (Algorithm 3)."""

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
        # 1. Propose or retrieve candidates
        if global_candidates is not None and feature_index in global_candidates:
            candidates = global_candidates[feature_index]
        else:
            feature_values = [features[i][feature_index] for i in range(len(features))]
            candidates = propose_candidates(feature_values, hessians, indices, eps)

        if len(candidates) < 2:
            continue

        candidates = sorted(candidates)
        non_missing = [i for i in indices if features[i][feature_index] is not None]

        # Evaluate splits at each candidate (except the last one)
        for s_v in candidates[:-1]:
            left_non_missing = [i for i in non_missing if features[i][feature_index] <= s_v]
            left_gradient_nm = sum(gradients[i] for i in left_non_missing)
            left_hessian_nm = sum(hessians[i] for i in left_non_missing)

            right_non_missing = [i for i in non_missing if features[i][feature_index] > s_v]
            right_gradient_nm = sum(gradients[i] for i in right_non_missing)
            right_hessian_nm = sum(hessians[i] for i in right_non_missing)

            # A. Enumerate missing goes to Right
            left_gradient = left_gradient_nm
            left_hessian = left_hessian_nm
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
                    best_gain = gain
                    idx = candidates.index(s_v)
                    next_s = candidates[idx + 1]
                    threshold = (s_v + next_s) / 2.0

                    left_indices = tuple(left_non_missing)
                    left_set = set(left_indices)
                    right_indices = tuple(i for i in indices if i not in left_set)

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

            # B. Enumerate missing goes to Left
            right_gradient = right_gradient_nm
            right_hessian = right_hessian_nm
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
                if gain > best_gain:
                    best_gain = gain
                    idx = candidates.index(s_v)
                    next_s = candidates[idx + 1]
                    threshold = (s_v + next_s) / 2.0

                    right_indices = tuple(right_non_missing)
                    right_set = set(right_indices)
                    left_indices = tuple(i for i in indices if i not in right_set)

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

    return best
