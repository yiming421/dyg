"""Shared primitives for uncertainty-aware TabICL utility routing."""

from __future__ import annotations

import gc
from typing import Protocol

import numpy as np


class RegressorArgs(Protocol):
    """Configuration consumed by :func:`fit_regressor_predict`."""

    n_estimators: int
    batch_size: int
    device: str


def recent_paired_query_indices(
    labels: np.ndarray,
    query_ids: np.ndarray,
    timestamps: np.ndarray,
    positive_count: int,
) -> np.ndarray:
    """Select the most-recent complete DTGB positive/negative query pairs.

    The helper is deliberately split-agnostic: callers decide whether the
    calibration pool is named ``train``, ``validation``, or something else.
    Returned rows retain their source-table order.
    """
    labels = np.asarray(labels, dtype=np.int64)
    query_ids = np.asarray(query_ids, dtype=np.int64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if positive_count < 1:
        raise ValueError("positive_count must be positive")
    if not (labels.ndim == query_ids.ndim == timestamps.ndim == 1):
        raise ValueError("labels, query_ids, and timestamps must be one-dimensional")
    if not (len(labels) == len(query_ids) == len(timestamps)):
        raise ValueError("labels, query_ids, and timestamps must have equal lengths")

    positive_indices = np.flatnonzero(labels == 1)
    if len(positive_indices) < positive_count:
        raise ValueError(
            f"Requested {positive_count} positives from {len(positive_indices)}"
        )
    order = np.lexsort(
        (
            positive_indices,
            query_ids[positive_indices],
            timestamps[positive_indices],
        )
    )
    selected_query_ids = query_ids[positive_indices[order[-positive_count:]]]
    if np.unique(selected_query_ids).size != positive_count:
        raise ValueError("Selected positive rows do not have unique query IDs")

    selected = np.flatnonzero(np.isin(query_ids, selected_query_ids))
    if len(selected) != 2 * positive_count:
        raise ValueError(
            "Selected queries must contain exactly one positive and one matched "
            f"negative; found {len(selected)} rows"
        )
    for query_id in selected_query_ids:
        rows = selected[query_ids[selected] == query_id]
        row_labels = labels[rows]
        if (
            len(rows) != 2
            or np.count_nonzero(row_labels == 1) != 1
            or np.count_nonzero(row_labels == 0) != 1
        ):
            raise ValueError(
                f"Query {int(query_id)} is not one positive/negative pair"
            )
        if timestamps[rows[0]] != timestamps[rows[1]]:
            raise ValueError(
                f"Query {int(query_id)} has mismatched positive/negative times"
            )
    return selected.astype(np.int64, copy=False)


def top_indices(priority: np.ndarray, count: int) -> np.ndarray:
    """Select descending priorities with stable lower-row-index tie breaking."""
    priority = np.asarray(priority, dtype=np.float64)
    indices = np.arange(len(priority), dtype=np.int64)
    order = np.lexsort((indices, -priority))
    return order[:count]


def robust_utility_target(utility: np.ndarray) -> np.ndarray:
    """Compress utility magnitude while retaining its sign."""
    utility = np.asarray(utility, dtype=np.float64)
    return np.sign(utility) * np.sqrt(np.abs(utility))


def fit_regressor_predict(
    X_context: np.ndarray,
    utility_context: np.ndarray,
    X_query: np.ndarray,
    *,
    seed: int,
    args: RegressorArgs,
) -> np.ndarray:
    """Fit the shared TabICL utility regressor and return float64 predictions."""
    import torch
    from tabicl import TabICLRegressor

    order = np.random.default_rng(seed + 30_000).permutation(len(utility_context))
    regressor = TabICLRegressor(
        n_estimators=args.n_estimators,
        batch_size=args.batch_size,
        kv_cache="repr",
        device=args.device,
        random_state=seed,
        verbose=False,
    )
    regressor.fit(X_context[order], utility_context[order])
    prediction = np.asarray(regressor.predict(X_query), dtype=np.float64)
    del regressor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prediction


def uncertainty_partition(
    uncertainty_priority: np.ndarray,
    *,
    core_count: int,
    candidate_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition uncertainty-ranked rows into core, shell, and candidates."""
    order = top_indices(uncertainty_priority, len(uncertainty_priority))
    core = order[:core_count]
    shell = order[core_count:candidate_count]
    candidate = order[:candidate_count]
    return core, shell, candidate


def core_shell_selection(
    uncertainty_priority: np.ndarray,
    learned_prediction: np.ndarray,
    *,
    route_count: int,
    core_count: int,
    candidate_count: int,
) -> np.ndarray:
    """Keep the uncertainty core and fill the remaining budget from the shell."""
    core, shell, _ = uncertainty_partition(
        uncertainty_priority,
        core_count=core_count,
        candidate_count=candidate_count,
    )
    # Match the global row-index tie break used by the frozen priority model.
    shell_order = np.lexsort((shell, -np.asarray(learned_prediction)[shell]))
    learned_shell_order = shell[shell_order[: route_count - core_count]]
    return np.concatenate((core, learned_shell_order))


def core_shell_priority(
    uncertainty_priority: np.ndarray,
    learned_prediction: np.ndarray,
    *,
    core_count: int,
    candidate_count: int,
) -> np.ndarray:
    """Encode core/shell selection as priorities for a frozen route model."""
    core, shell, _ = uncertainty_partition(
        uncertainty_priority,
        core_count=core_count,
        candidate_count=candidate_count,
    )
    shell_values = np.asarray(learned_prediction[shell], dtype=np.float64)
    if not np.all(np.isfinite(shell_values)):
        raise ValueError("Shell predictions contain non-finite values")
    spread = float(np.ptp(shell_values))
    margin = max(1.0, spread + 1.0)
    priority = np.full(len(uncertainty_priority), np.min(shell_values) - margin)
    priority[shell] = shell_values
    priority[core] = np.max(shell_values) + margin
    return priority


def selected_summary(
    selected: np.ndarray,
    baseline_selected: np.ndarray,
    utility: np.ndarray,
) -> dict[str, float | int]:
    """Summarize utility exchanged relative to a baseline route."""
    selected = np.asarray(selected, dtype=np.int64)
    baseline_selected = np.asarray(baseline_selected, dtype=np.int64)
    selected_mask = np.zeros(len(utility), dtype=bool)
    baseline_mask = np.zeros(len(utility), dtype=bool)
    selected_mask[selected] = True
    baseline_mask[baseline_selected] = True
    return {
        "selected_utility_sum": float(np.sum(utility[selected_mask])),
        "selected_utility_mean": float(np.mean(utility[selected_mask])),
        "overlap_with_uncertainty": int(np.sum(selected_mask & baseline_mask)),
        "learned_only_rows": int(np.sum(selected_mask & ~baseline_mask)),
        "learned_only_utility_sum": float(
            np.sum(utility[selected_mask & ~baseline_mask])
        ),
        "uncertainty_only_utility_sum": float(
            np.sum(utility[baseline_mask & ~selected_mask])
        ),
    }


__all__ = [
    "core_shell_priority",
    "core_shell_selection",
    "fit_regressor_predict",
    "recent_paired_query_indices",
    "robust_utility_target",
    "selected_summary",
    "top_indices",
    "uncertainty_partition",
]
