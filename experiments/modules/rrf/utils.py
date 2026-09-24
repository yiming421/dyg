"""
Shared Reciprocal Rank Fusion (RRF) utilities.
Used by both heuristic-only and LLM evaluation pipelines.
"""
import numpy as np


def descending_ranks(values, axis=-1):
    """
    Compute 1-based descending ranks with stable tie-break.
    """
    arr = np.asarray(values)
    if arr.ndim == 0:
        raise ValueError("descending_ranks expects at least 1D input.")

    if arr.ndim == 1:
        order = np.argsort(-arr, kind='mergesort')
        ranks = np.empty(order.shape[0], dtype=np.int64)
        ranks[order] = np.arange(1, order.shape[0] + 1, dtype=np.int64)
        return ranks

    axis = int(axis)
    if axis < 0:
        axis += arr.ndim
    if axis < 0 or axis >= arr.ndim:
        raise ValueError(f"Invalid axis={axis} for shape={arr.shape}")

    order = np.argsort(-arr, axis=axis, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.int64)
    axis_len = arr.shape[axis]
    rank_values_shape = [1] * arr.ndim
    rank_values_shape[axis] = axis_len
    rank_values = np.arange(1, axis_len + 1, dtype=np.int64).reshape(rank_values_shape)
    np.put_along_axis(ranks, order, rank_values, axis=axis)
    return ranks


def reciprocal_rank_fusion_from_ranks(rank_tensor, rrf_k=60, model_axis=0):
    """
    Fuse ranks from multiple models/components with RRF.

    Args:
        rank_tensor: array-like rank values.
        rrf_k: RRF denominator offset.
        model_axis: axis representing different component models.
    """
    ranks = np.asarray(rank_tensor, dtype=np.float64)
    if ranks.ndim < 1:
        raise ValueError("rank_tensor must be at least 1D.")
    return np.sum(1.0 / (float(rrf_k) + ranks), axis=int(model_axis))


def reciprocal_rank_fusion_from_scores(score_tensor, rrf_k=60, model_axis=0, rank_axis=-1, return_ranks=False):
    """
    Convert per-component scores to ranks, then fuse with RRF.

    Args:
        score_tensor: array-like with a model/component axis and a rank axis.
        rrf_k: RRF denominator offset.
        model_axis: axis representing different component models.
        rank_axis: axis along which ranking is performed (candidate axis).
        return_ranks: whether to return computed rank tensor.
    """
    scores = np.asarray(score_tensor, dtype=np.float64)
    if scores.ndim < 2:
        raise ValueError("score_tensor must be at least 2D (model_axis + rank_axis).")
    ranks = descending_ranks(scores, axis=rank_axis)
    fused = reciprocal_rank_fusion_from_ranks(ranks, rrf_k=rrf_k, model_axis=model_axis)
    if return_ranks:
        return fused, ranks
    return fused
