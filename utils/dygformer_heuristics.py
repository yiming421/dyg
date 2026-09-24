"""Structural heuristic features for DyGFormer pair scoring.

The sequence-scoped implementation deliberately mirrors DyGFormer's causal
receptive field: for a query at time ``t`` it retains only the latest ``K``
interactions of each endpoint before ``t``.  ``K`` is normally
``max_input_sequence_length - 1`` because DyGFormer reserves one token for the
endpoint itself.
"""

from __future__ import annotations

from typing import Tuple

import numba
import numpy as np
import torch
from utils.heuristic_scaling import FixedHeuristicScaling


HEURISTIC_FEATURE_NAMES = ("recency", "popularity", "past", "ra")


@numba.njit(parallel=True)
def _recent_interaction_heuristic_kernel(
    sources: np.ndarray,
    targets: np.ndarray,
    prediction_times: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    times: np.ndarray,
    recent_cap: int,
) -> np.ndarray:
    """Return absolute recency time, target degree, pair count, and local RA."""
    row_count = len(indptr) - 1
    output = np.zeros((len(sources), 4), dtype=np.float64)
    output[:, 0] = -1e15

    for row_idx in numba.prange(len(sources)):
        source = sources[row_idx]
        target = targets[row_idx]
        prediction_time = prediction_times[row_idx]
        if (
            source < 0
            or source >= row_count
            or target < 0
            or target >= row_count
        ):
            continue

        source_row_start = indptr[source]
        source_row_end = indptr[source + 1]
        source_cutoff = source_row_start + np.searchsorted(
            times[source_row_start:source_row_end], prediction_time
        )
        source_recent_start = max(source_row_start, source_cutoff - recent_cap)

        target_row_start = indptr[target]
        target_row_end = indptr[target + 1]
        target_cutoff = target_row_start + np.searchsorted(
            times[target_row_start:target_row_end], prediction_time
        )
        target_recent_start = max(target_row_start, target_cutoff - recent_cap)

        # A pair event is present in both undirected endpoint rows.  With two
        # independently truncated histories, one endpoint can retain more pair
        # events than the other; max(count_src, count_dst) is their deduplicated
        # union because both retained sets are suffixes in time.
        source_pair_count = 0
        target_pair_count = 0
        last_interaction = -1e15
        for position in range(source_recent_start, source_cutoff):
            if indices[position] == target:
                source_pair_count += 1
                if times[position] > last_interaction:
                    last_interaction = times[position]
        for position in range(target_recent_start, target_cutoff):
            if indices[position] == source:
                target_pair_count += 1
                if times[position] > last_interaction:
                    last_interaction = times[position]

        output[row_idx, 0] = last_interaction
        output[row_idx, 1] = float(target_cutoff - target_recent_start)
        output[row_idx, 2] = float(max(source_pair_count, target_pair_count))

        # The endpoint neighborhoods and the common neighbor's degree all use
        # the same latest-K causal scope.  Repeated endpoint interactions do not
        # create repeated common-neighbor contributions.
        ra_score = 0.0
        for source_position in range(source_recent_start, source_cutoff):
            common_neighbor = indices[source_position]
            if common_neighbor <= 0 or common_neighbor >= row_count:
                continue

            already_seen = False
            for prior_position in range(source_recent_start, source_position):
                if indices[prior_position] == common_neighbor:
                    already_seen = True
                    break
            if already_seen:
                continue

            found_in_target = False
            for target_position in range(target_recent_start, target_cutoff):
                if indices[target_position] == common_neighbor:
                    found_in_target = True
                    break
            if not found_in_target:
                continue

            neighbor_row_start = indptr[common_neighbor]
            neighbor_row_end = indptr[common_neighbor + 1]
            neighbor_cutoff = neighbor_row_start + np.searchsorted(
                times[neighbor_row_start:neighbor_row_end], prediction_time
            )
            degree = min(recent_cap, neighbor_cutoff - neighbor_row_start)
            if degree > 0:
                ra_score += 1.0 / float(degree)
        output[row_idx, 3] = ra_score

    return output


def _time_sorted_csr(neighbor_sampler) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_name = "_dygformer_heuristic_time_sorted_csr"
    cached = getattr(neighbor_sampler, cache_name, None)
    if cached is not None:
        return cached

    lengths = np.asarray(
        [len(row) for row in neighbor_sampler.nodes_neighbor_ids], dtype=np.int64
    )
    indptr = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    if int(indptr[-1]) == 0:
        indices = np.empty(0, dtype=np.int64)
        times = np.empty(0, dtype=np.float64)
    else:
        indices = np.concatenate(neighbor_sampler.nodes_neighbor_ids).astype(
            np.int64, copy=False
        )
        times = np.concatenate(neighbor_sampler.nodes_neighbor_times).astype(
            np.float64, copy=False
        )
    cached = (indptr, indices, times)
    setattr(neighbor_sampler, cache_name, cached)
    return cached


class RecentInteractionHeuristicExtractor(FixedHeuristicScaling):
    """Four structural features restricted to each node's latest-K history."""

    feature_names = HEURISTIC_FEATURE_NAMES

    def __init__(self, neighbor_sampler, recent_cap: int):
        if int(recent_cap) <= 0:
            raise ValueError("recent_cap must be positive")
        self.neighbor_sampler = neighbor_sampler
        self.recent_cap = int(recent_cap)
        self.indptr, self.indices, self.times = _time_sorted_csr(neighbor_sampler)

    def get_raw_features(
        self,
        sources: np.ndarray,
        targets: np.ndarray,
        prediction_times: np.ndarray,
        **_unused,
    ) -> np.ndarray:
        sources = np.asarray(sources, dtype=np.int64).reshape(-1)
        targets = np.asarray(targets, dtype=np.int64).reshape(-1)
        prediction_times = np.asarray(prediction_times, dtype=np.float64).reshape(-1)
        if not (len(sources) == len(targets) == len(prediction_times)):
            raise ValueError("sources, targets, and prediction_times must align")

        raw = _recent_interaction_heuristic_kernel(
            sources,
            targets,
            prediction_times,
            self.indptr,
            self.indices,
            self.times,
            self.recent_cap,
        )
        unseen = raw[:, 0] <= -1e14
        delta = prediction_times - raw[:, 0]
        delta[unseen] = 1e9
        output = np.empty_like(raw, dtype=np.float32)
        output[:, 0] = -np.log1p(np.clip(delta, a_min=0.0, a_max=None))
        output[:, 1] = np.log1p(raw[:, 1])
        output[:, 2] = np.log1p(raw[:, 2])
        output[:, 3] = np.log1p(raw[:, 3])
        return output


def build_dygformer_heuristic_extractor(
    *, neighbor_sampler, scope: str, recent_cap: int, use_gpu_heuristics: bool = False
):
    scope = str(scope).strip().lower()
    if scope == "sequence":
        return RecentInteractionHeuristicExtractor(
            neighbor_sampler=neighbor_sampler, recent_cap=recent_cap
        )
    if scope == "full":
        # Import lazily so ordinary DTGB baselines retain their original startup
        # path and dependencies.
        from experiments.modules.semantic_mlp.heuristic_components import (
            HeuristicFeatureExtractor,
        )

        return HeuristicFeatureExtractor(
            neighbor_sampler=neighbor_sampler,
            use_gpu_heuristics=use_gpu_heuristics,
            feature_names=HEURISTIC_FEATURE_NAMES,
        )
    raise ValueError(f"Unsupported DyGFormer heuristic scope: {scope!r}")


def prepare_pos_neg_heuristic_tensors(
    *,
    extractor,
    pos_sources: np.ndarray,
    pos_targets: np.ndarray,
    pos_times: np.ndarray,
    neg_sources: np.ndarray,
    neg_targets: np.ndarray,
    neg_times: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transform positive/negative rows using fixed training normalization."""
    pos_raw = extractor.get_raw_features(
        sources=pos_sources, targets=pos_targets, prediction_times=pos_times
    )
    neg_raw = extractor.get_raw_features(
        sources=neg_sources, targets=neg_targets, prediction_times=neg_times
    )
    combined = extractor.normalize_raw_features(
        np.concatenate([pos_raw, neg_raw], axis=0)
    )
    positive_count = len(pos_sources)
    tensor = torch.from_numpy(combined).to(device=device, dtype=dtype)
    return tensor[:positive_count], tensor[positive_count:]


__all__ = [
    "HEURISTIC_FEATURE_NAMES",
    "RecentInteractionHeuristicExtractor",
    "build_dygformer_heuristic_extractor",
    "prepare_pos_neg_heuristic_tensors",
]
