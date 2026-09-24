"""
Batched train-pool candidate containers for train-pool RRF evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrainPoolBatch:
    """
    Dense train-pool representation.

    Shapes:
      - sample_query_ids: (N,)
      - pool_sample_indices: (R,)
      - pool_source_ids: (R,)
      - pool_timestamps: (R,)
      - pool_targets: (R, C)
      - pool_valid_mask: (R, C)
      - pool_recency_scores: (R, C)
      - pool_past_scores: (R, C)

    Where:
      - N = number of original evaluation samples
      - R = total sampled train pools across samples
      - C = pool size
    """

    sample_query_ids: np.ndarray
    pool_sample_indices: np.ndarray
    pool_source_ids: np.ndarray
    pool_timestamps: np.ndarray
    pool_targets: np.ndarray
    pool_valid_mask: np.ndarray
    pool_recency_scores: np.ndarray
    pool_past_scores: np.ndarray
    anchor_col: int = 0

    def __post_init__(self):
        sample_query_ids = np.asarray(self.sample_query_ids)
        pool_sample_indices = np.asarray(self.pool_sample_indices)
        pool_source_ids = np.asarray(self.pool_source_ids)
        pool_timestamps = np.asarray(self.pool_timestamps)
        pool_targets = np.asarray(self.pool_targets)
        pool_valid_mask = np.asarray(self.pool_valid_mask)
        pool_recency_scores = np.asarray(self.pool_recency_scores)
        pool_past_scores = np.asarray(self.pool_past_scores)

        if sample_query_ids.ndim != 1:
            raise ValueError("sample_query_ids must be 1D.")
        if pool_sample_indices.ndim != 1:
            raise ValueError("pool_sample_indices must be 1D.")
        if pool_source_ids.ndim != 1:
            raise ValueError("pool_source_ids must be 1D.")
        if pool_timestamps.ndim != 1:
            raise ValueError("pool_timestamps must be 1D.")
        if pool_targets.ndim != 2:
            raise ValueError("pool_targets must be 2D.")
        if pool_valid_mask.shape != pool_targets.shape:
            raise ValueError("pool_valid_mask must match pool_targets shape.")
        if pool_recency_scores.shape != pool_targets.shape:
            raise ValueError("pool_recency_scores must match pool_targets shape.")
        if pool_past_scores.shape != pool_targets.shape:
            raise ValueError("pool_past_scores must match pool_targets shape.")

        num_pool_rows = pool_targets.shape[0]
        if pool_sample_indices.shape[0] != num_pool_rows:
            raise ValueError("pool_sample_indices length must match pool_targets rows.")
        if pool_source_ids.shape[0] != num_pool_rows:
            raise ValueError("pool_source_ids length must match pool_targets rows.")
        if pool_timestamps.shape[0] != num_pool_rows:
            raise ValueError("pool_timestamps length must match pool_targets rows.")

        num_pool_cols = int(pool_targets.shape[1]) if pool_targets.ndim == 2 else 0
        if num_pool_cols == 0:
            if num_pool_rows != 0:
                raise ValueError("pool_targets cannot have zero columns when pool rows are non-empty.")
        elif not (0 <= int(self.anchor_col) < num_pool_cols):
            raise ValueError("anchor_col must index a valid pool column.")

    @property
    def num_samples(self) -> int:
        return int(self.sample_query_ids.shape[0])

    @property
    def num_pool_rows(self) -> int:
        return int(self.pool_targets.shape[0])

    @property
    def pool_size(self) -> int:
        return int(self.pool_targets.shape[1]) if self.pool_targets.ndim == 2 else 0

    @classmethod
    def empty(cls) -> "TrainPoolBatch":
        return cls(
            sample_query_ids=np.empty(0, dtype=np.int64),
            pool_sample_indices=np.empty(0, dtype=np.int64),
            pool_source_ids=np.empty(0, dtype=np.int64),
            pool_timestamps=np.empty(0, dtype=np.float64),
            pool_targets=np.empty((0, 0), dtype=np.int64),
            pool_valid_mask=np.empty((0, 0), dtype=bool),
            pool_recency_scores=np.empty((0, 0), dtype=np.float64),
            pool_past_scores=np.empty((0, 0), dtype=np.float64),
            anchor_col=0,
        )
