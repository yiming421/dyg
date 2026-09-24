"""Small identity-keyed routing models for fixed-split experiments.

The normal learned-router path expects a checkpoint model exposing ``predict``.
For an expensive LLM evaluation we often want to freeze a route computed by a
separate model before starting inference.  ``IdentityPriorityModel`` provides
that bridge without recomputing the router or consulting evaluation labels: it
maps a stable pair of integer identity fields carried by DTGB samples to a
precomputed routing priority.
"""

from __future__ import annotations

import numpy as np


class IdentityPriorityModel:
    """Return precomputed priorities for two-column integer identity rows."""

    def __init__(
        self,
        query_ids,
        secondary_ids,
        priorities,
        *,
        default_priority: float | None = None,
    ) -> None:
        query_ids = np.asarray(query_ids, dtype=np.int64)
        secondary_ids = np.asarray(secondary_ids, dtype=np.int64)
        priorities = np.asarray(priorities, dtype=np.float64)
        if not (
            query_ids.ndim == secondary_ids.ndim == priorities.ndim == 1
            and len(query_ids) == len(secondary_ids) == len(priorities)
        ):
            raise ValueError("Identity priorities must be aligned one-dimensional arrays")
        if not np.all(np.isfinite(priorities)):
            raise ValueError("Identity priorities contain NaN or infinity")

        keys = list(zip(query_ids.tolist(), secondary_ids.tolist()))
        grouped: dict[tuple[int, int], list[float]] = {}
        for key, priority in zip(keys, priorities.tolist()):
            grouped.setdefault(key, []).append(float(priority))
        # Most identities are unique. DTGB can occasionally sample the true
        # target as its paired negative, however, making (query_id, target_id)
        # collide. Preserve deterministic row occurrence order instead of using
        # the evaluation label to disambiguate those rows.
        self._priority_by_key = {
            key: values[0] if len(values) == 1 else tuple(values)
            for key, values in grouped.items()
        }
        self.default_priority = (
            None if default_priority is None else float(default_priority)
        )

    def predict(self, features) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != 2:
            raise ValueError(
                "IdentityPriorityModel expects two integer identity columns; "
                f"received shape {features.shape}"
            )
        if not np.all(np.isfinite(features)):
            raise ValueError("Identity routing features contain NaN or infinity")

        query_ids = np.rint(features[:, 0]).astype(np.int64)
        secondary_ids = np.rint(features[:, 1]).astype(np.int64)
        output = np.empty(len(features), dtype=np.float64)
        missing: list[tuple[int, int]] = []
        occurrences: dict[tuple[int, int], int] = {}
        for row_idx, key in enumerate(zip(query_ids.tolist(), secondary_ids.tolist())):
            priority = self._priority_by_key.get(key)
            if priority is None:
                if self.default_priority is None:
                    missing.append(key)
                    continue
                priority = self.default_priority
            if isinstance(priority, (tuple, list, np.ndarray)):
                occurrence = occurrences.get(key, 0)
                if occurrence >= len(priority):
                    missing.append(key)
                    continue
                output[row_idx] = float(priority[occurrence])
                occurrences[key] = occurrence + 1
            else:
                output[row_idx] = float(priority)
        if missing:
            preview = ", ".join(str(key) for key in missing[:5])
            raise ValueError(
                f"No precomputed priority for {len(missing)} sample identities; "
                f"first keys: {preview}"
            )
        return output
