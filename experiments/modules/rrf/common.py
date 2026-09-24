"""
Lightweight shared helpers for RRF-related modules.
"""
from tqdm import tqdm


def group_indices_by_values(values):
    """Group positional indices by value while preserving first-seen order."""
    query_to_indices = {}
    for idx, value in enumerate(values):
        key = int(value)
        if key not in query_to_indices:
            query_to_indices[key] = []
        query_to_indices[key].append(idx)
    return query_to_indices


def group_sample_indices_by_query(samples, show_progress=False, progress_desc="RRF scoring"):
    """Group sample indices by query_id while preserving insertion order."""
    sample_iter = enumerate(samples)
    query_ids = []
    if show_progress:
        sample_iter = enumerate(
            tqdm(samples, total=len(samples), desc=f"{progress_desc}: index", ncols=100)
        )
    for idx, sample in sample_iter:
        query_ids.append(sample.get("query_id", idx))
    return group_indices_by_values(query_ids)


__all__ = ["group_indices_by_values", "group_sample_indices_by_query"]
