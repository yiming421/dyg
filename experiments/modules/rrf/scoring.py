"""
Standard RRF scoring helpers shared by LLM and heuristic evaluation paths.
"""
from bisect import bisect_right, insort
import time

import numpy as np
from tqdm import tqdm

from .common import group_sample_indices_by_query
from .utils import descending_ranks, reciprocal_rank_fusion_from_ranks

DEFAULT_RRF_HEURISTICS = (
    "recency",
    "popularity",
    "past_interactions",
    "resource_allocation",
)
SUPPORTED_RRF_HEURISTICS = DEFAULT_RRF_HEURISTICS + (
    "recent_degree",
    "global_recency",
    "semantic_smoothing",
)
HEURISTIC_SCORE_KEYS = {
    "recency": "recency_score",
    "popularity": "popularity_score",
    "past_interactions": "past_interactions_score",
    "resource_allocation": "resource_allocation_score",
    "recent_degree": "recent_degree_score",
    "global_recency": "global_recency_score",
    "semantic_smoothing": "semantic_smoothing_score",
}
HEURISTIC_RANK_KEYS = {
    "recency": "recency_rank",
    "popularity": "popularity_rank",
    "past_interactions": "past_interactions_rank",
    "resource_allocation": "resource_allocation_rank",
    "recent_degree": "recent_degree_rank",
    "global_recency": "global_recency_rank",
    "semantic_smoothing": "semantic_smoothing_rank",
}


def normalize_rrf_heuristics(rrf_heuristics=None):
    if rrf_heuristics is None:
        return DEFAULT_RRF_HEURISTICS

    if isinstance(rrf_heuristics, str):
        raw_items = [item.strip().lower() for item in rrf_heuristics.split(",")]
    else:
        raw_items = []
        for item in rrf_heuristics:
            if item is None:
                continue
            raw_items.append(str(item).strip().lower())

    normalized = []
    seen = set()
    for item in raw_items:
        if not item or item in seen:
            continue
        if item not in SUPPORTED_RRF_HEURISTICS:
            raise ValueError(
                f"Unsupported RRF heuristic '{item}'. "
                f"Use a subset of {list(SUPPORTED_RRF_HEURISTICS)}."
            )
        normalized.append(item)
        seen.add(item)

    if not normalized:
        raise ValueError(
            "At least one RRF heuristic must be selected. "
            f"Supported values: {list(SUPPORTED_RRF_HEURISTICS)}."
        )
    return tuple(normalized)


def _log_stage_timing(enabled, label, start_t, *, chunk=None, size=None):
    if not enabled:
        return
    prefix = "[RRF]"
    parts = []
    if chunk is not None:
        parts.append(f"chunk={chunk}")
    if size is not None:
        parts.append(f"size={size}")
    parts.append(f"{label}={time.perf_counter() - start_t:.2f}s")
    print(f"{prefix} " + " ".join(parts), flush=True)


def _normalize_embeddings(embeddings):
    import torch

    if isinstance(embeddings, torch.Tensor):
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    emb_arr = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb_arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb_arr / norms


def _compute_semantic_smoothing_scores_for_pairs(
    sources,
    targets,
    timestamps,
    *,
    semantic_context,
    show_progress=True,
    progress_desc="RRF scoring: semantic_smoothing",
):
    if semantic_context is None:
        raise ValueError(
            "semantic_context is required when 'semantic_smoothing' is selected for RRF."
        )

    import torch

    from ..heuristic_models import smooth_embeddings_by_time_window_torch

    sources = np.asarray(sources, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    scores = np.full(sources.shape[0], -1.0, dtype=np.float64)
    if sources.size == 0:
        return scores

    base_embeddings = semantic_context["embeddings"]
    entity_id_to_idx = semantic_context.get("entity_id_to_idx")
    entity_idx_lookup = semantic_context.get("entity_idx_lookup")
    history_src_indices = semantic_context["history_src_indices"]
    history_dst_indices = semantic_context["history_dst_indices"]
    history_timestamps = semantic_context["history_timestamps"]
    smooth_time_window = float(semantic_context["smooth_time_window"])
    smooth_steps = int(semantic_context["smooth_steps"])
    smooth_decay_gamma = semantic_context["smooth_decay_gamma"]
    smooth_undirected = bool(semantic_context["smooth_undirected"])
    semantic_device = str(semantic_context.get("device", "cpu"))

    timestamp_order = np.argsort(timestamps, kind="stable")
    ordered_timestamps = timestamps[timestamp_order]
    last_timestamp = None
    last_embeddings = None

    def get_embeddings_at_time(timestamp):
        nonlocal last_timestamp, last_embeddings

        if last_timestamp == timestamp and last_embeddings is not None:
            return last_embeddings

        cutoff = int(np.searchsorted(history_timestamps, timestamp, side="left"))
        if cutoff <= 0:
            current_embeddings = base_embeddings
        else:
            window_start = float(timestamp) - smooth_time_window
            left = int(np.searchsorted(history_timestamps, window_start, side="left"))
            src_hist = history_src_indices[left:cutoff]
            dst_hist = history_dst_indices[left:cutoff]
            ts_hist = history_timestamps[left:cutoff]
            valid = (src_hist >= 0) & (dst_hist >= 0)
            src_hist = src_hist[valid]
            dst_hist = dst_hist[valid]
            ts_hist = ts_hist[valid]

            if src_hist.size == 0:
                current_embeddings = base_embeddings
            else:
                with torch.inference_mode():
                    current_embeddings = smooth_embeddings_by_time_window_torch(
                        embeddings=base_embeddings,
                        src_node_ids=src_hist,
                        dst_node_ids=dst_hist,
                        node_interact_times=ts_hist,
                        time_window=smooth_time_window,
                        reference_time=float(timestamp),
                        num_steps=smooth_steps,
                        symmetric_norm=True,
                        decay_gamma=smooth_decay_gamma,
                        residual_alpha=0.0,
                        undirected=smooth_undirected,
                        device=semantic_device,
                        debug=False,
                    )
                current_embeddings = _normalize_embeddings(current_embeddings)

        last_timestamp = timestamp
        last_embeddings = current_embeddings
        return current_embeddings

    iter_indices = range(timestamp_order.shape[0])
    if show_progress:
        iter_indices = tqdm(
            iter_indices,
            total=timestamp_order.shape[0],
            desc=progress_desc,
            ncols=100,
        )

    ptr = 0
    while ptr < timestamp_order.shape[0]:
        order_pos = int(ptr)
        ts_value = ordered_timestamps[order_pos]
        group_end = order_pos + 1
        while (
            group_end < timestamp_order.shape[0]
            and ordered_timestamps[group_end] == ts_value
        ):
            group_end += 1

        group_indices = timestamp_order[order_pos:group_end]
        semantic_embeddings = get_embeddings_at_time(float(ts_value))

        group_sources = sources[group_indices]
        group_targets = targets[group_indices]

        mapped_sources = np.full(group_sources.shape, -1, dtype=np.int64)
        mapped_targets = np.full(group_targets.shape, -1, dtype=np.int64)
        if entity_idx_lookup is not None:
            src_mask = (group_sources >= 0) & (group_sources < int(entity_idx_lookup.shape[0]))
            tgt_mask = (group_targets >= 0) & (group_targets < int(entity_idx_lookup.shape[0]))
            mapped_sources[src_mask] = entity_idx_lookup[group_sources[src_mask]]
            mapped_targets[tgt_mask] = entity_idx_lookup[group_targets[tgt_mask]]
        else:
            for idx, node_id in enumerate(group_sources):
                mapped_sources[idx] = int(entity_id_to_idx.get(int(node_id), -1))
            for idx, node_id in enumerate(group_targets):
                mapped_targets[idx] = int(entity_id_to_idx.get(int(node_id), -1))

        valid = (mapped_sources >= 0) & (mapped_targets >= 0)
        if np.any(valid):
            if isinstance(semantic_embeddings, torch.Tensor):
                src_tensor = torch.as_tensor(
                    mapped_sources[valid],
                    dtype=torch.long,
                    device=semantic_embeddings.device,
                )
                tgt_tensor = torch.as_tensor(
                    mapped_targets[valid],
                    dtype=torch.long,
                    device=semantic_embeddings.device,
                )
                with torch.inference_mode():
                    sim_tensor = (
                        semantic_embeddings.index_select(0, src_tensor)
                        * semantic_embeddings.index_select(0, tgt_tensor)
                    ).sum(dim=1)
                scores[group_indices[valid]] = sim_tensor.detach().cpu().numpy().astype(np.float64)
            else:
                src_vecs = semantic_embeddings[mapped_sources[valid]]
                tgt_vecs = semantic_embeddings[mapped_targets[valid]]
                scores[group_indices[valid]] = np.sum(src_vecs * tgt_vecs, axis=1).astype(np.float64)

        if show_progress:
            iter_indices.update(group_end - order_pos)
        ptr = group_end

    if show_progress:
        iter_indices.close()

    return scores


def compute_selected_heuristic_scores_for_samples(
    samples,
    neighbor_sampler,
    *,
    heuristics,
    recent_degree_window=30.0,
    use_gpu_heuristics=False,
    score_batch_size=200000,
    show_progress=True,
    progress_desc="Heuristic scoring",
    semantic_context=None,
    directed_src_node_ids=None,
    directed_dst_node_ids=None,
    directed_node_interact_times=None,
):
    supported_prompt_heuristics = SUPPORTED_RRF_HEURISTICS + ("itemcf", "usercf")
    raw_heuristics = [item.strip().lower() for item in str(heuristics).split(",")] if isinstance(heuristics, str) else [
        str(item).strip().lower() for item in heuristics if item is not None
    ]
    selected_heuristics = tuple(dict.fromkeys(item for item in raw_heuristics if item))
    unknown = [item for item in selected_heuristics if item not in supported_prompt_heuristics]
    if unknown:
        raise ValueError(
            f"Unsupported selected heuristic(s): {', '.join(unknown)}. "
            f"Use a subset of {list(supported_prompt_heuristics)}."
        )
    if not samples:
        return {name: np.array([], dtype=np.float64) for name in selected_heuristics}

    from ..heuristic_models import (
        score_links_by_common_neighbors,
        score_links_by_global_recency,
        score_links_by_itemcf_cosine,
        score_links_by_past_interactions,
        score_links_by_popularity,
        score_links_by_recent_degree,
        score_links_by_recency,
        score_links_by_usercf_cosine,
    )

    total = len(samples)
    if show_progress:
        print(
            f"[RRF] Preparing selected heuristic arrays: "
            f"samples={total}, heuristics={','.join(selected_heuristics)}, "
            f"score_batch_size={int(score_batch_size)}",
            flush=True,
        )
    flat_src = np.empty(total, dtype=np.int64)
    flat_tgt = np.empty(total, dtype=np.int64)
    flat_ts = np.empty(total, dtype=np.float64)
    for idx, sample in enumerate(samples):
        flat_src[idx] = int(sample["source_id"])
        flat_tgt[idx] = int(sample["target_id"])
        flat_ts[idx] = float(sample["timestamp"])

    outputs = {}
    raw_recency = np.empty(total, dtype=np.float64) if "recency" in selected_heuristics else None
    if "popularity" in selected_heuristics:
        outputs["popularity"] = np.empty(total, dtype=np.float64)
    if "recent_degree" in selected_heuristics:
        outputs["recent_degree"] = np.empty(total, dtype=np.float64)
    if "past_interactions" in selected_heuristics:
        outputs["past_interactions"] = np.empty(total, dtype=np.float64)
    raw_ra = np.empty(total, dtype=np.float64) if "resource_allocation" in selected_heuristics else None
    if "global_recency" in selected_heuristics:
        outputs["global_recency"] = np.empty(total, dtype=np.float64)
    if "itemcf" in selected_heuristics:
        outputs["itemcf"] = np.empty(total, dtype=np.float64)
    if "usercf" in selected_heuristics:
        outputs["usercf"] = np.empty(total, dtype=np.float64)

    needs_directed_history = "itemcf" in selected_heuristics or "usercf" in selected_heuristics
    if needs_directed_history:
        directed_src_node_ids = directed_src_node_ids if directed_src_node_ids is not None else getattr(neighbor_sampler, "src_node_ids", None)
        directed_dst_node_ids = directed_dst_node_ids if directed_dst_node_ids is not None else getattr(neighbor_sampler, "dst_node_ids", None)
        directed_node_interact_times = directed_node_interact_times if directed_node_interact_times is not None else getattr(neighbor_sampler, "node_interact_times", None)
        if directed_src_node_ids is None or directed_dst_node_ids is None or directed_node_interact_times is None:
            data = getattr(neighbor_sampler, "data", None)
            directed_src_node_ids = getattr(data, "src_node_ids", directed_src_node_ids)
            directed_dst_node_ids = getattr(data, "dst_node_ids", directed_dst_node_ids)
            directed_node_interact_times = getattr(data, "node_interact_times", directed_node_interact_times)
        if directed_src_node_ids is None or directed_dst_node_ids is None or directed_node_interact_times is None:
            raise ValueError("itemcf/usercf prompt heuristics require directed history arrays on the neighbor sampler.")

    chunk_starts = range(0, total, score_batch_size)
    if show_progress:
        chunk_total = (total + score_batch_size - 1) // score_batch_size
        print(
            f"[RRF] Selected heuristic scoring chunks: {chunk_total}",
            flush=True,
        )
        chunk_starts = tqdm(
            chunk_starts,
            total=chunk_total,
            desc=progress_desc,
            ncols=100,
        )

    for chunk_idx, start in enumerate(chunk_starts, start=1):
        end = min(total, start + score_batch_size)
        sl = slice(start, end)
        chunk_size = end - start
        if show_progress:
            print(
                f"[RRF] selected chunk={chunk_idx} size={chunk_size} starting heuristic kernels",
                flush=True,
            )
        if raw_recency is not None:
            stage_t0 = time.perf_counter()
            raw_recency[sl] = score_links_by_recency(
                neighbor_sampler,
                flat_src[sl],
                flat_tgt[sl],
                flat_ts[sl],
            )
            _log_stage_timing(show_progress, "recency", stage_t0, chunk=chunk_idx, size=chunk_size)
        if "popularity" in outputs:
            stage_t0 = time.perf_counter()
            outputs["popularity"][sl] = score_links_by_popularity(
                neighbor_sampler,
                flat_src[sl],
                flat_tgt[sl],
                flat_ts[sl],
            )
            _log_stage_timing(show_progress, "popularity", stage_t0, chunk=chunk_idx, size=chunk_size)
        if "recent_degree" in outputs:
            stage_t0 = time.perf_counter()
            outputs["recent_degree"][sl] = score_links_by_recent_degree(
                neighbor_sampler,
                flat_src[sl],
                flat_tgt[sl],
                flat_ts[sl],
                window=float(recent_degree_window),
                mode="target",
            )
            _log_stage_timing(show_progress, "recent_degree", stage_t0, chunk=chunk_idx, size=chunk_size)
        if "past_interactions" in outputs:
            stage_t0 = time.perf_counter()
            outputs["past_interactions"][sl] = score_links_by_past_interactions(
                neighbor_sampler,
                flat_src[sl],
                flat_tgt[sl],
                flat_ts[sl],
            )
            _log_stage_timing(show_progress, "past_interactions", stage_t0, chunk=chunk_idx, size=chunk_size)
        if raw_ra is not None:
            stage_t0 = time.perf_counter()
            raw_ra[sl] = score_links_by_common_neighbors(
                neighbor_sampler,
                flat_src[sl],
                flat_tgt[sl],
                flat_ts[sl],
                mode="ra",
                use_gpu=use_gpu_heuristics,
            )
            _log_stage_timing(show_progress, "resource_allocation", stage_t0, chunk=chunk_idx, size=chunk_size)
        if "global_recency" in outputs:
            stage_t0 = time.perf_counter()
            outputs["global_recency"][sl] = score_links_by_global_recency(
                neighbor_sampler,
                flat_src[sl],
                flat_tgt[sl],
                flat_ts[sl],
            )
            _log_stage_timing(show_progress, "global_recency", stage_t0, chunk=chunk_idx, size=chunk_size)
        if "itemcf" in outputs:
            stage_t0 = time.perf_counter()
            outputs["itemcf"][sl] = score_links_by_itemcf_cosine(
                neighbor_sampler,
                flat_src[sl],
                flat_tgt[sl],
                flat_ts[sl],
                directed_src_node_ids=directed_src_node_ids,
                directed_dst_node_ids=directed_dst_node_ids,
                directed_node_interact_times=directed_node_interact_times,
            )
            _log_stage_timing(show_progress, "itemcf", stage_t0, chunk=chunk_idx, size=chunk_size)
        if "usercf" in outputs:
            stage_t0 = time.perf_counter()
            outputs["usercf"][sl] = score_links_by_usercf_cosine(
                neighbor_sampler,
                flat_src[sl],
                flat_tgt[sl],
                flat_ts[sl],
                directed_src_node_ids=directed_src_node_ids,
                directed_dst_node_ids=directed_dst_node_ids,
                directed_node_interact_times=directed_node_interact_times,
            )
            _log_stage_timing(show_progress, "usercf", stage_t0, chunk=chunk_idx, size=chunk_size)

    if raw_recency is not None:
        delta_t = flat_ts - raw_recency
        delta_t[raw_recency == -1] = 1e9
        outputs["recency"] = -np.log1p(delta_t.clip(min=0))
    if raw_ra is not None:
        outputs["resource_allocation"] = raw_ra
    if "semantic_smoothing" in selected_heuristics:
        outputs["semantic_smoothing"] = _compute_semantic_smoothing_scores_for_pairs(
            flat_src,
            flat_tgt,
            flat_ts,
            semantic_context=semantic_context,
            show_progress=show_progress,
            progress_desc=f"{progress_desc}: semantic_smoothing",
        )

    return outputs


def compute_rrf_scores_for_samples(
    samples,
    neighbor_sampler,
    rrf_k=60,
    use_gpu_heuristics=False,
    score_batch_size=200000,
    recent_degree_window=30.0,
    return_components=False,
    rrf_mode="query_local",
    sequential_rank_bins=1024,
    show_progress=True,
    progress_desc="RRF scoring",
    precomputed_recency_scores=None,
    precomputed_past_scores=None,
    rrf_heuristics=None,
    semantic_context=None,
):
    """
    Compute RRF-like scores in batch.

    rrf_mode:
      - query_local: classic per-query RRF over candidates in the same query.
      - sequential_pointwise: online pointwise fusion where each heuristic score is
        converted to a pseudo-rank using only historical score distributions from
        earlier queries (sorted by timestamp, then query id).

    Returns arrays: rrf_scores, ra_scores aligned with samples.
    If return_components=True, also returns per-sample heuristic score/rank dict.
    """
    valid_modes = {"query_local", "sequential_pointwise"}
    if rrf_mode not in valid_modes:
        raise ValueError(f"Unknown rrf_mode={rrf_mode}. Use one of {sorted(valid_modes)}.")
    sequential_rank_bins = max(2, int(sequential_rank_bins))
    selected_heuristics = normalize_rrf_heuristics(rrf_heuristics)

    if not samples:
        if return_components:
            return np.array([]), np.array([]), {}
        return np.array([]), np.array([])

    from ..heuristic_models import (
        score_links_by_common_neighbors,
        score_links_by_global_recency,
        score_links_by_past_interactions,
        score_links_by_popularity,
        score_links_by_recent_degree,
        score_links_by_recency,
    )

    query_to_indices = group_sample_indices_by_query(
        samples,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )

    query_items = []
    total = 0
    for qid, indices in query_to_indices.items():
        src = samples[indices[0]]["source_id"]
        ts = samples[indices[0]]["timestamp"]
        query_items.append((int(qid), indices, src, ts))
        total += len(indices)

    flat_src = np.empty(total, dtype=np.int64)
    flat_tgt = np.empty(total, dtype=np.int64)
    flat_ts = np.empty(total, dtype=np.float64)
    flat_sample_idx = np.empty(total, dtype=np.int64)
    query_offsets = np.zeros(len(query_items) + 1, dtype=np.int64)

    cursor = 0
    query_item_iter = enumerate(query_items)
    if show_progress:
        query_item_iter = enumerate(
            tqdm(query_items, total=len(query_items), desc=f"{progress_desc}: flatten", ncols=100)
        )
    for q_idx, (_, indices, src, ts) in query_item_iter:
        start = cursor
        for sample_idx in indices:
            flat_src[cursor] = src
            flat_tgt[cursor] = samples[sample_idx]["target_id"]
            flat_ts[cursor] = ts
            flat_sample_idx[cursor] = sample_idx
            cursor += 1
        query_offsets[q_idx] = start
    query_offsets[len(query_items)] = cursor

    heuristic_values = {}
    raw_recency = None
    if "recency" in selected_heuristics:
        if precomputed_recency_scores is not None:
            precomputed_recency_scores = np.asarray(precomputed_recency_scores, dtype=np.float64)
            if precomputed_recency_scores.shape[0] != len(samples):
                raise ValueError("precomputed_recency_scores must align with samples.")
            heuristic_values["recency"] = precomputed_recency_scores[flat_sample_idx]
        else:
            raw_recency = np.empty(total, dtype=np.float64)
    if "popularity" in selected_heuristics:
        heuristic_values["popularity"] = np.empty(total, dtype=np.float64)
    if "recent_degree" in selected_heuristics:
        heuristic_values["recent_degree"] = np.empty(total, dtype=np.float64)
    if "past_interactions" in selected_heuristics:
        if precomputed_past_scores is not None:
            precomputed_past_scores = np.asarray(precomputed_past_scores, dtype=np.float64)
            if precomputed_past_scores.shape[0] != len(samples):
                raise ValueError("precomputed_past_scores must align with samples.")
            heuristic_values["past_interactions"] = precomputed_past_scores[flat_sample_idx]
        else:
            heuristic_values["past_interactions"] = np.empty(total, dtype=np.float64)
    s_ra = np.empty(total, dtype=np.float64)
    if "global_recency" in selected_heuristics:
        heuristic_values["global_recency"] = np.empty(total, dtype=np.float64)

    chunk_starts = range(0, total, score_batch_size)
    if show_progress:
        chunk_total = (total + score_batch_size - 1) // score_batch_size
        chunk_starts = tqdm(
            chunk_starts,
            total=chunk_total,
            desc=f"{progress_desc}: heuristics",
            ncols=100,
        )

    for chunk_idx, start in enumerate(chunk_starts, start=1):
        end = min(total, start + score_batch_size)
        sl = slice(start, end)
        chunk_size = end - start
        if show_progress:
            print(
                f"[RRF] chunk={chunk_idx} size={chunk_size} starting heuristic kernels",
                flush=True,
            )

        if raw_recency is not None:
            stage_t0 = time.perf_counter()
            raw_recency[sl] = score_links_by_recency(
                neighbor_sampler,
                flat_src[sl],
                flat_tgt[sl],
                flat_ts[sl],
            )
            _log_stage_timing(show_progress, "recency", stage_t0, chunk=chunk_idx, size=chunk_size)
        if "popularity" in heuristic_values:
            stage_t0 = time.perf_counter()
            heuristic_values["popularity"][sl] = score_links_by_popularity(
                neighbor_sampler,
                flat_src[sl],
                flat_tgt[sl],
                flat_ts[sl],
            )
            _log_stage_timing(show_progress, "popularity", stage_t0, chunk=chunk_idx, size=chunk_size)
        if "recent_degree" in heuristic_values:
            stage_t0 = time.perf_counter()
            heuristic_values["recent_degree"][sl] = score_links_by_recent_degree(
                neighbor_sampler,
                flat_src[sl],
                flat_tgt[sl],
                flat_ts[sl],
                window=float(recent_degree_window),
                mode="target",
            )
            _log_stage_timing(show_progress, "recent_degree", stage_t0, chunk=chunk_idx, size=chunk_size)
        if "past_interactions" in heuristic_values and precomputed_past_scores is None:
            stage_t0 = time.perf_counter()
            heuristic_values["past_interactions"][sl] = score_links_by_past_interactions(
                neighbor_sampler,
                flat_src[sl],
                flat_tgt[sl],
                flat_ts[sl],
            )
            _log_stage_timing(show_progress, "past_interactions", stage_t0, chunk=chunk_idx, size=chunk_size)
        stage_t0 = time.perf_counter()
        s_ra[sl] = score_links_by_common_neighbors(
            neighbor_sampler,
            flat_src[sl],
            flat_tgt[sl],
            flat_ts[sl],
            mode="ra",
            use_gpu=use_gpu_heuristics,
        )
        _log_stage_timing(show_progress, "resource_allocation", stage_t0, chunk=chunk_idx, size=chunk_size)
        if "global_recency" in heuristic_values:
            stage_t0 = time.perf_counter()
            heuristic_values["global_recency"][sl] = score_links_by_global_recency(
                neighbor_sampler,
                flat_src[sl],
                flat_tgt[sl],
                flat_ts[sl],
            )
            _log_stage_timing(show_progress, "global_recency", stage_t0, chunk=chunk_idx, size=chunk_size)

    if raw_recency is not None:
        delta_t = flat_ts - raw_recency
        delta_t[raw_recency == -1] = 1e9
        heuristic_values["recency"] = -np.log1p(delta_t.clip(min=0))

    if "resource_allocation" in selected_heuristics:
        heuristic_values["resource_allocation"] = s_ra

    if "semantic_smoothing" in selected_heuristics:
        heuristic_values["semantic_smoothing"] = _compute_semantic_smoothing_scores_for_pairs(
            flat_src,
            flat_tgt,
            flat_ts,
            semantic_context=semantic_context,
            show_progress=show_progress,
            progress_desc=f"{progress_desc}: semantic_smoothing",
        )

    rrf_scores = np.zeros(len(samples), dtype=np.float64)
    ra_scores = np.zeros(len(samples), dtype=np.float64)
    selected_score_buffers = {
        name: np.zeros(len(samples), dtype=np.float64) for name in selected_heuristics
    }
    selected_rank_buffers = {
        name: np.zeros(len(samples), dtype=np.int64) for name in selected_heuristics
    }
    rrf_ranks = np.zeros(len(samples), dtype=np.int64)

    def _assign_query_results(start, end, local_scores, local_ranks):
        rank_tensor = np.stack(
            [local_ranks[name] for name in selected_heuristics],
            axis=0,
        )
        local_rrf = reciprocal_rank_fusion_from_ranks(rank_tensor, rrf_k=rrf_k, model_axis=0)
        local_rrf_rank = descending_ranks(local_rrf)
        for local_idx, flat_idx in enumerate(range(start, end)):
            sample_idx = flat_sample_idx[flat_idx]
            rrf_scores[sample_idx] = local_rrf[local_idx]
            ra_scores[sample_idx] = s_ra[flat_idx]
            for name in selected_heuristics:
                selected_score_buffers[name][sample_idx] = local_scores[name][local_idx]
                selected_rank_buffers[name][sample_idx] = local_ranks[name][local_idx]
            rrf_ranks[sample_idx] = local_rrf_rank[local_idx]

    if rrf_mode == "query_local":
        query_iter = range(len(query_items))
        if show_progress:
            query_iter = tqdm(
                query_iter,
                total=len(query_items),
                desc=f"{progress_desc}: fusion",
                ncols=100,
            )
        for q_idx in query_iter:
            start = query_offsets[q_idx]
            end = query_offsets[q_idx + 1]
            if end <= start:
                continue

            local_scores = {
                name: heuristic_values[name][start:end] for name in selected_heuristics
            }
            local_ranks = {
                name: descending_ranks(local_scores[name]) for name in selected_heuristics
            }
            _assign_query_results(start, end, local_scores, local_ranks)
    else:
        histories = {name: [] for name in selected_heuristics}

        def _online_pseudo_ranks(values, history):
            out = np.empty(values.shape[0], dtype=np.int64)
            if len(history) == 0:
                out.fill((sequential_rank_bins + 1) // 2)
                return out

            n_hist = len(history)
            for idx, val in enumerate(values):
                pos = bisect_right(history, float(val))
                pct = pos / n_hist
                rank = int(round((1.0 - pct) * (sequential_rank_bins - 1))) + 1
                rank = max(1, min(sequential_rank_bins, rank))
                out[idx] = rank
            return out

        def _update_history(history, values):
            for val in values:
                insort(history, float(val))

        ordered_query_indices = sorted(
            range(len(query_items)),
            key=lambda idx: (query_items[idx][3], query_items[idx][0], idx),
        )

        ptr = 0
        group_bar = None
        if show_progress:
            group_bar = tqdm(
                total=len(query_items),
                desc=f"{progress_desc}: fusion",
                ncols=100,
            )
        while ptr < len(ordered_query_indices):
            q_idx = ordered_query_indices[ptr]
            ts_anchor = query_items[q_idx][3]
            group = []
            while ptr < len(ordered_query_indices):
                candidate_idx = ordered_query_indices[ptr]
                if query_items[candidate_idx][3] != ts_anchor:
                    break
                group.append(candidate_idx)
                ptr += 1

            pending_values = {name: [] for name in selected_heuristics}

            for q_idx in group:
                start = query_offsets[q_idx]
                end = query_offsets[q_idx + 1]
                if end <= start:
                    continue

                local_scores = {
                    name: heuristic_values[name][start:end] for name in selected_heuristics
                }
                local_ranks = {
                    name: _online_pseudo_ranks(local_scores[name], histories[name])
                    for name in selected_heuristics
                }
                _assign_query_results(start, end, local_scores, local_ranks)

                for name in selected_heuristics:
                    pending_values[name].extend(local_scores[name].tolist())

            for name in selected_heuristics:
                _update_history(histories[name], pending_values[name])
            if group_bar is not None:
                group_bar.update(len(group))

        if group_bar is not None:
            group_bar.close()

    if not return_components:
        return rrf_scores, ra_scores

    component_scores = {"rrf_rank": rrf_ranks}
    for name in selected_heuristics:
        component_scores[HEURISTIC_SCORE_KEYS[name]] = selected_score_buffers[name]
        component_scores[HEURISTIC_RANK_KEYS[name]] = selected_rank_buffers[name]
    return rrf_scores, ra_scores, component_scores


__all__ = [
    "DEFAULT_RRF_HEURISTICS",
    "SUPPORTED_RRF_HEURISTICS",
    "compute_selected_heuristic_scores_for_samples",
    "compute_rrf_scores_for_samples",
    "group_sample_indices_by_query",
    "normalize_rrf_heuristics",
    "_compute_semantic_smoothing_scores_for_pairs",
]
