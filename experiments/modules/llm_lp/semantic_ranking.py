"""
Shared runtime helpers for query-time semantic ranking and smoothing.
"""
import numpy as np
import torch

from experiments.modules.heuristic_models import smooth_embeddings_by_time_window_torch


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def build_entity_idx_lookup(entity_id_to_idx):
    try:
        max_entity_id = max(int(eid) for eid in entity_id_to_idx.keys()) if entity_id_to_idx else -1
        num_indexed_entities = len(entity_id_to_idx) if entity_id_to_idx else 0
        if (
            max_entity_id >= 0
            and num_indexed_entities > 0
            and max_entity_id <= max(1024, 16 * num_indexed_entities)
        ):
            lookup = np.full(max_entity_id + 1, -1, dtype=np.int64)
            for entity_id, emb_idx in entity_id_to_idx.items():
                entity_id = int(entity_id)
                if entity_id < 0:
                    continue
                lookup[entity_id] = int(emb_idx)
            return lookup
    except Exception:
        return None
    return None


def prepare_semantic_embeddings_for_runtime(
    embeddings,
    *,
    semantic_use_smoothing,
    semantic_smoothing_device,
    warn_fn=None,
):
    semantic_embeddings_base = embeddings
    semantic_embeddings_for_smoothing = semantic_embeddings_base
    if not semantic_use_smoothing:
        return semantic_embeddings_base, semantic_embeddings_for_smoothing

    try:
        target_device = torch.device(semantic_smoothing_device)
        if isinstance(semantic_embeddings_base, np.ndarray):
            semantic_embeddings_for_smoothing = torch.from_numpy(semantic_embeddings_base).to(
                target_device
            )
        else:
            semantic_embeddings_for_smoothing = semantic_embeddings_base.to(target_device)
    except Exception as exc:
        if warn_fn is not None:
            warn_fn(
                "Failed to pre-stage semantic embeddings on "
                f"{semantic_smoothing_device} ({exc}); falling back to host-backed smoothing input."
            )
        semantic_embeddings_for_smoothing = semantic_embeddings_base

    return semantic_embeddings_base, semantic_embeddings_for_smoothing


def make_semantic_cache_state():
    return {
        "timestamp": None,
        "embeddings": None,
        "hits": 0,
        "misses": 0,
    }


def get_query_time_semantic_embeddings(
    timestamp,
    *,
    semantic_use_smoothing,
    semantic_embeddings_base,
    semantic_embeddings_for_smoothing,
    mapped_u_vals,
    mapped_i_vals,
    ts_vals_float,
    smooth_time_window,
    smooth_steps,
    smooth_decay_gamma,
    smooth_undirected,
    semantic_smoothing_device,
    cache_state,
):
    if not semantic_use_smoothing:
        return semantic_embeddings_base

    cached_timestamp = cache_state.get("timestamp")
    cached_embeddings = cache_state.get("embeddings")
    if cached_timestamp == timestamp and cached_embeddings is not None:
        cache_state["hits"] = int(cache_state.get("hits", 0)) + 1
        return cached_embeddings

    cache_state["misses"] = int(cache_state.get("misses", 0)) + 1
    cutoff = int(np.searchsorted(ts_vals_float, float(timestamp), side="left"))
    if cutoff <= 0:
        cache_state["timestamp"] = timestamp
        cache_state["embeddings"] = semantic_embeddings_base
        return semantic_embeddings_base

    window_start = float(timestamp) - float(smooth_time_window)
    left = int(np.searchsorted(ts_vals_float, window_start, side="left"))
    src_hist = mapped_u_vals[left:cutoff]
    dst_hist = mapped_i_vals[left:cutoff]
    t_hist = ts_vals_float[left:cutoff]
    valid = (src_hist >= 0) & (dst_hist >= 0)
    src_hist = src_hist[valid]
    dst_hist = dst_hist[valid]
    t_hist = t_hist[valid]

    if len(src_hist) == 0:
        current_embeddings = semantic_embeddings_for_smoothing
    else:
        with torch.inference_mode():
            smoothed = smooth_embeddings_by_time_window_torch(
                embeddings=semantic_embeddings_for_smoothing,
                src_node_ids=src_hist,
                dst_node_ids=dst_hist,
                node_interact_times=t_hist,
                time_window=smooth_time_window,
                reference_time=float(timestamp),
                num_steps=smooth_steps,
                symmetric_norm=True,
                decay_gamma=smooth_decay_gamma,
                residual_alpha=0.0,
                undirected=smooth_undirected,
                device=semantic_smoothing_device,
                debug=False,
            )
        if isinstance(smoothed, torch.Tensor):
            smoothed = torch.nn.functional.normalize(smoothed, p=2, dim=1)
            current_embeddings = smoothed
        else:
            current_embeddings = _l2_normalize_rows(smoothed)

    cache_state["timestamp"] = timestamp
    cache_state["embeddings"] = current_embeddings
    return current_embeddings


def compute_semantic_dot_scores(
    ref_id,
    candidate_node_ids,
    semantic_embs,
    *,
    entity_id_to_idx,
    entity_idx_lookup=None,
):
    num_candidates = len(candidate_node_ids)
    if num_candidates == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=bool)

    if isinstance(semantic_embs, torch.Tensor):
        scores = torch.full(
            (num_candidates,),
            -1.0,
            dtype=torch.float32,
            device=semantic_embs.device,
        )
        valid_mask = torch.zeros(
            (num_candidates,),
            dtype=torch.bool,
            device=semantic_embs.device,
        )
    else:
        scores = np.full(num_candidates, -1.0, dtype=np.float32)
        valid_mask = np.zeros(num_candidates, dtype=bool)
    if semantic_embs is None or (entity_id_to_idx is None and entity_idx_lookup is None):
        return scores, valid_mask

    ref_id = int(ref_id)
    ref_idx = None
    if (
        entity_idx_lookup is not None
        and ref_id >= 0
        and ref_id < int(entity_idx_lookup.shape[0])
    ):
        lookup_idx = int(entity_idx_lookup[ref_id])
        if lookup_idx >= 0:
            ref_idx = lookup_idx
    if ref_idx is None and entity_id_to_idx is not None:
        ref_idx = entity_id_to_idx.get(ref_id)
    if ref_idx is None:
        return scores, valid_mask

    candidate_ids = np.asarray(candidate_node_ids, dtype=np.int64)
    if entity_idx_lookup is not None:
        in_range = (candidate_ids >= 0) & (candidate_ids < int(entity_idx_lookup.shape[0]))
        mapped_indices = np.full(candidate_ids.shape, -1, dtype=np.int64)
        mapped_indices[in_range] = entity_idx_lookup[candidate_ids[in_range]]
        valid_mask_np = mapped_indices >= 0
    else:
        mapped_indices = np.full(candidate_ids.shape, -1, dtype=np.int64)
        for pos, node_id in enumerate(candidate_ids):
            other_idx = entity_id_to_idx.get(int(node_id))
            if other_idx is not None:
                mapped_indices[pos] = int(other_idx)
        valid_mask_np = mapped_indices >= 0

    if not np.any(valid_mask_np):
        return scores, valid_mask

    valid_pos_arr = np.flatnonzero(valid_mask_np)
    valid_entity_indices = mapped_indices[valid_pos_arr]

    if isinstance(semantic_embs, torch.Tensor):
        idx_tensor = torch.as_tensor(
            valid_entity_indices,
            dtype=torch.long,
            device=semantic_embs.device,
        )
        ref_vector = semantic_embs[int(ref_idx)]
        valid_pos_tensor = torch.as_tensor(
            valid_pos_arr,
            dtype=torch.long,
            device=semantic_embs.device,
        )
        with torch.inference_mode():
            sim_tensor = semantic_embs.index_select(0, idx_tensor).matmul(ref_vector)
        scores.index_copy_(0, valid_pos_tensor, sim_tensor.to(dtype=torch.float32))
        valid_mask.index_fill_(0, valid_pos_tensor, True)
    else:
        idx_arr = np.asarray(valid_entity_indices, dtype=np.int64)
        ref_vector = np.asarray(semantic_embs[int(ref_idx)])
        cand_vectors = np.asarray(semantic_embs[idx_arr])
        sim_values = cand_vectors.dot(ref_vector).astype(np.float32, copy=False)
        scores[valid_pos_arr] = sim_values
        valid_mask[valid_pos_arr] = True
    return scores, valid_mask
