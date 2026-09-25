"""
Prompt-context materialization and prompt-side key-signal calibration helpers.
"""
from bisect import bisect_left, bisect_right, insort
from collections import Counter, defaultdict
import time

import numpy as np
import torch
from tqdm import tqdm

from experiments.modules.heuristic_models import (
    score_links_by_common_neighbors,
    score_links_by_recent_degree,
)
from experiments.modules.llm_lp.eval_helpers import load_or_compute_embeddings
from experiments.modules.rrf.common import group_sample_indices_by_query
from experiments.modules.llm_lp.semantic_ranking import (
    build_entity_idx_lookup,
    compute_semantic_dot_scores,
    get_query_time_semantic_embeddings,
    make_semantic_cache_state,
    prepare_semantic_embeddings_for_runtime,
)
import os
import sys

_MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_MODULES_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.DataLoader import Data
from utils.utils import get_neighbor_sampler


def _quantile_pair(sorted_history):
    if not sorted_history:
        return None, None
    arr = np.asarray(sorted_history, dtype=np.float64)
    return float(np.quantile(arr, 0.3)), float(np.quantile(arr, 0.7))


def _bucket_from_thresholds(value, p30, p70, higher_is_better=True, missing_label="Modest"):
    if value is None:
        return missing_label
    if p30 is None or p70 is None or p30 == p70:
        return "Modest"
    if higher_is_better:
        if value >= p70:
            return "High"
        if value <= p30:
            return "Low"
    else:
        if value <= p30:
            return "High"
        if value >= p70:
            return "Low"
    return "Modest"


def _percentile_from_sorted_history(value, sorted_history, invert=False, missing_value=0.0):
    if value is None:
        return float(missing_value)
    n = len(sorted_history)
    if n <= 1:
        pct = 50.0
    else:
        lo = bisect_left(sorted_history, float(value))
        hi = bisect_right(sorted_history, float(value))
        avg_rank = 0.5 * (lo + hi - 1)
        pct = 100.0 * (avg_rank / float(n - 1))
    if invert:
        pct = 100.0 - pct
    return float(max(0.0, min(100.0, pct)))


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _assign_sequential_key_signal_calibration(samples):
    """
    Calibrate prompt-facing key signals without peeking at future eval samples.

    Queries are processed in timestamp order. All queries sharing the same timestamp
    are scored against the same history, then committed together to avoid
    same-timestamp leakage.
    """
    if not samples:
        return

    query_to_indices = group_sample_indices_by_query(samples)
    ordered_queries = sorted(
        [
            (int(qid), indices, int(samples[indices[0]].get("timestamp", 0)))
            for qid, indices in query_to_indices.items()
            if indices
        ],
        key=lambda item: (item[2], item[0]),
    )

    popularity_history = []
    source_pop_history = []
    target_pop_history = []
    interaction_history = []
    recency_history = []
    common_neighbor_history = []
    recent_degree_history = []
    global_recency_history = []
    itemcf_history = []
    usercf_history = []

    ptr = 0
    while ptr < len(ordered_queries):
        ts_anchor = ordered_queries[ptr][2]
        group = []
        while ptr < len(ordered_queries) and ordered_queries[ptr][2] == ts_anchor:
            group.append(ordered_queries[ptr])
            ptr += 1

        pop_p30, pop_p70 = _quantile_pair(popularity_history)
        inter_p30, inter_p70 = _quantile_pair(interaction_history)
        rec_p30, rec_p70 = _quantile_pair(recency_history)
        cn_p30, cn_p70 = _quantile_pair(common_neighbor_history)
        recent_deg_p30, recent_deg_p70 = _quantile_pair(recent_degree_history)
        global_rec_p30, global_rec_p70 = _quantile_pair(global_recency_history)
        itemcf_p30, itemcf_p70 = _quantile_pair(itemcf_history)
        usercf_p30, usercf_p70 = _quantile_pair(usercf_history)

        pending_popularity = []
        pending_source_pop = []
        pending_target_pop = []
        pending_interactions = []
        pending_recency = []
        pending_common_neighbor = []
        pending_recent_degree = []
        pending_global_recency = []
        pending_itemcf = []
        pending_usercf = []

        for _, indices, _ in group:
            for sample_idx in indices:
                sample = samples[sample_idx]

                source_pop_raw = float(sample.get("source_popularity_raw", 0.0))
                target_pop_raw = float(sample.get("target_popularity_raw", 0.0))
                interaction_raw = float(
                    sample.get("num_past_interactions_raw", sample.get("num_past_interactions", 0.0))
                )
                common_neighbor_raw = float(sample.get("common_neighbor_score", 0.0))
                recent_degree_raw = sample.get("heuristic_recent_degree_score")
                if recent_degree_raw is not None:
                    recent_degree_raw = float(recent_degree_raw)
                recency_raw = sample.get("last_interaction_delta")
                if recency_raw is not None:
                    recency_raw = float(recency_raw)
                global_recency_raw = sample.get("heuristic_global_recency_score")
                if global_recency_raw is not None:
                    global_recency_raw = float(global_recency_raw)
                    if global_recency_raw <= -1e14:
                        global_recency_raw = None
                itemcf_raw = sample.get("heuristic_itemcf_score")
                if itemcf_raw is not None:
                    itemcf_raw = float(itemcf_raw)
                usercf_raw = sample.get("heuristic_usercf_score")
                if usercf_raw is not None:
                    usercf_raw = float(usercf_raw)

                sample["source_popularity"] = _bucket_from_thresholds(
                    source_pop_raw,
                    pop_p30,
                    pop_p70,
                    higher_is_better=True,
                )
                sample["target_popularity"] = _bucket_from_thresholds(
                    target_pop_raw,
                    pop_p30,
                    pop_p70,
                    higher_is_better=True,
                )
                sample["num_past_interactions"] = _bucket_from_thresholds(
                    interaction_raw,
                    inter_p30,
                    inter_p70,
                    higher_is_better=True,
                )
                sample["common_neighbor_level"] = _bucket_from_thresholds(
                    common_neighbor_raw,
                    cn_p30,
                    cn_p70,
                    higher_is_better=True,
                )
                sample["recent_degree_level"] = _bucket_from_thresholds(
                    recent_degree_raw,
                    recent_deg_p30,
                    recent_deg_p70,
                    higher_is_better=True,
                    missing_label="Low",
                )
                sample["last_interaction_str"] = _bucket_from_thresholds(
                    recency_raw,
                    rec_p30,
                    rec_p70,
                    higher_is_better=False,
                    missing_label="No prior interactions",
                )
                sample["global_recency_level"] = _bucket_from_thresholds(
                    global_recency_raw,
                    global_rec_p30,
                    global_rec_p70,
                    higher_is_better=True,
                    missing_label="No prior target interactions",
                )
                sample["itemcf_level"] = _bucket_from_thresholds(
                    itemcf_raw,
                    itemcf_p30,
                    itemcf_p70,
                    higher_is_better=True,
                    missing_label="Low",
                )
                sample["usercf_level"] = _bucket_from_thresholds(
                    usercf_raw,
                    usercf_p30,
                    usercf_p70,
                    higher_is_better=True,
                    missing_label="Low",
                )

                sample["source_popularity_pct"] = _percentile_from_sorted_history(
                    source_pop_raw,
                    source_pop_history,
                )
                sample["target_popularity_pct"] = _percentile_from_sorted_history(
                    target_pop_raw,
                    target_pop_history,
                )
                sample["num_past_interactions_pct"] = _percentile_from_sorted_history(
                    interaction_raw,
                    interaction_history,
                )
                sample["common_neighbor_score_pct"] = _percentile_from_sorted_history(
                    common_neighbor_raw,
                    common_neighbor_history,
                )
                sample["recent_degree_pct"] = _percentile_from_sorted_history(
                    recent_degree_raw,
                    recent_degree_history,
                    missing_value=0.0,
                )
                sample["last_interaction_recency_pct"] = _percentile_from_sorted_history(
                    recency_raw,
                    recency_history,
                    invert=True,
                    missing_value=0.0,
                )
                sample["global_recency_pct"] = _percentile_from_sorted_history(
                    global_recency_raw,
                    global_recency_history,
                    missing_value=0.0,
                )
                sample["itemcf_pct"] = _percentile_from_sorted_history(
                    itemcf_raw,
                    itemcf_history,
                    missing_value=0.0,
                )
                sample["usercf_pct"] = _percentile_from_sorted_history(
                    usercf_raw,
                    usercf_history,
                    missing_value=0.0,
                )

                pending_popularity.extend([source_pop_raw, target_pop_raw])
                pending_source_pop.append(source_pop_raw)
                pending_target_pop.append(target_pop_raw)
                pending_interactions.append(interaction_raw)
                pending_common_neighbor.append(common_neighbor_raw)
                if recent_degree_raw is not None:
                    pending_recent_degree.append(recent_degree_raw)
                if recency_raw is not None:
                    pending_recency.append(recency_raw)
                if global_recency_raw is not None:
                    pending_global_recency.append(global_recency_raw)
                if itemcf_raw is not None:
                    pending_itemcf.append(itemcf_raw)
                if usercf_raw is not None:
                    pending_usercf.append(usercf_raw)

        for value in pending_popularity:
            insort(popularity_history, float(value))
        for value in pending_source_pop:
            insort(source_pop_history, float(value))
        for value in pending_target_pop:
            insort(target_pop_history, float(value))
        for value in pending_interactions:
            insort(interaction_history, float(value))
        for value in pending_recency:
            insort(recency_history, float(value))
        for value in pending_common_neighbor:
            insort(common_neighbor_history, float(value))
        for value in pending_recent_degree:
            insort(recent_degree_history, float(value))
        for value in pending_global_recency:
            insort(global_recency_history, float(value))
        for value in pending_itemcf:
            insort(itemcf_history, float(value))
        for value in pending_usercf:
            insort(usercf_history, float(value))


def _assign_contextual_key_signal_calibration(
    samples,
    history_as_source,
    history_as_target,
    edge_u_vals,
    edge_i_vals,
    edge_ts_vals,
    neighbor_sampler=None,
    use_gpu_heuristics=True,
    heuristic_recent_degree_window=30.0,
):
    """
    Calibrate prompt-facing key signals against local temporal reference pools.

    Popularity is compared against all active entities before timestamp t.
    Pairwise signals are compared against the source node's prior counterpart
    distribution before t.
    """
    if not samples:
        return

    query_to_indices = group_sample_indices_by_query(samples)
    ordered_queries = sorted(
        [
            (int(qid), indices, int(samples[indices[0]].get("timestamp", 0)))
            for qid, indices in query_to_indices.items()
            if indices
        ],
        key=lambda item: (item[2], item[0]),
    )

    entity_degree = {}
    entity_last_ts = {}
    edge_ptr = 0
    total_edges = len(edge_ts_vals)

    ptr = 0
    with tqdm(total=len(samples), desc="Contextual key-signal calibration") as pbar:
        while ptr < len(ordered_queries):
            ts_anchor = ordered_queries[ptr][2]
            while edge_ptr < total_edges and float(edge_ts_vals[edge_ptr]) < ts_anchor:
                edge_time = float(edge_ts_vals[edge_ptr])
                for node_id in (int(edge_u_vals[edge_ptr]), int(edge_i_vals[edge_ptr])):
                    entity_degree[node_id] = entity_degree.get(node_id, 0) + 1
                    prev_last = entity_last_ts.get(node_id)
                    if prev_last is None or edge_time > prev_last:
                        entity_last_ts[node_id] = edge_time
                edge_ptr += 1

            # Materialize the active degree distribution once per timestamp group
            # instead of maintaining a sorted Python list on every edge update.
            active_degrees = sorted(float(deg) for deg in entity_degree.values() if deg > 0)

            group = []
            while ptr < len(ordered_queries) and ordered_queries[ptr][2] == ts_anchor:
                group.append(ordered_queries[ptr])
                ptr += 1

            pop_p30, pop_p70 = _quantile_pair(active_degrees)

            for _, indices, timestamp in group:
                sample0 = samples[indices[0]]
                source_id = int(sample0.get("source_id"))

                counterpart_counts = Counter()
                counterpart_last_ts = {}

                out_hist = history_as_source.get(source_id, [])
                out_cutoff = bisect_left(out_hist, timestamp, key=lambda x: x[3])
                for evt in out_hist[:out_cutoff]:
                    other_id = int(evt[2])
                    if other_id == source_id:
                        continue
                    counterpart_counts[other_id] += 1
                    counterpart_last_ts[other_id] = int(evt[3])

                in_hist = history_as_target.get(source_id, [])
                in_cutoff = bisect_left(in_hist, timestamp, key=lambda x: x[3])
                for evt in in_hist[:in_cutoff]:
                    other_id = int(evt[0])
                    if other_id == source_id:
                        continue
                    evt_ts = int(evt[3])
                    prev_last = counterpart_last_ts.get(other_id)
                    if prev_last is None or evt_ts > prev_last:
                        counterpart_last_ts[other_id] = evt_ts

                query_targets = [int(samples[sample_idx]["target_id"]) for sample_idx in indices]
                # Keep bidirectional recency/RA references, but calibrate counts
                # against outgoing counterparts only, matching the directed raw count.
                reference_targets = list(counterpart_last_ts.keys())
                interaction_reference = sorted(
                    float(counterpart_counts.get(target_id, 0.0))
                    for target_id in counterpart_counts
                )
                recency_reference = sorted(
                    float(timestamp - counterpart_last_ts[target_id])
                    if target_id in counterpart_last_ts
                    else float(timestamp + 1)
                    for target_id in reference_targets
                )
                global_recency_reference = sorted(
                    float(last_ts - float(timestamp))
                    for last_ts in entity_last_ts.values()
                    if last_ts is not None
                )
                inter_p30, inter_p70 = _quantile_pair(interaction_reference)
                rec_p30, rec_p70 = _quantile_pair(recency_reference)
                global_rec_p30, global_rec_p70 = _quantile_pair(global_recency_reference)

                ra_by_target = {}
                ra_reference_targets = list(dict.fromkeys(reference_targets + query_targets))
                if neighbor_sampler is not None and ra_reference_targets:
                    src_arr = np.full(len(ra_reference_targets), source_id, dtype=np.int64)
                    tgt_arr = np.asarray(ra_reference_targets, dtype=np.int64)
                    ts_arr = np.full(len(ra_reference_targets), float(timestamp), dtype=np.float64)
                    ra_scores = score_links_by_common_neighbors(
                        neighbor_sampler,
                        src_arr,
                        tgt_arr,
                        ts_arr,
                        mode="ra",
                        use_gpu=use_gpu_heuristics,
                    )
                    ra_by_target = {
                        int(target_id): float(ra_scores[idx])
                        for idx, target_id in enumerate(tgt_arr)
                    }
                ra_distribution_targets = reference_targets if reference_targets else query_targets
                common_neighbor_reference = sorted(
                    float(ra_by_target[target_id])
                    for target_id in ra_distribution_targets
                    if target_id in ra_by_target
                )
                cn_p30, cn_p70 = _quantile_pair(common_neighbor_reference)

                recent_degree_by_target = {}
                recent_degree_reference = []
                recent_degree_targets = list(dict.fromkeys(reference_targets + query_targets))
                if neighbor_sampler is not None and recent_degree_targets:
                    src_arr = np.full(len(recent_degree_targets), source_id, dtype=np.int64)
                    tgt_arr = np.asarray(recent_degree_targets, dtype=np.int64)
                    ts_arr = np.full(len(recent_degree_targets), float(timestamp), dtype=np.float64)
                    recent_scores = score_links_by_recent_degree(
                        neighbor_sampler,
                        src_arr,
                        tgt_arr,
                        ts_arr,
                        window=float(heuristic_recent_degree_window),
                        mode="target",
                    )
                    recent_degree_by_target = {
                        int(target_id): float(recent_scores[idx])
                        for idx, target_id in enumerate(tgt_arr)
                    }
                    recent_distribution_targets = reference_targets if reference_targets else query_targets
                    recent_degree_reference = sorted(
                        float(recent_degree_by_target[target_id])
                        for target_id in recent_distribution_targets
                        if target_id in recent_degree_by_target
                    )
                recent_deg_p30, recent_deg_p70 = _quantile_pair(recent_degree_reference)

                for sample_idx in indices:
                    sample = samples[sample_idx]
                    source_pop_raw = float(sample.get("source_popularity_raw", 0.0))
                    target_pop_raw = float(sample.get("target_popularity_raw", 0.0))
                    interaction_raw = float(
                        sample.get("num_past_interactions_raw", sample.get("num_past_interactions", 0.0))
                    )
                    recency_raw = sample.get("last_interaction_delta")
                    if recency_raw is not None:
                        recency_raw = float(recency_raw)
                    global_recency_raw = sample.get("heuristic_global_recency_score")
                    if global_recency_raw is not None:
                        global_recency_raw = float(global_recency_raw)
                        if global_recency_raw <= -1e14:
                            global_recency_raw = None

                    target_id = int(sample.get("target_id"))
                    common_neighbor_raw = float(
                        ra_by_target.get(target_id, sample.get("common_neighbor_score", 0.0))
                    )
                    recent_degree_raw = recent_degree_by_target.get(
                        target_id, sample.get("heuristic_recent_degree_score")
                    )
                    if recent_degree_raw is not None:
                        recent_degree_raw = float(recent_degree_raw)

                    sample["source_popularity"] = _bucket_from_thresholds(
                        source_pop_raw,
                        pop_p30,
                        pop_p70,
                        higher_is_better=True,
                    )
                    sample["target_popularity"] = _bucket_from_thresholds(
                        target_pop_raw,
                        pop_p30,
                        pop_p70,
                        higher_is_better=True,
                    )
                    sample["num_past_interactions"] = _bucket_from_thresholds(
                        interaction_raw,
                        inter_p30,
                        inter_p70,
                        higher_is_better=True,
                    )
                    sample["common_neighbor_level"] = _bucket_from_thresholds(
                        common_neighbor_raw,
                        cn_p30,
                        cn_p70,
                        higher_is_better=True,
                    )
                    sample["recent_degree_level"] = _bucket_from_thresholds(
                        recent_degree_raw,
                        recent_deg_p30,
                        recent_deg_p70,
                        higher_is_better=True,
                        missing_label="Low",
                    )
                    sample["last_interaction_str"] = _bucket_from_thresholds(
                        recency_raw,
                        rec_p30,
                        rec_p70,
                        higher_is_better=False,
                        missing_label="No prior interactions",
                    )
                    sample["global_recency_level"] = _bucket_from_thresholds(
                        global_recency_raw,
                        global_rec_p30,
                        global_rec_p70,
                        higher_is_better=True,
                        missing_label="No prior target interactions",
                    )

                    sample["source_popularity_pct"] = _percentile_from_sorted_history(
                        source_pop_raw,
                        active_degrees,
                    )
                    sample["target_popularity_pct"] = _percentile_from_sorted_history(
                        target_pop_raw,
                        active_degrees,
                    )
                    sample["num_past_interactions_pct"] = _percentile_from_sorted_history(
                        interaction_raw,
                        interaction_reference,
                    )
                    sample["common_neighbor_score_pct"] = _percentile_from_sorted_history(
                        common_neighbor_raw,
                        common_neighbor_reference,
                    )
                    sample["recent_degree_pct"] = _percentile_from_sorted_history(
                        recent_degree_raw,
                        recent_degree_reference,
                        missing_value=0.0,
                    )
                    sample["last_interaction_recency_pct"] = _percentile_from_sorted_history(
                        recency_raw,
                        recency_reference,
                        invert=True,
                        missing_value=0.0,
                    )
                    sample["global_recency_pct"] = _percentile_from_sorted_history(
                        global_recency_raw,
                        global_recency_reference,
                        missing_value=0.0,
                    )
                    sample["common_neighbor_score"] = common_neighbor_raw
                    if recent_degree_raw is not None:
                        sample["heuristic_recent_degree_score"] = recent_degree_raw

                pbar.update(len(indices))


def calibrate_prompt_key_signals(
    samples,
    *,
    key_signal_reference="sequential_global",
    edges_df=None,
    history_as_source=None,
    history_as_target=None,
    edge_u_vals=None,
    edge_i_vals=None,
    edge_ts_vals=None,
    neighbor_sampler=None,
    use_gpu_heuristics=True,
    heuristic_recent_degree_window=30.0,
):
    from experiments.modules.llm_lp.training_protocol import history_edges_for_samples
    edges_df = history_edges_for_samples(edges_df, samples)
    if not samples:
        return samples

    key_signal_reference = str(key_signal_reference).strip().lower()
    if key_signal_reference not in {"sequential_global", "contextual"}:
        raise ValueError(
            "Unsupported key_signal_reference="
            f"{key_signal_reference}. Use sequential_global or contextual."
        )

    if key_signal_reference != "contextual":
        print("Calibrating key-signal buckets/percentiles sequentially (no future-query leakage)...")
        _assign_sequential_key_signal_calibration(samples)
        return samples

    if (
        history_as_source is None
        or history_as_target is None
        or edge_u_vals is None
        or edge_i_vals is None
        or edge_ts_vals is None
    ):
        if edges_df is None:
            print(
                "Contextual key-signal reference requested, but calibration histories are unavailable; "
                "falling back to sequential_global calibration."
            )
            _assign_sequential_key_signal_calibration(samples)
            return samples

        print("Pre-computing contextual calibration histories...")
        edges_df_sorted = edges_df.sort_values("ts", kind="stable")
        edge_u_vals = edges_df_sorted["u"].values
        edge_i_vals = edges_df_sorted["i"].values
        edge_ts_vals = edges_df_sorted["ts"].values
        history_as_source = defaultdict(list)
        history_as_target = defaultdict(list)
        for u, i, ts in tqdm(
            zip(edge_u_vals, edge_i_vals, edge_ts_vals),
            total=len(edge_u_vals),
            desc="Contextual indexing",
        ):
            event = (int(u), 0, int(i), int(ts))
            history_as_source[int(u)].append(event)
            history_as_target[int(i)].append(event)

    if neighbor_sampler is None:
        if edges_df is None and (edge_u_vals is None or edge_i_vals is None or edge_ts_vals is None):
            print(
                "Contextual key-signal reference requested, but neighbor_sampler is unavailable; "
                "falling back to sequential_global calibration."
            )
            _assign_sequential_key_signal_calibration(samples)
            return samples

        print("Initializing NeighborSampler for contextual key-signal calibration...")
        if edges_df is not None:
            src_node_ids = edges_df["u"].values.astype(np.int64)
            dst_node_ids = edges_df["i"].values.astype(np.int64)
            node_interact_times = edges_df["ts"].values.astype(np.float64)
        else:
            src_node_ids = np.asarray(edge_u_vals, dtype=np.int64)
            dst_node_ids = np.asarray(edge_i_vals, dtype=np.int64)
            node_interact_times = np.asarray(edge_ts_vals, dtype=np.float64)
        full_data = Data(
            src_node_ids=src_node_ids,
            dst_node_ids=dst_node_ids,
            node_interact_times=node_interact_times,
            edge_ids=np.arange(len(src_node_ids), dtype=np.int64),
            labels=np.zeros(len(src_node_ids), dtype=np.int64),
        )
        neighbor_sampler = get_neighbor_sampler(
            data=full_data,
            sample_neighbor_strategy="recent",
            time_scaling_factor=0.0,
            seed=1,
        )

    if neighbor_sampler is None:
        print(
            "Contextual key-signal reference requested, but neighbor_sampler is unavailable; "
            "falling back to sequential_global calibration."
        )
        _assign_sequential_key_signal_calibration(samples)
        return samples

    print(
        "Calibrating key signals with contextual reference pools "
        "(active-entity popularity + source-conditional pairwise baselines)..."
    )
    _assign_contextual_key_signal_calibration(
        samples=samples,
        history_as_source=history_as_source,
        history_as_target=history_as_target,
        edge_u_vals=edge_u_vals,
        edge_i_vals=edge_i_vals,
        edge_ts_vals=edge_ts_vals,
        neighbor_sampler=neighbor_sampler,
        use_gpu_heuristics=use_gpu_heuristics,
        heuristic_recent_degree_window=heuristic_recent_degree_window,
    )
    return samples

def materialize_samples_prompt_context(
    samples,
    *,
    edges_df,
    entity_map,
    history_window=47,
    semantic_history=False,
    semantic_topk=None,
    semantic_history_entity_mode=False,
    common_neighbors_semantic=False,
    semantic_use_smoothing=True,
    semantic_hub_penalty_alpha=0.0,
    semantic_fusion_alpha=0.4,
    semantic_fusion_tau=None,
    semantic_fusion_recency_speed=1.0,
    history_pool_size=None,
    history_pool_window=None,
    history_preserve_recent_k=0,
    embeddings=None,
    entity_id_to_idx=None,
    embedding_model="intfloat/e5-large-v2",
    embedding_cache=None,
    smooth_time_window=50.0,
    smooth_steps=1,
    smooth_decay_gamma=0.1,
    smooth_undirected=True,
    populate_prompt_lists=True,
    calibrate_key_signals=False,
    key_signal_reference="sequential_global",
    heuristic_recent_degree_window=30.0,
    use_gpu_heuristics=True,
    neighbor_sampler=None,
    monitor_label="Materializing prompt context",
):
    """
    Populate structural prompt fields onto an existing sample list.

    When populate_prompt_lists=False, raw structural fields needed for prompt
    key signals are computed, but expensive history/common-neighbor payloads are
    left empty so hybrid full-set preparation can stay lightweight.
    """
    from experiments.modules.llm_lp.training_protocol import history_edges_for_samples
    edges_df = history_edges_for_samples(edges_df, samples)
    if not samples:
        return samples

    if semantic_history_entity_mode:
        semantic_history = True

    stage_profile_enabled = str(monitor_label).strip().lower() == "hybrid prompt materialization"
    stage_profile = {
        "start_time": time.perf_counter(),
        "indexing_sec": 0.0,
        "sample_loop_sec": 0.0,
        "semantic_smoothing_sec": 0.0,
        "source_context_sec": 0.0,
        "target_context_sec": 0.0,
        "source_history_sec": 0.0,
        "target_history_sec": 0.0,
        "pair_stats_sec": 0.0,
        "common_neighbors_sec": 0.0,
        "calibration_sec": 0.0,
        "semantic_cache_hits": 0,
        "semantic_cache_misses": 0,
    }

    populate_prompt_lists = bool(populate_prompt_lists)
    calibrate_key_signals = bool(calibrate_key_signals)
    semantic_ranking_enabled = populate_prompt_lists and (semantic_history or common_neighbors_semantic)
    semantic_smoothing_device = "cpu"
    semantic_embeddings_base = embeddings
    semantic_embeddings_for_smoothing = embeddings
    entity_idx_lookup = None
    semantic_cache_state = make_semantic_cache_state()
    if semantic_ranking_enabled:
        if semantic_topk is None:
            semantic_topk = history_window
        else:
            semantic_topk = min(int(semantic_topk), int(history_window))

    source_history_desc = "most recent"
    target_history_desc = "most recent"
    common_neighbors_desc = "sorted by popularity"
    semantic_rank_fusion_alpha = float(np.clip(semantic_fusion_alpha, 0.0, 1.0))
    fusion_tau_desc = "auto" if semantic_fusion_tau is None else f"{float(semantic_fusion_tau):g}"
    recency_speed_desc = f"{float(semantic_fusion_recency_speed):g}"
    if semantic_history:
        pool_bits = []
        if history_pool_window is not None:
            pool_bits.append(f"last {int(history_pool_window)} time units")
        if history_pool_size is not None:
            pool_bits.append(f"last {int(history_pool_size)} events")
        pool_desc = " and ".join(pool_bits) if pool_bits else "a larger time pool"
        mode_desc = "smoothed E5" if semantic_use_smoothing else "raw E5"
        if semantic_hub_penalty_alpha > 0.0:
            mode_desc = f"{mode_desc}, hub-penalty={semantic_hub_penalty_alpha:g}"
        preserve_desc = ""
        if history_preserve_recent_k > 0 and not semantic_history_entity_mode:
            preserve_desc = f", preserve newest {history_preserve_recent_k}"
        if semantic_history_entity_mode:
            source_history_desc = (
                f"most recent counterpart entities (semantic tie-break, {mode_desc}, "
                f"grouped timestamps, from {pool_desc})"
            )
            target_history_desc = (
                f"most recent counterpart entities (semantic tie-break, {mode_desc}, "
                f"grouped timestamps, from {pool_desc})"
            )
        else:
            source_history_desc = (
                "most semantically related and recent "
                f"(rank-fusion, alpha={semantic_rank_fusion_alpha:g}, "
                f"recency_speed={recency_speed_desc}, tau={fusion_tau_desc}, "
                f"{mode_desc}{preserve_desc}, from {pool_desc})"
            )
            target_history_desc = (
                "most semantically related and recent "
                f"(rank-fusion, alpha={semantic_rank_fusion_alpha:g}, "
                f"recency_speed={recency_speed_desc}, tau={fusion_tau_desc}, "
                f"{mode_desc}{preserve_desc}, from {pool_desc})"
            )
    if common_neighbors_semantic:
        common_neighbors_desc = (
            "selected by semantic similarity to Target, then sorted by recency to Target"
        )

    history_preserve_recent_k = int(max(0, history_preserve_recent_k))
    if history_preserve_recent_k > history_window:
        history_preserve_recent_k = history_window
    if semantic_history and history_pool_size is None and history_pool_window is None:
        history_pool_size = history_window * 5

    if semantic_ranking_enabled:
        semantic_smoothing_device = "cuda" if semantic_use_smoothing and torch.cuda.is_available() else "cpu"
        if embeddings is None or entity_id_to_idx is None:
            embeddings, entity_id_to_idx = load_or_compute_embeddings(
                entity_map,
                embedding_model,
                embedding_cache,
            )
        embeddings = _l2_normalize_rows(embeddings)
        semantic_embeddings_base = embeddings
        entity_idx_lookup = build_entity_idx_lookup(entity_id_to_idx)
        semantic_embeddings_base, semantic_embeddings_for_smoothing = (
            prepare_semantic_embeddings_for_runtime(
                semantic_embeddings_base,
                semantic_use_smoothing=semantic_use_smoothing,
                semantic_smoothing_device=semantic_smoothing_device,
            )
        )
        if semantic_use_smoothing:
            print(
                "Prompt-context enrichment uses query-time smoothed cosine "
                f"(window={smooth_time_window}, steps={smooth_steps}, "
                f"decay_gamma={smooth_decay_gamma}, undirected={smooth_undirected}, "
                f"device={semantic_smoothing_device})."
            )
        else:
            print("Prompt-context enrichment uses original raw-embedding cosine (no smoothing).")

    edges_df_sorted = edges_df.sort_values("ts", kind="stable")
    u_vals = edges_df_sorted["u"].values
    r_vals = edges_df_sorted["r"].values
    i_vals = edges_df_sorted["i"].values
    ts_vals = edges_df_sorted["ts"].values
    ts_vals_float = ts_vals.astype(np.float64, copy=False)

    history_as_source = defaultdict(list)
    history_as_target = defaultdict(list)
    history_as_endpoint = defaultdict(list)
    pair_history = defaultdict(list)
    indexing_start = time.perf_counter()
    for u, r, i, ts in tqdm(
        zip(u_vals, r_vals, i_vals, ts_vals),
        total=len(u_vals),
        desc=f"{monitor_label}: indexing",
    ):
        u = int(u)
        r = int(r)
        i = int(i)
        ts = int(ts)
        event = (u, r, i, ts)
        history_as_source[u].append(event)
        history_as_target[i].append(event)
        history_as_endpoint[u].append(event)
        history_as_endpoint[i].append(event)
        pair_history[(u, i)].append(event)
    stage_profile["indexing_sec"] = time.perf_counter() - indexing_start

    dst_pop_norm_denom = np.log1p(
        float(max((len(value) for value in history_as_target.values()), default=1))
    )

    mapped_u_vals = None
    mapped_i_vals = None
    if semantic_ranking_enabled and semantic_use_smoothing:
        mapped_u_vals = np.array([entity_id_to_idx.get(int(u), -1) for u in u_vals], dtype=np.int64)
        mapped_i_vals = np.array([entity_id_to_idx.get(int(i), -1) for i in i_vals], dtype=np.int64)

    def get_recent_history(event_list, timestamp, limit=10):
        idx = bisect_left(event_list, timestamp, key=lambda x: x[3])
        start = max(0, idx - limit)
        return event_list[start:idx]

    def get_history_pool(event_list, timestamp, pool_size=None, pool_window=None):
        idx = bisect_left(event_list, timestamp, key=lambda x: x[3])
        start = 0
        if pool_size is not None:
            start = max(start, idx - pool_size)
        if pool_window is not None:
            cutoff = timestamp - pool_window
            start = max(start, bisect_left(event_list, cutoff, key=lambda x: x[3]))
        return event_list[start:idx]

    def select_ranked_events_with_recent_preservation(events, rank_scores, top_k, preserve_recent_k):
        if not events or top_k <= 0:
            return []
        if len(events) <= top_k:
            return list(events)

        preserve_recent_k = int(max(0, min(preserve_recent_k, top_k, len(events))))
        if preserve_recent_k <= 0:
            if isinstance(rank_scores, torch.Tensor):
                top_idx = (
                    torch.topk(rank_scores, k=int(top_k), largest=True)
                    .indices.detach()
                    .cpu()
                    .numpy()
                )
            else:
                top_idx = np.argpartition(rank_scores, -top_k)[-top_k:]
            selected = [events[int(i)] for i in top_idx]
            selected.sort(key=lambda x: x[3])
            return selected

        recent_start = len(events) - preserve_recent_k
        recent_idx = np.arange(recent_start, len(events), dtype=np.int64)
        remaining_slots = top_k - preserve_recent_k

        if remaining_slots <= 0 or recent_start <= 0:
            selected = [events[int(i)] for i in recent_idx[-top_k:]]
            selected.sort(key=lambda x: x[3])
            return selected

        older_scores = rank_scores[:recent_start]
        if recent_start <= remaining_slots:
            older_idx = np.arange(recent_start, dtype=np.int64)
        else:
            if isinstance(older_scores, torch.Tensor):
                older_idx = (
                    torch.topk(older_scores, k=int(remaining_slots), largest=True)
                    .indices.detach()
                    .cpu()
                    .numpy()
                )
            else:
                older_idx = np.argpartition(older_scores, -remaining_slots)[-remaining_slots:]

        selected_idx = np.concatenate([older_idx, recent_idx])
        selected = [events[int(i)] for i in selected_idx]
        selected.sort(key=lambda x: x[3])
        return selected

    last_smooth_ts = None
    last_smooth_emb = None

    def get_semantic_embeddings_at_time(timestamp):
        nonlocal last_smooth_emb, last_smooth_ts
        if not semantic_use_smoothing:
            return semantic_embeddings_base

        smooth_start = time.perf_counter() if stage_profile_enabled else None
        current_embeddings = get_query_time_semantic_embeddings(
            timestamp,
            semantic_use_smoothing=semantic_use_smoothing,
            semantic_embeddings_base=semantic_embeddings_base,
            semantic_embeddings_for_smoothing=semantic_embeddings_for_smoothing,
            mapped_u_vals=mapped_u_vals,
            mapped_i_vals=mapped_i_vals,
            ts_vals_float=ts_vals_float,
            smooth_time_window=smooth_time_window,
            smooth_steps=smooth_steps,
            smooth_decay_gamma=smooth_decay_gamma,
            smooth_undirected=smooth_undirected,
            semantic_smoothing_device=semantic_smoothing_device,
            cache_state=semantic_cache_state,
        )
        last_smooth_ts = semantic_cache_state["timestamp"]
        last_smooth_emb = semantic_cache_state["embeddings"]
        if stage_profile_enabled:
            stage_profile["semantic_cache_hits"] = int(semantic_cache_state["hits"])
            stage_profile["semantic_cache_misses"] = int(semantic_cache_state["misses"])
            stage_profile["semantic_smoothing_sec"] += time.perf_counter() - smooth_start
        return current_embeddings

    def select_topk_semantic(
        events,
        ref_id,
        other_id_fn,
        top_k,
        semantic_embs,
        query_ts=None,
        apply_hub_penalty=False,
        other_ids=None,
    ):
        def rank01_desc(values):
            if isinstance(values, torch.Tensor):
                n_values = int(values.numel())
                if n_values == 0:
                    return torch.empty(0, dtype=torch.float32, device=values.device)
                if n_values == 1:
                    return torch.ones(1, dtype=torch.float32, device=values.device)
                order = torch.argsort(values, descending=True)
                pos = torch.empty(n_values, dtype=torch.float32, device=values.device)
                pos.scatter_(
                    0,
                    order,
                    torch.arange(n_values, dtype=torch.float32, device=values.device),
                )
                return 1.0 - (pos / float(n_values - 1))

            n_values = len(values)
            if n_values == 0:
                return np.empty(0, dtype=np.float32)
            if n_values == 1:
                return np.array([1.0], dtype=np.float32)
            order = np.argsort(-values, kind="mergesort")
            pos = np.empty(n_values, dtype=np.int32)
            pos[order] = np.arange(n_values)
            return 1.0 - (pos.astype(np.float32) / float(n_values - 1))

        if not events:
            return []
        if semantic_embs is None or (entity_id_to_idx is None and entity_idx_lookup is None):
            return events[-top_k:] if len(events) > top_k else events
        if other_ids is None:
            if other_id_fn is None:
                other_ids = [int(evt[2]) for evt in events]
            else:
                other_ids = [int(other_id_fn(evt)) for evt in events]
        sims, _ = compute_semantic_dot_scores(
            ref_id,
            other_ids,
            semantic_embs,
            entity_id_to_idx=entity_id_to_idx,
            entity_idx_lookup=entity_idx_lookup,
        )
        if (
            apply_hub_penalty
            and semantic_hub_penalty_alpha > 0.0
            and query_ts is not None
            and dst_pop_norm_denom > 0.0
        ):
            penalties = np.zeros(len(other_ids), dtype=np.float32)
            for idx, other_id in enumerate(other_ids):
                other_hist_as_target = history_as_target.get(other_id, [])
                dst_deg_before_t = bisect_left(other_hist_as_target, query_ts, key=lambda x: x[3])
                pop_norm = np.log1p(float(dst_deg_before_t)) / dst_pop_norm_denom
                penalties[idx] = float(pop_norm)
            if isinstance(sims, torch.Tensor):
                sims = sims - float(semantic_hub_penalty_alpha) * torch.as_tensor(
                    penalties,
                    dtype=torch.float32,
                    device=sims.device,
                )
            else:
                sims = sims - float(semantic_hub_penalty_alpha) * penalties

        if query_ts is not None:
            if isinstance(sims, torch.Tensor):
                evt_ts = torch.as_tensor(
                    [float(evt[3]) for evt in events],
                    dtype=torch.float32,
                    device=sims.device,
                )
                ages = torch.clamp(float(query_ts) - evt_ts, min=0.0)
                if semantic_fusion_tau is None:
                    tau = float(torch.median(ages).item()) if int(ages.numel()) > 0 else 1.0
                else:
                    tau = float(semantic_fusion_tau)
                tau = max(tau, 1e-6)
                speed = max(float(semantic_fusion_recency_speed), 1e-6)
                recency_scores = torch.exp(
                    -speed * (ages / tau)
                ).to(dtype=torch.float32)
            else:
                evt_ts = np.array([float(evt[3]) for evt in events], dtype=np.float32)
                ages = np.maximum(float(query_ts) - evt_ts, 0.0)
                if semantic_fusion_tau is None:
                    tau = float(np.median(ages)) if len(ages) > 0 else 1.0
                else:
                    tau = float(semantic_fusion_tau)
                tau = max(tau, 1e-6)
                speed = max(float(semantic_fusion_recency_speed), 1e-6)
                recency_scores = np.exp(-speed * (ages / tau)).astype(np.float32)
            sim_rank = rank01_desc(sims)
            rec_rank = rank01_desc(recency_scores)
            rank_scores = semantic_rank_fusion_alpha * sim_rank + (1.0 - semantic_rank_fusion_alpha) * rec_rank
        else:
            rank_scores = sims

        return select_ranked_events_with_recent_preservation(
            events=events,
            rank_scores=rank_scores,
            top_k=top_k,
            preserve_recent_k=history_preserve_recent_k,
        )

    def select_topk_entity_series(entity_ts_map, ref_id, top_k, semantic_embs):
        if not entity_ts_map:
            return []

        entity_ids = [int(entity_id) for entity_id in entity_ts_map.keys()]
        scores, _ = compute_semantic_dot_scores(
            ref_id,
            entity_ids,
            semantic_embs,
            entity_id_to_idx=entity_id_to_idx,
            entity_idx_lookup=entity_idx_lookup,
        )
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy().astype(np.float32, copy=False)
        rows = []
        for entity_id, score in zip(entity_ids, scores):
            ts_list = entity_ts_map[int(entity_id)]
            rows.append((int(entity_id), float(score), [int(ts) for ts in ts_list]))

        rows.sort(key=lambda row: (row[2][-1], row[1], len(row[2])), reverse=True)
        if len(rows) > top_k:
            rows = rows[:top_k]
        return [(entity_id, sorted(ts_list)) for entity_id, _, ts_list in rows]

    def select_topk_semantic_nodes(node_ids, ref_id, top_k, semantic_embs):
        if not node_ids:
            return []
        if top_k <= 0 or len(node_ids) <= top_k:
            return list(node_ids)
        if semantic_embs is None or (entity_id_to_idx is None and entity_idx_lookup is None):
            return list(node_ids)

        node_ids_int = [int(node_id) for node_id in node_ids]
        scores, _ = compute_semantic_dot_scores(
            ref_id,
            node_ids_int,
            semantic_embs,
            entity_id_to_idx=entity_id_to_idx,
            entity_idx_lookup=entity_idx_lookup,
        )
        if isinstance(scores, torch.Tensor):
            top_k = int(min(top_k, len(node_ids_int)))
            selected_idx = (
                torch.topk(scores, k=top_k, largest=True)
                .indices.detach()
                .cpu()
                .tolist()
            )
            return [node_ids_int[int(idx)] for idx in selected_idx]

        rows = [(node_id, float(score)) for node_id, score in zip(node_ids_int, scores)]
        rows.sort(key=lambda row: row[1], reverse=True)
        return [node_id for node_id, _ in rows[:top_k]]

    def build_source_history(source_id, target_id, timestamp, semantic_embs=None):
        return build_target_history(target_id, source_id, timestamp, semantic_embs)

    def build_target_history(source_id, target_id, timestamp, semantic_embs=None):
        if semantic_history:
            pool_src = get_history_pool(
                history_as_source[target_id],
                timestamp,
                pool_size=history_pool_size,
                pool_window=history_pool_window,
            )
            pool_tar = get_history_pool(
                history_as_target[target_id],
                timestamp,
                pool_size=history_pool_size,
                pool_window=history_pool_window,
            )
            pool = sorted(pool_src + pool_tar, key=lambda x: x[3])
            if semantic_history_entity_mode:
                entity_ts_map = defaultdict(list)
                for evt in pool:
                    other_id = int(evt[2]) if int(evt[0]) == target_id else int(evt[0])
                    entity_ts_map[other_id].append(int(evt[3]))
                grouped = select_topk_entity_series(
                    entity_ts_map=entity_ts_map,
                    ref_id=source_id,
                    top_k=semantic_topk,
                    semantic_embs=semantic_embs,
                )
                return [], grouped

            def other_id(evt):
                return evt[2] if evt[0] == target_id else evt[0]

            return (
                select_topk_semantic(
                    pool,
                    source_id,
                    other_id,
                    semantic_topk,
                    semantic_embs,
                    query_ts=timestamp,
                    apply_hub_penalty=False,
                ),
                [],
            )

        return get_recent_history(
            history_as_endpoint[target_id], timestamp, limit=history_window
        ), []

    def build_common_neighbors_info(
        common_nodes,
        dst_id,
        src_neighbors_events,
        tar_neighbors_events,
        semantic_embs=None,
        timestamp=None,
    ):
        if not common_nodes:
            return []

        if common_neighbors_semantic:
            selected_semantic_nodes = set(
                select_topk_semantic_nodes(
                    node_ids=list(common_nodes),
                    ref_id=dst_id,
                    top_k=history_window,
                    semantic_embs=semantic_embs,
                )
            )

            rows = []
            for node_id in common_nodes:
                if int(node_id) not in selected_semantic_nodes:
                    continue
                target_ts = int(tar_neighbors_events[node_id][3])
                source_ts = int(src_neighbors_events[node_id][3])
                rows.append((int(node_id), target_ts, source_ts))
            rows.sort(key=lambda row: (row[1], row[2]), reverse=True)
            selected_nodes = [node_id for node_id, _, _ in rows[:history_window]]
            return [
                (node_id, src_neighbors_events[node_id], tar_neighbors_events[node_id])
                for node_id in selected_nodes
            ]

        degree_cache = {}

        def historical_degree(node_id):
            node_id = int(node_id)
            if node_id not in degree_cache:
                src_degree = bisect_left(history_as_source[node_id], timestamp, key=lambda x: x[3])
                dst_degree = bisect_left(history_as_target[node_id], timestamp, key=lambda x: x[3])
                degree_cache[node_id] = int(src_degree + dst_degree)
            return degree_cache[node_id]

        common_list = sorted(list(common_nodes), key=historical_degree)
        top_common = common_list[:history_window]
        return [
            (node_id, src_neighbors_events[node_id], tar_neighbors_events[node_id])
            for node_id in top_common
        ]

    def get_pair_stats(source_id, target_id, timestamp, include_history):
        forward = pair_history.get((source_id, target_id), [])
        reverse = pair_history.get((target_id, source_id), [])
        idx_forward = bisect_left(forward, timestamp, key=lambda x: x[3])
        idx_reverse = bisect_left(reverse, timestamp, key=lambda x: x[3])

        last_ts = None
        if idx_forward > 0:
            last_ts = int(forward[idx_forward - 1][3])
        if idx_reverse > 0:
            reverse_last_ts = int(reverse[idx_reverse - 1][3])
            if last_ts is None or reverse_last_ts > last_ts:
                last_ts = reverse_last_ts

        mutual_history = []
        if include_history and (idx_forward > 0 or idx_reverse > 0):
            mutual_all = sorted(forward[:idx_forward] + reverse[:idx_reverse], key=lambda x: x[3])
            mutual_history = mutual_all[-history_window:]

        # Mutual history/recency include both directions; the count is source -> target.
        return mutual_history, int(idx_forward), last_ts

    # Reuse source/target-side lookup work across samples sharing the same
    # endpoint and timestamp (common in DTGB positive+negative query bundles).
    source_context_cache = {}
    target_context_cache = {}
    pair_stats_cache = {}

    def get_source_context(source_id, timestamp):
        key = (int(source_id), int(timestamp))
        cached = source_context_cache.get(key)
        if cached is not None:
            return cached

        idx_s_source = bisect_left(history_as_source[source_id], timestamp, key=lambda x: x[3])
        idx_s_target = bisect_left(history_as_target[source_id], timestamp, key=lambda x: x[3])
        source_popularity_raw = int(idx_s_source + idx_s_target)

        src_neighbors_events = None
        if populate_prompt_lists:
            src_neighbors_events = {}
            for idx in range(idx_s_source):
                evt = history_as_source[source_id][idx]
                src_neighbors_events[evt[2]] = evt
            for idx in range(idx_s_target):
                evt = history_as_target[source_id][idx]
                src_neighbors_events[evt[0]] = evt

        source_history_static = None
        if populate_prompt_lists and (not semantic_history):
            source_history_static = get_recent_history(
                history_as_endpoint[source_id],
                timestamp,
                limit=history_window,
            )

        payload = {
            "idx_s_source": int(idx_s_source),
            "idx_s_target": int(idx_s_target),
            "source_popularity_raw": source_popularity_raw,
            "src_neighbors_events": src_neighbors_events,
            "source_history_static": source_history_static,
        }
        source_context_cache[key] = payload
        return payload

    def get_target_context(target_id, timestamp):
        key = (int(target_id), int(timestamp))
        cached = target_context_cache.get(key)
        if cached is not None:
            return cached

        idx_t_source = bisect_left(history_as_source[target_id], timestamp, key=lambda x: x[3])
        idx_t_target = bisect_left(history_as_target[target_id], timestamp, key=lambda x: x[3])
        target_popularity_raw = int(idx_t_source + idx_t_target)

        tar_neighbors_events = None
        if populate_prompt_lists:
            tar_neighbors_events = {}
            for idx in range(idx_t_source):
                evt = history_as_source[target_id][idx]
                tar_neighbors_events[evt[2]] = evt
            for idx in range(idx_t_target):
                evt = history_as_target[target_id][idx]
                tar_neighbors_events[evt[0]] = evt

        target_history_static = None
        if populate_prompt_lists and (not semantic_history):
            target_history_static = get_recent_history(
                history_as_endpoint[target_id], timestamp, limit=history_window
            )

        payload = {
            "idx_t_source": int(idx_t_source),
            "idx_t_target": int(idx_t_target),
            "target_popularity_raw": target_popularity_raw,
            "tar_neighbors_events": tar_neighbors_events,
            "target_history_static": target_history_static,
        }
        target_context_cache[key] = payload
        return payload

    def get_cached_pair_stats(source_id, target_id, timestamp, include_history):
        key = (int(source_id), int(target_id), int(timestamp), bool(include_history))
        cached = pair_stats_cache.get(key)
        if cached is not None:
            return cached
        payload = get_pair_stats(
            source_id=source_id,
            target_id=target_id,
            timestamp=timestamp,
            include_history=include_history,
        )
        pair_stats_cache[key] = payload
        return payload

    sample_iter = samples
    if populate_prompt_lists:
        sample_iter = tqdm(samples, total=len(samples), desc=monitor_label)

    sample_loop_start = time.perf_counter()
    for sample in sample_iter:
        source_id = int(sample["source_id"])
        target_id = int(sample["target_id"])
        timestamp = int(sample["timestamp"])

        sample["source_history_desc"] = source_history_desc
        sample["target_history_desc"] = target_history_desc
        sample["common_neighbors_desc"] = common_neighbors_desc

        semantic_embs_t = None
        if semantic_ranking_enabled:
            semantic_embs_t = get_semantic_embeddings_at_time(timestamp)

        source_ctx_start = time.perf_counter() if stage_profile_enabled else None
        source_ctx = get_source_context(source_id, timestamp)
        if stage_profile_enabled:
            stage_profile["source_context_sec"] += time.perf_counter() - source_ctx_start
        target_ctx_start = time.perf_counter() if stage_profile_enabled else None
        target_ctx = get_target_context(target_id, timestamp)
        if stage_profile_enabled:
            stage_profile["target_context_sec"] += time.perf_counter() - target_ctx_start

        source_history_list = []
        source_history_entities = []
        target_history_list = []
        target_history_entities = []
        common_neighbors_info = []

        if populate_prompt_lists:
            if semantic_history:
                source_hist_start = time.perf_counter() if stage_profile_enabled else None
                source_history_list, source_history_entities = build_source_history(
                    source_id,
                    target_id,
                    timestamp,
                    semantic_embs_t,
                )
                if stage_profile_enabled:
                    stage_profile["source_history_sec"] += time.perf_counter() - source_hist_start
                target_hist_start = time.perf_counter() if stage_profile_enabled else None
                target_history_list, target_history_entities = build_target_history(
                    source_id,
                    target_id,
                    timestamp,
                    semantic_embs_t,
                )
                if stage_profile_enabled:
                    stage_profile["target_history_sec"] += time.perf_counter() - target_hist_start
            else:
                source_history_list = list(source_ctx["source_history_static"] or [])
                target_history_list = list(target_ctx["target_history_static"] or [])

        pair_stats_start = time.perf_counter() if stage_profile_enabled else None
        mutual_history_list, num_past_interactions, last_ts = get_cached_pair_stats(
            source_id,
            target_id,
            timestamp,
            include_history=populate_prompt_lists,
        )
        if stage_profile_enabled:
            stage_profile["pair_stats_sec"] += time.perf_counter() - pair_stats_start

        sample["source_popularity_raw"] = int(source_ctx["source_popularity_raw"])
        sample["target_popularity_raw"] = int(target_ctx["target_popularity_raw"])
        sample["num_past_interactions"] = int(num_past_interactions)
        sample["num_past_interactions_raw"] = int(num_past_interactions)
        sample["last_interaction_delta"] = None if last_ts is None else int(timestamp - last_ts)

        if populate_prompt_lists:
            src_neighbors_events = source_ctx["src_neighbors_events"] or {}
            tar_neighbors_events = target_ctx["tar_neighbors_events"] or {}

            common_nodes = set(src_neighbors_events.keys()).intersection(tar_neighbors_events.keys())
            common_nodes.discard(source_id)
            common_nodes.discard(target_id)

            common_neighbors_start = time.perf_counter() if stage_profile_enabled else None
            common_neighbors_info = build_common_neighbors_info(
                common_nodes=common_nodes,
                dst_id=target_id,
                src_neighbors_events=src_neighbors_events,
                tar_neighbors_events=tar_neighbors_events,
                semantic_embs=semantic_embs_t,
                timestamp=timestamp,
            )
            if stage_profile_enabled:
                stage_profile["common_neighbors_sec"] += time.perf_counter() - common_neighbors_start

        sample["source_history"] = source_history_list
        sample["target_history"] = target_history_list
        sample["source_history_entities"] = source_history_entities
        sample["target_history_entities"] = target_history_entities
        sample["mutual_history"] = mutual_history_list
        sample["common_neighbors"] = common_neighbors_info
        sample["prompt_context_materialized"] = bool(populate_prompt_lists)
    stage_profile["sample_loop_sec"] = time.perf_counter() - sample_loop_start

    if calibrate_key_signals:
        calibration_start = time.perf_counter() if stage_profile_enabled else None
        calibrate_prompt_key_signals(
            samples=samples,
            key_signal_reference=key_signal_reference,
            edges_df=edges_df,
            history_as_source=history_as_source,
            history_as_target=history_as_target,
            edge_u_vals=u_vals,
            edge_i_vals=i_vals,
            edge_ts_vals=ts_vals,
            neighbor_sampler=neighbor_sampler,
            use_gpu_heuristics=use_gpu_heuristics,
            heuristic_recent_degree_window=heuristic_recent_degree_window,
        )
        if stage_profile_enabled:
            stage_profile["calibration_sec"] += time.perf_counter() - calibration_start

    if stage_profile_enabled:
        loop_total_sec = float(stage_profile["sample_loop_sec"])
        unique_timestamps = len({int(sample.get("timestamp", 0)) for sample in samples})
        cache_hits = int(stage_profile["semantic_cache_hits"])
        cache_misses = int(stage_profile["semantic_cache_misses"])

        def _loop_pct(seconds):
            if loop_total_sec <= 0.0:
                return 0.0
            return 100.0 * float(seconds) / float(loop_total_sec)

        print(
            "Hybrid prompt materialization bar profile: "
            f"samples={len(samples)}, unique_timestamps={unique_timestamps}, loop_total={loop_total_sec:.2f}s"
        )
        print(
            "  smoothing="
            f"{stage_profile['semantic_smoothing_sec']:.2f}s ({_loop_pct(stage_profile['semantic_smoothing_sec']):.1f}%), "
            "source_ctx="
            f"{stage_profile['source_context_sec']:.2f}s ({_loop_pct(stage_profile['source_context_sec']):.1f}%), "
            "target_ctx="
            f"{stage_profile['target_context_sec']:.2f}s ({_loop_pct(stage_profile['target_context_sec']):.1f}%)"
        )
        print(
            "  source_history="
            f"{stage_profile['source_history_sec']:.2f}s ({_loop_pct(stage_profile['source_history_sec']):.1f}%), "
            "target_history="
            f"{stage_profile['target_history_sec']:.2f}s ({_loop_pct(stage_profile['target_history_sec']):.1f}%), "
            "pair_stats="
            f"{stage_profile['pair_stats_sec']:.2f}s ({_loop_pct(stage_profile['pair_stats_sec']):.1f}%), "
            "common_neighbors="
            f"{stage_profile['common_neighbors_sec']:.2f}s ({_loop_pct(stage_profile['common_neighbors_sec']):.1f}%)"
        )
        if semantic_ranking_enabled and semantic_use_smoothing:
            print(
                "  semantic smoothing cache: "
                f"hits={cache_hits}, misses={cache_misses}"
            )

    return samples


__all__ = ["calibrate_prompt_key_signals", "materialize_samples_prompt_context"]
