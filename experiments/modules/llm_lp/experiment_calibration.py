#!/usr/bin/env python3
"""
Calibration and routing helpers for LLM link prediction experiments.
"""
import random
from bisect import bisect_left
from collections import defaultdict

import numpy as np

from experiments.modules.llm_lp.sample_finalize import finalize_test_samples
from experiments.modules.rrf.analysis import (
    compute_two_component_gmm_overlap_scores,
    fit_two_component_gaussian_mixture_1d,
)


def _resolve_finalize_smoothing_kwargs(args):
    return {
        "smooth_time_window": float(getattr(args, "smooth_time_window", 50.0)),
        "smooth_steps": int(getattr(args, "smooth_steps", 1)),
        "smooth_decay_gamma": getattr(args, "smooth_decay_gamma", 0.1),
        "smooth_undirected": bool(getattr(args, "smooth_undirected", True)),
    }


def _directed_edge_arrays(edges_df):
    edges_df_sorted = edges_df.sort_values("ts")
    return (
        edges_df_sorted["u"].values.astype(np.int64, copy=False),
        edges_df_sorted["i"].values.astype(np.int64, copy=False),
        edges_df_sorted["ts"].values.astype(np.float64, copy=False),
    )


def _uses_validation_sampled_threeway(mode):
    return str(mode).strip().lower() == "validation_sampled_threeway"


def _uses_validation_sampled_binary(mode):
    return str(mode).strip().lower() == "validation_sampled_binary"


def _resolve_validation_calibration_sizes(args):
    num_samples = int(args.validation_calibration_num_samples)
    if num_samples <= 0:
        num_samples = int(args.num_samples)
    negative_ratio = args.validation_calibration_negative_ratio
    if negative_ratio is None:
        negative_ratio = int(args.negative_ratio)
    else:
        negative_ratio = int(negative_ratio)
    return num_samples, negative_ratio


def _resolve_hybrid_alignment_calibration_sizes(args):
    num_samples = int(getattr(args, "hybrid_alignment_num_samples", 0))
    if num_samples <= 0:
        num_samples = int(args.validation_calibration_num_samples)
    if num_samples <= 0:
        num_samples = int(args.num_samples)

    negative_ratio = getattr(args, "hybrid_alignment_negative_ratio", None)
    if negative_ratio is None:
        negative_ratio = args.validation_calibration_negative_ratio
    if negative_ratio is None:
        negative_ratio = int(args.negative_ratio)
    else:
        negative_ratio = int(negative_ratio)
    return num_samples, negative_ratio


def _uses_validation_sampled_key_signal_reference(reference):
    return str(reference).strip().lower() == "validation_sampled_global"


def dataset_uses_dtgb_time_bucket(dataset_name):
    return str(dataset_name).strip().upper() == "GDELT"


def _select_eval_edges_and_negative_pool(
    edges_df,
    *,
    test_ratio,
    val_ratio,
    eval_split,
    apply_gdelt_time_bucket=False,
):
    edge_ts_raw = edges_df["ts"].values.astype(np.float64)
    edge_ts_for_split = edge_ts_raw
    if apply_gdelt_time_bucket:
        edge_ts_for_split = np.floor_divide(edge_ts_raw.astype(np.int64), 15).astype(np.float64)

    val_time = np.quantile(edge_ts_for_split, 1 - val_ratio - test_ratio)
    test_time = np.quantile(edge_ts_for_split, 1 - test_ratio)

    edge_src = edges_df["u"].values.astype(np.int64)
    edge_dst = edges_df["i"].values.astype(np.int64)
    edge_ts = edge_ts_for_split

    if eval_split == "transductive":
        eval_mask = edge_ts > test_time
        eval_edges = edges_df[eval_mask].copy()
        negative_dst_pool = np.unique(edge_dst).astype(np.int64)
    elif eval_split == "validation":
        eval_mask = np.logical_and(edge_ts >= val_time, edge_ts < test_time)
        eval_edges = edges_df[eval_mask].copy()
        negative_dst_pool = np.unique(edge_dst[edge_ts < test_time]).astype(np.int64)
    elif eval_split == "inductive":
        rng = random.Random(2020)
        node_set = set(edge_src).union(set(edge_dst))
        num_total_unique_node_ids = len(node_set)
        test_node_set = set(edge_src[edge_ts > val_time]).union(set(edge_dst[edge_ts > val_time]))
        new_test_node_set = set(
            rng.sample(list(test_node_set), int(0.1 * num_total_unique_node_ids))
        )

        new_test_source_mask = edges_df.u.map(lambda x: x in new_test_node_set).values
        new_test_destination_mask = edges_df.i.map(lambda x: x in new_test_node_set).values
        observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)
        train_mask = np.logical_and(edge_ts <= val_time, observed_edges_mask)

        train_node_set = set(edge_src[train_mask]).union(set(edge_dst[train_mask]))
        if len(train_node_set & new_test_node_set) != 0:
            raise RuntimeError("Inductive split construction violated DTGB new-node isolation.")
        new_node_set = node_set - train_node_set

        edge_contains_new_node_mask = np.array(
            [
                (src_node_id in new_node_set or dst_node_id in new_node_set)
                for src_node_id, dst_node_id in zip(edge_src, edge_dst)
            ]
        )
        eval_mask = np.logical_and(edge_ts > test_time, edge_contains_new_node_mask)
        eval_edges = edges_df[eval_mask].copy()
        negative_dst_pool = np.unique(edge_dst[eval_mask]).astype(np.int64)
    else:
        raise ValueError(f"Unsupported eval_split: {eval_split}")

    if negative_dst_pool.size == 0:
        raise RuntimeError(
            f"No destination nodes available for DTGB-style negative sampling in split '{eval_split}'."
        )
    return eval_edges, negative_dst_pool


def _build_lightweight_rrf_samples_for_split(
    edges_df,
    *,
    test_ratio,
    val_ratio,
    num_samples,
    negative_ratio,
    random_seed,
    eval_split,
    apply_gdelt_time_bucket=False,
):
    eval_edges, negative_dst_pool = _select_eval_edges_and_negative_pool(
        edges_df,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        eval_split=eval_split,
        apply_gdelt_time_bucket=apply_gdelt_time_bucket,
    )

    if len(eval_edges) > int(num_samples):
        positive_samples = eval_edges.sample(n=int(num_samples), random_state=int(random_seed))
    else:
        positive_samples = eval_edges

    rng = np.random.default_rng(int(random_seed))
    samples = []
    for query_id, row in enumerate(positive_samples.itertuples(index=False)):
        source_id = int(row.u)
        target_id = int(row.i)
        timestamp = int(row.ts)
        relation_id = int(row.r)
        timestamp_fields = {"timestamp": timestamp}
        if apply_gdelt_time_bucket:
            # The semantic GDELT backbone is built on the DTGB time axis
            # (raw timestamp // 15).  Keep the raw timestamp for prompt-side
            # consumers, but expose the same bucketed cutoff used by ordinary
            # train/test samples so temporal calibration cannot see the future.
            timestamp_fields["dtgb_timestamp"] = timestamp // 15

        samples.append(
            {
                "source_id": source_id,
                "target_id": target_id,
                **timestamp_fields,
                "relation_id": relation_id,
                "query_id": int(query_id),
                "label": 1,
            }
        )

        sampled_neg_indices = rng.integers(0, len(negative_dst_pool), size=int(negative_ratio))
        sampled_negatives = negative_dst_pool[sampled_neg_indices]
        for neg_target_id in sampled_negatives:
            samples.append(
                {
                    "source_id": source_id,
                    "target_id": int(neg_target_id),
                    **timestamp_fields,
                    "relation_id": relation_id,
                    "query_id": int(query_id),
                    "label": 0,
                }
            )
    return samples


def _attach_lightweight_structural_fields(samples, edges_df):
    if not samples:
        return samples

    print("Indexing lightweight structural histories for validation key-signal calibration...")
    edges_df_sorted = edges_df.sort_values("ts")
    u_vals = edges_df_sorted["u"].values.astype(np.int64, copy=False)
    r_vals = edges_df_sorted["r"].values.astype(np.int64, copy=False)
    i_vals = edges_df_sorted["i"].values.astype(np.int64, copy=False)
    ts_vals = edges_df_sorted["ts"].values.astype(np.int64, copy=False)

    history_as_source = defaultdict(list)
    history_as_target = defaultdict(list)
    pair_history_as_source = defaultdict(lambda: defaultdict(list))
    for u, r, i, ts in zip(u_vals, r_vals, i_vals, ts_vals):
        event = (int(u), int(r), int(i), int(ts))
        history_as_source[int(u)].append(event)
        history_as_target[int(i)].append(event)
        pair_history_as_source[int(u)][int(i)].append(event)

    popularity_cache = {}
    pair_stats_cache = {}

    def _popularity_before(node_id, timestamp):
        key = (int(node_id), int(timestamp))
        cached = popularity_cache.get(key)
        if cached is not None:
            return cached
        out_hist = history_as_source.get(int(node_id), [])
        in_hist = history_as_target.get(int(node_id), [])
        out_count = bisect_left(out_hist, int(timestamp), key=lambda x: x[3])
        in_count = bisect_left(in_hist, int(timestamp), key=lambda x: x[3])
        value = int(out_count + in_count)
        popularity_cache[key] = value
        return value

    def _pair_stats_before(src_id, dst_id, timestamp):
        key = (int(src_id), int(dst_id), int(timestamp))
        cached = pair_stats_cache.get(key)
        if cached is not None:
            return cached

        forward_hist = pair_history_as_source[int(src_id)].get(int(dst_id), [])
        backward_hist = pair_history_as_source[int(dst_id)].get(int(src_id), [])
        forward_count = bisect_left(forward_hist, int(timestamp), key=lambda x: x[3])
        backward_count = bisect_left(backward_hist, int(timestamp), key=lambda x: x[3])

        last_ts = None
        if forward_count > 0:
            last_ts = int(forward_hist[forward_count - 1][3])
        if backward_count > 0:
            backward_last_ts = int(backward_hist[backward_count - 1][3])
            if last_ts is None or backward_last_ts > last_ts:
                last_ts = backward_last_ts

        value = (
            int(forward_count + backward_count),
            None if last_ts is None else int(timestamp) - int(last_ts),
        )
        pair_stats_cache[key] = value
        return value

    print(
        "Attaching lightweight structural fields for validation key-signal calibration: "
        f"{len(samples)} samples"
    )
    for sample in samples:
        source_id = int(sample["source_id"])
        target_id = int(sample["target_id"])
        timestamp = int(sample["timestamp"])
        sample["source_popularity_raw"] = _popularity_before(source_id, timestamp)
        sample["target_popularity_raw"] = _popularity_before(target_id, timestamp)
        num_past_interactions, last_interaction_delta = _pair_stats_before(
            source_id,
            target_id,
            timestamp,
        )
        sample["num_past_interactions"] = int(num_past_interactions)
        sample["num_past_interactions_raw"] = int(num_past_interactions)
        sample["last_interaction_delta"] = last_interaction_delta
    return samples


def _quantile_pair_from_values(values, low_q=0.3, high_q=0.7):
    if values is None or len(values) == 0:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    return float(np.quantile(arr, low_q)), float(np.quantile(arr, high_q))


def _bucket_from_thresholds_local(
    value,
    low_t,
    high_t,
    *,
    higher_is_better=True,
    missing_label="Modest",
):
    if value is None:
        return missing_label
    if low_t is None or high_t is None or low_t == high_t:
        return "Modest"
    if higher_is_better:
        if float(value) >= float(high_t):
            return "High"
        if float(value) <= float(low_t):
            return "Low"
    else:
        if float(value) <= float(low_t):
            return "High"
        if float(value) >= float(high_t):
            return "Low"
    return "Modest"


def _percentile_from_sorted_values(value, sorted_values, *, invert=False, missing_value=0.0):
    if value is None:
        return float(missing_value)
    n = len(sorted_values)
    if n <= 1:
        pct = 50.0
    else:
        arr = np.asarray(sorted_values, dtype=np.float64)
        lo = int(np.searchsorted(arr, float(value), side="left"))
        hi = int(np.searchsorted(arr, float(value), side="right"))
        avg_rank = 0.5 * (lo + hi - 1)
        pct = 100.0 * (avg_rank / float(n - 1))
    if invert:
        pct = 100.0 - pct
    return float(max(0.0, min(100.0, pct)))


def _build_validation_sampled_key_signal_reference(samples):
    popularity_history = []
    source_pop_history = []
    target_pop_history = []
    interaction_history = []
    recency_history = []
    common_neighbor_history = []
    global_recency_history = []
    itemcf_history = []
    usercf_history = []

    for sample in samples:
        source_pop_raw = float(sample.get("source_popularity_raw", 0.0))
        target_pop_raw = float(sample.get("target_popularity_raw", 0.0))
        interaction_raw = float(
            sample.get("num_past_interactions_raw", sample.get("num_past_interactions", 0.0))
        )
        common_neighbor_raw = float(sample.get("common_neighbor_score", 0.0))
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

        popularity_history.extend([source_pop_raw, target_pop_raw])
        source_pop_history.append(source_pop_raw)
        target_pop_history.append(target_pop_raw)
        interaction_history.append(interaction_raw)
        common_neighbor_history.append(common_neighbor_raw)
        if recency_raw is not None:
            recency_history.append(recency_raw)
        if global_recency_raw is not None:
            global_recency_history.append(global_recency_raw)
        if itemcf_raw is not None:
            itemcf_history.append(itemcf_raw)
        if usercf_raw is not None:
            usercf_history.append(usercf_raw)

    return {
        "num_samples": int(len(samples)),
        "popularity_history": sorted(float(v) for v in popularity_history),
        "source_pop_history": sorted(float(v) for v in source_pop_history),
        "target_pop_history": sorted(float(v) for v in target_pop_history),
        "interaction_history": sorted(float(v) for v in interaction_history),
        "recency_history": sorted(float(v) for v in recency_history),
        "common_neighbor_history": sorted(float(v) for v in common_neighbor_history),
        "global_recency_history": sorted(float(v) for v in global_recency_history),
        "itemcf_history": sorted(float(v) for v in itemcf_history),
        "usercf_history": sorted(float(v) for v in usercf_history),
    }


def _apply_validation_sampled_key_signal_reference(samples, reference):
    def _history_array(name):
        return np.asarray(reference.get(name, []), dtype=np.float64)

    popularity_history = _history_array("popularity_history")
    source_pop_history = _history_array("source_pop_history")
    target_pop_history = _history_array("target_pop_history")
    interaction_history = _history_array("interaction_history")
    recency_history = _history_array("recency_history")
    common_neighbor_history = _history_array("common_neighbor_history")
    global_recency_history = _history_array("global_recency_history")
    itemcf_history = _history_array("itemcf_history")
    usercf_history = _history_array("usercf_history")

    pop_low, pop_high = _quantile_pair_from_values(popularity_history)
    inter_low, inter_high = _quantile_pair_from_values(interaction_history)
    rec_low, rec_high = _quantile_pair_from_values(recency_history)
    cn_low, cn_high = _quantile_pair_from_values(common_neighbor_history)
    global_rec_low, global_rec_high = _quantile_pair_from_values(global_recency_history)
    itemcf_low, itemcf_high = _quantile_pair_from_values(itemcf_history)
    usercf_low, usercf_high = _quantile_pair_from_values(usercf_history)

    total_samples = len(samples)
    print(
        "Applying validation-sampled key-signal reference: "
        f"samples={total_samples}, calibration_samples={reference.get('num_samples', 'unknown')}",
        flush=True,
    )
    progress_every = 100000
    for sample_idx, sample in enumerate(samples, start=1):
        source_pop_raw = float(sample.get("source_popularity_raw", 0.0))
        target_pop_raw = float(sample.get("target_popularity_raw", 0.0))
        interaction_raw = float(
            sample.get("num_past_interactions_raw", sample.get("num_past_interactions", 0.0))
        )
        common_neighbor_raw = float(sample.get("common_neighbor_score", 0.0))
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

        sample["source_popularity"] = _bucket_from_thresholds_local(
            source_pop_raw,
            pop_low,
            pop_high,
            higher_is_better=True,
        )
        sample["target_popularity"] = _bucket_from_thresholds_local(
            target_pop_raw,
            pop_low,
            pop_high,
            higher_is_better=True,
        )
        sample["num_past_interactions"] = _bucket_from_thresholds_local(
            interaction_raw,
            inter_low,
            inter_high,
            higher_is_better=True,
        )
        sample["common_neighbor_level"] = _bucket_from_thresholds_local(
            common_neighbor_raw,
            cn_low,
            cn_high,
            higher_is_better=True,
        )
        sample["last_interaction_str"] = _bucket_from_thresholds_local(
            recency_raw,
            rec_low,
            rec_high,
            higher_is_better=False,
            missing_label="No prior interactions",
        )
        sample["global_recency_level"] = _bucket_from_thresholds_local(
            global_recency_raw,
            global_rec_low,
            global_rec_high,
            higher_is_better=True,
            missing_label="No prior target interactions",
        )
        sample["itemcf_level"] = _bucket_from_thresholds_local(
            itemcf_raw,
            itemcf_low,
            itemcf_high,
            higher_is_better=True,
            missing_label="Low",
        )
        sample["usercf_level"] = _bucket_from_thresholds_local(
            usercf_raw,
            usercf_low,
            usercf_high,
            higher_is_better=True,
            missing_label="Low",
        )

        sample["source_popularity_pct"] = _percentile_from_sorted_values(
            source_pop_raw,
            source_pop_history,
        )
        sample["target_popularity_pct"] = _percentile_from_sorted_values(
            target_pop_raw,
            target_pop_history,
        )
        sample["num_past_interactions_pct"] = _percentile_from_sorted_values(
            interaction_raw,
            interaction_history,
        )
        sample["common_neighbor_score_pct"] = _percentile_from_sorted_values(
            common_neighbor_raw,
            common_neighbor_history,
        )
        sample["last_interaction_recency_pct"] = _percentile_from_sorted_values(
            recency_raw,
            recency_history,
            invert=True,
            missing_value=0.0,
        )
        sample["global_recency_pct"] = _percentile_from_sorted_values(
            global_recency_raw,
            global_recency_history,
            missing_value=0.0,
        )
        sample["itemcf_pct"] = _percentile_from_sorted_values(
            itemcf_raw,
            itemcf_history,
            missing_value=0.0,
        )
        sample["usercf_pct"] = _percentile_from_sorted_values(
            usercf_raw,
            usercf_history,
            missing_value=0.0,
        )
        if sample_idx % progress_every == 0:
            print(
                "Applying validation-sampled key-signal reference: "
                f"{sample_idx}/{total_samples} samples",
                flush=True,
            )
    print(
        "Applying validation-sampled key-signal reference complete: "
        f"samples={total_samples}",
        flush=True,
    )
    return samples


def _best_balanced_threshold(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    unique_scores = np.unique(scores)
    if unique_scores.size == 0:
        return 0.0

    best_threshold = float(unique_scores[0])
    best_score = -np.inf
    for threshold in unique_scores:
        preds = (scores >= float(threshold)).astype(np.int64)
        pos_mask = labels == 1
        neg_mask = labels == 0
        tpr = float(preds[pos_mask].mean()) if np.any(pos_mask) else 0.0
        tnr = float((1 - preds[neg_mask]).mean()) if np.any(neg_mask) else 0.0
        balanced_acc = 0.5 * (tpr + tnr)
        if balanced_acc > best_score:
            best_score = balanced_acc
            best_threshold = float(threshold)
    return best_threshold


def _derive_validation_sampled_threeway_thresholds(
    scores,
    labels,
    *,
    low_neg_quantile,
    high_pos_quantile,
):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    if pos_scores.size == 0 or neg_scores.size == 0:
        raise ValueError("Validation calibration requires both positive and negative samples.")

    low_threshold = float(np.quantile(neg_scores, float(low_neg_quantile)))
    high_threshold = float(np.quantile(pos_scores, float(high_pos_quantile)))
    collapsed_to_single_threshold = False
    if low_threshold > high_threshold:
        midpoint = _best_balanced_threshold(scores, labels)
        low_threshold = midpoint
        high_threshold = midpoint
        collapsed_to_single_threshold = True

    low_mask = scores < low_threshold
    middle_mask = (scores >= low_threshold) & (scores < high_threshold)
    high_mask = scores >= high_threshold

    summary = {
        "num_samples": int(scores.size),
        "num_positive": int((labels == 1).sum()),
        "num_negative": int((labels == 0).sum()),
        "low_threshold": float(low_threshold),
        "high_threshold": float(high_threshold),
        "low_neg_quantile": float(low_neg_quantile),
        "high_pos_quantile": float(high_pos_quantile),
        "collapsed_to_single_threshold": bool(collapsed_to_single_threshold),
        "low_coverage": float(low_mask.mean()) if scores.size > 0 else 0.0,
        "middle_coverage": float(middle_mask.mean()) if scores.size > 0 else 0.0,
        "high_coverage": float(high_mask.mean()) if scores.size > 0 else 0.0,
        "low_negative_rate": float((labels[low_mask] == 0).mean()) if np.any(low_mask) else None,
        "high_positive_rate": float((labels[high_mask] == 1).mean()) if np.any(high_mask) else None,
    }
    return summary


def _derive_validation_sampled_binary_threshold(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.size == 0:
        raise ValueError("Validation calibration requires at least one sample.")
    pos_mask = labels == 1
    neg_mask = labels == 0
    if not np.any(pos_mask) or not np.any(neg_mask):
        raise ValueError("Validation calibration requires both positive and negative samples.")

    threshold = float(_best_balanced_threshold(scores, labels))
    preds = (scores >= threshold).astype(np.int64)
    tpr = float(preds[pos_mask].mean()) if np.any(pos_mask) else 0.0
    tnr = float((1 - preds[neg_mask]).mean()) if np.any(neg_mask) else 0.0
    balanced_acc = 0.5 * (tpr + tnr)
    positive_rate = float(preds.mean()) if preds.size > 0 else 0.0
    return {
        "mode": "validation_sampled_binary",
        "num_samples": int(scores.size),
        "num_positive": int(pos_mask.sum()),
        "num_negative": int(neg_mask.sum()),
        "threshold": threshold,
        "low_threshold": threshold,
        "high_threshold": threshold,
        "balanced_accuracy": float(balanced_acc),
        "true_positive_rate": float(tpr),
        "true_negative_rate": float(tnr),
        "predicted_positive_rate": float(positive_rate),
    }


def _derive_validation_sampled_hybrid_uncertainty_band(
    scores,
    labels,
    *,
    target_fraction,
):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.size == 0:
        raise ValueError("Validation hybrid routing calibration requires at least one scored sample.")

    requested_fraction = float(max(0.0, min(1.0, target_fraction)))
    center_threshold = float(_best_balanced_threshold(scores, labels))
    target_count = int(round(requested_fraction * float(scores.size)))
    if requested_fraction > 0.0:
        target_count = max(1, target_count)
    target_count = min(int(scores.size), target_count)

    if target_count <= 0:
        return {
            "num_samples": int(scores.size),
            "num_positive": int((labels == 1).sum()),
            "num_negative": int((labels == 0).sum()),
            "center_threshold": center_threshold,
            "low_threshold": center_threshold,
            "high_threshold": center_threshold,
            "requested_target_fraction": requested_fraction,
            "requested_target_count": 0,
            "selected_count_realized": 0,
            "selected_fraction_realized": 0.0,
            "selected_positive_rate": None,
            "selected_negative_rate": None,
        }

    distance = np.abs(scores - center_threshold)
    order = np.lexsort((np.arange(scores.size), distance))
    chosen = order[:target_count]
    chosen_scores = scores[chosen]
    low_threshold = float(np.min(chosen_scores))
    high_threshold = float(np.nextafter(np.max(chosen_scores), np.inf))

    selected_mask = (scores >= low_threshold) & (scores < high_threshold)
    selected_count = int(selected_mask.sum())
    selected_positive_rate = float(labels[selected_mask].mean()) if selected_count > 0 else None

    return {
        "num_samples": int(scores.size),
        "num_positive": int((labels == 1).sum()),
        "num_negative": int((labels == 0).sum()),
        "center_threshold": center_threshold,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "requested_target_fraction": requested_fraction,
        "requested_target_count": int(target_count),
        "selected_count_realized": selected_count,
        "selected_fraction_realized": float(selected_count / max(scores.size, 1)),
        "selected_positive_rate": selected_positive_rate,
        "selected_negative_rate": (
            None if selected_positive_rate is None else float(1.0 - selected_positive_rate)
        ),
    }


def _derive_validation_sampled_hybrid_gmm_overlap_band(
    scores,
    labels,
    *,
    target_fraction,
):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.size == 0:
        raise ValueError("Validation GMM routing calibration requires at least one scored sample.")

    requested_fraction = float(max(0.0, min(1.0, target_fraction)))
    gmm = fit_two_component_gaussian_mixture_1d(scores)
    overlap_scores = compute_two_component_gmm_overlap_scores(scores, gmm)
    target_count = int(round(requested_fraction * float(scores.size)))
    if requested_fraction > 0.0:
        target_count = max(1, target_count)
    target_count = min(int(scores.size), target_count)
    center_threshold = float(np.mean(np.asarray(gmm["means"], dtype=np.float64)))

    if target_count <= 0:
        return {
            "num_samples": int(scores.size),
            "num_positive": int((labels == 1).sum()),
            "num_negative": int((labels == 0).sum()),
            "center_threshold": center_threshold,
            "ambiguity_threshold": 1.0,
            "requested_target_fraction": requested_fraction,
            "requested_target_count": 0,
            "selected_count_realized": 0,
            "selected_fraction_realized": 0.0,
            "selected_positive_rate": None,
            "selected_negative_rate": None,
            "gmm": gmm,
        }

    chosen = np.asarray(
        np.lexsort((np.arange(overlap_scores.size), -overlap_scores))[:target_count],
        dtype=np.int64,
    )
    chosen_overlap = overlap_scores[chosen]
    ambiguity_threshold = float(np.min(chosen_overlap))
    selected_mask = overlap_scores >= ambiguity_threshold
    selected_count = int(selected_mask.sum())
    selected_positive_rate = float(labels[selected_mask].mean()) if selected_count > 0 else None

    return {
        "num_samples": int(scores.size),
        "num_positive": int((labels == 1).sum()),
        "num_negative": int((labels == 0).sum()),
        "center_threshold": center_threshold,
        "ambiguity_threshold": ambiguity_threshold,
        "requested_target_fraction": requested_fraction,
        "requested_target_count": int(target_count),
        "selected_count_realized": selected_count,
        "selected_fraction_realized": float(selected_count / max(scores.size, 1)),
        "selected_positive_rate": selected_positive_rate,
        "selected_negative_rate": (
            None if selected_positive_rate is None else float(1.0 - selected_positive_rate)
        ),
        "gmm": gmm,
    }


def _apply_validation_sampled_threeway_labels(
    samples,
    *,
    low_threshold,
    high_threshold,
    assign_expert_prediction,
    assign_overall_signal,
    score_field="rrf_score",
):
    low_threshold = float(low_threshold)
    high_threshold = float(high_threshold)
    for sample in samples:
        score = float(sample.get(score_field, 0.0))
        if score < low_threshold:
            expert_value = "False"
            bucket_value = "Low"
        elif score >= high_threshold:
            expert_value = "True"
            bucket_value = "High"
        else:
            expert_value = "Neutral"
            bucket_value = "Modest"
        if assign_expert_prediction:
            sample["expert_prediction"] = expert_value
        if assign_overall_signal:
            sample["overall_structural_signal"] = bucket_value
    return samples


def _apply_validation_sampled_binary_labels(
    samples,
    *,
    threshold,
    assign_expert_prediction,
    assign_overall_signal,
    score_field="rrf_score",
):
    threshold = float(threshold)
    for sample in samples:
        score = float(sample.get(score_field, 0.0))
        if score >= threshold:
            expert_value = "True"
            bucket_value = "High"
        else:
            expert_value = "False"
            bucket_value = "Low"
        if assign_expert_prediction:
            sample["expert_prediction"] = expert_value
        if assign_overall_signal:
            sample["overall_structural_signal"] = bucket_value
    return samples


def _calibrate_validation_sampled_threeway_thresholds(
    *,
    args,
    edges,
    embeddings,
    entity_id_to_idx,
    random_seed,
    score_field="rrf_score",
    score_label="RRF",
    score_samples_fn=None,
):
    apply_dtgb_time_bucket = dataset_uses_dtgb_time_bucket(args.dataset_name)
    calibration_num_samples, calibration_negative_ratio = _resolve_validation_calibration_sizes(args)
    calibration_mode = str(args.expert_prediction_mode).strip().lower()
    if calibration_mode not in {"validation_sampled_threeway", "validation_sampled_binary"}:
        raise ValueError(
            "Validation-sampled prior calibration requires "
            "expert_prediction_mode in {'validation_sampled_threeway', 'validation_sampled_binary'}."
        )
    mode_label = (
        "three-way prior"
        if calibration_mode == "validation_sampled_threeway"
        else "binary prior"
    )
    print(
        "Calibrating validation-sampled "
        f"{mode_label}: "
        f"validation positives={calibration_num_samples}, "
        f"negative_ratio={calibration_negative_ratio}"
    )
    calibration_samples = _build_lightweight_rrf_samples_for_split(
        edges,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        num_samples=calibration_num_samples,
        negative_ratio=calibration_negative_ratio,
        random_seed=random_seed,
        eval_split="validation",
        apply_gdelt_time_bucket=apply_dtgb_time_bucket,
    )
    if score_samples_fn is None:
        finalize_test_samples(
            samples=calibration_samples,
            edges_df=edges,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            negative_ratio=calibration_negative_ratio,
            random_seed=random_seed,
            build_prompt_features=False,
            compute_expert_prediction=False,
            compute_rrf_scores=True,
            rrf_k=args.rrf_k,
            rrf_mode=args.rrf_mode,
            sequential_rank_bins=args.sequential_rank_bins,
            expert_prediction_mode="fixed_threshold",
            expert_prediction_fixed_threshold=args.expert_prediction_fixed_threshold,
            rrf_pointwise_pool_size=args.rrf_pointwise_pool_size,
            rrf_pointwise_num_pools=args.rrf_pointwise_num_pools,
            rrf_batch_size=args.rrf_batch_size,
            rrf_heuristics=args.rrf_heuristics,
            key_signal_reference="sequential_global",
            include_overall_structural_signal=False,
            overall_signal_low_threshold=args.overall_signal_low_threshold,
            overall_signal_high_threshold=args.overall_signal_high_threshold,
            embeddings=embeddings,
            entity_id_to_idx=entity_id_to_idx,
            apply_gdelt_time_bucket=apply_dtgb_time_bucket,
            **_resolve_finalize_smoothing_kwargs(args),
        )
    else:
        score_samples_fn(calibration_samples, score_field=score_field)
    scores = np.asarray(
        [float(sample.get(score_field, 0.0)) for sample in calibration_samples],
        dtype=np.float64,
    )
    labels = np.asarray([int(sample.get("label", 0)) for sample in calibration_samples], dtype=np.int64)
    if calibration_mode == "validation_sampled_binary":
        summary = _derive_validation_sampled_binary_threshold(scores, labels)
    else:
        summary = _derive_validation_sampled_threeway_thresholds(
            scores,
            labels,
            low_neg_quantile=args.validation_calibration_low_neg_quantile,
            high_pos_quantile=args.validation_calibration_high_pos_quantile,
        )
    summary["score_field"] = str(score_field)
    summary["score_label"] = str(score_label)
    if calibration_mode == "validation_sampled_binary":
        print(
            f"Validation-sampled binary threshold [{score_label}]: "
            f"threshold={summary['threshold']:.6f}, "
            f"balanced_acc={summary['balanced_accuracy']:.4f}, "
            f"TPR={summary['true_positive_rate']:.4f}, "
            f"TNR={summary['true_negative_rate']:.4f}, "
            f"pred_pos_rate={summary['predicted_positive_rate']:.4f}"
        )
    else:
        print(
            f"Validation-sampled three-way thresholds [{score_label}]: "
            f"low={summary['low_threshold']:.6f}, "
            f"high={summary['high_threshold']:.6f}, "
            f"(neg_q={summary['low_neg_quantile']:.2f}, "
            f"pos_q={summary['high_pos_quantile']:.2f}), "
            f"coverage(low/mid/high)="
            f"{summary['low_coverage']:.3f}/"
            f"{summary['middle_coverage']:.3f}/"
            f"{summary['high_coverage']:.3f}"
        )
        if summary["low_negative_rate"] is not None or summary["high_positive_rate"] is not None:
            low_rate = (
                f"{summary['low_negative_rate']:.3f}"
                if summary["low_negative_rate"] is not None
                else "n/a"
            )
            high_rate = (
                f"{summary['high_positive_rate']:.3f}"
                if summary["high_positive_rate"] is not None
                else "n/a"
            )
            print(
                "Validation-sampled band purity: "
                f"low_negative_rate={low_rate}, high_positive_rate={high_rate}"
            )
        if summary["collapsed_to_single_threshold"]:
            print("Validation-sampled calibration bands overlapped; collapsed to a single threshold.")
    return summary


def _calibrate_validation_sampled_key_signal_reference(
    *,
    args,
    edges,
    embeddings,
    entity_id_to_idx,
    random_seed,
):
    apply_dtgb_time_bucket = dataset_uses_dtgb_time_bucket(args.dataset_name)
    calibration_num_samples, calibration_negative_ratio = _resolve_validation_calibration_sizes(args)
    print(
        "Calibrating validation-sampled key-signal reference: "
        f"validation positives={calibration_num_samples}, "
        f"negative_ratio={calibration_negative_ratio}"
    )
    calibration_samples = _build_lightweight_rrf_samples_for_split(
        edges,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        num_samples=calibration_num_samples,
        negative_ratio=calibration_negative_ratio,
        random_seed=random_seed,
        eval_split="validation",
        apply_gdelt_time_bucket=apply_dtgb_time_bucket,
    )
    _attach_lightweight_structural_fields(
        calibration_samples,
        edges,
    )
    edge_u_vals, edge_i_vals, edge_ts_vals = _directed_edge_arrays(edges)
    finalize_test_samples(
        samples=calibration_samples,
        edges_df=edges,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        negative_ratio=calibration_negative_ratio,
        random_seed=random_seed,
        build_prompt_features=False,
        compute_expert_prediction=False,
        compute_rrf_scores=False,
        rrf_k=args.rrf_k,
        rrf_mode=args.rrf_mode,
        sequential_rank_bins=args.sequential_rank_bins,
        expert_prediction_mode="fixed_threshold",
        expert_prediction_fixed_threshold=args.expert_prediction_fixed_threshold,
        key_signal_fields=args.key_signal_fields,
        rrf_pointwise_pool_size=args.rrf_pointwise_pool_size,
        rrf_pointwise_num_pools=args.rrf_pointwise_num_pools,
        rrf_batch_size=args.rrf_batch_size,
        rrf_heuristics=args.rrf_heuristics,
        key_signal_reference="sequential_global",
        include_overall_structural_signal=False,
        overall_signal_low_threshold=args.overall_signal_low_threshold,
        overall_signal_high_threshold=args.overall_signal_high_threshold,
        embeddings=embeddings,
        entity_id_to_idx=entity_id_to_idx,
        apply_gdelt_time_bucket=apply_dtgb_time_bucket,
        edge_u_vals=edge_u_vals,
        edge_i_vals=edge_i_vals,
        edge_ts_vals=edge_ts_vals,
        **_resolve_finalize_smoothing_kwargs(args),
    )
    reference = _build_validation_sampled_key_signal_reference(calibration_samples)
    print(
        "Validation-sampled key-signal reference ready: "
        f"{reference['num_samples']} samples, "
        f"pop={len(reference['popularity_history'])}, "
        f"inter={len(reference['interaction_history'])}, "
        f"rec={len(reference['recency_history'])}, "
        f"cn={len(reference['common_neighbor_history'])}"
    )
    return reference


def _calibrate_validation_sampled_hybrid_uncertainty_band(
    *,
    args,
    edges,
    embeddings,
    entity_id_to_idx,
    random_seed,
    score_field="rrf_score",
    score_label="RRF",
    score_samples_fn=None,
):
    apply_dtgb_time_bucket = dataset_uses_dtgb_time_bucket(args.dataset_name)
    calibration_num_samples, calibration_negative_ratio = _resolve_validation_calibration_sizes(args)
    print(
        "Calibrating validation-sampled hybrid routing band: "
        f"validation positives={calibration_num_samples}, "
        f"negative_ratio={calibration_negative_ratio}, "
        f"target_fraction={float(args.hybrid_validation_target_fraction):.3f}"
    )
    calibration_samples = _build_lightweight_rrf_samples_for_split(
        edges,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        num_samples=calibration_num_samples,
        negative_ratio=calibration_negative_ratio,
        random_seed=random_seed,
        eval_split="validation",
        apply_gdelt_time_bucket=apply_dtgb_time_bucket,
    )
    if score_samples_fn is None:
        finalize_test_samples(
            samples=calibration_samples,
            edges_df=edges,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            negative_ratio=calibration_negative_ratio,
            random_seed=random_seed,
            build_prompt_features=False,
            compute_expert_prediction=False,
            compute_rrf_scores=True,
            rrf_k=args.rrf_k,
            rrf_mode=args.rrf_mode,
            sequential_rank_bins=args.sequential_rank_bins,
            expert_prediction_mode="fixed_threshold",
            expert_prediction_fixed_threshold=args.expert_prediction_fixed_threshold,
            rrf_pointwise_pool_size=args.rrf_pointwise_pool_size,
            rrf_pointwise_num_pools=args.rrf_pointwise_num_pools,
            rrf_batch_size=args.rrf_batch_size,
            rrf_heuristics=args.rrf_heuristics,
            key_signal_reference="sequential_global",
            include_overall_structural_signal=False,
            overall_signal_low_threshold=args.overall_signal_low_threshold,
            overall_signal_high_threshold=args.overall_signal_high_threshold,
            embeddings=embeddings,
            entity_id_to_idx=entity_id_to_idx,
            apply_gdelt_time_bucket=apply_dtgb_time_bucket,
            **_resolve_finalize_smoothing_kwargs(args),
        )
    else:
        score_samples_fn(calibration_samples, score_field=score_field)
    scores = np.asarray(
        [float(sample.get(score_field, 0.0)) for sample in calibration_samples],
        dtype=np.float64,
    )
    labels = np.asarray(
        [int(sample.get("label", 0)) for sample in calibration_samples],
        dtype=np.int64,
    )
    summary = _derive_validation_sampled_hybrid_uncertainty_band(
        scores,
        labels,
        target_fraction=args.hybrid_validation_target_fraction,
    )
    print(
        f"Validation-sampled hybrid routing band [{score_label}]: "
        f"center={summary['center_threshold']:.6f}, "
        f"low={summary['low_threshold']:.6f}, "
        f"high={summary['high_threshold']:.6f}, "
        f"requested_fraction={summary['requested_target_fraction']:.3f}, "
        f"realized_fraction={summary['selected_fraction_realized']:.3f}"
    )
    return summary


def _calibrate_validation_sampled_hybrid_gmm_overlap_band(
    *,
    args,
    edges,
    embeddings,
    entity_id_to_idx,
    random_seed,
    score_field="rrf_score",
    score_label="RRF",
    score_samples_fn=None,
):
    apply_dtgb_time_bucket = dataset_uses_dtgb_time_bucket(args.dataset_name)
    calibration_num_samples, calibration_negative_ratio = _resolve_validation_calibration_sizes(args)
    print(
        "Calibrating validation-sampled hybrid GMM routing band: "
        f"validation positives={calibration_num_samples}, "
        f"negative_ratio={calibration_negative_ratio}, "
        f"target_fraction={float(args.hybrid_validation_target_fraction):.3f}"
    )
    calibration_samples = _build_lightweight_rrf_samples_for_split(
        edges,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        num_samples=calibration_num_samples,
        negative_ratio=calibration_negative_ratio,
        random_seed=random_seed,
        eval_split="validation",
        apply_gdelt_time_bucket=apply_dtgb_time_bucket,
    )
    if score_samples_fn is None:
        finalize_test_samples(
            samples=calibration_samples,
            edges_df=edges,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            negative_ratio=calibration_negative_ratio,
            random_seed=random_seed,
            build_prompt_features=False,
            compute_expert_prediction=False,
            compute_rrf_scores=True,
            rrf_k=args.rrf_k,
            rrf_mode=args.rrf_mode,
            sequential_rank_bins=args.sequential_rank_bins,
            expert_prediction_mode="fixed_threshold",
            expert_prediction_fixed_threshold=args.expert_prediction_fixed_threshold,
            rrf_pointwise_pool_size=args.rrf_pointwise_pool_size,
            rrf_pointwise_num_pools=args.rrf_pointwise_num_pools,
            rrf_batch_size=args.rrf_batch_size,
            rrf_heuristics=args.rrf_heuristics,
            key_signal_reference="sequential_global",
            include_overall_structural_signal=False,
            overall_signal_low_threshold=args.overall_signal_low_threshold,
            overall_signal_high_threshold=args.overall_signal_high_threshold,
            embeddings=embeddings,
            entity_id_to_idx=entity_id_to_idx,
            apply_gdelt_time_bucket=apply_dtgb_time_bucket,
            **_resolve_finalize_smoothing_kwargs(args),
        )
    else:
        score_samples_fn(calibration_samples, score_field=score_field)

    scores = np.asarray(
        [float(sample.get(score_field, 0.0)) for sample in calibration_samples],
        dtype=np.float64,
    )
    labels = np.asarray(
        [int(sample.get("label", 0)) for sample in calibration_samples],
        dtype=np.int64,
    )
    summary = _derive_validation_sampled_hybrid_gmm_overlap_band(
        scores,
        labels,
        target_fraction=args.hybrid_validation_target_fraction,
    )
    print(
        f"Validation-sampled hybrid GMM routing [{score_label}]: "
        f"means=({summary['gmm']['means'][0]:.6f}, {summary['gmm']['means'][1]:.6f}), "
        f"weights=({summary['gmm']['weights'][0]:.3f}, {summary['gmm']['weights'][1]:.3f}), "
        f"ambiguity_threshold={summary['ambiguity_threshold']:.6f}, "
        f"requested_fraction={summary['requested_target_fraction']:.3f}, "
        f"realized_fraction={summary['selected_fraction_realized']:.3f}"
    )
    return summary


def _build_realized_discriminative_debug(
    samples,
    *,
    low_threshold,
    high_threshold,
    score_field="rrf_score",
    score_label="RRF",
):
    scores = np.asarray(
        [float(sample.get(score_field, 0.0)) for sample in samples],
        dtype=np.float64,
    )
    labels = np.asarray(
        [int(sample.get("label", 0)) for sample in samples],
        dtype=np.int64,
    )
    if scores.size == 0:
        return None

    band_masks = {
        "low": scores < float(low_threshold),
        "modest": (scores >= float(low_threshold)) & (scores < float(high_threshold)),
        "high": scores >= float(high_threshold),
    }
    bands = {}
    positive_rates = {}
    for band_name, band_mask in band_masks.items():
        count = int(band_mask.sum())
        positive_rate = None
        negative_rate = None
        if count > 0:
            positive_rate = float(labels[band_mask].mean())
            negative_rate = float(1.0 - positive_rate)
            positive_rates[band_name] = positive_rate
        bands[band_name] = {
            "count": count,
            "fraction": float(count / max(scores.size, 1)),
            "positive_rate": positive_rate,
            "negative_rate": negative_rate,
        }

    monotonic_non_decreasing = None
    if all(positive_rates.get(name) is not None for name in ("low", "modest", "high")):
        monotonic_non_decreasing = bool(
            positive_rates["low"] <= positive_rates["modest"] <= positive_rates["high"]
        )

    return {
        "debug_only_uses_eval_labels": True,
        "score_field": str(score_field),
        "score_label": str(score_label),
        "low_threshold": float(low_threshold),
        "high_threshold": float(high_threshold),
        "num_samples": int(scores.size),
        "num_positive": int((labels == 1).sum()),
        "num_negative": int((labels == 0).sum()),
        "bands": bands,
        "positive_rate_gap_high_minus_low": (
            float(positive_rates["high"] - positive_rates["low"])
            if positive_rates.get("high") is not None and positive_rates.get("low") is not None
            else None
        ),
        "monotonic_non_decreasing": monotonic_non_decreasing,
    }


def _build_expert_prediction_discriminative_debug(samples):
    if not samples:
        return None

    labels = np.asarray(
        [int(sample.get("label", 0)) for sample in samples],
        dtype=np.int64,
    )
    preferred_order = {"False": 0, "Neutral": 1, "True": 2}
    raw_categories = [str(sample.get("expert_prediction", "Unknown")) for sample in samples]
    ordered_categories = [
        category
        for category in sorted(
            set(raw_categories),
            key=lambda name: (preferred_order.get(name, 100), str(name)),
        )
        if category != "Unknown"
    ]
    if not ordered_categories:
        return None

    categories = {}
    for category in ordered_categories:
        mask = np.asarray([value == category for value in raw_categories], dtype=bool)
        count = int(mask.sum())
        positive_rate = None
        negative_rate = None
        if count > 0:
            positive_rate = float(labels[mask].mean())
            negative_rate = float(1.0 - positive_rate)
        categories[category] = {
            "count": count,
            "fraction": float(count / max(len(samples), 1)),
            "positive_rate": positive_rate,
            "negative_rate": negative_rate,
        }

    monotonic_non_decreasing = None
    positive_rates_in_order = [
        categories[category]["positive_rate"]
        for category in ordered_categories
        if categories[category]["positive_rate"] is not None
    ]
    if len(positive_rates_in_order) >= 2:
        monotonic_non_decreasing = bool(
            all(
                left <= right
                for left, right in zip(positive_rates_in_order[:-1], positive_rates_in_order[1:])
            )
        )

    gap = None
    if "False" in categories and "True" in categories:
        low_rate = categories["False"]["positive_rate"]
        high_rate = categories["True"]["positive_rate"]
        if low_rate is not None and high_rate is not None:
            gap = float(high_rate - low_rate)

    return {
        "debug_only_uses_eval_labels": True,
        "num_samples": int(len(samples)),
        "num_positive": int((labels == 1).sum()),
        "num_negative": int((labels == 0).sum()),
        "categories": categories,
        "category_order": ordered_categories,
        "positive_rate_gap_true_minus_false": gap,
        "monotonic_non_decreasing": monotonic_non_decreasing,
    }


def _build_key_signal_discriminative_debug(samples):
    if not samples:
        return None

    labels = np.asarray(
        [int(sample.get("label", 0)) for sample in samples],
        dtype=np.int64,
    )
    signal_specs = (
        ("target_popularity", "target_popularity"),
        ("past_interactions", "num_past_interactions"),
        ("interaction_recency", "last_interaction_str"),
        ("common_neighbor", "common_neighbor_level"),
        ("global_recency", "global_recency_level"),
    )
    preferred_order = {
        "Low": 0,
        "No prior interactions": 1,
        "No prior target interactions": 1,
        "Modest": 2,
        "Neutral": 2,
        "Unknown": 3,
        "High": 4,
    }

    def _order_key(name):
        return (preferred_order.get(name, 100), str(name))

    signal_debug = {}
    for debug_name, sample_key in signal_specs:
        categories = {}
        raw_values = [str(sample.get(sample_key, "Unknown")) for sample in samples]
        for category in sorted(set(raw_values), key=_order_key):
            mask = np.asarray([value == category for value in raw_values], dtype=bool)
            count = int(mask.sum())
            positive_rate = None
            negative_rate = None
            if count > 0:
                positive_rate = float(labels[mask].mean())
                negative_rate = float(1.0 - positive_rate)
            categories[category] = {
                "count": count,
                "fraction": float(count / max(len(samples), 1)),
                "positive_rate": positive_rate,
                "negative_rate": negative_rate,
            }

        monotonic_non_decreasing = None
        if all(
            name in categories and categories[name]["positive_rate"] is not None
            for name in ("Low", "Modest", "High")
        ):
            monotonic_non_decreasing = bool(
                categories["Low"]["positive_rate"]
                <= categories["Modest"]["positive_rate"]
                <= categories["High"]["positive_rate"]
            )

        signal_debug[debug_name] = {
            "categories": categories,
            "monotonic_non_decreasing_low_modest_high": monotonic_non_decreasing,
        }

    return {
        "debug_only_uses_eval_labels": True,
        "num_samples": int(len(samples)),
        "num_positive": int((labels == 1).sum()),
        "num_negative": int((labels == 0).sum()),
        "signals": signal_debug,
    }
