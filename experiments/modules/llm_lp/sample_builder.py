"""
Sample construction and prompt-feature preparation for LLM link prediction.
"""
import cProfile
import io
import os
import pstats
import random
import time
from bisect import bisect_left
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from experiments.modules.llm_lp.eval_helpers import load_or_compute_embeddings
from experiments.modules.llm_lp.prompt_context import materialize_samples_prompt_context
from experiments.modules.llm_lp.semantic_ranking import (
    build_entity_idx_lookup,
    compute_semantic_dot_scores,
    get_query_time_semantic_embeddings,
    make_semantic_cache_state,
    prepare_semantic_embeddings_for_runtime,
)
from experiments.modules.llm_lp.sample_finalize import (
    _l2_normalize_rows,
    _rrf_log,
    _rrf_warn,
    finalize_test_samples,
)
from utils.utils import NegativeEdgeSampler


def _precompute_negative_targets(
    *,
    mode,
    eval_split,
    edge_src,
    edge_dst,
    edge_ts,
    val_time,
    test_time,
    test_edges,
    positive_samples,
    negative_dst_pool,
    negative_ratio,
    batch_size,
):
    mode = str(mode or "dtgb_sampler").strip().lower()
    if mode not in {"dtgb_sampler", "pool"}:
        raise ValueError(
            "negative_sampling_mode must be one of dtgb_sampler/pool, "
            f"got {mode!r}."
        )
    if mode == "pool":
        print("Negative sample mode: split destination pool")
        return None

    negative_ratio = int(negative_ratio)
    if negative_ratio < 1:
        raise ValueError("negative_ratio must be >= 1")

    if str(eval_split).strip().lower() == "inductive":
        sampler_src = test_edges["u"].values.astype(np.int64, copy=False)
        sampler_dst = test_edges["i"].values.astype(np.int64, copy=False)
        sampler_times = test_edges["_dtgb_eval_ts"].values.astype(np.float64, copy=False)
        sampler_seed = 3
    else:
        split_key = str(eval_split).strip().lower()
        if split_key == "train":
            sampler_src = test_edges["u"].values.astype(np.int64, copy=False)
            sampler_dst = test_edges["i"].values.astype(np.int64, copy=False)
            sampler_times = test_edges["_dtgb_eval_ts"].values.astype(np.float64, copy=False)
        else:
            sampler_src = edge_src
            sampler_dst = edge_dst
            sampler_times = edge_ts.astype(np.float64, copy=False)
        sampler_seed = 0 if split_key == "train" else (1 if split_key == "validation" else 2)

    val_mask = np.logical_and(edge_ts <= test_time, edge_ts > val_time)
    val_times = edge_ts[val_mask]
    last_observed_time = float(val_times[-1]) if len(val_times) else float(test_time)
    sampler = NegativeEdgeSampler(
        src_node_ids=sampler_src,
        dst_node_ids=sampler_dst,
        interact_times=sampler_times,
        last_observed_time=last_observed_time,
        negative_sample_strategy="random",
        seed=sampler_seed,
    )

    num_positive = len(positive_samples)
    targets = np.empty((num_positive, negative_ratio), dtype=np.int64)
    batch_size = max(1, int(batch_size))
    for start in range(0, num_positive, batch_size):
        end = min(num_positive, start + batch_size)
        _, neg_dst = sampler.sample(size=(end - start) * negative_ratio)
        targets[start:end] = np.asarray(neg_dst, dtype=np.int64).reshape(
            end - start,
            negative_ratio,
        )

    print(
        "Negative sample mode: DTGB NegativeEdgeSampler "
        f"(strategy=random, seed={sampler_seed}, batch_size={batch_size})"
    )
    return targets


def _sanitize_profile_phase_tag(value):
    raw = str(value or "sample_creation").strip().lower()
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in raw)
    cleaned = cleaned.strip("_")
    return cleaned or "sample_creation"


def _resolve_profile_output_path(base_path, phase_label):
    if not base_path:
        return None
    base_path = str(base_path).strip()
    if not base_path:
        return None
    phase_tag = _sanitize_profile_phase_tag(phase_label)
    if "{phase}" in base_path:
        return base_path.format(phase=phase_tag)
    root, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".prof"
    return f"{root}.{phase_tag}{ext}"


def _emit_sample_creation_profile(
    profiler,
    *,
    phase_label,
    sort_by,
    top_n,
    output_path=None,
):
    if profiler is None:
        return

    resolved_output = _resolve_profile_output_path(output_path, phase_label)
    if resolved_output:
        try:
            profiler.dump_stats(resolved_output)
            print(f"[SampleProfile] Raw cProfile stats written: {resolved_output}")
        except Exception as exc:
            print(f"[SampleProfile][WARN] Failed to dump raw profile stats ({exc})")

    stats_stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stats_stream).strip_dirs()
    try:
        stats.sort_stats(sort_by)
        active_sort = sort_by
    except Exception:
        active_sort = "cumtime"
        stats.sort_stats(active_sort)
        print(
            f"[SampleProfile][WARN] Unsupported sort '{sort_by}', "
            f"falling back to '{active_sort}'."
        )
    stats.print_stats(int(max(1, top_n)))
    print(
        f"[SampleProfile] phase={phase_label}, sort={active_sort}, top_n={int(max(1, top_n))}"
    )
    report = stats_stream.getvalue().rstrip()
    if report:
        print(report)


def _build_sample_record(
    entity_map,
    relation_map,
    source_id,
    relation_id,
    target_id,
    timestamp,
    source_history,
    target_history,
    source_history_entities,
    target_history_entities,
    mutual_history,
    num_past_interactions,
    global_avg_interactions,
    source_popularity_raw,
    target_popularity_raw,
    last_interaction_delta,
    avg_node_popularity,
    common_neighbors,
    common_neighbors_desc,
    source_history_desc,
    target_history_desc,
    query_id,
    label,
    dtgb_timestamp=None,
):
    return {
        "source_id": source_id,
        "relation_id": relation_id,
        "target_id": target_id,
        "source_entity": entity_map.get(source_id, f"entity_{source_id}"),
        "relation": relation_map.get(relation_id, f"relation_{relation_id}"),
        "target_entity": entity_map.get(target_id, f"entity_{target_id}"),
        "timestamp": timestamp,
        "dtgb_timestamp": (
            timestamp if dtgb_timestamp is None else dtgb_timestamp
        ),
        "source_history": source_history,
        "target_history": target_history,
        "source_history_entities": source_history_entities,
        "target_history_entities": target_history_entities,
        "mutual_history": mutual_history,
        "num_past_interactions": num_past_interactions,
        "num_past_interactions_raw": num_past_interactions,
        "global_avg_interactions": global_avg_interactions,
        "source_popularity": "Modest",
        "target_popularity": "Modest",
        "source_popularity_raw": source_popularity_raw,
        "target_popularity_raw": target_popularity_raw,
        "last_interaction_delta": last_interaction_delta,
        "avg_node_popularity": avg_node_popularity,
        "common_neighbors": common_neighbors,
        "common_neighbors_desc": common_neighbors_desc,
        "source_history_desc": source_history_desc,
        "target_history_desc": target_history_desc,
        "prompt_context_materialized": bool(
            source_history
            or target_history
            or source_history_entities
            or target_history_entities
            or mutual_history
            or common_neighbors
        ),
        "query_id": int(query_id),
        "label": label,
        "rrf_score": 0.0,
        "rrf_rank": None,
        "common_neighbor_score": 0.0,
        "common_neighbor_level": "Modest",
        "global_recency_level": "No prior target interactions",
        "itemcf_level": "Low",
        "usercf_level": "Low",
        "last_interaction_str": "No prior interactions",
        "source_popularity_pct": 50.0,
        "target_popularity_pct": 50.0,
        "num_past_interactions_pct": 50.0,
        "common_neighbor_score_pct": 50.0,
        "last_interaction_recency_pct": 0.0,
        "global_recency_pct": 0.0,
        "itemcf_pct": 0.0,
        "usercf_pct": 0.0,
        "heuristic_recency_score": None,
        "heuristic_popularity_score": None,
        "heuristic_past_interactions_score": None,
        "heuristic_resource_allocation_score": None,
        "heuristic_global_recency_score": None,
        "heuristic_itemcf_score": None,
        "heuristic_usercf_score": None,
        "heuristic_semantic_smoothing_score": None,
        "heuristic_recency_rank": None,
        "heuristic_popularity_rank": None,
        "heuristic_past_interactions_rank": None,
        "heuristic_resource_allocation_rank": None,
        "heuristic_global_recency_rank": None,
        "heuristic_semantic_smoothing_rank": None,
    }


def create_test_samples(
    edges_df,
    entity_map,
    relation_map,
    test_ratio=0.15,
    val_ratio=0.15,
    num_samples=100,
    negative_ratio=1,
    random_seed=42,
    history_window=10,
    semantic_history=False,
    semantic_topk=None,
    semantic_history_entity_mode=False,
    common_neighbors_semantic=False,
    build_prompt_features=True,
    defer_prompt_context_materialization=False,
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
    compute_expert_prediction=True,
    compute_rrf_scores=None,
    rrf_k=60,
    rrf_mode="query_local",
    sequential_rank_bins=1024,
    expert_prediction_mode="sequential_running_median",
    expert_prediction_fixed_threshold=0.05,
    rrf_pointwise_pool_size=256,
    rrf_pointwise_num_pools=4,
    rrf_batch_size=200000,
    rrf_heuristics=None,
    eval_split="transductive",
    key_signal_reference="sequential_global",
    key_signal_fields=None,
    include_overall_structural_signal=False,
    overall_signal_low_threshold=0.0475,
    overall_signal_high_threshold=0.0510,
    defer_postprocessing=False,
    apply_gdelt_time_bucket=False,
    sample_creation_monitor_every=0,
    sample_creation_profile=False,
    sample_creation_profile_sort="cumtime",
    sample_creation_profile_top_n=40,
    sample_creation_profile_output=None,
    skip_key_signal_calibration=False,
    heuristic_recent_degree_window=30.0,
    positive_sample_mode="random",
    positive_sample_block_size=None,
    negative_sampling_mode="dtgb_sampler",
    negative_sampling_batch_size=256,
):
    """
    Create test samples following DTGB protocol.
    eval_split:
      - train: DTGB observed training edges through the validation boundary,
        excluding held-out inductive nodes
      - transductive: all edges in the test period
      - inductive: only test edges with at least one node unseen in train
      - validation: all edges in the validation period
    """
    np.random.seed(random_seed)

    if compute_rrf_scores is None:
        compute_rrf_scores = compute_expert_prediction
    rrf_mode = str(rrf_mode).strip().lower()
    if rrf_mode not in {"query_local", "sequential_pointwise", "train_pool_pointwise"}:
        raise ValueError(
            "Unsupported rrf_mode="
            f"{rrf_mode}. Use query_local, sequential_pointwise, or train_pool_pointwise."
        )
    expert_prediction_mode = str(expert_prediction_mode).strip().lower()
    if expert_prediction_mode not in {"global_median", "sequential_running_median", "fixed_threshold"}:
        raise ValueError(
            "Unsupported expert_prediction_mode="
            f"{expert_prediction_mode}. Use global_median, sequential_running_median, or fixed_threshold."
        )
    expert_prediction_fixed_threshold = float(expert_prediction_fixed_threshold)
    if not np.isfinite(expert_prediction_fixed_threshold):
        raise ValueError(
            "expert_prediction_fixed_threshold must be a finite float; "
            f"got {expert_prediction_fixed_threshold}."
        )
    overall_signal_low_threshold = float(overall_signal_low_threshold)
    overall_signal_high_threshold = float(overall_signal_high_threshold)
    if not np.isfinite(overall_signal_low_threshold) or not np.isfinite(overall_signal_high_threshold):
        raise ValueError(
            "overall_signal thresholds must be finite floats; "
            f"got low={overall_signal_low_threshold}, high={overall_signal_high_threshold}."
        )
    if overall_signal_low_threshold >= overall_signal_high_threshold:
        raise ValueError(
            "overall_signal_low_threshold must be < overall_signal_high_threshold; "
            f"got low={overall_signal_low_threshold}, high={overall_signal_high_threshold}."
        )
    rrf_pointwise_pool_size = max(2, int(rrf_pointwise_pool_size))
    rrf_pointwise_num_pools = max(1, int(rrf_pointwise_num_pools))

    if semantic_history_entity_mode:
        semantic_history = True

    build_prompt_features = bool(build_prompt_features)
    defer_prompt_context_materialization = bool(defer_prompt_context_materialization)
    materialize_prompt_context = build_prompt_features and (not defer_prompt_context_materialization)
    sample_creation_monitor_every = int(max(0, sample_creation_monitor_every))
    sample_creation_monitor_enabled = sample_creation_monitor_every > 0
    sample_creation_profile = bool(sample_creation_profile)
    sample_creation_profile_sort = str(sample_creation_profile_sort or "cumtime").strip().lower()
    supported_profile_sort_keys = {"cumtime", "tottime", "calls", "ncalls", "time"}
    if sample_creation_profile_sort not in supported_profile_sort_keys:
        raise ValueError(
            "Unsupported sample_creation_profile_sort="
            f"{sample_creation_profile_sort}. Use one of {sorted(supported_profile_sort_keys)}."
        )
    sample_creation_profile_top_n = int(sample_creation_profile_top_n)
    if sample_creation_profile_top_n < 1:
        raise ValueError("sample_creation_profile_top_n must be >= 1.")
    sample_creation_profile_output = (
        str(sample_creation_profile_output).strip()
        if sample_creation_profile_output
        else None
    )
    sample_creation_monitor_started_at = time.perf_counter() if sample_creation_monitor_enabled else None
    sample_creation_setup_sec = 0.0
    sample_creation_stage_sec = {
        "semantic_lookup": 0.0,
        "pos_history": 0.0,
        "pos_mutual": 0.0,
        "pos_neighbors": 0.0,
        "pos_common_neighbors": 0.0,
        "pos_record_build": 0.0,
        "neg_branch": 0.0,
    }
    sample_creation_positive_count = 0
    sample_creation_negative_count = 0
    sample_creation_last_report_count = 0
    sample_creation_semantic_calls = 0
    sample_creation_semantic_cache_hits = 0
    sample_creation_semantic_recomputes = 0
    key_signal_reference = str(key_signal_reference).strip().lower()
    if key_signal_reference not in {"sequential_global", "contextual"}:
        raise ValueError(
            "Unsupported key_signal_reference="
            f"{key_signal_reference}. Use sequential_global or contextual."
        )
    semantic_smoothing_device = "cpu"
    semantic_ranking_enabled = materialize_prompt_context and (
        semantic_history or common_neighbors_semantic
    )
    semantic_embeddings_base = embeddings
    semantic_embeddings_for_smoothing = embeddings
    entity_idx_lookup = None
    semantic_cache_state = make_semantic_cache_state()
    if semantic_ranking_enabled:
        if semantic_topk is None:
            semantic_topk = history_window
        else:
            semantic_topk = min(semantic_topk, history_window)
        semantic_smoothing_device = "cuda" if semantic_use_smoothing and torch.cuda.is_available() else "cpu"
        if semantic_history and history_pool_size is None and history_pool_window is None:
            history_pool_size = history_window * 5
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
                warn_fn=_rrf_warn,
            )
        )
        if semantic_use_smoothing:
            print("Semantic ranking uses query-time smoothed cosine (strict < current timestamp).")
            print(
                "Smoothing config for semantic ranking: "
                f"window={smooth_time_window}, steps={smooth_steps}, "
                f"decay_gamma={smooth_decay_gamma}, undirected={smooth_undirected}, "
                f"device={semantic_smoothing_device}"
            )
        else:
            print("Semantic ranking uses original raw-embedding cosine (no smoothing).")
        if semantic_hub_penalty_alpha > 0.0:
            print(
                "Semantic source-history hub penalty is enabled: "
                f"alpha={semantic_hub_penalty_alpha:.4f}"
            )

    history_preserve_recent_k = int(max(0, history_preserve_recent_k))
    if history_preserve_recent_k > history_window:
        print(
            f"INFO: history_preserve_recent_k={history_preserve_recent_k} exceeds "
            f"history_window={history_window}; clipping to {history_window}."
        )
        history_preserve_recent_k = history_window
    if semantic_history_entity_mode and history_preserve_recent_k > 0:
        print(
            "INFO: history_preserve_recent_k affects semantic event selection only; "
            "entity-centric semantic history remains recency-first."
        )

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

    edge_ts_raw = edges_df["ts"].values.astype(np.float64)
    edge_ts_for_split = edge_ts_raw
    if apply_gdelt_time_bucket:
        edge_ts_for_split = np.floor_divide(edge_ts_raw.astype(np.int64), 15).astype(np.float64)
        print("Applying DTGB GDELT time bucketing for split masks: ts // 15")

    val_time = np.quantile(edge_ts_for_split, 1 - val_ratio - test_ratio)
    test_time = np.quantile(edge_ts_for_split, 1 - test_ratio)

    print("\nData split (DTGB protocol):")
    print(f"  Train: t <= {val_time:.0f} (held-out inductive nodes excluded)")
    print(f"  Val: {val_time:.0f} < t <= {test_time:.0f}")
    if eval_split == "train":
        print(f"  Selected: observed train edges with t <= {val_time:.0f}")
    elif eval_split == "transductive":
        print(f"  Test: t > {test_time:.0f}")
    else:
        print(f"  Test: t > {test_time:.0f}")

    edge_src = edges_df["u"].values.astype(np.int64)
    edge_dst = edges_df["i"].values.astype(np.int64)
    edge_ts = edge_ts_for_split

    if eval_split == "train":
        split_random = random.Random(2020)
        node_set = set(edge_src).union(set(edge_dst))
        test_node_set = set(edge_src[edge_ts > val_time]).union(
            set(edge_dst[edge_ts > val_time])
        )
        held_out_nodes = set(
            split_random.sample(list(test_node_set), int(0.1 * len(node_set)))
        )
        observed_edges_mask = ~(
            np.isin(edge_src, np.fromiter(held_out_nodes, dtype=np.int64))
            | np.isin(edge_dst, np.fromiter(held_out_nodes, dtype=np.int64))
        )
        test_mask = np.logical_and(edge_ts <= val_time, observed_edges_mask)
        test_edges = edges_df[test_mask].copy()
        test_edges["_dtgb_eval_ts"] = edge_ts_for_split[test_mask]
        negative_dst_pool = np.unique(edge_dst[test_mask]).astype(np.int64)
        print(
            f"\nTrain set: {len(test_edges)} edges "
            f"(held-out inductive nodes excluded={len(held_out_nodes)})"
        )
    elif eval_split == "transductive":
        test_mask = edge_ts > test_time
        test_edges = edges_df[test_mask].copy()
        test_edges["_dtgb_eval_ts"] = edge_ts_for_split[test_mask]
        negative_dst_pool = np.unique(edge_dst).astype(np.int64)
        print(f"\nTest set (transductive): {len(test_edges)} edges")
    elif eval_split == "validation":
        test_mask = np.logical_and(edge_ts > val_time, edge_ts <= test_time)
        test_edges = edges_df[test_mask].copy()
        test_edges["_dtgb_eval_ts"] = edge_ts_for_split[test_mask]
        negative_dst_pool = np.unique(edge_dst[edge_ts <= test_time]).astype(np.int64)
        print(f"\nValidation set: {len(test_edges)} edges")
    elif eval_split == "inductive":
        random.seed(2020)
        node_set = set(edge_src).union(set(edge_dst))
        num_total_unique_node_ids = len(node_set)
        test_node_set = set(edge_src[edge_ts > val_time]).union(set(edge_dst[edge_ts > val_time]))
        new_test_node_set = set(
            random.sample(list(test_node_set), int(0.1 * num_total_unique_node_ids))
        )

        new_test_source_mask = edges_df.u.map(lambda x: x in new_test_node_set).values
        new_test_destination_mask = edges_df.i.map(lambda x: x in new_test_node_set).values
        observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)
        train_mask = np.logical_and(edge_ts <= val_time, observed_edges_mask)

        train_node_set = set(edge_src[train_mask]).union(set(edge_dst[train_mask]))
        assert len(train_node_set & new_test_node_set) == 0
        new_node_set = node_set - train_node_set

        edge_contains_new_node_mask = np.array(
            [
                (src_node_id in new_node_set or dst_node_id in new_node_set)
                for src_node_id, dst_node_id in zip(edge_src, edge_dst)
            ]
        )
        test_mask = edge_ts > test_time
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)
        test_edges = edges_df[new_node_test_mask].copy()
        test_edges["_dtgb_eval_ts"] = edge_ts_for_split[new_node_test_mask]
        negative_dst_pool = np.unique(edge_dst[new_node_test_mask]).astype(np.int64)
        print(
            f"\nTest set (inductive): {len(test_edges)} edges "
            f"(new_test_nodes={len(new_test_node_set)}, unseen_nodes={len(new_node_set)})"
        )
        print(
            f"{len(new_test_node_set)} nodes were used for inductive testing "
            "(never seen during training)"
        )
    else:
        raise ValueError(f"Unsupported eval_split: {eval_split}")

    if len(negative_dst_pool) == 0:
        raise RuntimeError(
            f"No destination nodes available for DTGB-style negative sampling in split '{eval_split}'."
        )
    print(f"DTGB negative dst candidate pool size: {len(negative_dst_pool)}")
    positive_sample_mode = str(positive_sample_mode or "random").strip().lower()
    if positive_sample_mode not in {"random", "contiguous", "most_recent"}:
        raise ValueError(
            "positive_sample_mode must be one of random/contiguous/most_recent, "
            f"got {positive_sample_mode!r}."
        )

    if len(test_edges) > num_samples:
        if positive_sample_mode == "most_recent":
            selected = test_edges.sort_values(
                "_dtgb_eval_ts", ascending=False, kind="mergesort"
            ).head(int(num_samples))
            positive_samples = selected.sort_values(
                "_dtgb_eval_ts", ascending=True, kind="mergesort"
            )
            print(
                "Positive sample mode: most_recent "
                f"(selected={len(positive_samples)}, total={len(test_edges)})"
            )
        elif positive_sample_mode == "contiguous":
            rng = np.random.RandomState(int(random_seed))
            block_size = (
                int(positive_sample_block_size)
                if positive_sample_block_size is not None
                else int(num_samples)
            )
            block_size = max(1, min(block_size, int(num_samples), len(test_edges)))
            candidate_starts = np.arange(0, len(test_edges), block_size, dtype=np.int64)
            rng.shuffle(candidate_starts)
            selected_blocks = []
            selected_count = 0
            for start_raw in candidate_starts:
                if selected_count >= int(num_samples):
                    break
                start = int(start_raw)
                end = min(start + block_size, len(test_edges))
                take = min(end - start, int(num_samples) - selected_count)
                if take <= 0:
                    continue
                selected_blocks.append((start, start + take))
                selected_count += take
            selected_blocks.sort(key=lambda item: item[0])
            selected_positions = np.concatenate(
                [
                    np.arange(start, end, dtype=np.int64)
                    for start, end in selected_blocks
                ],
                axis=0,
            )
            positive_samples = test_edges.iloc[selected_positions]
            print(
                "Positive sample mode: contiguous "
                f"(blocks={len(selected_blocks)}, block_size={block_size}, "
                f"selected={len(positive_samples)}, total={len(test_edges)})"
            )
        else:
            positive_samples = test_edges.sample(n=num_samples, random_state=random_seed)
            print("Positive sample mode: random")
    else:
        positive_samples = test_edges
        print("Positive sample mode: full_split")

    print(f"Sampled {len(positive_samples)} positive test samples")
    precomputed_negative_targets = _precompute_negative_targets(
        mode=negative_sampling_mode,
        eval_split=eval_split,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_ts=edge_ts,
        val_time=val_time,
        test_time=test_time,
        test_edges=test_edges,
        positive_samples=positive_samples,
        negative_dst_pool=negative_dst_pool,
        negative_ratio=negative_ratio,
        batch_size=negative_sampling_batch_size,
    )
    negative_target_rows = (
        None if precomputed_negative_targets is None else iter(precomputed_negative_targets)
    )

    if defer_prompt_context_materialization and build_prompt_features:
        print(
            "Hybrid deferred prompt materialization is enabled: "
            "building lightweight full-split samples before hybrid selection."
        )

        num_nodes = len(entity_map)
        num_edges = len(edges_df)
        total_pairs = (num_nodes * (num_nodes - 1)) / 2
        global_avg_interactions = num_edges / total_pairs if total_pairs > 0 else 0
        avg_node_popularity = (2 * num_edges) / num_nodes if num_nodes > 0 else 0

        samples = []
        deferred_profile = None
        if sample_creation_profile:
            deferred_profile = cProfile.Profile()
            deferred_profile.enable()
        for query_id, row in tqdm(
            positive_samples.iterrows(),
            total=len(positive_samples),
            desc="Creating samples",
        ):
            source_id = int(row["u"])
            relation_id = int(row["r"])
            target_id = int(row["i"])
            timestamp = int(row["ts"])
            dtgb_timestamp = int(row["_dtgb_eval_ts"]) if apply_gdelt_time_bucket else timestamp

            samples.append(
                _build_sample_record(
                    entity_map=entity_map,
                    relation_map=relation_map,
                    source_id=source_id,
                    relation_id=relation_id,
                    target_id=target_id,
                    timestamp=timestamp,
                    dtgb_timestamp=dtgb_timestamp,
                    source_history=[],
                    target_history=[],
                    source_history_entities=[],
                    target_history_entities=[],
                    mutual_history=[],
                    num_past_interactions=0,
                    global_avg_interactions=global_avg_interactions,
                    source_popularity_raw=0,
                    target_popularity_raw=0,
                    last_interaction_delta=None,
                    avg_node_popularity=avg_node_popularity,
                    common_neighbors=[],
                    common_neighbors_desc="sorted by popularity",
                    source_history_desc="most recent",
                    target_history_desc="most recent",
                    query_id=query_id,
                    label=1,
                )
            )

            if negative_target_rows is None:
                sampled_neg_indices = np.random.randint(0, len(negative_dst_pool), size=negative_ratio)
                sampled_negatives = negative_dst_pool[sampled_neg_indices]
            else:
                sampled_negatives = next(negative_target_rows)
            for neg_target_id in sampled_negatives:
                samples.append(
                    _build_sample_record(
                        entity_map=entity_map,
                        relation_map=relation_map,
                        source_id=source_id,
                        relation_id=relation_id,
                        target_id=int(neg_target_id),
                        timestamp=timestamp,
                        dtgb_timestamp=dtgb_timestamp,
                        source_history=[],
                        target_history=[],
                        source_history_entities=[],
                        target_history_entities=[],
                        mutual_history=[],
                        num_past_interactions=0,
                        global_avg_interactions=global_avg_interactions,
                        source_popularity_raw=0,
                        target_popularity_raw=0,
                        last_interaction_delta=None,
                        avg_node_popularity=avg_node_popularity,
                        common_neighbors=[],
                        common_neighbors_desc="sorted by popularity",
                        source_history_desc="most recent",
                        target_history_desc="most recent",
                        query_id=query_id,
                        label=0,
                    )
                )

        if deferred_profile is not None:
            deferred_profile.disable()
            _emit_sample_creation_profile(
                deferred_profile,
                phase_label=f"create_samples_{eval_split}_deferred_loop",
                sort_by=sample_creation_profile_sort,
                top_n=sample_creation_profile_top_n,
                output_path=sample_creation_profile_output,
            )

        materialize_samples_prompt_context(
            samples,
            edges_df=edges_df,
            entity_map=entity_map,
            history_window=history_window,
            semantic_history=semantic_history,
            semantic_topk=semantic_topk,
            semantic_history_entity_mode=semantic_history_entity_mode,
            common_neighbors_semantic=common_neighbors_semantic,
            semantic_use_smoothing=semantic_use_smoothing,
            semantic_hub_penalty_alpha=semantic_hub_penalty_alpha,
            semantic_fusion_alpha=semantic_fusion_alpha,
            semantic_fusion_tau=semantic_fusion_tau,
            semantic_fusion_recency_speed=semantic_fusion_recency_speed,
            history_pool_size=history_pool_size,
            history_pool_window=history_pool_window,
            history_preserve_recent_k=history_preserve_recent_k,
            embeddings=embeddings,
            entity_id_to_idx=entity_id_to_idx,
            embedding_model=embedding_model,
            embedding_cache=embedding_cache,
            smooth_time_window=smooth_time_window,
            smooth_steps=smooth_steps,
            smooth_decay_gamma=smooth_decay_gamma,
            smooth_undirected=smooth_undirected,
            populate_prompt_lists=False,
            monitor_label="Hybrid structural preparation",
        )
        deferred_count = sum(
            1 for sample in samples if not sample.get("prompt_context_materialized", False)
        )
        print(
            "Hybrid lightweight preparation complete: "
            f"deferred prompt context for {deferred_count}/{len(samples)} full-set samples."
        )

        num_pos = sum(1 for sample in samples if sample["label"] == 1)
        num_neg = sum(1 for sample in samples if sample["label"] == 0)
        print(f"\nCreated {len(samples)} total samples: {num_pos} positive, {num_neg} negative")

        if defer_postprocessing:
            print("Deferring RRF/key-signal postprocessing to a later phase.")
            return samples

        deferred_contextual_key_signals = (
            bool(build_prompt_features) and str(key_signal_reference).strip().lower() == "contextual"
        )
        if deferred_contextual_key_signals:
            print("Deferring prompt-side contextual key-signal calibration to selected-slice enrichment.")
        finalize_test_samples(
            samples=samples,
            edges_df=edges_df,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            negative_ratio=negative_ratio,
            random_seed=random_seed,
            build_prompt_features=(build_prompt_features and (not deferred_contextual_key_signals)),
            compute_expert_prediction=compute_expert_prediction,
            compute_rrf_scores=compute_rrf_scores,
            rrf_k=rrf_k,
            rrf_mode=rrf_mode,
            sequential_rank_bins=sequential_rank_bins,
            expert_prediction_mode=expert_prediction_mode,
            expert_prediction_fixed_threshold=expert_prediction_fixed_threshold,
            rrf_pointwise_pool_size=rrf_pointwise_pool_size,
            rrf_pointwise_num_pools=rrf_pointwise_num_pools,
            rrf_batch_size=rrf_batch_size,
            key_signal_reference=key_signal_reference,
            include_overall_structural_signal=include_overall_structural_signal,
            overall_signal_low_threshold=overall_signal_low_threshold,
            overall_signal_high_threshold=overall_signal_high_threshold,
            use_gpu_heuristics=True,
            apply_gdelt_time_bucket=apply_gdelt_time_bucket,
            skip_key_signal_calibration=skip_key_signal_calibration,
        )

        return samples

    need_event_histories = build_prompt_features
    if need_event_histories:
        print("Pre-computing entity histories...")
    else:
        print("Skipping entity-history pre-computation (prompt-context disabled).")

    edges_df_sorted = edges_df.sort_values("ts")
    u_vals = edges_df_sorted["u"].values
    r_vals = edges_df_sorted["r"].values
    i_vals = edges_df_sorted["i"].values
    ts_vals = edges_df_sorted["ts"].values
    ts_vals_float = ts_vals.astype(np.float64, copy=False)
    split_ts_vals = ts_vals
    if apply_gdelt_time_bucket:
        split_ts_vals = np.floor_divide(ts_vals.astype(np.int64), 15).astype(np.float64)

    history_as_source = defaultdict(list)
    history_as_target = defaultdict(list)
    pair_history_as_source = defaultdict(lambda: defaultdict(list))
    if need_event_histories:
        for u, r, i, ts in tqdm(zip(u_vals, r_vals, i_vals, ts_vals), total=len(u_vals), desc="Indexing"):
            u, r, i, ts = int(u), int(r), int(i), int(ts)
            event = (u, r, i, ts)
            history_as_source[u].append(event)
            history_as_target[i].append(event)
            pair_history_as_source[u][i].append(event)

    mapped_u_vals = None
    mapped_i_vals = None
    if semantic_ranking_enabled and semantic_use_smoothing:
        mapped_u_vals = np.array([entity_id_to_idx.get(int(u), -1) for u in u_vals], dtype=np.int64)
        mapped_i_vals = np.array([entity_id_to_idx.get(int(i), -1) for i in i_vals], dtype=np.int64)
        print("Semantic-ranking smoothing backend prepared (dynamic per-query timestamp).")

    num_nodes = len(entity_map)
    num_edges = len(edges_df)
    total_pairs = (num_nodes * (num_nodes - 1)) / 2
    global_avg_interactions = num_edges / total_pairs if total_pairs > 0 else 0
    avg_node_popularity = (2 * num_edges) / num_nodes if num_nodes > 0 else 0

    max_dst_degree = max((len(value) for value in history_as_target.values()), default=1)
    dst_pop_norm_denom = np.log1p(float(max_dst_degree))

    train_entity_ids = []
    if compute_rrf_scores and rrf_mode == "train_pool_pointwise":
        train_pool_cutoff_ts = float(val_time)
        train_node_ids = set()

        for u, r, i, split_ts in zip(u_vals, r_vals, i_vals, split_ts_vals):
            split_ts = float(split_ts)
            if split_ts >= train_pool_cutoff_ts:
                break
            u = int(u)
            i = int(i)

            train_node_ids.add(u)
            train_node_ids.add(i)

        train_entity_ids = [int(node_id) for node_id in train_node_ids if int(node_id) != 0]

    samples = []

    def get_recent_history(event_list, timestamp, limit=10):
        idx = bisect_left(event_list, timestamp, key=lambda x: x[3])
        start = max(0, idx - limit)
        return event_list[start:idx]

    def get_outgoing_pair_history(src_id, dst_id):
        src_id = int(src_id)
        dst_id = int(dst_id)
        return pair_history_as_source[src_id].get(dst_id, [])

    def merge_sorted_events_by_ts(left_events, right_events):
        if not left_events:
            return right_events
        if not right_events:
            return left_events

        merged = []
        left_idx = 0
        right_idx = 0
        left_len = len(left_events)
        right_len = len(right_events)

        while left_idx < left_len and right_idx < right_len:
            if left_events[left_idx][3] <= right_events[right_idx][3]:
                merged.append(left_events[left_idx])
                left_idx += 1
            else:
                merged.append(right_events[right_idx])
                right_idx += 1

        if left_idx < left_len:
            merged.extend(left_events[left_idx:])
        if right_idx < right_len:
            merged.extend(right_events[right_idx:])
        return merged

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
        nonlocal sample_creation_semantic_calls
        nonlocal sample_creation_semantic_cache_hits
        nonlocal sample_creation_semantic_recomputes
        if sample_creation_monitor_enabled:
            sample_creation_semantic_calls += 1
        if not semantic_use_smoothing:
            return semantic_embeddings_base
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
        if sample_creation_monitor_enabled:
            sample_creation_semantic_cache_hits = int(semantic_cache_state["hits"])
            sample_creation_semantic_recomputes = int(semantic_cache_state["misses"])
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
        sims, valid_mask = compute_semantic_dot_scores(
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
        if semantic_embs is None or entity_id_to_idx is None:
            return list(node_ids)

        ref_idx = entity_id_to_idx.get(ref_id)
        if ref_idx is None:
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
        if semantic_history:
            pool = get_history_pool(
                history_as_source[source_id],
                timestamp,
                pool_size=history_pool_size,
                pool_window=history_pool_window,
            )
            if semantic_history_entity_mode:
                entity_ts_map = defaultdict(list)
                for evt in pool:
                    entity_ts_map[int(evt[2])].append(int(evt[3]))
                grouped = select_topk_entity_series(
                    entity_ts_map=entity_ts_map,
                    ref_id=target_id,
                    top_k=semantic_topk,
                    semantic_embs=semantic_embs,
                )
                return [], grouped
            return (
                select_topk_semantic(
                    pool,
                    target_id,
                    None,
                    semantic_topk,
                    semantic_embs,
                    query_ts=timestamp,
                    apply_hub_penalty=True,
                ),
                [],
            )
        return get_recent_history(history_as_source[source_id], timestamp, limit=history_window), []

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

            target_other_ids = [
                int(evt[2]) if int(evt[0]) == int(target_id) else int(evt[0])
                for evt in pool
            ]

            return (
                select_topk_semantic(
                    pool,
                    source_id,
                    None,
                    semantic_topk,
                    semantic_embs,
                    query_ts=timestamp,
                    apply_hub_penalty=False,
                    other_ids=target_other_ids,
                ),
                [],
            )

        target_hist_as_source = get_recent_history(
            history_as_source[target_id],
            timestamp,
            limit=history_window,
        )
        target_hist_as_target = get_recent_history(
            history_as_target[target_id],
            timestamp,
            limit=history_window,
        )
        target_history_list = sorted(target_hist_as_source + target_hist_as_target, key=lambda x: x[3])
        if len(target_history_list) > history_window:
            target_history_list = target_history_list[-history_window:]
        return target_history_list, []

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

    def maybe_report_sample_creation_monitor(force=False):
        nonlocal sample_creation_last_report_count
        if not sample_creation_monitor_enabled or sample_creation_positive_count <= 0:
            return
        if force and sample_creation_last_report_count == sample_creation_positive_count:
            return
        if not force:
            if (sample_creation_positive_count - sample_creation_last_report_count) < sample_creation_monitor_every:
                return
        assert sample_creation_monitor_started_at is not None
        elapsed = time.perf_counter() - sample_creation_monitor_started_at
        accounted = sample_creation_setup_sec + sum(sample_creation_stage_sec.values())
        avg_ms = lambda key: (1000.0 * sample_creation_stage_sec[key] / sample_creation_positive_count)
        neg_ms_per_positive = 1000.0 * sample_creation_stage_sec["neg_branch"] / sample_creation_positive_count
        neg_ms_per_negative = 0.0
        if sample_creation_negative_count > 0:
            neg_ms_per_negative = 1000.0 * sample_creation_stage_sec["neg_branch"] / sample_creation_negative_count
        tqdm.write(
            "[SampleMonitor] "
            f"positives={sample_creation_positive_count}/{len(positive_samples)} "
            f"negatives={sample_creation_negative_count} "
            f"elapsed={elapsed:.1f}s "
            f"rate={sample_creation_positive_count / max(elapsed, 1e-9):.2f} pos/s "
            f"setup={sample_creation_setup_sec:.2f}s "
            f"accounted={100.0 * accounted / max(elapsed, 1e-9):.1f}%"
        )
        tqdm.write(
            "[SampleMonitor] avg ms/positive: "
            f"semantic_lookup={avg_ms('semantic_lookup'):.2f}, "
            f"pos_history={avg_ms('pos_history'):.2f}, "
            f"pos_mutual={avg_ms('pos_mutual'):.2f}, "
            f"pos_neighbors={avg_ms('pos_neighbors'):.2f}, "
            f"pos_common_neighbors={avg_ms('pos_common_neighbors'):.2f}, "
            f"neg_branch={neg_ms_per_positive:.2f}, "
            f"pos_record_build={avg_ms('pos_record_build'):.2f}"
        )
        if sample_creation_negative_count > 0:
            tqdm.write(
                "[SampleMonitor] negative-branch avg: "
                f"{neg_ms_per_negative:.2f} ms/negative sample"
            )
        if semantic_ranking_enabled:
            tqdm.write(
                "[SampleMonitor] semantic calls: "
                f"{sample_creation_semantic_calls}"
                + (
                    (
                        f", smoothing_recomputes={sample_creation_semantic_recomputes}, "
                        f"cache_hits={sample_creation_semantic_cache_hits}"
                    )
                    if semantic_use_smoothing
                    else ""
                )
            )
        sample_creation_last_report_count = sample_creation_positive_count

    if not build_prompt_features:
        _rrf_log("Prompt-context construction disabled: building minimal samples for RRF-only evaluation.")

    if sample_creation_monitor_enabled:
        assert sample_creation_monitor_started_at is not None
        sample_creation_setup_sec = time.perf_counter() - sample_creation_monitor_started_at

    loop_profile = None
    if sample_creation_profile:
        loop_profile = cProfile.Profile()
        loop_profile.enable()

    for query_id, row in tqdm(
        positive_samples.iterrows(),
        total=len(positive_samples),
        desc="Creating samples",
    ):
        source_id = int(row["u"])
        relation_id = int(row["r"])
        target_id = int(row["i"])
        timestamp = int(row["ts"])
        dtgb_timestamp = int(row["_dtgb_eval_ts"]) if apply_gdelt_time_bucket else timestamp

        semantic_embs_t = None
        source_history_list, source_history_entities = [], []
        target_history_list, target_history_entities = [], []
        mutual_history_list = []
        num_past_interactions = 0
        last_interaction_delta = None
        source_popularity_raw = 0
        target_popularity_raw = 0
        common_neighbors_info = []
        src_neighbors_events = {}

        if build_prompt_features:
            if semantic_ranking_enabled:
                semantic_t0 = time.perf_counter() if sample_creation_monitor_enabled else None
                semantic_embs_t = get_semantic_embeddings_at_time(timestamp)
                if sample_creation_monitor_enabled:
                    sample_creation_stage_sec["semantic_lookup"] += time.perf_counter() - semantic_t0
            else:
                semantic_embs_t = None

            pos_hist_t0 = time.perf_counter() if sample_creation_monitor_enabled else None
            source_history_list, source_history_entities = build_source_history(
                source_id,
                target_id,
                timestamp,
                semantic_embs_t,
            )
            target_history_list, target_history_entities = build_target_history(
                source_id,
                target_id,
                timestamp,
                semantic_embs_t,
            )
            if sample_creation_monitor_enabled:
                sample_creation_stage_sec["pos_history"] += time.perf_counter() - pos_hist_t0

            pos_mutual_t0 = time.perf_counter() if sample_creation_monitor_enabled else None
            s_to_t_all = get_outgoing_pair_history(source_id, target_id)
            t_to_s_all = get_outgoing_pair_history(target_id, source_id)
            mutual_all = merge_sorted_events_by_ts(s_to_t_all, t_to_s_all)
            mutual_history_list = get_recent_history(mutual_all, timestamp, limit=history_window)
            num_past_interactions = bisect_left(mutual_all, timestamp, key=lambda x: x[3])

            if mutual_history_list:
                last_ts = mutual_history_list[-1][3]
                last_interaction_delta = timestamp - last_ts
            if sample_creation_monitor_enabled:
                sample_creation_stage_sec["pos_mutual"] += time.perf_counter() - pos_mutual_t0

            pos_neighbors_t0 = time.perf_counter() if sample_creation_monitor_enabled else None
            idx_s_source = bisect_left(history_as_source[source_id], timestamp, key=lambda x: x[3])
            idx_s_target = bisect_left(history_as_target[source_id], timestamp, key=lambda x: x[3])
            source_popularity_raw = idx_s_source + idx_s_target

            idx_t_source = bisect_left(history_as_source[target_id], timestamp, key=lambda x: x[3])
            idx_t_target = bisect_left(history_as_target[target_id], timestamp, key=lambda x: x[3])
            target_popularity_raw = idx_t_source + idx_t_target

            for idx in range(idx_s_source):
                evt = history_as_source[source_id][idx]
                src_neighbors_events[evt[2]] = evt
            for idx in range(idx_s_target):
                evt = history_as_target[source_id][idx]
                src_neighbors_events[evt[0]] = evt

            tar_neighbors_events = {}
            for idx in range(idx_t_source):
                evt = history_as_source[target_id][idx]
                tar_neighbors_events[evt[2]] = evt
            for idx in range(idx_t_target):
                evt = history_as_target[target_id][idx]
                tar_neighbors_events[evt[0]] = evt

            common_nodes = set(src_neighbors_events.keys()).intersection(tar_neighbors_events.keys())
            common_nodes.discard(source_id)
            common_nodes.discard(target_id)
            if sample_creation_monitor_enabled:
                sample_creation_stage_sec["pos_neighbors"] += time.perf_counter() - pos_neighbors_t0

            pos_cn_t0 = time.perf_counter() if sample_creation_monitor_enabled else None
            common_neighbors_info = build_common_neighbors_info(
                common_nodes=common_nodes,
                dst_id=target_id,
                src_neighbors_events=src_neighbors_events,
                tar_neighbors_events=tar_neighbors_events,
                semantic_embs=semantic_embs_t,
                timestamp=timestamp,
            )
            if sample_creation_monitor_enabled:
                sample_creation_stage_sec["pos_common_neighbors"] += time.perf_counter() - pos_cn_t0

        pos_record_t0 = time.perf_counter() if sample_creation_monitor_enabled else None
        samples.append(
            _build_sample_record(
                entity_map=entity_map,
                relation_map=relation_map,
                source_id=source_id,
                relation_id=relation_id,
                target_id=target_id,
                timestamp=timestamp,
                dtgb_timestamp=dtgb_timestamp,
                source_history=source_history_list,
                target_history=target_history_list,
                source_history_entities=source_history_entities,
                target_history_entities=target_history_entities,
                mutual_history=mutual_history_list,
                num_past_interactions=num_past_interactions,
                global_avg_interactions=global_avg_interactions,
                source_popularity_raw=source_popularity_raw,
                target_popularity_raw=target_popularity_raw,
                last_interaction_delta=last_interaction_delta,
                avg_node_popularity=avg_node_popularity,
                common_neighbors=common_neighbors_info,
                common_neighbors_desc=common_neighbors_desc,
                source_history_desc=source_history_desc,
                target_history_desc=target_history_desc,
                query_id=query_id,
                label=1,
            )
        )
        if sample_creation_monitor_enabled:
            sample_creation_stage_sec["pos_record_build"] += time.perf_counter() - pos_record_t0

        if negative_target_rows is None:
            sampled_neg_indices = np.random.randint(0, len(negative_dst_pool), size=negative_ratio)
            sampled_negatives = negative_dst_pool[sampled_neg_indices]
        else:
            sampled_negatives = next(negative_target_rows)
        neg_branch_t0 = time.perf_counter() if sample_creation_monitor_enabled else None

        for neg_target_id in sampled_negatives:
            neg_target_id = int(neg_target_id)
            neg_source_history_list, neg_source_history_entities = [], []
            neg_target_history_list, neg_target_history_entities = [], []
            neg_mutual_history_list = []
            neg_num_past_interactions = 0
            neg_last_interaction_delta = None
            neg_target_popularity_raw = 0
            neg_common_neighbors_info = []

            if build_prompt_features:
                neg_source_history_list, neg_source_history_entities = build_source_history(
                    source_id,
                    neg_target_id,
                    timestamp,
                    semantic_embs_t,
                )
                neg_target_history_list, neg_target_history_entities = build_target_history(
                    source_id,
                    neg_target_id,
                    timestamp,
                    semantic_embs_t,
                )

                neg_s_to_t_all = get_outgoing_pair_history(source_id, neg_target_id)
                neg_t_to_s_all = get_outgoing_pair_history(neg_target_id, source_id)
                neg_mutual_all = merge_sorted_events_by_ts(neg_s_to_t_all, neg_t_to_s_all)
                neg_mutual_history_list = get_recent_history(
                    neg_mutual_all,
                    timestamp,
                    limit=history_window,
                )
                neg_num_past_interactions = bisect_left(neg_mutual_all, timestamp, key=lambda x: x[3])

                if neg_mutual_history_list:
                    last_ts = neg_mutual_history_list[-1][3]
                    neg_last_interaction_delta = timestamp - last_ts

                idx_neg_t_source = bisect_left(
                    history_as_source[neg_target_id],
                    timestamp,
                    key=lambda x: x[3],
                )
                idx_neg_t_target = bisect_left(
                    history_as_target[neg_target_id],
                    timestamp,
                    key=lambda x: x[3],
                )
                neg_target_popularity_raw = idx_neg_t_source + idx_neg_t_target

                neg_tar_neighbors_events = {}
                for idx in range(idx_neg_t_source):
                    evt = history_as_source[neg_target_id][idx]
                    neg_tar_neighbors_events[evt[2]] = evt
                for idx in range(idx_neg_t_target):
                    evt = history_as_target[neg_target_id][idx]
                    neg_tar_neighbors_events[evt[0]] = evt

                neg_common_nodes = set(src_neighbors_events.keys()).intersection(
                    neg_tar_neighbors_events.keys()
                )
                neg_common_nodes.discard(source_id)
                neg_common_nodes.discard(neg_target_id)

                neg_common_neighbors_info = build_common_neighbors_info(
                    common_nodes=neg_common_nodes,
                    dst_id=neg_target_id,
                    src_neighbors_events=src_neighbors_events,
                    tar_neighbors_events=neg_tar_neighbors_events,
                    semantic_embs=semantic_embs_t,
                    timestamp=timestamp,
                )

            samples.append(
                _build_sample_record(
                    entity_map=entity_map,
                    relation_map=relation_map,
                    source_id=source_id,
                    relation_id=relation_id,
                    target_id=neg_target_id,
                    timestamp=timestamp,
                    dtgb_timestamp=dtgb_timestamp,
                    source_history=neg_source_history_list,
                    target_history=neg_target_history_list,
                    source_history_entities=neg_source_history_entities,
                    target_history_entities=neg_target_history_entities,
                    mutual_history=neg_mutual_history_list,
                    num_past_interactions=neg_num_past_interactions,
                    global_avg_interactions=global_avg_interactions,
                    source_popularity_raw=source_popularity_raw,
                    target_popularity_raw=neg_target_popularity_raw,
                    last_interaction_delta=neg_last_interaction_delta,
                    avg_node_popularity=avg_node_popularity,
                    common_neighbors=neg_common_neighbors_info,
                    common_neighbors_desc=common_neighbors_desc,
                    source_history_desc=source_history_desc,
                    target_history_desc=target_history_desc,
                    query_id=query_id,
                    label=0,
                )
            )

        if sample_creation_monitor_enabled:
            sample_creation_positive_count += 1
            sample_creation_negative_count += len(sampled_negatives)
            sample_creation_stage_sec["neg_branch"] += time.perf_counter() - neg_branch_t0
            maybe_report_sample_creation_monitor()

    if loop_profile is not None:
        loop_profile.disable()
        _emit_sample_creation_profile(
            loop_profile,
            phase_label=f"create_samples_{eval_split}_full_loop",
            sort_by=sample_creation_profile_sort,
            top_n=sample_creation_profile_top_n,
            output_path=sample_creation_profile_output,
        )

    maybe_report_sample_creation_monitor(force=True)

    num_pos = sum(1 for sample in samples if sample["label"] == 1)
    num_neg = sum(1 for sample in samples if sample["label"] == 0)
    print(
        f"\nCreated {len(samples)} total samples: {num_pos} positive, {num_neg} negative",
        flush=True,
    )

    if defer_postprocessing:
        print("Deferring RRF/key-signal postprocessing to a later phase.", flush=True)
        return samples

    finalize_test_samples(
        samples=samples,
        edges_df=edges_df,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        negative_ratio=negative_ratio,
        random_seed=random_seed,
        build_prompt_features=build_prompt_features,
        compute_expert_prediction=compute_expert_prediction,
        compute_rrf_scores=compute_rrf_scores,
        rrf_k=rrf_k,
        rrf_mode=rrf_mode,
        sequential_rank_bins=sequential_rank_bins,
        expert_prediction_mode=expert_prediction_mode,
        expert_prediction_fixed_threshold=expert_prediction_fixed_threshold,
        key_signal_fields=key_signal_fields,
        rrf_pointwise_pool_size=rrf_pointwise_pool_size,
        rrf_pointwise_num_pools=rrf_pointwise_num_pools,
        rrf_batch_size=rrf_batch_size,
        rrf_heuristics=rrf_heuristics,
        key_signal_reference=key_signal_reference,
        include_overall_structural_signal=include_overall_structural_signal,
        overall_signal_low_threshold=overall_signal_low_threshold,
        overall_signal_high_threshold=overall_signal_high_threshold,
        history_as_source=history_as_source if build_prompt_features else None,
        history_as_target=history_as_target if build_prompt_features else None,
        edge_u_vals=u_vals if build_prompt_features else None,
        edge_i_vals=i_vals if build_prompt_features else None,
        edge_ts_vals=ts_vals if build_prompt_features else None,
        train_entity_ids=train_entity_ids if compute_rrf_scores and rrf_mode == "train_pool_pointwise" else None,
        use_gpu_heuristics=True,
        apply_gdelt_time_bucket=apply_gdelt_time_bucket,
        skip_key_signal_calibration=skip_key_signal_calibration,
        embeddings=embeddings,
        entity_id_to_idx=entity_id_to_idx,
        smooth_time_window=smooth_time_window,
        smooth_steps=smooth_steps,
        smooth_decay_gamma=smooth_decay_gamma,
        smooth_undirected=smooth_undirected,
        heuristic_recent_degree_window=heuristic_recent_degree_window,
    )

    return samples


__all__ = [
    "create_test_samples",
]
