"""
Sample postprocessing: RRF scoring, key-signal calibration, and expert labeling.
"""
from bisect import insort

import numpy as np
import torch

from experiments.modules.llm_lp.prompt_context import calibrate_prompt_key_signals
from experiments.modules.llm_lp.prompt_template import normalize_key_signal_fields
from experiments.modules.prediction_metrics import compute_dtgb_ap_auc, roc_auc_score
from experiments.modules.rrf.scoring import (
    SUPPORTED_RRF_HEURISTICS,
    compute_selected_heuristic_scores_for_samples,
    compute_rrf_scores_for_samples,
    normalize_rrf_heuristics,
)
from experiments.modules.rrf.train_pool import (
    TrainPoolBuildConfig,
    TrainPoolRRFConfig,
    compute_train_pool_rrf_scores,
)
from utils.DataLoader import Data
from utils.utils import get_neighbor_sampler


def _rrf_log(message):
    print(f"[RRF] {message}", flush=True)


def _rrf_warn(message):
    print(f"[RRF][WARN] {message}", flush=True)


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _build_entity_idx_lookup(entity_id_to_idx):
    if not entity_id_to_idx:
        return None
    try:
        max_entity_id = max(int(eid) for eid in entity_id_to_idx.keys())
        num_indexed_entities = len(entity_id_to_idx)
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


def _prepare_rrf_semantic_context(
    *,
    edges_df,
    embeddings,
    entity_id_to_idx,
    smooth_time_window,
    smooth_steps,
    smooth_decay_gamma,
    smooth_undirected,
):
    if embeddings is None or entity_id_to_idx is None:
        raise ValueError(
            "Semantic-smoothing RRF requires embeddings and entity_id_to_idx to be available."
        )

    normalized_embeddings = embeddings
    if isinstance(normalized_embeddings, torch.Tensor):
        normalized_embeddings = torch.nn.functional.normalize(normalized_embeddings, p=2, dim=1)
    else:
        normalized_embeddings = _l2_normalize_rows(
            np.asarray(normalized_embeddings, dtype=np.float32)
        )

    semantic_device = "cuda" if torch.cuda.is_available() else "cpu"
    if not isinstance(normalized_embeddings, torch.Tensor) and semantic_device == "cuda":
        try:
            normalized_embeddings = torch.from_numpy(normalized_embeddings).to(
                torch.device(semantic_device)
            )
        except Exception as exc:
            _rrf_warn(
                "Failed to stage semantic RRF embeddings on "
                f"{semantic_device} ({exc}); using CPU arrays instead."
            )
            semantic_device = "cpu"

    entity_idx_lookup = _build_entity_idx_lookup(entity_id_to_idx)
    sorted_edges = edges_df.sort_values("ts")
    u_vals = sorted_edges["u"].values.astype(np.int64)
    i_vals = sorted_edges["i"].values.astype(np.int64)
    ts_vals = sorted_edges["ts"].values.astype(np.float64)

    mapped_u_vals = np.full(u_vals.shape, -1, dtype=np.int64)
    mapped_i_vals = np.full(i_vals.shape, -1, dtype=np.int64)
    if entity_idx_lookup is not None:
        u_mask = (u_vals >= 0) & (u_vals < int(entity_idx_lookup.shape[0]))
        i_mask = (i_vals >= 0) & (i_vals < int(entity_idx_lookup.shape[0]))
        mapped_u_vals[u_mask] = entity_idx_lookup[u_vals[u_mask]]
        mapped_i_vals[i_mask] = entity_idx_lookup[i_vals[i_mask]]
    else:
        mapped_u_vals = np.array(
            [entity_id_to_idx.get(int(u), -1) for u in u_vals],
            dtype=np.int64,
        )
        mapped_i_vals = np.array(
            [entity_id_to_idx.get(int(i), -1) for i in i_vals],
            dtype=np.int64,
        )

    return {
        "embeddings": normalized_embeddings,
        "entity_id_to_idx": entity_id_to_idx,
        "entity_idx_lookup": entity_idx_lookup,
        "history_src_indices": mapped_u_vals,
        "history_dst_indices": mapped_i_vals,
        "history_timestamps": ts_vals,
        "smooth_time_window": float(smooth_time_window),
        "smooth_steps": int(smooth_steps),
        "smooth_decay_gamma": smooth_decay_gamma,
        "smooth_undirected": bool(smooth_undirected),
        "device": semantic_device,
    }


def _attach_rrf_scores(samples, rrf_scores, ra_scores, heuristic_components):
    for idx, (sample, rrf_value, ra_value) in enumerate(zip(samples, rrf_scores, ra_scores)):
        sample["rrf_score"] = float(rrf_value)
        sample["common_neighbor_score"] = float(ra_value)
        sample["rrf_rank"] = int(heuristic_components["rrf_rank"][idx])
        for heuristic_name in SUPPORTED_RRF_HEURISTICS:
            score_key = f"{heuristic_name}_score"
            rank_key = f"{heuristic_name}_rank"
            sample_score_key = f"heuristic_{heuristic_name}_score"
            sample_rank_key = f"heuristic_{heuristic_name}_rank"
            if score_key in heuristic_components:
                sample[sample_score_key] = float(heuristic_components[score_key][idx])
            else:
                sample[sample_score_key] = None
            if rank_key in heuristic_components:
                sample[sample_rank_key] = int(heuristic_components[rank_key][idx])
            else:
                sample[sample_rank_key] = None


def _resolve_prompt_structural_heuristic_fields(
    key_signal_fields,
    *,
    key_signal_reference="sequential_global",
    for_reference_build=False,
):
    if key_signal_fields is None:
        return ()
    selected_fields = normalize_key_signal_fields(key_signal_fields)
    needed = []
    if "common_neighbor" in selected_fields:
        if for_reference_build or str(key_signal_reference).strip().lower() != "contextual":
            needed.append("resource_allocation")
    if "recent_degree" in selected_fields:
        needed.append("recent_degree")
    if "global_recency" in selected_fields:
        needed.append("global_recency")
    if "itemcf" in selected_fields:
        needed.append("itemcf")
    if "usercf" in selected_fields:
        needed.append("usercf")
    return tuple(needed)


def _resolve_directed_edge_arrays(edges_df, edge_u_vals, edge_i_vals, edge_ts_vals):
    if edge_u_vals is not None and edge_i_vals is not None and edge_ts_vals is not None:
        return (
            np.asarray(edge_u_vals, dtype=np.int64),
            np.asarray(edge_i_vals, dtype=np.int64),
            np.asarray(edge_ts_vals, dtype=np.float64),
        )
    if edges_df is None:
        return edge_u_vals, edge_i_vals, edge_ts_vals

    edges_df_sorted = edges_df.sort_values("ts")
    return (
        edges_df_sorted["u"].values.astype(np.int64, copy=False),
        edges_df_sorted["i"].values.astype(np.int64, copy=False),
        edges_df_sorted["ts"].values.astype(np.float64, copy=False),
    )


def _attach_standalone_heuristic_scores(samples, heuristic_scores):
    if not heuristic_scores:
        return samples
    for idx, sample in enumerate(samples):
        if "resource_allocation" in heuristic_scores:
            value = float(heuristic_scores["resource_allocation"][idx])
            sample["common_neighbor_score"] = value
            sample["heuristic_resource_allocation_score"] = value
        if "global_recency" in heuristic_scores:
            sample["heuristic_global_recency_score"] = float(heuristic_scores["global_recency"][idx])
        if "recent_degree" in heuristic_scores:
            sample["heuristic_recent_degree_score"] = float(heuristic_scores["recent_degree"][idx])
        if "itemcf" in heuristic_scores:
            sample["heuristic_itemcf_score"] = float(heuristic_scores["itemcf"][idx])
        if "usercf" in heuristic_scores:
            sample["heuristic_usercf_score"] = float(heuristic_scores["usercf"][idx])
        if "recency" in heuristic_scores:
            sample["heuristic_recency_score"] = float(heuristic_scores["recency"][idx])
        if "popularity" in heuristic_scores:
            sample["heuristic_popularity_score"] = float(heuristic_scores["popularity"][idx])
        if "past_interactions" in heuristic_scores:
            sample["heuristic_past_interactions_score"] = float(
                heuristic_scores["past_interactions"][idx]
            )
        if "semantic_smoothing" in heuristic_scores:
            sample["heuristic_semantic_smoothing_score"] = float(
                heuristic_scores["semantic_smoothing"][idx]
            )
    return samples


def _bucket_overall_structural_signal(score, low_threshold, high_threshold):
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "Unknown"
    if value < float(low_threshold):
        return "Low"
    if value >= float(high_threshold):
        return "High"
    return "Modest"


def assign_expert_prediction_labels(
    samples,
    negative_ratio,
    expert_prediction_mode,
    expert_prediction_fixed_threshold=0.05,
    score_field="rrf_score",
    score_label="RRF",
):
    if not samples:
        return
    expert_prediction_mode = str(expert_prediction_mode).strip().lower()
    score_field = str(score_field).strip() or "rrf_score"
    score_label = str(score_label).strip() or score_field
    if expert_prediction_mode not in {
        "global_median",
        "sequential_running_median",
        "fixed_threshold",
    }:
        raise ValueError(
            "Unsupported expert_prediction_mode="
            f"{expert_prediction_mode}. Use global_median, sequential_running_median, or fixed_threshold."
        )

    all_scores = [float(sample.get(score_field, 0.0)) for sample in samples]
    labels = np.array([sample["label"] for sample in samples], dtype=np.int64)
    scores = np.array(all_scores, dtype=np.float64)

    if expert_prediction_mode == "global_median":
        _rrf_log(
            f"Expert prediction mode [{score_label}]: "
            "global_median (global eval-batch threshold; leaky baseline)."
        )
        median_score = float(np.median(all_scores))
        _rrf_log(f"Median score [{score_label}] (global batch): {median_score:.4f}")
        for sample in samples:
            sample["expert_prediction"] = (
                "True" if float(sample.get(score_field, 0.0)) >= median_score else "False"
            )
        thresholded_preds = (scores >= median_score).astype(np.int64)
        thresholded_labels = labels
    elif expert_prediction_mode == "fixed_threshold":
        fixed_threshold = float(expert_prediction_fixed_threshold)
        _rrf_log(
            f"Expert prediction mode [{score_label}]: fixed_threshold "
            f"(global fixed boundary={fixed_threshold:.6f})."
        )
        for sample in samples:
            sample["expert_prediction"] = (
                "True" if float(sample.get(score_field, 0.0)) >= fixed_threshold else "False"
            )
        thresholded_preds = (scores >= fixed_threshold).astype(np.int64)
        thresholded_labels = labels
    else:
        _rrf_log(
            f"Expert prediction mode [{score_label}]: "
            "sequential_running_median (no future-query threshold leakage)."
        )
        query_to_indices = {}
        for idx, sample in enumerate(samples):
            qid = sample.get("query_id", idx)
            query_to_indices.setdefault(qid, []).append(idx)

        ordered_queries = sorted(
            [
                (qid, indices, samples[indices[0]]["timestamp"])
                for qid, indices in query_to_indices.items()
            ],
            key=lambda item: (item[2], int(item[0])),
        )

        running_scores = []
        warmup_unknown_queries = 0
        ptr = 0
        while ptr < len(ordered_queries):
            ts_anchor = ordered_queries[ptr][2]
            group = []
            while ptr < len(ordered_queries) and ordered_queries[ptr][2] == ts_anchor:
                group.append(ordered_queries[ptr])
                ptr += 1

            if not running_scores:
                warmup_unknown_queries += len(group)
                for _, indices, _ in group:
                    for idx in indices:
                        samples[idx]["expert_prediction"] = "Unknown"
            else:
                n_scores = len(running_scores)
                if n_scores % 2 == 1:
                    median_score = running_scores[n_scores // 2]
                else:
                    median_score = 0.5 * (
                        running_scores[n_scores // 2 - 1] + running_scores[n_scores // 2]
                    )
                for _, indices, _ in group:
                    for idx in indices:
                        samples[idx]["expert_prediction"] = (
                            "True"
                            if float(samples[idx].get(score_field, 0.0)) >= median_score
                            else "False"
                        )

            for _, indices, _ in group:
                for idx in indices:
                    insort(running_scores, float(samples[idx].get(score_field, 0.0)))

        _rrf_log(
            f"Sequential expert warmup queries [{score_label}] "
            f"(prediction=Unknown): {warmup_unknown_queries}"
        )

        thresholded_preds = []
        thresholded_labels = []
        for sample in samples:
            pred = sample.get("expert_prediction", "Unknown")
            if pred == "Unknown":
                continue
            thresholded_preds.append(1 if pred == "True" else 0)
            thresholded_labels.append(int(sample["label"]))
        thresholded_preds = np.asarray(thresholded_preds, dtype=np.int64)
        thresholded_labels = np.asarray(thresholded_labels, dtype=np.int64)

    try:
        score_ap, score_auc = compute_dtgb_ap_auc(scores, labels, negative_ratio)
        if score_auc is None:
            score_auc = roc_auc_score(labels, scores)
        if thresholded_labels.size > 0:
            score_acc = (thresholded_preds == thresholded_labels).mean()
        else:
            score_acc = None
        if score_ap is None:
            if score_acc is None:
                _rrf_log(f"Baseline [{score_label}]: AUC={score_auc:.4f}, Acc=unavailable")
            else:
                _rrf_log(
                    f"Baseline [{score_label}]: AUC={score_auc:.4f}, Acc={score_acc:.4f}"
                )
        else:
            if score_acc is None:
                _rrf_log(
                    f"Baseline [{score_label}] (DTGB 1v1): "
                    f"AP={score_ap:.4f}, AUC={score_auc:.4f}, Acc=unavailable"
                )
            else:
                _rrf_log(
                    f"Baseline [{score_label}] (DTGB 1v1): "
                    f"AP={score_ap:.4f}, AUC={score_auc:.4f}, Acc={score_acc:.4f}"
                )
    except Exception as exc:
        _rrf_warn(f"Baseline AUC/Acc unavailable [{score_label}] ({exc})")


def _assign_expert_prediction_labels(
    samples,
    negative_ratio,
    expert_prediction_mode,
    expert_prediction_fixed_threshold=0.05,
):
    assign_expert_prediction_labels(
        samples=samples,
        negative_ratio=negative_ratio,
        expert_prediction_mode=expert_prediction_mode,
        expert_prediction_fixed_threshold=expert_prediction_fixed_threshold,
        score_field="rrf_score",
        score_label="RRF",
    )


def apply_overall_structural_signal_buckets(
    samples,
    *,
    low_threshold,
    high_threshold,
    score_field="rrf_score",
):
    score_field = str(score_field).strip() or "rrf_score"
    for sample in samples:
        sample["overall_structural_signal"] = _bucket_overall_structural_signal(
            sample.get(score_field, 0.0),
            low_threshold,
            high_threshold,
        )


def finalize_test_samples(
    samples,
    *,
    edges_df,
    val_ratio=0.15,
    test_ratio=0.15,
    negative_ratio=1,
    random_seed=42,
    build_prompt_features=True,
    compute_expert_prediction=True,
    compute_rrf_scores=None,
    key_signal_fields=None,
    standalone_heuristic_fields=None,
    rrf_k=60,
    rrf_mode="query_local",
    sequential_rank_bins=1024,
    expert_prediction_mode="sequential_running_median",
    expert_prediction_fixed_threshold=0.05,
    rrf_pointwise_pool_size=256,
    rrf_pointwise_num_pools=4,
    rrf_batch_size=200000,
    rrf_heuristics=None,
    key_signal_reference="sequential_global",
    include_overall_structural_signal=False,
    overall_signal_low_threshold=0.0475,
    overall_signal_high_threshold=0.0510,
    history_as_source=None,
    history_as_target=None,
    edge_u_vals=None,
    edge_i_vals=None,
    edge_ts_vals=None,
    train_entity_ids=None,
    use_gpu_heuristics=True,
    apply_gdelt_time_bucket=False,
    skip_key_signal_calibration=False,
    embeddings=None,
    entity_id_to_idx=None,
    smooth_time_window=50.0,
    smooth_steps=1,
    smooth_decay_gamma=0.1,
    smooth_undirected=True,
    heuristic_recent_degree_window=30.0,
):
    if compute_rrf_scores is None:
        compute_rrf_scores = compute_expert_prediction

    build_prompt_features = bool(build_prompt_features)
    selected_rrf_heuristics = normalize_rrf_heuristics(rrf_heuristics)
    key_signal_reference = str(key_signal_reference).strip().lower()
    if key_signal_reference not in {"sequential_global", "contextual"}:
        raise ValueError(
            "Unsupported key_signal_reference="
            f"{key_signal_reference}. Use sequential_global or contextual."
        )
    print(
        "[FinalizeSamples] start: "
        f"samples={len(samples)}, build_prompt_features={build_prompt_features}, "
        f"compute_rrf_scores={bool(compute_rrf_scores)}, "
        f"compute_expert_prediction={bool(compute_expert_prediction)}, "
        f"key_signal_reference={key_signal_reference}, "
        f"skip_key_signal_calibration={bool(skip_key_signal_calibration)}",
        flush=True,
    )

    if standalone_heuristic_fields is None:
        standalone_heuristic_fields = _resolve_prompt_structural_heuristic_fields(
            key_signal_fields,
            key_signal_reference=key_signal_reference,
        )
    else:
        standalone_heuristic_fields = tuple(
            normalize_rrf_heuristics(standalone_heuristic_fields)
        ) if standalone_heuristic_fields else ()

    needs_directed_prompt_heuristics = (
        "itemcf" in standalone_heuristic_fields
        or "usercf" in standalone_heuristic_fields
    )
    if needs_directed_prompt_heuristics:
        edge_u_vals, edge_i_vals, edge_ts_vals = _resolve_directed_edge_arrays(
            edges_df,
            edge_u_vals,
            edge_i_vals,
            edge_ts_vals,
        )

    need_neighbor_sampler = (
        bool(compute_rrf_scores)
        or bool(standalone_heuristic_fields)
        or (build_prompt_features and key_signal_reference == "contextual")
    )
    print(
        "[FinalizeSamples] resolved fields: "
        f"standalone_heuristics={','.join(standalone_heuristic_fields) if standalone_heuristic_fields else 'none'}, "
        f"need_neighbor_sampler={bool(need_neighbor_sampler)}",
        flush=True,
    )
    neighbor_sampler = None
    if need_neighbor_sampler:
        if compute_rrf_scores or standalone_heuristic_fields:
            _rrf_log("Initializing NeighborSampler for heuristic scoring...")
        else:
            print("Initializing NeighborSampler for contextual key-signal calibration...")
        full_data = Data(
            src_node_ids=edges_df["u"].values.astype(np.int64),
            dst_node_ids=edges_df["i"].values.astype(np.int64),
            node_interact_times=edges_df["ts"].values.astype(np.float64),
            edge_ids=np.arange(len(edges_df), dtype=np.int64),
            labels=np.zeros(len(edges_df), dtype=np.int64),
        )
        neighbor_sampler = get_neighbor_sampler(
            data=full_data,
            sample_neighbor_strategy="recent",
            time_scaling_factor=0.0,
            seed=1,
        )

    if compute_rrf_scores:
        _rrf_log(
            "Computing scores in mode="
            f"{rrf_mode} with heuristics={','.join(selected_rrf_heuristics)}..."
        )
        semantic_context = None
        if "semantic_smoothing" in selected_rrf_heuristics:
            semantic_context = _prepare_rrf_semantic_context(
                edges_df=edges_df,
                embeddings=embeddings,
                entity_id_to_idx=entity_id_to_idx,
                smooth_time_window=smooth_time_window,
                smooth_steps=smooth_steps,
                smooth_decay_gamma=smooth_decay_gamma,
                smooth_undirected=smooth_undirected,
            )
        if rrf_mode == "train_pool_pointwise":
            if train_entity_ids is None:
                cutoff_ts = edges_df["ts"].values.astype(np.float64)
                if apply_gdelt_time_bucket:
                    cutoff_ts = np.floor_divide(cutoff_ts.astype(np.int64), 15).astype(np.float64)
                val_time = np.quantile(cutoff_ts, 1 - val_ratio - test_ratio)
                edges_df_sorted = edges_df.sort_values("ts")
                u_vals = edges_df_sorted["u"].values
                i_vals = edges_df_sorted["i"].values
                ts_vals = edges_df_sorted["ts"].values
                split_ts_vals = ts_vals
                if apply_gdelt_time_bucket:
                    split_ts_vals = np.floor_divide(ts_vals.astype(np.int64), 15).astype(np.float64)
                train_node_ids = set()
                for u, i, split_ts in zip(u_vals, i_vals, split_ts_vals):
                    split_ts = float(split_ts)
                    if split_ts >= float(val_time):
                        break
                    train_node_ids.add(int(u))
                    train_node_ids.add(int(i))
                train_entity_ids = [
                    int(node_id) for node_id in train_node_ids if int(node_id) != 0
                ]
            rrf_scores, ra_scores, heuristic_components = compute_train_pool_rrf_scores(
                samples=samples,
                neighbor_sampler=neighbor_sampler,
                build_config=TrainPoolBuildConfig(
                    num_pools=rrf_pointwise_num_pools,
                    pool_size=rrf_pointwise_pool_size,
                    random_seed=random_seed,
                ),
                score_config=TrainPoolRRFConfig(
                    rrf_k=rrf_k,
                    score_batch_size=rrf_batch_size,
                    use_gpu_heuristics=use_gpu_heuristics,
                    rrf_heuristics=selected_rrf_heuristics,
                    semantic_context=semantic_context,
                ),
                train_entity_ids=train_entity_ids,
            )
        else:
            rrf_scores, ra_scores, heuristic_components = compute_rrf_scores_for_samples(
                samples=samples,
                neighbor_sampler=neighbor_sampler,
                rrf_k=rrf_k,
                use_gpu_heuristics=use_gpu_heuristics,
                score_batch_size=rrf_batch_size,
                recent_degree_window=heuristic_recent_degree_window,
                return_components=True,
                rrf_mode=rrf_mode,
                sequential_rank_bins=sequential_rank_bins,
                rrf_heuristics=selected_rrf_heuristics,
                semantic_context=semantic_context,
            )
        _attach_rrf_scores(samples, rrf_scores, ra_scores, heuristic_components)
        _rrf_log(f"Scored {len(samples)} samples.")
    else:
        _rrf_log("Skipping score computation.")
        if standalone_heuristic_fields:
            _rrf_log(
                "Computing standalone prompt heuristics: "
                f"{','.join(standalone_heuristic_fields)}"
            )
            heuristic_scores = compute_selected_heuristic_scores_for_samples(
                samples=samples,
                neighbor_sampler=neighbor_sampler,
                heuristics=standalone_heuristic_fields,
                recent_degree_window=heuristic_recent_degree_window,
                use_gpu_heuristics=use_gpu_heuristics,
                score_batch_size=rrf_batch_size,
                show_progress=True,
                progress_desc="Prompt heuristic scoring",
                directed_src_node_ids=edge_u_vals,
                directed_dst_node_ids=edge_i_vals,
                directed_node_interact_times=edge_ts_vals,
            )
            _attach_standalone_heuristic_scores(samples, heuristic_scores)

    if build_prompt_features and (not skip_key_signal_calibration):
        calibrate_prompt_key_signals(
            samples=samples,
            key_signal_reference=key_signal_reference,
            edges_df=edges_df,
            history_as_source=history_as_source,
            history_as_target=history_as_target,
            edge_u_vals=edge_u_vals,
            edge_i_vals=edge_i_vals,
            edge_ts_vals=edge_ts_vals,
            neighbor_sampler=neighbor_sampler,
            use_gpu_heuristics=use_gpu_heuristics,
            heuristic_recent_degree_window=heuristic_recent_degree_window,
        )
    elif build_prompt_features:
        print("Skipping key-signal calibration (external reference will be applied later).")
    else:
        print("Skipping key-signal calibration (prompt-context construction disabled).")

    if compute_expert_prediction and compute_rrf_scores:
        _assign_expert_prediction_labels(
            samples=samples,
            negative_ratio=negative_ratio,
            expert_prediction_mode=expert_prediction_mode,
            expert_prediction_fixed_threshold=expert_prediction_fixed_threshold,
        )
    else:
        for sample in samples:
            sample["expert_prediction"] = "Unknown"

    if include_overall_structural_signal:
        apply_overall_structural_signal_buckets(
            samples,
            low_threshold=float(overall_signal_low_threshold),
            high_threshold=float(overall_signal_high_threshold),
            score_field="rrf_score",
        )

    return samples


__all__ = [
    "assign_expert_prediction_labels",
    "apply_overall_structural_signal_buckets",
    "finalize_test_samples",
]
