"""
Budgeted hybrid evaluation: RRF on full set, LLM on a selected sample slice.
"""
import numpy as np

from .analysis import (
    build_dtgb_eval_batches,
    compute_rrf_selected_slice_proxy,
    minmax_scale_scores,
)
from ..prediction_metrics import compute_prediction_metrics


def prepare_backbone_score_space(raw_scores, score_space="minmax"):
    values = np.asarray(raw_scores, dtype=np.float64)
    mode = str(score_space).strip().lower()
    if mode == "minmax":
        return minmax_scale_scores(values)
    if mode == "raw":
        return values.copy()
    raise ValueError(f"Unknown backbone score space: {score_space}")


def fit_quantile_score_alignment(source_scores, target_scores):
    source = np.asarray(source_scores, dtype=np.float64)
    target = np.asarray(target_scores, dtype=np.float64)
    if source.size == 0 or target.size == 0:
        raise ValueError("Quantile score alignment requires non-empty source and target scores.")

    return {
        "mode": "quantile_match",
        "num_source": int(source.size),
        "num_target": int(target.size),
        "source_sorted": np.sort(source).astype(np.float64),
        "target_sorted": np.sort(target).astype(np.float64),
    }


def _fit_monotone_isotonic_values(x_values, y_values):
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        raise ValueError("Isotonic alignment requires equal-length non-empty x/y arrays.")

    order = np.argsort(x, kind="mergesort")
    x_sorted = x[order]
    y_sorted = y[order]

    unique_x = []
    mean_y = []
    counts = []
    for value_x, value_y in zip(x_sorted, y_sorted):
        if unique_x and value_x == unique_x[-1]:
            total_weight = counts[-1] + 1
            mean_y[-1] = (mean_y[-1] * counts[-1] + value_y) / float(total_weight)
            counts[-1] = total_weight
        else:
            unique_x.append(float(value_x))
            mean_y.append(float(value_y))
            counts.append(1)

    block_values = []
    block_weights = []
    block_sizes = []
    for target, weight in zip(mean_y, counts):
        block_values.append(float(target))
        block_weights.append(float(weight))
        block_sizes.append(1)
        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            merged_weight = block_weights[-2] + block_weights[-1]
            merged_value = (
                block_values[-2] * block_weights[-2]
                + block_values[-1] * block_weights[-1]
            ) / float(merged_weight)
            merged_size = block_sizes[-2] + block_sizes[-1]
            block_values[-2:] = [float(merged_value)]
            block_weights[-2:] = [float(merged_weight)]
            block_sizes[-2:] = [int(merged_size)]

    fitted_y = np.empty(len(unique_x), dtype=np.float64)
    start = 0
    for block_value, block_size in zip(block_values, block_sizes):
        end = start + int(block_size)
        fitted_y[start:end] = float(block_value)
        start = end

    return np.asarray(unique_x, dtype=np.float64), fitted_y


def fit_isotonic_score_alignment(source_scores, target_scores):
    source = np.asarray(source_scores, dtype=np.float64)
    target = np.asarray(target_scores, dtype=np.float64)
    if source.size == 0 or target.size == 0:
        raise ValueError("Isotonic score alignment requires non-empty source and target scores.")
    if source.size != target.size:
        raise ValueError("Isotonic score alignment requires source and target scores of equal length.")

    unique_x, fitted_y = _fit_monotone_isotonic_values(source, target)
    return {
        "mode": "isotonic_regression",
        "num_source": int(source.size),
        "num_target": int(target.size),
        "source_unique": unique_x,
        "target_fitted": fitted_y,
        "target_min": float(np.min(target)),
        "target_max": float(np.max(target)),
    }


def apply_quantile_score_alignment(scores, alignment):
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return values.copy()

    source_sorted = np.asarray(alignment["source_sorted"], dtype=np.float64)
    target_sorted = np.asarray(alignment["target_sorted"], dtype=np.float64)
    if source_sorted.size == 0 or target_sorted.size == 0:
        raise ValueError("Quantile score alignment received an empty calibration array.")
    if source_sorted.size == 1 or np.all(source_sorted == source_sorted[0]):
        return np.full_like(values, float(np.median(target_sorted)), dtype=np.float64)

    lo = np.searchsorted(source_sorted, values, side="left")
    hi = np.searchsorted(source_sorted, values, side="right")
    avg_rank = 0.5 * (lo + hi - 1)
    quantiles = avg_rank / float(max(source_sorted.size - 1, 1))
    quantiles = np.clip(quantiles, 0.0, 1.0)
    return np.quantile(target_sorted, quantiles)


def apply_isotonic_score_alignment(scores, alignment):
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return values.copy()

    source_unique = np.asarray(alignment["source_unique"], dtype=np.float64)
    target_fitted = np.asarray(alignment["target_fitted"], dtype=np.float64)
    if source_unique.size == 0 or target_fitted.size == 0:
        raise ValueError("Isotonic score alignment received an empty calibration array.")
    if source_unique.size != target_fitted.size:
        raise ValueError("Isotonic score alignment calibration arrays must have equal length.")
    if source_unique.size == 1:
        return np.full_like(values, float(target_fitted[0]), dtype=np.float64)

    return np.interp(
        values,
        source_unique,
        target_fitted,
        left=float(target_fitted[0]),
        right=float(target_fitted[-1]),
    )


def apply_score_alignment(scores, alignment):
    if alignment is None:
        return np.asarray(scores, dtype=np.float64).copy()

    mode = str(alignment.get("mode", "")).strip().lower()
    if mode in {"quantile_match", "validation_selected_quantile_match"}:
        return apply_quantile_score_alignment(scores, alignment)
    if mode in {"isotonic_regression", "validation_selected_isotonic_regression"}:
        return apply_isotonic_score_alignment(scores, alignment)
    raise ValueError(f"Unknown score alignment mode: {alignment.get('mode')}")


def _metric_delta(lhs, rhs):
    return {
        "ap": float(lhs["ap"] - rhs["ap"]),
        "auc": float(lhs["auc"] - rhs["auc"]),
        "accuracy": float(lhs["accuracy"] - rhs["accuracy"]),
        "ap_global": float(lhs["ap_global"] - rhs["ap_global"]),
        "auc_global": float(lhs["auc_global"] - rhs["auc_global"]),
    }


def _ks_distance_1d(values_a, values_b):
    arr_a = np.sort(np.asarray(values_a, dtype=np.float64))
    arr_b = np.sort(np.asarray(values_b, dtype=np.float64))
    if arr_a.size == 0 or arr_b.size == 0:
        return None
    grid = np.sort(np.concatenate([arr_a, arr_b]))
    cdf_a = np.searchsorted(arr_a, grid, side="right") / float(arr_a.size)
    cdf_b = np.searchsorted(arr_b, grid, side="right") / float(arr_b.size)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _wasserstein_distance_1d(values_a, values_b, num_quantiles=257):
    arr_a = np.asarray(values_a, dtype=np.float64)
    arr_b = np.asarray(values_b, dtype=np.float64)
    if arr_a.size == 0 or arr_b.size == 0:
        return None
    q = np.linspace(0.0, 1.0, int(max(3, num_quantiles)))
    qa = np.quantile(arr_a, q)
    qb = np.quantile(arr_b, q)
    return float(np.mean(np.abs(qa - qb)))


def _rank_graft_scores(rrf_selected, llm_selected):
    """
    Replace selected scores by reordering the selected RRF values with LLM rank.
    This preserves the selected score distribution while injecting LLM ordering.
    """
    llm_rank_desc = np.argsort(-llm_selected, kind="mergesort")
    rrf_values_desc = np.sort(rrf_selected)[::-1]
    grafted = np.empty_like(rrf_selected)
    grafted[llm_rank_desc] = rrf_values_desc
    return grafted


def _rank_graft_scores_per_dtgb_batch(
    samples,
    selected_indices,
    rrf_selected,
    llm_selected,
    dtgb_eval_batch_size,
):
    """Apply rank graft independently inside each DTGB evaluation batch."""
    grafted = np.empty_like(rrf_selected)
    selected_pos_by_sample_idx = {
        int(sample_idx): int(local_pos) for local_pos, sample_idx in enumerate(selected_indices)
    }

    for batch in build_dtgb_eval_batches(samples, dtgb_eval_batch_size):
        local_positions = [
            selected_pos_by_sample_idx[idx]
            for idx in batch["sample_indices"]
            if idx in selected_pos_by_sample_idx
        ]
        if not local_positions:
            continue
        local_idx = np.asarray(local_positions, dtype=np.int64)
        grafted[local_idx] = _rank_graft_scores(
            rrf_selected=rrf_selected[local_idx],
            llm_selected=llm_selected[local_idx],
        )

    return grafted


def _merge_selected_scores(
    samples,
    selected_indices,
    rrf_selected,
    llm_selected,
    merge_method,
    fusion_alpha,
    dtgb_eval_batch_size,
    selection_mode=None,
):
    method = str(merge_method).strip().lower()
    if method in {"rank_graft", "graft"}:
        selection_mode_key = str(selection_mode or "").strip().lower()
        if selection_mode_key in {
            "pointwise_fixed_threshold_band",
            "validation_sampled_uncertainty_band",
        }:
            return _rank_graft_scores(
                rrf_selected=rrf_selected,
                llm_selected=llm_selected,
            )
        return _rank_graft_scores_per_dtgb_batch(
            samples=samples,
            selected_indices=selected_indices,
            rrf_selected=rrf_selected,
            llm_selected=llm_selected,
            dtgb_eval_batch_size=dtgb_eval_batch_size,
        )

    if method in {"alpha", "raw_alpha"}:
        alpha = float(fusion_alpha)
        alpha = min(1.0, max(0.0, alpha))
        return alpha * llm_selected + (1.0 - alpha) * rrf_selected

    if method in {"llm_only", "replace"}:
        return llm_selected.copy()

    raise ValueError(
        f"Unknown hybrid merge method: {merge_method}. "
        "Use one of: rank_graft, alpha, llm_only."
    )


def _build_selection_meta(samples, selected_indices, dtgb_eval_batch_size):
    selected_index_set = {int(idx) for idx in selected_indices}
    selected_batch_ids = []
    for batch in build_dtgb_eval_batches(samples, dtgb_eval_batch_size):
        if any(idx in selected_index_set for idx in batch["sample_indices"]):
            selected_batch_ids.append(int(batch["batch_id"]))

    return {
        "enabled": True,
        "selection_mode": "explicit_sample_selection",
        "selected_count": int(len(selected_indices)),
        "selected_batches": int(len(selected_batch_ids)),
        "available_batches": int(len(build_dtgb_eval_batches(samples, dtgb_eval_batch_size))),
        "dtgb_eval_batch_size": int(dtgb_eval_batch_size),
        "selected_batch_ids": selected_batch_ids,
        "selected_sample_indices": [int(idx) for idx in selected_indices],
    }


def evaluate_budgeted_hybrid_rrf_llm(
    samples,
    run_llm_on_samples,
    selected_sample_indices=None,
    selection_meta=None,
    score_alignment=None,
    fusion_alpha=1.0,
    merge_method="rank_graft",
    dtgb_eval_batch_size=None,
    backbone_score_field="rrf_score",
    backbone_name="RRF",
    backbone_score_space="minmax",
    full_score_fusion=None,
):
    """
    Evaluate budgeted hybrid predictions on the full sample set.

    Args:
        samples: full evaluation samples containing precomputed `rrf_score`.
        selected_sample_indices: explicit sample-level selection for LLM reranking.
        run_llm_on_samples: callable(subset_samples) -> eval dict with `predictions`.
        selection_meta: optional metadata describing how the selected slice was built.
        fusion_alpha: alpha in alpha*LLM + (1-alpha)*RRF for selected samples.
        merge_method: selected-score merge method. One of:
            - rank_graft: keep selected RRF value set, reorder by LLM rank
              independently inside each DTGB evaluation batch
            - alpha/raw_alpha: alpha*LLM + (1-alpha)*RRF
            - llm_only: direct replacement by LLM score
    """
    if dtgb_eval_batch_size is None:
        raise ValueError(
            "dtgb_eval_batch_size is required for hybrid full-set metric aggregation."
        )

    selected_indices = [int(idx) for idx in (selected_sample_indices or [])]

    if not selected_indices:
        raise RuntimeError("No samples selected for hybrid LLM evaluation.")

    selected_samples = [samples[idx] for idx in selected_indices]
    selected_results = run_llm_on_samples(selected_samples)

    llm_selected_scores_raw = np.array(selected_results["predictions"], dtype=np.float64)
    if llm_selected_scores_raw.shape[0] != len(selected_indices):
        raise RuntimeError(
            "LLM prediction count does not match selected sample count "
            f"({llm_selected_scores_raw.shape[0]} vs {len(selected_indices)})."
        )

    backbone_scores_raw = np.array(
        [sample.get(backbone_score_field, 0.0) for sample in samples],
        dtype=np.float64,
    )
    backbone_scores = prepare_backbone_score_space(
        backbone_scores_raw,
        score_space=backbone_score_space,
    )
    full_score_fusion_metadata = None
    if full_score_fusion is not None:
        fusion_result = full_score_fusion(llm_selected_scores_raw)
        if not isinstance(fusion_result, dict) or "scores" not in fusion_result:
            raise RuntimeError(
                "full_score_fusion must return a mapping containing full 'scores'"
            )
        hybrid_scores = np.asarray(fusion_result["scores"], dtype=np.float64)
        if hybrid_scores.shape != backbone_scores.shape or not np.all(
            np.isfinite(hybrid_scores)
        ):
            raise RuntimeError(
                "full_score_fusion returned invalid full-split predictions"
            )
        llm_selected_scores = hybrid_scores[selected_indices]
        full_score_fusion_metadata = fusion_result.get("metadata")
        effective_merge_method = "tabicl_full_score_fusion"
    else:
        llm_selected_scores = apply_score_alignment(
            llm_selected_scores_raw, score_alignment
        )
        hybrid_scores = backbone_scores.copy()
        selection_mode = None
        if isinstance(selection_meta, dict):
            selection_mode = selection_meta.get("selection_mode")
        hybrid_scores[selected_indices] = _merge_selected_scores(
            samples=samples,
            selected_indices=selected_indices,
            rrf_selected=backbone_scores[selected_indices],
            llm_selected=llm_selected_scores,
            merge_method=merge_method,
            fusion_alpha=fusion_alpha,
            dtgb_eval_batch_size=dtgb_eval_batch_size,
            selection_mode=selection_mode,
        )
        effective_merge_method = str(merge_method)

    labels = np.array([sample["label"] for sample in samples], dtype=np.int64)
    hybrid_metrics = compute_prediction_metrics(
        hybrid_scores,
        labels,
        dtgb_eval_batch_size=dtgb_eval_batch_size,
    )
    backbone_metrics = compute_prediction_metrics(
        backbone_scores,
        labels,
        dtgb_eval_batch_size=dtgb_eval_batch_size,
    )

    selected_labels = labels[selected_indices]
    selected_llm_raw_metrics = compute_prediction_metrics(
        llm_selected_scores_raw,
        selected_labels,
        dtgb_eval_batch_size=None,
    )
    selected_llm_metrics = compute_prediction_metrics(
        llm_selected_scores,
        selected_labels,
        dtgb_eval_batch_size=None,
    )
    selected_backbone_metrics = compute_prediction_metrics(
        backbone_scores[selected_indices],
        selected_labels,
        dtgb_eval_batch_size=None,
    )
    selected_hybrid_metrics = compute_prediction_metrics(
        hybrid_scores[selected_indices],
        selected_labels,
        dtgb_eval_batch_size=None,
    )
    routing_debug = {
        "num_selected_samples": int(len(selected_indices)),
        "selected_fraction_realized": (
            float(len(selected_indices)) / float(len(samples)) if samples else 0.0
        ),
        "backbone_name": str(backbone_name),
        "backbone": {
            "ap": selected_backbone_metrics["ap"],
            "auc": selected_backbone_metrics["auc"],
            "accuracy": selected_backbone_metrics["accuracy"],
            "ap_global": selected_backbone_metrics["ap_global"],
            "auc_global": selected_backbone_metrics["auc_global"],
        },
        "llm_raw": {
            "ap": selected_llm_raw_metrics["ap"],
            "auc": selected_llm_raw_metrics["auc"],
            "accuracy": selected_llm_raw_metrics["accuracy"],
            "ap_global": selected_llm_raw_metrics["ap_global"],
            "auc_global": selected_llm_raw_metrics["auc_global"],
        },
        "delta_llm_raw_minus_backbone": _metric_delta(
            selected_llm_raw_metrics,
            selected_backbone_metrics,
        ),
    }
    alignment_debug = {
        "num_selected_samples": int(len(selected_indices)),
        "backbone_name": str(backbone_name),
        "backbone": {
            "ap": selected_backbone_metrics["ap"],
            "auc": selected_backbone_metrics["auc"],
            "accuracy": selected_backbone_metrics["accuracy"],
            "ap_global": selected_backbone_metrics["ap_global"],
            "auc_global": selected_backbone_metrics["auc_global"],
        },
        "llm_raw": {
            "ap": selected_llm_raw_metrics["ap"],
            "auc": selected_llm_raw_metrics["auc"],
            "accuracy": selected_llm_raw_metrics["accuracy"],
            "ap_global": selected_llm_raw_metrics["ap_global"],
            "auc_global": selected_llm_raw_metrics["auc_global"],
        },
        "llm_aligned": {
            "ap": selected_llm_metrics["ap"],
            "auc": selected_llm_metrics["auc"],
            "accuracy": selected_llm_metrics["accuracy"],
            "ap_global": selected_llm_metrics["ap_global"],
            "auc_global": selected_llm_metrics["auc_global"],
        },
        "delta_llm_aligned_minus_llm_raw": _metric_delta(
            selected_llm_metrics,
            selected_llm_raw_metrics,
        ),
        "distribution_distance_to_backbone": {
            "llm_raw": {
                "ks": _ks_distance_1d(
                    llm_selected_scores_raw,
                    backbone_scores[selected_indices],
                ),
                "wasserstein": _wasserstein_distance_1d(
                    llm_selected_scores_raw,
                    backbone_scores[selected_indices],
                ),
            },
            "llm_aligned": {
                "ks": _ks_distance_1d(
                    llm_selected_scores,
                    backbone_scores[selected_indices],
                ),
                "wasserstein": _wasserstein_distance_1d(
                    llm_selected_scores,
                    backbone_scores[selected_indices],
                ),
            },
        },
    }
    if str(backbone_name).strip().lower() == "rrf":
        routing_debug["rrf"] = dict(routing_debug["backbone"])
        routing_debug["delta_llm_raw_minus_rrf"] = dict(
            routing_debug["delta_llm_raw_minus_backbone"]
        )
        alignment_debug["rrf"] = dict(alignment_debug["backbone"])
        alignment_debug["distribution_distance_to_rrf"] = dict(
            alignment_debug["distribution_distance_to_backbone"]
        )

    selection_meta = selection_meta or _build_selection_meta(
        samples=samples,
        selected_indices=selected_indices,
        dtgb_eval_batch_size=dtgb_eval_batch_size,
    )
    proxy = compute_rrf_selected_slice_proxy(
        samples=selected_samples,
        llm_predictions=llm_selected_scores,
        selection_meta=selection_meta,
        score_field=backbone_score_field,
        backbone_name=backbone_name,
    )

    hybrid_metrics["detailed_results"] = selected_results.get("detailed_results", [])
    if isinstance(selected_results.get("token_usage"), dict):
        hybrid_metrics["token_usage"] = dict(selected_results["token_usage"])
    if isinstance(selected_results.get("prompt_embedding_shards"), list):
        hybrid_metrics["prompt_embedding_shards"] = list(
            selected_results["prompt_embedding_shards"]
        )
    hybrid_metrics["hybrid"] = {
        "enabled": True,
        "mode": "budgeted_backbone_llm",
        "backbone_name": str(backbone_name),
        "backbone_score_field": str(backbone_score_field),
        "backbone_score_space": str(backbone_score_space),
        "selection_mode": selection_meta.get("selection_mode", "explicit_sample_selection"),
        "merge_method": effective_merge_method,
        "fusion_alpha": float(fusion_alpha),
        "num_selected_batches": int(selection_meta.get("selected_batches", 0)),
        "num_selected_samples": int(len(selected_indices)),
        "selected_batch_ids": [
            int(batch_id) for batch_id in selection_meta.get("selected_batch_ids", [])
        ],
        "selected_sample_indices": [int(idx) for idx in selected_indices],
        "backbone_baseline_fullset": {
            "ap": backbone_metrics["ap"],
            "auc": backbone_metrics["auc"],
            "accuracy": backbone_metrics["accuracy"],
            "ap_global": backbone_metrics["ap_global"],
            "auc_global": backbone_metrics["auc_global"],
        },
        "uplift_vs_backbone": {
            "ap": hybrid_metrics["ap"] - backbone_metrics["ap"],
            "auc": hybrid_metrics["auc"] - backbone_metrics["auc"],
            "accuracy": hybrid_metrics["accuracy"] - backbone_metrics["accuracy"],
            "ap_global": hybrid_metrics["ap_global"] - backbone_metrics["ap_global"],
            "auc_global": hybrid_metrics["auc_global"] - backbone_metrics["auc_global"],
        },
        "selection": selection_meta,
        "score_alignment": (
            dict(full_score_fusion_metadata)
            if isinstance(full_score_fusion_metadata, dict)
            else (
                {
                    "mode": str(score_alignment.get("mode", "unknown")),
                    "num_source": int(score_alignment.get("num_source", 0)),
                    "num_target": int(score_alignment.get("num_target", 0)),
                    "target_name": score_alignment.get("target_name"),
                }
                if isinstance(score_alignment, dict)
                else None
            )
        ),
        "llm_selected_only_metrics": {
            "ap": selected_llm_metrics["ap"],
            "auc": selected_llm_metrics["auc"],
            "accuracy": selected_llm_metrics["accuracy"],
            "ap_global": selected_llm_metrics["ap_global"],
            "auc_global": selected_llm_metrics["auc_global"],
        },
        "selected_slice_metrics": {
            "backbone": {
                "ap": selected_backbone_metrics["ap"],
                "auc": selected_backbone_metrics["auc"],
                "accuracy": selected_backbone_metrics["accuracy"],
                "ap_global": selected_backbone_metrics["ap_global"],
                "auc_global": selected_backbone_metrics["auc_global"],
            },
            "llm_raw": {
                "ap": selected_llm_raw_metrics["ap"],
                "auc": selected_llm_raw_metrics["auc"],
                "accuracy": selected_llm_raw_metrics["accuracy"],
                "ap_global": selected_llm_raw_metrics["ap_global"],
                "auc_global": selected_llm_raw_metrics["auc_global"],
            },
            "llm_aligned": {
                "ap": selected_llm_metrics["ap"],
                "auc": selected_llm_metrics["auc"],
                "accuracy": selected_llm_metrics["accuracy"],
                "ap_global": selected_llm_metrics["ap_global"],
                "auc_global": selected_llm_metrics["auc_global"],
            },
            "hybrid_after_merge": {
                "ap": selected_hybrid_metrics["ap"],
                "auc": selected_hybrid_metrics["auc"],
                "accuracy": selected_hybrid_metrics["accuracy"],
                "ap_global": selected_hybrid_metrics["ap_global"],
                "auc_global": selected_hybrid_metrics["auc_global"],
            },
            "hybrid_uplift_vs_backbone": {
                "ap": selected_hybrid_metrics["ap"] - selected_backbone_metrics["ap"],
                "auc": selected_hybrid_metrics["auc"] - selected_backbone_metrics["auc"],
                "accuracy": selected_hybrid_metrics["accuracy"] - selected_backbone_metrics["accuracy"],
                "ap_global": selected_hybrid_metrics["ap_global"] - selected_backbone_metrics["ap_global"],
                "auc_global": selected_hybrid_metrics["auc_global"] - selected_backbone_metrics["auc_global"],
            },
        },
    }
    if str(backbone_name).strip().lower() == "rrf":
        hybrid_metrics["hybrid"]["mode"] = "budgeted_rrf_llm"
        hybrid_metrics["hybrid"]["rrf_baseline_fullset"] = dict(
            hybrid_metrics["hybrid"]["backbone_baseline_fullset"]
        )
        hybrid_metrics["hybrid"]["uplift_vs_rrf"] = dict(
            hybrid_metrics["hybrid"]["uplift_vs_backbone"]
        )
        hybrid_metrics["hybrid"]["selected_slice_metrics"]["rrf"] = dict(
            hybrid_metrics["hybrid"]["selected_slice_metrics"]["backbone"]
        )
        hybrid_metrics["hybrid"]["selected_slice_metrics"]["hybrid_uplift_vs_rrf"] = dict(
            hybrid_metrics["hybrid"]["selected_slice_metrics"]["hybrid_uplift_vs_backbone"]
        )
    if full_score_fusion is not None:
        hybrid_metrics["hybrid"]["mode"] = (
            "budgeted_tabicl_router_tabicl_full_score_fusion"
        )
    if "token_usage" in hybrid_metrics:
        hybrid_metrics["hybrid"]["llm_selected_token_usage"] = dict(hybrid_metrics["token_usage"])
    hybrid_metrics["rrf_uncertainty_proxy"] = proxy
    hybrid_metrics["hybrid_routing_debug"] = routing_debug
    hybrid_metrics["hybrid_alignment_debug"] = alignment_debug
    hybrid_metrics["selected_llm_predictions"] = llm_selected_scores_raw.tolist()
    hybrid_metrics["selected_llm_predictions_aligned"] = llm_selected_scores.tolist()

    return hybrid_metrics
