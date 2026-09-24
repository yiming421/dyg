"""
RRF downstream analysis helpers: DTGB-batch selection and score comparisons.
"""
import numpy as np

from .common import group_sample_indices_by_query
from ..prediction_metrics import compute_prediction_metrics


def fit_two_component_gaussian_mixture_1d(
    scores,
    *,
    max_iter=100,
    tol=1e-6,
    variance_floor=1e-6,
):
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        raise ValueError("GMM fitting requires at least one score.")

    if values.size == 1 or np.all(values == values[0]):
        mean = float(values[0])
        var = float(max(np.var(values), variance_floor))
        return {
            "weights": np.asarray([0.5, 0.5], dtype=np.float64),
            "means": np.asarray([mean, mean], dtype=np.float64),
            "variances": np.asarray([var, var], dtype=np.float64),
            "log_likelihood": None,
            "num_iterations": 0,
        }

    q25, q75 = np.quantile(values, [0.25, 0.75])
    means = np.asarray([float(q25), float(q75)], dtype=np.float64)
    overall_var = float(max(np.var(values), variance_floor))
    variances = np.asarray([overall_var, overall_var], dtype=np.float64)
    weights = np.asarray([0.5, 0.5], dtype=np.float64)
    prev_log_likelihood = None

    for iteration in range(int(max_iter)):
        std = np.sqrt(np.maximum(variances, variance_floor))
        centered = values[:, None] - means[None, :]
        gaussian = np.exp(-0.5 * (centered / std[None, :]) ** 2) / (
            np.sqrt(2.0 * np.pi) * std[None, :]
        )
        weighted = gaussian * weights[None, :]
        normalizer = np.maximum(weighted.sum(axis=1, keepdims=True), 1e-300)
        responsibilities = weighted / normalizer
        component_mass = np.maximum(responsibilities.sum(axis=0), 1e-12)

        weights = component_mass / float(values.size)
        means = (responsibilities * values[:, None]).sum(axis=0) / component_mass
        centered = values[:, None] - means[None, :]
        variances = (responsibilities * (centered ** 2)).sum(axis=0) / component_mass
        variances = np.maximum(variances, variance_floor)

        log_likelihood = float(np.log(normalizer[:, 0]).sum())
        if (
            prev_log_likelihood is not None
            and abs(log_likelihood - prev_log_likelihood) <= float(tol)
        ):
            break
        prev_log_likelihood = log_likelihood

    order = np.argsort(means, kind="mergesort")
    return {
        "weights": weights[order].astype(np.float64),
        "means": means[order].astype(np.float64),
        "variances": variances[order].astype(np.float64),
        "log_likelihood": prev_log_likelihood,
        "num_iterations": int(iteration + 1),
    }


def compute_two_component_gmm_overlap_scores(scores, gmm):
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return values.copy()

    weights = np.asarray(gmm["weights"], dtype=np.float64)
    means = np.asarray(gmm["means"], dtype=np.float64)
    variances = np.asarray(gmm["variances"], dtype=np.float64)
    if weights.size != 2 or means.size != 2 or variances.size != 2:
        raise ValueError("GMM overlap scoring expects exactly two components.")

    std = np.sqrt(np.maximum(variances, 1e-12))
    centered = values[:, None] - means[None, :]
    gaussian = np.exp(-0.5 * (centered / std[None, :]) ** 2) / (
        np.sqrt(2.0 * np.pi) * std[None, :]
    )
    weighted = gaussian * weights[None, :]
    normalizer = np.maximum(weighted.sum(axis=1, keepdims=True), 1e-300)
    responsibilities = weighted / normalizer
    return 1.0 - np.abs(responsibilities[:, 0] - responsibilities[:, 1])


def _select_topk_indices_by_score_desc(scores, top_k):
    values = np.asarray(scores, dtype=np.float64)
    if top_k is None or int(top_k) <= 0 or values.size == 0:
        return np.asarray([], dtype=np.int64)
    k = min(int(top_k), int(values.size))
    order = np.lexsort((np.arange(values.size), -values))
    return np.asarray(order[:k], dtype=np.int64)


def _centered_rank_window(num_items, top_k):
    """Return a centered contiguous rank window of size top_k."""
    k = min(int(top_k), int(num_items))
    if k <= 0:
        return 0, 0
    start = max(0, (int(num_items) - k) // 2)
    end = start + k
    return start, end


def build_dtgb_eval_batches(samples, dtgb_eval_batch_size):
    """
    Group consecutive query blocks into DTGB evaluation batches.

    `dtgb_eval_batch_size` is the number of positive queries per DTGB metric
    batch. The returned batches flatten those query blocks into the exact sample
    sequences that should be treated as one sortable rerank unit.
    """
    batch_size = int(dtgb_eval_batch_size)
    if batch_size < 1:
        raise ValueError("dtgb_eval_batch_size must be >= 1")

    query_to_indices = group_sample_indices_by_query(samples)
    query_groups = list(query_to_indices.items())
    batch_rows = []

    for batch_id, query_start in enumerate(range(0, len(query_groups), batch_size)):
        query_slice = query_groups[query_start: query_start + batch_size]
        batch_sample_indices = []
        batch_query_ids = []
        for query_id, indices in query_slice:
            batch_query_ids.append(int(query_id))
            batch_sample_indices.extend(int(idx) for idx in indices)

        batch_rows.append(
            {
                "batch_id": int(batch_id),
                "query_count": int(len(query_slice)),
                "query_ids": batch_query_ids,
                "sample_count": int(len(batch_sample_indices)),
                "sample_indices": batch_sample_indices,
            }
        )

    return batch_rows


def select_topk_rrf_middle_sample_indices(
    samples,
    top_k,
    dtgb_eval_batch_size,
    *,
    score_field="rrf_score",
):
    """
    Select top-k middle-ranked samples per DTGB evaluation batch by RRF order.

    Each DTGB batch is treated as one score sequence. Scores are sorted inside
    that batch, a centered contiguous rank window is selected, and the original
    sample indices are returned in original sample order.
    """
    if top_k is None or top_k <= 0:
        return [], {
            "enabled": False,
            "selection_mode": "per_dtgb_batch_middle_band",
            "requested_top_k": 0,
            "selected_count": 0,
            "selected_batches": 0,
            "available_batches": 0,
            "dtgb_eval_batch_size": int(dtgb_eval_batch_size),
            "selected_batch_ids": [],
            "selected_sample_indices": [],
        }

    batch_rows = build_dtgb_eval_batches(samples, dtgb_eval_batch_size)
    selected_sample_indices = []
    preview_rows = []

    for batch in batch_rows:
        batch_indices = batch["sample_indices"]
        if not batch_indices:
            continue

        batch_scores = np.array(
            [float(samples[idx].get(score_field, 0.0)) for idx in batch_indices],
            dtype=np.float64,
        )
        batch_order = np.argsort(batch_scores, kind="mergesort")
        start, end = _centered_rank_window(len(batch_indices), top_k)
        selected_positions = batch_order[start:end]
        batch_selected_indices = sorted(int(batch_indices[int(pos)]) for pos in selected_positions)

        selected_sample_indices.extend(batch_selected_indices)
        preview_rows.append(
            {
                "batch_id": int(batch["batch_id"]),
                "query_count": int(batch["query_count"]),
                "sample_count": int(batch["sample_count"]),
                "selected_count": int(len(batch_selected_indices)),
                "selected_sample_indices": batch_selected_indices,
                "selected_scores": [
                    float(samples[idx].get(score_field, 0.0))
                    for idx in batch_selected_indices
                ],
                "query_id_preview": batch["query_ids"][:10],
            }
        )

    selected_sample_indices.sort()
    return selected_sample_indices, {
        "enabled": True,
        "selection_mode": "per_dtgb_batch_middle_band",
        "requested_top_k": int(top_k),
        "selected_count": int(len(selected_sample_indices)),
        "selected_batches": int(len(preview_rows)),
        "available_batches": int(len(batch_rows)),
        "dtgb_eval_batch_size": int(dtgb_eval_batch_size),
        "selected_batch_ids": [int(row["batch_id"]) for row in preview_rows],
        "selected_sample_indices": [int(idx) for idx in selected_sample_indices],
        "batch_preview": preview_rows[:10],
    }


def select_rrf_middle_sample_indices_pointwise_threshold_band(
    samples,
    low_threshold,
    high_threshold,
    *,
    score_field="rrf_score",
):
    """
    Select samples whose raw RRF score lies inside a fixed middle threshold band.

    This is strict pointwise routing: each sample decision depends only on its own
    score and fixed thresholds, not on current test-batch composition.
    """
    low = float(low_threshold)
    high = float(high_threshold)
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError(
            "Pointwise threshold-band routing requires finite thresholds; "
            f"got low={low_threshold}, high={high_threshold}."
        )
    if low >= high:
        raise ValueError(
            "Pointwise threshold-band routing requires low < high; "
            f"got low={low:.6f}, high={high:.6f}."
        )

    selected_indices = []
    selected_scores = []
    for idx, sample in enumerate(samples):
        score = float(sample.get(score_field, 0.0))
        if low <= score < high:
            selected_indices.append(int(idx))
            selected_scores.append(score)

    return selected_indices, {
        "enabled": True,
        "selection_mode": "pointwise_fixed_threshold_band",
        "selection_rule": "route_if_low_le_score_lt_high",
        "low_threshold": low,
        "high_threshold": high,
        "selected_count": int(len(selected_indices)),
        "num_candidates": int(len(samples)),
        "selected_fraction_realized": (
            float(len(selected_indices)) / float(len(samples)) if samples else 0.0
        ),
        "selected_sample_indices": selected_indices,
        "selected_score_preview": [float(v) for v in selected_scores[:20]],
    }


def select_rrf_middle_sample_indices_validation_sampled_band(
    samples,
    calibration,
    *,
    score_field="rrf_score",
):
    selected_indices, meta = select_rrf_middle_sample_indices_pointwise_threshold_band(
        samples=samples,
        low_threshold=calibration["low_threshold"],
        high_threshold=calibration["high_threshold"],
        score_field=score_field,
    )
    enriched = dict(meta)
    enriched["selection_mode"] = "validation_sampled_uncertainty_band"
    enriched["selection_rule"] = "validation_frozen_middle_band_around_balanced_threshold"
    enriched["center_threshold"] = float(calibration["center_threshold"])
    enriched["validation_requested_target_fraction"] = float(
        calibration["requested_target_fraction"]
    )
    enriched["validation_requested_target_count"] = int(
        calibration["requested_target_count"]
    )
    enriched["validation_realized_fraction"] = float(
        calibration["selected_fraction_realized"]
    )
    enriched["validation_realized_count"] = int(
        calibration["selected_count_realized"]
    )
    return selected_indices, enriched


def select_rrf_middle_sample_indices_validation_sampled_gmm_overlap(
    samples,
    calibration,
    *,
    score_field="rrf_score",
):
    scores = np.asarray(
        [float(sample.get(score_field, 0.0)) for sample in samples],
        dtype=np.float64,
    )
    overlap_scores = compute_two_component_gmm_overlap_scores(
        scores,
        calibration["gmm"],
    )
    ambiguity_threshold = float(calibration["ambiguity_threshold"])
    selected_mask = overlap_scores >= ambiguity_threshold
    selected_indices = np.flatnonzero(selected_mask).astype(np.int64).tolist()

    enriched = {
        "enabled": True,
        "selection_mode": "validation_sampled_gmm_overlap_band",
        "selection_rule": "validation_frozen_two_component_gmm_overlap",
        "center_threshold": float(calibration["center_threshold"]),
        "ambiguity_threshold": ambiguity_threshold,
        "selected_count": int(len(selected_indices)),
        "num_candidates": int(len(samples)),
        "selected_fraction_realized": (
            float(len(selected_indices)) / float(len(samples)) if samples else 0.0
        ),
        "selected_sample_indices": [int(idx) for idx in selected_indices],
        "validation_requested_target_fraction": float(
            calibration["requested_target_fraction"]
        ),
        "validation_requested_target_count": int(
            calibration["requested_target_count"]
        ),
        "validation_realized_fraction": float(
            calibration["selected_fraction_realized"]
        ),
        "validation_realized_count": int(calibration["selected_count_realized"]),
        "gmm_weights": [float(v) for v in calibration["gmm"]["weights"]],
        "gmm_means": [float(v) for v in calibration["gmm"]["means"]],
        "gmm_variances": [float(v) for v in calibration["gmm"]["variances"]],
        "selected_overlap_preview": [
            float(overlap_scores[idx]) for idx in selected_indices[:20]
        ],
    }
    return selected_indices, enriched


def select_random_sample_indices(
    samples,
    target_fraction,
    *,
    random_seed,
):
    """
    Select a uniformly random sample-level slice at a fixed routing budget.

    This is intended as a naive routing ablation for validation-calibrated
    hybrid selectors. The target fraction uses the same round/max-one policy as
    validation-derived routing calibration.
    """
    num_candidates = int(len(samples))
    requested_fraction = float(max(0.0, min(1.0, target_fraction)))
    target_count = int(round(requested_fraction * float(num_candidates)))
    if requested_fraction > 0.0 and num_candidates > 0:
        target_count = max(1, target_count)
    target_count = min(num_candidates, target_count)

    if target_count <= 0:
        selected_indices = []
    else:
        rng = np.random.default_rng(int(random_seed))
        selected_indices = rng.choice(
            num_candidates,
            size=target_count,
            replace=False,
        ).astype(np.int64).tolist()
        selected_indices.sort()

    selected_labels = [
        int(samples[idx]["label"])
        for idx in selected_indices
        if "label" in samples[idx]
    ]
    selected_positive_rate = (
        float(np.mean(selected_labels)) if selected_labels else None
    )
    return selected_indices, {
        "enabled": True,
        "selection_mode": "random_sample",
        "selection_rule": "uniform_random_without_replacement",
        "random_seed": int(random_seed),
        "requested_target_fraction": requested_fraction,
        "requested_target_count": int(target_count),
        "selected_count": int(len(selected_indices)),
        "num_candidates": num_candidates,
        "selected_fraction_realized": (
            float(len(selected_indices)) / float(num_candidates)
            if num_candidates > 0
            else 0.0
        ),
        "selected_positive_rate": selected_positive_rate,
        "selected_negative_rate": (
            None if selected_positive_rate is None else float(1.0 - selected_positive_rate)
        ),
        "selected_sample_indices": [int(idx) for idx in selected_indices],
    }


def select_learned_router_indices(
    samples,
    target_fraction,
    *,
    router_checkpoint,
):
    """Select the global top predicted-utility fraction from a fitted router."""
    import joblib

    payload = joblib.load(router_checkpoint)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(
            f"Learned router checkpoint has no model payload: {router_checkpoint}"
        )
    feature_names = tuple(payload.get("feature_names", ()))
    if not feature_names:
        raise ValueError(
            f"Learned router checkpoint has no feature_names: {router_checkpoint}"
        )

    uncertainty_prior = payload.get("uncertainty_prior") or {}
    uncertainty_center = uncertainty_prior.get("center_threshold")

    features = np.full(
        (len(samples), len(feature_names)),
        np.nan,
        dtype=np.float64,
    )
    for row_idx, sample in enumerate(samples):
        for col_idx, name in enumerate(feature_names):
            if name == "semantic_abs_center_distance":
                raw_score = sample.get("semantic_mlp_score")
                value = (
                    abs(float(raw_score) - float(uncertainty_center))
                    if raw_score is not None and uncertainty_center is not None
                    else None
                )
            elif name == "last_interaction_missing":
                value = float(sample.get("last_interaction_delta") is None)
            elif name == "last_interaction_log1p":
                raw_delta = sample.get("last_interaction_delta")
                value = np.log1p(float(raw_delta)) if raw_delta is not None else None
            else:
                value = sample.get(name)
            if value is None:
                continue
            try:
                features[row_idx, col_idx] = float(value)
            except (TypeError, ValueError):
                continue

    predicted_utility = np.asarray(
        payload["model"].predict(features),
        dtype=np.float64,
    )
    if predicted_utility.shape != (len(samples),):
        raise ValueError(
            "Learned router returned an unexpected score shape: "
            f"{predicted_utility.shape} for {len(samples)} samples."
        )
    if not np.all(np.isfinite(predicted_utility)):
        raise ValueError("Learned router returned non-finite utility scores.")

    routing_priority = predicted_utility.copy()
    priority_blend = payload.get("priority_blend") or {}
    uncertainty_score = None
    if priority_blend:
        if uncertainty_center is None:
            raise ValueError(
                "Learned router priority_blend requires an uncertainty center."
            )
        score_field = str(uncertainty_prior.get("score_field", "semantic_mlp_score"))
        raw_scores = np.asarray(
            [float(sample[score_field]) for sample in samples],
            dtype=np.float64,
        )
        uncertainty_score = -np.abs(raw_scores - float(uncertainty_center))

        model_mean = float(priority_blend["model_score_mean"])
        model_std = max(float(priority_blend["model_score_std"]), 1e-12)
        uncertainty_mean = float(priority_blend["uncertainty_score_mean"])
        uncertainty_std = max(
            float(priority_blend["uncertainty_score_std"]),
            1e-12,
        )
        beta = float(priority_blend["uncertainty_beta"])
        routing_priority = (
            (predicted_utility - model_mean) / model_std
            + beta * (uncertainty_score - uncertainty_mean) / uncertainty_std
        )
        if not np.all(np.isfinite(routing_priority)):
            raise ValueError("Learned router returned non-finite blended priorities.")

    num_candidates = int(len(samples))
    requested_fraction = float(max(0.0, min(1.0, target_fraction)))
    target_count = int(round(requested_fraction * float(num_candidates)))
    if requested_fraction > 0.0 and num_candidates > 0:
        target_count = max(1, target_count)
    target_count = min(num_candidates, target_count)

    excluded_query_ids = {
        int(query_id) for query_id in payload.get("excluded_query_ids", ())
    }
    excluded_mask = np.asarray(
        [
            sample.get("query_id") is not None
            and int(sample["query_id"]) in excluded_query_ids
            for sample in samples
        ],
        dtype=bool,
    )
    eligible_indices = np.flatnonzero(~excluded_mask)
    if target_count > eligible_indices.size:
        raise ValueError(
            "Learned router has fewer eligible samples than the requested budget: "
            f"{eligible_indices.size} eligible for {target_count} calls."
        )
    eligible_order = np.lexsort(
        (eligible_indices, -routing_priority[eligible_indices])
    )
    routed_by_priority = eligible_indices[eligible_order[:target_count]]
    selected_indices = np.sort(routed_by_priority).astype(np.int64).tolist()
    selected_scores = predicted_utility[routed_by_priority]
    selected_priorities = routing_priority[routed_by_priority]
    return selected_indices, {
        "enabled": True,
        "selection_mode": "learned_router_top_fraction",
        "selection_rule": (
            "global_top_uncertainty_prior_plus_learned_auc_utility"
            if priority_blend
            else "global_top_predicted_llm_utility"
        ),
        "router_checkpoint": str(router_checkpoint),
        "router_schema_version": payload.get("schema_version", 1),
        "router_target": payload.get("target"),
        "router_training_examples": payload.get("training_examples"),
        "router_feature_names": list(feature_names),
        "router_priority_blend": priority_blend or None,
        "router_uncertainty_prior": uncertainty_prior or None,
        "excluded_training_query_count_configured": int(len(excluded_query_ids)),
        "excluded_candidate_count": int(excluded_mask.sum()),
        "eligible_candidate_count": int(eligible_indices.size),
        "requested_target_fraction": requested_fraction,
        "requested_target_count": int(target_count),
        "selected_count": int(len(selected_indices)),
        "num_candidates": num_candidates,
        "selected_fraction_realized": (
            float(len(selected_indices)) / float(num_candidates)
            if num_candidates > 0
            else 0.0
        ),
        "selected_router_score_min": (
            float(np.min(selected_scores)) if selected_scores.size else None
        ),
        "selected_router_score_max": (
            float(np.max(selected_scores)) if selected_scores.size else None
        ),
        "all_router_score_min": (
            float(np.min(predicted_utility)) if predicted_utility.size else None
        ),
        "all_router_score_max": (
            float(np.max(predicted_utility)) if predicted_utility.size else None
        ),
        "selected_routing_priority_min": (
            float(np.min(selected_priorities)) if selected_priorities.size else None
        ),
        "selected_routing_priority_max": (
            float(np.max(selected_priorities)) if selected_priorities.size else None
        ),
        "all_routing_priority_min": (
            float(np.min(routing_priority)) if routing_priority.size else None
        ),
        "all_routing_priority_max": (
            float(np.max(routing_priority)) if routing_priority.size else None
        ),
        "selected_uncertainty_score_min": (
            float(np.min(uncertainty_score[routed_by_priority]))
            if uncertainty_score is not None and routed_by_priority.size
            else None
        ),
        "selected_uncertainty_score_max": (
            float(np.max(uncertainty_score[routed_by_priority]))
            if uncertainty_score is not None and routed_by_priority.size
            else None
        ),
    }


def select_topk_rrf_middle_samples(
    samples,
    top_k,
    dtgb_eval_batch_size,
    *,
    score_field="rrf_score",
):
    """Materialize the DTGB-batch middle-band sample selection."""
    selected_sample_indices, meta = select_topk_rrf_middle_sample_indices(
        samples,
        top_k=top_k,
        dtgb_eval_batch_size=dtgb_eval_batch_size,
        score_field=score_field,
    )
    if not selected_sample_indices:
        return [], meta
    return [samples[idx] for idx in selected_sample_indices], meta


def compute_rrf_selected_slice_proxy(
    samples,
    llm_predictions,
    selection_meta=None,
    dtgb_eval_batch_size=None,
    *,
    score_field="rrf_score",
    backbone_name="rrf",
):
    """
    Compare LLM and raw RRF on the same selected sample slice.

    The selected slice is not a full DTGB evaluation batch anymore, so these
    metrics are intentionally pooled/global even if the caller runs full-set
    DTGB batchwise metrics elsewhere.
    """
    del dtgb_eval_batch_size

    labels = np.array([int(sample["label"]) for sample in samples], dtype=np.int64)
    llm_scores = np.asarray(llm_predictions, dtype=np.float64)
    backbone_scores = np.array(
        [float(sample.get(score_field, 0.0)) for sample in samples],
        dtype=np.float64,
    )

    llm_metrics = compute_prediction_metrics(llm_scores, labels, dtgb_eval_batch_size=None)
    backbone_metrics = compute_prediction_metrics(
        backbone_scores,
        labels,
        dtgb_eval_batch_size=None,
    )

    backbone_key = str(backbone_name).strip().lower().replace(" ", "_")
    if not backbone_key:
        backbone_key = "backbone"

    result = {
        "selection": selection_meta or {},
        "llm": llm_metrics,
        "backbone": backbone_metrics,
        "backbone_name": str(backbone_name),
        "delta_llm_minus_backbone": {
            "ap": float(llm_metrics["ap"] - backbone_metrics["ap"]),
            "auc": float(llm_metrics["auc"] - backbone_metrics["auc"]),
            "accuracy": float(llm_metrics["accuracy"] - backbone_metrics["accuracy"]),
            "ap_global": float(llm_metrics["ap_global"] - backbone_metrics["ap_global"]),
            "auc_global": float(llm_metrics["auc_global"] - backbone_metrics["auc_global"]),
        },
    }
    result[backbone_key] = backbone_metrics
    if backbone_key == "rrf":
        result["delta_llm_minus_rrf"] = dict(result["delta_llm_minus_backbone"])
    return result


def select_topk_middle_sample_indices_by_score(scores, top_k):
    """
    Select top-k sample indices closest to the global median score.
    Useful for metric-driven middle-region refinement without query coupling.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if top_k is None or top_k <= 0 or scores.size == 0:
        return [], {
            "enabled": False,
            "requested_top_k": 0,
            "selected_count": 0,
            "num_candidates": int(scores.size),
            "median_score": float(np.median(scores)) if scores.size > 0 else None,
            "selected_indices": [],
        }

    k = min(int(top_k), int(scores.size))
    median_score = float(np.median(scores))
    distance = np.abs(scores - median_score)
    order = np.lexsort((np.arange(scores.size), distance))
    selected = order[:k].tolist()

    return selected, {
        "enabled": True,
        "requested_top_k": int(top_k),
        "selected_count": int(k),
        "num_candidates": int(scores.size),
        "median_score": median_score,
        "selected_indices": [int(i) for i in selected],
    }


def minmax_scale_scores(scores):
    """
    Min-max scale scores to [0, 1]. If all values are equal, return 0.5.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return scores.copy()
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if hi <= lo:
        return np.full_like(scores, 0.5, dtype=np.float64)
    return (scores - lo) / (hi - lo)


__all__ = [
    "build_dtgb_eval_batches",
    "compute_rrf_selected_slice_proxy",
    "compute_two_component_gmm_overlap_scores",
    "fit_two_component_gaussian_mixture_1d",
    "minmax_scale_scores",
    "select_rrf_middle_sample_indices_validation_sampled_band",
    "select_rrf_middle_sample_indices_validation_sampled_gmm_overlap",
    "select_rrf_middle_sample_indices_pointwise_threshold_band",
    "select_topk_middle_sample_indices_by_score",
    "select_topk_rrf_middle_sample_indices",
    "select_topk_rrf_middle_samples",
]
