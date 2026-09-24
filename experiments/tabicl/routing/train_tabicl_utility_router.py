#!/usr/bin/env python3
"""Fit a dataset-agnostic TabICL router against aligned Llama utility.

Any standardized table split can provide the calibration support, and either
``test`` or ``inductive`` can be the deployment split. Query-grouped
out-of-fold corrector predictions provide two counterfactual scores for every
support row: no LLM call (p0) and an available Llama score (p1). A separate
TabICL classifier sees only causal pre-LLM graph/tabular features and learns
which rows belong to the top utility fraction. The final identity checkpoint
only freezes those learned priorities for an immediate exact-split run; it is
not a dataset-specific implementation of the router.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from tabicl import TabICLClassifier

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.seed_runs import DEFAULT_SEEDS, DEFAULT_ROUTER_SEEDS

from experiments.tabicl.tables.build_tabicl_train_llm_table import join_routed_debug  # noqa: E402
from experiments.tabicl.evaluation.evaluate_tabicl_train_llm_fusion import (  # noqa: E402
    LLM_FIELDS,
    STRUCTURAL_FIELDS,
    VARIANTS,
    batch_mean_metrics,
    build_feature_variants,
    empirical_percentile,
    fit_feature_state,
    metrics,
)
from experiments.modules.rrf.precomputed_router import (  # noqa: E402
    IdentityPriorityModel,
)
from experiments.modules.tabicl.utility_routing import (  # noqa: E402
    recent_paired_query_indices,
    top_indices,
)


DEFAULT_TABLE = "result/gdelt_tabicl_holdout_recent1k_llm_full_table.npz"
DEFAULT_SUPPORT_DEBUG = "result/gdelt_k10_tabicl_router_support2k_selected.jsonl"
DEFAULT_OUTPUT = "result/gdelt_tabicl_router2k_training.json"
DEFAULT_PREDICTIONS = "result/gdelt_tabicl_router2k_predictions.npz"
DEFAULT_ROUTE_CHECKPOINT = "saved_models/gdelt_tabicl_router2k_full_route.joblib"
ALIGNMENT_VARIANT = "score_fusion_no_heuristics"


def empirical_win_rate_left(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """P(score > reference), with half credit for ties."""
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    scores = np.asarray(scores, dtype=np.float64)
    left = np.searchsorted(reference, scores, side="left")
    right = np.searchsorted(reference, scores, side="right")
    return (left + 0.5 * (right - left)) / float(reference.size)


def empirical_win_rate_right(reference: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """P(reference > score), with half credit for ties."""
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    scores = np.asarray(scores, dtype=np.float64)
    left = np.searchsorted(reference, scores, side="left")
    right = np.searchsorted(reference, scores, side="right")
    return (reference.size - right + 0.5 * (right - left)) / float(reference.size)


def marginal_auc_utility(
    baseline_scores: np.ndarray,
    replacement_scores: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Per-row empirical AUC change when p1 replaces p0."""
    positives = baseline_scores[labels == 1]
    negatives = baseline_scores[labels == 0]
    utility = np.empty(labels.size, dtype=np.float64)
    positive_mask = labels == 1
    utility[positive_mask] = empirical_win_rate_left(
        replacement_scores[positive_mask], negatives
    ) - empirical_win_rate_left(baseline_scores[positive_mask], negatives)
    utility[~positive_mask] = empirical_win_rate_right(
        positives, replacement_scores[~positive_mask]
    ) - empirical_win_rate_right(positives, baseline_scores[~positive_mask])
    return utility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Metadata only; the standardized NPZ table is the data interface.",
    )
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--support-debug-jsonl", default=DEFAULT_SUPPORT_DEBUG)
    parser.add_argument(
        "--calibration-split",
        default="train",
        help="NPZ prefix supplying labeled calibration/support rows.",
    )
    parser.add_argument(
        "--support-debug-split",
        default=None,
        help=(
            "Split label stored in support JSONL rows; defaults to "
            "--calibration-split."
        ),
    )
    parser.add_argument(
        "--allow-extra-support-debug-rows",
        action="store_true",
        help=(
            "Accept a full-split all-call JSONL and retain only the selected "
            "calibration support rows."
        ),
    )
    parser.add_argument(
        "--deployment-split",
        choices=("test", "inductive"),
        default="test",
        help="Unlabeled NPZ split for which routing priorities are frozen.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--route-checkpoint", default=DEFAULT_ROUTE_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--positive-context", type=int, default=1000)
    parser.add_argument("--budget-fraction", type=float, default=0.20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--alignment-seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--router-seeds", nargs="+", type=int, default=list(DEFAULT_ROUTER_SEEDS))
    parser.add_argument("--n-estimators", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--metric-batch-size", type=int, default=256)
    parser.add_argument("--gnn-score-atol", type=float, default=1e-5)
    parser.add_argument(
        "--alignment-variant",
        choices=VARIANTS,
        default=ALIGNMENT_VARIANT,
        help=(
            "TabICL corrector whose counterfactual call/no-call utility is "
            "used as router supervision."
        ),
    )
    args = parser.parse_args()
    if not 0.0 < args.budget_fraction < 1.0:
        parser.error("--budget-fraction must be in (0, 1)")
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if not args.calibration_split.strip():
        parser.error("--calibration-split cannot be empty")
    args.calibration_split = args.calibration_split.strip()
    if args.support_debug_split is None:
        args.support_debug_split = args.calibration_split
    return args


def require_router_table_schema(
    payload: dict[str, np.ndarray],
    *,
    calibration_split: str,
    deployment_split: str,
) -> None:
    """Validate the shared table contract before any expensive TabICL fit."""
    required = {"route_center", "validation_gnn"}
    for split in (calibration_split, deployment_split):
        required.update(f"{split}_{field}" for field in STRUCTURAL_FIELDS)
        required.update(f"{split}_{field}" for field in LLM_FIELDS)
        required.update((f"{split}_source_ids", f"{split}_target_ids"))
    missing = sorted(required.difference(payload))
    if missing:
        raise KeyError(
            "TabICL router table is missing required arrays: " + ", ".join(missing)
        )

    for split in (calibration_split, deployment_split):
        expected = len(payload[f"{split}_labels"])
        fields = STRUCTURAL_FIELDS + LLM_FIELDS + ("source_ids", "target_ids")
        for field in fields:
            key = f"{split}_{field}"
            if len(payload[key]) != expected:
                raise ValueError(
                    f"{key} has {len(payload[key])} rows; expected {expected}"
                )


def subset_view(
    payload: dict[str, np.ndarray],
    *,
    source_split: str,
    indices: np.ndarray,
    target_split: str,
    diagnostics: dict[str, np.ndarray] | None = None,
    route_override: np.ndarray | None = None,
    available_override: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Materialize the exact schema expected by the alignment feature builder."""

    indices = np.asarray(indices, dtype=np.int64)
    view: dict[str, np.ndarray] = {
        "route_center": np.asarray(payload["route_center"]),
        "validation_gnn": np.asarray(payload["validation_gnn"]),
    }
    for field in STRUCTURAL_FIELDS:
        source = payload[f"{source_split}_{field}"]
        view[f"{target_split}_{field}"] = np.asarray(source[indices]).copy()
    if route_override is not None:
        route_override = np.asarray(route_override, dtype=bool)
        if len(route_override) != len(indices):
            raise ValueError("route_override length mismatch")
        view[f"{target_split}_route"] = route_override.copy()

    diagnostic_source = payload if diagnostics is None else diagnostics
    for field in LLM_FIELDS:
        key = f"{source_split}_{field}"
        if key not in diagnostic_source:
            raise KeyError(f"Missing diagnostic array {key}")
        view[f"{target_split}_{field}"] = np.asarray(
            diagnostic_source[key][indices]
        ).copy()
    if available_override is not None:
        available_override = np.asarray(available_override, dtype=bool)
        if len(available_override) != len(indices):
            raise ValueError("available_override length mismatch")
        view[f"{target_split}_llm_available"] = available_override.copy()
    return view


def fit_alignment_predict_both(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_no_call: np.ndarray,
    X_call: np.ndarray,
    *,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed + 10_000).permutation(len(y_context))
    classifier = TabICLClassifier(
        n_estimators=args.n_estimators,
        batch_size=args.batch_size,
        kv_cache="repr",
        device=args.device,
        random_state=seed,
        verbose=False,
    )
    classifier.fit(X_context[order], y_context[order])
    # Predict the two counterfactual worlds independently. TabICL's numerical
    # preprocessing may depend on the query table, so concatenating p0 and p1
    # would let the call-state distribution influence the no-call prediction.
    no_call_probability = np.asarray(
        classifier.predict_proba(X_no_call)[:, 1], dtype=np.float64
    )
    call_probability = np.asarray(
        classifier.predict_proba(X_call)[:, 1], dtype=np.float64
    )
    del classifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return no_call_probability, call_probability


def fit_router_predict(
    X_context: np.ndarray,
    route_labels: np.ndarray,
    X_query: np.ndarray,
    *,
    seed: int,
    args: argparse.Namespace,
) -> np.ndarray:
    order = np.random.default_rng(seed + 20_000).permutation(len(route_labels))
    classifier = TabICLClassifier(
        n_estimators=args.n_estimators,
        batch_size=args.batch_size,
        kv_cache="repr",
        device=args.device,
        random_state=seed,
        verbose=False,
    )
    classifier.fit(X_context[order], route_labels[order])
    probability = np.asarray(classifier.predict_proba(X_query)[:, 1], dtype=np.float64)
    del classifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probability


def router_features(
    payload: dict[str, np.ndarray],
    *,
    split: str,
    indices: np.ndarray,
    no_call_probability: np.ndarray,
    no_call_reference: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    indices = np.asarray(indices, dtype=np.int64)
    gnn = np.asarray(payload[f"{split}_gnn"][indices], dtype=np.float64)
    center = float(payload["route_center"])
    gnn_pct = empirical_percentile(payload["validation_gnn"], gnn)
    no_call_probability = np.asarray(no_call_probability, dtype=np.float64)
    if len(no_call_probability) != len(indices):
        raise ValueError("no-call probability length mismatch")
    no_call_pct = empirical_percentile(no_call_reference, no_call_probability)

    last_delta = np.asarray(
        payload[f"{split}_last_interaction_delta"][indices], dtype=np.float64
    ).copy()
    has_past = np.isfinite(last_delta).astype(np.float64)
    last_delta[~np.isfinite(last_delta)] = 1_000_000.0
    matrix = np.column_stack(
        (
            gnn,
            gnn_pct,
            np.abs(gnn - center),
            no_call_probability,
            np.abs(no_call_probability - 0.5),
            no_call_pct,
            no_call_pct - gnn_pct,
            np.log1p(
                np.maximum(payload[f"{split}_source_popularity"][indices], 0.0)
            ),
            np.log1p(
                np.maximum(payload[f"{split}_target_popularity"][indices], 0.0)
            ),
            np.log1p(
                np.maximum(payload[f"{split}_past_interactions"][indices], 0.0)
            ),
            np.log1p(np.maximum(last_delta, 0.0)),
            has_past,
            np.log1p(
                np.maximum(payload[f"{split}_common_neighbor"][indices], 0.0)
            ),
        )
    )
    names = [
        "gnn_raw",
        "gnn_validation_percentile",
        "gnn_distance_to_route_center",
        "tabicl_no_call_probability",
        "tabicl_no_call_confidence",
        "tabicl_no_call_support_percentile",
        "tabicl_no_call_minus_gnn_percentile_gap",
        "log_source_popularity",
        "log_target_popularity",
        "log_past_interactions",
        "log_last_interaction_delta",
        "has_past_interaction",
        "log_common_neighbor",
    ]
    if not np.all(np.isfinite(matrix)):
        bad = [names[i] for i in np.flatnonzero(~np.all(np.isfinite(matrix), axis=0))]
        raise ValueError(f"Non-finite router features: {bad}")
    return matrix.astype(np.float32), names


def policy_metrics(
    labels: np.ndarray,
    query_ids: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    chosen: np.ndarray,
    metric_batch_size: int,
) -> dict:
    hybrid = np.asarray(p0, dtype=np.float64).copy()
    hybrid[np.asarray(chosen, dtype=np.int64)] = p1[np.asarray(chosen, dtype=np.int64)]
    return {
        "global": metrics(labels, hybrid),
        "dtgb_batch_mean": batch_mean_metrics(
            labels, hybrid, query_ids, metric_batch_size
        ),
    }


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    with np.load(args.table, allow_pickle=False) as loaded:
        payload = {key: loaded[key] for key in loaded.files}
    calibration_split = args.calibration_split
    deployment_split = args.deployment_split
    require_router_table_schema(
        payload,
        calibration_split=calibration_split,
        deployment_split=deployment_split,
    )

    context_indices = recent_paired_query_indices(
        payload[f"{calibration_split}_labels"],
        payload[f"{calibration_split}_query_ids"],
        payload[f"{calibration_split}_timestamps"],
        args.positive_context,
    )
    context_mask = np.zeros(
        len(payload[f"{calibration_split}_labels"]), dtype=bool
    )
    context_mask[context_indices] = True
    full_diagnostics = join_routed_debug(
        payload=payload,
        split=calibration_split,
        debug_jsonl=args.support_debug_jsonl,
        route_mask=context_mask,
        expected_selected=len(context_indices),
        expected_debug_split=args.support_debug_split,
        gnn_score_atol=args.gnn_score_atol,
        allow_unselected=args.allow_extra_support_debug_rows,
    )

    labels = payload[f"{calibration_split}_labels"][context_indices].astype(
        np.int64
    )
    groups = payload[f"{calibration_split}_query_ids"][context_indices].astype(
        np.int64
    )
    if np.unique(groups).size != args.positive_context:
        raise ValueError("Context query groups are not one positive/negative pair each")
    relative = np.arange(len(context_indices), dtype=np.int64)
    splitter = GroupKFold(n_splits=args.folds)
    oof_p0_by_seed = {
        seed: np.full(len(context_indices), np.nan, dtype=np.float64)
        for seed in args.alignment_seeds
    }
    oof_p1_by_seed = {
        seed: np.full(len(context_indices), np.nan, dtype=np.float64)
        for seed in args.alignment_seeds
    }
    alignment_folds = []
    for fold, (train_rel, valid_rel) in enumerate(
        splitter.split(relative, labels, groups=groups)
    ):
        train_abs = context_indices[train_rel]
        valid_abs = context_indices[valid_rel]
        train_view = subset_view(
            payload,
            source_split=calibration_split,
            indices=train_abs,
            target_split="train",
        )
        state = fit_feature_state(train_view, np.arange(len(train_abs)))
        train_features, _ = build_feature_variants(train_view, "train", state)
        p0_view = subset_view(
            payload,
            source_split=calibration_split,
            indices=valid_abs,
            target_split="test",
            diagnostics=full_diagnostics,
            route_override=np.zeros(len(valid_abs), dtype=bool),
            available_override=np.zeros(len(valid_abs), dtype=bool),
        )
        p1_view = subset_view(
            payload,
            source_split=calibration_split,
            indices=valid_abs,
            target_split="test",
            diagnostics=full_diagnostics,
            route_override=np.ones(len(valid_abs), dtype=bool),
            available_override=np.ones(len(valid_abs), dtype=bool),
        )
        p0_features, _ = build_feature_variants(p0_view, "test", state)
        p1_features, _ = build_feature_variants(p1_view, "test", state)
        for seed in args.alignment_seeds:
            p0, p1 = fit_alignment_predict_both(
                train_features[args.alignment_variant],
                labels[train_rel],
                p0_features[args.alignment_variant],
                p1_features[args.alignment_variant],
                seed=seed,
                args=args,
            )
            oof_p0_by_seed[seed][valid_rel] = p0
            oof_p1_by_seed[seed][valid_rel] = p1
        alignment_folds.append(
            {
                "fold": fold,
                "train_rows": int(len(train_rel)),
                "validation_rows": int(len(valid_rel)),
                "query_overlap": int(
                    np.intersect1d(groups[train_rel], groups[valid_rel]).size
                ),
                "train_llm_available": int(
                    train_view["train_llm_available"].sum()
                ),
            }
        )
        print(f"Alignment OOF fold {fold + 1}/{args.folds} complete", flush=True)

    for seed in args.alignment_seeds:
        if not (
            np.all(np.isfinite(oof_p0_by_seed[seed]))
            and np.all(np.isfinite(oof_p1_by_seed[seed]))
        ):
            raise ValueError(f"Incomplete OOF alignment predictions for seed {seed}")
    oof_p0 = np.mean(np.vstack(list(oof_p0_by_seed.values())), axis=0)
    oof_p1 = np.mean(np.vstack(list(oof_p1_by_seed.values())), axis=0)
    utility = marginal_auc_utility(oof_p0, oof_p1, labels)
    route_count = max(1, int(round(args.budget_fraction * len(labels))))
    route_labels = np.zeros(len(labels), dtype=np.int64)
    route_labels[top_indices(utility, route_count)] = 1

    X_support, feature_names = router_features(
        payload,
        split=calibration_split,
        indices=context_indices,
        no_call_probability=oof_p0,
        no_call_reference=oof_p0,
    )
    oof_router_by_seed = {
        seed: np.full(len(labels), np.nan, dtype=np.float64)
        for seed in args.router_seeds
    }
    router_folds = []
    for fold, (train_rel, valid_rel) in enumerate(
        splitter.split(X_support, route_labels, groups=groups)
    ):
        for seed in args.router_seeds:
            oof_router_by_seed[seed][valid_rel] = fit_router_predict(
                X_support[train_rel],
                route_labels[train_rel],
                X_support[valid_rel],
                seed=seed,
                args=args,
            )
        router_folds.append(
            {
                "fold": fold,
                "train_rows": int(len(train_rel)),
                "validation_rows": int(len(valid_rel)),
                "query_overlap": int(
                    np.intersect1d(groups[train_rel], groups[valid_rel]).size
                ),
            }
        )
        print(f"Router OOF fold {fold + 1}/{args.folds} complete", flush=True)
    oof_router = np.mean(np.vstack(list(oof_router_by_seed.values())), axis=0)

    # Fit the completed sparse-availability alignment recipe on all support
    # rows and form the no-call representation used by the deployable router.
    context_train_view = subset_view(
        payload,
        source_split=calibration_split,
        indices=context_indices,
        target_split="train",
    )
    final_state = fit_feature_state(
        context_train_view, np.arange(len(context_indices))
    )
    context_features, _ = build_feature_variants(
        context_train_view, "train", final_state
    )
    deployment_indices = np.arange(
        len(payload[f"{deployment_split}_labels"]), dtype=np.int64
    )
    deployment_p0_view = subset_view(
        payload,
        source_split=deployment_split,
        indices=deployment_indices,
        target_split="test",
        route_override=np.zeros(len(deployment_indices), dtype=bool),
        available_override=np.zeros(len(deployment_indices), dtype=bool),
    )
    deployment_p0_features, _ = build_feature_variants(
        deployment_p0_view, "test", final_state
    )
    deployment_p0_by_seed = {}
    for seed in args.alignment_seeds:
        p0, _ = fit_alignment_predict_both(
            context_features[args.alignment_variant],
            labels,
            deployment_p0_features[args.alignment_variant],
            deployment_p0_features[args.alignment_variant],
            seed=seed,
            args=args,
        )
        deployment_p0_by_seed[seed] = p0
        print(
            f"Full-{deployment_split} no-call alignment seed {seed} complete",
            flush=True,
        )
    deployment_p0 = np.mean(
        np.vstack(list(deployment_p0_by_seed.values())), axis=0
    )
    X_deployment, deployment_feature_names = router_features(
        payload,
        split=deployment_split,
        indices=deployment_indices,
        no_call_probability=deployment_p0,
        no_call_reference=oof_p0,
    )
    if deployment_feature_names != feature_names:
        raise AssertionError("Router calibration/deployment feature schemas differ")

    deployment_router_by_seed = {}
    for seed in args.router_seeds:
        deployment_router_by_seed[seed] = fit_router_predict(
            X_support,
            route_labels,
            X_deployment,
            seed=seed,
            args=args,
        )
        print(
            f"Full-{deployment_split} router seed {seed} complete", flush=True
        )
    deployment_priority = np.mean(
        np.vstack(list(deployment_router_by_seed.values())), axis=0
    )
    deployment_route_count = max(
        1, int(round(args.budget_fraction * len(deployment_priority)))
    )
    selected_deployment = top_indices(
        deployment_priority, deployment_route_count
    )

    route_model = IdentityPriorityModel(
        payload[f"{deployment_split}_query_ids"],
        payload[f"{deployment_split}_target_ids"],
        deployment_priority,
        default_priority=None,
    )
    route_checkpoint = {
        "schema_version": 4,
        "model": route_model,
        "feature_names": ["query_id", "target_id"],
        "target": "tabicl_top_fraction_marginal_auc_utility",
        "training_examples": int(len(labels)),
        "training_query_groups": int(np.unique(groups).size),
        "router_feature_names": feature_names,
        "budget_fraction": float(args.budget_fraction),
        "selected_count": int(deployment_route_count),
        "source_table": str(args.table),
        "support_debug_jsonl": str(args.support_debug_jsonl),
        "dataset_name": args.dataset_name,
        "calibration_split": calibration_split,
        "deployment_split": deployment_split,
        "uses_evaluation_labels": False,
        "uses_test_labels": False,
        "identity_fields": ["query_id", "target_id"],
    }
    route_path = Path(args.route_checkpoint)
    route_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(route_checkpoint, route_path)

    uncertainty_priority = -np.abs(
        payload[f"{calibration_split}_gnn"][context_indices]
        - float(payload["route_center"])
    )
    support_policies = {
        "no_call": policy_metrics(
            labels,
            groups,
            oof_p0,
            oof_p1,
            np.empty(0, dtype=np.int64),
            args.metric_batch_size,
        ),
        "uncertainty": policy_metrics(
            labels,
            groups,
            oof_p0,
            oof_p1,
            top_indices(uncertainty_priority, route_count),
            args.metric_batch_size,
        ),
        "tabicl_router_oof": policy_metrics(
            labels,
            groups,
            oof_p0,
            oof_p1,
            top_indices(oof_router, route_count),
            args.metric_batch_size,
        ),
        "oracle_marginal_utility": policy_metrics(
            labels,
            groups,
            oof_p0,
            oof_p1,
            top_indices(utility, route_count),
            args.metric_batch_size,
        ),
        "all_call": policy_metrics(
            labels,
            groups,
            oof_p0,
            oof_p1,
            np.arange(len(labels), dtype=np.int64),
            args.metric_batch_size,
        ),
    }
    result = {
        "experiment": "generic_tabicl_utility_router",
        "dataset_name": args.dataset_name,
        "table": str(args.table),
        "support_debug_jsonl": str(args.support_debug_jsonl),
        "alignment_variant": args.alignment_variant,
        "protocol": {
            "calibration_split": calibration_split,
            "support_debug_split": args.support_debug_split,
            "support_debug_allows_extra_rows": bool(
                args.allow_extra_support_debug_rows
            ),
            "deployment_split": deployment_split,
            "support_rows": int(len(labels)),
            "support_query_groups": int(np.unique(groups).size),
            "support_positive_rows": int(np.sum(labels == 1)),
            "support_negative_rows": int(np.sum(labels == 0)),
            "original_sparse_alignment_llm_rows": int(
                payload[f"{calibration_split}_llm_available"][
                    context_indices
                ].sum()
            ),
            "counterfactual_llm_rows": int(
                full_diagnostics[f"{calibration_split}_llm_available"][
                    context_indices
                ].sum()
            ),
            "alignment_oof_folds": alignment_folds,
            "router_oof_folds": router_folds,
            "budget_fraction": float(args.budget_fraction),
            "support_route_count": int(route_count),
            "deployment_rows": int(len(deployment_priority)),
            "deployment_route_count": int(deployment_route_count),
            "evaluation_labels_used_for_routing": False,
        },
        "features": {
            "router": feature_names,
            "post_llm_fields_in_router": [],
            "target": "top_budget_fraction_of_oof_marginal_global_auc_utility",
        },
        "config": {
            "alignment_seeds": args.alignment_seeds,
            "router_seeds": args.router_seeds,
            "n_estimators": args.n_estimators,
            "batch_size": args.batch_size,
            "folds": args.folds,
            "device": args.device,
        },
        "support_target": {
            "utility_mean": float(np.mean(utility)),
            "utility_std": float(np.std(utility)),
            "utility_positive_fraction": float(np.mean(utility > 0.0)),
            "route_label_positive_fraction": float(np.mean(route_labels)),
        },
        "router_oof": {
            "winner_auc": float(roc_auc_score(route_labels, oof_router)),
            "winner_ap": float(average_precision_score(route_labels, oof_router)),
            "utility_spearman": float(spearmanr(utility, oof_router).statistic),
            "policies": support_policies,
        },
        "deployment_route": {
            "split": deployment_split,
            "priority_min": float(np.min(deployment_priority)),
            "priority_max": float(np.max(deployment_priority)),
            "selected_priority_min": float(
                np.min(deployment_priority[selected_deployment])
            ),
            "selected_priority_max": float(
                np.max(deployment_priority[selected_deployment])
            ),
            "selected_positive_count_debug_only_after_route": int(
                np.sum(
                    payload[f"{deployment_split}_labels"][selected_deployment]
                    == 1
                )
            ),
        },
        "artifacts": {
            "predictions": str(args.predictions),
            "route_checkpoint": str(args.route_checkpoint),
        },
        "runtime_seconds": float(time.perf_counter() - start),
    }

    prediction_payload = {
        "support_indices": context_indices,
        "support_query_ids": groups,
        "support_labels": labels,
        "support_oof_p0": oof_p0.astype(np.float32),
        "support_oof_p1": oof_p1.astype(np.float32),
        "support_utility": utility.astype(np.float32),
        "support_route_labels": route_labels,
        "support_oof_router_priority": oof_router.astype(np.float32),
        "deployment_split": np.asarray(deployment_split),
        "deployment_query_ids": payload[
            f"{deployment_split}_query_ids"
        ].astype(np.int64),
        "deployment_target_ids": payload[
            f"{deployment_split}_target_ids"
        ].astype(np.int64),
        "deployment_no_call_probability": deployment_p0.astype(np.float32),
        "deployment_router_priority": deployment_priority.astype(np.float32),
        "deployment_selected_indices": selected_deployment.astype(np.int64),
    }
    # Preserve the established prediction keys for existing transductive
    # consumers while making the primary schema split-neutral.
    if deployment_split == "test":
        prediction_payload.update(
            {
                "test_query_ids": prediction_payload["deployment_query_ids"],
                "test_target_ids": prediction_payload["deployment_target_ids"],
                "test_no_call_probability": prediction_payload[
                    "deployment_no_call_probability"
                ],
                "test_router_priority": prediction_payload[
                    "deployment_router_priority"
                ],
                "test_selected_indices": prediction_payload[
                    "deployment_selected_indices"
                ],
            }
        )
    for seed, values in oof_p0_by_seed.items():
        prediction_payload[f"support_oof_p0_seed_{seed}"] = values.astype(np.float32)
    for seed, values in oof_p1_by_seed.items():
        prediction_payload[f"support_oof_p1_seed_{seed}"] = values.astype(np.float32)
    for seed, values in oof_router_by_seed.items():
        prediction_payload[f"support_oof_router_seed_{seed}"] = values.astype(np.float32)
    for seed, values in deployment_p0_by_seed.items():
        prediction_payload[f"deployment_no_call_seed_{seed}"] = values.astype(
            np.float32
        )
        if deployment_split == "test":
            prediction_payload[f"test_no_call_seed_{seed}"] = prediction_payload[
                f"deployment_no_call_seed_{seed}"
            ]
    for seed, values in deployment_router_by_seed.items():
        prediction_payload[f"deployment_router_priority_seed_{seed}"] = (
            values.astype(np.float32)
        )
        if deployment_split == "test":
            prediction_payload[f"test_router_priority_seed_{seed}"] = (
                prediction_payload[f"deployment_router_priority_seed_{seed}"]
            )

    predictions_path = Path(args.predictions)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(predictions_path, **prediction_payload)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        "TabICL router training complete: "
        f"OOF winner AUC={result['router_oof']['winner_auc']:.6f}, "
        f"{deployment_split} route={deployment_route_count:,}/"
        f"{len(deployment_priority):,}",
        flush=True,
    )
    print(f"Route checkpoint: {route_path}", flush=True)
    print(f"Result: {output_path}", flush=True)


if __name__ == "__main__":
    main()
