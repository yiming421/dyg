#!/usr/bin/env python3
"""Evaluate train-context TabICL fusion with progressively richer LLM features.

The input table must contain the held-out GNN scores, routed LLM diagnostics,
and cheap structural columns for ``train`` and the selected evaluation split. Exactly
the most-recent 1,000 positive train queries (and their matched negatives) are
used as labeled TabICL context.  Every imputation statistic and every LLM
empirical reference is fitted only on LLM-available rows in that context.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tabicl import TabICLClassifier

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.seed_runs import DEFAULT_SEEDS


DEFAULT_TABLE = "result/gdelt_tabicl_holdout_recent1k_llm_full_table.npz"
DEFAULT_OUTPUT = "result/gdelt_tabicl_holdout_recent1k_llm_rich_fusion.json"
LLM_FIELDS = (
    "llm_scores",
    "llm_logprob_0",
    "llm_logprob_1",
    "llm_logit_margin",
    "llm_token_mass",
    "llm_entropy",
    "llm_prompt_tokens",
    "llm_available",
)
STRUCTURAL_FIELDS = (
    "gnn",
    "route",
    "labels",
    "timestamps",
    "source_popularity",
    "target_popularity",
    "past_interactions",
    "last_interaction_delta",
    "common_neighbor",
    "query_ids",
)
VARIANTS = (
    "structural",
    "llm_scalar",
    "score_fusion_no_heuristics",
    "score_engineered",
    "rich_raw",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--predictions",
        default=None,
        help="Prediction NPZ (default: <output stem>_predictions.npz)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--eval-split",
        choices=("test", "inductive"),
        default="test",
        help="NPZ prefix to evaluate (default preserves transductive test behavior).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=None,
        help="TabICL variants to fit (default: all variants in the established order).",
    )
    parser.add_argument("--positive-context", type=int, default=1000)
    parser.add_argument(
        "--allow-gnn-trained-context",
        action="store_true",
        help=(
            "Allow the labeled train context to contain rows previously seen by "
            "the frozen GNN. The TabICL fit still uses train rows only."
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--n-estimators", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--metric-batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.positive_context < 1:
        parser.error("--positive-context must be positive")
    if not args.seeds:
        parser.error("--seeds cannot be empty")
    if args.metric_batch_size < 1:
        parser.error("--metric-batch-size must be positive")
    if args.variants is None:
        args.variants = list(VARIANTS)
    else:
        # Preserve requested order while preventing accidental duplicate fits.
        args.variants = list(dict.fromkeys(args.variants))
    return args


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) == 0 or np.unique(labels).size != 2:
        raise ValueError("AP/AUC evaluation requires both labels")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Metric scores contain NaN or infinity")
    return {
        "ap": float(average_precision_score(labels, scores)),
        "auc": float(roc_auc_score(labels, scores)),
    }


def batch_mean_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    query_ids: np.ndarray,
    query_batch_size: int,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    query_ids = np.asarray(query_ids, dtype=np.int64)
    if len(labels) != len(scores) or len(labels) != len(query_ids):
        raise ValueError("DTGB metric labels, scores, and query IDs must align")
    if len(labels) % 2:
        raise ValueError("DTGB batch-mean evaluation requires paired rows")
    paired_labels = labels.reshape(-1, 2)
    paired_query_ids = query_ids.reshape(-1, 2)
    if not (
        np.all(paired_labels[:, 0] == 1)
        and np.all(paired_labels[:, 1] == 0)
    ):
        raise ValueError(
            "DTGB batch-mean evaluation expects [positive, negative] row pairs"
        )
    if not np.all(paired_query_ids[:, 0] == paired_query_ids[:, 1]):
        raise ValueError("Each DTGB positive/negative pair must share a query ID")
    if np.unique(paired_query_ids[:, 0]).size != len(paired_query_ids):
        raise ValueError("DTGB positive-query IDs must be unique")
    aps: list[float] = []
    aucs: list[float] = []
    query_count = len(paired_labels)
    for query_start in range(0, query_count, query_batch_size):
        query_end = min(query_start + query_batch_size, query_count)
        row_start = 2 * query_start
        row_end = 2 * query_end
        batch_labels = labels[row_start:row_end]
        batch_scores = scores[row_start:row_end]
        aps.append(float(average_precision_score(batch_labels, batch_scores)))
        aucs.append(float(roc_auc_score(batch_labels, batch_scores)))
    if not aps:
        raise ValueError("No two-class DTGB metric batches were found")
    return {
        "ap": float(np.mean(aps)),
        "auc": float(np.mean(aucs)),
        "batches": len(aps),
    }


def aggregate(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(reference) == 0 or not np.all(np.isfinite(reference)):
        raise ValueError("An empirical reference must be nonempty and finite")
    sorted_reference = np.sort(reference)
    ranks = np.searchsorted(sorted_reference, values, side="right")
    return ranks.astype(np.float64) / float(len(sorted_reference))


def require_table_schema(
    table: np.lib.npyio.NpzFile,
    eval_split: str = "test",
) -> None:
    required = {
        "route_center",
        "validation_gnn",
        "gnn_holdout_positive_count",
    }
    for split in ("train", eval_split):
        required.update(f"{split}_{field}" for field in STRUCTURAL_FIELDS)
        required.update(f"{split}_{field}" for field in LLM_FIELDS)
    missing = sorted(required.difference(table.files))
    if missing:
        raise KeyError(f"Combined table is missing required arrays: {missing}")

    for split in ("train", eval_split):
        expected = len(table[f"{split}_labels"])
        for field in STRUCTURAL_FIELDS + LLM_FIELDS:
            values = table[f"{split}_{field}"]
            if len(values) != expected:
                raise ValueError(
                    f"{split}_{field} has {len(values)} rows; expected {expected}"
                )


def llm_availability(
    table: np.lib.npyio.NpzFile,
    split: str,
) -> np.ndarray:
    route = table[f"{split}_route"].astype(bool)
    probability = table[f"{split}_llm_scores"].astype(np.float64)
    available = route & np.isfinite(probability)
    recorded = table[f"{split}_llm_available"].astype(bool)
    if not np.array_equal(available, recorded):
        mismatches = int(np.count_nonzero(available != recorded))
        raise ValueError(
            f"{split}_llm_available disagrees with route & finite probability "
            f"on {mismatches} rows"
        )
    if np.any((probability[available] < 0.0) | (probability[available] > 1.0)):
        raise ValueError(f"{split} has an available LLM probability outside [0, 1]")
    return available


def most_recent_query_indices(
    table: np.lib.npyio.NpzFile,
    positive_count: int,
) -> np.ndarray:
    labels = table["train_labels"].astype(np.int64)
    query_ids = table["train_query_ids"].astype(np.int64)
    timestamps = table["train_timestamps"].astype(np.float64)
    positive_indices = np.flatnonzero(labels == 1)
    if len(positive_indices) < positive_count:
        raise ValueError(
            f"Requested {positive_count} positives, but train has only "
            f"{len(positive_indices)}"
        )

    # The final key is the original row index, making equal-time selection
    # deterministic while retaining the most-recent-query protocol.
    order = np.lexsort(
        (
            positive_indices,
            query_ids[positive_indices],
            timestamps[positive_indices],
        )
    )
    selected_positive = positive_indices[order[-positive_count:]]
    selected_query_ids = query_ids[selected_positive]
    if np.unique(selected_query_ids).size != positive_count:
        raise ValueError("Selected positive context rows do not have unique query IDs")

    selected_mask = np.isin(query_ids, selected_query_ids)
    selected = np.flatnonzero(selected_mask)
    if len(selected) != 2 * positive_count:
        raise ValueError(
            "Selected train queries must contain exactly one positive and one "
            f"matched negative; found {len(selected)} rows"
        )
    for query_id in selected_query_ids:
        query_rows = selected[query_ids[selected] == query_id]
        query_labels = labels[query_rows]
        if (
            len(query_rows) != 2
            or np.count_nonzero(query_labels == 1) != 1
            or np.count_nonzero(query_labels == 0) != 1
        ):
            raise ValueError(
                f"Query {query_id} does not have exactly one positive and one negative"
            )
        if timestamps[query_rows[0]] != timestamps[query_rows[1]]:
            raise ValueError(f"Matched rows for query {query_id} have different times")

    # Preserve source-table order; fit_predict applies a seed-specific shuffle.
    return selected


def context_median(
    values: np.ndarray,
    context_available: np.ndarray,
    name: str,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    usable = context_available & np.isfinite(values)
    if not np.any(usable):
        raise ValueError(f"No finite available context values for {name}")
    return float(np.median(values[usable]))


@dataclass(frozen=True)
class FeatureState:
    context_indices: np.ndarray
    gnn_reference: np.ndarray
    gnn_available_context_reference: np.ndarray
    llm_probability_reference: np.ndarray
    fills: dict[str, float]


def fit_feature_state(
    table: np.lib.npyio.NpzFile,
    context_indices: np.ndarray,
) -> FeatureState:
    train_available = llm_availability(table, "train")
    context_available = train_available[context_indices]
    if not np.any(context_available):
        raise ValueError("Selected train context contains no routed LLM scores")

    def context_values(field: str) -> np.ndarray:
        return table[f"train_{field}"].astype(np.float64)[context_indices]

    probability = context_values("llm_scores")
    gnn = context_values("gnn")
    route_center = float(table["route_center"])
    gnn_reference = table["validation_gnn"].astype(np.float64)
    if not np.all(np.isfinite(gnn_reference)):
        raise ValueError("validation_gnn must be entirely finite")
    gnn_available_context_reference = gnn[context_available].copy()
    gnn_available_context_percentile = np.full(
        len(context_indices), np.nan, dtype=np.float64
    )
    gnn_available_context_percentile[context_available] = empirical_percentile(
        gnn_available_context_reference,
        gnn[context_available],
    )
    probability_reference = probability[context_available].copy()
    probability_percentile = np.full(len(context_indices), np.nan, dtype=np.float64)
    probability_percentile[context_available] = empirical_percentile(
        probability_reference,
        probability[context_available],
    )

    confidence = 2.0 * np.abs(probability - 0.5)
    signed_gap = (
        probability_percentile - gnn_available_context_percentile
    )
    absolute_gap = np.abs(signed_gap)
    hard_agreement = (
        (probability >= 0.5) == (gnn >= route_center)
    ).astype(np.float64)
    log_prompt_length = np.log1p(
        np.maximum(context_values("llm_prompt_tokens"), 0.0)
    )
    fills = {
        "llm_probability": context_median(
            probability, context_available, "llm_probability"
        ),
        "llm_logit_margin": context_median(
            context_values("llm_logit_margin"),
            context_available,
            "llm_logit_margin",
        ),
        "llm_entropy": context_median(
            context_values("llm_entropy"), context_available, "llm_entropy"
        ),
        "llm_confidence": context_median(
            confidence, context_available, "llm_confidence"
        ),
        "llm_context_percentile": context_median(
            probability_percentile,
            context_available,
            "llm_context_percentile",
        ),
        "llm_gnn_percentile_signed_gap": context_median(
            signed_gap,
            context_available,
            "llm_gnn_percentile_signed_gap",
        ),
        "llm_gnn_percentile_absolute_gap": context_median(
            absolute_gap,
            context_available,
            "llm_gnn_percentile_absolute_gap",
        ),
        "llm_gnn_route_center_agreement": context_median(
            hard_agreement,
            context_available,
            "llm_gnn_route_center_agreement",
        ),
        "llm_logprob_0": context_median(
            context_values("llm_logprob_0"), context_available, "llm_logprob_0"
        ),
        "llm_logprob_1": context_median(
            context_values("llm_logprob_1"), context_available, "llm_logprob_1"
        ),
        "llm_token_mass": context_median(
            context_values("llm_token_mass"), context_available, "llm_token_mass"
        ),
        "llm_log_prompt_tokens": context_median(
            log_prompt_length, context_available, "llm_log_prompt_tokens"
        ),
    }
    return FeatureState(
        context_indices=context_indices,
        gnn_reference=gnn_reference,
        gnn_available_context_reference=gnn_available_context_reference,
        llm_probability_reference=probability_reference,
        fills=fills,
    )


def filled(
    values: np.ndarray,
    usable: np.ndarray,
    fill_value: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).copy()
    keep = usable & np.isfinite(values)
    values[~keep] = fill_value
    return values


def build_feature_variants(
    table: np.lib.npyio.NpzFile,
    split: str,
    state: FeatureState,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    gnn = table[f"{split}_gnn"].astype(np.float64)
    if not np.all(np.isfinite(gnn)):
        raise ValueError(f"{split}_gnn contains nonfinite values")
    gnn_percentile = empirical_percentile(state.gnn_reference, gnn)
    available = llm_availability(table, split)
    probability = table[f"{split}_llm_scores"].astype(np.float64)
    center = float(table["route_center"])

    last_delta = table[f"{split}_last_interaction_delta"].astype(
        np.float64, copy=True
    )
    has_past = np.isfinite(last_delta).astype(np.float64)
    # Preserve the established 11-column structural control exactly. This is
    # a fixed semantic sentinel, not a statistic estimated from any split.
    last_delta[~np.isfinite(last_delta)] = 1_000_000.0
    structural_columns = [
        gnn,
        gnn_percentile,
        np.abs(gnn - center),
        available.astype(np.float64),
        table[f"{split}_timestamps"].astype(np.float64),
        np.log1p(
            np.maximum(
                table[f"{split}_source_popularity"].astype(np.float64), 0.0
            )
        ),
        np.log1p(
            np.maximum(
                table[f"{split}_target_popularity"].astype(np.float64), 0.0
            )
        ),
        np.log1p(
            np.maximum(
                table[f"{split}_past_interactions"].astype(np.float64), 0.0
            )
        ),
        np.log1p(np.maximum(last_delta, 0.0)),
        has_past,
        np.log1p(
            np.maximum(
                table[f"{split}_common_neighbor"].astype(np.float64), 0.0
            )
        ),
    ]
    structural_names = [
        "gnn_raw",
        "gnn_validation_percentile",
        "gnn_distance_to_route_center",
        "llm_available",
        "timestamp",
        "log_source_popularity",
        "log_target_popularity",
        "log_past_interactions",
        "log_last_interaction_delta",
        "has_past_interaction",
        "log_common_neighbor",
    ]

    probability_filled = filled(
        probability, available, state.fills["llm_probability"]
    )
    logit_margin = filled(
        table[f"{split}_llm_logit_margin"],
        available,
        state.fills["llm_logit_margin"],
    )
    entropy = filled(
        table[f"{split}_llm_entropy"],
        available,
        state.fills["llm_entropy"],
    )
    confidence = np.full(len(gnn), state.fills["llm_confidence"], dtype=np.float64)
    confidence[available] = 2.0 * np.abs(probability[available] - 0.5)
    probability_percentile = np.full(
        len(gnn), state.fills["llm_context_percentile"], dtype=np.float64
    )
    probability_percentile[available] = empirical_percentile(
        state.llm_probability_reference,
        probability[available],
    )
    gnn_available_context_percentile = np.full(len(gnn), np.nan, dtype=np.float64)
    gnn_available_context_percentile[available] = empirical_percentile(
        state.gnn_available_context_reference,
        gnn[available],
    )
    signed_gap = np.full(
        len(gnn),
        state.fills["llm_gnn_percentile_signed_gap"],
        dtype=np.float64,
    )
    signed_gap[available] = (
        probability_percentile[available]
        - gnn_available_context_percentile[available]
    )
    absolute_gap = np.full(
        len(gnn),
        state.fills["llm_gnn_percentile_absolute_gap"],
        dtype=np.float64,
    )
    absolute_gap[available] = np.abs(signed_gap[available])
    hard_agreement = np.full(
        len(gnn),
        state.fills["llm_gnn_route_center_agreement"],
        dtype=np.float64,
    )
    hard_agreement[available] = (
        (probability[available] >= 0.5) == (gnn[available] >= center)
    ).astype(np.float64)

    engineered_columns = [
        probability_filled,
        logit_margin,
        entropy,
        confidence,
        probability_percentile,
        signed_gap,
        absolute_gap,
        hard_agreement,
    ]
    engineered_names = [
        "llm_probability_or_context_fill",
        "llm_logit_margin_or_context_fill",
        "llm_entropy_or_context_fill",
        "llm_confidence_or_context_fill",
        "llm_train_context_percentile_or_context_fill",
        "llm_minus_gnn_train_available_context_percentile_gap_or_context_fill",
        "abs_llm_minus_gnn_train_available_context_percentile_gap_or_context_fill",
        "llm_gnn_route_center_agreement_or_context_fill",
    ]

    logprob_0_raw = table[f"{split}_llm_logprob_0"].astype(np.float64)
    logprob_1_raw = table[f"{split}_llm_logprob_1"].astype(np.float64)
    logprob_0_present = available & np.isfinite(logprob_0_raw)
    logprob_1_present = available & np.isfinite(logprob_1_raw)
    logprob_0 = filled(
        logprob_0_raw, logprob_0_present, state.fills["llm_logprob_0"]
    )
    logprob_1 = filled(
        logprob_1_raw, logprob_1_present, state.fills["llm_logprob_1"]
    )
    token_mass = filled(
        table[f"{split}_llm_token_mass"],
        available,
        state.fills["llm_token_mass"],
    )
    prompt_tokens = table[f"{split}_llm_prompt_tokens"].astype(np.float64)
    log_prompt_tokens = np.log1p(np.maximum(prompt_tokens, 0.0))
    log_prompt_tokens = filled(
        log_prompt_tokens,
        available & np.isfinite(prompt_tokens),
        state.fills["llm_log_prompt_tokens"],
    )
    rich_columns = [
        logprob_0,
        logprob_0_present.astype(np.float64),
        logprob_1,
        logprob_1_present.astype(np.float64),
        token_mass,
        log_prompt_tokens,
    ]
    rich_names = [
        "llm_logprob_0_or_context_fill",
        "llm_logprob_0_present",
        "llm_logprob_1_or_context_fill",
        "llm_logprob_1_present",
        "llm_binary_token_mass_or_context_fill",
        "log1p_llm_prompt_tokens_or_context_fill",
    ]

    columns_by_variant = {
        "structural": structural_columns,
        "llm_scalar": structural_columns + [probability_filled],
        # Keep the complete GNN/LLM score-derived block, but remove timestamp,
        # node-popularity, interaction-history, recency, and common-neighbor
        # heuristics.  The first four structural columns are all score or
        # routing derived: raw GNN, validation percentile, distance to the
        # routing center, and LLM availability.
        "score_fusion_no_heuristics": structural_columns[:4]
        + engineered_columns,
        "score_engineered": structural_columns + engineered_columns,
        "rich_raw": structural_columns + engineered_columns + rich_columns,
    }
    names_by_variant = {
        "structural": structural_names,
        "llm_scalar": structural_names + [engineered_names[0]],
        "score_fusion_no_heuristics": structural_names[:4] + engineered_names,
        "score_engineered": structural_names + engineered_names,
        "rich_raw": structural_names + engineered_names + rich_names,
    }
    features: dict[str, np.ndarray] = {}
    for variant in VARIANTS:
        matrix = np.column_stack(columns_by_variant[variant])
        if not np.all(np.isfinite(matrix)):
            bad_columns = [
                names_by_variant[variant][column]
                for column in np.flatnonzero(~np.all(np.isfinite(matrix), axis=0))
            ]
            raise ValueError(
                f"{split}/{variant} contains nonfinite features: {bad_columns}"
            )
        features[variant] = matrix.astype(np.float32)
    return features, names_by_variant


def fit_predict(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float]]:
    order = np.random.default_rng(seed + 10_000).permutation(len(y_context))
    classifier = TabICLClassifier(
        n_estimators=args.n_estimators,
        batch_size=args.batch_size,
        kv_cache="repr",
        device=args.device,
        random_state=seed,
        verbose=False,
    )
    start = time.perf_counter()
    classifier.fit(X_context[order], y_context[order])
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    scores = classifier.predict_proba(X_test)[:, 1]
    predict_seconds = time.perf_counter() - start
    del classifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.asarray(scores, dtype=np.float64), {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
    }


def metric_delta(
    result: dict[str, float | int],
    reference: dict[str, float | int],
) -> dict[str, float]:
    return {
        key: float(result[key] - reference[key])
        for key in ("ap", "auc")
    }


def prediction_path(args: argparse.Namespace) -> Path:
    if args.predictions:
        return Path(args.predictions)
    output = Path(args.output)
    return output.with_name(f"{output.stem}_predictions.npz")


def main() -> None:
    args = parse_args()
    with np.load(args.table, allow_pickle=False) as table:
        require_table_schema(table, args.eval_split)
        heldout_positive_count = int(table["gnn_holdout_positive_count"])
        if (
            not args.allow_gnn_trained_context
            and args.positive_context > heldout_positive_count
        ):
            raise ValueError(
                f"Requested {args.positive_context} positive context rows, but only "
                f"{heldout_positive_count} were excluded from GNN optimization"
            )
        context_indices = most_recent_query_indices(table, args.positive_context)
        if "train_context_selection" in table.files:
            context_selection = str(table["train_context_selection"].item())
        else:
            context_selection = (
                "most_recent_unique_positive_queries_with_matched_negatives"
            )
        y_train = table["train_labels"].astype(np.int64)
        y_context = y_train[context_indices]
        y_eval = table[f"{args.eval_split}_labels"].astype(np.int64)
        if (
            np.count_nonzero(y_context == 1) != args.positive_context
            or np.count_nonzero(y_context == 0) != args.positive_context
        ):
            raise ValueError("Context is not exactly balanced")

        state = fit_feature_state(table, context_indices)
        train_features, feature_names = build_feature_variants(table, "train", state)
        eval_features, eval_feature_names = build_feature_variants(
            table, args.eval_split, state
        )
        if feature_names != eval_feature_names:
            raise AssertionError("Train/evaluation feature schemas differ")

        train_available = llm_availability(table, "train")
        eval_available = llm_availability(table, args.eval_split)
        eval_gnn = table[f"{args.eval_split}_gnn"].astype(np.float64)
        eval_llm = table[f"{args.eval_split}_llm_scores"].astype(np.float64)
        eval_query_ids = table[f"{args.eval_split}_query_ids"].astype(np.int64)
        gnn_global = metrics(y_eval, eval_gnn)
        gnn_batch = batch_mean_metrics(
            y_eval, eval_gnn, eval_query_ids, args.metric_batch_size
        )
        routed_baselines = {
            "rows": int(eval_available.sum()),
            "fraction": float(eval_available.mean()),
            "gnn": metrics(y_eval[eval_available], eval_gnn[eval_available]),
            "llm": metrics(y_eval[eval_available], eval_llm[eval_available]),
        }

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        predictions = prediction_path(args)
        predictions.parent.mkdir(parents=True, exist_ok=True)
        results: dict = {
            "table": str(args.table),
            "predictions": str(predictions),
            "evaluation_split": args.eval_split,
            "context_protocol": {
                "selection": context_selection,
                "positive_rows": args.positive_context,
                "negative_rows": args.positive_context,
                "total_rows": int(len(context_indices)),
                "unique_query_ids": int(
                    np.unique(table["train_query_ids"][context_indices]).size
                ),
                "gnn_heldout_positive_rows": heldout_positive_count,
                "gnn_trained_context_allowed": bool(
                    args.allow_gnn_trained_context
                ),
                "llm_available_rows": int(train_available[context_indices].sum()),
                "llm_available_fraction": float(
                    train_available[context_indices].mean()
                ),
                "minimum_timestamp": float(
                    table["train_timestamps"][context_indices].min()
                ),
                "maximum_timestamp": float(
                    table["train_timestamps"][context_indices].max()
                ),
            },
            "leakage_controls": {
                "llm_references": "selected available train-context rows only",
                "fill_statistics": "selected available train-context rows only",
                "structural_gnn_percentile_reference": (
                    "validation_gnn scores without labels"
                ),
                "llm_gnn_gap_percentile_references": (
                    "separate LLM and GNN score distributions over the same "
                    "selected available train-context rows"
                ),
                "structural_missing_delta_sentinel": 1_000_000.0,
                "test_statistics_or_labels_used_for_features": False,
                "availability_definition": "route AND finite llm probability",
            },
            "rows": {
                "train_table": int(len(y_train)),
                args.eval_split: int(len(y_eval)),
                f"{args.eval_split}_llm_available": int(eval_available.sum()),
            },
            "config": {
                "seeds": args.seeds,
                "variants": args.variants,
                "n_estimators": args.n_estimators,
                "batch_size": args.batch_size,
                "kv_cache": "repr",
                "metric_batch_size": args.metric_batch_size,
                "device": args.device,
            },
            "features": {
                variant: feature_names[variant] for variant in args.variants
            },
            "feature_fills": state.fills,
            "llm_probability_reference": {
                "rows": int(len(state.llm_probability_reference)),
                "minimum": float(state.llm_probability_reference.min()),
                "median": float(np.median(state.llm_probability_reference)),
                "maximum": float(state.llm_probability_reference.max()),
            },
            "baselines": {
                "gnn_global": gnn_global,
                "gnn_dtgb_batch_mean": gnn_batch,
                "routed_slice": routed_baselines,
            },
            "runs": [],
        }
        if "train_context_pool_positive_queries" in table.files:
            results["context_protocol"]["recent_pool_positive_queries"] = int(
                table["train_context_pool_positive_queries"].item()
            )
        prediction_payload: dict[str, np.ndarray] = {
            "evaluation_split": np.asarray(args.eval_split),
            "labels": y_eval,
            "gnn": eval_gnn,
            # Keep the prediction artifact fully finite. The availability mask
            # distinguishes the context-fitted neutral fill from real scores.
            "llm_probability_context_fill": eval_features["llm_scalar"][:, -1],
            "llm_available": eval_available,
            "route": table[f"{args.eval_split}_route"].astype(bool),
            "query_ids": eval_query_ids,
            "context_indices": context_indices.astype(np.int64),
            "context_query_ids": table["train_query_ids"][context_indices].astype(
                np.int64
            ),
            "context_labels": y_context,
            "context_llm_available": train_available[context_indices],
        }

        total_runs = len(args.seeds) * len(args.variants)
        run_number = 0
        for seed in args.seeds:
            seed_runs: dict[str, dict] = {}
            seed_scores: dict[str, np.ndarray] = {}
            for variant in args.variants:
                run_number += 1
                print(
                    f"[{run_number}/{total_runs}] seed={seed}, variant={variant}, "
                    f"features={eval_features[variant].shape[1]}",
                    flush=True,
                )
                scores, timing = fit_predict(
                    train_features[variant][context_indices],
                    y_context,
                    eval_features[variant],
                    seed=seed,
                    args=args,
                )
                global_result = metrics(y_eval, scores)
                batch_result = batch_mean_metrics(
                    y_eval, scores, eval_query_ids, args.metric_batch_size
                )
                run = {
                    "seed": int(seed),
                    "variant": variant,
                    "feature_count": int(eval_features[variant].shape[1]),
                    "global": global_result,
                    "dtgb_batch_mean": batch_result,
                    "routed_slice": metrics(
                        y_eval[eval_available], scores[eval_available]
                    ),
                    "delta_vs_gnn_global": metric_delta(global_result, gnn_global),
                    "delta_vs_gnn_dtgb_batch_mean": metric_delta(
                        batch_result, gnn_batch
                    ),
                    "timing": timing,
                }
                seed_runs[variant] = run
                seed_scores[variant] = scores
                prediction_payload[f"{variant}_seed_{seed}"] = scores.astype(
                    np.float32
                )
                print(
                    f"  global AP/AUC={global_result['ap']:.8f}/"
                    f"{global_result['auc']:.8f}; GNN delta AUC="
                    f"{run['delta_vs_gnn_global']['auc']:+.8f}; "
                    f"fit/predict={timing['fit_seconds']:.2f}/"
                    f"{timing['predict_seconds']:.2f}s",
                    flush=True,
                )

            references = {
                "structural": "structural",
                "llm_scalar": "llm_scalar",
                "score_engineered": "score_engineered",
            }
            for variant in args.variants:
                run = seed_runs[variant]
                for reference_name, reference_variant in references.items():
                    if reference_variant not in seed_runs:
                        continue
                    reference = seed_runs[reference_variant]
                    run[
                        f"delta_vs_same_seed_{reference_name}_global"
                    ] = metric_delta(run["global"], reference["global"])
                    run[
                        f"delta_vs_same_seed_{reference_name}_dtgb_batch_mean"
                    ] = metric_delta(
                        run["dtgb_batch_mean"], reference["dtgb_batch_mean"]
                    )
                results["runs"].append(run)
            output.write_text(
                json.dumps(results, indent=2) + "\n", encoding="utf-8"
            )

        aggregates: dict[str, dict] = {}
        for variant in args.variants:
            selected = [run for run in results["runs"] if run["variant"] == variant]
            variant_aggregate = {
                "global_ap": aggregate([run["global"]["ap"] for run in selected]),
                "global_auc": aggregate([run["global"]["auc"] for run in selected]),
                "dtgb_batch_mean_ap": aggregate(
                    [run["dtgb_batch_mean"]["ap"] for run in selected]
                ),
                "dtgb_batch_mean_auc": aggregate(
                    [run["dtgb_batch_mean"]["auc"] for run in selected]
                ),
                "routed_slice_ap": aggregate(
                    [run["routed_slice"]["ap"] for run in selected]
                ),
                "routed_slice_auc": aggregate(
                    [run["routed_slice"]["auc"] for run in selected]
                ),
                "delta_vs_gnn_global_ap": aggregate(
                    [run["delta_vs_gnn_global"]["ap"] for run in selected]
                ),
                "delta_vs_gnn_global_auc": aggregate(
                    [run["delta_vs_gnn_global"]["auc"] for run in selected]
                ),
                "delta_vs_gnn_dtgb_batch_mean_ap": aggregate(
                    [run["delta_vs_gnn_dtgb_batch_mean"]["ap"] for run in selected]
                ),
                "delta_vs_gnn_dtgb_batch_mean_auc": aggregate(
                    [run["delta_vs_gnn_dtgb_batch_mean"]["auc"] for run in selected]
                ),
                "fit_seconds": aggregate(
                    [run["timing"]["fit_seconds"] for run in selected]
                ),
                "predict_seconds": aggregate(
                    [run["timing"]["predict_seconds"] for run in selected]
                ),
            }
            optional_delta_fields = (
                "delta_vs_same_seed_structural_global",
                "delta_vs_same_seed_llm_scalar_global",
                "delta_vs_same_seed_score_engineered_global",
                "delta_vs_same_seed_structural_dtgb_batch_mean",
                "delta_vs_same_seed_llm_scalar_dtgb_batch_mean",
                "delta_vs_same_seed_score_engineered_dtgb_batch_mean",
            )
            for field in optional_delta_fields:
                if not all(field in run for run in selected):
                    continue
                for metric_name in ("ap", "auc"):
                    variant_aggregate[f"{field}_{metric_name}"] = aggregate(
                        [run[field][metric_name] for run in selected]
                    )
            aggregates[variant] = variant_aggregate
        results["aggregates"] = aggregates
        ensembles: dict[str, dict] = {}
        for variant in args.variants:
            ensemble_scores = np.mean(
                np.vstack(
                    [
                        prediction_payload[f"{variant}_seed_{seed}"]
                        for seed in args.seeds
                    ]
                ),
                axis=0,
            )
            ensemble_global = metrics(y_eval, ensemble_scores)
            ensemble_batch = batch_mean_metrics(
                y_eval,
                ensemble_scores,
                eval_query_ids,
                args.metric_batch_size,
            )
            ensembles[variant] = {
                "aggregation": "probability_mean",
                "seeds": [int(seed) for seed in args.seeds],
                "global": ensemble_global,
                "dtgb_batch_mean": ensemble_batch,
                "routed_slice": metrics(
                    y_eval[eval_available], ensemble_scores[eval_available]
                ),
                "delta_vs_gnn_global": metric_delta(
                    ensemble_global, gnn_global
                ),
                "delta_vs_gnn_dtgb_batch_mean": metric_delta(
                    ensemble_batch, gnn_batch
                ),
            }
            prediction_payload[f"{variant}_ensemble"] = ensemble_scores.astype(
                np.float32
            )
        results["ensembles"] = ensembles
        output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        np.savez_compressed(predictions, **prediction_payload)

    print(f"Saved summary: {output}", flush=True)
    print(f"Saved predictions: {predictions}", flush=True)


if __name__ == "__main__":
    main()
