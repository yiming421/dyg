#!/usr/bin/env python3
"""Join routed LLM diagnostics into the held-out-GNN TabICL table.

The score exports contain LLM outputs only for rows selected by the frozen
uncertainty band.  This builder keeps that distinction explicit: ``*_route``
means eligible for LLM routing, while ``*_llm_available`` means an LLM forward
pass was actually exported for the row.  Missing LLM features remain NaN.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_BASE_TABLE = "result/gdelt_tabicl_holdout_recent1k_gnn_full_table.npz"
DEFAULT_FULL_RESULT = (
    "result/gdelt_k10_tabicl_holdout_recent1k_full_score_export.json"
)
DEFAULT_FULL_DEBUG = (
    "result/gdelt_k10_tabicl_holdout_recent1k_full_selected.jsonl"
)
DEFAULT_TRAIN_RESULT = (
    "result/gdelt_k10_tabicl_holdout_recent1k_train_score_export.json"
)
DEFAULT_TRAIN_DEBUG = (
    "result/gdelt_k10_tabicl_holdout_recent1k_train_selected.jsonl"
)
DEFAULT_CHECKPOINT = (
    "saved_models/"
    "gdelt_semantic_mlp_minilm_raw384_gcn_late_concat_holdout_recent1k.pt"
)
DEFAULT_DATASET_ROOT = "../DyLink_Datasets/GDELT"
DEFAULT_OUTPUT = "result/gdelt_tabicl_holdout_recent1k_llm_full_table.npz"

LLM_DEBUG_FIELDS = {
    "llm_scores": "prediction_score",
    "llm_logprob_0": "binary_logprob_0",
    "llm_logprob_1": "binary_logprob_1",
    "llm_logit_margin": "binary_logit_margin",
    "llm_token_mass": "binary_token_mass",
    "llm_entropy": "binary_entropy",
    "llm_prompt_tokens": "prompt_token_count",
}

EXPORT_COMPATIBILITY_FIELDS = (
    "model",
    "backend",
    "dataset",
    "entity_name_mode",
    "dtgb_eval_batch_size",
    "base_seed",
    "history_preserve_recent_k",
    "no_cot_binary_score_mode",
    "include_edge_type",
    "include_edge_type_except_target",
    "semantic_mlp_checkpoint",
    "validation_calibration_num_samples",
    "validation_calibration_negative_ratio",
    "key_signal_mode",
    "key_signal_reference",
    "key_signal_fields",
    "hybrid_backbone",
    "hybrid_selection_mode",
    "hybrid_validation_target_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-table", default=DEFAULT_BASE_TABLE)
    parser.add_argument("--full-result-json", default=DEFAULT_FULL_RESULT)
    parser.add_argument("--full-debug-jsonl", default=DEFAULT_FULL_DEBUG)
    parser.add_argument("--train-result-json", default=DEFAULT_TRAIN_RESULT)
    parser.add_argument("--train-debug-jsonl", default=DEFAULT_TRAIN_DEBUG)
    parser.add_argument("--gnn-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--heldout-positive-count", type=int, default=1000)
    parser.add_argument("--gnn-score-atol", type=float, default=1e-5)
    parser.add_argument(
        "--route-center",
        type=float,
        default=None,
        help=(
            "Exact validation-calibrated center when known. If omitted and "
            "the result JSON does not retain it, use the route-band midpoint; "
            "routing itself depends only on low/high."
        ),
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sample_identity(
    query_id: Any,
    source_id: Any,
    target_id: Any,
    timestamp: Any,
    label: Any,
) -> tuple[int, int, int, int, int]:
    return (
        int(query_id),
        int(source_id),
        int(target_id),
        int(timestamp),
        int(label),
    )


def row_identity(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return sample_identity(
        row["query_id"],
        row["source_id"],
        row["target_id"],
        row["timestamp"],
        row["label"],
    )


def table_identity_index(
    payload: dict[str, np.ndarray],
    split: str,
) -> dict[tuple[int, int, int, int, int], int]:
    keys = (
        f"{split}_query_ids",
        f"{split}_source_ids",
        f"{split}_target_ids",
        f"{split}_timestamps",
        f"{split}_labels",
    )
    missing = [key for key in keys if key not in payload]
    if missing:
        raise KeyError(f"Base table lacks {split} identity arrays: {missing}")

    arrays = [payload[key] for key in keys]
    lengths = {len(values) for values in arrays}
    if len(lengths) != 1:
        raise ValueError(f"Mismatched {split} identity-array lengths: {lengths}")

    index: dict[tuple[int, int, int, int, int], int] = {}
    for row_idx, values in enumerate(zip(*arrays)):
        key = sample_identity(*values)
        if key in index:
            raise ValueError(f"Duplicate base-table {split} identity: {key}")
        index[key] = row_idx
    return index


def load_trial_hybrid(path: str, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    trials = result.get("trials")
    if not isinstance(trials, list) or len(trials) != 1:
        raise ValueError(
            f"{label} score export must contain exactly one trial; got "
            f"{None if not isinstance(trials, list) else len(trials)}"
        )
    hybrid = trials[0].get("hybrid")
    if not isinstance(hybrid, dict) or not hybrid.get("enabled", False):
        raise ValueError(f"{label} score export does not contain an enabled hybrid run")
    selection = hybrid.get("selection")
    if not isinstance(selection, dict):
        raise ValueError(f"{label} score export lacks trial hybrid selection metadata")
    return result, hybrid


def verify_export_compatibility(
    full_result: dict[str, Any],
    train_result: dict[str, Any],
) -> None:
    mismatches = {
        field: (full_result.get(field), train_result.get(field))
        for field in EXPORT_COMPATIBILITY_FIELDS
        if full_result.get(field) != train_result.get(field)
    }
    if mismatches:
        raise ValueError(
            "Full/test and train LLM exports use incompatible configurations: "
            f"{mismatches}"
        )
    print(
        "Verified matching train/test LLM export configuration: "
        f"model={full_result.get('model')}, "
        f"binary_score={full_result.get('no_cot_binary_score_mode')}",
        flush=True,
    )


def extract_full_route_band(
    result: dict[str, Any],
    hybrid: dict[str, Any],
    route_center_override: float | None,
) -> tuple[float, float, float, float, str]:
    selection = hybrid["selection"]
    try:
        low = float(selection["low_threshold"])
        high = float(selection["high_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Full-result hybrid selection lacks a numeric route band") from exc

    center_value = route_center_override
    center_source = "cli_override" if center_value is not None else "selection"
    if center_value is None:
        center_value = selection.get("center_threshold")
    if center_value is None:
        # Compatibility with result files that retain the calibration beside
        # rather than inside selection metadata.
        center_source = "calibration"
        candidates = (
            hybrid.get("validation_sampled_uncertainty_band"),
            result.get("validation_sampled_hybrid_band"),
        )
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            center_value = candidate.get("center_threshold", candidate.get("center"))
            if center_value is not None:
                break
    if center_value is None:
        # Older score-export summaries retain the exact route endpoints but
        # omit the balanced-threshold center printed to the run log. The
        # center is descriptive for this table: low/high alone define every
        # route identity, so retain an explicit, reproducible midpoint.
        center_value = 0.5 * (low + high)
        center_source = "band_midpoint"
    center = float(center_value)

    target_fraction = selection.get(
        "validation_requested_target_fraction",
        result.get("hybrid_validation_target_fraction"),
    )
    if target_fraction is None:
        raise ValueError("Full-result score export lacks the requested route fraction")
    target_fraction = float(target_fraction)

    if not all(np.isfinite(value) for value in (center, low, high, target_fraction)):
        raise ValueError(
            f"Non-finite route metadata: center={center}, low={low}, "
            f"high={high}, target_fraction={target_fraction}"
        )
    if not low < high:
        raise ValueError(f"Invalid route band: low={low}, high={high}")
    if not low <= center < high:
        raise ValueError(
            f"Calibrated center is outside its route band: "
            f"center={center}, band=[{low}, {high})"
        )
    if not 0.0 <= target_fraction <= 1.0:
        raise ValueError(f"Invalid target route fraction: {target_fraction}")
    return center, low, high, target_fraction, center_source


def nullable_float(row: dict[str, Any], field: str) -> float:
    if field not in row:
        raise KeyError(f"Routed debug row lacks required field {field!r}")
    value = row[field]
    if value is None:
        return np.nan
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"Non-finite routed debug field {field}: {value!r}")
    return numeric


def join_routed_debug(
    *,
    payload: dict[str, np.ndarray],
    split: str,
    debug_jsonl: str,
    route_mask: np.ndarray,
    expected_selected: int,
    expected_debug_split: str,
    gnn_score_atol: float,
    allow_unselected: bool = False,
) -> dict[str, np.ndarray]:
    index = table_identity_index(payload, split)
    num_rows = len(route_mask)
    gnn = np.asarray(payload[f"{split}_gnn"], dtype=np.float64)
    if len(gnn) != num_rows or not np.all(np.isfinite(gnn)):
        raise ValueError(f"Invalid {split} GNN score array")

    columns = {
        output_name: np.full(num_rows, np.nan, dtype=np.float64)
        for output_name in LLM_DEBUG_FIELDS
    }
    available = np.zeros(num_rows, dtype=bool)
    seen = np.zeros(num_rows, dtype=bool)
    max_gnn_abs_diff = 0.0
    debug_count = 0

    with open(debug_jsonl, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("split")) != expected_debug_split:
                raise ValueError(
                    f"Unexpected split in {debug_jsonl}:{line_number}: "
                    f"{row.get('split')!r}, expected {expected_debug_split!r}"
                )
            key = row_identity(row)
            row_idx = index.get(key)
            if row_idx is None:
                raise ValueError(
                    f"Debug identity is absent from {split} table at "
                    f"{debug_jsonl}:{line_number}: {key}"
                )
            if seen[row_idx]:
                raise ValueError(
                    f"Duplicate routed debug identity at {debug_jsonl}:"
                    f"{line_number}: {key}"
                )
            seen[row_idx] = True
            if not route_mask[row_idx]:
                if allow_unselected:
                    continue
                raise ValueError(
                    f"Exported LLM row is outside the full-result route band at "
                    f"{debug_jsonl}:{line_number}: {key}"
                )

            score = nullable_float(row, "prediction_score")
            if not np.isfinite(score):
                raise ValueError(
                    f"Missing/non-finite LLM prediction score at "
                    f"{debug_jsonl}:{line_number}: {key}"
                )
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"LLM prediction score outside [0, 1]: {score}")

            debug_gnn = nullable_float(row, "semantic_mlp_score")
            if not np.isfinite(debug_gnn):
                raise ValueError(f"Missing semantic_mlp_score for routed row {key}")
            gnn_abs_diff = abs(float(gnn[row_idx]) - debug_gnn)
            max_gnn_abs_diff = max(max_gnn_abs_diff, gnn_abs_diff)
            if gnn_abs_diff > gnn_score_atol:
                raise ValueError(
                    f"GNN score mismatch for {key}: table={gnn[row_idx]:.12g}, "
                    f"debug={debug_gnn:.12g}, abs_diff={gnn_abs_diff:.3g}"
                )

            for output_name, debug_field in LLM_DEBUG_FIELDS.items():
                columns[output_name][row_idx] = nullable_float(row, debug_field)
            # The normalized score is the required, finite primary LLM feature.
            columns["llm_scores"][row_idx] = score
            available[row_idx] = True
            debug_count += 1

    if debug_count != expected_selected:
        raise ValueError(
            f"{split} routed count disagrees with its result JSON: "
            f"debug={debug_count}, result={expected_selected}"
        )
    if not np.array_equal(available, route_mask):
        missing_count = int(np.count_nonzero(route_mask & ~available))
        extra_count = int(np.count_nonzero(available & ~route_mask))
        raise ValueError(
            f"{split} routed debug identities do not exactly equal the route mask: "
            f"missing={missing_count}, extra={extra_count}"
        )

    result_columns = {
        f"{split}_{name}": values for name, values in columns.items()
    }
    result_columns[f"{split}_llm_available"] = available
    print(
        f"Joined {split} routed LLM diagnostics: rows={debug_count:,}, "
        f"GNN max abs diff={max_gnn_abs_diff:.3g}",
        flush=True,
    )
    return result_columns


def empty_llm_columns(split: str, num_rows: int) -> dict[str, np.ndarray]:
    columns = {
        f"{split}_{name}": np.full(num_rows, np.nan, dtype=np.float64)
        for name in LLM_DEBUG_FIELDS
    }
    columns[f"{split}_llm_available"] = np.zeros(num_rows, dtype=bool)
    return columns


def load_checkpoint_metadata(path: str) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to verify the GNN held-out train identities"
        ) from exc
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint payload in {path}")
    metadata = checkpoint.get("train_holdout_metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Checkpoint {path} lacks train_holdout_metadata")
    return {
        "count": int(checkpoint.get("train_holdout_recent_edges", -1)),
        "metadata": metadata,
    }


def verify_recent_positive_holdout(
    *,
    payload: dict[str, np.ndarray],
    checkpoint_path: str,
    dataset_root: str,
    expected_count: int,
) -> None:
    if expected_count <= 0:
        raise ValueError("--heldout-positive-count must be positive")

    labels = np.asarray(payload["train_labels"], dtype=np.int64)
    query_ids = np.asarray(payload["train_query_ids"], dtype=np.int64)
    timestamps = np.asarray(payload["train_timestamps"], dtype=np.float64)
    source_ids = np.asarray(payload["train_source_ids"], dtype=np.int64)
    target_ids = np.asarray(payload["train_target_ids"], dtype=np.int64)
    positive_indices = np.flatnonzero(labels == 1)
    positive_query_ids = query_ids[positive_indices]
    if np.unique(positive_query_ids).size != len(positive_indices):
        raise ValueError(
            "Train context contains duplicate positive query IDs; positive rows "
            "must be unique for this trial"
        )
    if len(positive_indices) < expected_count:
        raise ValueError(
            f"Only {len(positive_indices)} positive train rows are available; "
            f"cannot select the latest {expected_count}"
        )

    order = np.lexsort(
        (positive_query_ids, timestamps[positive_indices])
    )
    selected_indices = positive_indices[order[-expected_count:]]
    selected_query_ids = query_ids[selected_indices]
    if np.unique(selected_query_ids).size != expected_count:
        raise ValueError(
            f"Latest-positive context is not unique: expected={expected_count}, "
            f"unique={np.unique(selected_query_ids).size}"
        )

    checkpoint_info = load_checkpoint_metadata(checkpoint_path)
    if checkpoint_info["count"] != expected_count:
        raise ValueError(
            f"Checkpoint held-out count={checkpoint_info['count']} does not match "
            f"requested context count={expected_count}"
        )
    metadata = checkpoint_info["metadata"]
    if int(metadata.get("count", -1)) != expected_count:
        raise ValueError(
            f"Checkpoint holdout metadata count={metadata.get('count')} does not "
            f"match {expected_count}"
        )

    edge_path = Path(dataset_root) / "edge_list.csv"
    edges = pd.read_csv(edge_path)
    required_columns = {"u", "i", "ts"}
    if not required_columns.issubset(edges.columns):
        raise ValueError(
            f"{edge_path} lacks required columns {sorted(required_columns)}"
        )
    if not isinstance(edges.index, pd.RangeIndex) or edges.index.start != 0:
        raise ValueError("GDELT edge table must use its default integer row index")

    edge_src = edges["u"].to_numpy(dtype=np.int64)
    edge_dst = edges["i"].to_numpy(dtype=np.int64)
    raw_times = edges["ts"].to_numpy(dtype=np.float64)
    dtgb_times = np.floor_divide(raw_times.astype(np.int64), 15).astype(np.float64)
    val_time = float(np.quantile(dtgb_times, 0.70))

    node_set = set(edge_src).union(set(edge_dst))
    test_node_set = set(edge_src[dtgb_times > val_time]).union(
        set(edge_dst[dtgb_times > val_time])
    )
    heldout_nodes = set(
        random.Random(2020).sample(
            list(test_node_set),
            int(0.1 * len(node_set)),
        )
    )
    heldout_node_array = np.fromiter(heldout_nodes, dtype=np.int64)
    observed_mask = ~(
        np.isin(edge_src, heldout_node_array)
        | np.isin(edge_dst, heldout_node_array)
    )
    gnn_train_mask = (dtgb_times <= val_time) & observed_mask
    gnn_train_query_ids = np.flatnonzero(gnn_train_mask).astype(np.int64)
    if len(gnn_train_query_ids) <= expected_count:
        raise ValueError("Reconstructed GNN train split is too small for its holdout")
    reconstructed_holdout = gnn_train_query_ids[-expected_count:]
    reconstructed_remaining = gnn_train_query_ids[:-expected_count]

    selected_set = set(selected_query_ids.tolist())
    holdout_set = set(reconstructed_holdout.tolist())
    if selected_set != holdout_set:
        raise ValueError(
            "Latest train-context positives are not exactly the checkpoint's "
            f"reconstructed held-out GNN tail: only_context="
            f"{len(selected_set - holdout_set)}, only_holdout="
            f"{len(holdout_set - selected_set)}"
        )
    overlap = selected_set.intersection(reconstructed_remaining.tolist())
    if overlap:
        raise ValueError(
            f"Latest positive context overlaps GNN optimization rows: {len(overlap)}"
        )

    for row_idx in selected_indices:
        query_id = int(query_ids[row_idx])
        expected_identity = (
            int(edge_src[query_id]),
            int(edge_dst[query_id]),
            int(raw_times[query_id]),
        )
        table_identity = (
            int(source_ids[row_idx]),
            int(target_ids[row_idx]),
            int(timestamps[row_idx]),
        )
        if table_identity != expected_identity:
            raise ValueError(
                f"Train positive query {query_id} does not match edge_list.csv: "
                f"table={table_identity}, edge={expected_identity}"
            )

    heldout_src = edge_src[reconstructed_holdout]
    heldout_dst = edge_dst[reconstructed_holdout]
    heldout_times = dtgb_times[reconstructed_holdout]
    expected_first = [
        int(heldout_src[0]),
        int(heldout_dst[0]),
        float(heldout_times[0]),
    ]
    expected_last = [
        int(heldout_src[-1]),
        int(heldout_dst[-1]),
        float(heldout_times[-1]),
    ]
    if metadata.get("first_edge") != expected_first:
        raise ValueError(
            f"Checkpoint first held-out edge mismatch: metadata="
            f"{metadata.get('first_edge')}, reconstructed={expected_first}"
        )
    if metadata.get("last_edge") != expected_last:
        raise ValueError(
            f"Checkpoint last held-out edge mismatch: metadata="
            f"{metadata.get('last_edge')}, reconstructed={expected_last}"
        )
    if not np.isclose(float(metadata["min_time"]), float(heldout_times.min())):
        raise ValueError("Checkpoint held-out minimum timestamp mismatch")
    if not np.isclose(float(metadata["max_time"]), float(heldout_times.max())):
        raise ValueError("Checkpoint held-out maximum timestamp mismatch")

    print(
        "Verified train context isolation: "
        f"all positives unique={len(positive_indices):,}, latest={expected_count:,}, "
        "overlap with GNN optimization=0",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.gnn_score_atol < 0.0:
        raise ValueError("--gnn-score-atol must be nonnegative")

    with np.load(args.base_table, allow_pickle=False) as base:
        payload = {key: base[key] for key in base.files}

    full_result, full_hybrid = load_trial_hybrid(
        args.full_result_json, "full"
    )
    train_result, train_hybrid = load_trial_hybrid(
        args.train_result_json, "train"
    )
    verify_export_compatibility(full_result, train_result)
    center, low, high, target_fraction, center_source = extract_full_route_band(
        full_result,
        full_hybrid,
        args.route_center,
    )

    train_selection = train_hybrid["selection"]
    for name, expected in (
        ("low_threshold", low),
        ("high_threshold", high),
    ):
        value = train_selection.get(name)
        if value is None or not np.isclose(
            float(value), expected, rtol=0.0, atol=args.gnn_score_atol
        ):
            raise ValueError(
                f"Train/full route calibration mismatch for {name}: "
                f"train={value}, full={expected}"
            )

    payload.update(
        {
            "route_center": np.asarray(center, dtype=np.float64),
            "route_low": np.asarray(low, dtype=np.float64),
            "route_high": np.asarray(high, dtype=np.float64),
            "route_center_source": np.asarray(center_source),
            "target_route_fraction": np.asarray(
                target_fraction, dtype=np.float64
            ),
        }
    )
    route_masks: dict[str, np.ndarray] = {}
    for split in ("train", "validation", "test"):
        scores = np.asarray(payload[f"{split}_gnn"], dtype=np.float64)
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"Base table has non-finite {split} GNN scores")
        route = (scores >= low) & (scores < high)
        payload[f"{split}_route"] = route
        route_masks[split] = route

    payload.update(
        join_routed_debug(
            payload=payload,
            split="train",
            debug_jsonl=args.train_debug_jsonl,
            route_mask=route_masks["train"],
            expected_selected=int(train_selection["selected_count"]),
            expected_debug_split="train",
            gnn_score_atol=args.gnn_score_atol,
        )
    )
    payload.update(
        join_routed_debug(
            payload=payload,
            split="test",
            debug_jsonl=args.full_debug_jsonl,
            route_mask=route_masks["test"],
            expected_selected=int(full_hybrid["selection"]["selected_count"]),
            expected_debug_split="transductive",
            gnn_score_atol=args.gnn_score_atol,
        )
    )
    # No current-checkpoint validation LLM export is part of this experiment.
    # Explicitly erase the stale score column inherited from the base table.
    payload.update(empty_llm_columns("validation", len(route_masks["validation"])))

    verify_recent_positive_holdout(
        payload=payload,
        checkpoint_path=args.gnn_checkpoint,
        dataset_root=args.dataset_root,
        expected_count=args.heldout_positive_count,
    )

    payload.update(
        {
            "llm_full_result_json": np.asarray(args.full_result_json),
            "llm_full_debug_jsonl": np.asarray(args.full_debug_jsonl),
            "llm_train_result_json": np.asarray(args.train_result_json),
            "llm_train_debug_jsonl": np.asarray(args.train_debug_jsonl),
            "llm_diagnostic_schema_version": np.asarray(1, dtype=np.int64),
            "llm_model": np.asarray(str(full_result.get("model", ""))),
            "gnn_holdout_positive_count": np.asarray(
                args.heldout_positive_count, dtype=np.int64
            ),
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    print(
        f"Saved LLM-enriched TabICL table: {output}; "
        f"route center={center:.8g} ({center_source}), "
        f"band=[{low:.8g}, {high:.8g}), "
        f"train/test available={payload['train_llm_available'].sum():,}/"
        f"{payload['test_llm_available'].sum():,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
