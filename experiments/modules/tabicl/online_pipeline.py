"""Out-of-process TabICL routing and fusion for the main LLM evaluator.

The vLLM and TabICL environments intentionally remain separate.  This module
serializes an already-computed labeled train-context slice plus unlabeled
deployment features, invokes the shared TabICL router/fusion programs, and
returns their predictions to the main hybrid pipeline.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from utils.seed_runs import DEFAULT_SEEDS


_REPO_ROOT = Path(__file__).resolve().parents[3]
_ROUTER_SCRIPT = _REPO_ROOT / "experiments/tabicl/routing/train_tabicl_utility_router.py"
_FUSION_SCRIPT = (
    _REPO_ROOT / "experiments/tabicl/evaluation/evaluate_tabicl_train_llm_fusion.py"
)
_ALIGNMENT_VARIANT = "score_fusion_no_heuristics"
_ALIGNMENT_VARIANTS = frozenset({_ALIGNMENT_VARIANT, "score_engineered"})
_ALIGNMENT_SEEDS = DEFAULT_SEEDS


def _normalize_alignment_variant(value: str) -> str:
    variant = str(value).strip().lower()
    if variant not in _ALIGNMENT_VARIANTS:
        expected = ", ".join(sorted(_ALIGNMENT_VARIANTS))
        raise ValueError(
            f"Unsupported TabICL alignment variant {value!r}; expected {expected}"
        )
    return variant


def _values(
    samples: list[dict[str, Any]],
    field: str,
    *,
    default: float,
) -> np.ndarray:
    return np.asarray(
        [
            default if sample.get(field) is None else float(sample[field])
            for sample in samples
        ],
        dtype=np.float64,
    )


def _integers(samples: list[dict[str, Any]], field: str) -> np.ndarray:
    missing = [
        index for index, sample in enumerate(samples) if sample.get(field) is None
    ]
    if missing:
        raise ValueError(f"Router samples lack {field!r} at rows {missing[:5]}")
    return np.asarray([int(sample[field]) for sample in samples], dtype=np.int64)


def _binary_diagnostics(probability: np.ndarray) -> dict[str, np.ndarray]:
    probability = np.asarray(probability, dtype=np.float64)
    if probability.ndim != 1 or not np.all(np.isfinite(probability)):
        raise ValueError(
            "LLM probabilities must be finite and one-dimensional"
        )
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("LLM probabilities must lie in [0, 1]")
    epsilon = 1e-6
    clipped = np.clip(probability, epsilon, 1.0 - epsilon)
    logprob_0 = np.log1p(-clipped)
    logprob_1 = np.log(clipped)
    return {
        "llm_scores": probability,
        "llm_logprob_0": logprob_0,
        "llm_logprob_1": logprob_1,
        "llm_logit_margin": logprob_1 - logprob_0,
        "llm_token_mass": np.ones(len(probability), dtype=np.float64),
        "llm_entropy": -(clipped * logprob_1 + (1.0 - clipped) * logprob_0),
        # The established score-only alignment does not consume prompt length,
        # but a finite placeholder keeps the common table schema valid.
        "llm_prompt_tokens": np.zeros(len(probability), dtype=np.float64),
    }


def _structural_columns(
    samples: list[dict[str, Any]],
    *,
    backbone_score_field: str,
) -> dict[str, np.ndarray]:
    return {
        "gnn": _values(samples, backbone_score_field, default=np.nan),
        "labels": _integers(samples, "label"),
        "timestamps": _values(samples, "timestamp", default=np.nan),
        "source_popularity": _values(samples, "source_popularity_raw", default=0.0),
        "target_popularity": _values(samples, "target_popularity_raw", default=0.0),
        "past_interactions": _values(samples, "num_past_interactions_raw", default=0.0),
        "last_interaction_delta": _values(samples, "last_interaction_delta", default=np.nan),
        "common_neighbor": _values(samples, "common_neighbor_score", default=0.0),
        "source_ids": _integers(samples, "source_id"),
        "target_ids": _integers(samples, "target_id"),
        "query_ids": _integers(samples, "query_id"),
    }


def write_online_router_inputs(
    *,
    support_samples: list[dict[str, Any]],
    support_llm_scores: np.ndarray,
    deployment_samples: list[dict[str, Any]],
    backbone_score_field: str,
    route_center: float,
    budget_fraction: float,
    table_path: Path,
    support_debug_path: Path,
    context_selection: str = "most_recent",
) -> dict[str, int | float]:
    """Write the shared router table/debug contract from live evaluator rows."""
    if not support_samples or not deployment_samples:
        raise ValueError(
            "Online TabICL routing requires nonempty support and deployment rows"
        )
    if not 0.0 < float(budget_fraction) < 1.0:
        raise ValueError("Online TabICL budget_fraction must be in (0, 1)")
    if not np.isfinite(float(route_center)):
        raise ValueError("Online TabICL route_center must be finite")
    support_llm_scores = np.asarray(support_llm_scores, dtype=np.float64)
    if len(support_llm_scores) != len(support_samples):
        raise ValueError("Validation samples and LLM scores have different lengths")

    support = _structural_columns(
        support_samples, backbone_score_field=backbone_score_field
    )
    deployment = _structural_columns(
        deployment_samples, backbone_score_field=backbone_score_field
    )
    for label, columns in (("support", support), ("deployment", deployment)):
        if not np.all(np.isfinite(columns["gnn"])):
            raise ValueError(f"{label} backbone scores contain NaN or infinity")
        if not np.all(np.isfinite(columns["timestamps"])):
            raise ValueError(f"{label} timestamps contain NaN or infinity")

    diagnostics = _binary_diagnostics(support_llm_scores)
    route_count = max(1, int(round(float(budget_fraction) * len(support_samples))))
    distance = np.abs(support["gnn"] - float(route_center))
    row_indices = np.arange(len(distance), dtype=np.int64)
    uncertainty_order = np.lexsort((row_indices, distance))
    support_route = np.zeros(len(support_samples), dtype=bool)
    support_route[uncertainty_order[:route_count]] = True

    payload: dict[str, np.ndarray] = {
        "route_center": np.asarray(float(route_center), dtype=np.float64),
        "validation_gnn": support["gnn"].astype(np.float64),
        # The online protocol deliberately allows these train rows to have
        # participated in GNN optimization. Keep the legacy schema field
        # truthful and record the policy explicitly for downstream auditing.
        "gnn_holdout_positive_count": np.asarray(0, dtype=np.int64),
        "train_context_gnn_seen": np.asarray(True),
        "train_context_selection": np.asarray(str(context_selection)),
        "gnn_reference_split": np.asarray("train_context"),
    }
    for field, values in support.items():
        payload[f"train_{field}"] = np.asarray(values).copy()
    payload["train_route"] = support_route
    for field, values in diagnostics.items():
        sparse = np.full(len(values), np.nan, dtype=np.float64)
        sparse[support_route] = values[support_route]
        payload[f"train_{field}"] = sparse
    payload["train_llm_available"] = support_route.copy()

    for field, values in deployment.items():
        payload[f"test_{field}"] = np.asarray(values).copy()
    payload["test_route"] = np.zeros(len(deployment_samples), dtype=bool)
    for field in diagnostics:
        payload[f"test_{field}"] = np.full(
            len(deployment_samples), np.nan, dtype=np.float64
        )
    payload["test_llm_available"] = np.zeros(len(deployment_samples), dtype=bool)

    table_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(table_path, **payload)

    support_debug_path.parent.mkdir(parents=True, exist_ok=True)
    with support_debug_path.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(support_samples):
            row = {
                "split": "train",
                "query_id": int(sample["query_id"]),
                "source_id": int(sample["source_id"]),
                "target_id": int(sample["target_id"]),
                "timestamp": int(sample["timestamp"]),
                "label": int(sample["label"]),
                "semantic_mlp_score": float(support["gnn"][index]),
                "prediction_score": float(diagnostics["llm_scores"][index]),
                "binary_logprob_0": float(diagnostics["llm_logprob_0"][index]),
                "binary_logprob_1": float(diagnostics["llm_logprob_1"][index]),
                "binary_logit_margin": float(diagnostics["llm_logit_margin"][index]),
                "binary_token_mass": float(diagnostics["llm_token_mass"][index]),
                "binary_entropy": float(diagnostics["llm_entropy"][index]),
                "prompt_token_count": 0.0,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "support_rows": int(len(support_samples)),
        "support_positive_rows": int(np.count_nonzero(support["labels"] == 1)),
        "support_negative_rows": int(np.count_nonzero(support["labels"] == 0)),
        "support_initial_route_rows": int(support_route.sum()),
        "deployment_rows": int(len(deployment_samples)),
        "route_center": float(route_center),
    }


def prepare_online_tabicl_fusion_table(
    *,
    eval_split: str,
    trial: int,
    support_samples: list[dict[str, Any]],
    support_llm_scores: np.ndarray,
    deployment_samples: list[dict[str, Any]],
    backbone_score_field: str,
    route_center: float,
    budget_fraction: float,
    artifact_dir: str | Path,
    context_selection: str = "most_recent",
) -> tuple[str, dict[str, Any]]:
    """Prepare TabICL fusion inputs when routing is supplied externally.

    The support-side availability policy intentionally matches the automatic
    TabICL-router path.  Only deployment selection is delegated to the caller
    (for example, a validation-calibrated middle band).
    """
    artifact_root = Path(artifact_dir)
    run_root = artifact_root / f"trial{int(trial) + 1}_{eval_split}"
    table_path = run_root / "fusion_table.npz"
    support_debug_path = run_root / "support_allcall.jsonl"
    input_summary = write_online_router_inputs(
        support_samples=support_samples,
        support_llm_scores=support_llm_scores,
        deployment_samples=deployment_samples,
        backbone_score_field=backbone_score_field,
        route_center=route_center,
        budget_fraction=budget_fraction,
        table_path=table_path,
        support_debug_path=support_debug_path,
        context_selection=context_selection,
    )
    metadata = {
        **input_summary,
        "table": str(table_path),
        "support_debug": str(support_debug_path),
        "deployment_selection_external": True,
    }
    return str(table_path), metadata


def run_online_tabicl_router(
    *,
    dataset_name: str,
    eval_split: str,
    trial: int,
    support_samples: list[dict[str, Any]],
    support_llm_scores: np.ndarray,
    deployment_samples: list[dict[str, Any]],
    backbone_score_field: str,
    route_center: float,
    budget_fraction: float,
    tabicl_python: str,
    tabicl_device: str,
    cuda_visible_devices: str | None,
    artifact_dir: str | Path,
    folds: int,
    n_estimators: int,
    batch_size: int,
    alignment_variant: str = _ALIGNMENT_VARIANT,
    context_selection: str = "most_recent",
) -> tuple[list[int], dict[str, Any]]:
    """Fit the shared router in the TabICL environment and return selected rows."""
    alignment_variant = _normalize_alignment_variant(alignment_variant)
    artifact_root = Path(artifact_dir)
    run_root = artifact_root / f"trial{int(trial) + 1}_{eval_split}"
    table_path = run_root / "router_table.npz"
    support_debug_path = run_root / "support_allcall.jsonl"
    result_path = run_root / "router_training.json"
    predictions_path = run_root / "router_predictions.npz"
    checkpoint_path = run_root / "route.joblib"
    log_path = run_root / "router_fit.log"

    input_summary = write_online_router_inputs(
        support_samples=support_samples,
        support_llm_scores=support_llm_scores,
        deployment_samples=deployment_samples,
        backbone_score_field=backbone_score_field,
        route_center=route_center,
        budget_fraction=budget_fraction,
        table_path=table_path,
        support_debug_path=support_debug_path,
        context_selection=context_selection,
    )
    positive_context = int(input_summary["support_positive_rows"])
    if positive_context < int(folds):
        raise ValueError(
            f"TabICL support has {positive_context} query pairs for {folds} folds"
        )

    python_path = Path(tabicl_python).expanduser()
    if not python_path.is_file():
        raise FileNotFoundError(f"TabICL Python does not exist: {python_path}")
    command = [
        str(python_path),
        "-u",
        str(_ROUTER_SCRIPT),
        "--dataset-name",
        str(dataset_name),
        "--table",
        str(table_path),
        "--support-debug-jsonl",
        str(support_debug_path),
        "--calibration-split",
        "train",
        "--support-debug-split",
        "train",
        "--deployment-split",
        "test",
        "--positive-context",
        str(positive_context),
        "--budget-fraction",
        str(float(budget_fraction)),
        "--folds",
        str(int(folds)),
        "--n-estimators",
        str(int(n_estimators)),
        "--batch-size",
        str(int(batch_size)),
        "--alignment-variant",
        alignment_variant,
        "--device",
        str(tabicl_device),
        "--output",
        str(result_path),
        "--predictions",
        str(predictions_path),
        "--route-checkpoint",
        str(checkpoint_path),
    ]
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{_REPO_ROOT}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(_REPO_ROOT)
    )
    if cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")

    run_root.mkdir(parents=True, exist_ok=True)
    print(
        "Automatic TabICL router fit: "
        f"support={len(support_samples):,}, deployment={len(deployment_samples):,}, "
        f"budget={float(budget_fraction):.3f}, device={tabicl_device}",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(_REPO_ROOT),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            print(f"[TabICL router] {line}", end="", flush=True)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Automatic TabICL router failed with exit {return_code}; log={log_path}"
        )
    for required in (result_path, predictions_path, checkpoint_path):
        if not required.is_file():
            raise RuntimeError(f"TabICL router did not create {required}")

    with np.load(predictions_path, allow_pickle=False) as predictions:
        priorities = predictions["deployment_router_priority"].astype(np.float64)
        selected = predictions["deployment_selected_indices"].astype(np.int64)
        query_ids = predictions["deployment_query_ids"].astype(np.int64)
        target_ids = predictions["deployment_target_ids"].astype(np.int64)
    expected_query_ids = _integers(deployment_samples, "query_id")
    expected_target_ids = _integers(deployment_samples, "target_id")
    if not (
        np.array_equal(query_ids, expected_query_ids)
        and np.array_equal(target_ids, expected_target_ids)
    ):
        raise RuntimeError("TabICL router response identities do not match deployment rows")
    if priorities.shape != (len(deployment_samples),) or not np.all(
        np.isfinite(priorities)
    ):
        raise RuntimeError("TabICL router returned invalid deployment priorities")
    if np.any(selected < 0) or np.any(selected >= len(deployment_samples)):
        raise RuntimeError("TabICL router returned an out-of-range selected row")
    if np.unique(selected).size != len(selected):
        raise RuntimeError("TabICL router returned duplicate selected rows")
    expected_selected_count = max(
        1, round(float(budget_fraction) * len(deployment_samples))
    )
    if len(selected) != expected_selected_count:
        raise RuntimeError(
            "TabICL router selected an unexpected number of rows: "
            f"{len(selected)} != {expected_selected_count}"
        )

    selected_indices = sorted(int(index) for index in selected.tolist())
    selection = {
        "enabled": True,
        "selection_mode": "tabicl_router",
        "selection_rule": "global_top_train_fitted_tabicl_auc_utility",
        "requested_target_fraction": float(budget_fraction),
        "requested_target_count": int(expected_selected_count),
        "selected_count": int(len(selected_indices)),
        "num_candidates": int(len(deployment_samples)),
        "selected_fraction_realized": float(len(selected_indices))
        / float(len(deployment_samples)),
        "selected_sample_indices": selected_indices,
        "selected_router_score_min": float(np.min(priorities[selected])),
        "selected_router_score_max": float(np.max(priorities[selected])),
        "all_router_score_min": float(np.min(priorities)),
        "all_router_score_max": float(np.max(priorities)),
        "router_checkpoint": str(checkpoint_path),
        "router_training_result": str(result_path),
        "router_predictions": str(predictions_path),
        "router_log": str(log_path),
        "router_table": str(table_path),
        "router_support_debug": str(support_debug_path),
        "router_support_rows": int(len(support_samples)),
        "router_context_split": "train",
        "router_context_selection": str(context_selection),
        "router_context_gnn_seen": True,
        "router_alignment_variant": alignment_variant,
        "router_uses_evaluation_labels": False,
        "route_center": float(route_center),
    }
    return selected_indices, selection


def _write_routed_deployment_scores(
    *,
    table_path: Path,
    selected_indices: list[int] | np.ndarray,
    selected_llm_scores: np.ndarray,
) -> int:
    """Atomically add the actually routed deployment LLM scores to a table."""
    with np.load(table_path, allow_pickle=False) as loaded:
        payload = {key: loaded[key] for key in loaded.files}
    rows = len(payload["test_labels"])
    selected = np.asarray(selected_indices, dtype=np.int64)
    scores = np.asarray(selected_llm_scores, dtype=np.float64)
    if selected.ndim != 1 or scores.shape != selected.shape:
        raise ValueError("Selected TabICL fusion indices/scores must be aligned vectors")
    if np.any(selected < 0) or np.any(selected >= rows):
        raise ValueError("Selected TabICL fusion index is out of range")
    if np.unique(selected).size != len(selected):
        raise ValueError("Selected TabICL fusion indices contain duplicates")

    route = np.zeros(rows, dtype=bool)
    route[selected] = True
    payload["test_route"] = route
    payload["test_llm_available"] = route.copy()
    diagnostics = _binary_diagnostics(scores)
    for field, values in diagnostics.items():
        routed = np.full(rows, np.nan, dtype=np.float64)
        routed[selected] = values
        payload[f"test_{field}"] = routed

    temporary_path = table_path.with_name(f"{table_path.stem}.tmp.npz")
    np.savez_compressed(temporary_path, **payload)
    os.replace(temporary_path, table_path)
    return int(route.sum())


def run_online_tabicl_fusion(
    *,
    table_path: str | Path,
    selected_indices: list[int] | np.ndarray,
    selected_llm_scores: np.ndarray,
    deployment_samples: list[dict[str, Any]],
    tabicl_python: str,
    tabicl_device: str,
    cuda_visible_devices: str | None,
    positive_context: int,
    n_estimators: int,
    batch_size: int,
    metric_batch_size: int,
    alignment_variant: str = _ALIGNMENT_VARIANT,
    context_selection: str = "most_recent",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit the final train-context TabICL corrector and return full scores."""
    alignment_variant = _normalize_alignment_variant(alignment_variant)
    table_path = Path(table_path)
    if not table_path.is_file():
        raise FileNotFoundError(f"TabICL router table does not exist: {table_path}")
    routed_count = _write_routed_deployment_scores(
        table_path=table_path,
        selected_indices=selected_indices,
        selected_llm_scores=selected_llm_scores,
    )

    run_root = table_path.parent
    output_path = run_root / "tabicl_fusion.json"
    predictions_path = run_root / "tabicl_fusion_predictions.npz"
    log_path = run_root / "tabicl_fusion.log"
    python_path = Path(tabicl_python).expanduser()
    if not python_path.is_file():
        raise FileNotFoundError(f"TabICL Python does not exist: {python_path}")
    command = [
        str(python_path),
        "-u",
        str(_FUSION_SCRIPT),
        "--table",
        str(table_path),
        "--output",
        str(output_path),
        "--predictions",
        str(predictions_path),
        "--eval-split",
        "test",
        "--variants",
        alignment_variant,
        "--positive-context",
        str(int(positive_context)),
        "--seeds",
        *(str(seed) for seed in _ALIGNMENT_SEEDS),
        "--n-estimators",
        str(int(n_estimators)),
        "--batch-size",
        str(int(batch_size)),
        "--metric-batch-size",
        str(int(metric_batch_size)),
        "--device",
        str(tabicl_device),
        "--allow-gnn-trained-context",
    ]
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{_REPO_ROOT}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(_REPO_ROOT)
    )
    if cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(
        "Automatic TabICL final fusion: "
        f"context={int(positive_context) * 2:,}, routed={routed_count:,}, "
        f"deployment={len(deployment_samples):,}, device={tabicl_device}",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(_REPO_ROOT),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            print(f"[TabICL fusion] {line}", end="", flush=True)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Automatic TabICL fusion failed with exit {return_code}; log={log_path}"
        )
    for required in (output_path, predictions_path):
        if not required.is_file():
            raise RuntimeError(f"TabICL fusion did not create {required}")

    with np.load(predictions_path, allow_pickle=False) as predictions:
        expected_query_ids = _integers(deployment_samples, "query_id")
        if not np.array_equal(
            predictions["query_ids"].astype(np.int64), expected_query_ids
        ):
            raise RuntimeError("TabICL fusion response query identities do not match")
        seed_scores = [
            predictions[f"{alignment_variant}_seed_{seed}"].astype(np.float64)
            for seed in _ALIGNMENT_SEEDS
        ]
    fused_scores = np.mean(np.vstack(seed_scores), axis=0)
    if fused_scores.shape != (len(deployment_samples),) or not np.all(
        np.isfinite(fused_scores)
    ):
        raise RuntimeError("TabICL fusion returned invalid full-split scores")
    if np.any((fused_scores < 0.0) | (fused_scores > 1.0)):
        raise RuntimeError("TabICL fusion probabilities lie outside [0, 1]")

    metadata = {
        "mode": "tabicl",
        "variant": alignment_variant,
        "context_split": "train",
        "context_selection": str(context_selection),
        "context_positive_queries": int(positive_context),
        "context_rows": int(positive_context) * 2,
        "context_gnn_seen": True,
        "evaluation_labels_used_for_fit": False,
        "routed_rows": int(routed_count),
        "seeds": [int(seed) for seed in _ALIGNMENT_SEEDS],
        "seed_aggregation": "probability_mean",
        "table": str(table_path),
        "result": str(output_path),
        "predictions": str(predictions_path),
        "log": str(log_path),
    }
    return fused_scores, metadata


__all__ = [
    "prepare_online_tabicl_fusion_table",
    "run_online_tabicl_fusion",
    "run_online_tabicl_router",
    "write_online_router_inputs",
]
