#!/usr/bin/env python3
"""Score an exported prompt JSONL with the repository's vLLM binary scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.modules.llm_lp.eval import (
    _resolve_stop_token_ids,
    _score_no_cot_binary_forced,
    load_model_and_tokenizer,
)
from experiments.modules.prediction_metrics import compute_prediction_metrics


IDENTITY_FIELDS = (
    "row_index",
    "variant",
    "query_id",
    "source_id",
    "target_id",
    "relation_id",
    "timestamp",
    "dtgb_timestamp",
    "label",
    "eval_split",
    "prompt_sha256",
    "reference_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prompt_dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dtgb_eval_batch_size", type=int, default=256)
    parser.add_argument(
        "--variants",
        nargs="+",
        help="Score only these variant names (default: all rows).",
    )
    parser.add_argument(
        "--query_pairs_per_split",
        type=int,
        default=0,
        help="Fixed random screening sample per eval split; 0 keeps every pair.",
    )
    parser.add_argument("--selection_seed", type=int, default=42)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "prompt" not in row or "label" not in row:
                raise ValueError(f"Missing prompt/label at {path}:{line_number}")
            expected_hash = row.get("prompt_sha256")
            if expected_hash is not None:
                actual_hash = hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest()
                if actual_hash != expected_hash:
                    raise ValueError(f"Prompt hash mismatch at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No prompt rows found in {path}")
    return rows


def select_rows(
    rows: list[dict],
    *,
    variants: list[str] | None,
    query_pairs_per_split: int,
    selection_seed: int,
) -> tuple[list[dict], dict]:
    if variants:
        requested = list(dict.fromkeys(str(variant) for variant in variants))
        available = {str(row.get("variant")) for row in rows}
        missing = sorted(set(requested).difference(available))
        if missing:
            raise ValueError(f"Requested variants absent from dataset: {missing}")
        requested_set = set(requested)
        rows = [row for row in rows if str(row.get("variant")) in requested_set]
    else:
        requested = sorted({str(row.get("variant")) for row in rows})

    selected_pair_keys = None
    if query_pairs_per_split < 0:
        raise ValueError("query_pairs_per_split must be non-negative")
    if query_pairs_per_split:
        pairs_by_split: dict[str, list[tuple[str, int]]] = {}
        for row in rows:
            key = (str(row["eval_split"]), int(row["query_id"]))
            pairs_by_split.setdefault(key[0], []).append(key)
        rng = np.random.default_rng(int(selection_seed))
        selected_pair_keys = set()
        for split, repeated_keys in sorted(pairs_by_split.items()):
            keys = sorted(set(repeated_keys))
            count = min(int(query_pairs_per_split), len(keys))
            indices = sorted(rng.choice(len(keys), size=count, replace=False).tolist())
            selected_pair_keys.update(keys[index] for index in indices)
        rows = [
            row
            for row in rows
            if (str(row["eval_split"]), int(row["query_id"])) in selected_pair_keys
        ]

    counts: dict[tuple[str, int, str], int] = {}
    for row in rows:
        key = (str(row["eval_split"]), int(row["query_id"]), str(row.get("variant")))
        counts[key] = counts.get(key, 0) + 1
    malformed = [key for key, count in counts.items() if count != 2]
    if malformed:
        raise RuntimeError(f"Expected two candidates per split/query/variant: {malformed[:5]}")

    selection = {
        "variants": requested,
        "query_pairs_per_split": int(query_pairs_per_split),
        "selection_seed": int(selection_seed),
        "selected_query_pairs": int(
            len({(str(row["eval_split"]), int(row["query_id"])) for row in rows})
        ),
    }
    return rows, selection


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    prompt_path = Path(args.prompt_dataset).resolve()
    output_path = Path(args.output).resolve()
    rows = load_rows(prompt_path)
    rows, selection = select_rows(
        rows,
        variants=args.variants,
        query_pairs_per_split=args.query_pairs_per_split,
        selection_seed=args.selection_seed,
    )
    prompts = [str(row["prompt"]) for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)

    started = time.time()
    llm, tokenizer = load_model_and_tokenizer(
        args.model_path,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_utilization=args.gpu_memory_utilization,
    )
    loaded_seconds = time.time() - started
    stop_token_ids = _resolve_stop_token_ids(tokenizer)
    inference_started = time.time()
    results, token_usage, _, _ = _score_no_cot_binary_forced(
        llm=llm,
        tokenizer=tokenizer,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )
    inference_seconds = time.time() - inference_started
    scores = np.asarray([result["score"] for result in results], dtype=np.float64)
    metrics = compute_prediction_metrics(
        scores,
        labels,
        dtgb_eval_batch_size=args.dtgb_eval_batch_size,
    )

    prediction_rows = []
    for source, result in zip(rows, results):
        item = {field: source.get(field) for field in IDENTITY_FIELDS}
        item.update(
            {
                key: result.get(key)
                for key in (
                    "score",
                    "binary_logprob_0",
                    "binary_logprob_1",
                    "binary_logit_margin",
                    "binary_token_mass",
                    "binary_entropy",
                    "parse_method",
                    "prompt_token_count",
                )
            }
        )
        prediction_rows.append(item)

    payload = {
        "model_path": str(Path(args.model_path).resolve()),
        "prompt_dataset": str(prompt_path),
        "prompt_dataset_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        "scoring": "repository_vllm_forced_binary_next_token_probability",
        "max_model_len": int(args.max_model_len),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "num_samples": int(len(rows)),
        "num_positive": int(labels.sum()),
        "num_negative": int((1 - labels).sum()),
        "selection": selection,
        "model_load_seconds": float(loaded_seconds),
        "inference_seconds": float(inference_seconds),
        "rows_per_second": float(len(rows) / inference_seconds),
        "token_usage": jsonable(token_usage),
        "metrics": jsonable(metrics),
        "predictions": prediction_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(str(output_path) + ".tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary_path, output_path)
    print(json.dumps({key: payload[key] for key in (
        "num_samples", "model_load_seconds", "inference_seconds", "rows_per_second", "metrics"
    )}, indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
