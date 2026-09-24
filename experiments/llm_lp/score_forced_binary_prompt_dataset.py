#!/usr/bin/env python3
"""Score an exported link-prediction prompt JSONL with a Transformers CausalLM."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

_EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_EXPERIMENTS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.modules.llm_lp.peft import (
    PromptResponseCollator,
    _build_binary_token_groups,
    _score_binary_next_token_batch,
)
from experiments.modules.prediction_metrics import compute_prediction_metrics


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Score canonical exported prompts by P(next token=1) vs P(next token=0)."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prompt_dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--dtgb_eval_batch_size", type=int, default=256)
    parser.add_argument("--negative_ratio", type=int, default=1)
    parser.add_argument("--include_predictions", action="store_true")
    return parser


def _load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "prompt" not in row or "label" not in row:
                raise ValueError(f"Missing prompt/label at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No prompt rows found in {path}")
    return rows


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    args = build_arg_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.max_length < 1:
        raise ValueError("--max_length must be >= 1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this prompt-scoring experiment.")

    rows = _load_rows(args.prompt_dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).cuda()
    model.eval()
    model.config.use_cache = False

    collator = PromptResponseCollator(tokenizer)
    binary_token_groups = _build_binary_token_groups(tokenizer)
    device = next(model.parameters()).device
    predictions = []
    labels = []
    truncated = 0
    token_lengths = []
    started = time.time()

    for start in tqdm(range(0, len(rows), args.batch_size), desc="Forced-binary scoring"):
        batch_rows = rows[start : start + args.batch_size]
        examples = []
        for row in batch_rows:
            prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
            token_lengths.append(len(prompt_ids))
            if len(prompt_ids) > args.max_length:
                prompt_ids = prompt_ids[-args.max_length :]
                truncated += 1
            examples.append(
                {
                    "input_ids": prompt_ids,
                    "attention_mask": [1] * len(prompt_ids),
                    "labels": [-100] * len(prompt_ids),
                }
            )
        batch = collator(examples)
        predictions.extend(
            _score_binary_next_token_batch(
                model,
                batch,
                device=device,
                binary_token_groups=binary_token_groups,
            )
        )
        labels.extend(int(row["label"]) for row in batch_rows)

    metrics = compute_prediction_metrics(
        np.asarray(predictions, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        dtgb_eval_batch_size=args.dtgb_eval_batch_size,
    )
    payload = {
        "model_path": os.path.abspath(args.model_path),
        "prompt_dataset": os.path.abspath(args.prompt_dataset),
        "scoring": "forced_binary_next_token_probability",
        "dtype": str(dtype).replace("torch.", ""),
        "batch_size": int(args.batch_size),
        "max_length": int(args.max_length),
        "num_samples": len(rows),
        "num_positive": int(sum(labels)),
        "num_negative": int(len(labels) - sum(labels)),
        "num_truncated": int(truncated),
        "avg_prompt_tokens": float(np.mean(token_lengths)),
        "max_prompt_tokens": int(max(token_lengths)),
        "runtime_seconds": float(time.time() - started),
        "metrics": _jsonable(metrics),
    }
    if args.include_predictions:
        payload["predictions"] = [float(value) for value in predictions]
        payload["labels"] = labels

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, output_path)
    print(json.dumps(payload["metrics"], indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
