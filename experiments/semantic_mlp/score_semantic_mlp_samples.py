#!/usr/bin/env python3
import argparse
import json
import os
import sys

import numpy as np

_EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_EXPERIMENTS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.modules.semantic_mlp.backbone import SemanticMLPHybridBackbone
from experiments.modules.llm_lp.training_protocol import sample_history_scope


def main():
    parser = argparse.ArgumentParser(description="Score semantic MLP samples from JSONL.")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--embedding_npz", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--dtgb_eval_batch_size", type=int, default=256)
    parser.add_argument(
        "--semantic_mlp_source_init_override",
        default="auto",
        choices=["auto", "raw", "history_mean"],
    )
    parser.add_argument(
        "--semantic_mlp_temporal_mode",
        default="timestamp_rebuild",
        choices=["timestamp_rebuild", "rolling_replay"],
    )
    args = parser.parse_args()

    payload = np.load(args.embedding_npz)
    samples = []
    with open(args.input) as handle:
        for line in handle:
            if line.strip():
                samples.append(json.loads(line))

    scorer = SemanticMLPHybridBackbone.from_checkpoint(
        dataset_name=args.dataset_name,
        checkpoint_path=args.checkpoint_path,
        embeddings=payload["embeddings"],
        entity_ids_sorted=payload["entity_ids_sorted"].tolist(),
        device="cuda",
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        eval_positive_batch_size=args.dtgb_eval_batch_size,
        source_init_override=args.semantic_mlp_source_init_override,
        temporal_mode=args.semantic_mlp_temporal_mode,
        history_scope=sample_history_scope(samples),
    )
    scores = scorer.score_samples(samples)

    with open(args.output, "w") as handle:
        for sample, score in zip(samples, scores):
            handle.write(json.dumps({"idx": int(sample["idx"]), "score": float(score)}) + "\n")


if __name__ == "__main__":
    main()
