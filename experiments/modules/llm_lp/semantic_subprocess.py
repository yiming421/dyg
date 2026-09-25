import json
import os
import subprocess
import tempfile

import numpy as np


_EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SubprocessSemanticMLPScorer:
    score_field = "semantic_mlp_score"
    score_label = "Semantic MLP"
    scorer_type = "subprocess"

    def __init__(
        self,
        *,
        python_path,
        dataset_name,
        checkpoint_path,
        embeddings,
        entity_ids_sorted,
        val_ratio,
        test_ratio,
        eval_positive_batch_size,
        source_init_override,
        temporal_mode,
    ):
        if embeddings is None:
            raise ValueError("Subprocess semantic scorer requires preloaded embeddings.")
        self.python_path = str(python_path)
        self.dataset_name = str(dataset_name)
        self.checkpoint_path = str(checkpoint_path)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.entity_ids_sorted = np.asarray(entity_ids_sorted, dtype=np.int64)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.eval_positive_batch_size = int(eval_positive_batch_size)
        self.source_init_override = str(source_init_override)
        self.temporal_mode = str(temporal_mode)

    def annotate_samples(self, samples, *, score_field=None):
        if not samples:
            return samples
        target_field = str(score_field or self.score_field)
        script_path = os.path.join(
            _EXPERIMENTS_DIR,
            "semantic_mlp",
            "score_semantic_mlp_samples.py",
        )
        with tempfile.TemporaryDirectory(prefix="semantic_mlp_score_") as tmp_dir:
            input_path = os.path.join(tmp_dir, "samples.jsonl")
            output_path = os.path.join(tmp_dir, "scores.jsonl")
            embedding_path = os.path.join(tmp_dir, "embeddings.npz")
            np.savez(
                embedding_path,
                embeddings=self.embeddings,
                entity_ids_sorted=self.entity_ids_sorted,
            )
            with open(input_path, "w") as handle:
                for idx, sample in enumerate(samples):
                    row = {
                        "idx": int(idx),
                        "source_id": int(sample["source_id"]),
                        "target_id": int(sample["target_id"]),
                        "timestamp": float(sample.get("dtgb_timestamp", sample["timestamp"])),
                        "query_id": int(sample["query_id"]),
                    }
                    if "dtgb_timestamp" in sample:
                        row["dtgb_timestamp"] = float(sample["dtgb_timestamp"])
                    for key in ("graph_history_scope", "training_history_spec"):
                        if key in sample:
                            row[key] = sample[key]
                    handle.write(json.dumps(row) + "\n")

            cmd = [
                self.python_path,
                script_path,
                "--dataset_name",
                self.dataset_name,
                "--checkpoint_path",
                self.checkpoint_path,
                "--embedding_npz",
                embedding_path,
                "--input",
                input_path,
                "--output",
                output_path,
                "--val_ratio",
                str(self.val_ratio),
                "--test_ratio",
                str(self.test_ratio),
                "--dtgb_eval_batch_size",
                str(self.eval_positive_batch_size),
                "--semantic_mlp_source_init_override",
                self.source_init_override,
                "--semantic_mlp_temporal_mode",
                self.temporal_mode,
            ]
            subprocess.run(cmd, check=True)

            scores = [None] * len(samples)
            with open(output_path) as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    scores[int(row["idx"])] = float(row["score"])
            missing = sum(score is None for score in scores)
            if missing:
                raise RuntimeError(f"Semantic subprocess returned {missing} missing scores.")
            for sample, score in zip(samples, scores):
                sample[target_field] = float(score)
        return samples
