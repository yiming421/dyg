#!/usr/bin/env python3
"""Reproduce the frozen GIN recipes without an experiment-tracking account."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.gin.protocol import (  # noqa: E402
    TRAINER, TrialMetrics, adapt_source, install_hooks, sha256_file,
    validation_evaluator, verify_sources,
)

from utils.seed_runs import DEFAULT_SEEDS

DATASETS = ("GDELT", "ICEWS1819", "Enron", "Googlemap_CT")
SEEDS = DEFAULT_SEEDS


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def recipe(dataset):
    spec = json.loads((ROOT / "configs/gin" / f"{dataset}.json").read_text(encoding="utf-8"))
    config = spec["config"]
    required = {"dataset_name": dataset, "learnable_mp_type": "gin",
                "use_heuristic_features": True, "use_mplp_exact_features": False,
                "dtgb_eval_batch_size": 256, "val_num_negatives": 1,
                "eval_num_negatives": 1, "report_test": True}
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("Recipe does not satisfy the GIN main-table protocol")
    if spec["seeds"] != list(SEEDS):
        raise ValueError("Expected the five distinct training seeds 42, 43, 44, 45, 46")
    return spec


def training_arguments(spec, seed, data_root, embedding, checkpoint, gpu, smoke_test=False):
    config = {**spec["config"], "seed": seed}
    history = config.pop("history_truncation_mode")
    config.pop("report_test")
    if history == "K":
        config["smooth_time_window"] = 1e15
    elif history == "W":
        config["smooth_endpoint_topk_recent"] = 0
    else:
        raise ValueError("History mode must be K or W")
    if smoke_test:
        config["epochs"] = 1
    config.update(gpu=gpu, embedding_cache=str(embedding),
                  embedding_entity_name_mode=spec["embedding_entity_name_mode"],
                  entity_text_path=str(data_root / spec["dataset"] / "entity_text.csv"),
                  checkpoint_path=str(checkpoint))
    argv = []
    for key, value in config.items():
        argv.extend(["--" + key, str(value).lower() if isinstance(value, bool) else str(value)])
    return argv


class LocalRecord:
    """The small log/summary interface used by the frozen protocol adapter."""
    def __init__(self, directory, config):
        self.directory = directory
        self.config = config
        self.summary = {}

    def log(self, payload, step=None, commit=True):
        with (self.directory / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"step": step, **payload}, allow_nan=False) + "\n")


def inspect_inputs(spec, data_root, embedding):
    paths = {"embedding_cache": embedding,
             "edge_list.csv": data_root / spec["dataset"] / "edge_list.csv",
             "entity_text.csv": data_root / spec["dataset"] / "entity_text.csv"}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required inputs missing: " + ", ".join(missing))
    records = {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in paths.items()}
    mismatches = [key for key, value in records.items()
                  if value["sha256"] != spec["reference_input_sha256"][key]]
    return records, mismatches


def run_worker(args):
    if args.gpu < 0:
        # Numba heuristics probe CUDA during import, independently of PyTorch's
        # model device. Hide CUDA before either import for an actual CPU run.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch
    from experiments.modules.semantic_mlp import runtime
    from utils import DataLoader

    spec = recipe(args.dataset)
    seed = args.seeds[0]
    directory = args.output_root / args.dataset / f"seed-{seed}"
    pins = verify_sources(ROOT, ROOT / "configs/gin/source-pins.json")
    inputs, mismatches = inspect_inputs(spec, args.data_root, args.embedding_cache)
    if mismatches and not args.allow_input_mismatch:
        raise ValueError("Inputs differ from the audited run: " + ", ".join(mismatches)
                         + ". Check the files; use --allow-input-mismatch only for a labelled new experiment.")
    embeddings = np.load(args.embedding_cache, mmap_mode="r", allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[1] != 1024:
        raise ValueError("GIN requires an E5-large-v2 cache with shape (entities, 1024)")
    if any(not np.isfinite(embeddings[start:start + 8192]).all()
           for start in range(0, len(embeddings), 8192)):
        raise ValueError("Embedding cache contains nonfinite values")
    directory.mkdir(parents=True, exist_ok=False)
    checkpoint = directory / "best.pt"
    config = {**spec["config"], "seed": seed}
    if args.smoke_test:
        config["epochs"] = 1
    argv = training_arguments(spec, seed, args.data_root, args.embedding_cache,
                              checkpoint, args.gpu, args.smoke_test)
    manifest = {"dataset": args.dataset, "seed": seed, "config": config,
                "purpose": "smoke_test" if args.smoke_test else "reproduction",
                "reference_inputs_match": not mismatches, "input_mismatches": mismatches,
                "inputs": inputs, "sources": pins, "argv": argv,
                "runner_sha256": sha256_file(Path(__file__)),
                "protocol_sha256": sha256_file(ROOT / "experiments/gin/protocol.py"),
                "recipe_sha256": sha256_file(ROOT / "configs/gin" / f"{args.dataset}.json"),
                "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                "cuda": torch.version.cuda, "gpu": args.gpu,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "test_evaluated": False}
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "effective_config.json", config)
    local = LocalRecord(directory, config)
    metrics = TrialMetrics(local)
    finalized = False

    def finalize(trainer_args, test_metrics=None, new_node_metrics=None):
        nonlocal finalized
        if not checkpoint.is_file() or metrics.best is None:
            raise RuntimeError("Training ended without a validation-selected checkpoint")
        verify_sources(ROOT, ROOT / "configs/gin/source-pins.json")
        if any(sha256_file(Path(item["path"])) != item["sha256"] for item in inputs.values()):
            raise RuntimeError("An input file changed during training")
        report = metrics.log_test(test_metrics, new_node_metrics)
        fingerprints = {}
        for split, expected in spec["reference_evaluation_fingerprints"].items():
            fingerprints[split] = {key: local.summary[f"protocol/{split}_{key}"] for key in expected}
        if not mismatches and fingerprints != spec["reference_evaluation_fingerprints"]:
            raise RuntimeError("Query order or evaluation negatives differ from the reference protocol")
        result = {"status": "completed", "dataset": args.dataset, "seed": seed,
                  "purpose": manifest["purpose"], "reference_inputs_match": not mismatches,
                  "best": metrics.best, "checkpoint": str(checkpoint),
                  "checkpoint_sha256": sha256_file(checkpoint),
                  "validation_evaluations": metrics.evaluations,
                  "test_evaluated": metrics.test_reported, "test": report,
                  "evaluation_fingerprints": fingerprints}
        write_json(directory / "result.json", result)
        manifest["test_evaluated"] = metrics.test_reported
        write_json(directory / "manifest.json", manifest)
        write_json(directory / "summary.json", local.summary)
        finalized = True

    try:
        adapted = adapt_source((ROOT / TRAINER).read_text(encoding="utf-8"), report_test=True)
        # Direct the historical loader to the explicit input directory without
        # changing its split seed, timestamp conversion, ordering, or RNG calls.
        DataLoader._resolve_dataset_root = lambda dataset: str(args.data_root / dataset)
        namespace = {"__name__": "_dtgb_gin", "__file__": str(ROOT / TRAINER),
                     "_gin_metrics": metrics, "_gin_finalize": finalize,
                     "_gin_begin_test": metrics.begin_test}
        exec(compile(adapted, str(ROOT / TRAINER), "exec"), namespace)
        namespace["evaluate_split"] = validation_evaluator(runtime, torch, metrics, directory, report_test=True)
        install_hooks(namespace, runtime, torch, metrics, directory)
        previous_argv = sys.argv
        try:
            sys.argv = [str(ROOT / TRAINER), *argv]
            namespace["main"]()
        finally:
            sys.argv = previous_argv
        if not finalized:
            raise RuntimeError("Trainer returned without a completed result")
    except BaseException as exc:
        write_json(directory / "failure.json", {"status": "failed", "error_type": type(exc).__name__,
                                                "message": str(exc)})
        raise


def run(args):
    spec = recipe(args.dataset)
    for name in ("data_root", "embedding_cache", "output_root"):
        setattr(args, name, getattr(args, name).resolve())
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Each training seed may appear only once")
    if args.smoke_test and not args.allow_input_mismatch:
        raise ValueError("--smoke-test requires --allow-input-mismatch and produces no table result")
    jobs = []
    for seed in args.seeds:
        directory = args.output_root / args.dataset / f"seed-{seed}"
        if directory.exists():
            raise FileExistsError(f"Refusing to overwrite {directory}; choose another --output-root")
        jobs.append({"seed": seed, "output": str(directory),
                     "trainer_argv": training_arguments(spec, seed, args.data_root, args.embedding_cache,
                                                        directory / "best.pt", args.gpu, args.smoke_test)})
    if args.dry_run:
        print(json.dumps({"dataset": args.dataset, "jobs": jobs,
                          "test_policy": "validation_auc_best_checkpoint_once_per_test_split",
                          "tracking_account_required": False, "training_started": False}, indent=2))
        return
    if args.worker:
        if len(args.seeds) != 1:
            raise ValueError("A worker requires exactly one seed")
        run_worker(args)
        return
    # A fresh interpreter per seed avoids retained monkeypatches or GPU state.
    for seed in args.seeds:
        command = [sys.executable, str(Path(__file__).resolve()), "run", "--dataset", args.dataset,
                   "--data-root", str(args.data_root), "--embedding-cache", str(args.embedding_cache),
                   "--output-root", str(args.output_root), "--gpu", str(args.gpu),
                   "--seeds", str(seed), "--worker"]
        for flag in ("allow_input_mismatch", "smoke_test"):
            if getattr(args, flag):
                command.append("--" + flag.replace("_", "-"))
        subprocess.run(command, check=True, cwd=ROOT)


def summarize(args):
    output = {"metric": "DTGB batch-mean AUC, canonical metric groups of 256",
              "std": "sample standard deviation (ddof=1)", "datasets": {}}
    for dataset in args.datasets:
        records = []
        comparison = None
        for seed in args.seeds:
            directory = args.output_root / dataset / f"seed-{seed}"
            result_path = directory / "result.json"
            if not result_path.is_file() or (directory / "failure.json").exists():
                if args.allow_partial:
                    continue
                raise ValueError(f"Missing successful result for {dataset}, seed {seed}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if (result.get("status") != "completed" or not result.get("test_evaluated")
                    or result.get("seed") != seed or result.get("dataset") != dataset
                    or result.get("purpose") != "reproduction"):
                raise ValueError(f"Invalid or smoke-test result: {result_path}")
            if not result.get("reference_inputs_match") and not args.allow_input_mismatch:
                raise ValueError(f"Non-reference inputs in {result_path}; use --allow-input-mismatch to label the comparison")
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            shared = {"config": {key: value for key, value in manifest["config"].items() if key != "seed"},
                      "inputs": {key: value["sha256"] for key, value in manifest["inputs"].items()},
                      "sources": manifest["sources"], "protocol": manifest["protocol_sha256"],
                      "runner": manifest["runner_sha256"],
                      "recipe": manifest["recipe_sha256"],
                      "evaluation_fingerprints": result["evaluation_fingerprints"]}
            if comparison is not None and shared != comparison:
                raise ValueError(f"Cannot aggregate incompatible configurations, inputs, or protocols: {result_path}")
            comparison = shared
            values = {split: float(result["test"][f"test/{split}_auc"])
                      for split in ("transductive", "inductive")}
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in values.values()):
                raise ValueError(f"Invalid AUC in {result_path}")
            records.append({"seed": seed, "reference_inputs_match": result["reference_inputs_match"], **values})
        if not records:
            raise ValueError(f"No completed seeds for {dataset}")
        entry = {"n": len(records), "seeds": [r["seed"] for r in records], "records": records,
                 "target_seeds": list(SEEDS),
                 "missing_seeds": [seed for seed in SEEDS if seed not in {r["seed"] for r in records}],
                 "five_seed_complete": len(records) == len(SEEDS) and {r["seed"] for r in records} == set(SEEDS)}
        for split in ("transductive", "inductive"):
            values = [r[split] for r in records]
            entry[split + "_auc"] = {"mean": statistics.mean(values),
                                     "sample_std": statistics.stdev(values) if len(values) > 1 else None}
        output["datasets"][dataset] = entry
    print(json.dumps(output, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="Run one audited recipe over distinct seeds")
    run_parser.add_argument("--dataset", choices=DATASETS, required=True)
    run_parser.add_argument("--data-root", type=Path, required=True, help="Parent directory of the dataset folders")
    run_parser.add_argument("--embedding-cache", type=Path, required=True, help="Dataset-specific E5-large-v2 .npy")
    run_parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/gin")
    run_parser.add_argument("--seeds", type=int, choices=SEEDS, nargs="+", default=list(SEEDS))
    run_parser.add_argument("--gpu", type=int, default=0, help="Logical CUDA device, or -1 for CPU")
    run_parser.add_argument("--dry-run", action="store_true", help="Print exact commands without loading ML packages or writing files")
    run_parser.add_argument("--allow-input-mismatch", action="store_true", help="Label changed inputs as a new experiment")
    run_parser.add_argument("--smoke-test", action="store_true", help="One epoch; excluded from main-table aggregation")
    run_parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    run_parser.set_defaults(function=run)
    summary_parser = sub.add_parser("summarize", help="Aggregate completed distinct seeds; refuses partial results by default")
    summary_parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/gin")
    summary_parser.add_argument("--datasets", choices=DATASETS, nargs="+", default=list(DATASETS))
    summary_parser.add_argument("--seeds", choices=SEEDS, type=int, nargs="+", default=list(SEEDS))
    summary_parser.add_argument("--allow-partial", action="store_true", help="Report the actual n and incomplete five-seed status")
    summary_parser.add_argument("--allow-input-mismatch", action="store_true")
    summary_parser.set_defaults(function=summarize)
    args = parser.parse_args()
    if hasattr(args, "seeds") and len(set(args.seeds)) != len(args.seeds):
        parser.error("Repeated seeds would double-count a run")
    args.function(args)


if __name__ == "__main__":
    main()
