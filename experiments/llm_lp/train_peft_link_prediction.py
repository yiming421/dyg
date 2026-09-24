#!/usr/bin/env python3
"""
Train a direct PEFT baseline on the current link-prediction prompt pipeline.
"""
import argparse
import hashlib
import inspect
import json
import logging
import os
import shutil
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import torch
from transformers import Trainer, TrainingArguments

_EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_EXPERIMENTS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.seed_runs import add_seed_arguments, launch_seed_workers

from experiments.modules.llm_lp.eval_helpers import load_or_compute_embeddings
from experiments.modules.llm_lp.cli import apply_compact_activity_prompt_recipe
from experiments.modules.llm_lp.experiment import (
    _apply_validation_sampled_binary_labels,
    _apply_validation_sampled_key_signal_reference,
    _apply_validation_sampled_threeway_labels,
    _calibrate_validation_sampled_key_signal_reference,
    _calibrate_validation_sampled_threeway_thresholds,
    _uses_validation_sampled_binary,
    _uses_validation_sampled_key_signal_reference,
    _uses_validation_sampled_threeway,
    build_auto_summary_entity_map,
    build_prompt_entity_map,
    configure_runtime_logging,
    dataset_uses_dtgb_time_bucket,
    entity_name_mode_uses_compact_profile,
    load_dataset_data,
    load_summary_entity_map,
)
from experiments.modules.llm_lp.peft import (
    DEFAULT_LORA_TARGET_MODULES,
    GRAPH_PAIR_FEATURE_KEY,
    GraphPromptMLPOnlyModel,
    PromptResponseCollator,
    PromptResponseDataset,
    attach_hadamard_graph_pair_features,
    build_prompt_response_examples,
    compute_rrf_baseline_metrics,
    create_direct_peft_samples,
    evaluate_binary_completion_likelihood,
    extract_binary_scores_from_training_forward,
    get_distributed_training_context,
    load_frozen_base_model,
    load_peft_training_model,
    load_prompt_tokenizer,
    merge_peft_adapter,
    resolve_process_cuda_device,
    save_json,
    summarize_trainable_parameters,
)
from experiments.modules.llm_lp.prompt_context import (
    calibrate_prompt_key_signals,
    materialize_samples_prompt_context,
)
from experiments.modules.llm_lp.sample_builder import create_test_samples
from experiments.modules.llm_lp.prompt_template import (
    DEFAULT_KEY_SIGNAL_FIELDS,
    SUPPORTED_KEY_SIGNAL_FIELDS,
    normalize_key_signal_fields,
)
from experiments.modules.llm_lp.sample_finalize import (
    apply_overall_structural_signal_buckets,
    assign_expert_prediction_labels,
)
from experiments.modules.llm_lp.semantic_subprocess import SubprocessSemanticMLPScorer
from experiments.modules.llm_lp.training_protocol import (
    protocol_history_edges,
    resolve_training_protocol,
    validate_protocol_samples,
)
from experiments.modules.prediction_metrics import compute_prediction_metrics
from experiments.modules.rrf.analysis import select_topk_rrf_middle_sample_indices


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Direct PEFT training for LLM link prediction.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="GDELT",
        help="DTGB dataset folder name under ../DyLink_Datasets or ./DyLink_Datasets.",
    )
    parser.add_argument(
        "--entity_name_mode",
        type=str,
        default="auto",
        choices=["auto", "raw", "compressed", "compressed_profile"],
        help="Prompt-side entity display mode used when building PEFT prompts.",
    )
    parser.add_argument(
        "--entity_text_path",
        type=str,
        default=None,
        help="Optional override for the dataset entity_text CSV.",
    )
    parser.add_argument("--model_path", type=str, required=True, help="Base HF model path.")
    parser.add_argument("--output_dir", type=str, required=True, help="Training output directory.")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Optional Trainer checkpoint to resume, including adapter, optimizer, "
            "scheduler, RNG, and data-skip state."
        ),
    )
    parser.add_argument("--train_num_samples", type=int, default=5000, help="Positive train queries.")
    parser.add_argument("--val_num_samples", type=int, default=1000, help="Positive validation queries.")
    parser.add_argument("--negative_ratio", type=int, default=1, help="Negatives per positive query.")
    parser.add_argument(
        "--dtgb_eval_batch_size",
        type=int,
        default=256,
        help="Fixed positive-query batch size for DTGB-style AP/AUC averaging.",
    )
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument(
        "--train_split_name",
        type=str,
        default="train",
        choices=["train", "pretest"],
        help=(
            "Which timestamp window to use for PEFT training positives. "
            "Strict training uses the canonical DTGB train interval (<= val_time). "
            "'pretest' is available only for explicit legacy reproduction."
        ),
    )
    parser.add_argument(
        "--train_data_protocol",
        choices=["dtgb_strict", "legacy_time_only"],
        default="dtgb_strict",
        help=(
            "Default: canonical DTGB train-only positives, observed-training negative "
            "destinations, and reserved-node-excluded training histories. "
            "legacy_time_only reproduces the historical node-exposed sampler."
        ),
    )
    parser.add_argument(
        "--data_seed", type=int, default=2020,
        help="DTGB reserved-node split seed; independent of the SFT sampling seed.",
    )
    parser.add_argument("--history_window", type=int, default=100, help="Prompt history window.")
    parser.add_argument("--max_length", type=int, default=8192, help="Training sequence length cap.")
    add_seed_arguments(parser)
    parser.add_argument(
        "--vllm_prompt_variant",
        type=str,
        default="gdelt",
        choices=["gdelt", "temporal_link_prediction"],
        help="Prompt framing variant to match inference prompt construction.",
    )
    parser.add_argument(
        "--edge_sampling_strategy",
        type=str,
        default="most_recent",
        choices=["most_recent", "random"],
        help="How to subsample positives inside each split window.",
    )
    parser.add_argument(
        "--collapse_train_prompt_duplicates",
        action="store_true",
        help=(
            "Collapse positive edge rows that render the same training query before "
            "sampling. The identity is (source,target,time), plus relation when "
            "--include_edge_type is enabled."
        ),
    )
    parser.add_argument(
        "--train_inductive_num_samples",
        type=int,
        default=0,
        help=(
            "Reserve this many slots inside --train_num_samples for natural "
            "pseudo-inductive positives: the source or target is first observed at "
            "that timestamp in the dataset. The remaining positive slots use "
            "--edge_sampling_strategy."
        ),
    )

    parser.add_argument("--summary_mode", type=str, default="off", choices=["off", "entity_meaning", "full"])
    parser.add_argument(
        "--summary_entity_text_path",
        type=str,
        default=None,
        help="Summary CSV used when summary_mode != off.",
    )
    parser.add_argument("--summary_max_chars", type=int, default=120)

    parser.add_argument("--semantic_history", action="store_true")
    parser.add_argument("--semantic_history_entity_mode", action="store_true")
    parser.add_argument("--common_neighbors_semantic", action="store_true")
    parser.add_argument("--semantic_history_no_smoothing", action="store_true")
    parser.add_argument("--embedding_model", type=str, default="intfloat/e5-large-v2")
    parser.add_argument("--embedding_cache", type=str, default=None)
    parser.add_argument("--semantic_topk", type=int, default=None)
    parser.add_argument("--history_pool_size", type=int, default=None)
    parser.add_argument("--history_pool_window", type=int, default=None)
    parser.add_argument("--history_preserve_recent_k", type=int, default=10)
    parser.add_argument("--semantic_hub_penalty_alpha", type=float, default=0.0)
    parser.add_argument("--semantic_fusion_alpha", type=float, default=1.0)
    parser.add_argument("--semantic_fusion_tau", type=float, default=None)
    parser.add_argument("--semantic_fusion_recency_speed", type=float, default=1.0)

    parser.add_argument("--hide_key_signals", action="store_true")
    parser.add_argument("--hide_expert_prediction", action="store_true")
    parser.add_argument("--include_overall_structural_signal", action="store_true")
    parser.add_argument("--overall_signal_name", type=str, default="Overall structural signal")
    parser.add_argument(
        "--key_signal_fields",
        type=str,
        default=",".join(DEFAULT_KEY_SIGNAL_FIELDS),
        help=(
            "Comma-separated prompt key-signal subset. "
            f"Supported: {', '.join(SUPPORTED_KEY_SIGNAL_FIELDS)}"
        ),
    )
    parser.add_argument(
        "--heuristic_recent_degree_window",
        type=float,
        default=30.0,
        help="Hard lookback window for recent_degree prompt signals/RRF during training sample construction.",
    )
    parser.add_argument("--use_raw_key_signals", action="store_true")
    parser.add_argument("--use_percentile_key_signals", action="store_true")
    parser.add_argument("--include_edge_type", action="store_true")
    parser.add_argument("--mutual_timestamps_only", action="store_true")
    parser.add_argument("--mutual_timestamps_dedup", action="store_true")
    parser.add_argument(
        "--mutual_summary_count_recency",
        action="store_true",
        help=(
            "Replace the retained mutual timestamp list with direct-history "
            "existence, distinct displayed-time count, and latest-event recency."
        ),
    )
    parser.add_argument("--common_neighbors_names_only", action="store_true")
    parser.add_argument(
        "--compact_common_neighbors_top_k",
        type=int,
        default=0,
        help=(
            "Compact semantic common neighbors as a pool-size summary plus at most "
            "K ordered neighbors with endpoint-relative direction/time (0 disables)."
        ),
    )
    parser.add_argument(
        "--compact_common_neighbors_novel_only",
        action="store_true",
        help=(
            "Exclude semantic common-neighbor IDs already named in either rendered "
            "compact endpoint history, then backfill from later candidates up to K."
        ),
    )
    parser.add_argument("--history_table_aliases", action="store_true")
    parser.add_argument(
        "--anonymous_entity_aliases",
        action="store_true",
        help=(
            "Replace query endpoints with Source/Target and all other visible "
            "entity names with prompt-local N1, N2, ... aliases."
        ),
    )
    parser.add_argument("--natural_grouped_history", action="store_true")
    parser.add_argument("--natural_activity_summary", action="store_true")
    parser.add_argument("--natural_neighbor_names_only", action="store_true")
    parser.add_argument(
        "--natural_neighbor_top_k",
        type=int,
        default=5,
        help=(
            "For --natural_neighbor_names_only, retain the union of the top-K "
            "frequent and top-K recent partners per activity direction (default: 5)."
        ),
    )
    parser.add_argument(
        "--natural_activity_compact_top3",
        action="store_true",
        help=(
            "Deprecated compatibility flag for compact activity rendering. "
            "Prefer --compact_activity_prompt."
        ),
    )
    parser.add_argument(
        "--compact_activity_prompt",
        action="store_true",
        help=(
            "Unified compact prompt recipe: per-direction compact activity, "
            "timestamp-only deduplicated mutual history, no common-neighbor section "
            "unless --common_neighbors_semantic is set, no key-signal/expert sections, "
            "and minimal output guidance."
        ),
    )
    parser.add_argument(
        "--compact_activity_top_k",
        type=int,
        default=3,
        help=(
            "For --compact_activity_prompt, retain the union of the top-K frequent "
            "and top-K recent partners per activity direction (default: 3)."
        ),
    )
    parser.add_argument("--ablate_mutual_history", action="store_true")
    parser.add_argument("--ablate_common_neighbors", action="store_true")
    parser.add_argument("--ablate_source_history", action="store_true")
    parser.add_argument("--ablate_target_history", action="store_true")
    parser.add_argument("--ablate_source_target_history", action="store_true")
    parser.add_argument("--ablate_reasoning_guidance", action="store_true")

    parser.add_argument("--disable_rrf_scores", action="store_true")
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument(
        "--rrf_mode",
        type=str,
        default="train_pool_pointwise",
        choices=["query_local", "sequential_pointwise", "train_pool_pointwise"],
    )
    parser.add_argument("--sequential_rank_bins", type=int, default=1024)
    parser.add_argument(
        "--expert_prediction_mode",
        type=str,
        default="fixed_threshold",
        choices=[
            "global_median",
            "sequential_running_median",
            "fixed_threshold",
            "validation_sampled_threeway",
            "validation_sampled_binary",
        ],
    )
    parser.add_argument(
        "--expert_prediction_source",
        type=str,
        default="rrf",
        choices=["rrf", "semantic_mlp"],
        help="Structural score source used for prompt prior signal and overall signal buckets.",
    )
    parser.add_argument("--expert_prediction_fixed_threshold", type=float, default=0.05)
    parser.add_argument(
        "--validation_calibration_num_samples",
        type=int,
        default=0,
        help="Positive validation queries used for validation-sampled prior calibration.",
    )
    parser.add_argument(
        "--validation_calibration_negative_ratio",
        type=int,
        default=None,
        help="Negative ratio for validation-sampled prior calibration.",
    )
    parser.add_argument(
        "--validation_calibration_low_neg_quantile",
        type=float,
        default=0.75,
        help="Low-band quantile used by validation-sampled three-way prior calibration.",
    )
    parser.add_argument(
        "--validation_calibration_high_pos_quantile",
        type=float,
        default=0.25,
        help="High-band quantile used by validation-sampled three-way prior calibration.",
    )
    parser.add_argument(
        "--semantic_mlp_checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for semantic structural scoring when expert_prediction_source=semantic_mlp.",
    )
    parser.add_argument(
        "--semantic_mlp_score_python",
        type=str,
        default=None,
        help=(
            "Optional Python executable for out-of-process semantic MLP scoring. "
            "Use the DTGB environment when the training environment lacks compatible sparse ops."
        ),
    )
    parser.add_argument(
        "--semantic_mlp_source_init_override",
        type=str,
        default="auto",
        choices=["auto", "raw", "history_mean"],
        help=(
            "Override semantic source/user embedding initialization when loading a "
            "semantic MLP checkpoint for PEFT prompt prior scoring."
        ),
    )
    parser.add_argument(
        "--semantic_mlp_temporal_mode",
        type=str,
        default="timestamp_rebuild",
        choices=["timestamp_rebuild", "rolling_replay"],
        help="Temporal replay mode for semantic MLP scoring during PEFT sample construction.",
    )
    parser.add_argument("--rrf_pointwise_pool_size", type=int, default=256)
    parser.add_argument("--rrf_pointwise_num_pools", type=int, default=4)
    parser.add_argument("--rrf_batch_size", type=int, default=200000)
    parser.add_argument(
        "--key_signal_reference",
        type=str,
        default="contextual",
        choices=["sequential_global", "contextual", "validation_sampled_global"],
    )
    parser.add_argument("--overall_signal_low_threshold", type=float, default=0.0475)
    parser.add_argument("--overall_signal_high_threshold", type=float, default=0.0510)
    parser.set_defaults(apply_gdelt_time_bucket=None)
    parser.add_argument(
        "--enable_gdelt_time_bucket",
        dest="apply_gdelt_time_bucket",
        action="store_true",
        help="Force DTGB GDELT ts//15 bucketed split boundaries.",
    )
    parser.add_argument(
        "--disable_gdelt_time_bucket",
        dest="apply_gdelt_time_bucket",
        action="store_false",
        help="Disable DTGB GDELT ts//15 bucketed split boundaries.",
    )

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help=(
            "Optionally save a resumable Trainer checkpoint every N optimizer steps. "
            "When omitted, PEFT runs retain the existing epoch-only save behavior."
        ),
    )
    parser.add_argument(
        "--disable_live_train_auc",
        action="store_true",
        help=(
            "Disable train-AUC measurement, including rolling per-log train AUC/AP/accuracy "
            "and the post-training train-split scoring pass."
        ),
    )
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument(
        "--group_by_length",
        action="store_true",
        help=(
            "Group similarly sized tokenized examples into batches to reduce padding. "
            "The longest batch is scheduled early so memory failures surface quickly."
        ),
    )
    parser.add_argument(
        "--train_hybrid_middle_topk",
        type=int,
        default=0,
        help=(
            "If >0, preselect train samples using the eval hybrid middle-band rule: "
            "keep the centered RRF rank window of this size per DTGB batch."
        ),
    )

    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--disable_gradient_checkpointing", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_last_n_layers",
        type=int,
        default=0,
        help=(
            "Restrict LoRA adapters to the final N transformer layers. "
            "Use 0 (default) to adapt every layer."
        ),
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default=",".join(DEFAULT_LORA_TARGET_MODULES),
        help="Comma-separated module names.",
    )
    parser.add_argument(
        "--graph_prompt_mlp_only",
        action="store_true",
        help=(
            "Enable simple graph prompting with cached hadamard(source_emb, target_emb) features and "
            "train only a small MLP projector (base LLM frozen, no LoRA)."
        ),
    )
    parser.add_argument(
        "--graph_prompt_num_tokens",
        type=int,
        default=4,
        help="Number of virtual graph prompt tokens used by the graph projector.",
    )
    parser.add_argument(
        "--graph_prompt_dedicated_slot",
        action="store_true",
        help=(
            "Place graph prompt tokens into a dedicated GRAPH TOKENS section inside the prompt "
            "instead of prepending them before the whole prompt."
        ),
    )
    parser.add_argument(
        "--graph_prompt_special_token",
        type=str,
        default="<graph>",
        help="Special token used for dedicated in-prompt graph token slots.",
    )
    parser.add_argument(
        "--graph_prompt_hidden_dim",
        type=int,
        default=1024,
        help="Hidden size for the graph prompt MLP projector.",
    )
    parser.add_argument(
        "--graph_prompt_dropout",
        type=float,
        default=0.1,
        help="Dropout used in the graph prompt MLP projector.",
    )
    parser.add_argument(
        "--graph_prompt_output_l2_norm",
        action="store_true",
        help="L2-normalize each projected virtual token before prepending it.",
    )
    parser.add_argument(
        "--graph_prompt_zero_init_output",
        action="store_true",
        help="Zero-initialize the final projector layer so training starts from no graph-prefix effect.",
    )
    parser.add_argument(
        "--graph_prompt_mean_token_init",
        action="store_true",
        help=(
            "Initialize the graph projector output to repeated mean token embeddings, "
            "so graph tokens start from a friendly constant soft prompt instead of noisy vectors."
        ),
    )
    parser.add_argument(
        "--graph_prompt_token_space_align_weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary training loss weight that pulls projected virtual tokens toward the "
            "nearest pretrained token embedding."
        ),
    )
    parser.add_argument(
        "--graph_prompt_unfreeze_base_model",
        action="store_true",
        help=(
            "Unfreeze the full base LLM in graph-prompt mode. The fine-tuned base model is "
            "saved separately under <output_dir>/base_model; it is never merged."
        ),
    )
    parser.add_argument(
        "--graph_prompt_init_projector_dir",
        type=str,
        default=None,
        help=(
            "Initialize the graph projector from graph_projector.pt in this directory. "
            "Architecture-related graph-prompt arguments must match the checkpoint."
        ),
    )
    parser.add_argument(
        "--graph_prompt_freeze_projector",
        action="store_true",
        help="Freeze the graph projector while adapting the base LLM.",
    )

    parser.add_argument("--eval_after_training", action="store_true")
    parser.add_argument(
        "--eval_before_training",
        action="store_true",
        help="Run pre-training evaluation on the post-train eval split/sample slice.",
    )
    parser.add_argument("--eval_prompt_batch_size", type=int, default=1)
    parser.add_argument(
        "--post_train_eval_split",
        type=str,
        default="validation",
        choices=["validation", "transductive", "inductive"],
        help="Split used for post-training AUC evaluation.",
    )
    parser.add_argument(
        "--post_train_eval_num_samples",
        type=int,
        default=None,
        help="Positive-query count for post-training evaluation (defaults to val_num_samples).",
    )
    parser.add_argument(
        "--include_prediction_vectors",
        action="store_true",
        help="Keep raw predictions/labels arrays in saved eval JSON files.",
    )
    parser.add_argument("--merge_adapter", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--prepare_samples_only", action="store_true",
        help="Write sampled training identities and protocol audit, then exit before prompts/model loading.",
    )
    return parser


def _validate_training_protocol_args(args):
    """Fail closed before legacy exposure can be mislabeled as strict training."""
    if args.prepare_samples_only and (
            args.semantic_history or args.semantic_history_entity_mode
            or args.common_neighbors_semantic or args.graph_prompt_mlp_only
            or (args.expert_prediction_source == "semantic_mlp"
                and (not args.hide_expert_prediction or args.include_overall_structural_signal))):
        raise ValueError("--prepare_samples_only is model-free and requires the raw-history, no-semantic-model recipe.")
    if args.train_data_protocol == "legacy_time_only":
        warnings.warn(
            "Explicit legacy_time_only reproduction: held-out nodes are not excluded "
            "and pretest includes validation. Do not claim strict inductive training.",
            UserWarning,
        )
        return
    if args.train_split_name != "train":
        raise ValueError("dtgb_strict requires --train_split_name train; pretest includes validation.")
    expert_visible = not args.hide_expert_prediction or args.include_overall_structural_signal
    if (expert_visible and (
            _uses_validation_sampled_binary(args.expert_prediction_mode)
            or _uses_validation_sampled_threeway(args.expert_prediction_mode))):
        raise ValueError("dtgb_strict forbids validation-label-calibrated expert annotations in SFT inputs.")
    if not args.hide_key_signals and _uses_validation_sampled_key_signal_reference(args.key_signal_reference):
        raise ValueError("dtgb_strict forbids validation-calibrated key signals in SFT inputs.")
    config_path = os.path.join(args.output_dir, "run_config.json")
    if args.resume_from_checkpoint:
        if not os.path.isfile(config_path):
            raise ValueError("Strict resume requires the original strict run_config.json in output_dir.")
        with open(config_path, encoding="utf-8") as handle:
            previous = json.load(handle)
        fields = (
            "train_data_protocol", "train_split_name", "data_seed", "dataset_name",
            "val_ratio", "test_ratio", "apply_gdelt_time_bucket", "seed",
            "train_num_samples", "negative_ratio", "edge_sampling_strategy", "model_path",
        )
        if any(previous.get(field) != getattr(args, field) for field in fields):
            raise ValueError("Strict resume data/base provenance differs; do not resume a legacy adapter.")
        checkpoint = os.path.realpath(args.resume_from_checkpoint)
        output = os.path.realpath(args.output_dir)
        if os.path.commonpath([checkpoint, output]) != output:
            raise ValueError("Strict resume checkpoint must belong to this same strict output run.")
    elif int(os.environ.get("RANK", "0")) == 0 and os.path.isfile(config_path):
        raise FileExistsError("Refusing to overwrite an existing strict/legacy run; choose a fresh output_dir.")


def _resolve_lora_target_modules(raw_value: str):
    modules = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not modules:
        raise ValueError("lora_target_modules cannot be empty.")
    return modules


def _validate_strict_resume_samples(args, protocol, samples):
    if args.train_data_protocol != "dtgb_strict" or not args.resume_from_checkpoint:
        return
    protocol_path = os.path.join(args.output_dir, "training_protocol.json")
    manifest_path = os.path.join(args.output_dir, "train_sample_manifest_meta.json")
    if not os.path.isfile(protocol_path) or not os.path.isfile(manifest_path):
        raise ValueError("Strict resume requires saved graph and exact sampled-row provenance.")
    with open(protocol_path, encoding="utf-8") as handle:
        previous = json.load(handle)
    for key in ("full_graph_identity_sha256", "history_graph_identity_sha256",
                "reserved_node_ids_sha256", "negative_destination_pool_sha256"):
        if not previous.get(key) or previous[key] != protocol.metadata.get(key):
            raise ValueError(f"Strict resume graph/split provenance changed: {key}")
    digest = hashlib.sha256()
    fields = ("query_id", "label", "source_id", "relation_id", "target_id", "timestamp")
    for sample in samples:
        row = {key: int(sample[key]) for key in fields}
        digest.update((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
    with open(manifest_path, encoding="utf-8") as handle:
        previous_samples = json.load(handle)
    if previous_samples.get("sha256") != digest.hexdigest():
        raise ValueError("Strict resume training sample identities changed; refusing to overwrite provenance.")


def _save_train_sample_manifest(output_dir: str, samples):
    """Persist the exact sampled training pairs without prompt text."""
    manifest_path = os.path.join(output_dir, "train_sample_manifest.jsonl")
    digest = hashlib.sha256()
    positive_count = 0
    negative_count = 0
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for sample in samples:
            label = int(sample["label"])
            row = {
                "query_id": int(sample["query_id"]),
                "label": label,
                "source_id": int(sample["source_id"]),
                "relation_id": int(sample["relation_id"]),
                "target_id": int(sample["target_id"]),
                "timestamp": int(sample["timestamp"]),
            }
            line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
            if label == 1:
                positive_count += 1
            else:
                negative_count += 1

    metadata = {
        "path": os.path.basename(manifest_path),
        "sha256": digest.hexdigest(),
        "num_samples": int(len(samples)),
        "num_positive": int(positive_count),
        "num_negative": int(negative_count),
        "fields": [
            "query_id",
            "label",
            "source_id",
            "relation_id",
            "target_id",
            "timestamp",
        ],
    }
    save_json(os.path.join(output_dir, "train_sample_manifest_meta.json"), metadata)
    print(
        "Training sample manifest: "
        f"samples={metadata['num_samples']}, positives={positive_count}, "
        f"negatives={negative_count}, sha256={metadata['sha256']}"
    )
    return metadata


def _maybe_preselect_train_samples(samples, args):
    top_k = int(getattr(args, "train_hybrid_middle_topk", 0) or 0)
    if top_k <= 0:
        return samples, None

    selected_indices, selection_meta = select_topk_rrf_middle_sample_indices(
        samples=samples,
        top_k=top_k,
        dtgb_eval_batch_size=args.dtgb_eval_batch_size,
    )
    selected_samples = [samples[idx] for idx in selected_indices]
    compact_selection_meta = {
        "enabled": bool(selection_meta.get("enabled", True)),
        "selection_mode": selection_meta.get("selection_mode", "per_dtgb_batch_middle_band"),
        "requested_top_k": int(selection_meta.get("requested_top_k", top_k)),
        "selected_count": int(selection_meta.get("selected_count", len(selected_samples))),
        "selected_batches": int(selection_meta.get("selected_batches", 0)),
        "available_batches": int(selection_meta.get("available_batches", 0)),
        "dtgb_eval_batch_size": int(
            selection_meta.get("dtgb_eval_batch_size", args.dtgb_eval_batch_size)
        ),
    }
    selection_summary = {
        "enabled": True,
        "selection_mode": "per_dtgb_batch_middle_band",
        "requested_top_k": int(top_k),
        "dtgb_eval_batch_size": int(args.dtgb_eval_batch_size),
        "selected_count": int(len(selected_samples)),
        "selected_positive": int(sum(1 for sample in selected_samples if int(sample["label"]) == 1)),
        "selected_negative": int(sum(1 for sample in selected_samples if int(sample["label"]) == 0)),
        "total_before_selection": int(len(samples)),
        "total_after_selection": int(len(selected_samples)),
        "selected_fraction_realized": (
            float(len(selected_samples)) / float(len(samples)) if samples else 0.0
        ),
        "selection": compact_selection_meta,
    }
    return selected_samples, selection_summary


def _maybe_materialize_selected_train_prompt_context(
    *,
    samples,
    args,
    edges,
    entity_map,
    embeddings,
    entity_id_to_idx,
):
    if int(getattr(args, "train_hybrid_middle_topk", 0) or 0) <= 0:
        return
    if not samples:
        return

    materialize_samples_prompt_context(
        samples=samples,
        edges_df=edges,
        entity_map=entity_map,
        history_window=args.history_window,
        semantic_history=args.semantic_history,
        semantic_topk=args.semantic_topk,
        semantic_history_entity_mode=args.semantic_history_entity_mode,
        common_neighbors_semantic=args.common_neighbors_semantic,
        semantic_use_smoothing=(not args.semantic_history_no_smoothing),
        semantic_hub_penalty_alpha=args.semantic_hub_penalty_alpha,
        semantic_fusion_alpha=args.semantic_fusion_alpha,
        semantic_fusion_tau=args.semantic_fusion_tau,
        semantic_fusion_recency_speed=args.semantic_fusion_recency_speed,
        history_pool_size=args.history_pool_size,
        history_pool_window=args.history_pool_window,
        history_preserve_recent_k=args.history_preserve_recent_k,
        embeddings=embeddings,
        entity_id_to_idx=entity_id_to_idx,
        embedding_model=args.embedding_model,
        embedding_cache=args.embedding_cache,
        populate_prompt_lists=True,
        calibrate_key_signals=False,
        monitor_label="Materializing selected train prompt context",
    )
    calibrate_prompt_key_signals(
        samples=samples,
        key_signal_reference=getattr(
            args, "runtime_key_signal_reference", args.key_signal_reference
        ),
        edges_df=edges,
        heuristic_recent_degree_window=args.heuristic_recent_degree_window,
    )


def _build_training_args(args, has_eval_dataset: bool):
    dist_ctx = get_distributed_training_context()
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    save_strategy = (
        "no"
        if args.graph_prompt_mlp_only
        else ("steps" if args.save_steps is not None else "epoch")
    )
    training_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy=save_strategy,
        save_steps=args.save_steps if args.save_steps is not None else 500,
        save_total_limit=args.save_total_limit,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        group_by_length=args.group_by_length,
        seed=args.seed,
        local_rank=dist_ctx["local_rank"],
        ddp_find_unused_parameters=False if dist_ctx["is_distributed"] else None,
        ddp_backend="nccl" if dist_ctx["is_distributed"] and torch.cuda.is_available() else None,
        disable_tqdm=not dist_ctx["is_main_process"],
    )
    if "save_safetensors" in inspect.signature(TrainingArguments.__init__).parameters:
        training_kwargs["save_safetensors"] = True
    training_kwargs["eval_strategy"] = "epoch" if has_eval_dataset else "no"
    return TrainingArguments(**training_kwargs)


def _maybe_barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.cuda.is_available():
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        else:
            torch.distributed.barrier()


def _run_main_process_only(fn, *, enabled=True):
    if not enabled:
        return None
    _maybe_barrier()
    result = fn()
    _maybe_barrier()
    return result


def _maybe_destroy_process_group():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _strip_prediction_vectors(metrics):
    if not isinstance(metrics, dict):
        return metrics
    compact = dict(metrics)
    compact.pop("predictions", None)
    compact.pop("labels", None)
    baseline = compact.get("rrf_baseline")
    if isinstance(baseline, dict):
        baseline_compact = dict(baseline)
        baseline_compact.pop("predictions", None)
        baseline_compact.pop("labels", None)
        compact["rrf_baseline"] = baseline_compact
    return compact


def _metric_summary_for_console(metrics):
    if not isinstance(metrics, dict):
        return metrics
    keys = [
        "ap",
        "auc",
        "accuracy",
        "ap_global",
        "auc_global",
        "num_samples",
        "num_positive",
        "num_negative",
        "negative_ratio",
        "dtgb_eval_batch_size",
        "dtgb_num_metric_batches",
        "dtgb_metric_aggregation",
        "eval_split",
        "eval_seed",
    ]
    summary = {key: metrics[key] for key in keys if key in metrics}
    baseline = metrics.get("rrf_baseline")
    if isinstance(baseline, dict):
        baseline_keys = [
            "ap",
            "auc",
            "accuracy",
            "ap_global",
            "auc_global",
            "num_samples",
            "num_positive",
            "num_negative",
            "negative_ratio",
            "dtgb_eval_batch_size",
            "dtgb_num_metric_batches",
            "dtgb_metric_aggregation",
        ]
        summary["rrf_baseline"] = {
            key: baseline[key] for key in baseline_keys if key in baseline
        }
    return summary


def _build_validation_calibration_args(args):
    default_num_samples = int(args.val_num_samples)
    if default_num_samples <= 0:
        default_num_samples = int(args.train_num_samples)
    return SimpleNamespace(
        dataset_name=args.dataset_name,
        num_samples=default_num_samples,
        negative_ratio=args.negative_ratio,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        expert_prediction_mode=args.expert_prediction_mode,
        expert_prediction_fixed_threshold=args.expert_prediction_fixed_threshold,
        validation_calibration_num_samples=args.validation_calibration_num_samples,
        validation_calibration_negative_ratio=args.validation_calibration_negative_ratio,
        validation_calibration_low_neg_quantile=args.validation_calibration_low_neg_quantile,
        validation_calibration_high_pos_quantile=args.validation_calibration_high_pos_quantile,
        rrf_k=args.rrf_k,
        rrf_mode=args.rrf_mode,
        sequential_rank_bins=args.sequential_rank_bins,
        rrf_pointwise_pool_size=args.rrf_pointwise_pool_size,
        rrf_pointwise_num_pools=args.rrf_pointwise_num_pools,
        rrf_batch_size=args.rrf_batch_size,
        rrf_heuristics=None,
        key_signal_fields=args.key_signal_fields,
        overall_signal_low_threshold=args.overall_signal_low_threshold,
        overall_signal_high_threshold=args.overall_signal_high_threshold,
        smooth_time_window=50.0,
        smooth_steps=1,
        smooth_decay_gamma=0.1,
        smooth_undirected=True,
    )


class RollingTrainMetricTrainer(Trainer):
    def __init__(self, *args, tokenizer_for_metrics=None, enable_live_train_metrics=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._metric_tokenizer = tokenizer_for_metrics
        self._enable_live_train_metrics = bool(enable_live_train_metrics)
        self._binary_token_groups = None
        self._rolling_train_scores = []
        self._rolling_train_labels = []

    def _ensure_binary_token_groups(self):
        if self._binary_token_groups is None:
            from experiments.modules.llm_lp.peft import _build_binary_token_groups

            self._binary_token_groups = _build_binary_token_groups(self._metric_tokenizer)
        return self._binary_token_groups

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        if not self._enable_live_train_metrics:
            return (loss, outputs) if return_outputs else loss

        try:
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            binary_token_groups = self._ensure_binary_token_groups()
            scores, labels = extract_binary_scores_from_training_forward(
                model=model,
                batch=inputs,
                logits=logits.detach(),
                binary_token_groups=binary_token_groups,
            )
            if scores:
                self._rolling_train_scores.extend(scores)
                self._rolling_train_labels.extend(labels)
        except Exception:
            pass

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        if (
            self._enable_live_train_metrics
            and self._rolling_train_scores
            and len(set(self._rolling_train_labels)) >= 2
        ):
            metric_payload = compute_prediction_metrics(
                self._rolling_train_scores,
                self._rolling_train_labels,
            )
            logs["train_auc"] = float(metric_payload["auc"])
            logs["train_ap"] = float(metric_payload["ap"])
            logs["train_accuracy"] = float(metric_payload["accuracy"])
        self._rolling_train_scores.clear()
        self._rolling_train_labels.clear()
        if start_time is None:
            return super().log(logs)
        return super().log(logs, start_time)


def main():
    args = build_arg_parser().parse_args()
    if launch_seed_workers(args, "experiments.llm_lp.train_peft_link_prediction", "--output_dir"):
        return
    apply_compact_activity_prompt_recipe(args)
    if int(args.compact_common_neighbors_top_k) < 0:
        raise ValueError("--compact_common_neighbors_top_k must be >= 0")
    if int(args.train_inductive_num_samples) < 0:
        raise ValueError("--train_inductive_num_samples must be >= 0")
    if int(args.train_inductive_num_samples) > int(args.train_num_samples):
        raise ValueError(
            "--train_inductive_num_samples cannot exceed --train_num_samples"
        )
    if args.compact_common_neighbors_top_k and not args.common_neighbors_semantic:
        raise ValueError(
            "--compact_common_neighbors_top_k requires --common_neighbors_semantic"
        )
    if args.compact_common_neighbors_top_k and args.common_neighbors_names_only:
        raise ValueError(
            "--compact_common_neighbors_top_k cannot be combined with "
            "--common_neighbors_names_only"
        )
    if args.compact_common_neighbors_novel_only:
        if not args.compact_common_neighbors_top_k:
            raise ValueError(
                "--compact_common_neighbors_novel_only requires "
                "--compact_common_neighbors_top_k"
            )
        if not args.natural_activity_compact_top3:
            raise ValueError(
                "--compact_common_neighbors_novel_only requires compact activity histories"
            )
        if args.anonymous_entity_aliases:
            raise ValueError(
                "--compact_common_neighbors_novel_only cannot be combined with "
                "--anonymous_entity_aliases"
            )
    args.key_signal_fields = normalize_key_signal_fields(getattr(args, "key_signal_fields", None))
    if float(args.heuristic_recent_degree_window) <= 0.0:
        raise ValueError("--heuristic_recent_degree_window must be > 0.")
    if args.graph_prompt_mlp_only and args.load_in_4bit:
        raise ValueError(
            "--graph_prompt_mlp_only currently does not support --load_in_4bit. "
            "Disable 4-bit loading for projector-only training."
        )
    graph_projector_init_path = None
    if args.graph_prompt_init_projector_dir:
        if not args.graph_prompt_mlp_only:
            raise ValueError(
                "--graph_prompt_init_projector_dir requires --graph_prompt_mlp_only."
            )
        graph_projector_init_path = os.path.join(
            args.graph_prompt_init_projector_dir,
            "graph_projector.pt",
        )
        if not os.path.isfile(graph_projector_init_path):
            raise ValueError(
                f"Graph projector checkpoint not found: {graph_projector_init_path}"
            )
    if args.graph_prompt_unfreeze_base_model and not args.graph_prompt_mlp_only:
        raise ValueError(
            "--graph_prompt_unfreeze_base_model requires --graph_prompt_mlp_only."
        )
    if args.graph_prompt_freeze_projector and not args.graph_prompt_init_projector_dir:
        raise ValueError(
            "--graph_prompt_freeze_projector requires --graph_prompt_init_projector_dir."
        )
    if (
        (args.eval_before_training or args.eval_after_training)
        and args.post_train_eval_split == "validation"
        and int(args.val_num_samples) <= 0
    ):
        raise ValueError(
            "Validation evaluation requested but --val_num_samples <= 0. "
            "Set --post_train_eval_split to transductive/inductive, or set --val_num_samples > 0."
        )
    configure_runtime_logging()
    logging.getLogger("numba.cuda.cudadrv").setLevel(logging.WARNING)
    try:
        from numba.core.errors import NumbaPerformanceWarning

        warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
    except Exception:
        pass
    dist_ctx = get_distributed_training_context()
    resolved_cuda_device = resolve_process_cuda_device(dist_ctx["local_rank"])
    if resolved_cuda_device >= 0:
        torch.cuda.set_device(resolved_cuda_device)

    if args.apply_gdelt_time_bucket is None:
        args.apply_gdelt_time_bucket = dataset_uses_dtgb_time_bucket(args.dataset_name)
    _validate_training_protocol_args(args)
    if args.expert_prediction_source == "semantic_mlp" and not args.semantic_mlp_checkpoint:
        raise ValueError(
            "--semantic_mlp_checkpoint is required when --expert_prediction_source semantic_mlp."
        )
    if args.semantic_mlp_checkpoint and not os.path.isfile(args.semantic_mlp_checkpoint):
        raise ValueError(f"--semantic_mlp_checkpoint not found: {args.semantic_mlp_checkpoint}")
    if (
        args.semantic_mlp_score_python
        and not os.path.isfile(args.semantic_mlp_score_python)
        and shutil.which(args.semantic_mlp_score_python) is None
    ):
        raise ValueError(f"--semantic_mlp_score_python not found: {args.semantic_mlp_score_python}")

    os.makedirs(args.output_dir, exist_ok=True)

    edges, entity_map, relation_map = load_dataset_data(
        args.dataset_name,
        entity_text_path=args.entity_text_path,
    )
    prompt_entity_map = build_prompt_entity_map(
        args.dataset_name,
        entity_map,
        entity_name_mode=args.entity_name_mode,
    )

    if (
        entity_name_mode_uses_compact_profile(args.dataset_name, args.entity_name_mode)
        and str(args.summary_mode).strip().lower() == "off"
    ):
        args.summary_mode = "full"

    summary_map = None
    if args.summary_mode != "off":
        if args.summary_entity_text_path:
            summary_map = load_summary_entity_map(args.summary_entity_text_path)
        else:
            summary_map = build_auto_summary_entity_map(args.dataset_name, entity_map)
            if summary_map is None:
                raise ValueError(
                    "--summary_entity_text_path is required when --summary_mode is not off "
                    f"for dataset {args.dataset_name}"
                )

    expert_source_is_semantic = str(args.expert_prediction_source).strip().lower() == "semantic_mlp"
    expert_score_field = "semantic_mlp_score" if expert_source_is_semantic else "rrf_score"
    expert_score_label = "Semantic MLP" if expert_source_is_semantic else "RRF"
    use_validation_sampled_threeway = _uses_validation_sampled_threeway(args.expert_prediction_mode)
    use_validation_sampled_binary = _uses_validation_sampled_binary(args.expert_prediction_mode)
    use_validation_sampled_key_signals = _uses_validation_sampled_key_signal_reference(
        args.key_signal_reference
    )
    expert_signal_visible = bool((not args.hide_expert_prediction) or args.include_overall_structural_signal)
    need_semantic_expert_scorer = bool(expert_source_is_semantic and expert_signal_visible)

    embeddings = None
    entity_id_to_idx = None
    need_semantic_embeddings = (
        args.semantic_history
        or args.semantic_history_entity_mode
        or args.common_neighbors_semantic
        or args.graph_prompt_mlp_only
        or need_semantic_expert_scorer
    )
    if need_semantic_embeddings:
        embeddings, entity_id_to_idx = load_or_compute_embeddings(
            entity_map,
            args.embedding_model,
            args.embedding_cache,
        )

    semantic_backbone_scorer = None
    if need_semantic_expert_scorer:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Semantic MLP structural scoring requires CUDA. No GPU is available."
            )
        if args.semantic_mlp_score_python:
            semantic_backbone_scorer = SubprocessSemanticMLPScorer(
                python_path=args.semantic_mlp_score_python,
                dataset_name=args.dataset_name,
                checkpoint_path=args.semantic_mlp_checkpoint,
                embeddings=embeddings,
                entity_ids_sorted=sorted(entity_map.keys()),
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                eval_positive_batch_size=args.dtgb_eval_batch_size,
                source_init_override=args.semantic_mlp_source_init_override,
                temporal_mode=args.semantic_mlp_temporal_mode,
            )
        else:
            from experiments.modules.semantic_mlp.backbone import SemanticMLPHybridBackbone

            semantic_backbone_scorer = SemanticMLPHybridBackbone.from_checkpoint(
                dataset_name=args.dataset_name,
                checkpoint_path=args.semantic_mlp_checkpoint,
                embeddings=embeddings,
                entity_ids_sorted=sorted(entity_map.keys()),
                device="cuda",
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                eval_positive_batch_size=args.dtgb_eval_batch_size,
                source_init_override=args.semantic_mlp_source_init_override,
                temporal_mode=args.semantic_mlp_temporal_mode,
            )

    def annotate_expert_source_scores(samples, *, score_field=None):
        if not expert_source_is_semantic or not samples:
            return samples
        target_field = str(score_field or expert_score_field)
        if all(sample.get(target_field) is not None for sample in samples):
            return samples
        if semantic_backbone_scorer is None:
            raise RuntimeError("Semantic expert scoring requested but the scorer is unavailable.")
        semantic_backbone_scorer.annotate_samples(samples, score_field=target_field)
        return samples

    validation_sampled_calibration = None
    if expert_signal_visible and (use_validation_sampled_threeway or use_validation_sampled_binary):
        validation_sampled_calibration = _calibrate_validation_sampled_threeway_thresholds(
            args=_build_validation_calibration_args(args),
            edges=edges,
            embeddings=embeddings,
            entity_id_to_idx=entity_id_to_idx,
            random_seed=args.seed,
            score_field=expert_score_field,
            score_label=expert_score_label,
            score_samples_fn=(
                annotate_expert_source_scores if expert_source_is_semantic else None
            ),
        )

    rrf_drives_prior_signal = bool((not expert_source_is_semantic) and expert_signal_visible)
    include_expert_prediction = not args.hide_expert_prediction
    include_key_signals = not args.hide_key_signals
    validation_sampled_key_signal_reference = None
    if use_validation_sampled_key_signals and include_key_signals:
        validation_sampled_key_signal_reference = _calibrate_validation_sampled_key_signal_reference(
            args=_build_validation_calibration_args(args),
            edges=edges,
            embeddings=embeddings,
            entity_id_to_idx=entity_id_to_idx,
            random_seed=args.seed,
        )
    runtime_key_signal_reference = (
        "sequential_global"
        if use_validation_sampled_key_signals
        else str(args.key_signal_reference).strip().lower()
    )
    runtime_skip_key_signal_calibration = bool(use_validation_sampled_key_signals)
    args.runtime_key_signal_reference = runtime_key_signal_reference
    compute_rrf_scores = bool((not args.disable_rrf_scores) or rrf_drives_prior_signal)
    semantic_use_smoothing = not args.semantic_history_no_smoothing
    runtime_compute_expert_prediction = (
        include_expert_prediction
        and (not use_validation_sampled_threeway)
        and (not use_validation_sampled_binary)
        and (not expert_source_is_semantic)
    )
    runtime_include_overall_structural_signal = bool(
        args.include_overall_structural_signal and (not expert_source_is_semantic)
    )
    runtime_expert_prediction_mode = (
        "fixed_threshold"
        if (use_validation_sampled_threeway or use_validation_sampled_binary)
        else args.expert_prediction_mode
    )

    def apply_runtime_expert_signal_annotations(samples):
        if not samples:
            return samples
        if not expert_signal_visible:
            return samples
        if expert_source_is_semantic:
            annotate_expert_source_scores(samples, score_field=expert_score_field)
        if validation_sampled_calibration is not None:
            calibration_mode = str(
                validation_sampled_calibration.get("mode", "validation_sampled_threeway")
            ).strip().lower()
            if calibration_mode == "validation_sampled_binary":
                return _apply_validation_sampled_binary_labels(
                    samples,
                    threshold=validation_sampled_calibration["threshold"],
                    assign_expert_prediction=bool(include_expert_prediction),
                    assign_overall_signal=bool(args.include_overall_structural_signal),
                    score_field=expert_score_field,
                )
            return _apply_validation_sampled_threeway_labels(
                samples,
                low_threshold=validation_sampled_calibration["low_threshold"],
                high_threshold=validation_sampled_calibration["high_threshold"],
                assign_expert_prediction=bool(include_expert_prediction),
                assign_overall_signal=bool(args.include_overall_structural_signal),
                score_field=expert_score_field,
            )
        if expert_source_is_semantic:
            if include_expert_prediction:
                assign_expert_prediction_labels(
                    samples=samples,
                    negative_ratio=args.negative_ratio,
                    expert_prediction_mode=args.expert_prediction_mode,
                    expert_prediction_fixed_threshold=args.expert_prediction_fixed_threshold,
                    score_field=expert_score_field,
                    score_label=expert_score_label,
                )
            if args.include_overall_structural_signal:
                apply_overall_structural_signal_buckets(
                    samples,
                    low_threshold=args.overall_signal_low_threshold,
                    high_threshold=args.overall_signal_high_threshold,
                    score_field=expert_score_field,
                )
        return samples

    def apply_runtime_key_signal_annotations(samples):
        if (
            not samples
            or not include_key_signals
            or validation_sampled_key_signal_reference is None
        ):
            return samples
        return _apply_validation_sampled_key_signal_reference(
            samples,
            validation_sampled_key_signal_reference,
        )

    if dist_ctx["is_main_process"]:
        save_json(os.path.join(args.output_dir, "run_config.json"), vars(args))
        if dist_ctx["is_distributed"]:
            print(
                f"DDP mode enabled: world_size={dist_ctx['world_size']}, "
                f"local_rank={dist_ctx['local_rank']}"
            )

    train_samples, train_sampling_stats = create_direct_peft_samples(
        edges_df=edges,
        entity_map=prompt_entity_map,
        relation_map=relation_map,
        split_name=args.train_split_name,
        num_samples=args.train_num_samples,
        negative_ratio=args.negative_ratio,
        random_seed=args.seed,
        train_data_protocol=args.train_data_protocol,
        data_seed=args.data_seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        history_window=args.history_window,
        semantic_history=args.semantic_history,
        semantic_topk=args.semantic_topk,
        semantic_history_entity_mode=args.semantic_history_entity_mode,
        common_neighbors_semantic=args.common_neighbors_semantic,
        semantic_use_smoothing=semantic_use_smoothing,
        semantic_hub_penalty_alpha=args.semantic_hub_penalty_alpha,
        semantic_fusion_alpha=args.semantic_fusion_alpha,
        semantic_fusion_tau=args.semantic_fusion_tau,
        semantic_fusion_recency_speed=args.semantic_fusion_recency_speed,
        history_pool_size=args.history_pool_size,
        history_pool_window=args.history_pool_window,
        history_preserve_recent_k=args.history_preserve_recent_k,
        embeddings=embeddings,
        entity_id_to_idx=entity_id_to_idx,
        embedding_model=args.embedding_model,
        embedding_cache=args.embedding_cache,
        compute_expert_prediction=runtime_compute_expert_prediction,
        compute_rrf_scores=compute_rrf_scores,
        rrf_k=args.rrf_k,
        rrf_mode=args.rrf_mode,
        sequential_rank_bins=args.sequential_rank_bins,
        expert_prediction_mode=runtime_expert_prediction_mode,
        expert_prediction_fixed_threshold=args.expert_prediction_fixed_threshold,
        rrf_pointwise_pool_size=args.rrf_pointwise_pool_size,
        rrf_pointwise_num_pools=args.rrf_pointwise_num_pools,
        rrf_batch_size=args.rrf_batch_size,
        key_signal_reference=runtime_key_signal_reference,
        key_signal_fields=args.key_signal_fields,
        skip_key_signal_calibration=((not include_key_signals) or runtime_skip_key_signal_calibration),
        include_overall_structural_signal=runtime_include_overall_structural_signal,
        overall_signal_low_threshold=args.overall_signal_low_threshold,
        overall_signal_high_threshold=args.overall_signal_high_threshold,
        apply_gdelt_time_bucket=args.apply_gdelt_time_bucket,
        sampling_strategy=args.edge_sampling_strategy,
        sampling_skip_recent=(args.val_num_samples if args.train_split_name == "pretest" else 0),
        collapse_prompt_duplicates=args.collapse_train_prompt_duplicates,
        prompt_identity_includes_relation=args.include_edge_type,
        inductive_num_samples=args.train_inductive_num_samples,
        return_selection_stats=True,
        defer_prompt_context_materialization=(args.prepare_samples_only or int(args.train_hybrid_middle_topk) > 0),
        heuristic_recent_degree_window=args.heuristic_recent_degree_window,
    )
    train_selection_stats = None
    train_samples, train_selection_stats = _maybe_preselect_train_samples(train_samples, args)
    if not train_samples:
        raise RuntimeError("No train samples remain after train_hybrid_middle_topk preselection.")
    if dist_ctx["is_main_process"] and train_selection_stats is not None:
        selection = train_selection_stats["selection"]
        print(
            "Train hybrid middle-band selection: "
            f"kept {train_selection_stats['selected_count']}/{train_selection_stats['total_before_selection']} "
            f"samples across {selection.get('selected_batches', 0)} DTGB batches "
            f"(requested {selection.get('requested_top_k', args.train_hybrid_middle_topk)} per batch)."
        )
    training_protocol = resolve_training_protocol(
        edges, split_name=args.train_split_name,
        train_data_protocol=args.train_data_protocol, data_seed=args.data_seed,
        val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        apply_gdelt_time_bucket=args.apply_gdelt_time_bucket,
    )
    protocol_audit = validate_protocol_samples(train_samples, training_protocol)
    _validate_strict_resume_samples(args, training_protocol, train_samples)
    if dist_ctx["is_main_process"]:
        _save_train_sample_manifest(args.output_dir, train_samples)
        save_json(os.path.join(args.output_dir, "training_protocol.json"), {
            **training_protocol.metadata, "selected_sample_audit": protocol_audit,
            "preparation_only": bool(args.prepare_samples_only),
            "sampling": train_sampling_stats,
        })
    if args.prepare_samples_only:
        print("Training sample preparation complete; no prompts tokenized or model weights loaded.")
        _maybe_destroy_process_group()
        return
    _maybe_materialize_selected_train_prompt_context(
        samples=train_samples,
        args=args,
        edges=protocol_history_edges(edges, training_protocol),
        entity_map=prompt_entity_map,
        embeddings=embeddings,
        entity_id_to_idx=entity_id_to_idx,
    )
    protocol_audit = validate_protocol_samples(train_samples, training_protocol)
    if dist_ctx["is_main_process"]:
        save_json(os.path.join(args.output_dir, "training_protocol.json"), {
            **training_protocol.metadata, "selected_sample_audit": protocol_audit,
            "preparation_only": False, "sampling": train_sampling_stats,
        })
    apply_runtime_key_signal_annotations(train_samples)
    apply_runtime_expert_signal_annotations(train_samples)

    val_samples = []
    if args.val_num_samples > 0:
        val_samples = create_direct_peft_samples(
            edges_df=edges,
            entity_map=prompt_entity_map,
            relation_map=relation_map,
            split_name="validation",
            num_samples=args.val_num_samples,
            negative_ratio=args.negative_ratio,
            random_seed=args.seed,
            train_data_protocol=args.train_data_protocol,
            data_seed=args.data_seed,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            history_window=args.history_window,
            semantic_history=args.semantic_history,
            semantic_topk=args.semantic_topk,
            semantic_history_entity_mode=args.semantic_history_entity_mode,
            common_neighbors_semantic=args.common_neighbors_semantic,
            semantic_use_smoothing=semantic_use_smoothing,
            semantic_hub_penalty_alpha=args.semantic_hub_penalty_alpha,
            semantic_fusion_alpha=args.semantic_fusion_alpha,
            semantic_fusion_tau=args.semantic_fusion_tau,
            semantic_fusion_recency_speed=args.semantic_fusion_recency_speed,
            history_pool_size=args.history_pool_size,
            history_pool_window=args.history_pool_window,
            history_preserve_recent_k=args.history_preserve_recent_k,
            embeddings=embeddings,
            entity_id_to_idx=entity_id_to_idx,
            embedding_model=args.embedding_model,
            embedding_cache=args.embedding_cache,
            compute_expert_prediction=runtime_compute_expert_prediction,
            compute_rrf_scores=compute_rrf_scores,
            rrf_k=args.rrf_k,
            rrf_mode=args.rrf_mode,
            sequential_rank_bins=args.sequential_rank_bins,
            expert_prediction_mode=runtime_expert_prediction_mode,
            expert_prediction_fixed_threshold=args.expert_prediction_fixed_threshold,
            rrf_pointwise_pool_size=args.rrf_pointwise_pool_size,
            rrf_pointwise_num_pools=args.rrf_pointwise_num_pools,
            rrf_batch_size=args.rrf_batch_size,
            key_signal_reference=runtime_key_signal_reference,
            key_signal_fields=args.key_signal_fields,
            skip_key_signal_calibration=((not include_key_signals) or runtime_skip_key_signal_calibration),
            include_overall_structural_signal=runtime_include_overall_structural_signal,
            overall_signal_low_threshold=args.overall_signal_low_threshold,
            overall_signal_high_threshold=args.overall_signal_high_threshold,
            apply_gdelt_time_bucket=args.apply_gdelt_time_bucket,
            sampling_strategy=args.edge_sampling_strategy,
            heuristic_recent_degree_window=args.heuristic_recent_degree_window,
        )
        apply_runtime_key_signal_annotations(val_samples)
        apply_runtime_expert_signal_annotations(val_samples)

    graph_feature_meta = None
    if args.graph_prompt_mlp_only:
        if embeddings is None or entity_id_to_idx is None:
            raise RuntimeError(
                "Graph prompt MLP mode requires node embeddings. "
                "Provide --embedding_cache (recommended) or allow embedding computation."
            )
        train_graph_meta = attach_hadamard_graph_pair_features(
            train_samples,
            embeddings=embeddings,
            entity_id_to_idx=entity_id_to_idx,
            feature_key=GRAPH_PAIR_FEATURE_KEY,
        )
        val_graph_meta = attach_hadamard_graph_pair_features(
            val_samples,
            embeddings=embeddings,
            entity_id_to_idx=entity_id_to_idx,
            feature_key=GRAPH_PAIR_FEATURE_KEY,
        )
        graph_feature_meta = {
            "feature_key": GRAPH_PAIR_FEATURE_KEY,
            "train": train_graph_meta,
            "validation": val_graph_meta,
        }

    graph_prompt_special_token = None
    graph_prompt_num_tokens = 0
    if args.graph_prompt_mlp_only and args.graph_prompt_dedicated_slot:
        graph_prompt_special_token = str(args.graph_prompt_special_token).strip()
        graph_prompt_num_tokens = int(args.graph_prompt_num_tokens)

    tokenizer_for_prompts = load_prompt_tokenizer(
        args.model_path,
        graph_prompt_special_token=graph_prompt_special_token,
    )

    prompt_kwargs = {
        "summary_map": summary_map,
        "summary_mode": args.summary_mode,
        "summary_max_chars": args.summary_max_chars,
        "history_window": args.history_window,
        "include_key_signals": include_key_signals,
        "include_expert_prediction": include_expert_prediction,
        "include_overall_structural_signal": args.include_overall_structural_signal,
        "overall_structural_signal_name": args.overall_signal_name,
        "key_signal_fields": args.key_signal_fields,
        "use_raw_key_signals": args.use_raw_key_signals,
        "use_percentile_key_signals": args.use_percentile_key_signals,
        "include_edge_type": args.include_edge_type,
        "mutual_timestamps_only": args.mutual_timestamps_only,
        "mutual_timestamps_dedup": args.mutual_timestamps_dedup,
        "mutual_summary_count_recency": args.mutual_summary_count_recency,
        "common_neighbors_names_only": args.common_neighbors_names_only,
        "compact_common_neighbors_top_k": args.compact_common_neighbors_top_k,
        "compact_common_neighbors_novel_only": args.compact_common_neighbors_novel_only,
        "history_table_aliases": args.history_table_aliases,
        "anonymous_entity_aliases": args.anonymous_entity_aliases,
        "natural_grouped_history": args.natural_grouped_history,
        "natural_activity_summary": args.natural_activity_summary,
        "natural_neighbor_names_only": args.natural_neighbor_names_only,
        "natural_activity_compact_top3": args.natural_activity_compact_top3,
        "natural_activity_top_k": args.natural_activity_top_k,
        "ablate_mutual_history": args.ablate_mutual_history,
        "ablate_common_neighbors": args.ablate_common_neighbors,
        "ablate_source_history": args.ablate_source_history,
        "ablate_target_history": args.ablate_target_history,
        "ablate_source_target_history": args.ablate_source_target_history,
        "ablate_reasoning_guidance": args.ablate_reasoning_guidance,
        "prompt_variant": args.vllm_prompt_variant,
        "graph_prompt_special_token": graph_prompt_special_token,
        "graph_prompt_num_tokens": graph_prompt_num_tokens,
        "max_length": args.max_length,
    }

    train_examples, train_stats = build_prompt_response_examples(
        samples=train_samples,
        tokenizer=tokenizer_for_prompts,
        entity_map=prompt_entity_map,
        relation_map=relation_map,
        include_graph_pair_feature=args.graph_prompt_mlp_only,
        graph_pair_feature_key=GRAPH_PAIR_FEATURE_KEY,
        **prompt_kwargs,
    )
    val_examples, val_stats = build_prompt_response_examples(
        samples=val_samples,
        tokenizer=tokenizer_for_prompts,
        entity_map=prompt_entity_map,
        relation_map=relation_map,
        include_graph_pair_feature=args.graph_prompt_mlp_only,
        graph_pair_feature_key=GRAPH_PAIR_FEATURE_KEY,
        **prompt_kwargs,
    )

    dataset_stats = {
        "train": train_stats,
        "validation": val_stats,
        "train_sampling": train_sampling_stats,
    }
    if train_selection_stats is not None:
        dataset_stats["train_hybrid_middle_selection"] = train_selection_stats
    if dist_ctx["is_main_process"]:
        save_json(os.path.join(args.output_dir, "dataset_stats.json"), dataset_stats)
        if graph_feature_meta is not None:
            save_json(os.path.join(args.output_dir, "graph_feature_stats.json"), graph_feature_meta)

    eval_samples = None
    eval_rrf_baseline = None
    train_rrf_baseline = None
    if dist_ctx["is_main_process"]:
        train_rrf_baseline = compute_rrf_baseline_metrics(
            train_samples,
            dtgb_eval_batch_size=args.dtgb_eval_batch_size,
            negative_ratio=args.negative_ratio,
        )
    if (args.eval_before_training or args.eval_after_training) and dist_ctx["is_main_process"]:
        post_train_eval_num_samples = (
            args.val_num_samples
            if args.post_train_eval_num_samples is None
            else int(args.post_train_eval_num_samples)
        )
        if args.post_train_eval_split == "validation":
            eval_samples = val_samples
        else:
            eval_samples = create_test_samples(
                edges,
                prompt_entity_map,
                relation_map,
                test_ratio=args.test_ratio,
                val_ratio=args.val_ratio,
                num_samples=post_train_eval_num_samples,
                negative_ratio=args.negative_ratio,
                random_seed=args.seed,
                history_window=args.history_window,
                semantic_history=args.semantic_history,
                semantic_topk=args.semantic_topk,
                semantic_history_entity_mode=args.semantic_history_entity_mode,
                common_neighbors_semantic=args.common_neighbors_semantic,
                build_prompt_features=True,
                semantic_use_smoothing=semantic_use_smoothing,
                semantic_hub_penalty_alpha=args.semantic_hub_penalty_alpha,
                semantic_fusion_alpha=args.semantic_fusion_alpha,
                semantic_fusion_tau=args.semantic_fusion_tau,
                semantic_fusion_recency_speed=args.semantic_fusion_recency_speed,
                history_pool_size=args.history_pool_size,
                history_pool_window=args.history_pool_window,
                history_preserve_recent_k=args.history_preserve_recent_k,
                embeddings=embeddings,
                entity_id_to_idx=entity_id_to_idx,
                embedding_model=args.embedding_model,
                embedding_cache=args.embedding_cache,
                compute_expert_prediction=runtime_compute_expert_prediction,
                compute_rrf_scores=compute_rrf_scores,
                rrf_k=args.rrf_k,
                rrf_mode=args.rrf_mode,
                sequential_rank_bins=args.sequential_rank_bins,
                expert_prediction_mode=runtime_expert_prediction_mode,
                expert_prediction_fixed_threshold=args.expert_prediction_fixed_threshold,
                rrf_pointwise_pool_size=args.rrf_pointwise_pool_size,
                rrf_pointwise_num_pools=args.rrf_pointwise_num_pools,
                rrf_batch_size=args.rrf_batch_size,
                eval_split=args.post_train_eval_split,
                key_signal_reference=runtime_key_signal_reference,
                key_signal_fields=args.key_signal_fields,
                skip_key_signal_calibration=runtime_skip_key_signal_calibration,
                include_overall_structural_signal=runtime_include_overall_structural_signal,
                overall_signal_low_threshold=args.overall_signal_low_threshold,
                overall_signal_high_threshold=args.overall_signal_high_threshold,
                apply_gdelt_time_bucket=args.apply_gdelt_time_bucket,
                heuristic_recent_degree_window=args.heuristic_recent_degree_window,
            )
            apply_runtime_key_signal_annotations(eval_samples)
            apply_runtime_expert_signal_annotations(eval_samples)
            if args.graph_prompt_mlp_only:
                attach_hadamard_graph_pair_features(
                    eval_samples,
                    embeddings=embeddings,
                    entity_id_to_idx=entity_id_to_idx,
                    feature_key=GRAPH_PAIR_FEATURE_KEY,
                )
        eval_rrf_baseline = compute_rrf_baseline_metrics(
            eval_samples,
            dtgb_eval_batch_size=args.dtgb_eval_batch_size,
            negative_ratio=args.negative_ratio,
        )

    if semantic_backbone_scorer is not None:
        del semantic_backbone_scorer
        semantic_backbone_scorer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not train_examples:
        raise RuntimeError("No train examples were produced after tokenization.")

    if args.dry_run:
        if dist_ctx["is_main_process"]:
            print("Dry run complete. Dataset stats:")
            print(dataset_stats)
        return

    if args.graph_prompt_mlp_only:
        model, tokenizer = load_frozen_base_model(
            model_path=args.model_path,
            tokenizer=tokenizer_for_prompts,
        )
        if not train_examples:
            raise RuntimeError("No train examples found for graph prompt mode.")
        first_feature = train_examples[0].get(GRAPH_PAIR_FEATURE_KEY)
        if first_feature is None:
            raise RuntimeError(
                f"Missing {GRAPH_PAIR_FEATURE_KEY!r} in tokenized train examples for graph prompt mode."
            )
        graph_feature_dim = int(np.asarray(first_feature).shape[-1])
        model = GraphPromptMLPOnlyModel(
            base_model=model,
            graph_feature_dim=graph_feature_dim,
            num_virtual_tokens=args.graph_prompt_num_tokens,
            hidden_dim=args.graph_prompt_hidden_dim,
            dropout=args.graph_prompt_dropout,
            output_l2_norm=args.graph_prompt_output_l2_norm,
            zero_init_output=args.graph_prompt_zero_init_output,
            mean_token_init=args.graph_prompt_mean_token_init,
            token_space_align_weight=args.graph_prompt_token_space_align_weight,
            graph_prompt_injection_mode=(
                "dedicated_slot" if args.graph_prompt_dedicated_slot else "prepend"
            ),
            graph_prompt_special_token=graph_prompt_special_token,
            graph_prompt_special_token_id=(
                tokenizer.convert_tokens_to_ids(graph_prompt_special_token)
                if graph_prompt_special_token
                else None
            ),
            freeze_base_model=(not args.graph_prompt_unfreeze_base_model),
        )
        if graph_projector_init_path is not None:
            projector_state = torch.load(graph_projector_init_path, map_location="cpu")
            if not isinstance(projector_state, dict):
                raise RuntimeError("graph_projector.pt must contain a state_dict.")
            model.graph_projector.load_state_dict(projector_state, strict=True)
        if args.graph_prompt_freeze_projector:
            for param in model.graph_projector.parameters():
                param.requires_grad = False
        if args.graph_prompt_unfreeze_base_model:
            model.base_model.config.use_cache = False
            if not args.disable_gradient_checkpointing:
                model.base_model.gradient_checkpointing_enable()
                if hasattr(model.base_model, "enable_input_require_grads"):
                    model.base_model.enable_input_require_grads()
    else:
        lora_target_modules = _resolve_lora_target_modules(args.lora_target_modules)
        model, tokenizer = load_peft_training_model(
            model_path=args.model_path,
            tokenizer=tokenizer_for_prompts,
            load_in_4bit=args.load_in_4bit,
            gradient_checkpointing=(not args.disable_gradient_checkpointing),
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=lora_target_modules,
            lora_last_n_layers=args.lora_last_n_layers,
        )

    trainable_summary = summarize_trainable_parameters(model)
    if dist_ctx["is_main_process"]:
        save_json(os.path.join(args.output_dir, "trainable_parameters.json"), trainable_summary)
        print(trainable_summary)

    trainer = RollingTrainMetricTrainer(
        model=model,
        args=_build_training_args(args, has_eval_dataset=bool(val_examples)),
        train_dataset=PromptResponseDataset(train_examples),
        eval_dataset=PromptResponseDataset(val_examples) if val_examples else None,
        data_collator=PromptResponseCollator(tokenizer),
        tokenizer_for_metrics=tokenizer,
        enable_live_train_metrics=(not args.disable_live_train_auc),
    )

    pre_train_metrics = {}
    if args.eval_before_training:
        def _compute_pre_train_metrics():
            if not dist_ctx["is_main_process"]:
                return {}
            metrics = evaluate_binary_completion_likelihood(
                model=trainer.model,
                tokenizer=tokenizer,
                samples=eval_samples,
                entity_map=prompt_entity_map,
                relation_map=relation_map,
                negative_ratio=args.negative_ratio,
                batch_size=args.eval_prompt_batch_size,
                dtgb_eval_batch_size=args.dtgb_eval_batch_size,
                **prompt_kwargs,
            )
            metrics["eval_split"] = args.post_train_eval_split
            metrics["eval_seed"] = int(args.seed)
            metrics["rrf_baseline"] = dict(eval_rrf_baseline or {})
            pre_payload = (
                metrics
                if args.include_prediction_vectors
                else _strip_prediction_vectors(metrics)
            )
            save_json(os.path.join(args.output_dir, "pre_validation_metrics.json"), pre_payload)
            print({"pre_validation_metrics": _metric_summary_for_console(metrics)})
            return metrics

        pre_train_metrics = _run_main_process_only(_compute_pre_train_metrics, enabled=True) or {}

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    _maybe_barrier()

    adapter_dir = os.path.join(args.output_dir, "adapter")
    graph_projector_dir = os.path.join(args.output_dir, "graph_projector")
    if args.graph_prompt_mlp_only:
        if dist_ctx["is_main_process"]:
            unwrapped_model = trainer.model
            if hasattr(unwrapped_model, "module"):
                unwrapped_model = unwrapped_model.module
            if not isinstance(unwrapped_model, GraphPromptMLPOnlyModel):
                raise RuntimeError("Expected GraphPromptMLPOnlyModel in graph prompt mode.")
            graph_projector_base_model_path = args.model_path
            if args.graph_prompt_unfreeze_base_model:
                base_model_dir = os.path.join(args.output_dir, "base_model")
                unwrapped_model.base_model.save_pretrained(
                    base_model_dir,
                    safe_serialization=True,
                )
                tokenizer.save_pretrained(base_model_dir)
                graph_projector_base_model_path = os.path.abspath(base_model_dir)
            unwrapped_model.save_graph_projector(
                graph_projector_dir,
                base_model_path=graph_projector_base_model_path,
            )
            tokenizer.save_pretrained(graph_projector_dir)
    else:
        trainer.save_model(adapter_dir)
        if dist_ctx["is_main_process"]:
            tokenizer.save_pretrained(adapter_dir)
    _maybe_barrier()

    eval_model = trainer.model
    if hasattr(eval_model, "module"):
        eval_model = eval_model.module

    post_train_train_metrics = {}
    if not args.disable_live_train_auc:
        def _compute_post_train_train_metrics():
            if not dist_ctx["is_main_process"]:
                return {}
            metrics = evaluate_binary_completion_likelihood(
                model=eval_model,
                tokenizer=tokenizer,
                samples=train_samples,
                entity_map=prompt_entity_map,
                relation_map=relation_map,
                negative_ratio=args.negative_ratio,
                batch_size=args.eval_prompt_batch_size,
                dtgb_eval_batch_size=args.dtgb_eval_batch_size,
                **prompt_kwargs,
            )
            metrics["eval_split"] = args.train_split_name
            metrics["eval_seed"] = int(args.seed)
            metrics["rrf_baseline"] = dict(train_rrf_baseline or {})
            train_payload = (
                metrics
                if args.include_prediction_vectors
                else _strip_prediction_vectors(metrics)
            )
            save_json(os.path.join(args.output_dir, "train_metrics.json"), train_payload)
            print({"post_train_train_metrics": _metric_summary_for_console(metrics)})
            return metrics

        post_train_train_metrics = (
            _run_main_process_only(_compute_post_train_train_metrics, enabled=True) or {}
        )

    post_train_metrics = {}
    if args.eval_after_training:
        def _compute_post_train_metrics():
            if not dist_ctx["is_main_process"]:
                return {}
            if eval_samples is None:
                raise RuntimeError("Post-training evaluation samples were not prepared.")
            metrics = evaluate_binary_completion_likelihood(
                model=eval_model,
                tokenizer=tokenizer,
                samples=eval_samples,
                entity_map=prompt_entity_map,
                relation_map=relation_map,
                negative_ratio=args.negative_ratio,
                batch_size=args.eval_prompt_batch_size,
                dtgb_eval_batch_size=args.dtgb_eval_batch_size,
                **prompt_kwargs,
            )
            metrics["eval_split"] = args.post_train_eval_split
            metrics["eval_seed"] = int(args.seed)
            metrics["rrf_baseline"] = dict(eval_rrf_baseline or {})
            post_payload = (
                metrics
                if args.include_prediction_vectors
                else _strip_prediction_vectors(metrics)
            )
            save_json(os.path.join(args.output_dir, "validation_metrics.json"), post_payload)
            print({"post_validation_metrics": _metric_summary_for_console(metrics)})
            return metrics

        post_train_metrics = _run_main_process_only(_compute_post_train_metrics, enabled=True) or {}

    if dist_ctx["is_main_process"] and (pre_train_metrics or post_train_metrics or post_train_train_metrics):
        eval_summary = {
            "post_train_train_metrics": (
                post_train_train_metrics
                if args.include_prediction_vectors
                else _strip_prediction_vectors(post_train_train_metrics)
            )
            if post_train_train_metrics
            else None,
            "pre_train_metrics": (
                pre_train_metrics
                if args.include_prediction_vectors
                else _strip_prediction_vectors(pre_train_metrics)
            )
            if pre_train_metrics
            else None,
            "post_train_metrics": (
                post_train_metrics
                if args.include_prediction_vectors
                else _strip_prediction_vectors(post_train_metrics)
            )
            if post_train_metrics
            else None,
        }
        if pre_train_metrics and post_train_metrics:
            eval_summary["delta_post_minus_pre"] = {
                "auc": float(post_train_metrics["auc"] - pre_train_metrics["auc"]),
                "ap": float(post_train_metrics["ap"] - pre_train_metrics["ap"]),
                "accuracy": float(post_train_metrics["accuracy"] - pre_train_metrics["accuracy"]),
                "auc_global": float(post_train_metrics["auc_global"] - pre_train_metrics["auc_global"]),
                "ap_global": float(post_train_metrics["ap_global"] - pre_train_metrics["ap_global"]),
            }
        save_json(os.path.join(args.output_dir, "eval_comparison.json"), eval_summary)
    _maybe_barrier()

    if args.merge_adapter and dist_ctx["is_main_process"]:
        if args.graph_prompt_mlp_only:
            print(
                "Skipped --merge_adapter in --graph_prompt_mlp_only mode: "
                "no LoRA adapter is trained; projector is saved under graph_projector/."
            )
        else:
            merged_dir = os.path.join(args.output_dir, "merged")
            merge_peft_adapter(
                base_model_path=args.model_path,
                adapter_path=adapter_dir,
                output_dir=merged_dir,
            )
            print(f"Merged model saved to {merged_dir}")

    if (not pre_train_metrics) and (not post_train_metrics) and (not post_train_train_metrics) and dist_ctx["is_main_process"]:
        save_json(os.path.join(args.output_dir, "training_complete.json"), {"status": "ok"})
    _maybe_destroy_process_group()


if __name__ == "__main__":
    main()
