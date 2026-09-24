#!/usr/bin/env python3
"""
LLM-based Link Prediction Evaluation for DTGB datasets.
Following DTGB standard protocol: 70% train, 15% val, 15% test
Metrics: AP (Average Precision) and AUC-ROC
"""
import json
import os
import sys
import time

import numpy as np
import torch

_EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_EXPERIMENTS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.modules.llm_lp.data_parallel import (
    cleanup_data_parallel_run,
    init_data_parallel_context,
    merge_hybrid_selected_shards,
    merge_trial_shards,
    split_samples_for_rank,
    wait_for_rank_completions,
    wait_for_shared_payload,
    wait_for_hybrid_selected_shards,
    wait_for_hybrid_selection,
    wait_for_shared_samples,
    wait_for_trial_shards,
    write_shared_payload,
    write_hybrid_selected_shard,
    write_hybrid_selection,
    write_rank_completion,
    write_shared_samples,
    write_trial_shard,
)
from experiments.modules.llm_lp.history_compaction import compact_history_local
from experiments.modules.llm_lp.eval_helpers import load_or_compute_embeddings
from experiments.modules.llm_lp.cli import build_arg_parser, validate_args
from experiments.modules.llm_lp.eval import (
    evaluate,
    evaluate_rrf_only,
    evaluate_transformers,
    load_model_and_tokenizer,
    load_model_transformers,
    maybe_spawn_local_dp_workers,
    sanitize_torch_distributed_env_for_vllm,
    write_prediction_debug_log,
)
from experiments.modules.llm_lp.experiment import (
    _build_lightweight_rrf_samples_for_split,
    _build_expert_prediction_discriminative_debug,
    _apply_validation_sampled_key_signal_reference,
    _apply_validation_sampled_binary_labels,
    _apply_validation_sampled_threeway_labels,
    _build_key_signal_discriminative_debug,
    _build_realized_discriminative_debug,
    _calibrate_validation_sampled_hybrid_gmm_overlap_band,
    _calibrate_validation_sampled_hybrid_uncertainty_band,
    _calibrate_validation_sampled_key_signal_reference,
    _calibrate_validation_sampled_threeway_thresholds,
    _uses_validation_sampled_binary,
    _uses_validation_sampled_key_signal_reference,
    _uses_validation_sampled_threeway,
    _resolve_hybrid_alignment_calibration_sizes,
    apply_hybrid_debug_route_cap,
    build_auto_summary_entity_map,
    build_output_data,
    build_prompt_entity_map,
    build_trial_result_summary,
    configure_runtime_logging,
    dataset_uses_dtgb_time_bucket,
    entity_name_mode_uses_compact_profile,
    load_few_shot_examples,
    load_dataset_data,
    load_summary_entity_map,
    maybe_print_token_usage_summary,
    normalize_hybrid_selection,
    print_run_config,
)
from experiments.modules.llm_lp.prompt_context import materialize_samples_prompt_context
from experiments.modules.llm_lp.sample_builder import create_test_samples
from experiments.modules.llm_lp.sample_finalize import (
    apply_overall_structural_signal_buckets,
    assign_expert_prediction_labels,
    finalize_test_samples,
)
from experiments.modules.llm_lp.semantic_subprocess import SubprocessSemanticMLPScorer
from experiments.modules.llm_lp.split_runner import (
    SplitExecutionContext,
    execute_split,
)
from experiments.modules.llm_lp.openai_eval import (
    build_openai_client,
    evaluate_openai_chat,
)
from experiments.modules.prediction_metrics import compute_prediction_metrics
from experiments.modules.rrf.analysis import (
    select_rrf_middle_sample_indices_validation_sampled_gmm_overlap,
    select_rrf_middle_sample_indices_validation_sampled_band,
    select_rrf_middle_sample_indices_pointwise_threshold_band,
    select_topk_rrf_middle_sample_indices,
)
from experiments.modules.rrf.hybrid import (
    fit_isotonic_score_alignment,
    evaluate_budgeted_hybrid_rrf_llm,
    fit_quantile_score_alignment,
    prepare_backbone_score_space,
)


def run_evaluation(args):
    if args.capture_prompt_embeddings and "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        print(
            "Prompt embedding capture: setting VLLM_WORKER_MULTIPROC_METHOD=spawn "
            "to avoid inheriting an initialized CUDA runtime through fork."
        )
    apply_dtgb_time_bucket = dataset_uses_dtgb_time_bucket(args.dataset_name)
    if (
        entity_name_mode_uses_compact_profile(args.dataset_name, args.entity_name_mode)
        and str(args.summary_mode).strip().lower() == "off"
    ):
        args.summary_mode = "full"
    use_raw_key_signals = args.key_signal_mode == "raw"
    use_percentile_key_signals = args.key_signal_mode == "percentile"
    selected_key_signal_fields = tuple(args.key_signal_fields)

    dp_ctx = init_data_parallel_context(
        args.output,
        args.data_parallel_size,
        sync_dir=args.data_parallel_sync_dir,
    )
    if dp_ctx["enabled"] and args.tensor_parallel_size != 1:
        raise ValueError("Data parallel mode requires --tensor_parallel_size 1.")
    if dp_ctx["enabled"]:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            devices = [device.strip() for device in visible.split(",") if device.strip()]
            if len(devices) > 1:
                if dp_ctx["rank"] >= len(devices):
                    raise RuntimeError(
                        f"DP rank {dp_ctx['rank']} exceeds visible device list {devices}."
                    )
                os.environ["CUDA_VISIBLE_DEVICES"] = devices[dp_ctx["rank"]]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(dp_ctx["rank"])

    if args.debug_prediction_log and (not dp_ctx["enabled"] or dp_ctx["rank"] == 0):
        debug_dir = os.path.dirname(os.path.abspath(args.debug_prediction_log))
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
        with open(args.debug_prediction_log, "w"):
            pass
        print(f"Prediction debug log path: {args.debug_prediction_log}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    semantic_use_smoothing = not args.semantic_history_no_smoothing
    few_shot_examples = None
    if args.few_shot_path:
        few_shot_examples = load_few_shot_examples(
            args.few_shot_path,
            max_examples=args.few_shot_max_examples,
        )
    print_run_config(
        args=args,
        dp_ctx=dp_ctx,
        semantic_use_smoothing=semantic_use_smoothing,
        use_raw_key_signals=use_raw_key_signals,
        use_percentile_key_signals=use_percentile_key_signals,
        few_shot_examples_count=len(few_shot_examples) if few_shot_examples else 0,
    )

    if dp_ctx["enabled"] and (not args.use_transformers) and (not args.use_openai_api) and (not args.rrf_only) and (not args.export_prompt_dataset_dir):
        sanitize_torch_distributed_env_for_vllm()

    model = None
    tokenizer = None
    llm = None
    openai_client = None
    if args.export_prompt_dataset_dir:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )
        print(
            "Prompt-dataset export mode: tokenizer loaded; model inference is disabled."
        )
    elif args.rrf_only:
        print("RRF-only mode: skipping model load and LLM inference.")
    elif args.use_openai_api:
        openai_client = build_openai_client(
            api_key=args.openai_api_key,
            api_key_env=args.openai_api_key_env,
            base_url=args.openai_base_url,
            timeout_sec=args.openai_timeout_sec,
            max_retries=args.openai_max_retries,
        )
        print(f"OpenAI client ready (model={args.openai_model}).")
    elif args.use_transformers:
        model, tokenizer = load_model_transformers(
            args.model_path,
            quantization=args.quantization,
        )
    else:
        llm, tokenizer = load_model_and_tokenizer(
            args.model_path,
            args.quantization,
            args.tensor_parallel_size,
            args.max_model_len,
            args.gpu_utilization,
            args.enforce_eager_vllm,
            capture_prompt_embeddings=args.capture_prompt_embeddings,
            prompt_embedding_storage_dir=args.prompt_embedding_storage_dir,
            prompt_embedding_layer=args.prompt_embedding_layer,
        )

    edges, entity_map, relation_map = load_dataset_data(
        args.dataset_name,
        entity_text_path=args.entity_text_path,
        relation_text_path=args.relation_text_path,
    )
    prompt_entity_map = build_prompt_entity_map(
        args.dataset_name,
        entity_map,
        entity_name_mode=args.entity_name_mode,
        disable_stack_elec_prompt_cleaning=args.disable_stack_elec_prompt_cleaning,
    )
    summary_map = None
    if args.summary_mode != "off":
        if args.summary_entity_text_path:
            summary_map = load_summary_entity_map(args.summary_entity_text_path)
            summary_source = args.summary_entity_text_path
        else:
            summary_map = build_auto_summary_entity_map(args.dataset_name, entity_map)
            summary_source = "auto_compact_profiles"
            if summary_map is None:
                raise ValueError(
                    "--summary_entity_text_path is required when --summary_mode is not off "
                    f"for dataset {args.dataset_name}"
                )
        graph_node_ids = set(edges["u"].astype(int)).union(set(edges["i"].astype(int)))
        covered = len(graph_node_ids.intersection(set(summary_map.keys())))
        print(
            "Summary prompt mode: "
            f"{args.summary_mode} ({covered}/{len(graph_node_ids)} graph nodes covered), "
            f"max_chars={args.summary_max_chars}, source={summary_source}"
        )

    embeddings = None
    entity_id_to_idx = None
    expert_source_is_semantic = str(args.expert_prediction_source).strip().lower() == "semantic_mlp"
    expert_score_field = "semantic_mlp_score" if expert_source_is_semantic else "rrf_score"
    expert_score_label = "Semantic MLP" if expert_source_is_semantic else "RRF"
    hybrid_backbone_is_semantic = str(args.hybrid_backbone).strip().lower() == "semantic_mlp"
    hybrid_backbone_score_field = (
        "semantic_mlp_score" if hybrid_backbone_is_semantic else "rrf_score"
    )
    hybrid_backbone_label = "Semantic MLP" if hybrid_backbone_is_semantic else "RRF"
    rrf_uses_semantic_smoothing = "semantic_smoothing" in args.rrf_heuristics
    expert_signal_visible = bool((not args.hide_expert_prediction) or args.include_overall_structural_signal)
    need_semantic_expert_scorer_this_rank = bool(expert_source_is_semantic and expert_signal_visible)
    need_semantic_hybrid_scorer_this_rank = bool(
        hybrid_backbone_is_semantic
        and (not dp_ctx["enabled"] or dp_ctx["rank"] == 0)
    )
    need_embedding_preload = (
        (
            args.semantic_history
            or args.semantic_history_entity_mode
            or args.common_neighbors_semantic
            or need_semantic_hybrid_scorer_this_rank
            or need_semantic_expert_scorer_this_rank
            or rrf_uses_semantic_smoothing
        )
        and (
            (not args.rrf_only)
            or rrf_uses_semantic_smoothing
            or need_semantic_hybrid_scorer_this_rank
            or need_semantic_expert_scorer_this_rank
        )
    )
    if need_embedding_preload:
        embeddings, entity_id_to_idx = load_or_compute_embeddings(
            entity_map,
            args.embedding_model,
            args.embedding_cache,
        )

    semantic_mlp_embeddings = embeddings
    if (
        need_semantic_hybrid_scorer_this_rank
        or need_semantic_expert_scorer_this_rank
    ) and (
        args.semantic_mlp_embedding_model is not None
        or args.semantic_mlp_embedding_cache is not None
    ):
        semantic_mlp_embeddings, _ = load_or_compute_embeddings(
            entity_map,
            args.semantic_mlp_embedding_model or args.embedding_model,
            args.semantic_mlp_embedding_cache,
        )
        print(
            "Semantic MLP uses a dedicated embedding source: "
            f"model={args.semantic_mlp_embedding_model or args.embedding_model}, "
            f"cache={args.semantic_mlp_embedding_cache or 'none'}, "
            f"shape={tuple(semantic_mlp_embeddings.shape)}"
        )

    eval_splits = ["transductive", "inductive"] if args.eval_split == "both" else [args.eval_split]
    use_two_phase_postprocessing = (
        len(eval_splits) > 1 and not args.legacy_interleaved_split_processing
    )
    all_trial_results = {split_name: [] for split_name in eval_splits}
    validation_sampled_calibration_by_trial = []
    save_eval_details = bool(args.debug_prediction_log)
    use_validation_sampled_threeway = _uses_validation_sampled_threeway(args.expert_prediction_mode)
    use_validation_sampled_binary = _uses_validation_sampled_binary(args.expert_prediction_mode)
    use_validation_sampled_key_signals = _uses_validation_sampled_key_signal_reference(
        args.key_signal_reference
    )
    hybrid_selection_mode_key = str(args.hybrid_selection_mode).strip().lower()
    use_tabicl_router = hybrid_selection_mode_key in {
        "tabicl_router",
        "validation_fitted_tabicl_router",
    }
    use_validation_sampled_hybrid_band = (
        hybrid_selection_mode_key == "validation_sampled_uncertainty_band"
    )
    use_validation_sampled_hybrid_gmm = (
        hybrid_selection_mode_key == "validation_sampled_gmm_overlap_band"
    )
    hybrid_enabled = bool(
        args.hybrid_uncertain_topk_queries > 0
        or hybrid_selection_mode_key == "pointwise_fixed_threshold_band"
        or hybrid_selection_mode_key == "random_sample"
        or hybrid_selection_mode_key == "learned_router_top_fraction"
        or use_tabicl_router
        or use_validation_sampled_hybrid_band
        or use_validation_sampled_hybrid_gmm
    )
    semantic_backbone_scorer = None
    if hybrid_enabled:
        print(f"Hybrid backbone: {hybrid_backbone_label}")
    if not args.hide_expert_prediction or args.include_overall_structural_signal:
        print(f"Prior-signal source: {expert_score_label}")
    if need_semantic_hybrid_scorer_this_rank or need_semantic_expert_scorer_this_rank:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Semantic MLP structural scoring requires CUDA. "
                "No GPU is available in the current runtime."
            )
        requested_roles = []
        if need_semantic_expert_scorer_this_rank:
            requested_roles.append("prior-signal")
        if need_semantic_hybrid_scorer_this_rank:
            requested_roles.append("hybrid-backbone")
        print(
            "Loading semantic MLP structural scorer: "
            f"roles={','.join(requested_roles)}, checkpoint={args.semantic_mlp_checkpoint} on cuda"
        )
        if args.semantic_mlp_score_python:
            semantic_backbone_scorer = SubprocessSemanticMLPScorer(
                python_path=args.semantic_mlp_score_python,
                dataset_name=args.dataset_name,
                checkpoint_path=args.semantic_mlp_checkpoint,
                embeddings=semantic_mlp_embeddings,
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
                embeddings=semantic_mlp_embeddings,
                entity_ids_sorted=sorted(entity_map.keys()),
                device="cuda",
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                eval_positive_batch_size=args.dtgb_eval_batch_size,
                source_init_override=args.semantic_mlp_source_init_override,
                temporal_mode=args.semantic_mlp_temporal_mode,
            )
        print(
            "Semantic structural scorer ready: "
            f"field={semantic_backbone_scorer.score_field}, "
            f"scorer={semantic_backbone_scorer.scorer_type}, "
            f"checkpoint={args.semantic_mlp_checkpoint}"
        )

    def annotate_semantic_structural_scores(
        samples,
        *,
        active_split="unknown",
        score_field,
        score_label,
        purpose_label,
    ):
        if not samples:
            return samples
        print(
            f"Checking semantic {purpose_label} scores: "
            f"{len(samples)} samples, field={score_field}, split={active_split}.",
            flush=True,
        )
        existing_count = sum(1 for sample in samples if sample.get(score_field) is not None)
        if existing_count == len(samples):
            print(
                f"Semantic {purpose_label} scores already present: "
                f"{existing_count}/{len(samples)} samples.",
                flush=True,
            )
            return samples
        if semantic_backbone_scorer is None:
            raise RuntimeError(
                f"Semantic {purpose_label} scoring requested but scorer is unavailable."
            )
        print(
            f"Scoring semantic {purpose_label} samples: "
            f"{len(samples)} samples with {score_label}; "
            f"missing={len(samples) - existing_count} "
            f"for split={active_split}."
            ,
            flush=True,
        )
        semantic_score_t0 = time.perf_counter()
        semantic_backbone_scorer.annotate_samples(
            samples,
            score_field=score_field,
        )
        print(
            f"Semantic {purpose_label} scoring complete: "
            f"split={active_split}, samples={len(samples)}, "
            f"elapsed={time.perf_counter() - semantic_score_t0:.1f}s.",
            flush=True,
        )
        return samples

    def annotate_hybrid_backbone_scores(samples, *, active_split="unknown", score_field=None):
        if not hybrid_enabled or not hybrid_backbone_is_semantic or not samples:
            return samples
        return annotate_semantic_structural_scores(
            samples,
            active_split=active_split,
            score_field=(score_field or hybrid_backbone_score_field),
            score_label=hybrid_backbone_label,
            purpose_label="hybrid-backbone",
        )

    def annotate_expert_source_scores(samples, *, active_split="unknown", score_field=None):
        if not expert_source_is_semantic or not samples:
            return samples
        return annotate_semantic_structural_scores(
            samples,
            active_split=active_split,
            score_field=(score_field or expert_score_field),
            score_label=expert_score_label,
            purpose_label="prior-signal",
        )

    for trial in range(args.num_trials):
        trial_semantic_backbone_baseline_printed = set()

        def maybe_print_semantic_hybrid_backbone_baseline(samples, *, active_split="unknown"):
            if (not hybrid_backbone_is_semantic) or (not samples):
                return
            baseline_key = (int(trial), str(active_split))
            if baseline_key in trial_semantic_backbone_baseline_printed:
                return
            if any(sample.get(hybrid_backbone_score_field) is None for sample in samples):
                return
            predictions = np.asarray(
                [float(sample.get(hybrid_backbone_score_field, 0.0)) for sample in samples],
                dtype=np.float64,
            )
            labels = np.asarray([int(sample["label"]) for sample in samples], dtype=np.int64)
            metrics = compute_prediction_metrics(
                predictions,
                labels,
                dtgb_eval_batch_size=args.dtgb_eval_batch_size,
            )
            print(
                f"Semantic MLP immediate baseline [{active_split}]: "
                f"AP={metrics['ap']:.4f}, "
                f"AUC={metrics['auc']:.4f}, "
                f"Acc={metrics['accuracy']:.4f}, "
                f"AP_GLOBAL={metrics.get('ap_global', metrics['ap']):.4f}, "
                f"AUC_GLOBAL={metrics.get('auc_global', metrics['auc']):.4f}"
            )
            trial_semantic_backbone_baseline_printed.add(baseline_key)

        current_seed = args.seed + trial
        print(f"\n{'-' * 40}")
        print(f"Trial {trial + 1}/{args.num_trials} (Seed: {current_seed})")
        print("-" * 40)

        trial_validation_sampled_calibration = None
        trial_validation_sampled_key_signal_reference = None
        trial_validation_sampled_hybrid_band = None
        trial_hybrid_score_alignment = None
        need_threeway_calibration = (use_validation_sampled_threeway or use_validation_sampled_binary or args.enable_rrf_validation_band_debug) and (
            (not args.hide_expert_prediction) or args.include_overall_structural_signal
        )
        need_key_signal_calibration = use_validation_sampled_key_signals and (
            (not args.hide_key_signals) and (args.key_signal_mode != "raw")
        )
        need_hybrid_band_calibration = bool(
            hybrid_enabled
            and (
                use_validation_sampled_hybrid_band
                or use_validation_sampled_hybrid_gmm
                or use_tabicl_router
            )
        )
        if dp_ctx["enabled"]:
            calibration_payload_name = "validation_sampled_threeway"
            key_signal_payload_name = "validation_sampled_key_signal_reference"
            hybrid_payload_name = "validation_sampled_hybrid_band"
            if dp_ctx["rank"] == 0:
                if need_threeway_calibration:
                    trial_validation_sampled_calibration = _calibrate_validation_sampled_threeway_thresholds(
                        args=args,
                        edges=edges,
                        embeddings=embeddings,
                        entity_id_to_idx=entity_id_to_idx,
                        random_seed=current_seed,
                        score_field=expert_score_field,
                        score_label=expert_score_label,
                        score_samples_fn=(
                            annotate_expert_source_scores if expert_source_is_semantic else None
                        ),
                    )
                    write_shared_payload(
                        sync_dir=dp_ctx["sync_dir"],
                        run_id=dp_ctx["run_id"],
                        trial_idx=trial,
                        payload_name=calibration_payload_name,
                        payload=trial_validation_sampled_calibration,
                    )
                if need_key_signal_calibration:
                    trial_validation_sampled_key_signal_reference = (
                        _calibrate_validation_sampled_key_signal_reference(
                            args=args,
                            edges=edges,
                            embeddings=embeddings,
                            entity_id_to_idx=entity_id_to_idx,
                            random_seed=current_seed,
                        )
                    )
                    write_shared_payload(
                        sync_dir=dp_ctx["sync_dir"],
                        run_id=dp_ctx["run_id"],
                        trial_idx=trial,
                        payload_name=key_signal_payload_name,
                        payload=trial_validation_sampled_key_signal_reference,
                    )
                if need_hybrid_band_calibration:
                    calibrate_hybrid_band_fn = (
                        _calibrate_validation_sampled_hybrid_gmm_overlap_band
                        if use_validation_sampled_hybrid_gmm
                        else _calibrate_validation_sampled_hybrid_uncertainty_band
                    )
                    trial_validation_sampled_hybrid_band = calibrate_hybrid_band_fn(
                        args=args,
                        edges=edges,
                        embeddings=embeddings,
                        entity_id_to_idx=entity_id_to_idx,
                        random_seed=current_seed,
                        score_field=hybrid_backbone_score_field,
                        score_label=hybrid_backbone_label,
                        score_samples_fn=(
                            annotate_hybrid_backbone_scores
                            if hybrid_backbone_is_semantic
                            else None
                        ),
                    )
                    write_shared_payload(
                        sync_dir=dp_ctx["sync_dir"],
                        run_id=dp_ctx["run_id"],
                        trial_idx=trial,
                        payload_name=hybrid_payload_name,
                        payload=trial_validation_sampled_hybrid_band,
                    )
            else:
                if need_threeway_calibration:
                    trial_validation_sampled_calibration = wait_for_shared_payload(
                        sync_dir=dp_ctx["sync_dir"],
                        run_id=dp_ctx["run_id"],
                        trial_idx=trial,
                        payload_name=calibration_payload_name,
                        timeout_sec=dp_ctx["timeout_sec"],
                    )
                if need_key_signal_calibration:
                    trial_validation_sampled_key_signal_reference = wait_for_shared_payload(
                        sync_dir=dp_ctx["sync_dir"],
                        run_id=dp_ctx["run_id"],
                        trial_idx=trial,
                        payload_name=key_signal_payload_name,
                        timeout_sec=dp_ctx["timeout_sec"],
                    )
                if need_hybrid_band_calibration:
                    trial_validation_sampled_hybrid_band = wait_for_shared_payload(
                        sync_dir=dp_ctx["sync_dir"],
                        run_id=dp_ctx["run_id"],
                        trial_idx=trial,
                        payload_name=hybrid_payload_name,
                        timeout_sec=dp_ctx["timeout_sec"],
                    )
        else:
            if need_threeway_calibration:
                trial_validation_sampled_calibration = _calibrate_validation_sampled_threeway_thresholds(
                    args=args,
                    edges=edges,
                    embeddings=embeddings,
                    entity_id_to_idx=entity_id_to_idx,
                    random_seed=current_seed,
                    score_field=expert_score_field,
                    score_label=expert_score_label,
                    score_samples_fn=(
                        annotate_expert_source_scores if expert_source_is_semantic else None
                    ),
                )
            if need_key_signal_calibration:
                trial_validation_sampled_key_signal_reference = _calibrate_validation_sampled_key_signal_reference(
                    args=args,
                    edges=edges,
                    embeddings=embeddings,
                    entity_id_to_idx=entity_id_to_idx,
                    random_seed=current_seed,
                )
            if need_hybrid_band_calibration:
                calibrate_hybrid_band_fn = (
                    _calibrate_validation_sampled_hybrid_gmm_overlap_band
                    if use_validation_sampled_hybrid_gmm
                    else _calibrate_validation_sampled_hybrid_uncertainty_band
                )
                trial_validation_sampled_hybrid_band = calibrate_hybrid_band_fn(
                    args=args,
                    edges=edges,
                    embeddings=embeddings,
                    entity_id_to_idx=entity_id_to_idx,
                    random_seed=current_seed,
                    score_field=hybrid_backbone_score_field,
                    score_label=hybrid_backbone_label,
                    score_samples_fn=(
                        annotate_hybrid_backbone_scores
                        if hybrid_backbone_is_semantic
                        else None
                    ),
                )
        validation_sampled_calibration_by_trial.append(trial_validation_sampled_calibration)

        split_runtime = {}

        def maybe_apply_trial_validation_threeway(
            samples,
            include_expert_prediction,
            *,
            score_field=None,
        ):
            if not samples or trial_validation_sampled_calibration is None:
                return samples
            calibration_mode = str(
                trial_validation_sampled_calibration.get("mode", "validation_sampled_threeway")
            ).strip().lower()
            if calibration_mode == "validation_sampled_binary":
                return _apply_validation_sampled_binary_labels(
                    samples,
                    threshold=trial_validation_sampled_calibration["threshold"],
                    assign_expert_prediction=bool(include_expert_prediction),
                    assign_overall_signal=bool(args.include_overall_structural_signal),
                    score_field=(score_field or expert_score_field),
                )
            return _apply_validation_sampled_threeway_labels(
                samples,
                low_threshold=trial_validation_sampled_calibration["low_threshold"],
                high_threshold=trial_validation_sampled_calibration["high_threshold"],
                assign_expert_prediction=bool(include_expert_prediction),
                assign_overall_signal=bool(args.include_overall_structural_signal),
                score_field=(score_field or expert_score_field),
            )

        def maybe_apply_trial_validation_key_signals(samples):
            if not samples or trial_validation_sampled_key_signal_reference is None:
                return samples
            return _apply_validation_sampled_key_signal_reference(
                samples,
                trial_validation_sampled_key_signal_reference,
            )

        def maybe_apply_expert_signal_annotations(
            samples,
            *,
            include_expert_prediction,
            active_split,
        ):
            if not samples:
                return samples
            needs_expert_signal = bool(
                include_expert_prediction or args.include_overall_structural_signal
            )
            if not needs_expert_signal:
                return samples
            if expert_source_is_semantic:
                samples = annotate_expert_source_scores(
                    samples,
                    active_split=active_split,
                    score_field=expert_score_field,
                )
            if use_validation_sampled_threeway or use_validation_sampled_binary:
                return maybe_apply_trial_validation_threeway(
                    samples,
                    include_expert_prediction=include_expert_prediction,
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

        def apply_label_signal_annotations(
            samples,
            *,
            include_expert_prediction,
            active_split,
        ):
            print(
                f"Applying validation key-signal labels: split={active_split}, "
                f"samples={len(samples)}",
                flush=True,
            )
            stage_t0 = time.perf_counter()
            samples = maybe_apply_trial_validation_key_signals(samples)
            print(
                f"Validation key-signal labels complete: split={active_split}, "
                f"elapsed={time.perf_counter() - stage_t0:.1f}s",
                flush=True,
            )

            print(
                f"Applying expert signal labels: split={active_split}, "
                f"samples={len(samples)}",
                flush=True,
            )
            stage_t0 = time.perf_counter()
            samples = maybe_apply_expert_signal_annotations(
                samples,
                include_expert_prediction=include_expert_prediction,
                active_split=active_split,
            )
            print(
                f"Expert signal labels complete: split={active_split}, "
                f"elapsed={time.perf_counter() - stage_t0:.1f}s",
                flush=True,
            )
            return samples

        split_exec_ctx = SplitExecutionContext(
            args=args,
            dp_ctx=dp_ctx,
            model=model,
            tokenizer=tokenizer,
            llm=llm,
            openai_client=openai_client,
            edges=edges,
            prompt_entity_map=prompt_entity_map,
            relation_map=relation_map,
            summary_map=summary_map,
            embeddings=embeddings,
            entity_id_to_idx=entity_id_to_idx,
            semantic_use_smoothing=semantic_use_smoothing,
            selected_key_signal_fields=selected_key_signal_fields,
            use_raw_key_signals=use_raw_key_signals,
            use_percentile_key_signals=use_percentile_key_signals,
            save_eval_details=save_eval_details,
            few_shot_examples=few_shot_examples,
            apply_dtgb_time_bucket=apply_dtgb_time_bucket,
            eval_splits=eval_splits,
            use_two_phase_postprocessing=use_two_phase_postprocessing,
            hybrid_enabled=hybrid_enabled,
            hybrid_backbone_is_semantic=hybrid_backbone_is_semantic,
            hybrid_backbone_score_field=hybrid_backbone_score_field,
            hybrid_backbone_label=hybrid_backbone_label,
            expert_score_field=expert_score_field,
            expert_score_label=expert_score_label,
            use_validation_sampled_threeway=use_validation_sampled_threeway,
            use_validation_sampled_hybrid_band=use_validation_sampled_hybrid_band,
            use_validation_sampled_hybrid_gmm=use_validation_sampled_hybrid_gmm,
            current_seed=current_seed,
            trial=trial,
            split_runtime=split_runtime,
            all_trial_results=all_trial_results,
            trial_validation_sampled_calibration=trial_validation_sampled_calibration,
            trial_validation_sampled_hybrid_band=trial_validation_sampled_hybrid_band,
            trial_hybrid_score_alignment=trial_hybrid_score_alignment,
            maybe_apply_trial_validation_key_signals=maybe_apply_trial_validation_key_signals,
            maybe_apply_expert_signal_annotations=maybe_apply_expert_signal_annotations,
            annotate_hybrid_backbone_scores=annotate_hybrid_backbone_scores,
        )

        for split_idx, eval_split in enumerate(eval_splits):
            include_key_signals = not args.hide_key_signals
            include_expert_prediction = not args.hide_expert_prediction
            rrf_drives_prior_signal = bool(
                (not expert_source_is_semantic)
                and (include_expert_prediction or args.include_overall_structural_signal)
            )
            compute_rrf_scores = (
                args.force_compute_rrf_scores
                or rrf_drives_prior_signal
                or (hybrid_enabled and not hybrid_backbone_is_semantic)
            )
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
            runtime_key_signal_reference = (
                "sequential_global"
                if use_validation_sampled_key_signals
                else args.key_signal_reference
            )
            runtime_skip_key_signal_calibration = bool(
                (not include_key_signals) or use_validation_sampled_key_signals
            )
            split_runtime[eval_split] = {
                "include_key_signals": include_key_signals,
                "include_expert_prediction": include_expert_prediction,
                "compute_rrf_scores": compute_rrf_scores,
                "runtime_compute_expert_prediction": runtime_compute_expert_prediction,
                "runtime_include_overall_structural_signal": runtime_include_overall_structural_signal,
                "runtime_expert_prediction_mode": runtime_expert_prediction_mode,
                "runtime_key_signal_reference": runtime_key_signal_reference,
                "runtime_skip_key_signal_calibration": runtime_skip_key_signal_calibration,
            }
            if args.force_compute_rrf_scores and not (
                include_expert_prediction
                or args.include_overall_structural_signal
                or hybrid_enabled
            ):
                print("RRF scoring is force-enabled by --force_compute_rrf_scores.")

            def build_samples(active_split=eval_split, active_rrf=compute_rrf_scores):
                return create_test_samples(
                    edges,
                    prompt_entity_map,
                    relation_map,
                    test_ratio=args.test_ratio,
                    val_ratio=args.val_ratio,
                    num_samples=args.num_samples,
                    negative_ratio=args.negative_ratio,
                    negative_sampling_mode=args.negative_sampling_mode,
                    negative_sampling_batch_size=args.dtgb_eval_batch_size,
                    random_seed=current_seed,
                    positive_sample_mode=args.positive_sample_mode,
                    positive_sample_block_size=args.positive_sample_block_size,
                    history_window=args.history_window,
                    semantic_history=args.semantic_history,
                    semantic_history_entity_mode=args.semantic_history_entity_mode,
                    common_neighbors_semantic=args.common_neighbors_semantic,
                    build_prompt_features=(not args.rrf_only),
                    semantic_use_smoothing=semantic_use_smoothing,
                    semantic_hub_penalty_alpha=args.semantic_hub_penalty_alpha,
                    semantic_fusion_alpha=args.semantic_fusion_alpha,
                    semantic_fusion_tau=args.semantic_fusion_tau,
                    semantic_fusion_recency_speed=args.semantic_fusion_recency_speed,
                    semantic_topk=args.semantic_topk,
                    history_pool_size=args.history_pool_size,
                    history_pool_window=args.history_pool_window,
                    history_preserve_recent_k=args.history_preserve_recent_k,
                    embeddings=embeddings,
                    entity_id_to_idx=entity_id_to_idx,
                    embedding_model=args.embedding_model,
                    embedding_cache=args.embedding_cache,
                    compute_expert_prediction=runtime_compute_expert_prediction,
                    compute_rrf_scores=active_rrf,
                    rrf_k=args.rrf_k,
                    rrf_mode=args.rrf_mode,
                    sequential_rank_bins=args.sequential_rank_bins,
                    expert_prediction_mode=runtime_expert_prediction_mode,
                    expert_prediction_fixed_threshold=args.expert_prediction_fixed_threshold,
                    rrf_pointwise_pool_size=args.rrf_pointwise_pool_size,
                    rrf_pointwise_num_pools=args.rrf_pointwise_num_pools,
                    rrf_batch_size=args.rrf_batch_size,
                    rrf_heuristics=args.rrf_heuristics,
                    eval_split=active_split,
                    key_signal_reference=runtime_key_signal_reference,
                    key_signal_fields=selected_key_signal_fields,
                    include_overall_structural_signal=runtime_include_overall_structural_signal,
                    overall_signal_low_threshold=args.overall_signal_low_threshold,
                    overall_signal_high_threshold=args.overall_signal_high_threshold,
                    defer_postprocessing=use_two_phase_postprocessing,
                    apply_gdelt_time_bucket=apply_dtgb_time_bucket,
                    sample_creation_monitor_every=args.sample_creation_monitor_every,
                    sample_creation_profile=args.sample_creation_profile,
                    sample_creation_profile_sort=args.sample_creation_profile_sort,
                    sample_creation_profile_top_n=args.sample_creation_profile_top_n,
                    sample_creation_profile_output=args.sample_creation_profile_output,
                    defer_prompt_context_materialization=(hybrid_enabled and (not args.rrf_only)),
                    skip_key_signal_calibration=runtime_skip_key_signal_calibration,
                )

            if args.legacy_interleaved_split_processing:
                if len(eval_splits) > 1:
                    print(f"\n[Split: {eval_split}]")

                if dp_ctx["enabled"]:
                    if dp_ctx["rank"] == 0:
                        split_samples = build_samples()
                        split_samples = apply_label_signal_annotations(
                            split_samples,
                            include_expert_prediction=include_expert_prediction,
                            active_split=eval_split,
                        )
                        split_samples = annotate_hybrid_backbone_scores(
                            split_samples,
                            active_split=eval_split,
                        )
                        maybe_print_semantic_hybrid_backbone_baseline(
                            split_samples,
                            active_split=eval_split,
                        )
                        shared_path = write_shared_samples(
                            sync_dir=dp_ctx["sync_dir"],
                            run_id=dp_ctx["run_id"],
                            trial_idx=trial,
                            split_name=eval_split,
                            samples=split_samples,
                        )
                        print(
                            f"DP sample bundle saved: {shared_path} "
                            f"({len(split_samples)} samples)."
                        )
                    else:
                        split_samples = wait_for_shared_samples(
                            sync_dir=dp_ctx["sync_dir"],
                            run_id=dp_ctx["run_id"],
                            trial_idx=trial,
                            split_name=eval_split,
                            timeout_sec=dp_ctx["timeout_sec"],
                        )
                        split_samples = apply_label_signal_annotations(
                            split_samples,
                            include_expert_prediction=include_expert_prediction,
                            active_split=eval_split,
                        )
                        print(
                            f"DP sample bundle loaded: {len(split_samples)} samples "
                            f"for trial {trial + 1} [{eval_split}]."
                        )
                else:
                    split_samples = build_samples()
                    split_samples = apply_label_signal_annotations(
                        split_samples,
                        include_expert_prediction=include_expert_prediction,
                        active_split=eval_split,
                    )
                    split_samples = annotate_hybrid_backbone_scores(
                        split_samples,
                        active_split=eval_split,
                    )
                    maybe_print_semantic_hybrid_backbone_baseline(
                        split_samples,
                        active_split=eval_split,
                    )

                split_runtime[eval_split]["samples"] = split_samples
                execute_split(split_exec_ctx, split_idx, eval_split)
            else:
                if len(eval_splits) > 1:
                    print(f"\n[Split: {eval_split}]")
                    print("Preparing samples before inference...")

                if dp_ctx["enabled"]:
                    if use_two_phase_postprocessing:
                        if dp_ctx["rank"] == 0:
                            print(
                                f"Building deferred samples: split={eval_split}, "
                                f"num_positive={args.num_samples}",
                                flush=True,
                            )
                            split_samples = build_samples()
                            print(
                                f"Deferred sample build complete: split={eval_split}, "
                                f"samples={len(split_samples)}. "
                                "Annotating semantic hybrid backbone next.",
                                flush=True,
                            )
                            split_samples = annotate_hybrid_backbone_scores(
                                split_samples,
                                active_split=eval_split,
                            )
                            print(
                                f"Semantic hybrid backbone annotation complete: split={eval_split}.",
                                flush=True,
                            )
                            maybe_print_semantic_hybrid_backbone_baseline(
                                split_samples,
                                active_split=eval_split,
                            )
                        else:
                            split_samples = None
                            print(
                                f"Rank {dp_ctx['rank']} waiting for finalized sample bundle "
                                f"later for trial {trial + 1} [{eval_split}]."
                            )
                    else:
                        if dp_ctx["rank"] == 0:
                            split_samples = build_samples()
                            split_samples = apply_label_signal_annotations(
                                split_samples,
                                include_expert_prediction=include_expert_prediction,
                                active_split=eval_split,
                            )
                            split_samples = annotate_hybrid_backbone_scores(
                                split_samples,
                                active_split=eval_split,
                            )
                            maybe_print_semantic_hybrid_backbone_baseline(
                                split_samples,
                                active_split=eval_split,
                            )
                            shared_path = write_shared_samples(
                                sync_dir=dp_ctx["sync_dir"],
                                run_id=dp_ctx["run_id"],
                                trial_idx=trial,
                                split_name=eval_split,
                                samples=split_samples,
                            )
                            print(
                                f"DP sample bundle saved: {shared_path} "
                                f"({len(split_samples)} samples)."
                            )
                        else:
                            split_samples = wait_for_shared_samples(
                                sync_dir=dp_ctx["sync_dir"],
                                run_id=dp_ctx["run_id"],
                                trial_idx=trial,
                                split_name=eval_split,
                                timeout_sec=dp_ctx["timeout_sec"],
                            )
                            split_samples = apply_label_signal_annotations(
                                split_samples,
                                include_expert_prediction=include_expert_prediction,
                                active_split=eval_split,
                            )
                            print(
                                f"DP sample bundle loaded: {len(split_samples)} samples "
                                f"for trial {trial + 1} [{eval_split}]."
                            )
                else:
                    print(
                        f"Building samples: split={eval_split}, "
                        f"num_positive={args.num_samples}",
                        flush=True,
                    )
                    split_samples = build_samples()
                    print(
                        f"Sample build complete: split={eval_split}, "
                        f"samples={len(split_samples)}. Applying labels/signals next.",
                        flush=True,
                    )
                    split_samples = apply_label_signal_annotations(
                        split_samples,
                        include_expert_prediction=include_expert_prediction,
                        active_split=eval_split,
                    )
                    print(
                        f"Label/signal annotation complete: split={eval_split}. "
                        "Annotating semantic hybrid backbone next.",
                        flush=True,
                    )
                    split_samples = annotate_hybrid_backbone_scores(
                        split_samples,
                        active_split=eval_split,
                    )
                    print(
                        f"Semantic hybrid backbone annotation complete: split={eval_split}.",
                        flush=True,
                    )
                    maybe_print_semantic_hybrid_backbone_baseline(
                        split_samples,
                        active_split=eval_split,
                    )

                split_runtime[eval_split]["samples"] = split_samples

        if not args.legacy_interleaved_split_processing:
            for split_idx, eval_split in enumerate(eval_splits):
                execute_split(split_exec_ctx, split_idx, eval_split)

    if dp_ctx["enabled"] and dp_ctx["rank"] != 0:
        print("\nDP worker finished. Rank 0 will aggregate and write final results.")
        write_rank_completion(
            sync_dir=dp_ctx["sync_dir"],
            run_id=dp_ctx["run_id"],
            rank=dp_ctx["rank"],
        )
        return

    aggregated_metrics_by_split = {}

    print("\n" + "=" * 80)
    print("FINAL AGGREGATED RESULTS")
    print("=" * 80)
    print(f"Number of trials: {args.num_trials}")
    for split_name in eval_splits:
        split_results = all_trial_results.get(split_name, [])
        if len(split_results) == 0:
            continue
        aps = [result["ap"] for result in split_results]
        aucs = [result["auc"] for result in split_results]
        accs = [result["accuracy"] for result in split_results]
        aggregated_metrics_by_split[split_name] = {
            "ap_mean": float(np.mean(aps)),
            "ap_std": float(np.std(aps)),
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
        }
        if len(eval_splits) > 1:
            print(f"[{split_name}]")
        print(f"Average Precision (AP): {np.mean(aps):.4f} ± {np.std(aps):.4f}")
        print(f"AUC-ROC:              {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"Accuracy:             {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print("=" * 80)

    try:
        import wandb

        if wandb.run is None:
            wandb.init(project="dtgb_llm_link_prediction", config=vars(args))
        wandb_payload = {}
        if "transductive" in aggregated_metrics_by_split:
            wandb_payload["final/transductive_auc"] = aggregated_metrics_by_split[
                "transductive"
            ]["auc_mean"]
        if "inductive" in aggregated_metrics_by_split:
            wandb_payload["final/inductive_auc"] = aggregated_metrics_by_split[
                "inductive"
            ]["auc_mean"]
        if wandb_payload:
            wandb.log(wandb_payload)
    except ImportError:
        print("wandb is not installed; skipping final AUC logging.")

    output_data = build_output_data(
        args=args,
        eval_splits=eval_splits,
        aggregated_metrics_by_split=aggregated_metrics_by_split,
        all_trial_results=all_trial_results,
        validation_sampled_calibration_by_trial=validation_sampled_calibration_by_trial,
    )
    with open(args.output, "w") as handle:
        json.dump(output_data, handle, indent=2)

    print(f"\n✓ Results saved to {args.output}")
    if dp_ctx["enabled"]:
        write_rank_completion(
            sync_dir=dp_ctx["sync_dir"],
            run_id=dp_ctx["run_id"],
            rank=dp_ctx["rank"],
        )
        if dp_ctx["rank"] == 0:
            wait_for_rank_completions(
                sync_dir=dp_ctx["sync_dir"],
                run_id=dp_ctx["run_id"],
                dp_size=dp_ctx["size"],
                timeout_sec=dp_ctx["timeout_sec"],
            )
            if not args.keep_data_parallel_sync:
                removed_path = cleanup_data_parallel_run(
                    sync_dir=dp_ctx["sync_dir"],
                    run_id=dp_ctx["run_id"],
                    remove_sync_root_if_empty=True,
                )
                if removed_path:
                    print(f"DP sync cleanup complete: removed {removed_path}")


def main():
    configure_runtime_logging()
    parser = build_arg_parser()
    # Keep archived shared-source bytes reproducible while requiring an
    # explicit local checkpoint or model identifier at the public entrypoint.
    parser.set_defaults(model_path=None)
    args = parser.parse_args()
    needs_local_model = args.export_prompt_dataset_dir or not (
        args.rrf_only or args.use_openai_api
    )
    if needs_local_model and not args.model_path:
        parser.error("--model_path is required for local LLM evaluation or prompt export")
    validate_args(args)

    if maybe_spawn_local_dp_workers(
        args,
        script_path=os.path.abspath(__file__),
        argv=sys.argv[1:],
    ):
        return

    run_evaluation(args)


if __name__ == "__main__":
    main()
