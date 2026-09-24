#!/usr/bin/env python3
"""
Per-split evaluation runner for LLM link prediction experiments.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.modules.prediction_metrics import compute_prediction_metrics

from experiments.modules.llm_lp.data_parallel import (
    merge_hybrid_selected_shards,
    merge_trial_shards,
    split_samples_for_rank,
    wait_for_hybrid_selected_shards,
    wait_for_hybrid_selection,
    wait_for_shared_payload,
    wait_for_shared_samples,
    wait_for_trial_shards,
    write_hybrid_selected_shard,
    write_hybrid_selection,
    write_shared_payload,
    write_shared_samples,
    write_trial_shard,
)
from experiments.modules.llm_lp.history_compaction import compact_history_local
from experiments.modules.llm_lp.eval import (
    build_link_prediction_prompts,
    evaluate,
    evaluate_rrf_only,
    evaluate_transformers,
    export_link_prediction_prompt_dataset,
    write_prediction_debug_log,
)
from experiments.modules.llm_lp.experiment import (
    apply_hybrid_debug_route_cap,
    build_trial_result_summary,
    maybe_print_token_usage_summary,
    normalize_hybrid_selection,
)
from experiments.modules.llm_lp.experiment_calibration import (
    _build_expert_prediction_discriminative_debug,
    _build_key_signal_discriminative_debug,
    _build_lightweight_rrf_samples_for_split,
    _build_realized_discriminative_debug,
    _resolve_hybrid_alignment_calibration_sizes,
)
from experiments.modules.llm_lp.prompt_context import materialize_samples_prompt_context
from experiments.modules.llm_lp.sample_builder import create_test_samples
from experiments.modules.llm_lp.sample_finalize import finalize_test_samples
from experiments.modules.llm_lp.openai_eval import evaluate_openai_chat
from experiments.modules.rrf.analysis import (
    select_learned_router_indices,
    select_random_sample_indices,
    select_rrf_middle_sample_indices_pointwise_threshold_band,
    select_rrf_middle_sample_indices_validation_sampled_band,
    select_rrf_middle_sample_indices_validation_sampled_gmm_overlap,
    select_topk_rrf_middle_sample_indices,
)
from experiments.modules.rrf.hybrid import (
    evaluate_budgeted_hybrid_rrf_llm,
    fit_isotonic_score_alignment,
    fit_quantile_score_alignment,
    prepare_backbone_score_space,
)
from experiments.modules.tabicl.online_pipeline import (
    prepare_online_tabicl_fusion_table,
    run_online_tabicl_fusion,
    run_online_tabicl_router,
)


@dataclass
class SplitExecutionContext:
    args: Any
    dp_ctx: dict
    model: Any
    tokenizer: Any
    llm: Any
    openai_client: Any
    edges: Any
    prompt_entity_map: dict
    relation_map: dict
    summary_map: Any
    embeddings: Any
    entity_id_to_idx: Any
    semantic_use_smoothing: bool
    selected_key_signal_fields: tuple
    use_raw_key_signals: bool
    use_percentile_key_signals: bool
    save_eval_details: bool
    few_shot_examples: Any
    apply_dtgb_time_bucket: bool
    eval_splits: list
    use_two_phase_postprocessing: bool
    hybrid_enabled: bool
    hybrid_backbone_is_semantic: bool
    hybrid_backbone_score_field: str
    hybrid_backbone_label: str
    expert_score_field: str
    expert_score_label: str
    use_validation_sampled_threeway: bool
    use_validation_sampled_hybrid_band: bool
    use_validation_sampled_hybrid_gmm: bool
    current_seed: int
    trial: int
    split_runtime: dict
    all_trial_results: dict
    trial_validation_sampled_calibration: Any = None
    trial_validation_sampled_hybrid_band: Any = None
    trial_hybrid_score_alignment: Any = None
    trial_hybrid_alignment_calibration: Any = None
    maybe_apply_trial_validation_key_signals: Callable[[list], list] | None = None
    maybe_apply_expert_signal_annotations: Callable[..., list] | None = None
    annotate_hybrid_backbone_scores: Callable[..., list] | None = None


def _uniform_query_subset(
    samples: list[dict[str, Any]],
    *,
    positive_queries: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Uniformly retain complete positive/negative query pairs."""
    labels = np.asarray([int(sample["label"]) for sample in samples], dtype=np.int64)
    query_ids = np.asarray(
        [int(sample["query_id"]) for sample in samples], dtype=np.int64
    )
    positive_query_ids = query_ids[labels == 1]
    if np.unique(positive_query_ids).size != len(positive_query_ids):
        raise ValueError("TabICL context pool has duplicate positive query IDs")
    if positive_queries > len(positive_query_ids):
        raise ValueError(
            "TabICL uniform context request exceeds its recent pool: "
            f"{positive_queries} > {len(positive_query_ids)}"
        )
    selected_query_ids = np.random.RandomState(int(seed)).choice(
        positive_query_ids,
        size=int(positive_queries),
        replace=False,
    )
    selected_set = {int(value) for value in selected_query_ids.tolist()}
    selected = [
        sample for sample in samples if int(sample["query_id"]) in selected_set
    ]
    selected_labels = np.asarray(
        [int(sample["label"]) for sample in selected], dtype=np.int64
    )
    selected_queries = np.asarray(
        [int(sample["query_id"]) for sample in selected], dtype=np.int64
    )
    if (
        len(selected) != 2 * int(positive_queries)
        or np.count_nonzero(selected_labels == 1) != int(positive_queries)
        or np.count_nonzero(selected_labels == 0) != int(positive_queries)
        or np.unique(selected_queries).size != int(positive_queries)
    ):
        raise RuntimeError(
            "Uniform TabICL context sampling did not preserve matched query pairs"
        )
    return selected


def execute_split(ctx: SplitExecutionContext, split_idx: int, eval_split: str) -> None:
    args = ctx.args
    dp_ctx = ctx.dp_ctx
    include_key_signals = not args.hide_key_signals
    include_expert_prediction = ctx.split_runtime[eval_split]["include_expert_prediction"]
    compute_rrf_scores = ctx.split_runtime[eval_split]["compute_rrf_scores"]
    runtime_compute_expert_prediction = ctx.split_runtime[eval_split][
        "runtime_compute_expert_prediction"
    ]
    runtime_include_overall_structural_signal = ctx.split_runtime[eval_split][
        "runtime_include_overall_structural_signal"
    ]
    runtime_expert_prediction_mode = ctx.split_runtime[eval_split][
        "runtime_expert_prediction_mode"
    ]
    runtime_key_signal_reference = ctx.split_runtime[eval_split]["runtime_key_signal_reference"]
    runtime_skip_key_signal_calibration = ctx.split_runtime[eval_split][
        "runtime_skip_key_signal_calibration"
    ]
    samples = ctx.split_runtime[eval_split]["samples"]
    needs_contextual_key_signal_calibration = bool(
        include_key_signals and args.key_signal_reference == "contextual"
    )
    defer_contextual_key_signal_calibration = (
        ctx.hybrid_enabled
        and (not args.rrf_only)
        and needs_contextual_key_signal_calibration
    )

    if ctx.use_two_phase_postprocessing:
        if dp_ctx["enabled"]:
            if dp_ctx["rank"] == 0:
                if not ctx.split_runtime[eval_split].get("postprocessed", False):
                    if defer_contextual_key_signal_calibration:
                        print(
                            "Deferring prompt-side contextual key-signal calibration "
                            "to selected-slice enrichment."
                        )
                    samples = finalize_test_samples(
                        samples=samples,
                        edges_df=ctx.edges,
                        val_ratio=args.val_ratio,
                        test_ratio=args.test_ratio,
                        negative_ratio=args.negative_ratio,
                        random_seed=ctx.current_seed,
                        build_prompt_features=(
                            (not args.rrf_only) and (not defer_contextual_key_signal_calibration)
                        ),
                        compute_expert_prediction=runtime_compute_expert_prediction,
                        compute_rrf_scores=compute_rrf_scores,
                        rrf_k=args.rrf_k,
                        rrf_mode=args.rrf_mode,
                        sequential_rank_bins=args.sequential_rank_bins,
                        expert_prediction_mode=runtime_expert_prediction_mode,
                        expert_prediction_fixed_threshold=args.expert_prediction_fixed_threshold,
                        key_signal_fields=ctx.selected_key_signal_fields,
                        rrf_pointwise_pool_size=args.rrf_pointwise_pool_size,
                        rrf_pointwise_num_pools=args.rrf_pointwise_num_pools,
                        rrf_batch_size=args.rrf_batch_size,
                        rrf_heuristics=args.rrf_heuristics,
                        key_signal_reference=runtime_key_signal_reference,
                        include_overall_structural_signal=runtime_include_overall_structural_signal,
                        overall_signal_low_threshold=args.overall_signal_low_threshold,
                        overall_signal_high_threshold=args.overall_signal_high_threshold,
                        heuristic_recent_degree_window=args.heuristic_recent_degree_window,
                        use_gpu_heuristics=True,
                        apply_gdelt_time_bucket=ctx.apply_dtgb_time_bucket,
                        skip_key_signal_calibration=runtime_skip_key_signal_calibration,
                    )
                    samples = ctx.maybe_apply_trial_validation_key_signals(samples)
                    samples = ctx.maybe_apply_expert_signal_annotations(
                        samples,
                        include_expert_prediction=include_expert_prediction,
                        active_split=eval_split,
                    )
                    samples = ctx.annotate_hybrid_backbone_scores(
                        samples,
                        active_split=eval_split,
                    )
                    ctx.split_runtime[eval_split]["samples"] = samples
                    ctx.split_runtime[eval_split]["postprocessed"] = True
                    shared_path = write_shared_samples(
                        sync_dir=dp_ctx["sync_dir"],
                        run_id=dp_ctx["run_id"],
                        trial_idx=ctx.trial,
                        split_name=eval_split,
                        samples=samples,
                    )
                    print(f"DP sample bundle saved: {shared_path} ({len(samples)} samples).")
            else:
                if not ctx.split_runtime[eval_split].get("postprocessed", False):
                    samples = wait_for_shared_samples(
                        sync_dir=dp_ctx["sync_dir"],
                        run_id=dp_ctx["run_id"],
                        trial_idx=ctx.trial,
                        split_name=eval_split,
                        timeout_sec=dp_ctx["timeout_sec"],
                    )
                    samples = ctx.maybe_apply_trial_validation_key_signals(samples)
                    samples = ctx.maybe_apply_expert_signal_annotations(
                        samples,
                        include_expert_prediction=include_expert_prediction,
                        active_split=eval_split,
                    )
                    ctx.split_runtime[eval_split]["samples"] = samples
                    ctx.split_runtime[eval_split]["postprocessed"] = True
                    print(
                        f"DP sample bundle loaded: {len(samples)} samples "
                        f"for trial {ctx.trial + 1} [{eval_split}]."
                    )
        else:
            if not ctx.split_runtime[eval_split].get("postprocessed", False):
                if defer_contextual_key_signal_calibration:
                    print(
                        "Deferring prompt-side contextual key-signal calibration "
                        "to selected-slice enrichment.",
                        flush=True,
                    )
                print(
                    "Postprocessing full sample set before expert/hybrid scoring: "
                    f"split={eval_split}, samples={len(samples)}, "
                    f"compute_rrf_scores={bool(compute_rrf_scores)}, "
                    f"key_signal_reference={runtime_key_signal_reference}, "
                    f"skip_key_signal_calibration={bool(runtime_skip_key_signal_calibration)}",
                    flush=True,
                )
                samples = finalize_test_samples(
                    samples=samples,
                    edges_df=ctx.edges,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio,
                    negative_ratio=args.negative_ratio,
                    random_seed=ctx.current_seed,
                    build_prompt_features=(
                        (not args.rrf_only) and (not defer_contextual_key_signal_calibration)
                    ),
                    compute_expert_prediction=runtime_compute_expert_prediction,
                    compute_rrf_scores=compute_rrf_scores,
                    rrf_k=args.rrf_k,
                    rrf_mode=args.rrf_mode,
                    sequential_rank_bins=args.sequential_rank_bins,
                    expert_prediction_mode=runtime_expert_prediction_mode,
                    expert_prediction_fixed_threshold=args.expert_prediction_fixed_threshold,
                    key_signal_fields=ctx.selected_key_signal_fields,
                    rrf_pointwise_pool_size=args.rrf_pointwise_pool_size,
                    rrf_pointwise_num_pools=args.rrf_pointwise_num_pools,
                    rrf_batch_size=args.rrf_batch_size,
                    rrf_heuristics=args.rrf_heuristics,
                    key_signal_reference=runtime_key_signal_reference,
                    include_overall_structural_signal=runtime_include_overall_structural_signal,
                    overall_signal_low_threshold=args.overall_signal_low_threshold,
                    overall_signal_high_threshold=args.overall_signal_high_threshold,
                    heuristic_recent_degree_window=args.heuristic_recent_degree_window,
                    use_gpu_heuristics=True,
                    apply_gdelt_time_bucket=ctx.apply_dtgb_time_bucket,
                    skip_key_signal_calibration=runtime_skip_key_signal_calibration,
                )
                print(
                    "Full sample postprocessing complete; applying validation key signals "
                    "and semantic expert/hybrid annotations next.",
                    flush=True,
                )
                samples = ctx.maybe_apply_trial_validation_key_signals(samples)
                samples = ctx.maybe_apply_expert_signal_annotations(
                    samples,
                    include_expert_prediction=include_expert_prediction,
                    active_split=eval_split,
                )
                samples = ctx.annotate_hybrid_backbone_scores(
                    samples,
                    active_split=eval_split,
                )
                ctx.split_runtime[eval_split]["samples"] = samples
                ctx.split_runtime[eval_split]["postprocessed"] = True

    if len(ctx.eval_splits) > 1:
        print(f"\n[Split: {eval_split}]")

    def run_eval(
        current_samples,
        dtgb_eval_batch_size_override=None,
        active_eval_split=eval_split,
        capture_embeddings=False,
    ):
        if dtgb_eval_batch_size_override is None:
            metric_batch_size = args.dtgb_eval_batch_size
        elif dtgb_eval_batch_size_override is False:
            metric_batch_size = None
        else:
            metric_batch_size = int(dtgb_eval_batch_size_override)
        if args.export_prompt_dataset_dir:
            prompts = build_link_prediction_prompts(
                current_samples,
                ctx.tokenizer,
                ctx.prompt_entity_map,
                ctx.relation_map,
                summary_map=ctx.summary_map,
                summary_mode=args.summary_mode,
                summary_max_chars=args.summary_max_chars,
                history_window=args.history_window,
                include_key_signals=include_key_signals,
                include_expert_prediction=include_expert_prediction,
                include_overall_structural_signal=args.include_overall_structural_signal,
                overall_structural_signal_name=args.overall_signal_name,
                use_raw_key_signals=ctx.use_raw_key_signals,
                use_percentile_key_signals=ctx.use_percentile_key_signals,
                include_edge_type=args.include_edge_type,
                include_edge_type_except_target=args.include_edge_type_except_target,
                use_cot=not args.no_cot,
                mutual_timestamps_only=args.mutual_timestamps_only,
                mutual_timestamps_dedup=args.mutual_timestamps_dedup,
                mutual_summary_count_recency=args.mutual_summary_count_recency,
                common_neighbors_names_only=args.common_neighbors_names_only,
                compact_common_neighbors_top_k=args.compact_common_neighbors_top_k,
                compact_common_neighbors_novel_only=args.compact_common_neighbors_novel_only,
                history_table_aliases=args.history_table_aliases,
                anonymous_entity_aliases=args.anonymous_entity_aliases,
                natural_grouped_history=args.natural_grouped_history,
                natural_activity_summary=args.natural_activity_summary,
                natural_neighbor_names_only=args.natural_neighbor_names_only,
                natural_activity_compact_top3=args.natural_activity_compact_top3,
                natural_activity_top_k=args.natural_activity_top_k,
                ablate_mutual_history=args.ablate_mutual_history,
                ablate_common_neighbors=args.ablate_common_neighbors,
                ablate_source_history=args.ablate_source_history,
                ablate_target_history=args.ablate_target_history,
                ablate_source_target_history=args.ablate_source_target_history,
                ablate_reasoning_guidance=args.ablate_reasoning_guidance,
                no_cot_output_0_100=args.no_cot_output_0_100,
                prompt_variant=args.vllm_prompt_variant,
                few_shot_examples=ctx.few_shot_examples,
                key_signal_fields=ctx.selected_key_signal_fields,
                force_binary_answer_prefix=True,
            )
            dataset_path = export_link_prediction_prompt_dataset(
                args.export_prompt_dataset_dir,
                current_samples,
                prompts,
                eval_split=active_eval_split,
            )
            labels = np.asarray(
                [int(sample["label"]) for sample in current_samples], dtype=np.int64
            )
            metrics = compute_prediction_metrics(
                np.full(len(labels), 0.5, dtype=np.float64),
                labels,
                dtgb_eval_batch_size=metric_batch_size,
            )
            metrics.update(
                {
                    "predictions": np.full(len(labels), 0.5).tolist(),
                    "labels": labels.tolist(),
                    "parse_stats": {"mode": "prompt_dataset_export"},
                    "detailed_results": [],
                    "prompt_dataset_path": dataset_path,
                }
            )
            print(f"Prompt dataset exported: {dataset_path} ({len(labels)} rows)")
            return metrics
        if args.rrf_only:
            return evaluate_rrf_only(
                current_samples,
                negative_ratio=args.negative_ratio,
                save_details=ctx.save_eval_details,
                dtgb_eval_batch_size=metric_batch_size,
            )
        if args.use_transformers:
            return evaluate_transformers(
                ctx.model,
                ctx.tokenizer,
                current_samples,
                entity_map=ctx.prompt_entity_map,
                relation_map=ctx.relation_map,
                summary_map=ctx.summary_map,
                summary_mode=args.summary_mode,
                summary_max_chars=args.summary_max_chars,
                history_window=args.history_window,
                save_details=ctx.save_eval_details,
                use_cot=not args.no_cot,
                include_key_signals=include_key_signals,
                key_signal_fields=ctx.selected_key_signal_fields,
                include_expert_prediction=include_expert_prediction,
                include_overall_structural_signal=args.include_overall_structural_signal,
                overall_structural_signal_name=args.overall_signal_name,
                use_raw_key_signals=ctx.use_raw_key_signals,
                use_percentile_key_signals=ctx.use_percentile_key_signals,
                include_edge_type=args.include_edge_type,
                cot_max_tokens=args.cot_max_tokens,
                mutual_timestamps_only=args.mutual_timestamps_only,
                mutual_timestamps_dedup=args.mutual_timestamps_dedup,
                mutual_summary_count_recency=args.mutual_summary_count_recency,
                common_neighbors_names_only=args.common_neighbors_names_only,
                compact_common_neighbors_top_k=args.compact_common_neighbors_top_k,
                compact_common_neighbors_novel_only=args.compact_common_neighbors_novel_only,
                history_table_aliases=args.history_table_aliases,
                anonymous_entity_aliases=args.anonymous_entity_aliases,
                natural_grouped_history=args.natural_grouped_history,
                natural_activity_summary=args.natural_activity_summary,
                natural_neighbor_names_only=args.natural_neighbor_names_only,
                natural_activity_compact_top3=args.natural_activity_compact_top3,
                natural_activity_top_k=args.natural_activity_top_k,
                ablate_mutual_history=args.ablate_mutual_history,
                ablate_common_neighbors=args.ablate_common_neighbors,
                ablate_source_history=args.ablate_source_history,
                ablate_target_history=args.ablate_target_history,
                ablate_source_target_history=args.ablate_source_target_history,
                ablate_reasoning_guidance=args.ablate_reasoning_guidance,
                no_cot_output_0_100=args.no_cot_output_0_100,
                no_cot_binary_score_mode=args.no_cot_binary_score_mode,
                eval_split=active_eval_split,
                dtgb_eval_batch_size=metric_batch_size,
                few_shot_examples=ctx.few_shot_examples,
            )
        if args.use_openai_api:
            return evaluate_openai_chat(
                client=ctx.openai_client,
                model_name=args.openai_model,
                samples=current_samples,
                entity_map=ctx.prompt_entity_map,
                summary_map=ctx.summary_map,
                summary_mode=args.summary_mode,
                summary_max_chars=args.summary_max_chars,
                relation_map=ctx.relation_map,
                history_window=args.history_window,
                save_details=ctx.save_eval_details,
                use_cot=not args.no_cot,
                include_key_signals=include_key_signals,
                key_signal_fields=ctx.selected_key_signal_fields,
                include_expert_prediction=include_expert_prediction,
                include_overall_structural_signal=args.include_overall_structural_signal,
                overall_structural_signal_name=args.overall_signal_name,
                use_raw_key_signals=ctx.use_raw_key_signals,
                use_percentile_key_signals=ctx.use_percentile_key_signals,
                include_edge_type=args.include_edge_type,
                cot_max_tokens=args.cot_max_tokens,
                no_cot_max_tokens=args.openai_no_cot_max_tokens,
                mutual_timestamps_only=args.mutual_timestamps_only,
                mutual_timestamps_dedup=args.mutual_timestamps_dedup,
                mutual_summary_count_recency=args.mutual_summary_count_recency,
                common_neighbors_names_only=args.common_neighbors_names_only,
                compact_common_neighbors_top_k=args.compact_common_neighbors_top_k,
                compact_common_neighbors_novel_only=args.compact_common_neighbors_novel_only,
                history_table_aliases=args.history_table_aliases,
                anonymous_entity_aliases=args.anonymous_entity_aliases,
                natural_grouped_history=args.natural_grouped_history,
                natural_activity_summary=args.natural_activity_summary,
                natural_neighbor_names_only=args.natural_neighbor_names_only,
                natural_activity_compact_top3=args.natural_activity_compact_top3,
                natural_activity_top_k=args.natural_activity_top_k,
                ablate_mutual_history=args.ablate_mutual_history,
                ablate_common_neighbors=args.ablate_common_neighbors,
                ablate_source_history=args.ablate_source_history,
                ablate_target_history=args.ablate_target_history,
                ablate_source_target_history=args.ablate_source_target_history,
                ablate_reasoning_guidance=args.ablate_reasoning_guidance,
                no_cot_output_0_100=args.no_cot_output_0_100,
                no_cot_binary_score_mode=args.no_cot_binary_score_mode,
                eval_split=active_eval_split,
                dtgb_eval_batch_size=metric_batch_size,
                reasoning_effort=args.openai_reasoning_effort,
                temperature=args.openai_temperature,
                top_p=args.openai_top_p,
                openai_concurrency=args.openai_concurrency,
                openai_max_requests_per_sec=args.openai_max_requests_per_sec,
                few_shot_examples=ctx.few_shot_examples,
            )
        if args.enable_history_compaction_llm_only:
            compact_history_local(
                llm=ctx.llm,
                tokenizer=ctx.tokenizer,
                samples=current_samples,
                entity_map=ctx.prompt_entity_map,
                relation_map=ctx.relation_map,
                history_window=args.history_window,
                max_tokens=args.history_compaction_max_tokens,
                include_edge_type=args.include_edge_type,
                mutual_timestamps_only=args.mutual_timestamps_only,
                mutual_timestamps_dedup=args.mutual_timestamps_dedup,
                common_neighbors_names_only=args.common_neighbors_names_only,
            )
        return evaluate(
            ctx.llm,
            ctx.tokenizer,
            current_samples,
            entity_map=ctx.prompt_entity_map,
            relation_map=ctx.relation_map,
            summary_map=ctx.summary_map,
            summary_mode=args.summary_mode,
            summary_max_chars=args.summary_max_chars,
            history_window=args.history_window,
            save_details=ctx.save_eval_details,
            use_cot=not args.no_cot,
            include_key_signals=include_key_signals,
            key_signal_fields=ctx.selected_key_signal_fields,
            include_expert_prediction=include_expert_prediction,
            include_overall_structural_signal=args.include_overall_structural_signal,
            overall_structural_signal_name=args.overall_signal_name,
            use_raw_key_signals=ctx.use_raw_key_signals,
            use_percentile_key_signals=ctx.use_percentile_key_signals,
            include_edge_type=args.include_edge_type,
            include_edge_type_except_target=args.include_edge_type_except_target,
            cot_max_tokens=args.cot_max_tokens,
            mutual_timestamps_only=args.mutual_timestamps_only,
            mutual_timestamps_dedup=args.mutual_timestamps_dedup,
            mutual_summary_count_recency=args.mutual_summary_count_recency,
            common_neighbors_names_only=args.common_neighbors_names_only,
            compact_common_neighbors_top_k=args.compact_common_neighbors_top_k,
            compact_common_neighbors_novel_only=args.compact_common_neighbors_novel_only,
            history_table_aliases=args.history_table_aliases,
            anonymous_entity_aliases=args.anonymous_entity_aliases,
            natural_grouped_history=args.natural_grouped_history,
            natural_activity_summary=args.natural_activity_summary,
            natural_neighbor_names_only=args.natural_neighbor_names_only,
            natural_activity_compact_top3=args.natural_activity_compact_top3,
            natural_activity_top_k=args.natural_activity_top_k,
            ablate_mutual_history=args.ablate_mutual_history,
            ablate_common_neighbors=args.ablate_common_neighbors,
            ablate_source_history=args.ablate_source_history,
            ablate_target_history=args.ablate_target_history,
            ablate_source_target_history=args.ablate_source_target_history,
            ablate_reasoning_guidance=args.ablate_reasoning_guidance,
            no_cot_output_0_100=args.no_cot_output_0_100,
            no_cot_binary_score_mode=args.no_cot_binary_score_mode,
            prompt_variant=args.vllm_prompt_variant,
            eval_split=active_eval_split,
            dtgb_eval_batch_size=metric_batch_size,
            few_shot_examples=ctx.few_shot_examples,
            capture_prompt_embeddings=bool(
                args.capture_prompt_embeddings and capture_embeddings
            ),
            prompt_embedding_output_dir=args.prompt_embedding_output_dir,
            prompt_embedding_normalize=args.prompt_embedding_normalize,
            prompt_embedding_save_dtype=args.prompt_embedding_save_dtype,
        )

    def ensure_prompt_context_for_selected_samples(current_samples, total_selected_count):
        if args.rrf_only or not current_samples:
            return current_samples
        if all(sample.get("prompt_context_materialized", False) for sample in current_samples):
            return current_samples

        print(
            "Hybrid prompt materialization: "
            f"enriching {len(current_samples)}/{int(total_selected_count)} selected samples only."
        )
        materialize_samples_prompt_context(
            current_samples,
            edges_df=ctx.edges,
            entity_map=ctx.prompt_entity_map,
            history_window=args.history_window,
            semantic_history=args.semantic_history,
            semantic_topk=args.semantic_topk,
            semantic_history_entity_mode=args.semantic_history_entity_mode,
            common_neighbors_semantic=args.common_neighbors_semantic,
            semantic_use_smoothing=ctx.semantic_use_smoothing,
            semantic_hub_penalty_alpha=args.semantic_hub_penalty_alpha,
            semantic_fusion_alpha=args.semantic_fusion_alpha,
            semantic_fusion_tau=args.semantic_fusion_tau,
            semantic_fusion_recency_speed=args.semantic_fusion_recency_speed,
            history_pool_size=args.history_pool_size,
            history_pool_window=args.history_pool_window,
            history_preserve_recent_k=args.history_preserve_recent_k,
            embeddings=ctx.embeddings,
            entity_id_to_idx=ctx.entity_id_to_idx,
            embedding_model=args.embedding_model,
            embedding_cache=args.embedding_cache,
            populate_prompt_lists=True,
            calibrate_key_signals=needs_contextual_key_signal_calibration,
            key_signal_reference=args.key_signal_reference,
            heuristic_recent_degree_window=args.heuristic_recent_degree_window,
            monitor_label="Hybrid prompt materialization",
        )
        current_samples = ctx.maybe_apply_trial_validation_key_signals(current_samples)
        return current_samples

    def select_trial_hybrid_samples(current_samples):
        if args.hybrid_selection_mode == "pointwise_fixed_threshold_band":
            selected_sample_indices, selection_meta = (
                select_rrf_middle_sample_indices_pointwise_threshold_band(
                    samples=current_samples,
                    low_threshold=args.hybrid_pointwise_low_threshold,
                    high_threshold=args.hybrid_pointwise_high_threshold,
                    score_field=ctx.hybrid_backbone_score_field,
                )
            )
        elif args.hybrid_selection_mode == "random_sample":
            selected_sample_indices, selection_meta = select_random_sample_indices(
                samples=current_samples,
                target_fraction=args.hybrid_validation_target_fraction,
                random_seed=ctx.current_seed + split_idx * 1_000_003,
            )
        elif args.hybrid_selection_mode == "learned_router_top_fraction":
            selected_sample_indices, selection_meta = select_learned_router_indices(
                samples=current_samples,
                target_fraction=args.hybrid_validation_target_fraction,
                router_checkpoint=args.hybrid_router_checkpoint,
            )
        elif args.hybrid_selection_mode in {
            "tabicl_router",
            "validation_fitted_tabicl_router",
        }:
            return ensure_trial_tabicl_router(current_samples)
        elif ctx.use_validation_sampled_hybrid_band:
            if ctx.trial_validation_sampled_hybrid_band is None:
                raise RuntimeError(
                    "validation_sampled_uncertainty_band routing requires "
                    "a trial validation calibration payload."
                )
            selected_sample_indices, selection_meta = (
                select_rrf_middle_sample_indices_validation_sampled_band(
                    samples=current_samples,
                    calibration=ctx.trial_validation_sampled_hybrid_band,
                    score_field=ctx.hybrid_backbone_score_field,
                )
            )
        elif ctx.use_validation_sampled_hybrid_gmm:
            if ctx.trial_validation_sampled_hybrid_band is None:
                raise RuntimeError(
                    "validation_sampled_gmm_overlap_band routing requires "
                    "a trial validation calibration payload."
                )
            selected_sample_indices, selection_meta = (
                select_rrf_middle_sample_indices_validation_sampled_gmm_overlap(
                    samples=current_samples,
                    calibration=ctx.trial_validation_sampled_hybrid_band,
                    score_field=ctx.hybrid_backbone_score_field,
                )
            )
        else:
            selected_sample_indices, selection_meta = select_topk_rrf_middle_sample_indices(
                samples=current_samples,
                top_k=args.hybrid_uncertain_topk_queries,
                dtgb_eval_batch_size=args.dtgb_eval_batch_size,
                score_field=ctx.hybrid_backbone_score_field,
            )
        return normalize_hybrid_selection(
            selected_sample_indices=selected_sample_indices,
            selection_meta=selection_meta,
            total_samples=len(current_samples),
        )

    def ensure_trial_tabicl_train_context():
        if isinstance(ctx.trial_hybrid_alignment_calibration, dict):
            return ctx.trial_hybrid_alignment_calibration
        if dp_ctx["enabled"]:
            raise RuntimeError(
                "TabICL router/fusion currently requires --data_parallel_size 1"
            )

        context_positive_queries, context_negative_ratio = (
            _resolve_hybrid_alignment_calibration_sizes(args)
        )
        if int(context_negative_ratio) != 1:
            raise ValueError(
                "TabICL router/fusion requires exactly one matched negative per "
                "train-context positive query"
            )
        context_sampling = str(
            getattr(args, "tabicl_context_sampling", "most_recent")
        ).strip().lower()
        if context_sampling == "uniform_recent_pool":
            context_pool_positive_queries = int(
                getattr(args, "tabicl_context_pool_positive_queries", 0)
            )
            if context_pool_positive_queries < int(context_positive_queries):
                raise ValueError(
                    "--tabicl_context_pool_positive_queries must be at least "
                    "the TabICL alignment positive-query count: "
                    f"{context_pool_positive_queries} < {context_positive_queries}"
                )
            context_positive_sample_mode = "most_recent"
        elif context_sampling == "uniform_all_train":
            context_pool_positive_queries = 0
            context_positive_sample_mode = "random"
        else:
            context_pool_positive_queries = int(context_positive_queries)
            context_positive_sample_mode = "most_recent"
        pool_description = (
            "all_eligible_train"
            if context_sampling == "uniform_all_train"
            else str(context_pool_positive_queries)
        )
        print(
            "Building shared TabICL context from the train split: "
            f"sampling={context_sampling}, "
            f"positive_queries={context_positive_queries}, "
            f"positive_query_pool={pool_description}, "
            "negative_ratio=1, gnn_rows_may_have_been_seen=True",
            flush=True,
        )
        context_samples = create_test_samples(
            ctx.edges,
            ctx.prompt_entity_map,
            ctx.relation_map,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
            num_samples=(
                context_positive_queries
                if context_sampling == "uniform_all_train"
                else context_pool_positive_queries
            ),
            negative_ratio=context_negative_ratio,
            negative_sampling_mode=args.negative_sampling_mode,
            negative_sampling_batch_size=args.dtgb_eval_batch_size,
            random_seed=ctx.current_seed,
            positive_sample_mode=context_positive_sample_mode,
            history_window=args.history_window,
            semantic_history=args.semantic_history,
            semantic_history_entity_mode=args.semantic_history_entity_mode,
            common_neighbors_semantic=args.common_neighbors_semantic,
            build_prompt_features=(not args.rrf_only),
            defer_prompt_context_materialization=(not args.rrf_only),
            semantic_use_smoothing=ctx.semantic_use_smoothing,
            semantic_hub_penalty_alpha=args.semantic_hub_penalty_alpha,
            semantic_fusion_alpha=args.semantic_fusion_alpha,
            semantic_fusion_tau=args.semantic_fusion_tau,
            semantic_fusion_recency_speed=args.semantic_fusion_recency_speed,
            semantic_topk=args.semantic_topk,
            history_pool_size=args.history_pool_size,
            history_pool_window=args.history_pool_window,
            history_preserve_recent_k=args.history_preserve_recent_k,
            embeddings=ctx.embeddings,
            entity_id_to_idx=ctx.entity_id_to_idx,
            embedding_model=args.embedding_model,
            embedding_cache=args.embedding_cache,
            compute_expert_prediction=runtime_compute_expert_prediction,
            compute_rrf_scores=(
                args.force_compute_rrf_scores
                or runtime_compute_expert_prediction
                or runtime_include_overall_structural_signal
                or (not ctx.hybrid_backbone_is_semantic)
            ),
            rrf_k=args.rrf_k,
            rrf_mode=args.rrf_mode,
            sequential_rank_bins=args.sequential_rank_bins,
            expert_prediction_mode=runtime_expert_prediction_mode,
            expert_prediction_fixed_threshold=args.expert_prediction_fixed_threshold,
            rrf_pointwise_pool_size=args.rrf_pointwise_pool_size,
            rrf_pointwise_num_pools=args.rrf_pointwise_num_pools,
            rrf_batch_size=args.rrf_batch_size,
            rrf_heuristics=args.rrf_heuristics,
            eval_split="train",
            key_signal_reference=runtime_key_signal_reference,
            key_signal_fields=ctx.selected_key_signal_fields,
            include_overall_structural_signal=runtime_include_overall_structural_signal,
            overall_signal_low_threshold=args.overall_signal_low_threshold,
            overall_signal_high_threshold=args.overall_signal_high_threshold,
            defer_postprocessing=True,
            apply_gdelt_time_bucket=ctx.apply_dtgb_time_bucket,
            skip_key_signal_calibration=runtime_skip_key_signal_calibration,
            heuristic_recent_degree_window=args.heuristic_recent_degree_window,
        )
        if context_sampling == "uniform_recent_pool":
            context_samples = _uniform_query_subset(
                context_samples,
                positive_queries=int(context_positive_queries),
                seed=int(ctx.current_seed),
            )
        if context_sampling in {"uniform_recent_pool", "uniform_all_train"}:
            context_positive_timestamps = np.asarray(
                [
                    float(sample["timestamp"])
                    for sample in context_samples
                    if int(sample["label"]) == 1
                ],
                dtype=np.float64,
            )
            print(
                "Uniform TabICL context selected: "
                f"sampling={context_sampling}, "
                f"rows={len(context_samples)}, "
                f"unique_positive_timestamps="
                f"{np.unique(context_positive_timestamps).size}, "
                f"positive_timestamp_median="
                f"{np.median(context_positive_timestamps):g}, "
                f"positive_timestamp_range="
                f"{context_positive_timestamps.min():g}:"
                f"{context_positive_timestamps.max():g}",
                flush=True,
            )
        context_samples = finalize_test_samples(
            samples=context_samples,
            edges_df=ctx.edges,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            negative_ratio=context_negative_ratio,
            random_seed=ctx.current_seed,
            build_prompt_features=False,
            compute_expert_prediction=runtime_compute_expert_prediction,
            compute_rrf_scores=(
                args.force_compute_rrf_scores
                or runtime_compute_expert_prediction
                or runtime_include_overall_structural_signal
                or (not ctx.hybrid_backbone_is_semantic)
            ),
            rrf_k=args.rrf_k,
            rrf_mode=args.rrf_mode,
            sequential_rank_bins=args.sequential_rank_bins,
            expert_prediction_mode=runtime_expert_prediction_mode,
            expert_prediction_fixed_threshold=args.expert_prediction_fixed_threshold,
            key_signal_fields=ctx.selected_key_signal_fields,
            rrf_pointwise_pool_size=args.rrf_pointwise_pool_size,
            rrf_pointwise_num_pools=args.rrf_pointwise_num_pools,
            rrf_batch_size=args.rrf_batch_size,
            rrf_heuristics=args.rrf_heuristics,
            key_signal_reference=runtime_key_signal_reference,
            include_overall_structural_signal=runtime_include_overall_structural_signal,
            heuristic_recent_degree_window=args.heuristic_recent_degree_window,
            overall_signal_low_threshold=args.overall_signal_low_threshold,
            overall_signal_high_threshold=args.overall_signal_high_threshold,
            apply_gdelt_time_bucket=ctx.apply_dtgb_time_bucket,
            skip_key_signal_calibration=runtime_skip_key_signal_calibration,
        )
        context_samples = ctx.maybe_apply_expert_signal_annotations(
            context_samples,
            include_expert_prediction=include_expert_prediction,
            active_split="train",
        )
        context_samples = ctx.annotate_hybrid_backbone_scores(
            context_samples,
            active_split="train",
        )
        context_labels = np.asarray(
            [int(sample["label"]) for sample in context_samples], dtype=np.int64
        )
        context_query_ids = np.asarray(
            [int(sample["query_id"]) for sample in context_samples], dtype=np.int64
        )
        if (
            np.count_nonzero(context_labels == 1) != context_positive_queries
            or np.count_nonzero(context_labels == 0) != context_positive_queries
            or np.unique(context_query_ids).size != context_positive_queries
        ):
            raise RuntimeError(
                "TabICL train context is not exactly one positive/negative pair "
                "per query"
            )
        ensure_prompt_context_for_selected_samples(
            context_samples,
            total_selected_count=len(context_samples),
        )
        context_results = run_eval(
            context_samples,
            dtgb_eval_batch_size_override=False,
            active_eval_split="train",
        )
        context_llm_scores = np.asarray(
            context_results.get("predictions", []), dtype=np.float64
        )
        if context_llm_scores.shape != (len(context_samples),):
            raise RuntimeError(
                "TabICL train-context LLM predictions do not match context rows"
            )
        ctx.trial_hybrid_alignment_calibration = {
            "samples": context_samples,
            "llm_scores": context_llm_scores.copy(),
            "positive_queries": int(context_positive_queries),
            "pool_positive_queries": int(context_pool_positive_queries),
            "negative_ratio": int(context_negative_ratio),
            "split": "train",
            "selection": context_sampling,
            "gnn_context_seen": True,
        }
        ctx.trial_hybrid_score_alignment = {
            "mode": "tabicl",
            "fit_scope": f"train_{context_sampling}",
            "calibration_sample_count": int(len(context_samples)),
            "calibration_num_samples": int(context_positive_queries),
            "context_pool_positive_queries": int(context_pool_positive_queries),
            "calibration_negative_ratio": int(context_negative_ratio),
            "context_selection": context_sampling,
            "context_gnn_seen": True,
        }
        print(
            "Shared TabICL train context ready: "
            f"rows={len(context_samples)}, queries={context_positive_queries}, "
            "all_call_llm_rows="
            f"{len(context_llm_scores)}, gnn_context_seen=True",
            flush=True,
        )
        return ctx.trial_hybrid_alignment_calibration

    def ensure_trial_hybrid_score_alignment():
        if args.hybrid_score_alignment_mode == "off":
            return None
        if ctx.trial_hybrid_score_alignment is not None:
            return ctx.trial_hybrid_score_alignment
        if str(args.hybrid_score_alignment_mode).strip().lower() == "tabicl":
            ensure_trial_tabicl_train_context()
            return ctx.trial_hybrid_score_alignment
        alignment_payload_name = "hybrid_score_alignment"

        alignment_num_samples, alignment_negative_ratio = (
            _resolve_hybrid_alignment_calibration_sizes(args)
        )
        if dp_ctx["enabled"] and dp_ctx["rank"] != 0:
            alignment_dp_trial_idx = ctx.trial
            alignment_split_name = "alignment_validation"
            alignment_validation_samples = wait_for_shared_samples(
                sync_dir=dp_ctx["sync_dir"],
                run_id=dp_ctx["run_id"],
                trial_idx=alignment_dp_trial_idx,
                split_name=alignment_split_name,
                timeout_sec=dp_ctx["timeout_sec"],
            )
            print(
                f"DP alignment sample bundle loaded: {len(alignment_validation_samples)} samples "
                f"for trial {ctx.trial + 1} [{eval_split}]."
            )
            alignment_positions, alignment_eval_samples = split_samples_for_rank(
                samples=alignment_validation_samples,
                dp_size=dp_ctx["size"],
                dp_rank=dp_ctx["rank"],
            )
            print(
                f"DP alignment shard assignment: rank {dp_ctx['rank']}/{dp_ctx['size']} "
                f"handles {len(alignment_eval_samples)}/{len(alignment_validation_samples)} validation alignment samples."
            )
            if alignment_eval_samples:
                alignment_local_results = run_eval(
                    alignment_eval_samples,
                    dtgb_eval_batch_size_override=False,
                    active_eval_split="validation",
                )
            else:
                alignment_local_results = {
                    "predictions": [],
                    "labels": [],
                    "parse_stats": {},
                    "detailed_results": [],
                }
            alignment_shard_path = write_hybrid_selected_shard(
                sync_dir=dp_ctx["sync_dir"],
                run_id=dp_ctx["run_id"],
                trial_idx=alignment_dp_trial_idx,
                split_name=alignment_split_name,
                rank=dp_ctx["rank"],
                selected_count=len(alignment_validation_samples),
                selected_positions=alignment_positions,
                trial_results=alignment_local_results,
            )
            print(f"DP alignment shard saved: {alignment_shard_path}")
            ctx.trial_hybrid_score_alignment = wait_for_shared_payload(
                sync_dir=dp_ctx["sync_dir"],
                run_id=dp_ctx["run_id"],
                trial_idx=ctx.trial,
                payload_name=alignment_payload_name,
                timeout_sec=dp_ctx["timeout_sec"],
            )
            return ctx.trial_hybrid_score_alignment

        print(
            "Calibrating hybrid score alignment on sampled validation slice: "
            f"validation positives={alignment_num_samples}, "
            f"negative_ratio={alignment_negative_ratio}, "
            f"mode={args.hybrid_score_alignment_mode}"
        )
        calibration_samples = _build_lightweight_rrf_samples_for_split(
            ctx.edges,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
            num_samples=alignment_num_samples,
            negative_ratio=alignment_negative_ratio,
            random_seed=ctx.current_seed,
            eval_split="validation",
            apply_gdelt_time_bucket=ctx.apply_dtgb_time_bucket,
        )
        calibration_samples = finalize_test_samples(
            samples=calibration_samples,
            edges_df=ctx.edges,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            negative_ratio=alignment_negative_ratio,
            random_seed=ctx.current_seed,
            build_prompt_features=False,
            compute_expert_prediction=runtime_compute_expert_prediction,
            compute_rrf_scores=(
                args.force_compute_rrf_scores
                or runtime_compute_expert_prediction
                or runtime_include_overall_structural_signal
                or (not ctx.hybrid_backbone_is_semantic)
            ),
            rrf_k=args.rrf_k,
            rrf_mode=args.rrf_mode,
            sequential_rank_bins=args.sequential_rank_bins,
            expert_prediction_mode=runtime_expert_prediction_mode,
            expert_prediction_fixed_threshold=args.expert_prediction_fixed_threshold,
            key_signal_fields=ctx.selected_key_signal_fields,
            rrf_pointwise_pool_size=args.rrf_pointwise_pool_size,
            rrf_pointwise_num_pools=args.rrf_pointwise_num_pools,
            rrf_batch_size=args.rrf_batch_size,
            rrf_heuristics=args.rrf_heuristics,
            key_signal_reference=runtime_key_signal_reference,
            include_overall_structural_signal=runtime_include_overall_structural_signal,
            heuristic_recent_degree_window=args.heuristic_recent_degree_window,
            overall_signal_low_threshold=args.overall_signal_low_threshold,
            overall_signal_high_threshold=args.overall_signal_high_threshold,
            apply_gdelt_time_bucket=ctx.apply_dtgb_time_bucket,
            skip_key_signal_calibration=runtime_skip_key_signal_calibration,
        )
        calibration_samples = ctx.maybe_apply_expert_signal_annotations(
            calibration_samples,
            include_expert_prediction=include_expert_prediction,
            active_split="validation",
        )
        calibration_samples = ctx.annotate_hybrid_backbone_scores(
            calibration_samples,
            active_split="validation",
        )

        alignment_fit_scope = str(
            getattr(args, "hybrid_score_alignment_fit_scope", "validation_all")
        ).strip().lower()
        alignment_validation_samples = calibration_samples
        if alignment_fit_scope == "validation_selected":
            alignment_selected_indices, alignment_selection = select_trial_hybrid_samples(
                calibration_samples
            )
            if not alignment_selected_indices:
                raise RuntimeError(
                    "Hybrid score alignment fit scope 'validation_selected' produced "
                    "zero routed validation calibration samples."
                )
            alignment_validation_samples = [
                calibration_samples[idx] for idx in alignment_selected_indices
            ]
            print(
                "Hybrid score alignment fit scope: validation_selected "
                f"({len(alignment_validation_samples)}/{len(calibration_samples)} "
                "validation samples kept after routing)."
            )
        else:
            alignment_selection = None
            print(
                "Hybrid score alignment fit scope: validation_all "
                f"({len(alignment_validation_samples)} validation samples)."
            )
        if not alignment_validation_samples:
            raise RuntimeError(
                "Hybrid score alignment calibration produced zero validation samples."
            )

        for sample in alignment_validation_samples:
            source_id = int(sample["source_id"])
            target_id = int(sample["target_id"])
            relation_id = int(sample["relation_id"])
            sample.setdefault("source_entity", ctx.prompt_entity_map.get(source_id, f"entity_{source_id}"))
            sample.setdefault("target_entity", ctx.prompt_entity_map.get(target_id, f"entity_{target_id}"))
            sample.setdefault("relation", ctx.relation_map.get(relation_id, f"relation_{relation_id}"))
            sample.setdefault("source_history", [])
            sample.setdefault("target_history", [])
            sample.setdefault("source_history_entities", [])
            sample.setdefault("target_history_entities", [])
            sample.setdefault("mutual_history", [])
            sample.setdefault("common_neighbors", [])
            sample.setdefault("common_neighbors_desc", "sorted by popularity")
            sample.setdefault("source_history_desc", "most recent")
            sample.setdefault("target_history_desc", "most recent")
            sample.setdefault("prompt_context_materialized", False)

        if dp_ctx["enabled"]:
            alignment_dp_trial_idx = ctx.trial
            alignment_split_name = "alignment_validation"
            if dp_ctx["rank"] == 0:
                ensure_prompt_context_for_selected_samples(
                    alignment_validation_samples,
                    total_selected_count=len(alignment_validation_samples),
                )
                alignment_bundle_path = write_shared_samples(
                    sync_dir=dp_ctx["sync_dir"],
                    run_id=dp_ctx["run_id"],
                    trial_idx=alignment_dp_trial_idx,
                    split_name=alignment_split_name,
                    samples=alignment_validation_samples,
                )
                print(
                    "DP alignment sample bundle saved: "
                    f"{alignment_bundle_path} ({len(alignment_validation_samples)} samples)."
                )
            else:
                alignment_validation_samples = wait_for_shared_samples(
                    sync_dir=dp_ctx["sync_dir"],
                    run_id=dp_ctx["run_id"],
                    trial_idx=alignment_dp_trial_idx,
                    split_name=alignment_split_name,
                    timeout_sec=dp_ctx["timeout_sec"],
                )
                print(
                    f"DP alignment sample bundle loaded: {len(alignment_validation_samples)} samples "
                    f"for trial {ctx.trial + 1} [{eval_split}]."
                )

            alignment_positions, alignment_eval_samples = split_samples_for_rank(
                samples=alignment_validation_samples,
                dp_size=dp_ctx["size"],
                dp_rank=dp_ctx["rank"],
            )
            print(
                f"DP alignment shard assignment: rank {dp_ctx['rank']}/{dp_ctx['size']} "
                f"handles {len(alignment_eval_samples)}/{len(alignment_validation_samples)} validation alignment samples."
            )
            if alignment_eval_samples:
                alignment_local_results = run_eval(
                    alignment_eval_samples,
                    dtgb_eval_batch_size_override=False,
                    active_eval_split="validation",
                )
            else:
                alignment_local_results = {
                    "predictions": [],
                    "labels": [],
                    "parse_stats": {},
                    "detailed_results": [],
                }
            alignment_shard_path = write_hybrid_selected_shard(
                sync_dir=dp_ctx["sync_dir"],
                run_id=dp_ctx["run_id"],
                trial_idx=alignment_dp_trial_idx,
                split_name=alignment_split_name,
                rank=dp_ctx["rank"],
                selected_count=len(alignment_validation_samples),
                selected_positions=alignment_positions,
                trial_results=alignment_local_results,
            )
            print(f"DP alignment shard saved: {alignment_shard_path}")

            if dp_ctx["rank"] == 0:
                alignment_shard_paths = wait_for_hybrid_selected_shards(
                    sync_dir=dp_ctx["sync_dir"],
                    run_id=dp_ctx["run_id"],
                    trial_idx=alignment_dp_trial_idx,
                    split_name=alignment_split_name,
                    dp_size=dp_ctx["size"],
                    timeout_sec=dp_ctx["timeout_sec"],
                )
                alignment_validation_results = merge_hybrid_selected_shards(
                    alignment_shard_paths,
                    selected_count=len(alignment_validation_samples),
                )
                print(
                    f"DP alignment merge complete for trial {ctx.trial + 1} [{eval_split}]: "
                    f"{len(alignment_shard_paths)} shards -> {len(alignment_validation_samples)} validation samples"
                )
            else:
                return None
        else:
            ensure_prompt_context_for_selected_samples(
                alignment_validation_samples,
                total_selected_count=len(alignment_validation_samples),
            )
            alignment_validation_results = run_eval(
                alignment_validation_samples,
                dtgb_eval_batch_size_override=False,
                active_eval_split="validation",
            )

        llm_validation_scores = np.asarray(
            alignment_validation_results.get("predictions", []),
            dtype=np.float64,
        )
        if llm_validation_scores.shape[0] != len(alignment_validation_samples):
            raise RuntimeError(
                "Hybrid score alignment calibration predictions do not match "
                "the validation calibration sample size."
            )
        ctx.trial_hybrid_alignment_calibration = {
            "samples": alignment_validation_samples,
            "llm_scores": llm_validation_scores.copy(),
        }
        backbone_validation_scores = prepare_backbone_score_space(
            np.asarray(
                [
                    float(sample.get(ctx.hybrid_backbone_score_field, 0.0))
                    for sample in alignment_validation_samples
                ],
                dtype=np.float64,
            ),
            score_space=getattr(args, "hybrid_backbone_score_space", "minmax"),
        )
        alignment_mode = str(args.hybrid_score_alignment_mode).strip().lower()
        if alignment_mode == "validation_selected_quantile_match":
            ctx.trial_hybrid_score_alignment = fit_quantile_score_alignment(
                llm_validation_scores,
                backbone_validation_scores,
            )
            ctx.trial_hybrid_score_alignment["mode"] = "validation_selected_quantile_match"
            ctx.trial_hybrid_score_alignment["target_name"] = str(ctx.hybrid_backbone_score_field)
        elif alignment_mode == "validation_selected_isotonic_regression":
            ctx.trial_hybrid_score_alignment = fit_isotonic_score_alignment(
                llm_validation_scores,
                backbone_validation_scores,
            )
            ctx.trial_hybrid_score_alignment["mode"] = "validation_selected_isotonic_regression"
            ctx.trial_hybrid_score_alignment["target_name"] = str(ctx.hybrid_backbone_score_field)
        else:
            raise ValueError(
                f"Unsupported hybrid score alignment mode: {args.hybrid_score_alignment_mode}"
            )
        ctx.trial_hybrid_score_alignment["fit_scope"] = alignment_fit_scope
        ctx.trial_hybrid_score_alignment["calibration_sample_count"] = int(
            len(alignment_validation_samples)
        )
        ctx.trial_hybrid_score_alignment["calibration_total_sample_count"] = int(
            len(calibration_samples)
        )
        ctx.trial_hybrid_score_alignment["calibration_num_samples"] = int(alignment_num_samples)
        ctx.trial_hybrid_score_alignment["calibration_negative_ratio"] = int(
            alignment_negative_ratio
        )
        ctx.trial_hybrid_score_alignment["calibration_selection_mode"] = (
            alignment_selection.get("selection_mode")
            if isinstance(alignment_selection, dict)
            else "full_validation_slice"
        )
        if dp_ctx["enabled"] and dp_ctx["rank"] == 0:
            write_shared_payload(
                sync_dir=dp_ctx["sync_dir"],
                run_id=dp_ctx["run_id"],
                trial_idx=ctx.trial,
                payload_name=alignment_payload_name,
                payload=ctx.trial_hybrid_score_alignment,
            )
        print(
            "Hybrid score alignment ready: "
            f"mode={ctx.trial_hybrid_score_alignment.get('mode', 'unknown')}, "
            f"fit_scope={alignment_fit_scope}, "
            f"validation_samples={len(alignment_validation_samples)}"
        )
        return ctx.trial_hybrid_score_alignment

    def ensure_trial_tabicl_router(current_samples):
        cached = ctx.split_runtime[eval_split].get("tabicl_router_selection")
        if cached is not None:
            return cached
        if dp_ctx["enabled"]:
            raise RuntimeError(
                "tabicl_router currently requires "
                "--data_parallel_size 1"
            )

        # The same all-call train slice supplies grouped-OOF router targets and
        # the sparse-context final TabICL fusion model.
        ensure_trial_hybrid_score_alignment()
        calibration = ctx.trial_hybrid_alignment_calibration
        if not isinstance(calibration, dict):
            raise RuntimeError(
                "TabICL routing could not access train-context LLM rows"
            )
        band = ctx.trial_validation_sampled_hybrid_band
        if not isinstance(band, dict) or band.get("center_threshold") is None:
            raise RuntimeError(
                "TabICL routing requires a validation-calibrated backbone center"
            )

        configured_root = getattr(args, "tabicl_router_artifact_dir", None)
        if configured_root:
            artifact_root = Path(configured_root)
        else:
            output_path = Path(args.output)
            artifact_root = output_path.with_name(
                f"{output_path.stem}_tabicl_router"
            )
        selected, selection = run_online_tabicl_router(
            dataset_name=args.dataset_name,
            eval_split=eval_split,
            trial=ctx.trial,
            support_samples=calibration["samples"],
            support_llm_scores=calibration["llm_scores"],
            deployment_samples=current_samples,
            backbone_score_field=ctx.hybrid_backbone_score_field,
            route_center=float(band["center_threshold"]),
            budget_fraction=float(args.hybrid_validation_target_fraction),
            tabicl_python=args.tabicl_router_python,
            tabicl_device=args.tabicl_router_device,
            cuda_visible_devices=args.tabicl_router_cuda_visible_devices,
            artifact_dir=artifact_root,
            folds=int(args.tabicl_router_folds),
            n_estimators=int(args.tabicl_router_n_estimators),
            batch_size=int(args.tabicl_router_batch_size),
            alignment_variant=getattr(
                args,
                "tabicl_alignment_variant",
                "score_fusion_no_heuristics",
            ),
            context_selection=str(calibration["selection"]),
        )
        cached = (selected, selection)
        ctx.split_runtime[eval_split]["tabicl_router_selection"] = cached
        return cached

    if ctx.hybrid_enabled:
        dp_trial_idx = ctx.trial * len(ctx.eval_splits) + split_idx

        if dp_ctx["enabled"] and dp_ctx["rank"] != 0:
            selected_sample_indices, hybrid_selection = wait_for_hybrid_selection(
                sync_dir=dp_ctx["sync_dir"],
                run_id=dp_ctx["run_id"],
                trial_idx=dp_trial_idx,
                split_name=eval_split,
                timeout_sec=dp_ctx["timeout_sec"],
            )
            selected_sample_indices, hybrid_selection = normalize_hybrid_selection(
                selected_sample_indices=selected_sample_indices,
                selection_meta=hybrid_selection,
                total_samples=len(samples),
            )
            print(
                f"DP hybrid selection loaded: {len(selected_sample_indices)} samples "
                f"for trial {ctx.trial + 1} [{eval_split}]."
            )
        else:
            selected_sample_indices, hybrid_selection = select_trial_hybrid_samples(samples)
            selected_sample_indices, hybrid_selection = apply_hybrid_debug_route_cap(
                samples,
                selected_sample_indices,
                hybrid_selection,
                score_field=ctx.hybrid_backbone_score_field,
                max_routed_samples=int(getattr(args, "hybrid_debug_max_routed_samples", 0)),
            )
            selected_sample_indices, hybrid_selection = normalize_hybrid_selection(
                selected_sample_indices=selected_sample_indices,
                selection_meta=hybrid_selection,
                total_samples=len(samples),
            )
            if dp_ctx["enabled"]:
                selection_path = write_hybrid_selection(
                    sync_dir=dp_ctx["sync_dir"],
                    run_id=dp_ctx["run_id"],
                    trial_idx=dp_trial_idx,
                    split_name=eval_split,
                    selected_sample_indices=selected_sample_indices,
                    selection_meta=hybrid_selection,
                )
                print(f"DP hybrid selection saved: {selection_path}")

        selection_mode = str(hybrid_selection.get("selection_mode", "")).strip().lower()
        if selection_mode == "pointwise_fixed_threshold_band":
            print(
                f"Hybrid selection [{ctx.hybrid_backbone_label}]: {hybrid_selection['selected_count']} middle-band samples "
                f"(strict pointwise thresholds: "
                f"{hybrid_selection.get('low_threshold', args.hybrid_pointwise_low_threshold):.6f} "
                f"<= score < "
                f"{hybrid_selection.get('high_threshold', args.hybrid_pointwise_high_threshold):.6f})."
            )
        elif selection_mode == "validation_sampled_uncertainty_band":
            print(
                f"Hybrid selection [{ctx.hybrid_backbone_label}]: {hybrid_selection['selected_count']} pointwise uncertainty-band samples "
                f"(validation-frozen center={hybrid_selection.get('center_threshold', 0.0):.6f}, "
                f"low={hybrid_selection.get('low_threshold', 0.0):.6f}, "
                f"high={hybrid_selection.get('high_threshold', 0.0):.6f}, "
                f"requested_fraction={hybrid_selection.get('validation_requested_target_fraction', 0.0):.3f}, "
                f"realized_fraction={hybrid_selection.get('selected_fraction_realized', 0.0):.3f})."
            )
        elif selection_mode == "validation_sampled_gmm_overlap_band":
            means = hybrid_selection.get("gmm_means", [0.0, 0.0])
            weights = hybrid_selection.get("gmm_weights", [0.0, 0.0])
            print(
                f"Hybrid selection [{ctx.hybrid_backbone_label}]: {hybrid_selection['selected_count']} GMM-overlap samples "
                f"(means=({float(means[0]):.6f}, {float(means[1]):.6f}), "
                f"weights=({float(weights[0]):.3f}, {float(weights[1]):.3f}), "
                f"ambiguity_threshold={hybrid_selection.get('ambiguity_threshold', 0.0):.6f}, "
                f"requested_fraction={hybrid_selection.get('validation_requested_target_fraction', 0.0):.3f}, "
                f"realized_fraction={hybrid_selection.get('selected_fraction_realized', 0.0):.3f})."
            )
        elif selection_mode == "random_sample":
            print(
                f"Hybrid selection [{ctx.hybrid_backbone_label}]: {hybrid_selection['selected_count']} random samples "
                f"(requested_fraction={hybrid_selection.get('requested_target_fraction', 0.0):.3f}, "
                f"realized_fraction={hybrid_selection.get('selected_fraction_realized', 0.0):.3f}, "
                f"seed={hybrid_selection.get('random_seed', ctx.current_seed)})."
            )
        elif selection_mode == "learned_router_top_fraction":
            print(
                f"Hybrid selection [{ctx.hybrid_backbone_label}]: "
                f"{hybrid_selection['selected_count']} learned-router samples "
                f"from {hybrid_selection.get('num_candidates', len(samples))} full-split candidates "
                f"(requested_fraction={hybrid_selection.get('requested_target_fraction', 0.0):.3f}, "
                f"realized_fraction={hybrid_selection.get('selected_fraction_realized', 0.0):.3f}, "
                f"checkpoint={hybrid_selection.get('router_checkpoint')})."
            )
        elif selection_mode == "tabicl_router":
            print(
                f"Hybrid selection [{ctx.hybrid_backbone_label}]: "
                f"{hybrid_selection['selected_count']} train-fitted TabICL "
                f"router samples from "
                f"{hybrid_selection.get('num_candidates', len(samples))} candidates "
                f"(requested_fraction="
                f"{hybrid_selection.get('requested_target_fraction', 0.0):.3f}, "
                f"checkpoint={hybrid_selection.get('router_checkpoint')})."
            )
        else:
            print(
                f"Hybrid selection [{ctx.hybrid_backbone_label}]: {hybrid_selection['selected_count']} middle-band samples "
                f"across {hybrid_selection.get('selected_batches', 0)} DTGB batches "
                f"(requested {hybrid_selection['requested_top_k']} per DTGB batch)."
            )
        if hybrid_selection.get("debug_route_cap_enabled"):
            print(
                "Hybrid debug route cap applied: "
                f"{hybrid_selection.get('selected_count_before_debug_cap', 0)} -> "
                f"{hybrid_selection.get('selected_count', 0)} "
                f"(cap={hybrid_selection.get('debug_route_cap', 0)}, "
                f"priority={hybrid_selection.get('debug_route_cap_reason', 'selection_order')})."
            )
        if not selected_sample_indices:
            raise RuntimeError("No valid middle-band samples selected for hybrid evaluation.")

        if dp_ctx["enabled"]:
            selected_samples = [samples[idx] for idx in selected_sample_indices]
            selected_positions, selected_eval_samples = split_samples_for_rank(
                samples=selected_samples,
                dp_size=dp_ctx["size"],
                dp_rank=dp_ctx["rank"],
            )
            print(
                f"DP hybrid shard assignment: rank {dp_ctx['rank']}/{dp_ctx['size']} "
                f"handles {len(selected_eval_samples)}/{len(selected_samples)} selected samples."
            )
            if selected_eval_samples:
                ensure_prompt_context_for_selected_samples(
                    selected_eval_samples,
                    total_selected_count=len(selected_samples),
                )
                local_selected_results = run_eval(
                    selected_eval_samples,
                    dtgb_eval_batch_size_override=False,
                    capture_embeddings=True,
                )
            else:
                local_selected_results = {
                    "predictions": [],
                    "labels": [],
                    "parse_stats": {},
                    "detailed_results": [],
                }
            selected_shard_path = write_hybrid_selected_shard(
                sync_dir=dp_ctx["sync_dir"],
                run_id=dp_ctx["run_id"],
                trial_idx=dp_trial_idx,
                split_name=eval_split,
                rank=dp_ctx["rank"],
                selected_count=len(selected_samples),
                selected_positions=selected_positions,
                trial_results=local_selected_results,
            )
            print(f"DP hybrid selected shard saved: {selected_shard_path}")
            hybrid_score_alignment = ensure_trial_hybrid_score_alignment()

            if dp_ctx["rank"] == 0:
                selected_shard_paths = wait_for_hybrid_selected_shards(
                    sync_dir=dp_ctx["sync_dir"],
                    run_id=dp_ctx["run_id"],
                    trial_idx=dp_trial_idx,
                    split_name=eval_split,
                    dp_size=dp_ctx["size"],
                    timeout_sec=dp_ctx["timeout_sec"],
                )
                merged_selected_results = merge_hybrid_selected_shards(
                    selected_shard_paths,
                    selected_count=len(selected_samples),
                )
                print(
                    f"DP hybrid selected merge complete for trial {ctx.trial + 1} [{eval_split}]: "
                    f"{len(selected_shard_paths)} shards -> {len(selected_samples)} selected samples"
                )
                if len(merged_selected_results.get("predictions", [])) != len(selected_samples):
                    raise RuntimeError(
                        "Merged DP hybrid selected predictions do not match selected sample count."
                    )

                def run_llm_on_merged_selected_samples(current_samples):
                    if len(current_samples) != len(selected_samples):
                        raise RuntimeError(
                            "Hybrid selected-slice size mismatch between selection and merged DP shard outputs."
                        )
                    return merged_selected_results

                results = evaluate_budgeted_hybrid_rrf_llm(
                    samples=samples,
                    selected_sample_indices=selected_sample_indices,
                    run_llm_on_samples=run_llm_on_merged_selected_samples,
                    selection_meta=hybrid_selection,
                    score_alignment=hybrid_score_alignment,
                    fusion_alpha=args.hybrid_fusion_alpha,
                    merge_method=args.hybrid_merge_method,
                    dtgb_eval_batch_size=args.dtgb_eval_batch_size,
                    backbone_score_field=ctx.hybrid_backbone_score_field,
                    backbone_name=ctx.hybrid_backbone_label,
                    backbone_score_space=getattr(args, "hybrid_backbone_score_space", "minmax"),
                )
            else:
                results = None
        else:
            hybrid_score_alignment = ensure_trial_hybrid_score_alignment()
            full_score_fusion = None
            if str(args.hybrid_score_alignment_mode).strip().lower() == "tabicl":
                calibration = ctx.trial_hybrid_alignment_calibration
                if not isinstance(calibration, dict):
                    raise RuntimeError(
                        "TabICL final fusion lacks its shared train context"
                    )
                router_table = hybrid_selection.get("router_table")
                if not router_table:
                    band = ctx.trial_validation_sampled_hybrid_band
                    if (
                        args.hybrid_selection_mode
                        != "validation_sampled_uncertainty_band"
                        or not isinstance(band, dict)
                        or band.get("center_threshold") is None
                    ):
                        raise RuntimeError(
                            "TabICL final fusion requires either tabicl_router "
                            "selection or a validation-calibrated uncertainty band"
                        )
                    configured_root = getattr(
                        args, "tabicl_router_artifact_dir", None
                    )
                    if configured_root:
                        artifact_root = Path(configured_root)
                    else:
                        output_path = Path(args.output)
                        artifact_root = output_path.with_name(
                            f"{output_path.stem}_tabicl_alignment"
                        )
                    router_table, table_metadata = (
                        prepare_online_tabicl_fusion_table(
                            eval_split=eval_split,
                            trial=ctx.trial,
                            support_samples=calibration["samples"],
                            support_llm_scores=calibration["llm_scores"],
                            deployment_samples=samples,
                            backbone_score_field=ctx.hybrid_backbone_score_field,
                            route_center=float(band["center_threshold"]),
                            budget_fraction=float(
                                args.hybrid_validation_target_fraction
                            ),
                            artifact_dir=artifact_root,
                            context_selection=str(calibration["selection"]),
                        )
                    )
                    hybrid_selection["tabicl_fusion_table"] = router_table
                    hybrid_selection["tabicl_fusion_table_metadata"] = table_metadata

                def full_score_fusion(llm_selected_scores_raw):
                    fused_scores, fusion_metadata = run_online_tabicl_fusion(
                        table_path=router_table,
                        selected_indices=selected_sample_indices,
                        selected_llm_scores=llm_selected_scores_raw,
                        deployment_samples=samples,
                        tabicl_python=args.tabicl_router_python,
                        tabicl_device=args.tabicl_router_device,
                        cuda_visible_devices=args.tabicl_router_cuda_visible_devices,
                        positive_context=int(calibration["positive_queries"]),
                        n_estimators=int(args.tabicl_router_n_estimators),
                        batch_size=int(args.tabicl_router_batch_size),
                        metric_batch_size=int(args.dtgb_eval_batch_size),
                        alignment_variant=getattr(
                            args,
                            "tabicl_alignment_variant",
                            "score_fusion_no_heuristics",
                        ),
                        context_selection=str(calibration["selection"]),
                    )
                    return {
                        "scores": fused_scores,
                        "metadata": fusion_metadata,
                    }

            results = evaluate_budgeted_hybrid_rrf_llm(
                samples=samples,
                selected_sample_indices=selected_sample_indices,
                run_llm_on_samples=lambda current_samples: run_eval(
                    ensure_prompt_context_for_selected_samples(
                        current_samples,
                        total_selected_count=len(selected_sample_indices),
                    ),
                    dtgb_eval_batch_size_override=False,
                    capture_embeddings=True,
                ),
                selection_meta=hybrid_selection,
                score_alignment=hybrid_score_alignment,
                fusion_alpha=args.hybrid_fusion_alpha,
                merge_method=args.hybrid_merge_method,
                dtgb_eval_batch_size=args.dtgb_eval_batch_size,
                backbone_score_field=ctx.hybrid_backbone_score_field,
                backbone_name=ctx.hybrid_backbone_label,
                backbone_score_space=getattr(args, "hybrid_backbone_score_space", "minmax"),
                full_score_fusion=full_score_fusion,
            )

        if results is None:
            print(
                f"  -> Trial {ctx.trial + 1} [{eval_split}]: rank {dp_ctx['rank']} hybrid shard complete."
            )
            return

        proxy = results.get("rrf_uncertainty_proxy", {})
        llm_slice = proxy.get("llm", {})
        backbone_slice = proxy.get("backbone", proxy.get("rrf", {}))
        slice_delta = proxy.get(
            "delta_llm_minus_backbone",
            proxy.get("delta_llm_minus_rrf", {}),
        )
        print(
            "Selected Middle-band Slice: "
            f"LLM(AP={llm_slice.get('ap', 0.0):.4f}, AUC={llm_slice.get('auc', 0.0):.4f}, Acc={llm_slice.get('accuracy', 0.0):.4f}) | "
            f"{ctx.hybrid_backbone_label}(AP={backbone_slice.get('ap', 0.0):.4f}, AUC={backbone_slice.get('auc', 0.0):.4f}, Acc={backbone_slice.get('accuracy', 0.0):.4f})"
        )
        print(
            "Selected Middle-band Slice: "
            f"LLM vs {ctx.hybrid_backbone_label} -> Delta AP={slice_delta.get('ap', 0.0):.4f}, "
            f"Delta AUC={slice_delta.get('auc', 0.0):.4f}, "
            f"Delta Acc={slice_delta.get('accuracy', 0.0):.4f}"
        )

        uplift = results.get("hybrid", {}).get(
            "uplift_vs_backbone",
            results.get("hybrid", {}).get("uplift_vs_rrf", {}),
        )
        print(
            f"Hybrid Full-set Uplift vs {ctx.hybrid_backbone_label} ({args.hybrid_merge_method}): AP={uplift.get('ap', 0.0):.4f}, "
            f"AUC={uplift.get('auc', 0.0):.4f}, "
            f"Acc={uplift.get('accuracy', 0.0):.4f}"
        )

        def _fmt_dbg(value):
            return "n/a" if value is None else f"{float(value):.4f}"

        routing_debug = results.get("hybrid_routing_debug", {})
        if routing_debug:
            routing_delta = routing_debug.get(
                "delta_llm_raw_minus_backbone",
                routing_debug.get("delta_llm_raw_minus_rrf", {}),
            )
            print(
                "Hybrid routing debug: "
                f"selected={routing_debug.get('num_selected_samples', 0)} "
                f"({routing_debug.get('selected_fraction_realized', 0.0):.3f}), "
                f"{ctx.hybrid_backbone_label}(AUC={_fmt_dbg(routing_debug.get('backbone', routing_debug.get('rrf', {})).get('auc'))}) vs "
                f"LLM-raw(AUC={_fmt_dbg(routing_debug.get('llm_raw', {}).get('auc'))}), "
                f"delta_auc={_fmt_dbg(routing_delta.get('auc'))}"
            )
        alignment_debug = results.get("hybrid_alignment_debug", {})
        if alignment_debug:
            dist = alignment_debug.get(
                "distribution_distance_to_backbone",
                alignment_debug.get("distribution_distance_to_rrf", {}),
            )
            raw_dist = dist.get("llm_raw", {})
            aligned_dist = dist.get("llm_aligned", {})
            delta_align = alignment_debug.get("delta_llm_aligned_minus_llm_raw", {})
            print(
                "Hybrid alignment debug: "
                f"LLM-raw(AUC={_fmt_dbg(alignment_debug.get('llm_raw', {}).get('auc'))}) -> "
                f"LLM-aligned(AUC={_fmt_dbg(alignment_debug.get('llm_aligned', {}).get('auc'))}), "
                f"delta_auc={_fmt_dbg(delta_align.get('auc'))}, "
                f"KS(raw/aligned->{ctx.hybrid_backbone_label})={_fmt_dbg(raw_dist.get('ks'))}/{_fmt_dbg(aligned_dist.get('ks'))}, "
                f"W1(raw/aligned->{ctx.hybrid_backbone_label})={_fmt_dbg(raw_dist.get('wasserstein'))}/{_fmt_dbg(aligned_dist.get('wasserstein'))}"
            )
        selected_slice = results.get("hybrid", {}).get("selected_slice_metrics", {})
        selected_backbone = selected_slice.get("backbone", selected_slice.get("rrf", {}))
        selected_llm = selected_slice.get("llm_raw", {})
        selected_llm_aligned = selected_slice.get("llm_aligned", {})
        selected_hybrid = selected_slice.get("hybrid_after_merge", {})
        print(
            "Selected-slice Metrics: "
            f"{ctx.hybrid_backbone_label}(AP={selected_backbone.get('ap', 0.0):.4f}, AUC={selected_backbone.get('auc', 0.0):.4f}, Acc={selected_backbone.get('accuracy', 0.0):.4f}) | "
            f"LLM(AP={selected_llm.get('ap', 0.0):.4f}, AUC={selected_llm.get('auc', 0.0):.4f}, Acc={selected_llm.get('accuracy', 0.0):.4f}) | "
            f"Hybrid(AP={selected_hybrid.get('ap', 0.0):.4f}, AUC={selected_hybrid.get('auc', 0.0):.4f}, Acc={selected_hybrid.get('accuracy', 0.0):.4f})"
        )
        if args.hybrid_score_alignment_mode != "off" and selected_llm_aligned:
            print(
                "Selected-slice Aligned LLM: "
                f"AP={selected_llm_aligned.get('ap', 0.0):.4f}, "
                f"AUC={selected_llm_aligned.get('auc', 0.0):.4f}, "
                f"Acc={selected_llm_aligned.get('accuracy', 0.0):.4f}"
            )
        llm_auc_delta = selected_llm.get("auc", 0.0) - selected_backbone.get("auc", 0.0)
        llm_ap_delta = selected_llm.get("ap", 0.0) - selected_backbone.get("ap", 0.0)
        print(
            "Selected Middle-band Slice: "
            f"LLM vs {ctx.hybrid_backbone_label} -> Delta AP={llm_ap_delta:.4f}, Delta AUC={llm_auc_delta:.4f}"
        )
    else:
        if dp_ctx["enabled"]:
            sample_indices, eval_samples = split_samples_for_rank(
                samples=samples,
                dp_size=dp_ctx["size"],
                dp_rank=dp_ctx["rank"],
            )
            print(
                f"DP shard assignment: rank {dp_ctx['rank']}/{dp_ctx['size']} "
                f"handles {len(eval_samples)}/{len(samples)} samples."
            )
        else:
            sample_indices = list(range(len(samples)))
            eval_samples = samples

        local_results = run_eval(eval_samples, capture_embeddings=True)
        if dp_ctx["enabled"]:
            dp_trial_idx = ctx.trial * len(ctx.eval_splits) + split_idx
            shard_path = write_trial_shard(
                sync_dir=dp_ctx["sync_dir"],
                run_id=dp_ctx["run_id"],
                trial_idx=dp_trial_idx,
                rank=dp_ctx["rank"],
                total_samples=len(samples),
                sample_indices=sample_indices,
                trial_results=local_results,
            )
            print(f"DP shard saved: {shard_path}")
            if dp_ctx["rank"] == 0:
                shard_paths = wait_for_trial_shards(
                    sync_dir=dp_ctx["sync_dir"],
                    run_id=dp_ctx["run_id"],
                    trial_idx=dp_trial_idx,
                    dp_size=dp_ctx["size"],
                    timeout_sec=dp_ctx["timeout_sec"],
                )
                results = merge_trial_shards(
                    shard_paths,
                    total_samples=len(samples),
                    dtgb_eval_batch_size=args.dtgb_eval_batch_size,
                )
                print(
                    f"DP merge complete for trial {ctx.trial + 1} [{eval_split}]: "
                    f"{len(shard_paths)} shards -> {len(samples)} samples"
                )
            else:
                results = None
        else:
            results = local_results

    if results is None:
        print(f"  -> Trial {ctx.trial + 1} [{eval_split}]: rank {dp_ctx['rank']} shard complete.")
        return

    if args.debug_prediction_log and (not dp_ctx["enabled"] or dp_ctx["rank"] == 0):
        written = write_prediction_debug_log(
            path=args.debug_prediction_log,
            detailed_results=results.get("detailed_results", []),
            trial_idx=ctx.trial + 1,
            eval_split=eval_split,
        )
        if written > 0:
            print(f"Prediction debug rows written: {written}")

    if ctx.trial_validation_sampled_calibration is not None:
        band_debug = _build_realized_discriminative_debug(
            samples,
            low_threshold=ctx.trial_validation_sampled_calibration["low_threshold"],
            high_threshold=ctx.trial_validation_sampled_calibration["high_threshold"],
            score_field=ctx.expert_score_field,
            score_label=ctx.expert_score_label,
        )
        results["rrf_validation_band_debug"] = band_debug
        if ctx.use_validation_sampled_threeway:
            results["validation_sampled_threeway_debug"] = band_debug
        if band_debug is not None:
            low_rate = band_debug["bands"]["low"]["positive_rate"]
            modest_rate = band_debug["bands"]["modest"]["positive_rate"]
            high_rate = band_debug["bands"]["high"]["positive_rate"]

            def _fmt_rate(value):
                return "n/a" if value is None else f"{float(value):.3f}"

            gap_value = band_debug.get("positive_rate_gap_high_minus_low")
            gap_text = "n/a" if gap_value is None else f"{float(gap_value):.3f}"
            monotonic_value = band_debug.get("monotonic_non_decreasing")
            monotonic_text = "n/a" if monotonic_value is None else str(bool(monotonic_value))
            print(
                f"  -> Trial {ctx.trial + 1} [{eval_split}] "
                f"Expert/{ctx.expert_score_label} validation-band test debug: "
                f"pos_rate(low/modest/high)="
                f"{_fmt_rate(low_rate)}/{_fmt_rate(modest_rate)}/{_fmt_rate(high_rate)}, "
                f"gap(high-low)={gap_text}, monotonic={monotonic_text}"
            )

    if include_expert_prediction:
        expert_debug = _build_expert_prediction_discriminative_debug(samples)
        results["expert_discrimination_debug"] = expert_debug
        if expert_debug is not None:
            ordered_categories = expert_debug.get("category_order", [])
            category_rates = [
                expert_debug.get("categories", {}).get(category, {}).get("positive_rate")
                for category in ordered_categories
            ]

            def _fmt_rate(value):
                return "n/a" if value is None else f"{float(value):.3f}"

            categories_text = "/".join(ordered_categories)
            rates_text = "/".join(_fmt_rate(value) for value in category_rates)
            gap_value = expert_debug.get("positive_rate_gap_true_minus_false")
            gap_text = "n/a" if gap_value is None else f"{float(gap_value):.3f}"
            monotonic_value = expert_debug.get("monotonic_non_decreasing")
            monotonic_text = "n/a" if monotonic_value is None else str(bool(monotonic_value))
            print(
                f"  -> Trial {ctx.trial + 1} [{eval_split}] Expert prior runtime debug: "
                f"pos_rate({categories_text})={rates_text}, "
                f"gap(true-false)={gap_text}, monotonic={monotonic_text}"
            )

    if include_key_signals:
        key_signal_debug = _build_key_signal_discriminative_debug(samples)
        results["key_signal_discrimination_debug"] = key_signal_debug
        if key_signal_debug is not None:
            signal_order_map = {
                "target_popularity": ("target_popularity", "target_popularity"),
                "past_interactions": ("past_interactions", "past_interactions"),
                "recency": ("interaction_recency", "interaction_recency"),
                "common_neighbor": ("common_neighbor", "common_neighbor"),
                "global_recency": ("global_recency", "global_recency"),
            }
            signal_order = [
                signal_order_map[field]
                for field in args.key_signal_fields
                if field in signal_order_map
            ]

            def _fmt_rate(value):
                return "n/a" if value is None else f"{float(value):.3f}"

            for debug_key, label in signal_order:
                signal_info = key_signal_debug["signals"].get(debug_key, {})
                categories = signal_info.get("categories", {})
                ordered_names = []
                if label == "interaction_recency" and "No prior interactions" in categories:
                    ordered_names.extend(["No prior interactions"])
                if label == "global_recency" and "No prior target interactions" in categories:
                    ordered_names.extend(["No prior target interactions"])
                ordered_names.extend(["Low", "Modest", "High"])
                ordered_names = [name for name in ordered_names if name in categories]
                ordered_rates = [
                    categories.get(name, {}).get("positive_rate") for name in ordered_names
                ]
                monotonic_value = signal_info.get("monotonic_non_decreasing_low_modest_high")
                monotonic_text = (
                    "n/a" if monotonic_value is None else str(bool(monotonic_value))
                )
                category_text = "/".join(
                    name.lower().replace(" ", "_") for name in ordered_names
                )
                rate_text = "/".join(_fmt_rate(value) for value in ordered_rates)
                print(
                    f"  -> Trial {ctx.trial + 1} [{eval_split}] Key-signal debug [{label}]: "
                    f"pos_rate({category_text})={rate_text}, "
                    f"monotonic={monotonic_text}"
                )

    ctx.all_trial_results[eval_split].append(build_trial_result_summary(results))
    maybe_print_token_usage_summary(
        results=results,
        trial_idx=ctx.trial + 1,
        eval_split=eval_split,
    )
    print(
        f"  -> Trial {ctx.trial + 1} [{eval_split}] Results: "
        f"AP={results['ap']:.4f}, AUC={results['auc']:.4f}, Acc={results['accuracy']:.4f}"
    )
