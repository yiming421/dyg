"""
RRF package: scoring backends, train-pool infrastructure, and RRF-specific analysis.
"""

__all__ = [
    "TrainPoolBuildConfig",
    "TrainPoolRRFConfig",
    "build_dtgb_eval_batches",
    "compute_rrf_selected_slice_proxy",
    "compute_rrf_scores_for_samples",
    "compute_train_pool_rrf_scores",
    "evaluate_budgeted_hybrid_rrf_llm",
    "group_indices_by_values",
    "group_sample_indices_by_query",
    "minmax_scale_scores",
    "select_rrf_middle_sample_indices_pointwise_threshold_band",
    "select_topk_middle_sample_indices_by_score",
    "select_topk_rrf_middle_sample_indices",
    "select_topk_rrf_middle_samples",
]
