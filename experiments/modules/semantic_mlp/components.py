from __future__ import annotations

from experiments.modules.heuristic_models import (
    HAS_CUDA,
    score_links_by_common_neighbors,
    score_links_by_global_recency,
    score_links_by_itemcf_cosine,
    score_links_by_past_interactions,
    score_links_by_popularity,
    score_links_by_recent_degree,
    score_links_by_recency,
    score_links_by_usercf_cosine,
    smooth_embeddings_by_time_window_torch,
)
from experiments.modules.prediction_metrics import compute_prediction_metrics
from experiments.modules.semantic_mlp.graph_components import *
from experiments.modules.semantic_mlp.graph_components import (
    _build_neighbor_sparse,
    _compute_mplp_exact_features_local_cpu,
    _elem2spm,
    _get_cpu_csr_from_sparse_adj,
    _spm2elem,
    _spmdiff,
    _spmoverlap,
)
from experiments.modules.semantic_mlp.heuristic_components import *
from experiments.modules.semantic_mlp.runtime_components import *
from experiments.modules.semantic_mlp.runtime_components import (
    _negative_precompute_debug_log,
    _profile_stage_elapsed,
    _profile_stage_start,
)
