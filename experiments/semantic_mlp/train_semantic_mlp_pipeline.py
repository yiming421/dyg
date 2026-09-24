#!/usr/bin/env python3
"""
Standalone training pipeline for semantic link scoring with a pair scorer.

This script:
1) Computes/loads base text embeddings
2) Applies semantic smoothing over temporal graph history
3) Trains a pair scorer on (src_emb, dst_emb) with negative sampling
4) Evaluates AP/AUC/MRR on transductive and inductive test splits
"""

import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

_EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_EXPERIMENTS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.modules.heuristic_models import (
    HAS_CUDA,
    precompute_entity_embeddings,
    smooth_embeddings_by_time_window_torch,
)
from experiments.modules.heuristic_semantic_models import (
    build_source_history_mean_initialized_embeddings,
)
from experiments.modules.llm_lp.cli import (
    build_semantic_mlp_arg_parser,
    validate_semantic_mlp_args,
)
from experiments.modules.llm_lp.experiment import build_prompt_entity_map
from utils.seed_runs import launch_seed_workers
from utils.DataLoader import get_link_prediction_data, get_idx_data_loader
from utils.utils import NegativeEdgeSampler, get_neighbor_sampler, set_random_seed

from experiments.modules.semantic_mlp.models import (
    CrossAttention,
    CrossAttentionLayer,
    FeedForwardCrossAttention,
    LearnableEntityEmbeddingTable,
    MultiHeadCrossAttentionByHand,
    RelativeTimeEncoder,
    SemanticCrossAttention,
    SemanticDyGFormerLiteScorer,
    SemanticMLP,
    SemanticNCNScorer,
    SemanticSeqFilterScorer,
    TemporalNeighborIndex,
    TemporalSelfAttentionPooling,
    build_lookup_tensor,
    build_lr_scheduler,
    make_activation_layer,
    make_subset,
    maybe_filter_train_edges,
    str2bool,
)

from experiments.modules.semantic_mlp.runtime import (
    HeuristicFeatureExtractor,
    HeuristicFusionHead,
    LearnableGCNEncoder,
    LearnableGINEncoder,
    MPLPExactFusionHead,
    RollingSmoothedEmbeddingProvider,
    build_binary_history_adj,
    build_two_hop_binary_adj,
    build_precomputed_train_negative_pool,
    build_precomputed_negative_queries,
    evaluate_split,
    fit_pairwise_ridge,
    train_one_epoch,
)
from experiments.modules.semantic_mlp.ridge import (
    FixedRandomProjection,
    SemanticRidgeScorer,
)
from experiments.modules.semantic_mlp.heuristic_components import (
    build_googlemap_city_zip_ids,
)
from experiments.modules.semantic_mlp.graph_components import (
    SEMANTIC_SOURCE_INIT_VERSION,
)


def _build_structural_seed_features(
    *,
    num_nodes: int,
    src_node_ids: np.ndarray,
    dst_node_ids: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    src = np.asarray(src_node_ids, dtype=np.int64)
    dst = np.asarray(dst_node_ids, dtype=np.int64)
    out_degree = np.bincount(src[(src >= 0) & (src < num_nodes)], minlength=num_nodes).astype(np.float32)
    in_degree = np.bincount(dst[(dst >= 0) & (dst < num_nodes)], minlength=num_nodes).astype(np.float32)
    total_degree = out_degree + in_degree
    features = np.stack(
        [
            (out_degree > 0).astype(np.float32),
            (in_degree > 0).astype(np.float32),
            np.log1p(out_degree),
            np.log1p(in_degree),
            np.log1p(total_degree),
        ],
        axis=1,
    ).astype(np.float32)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    features = (features - mean) / np.clip(std, a_min=1e-6, a_max=None)
    return torch.from_numpy(features).to(device)


def _resolve_relation_feature_path(dataset_name: str, configured_path: str = None) -> str:
    candidates = []
    if configured_path:
        candidates.append(configured_path)
    candidates.extend(
        [
            os.path.join("..", "DyLink_Datasets", dataset_name, "r_feat.npy"),
            os.path.join(_REPO_ROOT, "DyLink_Datasets", dataset_name, "r_feat.npy"),
        ]
    )
    for path in candidates:
        if path and os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "Temporal-relational GCN requires relation features. Checked: "
        + ", ".join(os.path.abspath(path) for path in candidates if path)
    )


def main():
    args = validate_semantic_mlp_args(
        build_semantic_mlp_arg_parser(str2bool).parse_args()
    )
    if launch_seed_workers(args, "experiments.semantic_mlp.train_semantic_mlp_pipeline",
                           "--checkpoint_path", output_is_file=True):
        return

    if args.scorer_type == 'ridge':
        if args.use_learnable_entity_embeddings and not args.freeze_learnable_entity_embeddings:
            raise ValueError(
                "scorer_type=ridge cannot train entity embeddings; use text embeddings or "
                "freeze the entity embedding table."
            )
        if args.use_structural_seed_features or args.semantic_project_dim > 0:
            raise ValueError(
                "scorer_type=ridge does not support trainable structural seeds or "
                "--semantic_project_dim. Use --ridge_projection_dim for its fixed projection."
            )
        if args.use_learnable_gcn:
            print(
                "Semantic ridge uses the existing parameter-free smoothing operator; "
                "disabling --use_learnable_gcn."
            )
            args.use_learnable_gcn = False
        args.semantic_aux_fusion_mode = 'late_concat'
        args.semantic_mlp_pair_feature_mode = 'hadamard'

    FIXED_CROSS_ATTN_UNDIRECTED_HISTORY = True
    FIXED_ATTN_MP_LAYERS = 1
    FIXED_ATTN_MP_RESIDUAL = True
    FIXED_TIME_ENCODER_RBF_GAMMA = 16.0

    set_random_seed(args.seed)
    if args.gpu < 0:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if args.gpu >= num_gpus:
            raise ValueError(f"--gpu={args.gpu} is invalid: only {num_gpus} CUDA device(s) available.")
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(device)
    else:
        print("CUDA is not available. Falling back to CPU.")
        device = torch.device('cpu')
    semantic_smoothing_enabled = args.use_semantic_smoothing
    rolling_enabled = args.rolling_smoothing and semantic_smoothing_enabled
    source_init_requested = str(args.semantic_smoothing_source_init).strip().lower()
    if not semantic_smoothing_enabled:
        source_init_effective = 'raw'
        source_init_temporal_mode = 'disabled'
    elif args.use_learnable_entity_embeddings:
        # Preserve the documented behavior for learned ID tables: their source
        # representations are learned directly rather than replaced by history.
        source_init_effective = 'raw'
        source_init_temporal_mode = 'raw'
    else:
        source_init_effective = source_init_requested
        if source_init_effective == 'raw':
            source_init_temporal_mode = 'raw'
        elif rolling_enabled:
            source_init_temporal_mode = 'rolling_history_before_batch'
        else:
            source_init_temporal_mode = 'static_history'
    if args.use_heuristic_features and args.use_gpu_heuristics and not HAS_CUDA:
        print("CUDA heuristic kernels are unavailable; RA heuristic will run on CPU.")

    print("=" * 80)
    print("Semantic Pair-Scorer Training Pipeline")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Negative strategy: {args.negative_strategy}")
    print(
        f"Optimizer: {args.optimizer} "
        f"(lr={args.lr}, weight_decay={args.weight_decay}, scheduler={args.scheduler})"
    )
    print(f"Scorer: {args.scorer_type}")
    if args.scorer_type == 'mlp':
        print(f"Semantic MLP pair feature mode: {args.semantic_mlp_pair_feature_mode}")
    if args.use_heuristic_features:
        print(
            f"Heuristic fusion: enabled "
            f"(features={','.join(args.heuristic_feature_names)}, "
            f"mode={args.semantic_aux_fusion_mode}, "
            f"recency_directed={args.heuristic_recency_directed}, "
            f"pop_decay={args.heuristic_popularity_decay}, "
            f"recent_window={args.heuristic_recent_degree_window}, "
            f"gpu_ra={args.use_gpu_heuristics and HAS_CUDA})"
        )
    if args.use_mplp_exact_features:
        print(
            "High-order structural fusion: exact MPLP-style overlap features enabled "
            f"(mplp_signature_dim={args.mplp_signature_dim}; reserved for non-exact variants)"
        )
    if args.scorer_type == 'cross_attention':
        print(
            f"Cross-attention config: heads={args.cross_attn_heads}, "
            f"layers={args.cross_attn_num_layers}, "
            f"num_neighbors={args.cross_attn_num_neighbors}, "
            f"use_pos={args.cross_attn_use_pos}, "
            f"add_time_to_history={args.cross_attn_add_time_to_history}, "
            f"use_raw_embeddings={args.cross_attn_use_raw_embeddings}, "
            f"hidden_dropout={args.cross_attn_hidden_dropout}, "
            f"attn_dropout={args.cross_attn_attn_dropout}, "
            f"emb_dropout={args.cross_attn_emb_dropout}, "
            f"undirected_history={FIXED_CROSS_ATTN_UNDIRECTED_HISTORY}, "
            f"time_encoder={args.time_encoder_type}, "
            f"mask_padding={args.time_encoder_mask_padding}"
        )
    elif args.scorer_type == 'dygformer_lite':
        print(
            f"DyGFormer-lite config: heads={args.dygformer_heads}, "
            f"layers={args.dygformer_num_layers}, "
            f"num_neighbors={args.dygformer_num_neighbors}, "
            f"add_time={args.dygformer_add_time}, "
            f"hidden_dim={args.hidden_dim}, dropout={args.dropout}, "
            f"undirected_history={FIXED_CROSS_ATTN_UNDIRECTED_HISTORY}, "
            f"time_encoder={args.time_encoder_type}, "
            f"mask_padding={args.time_encoder_mask_padding}"
        )
    elif args.scorer_type == 'ncn':
        print(
            f"NCN config: num_neighbors={args.ncn_num_neighbors}, "
            f"hidden_dim={args.hidden_dim}, layers={args.num_layers}, "
            f"dropout={args.dropout}"
        )
    elif args.scorer_type == 'seqfilter':
        print(
            f"SeqFilter config: num_neighbors={args.seqfilter_num_neighbors}, "
            f"tau={args.seqfilter_tau}, kernel={args.seqfilter_kernel_size}, "
            f"soft_mask={args.seqfilter_use_soft_mask}, hidden_dim={args.hidden_dim}, "
            f"layers={args.num_layers}, dropout={args.dropout}, "
            f"time_encoder={args.time_encoder_type}"
        )
    if args.use_learnable_gcn:
        if args.learnable_mp_type == 'gcn':
            print(
                f"Learnable MP: gcn "
                f"(layers={args.gcn_num_layers}, hidden={args.hidden_dim}, "
                f"linear_transform={args.gcn_use_linear_transform}, "
                f"dropout={args.dropout}; adjacency=tied_to_smoothing)"
            )
            if args.use_temporal_relational_gcn:
                print(
                    "Temporal-relational GCN branch: enabled "
                    f"(relation_features=768d, rank={args.temporal_relational_rank}, "
                    f"time_basis={args.temporal_relational_time_basis_dim}, "
                    "direction=out/in, cache=per_timestamp)"
                )
        elif args.learnable_mp_type == 'gin':
            if args.gin_nonparametric:
                print(
                    f"Learnable MP: gin_nonparametric "
                    f"(layers={args.gcn_num_layers}; "
                    f"operator={args.gin_nonparametric_norm.upper()}(x+sum_neighbors), "
                    f"affine=false, self_coefficient=1, outer_residual=false, "
                    f"adjacency=tied_to_smoothing)"
                )
            else:
                print(
                    f"Learnable MP: gin "
                    f"(layers={args.gcn_num_layers}, mlp_hidden={args.hidden_dim}, "
                    f"dropout={args.dropout}; aggregation=sum, adjacency=tied_to_smoothing)"
                )
        else:
            print(
                f"Learnable MP: attn_pool "
                f"(layers={FIXED_ATTN_MP_LAYERS}, heads={args.attn_mp_heads}, "
                f"neighbors={args.attn_mp_num_neighbors}, dropout={args.dropout}, "
                f"residual={FIXED_ATTN_MP_RESIDUAL}, time_encoder={args.time_encoder_type}, "
                f"mask_padding={args.time_encoder_mask_padding}; "
                f"history=tied_to_temporal_recent_sequence)"
            )
    print(f"Semantic smoothing: {semantic_smoothing_enabled}")
    print(f"Rolling smoothing: {rolling_enabled}")
    if args.smooth_endpoint_topk_recent > 0:
        print(
            "Semantic adjacency perception cap: "
            f"latest {args.smooth_endpoint_topk_recent} interactions per node "
            f"(mode={args.smooth_endpoint_topk_mode}, "
            f"inside time_window={args.smooth_time_window})"
        )
    if args.use_learnable_entity_embeddings:
        base_mode_desc = "learnable-from-scratch"
        if args.freeze_learnable_entity_embeddings:
            base_mode_desc = "frozen-random"
        print(
            f"Base entity features: {base_mode_desc} "
            f"(dim={args.learnable_entity_embedding_dim}, init={args.learnable_entity_embedding_init}, "
            f"freeze={args.freeze_learnable_entity_embeddings}, raw_node_id_aligned=true)"
        )
    else:
        print("Base entity features: text-derived embeddings")
    if args.use_structural_seed_features:
        print("Structural seed features: enabled (5 dims -> embedding dim -> graph propagation)")
    print(f"DataLoader workers: {args.dataloader_num_workers}")
    if args.debug_negative_precompute:
        print(
            f"Negative-precompute debug: enabled "
            f"(every {args.debug_negative_precompute_every} batches)"
        )
    print(f"Batch size (train/eval): {args.train_batch_size}/{args.eval_batch_size}")
    if args.profile_runtime:
        print(
            f"Runtime profiling: enabled "
            f"(skip {args.profile_skip_batches}, then profile {args.profile_batches} batches per split per epoch)"
        )
    if args.torch_profile_steps > 0:
        print(
            f"torch.profiler: enabled "
            f"(first epoch only, warmup={args.torch_profile_warmup_steps}, active={args.torch_profile_steps}, "
            f"dir={args.torch_profile_dir})"
        )
    if args.train_rand_ratio is not None:
        print(f"Training negative mix: random={args.train_rand_ratio:.2f}, historical={1.0 - args.train_rand_ratio:.2f}")
    if args.historical_neg_min_gap > 0.0:
        print(f"Historical negatives min gap: {args.historical_neg_min_gap}")
    if args.historical_neg_strict_gap:
        print("Historical negatives strict gap: enabled")

    wandb = None
    if args.scorer_type != 'ridge':
        try:
            import wandb
        except ImportError as e:
            raise ImportError("wandb is required. Install with `pip install wandb`.") from e
        wandb.init(
            project="semantic_mlp_pipeline",
            name=f"{args.dataset_name}_semantic_mlp",
            config=vars(args),
        )

    class DataArgs:
        use_feature = 'None'
        model_name = 'SemanticMLP'

    _, _, full_data, train_data_raw, val_data, test_data, _, new_node_test_data, _ = get_link_prediction_data(
        dataset_name=args.dataset_name,
        val_ratio=0.15,
        test_ratio=0.15,
        args=DataArgs(),
    )

    relation_features_t = None
    relation_feature_path = None
    if args.use_temporal_relational_gcn:
        relation_feature_path = _resolve_relation_feature_path(
            args.dataset_name,
            args.temporal_relational_edge_feature_path,
        )
        relation_features_np = np.load(relation_feature_path)
        if relation_features_np.ndim != 2:
            raise ValueError(
                f"Relation feature table must be 2D, got {relation_features_np.shape}."
            )
        max_relation_id = int(np.max(full_data.edge_ids))
        if max_relation_id >= relation_features_np.shape[0]:
            raise ValueError(
                "Relation feature table does not cover graph relation IDs: "
                f"max_id={max_relation_id}, rows={relation_features_np.shape[0]}."
            )
        relation_features_t = torch.from_numpy(
            np.asarray(relation_features_np, dtype=np.float32)
        ).to(device)
        print(
            f"Loaded relation features from {relation_feature_path}: "
            f"shape={tuple(relation_features_t.shape)}"
        )

    train_data, kept_train, total_train = maybe_filter_train_edges(
        train_data=train_data_raw,
        cutoff_time=args.train_edge_cutoff_time,
        cutoff_ratio=args.train_edge_cutoff_ratio,
    )

    train_holdout_metadata = {
        'count': 0,
        'min_time': None,
        'max_time': None,
        'first_edge': None,
        'last_edge': None,
    }
    if args.train_holdout_recent_edges > 0:
        holdout_count = int(args.train_holdout_recent_edges)
        if holdout_count >= len(train_data.src_node_ids):
            raise ValueError(
                "--train_holdout_recent_edges must leave at least one training edge; "
                f"requested={holdout_count}, available={len(train_data.src_node_ids)}"
            )
        heldout_slice = slice(len(train_data.src_node_ids) - holdout_count, None)
        heldout_src = train_data.src_node_ids[heldout_slice]
        heldout_dst = train_data.dst_node_ids[heldout_slice]
        heldout_times = train_data.node_interact_times[heldout_slice]
        train_holdout_metadata = {
            'count': holdout_count,
            'min_time': float(np.min(heldout_times)),
            'max_time': float(np.max(heldout_times)),
            'first_edge': [
                int(heldout_src[0]),
                int(heldout_dst[0]),
                float(heldout_times[0]),
            ],
            'last_edge': [
                int(heldout_src[-1]),
                int(heldout_dst[-1]),
                float(heldout_times[-1]),
            ],
        }
        keep_mask = np.ones(len(train_data.src_node_ids), dtype=bool)
        keep_mask[-holdout_count:] = False
        train_data = make_subset(train_data, keep_mask)
        kept_train = len(train_data.src_node_ids)
        print(
            "Held out recent train tail from GNN optimization: "
            f"count={holdout_count:,}, time=[{train_holdout_metadata['min_time']:.0f}, "
            f"{train_holdout_metadata['max_time']:.0f}], "
            f"remaining={kept_train:,}"
        )

    print(f"Train edges: {kept_train:,} / {total_train:,} (after cutoff)")
    print(f"Val edges: {len(val_data.src_node_ids):,}")
    print(f"Test edges (transductive): {len(test_data.src_node_ids):,}")
    print(f"Test edges (inductive): {len(new_node_test_data.src_node_ids):,}")

    max_node_id = int(max(full_data.src_node_ids.max(), full_data.dst_node_ids.max()))

    import pandas as pd

    entity_text_path = (
        args.entity_text_path
        if getattr(args, "entity_text_path", None)
        else f'../DyLink_Datasets/{args.dataset_name}/entity_text.csv'
    )
    entity_embedding_table = None
    transformed_count = 0
    if args.use_learnable_entity_embeddings:
        if args.embedding_cache:
            print("Learnable entity embeddings enabled; ignoring --embedding_cache.")
        if not os.path.exists(entity_text_path):
            entity_text_path = None
        entity_ids = list(range(max_node_id + 1))
        entity_embedding_table = LearnableEntityEmbeddingTable(
            num_entities=len(entity_ids),
            embedding_dim=args.learnable_entity_embedding_dim,
            init_mode=args.learnable_entity_embedding_init,
            freeze=args.freeze_learnable_entity_embeddings,
        ).to(device)
        base_emb_t = entity_embedding_table()
        embedding_mode_desc = (
            "frozen orthogonal-noise"
            if args.freeze_learnable_entity_embeddings and args.learnable_entity_embedding_init == 'orthogonal'
            else (
                "frozen random"
                if args.freeze_learnable_entity_embeddings
                else "learnable"
            )
        )
        print(
            f"Initialized {embedding_mode_desc} entity embedding table with {len(entity_ids):,} rows "
            f"(covers node ids 0..{max_node_id}, dim={args.learnable_entity_embedding_dim}, "
            f"init={args.learnable_entity_embedding_init})"
        )
    else:
        if not os.path.exists(entity_text_path):
            raise FileNotFoundError(f"Missing entity text file: {entity_text_path}")

        entity_text_df = pd.read_csv(entity_text_path)
        raw_entity_texts = dict(zip(entity_text_df['i'], entity_text_df['text']))
        entity_texts = build_prompt_entity_map(
            args.dataset_name,
            raw_entity_texts,
            entity_name_mode=args.embedding_entity_name_mode,
        )
        entity_ids = sorted(entity_texts.keys())
        transformed_count = sum(
            1
            for entity_id in entity_ids
            if entity_texts.get(entity_id) != raw_entity_texts.get(entity_id)
        )

        print(
            f"Loaded {len(entity_ids):,} entity texts "
            f"from {entity_text_path} "
            f"(embedding_entity_name_mode={args.embedding_entity_name_mode}, "
            f"transformed={transformed_count:,})"
        )

        if args.embedding_cache and os.path.exists(args.embedding_cache):
            print(f"Loading base embeddings from {args.embedding_cache}")
            if str(args.embedding_entity_name_mode).strip().lower() != 'raw':
                print(
                    "[WARN] Reusing --embedding_cache with non-raw entity text mode. "
                    "Make sure this cache was built with the same --embedding_entity_name_mode."
                )
            base_embeddings = np.load(args.embedding_cache)
        else:
            base_embeddings, _ = precompute_entity_embeddings(
                entity_texts,
                model_name=args.embedding_model,
                device='cuda' if torch.cuda.is_available() else 'cpu',
            )
            if args.embedding_cache:
                cache_dir = os.path.dirname(args.embedding_cache)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                np.save(args.embedding_cache, base_embeddings)
                print(f"Saved base embeddings to {args.embedding_cache}")

        base_emb_t = torch.from_numpy(base_embeddings).float().to(device)
        base_emb_t = base_emb_t / torch.norm(base_emb_t, dim=1, keepdim=True).clamp(min=1e-12)

    if args.strict_no_leakage:
        test_start_time = float(test_data.node_interact_times[0])
        hist_mask = full_data.node_interact_times < test_start_time
        smooth_src = full_data.src_node_ids[hist_mask]
        smooth_dst = full_data.dst_node_ids[hist_mask]
        smooth_times = full_data.node_interact_times[hist_mask]
        print(f"Smoothing graph (strict): {len(smooth_src):,} edges (train+val)")
    else:
        smooth_src = full_data.src_node_ids
        smooth_dst = full_data.dst_node_ids
        smooth_times = full_data.node_interact_times
        print(f"Smoothing graph (full): {len(smooth_src):,} edges")

    if args.smooth_cutoff_time is not None:
        smooth_mask = smooth_times < float(args.smooth_cutoff_time)
        smooth_src = smooth_src[smooth_mask]
        smooth_dst = smooth_dst[smooth_mask]
        smooth_times = smooth_times[smooth_mask]
        print(f"Applied smoothing cutoff @ t<{args.smooth_cutoff_time}: {len(smooth_src):,} edges")

    if len(smooth_src) == 0:
        raise ValueError("No edges left for smoothing. Relax smoothing cutoff/time-window settings.")

    # Rolling temporal state reads the complete authoritative stream and
    # applies a strict time cutoff inside each batch. This is leakage-safe and
    # prevents inductive-only evaluation from omitting interleaved
    # transductive interactions. Keep the optional explicit smoothing cutoff.
    rolling_src = np.asarray(full_data.src_node_ids, dtype=np.int64)
    rolling_dst = np.asarray(full_data.dst_node_ids, dtype=np.int64)
    rolling_times = np.asarray(full_data.node_interact_times, dtype=np.float64)
    rolling_edge_ids = np.asarray(full_data.edge_ids, dtype=np.int64)
    if args.smooth_cutoff_time is not None:
        rolling_mask = rolling_times < float(args.smooth_cutoff_time)
        rolling_src = rolling_src[rolling_mask]
        rolling_dst = rolling_dst[rolling_mask]
        rolling_times = rolling_times[rolling_mask]
        rolling_edge_ids = rolling_edge_ids[rolling_mask]

    smoothing_base_emb_t = base_emb_t
    if semantic_smoothing_enabled and source_init_requested != source_init_effective:
        print(
            "Semantic source init requested "
            f"{source_init_requested}, effective {source_init_effective} "
            "(--use_learnable_entity_embeddings=true)."
        )
    elif source_init_effective == 'history_mean' and rolling_enabled:
        print(
            "Semantic source init: causal rolling history_mean "
            "(events strictly before each batch)."
        )
    elif source_init_effective == 'history_mean':
        lookup_for_init = build_lookup_tensor(entity_ids, max_node_id, torch.device('cpu'))
        smoothing_base_emb_t, replaced_nodes = build_source_history_mean_initialized_embeddings(
            base_embeddings=base_emb_t,
            node_id_lookup=lookup_for_init,
            src_node_ids=smooth_src,
            dst_node_ids=smooth_dst,
        )
        print(
            "Semantic smoothing init updated "
            f"{replaced_nodes:,} source/user embeddings from static history."
        )

    raw_semantic_input_dim = int(base_emb_t.shape[1])
    ridge_projection = None
    if args.scorer_type == 'ridge' and args.ridge_projection_dim > raw_semantic_input_dim:
        raise ValueError(
            f"--ridge_projection_dim ({args.ridge_projection_dim}) must be <= "
            f"the semantic input width ({raw_semantic_input_dim})."
        )
    if args.scorer_type == 'ridge' and 0 < args.ridge_projection_dim < raw_semantic_input_dim:
        ridge_projection = FixedRandomProjection(
            input_dim=raw_semantic_input_dim,
            output_dim=args.ridge_projection_dim,
            seed=args.seed,
        ).to(device)
        with torch.inference_mode():
            projected_base = torch.nn.functional.normalize(
                ridge_projection(base_emb_t),
                dim=1,
            )
            if smoothing_base_emb_t is base_emb_t:
                projected_smoothing_base = projected_base
            else:
                projected_smoothing_base = torch.nn.functional.normalize(
                    ridge_projection(smoothing_base_emb_t),
                    dim=1,
                )
        base_emb_t = projected_base
        smoothing_base_emb_t = projected_smoothing_base
        print(
            "Semantic ridge fixed projection: "
            f"{raw_semantic_input_dim} -> {args.ridge_projection_dim} (seed={args.seed})."
        )
    elif args.scorer_type == 'ridge':
        args.ridge_projection_dim = 0
        print(f"Semantic ridge fixed projection disabled (width={raw_semantic_input_dim}).")

    mp_replaces_smoothing = args.use_learnable_gcn
    graph_mp_uses_smoothing_adj = args.use_learnable_gcn and args.learnable_mp_type in ('gcn', 'gin')

    smoothed_embeddings = None
    static_mp_adj = None
    static_smoothing_adj = None
    if not semantic_smoothing_enabled:
        if args.smoothed_embedding_cache:
            print("Semantic smoothing disabled; ignoring --smoothed_embedding_cache.")
        if source_init_requested != 'raw':
            print("Semantic smoothing disabled; ignoring --semantic_smoothing_source_init.")
        smoothed_embeddings = base_emb_t
    elif rolling_enabled:
        if args.smoothed_embedding_cache:
            if args.use_learnable_entity_embeddings:
                print(
                    "Learnable entity embeddings use dynamic rolling smoothing; "
                    "ignoring --smoothed_embedding_cache."
                )
            else:
                print("Rolling smoothing enabled; ignoring --smoothed_embedding_cache (static cache only).")
    else:
        if mp_replaces_smoothing:
            if args.smoothed_embedding_cache and os.path.exists(args.smoothed_embedding_cache):
                if graph_mp_uses_smoothing_adj:
                    print(
                        "Learnable graph MP replaces heuristic smoothing; "
                        "ignoring --smoothed_embedding_cache and building only graph operator."
                    )
                else:
                    print("Learnable attn-pool replaces heuristic smoothing; ignoring --smoothed_embedding_cache.")
            if graph_mp_uses_smoothing_adj:
                # Build the exact smoothing operator from the same smoothing heuristic path.
                if args.learnable_mp_type == 'gcn':
                    _, static_mp_adj = smooth_embeddings_by_time_window_torch(
                        embeddings=smoothing_base_emb_t,
                        src_node_ids=smooth_src,
                        dst_node_ids=smooth_dst,
                        node_interact_times=smooth_times,
                        time_window=args.smooth_time_window,
                        num_steps=1,
                        symmetric_norm=True,
                        decay_gamma=args.smooth_decay_gamma,
                        undirected=args.smooth_undirected,
                        log_dampen=args.smooth_log_dampen,
                        supernode_strength=args.smooth_supernode_strength,
                        endpoint_topk_recent=args.smooth_endpoint_topk_recent,
                        endpoint_topk_mode=args.smooth_endpoint_topk_mode,
                        return_norm_adj=True,
                    )
                else:
                    _, static_mp_adj = smooth_embeddings_by_time_window_torch(
                        embeddings=smoothing_base_emb_t,
                        src_node_ids=smooth_src,
                        dst_node_ids=smooth_dst,
                        node_interact_times=smooth_times,
                        time_window=args.smooth_time_window,
                        num_steps=1,
                        symmetric_norm=True,
                        decay_gamma=args.smooth_decay_gamma,
                        undirected=args.smooth_undirected,
                        log_dampen=args.smooth_log_dampen,
                        supernode_strength=args.smooth_supernode_strength,
                        endpoint_topk_recent=args.smooth_endpoint_topk_recent,
                        endpoint_topk_mode=args.smooth_endpoint_topk_mode,
                        return_sum_adj=True,
                    )
        else:
            if args.use_learnable_entity_embeddings or args.use_structural_seed_features:
                if args.smoothed_embedding_cache:
                    print(
                        "Dynamic base embeddings use static-smoothing application; "
                        "ignoring --smoothed_embedding_cache."
                    )
                _, static_smoothing_adj = smooth_embeddings_by_time_window_torch(
                    embeddings=smoothing_base_emb_t,
                    src_node_ids=smooth_src,
                    dst_node_ids=smooth_dst,
                    node_interact_times=smooth_times,
                    time_window=args.smooth_time_window,
                    num_steps=1,
                    symmetric_norm=True,
                    decay_gamma=args.smooth_decay_gamma,
                    undirected=args.smooth_undirected,
                    log_dampen=args.smooth_log_dampen,
                    supernode_strength=args.smooth_supernode_strength,
                    endpoint_topk_recent=args.smooth_endpoint_topk_recent,
                    endpoint_topk_mode=args.smooth_endpoint_topk_mode,
                    return_norm_adj=True,
                )
            elif args.smoothed_embedding_cache and os.path.exists(args.smoothed_embedding_cache):
                print(f"Loading smoothed embeddings from {args.smoothed_embedding_cache}")
                smoothed_np = np.load(args.smoothed_embedding_cache)
                smoothed_embeddings = torch.from_numpy(smoothed_np).float().to(device)
            else:
                print("Computing smoothed embeddings...")
                smoothed_embeddings = smooth_embeddings_by_time_window_torch(
                    embeddings=smoothing_base_emb_t,
                    src_node_ids=smooth_src,
                    dst_node_ids=smooth_dst,
                    node_interact_times=smooth_times,
                    time_window=args.smooth_time_window,
                    num_steps=args.smooth_steps,
                    symmetric_norm=True,
                    decay_gamma=args.smooth_decay_gamma,
                    undirected=args.smooth_undirected,
                    log_dampen=args.smooth_log_dampen,
                    supernode_strength=args.smooth_supernode_strength,
                    endpoint_topk_recent=args.smooth_endpoint_topk_recent,
                    endpoint_topk_mode=args.smooth_endpoint_topk_mode,
                    return_norm_adj=False,
                )
                smoothed_embeddings = smoothed_embeddings.float()
                smoothed_embeddings = smoothed_embeddings / torch.norm(smoothed_embeddings, dim=1, keepdim=True).clamp(min=1e-12)

                if args.smoothed_embedding_cache:
                    cache_dir = os.path.dirname(args.smoothed_embedding_cache)
                    if cache_dir:
                        os.makedirs(cache_dir, exist_ok=True)
                    np.save(args.smoothed_embedding_cache, smoothed_embeddings.detach().cpu().numpy())
                    print(f"Saved smoothed embeddings to {args.smoothed_embedding_cache}")
    lookup = build_lookup_tensor(entity_ids, max_node_id, device)
    cross_attn_neighbor_index = None
    if args.scorer_type in {'cross_attention', 'dygformer_lite'}:
        index_label = "cross-attention" if args.scorer_type == 'cross_attention' else "DyGFormer-lite"
        print(f"Building temporal neighbor index for {index_label} scorer...")
        cross_attn_neighbor_index = TemporalNeighborIndex(
            src_node_ids=smooth_src,
            dst_node_ids=smooth_dst,
            node_interact_times=smooth_times,
            max_node_id=max_node_id,
            undirected=FIXED_CROSS_ATTN_UNDIRECTED_HISTORY,
        )
    ncn_neighbor_index = None
    if args.scorer_type == 'ncn':
        # Keep NCN on its own binary temporal history graph rather than reusing any
        # smoothing operator state. This keeps the overlap structure explicit and
        # avoids supernode/weighted-adjacency semantics leaking into the scorer.
        print("Building temporal neighbor index for NCN scorer...")
        ncn_neighbor_index = TemporalNeighborIndex(
            src_node_ids=smooth_src,
            dst_node_ids=smooth_dst,
            node_interact_times=smooth_times,
            max_node_id=max_node_id,
            undirected=True,
        )
    seqfilter_neighbor_index = None
    if args.scorer_type == 'seqfilter':
        print("Building temporal neighbor index for SeqFilter scorer...")
        seqfilter_neighbor_index = TemporalNeighborIndex(
            src_node_ids=smooth_src,
            dst_node_ids=smooth_dst,
            node_interact_times=smooth_times,
            max_node_id=max_node_id,
            undirected=args.smooth_undirected,
        )
    scorer_neighbor_index = (
        cross_attn_neighbor_index
        if args.scorer_type in {'cross_attention', 'dygformer_lite'}
        else (ncn_neighbor_index if args.scorer_type == 'ncn' else seqfilter_neighbor_index)
    )
    mp_neighbor_index = None
    if args.use_learnable_gcn and args.learnable_mp_type == 'attn_pool':
        print("Building temporal neighbor index for attention-pooling message passing...")
        mp_neighbor_index = TemporalNeighborIndex(
            src_node_ids=smooth_src,
            dst_node_ids=smooth_dst,
            node_interact_times=smooth_times,
            max_node_id=max_node_id,
            undirected=args.smooth_undirected,
        )
    node_city_ids = None
    node_zip_ids = None
    location_feature_names = {"city_preference", "zip_preference"} & set(
        args.heuristic_feature_names
    )
    if args.use_heuristic_features and location_feature_names:
        if str(args.dataset_name) != "Googlemap_CT":
            raise ValueError(
                "city_preference/zip_preference currently require dataset_name=Googlemap_CT."
            )
        if not os.path.exists(entity_text_path):
            raise FileNotFoundError(
                "Googlemap city/ZIP preference features require entity_text.csv."
            )
        if "entity_text_df" not in locals():
            entity_text_df = pd.read_csv(entity_text_path)
        node_city_ids, node_zip_ids, city_to_id, zip_to_id = build_googlemap_city_zip_ids(
            entity_text_df=entity_text_df,
            max_node_id=max_node_id,
        )
        print(
            "Loaded Googlemap preference metadata: "
            f"city_nodes={int(np.sum(node_city_ids >= 0)):,} ({len(city_to_id):,} cities), "
            f"zip_nodes={int(np.sum(node_zip_ids >= 0)):,} ({len(zip_to_id):,} ZIPs)"
        )

    heuristic_extractor = None
    if args.use_heuristic_features:
        print("Building heuristic feature extractor...")
        # Every heuristic kernel applies interaction_time < prediction_time.
        # Supplying full_data is therefore causal and lets test-period history
        # evolve instead of freezing all features at the test boundary.
        heuristic_graph_data = SimpleNamespace(
            src_node_ids=np.asarray(full_data.src_node_ids, dtype=np.int64),
            dst_node_ids=np.asarray(full_data.dst_node_ids, dtype=np.int64),
            edge_ids=np.arange(len(full_data.src_node_ids), dtype=np.int64),
            node_interact_times=np.asarray(full_data.node_interact_times, dtype=np.float64),
        )
        heuristic_neighbor_sampler = get_neighbor_sampler(
            data=heuristic_graph_data,
            sample_neighbor_strategy='recent',
            seed=args.seed,
        )
        heuristic_extractor = HeuristicFeatureExtractor(
            neighbor_sampler=heuristic_neighbor_sampler,
            directed_src_node_ids=np.asarray(full_data.src_node_ids, dtype=np.int64),
            directed_dst_node_ids=np.asarray(full_data.dst_node_ids, dtype=np.int64),
            directed_node_interact_times=np.asarray(
                full_data.node_interact_times, dtype=np.float64
            ),
            use_gpu_heuristics=args.use_gpu_heuristics,
            popularity_decay=args.heuristic_popularity_decay,
            recent_degree_window=args.heuristic_recent_degree_window,
            score_batch_size=args.heuristic_score_batch_size,
            recency_directed=args.heuristic_recency_directed,
            feature_names=args.heuristic_feature_names,
            node_city_ids=node_city_ids,
            node_zip_ids=node_zip_ids,
        )

    train_loader = get_idx_data_loader(
        list(range(len(train_data.src_node_ids))),
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )
    train_eval_loader = None
    if args.report_train_metrics:
        train_eval_loader = get_idx_data_loader(
            list(range(len(train_data.src_node_ids))),
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.dataloader_num_workers,
        )
    val_loader = get_idx_data_loader(
        list(range(len(val_data.src_node_ids))),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )
    test_loader = get_idx_data_loader(
        list(range(len(test_data.src_node_ids))),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )
    new_node_test_loader = get_idx_data_loader(
        list(range(len(new_node_test_data.src_node_ids))),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )

    train_neg_sampler = NegativeEdgeSampler(
        src_node_ids=train_data.src_node_ids,
        dst_node_ids=train_data.dst_node_ids,
        interact_times=train_data.node_interact_times,
        last_observed_time=float(train_data.node_interact_times[-1]),
        negative_sample_strategy=args.negative_strategy,
        seed=0,
        historical_min_gap=args.historical_neg_min_gap,
        historical_sample_per_src=True,
    )
    train_eval_neg_sampler = None
    if args.report_train_metrics:
        train_eval_neg_sampler = NegativeEdgeSampler(
            src_node_ids=train_data.src_node_ids,
            dst_node_ids=train_data.dst_node_ids,
            interact_times=train_data.node_interact_times,
            last_observed_time=float(train_data.node_interact_times[-1]),
            negative_sample_strategy=args.negative_strategy,
            seed=4,
            historical_min_gap=args.historical_neg_min_gap,
            historical_sample_per_src=True,
        )
    train_neg_sampler_random = None
    train_neg_sampler_historical = None
    if args.train_rand_ratio is not None:
        train_neg_sampler_random = NegativeEdgeSampler(
            src_node_ids=train_data.src_node_ids,
            dst_node_ids=train_data.dst_node_ids,
            interact_times=train_data.node_interact_times,
            last_observed_time=float(train_data.node_interact_times[-1]),
            negative_sample_strategy='random',
            seed=0,
            historical_min_gap=args.historical_neg_min_gap,
            historical_sample_per_src=True,
        )
        train_neg_sampler_historical = NegativeEdgeSampler(
            src_node_ids=train_data.src_node_ids,
            dst_node_ids=train_data.dst_node_ids,
            interact_times=train_data.node_interact_times,
            last_observed_time=float(train_data.node_interact_times[-1]),
            negative_sample_strategy='historical',
            seed=11,
            historical_min_gap=args.historical_neg_min_gap,
            historical_sample_per_src=True,
        )
    val_neg_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids,
        dst_node_ids=full_data.dst_node_ids,
        interact_times=full_data.node_interact_times,
        last_observed_time=float(val_data.node_interact_times[-1]),
        negative_sample_strategy=args.negative_strategy,
        seed=1,
        historical_min_gap=args.historical_neg_min_gap,
        historical_sample_per_src=True,
    )
    test_neg_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids,
        dst_node_ids=full_data.dst_node_ids,
        interact_times=full_data.node_interact_times,
        last_observed_time=float(val_data.node_interact_times[-1]),
        negative_sample_strategy=args.negative_strategy,
        seed=2,
        historical_min_gap=args.historical_neg_min_gap,
        historical_sample_per_src=True,
    )
    new_node_test_neg_sampler = NegativeEdgeSampler(
        src_node_ids=new_node_test_data.src_node_ids,
        dst_node_ids=new_node_test_data.dst_node_ids,
        interact_times=new_node_test_data.node_interact_times,
        last_observed_time=float(val_data.node_interact_times[-1]),
        negative_sample_strategy=args.negative_strategy,
        seed=3,
        historical_min_gap=args.historical_neg_min_gap,
        historical_sample_per_src=True,
    )

    train_pos_raw_heuristic_features = None
    val_pos_raw_heuristic_features = None
    val_neg_raw_heuristic_features = None
    test_pos_raw_heuristic_features = None
    test_neg_raw_heuristic_features = None
    new_node_pos_raw_heuristic_features = None
    new_node_neg_raw_heuristic_features = None
    if heuristic_extractor is not None:
        print("Precomputing cached heuristic features for fixed splits...")
        train_pos_raw_heuristic_features = heuristic_extractor.precompute_raw_features(
            sources=train_data.src_node_ids,
            targets=train_data.dst_node_ids,
            prediction_times=train_data.node_interact_times,
            desc="Heuristics: train positives",
        )
        val_pos_raw_heuristic_features = heuristic_extractor.precompute_raw_features(
            sources=val_data.src_node_ids,
            targets=val_data.dst_node_ids,
            prediction_times=val_data.node_interact_times,
            desc="Heuristics: val positives",
        )
        val_neg_src, val_neg_dst, val_neg_times = build_precomputed_negative_queries(
            data_loader=val_loader,
            data_source=val_data,
            neg_sampler=val_neg_sampler,
            num_negatives=args.val_num_negatives,
            desc="Sample val negatives",
            debug=args.debug_negative_precompute,
            debug_every=args.debug_negative_precompute_every,
        )
        val_neg_raw_heuristic_features = heuristic_extractor.precompute_raw_features(
            sources=val_neg_src,
            targets=val_neg_dst,
            prediction_times=val_neg_times,
            desc="Heuristics: val negatives",
        )
        test_pos_raw_heuristic_features = heuristic_extractor.precompute_raw_features(
            sources=test_data.src_node_ids,
            targets=test_data.dst_node_ids,
            prediction_times=test_data.node_interact_times,
            desc="Heuristics: test positives",
        )
        test_neg_src, test_neg_dst, test_neg_times = build_precomputed_negative_queries(
            data_loader=test_loader,
            data_source=test_data,
            neg_sampler=test_neg_sampler,
            num_negatives=args.eval_num_negatives,
            desc="Sample test negatives",
            debug=args.debug_negative_precompute,
            debug_every=args.debug_negative_precompute_every,
        )
        test_neg_raw_heuristic_features = heuristic_extractor.precompute_raw_features(
            sources=test_neg_src,
            targets=test_neg_dst,
            prediction_times=test_neg_times,
            desc="Heuristics: test negatives",
        )
        new_node_pos_raw_heuristic_features = heuristic_extractor.precompute_raw_features(
            sources=new_node_test_data.src_node_ids,
            targets=new_node_test_data.dst_node_ids,
            prediction_times=new_node_test_data.node_interact_times,
            desc="Heuristics: inductive positives",
        )
        new_node_neg_src, new_node_neg_dst, new_node_neg_times = build_precomputed_negative_queries(
            data_loader=new_node_test_loader,
            data_source=new_node_test_data,
            neg_sampler=new_node_test_neg_sampler,
            num_negatives=args.eval_num_negatives,
            desc="Sample inductive negatives",
            debug=args.debug_negative_precompute,
            debug_every=args.debug_negative_precompute_every,
        )
        new_node_neg_raw_heuristic_features = heuristic_extractor.precompute_raw_features(
            sources=new_node_neg_src,
            targets=new_node_neg_dst,
            prediction_times=new_node_neg_times,
            desc="Heuristics: inductive negatives",
        )

    train_rolling_provider = None
    val_rolling_provider = None
    test_rolling_provider = None
    new_node_test_rolling_provider = None
    static_ncn_adj = None
    static_mplp_exact_adj2 = None

    needs_rolling_provider = rolling_enabled and (
        not mp_replaces_smoothing
        or graph_mp_uses_smoothing_adj
        or source_init_effective == 'history_mean'
    )
    if needs_rolling_provider:
        def _make_provider(split_start_time: float):
            return RollingSmoothedEmbeddingProvider(
                base_embeddings=smoothing_base_emb_t,
                lookup=lookup,
                init_src_node_ids=rolling_src,
                init_dst_node_ids=rolling_dst,
                init_node_interact_times=rolling_times,
                smooth_time_window=args.smooth_time_window,
                smooth_steps=args.smooth_steps,
                smooth_decay_gamma=args.smooth_decay_gamma,
                smooth_undirected=args.smooth_undirected,
                smooth_log_dampen=args.smooth_log_dampen,
                smooth_supernode_strength=args.smooth_supernode_strength,
                smooth_endpoint_topk_recent=args.smooth_endpoint_topk_recent,
                smooth_endpoint_topk_mode=args.smooth_endpoint_topk_mode,
                export_norm_adj=(args.use_learnable_gcn and args.learnable_mp_type == 'gcn'),
                export_sum_adj=(args.use_learnable_gcn and args.learnable_mp_type == 'gin'),
                export_binary_adj=(args.scorer_type == 'ncn'),
                export_binary_two_hop_adj=args.use_mplp_exact_features,
                source_init=source_init_effective,
                apply_smoothing=(not mp_replaces_smoothing or graph_mp_uses_smoothing_adj),
                history_is_complete=True,
                materialize_smoothed_embeddings=not args.use_learnable_gcn,
                init_edge_ids=rolling_edge_ids,
                export_temporal_relational_context=args.use_temporal_relational_gcn,
                temporal_relational_num_relations=(
                    int(relation_features_t.size(0))
                    if relation_features_t is not None
                    else 0
                ),
                temporal_relational_time_basis_dim=(
                    args.temporal_relational_time_basis_dim
                ),
            )

        train_rolling_provider = _make_provider(float(np.min(train_data.node_interact_times)))
        val_rolling_provider = _make_provider(float(np.min(val_data.node_interact_times)))
        test_rolling_provider = _make_provider(float(np.min(test_data.node_interact_times)))
        new_node_test_rolling_provider = _make_provider(float(np.min(new_node_test_data.node_interact_times)))
    if not semantic_smoothing_enabled:
        active_embeddings = base_emb_t
    elif args.use_learnable_entity_embeddings:
        active_embeddings = base_emb_t
    elif args.use_structural_seed_features or mp_replaces_smoothing:
        # Static paths materialize dynamic features from the initialized table;
        # rolling paths let the provider initialize each materialized batch.
        active_embeddings = base_emb_t if rolling_enabled else smoothing_base_emb_t
    else:
        active_embeddings = base_emb_t if rolling_enabled else smoothed_embeddings
    if (args.scorer_type == 'ncn' or args.use_mplp_exact_features) and not rolling_enabled:
        print("Building static binary history graph...")
        static_ncn_adj = build_binary_history_adj(
            src_node_ids=smooth_src,
            dst_node_ids=smooth_dst,
            lookup=lookup,
            num_rows=int(active_embeddings.shape[0]),
            undirected=True,
        )
        if args.use_mplp_exact_features:
            print("Building static exact MPLP-style 2-hop history graph...")
            static_mplp_exact_adj2 = build_two_hop_binary_adj(static_ncn_adj)
    raw_input_dim = int(active_embeddings.shape[1])
    input_dim = raw_input_dim
    structural_seed_features = None
    structural_seed_projector = None
    if args.use_structural_seed_features:
        structural_seed_features = _build_structural_seed_features(
            num_nodes=int(active_embeddings.shape[0]),
            src_node_ids=train_data.src_node_ids,
            dst_node_ids=train_data.dst_node_ids,
            device=device,
        )
        structural_seed_projector = nn.Linear(
            int(structural_seed_features.shape[1]),
            raw_input_dim,
            bias=False,
        ).to(device)
        print(
            "Structural seed projector initialized "
            f"({int(structural_seed_features.shape[1])} -> {raw_input_dim}); "
            "projected train-history seeds are added before smoothing/message passing."
        )
    semantic_projector = None
    if args.semantic_project_dim > 0:
        if int(args.semantic_project_dim) > raw_input_dim:
            raise ValueError(
                f"--semantic_project_dim ({args.semantic_project_dim}) must be <= raw input dim ({raw_input_dim})."
            )
        input_dim = int(args.semantic_project_dim)
        semantic_projector = nn.Linear(raw_input_dim, input_dim, bias=False).to(device)
        nn.init.orthogonal_(semantic_projector.weight)
        print(
            "Semantic pre-projector initialized "
            f"({raw_input_dim} -> {input_dim}); smoothing/message passing and scoring use the projected width."
        )
    semantic_mlp_auxiliary_dim = 0
    if args.scorer_type in {'mlp', 'ridge'} and args.semantic_aux_fusion_mode == 'late_concat':
        if args.use_heuristic_features:
            semantic_mlp_auxiliary_dim += len(args.heuristic_feature_names)
        if args.use_mplp_exact_features:
            semantic_mlp_auxiliary_dim += MPLPExactFusionHead.feature_dim
        print(
            "Semantic pair-scorer auxiliary input concat: "
            f"enabled (aux_dim={semantic_mlp_auxiliary_dim})"
        )
    if args.scorer_type == 'mlp':
        model = SemanticMLP(
            input_dim=input_dim,
            auxiliary_dim=semantic_mlp_auxiliary_dim,
            pair_feature_mode=args.semantic_mlp_pair_feature_mode,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            activation=args.activation,
            use_layernorm=args.use_layernorm,
        ).to(device)
    elif args.scorer_type == 'ridge':
        model = SemanticRidgeScorer(
            input_dim=input_dim,
            auxiliary_dim=semantic_mlp_auxiliary_dim,
            semantic_feature_mode=args.ridge_semantic_feature_mode,
        ).to(device)
    elif args.scorer_type == 'cross_attention':
        model = SemanticCrossAttention(
            input_dim=input_dim,
            num_layers=args.cross_attn_num_layers,
            num_heads=args.cross_attn_heads,
            hidden_dropout_prob=args.cross_attn_hidden_dropout,
            attn_dropout_prob=args.cross_attn_attn_dropout,
            emb_dropout_prob=args.cross_attn_emb_dropout,
            activation=args.activation,
            use_pos=args.cross_attn_use_pos,
            max_seq_length=args.cross_attn_num_neighbors,
            add_time_to_history=args.cross_attn_add_time_to_history,
            time_encoder_type=args.time_encoder_type,
            time_encoder_mask_padding=args.time_encoder_mask_padding,
            time_encoder_fourier_dim=args.time_encoder_fourier_dim,
            time_encoder_rbf_dim=args.time_encoder_rbf_dim,
            time_encoder_rbf_gamma=FIXED_TIME_ENCODER_RBF_GAMMA,
        ).to(device)
    elif args.scorer_type == 'dygformer_lite':
        model = SemanticDyGFormerLiteScorer(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.dygformer_num_layers,
            head_num_layers=args.num_layers,
            num_neighbors=args.dygformer_num_neighbors,
            num_heads=args.dygformer_heads,
            dropout=args.dropout,
            activation=args.activation,
            use_layernorm=args.use_layernorm,
            add_time=args.dygformer_add_time,
            time_encoder_type=args.time_encoder_type,
            time_encoder_fourier_dim=args.time_encoder_fourier_dim,
            time_encoder_rbf_dim=args.time_encoder_rbf_dim,
            time_encoder_rbf_gamma=FIXED_TIME_ENCODER_RBF_GAMMA,
            time_encoder_mask_padding=args.time_encoder_mask_padding,
        ).to(device)
    elif args.scorer_type == 'ncn':
        model = SemanticNCNScorer(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            activation=args.activation,
            use_layernorm=args.use_layernorm,
        ).to(device)
    elif args.scorer_type == 'seqfilter':
        model = SemanticSeqFilterScorer(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_neighbors=args.seqfilter_num_neighbors,
            tau=args.seqfilter_tau,
            kernel_size=args.seqfilter_kernel_size,
            dropout=args.dropout,
            activation=args.activation,
            use_layernorm=args.use_layernorm,
            time_encoder_type=args.time_encoder_type,
            time_encoder_fourier_dim=args.time_encoder_fourier_dim,
            time_encoder_rbf_dim=args.time_encoder_rbf_dim,
            time_encoder_rbf_gamma=FIXED_TIME_ENCODER_RBF_GAMMA,
            use_soft_mask=args.seqfilter_use_soft_mask,
        ).to(device)
    else:
        raise ValueError(f"Unsupported scorer type: {args.scorer_type}")

    if args.scorer_type == 'ridge':
        print("Precomputing the single ridge training negative pool...")
        ridge_neg_src, ridge_neg_dst, ridge_neg_times = build_precomputed_train_negative_pool(
            train_loader=train_loader,
            train_data=train_data,
            train_num_negatives=args.train_num_negatives,
            train_neg_sampler=train_neg_sampler,
            train_neg_sampler_random=train_neg_sampler_random,
            train_neg_sampler_historical=train_neg_sampler_historical,
            train_rand_ratio=args.train_rand_ratio,
            historical_strict_gap=args.historical_neg_strict_gap,
            desc="Sample ridge train negatives",
            debug=args.debug_negative_precompute,
            debug_every=args.debug_negative_precompute_every,
        )
        ridge_neg_raw_features = None
        if heuristic_extractor is not None:
            ridge_neg_raw_features = heuristic_extractor.precompute_raw_features(
                sources=ridge_neg_src,
                targets=ridge_neg_dst,
                prediction_times=ridge_neg_times,
                desc="Heuristics: ridge train negatives",
                store_in_cache=False,
            )

        fit_start = time.perf_counter()
        ridge_stats = fit_pairwise_ridge(
            model=model,
            train_loader=train_loader,
            train_data=train_data,
            embeddings=active_embeddings,
            lookup=lookup,
            train_num_negatives=args.train_num_negatives,
            train_neg_src_all=ridge_neg_src,
            train_neg_dst_all=ridge_neg_dst,
            lambda_value=args.ridge_lambda,
            rolling_provider=train_rolling_provider,
            static_ncn_adj=static_ncn_adj,
            static_mplp_exact_adj2=static_mplp_exact_adj2,
            heuristic_extractor=heuristic_extractor,
            use_mplp_exact_features=args.use_mplp_exact_features,
            static_smoothing_adj=static_smoothing_adj,
            static_smoothing_steps=args.smooth_steps,
            train_pos_raw_heuristic_features=train_pos_raw_heuristic_features,
            train_neg_raw_heuristic_features=ridge_neg_raw_features,
        )
        fit_elapsed = time.perf_counter() - fit_start
        print(
            "Ridge fit complete: "
            f"pairs={ridge_stats.pair_count:,}, features={ridge_stats.feature_dim}, "
            f"lambda={ridge_stats.lambda_value:g}, coef_norm={ridge_stats.coefficient_norm:.4f}, "
            f"elapsed={fit_elapsed:.2f}s"
        )

        def _evaluate_ridge(
            *, data_loader, data_source, neg_sampler, num_negatives, desc,
            rolling_provider, pos_raw_features, neg_raw_features,
        ):
            return evaluate_split(
                model=model,
                data_loader=data_loader,
                data_source=data_source,
                neg_sampler=neg_sampler,
                embeddings=active_embeddings,
                lookup=lookup,
                num_negatives=num_negatives,
                desc=desc,
                rolling_provider=rolling_provider,
                static_ncn_adj=static_ncn_adj,
                static_mplp_exact_adj2=static_mplp_exact_adj2,
                heuristic_extractor=heuristic_extractor,
                heuristic_fusion=None,
                mplp_exact_fusion=None,
                semantic_aux_fusion_mode='late_concat',
                use_mplp_exact_input_features=args.use_mplp_exact_features,
                static_smoothing_adj=static_smoothing_adj,
                static_smoothing_steps=args.smooth_steps,
                precomputed_pos_raw_heuristic_features=pos_raw_features,
                precomputed_neg_raw_heuristic_features=neg_raw_features,
                dtgb_eval_batch_size=args.dtgb_eval_batch_size,
            )

        val_metrics = _evaluate_ridge(
            data_loader=val_loader,
            data_source=val_data,
            neg_sampler=val_neg_sampler,
            num_negatives=args.val_num_negatives,
            desc="Val ridge",
            rolling_provider=val_rolling_provider,
            pos_raw_features=val_pos_raw_heuristic_features,
            neg_raw_features=val_neg_raw_heuristic_features,
        )
        test_metrics = _evaluate_ridge(
            data_loader=test_loader,
            data_source=test_data,
            neg_sampler=test_neg_sampler,
            num_negatives=args.eval_num_negatives,
            desc="Test ridge",
            rolling_provider=test_rolling_provider,
            pos_raw_features=test_pos_raw_heuristic_features,
            neg_raw_features=test_neg_raw_heuristic_features,
        )
        new_node_metrics = _evaluate_ridge(
            data_loader=new_node_test_loader,
            data_source=new_node_test_data,
            neg_sampler=new_node_test_neg_sampler,
            num_negatives=args.eval_num_negatives,
            desc="Test-inductive ridge",
            rolling_provider=new_node_test_rolling_provider,
            pos_raw_features=new_node_pos_raw_heuristic_features,
            neg_raw_features=new_node_neg_raw_heuristic_features,
        )
        print(
            "Ridge results | "
            f"val_ap={val_metrics['average_precision']:.4f} val_auc={val_metrics['roc_auc']:.4f} "
            f"val_mrr={val_metrics['mrr']:.4f} | "
            f"test_ap={test_metrics['average_precision']:.4f} "
            f"test_auc={test_metrics['roc_auc']:.4f} test_mrr={test_metrics['mrr']:.4f} | "
            f"test_ind_ap={new_node_metrics['average_precision']:.4f} "
            f"test_ind_auc={new_node_metrics['roc_auc']:.4f} "
            f"test_ind_mrr={new_node_metrics['mrr']:.4f}"
        )

        checkpoint_dir = os.path.dirname(args.checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        ridge_checkpoint = {
            'state_dict': model.state_dict(),
            'ridge_projection_state_dict': (
                ridge_projection.state_dict() if ridge_projection is not None else None
            ),
            'model_config': {
                'scorer_type': 'ridge',
                'input_dim': input_dim,
                'raw_input_dim': raw_semantic_input_dim,
                'auxiliary_dim': semantic_mlp_auxiliary_dim,
                'pair_feature_mode': 'hadamard',
                'semantic_aux_fusion_mode': 'late_concat',
                'ridge_lambda': args.ridge_lambda,
                'ridge_projection_dim': args.ridge_projection_dim,
                'ridge_projection_seed': args.seed,
                'ridge_semantic_feature_mode': args.ridge_semantic_feature_mode,
            },
            'gcn_config': {'use_learnable_gcn': False},
            'heuristic_config': {
                'use_heuristic_features': args.use_heuristic_features,
                'heuristic_feature_names': list(args.heuristic_feature_names),
                'heuristic_recency_directed': args.heuristic_recency_directed,
                'heuristic_popularity_decay': args.heuristic_popularity_decay,
                'heuristic_recent_degree_window': args.heuristic_recent_degree_window,
                'use_gpu_heuristics': args.use_gpu_heuristics,
                'semantic_aux_fusion_mode': 'late_concat',
                'entity_text_path': entity_text_path,
            },
            'structural_config': {
                'use_mplp_exact_features': args.use_mplp_exact_features,
                'mplp_signature_dim': args.mplp_signature_dim,
                'semantic_aux_fusion_mode': 'late_concat',
            },
            'dataset_name': args.dataset_name,
            'rolling_smoothing': rolling_enabled,
            'strict_no_leakage': args.strict_no_leakage,
            'train_edge_cutoff_time': args.train_edge_cutoff_time,
            'train_edge_cutoff_ratio': args.train_edge_cutoff_ratio,
            'train_holdout_recent_edges': args.train_holdout_recent_edges,
            'train_holdout_metadata': train_holdout_metadata,
            'smoothing': {
                'enabled': semantic_smoothing_enabled,
                'time_window': args.smooth_time_window,
                'steps': args.smooth_steps,
                'decay_gamma': args.smooth_decay_gamma,
                'undirected': args.smooth_undirected,
                'log_dampen': args.smooth_log_dampen,
                'supernode_strength': args.smooth_supernode_strength,
                'endpoint_topk_recent': args.smooth_endpoint_topk_recent,
                'endpoint_topk_mode': args.smooth_endpoint_topk_mode,
                'smooth_cutoff_time': args.smooth_cutoff_time,
                'source_init': source_init_effective,
                'source_init_requested': source_init_requested,
                'source_init_effective': source_init_effective,
                'source_init_version': SEMANTIC_SOURCE_INIT_VERSION,
                'source_init_temporal_mode': source_init_temporal_mode,
            },
            'ridge_fit_stats': vars(ridge_stats),
            'best_val_ap': val_metrics['average_precision'],
        }
        torch.save(ridge_checkpoint, args.checkpoint_path)
        print(f"Saved semantic ridge checkpoint to {args.checkpoint_path}")
        return

    heuristic_fusion = None
    if args.use_heuristic_features and args.semantic_aux_fusion_mode != 'late_concat':
        heuristic_fusion = HeuristicFusionHead(
            feature_dim=len(args.heuristic_feature_names)
        ).to(device)
    mplp_exact_fusion = None
    if args.use_mplp_exact_features and args.semantic_aux_fusion_mode != 'late_concat':
        mplp_exact_fusion = MPLPExactFusionHead().to(device)

    gcn_encoder = None
    if args.use_learnable_gcn:
        if args.learnable_mp_type == 'gcn':
            gcn_encoder = LearnableGCNEncoder(
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.gcn_num_layers,
                dropout=args.dropout,
                activation='relu',
                use_layernorm=True,
                residual=True,
                use_linear_transform=args.gcn_use_linear_transform,
                relation_features=relation_features_t,
                temporal_relational_rank=args.temporal_relational_rank,
                temporal_relational_time_basis_dim=(
                    args.temporal_relational_time_basis_dim
                ),
            ).to(device)
        elif args.learnable_mp_type == 'gin':
            gcn_encoder = LearnableGINEncoder(
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.gcn_num_layers,
                dropout=args.dropout,
                activation='relu',
                use_layernorm=True,
                residual=True,
                nonparametric=args.gin_nonparametric,
                nonparametric_norm=args.gin_nonparametric_norm,
            ).to(device)
        elif args.learnable_mp_type == 'attn_pool':
            if input_dim % args.attn_mp_heads != 0:
                raise ValueError(
                    f"input_dim ({input_dim}) must be divisible by --attn_mp_heads ({args.attn_mp_heads})."
                )
            gcn_encoder = TemporalSelfAttentionPooling(
                input_dim=input_dim,
                num_layers=FIXED_ATTN_MP_LAYERS,
                num_heads=args.attn_mp_heads,
                dropout=args.dropout,
                activation=args.activation,
                residual=FIXED_ATTN_MP_RESIDUAL,
                time_encoder_type=args.time_encoder_type,
                time_encoder_mask_padding=args.time_encoder_mask_padding,
                time_encoder_fourier_dim=args.time_encoder_fourier_dim,
                time_encoder_rbf_dim=args.time_encoder_rbf_dim,
                time_encoder_rbf_gamma=FIXED_TIME_ENCODER_RBF_GAMMA,
            ).to(device)
        else:
            raise ValueError(f"Unsupported learnable_mp_type: {args.learnable_mp_type}")

    trainable_params = list(model.parameters())
    if entity_embedding_table is not None:
        trainable_params.extend(
            param for param in entity_embedding_table.parameters() if param.requires_grad
        )
    if structural_seed_projector is not None:
        trainable_params.extend(structural_seed_projector.parameters())
    if semantic_projector is not None:
        trainable_params.extend(semantic_projector.parameters())
    if gcn_encoder is not None:
        trainable_params.extend(gcn_encoder.parameters())
    if heuristic_fusion is not None:
        trainable_params.extend(heuristic_fusion.parameters())
    if mplp_exact_fusion is not None:
        trainable_params.extend(mplp_exact_fusion.parameters())

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    criterion = nn.BCEWithLogitsLoss()

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else int(total_steps * args.warmup_ratio)
    scheduler = build_lr_scheduler(
        optimizer=optimizer,
        scheduler_type=args.scheduler,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    best_val_ap = -1.0
    epochs_no_improve = 0
    profile_n = args.profile_batches if args.profile_runtime else 0

    def _run_train_epoch_with_optional_torch_profiler(epoch: int):
        if args.torch_profile_steps <= 0 or epoch != 1:
            return train_one_epoch(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                train_loader=train_loader,
                train_data=train_data,
                train_neg_sampler=train_neg_sampler,
                embeddings=active_embeddings,
                lookup=lookup,
                train_num_negatives=args.train_num_negatives,
                grad_clip_norm=args.grad_clip_norm,
                label_smoothing=args.label_smoothing,
                rolling_provider=train_rolling_provider,
                gcn_encoder=gcn_encoder,
                learnable_mp_type=args.learnable_mp_type,
                mp_adj=static_mp_adj,
                mp_neighbor_index=mp_neighbor_index,
                mp_num_neighbors=args.attn_mp_num_neighbors,
                neighbor_index=scorer_neighbor_index,
                static_ncn_adj=static_ncn_adj,
                static_mplp_exact_adj2=static_mplp_exact_adj2,
                cross_attn_num_neighbors=args.cross_attn_num_neighbors,
                ncn_num_neighbors=args.ncn_num_neighbors,
                seqfilter_num_neighbors=args.seqfilter_num_neighbors,
                cross_attn_use_raw_embeddings=args.cross_attn_use_raw_embeddings,
                train_neg_sampler_random=train_neg_sampler_random,
                train_neg_sampler_historical=train_neg_sampler_historical,
                train_rand_ratio=args.train_rand_ratio,
                historical_strict_gap=args.historical_neg_strict_gap,
                heuristic_extractor=heuristic_extractor,
                heuristic_fusion=heuristic_fusion,
                mplp_exact_fusion=mplp_exact_fusion,
                semantic_aux_fusion_mode=args.semantic_aux_fusion_mode,
                use_mplp_exact_input_features=(
                    args.semantic_aux_fusion_mode == 'late_concat' and args.use_mplp_exact_features
                ),
                entity_embedding_table=entity_embedding_table,
                structural_seed_features=structural_seed_features,
                structural_seed_projector=structural_seed_projector,
                semantic_projector=semantic_projector,
                static_smoothing_adj=static_smoothing_adj,
                static_smoothing_steps=args.smooth_steps,
                train_pos_raw_heuristic_features=train_pos_raw_heuristic_features,
                train_neg_src_all=train_neg_src_epoch,
                train_neg_dst_all=train_neg_dst_epoch,
                train_neg_raw_heuristic_features=train_neg_raw_heuristic_features_epoch,
                profile_batches=profile_n,
                profile_skip_batches=args.profile_skip_batches,
                profile_out=train_profile,
                profile_label=f"train@epoch{epoch:02d}",
                profile_print_early=args.profile_runtime,
                torch_profiler=None,
            )

        os.makedirs(args.torch_profile_dir, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        sort_key = "self_cpu_time_total"
        if device.type == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)
            sort_key = "self_cuda_time_total"
        wait_steps = 0
        warmup_steps = max(0, int(args.torch_profile_warmup_steps))
        active_steps = max(1, int(args.torch_profile_steps))
        schedule = torch.profiler.schedule(wait=wait_steps, warmup=warmup_steps, active=active_steps, repeat=1)
        worker_name = f"{args.dataset_name}_epoch{epoch:02d}"
        summary_path = os.path.join(args.torch_profile_dir, f"{worker_name}_summary.txt")
        trace_handler = torch.profiler.tensorboard_trace_handler(
            args.torch_profile_dir,
            worker_name=worker_name,
        )

        def _on_trace_ready(profile) -> None:
            trace_handler(profile)
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(profile.key_averages().table(
                    sort_by=sort_key,
                    row_limit=int(args.torch_profile_row_limit),
                ))
                f.write("\n")
            print(f"torch.profiler summary written to {summary_path}")

        print(
            f"Running torch.profiler for epoch {epoch:02d} "
            f"(warmup={warmup_steps}, active={active_steps})..."
        )
        with torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=_on_trace_ready,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as torch_prof:
            result = train_one_epoch(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                train_loader=train_loader,
                train_data=train_data,
                train_neg_sampler=train_neg_sampler,
                embeddings=active_embeddings,
                lookup=lookup,
                train_num_negatives=args.train_num_negatives,
                grad_clip_norm=args.grad_clip_norm,
                label_smoothing=args.label_smoothing,
                rolling_provider=train_rolling_provider,
                gcn_encoder=gcn_encoder,
                learnable_mp_type=args.learnable_mp_type,
                mp_adj=static_mp_adj,
                mp_neighbor_index=mp_neighbor_index,
                mp_num_neighbors=args.attn_mp_num_neighbors,
                neighbor_index=scorer_neighbor_index,
                static_ncn_adj=static_ncn_adj,
                static_mplp_exact_adj2=static_mplp_exact_adj2,
                cross_attn_num_neighbors=args.cross_attn_num_neighbors,
                ncn_num_neighbors=args.ncn_num_neighbors,
                seqfilter_num_neighbors=args.seqfilter_num_neighbors,
                cross_attn_use_raw_embeddings=args.cross_attn_use_raw_embeddings,
                train_neg_sampler_random=train_neg_sampler_random,
                train_neg_sampler_historical=train_neg_sampler_historical,
                train_rand_ratio=args.train_rand_ratio,
                historical_strict_gap=args.historical_neg_strict_gap,
                heuristic_extractor=heuristic_extractor,
                heuristic_fusion=heuristic_fusion,
                mplp_exact_fusion=mplp_exact_fusion,
                semantic_aux_fusion_mode=args.semantic_aux_fusion_mode,
                use_mplp_exact_input_features=(
                    args.semantic_aux_fusion_mode == 'late_concat' and args.use_mplp_exact_features
                ),
                entity_embedding_table=entity_embedding_table,
                structural_seed_features=structural_seed_features,
                structural_seed_projector=structural_seed_projector,
                semantic_projector=semantic_projector,
                static_smoothing_adj=static_smoothing_adj,
                static_smoothing_steps=args.smooth_steps,
                train_pos_raw_heuristic_features=train_pos_raw_heuristic_features,
                train_neg_src_all=train_neg_src_epoch,
                train_neg_dst_all=train_neg_dst_epoch,
                train_neg_raw_heuristic_features=train_neg_raw_heuristic_features_epoch,
                profile_batches=profile_n,
                profile_skip_batches=args.profile_skip_batches,
                profile_out=train_profile,
                profile_label=f"train@epoch{epoch:02d}",
                profile_print_early=args.profile_runtime,
                torch_profiler=torch_prof,
            )
        return result

    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        train_neg_src_epoch = None
        train_neg_dst_epoch = None
        train_neg_raw_heuristic_features_epoch = None
        if args.freeze_train_negatives_per_epoch:
            print(f"Precomputing train negatives for epoch {epoch:02d}...")
            train_neg_src_epoch, train_neg_dst_epoch, train_neg_times_epoch = build_precomputed_train_negative_pool(
                train_loader=train_loader,
                train_data=train_data,
                train_num_negatives=args.train_num_negatives,
                train_neg_sampler=train_neg_sampler,
                train_neg_sampler_random=train_neg_sampler_random,
                train_neg_sampler_historical=train_neg_sampler_historical,
                train_rand_ratio=args.train_rand_ratio,
                historical_strict_gap=args.historical_neg_strict_gap,
                desc=f"Sample train negatives @ epoch {epoch:02d}",
                debug=args.debug_negative_precompute,
                debug_every=args.debug_negative_precompute_every,
            )
            if heuristic_extractor is not None:
                train_neg_raw_heuristic_features_epoch = heuristic_extractor.precompute_raw_features(
                    sources=train_neg_src_epoch,
                    targets=train_neg_dst_epoch,
                    prediction_times=train_neg_times_epoch,
                    desc=f"Heuristics: train negatives @ epoch {epoch:02d}",
                    store_in_cache=False,
                )

        train_profile = {}
        train_loss, lr_now, train_steps = _run_train_epoch_with_optional_torch_profiler(epoch)

        train_eval_metrics = None
        if args.report_train_metrics:
            train_eval_metrics = evaluate_split(
                model=model,
                data_loader=train_eval_loader,
                data_source=train_data,
                neg_sampler=train_eval_neg_sampler,
                embeddings=active_embeddings,
                lookup=lookup,
                num_negatives=args.val_num_negatives,
                desc="Train-Eval",
                rolling_provider=train_rolling_provider,
                gcn_encoder=gcn_encoder,
                learnable_mp_type=args.learnable_mp_type,
                mp_adj=static_mp_adj,
                mp_neighbor_index=mp_neighbor_index,
                mp_num_neighbors=args.attn_mp_num_neighbors,
                neighbor_index=scorer_neighbor_index,
                static_ncn_adj=static_ncn_adj,
                static_mplp_exact_adj2=static_mplp_exact_adj2,
                cross_attn_num_neighbors=args.cross_attn_num_neighbors,
                ncn_num_neighbors=args.ncn_num_neighbors,
                seqfilter_num_neighbors=args.seqfilter_num_neighbors,
                cross_attn_use_raw_embeddings=args.cross_attn_use_raw_embeddings,
                heuristic_extractor=heuristic_extractor,
                heuristic_fusion=heuristic_fusion,
                mplp_exact_fusion=mplp_exact_fusion,
                semantic_aux_fusion_mode=args.semantic_aux_fusion_mode,
                use_mplp_exact_input_features=(
                    args.semantic_aux_fusion_mode == 'late_concat' and args.use_mplp_exact_features
                ),
                entity_embedding_table=entity_embedding_table,
                structural_seed_features=structural_seed_features,
                structural_seed_projector=structural_seed_projector,
                semantic_projector=semantic_projector,
                static_smoothing_adj=static_smoothing_adj,
                static_smoothing_steps=args.smooth_steps,
                precomputed_pos_raw_heuristic_features=train_pos_raw_heuristic_features,
                precomputed_neg_raw_heuristic_features=None,
                dtgb_eval_batch_size=args.dtgb_eval_batch_size,
            )

        val_profile = {}
        val_metrics = evaluate_split(
            model=model,
            data_loader=val_loader,
            data_source=val_data,
            neg_sampler=val_neg_sampler,
            embeddings=active_embeddings,
            lookup=lookup,
            num_negatives=args.val_num_negatives,
            desc="Val",
            rolling_provider=val_rolling_provider,
            gcn_encoder=gcn_encoder,
            learnable_mp_type=args.learnable_mp_type,
            mp_adj=static_mp_adj,
            mp_neighbor_index=mp_neighbor_index,
            mp_num_neighbors=args.attn_mp_num_neighbors,
            neighbor_index=scorer_neighbor_index,
            static_ncn_adj=static_ncn_adj,
            static_mplp_exact_adj2=static_mplp_exact_adj2,
            cross_attn_num_neighbors=args.cross_attn_num_neighbors,
            ncn_num_neighbors=args.ncn_num_neighbors,
            seqfilter_num_neighbors=args.seqfilter_num_neighbors,
            cross_attn_use_raw_embeddings=args.cross_attn_use_raw_embeddings,
            heuristic_extractor=heuristic_extractor,
            heuristic_fusion=heuristic_fusion,
            mplp_exact_fusion=mplp_exact_fusion,
            semantic_aux_fusion_mode=args.semantic_aux_fusion_mode,
            use_mplp_exact_input_features=(
                args.semantic_aux_fusion_mode == 'late_concat' and args.use_mplp_exact_features
            ),
            entity_embedding_table=entity_embedding_table,
            structural_seed_features=structural_seed_features,
            structural_seed_projector=structural_seed_projector,
            semantic_projector=semantic_projector,
            static_smoothing_adj=static_smoothing_adj,
            static_smoothing_steps=args.smooth_steps,
            precomputed_pos_raw_heuristic_features=val_pos_raw_heuristic_features,
            precomputed_neg_raw_heuristic_features=val_neg_raw_heuristic_features,
            dtgb_eval_batch_size=args.dtgb_eval_batch_size,
            profile_batches=profile_n,
            profile_skip_batches=args.profile_skip_batches,
            profile_out=val_profile,
        )

        train_metric_text = ""
        if train_eval_metrics is not None:
            train_metric_text = (
                f"train_ap={train_eval_metrics['average_precision']:.4f} | "
                f"train_auc={train_eval_metrics['roc_auc']:.4f} | "
                f"train_mrr={train_eval_metrics['mrr']:.4f} | "
            )
        print(
            f"Epoch {epoch:02d} | "
            f"steps={train_steps} | "
            f"lr={lr_now:.2e} | "
            f"loss={train_loss:.4f} | "
            f"{train_metric_text}"
            f"val_ap={val_metrics['average_precision']:.4f} | "
            f"val_auc={val_metrics['roc_auc']:.4f} | "
            f"val_mrr={val_metrics['mrr']:.4f}"
        )

        test_profile = {}
        test_metrics_epoch = evaluate_split(
            model=model,
            data_loader=test_loader,
            data_source=test_data,
            neg_sampler=test_neg_sampler,
            embeddings=active_embeddings,
            lookup=lookup,
            num_negatives=args.eval_num_negatives,
            desc=f"Test-Transductive@Epoch{epoch:02d}",
            rolling_provider=test_rolling_provider,
            gcn_encoder=gcn_encoder,
            learnable_mp_type=args.learnable_mp_type,
            mp_adj=static_mp_adj,
            mp_neighbor_index=mp_neighbor_index,
            mp_num_neighbors=args.attn_mp_num_neighbors,
            neighbor_index=scorer_neighbor_index,
            static_ncn_adj=static_ncn_adj,
            static_mplp_exact_adj2=static_mplp_exact_adj2,
            cross_attn_num_neighbors=args.cross_attn_num_neighbors,
            ncn_num_neighbors=args.ncn_num_neighbors,
            seqfilter_num_neighbors=args.seqfilter_num_neighbors,
            cross_attn_use_raw_embeddings=args.cross_attn_use_raw_embeddings,
            heuristic_extractor=heuristic_extractor,
            heuristic_fusion=heuristic_fusion,
            mplp_exact_fusion=mplp_exact_fusion,
            semantic_aux_fusion_mode=args.semantic_aux_fusion_mode,
            use_mplp_exact_input_features=(
                args.semantic_aux_fusion_mode == 'late_concat' and args.use_mplp_exact_features
            ),
            entity_embedding_table=entity_embedding_table,
            structural_seed_features=structural_seed_features,
            structural_seed_projector=structural_seed_projector,
            semantic_projector=semantic_projector,
            static_smoothing_adj=static_smoothing_adj,
            static_smoothing_steps=args.smooth_steps,
            precomputed_pos_raw_heuristic_features=test_pos_raw_heuristic_features,
            precomputed_neg_raw_heuristic_features=test_neg_raw_heuristic_features,
            dtgb_eval_batch_size=args.dtgb_eval_batch_size,
            profile_batches=profile_n,
            profile_skip_batches=args.profile_skip_batches,
            profile_out=test_profile,
        )
        new_node_test_profile = {}
        new_node_metrics_epoch = evaluate_split(
            model=model,
            data_loader=new_node_test_loader,
            data_source=new_node_test_data,
            neg_sampler=new_node_test_neg_sampler,
            embeddings=active_embeddings,
            lookup=lookup,
            num_negatives=args.eval_num_negatives,
            desc=f"Test-Inductive@Epoch{epoch:02d}",
            rolling_provider=new_node_test_rolling_provider,
            gcn_encoder=gcn_encoder,
            learnable_mp_type=args.learnable_mp_type,
            mp_adj=static_mp_adj,
            mp_neighbor_index=mp_neighbor_index,
            mp_num_neighbors=args.attn_mp_num_neighbors,
            neighbor_index=scorer_neighbor_index,
            static_ncn_adj=static_ncn_adj,
            static_mplp_exact_adj2=static_mplp_exact_adj2,
            cross_attn_num_neighbors=args.cross_attn_num_neighbors,
            ncn_num_neighbors=args.ncn_num_neighbors,
            seqfilter_num_neighbors=args.seqfilter_num_neighbors,
            cross_attn_use_raw_embeddings=args.cross_attn_use_raw_embeddings,
            heuristic_extractor=heuristic_extractor,
            heuristic_fusion=heuristic_fusion,
            mplp_exact_fusion=mplp_exact_fusion,
            semantic_aux_fusion_mode=args.semantic_aux_fusion_mode,
            use_mplp_exact_input_features=(
                args.semantic_aux_fusion_mode == 'late_concat' and args.use_mplp_exact_features
            ),
            entity_embedding_table=entity_embedding_table,
            structural_seed_features=structural_seed_features,
            structural_seed_projector=structural_seed_projector,
            semantic_projector=semantic_projector,
            static_smoothing_adj=static_smoothing_adj,
            static_smoothing_steps=args.smooth_steps,
            precomputed_pos_raw_heuristic_features=new_node_pos_raw_heuristic_features,
            precomputed_neg_raw_heuristic_features=new_node_neg_raw_heuristic_features,
            dtgb_eval_batch_size=args.dtgb_eval_batch_size,
            profile_batches=profile_n,
            profile_skip_batches=args.profile_skip_batches,
            profile_out=new_node_test_profile,
        )
        print(
            f"  test_ap={test_metrics_epoch['average_precision']:.4f} | "
            f"test_auc={test_metrics_epoch['roc_auc']:.4f} | "
            f"test_mrr={test_metrics_epoch['mrr']:.4f} | "
            f"test_ind_ap={new_node_metrics_epoch['average_precision']:.4f} | "
            f"test_ind_auc={new_node_metrics_epoch['roc_auc']:.4f} | "
            f"test_ind_mrr={new_node_metrics_epoch['mrr']:.4f}"
        )
        if args.profile_runtime:
            print(
                f"  profile/train: prep={train_profile['prepare_ms']:.1f}ms ({train_profile['prepare_pct']:.1f}%), "
                f"neg={train_profile['neg_ms']:.1f}ms ({train_profile['neg_pct']:.1f}%), "
                f"mp={train_profile['mp_ms']:.1f}ms ({train_profile['mp_pct']:.1f}%), "
                f"heur={train_profile['heur_ms']:.1f}ms ({train_profile['heur_pct']:.1f}%), "
                f"mplp={train_profile['mplp_ms']:.1f}ms ({train_profile['mplp_pct']:.1f}%), "
                f"score={train_profile['score_ms']:.1f}ms ({train_profile['score_pct']:.1f}%), "
                f"back={train_profile['backward_ms']:.1f}ms ({train_profile['backward_pct']:.1f}%), "
                f"commit={train_profile['commit_ms']:.1f}ms ({train_profile['commit_pct']:.1f}%), "
                f"other={train_profile['other_ms']:.1f}ms ({train_profile['other_pct']:.1f}%), "
                f"total={train_profile['total_ms']:.1f}ms"
            )
            if args.scorer_type == 'ncn' or args.use_mplp_exact_features:
                print(
                    f"    ncn/train: row_slice={train_profile['ncn_row_slice_ms']:.1f}ms, "
                    f"overlap={train_profile['ncn_overlap_ms']:.1f}ms, "
                    f"aggregate={train_profile['ncn_aggregate_ms']:.1f}ms, "
                    f"hist_fetch={train_profile['ncn_history_fetch_ms']:.1f}ms, "
                    f"marshal={train_profile['ncn_marshal_ms']:.1f}ms, "
                    f"sparse_build={train_profile['ncn_sparse_build_ms']:.1f}ms, "
                    f"edges={train_profile['ncn_edges']:.0f}"
                )
            print(
                f"  profile/val: prep={val_profile['prepare_ms']:.1f}ms ({val_profile['prepare_pct']:.1f}%), "
                f"neg={val_profile['neg_ms']:.1f}ms ({val_profile['neg_pct']:.1f}%), "
                f"mp={val_profile['mp_ms']:.1f}ms ({val_profile['mp_pct']:.1f}%), "
                f"heur={val_profile['heur_ms']:.1f}ms ({val_profile['heur_pct']:.1f}%), "
                f"mplp={val_profile['mplp_ms']:.1f}ms ({val_profile['mplp_pct']:.1f}%), "
                f"score={val_profile['score_ms']:.1f}ms ({val_profile['score_pct']:.1f}%), "
                f"commit={val_profile['commit_ms']:.1f}ms ({val_profile['commit_pct']:.1f}%), "
                f"other={val_profile['other_ms']:.1f}ms ({val_profile['other_pct']:.1f}%), "
                f"total={val_profile['total_ms']:.1f}ms"
            )
            if args.scorer_type == 'ncn' or args.use_mplp_exact_features:
                print(
                    f"    ncn/val: row_slice={val_profile['ncn_row_slice_ms']:.1f}ms, "
                    f"overlap={val_profile['ncn_overlap_ms']:.1f}ms, "
                    f"aggregate={val_profile['ncn_aggregate_ms']:.1f}ms, "
                    f"hist_fetch={val_profile['ncn_history_fetch_ms']:.1f}ms, "
                    f"marshal={val_profile['ncn_marshal_ms']:.1f}ms, "
                    f"sparse_build={val_profile['ncn_sparse_build_ms']:.1f}ms, "
                    f"edges={val_profile['ncn_edges']:.0f}"
                )
            print(
                f"  profile/test: prep={test_profile['prepare_ms']:.1f}ms ({test_profile['prepare_pct']:.1f}%), "
                f"neg={test_profile['neg_ms']:.1f}ms ({test_profile['neg_pct']:.1f}%), "
                f"mp={test_profile['mp_ms']:.1f}ms ({test_profile['mp_pct']:.1f}%), "
                f"heur={test_profile['heur_ms']:.1f}ms ({test_profile['heur_pct']:.1f}%), "
                f"mplp={test_profile['mplp_ms']:.1f}ms ({test_profile['mplp_pct']:.1f}%), "
                f"score={test_profile['score_ms']:.1f}ms ({test_profile['score_pct']:.1f}%), "
                f"commit={test_profile['commit_ms']:.1f}ms ({test_profile['commit_pct']:.1f}%), "
                f"other={test_profile['other_ms']:.1f}ms ({test_profile['other_pct']:.1f}%), "
                f"total={test_profile['total_ms']:.1f}ms"
            )
            if args.scorer_type == 'ncn' or args.use_mplp_exact_features:
                print(
                    f"    ncn/test: row_slice={test_profile['ncn_row_slice_ms']:.1f}ms, "
                    f"overlap={test_profile['ncn_overlap_ms']:.1f}ms, "
                    f"aggregate={test_profile['ncn_aggregate_ms']:.1f}ms, "
                    f"hist_fetch={test_profile['ncn_history_fetch_ms']:.1f}ms, "
                    f"marshal={test_profile['ncn_marshal_ms']:.1f}ms, "
                    f"sparse_build={test_profile['ncn_sparse_build_ms']:.1f}ms, "
                    f"edges={test_profile['ncn_edges']:.0f}"
                )
            print(
                f"  profile/test_ind: prep={new_node_test_profile['prepare_ms']:.1f}ms ({new_node_test_profile['prepare_pct']:.1f}%), "
                f"neg={new_node_test_profile['neg_ms']:.1f}ms ({new_node_test_profile['neg_pct']:.1f}%), "
                f"mp={new_node_test_profile['mp_ms']:.1f}ms ({new_node_test_profile['mp_pct']:.1f}%), "
                f"heur={new_node_test_profile['heur_ms']:.1f}ms ({new_node_test_profile['heur_pct']:.1f}%), "
                f"mplp={new_node_test_profile['mplp_ms']:.1f}ms ({new_node_test_profile['mplp_pct']:.1f}%), "
                f"score={new_node_test_profile['score_ms']:.1f}ms ({new_node_test_profile['score_pct']:.1f}%), "
                f"commit={new_node_test_profile['commit_ms']:.1f}ms ({new_node_test_profile['commit_pct']:.1f}%), "
                f"other={new_node_test_profile['other_ms']:.1f}ms ({new_node_test_profile['other_pct']:.1f}%), "
                f"total={new_node_test_profile['total_ms']:.1f}ms"
            )
            if args.scorer_type == 'ncn' or args.use_mplp_exact_features:
                print(
                    f"    ncn/test_ind: row_slice={new_node_test_profile['ncn_row_slice_ms']:.1f}ms, "
                    f"overlap={new_node_test_profile['ncn_overlap_ms']:.1f}ms, "
                    f"aggregate={new_node_test_profile['ncn_aggregate_ms']:.1f}ms, "
                    f"hist_fetch={new_node_test_profile['ncn_history_fetch_ms']:.1f}ms, "
                    f"marshal={new_node_test_profile['ncn_marshal_ms']:.1f}ms, "
                    f"sparse_build={new_node_test_profile['ncn_sparse_build_ms']:.1f}ms, "
                    f"edges={new_node_test_profile['ncn_edges']:.0f}"
                )

        if val_metrics['average_precision'] > (best_val_ap + args.early_stopping_min_delta):
            best_val_ap = val_metrics['average_precision']
            epochs_no_improve = 0
            ckpt = {
                'state_dict': model.state_dict(),
                'entity_embedding_state_dict': (
                    entity_embedding_table.state_dict() if entity_embedding_table is not None else None
                ),
                'structural_seed_projector_state_dict': (
                    structural_seed_projector.state_dict() if structural_seed_projector is not None else None
                ),
                'semantic_projector_state_dict': (
                    semantic_projector.state_dict() if semantic_projector is not None else None
                ),
                'gcn_state_dict': gcn_encoder.state_dict() if gcn_encoder is not None else None,
                'heuristic_fusion_state_dict': heuristic_fusion.state_dict() if heuristic_fusion is not None else None,
                'mplp_exact_fusion_state_dict': (
                    mplp_exact_fusion.state_dict() if mplp_exact_fusion is not None else None
                ),
                'model_config': {
                    'scorer_type': args.scorer_type,
                    'input_dim': input_dim,
                    'raw_input_dim': raw_input_dim,
                    'semantic_project_dim': args.semantic_project_dim,
                    'auxiliary_dim': semantic_mlp_auxiliary_dim,
                    'pair_feature_mode': args.semantic_mlp_pair_feature_mode,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'dropout': args.dropout,
                    'activation': args.activation,
                    'use_layernorm': args.use_layernorm,
                    'cross_attn_heads': args.cross_attn_heads,
                    'cross_attn_num_layers': args.cross_attn_num_layers,
                    'cross_attn_num_neighbors': args.cross_attn_num_neighbors,
                    'dygformer_heads': args.dygformer_heads,
                    'dygformer_num_layers': args.dygformer_num_layers,
                    'dygformer_num_neighbors': args.dygformer_num_neighbors,
                    'dygformer_add_time': args.dygformer_add_time,
                    'ncn_num_neighbors': args.ncn_num_neighbors,
                    'seqfilter_num_neighbors': args.seqfilter_num_neighbors,
                    'seqfilter_tau': args.seqfilter_tau,
                    'seqfilter_kernel_size': args.seqfilter_kernel_size,
                    'seqfilter_use_soft_mask': args.seqfilter_use_soft_mask,
                    'cross_attn_use_pos': args.cross_attn_use_pos,
                    'cross_attn_add_time_to_history': args.cross_attn_add_time_to_history,
                    'cross_attn_use_raw_embeddings': args.cross_attn_use_raw_embeddings,
                    'cross_attn_hidden_dropout': args.cross_attn_hidden_dropout,
                    'cross_attn_attn_dropout': args.cross_attn_attn_dropout,
                    'cross_attn_emb_dropout': args.cross_attn_emb_dropout,
                    'cross_attn_undirected_history': FIXED_CROSS_ATTN_UNDIRECTED_HISTORY,
                    'time_encoder_type': args.time_encoder_type,
                    'time_encoder_mask_padding': args.time_encoder_mask_padding,
                    'time_encoder_fourier_dim': args.time_encoder_fourier_dim,
                    'time_encoder_rbf_dim': args.time_encoder_rbf_dim,
                    'time_encoder_rbf_gamma': FIXED_TIME_ENCODER_RBF_GAMMA,
                    'semantic_aux_fusion_mode': args.semantic_aux_fusion_mode,
                },
                'gcn_config': {
                    'use_learnable_gcn': args.use_learnable_gcn,
                    'learnable_mp_type': args.learnable_mp_type,
                    'gcn_num_layers': args.gcn_num_layers,
                    'gcn_hidden_dim': args.hidden_dim,
                    'gcn_dropout': args.dropout,
                    'gcn_activation': 'relu',
                    'gcn_use_layernorm': True,
                    'gcn_residual': True,
                    'gcn_use_linear_transform': args.gcn_use_linear_transform,
                    'use_temporal_relational_gcn': args.use_temporal_relational_gcn,
                    'temporal_relational_rank': args.temporal_relational_rank,
                    'temporal_relational_time_basis_dim': (
                        args.temporal_relational_time_basis_dim
                    ),
                    'temporal_relational_edge_feature_path': relation_feature_path,
                    'gin_nonparametric': args.gin_nonparametric,
                    'gin_nonparametric_norm': args.gin_nonparametric_norm,
                    'gcn_undirected': args.smooth_undirected,
                    'attn_mp_heads': args.attn_mp_heads,
                    'attn_mp_layers': FIXED_ATTN_MP_LAYERS,
                    'attn_mp_num_neighbors': args.attn_mp_num_neighbors,
                    'attn_mp_dropout': args.dropout,
                    'attn_mp_residual': FIXED_ATTN_MP_RESIDUAL,
                    'time_encoder_type': args.time_encoder_type,
                    'time_encoder_mask_padding': args.time_encoder_mask_padding,
                    'time_encoder_fourier_dim': args.time_encoder_fourier_dim,
                    'time_encoder_rbf_dim': args.time_encoder_rbf_dim,
                    'time_encoder_rbf_gamma': FIXED_TIME_ENCODER_RBF_GAMMA,
                },
                'optimizer_config': {
                    'lr': args.lr,
                    'weight_decay': args.weight_decay,
                    'scheduler': args.scheduler,
                    'warmup_steps': warmup_steps,
                    'min_lr_ratio': args.min_lr_ratio,
                    'grad_clip_norm': args.grad_clip_norm,
                    'label_smoothing': args.label_smoothing,
                },
                'dataset_name': args.dataset_name,
                'entity_text_config': {
                    'entity_text_path': entity_text_path,
                    'embedding_model': args.embedding_model,
                    'embedding_entity_name_mode': args.embedding_entity_name_mode,
                    'use_learnable_entity_embeddings': args.use_learnable_entity_embeddings,
                    'learnable_entity_embedding_dim': args.learnable_entity_embedding_dim,
                    'learnable_entity_embedding_init': args.learnable_entity_embedding_init,
                    'freeze_learnable_entity_embeddings': args.freeze_learnable_entity_embeddings,
                    'use_structural_seed_features': args.use_structural_seed_features,
                    'semantic_project_dim': args.semantic_project_dim,
                    'structural_seed_feature_dim': (
                        int(structural_seed_features.shape[1]) if structural_seed_features is not None else 0
                    ),
                    'num_entities': len(entity_ids),
                    'transformed_entities': int(transformed_count),
                },
                'rolling_smoothing': rolling_enabled,
                'strict_no_leakage': args.strict_no_leakage,
                'heuristic_config': {
                    'use_heuristic_features': args.use_heuristic_features,
                    'heuristic_feature_names': list(args.heuristic_feature_names),
                    'heuristic_recency_directed': args.heuristic_recency_directed,
                    'heuristic_popularity_decay': args.heuristic_popularity_decay,
                    'heuristic_recent_degree_window': args.heuristic_recent_degree_window,
                    'use_gpu_heuristics': args.use_gpu_heuristics,
                    'semantic_aux_fusion_mode': args.semantic_aux_fusion_mode,
                    'entity_text_path': entity_text_path,
                },
                'structural_config': {
                    'use_mplp_exact_features': args.use_mplp_exact_features,
                    'mplp_signature_dim': args.mplp_signature_dim,
                    'semantic_aux_fusion_mode': args.semantic_aux_fusion_mode,
                },
                'train_edge_cutoff_time': args.train_edge_cutoff_time,
                'train_edge_cutoff_ratio': args.train_edge_cutoff_ratio,
                'train_holdout_recent_edges': args.train_holdout_recent_edges,
                'train_holdout_metadata': train_holdout_metadata,
                'smoothing': {
                    'enabled': semantic_smoothing_enabled,
                    'time_window': args.smooth_time_window,
                    'steps': args.smooth_steps,
                    'decay_gamma': args.smooth_decay_gamma,
                    'undirected': args.smooth_undirected,
                    'log_dampen': args.smooth_log_dampen,
                    'supernode_strength': args.smooth_supernode_strength,
                    'endpoint_topk_recent': args.smooth_endpoint_topk_recent,
                    'endpoint_topk_mode': args.smooth_endpoint_topk_mode,
                    'smooth_cutoff_time': args.smooth_cutoff_time,
                    'source_init': source_init_effective,
                    'source_init_requested': source_init_requested,
                    'source_init_effective': source_init_effective,
                    'source_init_version': SEMANTIC_SOURCE_INIT_VERSION,
                    'source_init_temporal_mode': source_init_temporal_mode,
                },
                'best_val_ap': best_val_ap,
            }
            torch.save(ckpt, args.checkpoint_path)
            print(f"Saved new best checkpoint to {args.checkpoint_path}")
        else:
            epochs_no_improve += 1

        log_payload = {
            'epoch': epoch,
            'train/loss': train_loss,
            'train/lr': lr_now,
            'train/steps': train_steps,
            'val/ap': val_metrics['average_precision'],
            'val/auc': val_metrics['roc_auc'],
            'val/mrr': val_metrics['mrr'],
            'val/best_ap': best_val_ap,
            'epoch_test/transductive_ap': test_metrics_epoch['average_precision'],
            'epoch_test/transductive_auc': test_metrics_epoch['roc_auc'],
            'epoch_test/transductive_mrr': test_metrics_epoch['mrr'],
            'epoch_test/inductive_ap': new_node_metrics_epoch['average_precision'],
            'epoch_test/inductive_auc': new_node_metrics_epoch['roc_auc'],
            'epoch_test/inductive_mrr': new_node_metrics_epoch['mrr'],
        }
        if train_eval_metrics is not None:
            log_payload.update({
                'train/ap': train_eval_metrics['average_precision'],
                'train/auc': train_eval_metrics['roc_auc'],
                'train/mrr': train_eval_metrics['mrr'],
            })
        if args.profile_runtime:
            log_payload.update({
                'profile/train_prepare_ms': train_profile['prepare_ms'],
                'profile/train_prepare_pct': train_profile['prepare_pct'],
                'profile/train_heur_ms': train_profile['heur_ms'],
                'profile/train_heur_pct': train_profile['heur_pct'],
                'profile/train_score_ms': train_profile['score_ms'],
                'profile/train_score_pct': train_profile['score_pct'],
                'profile/train_score_id_to_tensor_ms': train_profile['score_id_to_tensor_ms'],
                'profile/train_score_lookup_ms': train_profile['score_lookup_ms'],
                'profile/train_score_neighbor_fetch_ms': train_profile['score_neighbor_fetch_ms'],
                'profile/train_score_neighbor_marshal_ms': train_profile['score_neighbor_marshal_ms'],
                'profile/train_score_gather_ms': train_profile['score_gather_ms'],
                'profile/train_score_forward_ms': train_profile['score_forward_ms'],
                'profile/train_backward_ms': train_profile['backward_ms'],
                'profile/train_backward_pct': train_profile['backward_pct'],
                'profile/train_total_ms': train_profile['total_ms'],
                'profile/val_prepare_ms': val_profile['prepare_ms'],
                'profile/val_prepare_pct': val_profile['prepare_pct'],
                'profile/val_heur_ms': val_profile['heur_ms'],
                'profile/val_heur_pct': val_profile['heur_pct'],
                'profile/val_score_ms': val_profile['score_ms'],
                'profile/val_score_pct': val_profile['score_pct'],
                'profile/val_score_id_to_tensor_ms': val_profile['score_id_to_tensor_ms'],
                'profile/val_score_lookup_ms': val_profile['score_lookup_ms'],
                'profile/val_score_neighbor_fetch_ms': val_profile['score_neighbor_fetch_ms'],
                'profile/val_score_neighbor_marshal_ms': val_profile['score_neighbor_marshal_ms'],
                'profile/val_score_gather_ms': val_profile['score_gather_ms'],
                'profile/val_score_forward_ms': val_profile['score_forward_ms'],
                'profile/val_total_ms': val_profile['total_ms'],
                'profile/test_prepare_ms': test_profile['prepare_ms'],
                'profile/test_prepare_pct': test_profile['prepare_pct'],
                'profile/test_heur_ms': test_profile['heur_ms'],
                'profile/test_heur_pct': test_profile['heur_pct'],
                'profile/test_score_ms': test_profile['score_ms'],
                'profile/test_score_pct': test_profile['score_pct'],
                'profile/test_score_id_to_tensor_ms': test_profile['score_id_to_tensor_ms'],
                'profile/test_score_lookup_ms': test_profile['score_lookup_ms'],
                'profile/test_score_neighbor_fetch_ms': test_profile['score_neighbor_fetch_ms'],
                'profile/test_score_neighbor_marshal_ms': test_profile['score_neighbor_marshal_ms'],
                'profile/test_score_gather_ms': test_profile['score_gather_ms'],
                'profile/test_score_forward_ms': test_profile['score_forward_ms'],
                'profile/test_total_ms': test_profile['total_ms'],
                'profile/new_test_prepare_ms': new_node_test_profile['prepare_ms'],
                'profile/new_test_prepare_pct': new_node_test_profile['prepare_pct'],
                'profile/new_test_heur_ms': new_node_test_profile['heur_ms'],
                'profile/new_test_heur_pct': new_node_test_profile['heur_pct'],
                'profile/new_test_score_ms': new_node_test_profile['score_ms'],
                'profile/new_test_score_pct': new_node_test_profile['score_pct'],
                'profile/new_test_score_id_to_tensor_ms': new_node_test_profile['score_id_to_tensor_ms'],
                'profile/new_test_score_lookup_ms': new_node_test_profile['score_lookup_ms'],
                'profile/new_test_score_neighbor_fetch_ms': new_node_test_profile['score_neighbor_fetch_ms'],
                'profile/new_test_score_neighbor_marshal_ms': new_node_test_profile['score_neighbor_marshal_ms'],
                'profile/new_test_score_gather_ms': new_node_test_profile['score_gather_ms'],
                'profile/new_test_score_forward_ms': new_node_test_profile['score_forward_ms'],
                'profile/new_test_total_ms': new_node_test_profile['total_ms'],
            })
        wandb.log(log_payload, step=epoch)

        if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}: no val AP improvement for {epochs_no_improve} epochs")
            break

    if not os.path.exists(args.checkpoint_path):
        raise RuntimeError("No checkpoint saved. Training may have produced zero valid batches.")

    ckpt = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    if entity_embedding_table is not None:
        if ckpt.get('entity_embedding_state_dict') is None:
            raise RuntimeError(
                "Checkpoint does not contain entity_embedding_state_dict but "
                "--use_learnable_entity_embeddings=true."
            )
        entity_embedding_table.load_state_dict(ckpt['entity_embedding_state_dict'])
    if structural_seed_projector is not None:
        if ckpt.get('structural_seed_projector_state_dict') is None:
            raise RuntimeError(
                "Checkpoint does not contain structural_seed_projector_state_dict but "
                "--use_structural_seed_features=true."
            )
        structural_seed_projector.load_state_dict(ckpt['structural_seed_projector_state_dict'])
    if semantic_projector is not None:
        if ckpt.get('semantic_projector_state_dict') is None:
            raise RuntimeError(
                "Checkpoint does not contain semantic_projector_state_dict but "
                "--semantic_project_dim > 0."
            )
        semantic_projector.load_state_dict(ckpt['semantic_projector_state_dict'])
    if gcn_encoder is not None:
        if ckpt.get('gcn_state_dict') is None:
            raise RuntimeError("Checkpoint does not contain gcn_state_dict but --use_learnable_gcn=true.")
        gcn_encoder.load_state_dict(ckpt['gcn_state_dict'])
    if heuristic_fusion is not None:
        if ckpt.get('heuristic_fusion_state_dict') is None:
            raise RuntimeError("Checkpoint does not contain heuristic_fusion_state_dict but --use_heuristic_features=true.")
        heuristic_fusion.load_state_dict(ckpt['heuristic_fusion_state_dict'])
    if mplp_exact_fusion is not None:
        if ckpt.get('mplp_exact_fusion_state_dict') is None:
            raise RuntimeError("Checkpoint does not contain mplp_exact_fusion_state_dict but --use_mplp_exact_features=true.")
        mplp_exact_fusion.load_state_dict(ckpt['mplp_exact_fusion_state_dict'])

    print("\nEvaluating best checkpoint on test splits...")

    test_metrics = evaluate_split(
        model=model,
        data_loader=test_loader,
        data_source=test_data,
        neg_sampler=test_neg_sampler,
        embeddings=active_embeddings,
        lookup=lookup,
        num_negatives=args.eval_num_negatives,
        desc="Test-Transductive",
        rolling_provider=test_rolling_provider,
        gcn_encoder=gcn_encoder,
        learnable_mp_type=args.learnable_mp_type,
        mp_adj=static_mp_adj,
        mp_neighbor_index=mp_neighbor_index,
        mp_num_neighbors=args.attn_mp_num_neighbors,
        neighbor_index=scorer_neighbor_index,
        static_ncn_adj=static_ncn_adj,
        static_mplp_exact_adj2=static_mplp_exact_adj2,
        cross_attn_num_neighbors=args.cross_attn_num_neighbors,
        ncn_num_neighbors=args.ncn_num_neighbors,
        seqfilter_num_neighbors=args.seqfilter_num_neighbors,
        cross_attn_use_raw_embeddings=args.cross_attn_use_raw_embeddings,
        heuristic_extractor=heuristic_extractor,
        heuristic_fusion=heuristic_fusion,
        mplp_exact_fusion=mplp_exact_fusion,
        semantic_aux_fusion_mode=args.semantic_aux_fusion_mode,
        use_mplp_exact_input_features=(
            args.semantic_aux_fusion_mode == 'late_concat' and args.use_mplp_exact_features
        ),
        entity_embedding_table=entity_embedding_table,
        structural_seed_features=structural_seed_features,
        structural_seed_projector=structural_seed_projector,
        semantic_projector=semantic_projector,
        static_smoothing_adj=static_smoothing_adj,
        static_smoothing_steps=args.smooth_steps,
        precomputed_pos_raw_heuristic_features=test_pos_raw_heuristic_features,
        precomputed_neg_raw_heuristic_features=test_neg_raw_heuristic_features,
        dtgb_eval_batch_size=args.dtgb_eval_batch_size,
        time_bucket_count=args.eval_time_buckets,
    )

    new_node_metrics = evaluate_split(
        model=model,
        data_loader=new_node_test_loader,
        data_source=new_node_test_data,
        neg_sampler=new_node_test_neg_sampler,
        embeddings=active_embeddings,
        lookup=lookup,
        num_negatives=args.eval_num_negatives,
        desc="Test-Inductive",
        rolling_provider=new_node_test_rolling_provider,
        gcn_encoder=gcn_encoder,
        learnable_mp_type=args.learnable_mp_type,
        mp_adj=static_mp_adj,
        mp_neighbor_index=mp_neighbor_index,
        mp_num_neighbors=args.attn_mp_num_neighbors,
        neighbor_index=scorer_neighbor_index,
        static_ncn_adj=static_ncn_adj,
        static_mplp_exact_adj2=static_mplp_exact_adj2,
        cross_attn_num_neighbors=args.cross_attn_num_neighbors,
        ncn_num_neighbors=args.ncn_num_neighbors,
        seqfilter_num_neighbors=args.seqfilter_num_neighbors,
        cross_attn_use_raw_embeddings=args.cross_attn_use_raw_embeddings,
        heuristic_extractor=heuristic_extractor,
        heuristic_fusion=heuristic_fusion,
        mplp_exact_fusion=mplp_exact_fusion,
        semantic_aux_fusion_mode=args.semantic_aux_fusion_mode,
        use_mplp_exact_input_features=(
            args.semantic_aux_fusion_mode == 'late_concat' and args.use_mplp_exact_features
        ),
        entity_embedding_table=entity_embedding_table,
        structural_seed_features=structural_seed_features,
        structural_seed_projector=structural_seed_projector,
        semantic_projector=semantic_projector,
        static_smoothing_adj=static_smoothing_adj,
        static_smoothing_steps=args.smooth_steps,
        precomputed_pos_raw_heuristic_features=new_node_pos_raw_heuristic_features,
        precomputed_neg_raw_heuristic_features=new_node_neg_raw_heuristic_features,
        dtgb_eval_batch_size=args.dtgb_eval_batch_size,
        time_bucket_count=args.eval_time_buckets,
    )

    def _print_time_bucket_metrics(title: str, metrics: dict) -> None:
        bucket_metrics = metrics.get('time_bucket_metrics')
        if not bucket_metrics:
            return
        print(f"\n{title} Time Buckets")
        print("-" * 80)
        print(f"{'Bucket':<8}{'Time Range':<24}{'Queries':<10}{'AP':<10}{'AUC':<10}{'MRR':<10}")
        for bucket in bucket_metrics:
            time_range = f"[{bucket['time_min']:.0f}, {bucket['time_max']:.0f}]"
            print(
                f"{bucket['bucket_index']:<8}"
                f"{time_range:<24}"
                f"{bucket['num_queries']:<10}"
                f"{bucket['average_precision']:<10.4f}"
                f"{bucket['roc_auc']:<10.4f}"
                f"{bucket['mrr']:<10.4f}"
            )

    print("\n" + "=" * 80)
    print("FINAL RESULTS (Semantic Pair Scorer on Smoothed Embeddings)")
    print("=" * 80)
    print(f"{'Metric':<16}{'Transductive':<16}{'Inductive':<16}")
    print("-" * 48)
    print(f"{'AP':<16}{test_metrics['average_precision']:<16.4f}{new_node_metrics['average_precision']:<16.4f}")
    print(f"{'AUC':<16}{test_metrics['roc_auc']:<16.4f}{new_node_metrics['roc_auc']:<16.4f}")
    print(
        f"{'AP_GLOBAL':<16}"
        f"{test_metrics['average_precision_global']:<16.4f}"
        f"{new_node_metrics['average_precision_global']:<16.4f}"
    )
    print(
        f"{'AUC_GLOBAL':<16}"
        f"{test_metrics['roc_auc_global']:<16.4f}"
        f"{new_node_metrics['roc_auc_global']:<16.4f}"
    )
    print(f"{'MRR':<16}{test_metrics['mrr']:<16.4f}{new_node_metrics['mrr']:<16.4f}")
    print(
        f"DTGB metric aggregation: transductive={test_metrics['dtgb_metric_aggregation']} "
        f"(batch_size={test_metrics['dtgb_eval_batch_size']}), "
        f"inductive={new_node_metrics['dtgb_metric_aggregation']} "
        f"(batch_size={new_node_metrics['dtgb_eval_batch_size']})"
    )
    _print_time_bucket_metrics("Transductive", test_metrics)
    _print_time_bucket_metrics("Inductive", new_node_metrics)
    print("=" * 80)
    print("FINAL_METRICS_JSON " + json.dumps({
        'dataset': args.dataset_name,
        'seed': int(args.seed),
        'history_cap': int(args.smooth_endpoint_topk_recent),
        'auc_test': float(test_metrics['roc_auc']),
        'auc_new_node': float(new_node_metrics['roc_auc']),
        'ap_test': float(test_metrics['average_precision']),
        'ap_new_node': float(new_node_metrics['average_precision']),
        'aggregation': {
            'test': test_metrics['dtgb_metric_aggregation'],
            'new_node': new_node_metrics['dtgb_metric_aggregation'],
        },
        'dtgb_eval_batch_size': {
            'test': test_metrics['dtgb_eval_batch_size'],
            'new_node': new_node_metrics['dtgb_eval_batch_size'],
        },
    }, allow_nan=False, sort_keys=True), flush=True)

    wandb.log({
        'test/transductive_ap': test_metrics['average_precision'],
        'test/transductive_auc': test_metrics['roc_auc'],
        'test/transductive_ap_global': test_metrics['average_precision_global'],
        'test/transductive_auc_global': test_metrics['roc_auc_global'],
        'test/transductive_mrr': test_metrics['mrr'],
        'test/inductive_ap': new_node_metrics['average_precision'],
        'test/inductive_auc': new_node_metrics['roc_auc'],
        'test/inductive_ap_global': new_node_metrics['average_precision_global'],
        'test/inductive_auc_global': new_node_metrics['roc_auc_global'],
        'test/inductive_mrr': new_node_metrics['mrr'],
        'test/transductive_dtgb_eval_batch_size': test_metrics['dtgb_eval_batch_size'],
        'test/transductive_dtgb_num_metric_batches': test_metrics['dtgb_num_metric_batches'],
        'test/inductive_dtgb_eval_batch_size': new_node_metrics['dtgb_eval_batch_size'],
        'test/inductive_dtgb_num_metric_batches': new_node_metrics['dtgb_num_metric_batches'],
    })
    wandb.finish()


if __name__ == '__main__':
    main()
