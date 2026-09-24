from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

try:
    from torch_sparse import SparseTensor
except ImportError:  # pragma: no cover
    SparseTensor = None

from experiments.modules.prediction_metrics import compute_prediction_metrics
from utils.utils import NegativeEdgeSampler
from experiments.modules.semantic_mlp.components import (
    HEURISTIC_FEATURE_NAME_ALIASES,
    HAS_CUDA,
    SUPPORTED_HEURISTIC_FEATURE_NAMES,
    GINLayer,
    HeuristicFeatureExtractor,
    HeuristicFusionHead,
    LearnableGCNEncoder,
    LearnableGINEncoder,
    MPLPExactFusionHead,
    RollingSmoothedEmbeddingProvider,
    SimpleGCNConv,
    _build_neighbor_sparse,
    _elem2spm,
    _negative_precompute_debug_log,
    _profile_stage_elapsed,
    _profile_stage_start,
    _spm2elem,
    _spmdiff,
    _spmoverlap,
    apply_attention_pool_message_passing_for_nodes,
    apply_static_smoothing_operator,
    build_binary_history_adj,
    build_semantic_mlp_auxiliary_features,
    build_precomputed_negative_queries,
    build_precomputed_train_negative_pool,
    build_two_hop_binary_adj,
    compute_mplp_exact_features,
    fuse_pos_neg_logits_with_heuristics,
    fuse_pos_neg_logits_with_mplp_exact,
    materialize_base_embeddings,
    sample_negatives,
    sample_negatives_rand_hist_ratio,
    score_links_by_common_neighbors,
    score_links_by_global_recency,
    score_links_by_itemcf_cosine,
    score_links_by_past_interactions,
    score_links_by_popularity,
    score_links_by_recent_degree,
    score_links_by_recency,
    score_links_by_usercf_cosine,
    score_edge_batch,
    smooth_embeddings_by_time_window_torch,
    spmm_add,
)
from experiments.modules.semantic_mlp.ridge import (
    PairwiseRidgeAccumulator,
    SemanticRidgeScorer,
    build_ridge_edge_features,
)


def _unique_embedding_rows(node_ids: torch.Tensor, lookup: torch.Tensor) -> torch.Tensor:
    valid_ids = (node_ids >= 0) & (node_ids < lookup.numel())
    rows = lookup[node_ids[valid_ids]]
    return torch.unique(rows[rows >= 0])


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    train_loader,
    train_data,
    train_neg_sampler: NegativeEdgeSampler,
    embeddings: torch.Tensor,
    lookup: torch.Tensor,
    train_num_negatives: int,
    grad_clip_norm: float,
    label_smoothing: float,
    rolling_provider: Optional[RollingSmoothedEmbeddingProvider] = None,
    gcn_encoder: Optional[nn.Module] = None,
    learnable_mp_type: str = 'gcn',
    mp_adj: Optional[torch.Tensor] = None,
    mp_neighbor_index: Optional[TemporalNeighborIndex] = None,
    mp_num_neighbors: int = 50,
    neighbor_index: Optional[TemporalNeighborIndex] = None,
    static_ncn_adj: Optional[SparseTensor] = None,
    static_mplp_exact_adj2: Optional[SparseTensor] = None,
    cross_attn_num_neighbors: int = 20,
    ncn_num_neighbors: int = 50,
    seqfilter_num_neighbors: int = 32,
    cross_attn_use_raw_embeddings: bool = False,
    train_neg_sampler_random: Optional[NegativeEdgeSampler] = None,
    train_neg_sampler_historical: Optional[NegativeEdgeSampler] = None,
    train_rand_ratio: Optional[float] = None,
    historical_strict_gap: bool = False,
    heuristic_extractor: Optional[HeuristicFeatureExtractor] = None,
    heuristic_fusion: Optional[HeuristicFusionHead] = None,
    mplp_exact_fusion: Optional[MPLPExactFusionHead] = None,
    semantic_aux_fusion_mode: str = 'residual',
    use_mplp_exact_input_features: bool = False,
    entity_embedding_table: Optional[nn.Module] = None,
    structural_seed_features: Optional[torch.Tensor] = None,
    structural_seed_projector: Optional[nn.Module] = None,
    semantic_projector: Optional[nn.Module] = None,
    static_smoothing_adj: Optional[torch.Tensor] = None,
    static_smoothing_steps: int = 1,
    train_pos_raw_heuristic_features: Optional[np.ndarray] = None,
    train_neg_src_all: Optional[np.ndarray] = None,
    train_neg_dst_all: Optional[np.ndarray] = None,
    train_neg_raw_heuristic_features: Optional[np.ndarray] = None,
    profile_batches: int = 0,
    profile_skip_batches: int = 0,
    profile_out: Optional[Dict[str, float]] = None,
    profile_label: Optional[str] = None,
    profile_print_early: bool = False,
    torch_profiler=None,
) -> Tuple[float, float, int]:
    model.train()
    if gcn_encoder is not None:
        gcn_encoder.train()
    if heuristic_fusion is not None:
        heuristic_fusion.train()
    if mplp_exact_fusion is not None:
        mplp_exact_fusion.train()
    if structural_seed_projector is not None:
        structural_seed_projector.train()
    if semantic_projector is not None:
        semantic_projector.train()
    # Do NOT reset train negative sampler each epoch; this increases negative diversity over epochs.
    # (Validation/Test samplers are still reset for reproducible evaluation.)
    if rolling_provider is not None:
        rolling_provider.reset_history()

    total_loss = 0.0
    num_steps = 0
    profile_enabled = profile_batches > 0
    profile_seen = 0
    early_profile_printed = False
    prof_prepare = 0.0
    prof_neg = 0.0
    prof_mp = 0.0
    prof_heur = 0.0
    prof_heur_lookup = 0.0
    prof_heur_recency = 0.0
    prof_heur_popularity = 0.0
    prof_heur_past = 0.0
    prof_heur_ra = 0.0
    prof_heur_build = 0.0
    prof_heur_update = 0.0
    prof_heur_post = 0.0
    prof_heur_queries = 0.0
    prof_heur_misses = 0.0
    prof_mplp = 0.0
    prof_score = 0.0
    prof_score_id_to_tensor = 0.0
    prof_score_lookup = 0.0
    prof_score_neighbor_fetch = 0.0
    prof_score_neighbor_marshal = 0.0
    prof_score_gather = 0.0
    prof_score_forward = 0.0
    prof_score_valid_edges = 0.0
    prof_ncn_history_fetch = 0.0
    prof_ncn_marshal = 0.0
    prof_ncn_sparse_build = 0.0
    prof_ncn_row_slice = 0.0
    prof_ncn_overlap = 0.0
    prof_ncn_aggregate = 0.0
    prof_ncn_edges = 0.0
    prof_backward = 0.0
    prof_commit = 0.0
    prof_total = 0.0
    prof_zero_grad = 0.0
    prof_batch_unpack = 0.0
    prof_base_embed = 0.0
    prof_static_smooth = 0.0
    prof_scheduler = 0.0
    prof_label_assembly = 0.0
    prof_loss_compute = 0.0
    prof_backward_grad = 0.0
    prof_grad_clip = 0.0
    prof_opt_step = 0.0
    prof_loss_item = 0.0
    prof_bookkeeping = 0.0

    def _build_profile_summary_dict():
        if profile_seen > 0 and prof_total > 0:
            inv = 1000.0 / float(profile_seen)
            prof_other = prof_total - (
                prof_prepare + prof_neg + prof_mp + prof_heur + prof_mplp + prof_score + prof_backward + prof_commit
                + prof_zero_grad + prof_batch_unpack + prof_base_embed + prof_static_smooth
                + prof_scheduler + prof_label_assembly + prof_bookkeeping
            )
            return {
                'batches': float(profile_seen),
                'prepare_ms': prof_prepare * inv,
                'neg_ms': prof_neg * inv,
                'mp_ms': prof_mp * inv,
                'heur_ms': prof_heur * inv,
                'mplp_ms': prof_mplp * inv,
                'heur_lookup_ms': prof_heur_lookup * inv,
                'heur_recency_ms': prof_heur_recency * inv,
                'heur_popularity_ms': prof_heur_popularity * inv,
                'heur_past_ms': prof_heur_past * inv,
                'heur_ra_ms': prof_heur_ra * inv,
                'heur_build_ms': prof_heur_build * inv,
                'heur_update_ms': prof_heur_update * inv,
                'heur_post_ms': prof_heur_post * inv,
                'heur_queries': prof_heur_queries / float(profile_seen),
                'heur_misses': prof_heur_misses / float(profile_seen),
                'score_ms': prof_score * inv,
                'score_id_to_tensor_ms': prof_score_id_to_tensor * inv,
                'score_lookup_ms': prof_score_lookup * inv,
                'score_neighbor_fetch_ms': prof_score_neighbor_fetch * inv,
                'score_neighbor_marshal_ms': prof_score_neighbor_marshal * inv,
                'score_gather_ms': prof_score_gather * inv,
                'score_forward_ms': prof_score_forward * inv,
                'score_valid_edges': prof_score_valid_edges / float(profile_seen),
                'ncn_history_fetch_ms': prof_ncn_history_fetch * inv,
                'ncn_marshal_ms': prof_ncn_marshal * inv,
                'ncn_sparse_build_ms': prof_ncn_sparse_build * inv,
                'ncn_row_slice_ms': prof_ncn_row_slice * inv,
                'ncn_overlap_ms': prof_ncn_overlap * inv,
                'ncn_aggregate_ms': prof_ncn_aggregate * inv,
                'ncn_edges': prof_ncn_edges / float(profile_seen),
                'backward_ms': prof_backward * inv,
                'commit_ms': prof_commit * inv,
                'zero_grad_ms': prof_zero_grad * inv,
                'batch_unpack_ms': prof_batch_unpack * inv,
                'base_embed_ms': prof_base_embed * inv,
                'static_smooth_ms': prof_static_smooth * inv,
                'scheduler_ms': prof_scheduler * inv,
                'label_assembly_ms': prof_label_assembly * inv,
                'loss_compute_ms': prof_loss_compute * inv,
                'backward_grad_ms': prof_backward_grad * inv,
                'grad_clip_ms': prof_grad_clip * inv,
                'opt_step_ms': prof_opt_step * inv,
                'loss_item_ms': prof_loss_item * inv,
                'bookkeeping_ms': prof_bookkeeping * inv,
                'other_ms': prof_other * inv,
                'total_ms': prof_total * inv,
                'prepare_pct': 100.0 * prof_prepare / prof_total,
                'neg_pct': 100.0 * prof_neg / prof_total,
                'mp_pct': 100.0 * prof_mp / prof_total,
                'heur_pct': 100.0 * prof_heur / prof_total,
                'mplp_pct': 100.0 * prof_mplp / prof_total,
                'score_pct': 100.0 * prof_score / prof_total,
                'backward_pct': 100.0 * prof_backward / prof_total,
                'commit_pct': 100.0 * prof_commit / prof_total,
                'other_pct': 100.0 * prof_other / prof_total,
            }
        return {
            'batches': 0.0,
            'prepare_ms': 0.0,
            'neg_ms': 0.0,
            'mp_ms': 0.0,
            'heur_ms': 0.0,
            'mplp_ms': 0.0,
            'heur_lookup_ms': 0.0,
            'heur_recency_ms': 0.0,
            'heur_popularity_ms': 0.0,
            'heur_past_ms': 0.0,
            'heur_ra_ms': 0.0,
            'heur_build_ms': 0.0,
            'heur_update_ms': 0.0,
            'heur_post_ms': 0.0,
            'heur_queries': 0.0,
            'heur_misses': 0.0,
            'score_ms': 0.0,
            'ncn_history_fetch_ms': 0.0,
            'ncn_marshal_ms': 0.0,
            'ncn_sparse_build_ms': 0.0,
            'ncn_row_slice_ms': 0.0,
            'ncn_overlap_ms': 0.0,
            'ncn_aggregate_ms': 0.0,
            'ncn_edges': 0.0,
            'backward_ms': 0.0,
            'commit_ms': 0.0,
            'zero_grad_ms': 0.0,
            'batch_unpack_ms': 0.0,
            'base_embed_ms': 0.0,
            'static_smooth_ms': 0.0,
            'scheduler_ms': 0.0,
            'label_assembly_ms': 0.0,
            'loss_compute_ms': 0.0,
            'backward_grad_ms': 0.0,
            'grad_clip_ms': 0.0,
            'opt_step_ms': 0.0,
            'loss_item_ms': 0.0,
            'bookkeeping_ms': 0.0,
            'other_ms': 0.0,
            'total_ms': 0.0,
            'prepare_pct': 0.0,
            'neg_pct': 0.0,
            'mp_pct': 0.0,
            'heur_pct': 0.0,
            'mplp_pct': 0.0,
            'score_pct': 0.0,
            'backward_pct': 0.0,
            'commit_pct': 0.0,
            'other_pct': 0.0,
        }

    pos_label_value = 1.0 - label_smoothing
    neg_label_value = label_smoothing

    for batch_id, batch_indices in enumerate(tqdm(train_loader, desc="Train", ncols=100)):
        do_profile = profile_enabled and (batch_id >= profile_skip_batches) and (profile_seen < profile_batches)
        batch_t0 = time.perf_counter() if do_profile else None

        t_start = time.perf_counter() if do_profile else None
        optimizer.zero_grad(set_to_none=True)
        if do_profile:
            prof_zero_grad += time.perf_counter() - t_start

        t_start = time.perf_counter() if do_profile else None
        batch_indices = batch_indices.numpy()
        src = train_data.src_node_ids[batch_indices]
        dst = train_data.dst_node_ids[batch_indices]
        times = train_data.node_interact_times[batch_indices]
        if do_profile:
            prof_batch_unpack += time.perf_counter() - t_start

        t_start = time.perf_counter() if do_profile else None
        current_base_embeddings = materialize_base_embeddings(
            embeddings=embeddings,
            entity_embedding_table=entity_embedding_table,
            structural_seed_features=structural_seed_features,
            structural_seed_projector=structural_seed_projector,
        )
        if semantic_projector is not None:
            current_base_embeddings = F.normalize(
                semantic_projector(current_base_embeddings),
                dim=1,
            )
        if do_profile:
            prof_base_embed += time.perf_counter() - t_start

        if rolling_provider is not None:
            t_start = time.perf_counter() if do_profile else None
            rolling_provider.set_base_embeddings(current_base_embeddings)
            rolling_provider.prepare_batch(times)
            if do_profile:
                prof_prepare += time.perf_counter() - t_start
            # For learnable message passing, replace heuristic smoothing as feature source.
            if gcn_encoder is not None:
                current_embeddings = rolling_provider.current_base_embeddings
            else:
                current_embeddings = rolling_provider.current_embeddings
        else:
            current_embeddings = current_base_embeddings
            if static_smoothing_adj is not None:
                t_start = time.perf_counter() if do_profile else None
                current_embeddings = apply_static_smoothing_operator(
                    embeddings=current_embeddings,
                    norm_adj=static_smoothing_adj,
                    num_steps=static_smoothing_steps,
                )
                if do_profile:
                    prof_static_smooth += time.perf_counter() - t_start

        neg_index = None
        if train_neg_src_all is not None and train_neg_dst_all is not None:
            t_start = time.perf_counter() if do_profile else None
            neg_index = (
                batch_indices[:, None] * train_num_negatives
                + np.arange(train_num_negatives, dtype=np.int64)[None, :]
            ).reshape(-1)
            neg_src = train_neg_src_all[neg_index]
            neg_dst = train_neg_dst_all[neg_index]
            if do_profile:
                prof_neg += time.perf_counter() - t_start
        elif train_rand_ratio is None:
            t_start = time.perf_counter() if do_profile else None
            neg_src, neg_dst = sample_negatives(
                neg_sampler=train_neg_sampler,
                src=src,
                dst=dst,
                times=times,
                num_negatives=train_num_negatives,
            )
            if do_profile:
                prof_neg += time.perf_counter() - t_start
        else:
            if train_neg_sampler_random is None or train_neg_sampler_historical is None:
                raise ValueError("Mixed sampling requested but mixed samplers are not initialized.")
            t_start = time.perf_counter() if do_profile else None
            neg_src, neg_dst = sample_negatives_rand_hist_ratio(
                neg_sampler_random=train_neg_sampler_random,
                neg_sampler_historical=train_neg_sampler_historical,
                src=src,
                dst=dst,
                times=times,
                num_negatives=train_num_negatives,
                rand_ratio=train_rand_ratio,
                strict_historical_gap=historical_strict_gap,
            )
            if do_profile:
                prof_neg += time.perf_counter() - t_start

        t_start = _profile_stage_start(embeddings.device, do_profile)
        packed_ids = np.concatenate([src, dst, neg_src, neg_dst], axis=0)
        packed_ids_t = torch.as_tensor(packed_ids, dtype=torch.long, device=embeddings.device)
        src_end = len(src)
        dst_end = src_end + len(dst)
        neg_src_end = dst_end + len(neg_src)
        src_t = packed_ids_t[:src_end]
        dst_t = packed_ids_t[src_end:dst_end]
        neg_src_t = packed_ids_t[dst_end:neg_src_end]
        neg_dst_t = packed_ids_t[neg_src_end:]
        if do_profile:
            prof_score_id_to_tensor += _profile_stage_elapsed(t_start, embeddings.device)

        if gcn_encoder is not None:
            t_start = time.perf_counter() if do_profile else None
            if learnable_mp_type == 'gcn':
                current_mp_adj = mp_adj
                if rolling_provider is not None and rolling_provider.current_norm_adj is not None:
                    current_mp_adj = rolling_provider.current_norm_adj
                if current_mp_adj is None:
                    raise ValueError("gcn_encoder is set but no smoothing-aligned norm adjacency is available.")
                temporal_relational_context = (
                    rolling_provider.current_temporal_relational_context
                    if rolling_provider is not None
                    else None
                )
                current_embeddings = gcn_encoder(
                    current_embeddings,
                    current_mp_adj,
                    temporal_relational_context=temporal_relational_context,
                    temporal_relational_query_rows=(
                        _unique_embedding_rows(packed_ids_t, lookup)
                        if getattr(gcn_encoder, "use_temporal_relational", False)
                        else None
                    ),
                )
                current_embeddings = F.normalize(current_embeddings, dim=1)
            elif learnable_mp_type == 'gin':
                current_mp_adj = mp_adj
                if rolling_provider is not None and rolling_provider.current_sum_adj is not None:
                    current_mp_adj = rolling_provider.current_sum_adj
                if current_mp_adj is None:
                    raise ValueError("gcn_encoder is set but no smoothing-aligned sum adjacency is available.")
                current_embeddings = gcn_encoder(current_embeddings, current_mp_adj)
                current_embeddings = F.normalize(current_embeddings, dim=1)
            elif learnable_mp_type == 'attn_pool':
                if mp_neighbor_index is None:
                    raise ValueError("attn_pool message passing requires mp_neighbor_index.")
                query_time = float(np.min(times))
                query_nodes = np.concatenate([src, dst, neg_src, neg_dst], axis=0)
                current_embeddings = apply_attention_pool_message_passing_for_nodes(
                    mp_encoder=gcn_encoder,
                    embeddings=current_embeddings,
                    lookup=lookup,
                    query_node_ids=query_nodes,
                    query_time=query_time,
                    neighbor_index=mp_neighbor_index,
                    num_neighbors=mp_num_neighbors,
                )
            else:
                raise ValueError(f"Unsupported learnable_mp_type: {learnable_mp_type}")
            if do_profile:
                prof_mp += time.perf_counter() - t_start

        scoring_embeddings = current_embeddings
        if cross_attn_use_raw_embeddings and getattr(model, 'requires_neighbor_context', False):
            scoring_embeddings = current_base_embeddings

        batch_ncn_adj = static_ncn_adj
        batch_mplp_exact_adj2 = static_mplp_exact_adj2
        if rolling_provider is not None:
            batch_ncn_adj = rolling_provider.current_binary_adj
            batch_mplp_exact_adj2 = rolling_provider.current_binary_two_hop_adj

        use_mlp_input_aux = semantic_aux_fusion_mode == 'late_concat'
        combined_aux_features = None
        if use_mlp_input_aux:
            combined_src = np.concatenate([src, neg_src], axis=0)
            combined_dst = np.concatenate([dst, neg_dst], axis=0)
            combined_times = np.concatenate([times, np.repeat(times, train_num_negatives)], axis=0)
            combined_raw_heuristic_features = None
            if (
                train_pos_raw_heuristic_features is not None
                and train_neg_raw_heuristic_features is not None
                and neg_index is not None
            ):
                combined_raw_heuristic_features = np.concatenate(
                    [
                        train_pos_raw_heuristic_features[batch_indices],
                        train_neg_raw_heuristic_features[neg_index],
                    ],
                    axis=0,
                )
            combined_aux_features = build_semantic_mlp_auxiliary_features(
                heuristic_extractor=heuristic_extractor,
                lookup=lookup,
                sources=combined_src,
                targets=combined_dst,
                prediction_times=combined_times,
                raw_heuristic_features=combined_raw_heuristic_features,
                ncn_adj=batch_ncn_adj if use_mplp_exact_input_features else None,
                two_hop_adj=batch_mplp_exact_adj2 if use_mplp_exact_input_features else None,
                output_device=embeddings.device,
                output_dtype=scoring_embeddings.dtype,
                normalize_heuristics=not getattr(model, 'expects_raw_auxiliary_features', False),
            )

        t_start = _profile_stage_start(embeddings.device, do_profile)
        score_profile = {} if do_profile else None
        pos_logits, pos_valid = score_edge_batch(
            model,
            scoring_embeddings,
            lookup,
            src_t,
            dst_t,
            auxiliary_features=(
                None if combined_aux_features is None else combined_aux_features[:len(src)]
            ),
            node_interact_times=times,
            neighbor_index=neighbor_index,
            ncn_adj=batch_ncn_adj,
            cross_attn_num_neighbors=cross_attn_num_neighbors,
            ncn_num_neighbors=ncn_num_neighbors,
            seqfilter_num_neighbors=seqfilter_num_neighbors,
            profile_out=score_profile,
        )
        neg_times = np.repeat(times, train_num_negatives)
        neg_logits, neg_valid = score_edge_batch(
            model,
            scoring_embeddings,
            lookup,
            neg_src_t,
            neg_dst_t,
            auxiliary_features=(
                None if combined_aux_features is None else combined_aux_features[len(src):]
            ),
            node_interact_times=neg_times,
            neighbor_index=neighbor_index,
            ncn_adj=batch_ncn_adj,
            cross_attn_num_neighbors=cross_attn_num_neighbors,
            ncn_num_neighbors=ncn_num_neighbors,
            seqfilter_num_neighbors=seqfilter_num_neighbors,
            profile_out=score_profile,
        )
        if do_profile:
            prof_score += _profile_stage_elapsed(t_start, embeddings.device)
            prof_score_id_to_tensor += score_profile.get('score_id_to_tensor_s', 0.0)
            prof_score_lookup += score_profile.get('score_lookup_s', 0.0)
            prof_score_neighbor_fetch += score_profile.get('score_neighbor_fetch_s', 0.0)
            prof_score_neighbor_marshal += score_profile.get('score_neighbor_marshal_s', 0.0)
            prof_score_gather += score_profile.get('score_gather_s', 0.0)
            prof_score_forward += score_profile.get('score_forward_s', 0.0)
            prof_score_valid_edges += score_profile.get('score_valid_edges', 0.0)
            prof_ncn_history_fetch += score_profile.get('ncn_history_fetch_s', 0.0)
            prof_ncn_marshal += score_profile.get('ncn_marshal_s', 0.0)
            prof_ncn_sparse_build += score_profile.get('ncn_sparse_build_s', 0.0)
            prof_ncn_row_slice += score_profile.get('ncn_row_slice_s', 0.0)
            prof_ncn_overlap += score_profile.get('ncn_overlap_s', 0.0)
            prof_ncn_aggregate += score_profile.get('ncn_aggregate_s', 0.0)
            prof_ncn_edges += score_profile.get('ncn_edge_count', 0.0)

        if not use_mlp_input_aux:
            t_start = time.perf_counter() if do_profile else None
            pos_logits, neg_logits, mplp_elapsed = fuse_pos_neg_logits_with_mplp_exact(
                mplp_exact_fusion=mplp_exact_fusion,
                lookup=lookup,
                pos_logits=pos_logits,
                neg_logits=neg_logits,
                pos_src=src,
                pos_dst=dst,
                neg_src=neg_src,
                neg_dst=neg_dst,
                ncn_adj=batch_ncn_adj,
                two_hop_adj=batch_mplp_exact_adj2,
            )
            if do_profile:
                prof_mplp += time.perf_counter() - t_start

            heur_profile = {} if do_profile else None
            pos_logits, neg_logits, heuristic_elapsed = fuse_pos_neg_logits_with_heuristics(
                heuristic_extractor=heuristic_extractor,
                heuristic_fusion=heuristic_fusion,
                pos_logits=pos_logits,
                neg_logits=neg_logits,
                pos_src=src,
                pos_dst=dst,
                pos_times=times,
                neg_src=neg_src,
                neg_dst=neg_dst,
                neg_times=neg_times,
                pos_raw_features=(
                    train_pos_raw_heuristic_features[batch_indices]
                    if train_pos_raw_heuristic_features is not None
                    else None
                ),
                neg_raw_features=(
                    train_neg_raw_heuristic_features[neg_index]
                    if train_neg_raw_heuristic_features is not None
                    else None
                ),
                profile_out=heur_profile,
            )
            if do_profile:
                prof_heur += heuristic_elapsed
                prof_heur_lookup += heur_profile.get('lookup_s', 0.0)
                prof_heur_recency += heur_profile.get('recency_s', 0.0)
                prof_heur_popularity += heur_profile.get('popularity_s', 0.0)
                prof_heur_past += heur_profile.get('past_s', 0.0)
                prof_heur_ra += heur_profile.get('ra_s', 0.0)
                prof_heur_build += heur_profile.get('build_s', 0.0)
                prof_heur_update += heur_profile.get('update_s', 0.0)
                prof_heur_post += heur_profile.get('post_s', 0.0)
                prof_heur_queries += heur_profile.get('query_count', 0.0)
                prof_heur_misses += heur_profile.get('miss_count', 0.0)

        if not pos_valid.any() and not neg_valid.any():
            if do_profile:
                prof_total += time.perf_counter() - batch_t0
                profile_seen += 1
            if torch_profiler is not None:
                torch_profiler.step()
            continue

        t_start = time.perf_counter() if do_profile else None
        train_scores = torch.cat([pos_logits[pos_valid], neg_logits[neg_valid]], dim=0)
        train_labels = torch.cat([
            torch.full((int(pos_valid.sum().item()),), pos_label_value, device=embeddings.device),
            torch.full((int(neg_valid.sum().item()),), neg_label_value, device=embeddings.device),
        ], dim=0)
        if do_profile:
            prof_label_assembly += time.perf_counter() - t_start

        if train_scores.numel() == 0:
            if do_profile:
                prof_total += time.perf_counter() - batch_t0
                profile_seen += 1
            if torch_profiler is not None:
                torch_profiler.step()
            continue

        loss_compute_elapsed = 0.0
        backward_grad_elapsed = 0.0
        grad_clip_elapsed = 0.0
        opt_step_elapsed = 0.0

        t_start = _profile_stage_start(embeddings.device, do_profile)
        loss = criterion(train_scores, train_labels)
        if do_profile:
            loss_compute_elapsed = _profile_stage_elapsed(t_start, embeddings.device)
            prof_loss_compute += loss_compute_elapsed

        t_start = _profile_stage_start(embeddings.device, do_profile)
        loss.backward()
        if do_profile:
            backward_grad_elapsed = _profile_stage_elapsed(t_start, embeddings.device)
            prof_backward_grad += backward_grad_elapsed

        if grad_clip_norm > 0:
            clip_params = list(model.parameters())
            if entity_embedding_table is not None:
                clip_params.extend(entity_embedding_table.parameters())
            if structural_seed_projector is not None:
                clip_params.extend(structural_seed_projector.parameters())
            if semantic_projector is not None:
                clip_params.extend(semantic_projector.parameters())
            if gcn_encoder is not None:
                clip_params.extend(gcn_encoder.parameters())
            if heuristic_fusion is not None:
                clip_params.extend(heuristic_fusion.parameters())
            if mplp_exact_fusion is not None:
                clip_params.extend(mplp_exact_fusion.parameters())
            t_start = _profile_stage_start(embeddings.device, do_profile)
            torch.nn.utils.clip_grad_norm_(clip_params, grad_clip_norm)
            if do_profile:
                grad_clip_elapsed = _profile_stage_elapsed(t_start, embeddings.device)
                prof_grad_clip += grad_clip_elapsed
        t_start = _profile_stage_start(embeddings.device, do_profile)
        optimizer.step()
        if do_profile:
            opt_step_elapsed = _profile_stage_elapsed(t_start, embeddings.device)
            prof_opt_step += opt_step_elapsed
            prof_backward += loss_compute_elapsed + backward_grad_elapsed + grad_clip_elapsed + opt_step_elapsed

        if scheduler is not None:
            t_start = time.perf_counter() if do_profile else None
            scheduler.step()
            if do_profile:
                prof_scheduler += time.perf_counter() - t_start

        if rolling_provider is not None:
            t_start = time.perf_counter() if do_profile else None
            rolling_provider.commit_batch(src, dst, times)
            if do_profile:
                prof_commit += time.perf_counter() - t_start

        t_start = _profile_stage_start(embeddings.device, do_profile)
        loss_item_value = float(loss.item())
        if do_profile:
            prof_loss_item += _profile_stage_elapsed(t_start, embeddings.device)

        t_start = time.perf_counter() if do_profile else None
        total_loss += loss_item_value
        num_steps += 1
        if do_profile:
            prof_bookkeeping += time.perf_counter() - t_start
        if do_profile:
            prof_total += time.perf_counter() - batch_t0
            profile_seen += 1
            if (
                profile_print_early
                and not early_profile_printed
                and profile_seen >= profile_batches
            ):
                early_profile = _build_profile_summary_dict()
                label = str(profile_label or "train")
                print(
                    f"[EarlyProfile {label}] "
                    f"batches={int(early_profile['batches'])} | "
                    f"prep={early_profile['prepare_ms']:.1f}ms ({early_profile['prepare_pct']:.1f}%) | "
                    f"neg={early_profile['neg_ms']:.1f}ms ({early_profile['neg_pct']:.1f}%) | "
                    f"mp={early_profile['mp_ms']:.1f}ms ({early_profile['mp_pct']:.1f}%) | "
                    f"heur={early_profile['heur_ms']:.1f}ms ({early_profile['heur_pct']:.1f}%) | "
                    f"mplp={early_profile['mplp_ms']:.1f}ms ({early_profile['mplp_pct']:.1f}%) | "
                    f"score={early_profile['score_ms']:.1f}ms ({early_profile['score_pct']:.1f}%) | "
                    f"back={early_profile['backward_ms']:.1f}ms ({early_profile['backward_pct']:.1f}%) | "
                    f"commit={early_profile['commit_ms']:.1f}ms ({early_profile['commit_pct']:.1f}%) | "
                    f"other={early_profile['other_ms']:.1f}ms ({early_profile['other_pct']:.1f}%) | "
                    f"total={early_profile['total_ms']:.1f}ms"
                )
                early_profile_printed = True
        if torch_profiler is not None:
            torch_profiler.step()

    lr = float(optimizer.param_groups[0]['lr'])
    if profile_out is not None:
        profile_out.clear()
        profile_out.update(_build_profile_summary_dict())
    return total_loss / max(1, num_steps), lr, num_steps


@torch.inference_mode()
def fit_pairwise_ridge(
    *,
    model: SemanticRidgeScorer,
    train_loader,
    train_data,
    embeddings: torch.Tensor,
    lookup: torch.Tensor,
    train_num_negatives: int,
    train_neg_src_all: np.ndarray,
    train_neg_dst_all: np.ndarray,
    lambda_value: float,
    rolling_provider: Optional[RollingSmoothedEmbeddingProvider] = None,
    static_ncn_adj: Optional[SparseTensor] = None,
    static_mplp_exact_adj2: Optional[SparseTensor] = None,
    heuristic_extractor: Optional[HeuristicFeatureExtractor] = None,
    use_mplp_exact_features: bool = False,
    static_smoothing_adj: Optional[torch.Tensor] = None,
    static_smoothing_steps: int = 1,
    train_pos_raw_heuristic_features: Optional[np.ndarray] = None,
    train_neg_raw_heuristic_features: Optional[np.ndarray] = None,
):
    """Fit a pairwise ridge head in one chronological, leakage-safe pass."""

    if train_num_negatives < 1:
        raise ValueError("train_num_negatives must be >= 1.")
    expected_negatives = len(train_data.src_node_ids) * int(train_num_negatives)
    if len(train_neg_src_all) != expected_negatives or len(train_neg_dst_all) != expected_negatives:
        raise ValueError(
            "Precomputed ridge negative pool has the wrong size: "
            f"expected {expected_negatives}, got "
            f"src={len(train_neg_src_all)}, dst={len(train_neg_dst_all)}."
        )

    model.eval()
    accumulator = PairwiseRidgeAccumulator(model.feature_dim)
    if rolling_provider is not None:
        rolling_provider.reset_history()

    for batch_indices_t in tqdm(train_loader, desc="Fit ridge", ncols=100):
        batch_indices = batch_indices_t.numpy()
        src = train_data.src_node_ids[batch_indices]
        dst = train_data.dst_node_ids[batch_indices]
        times = train_data.node_interact_times[batch_indices]

        if rolling_provider is not None:
            rolling_provider.set_base_embeddings(embeddings)
            rolling_provider.prepare_batch(times)
            current_embeddings = rolling_provider.current_embeddings
            batch_ncn_adj = rolling_provider.current_binary_adj
            batch_mplp_exact_adj2 = rolling_provider.current_binary_two_hop_adj
        else:
            current_embeddings = embeddings
            if static_smoothing_adj is not None:
                current_embeddings = apply_static_smoothing_operator(
                    embeddings=current_embeddings,
                    norm_adj=static_smoothing_adj,
                    num_steps=static_smoothing_steps,
                )
            batch_ncn_adj = static_ncn_adj
            batch_mplp_exact_adj2 = static_mplp_exact_adj2

        neg_index = (
            batch_indices[:, None] * int(train_num_negatives)
            + np.arange(train_num_negatives, dtype=np.int64)[None, :]
        ).reshape(-1)
        neg_src = train_neg_src_all[neg_index]
        neg_dst = train_neg_dst_all[neg_index]
        neg_times = np.repeat(times, train_num_negatives)

        combined_aux_features = None
        if model.auxiliary_dim > 0:
            combined_raw_features = None
            if (
                train_pos_raw_heuristic_features is not None
                and train_neg_raw_heuristic_features is not None
            ):
                combined_raw_features = np.concatenate(
                    [
                        train_pos_raw_heuristic_features[batch_indices],
                        train_neg_raw_heuristic_features[neg_index],
                    ],
                    axis=0,
                )
            combined_aux_features = build_semantic_mlp_auxiliary_features(
                heuristic_extractor=heuristic_extractor,
                lookup=lookup,
                sources=np.concatenate([src, neg_src], axis=0),
                targets=np.concatenate([dst, neg_dst], axis=0),
                prediction_times=np.concatenate([times, neg_times], axis=0),
                raw_heuristic_features=combined_raw_features,
                ncn_adj=batch_ncn_adj if use_mplp_exact_features else None,
                two_hop_adj=batch_mplp_exact_adj2 if use_mplp_exact_features else None,
                output_device=embeddings.device,
                output_dtype=current_embeddings.dtype,
                normalize_heuristics=False,
            )

        pos_aux = None if combined_aux_features is None else combined_aux_features[:len(src)]
        neg_aux = None if combined_aux_features is None else combined_aux_features[len(src):]
        pos_features, pos_valid = build_ridge_edge_features(
            model=model,
            embeddings=current_embeddings,
            lookup=lookup,
            src_ids=src,
            dst_ids=dst,
            auxiliary_features=pos_aux,
        )
        neg_features, neg_valid = build_ridge_edge_features(
            model=model,
            embeddings=current_embeddings,
            lookup=lookup,
            src_ids=neg_src,
            dst_ids=neg_dst,
            auxiliary_features=neg_aux,
        )
        pair_valid = pos_valid.repeat_interleave(train_num_negatives) & neg_valid
        if pair_valid.any():
            differences = (
                pos_features.repeat_interleave(train_num_negatives, dim=0)
                - neg_features
            )
            accumulator.update(differences[pair_valid])

        if rolling_provider is not None:
            rolling_provider.commit_batch(src, dst, times)

    coefficient, feature_scale, stats = accumulator.solve(lambda_value)
    model.set_solution(coefficient, feature_scale)
    return stats


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    data_loader,
    data_source,
    neg_sampler: NegativeEdgeSampler,
    embeddings: torch.Tensor,
    lookup: torch.Tensor,
    num_negatives: int,
    desc: str,
    rolling_provider: Optional[RollingSmoothedEmbeddingProvider] = None,
    gcn_encoder: Optional[nn.Module] = None,
    learnable_mp_type: str = 'gcn',
    mp_adj: Optional[torch.Tensor] = None,
    mp_neighbor_index: Optional[TemporalNeighborIndex] = None,
    mp_num_neighbors: int = 50,
    neighbor_index: Optional[TemporalNeighborIndex] = None,
    static_ncn_adj: Optional[SparseTensor] = None,
    static_mplp_exact_adj2: Optional[SparseTensor] = None,
    cross_attn_num_neighbors: int = 20,
    ncn_num_neighbors: int = 50,
    seqfilter_num_neighbors: int = 32,
    cross_attn_use_raw_embeddings: bool = False,
    heuristic_extractor: Optional[HeuristicFeatureExtractor] = None,
    heuristic_fusion: Optional[HeuristicFusionHead] = None,
    mplp_exact_fusion: Optional[MPLPExactFusionHead] = None,
    semantic_aux_fusion_mode: str = 'residual',
    use_mplp_exact_input_features: bool = False,
    entity_embedding_table: Optional[nn.Module] = None,
    structural_seed_features: Optional[torch.Tensor] = None,
    structural_seed_projector: Optional[nn.Module] = None,
    semantic_projector: Optional[nn.Module] = None,
    static_smoothing_adj: Optional[torch.Tensor] = None,
    static_smoothing_steps: int = 1,
    precomputed_pos_raw_heuristic_features: Optional[np.ndarray] = None,
    precomputed_neg_raw_heuristic_features: Optional[np.ndarray] = None,
    dtgb_eval_batch_size: Optional[int] = None,
    time_bucket_count: int = 0,
    profile_batches: int = 0,
    profile_skip_batches: int = 0,
    profile_out: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    model.eval()
    if gcn_encoder is not None:
        gcn_encoder.eval()
    if heuristic_fusion is not None:
        heuristic_fusion.eval()
    if mplp_exact_fusion is not None:
        mplp_exact_fusion.eval()
    if structural_seed_projector is not None:
        structural_seed_projector.eval()
    if semantic_projector is not None:
        semantic_projector.eval()
    neg_sampler.reset_random_state()
    if rolling_provider is not None:
        rolling_provider.reset_history()

    all_scores = []
    all_labels = []
    mrr_list = []
    bucket_times = [] if time_bucket_count and time_bucket_count > 0 else None
    bucket_pos_scores = [] if time_bucket_count and time_bucket_count > 0 else None
    bucket_neg_scores = [] if time_bucket_count and time_bucket_count > 0 else None
    bucket_rr = [] if time_bucket_count and time_bucket_count > 0 else None
    profile_enabled = profile_batches > 0
    profile_seen = 0
    prof_prepare = 0.0
    prof_neg = 0.0
    prof_mp = 0.0
    prof_heur = 0.0
    prof_heur_lookup = 0.0
    prof_heur_recency = 0.0
    prof_heur_popularity = 0.0
    prof_heur_past = 0.0
    prof_heur_ra = 0.0
    prof_heur_build = 0.0
    prof_heur_update = 0.0
    prof_heur_post = 0.0
    prof_heur_queries = 0.0
    prof_heur_misses = 0.0
    prof_mplp = 0.0
    prof_score = 0.0
    prof_score_id_to_tensor = 0.0
    prof_score_lookup = 0.0
    prof_score_neighbor_fetch = 0.0
    prof_score_neighbor_marshal = 0.0
    prof_score_gather = 0.0
    prof_score_forward = 0.0
    prof_score_valid_edges = 0.0
    prof_ncn_history_fetch = 0.0
    prof_ncn_marshal = 0.0
    prof_ncn_sparse_build = 0.0
    prof_ncn_row_slice = 0.0
    prof_ncn_overlap = 0.0
    prof_ncn_aggregate = 0.0
    prof_ncn_edges = 0.0
    prof_commit = 0.0
    prof_total = 0.0

    for batch_id, batch_indices in enumerate(tqdm(data_loader, desc=desc, ncols=100)):
        do_profile = profile_enabled and (batch_id >= profile_skip_batches) and (profile_seen < profile_batches)
        batch_t0 = time.perf_counter() if do_profile else None

        batch_indices = batch_indices.numpy()
        src = data_source.src_node_ids[batch_indices]
        dst = data_source.dst_node_ids[batch_indices]
        times = data_source.node_interact_times[batch_indices]
        current_base_embeddings = materialize_base_embeddings(
            embeddings=embeddings,
            entity_embedding_table=entity_embedding_table,
            structural_seed_features=structural_seed_features,
            structural_seed_projector=structural_seed_projector,
        )
        if semantic_projector is not None:
            current_base_embeddings = F.normalize(
                semantic_projector(current_base_embeddings),
                dim=1,
            )

        if rolling_provider is not None:
            t_start = time.perf_counter() if do_profile else None
            rolling_provider.set_base_embeddings(current_base_embeddings)
            rolling_provider.prepare_batch(times)
            if do_profile:
                prof_prepare += time.perf_counter() - t_start
            # For learnable message passing, replace heuristic smoothing as feature source.
            if gcn_encoder is not None:
                current_embeddings = rolling_provider.current_base_embeddings
            else:
                current_embeddings = rolling_provider.current_embeddings
        else:
            current_embeddings = current_base_embeddings
            if static_smoothing_adj is not None:
                current_embeddings = apply_static_smoothing_operator(
                    embeddings=current_embeddings,
                    norm_adj=static_smoothing_adj,
                    num_steps=static_smoothing_steps,
                )

        batch_size = len(src)
        t_start = time.perf_counter() if do_profile else None
        neg_src, neg_dst = sample_negatives(
            neg_sampler=neg_sampler,
            src=src,
            dst=dst,
            times=times,
            num_negatives=num_negatives,
        )
        if do_profile:
            prof_neg += time.perf_counter() - t_start
        t_start = _profile_stage_start(embeddings.device, do_profile)
        packed_ids = np.concatenate([src, dst, neg_src, neg_dst], axis=0)
        packed_ids_t = torch.as_tensor(packed_ids, dtype=torch.long, device=embeddings.device)
        src_end = len(src)
        dst_end = src_end + len(dst)
        neg_src_end = dst_end + len(neg_src)
        src_t = packed_ids_t[:src_end]
        dst_t = packed_ids_t[src_end:dst_end]
        neg_src_t = packed_ids_t[dst_end:neg_src_end]
        neg_dst_t = packed_ids_t[neg_src_end:]
        if do_profile:
            prof_score_id_to_tensor += _profile_stage_elapsed(t_start, embeddings.device)

        if gcn_encoder is not None:
            t_start = time.perf_counter() if do_profile else None
            if learnable_mp_type == 'gcn':
                current_mp_adj = mp_adj
                if rolling_provider is not None and rolling_provider.current_norm_adj is not None:
                    current_mp_adj = rolling_provider.current_norm_adj
                if current_mp_adj is None:
                    raise ValueError("gcn_encoder is set but no smoothing-aligned norm adjacency is available.")
                temporal_relational_context = (
                    rolling_provider.current_temporal_relational_context
                    if rolling_provider is not None
                    else None
                )
                current_embeddings = gcn_encoder(
                    current_embeddings,
                    current_mp_adj,
                    temporal_relational_context=temporal_relational_context,
                    temporal_relational_query_rows=(
                        _unique_embedding_rows(packed_ids_t, lookup)
                        if getattr(gcn_encoder, "use_temporal_relational", False)
                        else None
                    ),
                )
                current_embeddings = F.normalize(current_embeddings, dim=1)
            elif learnable_mp_type == 'gin':
                current_mp_adj = mp_adj
                if rolling_provider is not None and rolling_provider.current_sum_adj is not None:
                    current_mp_adj = rolling_provider.current_sum_adj
                if current_mp_adj is None:
                    raise ValueError("gcn_encoder is set but no smoothing-aligned sum adjacency is available.")
                current_embeddings = gcn_encoder(current_embeddings, current_mp_adj)
                current_embeddings = F.normalize(current_embeddings, dim=1)
            elif learnable_mp_type == 'attn_pool':
                if mp_neighbor_index is None:
                    raise ValueError("attn_pool message passing requires mp_neighbor_index.")
                query_time = float(np.min(times))
                query_nodes = np.concatenate([src, dst, neg_src, neg_dst], axis=0)
                current_embeddings = apply_attention_pool_message_passing_for_nodes(
                    mp_encoder=gcn_encoder,
                    embeddings=current_embeddings,
                    lookup=lookup,
                    query_node_ids=query_nodes,
                    query_time=query_time,
                    neighbor_index=mp_neighbor_index,
                    num_neighbors=mp_num_neighbors,
                )
            else:
                raise ValueError(f"Unsupported learnable_mp_type: {learnable_mp_type}")
            if do_profile:
                prof_mp += time.perf_counter() - t_start

        scoring_embeddings = current_embeddings
        if cross_attn_use_raw_embeddings and getattr(model, 'requires_neighbor_context', False):
            scoring_embeddings = current_base_embeddings

        batch_ncn_adj = static_ncn_adj
        batch_mplp_exact_adj2 = static_mplp_exact_adj2
        if rolling_provider is not None:
            batch_ncn_adj = rolling_provider.current_binary_adj
            batch_mplp_exact_adj2 = rolling_provider.current_binary_two_hop_adj

        use_mlp_input_aux = semantic_aux_fusion_mode == 'late_concat'
        combined_aux_features = None
        if use_mlp_input_aux:
            combined_src = np.concatenate([src, neg_src], axis=0)
            combined_dst = np.concatenate([dst, neg_dst], axis=0)
            combined_times = np.concatenate([times, np.repeat(times, num_negatives)], axis=0)
            combined_raw_heuristic_features = None
            if (
                precomputed_pos_raw_heuristic_features is not None
                and precomputed_neg_raw_heuristic_features is not None
            ):
                neg_feature_index = (
                    batch_indices[:, None] * num_negatives
                    + np.arange(num_negatives, dtype=np.int64)[None, :]
                ).reshape(-1)
                combined_raw_heuristic_features = np.concatenate(
                    [
                        precomputed_pos_raw_heuristic_features[batch_indices],
                        precomputed_neg_raw_heuristic_features[neg_feature_index],
                    ],
                    axis=0,
                )
            combined_aux_features = build_semantic_mlp_auxiliary_features(
                heuristic_extractor=heuristic_extractor,
                lookup=lookup,
                sources=combined_src,
                targets=combined_dst,
                prediction_times=combined_times,
                raw_heuristic_features=combined_raw_heuristic_features,
                ncn_adj=batch_ncn_adj if use_mplp_exact_input_features else None,
                two_hop_adj=batch_mplp_exact_adj2 if use_mplp_exact_input_features else None,
                output_device=embeddings.device,
                output_dtype=scoring_embeddings.dtype,
                normalize_heuristics=not getattr(model, 'expects_raw_auxiliary_features', False),
                profile_out=None,
            )

        t_start = _profile_stage_start(embeddings.device, do_profile)
        score_profile = {} if do_profile else None
        pos_logits, _ = score_edge_batch(
            model,
            scoring_embeddings,
            lookup,
            src_t,
            dst_t,
            auxiliary_features=(
                None if combined_aux_features is None else combined_aux_features[:len(src)]
            ),
            node_interact_times=times,
            neighbor_index=neighbor_index,
            ncn_adj=batch_ncn_adj,
            cross_attn_num_neighbors=cross_attn_num_neighbors,
            ncn_num_neighbors=ncn_num_neighbors,
            seqfilter_num_neighbors=seqfilter_num_neighbors,
            profile_out=score_profile,
        )
        neg_times = np.repeat(times, num_negatives)
        neg_logits, _ = score_edge_batch(
            model,
            scoring_embeddings,
            lookup,
            neg_src_t,
            neg_dst_t,
            auxiliary_features=(
                None if combined_aux_features is None else combined_aux_features[len(src):]
            ),
            node_interact_times=neg_times,
            neighbor_index=neighbor_index,
            ncn_adj=batch_ncn_adj,
            cross_attn_num_neighbors=cross_attn_num_neighbors,
            ncn_num_neighbors=ncn_num_neighbors,
            seqfilter_num_neighbors=seqfilter_num_neighbors,
            profile_out=score_profile,
        )
        if do_profile:
            prof_score += _profile_stage_elapsed(t_start, embeddings.device)
            prof_score_id_to_tensor += score_profile.get('score_id_to_tensor_s', 0.0)
            prof_score_lookup += score_profile.get('score_lookup_s', 0.0)
            prof_score_neighbor_fetch += score_profile.get('score_neighbor_fetch_s', 0.0)
            prof_score_neighbor_marshal += score_profile.get('score_neighbor_marshal_s', 0.0)
            prof_score_gather += score_profile.get('score_gather_s', 0.0)
            prof_score_forward += score_profile.get('score_forward_s', 0.0)
            prof_score_valid_edges += score_profile.get('score_valid_edges', 0.0)
            prof_ncn_history_fetch += score_profile.get('ncn_history_fetch_s', 0.0)
            prof_ncn_marshal += score_profile.get('ncn_marshal_s', 0.0)
            prof_ncn_sparse_build += score_profile.get('ncn_sparse_build_s', 0.0)
            prof_ncn_row_slice += score_profile.get('ncn_row_slice_s', 0.0)
            prof_ncn_overlap += score_profile.get('ncn_overlap_s', 0.0)
            prof_ncn_aggregate += score_profile.get('ncn_aggregate_s', 0.0)
            prof_ncn_edges += score_profile.get('ncn_edge_count', 0.0)

        if not use_mlp_input_aux:
            t_start = time.perf_counter() if do_profile else None
            pos_logits, neg_logits, _ = fuse_pos_neg_logits_with_mplp_exact(
                mplp_exact_fusion=mplp_exact_fusion,
                lookup=lookup,
                pos_logits=pos_logits,
                neg_logits=neg_logits,
                pos_src=src,
                pos_dst=dst,
                neg_src=neg_src,
                neg_dst=neg_dst,
                ncn_adj=batch_ncn_adj,
                two_hop_adj=batch_mplp_exact_adj2,
            )
            if do_profile:
                prof_mplp += time.perf_counter() - t_start

            heur_profile = {} if do_profile else None
            pos_logits, neg_logits, heuristic_elapsed = fuse_pos_neg_logits_with_heuristics(
                heuristic_extractor=heuristic_extractor,
                heuristic_fusion=heuristic_fusion,
                pos_logits=pos_logits,
                neg_logits=neg_logits,
                pos_src=src,
                pos_dst=dst,
                pos_times=times,
                neg_src=neg_src,
                neg_dst=neg_dst,
                neg_times=neg_times,
                pos_raw_features=(
                    precomputed_pos_raw_heuristic_features[batch_indices]
                    if precomputed_pos_raw_heuristic_features is not None
                    else None
                ),
                neg_raw_features=(
                    precomputed_neg_raw_heuristic_features[
                        (
                            batch_indices[:, None] * num_negatives
                            + np.arange(num_negatives, dtype=np.int64)[None, :]
                        ).reshape(-1)
                    ]
                    if precomputed_neg_raw_heuristic_features is not None
                    else None
                ),
                profile_out=heur_profile,
            )
            if do_profile:
                prof_heur += heuristic_elapsed
                prof_heur_lookup += heur_profile.get('lookup_s', 0.0)
                prof_heur_recency += heur_profile.get('recency_s', 0.0)
                prof_heur_popularity += heur_profile.get('popularity_s', 0.0)
                prof_heur_past += heur_profile.get('past_s', 0.0)
                prof_heur_ra += heur_profile.get('ra_s', 0.0)
                prof_heur_build += heur_profile.get('build_s', 0.0)
                prof_heur_update += heur_profile.get('update_s', 0.0)
                prof_heur_post += heur_profile.get('post_s', 0.0)
                prof_heur_queries += heur_profile.get('query_count', 0.0)
                prof_heur_misses += heur_profile.get('miss_count', 0.0)

        neg_scores_matrix = neg_logits.reshape(batch_size, num_negatives)

        # MRR (1 vs K)
        ranks = 1 + torch.sum(neg_scores_matrix >= pos_logits[:, None], dim=1)
        reciprocal_ranks = (1.0 / ranks.float()).cpu().numpy().astype(np.float64, copy=False)
        mrr_list.extend(reciprocal_ranks.tolist())

        # DTGB metric layout is [pos, neg1, neg2, ...] per query.
        pos_np = pos_logits.detach().float().cpu().numpy().astype(np.float64, copy=False)
        neg_np = neg_scores_matrix.detach().float().cpu().numpy().astype(np.float64, copy=False)
        if bucket_times is not None:
            bucket_times.append(np.asarray(times, dtype=np.float64))
            bucket_pos_scores.append(pos_np.copy())
            bucket_neg_scores.append(neg_np.copy())
            bucket_rr.append(reciprocal_ranks.copy())
        batch_scores = np.empty(batch_size * (1 + num_negatives), dtype=np.float64)
        batch_labels = np.empty(batch_size * (1 + num_negatives), dtype=np.int64)
        for row_idx in range(batch_size):
            start = row_idx * (1 + num_negatives)
            batch_scores[start] = float(pos_np[row_idx])
            batch_labels[start] = 1
            batch_scores[start + 1 : start + 1 + num_negatives] = neg_np[row_idx]
            batch_labels[start + 1 : start + 1 + num_negatives] = 0
        all_scores.append(batch_scores)
        all_labels.append(batch_labels)

        if rolling_provider is not None:
            t_start = time.perf_counter() if do_profile else None
            rolling_provider.commit_batch(src, dst, times)
            if do_profile:
                prof_commit += time.perf_counter() - t_start

        if do_profile:
            prof_total += time.perf_counter() - batch_t0
            profile_seen += 1

    if all_scores:
        flat_scores = np.concatenate(all_scores, axis=0)
        flat_labels = np.concatenate(all_labels, axis=0)
        metric_payload = compute_prediction_metrics(
            flat_scores,
            flat_labels,
            dtgb_eval_batch_size=dtgb_eval_batch_size,
        )
    else:
        metric_payload = {
            'ap': 0.0,
            'auc': 0.0,
            'ap_global': 0.0,
            'auc_global': 0.0,
            'accuracy': 0.0,
            'dtgb_eval_batch_size': (
                int(dtgb_eval_batch_size) if dtgb_eval_batch_size is not None else None
            ),
            'dtgb_num_metric_batches': 0,
            'dtgb_metric_aggregation': 'pooled',
        }

    metrics = {
        'average_precision': float(metric_payload['ap']),
        'roc_auc': float(metric_payload['auc']),
        'mrr': float(np.mean(mrr_list)) if mrr_list else 0.0,
        'average_precision_global': float(metric_payload.get('ap_global', metric_payload['ap'])),
        'roc_auc_global': float(metric_payload.get('auc_global', metric_payload['auc'])),
        'accuracy': float(metric_payload.get('accuracy', 0.0)),
        'dtgb_eval_batch_size': metric_payload.get('dtgb_eval_batch_size'),
        'dtgb_num_metric_batches': int(metric_payload.get('dtgb_num_metric_batches', 0)),
        'dtgb_metric_aggregation': str(metric_payload.get('dtgb_metric_aggregation', 'pooled')),
    }
    if bucket_times is not None and bucket_times:
        flat_times = np.concatenate(bucket_times, axis=0)
        flat_pos = np.concatenate(bucket_pos_scores, axis=0)
        flat_neg = np.concatenate(bucket_neg_scores, axis=0)
        flat_rr = np.concatenate(bucket_rr, axis=0)
        num_queries = flat_times.shape[0]
        if num_queries > 0:
            quantiles = np.linspace(0.0, 1.0, int(time_bucket_count) + 1, dtype=np.float64)
            boundaries = np.quantile(flat_times, quantiles)
            bucket_metrics = []
            for bucket_idx in range(int(time_bucket_count)):
                bucket_lo = float(boundaries[bucket_idx])
                bucket_hi = float(boundaries[bucket_idx + 1])
                if bucket_idx == int(time_bucket_count) - 1:
                    mask = (flat_times >= bucket_lo) & (flat_times <= bucket_hi)
                else:
                    mask = (flat_times >= bucket_lo) & (flat_times < bucket_hi)
                if not np.any(mask):
                    continue
                sel_pos = flat_pos[mask]
                sel_neg = flat_neg[mask]
                bucket_scores = np.empty(sel_pos.shape[0] * (1 + num_negatives), dtype=np.float64)
                bucket_labels = np.empty(sel_pos.shape[0] * (1 + num_negatives), dtype=np.int64)
                for row_idx in range(sel_pos.shape[0]):
                    start = row_idx * (1 + num_negatives)
                    bucket_scores[start] = float(sel_pos[row_idx])
                    bucket_labels[start] = 1
                    bucket_scores[start + 1 : start + 1 + num_negatives] = sel_neg[row_idx]
                    bucket_labels[start + 1 : start + 1 + num_negatives] = 0
                bucket_payload = compute_prediction_metrics(
                    bucket_scores,
                    bucket_labels,
                    dtgb_eval_batch_size=dtgb_eval_batch_size,
                )
                bucket_metrics.append({
                    'bucket_index': int(bucket_idx + 1),
                    'time_min': bucket_lo,
                    'time_max': bucket_hi,
                    'num_queries': int(sel_pos.shape[0]),
                    'average_precision': float(bucket_payload['ap']),
                    'roc_auc': float(bucket_payload['auc']),
                    'mrr': float(np.mean(flat_rr[mask])),
                })
            metrics['time_bucket_metrics'] = bucket_metrics
    if profile_out is not None:
        profile_out.clear()
        if profile_seen > 0 and prof_total > 0:
            inv = 1000.0 / float(profile_seen)
            prof_other = prof_total - (
                prof_prepare + prof_neg + prof_mp + prof_heur + prof_mplp + prof_score + prof_commit
            )
            profile_out.update({
                'batches': float(profile_seen),
                'prepare_ms': prof_prepare * inv,
                'neg_ms': prof_neg * inv,
                'mp_ms': prof_mp * inv,
                'heur_ms': prof_heur * inv,
                'mplp_ms': prof_mplp * inv,
                'heur_lookup_ms': prof_heur_lookup * inv,
                'heur_recency_ms': prof_heur_recency * inv,
                'heur_popularity_ms': prof_heur_popularity * inv,
                'heur_past_ms': prof_heur_past * inv,
                'heur_ra_ms': prof_heur_ra * inv,
                'heur_build_ms': prof_heur_build * inv,
                'heur_update_ms': prof_heur_update * inv,
                'heur_post_ms': prof_heur_post * inv,
                'heur_queries': prof_heur_queries / float(profile_seen),
                'heur_misses': prof_heur_misses / float(profile_seen),
                'score_ms': prof_score * inv,
                'score_id_to_tensor_ms': prof_score_id_to_tensor * inv,
                'score_lookup_ms': prof_score_lookup * inv,
                'score_neighbor_fetch_ms': prof_score_neighbor_fetch * inv,
                'score_neighbor_marshal_ms': prof_score_neighbor_marshal * inv,
                'score_gather_ms': prof_score_gather * inv,
                'score_forward_ms': prof_score_forward * inv,
                'score_valid_edges': prof_score_valid_edges / float(profile_seen),
                'ncn_history_fetch_ms': prof_ncn_history_fetch * inv,
                'ncn_marshal_ms': prof_ncn_marshal * inv,
                'ncn_sparse_build_ms': prof_ncn_sparse_build * inv,
                'ncn_row_slice_ms': prof_ncn_row_slice * inv,
                'ncn_overlap_ms': prof_ncn_overlap * inv,
                'ncn_aggregate_ms': prof_ncn_aggregate * inv,
                'ncn_edges': prof_ncn_edges / float(profile_seen),
                'commit_ms': prof_commit * inv,
                'other_ms': prof_other * inv,
                'total_ms': prof_total * inv,
                'prepare_pct': 100.0 * prof_prepare / prof_total,
                'neg_pct': 100.0 * prof_neg / prof_total,
                'mp_pct': 100.0 * prof_mp / prof_total,
                'heur_pct': 100.0 * prof_heur / prof_total,
                'mplp_pct': 100.0 * prof_mplp / prof_total,
                'score_pct': 100.0 * prof_score / prof_total,
                'commit_pct': 100.0 * prof_commit / prof_total,
                'other_pct': 100.0 * prof_other / prof_total,
            })
        else:
            profile_out.update({
                'batches': 0.0,
                'prepare_ms': 0.0,
                'neg_ms': 0.0,
                'mp_ms': 0.0,
                'heur_ms': 0.0,
                'mplp_ms': 0.0,
                'heur_lookup_ms': 0.0,
                'heur_recency_ms': 0.0,
                'heur_popularity_ms': 0.0,
                'heur_past_ms': 0.0,
                'heur_ra_ms': 0.0,
                'heur_build_ms': 0.0,
                'heur_update_ms': 0.0,
                'heur_post_ms': 0.0,
                'heur_queries': 0.0,
                'heur_misses': 0.0,
                'score_ms': 0.0,
                'score_id_to_tensor_ms': 0.0,
                'score_lookup_ms': 0.0,
                'score_neighbor_fetch_ms': 0.0,
                'score_neighbor_marshal_ms': 0.0,
                'score_gather_ms': 0.0,
                'score_forward_ms': 0.0,
                'score_valid_edges': 0.0,
                'ncn_history_fetch_ms': 0.0,
                'ncn_marshal_ms': 0.0,
                'ncn_sparse_build_ms': 0.0,
                'ncn_row_slice_ms': 0.0,
                'ncn_overlap_ms': 0.0,
                'ncn_aggregate_ms': 0.0,
                'ncn_edges': 0.0,
                'commit_ms': 0.0,
                'other_ms': 0.0,
                'total_ms': 0.0,
                'prepare_pct': 0.0,
                'neg_pct': 0.0,
                'mp_pct': 0.0,
                'heur_pct': 0.0,
                'mplp_pct': 0.0,
                'score_pct': 0.0,
                'commit_pct': 0.0,
                'other_pct': 0.0,
            })
    return metrics
