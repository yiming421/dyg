#!/usr/bin/env python3
"""
Evaluate heuristic baselines using DTGB's standard evaluation pipeline
Follows the exact same protocol as other DTGB models
"""
import numpy as np
import torch
import time
import argparse
import os
import sys
import re
from functools import partial
from tqdm import tqdm
from sklearn.metrics import pairwise_distances

_EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_EXPERIMENTS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.DataLoader import get_link_prediction_data, get_idx_data_loader
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from utils.metrics import get_link_prediction_metrics
from experiments.modules.heuristic_models import (score_links_by_recency, score_links_by_popularity,
                                                  score_links_by_recent_degree, score_links_by_past_interactions, score_links_by_global_recency,
                                                  score_links_by_itemcf_cosine, score_links_by_markov_transition, score_links_by_usercf_cosine,
                                                  score_links_by_personalized_location,
                                                  score_links_by_common_neighbors, score_links_by_semantic_similarity,
                                                  score_links_by_semantic_history_mean,
                                                  score_links_by_semantic_history_query_conditioned,
                                                  precompute_entity_embeddings, smooth_embeddings_by_time_window_torch)
from experiments.modules.llm_lp.experiment import build_prompt_entity_map
from experiments.modules.rrf.utils import reciprocal_rank_fusion_from_scores


def build_source_history_mean_initialized_embeddings(
    base_embeddings: np.ndarray,
    node_id_lookup: np.ndarray,
    src_node_ids: np.ndarray,
    dst_node_ids: np.ndarray,
):
    """
    Replace each source-node embedding with the mean of its observed destination/item
    embeddings under the supplied edge set. Destination embeddings remain unchanged.
    """
    base = np.asarray(base_embeddings, dtype=np.float32)
    lookup = np.asarray(node_id_lookup, dtype=np.int64)
    src_arr = np.asarray(src_node_ids, dtype=np.int64)
    dst_arr = np.asarray(dst_node_ids, dtype=np.int64)

    updated = base.copy()
    sums = np.zeros_like(base, dtype=np.float32)
    counts = np.zeros(base.shape[0], dtype=np.int64)

    valid = (
        (src_arr >= 0) & (src_arr < len(lookup)) &
        (dst_arr >= 0) & (dst_arr < len(lookup))
    )
    if not np.any(valid):
        return updated, 0

    src_emb_idx = lookup[src_arr[valid]]
    dst_emb_idx = lookup[dst_arr[valid]]
    valid_pairs = (src_emb_idx >= 0) & (dst_emb_idx >= 0)
    if not np.any(valid_pairs):
        return updated, 0

    src_emb_idx = src_emb_idx[valid_pairs]
    dst_emb_idx = dst_emb_idx[valid_pairs]

    np.add.at(sums, src_emb_idx, base[dst_emb_idx])
    np.add.at(counts, src_emb_idx, 1)

    replace_mask = counts > 0
    if np.any(replace_mask):
        updated[replace_mask] = sums[replace_mask] / counts[replace_mask, None]
        norms = np.linalg.norm(updated[replace_mask], axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        updated[replace_mask] = updated[replace_mask] / norms

    return updated, int(np.sum(replace_mask))


_GOOGLEMAP_CITY_STATE_RE = re.compile(
    r",\s*([^,.\n]+?)\s*,\s*([A-Z]{2})(?:\s+\d{5}(?:-\d{4})?)?(?=\.|,|$)"
)


def extract_googlemap_city_state(text) -> str:
    """
    Extract coarse location as "City, ST" from Googlemap business-card text.
    User-name rows and malformed item rows return an empty string.
    """
    if not isinstance(text, str):
        return ""
    clean = " ".join(text.strip().split())
    if not clean or not clean.startswith("Name:"):
        return ""

    address_segment = clean
    category_pos = address_segment.find(". Category:")
    if category_pos >= 0:
        address_segment = address_segment[:category_pos]

    matches = _GOOGLEMAP_CITY_STATE_RE.findall(address_segment)
    if not matches:
        return ""

    city, state = matches[-1]
    city = city.strip()
    state = state.strip().upper()
    if not city or len(state) != 2:
        return ""
    return f"{city}, {state}"


def build_googlemap_location_ids(entity_text_df, max_node_id: int):
    """
    Build node_id -> compact location id lookup for Googlemap item/business nodes.
    Nodes without a parsed city/state get -1.
    """
    location_ids = np.full(max_node_id + 1, -1, dtype=np.int64)
    location_to_id = {}

    for node_id, text in zip(entity_text_df['i'], entity_text_df['text']):
        node_id = int(node_id)
        if node_id < 0 or node_id > max_node_id:
            continue
        city_state = extract_googlemap_city_state(text)
        if not city_state:
            continue
        loc_id = location_to_id.get(city_state)
        if loc_id is None:
            loc_id = len(location_to_id)
            location_to_id[city_state] = loc_id
        location_ids[node_id] = loc_id

    return location_ids, location_to_id


def _iter_online_scorers(scoring_function):
    scorers = scoring_function if isinstance(scoring_function, list) else [scoring_function]
    seen = set()
    for scorer in scorers:
        if hasattr(scorer, 'prepare_batch') and hasattr(scorer, 'commit_batch'):
            sid = id(scorer)
            if sid in seen:
                continue
            seen.add(sid)
            yield scorer


class FullResmoothPerBatchSemanticScorer:
    """
    Performance-first online semantic scorer:
    - Recompute full non-parameterized GCN smoothing before every eval batch
      using history edges strictly before the current batch timestamp.
    - Commit current batch positive edges after scoring.
    """
    def __init__(self,
                 embeddings,
                 entity_id_to_idx,
                 init_src_node_ids: np.ndarray,
                 init_dst_node_ids: np.ndarray,
                 init_node_interact_times: np.ndarray,
                 time_window: float,
                 num_steps: int,
                 symmetric_norm: bool,
                 decay_gamma: float,
                 residual_alpha: float,
                 undirected: bool,
                 layer_norm: str,
                 pre_norm: str,
                 debug: bool,
                 log_dampen: bool,
                 supernode_strength: float,
                 hetero_highpass_coef: float,
                 hetero_tau: float,
                 hetero_filter_tau: float,
                 hetero_filter_keep_weight: float,
                 endpoint_topk_recent: int,
                 endpoint_topk_ensure_coverage: bool,
                 score_fn,
                 score_kwargs,
                 device: str):
        self.embeddings = embeddings
        self.entity_id_to_idx = entity_id_to_idx

        self.init_src_node_ids = np.asarray(init_src_node_ids).copy()
        self.init_dst_node_ids = np.asarray(init_dst_node_ids).copy()
        self.init_node_interact_times = np.asarray(init_node_interact_times).copy()

        self.time_window = time_window
        self.num_steps = num_steps
        self.symmetric_norm = symmetric_norm
        self.decay_gamma = decay_gamma
        self.residual_alpha = residual_alpha
        self.undirected = undirected
        self.layer_norm = layer_norm
        self.pre_norm = pre_norm
        self.debug = debug
        self.log_dampen = log_dampen
        self.supernode_strength = supernode_strength
        self.hetero_highpass_coef = hetero_highpass_coef
        self.hetero_tau = hetero_tau
        self.hetero_filter_tau = hetero_filter_tau
        self.hetero_filter_keep_weight = hetero_filter_keep_weight
        self.endpoint_topk_recent = endpoint_topk_recent
        self.endpoint_topk_ensure_coverage = endpoint_topk_ensure_coverage
        self.score_fn = score_fn
        self.score_kwargs = dict(score_kwargs or {})
        self.device = device

        self.history_src_node_ids = None
        self.history_dst_node_ids = None
        self.history_node_interact_times = None
        self.current_smoothed_embeddings = None
        self.current_batch_time = None
        self.reset_history()

    def reset_history(self):
        self.history_src_node_ids = self.init_src_node_ids.copy()
        self.history_dst_node_ids = self.init_dst_node_ids.copy()
        self.history_node_interact_times = self.init_node_interact_times.copy()
        self.current_smoothed_embeddings = None
        self.current_batch_time = None

    def _resmooth_for_time(self, batch_start_time: float):
        # Strict temporal protocol: only edges strictly before current batch.
        time_mask = self.history_node_interact_times < batch_start_time
        if self.time_window is None:
            window_mask = np.ones_like(time_mask, dtype=bool)
        else:
            window_mask = self.history_node_interact_times >= (batch_start_time - self.time_window)
        mask = time_mask & window_mask

        cur_src = self.history_src_node_ids[mask]
        cur_dst = self.history_dst_node_ids[mask]
        cur_times = self.history_node_interact_times[mask]

        if len(cur_src) == 0:
            smoothed = self.embeddings
        else:
            # Window is already applied by explicit mask above.
            smoothed = smooth_embeddings_by_time_window_torch(
                embeddings=self.embeddings,
                src_node_ids=cur_src,
                dst_node_ids=cur_dst,
                node_interact_times=cur_times,
                time_window=1e30,
                num_steps=self.num_steps,
                symmetric_norm=self.symmetric_norm,
                decay_gamma=self.decay_gamma,
                residual_alpha=self.residual_alpha,
                undirected=self.undirected,
                layer_norm=self.layer_norm,
                pre_norm=self.pre_norm,
                device=self.device,
                debug=self.debug,
                log_dampen=self.log_dampen,
                supernode_strength=self.supernode_strength,
                reference_time=batch_start_time,
                hetero_highpass_coef=self.hetero_highpass_coef,
                hetero_tau=self.hetero_tau,
                hetero_filter_tau=self.hetero_filter_tau,
                hetero_filter_keep_weight=self.hetero_filter_keep_weight,
                endpoint_topk_recent=self.endpoint_topk_recent,
                endpoint_topk_ensure_coverage=self.endpoint_topk_ensure_coverage
            )

        norms = torch.norm(smoothed, dim=1, keepdim=True)
        norms[norms == 0] = 1.0
        self.current_smoothed_embeddings = smoothed / norms
        self.current_batch_time = batch_start_time

    def prepare_batch(self, batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times):
        if len(batch_node_interact_times) == 0:
            return
        batch_start_time = float(batch_node_interact_times[0])
        if self.current_batch_time != batch_start_time:
            self._resmooth_for_time(batch_start_time)

    def commit_batch(self, batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times):
        if len(batch_src_node_ids) == 0:
            return
        self.history_src_node_ids = np.concatenate([self.history_src_node_ids, np.asarray(batch_src_node_ids)])
        self.history_dst_node_ids = np.concatenate([self.history_dst_node_ids, np.asarray(batch_dst_node_ids)])
        self.history_node_interact_times = np.concatenate([self.history_node_interact_times, np.asarray(batch_node_interact_times)])

    def __call__(self, neighbor_sampler, sources, targets, prediction_times):
        if self.current_smoothed_embeddings is None:
            self.prepare_batch(None, None, prediction_times)
        return self.score_fn(
            neighbor_sampler=neighbor_sampler,
            sources=sources,
            targets=targets,
            prediction_times=prediction_times,
            embeddings=self.current_smoothed_embeddings,
            entity_id_to_idx=self.entity_id_to_idx,
            **self.score_kwargs,
        )


def evaluate_heuristic_baseline_dtgb(scoring_function,
                                     neighbor_sampler,
                                     evaluate_idx_data_loader,
                                     evaluate_neg_edge_sampler,
                                     evaluate_data):
    """
    Evaluate a heuristic baseline following DTGB's evaluation protocol

    This mirrors evaluate_model_link_prediction() from evaluate_models_utils.py
    but uses a heuristic scoring function instead of a neural model

    Args:
        scoring_function: Function(neighbor_sampler, sources, targets, times) -> scores
        neighbor_sampler: NeighborSampler for retrieving historical interactions
        evaluate_idx_data_loader: DataLoader with evaluation indices
        evaluate_neg_edge_sampler: NegativeEdgeSampler for negative samples
        evaluate_data: Data object with test edges

    Returns:
        evaluate_metrics: list of metric dicts (AP, AUC)
    """
    # Reset random state for consistent negative sampling
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    evaluate_metrics = []
    evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)

    for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
        evaluate_data_indices = evaluate_data_indices.numpy()

        # Get positive edges
        batch_src_node_ids = evaluate_data.src_node_ids[evaluate_data_indices]
        batch_dst_node_ids = evaluate_data.dst_node_ids[evaluate_data_indices]
        batch_node_interact_times = evaluate_data.node_interact_times[evaluate_data_indices]

        # Sample negative edges (following DTGB protocol)
        if evaluate_neg_edge_sampler.negative_sample_strategy != 'random':
            batch_neg_src_node_ids, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(
                size=len(batch_src_node_ids),
                batch_src_node_ids=batch_src_node_ids,
                batch_dst_node_ids=batch_dst_node_ids,
                current_batch_start_time=batch_node_interact_times[0],
                current_batch_end_time=batch_node_interact_times[-1]
            )
        else:
            _, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids))
            batch_neg_src_node_ids = batch_src_node_ids

        # Score positive edges
        positive_scores = scoring_function(
            neighbor_sampler=neighbor_sampler,
            sources=batch_src_node_ids,
            targets=batch_dst_node_ids,
            prediction_times=batch_node_interact_times
        )

        # Score negative edges
        negative_scores = scoring_function(
            neighbor_sampler=neighbor_sampler,
            sources=batch_neg_src_node_ids,
            targets=batch_neg_dst_node_ids,
            prediction_times=batch_node_interact_times
        )

        # Convert to torch tensors and create labels (1 for positive, 0 for negative)
        predicts = torch.from_numpy(np.concatenate([positive_scores, negative_scores])).float()
        labels = torch.cat([torch.ones(len(positive_scores)), torch.zeros(len(negative_scores))])

        # Compute metrics using DTGB's metric function
        evaluate_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

    return evaluate_metrics


    return evaluate_metrics


def evaluate_heuristic_combined(scoring_function,
                                neighbor_sampler,
                                evaluate_idx_data_loader,
                                evaluate_neg_edge_sampler,
                                evaluate_data,
                                num_negatives=100,
                                fusion_strategy=None,
                                max_eval_batches=None):
    """
    Evaluate both Standard AP/AUC (1 vs 1) and MRR (1 vs K) in one pass.
    Supports RRF fusion if scoring_function is a list.
    """
    evaluate_neg_edge_sampler.reset_random_state()
    mrr_list = []
    ap_list = []
    auc_list = []
    
    is_ensemble = isinstance(scoring_function, list)
    rrf_k = 60
    
    total_sample_time = 0.0
    total_score_time = 0.0
    online_scorers = list(_iter_online_scorers(scoring_function))
    
    evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
    for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
        if max_eval_batches is not None and batch_idx >= max_eval_batches:
            break
        evaluate_data_indices = evaluate_data_indices.numpy()

        # Get positive edges
        batch_src_node_ids = evaluate_data.src_node_ids[evaluate_data_indices]
        batch_dst_node_ids = evaluate_data.dst_node_ids[evaluate_data_indices]
        batch_node_interact_times = evaluate_data.node_interact_times[evaluate_data_indices]

        batch_size = len(batch_src_node_ids)
        for online_scorer in online_scorers:
            online_scorer.prepare_batch(batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times)

        # 2. Sample Negatives (K per positive) FIRST
        # We need negatives to form the candidate set for Ranking (RRF needs the set)
        t1 = time.time()
        
        flat_neg_src = None
        flat_neg_dst = None
        
        if evaluate_neg_edge_sampler.negative_sample_strategy != 'random':
            ns, nd = evaluate_neg_edge_sampler.sample(
                size=batch_size * num_negatives,
                batch_src_node_ids=np.repeat(batch_src_node_ids, num_negatives),
                batch_dst_node_ids=np.repeat(batch_dst_node_ids, num_negatives),
                current_batch_start_time=batch_node_interact_times[0],
                current_batch_end_time=batch_node_interact_times[-1]
            )
            flat_neg_src = ns
            flat_neg_dst = nd
        else:
            _, nd = evaluate_neg_edge_sampler.sample(size=batch_size * num_negatives)
            flat_neg_src = np.repeat(batch_src_node_ids, num_negatives)
            flat_neg_dst = nd

        total_sample_time += time.time() - t1
        flat_neg_times = np.repeat(batch_node_interact_times, num_negatives)

        # Helper to compute scores for a given scorer
        def compute_all_scores(scorer):
            # Positives
            pos = scorer(
                neighbor_sampler=neighbor_sampler,
                sources=batch_src_node_ids,
                targets=batch_dst_node_ids,
                prediction_times=batch_node_interact_times
            )
            # Negatives
            neg_flat = scorer(
                neighbor_sampler=neighbor_sampler,
                sources=flat_neg_src,
                targets=flat_neg_dst,
                prediction_times=flat_neg_times
            )
            neg_mat = neg_flat.reshape(batch_size, num_negatives)
            return pos, neg_mat

        t2 = time.time()
        
        ap_auc_pos_scores = None
        ap_auc_neg_scores_1vs1 = None

        if not is_ensemble:
            # Standard single model
            pos_scores, neg_scores_matrix = compute_all_scores(scoring_function)
        else:
            strategy = fusion_strategy or 'rrf'
            # Compute all component outputs once (used by both HyperFusion and RRF)
            component_pos = []
            component_neg = []
            for scorer in scoring_function:
                p, n = compute_all_scores(scorer)
                component_pos.append(p)
                component_neg.append(n)

            component_pos = np.stack(component_pos)  # (M, B)
            component_neg = np.stack(component_neg)  # (M, B, K)

            if strategy == 'hyperfusion':
                d_pos = pairwise_distances(component_pos, metric='cosine')
                d_neg = pairwise_distances(component_neg.reshape(component_neg.shape[0], -1), metric='cosine')

                d_all = [d_pos, d_neg]
                num_models = component_pos.shape[0]
                H = np.zeros((num_models, len(d_all)))

                for idx, dd in enumerate(d_all):
                    where = np.argwhere((dd > 0) & (dd < 0.1))
                    for w in where:
                        for i in w:
                            H[i, idx] = 1

                A = H @ H.T
                # Ensure self-connections so we never zero out all models
                A = A + np.eye(num_models)
                fused_pos = A @ component_pos
                fused_neg = A @ component_neg.reshape(num_models, -1)

                pos_scores = fused_pos.sum(0)
                neg_scores_matrix = fused_neg.reshape(num_models, batch_size, num_negatives).sum(0)
            else:
                # RRF Fusion
                combined_scores = np.concatenate(
                    [component_pos[:, :, None], component_neg],
                    axis=2
                )  # (num_models, batch, 1+num_negatives)
                rrf_scores_matrix = reciprocal_rank_fusion_from_scores(
                    combined_scores,
                    rrf_k=rrf_k,
                    model_axis=0,
                    rank_axis=2
                )

                pos_scores = rrf_scores_matrix[:, 0]
                neg_scores_matrix = rrf_scores_matrix[:, 1:]

                if strategy == 'rrf_eval_batch':
                    # For AP/AUC, fuse ranks over the current evaluation batch
                    # (B positives + B first negatives), while MRR keeps the
                    # standard per-query 1-vs-K candidate set above.
                    batch_scores_1v1 = np.concatenate(
                        [component_pos, component_neg[:, :, 0]],
                        axis=1
                    )  # (num_models, 2 * batch_size)
                    fused_batch_scores_1v1 = reciprocal_rank_fusion_from_scores(
                        batch_scores_1v1,
                        rrf_k=rrf_k,
                        model_axis=0,
                        rank_axis=1
                    )
                    ap_auc_pos_scores = fused_batch_scores_1v1[:batch_size]
                    ap_auc_neg_scores_1vs1 = fused_batch_scores_1v1[batch_size:]
            
        total_score_time += time.time() - t2

        # 3. Compute MRR (using ALL negatives)
        ranks = 1 + np.sum(neg_scores_matrix >= pos_scores[:, None], axis=1)
        reciprocal_ranks = 1.0 / ranks
        mrr_list.extend(reciprocal_ranks)
        
        # 4. Compute Standard AP/AUC (using 1st negative only)
        # This replicates the standard 1-vs-1 evaluation
        if ap_auc_pos_scores is None:
            ap_auc_pos_scores = pos_scores
            ap_auc_neg_scores_1vs1 = neg_scores_matrix[:, 0]
        
        predicts = torch.from_numpy(np.concatenate([ap_auc_pos_scores, ap_auc_neg_scores_1vs1])).float()
        labels = torch.cat([torch.ones(len(ap_auc_pos_scores)), torch.zeros(len(ap_auc_neg_scores_1vs1))])
        
        metrics = get_link_prediction_metrics(predicts=predicts, labels=labels)
        ap_list.append(metrics['average_precision'])
        auc_list.append(metrics['roc_auc'])
        for online_scorer in online_scorers:
            online_scorer.commit_batch(batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times)
        
    print(f"  Timing Breakdown: Sampling={total_sample_time:.2f}s, Scoring={total_score_time:.2f}s")
    
    return {
        'mrr': np.mean(mrr_list),
        'average_precision': np.mean(ap_list),
        'roc_auc': np.mean(auc_list)
    }


def main():
    """
    Run evaluation for heuristic baselines following DTGB's exact protocol
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate heuristic baselines')
    parser.add_argument('--dataset_name', type=str, default='GDELT',
                        help='Dataset name under ../DyLink_Datasets/')
    parser.add_argument('--metric', type=str, default='ap_auc', choices=['all', 'ap_auc', 'mrr'],
                        help='Metric to evaluate: all (AP/AUC + MRR), ap_auc (Standard), mrr (Ranking)')
    parser.add_argument('--num_negatives', type=int, default=100,
                        help='Number of negatives per positive for MRR evaluation')
    parser.add_argument('--negative_strategy', type=str, default='random', choices=['random', 'historical'],
                        help='Negative sampling strategy: random (easy), historical (hard, requires history)')
    parser.add_argument('--baseline', type=str, default='all',
                        choices=['recency', 'popularity', 'recent_degree', 'past_interactions', 'global_recency',
                                 'itemcf_cosine', 'markov_transition', 'usercf_cosine', 'personalized_location', 'cn', 'aa', 'ra', 'semantic', 'semantic_smooth', 'all', 'rrf', 'hyperfusion'],
                        help='Which baseline to evaluate (default: all)')
    parser.add_argument('--rrf_mode', type=str, default='local', choices=['local', 'eval_batch'],
                        help='RRF mode: local ranks within each query candidate set; eval_batch keeps local RRF for MRR but fuses AP/AUC ranks over the whole eval batch.')
    parser.add_argument('--decay', type=float, default=0.0,
                        help='Time decay factor for popularity (0=no decay, >0=exponential decay)')
    parser.add_argument('--recent_degree_window', type=float, default=30.0,
                        help='Lookback window for recent-degree popularity (count events in [t-window, t)).')
    parser.add_argument('--popularity_include_source', action='store_true',
                        help='If set, Popularity score is pop(src)+pop(dst) (temporal degrees before t_pred); default is target-only pop(dst).')
    parser.add_argument('--recency_directed', action='store_true',
                        help='If set, compute Recency using directed history only (src->dst). Default uses undirected history (DTGB neighbor sampler).')
    parser.add_argument('--smooth_time_window', type=float, default=140.0,
                        help='Time window for semantic smoothing (edges with t >= max_t - window)')
    parser.add_argument('--smooth_steps', type=int, default=1,
                        help='Number of propagation steps for semantic smoothing')
    parser.add_argument('--smooth_decay_gamma', type=float, default=None,
                        help='Time-decay rate for smoothing weights (exp(-gamma * delta_t)); if unset, edges are unweighted')
    parser.add_argument('--smooth_residual_alpha', type=float, default=0.0,
                        help='Residual connection weight (0=no residual, >0=mix original embeddings to prevent over-smoothing)')
    parser.add_argument('--smooth_debug', action='store_true',
                        help='Enable debug output for GCN smoothing diagnostics')
    parser.add_argument('--smooth_undirected', action='store_true',
                        help='If set, smoothing uses undirected graph (add both u->v and v->u). Default is directed.')
    parser.add_argument('--smooth_layer_norm', type=str, default=None, choices=[None, 'ln'],
                        help='Apply non-parameterized normalization between propagation steps: ln=LayerNorm (per row, mean=0 std=1)')
    parser.add_argument('--smooth_pre_norm', type=str, default=None, choices=[None, 'zscore'],
                        help='Apply normalization to initial embeddings before smoothing: zscore (population standardization per dim)')
    parser.add_argument('--smooth_log_dampen', action='store_true',
                        help='Apply log(1+x) to edge weights (interaction counts) before smoothing to dampen hubs')
    parser.add_argument('--smooth_supernode_strength', type=float, default=0.0,
                        help='If > 0, add a virtual supernode connected to all nodes with this edge weight during smoothing')
    parser.add_argument('--smooth_hetero_highpass_coef', type=float, default=0.0,
                        help='If > 0, apply high-pass correction on heterophilous edges during smoothing')
    parser.add_argument('--smooth_hetero_tau', type=float, default=0.0,
                        help='Cosine threshold for heterophilous edges (cos < tau)')
    parser.add_argument('--smooth_hetero_filter_tau', type=float, default=None,
                        help='If set, downweight edges with cos < tau during smoothing')
    parser.add_argument('--smooth_hetero_filter_keep_weight', type=float, default=1.0,
                        help='Weight multiplier for filtered edges (0=drop, 1=no effect)')
    parser.add_argument('--smooth_endpoint_topk_recent', type=int, default=None,
                        help='If set (>0), keep at most top-k recent incident edges per endpoint before smoothing')
    parser.add_argument('--smooth_endpoint_topk_ensure_coverage', action='store_true',
                        help='When endpoint top-k is enabled, force active endpoints to keep at least one edge')
    parser.add_argument('--smooth_full_resmooth_per_batch', action='store_true',
                        help='If set, recompute semantic smoothing from scratch before each eval batch, then commit batch positives')
    parser.add_argument('--embedding_cache', type=str, default=None,
                        help='Path to load/save base semantic embeddings (npz/npy). If exists, loads instead of recomputing.')
    parser.add_argument('--smoothed_embedding_cache', type=str, default=None,
                        help='Path to load/save smoothed embeddings (npz/npy). If exists, loads instead of recomputing.')
    parser.add_argument('--embedding_model', type=str, default='intfloat/e5-large-v2',
                        help='Embedding model name or local path for semantic baselines.')
    parser.add_argument('--entity_text_path', type=str, default=None,
                        help='Optional path to an alternate entity-text CSV (expects at least columns i,text).')
    parser.add_argument('--semantic_entity_name_mode', type=str, default='raw',
                        choices=['raw', 'auto', 'compressed', 'compressed_profile'],
                        help=('Optional prompt-style entity cleaning/compression to apply before semantic embedding. '
                              'This reuses the main LLM pipeline entity formatter. '
                              'Use auto/compressed for Stack_elec if you want semantic heuristics to embed the cleaned text instead of raw text.'))
    parser.add_argument('--semantic_mode', type=str, default='auto', choices=['auto', 'off', 'required'],
                        help='Semantic handling: auto=use entity_text.csv if present, off=disable semantic scorers, required=error if semantic inputs missing.')
    parser.add_argument('--semantic_source_profile', type=str, default='raw', choices=['raw', 'history_mean', 'query_conditioned_mean'],
                        help='How to form the source-side semantic vector: raw=node text embedding, history_mean=mean of historical neighbor/item embeddings before query time, query_conditioned_mean=target-conditioned weighted average of historical neighbor/item embeddings.')
    parser.add_argument('--semantic_query_temperature', type=float, default=0.2,
                        help='Softmax temperature for query_conditioned_mean source profiling (lower = sharper focus on candidate-similar history).')
    parser.add_argument('--semantic_smoothing_source_init', type=str, default='raw', choices=['raw', 'history_mean'],
                        help='How to initialize source/user embeddings before semantic smoothing: raw=keep raw node text embeddings, history_mean=replace each source/user node with the mean of its observed destination/item embeddings before smoothing.')
    parser.add_argument('--strict_no_leakage', action='store_true', default=True,
                        help='If set (default), use only Train+Val edges for smoothing graph. Prevents leakage.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for evaluation (default: 256, matching train_link_prediction.py)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of DataLoader workers for evaluation index loaders (default: 2; use 0 if multiprocessing semaphores are unavailable).')
    parser.add_argument('--max_eval_batches', type=int, default=0,
                        help='If >0, evaluate only this many batches per split (debug/quick checks).')
    parser.add_argument('--gpu_index', type=int, default=0,
                        help='CUDA GPU index for semantic embedding/smoothing/scoring (default: 0)')
    
    # GPU Heuristics
    parser.add_argument('--use_gpu_heuristics', action='store_true', help='Use GPU block-parallel kernels for CN/AA/RA')
    
    cmd_args = parser.parse_args()
    if cmd_args.gpu_index < 0:
        raise ValueError(f"--gpu_index must be >= 0, got {cmd_args.gpu_index}")

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if cmd_args.gpu_index >= num_gpus:
            raise ValueError(f"--gpu_index={cmd_args.gpu_index} out of range; available CUDA devices: 0..{num_gpus - 1}")
        semantic_device = f'cuda:{cmd_args.gpu_index}'
    else:
        semantic_device = 'cpu'

    if cmd_args.semantic_source_profile == 'history_mean':
        semantic_score_fn = score_links_by_semantic_history_mean
        semantic_score_kwargs = {}
    elif cmd_args.semantic_source_profile == 'query_conditioned_mean':
        semantic_score_fn = score_links_by_semantic_history_query_conditioned
        semantic_score_kwargs = {
            'attention_temperature': cmd_args.semantic_query_temperature,
        }
    else:
        semantic_score_fn = score_links_by_semantic_similarity
        semantic_score_kwargs = {}

    embedding_model_label = os.path.basename(os.path.normpath(cmd_args.embedding_model)) or cmd_args.embedding_model

    print("="*80)
    print("Heuristic Baselines Evaluation - DTGB Protocol")
    print("="*80)
    print(f"Dataset: {cmd_args.dataset_name}")
    print(f"Semantic device: {semantic_device}")
    print(f"Embedding model: {cmd_args.embedding_model}")
    print(f"Semantic entity name mode: {cmd_args.semantic_entity_name_mode}")
    if cmd_args.strict_no_leakage:
        print("NOTE: STRICT NO LEAKAGE mode enabled. Smoothing uses only Train+Val edges.")

    # Load data using DTGB's loader
    print("\nLoading data...")

    class Args:
        use_feature = 'None'
        model_name = 'HeuristicBaseline'
        skip_raw_features = True

    args = Args()

    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, \
        new_node_val_data, new_node_test_data, cat_num = get_link_prediction_data(
            dataset_name=cmd_args.dataset_name,
            val_ratio=0.15,
            test_ratio=0.15,
            args=args
        )

    print(f"✓ Data loaded")
    print(f"  Train: {train_data.num_interactions:,} edges")
    print(f"  Val: {val_data.num_interactions:,} edges")
    print(f"  Test (transductive): {test_data.num_interactions:,} edges")
    print(f"  Test (inductive): {new_node_test_data.num_interactions:,} edges")

    # Define the graph for smoothing
    if cmd_args.strict_no_leakage:
        # Cutoff at start of test
        test_start_time = test_data.node_interact_times[0]
        # Find index in full_data where time < test_start_time
        # Since full_data is sorted by time
        mask = full_data.node_interact_times < test_start_time
        num_observed = np.sum(mask)
        print(f"\nConstructing smoothing graph (Strict No Leakage)...")
        print(f"  Using {num_observed:,} edges (Train + Val) out of {len(full_data.src_node_ids):,} total.")
        
        smooth_src = full_data.src_node_ids[:num_observed]
        smooth_dst = full_data.dst_node_ids[:num_observed]
        smooth_times = full_data.node_interact_times[:num_observed]
    else:
        print(f"\nConstructing smoothing graph (Full Graph - Potential Leakage)...")
        smooth_src = full_data.src_node_ids
        smooth_dst = full_data.dst_node_ids
        smooth_times = full_data.node_interact_times

    # Build neighbor sampler (DTGB uses full_data for all evaluation)
    print("\nBuilding neighbor sampler on FULL DATA (DTGB protocol)...")
    start = time.time()
    full_neighbor_sampler = get_neighbor_sampler(
        data=full_data,
        sample_neighbor_strategy='recent',
        time_scaling_factor=0.0,
        seed=1  # Same seed as DTGB
    )
    print(f"✓ Built in {time.time() - start:.2f}s")

    # Initialize negative samplers (same seeds as DTGB)
    print(f"\nInitializing negative samplers (strategy: {cmd_args.negative_strategy})...")
    test_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids,
        dst_node_ids=full_data.dst_node_ids,
        interact_times=full_data.node_interact_times,
        last_observed_time=val_data.node_interact_times[-1],
        negative_sample_strategy=cmd_args.negative_strategy,
        seed=2
    )
    new_node_test_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=new_node_test_data.src_node_ids,
        dst_node_ids=new_node_test_data.dst_node_ids,
        interact_times=new_node_test_data.node_interact_times,
        last_observed_time=val_data.node_interact_times[-1],
        negative_sample_strategy=cmd_args.negative_strategy,
        seed=3
    )

    # Get data loaders
    print(f"Creating data loaders (Batch Size: {cmd_args.batch_size})...")
    test_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(test_data.src_node_ids))),
        batch_size=cmd_args.batch_size,
        shuffle=False,
        num_workers=cmd_args.num_workers,
    )
    new_node_test_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(new_node_test_data.src_node_ids))),
        batch_size=cmd_args.batch_size,
        shuffle=False,
        num_workers=cmd_args.num_workers,
    )

    # Load entity texts and precompute embeddings if semantic baselines are needed
    embeddings, entity_id_to_idx, smoothed_embeddings = None, None, None
    semantic_smooth_scorer = None
    node_location_ids, location_to_id = None, None
    semantic_requested_by_baseline = cmd_args.baseline in ('semantic', 'semantic_smooth', 'all', 'rrf', 'hyperfusion')
    location_requested_by_baseline = cmd_args.baseline in ('personalized_location', 'all')
    if cmd_args.semantic_mode == 'off' and cmd_args.baseline in ('semantic', 'semantic_smooth'):
        raise ValueError("--semantic_mode off is incompatible with --baseline semantic/semantic_smooth")

    entity_text_path = cmd_args.entity_text_path or os.path.join('..', 'DyLink_Datasets', cmd_args.dataset_name, 'entity_text.csv')
    metadata_requested_by_baseline = semantic_requested_by_baseline or location_requested_by_baseline
    if metadata_requested_by_baseline and not os.path.exists(entity_text_path):
        if cmd_args.baseline in ('semantic', 'semantic_smooth') or cmd_args.semantic_mode == 'required':
            raise FileNotFoundError(f"Semantic inputs missing: {entity_text_path}")
        if cmd_args.baseline == 'personalized_location':
            raise FileNotFoundError(f"Personalized location requires entity text metadata: {entity_text_path}")
        print(f"\nWarning: {entity_text_path} not found. Continuing without semantic scorers.")

    entity_text_df = None
    entity_texts = None
    if metadata_requested_by_baseline and os.path.exists(entity_text_path):
        import pandas as pd
        entity_text_df = pd.read_csv(entity_text_path)
        entity_texts = dict(zip(entity_text_df['i'], entity_text_df['text']))

    if location_requested_by_baseline and entity_text_df is not None:
        max_node_id = int(max(full_data.src_node_ids.max(), full_data.dst_node_ids.max()))
        node_location_ids, location_to_id = build_googlemap_location_ids(
            entity_text_df=entity_text_df,
            max_node_id=max_node_id,
        )
        covered_location_nodes = int(np.sum(node_location_ids >= 0))
        if cmd_args.baseline == 'personalized_location' and covered_location_nodes == 0:
            raise ValueError(
                "Personalized location could not parse any city/state metadata. "
                "This baseline is currently intended for Googlemap-style entity_text.csv."
            )
        if covered_location_nodes > 0:
            print(
                "\nLoaded personalized-location metadata: "
                f"{covered_location_nodes:,} nodes with {len(location_to_id):,} city/state values."
            )

    if semantic_requested_by_baseline and entity_texts is not None and cmd_args.semantic_mode != 'off':
        print(f"\nLoading entity texts for semantic similarity from: {entity_text_path}")
        print(f"✓ Loaded {len(entity_texts)} entity texts")
        if cmd_args.semantic_entity_name_mode != 'raw':
            entity_texts = build_prompt_entity_map(
                cmd_args.dataset_name,
                entity_texts,
                entity_name_mode=cmd_args.semantic_entity_name_mode,
            )
            print(
                "✓ Applied prompt-style entity cleaning/compression before semantic embedding "
                f"(mode={cmd_args.semantic_entity_name_mode})"
            )
        graph_node_ids = set(full_data.src_node_ids).union(set(full_data.dst_node_ids))
        covered_graph_nodes = len(graph_node_ids.intersection(set(int(x) for x in entity_texts.keys())))
        if covered_graph_nodes < len(graph_node_ids):
            missing = len(graph_node_ids) - covered_graph_nodes
            print(
                f"Warning: entity-text file covers {covered_graph_nodes}/{len(graph_node_ids)} graph nodes. "
                f"{missing} nodes will fall back to missing semantic embeddings."
            )

        # Precompute or load cached semantic embeddings
        if cmd_args.embedding_cache and os.path.exists(cmd_args.embedding_cache):
            print(f"Loading base embeddings from cache: {cmd_args.embedding_cache}")
            embeddings = np.load(cmd_args.embedding_cache)
        else:
            embeddings, _ = precompute_entity_embeddings(
                entity_texts, model_name=cmd_args.embedding_model, device=semantic_device
            )
            if cmd_args.embedding_cache:
                np.save(cmd_args.embedding_cache, embeddings)
                print(f"Saved base embeddings to cache: {cmd_args.embedding_cache}")

        # Rebuild entity_id_to_idx (sorted order must match embeddings rows)
        entity_ids = sorted(entity_texts.keys())
        entity_id_to_idx_dict = {eid: idx for idx, eid in enumerate(entity_ids)}
        
        # Build Lookup Tensor for Fast GPU Scoring
        # Size = max(node_id) + 1
        max_node_id = max(full_data.src_node_ids.max(), full_data.dst_node_ids.max())
        lookup_tensor = torch.full((max_node_id + 1,), -1, dtype=torch.long)
        
        # Fill lookup
        # We need to map: node_id -> embedding_index
        # entity_ids contains the node IDs that have embeddings
        # The index in entity_ids corresponds to the row in `embeddings`
        
        # Optimizing fill:
        valid_nids = [eid for eid in entity_ids if eid <= max_node_id]
        valid_indices = [idx for idx, eid in enumerate(entity_ids) if eid <= max_node_id]
        
        lookup_tensor[valid_nids] = torch.tensor(valid_indices, dtype=torch.long)
        
        # Use the lookup tensor as the mapping
        entity_id_to_idx = lookup_tensor.to(semantic_device)

        raw_semantic_requested = cmd_args.baseline in ('semantic', 'all')
        smoothing_requested = cmd_args.baseline in ('semantic_smooth', 'all', 'rrf', 'hyperfusion')

        smoothing_input_embeddings = embeddings
        if (
            cmd_args.semantic_smoothing_source_init == 'history_mean'
            and smoothing_requested
        ):
            print(
                "\nSemantic smoothing init: replacing source/user node text embeddings "
                "with mean destination/item embeddings from the smoothing graph."
            )
            smoothing_input_embeddings, replaced_nodes = build_source_history_mean_initialized_embeddings(
                base_embeddings=embeddings,
                node_id_lookup=lookup_tensor.numpy(),
                src_node_ids=smooth_src,
                dst_node_ids=smooth_dst,
            )
            print(
                "✓ Semantic smoothing init updated "
                f"{replaced_nodes:,} source/user embeddings before smoothing."
            )

        # Only materialize the full embedding table on GPU for scorers that use it.
        # Yelp's e5 cache is ~8 GiB, so semantic_smooth-only runs should not keep
        # both raw and smoothing input embeddings resident on the GPU.
        smoothing_reuses_raw_embeddings = smoothing_input_embeddings is embeddings

        if raw_semantic_requested:
            embeddings = torch.from_numpy(embeddings).to(semantic_device)
        else:
            embeddings = None

        if smoothing_requested:
            if raw_semantic_requested and smoothing_reuses_raw_embeddings:
                smoothing_input_embeddings = embeddings
            else:
                smoothing_input_embeddings = torch.from_numpy(smoothing_input_embeddings).to(semantic_device)
        else:
            smoothing_input_embeddings = None

        if smoothing_requested:
            if cmd_args.smooth_full_resmooth_per_batch:
                print("\nSemantic Smooth mode: FULL RE-SMOOTH PER BATCH (performance-first, very slow).")
                semantic_smooth_scorer = FullResmoothPerBatchSemanticScorer(
                    embeddings=smoothing_input_embeddings,
                    entity_id_to_idx=entity_id_to_idx,
                    init_src_node_ids=smooth_src,
                    init_dst_node_ids=smooth_dst,
                    init_node_interact_times=smooth_times,
                    time_window=cmd_args.smooth_time_window,
                    num_steps=cmd_args.smooth_steps,
                    symmetric_norm=True,
                    decay_gamma=cmd_args.smooth_decay_gamma,
                    residual_alpha=cmd_args.smooth_residual_alpha,
                    undirected=cmd_args.smooth_undirected,
                    layer_norm=cmd_args.smooth_layer_norm,
                    pre_norm=cmd_args.smooth_pre_norm,
                    debug=cmd_args.smooth_debug,
                    log_dampen=cmd_args.smooth_log_dampen,
                    supernode_strength=cmd_args.smooth_supernode_strength,
                    hetero_highpass_coef=cmd_args.smooth_hetero_highpass_coef,
                    hetero_tau=cmd_args.smooth_hetero_tau,
                    hetero_filter_tau=cmd_args.smooth_hetero_filter_tau,
                    hetero_filter_keep_weight=cmd_args.smooth_hetero_filter_keep_weight,
                    endpoint_topk_recent=cmd_args.smooth_endpoint_topk_recent,
                    endpoint_topk_ensure_coverage=cmd_args.smooth_endpoint_topk_ensure_coverage,
                    score_fn=semantic_score_fn,
                    score_kwargs=semantic_score_kwargs,
                    device=semantic_device
                )
            else:
                if cmd_args.smoothed_embedding_cache and os.path.exists(cmd_args.smoothed_embedding_cache):
                    print(f"Loading smoothed embeddings from cache: {cmd_args.smoothed_embedding_cache}")
                    smoothed_embeddings = np.load(cmd_args.smoothed_embedding_cache)
                    # Convert loaded numpy to tensor
                    smoothed_embeddings = torch.from_numpy(smoothed_embeddings).to(semantic_device)
                else:
                    graph_type = "undirected" if cmd_args.smooth_undirected else "directed"
                    layer_norm_info = f", layer_norm={cmd_args.smooth_layer_norm}" if cmd_args.smooth_layer_norm else ""
                    pre_norm_info = f", pre_norm={cmd_args.smooth_pre_norm}" if cmd_args.smooth_pre_norm else ""
                    print(f"\nSmoothing embeddings with time-window GCN ({graph_type}{layer_norm_info}{pre_norm_info})...")
                    smoothed = smooth_embeddings_by_time_window_torch(
                        embeddings=smoothing_input_embeddings, # Already a tensor on GPU
                        src_node_ids=smooth_src,
                        dst_node_ids=smooth_dst,
                        node_interact_times=smooth_times,
                        time_window=cmd_args.smooth_time_window,
                        num_steps=cmd_args.smooth_steps,
                        symmetric_norm=True,
                        decay_gamma=cmd_args.smooth_decay_gamma,
                        residual_alpha=cmd_args.smooth_residual_alpha,
                        undirected=cmd_args.smooth_undirected,
                        layer_norm=cmd_args.smooth_layer_norm,
                        pre_norm=cmd_args.smooth_pre_norm,
                        device=semantic_device,
                        debug=cmd_args.smooth_debug,
                        log_dampen=cmd_args.smooth_log_dampen,
                        supernode_strength=cmd_args.smooth_supernode_strength,
                        hetero_highpass_coef=cmd_args.smooth_hetero_highpass_coef,
                        hetero_tau=cmd_args.smooth_hetero_tau,
                        hetero_filter_tau=cmd_args.smooth_hetero_filter_tau,
                        hetero_filter_keep_weight=cmd_args.smooth_hetero_filter_keep_weight,
                        endpoint_topk_recent=cmd_args.smooth_endpoint_topk_recent,
                        endpoint_topk_ensure_coverage=cmd_args.smooth_endpoint_topk_ensure_coverage
                    )
                    smoothed_embeddings = smoothed # Already on GPU

                    # Normalize
                    norms = torch.norm(smoothed_embeddings, dim=1, keepdim=True)
                    norms[norms == 0] = 1.0
                    smoothed_embeddings = smoothed_embeddings / norms

                    if cmd_args.smoothed_embedding_cache:
                        np.save(cmd_args.smoothed_embedding_cache, smoothed_embeddings.cpu().numpy())
                        print(f"Saved smoothed embeddings to cache: {cmd_args.smoothed_embedding_cache}")

    # Select baselines to evaluate
    # Recency scorer (optionally directed)
    recency_name = "Recency (directed)" if cmd_args.recency_directed else "Recency (undirected)"
    recency_scorer = partial(
        score_links_by_recency,
        directed=cmd_args.recency_directed,
        directed_src_node_ids=full_data.src_node_ids,
        directed_dst_node_ids=full_data.dst_node_ids,
        directed_node_interact_times=full_data.node_interact_times
    )

    # Create popularity scorer with decay parameter
    popularity_mode = 'sum' if cmd_args.popularity_include_source else 'target'
    popularity_scorer = partial(score_links_by_popularity, decay=cmd_args.decay, mode=popularity_mode)
    recent_degree_scorer = partial(
        score_links_by_recent_degree,
        window=cmd_args.recent_degree_window,
        mode=popularity_mode,
    )
    itemcf_cosine_scorer = partial(
        score_links_by_itemcf_cosine,
        directed_src_node_ids=full_data.src_node_ids,
        directed_dst_node_ids=full_data.dst_node_ids,
        directed_node_interact_times=full_data.node_interact_times,
    )
    markov_transition_scorer = partial(
        score_links_by_markov_transition,
        directed_src_node_ids=full_data.src_node_ids,
        directed_dst_node_ids=full_data.dst_node_ids,
        directed_node_interact_times=full_data.node_interact_times,
    )
    usercf_cosine_scorer = partial(
        score_links_by_usercf_cosine,
        directed_src_node_ids=full_data.src_node_ids,
        directed_dst_node_ids=full_data.dst_node_ids,
        directed_node_interact_times=full_data.node_interact_times,
    )
    personalized_location_scorer = None
    if node_location_ids is not None and int(np.sum(node_location_ids >= 0)) > 0:
        personalized_location_scorer = partial(
            score_links_by_personalized_location,
            directed_src_node_ids=full_data.src_node_ids,
            directed_dst_node_ids=full_data.dst_node_ids,
            directed_node_interact_times=full_data.node_interact_times,
            node_location_ids=node_location_ids,
        )

    # Add info to name
    popularity_base = "Popularity (src+dst)" if cmd_args.popularity_include_source else "Popularity (dst only)"
    popularity_name = f"{popularity_base}, decay={cmd_args.decay}" if cmd_args.decay > 0 else popularity_base
    recent_degree_base = "Recent Degree (src+dst)" if cmd_args.popularity_include_source else "Recent Degree (dst only)"
    recent_degree_name = f"{recent_degree_base}, window={cmd_args.recent_degree_window:g}"

    # Create CN/AA/RA scorers with mode parameter
    cn_scorer = partial(score_links_by_common_neighbors, mode='cn', use_gpu=cmd_args.use_gpu_heuristics)
    aa_scorer = partial(score_links_by_common_neighbors, mode='aa', use_gpu=cmd_args.use_gpu_heuristics)
    ra_scorer = partial(score_links_by_common_neighbors, mode='ra', use_gpu=cmd_args.use_gpu_heuristics)

    # Create semantic scorers
    semantic_scorer = partial(
        semantic_score_fn,
        embeddings=embeddings,
        entity_id_to_idx=entity_id_to_idx,
        **semantic_score_kwargs,
    ) if embeddings is not None else None
    if semantic_smooth_scorer is None and smoothed_embeddings is not None:
        semantic_smooth_scorer = partial(
            semantic_score_fn,
            embeddings=smoothed_embeddings,
            entity_id_to_idx=entity_id_to_idx,
            **semantic_score_kwargs,
        )

    available_baselines = {
        'recency': (recency_name, recency_scorer),
        'popularity': (popularity_name, popularity_scorer),
        'recent_degree': (recent_degree_name, recent_degree_scorer),
        'past_interactions': ("Past Interactions", score_links_by_past_interactions),
        'global_recency': ("Global Recency", score_links_by_global_recency),
        'itemcf_cosine': ("ItemCF Cosine", itemcf_cosine_scorer),
        'markov_transition': ("Markov Transition (last timestamp basket)", markov_transition_scorer),
        'usercf_cosine': ("UserCF Cosine", usercf_cosine_scorer),
        'cn': ("Common Neighbors (CN)", cn_scorer),
        'aa': ("Adamic-Adar (AA)", aa_scorer),
        'ra': ("Resource Allocation (RA)", ra_scorer)
    }
    if personalized_location_scorer is not None:
        available_baselines['personalized_location'] = (
            "Personalized Location (city/state history frequency)",
            personalized_location_scorer,
        )

    if semantic_scorer is not None:
        semantic_name = f"Semantic Similarity ({embedding_model_label})"
        if cmd_args.semantic_source_profile == 'history_mean':
            semantic_name += " [src=history_mean]"
        elif cmd_args.semantic_source_profile == 'query_conditioned_mean':
            semantic_name += f" [src=query_conditioned_mean,temp={cmd_args.semantic_query_temperature:g}]"
        available_baselines['semantic'] = (semantic_name, semantic_scorer)
    if semantic_smooth_scorer is not None:
        graph_type = "undirected" if cmd_args.smooth_undirected else "directed"
        smooth_name = f"Semantic Similarity ({embedding_model_label}, Smoothed, {graph_type}, window={cmd_args.smooth_time_window}, steps={cmd_args.smooth_steps}"
        if cmd_args.smooth_layer_norm:
            smooth_name += f", layer_norm={cmd_args.smooth_layer_norm}"
        if cmd_args.smooth_pre_norm:
            smooth_name += f", pre_norm={cmd_args.smooth_pre_norm}"
        if cmd_args.smooth_residual_alpha > 0:
            smooth_name += f", residual_alpha={cmd_args.smooth_residual_alpha}"
        if cmd_args.smooth_log_dampen:
            smooth_name += ", log_dampen=True"
        if cmd_args.smooth_supernode_strength > 0:
            smooth_name += f", supernode={cmd_args.smooth_supernode_strength}"
        if cmd_args.smooth_hetero_highpass_coef > 0:
            smooth_name += f", hetero_hp={cmd_args.smooth_hetero_highpass_coef}, tau={cmd_args.smooth_hetero_tau}"
        if cmd_args.smooth_hetero_filter_tau is not None and cmd_args.smooth_hetero_filter_keep_weight < 1.0:
            smooth_name += f", hetero_filter_tau={cmd_args.smooth_hetero_filter_tau}, keep={cmd_args.smooth_hetero_filter_keep_weight}"
        if cmd_args.smooth_endpoint_topk_recent is not None and cmd_args.smooth_endpoint_topk_recent > 0:
            smooth_name += f", endpoint_topk={cmd_args.smooth_endpoint_topk_recent}"
            if cmd_args.smooth_endpoint_topk_ensure_coverage:
                smooth_name += ", endpoint_cover=True"
        if cmd_args.smooth_full_resmooth_per_batch:
            smooth_name += ", full_resmooth_per_batch=True"
        smooth_name += ")"
        if cmd_args.semantic_source_profile == 'history_mean':
            smooth_name += " [src=history_mean]"
        elif cmd_args.semantic_source_profile == 'query_conditioned_mean':
            smooth_name += f" [src=query_conditioned_mean,temp={cmd_args.semantic_query_temperature:g}]"
        available_baselines['semantic_smooth'] = (smooth_name, semantic_smooth_scorer)

    baselines = []
    if cmd_args.baseline == 'all':
        baselines = [(name, scorer, None) for name, scorer in available_baselines.values()]
    elif cmd_args.baseline in ['rrf', 'hyperfusion']:
        components = []
        if 'recency' in available_baselines: components.append(available_baselines['recency'][1])
        if 'popularity' in available_baselines: components.append(available_baselines['popularity'][1])
        if 'past_interactions' in available_baselines: components.append(available_baselines['past_interactions'][1])
        if 'ra' in available_baselines: components.append(available_baselines['ra'][1])
        if 'semantic_smooth' in available_baselines:
            components.append(available_baselines['semantic_smooth'][1])
        else:
            print("Warning: Semantic Smooth scorer not available (maybe no cache/embeddings loaded?), skipping for fusion.")

        fusion_label = "HyperFusion" if cmd_args.baseline == 'hyperfusion' else "RRF"
        if cmd_args.baseline == 'hyperfusion':
            fusion_strategy = 'hyperfusion'
            fusion_name = f"{fusion_label} Fusion of {len(components)} heuristics"
        else:
            fusion_strategy = 'rrf' if cmd_args.rrf_mode == 'local' else 'rrf_eval_batch'
            mode_suffix = "local" if cmd_args.rrf_mode == 'local' else 'eval-batch'
            fusion_name = f"{fusion_label} ({mode_suffix}) Fusion of {len(components)} heuristics"
        baselines = [(fusion_name, components, fusion_strategy)]
    else:
        name, scorer = available_baselines[cmd_args.baseline]
        baselines = [(name, scorer, None)]

    print(f"\nEvaluating: {', '.join([name for name, _, _ in baselines])}")

    all_results = {}

    for baseline_name, scoring_function, fusion_strategy in baselines:
        print("\n" + "="*80)
        print(f"{baseline_name.upper()} BASELINE")
        print("="*80)

        # Optimization: If metric is ONLY ap_auc, we only need 1 negative per positive
        # This makes evaluation 100x faster (1 vs 1 instead of 1 vs 100)
        eval_num_negatives = 1 if cmd_args.metric == 'ap_auc' else cmd_args.num_negatives

        # Transductive test
        print(f"\nTransductive Test:")
        start = time.time()
        for online_scorer in _iter_online_scorers(scoring_function):
            online_scorer.reset_history()
        
        # Always use combined evaluator which supports RRF (list of scorers)
        results = evaluate_heuristic_combined(
            scoring_function=scoring_function,
            neighbor_sampler=full_neighbor_sampler,
            evaluate_idx_data_loader=test_idx_data_loader,
            evaluate_neg_edge_sampler=test_neg_edge_sampler,
            evaluate_data=test_data,
            num_negatives=eval_num_negatives,
            fusion_strategy=fusion_strategy,
            max_eval_batches=(cmd_args.max_eval_batches if cmd_args.max_eval_batches > 0 else None)
        )
        test_metrics_dict = results
        
        if cmd_args.metric in ['ap_auc', 'all']:
            print(f"  AP:  {results['average_precision']:.4f}")
            print(f"  AUC: {results['roc_auc']:.4f}")
        if cmd_args.metric in ['mrr', 'all']:
            print(f"  MRR: {results['mrr']:.4f}")

        test_time = time.time() - start
        print(f"  Time: {test_time:.2f}s")

        # Inductive test
        print(f"\nInductive Test (New Nodes):")
        start = time.time()
        for online_scorer in _iter_online_scorers(scoring_function):
            online_scorer.reset_history()
        
        results = evaluate_heuristic_combined(
            scoring_function=scoring_function,
            neighbor_sampler=full_neighbor_sampler,
            evaluate_idx_data_loader=new_node_test_idx_data_loader,
            evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
            evaluate_data=new_node_test_data,
            num_negatives=eval_num_negatives,
            fusion_strategy=fusion_strategy,
            max_eval_batches=(cmd_args.max_eval_batches if cmd_args.max_eval_batches > 0 else None)
        )
        new_node_test_metrics_dict = results
        
        if cmd_args.metric in ['ap_auc', 'all']:
            print(f"  AP:  {results['average_precision']:.4f}")
            print(f"  AUC: {results['roc_auc']:.4f}")
        if cmd_args.metric in ['mrr', 'all']:
            print(f"  MRR: {results['mrr']:.4f}")

        new_node_test_time = time.time() - start
        print(f"  Time: {new_node_test_time:.2f}s")

        # Store results
        all_results[baseline_name] = {
            'transductive': test_metrics_dict,
            'inductive': new_node_test_metrics_dict
        }

    # Final comparison
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    if cmd_args.metric in ['mrr', 'all']:
        print(f"\n{'Baseline':<15} {'Transductive AP':<18} {'Transductive MRR':<18} {'Inductive AP':<15} {'Inductive MRR':<15}")
        print("-" * 80)
        for baseline_name, _, _ in baselines:
            trans_ap = all_results[baseline_name]['transductive']['average_precision']
            trans_mrr = all_results[baseline_name]['transductive']['mrr']
            ind_ap = all_results[baseline_name]['inductive']['average_precision']
            ind_mrr = all_results[baseline_name]['inductive']['mrr']
            print(f"{baseline_name:<15} {trans_ap:<18.4f} {trans_mrr:<18.4f} {ind_ap:<15.4f} {ind_mrr:<15.4f}")
    else:
        print(f"\n{'Baseline':<15} {'Transductive AP':<18} {'Transductive AUC':<18} {'Inductive AP':<15} {'Inductive AUC':<15}")
        print("-" * 80)
        for baseline_name, _, _ in baselines:
            trans_ap = all_results[baseline_name]['transductive']['average_precision']
            trans_auc = all_results[baseline_name]['transductive']['roc_auc']
            ind_ap = all_results[baseline_name]['inductive']['average_precision']
            ind_auc = all_results[baseline_name]['inductive']['roc_auc']
            print(f"{baseline_name:<15} {trans_ap:<18.4f} {trans_auc:<18.4f} {ind_ap:<15.4f} {ind_auc:<15.4f}")

    print("="*80)


if __name__ == "__main__":
    main()
