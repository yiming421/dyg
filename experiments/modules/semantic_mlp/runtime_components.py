from __future__ import annotations

import time
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from torch_sparse import SparseTensor
    from torch_sparse.matmul import spmm_add
except ImportError:  # pragma: no cover
    SparseTensor = None
    spmm_add = None

from experiments.modules.semantic_mlp.graph_components import (
    _build_neighbor_sparse,
    _spmoverlap,
)
from utils.utils import NegativeEdgeSampler

def _negative_precompute_debug_log(desc: str, message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[NegPrecomputeDebug][{ts}][{desc}] {message}", flush=True)


def _profile_stage_start(device: Optional[torch.device], enabled: bool) -> Optional[float]:
    if not enabled:
        return None
    if device is not None and device.type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _profile_stage_elapsed(start_t: Optional[float], device: Optional[torch.device]) -> float:
    if start_t is None:
        return 0.0
    if device is not None and device.type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter() - start_t

def sample_negatives(
    neg_sampler: NegativeEdgeSampler,
    src: np.ndarray,
    dst: np.ndarray,
    times: np.ndarray,
    num_negatives: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if num_negatives < 1:
        raise ValueError("num_negatives must be >= 1")

    repeated_src = np.repeat(src, num_negatives)
    repeated_dst = np.repeat(dst, num_negatives)

    if neg_sampler.negative_sample_strategy != 'random':
        batch_start_time = float(np.min(times))
        batch_end_time = float(np.max(times))
        neg_src, neg_dst = neg_sampler.sample(
            size=len(repeated_src),
            batch_src_node_ids=repeated_src,
            batch_dst_node_ids=repeated_dst,
            current_batch_start_time=batch_start_time,
            current_batch_end_time=batch_end_time,
        )
    else:
        _, neg_dst = neg_sampler.sample(size=len(repeated_src))
        neg_src = repeated_src

    return neg_src, neg_dst


def sample_negatives_rand_hist_ratio(
    neg_sampler_random: NegativeEdgeSampler,
    neg_sampler_historical: NegativeEdgeSampler,
    src: np.ndarray,
    dst: np.ndarray,
    times: np.ndarray,
    num_negatives: int,
    rand_ratio: float,
    strict_historical_gap: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Training-only mixed negative sampling.
    rand_ratio = fraction of random negatives, (1-rand_ratio) = historical negatives.
    """
    if num_negatives < 1:
        raise ValueError("num_negatives must be >= 1")
    if not (0.0 <= rand_ratio <= 1.0):
        raise ValueError("rand_ratio must be in [0, 1]")

    repeated_src = np.repeat(src, num_negatives)
    repeated_dst = np.repeat(dst, num_negatives)
    total = len(repeated_src)

    num_rand = int(round(total * rand_ratio))
    num_rand = max(0, min(num_rand, total))
    num_hist = total - num_rand

    perm = np.random.permutation(total)
    rand_idx = perm[:num_rand]
    hist_idx = perm[num_rand:]

    neg_src = np.empty(total, dtype=np.int64)
    neg_dst = np.empty(total, dtype=np.int64)

    if num_rand > 0:
        _, rand_dst = neg_sampler_random.sample(size=num_rand)
        neg_src[rand_idx] = repeated_src[rand_idx]
        neg_dst[rand_idx] = rand_dst.astype(np.int64, copy=False)

    if num_hist > 0:
        batch_start_time = float(np.min(times))
        batch_end_time = float(np.max(times))
        hist_src, hist_dst = neg_sampler_historical.sample(
            size=num_hist,
            batch_src_node_ids=repeated_src[hist_idx],
            batch_dst_node_ids=repeated_dst[hist_idx],
            current_batch_start_time=batch_start_time,
            current_batch_end_time=batch_end_time,
            allow_random_fallback=(not strict_historical_gap),
        )
        hist_src = hist_src.astype(np.int64, copy=False)
        hist_dst = hist_dst.astype(np.int64, copy=False)
        hist_filled = np.zeros(num_hist, dtype=bool)
        src_slots = {}
        src_slot_ptr = {}
        for rel_pos, global_idx in enumerate(hist_idx):
            src_val = int(repeated_src[global_idx])
            if src_val not in src_slots:
                src_slots[src_val] = []
                src_slot_ptr[src_val] = 0
            src_slots[src_val].append(rel_pos)

        for src_val, dst_val in zip(hist_src, hist_dst):
            src_i = int(src_val)
            if src_i not in src_slots:
                continue
            ptr = src_slot_ptr[src_i]
            slots = src_slots[src_i]
            if ptr >= len(slots):
                continue
            rel_pos = slots[ptr]
            src_slot_ptr[src_i] = ptr + 1
            global_idx = hist_idx[rel_pos]
            neg_src[global_idx] = repeated_src[global_idx]
            neg_dst[global_idx] = int(dst_val)
            hist_filled[rel_pos] = True

        # In strict mode, historical sampler may return fewer usable samples.
        # Reassign missing quota to regular random negatives (source-conditioned).
        missing = int((~hist_filled).sum())
        if missing > 0:
            miss_idx = hist_idx[~hist_filled]
            _, miss_rand_dst = neg_sampler_random.sample(size=missing)
            neg_src[miss_idx] = repeated_src[miss_idx]
            neg_dst[miss_idx] = miss_rand_dst.astype(np.int64, copy=False)

    return neg_src, neg_dst


def build_precomputed_negative_queries(
    data_loader,
    data_source,
    neg_sampler: NegativeEdgeSampler,
    num_negatives: int,
    desc: str,
    debug: bool = False,
    debug_every: int = 25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    neg_sampler.reset_random_state()
    total = len(data_source.src_node_ids) * int(num_negatives)
    neg_src_all = np.empty(total, dtype=np.int64)
    neg_dst_all = np.empty(total, dtype=np.int64)
    neg_times_all = np.empty(total, dtype=np.float64)
    cursor = 0

    if debug:
        total_batches = len(data_loader)
        debug_every = max(1, int(debug_every))
        progress = tqdm(total=total_batches, desc=desc, ncols=100)
        iterator = iter(data_loader)
        loop_start = time.perf_counter()
        _negative_precompute_debug_log(
            desc,
            f"start total_batches={total_batches}, total_queries={total}, num_negatives={num_negatives}",
        )

        for batch_id in range(total_batches):
            should_log = batch_id < 5 or ((batch_id + 1) % debug_every == 0) or (batch_id + 1 == total_batches)
            fetch_t0 = time.perf_counter()
            if should_log:
                _negative_precompute_debug_log(
                    desc,
                    f"waiting_for_batch batch={batch_id + 1}/{total_batches} cursor={cursor}",
                )
            batch_indices = next(iterator)
            fetch_elapsed = time.perf_counter() - fetch_t0
            if should_log:
                _negative_precompute_debug_log(
                    desc,
                    f"fetched_batch batch={batch_id + 1}/{total_batches} fetch_s={fetch_elapsed:.4f}",
                )
            batch_indices = batch_indices.numpy()
            src = data_source.src_node_ids[batch_indices]
            dst = data_source.dst_node_ids[batch_indices]
            times = data_source.node_interact_times[batch_indices]
            sample_t0 = time.perf_counter()
            if should_log:
                _negative_precompute_debug_log(
                    desc,
                    f"sampling_negatives batch={batch_id + 1}/{total_batches} batch_size={len(src)}",
                )
            neg_src, neg_dst = sample_negatives(
                neg_sampler=neg_sampler,
                src=src,
                dst=dst,
                times=times,
                num_negatives=num_negatives,
            )
            sample_elapsed = time.perf_counter() - sample_t0
            if should_log:
                _negative_precompute_debug_log(
                    desc,
                    f"sampled_negatives batch={batch_id + 1}/{total_batches} sample_s={sample_elapsed:.4f}",
                )
            batch_total = len(neg_src)
            write_t0 = time.perf_counter()
            neg_src_all[cursor:cursor + batch_total] = neg_src
            neg_dst_all[cursor:cursor + batch_total] = neg_dst
            neg_times_all[cursor:cursor + batch_total] = np.repeat(times, num_negatives)
            cursor += batch_total
            write_elapsed = time.perf_counter() - write_t0
            if should_log:
                _negative_precompute_debug_log(
                    desc,
                    f"stored_batch batch={batch_id + 1}/{total_batches} batch_total={batch_total} "
                    f"write_s={write_elapsed:.4f} total_s={time.perf_counter() - loop_start:.4f}",
                )
            progress.update(1)

        progress.close()
        _negative_precompute_debug_log(desc, f"done total_s={time.perf_counter() - loop_start:.4f}")
        return neg_src_all, neg_dst_all, neg_times_all

    for batch_indices in tqdm(data_loader, desc=desc, ncols=100):
        batch_indices = batch_indices.numpy()
        src = data_source.src_node_ids[batch_indices]
        dst = data_source.dst_node_ids[batch_indices]
        times = data_source.node_interact_times[batch_indices]
        neg_src, neg_dst = sample_negatives(
            neg_sampler=neg_sampler,
            src=src,
            dst=dst,
            times=times,
            num_negatives=num_negatives,
        )
        batch_total = len(neg_src)
        neg_src_all[cursor:cursor + batch_total] = neg_src
        neg_dst_all[cursor:cursor + batch_total] = neg_dst
        neg_times_all[cursor:cursor + batch_total] = np.repeat(times, num_negatives)
        cursor += batch_total

    return neg_src_all, neg_dst_all, neg_times_all


def build_precomputed_train_negative_pool(
    train_loader,
    train_data,
    train_num_negatives: int,
    train_neg_sampler: Optional[NegativeEdgeSampler] = None,
    train_neg_sampler_random: Optional[NegativeEdgeSampler] = None,
    train_neg_sampler_historical: Optional[NegativeEdgeSampler] = None,
    train_rand_ratio: Optional[float] = None,
    historical_strict_gap: bool = False,
    desc: str = "Sample train negatives",
    debug: bool = False,
    debug_every: int = 25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = len(train_data.src_node_ids) * int(train_num_negatives)
    neg_src_all = np.empty(total, dtype=np.int64)
    neg_dst_all = np.empty(total, dtype=np.int64)
    neg_times_all = np.empty(total, dtype=np.float64)
    cursor = 0

    if debug:
        total_batches = len(train_loader)
        debug_every = max(1, int(debug_every))
        progress = tqdm(total=total_batches, desc=desc, ncols=100)
        iterator = iter(train_loader)
        loop_start = time.perf_counter()
        _negative_precompute_debug_log(
            desc,
            f"start total_batches={total_batches}, total_queries={total}, train_num_negatives={train_num_negatives}",
        )

        for batch_id in range(total_batches):
            should_log = batch_id < 5 or ((batch_id + 1) % debug_every == 0) or (batch_id + 1 == total_batches)
            fetch_t0 = time.perf_counter()
            if should_log:
                _negative_precompute_debug_log(
                    desc,
                    f"waiting_for_batch batch={batch_id + 1}/{total_batches} cursor={cursor}",
                )
            batch_indices = next(iterator)
            fetch_elapsed = time.perf_counter() - fetch_t0
            if should_log:
                _negative_precompute_debug_log(
                    desc,
                    f"fetched_batch batch={batch_id + 1}/{total_batches} fetch_s={fetch_elapsed:.4f}",
                )
            batch_indices = batch_indices.numpy()
            src = train_data.src_node_ids[batch_indices]
            dst = train_data.dst_node_ids[batch_indices]
            times = train_data.node_interact_times[batch_indices]
            sample_t0 = time.perf_counter()
            if should_log:
                _negative_precompute_debug_log(
                    desc,
                    f"sampling_negatives batch={batch_id + 1}/{total_batches} batch_size={len(src)}",
                )

            if train_rand_ratio is None:
                if train_neg_sampler is None:
                    raise ValueError("train_neg_sampler is required when train_rand_ratio is None.")
                neg_src, neg_dst = sample_negatives(
                    neg_sampler=train_neg_sampler,
                    src=src,
                    dst=dst,
                    times=times,
                    num_negatives=train_num_negatives,
                )
            else:
                if train_neg_sampler_random is None or train_neg_sampler_historical is None:
                    raise ValueError(
                        "train_neg_sampler_random and train_neg_sampler_historical are required "
                        "when train_rand_ratio is set."
                    )
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

            sample_elapsed = time.perf_counter() - sample_t0
            if should_log:
                _negative_precompute_debug_log(
                    desc,
                    f"sampled_negatives batch={batch_id + 1}/{total_batches} sample_s={sample_elapsed:.4f}",
                )
            batch_total = len(neg_src)
            write_t0 = time.perf_counter()
            neg_src_all[cursor:cursor + batch_total] = neg_src
            neg_dst_all[cursor:cursor + batch_total] = neg_dst
            neg_times_all[cursor:cursor + batch_total] = np.repeat(times, train_num_negatives)
            cursor += batch_total
            write_elapsed = time.perf_counter() - write_t0
            if should_log:
                _negative_precompute_debug_log(
                    desc,
                    f"stored_batch batch={batch_id + 1}/{total_batches} batch_total={batch_total} "
                    f"write_s={write_elapsed:.4f} total_s={time.perf_counter() - loop_start:.4f}",
                )
            progress.update(1)

        progress.close()
        _negative_precompute_debug_log(desc, f"done total_s={time.perf_counter() - loop_start:.4f}")
        return neg_src_all, neg_dst_all, neg_times_all

    for batch_indices in tqdm(train_loader, desc=desc, ncols=100):
        batch_indices = batch_indices.numpy()
        src = train_data.src_node_ids[batch_indices]
        dst = train_data.dst_node_ids[batch_indices]
        times = train_data.node_interact_times[batch_indices]

        if train_rand_ratio is None:
            if train_neg_sampler is None:
                raise ValueError("train_neg_sampler is required when train_rand_ratio is None.")
            neg_src, neg_dst = sample_negatives(
                neg_sampler=train_neg_sampler,
                src=src,
                dst=dst,
                times=times,
                num_negatives=train_num_negatives,
            )
        else:
            if train_neg_sampler_random is None or train_neg_sampler_historical is None:
                raise ValueError(
                    "train_neg_sampler_random and train_neg_sampler_historical are required "
                    "when train_rand_ratio is set."
                )
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

        batch_total = len(neg_src)
        neg_src_all[cursor:cursor + batch_total] = neg_src
        neg_dst_all[cursor:cursor + batch_total] = neg_dst
        neg_times_all[cursor:cursor + batch_total] = np.repeat(times, train_num_negatives)
        cursor += batch_total

    return neg_src_all, neg_dst_all, neg_times_all

def score_edge_batch(
    model: nn.Module,
    embeddings: torch.Tensor,
    lookup: torch.Tensor,
    src_ids: Union[np.ndarray, torch.Tensor],
    dst_ids: Union[np.ndarray, torch.Tensor],
    auxiliary_features: Optional[torch.Tensor] = None,
    node_interact_times: Optional[Union[np.ndarray, torch.Tensor]] = None,
    neighbor_index: Optional[TemporalNeighborIndex] = None,
    ncn_adj: Optional[SparseTensor] = None,
    cross_attn_num_neighbors: int = 20,
    ncn_num_neighbors: int = 50,
    seqfilter_num_neighbors: int = 32,
    default_value: float = 0.0,
    profile_out: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        scores: [N] logits
        valid_mask: [N] bool, whether both endpoints have semantic embeddings
    """
    device = embeddings.device

    if torch.is_tensor(src_ids):
        src = src_ids if src_ids.device == device and src_ids.dtype == torch.long else src_ids.to(device=device, dtype=torch.long)
    else:
        t_stage = _profile_stage_start(device, profile_out is not None)
        src = torch.from_numpy(np.asarray(src_ids, dtype=np.int64)).long().to(device)
        if profile_out is not None:
            profile_out['score_id_to_tensor_s'] = profile_out.get('score_id_to_tensor_s', 0.0) + _profile_stage_elapsed(t_stage, device)
    if torch.is_tensor(dst_ids):
        dst = dst_ids if dst_ids.device == device and dst_ids.dtype == torch.long else dst_ids.to(device=device, dtype=torch.long)
    else:
        t_stage = _profile_stage_start(device, profile_out is not None)
        dst = torch.from_numpy(np.asarray(dst_ids, dtype=np.int64)).long().to(device)
        if profile_out is not None:
            profile_out['score_id_to_tensor_s'] = profile_out.get('score_id_to_tensor_s', 0.0) + _profile_stage_elapsed(t_stage, device)

    t_stage = _profile_stage_start(device, profile_out is not None)
    max_id = lookup.size(0)
    in_range = (src >= 0) & (src < max_id) & (dst >= 0) & (dst < max_id)

    src_idx = torch.full_like(src, -1)
    dst_idx = torch.full_like(dst, -1)

    src_idx[in_range] = lookup[src[in_range]]
    dst_idx[in_range] = lookup[dst[in_range]]

    valid_mask = in_range & (src_idx >= 0) & (dst_idx >= 0)
    if profile_out is not None:
        profile_out['score_lookup_s'] = profile_out.get('score_lookup_s', 0.0) + _profile_stage_elapsed(t_stage, device)
        profile_out['score_valid_edges'] = profile_out.get('score_valid_edges', 0.0) + float(valid_mask.sum().item())

    scores = torch.full((int(src.numel()),), float(default_value), dtype=torch.float32, device=device)
    if valid_mask.any():
        if getattr(model, 'requires_pair_neighbor_context', False):
            if node_interact_times is None:
                raise ValueError("Pair-history scorer requires node_interact_times.")
            if neighbor_index is None:
                raise ValueError("Pair-history scorer requires neighbor_index.")

            valid_mask_np = valid_mask.detach().cpu().numpy()
            src_ids_np = (
                src_ids.detach().cpu().numpy()
                if torch.is_tensor(src_ids)
                else np.asarray(src_ids, dtype=np.int64)
            )
            dst_ids_np = (
                dst_ids.detach().cpu().numpy()
                if torch.is_tensor(dst_ids)
                else np.asarray(dst_ids, dtype=np.int64)
            )
            time_ids_np = (
                node_interact_times.detach().cpu().numpy()
                if torch.is_tensor(node_interact_times)
                else np.asarray(node_interact_times, dtype=np.float64)
            )
            valid_src_ids = src_ids_np[valid_mask_np]
            valid_dst_ids = dst_ids_np[valid_mask_np]
            valid_times = time_ids_np[valid_mask_np]
            num_neighbors = int(getattr(model, 'num_neighbors', seqfilter_num_neighbors))

            t_stage = _profile_stage_start(device, profile_out is not None)
            src_neigh_ids_np, src_neigh_times_np, src_neigh_mask_np = neighbor_index.get_recent_neighbors(
                node_ids=valid_src_ids,
                query_times=valid_times,
                num_neighbors=num_neighbors,
                return_times=True,
            )
            dst_neigh_ids_np, dst_neigh_times_np, dst_neigh_mask_np = neighbor_index.get_recent_neighbors(
                node_ids=valid_dst_ids,
                query_times=valid_times,
                num_neighbors=num_neighbors,
                return_times=True,
            )
            if profile_out is not None:
                profile_out['score_neighbor_fetch_s'] = profile_out.get('score_neighbor_fetch_s', 0.0) + _profile_stage_elapsed(t_stage, None)

            def _marshal_history(neigh_ids_np, neigh_times_np, neigh_mask_np):
                neigh_nodes = torch.from_numpy(neigh_ids_np).long().to(device)
                neigh_hist_mask = torch.from_numpy(neigh_mask_np).to(device=device, dtype=torch.bool)
                neigh_in_range = (neigh_nodes >= 0) & (neigh_nodes < max_id)
                neigh_idx = torch.full_like(neigh_nodes, -1)
                if neigh_in_range.any():
                    neigh_idx[neigh_in_range] = lookup[neigh_nodes[neigh_in_range]]
                neigh_valid = neigh_hist_mask & (neigh_idx >= 0)
                safe_neigh_idx = neigh_idx.clone()
                safe_neigh_idx[~neigh_valid] = 0
                history_emb = embeddings[safe_neigh_idx]
                history_emb = history_emb * neigh_valid.unsqueeze(-1).to(history_emb.dtype)
                history_times = torch.from_numpy(neigh_times_np).to(device=device, dtype=torch.float32)
                query_times_t = torch.from_numpy(valid_times).to(device=device, dtype=torch.float32).unsqueeze(1)
                history_delta_t = torch.clamp(query_times_t - history_times, min=0.0)
                history_delta_t = history_delta_t.masked_fill(~neigh_valid, 0.0)
                history_times = history_times.masked_fill(~neigh_valid, 0.0)
                return history_emb, neigh_valid, history_delta_t, history_times

            t_stage = _profile_stage_start(device, profile_out is not None)
            src_history_emb, src_history_mask, src_history_delta_t, src_history_times = _marshal_history(
                src_neigh_ids_np,
                src_neigh_times_np,
                src_neigh_mask_np,
            )
            dst_history_emb, dst_history_mask, dst_history_delta_t, dst_history_times = _marshal_history(
                dst_neigh_ids_np,
                dst_neigh_times_np,
                dst_neigh_mask_np,
            )
            if profile_out is not None:
                profile_out['score_neighbor_marshal_s'] = profile_out.get('score_neighbor_marshal_s', 0.0) + _profile_stage_elapsed(t_stage, device)

            t_stage = _profile_stage_start(device, profile_out is not None)
            src_valid_emb = embeddings[src_idx[valid_mask]]
            dst_valid_emb = embeddings[dst_idx[valid_mask]]
            if profile_out is not None:
                profile_out['score_gather_s'] = profile_out.get('score_gather_s', 0.0) + _profile_stage_elapsed(t_stage, device)

            t_stage = _profile_stage_start(device, profile_out is not None)
            scores[valid_mask] = model(
                src_emb=src_valid_emb,
                dst_emb=dst_valid_emb,
                src_history_emb=src_history_emb,
                src_history_mask=src_history_mask,
                src_history_delta_t=src_history_delta_t,
                src_history_times=src_history_times,
                dst_history_emb=dst_history_emb,
                dst_history_mask=dst_history_mask,
                dst_history_delta_t=dst_history_delta_t,
                dst_history_times=dst_history_times,
            )
            if profile_out is not None:
                profile_out['score_forward_s'] = profile_out.get('score_forward_s', 0.0) + _profile_stage_elapsed(t_stage, device)
        elif getattr(model, 'requires_neighbor_context', False):
            if node_interact_times is None:
                raise ValueError("Cross-attention scorer requires node_interact_times.")
            if neighbor_index is None:
                raise ValueError("Cross-attention scorer requires neighbor_index.")

            valid_mask_np = valid_mask.detach().cpu().numpy()
            src_ids_np = (
                src_ids.detach().cpu().numpy()
                if torch.is_tensor(src_ids)
                else np.asarray(src_ids, dtype=np.int64)
            )
            time_ids_np = (
                node_interact_times.detach().cpu().numpy()
                if torch.is_tensor(node_interact_times)
                else np.asarray(node_interact_times, dtype=np.float64)
            )
            valid_src_ids = src_ids_np[valid_mask_np]
            valid_times = time_ids_np[valid_mask_np]
            t_stage = _profile_stage_start(device, profile_out is not None)
            neigh_ids_np, neigh_times_np, neigh_mask_np = neighbor_index.get_recent_neighbors(
                node_ids=valid_src_ids,
                query_times=valid_times,
                num_neighbors=cross_attn_num_neighbors,
                return_times=True,
            )
            if profile_out is not None:
                profile_out['score_neighbor_fetch_s'] = profile_out.get('score_neighbor_fetch_s', 0.0) + _profile_stage_elapsed(t_stage, None)

            t_stage = _profile_stage_start(device, profile_out is not None)
            neigh_nodes = torch.from_numpy(neigh_ids_np).long().to(device)
            neigh_hist_mask = torch.from_numpy(neigh_mask_np).to(device=device, dtype=torch.bool)

            neigh_in_range = (neigh_nodes >= 0) & (neigh_nodes < max_id)
            neigh_idx = torch.full_like(neigh_nodes, -1)
            if neigh_in_range.any():
                neigh_idx[neigh_in_range] = lookup[neigh_nodes[neigh_in_range]]

            neigh_valid = neigh_hist_mask & (neigh_idx >= 0)
            safe_neigh_idx = neigh_idx.clone()
            safe_neigh_idx[~neigh_valid] = 0
            if profile_out is not None:
                profile_out['score_neighbor_marshal_s'] = profile_out.get('score_neighbor_marshal_s', 0.0) + _profile_stage_elapsed(t_stage, device)

            t_stage = _profile_stage_start(device, profile_out is not None)
            src_history_emb = embeddings[safe_neigh_idx]
            src_history_emb = src_history_emb * neigh_valid.unsqueeze(-1).to(src_history_emb.dtype)
            neigh_times_t = torch.from_numpy(neigh_times_np).to(device=device, dtype=torch.float32)
            query_times_t = torch.from_numpy(valid_times).to(device=device, dtype=torch.float32).unsqueeze(1)
            src_history_delta_t = torch.clamp(query_times_t - neigh_times_t, min=0.0)
            src_history_delta_t = src_history_delta_t.masked_fill(~neigh_valid, 0.0)
            dst_valid_emb = embeddings[dst_idx[valid_mask]]
            if profile_out is not None:
                profile_out['score_gather_s'] = profile_out.get('score_gather_s', 0.0) + _profile_stage_elapsed(t_stage, device)

            t_stage = _profile_stage_start(device, profile_out is not None)
            scores[valid_mask] = model(
                dst_emb=dst_valid_emb,
                src_history_emb=src_history_emb,
                src_history_mask=neigh_valid,
                src_history_delta_t=src_history_delta_t,
            )
            if profile_out is not None:
                profile_out['score_forward_s'] = profile_out.get('score_forward_s', 0.0) + _profile_stage_elapsed(t_stage, device)
        elif getattr(model, 'requires_common_neighbor_context', False):
            if SparseTensor is None or spmm_add is None:
                raise RuntimeError("NCN scorer requires torch_sparse to be installed.")

            t_stage = _profile_stage_start(device, profile_out is not None)
            src_valid_emb = embeddings[src_idx[valid_mask]]
            dst_valid_emb = embeddings[dst_idx[valid_mask]]
            valid_src_idx = src_idx[valid_mask]
            valid_dst_idx = dst_idx[valid_mask]
            if profile_out is not None:
                profile_out['score_gather_s'] = profile_out.get('score_gather_s', 0.0) + _profile_stage_elapsed(t_stage, device)
            if ncn_adj is not None:
                t_stage = _profile_stage_start(device, profile_out is not None)
                src_sparse = ncn_adj[valid_src_idx]
                dst_sparse = ncn_adj[valid_dst_idx]
                if profile_out is not None:
                    profile_out['ncn_row_slice_s'] = profile_out.get('ncn_row_slice_s', 0.0) + _profile_stage_elapsed(t_stage, device)
                t_stage = _profile_stage_start(device, profile_out is not None)
                cn_sparse = _spmoverlap(src_sparse, dst_sparse)
                if profile_out is not None:
                    profile_out['ncn_overlap_s'] = profile_out.get('ncn_overlap_s', 0.0) + _profile_stage_elapsed(t_stage, device)
            else:
                if node_interact_times is None:
                    raise ValueError("NCN scorer requires node_interact_times when ncn_adj is not provided.")
                if neighbor_index is None:
                    raise ValueError("NCN scorer requires neighbor_index when ncn_adj is not provided.")

                valid_mask_np = valid_mask.detach().cpu().numpy()
                src_ids_np = (
                    src_ids.detach().cpu().numpy()
                    if torch.is_tensor(src_ids)
                    else np.asarray(src_ids, dtype=np.int64)
                )
                dst_ids_np = (
                    dst_ids.detach().cpu().numpy()
                    if torch.is_tensor(dst_ids)
                    else np.asarray(dst_ids, dtype=np.int64)
                )
                time_ids_np = (
                    node_interact_times.detach().cpu().numpy()
                    if torch.is_tensor(node_interact_times)
                    else np.asarray(node_interact_times, dtype=np.float64)
                )
                valid_src_ids = src_ids_np[valid_mask_np]
                valid_dst_ids = dst_ids_np[valid_mask_np]
                valid_times = time_ids_np[valid_mask_np]

                t_stage = _profile_stage_start(device, profile_out is not None)
                src_neigh_ids_np, src_neigh_mask_np = neighbor_index.get_recent_neighbors(
                    node_ids=valid_src_ids,
                    query_times=valid_times,
                    num_neighbors=ncn_num_neighbors,
                    return_times=False,
                )
                dst_neigh_ids_np, dst_neigh_mask_np = neighbor_index.get_recent_neighbors(
                    node_ids=valid_dst_ids,
                    query_times=valid_times,
                    num_neighbors=ncn_num_neighbors,
                    return_times=False,
                )
                if profile_out is not None:
                    profile_out['ncn_history_fetch_s'] = profile_out.get('ncn_history_fetch_s', 0.0) + _profile_stage_elapsed(t_stage, None)

                t_stage = _profile_stage_start(device, profile_out is not None)
                src_neigh_nodes = torch.from_numpy(src_neigh_ids_np).long().to(device)
                dst_neigh_nodes = torch.from_numpy(dst_neigh_ids_np).long().to(device)
                src_neigh_mask = torch.from_numpy(src_neigh_mask_np).to(device=device, dtype=torch.bool)
                dst_neigh_mask = torch.from_numpy(dst_neigh_mask_np).to(device=device, dtype=torch.bool)

                src_neigh_in_range = (src_neigh_nodes >= 0) & (src_neigh_nodes < max_id)
                dst_neigh_in_range = (dst_neigh_nodes >= 0) & (dst_neigh_nodes < max_id)
                src_neigh_idx = torch.full_like(src_neigh_nodes, -1)
                dst_neigh_idx = torch.full_like(dst_neigh_nodes, -1)
                if src_neigh_in_range.any():
                    src_neigh_idx[src_neigh_in_range] = lookup[src_neigh_nodes[src_neigh_in_range]]
                if dst_neigh_in_range.any():
                    dst_neigh_idx[dst_neigh_in_range] = lookup[dst_neigh_nodes[dst_neigh_in_range]]

                src_valid_hist = src_neigh_mask & (src_neigh_idx >= 0)
                dst_valid_hist = dst_neigh_mask & (dst_neigh_idx >= 0)
                if profile_out is not None:
                    profile_out['ncn_marshal_s'] = profile_out.get('ncn_marshal_s', 0.0) + _profile_stage_elapsed(t_stage, device)

                t_stage = _profile_stage_start(device, profile_out is not None)
                src_sparse = _build_neighbor_sparse(src_neigh_idx, src_valid_hist, embeddings.size(0))
                dst_sparse = _build_neighbor_sparse(dst_neigh_idx, dst_valid_hist, embeddings.size(0))
                if profile_out is not None:
                    profile_out['ncn_sparse_build_s'] = profile_out.get('ncn_sparse_build_s', 0.0) + _profile_stage_elapsed(t_stage, device)
                t_stage = _profile_stage_start(device, profile_out is not None)
                cn_sparse = _spmoverlap(src_sparse, dst_sparse)
                if profile_out is not None:
                    profile_out['ncn_overlap_s'] = profile_out.get('ncn_overlap_s', 0.0) + _profile_stage_elapsed(t_stage, device)

            t_stage = _profile_stage_start(device, profile_out is not None)
            cn_emb = spmm_add(cn_sparse, embeddings)
            has_common = cn_sparse.sum(dim=-1).to_dense() > 0
            if profile_out is not None:
                profile_out['ncn_aggregate_s'] = profile_out.get('ncn_aggregate_s', 0.0) + _profile_stage_elapsed(t_stage, device)
                profile_out['ncn_edge_count'] = profile_out.get('ncn_edge_count', 0.0) + float(valid_src_idx.numel())

            t_stage = _profile_stage_start(device, profile_out is not None)
            scores[valid_mask] = model(
                src_emb=src_valid_emb,
                dst_emb=dst_valid_emb,
                cn_emb=cn_emb,
                has_common_neighbors=has_common,
            )
            if profile_out is not None:
                profile_out['score_forward_s'] = profile_out.get('score_forward_s', 0.0) + _profile_stage_elapsed(t_stage, device)
        else:
            t_stage = _profile_stage_start(device, profile_out is not None)
            src_valid_emb = embeddings[src_idx[valid_mask]]
            dst_valid_emb = embeddings[dst_idx[valid_mask]]
            valid_auxiliary_features = None
            if auxiliary_features is not None:
                valid_auxiliary_features = auxiliary_features.to(device=device)[valid_mask]
            if profile_out is not None:
                profile_out['score_gather_s'] = profile_out.get('score_gather_s', 0.0) + _profile_stage_elapsed(t_stage, device)
            t_stage = _profile_stage_start(device, profile_out is not None)
            scores[valid_mask] = model(
                src_valid_emb,
                dst_valid_emb,
                auxiliary_features=valid_auxiliary_features,
            )
            if profile_out is not None:
                profile_out['score_forward_s'] = profile_out.get('score_forward_s', 0.0) + _profile_stage_elapsed(t_stage, device)

    return scores, valid_mask
