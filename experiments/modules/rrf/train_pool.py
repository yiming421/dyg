"""
Train-pool builder and batched RRF engine.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from .common import group_indices_by_values
from .scoring import (
    HEURISTIC_RANK_KEYS,
    HEURISTIC_SCORE_KEYS,
    _compute_semantic_smoothing_scores_for_pairs,
    normalize_rrf_heuristics,
)
from .utils import descending_ranks, reciprocal_rank_fusion_from_ranks
from .train_pool_batch import TrainPoolBatch
from ..heuristic_models import (
    build_csr_from_neighbor_sampler,
    score_links_by_common_neighbors,
    score_links_by_global_recency,
    score_links_by_popularity,
)


def _as_int64_array(values):
    if values is None:
        return np.empty(0, dtype=np.int64)
    if isinstance(values, np.ndarray):
        return values.astype(np.int64, copy=False).reshape(-1)
    return np.asarray([int(x) for x in values], dtype=np.int64)


@dataclass(frozen=True)
class TrainPoolBuildConfig:
    num_pools: int = 4
    pool_size: int = 256
    random_seed: int = 42


@dataclass(frozen=True)
class TrainPoolRRFConfig:
    rrf_k: int = 60
    score_batch_size: int = 200000
    use_gpu_heuristics: bool = False
    rrf_heuristics: tuple[str, ...] | str | None = None
    semantic_context: dict | None = None


class TrainPoolBatchBuilder:
    """
    Build dense train-pool batches from evaluation samples and train priors.
    """

    def __init__(
        self,
        neighbor_sampler,
        train_entity_ids,
    ):
        self.neighbor_sampler = neighbor_sampler
        self.train_entity_ids = _as_int64_array(train_entity_ids)
        self._unseen_recency_score = float(-np.log1p(1e9))
        self._source_time_feature_cache = {}
        self._ensure_source_csr()

    def _ensure_source_csr(self):
        if hasattr(self.neighbor_sampler, '_csr_indptr'):
            self._source_indptr = self.neighbor_sampler._csr_indptr
            self._source_indices = self.neighbor_sampler._csr_indices
            self._source_times = self.neighbor_sampler._csr_times
            return

        indptr, indices, times = build_csr_from_neighbor_sampler(self.neighbor_sampler)
        self.neighbor_sampler._csr_indptr = indptr
        self.neighbor_sampler._csr_indices = indices
        self.neighbor_sampler._csr_times = times
        self._source_indptr = indptr
        self._source_indices = indices
        self._source_times = times

    def _append_candidates(self, selected, seen, src, candidates, need, rng):
        if need <= 0 or candidates.size == 0:
            return

        blocked_count = len(seen) + (0 if src in seen else 1)
        available = int(candidates.size) - blocked_count
        if available <= 0:
            return

        take = min(int(need), available)
        if candidates.size >= 1024 and take <= max(512, candidates.size // 4):
            picked = []
            picked_set = set()
            draw_size = min(int(candidates.size), max(32, take * 2))

            while len(picked) < take:
                draw_idx = rng.integers(0, int(candidates.size), size=draw_size)
                draw_vals = candidates[draw_idx]
                for node_id in draw_vals:
                    node_id = int(node_id)
                    if node_id == src or node_id in seen or node_id in picked_set:
                        continue
                    picked_set.add(node_id)
                    picked.append(node_id)
                    if len(picked) >= take:
                        break
                if len(picked) < take:
                    draw_size = min(int(candidates.size), draw_size * 2)
        else:
            blocked = np.fromiter(seen | {src}, dtype=np.int64, count=len(seen) + (0 if src in seen else 1))
            filtered = candidates[~np.isin(candidates, blocked, assume_unique=False)]
            if filtered.size == 0:
                return
            if filtered.size > take:
                pick_idx = rng.choice(filtered.size, size=take, replace=False)
                picked = filtered[pick_idx].tolist()
            else:
                picked = filtered.tolist()

        for node_id in picked:
            seen.add(node_id)
            selected.append(node_id)

    def _sample_pool_targets(self, sample, rng, pool_size):
        src = int(sample['source_id'])
        true_target = int(sample['target_id'])

        selected = [true_target]
        seen = {true_target}
        need = int(pool_size) - 1

        self._append_candidates(selected, seen, src, self.train_entity_ids, need, rng)

        return selected

    def _pool_seed(self, base_seed, sample_idx, pool_id):
        return int(base_seed) + int(sample_idx) * 1000003 + int(pool_id) * 9176

    def _filter_primary_candidates(self, src, true_target):
        candidates = self.train_entity_ids
        if candidates.size == 0:
            return candidates
        mask = candidates != int(src)
        if int(true_target) != int(src):
            mask &= candidates != int(true_target)
        return candidates[mask]

    def _sample_pool_rows(self, sample, sample_idx, num_pools, pool_size, base_seed):
        src = int(sample['source_id'])
        true_target = int(sample['target_id'])
        need = max(0, int(pool_size) - 1)

        pool_rows = np.full((num_pools, pool_size), -1, dtype=np.int64)
        pool_valid_mask = np.zeros((num_pools, pool_size), dtype=bool)
        pool_rows[:, 0] = true_target
        pool_valid_mask[:, 0] = True

        primary_candidates = self._filter_primary_candidates(src, true_target)
        if primary_candidates.size >= need:
            for pool_id in range(num_pools):
                if need <= 0:
                    continue
                seed = self._pool_seed(base_seed, sample_idx, pool_id)
                rng = np.random.default_rng(seed)
                pick_idx = rng.choice(primary_candidates.size, size=need, replace=False)
                picked = primary_candidates[pick_idx]
                pool_rows[pool_id, 1:1 + need] = picked
                pool_valid_mask[pool_id, :1 + need] = True
            return pool_rows, pool_valid_mask

        for pool_id in range(num_pools):
            seed = self._pool_seed(base_seed, sample_idx, pool_id)
            rng = np.random.default_rng(seed)
            selected = self._sample_pool_targets(sample, rng, pool_size)
            width = len(selected)
            pool_rows[pool_id, :width] = selected
            pool_valid_mask[pool_id, :width] = True
        return pool_rows, pool_valid_mask

    def _get_source_time_features(self, src, timestamp):
        key = (int(src), int(timestamp))
        cached = self._source_time_feature_cache.get(key)
        if cached is not None:
            return cached

        row_start = int(self._source_indptr[int(src)])
        row_end = int(self._source_indptr[int(src) + 1])
        if row_end <= row_start:
            cached = (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )
            self._source_time_feature_cache[key] = cached
            return cached

        row_times = self._source_times[row_start:row_end]
        cutoff = int(np.searchsorted(row_times, float(timestamp), side='left'))
        if cutoff <= 0:
            cached = (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )
            self._source_time_feature_cache[key] = cached
            return cached

        hist_targets = self._source_indices[row_start:row_start + cutoff]
        hist_times = self._source_times[row_start:row_start + cutoff]
        order = np.argsort(hist_targets, kind='mergesort')
        sorted_targets = hist_targets[order]
        sorted_times = hist_times[order]
        unique_targets, counts = np.unique(sorted_targets, return_counts=True)
        last_indices = np.cumsum(counts) - 1
        last_times = sorted_times[last_indices]

        cached = (
            unique_targets.astype(np.int64, copy=False),
            counts.astype(np.float64, copy=False),
            last_times.astype(np.float64, copy=False),
        )
        self._source_time_feature_cache[key] = cached
        return cached

    def _score_pool_targets_fast(self, src, timestamp, pool_targets):
        target_ids, counts, last_times = self._get_source_time_features(src, timestamp)
        pool_arr = np.asarray(pool_targets, dtype=np.int64)
        recency = np.full(pool_arr.shape[0], self._unseen_recency_score, dtype=np.float64)
        past = np.zeros(pool_arr.shape[0], dtype=np.float64)

        if target_ids.size == 0 or pool_arr.size == 0:
            return recency, past

        insert_pos = np.searchsorted(target_ids, pool_arr)
        valid = insert_pos < target_ids.size
        if not np.any(valid):
            return recency, past

        matched = np.zeros(pool_arr.shape[0], dtype=bool)
        matched[valid] = target_ids[insert_pos[valid]] == pool_arr[valid]
        if not np.any(matched):
            return recency, past

        matched_pos = insert_pos[matched]
        past[matched] = counts[matched_pos]
        age = np.maximum(float(timestamp) - last_times[matched_pos], 0.0)
        recency[matched] = -np.log1p(age)
        return recency, past

    def _score_pool_target_matrix_fast(self, src, timestamp, pool_targets, pool_valid_mask):
        recency = np.full(pool_targets.shape, self._unseen_recency_score, dtype=np.float64)
        past = np.zeros(pool_targets.shape, dtype=np.float64)

        if pool_targets.size == 0:
            return recency, past

        valid_pos = np.flatnonzero(pool_valid_mask.reshape(-1))
        if valid_pos.size == 0:
            recency[~pool_valid_mask] = -np.inf
            past[~pool_valid_mask] = -np.inf
            return recency, past

        target_ids, counts, last_times = self._get_source_time_features(src, timestamp)
        if target_ids.size == 0:
            recency[~pool_valid_mask] = -np.inf
            past[~pool_valid_mask] = -np.inf
            return recency, past

        flat_targets = pool_targets.reshape(-1)
        flat_recency = recency.reshape(-1)
        flat_past = past.reshape(-1)
        lookup_targets = flat_targets[valid_pos]
        insert_pos = np.searchsorted(target_ids, lookup_targets)
        within = insert_pos < target_ids.size
        if np.any(within):
            matched = np.zeros(valid_pos.shape[0], dtype=bool)
            matched[within] = target_ids[insert_pos[within]] == lookup_targets[within]
            if np.any(matched):
                matched_pos = insert_pos[matched]
                write_pos = valid_pos[matched]
                flat_past[write_pos] = counts[matched_pos]
                age = np.maximum(float(timestamp) - last_times[matched_pos], 0.0)
                flat_recency[write_pos] = -np.log1p(age)

        recency[~pool_valid_mask] = -np.inf
        past[~pool_valid_mask] = -np.inf
        return recency, past

    def build(self, samples, config: TrainPoolBuildConfig, show_progress=True, progress_desc="RRF scoring: pool build"):
        if not samples:
            return TrainPoolBatch.empty()

        num_samples = len(samples)
        num_pools = max(1, int(config.num_pools))
        pool_size = max(2, int(config.pool_size))
        num_pool_rows = num_samples * num_pools

        sample_query_ids = np.empty(num_samples, dtype=np.int64)
        pool_sample_indices = np.repeat(np.arange(num_samples, dtype=np.int64), num_pools)
        pool_source_ids = np.empty(num_pool_rows, dtype=np.int64)
        pool_timestamps = np.empty(num_pool_rows, dtype=np.float64)
        pool_targets = np.full((num_pool_rows, pool_size), -1, dtype=np.int64)
        pool_valid_mask = np.zeros((num_pool_rows, pool_size), dtype=bool)
        pool_recency_scores = np.full((num_pool_rows, pool_size), -np.inf, dtype=np.float64)
        pool_past_scores = np.full((num_pool_rows, pool_size), -np.inf, dtype=np.float64)

        sample_iter = enumerate(samples)
        if show_progress:
            sample_iter = enumerate(tqdm(samples, desc=progress_desc, ncols=100))

        for sample_idx, sample in sample_iter:
            src = int(sample['source_id'])
            ts = float(sample['timestamp'])
            sample_query_ids[sample_idx] = int(sample.get('query_id', sample_idx))
            row_start = sample_idx * num_pools
            row_end = row_start + num_pools
            pool_source_ids[row_start:row_end] = src
            pool_timestamps[row_start:row_end] = ts

            sample_pool_targets, sample_pool_valid_mask = self._sample_pool_rows(
                sample=sample,
                sample_idx=sample_idx,
                num_pools=num_pools,
                pool_size=pool_size,
                base_seed=config.random_seed,
            )
            sample_recency_scores, sample_past_scores = self._score_pool_target_matrix_fast(
                src=src,
                timestamp=ts,
                pool_targets=sample_pool_targets,
                pool_valid_mask=sample_pool_valid_mask,
            )

            pool_targets[row_start:row_end] = sample_pool_targets
            pool_valid_mask[row_start:row_end] = sample_pool_valid_mask
            pool_recency_scores[row_start:row_end] = sample_recency_scores
            pool_past_scores[row_start:row_end] = sample_past_scores

        return TrainPoolBatch(
            sample_query_ids=sample_query_ids,
            pool_sample_indices=pool_sample_indices,
            pool_source_ids=pool_source_ids,
            pool_timestamps=pool_timestamps,
            pool_targets=pool_targets,
            pool_valid_mask=pool_valid_mask,
            pool_recency_scores=pool_recency_scores,
            pool_past_scores=pool_past_scores,
            anchor_col=0,
        )


class TrainPoolRRFEngine:
    """
    Score and fuse dense train-pool batches without routing through sample dicts.
    """

    def __init__(self, neighbor_sampler, config: TrainPoolRRFConfig):
        self.neighbor_sampler = neighbor_sampler
        self.config = config
        self.selected_heuristics = normalize_rrf_heuristics(config.rrf_heuristics)

    def _empty_outputs(self, num_samples):
        zeros = np.zeros(num_samples, dtype=np.float64)
        ones = np.ones(num_samples, dtype=np.int64)
        component_scores = {'rrf_rank': ones.copy()}
        for name in self.selected_heuristics:
            component_scores[HEURISTIC_SCORE_KEYS[name]] = zeros.copy()
            component_scores[HEURISTIC_RANK_KEYS[name]] = ones.copy()
        return zeros.copy(), zeros.copy(), component_scores

    def _pool_rows_per_block(self, pool_size):
        return max(1, int(self.config.score_batch_size) // max(1, int(pool_size)))

    def score(self, batch: TrainPoolBatch, show_progress=True, progress_desc="RRF scoring"):
        if batch.num_samples == 0 or batch.num_pool_rows == 0:
            return self._empty_outputs(batch.num_samples)

        n_samples = batch.num_samples
        counts = np.zeros(n_samples, dtype=np.int64)
        rrf_scores = np.zeros(n_samples, dtype=np.float64)
        ra_scores = np.zeros(n_samples, dtype=np.float64)
        selected_score_buffers = {
            name: np.zeros(n_samples, dtype=np.float64) for name in self.selected_heuristics
        }
        selected_rank_buffers = {
            name: np.ones(n_samples, dtype=np.int64) for name in self.selected_heuristics
        }
        rrf_ranks = np.ones(n_samples, dtype=np.int64)

        row_step = self._pool_rows_per_block(batch.pool_size)
        row_starts = range(0, batch.num_pool_rows, row_step)
        if show_progress:
            row_starts = tqdm(
                row_starts,
                total=(batch.num_pool_rows + row_step - 1) // row_step,
                desc=f"{progress_desc}: pooled blocks",
                ncols=100,
            )

        for row_start in row_starts:
            row_end = min(batch.num_pool_rows, row_start + row_step)
            block_targets = batch.pool_targets[row_start:row_end]
            block_valid_mask = batch.pool_valid_mask[row_start:row_end]
            block_recency = batch.pool_recency_scores[row_start:row_end].copy()
            block_past = batch.pool_past_scores[row_start:row_end].copy()
            block_popularity = np.full(block_targets.shape, -np.inf, dtype=np.float64)
            block_ra = np.full(block_targets.shape, -np.inf, dtype=np.float64)
            block_global_recency = np.full(block_targets.shape, -np.inf, dtype=np.float64)
            block_semantic = np.full(block_targets.shape, -1.0, dtype=np.float64)

            flat_valid = np.flatnonzero(block_valid_mask.reshape(-1))
            if flat_valid.size > 0:
                row_ids, col_ids = np.divmod(flat_valid, batch.pool_size)
                block_sources = batch.pool_source_ids[row_start:row_end]
                block_times = batch.pool_timestamps[row_start:row_end]
                flat_src = block_sources[row_ids]
                flat_tgt = block_targets[row_ids, col_ids]
                flat_ts = block_times[row_ids]

                flat_popularity = None
                if "popularity" in self.selected_heuristics:
                    flat_popularity = score_links_by_popularity(
                        self.neighbor_sampler,
                        flat_src,
                        flat_tgt,
                        flat_ts,
                    )
                flat_ra = score_links_by_common_neighbors(
                    self.neighbor_sampler,
                    flat_src,
                    flat_tgt,
                    flat_ts,
                    mode='ra',
                    use_gpu=self.config.use_gpu_heuristics,
                )
                flat_global_recency = None
                if "global_recency" in self.selected_heuristics:
                    flat_global_recency = score_links_by_global_recency(
                        self.neighbor_sampler,
                        flat_src,
                        flat_tgt,
                        flat_ts,
                    )
                flat_semantic = None
                if "semantic_smoothing" in self.selected_heuristics:
                    flat_semantic = _compute_semantic_smoothing_scores_for_pairs(
                        flat_src,
                        flat_tgt,
                        flat_ts,
                        semantic_context=self.config.semantic_context,
                        show_progress=False,
                    )
                if flat_popularity is not None:
                    block_popularity[row_ids, col_ids] = flat_popularity
                block_ra[row_ids, col_ids] = flat_ra
                if flat_global_recency is not None:
                    block_global_recency[row_ids, col_ids] = flat_global_recency
                if flat_semantic is not None:
                    block_semantic[row_ids, col_ids] = flat_semantic

            block_scores = {
                "recency": block_recency,
                "popularity": block_popularity,
                "past_interactions": block_past,
                "resource_allocation": block_ra,
                "global_recency": block_global_recency,
                "semantic_smoothing": block_semantic,
            }
            block_ranks = {
                name: descending_ranks(block_scores[name], axis=1)
                for name in self.selected_heuristics
            }

            rank_tensor = np.stack(
                [block_ranks[name] for name in self.selected_heuristics],
                axis=0,
            )
            block_rrf = reciprocal_rank_fusion_from_ranks(rank_tensor, rrf_k=self.config.rrf_k, model_axis=0)
            block_rrf_rank = descending_ranks(block_rrf, axis=1)

            block_sample_indices = batch.pool_sample_indices[row_start:row_end]
            anchor_col = int(batch.anchor_col)
            np.add.at(counts, block_sample_indices, 1)
            np.add.at(rrf_scores, block_sample_indices, block_rrf[:, anchor_col])
            np.add.at(ra_scores, block_sample_indices, block_ra[:, anchor_col])
            np.add.at(rrf_ranks, block_sample_indices, block_rrf_rank[:, anchor_col])
            for name in self.selected_heuristics:
                np.add.at(
                    selected_score_buffers[name],
                    block_sample_indices,
                    block_scores[name][:, anchor_col],
                )

        valid = counts > 0
        rrf_scores[valid] /= counts[valid]
        ra_scores[valid] /= counts[valid]
        for name in self.selected_heuristics:
            selected_score_buffers[name][valid] /= counts[valid]

        query_groups = group_indices_by_values(batch.sample_query_ids)
        group_iter = query_groups.values()
        if show_progress:
            group_iter = tqdm(group_iter, total=len(query_groups), desc=f"{progress_desc}: rerank", ncols=100)
        for indices in group_iter:
            idx_arr = np.asarray(indices, dtype=np.int64)
            for name in self.selected_heuristics:
                selected_rank_buffers[name][idx_arr] = descending_ranks(selected_score_buffers[name][idx_arr])
            rrf_ranks[idx_arr] = descending_ranks(rrf_scores[idx_arr])

        component_scores = {'rrf_rank': rrf_ranks}
        for name in self.selected_heuristics:
            component_scores[HEURISTIC_SCORE_KEYS[name]] = selected_score_buffers[name]
            component_scores[HEURISTIC_RANK_KEYS[name]] = selected_rank_buffers[name]
        return rrf_scores, ra_scores, component_scores


def compute_train_pool_rrf_scores(
    samples,
    neighbor_sampler,
    build_config: TrainPoolBuildConfig,
    score_config: TrainPoolRRFConfig,
    train_entity_ids,
    show_progress=True,
):
    builder = TrainPoolBatchBuilder(
        neighbor_sampler=neighbor_sampler,
        train_entity_ids=train_entity_ids,
    )
    batch = builder.build(
        samples,
        config=build_config,
        show_progress=show_progress,
        progress_desc="RRF scoring: pool build",
    )
    engine = TrainPoolRRFEngine(neighbor_sampler=neighbor_sampler, config=score_config)
    return engine.score(batch, show_progress=show_progress, progress_desc="RRF scoring")
