from __future__ import annotations

import re
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from experiments.modules.heuristic_models import (
    HAS_CUDA,
    score_links_by_common_neighbors,
    score_links_by_global_recency,
    score_links_by_itemcf_cosine,
    score_links_by_past_interactions,
    score_links_by_personalized_location,
    score_links_by_popularity,
    score_links_by_recent_degree,
    score_links_by_recency,
    score_links_by_usercf_cosine,
)
from experiments.modules.semantic_mlp.graph_components import (
    SparseTensor,
    compute_mplp_exact_features,
)

SUPPORTED_HEURISTIC_FEATURE_NAMES = (
    "recency",
    "popularity",
    "recent_degree",
    "global_recency",
    "past",
    "ra",
    "itemcf",
    "usercf",
    "city_preference",
    "zip_preference",
)
HEURISTIC_FEATURE_NAME_ALIASES = {
    "itemcf_cosine": "itemcf",
    "usercf_cosine": "usercf",
    "personalized_city": "city_preference",
    "personalized_zip": "zip_preference",
}

_GOOGLEMAP_CITY_ZIP_RE = re.compile(
    r",\s*([^,.\n]+?)\s*,\s*([A-Z]{2})(?:\s+(\d{5})(?:-\d{4})?)?(?=\.|,|$)"
)


def extract_googlemap_city_zip(text: object) -> Tuple[str, str]:
    """Return (city/state, ZIP5) parsed from a Googlemap business card."""
    if not isinstance(text, str):
        return "", ""
    clean = " ".join(text.strip().split())
    if not clean.startswith("Name:"):
        return "", ""
    category_pos = clean.find(". Category:")
    address_segment = clean if category_pos < 0 else clean[:category_pos]
    matches = _GOOGLEMAP_CITY_ZIP_RE.findall(address_segment)
    if not matches:
        return "", ""
    city, state, zip_code = matches[-1]
    city = city.strip()
    state = state.strip().upper()
    if not city or len(state) != 2:
        return "", ""
    return f"{city}, {state}", zip_code.strip()


def build_googlemap_city_zip_ids(entity_text_df, max_node_id: int):
    """Build node-indexed compact city/state and ZIP identifiers."""
    city_ids = np.full(max_node_id + 1, -1, dtype=np.int64)
    zip_ids = np.full(max_node_id + 1, -1, dtype=np.int64)
    city_to_id = {}
    zip_to_id = {}

    for node_id, text in zip(entity_text_df["i"], entity_text_df["text"]):
        node_id = int(node_id)
        if node_id < 0 or node_id > max_node_id:
            continue
        city, zip_code = extract_googlemap_city_zip(text)
        if city:
            city_ids[node_id] = city_to_id.setdefault(city, len(city_to_id))
        if zip_code:
            zip_ids[node_id] = zip_to_id.setdefault(zip_code, len(zip_to_id))

    return city_ids, zip_ids, city_to_id, zip_to_id

class HeuristicFeatureExtractor:
    """
    CRAFT-style edge heuristics with configurable feature subset.
    """

    def __init__(
        self,
        neighbor_sampler,
        directed_src_node_ids: Optional[np.ndarray] = None,
        directed_dst_node_ids: Optional[np.ndarray] = None,
        directed_node_interact_times: Optional[np.ndarray] = None,
        use_gpu_heuristics: bool = True,
        popularity_decay: float = 0.0,
        recent_degree_window: float = 50.0,
        score_batch_size: int = 200000,
        ablate_recency: bool = False,
        recency_directed: bool = False,
        feature_names: Optional[Tuple[str, ...]] = None,
        node_city_ids: Optional[np.ndarray] = None,
        node_zip_ids: Optional[np.ndarray] = None,
    ):
        self.neighbor_sampler = neighbor_sampler
        self.use_gpu_heuristics = bool(use_gpu_heuristics and HAS_CUDA)
        self.popularity_decay = float(popularity_decay)
        self.recent_degree_window = float(recent_degree_window)
        self.score_batch_size = max(1, int(score_batch_size))
        self.ablate_recency = bool(ablate_recency)
        self.recency_directed = bool(recency_directed)
        self.feature_names = self._normalize_feature_names(feature_names)
        self.directed_src_node_ids = (
            None if directed_src_node_ids is None
            else np.asarray(directed_src_node_ids, dtype=np.int64).reshape(-1)
        )
        self.directed_dst_node_ids = (
            None if directed_dst_node_ids is None
            else np.asarray(directed_dst_node_ids, dtype=np.int64).reshape(-1)
        )
        self.directed_node_interact_times = (
            None if directed_node_interact_times is None
            else np.asarray(directed_node_interact_times, dtype=np.float64).reshape(-1)
        )
        self.node_city_ids = (
            None if node_city_ids is None
            else np.asarray(node_city_ids, dtype=np.int64).reshape(-1)
        )
        self.node_zip_ids = (
            None if node_zip_ids is None
            else np.asarray(node_zip_ids, dtype=np.int64).reshape(-1)
        )
        self.feature_dim = len(self.feature_names)
        if "recent_degree" in self.feature_names and self.recent_degree_window <= 0.0:
            raise ValueError("recent_degree_window must be > 0 when recent_degree is enabled.")
        if self.recency_directed and "recency" in self.feature_names:
            if (
                self.directed_src_node_ids is None
                or self.directed_dst_node_ids is None
                or self.directed_node_interact_times is None
            ):
                raise ValueError(
                    "Directed recency requires directed source, destination, and time arrays."
                )
        if {"itemcf", "usercf", "city_preference", "zip_preference"} & set(self.feature_names):
            if (
                self.directed_src_node_ids is None
                or self.directed_dst_node_ids is None
                or self.directed_node_interact_times is None
            ):
                raise ValueError(
                    "Collaborative/location preference features require directed_src_node_ids, "
                    "directed_dst_node_ids, and directed_node_interact_times."
                )
        if "city_preference" in self.feature_names and self.node_city_ids is None:
            raise ValueError("city_preference requires node_city_ids.")
        if "zip_preference" in self.feature_names and self.node_zip_ids is None:
            raise ValueError("zip_preference requires node_zip_ids.")
        self._feature_cache: Dict[Tuple[int, int, float], np.ndarray] = {}
        self._debug_runtime_batch_logs = 0
        self._debug_cache_miss_logs = 0

    @staticmethod
    def _normalize_feature_names(feature_names: Optional[Tuple[str, ...]]) -> Tuple[str, ...]:
        if feature_names is None:
            return SUPPORTED_HEURISTIC_FEATURE_NAMES
        normalized = tuple(
            HEURISTIC_FEATURE_NAME_ALIASES.get(str(name).strip().lower(), str(name).strip().lower())
            for name in feature_names
            if str(name).strip()
        )
        if not normalized:
            raise ValueError("Heuristic feature extractor requires at least one feature.")
        unknown = [name for name in normalized if name not in SUPPORTED_HEURISTIC_FEATURE_NAMES]
        if unknown:
            raise ValueError(
                "Unsupported heuristic feature(s): "
                f"{', '.join(unknown)}. Supported values: {', '.join(SUPPORTED_HEURISTIC_FEATURE_NAMES)}."
            )
        return tuple(name for name in SUPPORTED_HEURISTIC_FEATURE_NAMES if name in set(normalized))

    @staticmethod
    def _minmax_norm(features: np.ndarray) -> np.ndarray:
        mins = np.min(features, axis=0, keepdims=True)
        maxs = np.max(features, axis=0, keepdims=True)
        denom = maxs - mins
        denom[denom < 1e-12] = 1.0
        out = (features - mins) / denom
        return np.clip(out, 0.0, 1.0)

    @staticmethod
    def normalize_raw_features(features: np.ndarray) -> np.ndarray:
        return HeuristicFeatureExtractor._minmax_norm(
            np.asarray(features, dtype=np.float32)
        ).astype(np.float32)

    @staticmethod
    def _cache_key(source: int, target: int, prediction_time: float) -> Tuple[int, int, float]:
        return int(source), int(target), float(prediction_time)

    def _build_raw_feature_matrix(
        self,
        *,
        raw_recency: Optional[np.ndarray],
        raw_popularity: Optional[np.ndarray],
        raw_recent_degree: Optional[np.ndarray],
        raw_global_recency: Optional[np.ndarray],
        raw_past: Optional[np.ndarray],
        raw_ra: Optional[np.ndarray],
        raw_itemcf: Optional[np.ndarray],
        raw_usercf: Optional[np.ndarray],
        raw_city_preference: Optional[np.ndarray],
        raw_zip_preference: Optional[np.ndarray],
        prediction_times: np.ndarray,
    ) -> np.ndarray:
        feature_columns = []
        if "recency" in self.feature_names:
            assert raw_recency is not None
            unseen = raw_recency <= -1e14
            delta_t = prediction_times - raw_recency
            delta_t[unseen] = 1e9
            feature_columns.append(-np.log1p(np.clip(delta_t, a_min=0.0, a_max=None)))
        if "popularity" in self.feature_names:
            assert raw_popularity is not None
            feature_columns.append(np.log1p(raw_popularity))
        if "recent_degree" in self.feature_names:
            assert raw_recent_degree is not None
            feature_columns.append(np.log1p(raw_recent_degree))
        if "global_recency" in self.feature_names:
            assert raw_global_recency is not None
            global_delta = -raw_global_recency
            unseen = raw_global_recency <= -1e14
            global_delta[unseen] = 1e9
            feature_columns.append(-np.log1p(np.clip(global_delta, a_min=0.0, a_max=None)))
        if "past" in self.feature_names:
            assert raw_past is not None
            feature_columns.append(np.log1p(raw_past))
        if "ra" in self.feature_names:
            assert raw_ra is not None
            feature_columns.append(np.log1p(raw_ra))
        if "itemcf" in self.feature_names:
            assert raw_itemcf is not None
            feature_columns.append(np.log1p(raw_itemcf))
        if "usercf" in self.feature_names:
            assert raw_usercf is not None
            feature_columns.append(np.log1p(raw_usercf))
        if "city_preference" in self.feature_names:
            assert raw_city_preference is not None
            feature_columns.append(raw_city_preference)
        if "zip_preference" in self.feature_names:
            assert raw_zip_preference is not None
            feature_columns.append(raw_zip_preference)
        return np.stack(feature_columns, axis=1).astype(np.float32)

    def _compute_raw_features_batched(
        self,
        sources: np.ndarray,
        targets: np.ndarray,
        prediction_times: np.ndarray,
        desc: Optional[str] = None,
        show_progress: bool = False,
        profile_out: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        total = len(sources)
        if total == 0:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        if (not show_progress) and self._debug_runtime_batch_logs < 4:
            print(
                "[SemanticHeuristics] runtime feature request: "
                f"total={int(total)}, chunk_size={int(self.score_batch_size)}, "
                f"use_gpu_ra={self.use_gpu_heuristics}, desc={desc or 'runtime'}"
            )
            self._debug_runtime_batch_logs += 1

        features = np.empty((total, self.feature_dim), dtype=np.float32)
        chunk_starts = range(0, total, self.score_batch_size)
        num_chunks = (total + self.score_batch_size - 1) // self.score_batch_size
        progress_bar = None
        if show_progress:
            progress_bar = tqdm(
                total=num_chunks,
                desc=desc or "Heuristic precompute",
                ncols=100,
            )

        for chunk_idx, start in enumerate(chunk_starts, start=1):
            end = min(total, start + self.score_batch_size)
            sl = slice(start, end)
            if show_progress:
                print(
                    "[SemanticHeuristics] precompute chunk start: "
                    f"chunk={chunk_idx}/{num_chunks}, start={int(start)}, "
                    f"end={int(end)}, size={int(end - start)}, "
                    f"use_gpu_ra={self.use_gpu_heuristics}"
                )
            elif self._debug_runtime_batch_logs < 8:
                print(
                    "[SemanticHeuristics] runtime feature chunk: "
                    f"start={int(start)}, end={int(end)}, size={int(end - start)}, "
                    f"use_gpu_ra={self.use_gpu_heuristics}"
                )
                self._debug_runtime_batch_logs += 1
            chunk_t0 = time.perf_counter()
            t_start = time.perf_counter() if profile_out is not None or show_progress else None
            if show_progress and "recency" in self.feature_names:
                print(
                    "[SemanticHeuristics] chunk stage start: "
                    f"chunk={chunk_idx}/{num_chunks}, stage=recency"
                )
            raw_recency = None
            if "recency" in self.feature_names:
                if self.ablate_recency:
                    raw_recency = np.full(end - start, -1e15, dtype=np.float64)
                else:
                    raw_recency = score_links_by_recency(
                        neighbor_sampler=self.neighbor_sampler,
                        sources=sources[sl],
                        targets=targets[sl],
                        prediction_times=prediction_times[sl],
                        directed=self.recency_directed,
                        directed_src_node_ids=self.directed_src_node_ids,
                        directed_dst_node_ids=self.directed_dst_node_ids,
                        directed_node_interact_times=self.directed_node_interact_times,
                    )
                    if self.recency_directed:
                        # The legacy kernel returns last_time - prediction_time even
                        # though its public contract says absolute last_time.  Preserve
                        # the old undirected checkpoint path byte-for-byte, but convert
                        # the new directed intervention back to an absolute timestamp so
                        # _build_raw_feature_matrix computes the intended positive gap.
                        seen = raw_recency > -1e14
                        raw_recency = raw_recency.copy()
                        raw_recency[seen] += prediction_times[sl][seen]
            if profile_out is not None:
                profile_out['recency_s'] = profile_out.get('recency_s', 0.0) + (time.perf_counter() - t_start)
            recency_elapsed = (time.perf_counter() - t_start) if show_progress else None
            if show_progress and "recency" in self.feature_names:
                print(
                    "[SemanticHeuristics] chunk stage done: "
                    f"chunk={chunk_idx}/{num_chunks}, stage=recency, "
                    f"elapsed_s={recency_elapsed:.2f}, ablated={self.ablate_recency}, "
                    f"directed={self.recency_directed}"
                )

            t_start = time.perf_counter() if profile_out is not None or show_progress else None
            if show_progress and "popularity" in self.feature_names:
                print(
                    "[SemanticHeuristics] chunk stage start: "
                    f"chunk={chunk_idx}/{num_chunks}, stage=popularity"
                )
            raw_popularity = None
            if "popularity" in self.feature_names:
                raw_popularity = score_links_by_popularity(
                    neighbor_sampler=self.neighbor_sampler,
                    sources=sources[sl],
                    targets=targets[sl],
                    prediction_times=prediction_times[sl],
                    decay=self.popularity_decay,
                )
            if profile_out is not None:
                profile_out['popularity_s'] = profile_out.get('popularity_s', 0.0) + (time.perf_counter() - t_start)
            popularity_elapsed = (time.perf_counter() - t_start) if show_progress else None
            if show_progress and "popularity" in self.feature_names:
                print(
                    "[SemanticHeuristics] chunk stage done: "
                    f"chunk={chunk_idx}/{num_chunks}, stage=popularity, elapsed_s={popularity_elapsed:.2f}"
                )

            t_start = time.perf_counter() if profile_out is not None or show_progress else None
            if show_progress and "recent_degree" in self.feature_names:
                print(
                    "[SemanticHeuristics] chunk stage start: "
                    f"chunk={chunk_idx}/{num_chunks}, stage=recent_degree"
                )
            raw_recent_degree = None
            if "recent_degree" in self.feature_names:
                raw_recent_degree = score_links_by_recent_degree(
                    neighbor_sampler=self.neighbor_sampler,
                    sources=sources[sl],
                    targets=targets[sl],
                    prediction_times=prediction_times[sl],
                    window=self.recent_degree_window,
                )
            if profile_out is not None:
                profile_out['recent_degree_s'] = profile_out.get('recent_degree_s', 0.0) + (time.perf_counter() - t_start)
            recent_degree_elapsed = (time.perf_counter() - t_start) if show_progress else None
            if show_progress and "recent_degree" in self.feature_names:
                print(
                    "[SemanticHeuristics] chunk stage done: "
                    f"chunk={chunk_idx}/{num_chunks}, stage=recent_degree, elapsed_s={recent_degree_elapsed:.2f}"
                )

            t_start = time.perf_counter() if profile_out is not None or show_progress else None
            if show_progress and "global_recency" in self.feature_names:
                print(
                    "[SemanticHeuristics] chunk stage start: "
                    f"chunk={chunk_idx}/{num_chunks}, stage=global_recency"
                )
            raw_global_recency = None
            if "global_recency" in self.feature_names:
                raw_global_recency = score_links_by_global_recency(
                    neighbor_sampler=self.neighbor_sampler,
                    sources=sources[sl],
                    targets=targets[sl],
                    prediction_times=prediction_times[sl],
                )
            if profile_out is not None:
                profile_out['global_recency_s'] = profile_out.get('global_recency_s', 0.0) + (time.perf_counter() - t_start)
            global_recency_elapsed = (time.perf_counter() - t_start) if show_progress else None
            if show_progress and "global_recency" in self.feature_names:
                print(
                    "[SemanticHeuristics] chunk stage done: "
                    f"chunk={chunk_idx}/{num_chunks}, stage=global_recency, elapsed_s={global_recency_elapsed:.2f}"
                )

            t_start = time.perf_counter() if profile_out is not None or show_progress else None
            if show_progress and "past" in self.feature_names:
                print(
                    "[SemanticHeuristics] chunk stage start: "
                    f"chunk={chunk_idx}/{num_chunks}, stage=past"
                )
            raw_past = None
            if "past" in self.feature_names:
                raw_past = score_links_by_past_interactions(
                    neighbor_sampler=self.neighbor_sampler,
                    sources=sources[sl],
                    targets=targets[sl],
                    prediction_times=prediction_times[sl],
                )
            if profile_out is not None:
                profile_out['past_s'] = profile_out.get('past_s', 0.0) + (time.perf_counter() - t_start)
            past_elapsed = (time.perf_counter() - t_start) if show_progress else None
            if show_progress and "past" in self.feature_names:
                print(
                    "[SemanticHeuristics] chunk stage done: "
                    f"chunk={chunk_idx}/{num_chunks}, stage=past, elapsed_s={past_elapsed:.2f}"
                )

            t_start = time.perf_counter() if profile_out is not None or show_progress else None
            if show_progress and "ra" in self.feature_names:
                print(
                    "[SemanticHeuristics] chunk stage start: "
                    f"chunk={chunk_idx}/{num_chunks}, stage=ra"
                )
            raw_ra = None
            if "ra" in self.feature_names:
                raw_ra = score_links_by_common_neighbors(
                    neighbor_sampler=self.neighbor_sampler,
                    sources=sources[sl],
                    targets=targets[sl],
                    prediction_times=prediction_times[sl],
                    mode='ra',
                    use_gpu=self.use_gpu_heuristics,
                )
            if profile_out is not None:
                profile_out['ra_s'] = profile_out.get('ra_s', 0.0) + (time.perf_counter() - t_start)
            ra_elapsed = (time.perf_counter() - t_start) if show_progress else None
            if show_progress and "ra" in self.feature_names:
                print(
                    "[SemanticHeuristics] chunk stage done: "
                    f"chunk={chunk_idx}/{num_chunks}, stage=ra, elapsed_s={ra_elapsed:.2f}"
                )

            t_start = time.perf_counter() if profile_out is not None or show_progress else None
            if show_progress and "itemcf" in self.feature_names:
                print(
                    "[SemanticHeuristics] chunk stage start: "
                    f"chunk={chunk_idx}/{num_chunks}, stage=itemcf"
                )
            raw_itemcf = None
            if "itemcf" in self.feature_names:
                raw_itemcf = score_links_by_itemcf_cosine(
                    neighbor_sampler=self.neighbor_sampler,
                    sources=sources[sl],
                    targets=targets[sl],
                    prediction_times=prediction_times[sl],
                    directed_src_node_ids=self.directed_src_node_ids,
                    directed_dst_node_ids=self.directed_dst_node_ids,
                    directed_node_interact_times=self.directed_node_interact_times,
                )
            if profile_out is not None:
                profile_out['itemcf_s'] = profile_out.get('itemcf_s', 0.0) + (time.perf_counter() - t_start)
            itemcf_elapsed = (time.perf_counter() - t_start) if show_progress else None
            if show_progress and "itemcf" in self.feature_names:
                print(
                    "[SemanticHeuristics] chunk stage done: "
                    f"chunk={chunk_idx}/{num_chunks}, stage=itemcf, elapsed_s={itemcf_elapsed:.2f}"
                )

            t_start = time.perf_counter() if profile_out is not None or show_progress else None
            if show_progress and "usercf" in self.feature_names:
                print(
                    "[SemanticHeuristics] chunk stage start: "
                    f"chunk={chunk_idx}/{num_chunks}, stage=usercf"
                )
            raw_usercf = None
            if "usercf" in self.feature_names:
                raw_usercf = score_links_by_usercf_cosine(
                    neighbor_sampler=self.neighbor_sampler,
                    sources=sources[sl],
                    targets=targets[sl],
                    prediction_times=prediction_times[sl],
                    directed_src_node_ids=self.directed_src_node_ids,
                    directed_dst_node_ids=self.directed_dst_node_ids,
                    directed_node_interact_times=self.directed_node_interact_times,
                )
            if profile_out is not None:
                profile_out['usercf_s'] = profile_out.get('usercf_s', 0.0) + (time.perf_counter() - t_start)
            usercf_elapsed = (time.perf_counter() - t_start) if show_progress else None
            if show_progress and "usercf" in self.feature_names:
                print(
                    "[SemanticHeuristics] chunk stage done: "
                    f"chunk={chunk_idx}/{num_chunks}, stage=usercf, elapsed_s={usercf_elapsed:.2f}"
                )

            t_start = time.perf_counter() if profile_out is not None or show_progress else None
            if show_progress and "city_preference" in self.feature_names:
                print(
                    "[SemanticHeuristics] chunk stage start: "
                    f"chunk={chunk_idx}/{num_chunks}, stage=city_preference"
                )
            raw_city_preference = None
            if "city_preference" in self.feature_names:
                raw_city_preference = score_links_by_personalized_location(
                    neighbor_sampler=self.neighbor_sampler,
                    sources=sources[sl],
                    targets=targets[sl],
                    prediction_times=prediction_times[sl],
                    directed_src_node_ids=self.directed_src_node_ids,
                    directed_dst_node_ids=self.directed_dst_node_ids,
                    directed_node_interact_times=self.directed_node_interact_times,
                    node_location_ids=self.node_city_ids,
                )
            if profile_out is not None:
                profile_out['city_preference_s'] = profile_out.get('city_preference_s', 0.0) + (time.perf_counter() - t_start)
            city_preference_elapsed = (time.perf_counter() - t_start) if show_progress else None
            if show_progress and "city_preference" in self.feature_names:
                print(
                    "[SemanticHeuristics] chunk stage done: "
                    f"chunk={chunk_idx}/{num_chunks}, stage=city_preference, "
                    f"elapsed_s={city_preference_elapsed:.2f}"
                )

            t_start = time.perf_counter() if profile_out is not None or show_progress else None
            if show_progress and "zip_preference" in self.feature_names:
                print(
                    "[SemanticHeuristics] chunk stage start: "
                    f"chunk={chunk_idx}/{num_chunks}, stage=zip_preference"
                )
            raw_zip_preference = None
            if "zip_preference" in self.feature_names:
                raw_zip_preference = score_links_by_personalized_location(
                    neighbor_sampler=self.neighbor_sampler,
                    sources=sources[sl],
                    targets=targets[sl],
                    prediction_times=prediction_times[sl],
                    directed_src_node_ids=self.directed_src_node_ids,
                    directed_dst_node_ids=self.directed_dst_node_ids,
                    directed_node_interact_times=self.directed_node_interact_times,
                    node_location_ids=self.node_zip_ids,
                )
            if profile_out is not None:
                profile_out['zip_preference_s'] = profile_out.get('zip_preference_s', 0.0) + (time.perf_counter() - t_start)
            zip_preference_elapsed = (time.perf_counter() - t_start) if show_progress else None
            if show_progress and "zip_preference" in self.feature_names:
                print(
                    "[SemanticHeuristics] chunk stage done: "
                    f"chunk={chunk_idx}/{num_chunks}, stage=zip_preference, "
                    f"elapsed_s={zip_preference_elapsed:.2f}"
                )

            t_start = time.perf_counter() if profile_out is not None or show_progress else None
            if show_progress:
                print(
                    "[SemanticHeuristics] chunk stage start: "
                    f"chunk={chunk_idx}/{num_chunks}, stage=build"
                )
            features[sl] = self._build_raw_feature_matrix(
                raw_recency=raw_recency,
                raw_popularity=raw_popularity,
                raw_recent_degree=raw_recent_degree,
                raw_global_recency=raw_global_recency,
                raw_past=raw_past,
                raw_ra=raw_ra,
                raw_itemcf=raw_itemcf,
                raw_usercf=raw_usercf,
                raw_city_preference=raw_city_preference,
                raw_zip_preference=raw_zip_preference,
                prediction_times=prediction_times[sl],
            )
            if profile_out is not None:
                profile_out['build_s'] = profile_out.get('build_s', 0.0) + (time.perf_counter() - t_start)
            build_elapsed = (time.perf_counter() - t_start) if show_progress else None
            if show_progress:
                print(
                    "[SemanticHeuristics] chunk stage done: "
                    f"chunk={chunk_idx}/{num_chunks}, stage=build, elapsed_s={build_elapsed:.2f}"
                )
            if show_progress:
                chunk_elapsed = time.perf_counter() - chunk_t0
                selected_elapsed = {
                    "recency": recency_elapsed,
                    "popularity": popularity_elapsed,
                    "recent_degree": recent_degree_elapsed,
                    "global_recency": global_recency_elapsed,
                    "past": past_elapsed,
                    "ra": ra_elapsed,
                    "itemcf": itemcf_elapsed,
                    "usercf": usercf_elapsed,
                    "city_preference": city_preference_elapsed,
                    "zip_preference": zip_preference_elapsed,
                }
                elapsed_text = ", ".join(
                    f"{name}_s={selected_elapsed[name]:.2f}"
                    for name in self.feature_names
                    if selected_elapsed.get(name) is not None
                )
                if elapsed_text:
                    elapsed_text = f"{elapsed_text}, "
                print(
                    "[SemanticHeuristics] precompute chunk done: "
                    f"chunk={chunk_idx}/{num_chunks}, "
                    f"{elapsed_text}"
                    f"build_s={build_elapsed:.2f}, "
                    f"total_s={chunk_elapsed:.2f}"
                )
                progress_bar.update(1)

        if progress_bar is not None:
            progress_bar.close()

        return features

    def _store_features_in_cache(
        self,
        sources: np.ndarray,
        targets: np.ndarray,
        prediction_times: np.ndarray,
        features: np.ndarray,
    ) -> None:
        for idx in range(len(sources)):
            self._feature_cache[self._cache_key(sources[idx], targets[idx], prediction_times[idx])] = (
                features[idx].copy()
            )

    def precompute_raw_features(
        self,
        sources: np.ndarray,
        targets: np.ndarray,
        prediction_times: np.ndarray,
        desc: str,
        store_in_cache: bool = True,
    ) -> np.ndarray:
        sources = np.asarray(sources, dtype=np.int64).reshape(-1)
        targets = np.asarray(targets, dtype=np.int64).reshape(-1)
        prediction_times = np.asarray(prediction_times, dtype=np.float64).reshape(-1)
        features = self._compute_raw_features_batched(
            sources=sources,
            targets=targets,
            prediction_times=prediction_times,
            desc=desc,
            show_progress=True,
        )
        if store_in_cache:
            self._store_features_in_cache(sources, targets, prediction_times, features)
        return features

    def get_raw_features(
        self,
        sources: np.ndarray,
        targets: np.ndarray,
        prediction_times: np.ndarray,
        profile_out: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        sources = np.asarray(sources, dtype=np.int64).reshape(-1)
        targets = np.asarray(targets, dtype=np.int64).reshape(-1)
        prediction_times = np.asarray(prediction_times, dtype=np.float64).reshape(-1)
        if len(sources) == 0:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        features = np.empty((len(sources), self.feature_dim), dtype=np.float32)
        missing_indices = []
        missing_sources = []
        missing_targets = []
        missing_times = []
        lookup_t0 = time.perf_counter() if profile_out is not None else None

        for idx in range(len(sources)):
            cache_key = self._cache_key(sources[idx], targets[idx], prediction_times[idx])
            cached = self._feature_cache.get(cache_key)
            if cached is None:
                missing_indices.append(idx)
                missing_sources.append(sources[idx])
                missing_targets.append(targets[idx])
                missing_times.append(prediction_times[idx])
            else:
                features[idx] = cached

        if profile_out is not None:
            profile_out['lookup_s'] = profile_out.get('lookup_s', 0.0) + (time.perf_counter() - lookup_t0)
            profile_out['query_count'] = profile_out.get('query_count', 0.0) + float(len(sources))
            profile_out['miss_count'] = profile_out.get('miss_count', 0.0) + float(len(missing_indices))

        if missing_indices:
            if self._debug_cache_miss_logs < 6:
                print(
                    "[SemanticHeuristics] cache miss refill: "
                    f"missing={len(missing_indices)}/{len(sources)}, "
                    f"use_gpu_ra={self.use_gpu_heuristics}"
                )
                self._debug_cache_miss_logs += 1
            missing_features = self._compute_raw_features_batched(
                sources=np.asarray(missing_sources, dtype=np.int64),
                targets=np.asarray(missing_targets, dtype=np.int64),
                prediction_times=np.asarray(missing_times, dtype=np.float64),
                profile_out=profile_out,
            )
            update_t0 = time.perf_counter() if profile_out is not None else None
            for local_idx, batch_idx in enumerate(missing_indices):
                feature_row = missing_features[local_idx]
                features[batch_idx] = feature_row
                self._feature_cache[
                    self._cache_key(sources[batch_idx], targets[batch_idx], prediction_times[batch_idx])
                ] = feature_row.copy()
            if profile_out is not None:
                profile_out['update_s'] = profile_out.get('update_s', 0.0) + (time.perf_counter() - update_t0)

        return features

    def compute_normalized_features(
        self,
        sources: np.ndarray,
        targets: np.ndarray,
        prediction_times: np.ndarray,
    ) -> np.ndarray:
        return self.normalize_raw_features(
            self.get_raw_features(
                sources=sources,
                targets=targets,
                prediction_times=prediction_times,
            )
        )


class HeuristicFusionHead(nn.Module):
    """
    Learnable additive fusion in logit space.
    final_logit = semantic_logit + w(heuristics)
    """

    def __init__(self, feature_dim: int = 4):
        super().__init__()
        self.linear = nn.Linear(int(feature_dim), 1, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, semantic_logits: torch.Tensor, heuristic_features: torch.Tensor) -> torch.Tensor:
        delta = self.linear(heuristic_features).squeeze(-1)
        return semantic_logits + delta

def build_semantic_mlp_auxiliary_features(
    *,
    heuristic_extractor: Optional[HeuristicFeatureExtractor],
    lookup: torch.Tensor,
    sources: np.ndarray,
    targets: np.ndarray,
    prediction_times: np.ndarray,
    raw_heuristic_features: Optional[np.ndarray],
    ncn_adj: Optional[SparseTensor],
    two_hop_adj: Optional[SparseTensor],
    output_device: torch.device,
    output_dtype: torch.dtype,
    normalize_heuristics: bool = True,
    profile_out: Optional[Dict[str, float]] = None,
) -> Optional[torch.Tensor]:
    feature_tensors = []

    if heuristic_extractor is not None:
        if raw_heuristic_features is None:
            raw_heuristic_features = heuristic_extractor.get_raw_features(
                sources=sources,
                targets=targets,
                prediction_times=prediction_times,
                profile_out=profile_out,
            )
        else:
            raw_heuristic_features = np.asarray(raw_heuristic_features, dtype=np.float32)
        if normalize_heuristics:
            heuristic_values = heuristic_extractor.normalize_raw_features(raw_heuristic_features)
        else:
            # Ridge fitting learns one stable, training-wide scale per feature.  Do not
            # make its inputs depend on the composition or size of the current batch.
            heuristic_values = np.asarray(raw_heuristic_features, dtype=np.float32)
        feature_tensors.append(
            torch.from_numpy(heuristic_values).to(device=output_device, dtype=output_dtype)
        )

    if ncn_adj is not None:
        structural_features, _ = compute_mplp_exact_features(
            lookup=lookup,
            src_ids=sources,
            dst_ids=targets,
            ncn_adj=ncn_adj,
            two_hop_adj=two_hop_adj,
            output_device=output_device,
            output_dtype=output_dtype,
        )
        feature_tensors.append(structural_features)

    if not feature_tensors:
        return None
    return torch.cat(feature_tensors, dim=1)


def fuse_pos_neg_logits_with_heuristics(
    heuristic_extractor: Optional[HeuristicFeatureExtractor],
    heuristic_fusion: Optional[HeuristicFusionHead],
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    pos_src: np.ndarray,
    pos_dst: np.ndarray,
    pos_times: np.ndarray,
    neg_src: np.ndarray,
    neg_dst: np.ndarray,
    neg_times: np.ndarray,
    pos_raw_features: Optional[np.ndarray] = None,
    neg_raw_features: Optional[np.ndarray] = None,
    profile_out: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    if heuristic_extractor is None or heuristic_fusion is None:
        return pos_logits, neg_logits, 0.0

    t0 = time.perf_counter()
    if pos_raw_features is None:
        pos_raw_features = heuristic_extractor.get_raw_features(
            sources=pos_src,
            targets=pos_dst,
            prediction_times=pos_times,
            profile_out=profile_out,
        )
    else:
        t_wrap = time.perf_counter() if profile_out is not None else None
        pos_raw_features = np.asarray(pos_raw_features, dtype=np.float32)
        if profile_out is not None:
            profile_out['post_s'] = profile_out.get('post_s', 0.0) + (time.perf_counter() - t_wrap)
    if neg_raw_features is None:
        neg_raw_features = heuristic_extractor.get_raw_features(
            sources=neg_src,
            targets=neg_dst,
            prediction_times=neg_times,
            profile_out=profile_out,
        )
    else:
        t_wrap = time.perf_counter() if profile_out is not None else None
        neg_raw_features = np.asarray(neg_raw_features, dtype=np.float32)
        if profile_out is not None:
            profile_out['post_s'] = profile_out.get('post_s', 0.0) + (time.perf_counter() - t_wrap)

    t_post = time.perf_counter() if profile_out is not None else None
    combined_features = heuristic_extractor.normalize_raw_features(
        np.concatenate([pos_raw_features, neg_raw_features], axis=0)
    )
    bs = len(pos_src)
    pos_features = torch.from_numpy(combined_features[:bs]).to(device=pos_logits.device, dtype=pos_logits.dtype)
    neg_features = torch.from_numpy(combined_features[bs:]).to(device=neg_logits.device, dtype=neg_logits.dtype)
    fused_pos = heuristic_fusion(pos_logits, pos_features)
    fused_neg = heuristic_fusion(neg_logits, neg_features)
    if profile_out is not None:
        profile_out['post_s'] = profile_out.get('post_s', 0.0) + (time.perf_counter() - t_post)
    elapsed = time.perf_counter() - t0
    return fused_pos, fused_neg, elapsed
