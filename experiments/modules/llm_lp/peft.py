"""
Direct PEFT helpers for the current LLM link-prediction prompt pipeline.
"""
from __future__ import annotations

import importlib.util
import json
import glob
import os
import shutil
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from experiments.modules.llm_lp.eval import (
    _build_forced_binary_scoring_prompt,
    _build_prompt_for_sample,
)
from experiments.modules.llm_lp.prompt_context import materialize_samples_prompt_context
from experiments.modules.llm_lp.sample_finalize import finalize_test_samples
from experiments.modules.llm_lp.prompt_template import DEFAULT_KEY_SIGNAL_FIELDS
from experiments.modules.llm_lp.train_sampling import (
    POSITIVE_SAMPLING_GROUP_COLUMN,
    PROMPT_MULTIPLICITY_COLUMN,
    SOURCE_FIRST_TOUCH_COLUMN,
    TARGET_FIRST_TOUCH_COLUMN,
    select_training_positive_edges,
)
from experiments.modules.llm_lp.training_protocol import (
    protocol_history_edges,
    resolve_training_protocol,
    validate_protocol_samples,
)
from experiments.modules.prediction_metrics import compute_prediction_metrics


DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

DEFAULT_BINARY_COMPLETIONS = (
    "The answer is: 0",
    "The answer is: 1",
)

GRAPH_PAIR_FEATURE_KEY = "graph_pair_feature"
DEFAULT_GRAPH_PROMPT_SPECIAL_TOKEN = "<graph>"
GRAPH_PROMPT_INSERTION_MODES = ("prepend", "dedicated_slot")


def _safe_l2_normalize(tensor: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    norms = tensor.norm(p=2, dim=dim, keepdim=True)
    normalized = tensor / norms.clamp_min(float(eps))
    return torch.where(norms > float(eps), normalized, tensor)


def get_distributed_training_context():
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = local_rank >= 0 and world_size > 1
    is_main_process = rank == 0
    return {
        "local_rank": local_rank,
        "rank": rank,
        "world_size": world_size,
        "is_distributed": is_distributed,
        "is_main_process": is_main_process,
    }


def resolve_process_cuda_device(local_rank: int) -> int:
    if not torch.cuda.is_available():
        return -1
    device_count = int(torch.cuda.device_count())
    if device_count <= 0:
        return -1
    local_rank = int(local_rank)
    if 0 <= local_rank < device_count:
        return local_rank
    # Local DP workers may set CUDA_VISIBLE_DEVICES to a single GPU while still
    # carrying rank metadata like LOCAL_RANK=1/2/3. In that case only cuda:0 is
    # valid inside the process.
    if device_count == 1:
        return 0
    raise RuntimeError(
        f"Invalid local_rank={local_rank} for process-visible CUDA device_count={device_count}."
    )


def _choose_torch_dtype(model_path: str) -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    model_path_lower = str(model_path).lower()
    if "llama-3.2-1b" in model_path_lower:
        return torch.bfloat16
    return torch.float16


def _load_tokenizer(model_path: str):
    tokenizer_kwargs = {}
    if "qwen" in str(model_path).lower():
        tokenizer_kwargs["trust_remote_code"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def ensure_graph_prompt_special_token(
    tokenizer,
    special_token: str = DEFAULT_GRAPH_PROMPT_SPECIAL_TOKEN,
):
    special_token = str(special_token or "").strip()
    if not special_token:
        raise ValueError("graph prompt special token must be a non-empty string.")
    vocab = tokenizer.get_vocab()
    if special_token in vocab:
        return 0
    existing = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    if special_token in existing:
        return 0
    return tokenizer.add_special_tokens(
        {"additional_special_tokens": existing + [special_token]}
    )


def load_prompt_tokenizer(
    model_path: str,
    *,
    graph_prompt_special_token: str | None = None,
):
    tokenizer = _load_tokenizer(model_path)
    if graph_prompt_special_token:
        ensure_graph_prompt_special_token(
            tokenizer,
            special_token=graph_prompt_special_token,
        )
    return tokenizer


def attach_hadamard_graph_pair_features(
    samples,
    *,
    embeddings,
    entity_id_to_idx,
    feature_key: str = GRAPH_PAIR_FEATURE_KEY,
):
    if embeddings is None or entity_id_to_idx is None:
        raise ValueError("embeddings and entity_id_to_idx are required for graph-prompt features.")
    emb_arr = np.asarray(embeddings)
    if emb_arr.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape={tuple(emb_arr.shape)}")
    feat_dim = int(emb_arr.shape[1])
    if feat_dim <= 0:
        raise ValueError("embeddings feature dimension must be positive.")

    zero_feat = np.zeros((feat_dim,), dtype=np.float32)
    missing_count = 0
    for sample in samples:
        src = int(sample["source_id"])
        dst = int(sample["target_id"])
        src_idx = entity_id_to_idx.get(src)
        dst_idx = entity_id_to_idx.get(dst)
        if src_idx is None or dst_idx is None:
            feat = zero_feat
            missing_count += 1
        else:
            feat = np.asarray(emb_arr[int(src_idx)] * emb_arr[int(dst_idx)], dtype=np.float32)
        sample[feature_key] = feat

    return {
        "feature_key": str(feature_key),
        "feature_dim": int(feat_dim),
        "num_samples": int(len(samples)),
        "num_missing_entity_pairs": int(missing_count),
    }


def _build_lightweight_sample_record(
    *,
    entity_map,
    relation_map,
    source_id: int,
    relation_id: int,
    target_id: int,
    timestamp: int,
    global_avg_interactions: float,
    avg_node_popularity: float,
    query_id: int,
    label: int,
):
    return {
        "source_id": int(source_id),
        "relation_id": int(relation_id),
        "target_id": int(target_id),
        "source_entity": entity_map.get(int(source_id), f"entity_{int(source_id)}"),
        "relation": relation_map.get(int(relation_id), f"relation_{int(relation_id)}"),
        "target_entity": entity_map.get(int(target_id), f"entity_{int(target_id)}"),
        "timestamp": int(timestamp),
        "source_history": [],
        "target_history": [],
        "source_history_entities": [],
        "target_history_entities": [],
        "mutual_history": [],
        "num_past_interactions": 0,
        "num_past_interactions_raw": 0,
        "global_avg_interactions": float(global_avg_interactions),
        "source_popularity": "Modest",
        "target_popularity": "Modest",
        "source_popularity_raw": 0,
        "target_popularity_raw": 0,
        "last_interaction_delta": None,
        "avg_node_popularity": float(avg_node_popularity),
        "common_neighbors": [],
        "common_neighbors_desc": "sorted by popularity",
        "source_history_desc": "most recent",
        "target_history_desc": "most recent",
        "prompt_context_materialized": False,
        "query_id": int(query_id),
        "label": int(label),
        "rrf_score": 0.0,
        "rrf_rank": None,
        "common_neighbor_score": 0.0,
        "common_neighbor_level": "Modest",
        "global_recency_level": "No prior target interactions",
        "itemcf_level": "Low",
        "usercf_level": "Low",
        "last_interaction_str": "No prior interactions",
        "source_popularity_pct": 50.0,
        "target_popularity_pct": 50.0,
        "num_past_interactions_pct": 50.0,
        "common_neighbor_score_pct": 50.0,
        "last_interaction_recency_pct": 0.0,
        "global_recency_pct": 0.0,
        "itemcf_pct": 0.0,
        "usercf_pct": 0.0,
        "heuristic_recency_score": None,
        "heuristic_popularity_score": None,
        "heuristic_past_interactions_score": None,
        "heuristic_resource_allocation_score": None,
        "heuristic_global_recency_score": None,
        "heuristic_itemcf_score": None,
        "heuristic_usercf_score": None,
        "heuristic_semantic_smoothing_score": None,
        "heuristic_recency_rank": None,
        "heuristic_popularity_rank": None,
        "heuristic_past_interactions_rank": None,
        "heuristic_resource_allocation_rank": None,
        "heuristic_global_recency_rank": None,
        "heuristic_semantic_smoothing_rank": None,
    }


def _compute_split_times(edges_df, val_ratio: float, test_ratio: float, apply_gdelt_time_bucket: bool):
    edge_ts_raw = edges_df["ts"].values.astype(np.float64)
    edge_ts_for_split = edge_ts_raw
    if apply_gdelt_time_bucket:
        edge_ts_for_split = np.floor_divide(edge_ts_raw.astype(np.int64), 15).astype(np.float64)
    val_time = float(np.quantile(edge_ts_for_split, 1 - val_ratio - test_ratio))
    test_time = float(np.quantile(edge_ts_for_split, 1 - test_ratio))
    return edge_ts_for_split, val_time, test_time


def _select_positive_edges(
    edges_df,
    *,
    split_name: str,
    val_ratio: float,
    test_ratio: float,
    apply_gdelt_time_bucket: bool,
    train_data_protocol: str = "dtgb_strict",
    data_seed: int = 2020,
):
    protocol = resolve_training_protocol(
        edges_df, split_name=split_name, train_data_protocol=train_data_protocol,
        data_seed=data_seed, val_ratio=val_ratio, test_ratio=test_ratio,
        apply_gdelt_time_bucket=apply_gdelt_time_bucket,
    )
    split_edges = edges_df.loc[protocol.positive_mask].copy()
    if len(protocol.negative_dst_pool) == 0:
        raise RuntimeError(f"No destination nodes available for split {split_name!r}.")
    return split_edges, protocol.negative_dst_pool, protocol.val_time, protocol.test_time


def create_direct_peft_samples(
    edges_df,
    entity_map,
    relation_map,
    *,
    split_name: str,
    num_samples: int,
    negative_ratio: int,
    random_seed: int,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    history_window: int = 47,
    semantic_history: bool = False,
    semantic_topk: Optional[int] = None,
    semantic_history_entity_mode: bool = False,
    common_neighbors_semantic: bool = False,
    semantic_use_smoothing: bool = True,
    semantic_hub_penalty_alpha: float = 0.0,
    semantic_fusion_alpha: float = 1.0,
    semantic_fusion_tau: Optional[float] = None,
    semantic_fusion_recency_speed: float = 1.0,
    history_pool_size: Optional[int] = None,
    history_pool_window: Optional[int] = None,
    history_preserve_recent_k: int = 10,
    embeddings=None,
    entity_id_to_idx=None,
    embedding_model: str = "intfloat/e5-large-v2",
    embedding_cache: Optional[str] = None,
    smooth_time_window: float = 50.0,
    smooth_steps: int = 1,
    smooth_decay_gamma: float = 0.1,
    smooth_undirected: bool = True,
    compute_expert_prediction: bool = True,
    compute_rrf_scores: Optional[bool] = None,
    rrf_k: int = 60,
    rrf_mode: str = "query_local",
    sequential_rank_bins: int = 1024,
    expert_prediction_mode: str = "sequential_running_median",
    expert_prediction_fixed_threshold: float = 0.05,
    rrf_pointwise_pool_size: int = 256,
    rrf_pointwise_num_pools: int = 4,
    rrf_batch_size: int = 200000,
    rrf_heuristics=None,
    key_signal_reference: str = "sequential_global",
    key_signal_fields: Optional[Sequence[str]] = None,
    skip_key_signal_calibration: bool = False,
    include_overall_structural_signal: bool = False,
    overall_signal_low_threshold: float = 0.0475,
    overall_signal_high_threshold: float = 0.0510,
    apply_gdelt_time_bucket: bool = True,
    sampling_strategy: str = "most_recent",
    sampling_skip_recent: int = 0,
    collapse_prompt_duplicates: bool = False,
    prompt_identity_includes_relation: bool = False,
    inductive_num_samples: int = 0,
    return_selection_stats: bool = False,
    defer_prompt_context_materialization: bool = False,
    heuristic_recent_degree_window: float = 30.0,
    train_data_protocol: str = "dtgb_strict",
    data_seed: int = 2020,
):
    if compute_rrf_scores is None:
        compute_rrf_scores = bool(compute_expert_prediction or include_overall_structural_signal)

    protocol = resolve_training_protocol(
        edges_df, split_name=split_name, train_data_protocol=train_data_protocol,
        data_seed=data_seed, val_ratio=val_ratio, test_ratio=test_ratio,
        apply_gdelt_time_bucket=apply_gdelt_time_bucket,
    )
    split_edges = edges_df.loc[protocol.positive_mask].copy()
    negative_dst_pool, val_time, test_time = protocol.negative_dst_pool, protocol.val_time, protocol.test_time
    if len(negative_dst_pool) == 0:
        raise RuntimeError(f"No destination nodes available for split {split_name!r}.")
    strict_training = protocol.name == "dtgb_strict" and protocol.split_name == "train"
    history_edges = protocol_history_edges(edges_df, protocol) if strict_training else edges_df

    if len(split_edges) == 0:
        raise RuntimeError(f"No positive edges found for split {split_name!r}.")

    rng = np.random.default_rng(random_seed)
    positive_edges, selection_stats = select_training_positive_edges(
        split_edges,
        num_samples=num_samples,
        sampling_strategy=sampling_strategy,
        random_seed=random_seed,
        sampling_skip_recent=sampling_skip_recent,
        collapse_prompt_duplicates=collapse_prompt_duplicates,
        prompt_identity_includes_relation=prompt_identity_includes_relation,
        inductive_num_samples=inductive_num_samples,
        first_seen_edges=history_edges,
    )
    print(
        f"{split_name.capitalize()} positive selection: "
        f"selected={selection_stats['selected_positive_queries']}, "
        f"base={selection_stats['base_selected']}, "
        f"reserved_first_touch={selection_stats['inductive_reserved_selected']}, "
        f"all_first_touch={selection_stats['first_touch_selected_total']}, "
        f"collapsed_duplicates={selection_stats['collapsed_duplicate_rows']}"
    )

    num_nodes = len(protocol.observed_train_node_ids) if strict_training else len(entity_map)
    num_edges = len(history_edges)
    total_pairs = (num_nodes * (num_nodes - 1)) / 2
    global_avg_interactions = num_edges / total_pairs if total_pairs > 0 else 0.0
    avg_node_popularity = (2 * num_edges) / num_nodes if num_nodes > 0 else 0.0

    samples = []
    positive_rows = list(positive_edges.itertuples(index=False))
    for query_id, row in enumerate(tqdm(positive_rows, desc=f"Building {split_name} PEFT samples"), start=0):
        source_id = int(row.u)
        relation_id = int(row.r)
        target_id = int(row.i)
        timestamp = int(row.ts)

        positive_sample = _build_lightweight_sample_record(
            entity_map=entity_map,
            relation_map=relation_map,
            source_id=source_id,
            relation_id=relation_id,
            target_id=target_id,
            timestamp=timestamp,
            global_avg_interactions=global_avg_interactions,
            avg_node_popularity=avg_node_popularity,
            query_id=query_id,
            label=1,
        )
        positive_sample.update(
            {
                POSITIVE_SAMPLING_GROUP_COLUMN: str(
                    getattr(row, POSITIVE_SAMPLING_GROUP_COLUMN)
                ),
                PROMPT_MULTIPLICITY_COLUMN: int(
                    getattr(row, PROMPT_MULTIPLICITY_COLUMN)
                ),
                SOURCE_FIRST_TOUCH_COLUMN: bool(
                    getattr(row, SOURCE_FIRST_TOUCH_COLUMN)
                ),
                TARGET_FIRST_TOUCH_COLUMN: bool(
                    getattr(row, TARGET_FIRST_TOUCH_COLUMN)
                ),
            }
        )
        samples.append(positive_sample)

        sampled_negatives = negative_dst_pool[
            rng.integers(0, len(negative_dst_pool), size=int(negative_ratio))
        ]
        for neg_target_id in sampled_negatives:
            negative_sample = _build_lightweight_sample_record(
                entity_map=entity_map,
                relation_map=relation_map,
                source_id=source_id,
                relation_id=relation_id,
                target_id=int(neg_target_id),
                timestamp=timestamp,
                global_avg_interactions=global_avg_interactions,
                avg_node_popularity=avg_node_popularity,
                query_id=query_id,
                label=0,
            )
            negative_sample.update(
                {
                    POSITIVE_SAMPLING_GROUP_COLUMN: str(
                        getattr(row, POSITIVE_SAMPLING_GROUP_COLUMN)
                    ),
                    PROMPT_MULTIPLICITY_COLUMN: int(
                        getattr(row, PROMPT_MULTIPLICITY_COLUMN)
                    ),
                }
            )
            samples.append(negative_sample)

    if not samples:
        print(
            f"{split_name.capitalize()} samples ready: 0 total "
            f"(0 positive queries, negative_ratio={negative_ratio}, "
            f"val_time={val_time:.0f}, test_time={test_time:.0f})."
        )
        selection_stats["training_protocol"] = {
            **protocol.metadata, "sample_audit": validate_protocol_samples(samples, protocol),
        }
        if return_selection_stats:
            return samples, selection_stats
        return samples

    from experiments.modules.llm_lp.training_protocol import tag_training_history
    tag_training_history(samples, protocol)
    # Fail closed before any prompt construction, then recheck materialized evidence.
    validate_protocol_samples(samples, protocol)
    if not defer_prompt_context_materialization:
        materialize_samples_prompt_context(
            samples=samples,
            edges_df=history_edges,
            entity_map=entity_map,
            history_window=history_window,
            semantic_history=semantic_history,
            semantic_topk=semantic_topk,
            semantic_history_entity_mode=semantic_history_entity_mode,
            common_neighbors_semantic=common_neighbors_semantic,
            semantic_use_smoothing=semantic_use_smoothing,
            semantic_hub_penalty_alpha=semantic_hub_penalty_alpha,
            semantic_fusion_alpha=semantic_fusion_alpha,
            semantic_fusion_tau=semantic_fusion_tau,
            semantic_fusion_recency_speed=semantic_fusion_recency_speed,
            history_pool_size=history_pool_size,
            history_pool_window=history_pool_window,
            history_preserve_recent_k=history_preserve_recent_k,
            embeddings=embeddings,
            entity_id_to_idx=entity_id_to_idx,
            embedding_model=embedding_model,
            embedding_cache=embedding_cache,
            populate_prompt_lists=True,
            calibrate_key_signals=False,
            heuristic_recent_degree_window=heuristic_recent_degree_window,
            monitor_label=f"Materializing {split_name} prompt context",
        )

    finalize_test_samples(
        samples=samples,
        edges_df=history_edges,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        negative_ratio=negative_ratio,
        random_seed=random_seed,
        build_prompt_features=(not defer_prompt_context_materialization),
        compute_expert_prediction=compute_expert_prediction,
        compute_rrf_scores=compute_rrf_scores,
        rrf_k=rrf_k,
        rrf_mode=rrf_mode,
        sequential_rank_bins=sequential_rank_bins,
        expert_prediction_mode=expert_prediction_mode,
        expert_prediction_fixed_threshold=expert_prediction_fixed_threshold,
        rrf_pointwise_pool_size=rrf_pointwise_pool_size,
        rrf_pointwise_num_pools=rrf_pointwise_num_pools,
        rrf_batch_size=rrf_batch_size,
        rrf_heuristics=rrf_heuristics,
        key_signal_reference=key_signal_reference,
        key_signal_fields=key_signal_fields,
        skip_key_signal_calibration=skip_key_signal_calibration,
        include_overall_structural_signal=include_overall_structural_signal,
        overall_signal_low_threshold=overall_signal_low_threshold,
        overall_signal_high_threshold=overall_signal_high_threshold,
        apply_gdelt_time_bucket=apply_gdelt_time_bucket,
        embeddings=embeddings,
        entity_id_to_idx=entity_id_to_idx,
        smooth_time_window=smooth_time_window,
        smooth_steps=smooth_steps,
        smooth_decay_gamma=smooth_decay_gamma,
        smooth_undirected=smooth_undirected,
        heuristic_recent_degree_window=heuristic_recent_degree_window,
        # Supplying the canonical observed pool prevents finalize_test_samples
        # from deriving new split quantiles from the already-filtered graph.
        train_entity_ids=(protocol.observed_train_node_ids[protocol.observed_train_node_ids != 0].tolist()
                          if protocol.name == "dtgb_strict" else None),
    )
    selection_stats["training_protocol"] = {
        **protocol.metadata, "sample_audit": validate_protocol_samples(samples, protocol),
    }

    print(
        f"{split_name.capitalize()} samples ready: {len(samples)} total "
        f"({len(positive_rows)} positive queries, negative_ratio={negative_ratio}, "
        f"val_time={val_time:.0f}, test_time={test_time:.0f})."
    )
    if return_selection_stats:
        return samples, selection_stats
    return samples


def _tokenize_prompt_completion(
    *,
    tokenizer,
    prompt_text: str,
    completion_text: str,
    max_length: int,
):
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt_text + completion_text, add_special_tokens=False)

    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Prompt tokenization is not a prefix of prompt+completion tokenization.")

    completion_ids = full_ids[len(prompt_ids) :]
    if not completion_ids:
        raise ValueError("Completion text did not produce any tokens.")

    truncated = False
    max_prompt_tokens = int(max_length) - len(completion_ids)
    if max_prompt_tokens <= 0:
        return None
    if len(prompt_ids) > max_prompt_tokens:
        prompt_ids = prompt_ids[-max_prompt_tokens:]
        truncated = True

    input_ids = prompt_ids + completion_ids
    labels = ([-100] * len(prompt_ids)) + completion_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "prompt_length": len(prompt_ids),
        "completion_length": len(completion_ids),
        "total_length": len(input_ids),
        "truncated": truncated,
    }


def _tokenize_forced_binary_label(
    *,
    tokenizer,
    prompt_text: str,
    label_text: str,
    max_length: int,
):
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt_text + label_text, add_special_tokens=False)

    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Prompt tokenization is not a prefix of prompt+label tokenization.")

    label_ids = full_ids[len(prompt_ids) :]
    if not label_ids:
        raise ValueError("Binary label text did not produce any tokens.")

    truncated = False
    max_prompt_tokens = int(max_length) - len(label_ids)
    if max_prompt_tokens <= 0:
        return None
    if len(prompt_ids) > max_prompt_tokens:
        prompt_ids = prompt_ids[-max_prompt_tokens:]
        truncated = True

    input_ids = prompt_ids + label_ids
    labels = ([-100] * len(prompt_ids)) + label_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "prompt_length": len(prompt_ids),
        "completion_length": len(label_ids),
        "total_length": len(input_ids),
        "truncated": truncated,
    }


class PromptResponseDataset(Dataset):
    def __init__(self, examples: Sequence[dict]):
        self.examples = list(examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class PromptResponseCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = int(tokenizer.pad_token_id)

    def __call__(self, features: Sequence[dict]):
        batch_size = len(features)
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

        graph_features = []
        has_graph_features = any(GRAPH_PAIR_FEATURE_KEY in feature for feature in features)
        if has_graph_features and (not all(GRAPH_PAIR_FEATURE_KEY in feature for feature in features)):
            raise ValueError("All features in a batch must include graph_pair_feature when enabled.")

        for row_idx, feature in enumerate(features):
            seq_len = len(feature["input_ids"])
            input_ids[row_idx, :seq_len] = torch.tensor(feature["input_ids"], dtype=torch.long)
            attention_mask[row_idx, :seq_len] = torch.tensor(feature["attention_mask"], dtype=torch.long)
            labels[row_idx, :seq_len] = torch.tensor(feature["labels"], dtype=torch.long)
            if has_graph_features:
                graph_features.append(torch.as_tensor(feature[GRAPH_PAIR_FEATURE_KEY], dtype=torch.float32))

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if has_graph_features:
            batch[GRAPH_PAIR_FEATURE_KEY] = torch.stack(graph_features, dim=0)
        return batch


def build_prompt_response_examples(
    *,
    samples,
    tokenizer,
    entity_map,
    relation_map,
    summary_map=None,
    summary_mode: str = "off",
    summary_max_chars: int = 120,
    history_window: int = 47,
    include_key_signals: bool = True,
    include_expert_prediction: bool = True,
    include_overall_structural_signal: bool = False,
    overall_structural_signal_name: str = "Overall structural signal",
    key_signal_fields: Sequence[str] = DEFAULT_KEY_SIGNAL_FIELDS,
    use_raw_key_signals: bool = False,
    use_percentile_key_signals: bool = False,
    include_edge_type: bool = False,
    mutual_timestamps_only: bool = False,
    mutual_timestamps_dedup: bool = False,
    mutual_summary_count_recency: bool = False,
    common_neighbors_names_only: bool = False,
    compact_common_neighbors_top_k: int = 0,
    compact_common_neighbors_novel_only: bool = False,
    history_table_aliases: bool = False,
    anonymous_entity_aliases: bool = False,
    natural_grouped_history: bool = False,
    natural_activity_summary: bool = False,
    natural_neighbor_names_only: bool = False,
    natural_activity_compact_top3: bool = False,
    natural_activity_top_k: int = 3,
    ablate_mutual_history: bool = False,
    ablate_common_neighbors: bool = False,
    ablate_source_history: bool = False,
    ablate_target_history: bool = False,
    ablate_source_target_history: bool = False,
    ablate_reasoning_guidance: bool = False,
    prompt_variant: str = "gdelt",
    max_length: int = 4096,
    include_graph_pair_feature: bool = False,
    graph_pair_feature_key: str = GRAPH_PAIR_FEATURE_KEY,
    graph_prompt_special_token: str | None = None,
    graph_prompt_num_tokens: int = 0,
):
    examples = []
    stats = {
        "num_samples": 0,
        "num_examples": 0,
        "num_truncated": 0,
        "num_skipped_too_long": 0,
        "avg_prompt_tokens": 0.0,
        "p95_prompt_tokens": 0.0,
        "max_prompt_tokens": 0,
        "avg_total_tokens": 0.0,
        "p95_total_tokens": 0.0,
        "max_total_tokens": 0,
    }
    prompt_lengths = []
    total_lengths = []

    iterator = tqdm(samples, total=len(samples), desc="Tokenizing PEFT supervision")
    for sample in iterator:
        prompt_text = _build_forced_binary_scoring_prompt(
            _build_prompt_for_sample(
                sample=sample,
                tokenizer=tokenizer,
                entity_map=entity_map,
                summary_map=summary_map,
                summary_mode=summary_mode,
                summary_max_chars=summary_max_chars,
                relation_map=relation_map,
                history_window=history_window,
                include_key_signals=include_key_signals,
                include_expert_prediction=include_expert_prediction,
                include_overall_structural_signal=include_overall_structural_signal,
                overall_structural_signal_name=overall_structural_signal_name,
                key_signal_fields=key_signal_fields,
                use_raw_key_signals=use_raw_key_signals,
                use_percentile_key_signals=use_percentile_key_signals,
                include_edge_type=include_edge_type,
                use_cot=False,
                mutual_timestamps_only=mutual_timestamps_only,
                mutual_timestamps_dedup=mutual_timestamps_dedup,
                mutual_summary_count_recency=mutual_summary_count_recency,
                common_neighbors_names_only=common_neighbors_names_only,
                compact_common_neighbors_top_k=compact_common_neighbors_top_k,
                compact_common_neighbors_novel_only=compact_common_neighbors_novel_only,
                history_table_aliases=history_table_aliases,
                anonymous_entity_aliases=anonymous_entity_aliases,
                natural_grouped_history=natural_grouped_history,
                natural_activity_summary=natural_activity_summary,
                natural_neighbor_names_only=natural_neighbor_names_only,
                natural_activity_compact_top3=natural_activity_compact_top3,
                natural_activity_top_k=natural_activity_top_k,
                ablate_mutual_history=ablate_mutual_history,
                ablate_common_neighbors=ablate_common_neighbors,
                ablate_source_history=ablate_source_history,
                ablate_target_history=ablate_target_history,
                ablate_source_target_history=ablate_source_target_history,
                ablate_reasoning_guidance=ablate_reasoning_guidance,
                no_cot_output_0_100=False,
                prompt_variant=prompt_variant,
                few_shot_examples=None,
                graph_prompt_special_token=graph_prompt_special_token,
                graph_prompt_num_tokens=graph_prompt_num_tokens,
            )
        )
        label_text = str(int(sample["label"]))
        example = _tokenize_forced_binary_label(
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            label_text=label_text,
            max_length=max_length,
        )
        stats["num_samples"] += 1
        if example is None:
            stats["num_skipped_too_long"] += 1
            continue

        prompt_lengths.append(int(example["prompt_length"]))
        total_lengths.append(int(example["total_length"]))
        if example["truncated"]:
            stats["num_truncated"] += 1

        row = {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
            "labels": example["labels"],
        }
        if include_graph_pair_feature:
            feature_value = sample.get(graph_pair_feature_key)
            if feature_value is None:
                raise ValueError(
                    f"Sample missing required {graph_pair_feature_key!r} for graph-prompt mode."
                )
            row[graph_pair_feature_key] = np.asarray(feature_value, dtype=np.float32)
        examples.append(row)

    stats["num_examples"] = len(examples)
    if prompt_lengths:
        stats["avg_prompt_tokens"] = float(np.mean(prompt_lengths))
        stats["p95_prompt_tokens"] = float(np.percentile(prompt_lengths, 95))
        stats["max_prompt_tokens"] = int(np.max(prompt_lengths))
    if total_lengths:
        stats["avg_total_tokens"] = float(np.mean(total_lengths))
        stats["p95_total_tokens"] = float(np.percentile(total_lengths, 95))
        stats["max_total_tokens"] = int(np.max(total_lengths))
    return examples, stats


class GraphPromptProjector(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        model_dim: int,
        num_virtual_tokens: int = 4,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        output_l2_norm: bool = False,
        zero_init_output: bool = False,
        mean_token_init: bool = False,
        mean_token_embedding: torch.Tensor | None = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.num_virtual_tokens = int(num_virtual_tokens)
        hidden_dim = int(hidden_dim)
        if self.input_dim <= 0:
            raise ValueError("GraphPromptProjector input_dim must be > 0.")
        if self.model_dim <= 0:
            raise ValueError("GraphPromptProjector model_dim must be > 0.")
        if self.num_virtual_tokens <= 0:
            raise ValueError("GraphPromptProjector num_virtual_tokens must be > 0.")
        if hidden_dim <= 0:
            raise ValueError("GraphPromptProjector hidden_dim must be > 0.")

        self.input_norm = nn.LayerNorm(self.input_dim)
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(hidden_dim, self.num_virtual_tokens * self.model_dim)
        self.output_l2_norm = bool(output_l2_norm)
        self.zero_init_output = bool(zero_init_output)
        self.mean_token_init = bool(mean_token_init)
        if self.zero_init_output and self.mean_token_init:
            raise ValueError("zero_init_output and mean_token_init are mutually exclusive.")
        if self.zero_init_output:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
        elif self.mean_token_init:
            if mean_token_embedding is None:
                raise ValueError("mean_token_embedding is required when mean_token_init=True.")
            mean_token_embedding = torch.as_tensor(mean_token_embedding, dtype=self.fc2.bias.dtype)
            if mean_token_embedding.numel() != int(self.model_dim):
                raise ValueError(
                    "mean_token_embedding size mismatch: "
                    f"expected {self.model_dim}, got {mean_token_embedding.numel()}."
                )
            repeated_bias = mean_token_embedding.reshape(1, self.model_dim).repeat(
                self.num_virtual_tokens,
                1,
            )
            nn.init.zeros_(self.fc2.weight)
            with torch.no_grad():
                self.fc2.bias.copy_(repeated_bias.reshape(-1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(features)
        x = self.fc1(x)
        x = self.hidden_norm(x)
        x = self.act(x)
        x = self.dropout(x)
        out = self.fc2(x)
        out = out.view(features.size(0), self.num_virtual_tokens, self.model_dim)
        if self.output_l2_norm:
            out = _safe_l2_normalize(out, dim=-1)
        return out


class GraphPromptMLPOnlyModel(nn.Module):
    supports_graph_pair_feature = True

    def __init__(
        self,
        *,
        base_model: nn.Module,
        graph_feature_dim: int,
        num_virtual_tokens: int = 4,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        output_l2_norm: bool = False,
        zero_init_output: bool = False,
        mean_token_init: bool = False,
        token_space_align_weight: float = 0.0,
        graph_prompt_injection_mode: str = "prepend",
        graph_prompt_special_token: str | None = None,
        graph_prompt_special_token_id: int | None = None,
        freeze_base_model: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = getattr(base_model, "config", None)
        self.model_dim = self._resolve_model_dim(base_model)
        self.graph_projector = GraphPromptProjector(
            input_dim=int(graph_feature_dim),
            model_dim=int(self.model_dim),
            num_virtual_tokens=int(num_virtual_tokens),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            output_l2_norm=bool(output_l2_norm),
            zero_init_output=bool(zero_init_output),
            mean_token_init=bool(mean_token_init),
            mean_token_embedding=base_model.get_input_embeddings().weight.detach().mean(dim=0),
        )
        self.num_virtual_tokens = int(num_virtual_tokens)
        self.graph_feature_dim = int(graph_feature_dim)
        self.output_l2_norm = bool(output_l2_norm)
        self.zero_init_output = bool(zero_init_output)
        self.mean_token_init = bool(mean_token_init)
        self.token_space_align_weight = float(token_space_align_weight)
        graph_prompt_injection_mode = str(graph_prompt_injection_mode or "prepend").strip().lower()
        if graph_prompt_injection_mode not in GRAPH_PROMPT_INSERTION_MODES:
            raise ValueError(
                f"graph_prompt_injection_mode must be one of {GRAPH_PROMPT_INSERTION_MODES}, "
                f"got {graph_prompt_injection_mode!r}."
            )
        self.graph_prompt_injection_mode = graph_prompt_injection_mode
        self.graph_prompt_special_token = (
            str(graph_prompt_special_token).strip() if graph_prompt_special_token else None
        )
        self.graph_prompt_special_token_id = (
            None if graph_prompt_special_token_id is None else int(graph_prompt_special_token_id)
        )
        if self.graph_prompt_injection_mode == "dedicated_slot":
            if self.graph_prompt_special_token_id is None:
                raise ValueError(
                    "graph_prompt_special_token_id is required for dedicated_slot graph prompting."
                )
        if freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False

    @staticmethod
    def _resolve_model_dim(base_model) -> int:
        cfg = getattr(base_model, "config", None)
        for attr in ("hidden_size", "d_model", "n_embd"):
            value = getattr(cfg, attr, None) if cfg is not None else None
            if value is not None:
                return int(value)
        emb = base_model.get_input_embeddings()
        if emb is None or not hasattr(emb, "embedding_dim"):
            raise RuntimeError("Cannot resolve base model hidden size for graph projector.")
        return int(emb.embedding_dim)

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def _compute_token_space_alignment_loss(self, prefix_embeds: torch.Tensor) -> torch.Tensor:
        token_embed_weight = self.get_input_embeddings().weight.detach()
        prefix_unit = _safe_l2_normalize(prefix_embeds.float(), dim=-1)
        token_unit = _safe_l2_normalize(token_embed_weight.float(), dim=-1)
        flat_prefix = prefix_unit.reshape(-1, prefix_unit.size(-1))
        cosine_scores = torch.matmul(flat_prefix, token_unit.transpose(0, 1))
        nearest_cosine = cosine_scores.max(dim=-1).values
        return (1.0 - nearest_cosine).mean().to(prefix_embeds.dtype)

    def _build_prefix_embeds(
        self,
        *,
        input_embeds: torch.Tensor,
        graph_pair_feature: torch.Tensor,
    ) -> torch.Tensor:
        projector_param = next(self.graph_projector.parameters(), None)
        projector_dtype = (
            projector_param.dtype if projector_param is not None else input_embeds.dtype
        )
        graph_pair_feature = graph_pair_feature.to(
            device=input_embeds.device,
            dtype=projector_dtype,
        )
        prefix_embeds = self.graph_projector(graph_pair_feature)
        if prefix_embeds.dtype != input_embeds.dtype:
            prefix_embeds = prefix_embeds.to(dtype=input_embeds.dtype)
        return prefix_embeds

    def _inject_graph_prompt_into_dedicated_slot(
        self,
        *,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        prefix_embeds: torch.Tensor,
    ) -> torch.Tensor:
        slot_token_id = int(self.graph_prompt_special_token_id)
        slot_mask = input_ids.eq(slot_token_id)
        slot_counts = slot_mask.sum(dim=1)
        expected = int(self.num_virtual_tokens)
        if not bool(torch.all(slot_counts == expected).item()):
            observed = sorted({int(v) for v in slot_counts.detach().cpu().tolist()})
            raise ValueError(
                "Dedicated graph-token slot count mismatch. "
                f"Expected {expected} occurrences of token_id={slot_token_id}, got counts={observed}."
            )
        full_embeds = input_embeds.clone()
        for row_idx in range(int(input_ids.size(0))):
            slot_positions = torch.nonzero(slot_mask[row_idx], as_tuple=False).flatten()
            full_embeds[row_idx, slot_positions, :] = prefix_embeds[row_idx]
        return full_embeds

    def build_input_embeds(
        self,
        *,
        input_ids: torch.Tensor,
        graph_pair_feature: torch.Tensor | None = None,
    ):
        if input_ids is None:
            raise ValueError("input_ids are required for graph prompt input-embed construction.")
        input_embeds = self.get_input_embeddings()(input_ids)
        if graph_pair_feature is None:
            return input_embeds, None
        prefix_embeds = self._build_prefix_embeds(
            input_embeds=input_embeds,
            graph_pair_feature=graph_pair_feature,
        )
        if self.graph_prompt_injection_mode == "dedicated_slot":
            full_embeds = self._inject_graph_prompt_into_dedicated_slot(
                input_ids=input_ids,
                input_embeds=input_embeds,
                prefix_embeds=prefix_embeds,
            )
        else:
            full_embeds = torch.cat([prefix_embeds, input_embeds], dim=1)
        return full_embeds, prefix_embeds

    def forward(
        self,
        *,
        input_ids=None,
        attention_mask=None,
        labels=None,
        graph_pair_feature=None,
        **kwargs,
    ):
        if graph_pair_feature is None:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        if input_ids is None:
            raise ValueError("input_ids are required when graph_pair_feature is provided.")

        full_embeds, prefix_embeds = self.build_input_embeds(
            input_ids=input_ids,
            graph_pair_feature=graph_pair_feature,
        )
        if attention_mask is None:
            attention_mask = torch.ones(
                input_ids.size(),
                dtype=torch.long,
                device=input_ids.device,
            )
        if self.graph_prompt_injection_mode == "dedicated_slot":
            full_attention = attention_mask
            full_labels = labels
        else:
            prefix_mask = torch.ones(
                (attention_mask.size(0), self.num_virtual_tokens),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            full_attention = torch.cat([prefix_mask, attention_mask], dim=1)
            full_labels = None
            if labels is not None:
                prefix_labels = torch.full(
                    (labels.size(0), self.num_virtual_tokens),
                    -100,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                full_labels = torch.cat([prefix_labels, labels], dim=1)

        kwargs = dict(kwargs)
        kwargs.pop("input_ids", None)
        kwargs.pop("inputs_embeds", None)
        outputs = self.base_model(
            inputs_embeds=full_embeds,
            attention_mask=full_attention,
            labels=full_labels,
            **kwargs,
        )
        if (
            labels is not None
            and self.token_space_align_weight > 0.0
            and getattr(outputs, "loss", None) is not None
        ):
            align_loss = self._compute_token_space_alignment_loss(prefix_embeds)
            outputs.loss = outputs.loss + (self.token_space_align_weight * align_loss)
            try:
                outputs["token_space_align_loss"] = align_loss.detach()
            except Exception:
                pass
        return outputs

    def save_graph_projector(self, output_dir: str, *, base_model_path: str):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            self.graph_projector.state_dict(),
            os.path.join(output_dir, "graph_projector.pt"),
        )
        payload = {
            "base_model_path": str(base_model_path),
            "graph_feature_dim": int(self.graph_feature_dim),
            "num_virtual_tokens": int(self.num_virtual_tokens),
            "model_dim": int(self.model_dim),
            "output_l2_norm": bool(self.output_l2_norm),
            "zero_init_output": bool(self.zero_init_output),
            "mean_token_init": bool(self.mean_token_init),
            "graph_prompt_injection_mode": str(self.graph_prompt_injection_mode),
            "graph_prompt_special_token": self.graph_prompt_special_token,
        }
        with open(
            os.path.join(output_dir, "graph_projector_config.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)


def load_frozen_base_model(
    *,
    model_path: str,
    tokenizer=None,
):
    if tokenizer is None:
        tokenizer = _load_tokenizer(model_path)

    trust_remote_code = "qwen" in str(model_path).lower()
    torch_dtype = _choose_torch_dtype(model_path)
    dist_ctx = get_distributed_training_context()
    resolved_cuda_device = resolve_process_cuda_device(dist_ctx["local_rank"])
    if resolved_cuda_device >= 0:
        torch.cuda.set_device(resolved_cuda_device)

    model_kwargs = {
        "dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = True

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs,
    )
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    return model, tokenizer


def load_peft_training_model(
    *,
    model_path: str,
    tokenizer=None,
    load_in_4bit: bool = False,
    gradient_checkpointing: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Sequence[str] = DEFAULT_LORA_TARGET_MODULES,
    lora_last_n_layers: int = 0,
):
    if tokenizer is None:
        tokenizer = _load_tokenizer(model_path)

    trust_remote_code = "qwen" in str(model_path).lower()
    torch_dtype = _choose_torch_dtype(model_path)
    dist_ctx = get_distributed_training_context()
    resolved_cuda_device = resolve_process_cuda_device(dist_ctx["local_rank"])
    if resolved_cuda_device >= 0:
        torch.cuda.set_device(resolved_cuda_device)

    model_kwargs = {
        "dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = True

    if load_in_4bit:
        if importlib.util.find_spec("bitsandbytes") is None:
            raise ImportError(
                "bitsandbytes is not installed, but --load_in_4bit was requested."
            )
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs,
    )
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    last_n_layers = int(lora_last_n_layers)
    if last_n_layers < 0:
        raise ValueError("lora_last_n_layers must be >= 0")

    layers_to_transform = None
    if last_n_layers > 0:
        text_config = model.config
        get_text_config = getattr(text_config, "get_text_config", None)
        if callable(get_text_config):
            try:
                text_config = get_text_config()
            except TypeError:
                text_config = get_text_config(decoder=True)
        num_hidden_layers = getattr(text_config, "num_hidden_layers", None)
        if num_hidden_layers is None:
            raise ValueError(
                "Could not determine num_hidden_layers; cannot apply --lora_last_n_layers."
            )
        num_hidden_layers = int(num_hidden_layers)
        if last_n_layers > num_hidden_layers:
            raise ValueError(
                "lora_last_n_layers cannot exceed the model depth: "
                f"requested {last_n_layers}, model has {num_hidden_layers}."
            )
        layers_to_transform = list(
            range(num_hidden_layers - last_n_layers, num_hidden_layers)
        )
        if dist_ctx["is_main_process"]:
            print(
                "Restricting LoRA to the final transformer layers: "
                f"{layers_to_transform[0]}..{layers_to_transform[-1]} "
                f"({last_n_layers}/{num_hidden_layers} layers)."
            )

    peft_config = LoraConfig(
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        target_modules=list(lora_target_modules),
        layers_to_transform=layers_to_transform,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    return model, tokenizer


def summarize_trainable_parameters(model):
    trainable = 0
    total = 0
    for param in model.parameters():
        total += int(param.numel())
        if param.requires_grad:
            trainable += int(param.numel())
    ratio = (100.0 * trainable / total) if total > 0 else 0.0
    return {
        "trainable_params": int(trainable),
        "total_params": int(total),
        "trainable_percent": float(ratio),
    }


def _resolve_model_device(model):
    model_device = getattr(model, "device", None)
    if model_device is not None:
        return model_device
    return next(model.parameters()).device


def _graph_prompt_logit_prefix_offset(model, batch) -> int:
    if GRAPH_PAIR_FEATURE_KEY not in batch:
        return 0
    if getattr(model, "graph_prompt_injection_mode", "prepend") != "prepend":
        return 0
    return int(getattr(model, "num_virtual_tokens", 0))


def _sequence_log_likelihoods(model, batch, *, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    model_kwargs = {}
    if GRAPH_PAIR_FEATURE_KEY in batch:
        model_kwargs[GRAPH_PAIR_FEATURE_KEY] = batch[GRAPH_PAIR_FEATURE_KEY].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **model_kwargs,
        )
        logits = outputs.logits
    prefix_len = _graph_prompt_logit_prefix_offset(model, batch)
    if prefix_len > 0 and logits.size(1) > prefix_len:
        logits = logits[:, prefix_len:, :]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    flat_token_nll = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )
    token_nll = flat_token_nll.view(shift_labels.size(0), shift_labels.size(1))
    valid_mask = shift_labels != -100
    seq_logps = -(token_nll * valid_mask).sum(dim=-1)

    del outputs
    del logits
    del shift_logits
    del flat_token_nll
    del token_nll
    return seq_logps.detach().cpu()


def _build_binary_token_groups(tokenizer):
    zero_ids = []
    one_ids = []
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    for token_id in range(vocab_size):
        token_text = tokenizer.decode([token_id]).strip()
        if token_text == "0":
            zero_ids.append(token_id)
        elif token_text == "1":
            one_ids.append(token_id)
    if not zero_ids or not one_ids:
        raise RuntimeError("Could not resolve tokenizer token ids for binary labels 0/1.")
    return {
        "0": torch.tensor(sorted(set(zero_ids)), dtype=torch.long),
        "1": torch.tensor(sorted(set(one_ids)), dtype=torch.long),
    }


def _score_binary_next_token_batch(model, batch, *, device, binary_token_groups):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    model_kwargs = {}
    if GRAPH_PAIR_FEATURE_KEY in batch:
        model_kwargs[GRAPH_PAIR_FEATURE_KEY] = batch[GRAPH_PAIR_FEATURE_KEY].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **model_kwargs,
        )
        logits = outputs.logits

    last_indices = attention_mask.sum(dim=1) - 1
    prefix_offset = _graph_prompt_logit_prefix_offset(model, batch)
    if prefix_offset > 0:
        last_indices = last_indices + prefix_offset
    batch_indices = torch.arange(logits.size(0), device=logits.device)
    next_token_logits = logits[batch_indices, last_indices, :]

    zero_ids = binary_token_groups["0"].to(logits.device)
    one_ids = binary_token_groups["1"].to(logits.device)
    zero_scores = torch.logsumexp(next_token_logits.index_select(dim=-1, index=zero_ids), dim=-1)
    one_scores = torch.logsumexp(next_token_logits.index_select(dim=-1, index=one_ids), dim=-1)
    score = torch.sigmoid(one_scores - zero_scores)

    del outputs
    del logits
    del next_token_logits
    return score.detach().cpu().tolist()


def extract_binary_scores_from_training_forward(
    *,
    model,
    batch,
    logits,
    binary_token_groups,
):
    labels = batch["labels"].to(logits.device)
    prefix_offset = _graph_prompt_logit_prefix_offset(model, batch)

    zero_ids = binary_token_groups["0"].to(logits.device)
    one_ids = binary_token_groups["1"].to(logits.device)

    scores = []
    gold_labels = []
    for row_idx in range(int(labels.size(0))):
        label_positions = torch.nonzero(labels[row_idx] != -100, as_tuple=False).flatten()
        if label_positions.numel() == 0:
            continue
        first_label_pos = int(label_positions[0].item())
        if first_label_pos <= 0:
            continue
        gold_token_id = int(labels[row_idx, first_label_pos].item())
        if bool((zero_ids == gold_token_id).any().item()):
            gold_label = 0
        elif bool((one_ids == gold_token_id).any().item()):
            gold_label = 1
        else:
            continue

        score_pos = first_label_pos - 1 + prefix_offset
        if score_pos < 0 or score_pos >= int(logits.size(1)):
            continue
        next_token_logits = logits[row_idx, score_pos, :]
        zero_scores = torch.logsumexp(next_token_logits.index_select(dim=-1, index=zero_ids), dim=-1)
        one_scores = torch.logsumexp(next_token_logits.index_select(dim=-1, index=one_ids), dim=-1)
        score = torch.sigmoid(one_scores - zero_scores)
        scores.append(float(score.detach().item()))
        gold_labels.append(int(gold_label))
    return scores, gold_labels


def evaluate_binary_completion_likelihood(
    *,
    model,
    tokenizer,
    samples,
    entity_map,
    relation_map,
    summary_map=None,
    summary_mode: str = "off",
    summary_max_chars: int = 120,
    history_window: int = 47,
    include_key_signals: bool = True,
    include_expert_prediction: bool = True,
    include_overall_structural_signal: bool = False,
    overall_structural_signal_name: str = "Overall structural signal",
    key_signal_fields: Sequence[str] = DEFAULT_KEY_SIGNAL_FIELDS,
    use_raw_key_signals: bool = False,
    use_percentile_key_signals: bool = False,
    include_edge_type: bool = False,
    mutual_timestamps_only: bool = False,
    mutual_timestamps_dedup: bool = False,
    mutual_summary_count_recency: bool = False,
    common_neighbors_names_only: bool = False,
    compact_common_neighbors_top_k: int = 0,
    compact_common_neighbors_novel_only: bool = False,
    history_table_aliases: bool = False,
    anonymous_entity_aliases: bool = False,
    natural_grouped_history: bool = False,
    natural_activity_summary: bool = False,
    natural_neighbor_names_only: bool = False,
    natural_activity_compact_top3: bool = False,
    natural_activity_top_k: int = 3,
    ablate_mutual_history: bool = False,
    ablate_common_neighbors: bool = False,
    ablate_source_history: bool = False,
    ablate_target_history: bool = False,
    ablate_source_target_history: bool = False,
    ablate_reasoning_guidance: bool = False,
    prompt_variant: str = "gdelt",
    graph_prompt_special_token: str | None = None,
    graph_prompt_num_tokens: int = 0,
    max_length: int = 4096,
    batch_size: int = 4,
    negative_ratio: Optional[int] = None,
    dtgb_eval_batch_size: Optional[int] = None,
):
    model_was_training = model.training
    model.eval()
    device = _resolve_model_device(model)
    collator = PromptResponseCollator(tokenizer)
    binary_token_groups = _build_binary_token_groups(tokenizer)

    predictions = []
    labels = []
    prompt_buffer = []
    label_buffer = []

    def flush_prompt_buffer():
        if not prompt_buffer:
            return
        prompt_examples = []
        for prompt_item in prompt_buffer:
            prompt_text = prompt_item["prompt_text"]
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            if len(prompt_ids) > int(max_length):
                prompt_ids = prompt_ids[-int(max_length):]
            row = {
                "input_ids": prompt_ids,
                "attention_mask": [1] * len(prompt_ids),
                "labels": [-100] * len(prompt_ids),
            }
            if GRAPH_PAIR_FEATURE_KEY in prompt_item:
                row[GRAPH_PAIR_FEATURE_KEY] = np.asarray(
                    prompt_item[GRAPH_PAIR_FEATURE_KEY],
                    dtype=np.float32,
                )
            prompt_examples.append(row)
        batch = collator(prompt_examples)
        predictions.extend(
            _score_binary_next_token_batch(
                model,
                batch,
                device=device,
                binary_token_groups=binary_token_groups,
            )
        )
        labels.extend(int(label) for label in label_buffer)
        prompt_buffer.clear()
        label_buffer.clear()

    for sample in tqdm(samples, total=len(samples), desc="Scoring validation prompts"):
        prompt_text = _build_forced_binary_scoring_prompt(
            _build_prompt_for_sample(
                sample=sample,
                tokenizer=tokenizer,
                entity_map=entity_map,
                summary_map=summary_map,
                summary_mode=summary_mode,
                summary_max_chars=summary_max_chars,
                relation_map=relation_map,
                history_window=history_window,
                include_key_signals=include_key_signals,
                include_expert_prediction=include_expert_prediction,
                include_overall_structural_signal=include_overall_structural_signal,
                overall_structural_signal_name=overall_structural_signal_name,
                key_signal_fields=key_signal_fields,
                use_raw_key_signals=use_raw_key_signals,
                use_percentile_key_signals=use_percentile_key_signals,
                include_edge_type=include_edge_type,
                use_cot=False,
                mutual_timestamps_only=mutual_timestamps_only,
                mutual_timestamps_dedup=mutual_timestamps_dedup,
                mutual_summary_count_recency=mutual_summary_count_recency,
                common_neighbors_names_only=common_neighbors_names_only,
                compact_common_neighbors_top_k=compact_common_neighbors_top_k,
                compact_common_neighbors_novel_only=compact_common_neighbors_novel_only,
                history_table_aliases=history_table_aliases,
                anonymous_entity_aliases=anonymous_entity_aliases,
                natural_grouped_history=natural_grouped_history,
                natural_activity_summary=natural_activity_summary,
                natural_neighbor_names_only=natural_neighbor_names_only,
                natural_activity_compact_top3=natural_activity_compact_top3,
                natural_activity_top_k=natural_activity_top_k,
                ablate_mutual_history=ablate_mutual_history,
                ablate_common_neighbors=ablate_common_neighbors,
                ablate_source_history=ablate_source_history,
                ablate_target_history=ablate_target_history,
                ablate_source_target_history=ablate_source_target_history,
                ablate_reasoning_guidance=ablate_reasoning_guidance,
                no_cot_output_0_100=False,
                prompt_variant=prompt_variant,
                few_shot_examples=None,
                graph_prompt_special_token=graph_prompt_special_token,
                graph_prompt_num_tokens=graph_prompt_num_tokens,
            )
        )
        item = {"prompt_text": prompt_text}
        if GRAPH_PAIR_FEATURE_KEY in sample:
            item[GRAPH_PAIR_FEATURE_KEY] = sample[GRAPH_PAIR_FEATURE_KEY]
        prompt_buffer.append(item)
        label_buffer.append(int(sample["label"]))
        if len(prompt_buffer) >= int(batch_size):
            flush_prompt_buffer()

    flush_prompt_buffer()
    metrics = compute_prediction_metrics(
        predictions,
        labels,
        dtgb_eval_batch_size=dtgb_eval_batch_size,
    )
    if negative_ratio is not None:
        metrics["negative_ratio"] = int(negative_ratio)
    if model_was_training:
        model.train()
    return metrics


def compute_rrf_baseline_metrics(
    samples,
    *,
    dtgb_eval_batch_size: Optional[int] = None,
    negative_ratio: Optional[int] = None,
):
    predictions = np.array([float(sample.get("rrf_score", 0.0)) for sample in samples], dtype=np.float64)
    labels = np.array([int(sample["label"]) for sample in samples], dtype=np.int64)
    metrics = compute_prediction_metrics(
        predictions,
        labels,
        dtgb_eval_batch_size=dtgb_eval_batch_size,
    )
    if negative_ratio is not None:
        metrics["negative_ratio"] = int(negative_ratio)
    return metrics


def save_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def merge_peft_adapter(
    *,
    base_model_path: str,
    adapter_path: str,
    output_dir: str,
):
    tokenizer = _load_tokenizer(base_model_path)
    trust_remote_code = "qwen" in str(base_model_path).lower()
    model_kwargs = {"dtype": _choose_torch_dtype(base_model_path), "device_map": "cpu"}
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
    if base_model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        base_model.resize_token_embeddings(len(tokenizer))
    merged = PeftModel.from_pretrained(base_model, adapter_path).merge_and_unload()
    os.makedirs(output_dir, exist_ok=True)
    merged.save_pretrained(output_dir)
    base_config_path = os.path.join(base_model_path, "config.json")
    merged_config_path = os.path.join(output_dir, "config.json")
    if os.path.isfile(base_config_path) and os.path.isfile(merged_config_path):
        try:
            with open(base_config_path, "r", encoding="utf-8") as handle:
                base_cfg = json.load(handle)
            with open(merged_config_path, "r", encoding="utf-8") as handle:
                merged_cfg = json.load(handle)
            base_transformers_version = base_cfg.get("transformers_version")
            if isinstance(base_transformers_version, str) and base_transformers_version:
                merged_cfg["transformers_version"] = base_transformers_version
                with open(merged_config_path, "w", encoding="utf-8") as handle:
                    json.dump(merged_cfg, handle, indent=2, ensure_ascii=True)
        except Exception:
            pass
    copied = 0
    for pattern in (
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "tokenizer.model.v*",
        "special_tokens_map.json",
        "added_tokens.json",
    ):
        for source_path in glob.glob(os.path.join(base_model_path, pattern)):
            if not os.path.isfile(source_path):
                continue
            target_path = os.path.join(output_dir, os.path.basename(source_path))
            if os.path.abspath(source_path) == os.path.abspath(target_path):
                continue
            shutil.copy2(source_path, target_path)
            copied += 1
    if copied == 0:
        tokenizer.save_pretrained(output_dir)
    return output_dir


__all__ = [
    "GRAPH_PAIR_FEATURE_KEY",
    "GraphPromptMLPOnlyModel",
    "DEFAULT_LORA_TARGET_MODULES",
    "PromptResponseCollator",
    "PromptResponseDataset",
    "attach_hadamard_graph_pair_features",
    "build_prompt_response_examples",
    "create_direct_peft_samples",
    "evaluate_binary_completion_likelihood",
    "extract_binary_scores_from_training_forward",
    "get_distributed_training_context",
    "load_frozen_base_model",
    "load_peft_training_model",
    "load_prompt_tokenizer",
    "merge_peft_adapter",
    "resolve_process_cuda_device",
    "save_json",
    "summarize_trainable_parameters",
]
