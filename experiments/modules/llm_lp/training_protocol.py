"""Explicit PEFT training boundaries aligned with canonical DTGB splitting.

Split quantiles and reserved nodes are resolved from the ORIGINAL full graph,
before filtering. Strict training queries, negative destinations, histories,
and graph-wide prompt statistics then use only the observed training graph.
Validation queries retain canonical boundaries and causal full-graph history.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random

import numpy as np


TRAIN_DATA_PROTOCOLS = ("dtgb_strict", "legacy_time_only")


def _node_ids_sha256(node_ids):
    return hashlib.sha256(np.asarray(node_ids, dtype="<i8").tobytes()).hexdigest()


def _graph_sha256(edges_df, mask=None):
    """Hash ordered identity columns without platform-endian/dtype ambiguity."""
    digest = hashlib.sha256(b"dtgb_graph_u_r_i_ts_v1\n")
    for name, dtype in (("u", "<i8"), ("r", "<i8"), ("i", "<i8"), ("ts", "<f8")):
        values = np.asarray(edges_df[name], dtype=dtype)
        if mask is not None:
            values = values[mask]
        digest.update(name.encode("ascii") + b"\n")
        digest.update(np.asarray([len(values)], dtype="<i8").tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class TrainingProtocol:
    name: str
    split_name: str
    split_times: np.ndarray
    val_time: float
    test_time: float
    reserved_node_ids: np.ndarray
    observed_train_node_ids: np.ndarray
    positive_mask: np.ndarray
    history_mask: np.ndarray
    negative_dst_pool: np.ndarray
    apply_gdelt_time_bucket: bool
    metadata: dict


def resolve_training_protocol(
    edges_df, *, split_name, train_data_protocol="dtgb_strict", data_seed=2020,
    val_ratio=.15, test_ratio=.15, apply_gdelt_time_bucket=False,
):
    """Return immutable-array selections; never recompute splits after filtering.

    The unsorted ``list(test_node_set)`` sampling order intentionally reproduces
    utils.DataLoader. Sorting that population would silently change the split.
    Sorted selected node IDs are recorded for reproducibility and auditing.
    """
    name, split = str(train_data_protocol).strip().lower(), str(split_name).strip().lower()
    if name not in TRAIN_DATA_PROTOCOLS:
        raise ValueError(f"Unknown train_data_protocol={name!r}")
    if split not in {"train", "pretest", "validation"}:
        raise ValueError(f"Unsupported split_name={split!r}")
    if name == "dtgb_strict" and split == "pretest":
        raise ValueError("dtgb_strict forbids pretest training; use split_name='train'")
    if not (np.isfinite(val_ratio) and np.isfinite(test_ratio)
            and 0 <= val_ratio < 1 and 0 <= test_ratio < 1 and val_ratio + test_ratio < 1):
        raise ValueError("Require finite nonnegative split ratios with sum below one")
    raw = np.asarray(edges_df["ts"], dtype=np.float64)
    src, dst = np.asarray(edges_df["u"]), np.asarray(edges_df["i"])
    if (raw.ndim != 1 or not len(raw) or src.shape != raw.shape or dst.shape != raw.shape
            or not np.isfinite(raw).all() or not np.isfinite(src).all() or not np.isfinite(dst).all()
            or not np.equal(src, np.floor(src)).all() or not np.equal(dst, np.floor(dst)).all()):
        raise ValueError("Require nonempty aligned finite timestamps and integer node IDs")
    src, dst = src.astype(np.int64), dst.astype(np.int64)
    times = raw.copy()
    if apply_gdelt_time_bucket:
        times = (np.floor(raw / 15) if name == "dtgb_strict" else
                 np.floor_divide(raw.astype(np.int64), 15).astype(np.float64))
    val_time, test_time = map(float, np.quantile(times, [1 - val_ratio - test_ratio, 1 - test_ratio]))
    reserved = np.array([], dtype=np.int64)
    observed_mask = np.ones(len(raw), dtype=bool)
    if name == "dtgb_strict":
        node_set = set(src).union(set(dst))
        test_node_set = set(src[times > val_time]).union(set(dst[times > val_time]))
        count = int(.1 * len(node_set))
        if count > len(test_node_set):
            raise ValueError("Canonical DTGB reserved-node population is too small for its prescribed sample")
        selected = random.Random(int(data_seed)).sample(list(test_node_set), count)
        reserved = np.asarray(sorted(selected), dtype=np.int64)
        observed_mask = ~(np.isin(src, reserved) | np.isin(dst, reserved))
    observed_train = (times <= val_time) & observed_mask
    observed_nodes = np.unique(np.r_[src[observed_train], dst[observed_train]])
    if name == "dtgb_strict":
        mask = observed_train if split == "train" else ((times > val_time) & (times <= test_time))
        pool_mask = observed_train if split == "train" else times <= test_time
        history_mask = observed_train if split == "train" else np.ones(len(raw), dtype=bool)
    else:
        if split == "train":
            mask, pool_mask = times < val_time, times < val_time
        elif split == "pretest":
            mask, pool_mask = times < test_time, times < test_time
        else:
            mask, pool_mask = (times >= val_time) & (times < test_time), times < test_time
        history_mask = np.ones(len(raw), dtype=bool)
    pool = np.unique(dst[pool_mask])
    metadata = {
        "name": name, "split_name": split, "data_seed": int(data_seed),
        "val_ratio": float(val_ratio), "test_ratio": float(test_ratio),
        "val_time": val_time, "test_time": test_time,
        "apply_gdelt_time_bucket": bool(apply_gdelt_time_bucket),
        "split_cutoffs_computed_from_original_full_graph": True,
        "full_graph_rows": len(raw), "positive_pool_rows": int(mask.sum()),
        "history_graph_rows": int(history_mask.sum()),
        "negative_destination_pool_size": len(pool),
        "reserved_node_count": len(reserved), "reserved_node_ids": reserved.tolist(),
        "reserved_node_ids_sha256": _node_ids_sha256(reserved),
        "observed_train_node_count": len(observed_nodes),
        "observed_train_node_ids_sha256": _node_ids_sha256(observed_nodes),
        "full_graph_identity_sha256": _graph_sha256(edges_df),
        "history_graph_identity_sha256": _graph_sha256(edges_df, history_mask),
        "observed_training_graph_identity_sha256": _graph_sha256(edges_df, observed_train),
        "negative_destination_pool_sha256": _node_ids_sha256(pool),
        "graph_identity_hash_schema": "dtgb_graph_u_r_i_ts_v1; ordered column-major little-endian int64 u/r/i and float64 ts",
        "query_boundary": ("t <= val_time" if split == "train" else "val_time < t <= test_time")
            if name == "dtgb_strict" else "legacy half-open time-only window",
        "history_policy": "observed_training_graph_only" if name == "dtgb_strict" and split == "train"
            else "full_graph_causal_per_query_history",
        "history_per_query_cutoff": "strictly earlier RAW timestamp than the query",
        "strict_train_node_isolation": name == "dtgb_strict" and split == "train",
        "global_prompt_statistics_source": "observed_training_graph_only"
            if name == "dtgb_strict" and split == "train" else "legacy/full_graph",
        "legacy_protocol_preserved_explicitly": name == "legacy_time_only",
    }
    for array in (times, reserved, observed_nodes, mask, history_mask, pool):
        array.setflags(write=False)
    return TrainingProtocol(name, split, times, val_time, test_time, reserved, observed_nodes,
                            mask, history_mask, pool, bool(apply_gdelt_time_bucket), metadata)


def protocol_history_edges(edges_df, protocol):
    """Return the exact history/statistics graph specified by a resolved protocol."""
    if len(edges_df) != len(protocol.history_mask):
        raise ValueError("History source differs in length from the resolved full graph")
    if _graph_sha256(edges_df) != protocol.metadata["full_graph_identity_sha256"]:
        raise ValueError("History source identities differ from the resolved full graph")
    result = edges_df.loc[protocol.history_mask].copy()
    if protocol.name == "dtgb_strict" and protocol.split_name == "train":
        endpoints = np.r_[result["u"].to_numpy(), result["i"].to_numpy()]
        if np.isin(endpoints, protocol.reserved_node_ids).any():
            raise ValueError("Reserved-node exposure in strict training history graph")
        if np.any(protocol.split_times[protocol.history_mask] > protocol.val_time):
            raise ValueError("Validation/future row in strict training history graph")
    return result


def validate_protocol_samples(samples, protocol):
    """Fail closed on strict query endpoints/time and materialized history evidence.

    Handles the current raw event, grouped entity-history and common-neighbor
    schemas. Deferred samples are checked now for queries, and callers must run
    this again after deferred prompt materialization. Free-form entity text or
    third-party summaries are not proven free of arbitrary textual references.
    """
    strict = protocol.name == "dtgb_strict"
    train = strict and protocol.split_name == "train"
    reserved = set(map(int, protocol.reserved_node_ids))
    observed = set(map(int, protocol.observed_train_node_ids))
    pool = set(map(int, protocol.negative_dst_pool))
    event_count, mentions, materialized = 0, 0, 0
    for sample in samples:
        ts = float(sample["timestamp"])
        tick = np.floor(ts / 15) if protocol.apply_gdelt_time_bucket else ts
        src, dst = int(sample["source_id"]), int(sample["target_id"])
        label = sample["label"]
        if not np.isfinite(ts) or label not in (0, 1):
            raise ValueError("Invalid supervised sample timestamp or label")
        if strict:
            valid_time = tick <= protocol.val_time if train else protocol.val_time < tick <= protocol.test_time
            if not valid_time:
                raise ValueError("Supervised sample violates the canonical strict split boundary")
        if train and (src in reserved or dst in reserved):
            raise ValueError("Reserved-node exposure in strict training query endpoint")
        if train and (src not in observed or dst not in observed):
            raise ValueError("Strict training query endpoint is outside the observed-training nodes")
        if train and label == 0 and dst not in pool:
            raise ValueError("Strict training negative destination is outside the observed-training pool")

        def check_event(event):
            nonlocal event_count
            if len(event) != 4:
                raise ValueError("Unrecognized raw history event schema")
            u, _, v, history_ts = event
            history_ts = float(history_ts)
            if strict and (not np.isfinite(history_ts) or history_ts >= ts):
                raise ValueError("History evidence must be strictly earlier than its query")
            if train and (int(u) in reserved or int(v) in reserved):
                raise ValueError("Reserved-node exposure in strict training history event")
            if train and (int(u) not in observed or int(v) not in observed):
                raise ValueError("Strict training history event is outside the observed-training nodes")
            event_count += 1

        if strict:
            for field in ("source_history", "target_history", "mutual_history"):
                for event in sample.get(field, []):
                    check_event(event)
            for field in ("source_history_entities", "target_history_entities"):
                for node, timestamps in sample.get(field, []):
                    if train and int(node) in reserved:
                        raise ValueError("Reserved-node exposure in grouped training history")
                    if train and int(node) not in observed:
                        raise ValueError("Grouped training history is outside the observed-training nodes")
                    if any(not np.isfinite(t) or float(t) >= ts for t in timestamps):
                        raise ValueError("Grouped history evidence must precede its query")
                    mentions += 1
            for node, source_event, target_event in sample.get("common_neighbors", []):
                if train and int(node) in reserved:
                    raise ValueError("Reserved-node exposure in training common neighbors")
                if train and int(node) not in observed:
                    raise ValueError("Training common neighbor is outside the observed-training nodes")
                check_event(source_event)
                check_event(target_event)
            materialized += int(bool(sample.get("prompt_context_materialized", False)))
    return {
        "validated_samples": len(samples), "materialized_samples_checked": materialized,
        "history_events_checked": event_count, "grouped_history_entities_checked": mentions,
        "strict_checks_applied": strict, "reserved_training_exposure_found": False if train else None,
        "query_time_violations_found": False if strict else None,
        "requires_revalidation_after_deferred_materialization": strict and materialized < len(samples),
        "free_form_entity_text_and_external_summary_contents_audited": False,
    }


__all__ = ["TRAIN_DATA_PROTOCOLS", "TrainingProtocol", "resolve_training_protocol",
           "protocol_history_edges", "validate_protocol_samples"]
