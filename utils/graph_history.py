"""Select graph history without changing the canonical data split."""
from types import SimpleNamespace

import numpy as np


TRAIN_HISTORY_POLICY = "observed_train_only_v1"


def require_training_history_table(table):
    """Check graph provenance before fitting on saved context features."""
    value = table.get("train_history_policy")
    if value is None or str(np.asarray(value).item()) != TRAIN_HISTORY_POLICY:
        raise ValueError("Train context must be generated with observed_train_only_v1 history")


def graph_history(data, cutoff_time=None):
    """Copy a resolved split, with an optional additional strict time cutoff."""
    times = np.asarray(data.node_interact_times, dtype=np.float64)
    mask = np.ones(len(times), dtype=bool)
    if cutoff_time is not None:
        mask &= times < float(cutoff_time)
    return SimpleNamespace(
        src_node_ids=np.asarray(data.src_node_ids, dtype=np.int64)[mask],
        dst_node_ids=np.asarray(data.dst_node_ids, dtype=np.int64)[mask],
        node_interact_times=times[mask],
        edge_ids=np.asarray(data.edge_ids, dtype=np.int64)[mask],
    )
