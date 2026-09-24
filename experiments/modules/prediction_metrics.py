"""
Lightweight prediction metrics helpers shared by evaluation paths.
"""
import numpy as np

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:
    def roc_auc_score(y_true, y_score):
        labels = np.asarray(y_true, dtype=np.int64)
        scores = np.asarray(y_score, dtype=np.float64)
        pos_mask = labels == 1
        neg_mask = labels == 0
        n_pos = int(np.sum(pos_mask))
        n_neg = int(np.sum(neg_mask))
        if n_pos == 0 or n_neg == 0:
            raise ValueError("roc_auc_score requires both positive and negative labels.")

        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        sorted_labels = labels[order]
        ranks = np.empty(sorted_scores.shape[0], dtype=np.float64)
        idx = 0
        while idx < sorted_scores.shape[0]:
            end = idx + 1
            while end < sorted_scores.shape[0] and sorted_scores[end] == sorted_scores[idx]:
                end += 1
            avg_rank = 0.5 * ((idx + 1) + end)
            ranks[idx:end] = avg_rank
            idx = end

        rank_sum_pos = float(np.sum(ranks[sorted_labels == 1]))
        u_stat = rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)
        return float(u_stat / float(n_pos * n_neg))

    def average_precision_score(y_true, y_score):
        labels = np.asarray(y_true, dtype=np.int64)
        scores = np.asarray(y_score, dtype=np.float64)
        n_pos = int(np.sum(labels == 1))
        if n_pos == 0:
            raise ValueError("average_precision_score requires at least one positive label.")

        order = np.argsort(-scores, kind="mergesort")
        sorted_labels = labels[order]
        tp = np.cumsum(sorted_labels == 1)
        precision = tp / np.arange(1, sorted_labels.shape[0] + 1, dtype=np.float64)
        return float(np.sum(precision[sorted_labels == 1]) / float(n_pos))


def compute_dtgb_ap_auc(predictions, labels, negative_ratio):
    """
    DTGB-style AP/AUC: 1 positive vs 1 matched negative per query (1-vs-1).
    Assumes samples are ordered as [pos, neg1, neg2, ...] per query.
    Returns (ap, auc) or (None, None) if not computable.
    """
    if negative_ratio is None or negative_ratio < 1:
        return None, None
    block = 1 + negative_ratio
    if len(labels) % block != 0:
        return None, None

    num_blocks = len(labels) // block
    pos_scores = np.empty(num_blocks, dtype=np.float64)
    neg_scores = np.empty(num_blocks, dtype=np.float64)
    valid = 0

    for block_idx in range(num_blocks):
        start = block_idx * block
        end = start + block
        block_labels = labels[start:end]
        block_scores = predictions[start:end]

        pos_idx = np.where(block_labels == 1)[0]
        neg_idx = np.where(block_labels == 0)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue

        pos_scores[valid] = block_scores[pos_idx[0]]
        neg_scores[valid] = block_scores[neg_idx[0]]
        valid += 1

    if valid == 0:
        return None, None

    pos_scores = pos_scores[:valid]
    neg_scores = neg_scores[:valid]
    combined_scores = np.concatenate([pos_scores, neg_scores])
    combined_labels = np.concatenate([np.ones(valid), np.zeros(valid)])

    ap = average_precision_score(combined_labels, combined_scores)
    auc = roc_auc_score(combined_labels, combined_scores)
    return ap, auc


def compute_dtgb_ap_auc_batchwise(predictions, labels, negative_ratio, eval_batch_size):
    """
    DTGB-style AP/AUC averaged across fixed evaluation batches.

    `eval_batch_size` is the number of positive queries per evaluation batch,
    matching the DTGB loaders that iterate over positive edges in fixed-size
    batches and compute AP/AUC per batch before averaging.
    """
    if negative_ratio is None or negative_ratio < 1:
        return None, None, 0
    if eval_batch_size is None or int(eval_batch_size) < 1:
        return None, None, 0

    block = 1 + negative_ratio
    if len(labels) % block != 0:
        return None, None, 0

    num_queries = len(labels) // block
    batch_size = int(eval_batch_size)
    ap_list = []
    auc_list = []

    for query_start in range(0, num_queries, batch_size):
        query_end = min(num_queries, query_start + batch_size)
        sample_start = query_start * block
        sample_end = query_end * block
        ap, auc = compute_dtgb_ap_auc(
            predictions[sample_start:sample_end],
            labels[sample_start:sample_end],
            negative_ratio,
        )
        if ap is None or auc is None:
            continue
        ap_list.append(float(ap))
        auc_list.append(float(auc))

    if not ap_list:
        return None, None, 0

    return float(np.mean(ap_list)), float(np.mean(auc_list)), int(len(ap_list))


def compute_prediction_metrics(scores, labels, dtgb_eval_batch_size=None):
    """
    Compute DTGB AP/AUC (when valid), global AP/AUC, and threshold-0.5 accuracy.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    num_pos = int(labels.sum())
    num_neg = int((1 - labels).sum())
    neg_ratio = None
    if num_pos > 0 and num_neg % num_pos == 0:
        neg_ratio = num_neg // num_pos

    dtgb_batch_count = 0
    dtgb_metric_aggregation = "pooled"
    binary_preds = (scores > 0.5).astype(np.int64)
    accuracy = float((binary_preds == labels).mean())

    if num_pos == 0 or num_neg == 0:
        nan = float("nan")
        return {
            "auc": nan,
            "ap": nan,
            "auc_global": nan,
            "ap_global": nan,
            "accuracy": accuracy,
            "num_samples": int(len(labels)),
            "num_positive": num_pos,
            "num_negative": num_neg,
            "predictions": scores.tolist(),
            "labels": labels.tolist(),
            "dtgb_eval_batch_size": (
                int(dtgb_eval_batch_size) if dtgb_eval_batch_size is not None else None
            ),
            "dtgb_num_metric_batches": int(dtgb_batch_count),
            "dtgb_metric_aggregation": dtgb_metric_aggregation,
        }

    if dtgb_eval_batch_size is not None:
        ap, auc, dtgb_batch_count = compute_dtgb_ap_auc_batchwise(
            scores,
            labels,
            neg_ratio,
            dtgb_eval_batch_size,
        )
        if ap is not None and auc is not None:
            dtgb_metric_aggregation = "batch_mean"
    else:
        ap, auc = compute_dtgb_ap_auc(scores, labels, neg_ratio)

    ap_global = average_precision_score(labels, scores)
    auc_global = roc_auc_score(labels, scores)
    if ap is None or auc is None:
        ap, auc = ap_global, auc_global
        if dtgb_eval_batch_size is not None:
            dtgb_metric_aggregation = "global_fallback"

    return {
        "auc": float(auc),
        "ap": float(ap),
        "auc_global": float(auc_global),
        "ap_global": float(ap_global),
        "accuracy": accuracy,
        "num_samples": int(len(labels)),
        "num_positive": num_pos,
        "num_negative": num_neg,
        "predictions": scores.tolist(),
        "labels": labels.tolist(),
        "dtgb_eval_batch_size": (
            int(dtgb_eval_batch_size) if dtgb_eval_batch_size is not None else None
        ),
        "dtgb_num_metric_batches": int(dtgb_batch_count),
        "dtgb_metric_aggregation": dtgb_metric_aggregation,
    }


__all__ = [
    "average_precision_score",
    "compute_dtgb_ap_auc",
    "compute_dtgb_ap_auc_batchwise",
    "compute_prediction_metrics",
    "roc_auc_score",
]
