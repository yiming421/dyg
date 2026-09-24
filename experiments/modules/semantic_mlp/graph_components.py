from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torch_sparse import SparseTensor
    from torch_sparse.matmul import spmm_add
except ImportError:  # pragma: no cover
    SparseTensor = None
    spmm_add = None

from experiments.modules.heuristic_models import smooth_embeddings_by_time_window_torch


SEMANTIC_SOURCE_INIT_VERSION = 2
SEMANTIC_SOURCE_INIT_CHOICES = {"raw", "history_mean"}


def normalize_semantic_source_init(value: str, *, field_name: str = "source_init") -> str:
    normalized = str(value).strip().lower()
    if normalized not in SEMANTIC_SOURCE_INIT_CHOICES:
        raise ValueError(
            f"{field_name} must be one of {sorted(SEMANTIC_SOURCE_INIT_CHOICES)}, "
            f"got {value!r}."
        )
    return normalized


def resolve_checkpoint_semantic_source_init(
    *,
    smoothing_config: Dict,
    gcn_config: Dict,
    rolling_smoothing: bool,
    source_init_override: str = "auto",
) -> Tuple[str, str]:
    """Resolve checkpoint source-init semantics, including the legacy runtime bug.

    Checkpoints written before ``SEMANTIC_SOURCE_INIT_VERSION`` recorded the
    requested CLI option even when the runtime actually selected raw embeddings.
    New checkpoints record the effective behavior explicitly.
    """
    override = str(source_init_override).strip().lower()
    if override not in {"auto", *SEMANTIC_SOURCE_INIT_CHOICES}:
        raise ValueError(
            "source_init_override must be one of auto/raw/history_mean, "
            f"got {source_init_override!r}."
        )
    if override != "auto":
        return override, "explicit_override"

    effective = str(smoothing_config.get("source_init_effective", "")).strip().lower()
    if effective in SEMANTIC_SOURCE_INIT_CHOICES:
        return effective, "checkpoint_effective_metadata"

    try:
        version = int(smoothing_config.get("source_init_version", 0))
    except (TypeError, ValueError):
        version = 0
    recorded = str(smoothing_config.get("source_init", "")).strip().lower()
    if version >= SEMANTIC_SOURCE_INIT_VERSION and recorded in SEMANTIC_SOURCE_INIT_CHOICES:
        return recorded, "checkpoint_versioned_metadata"

    if bool(rolling_smoothing):
        # Every legacy rolling path replaced the provider's initialized table
        # with active raw embeddings before each batch.
        return "raw", "legacy_rolling_compatibility"

    if bool(gcn_config.get("use_learnable_gcn", False)):
        # Static learnable GCN/GIN/attention-pool paths likewise selected the
        # active raw table instead of the initialized smoothing table.
        return "raw", "legacy_learnable_mp_compatibility"

    if recorded in SEMANTIC_SOURCE_INIT_CHOICES:
        return recorded, "legacy_non_mp_metadata"
    return "raw", "missing_metadata_default"

def _elem2spm(element: Tensor, sizes: Tuple[int, int]) -> SparseTensor:
    col = torch.bitwise_and(element, 0xFFFFFFFF)
    row = torch.bitwise_right_shift(element, 32)
    return SparseTensor(row=row, col=col, sparse_sizes=sizes).to_device(element.device).fill_value_(1.0)


def _spm2elem(spm: SparseTensor) -> Tensor:
    return torch.bitwise_left_shift(spm.storage.row(), 32).add_(spm.storage.col())


def _spmoverlap(adj1: SparseTensor, adj2: SparseTensor) -> SparseTensor:
    if adj1.sizes() != adj2.sizes():
        raise ValueError(f"Sparse overlap shape mismatch: {adj1.sizes()} vs {adj2.sizes()}.")
    element1 = _spm2elem(adj1)
    element2 = _spm2elem(adj2)
    if element1.numel() == 0 or element2.numel() == 0:
        return _elem2spm(element1[:0], adj1.sizes())
    if element2.shape[0] > element1.shape[0]:
        element1, element2 = element2, element1
    idx = torch.searchsorted(element1[:-1], element2)
    mask = element1[idx] == element2
    return _elem2spm(element2[mask], adj1.sizes())


def _spmdiff(adj1: SparseTensor, adj2: SparseTensor) -> SparseTensor:
    element1 = _spm2elem(adj1)
    element2 = _spm2elem(adj2)

    if element1.numel() == 0:
        return _elem2spm(element1, adj1.sizes())

    idx = torch.searchsorted(element1[:-1], element2)
    matched_mask = element1[idx] == element2

    keep_mask = torch.ones_like(element1, dtype=torch.bool)
    keep_mask[idx[matched_mask]] = False
    return _elem2spm(element1[keep_mask], adj1.sizes())


def _build_neighbor_sparse(
    neigh_idx: torch.Tensor,
    neigh_valid: torch.Tensor,
    num_cols: int,
) -> SparseTensor:
    if SparseTensor is None:
        raise RuntimeError("torch_sparse is required for the NCN sparse-overlap scorer.")
    if neigh_idx.ndim != 2 or neigh_valid.ndim != 2:
        raise ValueError("Expected 2D neighbor index/mask tensors.")
    if neigh_idx.shape != neigh_valid.shape:
        raise ValueError(
            f"Neighbor index/mask shape mismatch: {tuple(neigh_idx.shape)} vs {tuple(neigh_valid.shape)}."
        )
    row_ids = torch.arange(neigh_idx.size(0), device=neigh_idx.device, dtype=torch.long).unsqueeze(1)
    row_ids = row_ids.expand_as(neigh_idx)
    flat_mask = neigh_valid.reshape(-1)
    if not flat_mask.any():
        return SparseTensor(
            row=torch.empty((0,), dtype=torch.long, device=neigh_idx.device),
            col=torch.empty((0,), dtype=torch.long, device=neigh_idx.device),
            sparse_sizes=(int(neigh_idx.size(0)), int(num_cols)),
        ).to_device(neigh_idx.device).fill_value_(1.0)
    row = row_ids.reshape(-1)[flat_mask]
    col = neigh_idx.reshape(-1)[flat_mask]
    return SparseTensor(row=row, col=col, sparse_sizes=(int(neigh_idx.size(0)), int(num_cols))).to_device(
        neigh_idx.device
    ).coalesce().fill_value_(1.0)


def build_binary_history_adj(
    *,
    src_node_ids: np.ndarray,
    dst_node_ids: np.ndarray,
    lookup: torch.Tensor,
    num_rows: int,
    undirected: bool = True,
) -> Optional[SparseTensor]:
    if SparseTensor is None:
        raise RuntimeError("torch_sparse is required for the NCN sparse-overlap scorer.")

    src_np = np.asarray(src_node_ids, dtype=np.int64).reshape(-1)
    dst_np = np.asarray(dst_node_ids, dtype=np.int64).reshape(-1)
    if src_np.size == 0 or dst_np.size == 0:
        return SparseTensor(
            row=torch.empty((0,), dtype=torch.long, device=lookup.device),
            col=torch.empty((0,), dtype=torch.long, device=lookup.device),
            sparse_sizes=(int(num_rows), int(num_rows)),
        ).to_device(lookup.device).fill_value_(1.0)

    device = lookup.device
    src = torch.from_numpy(src_np).long().to(device)
    dst = torch.from_numpy(dst_np).long().to(device)
    max_id = lookup.size(0)
    in_range = (src >= 0) & (src < max_id) & (dst >= 0) & (dst < max_id)
    if not in_range.any():
        return SparseTensor(
            row=torch.empty((0,), dtype=torch.long, device=device),
            col=torch.empty((0,), dtype=torch.long, device=device),
            sparse_sizes=(int(num_rows), int(num_rows)),
        ).to_device(device).fill_value_(1.0)

    src = src[in_range]
    dst = dst[in_range]
    src_idx = lookup[src]
    dst_idx = lookup[dst]
    valid = (src_idx >= 0) & (dst_idx >= 0)
    if not valid.any():
        return SparseTensor(
            row=torch.empty((0,), dtype=torch.long, device=device),
            col=torch.empty((0,), dtype=torch.long, device=device),
            sparse_sizes=(int(num_rows), int(num_rows)),
        ).to_device(device).fill_value_(1.0)

    row = src_idx[valid]
    col = dst_idx[valid]
    if undirected:
        row_orig = row
        col_orig = col
        row = torch.cat([row_orig, col_orig], dim=0)
        col = torch.cat([col_orig, row_orig], dim=0)
    return SparseTensor(row=row, col=col, sparse_sizes=(int(num_rows), int(num_rows))).to_device(device).coalesce().fill_value_(1.0)


def build_two_hop_binary_adj(adj: Optional[SparseTensor]) -> Optional[SparseTensor]:
    if adj is None:
        return None
    if SparseTensor is None:
        raise RuntimeError("torch_sparse is required for exact MPLP-style structural features.")
    adj_device = adj.storage.row().device
    if adj_device.type == 'cuda':
        try:
            from spspmm import spspmm as cuda_spspmm

            with torch.cuda.device(adj_device):
                adj_coo = adj.to_torch_sparse_coo_tensor().coalesce()
                adj_two_walks_coo = cuda_spspmm(adj_coo, adj_coo, 'alg3').coalesce()
            edge_index = adj_two_walks_coo.indices()
            adj_two_walks = SparseTensor(
                row=edge_index[0],
                col=edge_index[1],
                sparse_sizes=adj.sizes(),
                is_sorted=True,
                trust_data=True,
            ).to_device(adj_device).coalesce().fill_value_(1.0)
        except ImportError:
            adj_two_walks = adj @ adj
    else:
        adj_two_walks = adj @ adj
    adj_with_self = adj.fill_diag(1)
    return _spmdiff(adj_two_walks, adj_with_self)


def _get_cpu_csr_from_sparse_adj(adj: SparseTensor) -> Tuple[np.ndarray, np.ndarray]:
    cache = getattr(adj, "_dtgb_cpu_csr_cache", None)
    if cache is not None:
        return cache

    row = adj.storage.row().detach().cpu().numpy().astype(np.int64, copy=False)
    col = adj.storage.col().detach().cpu().numpy().astype(np.int64, copy=False)
    num_rows = int(adj.sizes()[0])
    rowptr = np.zeros(num_rows + 1, dtype=np.int64)
    if row.size > 0:
        np.add.at(rowptr, row + 1, 1)
    np.cumsum(rowptr, out=rowptr)
    cache = (rowptr, col)
    try:
        setattr(adj, "_dtgb_cpu_csr_cache", cache)
    except Exception:
        pass
    return cache


def _compute_mplp_exact_features_local_cpu(
    *,
    lookup: torch.Tensor,
    src_ids: np.ndarray,
    dst_ids: np.ndarray,
    ncn_adj: SparseTensor,
    output_device: torch.device,
    output_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    src = torch.from_numpy(np.asarray(src_ids, dtype=np.int64)).long().to(output_device)
    dst = torch.from_numpy(np.asarray(dst_ids, dtype=np.int64)).long().to(output_device)

    max_id = lookup.size(0)
    in_range = (src >= 0) & (src < max_id) & (dst >= 0) & (dst < max_id)
    src_idx = torch.full_like(src, -1)
    dst_idx = torch.full_like(dst, -1)
    src_idx[in_range] = lookup[src[in_range]]
    dst_idx[in_range] = lookup[dst[in_range]]
    valid_mask = in_range & (src_idx >= 0) & (dst_idx >= 0)

    features = torch.zeros(
        (len(src_ids), MPLPExactFusionHead.feature_dim),
        device=output_device,
        dtype=output_dtype,
    )
    if not valid_mask.any():
        return features, valid_mask

    rowptr, col = _get_cpu_csr_from_sparse_adj(ncn_adj)
    valid_cpu = valid_mask.detach().cpu().numpy()
    src_idx_cpu = src_idx.detach().cpu().numpy().astype(np.int64, copy=False)
    dst_idx_cpu = dst_idx.detach().cpu().numpy().astype(np.int64, copy=False)
    valid_src = src_idx_cpu[valid_cpu]
    valid_dst = dst_idx_cpu[valid_cpu]
    needed_nodes = np.unique(np.concatenate([valid_src, valid_dst], axis=0))

    one_hop_cache: Dict[int, np.ndarray] = {}
    two_hop_cache: Dict[int, np.ndarray] = {}

    def one_hop(node: int) -> np.ndarray:
        cached = one_hop_cache.get(node)
        if cached is not None:
            return cached
        start = int(rowptr[node])
        end = int(rowptr[node + 1])
        neigh = col[start:end]
        one_hop_cache[node] = neigh
        return neigh

    def two_hop(node: int) -> np.ndarray:
        cached = two_hop_cache.get(node)
        if cached is not None:
            return cached
        neigh1 = one_hop(node)
        if neigh1.size == 0:
            neigh2 = np.empty((0,), dtype=np.int64)
        else:
            expanded = [one_hop(int(mid)) for mid in neigh1]
            expanded = [arr for arr in expanded if arr.size > 0]
            if expanded:
                neigh2 = np.unique(np.concatenate(expanded, axis=0))
                excluded = np.concatenate([neigh1, np.asarray([node], dtype=np.int64)], axis=0)
                neigh2 = np.setdiff1d(neigh2, excluded, assume_unique=True)
            else:
                neigh2 = np.empty((0,), dtype=np.int64)
        two_hop_cache[node] = neigh2
        return neigh2

    for node in needed_nodes:
        node_int = int(node)
        one_hop(node_int)
        two_hop(node_int)

    valid_features = np.zeros((valid_src.shape[0], MPLPExactFusionHead.feature_dim), dtype=np.float32)
    for row_idx, (src_node, dst_node) in enumerate(zip(valid_src, valid_dst)):
        src_node = int(src_node)
        dst_node = int(dst_node)
        src_one = one_hop(src_node)
        dst_one = one_hop(dst_node)
        src_two = two_hop(src_node)
        dst_two = two_hop(dst_node)

        count_1_1 = float(np.intersect1d(src_one, dst_one, assume_unique=True).size)
        count_1_2 = float(
            np.intersect1d(src_one, dst_two, assume_unique=True).size
            + np.intersect1d(src_two, dst_one, assume_unique=True).size
        )
        count_2_2 = float(np.intersect1d(src_two, dst_two, assume_unique=True).size)
        degree_one_src = float(src_one.size)
        degree_one_dst = float(dst_one.size)
        degree_two_src = float(src_two.size)
        degree_two_dst = float(dst_two.size)
        count_1_inf = max(degree_one_src + degree_one_dst - 2.0 * count_1_1 - count_1_2, 0.0)
        count_2_inf = max(degree_two_src + degree_two_dst - 2.0 * count_2_2 - count_1_2, 0.0)
        valid_features[row_idx] = np.log1p(
            np.asarray([count_1_1, count_1_2, count_2_2, count_1_inf, count_2_inf], dtype=np.float32)
        )

    valid_tensor = torch.from_numpy(valid_features).to(device=output_device, dtype=output_dtype)
    features[valid_mask] = valid_tensor
    return features, valid_mask

class MPLPExactFusionHead(nn.Module):
    """
    Learnable additive fusion for exact MPLP-style high-order structure features.
    """

    feature_dim = 5

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(self.feature_dim, 1, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, semantic_logits: torch.Tensor, structural_features: torch.Tensor) -> torch.Tensor:
        delta = self.linear(structural_features).squeeze(-1)
        return semantic_logits + delta


def compute_mplp_exact_features(
    *,
    lookup: torch.Tensor,
    src_ids: np.ndarray,
    dst_ids: np.ndarray,
    ncn_adj: Optional[SparseTensor],
    two_hop_adj: Optional[SparseTensor],
    output_device: torch.device,
    output_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if ncn_adj is None:
        return (
            torch.zeros((len(src_ids), MPLPExactFusionHead.feature_dim), device=output_device, dtype=output_dtype),
            torch.zeros((len(src_ids),), device=output_device, dtype=torch.bool),
        )
    if SparseTensor is None:
        raise RuntimeError("torch_sparse is required for exact MPLP-style structural features.")
    if two_hop_adj is None:
        return _compute_mplp_exact_features_local_cpu(
            lookup=lookup,
            src_ids=src_ids,
            dst_ids=dst_ids,
            ncn_adj=ncn_adj,
            output_device=output_device,
            output_dtype=output_dtype,
        )

    src = torch.from_numpy(np.asarray(src_ids, dtype=np.int64)).long().to(output_device)
    dst = torch.from_numpy(np.asarray(dst_ids, dtype=np.int64)).long().to(output_device)

    max_id = lookup.size(0)
    in_range = (src >= 0) & (src < max_id) & (dst >= 0) & (dst < max_id)
    src_idx = torch.full_like(src, -1)
    dst_idx = torch.full_like(dst, -1)
    src_idx[in_range] = lookup[src[in_range]]
    dst_idx[in_range] = lookup[dst[in_range]]
    valid_mask = in_range & (src_idx >= 0) & (dst_idx >= 0)

    features = torch.zeros(
        (len(src_ids), MPLPExactFusionHead.feature_dim),
        device=output_device,
        dtype=output_dtype,
    )
    if not valid_mask.any():
        return features, valid_mask

    valid_src_idx = src_idx[valid_mask]
    valid_dst_idx = dst_idx[valid_mask]
    adj_device = ncn_adj.storage.row().device
    valid_src_idx = valid_src_idx.to(adj_device)
    valid_dst_idx = valid_dst_idx.to(adj_device)

    src_one = ncn_adj[valid_src_idx]
    dst_one = ncn_adj[valid_dst_idx]
    src_two = two_hop_adj[valid_src_idx]
    dst_two = two_hop_adj[valid_dst_idx]

    count_1_1 = _spmoverlap(src_one, dst_one).sum(dim=-1).to_dense().to(dtype=output_dtype, device=output_device)
    count_1_2 = (
        _spmoverlap(src_one, dst_two).sum(dim=-1).to_dense().to(dtype=output_dtype, device=output_device)
        + _spmoverlap(src_two, dst_one).sum(dim=-1).to_dense().to(dtype=output_dtype, device=output_device)
    )
    count_2_2 = _spmoverlap(src_two, dst_two).sum(dim=-1).to_dense().to(dtype=output_dtype, device=output_device)

    degree_one_src = src_one.sum(dim=-1).to_dense().to(dtype=output_dtype, device=output_device)
    degree_one_dst = dst_one.sum(dim=-1).to_dense().to(dtype=output_dtype, device=output_device)
    degree_two_src = src_two.sum(dim=-1).to_dense().to(dtype=output_dtype, device=output_device)
    degree_two_dst = dst_two.sum(dim=-1).to_dense().to(dtype=output_dtype, device=output_device)

    count_1_inf = torch.clamp(degree_one_src + degree_one_dst - 2.0 * count_1_1 - count_1_2, min=0.0)
    count_2_inf = torch.clamp(degree_two_src + degree_two_dst - 2.0 * count_2_2 - count_1_2, min=0.0)

    valid_features = torch.stack(
        [
            count_1_1,
            count_1_2,
            count_2_2,
            count_1_inf,
            count_2_inf,
        ],
        dim=1,
    )
    features[valid_mask] = torch.log1p(valid_features)
    return features, valid_mask

def fuse_pos_neg_logits_with_mplp_exact(
    mplp_exact_fusion: Optional[MPLPExactFusionHead],
    lookup: torch.Tensor,
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    pos_src: np.ndarray,
    pos_dst: np.ndarray,
    neg_src: np.ndarray,
    neg_dst: np.ndarray,
    ncn_adj: Optional[SparseTensor],
    two_hop_adj: Optional[SparseTensor],
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    if mplp_exact_fusion is None:
        return pos_logits, neg_logits, 0.0

    t0 = time.perf_counter()
    if ncn_adj is not None and two_hop_adj is None:
        combined_src = np.concatenate(
            [
                np.asarray(pos_src, dtype=np.int64),
                np.asarray(neg_src, dtype=np.int64),
            ],
            axis=0,
        )
        combined_dst = np.concatenate(
            [
                np.asarray(pos_dst, dtype=np.int64),
                np.asarray(neg_dst, dtype=np.int64),
            ],
            axis=0,
        )
        combined_features, combined_valid = compute_mplp_exact_features(
            lookup=lookup,
            src_ids=combined_src,
            dst_ids=combined_dst,
            ncn_adj=ncn_adj,
            two_hop_adj=None,
            output_device=pos_logits.device,
            output_dtype=pos_logits.dtype,
        )
        pos_count = len(pos_src)
        pos_features = combined_features[:pos_count]
        neg_features = combined_features[pos_count:].to(dtype=neg_logits.dtype, device=neg_logits.device)
        pos_valid = combined_valid[:pos_count]
        neg_valid = combined_valid[pos_count:].to(device=neg_logits.device)
    else:
        pos_features, pos_valid = compute_mplp_exact_features(
            lookup=lookup,
            src_ids=pos_src,
            dst_ids=pos_dst,
            ncn_adj=ncn_adj,
            two_hop_adj=two_hop_adj,
            output_device=pos_logits.device,
            output_dtype=pos_logits.dtype,
        )
        neg_features, neg_valid = compute_mplp_exact_features(
            lookup=lookup,
            src_ids=neg_src,
            dst_ids=neg_dst,
            ncn_adj=ncn_adj,
            two_hop_adj=two_hop_adj,
            output_device=neg_logits.device,
            output_dtype=neg_logits.dtype,
        )

    if pos_valid.any():
        pos_logits = pos_logits.clone()
        pos_logits[pos_valid] = mplp_exact_fusion(pos_logits[pos_valid], pos_features[pos_valid])
    if neg_valid.any():
        neg_logits = neg_logits.clone()
        neg_logits[neg_valid] = mplp_exact_fusion(neg_logits[neg_valid], neg_features[neg_valid])
    return pos_logits, neg_logits, time.perf_counter() - t0

class SimpleGCNConv(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_linear_transform: bool = True,
    ):
        super().__init__()
        self.use_linear_transform = bool(use_linear_transform)
        if self.use_linear_transform:
            self.linear = nn.Linear(input_dim, output_dim, bias=False)
        elif int(input_dim) != int(output_dim):
            raise ValueError(
                "A transformation-free GCN layer requires input_dim == output_dim, "
                f"got {input_dim} != {output_dim}."
            )
        else:
            self.linear = nn.Identity()

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(norm_adj, self.linear(x))


class LearnableGCNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        activation: str = 'relu',
        use_layernorm: bool = True,
        residual: bool = True,
        use_linear_transform: bool = True,
        relation_features: Optional[torch.Tensor] = None,
        temporal_relational_rank: int = 32,
        temporal_relational_time_basis_dim: int = 16,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.residual = bool(residual)
        self.use_linear_transform = bool(use_linear_transform)
        self.dropout = nn.Dropout(float(dropout))
        self.use_temporal_relational = relation_features is not None

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'silu':
            self.act = nn.SiLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported GCN activation: {activation}")

        if self.use_linear_transform:
            dims = [int(input_dim)] + [int(hidden_dim)] * (num_layers - 1) + [int(input_dim)]
        else:
            # Pure GCN propagation keeps the feature width fixed at every layer:
            # H^(l+1) = A_norm H^l. LayerNorm/residual behavior is unchanged.
            dims = [int(input_dim)] * (num_layers + 1)
        self.layers = nn.ModuleList([
            SimpleGCNConv(
                dims[i],
                dims[i + 1],
                use_linear_transform=self.use_linear_transform,
            )
            for i in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(dims[i + 1]) if use_layernorm else nn.Identity()
            for i in range(num_layers)
        ])

        if self.use_temporal_relational:
            relation_features = torch.as_tensor(relation_features, dtype=torch.float32)
            if relation_features.ndim != 2 or relation_features.size(0) < 1:
                raise ValueError(
                    "relation_features must be a non-empty [num_relations, feature_dim] tensor."
                )
            rank = int(temporal_relational_rank)
            time_basis_dim = int(temporal_relational_time_basis_dim)
            if rank < 1 or time_basis_dim < 2:
                raise ValueError(
                    "temporal_relational_rank must be >= 1 and "
                    "temporal_relational_time_basis_dim must be >= 2."
                )
            self.temporal_relational_rank = rank
            self.temporal_relational_time_basis_dim = time_basis_dim
            self.register_buffer("relation_features", relation_features)
            self.relation_projector = nn.Linear(
                int(relation_features.size(1)), rank, bias=False
            )
            self.temporal_relational_fusion = nn.Linear(
                2 * rank + 2 * time_basis_dim + 2,
                int(input_dim),
                bias=False,
            )
            # Start as a small residual correction to the established GCN path.
            self.temporal_relational_scale = nn.Parameter(torch.tensor(0.1))

    def _temporal_relational_residual(
        self,
        context: Dict[str, torch.Tensor],
        *,
        output_dtype: torch.dtype,
        query_rows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        required = {
            "out_relation_profile",
            "in_relation_profile",
            "out_time_profile",
            "in_time_profile",
            "direction_mask",
        }
        missing = required - set(context)
        if missing:
            raise ValueError(
                "Temporal-relational GCN context is missing: " + ", ".join(sorted(missing))
            )

        relation_codes = self.relation_projector(self.relation_features)
        num_rows = int(context["out_time_profile"].size(0))
        if query_rows is None:
            selected_rows = torch.arange(
                num_rows, dtype=torch.long, device=relation_codes.device
            )
        else:
            selected_rows = torch.unique(query_rows.to(device=relation_codes.device))
            selected_rows = selected_rows[
                (selected_rows >= 0) & (selected_rows < num_rows)
            ]
        out_relation = context["out_relation_profile"].index_select(
            0, selected_rows
        ) @ relation_codes
        in_relation = context["in_relation_profile"].index_select(
            0, selected_rows
        ) @ relation_codes
        out_time = context["out_time_profile"].index_select(0, selected_rows)
        in_time = context["in_time_profile"].index_select(0, selected_rows)
        direction_mask = context["direction_mask"].index_select(
            0, selected_rows
        ).to(dtype=relation_codes.dtype)
        fused = self.temporal_relational_fusion(
            torch.cat(
                [
                    out_relation,
                    in_relation,
                    out_time,
                    in_time,
                    direction_mask,
                ],
                dim=1,
            )
        )
        residual = (torch.tanh(self.temporal_relational_scale) * fused).to(output_dtype)
        return selected_rows, residual

    def forward(
        self,
        x: torch.Tensor,
        norm_adj: torch.Tensor,
        temporal_relational_context: Optional[Dict[str, torch.Tensor]] = None,
        temporal_relational_query_rows: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Match smoothing operator shape exactly, including optional virtual supernode.
        if norm_adj.size(0) == x.size(0):
            h = x
            drop_supernode = False
        elif norm_adj.size(0) == x.size(0) + 1:
            super_emb = x.mean(dim=0, keepdim=True)
            h = torch.cat([x, super_emb], dim=0)
            drop_supernode = True
        else:
            raise ValueError(
                f"GCN adj shape mismatch: adj={tuple(norm_adj.shape)}, x={tuple(x.shape)}."
            )

        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            h_next = layer(h, norm_adj)
            h_next = norm(h_next)
            if i < len(self.layers) - 1:
                h_next = self.act(h_next)
                h_next = self.dropout(h_next)
            if self.residual and h_next.shape == h.shape:
                h = h_next + h
            else:
                h = h_next
        if self.use_temporal_relational:
            if temporal_relational_context is None:
                raise ValueError(
                    "Temporal-relational GCN is enabled but no rolling context was supplied."
                )
            node_h = h[:x.size(0)] if drop_supernode else h
            selected_rows, residual = self._temporal_relational_residual(
                temporal_relational_context,
                output_dtype=node_h.dtype,
                query_rows=temporal_relational_query_rows,
            )
            node_h = node_h.index_add(0, selected_rows, residual)
            if drop_supernode:
                return node_h
            h = node_h
        if drop_supernode:
            return h[:x.size(0)]
        return h


class GINLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation: str = 'relu',
        use_layernorm: bool = True,
    ):
        super().__init__()
        if activation == 'relu':
            act = nn.ReLU()
        elif activation == 'silu':
            act = nn.SiLU()
        elif activation == 'gelu':
            act = nn.GELU()
        else:
            raise ValueError(f"Unsupported GIN activation: {activation}")

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act,
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, input_dim),
        )
        self.norm = nn.LayerNorm(input_dim) if use_layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.mlp(x))


class NonParametricGINLayer(nn.Module):
    """Self-inclusive sum pooling followed by a parameter-free normalization."""

    def __init__(self, input_dim: int, norm_type: str = 'bn'):
        super().__init__()
        self.norm_type = str(norm_type).lower()
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm1d(
                int(input_dim),
                affine=False,
                track_running_stats=False,
            )
        elif self.norm_type == 'ln':
            self.norm = nn.LayerNorm(
                int(input_dim),
                elementwise_affine=False,
            )
        else:
            raise ValueError(
                f"Unsupported nonparametric GIN norm: {norm_type}. "
                "Expected one of: ln, bn."
            )

    def forward(self, self_and_neighbor_sum: torch.Tensor) -> torch.Tensor:
        return self.norm(self_and_neighbor_sum)


class LearnableGINEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        activation: str = 'relu',
        use_layernorm: bool = True,
        residual: bool = True,
        nonparametric: bool = False,
        nonparametric_norm: str = 'bn',
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.nonparametric = bool(nonparametric)
        self.nonparametric_norm = str(nonparametric_norm).lower()
        if self.nonparametric:
            # Fixed-coefficient self loop plus neighbor sum: norm(x + A @ x).
            # There is no learned epsilon, feature transformation, affine norm,
            # or outer residual, so this branch remains parameter-free.
            self.residual = False
            self.layers = nn.ModuleList([
                NonParametricGINLayer(
                    input_dim=int(input_dim),
                    norm_type=self.nonparametric_norm,
                )
                for _ in range(num_layers)
            ])
            self.register_parameter('eps', None)
        else:
            self.residual = bool(residual)
            self.layers = nn.ModuleList([
                GINLayer(
                    input_dim=int(input_dim),
                    hidden_dim=int(hidden_dim),
                    dropout=float(dropout),
                    activation=activation,
                    use_layernorm=use_layernorm,
                )
                for _ in range(num_layers)
            ])
            self.eps = nn.Parameter(torch.zeros(num_layers, dtype=torch.float32))

    def forward(self, x: torch.Tensor, sum_adj: torch.Tensor) -> torch.Tensor:
        if sum_adj.size(0) == x.size(0):
            h = x
            drop_supernode = False
        elif sum_adj.size(0) == x.size(0) + 1:
            super_emb = x.mean(dim=0, keepdim=True)
            h = torch.cat([x, super_emb], dim=0)
            drop_supernode = True
        else:
            raise ValueError(
                f"GIN adj shape mismatch: adj={tuple(sum_adj.shape)}, x={tuple(x.shape)}."
            )

        for i, layer in enumerate(self.layers):
            neigh_sum = torch.sparse.mm(sum_adj, h)
            if self.nonparametric:
                h_next = layer(h + neigh_sum)
            else:
                h_next = layer((1.0 + self.eps[i]) * h + neigh_sum)
            if self.residual and h_next.shape == h.shape:
                h = h + h_next
            else:
                h = h_next
        if drop_supernode:
            return h[:x.size(0)]
        return h


def _relative_time_basis(
    delta_times: torch.Tensor,
    *,
    time_window: float,
    basis_dim: int,
) -> torch.Tensor:
    """Fixed compact basis whose node-level mixture is learned by the encoder."""
    if basis_dim < 2:
        raise ValueError("basis_dim must be >= 2.")
    scale = max(float(time_window), 1e-12)
    x = torch.log1p(delta_times.clamp(min=0.0)) / np.log1p(scale)
    if basis_dim == 2:
        return torch.stack([torch.ones_like(x), x], dim=1)
    num_periodic = basis_dim - 2
    num_frequencies = (num_periodic + 1) // 2
    frequencies = torch.arange(
        1,
        num_frequencies + 1,
        dtype=x.dtype,
        device=x.device,
    )
    phases = np.pi * x.unsqueeze(1) * frequencies.unsqueeze(0)
    periodic = torch.stack([torch.sin(phases), torch.cos(phases)], dim=2)
    periodic = periodic.flatten(1)[:, :num_periodic]
    return torch.cat([torch.ones_like(x).unsqueeze(1), x.unsqueeze(1), periodic], dim=1)


def build_temporal_relational_context(
    *,
    src_node_ids: np.ndarray,
    dst_node_ids: np.ndarray,
    edge_ids: np.ndarray,
    node_interact_times: np.ndarray,
    reference_time: float,
    time_window: float,
    decay_gamma: float,
    lookup: torch.Tensor,
    num_rows: int,
    num_relations: int,
    time_basis_dim: int,
    output_dtype: torch.dtype,
    rows_are_precomputed: bool = False,
) -> Dict[str, torch.Tensor]:
    """Build degree-normalized outgoing/incoming relation and age summaries.

    The returned tensors contain no learned values, so a rolling provider can
    safely cache them while encoder parameters change during optimization.
    """
    device = lookup.device
    if rows_are_precomputed:
        src_rows = torch.as_tensor(src_node_ids, device=device).long().reshape(-1)
        dst_rows = torch.as_tensor(dst_node_ids, device=device).long().reshape(-1)
        relation_ids = torch.as_tensor(edge_ids, device=device).long().reshape(-1)
        event_times = torch.as_tensor(
            node_interact_times, dtype=output_dtype, device=device
        ).reshape(-1)
        if not (
            src_rows.numel()
            == dst_rows.numel()
            == relation_ids.numel()
            == event_times.numel()
        ):
            raise ValueError("Temporal-relational history tensors must have equal length.")
    else:
        src_np = np.asarray(src_node_ids, dtype=np.int64).reshape(-1)
        dst_np = np.asarray(dst_node_ids, dtype=np.int64).reshape(-1)
        rel_np = np.asarray(edge_ids, dtype=np.int64).reshape(-1)
        times_np = np.asarray(node_interact_times, dtype=np.float64).reshape(-1)
        if not (len(src_np) == len(dst_np) == len(rel_np) == len(times_np)):
            raise ValueError("Temporal-relational history arrays must have equal length.")

        window_start = float(reference_time) - float(time_window)
        valid_time_np = (times_np < float(reference_time)) & (times_np >= window_start)
        if not np.all(valid_time_np):
            src_np = src_np[valid_time_np]
            dst_np = dst_np[valid_time_np]
            rel_np = rel_np[valid_time_np]
            times_np = times_np[valid_time_np]

        src_ids = torch.as_tensor(src_np, dtype=torch.long, device=device)
        dst_ids = torch.as_tensor(dst_np, dtype=torch.long, device=device)
        relation_ids = torch.as_tensor(rel_np, dtype=torch.long, device=device)
        event_times = torch.as_tensor(times_np, dtype=output_dtype, device=device)
        valid_ids = (
            (src_ids >= 0)
            & (src_ids < lookup.numel())
            & (dst_ids >= 0)
            & (dst_ids < lookup.numel())
        )
        src_rows = torch.full_like(src_ids, -1)
        dst_rows = torch.full_like(dst_ids, -1)
        src_rows[valid_ids] = lookup[src_ids[valid_ids]]
        dst_rows[valid_ids] = lookup[dst_ids[valid_ids]]

    if not rows_are_precomputed:
        valid_rows = (
            (src_rows >= 0)
            & (src_rows < int(num_rows))
            & (dst_rows >= 0)
            & (dst_rows < int(num_rows))
            & (relation_ids >= 0)
            & (relation_ids < int(num_relations))
        )
        src_rows = src_rows[valid_rows]
        dst_rows = dst_rows[valid_rows]
        relation_ids = relation_ids[valid_rows]
        event_times = event_times[valid_rows]

    if src_rows.numel() == 0:
        zero_time = torch.zeros(
            (int(num_rows), int(time_basis_dim)), dtype=output_dtype, device=device
        )
        zero_relation = torch.zeros(
            (int(num_rows), int(num_relations)), dtype=output_dtype, device=device
        )
        return {
            "out_relation_profile": zero_relation,
            "in_relation_profile": zero_relation.clone(),
            "out_time_profile": zero_time,
            "in_time_profile": zero_time.clone(),
            "direction_mask": torch.zeros(
                (int(num_rows), 2), dtype=torch.bool, device=device
            ),
        }

    delta_times = float(reference_time) - event_times
    weights = torch.exp(-float(decay_gamma) * delta_times)
    out_degree = torch.zeros(int(num_rows), dtype=output_dtype, device=device)
    in_degree = torch.zeros_like(out_degree)
    out_degree.index_add_(0, src_rows, weights)
    in_degree.index_add_(0, dst_rows, weights)
    out_values = weights / out_degree.index_select(0, src_rows).clamp(min=1e-12)
    in_values = weights / in_degree.index_select(0, dst_rows).clamp(min=1e-12)

    out_relation = torch.zeros(
        (int(num_rows), int(num_relations)), dtype=output_dtype, device=device
    )
    in_relation = torch.zeros_like(out_relation)
    out_relation.index_put_(
        (src_rows, relation_ids), out_values, accumulate=True
    )
    in_relation.index_put_(
        (dst_rows, relation_ids), in_values, accumulate=True
    )

    time_basis = _relative_time_basis(
        delta_times,
        time_window=float(time_window),
        basis_dim=int(time_basis_dim),
    )
    out_time = torch.zeros(
        (int(num_rows), int(time_basis_dim)), dtype=output_dtype, device=device
    )
    in_time = torch.zeros_like(out_time)
    out_time.index_add_(0, src_rows, time_basis * out_values.unsqueeze(1))
    in_time.index_add_(0, dst_rows, time_basis * in_values.unsqueeze(1))
    return {
        "out_relation_profile": out_relation,
        "in_relation_profile": in_relation,
        "out_time_profile": out_time,
        "in_time_profile": in_time,
        "direction_mask": torch.stack([out_degree > 0, in_degree > 0], dim=1),
    }


class RollingSmoothedEmbeddingProvider:
    """
    Per-batch rolling semantic smoothing:
    - optionally initialize each source from all of its history before the batch
    - recompute smoothed embeddings at batch time using history before batch
    - either commit current positives or read from an authoritative full stream

    ``history_is_complete=True`` is the preferred temporal-evaluation mode. The
    input arrays may contain future edges because every materialized graph uses
    a strict timestamp slice ``[batch_time - window, batch_time)``. This keeps
    evaluation independent of which subset of positives happens to be scored.
    """
    def __init__(
        self,
        base_embeddings: torch.Tensor,
        lookup: torch.Tensor,
        init_src_node_ids: np.ndarray,
        init_dst_node_ids: np.ndarray,
        init_node_interact_times: np.ndarray,
        smooth_time_window: float,
        smooth_steps: int,
        smooth_decay_gamma: float,
        smooth_undirected: bool,
        smooth_log_dampen: bool,
        smooth_supernode_strength: float,
        smooth_endpoint_topk_recent: int = 0,
        smooth_endpoint_topk_mode: str = "per_node",
        export_norm_adj: bool = False,
        export_sum_adj: bool = False,
        export_binary_adj: bool = False,
        export_binary_two_hop_adj: bool = False,
        smooth_min_time: Optional[float] = None,
        source_init: str = "raw",
        apply_smoothing: bool = True,
        history_is_complete: bool = False,
        materialize_smoothed_embeddings: bool = True,
        init_edge_ids: Optional[np.ndarray] = None,
        export_temporal_relational_context: bool = False,
        temporal_relational_num_relations: int = 0,
        temporal_relational_time_basis_dim: int = 16,
    ):
        self.base_embeddings = base_embeddings
        self.lookup = lookup
        init_src = np.asarray(init_src_node_ids, dtype=np.int64).reshape(-1)
        init_dst = np.asarray(init_dst_node_ids, dtype=np.int64).reshape(-1)
        init_times = np.asarray(init_node_interact_times, dtype=np.float64).reshape(-1)
        init_edges = None
        if init_edge_ids is not None:
            init_edges = np.asarray(init_edge_ids, dtype=np.int64).reshape(-1)
        if not (len(init_src) == len(init_dst) == len(init_times)):
            raise ValueError(
                "Rolling history source, destination, and time arrays must have "
                "equal length."
            )
        if init_edges is not None and len(init_edges) != len(init_times):
            raise ValueError("Rolling relation IDs must align with rolling history arrays.")
        if len(init_times) < 2 or np.all(init_times[1:] >= init_times[:-1]):
            self.init_src_node_ids = init_src.copy()
            self.init_dst_node_ids = init_dst.copy()
            self.init_node_interact_times = init_times.copy()
            self.init_edge_ids = None if init_edges is None else init_edges.copy()
        else:
            init_order = np.argsort(init_times, kind="mergesort")
            self.init_src_node_ids = init_src[init_order].copy()
            self.init_dst_node_ids = init_dst[init_order].copy()
            self.init_node_interact_times = init_times[init_order].copy()
            self.init_edge_ids = (
                None if init_edges is None else init_edges[init_order].copy()
            )

        self.smooth_time_window = float(smooth_time_window)
        self.smooth_steps = int(smooth_steps)
        self.smooth_decay_gamma = float(smooth_decay_gamma)
        self.smooth_undirected = bool(smooth_undirected)
        self.smooth_log_dampen = bool(smooth_log_dampen)
        self.smooth_supernode_strength = float(smooth_supernode_strength)
        self.smooth_endpoint_topk_recent = int(smooth_endpoint_topk_recent)
        self.smooth_endpoint_topk_mode = str(smooth_endpoint_topk_mode)
        self.export_norm_adj = bool(export_norm_adj)
        self.export_sum_adj = bool(export_sum_adj)
        self.export_binary_adj = bool(export_binary_adj)
        self.export_binary_two_hop_adj = bool(export_binary_two_hop_adj)
        self.smooth_min_time = smooth_min_time
        self.source_init = normalize_semantic_source_init(source_init)
        self.apply_smoothing = bool(apply_smoothing)
        self.history_is_complete = bool(history_is_complete)
        self.materialize_smoothed_embeddings = bool(materialize_smoothed_embeddings)
        self.export_temporal_relational_context = bool(
            export_temporal_relational_context
        )
        self.temporal_relational_num_relations = int(
            temporal_relational_num_relations
        )
        self.temporal_relational_time_basis_dim = int(
            temporal_relational_time_basis_dim
        )
        if self.export_temporal_relational_context:
            if self.init_edge_ids is None:
                raise ValueError(
                    "Temporal-relational rolling context requires init_edge_ids."
                )
            if self.temporal_relational_num_relations < 1:
                raise ValueError("temporal_relational_num_relations must be >= 1.")
            if not self.history_is_complete:
                raise ValueError(
                    "The device-cached temporal-relational path requires "
                    "history_is_complete=True."
                )

        self._temporal_relational_src_rows = None
        self._temporal_relational_dst_rows = None
        self._temporal_relational_edge_ids = None
        self._temporal_relational_times = None
        if self.export_temporal_relational_context:
            device = self.lookup.device
            src_ids_t = torch.as_tensor(
                self.init_src_node_ids, dtype=torch.long, device=device
            )
            dst_ids_t = torch.as_tensor(
                self.init_dst_node_ids, dtype=torch.long, device=device
            )
            valid_ids = (
                (src_ids_t >= 0)
                & (src_ids_t < self.lookup.numel())
                & (dst_ids_t >= 0)
                & (dst_ids_t < self.lookup.numel())
            )
            src_rows_t = torch.full_like(src_ids_t, -1)
            dst_rows_t = torch.full_like(dst_ids_t, -1)
            src_rows_t[valid_ids] = self.lookup[src_ids_t[valid_ids]]
            dst_rows_t[valid_ids] = self.lookup[dst_ids_t[valid_ids]]
            edge_ids_t = torch.as_tensor(
                self.init_edge_ids, dtype=torch.long, device=device
            )
            valid_cached_history = (
                (src_rows_t >= 0)
                & (src_rows_t < int(self.base_embeddings.size(0)))
                & (dst_rows_t >= 0)
                & (dst_rows_t < int(self.base_embeddings.size(0)))
                & (edge_ids_t >= 0)
                & (edge_ids_t < self.temporal_relational_num_relations)
            )
            if not bool(valid_cached_history.all().item()):
                raise ValueError(
                    "Temporal-relational history contains an unmapped node or "
                    "relation ID outside the relation feature table."
                )
            # int32 halves the persistent index cache; each small active window
            # is promoted to int64 only when consumed by index_add.
            self._temporal_relational_src_rows = src_rows_t.to(torch.int32)
            self._temporal_relational_dst_rows = dst_rows_t.to(torch.int32)
            self._temporal_relational_edge_ids = edge_ids_t.to(torch.int32)
            self._temporal_relational_times = torch.as_tensor(
                self.init_node_interact_times, dtype=torch.float32, device=device
            )

        self.history_src_node_ids = None
        self.history_dst_node_ids = None
        self.history_node_interact_times = None
        self.history_edge_ids = None
        self.current_batch_time = None
        self.current_base_embeddings = None
        self.current_embeddings = None
        self.current_norm_adj = None
        self.current_sum_adj = None
        self.current_binary_adj = None
        self.current_binary_two_hop_adj = None
        self.current_temporal_relational_context = None
        self._base_embedding_signature = None
        self._source_history_sums = None
        self._source_history_counts = None
        self._source_history_ptr = 0
        self.reset_history()

    @staticmethod
    def _embedding_signature(embeddings: torch.Tensor):
        return (
            int(embeddings.data_ptr()),
            int(getattr(embeddings, "_version", 0)),
            tuple(embeddings.shape),
            embeddings.dtype,
            embeddings.device,
        )

    def _invalidate_source_init_cache(self):
        self._source_history_sums = None
        self._source_history_counts = None
        self._source_history_ptr = 0
        self.current_base_embeddings = self.base_embeddings

    def set_base_embeddings(self, base_embeddings: torch.Tensor):
        """Update the batch's materialized base tensor without losing source init."""
        signature = self._embedding_signature(base_embeddings)
        if signature != self._base_embedding_signature:
            self.base_embeddings = base_embeddings
            self._base_embedding_signature = signature
            self._invalidate_source_init_cache()
            # A changed trainable/projected base must be reprocessed even when
            # the next loader batch begins at the same timestamp.
            self.current_batch_time = None
        else:
            self.base_embeddings = base_embeddings

    def _history_rows(self, start: int, end: int):
        if end <= start:
            empty = torch.empty((0,), dtype=torch.long, device=self.lookup.device)
            return empty, empty

        src_ids = (
            torch.from_numpy(self.history_src_node_ids[start:end])
            .long()
            .to(self.lookup.device)
        )
        dst_ids = (
            torch.from_numpy(self.history_dst_node_ids[start:end])
            .long()
            .to(self.lookup.device)
        )
        in_range = (
            (src_ids >= 0)
            & (src_ids < self.lookup.size(0))
            & (dst_ids >= 0)
            & (dst_ids < self.lookup.size(0))
        )
        if not in_range.any():
            empty = torch.empty((0,), dtype=torch.long, device=self.lookup.device)
            return empty, empty

        src_rows = self.lookup[src_ids[in_range]]
        dst_rows = self.lookup[dst_ids[in_range]]
        valid = (
            (src_rows >= 0)
            & (src_rows < self.base_embeddings.size(0))
            & (dst_rows >= 0)
            & (dst_rows < self.base_embeddings.size(0))
        )
        return src_rows[valid].to(self.base_embeddings.device), dst_rows[valid].to(
            self.base_embeddings.device
        )

    def _recompute_history_mean(self, history_end: int) -> torch.Tensor:
        base = self.base_embeddings
        src_rows, dst_rows = self._history_rows(0, history_end)
        if src_rows.numel() == 0:
            return base

        sums = torch.zeros_like(base)
        sums = sums.index_add(0, src_rows, base.index_select(0, dst_rows))
        counts = torch.zeros(base.size(0), dtype=base.dtype, device=base.device)
        counts = counts.index_add(
            0,
            src_rows,
            torch.ones(src_rows.numel(), dtype=base.dtype, device=base.device),
        )
        replace_mask = counts > 0
        means = sums / counts.clamp(min=1).unsqueeze(1)
        means = F.normalize(means, p=2, dim=1)
        return torch.where(replace_mask.unsqueeze(1), means, base)

    def _incremental_history_mean(self, history_end: int) -> torch.Tensor:
        base = self.base_embeddings
        if self._source_history_sums is None:
            self._source_history_sums = torch.zeros_like(base)
            self._source_history_counts = torch.zeros(
                base.size(0), dtype=torch.long, device=base.device
            )
            self.current_base_embeddings = base.clone()
            self._source_history_ptr = 0

        if history_end < self._source_history_ptr:
            self._invalidate_source_init_cache()
            return self._incremental_history_mean(history_end)

        src_rows, dst_rows = self._history_rows(self._source_history_ptr, history_end)
        if src_rows.numel() > 0:
            self._source_history_sums.index_add_(
                0, src_rows, base.index_select(0, dst_rows)
            )
            self._source_history_counts.index_add_(
                0,
                src_rows,
                torch.ones(src_rows.numel(), dtype=torch.long, device=base.device),
            )
            changed_rows = torch.unique(src_rows)
            changed_means = self._source_history_sums.index_select(0, changed_rows)
            changed_counts = self._source_history_counts.index_select(0, changed_rows)
            changed_means = changed_means / changed_counts.unsqueeze(1).to(base.dtype)
            changed_means = F.normalize(changed_means, p=2, dim=1)
            self.current_base_embeddings[changed_rows] = changed_means

        self._source_history_ptr = history_end
        return self.current_base_embeddings

    def _source_initialized_embeddings(self, batch_start_time: float) -> torch.Tensor:
        if self.source_init == "raw":
            return self.base_embeddings

        history_end = int(
            np.searchsorted(
                self.history_node_interact_times,
                float(batch_start_time),
                side="left",
            )
        )
        # Frozen semantic tables can update their running source means in O(new
        # edges). Trainable/projected tensors need a fresh differentiable mean.
        cacheable = (
            not self.base_embeddings.requires_grad
            and self.base_embeddings.grad_fn is None
        )
        if cacheable:
            return self._incremental_history_mean(history_end)
        return self._recompute_history_mean(history_end)

    def reset_history(self):
        if self.history_is_complete:
            # Authoritative history is immutable; share the sorted arrays rather
            # than copying the entire dataset once per split/provider.
            self.history_src_node_ids = self.init_src_node_ids
            self.history_dst_node_ids = self.init_dst_node_ids
            self.history_node_interact_times = self.init_node_interact_times
            self.history_edge_ids = self.init_edge_ids
        else:
            self.history_src_node_ids = self.init_src_node_ids.copy()
            self.history_dst_node_ids = self.init_dst_node_ids.copy()
            self.history_node_interact_times = self.init_node_interact_times.copy()
            self.history_edge_ids = (
                None if self.init_edge_ids is None else self.init_edge_ids.copy()
            )
        self.current_batch_time = None
        self._base_embedding_signature = self._embedding_signature(self.base_embeddings)
        self._invalidate_source_init_cache()
        self.current_base_embeddings = self.base_embeddings
        self.current_embeddings = self.base_embeddings
        self.current_norm_adj = None
        self.current_sum_adj = None
        self.current_binary_adj = None
        self.current_binary_two_hop_adj = None
        self.current_temporal_relational_context = None

    def prepare_batch(self, batch_node_interact_times: np.ndarray):
        if len(batch_node_interact_times) == 0:
            return

        batch_start_time = float(np.min(batch_node_interact_times))
        if self.current_batch_time == batch_start_time:
            return

        initialized_base_embeddings = self._source_initialized_embeddings(batch_start_time)
        self.current_base_embeddings = initialized_base_embeddings

        needs_window_history = (
            self.apply_smoothing
            or self.export_binary_adj
            or self.export_binary_two_hop_adj
            or self.export_temporal_relational_context
        )
        if not needs_window_history or len(self.history_src_node_ids) == 0:
            history_start = 0
            history_end = 0
            hist_src = np.empty((0,), dtype=np.int64)
            hist_dst = np.empty((0,), dtype=np.int64)
            hist_times = np.empty((0,), dtype=np.float64)
            hist_edges = np.empty((0,), dtype=np.int64)
        else:
            window_start_time = batch_start_time - self.smooth_time_window
            if self.smooth_min_time is not None:
                window_start_time = max(window_start_time, float(self.smooth_min_time))
            history_start = int(
                np.searchsorted(
                    self.history_node_interact_times,
                    window_start_time,
                    side="left",
                )
            )
            history_end = int(
                np.searchsorted(
                    self.history_node_interact_times,
                    batch_start_time,
                    side="left",
                )
            )
            hist_src = self.history_src_node_ids[history_start:history_end]
            hist_dst = self.history_dst_node_ids[history_start:history_end]
            hist_times = self.history_node_interact_times[history_start:history_end]
            hist_edges = (
                np.empty((0,), dtype=np.int64)
                if self.history_edge_ids is None
                else self.history_edge_ids[history_start:history_end]
            )

        if not self.apply_smoothing:
            smoothed = initialized_base_embeddings
            self.current_norm_adj = None
            self.current_sum_adj = None
        elif len(hist_src) == 0 and not self.export_norm_adj and not self.export_sum_adj:
            smoothed = initialized_base_embeddings
            self.current_norm_adj = None
            self.current_sum_adj = None
        else:
            # Learnable GCN/GIN paths consume only the smoothing-aligned sparse
            # operator and deliberately replace the parameter-free smoothed
            # features with message-passing output.  Build that operator with a
            # detached one-column view so the unused 1024-d sparse propagation
            # does not dominate memory or runtime on large E5 graphs.
            smoothing_input = initialized_base_embeddings
            if not self.materialize_smoothed_embeddings:
                smoothing_input = initialized_base_embeddings[:, :1].detach()
            smooth_out = smooth_embeddings_by_time_window_torch(
                embeddings=smoothing_input,
                src_node_ids=hist_src,
                dst_node_ids=hist_dst,
                node_interact_times=hist_times,
                time_window=self.smooth_time_window,
                num_steps=self.smooth_steps,
                symmetric_norm=True,
                decay_gamma=self.smooth_decay_gamma,
                undirected=self.smooth_undirected,
                log_dampen=self.smooth_log_dampen,
                supernode_strength=self.smooth_supernode_strength,
                endpoint_topk_recent=self.smooth_endpoint_topk_recent,
                endpoint_topk_mode=self.smooth_endpoint_topk_mode,
                reference_time=batch_start_time,
                device=initialized_base_embeddings.device,
                return_norm_adj=self.export_norm_adj,
                return_sum_adj=self.export_sum_adj,
            )
            if self.export_norm_adj and self.export_sum_adj:
                smoothed, self.current_norm_adj, self.current_sum_adj = smooth_out
            elif self.export_norm_adj:
                smoothed, self.current_norm_adj = smooth_out
                self.current_sum_adj = None
            elif self.export_sum_adj:
                smoothed, self.current_sum_adj = smooth_out
                self.current_norm_adj = None
            else:
                smoothed = smooth_out
                self.current_norm_adj = None
                self.current_sum_adj = None

            if not self.materialize_smoothed_embeddings:
                smoothed = initialized_base_embeddings

        if self.export_binary_adj or self.export_binary_two_hop_adj:
            self.current_binary_adj = build_binary_history_adj(
                src_node_ids=hist_src,
                dst_node_ids=hist_dst,
                lookup=self.lookup,
                num_rows=int(initialized_base_embeddings.size(0)),
                undirected=True,
            )
        else:
            self.current_binary_adj = None

        if self.export_binary_two_hop_adj:
            try:
                self.current_binary_two_hop_adj = build_two_hop_binary_adj(self.current_binary_adj)
            except torch.cuda.OutOfMemoryError:
                if self.current_binary_adj is not None and self.current_binary_adj.storage.row().device.type == 'cuda':
                    torch.cuda.empty_cache()
                self.current_binary_two_hop_adj = None
        else:
            self.current_binary_two_hop_adj = None

        if self.export_temporal_relational_context:
            self.current_temporal_relational_context = build_temporal_relational_context(
                src_node_ids=self._temporal_relational_src_rows[history_start:history_end],
                dst_node_ids=self._temporal_relational_dst_rows[history_start:history_end],
                edge_ids=self._temporal_relational_edge_ids[history_start:history_end],
                node_interact_times=self._temporal_relational_times[history_start:history_end],
                reference_time=batch_start_time,
                time_window=self.smooth_time_window,
                decay_gamma=self.smooth_decay_gamma,
                lookup=self.lookup,
                num_rows=int(initialized_base_embeddings.size(0)),
                num_relations=self.temporal_relational_num_relations,
                time_basis_dim=self.temporal_relational_time_basis_dim,
                output_dtype=initialized_base_embeddings.dtype,
                rows_are_precomputed=True,
            )
        else:
            self.current_temporal_relational_context = None

        smoothed = smoothed / torch.norm(smoothed, dim=1, keepdim=True).clamp(min=1e-12)
        self.current_embeddings = smoothed
        self.current_batch_time = batch_start_time

    def commit_batch(
        self,
        batch_src_node_ids: np.ndarray,
        batch_dst_node_ids: np.ndarray,
        batch_node_interact_times: np.ndarray,
    ):
        if len(batch_src_node_ids) == 0:
            return

        batch_src = np.asarray(batch_src_node_ids, dtype=np.int64).reshape(-1)
        batch_dst = np.asarray(batch_dst_node_ids, dtype=np.int64).reshape(-1)
        batch_times = np.asarray(batch_node_interact_times, dtype=np.float64).reshape(-1)
        if not (len(batch_src) == len(batch_dst) == len(batch_times)):
            raise ValueError("Committed rolling-history arrays must have equal length.")
        if self.history_is_complete:
            # The authoritative stream already contains these real edges. A
            # no-op avoids duplicates and makes results independent of whether
            # the caller scores the full, transductive, or inductive subset.
            return
        batch_order = np.argsort(batch_times, kind="mergesort")
        batch_src = batch_src[batch_order]
        batch_dst = batch_dst[batch_order]
        batch_times = batch_times[batch_order]

        out_of_order = (
            len(self.history_node_interact_times) > 0
            and len(batch_times) > 0
            and batch_times[0] < self.history_node_interact_times[-1]
        )
        self.history_src_node_ids = np.concatenate([self.history_src_node_ids, batch_src])
        self.history_dst_node_ids = np.concatenate([self.history_dst_node_ids, batch_dst])
        self.history_node_interact_times = np.concatenate(
            [self.history_node_interact_times, batch_times]
        )
        if out_of_order:
            order = np.argsort(self.history_node_interact_times, kind="mergesort")
            self.history_src_node_ids = self.history_src_node_ids[order]
            self.history_dst_node_ids = self.history_dst_node_ids[order]
            self.history_node_interact_times = self.history_node_interact_times[order]
            self._invalidate_source_init_cache()
            self.current_batch_time = None

def apply_attention_pool_message_passing_for_nodes(
    mp_encoder: TemporalSelfAttentionPooling,
    embeddings: torch.Tensor,
    lookup: torch.Tensor,
    query_node_ids: np.ndarray,
    query_time: float,
    neighbor_index: TemporalNeighborIndex,
    num_neighbors: int,
) -> torch.Tensor:
    if num_neighbors <= 0:
        raise ValueError("num_neighbors must be > 0.")

    query_nodes = np.asarray(query_node_ids, dtype=np.int64)
    if query_nodes.size == 0:
        return embeddings
    query_nodes = np.unique(query_nodes)

    device = embeddings.device
    query_nodes_t = torch.from_numpy(query_nodes).long().to(device)
    in_range = (query_nodes_t >= 0) & (query_nodes_t < lookup.size(0))
    if not in_range.any():
        return embeddings
    query_nodes_t = query_nodes_t[in_range]
    query_rows_t = lookup[query_nodes_t]
    row_valid = query_rows_t >= 0
    if not row_valid.any():
        return embeddings
    query_nodes_t = query_nodes_t[row_valid]
    query_rows_t = query_rows_t[row_valid]

    query_nodes_np = query_nodes_t.detach().cpu().numpy()
    query_times = np.full((query_nodes_np.shape[0],), float(query_time), dtype=np.float64)
    neigh_ids_np, neigh_times_np, neigh_mask_np = neighbor_index.get_recent_neighbors(
        node_ids=query_nodes_np,
        query_times=query_times,
        num_neighbors=num_neighbors,
        return_times=True,
    )

    neigh_nodes = torch.from_numpy(neigh_ids_np).long().to(device)
    neigh_hist_mask = torch.from_numpy(neigh_mask_np).to(device=device, dtype=torch.bool)
    neigh_in_range = (neigh_nodes >= 0) & (neigh_nodes < lookup.size(0))
    neigh_idx = torch.full_like(neigh_nodes, -1)
    if neigh_in_range.any():
        neigh_idx[neigh_in_range] = lookup[neigh_nodes[neigh_in_range]]
    neigh_valid = neigh_hist_mask & (neigh_idx >= 0)

    safe_neigh_idx = neigh_idx.clone()
    safe_neigh_idx[~neigh_valid] = 0
    history_emb = embeddings[safe_neigh_idx]
    history_emb = history_emb * neigh_valid.unsqueeze(-1).to(history_emb.dtype)
    neigh_times_t = torch.from_numpy(neigh_times_np).to(device=device, dtype=torch.float32)
    history_delta_t = torch.full_like(neigh_times_t, float(query_time)) - neigh_times_t
    history_delta_t = torch.clamp(history_delta_t, min=0.0)
    history_delta_t = history_delta_t.masked_fill(~neigh_valid, 0.0)

    query_emb = embeddings[query_rows_t]
    updated_query_emb = mp_encoder(
        node_emb=query_emb,
        history_emb=history_emb,
        history_mask=neigh_valid,
        history_delta_t=history_delta_t,
    )

    out_embeddings = embeddings.clone()
    out_embeddings[query_rows_t] = updated_query_emb
    return out_embeddings

def materialize_base_embeddings(
    embeddings: torch.Tensor,
    entity_embedding_table: Optional[nn.Module] = None,
    structural_seed_features: Optional[torch.Tensor] = None,
    structural_seed_projector: Optional[nn.Module] = None,
) -> torch.Tensor:
    if entity_embedding_table is None:
        out = embeddings
    else:
        out = entity_embedding_table()
    if structural_seed_projector is not None:
        if structural_seed_features is None:
            raise ValueError("structural_seed_projector is set but structural_seed_features is None.")
        seed = structural_seed_projector(
            structural_seed_features.to(device=out.device, dtype=out.dtype)
        )
        out = out + seed
        out = F.normalize(out, dim=1)
    return out


def apply_static_smoothing_operator(
    embeddings: torch.Tensor,
    norm_adj: Optional[torch.Tensor],
    num_steps: int,
) -> torch.Tensor:
    if norm_adj is None:
        return embeddings

    steps = max(1, int(num_steps))
    if norm_adj.size(0) == embeddings.size(0):
        out = embeddings
        drop_supernode = False
    elif norm_adj.size(0) == embeddings.size(0) + 1:
        super_emb = embeddings.mean(dim=0, keepdim=True)
        out = torch.cat([embeddings, super_emb], dim=0)
        drop_supernode = True
    else:
        raise ValueError(
            f"Static smoothing adj shape mismatch: adj={tuple(norm_adj.shape)}, "
            f"embeddings={tuple(embeddings.shape)}."
        )

    for _ in range(steps):
        out = torch.sparse.mm(norm_adj, out)

    if drop_supernode:
        out = out[:embeddings.size(0)]
    return F.normalize(out, dim=1)
