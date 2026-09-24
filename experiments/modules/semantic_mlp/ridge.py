from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedRandomProjection(nn.Module):
    """Seeded, frozen Rademacher projection used before graph propagation."""

    def __init__(self, input_dim: int, output_dim: int, seed: int = 42):
        super().__init__()
        input_dim = int(input_dim)
        output_dim = int(output_dim)
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Projection dimensions must be positive.")
        if output_dim > input_dim:
            raise ValueError(
                f"Projection output_dim ({output_dim}) must be <= input_dim ({input_dim})."
            )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        signs = torch.randint(
            0,
            2,
            (output_dim, input_dim),
            generator=generator,
            dtype=torch.int8,
        )
        weight = signs.to(torch.float32).mul_(2.0).sub_(1.0)
        weight.div_(float(output_dim) ** 0.5)
        self.register_buffer("weight", weight)

    @property
    def input_dim(self) -> int:
        return int(self.weight.size(1))

    @property
    def output_dim(self) -> int:
        return int(self.weight.size(0))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.linear(embeddings, self.weight.to(dtype=embeddings.dtype))


class SemanticRidgeScorer(nn.Module):
    """Diagonal semantic interaction plus auxiliary features and a fitted ridge head."""

    expects_raw_auxiliary_features = True

    def __init__(
        self,
        input_dim: int,
        auxiliary_dim: int = 0,
        semantic_feature_mode: str = "hadamard",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.auxiliary_dim = int(auxiliary_dim)
        self.semantic_feature_mode = str(semantic_feature_mode).strip().lower()
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if self.auxiliary_dim < 0:
            raise ValueError("auxiliary_dim must be non-negative.")
        if self.semantic_feature_mode not in {"hadamard", "cosine"}:
            raise ValueError("semantic_feature_mode must be one of: hadamard, cosine.")
        semantic_dim = self.input_dim if self.semantic_feature_mode == "hadamard" else 1
        feature_dim = semantic_dim + self.auxiliary_dim
        self.register_buffer("feature_scale", torch.ones(feature_dim, dtype=torch.float32))
        self.register_buffer("coefficient", torch.zeros(feature_dim, dtype=torch.float32))
        self.register_buffer("bias", torch.zeros((), dtype=torch.float32))

    @property
    def feature_dim(self) -> int:
        return int(self.coefficient.numel())

    def build_features(
        self,
        src_emb: torch.Tensor,
        dst_emb: torch.Tensor,
        auxiliary_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        semantic_product = src_emb * dst_emb
        if self.semantic_feature_mode == "cosine":
            # Propagated embeddings are normalized by the surrounding pipeline, so
            # this dot product is their cosine similarity. Keep a defensive norm for
            # direct callers and older cached embeddings.
            semantic_features = F.cosine_similarity(
                src_emb,
                dst_emb,
                dim=1,
                eps=1e-8,
            ).unsqueeze(1)
        else:
            semantic_features = semantic_product
        parts = [semantic_features]
        if self.auxiliary_dim > 0:
            if auxiliary_features is None:
                raise ValueError(
                    "SemanticRidgeScorer requires auxiliary features because auxiliary_dim > 0."
                )
            if auxiliary_features.ndim != 2 or auxiliary_features.size(1) != self.auxiliary_dim:
                raise ValueError(
                    "Auxiliary feature shape mismatch: "
                    f"expected [N, {self.auxiliary_dim}], got {tuple(auxiliary_features.shape)}."
                )
            parts.append(auxiliary_features.to(device=src_emb.device, dtype=src_emb.dtype))
        return torch.cat(parts, dim=1)

    def set_solution(self, coefficient: torch.Tensor, feature_scale: torch.Tensor) -> None:
        coefficient = torch.as_tensor(coefficient, dtype=torch.float32, device=self.coefficient.device)
        feature_scale = torch.as_tensor(feature_scale, dtype=torch.float32, device=self.feature_scale.device)
        if coefficient.shape != self.coefficient.shape:
            raise ValueError(
                f"Coefficient shape mismatch: expected {tuple(self.coefficient.shape)}, "
                f"got {tuple(coefficient.shape)}."
            )
        if feature_scale.shape != self.feature_scale.shape:
            raise ValueError(
                f"Feature-scale shape mismatch: expected {tuple(self.feature_scale.shape)}, "
                f"got {tuple(feature_scale.shape)}."
            )
        self.coefficient.copy_(coefficient)
        self.feature_scale.copy_(feature_scale.clamp_min(1e-8))

    def forward(
        self,
        src_emb: torch.Tensor,
        dst_emb: torch.Tensor,
        auxiliary_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = self.build_features(src_emb, dst_emb, auxiliary_features)
        scale = self.feature_scale.to(device=features.device, dtype=features.dtype)
        coefficient = self.coefficient.to(device=features.device, dtype=features.dtype)
        return (features / scale).matmul(coefficient) + self.bias.to(
            device=features.device,
            dtype=features.dtype,
        )


def build_ridge_edge_features(
    *,
    model: SemanticRidgeScorer,
    embeddings: torch.Tensor,
    lookup: torch.Tensor,
    src_ids: Union[np.ndarray, torch.Tensor],
    dst_ids: Union[np.ndarray, torch.Tensor],
    auxiliary_features: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build aligned edge features and a validity mask for raw node ids."""

    device = embeddings.device
    src = torch.as_tensor(src_ids, dtype=torch.long, device=device)
    dst = torch.as_tensor(dst_ids, dtype=torch.long, device=device)
    max_id = int(lookup.size(0))
    in_range = (src >= 0) & (src < max_id) & (dst >= 0) & (dst < max_id)
    src_idx = torch.full_like(src, -1)
    dst_idx = torch.full_like(dst, -1)
    src_idx[in_range] = lookup[src[in_range]]
    dst_idx[in_range] = lookup[dst[in_range]]
    valid = in_range & (src_idx >= 0) & (dst_idx >= 0)

    features = torch.zeros(
        (int(src.numel()), model.feature_dim),
        dtype=embeddings.dtype,
        device=device,
    )
    if valid.any():
        valid_aux = None
        if auxiliary_features is not None:
            valid_aux = auxiliary_features.to(device=device, dtype=embeddings.dtype)[valid]
        features[valid] = model.build_features(
            embeddings[src_idx[valid]],
            embeddings[dst_idx[valid]],
            valid_aux,
        )
    return features, valid


@dataclass
class RidgeFitStats:
    pair_count: int
    feature_dim: int
    lambda_value: float
    min_feature_scale: float
    max_feature_scale: float
    coefficient_norm: float


class PairwiseRidgeAccumulator:
    """Streaming sufficient statistics for min ||D w - 1||^2 + lambda ||w||^2."""

    def __init__(self, feature_dim: int):
        feature_dim = int(feature_dim)
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        self.feature_dim = feature_dim
        self.gram = torch.zeros((feature_dim, feature_dim), dtype=torch.float64)
        self.rhs = torch.zeros(feature_dim, dtype=torch.float64)
        self.pair_count = 0

    def update(self, differences: torch.Tensor) -> None:
        if differences.ndim != 2 or differences.size(1) != self.feature_dim:
            raise ValueError(
                f"Expected differences [N, {self.feature_dim}], got {tuple(differences.shape)}."
            )
        if differences.numel() == 0:
            return
        finite = torch.isfinite(differences).all(dim=1)
        differences = differences[finite]
        if differences.numel() == 0:
            return

        # Fast device-side GEMM; retain stable float64 running totals on the host.
        diff = differences.detach().to(dtype=torch.float32)
        batch_gram = diff.transpose(0, 1).matmul(diff).to(device="cpu", dtype=torch.float64)
        batch_rhs = diff.sum(dim=0).to(device="cpu", dtype=torch.float64)
        self.gram.add_(batch_gram)
        self.rhs.add_(batch_rhs)
        self.pair_count += int(diff.size(0))

    def solve(self, lambda_value: float) -> Tuple[torch.Tensor, torch.Tensor, RidgeFitStats]:
        if self.pair_count <= 0:
            raise ValueError("Cannot fit ridge scorer without any valid positive-negative pairs.")
        lambda_value = float(lambda_value)
        if lambda_value <= 0.0:
            raise ValueError("ridge lambda must be > 0.")

        gram = self.gram / float(self.pair_count)
        rhs = self.rhs / float(self.pair_count)
        scale = torch.sqrt(torch.diagonal(gram).clamp_min(0.0))
        scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale))
        scaled_gram = gram / scale[:, None] / scale[None, :]
        scaled_rhs = rhs / scale
        system = scaled_gram + lambda_value * torch.eye(self.feature_dim, dtype=torch.float64)
        coefficient = torch.linalg.solve(system, scaled_rhs)

        stats = RidgeFitStats(
            pair_count=self.pair_count,
            feature_dim=self.feature_dim,
            lambda_value=lambda_value,
            min_feature_scale=float(scale.min().item()),
            max_feature_scale=float(scale.max().item()),
            coefficient_norm=float(coefficient.norm().item()),
        )
        return coefficient.float(), scale.float(), stats


__all__ = [
    "FixedRandomProjection",
    "PairwiseRidgeAccumulator",
    "RidgeFitStats",
    "SemanticRidgeScorer",
    "build_ridge_edge_features",
]
