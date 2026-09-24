from __future__ import annotations

import torch
import torch.nn as nn


class SparseLLMResidualFusion(nn.Module):
    """Optional LLM correction on top of a fully supervised graph logit."""

    def __init__(
        self,
        graph_dim: int,
        llm_dim: int,
        adapter_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        fusion_input: str = "embedding_and_margin",
    ):
        super().__init__()
        if fusion_input not in {"margin", "embedding", "embedding_and_margin"}:
            raise ValueError(f"Unsupported fusion_input={fusion_input!r}.")
        self.fusion_input = str(fusion_input)
        self.use_embedding = fusion_input in {"embedding", "embedding_and_margin"}
        self.use_margin = fusion_input in {"margin", "embedding_and_margin"}
        self.llm_adapter = None
        correction_input_dim = int(graph_dim)
        if self.use_embedding:
            self.llm_adapter = nn.Sequential(
                nn.Linear(int(llm_dim), int(adapter_dim)),
                nn.GELU(),
            )
            correction_input_dim += int(adapter_dim)
        if self.use_margin:
            correction_input_dim += 1
        self.correction = nn.Sequential(
            nn.Linear(correction_input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(
        self,
        base_logits: torch.Tensor,
        graph_features: torch.Tensor,
        llm_embeddings: torch.Tensor | None = None,
        llm_available: torch.Tensor | None = None,
        llm_margin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base_logits = base_logits.reshape(-1)
        if llm_available is None:
            llm_available = torch.ones_like(base_logits, dtype=torch.bool)
        else:
            llm_available = llm_available.reshape(-1).bool()
        if graph_features.size(0) != base_logits.numel():
            raise ValueError("graph_features and base_logits must have the same row count.")
        if self.use_embedding and llm_embeddings is None:
            raise ValueError(f"llm_embeddings are required for {self.fusion_input!r} fusion.")
        if self.use_embedding and llm_embeddings.size(0) != base_logits.numel():
            raise ValueError("llm_embeddings and base_logits must have the same row count.")
        if self.use_margin and llm_margin is None:
            raise ValueError(f"llm_margin is required for {self.fusion_input!r} fusion.")
        if self.use_margin and llm_margin.numel() != base_logits.numel():
            raise ValueError("llm_margin and base_logits must have the same row count.")

        fused_logits = base_logits.clone()
        if llm_available.any():
            features = [graph_features[llm_available]]
            if self.use_embedding:
                features.append(self.llm_adapter(llm_embeddings[llm_available]))
            if self.use_margin:
                features.append(llm_margin.reshape(-1, 1)[llm_available])
            correction = self.correction(torch.cat(features, dim=-1)).squeeze(-1)
            fused_logits[llm_available] = fused_logits[llm_available] + correction
        return fused_logits
