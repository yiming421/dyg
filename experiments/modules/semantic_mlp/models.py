from __future__ import annotations

import argparse
import math
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ('true', '1', 'yes', 'y', 't'):
        return True
    if s in ('false', '0', 'no', 'n', 'f'):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value: true/false")


def make_activation_layer(activation: str) -> nn.Module:
    if activation == 'relu':
        return nn.ReLU()
    if activation == 'silu':
        return nn.SiLU()
    if activation == 'gelu':
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {activation}")


class SemanticMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        auxiliary_dim: int = 0,
        pair_feature_mode: str = 'concat',
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_layernorm: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        make_activation_layer(activation)

        layers = []
        self.auxiliary_dim = int(auxiliary_dim)
        if self.auxiliary_dim < 0:
            raise ValueError("auxiliary_dim must be >= 0")
        self.pair_feature_mode = str(pair_feature_mode)
        if self.pair_feature_mode not in {'concat', 'hadamard'}:
            raise ValueError("pair_feature_mode must be one of: concat, hadamard")
        pair_dim = input_dim * 2 if self.pair_feature_mode == 'concat' else input_dim
        in_dim = pair_dim + self.auxiliary_dim

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(make_activation_layer(activation))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    @property
    def pair_representation_dim(self) -> int:
        """Width of the representation immediately before the scalar link head."""
        return int(self.mlp[-1].in_features)

    def encode_pair(
        self,
        src_emb: torch.Tensor,
        dst_emb: torch.Tensor,
        auxiliary_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the checkpoint-compatible pair representation used by the head."""
        if self.pair_feature_mode == 'concat':
            parts = [src_emb, dst_emb]
        else:
            parts = [src_emb * dst_emb]
        if self.auxiliary_dim > 0:
            if auxiliary_features is None:
                raise ValueError("SemanticMLP was built with auxiliary_dim > 0 but no auxiliary_features were provided.")
            if auxiliary_features.ndim != 2 or auxiliary_features.size(1) != self.auxiliary_dim:
                raise ValueError(
                    "auxiliary_features shape mismatch: "
                    f"expected [N, {self.auxiliary_dim}], got {tuple(auxiliary_features.shape)}"
                )
            parts.append(auxiliary_features.to(device=src_emb.device, dtype=src_emb.dtype))
        representation = torch.cat(parts, dim=1)
        for layer in list(self.mlp.children())[:-1]:
            representation = layer(representation)
        return representation

    def forward(
        self,
        src_emb: torch.Tensor,
        dst_emb: torch.Tensor,
        auxiliary_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        representation = self.encode_pair(src_emb, dst_emb, auxiliary_features)
        return self.mlp[-1](representation).squeeze(-1)


class SemanticNCNScorer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_layernorm: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        def make_branch() -> nn.Sequential:
            layers = [nn.Linear(input_dim, hidden_dim)]
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(make_activation_layer(activation))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            return nn.Sequential(*layers)

        head_layers = []
        in_dim = hidden_dim
        for _ in range(num_layers - 1):
            head_layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layernorm:
                head_layers.append(nn.LayerNorm(hidden_dim))
            head_layers.append(make_activation_layer(activation))
            head_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        head_layers.append(nn.Linear(in_dim, 1))

        self.xij_branch = make_branch()
        self.xcn_branch = make_branch()
        self.head = nn.Sequential(*head_layers)
        self.beta = nn.Parameter(torch.ones(1, dtype=torch.float32))

    @property
    def requires_common_neighbor_context(self) -> bool:
        return True

    def forward(
        self,
        src_emb: torch.Tensor,
        dst_emb: torch.Tensor,
        cn_emb: torch.Tensor,
        has_common_neighbors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        xij = self.xij_branch(src_emb * dst_emb)
        xcn = self.xcn_branch(cn_emb)
        if has_common_neighbors is not None:
            xcn = xcn * has_common_neighbors.unsqueeze(1).to(dtype=xcn.dtype)
        return self.head(xij + self.beta * xcn).squeeze(-1)


class FrequencySelectiveStructureEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_neighbors: int,
        tau: float = 0.2,
        kernel_size: int = 3,
        dropout: float = 0.1,
        activation: str = 'gelu',
        time_encoder_type: str = 'mlp',
        time_encoder_fourier_dim: int = 32,
        time_encoder_rbf_dim: int = 32,
        time_encoder_rbf_gamma: float = 16.0,
        use_soft_mask: bool = True,
    ):
        super().__init__()
        if num_neighbors <= 0:
            raise ValueError("num_neighbors must be > 0.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")

        self.input_dim = int(input_dim)
        self.num_neighbors = int(num_neighbors)
        self.tau = float(tau)
        self.use_soft_mask = bool(use_soft_mask)
        self.time_encoder = RelativeTimeEncoder(
            output_dim=input_dim,
            encoder_type=time_encoder_type,
            fourier_dim=time_encoder_fourier_dim,
            rbf_dim=time_encoder_rbf_dim,
            rbf_gamma=time_encoder_rbf_gamma,
        )
        freq_bins = self.num_neighbors // 2 + 1
        self.freq_gain_real = nn.Parameter(torch.ones(freq_bins, input_dim))
        self.freq_gain_imag = nn.Parameter(torch.zeros(freq_bins, input_dim))
        self.local_conv = nn.Conv1d(
            input_dim,
            input_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=1,
        )
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = make_activation_layer(activation)

    def forward(
        self,
        history_emb: torch.Tensor,
        history_mask: torch.Tensor,
        history_delta_t: torch.Tensor,
    ) -> torch.Tensor:
        if history_emb.ndim != 3:
            raise ValueError("history_emb must have shape [B, L, D].")
        if history_mask.ndim != 2:
            raise ValueError("history_mask must have shape [B, L].")
        if history_delta_t.ndim != 2:
            raise ValueError("history_delta_t must have shape [B, L].")

        history_time_emb = self.time_encoder(history_delta_t)
        x = history_emb + history_time_emb
        x = x * history_mask.unsqueeze(-1).to(dtype=x.dtype)

        spectrum = torch.fft.rfft(x, n=self.num_neighbors, dim=1, norm='ortho')
        energy = spectrum.abs().pow(2).mean(dim=-1, keepdim=True)
        max_energy = energy.amax(dim=1, keepdim=True).clamp(min=1e-12)
        threshold = self.tau * max_energy
        if self.use_soft_mask:
            mask = torch.sigmoid((energy - threshold) / max_energy)
        else:
            mask = (energy >= threshold).to(dtype=x.dtype)
        spectrum = spectrum * mask

        gain = torch.complex(
            self.freq_gain_real.to(dtype=x.dtype, device=x.device),
            self.freq_gain_imag.to(dtype=x.dtype, device=x.device),
        ).unsqueeze(0)
        filtered = torch.fft.irfft(spectrum * gain, n=self.num_neighbors, dim=1, norm='ortho')

        filtered = filtered * history_mask.unsqueeze(-1).to(dtype=filtered.dtype)
        conv_out = self.local_conv(filtered.transpose(1, 2)).transpose(1, 2)
        conv_out = self.activation(conv_out)
        conv_out = self.dropout(conv_out)
        conv_out = conv_out * history_mask.unsqueeze(-1).to(dtype=conv_out.dtype)

        denom = history_mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=conv_out.dtype)
        pooled = conv_out.sum(dim=1) / denom
        return self.layer_norm(pooled)


class NodeRhythmEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_neighbors: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        time_encoder_type: str = 'mlp',
        time_encoder_fourier_dim: int = 32,
        time_encoder_rbf_dim: int = 32,
        time_encoder_rbf_gamma: float = 16.0,
    ):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        self.num_neighbors = int(num_neighbors)
        self.absolute_time_encoder = RelativeTimeEncoder(
            output_dim=input_dim,
            encoder_type=time_encoder_type,
            fourier_dim=time_encoder_fourier_dim,
            rbf_dim=time_encoder_rbf_dim,
            rbf_gamma=time_encoder_rbf_gamma,
        )
        self.recency_encoder = RelativeTimeEncoder(
            output_dim=input_dim,
            encoder_type=time_encoder_type,
            fourier_dim=time_encoder_fourier_dim,
            rbf_dim=time_encoder_rbf_dim,
            rbf_gamma=time_encoder_rbf_gamma,
        )
        self.depthwise_conv = nn.Conv1d(
            input_dim,
            input_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=input_dim,
        )
        self.mix = nn.Linear(input_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        history_times: torch.Tensor,
        history_mask: torch.Tensor,
        history_delta_t: torch.Tensor,
    ) -> torch.Tensor:
        if history_times.ndim != 2:
            raise ValueError("history_times must have shape [B, L].")
        if history_mask.ndim != 2:
            raise ValueError("history_mask must have shape [B, L].")
        if history_delta_t.ndim != 2:
            raise ValueError("history_delta_t must have shape [B, L].")

        time_emb = self.absolute_time_encoder(history_times)
        time_emb = time_emb * history_mask.unsqueeze(-1).to(dtype=time_emb.dtype)
        rhythm_seq = self.depthwise_conv(time_emb.transpose(1, 2)).transpose(1, 2)
        rhythm_seq = rhythm_seq * history_mask.unsqueeze(-1).to(dtype=rhythm_seq.dtype)
        denom = history_mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=rhythm_seq.dtype)
        rhythm = rhythm_seq.sum(dim=1) / denom

        large_delta = torch.full_like(history_delta_t, float('inf'))
        masked_delta = torch.where(history_mask, history_delta_t, large_delta)
        recency = masked_delta.amin(dim=1, keepdim=True)
        recency = torch.where(torch.isfinite(recency), recency, torch.zeros_like(recency))
        recency_emb = self.recency_encoder(recency).squeeze(1)

        out = self.mix(rhythm + recency_emb)
        out = self.dropout(out)
        return self.layer_norm(out)


class SemanticSeqFilterScorer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_neighbors: int = 32,
        tau: float = 0.2,
        kernel_size: int = 3,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_layernorm: bool = True,
        time_encoder_type: str = 'mlp',
        time_encoder_fourier_dim: int = 32,
        time_encoder_rbf_dim: int = 32,
        time_encoder_rbf_gamma: float = 16.0,
        use_soft_mask: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.num_neighbors = int(num_neighbors)
        self.structure_encoder = FrequencySelectiveStructureEncoder(
            input_dim=input_dim,
            num_neighbors=num_neighbors,
            tau=tau,
            kernel_size=kernel_size,
            dropout=dropout,
            activation=activation,
            time_encoder_type=time_encoder_type,
            time_encoder_fourier_dim=time_encoder_fourier_dim,
            time_encoder_rbf_dim=time_encoder_rbf_dim,
            time_encoder_rbf_gamma=time_encoder_rbf_gamma,
            use_soft_mask=use_soft_mask,
        )
        self.rhythm_encoder = NodeRhythmEncoder(
            input_dim=input_dim,
            num_neighbors=num_neighbors,
            kernel_size=kernel_size,
            dropout=dropout,
            time_encoder_type=time_encoder_type,
            time_encoder_fourier_dim=time_encoder_fourier_dim,
            time_encoder_rbf_dim=time_encoder_rbf_dim,
            time_encoder_rbf_gamma=time_encoder_rbf_gamma,
        )
        self.node_layer_norm = nn.LayerNorm(input_dim)

        layers = []
        in_dim = input_dim * 4
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(make_activation_layer(activation))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*layers)

    @property
    def requires_pair_neighbor_context(self) -> bool:
        return True

    def encode_node(
        self,
        node_emb: torch.Tensor,
        history_emb: torch.Tensor,
        history_mask: torch.Tensor,
        history_delta_t: torch.Tensor,
        history_times: torch.Tensor,
    ) -> torch.Tensor:
        structure = self.structure_encoder(
            history_emb=history_emb,
            history_mask=history_mask,
            history_delta_t=history_delta_t,
        )
        rhythm = self.rhythm_encoder(
            history_times=history_times,
            history_mask=history_mask,
            history_delta_t=history_delta_t,
        )
        has_history = history_mask.any(dim=1, keepdim=True)
        fused = torch.tanh(node_emb + structure + rhythm)
        fused = self.node_layer_norm(fused)
        return torch.where(has_history, fused, node_emb)

    def forward(
        self,
        src_emb: torch.Tensor,
        dst_emb: torch.Tensor,
        src_history_emb: torch.Tensor,
        src_history_mask: torch.Tensor,
        src_history_delta_t: torch.Tensor,
        src_history_times: torch.Tensor,
        dst_history_emb: torch.Tensor,
        dst_history_mask: torch.Tensor,
        dst_history_delta_t: torch.Tensor,
        dst_history_times: torch.Tensor,
    ) -> torch.Tensor:
        z_src = self.encode_node(
            node_emb=src_emb,
            history_emb=src_history_emb,
            history_mask=src_history_mask,
            history_delta_t=src_history_delta_t,
            history_times=src_history_times,
        )
        z_dst = self.encode_node(
            node_emb=dst_emb,
            history_emb=dst_history_emb,
            history_mask=dst_history_mask,
            history_delta_t=dst_history_delta_t,
            history_times=dst_history_times,
        )
        pair = torch.cat([z_src, z_dst, z_src * z_dst, torch.abs(z_src - z_dst)], dim=1)
        return self.head(pair).squeeze(-1)


class SemanticDyGFormerLiteScorer(nn.Module):
    """
    DyGFormer-style lightweight pair scorer without co-occurrence features.
    Tokens are [src, dst, recent src neighbors..., recent dst neighbors...].
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        head_num_layers: int = 2,
        num_neighbors: int = 20,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_layernorm: bool = True,
        add_time: bool = True,
        time_encoder_type: str = 'mlp',
        time_encoder_fourier_dim: int = 32,
        time_encoder_rbf_dim: int = 32,
        time_encoder_rbf_gamma: float = 16.0,
        time_encoder_mask_padding: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if head_num_layers < 1:
            raise ValueError("head_num_layers must be >= 1")
        if num_neighbors <= 0:
            raise ValueError("num_neighbors must be > 0.")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0.")
        if input_dim % num_heads != 0:
            raise ValueError(f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads}).")

        self.num_neighbors = int(num_neighbors)
        self.add_time = bool(add_time)
        self.time_encoder_mask_padding = bool(time_encoder_mask_padding)
        self.type_embedding = nn.Embedding(4, input_dim)
        nn.init.normal_(self.type_embedding.weight, mean=0.0, std=0.02)
        self.time_encoder = RelativeTimeEncoder(
            output_dim=input_dim,
            encoder_type=time_encoder_type,
            fourier_dim=time_encoder_fourier_dim,
            rbf_dim=time_encoder_rbf_dim,
            rbf_gamma=time_encoder_rbf_gamma,
        )
        encoder_activation = 'gelu' if activation == 'silu' else activation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=dropout,
            activation=encoder_activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_norm = nn.LayerNorm(input_dim) if use_layernorm else nn.Identity()
        self.input_dropout = nn.Dropout(dropout)

        head_layers = []
        in_dim = input_dim * 4
        for _ in range(head_num_layers - 1):
            head_layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layernorm:
                head_layers.append(nn.LayerNorm(hidden_dim))
            head_layers.append(make_activation_layer(activation))
            head_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        head_layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*head_layers)

    @property
    def requires_pair_neighbor_context(self) -> bool:
        return True

    def forward(
        self,
        src_emb: torch.Tensor,
        dst_emb: torch.Tensor,
        src_history_emb: torch.Tensor,
        src_history_mask: torch.Tensor,
        src_history_delta_t: torch.Tensor,
        src_history_times: torch.Tensor,
        dst_history_emb: torch.Tensor,
        dst_history_mask: torch.Tensor,
        dst_history_delta_t: torch.Tensor,
        dst_history_times: torch.Tensor,
    ) -> torch.Tensor:
        if src_emb.ndim != 2 or dst_emb.ndim != 2:
            raise ValueError("src_emb and dst_emb must have shape [B, D].")
        if src_history_emb.ndim != 3 or dst_history_emb.ndim != 3:
            raise ValueError("history embeddings must have shape [B, L, D].")
        if src_history_mask.ndim != 2 or dst_history_mask.ndim != 2:
            raise ValueError("history masks must have shape [B, L].")

        batch_size = src_emb.shape[0]
        device = src_emb.device
        pair_tokens = torch.stack([src_emb, dst_emb], dim=1)
        tokens = torch.cat([pair_tokens, src_history_emb, dst_history_emb], dim=1)

        pair_mask = torch.ones((batch_size, 2), dtype=torch.bool, device=device)
        token_mask = torch.cat([pair_mask, src_history_mask, dst_history_mask], dim=1)

        src_type = torch.full_like(src_history_mask, 2, dtype=torch.long, device=device)
        dst_type = torch.full_like(dst_history_mask, 3, dtype=torch.long, device=device)
        pair_type = torch.tensor([0, 1], dtype=torch.long, device=device).view(1, 2).expand(batch_size, 2)
        type_ids = torch.cat([pair_type, src_type, dst_type], dim=1)
        tokens = tokens + self.type_embedding(type_ids)

        if self.add_time:
            pair_delta_t = torch.zeros((batch_size, 2), dtype=src_history_delta_t.dtype, device=device)
            delta_t = torch.cat([pair_delta_t, src_history_delta_t, dst_history_delta_t], dim=1)
            time_emb = self.time_encoder(delta_t)
            if self.time_encoder_mask_padding:
                time_emb = time_emb * token_mask.unsqueeze(-1).to(dtype=time_emb.dtype)
            tokens = tokens + time_emb

        tokens = self.input_norm(tokens)
        tokens = self.input_dropout(tokens)
        encoded = self.transformer(tokens, src_key_padding_mask=~token_mask)

        z_src = encoded[:, 0, :]
        z_dst = encoded[:, 1, :]
        pair = torch.cat([z_src, z_dst, z_src * z_dst, torch.abs(z_src - z_dst)], dim=1)
        return self.head(pair).squeeze(-1)


class MultiHeadCrossAttentionByHand(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        layer_norm_eps: float,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads}).")

        self.num_attention_heads = int(num_heads)
        self.attention_head_size = int(hidden_size // num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(float(self.attention_head_size))

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_x_shape)

    def forward(self, query: torch.Tensor, attention_mask: torch.Tensor, key: Optional[torch.Tensor] = None) -> torch.Tensor:
        if key is None:
            key = query

        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(key)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        query_layer = query_layer.permute(0, 2, 1, 3)
        key_layer = key_layer.permute(0, 2, 3, 1)
        value_layer = value_layer.permute(0, 2, 1, 3)

        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores = attention_scores + attention_mask

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + query
        return hidden_states


class FeedForwardCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        inner_size: int,
        hidden_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.hidden_act = hidden_act
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def apply_act(self, x: torch.Tensor) -> torch.Tensor:
        if self.hidden_act == 'gelu':
            return F.gelu(x)
        if self.hidden_act == 'relu':
            return F.relu(x)
        if self.hidden_act == 'swish':
            return x * torch.sigmoid(x)
        if self.hidden_act == 'tanh':
            return torch.tanh(x)
        if self.hidden_act == 'sigmoid':
            return torch.sigmoid(x)
        raise ValueError(f"Unsupported cross-attention activation: {self.hidden_act}")

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.apply_act(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
        intermediate_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.multi_head_attention = MultiHeadCrossAttentionByHand(
            num_heads=num_heads,
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            layer_norm_eps=layer_norm_eps,
        )
        self.feed_forward = FeedForwardCrossAttention(
            hidden_size=hidden_size,
            inner_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
        )

    def forward(self, query: torch.Tensor, attention_mask: torch.Tensor, key: Optional[torch.Tensor] = None) -> torch.Tensor:
        attention_output = self.multi_head_attention(query=query, attention_mask=attention_mask, key=key)
        return self.feed_forward(attention_output)


class CrossAttention(nn.Module):
    def __init__(
        self,
        num_layers: int = 2,
        num_heads: int = 2,
        hidden_size: int = 64,
        inner_size: int = 256,
        hidden_dropout_prob: float = 0.5,
        attn_dropout_prob: float = 0.5,
        hidden_act: str = 'gelu',
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(
                num_heads=num_heads,
                hidden_size=hidden_size,
                intermediate_size=inner_size,
                hidden_dropout_prob=hidden_dropout_prob,
                attn_dropout_prob=attn_dropout_prob,
                hidden_act=hidden_act,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        query: torch.Tensor,
        attention_mask: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        output_all_encoded_layers: bool = True,
    ):
        all_encoder_layers = []
        for layer_module in self.layers:
            query = layer_module(query=query, attention_mask=attention_mask, key=key)
            if output_all_encoded_layers:
                all_encoder_layers.append(query)
        if not output_all_encoded_layers:
            all_encoder_layers.append(query)
        return all_encoder_layers


class RelativeTimeEncoder(nn.Module):
    """
    Encode non-negative relative times (delta_t) into hidden vectors.
    Supported styles:
    - mlp: linear projection over log1p(delta_t), CRAFT-like lightweight mapping
    - fourier: fixed sinusoidal basis + linear projection
    - rbf: Gaussian RBF basis + linear projection
    """

    def __init__(
        self,
        output_dim: int,
        encoder_type: str = 'mlp',
        fourier_dim: int = 32,
        rbf_dim: int = 32,
        rbf_gamma: float = 16.0,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.encoder_type = str(encoder_type)

        if self.encoder_type == 'mlp':
            self.proj = nn.Linear(1, self.output_dim)
        elif self.encoder_type == 'fourier':
            num_freq = max(1, int(fourier_dim // 2))
            # Geometric frequencies over log-time, similar spirit to transformer PE.
            freq = torch.exp(torch.linspace(0.0, math.log(10000.0), steps=num_freq))
            self.register_buffer('fourier_freq', freq)
            self.proj = nn.Linear(num_freq * 2, self.output_dim)
        elif self.encoder_type == 'rbf':
            num_centers = max(2, int(rbf_dim))
            centers = torch.linspace(0.0, 1.0, steps=num_centers)
            self.register_buffer('rbf_centers', centers)
            self.rbf_gamma = float(rbf_gamma)
            self.proj = nn.Linear(num_centers, self.output_dim)
        else:
            raise ValueError(f"Unsupported time encoder type: {self.encoder_type}")

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        if delta_t.ndim != 2:
            raise ValueError("delta_t must have shape [B, L].")
        x = torch.log1p(torch.clamp(delta_t, min=0.0)).unsqueeze(-1)  # [B, L, 1]

        if self.encoder_type == 'mlp':
            return self.proj(x)

        if self.encoder_type == 'fourier':
            freq = self.fourier_freq.to(dtype=x.dtype, device=x.device).view(1, 1, -1)
            angles = x * freq
            basis = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
            return self.proj(basis)

        x_norm = x / (1.0 + x)
        centers = self.rbf_centers.to(dtype=x.dtype, device=x.device).view(1, 1, -1)
        basis = torch.exp(-self.rbf_gamma * (x_norm - centers).pow(2))
        return self.proj(basis)


class SemanticCrossAttention(nn.Module):
    """
    CRAFT-style cross-attention scorer:
    - query: destination candidate embedding
    - key/value: source historical neighbor sequence embeddings
    - strict time cutoff is handled outside this module when constructing source history
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int = 2,
        num_heads: int = 2,
        hidden_dropout_prob: float = 0.5,
        attn_dropout_prob: float = 0.5,
        emb_dropout_prob: float = 0.1,
        activation: str = 'gelu',
        layer_norm_eps: float = 1e-12,
        use_pos: bool = False,
        max_seq_length: int = 512,
        add_time_to_history: bool = True,
        time_encoder_type: str = 'mlp',
        time_encoder_fourier_dim: int = 32,
        time_encoder_rbf_dim: int = 32,
        time_encoder_rbf_gamma: float = 16.0,
        time_encoder_mask_padding: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if num_heads < 1:
            raise ValueError("num_heads must be >= 1")
        if input_dim % num_heads != 0:
            raise ValueError(f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads}).")

        hidden_act = 'swish' if activation == 'silu' else activation
        self.cross_attention = CrossAttention(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=input_dim,
            inner_size=input_dim * 4,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
        )
        self.use_pos = bool(use_pos)
        self.max_seq_length = int(max_seq_length)
        self.add_time_to_history = bool(add_time_to_history)
        if self.use_pos:
            if self.max_seq_length <= 0:
                raise ValueError("max_seq_length must be > 0 when use_pos is enabled.")
            self.position_embedding = nn.Embedding(self.max_seq_length, input_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.history_layer_norm = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        self.query_layer_norm = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        self.input_dropout = nn.Dropout(emb_dropout_prob)
        self.time_encoder = RelativeTimeEncoder(
            output_dim=input_dim,
            encoder_type=time_encoder_type,
            fourier_dim=time_encoder_fourier_dim,
            rbf_dim=time_encoder_rbf_dim,
            rbf_gamma=time_encoder_rbf_gamma,
        )
        self.time_encoder_mask_padding = bool(time_encoder_mask_padding)
        self.output_layer = nn.Linear(input_dim, 1)

    @property
    def requires_neighbor_context(self) -> bool:
        return True

    @staticmethod
    def build_attention_mask(query_mask: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        # Boolean outer-product mask with shape [B, 1, Q, K].
        mask = (query_mask.unsqueeze(2) & key_mask.unsqueeze(1)).unsqueeze(1)
        attention_mask = torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)
        attention_mask = attention_mask.masked_fill(~mask, -10000.0)
        return attention_mask

    def forward(
        self,
        dst_emb: torch.Tensor,
        src_history_emb: torch.Tensor,
        src_history_mask: torch.Tensor,
        src_history_delta_t: torch.Tensor,
    ) -> torch.Tensor:
        if dst_emb.ndim != 2:
            raise ValueError("dst_emb must have shape [B, D].")
        if src_history_emb.ndim != 3:
            raise ValueError("src_history_emb must have shape [B, L, D].")
        if src_history_mask.ndim != 2:
            raise ValueError("src_history_mask must have shape [B, L].")
        if src_history_delta_t.ndim != 2:
            raise ValueError("src_history_delta_t must have shape [B, L].")

        if self.use_pos:
            seq_len = int(src_history_emb.shape[1])
            if seq_len > self.max_seq_length:
                raise ValueError(
                    f"src_history length ({seq_len}) exceeds max_seq_length ({self.max_seq_length})."
                )
            pos_ids = torch.arange(seq_len, dtype=torch.long, device=src_history_emb.device).unsqueeze(0)
            pos_emb = self.position_embedding(pos_ids)
            pos_emb = pos_emb * src_history_mask.unsqueeze(-1).to(pos_emb.dtype)
            src_history_emb = src_history_emb + pos_emb

        if self.add_time_to_history:
            history_time_emb = self.time_encoder(src_history_delta_t)
            if self.time_encoder_mask_padding:
                history_time_emb = history_time_emb * src_history_mask.unsqueeze(-1).to(history_time_emb.dtype)
            src_history_emb = src_history_emb + history_time_emb
        src_history_emb = self.history_layer_norm(src_history_emb)
        src_history_emb = self.input_dropout(src_history_emb)

        query = self.query_layer_norm(dst_emb)
        query = self.input_dropout(query).unsqueeze(1)  # [B, 1, D]
        query_mask = torch.ones((dst_emb.shape[0], 1), dtype=torch.bool, device=dst_emb.device)
        attention_mask = self.build_attention_mask(query_mask=query_mask, key_mask=src_history_mask)
        output = self.cross_attention(
            query=query,
            attention_mask=attention_mask,
            key=src_history_emb,
            output_all_encoded_layers=True,
        )[-1]
        has_history = src_history_mask.any(dim=1, keepdim=True).unsqueeze(-1)
        output = torch.where(has_history, output, query)
        logits = self.output_layer(output).squeeze(-1).squeeze(-1)
        return logits


class TemporalSelfAttentionPooling(nn.Module):
    """
    Lightweight attention-pooling message passing:
    - query: current node embedding
    - key/value: recent interaction-neighbor sequence embeddings of the same node
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = 'gelu',
        residual: bool = True,
        layer_norm_eps: float = 1e-12,
        time_encoder_type: str = 'mlp',
        time_encoder_fourier_dim: int = 32,
        time_encoder_rbf_dim: int = 32,
        time_encoder_rbf_gamma: float = 16.0,
        time_encoder_mask_padding: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if num_heads < 1:
            raise ValueError("num_heads must be >= 1")
        if input_dim % num_heads != 0:
            raise ValueError(f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads}).")

        hidden_act = 'swish' if activation == 'silu' else activation
        self.cross_attention = CrossAttention(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=input_dim,
            inner_size=input_dim * 4,
            hidden_dropout_prob=dropout,
            attn_dropout_prob=dropout,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
        )
        self.time_encoder = RelativeTimeEncoder(
            output_dim=input_dim,
            encoder_type=time_encoder_type,
            fourier_dim=time_encoder_fourier_dim,
            rbf_dim=time_encoder_rbf_dim,
            rbf_gamma=time_encoder_rbf_gamma,
        )
        self.time_encoder_mask_padding = bool(time_encoder_mask_padding)
        self.residual = bool(residual)

    def forward(
        self,
        node_emb: torch.Tensor,
        history_emb: torch.Tensor,
        history_mask: torch.Tensor,
        history_delta_t: torch.Tensor,
    ) -> torch.Tensor:
        if node_emb.ndim != 2:
            raise ValueError("node_emb must have shape [B, D].")
        if history_emb.ndim != 3:
            raise ValueError("history_emb must have shape [B, L, D].")
        if history_mask.ndim != 2:
            raise ValueError("history_mask must have shape [B, L].")
        if history_delta_t.ndim != 2:
            raise ValueError("history_delta_t must have shape [B, L].")

        history_time_emb = self.time_encoder(history_delta_t)
        if self.time_encoder_mask_padding:
            history_time_emb = history_time_emb * history_mask.unsqueeze(-1).to(history_time_emb.dtype)
        history_emb = history_emb + history_time_emb

        query = node_emb.unsqueeze(1)  # [B, 1, D]
        query_mask = torch.ones((node_emb.shape[0], 1), dtype=torch.bool, device=node_emb.device)
        attention_mask = SemanticCrossAttention.build_attention_mask(query_mask=query_mask, key_mask=history_mask)
        pooled = self.cross_attention(
            query=query,
            attention_mask=attention_mask,
            key=history_emb,
            output_all_encoded_layers=True,
        )[-1].squeeze(1)
        if self.residual:
            pooled = pooled + node_emb
        has_history = history_mask.any(dim=1)
        pooled = torch.where(has_history.unsqueeze(1), pooled, node_emb)
        return pooled


class TemporalNeighborIndex:
    """
    Query-time temporal neighbor index:
    for each node u and query time t, returns recent neighbors with interactions strictly before t.
    """

    def __init__(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        max_node_id: int,
        undirected: bool = True,
    ):
        self.max_node_id = int(max_node_id)
        buckets_neighbors = [[] for _ in range(self.max_node_id + 1)]
        buckets_times = [[] for _ in range(self.max_node_id + 1)]

        for src, dst, ts in zip(src_node_ids, dst_node_ids, node_interact_times):
            src_i = int(src)
            dst_i = int(dst)
            t_f = float(ts)
            if 0 <= src_i <= self.max_node_id:
                buckets_neighbors[src_i].append(dst_i)
                buckets_times[src_i].append(t_f)
            if undirected and 0 <= dst_i <= self.max_node_id:
                buckets_neighbors[dst_i].append(src_i)
                buckets_times[dst_i].append(t_f)

        self.node_neighbors = []
        self.node_times = []
        for node_id in range(self.max_node_id + 1):
            if len(buckets_times[node_id]) == 0:
                self.node_neighbors.append(np.empty((0,), dtype=np.int64))
                self.node_times.append(np.empty((0,), dtype=np.float64))
                continue
            node_times = np.asarray(buckets_times[node_id], dtype=np.float64)
            sort_idx = np.argsort(node_times, kind='stable')
            self.node_times.append(node_times[sort_idx])
            self.node_neighbors.append(np.asarray(buckets_neighbors[node_id], dtype=np.int64)[sort_idx])

    def get_recent_neighbors(
        self,
        node_ids: np.ndarray,
        query_times: np.ndarray,
        num_neighbors: int,
        return_times: bool = False,
    ):
        if num_neighbors <= 0:
            raise ValueError("num_neighbors must be > 0.")

        node_ids = np.asarray(node_ids, dtype=np.int64)
        query_times = np.asarray(query_times, dtype=np.float64)
        if node_ids.shape[0] != query_times.shape[0]:
            raise ValueError("node_ids and query_times must have the same length.")

        neighbors = np.full((len(node_ids), num_neighbors), -1, dtype=np.int64)
        neighbor_times = np.zeros((len(node_ids), num_neighbors), dtype=np.float64)
        valid_mask = np.zeros((len(node_ids), num_neighbors), dtype=bool)

        for i, (node_id, query_time) in enumerate(zip(node_ids, query_times)):
            if node_id < 0 or node_id > self.max_node_id:
                continue
            times = self.node_times[node_id]
            if times.shape[0] == 0:
                continue
            end = int(np.searchsorted(times, float(query_time), side='left'))  # strict cutoff: t' < t
            if end <= 0:
                continue
            start = max(0, end - num_neighbors)
            chosen = self.node_neighbors[node_id][start:end]
            chosen_times = self.node_times[node_id][start:end]
            take = int(chosen.shape[0])
            neighbors[i, :take] = chosen
            neighbor_times[i, :take] = chosen_times
            valid_mask[i, :take] = True

        if return_times:
            return neighbors, neighbor_times, valid_mask
        return neighbors, valid_mask


def make_subset(data, mask: np.ndarray):
    subset = SimpleNamespace(
        src_node_ids=data.src_node_ids[mask],
        dst_node_ids=data.dst_node_ids[mask],
        node_interact_times=data.node_interact_times[mask],
        num_interactions=int(mask.sum()),
    )
    for name in ('edge_ids', 'labels'):
        if hasattr(data, name):
            setattr(subset, name, getattr(data, name)[mask])
    return subset


def build_lookup_tensor(entity_ids_sorted, max_node_id: int, device: torch.device) -> torch.Tensor:
    lookup = torch.full((max_node_id + 1,), -1, dtype=torch.long, device=device)
    entity_ids_np = np.asarray(entity_ids_sorted, dtype=np.int64)
    valid_mask = entity_ids_np <= max_node_id

    valid_node_ids = torch.from_numpy(entity_ids_np[valid_mask]).to(device=device, dtype=torch.long)
    valid_emb_rows = torch.from_numpy(np.where(valid_mask)[0]).to(device=device, dtype=torch.long)
    lookup[valid_node_ids] = valid_emb_rows
    return lookup


class LearnableEntityEmbeddingTable(nn.Module):
    def __init__(
        self,
        num_entities: int,
        embedding_dim: int,
        init_std: float = 0.02,
        init_mode: str = "normal",
        freeze: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(int(num_entities), int(embedding_dim))
        init_mode = str(init_mode).strip().lower()
        if init_mode == "normal":
            nn.init.normal_(self.embedding.weight, mean=0.0, std=float(init_std))
        elif init_mode == "orthogonal":
            nn.init.orthogonal_(self.embedding.weight)
        elif init_mode == "zero":
            nn.init.zeros_(self.embedding.weight)
        else:
            raise ValueError(f"Unsupported learnable entity embedding init_mode: {init_mode}")
        if freeze:
            self.embedding.weight.requires_grad_(False)

    def forward(self) -> torch.Tensor:
        return F.normalize(self.embedding.weight, dim=1)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
):
    if scheduler_type == 'none':
        return None

    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(step: int) -> float:
        step = int(step)

        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        denom = max(1, total_steps - warmup_steps)
        progress = float(step - warmup_steps) / float(denom)
        progress = min(max(progress, 0.0), 1.0)

        if scheduler_type == 'cosine':
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
        if scheduler_type == 'linear':
            return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)

        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def maybe_filter_train_edges(train_data, cutoff_time: Optional[float], cutoff_ratio: float):
    n_total = len(train_data.src_node_ids)
    mask = np.ones(n_total, dtype=bool)

    if cutoff_time is not None:
        # Recency cutoff: drop old edges before cutoff_time.
        mask &= train_data.node_interact_times >= float(cutoff_time)

    if cutoff_ratio < 1.0:
        keep_n = max(1, int(n_total * cutoff_ratio))
        ratio_mask = np.zeros(n_total, dtype=bool)
        # Keep the most recent portion of train edges.
        ratio_mask[-keep_n:] = True
        mask &= ratio_mask

    if int(mask.sum()) == 0:
        raise ValueError("Training edge cutoff removed all train edges. Relax cutoff settings.")

    return make_subset(train_data, mask), int(mask.sum()), n_total

__all__ = [
    "CrossAttention",
    "CrossAttentionLayer",
    "FeedForwardCrossAttention",
    "LearnableEntityEmbeddingTable",
    "MultiHeadCrossAttentionByHand",
    "RelativeTimeEncoder",
    "SemanticCrossAttention",
    "SemanticDyGFormerLiteScorer",
    "SemanticMLP",
    "SemanticNCNScorer",
    "SemanticSeqFilterScorer",
    "TemporalNeighborIndex",
    "TemporalSelfAttentionPooling",
    "build_lookup_tensor",
    "build_lr_scheduler",
    "make_activation_layer",
    "make_subset",
    "maybe_filter_train_edges",
    "str2bool",
]
