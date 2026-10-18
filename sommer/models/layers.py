"""Reusable building blocks for models (formerly baukasten.py)."""
from __future__ import annotations

import torch
import torch.nn as nn

from sommer.constants import MAX_PEPTIDE_LEN


class PositionalEncoding(nn.Module):
    """learned absolute positions; without it ACAB and AABC would look the same"""

    def __init__(self, hidden_dimension: int, max_peptide_length: int = MAX_PEPTIDE_LEN):
        super().__init__()
        self.position_embedding = nn.Embedding(max_peptide_length, hidden_dimension)

    def forward(self, residue_features: torch.Tensor) -> torch.Tensor:
        position_idxs = torch.arange(residue_features.size(1), device=residue_features.device)
        return residue_features + self.position_embedding(position_idxs).unsqueeze(0)


class TransformerEncoder(nn.Module):
    """pre-norm transformer encoder as in Attention is All you Need; optionally causal"""

    def __init__(self, hidden_dimension: int, layer_count: int, attention_head_count: int,
                 bidirectional: bool = True, dropout: float = 0.1):
        super().__init__()
        assert hidden_dimension % attention_head_count == 0  # clean division to heads
        self.bidirectional = bidirectional
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dimension,
            nhead=attention_head_count,
            dim_feedforward=4 * hidden_dimension,
            dropout=dropout,
            activation="gelu",  # relu might be faster but this is default for most modern transformers
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(transformer_layer, num_layers=layer_count,
                                             norm=nn.LayerNorm(hidden_dimension), enable_nested_tensor=False)

    def forward(self, residue_features: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        attention_mask = None
        if not self.bidirectional:
            # mask out attention to later positions (upper triangle)
            n = residue_features.size(1)
            attention_mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=residue_features.device), diagonal=1)
        return self.encoder(residue_features, mask=attention_mask, src_key_padding_mask=padding_mask)


def masked_mean_pooling(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """global representation without letting padding distort it"""
    weights = mask.unsqueeze(-1).to(dtype=features.dtype)
    return (features * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)


def place_bonds(bond_out: torch.Tensor, n_res: torch.Tensor) -> torch.Tensor:
    """per-bond (b, y) predictions -> 58-slot vector

    bond_out (B, L-1, 2): bond i (between residue i and i+1) yields b_{i+1} and y_{L-1-i}.
    Bonds beyond the peptide length are zeroed.
    """
    n = torch.arange(1, bond_out.size(1) + 1, device=bond_out.device).unsqueeze(0)
    n_res = n_res.unsqueeze(1)
    valid = n < n_res
    b = bond_out[..., 0]
    y = bond_out[..., 1].gather(1, (n_res - 1 - n).clamp(min=0))
    return (torch.stack([b, y], dim=-1) * valid.unsqueeze(-1)).flatten(1)
