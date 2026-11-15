"""Bidirectional GRU baseline in the spirit of Prosit (formerly in train_deep.py)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sommer.constants import N_ION_TYPES, PAD_IDX, VECTOR_DIM, VOCAB_SIZE
from sommer.models.base import FragmentModel
from sommer.models.layers import place_bonds
from sommer.models.registry import register_model

ACTIVATIONS = {"softplus": F.softplus, "sigmoid": torch.sigmoid}


@register_model
class PrositLike(FragmentModel):
    """
    head="pooled": attention pooling over residues -> all 58 slots at once
    head="bond":   one (b, y) pair per peptide bond, placed with place_bonds
    """

    def __init__(self, emb: int = 64, hidden: int = 128, head: str = "pooled", activation: str = "softplus",
                 dropout: float = 0.2, n_charge: int = 6):
        super().__init__()
        if head not in ("pooled", "bond"):
            raise ValueError(f"unknown head {head!r}, expected 'pooled' or 'bond'")
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}, expected one of {sorted(ACTIVATIONS)}")
        self.head_type = head
        self.final_act = ACTIVATIONS[activation]
        self.tok_emb = nn.Embedding(VOCAB_SIZE, emb, padding_idx=PAD_IDX)
        # indexed by the raw charge (kept for compatibility with legacy checkpoints)
        self.charge_emb = nn.Embedding(n_charge + 1, 16)
        self.gru = nn.GRU(emb + 16, hidden, batch_first=True, bidirectional=True, num_layers=2, dropout=dropout)
        if head == "pooled":
            self.attn = nn.Linear(2 * hidden, 1)
            self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(hidden, VECTOR_DIM))
        else:
            self.bond_head = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                           nn.Linear(hidden, N_ION_TYPES))

    def forward(self, tokens: torch.Tensor, precursor_charge: torch.Tensor) -> torch.Tensor:
        residue_mask = tokens != PAD_IDX                                  # (B, L)
        x = self.tok_emb(tokens)                                          # (B, L, emb)
        cz = self.charge_emb(precursor_charge).unsqueeze(1).expand(-1, x.size(1), -1)
        h, _ = self.gru(torch.cat([x, cz], dim=-1))                       # (B, L, 2H)
        if self.head_type == "bond":
            bonds = torch.cat([h[:, :-1], h[:, 1:]], dim=-1)
            return place_bonds(self.final_act(self.bond_head(bonds)), residue_mask.sum(dim=1))
        # attention pooling over valid (non-pad) positions
        score = self.attn(h).masked_fill(~residue_mask.unsqueeze(-1), -1e9)
        pooled = (torch.softmax(score, dim=1) * h).sum(dim=1)             # (B, 2H)
        return self.final_act(self.head(pooled))
