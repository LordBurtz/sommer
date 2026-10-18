"""Transformer encoder with a per-cleavage-site head (formerly refac/models.py)."""
from __future__ import annotations

import torch
import torch.nn as nn

from sommer.constants import CHARGES, PAD_IDX, TOKEN_TO_IDX
from sommer.models.base import FragmentModel
from sommer.models.layers import PositionalEncoding, TransformerEncoder, masked_mean_pooling, place_bonds
from sommer.models.registry import register_model

MOD_C, MOD_M = TOKEN_TO_IDX["C[UNIMOD:4]"], TOKEN_TO_IDX["M[UNIMOD:35]"]
C, M = TOKEN_TO_IDX["C"], TOKEN_TO_IDX["M"]


@register_model
class PeptideEncoder(FragmentModel):
    """
    Modified residues share the embedding of their base amino acid plus a modification embedding.
    Each cleavage site sees its two flanking residues and a global (pooled + charge) peptide vector.
    """

    def __init__(self, aa_vocab_size: int = 21, mod_vocab_size: int = 2, precursor_vocab_size: int = len(CHARGES),
                 hidden_dimensions: int = 256, transformer_layers: int = 6, attention_heads: int = 8):
        super().__init__()
        self.amino_acid_embedding = nn.Embedding(aa_vocab_size, hidden_dimensions)
        self.modification_embedding = nn.Embedding(mod_vocab_size, hidden_dimensions)
        self.precursor_charge_embedding = nn.Embedding(precursor_vocab_size, hidden_dimensions)
        self.positional_encoding = PositionalEncoding(hidden_dimensions)
        self.sequence_transformer = TransformerEncoder(
            hidden_dimension=hidden_dimensions,
            layer_count=transformer_layers,
            attention_head_count=attention_heads,
            bidirectional=True,
        )
        self.global_peptide_projection = nn.Sequential(
            nn.Linear(hidden_dimensions, hidden_dimensions),
            nn.GELU(),
            nn.Linear(hidden_dimensions, hidden_dimensions),
        )
        self.cleavage_site_projection = nn.Linear(3 * hidden_dimensions, hidden_dimensions)
        self.fragment_head = nn.Sequential(
            nn.Linear(hidden_dimensions, hidden_dimensions),
            nn.GELU(),
            nn.Linear(hidden_dimensions, 2),
            nn.Softplus(),
        )

    def forward(self, tokens: torch.Tensor, precursor_charge: torch.Tensor) -> torch.Tensor:
        residue_mask = tokens != PAD_IDX

        base_residue = torch.where(tokens == MOD_C, C, torch.where(tokens == MOD_M, M, tokens))
        is_modified = ((tokens == MOD_C) | (tokens == MOD_M)).long()
        residue_features = self.amino_acid_embedding(base_residue) + self.modification_embedding(is_modified)
        residue_features = self.positional_encoding(residue_features)
        residue_features = self.sequence_transformer(residue_features, padding_mask=~residue_mask)

        global_features = self.global_peptide_projection(
            masked_mean_pooling(residue_features, residue_mask)
            + self.precursor_charge_embedding(self.charge_index(precursor_charge))
        )
        cleavage_features = self.cleavage_site_projection(torch.cat([
            residue_features[:, :-1],
            residue_features[:, 1:],
            global_features.unsqueeze(1).expand(-1, residue_features.size(1) - 1, -1),
        ], dim=-1))
        return place_bonds(self.fragment_head(cleavage_features), residue_mask.sum(dim=1))
