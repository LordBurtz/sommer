import torch
import torch.nn as nn

from baukasten import (
    PositionalEncoding,
    TransformerEncoder,
    masked_mean_pooling
)

"""
Model 1: some description
"""

class PeptideEncoder(nn.Module):

    # Init with default config
    def __init__(
            self,
            aa_vocab_size: int,
            mod_vocab_size: int,
            precursor_vocab_size: int,
            hidden_dimensions: int = 256,
            transformer_layers: int = 6,
            attention_heads: int = 8
    ):
        super().__init__()

        # Create emneddings for amino acids, modifications, precursor charges
        self.amino_acid_embedding = nn.Embedding(
            aa_vocab_size, hidden_dimensions
        )

        self.modification_emnedding = nn.Embedding(
            mod_vocab_size, hidden_dimensions
        )

        self.precursor_charge_emnedding = nn.Embedding(
            precursor_vocab_size, hidden_dimensions
        )

        self.positional_encoding = PositionalEncoding(hidden_dimensions)

        self.sequence_transformer = TransformerEncoder(
            hidden_dimension = hidden_dimensions,
            layer_count = transformer_layers,
            attention_head_count = attention_heads,
            bidirectional = True
        )

        self.global_peptide_projection = nn.Sequential(
            nn.Linear(hidden_dimensions, hidden_dimensions),
            nn.GELU(),
            nn.Linear(hidden_dimensions, hidden_dimensions)
        )

        self.cleavage_site_projection = nn.Linear(
            3 * hidden_dimensions,
            hidden_dimensions,
        )

        self.hidden_dimension = hidden_dimensions

    def forward(
            self,
            amino_acid_ids,
            modification_ids,
            precursor_charge_ids,
            residue_mask,
    ):
        # Feature emnedding
        residue_features = self.amino_acid_embedding(amino_acid_ids)
        modification_features = self.modification_emnedding(modification_ids)
        precursor_charge_features = self.precursor_charge_emnedding(precursor_charge_ids)

        # Embedd features together into one
        residue_features = residue_features + modification_features
        residue_features = self.positional_encoding(residue_features)
        masked_residual_features = self.sequence_transformer(
            residue_features,
            padding_mask = ~residue_mask
        )

        # Pool together with precursor charge

        global_peptide_features = masked_mean_pooling(
            masked_residual_features,
            residue_mask,
        )

        global_peptide_features = self.global_peptide_projection(
            global_peptide_features
        )

        # Fridolin Karger: recheck this, its 4am lol

        left_residue_features = masked_residual_features[:, :-1, :]
        right_residue_features = masked_residual_features[:, 1:, :]

        repeated_global_features = global_peptide_features.unsqueeze(1).expand(
            -1,
            left_residue_features.size(1),
            -1,
        )

        cleavage_features = torch.cat(
            [
                left_residue_features,
                right_residue_features,
                repeated_global_features,
            ],
            dim=-1,
        )

        cleavage_features = self.cleavage_site_projection(cleavage_features)

        return {
            "residue_features": masked_residual_features,
            "global_peptide_features": global_peptide_features,
            "cleavage_features": cleavage_features,
        }

model = PeptideEncoder(
    aa_vocab_size=21,
    mod_vocab_size=2,
    precursor_vocab_size=4,
)

model.eval()

amino_acid_ids = torch.tensor([
    [1, 2, 3, 4],  # Example peptide: AGKP
    [1, 2, 3, 0],  # Example peptide: AGK
])

precursor_charge_ids = torch.tensor([2, 3])
modification_ids = torch.zeros_like(amino_acid_ids)

residue_mask = amino_acid_ids != 0

with torch.no_grad():
    output = model(
        amino_acid_ids,
        modification_ids,
        precursor_charge_ids,
        residue_mask,
    )

for name, tensor in output.items():
    print(name, tensor.shape if torch.is_tensor(tensor) else type(tensor))
