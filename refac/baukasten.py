import torch
import torch.nn as nn

# --------------------------------
# Embedding implementations
# --------------------------------

"""
This injects information about the order of the AAs in the sequence
Without it ACAB and AABC would look the same
"""
class PositionalEncoding(nn.Module):
    def __init__(
            self,
            hidden_dimension: int,
            max_peptide_length = 30
    ):
        super().__init__()

        self.position_embedding = nn.Embedding(
            max_peptide_length,
            hidden_dimension,
        )

    def forward(self, residue_features):
        _batch_size, peptide_length, _hidden_dimension = residue_features.shape

        position_idxs = torch.arange(
            peptide_length,
        )

        position_features = self.position_embedding(position_idxs)
        encoded_residues = residue_features + position_features.unsqueeze(0)

        return encoded_residues

"""
Use RoPE: rotate vector and by an angle proportional to the distance to the start
This allows the attention mechanism to naturally depend on the relative distance between two tokens
Seen in Gemma and Llama
"""
class RoPE_Encoding(nn.Module):
    def __init__():
        pass

    def forward(self, residue_features):
        pass


# -------------------------------
# Transformer implementation
# -------------------------------
"""
Uses the Transformer Architecture as in Attention is All you Need
"""
class TransformerEncoder(nn.Module):
    def __init__(
            self,
            hidden_dimension,
            layer_count,
            attention_head_count,
            bidirectional=True,
            dropout=0.1,
    ):
        super().__init__()

        assert hidden_dimension % attention_head_count == 0 # assert clean division to heads

        self.bidirectional = bidirectional

        transformer_layer = nn.TransformerEncoderLayer(
            d_model= hidden_dimension,
            nhead = attention_head_count,
            dim_feedforward = 4 * hidden_dimension,
            dropout = dropout,
            activation = "gelu", # relu might be faster but this is default for most modern transformers
            batch_first = True,
            norm_first = True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer = transformer_layer,
            num_layers = layer_count,
            norm = nn.LayerNorm(hidden_dimension),
        )

    def forward(self, residue_features, padding_mask=None):
        attention_mask = None

        # Create an attention mask to mask out the backward attention by
        # Zeroing the bottom half of the diagonally split matrixk
        if not self.bidirectional:
            sequence_length = residue_features.shape[1]

            attention_mask = torch.triu(
                torch.ones(
                    sequence_length,
                    sequence_length,
                    dtype=torch.bool,
                ),
                diagonal = 1
            )

        return self.encoder(
            residue_features,
            mask = attention_mask,
            src_key_padding_mask = padding_mask
        )

# ------------------------------------
# Implementing pooling
# ------------------------------------

"""
Masked mean pooling produces global representation without letting padding distort
"""
def masked_mean_pooling(features, mask):

    valid_weights = mask.unsqueeze(-1).to(
        dtype = features.dtype,
    )

    masked_features = features * valid_weights

    summed_features = masked_features.sum(dim=1)

    valid_residues_count = valid_weights.sum(dim=1).clamp(min=1)

    mean_features = summed_features / valid_residues_count

    return mean_features
