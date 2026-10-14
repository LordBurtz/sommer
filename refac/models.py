import torch
import torch.nn as nn

from baukasten import PositionalEncoding, TransformerEncoder, masked_mean_pooling


class PeptideEncoder(nn.Module):
    def __init__(
        self,
        aa_vocab_size: int,
        mod_vocab_size: int,
        precursor_vocab_size: int,
        hidden_dimensions: int = 256,
        transformer_layers: int = 6,
        attention_heads: int = 8,
    ):
        super().__init__()
        self.amino_acid_embedding = nn.Embedding(aa_vocab_size, hidden_dimensions)
        self.modification_embedding = nn.Embedding(mod_vocab_size, hidden_dimensions)
        self.precursor_charge_embedding = nn.Embedding(
            precursor_vocab_size, hidden_dimensions
        )
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
        self.cleavage_site_projection = nn.Linear(
            3 * hidden_dimensions, hidden_dimensions
        )
        self.fragment_head = nn.Sequential(
            nn.Linear(hidden_dimensions, hidden_dimensions),
            nn.GELU(),
            nn.Linear(hidden_dimensions, 2),
            nn.Softplus(),
        )

    def forward(self, tokens, precursor_charge_ids, frag=None):
        residue_mask = tokens != 0

        # Token IDs 21 and 22 are modified C and M; both use the modification embedding.
        residue_features = self.amino_acid_embedding(
            torch.where(tokens == 21, 2, torch.where(tokens == 22, 11, tokens))
        ) + self.modification_embedding((tokens >= 21).long())

        residue_features = self.positional_encoding(residue_features)

        residue_features = self.sequence_transformer(
            residue_features,
            padding_mask=~residue_mask,
        )

        global_features = self.global_peptide_projection(
            masked_mean_pooling(residue_features, residue_mask)
            # as we accept 2..5 we have to subtract minus 2
            + self.precursor_charge_embedding(precursor_charge_ids - 2)
        )
        cleavage_features = self.cleavage_site_projection(
            torch.cat(
                [
                    residue_features[:, :-1],
                    residue_features[:, 1:],
                    global_features.unsqueeze(1).expand(
                        -1, residue_features.size(1) - 1, -1
                    ),
                ],
                dim=-1,
            )
        )
        fragment_intensities = self.fragment_head(cleavage_features)

        fragment_numbers = torch.arange(
            1, fragment_intensities.size(1) + 1
        ).unsqueeze(0)
        peptide_lengths = residue_mask.sum(dim=1, keepdim=True)
        y_positions = (peptide_lengths - 1 - fragment_numbers).clamp(
            0, fragment_intensities.size(1) - 1
        )

        return (
            torch.stack(
                [
                    fragment_intensities[..., 0],
                    fragment_intensities[..., 1].gather(1, y_positions),
                ],
                dim=-1,
            )
            * (fragment_numbers < peptide_lengths).unsqueeze(-1)
        ).flatten(1)


if __name__ == "__main__":
    model = PeptideEncoder(aa_vocab_size=21, mod_vocab_size=2, precursor_vocab_size=4)
    model.eval()

    # A, modified C, D, modified M, K; pad to the trainer's sequence length.
    tokens = torch.zeros((1, 30), dtype=torch.long)
    tokens[0, :5] = torch.tensor([1, 21, 3, 22, 9])
    charge = torch.tensor([2])

    with torch.inference_mode():
        spectrum = model(tokens, charge)[0].reshape(-1, 2)

    print(spectrum)

    print("Predicted b/y intensities for A-C[UNIMOD:4]-D-M[UNIMOD:35]-K:")
    for cleavage, (b_intensity, y_intensity) in enumerate(spectrum[:4], start=1):
        print(f"b{cleavage}: {b_intensity:.3f}  y{cleavage}: {y_intensity:.3f}")
