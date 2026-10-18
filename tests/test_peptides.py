import numpy as np
import pandas as pd
import pytest

from sommer import peptides
from sommer.constants import AA_MONO_MASS, VECTOR_DIM
from sommer.paths import ROOT


def test_mono_masses_match_xlsx():
    xlsx = pd.read_excel(ROOT / "amino_acid_masses.xlsx").dropna(subset=["code"])
    for aa, mass in zip(xlsx["code"], xlsx["calculated mono mass"]):
        assert abs(AA_MONO_MASS[aa] - mass) < 1e-6, (aa, AA_MONO_MASS[aa], mass)


def test_parse_and_encode():
    seq = "SM[UNIMOD:35]AC[UNIMOD:4]K"
    assert peptides.parse_peptide(seq) == ["S", "M[UNIMOD:35]", "A", "C[UNIMOD:4]", "K"]
    assert peptides.peptide_length(seq) == 5 and peptides.is_modified(seq)
    tokens = peptides.encode_tokens(seq)
    assert (tokens[:5] != 0).all() and (tokens[5:] == 0).all()
    with pytest.raises(ValueError):
        peptides.encode_tokens("A" * 31)


def test_slots_and_mask():
    for slot in range(VECTOR_DIM):
        assert peptides.fragment_slot(*peptides.slot_to_fragment(slot)) == slot
    mask = peptides.valid_slot_mask("GASPK")
    assert mask.sum() == 8 and mask[:8].all()
    vec = peptides.ions_to_vector(["b1", "y4"], [2.0, 4.0])
    assert vec[peptides.fragment_slot("y", 4)] == 1.0 and vec[peptides.fragment_slot("b", 1)] == 0.5
    table = peptides.prediction_table("GASPK", vec)
    assert len(table) == 8 and list(table.columns) == ["Fragment ion", "Fragment Number", "Intensity"]
