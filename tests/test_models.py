import numpy as np
import pytest
import torch

from sommer import peptides
from sommer.constants import MAX_FRAGMENTS, VECTOR_DIM
from sommer.models import MODELS, FragmentModel, build_model, register_model
from sommer.models.layers import place_bonds

SMALL = {"PrositLike": {"emb": 8, "hidden": 8}, "PrositLike_bond": {"emb": 8, "hidden": 8, "head": "bond"},
         "PeptideEncoder": {"hidden_dimensions": 16, "transformer_layers": 1, "attention_heads": 2}}


def batch(seqs=("GASPK", "AC[UNIMOD:4]DM[UNIMOD:35]K", "LTQETNRV"), charges=(2, 3, 4)):
    return torch.from_numpy(peptides.encode_batch(seqs)), torch.tensor(charges)


def test_place_bonds_length_5():
    tokens = torch.from_numpy(peptides.encode_tokens("GASPK")).unsqueeze(0)
    bond_out = torch.full((1, MAX_FRAGMENTS, 2), 999.0)
    bond_out[0, :4, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    bond_out[0, :4, 1] = torch.tensor([10.0, 20.0, 30.0, 40.0])
    out = place_bonds(bond_out, (tokens != 0).sum(dim=1))[0]
    slot = lambda ion, n: out[peptides.fragment_slot(ion, n)].item()
    assert [slot("b", n) for n in range(1, 5)] == [1.0, 2.0, 3.0, 4.0]
    assert [slot("y", n) for n in range(1, 5)] == [40.0, 30.0, 20.0, 10.0]   # y1 from the last bond
    assert (out[8:] == 0).all()


@pytest.mark.parametrize("key", SMALL)
def test_registered_models_follow_contract(key):
    name = key.split("_")[0]
    model, params = build_model(name, SMALL[key])
    assert isinstance(model, FragmentModel) and model.name == name
    assert set(SMALL[key]) <= set(params)          # full params incl. defaults are recorded
    model.eval()
    out = model(*batch())
    assert out.shape == (3, VECTOR_DIM) and torch.isfinite(out).all() and (out >= 0).all()
    pred = model.predict(["GASPK", "LTQETNRV"], [2, 4])
    assert pred.shape == (2, VECTOR_DIM)


def test_all_registered_models_are_covered():
    assert {k.split("_")[0] for k in SMALL} == set(MODELS) - {"TinyTestModel"}


def test_new_model_only_needs_subclass_and_decorator():
    @register_model
    class TinyTestModel(FragmentModel):
        def __init__(self, width: int = 4):
            super().__init__()
            self.embedding = torch.nn.Embedding(30, width)
            self.head = torch.nn.Linear(width, VECTOR_DIM)

        def forward(self, tokens, precursor_charge):
            return torch.nn.functional.softplus(self.head(self.embedding(tokens).mean(dim=1)))

    model, params = build_model("TinyTestModel")
    assert params == {"width": 4} and model(*batch()).shape == (3, VECTOR_DIM)
    with pytest.raises(TypeError):
        register_model(torch.nn.Linear)
