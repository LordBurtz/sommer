"""checks bond -> slot placement of the per-bond head. run: uv run python test_train_deep.py"""
import torch

import common as c
from train_deep import place_bonds

PEPTIDE = "GASPK"
B_VALUES = [1.0, 2.0, 3.0, 4.0]
Y_VALUES = [10.0, 20.0, 30.0, 40.0]
PAD_VALUE = 999.0


def test_place_bonds_length_5():
    tokens = torch.from_numpy(c.encode_tokens(PEPTIDE)).unsqueeze(0)
    n_res = (tokens != 0).sum(dim=1)
    assert n_res.item() == 5

    bond_out = torch.full((1, c.MAX_FRAGMENTS, 2), PAD_VALUE)
    bond_out[0, :4, 0] = torch.tensor(B_VALUES)
    bond_out[0, :4, 1] = torch.tensor(Y_VALUES)

    out = place_bonds(bond_out, n_res)[0]
    assert out.shape == (c.VECTOR_DIM,)

    slot = lambda ion, n: out[c.fragment_slot(ion, n)].item()
    assert slot("y", 1) == Y_VALUES[-1], "y1 must come from the last bond"
    assert slot("y", 4) == Y_VALUES[0], "y4 must come from the first bond"

    expected = {}
    for n in range(1, 5):
        expected[("b", n)] = B_VALUES[n - 1]
        expected[("y", n)] = Y_VALUES[4 - n]
    for (ion, n), value in expected.items():
        assert slot(ion, n) == value, (ion, n, slot(ion, n), value)

    filled = {c.fragment_slot(ion, n) for ion, n in expected}
    rest = [i for i in range(c.VECTOR_DIM) if i not in filled]
    assert (out[rest] == 0).all(), "slots of non-existent fragments must be 0"
    assert PAD_VALUE not in out.tolist(), "padding bonds must not leak into the output"
    return out


if __name__ == "__main__":
    out = test_place_bonds_length_5()
    print(f"{PEPTIDE} (5 residues, bonds 0-3)")
    for n in range(1, 5):
        print(f"  b{n} = {out[c.fragment_slot('b', n)].item():4.0f}  (bond {n - 1})"
              f"    y{n} = {out[c.fragment_slot('y', n)].item():4.0f}  (bond {4 - n})")
    print(f"  other {c.VECTOR_DIM - 8} slots: all 0")
    print("ok")
