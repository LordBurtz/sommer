""" checks f common.py (so we definitely dont uee old wrong values). run: uv run python test_common.py"""
import pandas as pd

import common as c


def test_mono_masses_match_xlsx():
    xlsx = pd.read_excel("amino_acid_masses.xlsx").dropna(subset=["code"])
    ref = dict(zip(xlsx["code"], xlsx["calculated mono mass"]))
    for aa in "AGK":
        assert abs(c.AA_MONO_MASS[aa] - ref[aa]) < 1e-6, (aa, c.AA_MONO_MASS[aa], ref[aa])
    # full table chceck
    for aa, m in ref.items():
        assert abs(c.AA_MONO_MASS[aa] - m) < 1e-6, (aa, c.AA_MONO_MASS[aa], m)


if __name__ == "__main__":
    test_mono_masses_match_xlsx()
    print("ok")
