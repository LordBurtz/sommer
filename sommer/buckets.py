"""Per-bucket breakdown of the spectral angle: precursor charge x peptide length bin x modified.

modified = any [UNIMOD:x] token; every cysteine in the data is C[UNIMOD:4], so modified includes all Cys peptides.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sommer.constants import CHARGES

LENGTH_BINS = {"7-12": (7, 12), "13-18": (13, 18), "19-24": (19, 24), "25-30": (25, 30)}


def length_bin(length: np.ndarray) -> np.ndarray:
    out = np.full(len(length), "other", dtype=object)
    for name, (lo, hi) in LENGTH_BINS.items():
        out[(length >= lo) & (length <= hi)] = name
    return out


def bucket_table(sa: np.ndarray, charge: np.ndarray, length: np.ndarray, modified: np.ndarray) -> pd.DataFrame:
    """n, mean and median SA per bucket; every bucket is listed, empty ones with n = 0"""
    df = pd.DataFrame({"precursor_charge": charge, "length_bin": length_bin(length),
                       "modified": np.where(modified, "yes", "no"), "sa": sa})
    bins = list(LENGTH_BINS) + (["other"] if (df["length_bin"] == "other").any() else [])
    full = pd.MultiIndex.from_product([list(CHARGES), bins, ["no", "yes"]],
                                      names=["precursor_charge", "length_bin", "modified"])
    g = df.groupby(["precursor_charge", "length_bin", "modified"])["sa"]
    out = pd.DataFrame({"n": g.size(), "mean_sa": g.mean(), "median_sa": g.median()}).reindex(full)
    out["n"] = out["n"].fillna(0).astype(int)
    return out.reset_index()


def to_records(table: pd.DataFrame) -> list[dict]:
    """JSON-safe rows for the run log (NaN of empty buckets -> None)"""
    return [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}
            for row in table.astype(object).to_dict("records")]
