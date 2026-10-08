"""Build one fixed, leak-free train/val/test split -> splits.parquet

Groups are stripped peptide sequences (no peptide, modified or not, appears in
more than one split); stratified by acquisition method x precursor charge.
Fold 0 = test, fold 1 = val, folds 2-9 = train (~80/10/10).

The test split is held out: do not train, tune or evaluate on it until the end.

Usage: uv run python make_split.py [--data spectra_with_charge.parquet] [--out splits.parquet]
"""
import argparse
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import common as c

KEEP_CHARGES = (2, 3, 4)
DROP_FRAGS = ("DDA-longexcl",)
N_SPLITS = 10
SEED = 42

# "..._01_01-2xIT_2xHCD-1h-R1" -> "2xIT_2xHCD-1h"
_FRAG_RE = re.compile(r"_\d{2}_\d{2}-(.+?)(?:-R\d+)?$")


def frag_from_raw_file(raw_file: str) -> str:
    m = _FRAG_RE.search(raw_file)
    if m is None:
        raise ValueError(f"cannot parse acquisition method from {raw_file!r}")
    return m.group(1)


def make_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["raw_file", "scan_number", "peptide_sequence", "precursor_charge", "charge_source"]].copy()
    df["stripped_sequence"] = df["peptide_sequence"].map(c.strip_mods)
    df["frag"] = df["raw_file"].map(frag_from_raw_file)

    n0 = len(df)
    df = df[df["precursor_charge"].isin(KEEP_CHARGES) & ~df["frag"].isin(DROP_FRAGS)]
    df = df.reset_index(drop=True)
    print(f"kept {len(df):,} / {n0:,} spectra (charges {KEEP_CHARGES}, dropped frags {DROP_FRAGS})")

    strata = df["frag"] + "_z" + df["precursor_charge"].astype(str)
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold = np.full(len(df), -1)
    for k, (_, idx) in enumerate(sgkf.split(df, strata, groups=df["stripped_sequence"])):
        fold[idx] = k
    assert (fold >= 0).all()
    df["split"] = np.where(fold == 0, "test", np.where(fold == 1, "val", "train"))
    return df


def check_and_report(df: pd.DataFrame) -> None:
    n_splits_per_seq = df.groupby("stripped_sequence")["split"].nunique()
    leaked = n_splits_per_seq[n_splits_per_seq > 1]
    assert leaked.empty, f"{len(leaked)} stripped sequences appear in more than one split"
    assert not df.duplicated(["raw_file", "scan_number"]).any()
    print("leakage check passed: every stripped_sequence is in exactly one split")

    order = ["train", "val", "test"]
    sizes = df["split"].value_counts().reindex(order)
    print("\nsize per split:")
    print(pd.DataFrame({"n": sizes, "frac": (sizes / len(df)).round(4),
                        "n_peptides": df.groupby("split")["stripped_sequence"].nunique().reindex(order)}))
    for col in ("precursor_charge", "frag", "charge_source"):
        print(f"\n{col} proportions per split:")
        print(pd.crosstab(df[col], df["split"], normalize="columns")[order].round(4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="spectra_with_charge.parquet")
    ap.add_argument("--out", default="splits.parquet")
    args = ap.parse_args()

    df = make_split(pd.read_parquet(args.data))
    check_and_report(df)
    df[["raw_file", "scan_number", "split"]].to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
