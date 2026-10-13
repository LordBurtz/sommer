"""add a `fold` column (0-9) to refac/clean_spectra_split.parquet without changing the split.

Recomputes the 10 StratifiedGroupKFold folds exactly like refac/01_data_cleansing.ipynb
(strat = frag + "_z" + precursor_charge, groups = stripped_sequence, random_state 187) and only
writes if fold 0 == the current test rows and fold 1 == the current val rows, row for row.
Folds 2-9 are the train rows; train_deep.py --val-fold K uses fold K as val, 0 stays test.
Every other column is kept as is. Safe to rerun (an existing fold column is recomputed and checked).
Usage:
  uv run python add_folds.py
  uv run python add_folds.py --data refac/clean_spectra_split.parquet
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedGroupKFold

RANDOM_STATE = 187
N_SPLITS = 10
TEST_FOLD, VAL_FOLD = 0, 1


def compute_folds(df) -> np.ndarray:
    """fold number per row, same StratifiedGroupKFold call as refac/01 step 3."""
    strat = df["frag"] + "_z" + df["precursor_charge"].astype(str)
    groups = df["stripped_sequence"]
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold = np.full(len(df), -1, dtype=np.int64)
    for k, (_, idx) in enumerate(sgkf.split(df, strat, groups)):
        fold[idx] = k
    assert (fold >= 0).all(), "some rows are in no fold"
    return fold


def check_against_split(fold: np.ndarray, split: np.ndarray) -> None:
    for k, name in ((TEST_FOLD, "test"), (VAL_FOLD, "val")):
        rows_fold, rows_split = np.flatnonzero(fold == k), np.flatnonzero(split == name)
        assert np.array_equal(rows_fold, rows_split), (
            f"fold {k} != current {name} rows ({len(rows_fold):,} vs {len(rows_split):,} rows, "
            f"{len(np.setxor1d(rows_fold, rows_split)):,} differ). Nothing written.")
    assert ((fold > VAL_FOLD) == (split == "train")).all(), "folds 2-9 != current train rows. Nothing written."


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("refac/clean_spectra_split.parquet"))
    args = ap.parse_args(argv)

    table = pq.read_table(args.data)
    if "fold" in table.column_names:
        table = table.drop_columns(["fold"])
    df = table.select(["frag", "precursor_charge", "stripped_sequence", "split"]).to_pandas()
    split = df["split"].to_numpy()

    fold = compute_folds(df)
    check_against_split(fold, split)
    print(f"[ok] fold {TEST_FOLD} == test, fold {VAL_FOLD} == val, folds 2-{N_SPLITS - 1} == train")

    tmp = args.data.with_suffix(".tmp.parquet")
    pq.write_table(table.append_column("fold", pa.array(fold)), tmp)
    os.replace(tmp, args.data)

    back = pq.read_table(args.data)
    assert back.drop_columns(["fold"]).equals(table), "written file differs from the original apart from fold"
    assert np.array_equal(back["fold"].to_numpy(), fold)
    counts = np.bincount(fold, minlength=N_SPLITS)
    print("rows per fold: " + "  ".join(f"{k}: {n:,}" for k, n in enumerate(counts)))
    print(f"wrote fold column to {args.data}")


if __name__ == "__main__":
    main()
