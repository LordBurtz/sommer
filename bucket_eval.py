"""score one checkpoint on its own val fold, per precursor charge x peptide length bin x modified
modified = any [UNIMOD:x] token (C[UNIMOD:4] or
M[UNIMOD:35]); every cysteine in the data is C[UNIMOD:4], so modified = yes includes all Cys peptides
Usage:
  uv run python bucket_eval.py checkpoints/<tag>_<UTC start>.pt
  uv run python bucket_eval.py checkpoints/<a>.pt --data-dir /content/drive/MyDrive/sommer
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import common as c
import train_deep as td

CHARGES = (2, 3, 4)
LENGTH_BINS = {"7-12": (7, 12), "13-18": (13, 18), "19-24": (19, 24), "25-30": (25, 30)}
MOD_IDX = [c.TOKEN_TO_IDX[t] for t in c.MOD_TOKENS]


def per_spectrum_sa(pred: np.ndarray, Y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pred = np.clip(pred, 0.0, 1.0)                    # as in common.evaluate_vectors
    return np.array([c.spectral_angle(t, p, m) for t, p, m in zip(Y, pred, mask.astype(bool))])


def length_bin(length: np.ndarray) -> np.ndarray:
    out = np.full(len(length), "", dtype=object)
    for name, (lo, hi) in LENGTH_BINS.items():
        out[(length >= lo) & (length <= hi)] = name
    assert (out != "").all(), f"peptide lengths outside the bins: {sorted(set(length[out == '']))}"
    return out


def bucket_table(sa: np.ndarray, charge: np.ndarray, length: np.ndarray, modified: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"precursor_charge": charge, "length_bin": length_bin(length),
                       "modified": np.where(modified, "yes", "no"), "sa": sa})
    full = pd.MultiIndex.from_product([CHARGES, list(LENGTH_BINS), ["no", "yes"]],
                                      names=["precursor_charge", "length_bin", "modified"])
    g = df.groupby(["precursor_charge", "length_bin", "modified"])["sa"]
    out = pd.DataFrame({"n": g.size(), "mean_sa": g.mean(), "median_sa": g.median()}).reindex(full)
    out["n"] = out["n"].fillna(0).astype(int)
    return out.reset_index()


def print_table(t: pd.DataFrame, sa: np.ndarray) -> None:
    rows = [[str(r.precursor_charge), r.length_bin, r.modified, f"{r.n:,}",
             "-" if r.n == 0 else f"{r.mean_sa:.4f}", "-" if r.n == 0 else f"{r.median_sa:.4f}"]
            for r in t.itertuples()]
    rows.append(["all", "", "", f"{len(sa):,}", f"{sa.mean():.4f}", f"{np.median(sa):.4f}"])
    headers = ["z", "length", "modified", "n", "mean SA", "median SA"]
    table = [headers] + rows
    widths = [max(len(row[j]) for row in table) for j in range(len(headers))]
    for n, row in enumerate(table):
        if n == len(table) - 1:
            print("  ".join("-" * w for w in widths))
        print("  ".join(v.ljust(w) if j in (1, 2) else v.rjust(w) for j, (v, w) in enumerate(zip(row, widths))))
        if n == 0:
            print("  ".join("-" * w for w in widths))


def main(argv=None) -> pd.DataFrame:
    defaults = td.Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--data-dir", default=defaults.data_dir)
    ap.add_argument("--data-file", default=defaults.data_file)
    ap.add_argument("--out-dir", default=defaults.out_dir)
    ap.add_argument("--cache-file", default=defaults.cache_file)
    args = ap.parse_args(argv)
    if not args.checkpoint.is_file() or args.checkpoint.stat().st_size == 0:
        sys.exit(f"{args.checkpoint}: missing / empty")

    device = td.pick_device()
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count())
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = td.Config(**ckpt["config"])
    model = td.build_model(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])

    A = td.load_arrays(td.Config(data_dir=args.data_dir, data_file=args.data_file,
                                 out_dir=args.out_dir, cache_file=args.cache_file))
    assert (A["fold"] != td.TEST_FOLD).all(), "test rows in the training arrays"
    va_idx = np.where(A["fold"] == cfg.val_fold)[0]
    tensors = tuple(torch.from_numpy(np.array(A[k])).to(device) for k in ("tokens", "charge", "Y", "mask", "frag"))
    pred = td.predict(model, va_idx, tensors, 4096)
    Y, M, Z, tok = A["Y"][va_idx], A["mask"][va_idx], A["charge"][va_idx], A["tokens"][va_idx]

    sa = per_spectrum_sa(pred, Y, M)
    overall = td.evaluate(pred, Y, M, Z)["val_spectral_angle"]
    assert np.isclose(sa.mean(), overall, rtol=0, atol=1e-9), (sa.mean(), overall)
    t = bucket_table(sa, Z, (tok != 0).sum(axis=1), np.isin(tok, MOD_IDX).any(axis=1))

    print(f"{cfg.tag}  ({args.checkpoint})")
    print(f"val fold {cfg.val_fold}: {len(va_idx):,} spectra, fingerprint {td.val_fingerprint(A, va_idx)}  |  "
          f"val SA {overall:.4f} (checkpoint: {ckpt['val_sa']:.4f}{', smoke subset' if cfg.smoke else ''})\n")
    print_table(t, sa)

    out = td.resolve(args.out_dir, f"results/bucket_eval_{args.checkpoint.stem}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    t.insert(0, "val_fold", cfg.val_fold)
    t.insert(0, "checkpoint", str(args.checkpoint))
    t.to_csv(out, index=False)
    print(f"\nwritten to {out}")
    return t


if __name__ == "__main__":
    main()
