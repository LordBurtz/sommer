"""score finished checkpoints on the test split. The only script that loads the test split.

  results/test_results.json   one entry per checkpoint given (rewritten on every call): scores,
                              plus a good / average / poor example spectrum for make_figures.py
Scoring is train_deep.predict + train_deep.evaluate, the same functions as the validation scores,
on the raw model outputs (no post-processing beyond what the validation scoring does).
Usage:
  uv run python evaluate_test.py checkpoints/<a>.pt [checkpoints/<b>.pt ...]
  uv run python evaluate_test.py checkpoints/<a>.pt --data-dir /content/drive/MyDrive/sommer
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import common as c
import train_deep as td

SPLIT = "test"
CHARGES = ("2", "3", "4")
EXAMPLES = {"good": 0.9, "average": 0.5, "poor": 0.1}    # quantile of the per-spectrum spectral angle


def load_test(data_path: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    df = pd.read_parquet(data_path, columns=td.READ_COLUMNS, filters=[("split", "==", SPLIT)])
    assert len(df) and set(df["split"]) == {SPLIT}
    df = df.sort_values(["raw_file", "scan_number"]).reset_index(drop=True)
    return df, td.encode(df)


def examples(df: pd.DataFrame, A: dict[str, np.ndarray], pred: np.ndarray) -> list[dict]:
    # per spectrum through the same scoring function as the overall numbers
    sa = np.array([c.evaluate_vectors(A["Y"][i:i + 1], pred[i:i + 1], A["mask"][i:i + 1].astype(bool))["spectral_angle"]
                   for i in range(len(pred))])
    order = np.argsort(sa, kind="stable")
    out = []
    for label, q in EXAMPLES.items():
        i = int(order[round(q * (len(order) - 1))])
        row = df.iloc[i]
        out.append({"label": label, "quantile": q, "raw_file": row["raw_file"], "scan_number": int(row["scan_number"]),
                    "peptide_sequence": row["peptide_sequence"], "precursor_charge": int(row["precursor_charge"]),
                    "frag": row["frag"], "spectral_angle": float(sa[i]),
                    "measured": A["Y"][i].tolist(), "predicted": pred[i].tolist()})
    return out


def score(path: Path, df: pd.DataFrame, A: dict[str, np.ndarray], tensors, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device)
    cfg = td.Config(**ckpt["config"])
    model = td.build_model(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    pred = td.predict(model, np.arange(len(A["Y"])), tensors, 4096)
    return {
        "checkpoint": str(path),
        "tag": cfg.tag,
        "activation": cfg.activation,
        "head": cfg.head,
        "use_frag": cfg.use_frag,
        "seed": cfg.seed,
        "best_epoch": ckpt["epoch"],
        "val_spectral_angle": ckpt["val_sa"],
        **td.evaluate(pred, A["Y"], A["mask"], A["charge"], split=SPLIT),
        "examples": examples(df, A, pred),
    }


def fmt(x) -> str:
    return "-" if x is None else f"{x:.4f}"


def print_table(models: list[dict]) -> None:
    headers = ["#", "name", "activation", "val SA", f"{SPLIT} SA", f"{SPLIT} R",
               *(f"{m} z={z}" for z in CHARGES for m in ("SA", "R"))]
    rows = [[str(i + 1), m["tag"], m["activation"], fmt(m["val_spectral_angle"]),
             fmt(m[f"{SPLIT}_spectral_angle"]), fmt(m[f"{SPLIT}_pearson"]),
             *(fmt(m["by_charge"][z][k]) for z in CHARGES for k in ("spectral_angle", "pearson"))]
            for i, m in enumerate(models)]
    table = [headers] + rows
    widths = [max(len(row[j]) for row in table) for j in range(len(headers))]
    for n, row in enumerate(table):
        print("  ".join(v.ljust(w) if j in (1, 2) else v.rjust(w) for j, (v, w) in enumerate(zip(row, widths))))
        if n == 0:
            print("  ".join("-" * w for w in widths))
    print("\nR = Pearson correlation, val SA = from the checkpoint's own run")
    if any(m["activation"] != "sigmoid" for m in models):
        print("non-sigmoid models: predictions outside [0, 1] are clipped by the shared scoring, as in validation")


def main(argv=None) -> list[dict]:
    defaults = td.Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--data-dir", default=defaults.data_dir)
    ap.add_argument("--data-file", default=defaults.data_file)
    ap.add_argument("--out-dir", default=defaults.out_dir)
    args = ap.parse_args(argv)

    for p in args.checkpoints:
        if not p.is_file() or p.stat().st_size == 0:
            sys.exit(f"{p}: missing or empty (a run that stops before its first epoch leaves an empty file)")

    device = td.pick_device()
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count())
    data = td.resolve(args.data_dir, args.data_file)
    df, A = load_test(data)
    tensors = tuple(torch.from_numpy(np.array(A[k])).to(device) for k in ("tokens", "charge", "Y", "mask", "frag"))
    print(f"device: {device}  n_test={len(A['Y']):,}")

    models = []
    for p in args.checkpoints:
        models.append(score(p, df, A, tensors, device))
        print(f"scored {p}")

    out = td.resolve(args.out_dir, "results/test_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": td.git_commit(),
        "data": str(data),
        "n_test": int(len(A["Y"])),
        "test_fingerprint": td.val_fingerprint(A, np.arange(len(A["Y"]))),
        "models": models,
    }, indent=2) + "\n")

    n = models[0]["by_charge"]
    print(f"\ntest set: {len(A['Y']):,} spectra ({', '.join(f'z={z}: {n[z]['n']:,}' for z in CHARGES)})\n")
    print_table(models)
    print(f"\nwritten to {out}")
    return models


if __name__ == "__main__":
    main()
