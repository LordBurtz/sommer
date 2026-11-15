"""lookup-table baseline (fit_lookup / predict_lookup from 03_intensity_baseline.ipynb) on the refac split folds.

Per fold K: fit on folds 1-9 except K, score on fold K, i.e. the same train/val as
train_deep.py --val-fold K. Same arrays (train_deep.load_arrays, test never loaded) and the same
scoring (train_deep.evaluate) as the deep model, so the numbers and val_fingerprint are comparable
with results/runs.jsonl. Lookup: mean base-peak-normalised intensity per (ion, fragment number
bucket, relative position bucket, charge), backing off to (ion, number bucket, charge), then
(ion, position bucket), then the global mean. std = sample std (ddof=1) over folds.
Usage:
  uv run python baseline_lookup.py                 # folds 1-5
  uv run python baseline_lookup.py --folds 1 2 3
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import common as c
import train_deep as td
from compare_runs import CHARGES

N_RELPOS_BINS = 10
MAX_NUM_BUCKET = 20
LEVELS = (["ion", "nb", "rb", "charge"], ["ion", "nb", "charge"], ["ion", "rb"])   # most to least specific


def explode_fragments(A: dict[str, np.ndarray], idx: np.ndarray) -> pd.DataFrame:
    """one row per existing fragment (masked slot) of the spectra idx, with the lookup keys of 03."""
    spec, slot = np.nonzero(A["mask"][idx].astype(bool))
    L = (A["tokens"][idx] != 0).sum(axis=1)[spec]
    number = slot // c.N_ION_TYPES + 1
    relpos = number / L
    return pd.DataFrame({
        "spec": spec, "slot": slot,
        "ion": slot % c.N_ION_TYPES,                                  # 0 = b, 1 = y
        "nb": np.minimum(number, MAX_NUM_BUCKET),
        "rb": np.minimum((relpos * N_RELPOS_BINS).astype(int), N_RELPOS_BINS - 1),
        "charge": A["charge"][idx][spec],
        "target": A["Y"][idx][spec, slot],
    })


def fit_lookup(frag: pd.DataFrame) -> tuple[list[pd.Series], float]:
    return [frag.groupby(keys)["target"].mean() for keys in LEVELS], float(frag["target"].mean())


def predict_lookup(frag: pd.DataFrame, table: tuple[list[pd.Series], float]) -> np.ndarray:
    tables, global_mean = table
    pred = np.full(len(frag), np.nan)
    for keys, t in zip(LEVELS, tables):
        todo = np.isnan(pred)
        if not todo.any():
            break
        pred[todo] = t.reindex(pd.MultiIndex.from_frame(frag.loc[todo, keys])).to_numpy()
    return np.where(np.isnan(pred), global_mean, pred)


def stats(values: list[float]) -> dict:
    v = np.array(values, dtype=float)
    return {"mean": v.mean(), "std": v.std(ddof=1) if len(v) > 1 else None, "median": float(np.median(v))}


def pm(s: dict) -> str:
    std = "-" if s["std"] is None else f"{s['std']:.4f}"
    return f"{s['mean']:.4f} ± {std}"


def print_table(headers: list[str], rows: list[list[str]], left: set[int]) -> None:
    table = [headers] + rows
    widths = [max(len(row[j]) for row in table) for j in range(len(headers))]
    for n, row in enumerate(table):
        print("  ".join(v.ljust(w) if j in left else v.rjust(w) for j, (v, w) in enumerate(zip(row, widths))))
        if n == 0:
            print("  ".join("-" * w for w in widths))


def run_fold(A: dict[str, np.ndarray], k: int) -> dict:
    is_val = A["fold"] == k
    tr_idx, va_idx = np.where(~is_val)[0], np.where(is_val)[0]
    table = fit_lookup(explode_fragments(A, tr_idx))
    frag_va = explode_fragments(A, va_idx)
    P = np.zeros((len(va_idx), c.VECTOR_DIM), dtype=np.float32)
    P[frag_va["spec"], frag_va["slot"]] = predict_lookup(frag_va, table)
    res = td.evaluate(P, A["Y"][va_idx], A["mask"][va_idx], A["charge"][va_idx])
    return {"val_fold": k, "n_train": len(tr_idx), "n_val": len(va_idx),
            "val_fingerprint": td.val_fingerprint(A, va_idx), **res}


def main(argv=None) -> list[dict]:
    defaults = td.Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--data-dir", default=defaults.data_dir)
    ap.add_argument("--data-file", default=defaults.data_file)
    ap.add_argument("--out-dir", default=defaults.out_dir)
    ap.add_argument("--cache-file", default=defaults.cache_file)
    args = ap.parse_args(argv)
    bad = [k for k in args.folds if not 1 <= k < td.N_FOLDS]
    if bad:
        ap.error(f"folds must be 1..{td.N_FOLDS - 1} (fold {td.TEST_FOLD} is test), got {bad}")

    A = td.load_arrays(td.Config(data_dir=args.data_dir, data_file=args.data_file,
                                 out_dir=args.out_dir, cache_file=args.cache_file))
    assert (A["fold"] != td.TEST_FOLD).all(), "test rows in the training arrays"
    results = []
    for k in args.folds:
        results.append(run_fold(A, k))
        print(f"fold {k} done", flush=True)

    print("\nlookup baseline, per val fold\n")
    print_table(["fold", "n_train", "n_val", "val SA", "Pearson", *(f"z={z}" for z in CHARGES), "val_fingerprint"],
                [[str(r["val_fold"]), f"{r['n_train']:,}", f"{r['n_val']:,}", f"{r['val_spectral_angle']:.4f}",
                  f"{r['val_pearson']:.4f}", *(f"{r['by_charge'][z]['spectral_angle']:.4f}" for z in CHARGES),
                  r["val_fingerprint"]] for r in results], {8})
    rows = {"val SA": stats([r["val_spectral_angle"] for r in results]),
            "Pearson": stats([r["val_pearson"] for r in results]),
            **{f"z={z}": stats([r["by_charge"][z]["spectral_angle"] for r in results]) for z in CHARGES}}
    print(f"\nacross folds {', '.join(map(str, args.folds))} (n = {len(results)}): mean ± std\n")
    print_table(["", "mean ± std", "median"], [[k, pm(s), f"{s['median']:.4f}"] for k, s in rows.items()], {0})
    return results


if __name__ == "__main__":
    main()
