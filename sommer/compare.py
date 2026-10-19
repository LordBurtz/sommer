"""Compare recently trained models with previously logged runs and with Koina models.

All scores come from results/run_log.jsonl. Koina models are scored once per validation set (keyed by
its fingerprint) through the same evaluation as our models and cached in the log as source="koina".

  uv run python -m sommer.compare                 # leaderboard + recent runs vs. previous best vs. Koina
  uv run python -m sommer.compare --recent 5 --no-koina
  uv run python -m sommer.compare --koina-models Prosit_2020_intensity_CID ms2pip_HCD2021 --force-koina
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sommer import evaluation, tracking
from sommer.data import SpectrumSet, load_dataset
from sommer.koina import DEFAULT_MODELS, KoinaModel
from sommer.paths import RUN_LOG

CHARGE_COLUMNS = ["z=2", "z=3", "z=4"]
COLUMNS = ["label", "source", "val_set", "n_params", "val_sa", "val_pearson", *CHARGE_COLUMNS, "date"]


def label(record: dict) -> str:
    if record["source"] == "koina":
        return f"Koina {record['model']}"
    if record["source"].startswith("legacy"):
        return f"{record['model']} (legacy {record['timestamp'][:16].replace('T', ' ')})"
    return f"{record['model']} ({record['run_id'][-8:]})"


def score_koina(val: SpectrumSet, models=DEFAULT_MODELS, force: bool = False, run_log: Path = RUN_LOG) -> list[dict]:
    """score Koina models on the val set unless a record for this val fingerprint exists"""
    fingerprint = val.fingerprint()
    cached = {r["model"]: r for r in tracking.load_records(run_log)
              if r["source"] == "koina" and r["data"]["val_fingerprint"] == fingerprint}
    records = []
    for name in models:
        if name in cached and not force:
            records.append(cached[name])
            continue
        koina = KoinaModel(name)
        started = tracking.utc_now()
        record = {
            "run_id": f"koina_{name}_{tracking.new_run_id(started)}", "timestamp": started.isoformat(timespec="seconds"),
            "source": "koina", "mode": "full", "model": name, "model_params": koina.assumptions, "n_params": None,
            "data": {"n_val": len(val), "val_fingerprint": fingerprint},
            **evaluation.evaluate(koina, val),
            "seconds": None, "git_commit": tracking.git_commit(), "checkpoint": None,
        }
        record["seconds"] = round((tracking.utc_now() - started).total_seconds(), 1)
        tracking.append_record(record, run_log)
        records.append(record)
    return records


def leaderboard(val_fingerprint: str | None = None, run_log: Path = RUN_LOG, include_other: bool = False,
                print_table: bool = True) -> pd.DataFrame:
    """all full runs ranked by val SA

    val_set: "current" = same val fingerprint, "legacy" = old run on the val split (no fingerprint),
    "other" = scored on different val data (hidden unless include_other).
    """
    rows = []
    for r in tracking.load_records(run_log):
        if r.get("mode") != "full":
            continue
        fp = r.get("data", {}).get("val_fingerprint")
        val_set = "legacy" if fp is None else ("current" if val_fingerprint in (None, fp) else "other")
        rows.append({
            "label": label(r),
            "source": r["source"], "val_set": val_set, "model": r["model"], "n_params": r.get("n_params"),
            "val_sa": r["val_sa"], "val_pearson": r.get("val_pearson"),
            **{f"z={z}": r["by_charge"].get(z, {}).get("sa") for z in ("2", "3", "4")},
            "timestamp": pd.Timestamp(r["timestamp"]), "date": r["timestamp"][:16].replace("T", " "), "run_id": r["run_id"], "checkpoint": r.get("checkpoint"),
        })
    table = pd.DataFrame(rows, columns=COLUMNS + ["model", "timestamp", "run_id", "checkpoint"])
    if not include_other:
        table = table[table["val_set"] != "other"]
    table = table.sort_values("val_sa", ascending=False, kind="stable").reset_index(drop=True)
    if print_table:
        print(format_table(table[COLUMNS]))
    return table


def compare_recent(table: pd.DataFrame, recent: int = 3, print_table: bool = True) -> pd.DataFrame:
    """the `recent` newest pipeline runs vs. the best earlier run and the best Koina model"""
    ours = table[table["source"] != "koina"].sort_values("timestamp")
    new = ours[ours["source"] == "pipeline"].tail(recent)
    koina = table[table["source"] == "koina"]
    best_koina = koina.iloc[0] if len(koina) else None
    rows = []
    for _, run in new.iloc[::-1].iterrows():
        earlier = ours[(ours["timestamp"] < run["timestamp"]) & (ours["run_id"] != run["run_id"])]
        best_prev = earlier.sort_values("val_sa", ascending=False).iloc[0] if len(earlier) else None
        row = {"label": run["label"], "date": run["date"], "val_sa": run["val_sa"],
               "rank": int(table.index[table["run_id"] == run["run_id"]][0]) + 1}
        for prefix, ref in (("prev_best", best_prev), ("koina", best_koina)):
            row[prefix] = None if ref is None else ref["label"]
            row[f"Δ {prefix}"] = None if ref is None else run["val_sa"] - ref["val_sa"]
            for z in CHARGE_COLUMNS:
                row[f"Δ {prefix} {z}"] = (None if ref is None or pd.isna(run[z]) or pd.isna(ref[z])
                                          else run[z] - ref[z])
        rows.append(row)
    out = pd.DataFrame(rows)
    if print_table:
        print(format_table(out) if len(out) else "no pipeline runs logged yet")
    return out


def bucket_frame(run_ids: list[str], run_log: Path = RUN_LOG) -> pd.DataFrame:
    """long table of per-bucket SA for the given runs (legacy runs have no buckets)"""
    frames = []
    for r in tracking.load_records(run_log):
        if r["run_id"] in run_ids and r.get("buckets"):
            frames.append(pd.DataFrame(r["buckets"]).assign(label=label(r), run_id=r["run_id"]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def format_table(table: pd.DataFrame) -> str:
    return table.to_string(index=False, na_rep="-", float_format=lambda v: f"{v:.4f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recent", type=int, default=3, help="number of newest pipeline runs to compare")
    ap.add_argument("--no-koina", action="store_true", help="do not query Koina for missing scores")
    ap.add_argument("--force-koina", action="store_true", help="re-score Koina models even if cached")
    ap.add_argument("--koina-models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--include-other", action="store_true", help="also show runs scored on other val data")
    args = ap.parse_args(argv)

    val = load_dataset()["val"]
    if not args.no_koina:
        score_koina(val, args.koina_models, force=args.force_koina)
    print("\n=== leaderboard (validation split) ===")
    table = leaderboard(val.fingerprint(), include_other=args.include_other)
    print(f"\n=== {args.recent} most recent runs vs. previous best and Koina ===")
    compare_recent(table, args.recent)


if __name__ == "__main__":
    main()
