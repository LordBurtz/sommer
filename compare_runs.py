"""table of all runs in results/runs.jsonl with val SA and difference to the baseline.
run: uv run python compare_runs.py [--file results/runs.jsonl] [--baseline <row number or run name>]
"""
import argparse
import json
import sys
from pathlib import Path

DEFAULT_OPTIONS = {"head": "pooled", "use_frag": False, "activation": "softplus"}
BASELINE_NAME = "deep_baseline"
CHARGES = ("2", "3", "4")


def load_runs(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def options(run: dict) -> dict:
    hp = run.get("hyperparameters", {})
    return {k: hp.get(k, v) for k, v in DEFAULT_OPTIONS.items()}


def find_baseline(runs: list[dict], choice: str | None) -> int | None:
    if choice is not None and choice.isdigit():
        i = int(choice) - 1
        return i if 0 <= i < len(runs) else None
    if choice is not None:
        hits = [i for i, r in enumerate(runs) if r.get("tag") == choice]
    else:
        hits = ([i for i, r in enumerate(runs) if r.get("tag") == BASELINE_NAME]
                or [i for i, r in enumerate(runs) if options(r) == DEFAULT_OPTIONS])
    return hits[-1] if hits else None


def fmt(x, spec="{:.4f}") -> str:
    return "-" if x is None else spec.format(x)


def build_rows(runs: list[dict], base_i: int | None) -> tuple[list[list[str]], bool, bool]:
    base = runs[base_i] if base_i is not None else None
    other_val = unchecked = False
    rows = []
    for i, r in enumerate(runs):
        o = options(r)
        by_charge = r.get("by_charge", {})
        fp, base_fp = r.get("val_fingerprint"), base and base.get("val_fingerprint")
        if base is None:
            delta = "-"
        elif i == base_i:
            delta = "baseline"
        elif fp and base_fp and fp != base_fp:
            delta, other_val = "other val", True
        else:
            delta = f"{r['val_spectral_angle'] - base['val_spectral_angle']:+.4f}"
            if not (fp and base_fp):
                delta, unchecked = delta + " ?", True
        rows.append([
            str(i + 1),
            r.get("timestamp", "")[:16].replace("T", " "),
            r.get("tag", "?"),
            o["head"],
            "yes" if o["use_frag"] else "no",
            o["activation"],
            fmt(r.get("n_params"), "{:,}"),
            fmt(r.get("val_spectral_angle")),
            *(fmt(by_charge.get(z, {}).get("spectral_angle")) for z in CHARGES),
            delta,
        ])
    return rows, other_val, unchecked


def print_table(rows: list[list[str]]) -> None:
    headers = ["#", "when (UTC)", "name", "head", "frag", "activation", "params",
               "val SA", "z=2", "z=3", "z=4", "Δ SA"]
    right = {0, 6, 7, 8, 9, 10, 11}
    table = [headers] + rows
    widths = [max(len(row[j]) for row in table) for j in range(len(headers))]
    for n, row in enumerate(table):
        print("  ".join(v.rjust(w) if j in right else v.ljust(w) for j, (v, w) in enumerate(zip(row, widths))))
        if n == 0:
            print("  ".join("-" * w for w in widths))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="results/runs.jsonl")
    ap.add_argument("--baseline", default=None)
    args = ap.parse_args(argv)

    path = Path(args.file)
    if not path.exists():
        sys.exit(f"{path} not found, no runs logged yet")
    runs = load_runs(path)
    if not runs:
        sys.exit(f"{path} is empty")

    base_i = find_baseline(runs, args.baseline)
    if base_i is None:
        print(f"no baseline found ({args.baseline or 'no run with default options'}), showing scores only\n")
    else:
        b = runs[base_i]
        print(f"baseline: #{base_i + 1} {b['tag']}  val SA {b['val_spectral_angle']:.4f}\n")

    rows, other_val, unchecked = build_rows(runs, base_i)
    print_table(rows)
    if other_val or unchecked:
        print()
    if other_val:
        print("other val = scored on a different validation set than the baseline, difference not meaningful")
    if unchecked:
        print("? = logged without a validation fingerprint, so the validation set could not be checked")


if __name__ == "__main__":
    main()
