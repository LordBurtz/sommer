"""figures for the report. Every number is read from the result files, none is typed in.

  results/figures/val_steps.png        validation spectral angle per model step   <- results/runs.jsonl
  results/figures/test_by_charge.png   test SA per charge, pooled sigmoid baseline
                                       vs final per-bond model                    <- results/test_results.json
  results/figures/test_examples.png    good / average / poor test spectrum of the
                                       final model as mirror plots                <- results/test_results.json
Steps are matched on head / use_frag / activation (as in compare_runs.py); several runs of one step
(seeds) are averaged and shown as dots. test_results.json comes from evaluate_test.py.
Usage:
  uv run python make_figures.py
  uv run python make_figures.py --runs results/smoke_runs.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as c

# (label, head, use_frag, activation)
STEPS = [
    ("baseline, softplus", "pooled", False, "softplus"),
    ("baseline, sigmoid", "pooled", False, "sigmoid"),
    ("pooled + frag, sigmoid", "pooled", True, "sigmoid"),
    ("per-bond, sigmoid", "bond", False, "sigmoid"),
    ("per-bond + frag, sigmoid", "bond", True, "sigmoid"),
]
BASELINE, FINAL = STEPS[1], STEPS[-1]
CHARGES = ("2", "3", "4")
STEP_KEYS = ("seed", "head", "use_frag", "activation")

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"


class FigureError(Exception):
    pass


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 9.5, "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 10.5, "axes.titlelocation": "left", "axes.titleweight": "semibold",
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "axes.edgecolor": AXIS, "axes.linewidth": 1, "grid.color": GRID, "grid.linewidth": 1,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })


def options(entry: dict) -> tuple:
    hp = entry.get("hyperparameters", entry)        # runs.jsonl nests them, test_results.json doesn't
    return hp.get("head", "pooled"), hp.get("use_frag", False), hp.get("activation", "softplus")


def matching(entries: list[dict], step: tuple) -> list[dict]:
    return [e for e in entries if options(e) == step[1:]]


def flags(step: tuple) -> str:
    _, head, use_frag, activation = step
    f = [f"--head {head}" * (head != "pooled"), "--use-frag" * use_frag,
         f"--activation {activation}" * (activation != "softplus")]
    return " ".join(x for x in f if x) or "no flags"


def read(path: Path):
    if not path.exists():
        raise FigureError(f"{path} not found")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in open(path) if line.strip()]
    return json.loads(path.read_text())


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def dots(ax, x, y, horizontal=False) -> None:
    ax.scatter(y if horizontal else x, x if horizontal else y, s=22, color=INK,
               edgecolor=SURFACE, linewidth=1.5, zorder=3)


def fig_val_steps(path: Path):
    runs = read(path)
    groups = [matching(runs, s) for s in STEPS]
    missing = [f"{s[0]} ({flags(s)})" for s, g in zip(STEPS, groups) if not g]
    if missing:
        raise FigureError(f"{path} has no run for: " + "; ".join(missing))
    if len({r.get("val_fingerprint") for g in groups for r in g}) > 1:
        raise FigureError(f"the runs in {path} were not all scored on the same validation set (val_fingerprint)")
    hps = [r["hyperparameters"] for g in groups for r in g]
    differ = sorted(k for k in set().union(*hps)
                    if k not in STEP_KEYS and len({json.dumps(h.get(k)) for h in hps}) > 1)
    for s, g in zip(STEPS, groups):
        print(f"  {s[0]}: {len(g)} run(s), seeds {sorted(r['hyperparameters'].get('seed') for r in g)}")
    if differ:
        print(f"  note: the runs also differ in {', '.join(differ)}, so this is not a controlled comparison")

    vals = [[r["val_spectral_angle"] for r in g] for g in groups]
    means = [float(np.mean(v)) for v in vals]
    y = np.arange(len(STEPS))[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.barh(y, means, height=0.5, color=BLUE, zorder=2)
    for yi, v, m in zip(y, vals, means):
        if len(v) > 1:
            dots(ax, [yi] * len(v), v, horizontal=True)
        delta = "" if yi == y[0] else f"   {m - means[0]:+.4f}"
        ax.text(max(v) + 0.012, yi, f"{m:.4f}{delta}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, [s[0] for s in STEPS])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_xlabel("validation spectral angle (1 = identical spectra)")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)
    ax.set_title("Validation spectral angle per model step", pad=20)
    multi = any(len(v) > 1 for v in vals)
    ax.text(0, 1.03, f"n = {groups[0][0]['n_val']:,} validation spectra · difference to the first step"
            + (" · bar = mean, dots = single runs (seeds)" if multi else ""),
            transform=ax.transAxes, color=INK_2, fontsize=8.5)
    return fig


def fig_test_by_charge(path: Path):
    models = read(path)["models"]
    series = [(BASELINE, BLUE), (FINAL, ORANGE)]
    groups = [matching(models, s) for s, _ in series]
    missing = [s[0] for (s, _), g in zip(series, groups) if not g]
    if missing:
        raise FigureError(f"{path} has no checkpoint for: {', '.join(missing)} (pass it to evaluate_test.py)")

    x = np.arange(len(CHARGES))
    w, gap = 0.34, 0.02
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k, (((label, *_), color), g) in enumerate(zip(series, groups)):
        vals = np.array([[m["by_charge"][z]["spectral_angle"] for z in CHARGES] for m in g])
        pos = x + (k - 0.5) * (w + gap)
        mean = vals.mean(axis=0)
        name = label + (" (pooled baseline)" if k == 0 else " (final)") + (f", mean of {len(g)}" if len(g) > 1 else "")
        ax.bar(pos, mean, width=w, color=color, label=name, zorder=2)
        for p, col, m in zip(pos, vals.T, mean):
            if len(col) > 1:
                dots(ax, [p] * len(col), col)
            ax.text(p, col.max() + 0.015, f"{m:.3f}", ha="center", va="bottom", color=INK, fontsize=8.5)
    n = groups[0][0]["by_charge"]
    ax.set_xticks(x, [f"z = {z}\nn = {n[z]['n']:,}" for z in CHARGES])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("test spectral angle")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Test spectral angle per precursor charge", pad=34)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, fontsize=8.5, handlelength=1.2)
    return fig


def fig_examples(path: Path):
    final = matching(read(path)["models"], FINAL)
    if not final:
        raise FigureError(f"{path} has no checkpoint for: {FINAL[0]} (pass it to evaluate_test.py)")
    model = final[0]
    if "examples" not in model:
        raise FigureError(f"{path} has no example spectra, rerun evaluate_test.py")
    print(f"  example spectra of {model['checkpoint']}")

    fig, axes = plt.subplots(len(model["examples"]), 1, figsize=(8, 2.7 * len(model["examples"])))
    for ax, ex in zip(np.atleast_1d(axes), model["examples"]):
        mask = c.valid_slot_mask(ex["peptide_sequence"]).astype(bool)
        meas = np.asarray(ex["measured"]) * mask
        pred = np.asarray(ex["predicted"]) * mask
        pred = pred / pred.max() if pred.max() > 0 else pred
        num = np.arange(1, mask.sum() // 2 + 1)
        for ion, color, off in (("b", BLUE, -0.18), ("y", ORANGE, 0.18)):
            slots = [c.fragment_slot(ion, k) for k in num]
            ax.bar(num + off, meas[slots], width=0.32, color=color, label=f"{ion} ions", zorder=2)
            ax.bar(num + off, -pred[slots], width=0.32, color=color, zorder=2)
        ax.axhline(0, color=AXIS, lw=1, zorder=3)
        ax.set_ylim(-1.12, 1.12)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1], ["1", "0.5", "0", "0.5", "1"])
        ax.set_xticks(num)
        ax.set_xlim(0.4, num[-1] + 0.6)
        ax.tick_params(length=0)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.spines[["left", "bottom"]].set_visible(False)
        for yv, word in ((0.5, "measured"), (-0.5, "predicted")):
            ax.annotate(word, xy=(1.01, yv), xycoords=("axes fraction", "data"), va="center", color=INK_2, fontsize=8.5)
        ax.set_title(f"{ex['label']}: SA {ex['spectral_angle']:.3f} ({ordinal(round(ex['quantile'] * 100))} percentile)")
        ax.set_title(f"{ex['peptide_sequence']} · z = {ex['precursor_charge']} · {ex['frag']}", loc="right",
                     color=INK_2, fontsize=8.5, fontweight="normal")
    np.atleast_1d(axes)[-1].set_xlabel("ion number")
    fig.supylabel("relative intensity", color=INK_2, fontsize=9)
    fig.suptitle("Test spectra of the final model: measured up, predicted down",
                 x=0.01, y=0.995, ha="left", fontsize=11, fontweight="semibold")
    fig.text(0.01, 0.962, f"{model['tag']} · each side scaled to its highest peak", color=INK_2, fontsize=8.5)
    fig.legend(*np.atleast_1d(axes)[0].get_legend_handles_labels(), loc="upper left", ncol=2, fontsize=8.5,
               bbox_to_anchor=(0.005, 0.955), handlelength=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("results/runs.jsonl"))
    ap.add_argument("--test", type=Path, default=Path("results/test_results.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    args = ap.parse_args(argv)

    style()
    figures = {
        "val_steps.png": lambda: fig_val_steps(args.runs),
        "test_by_charge.png": lambda: fig_test_by_charge(args.test),
        "test_examples.png": lambda: fig_examples(args.test),
    }
    errors = []
    for name, make in figures.items():
        print(name)
        try:
            fig = make()
        except FigureError as e:
            errors.append(f"{name}: {e}")
            continue
        args.out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_dir / name, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {args.out_dir / name}")
    if errors:
        sys.exit("skipped:\n  " + "\n  ".join(errors))


if __name__ == "__main__":
    main()
