"""Figures for training runs and comparisons. Built on matplotlib.figure.Figure (no pyplot state), so they
work headless and display inline in notebooks when returned."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

LOSS_COLOR, VAL_COLOR, MUTED = "#4c72b0", "#dd8452", "grey"


def plot_curve(record: dict, path: Path | None = None) -> Figure:
    """train loss and val SA per epoch, best epoch marked"""
    history, best = record["history"], record["best_epoch"]
    epochs = [h["epoch"] for h in history]
    fig = Figure(figsize=(7, 4), layout="tight")
    ax1 = fig.add_subplot()
    ax1.plot(epochs, [h["train_loss"] for h in history], color=LOSS_COLOR, marker="o", ms=3)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color=LOSS_COLOR)
    ax2 = ax1.twinx()
    ax2.plot(epochs, [h["val_sa"] for h in history], color=VAL_COLOR, marker="o", ms=3)
    ax2.set_ylabel("val spectral angle", color=VAL_COLOR)
    ax2.axvline(best, color=MUTED, ls="--", lw=1)
    ax1.set_title(f"{record['model']}: best val SA {record['val_sa']:.4f} @ epoch {best}")
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
    return fig


def plot_charge_comparison(table: pd.DataFrame, label_column: str = "label") -> Figure:
    """grouped bars: val SA per precursor charge for each row of a leaderboard table"""
    charges = [c for c in ("z=2", "z=3", "z=4") if c in table]
    fig = Figure(figsize=(max(6, 1.2 * len(table) * len(charges) / 2), 4), layout="tight")
    ax = fig.add_subplot()
    width = 0.8 / max(len(table), 1)
    for i, (_, row) in enumerate(table.iterrows()):
        xs = [j + i * width for j in range(len(charges))]
        ax.bar(xs, [row[c] for c in charges], width=width, label=row[label_column])
    ax.set_xticks([j + width * (len(table) - 1) / 2 for j in range(len(charges))], charges)
    ax.set_ylabel("mean val spectral angle")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="y", alpha=0.25)
    return fig


def plot_bucket_comparison(buckets: pd.DataFrame, charge: int = 2) -> Figure:
    """mean SA error (1 - SA) by length bin for one charge, one line per model and modified state

    ``buckets``: long table with columns label, precursor_charge, length_bin, modified, n, mean_sa
    """
    frame = buckets[(buckets["precursor_charge"] == charge) & (buckets["n"] > 0)]
    fig = Figure(figsize=(9, 4.5), layout="constrained")
    ax = fig.add_subplot()
    colors = {label: f"C{i}" for i, label in enumerate(frame["label"].unique())}
    for (label, modified), group in frame.groupby(["label", "modified"], sort=False):
        unmodified = modified == "no"
        ax.plot(group["length_bin"], 1 - group["mean_sa"], color=colors[label], marker="o" if unmodified else "s",
                ls="-" if unmodified else "--", label=f"{label} ({'unmod' if unmodified else 'mod'})")
    ax.set(xlabel="peptide length", ylabel="mean SA error (1 - SA)", title=f"precursor charge {charge}")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    return fig
