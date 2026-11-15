"""Spectrum similarity metrics. One implementation, used by training, comparison and Koina scoring.

All metrics are computed per spectrum over the valid (masked) slots and are invariant to the scale of
the prediction, so models may output unnormalised intensities.
"""
from __future__ import annotations

import numpy as np

from sommer.constants import CHARGES

EPS = 1e-8


def spectral_angle(target: np.ndarray, pred: np.ndarray, mask: np.ndarray, eps: float = EPS) -> np.ndarray:
    """per-spectrum spectral angle similarity, SA = 1 - 2 * arccos(cos) / pi (1 = identical, 0 = orthogonal)

    Inputs are ``(n, VECTOR_DIM)``; a spectrum with an all-zero prediction or target scores 0.
    """
    t, p = _masked(target, pred, mask)
    nt, np_ = np.linalg.norm(t, axis=1), np.linalg.norm(p, axis=1)
    ok = (nt >= eps) & (np_ >= eps)
    cos = np.clip((t * p).sum(axis=1) / np.where(ok, nt * np_, 1.0), -1.0, 1.0)
    return np.where(ok, 1.0 - 2.0 * np.arccos(cos) / np.pi, 0.0)


def pearson(target: np.ndarray, pred: np.ndarray, mask: np.ndarray, eps: float = EPS) -> np.ndarray:
    """per-spectrum Pearson correlation over the valid slots"""
    t, p = _masked(target, pred, mask)
    m = np.asarray(mask, dtype=bool)
    n = m.sum(axis=1)
    t = np.where(m, t - t.sum(axis=1, keepdims=True) / np.maximum(n, 1)[:, None], 0.0)
    p = np.where(m, p - p.sum(axis=1, keepdims=True) / np.maximum(n, 1)[:, None], 0.0)
    denom = np.linalg.norm(t, axis=1) * np.linalg.norm(p, axis=1)
    ok = (n >= 2) & (denom >= eps)
    return np.where(ok, (t * p).sum(axis=1) / np.where(ok, denom, 1.0), 0.0)


def summarize(sa: np.ndarray, r: np.ndarray, charge: np.ndarray) -> dict:
    """mean SA / Pearson overall and per precursor charge (absent charge -> None, not 0)"""
    by_charge = {}
    for z in CHARGES:
        sel = charge == z
        by_charge[str(z)] = ({"sa": float(sa[sel].mean()), "pearson": float(r[sel].mean()), "n": int(sel.sum())}
                             if sel.any() else {"sa": None, "pearson": None, "n": 0})
    return {"val_sa": float(sa.mean()), "val_pearson": float(r.mean()), "n_val": int(len(sa)),
            "by_charge": by_charge}


def _masked(target, pred, mask) -> tuple[np.ndarray, np.ndarray]:
    target, pred = np.asarray(target, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    if target.shape != pred.shape:
        raise ValueError(f"prediction shape {pred.shape} != target shape {target.shape}")
    if not np.isfinite(pred).all():
        raise ValueError("predictions contain NaN/inf")
    m = np.asarray(mask, dtype=bool)
    return target * m, pred * m
