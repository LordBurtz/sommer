import numpy as np

from sommer import metrics
from sommer.constants import VECTOR_DIM


def reference_sa(t, p, m, eps=1e-8):
    """legacy/common.py spectral_angle, one spectrum at a time"""
    t, p = t[m].astype(np.float64), p[m].astype(np.float64)
    nt, np_ = np.linalg.norm(t), np.linalg.norm(p)
    if nt < eps or np_ < eps:
        return 0.0
    return 1.0 - 2.0 * np.arccos(np.clip(t @ p / (nt * np_), -1, 1)) / np.pi


def reference_pearson(t, p, m, eps=1e-8):
    t, p = t[m].astype(np.float64), p[m].astype(np.float64)
    if t.size < 2:
        return 0.0
    t, p = t - t.mean(), p - p.mean()
    d = np.linalg.norm(t) * np.linalg.norm(p)
    return 0.0 if d < eps else float(t @ p / d)


def random_spectra(n=200, seed=0):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(7, 31, n)
    mask = np.arange(VECTOR_DIM)[None, :] < 2 * (lengths[:, None] - 1)
    target = rng.random((n, VECTOR_DIM)) * mask
    pred = rng.random((n, VECTOR_DIM)) * 3.0           # unnormalised, garbage in padded slots
    pred[0] = 0.0                                      # all-zero prediction scores 0
    return target, pred, mask


def test_vectorised_matches_reference():
    target, pred, mask = random_spectra()
    sa, r = metrics.spectral_angle(target, pred, mask), metrics.pearson(target, pred, mask)
    np.testing.assert_allclose(sa, [reference_sa(*x) for x in zip(target, pred, mask)], atol=1e-10)
    np.testing.assert_allclose(r, [reference_pearson(*x) for x in zip(target, pred, mask)], atol=1e-10)
    assert sa[0] == 0.0


def test_scale_invariant_and_perfect():
    target, pred, mask = random_spectra()
    np.testing.assert_allclose(metrics.spectral_angle(target, pred, mask),
                               metrics.spectral_angle(target, 7 * pred, mask), atol=1e-10)
    np.testing.assert_allclose(metrics.spectral_angle(target[1:], target[1:] * 5, mask[1:]), 1.0, atol=1e-6)


def test_summarize_absent_charge_is_none():
    out = metrics.summarize(np.array([0.5, 0.7]), np.array([0.1, 0.3]), np.array([2, 2]))
    assert out["by_charge"]["3"] == {"sa": None, "pearson": None, "n": 0}
    assert abs(out["val_sa"] - 0.6) < 1e-12 and out["by_charge"]["2"]["n"] == 2
