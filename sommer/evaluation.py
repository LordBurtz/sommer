"""Score predictions on a SpectrumSet. Shared by the trainer, checkpoint re-evaluation and Koina."""
from __future__ import annotations

import numpy as np

from sommer import buckets, metrics
from sommer.data import SpectrumSet
from sommer.models.base import FragmentModel, Predictor, predict_tensors


def predict(model: FragmentModel | Predictor, data: SpectrumSet, batch_size: int = 4096) -> np.ndarray:
    if isinstance(model, FragmentModel):
        return predict_tensors(model, data.tokens, data.charge, batch_size)
    return model.predict(data.sequence.tolist(), data.charge.tolist())


def score(pred: np.ndarray, data: SpectrumSet, with_buckets: bool = True) -> dict:
    """val_sa, val_pearson, n_val, by_charge (+ buckets) as stored in the run log"""
    sa = metrics.spectral_angle(data.target, pred, data.mask)
    result = metrics.summarize(sa, metrics.pearson(data.target, pred, data.mask), data.charge)
    if with_buckets:
        result["buckets"] = buckets.to_records(buckets.bucket_table(sa, data.charge, data.length, data.modified))
    return result


def evaluate(model: FragmentModel | Predictor, data: SpectrumSet, with_buckets: bool = True) -> dict:
    return score(predict(model, data), data, with_buckets)
