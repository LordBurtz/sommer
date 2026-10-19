"""The interface every fragment-intensity model implements.

    (peptide sequence, precursor charge) -> 58 b/y intensities

Torch models subclass :class:`FragmentModel` and implement ``__init__`` + ``forward``. Everything else
(training, validation, checkpoints, comparison) only relies on this contract. Non-torch predictors
(e.g. Koina) implement the :class:`Predictor` protocol and are scored by the same evaluation code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, Sequence

import numpy as np
import torch
import torch.nn as nn

from sommer import peptides
from sommer.constants import CHARGES


class Predictor(Protocol):
    name: str

    def predict(self, sequences: Sequence[str], charges: Sequence[int]) -> np.ndarray:
        """(n, VECTOR_DIM) intensities, slot layout of peptides.fragment_slot, any non-negative scale"""
        ...


class FragmentModel(nn.Module, ABC):
    """Base class for all trainable models.

    ``forward(tokens, precursor_charge)``
      tokens            (B, MAX_PEPTIDE_LEN) int64, see peptides.encode_tokens, 0 = padding
      precursor_charge  (B,) int64, the raw charge (2, 3 or 4)
      returns           (B, VECTOR_DIM) intensities; slot = (fragment number - 1) * 2 + {b: 0, y: 1}

    Slots a peptide cannot have are masked out by loss and metrics, and both are scale invariant,
    so the output needs no normalisation. The constructor must take keyword arguments only (they are
    stored in the checkpoint to rebuild the model).
    """

    name: str = ""   # set by @register_model

    @abstractmethod
    def forward(self, tokens: torch.Tensor, precursor_charge: torch.Tensor) -> torch.Tensor:
        ...

    @staticmethod
    def charge_index(precursor_charge: torch.Tensor) -> torch.Tensor:
        """precursor charge -> 0-based index for an nn.Embedding of size len(CHARGES)"""
        return precursor_charge - min(CHARGES)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.inference_mode()
    def predict(self, sequences: Sequence[str], charges: Sequence[int], batch_size: int = 4096) -> np.ndarray:
        """Predictor protocol on raw sequences, e.g. for comparison with Koina"""
        return predict_tensors(self, peptides.encode_batch(sequences), np.asarray(charges, dtype=np.int64),
                               batch_size)


@torch.inference_mode()
def predict_tensors(model: FragmentModel, tokens: np.ndarray, charge: np.ndarray,
                    batch_size: int = 4096) -> np.ndarray:
    """batched eval-mode forward over pre-encoded inputs"""
    was_training = model.training
    model.eval()
    device = model.device
    out = [model(torch.tensor(tokens[s:s + batch_size], device=device),
                 torch.tensor(charge[s:s + batch_size], device=device)).float().cpu().numpy()
           for s in range(0, len(tokens), batch_size)]
    model.train(was_training)
    return np.concatenate(out)
