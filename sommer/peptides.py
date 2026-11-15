"""Peptide parsing, token encoding and the dense 58-slot b/y intensity vector."""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from sommer.constants import (AA_MONO_MASS, ION_TYPES, MAX_FRAGMENTS, MAX_PEPTIDE_LEN, MOD_MASS, N_ION_TYPES,
                              TOKEN_TO_IDX, VECTOR_DIM, WATER_MASS)

_TOKEN_RE = re.compile(r"[A-Z](?:\[UNIMOD:\d+\])?")
_MOD_RE = re.compile(r"\[UNIMOD:\d+\]")


def parse_peptide(sequence: str) -> list[str]:
    """split a peptide into AAs and modifications

    ``"SM[UNIMOD:35]ASK"`` -> ``["S", "M[UNIMOD:35]", "A", "S", "K"]``
    """
    return _TOKEN_RE.findall(sequence)


def strip_mods(sequence: str) -> str:
    """returns only the AAs, strips the modifications"""
    return _MOD_RE.sub("", sequence)


def peptide_length(sequence: str) -> int:
    """returns number of AAs WITHOUT modifications"""
    return len(strip_mods(sequence))


def is_modified(sequence: str) -> bool:
    return _MOD_RE.search(sequence) is not None


def peptide_mass(sequence: str) -> float:
    """monoisotopic mass of a (possibly modified) peptide"""
    tokens = parse_peptide(sequence)
    mass = sum(AA_MONO_MASS[t[0]] for t in tokens) + WATER_MASS
    return mass + sum(MOD_MASS[t] for t in tokens if t in MOD_MASS)


def encode_tokens(sequence: str, max_len: int = MAX_PEPTIDE_LEN) -> np.ndarray:
    """integer encode a peptide (0-padded)"""
    tokens = parse_peptide(sequence)
    if len(tokens) > max_len:
        raise ValueError(f"{sequence!r} is longer than {max_len} residues")
    idx = np.zeros(max_len, dtype=np.int64)
    for i, tok in enumerate(tokens):
        idx[i] = TOKEN_TO_IDX[tok]
    return idx


def encode_batch(sequences: Iterable[str]) -> np.ndarray:
    return np.stack([encode_tokens(s) for s in sequences])


# --------------------------------------------------------------------------- #
# vector layout
# --------------------------------------------------------------------------- #

def fragment_slot(ion_type: str, number: int) -> int:
    """map (ion_type, fragment_number) to a dense vector index

    Layout: slot = (number - 1) * 2 + {b:0, y:1}.
    """
    return (number - 1) * N_ION_TYPES + ION_TYPES.index(ion_type)


def slot_to_fragment(slot: int) -> tuple[str, int]:
    """inverse of :func:`fragment_slot`"""
    return ION_TYPES[slot % N_ION_TYPES], slot // N_ION_TYPES + 1


def ions_to_vector(matched_ions: Iterable[str], intensities: Iterable[float], normalize: bool = True) -> np.ndarray:
    """dense length-``VECTOR_DIM`` vector from ragged ``"b{n}"``/``"y{n}"`` annotations

    ``normalize`` divides by the base peak so the strongest fragment becomes 1.0.
    """
    vec = np.zeros(VECTOR_DIM, dtype=np.float32)
    for ion, intensity in zip(matched_ions, intensities):
        ion_type, number = ion[0], int(ion[1:])
        if ion_type in ION_TYPES and 1 <= number <= MAX_FRAGMENTS:
            vec[fragment_slot(ion_type, number)] = intensity
    if normalize and vec.max() > 0:
        vec /= vec.max()
    return vec


def valid_slot_mask(sequence: str) -> np.ndarray:
    """boolean mask of the fragments a peptide can have

    A length-L peptide only has fragments up to L-1, the rest is masked out of loss and metrics.
    """
    mask = np.zeros(VECTOR_DIM, dtype=bool)
    mask[: (peptide_length(sequence) - 1) * N_ION_TYPES] = True
    return mask


def prediction_table(sequence: str, pred_vec: np.ndarray, drop_zero: bool = False) -> pd.DataFrame:
    """render the existing fragments of a dense prediction into the task's output format"""
    rows = []
    for slot in np.flatnonzero(valid_slot_mask(sequence)):
        ion_type, number = slot_to_fragment(int(slot))
        intensity = float(max(pred_vec[slot], 0.0))
        if drop_zero and intensity <= 0:
            continue
        rows.append({"Fragment ion": ion_type.upper(), "Fragment Number": number, "Intensity": intensity})
    out = pd.DataFrame(rows, columns=["Fragment ion", "Fragment Number", "Intensity"])
    return out.sort_values(["Fragment ion", "Fragment Number"]).reset_index(drop=True)
