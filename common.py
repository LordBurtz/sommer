from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

MAX_PEPTIDE_LEN = 30
# A peptide of length L produces L-1 b ions and L-1 y ions.
MAX_FRAGMENTS = MAX_PEPTIDE_LEN - 1            # 29 fragment numbers
ION_TYPES = ("b", "y")
N_ION_TYPES = len(ION_TYPES)
VECTOR_DIM = MAX_FRAGMENTS * N_ION_TYPES       # 58 slots per spectrum

# 1, 6 are removed
VALID_CHARGES = (2, 3, 4, 5)

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
MOD_TOKENS = ["C[UNIMOD:4]", "M[UNIMOD:35]"]

# for the DL model
TOKENS = ["<pad>"] + AMINO_ACIDS + MOD_TOKENS
TOKEN_TO_IDX = {tok: i for i, tok in enumerate(TOKENS)}
VOCAB_SIZE = len(TOKENS)

AA_BASIC = set("KRH")
AA_ACIDIC = set("DE")

# masses taken from https://proteomicsresource.washington.edu/protocols06/masses.php
AA_MONO_MASS = {
    'A': 71.0779,'R': 156.18568000000002,'N': 114.10264000000001,
    'D': 115.08739999999999,'C': 103.1429,'E': 129.11398,'Q': 128.12922,
    'G': 57.051320000000004,'H': 137.13928,'I': 113.15763999999999,
    'L': 113.15763999999999,'K': 128.17228,'M': 131.19606,'F': 147.17386000000002,
    'P': 97.11518,'S': 87.0773,'T': 101.10388,'W': 186.2099,'Y': 163.17326,
    'V': 99.13105999999999,'O': 237.29815999999997,'U': 150.03789999999998
}

MOD_MASS = {"C[UNIMOD:4]": 57.02146, "M[UNIMOD:35]": 15.99491}

_TOKEN_RE = re.compile(r"[A-Z](?:\[UNIMOD:\d+\])?")
_MOD_RE = re.compile(r"\[UNIMOD:\d+\]")

# ---------------------------------------------------------------------------
# Data loading / cleaning
# ---------------------------------------------------------------------------

def load_clean_merged(
    data_path: str = "../data_for_student.parquet",
    meta_path: str = "../metadata_for_student.parquet",
    drop_charges: tuple[int, ...] = (1, 6),
) -> pd.DataFrame:

    df = pd.read_parquet(data_path)
    meta = pd.read_parquet(meta_path)

    df = df.drop_duplicates(["raw_file", "scan_number"], keep=False)
    meta = meta.drop_duplicates(["raw_file", "scan_number"], keep=False)

    merged = df.merge(meta, on=["raw_file", "scan_number"], how="left")
    merged = merged[~merged["precursor_charge"].isin(drop_charges)].copy()

    merged["pep_len"] = merged["peptide_sequence"].map(peptide_length)
    return merged.reset_index(drop=True)




# --------------------------------------------------------------------------- #
# Peptide parsing
# --------------------------------------------------------------------------- #

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


def encode_tokens(sequence: str, max_len: int = MAX_PEPTIDE_LEN) -> np.ndarray:
    """integer encode a peptide (0-padded)"""
    idx = np.zeros(max_len, dtype=np.int64)
    for i, tok in enumerate(parse_peptide(sequence)[:max_len]):
        idx[i] = TOKEN_TO_IDX.get(tok, TOKEN_TO_IDX.get(strip_mods(tok), 0))
    return idx


# --------------------------------------------------------------------------- #
# vector stuff
# --------------------------------------------------------------------------- #

def fragment_slot(ion_type: str, number: int) -> int:
    """map (ion_type, fragment_number) to a dense vector index

    Layout: slot = (number - 1) * 2 + {b:0, y:1}.
    """
    return (number - 1) * N_ION_TYPES + ION_TYPES.index(ion_type)


def slot_to_fragment(slot: int) -> tuple[str, int]:
    """inverse of :func:`fragment_slot`"""
    number = slot // N_ION_TYPES + 1
    ion_type = ION_TYPES[slot % N_ION_TYPES]
    return ion_type, number


def ions_to_vector(
    matched_ions: Iterable[str],
    intensities: Iterable[float],
    normalize: bool = True,
) -> np.ndarray:
    """Build a dense length-``VECTOR_DIM`` vector from the ragged storage.

    Parameters
    ----------
    matched_ions : sequence of ``"b{n}"`` / ``"y{n}"`` strings.
    intensities : matching raw intensity values.
    normalize : if True, divide by the maximum intensity (base-peak scaling)
        so the strongest fragment becomes 1.0.
    
    todo [FK] ki generiert, war honestly zu dumm das richtig zu machen
    todo: rewrite by hand?
    """
    vec = np.zeros(VECTOR_DIM, dtype=np.float32)
    intensities = np.asarray(list(intensities), dtype=np.float64)
    for ion, inten in zip(matched_ions, intensities):
        ion_type = ion[0]
        number = int(ion[1:])
        if ion_type in ION_TYPES and 1 <= number <= MAX_FRAGMENTS:
            vec[fragment_slot(ion_type, number)] = inten
    if normalize:
        m = vec.max()
        if m > 0:
            vec /= m
    return vec


def valid_slot_mask(sequence: str) -> np.ndarray:
    """boolean mask of fragments in sequence

    A length-L peptide only has fragments up to L-1, the rest is just not there
    and therefore is masked out to prevent skewing loss and evaluation metrics
    """
    L = peptide_length(sequence)
    mask = np.zeros(VECTOR_DIM, dtype=bool)
    for number in range(1, L):
        for ion_type in ION_TYPES:
            mask[fragment_slot(ion_type, number)] = True
    return mask


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def spectral_angle(
    true_vec: np.ndarray,
    pred_vec: np.ndarray,
    mask: np.ndarray | None = None,
    eps: float = 1e-8,
) -> float:
    """Spectral angle similarity (1 = identical, 0 = orthogonal)

    SA = 1 - 2 * arccos(cosine_similarity) / pi. 
    ``masks`` makes it only look at actual existing fragments
    """
    t = np.asarray(true_vec, dtype=np.float64).ravel()
    p = np.asarray(pred_vec, dtype=np.float64).ravel()
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).ravel()
        t, p = t[mask], p[mask]
    nt = np.linalg.norm(t)
    np_ = np.linalg.norm(p)
    if nt < eps or np_ < eps:
        return 0.0
    cos = np.clip(np.dot(t, p) / (nt * np_), -1.0, 1.0)
    return float(1.0 - 2.0 * np.arccos(cos) / np.pi)


def pearson_corr(
    true_vec: np.ndarray,
    pred_vec: np.ndarray,
    mask: np.ndarray | None = None,
    eps: float = 1e-8,
) -> float:
    """Pearson correlation"""
    t = np.asarray(true_vec, dtype=np.float64).ravel()
    p = np.asarray(pred_vec, dtype=np.float64).ravel()
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).ravel()
        t, p = t[mask], p[mask]
    if t.size < 2:
        return 0.0
    t = t - t.mean()
    p = p - p.mean()
    denom = np.linalg.norm(t) * np.linalg.norm(p)
    if denom < eps:
        return 0.0
    return float(np.dot(t, p) / denom)


def evaluate_vectors(
    true_mat: np.ndarray,
    pred_mat: np.ndarray,
    masks: np.ndarray,
) -> dict[str, float]:
    """Calculate mean spectral angle and pearson over a batch of spectra

    All inputs should be 2D arrays of shape ``(n_spectra, VECTOR_DIM)``
    All predictions are clipped to ``[0, 1]`` before scoring
    """
    pred_mat = np.clip(pred_mat, 0.0, 1.0)
    sa, pe = [], []
    for t, p, m in zip(true_mat, pred_mat, masks):
        sa.append(spectral_angle(t, p, m))
        pe.append(pearson_corr(t, p, m))
    return {
        "spectral_angle": float(np.mean(sa)),
        "pearson": float(np.mean(pe)),
        "n": int(len(sa)),
    }


# --------------------------------------------------------------------------- #
# Output formatting
# --------------------------------------------------------------------------- #

def prediction_table(
    sequence: str,
    pred_vec: np.ndarray,
    drop_zero: bool = False,
) -> pd.DataFrame:
    """
    Render not masked fragments from dense prediction vector into output format
    """
    mask = valid_slot_mask(sequence)
    rows = []
    for slot in np.where(mask)[0]:
        ion_type, number = slot_to_fragment(int(slot))
        intensity = float(np.clip(pred_vec[slot], 0.0, 1.0))
        if drop_zero and intensity <= 0:
            continue
        rows.append(
            {
                "Fragment ion": ion_type.upper(),
                "Fragment Number": number,
                "Intensity": intensity,
            }
        )
    out = pd.DataFrame(rows, columns=["Fragment ion", "Fragment Number", "Intensity"])
    return out.sort_values(["Fragment ion", "Fragment Number"]).reset_index(drop=True)

