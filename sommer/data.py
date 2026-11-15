"""Train/validation data: loading, encoding, caching.

Reads ``refac/spectra_with_charge.parquet`` (output of refac/02). Only rows with split train/val and rows
whose charge was imputed by refac/02 (split = null) are ever materialised; test rows are filtered out at
the parquet level and there is no code path that returns them.
Imputed rows go to train by default; ``include_imputed=False`` restricts train to annotated spectra.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from sommer import peptides
from sommer.constants import CHARGES, MAX_PEPTIDE_LEN
from sommer.paths import CACHE_DIR, DATASET_FILE

# bump when the encoding of tokens / targets / masks changes, invalidates every cache
ENCODING_VERSION = 1
READ_COLUMNS = ["raw_file", "scan_number", "peptide_sequence", "precursor_charge",
                "matched_ions", "intensities_raw", "split", "charge_source"]
ALLOWED_SPLITS = ("train", "val")


@dataclass(frozen=True)
class SpectrumSet:
    """Encoded spectra of one split; all arrays share the first dimension."""
    sequence: np.ndarray   # (n,) str, ProForma-like with [UNIMOD:x]
    tokens: np.ndarray     # (n, MAX_PEPTIDE_LEN) int64, 0 = padding
    charge: np.ndarray     # (n,) int64 precursor charge in CHARGES
    target: np.ndarray     # (n, VECTOR_DIM) float32, base-peak normalised
    mask: np.ndarray       # (n, VECTOR_DIM) bool, fragments the peptide can have
    imputed: np.ndarray    # (n,) bool, charge predicted by refac/02

    def __len__(self) -> int:
        return len(self.charge)

    def subset(self, idx: np.ndarray) -> "SpectrumSet":
        return SpectrumSet(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})

    def sample(self, n: int | None, seed: int) -> "SpectrumSet":
        """reproducible random subset of n spectra (all if n is None or >= len)"""
        if n is None or n >= len(self):
            return self
        return self.subset(np.sort(np.random.default_rng(seed).choice(len(self), n, replace=False)))

    @property
    def length(self) -> np.ndarray:
        return (self.tokens != 0).sum(axis=1)

    @property
    def modified(self) -> np.ndarray:
        return np.array([peptides.is_modified(s) for s in self.sequence])

    def fingerprint(self) -> str:
        """content hash; equal fingerprints = scores are directly comparable"""
        h = hashlib.sha256()
        for name in ("tokens", "charge", "target"):
            a = np.ascontiguousarray(getattr(self, name))
            h.update(f"{name}:{a.shape}:{a.dtype}".encode())
            h.update(a.tobytes())
        return h.hexdigest()[:16]


def load_dataset(path: Path = DATASET_FILE, include_imputed: bool = True,
                 cache: bool = True) -> dict[str, SpectrumSet]:
    """{"train": ..., "val": ...}; never returns test rows"""
    path = Path(path).resolve()
    stat = path.stat()
    key = hashlib.sha256(f"v{ENCODING_VERSION}:{path}:{stat.st_size}:{stat.st_mtime_ns}:{CHARGES}:"
                         f"{include_imputed}".encode()).hexdigest()[:16]
    cache_paths = {split: CACHE_DIR / f"dataset_{split}_{key}.npz" for split in ALLOWED_SPLITS}
    if cache and all(p.exists() for p in cache_paths.values()):
        return {split: _load_npz(p) for split, p in cache_paths.items()}

    frame = _read_frame(path, include_imputed)
    data = {split: encode(frame[frame["role"] == split]) for split in ALLOWED_SPLITS}
    for split, part in data.items():
        print(f"{split}: {len(part):,} spectra ({int(part.imputed.sum()):,} with imputed charge)")
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for split, p in cache_paths.items():
            np.savez(p, **{f.name: getattr(data[split], f.name) for f in fields(SpectrumSet)})
    return data


def encode(frame: pd.DataFrame) -> SpectrumSet:
    seqs = frame["peptide_sequence"].tolist()
    return SpectrumSet(
        sequence=np.array(seqs, dtype=str),
        tokens=peptides.encode_batch(seqs),
        charge=frame["precursor_charge"].to_numpy(dtype=np.int64),
        target=np.stack([peptides.ions_to_vector(i, v) for i, v in
                         zip(frame["matched_ions"], frame["intensities_raw"])]).astype(np.float32),
        mask=np.stack([peptides.valid_slot_mask(s) for s in seqs]),
        imputed=(frame["charge_source"] == "predicted").to_numpy(dtype=bool),
    )


def _read_frame(path: Path, include_imputed: bool) -> pd.DataFrame:
    """train/val rows (+ imputed rows assigned to train), filtered to CHARGES, sorted deterministically"""
    annotated = pd.read_parquet(path, columns=READ_COLUMNS, filters=[("split", "in", list(ALLOWED_SPLITS))])
    if not set(annotated["split"]) <= set(ALLOWED_SPLITS):
        raise RuntimeError("non train/val rows were read")
    annotated["role"] = annotated["split"]
    parts = [annotated]
    if include_imputed:
        imputed = pd.read_parquet(path, columns=READ_COLUMNS, filters=[("charge_source", "==", "predicted")])
        if imputed["split"].notna().any():
            raise RuntimeError("imputed rows must have no split assignment")
        parts.append(imputed.assign(role="train"))
    frame = pd.concat(parts, ignore_index=True)

    excluded = ~frame["precursor_charge"].isin(CHARGES)
    if excluded.any():
        print(f"excluding {int(excluded.sum()):,} spectra with precursor charge outside {CHARGES}")
        frame = frame[~excluded]
    lengths = frame["peptide_sequence"].map(peptides.peptide_length)
    if not lengths.between(2, MAX_PEPTIDE_LEN).all():
        raise ValueError(f"peptide lengths must be within 2..{MAX_PEPTIDE_LEN}, truncation is not allowed")
    return frame.sort_values(["role", "raw_file", "scan_number"]).reset_index(drop=True)


def _load_npz(path: Path) -> SpectrumSet:
    with np.load(path, allow_pickle=False) as z:
        return SpectrumSet(**{f.name: z[f.name] for f in fields(SpectrumSet)})
