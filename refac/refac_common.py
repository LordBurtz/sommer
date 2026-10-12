"""Shared helpers for the `refac/` notebooks.

Self-contained on purpose: nothing outside `refac/` is imported.
"""

from __future__ import annotations

import re

CLASSES = (2, 3, 4)

AA_BASIC = set("KRH")
AA_ACIDIC = set("DE")

# masses taken from https://proteomicsresource.washington.edu/protocols06/masses.php
AA_MONO_MASS = {
    "A": 71.037114, "R": 156.101111, "N": 114.042927, "D": 115.026943,
    "C": 103.009185, "E": 129.042593, "Q": 128.058578, "G": 57.021464,
    "H": 137.058912, "I": 113.084064, "L": 113.084064, "K": 128.094963,
    "M": 131.040485, "F": 147.068414, "P": 97.052764, "S": 87.032028,
    "T": 101.047679, "W": 186.079313, "Y": 163.063329, "V": 99.068414,
    "O": 237.147727, "U": 150.953633,
}

MOD_MASS = {"C[UNIMOD:4]": 57.02146, "M[UNIMOD:35]": 15.99491}

WATER_MASS = 18.01056

_TOKEN_RE = re.compile(r"[A-Z](?:\[UNIMOD:\d+\])?")
_MOD_RE = re.compile(r"\[UNIMOD:\d+\]")


def parse_peptide(sequence: str) -> list[str]:
    """Split a peptide into amino acids and modifications.

    ``"SM[UNIMOD:35]ASK"`` -> ``["S", "M[UNIMOD:35]", "A", "S", "K"]``
    """
    return _TOKEN_RE.findall(sequence)


def strip_mods(sequence: str) -> str:
    """Return the bare amino-acid sequence, modifications removed."""
    return _MOD_RE.sub("", sequence)


def peptide_length(sequence: str) -> int:
    """Number of amino acids, ignoring modifications."""
    return len(strip_mods(sequence))


def peptide_mass(sequence: str) -> float:
    """Monoisotopic mass of a (possibly modified) peptide."""
    tokens = parse_peptide(sequence)
    mass = sum(AA_MONO_MASS[t[0]] for t in tokens) + WATER_MASS
    return mass + sum(MOD_MASS[t] for t in tokens if t in MOD_MASS)
