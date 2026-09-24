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
    "A": 71.0779, "R": 156.18568000000002, "N": 114.10264000000001,
    "D": 115.08739999999999, "C": 103.1429, "E": 129.11398, "Q": 128.12922,
    "G": 57.051320000000004, "H": 137.13928, "I": 113.15763999999999,
    "L": 113.15763999999999, "K": 128.17228, "M": 131.19606, "F": 147.17386000000002,
    "P": 97.11518, "S": 87.0773, "T": 101.10388, "W": 186.2099, "Y": 163.17326,
    "V": 99.13105999999999, "O": 237.29815999999997, "U": 150.03789999999998,
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
