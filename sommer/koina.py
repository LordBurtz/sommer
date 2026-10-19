"""Koina (https://koina.wilhelmlab.org) models as Predictors, so they are scored by the same code as ours.

Only the singly charged b/y annotations are used. Inputs the CID task does not define (collision energy,
instrument) are fixed assumptions, recorded with every score.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Sequence
from urllib.request import Request, urlopen

import numpy as np

from sommer import peptides
from sommer.constants import VECTOR_DIM

URL = "https://koina.wilhelmlab.org/v2/models/{model}/infer"
DEFAULT_MODELS = (
    "Prosit_2020_intensity_CID",
    "Prosit_2025_intensity_22PTM",
)
COLLISION_ENERGY = 35.0
FRAGMENTATION_TYPE = "CID"
INSTRUMENT_TYPE = "QE"
_ANNOTATION = re.compile(r"([by])(\d+)(?:\+1|\^1)?")

# extra inputs some models require besides sequence and charge
EXTRA_INPUTS = {
    "Prosit_2025_intensity_22PTM": ("collision_energies", "fragmentation_types"),
    "AlphaPeptDeep_ms2_generic": ("collision_energies", "instrument_types"),
    "Altimeter_2024_intensities": ("collision_energies",),
}
MAX_BATCH = {"Altimeter_2024_intensities": 100}


@dataclass
class KoinaModel:
    name: str
    batch_size: int = 1000
    timeout: int = 120
    verbose: bool = True
    assumptions: dict = field(default_factory=lambda: {"collision_energy": COLLISION_ENERGY,
                                                       "fragmentation_type": FRAGMENTATION_TYPE,
                                                       "instrument_type": INSTRUMENT_TYPE})

    def predict(self, sequences: Sequence[str], charges: Sequence[int]) -> np.ndarray:
        sequences, charges = list(sequences), [int(z) for z in charges]
        step = min(self.batch_size, MAX_BATCH.get(self.name, self.batch_size))
        out = np.zeros((len(sequences), VECTOR_DIM), dtype=np.float32)
        for s in range(0, len(sequences), step):
            out[s:s + step] = self._infer(sequences[s:s + step], charges[s:s + step])
            if self.verbose:
                print(f"\r{self.name}: {min(s + step, len(sequences)):,}/{len(sequences):,}", end="", flush=True)
        if self.verbose:
            print()
        return out

    def _infer(self, sequences: list[str], charges: list[int]) -> np.ndarray:
        n = len(sequences)
        values = {
            "peptide_sequences": ("BYTES", sequences),
            "precursor_charges": ("INT32", charges),
            "collision_energies": ("FP32", [self.assumptions["collision_energy"]] * n),
            "fragmentation_types": ("BYTES", [self.assumptions["fragmentation_type"]] * n),
            "instrument_types": ("BYTES", [self.assumptions["instrument_type"]] * n),
        }
        names = ("peptide_sequences", "precursor_charges") + EXTRA_INPUTS.get(self.name, ())
        payload = {"inputs": [{"name": k, "datatype": values[k][0], "shape": [n, 1], "data": values[k][1]}
                              for k in names]}
        request = Request(URL.format(model=self.name), data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=self.timeout) as response:
            outputs = {o["name"]: o for o in json.load(response)["outputs"]}
        annotation = outputs.get("annotation") or outputs["annotations"]
        annotations = np.asarray(annotation["data"]).reshape(annotation["shape"])
        intensities = np.asarray(outputs["intensities"]["data"], dtype=np.float64).reshape(
            outputs["intensities"]["shape"])
        return np.stack([annotations_to_vector(a, i) for a, i in zip(annotations, intensities)])


def annotations_to_vector(annotations: Sequence[str | bytes], intensities: Sequence[float]) -> np.ndarray:
    """Koina annotations (``y1+1``, ``b3^1``, ...) -> 58-slot vector; other charges/ion types are dropped"""
    ions, values = [], []
    for annotation, intensity in zip(annotations, intensities):
        if isinstance(annotation, bytes):
            annotation = annotation.decode()
        if (match := _ANNOTATION.fullmatch(annotation)) is not None:
            ions.append(match.group(1) + match.group(2))
            values.append(max(0.0, float(intensity)))   # Koina marks impossible fragments with -1
    return peptides.ions_to_vector(ions, values, normalize=False)
