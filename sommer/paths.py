"""Filesystem layout. All paths are absolute so code works from the repo root and from refac/."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "refac"
# written by refac/02: train + val rows plus rows with imputed charge (split = null), never test
DATASET_FILE = DATA_DIR / "spectra_with_charge.parquet"

RESULTS_DIR = ROOT / "results"
RUN_LOG = RESULTS_DIR / "run_log.jsonl"                    # full runs, migrated legacy runs, Koina
PRELIMINARY_LOG = RESULTS_DIR / "preliminary_runs.jsonl"   # smoke runs and search trials
CURVE_DIR = RESULTS_DIR / "curves"
CHECKPOINT_DIR = ROOT / "checkpoints"
CACHE_DIR = ROOT / "cache"
