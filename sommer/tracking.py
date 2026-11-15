"""Run log (JSON lines) and checkpoints.

One record per run in results/run_log.jsonl (full runs, migrated legacy runs, Koina scores);
smoke runs and search trials go to results/preliminary_runs.jsonl. Record fields:

  run_id, timestamp, source ("pipeline" | "koina" | "legacy:train_deep" | "legacy:08"), mode,
  model, model_params, n_params, train_config, data {include_imputed, n_train, n_val, train_fingerprint,
  val_fingerprint}, val_sa, val_pearson, by_charge {"2": {sa, pearson, n}, ...}, buckets,
  epochs_trained, best_epoch, seconds, device, git_commit, checkpoint, history
"""
from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch

from sommer.models.base import FragmentModel
from sommer.models.registry import build_model
from sommer.paths import CHECKPOINT_DIR, ROOT, RUN_LOG


def new_run_id(started: datetime) -> str:
    return f"{started:%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True,
                                        cwd=ROOT, stderr=subprocess.DEVNULL).strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def append_record(record: dict, path: Path = RUN_LOG) -> None:
    payload = json.dumps(record, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(payload + "\n")


def load_records(path: Path = RUN_LOG) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def save_checkpoint(model: FragmentModel, record: dict, path: Path) -> Path:
    """state dict + the run record (model name and constructor params rebuild the model); atomic write"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "record": record}, tmp)
    tmp.replace(path)
    return path


def checkpoint_path(run_id: str, model_name: str, folder: Path = CHECKPOINT_DIR) -> Path:
    return Path(folder) / f"{model_name}_{run_id}.pt"


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[FragmentModel, dict]:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    record = saved["record"]
    model, _ = build_model(record["model"], record["model_params"])
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval(), record
