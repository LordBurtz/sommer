"""One-off conversion of the old run logs into results/run_log.jsonl.

  results/runs.jsonl                  legacy/train_deep.py (PrositLike, GRU)
  refac/results/pipeline_runs.jsonl   legacy/08_training_pipeline.ipynb

Only runs scored on the val split (train_deep --val-fold 1) are migrated; idempotent via run_id.
Legacy runs trained on annotated spectra only.

Scores logged before 2026-10-16 (commit 7c4f171) clipped predictions to [0, 1], later ones are scale free
like sommer.metrics. When a run's checkpoint still exists it is therefore re-scored on the current val
set with the shared evaluation (exact reproduction was verified for all existing checkpoints); the old
number is kept as logged_val_sa. Runs without checkpoint keep their logged scores (val_set "legacy").

  uv run python -m sommer.legacy_logs
"""
from __future__ import annotations

from pathlib import Path

import torch

from sommer import evaluation, tracking
from sommer.data import SpectrumSet
from sommer.models import FragmentModel, build_model
from sommer.paths import DATA_DIR, RESULTS_DIR, ROOT, RUN_LOG

TRAIN_DEEP_LOG = RESULTS_DIR / "runs.jsonl"
PIPELINE_LOG = DATA_DIR / "results" / "pipeline_runs.jsonl"


def from_train_deep(r: dict) -> dict | None:
    if r.get("val_fold", 1) != 1:
        return None
    hp = r["hyperparameters"]
    return {
        "run_id": f"legacy_{r['tag']}_{r['timestamp']}", "timestamp": r["timestamp"],
        "source": "legacy:train_deep", "mode": "full", "model": r["model"],
        "model_params": {k: hp[k] for k in ("emb", "hidden", "head", "activation") if k in hp}
                        | ({"use_frag": True} if hp.get("use_frag") else {}),
        "n_params": r["n_params"], "train_config": hp,
        "data": {"n_train": r["n_train"], "n_train_imputed": 0, "n_val": r["n_val"],
                 "train_fingerprint": None, "val_fingerprint": None},
        "val_sa": r["val_spectral_angle"], "val_pearson": r["val_pearson"], "n_val": r["n_val"],
        "by_charge": {z: {"sa": v["spectral_angle"], "pearson": v["pearson"], "n": v["n"]}
                      for z, v in r["by_charge"].items()},
        "epochs_trained": r["epochs_trained"], "best_epoch": r["best_epoch"],
        "seconds": round(r["train_minutes"] * 60, 1), "device": r.get("device"), "git_commit": r["git_commit"],
        "checkpoint": r["checkpoint"], "tag": r["tag"], "history": r.get("history", []),
    }


def from_pipeline(r: dict) -> dict | None:
    if r.get("mode") != "full":
        return None
    return {
        "run_id": f"legacy_{r['run_id']}", "timestamp": r["timestamp"], "source": "legacy:08", "mode": "full",
        "model": r["model"], "model_params": r["model_params"], "n_params": r["params"],
        "train_config": r["config"],
        "data": {"n_train": r["n_train"], "n_train_imputed": 0, "n_val": r["n_val"],
                 "train_fingerprint": None, "val_fingerprint": None},
        "val_sa": r["val SA"], "val_pearson": None, "n_val": r["n_val"],
        "by_charge": {str(z): {"sa": r.get(f"z={z}"), "pearson": None, "n": None} for z in (2, 3, 4)},
        "epochs_trained": r["epochs trained"], "best_epoch": r["best_epoch"], "seconds": r["seconds"],
        "device": None, "git_commit": r["git_commit"], "checkpoint": r["checkpoint"], "history": [],
    }


def load_legacy_checkpoint(path: Path) -> tuple[FragmentModel, dict] | None:
    """train_deep ({state_dict, config}) or 08 ({state_dict, record}) checkpoint -> model; the
    architecture is read from the state dict because some train_deep records mislabel it"""
    path = Path(path) if Path(path).is_absolute() else ROOT / path
    if not path.is_file() or path.stat().st_size == 0:
        return None
    saved = torch.load(path, map_location="cpu", weights_only=False)
    state, cfg = saved["state_dict"], saved.get("config") or saved["record"].get("model_params", {})
    if "frag_emb.weight" in state:
        return None                      # acquisition method as input: not part of the interface anymore
    if "tok_emb.weight" in state:
        name, params = "PrositLike", {k: cfg[k] for k in ("emb", "hidden", "head", "activation") if k in cfg}
    else:
        name = "PeptideEncoder"
        params = {"hidden_dimensions": state["amino_acid_embedding.weight"].shape[1],
                  "precursor_vocab_size": state["precursor_charge_embedding.weight"].shape[0],
                  "transformer_layers": len({k.split(".")[3] for k in state if k.startswith("sequence_transformer.encoder.layers.")})}
        params |= {k: cfg[k] for k in ("attention_heads",) if k in cfg}
    model, params = build_model(name, params)
    model.load_state_dict(state)
    return model.eval(), params


def rescore(record: dict, val: SpectrumSet) -> dict:
    loaded = load_legacy_checkpoint(record["checkpoint"]) if record.get("checkpoint") else None
    if loaded is None:
        return record
    model, params = loaded
    scores = evaluation.evaluate(model, val)
    return record | {"model": model.name, "model_params": params, "logged_val_sa": record["val_sa"], **scores,
                     "data": record["data"] | {"n_val": len(val), "val_fingerprint": val.fingerprint()}}


def migrate(val: SpectrumSet | None = None, run_log: Path = RUN_LOG) -> int:
    """append converted legacy runs to the run log; re-score them on `val` when given"""
    known = {r["run_id"] for r in tracking.load_records(run_log)}
    added = skipped = 0
    for path, convert in ((TRAIN_DEEP_LOG, from_train_deep), (PIPELINE_LOG, from_pipeline)):
        for old in tracking.load_records(path):
            new = convert(old)
            if new is None:
                skipped += 1
            elif new["run_id"] not in known:
                new = rescore(new, val) if val is not None else new
                tracking.append_record(new, run_log)
                known.add(new["run_id"])
                added += 1
    print(f"migrated {added} legacy runs into {run_log} ({skipped} not scored on the val split, skipped)")
    return added


if __name__ == "__main__":
    from sommer.data import load_dataset
    migrate(load_dataset()["val"])
