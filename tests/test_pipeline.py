"""End to end on fabricated spectra: data guard, training, logging, checkpoints, search, comparison."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from sommer import compare, data as data_module, evaluation, koina, legacy_logs, tracking
from sommer.constants import VECTOR_DIM
from sommer.data import load_dataset
from sommer.training import TrainConfig, Trainer, search

SEQS = ["GASPK", "AC[UNIMOD:4]DM[UNIMOD:35]K", "LTQETNRV", "PEPTIDEK", "ACDKLMNPQR", "GGGGSSSSK", "MMKR"]
TEST_SEQ = "WWWWWWWK"
SMALL = {"emb": 8, "hidden": 8}


def fabricate(path):
    rows = []
    for split, n in (("train", 14), ("val", 7), ("test", 5), (None, 3)):
        for i in range(n):
            seq = TEST_SEQ if split == "test" else SEQS[i % len(SEQS)]
            rows.append({
                "raw_file": f"f_{split}", "scan_number": i, "peptide_sequence": seq,
                "precursor_charge": [2, 3, 4][i % 3] if i != 6 else 5,          # one charge 5 per split
                "matched_ions": ["b1", "y1", "b2", "y2"], "intensities_raw": [2.0, 1.0, 0.5 + i, 3.0],
                "split": split, "charge_source": "predicted" if split is None else "annotated",
            })
    pd.DataFrame(rows).to_parquet(path, index=False)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "CACHE_DIR", tmp_path / "cache")
    fabricate(tmp_path / "spectra.parquet")
    dirs = {"run_log": tmp_path / "run_log.jsonl", "preliminary_log": tmp_path / "prelim.jsonl",
            "checkpoint_dir": tmp_path / "ckpt", "curve_dir": tmp_path / "curves"}
    return tmp_path, dirs


def test_loader_never_returns_test_rows_and_adds_imputed(env):
    tmp, _ = env
    data = load_dataset(tmp / "spectra.parquet")
    assert len(data["train"]) == 13 + 3 and len(data["val"]) == 6          # one charge 5 per split dropped
    assert data["train"].imputed.sum() == 3 and not data["val"].imputed.any()
    assert TEST_SEQ not in set(data["train"].sequence) | set(data["val"].sequence)
    assert set(np.unique(data["train"].charge)) <= {2, 3, 4}
    cached = load_dataset(tmp / "spectra.parquet")
    assert cached["train"].fingerprint() == data["train"].fingerprint()
    assert len(load_dataset(tmp / "spectra.parquet", include_imputed=False)["train"]) == 13


def test_full_run_logs_checkpoint_and_reproduces_score(env):
    tmp, dirs = env
    data = load_dataset(tmp / "spectra.parquet")
    cfg = TrainConfig(batch_size=4, max_epochs=2, patience=1)
    result = Trainer(cfg, device="cpu", verbose=False, **dirs).fit("PrositLike", data, SMALL)
    record = result.record
    logged = tracking.load_records(dirs["run_log"])
    assert len(logged) == 1 and logged[0]["run_id"] == record["run_id"]
    assert record["data"]["n_train_imputed"] == 3 and record["model_params"]["head"] == "pooled"
    assert set(record["by_charge"]) == {"2", "3", "4"} and record["buckets"]
    model, saved = tracking.load_checkpoint(record["checkpoint"])
    again = evaluation.evaluate(model, data["val"])
    assert math.isclose(again["val_sa"], record["val_sa"], abs_tol=1e-6)
    assert (dirs["curve_dir"]).exists()


def test_preliminary_and_search_stay_off_the_leaderboard(env):
    tmp, dirs = env
    data = load_dataset(tmp / "spectra.parquet")
    cfg = TrainConfig(batch_size=4, preliminary=True, preliminary_train=5, preliminary_val=3, preliminary_epochs=1)
    record = Trainer(cfg, device="cpu", verbose=False, **dirs).fit("PrositLike", data, SMALL).record
    assert record["data"]["n_train"] == 5 and record["checkpoint"] is None
    trials = search("PrositLike", data, {"hidden": [4, 8]}, {"lr": [1e-3, 1e-4]}, cfg, strategy="random",
                    trials=3, device="cpu", verbose=False, **dirs)
    assert len(trials) == 3 and trials["val_sa"].is_monotonic_decreasing
    assert not dirs["run_log"].exists()
    assert len(tracking.load_records(dirs["preliminary_log"])) == 4


def test_tweaked_training_algorithm(env):
    tmp, dirs = env
    data = load_dataset(tmp / "spectra.parquet")
    cfg = TrainConfig(batch_size=4, max_epochs=2, loss="masked_mse", optimizer="adamw", scheduler="cosine",
                      grad_clip=1.0, preliminary=True)
    assert Trainer(cfg, device="cpu", verbose=False, **dirs).fit("PrositLike", data, SMALL).record["val_sa"] >= 0


def test_compare_against_previous_and_koina(env, monkeypatch):
    tmp, dirs = env
    data = load_dataset(tmp / "spectra.parquet")
    cfg = TrainConfig(batch_size=4, max_epochs=1, patience=1)
    for _ in range(2):
        Trainer(cfg, device="cpu", verbose=False, **dirs).fit("PrositLike", data, SMALL)

    calls = []
    def fake_predict(self, sequences, charges):
        calls.append(len(sequences))
        return np.ones((len(sequences), VECTOR_DIM))
    monkeypatch.setattr(koina.KoinaModel, "predict", fake_predict)
    compare.score_koina(data["val"], ["FakeKoina"], run_log=dirs["run_log"])
    compare.score_koina(data["val"], ["FakeKoina"], run_log=dirs["run_log"])     # cached
    assert calls == [len(data["val"])]

    table = compare.leaderboard(data["val"].fingerprint(), run_log=dirs["run_log"], print_table=False)
    assert len(table) == 3 and table["val_sa"].is_monotonic_decreasing
    recent = compare.compare_recent(table, recent=2, print_table=False)
    assert len(recent) == 2 and recent["koina"].eq("Koina FakeKoina").all()
    assert recent["prev_best"].iloc[-1] is None              # oldest run has nothing earlier
    assert not compare.bucket_frame(table["run_id"].tolist(), dirs["run_log"]).empty


def test_koina_annotation_parsing():
    vec = koina.annotations_to_vector([b"y1+1", "b2+1", "y3+2", "b1^1", "y5+1"], [1.0, 0.5, 0.9, 0.2, -1.0])
    from sommer.peptides import fragment_slot
    assert vec[fragment_slot("y", 1)] == 1.0 and vec[fragment_slot("b", 2)] == 0.5
    assert vec[fragment_slot("b", 1)] == pytest.approx(0.2) and vec[fragment_slot("y", 3)] == 0.0
    assert vec[fragment_slot("y", 5)] == 0.0


def test_legacy_conversion():
    old = {"timestamp": "2026-10-14T01:48:04+00:00", "git_commit": "x", "model": "PrositLike", "tag": "t",
           "hyperparameters": {"emb": 64, "hidden": 128, "head": "pooled", "activation": "softplus", "lr": 1e-3},
           "n_params": 1, "n_train": 2, "n_val": 3, "epochs_trained": 4, "best_epoch": 3, "train_minutes": 1.0,
           "val_spectral_angle": 0.8, "val_pearson": 0.9, "checkpoint": "c.pt",
           "by_charge": {"2": {"spectral_angle": 0.85, "pearson": 0.9, "n": 2}}}
    new = legacy_logs.from_train_deep(old)
    assert new["val_sa"] == 0.8 and new["by_charge"]["2"]["sa"] == 0.85 and new["data"]["val_fingerprint"] is None
    assert legacy_logs.from_train_deep(old | {"val_fold": 3}) is None
    json.dumps(new, allow_nan=False)
