# Fragment ion intensity prediction (CID MS/MS)

Predict the intensities of singly charged b/y fragment ions from a peptide sequence and its precursor
charge (2, 3, 4). the test split is never used

## Setup
```
uv sync                      # creates .venv, installs dependencies and the `sommer` package (editable)
source .venv/bin/activate
uv run pytest                # tests on fabricated data, no research data needed
```

## Pipeline
Notebooks run inside `refac/`, the `sommer` package works from anywhere.

| step | entry point | reads | writes |
|---|---|---|---|
| 1. merge, clean, split | `refac/01_data_cleansing.ipynb` | `data_for_student.parquet`, `metadata_for_student.parquet` | `refac/clean_spectra_*split.parquet`, `refac/data_without_precursor_charge.parquet` |
| 2. impute precursor charge | `refac/02_precursor_charge.ipynb` | train/val split, rows without charge | CatBoost model, `refac/spectra_with_charge.parquet` (train + val + imputed, no test) |
| 3. train | `refac/03_train.ipynb` or `python -m sommer.train` | `refac/spectra_with_charge.parquet` | `results/run_log.jsonl`, `checkpoints/`, `results/curves/` |
| 4. compare | `refac/04_compare.ipynb` or `python -m sommer.compare` | `results/run_log.jsonl`, Koina API | Koina scores cached in the run log |

Split: `StratifiedGroupKFold(10, random_state=187)` grouped by `stripped_sequence`, stratified on
`frag` x `precursor_charge`; fold 0 = test, fold 1 = val, folds 2-9 = train. Training uses train + the
spectra whose charge was imputed in step 2; validation uses annotated val spectra only.

```
uv run python -m sommer.train --list
uv run python -m sommer.train --model PrositLike --preliminary            # smoke run, ~seconds
uv run python -m sommer.train --model PeptideEncoder -p hidden_dimensions=256 -p transformer_layers=6 \
    --lr 6e-4 --weight-decay 1.8e-6 --batch-size 128 --max-epochs 100 --patience 15
uv run python -m sommer.compare --recent 3                                # --no-koina to stay offline
```

## Code layout (`sommer/`)
```
constants.py  peptides.py      vocabulary, masses, parsing, 58-slot b/y vector (slot = (n-1)*2 + {b:0, y:1})
metrics.py                     spectral angle + Pearson (per spectrum, masked, scale invariant), per-charge summary
buckets.py                     charge x length bin x modified breakdown
data.py                        load/encode/cache train + val (test rows are filtered out at parquet level)
models/base.py                 FragmentModel: forward(tokens, precursor_charge) -> (B, 58); Predictor protocol
models/registry.py             @register_model, build_model(name, params)
models/layers.py               shared blocks (positional encoding, transformer, pooling, place_bonds)
models/prosit_like.py          GRU baseline;  models/peptide_encoder.py  transformer
training/config.py             TrainConfig: every knob of the training algorithm
training/components.py         losses / optimizers / schedulers by name
training/trainer.py            generic Trainer (early stopping on val SA, best epoch restored, logging)
training/search.py             grid / random hyperparameter search (preliminary runs)
evaluation.py                  one scoring function for our models, checkpoints and Koina
tracking.py                    run log (JSON lines) + checkpoints
koina.py  compare.py           Koina predictor, leaderboard and recent-vs-previous-vs-Koina comparison
legacy_logs.py                 one-off migration of the old run logs (re-scores old checkpoints)
```

### Adding a model
Create `sommer/models/my_model.py`; it is discovered automatically:
```python
from sommer.models import FragmentModel, register_model

@register_model
class MyModel(FragmentModel):
    def __init__(self, hidden: int = 128):          # keyword args only, stored in the checkpoint
        super().__init__()
        ...

    def forward(self, tokens, precursor_charge):    # (B, 30) int64, (B,) int64 raw charge -> (B, 58)
        ...
```
Then `python -m sommer.train --model MyModel -p hidden=256`. Add it to `SMALL` in `tests/test_models.py`.

### Changing the training algorithm
- a different loss / optimizer / schedule: a `TrainConfig` field (`loss="masked_mse"`, `scheduler="cosine"`, `grad_clip=1.0`, ...)
- a new one: add an entry to the dicts in `training/components.py`
- anything else: subclass `Trainer` and override one step (`build_optimizer`, `build_scheduler`,
  `compute_loss`, `train_epoch`, `validate`); example in `refac/03_train.ipynb`

### Run log
`results/run_log.jsonl` holds one record per full run (model, params, config, data fingerprints, val SA /
Pearson overall, per charge and per bucket, history, checkpoint, git commit), Koina scores
(`source="koina"`) and migrated legacy runs (`source="legacy:*"`). Smoke runs and search trials go to
`results/preliminary_runs.jsonl` and never reach the leaderboard. Runs are comparable when their
`val_fingerprint` matches.

Metric note: scores logged by the old tooling before 2026-10-16 clipped predictions to [0, 1]; the
current metric is scale invariant. Migrated runs with a checkpoint were re-scored (old value kept as
`logged_val_sa`).

## Legacy
`legacy/` holds superseded code and outputs for reference (old split and/or average masses; numbers are
not comparable): the top-level 01-05 notebooks, `train_deep.py`, `common.py`, `08_training_pipeline.ipynb`,
the Colab driver `06_deep_baseline.ipynb`, `evaluate_test.py`, `make_figures.py`, `bucket_eval.py`,
`baseline_lookup.py`, `predict_koina.py`, ... `refac/07_model_diagnostics.ipynb` still imports the legacy
modules and has not been ported yet.
