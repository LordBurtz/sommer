# How to run
Sync all dependencies to local
```python
$ uv sync
```

create the virtual environemnt
```python
$ uv venv
```

sync dependencies
```python
$ uv sync
```

activate the virtual environment
```python
$ source .venv/bin/activate
```

add a package
```python
$ uv add
```

# Current tooling
Pipeline in order, run from repo root (except `refac/` notebooks, which run inside `refac/`)

Split (refac/01): 10 folds, `StratifiedGroupKFold(random_state=187)`, grouped by `stripped_sequence`,
stratified on `frag`x`precursor_charge`
Fold 0 = test, fold 1 = val, folds 2-9 = train. 
split is fixed. CV moves val fold over folds 1-9 and never touches fold 0.

1. **`refac/01_data_cleansing.ipynb`**: cleaning and split. Reads `data_for_student.parquet` +
   `metadata_for_student.parquet`, writes `refac/clean_spectra_split.parquet` (`split`column),
   `_train/_val/_test_split` convenience files & `refac/data_without_precursor_charge.parquet`.
   ```
   cd refac && uv run jupyter nbconvert --to notebook --execute --inplace 01_data_cleansing.ipynb
   ```
2. **`add_folds.py`**: adds `fold` column (0-9) to `refac/clean_spectra_split.parquet`. 
   Recomputes folds of 01 and writes only if fold 0 == test and fold 1 == val (row by row)
   Takes ~15 min
   ```
   uv run python add_folds.py
   ```
   After rerunning 01: run `add_folds.py` again & lete `cache/deep_arrays_refac_trainval.npz`
   (the cache only rebuilt when a key is missing, not when data changes)

3. **`refac/02_precursor_charge.ipynb`**: CatBoost precursor charge classifier + Optuna search, every config
   appended to `refac/model_training_metadata_precursor_charge.json`. 
   Needs `optuna`, which is not in
   `pyproject.toml` yet
   ```
   cd refac && uv run jupyter nbconvert --to notebook --execute --inplace 02_precursor_charge.ipynb
   ```
4. **`train_deep.py`**: deep intensity model (b/y, 58 slots), early stopping on val spect angle
   `--val-fold K` (1-9, default 1): val = fold K, train = the other folds 1-9. Appends one record to
   `results/runs.jsonl` (smoke: `results/smoke_runs.jsonl`), best checkpoint to
   `checkpoints/<tag>_<UTC start>.pt`, curve to `results/<tag>_curve.png`. The tag ends in `_fold<K>_seed<N>`.
   ```
   uv run python train_deep.py --smoke              # 2k spectra, 1 epoch: run this first
   uv run python train_deep.py                      # full run, val = fold 1
   uv run python train_deep.py --head bond --use-frag --activation sigmoid --val-fold 3
   ```
5. **`06_deep_baseline.ipynb`**: Colab driver around `train_deep.main` (clones `master`, data from Drive).
   The Drive copy of `clean_spectra_split.parquet` needs the `fold` column: upload the local file after
   step 2, or run `!python add_folds.py --data /content/drive/MyDrive/sommer/clean_spectra_split.parquet`.
6. **`compare_runs.py`**: all runs, val SA overall and per charge, difference baseline run
   Runs scored on another val fold show "other val"
   ```
   uv run python compare_runs.py
   uv run python compare_runs.py --file results/smoke_runs.jsonl
   ```
7. **`baseline_lookup.py`**: lookup-table baseline (from 03) on folds 1-5, with same train/val, arrays
   & scoring as`train_deep.py`.
   ```
   uv run python baseline_lookup.py
   ```
8. **`bucket_eval.py`**: one checkpoint on its own val fold, by precursor chargexlength (7-12, 13-18,
   19-24, 25-30) x modified, with n, mean and median SA. Writes `results/bucket_eval_<checkpoint>.csv`
   ```
   uv run python bucket_eval.py checkpoints/<tag>_<UTC start>.pt
   ```
9. **`evaluate_test.py`**:only script that loads the test split (fold 0)
   Writes `results/test_results.json`
   ```
   uv run python evaluate_test.py checkpoints/<a>.pt checkpoints/<b>.pt
   ```
10. **`make_figures.py`**: report figures from `results/runs.jsonl` + `results/test_results.json`
    into `results/figures/`.
    ```
    uv run python make_figures.py
    ```

Tests:
```
uv run python test_common.py
uv run python test_train_deep.py
```
Shared code: `common.py` (tokens, 58-slot b/y vector, spectral angle, monoisotopic masses) and
`refac/refac_common.py` (for the refac notebooks, imports nothing from outside `refac/`).

## Legacy (old split and/or avg masses)
Reference only. numbers are not comparable with the ones above
- `01_data_cleaning.ipynb`, `02_precursor_charge.ipynb` (top level): replaced by `refac/01` and `refac/02`
- `03_intensity_baseline.ipynb`: lookup + ridge on an 80k subsample with its own GroupKFold
  (the lookup is now `baseline_lookup.py`)
- `04_intensity_gbm.ipynb`, `05_intensity_deep.ipynb`
- `aa_data_cleansing.ipynb`, `data_exploration.ipynb`, `predict-precursorcharge.ipynb`
- `SyntheticDataGeneration.ipynb`: Colab, own KFold/GroupKFold on the raw data
- `common_tools.py`: average masses labelled as monoisotopic, don't copy from it
- `clean_spectra.parquet`, `spectra_with_charge.parquet`, `*.pkl`, `*.model`, `approach*_score.npy`:
  outputs of the old notebooks
- `not_needed/`: retired, old split leaked the same peptide (with different mods) across train and test
