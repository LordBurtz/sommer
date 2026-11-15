"""Hyperparameter search. Strategies only generate candidates; training is always Trainer.fit in
preliminary mode on the same subsample, trials go to results/preliminary_runs.jsonl. Promote a winner
with a separate full run."""
from __future__ import annotations

import math
import uuid
from dataclasses import replace
from itertools import product
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from sommer.data import SpectrumSet
from sommer.training.config import TrainConfig
from sommer.training.trainer import Trainer

Space = dict[tuple[str, str], list]   # (group "model" | "training", key) -> options


def grid_strategy(space: Space, trials: int, seed: int) -> Iterator[dict]:
    for values in product(*space.values()):
        yield dict(zip(space, values))


def random_strategy(space: Space, trials: int, seed: int) -> Iterator[dict]:
    """up to `trials` distinct combinations, reproducible, without materialising the grid"""
    total = math.prod(len(v) for v in space.values())
    for index in np.random.default_rng(seed).choice(total, min(total, trials), replace=False):
        candidate = {}
        for key, options in reversed(list(space.items())):
            index, offset = divmod(int(index), len(options))
            candidate[key] = options[offset]
        yield candidate


STRATEGIES: dict[str, Callable[[Space, int, int], Iterator[dict]]] = {"grid": grid_strategy,
                                                                     "random": random_strategy}


def search(model_name: str, data: dict[str, SpectrumSet], model_space: dict[str, list],
           training_space: dict[str, list] | None = None, config: TrainConfig = TrainConfig(),
           strategy: str = "random", trials: int = 8, **trainer_kwargs) -> pd.DataFrame:
    """train each candidate (model params x TrainConfig overrides) and rank by val SA"""
    space = {(group, key): list(values)
             for group, entries in (("model", model_space), ("training", training_space or {}))
             for key, values in entries.items()}
    if any(not v for v in space.values()):
        raise ValueError("every search dimension needs at least one value")
    search_id = uuid.uuid4().hex[:8]
    rows = []
    for candidate in STRATEGIES[strategy](space, trials, config.seed):
        model_params = {k: v for (g, k), v in candidate.items() if g == "model"}
        overrides = {k: v for (g, k), v in candidate.items() if g == "training"}
        trainer = Trainer(replace(config, preliminary=True, **overrides), **trainer_kwargs)
        record = trainer.fit(model_name, data, model_params,
                             extra={"search_id": search_id, "training_overrides": overrides}).record
        rows.append({"search_id": search_id, "model": model_name, "model_params": model_params,
                     "training_overrides": overrides, "val_sa": record["val_sa"], "run_id": record["run_id"]})
    return pd.DataFrame(rows).sort_values("val_sa", ascending=False).reset_index(drop=True)
