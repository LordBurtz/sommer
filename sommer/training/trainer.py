"""Generic trainer: works for every registered FragmentModel.

The loop is split into small overridable steps (build_optimizer, build_scheduler, compute_loss,
train_epoch, validate). Most changes are a TrainConfig field or a new entry in components.py; for
anything else subclass Trainer and override one step.
"""
from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch

from sommer import evaluation, metrics, tracking
from sommer.data import SpectrumSet
from sommer.models.base import FragmentModel
from sommer.models.registry import build_model
from sommer.paths import CHECKPOINT_DIR, CURVE_DIR, PRELIMINARY_LOG, RUN_LOG
from sommer.plots import plot_curve
from sommer.training import components
from sommer.training.config import TrainConfig


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@dataclass
class Batch:
    tokens: torch.Tensor
    charge: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor


@dataclass
class TrainResult:
    model: FragmentModel
    record: dict


class Trainer:
    def __init__(self, config: TrainConfig = TrainConfig(), device: str | torch.device | None = None,
                 run_log=RUN_LOG, preliminary_log=PRELIMINARY_LOG, checkpoint_dir=CHECKPOINT_DIR,
                 curve_dir=CURVE_DIR, verbose: bool = True):
        self.config = config
        self.device = torch.device(device) if device is not None else pick_device()
        self.run_log, self.preliminary_log = run_log, preliminary_log
        self.checkpoint_dir, self.curve_dir = checkpoint_dir, curve_dir
        self.verbose = verbose
        self.loss_fn = components.resolve(components.LOSSES, config.loss, "loss")

    # ------------------------------------------------------------------ steps (override to tweak)

    def build_optimizer(self, model: FragmentModel) -> torch.optim.Optimizer:
        return components.resolve(components.OPTIMIZERS, self.config.optimizer, "optimizer")(
            model.parameters(), self.config)

    def build_scheduler(self, optimizer: torch.optim.Optimizer):
        return components.build_scheduler(optimizer, self.config)

    def compute_loss(self, model: FragmentModel, batch: Batch) -> torch.Tensor:
        pred = model(batch.tokens, batch.charge)
        if pred.shape != batch.target.shape:
            raise ValueError(f"{type(model).__name__} returned {tuple(pred.shape)}, "
                             f"expected {tuple(batch.target.shape)}")
        return self.loss_fn(pred, batch.target, batch.mask)

    def train_epoch(self, model: FragmentModel, data: SpectrumSet, optimizer: torch.optim.Optimizer,
                    generator: torch.Generator) -> float:
        model.train()
        total = 0.0
        for batch in self.batches(data, shuffle=True, generator=generator):
            optimizer.zero_grad(set_to_none=True)
            loss = self.compute_loss(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            if self.config.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
            optimizer.step()
            total += loss.item() * len(batch.tokens)
        return total / len(data)

    def validate(self, model: FragmentModel, data: SpectrumSet) -> float:
        """model-selection score (higher is better): mean validation spectral angle"""
        pred = evaluation.predict(model, data)
        return float(metrics.spectral_angle(data.target, pred, data.mask).mean())

    # ------------------------------------------------------------------ main loop

    def fit(self, model_name: str, data: dict[str, SpectrumSet], model_params: dict | None = None,
            extra: dict | None = None) -> TrainResult:
        """train one model with early stopping on val SA, restore the best epoch, score and log it

        ``extra`` is merged into the logged record (e.g. a note or search id).
        """
        cfg = self.config
        set_seed(cfg.seed)
        if self.device.type == "cpu":
            torch.set_num_threads(os.cpu_count())
        started = tracking.utc_now()
        run_id = tracking.new_run_id(started)
        train = data["train"].sample(cfg.preliminary_train if cfg.preliminary else None, cfg.seed)
        val = data["val"].sample(cfg.preliminary_val if cfg.preliminary else None, cfg.seed)

        model, params = build_model(model_name, model_params)
        model.to(self.device)
        optimizer = self.build_optimizer(model)
        scheduler = self.build_scheduler(optimizer)
        generator = torch.Generator().manual_seed(cfg.seed)
        n_params = sum(p.numel() for p in model.parameters())
        ckpt = None if cfg.preliminary else tracking.checkpoint_path(run_id, model_name, self.checkpoint_dir)
        self.log(f"{model_name} [{cfg.mode}] params={n_params:,} n_train={len(train):,} n_val={len(val):,} "
                 f"device={self.device}")

        history, best_sa, best_epoch, best_state, stale = [], -math.inf, 0, None, 0
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(model, train, optimizer, generator)
            val_sa = self.validate(model, val)
            if not math.isfinite(val_sa):
                raise FloatingPointError("non-finite validation score")
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(1 - val_sa)
            elif scheduler is not None:
                scheduler.step()
            history.append({"epoch": epoch, "train_loss": train_loss, "val_sa": val_sa,
                            "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.time() - t0, 2)})
            improved = val_sa > best_sa
            if improved:
                best_sa, best_epoch, stale = val_sa, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            self.log(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val SA={val_sa:.4f}  "
                     f"lr={history[-1]['lr']:.1e}  {history[-1]['seconds']:.0f}s{'  *' if improved else ''}")
            if stale >= cfg.patience:
                self.log(f"early stop: no val SA improvement for {cfg.patience} epochs")
                break

        model.load_state_dict(best_state)
        model.eval()
        record = {
            "run_id": run_id, "timestamp": started.isoformat(timespec="seconds"), "source": "pipeline",
            "mode": cfg.mode, "model": model_name, "model_params": params, "n_params": n_params,
            "train_config": cfg.to_dict(),
            "data": {"n_train": len(train), "n_train_imputed": int(train.imputed.sum()),
                     "n_val": len(val), "train_fingerprint": train.fingerprint(),
                     "val_fingerprint": val.fingerprint()},
            **evaluation.evaluate(model, val),
            "epochs_trained": len(history), "best_epoch": best_epoch,
            "seconds": round((tracking.utc_now() - started).total_seconds(), 1),
            "device": str(self.device), "git_commit": tracking.git_commit(),
            "checkpoint": str(ckpt) if ckpt else None, **(extra or {}), "history": history,
        }
        self.persist(model, record, ckpt)
        self.log(f"best epoch {best_epoch}: val SA={record['val_sa']:.4f}  pearson={record['val_pearson']:.4f}  "
                 + "  ".join(f"z={z}: {r['sa']:.4f}" for z, r in record["by_charge"].items() if r["n"]))
        return TrainResult(model, record)

    def persist(self, model: FragmentModel, record: dict, ckpt) -> None:
        if ckpt is None:
            tracking.append_record({k: v for k, v in record.items() if k != "history"}, self.preliminary_log)
            return
        tracking.save_checkpoint(model, record, ckpt)
        tracking.append_record(record, self.run_log)
        plot_curve(record, self.curve_dir / f"{record['model']}_{record['run_id']}.png")
        self.log(f"logged to {self.run_log}, checkpoint {ckpt}")

    # ------------------------------------------------------------------ helpers

    def batches(self, data: SpectrumSet, shuffle: bool, generator: torch.Generator | None = None):
        n = len(data)
        order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
        arrays = (data.tokens, data.charge, data.target, data.mask.astype(np.float32))
        for s in range(0, n, self.config.batch_size):
            idx = order[s:s + self.config.batch_size].numpy()
            yield Batch(*(torch.from_numpy(a[idx]).to(self.device) for a in arrays))

    def log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)


def run_experiments(experiments: dict[str, dict], data: dict[str, SpectrumSet], config: TrainConfig = TrainConfig(),
                    **trainer_kwargs) -> list[dict]:
    """train several {model name: constructor params} with the same config; returns the records"""
    records = []
    for name, params in experiments.items():
        result = Trainer(config, **trainer_kwargs).fit(name, data, params)
        records.append(result.record)
        del result
    return records

