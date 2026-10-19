"""Pluggable pieces of the training algorithm, selected by name in TrainConfig.

Add an entry here to make a new loss / optimizer / scheduler available to every model.
Losses take (pred, target, mask) of shape (B, VECTOR_DIM) and return a scalar.
"""
from __future__ import annotations

import math
from typing import Callable

import torch

from sommer.training.config import TrainConfig


def masked_spectral_angle_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                               eps: float = 1e-8) -> torch.Tensor:
    """1 - SA, averaged over the batch"""
    pred, target = pred * mask, target * mask
    cos = (pred * target).sum(dim=1) / (pred.norm(dim=1) * target.norm(dim=1) + eps)
    return (2.0 * torch.arccos(cos.clamp(-1 + 1e-7, 1 - 1e-7)) / math.pi).mean()


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE on base-peak normalised predictions (scale-free like SA)"""
    pred = pred * mask
    pred = pred / pred.amax(dim=1, keepdim=True).clamp(min=1e-8)
    return (((pred - target) * mask) ** 2).sum() / mask.sum().clamp(min=1)


LOSSES: dict[str, Callable[..., torch.Tensor]] = {
    "masked_spectral_angle": masked_spectral_angle_loss,
    "masked_mse": masked_mse_loss,
}

OPTIMIZERS: dict[str, Callable[..., torch.optim.Optimizer]] = {
    "adam": lambda params, cfg: torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay),
    "adamw": lambda params, cfg: torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay),
    "sgd": lambda params, cfg: torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9),
}

# plateau schedulers are stepped with the validation loss (1 - val SA), all others once per epoch;
# TrainConfig.scheduler_params override these defaults
SCHEDULER_DEFAULTS: dict[str, dict] = {
    "none": {},
    "plateau": {"factor": 0.5, "patience": 1},
    "cosine": {},
    "step": {"step_size": 10, "gamma": 0.5},
}
SCHEDULERS: dict[str, Callable[..., object]] = {
    "none": lambda opt, cfg, **kw: None,
    "plateau": lambda opt, cfg, **kw: torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **kw),
    "cosine": lambda opt, cfg, **kw: torch.optim.lr_scheduler.CosineAnnealingLR(opt, **({"T_max": cfg.epochs} | kw)),
    "step": lambda opt, cfg, **kw: torch.optim.lr_scheduler.StepLR(opt, **kw),
}


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig):
    factory = resolve(SCHEDULERS, cfg.scheduler, "scheduler")
    return factory(optimizer, cfg, **(SCHEDULER_DEFAULTS.get(cfg.scheduler, {}) | cfg.scheduler_params))


def resolve(table: dict, name: str, kind: str):
    if name not in table:
        raise KeyError(f"unknown {kind} {name!r}, available: {sorted(table)}")
    return table[name]
