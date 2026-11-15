"""Training configuration. Every knob of the training algorithm lives here; components are chosen by
name from sommer.training.components, so most experiments only change a TrainConfig."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class TrainConfig:
    # objective / optimisation (names resolve in components.py)
    loss: str = "masked_spectral_angle"
    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 1e-5
    scheduler: str = "plateau"
    scheduler_params: dict = field(default_factory=dict)   # overrides components.SCHEDULER_DEFAULTS
    grad_clip: float | None = None
    batch_size: int = 512
    # early stopping on validation spectral angle, the best epoch is kept
    max_epochs: int = 50
    patience: int = 5
    seed: int = 1312
    # preliminary = quick sanity run on a subsample; logged separately, never on the leaderboard
    preliminary: bool = False
    preliminary_train: int = 2000
    preliminary_val: int = 500
    preliminary_epochs: int = 2

    def __post_init__(self):
        for name in ("batch_size", "max_epochs", "patience", "preliminary_train", "preliminary_val",
                     "preliminary_epochs"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("lr must be positive and weight_decay non-negative")

    @property
    def mode(self) -> str:
        return "preliminary" if self.preliminary else "full"

    @property
    def epochs(self) -> int:
        return min(self.max_epochs, self.preliminary_epochs) if self.preliminary else self.max_epochs

    def to_dict(self) -> dict:
        return asdict(self)
