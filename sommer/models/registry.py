"""Name -> model class registry. Decorate a FragmentModel subclass with @register_model to make it
available to the trainer, checkpoints and the comparison by name."""
from __future__ import annotations

import inspect

from sommer.models.base import FragmentModel

MODELS: dict[str, type[FragmentModel]] = {}


def register_model(cls: type[FragmentModel] | None = None, *, name: str | None = None):
    def wrap(cls: type[FragmentModel]) -> type[FragmentModel]:
        if not issubclass(cls, FragmentModel):
            raise TypeError(f"{cls.__name__} must subclass FragmentModel")
        key = name or cls.__name__
        if key in MODELS and MODELS[key] is not cls and MODELS[key].__qualname__ != cls.__qualname__:
            raise ValueError(f"model name {key!r} is already registered by {MODELS[key]}")
        cls.name = key
        MODELS[key] = cls
        return cls
    return wrap(cls) if cls is not None else wrap


def build_model(name: str, params: dict | None = None) -> tuple[FragmentModel, dict]:
    """instantiate a registered model; returns (model, full constructor params incl. defaults)"""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}, registered: {sorted(MODELS)}")
    cls = MODELS[name]
    bound = inspect.signature(cls).bind(**(params or {}))
    bound.apply_defaults()
    return cls(**bound.arguments), dict(bound.arguments)
