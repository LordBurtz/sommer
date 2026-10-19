"""Models. Every module in this package is imported on load, so a new model only needs its own file:

    from sommer.models import FragmentModel, register_model

    @register_model
    class MyModel(FragmentModel):
        def __init__(self, hidden: int = 64): ...
        def forward(self, tokens, precursor_charge): ...   # -> (B, 58)
"""
import importlib
import pkgutil

from sommer.models.base import FragmentModel, Predictor, predict_tensors
from sommer.models.registry import MODELS, build_model, register_model

for _module in pkgutil.iter_modules(__path__):
    if _module.name not in ("base", "registry", "layers"):
        importlib.import_module(f"{__name__}.{_module.name}")

__all__ = ["FragmentModel", "Predictor", "predict_tensors", "MODELS", "build_model", "register_model"]
