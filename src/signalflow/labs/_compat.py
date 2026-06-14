"""Compatibility shims reproducing OLD-core symbols over the core.

The labs code predates the rewrite and imports symbols (``register``,
``validator``, ``SfTorchModuleMixin``, ``SfComponentType``, ``default_registry``,
``SignalValidator``, ``Signals``) that no longer exist in ``signalflow``. This
module re-expresses each of them on top of the ``registry`` and contracts so
the neural-network modules keep registering and importing unchanged.
"""


from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Literal, TypeVar

import polars as pl

from signalflow.enums import ComponentType
from signalflow.registry import registry

T = TypeVar("T", bound=type)

default_registry = registry


class SfComponentType:
    """Old component-type names mapped onto the ``ComponentType`` set."""

    TORCH_MODULE = ComponentType.MODEL
    VALIDATOR = ComponentType.MODEL
    MODEL = ComponentType.MODEL


def register(name: str, *, override: bool = True) -> Callable[[T], T]:
    """Register a neural component into the registry under the MODEL type."""

    def decorator(cls: T) -> T:
        cls._sf_name = name
        cls._sf_type = ComponentType.MODEL
        registry.register(ComponentType.MODEL, name, cls, role="model", override=override)
        return cls

    return decorator


def validator(name: str, **_kwargs: Any) -> Callable[[T], T]:
    """Register a validator into the registry, tolerating legacy kwargs."""

    def decorator(cls: T) -> T:
        cls._sf_name = name
        cls._sf_type = ComponentType.MODEL
        registry.register(ComponentType.MODEL, name, cls, role="validator", override=True)
        return cls

    return decorator


class Signals:
    """Thin wrapper around a polars frame of (pair, timestamp, ...) signals."""

    def __init__(self, frame: pl.DataFrame):
        self.frame = frame

    def __getattr__(self, item: str) -> Any:
        return getattr(self.frame, item)

    def __len__(self) -> int:
        return self.frame.height

    def __repr__(self) -> str:
        return f"Signals({self.frame.height} rows)"


class SfTorchModuleMixin(ABC):
    """Mixin for SignalFlow neural modules: default params and search space."""

    component_type = SfComponentType.TORCH_MODULE

    @classmethod
    @abstractmethod
    def default_params(cls) -> dict:
        """Default constructor parameters for this module."""
        ...

    @classmethod
    @abstractmethod
    def search_space(cls, model_size: Literal["small", "medium", "large"] = "small") -> dict:
        """Hyperparameter search space for tuning."""
        ...


class SignalValidator:
    """Base validator usable as a validator slot (meta-labeler)."""

    component_type = SfComponentType.VALIDATOR
    output = "p_success"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.model: Any | None = None
        self.feature_columns: list[str] | None = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return getattr(self, "_fitted", False)

    def fit(self, *args: Any, **kwargs: Any) -> "SignalValidator":
        """Train the validator. Subclasses implement the real logic."""
        raise NotImplementedError("Subclasses must implement fit()")

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Predict signal quality. Subclasses implement the real logic."""
        raise NotImplementedError("Subclasses must implement predict()")

    def predict_oos(self, *args: Any, **kwargs: Any) -> Any:
        """Leak-free predictions over the training span (defaults to predict)."""
        return self.predict(*args, **kwargs)
