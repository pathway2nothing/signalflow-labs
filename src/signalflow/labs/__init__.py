"""signalflow.labs - neural-network and RL extension plugin for SignalFlow.

Importing this package registers the RL strategy into the registry. The neural
building blocks (``encoder``, ``head``, ``loss``) and the parked temporal stack
(``data``, ``model``, ``validator``) are imported lazily on first attribute access,
so registering the plugin (``sf list``) never pulls in torch or lightning.
"""

import importlib

from loguru import logger

SIGNALFLOW_PLUGIN_API = 1

from signalflow.labs.backend import TorchMLPBackend  # torch is imported inside its methods

_LAZY_SUBMODULES = frozenset({"loss", "encoder", "head", "layer", "data", "model", "validator"})


def _load_strategy() -> None:
    """Register ``strategy: rl`` (needs the ``[rl]`` extra; skipped with a warning otherwise)."""
    try:
        from signalflow.labs.strategy import RLStrategy as _RL
        from signalflow.labs.strategy import make_env as _mk

        globals()["RLStrategy"], globals()["make_env"] = _RL, _mk
    except Exception as exc:
        logger.warning(f"signalflow.labs: RL strategy unavailable ({type(exc).__name__}: {exc})")


def __getattr__(name: str) -> object:
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"signalflow.labs.{name}")
        globals()[name] = module
        return module
    if name in ("RLStrategy", "make_env"):
        raise ImportError(
            f"signalflow.labs.{name} needs the [rl] extra: pip install 'signalflow-labs[rl]' (stable-baselines3, gymnasium)"
        )
    raise AttributeError(f"module 'signalflow.labs' has no attribute {name!r}")


_load_strategy()

__all__ = ["TorchMLPBackend", "RLStrategy", "make_env"]
