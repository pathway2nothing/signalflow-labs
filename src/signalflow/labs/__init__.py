"""signalflow.labs - neural-network and RL extension plugin for SignalFlow.

Importing this package registers the RL strategy into the registry and exposes
the neural building blocks (encoders, heads, losses) and the parked temporal
stack as importable modules. Encoders and heads are bare ``nn.Module`` s, not
model-contract components, so they are not registered. Each submodule import is
guarded so an unavailable optional dependency never breaks the import.
"""

from loguru import logger

from signalflow.labs.backend import TorchMLPBackend

_GUARDED = [
    "signalflow.labs.loss",
    "signalflow.labs.encoder",
    "signalflow.labs.head",
    "signalflow.labs.data",
    "signalflow.labs.model",
    "signalflow.labs.validator",
    "signalflow.labs.strategy",
]

RLStrategy = None
make_env = None


def _load_all() -> None:
    """Import every submodule so the entry point registers all components."""
    import importlib

    for name in _GUARDED:
        try:
            importlib.import_module(name)
        except Exception as exc:
            logger.warning(f"signalflow.labs: skipped {name} ({type(exc).__name__}: {exc})")

    global RLStrategy, make_env
    try:
        from signalflow.labs.strategy import RLStrategy as _RL
        from signalflow.labs.strategy import make_env as _mk

        RLStrategy, make_env = _RL, _mk
    except Exception as exc:
        logger.warning(f"signalflow.labs: RL strategy unavailable ({type(exc).__name__}: {exc})")


_load_all()

__all__ = ["TorchMLPBackend", "RLStrategy", "make_env"]
