"""signalflow.labs - neural-network and RL extension plugin for SignalFlow V5.

Importing this package registers its components (encoders, heads, the temporal
classifier/validator, and the RL strategy) into the V5 registry. Each submodule
import is guarded so an unavailable optional dependency never breaks the import.
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
