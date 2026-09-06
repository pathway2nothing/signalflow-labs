# Changelog (signalflow-labs)

## [Unreleased]

### Changed (breaking - pre-1.0, no compatibility aliases)

- `lightning` moved to the `[temporal]` extra; the live components (`TorchMLPBackend`,
  `RLStrategy`, `make_env`) need only torch (and the `[rl]` extra).
- `signalflow.labs` imports its neural building blocks and the parked temporal stack
  lazily; registering the plugin never imports torch or lightning (`tests/test_lazy_import.py`).
- `RLStrategy`/`make_env` raise `ImportError` on access without the `[rl]` extra
  instead of being `None`; `requirements.txt` removed (pyproject is the source).
