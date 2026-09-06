"""Registering the plugin (what `sf list` does) must not import torch or lightning."""

import subprocess
import sys


def test_registry_snapshot_does_not_import_torch():
    code = (
        "import sys, signalflow as sf; sf.registry.snapshot(); "
        "assert 'strategy' in sf.registry.snapshot() and 'rl' in sf.registry.list(sf.ComponentType.STRATEGY); "
        "heavy = [m for m in ('torch', 'lightning', 'stable_baselines3') if m in sys.modules]; "
        "assert not heavy, heavy; print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]


def test_lazy_submodules_still_import():
    import signalflow.labs as labs

    assert labs.encoder is not None and labs.head is not None and labs.loss is not None
