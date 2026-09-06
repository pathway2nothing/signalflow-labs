"""RLStrategy + make_env against the current core: decide, schema guard, env stepping, flow round-trip."""

import numpy as np
import polars as pl
import pytest
from signalflow.engine.types import PortfolioSnapshot
from signalflow.errors import ArtifactError, SchemaVersionError
from signalflow.strategy.observation import OBSERVATION_SCHEMA_VERSION, Observation

import signalflow as sf
import signalflow.labs as labs
from signalflow.labs.strategy import RLStrategy, action_to_intents, make_env


class _FixedPolicy:
    """Picklable stand-in for an SB3 policy: always returns one action."""

    def __init__(self, action: int):
        self.action = action

    def predict(self, vector, deterministic=True):
        return np.array([self.action]), None


@pytest.fixture(scope="module")
def ds():
    return sf.dataset("synthetic", pairs=["BTCUSDT", "ETHUSDT"], start="2023-01-01", end="2023-02-01", interval="1h")


def _obs(signals: pl.DataFrame, equity: float = 10_000.0, positions=None) -> Observation:
    snap = PortfolioSnapshot(ts=None, target="USDT", balances={"USDT": equity}, positions=positions or {}, equity=equity)
    return Observation(ts=None, signals=signals, portfolio=snap, mandate={})


def _rise_signals():
    return pl.DataFrame({"pair": ["BTCUSDT", "ETHUSDT"], "ts": [None, None], "signal": ["rise", "rise"], "p_success": [0.6, 0.9]})


def test_hold_open_close_intents():
    obs = _obs(_rise_signals())
    assert action_to_intents(0, obs, 0.1) == []
    opened = action_to_intents(1, obs, 0.1)
    assert len(opened) == 1 and opened[0].pair == "ETHUSDT" and opened[0].notional == pytest.approx(1_000.0)
    assert action_to_intents(2, obs, 0.1) == []  # nothing held -> nothing to close


def test_decide_routes_through_policy_and_checks_schema():
    strat = RLStrategy(model=_FixedPolicy(1), size_pct=0.2)
    intents = strat.decide(_obs(_rise_signals()))
    assert [i.pair for i in intents] == ["ETHUSDT"]

    stale = RLStrategy(model=_FixedPolicy(0), schema_version=OBSERVATION_SCHEMA_VERSION + 1)
    with pytest.raises(SchemaVersionError):
        stale.decide(_obs(_rise_signals()))


def test_strategy_requires_a_policy():
    with pytest.raises(sf.FlowConfigError):
        RLStrategy()


def test_make_env_reset_and_step(ds):
    base = sf.Flow(name="rl", detectors=[sf.SmaCrossDetector()])
    env = make_env(base, ds)
    vec, info = env.reset()
    assert vec.shape == env.observation_space.shape == (5,)
    assert env.action_space.n == 3
    total = 0.0
    for action in (1, 0, 2, 0):
        vec, reward, terminated, truncated, info = env.step(action)
        assert vec.shape == (5,) and not terminated and not truncated
        assert "equity" in info
        total += reward
    assert np.isfinite(total)


def test_flow_round_trip_with_pickled_policy(ds, tmp_path):
    base = sf.Flow(name="rl", detectors=[sf.SmaCrossDetector()])
    flow = base.replace(strategy=RLStrategy(model=_FixedPolicy(1), size_pct=0.05))
    run = flow.backtest(ds, capital=10_000)

    path = tmp_path / "rl.yaml"
    flow.save(str(path), model_dir=str(tmp_path / "models"))
    assert flow.strategy.policy_uri.endswith("strategy/policy.pkl")

    loaded = sf.Flow.load(str(path))
    assert isinstance(loaded.strategy, RLStrategy)
    assert loaded.strategy.size_pct == 0.05
    assert loaded.strategy.schema_version == OBSERVATION_SCHEMA_VERSION
    again = loaded.backtest(ds, capital=10_000)
    assert again.final_equity == run.final_equity
    assert len(again.fills) == len(run.fills)


def test_save_without_model_dir_raises(ds, tmp_path):
    flow = sf.Flow(name="rl", detectors=[sf.SmaCrossDetector()], strategy=RLStrategy(model=_FixedPolicy(0)))
    with pytest.raises(ArtifactError, match="model_dir"):
        flow.save(str(tmp_path / "rl.yaml"))


def test_flow_round_trip_with_sb3_policy(ds, tmp_path):
    sb3 = pytest.importorskip("stable_baselines3")
    base = sf.Flow(name="rl", detectors=[sf.SmaCrossDetector()])
    env = make_env(base, ds)
    policy = sb3.PPO("MlpPolicy", env, n_steps=32, batch_size=16, verbose=0, device="cpu", seed=0).learn(64)
    flow = base.replace(strategy=labs.RLStrategy(model=policy, size_pct=0.1))
    run = flow.backtest(ds, capital=10_000)

    path = tmp_path / "rl.yaml"
    flow.save(str(path), model_dir=str(tmp_path / "models"))
    assert flow.strategy.policy_uri.endswith("strategy/policy.zip")
    assert flow.strategy.policy_class.startswith("stable_baselines3")

    loaded = sf.Flow.load(str(path))
    assert type(loaded.strategy.model) is type(policy)
    again = loaded.backtest(ds, capital=10_000)
    assert again.final_equity == run.final_equity
    assert len(again.fills) == len(run.fills)
