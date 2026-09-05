"""
RLStrategy + the RL training env.

The RL training environment *is* the Engine replay: a gymnasium ``Env`` that
walks one bar at a time, builds the exact same :class:`Observation` the live loop
builds, and rewards the log-change in equity. Train/live parity is therefore
structural - the policy's observation is always ``Observation.to_vector()``, the
identical vector the live ``RLStrategy`` feeds to ``model.predict`` each bar.

The action space is a tiny ``Discrete(3)``:

* ``0`` - hold (no intents),
* ``1`` - open one position (``size_pct`` of equity) on the strongest RISE signal
  pair not already held,
* ``2`` - close all open positions.

A frozen policy carries the :data:`OBSERVATION_SCHEMA_VERSION` it trained against;
:meth:`RLStrategy.decide` asserts it against the live observation and raises
:class:`SchemaVersionError` on mismatch, so a changed observation layout fails
loudly instead of silently desyncing.
"""


import importlib
import math
import os
from dataclasses import dataclass

import cloudpickle
import numpy as np

from signalflow.decorators import strategy
from signalflow.engine.types import Intent
from signalflow.enums import RISE, IntentKind, Side
from signalflow.errors import ArtifactError, FlowConfigError, SchemaVersionError
from signalflow.strategy.base import Strategy
from signalflow.strategy.observation import OBSERVATION_SCHEMA_VERSION, Observation

_HOLD, _OPEN, _CLOSE = 0, 1, 2


def _strongest_rise(obs: Observation) -> str | None:
    """Pair of the highest-confidence RISE signal not already held, else None."""
    sig = obs.signals
    if "signal" not in sig.columns or sig.height == 0:
        return None
    held = obs.portfolio.positions
    rises = sig.filter(sig.get_column("signal") == RISE)
    if rises.height == 0:
        return None
    rows = rises.to_dicts()
    if "p_success" in rises.columns:
        rows.sort(key=lambda r: (r.get("p_success") is not None, r.get("p_success") or 0.0), reverse=True)
    for row in rows:
        pair = row["pair"]
        if pair not in held:
            return pair
    return None


def action_to_intents(action: int, obs: Observation, size_pct: float) -> list[Intent]:
    """Translate a Discrete(3) action into open/close intents for one bar."""
    port = obs.portfolio
    if action == _CLOSE:
        return [
            Intent(pair, IntentKind.CLOSE, Side.SELL, qty=pos.qty, reason="rl_close")
            for pair, pos in port.positions.items()
            if pos.qty > 0
        ]
    if action == _OPEN:
        pair = _strongest_rise(obs)
        if pair is None:
            return []
        notional = size_pct * port.equity
        if notional <= 0:
            return []
        return [Intent(pair, IntentKind.OPEN, Side.BUY, notional=notional, reason="rl_open")]
    return []


def _policy_class_path(policy: object) -> str:
    cls = type(policy)
    return f"{cls.__module__}:{cls.__qualname__}"


def _is_sb3_like(policy: object) -> bool:
    return callable(getattr(policy, "save", None)) and callable(getattr(type(policy), "load", None))


def _save_policy(policy: object, directory: str) -> tuple[str, str | None]:
    """Persist ``policy`` under ``directory``; return ``(uri, loader_class_path)``."""
    os.makedirs(directory, exist_ok=True)
    if _is_sb3_like(policy):
        path = os.path.join(directory, "policy.zip")
        policy.save(path)
        return f"file://{path}", _policy_class_path(policy)
    path = os.path.join(directory, "policy.pkl")
    with open(path, "wb") as fh:
        cloudpickle.dump(policy, fh)
    return f"file://{path}", None


def _load_policy(uri: str, policy_class: str | None) -> object:
    scheme, _, location = uri.partition("://")
    if scheme not in ("file", "") or not location:
        raise ArtifactError(f"RLStrategy: unsupported policy uri {uri!r} (only file:// is supported)")
    if not os.path.exists(location):
        raise ArtifactError(f"RLStrategy: policy artifact not found at {location}")
    if policy_class:
        module, _, qualname = policy_class.partition(":")
        cls: object = importlib.import_module(module)
        for part in qualname.split("."):
            cls = getattr(cls, part)
        return cls.load(location, device="cpu")  # type: ignore[attr-defined]
    with open(location, "rb") as fh:
        return cloudpickle.load(fh)


@strategy("rl")
@dataclass
class RLStrategy(Strategy):
    """Wrap a trained SB3-like policy (``.predict(obs_vector)``) as a strategy.

    Pass the trained policy as ``model=``; ``flow.save(path, model_dir=...)`` then
    stores it next to the forecast artifacts (SB3 ``.zip`` when the policy has
    ``save``/``load``, cloudpickle otherwise) and pins ``policy_uri`` so
    ``Flow.load`` rebuilds the strategy byte-for-byte. The observation schema
    version the policy trained against is recorded in the config, so the guard
    in :meth:`decide` survives the round-trip even though SB3 drops attributes.
    """

    model: object = None
    size_pct: float = 0.1
    policy_uri: str | None = None
    policy_class: str | None = None
    schema_version: int | None = None

    def __post_init__(self) -> None:
        if self.model is None and self.policy_uri:
            self.model = _load_policy(self.policy_uri, self.policy_class)
        if self.model is None:
            raise FlowConfigError("RLStrategy needs a trained policy: pass model= or policy_uri=")
        if self.schema_version is None:
            self.schema_version = int(getattr(self.model, "observation_schema_version", OBSERVATION_SCHEMA_VERSION))

    @property
    def _schema_version(self) -> int:
        return int(self.schema_version if self.schema_version is not None else OBSERVATION_SCHEMA_VERSION)

    def to_config(self) -> dict:
        """Config without the live policy object; the policy travels as ``policy_uri``."""
        return {
            "name": self.name,
            "params": {
                "size_pct": self.size_pct,
                "policy_uri": self.policy_uri,
                "policy_class": self.policy_class,
                "schema_version": self._schema_version,
            },
        }

    @classmethod
    def from_config(cls, cfg: dict) -> "RLStrategy":
        params = dict(cfg.get("params") or {})
        if not params.get("policy_uri"):
            raise ArtifactError("RLStrategy config has no policy_uri; save the flow with model_dir= first")
        return cls(model=None, **params)

    def save_artifacts(self, model_dir: str | None) -> None:
        """Persist the policy under ``model_dir/strategy`` and pin ``policy_uri`` (no-op once pinned)."""
        if self.policy_uri:
            return
        if not model_dir:
            raise ArtifactError("RLStrategy has no pinned policy_uri; pass model_dir= to flow.save")
        self.policy_uri, self.policy_class = _save_policy(self.model, os.path.join(model_dir, "strategy"))

    def decide(self, obs: Observation) -> list[Intent]:
        if obs.schema_version != self._schema_version:
            raise SchemaVersionError(
                f"RL policy trained on observation schema v{self._schema_version} "
                f"but received v{obs.schema_version}; retrain or reload the matching policy"
            )
        vector = obs.to_vector().reshape(1, -1)
        out = self.model.predict(vector, deterministic=True)
        action = out[0] if isinstance(out, tuple) else out
        action = int(np.asarray(action).reshape(-1)[0])
        return action_to_intents(action, obs, self.size_pct)


def make_env(flow, data, reward: str = "log_equity_delta"):
    """Build the gymnasium training env over ``flow`` + ``data`` (Engine replay)."""
    import gymnasium as gym
    from gymnasium import spaces

    from signalflow.engine.broker import SimBroker
    from signalflow.engine.engine import Engine
    from signalflow.flow.loop import enriched_signals

    try:
        from signalflow.flow.loop import orders_from_intents
    except ImportError:
        from signalflow.flow.loop import _orders as orders_from_intents

    if reward != "log_equity_delta":
        raise ValueError(f"unsupported reward {reward!r}; only 'log_equity_delta' is implemented")

    class SignalFlowEnv(gym.Env):
        metadata = {"render_modes": []}
        observation_schema_version = OBSERVATION_SCHEMA_VERSION

        def __init__(self) -> None:
            super().__init__()
            self.flow = flow
            self.data = data
            self.target = data.quote
            self.size_pct = getattr(getattr(flow.strategy, "entry", None), "size_pct", 0.1)
            self.broker = SimBroker(quote=flow.quote)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(5,), dtype=np.float64
            )
            self.action_space = spaces.Discrete(3)
            self._bars: list = []
            self._signals_by_ts: dict = {}
            self._i = 0
            self._prev_equity = 0.0
            self._peak = float("-inf")
            self._build_signals()

        def _build_signals(self) -> None:
            import polars as pl

            try:
                from signalflow.flow.loop import EMPTY_SIGNALS_SCHEMA
            except ImportError:
                from signalflow.flow.loop import _EMPTY_SIGNALS_SCHEMA as EMPTY_SIGNALS_SCHEMA

            self._empty_schema = EMPTY_SIGNALS_SCHEMA
            signals = enriched_signals(self.flow, self.data)
            by_ts: dict = {}
            if signals.height:
                for key, df in signals.group_by("ts"):
                    by_ts[key[0] if isinstance(key, tuple) else key] = df
            self._signals_by_ts = by_ts
            self._pl = pl

        def _observation(self, bar) -> Observation:
            snap = self.engine.snapshot(bar.ts, bar.prices)
            sig = self._signals_by_ts.get(bar.ts)
            if sig is None:
                sig = self._pl.DataFrame(schema=self._empty_schema)
            return Observation(bar.ts, sig, snap, {})

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.engine = Engine(self._initial_capital(), target=self.target, quote=self.data.quote)
            self._bars = list(self.data.iter_bars())
            self._i = 0
            self._peak = float("-inf")
            if not self._bars:
                self._prev_equity = 0.0
                return self.observation_space.sample() * 0.0, {}
            bar = self._bars[0]
            self._prev_equity = self.engine.equity(bar.prices)
            obs = self._observation(bar)
            return obs.to_vector(), {}

        def _initial_capital(self) -> float:
            return 10_000.0

        def step(self, action):
            bar = self._bars[self._i]
            obs = self._observation(bar)
            snap = obs.portfolio
            self._peak = max(self._peak, snap.equity)

            intents = action_to_intents(int(action), obs, self.size_pct)
            intents = self.flow.risk.clip(intents, snap, self._peak)
            fills = self.broker.execute(orders_from_intents(intents, bar.prices, bar.ts), bar)
            self.engine.apply(fills)

            equity = self.engine.equity(bar.prices)
            prev = self._prev_equity
            reward = math.log(equity / prev) if prev > 0 and equity > 0 else 0.0
            self._prev_equity = equity

            self._i += 1
            terminated = False
            truncated = self._i >= len(self._bars)
            if truncated:
                next_vec = obs.to_vector() * 0.0
            else:
                next_vec = self._observation(self._bars[self._i]).to_vector()
            info = {"equity": equity}
            return next_vec, float(reward), terminated, truncated, info

    return SignalFlowEnv()
