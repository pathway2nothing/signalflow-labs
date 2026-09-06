<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg" width="120">
  <source media="(prefers-color-scheme: light)" srcset="assets/logo.svg" width="120">
  <img alt="SignalFlow" src="assets/logo.png" width="120">
</picture>

# signalflow-labs

**Torch & RL extension for SignalFlow - a torch backend for `ForecastModel`, an RL strategy with its gymnasium env, and a parked neural time-series stack**

<p>
<a href="https://pypi.org/project/signalflow-labs/"><img src="https://img.shields.io/badge/version-0.8.5-7c3aed" alt="Version"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-3b82f6?logo=python&logoColor=white" alt="Python 3.12+"></a>
<img src="https://img.shields.io/badge/pytorch-ef4444?logo=pytorch&logoColor=white" alt="PyTorch">
<img src="https://img.shields.io/badge/lightning-792ee5?logo=lightning&logoColor=white" alt="Lightning">
</p>

</div>

---

Part of the [SignalFlow](https://github.com/pathway2nothing/sf-project) ecosystem.

A plugin for `signalflow-trading`. Installing it registers the RL strategy with
the core registry via the `signalflow.components` entry point; everything else is
importable from `signalflow.labs`.

## Installation

```bash
pip install signalflow-labs           # core: torch
pip install "signalflow-labs[rl]"     # + stable-baselines3, gymnasium (RLStrategy, make_env)
pip install "signalflow-labs[temporal]"  # + lightning (parked temporal stack)
# or, from the core:
pip install "signalflow-trading[labs]"
```

**Requires:** Python ≥ 3.12, signalflow-trading ≥ 0.8.5, PyTorch ≥ 2.2. Lightning ≥ 2.5 only with the `[temporal]` extra.

## What is live

Two components are wired into the core `Flow` contract and covered by tests.

### `TorchMLPBackend` - a torch backend for `ForecastModel`

A small MLP classifier with the sklearn estimator surface (`get_params` /
`set_params` / `fit` / `predict_proba`), so `ForecastModel` treats it like any
other backend: embargoed walk-forward, out-of-fold predictions, WoE encoding,
and `flow.save` / `Flow.load` all work unchanged.

```python
import signalflow as sf
import signalflow.labs as labs

ds = sf.dataset("synthetic", pairs=["BTCUSDT"], start="2023-01-01", interval="1h")

model = sf.ForecastModel(
    backend=labs.TorchMLPBackend(hidden_sizes=(64, 32), epochs=50, lr=1e-3),
    target=sf.FixedHorizon(bars=12),
    features=sf.FeaturePipeline(sf.SMA(10), sf.SMA(20), sf.SMA(50)),
)
model.fit(ds)

flow = sf.Flow(name="mlp_rise", forecasts={"rise": model},
               detectors=[sf.ThresholdDetector(forecast="rise", p_min=0.6)],
               strategy=sf.RulesStrategy())
print(flow.backtest(ds, capital=50_000).scorecard())
```

### `RLStrategy` + `make_env` - reinforcement learning in the strategy slot

`make_env(flow, ds)` builds a gymnasium env that *is* the Engine replay: one bar
per step, the exact `Observation.to_vector()` the live loop builds, reward =
log-change in equity, `Discrete(3)` actions (hold / open the strongest RISE pair
/ close all). Train/live parity is structural, and the observation schema version
travels with the policy so a changed layout fails loudly.

```python
from stable_baselines3 import PPO

base = sf.Flow(name="rl", detectors=[sf.SmaCrossDetector()])
env = labs.make_env(base, ds)
policy = PPO("MlpPolicy", env).learn(10_000)

flow = base.replace(strategy=labs.RLStrategy(model=policy, size_pct=0.1))
run = flow.backtest(ds, capital=50_000)

flow.save("flows/rl.yaml", model_dir="flows/models")   # policy -> flows/models/strategy/policy.zip
same = sf.Flow.load("flows/rl.yaml")                   # byte-identical backtest
```

Deploy is data here too: the policy is stored next to the forecast artifacts
(SB3 `.zip` when the policy has `save`/`load`, cloudpickle otherwise) and pinned
as `policy_uri` in the flow YAML.

## What is parked

The pre-V5 neural time-series stack is kept as importable building blocks. It is
**not registered** in the component registry (encoders and heads are bare
`nn.Module`s, not model-contract components) and its integration tests are
skipped until it is re-adapted to `Dataset` / `predict_oos`.

| Module | Contents | Status |
|--------|----------|--------|
| `signalflow.labs.encoder` | 16 sequence encoders: `Conv1d`, `ConvTran`, `gMLP`, `GRU`, `InceptionTime`, `iTransformer`, `LSTM`, `Mamba`, `OmniScaleCNN`, `PatchTST`, `ResNet1d`, `TCN`, `Transformer`, `TSMixer`, `XceptionTime`, `XCM` | usable `nn.Module`s, unit-tested |
| `signalflow.labs.head` | 7 heads: `Linear`, `MLP`, `Residual`, `Attention`, `OrdinalRegression`, `Distribution`, `ClassificationWithConfidence` | usable `nn.Module`s, unit-tested |
| `signalflow.labs.loss` | `FocalLoss`, `DiceLoss`, `LDAMLoss`, `SymmetricCrossEntropyLoss` | usable, unit-tested |
| `signalflow.labs.data` | `TimeSeriesPreprocessor`, `SignalWindowDataset`, `SignalDataModule` | parked (keyed on `timestamp`, not `ts`) |
| `signalflow.labs.model` | `TemporalClassificator` (Lightning module: encoder + head) | parked |
| `signalflow.labs.validator` | `TemporalValidator` | parked (no `predict`, never fitted) |

`signalflow.labs._compat` holds the shims that keep these modules importing
against the current core.

## Package structure

| Module | Description |
|--------|-------------|
| `signalflow.labs.backend` | `TorchMLPBackend` for `ForecastModel` |
| `signalflow.labs.strategy` | `RLStrategy`, `make_env`, `action_to_intents` |
| `signalflow.labs.encoder` / `head` / `loss` | neural building blocks (parked stack) |
| `signalflow.labs.data` / `model` / `validator` | temporal classification stack (parked) |

## Tests

```bash
pytest            # live components run; parked-stack integration tests are skipped
```

---

**License:** MIT &ensp;·&ensp; Part of [SignalFlow](https://github.com/pathway2nothing/sf-project)
