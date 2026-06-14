<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg" width="120">
  <source media="(prefers-color-scheme: light)" srcset="assets/logo.svg" width="120">
  <img alt="SignalFlow" src="assets/logo.png" width="120">
</picture>

# signalflow-labs

**Neural-network & RL extension for SignalFlow - 14 encoders, 7 heads, 4 losses, RL strategy**

<p>
<a href="https://pypi.org/project/signalflow-labs/"><img src="https://img.shields.io/badge/version-0.8.2-7c3aed" alt="Version"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-3b82f6?logo=python&logoColor=white" alt="Python 3.12+"></a>
<img src="https://img.shields.io/badge/pytorch-ef4444?logo=pytorch&logoColor=white" alt="PyTorch">
<img src="https://img.shields.io/badge/lightning-792ee5?logo=lightning&logoColor=white" alt="Lightning">
</p>

</div>

---

Part of the [SignalFlow](https://github.com/pathway2nothing/sf-project) ecosystem.

A plugin: a PyTorch/Lightning library for financial time-series classification
(modular encoders, classification heads, loss functions) plus a reinforcement-learning
strategy. Installing it auto-registers its components with the core `signalflow`
registry via entry points.

## Installation

```bash
pip install signalflow-labs           # core: torch + lightning
pip install "signalflow-labs[rl]"     # + stable-baselines3, gymnasium (RL strategy)
# or, from the core:
pip install "signalflow-trading[labs]"
```

**Requires:** Python ≥ 3.12, signalflow-trading ≥ 0.8.0, PyTorch ≥ 2.2, Lightning ≥ 2.5.

## Quick Start

```python
from signalflow.labs.encoder import TransformerEncoder
from signalflow.labs.head import MLPClassifierHead
from signalflow.labs.model import TemporalClassificator
from signalflow.labs.data import SignalDataModule
import lightning as pl

# Create model
model = TemporalClassificator(
    encoder_type="encoder/transformer",
    encoder_params={"d_model": 64, "nhead": 4, "num_layers": 2},
    head_type="head/cls/mlp",
    head_params={"hidden_sizes": [32]},
    num_classes=3,  # fall, neutral, rise
)

# Create data module
dm = SignalDataModule(
    data=df,
    window_size=60,
    batch_size=32,
    split_strategy="temporal",
)

# Train
trainer = pl.Trainer(max_epochs=50, accelerator="auto")
trainer.fit(model, dm)
```

## Encoders (14)

| Encoder | Architecture | Best For |
|---------|-------------|----------|
| `LSTMEncoder` | Bidirectional LSTM | Sequential patterns |
| `GRUEncoder` | Gated Recurrent Unit | Faster training |
| `TCNEncoder` | Temporal Convolutional Network | Long-range dependencies |
| `TransformerEncoder` | Self-attention + positional encoding | Complex relationships |
| `PatchTSTEncoder` | Patch-based Transformer | Multivariate time series |
| `TSMixerEncoder` | All-MLP (Google 2023) | Efficient mixing |
| `InceptionTimeEncoder` | Multi-scale convolutions | Multi-resolution features |
| `ResNet1dEncoder` | 1D ResNet | Deep representations |
| `XceptionTimeEncoder` | Depthwise separable conv | Efficient computation |
| `Conv1dEncoder` | 1D CNN | Local patterns |
| `XCMEncoder` | Cross-Channel Mixing | Channel interactions |
| `gMLPEncoder` | Gating MLP | Spatial/channel gating |
| `OmniScaleCNNEncoder` | Multi-scale CNN | Scale-invariant features |
| `ConvTranEncoder` | Conv + Transformer hybrid | Combined strengths |

## Classification Heads (7)

| Head | Use Case |
|------|----------|
| `LinearClassifierHead` | Simple baseline |
| `MLPClassifierHead` | Non-linear classification |
| `ResidualClassifierHead` | Deep with skip connections |
| `AttentionClassifierHead` | Attention-weighted pooling |
| `OrdinalRegressionHead` | Ordered classes (fall < neutral < rise) |
| `DistributionHead` | Probability distributions |
| `ClassificationWithConfidenceHead` | Class + confidence score |

## Loss Functions (4)

| Loss | Purpose |
|------|---------|
| `FocalLoss` | Class imbalance - down-weights easy examples |
| `DiceLoss` | Imbalanced multi-class |
| `LDAMLoss` | Large margin for rare classes |
| `SymmetricCrossEntropyLoss` | Noisy labels |

## SignalFlow Integration

Importing `signalflow.labs` registers its components. The RL strategy plugs into a
`Flow` as the strategy slot (needs the `[rl]` extra):

```python
import signalflow as sf
import signalflow.labs as labs               # registers neural + RL components
from stable_baselines3 import PPO

base = sf.Flow(name="rl", detectors=[sf.SmaCrossDetector()])
env = labs.make_env(base, ds)                # gymnasium env over an Engine replay
policy = PPO("MlpPolicy", env).learn(10_000)

flow = base.replace(strategy=labs.RLStrategy(model=policy, size_pct=0.1))
run = flow.backtest(ds, capital=50_000)
print(run.scorecard())
```

The neural `TemporalClassificator` can also back a `ForecastModel` for signal
validation; see `signalflow.labs.validator`.

## Package Structure

| Module | Description |
|--------|-------------|
| `signalflow.labs.data` | Data loading, windowing, temporal splitting |
| `signalflow.labs.encoder` | 14 feature encoding architectures |
| `signalflow.labs.head` | 7 output head architectures |
| `signalflow.labs.layer` | Custom neural network layers |
| `signalflow.labs.loss` | 4 specialized loss functions |
| `signalflow.labs.model` | `TemporalClassificator` - complete model |
| `signalflow.labs.validator` | SignalFlow validator integration |
| `signalflow.labs.strategy` | `RLStrategy` + `make_env` (RL) |
| `signalflow.labs.backend` | `TorchMLPBackend` for `ForecastModel` |

---

**License:** MIT &ensp;·&ensp; Part of [SignalFlow](https://github.com/pathway2nothing/sf-project)
