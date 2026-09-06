"""TorchMLPBackend as a ForecastModel backend: sklearn surface, training, and flow round-trip."""

import numpy as np
import pytest
from sklearn.base import clone

import signalflow as sf
from signalflow.labs import TorchMLPBackend


@pytest.fixture(scope="module")
def ds():
    return sf.data("synthetic", pairs=["BTCUSDT", "ETHUSDT"], start="2023-01-01", end="2023-04-01", interval="1h")


def test_sklearn_surface_and_clone():
    be = TorchMLPBackend(hidden_sizes=(8,), epochs=2, lr=1e-2, seed=3)
    params = be.get_params()
    assert params == {"hidden_sizes": (8,), "epochs": 2, "lr": 1e-2, "batch_size": 256, "weight_decay": 0.0, "seed": 3}
    cloned = clone(be)
    assert cloned is not be and cloned.get_params() == params
    assert be.set_params(epochs=5).epochs == 5


def test_fit_predict_proba_shapes_and_determinism():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    a = TorchMLPBackend(hidden_sizes=(8,), epochs=3, seed=1).fit(X, y)
    b = TorchMLPBackend(hidden_sizes=(8,), epochs=3, seed=1).fit(X, y)
    pa, pb = a.predict_proba(X), b.predict_proba(X)
    assert pa.shape == (200, 2)
    assert np.allclose(pa.sum(axis=1), 1.0, atol=1e-5)
    assert np.array_equal(pa, pb)
    assert a.predict(X).shape == (200,)
    assert a.n_features_in_ == 4 and list(a.classes_) == [0, 1]


def test_forecast_model_with_torch_backend_and_flow_round_trip(ds, tmp_path):
    model = sf.ForecastModel(
        backend=TorchMLPBackend(hidden_sizes=(8,), epochs=2),
        target=sf.FixedHorizon(bars=12),
        features=sf.FeaturePipeline(sf.SMA(10), sf.SMA(20), sf.SMA(50)),
    )
    model.fit(ds)
    assert model.is_fitted
    oos = model.predict_oos(ds).get_column(model.output)
    assert oos.drop_nulls().len() > 0

    flow = sf.Flow(
        name="mlp",
        forecasts={"rise": model},
        detectors=[sf.ThresholdDetector(forecast="rise", p_min=0.5)],
        strategy=sf.RulesStrategy(),
    )
    run = flow.backtest(ds, capital=10_000)
    path = tmp_path / "mlp.yaml"
    flow.save(str(path), model_dir=str(tmp_path / "models"))
    loaded = sf.Flow.load(str(path))
    assert loaded.backtest(ds, capital=10_000).final_equity == run.final_equity
