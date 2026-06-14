"""Torch MLP backend for the ForecastModel (sklearn-compatible)."""


import numpy as np


class TorchMLPBackend:
    """A small torch MLP classifier usable as a ForecastModel ``backend``.

    Trains on the 2D WoE-encoded tabular matrix the ForecastModel builds and
    exposes the sklearn estimator surface (``get_params``/``set_params``,
    ``fit``, ``predict_proba``) so ``sklearn.base.clone`` round-trips it.
    """

    def __init__(
        self,
        hidden_sizes: "tuple[int, ...]" = (64, 32),
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 256,
        weight_decay: float = 0.0,
        seed: int = 0,
    ):
        self.hidden_sizes = hidden_sizes
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.seed = seed

    def get_params(self, deep: bool = True) -> dict:
        """Return constructor parameters for ``sklearn.base.clone``."""
        return {
            "hidden_sizes": self.hidden_sizes,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
        }

    def set_params(self, **params) -> "TorchMLPBackend":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _build(self, n_features: int) -> "object":
        import torch
        from torch import nn

        layers: list = []
        prev = n_features
        for size in self.hidden_sizes:
            layers += [nn.Linear(prev, size), nn.ReLU()]
            prev = size
        layers.append(nn.Linear(prev, 2))
        return nn.Sequential(*layers)

    def fit(self, X, y, sample_weight=None) -> "TorchMLPBackend":
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        Xa = np.asarray(X, dtype=np.float32)
        ya = np.asarray(y).astype(np.int64).reshape(-1)
        self.classes_ = np.unique(ya)
        self.n_features_in_ = Xa.shape[1]
        self._net = self._build(Xa.shape[1])

        xt = torch.from_numpy(Xa)
        yt = torch.from_numpy(ya)
        wt = (
            torch.from_numpy(np.asarray(sample_weight, dtype=np.float32).reshape(-1))
            if sample_weight is not None
            else torch.ones(len(ya), dtype=torch.float32)
        )

        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        n = len(ya)
        self._net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                opt.zero_grad()
                logits = self._net(xt[idx])
                loss = (loss_fn(logits, yt[idx]) * wt[idx]).mean()
                loss.backward()
                opt.step()
        self._net.eval()
        return self

    def predict_proba(self, X) -> np.ndarray:
        import torch

        Xa = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            logits = self._net(torch.from_numpy(Xa))
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
