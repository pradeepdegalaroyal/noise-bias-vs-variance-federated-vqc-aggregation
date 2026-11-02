"""Data loading and non-IID client partitioning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class FederatedData:
    """Angle-encoded train/test split and the per-client index sets of the training part."""

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    client_idx: list[np.ndarray]


def dirichlet_partition(y: np.ndarray, n_clients: int, beta: float, rng: np.random.Generator,
                        min_size: int = 8) -> list[np.ndarray]:
    """Label-skewed split: per class, client shares are drawn from Dirichlet(beta)."""
    for _ in range(200):
        parts: list[list[int]] = [[] for _ in range(n_clients)]
        for c in np.unique(y):
            idx = rng.permutation(np.flatnonzero(y == c))
            share = rng.dirichlet(beta * np.ones(n_clients))
            cuts = (np.cumsum(share)[:-1] * len(idx)).astype(int)
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.array(p) for p in parts]
    raise RuntimeError("could not draw a Dirichlet partition with the requested minimum client size")


def make_federated_data(seed: int, n_clients: int = 4, beta: float = 1.0, n_features: int = 4) -> FederatedData:
    """Breast-cancer data -> standardise -> PCA(4) -> min-max angles in [0, pi]; labels in {-1, +1}."""
    x, y = load_breast_cancer(return_X_y=True)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3, stratify=y, random_state=seed)
    scaler = StandardScaler().fit(xtr)
    xtr, xte = scaler.transform(xtr), scaler.transform(xte)
    pca = PCA(n_components=n_features, random_state=seed).fit(xtr)
    xtr, xte = pca.transform(xtr), pca.transform(xte)
    mm = MinMaxScaler(feature_range=(0.0, np.pi)).fit(xtr)
    xtr, xte = np.clip(mm.transform(xtr), 0, np.pi), np.clip(mm.transform(xte), 0, np.pi)
    rng = np.random.default_rng(seed)
    idx = dirichlet_partition(ytr, n_clients, beta, rng)
    to_t = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    return FederatedData(to_t(xtr), to_t(2.0 * ytr - 1.0), to_t(xte), to_t(2.0 * yte - 1.0), idx)
