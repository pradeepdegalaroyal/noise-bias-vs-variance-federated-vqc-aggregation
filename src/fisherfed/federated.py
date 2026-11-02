"""Federated training of a variational quantum classifier on heterogeneous noisy clients.

Each round every client estimates the gradient of the squared loss on a local minibatch with the
parameter-shift rule, from shot-limited expectation values measured on its own noisy device.
The server combines the client gradients with an aggregation rule (:class:`Aggregator`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from .data import FederatedData, make_federated_data
from .simulator import N_PARAMS, DeviceNoise, NoisyClassifierSimulator, sample_expectation

__all__ = ["FedConfig", "Aggregator", "AGGREGATORS", "run_federated", "run_classical", "client_noise_levels"]


@dataclass(frozen=True)
class FedConfig:
    """Experiment configuration (one point of a sweep)."""

    n_clients: int = 4
    rounds: int = 50
    batch: int = 6
    lr: float = 0.1
    shots: int | None = 64          # shots per circuit; None means exact expectation values
    p_med: float = 0.04             # geometric-mean two-qubit error probability across clients
    sigma: float = 0.8              # spread (std of ln p) of the error probability across clients
    noise_kind: str = "depolarizing"
    beta: float = 1.0               # Dirichlet concentration of the label split (smaller = more skewed)
    cal_probes: int = 24            # random probe circuits per client for the attenuation calibration
    cal_shots: int = 512            # shots per probe circuit
    cal_error: float = 0.0          # extra log-normal error injected into the calibrated attenuation
    eval_every: int = 5


@dataclass(frozen=True)
class Aggregator:
    """Server-side rule.

    Attributes:
        name: label used in result tables.
        mode: ``"quantum"`` (noisy clients), ``"oracle"`` (ideal devices, exact gradients) or ``"classical"``.
        corrected: clients divide expectation values and derivatives by the calibrated attenuation.
        alpha: client weight is ``n_i * lam_i ** alpha`` (``alpha = 0`` is plain data-size FedAvg).
    """

    name: str
    mode: str = "quantum"
    corrected: bool = False
    alpha: float = 0.0


AGGREGATORS: dict[str, Aggregator] = {
    "fedavg": Aggregator("FedAvg (data-size weights)"),
    "reliability": Aggregator("Reliability-weighted (alpha=1)", alpha=1.0),
    "corrected": Aggregator("Attenuation-corrected, data-size weights", corrected=True),
    "fisher": Aggregator("Fisher-weighted, uncorrected (alpha=2)", alpha=2.0),
    "proposed": Aggregator("Corrected + Fisher-weighted (alpha=2)", corrected=True, alpha=2.0),
    "oracle": Aggregator("Ideal devices, exact gradients", mode="oracle"),
    "classical": Aggregator("Classical MLP (19 parameters)", mode="classical"),
}


def client_noise_levels(cfg: FedConfig, rng: np.random.Generator) -> np.ndarray:
    """Two-qubit error probabilities of the clients: log-normal with exactly ``sigma`` spread."""
    z = rng.standard_normal(cfg.n_clients)
    if cfg.n_clients > 1 and cfg.sigma > 0:
        z = (z - z.mean()) / z.std()
    else:
        z = np.zeros(cfg.n_clients)
    return np.clip(cfg.p_med * np.exp(cfg.sigma * z), 1e-4, 0.3)


def _slope(noisy: torch.Tensor, ideal: torch.Tensor) -> float:
    return float((noisy * ideal).sum() / (ideal * ideal).sum())


def calibrate_attenuation(noisy: NoisyClassifierSimulator, ideal: NoisyClassifierSimulator, probes: int,
                          shots: int | None, gen: torch.Generator) -> float:
    """Estimate the signal attenuation ``lam`` as the through-origin slope of noisy vs ideal ``<Z_0>``."""
    x = torch.rand(probes, 4, generator=gen) * np.pi
    th = torch.rand(probes, N_PARAMS, generator=gen) * 2 * np.pi
    e_noisy = sample_expectation(noisy.expectation(x, th), shots, gen)
    return float(np.clip(_slope(e_noisy, ideal.expectation(x, th)), 0.05, 1.0))


class Adam:
    """Minimal Adam optimiser (avoids importing ``torch.optim``, whose import costs seconds per worker)."""

    def __init__(self, params: list[torch.Tensor], lr: float, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8) -> None:
        self.params, self.lr, self.betas, self.eps, self.t = params, lr, betas, eps, 0
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]

    @torch.no_grad()
    def step(self) -> None:
        """Apply one update using the ``.grad`` attribute of every parameter."""
        self.t += 1
        b1, b2 = self.betas
        for p, m, v in zip(self.params, self.m, self.v):
            m.mul_(b1).add_(p.grad, alpha=1 - b1)
            v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
            p.sub_(self.lr * (m / (1 - b1**self.t)) / ((v / (1 - b2**self.t)).sqrt() + self.eps))


def _shift_matrix() -> torch.Tensor:
    shift = torch.zeros(1 + 2 * N_PARAMS, N_PARAMS)
    idx = torch.arange(N_PARAMS)
    shift[1 + idx, idx] = np.pi / 2
    shift[1 + N_PARAMS + idx, idx] = -np.pi / 2
    return shift


_SHIFT = _shift_matrix()


def client_gradient(sim: NoisyClassifierSimulator, theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
                    shots: int | None, gen: torch.Generator, lam: float | None) -> torch.Tensor:
    """Parameter-shift gradient of the mean squared loss from shot-limited noisy expectations.

    If ``lam`` is given the client works with ``f / lam`` (clipped to [-1, 1]) and ``df / lam``,
    which removes the first-order attenuation of the device.
    """
    b = x.shape[0]
    th = (theta[None, None, :] + _SHIFT[None]).expand(b, -1, -1).reshape(-1, N_PARAMS)
    xs = x[:, None, :].expand(b, 1 + 2 * N_PARAMS, 4).reshape(-1, 4)
    e = sample_expectation(sim.expectation(xs, th), shots, gen).reshape(b, 1 + 2 * N_PARAMS)
    f, df = e[:, 0], (e[:, 1:1 + N_PARAMS] - e[:, 1 + N_PARAMS:]) / 2
    if lam is not None:
        f, df = (f / lam).clamp(-1.0, 1.0), df / lam
    return (2.0 * (f - y)[:, None] * df).mean(0)


def _evaluate(sim: NoisyClassifierSimulator, theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    f = sim.expectation(x, theta[None, :].expand(x.shape[0], -1))
    return float(((f - y) ** 2).mean()), float((torch.sign(f) == y).float().mean())


def _batches(data: FederatedData, cfg: FedConfig, rng: np.random.Generator) -> list[np.ndarray]:
    return [rng.choice(ix, size=min(cfg.batch, len(ix)), replace=False) for ix in data.client_idx]


def _result(cfg: FedConfig, agg: Aggregator, seed: int, curve: list[tuple[int, float, float]],
            extra: dict[str, Any]) -> dict[str, Any]:
    return {"method": agg.name, "seed": seed, "final_mse": curve[-1][1], "final_acc": curve[-1][2],
            "curve": curve, **extra}


def run_federated(cfg: FedConfig, agg: Aggregator, seed: int, data: FederatedData | None = None) -> dict[str, Any]:
    """Train the quantum classifier federatedly and report ideal-device test loss and accuracy."""
    if agg.mode == "classical":
        return run_classical(cfg, agg, seed, data)
    data = data or make_federated_data(seed, cfg.n_clients, cfg.beta)
    rng = np.random.default_rng(seed)
    p_levels = client_noise_levels(cfg, np.random.default_rng(seed + 11))
    gen = torch.Generator().manual_seed(10_000 + seed)
    ideal = NoisyClassifierSimulator()
    oracle = agg.mode == "oracle"
    sims = [ideal if oracle else NoisyClassifierSimulator(DeviceNoise(cfg.noise_kind, float(p))) for p in p_levels]
    shots = None if oracle else cfg.shots
    n_i = np.array([len(ix) for ix in data.client_idx], dtype=float)

    lam_hat = np.ones(cfg.n_clients)
    lam_true = np.ones(cfg.n_clients)
    cal_shots = 0
    if not oracle:
        cal_gen = torch.Generator().manual_seed(20_000 + seed)
        cal_noise = np.random.default_rng(seed + 29)
        for i, sim in enumerate(sims):
            lam_hat[i] = calibrate_attenuation(sim, ideal, cfg.cal_probes, cfg.cal_shots if cfg.shots else None, cal_gen)
            lam_true[i] = calibrate_attenuation(sim, ideal, 200, None, cal_gen)
        lam_hat = np.clip(lam_hat * np.exp(cfg.cal_error * cal_noise.standard_normal(cfg.n_clients)), 0.05, 1.0)
        cal_shots = cfg.n_clients * cfg.cal_probes * (cfg.cal_shots if cfg.shots else 0)
    weights = n_i * lam_hat ** agg.alpha
    weights = weights / weights.sum()

    theta = torch.tensor(np.random.default_rng(seed + 3).normal(0.0, 1.0, N_PARAMS), dtype=torch.float32, requires_grad=True)
    opt = Adam([theta], lr=cfg.lr)
    curve: list[tuple[int, float, float]] = []
    for r in range(1, cfg.rounds + 1):
        grads = []
        for i, ix in enumerate(_batches(data, cfg, rng)):
            lam = float(lam_hat[i]) if (agg.corrected and not oracle) else None
            grads.append(client_gradient(sims[i], theta.detach(), data.x_train[ix], data.y_train[ix], shots, gen, lam))
        theta.grad = (torch.stack(grads) * torch.tensor(weights, dtype=torch.float32)[:, None]).sum(0)
        opt.step()
        if r % cfg.eval_every == 0 or r == cfg.rounds:
            mse, acc = _evaluate(ideal, theta.detach(), data.x_test, data.y_test)
            curve.append((r, mse, acc))
    per_round = cfg.n_clients * cfg.batch * (1 + 2 * N_PARAMS) * (cfg.shots or 0)
    extra = {"lam_hat": lam_hat.tolist(), "lam_true": lam_true.tolist(), "p_levels": p_levels.tolist(),
             "weights": weights.tolist(), "shots_total": per_round * cfg.rounds + cal_shots,
             "calibration_shots": cal_shots}
    return _result(cfg, agg, seed, curve, extra)


def run_classical(cfg: FedConfig, agg: Aggregator, seed: int, data: FederatedData | None = None) -> dict[str, Any]:
    """Federated Adam-SGD of a 4-3-1 tanh network (19 parameters) on the same splits and minibatches."""
    data = data or make_federated_data(seed, cfg.n_clients, cfg.beta)
    rng = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(seed + 3)
    w1 = (torch.randn(4, 3, generator=g) * 0.5).requires_grad_()
    b1 = torch.zeros(3, requires_grad=True)
    w2 = (torch.randn(3, generator=g) * 0.5).requires_grad_()
    b2 = torch.zeros((), requires_grad=True)
    params = [w1, b1, w2, b2]
    opt = Adam(params, lr=cfg.lr)
    n_i = np.array([len(ix) for ix in data.client_idx], dtype=float)
    weights = torch.tensor(n_i / n_i.sum(), dtype=torch.float32)

    def net(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(torch.tanh(torch.tanh(x - np.pi / 2) @ w1 + b1) @ w2 + b2)

    curve: list[tuple[int, float, float]] = []
    for r in range(1, cfg.rounds + 1):
        grads = []
        for ix in _batches(data, cfg, rng):
            loss = ((net(data.x_train[ix]) - data.y_train[ix]) ** 2).mean()
            grads.append(torch.autograd.grad(loss, params))
        for k, p in enumerate(params):
            p.grad = sum(weights[i] * grads[i][k] for i in range(cfg.n_clients))
        opt.step()
        if r % cfg.eval_every == 0 or r == cfg.rounds:
            with torch.no_grad():
                f = net(data.x_test)
                curve.append((r, float(((f - data.y_test) ** 2).mean()), float((torch.sign(f) == data.y_test).float().mean())))
    return _result(cfg, agg, seed, curve, {"n_params": 19})


def with_updates(cfg: FedConfig, **kw: Any) -> FedConfig:
    """Return a copy of ``cfg`` with the given fields replaced."""
    return replace(cfg, **kw)
