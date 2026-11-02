"""Fast checks of the federated pipeline."""

import numpy as np
import torch

from fisherfed.data import dirichlet_partition, make_federated_data
from fisherfed.federated import (AGGREGATORS, FedConfig, calibrate_attenuation, client_gradient,
                                 client_noise_levels, run_federated)
from fisherfed.simulator import N_PARAMS, DeviceNoise, NoisyClassifierSimulator


def test_partition_is_disjoint_and_complete() -> None:
    y = np.random.default_rng(0).integers(0, 2, 300)
    parts = dirichlet_partition(y, 4, 1.0, np.random.default_rng(1))
    allidx = np.concatenate(parts)
    assert len(allidx) == len(set(allidx.tolist())) == 300
    assert min(len(p) for p in parts) >= 8


def test_noise_levels_have_requested_spread() -> None:
    cfg = FedConfig(n_clients=6, sigma=0.8, p_med=0.04)
    p = client_noise_levels(cfg, np.random.default_rng(0))
    assert abs(np.log(p).mean() - np.log(0.04)) < 1e-6
    assert abs(np.log(p).std() - 0.8) < 1e-6
    assert np.allclose(client_noise_levels(FedConfig(sigma=0.0), np.random.default_rng(0)), 0.04)


def test_calibration_recovers_attenuation() -> None:
    sim = NoisyClassifierSimulator(DeviceNoise("depolarizing", 0.05))
    gen = torch.Generator().manual_seed(0)
    exact = calibrate_attenuation(sim, NoisyClassifierSimulator(), 100, None, gen)
    shot = calibrate_attenuation(sim, NoisyClassifierSimulator(), 40, 2000, gen)
    assert 0.1 < exact < 0.95
    assert abs(shot - exact) < 0.1


def test_exact_gradient_matches_autograd_free_finite_difference() -> None:
    sim = NoisyClassifierSimulator()
    gen = torch.Generator().manual_seed(0)
    th = torch.rand(N_PARAMS, generator=gen) * 6
    x, y = torch.rand(3, 4, generator=gen) * 3, torch.tensor([1.0, -1.0, 1.0])
    g = client_gradient(sim, th, x, y, None, gen, None)

    def loss(t: torch.Tensor) -> float:
        f = sim.expectation(x, t[None, :].expand(3, -1))
        return float(((f - y) ** 2).mean())

    h, j = 1e-2, 3
    up, dn = th.clone(), th.clone()
    up[j] += h
    dn[j] -= h
    assert abs(float(g[j]) - (loss(up) - loss(dn)) / (2 * h)) < 2e-3


def test_quick_runs_for_every_aggregator() -> None:
    cfg = FedConfig(rounds=2, eval_every=1, batch=2, shots=16, cal_probes=4, cal_shots=32)
    data = make_federated_data(0, cfg.n_clients)
    for key, agg in AGGREGATORS.items():
        r = run_federated(cfg, agg, 0, data)
        assert 0.0 <= r["final_acc"] <= 1.0 and np.isfinite(r["final_mse"]), key
        if agg.mode == "quantum":
            assert abs(sum(r["weights"]) - 1.0) < 1e-9


def test_alpha_zero_weights_follow_data_size() -> None:
    cfg = FedConfig(rounds=1, eval_every=1, batch=2, shots=8, cal_probes=2, cal_shots=8)
    data = make_federated_data(1, cfg.n_clients)
    r = run_federated(cfg, AGGREGATORS["fedavg"], 1, data)
    n = np.array([len(ix) for ix in data.client_idx], dtype=float)
    assert np.allclose(r["weights"], n / n.sum())


def test_label_skew_parameter_changes_the_split() -> None:
    def skew(beta: float) -> float:
        d = make_federated_data(0, 4, beta)
        frac = [float((d.y_train[ix] > 0).float().mean()) for ix in d.client_idx]
        return float(np.std(frac))

    assert skew(0.1) > skew(100.0)
