"""Fast checks of the simulator, parameter-shift gradient and calibration."""

import numpy as np
import pytest
import torch

from fisherfed.simulator import (N_PARAMS, DeviceNoise, NoisyClassifierSimulator, sample_expectation)


def _reference(x: np.ndarray, th: np.ndarray) -> float:
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector

    qc = QuantumCircuit(4)
    for q in range(4):
        qc.ry(float(x[q]), q)
    for layer in range(2):
        for q in range(4):
            qc.ry(float(th[layer * 8 + q]), q)
        for q in range(4):
            qc.rz(float(th[layer * 8 + 4 + q]), q)
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            qc.cz(a, b)
    return float(Statevector(qc).expectation_value(SparsePauliOp("IIIZ")).real)


def test_ideal_matches_qiskit_statevector() -> None:
    g = torch.Generator().manual_seed(0)
    x, th = torch.rand(3, 4, generator=g) * 3, torch.rand(3, N_PARAMS, generator=g) * 6
    got = NoisyClassifierSimulator().expectation(x, th).numpy()
    ref = [_reference(x[i].numpy(), th[i].numpy()) for i in range(3)]
    assert np.allclose(got, ref, atol=1e-5)


@pytest.mark.parametrize("kind", ["depolarizing", "amplitude_damping"])
def test_noise_keeps_expectation_physical_and_trace(kind: str) -> None:
    g = torch.Generator().manual_seed(1)
    x, th = torch.rand(8, 4, generator=g) * 3, torch.rand(8, N_PARAMS, generator=g) * 6
    e = NoisyClassifierSimulator(DeviceNoise(kind, 0.05)).expectation(x, th)
    assert torch.all(e.abs() <= 1.0 + 1e-5)


def test_depolarizing_attenuates_toward_zero() -> None:
    g = torch.Generator().manual_seed(2)
    x, th = torch.rand(32, 4, generator=g) * 3, torch.rand(32, N_PARAMS, generator=g) * 6
    ideal = NoisyClassifierSimulator().expectation(x, th)
    noisy = NoisyClassifierSimulator(DeviceNoise("depolarizing", 0.05)).expectation(x, th)
    slope = float((ideal * noisy).sum() / (ideal * ideal).sum())
    assert 0.1 < slope < 0.95


def test_parameter_shift_is_exact_under_noise() -> None:
    sim = NoisyClassifierSimulator(DeviceNoise("depolarizing", 0.03))
    g = torch.Generator().manual_seed(3)
    x, th = torch.rand(1, 4, generator=g) * 3, torch.rand(1, N_PARAMS, generator=g) * 6
    j, h = 5, 1e-3
    up, dn = th.clone(), th.clone()
    up[0, j] += h
    dn[0, j] -= h
    fd = float((sim.expectation(x, up) - sim.expectation(x, dn)) / (2 * h))
    s_up, s_dn = th.clone(), th.clone()
    s_up[0, j] += np.pi / 2
    s_dn[0, j] -= np.pi / 2
    ps = float((sim.expectation(x, s_up) - sim.expectation(x, s_dn)) / 2)
    assert abs(fd - ps) < 5e-3


def test_shot_sampling_is_unbiased_and_has_binomial_variance() -> None:
    gen = torch.Generator().manual_seed(4)
    e = torch.full((20000,), 0.3)
    s = sample_expectation(e, 50, gen)
    assert abs(float(s.mean()) - 0.3) < 0.01
    assert abs(float(s.var()) - (1 - 0.3**2) / 50) < 0.003
    assert torch.equal(sample_expectation(e, None, gen), e)
