"""Batched density-matrix simulator for a 4-qubit hardware-efficient classifier with device noise.

The circuit is: angle encoding (RY) -> L x [RY layer, RZ layer, CZ ring] -> readout of <Z_0>.
Noise is applied through precomputed superoperators (one 256 x 256 matrix per device), so a batch
of circuits costs a handful of batched matrix products.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch

N_QUBITS = 4
DIM = 2**N_QUBITS
N_LAYERS = 2
N_PARAMS = N_LAYERS * 2 * N_QUBITS  # one RY and one RZ angle per qubit per layer
RING = [(0, 1), (1, 2), (2, 3), (3, 0)]
_KET = "abcd"
_BRA = "efgh"


@dataclass(frozen=True)
class DeviceNoise:
    """Noise model of one client device.

    Attributes:
        kind: ``"ideal"``, ``"depolarizing"`` or ``"amplitude_damping"``.
        p2: error probability per two-qubit gate (per qubit for amplitude damping).
        p1_ratio: single-qubit error probability as a fraction of ``p2``.
    """

    kind: str = "ideal"
    p2: float = 0.0
    p1_ratio: float = 0.1

    @property
    def p1(self) -> float:
        """Single-qubit error probability."""
        return self.p2 * self.p1_ratio


def _depolarize(t: torch.Tensor, qubits: Sequence[int], p: float) -> torch.Tensor:
    """Replace the reduced state on ``qubits`` by the maximally mixed state with probability ``p``."""
    kets = _KET
    bras = "".join(_KET[q] if q in qubits else _BRA[q] for q in range(N_QUBITS))
    kept = "".join(c for q, c in enumerate(_KET) if q not in qubits) + "".join(
        c for q, c in enumerate(_BRA) if q not in qubits
    )
    reduced = torch.einsum(f"z{kets}{bras}->z{kept}", t)
    ops = ",".join(f"{_KET[q]}{_BRA[q]}" for q in qubits)
    eye = [torch.eye(2, dtype=t.dtype) for _ in qubits]
    mixed = torch.einsum(f"z{kept},{ops}->z{_KET}{_BRA}", reduced, *eye) / 2 ** len(qubits)
    return (1 - p) * t + p * mixed


def _amp_damp(t: torch.Tensor, qubit: int, gamma: float) -> torch.Tensor:
    """Amplitude damping with decay probability ``gamma`` on one qubit."""
    k0 = torch.tensor([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=t.dtype)
    k1 = torch.tensor([[0, np.sqrt(gamma)], [0, 0]], dtype=t.dtype)
    k, b = _KET[qubit], _BRA[qubit]
    in_k = _KET.replace(k, "x")
    in_b = _BRA.replace(b, "y")
    out = torch.zeros_like(t)
    for kr in (k0, k1):
        out = out + torch.einsum(f"z{in_k}{in_b},{k}x,{b}y->z{_KET}{_BRA}", t, kr, kr.conj())
    return out


def _channel(rho: torch.Tensor, kind: str, qubits: Sequence[int], p: float) -> torch.Tensor:
    """Apply a noise channel to a batch of density matrices of shape (N, 16, 16)."""
    n = rho.shape[0]
    t = rho.reshape(n, *([2] * (2 * N_QUBITS)))
    if kind == "depolarizing":
        t = _depolarize(t, qubits, p)
    elif kind == "amplitude_damping":
        for q in qubits:
            t = _amp_damp(t, q, p)
    else:
        raise ValueError(f"unknown noise kind {kind}")
    return t.reshape(n, DIM, DIM)


def _superop(fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Matrix ``S`` with ``vec(channel(rho)) = vec(rho) @ S``, built from the matrix-unit basis."""
    basis = torch.eye(DIM * DIM, dtype=torch.complex128).reshape(DIM * DIM, DIM, DIM)
    s = fn(basis).reshape(DIM * DIM, DIM * DIM)
    if float(s.imag.abs().max()) > 1e-12:
        raise ValueError("superoperator of a real-Kraus channel must be real")
    return s.real.to(torch.float32)


def _bit_table() -> torch.Tensor:
    idx = torch.arange(DIM)
    return torch.stack([(idx >> (N_QUBITS - 1 - q)) & 1 for q in range(N_QUBITS)], dim=1).double()


def _cz_ring_diag(bits: torch.Tensor) -> torch.Tensor:
    phase = torch.ones(DIM, dtype=torch.float64)
    for a, b in RING:
        phase = phase * (1 - 2 * bits[:, a] * bits[:, b])
    return phase.to(torch.complex64)


class NoisyClassifierSimulator:
    """Density-matrix simulator of the encoded variational circuit on one (possibly noisy) device."""

    def __init__(self, noise: DeviceNoise | None = None) -> None:
        self.noise = noise or DeviceNoise()
        bits = _bit_table()
        self._sgn = (1.0 - 2.0 * bits).float()  # +1 for |0>, -1 for |1>
        self._z0 = self._sgn[:, 0]
        self._cz = _cz_ring_diag(bits)
        self._s_enc: torch.Tensor | None = None
        self._s_layer: torch.Tensor | None = None
        if self.noise.kind != "ideal":
            self._build_superops()

    def _noise_1q(self, rho: torch.Tensor) -> torch.Tensor:
        kind, p = self.noise.kind, self.noise.p1
        for q in range(N_QUBITS):
            rho = _channel(rho, kind, [q], p)
        return rho

    def _cz_then_noise_2q(self, rho: torch.Tensor) -> torch.Tensor:
        rho = self._noise_1q(rho)
        rho = self._cz[None, :, None] * rho * self._cz.conj()[None, None, :]
        kind, p = self.noise.kind, self.noise.p2
        if kind == "amplitude_damping":
            for q in range(N_QUBITS):
                rho = _channel(rho, kind, [q], p)
            return rho
        for pair in RING:
            rho = _channel(rho, kind, list(pair), p)
        return rho

    def _build_superops(self) -> None:
        self._s_enc = _superop(self._noise_1q)
        self._s_layer = _superop(self._cz_then_noise_2q)

    @staticmethod
    def _ry_kron(ang: torch.Tensor) -> torch.Tensor:
        c, s = torch.cos(ang / 2), torch.sin(ang / 2)
        m = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(torch.complex64)
        k = m[:, 0]
        for q in range(1, N_QUBITS):
            k = (k[:, :, None, :, None] * m[:, q][:, None, :, None, :]).reshape(k.shape[0], k.shape[1] * 2, k.shape[2] * 2)
        return k

    @staticmethod
    def _encode(x: torch.Tensor) -> torch.Tensor:
        v = torch.stack([torch.cos(x / 2), torch.sin(x / 2)], -1).to(torch.complex64)
        psi = v[:, 0]
        for q in range(1, N_QUBITS):
            psi = (psi[:, :, None] * v[:, q][:, None, :]).reshape(psi.shape[0], -1)
        return psi[:, :, None] * psi.conj()[:, None, :]

    @staticmethod
    def _apply_super(rho: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Apply a real superoperator by multiplying the stacked real and imaginary parts."""
        b = rho.shape[0]
        r = torch.view_as_real(rho.reshape(b, DIM * DIM)).permute(0, 2, 1).reshape(2 * b, DIM * DIM)
        out = (r @ s).reshape(b, 2, DIM * DIM).permute(0, 2, 1).contiguous()
        return torch.view_as_complex(out).reshape(b, DIM, DIM)

    def expectation(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Exact noisy ``<Z_0>`` for inputs ``x`` of shape (B, 4) and parameters ``theta`` of shape (B, 16)."""
        rho = self._encode(x.float())
        if self._s_enc is not None:
            rho = self._apply_super(rho, self._s_enc)
        th = theta.float().reshape(-1, N_LAYERS, 2, N_QUBITS)
        for layer in range(N_LAYERS):
            ky = self._ry_kron(th[:, layer, 0])
            phase = torch.exp(-0.5j * (th[:, layer, 1] @ self._sgn.T)).to(torch.complex64)
            u = phase[:, :, None] * ky
            rho = u @ rho @ u.mH
            if self._s_layer is not None:
                rho = self._apply_super(rho, self._s_layer)
            else:
                rho = self._cz[None, :, None] * rho * self._cz.conj()[None, None, :]
        return (torch.diagonal(rho, dim1=1, dim2=2).real * self._z0).sum(-1)


def sample_expectation(e: torch.Tensor, shots: int | None, gen: torch.Generator) -> torch.Tensor:
    """Finite-shot estimate of a +-1 observable with exact mean ``e``; ``shots=None`` returns ``e``."""
    if shots is None:
        return e
    prob = ((1 + e) / 2).clamp(0.0, 1.0)
    counts = torch.binomial(torch.full_like(prob, float(shots)), prob, generator=gen)
    return 2 * counts / shots - 1
