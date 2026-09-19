from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from iqm.qiskit_iqm import IQMProvider, transpile_to_IQM
from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from ..types import CircuitSpec


@lru_cache(maxsize=1)
def get_backend() -> IQMBackendBase:
    """Create one backend per process; QPU execution must be explicitly enabled."""
    mode = os.getenv("IQM_BACKEND", "fake").lower()
    if mode == "fake":
        return IQMFakeGarnet()
    if mode == "sim":
        return AerSimulator()
    if mode != "qpu":
        raise ValueError("IQM_BACKEND must be 'fake' or 'qpu'")
    url = os.getenv("IQM_SERVER_URL")
    if not url:
        raise ValueError("IQM_SERVER_URL is required when IQM_BACKEND=qpu")
    # IQMProvider reads authentication (including IQM_TOKEN) from the environment.
    return IQMProvider(
        url=url,
        quantum_computer=os.getenv("IQM_QUANTUM_COMPUTER", "garnet"),
    ).get_backend()


def stitch_and_run(
    circuits: list[CircuitSpec],
    batch_size: int = 1,
    *,
    shots: int | None = None,
) -> np.ndarray:
    """Estimate P(0) >= 0.5 for each sample using measured IQM circuits."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if shots is None:
        shots = int(os.getenv("IQM_SHOTS", "1024"))
    if shots < 1:
        raise ValueError("shots must be >= 1")
    if not circuits:
        return np.asarray([], dtype=bool)

    backend = get_backend()
    if batch_size > backend.num_qubits:
        raise ValueError(
            f"batch_size must be <= {backend.num_qubits} for this IQM backend"
        )

    results: list[bool] = []
    for start in range(0, len(circuits), batch_size):
        batch = circuits[start : start + batch_size]
        stitched = QuantumCircuit(len(batch), len(batch))
        for qubit, (feature_angle, theta_z, theta_y) in enumerate(batch):
            stitched.ry(feature_angle, qubit)
            stitched.rz(theta_z, qubit)
            stitched.ry(theta_y, qubit)
            stitched.measure(qubit, qubit)

        compiled = transpile_to_IQM(stitched, backend=backend)
        counts = backend.run(compiled, shots=shots).result().get_counts()
        total_shots = sum(counts.values())
        if total_shots == 0:
            raise RuntimeError("IQM backend returned no measurement counts")
        # Qiskit displays classical bit 0 on the right, regardless of qubit layout.
        results.extend(
            sum(count for bits, count in counts.items() if bits[-1 - qubit] == "0")
            >= total_shots / 2
            for qubit in range(len(batch))
        )
    return np.asarray(results, dtype=bool)
