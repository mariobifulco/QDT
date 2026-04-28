from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


CircuitSpec = tuple[float, float, float]


def _run_batch(circuits: list[CircuitSpec]) -> list[bool]:
    if not circuits:
        return []

    stitched = QuantumCircuit(len(circuits))
    for qubit, (feature_angle, theta_z, theta_y) in enumerate(circuits):
        stitched.ry(feature_angle, qubit)
        stitched.rz(theta_z, qubit)
        stitched.ry(theta_y, qubit)

    statevector = Statevector.from_instruction(stitched)
    return [statevector.probabilities([qubit])[0] >= 0.5 for qubit in range(len(circuits))]


def stitch_and_run(circuits: list[CircuitSpec], batch_size: int = 1) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    results: list[bool] = []
    for start in range(0, len(circuits), batch_size):
        results.extend(_run_batch(circuits[start:start + batch_size]))
    return np.asarray(results, dtype=bool)
