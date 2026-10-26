from __future__ import annotations

import cudaq
import numpy as np

from qdt.types import CircuitSpec

cudaq.set_target("nvidia")


def _run_batch(circuits: list[CircuitSpec]) -> list[bool]:
    if not circuits:
        return []

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(len(circuits))

    for qubit_index, (feature_angle, theta_z, theta_y) in enumerate(circuits):
        kernel.ry(feature_angle, qubits[qubit_index])
        kernel.rz(theta_z, qubits[qubit_index])
        kernel.ry(theta_y, qubits[qubit_index])

    statevector = np.array(cudaq.get_state(kernel), copy=False)
    probs = np.abs(statevector) ** 2
    indices = np.arange(len(probs), dtype=np.int64)

    results: list[bool] = []
    for qubit_index in range(len(circuits)):
        mask = ((indices >> qubit_index) & 1) == 0
        prob_zero = float(np.sum(probs[mask]))
        results.append(prob_zero >= 0.5)

    return results


def stitch_and_run(circuits: list[CircuitSpec], batch_size: int = 1) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    results: list[bool] = []
    for start in range(0, len(circuits), batch_size):
        results.extend(_run_batch(circuits[start : start + batch_size]))
    return np.asarray(results, dtype=bool)
