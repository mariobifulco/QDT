import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    amplitude_damping_error,
    depolarizing_error,
    phase_damping_error,
)

from . import circuit_array


def gate_error(config):
    error = None
    for rate, channel in (
        (config.depolarizing, lambda p: depolarizing_error(p, 1)),
        (config.amplitude_damping, amplitude_damping_error),
        (config.phase_damping, phase_damping_error),
    ):
        if rate:
            next_error = channel(rate)
            error = next_error if error is None else error.compose(next_error)
    return error


class Adapter:
    def __init__(self, config):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.calls = self.circuits = 0
        self.device_name = (
            "density_matrix" if any(config.noise_rates) else "statevector"
        )
        noise = NoiseModel(basis_gates=["ry", "rz"])
        error = gate_error(config)
        if error is not None:
            noise.add_all_qubit_quantum_error(error, ["ry", "rz"])
        if config.readout_error:
            p = config.readout_error
            noise.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
        self.backend = AerSimulator(
            method=self.device_name,
            noise_model=noise,
            max_parallel_threads=config.threads,
            fusion_enable=False,
        )

    def probabilities(self, circuits, batch_size=1):
        angles = circuit_array(circuits, batch_size)
        if not len(angles):
            return np.empty(0)
        self.calls += 1
        self.circuits += len(circuits)
        probabilities = []
        for start in range(0, len(circuits), batch_size):
            experiments = []
            for a, z, b in angles[start : start + batch_size]:
                qc = QuantumCircuit(1)
                qc.ry(a, 0)
                qc.rz(z, 0)
                qc.ry(b, 0)
                if self.config.shots:
                    qc.measure_all()
                else:
                    qc.save_expectation_value(Pauli("Z"), [0], label="z")
                experiments.append(qc)
            result = self.backend.run(
                experiments,
                shots=self.config.shots or 1,
                seed_simulator=int(self.rng.integers(0, 2**31 - 1)),
            ).result()
            if not result.success:
                raise RuntimeError(f"Aer simulation failed: {result.status}")
            for index in range(len(experiments)):
                if self.config.shots:
                    counts = result.get_counts(index)
                    total = sum(counts.values())
                    if not total:
                        raise RuntimeError("Aer returned no measurement counts")
                    p = counts.get("0", 0) / total
                else:
                    p = (1 + float(result.data(index)["z"])) / 2
                    r = self.config.readout_error
                    p = p * (1 - r) + (1 - p) * r
                probabilities.append(p)
        return np.clip(probabilities, 0, 1)
