import numpy as np
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    amplitude_damping_error,
    depolarizing_error,
    phase_damping_error,
)

from ..circuit import SINGLE_GATES, TWO_GATES
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
        self.circuit = config.circuit
        self.rng = np.random.default_rng(config.seed)
        self.calls = self.circuits = 0
        self.device_name = (
            "density_matrix" if any(config.noise_rates[:3]) else "statevector"
        )
        noise = NoiseModel(basis_gates=sorted(SINGLE_GATES | TWO_GATES))
        error = gate_error(config)
        if error is not None:
            noise.add_all_qubit_quantum_error(error, sorted(SINGLE_GATES))
            noise.add_all_qubit_quantum_error(error.tensor(error), sorted(TWO_GATES))
        if config.readout_error:
            p = config.readout_error
            noise.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
        self.backend = AerSimulator(
            method=self.device_name,
            noise_model=noise,
            max_parallel_threads=config.threads,
            fusion_enable=False,
            runtime_parameter_bind_enable=True,
        )
        self.template = self.circuit.template.copy()
        if config.shots:
            from qiskit import ClassicalRegister

            self.template.add_register(ClassicalRegister(1))
            self.template.measure(self.circuit.readout_qubit, 0)
        else:
            self.template.save_expectation_value(
                Pauli("Z"), [self.circuit.readout_qubit], label="z"
            )

    def probabilities(self, circuits, batch_size=1):
        angles = circuit_array(circuits, batch_size, self.circuit.num_inputs)
        if not len(angles):
            return np.empty(0)
        self.calls += 1
        self.circuits += len(circuits)
        probabilities = []
        for start in range(0, len(circuits), batch_size):
            batch = angles[start : start + batch_size]
            bindings = {
                parameter: batch[:, column].tolist()
                for column, parameter in enumerate(self.circuit.parameters)
            }
            result = self.backend.run(
                self.template,
                parameter_binds=[bindings],
                shots=self.config.shots or 1,
                seed_simulator=int(self.rng.integers(0, 2**31 - 1)),
            ).result()
            if not result.success:
                raise RuntimeError(f"Aer simulation failed: {result.status}")
            for index in range(len(batch)):
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
