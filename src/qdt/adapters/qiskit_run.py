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
from ..packing import compose_copies, marginal_zeros, packed_batches
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
        self.rng = np.random.default_rng(config.seed_for("simulator"))
        self.calls = self.circuits = 0
        self.device_name = config.aer_method
        if self.device_name == "automatic":
            self.device_name = (
                "density_matrix" if any(config.noise_rates[:3]) else "statevector"
            )
        if config.aer_device not in AerSimulator().available_devices():
            raise ValueError(
                f"Aer device {config.aer_device} is unavailable; install a compatible Aer build"
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
            device=config.aer_device,
            noise_model=noise,
            max_parallel_threads=config.threads,
            fusion_enable=True,
            runtime_parameter_bind_enable=True,
        )
        self.last_metadata = []
        self.templates = {}
        self.template = self._template(config.packing_factor)[0]

    def _template(self, copies):
        if copies not in self.templates:
            qc, parameters, readouts = compose_copies(self.circuit, copies)
            if self.config.shots:
                from qiskit import ClassicalRegister

                qc.add_register(ClassicalRegister(copies))
                qc.measure(readouts, range(copies))
            else:
                for i, readout in enumerate(readouts):
                    qc.save_expectation_value(Pauli("Z"), [readout], label=f"z{i}")
            self.templates[copies] = qc, parameters
        return self.templates[copies]

    def probabilities(self, circuits, batch_size=1):
        angles = circuit_array(circuits, batch_size, self.circuit.num_inputs)
        if not len(angles):
            return np.empty(0)
        self.calls += 1
        self.circuits += len(angles)
        probabilities = np.empty(len(angles))
        for start, batch in packed_batches(
            angles, self.config.packing_factor, batch_size
        ):
            groups, copies, width = batch.shape
            template, parameters = self._template(copies)
            values = batch.reshape(groups, copies * width)
            bindings = {p: values[:, i].tolist() for i, p in enumerate(parameters)}
            result = self.backend.run(
                template,
                parameter_binds=[bindings],
                shots=self.config.shots or 1,
                seed_simulator=int(self.rng.integers(0, 2**31 - 1)),
            ).result()
            self.last_metadata = [record.metadata for record in result.results]
            if any(
                record.get("device") != self.config.aer_device
                for record in self.last_metadata
            ):
                raise RuntimeError("Aer executed on a different device than requested")
            if not result.success:
                raise RuntimeError(f"Aer simulation failed: {result.status}")
            for index in range(groups):
                if self.config.shots:
                    p = marginal_zeros(result.get_counts(index), copies)
                else:
                    p = np.array(
                        [
                            (1 + float(result.data(index)[f"z{i}"])) / 2
                            for i in range(copies)
                        ]
                    )
                    r = self.config.readout_error
                    p = p * (1 - r) + (1 - p) * r
                offset = start + index * copies
                probabilities[offset : offset + copies] = p
        return np.clip(probabilities, 0, 1)
