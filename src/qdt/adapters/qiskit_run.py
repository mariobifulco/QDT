import time

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


def aer_backend(config, method, noise):
    if config.aer_device not in AerSimulator().available_devices():
        raise ValueError(
            f"Aer device {config.aer_device} is unavailable; install a compatible Aer build"
        )
    return AerSimulator(
        method=method,
        device=config.aer_device,
        noise_model=noise,
        max_parallel_threads=config.threads,
        max_memory_mb=config.max_memory_mb,
        max_parallel_experiments=config.parallel_experiments,
        max_parallel_shots=config.parallel_shots,
        fusion_enable=True,
        runtime_parameter_bind_enable=True,
    )


class Adapter:
    def _initialize(self, config):
        self.config = config
        self.circuit = config.circuit
        self.rng = np.random.default_rng(config.seed_for("simulator"))
        self.calls = self.circuits = 0
        self.requests = self.composite_circuits = self.physical_shots = (
            self.logical_shots
        ) = 0
        self.records = []
        self.last_metadata = []
        self.templates = {}
        self.template_metadata = {}
        self.template_s = 0.0

    @property
    def statistics(self):
        return {
            "probability_calls": self.calls,
            "logical_circuits": self.circuits,
            "backend_requests": self.requests,
            "composite_circuits": self.composite_circuits,
            "composite_shots": self.physical_shots,
            "logical_shots": self.logical_shots,
            "template_s": self.template_s,
        }

    def __init__(self, config):
        self._initialize(config)
        self.device_name = config.aer_method
        if self.device_name == "automatic":
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
        self.backend = aer_backend(config, self.device_name, noise)
        self.template = self._template(config.packing_factor)[0]

    def _template(self, copies):
        if copies not in self.templates:
            start = time.perf_counter()
            qc, parameters, readouts = compose_copies(self.circuit, copies)
            if self.config.shots:
                from qiskit import ClassicalRegister

                qc.add_register(ClassicalRegister(copies))
                qc.measure(readouts, range(copies))
            else:
                for i, readout in enumerate(readouts):
                    qc.save_expectation_value(Pauli("Z"), [readout], label=f"z{i}")
            self.templates[copies] = qc, parameters
            self.template_s += time.perf_counter() - start
        return self.templates[copies]

    def _get_backend(self, copies):
        return self.backend

    def _correct_readout(self, p, copies):
        r = self.config.readout_error
        return p * (1 - r) + (1 - p) * r

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
            bindings = {
                p: values[:, i].tolist()
                for i, p in enumerate(parameters)
                if p in template.parameters
            }
            backend = self._get_backend(copies)
            self.requests += 1
            self.composite_circuits += groups
            self.physical_shots += groups * self.config.shots
            self.logical_shots += groups * copies * self.config.shots
            seed = int(self.rng.integers(0, 2**31 - 1))
            request = {
                "request": self.requests,
                "probability_call": self.calls,
                "logical_start": start,
                "requested_packing": self.config.packing_factor,
                "active_packing": copies,
                "composite_circuits": groups,
                "shots": self.config.shots,
                "seed_simulator": seed,
                "layout": self.template_metadata.get(copies),
                "experiments": [],
            }
            self.records.append(request)
            begin = time.perf_counter()
            result = backend.run(
                template if bindings else [template] * groups,
                **({"parameter_binds": [bindings]} if bindings else {}),
                shots=self.config.shots or 1,
                seed_simulator=seed,
            ).result()
            request["submit_wait_s"] = time.perf_counter() - begin
            request["aer_time_taken_s"] = result.time_taken
            request["success"] = bool(result.success)
            self.last_metadata = [record.metadata for record in result.results]
            if any(
                record.get("device") != self.config.aer_device
                for record in self.last_metadata
            ):
                raise RuntimeError("Aer executed on a different device than requested")
            if not result.success:
                raise RuntimeError(f"Aer simulation failed: {result.status}")
            if len(result.results) != groups:
                raise RuntimeError("Unexpected number of Aer results")
            for index in range(groups):
                if self.config.shots:
                    counts = result.get_counts(index)
                    if sum(counts.values()) != self.config.shots:
                        raise RuntimeError("Unexpected number of measurement shots")
                    p = marginal_zeros(counts, copies)
                else:
                    p = np.array(
                        [
                            (1 + float(result.data(index)[f"z{i}"])) / 2
                            for i in range(copies)
                        ]
                    )
                    p = self._correct_readout(p, copies)
                p = np.clip(p, 0, 1)
                offset = start + index * copies
                probabilities[offset : offset + copies] = p
                request["experiments"].append(
                    {
                        "logical_indices": list(range(offset, offset + copies)),
                        "inputs": batch[index].tolist(),
                        "probabilities": p.tolist(),
                        "counts": counts if self.config.shots else None,
                        "metadata": self.last_metadata[index],
                    }
                )
        return np.clip(probabilities, 0, 1)
