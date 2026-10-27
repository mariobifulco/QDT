import os

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator

from . import circuit_array
from .qiskit_run import Adapter as AerAdapter


class Adapter:
    def __init__(self, config):
        self.config = config
        self.circuit = config.circuit
        self.calls = self.circuits = 0
        self.rng = np.random.default_rng(config.seed)
        self.mode = os.getenv("IQM_BACKEND", "fake").lower()
        if self.mode not in {"fake", "sim", "qpu"}:
            raise ValueError("IQM_BACKEND must be fake, sim or qpu")
        self.device_name = f"iqm:{self.mode}"
        if self.mode == "sim":
            self.simulator = AerAdapter(config)
            self.device_name += f":{self.simulator.device_name}"
            return
        if config.shots == 0:
            raise ValueError(
                "IQM fake/qpu require shots > 0; use IQM_BACKEND=sim for exact probabilities"
            )
        if any(config.noise_rates):
            raise ValueError(
                "IQM fake/qpu use device noise; custom noise requires IQM_BACKEND=sim"
            )
        try:
            from iqm.qiskit_iqm import IQMProvider, transpile_to_IQM
            from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
        except ImportError as error:
            raise ImportError("IQM support requires: uv sync --extra iqm") from error
        self.transpile = transpile_to_IQM
        if self.mode == "fake":
            self.backend = IQMFakeGarnet()
            self.simulator = AerSimulator(
                method="matrix_product_state",
                noise_model=self.backend.noise_model,
                max_parallel_threads=config.threads,
            )
        else:
            url = os.getenv("IQM_SERVER_URL")
            if not url:
                raise ValueError("IQM_BACKEND=qpu requires IQM_SERVER_URL")
            self.backend = IQMProvider(
                url=url,
                quantum_computer=os.getenv("IQM_QUANTUM_COMPUTER", "garnet"),
            ).get_backend()
        self.capacity = self.backend.num_qubits // self.circuit.num_qubits
        if not self.capacity:
            raise ValueError("Circuit exceeds IQM backend qubit capacity")
        self.templates = {}

    def _template(self, size):
        if size not in self.templates:
            parameters = ParameterVector("batch", size * self.circuit.num_inputs)
            qc = QuantumCircuit(size * self.circuit.num_qubits, size)
            for sample in range(size):
                offset = sample * self.circuit.num_inputs
                bindings = dict(
                    zip(
                        self.circuit.parameters,
                        parameters[offset : offset + self.circuit.num_inputs],
                        strict=True,
                    )
                )
                circuit = self.circuit.template.assign_parameters(
                    bindings, flat_input=True
                )
                wires = list(
                    range(
                        sample * self.circuit.num_qubits,
                        (sample + 1) * self.circuit.num_qubits,
                    )
                )
                qc.compose(circuit, qubits=wires, inplace=True)
                qc.measure(wires[self.circuit.readout_qubit], sample)
            compiled = self.transpile(
                qc, backend=self.backend, seed_transpiler=self.config.seed
            )
            self.templates[size] = compiled, parameters
        return self.templates[size]

    def probabilities(self, circuits, batch_size=1):
        angles = circuit_array(circuits, batch_size, self.circuit.num_inputs)
        if not len(angles):
            return np.empty(0)
        self.calls += 1
        self.circuits += len(angles)
        if self.mode == "sim":
            return self.simulator.probabilities(angles, batch_size)
        batch_size = min(batch_size, self.capacity)
        probabilities = []
        for start in range(0, len(angles), batch_size):
            batch = angles[start : start + batch_size]
            template, parameters = self._template(len(batch))
            compiled = template.assign_parameters(
                dict(zip(parameters, batch.ravel(), strict=True)),
                flat_input=True,
                strict=False,
            )
            executor = self.simulator if self.mode == "fake" else self.backend
            options = (
                {"seed_simulator": int(self.rng.integers(0, 2**31 - 1))}
                if self.mode == "fake"
                else {}
            )
            result = executor.run(compiled, shots=self.config.shots, **options).result()
            if not result.success:
                raise RuntimeError(f"IQM execution failed: {result.status}")
            counts = result.get_counts()
            total = sum(counts.values())
            if not total:
                raise RuntimeError("IQM returned no measurement counts")
            probabilities.extend(
                sum(
                    n
                    for bits, n in counts.items()
                    if bits.replace(" ", "")[-1 - q] == "0"
                )
                / total
                for q in range(len(batch))
            )
        return np.asarray(probabilities)
