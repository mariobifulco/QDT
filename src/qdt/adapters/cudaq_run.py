import os

import numpy as np
from qiskit.quantum_info import Kraus

from ..circuit import ROTATIONS, TWO_GATES
from . import circuit_array
from .qiskit_run import gate_error


class Adapter:
    def __init__(self, config):
        try:
            import cudaq
        except ImportError as error:
            raise ImportError(
                "CUDA-Q support requires: uv sync --extra cudaq"
            ) from error
        self.runtime = cudaq
        self.config = config
        self.circuit = config.circuit
        self.calls = self.circuits = 0
        self.rng = np.random.default_rng(config.seed)
        error = gate_error(config)
        default_target = (
            "density-matrix-cpu"
            if error is not None
            else "nvidia"
            if cudaq.num_available_gpus()
            else "qpp-cpu"
        )
        self.target = os.getenv("CUDAQ_TARGET", default_target)
        cudaq.set_target(self.target)
        if not cudaq.get_target().simulator:
            raise ValueError("The CUDA-Q adapter requires a simulator target")
        if error is not None and self.target not in {"density-matrix-cpu", "nvidia"}:
            raise ValueError("Gate noise requires density-matrix-cpu or nvidia")
        if (
            error is not None
            and not config.shots
            and self.target != "density-matrix-cpu"
        ):
            raise ValueError(
                "Exact noisy probabilities require CUDAQ_TARGET=density-matrix-cpu"
            )
        self.device_name = f"cudaq:{self.target}"
        self.noise = cudaq.NoiseModel()
        if error is not None:
            channel = cudaq.KrausChannel(Kraus(error).data)
            pair_channel = cudaq.KrausChannel(Kraus(error.tensor(error)).data)
            for name in {g[0] for g in self.circuit.operations}:
                gate, controls = (name[1], 1) if name in {"cx", "cz"} else (name, 0)
                self.noise.add_all_qubit_channel(
                    gate, pair_channel if name in TWO_GATES else channel, controls
                )
        self.observable = cudaq.spin.z(self.circuit.readout_qubit)
        self.kernel, *inputs = cudaq.make_kernel(*([float] * self.circuit.num_inputs))
        qubits = self.kernel.qalloc(self.circuit.num_qubits)
        for name, *args in self.circuit.operations:
            if name in ROTATIONS:
                wire, angle = args
                angle = (
                    inputs[self.circuit.input_indices[angle]]
                    if isinstance(angle, str)
                    else float(angle)
                )
                getattr(self.kernel, name)(angle, qubits[wire])
            else:
                getattr(self.kernel, name)(*(qubits[q] for q in args))
        if error is not None:
            self.kernel.disable_quantum_optimization()

    def probabilities(self, circuits, batch_size=1):
        angles = circuit_array(circuits, batch_size, self.circuit.num_inputs)
        if not len(angles):
            return np.empty(0)
        cudaq = self.runtime
        if cudaq.get_target().name != self.target:
            cudaq.set_target(self.target)
        self.calls += 1
        self.circuits += len(angles)
        r = self.config.readout_error
        if not self.config.shots:
            probabilities = []
            for start in range(0, len(angles), batch_size):
                batch = angles[start : start + batch_size]
                results = cudaq.observe(
                    self.kernel,
                    self.observable,
                    *batch.T.tolist(),
                    noise_model=self.noise,
                )
                probabilities.extend(
                    (1 + result.expectation()) / 2 for result in results
                )
            p = np.asarray(probabilities)
            return np.clip(p * (1 - r) + (1 - p) * r, 0, 1)
        probabilities = []
        for row in angles:
            arguments = row.tolist()
            cudaq.set_random_seed(int(self.rng.integers(0, 2**31 - 1)))
            counts = cudaq.sample(
                self.kernel,
                *arguments,
                shots_count=self.config.shots,
                noise_model=self.noise,
            )
            total = sum(counts.values())
            if total != self.config.shots:
                raise RuntimeError("CUDA-Q returned an unexpected number of shots")
            zeros = sum(
                count
                for bits, count in counts.items()
                if bits[self.circuit.readout_qubit] == "0"
            )
            if r:
                zeros = self.rng.binomial(zeros, 1 - r) + self.rng.binomial(
                    total - zeros, r
                )
            probabilities.append(zeros / total)
        return np.clip(probabilities, 0, 1)
