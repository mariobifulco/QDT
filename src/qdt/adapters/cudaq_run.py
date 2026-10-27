import os

import numpy as np
from qiskit.quantum_info import Kraus

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
            for gate in ("ry", "rz"):
                self.noise.add_all_qubit_channel(gate, channel)
        self.kernel, a, z, b = cudaq.make_kernel(float, float, float)
        qubit = self.kernel.qalloc(1)[0]
        self.kernel.ry(a, qubit)
        self.kernel.rz(z, qubit)
        self.kernel.ry(b, qubit)

    def probabilities(self, circuits, batch_size=1):
        angles = circuit_array(circuits, batch_size)
        if not len(angles):
            return np.empty(0)
        cudaq = self.runtime
        cudaq.set_target(self.target)
        self.calls += 1
        self.circuits += len(angles)
        probabilities = []
        for a, z, b in angles:
            arguments = (float(a), float(z), float(b))
            r = self.config.readout_error
            if self.config.shots:
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
                zeros = counts.count("0")
                if r:
                    zeros = self.rng.binomial(zeros, 1 - r) + self.rng.binomial(
                        total - zeros, r
                    )
                p = zeros / total
            else:
                z_value = cudaq.observe(
                    self.kernel,
                    cudaq.spin.z(0),
                    *arguments,
                    noise_model=self.noise,
                ).expectation()
                p = (1 + z_value) / 2
                p = p * (1 - r) + (1 - p) * r
            probabilities.append(p)
        return np.clip(probabilities, 0, 1)
