import os

import cudaq
import numpy as np


class Adapter:
    def __init__(self, config):
        self.config = config
        self.calls = self.circuits = 0
        self.rng = np.random.default_rng(config.seed)
        cudaq.set_target(os.getenv("CUDAQ_TARGET", "nvidia"))

    def probabilities(self, circuits, batch_size=1):
        if not 1 <= batch_size <= 20:
            raise ValueError("CUDA-Q statevector batches must have 1 to 20 qubits")
        self.calls += 1
        self.circuits += len(circuits)
        results = []
        for start in range(0, len(circuits), batch_size):
            batch = circuits[start : start + batch_size]
            kernel = cudaq.make_kernel()
            qubits = kernel.qalloc(len(batch))
            for q, (a, z, b) in enumerate(batch):
                kernel.ry(a, qubits[q])
                kernel.rz(z, qubits[q])
                kernel.ry(b, qubits[q])
            probabilities = np.abs(np.asarray(cudaq.get_state(kernel))) ** 2
            indices = np.arange(len(probabilities), dtype=np.int64)
            results.extend(
                float(probabilities[((indices >> q) & 1) == 0].sum())
                for q in range(len(batch))
            )
        p = np.clip(results, 0, 1)
        return (
            self.rng.binomial(self.config.shots, p) / self.config.shots
            if self.config.shots
            else p
        )
