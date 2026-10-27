import numpy as np
from pytket import Circuit
from pytket.extensions.qiskit import AerBackend


class Adapter:
    def __init__(self, config):
        self.config = config
        self.calls = self.circuits = 0
        self.backend = AerBackend()
        self.backend._qiskit_backend.set_options(method="matrix_product_state")
        self.rng = np.random.default_rng(config.seed)

    def probabilities(self, circuits, batch_size=1):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.calls += 1
        self.circuits += len(circuits)
        probabilities = []
        for start in range(0, len(circuits), batch_size):
            batch = circuits[start : start + batch_size]
            qc = Circuit(len(batch), len(batch))
            for q, (a, z, b) in enumerate(batch):
                qc.Ry(a / np.pi, q).Rz(z / np.pi, q).Ry(b / np.pi, q).Measure(q, q)
            compiled = self.backend.get_compiled_circuit(qc, optimisation_level=0)
            counts = self.backend.run_circuit(
                compiled,
                n_shots=self.config.shots,
                seed=int(self.rng.integers(0, 2**31 - 1)),
            ).get_counts()
            total = sum(counts.values())
            if not total:
                raise RuntimeError("pytket returned no measurement counts")
            probabilities.extend(
                sum(n for bits, n in counts.items() if bits[q] == 0) / total
                for q in range(len(batch))
            )
        return np.asarray(probabilities)
