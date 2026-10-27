import os

import numpy as np
from iqm.qiskit_iqm import IQMProvider, transpile_to_IQM
from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


class Adapter:
    def __init__(self, config):
        self.config = config
        self.calls = self.circuits = 0
        self.mode = os.getenv("IQM_BACKEND", "fake").lower()
        if self.mode == "fake":
            self.backend = IQMFakeGarnet()
        elif self.mode == "sim":
            self.backend = AerSimulator()
        elif self.mode == "qpu":
            self.backend = IQMProvider(
                url=os.environ["IQM_SERVER_URL"],
                quantum_computer=os.getenv("IQM_QUANTUM_COMPUTER", "garnet"),
            ).get_backend()
        else:
            raise ValueError("IQM_BACKEND must be fake, sim or qpu")
        self.rng = np.random.default_rng(config.seed)

    def probabilities(self, circuits, batch_size=1):
        if not 1 <= batch_size <= self.backend.num_qubits:
            raise ValueError("batch_size exceeds IQM backend capacity")
        self.calls += 1
        self.circuits += len(circuits)
        probabilities = []
        for start in range(0, len(circuits), batch_size):
            batch = circuits[start : start + batch_size]
            qc = QuantumCircuit(len(batch), len(batch))
            for q, (a, z, b) in enumerate(batch):
                qc.ry(a, q)
                qc.rz(z, q)
                qc.ry(b, q)
                qc.measure(q, q)
            if self.mode == "sim":
                compiled = transpile(
                    qc,
                    self.backend,
                    optimization_level=0,
                    seed_transpiler=self.config.seed,
                )
                options = {"seed_simulator": int(self.rng.integers(0, 2**31 - 1))}
            else:
                compiled = transpile_to_IQM(qc, backend=self.backend)
                options = {}
            counts = (
                self.backend.run(compiled, shots=self.config.shots, **options)
                .result()
                .get_counts()
            )
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
