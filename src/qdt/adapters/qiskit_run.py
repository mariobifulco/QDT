import numpy as np

from . import CircuitSpec


class Adapter:
    def __init__(self, config):
        self.config = config
        self.rng = np.random.default_rng(self.config.seed)
        self.calls = 0
        self.circuits = 0
        self._backend = None

    def _get_backend(self):
        if self._backend is None:
            from qiskit_aer import AerSimulator
            from qiskit_aer.noise import (
                NoiseModel,
                ReadoutError,
                amplitude_damping_error,
                depolarizing_error,
                phase_damping_error,
            )

            cfg = self.config
            noise = NoiseModel(basis_gates=["ry", "rz"])
            error = None
            for rate, channel in (
                (cfg.depolarizing, lambda p: depolarizing_error(p, 1)),
                (cfg.amplitude_damping, amplitude_damping_error),
                (cfg.phase_damping, phase_damping_error),
            ):
                if rate:
                    next_error = channel(rate)
                    error = next_error if error is None else error.compose(next_error)
            if error is not None:
                noise.add_all_qubit_quantum_error(error, ["ry", "rz"])
            if cfg.readout_error:
                p = cfg.readout_error
                noise.add_all_qubit_readout_error(
                    ReadoutError([[1 - p, p], [p, 1 - p]])
                )
            method = (
                "matrix_product_state"
                if cfg.backend == "aer_stitched" and cfg.shots
                else "density_matrix"
            )
            self._backend = AerSimulator(
                method=method,
                noise_model=noise,
                max_parallel_threads=cfg.threads,
                fusion_enable=False,
            )
        return self._backend

    def probabilities(self, circuits: list[CircuitSpec], batch_size: int = 1):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if (
            self.config.backend == "aer_stitched"
            and self.config.shots == 0
            and min(batch_size, len(circuits)) > 12
        ):
            raise ValueError(
                "Exact stitched density matrices are limited to 12 qubits; "
                "use factorized aer or finite shots"
            )
        angles = np.asarray(circuits, dtype=float)
        if not len(angles):
            return np.empty(0, dtype=float)
        if angles.shape != (len(circuits), 3) or not np.isfinite(angles).all():
            raise ValueError(
                "Expected finite (feature_angle, theta_z, theta_y) triples"
            )
        self.calls += 1
        self.circuits += len(circuits)
        if self.config.backend == "analytic":
            a, z, b = angles.T
            p = np.clip(
                (1 + np.cos(a) * np.cos(b) - np.sin(a) * np.cos(z) * np.sin(b)) / 2,
                0,
                1,
            )
            if self.config.shots:
                p = self.rng.binomial(self.config.shots, p) / self.config.shots
            return p

        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Pauli

        backend = self._get_backend()
        probabilities = []
        for start in range(0, len(circuits), batch_size):
            batch = circuits[start : start + batch_size]
            groups = (
                [batch]
                if self.config.backend == "aer_stitched"
                else [[c] for c in batch]
            )
            experiments = []
            for group in groups:
                qc = QuantumCircuit(len(group))
                for q, (a, z, b) in enumerate(group):
                    qc.ry(a, q)
                    qc.rz(z, q)
                    qc.ry(b, q)
                    if not self.config.shots:
                        qc.save_expectation_value(Pauli("Z"), [q], label=f"z{q}")
                if self.config.shots:
                    qc.measure_all()
                experiments.append(qc)
            result = backend.run(
                experiments,
                shots=self.config.shots or 1,
                seed_simulator=int(self.rng.integers(0, 2**31 - 1)),
            ).result()
            if not result.success:
                raise RuntimeError(f"Aer simulation failed: {result.status}")
            for i, group in enumerate(groups):
                if self.config.shots:
                    counts = result.get_counts(i)
                    total = sum(counts.values())
                    for q in range(len(group)):
                        probabilities.append(
                            sum(
                                n
                                for bits, n in counts.items()
                                if bits.replace(" ", "")[-1 - q] == "0"
                            )
                            / total
                        )
                else:
                    for q in range(len(group)):
                        p = (1 + float(result.data(i)[f"z{q}"])) / 2
                        r = self.config.readout_error
                        probabilities.append(p * (1 - r) + (1 - p) * r)
        return np.clip(probabilities, 0, 1)
