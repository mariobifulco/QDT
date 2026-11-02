import json
from dataclasses import dataclass
from pathlib import Path

from .circuit import Circuit
from .data import DATASET_KEYS


@dataclass(frozen=True)
class Config:
    circuit: Circuit
    backend: str = "aer"
    ibm_backend: str | None = None
    job_journal: str | None = None
    max_qpu_jobs: int = 0
    max_qpu_shots: int = 0
    qpu_timeout_s: int = 300
    qpu_max_execution_time_s: int = 300
    dataset: str = "iris"
    synthetic_samples: int = 600
    data_seed: int = 1729
    data_dir: str = "datasets"
    seed: int = 42
    validation_size: float = (
        0.0  # Fraction of the full dataset; zero keeps legacy splits.
    )
    split_seed: int | None = None
    optimizer_seed: int | None = None
    simulator_seed: int | None = None
    transpiler_seed: int = 0
    test_size: float = 0.2
    max_train_samples: int | None = None
    batch_size: int = 64  # Composite circuits per backend request.
    packing_factor: int = 1
    aer_device: str = "CPU"
    aer_method: str = "automatic"
    max_memory_mb: int = 4096
    parallel_experiments: int = 1
    parallel_shots: int = 1
    shots: int = 0  # Exact probabilities; positive values enable shot sampling.
    depolarizing: float = 0.0
    amplitude_damping: float = 0.0
    phase_damping: float = 0.0
    readout_error: float = 0.0
    fake_qpu: str | None = None  # Bundled provider:name or file:path; always local Aer.
    qpu_layout: tuple[int, ...] | None = None
    threads: int = 1
    max_evals: int = 100  # Per candidate and node, shared across restarts.
    restarts: int = 2
    max_depth: int = 2
    min_samples_leaf: int = 2
    top_k: int | None = 4  # Initial supervised filter; null retains all active columns.
    corr_threshold: float = 0.95

    def __post_init__(self):
        for name in (
            "seed",
            "max_qpu_jobs",
            "max_qpu_shots",
            "qpu_timeout_s",
            "qpu_max_execution_time_s",
            "synthetic_samples",
            "data_seed",
            "split_seed",
            "optimizer_seed",
            "simulator_seed",
            "transpiler_seed",
            "batch_size",
            "packing_factor",
            "max_memory_mb",
            "parallel_experiments",
            "parallel_shots",
            "shots",
            "threads",
            "max_evals",
            "restarts",
            "max_depth",
            "min_samples_leaf",
            "max_train_samples",
            "top_k",
        ):
            value = getattr(self, name)
            if value is None and name in {
                "max_train_samples",
                "top_k",
                "split_seed",
                "optimizer_seed",
                "simulator_seed",
            }:
                continue
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer")
        if isinstance(self.circuit, dict):
            object.__setattr__(self, "circuit", Circuit(**self.circuit))
        if not isinstance(self.circuit, Circuit):
            raise ValueError("circuit must be a Circuit or its JSON definition")
        if min(self.max_memory_mb, self.parallel_experiments, self.parallel_shots) < 1:
            raise ValueError("Aer memory and parallelism limits must be positive")
        if self.synthetic_samples < 20 or self.data_seed < 0:
            raise ValueError(
                "synthetic_samples must be >= 20 and data_seed nonnegative"
            )
        if self.packing_factor < 1:
            raise ValueError("packing_factor must be positive")
        if self.aer_device not in {"CPU", "GPU"}:
            raise ValueError("aer_device must be CPU or GPU")
        if self.aer_method not in {"automatic", "statevector", "density_matrix"}:
            raise ValueError("Unsupported Aer simulation method")
        if (
            self.aer_method == "statevector"
            and any(self.noise_rates[:3])
            and not self.shots
        ):
            raise ValueError("Exact gate-noise simulation requires density_matrix")
        if self.backend not in {"aer", "ibm"}:
            raise ValueError(
                "backend must be aer or ibm; use fake_qpu for local profiles"
            )
        if (
            min(self.max_qpu_jobs, self.max_qpu_shots) < 0
            or min(self.qpu_timeout_s, self.qpu_max_execution_time_s) < 1
        ):
            raise ValueError("QPU budgets must be nonnegative and timeouts positive")
        if self.backend == "ibm":
            if (
                not isinstance(self.ibm_backend, str)
                or not self.ibm_backend.strip()
                or not isinstance(self.job_journal, str)
                or not self.job_journal.strip()
                or min(self.shots, self.max_qpu_jobs, self.max_qpu_shots) < 1
                or self.fake_qpu is not None
                or any(self.noise_rates)
            ):
                raise ValueError(
                    "IBM inference needs a backend, journal, positive shots/job budgets, and no simulated noise"
                )
        if self.fake_qpu is not None:
            if not isinstance(self.fake_qpu, str) or not self.fake_qpu.strip():
                raise ValueError(
                    "fake_qpu must be a nonempty provider:name or file:path"
                )
            if self.backend != "aer" or any(self.noise_rates):
                raise ValueError(
                    "fake_qpu requires backend=aer and custom noise rates=0"
                )
        if self.qpu_layout is not None:
            if (
                (self.fake_qpu is None and self.backend != "ibm")
                or not isinstance(self.qpu_layout, (list, tuple))
                or len(self.qpu_layout) != self.packing_factor * self.circuit.num_qubits
                or any(type(q) is not int or q < 0 for q in self.qpu_layout)
                or len(set(self.qpu_layout)) != len(self.qpu_layout)
            ):
                raise ValueError(
                    "qpu_layout needs distinct physical qubits for every packed circuit qubit"
                )
            object.__setattr__(self, "qpu_layout", tuple(self.qpu_layout))
        if self.dataset not in DATASET_KEYS:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        for name in (
            "split_seed",
            "optimizer_seed",
            "simulator_seed",
            "transpiler_seed",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be nonnegative")
        minimum = self.circuit.num_parameters + 2
        if self.restarts < 1 or self.max_evals < minimum * self.restarts:
            raise ValueError(
                f"max_evals must be >= {minimum} * restarts for this circuit"
            )
        if self.top_k is not None and self.top_k < self.circuit.num_features:
            raise ValueError("top_k must retain at least circuit.num_features columns")
        if self.seed < 0 or self.batch_size < 1 or self.shots < 0 or self.threads < 1:
            raise ValueError("Invalid seed, batch size, shots or threads")
        if (
            self.max_depth < 0
            or self.min_samples_leaf < 1
            or not 0 < self.test_size < 1
            or not 0 <= self.validation_size < 1 - self.test_size
            or (self.split_seed is not None and self.split_seed < 0)
        ):
            raise ValueError("Invalid tree size or test fraction")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be positive or null")
        if not 0 < self.corr_threshold <= 1:
            raise ValueError("corr_threshold must be in (0, 1]")
        if self.max_train_samples is not None and self.max_train_samples < 2:
            raise ValueError("max_train_samples must be >= 2")
        if any(not 0 <= rate <= 1 for rate in self.noise_rates):
            raise ValueError("Noise probabilities must be in [0, 1]")

    def seed_for(self, stage):
        value = getattr(self, f"{stage}_seed")
        return self.seed if value is None else value

    @property
    def noise_rates(self):
        return (
            self.depolarizing,
            self.amplitude_damping,
            self.phase_damping,
            self.readout_error,
        )

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))
