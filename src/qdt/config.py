import json
from dataclasses import dataclass, field
from pathlib import Path

from .circuit import Circuit
from .data import DATASET_KEYS


@dataclass(frozen=True)
class Config:
    backend: str = "aer"
    circuit: Circuit = field(default_factory=Circuit)
    dataset: str = "iris"
    data_dir: str = "datasets"
    seed: int = 42
    test_size: float = 0.2
    max_train_samples: int | None = None
    batch_size: int = 64
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
            "batch_size",
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
            if value is None and name in {"max_train_samples", "top_k"}:
                continue
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer")
        if isinstance(self.circuit, dict):
            object.__setattr__(self, "circuit", Circuit(**self.circuit))
        if not isinstance(self.circuit, Circuit):
            raise ValueError("circuit must be a Circuit or its JSON definition")
        if self.backend not in {"aer", "iqm", "cudaq"}:
            raise ValueError("backend must be aer, iqm or cudaq")
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
                self.fake_qpu is None
                or not isinstance(self.qpu_layout, (list, tuple))
                or len(self.qpu_layout) != self.circuit.num_qubits
                or any(type(q) is not int or q < 0 for q in self.qpu_layout)
                or len(set(self.qpu_layout)) != len(self.qpu_layout)
            ):
                raise ValueError(
                    "qpu_layout needs one distinct physical qubit per circuit qubit"
                )
            object.__setattr__(self, "qpu_layout", tuple(self.qpu_layout))
        if self.dataset not in DATASET_KEYS:
            raise ValueError(f"Unknown dataset: {self.dataset}")
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
