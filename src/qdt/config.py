import json
from dataclasses import dataclass
from pathlib import Path

from .data import DATASET_KEYS


@dataclass(frozen=True)
class Config:
    backend: str = "aer"
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
    threads: int = 1
    max_evals: int = 100  # Per candidate and node, shared across restarts.
    restarts: int = 2
    max_depth: int = 2
    min_samples_leaf: int = 2
    top_k: int | None = 4  # Initial supervised filter; null retains all active columns.
    corr_threshold: float = 0.95

    def __post_init__(self):
        if self.backend not in {"aer", "iqm", "cudaq"}:
            raise ValueError("backend must be aer, iqm or cudaq")
        if self.dataset not in DATASET_KEYS:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        if self.restarts < 1 or self.max_evals < 4 * self.restarts:
            raise ValueError("max_evals must be >= 4 * restarts")
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
