import json
from dataclasses import dataclass
from pathlib import Path

from .data import DATASET_KEYS


@dataclass(frozen=True)
class Config:
    dataset: str = "iris"
    data_dir: str = "datasets"
    seed: int = 42
    test_size: float = 0.2
    max_train_samples: int | None = None
    backend: str = "analytic"
    batch_size: int = 8
    shots: int = 0
    depolarizing: float = 0.0
    amplitude_damping: float = 0.0
    phase_damping: float = 0.0
    readout_error: float = 0.0
    threads: int = 1
    optimizer: str = "cobyla"
    max_evals: int = 40
    restarts: int = 2
    max_depth: int = 2
    min_samples_leaf: int = 2
    top_k: int | None = 4  # None searches all nonconstant features.
    corr_threshold: float = 0.95

    def __post_init__(self):
        if self.dataset not in DATASET_KEYS:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        if self.backend not in {
            "analytic",
            "aer",
            "aer_stitched",
            "iqm",
            "cudaq",
            "quantinuum",
        }:
            raise ValueError(f"Unknown backend: {self.backend}")
        if self.optimizer not in {"cobyla", "lbfgs"}:
            raise ValueError("optimizer must be cobyla or lbfgs")
        minimum = 5 if self.optimizer == "lbfgs" else 4
        if self.restarts < 1 or self.max_evals < minimum * self.restarts:
            raise ValueError(f"max_evals must be >= {minimum} * restarts")
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
        rates = (
            self.depolarizing,
            self.amplitude_damping,
            self.phase_damping,
            self.readout_error,
        )
        if any(not 0 <= p <= 1 for p in rates):
            raise ValueError("Noise probabilities must be in [0, 1]")
        if self.backend not in {"aer", "aer_stitched"} and any(rates):
            raise ValueError("Configured noise channels require an Aer backend")
        if self.backend in {"iqm", "quantinuum"} and not self.shots:
            raise ValueError("This adapter requires shots > 0")
        if self.backend == "aer_stitched" and not self.shots and self.batch_size > 12:
            raise ValueError(
                "Exact stitched density matrices support at most 12 qubits"
            )

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))
