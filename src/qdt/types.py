from dataclasses import dataclass

CircuitSpec = tuple[float, float, float]


@dataclass(frozen=True)
class RunRecord:
    dataset: str
    batch_size: int
    train_samples: int
    test_samples: int
    original_dim: int
    selected_dim: int
    classical_accuracy: float
    classical_train_s: float
    classical_predict_s: float
    classical_total_s: float
    quantum_accuracy: float
    quantum_train_s: float
    quantum_predict_s: float
    quantum_total_s: float
    train_peak_python_mib: float
    predict_peak_python_mib: float
