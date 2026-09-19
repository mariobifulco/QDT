from dataclasses import dataclass
from quantum_tree import fit, predict
from data import PreparedDataset
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from memory_profiler import memory_usage
from sklearn.metrics import accuracy_score
import csv
import time
from dataclasses import asdict
from pathlib import Path


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


CircuitSpec = tuple[float, float, float]


def timed(fn) -> tuple[object, float]:
    start = time.perf_counter()
    return fn(), time.perf_counter() - start


def run_quantum_experiment(
    dataset: PreparedDataset,
    classical_accuracy: float,
    classical_train_s: float,
    classical_predict_s: float,
    batch_size: int,
    depth: int,
) -> RunRecord:
    start_train = time.perf_counter()
    mem_train, tree = memory_usage(
        (
            fit,
            (dataset.X_train, dataset.y_train, 0),
            {
                "max_depth": depth,
                "batch_size": batch_size,
            },
        ),
        retval=True,
        interval=0.1,
        max_iterations=1,
    )
    end_train = time.perf_counter()
    train_s = end_train - start_train

    start_eval = time.perf_counter()
    mem_eval, predictions = memory_usage(
        (predict, (dataset.X_test, tree, batch_size)),
        retval=True,
        interval=0.1,
        max_iterations=1,
    )
    end_eval = time.perf_counter()
    eval_s = end_eval - start_eval

    return RunRecord(
        dataset=dataset.name,
        batch_size=batch_size,
        train_samples=len(dataset.X_train),
        test_samples=len(dataset.X_test),
        original_dim=dataset.original_dim,
        selected_dim=dataset.selected_dim,
        classical_accuracy=classical_accuracy,
        classical_train_s=classical_train_s,
        classical_predict_s=classical_predict_s,
        classical_total_s=classical_train_s + classical_predict_s,
        quantum_accuracy=float(accuracy_score(dataset.y_test, predictions)),
        quantum_train_s=train_s,
        quantum_predict_s=eval_s,
        quantum_total_s=train_s + eval_s,
        train_peak_python_mib=max(mem_train),
        predict_peak_python_mib=max(mem_eval),
    )


def run_classical_baseline(
    dataset: PreparedDataset,
    max_depth: int,
    random_state: int | None = None,
) -> tuple[float, float, float]:
    classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    _, train_s = timed(lambda: classifier.fit(dataset.X_train, dataset.y_train))
    predictions, predict_s = timed(lambda: classifier.predict(dataset.X_test))
    accuracy = float(accuracy_score(dataset.y_test, predictions))
    return accuracy, train_s, predict_s


def export_csv(records: list, csv_path: str) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with csv_path.open("a" if file_exists else "w", newline="") as handle:
        if not records:
            return
        writer = csv.DictWriter(
            handle, fieldnames=records[0].__class__.__dataclass_fields__.keys()
        )
        if not file_exists:
            writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def gini_impurity(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return float(1.0 - np.sum(probs**2))


def majority_class(labels: np.ndarray) -> int:
    values, counts = np.unique(labels, return_counts=True)
    return int(values[np.argmax(counts)]) if len(values) else 0


def split_score(left_mask: np.ndarray, labels: np.ndarray) -> float:
    left = labels[left_mask]
    right = labels[~left_mask]

    if len(left) == 0 or len(right) == 0:
        return 1.0

    total = len(labels)
    return (len(left) * gini_impurity(left) + len(right) * gini_impurity(right)) / total


@dataclass
class QNode:
    feature_idx: int | None = None
    theta: np.ndarray | None = None
    left: "QNode | None" = None
    right: "QNode | None" = None
    value: int | None = None
