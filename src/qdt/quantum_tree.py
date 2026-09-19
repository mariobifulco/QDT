from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize

# from .adapters.qiskit_run import stitch_and_run
# from .adapters.cudaq_run import stitch_and_run
from .adapters.iqm_run import stitch_and_run
from .types import CircuitSpec


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


NUM_PARAMS = 2


def build_circuit(feature_value: float, theta: np.ndarray) -> CircuitSpec:
    return (
        float(feature_value) * np.pi,
        float(theta[0]),
        float(theta[1]),
    )


def run_split(
    X: np.ndarray,
    feature_idx: int,
    theta: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    circuits = [build_circuit(row[feature_idx], theta) for row in X]
    return stitch_and_run(circuits, batch_size=batch_size)


def best_quantum_split(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray] | None:
    best_score = float("inf")
    best_split = None

    for feature_idx in range(X.shape[1]):
        initial_theta = np.random.uniform(0.0, 2.0 * np.pi, NUM_PARAMS)
        result = minimize(
            lambda theta: split_score(
                run_split(X, feature_idx, theta, batch_size),
                y,
            ),
            initial_theta,
            method="COBYLA",
            options={"maxiter": 30},
            tol=1e-3,
        )

        theta = np.asarray(result.x, dtype=float)
        left_mask = run_split(X, feature_idx, theta, batch_size)
        left_indices = np.flatnonzero(left_mask)
        right_indices = np.flatnonzero(~left_mask)

        if left_indices.size == 0 or right_indices.size == 0:
            continue

        score = float(result.fun)
        if score < best_score:
            best_score = score
            best_split = (feature_idx, theta, left_indices, right_indices)

    return best_split


def fit(
    X: np.ndarray,
    y: np.ndarray,
    depth: int = 0,
    max_depth: int = 10,
    batch_size: int = 1,
) -> QNode:
    if depth >= max_depth or np.unique(y).size == 1 or len(y) < 5:
        return QNode(value=majority_class(y))

    split = best_quantum_split(X, y, batch_size)
    if split is None:
        return QNode(value=majority_class(y))

    feature_idx, theta, left_indices, right_indices = split
    return QNode(
        feature_idx=feature_idx,
        theta=theta,
        left=fit(
            X[left_indices],
            y[left_indices],
            depth=depth + 1,
            max_depth=max_depth,
            batch_size=batch_size,
        ),
        right=fit(
            X[right_indices],
            y[right_indices],
            depth=depth + 1,
            max_depth=max_depth,
            batch_size=batch_size,
        ),
    )


def predict(X: np.ndarray, tree: QNode, batch_size: int) -> np.ndarray:
    predictions = np.empty(len(X), dtype=int)
    queue: list[tuple[QNode, np.ndarray]] = [(tree, np.arange(len(X)))]

    while queue:
        node, indices = queue.pop()
        if node.value is not None:
            predictions[indices] = node.value
            continue

        left_mask = run_split(
            X[indices],
            int(node.feature_idx),
            node.theta,
            batch_size,
        )
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]

        if len(left_indices):
            queue.append((node.left, left_indices))
        if len(right_indices):
            queue.append((node.right, right_indices))

    return predictions
