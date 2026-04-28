from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CircuitSpec = tuple[float, float, float]


def gini_impurity(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return float(1.0 - np.sum(probs ** 2))


def majority_class(labels: np.ndarray) -> int:
    values, counts = np.unique(labels, return_counts=True)
    return int(values[np.argmax(counts)]) if len(values) else 0


def split_score(left_mask: np.ndarray, labels: np.ndarray) -> float:
    left = labels[left_mask]
    right = labels[~left_mask]

    if len(left) == 0 or len(right) == 0:
        return 1.0

    total = len(labels)
    return (
        len(left) * gini_impurity(left)
        + len(right) * gini_impurity(right)
    ) / total


@dataclass
class QNode:
    feature_idx: int | None = None
    theta: np.ndarray | None = None
    left: "QNode | None" = None
    right: "QNode | None" = None
    value: int | None = None
