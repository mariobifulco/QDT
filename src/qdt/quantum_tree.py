from dataclasses import dataclass

import numpy as np

from .adapters import get_adapter
from .config import Config
from .optimization import optimize_split, split_score


@dataclass
class QNode:
    feature_idx: int | None = None
    theta: tuple[float, float] | None = None
    left: "QNode | None" = None
    right: "QNode | None" = None
    value: int | None = None


def run_probabilities(values, theta, adapter, config):
    circuits = [
        (float(value) * np.pi, float(theta[0]), float(theta[1])) for value in values
    ]
    return adapter.probabilities(circuits, config.batch_size)


def fit(X, y, config=None, *, adapter=None):
    config = config or Config()
    adapter = adapter or get_adapter(config)
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    if (
        X.ndim != 2
        or not X.shape[1]
        or y.ndim != 1
        or len(X) != len(y)
        or not len(y)
        or not np.isfinite(X).all()
    ):
        raise ValueError("Expected finite nonempty X (n, d) and aligned y (n,)")
    rng = np.random.default_rng(config.seed)

    def grow(features, labels, depth):
        classes, counts = np.unique(labels, return_counts=True)
        leaf = QNode(value=int(classes[counts.argmax()]))
        if (
            depth >= config.max_depth
            or len(classes) == 1
            or len(labels) < 2 * config.min_samples_leaf
            or not np.any(features.var(axis=0) > 1e-12)
        ):
            return leaf
        best_score, best = 1.0, None
        for feature in np.flatnonzero(features.var(axis=0) > 1e-12):

            def probabilities(theta, feature=feature):
                return run_probabilities(features[:, feature], theta, adapter, config)

            split = optimize_split(probabilities, labels, config, rng)
            if split is not None:
                theta, mask = split
                score = split_score(mask, labels)
                if score < best_score:
                    best_score, best = score, (feature, theta, mask)
        if best is None:
            return leaf
        feature, theta, mask = best
        return QNode(
            feature_idx=int(feature),
            theta=tuple(float(t) for t in theta),
            left=grow(features[mask], labels[mask], depth + 1),
            right=grow(features[~mask], labels[~mask], depth + 1),
        )

    return grow(X, y, 0)


def predict(X, tree, config=None, *, adapter=None):
    config = config or Config()
    adapter = adapter or get_adapter(config)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("Expected finite X (n, d)")
    predictions = np.empty(len(X), dtype=int)
    queue = [(tree, np.arange(len(X)))]
    while queue:
        node, indices = queue.pop()
        if not len(indices):
            continue
        if node.value is not None:
            predictions[indices] = node.value
        else:
            mask = (
                run_probabilities(
                    X[indices, node.feature_idx], node.theta, adapter, config
                )
                >= 0.5
            )
            queue.extend(((node.left, indices[mask]), (node.right, indices[~mask])))
    return predictions
