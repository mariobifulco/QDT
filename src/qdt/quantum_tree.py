from dataclasses import dataclass

import numpy as np

from .adapters import get_adapter
from .config import Config
from .optimization import optimize_split, split_score


@dataclass
class QNode:
    feature_indices: tuple[int, ...] | None = None
    theta: tuple[float, ...] | None = None
    left: "QNode | None" = None
    right: "QNode | None" = None
    value: int | None = None


def run_probabilities(values, theta, adapter, config):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    theta = np.asarray(theta, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != config.circuit.num_features
        or theta.shape != (config.circuit.num_parameters,)
    ):
        raise ValueError("Data and theta dimensions do not match the circuit")
    inputs = np.column_stack(
        (values * np.pi, np.broadcast_to(theta, (len(values), len(theta))))
    )
    return adapter.probabilities(inputs, config.batch_size)


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
    width = config.circuit.num_features
    if X.shape[1] < width:
        raise ValueError(
            "Selected features are fewer than circuit inputs; increase top_k or corr_threshold"
        )
    groups = [tuple(range(i, i + width)) for i in range(X.shape[1] - width + 1)]
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
        active = features.var(axis=0) > 1e-12
        for group in groups:
            if not active[list(group)].any():
                continue

            def probabilities(theta, group=group):
                return run_probabilities(features[:, group], theta, adapter, config)

            split = optimize_split(probabilities, labels, config, rng)
            if split is not None:
                theta, mask = split
                score = split_score(mask, labels)
                if score < best_score:
                    best_score, best = score, (group, theta, mask)
        if best is None:
            return leaf
        group, theta, mask = best
        return QNode(
            feature_indices=group,
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
                    X[indices][:, node.feature_indices], node.theta, adapter, config
                )
                >= 0.5
            )
            queue.extend(((node.left, indices[mask]), (node.right, indices[~mask])))
    return predictions
