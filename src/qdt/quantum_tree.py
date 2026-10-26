from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .selection import select_candidates
from .simulation import CircuitRunner
from .types import CircuitSpec


def gini_impurity(labels: np.ndarray) -> float:
    if not len(labels):
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    return float(1.0 - np.sum((counts / counts.sum()) ** 2))


def majority_class(labels: np.ndarray) -> int:
    values, counts = np.unique(labels, return_counts=True)
    return int(values[np.argmax(counts)]) if len(values) else 0


def split_score(left_mask: np.ndarray, labels: np.ndarray) -> float:
    left, right = labels[left_mask], labels[~left_mask]
    if not len(left) or not len(right):
        return 1.0
    return (len(left) * gini_impurity(left) + len(right) * gini_impurity(right)) / len(
        labels
    )


def soft_split_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Weighted Gini using P(0) as fractional left membership."""
    score = 0.0
    for weights in (probabilities, 1 - probabilities):
        mass = weights.sum()
        if mass > 1e-12:
            proportions = (
                np.array([weights[labels == c].sum() for c in np.unique(labels)]) / mass
            )
            score += mass / len(labels) * (1 - np.sum(proportions**2))
    return float(score)


@dataclass(frozen=True)
class TreeConfig:
    feature_selection: str = "f_classif"
    top_k: int | None = 4
    corr_threshold: float | None = 0.95
    optimizer: str = "cobyla"
    objective: str = "soft"
    max_evals: int = 40  # total per candidate, shared across restarts
    restarts: int = 2
    min_samples_leaf: int = 2
    min_impurity_decrease: float = 1e-8
    seed: int = 42

    def __post_init__(self):
        if self.optimizer not in {"cobyla", "spsa", "random"}:
            raise ValueError("optimizer must be cobyla, spsa or random")
        if self.objective not in {"hard", "soft"}:
            raise ValueError("objective must be hard or soft")
        if self.restarts < 1 or self.max_evals < 4 * self.restarts:
            raise ValueError("max_evals must be >= 4 * restarts")
        if self.min_samples_leaf < 1 or self.min_impurity_decrease < 0:
            raise ValueError("Invalid split stopping parameters")
        if self.feature_selection not in {"all", "f_classif", "mutual_info"}:
            raise ValueError("Unknown feature selector")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be positive or None")
        if self.corr_threshold is not None and not 0 < self.corr_threshold <= 1:
            raise ValueError("corr_threshold must be in (0, 1]")


@dataclass
class QNode:
    feature_idx: int | None = None
    theta: np.ndarray | None = None
    left: QNode | None = None
    right: QNode | None = None
    value: int | None = None


NUM_PARAMS = 2


def build_circuit(feature_value: float, theta: np.ndarray) -> CircuitSpec:
    return float(feature_value) * np.pi, float(theta[0]), float(theta[1])


def run_split(X, feature_idx, theta, batch_size, runner=None):
    runner = runner or CircuitRunner()
    return runner.split(
        [build_circuit(v, theta) for v in X[:, feature_idx]], batch_size
    )


class _BudgetReached(Exception):
    pass


def best_quantum_split(
    X, y, batch_size, *, runner=None, config=None, rng=None, diagnostics=None
):
    cfg = config or TreeConfig()
    runner = runner or CircuitRunner()
    rng = rng if rng is not None else np.random.default_rng(cfg.seed)
    candidates = select_candidates(
        X,
        y,
        method=cfg.feature_selection,
        top_k=cfg.top_k,
        corr_threshold=cfg.corr_threshold,
        seed=cfg.seed,
    )
    parent_score = gini_impurity(y)
    best_score = parent_score
    best_split = None
    for feature_idx in candidates:
        nfev = 0
        feature_best = 1.0
        for restart in range(cfg.restarts):
            allowance = cfg.max_evals // cfg.restarts + (
                restart < cfg.max_evals % cfg.restarts
            )
            stop_at = nfev + allowance

            def objective(theta, stop_at=stop_at, feature_idx=feature_idx):
                nonlocal nfev, best_score, best_split, feature_best
                if nfev >= stop_at:
                    raise _BudgetReached
                theta = np.asarray(theta) % (2 * np.pi)
                p = runner.probabilities(
                    [build_circuit(v, theta) for v in X[:, feature_idx]],
                    batch_size,
                )
                nfev += 1
                mask = p >= 0.5
                hard_score = split_score(mask, y)
                feature_best = min(feature_best, hard_score)
                if (
                    min(mask.sum(), (~mask).sum()) >= cfg.min_samples_leaf
                    and hard_score < best_score - 1e-12
                    and parent_score - hard_score >= cfg.min_impurity_decrease
                ):
                    best_score = hard_score
                    # Save the score and partition from THIS evaluation. A second
                    # shot-based execution would describe a different partition.
                    best_split = (
                        int(feature_idx),
                        theta.copy(),
                        np.flatnonzero(mask),
                        np.flatnonzero(~mask),
                    )
                return soft_split_score(p, y) if cfg.objective == "soft" else hard_score

            theta = (
                np.zeros(NUM_PARAMS)
                if restart == 0
                else rng.uniform(0, 2 * np.pi, NUM_PARAMS)
            )
            if cfg.optimizer == "cobyla":
                try:
                    minimize(
                        objective,
                        theta,
                        method="COBYLA",
                        tol=1e-4,
                        options={"maxiter": allowance, "rhobeg": 1.0},
                    )
                except _BudgetReached:
                    pass
            elif cfg.optimizer == "random":
                objective(theta)
                while nfev < stop_at:
                    objective(rng.uniform(0, 2 * np.pi, NUM_PARAMS))
            else:
                objective(theta)
                step = 0
                while nfev + 3 <= stop_at:
                    delta = rng.choice([-1.0, 1.0], size=NUM_PARAMS)
                    c = 0.2 / (step + 1) ** 0.101
                    a = 0.5 / (step + 1) ** 0.602
                    gradient = (
                        (objective(theta + c * delta) - objective(theta - c * delta))
                        / (2 * c)
                        * delta
                    )
                    theta = (theta - a * gradient) % (2 * np.pi)
                    objective(theta)
                    step += 1
                while nfev < stop_at:
                    objective(theta)
        if diagnostics is not None:
            diagnostics.append(
                {
                    "samples": len(y),
                    "feature": int(feature_idx),
                    "evaluations": nfev,
                    "best_hard_gini": feature_best,
                }
            )
    return best_split


def fit(
    X,
    y,
    depth=0,
    max_depth=10,
    batch_size=1,
    *,
    runner=None,
    config=None,
    diagnostics=None,
):
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
    if max_depth < 0 or batch_size < 1:
        raise ValueError("max_depth must be >= 0 and batch_size >= 1")
    cfg = config or TreeConfig()
    runner = runner or CircuitRunner()
    rng = np.random.default_rng(cfg.seed)

    def grow(features, labels, level):
        if (
            level >= max_depth
            or np.unique(labels).size == 1
            or len(labels) < max(5, 2 * cfg.min_samples_leaf)
        ):
            return QNode(value=majority_class(labels))
        split = best_quantum_split(
            features,
            labels,
            batch_size,
            runner=runner,
            config=cfg,
            rng=rng,
            diagnostics=diagnostics,
        )
        if split is None:
            return QNode(value=majority_class(labels))
        idx, theta, left, right = split
        return QNode(
            feature_idx=idx,
            theta=theta,
            left=grow(features[left], labels[left], level + 1),
            right=grow(features[right], labels[right], level + 1),
        )

    return grow(X, y, depth)


def predict(X, tree, batch_size=1, *, runner=None):
    runner = runner or CircuitRunner()
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not np.isfinite(X).all() or batch_size < 1:
        raise ValueError("Expected finite X (n, d) and batch_size >= 1")
    predictions = np.empty(len(X), dtype=int)
    queue = [(tree, np.arange(len(X)))]
    while queue:
        node, indices = queue.pop()
        if node.value is not None:
            predictions[indices] = node.value
            continue
        left_mask = run_split(
            X[indices], int(node.feature_idx), node.theta, batch_size, runner
        )
        if left_mask.any():
            queue.append((node.left, indices[left_mask]))
        if (~left_mask).any():
            queue.append((node.right, indices[~left_mask]))
    return predictions
