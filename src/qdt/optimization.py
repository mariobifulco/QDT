import numpy as np
from scipy.optimize import minimize


def gini(labels):
    _, counts = np.unique(labels, return_counts=True)
    return _gini_counts(counts)


def _gini_counts(counts):
    mass = counts.sum()
    return float(1 - np.sum((counts / mass) ** 2)) if mass else 0.0


def split_score(mask, labels):
    _, classes, totals = np.unique(labels, return_inverse=True, return_counts=True)
    return _split_score(mask, classes, totals)


def _split_score(mask, classes, totals):
    if not mask.any() or mask.all():
        return 1.0
    left = np.bincount(classes[mask], minlength=len(totals))
    right = totals - left
    return float(
        (left.sum() * _gini_counts(left) + right.sum() * _gini_counts(right))
        / len(classes)
    )


def soft_gini(probabilities, labels):
    _, classes = np.unique(labels, return_inverse=True)
    return _soft_gini(probabilities, classes)


def _soft_gini(probabilities, classes):
    score = 0.0
    for weights in (probabilities, 1 - probabilities):
        mass = weights.sum()
        if mass > 1e-12:
            proportions = np.bincount(classes, weights=weights) / mass
            score += mass / len(classes) * (1 - np.sum(proportions**2))
    return float(score)


def optimize_split(probabilities, labels, config, rng):
    _, classes, totals = np.unique(labels, return_inverse=True, return_counts=True)
    best_score, best = _gini_counts(totals), None
    for restart in range(config.restarts):
        budget = config.max_evals // config.restarts + (
            restart < config.max_evals % config.restarts
        )

        def objective(theta):
            nonlocal best_score, best
            theta = np.asarray(theta) % (2 * np.pi)
            p = probabilities(theta)
            mask = p >= 0.5
            score = _split_score(mask, classes, totals)
            if (
                min(mask.sum(), (~mask).sum()) >= config.min_samples_leaf
                and score < best_score - 1e-12
            ):
                best_score, best = score, (theta.copy(), mask.copy())
            return _soft_gini(p, classes)

        minimize(
            objective,
            rng.uniform(0, 2 * np.pi, config.circuit.num_parameters),
            method="COBYLA",
            options={"maxiter": budget},
        )
    return best
