import numpy as np
from scipy.optimize import minimize


def gini(labels):
    _, counts = np.unique(labels, return_counts=True)
    return float(1 - np.sum((counts / counts.sum()) ** 2)) if len(labels) else 0.0


def split_score(mask, labels):
    if not mask.any() or mask.all():
        return 1.0
    return float(
        (mask.sum() * gini(labels[mask]) + (~mask).sum() * gini(labels[~mask]))
        / len(labels)
    )


def soft_gini(probabilities, labels):
    _, classes = np.unique(labels, return_inverse=True)
    score = 0.0
    for weights in (probabilities, 1 - probabilities):
        mass = weights.sum()
        if mass > 1e-12:
            proportions = np.bincount(classes, weights=weights) / mass
            score += mass / len(labels) * (1 - np.sum(proportions**2))
    return float(score)


def optimize_split(probabilities, labels, config, rng):
    best_score, best = gini(labels), None
    for restart in range(config.restarts):
        budget = config.max_evals // config.restarts + (
            restart < config.max_evals % config.restarts
        )

        def objective(theta):
            nonlocal best_score, best
            theta = np.asarray(theta) % (2 * np.pi)
            p = probabilities(theta)
            mask = p >= 0.5
            score = split_score(mask, labels)
            if (
                min(mask.sum(), (~mask).sum()) >= config.min_samples_leaf
                and score < best_score - 1e-12
            ):
                best_score, best = score, (theta.copy(), mask.copy())
            return soft_gini(p, labels)

        minimize(
            objective,
            rng.uniform(0, 2 * np.pi, 2),
            method="COBYLA",
            options={"maxiter": budget},
        )
    return best
