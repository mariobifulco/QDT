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
    loss, derivative = 0.0, np.zeros(len(labels))
    for weights, sign in ((probabilities, 1), (1 - probabilities, -1)):
        mass = weights.sum()
        if mass <= 1e-12:
            continue
        proportions = np.bincount(classes, weights=weights) / mass
        squared = np.sum(proportions**2)
        loss += mass / len(labels) * (1 - squared)
        derivative += sign * (squared - 2 * proportions[classes]) / len(labels)
    return float(loss), derivative


def loss_and_gradient(probabilities, theta, labels, *, nominal=None):
    loss, derivative = soft_gini(
        probabilities(theta) if nominal is None else nominal, labels
    )
    gradient = np.empty(len(theta))
    for parameter in range(len(theta)):
        shift = np.zeros(len(theta))
        shift[parameter] = np.pi / 2
        gradient[parameter] = (
            derivative
            @ (probabilities(theta + shift) - probabilities(theta - shift))
            / 2
        )
    return loss, gradient


class _BudgetReached(Exception):
    pass


def optimize_split(probabilities, labels, config, rng, *, jacobian=None):
    best_score, best = gini(labels), None
    evaluations = 0
    for restart in range(config.restarts):
        budget = config.max_evals // config.restarts + (
            restart < config.max_evals % config.restarts
        )
        stop_at = evaluations + budget

        def evaluate(theta, stop_at=stop_at):
            nonlocal evaluations
            if evaluations >= stop_at:
                raise _BudgetReached
            evaluations += 1
            return probabilities(np.asarray(theta) % (2 * np.pi))

        def objective(theta, stop_at=stop_at):
            nonlocal best_score, best, evaluations
            required = (
                (2 if jacobian is not None else 5) if config.optimizer == "lbfgs" else 1
            )
            if evaluations + required > stop_at:
                raise _BudgetReached
            p = evaluate(theta)
            mask = p >= 0.5
            score = split_score(mask, labels)
            if (
                min(mask.sum(), (~mask).sum()) >= config.min_samples_leaf
                and score < best_score - 1e-12
            ):
                best_score, best = score, (np.asarray(theta) % (2 * np.pi), mask.copy())
            if config.optimizer == "cobyla":
                return soft_gini(p, labels)[0]
            if jacobian is not None:
                # One forward evaluation plus one native Jacobian request.
                evaluations += 1
                loss, derivative = soft_gini(p, labels)
                return loss, derivative @ jacobian(np.asarray(theta) % (2 * np.pi))
            return loss_and_gradient(evaluate, theta, labels, nominal=p)

        try:
            minimize(
                objective,
                rng.uniform(0, 2 * np.pi, 2),
                method="L-BFGS-B" if config.optimizer == "lbfgs" else "COBYLA",
                jac=True if config.optimizer == "lbfgs" else None,
                options={"maxiter": budget},
            )
        except _BudgetReached:
            pass
    return best
