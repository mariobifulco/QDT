import warnings

import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif


def select_candidates(
    X, y, *, method="f_classif", top_k=4, corr_threshold=0.95, seed=42
):
    """Rank node-training features, greedily discard redundant candidates.

    Correlations are computed against at most k retained columns: O(n*d*k),
    without constructing a d-by-d correlation matrix. Returned indices refer
    to the input columns; no test data is accepted by this API.
    """
    if method not in {"all", "f_classif", "mutual_info"}:
        raise ValueError("Unknown feature selection method")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1 or None")
    if corr_threshold is not None and not 0 < corr_threshold <= 1:
        raise ValueError("corr_threshold must be in (0, 1]")
    if method == "all":
        return np.arange(X.shape[1])
    active = np.flatnonzero(np.var(X, axis=0) > 1e-12)
    if not len(active):
        return active
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # MI's nearest-neighbor estimator cannot estimate singleton classes.
        if method == "f_classif" or np.unique(y, return_counts=True)[1].min() < 2:
            scores = f_classif(X[:, active], y)[0]
        else:
            scores = mutual_info_classif(X[:, active], y, random_state=seed)
    scores = np.nan_to_num(scores, nan=0, posinf=np.finfo(float).max)
    ranking = active[np.argsort(-scores, kind="stable")]
    centered = X - X.mean(axis=0)
    norms = np.linalg.norm(centered, axis=0)
    selected = []
    for idx in ranking:
        if selected and corr_threshold is not None:
            correlations = np.abs(centered[:, selected].T @ centered[:, idx])
            correlations /= norms[selected] * norms[idx]
            if np.any(correlations > corr_threshold):
                continue
        selected.append(int(idx))
        if top_k is not None and len(selected) >= top_k:
            break
    return np.asarray(selected, dtype=int)
