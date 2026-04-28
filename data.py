from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split


EPSILON = 1e-9


@dataclass(frozen=True)
class PreparedDataset:
    name: str
    original_dim: int
    selected_dim: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def _load_dataset(loader: Callable[[], object]) -> tuple[np.ndarray, np.ndarray]:
    data = loader()
    return np.asarray(data.data, dtype=float), np.asarray(data.target, dtype=int)


DATASET_LOADERS: dict[str, tuple[str, Callable[[], tuple[np.ndarray, np.ndarray]]]] = {
    "iris": ("Iris", lambda: _load_dataset(load_iris)),
    "wine": ("Wine", lambda: _load_dataset(load_wine)),
    "breast_cancer": ("Breast Cancer", lambda: _load_dataset(load_breast_cancer)),
}


def _normalize_from_train(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scales = np.maximum(np.abs(X_train).max(axis=0), EPSILON)
    return X_train / scales, X_test / scales


def _remove_train_quasi_duplicates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tol: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if tol is None or len(X_train) == 0:
        return X_train, y_train

    rounded = np.round(X_train, decimals=max(0, int(np.ceil(-np.log10(tol)))))
    _, keep = np.unique(rounded, axis=0, return_index=True)
    keep.sort()

    if np.unique(y_train[keep]).size < 2:
        return X_train, y_train

    return X_train[keep], y_train[keep]


def _remove_train_outliers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if alpha is None or len(X_train) == 0 or X_train.shape[0] <= X_train.shape[1]:
        return X_train, y_train

    mcd = MinCovDet(random_state=0).fit(X_train)
    diff = X_train - mcd.location_
    cov_inv = np.linalg.pinv(mcd.covariance_)
    distance = np.sum(diff @ cov_inv * diff, axis=1)
    keep = distance <= chi2.ppf(alpha, df=X_train.shape[1])

    if np.unique(y_train[keep]).size < 2:
        return X_train, y_train

    return X_train[keep], y_train[keep]


def _select_features_from_train(
    X_train: np.ndarray,
    X_test: np.ndarray,
    var_thresh: float,
    corr_thresh: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    variances = X_train.var(axis=0)
    keep = variances > var_thresh
    if not keep.any():
        keep[np.argmax(variances)] = True

    X_train = X_train[:, keep]
    X_test = X_test[:, keep]

    if corr_thresh is None or X_train.shape[1] < 2:
        return X_train, X_test

    corr = np.nan_to_num(np.corrcoef(X_train, rowvar=False))
    upper = np.abs(np.triu(corr, k=1))
    keep = ~np.any(upper > corr_thresh, axis=0)
    if not keep.any():
        keep[0] = True

    return X_train[:, keep], X_test[:, keep]


def prepare_dataset(
    dataset_key: str,
    seed: int,
    test_size: float = 0.2,
    var_thresh: float = 0.001,
    corr_thresh: float | None = 0.90,
    quasi_dup_tol: float | None = 1e-8,
    outlier_alpha: float | None = 0.975,
) -> PreparedDataset:
    name, loader = DATASET_LOADERS[dataset_key]
    X_raw, y_raw = loader()

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw,
        y_raw,
        test_size=test_size,
        stratify=y_raw,
        random_state=seed
    )

    X_train, X_test = _normalize_from_train(X_train, X_test)
    X_train, y_train = _remove_train_quasi_duplicates(
        X_train, y_train, quasi_dup_tol)
    X_train, y_train = _remove_train_outliers(X_train, y_train, outlier_alpha)
    X_train, X_test = _select_features_from_train(
        X_train, X_test, var_thresh, corr_thresh)

    return PreparedDataset(
        name=name,
        original_dim=X_raw.shape[1],
        selected_dim=X_train.shape[1],
        X_train=X_train,
        y_train=y_train.astype(int),
        X_test=X_test,
        y_test=y_test.astype(int),
    )


def get_data(
    dataset_name: str,
    seed: int,
    test_size: float = 0.2,
    var_thresh: float = 0.001,
    corr_thresh: float | None = 0.90,
    quasi_dup_tol: float | None = 1e-8,
    outlier_alpha: float | None = 0.975,
) -> PreparedDataset:
    return prepare_dataset(
        dataset_name,
        seed=seed,
        test_size=test_size,
        var_thresh=var_thresh,
        corr_thresh=corr_thresh,
        quasi_dup_tol=quasi_dup_tol,
        outlier_alpha=outlier_alpha,
    )
