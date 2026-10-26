from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_classification,
)
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
    feature_indices: tuple[int, ...] = ()
    data_sha256: str = ""


def _load_dataset(loader: Callable[[], object]) -> tuple[np.ndarray, np.ndarray]:
    data = loader()
    return np.asarray(data.data, dtype=float), np.asarray(data.target, dtype=int)


DATASET_LOADERS: dict[str, tuple[str, Callable[[], tuple[np.ndarray, np.ndarray]]]] = {
    "iris": ("Iris", lambda: _load_dataset(load_iris)),
    "wine": ("Wine", lambda: _load_dataset(load_wine)),
    "breast_cancer": ("Breast Cancer", lambda: _load_dataset(load_breast_cancer)),
    "digits": ("Digits", lambda: _load_dataset(load_digits)),
    "synthetic": (
        "Synthetic (100 features, 5 informative)",
        lambda: make_classification(
            n_samples=600,
            n_features=100,
            n_informative=5,
            n_redundant=5,
            n_clusters_per_class=1,
            random_state=1729,
        ),
    ),
}


def _xor_data():
    rng = np.random.default_rng(1729)
    X = rng.uniform(-1, 1, (600, 20))
    return X, ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)


DATASET_LOADERS["synthetic_xor"] = ("Synthetic XOR", _xor_data)
UCI_DATASETS = {
    "ionosphere": ("Ionosphere", 52, "ionosphere.data", 34, {"b": 0, "g": 1}),
    "sonar": ("Sonar", 151, "sonar.all-data", 60, {"R": 0, "M": 1}),
}
DATASET_KEYS = tuple(DATASET_LOADERS) + tuple(UCI_DATASETS)


def load_uci(key: str, data_dir: Path, *, download=False):
    name, dataset_id, member, dim, classes = UCI_DATASETS[key]
    path = Path(data_dir) / member
    if not path.exists():
        if not download:
            raise FileNotFoundError(
                f"Missing {path}; run qdt-campaign fetch --data-dir {data_dir}"
            )
        # Read just the expected member; never extract arbitrary archive paths.
        slug = "connectionist+bench+sonar+mines+vs+rocks" if key == "sonar" else key
        url = f"https://archive.ics.uci.edu/static/public/{dataset_id}/{slug}.zip"
        with urlopen(url, timeout=60) as response:
            archive = response.read()
        with ZipFile(BytesIO(archive)) as zipped:
            matches = [n for n in zipped.namelist() if Path(n).name == member]
            if len(matches) != 1:
                raise ValueError(f"Expected one {member} in UCI archive")
            raw = zipped.read(matches[0])
        path.parent.mkdir(parents=True, exist_ok=True)
        # fetch is run once on the login node before scheduling any jobs.
        path.write_bytes(raw)
    rows = np.loadtxt(path, delimiter=",", dtype=str)
    if rows.ndim != 2 or rows.shape[1] != dim + 1:
        raise ValueError(f"Invalid UCI dataset: {path}")
    X = rows[:, :-1].astype(float)
    y = np.array([classes[label] for label in rows[:, -1]], dtype=int)
    return name, X, y


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    variances = X_train.var(axis=0)
    keep = variances > var_thresh
    if not keep.any():
        keep[np.argmax(variances)] = True

    indices = np.flatnonzero(keep)
    X_train = X_train[:, keep]
    X_test = X_test[:, keep]

    if corr_thresh is None or X_train.shape[1] < 2:
        return X_train, X_test, indices

    corr = np.nan_to_num(np.corrcoef(X_train, rowvar=False))
    upper = np.abs(np.triu(corr, k=1))
    keep = ~np.any(upper > corr_thresh, axis=0)
    if not keep.any():
        keep[0] = True

    return X_train[:, keep], X_test[:, keep], indices[keep]


def prepare_dataset(
    dataset_key: str,
    seed: int,
    test_size: float = 0.2,
    var_thresh: float = 0.0,
    corr_thresh: float | None = None,
    quasi_dup_tol: float | None = None,
    outlier_alpha: float | None = None,
    data_dir: Path = Path("datasets"),
    max_train_samples: int | None = None,
) -> PreparedDataset:
    if dataset_key in UCI_DATASETS:
        name, X_raw, y_raw = load_uci(dataset_key, data_dir)
    else:
        name, loader = DATASET_LOADERS[dataset_key]
        X_raw, y_raw = loader()
    X_raw, y_raw = np.asarray(X_raw, dtype=float), np.asarray(y_raw, dtype=int)
    if not np.isfinite(X_raw).all():
        raise ValueError("Dataset contains non-finite values")
    if var_thresh < 0 or quasi_dup_tol is not None and quasi_dup_tol <= 0:
        raise ValueError("Invalid variance threshold or duplicate tolerance")
    if outlier_alpha is not None and not 0 < outlier_alpha < 1:
        raise ValueError("outlier_alpha must be in (0, 1)")
    data_sha256 = hashlib.sha256(
        X_raw.astype("<f8").tobytes() + y_raw.astype("<i8").tobytes()
    ).hexdigest()

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=test_size, stratify=y_raw, random_state=seed
    )

    if max_train_samples is not None:
        if max_train_samples < 2 * np.unique(y_train).size:
            raise ValueError(
                "max_train_samples must allow at least two samples per class"
            )
        if max_train_samples < len(y_train):
            X_train, _, y_train, _ = train_test_split(
                X_train,
                y_train,
                train_size=max_train_samples,
                stratify=y_train,
                random_state=seed,
            )

    X_train, X_test = _normalize_from_train(X_train, X_test)
    X_train, y_train = _remove_train_quasi_duplicates(X_train, y_train, quasi_dup_tol)
    X_train, y_train = _remove_train_outliers(X_train, y_train, outlier_alpha)
    X_train, X_test, feature_indices = _select_features_from_train(
        X_train, X_test, var_thresh, corr_thresh
    )

    return PreparedDataset(
        name=name,
        original_dim=X_raw.shape[1],
        selected_dim=X_train.shape[1],
        X_train=X_train,
        y_train=y_train.astype(int),
        X_test=X_test,
        y_test=y_test.astype(int),
        feature_indices=tuple(int(i) for i in feature_indices),
        data_sha256=data_sha256,
    )


def get_data(
    dataset_name: str,
    seed: int,
    test_size: float = 0.2,
    var_thresh: float = 0.0,
    corr_thresh: float | None = None,
    quasi_dup_tol: float | None = None,
    outlier_alpha: float | None = None,
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
