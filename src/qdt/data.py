import hashlib
import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_classification,
)
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split

LOADERS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
}
UCI = {
    "ionosphere": (52, "ionosphere", "ionosphere.data", 34, {"b": 0, "g": 1}),
    "sonar": (
        151,
        "connectionist+bench+sonar+mines+vs+rocks",
        "sonar.all-data",
        60,
        {"R": 0, "M": 1},
    ),
}
DATASET_KEYS = (*LOADERS, *UCI, "synthetic", "synthetic_xor")


@dataclass
class Preprocessing:
    original_dim: int
    feature_indices: list[int]
    scales: list[float]

    def __post_init__(self):
        if (
            type(self.original_dim) is not int
            or self.original_dim < 1
            or not self.feature_indices
            or any(
                type(i) is not int or not 0 <= i < self.original_dim
                for i in self.feature_indices
            )
            or len(set(self.feature_indices)) != len(self.feature_indices)
            or len(self.scales) != len(self.feature_indices)
            or not np.isfinite(self.scales).all()
            or np.any(np.asarray(self.scales) <= 0)
        ):
            raise ValueError("Invalid preprocessing dimensions, features or scales")

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.original_dim or not np.isfinite(X).all():
            raise ValueError(
                "Expected finite raw features with the original column count"
            )
        return X[:, self.feature_indices] / self.scales


@dataclass
class PreparedDataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_indices: np.ndarray
    original_dim: int
    X_validation: np.ndarray
    y_validation: np.ndarray
    preprocessing: Preprocessing
    split_indices: dict
    dataset_sha256: str


def load_uci(key, data_dir, *, download=False):
    dataset_id, slug, filename, dimensions, classes = UCI[key]
    path = Path(data_dir) / filename
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Missing {path}; run qdt --download-data first")
        with urlopen(
            f"https://archive.ics.uci.edu/static/public/{dataset_id}/{slug}.zip",
            timeout=60,
        ) as response:
            archive = response.read()
        with ZipFile(BytesIO(archive)) as zipped:
            matches = [
                name for name in zipped.namelist() if Path(name).name == filename
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one {filename} in UCI archive")
            raw = zipped.read(matches[0])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    rows = np.loadtxt(path, delimiter=",", dtype=str)
    if rows.ndim != 2 or rows.shape[1] != dimensions + 1:
        raise ValueError(f"Invalid dataset: {path}")
    return rows[:, :-1].astype(float), np.array(
        [classes[label] for label in rows[:, -1]]
    )


def select_features(X, y, config):
    active = np.flatnonzero(X.var(axis=0) > 1e-12)
    if config.top_k is None or not len(active):
        return active
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scores = np.nan_to_num(f_classif(X[:, active], y)[0], nan=0)
    ranking = active[np.argsort(-scores, kind="stable")]
    centered = X - X.mean(axis=0)
    norms = np.linalg.norm(centered, axis=0)
    selected = []
    for feature in ranking:
        if selected:
            correlation = np.abs(centered[:, selected].T @ centered[:, feature]) / (
                norms[selected] * norms[feature]
            )
            if np.any(correlation > config.corr_threshold):
                continue
        selected.append(int(feature))
        if len(selected) == config.top_k:
            break
    return selected


def load_dataset(config):
    if config.dataset in LOADERS:
        bunch = LOADERS[config.dataset]()
        X, y = bunch.data, bunch.target
    elif config.dataset in UCI:
        X, y = load_uci(config.dataset, config.data_dir)
    elif config.dataset == "synthetic":
        X, y = make_classification(
            n_samples=600,
            n_features=100,
            n_informative=5,
            n_redundant=5,
            n_clusters_per_class=1,
            random_state=1729,
        )
    elif config.dataset == "synthetic_xor":
        X = np.random.default_rng(1729).uniform(-1, 1, (600, 20))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
    if not np.isfinite(X).all():
        raise ValueError("Dataset contains non-finite values")
    return X, y


def prepare_dataset(config):
    X, y = load_dataset(config)
    indices = np.arange(len(y))
    seed = config.seed if config.split_seed is None else config.split_seed
    train, test = train_test_split(
        indices, test_size=config.test_size, stratify=y, random_state=seed
    )
    validation = np.empty(0, dtype=int)
    if config.validation_size:
        train, validation = train_test_split(
            train,
            test_size=config.validation_size / (1 - config.test_size),
            stratify=y[train],
            random_state=seed,
        )
    if config.max_train_samples is not None and config.max_train_samples < len(train):
        train, _ = train_test_split(
            train,
            train_size=config.max_train_samples,
            stratify=y[train],
            random_state=seed,
        )
    raw_train = X[train]
    keep = np.flatnonzero(raw_train.var(axis=0) > 1e-12)
    if not len(keep):
        keep = np.array([0])
    scales = np.maximum(np.abs(raw_train[:, keep]).max(axis=0), 1e-9)
    selected = select_features(raw_train[:, keep] / scales, y[train], config)
    if not len(selected):
        selected = np.array([0])
    features = keep[selected]
    preprocessing = Preprocessing(
        X.shape[1], features.tolist(), scales[selected].tolist()
    )
    digest = hashlib.sha256(
        X.astype("<f8").tobytes() + y.astype("<i8").tobytes()
    ).hexdigest()
    return PreparedDataset(
        preprocessing.transform(X[train]),
        preprocessing.transform(X[test]),
        y[train],
        y[test],
        features,
        X.shape[1],
        preprocessing.transform(X[validation]),
        y[validation],
        preprocessing,
        {
            "train": train.tolist(),
            "validation": validation.tolist(),
            "test": test.tolist(),
        },
        digest,
    )
