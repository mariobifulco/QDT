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
class PreparedDataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_indices: np.ndarray


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


def prepare_dataset(config):
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, stratify=y, random_state=config.seed
    )
    if config.max_train_samples is not None and config.max_train_samples < len(y_train):
        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=config.max_train_samples,
            stratify=y_train,
            random_state=config.seed,
        )
    keep = np.flatnonzero(X_train.var(axis=0) > 1e-12)
    if not len(keep):
        keep = np.array([0])  # Retain a column so a constant dataset can form a leaf.
    scales = np.maximum(np.abs(X_train[:, keep]).max(axis=0), 1e-9)
    return PreparedDataset(
        X_train[:, keep] / scales, X_test[:, keep] / scales, y_train, y_test, keep
    )
