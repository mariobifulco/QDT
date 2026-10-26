import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .data import DATASET_KEYS, prepare_dataset
from .quantum_tree import QNode, TreeConfig, fit, predict
from .selection import select_candidates
from .simulation import CircuitRunner, SimulationConfig


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "iris"
    seed: int = 42  # stratified split seed; optimizer/simulator seeds are separate
    depth: int = 2
    batch_size: int = 8
    test_size: float = 0.2
    max_train_samples: int | None = None
    data_dir: str = "datasets"
    tree: TreeConfig = field(default_factory=TreeConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    def __post_init__(self):
        if self.dataset not in DATASET_KEYS:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        if self.depth < 1 or self.batch_size < 1 or not 0 < self.test_size < 1:
            raise ValueError("Invalid depth, batch size or test fraction")
        if self.seed < 0 or self.tree.seed < 0 or self.simulation.seed < 0:
            raise ValueError("Seeds must be nonnegative")
        if self.max_train_samples is not None and self.max_train_samples < 2:
            raise ValueError("max_train_samples must be >= 2")
        if (
            self.simulation.backend == "aer_stitched"
            and not self.simulation.shots
            and self.batch_size > 12
        ):
            raise ValueError("Exact aer_stitched supports batch sizes <= 12")

    @classmethod
    def from_dict(cls, raw):
        raw = dict(raw)
        raw["tree"] = TreeConfig(**raw.get("tree", {}))
        raw["simulation"] = SimulationConfig(**raw.get("simulation", {}))
        return cls(**raw)

    @property
    def run_id(self):
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()[:20]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def source_digest():
    package = Path(__file__).resolve().parent
    root = package.parents[1]
    digest = hashlib.sha256()
    # Hash the installed package too, rather than assuming an editable checkout.
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    for name in ("pyproject.toml", "uv.lock"):
        path = root / name
        if path.exists():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def provenance():
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=Path(__file__).resolve().parents[2],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "source_sha256": source_digest(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": sys.version,
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "versions": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "scikit-learn", "qiskit", "qiskit-aer")
            },
        },
        "slurm": {
            k: os.environ[k]
            for k in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK")
            if k in os.environ
        },
    }


def metrics(y, prediction, labels):
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=labels).tolist(),
    }


def tree_dict(node: QNode, feature_indices):
    if node.value is not None:
        return {"value": int(node.value)}
    return {
        "feature_idx": int(node.feature_idx),
        "original_feature_idx": int(feature_indices[node.feature_idx]),
        "theta": node.theta.tolist(),
        "left": tree_dict(node.left, feature_indices),
        "right": tree_dict(node.right, feature_indices),
    }


def peak_rss_mib():
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024**2 if sys.platform == "darwin" else 1024)
    except ImportError:
        return None


def run_experiment(config: ExperimentConfig, output_dir: Path):
    directory = Path(output_dir) / config.run_id
    directory.mkdir(parents=True, exist_ok=True)
    metadata = provenance()
    dataset = prepare_dataset(
        config.dataset,
        config.seed,
        test_size=config.test_size,
        max_train_samples=config.max_train_samples,
        data_dir=Path(config.data_dir),
    )
    result_path = directory / "result.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if (
            previous["config"] != json.loads(json.dumps(asdict(config)))
            or previous["provenance"]["source_sha256"] != metadata["source_sha256"]
            or previous["provenance"]["versions"] != metadata["versions"]
            or previous["data_sha256"] != dataset.data_sha256
        ):
            raise ValueError(f"Stale result in {directory}; use a new output directory")
        if not all(
            (directory / f).exists()
            for f in ("tree.json", "predictions.npz", "diagnostics.json")
        ):
            raise ValueError(
                f"Incomplete artifacts in {directory}; inspect before rerunning"
            )
        print(f"Already completed: {config.run_id}", flush=True)
        return previous
    lock = directory / ".running"
    # Exclusive ownership prevents duplicate array tasks from clobbering a run.
    with lock.open("x") as handle:
        handle.write(f"{socket.gethostname()} pid={os.getpid()}\n")
    try:
        labels = np.unique(np.concatenate([dataset.y_train, dataset.y_test]))
        predictions = {"y_test": dataset.y_test}
        baselines = {}
        start = time.perf_counter()
        root_features = select_candidates(
            dataset.X_train,
            dataset.y_train,
            method=config.tree.feature_selection,
            top_k=config.tree.top_k,
            corr_threshold=config.tree.corr_threshold,
            seed=config.tree.seed,
        )
        filter_train_s = time.perf_counter() - start
        # Include a baseline with the same root filter, in addition to full-data CART.
        classifiers = {
            "dummy": DummyClassifier(strategy="most_frequent"),
            "cart": DecisionTreeClassifier(
                max_depth=config.depth,
                min_samples_leaf=config.tree.min_samples_leaf,
                random_state=config.seed,
            ),
            "logistic": LogisticRegression(max_iter=2000, random_state=config.seed),
            "rbf_svc": SVC(),
        }
        if len(root_features):
            classifiers["cart_filtered"] = DecisionTreeClassifier(
                max_depth=config.depth,
                min_samples_leaf=config.tree.min_samples_leaf,
                random_state=config.seed,
            )
        for name, classifier in classifiers.items():
            columns = root_features if name == "cart_filtered" else slice(None)
            start = time.perf_counter()
            classifier.fit(dataset.X_train[:, columns], dataset.y_train)
            train_s = time.perf_counter() - start
            if name == "cart_filtered":
                train_s += filter_train_s
            start = time.perf_counter()
            prediction = classifier.predict(dataset.X_test[:, columns])
            predict_s = time.perf_counter() - start
            baselines[name] = dict(
                metrics(dataset.y_test, prediction, labels),
                train_s=train_s,
                predict_s=predict_s,
            )
            predictions[name] = prediction
        runner = CircuitRunner(config.simulation)
        diagnostics = []
        start = time.perf_counter()
        tree = fit(
            dataset.X_train,
            dataset.y_train,
            max_depth=config.depth,
            batch_size=config.batch_size,
            config=config.tree,
            runner=runner,
            diagnostics=diagnostics,
        )
        train_s = time.perf_counter() - start
        train_cost = runner.stats()
        # Independent prediction stream avoids coupling shot noise to optimizer work.
        prediction_runner = CircuitRunner(config.simulation)
        prediction_runner.rng = np.random.default_rng(
            np.random.SeedSequence([config.simulation.seed, 1])
        )
        start = time.perf_counter()
        prediction = predict(
            dataset.X_test, tree, config.batch_size, runner=prediction_runner
        )
        predict_s = time.perf_counter() - start
        predictions["qdt"] = prediction
        result = {
            "schema_version": 1,
            "run_id": config.run_id,
            "config": asdict(config),
            "provenance": metadata,
            "data_sha256": dataset.data_sha256,
            "prepared_sha256": hashlib.sha256(
                b"".join(
                    a.tobytes()
                    for a in (
                        dataset.X_train,
                        dataset.y_train,
                        dataset.X_test,
                        dataset.y_test,
                    )
                )
            ).hexdigest(),
            "train_samples": len(dataset.y_train),
            "test_samples": len(dataset.y_test),
            "original_dim": dataset.original_dim,
            "selected_dim": dataset.selected_dim,
            "feature_indices": list(dataset.feature_indices),
            "class_labels": labels.tolist(),
            "qdt": dict(
                metrics(dataset.y_test, prediction, labels),
                train_s=train_s,
                predict_s=predict_s,
                train_cost=train_cost,
                predict_cost=prediction_runner.stats(),
            ),
            "baselines": baselines,
            "process_peak_rss_mib": peak_rss_mib(),
        }
        write_json(directory / "tree.json", tree_dict(tree, dataset.feature_indices))
        write_json(directory / "diagnostics.json", diagnostics)
        np.savez_compressed(directory / "predictions.npz", **predictions)
        # result.json is written last and is the completion marker.
        write_json(result_path, result)
        (directory / "failed.json").unlink(missing_ok=True)
        print(
            f"Completed {config.run_id}: QDT accuracy={result['qdt']['accuracy']:.4f}",
            flush=True,
        )
        return result
    except Exception as exc:
        write_json(
            directory / "failed.json", {"error": repr(exc), "config": asdict(config)}
        )
        raise
    finally:
        lock.unlink()
