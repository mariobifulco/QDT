"""Train once, then evaluate frozen models without refitting preprocessing."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .adapters import get_adapter
from .config import Config
from .data import dataset_checksum, load_dataset, prepare_dataset
from .model import Model
from .quantum_tree import fit


def provenance():
    def git(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("qiskit", "qiskit-aer", "numpy", "scipy", "scikit-learn")
        },
    }


def metrics(y, predictions):
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "predictions": predictions.tolist(),
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def train_model(config):
    start = time.perf_counter()
    data = prepare_dataset(config)
    setup = time.perf_counter()
    adapter = get_adapter(config)
    setup_s = time.perf_counter() - setup
    trace = []
    training = time.perf_counter()
    tree = fit(data.X_train, data.y_train, config, adapter=adapter, trace=trace)
    training_s = time.perf_counter() - training
    return Model(
        config,
        data.preprocessing,
        tree,
        np.unique(data.y_train).tolist(),
        {
            "dataset_sha256": data.dataset_sha256,
            "split_indices": data.split_indices,
            "training": {
                "provenance": provenance(),
                "trace": trace,
                "setup_s": setup_s,
                "training_s": training_s,
                "total_s": time.perf_counter() - start,
                "probability_calls": adapter.calls,
                "logical_circuits": adapter.circuits,
                "seeds": {
                    stage: config.seed_for(stage)
                    for stage in ("split", "optimizer", "simulator", "transpiler")
                },
            },
        },
    )


def evaluate(model, *, execution=None, split="validation", inputs=None):
    start = time.perf_counter()
    config = model.execution_config(execution)
    y = None
    indices = None
    if inputs is None:
        X, labels = load_dataset(model.config)
        if dataset_checksum(X, labels) != model.data["dataset_sha256"]:
            raise ValueError("Dataset checksum differs from the frozen model")
        indices = model.data["split_indices"][split]
        if not indices:
            raise ValueError(f"The frozen model has no {split} samples")
        X, y = X[indices], labels[indices]
    else:
        X = inputs
    setup = time.perf_counter()
    adapter = get_adapter(config)
    setup_s = time.perf_counter() - setup
    prediction = time.perf_counter()
    predictions = model.predict(X, execution=execution, adapter=adapter)
    predict_s = time.perf_counter() - prediction
    result = {
        "schema_version": 1,
        "model_id": model.model_id,
        "split": split if inputs is None else "external",
        "sample_indices": indices,
        "execution": execution or {},
        "setup_s": setup_s,
        "predict_s": predict_s,
        "total_s": time.perf_counter() - start,
        "device": adapter.device_name,
        "probability_calls": adapter.calls,
        "logical_circuits": adapter.circuits,
        "provenance": provenance(),
    }
    result.update(
        metrics(y, predictions)
        if y is not None
        else {"predictions": predictions.tolist()}
    )
    if y is not None:
        result["y_true"] = y.tolist()
    return result


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument(
        "--output", required=True, type=Path, help="Complete frozen model JSON"
    )
    train.add_argument("--validation-size", type=float)
    infer = sub.add_parser("predict")
    infer.add_argument("--model", required=True, type=Path)
    infer.add_argument(
        "--execution", type=Path, help="JSON containing execution overrides only"
    )
    infer.add_argument(
        "--inputs", type=Path, help="JSON matrix of raw features, original column order"
    )
    infer.add_argument(
        "--split", choices=("train", "validation", "test"), default="validation"
    )
    infer.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    if args.command == "train":
        config = Config.load(args.config)
        if args.validation_size is not None:
            config = replace(config, validation_size=args.validation_size)
        model = train_model(config)
        model.save(args.output)
        print(f"Saved model {model.model_id} to {args.output}")
    else:
        result = evaluate(
            Model.load(args.model),
            execution=json.loads(args.execution.read_text())
            if args.execution
            else None,
            inputs=json.loads(args.inputs.read_text()) if args.inputs else None,
            split=args.split,
        )
        write_json(args.output, result)
        print(f"Saved {len(result['predictions'])} predictions to {args.output}")
