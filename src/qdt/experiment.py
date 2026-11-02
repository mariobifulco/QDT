"""Train once, then evaluate frozen models without refitting preprocessing."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .adapters import get_adapter
from .config import Config
from .data import dataset_checksum, load_dataset, prepare_dataset
from .model import Model, write_json
from .quantum_tree import fit


def provenance():
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else None

    versions = {}
    for name in (
        "qiskit",
        "qiskit-aer",
        "qiskit-aer-gpu",
        "numpy",
        "scipy",
        "scikit-learn",
        "iqm-client",
        "qiskit-ibm-runtime",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    source = hashlib.sha256()
    for path in sorted((root / "src" / "qdt").rglob("*.py")):
        source.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    lock = root / "uv.lock"
    return {
        "commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "source_sha256": source.hexdigest(),
        "allocated_resources": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "SLURM_CPUS_PER_TASK",
                "SLURM_JOB_GPUS",
                "SLURM_GPUS_ON_NODE",
                "SLURM_MEM_PER_NODE",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        "lockfile_sha256": hashlib.sha256(lock.read_bytes()).hexdigest()
        if lock.exists()
        else None,
    }


def memory_usage():
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
    except ImportError:
        peak_bytes = None
    return {
        "process_peak_rss_bytes": peak_bytes,
        "scope": "process lifetime",
        "gpu_peak_memory_bytes": None,
    }


def metrics(y, predictions):
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "predictions": predictions.tolist(),
    }


def train_model(config):
    if config.backend != "aer":
        raise ValueError(
            "Training uses local simulation; use predict for IBM inference"
        )
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
                "statistics": adapter.statistics,
                "backend_records": adapter.records,
                "memory": memory_usage(),
                "probability_calls": adapter.calls,
                "logical_circuits": adapter.circuits,
                "seeds": {
                    stage: config.seed_for(stage)
                    for stage in (
                        "data",
                        "split",
                        "optimizer",
                        "simulator",
                        "transpiler",
                    )
                },
            },
        },
    )


def evaluate(
    model,
    *,
    execution=None,
    split="validation",
    inputs=None,
    records_output=None,
    max_samples=None,
):
    start = time.perf_counter()
    if max_samples is not None and (type(max_samples) is not int or max_samples < 1):
        raise ValueError("max_samples must be a positive integer")
    config = model.execution_config(execution)
    y = None
    indices = None
    if inputs is None:
        X, labels = load_dataset(model.config)
        if dataset_checksum(X, labels) != model.data["dataset_sha256"]:
            raise ValueError("Dataset checksum differs from the frozen model")
        indices = model.data["split_indices"][split][:max_samples]
        if not indices:
            raise ValueError(f"The frozen model has no {split} samples")
        X, y = X[indices], labels[indices]
    else:
        X = inputs[:max_samples]
    setup = time.perf_counter()
    adapter = get_adapter(config)
    setup_s = time.perf_counter() - setup
    prediction = time.perf_counter()
    try:
        predictions = model.predict(X, execution=execution, adapter=adapter)
    finally:
        if records_output is not None:
            write_json(
                records_output,
                {"statistics": adapter.statistics, "backend_records": adapter.records},
            )
    predict_s = time.perf_counter() - prediction
    result = {
        "schema_version": 1,
        "model_id": model.model_id,
        "split": split if inputs is None else "external",
        "sample_indices": indices,
        "execution": execution or {},
        "resolved_config": asdict(config),
        "statistics": adapter.statistics,
        "backend_records": adapter.records,
        "memory": memory_usage(),
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
    result["total_s"] = time.perf_counter() - start
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
    train.add_argument("--aer-device", choices=("CPU", "GPU"))
    train.add_argument("--threads", type=int)
    train.add_argument(
        "--seed",
        type=int,
        help="Fallback optimizer/simulator seed; explicit split seed stays fixed",
    )
    infer = sub.add_parser("predict")
    infer.add_argument("--model", required=True, type=Path)
    infer.add_argument(
        "--max-samples",
        type=int,
        help="Limit to the first N split/input rows for control runs",
    )
    settings = infer.add_mutually_exclusive_group()
    settings.add_argument(
        "--execution", type=Path, help="JSON containing execution overrides only"
    )
    settings.add_argument(
        "--policy", type=Path, help="Validated fixed-setting policy JSON"
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
        overrides = {
            key: getattr(args, key)
            for key in ("validation_size", "seed", "aer_device", "threads")
            if getattr(args, key) is not None
        }
        config = replace(config, **overrides)
        model = train_model(config)
        model.save(args.output)
        print(f"Saved model {model.model_id} to {args.output}")
    else:
        model = Model.load(args.model)
        execution = json.loads(args.execution.read_text()) if args.execution else None
        policy_id = None
        if args.policy:
            from .selection import policy_execution

            execution, policy_id = policy_execution(args.policy, model)
        result = evaluate(
            model,
            execution=execution,
            inputs=json.loads(args.inputs.read_text()) if args.inputs else None,
            split=args.split,
            max_samples=args.max_samples,
        )
        result["policy_id"] = policy_id
        write_json(args.output, result)
        print(f"Saved {len(result['predictions'])} predictions to {args.output}")
