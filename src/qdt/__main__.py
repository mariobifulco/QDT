"""Run one configuration: python -m qdt --config configs/aer_ideale.json."""

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

from .adapters import get_adapter
from .config import Config
from .data import DATASET_KEYS, UCI, load_uci, prepare_dataset
from .quantum_tree import fit, predict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", choices=DATASET_KEYS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--fake-qpu",
        help="Local QPU model: iqm:name, aqt:offline or file:path (downloaded IBM profile)",
    )
    parser.add_argument(
        "--list-qpus", action="store_true", help="List bundled local QPU models"
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Cache UCI datasets before compute jobs",
    )
    args = parser.parse_args(argv)
    if args.list_qpus:
        from .qpu import list_qpus

        print(json.dumps(list_qpus(), indent=2))
        return
    config = Config.load(args.config) if args.config else None
    if args.download_data:
        for key in UCI:
            load_uci(key, config.data_dir if config else "datasets", download=True)
        return
    if config is None:
        parser.error("--config is required and must include a circuit definition")
    overrides = {
        key: getattr(args, key)
        for key in ("dataset", "seed", "fake_qpu")
        if getattr(args, key) is not None
    }
    config = replace(config, **overrides)
    if args.output and args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    data = prepare_dataset(config)
    adapter = get_adapter(config)
    start = time.perf_counter()
    tree = fit(data.X_train, data.y_train, config, adapter=adapter)
    train_s = time.perf_counter() - start
    prediction_adapter = get_adapter(replace(config, seed=config.seed + 1))
    start = time.perf_counter()
    quantum_predictions = predict(data.X_test, tree, config, adapter=prediction_adapter)
    predict_s = time.perf_counter() - start
    result = {
        "config": asdict(config),
        "tree": asdict(tree),
        "feature_indices": data.feature_indices.tolist(),
        "original_dim": data.original_dim,
        "selected_dim": data.X_train.shape[1],
        "y_test": data.y_test.tolist(),
        "train_samples": len(data.y_train),
        "test_samples": len(data.y_test),
        "train_probability_calls": adapter.calls,
        "train_sample_circuits": adapter.circuits,
    }
    result["qdt"] = {
        "train_s": train_s,
        "predict_s": predict_s,
        "device": adapter.device_name,
        "optimizer": "cobyla",
    }
    if hasattr(adapter, "profile_metadata"):
        result["qdt"]["noise_profile"] = adapter.profile_metadata
    baseline = DecisionTreeClassifier(
        max_depth=config.max_depth or None,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.seed,
    )
    if config.max_depth == 0:
        from sklearn.dummy import DummyClassifier

        baseline = DummyClassifier(strategy="most_frequent")
    start = time.perf_counter()
    baseline.fit(data.X_train, data.y_train)
    baseline_train_s = time.perf_counter() - start
    start = time.perf_counter()
    classical_predictions = baseline.predict(data.X_test)
    result["cart"] = {
        "train_s": baseline_train_s,
        "predict_s": time.perf_counter() - start,
    }
    for name, predictions in (
        ("qdt", quantum_predictions),
        ("cart", classical_predictions),
    ):
        result[name].update(
            accuracy=float(accuracy_score(data.y_test, predictions)),
            balanced_accuracy=float(balanced_accuracy_score(data.y_test, predictions)),
            macro_f1=float(
                f1_score(data.y_test, predictions, average="macro", zero_division=0)
            ),
            predictions=predictions.tolist(),
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(
        f"QDT accuracy={result['qdt']['accuracy']:.4f}, CART={result['cart']['accuracy']:.4f}, train={train_s:.3f}s"
    )


if __name__ == "__main__":
    main()
