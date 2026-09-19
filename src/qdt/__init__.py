from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .data import get_data
from .quantum_tree import stitch_and_run
from .plot_stitching_results import plot_memory, plot_runtime
from .util import RunRecord, export_csv, run_classical_baseline, run_quantum_experiment


def save_summary(records: list[RunRecord], output_dir: Path) -> None:
    frame = pd.DataFrame([asdict(record) for record in records])
    first, last = records[0], records[-1]
    best_accuracy = max(records, key=lambda r: r.quantum_accuracy)
    fastest_train = min(records, key=lambda r: r.quantum_train_s)
    fastest_predict = min(records, key=lambda r: r.quantum_predict_s)
    summary = {
        "best_accuracy_batch": best_accuracy.batch_size,
        "best_quantum_accuracy": best_accuracy.quantum_accuracy,
        "classical_accuracy": first.classical_accuracy,
        "fastest_train_batch": fastest_train.batch_size,
        "fastest_train_s": fastest_train.quantum_train_s,
        "best_train_speedup_vs_b1": first.quantum_train_s
        / fastest_train.quantum_train_s,
        "fastest_predict_batch": fastest_predict.batch_size,
        "fastest_predict_s": fastest_predict.quantum_predict_s,
        "best_predict_speedup_vs_b1": first.quantum_predict_s
        / fastest_predict.quantum_predict_s,
        "train_memory_b1_mib": first.train_peak_python_mib,
        f"train_memory_b{last.batch_size}_mib": last.train_peak_python_mib,
        f"train_memory_growth_b{last.batch_size}_over_b1": last.train_peak_python_mib
        / first.train_peak_python_mib,
    }
    pd.DataFrame([summary]).to_csv(output_dir / "summary_stats.csv", index=False)
    print(f"Summary: {summary}", flush=True)

    # Reuse the existing PDF plots; one observation per batch has no error band.
    stats = frame.rename(columns={c: f"{c}_mean" for c in frame if c != "batch_size"})
    for column in frame:
        if column != "batch_size":
            stats[f"{column}_std"] = 0.0
    plot_runtime({"iris": stats}, output_dir, annotate=False)
    plot_memory({"iris": stats}, output_dir, annotate=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Iris: one iteration per batch, from 1 to 18, with terminal logging."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("results/iris"))
    args = parser.parse_args(argv)
    if args.depth < 1:
        parser.error("--depth must be >= 1")

    config = {
        "dataset": "iris",
        "iterations": 1,
        "batch_sizes": list(range(5, 6)),
        "seed": args.seed,
        "max_depth": args.depth,
        "test_size": 0.2,
        "adapter": stitch_and_run.__module__,
    }
    if config["adapter"] == "adapters.iqm_run":
        config.update(
            iqm_backend=os.getenv("IQM_BACKEND", "fake"),
            iqm_shots=int(os.getenv("IQM_SHOTS", "1024")),
        )
    output_dir = (
        args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "iris.csv"
    print(f"Config: {config}", flush=True)
    print(f"Results: {output_dir}", flush=True)
    dataset = get_data("iris", test_size=0.2, seed=args.seed)
    baseline = run_classical_baseline(
        dataset, max_depth=args.depth, random_state=args.seed
    )
    records = []
    for batch_size in config["batch_sizes"]:
        print(f"Iris — batch {batch_size}/{len(config['batch_sizes'])}", flush=True)
        np.random.seed(args.seed)
        record = run_quantum_experiment(
            dataset, *baseline, batch_size=batch_size, depth=args.depth
        )
        export_csv([record], str(csv_path))
        records.append(record)
        print(f"Metrics: {asdict(record)}", flush=True)
    save_summary(records, output_dir)
    print(f"Completed batches: {len(records)}. Results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
