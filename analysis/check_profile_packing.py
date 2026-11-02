"""Compare packed marginals with isolated execution on the SAME physical partitions."""

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np

from qdt.adapters import get_adapter
from qdt.data import dataset_checksum, load_dataset
from qdt.experiment import provenance
from qdt.model import Model, write_json
from qdt.quantum_tree import run_probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--packing", type=int, required=True)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.atol) or args.atol < 0:
        parser.error("atol must be finite and nonnegative")
    if args.packing < 2 or args.max_samples < 1:
        parser.error("Use packing >= 2 and max-samples >= 1")
    if args.output.exists():
        parser.error("Output already exists")
    model = Model.load(args.model)
    if model.tree.value is not None:
        parser.error("This root-node diagnostic requires a nonleaf model")
    X, y = load_dataset(model.config)
    if dataset_checksum(X, y) != model.data["dataset_sha256"]:
        raise ValueError("Dataset changed")
    indices = model.data["split_indices"]["validation"][: args.max_samples]
    if not indices:
        raise ValueError("Model has no validation data")
    features = model.preprocessing.transform(X[indices])[
        :, list(model.tree.feature_indices)
    ]
    config = replace(
        model.config,
        backend="aer",
        aer_device="CPU",
        aer_method="density_matrix",
        fake_qpu=args.profile,
        shots=0,
        packing_factor=args.packing,
        qpu_layout=None,
        depolarizing=0,
        amplitude_damping=0,
        phase_damping=0,
        readout_error=0,
    )
    adapter = get_adapter(config)
    packed = run_probabilities(features, model.tree.theta, adapter, config)
    reference = np.empty(len(features))
    isolated_records = []
    for copy in range(args.packing):
        selected = np.arange(copy, len(features), args.packing)
        if not len(selected):
            continue
        width = config.circuit.num_qubits
        local = replace(
            config,
            packing_factor=1,
            qpu_layout=adapter.selected_layout[copy * width : (copy + 1) * width],
        )
        isolated = get_adapter(local)
        reference[selected] = run_probabilities(
            features[selected], model.tree.theta, isolated, local
        )
        isolated_records.append(isolated.records)
    error = float(np.max(np.abs(reference - packed)))
    disagreements = int(np.count_nonzero((reference >= 0.5) != (packed >= 0.5)))
    write_json(
        args.output,
        {
            "model_id": model.model_id,
            "sample_indices": indices,
            "profile": args.profile,
            "layout": adapter.selected_layout,
            "packing_factor": args.packing,
            "packed": packed.tolist(),
            "isolated_on_same_partitions": reference.tolist(),
            "max_absolute_error": error,
            "tolerance": args.atol,
            "decision_disagreements": disagreements,
            "minimum_reference_margin": float(np.min(np.abs(reference - 0.5))),
            "aer_precision": adapter.backend.options.precision,
            "passed": error <= args.atol and disagreements == 0,
            "profile_metadata": adapter.profile_metadata,
            "packed_records": adapter.records,
            "isolated_records": isolated_records,
            "provenance": provenance(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "Local CPU independent-noise model; this does not measure physical crosstalk.",
        },
    )
    if error > args.atol or disagreements:
        raise RuntimeError(
            f"Packed marginals differ from matched isolated references: {error}"
        )
    print(f"Matched-partition check passed: max absolute error {error:.3g}")


if __name__ == "__main__":
    main()
