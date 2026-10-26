import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from .data import UCI_DATASETS, load_uci
from .experiments import ExperimentConfig, run_experiment, write_json


def read_manifest(path):
    value = json.loads(Path(path).read_text())
    if value.get("schema_version") != 1 or not value.get("runs"):
        raise ValueError("Expected nonempty version-1 campaign manifest")
    configs = [ExperimentConfig.from_dict(raw) for raw in value["runs"]]
    if len({c.run_id for c in configs}) != len(configs):
        raise ValueError("Manifest contains duplicate configurations")
    return configs


def plan(grid_path, manifest_path):
    grid = json.loads(Path(grid_path).read_text())
    if set(grid) - {"base", "grid"}:
        raise ValueError("Grid accepts only base and grid keys")
    axes = grid.get("grid", {})
    if any(not isinstance(v, list) or not v for v in axes.values()):
        raise ValueError("Grid axes must be nonempty lists")
    total = int(np.prod([len(v) for v in axes.values()], dtype=object))
    if total > 100000:
        raise ValueError("Grid exceeds 100000 runs; split it into smaller campaigns")
    runs = []
    for values in itertools.product(*axes.values()):
        raw = json.loads(json.dumps(grid.get("base", {})))
        for dotted_key, value in zip(axes, values, strict=True):
            parts = dotted_key.split(".")
            if len(parts) == 1:
                raw[parts[0]] = value
            elif len(parts) == 2 and parts[0] in {"tree", "simulation"}:
                raw.setdefault(parts[0], {})[parts[1]] = value
            else:
                raise ValueError(f"Unknown grid key: {dotted_key}")
        for section in ("tree", "simulation"):
            raw.setdefault(section, {}).setdefault("seed", raw.get("seed", 42))
        runs.append(asdict(ExperimentConfig.from_dict(raw)))
    ids = [ExperimentConfig.from_dict(raw).run_id for raw in runs]
    if len(set(ids)) != len(ids):
        raise ValueError("Grid contains duplicate runs")
    path = Path(manifest_path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {"schema_version": 1, "runs": runs})
    print(f"Prepared {len(runs)} runs; Slurm array: 0-{len(runs) - 1}")
    return runs


def collect(manifest_path, output_dir, allow_incomplete=False):
    configurations = read_manifest(manifest_path)
    rows = []
    missing = []
    for cfg in configurations:
        path = Path(output_dir) / cfg.run_id / "result.json"
        if not path.exists():
            missing.append(cfg.run_id)
            continue
        result = json.loads(path.read_text())
        if result["run_id"] != cfg.run_id or result["config"] != asdict(cfg):
            raise ValueError(f"Result configuration mismatch: {path}")
        group = asdict(cfg)
        group.pop("seed")
        group["tree"].pop("seed")
        group["simulation"].pop("seed")
        # Keep distinct code, dependencies and dataset versions in separate groups.
        group["source_sha256"] = result["provenance"]["source_sha256"]
        group["versions"] = result["provenance"]["versions"]
        group["data_sha256"] = result["data_sha256"]
        group_key = json.dumps(group, sort_keys=True)
        for model, values in {"qdt": result["qdt"], **result["baselines"]}.items():
            row = {
                "run_id": cfg.run_id,
                "dataset": cfg.dataset,
                "seed": cfg.seed,
                "model_seed": cfg.tree.seed,
                "simulator_seed": cfg.simulation.seed,
                "model": model,
                "group": group_key,
                "process_peak_rss_mib": result["process_peak_rss_mib"],
            }
            row.update({k: v for k, v in values.items() if isinstance(v, (int, float))})
            if model == "qdt":
                row.update({f"train_{k}": v for k, v in values["train_cost"].items()})
            rows.append(row)
    if missing and not allow_incomplete:
        raise ValueError(
            f"Missing {len(missing)}/{len(configurations)} runs; first: {missing[:5]}"
        )
    if not rows:
        raise ValueError("No completed runs")
    frame = pd.DataFrame(rows)
    summary = []
    for (group, model), observations in frame.groupby(["group", "model"], sort=True):
        # Multiple optimizer/shot seeds on the same split are not independent datasets.
        by_split = observations.groupby("seed")
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "train_s",
            "predict_s",
        ):
            means = by_split[metric].mean()
            n = len(means)
            mean = float(means.mean())
            std = float(means.std(ddof=1)) if n > 1 else None
            half = float(t.ppf(0.975, n - 1) * std / np.sqrt(n)) if n > 1 else None
            summary.append(
                {
                    "group": group,
                    "model": model,
                    "metric": metric,
                    "runs": len(observations),
                    "split_seeds": n,
                    "mean": mean,
                    "std_between_splits": std,
                    "ci95_low": mean - half if half is not None else None,
                    "ci95_high": mean + half if half is not None else None,
                }
            )
    output = Path(output_dir)
    for name, table in (("runs.csv", frame), ("summary.csv", pd.DataFrame(summary))):
        temporary = output / (name + ".tmp")
        table.to_csv(temporary, index=False)
        temporary.replace(output / name)
    write_json(
        output / "collection.json",
        {
            "expected": len(configurations),
            "completed": len(configurations) - len(missing),
            "missing": missing,
        },
    )
    print(
        f"Collected {len(configurations) - len(missing)}/{len(configurations)} runs into {output}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="Download UCI data before compute jobs")
    fetch.add_argument("--data-dir", type=Path, default=Path("datasets"))
    fetch.add_argument(
        "--datasets", nargs="+", choices=list(UCI_DATASETS), default=list(UCI_DATASETS)
    )
    prepare = sub.add_parser("plan")
    prepare.add_argument("--grid", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path, required=True)
    count = sub.add_parser("count")
    count.add_argument("--manifest", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    aggregate = sub.add_parser("collect")
    aggregate.add_argument("--manifest", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        for key in args.datasets:
            name, X, _ = load_uci(key, args.data_dir, download=True)
            print(f"Cached {name}: {X.shape}")
    elif args.command == "plan":
        plan(args.grid, args.manifest)
    elif args.command == "count":
        print(len(read_manifest(args.manifest)))
    elif args.command == "run":
        configs = read_manifest(args.manifest)
        if not 0 <= args.index < len(configs):
            parser.error("index outside campaign manifest")
        run_experiment(configs[args.index], args.output_dir)
    else:
        collect(args.manifest, args.output_dir, args.allow_incomplete)


if __name__ == "__main__":
    main()
