def main(argv=None):
    import argparse
    from dataclasses import asdict
    from pathlib import Path

    from .data import DATASET_KEYS
    from .experiments import ExperimentConfig, run_experiment
    from .quantum_tree import TreeConfig
    from .simulation import SimulationConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_KEYS, default="iris")
    parser.add_argument("--seed", "--iteration", type=int, default=42)
    parser.add_argument("--optimizer-seed", type=int)
    parser.add_argument("--simulator-seed", type=int)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=8)
    parser.add_argument(
        "--backend", choices=["analytic", "aer", "aer_stitched"], default="analytic"
    )
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    for name in ("depolarizing", "amplitude-damping", "phase-damping", "readout-error"):
        parser.add_argument(f"--{name}", type=float, default=0)
    parser.add_argument(
        "--feature-selection",
        choices=["all", "f_classif", "mutual_info"],
        default="f_classif",
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    parser.add_argument(
        "--optimizer", choices=["cobyla", "spsa", "random"], default="cobyla"
    )
    parser.add_argument("--objective", choices=["soft", "hard"], default="soft")
    parser.add_argument("--max-evals", type=int, default=40)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/single"))
    args = parser.parse_args(argv)
    sim = SimulationConfig(
        **{k: getattr(args, k) for k in asdict(SimulationConfig()) if k != "seed"},
        seed=args.simulator_seed if args.simulator_seed is not None else args.seed,
    )
    tree = TreeConfig(
        **{
            k: getattr(args, k)
            for k in (
                "feature_selection",
                "top_k",
                "corr_threshold",
                "optimizer",
                "objective",
                "max_evals",
                "restarts",
                "min_samples_leaf",
            )
        },
        seed=args.optimizer_seed if args.optimizer_seed is not None else args.seed,
    )
    config = ExperimentConfig(
        dataset=args.dataset,
        seed=args.seed,
        depth=args.depth,
        batch_size=args.batch_size,
        test_size=args.test_size,
        max_train_samples=args.max_train_samples,
        data_dir=str(args.data_dir),
        simulation=sim,
        tree=tree,
    )
    run_experiment(config, args.output_dir)
