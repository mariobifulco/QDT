"""Rebuild campaign tables and paired comparisons without executing circuits."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .campaign import completed_result, load_plan
from .experiment import metrics
from .model import write_json


def paired_summary(quality_differences, speedups):
    differences, ratios = np.asarray(quality_differences), np.asarray(speedups)
    if not len(differences):
        raise ValueError("No paired models")
    quality_ci = speed_ci = None
    if len(differences) >= 2:
        draws = np.random.default_rng(0).integers(
            len(differences), size=(2000, len(differences))
        )
        quality_ci = np.quantile(
            differences[draws].mean(axis=1), [0.025, 0.975]
        ).tolist()
        speed_ci = np.quantile(
            np.median(ratios[draws], axis=1), [0.025, 0.975]
        ).tolist()
    return {
        "paired_models": len(differences),
        "mean_quality_difference": float(differences.mean()),
        "quality_difference_ci95": quality_ci,
        "median_speedup": float(np.median(ratios)),
        "speedup_ci95": speed_ci,
    }


def analyze(plan_path, output, *, reference=None, allow_incomplete=False, plots=False):
    if plots:
        from matplotlib import pyplot as plt

    plan_path, output = Path(plan_path), Path(output)
    campaign = load_plan(plan_path)
    reference = reference or campaign["manifest"]["executions"][0]["name"]
    if reference not in [item["name"] for item in campaign["manifest"]["executions"]]:
        raise ValueError("Unknown reference execution")
    rows, missing = [], []
    groups = defaultdict(list)
    for run in campaign["runs"]:
        result = completed_result(plan_path.parent, run)
        if result is None:
            missing.append(run["run_id"])
            continue
        recomputed = metrics(
            np.asarray(result["y_true"]), np.asarray(result["predictions"])
        )
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            if not np.isclose(recomputed[metric], result[metric], atol=1e-12, rtol=0):
                raise ValueError("Stored metrics differ from raw predictions")
        config, stats = result["resolved_config"], result["statistics"]
        row = {
            "run_id": run["run_id"],
            "model_id": run["model_id"],
            "dataset": config["dataset"],
            "name": run["name"],
            "repetition": run["repetition"],
            "split": run["split"],
            "packing_factor": config["packing_factor"],
            "batch_size": config["batch_size"],
            "shots": config["shots"],
            "aer_device": config["aer_device"],
            "aer_method": config["aer_method"],
            "total_s": result["total_s"],
            "setup_s": result["setup_s"],
            "predict_s": result["predict_s"],
            "balanced_accuracy": result["balanced_accuracy"],
            "logical_circuits": stats["logical_circuits"],
            "backend_requests": stats["backend_requests"],
            "composite_circuits": stats["composite_circuits"],
            "composite_shots": stats["composite_shots"],
            "process_peak_rss_bytes": result["memory"]["process_peak_rss_bytes"],
        }
        rows.append(row)
        groups[(row["dataset"], row["name"], row["model_id"])].append(row)
    if missing and not allow_incomplete:
        raise ValueError(
            f"{len(missing)} missing runs; use allow_incomplete only for exploratory reports"
        )
    if not rows:
        raise ValueError("No completed runs to analyze")
    comparisons = []
    for dataset, name in sorted({(row["dataset"], row["name"]) for row in rows}):
        differences, ratios = [], []
        for key, values in groups.items():
            if key[:2] != (dataset, name):
                continue
            baseline = groups.get((dataset, reference, key[2]))
            if not baseline:
                continue
            differences.append(
                np.mean([v["balanced_accuracy"] for v in values])
                - np.mean([v["balanced_accuracy"] for v in baseline])
            )
            ratios.append(
                np.median([v["total_s"] for v in baseline])
                / np.median([v["total_s"] for v in values])
            )
        if differences:
            comparisons.append(
                {
                    "dataset": dataset,
                    "name": name,
                    "reference": reference,
                    **paired_summary(differences, ratios),
                }
            )
    report = {
        "schema_version": 1,
        "campaign_id": campaign["campaign_id"],
        "completed_runs": len(rows),
        "planned_runs": len(campaign["runs"]),
        "missing_runs": missing,
        "failed_attempts": len(
            list(plan_path.parent.glob("runs/*/attempt-*/failure.json"))
        ),
        "reference": reference,
        "comparisons": comparisons,
        "method": "Mean quality difference and median time-ratio across paired frozen models; "
        "timing repetitions collapsed within each model. Percentile bootstrap: "
        "2000 model-level resamples, seed 0. No CI for a single paired model. "
        "Exploratory: this does not establish non-inferiority or quantum advantage.",
    }
    if output.exists() and any(output.iterdir()):
        raise ValueError("Analysis output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runs.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "comparisons.json", report)
    lines = [
        "# Exploratory campaign analysis",
        "",
        f"Campaign: `{campaign['campaign_id']}`. Completed: {len(rows)}/{len(campaign['runs'])}. "
        f"Reference: `{reference}`.",
        "",
        report["method"],
        "",
        "| Dataset | Execution | Paired models | Quality difference | Median speedup |",
        "|---|---|---:|---:|---:|",
    ]
    for item in comparisons:
        lines.append(
            f"| {item['dataset']} | {item['name']} | {item['paired_models']} | "
            f"{item['mean_quality_difference']:.6f} | {item['median_speedup']:.3f} |"
        )
    lines += [
        "",
        "Confidence intervals and missing-run IDs are in `comparisons.json`. "
        "Detailed counters and timings are in `runs.csv`. Process peak RSS is lifetime-wide, "
        "not an isolated per-run allocation. Total time includes dataset loading, setup, "
        "prediction, metrics, and provenance; it excludes model-file loading and final artifact writing.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    if plots:
        plt.switch_backend("Agg")
        fig, ax = plt.subplots()
        for name in sorted({row["name"] for row in rows}):
            selected = [row for row in rows if row["name"] == name]
            ax.scatter(
                [r["total_s"] for r in selected],
                [r["balanced_accuracy"] for r in selected],
                label=name,
            )
        ax.set(
            xlabel="Inference total time (s)",
            ylabel="Balanced accuracy",
            title="Exploratory quality–time observations",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "quality_time.pdf")
        fig.savefig(output / "quality_time.png", dpi=160)
        plt.close(fig)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--plots", action="store_true", help="Requires the analysis extra"
    )
    args = parser.parse_args(argv)
    report = analyze(
        args.plan,
        args.output,
        reference=args.reference,
        allow_incomplete=args.allow_incomplete,
        plots=args.plots,
    )
    print(
        json.dumps(
            {"completed_runs": report["completed_runs"], "output": str(args.output)}
        )
    )


if __name__ == "__main__":
    main()
