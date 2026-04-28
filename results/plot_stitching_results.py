from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent
DATASETS = {
    "iris": {
        "label": "Iris",
        "color": "#1f77b4",
    },
    "wine": {
        "label": "Wine",
        "color": "#ff7f0e",
    },
}


def aggregate_dataset(dataset_key: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / f"{dataset_key}.csv")
    grouped = (
        df.groupby("batch_size")
        .agg(
            classical_accuracy_mean=("classical_accuracy", "mean"),
            classical_accuracy_std=("classical_accuracy", "std"),
            quantum_accuracy_mean=("quantum_accuracy", "mean"),
            quantum_accuracy_std=("quantum_accuracy", "std"),
            quantum_train_s_mean=("quantum_train_s", "mean"),
            quantum_train_s_std=("quantum_train_s", "std"),
            quantum_predict_s_mean=("quantum_predict_s", "mean"),
            quantum_predict_s_std=("quantum_predict_s", "std"),
            train_peak_python_mib_mean=("train_peak_python_mib", "mean"),
            train_peak_python_mib_std=("train_peak_python_mib", "std"),
            predict_peak_python_mib_mean=("predict_peak_python_mib", "mean"),
            predict_peak_python_mib_std=("predict_peak_python_mib", "std"),
        )
        .reset_index()
    )
    return grouped


def build_summary_row(dataset_key: str, stats: pd.DataFrame) -> dict[str, float | int | str]:
    first = stats.iloc[0]
    best_qdt_idx = stats["quantum_accuracy_mean"].idxmax()
    best_train_idx = stats["quantum_train_s_mean"].idxmin()
    best_predict_idx = stats["quantum_predict_s_mean"].idxmin()
    best_qdt = stats.loc[best_qdt_idx]
    best_train = stats.loc[best_train_idx]
    best_predict = stats.loc[best_predict_idx]

    train_b1 = float(stats.loc[stats["batch_size"] == 1, "quantum_train_s_mean"].iloc[0])
    predict_b1 = float(stats.loc[stats["batch_size"] == 1, "quantum_predict_s_mean"].iloc[0])
    mem_b1 = float(
        stats.loc[stats["batch_size"] == 1, "train_peak_python_mib_mean"].iloc[0]
    )
    mem_b20 = float(
        stats.loc[stats["batch_size"] == 20, "train_peak_python_mib_mean"].iloc[0]
    )

    return {
        "dataset": DATASETS[dataset_key]["label"],
        "classical_accuracy_mean": float(first["classical_accuracy_mean"]),
        "classical_accuracy_std": float(first["classical_accuracy_std"]),
        "best_qdt_factor": int(best_qdt["batch_size"]),
        "best_qdt_accuracy_mean": float(best_qdt["quantum_accuracy_mean"]),
        "best_qdt_accuracy_std": float(best_qdt["quantum_accuracy_std"]),
        "worst_qdt_accuracy_mean": float(stats["quantum_accuracy_mean"].min()),
        "best_train_factor": int(best_train["batch_size"]),
        "best_train_time_s": float(best_train["quantum_train_s_mean"]),
        "best_train_speedup": train_b1 / float(best_train["quantum_train_s_mean"]),
        "best_predict_factor": int(best_predict["batch_size"]),
        "best_predict_time_s": float(best_predict["quantum_predict_s_mean"]),
        "best_predict_speedup": predict_b1 / float(best_predict["quantum_predict_s_mean"]),
        "train_memory_b1_mib": mem_b1,
        "train_memory_b20_mib": mem_b20,
        "train_memory_growth_b20_over_b1": mem_b20 / mem_b1,
    }


def plot_runtime(stats_by_dataset: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    train_ax, predict_ax = axes

    for dataset_key, stats in stats_by_dataset.items():
        meta = DATASETS[dataset_key]
        x = stats["batch_size"]
        color = meta["color"]
        label = meta["label"]

        train_mean = stats["quantum_train_s_mean"]
        train_std = stats["quantum_train_s_std"]
        train_ax.plot(x, train_mean, color=color, linewidth=2.0, label=label)
        train_ax.fill_between(x, train_mean - train_std, train_mean + train_std, color=color, alpha=0.18)

        predict_mean = stats["quantum_predict_s_mean"]
        predict_std = stats["quantum_predict_s_std"]
        predict_ax.plot(x, predict_mean, color=color, linewidth=2.0, label=label)
        predict_ax.fill_between(
            x,
            predict_mean - predict_std,
            predict_mean + predict_std,
            color=color,
            alpha=0.18,
        )

    train_ax.set_title("Training Time")
    train_ax.set_xlabel("Stitching factor")
    train_ax.set_ylabel("Time [s]")
    train_ax.set_yscale("log")
    train_ax.grid(True, which="both", alpha=0.25)

    predict_ax.set_title("Prediction Time")
    predict_ax.set_xlabel("Stitching factor")
    predict_ax.set_ylabel("Time [s]")
    predict_ax.grid(True, alpha=0.25)

    handles, labels = train_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    for suffix in ("pdf", "png"):
        fig.savefig(RESULTS_DIR / f"runtime_stitching.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_memory(stats_by_dataset: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    train_ax, predict_ax = axes

    for dataset_key, stats in stats_by_dataset.items():
        meta = DATASETS[dataset_key]
        x = stats["batch_size"]
        color = meta["color"]
        label = meta["label"]

        train_mean = stats["train_peak_python_mib_mean"]
        train_std = stats["train_peak_python_mib_std"]
        train_ax.plot(x, train_mean, color=color, linewidth=2.0, label=label)
        train_ax.fill_between(x, train_mean - train_std, train_mean + train_std, color=color, alpha=0.18)

        predict_mean = stats["predict_peak_python_mib_mean"]
        predict_std = stats["predict_peak_python_mib_std"]
        predict_ax.plot(x, predict_mean, color=color, linewidth=2.0, label=label)
        predict_ax.fill_between(
            x,
            predict_mean - predict_std,
            predict_mean + predict_std,
            color=color,
            alpha=0.18,
        )

    train_ax.set_title("Training Peak Memory")
    train_ax.set_xlabel("Stitching factor")
    train_ax.set_ylabel("Memory [MiB]")
    train_ax.set_yscale("log")
    train_ax.grid(True, which="both", alpha=0.25)

    predict_ax.set_title("Prediction Peak Memory")
    predict_ax.set_xlabel("Stitching factor")
    predict_ax.set_ylabel("Memory [MiB]")
    predict_ax.set_yscale("log")
    predict_ax.grid(True, which="both", alpha=0.25)

    handles, labels = train_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    for suffix in ("pdf", "png"):
        fig.savefig(RESULTS_DIR / f"memory_stitching.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    stats_by_dataset = {dataset_key: aggregate_dataset(dataset_key) for dataset_key in DATASETS}
    plot_runtime(stats_by_dataset)
    plot_memory(stats_by_dataset)

    summary = pd.DataFrame(
        [build_summary_row(dataset_key, stats) for dataset_key, stats in stats_by_dataset.items()]
    )
    summary.to_csv(RESULTS_DIR / "summary_stats.csv", index=False)


if __name__ == "__main__":
    main()
