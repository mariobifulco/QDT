from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent
DATASETS = {
    "iris": {
        "label": "Iris",
        "color": "#1f77b4",
        "linestyle": "-",
    },
    "wine": {
        "label": "Wine",
        "color": "#ff7f0e",
        "linestyle": "--",
    },
    "breast_cancer": {
        "label": "Breast Cancer",
        "color": "#10c100",
        "linestyle": ":",
    },
}


def build_summary_row(
    dataset_key: str, stats: pd.DataFrame
) -> dict[str, float | int | str]:
    first = stats.iloc[0]
    best_qdt_idx = stats["quantum_accuracy_mean"].idxmax()
    best_train_idx = stats["quantum_train_s_mean"].idxmin()
    best_predict_idx = stats["quantum_predict_s_mean"].idxmin()
    best_qdt = stats.loc[best_qdt_idx]
    best_train = stats.loc[best_train_idx]
    best_predict = stats.loc[best_predict_idx]

    train_b1 = float(
        stats.loc[stats["batch_size"] == 1, "quantum_train_s_mean"].iloc[0]
    )
    predict_b1 = float(
        stats.loc[stats["batch_size"] == 1, "quantum_predict_s_mean"].iloc[0]
    )
    mem_b1 = float(
        stats.loc[stats["batch_size"] == 1, "train_peak_python_mib_mean"].iloc[0]
    )
    mem_b21 = float(
        stats.loc[stats["batch_size"] == 21, "train_peak_python_mib_mean"].iloc[0]
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
        "best_predict_speedup": predict_b1
        / float(best_predict["quantum_predict_s_mean"]),
        "train_memory_b1_mib": mem_b1,
        "train_memory_b21_mib": mem_b21,
        "train_memory_growth_b21_over_b1": mem_b21 / mem_b1,
    }


def add_training_annotations(ax: plt.Axes) -> None:
    x1 = 12
    x2 = 18
    ax.axvline(x=x1, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(x=x2, color="gray", linestyle="--", alpha=0.7)

    xmin, xmax = ax.get_xlim()
    trans = ax.get_xaxis_transform()
    bbox_props = dict(
        boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="lightgray"
    )

    ax.text(
        (xmin + x1) / 2,
        0.95,
        "under-batched",
        transform=trans,
        ha="center",
        va="top",
        fontsize=9,
        bbox=bbox_props,
    )

    ax.text(
        (x1 + x2) / 2,
        0.95,
        "sweet spot",
        transform=trans,
        ha="center",
        va="top",
        fontsize=9,
        bbox=bbox_props,
    )

    ax.text(
        (x2 + xmax) / 2,
        0.95,
        "over-b.",
        transform=trans,
        ha="center",
        va="top",
        fontsize=9,
        bbox=bbox_props,
    )


def plot_runtime(
    stats_by_dataset: dict[str, pd.DataFrame],
    output_dir: Path = RESULTS_DIR,
    annotate: bool = True,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    train_ax, predict_ax = axes

    for dataset_key, stats in stats_by_dataset.items():
        meta = DATASETS[dataset_key]
        x = stats["batch_size"]
        color = meta["color"]
        label = meta["label"]
        linestyle = meta["linestyle"]

        train_mean = stats["quantum_train_s_mean"]
        train_std = stats["quantum_train_s_std"]
        train_ax.plot(
            x, train_mean, color=color, linestyle=linestyle, linewidth=2.0, label=label
        )
        train_ax.fill_between(
            x, train_mean - train_std, train_mean + train_std, color=color, alpha=0.18
        )

        predict_mean = stats["quantum_predict_s_mean"]
        predict_std = stats["quantum_predict_s_std"]
        predict_ax.plot(
            x,
            predict_mean,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
        predict_ax.fill_between(
            x,
            predict_mean - predict_std,
            predict_mean + predict_std,
            color=color,
            alpha=0.18,
        )

    train_ax.set_title("Training Time")
    train_ax.set_xlabel("Batching factor")
    train_ax.set_ylabel("Time [s]")
    train_ax.set_yscale("log")
    train_ax.grid(True, which="both", alpha=0.25)
    train_ax.legend(loc="lower left", frameon=True)

    train_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if annotate:
        add_training_annotations(train_ax)

    predict_ax.set_title("Prediction Time")
    predict_ax.set_xlabel("Batching factor")
    predict_ax.set_ylabel("Time [s]")
    predict_ax.grid(True, alpha=0.25)
    predict_ax.legend(loc="upper right", frameon=True)

    predict_ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.savefig(output_dir / "runtime_batching.pdf", dpi=240, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_memory(
    stats_by_dataset: dict[str, pd.DataFrame],
    output_dir: Path = RESULTS_DIR,
    annotate: bool = True,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    train_ax, predict_ax = axes

    for dataset_key, stats in stats_by_dataset.items():
        meta = DATASETS[dataset_key]
        x = stats["batch_size"]
        color = meta["color"]
        label = meta["label"]
        linestyle = meta["linestyle"]

        train_mean = stats["train_peak_python_mib_mean"]
        train_std = stats["train_peak_python_mib_std"]
        train_ax.plot(
            x, train_mean, color=color, linestyle=linestyle, linewidth=2.0, label=label
        )
        train_ax.fill_between(
            x, train_mean - train_std, train_mean + train_std, color=color, alpha=0.18
        )

        predict_mean = stats["predict_peak_python_mib_mean"]
        predict_std = stats["predict_peak_python_mib_std"]
        predict_ax.plot(
            x,
            predict_mean,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
        predict_ax.fill_between(
            x,
            predict_mean - predict_std,
            predict_mean + predict_std,
            color=color,
            alpha=0.18,
        )

    train_ax.set_title("Training Peak Memory")
    train_ax.set_xlabel("Batching factor")
    train_ax.set_ylabel("Memory [MiB]")
    train_ax.set_yscale("log")
    train_ax.grid(True, which="both", alpha=0.25)
    train_ax.legend(loc="upper left", frameon=True)
    train_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if annotate:
        add_training_annotations(train_ax)

    predict_ax.set_title("Prediction Peak Memory")
    predict_ax.set_xlabel("Batching factor")
    predict_ax.set_ylabel("Memory [MiB]")
    predict_ax.set_yscale("log")
    predict_ax.grid(True, which="both", alpha=0.25)
    predict_ax.legend(loc="upper left", frameon=True)
    predict_ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.savefig(output_dir / "memory_batching.pdf", dpi=240, bbox_inches="tight")
    plt.close(fig)
    return fig


def main() -> None:
    stats_by_dataset = {
        dataset_key: pd.read_csv(f"{dataset_key}_mean.csv") for dataset_key in DATASETS
    }
    plot_runtime(stats_by_dataset)
    plot_memory(stats_by_dataset)

    summary = pd.DataFrame(
        [
            build_summary_row(dataset_key, stats)
            for dataset_key, stats in stats_by_dataset.items()
        ]
    )
    summary.to_csv(RESULTS_DIR / "summary_stats.csv", index=False)


if __name__ == "__main__":
    main()
