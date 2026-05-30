from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path


METRICS = ["mse", "mae", "nlpd", "coverage", "mean_prediction", "mean_variance", "loss_final"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparison plots from an EM benchmark sweep CSV.")
    parser.add_argument("--csv", required=True, help="Path to sweep CSV.")
    parser.add_argument("--out-dir", default=None, help="Directory for plots. Defaults to <csv-stem>_plots next to CSV.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.with_name(f"{csv_path.stem}_plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

    data = read_sweep_csv(csv_path)
    plot_metric_grid(data, out_dir / "metric_summary.png")
    for metric in METRICS:
        plot_metric(data, metric, out_dir / f"{metric}.png")
    print(f"Wrote plots to {out_dir}")


def read_sweep_csv(path: Path) -> dict[str, list[dict[str, float]]]:
    by_method: dict[str, list[dict[str, float]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            by_method[method].append(
                {
                    "step": int(row["step"]),
                    "selected_index": int(row["selected_index"]),
                    **{metric: float(row[metric]) for metric in METRICS},
                }
            )
    for rows in by_method.values():
        rows.sort(key=lambda item: item["step"])
    return dict(by_method)


def plot_metric_grid(data: dict[str, list[dict[str, float]]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cols = 3
    n_rows = (len(METRICS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows), sharex=True)
    for ax, metric in zip(axes.flat, METRICS):
        _plot_metric_on_axis(ax, data, metric)
    for ax in axes.flat[len(METRICS) :]:
        ax.axis("off")
    fig.suptitle("Benchmark Sweep Metrics", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_metric(data: dict[str, list[dict[str, float]]], metric: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _plot_metric_on_axis(ax, data, metric)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_metric_on_axis(ax, data: dict[str, list[dict[str, float]]], metric: str) -> None:
    for method, rows in data.items():
        steps = [row["step"] for row in rows]
        values = [row[metric] for row in rows]
        ax.plot(steps, values, label=method, linewidth=1.8)
    ax.set_title(metric.replace("_", " ").title())
    ax.set_xlabel("BO Step")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)


if __name__ == "__main__":
    main()
