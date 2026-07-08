#!/usr/bin/env python3
"""Plot paper-facing prediction-set figures from evaluator JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BUCKET_ORDER = ["0", "1-2", "3-10", "11-100", ">100"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", type=str, required=True)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def load_summary(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def output_dir_for(metrics_path: Path, requested: str | None) -> Path:
    output_dir = Path(requested) if requested else metrics_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def short_name(name: str) -> str:
    return name.replace("conformal_", "conf_").replace("_0.95", "").replace("_0.05", "")


def plot_coverage_efficiency(summary: dict[str, Any], output_dir: Path) -> None:
    methods = summary["methods"]
    names = list(methods)
    x = [methods[name]["avg_set_size"] for name in names]
    y = [methods[name]["coverage"] for name in names]
    mass = [methods[name]["avg_retained_mass"] for name in names]

    plt.figure(figsize=(8.5, 5.5))
    scatter = plt.scatter(x, y, s=90, c=mass, cmap="viridis", edgecolors="black", linewidths=0.5)
    for name, x_val, y_val in zip(names, x, y):
        plt.annotate(short_name(name), (x_val, y_val), fontsize=8, xytext=(5, 5), textcoords="offset points")
    plt.xlabel("Average support size")
    plt.ylabel("True-token coverage")
    plt.title("Prediction-set coverage vs support efficiency")
    plt.grid(alpha=0.25)
    cb = plt.colorbar(scatter)
    cb.set_label("Average retained probability mass")
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_set_coverage_efficiency.png", dpi=180)
    plt.close()


def plot_bucket_coverage(summary: dict[str, Any], output_dir: Path) -> None:
    methods = summary["methods"]
    names = list(methods)
    x = np.arange(len(BUCKET_ORDER))
    width = min(0.8 / max(len(names), 1), 0.12)

    plt.figure(figsize=(11, 5.5))
    for idx, name in enumerate(names):
        values = []
        for bucket in BUCKET_ORDER:
            coverage = methods[name]["bucket_summary"].get(bucket, {}).get("coverage")
            values.append(np.nan if coverage is None else coverage)
        offset = (idx - (len(names) - 1) / 2) * width
        plt.bar(x + offset, values, width=width, label=short_name(name))
    plt.xticks(x, BUCKET_ORDER)
    plt.ylim(0, 1.05)
    plt.xlabel("Target-token frequency bucket in calibration/eval corpus")
    plt.ylabel("True-token coverage")
    plt.title("Coverage by target-token frequency bucket")
    plt.grid(alpha=0.25, axis="y")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_set_bucket_coverage.png", dpi=180)
    plt.close()


def plot_distribution_summary(summary: dict[str, Any], output_dir: Path) -> None:
    methods = summary["methods"]
    names = list(methods)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    avg_sizes = [methods[name]["avg_set_size"] for name in names]
    ax.bar([short_name(name) for name in names], avg_sizes)
    ax.set_yscale("log")
    ax.set_ylabel("Average support size (log)")
    ax.set_title("Support size")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    retained_mass = [methods[name]["avg_retained_mass"] for name in names]
    ax.bar([short_name(name) for name in names], retained_mass)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Average retained mass")
    ax.set_title("Retained probability mass")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[2]
    for name in names:
        qs = methods[name].get("set_size_quantiles", {})
        if not qs:
            continue
        xs = [10, 25, 50, 75, 90, 95, 99]
        ys = [qs[str(q)] for q in xs]
        ax.plot(xs, ys, marker="o", label=short_name(name))
    ax.set_yscale("log")
    ax.set_xlabel("Quantile")
    ax.set_ylabel("Support size (log)")
    ax.set_title("Support-size quantiles")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "prediction_set_distribution_summary.png", dpi=180)
    plt.close()


def write_figure_summary(summary: dict[str, Any], output_dir: Path) -> None:
    methods = summary["methods"]
    compact = {
        "model": summary.get("model"),
        "dataset": summary.get("dataset"),
        "n_positions": summary.get("n_positions", summary.get("n_eval")),
        "q_hat": summary.get("q_hat"),
        "methods": {
            name: {
                "coverage": info["coverage"],
                "avg_set_size": info["avg_set_size"],
                "median_set_size": info["median_set_size"],
                "avg_retained_mass": info["avg_retained_mass"],
            }
            for name, info in methods.items()
        },
    }
    with open(output_dir / "prediction_set_figure_summary.json", "w") as f:
        json.dump(compact, f, indent=2)


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    summary = load_summary(str(metrics_path))
    output_dir = output_dir_for(metrics_path, args.output_dir)

    plot_coverage_efficiency(summary, output_dir)
    plot_bucket_coverage(summary, output_dir)
    plot_distribution_summary(summary, output_dir)
    write_figure_summary(summary, output_dir)

    print(f"[plot-prediction-sets] wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
