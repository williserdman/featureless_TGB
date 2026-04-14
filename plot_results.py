from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_DATASETS = ["tgbn-trade", "tgbn-genre"]
DEFAULT_MODELS = ["tgn", "tgat", "dyrep", "evolvegcn", "sage"]
VALID_MODES = {"full", "unweighted_ones", "gaussian_noise", "temporal_delta"}

OLD_FIELDS = [
    "dataset",
    "model",
    "feature_mode",
    "noise_std",
    "noise_seed",
    "base_lr",
    "effective_lr",
    "grad_clip_norm",
    "epochs",
    "max_events",
    "temporal_sampling",
    "temporal_stride",
    "temporal_ratio",
    "temporal_seed",
    "primary_metric",
    "test_primary",
    "test_ndcg",
    "test_auroc",
    "test_ap",
    "test_processed_events",
    "test_labeled_timestamps",
    "test_empty_label_events",
    "test_label_batches",
    "final_train_primary",
    "final_train_ndcg",
    "final_train_auroc",
    "final_train_ap",
    "final_val_primary",
    "final_val_ndcg",
    "final_val_auroc",
    "final_val_ap",
]

NEW_FIELDS = [
    "dataset",
    "model",
    "feature_mode",
    "feature_dim",
    "noise_std",
    "noise_seed",
    "temporal_ema_alpha",
    "pagerank_interval",
    "recency_tau",
    "base_lr",
    "effective_lr",
    "grad_clip_norm",
    "epochs",
    "max_events",
    "temporal_sampling",
    "temporal_stride",
    "temporal_ratio",
    "temporal_seed",
    "primary_metric",
    "test_primary",
    "test_ndcg",
    "test_auroc",
    "test_ap",
    "test_processed_events",
    "test_labeled_timestamps",
    "test_empty_label_events",
    "test_label_batches",
    "final_train_primary",
    "final_train_ndcg",
    "final_train_auroc",
    "final_train_ap",
    "final_val_primary",
    "final_val_ndcg",
    "final_val_auroc",
    "final_val_ap",
]

LATEST_FIELDS = [
    "dataset",
    "model",
    "feature_mode",
    "feature_dim",
    "noise_std",
    "noise_seed",
    "node_feature_mode",
    "node_feature_dim",
    "node_feature_noise_std",
    "node_feature_noise_seed",
    "base_lr",
    "effective_lr",
    "grad_clip_norm",
    "epochs",
    "max_events",
    "temporal_sampling",
    "temporal_stride",
    "temporal_ratio",
    "temporal_seed",
    "primary_metric",
    "test_primary",
    "test_ndcg",
    "test_auroc",
    "test_ap",
    "test_processed_events",
    "test_labeled_timestamps",
    "test_empty_label_events",
    "test_label_batches",
    "final_train_primary",
    "final_train_ndcg",
    "final_train_auroc",
    "final_train_ap",
    "final_val_primary",
    "final_val_ndcg",
    "final_val_auroc",
    "final_val_ap",
]


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def metric_column(metric_name: str) -> str:
    metric_name = metric_name.lower()
    mapping = {
        "ndcg": "test_ndcg",
        "auroc": "test_auroc",
        "ap": "test_ap",
    }
    if metric_name not in mapping:
        raise ValueError(f"Unsupported metric: {metric_name}")
    return mapping[metric_name]


def parse_results_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.reader(fp)
        _ = next(reader, None)
        for raw in reader:
            if len(raw) == len(OLD_FIELDS):
                row = dict(zip(OLD_FIELDS, raw))
            elif len(raw) == len(NEW_FIELDS):
                row = dict(zip(NEW_FIELDS, raw))
            elif len(raw) == len(LATEST_FIELDS):
                row = dict(zip(LATEST_FIELDS, raw))
            else:
                continue

            if row.get("feature_mode") not in VALID_MODES:
                continue
            rows.append(row)

    return rows


def select_rows_by_key(
    rows: list[dict[str, str]],
    *,
    metric_key: str,
    max_events: int | None,
) -> dict[tuple[str, str, str], dict[str, str]]:
    selected: dict[tuple[str, str, str], dict[str, str]] = {}

    for row in rows:
        if max_events is not None:
            row_events = row.get("max_events", "")
            if str(max_events) != str(row_events):
                continue

        key = (row["dataset"], row["model"], row["feature_mode"])
        current = selected.get(key)
        if current is None:
            selected[key] = row
            continue

        cur_val = to_float(current.get(metric_key, "nan"))
        new_val = to_float(row.get(metric_key, "nan"))

        cur_finite = np.isfinite(cur_val)
        new_finite = np.isfinite(new_val)
        if new_finite and not cur_finite:
            selected[key] = row
            continue
        if new_finite == cur_finite:
            selected[key] = row

    return selected


def plot_single_metric(
    *,
    datasets: list[str],
    metric: str,
    selected: dict[tuple[str, str, str], dict[str, str]],
    output: Path,
) -> None:
    metric_key = metric_column(metric)
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(6 * len(datasets), 4.8),
        sharey=True,
        constrained_layout=True,
    )
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        full_vals: list[float] = []
        noise_vals: list[float] = []
        for model in DEFAULT_MODELS:
            full_row = selected.get((dataset, model, "full"))
            noise_row = selected.get((dataset, model, "gaussian_noise"))
            full_vals.append(float("nan") if full_row is None else to_float(full_row.get(metric_key, "nan")))
            noise_vals.append(float("nan") if noise_row is None else to_float(noise_row.get(metric_key, "nan")))

        x = np.arange(len(DEFAULT_MODELS))
        width = 0.38
        ax.bar(x - width / 2, full_vals, width=width, label="Full Features")
        ax.bar(x + width / 2, noise_vals, width=width, label="Gaussian Noise")
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in DEFAULT_MODELS])
        ax.set_title(dataset)
        ax.set_xlabel("Model")
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel(metric.upper())
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Feature Ablation: Full vs Gaussian Noise ({metric.upper()})", y=1.08)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")


def plot_all_metrics(
    *,
    datasets: list[str],
    selected: dict[tuple[str, str, str], dict[str, str]],
    output: Path,
) -> None:
    metrics = ["ndcg", "ap", "auroc"]
    fig, axes = plt.subplots(
        len(metrics),
        len(datasets),
        figsize=(6 * len(datasets), 3.8 * len(metrics)),
        sharex=True,
        constrained_layout=True,
    )
    if len(datasets) == 1:
        axes = np.array(axes).reshape(len(metrics), 1)

    for r, metric in enumerate(metrics):
        metric_key = metric_column(metric)
        for c, dataset in enumerate(datasets):
            ax = axes[r, c]
            full_vals: list[float] = []
            noise_vals: list[float] = []
            for model in DEFAULT_MODELS:
                full_row = selected.get((dataset, model, "full"))
                noise_row = selected.get((dataset, model, "gaussian_noise"))
                full_vals.append(float("nan") if full_row is None else to_float(full_row.get(metric_key, "nan")))
                noise_vals.append(float("nan") if noise_row is None else to_float(noise_row.get(metric_key, "nan")))

            x = np.arange(len(DEFAULT_MODELS))
            width = 0.38
            ax.bar(x - width / 2, full_vals, width=width, label="Full Features")
            ax.bar(x + width / 2, noise_vals, width=width, label="Gaussian Noise")
            ax.set_xticks(x)
            ax.set_xticklabels([m.upper() for m in DEFAULT_MODELS])
            if r == 0:
                ax.set_title(dataset)
            if c == 0:
                ax.set_ylabel(metric.upper())
            ax.grid(axis="y", alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Feature Ablation: Full vs Gaussian Noise (NDCG, AP, AUROC)", y=1.06)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot full-vs-noise baseline comparison.")
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("results/baseline_results.csv"),
        help="CSV produced by main.py",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="ndcg",
        choices=["ndcg", "auroc", "ap", "all"],
        help="Metric to visualize on y-axis.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional filter so only rows with this max_events value are plotted.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to include in the chart.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/full_vs_noise_drop.png"),
        help="Output figure path.",
    )
    args = parser.parse_args()

    if not args.results_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {args.results_csv}")

    rows = parse_results_rows(args.results_csv)
    if not rows:
        raise RuntimeError("No parseable rows found in results CSV.")

    selected = select_rows_by_key(rows, metric_key="test_ndcg", max_events=args.max_events)

    if args.metric == "all":
        plot_all_metrics(datasets=args.datasets, selected=selected, output=args.output)
    else:
        plot_single_metric(
            datasets=args.datasets,
            metric=args.metric,
            selected=selected,
            output=args.output,
        )

    print(f"Saved chart: {args.output}")

    for dataset in args.datasets:
        print(f"\n{dataset} values:")
        for model in DEFAULT_MODELS:
            full_row = selected.get((dataset, model, "full"))
            noise_row = selected.get((dataset, model, "gaussian_noise"))
            ndcg_full = float("nan") if full_row is None else to_float(full_row.get("test_ndcg", "nan"))
            ndcg_noise = float("nan") if noise_row is None else to_float(noise_row.get("test_ndcg", "nan"))
            ap_full = float("nan") if full_row is None else to_float(full_row.get("test_ap", "nan"))
            ap_noise = float("nan") if noise_row is None else to_float(noise_row.get("test_ap", "nan"))
            auroc_full = float("nan") if full_row is None else to_float(full_row.get("test_auroc", "nan"))
            auroc_noise = float("nan") if noise_row is None else to_float(noise_row.get("test_auroc", "nan"))
            print(
                f"  {model}: full(ndcg/ap/auroc)={ndcg_full:.4f}/{ap_full:.4f}/{auroc_full:.4f} "
                f"noise={ndcg_noise:.4f}/{ap_noise:.4f}/{auroc_noise:.4f}"
            )


if __name__ == "__main__":
    main()
