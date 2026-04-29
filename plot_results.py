from __future__ import annotations

import argparse
import csv
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_DATASETS = ["tgbn-trade", "tgbn-genre"]
DEFAULT_MODELS = ["tgn", "tgat", "dyrep", "evolvegcn", "sage"]
EDGE_MODES = ["full", "unweighted_ones", "gaussian_noise", "temporal_delta"]
NODE_MODES = [
    "none",
    "gaussian_noise",
    "snapshot_pagerank",
    "snapshot_gae",
    "snapshot_node2vec",
    "snapshot_deepwalk",
]

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

            row.setdefault("feature_mode", "full")
            row.setdefault("node_feature_mode", "none")
            rows.append(row)

    return rows


def select_rows(
    rows: list[dict[str, str]],
    *,
    ablation_kind: str,
    variant_modes: list[str],
    fixed_feature_mode: str,
    fixed_node_feature_mode: str,
    metric_key: str,
    max_events: int | None,
) -> dict[tuple[str, str, str], dict[str, str]]:
    selected: dict[tuple[str, str, str], dict[str, str]] = {}

    for row in rows:
        if max_events is not None and str(max_events) != str(row.get("max_events", "")):
            continue

        feature_mode = row.get("feature_mode", "full")
        node_mode = row.get("node_feature_mode", "none")

        if ablation_kind == "edge":
            if node_mode != fixed_node_feature_mode:
                continue
            variant = feature_mode
        else:
            if feature_mode != fixed_feature_mode:
                continue
            variant = node_mode

        if variant not in variant_modes:
            continue

        key = (row["dataset"], row["model"], variant)
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


def present_variant_modes(
    rows: list[dict[str, str]],
    *,
    ablation_kind: str,
    candidate_modes: list[str],
    fixed_feature_mode: str,
    fixed_node_feature_mode: str,
    max_events: int | None,
) -> list[str]:
    present: list[str] = []
    seen = set()
    for row in rows:
        if max_events is not None and str(max_events) != str(row.get("max_events", "")):
            continue

        feature_mode = row.get("feature_mode", "full")
        node_mode = row.get("node_feature_mode", "none")

        if ablation_kind == "edge":
            if node_mode != fixed_node_feature_mode:
                continue
            mode = feature_mode
        else:
            if feature_mode != fixed_feature_mode:
                continue
            mode = node_mode

        if mode in candidate_modes and mode not in seen:
            seen.add(mode)
            present.append(mode)
    return present


def _label_for_mode(ablation_kind: str, mode: str) -> str:
    if ablation_kind == "edge":
        mapping = {
            "full": "Edge: full",
            "unweighted_ones": "Edge: ones",
            "gaussian_noise": "Edge: noise",
            "temporal_delta": "Edge: delta-t",
        }
        return mapping.get(mode, mode)

    mapping = {
        "none": "Node-init: none",
        "gaussian_noise": "Node-init: noise",
        "snapshot_pagerank": "Node-init: pagerank",
        "snapshot_gae": "Node-init: GAE",
        "snapshot_node2vec": "Node-init: node2vec",
        "snapshot_deepwalk": "Node-init: deepwalk",
    }
    return mapping.get(mode, mode)


def plot_single_metric(
    *,
    datasets: list[str],
    metric: str,
    ablation_kind: str,
    variant_modes: list[str],
    fixed_feature_mode: str,
    fixed_node_feature_mode: str,
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

    width = 0.82 / max(1, len(variant_modes))
    x = np.arange(len(DEFAULT_MODELS))

    for ax, dataset in zip(axes, datasets):
        for i, variant in enumerate(variant_modes):
            vals: list[float] = []
            for model in DEFAULT_MODELS:
                row = selected.get((dataset, model, variant))
                vals.append(float("nan") if row is None else to_float(row.get(metric_key, "nan")))
            offset = (i - (len(variant_modes) - 1) / 2.0) * width
            ax.bar(x + offset, vals, width=width, label=_label_for_mode(ablation_kind, variant))

        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in DEFAULT_MODELS])
        ax.set_title(dataset)
        ax.set_xlabel("Model")
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel(metric.upper())
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(3, len(variant_modes)), frameon=False, bbox_to_anchor=(0.5, 1.02))

    if ablation_kind == "edge":
        title = f"Edge-Message Ablation ({metric.upper()}) | fixed node_feature_mode={fixed_node_feature_mode}"
    else:
        title = f"Node-Initialization Ablation ({metric.upper()}) | fixed feature_mode={fixed_feature_mode}"
    fig.suptitle(title, y=1.08)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")


def plot_all_metrics(
    *,
    datasets: list[str],
    ablation_kind: str,
    variant_modes: list[str],
    fixed_feature_mode: str,
    fixed_node_feature_mode: str,
    selected: dict[tuple[str, str, str], dict[str, str]],
    output: Path,
) -> None:
    metrics = ["ndcg", "ap", "auroc"]
    fig, axes = plt.subplots(
        len(metrics),
        len(datasets),
        figsize=(6 * len(datasets), 3.9 * len(metrics)),
        sharex=True,
        constrained_layout=True,
    )
    if len(datasets) == 1:
        axes = np.array(axes).reshape(len(metrics), 1)

    width = 0.82 / max(1, len(variant_modes))
    x = np.arange(len(DEFAULT_MODELS))

    for r, metric in enumerate(metrics):
        metric_key = metric_column(metric)
        for c, dataset in enumerate(datasets):
            ax = axes[r, c]
            for i, variant in enumerate(variant_modes):
                vals: list[float] = []
                for model in DEFAULT_MODELS:
                    row = selected.get((dataset, model, variant))
                    vals.append(float("nan") if row is None else to_float(row.get(metric_key, "nan")))
                offset = (i - (len(variant_modes) - 1) / 2.0) * width
                ax.bar(x + offset, vals, width=width, label=_label_for_mode(ablation_kind, variant))

            ax.set_xticks(x)
            ax.set_xticklabels([m.upper() for m in DEFAULT_MODELS])
            if r == 0:
                ax.set_title(dataset)
            if c == 0:
                ax.set_ylabel(metric.upper())
            ax.grid(axis="y", alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(3, len(variant_modes)), frameon=False, bbox_to_anchor=(0.5, 1.02))

    if ablation_kind == "edge":
        title = f"Edge-Message Ablation (NDCG, AP, AUROC) | fixed node_feature_mode={fixed_node_feature_mode}"
    else:
        title = f"Node-Initialization Ablation (NDCG, AP, AUROC) | fixed feature_mode={fixed_feature_mode}"
    fig.suptitle(title, y=1.06)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot edge-message or node-initialization ablation results.")
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("results/baseline_results.csv"),
        help="CSV produced by main.py",
    )
    parser.add_argument(
        "--ablation-kind",
        type=str,
        default="edge",
        choices=["edge", "node"],
        help="Which pathway is ablated in the figure.",
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
        "--fixed-feature-mode",
        type=str,
        default="full",
        help="For node ablations: keep rows with this feature_mode.",
    )
    parser.add_argument(
        "--fixed-node-feature-mode",
        type=str,
        default="none",
        help="For edge ablations: keep rows with this node_feature_mode.",
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
        default=Path("results/ablation_plot.png"),
        help="Output figure path.",
    )
    args = parser.parse_args()

    if not args.results_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {args.results_csv}")

    rows = parse_results_rows(args.results_csv)
    if not rows:
        raise RuntimeError("No parseable rows found in results CSV.")

    candidate_modes = EDGE_MODES if args.ablation_kind == "edge" else NODE_MODES
    variant_modes = present_variant_modes(
        rows,
        ablation_kind=args.ablation_kind,
        candidate_modes=candidate_modes,
        fixed_feature_mode=args.fixed_feature_mode,
        fixed_node_feature_mode=args.fixed_node_feature_mode,
        max_events=args.max_events,
    )
    if not variant_modes:
        raise RuntimeError(
            "No rows match the requested ablation/fixed-mode filters. "
            f"ablation_kind={args.ablation_kind}, "
            f"fixed_feature_mode={args.fixed_feature_mode}, "
            f"fixed_node_feature_mode={args.fixed_node_feature_mode}, "
            f"max_events={args.max_events}."
        )
    if len(variant_modes) == 1:
        warnings.warn(
            f"Only one variant mode is present in filtered data: {variant_modes[0]}. "
            "The plot will show a single series."
        )

    selected = select_rows(
        rows,
        ablation_kind=args.ablation_kind,
        variant_modes=variant_modes,
        fixed_feature_mode=args.fixed_feature_mode,
        fixed_node_feature_mode=args.fixed_node_feature_mode,
        metric_key="test_ndcg",
        max_events=args.max_events,
    )

    if args.metric == "all":
        plot_all_metrics(
            datasets=args.datasets,
            ablation_kind=args.ablation_kind,
            variant_modes=variant_modes,
            fixed_feature_mode=args.fixed_feature_mode,
            fixed_node_feature_mode=args.fixed_node_feature_mode,
            selected=selected,
            output=args.output,
        )
    else:
        plot_single_metric(
            datasets=args.datasets,
            metric=args.metric,
            ablation_kind=args.ablation_kind,
            variant_modes=variant_modes,
            fixed_feature_mode=args.fixed_feature_mode,
            fixed_node_feature_mode=args.fixed_node_feature_mode,
            selected=selected,
            output=args.output,
        )

    print(f"Saved chart: {args.output}")
    print(
        f"Ablation kind={args.ablation_kind}, "
        f"fixed_feature_mode={args.fixed_feature_mode}, "
        f"fixed_node_feature_mode={args.fixed_node_feature_mode}"
    )


if __name__ == "__main__":
    main()
