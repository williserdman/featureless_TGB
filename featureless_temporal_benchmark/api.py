from __future__ import annotations

from pathlib import Path

import torch
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset

from main import train_dataset


def inspect_dataset(dataset_name: str, root: str = "datasets") -> dict[str, object]:
    """Return a compact summary for a TGB node dataset."""
    dataset = PyGNodePropPredDataset(name=dataset_name, root=root)
    data = dataset.get_TemporalData()

    src = data.src
    dst = data.dst
    msg = data.msg
    t = data.t

    if src is None or dst is None or msg is None or t is None:
        raise RuntimeError("TemporalData is missing one or more required tensors.")

    num_nodes = int(torch.max(torch.stack([src.max(), dst.max()])).item()) + 1

    return {
        "dataset": dataset_name,
        "num_events": int(src.numel()),
        "num_nodes": num_nodes,
        "msg_dim": int(msg.size(-1)),
        "unique_timestamps": int(torch.unique(t).numel()),
        "train_events": int(dataset.train_mask.sum()),
        "val_events": int(dataset.val_mask.sum()),
        "test_events": int(dataset.test_mask.sum()),
        "eval_metric": str(dataset.eval_metric),
        "num_classes": int(dataset.num_classes),
    }


def run_experiment(
    *,
    dataset: str,
    model: str,
    root: str = "datasets",
    epochs: int = 2,
    lr: float = 1e-3,
    memory_dim: int = 128,
    time_dim: int = 32,
    max_events: int | None = None,
    device: str | None = None,
    tgat_heads: int = 4,
    temporal_sampling: str = "none",
    temporal_stride: int = 2,
    temporal_ratio: float = 0.5,
    temporal_seed: int = 42,
    feature_mode: str = "full",
    feature_dim: int = 128,
    noise_std: float = 1.0,
    noise_seed: int = 12345,
    embedding_dim: int = 64,
    walk_length: int = 20,
    walks_per_node: int = 3,
    context_size: int = 5,
    node2vec_p: float = 1.0,
    node2vec_q: float = 1.0,
    node_feature_mode: str = "none",
    node_feature_dim: int = 64,
    node_feature_noise_std: float = 1.0,
    node_feature_noise_seed: int = 54321,
    grad_clip_norm: float = 1.0,
    tgat_lr_mult: float = 0.1,
    results_csv: str | Path | None = None,
) -> dict[str, object]:
    """Run one dataset-model experiment and return structured results."""

    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_obj = torch.device(device_name)

    csv_path = None if results_csv is None else Path(results_csv)

    return train_dataset(
        dataset_name=dataset,
        model_name=model,
        root=root,
        epochs=epochs,
        lr=lr,
        memory_dim=memory_dim,
        time_dim=time_dim,
        max_events=max_events,
        device=device_obj,
        tgat_heads=tgat_heads,
        temporal_sampling=temporal_sampling,
        temporal_stride=temporal_stride,
        temporal_ratio=temporal_ratio,
        temporal_seed=temporal_seed,
        feature_mode=feature_mode,
        feature_dim=feature_dim,
        noise_std=noise_std,
        noise_seed=noise_seed,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        walks_per_node=walks_per_node,
        context_size=context_size,
        node2vec_p=node2vec_p,
        node2vec_q=node2vec_q,
        node_feature_mode=node_feature_mode,
        node_feature_dim=node_feature_dim,
        node_feature_noise_std=node_feature_noise_std,
        node_feature_noise_seed=node_feature_noise_seed,
        grad_clip_norm=grad_clip_norm,
        tgat_lr_mult=tgat_lr_mult,
        results_csv=csv_path,
    )


def run_recovery_suite(
    *,
    dataset: str,
    models: list[str],
    root: str = "datasets",
    epochs: int = 1,
    max_events: int | None = 50000,
    feature_modes: list[str] | None = None,
    node_feature_modes: list[str] | None = None,
    feature_dim: int = 128,
    node_feature_dim: int = 64,
    device: str | None = None,
    results_csv: str | Path | None = None,
) -> list[dict[str, object]]:
    """Run a concise full-vs-recovery suite and return all run outputs."""

    modes = feature_modes or [
        "full",
        "unweighted_ones",
        "gaussian_noise",
        "temporal_delta",
    ]
    node_modes = node_feature_modes or ["none", "gaussian_noise", "snapshot_pagerank", "snapshot_node2vec", "snapshot_deepwalk"]

    outputs: list[dict[str, object]] = []
    for model in models:
        for mode in modes:
            for node_mode in node_modes:
                out = run_experiment(
                    dataset=dataset,
                    model=model,
                    root=root,
                    epochs=epochs,
                    max_events=max_events,
                    feature_mode=mode,
                    feature_dim=feature_dim,
                    node_feature_mode=node_mode,
                    node_feature_dim=node_feature_dim,
                    device=device,
                    results_csv=results_csv,
                )
                outputs.append(out)

    return outputs
