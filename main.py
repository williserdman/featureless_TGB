import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import warnings

import torch

from metrics_utils import compute_auroc_ap
from models import BaseTemporalModel, build_model
from sampling import temporal_subsample_indices
from synthetic_features import (
    FeatureEngineConfig,
    NodeFeatureConfig,
    NodeFeatureGenerator,
    SyntheticFeatureEngine,
)
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset
from tgb.nodeproppred.evaluate import Evaluator
from torch import Tensor, nn

DEFAULT_DATASETS = ["tgbn-trade", "tgbn-genre", "tgbn-reddit", "tgbn-token"]
DEFAULT_MODELS = ["tgn", "tgat", "dyrep", "evolvegcn", "sage"]
RESULTS_FIELDNAMES = [
    "dataset",
    "model",
    "feature_mode",
    "feature_dim",
    "noise_std",
    "noise_seed",
    "temporal_ema_alpha",
    "pagerank_interval",
    "recency_tau",
    "embedding_dim",
    "walk_length",
    "walks_per_node",
    "context_size",
    "node2vec_p",
    "node2vec_q",
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


# struct to hold the output
@dataclass
class SplitOutput:
    avg_loss: float
    primary_metric_name: str
    metric_value: float
    ndcg: float
    auroc: float
    ap: float
    processed_events: int
    labeled_timestamps: int
    empty_label_events: int
    label_batches: int


def _fmt_metric(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value:.4f}"


def append_results_row(*, file_path: Path, row: dict[str, object]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)

    target_path = file_path
    if file_path.exists():
        with file_path.open("r", newline="", encoding="utf-8") as fp:
            first_line = fp.readline().strip()
        if first_line:
            existing_header = first_line.split(",")
            if existing_header != RESULTS_FIELDNAMES:
                target_path = file_path.with_name(f"{file_path.stem}_v2{file_path.suffix}")
                warnings.warn(
                    f"Results CSV schema mismatch in {file_path}. "
                    f"Appending new rows to {target_path} with the current schema."
                )

    normalized_row = {field: row.get(field, "") for field in RESULTS_FIELDNAMES}

    write_header = not target_path.exists()
    with target_path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=RESULTS_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(normalized_row)


# processes a batch
def consume_label_batches(
    *,  # forces all arguments to be passed by keyword
    dataset: PyGNodePropPredDataset,  # temporal node property dataset
    model: BaseTemporalModel,  # see models.py
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,  # loss fn
    current_time: int,  # current timestamp
    device: torch.device,
    grad_clip_norm: float,
) -> tuple[float, int, list[Tensor], list[Tensor]]:
    """Processes a batch of nodes"""
    total_loss = 0.0
    num_batches = 0
    y_true_batches: list[Tensor] = []
    y_pred_batches: list[Tensor] = []

    # loop to fetch batches of node labels for current time
    while True:
        label_tuple = dataset.get_node_label(
            current_time
        )  # this is labels for all the nodes
        if label_tuple is None:
            break

        _, label_nodes, labels = label_tuple  # break up what the dataset holds
        label_nodes = label_nodes.to(device)  # node ids
        labels = labels.to(device)  # node labels

        logits = model.get_logits(
            label_nodes=label_nodes, current_time=current_time
        )  # get prediction from model

        # step the model
        if optimizer is not None:
            optimizer.zero_grad()
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            model.detach_state()
            total_loss += float(loss.item())

        num_batches += 1
        y_true_batches.append(labels.detach().cpu())  # logging done on the CPU
        y_pred_batches.append(torch.sigmoid(logits).detach().cpu())

    return total_loss, num_batches, y_true_batches, y_pred_batches  # returned


def run_split(
    *,
    dataset: PyGNodePropPredDataset,
    data,
    split_mask: Tensor,
    evaluator: Evaluator,
    model: BaseTemporalModel,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
    eval_metric: str,
    max_events: int | None,
    temporal_sampling: str,
    temporal_stride: int,
    temporal_ratio: float,
    temporal_seed: int,
    grad_clip_norm: float,
    split_name: str,
    log_sampling: bool,
    feature_engine: SyntheticFeatureEngine,
    node_feature_generator: NodeFeatureGenerator | None,
) -> SplitOutput:
    """Handles full pass over a specific data split (train/test/val)"""

    # resamples the indices based on the temporal masking that we do to lighted the computational load
    # just striding, max events limiter is handled elsewhere
    edge_idx, report = temporal_subsample_indices(
        split_mask=split_mask,
        timestamps=data.t,
        strategy=temporal_sampling,
        stride=temporal_stride,
        ratio=temporal_ratio,
        seed=temporal_seed,
    )

    if log_sampling and report.strategy != "none":
        print(
            f"Sampling[{report.strategy}] kept {report.kept_events}/{report.original_events} "
            f"({100.0 * report.retention:.1f}%), est_speedup={report.speedup_estimate:.2f}x"
        )

    # max events, maybe combine into the temporal striding idea? TODO
    if max_events is not None:
        edge_idx = edge_idx[:max_events]

    total_loss = 0.0
    total_label_batches = 0
    y_true_all: list[Tensor] = []
    y_pred_all: list[Tensor] = []
    labeled_timestamps = 0
    empty_label_events = 0

    for idx in edge_idx.tolist():
        cur_t = int(data.t[idx].item())
        if node_feature_generator is not None:
            node_features = node_feature_generator.node_features_for_time(current_time=cur_t)
            model.inject_node_features(node_features=node_features.to(device))

        # can process
        loss, n_batches, y_true, y_pred = (
            consume_label_batches(  # predict node properties/train model
                dataset=dataset,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                current_time=cur_t,
                device=device,
                grad_clip_norm=grad_clip_norm,
            )
        )
        total_loss += loss
        total_label_batches += n_batches
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)
        if n_batches > 0:
            labeled_timestamps += 1
        else:
            empty_label_events += 1

        # slice dataset to get source, dest, timestamp, msg for the current event. moves to proper device and then updates model
        # maybe should be doing all this before initializing the training loop?
        src = data.src[idx : idx + 1].to(device)
        dst = data.dst[idx : idx + 1].to(device)
        t = data.t[idx : idx + 1].to(device)
        src_i = int(data.src[idx].item())
        dst_i = int(data.dst[idx].item())
        msg = feature_engine.event_message(
            event_index=idx,
            src=src_i,
            dst=dst_i,
            current_time=cur_t,
            base_msg=data.msg[idx : idx + 1],
        )
        msg = msg.to(device)
        model.update_event(src=src, dst=dst, t=t, msg=msg)
        feature_engine.update_state(src=src_i, dst=dst_i, current_time=cur_t)
        if node_feature_generator is not None:
            node_feature_generator.update_state(src=src_i, dst=dst_i, current_time=cur_t)

    if y_true_all:
        y_true_cat = torch.cat(
            y_true_all, dim=0
        )  # predictions concatenated into one large tensor
        y_pred_cat = torch.cat(y_pred_all, dim=0)
        metric_value = float(
            evaluator.eval(  # evaluation
                {
                    "y_true": y_true_cat,
                    "y_pred": y_pred_cat,
                    "eval_metric": [eval_metric],
                }
            )[eval_metric]
        )
        auroc, ap = compute_auroc_ap(y_true=y_true_cat, y_pred=y_pred_cat)
    else:
        if optimizer is not None:
            warnings.warn(
                f"Split {split_name}: no labels were processed across {int(edge_idx.numel())} events. "
                "This often means --max-events is too small for the first labeled timestamp in this split "
                "(for tgbn-trade, 5000 is often too small; try 20000+), or labels are sparse in this time window."
            )
        metric_value = float("nan")
        auroc = float("nan")
        ap = float("nan")

    if total_label_batches == 0:
        avg_loss = float("nan")
    else:
        avg_loss = total_loss / total_label_batches

    if log_sampling:
        print(
            f"Split[{split_name}] events={int(edge_idx.numel())} "
            f"labeled_timestamps={labeled_timestamps} "
            f"empty_label_events={empty_label_events} "
            f"label_batches={total_label_batches}"
        )

    ndcg = metric_value if eval_metric.lower() == "ndcg" else float("nan")
    return SplitOutput(
        avg_loss=avg_loss,
        primary_metric_name=eval_metric,
        metric_value=metric_value,
        ndcg=ndcg,
        auroc=auroc,
        ap=ap,
        processed_events=int(edge_idx.numel()),
        labeled_timestamps=labeled_timestamps,
        empty_label_events=empty_label_events,
        label_batches=total_label_batches,
    )  # returns SplitOutput


def train_dataset(
    dataset_name: str,
    model_name: str,
    root: str,
    epochs: int,
    lr: float,
    memory_dim: int,
    time_dim: int,
    max_events: int | None,
    device: torch.device,
    tgat_heads: int,
    temporal_sampling: str,
    temporal_stride: int,
    temporal_ratio: float,
    temporal_seed: int,
    feature_mode: str,
    feature_dim: int,
    noise_std: float,
    noise_seed: int,
    temporal_ema_alpha: float,
    pagerank_interval: int,
    recency_tau: float,
    embedding_dim: int,
    walk_length: int,
    walks_per_node: int,
    context_size: int,
    node2vec_p: float,
    node2vec_q: float,
    node_feature_mode: str,
    node_feature_dim: int,
    node_feature_noise_std: float,
    node_feature_noise_seed: int,
    grad_clip_norm: float,
    tgat_lr_mult: float,
    results_csv: Path | None,
) -> dict[str, object]:
    print(f"\n=== Dataset: {dataset_name} | Model: {model_name} ===")
    # training for specific model and dataset
    dataset = PyGNodePropPredDataset(name=dataset_name, root=root)  # defines dataset
    evaluator = Evaluator(name=dataset_name)  # create evaluator
    data = dataset.get_TemporalData()  # loads data

    # verify integrity of data, moves to device
    if data.src is None or data.dst is None or data.t is None or data.msg is None:
        raise RuntimeError("TemporalData is missing one or more required tensors.")

    src_tensor = data.src
    dst_tensor = data.dst
    msg_tensor = data.msg

    data = data.to(device)
    num_nodes = (
        int(torch.max(torch.stack([src_tensor.max(), dst_tensor.max()])).item()) + 1
    )
    full_msg_dim = int(msg_tensor.size(-1))
    raw_msg_dim = full_msg_dim
    if feature_mode != "full":
        raw_msg_dim = feature_dim
    num_classes = int(dataset.num_classes)
    eval_metric = str(dataset.eval_metric)
    effective_node_feature_dim = node_feature_dim if node_feature_mode != "none" else None

    feature_engine = SyntheticFeatureEngine(
        FeatureEngineConfig(
            mode=feature_mode,
            num_nodes=num_nodes,
            feature_dim=raw_msg_dim,
            noise_seed=noise_seed,
            noise_std=noise_std,
            temporal_ema_alpha=temporal_ema_alpha,
            pagerank_interval=pagerank_interval,
            recency_tau=recency_tau,
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            walks_per_node=walks_per_node,
            context_size=context_size,
            node2vec_p=node2vec_p,
            node2vec_q=node2vec_q,
        )
    )

    node_feature_generator: NodeFeatureGenerator | None = None
    if effective_node_feature_dim is not None:
        node_feature_generator = NodeFeatureGenerator(
            NodeFeatureConfig(
                mode=node_feature_mode,
                num_nodes=num_nodes,
                node_feature_dim=effective_node_feature_dim,
                noise_seed=node_feature_noise_seed,
                noise_std=node_feature_noise_std,
                temporal_ema_alpha=temporal_ema_alpha,
                pagerank_interval=pagerank_interval,
                recency_tau=recency_tau,
                embedding_dim=embedding_dim,
                walk_length=walk_length,
                walks_per_node=walks_per_node,
                context_size=context_size,
                node2vec_p=node2vec_p,
                node2vec_q=node2vec_q,
            )
        )

    model = build_model(
        model_name=model_name,
        num_nodes=num_nodes,
        raw_msg_dim=raw_msg_dim,
        memory_dim=memory_dim,
        time_dim=time_dim,
        num_classes=num_classes,
        tgat_heads=tgat_heads,
        node_feature_dim=effective_node_feature_dim,
    ).to(
        device
    )  # creates one of our specified models

    # boilerplate
    effective_lr = lr * tgat_lr_mult if model_name.lower() == "tgat" else lr
    optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr)
    criterion = nn.BCEWithLogitsLoss()
    if model_name.lower() == "tgat" and tgat_lr_mult != 1.0:
        print(f"TGAT LR override enabled: base_lr={lr:.2e}, effective_lr={effective_lr:.2e}")

    final_train_out: SplitOutput | None = None
    final_val_out: SplitOutput | None = None

    for epoch in range(1, epochs + 1):
        model.reset_state()  # reset at beginning of each epoch
        feature_engine.reset()
        if node_feature_generator is not None:
            node_feature_generator.reset()
        dataset.reset_label_time()  # time reset to 0

        model.train()  # model in train mode
        train_out = run_split(  # run a pass through our model, computes loss and metric
            dataset=dataset,
            data=data,
            split_mask=dataset.train_mask,  # only use training subset
            evaluator=evaluator,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            eval_metric=eval_metric,
            max_events=max_events,
            temporal_sampling=temporal_sampling,
            temporal_stride=temporal_stride,
            temporal_ratio=temporal_ratio,
            temporal_seed=temporal_seed,
            grad_clip_norm=grad_clip_norm,
            split_name="train",
            log_sampling=(epoch == 1),
            feature_engine=feature_engine,
            node_feature_generator=node_feature_generator,
        )

        model.eval()  # eval mode
        with torch.no_grad():
            val_out = run_split(
                dataset=dataset,
                data=data,
                split_mask=dataset.val_mask,  # only use validation subset
                evaluator=evaluator,
                model=model,
                optimizer=None,
                criterion=criterion,
                device=device,
                eval_metric=eval_metric,
                max_events=max_events,
                temporal_sampling=temporal_sampling,
                temporal_stride=temporal_stride,
                temporal_ratio=temporal_ratio,
                temporal_seed=temporal_seed,
                grad_clip_norm=0.0,
                split_name="val",
                log_sampling=(epoch == 1),
                feature_engine=feature_engine,
                node_feature_generator=node_feature_generator,
            )

        final_train_out = train_out
        final_val_out = val_out
        print(
            f"Epoch {epoch:02d} | train_loss={train_out.avg_loss:.4f} "
            f"train_{eval_metric}={_fmt_metric(train_out.metric_value)} "
            f"train_auroc={_fmt_metric(train_out.auroc)} "
            f"train_ap={_fmt_metric(train_out.ap)} "
            f"val_{eval_metric}={_fmt_metric(val_out.metric_value)} "
            f"val_auroc={_fmt_metric(val_out.auroc)} "
            f"val_ap={_fmt_metric(val_out.ap)}"
        )

    model.reset_state()  # after training another final check
    feature_engine.reset()
    if node_feature_generator is not None:
        node_feature_generator.reset()
    dataset.reset_label_time()
    model.eval()
    with torch.no_grad():
        _ = run_split(
            dataset=dataset,
            data=data,
            split_mask=dataset.train_mask,
            evaluator=evaluator,
            model=model,
            optimizer=None,
            criterion=criterion,
            device=device,
            eval_metric=eval_metric,
            max_events=max_events,
            temporal_sampling=temporal_sampling,
            temporal_stride=temporal_stride,
            temporal_ratio=temporal_ratio,
            temporal_seed=temporal_seed,
            grad_clip_norm=0.0,
            split_name="train_eval",
            log_sampling=False,
            feature_engine=feature_engine,
            node_feature_generator=node_feature_generator,
        )
        _ = run_split(
            dataset=dataset,
            data=data,
            split_mask=dataset.val_mask,
            evaluator=evaluator,
            model=model,
            optimizer=None,
            criterion=criterion,
            device=device,
            eval_metric=eval_metric,
            max_events=max_events,
            temporal_sampling=temporal_sampling,
            temporal_stride=temporal_stride,
            temporal_ratio=temporal_ratio,
            temporal_seed=temporal_seed,
            grad_clip_norm=0.0,
            split_name="val_eval",
            log_sampling=False,
            feature_engine=feature_engine,
            node_feature_generator=node_feature_generator,
        )
        test_out = run_split(
            dataset=dataset,
            data=data,
            split_mask=dataset.test_mask,  # testing split
            evaluator=evaluator,
            model=model,
            optimizer=None,
            criterion=criterion,
            device=device,
            eval_metric=eval_metric,
            max_events=max_events,
            temporal_sampling=temporal_sampling,
            temporal_stride=temporal_stride,
            temporal_ratio=temporal_ratio,
            temporal_seed=temporal_seed,
            grad_clip_norm=0.0,
            split_name="test",
            log_sampling=True,
            feature_engine=feature_engine,
            node_feature_generator=node_feature_generator,
        )

    print(
        f"Test {eval_metric}: {_fmt_metric(test_out.metric_value)} | "
        f"AUROC: {_fmt_metric(test_out.auroc)} | AP: {_fmt_metric(test_out.ap)}"
    )

    row = {
            "dataset": dataset_name,
            "model": model_name,
            "feature_mode": feature_mode,
            "feature_dim": raw_msg_dim,
            "noise_std": noise_std,
            "noise_seed": noise_seed,
            "temporal_ema_alpha": temporal_ema_alpha,
            "pagerank_interval": pagerank_interval,
            "recency_tau": recency_tau,
            "embedding_dim": embedding_dim,
            "walk_length": walk_length,
            "walks_per_node": walks_per_node,
            "context_size": context_size,
            "node2vec_p": node2vec_p,
            "node2vec_q": node2vec_q,
            "node_feature_mode": node_feature_mode,
            "node_feature_dim": 0 if effective_node_feature_dim is None else effective_node_feature_dim,
            "node_feature_noise_std": node_feature_noise_std,
            "node_feature_noise_seed": node_feature_noise_seed,
            "base_lr": lr,
            "effective_lr": effective_lr,
            "grad_clip_norm": grad_clip_norm,
            "epochs": epochs,
            "max_events": max_events,
            "temporal_sampling": temporal_sampling,
            "temporal_stride": temporal_stride,
            "temporal_ratio": temporal_ratio,
            "temporal_seed": temporal_seed,
            "primary_metric": eval_metric,
            "test_primary": test_out.metric_value,
            "test_ndcg": test_out.ndcg,
            "test_auroc": test_out.auroc,
            "test_ap": test_out.ap,
            "test_processed_events": test_out.processed_events,
            "test_labeled_timestamps": test_out.labeled_timestamps,
            "test_empty_label_events": test_out.empty_label_events,
            "test_label_batches": test_out.label_batches,
            "final_train_primary": float("nan") if final_train_out is None else final_train_out.metric_value,
            "final_train_ndcg": float("nan") if final_train_out is None else final_train_out.ndcg,
            "final_train_auroc": float("nan") if final_train_out is None else final_train_out.auroc,
            "final_train_ap": float("nan") if final_train_out is None else final_train_out.ap,
            "final_val_primary": float("nan") if final_val_out is None else final_val_out.metric_value,
            "final_val_ndcg": float("nan") if final_val_out is None else final_val_out.ndcg,
            "final_val_auroc": float("nan") if final_val_out is None else final_val_out.auroc,
            "final_val_ap": float("nan") if final_val_out is None else final_val_out.ap,
        }

    if results_csv is not None:
        append_results_row(file_path=results_csv, row=row)

    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train temporal baselines and advanced models on TGB node classification datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=DEFAULT_DATASETS,
        help="Datasets to train sequentially.",
    )
    parser.add_argument(
        "--root", type=str, default="datasets", help="Dataset root path."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tgn"],
        choices=DEFAULT_MODELS,
        help="Models to train sequentially.",
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Training epochs per dataset."
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--memory-dim", type=int, default=128, help="TGN memory embedding dimension."
    )
    parser.add_argument(
        "--time-dim", type=int, default=32, help="TGN time encoding dimension."
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional cap on events per split to speed up quick tests.",
    )
    parser.add_argument(
        "--tgat-heads",
        type=int,
        default=4,
        help="Attention heads for TGAT model.",
    )
    parser.add_argument(
        "--temporal-sampling",
        type=str,
        default="none",
        choices=["none", "stride", "uniform"],
        help="Temporal subsampling strategy.",
    )
    parser.add_argument(
        "--temporal-stride",
        type=int,
        default=2,
        help="Stride k when --temporal-sampling=stride.",
    )
    parser.add_argument(
        "--temporal-ratio",
        type=float,
        default=0.5,
        help="Keep ratio when --temporal-sampling=uniform.",
    )
    parser.add_argument(
        "--temporal-seed",
        type=int,
        default=42,
        help="Random seed for temporal sampling.",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="full",
        choices=[
            "full",
            "gaussian_noise",
            "temporal_heuristics",
            "snapshot_pagerank",
            "recency_pagerank",
            "snapshot_node2vec",
            "snapshot_deepwalk",
            "recency_node2vec",
        ],
        help="Feature source for event messages.",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=128,
        help="Synthetic feature dimensionality for non-full feature modes.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=64,
        help="Base embedding dimension for snapshot node-walk feature modes.",
    )
    parser.add_argument(
        "--walk-length",
        type=int,
        default=20,
        help="Random-walk length for snapshot_node2vec/snapshot_deepwalk/recency_node2vec.",
    )
    parser.add_argument(
        "--walks-per-node",
        type=int,
        default=3,
        help="Number of random walks started per active node.",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=5,
        help="Context window used for walk co-occurrence embeddings.",
    )
    parser.add_argument(
        "--node2vec-p",
        type=float,
        default=1.0,
        help="Node2Vec return parameter p for snapshot_node2vec/recency_node2vec.",
    )
    parser.add_argument(
        "--node2vec-q",
        type=float,
        default=1.0,
        help="Node2Vec in-out parameter q for snapshot_node2vec/recency_node2vec.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=1.0,
        help="Standard deviation for Gaussian noise when --feature-mode=gaussian_noise.",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=12345,
        help="Seed used to deterministically generate Gaussian noise features.",
    )
    parser.add_argument(
        "--node-feature-mode",
        type=str,
        default="none",
        choices=[
            "none",
            "gaussian_noise",
            "temporal_heuristics",
            "snapshot_pagerank",
            "recency_pagerank",
            "snapshot_node2vec",
            "snapshot_deepwalk",
            "recency_node2vec",
        ],
        help="Optional node initialization feature source; 'none' disables node-feature injection.",
    )
    parser.add_argument(
        "--node-feature-dim",
        type=int,
        default=64,
        help="Dimensionality of injected node initialization features.",
    )
    parser.add_argument(
        "--node-feature-noise-std",
        type=float,
        default=1.0,
        help="Standard deviation for node features when --node-feature-mode=gaussian_noise.",
    )
    parser.add_argument(
        "--node-feature-noise-seed",
        type=int,
        default=54321,
        help="Seed used to deterministically generate node Gaussian features.",
    )
    parser.add_argument(
        "--temporal-ema-alpha",
        type=float,
        default=0.1,
        help="EMA alpha for temporal heuristic features.",
    )
    parser.add_argument(
        "--pagerank-interval",
        type=int,
        default=100,
        help="Recompute interval (in events) for causal PageRank in temporal heuristics.",
    )
    parser.add_argument(
        "--recency-tau",
        type=float,
        default=1000.0,
        help="Exponential decay timescale for recency-weighted PageRank mode.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Max norm for gradient clipping (<=0 disables clipping).",
    )
    parser.add_argument(
        "--tgat-lr-mult",
        type=float,
        default=0.1,
        help="Multiplier applied to --lr for TGAT only.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("results/baseline_results.csv"),
        help="CSV file where per-run summary rows are appended.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device to use.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    for dataset_name in args.datasets:
        for model_name in args.models:
            train_dataset(
                dataset_name=dataset_name,
                model_name=model_name,
                root=args.root,
                epochs=args.epochs,
                lr=args.lr,
                memory_dim=args.memory_dim,
                time_dim=args.time_dim,
                max_events=args.max_events,
                device=device,
                tgat_heads=args.tgat_heads,
                temporal_sampling=args.temporal_sampling,
                temporal_stride=args.temporal_stride,
                temporal_ratio=args.temporal_ratio,
                temporal_seed=args.temporal_seed,
                feature_mode=args.feature_mode,
                feature_dim=args.feature_dim,
                noise_std=args.noise_std,
                noise_seed=args.noise_seed,
                temporal_ema_alpha=args.temporal_ema_alpha,
                pagerank_interval=args.pagerank_interval,
                recency_tau=args.recency_tau,
                embedding_dim=args.embedding_dim,
                walk_length=args.walk_length,
                walks_per_node=args.walks_per_node,
                context_size=args.context_size,
                node2vec_p=args.node2vec_p,
                node2vec_q=args.node2vec_q,
                node_feature_mode=args.node_feature_mode,
                node_feature_dim=args.node_feature_dim,
                node_feature_noise_std=args.node_feature_noise_std,
                node_feature_noise_seed=args.node_feature_noise_seed,
                grad_clip_norm=args.grad_clip_norm,
                tgat_lr_mult=args.tgat_lr_mult,
                results_csv=args.results_csv,
            )


if __name__ == "__main__":
    main()
