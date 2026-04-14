# Featureless Temporal Benchmark

Temporal node-property benchmark runner for TGB node datasets.

This repo now keeps the two recovery pathways strictly separate:

- Edge-message pathway: controlled by `--feature-mode`
- Node-initialization pathway: controlled by `--node-feature-mode`

## Datasets

Supported TGB node datasets:
- `tgbn-trade`
- `tgbn-genre`
- `tgbn-reddit`
- `tgbn-token`

These datasets do not provide dense initial node feature matrices in this pipeline, so all node identity signals are generated causally inside the benchmark when requested.

## Models

Supported models:
- `tgn`
- `tgat`
- `dyrep`
- `evolvegcn`
- `sage`

These are benchmarked temporal baselines with a shared temporal-memory backbone and model-specific readout heads.

## Edge-Message Pathway

Use `--feature-mode` for the event message tensor that is passed into the model at each interaction.

Supported values:
- `full`: original TGB edge message
- `unweighted_ones`: constant all-ones tensor of shape `1 x d`
- `gaussian_noise`: seeded random noise tensor of shape `1 x d`
- `temporal_delta`: projected time since the same source-destination pair last interacted

Interpretation:
- `full` measures the fully featured upper baseline.
- `unweighted_ones` tests whether the model only needs to know that an event occurred.
- `gaussian_noise` is the pure starvation baseline.
- `temporal_delta` tests a very small timing-only signal.

## Node-Initialization Pathway

Use `--node-feature-mode` to inject causal node initialization features before event processing.

Supported values:
- `none`
- `gaussian_noise`
- `snapshot_pagerank`
- `snapshot_node2vec`
- `snapshot_deepwalk`

Interpretation:
- `gaussian_noise` is the naive node-init baseline.
- `snapshot_pagerank` injects a causal PageRank identity signal.
- `snapshot_node2vec` injects causal walk-based structural features.
- `snapshot_deepwalk` injects causal walk-based structural features with uniform transitions.

## Metrics

Primary benchmark metric:
- `ndcg` from the TGB evaluator

Supplemental diagnostics:
- `auroc`
- `ap`

`auroc` and `ap` are computed on binarized relevance labels (`label > 0`) and may be `NaN` in sparse windows.

## Installation

```bash
uv sync
```

For editable install:

```bash
pip install -e .
```

## Quick Start

```bash
uv run python main.py \
  --datasets tgbn-trade \
  --models tgn \
  --epochs 1 \
  --max-events 5000 \
  --device cpu
```

## Recommended Ablation Protocol

### Edge-message ablations

Fully featured baseline:

```bash
uv run python main.py \
  --datasets tgbn-trade tgbn-genre \
  --models tgn tgat dyrep evolvegcn sage \
  --epochs 1 \
  --max-events 50000 \
  --feature-mode full \
  --results-csv results/edge_ablation.csv \
  --device cpu
```

Interaction-only baseline:

```bash
uv run python main.py \
  --datasets tgbn-trade tgbn-genre \
  --models tgn tgat dyrep evolvegcn sage \
  --epochs 1 \
  --max-events 50000 \
  --feature-mode unweighted_ones \
  --feature-dim 128 \
  --results-csv results/edge_ablation.csv \
  --device cpu
```

Noise baseline:

```bash
uv run python main.py \
  --datasets tgbn-trade tgbn-genre \
  --models tgn tgat dyrep evolvegcn sage \
  --epochs 1 \
  --max-events 50000 \
  --feature-mode gaussian_noise \
  --feature-dim 128 \
  --noise-std 1.0 \
  --noise-seed 12345 \
  --results-csv results/edge_ablation.csv \
  --device cpu
```

Timing-only baseline:

```bash
uv run python main.py \
  --datasets tgbn-trade tgbn-genre \
  --models tgn tgat dyrep evolvegcn sage \
  --epochs 1 \
  --max-events 50000 \
  --feature-mode temporal_delta \
  --feature-dim 128 \
  --results-csv results/edge_ablation.csv \
  --device cpu
```

### Node-initialization ablations

```bash
uv run python main.py \
  --datasets tgbn-trade tgbn-genre \
  --models tgn tgat dyrep evolvegcn sage \
  --epochs 1 \
  --max-events 50000 \
  --feature-mode full \
  --node-feature-mode snapshot_pagerank \
  --node-feature-dim 64 \
  --results-csv results/node_ablation.csv \
  --device cpu
```

Use `--node-feature-mode gaussian_noise`, `snapshot_node2vec`, or `snapshot_deepwalk` to test the other node-init baselines while keeping the edge messages fixed.

## Developer API

```python
from featureless_temporal_benchmark import inspect_dataset, run_experiment, run_recovery_suite

summary = inspect_dataset("tgbn-trade")
print(summary["num_events"], summary["msg_dim"], summary["eval_metric"])

single = run_experiment(
    dataset="tgbn-trade",
    model="tgn",
    feature_mode="temporal_delta",
    node_feature_mode="snapshot_pagerank",
    feature_dim=128,
    node_feature_dim=64,
    epochs=1,
    max_events=20000,
    results_csv="results/api_runs.csv",
)
print(single["test_ndcg"], single["test_ap"], single["test_auroc"])

suite = run_recovery_suite(
    dataset="tgbn-trade",
    models=["tgn", "tgat", "dyrep", "evolvegcn", "sage"],
    feature_modes=["full", "unweighted_ones", "gaussian_noise", "temporal_delta"],
    node_feature_modes=["none", "gaussian_noise", "snapshot_pagerank", "snapshot_node2vec", "snapshot_deepwalk"],
    epochs=1,
    max_events=20000,
    results_csv="results/api_suite.csv",
)
print(len(suite))
```

## Notes

- Small `--max-events` values can produce zero train label batches on sparse windows, especially for `tgbn-trade`.
- TGAT uses a reduced effective learning rate by default via `--tgat-lr-mult`.
- Gradient clipping is enabled by default for stability.

## Tests

```bash
uv run python -m unittest -v tests/test_masking_and_labels.py
uv run python -m unittest -v tests/test_synthetic_features.py
```
