#!/usr/bin/env zsh
set -euo pipefail

cd /Users/williserdman/Documents/school/spring2026/graph_ml/featureless_temporal_benchmark

# Optional: run unit tests first
uv run python -m unittest -v tests/test_masking_and_labels.py tests/test_synthetic_features.py

RESULTS_CSV=results/full_tests_20k.csv
LOG_FILE=results/full_tests_20k.log

# Set INCLUDE_SLOW_WALK_MODES=1 to also run snapshot_node2vec/snapshot_deepwalk.
INCLUDE_SLOW_WALK_MODES=${INCLUDE_SLOW_WALK_MODES:-0}

# Snapshot GAE defaults tuned for 20k capped runs.
GAE_REFRESH_INTERVAL=${GAE_REFRESH_INTERVAL:-512}
GAE_STEPS=${GAE_STEPS:-8}
GAE_LR=${GAE_LR:-0.05}
GAE_MAX_EDGES=${GAE_MAX_EDGES:-50000}
GAE_BATCH_SIZE=${GAE_BATCH_SIZE:-4096}

DATASETS=(tgbn-trade tgbn-genre)
MODELS=(tgn tgat dyrep evolvegcn sage)

EDGE_MODES=(full unweighted_ones gaussian_noise temporal_delta)
NODE_MODES=(none gaussian_noise snapshot_pagerank snapshot_gae)
if [[ "$INCLUDE_SLOW_WALK_MODES" == "1" ]]; then
  NODE_MODES+=(snapshot_node2vec snapshot_deepwalk)
fi

: > "$LOG_FILE"

# 1) Edge-message ablation: vary edge mode, fix node init to none
for model in "${MODELS[@]}"; do
  for edge_mode in "${EDGE_MODES[@]}"; do
    echo "RUN edge ablation | model=$model | edge=$edge_mode | node=none" | tee -a "$LOG_FILE"
    uv run python main.py \
      --datasets "${DATASETS[@]}" \
      --models "$model" \
      --epochs 5 \
      --device cpu \
      --max-events 20000 \
      --temporal-sampling stride \
      --temporal-stride 20 \
      --feature-mode "$edge_mode" \
      --node-feature-mode none \
      --results-csv "$RESULTS_CSV" \
      --continue-on-error \
      >> "$LOG_FILE" 2>&1
  done
done

# 2) Node-init ablation: vary node mode, fix edge mode to full
for model in "${MODELS[@]}"; do
  for node_mode in "${NODE_MODES[@]}"; do
    echo "RUN node ablation | model=$model | edge=full | node=$node_mode" | tee -a "$LOG_FILE"
    uv run python main.py \
      --datasets "${DATASETS[@]}" \
      --models "$model" \
      --epochs 5 \
      --device cpu \
      --max-events 20000 \
      --temporal-sampling stride \
      --temporal-stride 20 \
      --feature-mode full \
      --node-feature-mode "$node_mode" \
      --gae-refresh-interval "$GAE_REFRESH_INTERVAL" \
      --gae-steps "$GAE_STEPS" \
      --gae-lr "$GAE_LR" \
      --gae-max-edges "$GAE_MAX_EDGES" \
      --gae-batch-size "$GAE_BATCH_SIZE" \
      --results-csv "$RESULTS_CSV" \
      --continue-on-error \
      >> "$LOG_FILE" 2>&1
  done
done

# 3) Plot summaries
uv run python plot_results.py \
  --results-csv "$RESULTS_CSV" \
  --ablation-kind edge \
  --fixed-node-feature-mode none \
  --metric all \
  --max-events 20000 \
  --datasets tgbn-trade tgbn-genre \
  --output results/edge_ablation_all_20k.png

uv run python plot_results.py \
  --results-csv "$RESULTS_CSV" \
  --ablation-kind node \
  --fixed-feature-mode full \
  --metric all \
  --max-events 20000 \
  --datasets tgbn-trade tgbn-genre \
  --output results/node_ablation_all_20k.png

echo "Done. Results: $RESULTS_CSV"
echo "Log: $LOG_FILE"
