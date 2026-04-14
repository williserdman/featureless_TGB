#!/bin/bash
#SBATCH --job-name=tgb_recovery_matrix
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/recovery_%A_%a.log
#SBATCH --error=slurm_logs/recovery_%A_%a.err
#SBATCH --array=0-9

set -euo pipefail

# -------- User-configurable environment (override with sbatch --export=...) --------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}}"
DATASET_ROOT="${DATASET_ROOT:-datasets}"
COPY_DATA_TO_TMPDIR="${COPY_DATA_TO_TMPDIR:-0}"          # 1 to rsync dataset root to $TMPDIR
DATASET_SOURCE="${DATASET_SOURCE:-}"                     # required if COPY_DATA_TO_TMPDIR=1

EPOCHS="${EPOCHS:-1}"
MAX_EVENTS="${MAX_EVENTS:-50000}"
DEVICE="${DEVICE:-cuda}"
FEATURE_DIM="${FEATURE_DIM:-128}"
NODE_FEATURE_DIM="${NODE_FEATURE_DIM:-64}"

TEMPORAL_EMA_ALPHA="${TEMPORAL_EMA_ALPHA:-0.1}"
PAGERANK_INTERVAL="${PAGERANK_INTERVAL:-100}"
RECENCY_TAU="${RECENCY_TAU:-1000.0}"
EMBEDDING_DIM="${EMBEDDING_DIM:-64}"
WALK_LENGTH="${WALK_LENGTH:-20}"
WALKS_PER_NODE="${WALKS_PER_NODE:-3}"
CONTEXT_SIZE="${CONTEXT_SIZE:-5}"
NODE2VEC_P="${NODE2VEC_P:-1.0}"
NODE2VEC_Q="${NODE2VEC_Q:-1.0}"

SCRATCH_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
RESULTS_DIR="${RESULTS_DIR:-${SCRATCH_ROOT}/featureless_tgb_results}"
RESULTS_CSV="${RESULTS_CSV:-${RESULTS_DIR}/recovery_matrix_${SLURM_JOB_ID}.csv}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-featureless-tgb-313}"

if command -v conda >/dev/null 2>&1; then
	eval "$(conda shell.bash hook)"
	conda activate "$CONDA_ENV_NAME"
fi

cd "$PROJECT_DIR"
mkdir -p "$RESULTS_DIR"

if [[ "$COPY_DATA_TO_TMPDIR" == "1" ]]; then
	if [[ -z "$DATASET_SOURCE" ]]; then
		echo "ERROR: DATASET_SOURCE must be set when COPY_DATA_TO_TMPDIR=1"
		exit 1
	fi
	if [[ -z "${TMPDIR:-}" ]]; then
		echo "ERROR: TMPDIR is not set by scheduler"
		exit 1
	fi

	echo "Copying dataset root from $DATASET_SOURCE to $TMPDIR/datasets_local"
	mkdir -p "$TMPDIR/datasets_local"
	rsync -a "$DATASET_SOURCE"/ "$TMPDIR/datasets_local"/
	DATASET_ROOT="$TMPDIR/datasets_local"
fi

DATASETS=("tgbn-trade" "tgbn-genre")
MODELS=("tgn" "tgat" "dyrep" "evolvegcn" "sage")

EDGE_MODES=(
	"full"
	"gaussian_noise"
	"temporal_heuristics"
	"snapshot_pagerank"
	"recency_pagerank"
	"snapshot_node2vec"
	"snapshot_deepwalk"
	"recency_node2vec"
)

NODE_MODES=(
	"none"
	"gaussian_noise"
	"temporal_heuristics"
	"snapshot_pagerank"
	"recency_pagerank"
	"snapshot_node2vec"
	"snapshot_deepwalk"
	"recency_node2vec"
)

N_DATASETS=${#DATASETS[@]}
N_MODELS=${#MODELS[@]}
TOTAL=$((N_DATASETS * N_MODELS))

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= TOTAL )); then
	echo "ERROR: SLURM_ARRAY_TASK_ID=$TASK_ID out of range [0, $((TOTAL - 1))]"
	exit 1
fi

DATASET_INDEX=$((TASK_ID / N_MODELS))
MODEL_INDEX=$((TASK_ID % N_MODELS))
DATASET_NAME="${DATASETS[$DATASET_INDEX]}"
MODEL_NAME="${MODELS[$MODEL_INDEX]}"

echo "============================================================"
echo "Job ${SLURM_JOB_ID:-na} task ${TASK_ID}: dataset=${DATASET_NAME} model=${MODEL_NAME}"
echo "results_csv=${RESULTS_CSV}"
echo "dataset_root=${DATASET_ROOT}"
echo "============================================================"

run_main() {
	local feature_mode="$1"
	local node_feature_mode="$2"

	python "$PROJECT_DIR/main.py" \
		--datasets "$DATASET_NAME" \
		--models "$MODEL_NAME" \
		--root "$DATASET_ROOT" \
		--epochs "$EPOCHS" \
		--max-events "$MAX_EVENTS" \
		--device "$DEVICE" \
		--feature-mode "$feature_mode" \
		--feature-dim "$FEATURE_DIM" \
		--node-feature-mode "$node_feature_mode" \
		--node-feature-dim "$NODE_FEATURE_DIM" \
		--temporal-ema-alpha "$TEMPORAL_EMA_ALPHA" \
		--pagerank-interval "$PAGERANK_INTERVAL" \
		--recency-tau "$RECENCY_TAU" \
		--embedding-dim "$EMBEDDING_DIM" \
		--walk-length "$WALK_LENGTH" \
		--walks-per-node "$WALKS_PER_NODE" \
		--context-size "$CONTEXT_SIZE" \
		--node2vec-p "$NODE2VEC_P" \
		--node2vec-q "$NODE2VEC_Q" \
		--results-csv "$RESULTS_CSV"
}

echo "[Phase 1] Edge feature recovery (node features disabled)"
for edge_mode in "${EDGE_MODES[@]}"; do
	echo "Running edge_mode=${edge_mode}, node_feature_mode=none"
	run_main "$edge_mode" "none"
done

echo "[Phase 2] Node initialization recovery (edge features fixed to full)"
for node_mode in "${NODE_MODES[@]}"; do
	echo "Running feature_mode=full, node_feature_mode=${node_mode}"
	run_main "full" "$node_mode"
done

echo "Task complete for dataset=${DATASET_NAME} model=${MODEL_NAME}"
