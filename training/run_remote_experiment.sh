#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-hr-ft}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"
DATASET="${DATASET:-docs/ft-data/train.jsonl}"
MAX_STEPS="${MAX_STEPS:-10}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-smoke}"
RUN_DIR="$ROOT_DIR/outputs/experiments/$RUN_ID"
if [[ "$DATASET" = /* ]]; then
  DATASET_PATH="$DATASET"
else
  DATASET_PATH="$ROOT_DIR/$DATASET"
fi

mkdir -p "$RUN_DIR" "$ROOT_DIR/.cache/huggingface" "$ROOT_DIR/.cache/torch"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT_DIR/.cache/torch}"
export PYTHONPATH="$ROOT_DIR/training/src:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

cat > "$RUN_DIR/purpose.md" <<EOF
# FT experiment purpose

- Objective: Teach the HR assistant to classify user intent, ask for missing
  information, reject unsafe requests, and emit a validated JSON execution plan.
- Out of scope: Memorizing mutable company policies. Those remain in the RAG store.
- Base model: $MODEL
- Method: Supervised fine-tuning with LoRA through Unsloth.
- Training data: $DATASET_PATH (synthetic HR conversations).
- Held-out data: docs/ft-data/validation.jsonl and test.jsonl. These are not
  included in the training input and are reserved for model selection/evaluation.
- Dataset generator seed: 3407.
- Maximum optimizer steps for this run: $MAX_STEPS.
EOF

{
  echo "run_id=$RUN_ID"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "working_directory=$ROOT_DIR"
  echo "conda_environment=$CONDA_ENV"
  echo "model=$MODEL"
  echo "dataset=$DATASET_PATH"
  echo "max_steps=$MAX_STEPS"
} > "$RUN_DIR/run.env"

sha256sum \
  "$DATASET_PATH" \
  "$ROOT_DIR/docs/ft-data/validation.jsonl" \
  "$ROOT_DIR/docs/ft-data/test.jsonl" \
  "$ROOT_DIR/docs/ft-data/stats.json" > "$RUN_DIR/dataset_sha256.txt"
cp "$ROOT_DIR/docs/ft-data/stats.json" "$RUN_DIR/dataset_stats.json"
if [[ -f "$(dirname "$DATASET_PATH")/manifest.json" ]]; then
  cp "$(dirname "$DATASET_PATH")/manifest.json" "$RUN_DIR/dataset_manifest.json"
fi
nvidia-smi > "$RUN_DIR/nvidia-smi.txt"
python --version > "$RUN_DIR/python-version.txt" 2>&1
python -m pip freeze > "$RUN_DIR/pip-freeze.txt"

python "$ROOT_DIR/training/validate_dataset.py" \
  "$DATASET_PATH" > "$RUN_DIR/dataset-validation.log" 2>&1
validation_status=$?
if [[ $validation_status -ne 0 ]]; then
  echo "status=dataset_validation_failed" >> "$RUN_DIR/run.env"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN_DIR/run.env"
  exit "$validation_status"
fi

set +e
python "$ROOT_DIR/training/train_unsloth.py" \
  --model "$MODEL" \
  --dataset "$DATASET_PATH" \
  --output "$RUN_DIR/adapter" \
  --max-steps "$MAX_STEPS" 2>&1 | tee "$RUN_DIR/training.log"
training_status=${PIPESTATUS[0]}
set -e

echo "training_exit_code=$training_status" >> "$RUN_DIR/run.env"
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN_DIR/run.env"
if [[ $training_status -eq 0 ]]; then
  echo "status=completed" >> "$RUN_DIR/run.env"
else
  echo "status=failed" >> "$RUN_DIR/run.env"
fi
exit "$training_status"
