#!/usr/bin/env bash
# Run once per diagnosis weight on a separately assigned free GPU.
set -euo pipefail
: "${MODEL_PATH:?Set the local base checkpoint}"
: "${INITIAL_ADAPTER:?Set the SFT120 adapter}"
: "${CASES_PATH:?Set the immutable flywheel training release}"
: "${DEV_CASES:?Set the previously inspected development cases}"
: "${OUTPUT_ROOT:?Set a fresh output root}"
WEIGHT="${1:?usage: run_retention_ablation.sh <diagnosis-weight>}"
mkdir "$OUTPUT_ROOT"
python -m recipes.telelogs.training.supervised "$CASES_PATH" \
  --model "$MODEL_PATH" --adapter "$INITIAL_ADAPTER" \
  --output "$OUTPUT_ROOT/train" --steps 100 --seed 42 \
  --lr 2e-5 --replay-kl 0.2 --diagnosis-weight "$WEIGHT"
python -m recipes.telelogs.evaluation.run_agent "$DEV_CASES" \
  --local --model "$MODEL_PATH" --adapter "$OUTPUT_ROOT/train/adapter" \
  --max-steps 8 --output "$OUTPUT_ROOT/dev.jsonl"
