#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_PATH:?Set the local checkpoint}"
: "${BASE_CASES:?Set the same 32 single-cause training rows}"
: "${CASES_PATH:?Set the immutable 32+8 flywheel release}"
: "${DEV_CASES:?Set the previously inspected development cases}"
: "${OUTPUT_ROOT:?Set a fresh output root}"
mkdir "$OUTPUT_ROOT"
python -m recipes.telelogs.evaluation.run_agent "$DEV_CASES" --local \
  --model "$MODEL_PATH" --max-steps 8 --output "$OUTPUT_ROOT/base-dev.jsonl"
python -m recipes.telelogs.training.supervised "$BASE_CASES" \
  --model "$MODEL_PATH" --output "$OUTPUT_ROOT/sft" \
  --steps 120 --seed 42 --lr 1e-4 --diagnosis-weight 1
python -m recipes.telelogs.evaluation.run_agent "$DEV_CASES" --local \
  --model "$MODEL_PATH" --adapter "$OUTPUT_ROOT/sft/adapter" \
  --max-steps 8 --output "$OUTPUT_ROOT/sft-dev.jsonl"
python -m recipes.telelogs.training.supervised "$CASES_PATH" \
  --model "$MODEL_PATH" --adapter "$OUTPUT_ROOT/sft/adapter" \
  --output "$OUTPUT_ROOT/flywheel" --steps 100 --seed 42 \
  --lr 2e-5 --replay-kl 0.2 --diagnosis-weight 1
python -m recipes.telelogs.evaluation.run_agent "$DEV_CASES" --local \
  --model "$MODEL_PATH" --adapter "$OUTPUT_ROOT/flywheel/adapter" \
  --max-steps 8 --output "$OUTPUT_ROOT/flywheel-dev.jsonl"
