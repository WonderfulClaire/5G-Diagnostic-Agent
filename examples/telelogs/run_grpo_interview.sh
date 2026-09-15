#!/usr/bin/env bash
# Small GRPO teaching run. Validate backend/LoRA support on the GPU host first.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"
: "${DATA_DIR:?Set DATA_DIR to a train-derived training/development split directory}"
: "${MODEL_PATH:?Set MODEL_PATH to the checked model path on the GPU host}"
: "${CUDA_VISIBLE_DEVICES:?Select a GPU after checking live occupancy}"
test -f "$DATA_DIR/train.parquet"
test -f "$DATA_DIR/dev.parquet"

export EXP_NAME="${EXP_NAME:-telelogs_grpo_interview_$(date -u +%Y%m%dT%H%M%SZ)}"
export TRAIN_BATCH_SIZE=4
export PPO_MINI_BATCH_SIZE=4
export PPO_MICRO_BATCH_SIZE=1
export N_GPUS_PER_NODE=1
export MAX_PROMPT_LENGTH=3072
export MAX_RESPONSE_LENGTH=512
export MAX_STEPS=6
export SAVE_FREQ=10
export TEST_FREQ=10
export TOTAL_EPOCHS=1

# The base launcher uses GAE/n=1; explicit final overrides switch to GRPO/n=4.
# This backend chooses whether to instantiate a critic from adv_estimator.
# Supply --cfg job for a configuration-only check before actual training.
exec bash examples/telelogs/run_steppo.sh \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=16 \
    data.val_files="$DATA_DIR/dev.parquet" \
    trainer.total_training_steps=20 \
    trainer.resume_mode=disable \
    trainer.validation_data_dir="$PROJECT_DIR/runs/$EXP_NAME/validation" \
    trainer.default_local_dir="$PROJECT_DIR/runs/$EXP_NAME/checkpoints" \
    "$@"
