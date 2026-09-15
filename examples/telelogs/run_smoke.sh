#!/usr/bin/env bash
set -euo pipefail

export EXP_NAME="${EXP_NAME:-telelogs_smoke}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-3072}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
export MAX_STEPS="${MAX_STEPS:-6}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export TEST_FREQ="${TEST_FREQ:-1}"

exec bash "$(dirname "$0")/run_steppo.sh" "$@"
