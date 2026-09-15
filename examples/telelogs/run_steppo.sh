#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="$PROJECT_DIR/recipes/telelogs/base.yaml"
DATA_DIR="${DATA_DIR:-$HOME/data/telelogs_agent_r1}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
EXP_NAME="${EXP_NAME:-telelogs_steppo_qwen3_4b}"

python3 -m agent_r1.trainer.main_agent_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/dev.parquet" \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-64}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-4096}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-1536}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-6}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-32}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE:-4}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD:-True}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD:-True}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE:-1}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.55}" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH:-4096}" \
    actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH:-1536}" \
    actor_rollout_ref.rollout.agent.agent_flow_config_path="$CONFIG_PATH" \
    actor_rollout_ref.rollout.agent.default_agent_flow=telelogs_rca \
    actor_rollout_ref.rollout.agent.max_steps="${MAX_STEPS:-8}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE:-1}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.path="$MODEL_PATH" \
    critic.optim.lr="${CRITIC_LR:-1e-5}" \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE:-4}" \
    critic.model.fsdp_config.param_offload="${CRITIC_PARAM_OFFLOAD:-True}" \
    critic.model.fsdp_config.optimizer_offload="${CRITIC_OPTIMIZER_OFFLOAD:-True}" \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    custom_reward_function.path=recipes/telelogs/reward_fn.py \
    custom_reward_function.name=compute_score \
    trainer.project_name=agent_r1_telelogs \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-1}" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ:-5}" \
    trainer.test_freq="${TEST_FREQ:-1}" \
    trainer.total_epochs="${TOTAL_EPOCHS:-3}" "$@"
