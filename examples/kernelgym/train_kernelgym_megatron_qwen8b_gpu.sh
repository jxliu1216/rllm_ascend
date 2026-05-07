#!/bin/bash

#pkill -9 python
#pkill -9 torchrun
set -euo pipefail
set -x

# ── vLLM / PyTorch 环境变量 ───────────────────────────────────────────────────
export NCCL_ALGO=Ring,Tree
export FORCE_BUILD=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"

# export CUDA_LAUNCH_BLOCKING=1
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export VLLM_LOGGING_LEVEL=WARN

export RAY_DEBUG_POST_MORTEM=0

RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
export PYTHONPATH=$PYTHONPATH:$RLLM_DIR

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1

# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

MODEL_PATH=/home/docker/drkernel-8b-codestart

echo "正在重启 Ray 集群..."
ray stop --force
rm -rf /tmp/ray/*
sleep 2
ray start --head \
    --port 6379 \
    --dashboard-host 0.0.0.0 \
    --dashboard-port 8265 \
    --disable-usage-stats \
    --node-ip-address ${MASTER_ADDR}

ARGS=(
  # =========================
  # algorithm
  # =========================
  algorithm.adv_estimator=grpo
  algorithm.kl_ctrl.kl_coef=0.001

  # =========================
  # data
  # =========================
  data.train_batch_size=16
  data.val_batch_size=16
  data.max_prompt_length=24576      # 24K
  data.max_response_length=8192    # 18K

  # =========================
  # actor_rollout_ref - common
  # =========================
  actor_rollout_ref.hybrid_engine=True
  # actor_rollout_ref.model.use_shm=True
  actor_rollout_ref.model.path=$MODEL_PATH
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True

  # =========================
  # actor - optimization / PPO
  # =========================
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
  actor_rollout_ref.actor.ppo_mini_batch_size=8     # 实际 minibatch_size = minibatch_size * n_rollout // dp_size
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0.0
  actor_rollout_ref.actor.clip_ratio_low=0.2
  actor_rollout_ref.actor.clip_ratio_high=0.28

  # =========================
  # actor.megatron
  # =========================
  actor_rollout_ref.actor.strategy=megatron
  actor_rollout_ref.actor.megatron.use_mbridge=True
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=False

  actor_rollout_ref.actor.megatron.param_offload=True
  actor_rollout_ref.actor.megatron.grad_offload=True
  # actor_rollout_ref.actor.megatron.optimizer_offload=True

  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
  actor_rollout_ref.actor.megatron.context_parallel_size=1
  actor_rollout_ref.actor.megatron.expert_model_parallel_size=1
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
  +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size=1
  #+actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
  +actor_rollout_ref.actor.checkpoint.save_contents="['model']"

  # =========================
  # actor optimizer override
  # =========================
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
#   +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
  +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True

  # =========================
  # ref
  # =========================
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1
  actor_rollout_ref.ref.megatron.context_parallel_size=1
  actor_rollout_ref.ref.megatron.expert_model_parallel_size=1
  actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1
  actor_rollout_ref.ref.megatron.param_offload=True
  actor_rollout_ref.ref.megatron.use_mbridge=True
  actor_rollout_ref.ref.megatron.use_dist_checkpointing=False

  # =========================
  # rollout
  # =========================
  actor_rollout_ref.rollout.tensor_model_parallel_size=2
  actor_rollout_ref.rollout.calculate_log_probs=True        # 记录 训推的 log_prob 确认是否存在diff
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.enforce_eager=False
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7
  actor_rollout_ref.rollout.max_model_len=32768
  actor_rollout_ref.rollout.n=16
  actor_rollout_ref.rollout.val_kwargs.n=4
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  ++actor_rollout_ref.rollout.engine_kwargs.max_num_seqs=64
  ++actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096

  rllm.rejection_sample.enable=True

  # =========================
  # rllm
  # =========================
  rllm.mask_truncated_samples=False
  +rllm.agent.engine_args.n_parallel_agents=96
  rllm.agent.max_steps=3
  rllm.stepwise_advantage.enable=True
  rllm.stepwise_advantage.mode=broadcast #per_step

  # =========================
  # trainer
  # =========================
  trainer.critic_warmup=0
  trainer.logger=[console,wandb]
  trainer.project_name=rllm-agent
  trainer.experiment_name=kernelgym-qwen30b
  trainer.val_before_train=False
  trainer.n_gpus_per_node=4
  trainer.nnodes=1
  trainer.device=cuda
  trainer.save_freq=20
  trainer.test_freq=20
  trainer.default_hdfs_dir=null
  trainer.total_epochs=100

  # # =========================
  # # kernel
  # # =========================

  reward_model.max_turns=3
  reward_model.reference_backend=triton
  reward_model.server_url="http://127.0.0.1:8002"
  reward_model.reward_func_name=calculate_reward_weighted
  reward_model.init_correct_weight=0.5
  reward_model.init_performance_weight=0.5

  reward_model.speedup_reward_upper_bound=3.0
  reward_model.speedup_reward_lower_bound=0.0

  reward_model.reward_policy.penalties.penalty_score=0.0
  reward_model.reward_policy.penalties.compilation_fail=-0.5
  reward_model.reward_policy.penalties.correctness_fail=-0.3
  reward_model.reward_policy.penalties.perf_degrade=-0.1

  reward_model.timeout=120
  reward_model.max_retries=2
  reward_model.task_timeout=600
  reward_model.task_timeout_in_client=150
  reward_model.num_perf_trials=100
  reward_model.num_correct_trials=5
  reward_model.enable_profiling=true
  reward_model.verbose_errors=true

  reward_model.reward_weights.compilation=0.3
  reward_model.reward_weights.correctness=0.4
  reward_model.reward_weights.performance=0.3

  reward_model.coverage_reward.reward_type=time_coverage
  reward_model.coverage_reward.enable=false
  reward_model.coverage_reward.weight=0.25
)

# python3 -m examples.kernelgym.train_kernelgym "${ARGS[@]}"


#ray job submit --address="http://${MASTER_ADDR}:8265" \
#    -- \
/usr/bin/python3 -m examples.kernelgym.train_kernelgym "${ARGS[@]}" 2>&1| tee "gpu_training_$(date +%Y%m%d_%H%M%S).log"