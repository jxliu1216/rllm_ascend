#!/bin/bash

pkill -9 python
pkill -9 torchrun
set -euo pipefail
set -x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $ROOT_DIR/common_env.sh

/workspace/kernelGym-npu-main/start_all_with_monitor.sh --env-file /workspace/kernelGym-npu-main/.env_p8335_n1516

vllm serve /home/g00841271/cszhou_sft_weight/global_step_50 \
  --dtype bfloat16 \
  --served-model-name eval_model \
  --max_model_len 98304 \
  --max_num_seqs 256 \
  --enable_prefix_caching \
  --max_num_batched_tokens 16384 \
  --enable_chunked_prefill \
  --gpu_memory_utilization 0.8 \
  --seed 0 \
  --api-server-count 8 \
  --scheduling_policy fcfs \
  --override_generation_config '{"temperature": 1.0, "top_k": -1, "top_p": 1.0, "repetition_penalty": 1.0}' \
  --tensor-parallel-size 1 \
  --data-parallel-size 12 \
  --host 0.0.0.0 \
  --port 8120
