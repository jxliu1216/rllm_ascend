#!/bin/bash

pkill -9 python
pkill -9 torchrun
set -euo pipefail
set -x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $ROOT_DIR/common_env.sh

/home/g00841271/kernelGym-npu-main/start_all_with_monitor.sh --env-file /home/g00841271/kernelGym-npu-main/.env_p8335_n1516

vllm serve /home/docker/cszhou0506_global_step_50 \
  --dtype bfloat16 \
  --served-model-name eval_model \
  --max_model_len 98304 \
  --max_num_seqs 1024 \
  --enable_chunked_prefill \
  --max_num_batched_tokens 8192 \
  --enable_prefix_caching \
  --gpu_memory_utilization 0.8 \
  --seed 0 \
  --override_generation_config '{"temperature": 1.0, "top_k": -1, "top_p": 1.0, "repetition_penalty": 1.0, "max_new_tokens": 30000}' \
  --scheduling_policy fcfs \
  --tensor-parallel-size 2 \
  --data-parallel-size 6 \
  --host 0.0.0.0 \
  --port 8120