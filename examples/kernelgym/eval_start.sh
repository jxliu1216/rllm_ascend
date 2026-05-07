python scripts/eval_pass_at_k.py \
    --vllm-url http://localhost:8120/v1 \
    --model-name eval_model \
    --kernelgym-url http://localhost:8002 \
    --data-path data/kernelbench_train.jsonl \
    --output-dir results/pass_at_k \
    --num-rollouts 10 \
    --max-turns 5 \
    --num-workers 72 \
    --k-values 1,5,10