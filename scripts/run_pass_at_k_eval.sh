#!/bin/bash
# PASS@K Evaluation Script
# This script runs the complete PASS@K evaluation pipeline

set -e

# Configuration
VLLM_URL="${VLLM_URL:-http://localhost:8000/v1}"
KERNELGYM_URL="${KERNELGYM_URL:-http://localhost:8002}"
DATA_PATH="${DATA_PATH:-data/kernelbench_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/pass_at_k_$(date +%Y%m%d_%H%M%S)}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-10}"
MAX_TURNS="${MAX_TURNS:-3}"
K_VALUES="${K_VALUES:-1,5,10}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LIMIT_PROBLEMS="${LIMIT_PROBLEMS:-}"
RESUME="${RESUME:-0}"
TRAIN_ID="${TRAIN_ID:-}"

echo "=========================================="
echo "PASS@K Evaluation for KernelGym"
echo "=========================================="
echo "VLLM URL: $VLLM_URL"
echo "KernelGym URL: $KERNELGYM_URL"
echo "Data Path: $DATA_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "Num Rollouts: $NUM_ROLLOUTS"
echo "Max Turns: $MAX_TURNS"
echo "K Values: $K_VALUES"
echo "Num Workers: $NUM_WORKERS"
echo "Resume: $RESUME"
echo "Train ID: ${TRAIN_ID:-<auto>}"
echo "=========================================="

# Prepare data if needed
if [ ! -f "$DATA_PATH" ]; then
    echo "Data file not found. Preparing KernelBench data..."
    python -m examples.kernelgym.prepare_kernelbench_data
fi

# Run evaluation
echo "Starting PASS@K evaluation..."
CMD="python scripts/eval_pass_at_k.py \
    --vllm-url $VLLM_URL \
    --kernelgym-url $KERNELGYM_URL \
    --data-path $DATA_PATH \
    --output-dir $OUTPUT_DIR \
    --num-rollouts $NUM_ROLLOUTS \
    --max-turns $MAX_TURNS \
    --k-values $K_VALUES \
    --num-workers $NUM_WORKERS"

if [ -n "$LIMIT_PROBLEMS" ]; then
    CMD="$CMD --limit-problems $LIMIT_PROBLEMS"
fi

if [ -n "$TRAIN_ID" ]; then
    CMD="$CMD --train-id $TRAIN_ID"
fi

if [ "$RESUME" = "1" ] || [ "$RESUME" = "true" ]; then
    CMD="$CMD --resume"
fi

$CMD

# Run analysis
echo "Running analysis..."
python scripts/analyze_pass_at_k.py \
    --db-path $OUTPUT_DIR/interactions.db \
    --output-dir $OUTPUT_DIR/analysis \
    --k-values $K_VALUES

echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
