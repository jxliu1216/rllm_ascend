# PASS@K Evaluation for KernelGym

This directory contains scripts for evaluating model performance on KernelBench using PASS@K metrics.

## Overview

The evaluation system measures:
- **PASS@K Accuracy**: Probability of getting at least one correct solution within K attempts
- **Speedup Rate**: Performance improvement of optimized kernels over baseline
- **Multi-turn Feedback**: Iterative refinement with N rounds of feedback

## Files

- `eval_pass_at_k.py` - Main evaluation script
- `analyze_pass_at_k.py` - Analysis and visualization script
- `run_pass_at_k_eval.sh` - Convenience shell script

## Quick Start

### 1. Prepare Data

```bash
# Download and prepare KernelBench level1 data
python -m examples.kernelgym.prepare_kernelbench_data
```

### 2. Run Evaluation

```bash
# Using the shell script
./scripts/run_pass_at_k_eval.sh

# Or directly with Python
python scripts/eval_pass_at_k.py \
    --vllm-url http://localhost:8000/v1 \
    --kernelgym-url http://localhost:8002 \
    --data-path data/kernelbench_train.jsonl \
    --output-dir results/pass_at_k \
    --num-rollouts 10 \
    --max-turns 3 \
    --k-values 1,5,10
```

### 3. Analyze Results

```bash
python scripts/analyze_pass_at_k.py \
    --db-path results/pass_at_k/interactions.db \
    --output-dir results/pass_at_k/analysis
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_URL` | `http://localhost:8000/v1` | OpenAI-compatible LLM endpoint |
| `KERNELGYM_URL` | `http://localhost:8002` | KernelGym server URL |
| `DATA_PATH` | `data/kernelbench_train.jsonl` | Path to KernelBench data |
| `OUTPUT_DIR` | `results/pass_at_k_<timestamp>` | Output directory |
| `NUM_ROLLOUTS` | `10` | Number of rollouts per problem |
| `MAX_TURNS` | `3` | Maximum feedback iterations |
| `K_VALUES` | `1,5,10` | K values for Pass@K calculation |
| `NUM_WORKERS` | `4` | Parallel workers |
| `LIMIT_PROBLEMS` | (empty) | Limit problems for testing |

### Command Line Arguments

```
--vllm-url          OpenAI-compatible LLM base URL
--vllm-api-key      API key for LLM (default: EMPTY)
--model-name        Model name (default: default)
--kernelgym-url     KernelGym server URL
--data-path         Path to KernelBench data
--hf-split          HuggingFace split name (default: level_1)
--output-dir        Output directory
--num-rollouts      Number of rollouts per problem
--max-turns         Maximum turns per rollout
--k-values          Comma-separated K values
--temperature       Sampling temperature (default: 0.6)
--max-tokens        Maximum tokens per response (default: 4096)
--task-timeout      Task timeout in seconds (default: 300)
--num-workers       Number of parallel workers
--limit-problems    Limit number of problems for testing
```

## Output Structure

```
results/pass_at_k/
├── interactions.db          # SQLite database with all interaction data
├── summary.json             # Summary statistics
├── pass_at_k_analysis.png   # Overview plots
├── overall_pass_at_k.png    # Overall Pass@K bar chart
└── analysis/
    ├── analysis_results.json    # Detailed analysis results
    ├── analysis_report.txt      # Text report
    ├── analysis_overview.png    # Overview visualization
    ├── problem_rankings.png     # Top problems by metrics
    └── pass_at_k_boxplot.png    # Pass@K distribution
```

## Database Schema

### rollouts table
```sql
CREATE TABLE rollouts (
    id INTEGER PRIMARY KEY,
    rollout_id TEXT,
    problem_id TEXT,
    timestamp TEXT,
    best_speedup REAL,
    final_correct INTEGER,
    final_compiled INTEGER,
    total_reward REAL
);
```

### turns table
```sql
CREATE TABLE turns (
    id INTEGER PRIMARY KEY,
    rollout_id TEXT,
    turn INTEGER,
    compiled INTEGER,
    correctness INTEGER,
    speedup REAL,
    reward REAL,
    error TEXT,
    kernel_code TEXT,
    response TEXT,
    server_result TEXT
);
```

### problem_stats table
```sql
CREATE TABLE problem_stats (
    id INTEGER PRIMARY KEY,
    problem_id TEXT UNIQUE,
    num_rollouts INTEGER,
    num_passed INTEGER,
    pass_at_1 REAL,
    pass_at_5 REAL,
    pass_at_10 REAL,
    best_speedup REAL,
    avg_speedup REAL,
    avg_reward REAL
);
```

## Metrics

### PASS@K Calculation

PASS@K is calculated using the unbiased estimator:

```
pass@k = 1 - ∏(n - k + 1) / (n - c + 1) for i in range(k)
```

Where:
- `n` = total number of samples
- `c` = number of correct samples
- `k` = number of samples to consider

The concrete `k` values are controlled by `--k-values` in the evaluation and analysis scripts.

### Speedup Rate

Speedup rate is calculated as:
- For each problem, find the best speedup among all passing rollouts
- Report the distribution of best speedups across problems

### Reward Calculation

Rewards are calculated per turn:
- Compilation failure: -0.5
- Correctness failure: -0.3
- Speedup >= 3.0x: 1.0
- Speedup >= 2.0x: 0.8
- Speedup >= 1.5x: 0.6
- Speedup >= 1.2x: 0.4
- Speedup >= 1.0x: 0.2
- Speedup < 1.0x: -0.1

## Example Usage

### Quick Test (5 problems, 3 rollouts each)

```bash
python scripts/eval_pass_at_k.py \
    --num-rollouts 3 \
    --limit-problems 5 \
    --output-dir results/quick_test
```

### Full Evaluation (all problems, 10 rollouts)

```bash
python scripts/eval_pass_at_k.py \
    --num-rollouts 10 \
    --max-turns 5 \
    --k-values 1,5,10,20 \
    --output-dir results/full_eval
```

### Resume from Database

If you have an existing database, you can analyze it directly:

```bash
python scripts/analyze_pass_at_k.py \
    --db-path results/pass_at_k/interactions.db \
    --output-dir results/pass_at_k/analysis
```

## Prerequisites

1. Running vLLM server with the model to evaluate
2. Running KernelGym server
3. KernelBench data prepared

```bash
# Start vLLM server (example)
python -m vllm.entrypoints.openai.api_server \
    --model <model_path> \
    --port 8000

# Start KernelGym server
# (Refer to KernelGym documentation)
```

## Troubleshooting

### Connection Errors

If you see connection errors:
1. Verify vLLM server is running: `curl http://localhost:8000/v1/models`
2. Verify KernelGym server is running: `curl http://localhost:8002/status`

### Timeout Errors

Increase timeout values:
```bash
python scripts/eval_pass_at_k.py \
    --task-timeout 600 \
    --num-workers 2
```

### Memory Issues

Reduce parallelism:
```bash
python scripts/eval_pass_at_k.py \
    --num-workers 1 \
    --limit-problems 10
```
