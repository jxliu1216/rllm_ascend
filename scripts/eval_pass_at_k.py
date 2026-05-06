"""
PASS@K Evaluation Script for KernelGym.

Uses rllm's KernelGymEnv + KernelAgent for evaluation instead of
duplicating HTTP client, reward calculation, and message construction logic.

Usage:
    python scripts/eval_pass_at_k.py \
        --vllm-url http://localhost:8000/v1 \
        --kernelgym-url http://localhost:8002 \
        --output-dir results/pass_at_k \
        --num-rollouts 10 \
        --max-turns 3 \
        --k-values 1,5,10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from openai import OpenAI

from rllm.agents.kernelgym_agent import KernelAgent
from rllm.environments.kernelgym.kernelgym_env import KernelGymEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_pass_at_k")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    turn: int
    compiled: bool = False
    correctness: bool = False
    speedup: float = 0.0
    reward: float = 0.0
    error: str | None = None
    kernel_code: str | None = None
    response: str | None = None
    server_result: dict | None = None


@dataclass
class RolloutResult:
    rollout_id: str
    problem_id: str
    turns: list[TurnResult] = field(default_factory=list)
    best_speedup: float = 0.0
    best_turn: int = -1
    final_correct: bool = False
    final_compiled: bool = False
    total_reward: float = 0.0


@dataclass
class ProblemStats:
    problem_id: str
    num_rollouts: int = 0
    num_passed: int = 0
    pass_at_k: dict[int, float] = field(default_factory=dict)
    best_speedup: float = 0.0
    avg_speedup: float = 0.0
    avg_reward: float = 0.0


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_env_config(args: argparse.Namespace) -> OmegaConf:
    """Build KernelGymEnv config from CLI args."""
    return OmegaConf.create({
        "server_url": args.kernelgym_url,
        "timeout": args.task_timeout,
        "rate_limit": args.rate_limit,
        "acquire_timeout": args.acquire_timeout,
        "task_timeout": args.task_timeout,
        "task_timeout_in_client": args.task_timeout + 120,
        "max_retries": args.max_retries,
        "reward_func_name": args.reward_func,
        "init_correct_weight": args.init_correct_weight,
        "init_performance_weight": args.init_performance_weight,
        "speedup_eps": args.speedup_eps,
        "speedup_reward_upper_bound": args.speedup_reward_upper_bound,
        "speedup_reward_lower_bound": args.speedup_reward_lower_bound,
        "num_perf_trials": args.num_perf_trials,
        "num_correct_trials": args.num_correct_trials,
        "enable_profiling": args.enable_profiling,
        "verbose_errors": args.verbose_errors,
        "detect_decoy_kernel": args.detect_decoy_kernel,
        "reference_backend": args.backend,
        "max_turns": args.max_turns,
        "coverage_reward": {
            "enable": False,
            "weight": 0.0,
            "reward_type": "time_coverage",
        },
        "reward_policy": {
            "penalties": {
                "penalty_score": args.penalty_score,
                "compilation_fail": args.compilation_fail_penalty,
                "correctness_fail": args.correctness_fail_penalty,
                "perf_degrade": args.perf_degrade_penalty,
            },
        },
    })


# ---------------------------------------------------------------------------
# PASS@K metric
# ---------------------------------------------------------------------------

def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    if n - k < 0:
        return 0.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)) if n > c else 1.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_kernelbench_data(data_path: str, hf_split: str = "level_1") -> list[dict]:
    """Load KernelBench data from parquet, jsonl, or HuggingFace dataset."""
    tasks = []
    path = Path(data_path)

    if path.suffix in (".parquet", ".jsonl") or path.exists():
        import pandas as pd
        if path.suffix == ".jsonl":
            df = pd.read_json(data_path, lines=True)
        else:
            df = pd.read_parquet(data_path)

        for idx, row in df.iterrows():
            row_dict = row.to_dict()

            # Format A: KernelBench raw (code, name, level, problem_id)
            if "code" in row_dict and "name" in row_dict:
                level = row_dict.get("level", "")
                pid = row_dict.get("problem_id", idx)
                name = row_dict.get("name", "")
                tid = f"level{level}_{pid}_{name}" if level else f"task_{pid}"
                tasks.append({
                    "problem_id": tid,
                    "reference_code": row_dict["code"],
                    "entry_point": "Model",
                })
            # Format B: extra_info dict (DR.Kernel parquet)
            elif "extra_info" in row_dict:
                extra = row_dict.get("extra_info", {}) or {}
                if isinstance(extra, str):
                    extra = json.loads(extra)
                ref_code = extra.get("ground_truth", extra.get("task_code", ""))
                if ref_code:
                    entry = extra.get("entry_point", "Model")
                    tid = extra.get("uuid", extra.get("op_name", f"task_{idx}"))
                    tasks.append({
                        "problem_id": str(tid),
                        "reference_code": ref_code,
                        "entry_point": entry,
                    })
            # Format C: rllm JSONL (reference_code, problem_id, entry_point)
            elif "reference_code" in row_dict:
                tasks.append({
                    "problem_id": row_dict.get("problem_id", f"task_{idx}"),
                    "reference_code": row_dict["reference_code"],
                    "entry_point": row_dict.get("entry_point", "Model"),
                })
            # Format D: HuggingFace-style (code field only)
            elif "code" in row_dict:
                level = row_dict.get("level", "")
                pid = row_dict.get("problem_id", idx)
                name = row_dict.get("name", "")
                tasks.append({
                    "problem_id": f"level{level}_{pid}_{name}" if level else f"task_{pid}",
                    "reference_code": row_dict["code"],
                    "entry_point": "Model",
                })
    else:
        from datasets import load_dataset
        ds = load_dataset("ScalingIntelligence/KernelBench", split=hf_split)
        for idx, row in enumerate(ds):
            row_dict = dict(row)
            level = row_dict.get("level", "")
            pid = row_dict.get("problem_id", idx)
            name = row_dict.get("name", "")
            tasks.append({
                "problem_id": f"level{level}_{pid}_{name}" if level else f"task_{pid}",
                "reference_code": row_dict["code"],
                "entry_point": "Model",
            })

    logger.info("Loaded %d tasks from %s", len(tasks), data_path)
    return tasks


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

class InteractionDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rollouts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rollout_id TEXT NOT NULL,
                problem_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                best_speedup REAL,
                final_correct INTEGER,
                final_compiled INTEGER,
                total_reward REAL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rollout_id TEXT NOT NULL,
                turn INTEGER NOT NULL,
                compiled INTEGER,
                correctness INTEGER,
                speedup REAL,
                reward REAL,
                error TEXT,
                kernel_code TEXT,
                response TEXT,
                server_result TEXT,
                FOREIGN KEY (rollout_id) REFERENCES rollouts(rollout_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS problem_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem_id TEXT NOT NULL UNIQUE,
                num_rollouts INTEGER,
                num_passed INTEGER,
                pass_at_1 REAL,
                pass_at_5 REAL,
                pass_at_10 REAL,
                best_speedup REAL,
                avg_speedup REAL,
                avg_reward REAL
            )
        """)

        conn.commit()
        conn.close()

    def save_rollout(self, result: RolloutResult):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO rollouts (rollout_id, problem_id, timestamp, best_speedup, final_correct, final_compiled, total_reward)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            result.rollout_id,
            result.problem_id,
            datetime.now().isoformat(),
            result.best_speedup,
            int(result.final_correct),
            int(result.final_compiled),
            result.total_reward,
        ))

        for turn in result.turns:
            cursor.execute("""
                INSERT INTO turns (rollout_id, turn, compiled, correctness, speedup, reward, error, kernel_code, response, server_result)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.rollout_id,
                turn.turn,
                int(turn.compiled),
                int(turn.correctness),
                turn.speedup,
                turn.reward,
                turn.error,
                turn.kernel_code,
                turn.response,
                json.dumps(turn.server_result) if turn.server_result else None,
            ))

        conn.commit()
        conn.close()

    def save_problem_stats(self, stats: ProblemStats):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO problem_stats
            (problem_id, num_rollouts, num_passed, pass_at_1, pass_at_5, pass_at_10, best_speedup, avg_speedup, avg_reward)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            stats.problem_id,
            stats.num_rollouts,
            stats.num_passed,
            stats.pass_at_k.get(1, 0.0),
            stats.pass_at_k.get(5, 0.0),
            stats.pass_at_k.get(10, 0.0),
            stats.best_speedup,
            stats.avg_speedup,
            stats.avg_reward,
        ))

        conn.commit()
        conn.close()

    def get_all_rollouts(self) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM rollouts")
        columns = [desc[0] for desc in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return rows

    def get_problem_rollouts(self, problem_id: str) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM rollouts WHERE problem_id = ?", (problem_id,))
        columns = [desc[0] for desc in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return rows


# ---------------------------------------------------------------------------
# Core rollout logic (uses rllm KernelGymEnv + KernelAgent)
# ---------------------------------------------------------------------------

def run_single_rollout(
    problem_id: str,
    reference_code: str,
    entry_point: str,
    llm: OpenAI,
    env_config: OmegaConf,
    model_name: str,
    max_turns: int,
    temperature: float,
    max_tokens: int,
    task_timeout: int,
    global_steps: int = 0,
) -> RolloutResult:
    rollout_id = f"{problem_id}_{__import__('uuid').uuid4().hex[:8]}"
    result = RolloutResult(rollout_id=rollout_id, problem_id=problem_id)

    task = {
        "problem_id": problem_id,
        "reference_code": reference_code,
        "entry_point": entry_point,
        "is_valid": True,
    }

    env = KernelGymEnv(task=task, config=env_config)
    agent = KernelAgent()

    obs, _ = env.reset(task=task)
    agent.reset()

    for turn_idx in range(max_turns):
        # Build messages for this turn via agent
        agent.update_from_env(observation=obs, reward=0.0, done=False, info={})

        try:
            completion = llm.chat.completions.create(
                model=model_name,
                messages=agent.chat_completions,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as e:
            logger.error("[%s] Turn %d LLM error: %s", problem_id, turn_idx + 1, e)
            result.turns.append(TurnResult(turn=turn_idx, error=f"LLM error: {e}"))
            break

        action = agent.update_from_model(response_text)

        # Step the env with the extracted kernel code
        obs, reward, done, info = env.step(action.action, global_steps=global_steps)

        # Collect turn results from env's meta_info_history
        meta = env.meta_info_history[-1] if env.meta_info_history else {}
        compiled = meta.get("compilation", False)
        correctness = meta.get("correctness", False)
        speedup = meta.get("performance", 0.0)
        error = meta.get("error")

        tr = TurnResult(
            turn=turn_idx,
            compiled=compiled,
            correctness=correctness,
            speedup=speedup,
            reward=reward,
            error=error,
            kernel_code=action.action,
            response=response_text,
            server_result=meta.get("server_result"),
        )
        result.turns.append(tr)
        result.total_reward += reward

        logger.info(
            "[%s] Turn %d: compiled=%s correct=%s speedup=%.2fx reward=%.2f",
            problem_id, turn_idx + 1, compiled, correctness, speedup, reward,
        )

        if correctness and speedup > result.best_speedup:
            result.best_speedup = speedup
            result.best_turn = turn_idx
            result.final_correct = True

        if compiled:
            result.final_compiled = True

        if done:
            break

    env.close()
    return result


# ---------------------------------------------------------------------------
# Problem stats
# ---------------------------------------------------------------------------

def compute_problem_stats(
    problem_id: str,
    rollout_results: list[RolloutResult],
    k_values: list[int],
) -> ProblemStats:
    stats = ProblemStats(problem_id=problem_id)
    stats.num_rollouts = len(rollout_results)

    passed_rollouts = [r for r in rollout_results if r.final_correct]
    stats.num_passed = len(passed_rollouts)

    for k in k_values:
        stats.pass_at_k[k] = calculate_pass_at_k(stats.num_rollouts, stats.num_passed, k)

    if passed_rollouts:
        speedups = [r.best_speedup for r in passed_rollouts]
        stats.best_speedup = max(speedups)
        stats.avg_speedup = np.mean(speedups)

    all_rewards = [r.total_reward for r in rollout_results]
    stats.avg_reward = np.mean(all_rewards) if all_rewards else 0.0

    return stats


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    problem_stats: list[ProblemStats],
    output_dir: str,
    k_values: list[int],
):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    for k in k_values:
        pass_rates = [s.pass_at_k.get(k, 0.0) for s in problem_stats]
        ax1.hist(pass_rates, bins=20, alpha=0.5, label=f'Pass@{k}')
    ax1.set_xlabel('Pass Rate')
    ax1.set_ylabel('Number of Problems')
    ax1.set_title('Distribution of Pass@K Rates')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    problem_ids = [s.problem_id[:20] for s in problem_stats[:20]]
    pass_at_1 = [s.pass_at_k.get(1, 0.0) for s in problem_stats[:20]]
    pass_at_5 = [s.pass_at_k.get(5, 0.0) for s in problem_stats[:20]]
    x = np.arange(len(problem_ids))
    width = 0.35
    ax2.bar(x - width / 2, pass_at_1, width, label='Pass@1')
    ax2.bar(x + width / 2, pass_at_5, width, label='Pass@5')
    ax2.set_xlabel('Problem ID')
    ax2.set_ylabel('Pass Rate')
    ax2.set_title('Pass@K by Problem (First 20)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(problem_ids, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    speedups = [s.best_speedup for s in problem_stats if s.best_speedup > 0]
    if speedups:
        ax3.hist(speedups, bins=20, color='green', alpha=0.7)
        ax3.axvline(x=1.0, color='red', linestyle='--', label='Baseline (1.0x)')
        ax3.set_xlabel('Best Speedup')
        ax3.set_ylabel('Number of Problems')
        ax3.set_title('Distribution of Best Speedups')
        ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    rewards = [s.avg_reward for s in problem_stats]
    ax4.hist(rewards, bins=20, color='purple', alpha=0.7)
    ax4.set_xlabel('Average Reward')
    ax4.set_ylabel('Number of Problems')
    ax4.set_title('Distribution of Average Rewards')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pass_at_k_analysis.png'), dpi=150)
    plt.close()

    fig2, ax = plt.subplots(figsize=(10, 6))
    overall_pass_at_k = {}
    for k in k_values:
        total_rollouts = sum(s.num_rollouts for s in problem_stats)
        total_passed = sum(s.num_passed for s in problem_stats)
        overall_pass_at_k[k] = calculate_pass_at_k(total_rollouts, total_passed, k)

    ax.bar([f'Pass@{k}' for k in k_values], [overall_pass_at_k[k] for k in k_values], color='steelblue')
    ax.set_ylabel('Pass Rate')
    ax.set_title('Overall Pass@K Performance')
    ax.set_ylim(0, 1)
    for i, k in enumerate(k_values):
        ax.text(i, overall_pass_at_k[k] + 0.02, f'{overall_pass_at_k[k]:.2%}', ha='center')
    ax.grid(True, alpha=0.3, axis='y')
    plt.savefig(os.path.join(output_dir, 'overall_pass_at_k.png'), dpi=150)
    plt.close()

    logger.info("Plots saved to %s", output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PASS@K Evaluation for KernelGym (uses rllm KernelGymEnv + KernelAgent)")
    # LLM
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1", help="OpenAI-compatible LLM base URL")
    parser.add_argument("--vllm-api-key", default="EMPTY", help="API key for LLM")
    parser.add_argument("--model-name", default="default", help="Model name")
    # KernelGym
    parser.add_argument("--kernelgym-url", default="http://localhost:8002", help="KernelGym server URL")
    parser.add_argument("--backend", default="triton", help="Kernel backend (triton/cuda)")
    # Data
    parser.add_argument("--data-path", default="data/kernelbench_train.jsonl", help="Path to KernelBench data")
    parser.add_argument("--hf-split", default="level_1", help="HuggingFace split name")
    # Output
    parser.add_argument("--output-dir", default="results/pass_at_k", help="Output directory")
    # Evaluation
    parser.add_argument("--num-rollouts", type=int, default=10, help="Number of rollouts per problem")
    parser.add_argument("--max-turns", type=int, default=3, help="Maximum turns per rollout")
    parser.add_argument("--k-values", default="1,5,10", help="Comma-separated K values for Pass@K")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum tokens per response")
    parser.add_argument("--task-timeout", type=int, default=300, help="Task timeout in seconds")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--limit-problems", type=int, default=None, help="Limit number of problems (for testing)")
    # Env config
    parser.add_argument("--rate-limit", type=int, default=10, help="Rate limit for HTTP worker")
    parser.add_argument("--acquire-timeout", type=int, default=30, help="Rate limiter acquire timeout")
    parser.add_argument("--max-retries", type=int, default=3, help="Max submission retries")
    parser.add_argument("--reward-func", default="calculate_reward_like_kernel",
                        help="Reward function name (calculate_reward_like_kernel/calculate_reward_weighted/calculate_reward_speedup)")
    parser.add_argument("--init-correct-weight", type=float, default=0.5, help="Initial correctness weight in reward")
    parser.add_argument("--init-performance-weight", type=float, default=0.5, help="Initial performance weight in reward")
    parser.add_argument("--speedup-eps", type=float, default=0.05, help="Speedup epsilon threshold")
    parser.add_argument("--speedup-reward-upper-bound", type=float, default=5.0, help="Speedup reward upper bound")
    parser.add_argument("--speedup-reward-lower-bound", type=float, default=0.0, help="Speedup reward lower bound")
    parser.add_argument("--num-perf-trials", type=int, default=100, help="Number of performance trials")
    parser.add_argument("--num-correct-trials", type=int, default=5, help="Number of correctness trials")
    parser.add_argument("--enable-profiling", action="store_true", default=False, help="Enable CUDA profiling")
    parser.add_argument("--verbose-errors", action="store_true", default=True, help="Enable verbose errors")
    parser.add_argument("--detect-decoy-kernel", action="store_true", default=True, help="Detect decoy kernels")
    # Penalties
    parser.add_argument("--penalty-score", type=float, default=-1.0, help="Penalty score for failures")
    parser.add_argument("--compilation-fail-penalty", type=float, default=-0.5, help="Compilation failure penalty")
    parser.add_argument("--correctness-fail-penalty", type=float, default=-0.3, help="Correctness failure penalty")
    parser.add_argument("--perf-degrade-penalty", type=float, default=-0.1, help="Performance degradation penalty")

    args = parser.parse_args()
    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    os.makedirs(args.output_dir, exist_ok=True)
    db_path = os.path.join(args.output_dir, "interactions.db")
    db = InteractionDatabase(db_path)

    env_config = build_env_config(args)

    llm = OpenAI(base_url=args.vllm_url, api_key=args.vllm_api_key)

    # Auto-detect model name if not provided
    model_name = args.model_name
    if model_name == "default":
        try:
            models = llm.models.list()
            model_name = models.data[0].id if models.data else "default"
        except Exception:
            pass

    # Health check
    import httpx
    try:
        health = httpx.get(f"{args.kernelgym_url}/health", timeout=10)
        logger.info("KernelGym health: %s", health.status_code)
    except Exception as e:
        logger.warning("KernelGym health check failed: %s (continuing anyway)", e)

    tasks = load_kernelbench_data(args.data_path, args.hf_split)
    if args.limit_problems:
        tasks = tasks[:args.limit_problems]

    all_results: dict[str, list[RolloutResult]] = {}

    logger.info("Starting PASS@K evaluation: %d problems, %d rollouts each, %d max turns",
                len(tasks), args.num_rollouts, args.max_turns)

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures: dict[Any, tuple[str, int]] = {}
        for task in tasks:
            for rollout_idx in range(args.num_rollouts):
                future = executor.submit(
                    run_single_rollout,
                    task["problem_id"],
                    task["reference_code"],
                    task["entry_point"],
                    llm,
                    env_config,
                    model_name,
                    args.max_turns,
                    args.temperature,
                    args.max_tokens,
                    args.task_timeout,
                    0,
                )
                futures[future] = (task["problem_id"], rollout_idx)

        for future in as_completed(futures):
            problem_id, rollout_idx = futures[future]
            try:
                result = future.result(timeout=args.task_timeout * args.max_turns + 120)
                if problem_id not in all_results:
                    all_results[problem_id] = []
                all_results[problem_id].append(result)
                db.save_rollout(result)
                logger.info("[%s] Rollout %d/%d completed: correct=%s speedup=%.2f",
                           problem_id, rollout_idx + 1, args.num_rollouts,
                           result.final_correct, result.best_speedup)
            except Exception as e:
                logger.error("[%s] Rollout %d failed: %s", problem_id, rollout_idx, e)

    # Compute per-problem and overall stats
    problem_stats: list[ProblemStats] = []
    for problem_id, results in all_results.items():
        stats = compute_problem_stats(problem_id, results, k_values)
        problem_stats.append(stats)
        db.save_problem_stats(stats)
        logger.info(
            "[%s] Stats: Pass@1=%.2f%% Pass@5=%.2f%% BestSpeedup=%.2fx",
            problem_id,
            stats.pass_at_k.get(1, 0) * 100,
            stats.pass_at_k.get(5, 0) * 100,
            stats.best_speedup,
        )

    total_rollouts = sum(s.num_rollouts for s in problem_stats)
    total_passed = sum(s.num_passed for s in problem_stats)
    overall_pass_at_k = {}
    for k in k_values:
        overall_pass_at_k[k] = calculate_pass_at_k(total_rollouts, total_passed, k)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_problems": len(tasks),
            "num_rollouts_per_problem": args.num_rollouts,
            "max_turns": args.max_turns,
            "k_values": k_values,
            "reward_func": args.reward_func,
        },
        "overall_metrics": {
            "total_rollouts": total_rollouts,
            "total_passed": total_passed,
            "pass_at_k": {f"pass@{k}": v for k, v in overall_pass_at_k.items()},
        },
        "problem_stats": [asdict(s) for s in problem_stats],
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary saved to %s", summary_path)

    plot_results(problem_stats, args.output_dir, k_values)

    print("\n" + "=" * 60)
    print("PASS@K EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Problems: {len(tasks)}")
    print(f"Rollouts per Problem: {args.num_rollouts}")
    print(f"Max Turns: {args.max_turns}")
    print("-" * 60)
    for k in k_values:
        print(f"Overall Pass@{k}: {overall_pass_at_k[k]:.2%}")
    print("-" * 60)
    print(f"Total Passed Rollouts: {total_passed}/{total_rollouts}")
    print(f"Database: {db_path}")
    print(f"Plots: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
