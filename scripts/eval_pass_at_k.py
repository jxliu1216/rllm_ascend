"""
PASS@K Evaluation Script for KernelGym.

This version reuses rLLM's KernelGym components directly:
- `KernelAgent` for multi-turn prompt/state/code extraction
- `KernelGymEnv.step()` for server payload, reward, and episode state
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sqlite3
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATEFMT = "%H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
)
logger = logging.getLogger("eval_pass_at_k")

MESSAGE_PASSTHROUGH_MARKER = "<|message_passthrough|>"
LEGACY_MESSAGE_PASSTHROUGH_MARKER = "<|message_passtrhough|>"


def split_message_passthrough(action: str) -> tuple[str, str]:
    for marker in (MESSAGE_PASSTHROUGH_MARKER, LEGACY_MESSAGE_PASSTHROUGH_MARKER):
        if marker in action:
            kernel_code, llm_messages = action.split(marker, 1)
            return kernel_code.strip(), llm_messages
    return action, ""


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, tqdm_cls):
        super().__init__()
        self._tqdm = tqdm_cls

    def emit(self, record: logging.LogRecord):
        try:
            self._tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


@contextmanager
def quiet_worker_output(enabled: bool):
    if not enabled:
        yield
        return

    root = logging.getLogger()
    old_handlers = root.handlers[:]
    old_level = root.level
    with open(os.devnull, "w") as devnull:
        root.handlers = [logging.NullHandler()]
        root.setLevel(logging.CRITICAL)
        try:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                yield
        finally:
            root.handlers = old_handlers
            root.setLevel(old_level)


class ProgressBar:
    def __init__(self, total: int, desc: str):
        self.total = total
        self.desc = desc
        self.count = 0
        self._bar = None
        self._log_every = max(1, total // 20) if total else 1
        self._root_logger = None
        self._old_handlers = None

    def __enter__(self):
        try:
            from tqdm import tqdm

            self._bar = tqdm(total=self.total, desc=self.desc, unit="rollout", dynamic_ncols=True, position=0)
            self._root_logger = logging.getLogger()
            self._old_handlers = self._root_logger.handlers[:]
            handler = TqdmLoggingHandler(tqdm)
            handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATEFMT))
            self._root_logger.handlers = [handler]
        except Exception:
            logger.info("%s progress: 0/%d", self.desc, self.total)
        return self

    def update(self, *, passed: int, failed: int):
        self.count += 1
        if self._bar is not None:
            self._bar.update(1)
            self._bar.set_postfix(passed=passed, failed=failed)
        elif self.count == self.total or self.count % self._log_every == 0:
            logger.info("%s progress: %d/%d passed=%d failed=%d", self.desc, self.count, self.total, passed, failed)

    def __exit__(self, exc_type, exc, tb):
        if self._root_logger is not None and self._old_handlers is not None:
            self._root_logger.handlers = self._old_handlers
        if self._bar is not None:
            self._bar.close()


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


class InteractionDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
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
            """
        )
        cursor.execute(
            """
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
            """
        )
        cursor.execute(
            """
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
            """
        )
        conn.commit()
        conn.close()

    def save_rollout(self, result: RolloutResult):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO rollouts (rollout_id, problem_id, timestamp, best_speedup, final_correct, final_compiled, total_reward)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.rollout_id,
                result.problem_id,
                datetime.now().isoformat(),
                result.best_speedup,
                int(result.final_correct),
                int(result.final_compiled),
                result.total_reward,
            ),
        )
        for turn in result.turns:
            cursor.execute(
                """
                INSERT INTO turns (rollout_id, turn, compiled, correctness, speedup, reward, error, kernel_code, response, server_result)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
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
                ),
            )
        conn.commit()
        conn.close()

    def save_problem_stats(self, stats: ProblemStats):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO problem_stats
            (problem_id, num_rollouts, num_passed, pass_at_1, pass_at_5, pass_at_10, best_speedup, avg_speedup, avg_reward)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stats.problem_id,
                stats.num_rollouts,
                stats.num_passed,
                stats.pass_at_k.get(1, 0.0),
                stats.pass_at_k.get(5, 0.0),
                stats.pass_at_k.get(10, 0.0),
                stats.best_speedup,
                stats.avg_speedup,
                stats.avg_reward,
            ),
        )
        conn.commit()
        conn.close()


def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    if n - k < 0:
        return 0.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)) if n > c else 1.0


def build_reward_config(args: argparse.Namespace):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "max_turns": args.max_turns,
            "server_url": args.kernelgym_url,
            "rate_limit": args.rate_limit,
            "acquire_timeout": args.acquire_timeout,
            "timeout": args.task_timeout_in_client,
            "max_retries": args.max_retries,
            "task_timeout": args.task_timeout,
            "task_timeout_in_client": args.task_timeout_in_client,
            "reward_func_name": args.reward_func_name,
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
            "reference_backend": args.reference_backend,
            "train_id": args.train_id,
            "reward_policy": {
                "penalties": {
                    "penalty_score": args.penalty_score,
                    "compilation_fail": args.compilation_fail_penalty,
                    "correctness_fail": args.correctness_fail_penalty,
                    "perf_degrade": args.perf_degrade_penalty,
                }
            },
            "coverage_reward": {
                "reward_type": "time_coverage",
                "enable": False,
                "weight": 0.0,
            },
        }
    )


def run_single_rollout(
    task: dict[str, Any],
    args: argparse.Namespace,
    model_name: str,
) -> RolloutResult:
    with quiet_worker_output(getattr(args, "quiet_worker_output", True)):
        return _run_single_rollout(task, args, model_name)


def _run_single_rollout(
    task: dict[str, Any],
    args: argparse.Namespace,
    model_name: str,
) -> RolloutResult:
    from rllm.agents.kernelgym_agent import KernelAgent
    from rllm.environments.kernelgym.kernelgym_env import KernelGymEnv

    problem_id = task["problem_id"]
    reference_code = task["reference_code"]
    entry_point = task.get("entry_point", "Model")
    rollout_id = f"{problem_id}_{uuid4().hex[:8]}"
    result = RolloutResult(rollout_id=rollout_id, problem_id=problem_id)

    from openai import OpenAI

    llm = OpenAI(base_url=args.vllm_url, api_key=args.vllm_api_key)
    env = KernelGymEnv(
        task={
            "problem_id": problem_id,
            "task_id": problem_id,
            "reference_code": reference_code,
            "entry_point": entry_point,
            "is_valid": args.is_valid_eval,
            "eval_tag": args.eval_tag,
        },
        config=build_reward_config(args),
        message_passthrough=True,
    )
    agent = KernelAgent(message_passthrough=True)
    try:
        obs, info = env.reset()
        agent.reset()
        agent.update_from_env(obs, reward=0.0, done=False, info=info)

        for turn_idx in range(args.max_turns):
            try:
                completion = llm.chat.completions.create(
                    model=model_name,
                    messages=agent.chat_completions,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as e:
                result.turns.append(TurnResult(turn=turn_idx, error=f"LLM error: {e}"))
                break

            action = agent.update_from_model(response_text).action
            next_obs, reward, done, step_info = env.step(action, global_steps=0)
            meta = env.meta_info_history[-1] if env.meta_info_history else {}
            eval_result = meta.get("server_result", next_obs)
            compiled = bool(meta.get("compilation", eval_result.get("compiled", False)))
            correctness = bool(meta.get("correctness", eval_result.get("correctness", False)))
            speedup = float(meta.get("performance", eval_result.get("speedup") or 0.0) or 0.0)
            error = meta.get("error") or eval_result.get("error_message") or eval_result.get("error")
            kernel_code, _ = split_message_passthrough(action)

            tr = TurnResult(
                turn=turn_idx,
                compiled=compiled,
                correctness=correctness,
                speedup=speedup,
                reward=float(reward),
                error=error,
                kernel_code=kernel_code,
                response=response_text,
                server_result=eval_result,
            )
            result.turns.append(tr)
            result.total_reward += float(reward)
            logger.debug(
                "[%s] Turn %d: compiled=%s correct=%s speedup=%.2fx reward=%.3f",
                problem_id,
                turn_idx + 1,
                compiled,
                correctness,
                speedup,
                float(reward),
            )
            if correctness:
                result.final_correct = True
                if result.best_turn < 0 or speedup > result.best_speedup:
                    result.best_speedup = speedup
                    result.best_turn = turn_idx
            if compiled:
                result.final_compiled = True

            agent.update_from_env(next_obs, reward=float(reward), done=done, info=step_info)
            if done:
                break
    finally:
        env.close()

    return result


def load_kernelbench_data(data_path: str, hf_split: str = "level_1") -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    path = Path(data_path)
    if path.suffix in (".parquet", ".jsonl") or path.exists():
        import pandas as pd

        if path.suffix == ".jsonl":
            df = pd.read_json(data_path, lines=True)
        else:
            df = pd.read_parquet(data_path)
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            if "task" in row_dict:
                t = row_dict["task"]
                if isinstance(t, str):
                    t = json.loads(t)
                tasks.append(
                    {
                        "problem_id": t.get("problem_id", f"task_{idx}"),
                        "reference_code": t.get("reference_code", ""),
                        "entry_point": t.get("entry_point", "Model"),
                    }
                )
            elif "reference_code" in row_dict:
                tasks.append(
                    {
                        "problem_id": row_dict.get("problem_id", f"task_{idx}"),
                        "reference_code": row_dict["reference_code"],
                        "entry_point": row_dict.get("entry_point", "Model"),
                    }
                )
            elif "extra_info" in row_dict:
                extra = row_dict.get("extra_info", {}) or {}
                if isinstance(extra, str):
                    extra = json.loads(extra)
                ref_code = extra.get("ground_truth", extra.get("task_code", ""))
                if ref_code:
                    tasks.append(
                        {
                            "problem_id": str(extra.get("uuid", extra.get("op_name", f"task_{idx}"))),
                            "reference_code": ref_code,
                            "entry_point": extra.get("entry_point", "Model"),
                        }
                    )
            elif "code" in row_dict:
                level = row_dict.get("level", "")
                pid = row_dict.get("problem_id", idx)
                name = row_dict.get("name", "")
                tasks.append(
                    {
                        "problem_id": f"level{level}_{pid}_{name}" if level else f"task_{pid}",
                        "reference_code": row_dict["code"],
                        "entry_point": "Model",
                    }
                )
    else:
        from datasets import load_dataset

        ds = load_dataset("ScalingIntelligence/KernelBench", split=hf_split)
        for idx, row in enumerate(ds):
            level = row.get("level", "")
            pid = row.get("problem_id", idx)
            name = row.get("name", "")
            tasks.append(
                {
                    "problem_id": f"level{level}_{pid}_{name}" if level else f"task_{pid}",
                    "reference_code": row["code"],
                    "entry_point": "Model",
                }
            )
    logger.info("Loaded %d tasks from %s", len(tasks), data_path)
    return tasks


def compute_problem_stats(problem_id: str, rollout_results: list[RolloutResult], k_values: list[int]) -> ProblemStats:
    stats = ProblemStats(problem_id=problem_id)
    stats.num_rollouts = len(rollout_results)
    passed_rollouts = [r for r in rollout_results if r.final_correct]
    stats.num_passed = len(passed_rollouts)
    for k in k_values:
        stats.pass_at_k[k] = calculate_pass_at_k(stats.num_rollouts, stats.num_passed, k)
    if passed_rollouts:
        speedups = [r.best_speedup for r in passed_rollouts]
        stats.best_speedup = max(speedups)
        stats.avg_speedup = float(np.mean(speedups))
    all_rewards = [r.total_reward for r in rollout_results]
    stats.avg_reward = float(np.mean(all_rewards)) if all_rewards else 0.0
    return stats


def plot_results(problem_stats: list[ProblemStats], output_dir: str, k_values: list[int]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1 = axes[0, 0]
    for k in k_values:
        pass_rates = [s.pass_at_k.get(k, 0.0) for s in problem_stats]
        ax1.hist(pass_rates, bins=20, alpha=0.5, label=f"Pass@{k}")
    ax1.set_xlabel("Pass Rate")
    ax1.set_ylabel("Number of Problems")
    ax1.set_title("Distribution of Pass@K Rates")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    problem_ids = [s.problem_id[:20] for s in problem_stats[:20]]
    pass_at_1 = [s.pass_at_k.get(1, 0.0) for s in problem_stats[:20]]
    pass_at_5 = [s.pass_at_k.get(5, 0.0) for s in problem_stats[:20]]
    x = np.arange(len(problem_ids))
    width = 0.35
    ax2.bar(x - width / 2, pass_at_1, width, label="Pass@1")
    ax2.bar(x + width / 2, pass_at_5, width, label="Pass@5")
    ax2.set_xlabel("Problem ID")
    ax2.set_ylabel("Pass Rate")
    ax2.set_title("Pass@K by Problem (First 20)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(problem_ids, rotation=45, ha="right")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    speedups = [s.best_speedup for s in problem_stats if s.best_speedup > 0]
    if speedups:
        ax3.hist(speedups, bins=20, color="green", alpha=0.7)
        ax3.axvline(x=1.0, color="red", linestyle="--", label="Baseline (1.0x)")
        ax3.set_xlabel("Best Speedup")
        ax3.set_ylabel("Number of Problems")
        ax3.set_title("Distribution of Best Speedups")
        ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    rewards = [s.avg_reward for s in problem_stats]
    ax4.hist(rewards, bins=20, color="purple", alpha=0.7)
    ax4.set_xlabel("Average Reward")
    ax4.set_ylabel("Number of Problems")
    ax4.set_title("Distribution of Average Rewards")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pass_at_k_analysis.png"), dpi=150)
    plt.close()

    fig2, ax = plt.subplots(figsize=(10, 6))
    total_rollouts = sum(s.num_rollouts for s in problem_stats)
    total_passed = sum(s.num_passed for s in problem_stats)
    overall_pass_at_k = {k: calculate_pass_at_k(total_rollouts, total_passed, k) for k in k_values}
    ax.bar([f"Pass@{k}" for k in k_values], [overall_pass_at_k[k] for k in k_values], color="steelblue")
    ax.set_ylabel("Pass Rate")
    ax.set_title("Overall Pass@K Performance")
    ax.set_ylim(0, 1)
    for i, k in enumerate(k_values):
        ax.text(i, overall_pass_at_k[k] + 0.02, f"{overall_pass_at_k[k]:.2%}", ha="center")
    ax.grid(True, alpha=0.3, axis="y")
    plt.savefig(os.path.join(output_dir, "overall_pass_at_k.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="PASS@K Evaluation for KernelGym (rLLM-native)")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--vllm-api-key", default="EMPTY")
    parser.add_argument("--model-name", default="default")
    parser.add_argument("--kernelgym-url", default="http://localhost:8002")
    parser.add_argument("--data-path", default="data/kernelbench_train.jsonl")
    parser.add_argument("--hf-split", default="level_1")
    parser.add_argument("--output-dir", default="results/pass_at_k")
    parser.add_argument("--num-rollouts", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--k-values", default="1,5,10")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-problems", type=int, default=None)
    parser.add_argument("--reference-backend", default="triton")
    parser.add_argument("--reward-func-name", default="calculate_reward_weighted")
    parser.add_argument("--init-correct-weight", type=float, default=0.5)
    parser.add_argument("--init-performance-weight", type=float, default=0.5)
    parser.add_argument("--speedup-eps", type=float, default=0.01)
    parser.add_argument("--speedup-reward-upper-bound", type=float, default=3.0)
    parser.add_argument("--speedup-reward-lower-bound", type=float, default=0.0)
    parser.add_argument("--penalty-score", type=float, default=0.0)
    parser.add_argument("--compilation-fail-penalty", type=float, default=-0.5)
    parser.add_argument("--correctness-fail-penalty", type=float, default=-0.3)
    parser.add_argument("--perf-degrade-penalty", type=float, default=-0.1)
    parser.add_argument("--task-timeout", type=int, default=180)
    parser.add_argument("--task-timeout-in-client", type=int, default=360)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--num-perf-trials", type=int, default=20)
    parser.add_argument("--enable-profiling", action="store_true", default=True)
    parser.add_argument("--disable-profiling", dest="enable_profiling", action="store_false")
    parser.add_argument("--verbose-errors", action="store_true", default=True)
    parser.add_argument("--quiet-errors", dest="verbose_errors", action="store_false")
    parser.add_argument("--detect-decoy-kernel", action="store_true", default=True)
    parser.add_argument("--disable-detect-decoy-kernel", dest="detect_decoy_kernel", action="store_false")
    parser.add_argument("--is-valid-eval", action="store_true", default=True)
    parser.add_argument("--is-train-eval", dest="is_valid_eval", action="store_false")
    parser.add_argument("--eval-tag", default="")
    parser.add_argument(
        "--train-id",
        default="",
        help="Training/evaluation run id shown in KernelGym dashboard. Auto-generated when omitted.",
    )
    parser.add_argument("--rate-limit", type=int, default=64)
    parser.add_argument("--acquire-timeout", type=int, default=2400)
    parser.add_argument(
        "--worker-backend",
        choices=("process", "thread"),
        default="process",
        help="Use process workers by default so signal-based timeouts run in worker main threads.",
    )
    parser.add_argument(
        "--process-start-method",
        choices=("spawn", "fork", "forkserver"),
        default="spawn",
        help="Multiprocessing start method when --worker-backend=process.",
    )
    parser.add_argument(
        "--show-worker-logs",
        dest="quiet_worker_output",
        action="store_false",
        help="Show stdout/stderr from worker processes. Disabled by default so the progress bar stays at the bottom.",
    )
    parser.set_defaults(quiet_worker_output=False)
    args = parser.parse_args()
    if not args.train_id:
        args.train_id = f"pass_at_k_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"

    k_values = [int(k.strip()) for k in args.k_values.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)
    db = InteractionDatabase(os.path.join(args.output_dir, "interactions.db"))
    tasks = load_kernelbench_data(args.data_path, args.hf_split)
    if args.limit_problems:
        tasks = tasks[: args.limit_problems]

    model_name = args.model_name
    if model_name == "default":
        try:
            from openai import OpenAI

            llm = OpenAI(base_url=args.vllm_url, api_key=args.vllm_api_key)
            models = llm.models.list()
            model_name = models.data[0].id if models.data else model_name
        except Exception as e:
            logger.warning("Model auto-detection failed: %s; using %s", e, model_name)

    logger.info(
        "Starting PASS@K evaluation: %d problems, %d rollouts each, train_id=%s",
        len(tasks),
        args.num_rollouts,
        args.train_id,
    )
    all_results: dict[str, list[RolloutResult]] = {}
    if args.worker_backend == "process":
        executor_factory = lambda: ProcessPoolExecutor(
            max_workers=args.num_workers,
            mp_context=mp.get_context(args.process_start_method),
        )
    else:
        logger.warning("Using thread workers; signal-based timeouts may fail outside the main thread")
        executor_factory = lambda: ThreadPoolExecutor(max_workers=args.num_workers)

    with executor_factory() as executor:
        futures: dict[Any, str] = {}
        for task in tasks:
            for _ in range(args.num_rollouts):
                future = executor.submit(run_single_rollout, task, args, model_name)
                futures[future] = task["problem_id"]
        passed_rollouts = 0
        failed_rollouts = 0
        with ProgressBar(total=len(futures), desc="Rollouts") as progress:
            for future in as_completed(futures):
                problem_id = futures[future]
                try:
                    result = future.result(timeout=args.task_timeout_in_client * args.max_turns + 120)
                    all_results.setdefault(problem_id, []).append(result)
                    db.save_rollout(result)
                    if result.final_correct:
                        passed_rollouts += 1
                    logger.info(
                        "[%s] Rollout completed: correct=%s speedup=%.2f",
                        problem_id,
                        result.final_correct,
                        result.best_speedup,
                    )
                except Exception as e:
                    failed_rollouts += 1
                    logger.error("[%s] Rollout failed: %s", problem_id, e)
                finally:
                    progress.update(passed=passed_rollouts, failed=failed_rollouts)

    problem_stats: list[ProblemStats] = []
    for problem_id, results in all_results.items():
        stats = compute_problem_stats(problem_id, results, k_values)
        problem_stats.append(stats)
        db.save_problem_stats(stats)

    total_rollouts = sum(s.num_rollouts for s in problem_stats)
    total_passed = sum(s.num_passed for s in problem_stats)
    overall_pass_at_k = {k: calculate_pass_at_k(total_rollouts, total_passed, k) for k in k_values}
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_problems": len(tasks),
            "num_rollouts_per_problem": args.num_rollouts,
            "max_turns": args.max_turns,
            "k_values": k_values,
            "reward_func_name": args.reward_func_name,
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
    print(f"Database: {os.path.join(args.output_dir, 'interactions.db')}")
    print(f"Plots: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
