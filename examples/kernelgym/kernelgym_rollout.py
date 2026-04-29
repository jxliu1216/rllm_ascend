"""KernelGYM rollout function for fully-async training.

Wraps the multi-turn agent-environment interaction into a single
``async def rollout_fn(client, tokenizer, **kwargs) -> Trajectory``
that the ``RolloutExecutor`` can drive concurrently.

The function mirrors the conversation flow from ``KernelAgent`` +
``KernelGymEnv`` but talks to the LLM through ``RolloutClient``
(HTTP-based, version-aware) instead of the synchronous
``AgentExecutionEngine``.

Eval + reward are aligned with rllm-071: ``POST /evaluate`` then poll
``/status`` + ``/results``, then the same reward functions as
``KernelGymEnv`` (``calculate_reward_like_kernel`` / weighted / speedup).
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from typing import Any, Optional

from rllm.agents.kernelgym_agent import KernelAgent
from rllm.environments.kernelgym.kernelgym_eval_hybrid_async import (
    async_submit_and_poll,
)
from rllm.environments.kernelgym.kernelgym_reward_ops import (
    KernelGymHybridRewardParams,
    KernelGymRewardOps,
    preflight_failure_result,
    preflight_validate,
)

from rllm.experimental.fully_async.protocol import Trajectory

logger = logging.getLogger(__name__)

_MIN_KERNEL_CODE_CHARS = 10

# ---------------------------------------------------------------------------
# Feedback builder
# ---------------------------------------------------------------------------


def _build_feedback(
    compiled: bool,
    correctness: Optional[bool],
    speedup: Optional[float],
    error_message: Optional[str],
) -> str:
    lines: list[str] = []
    if not compiled:
        lines.append("❌ Compilation FAILED.")
        if error_message:
            lines.append(f"Error:\n{error_message}")
        lines.append("Please fix the compilation error and resubmit your kernel.")
    elif not correctness:
        lines.append("✅ Compilation succeeded.")
        lines.append("❌ Correctness check FAILED — outputs do not match the reference.")
        if error_message:
            lines.append(f"Error:\n{error_message}")
        lines.append("Please fix the numerical correctness issue and resubmit.")
    else:
        speedup_str = f"{speedup:.2f}x" if speedup is not None else "N/A"
        lines.append("✅ Compilation succeeded.")
        lines.append("✅ Correctness check PASSED.")
        lines.append(f"Speedup over reference: {speedup_str}")
        lines.append("Can you further optimize the kernel?")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hybrid evaluate + rllm-071 reward (async)
# ---------------------------------------------------------------------------

_EVAL_SEMAPHORE: asyncio.Semaphore | None = None


def _get_global_eval_semaphore() -> asyncio.Semaphore:
    global _EVAL_SEMAPHORE
    if _EVAL_SEMAPHORE is None:
        _EVAL_SEMAPHORE = asyncio.Semaphore(32)
    return _EVAL_SEMAPHORE


async def _mock_raw_eval_result(task_id: str) -> dict[str, Any]:
    """Plausible server-shaped dict for ``reward_ops.apply_reward`` (no HTTP)."""
    await asyncio.sleep(0.01)
    compiled = random.random() < 0.82
    correctness = compiled and random.random() < 0.65
    speedup = round(random.uniform(0.85, 2.4), 3) if correctness else None
    return {
        "status": "completed",
        "task_id": task_id,
        "compiled": compiled,
        "correctness": correctness,
        "speedup": speedup,
        "error_message": None,
        "decoy_kernel": False,
    }


async def _evaluate_and_score_turn(
    task: dict,
    kernel_code: str,
    *,
    server_url: str,
    reward_ops: KernelGymRewardOps,
    num_correct_trials: int,
    num_perf_trials: int,
    per_task_timeout: int,
    per_task_timeout_in_client: int,
    max_retries: int | None,
    rate_limit: int,
    acquire_timeout: int,
    reference_backend: str | None,
    toolkit: str,
    backend_adapter: str,
    workflow: str,
    is_valid: bool,
    detect_decoy_kernel: bool,
    enable_profiling: bool,
    verbose_errors: bool,
    rerun_on_anomaly_speedup: bool,
    mock_eval: bool = False,
) -> dict[str, Any]:
    """Run async submit/poll, apply hybrid reward, optional anomaly re-run."""
    kc = (kernel_code or "").strip()
    problem_id = str(task.get("problem_id", "task"))
    suffix = uuid.uuid4().hex[:16]
    max_task_id_len = 100
    prefix_max_len = max_task_id_len - (1 + len(suffix))
    safe_prefix = problem_id[:prefix_max_len] if prefix_max_len > 0 else "task"
    task_id = f"{safe_prefix}_{suffix}"

    entry_point = str(task.get("entry_point", "Model"))
    ref_code = str(task.get("reference_code", ""))

    def _synthetic_fail(msg: str) -> dict[str, Any]:
        ret = {
            "status": "failed",
            "error_message": msg,
            "error": msg,
            "task_id": task_id,
            "compiled": False,
            "correctness": False,
            "speedup": 0.0,
        }
        return reward_ops.apply_reward(ret)

    if len(kc) < _MIN_KERNEL_CODE_CHARS:
        return _synthetic_fail(
            f"kernel_code too short ({len(kc)} chars; min {_MIN_KERNEL_CODE_CHARS})"
        )

    ok, missing = preflight_validate(ref_code, kc, entry_point)
    if not ok:
        logger.info("KernelGYM preflight failed: %s", missing)
        pre = preflight_failure_result(
            f"Client validation failed: missing {missing}",
            penalty=reward_ops.p.penalty_score,
            task_id=task_id,
        )
        return reward_ops.apply_reward(pre)

    if mock_eval:
        logger.debug("KernelGYM mock_eval: skip HTTP, task_id=%s", task_id)
        raw = await _mock_raw_eval_result(task_id)
        merged = reward_ops.apply_reward(raw)
        return merged

    p_timeout = int(task.get("task_timeout", per_task_timeout))
    p_client = int(
        task.get("task_timeout_in_client", per_task_timeout_in_client)
    )
    if p_client < p_timeout:
        p_client = p_timeout

    backend_value = task.get("reference_backend", reference_backend)
    if backend_value is None:
        backend_value = task.get("backend", "cuda")

    payload: dict[str, Any] = {
        # Always use locally-generated task_id: never inherit from dataset row to
        # guarantee uniqueness across concurrent rollouts (H2 fix).
        "task_id": task_id,
        "reference_code": ref_code,
        "kernel_code": kc,
        "backend": backend_value,
        "num_correct_trials": int(task.get("num_correct_trials", num_correct_trials)),
        "num_perf_trials": int(task.get("num_perf_trials", num_perf_trials)),
        "timeout": p_timeout,
        "priority": "normal",
        "entry_point": entry_point,
        "is_valid": bool(task.get("is_valid", is_valid)),
        "verbose_errors": bool(task.get("verbose_errors", verbose_errors)),
        "enable_profiling": bool(task.get("enable_profiling", enable_profiling)),
        "detect_decoy_kernel": bool(task.get("detect_decoy_kernel", detect_decoy_kernel)),
        "reference_backend": task.get("reference_backend", reference_backend),
    }
    if payload["is_valid"]:
        payload["detect_decoy_kernel"] = True

    # Optional fields for server versions that expect KernelBench-style metadata
    payload["toolkit"] = str(task.get("toolkit", toolkit))
    payload["backend_adapter"] = str(task.get("backend_adapter", backend_adapter))
    payload["workflow"] = str(task.get("workflow", workflow))

    async def _one_eval(override_task_id: str | None = None) -> dict[str, Any]:
        # Use override_task_id when provided (e.g. anomaly rerun) so each
        # submission has a unique task_id and the server cannot return a cached
        # result from the previous attempt (H1 fix).
        p = payload if override_task_id is None else {**payload, "task_id": override_task_id}
        sema = _get_global_eval_semaphore()
        async with sema:
            raw = await async_submit_and_poll(
                server_url,
                p,
                default_timeout=p_timeout,
                client_timeout=p_client,
                max_retries=max_retries,
                acquire_timeout=acquire_timeout,
                rate_limit=rate_limit,
            )
        merged = reward_ops.apply_reward(raw)
        return merged

    merged = await _one_eval()

    su = float(merged.get("speedup") or 0.0)
    if (
        rerun_on_anomaly_speedup
        and su > reward_ops.p.speedup_reward_upper_bound
    ):
        logger.warning(
            "KernelGYM: speedup %.4f > upper bound %.4f, re-executing once",
            su,
            reward_ops.p.speedup_reward_upper_bound,
        )
        # Generate a fresh task_id so the server treats this as a new request
        # and does not return the anomalous cached result (H1 fix).
        rerun_task_id = f"{safe_prefix}_{uuid.uuid4().hex[:16]}"
        merged = await _one_eval(override_task_id=rerun_task_id)

    return merged


# ---------------------------------------------------------------------------
# Core rollout: multi-turn agent loop
# ---------------------------------------------------------------------------


async def kernelgym_rollout(
    client,
    task: dict,
    max_turns: int = 3,
    system_prompt: str | None = None,
    kernel_server_url: str = "http://localhost:8000",
    kernel_eval_timeout: int = 300,
    kernel_eval_client_timeout: int | None = None,
    num_correct_trials: int = 5,
    num_perf_trials: int = 100,
    toolkit: str = "kernelbench",
    backend_adapter: str = "kernelbench",
    backend: str = "cuda",
    reference_backend: str | None = None,
    is_valid: bool = True,
    detect_decoy_kernel: bool = True,
    enable_profiling: bool = True,
    verbose_errors: bool = True,
    max_retries: int | None = 3,
    rate_limit: int = 8,
    acquire_timeout: int = 120,
    reward_params: KernelGymHybridRewardParams | None = None,
    reward_ops: KernelGymRewardOps | None = None,
    workflow: str = "kernelbench",
    rerun_on_anomaly_speedup: bool = True,
    early_exit_on_correct: bool = False,
    sampling_params: dict | None = None,
    mock_eval: bool = False,
) -> dict[str, Any]:
    """Run a complete KernelGYM multi-turn episode (071-aligned eval + reward)."""
    if reward_ops is not None:
        rops = reward_ops
    else:
        rp = reward_params or KernelGymHybridRewardParams()
        rops = KernelGymRewardOps(rp)

    agent = KernelAgent(system_prompt=system_prompt)
    agent.reset()
    agent.update_from_env(observation=task, reward=0.0, done=False, info={})

    client_t = int(
        kernel_eval_client_timeout
        if kernel_eval_client_timeout is not None
        else max(kernel_eval_timeout, 300)
    )

    protocol_trajectory = Trajectory(sequences=[], reward=0, metadata={})

    sampling_params = sampling_params or {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
    }

    total_completion_tokens = 0
    total_generation_time = 0.0
    total_eval_time = 0.0
    last_reward = 0.0
    best_reward: Optional[float] = None
    last_result: dict[str, Any] = {}
    overlong = False
    num_turns = 0
    ref_b = reference_backend if reference_backend is not None else backend

    try:
        for turn in range(max_turns):
            num_turns = turn + 1

            gen_start = time.time()
            messages = agent.chat_completions
            response_msg, output = await client.chat_completion(
                messages, sampling_params=sampling_params
            )
            gen_time = time.time() - gen_start
            total_generation_time += gen_time

            action = agent.update_from_model(response_msg.get("content") or "")
            protocol_trajectory.append(output.to_sequence())
            total_completion_tokens += output.num_output_tokens

            eval_start = time.time()
            last_result = await _evaluate_and_score_turn(
                task,
                action.action,
                server_url=kernel_server_url,
                reward_ops=rops,
                num_correct_trials=num_correct_trials,
                num_perf_trials=num_perf_trials,
                per_task_timeout=kernel_eval_timeout,
                per_task_timeout_in_client=client_t,
                max_retries=max_retries,
                rate_limit=rate_limit,
                acquire_timeout=acquire_timeout,
                reference_backend=ref_b,
                toolkit=toolkit,
                backend_adapter=backend_adapter,
                workflow=workflow,
                is_valid=is_valid,
                detect_decoy_kernel=detect_decoy_kernel,
                enable_profiling=enable_profiling,
                verbose_errors=verbose_errors,
                rerun_on_anomaly_speedup=rerun_on_anomaly_speedup,
                mock_eval=mock_eval,
            )
            total_eval_time += time.time() - eval_start

            compiled = bool(last_result.get("compiled", False))
            correctness = last_result.get("correctness")
            speedup = last_result.get("speedup")
            if isinstance(speedup, (int, float)):
                spv: Optional[float] = float(speedup)
            else:
                spv = None
            error_msg: str | None = last_result.get("error")
            if not error_msg:
                error_msg = last_result.get("error_message")

            last_reward = rops.score_from_merged(last_result)
            best_reward = (
                last_reward if best_reward is None else max(best_reward, last_reward)
            )

            if turn == max_turns - 1:
                break

            # early_exit_on_correct=True: stop as soon as the kernel compiles and
            # is correct, regardless of speedup.  For speedup-based rewards
            # (calculate_reward_speedup / calculate_reward_weighted) leave this
            # False so subsequent turns can continue to improve performance.
            if early_exit_on_correct and compiled and correctness:
                break

            feedback = _build_feedback(
                compiled,
                bool(correctness) if correctness is not None else None,
                spv,
                error_msg,
            )
            agent.update_from_env(
                observation=feedback,
                reward=last_reward,
                done=False,
                info=last_result,
            )

    except Exception as exc:  # noqa: BLE001
        import traceback

        overlong = True
        logger.warning(
            "KernelGYM rollout error for %s: %s\n%s",
            task.get("problem_id", "?"),
            exc,
            traceback.format_exc(),
        )

    if overlong:
        for seq in protocol_trajectory.sequences:
            seq.response_masks = [0] * len(seq.response_masks)

    lr = last_result
    metrics = {
        "num_turns": num_turns,
        "total_generation_time": total_generation_time,
        "total_eval_time": total_eval_time,
        "total_completion_tokens": total_completion_tokens,
        "last_reward": last_reward,
        "best_reward": best_reward if best_reward is not None else 0.0,
        "compiled": float(lr.get("compiled", False)),
        "correctness": float(lr.get("correctness", False) or False),
        "speedup": float(lr.get("speedup") or 0.0),
        "problem_id": task.get("problem_id", "unknown"),
        "overlong": float(overlong),
        "merged_step": len(protocol_trajectory.merge()),
    }

    return {
        "trajectory": protocol_trajectory,
        "messages": agent.messages,
        "reward": last_reward,
        "metrics": metrics,
    }
