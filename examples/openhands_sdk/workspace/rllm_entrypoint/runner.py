"""
runner.py — Main conversation execution logic for the rllm entrypoint.

Responsibilities:
  1. Build OpenHands SDK objects (LLM, Agent, Conversation).
  2. Wire the event callback into the Conversation.
  3. Start background pause controller thread.
  4. Run the conversation loop, honoring pause/resume signals.
  5. Emit lifecycle events (startup / heartbeat / evaluate / finish).
  6. Report final metrics and clean up background threads.

v2 (event-driven):
  The StateUploader background thread has been removed.  Instead, each
  OpenHands SDK event is pushed to the gateway immediately inside the
  event callback.  Heartbeats are emitted from a lightweight timer thread
  so the gateway always has a liveness signal even during long LLM calls.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from rllm_entrypoint.api_client import ObserverClient
from rllm_entrypoint.config import cfg
from rllm_entrypoint.events import (
    make_event_callback,
    push_evaluate_event,
    push_finish_event,
    push_heartbeat_event,
    push_startup_event,
)
from rllm_entrypoint.pause_ctrl import PauseController
from rllm_entrypoint.state import AgentPhase, RunState

from openhands.sdk.context import Skill
from openhands.sdk.context.skills import load_project_skills, load_skills_from_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_HARDCODED_SYSTEM_PROMPT = """You are a helpful assistant. Use the provided tools to complete the task. Be concise.
"""


def _write_system_prompt() -> None:
    os.makedirs(os.path.dirname(cfg.system_prompt_path) or "/tmp", exist_ok=True)
    with open(cfg.system_prompt_path, "w", encoding="utf-8") as f:
        f.write(_HARDCODED_SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Workspace skill merging (kept here to avoid monolithic entrypoint)
# ---------------------------------------------------------------------------

def merge_workspace_skills(workspace_base: str, task_scope: Skill=None) -> list:
    """Load only AGENTS.md and skill descriptors (SKILL.md), skip large references."""
    ws = Path(workspace_base)
    skills: list = []

    # Load AGENTS.md as a skill
    agents_md = ws / "AGENTS.md"
    if agents_md.exists():
        try:
            from openhands.sdk.context.skills.skill import Skill
            skill = Skill.load(agents_md)
            skills.append(skill)
        except Exception:
            logger.warning("Failed to load AGENTS.md as skill")

    # Load only SKILL.md from each skill directory, skip references/
    skills_dir = ws / ".agents" / "skills"
    if skills_dir.is_dir():
        for skill_dir in skills_dir.iterdir():
            if skill_dir.is_dir():
                skill_md = skill_dir / "SKILL.md"
                if skill_md.exists():
                    try:
                        from openhands.sdk.context.skills.skill import Skill
                        skill = Skill.load(skill_md)
                        skills.append(skill)
                    except Exception:
                        logger.warning(f"Failed to load skill from {skill_md}")

    return skills


# ---------------------------------------------------------------------------
# Env-var snapshot (redact secrets)
# ---------------------------------------------------------------------------

_SECRET_PATTERNS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")


def _safe_env_snapshot() -> dict[str, str]:
    result = {}
    for k, v in os.environ.items():
        upper = k.upper()
        if any(pat in upper for pat in _SECRET_PATTERNS):
            result[k] = "***REDACTED***"
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Heartbeat timer
# ---------------------------------------------------------------------------

class _HeartbeatTimer:
    """Periodically emits a HeartbeatEvent via push_heartbeat_event().

    Uses a daemon thread so it never blocks program exit.
    """

    def __init__(self, client: ObserverClient, run_state: RunState, interval_s: float = 15.0) -> None:
        self._client = client
        self._state = run_state
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop,
            name="rllm-heartbeat",
            daemon=True,
        )

    def start(self) -> None:
        if self._client._enabled:
            self._thread.start()
            logger.info("[heartbeat] Timer started (interval=%.1fs)", self._interval)

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float = 5.0) -> None:
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _loop(self) -> None:
        while not self._stop.wait(timeout=self._interval):
            try:
                push_heartbeat_event(self._client, self._state)
            except Exception:
                logger.debug("[heartbeat] push failed", exc_info=True)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def build_run_state() -> RunState:
    """Populate a RunState with all static config values."""
    state = RunState(
        session_id=cfg.session_id,
        session_label=cfg.session_label,
        task_instruction=cfg.task_instruction,
        workspace_base=cfg.workspace_base,
        llm_model=cfg.llm_model,
        llm_base_url=cfg.llm_base_url,
        max_iterations=cfg.max_iterations,
        ops_name=cfg.operator_name,
        ops_arch=cfg.operator_arch,
        is_running=False,
        phase=AgentPhase.INITIALIZING,
        env_vars=_safe_env_snapshot(),
        extra_metadata=_parse_extra_metadata(),
        
    )
    return state


def _parse_extra_metadata() -> dict[str, Any]:
    if not cfg.extra_metadata:
        return {}
    try:
        return json.loads(cfg.extra_metadata)
    except json.JSONDecodeError:
        return {"raw": cfg.extra_metadata}


def run() -> int:
    """Build SDK objects, run the conversation, return exit code."""
    from openhands.sdk import LLM, Agent, AgentContext, Conversation
    from openhands.sdk.tool import Tool
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.terminal import TerminalTool

    # ── Validate config ───────────────────────────────────────────────────
    if not cfg.llm_base_url:
        logger.error("LLM_BASE_URL is not set. Exiting.")
        return 1
    if not cfg.task_instruction:
        logger.error("No task instruction provided. Exiting.")
        return 1

    logger.info("LLM_BASE_URL : %s...", cfg.llm_base_url[:80])
    logger.info("LLM_MODEL    : %s", cfg.llm_model)
    logger.info("WORKSPACE    : %s", cfg.workspace_base)
    logger.info("MAX_ITER     : %d", cfg.max_iterations)
    logger.info("SESSION_ID   : %s", cfg.session_id)
    logger.info("TASK         : %.120s", cfg.task_instruction)
    if cfg.observer_api_url:
        logger.info("OBSERVER_URL : %s", cfg.observer_api_url)
    else:
        logger.info("OBSERVER_URL : (disabled — running standalone)")

    # ── Write system prompt ───────────────────────────────────────────────
    _write_system_prompt()

    # ── Initialise RunState ───────────────────────────────────────────────
    run_state = build_run_state()

    # ── Initialise observer client ────────────────────────────────────────
    client = ObserverClient(
        base_url=cfg.observer_api_url,
        session_id=cfg.session_id,
    )

    # ── Push startup lifecycle event ──────────────────────────────────────
    push_startup_event(client, run_state)

    # ── Start heartbeat timer ─────────────────────────────────────────────
    heartbeat = _HeartbeatTimer(
        client=client,
        run_state=run_state,
        interval_s=cfg.upload_interval_s,   # reuse the existing interval config
    )
    heartbeat.start()

    # ── Start pause controller ────────────────────────────────────────────
    pause_ctrl = PauseController(run_state, client, poll_interval_s=cfg.pause_poll_interval_s)
    pause_ctrl.start()

    # ── Build event callback (event-driven push) ──────────────────────────
    _event_cb = make_event_callback(run_state, client)

    # ── Build OpenHands SDK objects ───────────────────────────────────────
    llm = LLM(
        usage_id="rllm-openhands",
        model=cfg.llm_model,
        api_key=SecretStr(cfg.llm_api_key),
        base_url=cfg.llm_base_url or None,
        max_output_tokens=4096,
        extra_headers={"User-Agent": "KimiCLI/1.6"},
        reasoning_effort="none",
        enable_encrypted_reasoning=False,
        native_tool_calling=False,
    )

#     _task_scope = (
#     "You are a Triton-Ascend kernel generation agent. "
#     f"Target architecture: {cfg.operator_backend}. "
#     "Follow AGENTS.md and INSTRUCTIONS.md strictly. "
#     f"Implement ModelNew with @triton.jit kernels in src/{cfg.operat0r_name}_triton_ascend_impl.py. "
#     "All core computation MUST be in Triton kernels — no PyTorch ops in forward(). "
#     f"Do NOT modify tools/. Verify by running: bash tools/operator_pipeline.sh --op_name {cfg.operator_name}. "
#     "Iterate until metrics.json reports success. Summarize results when done."
# )

    # _task_skill = Skill(
    #     name="task_scope",
    #     content=_task_scope,
    #     trigger=None,
    # )

    try:
        _merged_skills = merge_workspace_skills(cfg.workspace_base)
    except Exception:
        logger.exception("merge_workspace_skills failed; falling back to task_scope only")
        _merged_skills = []

    agent_context = AgentContext(
        skills=_merged_skills,
        load_public_skills=False,
        system_message_suffix=(
            f"Workspace directory: {cfg.workspace_base}. "
            f"Maximum iterations budget: {cfg.max_iterations}. "
            f"Session ID: {cfg.session_id}."
        ),
    )

    agent = Agent(
        llm=llm,
        tools=[
            Tool(name=TerminalTool.name),
            Tool(name=FileEditorTool.name),
        ],
        agent_context=agent_context,
        system_prompt_filename=cfg.system_prompt_path,
    )

    conversation = Conversation(
        agent=agent,
        workspace=cfg.workspace_base,
        max_iteration_per_run=cfg.max_iterations,
        callbacks=[_event_cb],
    )

    # Wire pause controller to the live conversation object
    pause_ctrl.set_conversation(conversation)

    # Reflect conversation ID in state
    run_state.set_conversation_status("idle", str(conversation.id))

    # ── Run ───────────────────────────────────────────────────────────────
    exit_code = 0
    run_state.set_running(True)
    run_state.set_phase(AgentPhase.WAITING_FOR_REPLY)

    try:
        conversation.send_message(cfg.task_instruction)

        while True:
            run_state.set_conversation_status(
                conversation.state.execution_status.value,
                str(conversation.id),
            )
            run_state.set_phase(AgentPhase.WAITING_FOR_REPLY)

            conversation.run()

            # Check if resume was requested by pause controller
            if pause_ctrl.resume_requested:
                logger.info("[runner] Resume requested — re-entering conversation.run()")
                pause_ctrl.resume_requested = False
                continue

            # Check conversation terminal status
            exec_status = conversation.state.execution_status.value
            run_state.set_conversation_status(exec_status, str(conversation.id))

            if exec_status in ("finished", "error", "stuck"):
                logger.info("[runner] Conversation reached terminal status: %s", exec_status)
                break

            # Paused but no resume pending: wait for resume signal
            if exec_status == "paused":
                logger.info("[runner] Conversation paused. Waiting for resume signal...")
                while not pause_ctrl.resume_requested and not pause_ctrl._stop_event.is_set():
                    time.sleep(0.5)
                    exec_status = conversation.state.execution_status.value
                    if exec_status not in ("paused",):
                        break
                if pause_ctrl.resume_requested:
                    pause_ctrl.resume_requested = False
                    logger.info("[runner] Resuming after pause...")
                    continue
                break

            # All other (idle / waiting_for_confirmation) → done
            break

        # Log LLM cost
        cost = 0.0
        llm_calls = run_state.total_llm_calls
        if llm.metrics is not None:
            cost = float(llm.metrics.accumulated_cost or 0.0)
            run_state.update_metrics(cost=cost, llm_calls=llm_calls)
            logger.info("EXAMPLE_COST: %s", cost)

        exec_status = conversation.state.execution_status.value
        logger.info("Conversation completed. Status=%s", exec_status)
        run_state.set_phase(AgentPhase.FINISHED)

        # ── Evaluation event ──────────────────────────────────────────────
        push_evaluate_event(
            client,
            run_state,
            status=exec_status,
            cost=cost,
            llm_calls=llm_calls,
            iterations=run_state.iteration,
        )

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        run_state.set_phase(AgentPhase.ERROR)
        run_state.set_error("KeyboardInterrupt")
        exit_code = 1
    except Exception as exc:
        logger.exception("Unhandled exception in runner.")
        run_state.set_phase(AgentPhase.ERROR)
        run_state.set_error(str(exc))
        exit_code = 1
    finally:
        run_state.set_running(False)

        # ── Finish event (always) ─────────────────────────────────────────
        try:
            push_finish_event(
                client,
                run_state,
                exit_code=exit_code,
                reason="error" if exit_code != 0 else "completed",
            )
        except Exception:
            logger.debug("[runner] push_finish_event failed", exc_info=True)

        # ── Stop background threads ───────────────────────────────────────
        heartbeat.stop()
        pause_ctrl.stop()
        heartbeat.join(timeout=5)

    return exit_code
