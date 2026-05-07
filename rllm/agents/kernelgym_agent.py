"""KernelGYM agent for rllm.

Drives an LLM to iteratively write and refine CUDA/Triton kernels inside the
KernelGymEnv. Formats the initial system + user prompt and appends evaluation
feedback on subsequent turns so the model can revise its kernel implementation.
"""

from __future__ import annotations

import copy
import logging
import re
import json
from typing import Any, Optional

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory

logger = logging.getLogger(__name__)


MESSAGE_PASSTHROUGH_MARKER = "<|message_passthrough|>"


##########
_SYSTEM_PROMPT = """\
You are looking at this PyTorch code and thinking it could be optimized with Triton. You need to create a Triton version with the `ModelNew`. This triton version must be execution on Ascend NPU platforms.

Please firstly analyze this code and think hard how you can optimize it. YOU MUST wrap your final code in a ```triton ... ``` code block. No other code block markers are acceptable.

**Please output and show your thinking, plan,
analysis etc., before your coding, which should be as
more as possible.**

Here's the PyTorch code:

"""

##########
_INITIAL_USER_TEMPLATE = """
```python
{reference_code}
```
"""

##########
_REVISION_USER_TEMPLATE = """\
Now you have received the server feedback for your last implementation. Based on that and all your previous responses, improve the implementation.

Here is the server feedback. Please refer to this feedback to improve the implementation:
Server feedback (status/metrics/errors):
{feedback}

Return an improved Triton implementation named `ModelNew` as a single ```python``` block. Let's think step by step.
"""

# TODO. fix err: "RuntimeError: grid should be less than 65536!"
# TODO. ValueError('Did you forget to add @triton.jit ? (`_builder` argument must be provided outside of JIT functions.)')
# TODO. RuntimeError: ModelNew requires x.device.type == 'ascend'
# TODO. ValueError('program_id axis must be 0, 1, or 2 but got 3')


def extract_reference_code(solution_str: str) -> str:
    """
    从解决方案字符串中提取参考代码
    
    Args:
        solution_str: 包含提示和响应的完整字符串
        
    Returns:
        提取的参考代码
    """
    # 查找参考实现标记
    patterns = [
        r"# Reference Implementation\s*\n(.*?)(?=# Your Task|# Generate|$)",
        r"```python\s*# Reference\s*\n(.*?)```",
        r"# PyTorch Reference:\s*\n(.*?)(?=# Task|# Generate|$)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, solution_str, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # 如果没有找到特定标记，尝试提取第一个 Python 代码块
    code_block_match = re.search(r"```python\s*\n(.*?)```", solution_str, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()
    
    # 回退到整个字符串
    return solution_str


def extract_kernel_code(solution_str: str) -> str:
    """
    从解决方案字符串中提取内核代码
    
    Args:
        solution_str: 包含提示和响应的完整字符串
        
    Returns:
        提取的内核代码
    """
    # 查找内核实现标记
    patterns = [
        r"# Kernel Implementation\s*\n(.*?)(?=# End|$)",
        r"```python\s*# Kernel\s*\n(.*?)```",
        r"# Your implementation:\s*\n(.*?)(?=# End|$)",
        r"# Generated kernel:\s*\n(.*?)(?=# End|$)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, solution_str, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # 如果没有找到特定标记，尝试提取最后一个代码块
    code_blocks = re.findall(r"```(?:\w+)?\s*\n?(.*?)```", solution_str, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()
    
    # 回退：假设整个响应就是内核代码
    return solution_str


class KernelAgent(BaseAgent):
    """Agent that iteratively writes and refines GPU kernels within KernelGymEnv.

    Args:
        accumulate_thinking: If True, keep ``<think>...</think>`` blocks from
            previous turns in the conversation history (default False).
        system_prompt: Custom system prompt. If None, uses the built-in kernel
            engineering prompt.
    """

    def __init__(
        self,
        accumulate_thinking: bool = False,
        system_prompt: Optional[str] = None,
        message_passthrough: bool = False, 
    ):
        self.accumulate_thinking = accumulate_thinking
        self._system_prompt = system_prompt or _SYSTEM_PROMPT
        self._trajectory = Trajectory()
        self.messages: list[dict[str, str]] = []

        self.message_passthrough = message_passthrough

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset agent state at the start of a new episode."""
        self._trajectory = Trajectory()
        self.messages = []

    def update_from_env(
        self,
        observation: Any,
        reward: float,
        done: bool,
        info: dict,
        **kwargs: Any,
    ) -> None:
        # 第一轮，需要初始化
        if not self._trajectory.steps:
            assert isinstance(observation, dict), ("Initial observation must be the task dict.")
            
            #! 构造并初始化
            if not self.messages:
                self.messages.append({"role": "system", "content": self._system_prompt})
            reference_code = observation.get("reference_code", "")
            user_content = _INITIAL_USER_TEMPLATE.format( reference_code=reference_code )
            self.messages.append({"role": "user", "content": user_content})

            new_step = Step(observation=observation)
            self._trajectory.steps.append(new_step)

        else:
            cur_step = self.get_current_state()
            if cur_step is not None:
                cur_step.reward = reward
                cur_step.done = done
                cur_step.info = info

            if done:
                return

            user_content = _REVISION_USER_TEMPLATE.format(feedback=observation)
            self.messages.append({"role": "user", "content": user_content})

            new_step = Step(observation=observation)
            self._trajectory.steps.append(new_step)


    def update_from_model(self, response: str, **kwargs: Any) -> Action:
        self.messages.append({"role": "assistant", "content": response})

        cur_step = self.get_current_state()
        assert cur_step is not None

        cur_step.chat_completions = list(self.chat_completions)
        cur_step.model_response = response

        kernel_code = extract_kernel_code(response)
        #! Patch 1. 直接将没有继承 nn.Module 的 ModelNew 进行替换。
        if "class ModelNew:" in kernel_code:
            kernel_code = kernel_code.replace("class ModelNew:", "class ModelNew(nn.Module):")
        
        action_content = kernel_code.strip()

        if self.message_passthrough:
            action_content += MESSAGE_PASSTHROUGH_MARKER + json.dumps(self.messages)
            pass
        
        action = Action(action=action_content)
        cur_step.action = action

        return action

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        messages = copy.deepcopy(self.messages)
        if not self.accumulate_thinking:
            for msg in messages[:-1]:
                if msg["role"] == "assistant":
                    _, sep, after = msg["content"].partition("</think>")
                    if sep:
                        msg["content"] = after
        return messages

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def get_current_state(self) -> Step | None:
        if not self._trajectory.steps:
            return None
        return self._trajectory.steps[-1]
