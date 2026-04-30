from __future__ import annotations

import logging
import re
import os
import sys
import uuid
import time
import random
import json
import ray
import httpx
import omegaconf
from verl.tools.sandbox_fusion_tools import TokenBucketWorker
from typing import Any, Dict, Optional, Tuple, List, Sequence, Callable
from rllm.environments.base.multi_turn_env import MultiTurnEnvironment

logger = logging.getLogger(__name__)


class _HybridHttpWorker:
    def __init__(self, server_url: str, rate_limit: int, default_timeout: int, acquire_timeout: int) -> None:
        self.server_url = server_url
        # print(f"[DEBUG] Default timeout: {default_timeout}")
        self.default_timeout = int(default_timeout)
        self.acquire_timeout = int(acquire_timeout)
        self._limits = httpx.Limits(max_keepalive_connections=64, max_connections=128, keepalive_expiry=30.0)
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=self.default_timeout, write=10.0, pool=5.0),
            limits=self._limits,
            headers={"Content-Type": "application/json"},
            proxy=None,
            trust_env=False,
        )
        # TokenBucketWorker 是一个全局视角的 token 计数器
        # self._rate_limit_worker = TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def _backoff(self, attempt: int, base: int = 2, cap: int = 30) -> float:
        return min(base ** attempt, cap)

    def get_token_in_use(self) -> int:
        # try:
        #     return ray.get(self._rate_limit_worker.get_current_count.remote())
        # except Exception:
        #     return -1
        return 0
    
    def shutdown(self):
        self._client.close()

    def submit_and_poll(self, task_data: Dict[str, Any], client_timeout: int, max_retries: Optional[int]) -> Dict[str, Any]:
        """Submit task and poll for results.

        Args:
            task_data: Task payload including server-side timeout in task_data["timeout"]
            client_timeout: Client-side total timeout including queue wait + execution time
            max_retries: Max retry attempts for submission failures
        """
        start_ts = time.time()
        # Rate-limit only during submission; polling does not consume tokens.
        try:
            # Submit with limited retries: 429/503/timeout/connect errors.
            attempt = 0
            unlimited = max_retries is None or max_retries == -1
            while unlimited or attempt < (max_retries or 0):
                try:
                #     # Acquire token with timeout.
                #     acquire_ref = self._rate_limit_worker.acquire.remote()
                #     ready, _ = ray.wait([acquire_ref], timeout=self.acquire_timeout)
                #     if not ready:
                #         try:
                #             curr = ray.get(self._rate_limit_worker.get_current_count.remote())
                #         except Exception:
                #             curr = -1
                #         print(f"[HybridWorker] acquire timeout tokens_in_use={curr}")
                #         return {"status": "failed", "error_message": "rate limiter acquire timeout"}
                    # Log once on first attempt to help debug "server did not receive request".
                    if attempt == 0:
                        print(f"[HybridWorker] POST /evaluate task_id={task_data.get('task_id', '')} url={self.server_url}")
                    resp = self._client.post(f"{self.server_url}/evaluate", json=task_data)
                    # Log status code to help diagnose non-200 responses.
                    try:
                        print(f"[HybridWorker] POST /evaluate resp={resp.status_code} task_id={task_data.get('task_id','')}")
                    except Exception:
                        pass
                    # Release token immediately after submission.
                    # try:
                    #     self._rate_limit_worker.release.remote()
                    # except Exception:
                    #     pass
                    if resp.status_code == 200:
                        logger.debug(f"[HybridWorker] HTTP 200 OK. errcode(reason): {json.loads(resp.content.decode('utf-8') or '{}').get('error_code',None)}")
                        break
                    if resp.status_code in (429, 503):
                        time.sleep(self._backoff(attempt, base=2 if resp.status_code == 429 else 5))
                        attempt += 1
                        continue
                    resp.raise_for_status()
                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    # try:
                    #     self._rate_limit_worker.release.remote()
                    # except Exception:
                    #     pass
                    if unlimited or attempt < (max_retries or 0) - 1:
                        time.sleep(self._backoff(attempt))
                        attempt += 1
                        continue
                    return {"status": "failed", "error_message": str(e)}
                except Exception as e:
                    # try:
                    #     self._rate_limit_worker.release.remote()
                    # except Exception:
                    #     pass
                    return {"status": "failed", "error_message": str(e)}

            # Poll status at a fixed 1s interval.
            task_id = task_data.get("task_id", "")
            last_status = None
            while time.time() - start_ts < client_timeout:
                try:
                    s = self._client.get(f"{self.server_url}/status/{task_id}")
                    if s.status_code == 200:
                        data = s.json()
                        status = data.get("status", "unknown")
                        if status != last_status:
                            last_status = status
                            try:
                                print(f"[HybridWorker] STATUS task_id={task_id} -> {status}")
                            except Exception:
                                pass
                        if status in ("completed", "failed", "timeout", "cancelled"):
                            # TODO. 这里的 completed failed timeout cancelled 分级不正确
                            #! 我发现有许多执行失败的 被标记为 failed 但是模型并不知道哪里出错。
                            #! 因此在这里，我暂时将 completed 这个逻辑注释掉，让每一次报错都能直接透传至 LLM
                            # if status == "completed":
                            r = self._client.get(f"{self.server_url}/results/{task_id}")
                            if r.status_code == 200:
                                result = r.json()
                                result["status"] = status
                                logger.debug(f"[HybridWorker] PULL Result: {result}")
                                return result
                            return {"status": status, "error_message": f"Failed to fetch results: HTTP {r.status_code}"}
                            # return {"status": status, "error_message": data.get("error_message", f"Task {status}")}
                except Exception:
                    pass
                time.sleep(1.0)

            return {"status": "timeout", "error_message": f"Task timeout after {client_timeout}s (client-side)"}
        finally:
            # No need to release here (already released during submission).
            pass


class KernelGymEnv(MultiTurnEnvironment):
    '''rllm 中使用的 KernelGym 包装(KernelGymEnv) 与 Kernelgym 不同，
    kernelgym中评估kernel是以batch输入的, batch交给一个RewardClient后, 
    由RewardClient分发到多个 Ray Http 客户端上, 而后获得返回。
    在 KernelGymEnv(rllm) 中, 数据是以单条输入的, 因此我们不再使用这种结构, 
    转而直接发送Http请求。
    '''
    def __init__(self, task: dict | None = None, config:omegaconf.DictConfig = omegaconf.DictConfig({}), message_passthrough:bool = False):
        super().__init__(task=task, max_turns=config.max_turns)

        assert task is not None
        self.session_uuid = uuid.uuid4().hex[:16]
        task["task_id"] = task.get("problem_id", "undefined")
        #! 任务相关的输入
        self.problem_id = task.get("task_id")
        self.reference_code = task.get("reference_code", "")
        self.entry_point = task.get("entry_point", "")
        self.is_valid = task.get("is_valid", True)
        # TODO. 检查下这他妈是啥参数
        self.uuid = task.get("problem_id", None)
        self.task = task
        
        # 用于存储每次 get_reward_and_next_obs 返回的 meta_data
        self.meta_info_history = list()
        
        #! 配置信息
        self.config = config
        self.server_url = str(config["server_url"])
        self.timeout = float(config.timeout)
        self.rate_limit = int(config.rate_limit)
        if self.rate_limit <= 0:
            self.rate_limit = 1
        self.acquire_timeout = int(config.acquire_timeout)

        self.task_timeout =  int(getattr(config, 'task_timeout', self.timeout))
        self.task_timeout_in_client = int(getattr(config, 'task_timeout_in_client', self.timeout))
        self.max_retries = config.max_retries

        self.reward_func_name = config.reward_func_name
        self.init_correct_weight = float(config.init_correct_weight)
        self.init_performance_weight = float(config.init_performance_weight)
        self.speedup_eps = float(config.speedup_eps)
        self.penalty_score = float(config.reward_policy.penalties.penalty_score)
        self.speedup_reward_upper_bound = float(config.speedup_reward_upper_bound)
        self.speedup_reward_lower_bound = float(config.speedup_reward_lower_bound)

        self.num_perf_trials = config.num_perf_trials
        self.num_correct_trials = config.num_correct_trials
        self.enable_profiling = config.enable_profiling
        self.verbose_errors = config.verbose_errors
        self.detect_decoy_kernel = config.detect_decoy_kernel
        self.reference_backend = config.reference_backend
        self.train_id = str(getattr(config, "train_id", "") or "")

        self._worker = _HybridHttpWorker(
            self.server_url, self.rate_limit, int(self.timeout), self.acquire_timeout
        )
        
        # 指示消息是否会直接透传至KernelGym
        self.message_passthrough = message_passthrough

        self.logger = logging.getLogger(__name__)


    def calculate_reward_like_kernel(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if result.get("status") != "completed":
            error_message = result.get("error_message", "Task failed")          # TODO. 这里获取的message有问题，总是Task Failed
            if error_message == "Task failed":
                error_message = result.get("error", "Task failed")
            logger.warning(f"[HybridClient] calculate_reward_like_kernel error_message: {error_message}")
            logger.debug(f"[HybridClient] Task failed result: {result}")
            return {
                "reward": -1.0,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "error": error_message,
            }
        # Server returned a decoy kernel; force -1 and carry the marker.
        if result.get("decoy_kernel", False):
            try:
                print("[HybridClient] decoy_kernel detected; forcing reward -1")
            except Exception:
                pass
            return {
                "reward": -1.0,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "decoy_kernel": True,
                "error": "Reward hacking: Decoy kernel detected",
                "score": -1.0,
            }
        correctness = result.get("correctness", False)
        speedup = result.get("speedup", 0.0)
        compiled = result.get("compiled", False)

        penalties = self.config.reward_policy.penalties
        compilation_fail_penalty = float(penalties.get("compilation_fail", -0.5))
        correctness_fail_penalty = float(penalties.get("correctness_fail", -0.3))
        perf_degrade_penalty = float(penalties.get("perf_degrade", -0.1))

        if not compiled:
            reward = compilation_fail_penalty
        elif not correctness:
            reward = correctness_fail_penalty
        else:
            if speedup >= 3.0:
                reward = 1.0
            elif speedup >= 2.0:
                reward = 0.8
            elif speedup >= 1.5:
                reward = 0.6
            elif speedup >= 1.2:
                reward = 0.4
            elif speedup >= 1.0:
                reward = 0.2
            else:
                reward = perf_degrade_penalty
        return {
            "reward": reward,
            "speedup": speedup,
            "success": compiled and correctness,
            "correctness": correctness,
            "compiled": compiled,
            "score": reward,
        }


    def compute_coverage_reward(self, result: Dict[str, Any]) -> Dict[str, Any]:
        # Some server versions put coverage fields in metadata, possibly with plural names; normalize here.
        metadata = result.get("metadata") or {}

        def _get_field(*keys: str, default: int = 0) -> int:
            for k in keys:
                if k in metadata:
                    return metadata.get(k) or default
                if k in result:
                    return result.get(k) or default
            return default

        num_custom_kernel = _get_field("num_custom_kernels", "num_custom_kernel")
        num_total_kernels = _get_field("num_total_kernels", "num_total_kernel", "num_total_kernels")
        custom_kernel_cuda_time_in_profiling_us = _get_field("custom_kernel_cuda_time_in_profiling_us")
        total_kernel_run_time_in_profiling_us = _get_field("total_kernel_run_time_in_profiling_us")

        # Only log keys once when all fields are missing to aid debugging.
        if (
            not num_custom_kernel
            and not num_total_kernels
            and "num_custom_kernel" not in result
            and "num_total_kernels" not in result
            and "num_custom_kernels" not in metadata
            and "num_total_kernels" not in metadata
        ):
            try:
                print(f"[HybridClient] coverage fields missing, fallback to 0: keys={list(result.keys())}")
            except Exception:
                pass

        num_coverage = 0
        if num_total_kernels > 0:
            num_coverage = num_custom_kernel / num_total_kernels


        time_coverage = 0
        if total_kernel_run_time_in_profiling_us > 0:
            time_coverage = custom_kernel_cuda_time_in_profiling_us / total_kernel_run_time_in_profiling_us

        if self.config.coverage_reward.reward_type == "time_coverage":
            coverage = time_coverage
        elif self.config.coverage_reward.reward_type == "number_coverage":
            coverage = num_coverage
        else:
            raise ValueError(f"Invalid reward type: {self.config.coverage_reward.reward_type}")

        return {
            "coverage": coverage,
            "num_custom_kernel": num_custom_kernel,
            "num_total_kernels": num_total_kernels,
            "custom_kernel_cuda_time_in_profiling_us": custom_kernel_cuda_time_in_profiling_us,
            "total_kernel_run_time_in_profiling_us": total_kernel_run_time_in_profiling_us,
        }


    #!!! 可选
    def calculate_reward_weighted(self, result: Dict[str, Any]) -> Dict[str, Any]:
        penalty_score = self.penalty_score

        if result.get("status") != "completed":
            error_message = result.get("error_message", "Task failed")
            if error_message == "Task failed":
                error_message = result.get("error", "Task failed")
            print(f"[HybridClient] calculate_reward_like_kernel error_message: {error_message}")
            logger.debug(f"[HybridClient] Task failed result: {result}")

            return_result = {
                "reward": penalty_score,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "error": error_message,
            }

            for key in result.keys():
                if key not in return_result:
                    return_result[key] = result[key]

            return return_result
        # Server returned a decoy kernel; force penalty and carry the marker.
        # TODO Temporary disable decoy kernel detection
        if result.get("decoy_kernel", False):
            try:
                print("[HybridClient] decoy_kernel detected; forcing reward -1")
            except Exception:
                pass
            return {
                "reward": penalty_score,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "decoy_kernel": True,
                "error": "Reward hacking: Decoy kernel detected",
                "score": penalty_score,
            }
        correctness = result.get("correctness", False)
        speedup = result.get("speedup", 0.0)
        compiled = result.get("compiled", False)
        # In fact, profiling is always None here since it is actually inside metadata
        profiling = result.get("profiling", None) 

        if speedup is None:
            speedup = 0.0

        is_speedup_positive = speedup >= (1 + self.speedup_eps) # ignore too small speedup

        reward = self.init_correct_weight * correctness + self.init_performance_weight * is_speedup_positive

        num_custom_kernel = 0
        num_total_kernels = 0
        custom_kernel_cuda_time_in_profiling_us = 0
        total_kernel_run_time_in_profiling_us = 0
        # if self.reward_config.coverage_reward.enable and correctness:
        final_reward = reward
        if correctness:
            coverage_dict = self.compute_coverage_reward(result)
            coverage = coverage_dict["coverage"]
            num_custom_kernel = coverage_dict["num_custom_kernel"]
            num_total_kernels = coverage_dict["num_total_kernels"]
            custom_kernel_cuda_time_in_profiling_us = coverage_dict["custom_kernel_cuda_time_in_profiling_us"]
            total_kernel_run_time_in_profiling_us = coverage_dict["total_kernel_run_time_in_profiling_us"]
            print(f"[DEBUG] coverage: {coverage}")
            print(f"[DEBUG] num_custom_kernel: {num_custom_kernel}")
            print(f"[DEBUG] num_total_kernels: {num_total_kernels}")
            print(f"[DEBUG] custom_kernel_cuda_time_in_profiling_us: {custom_kernel_cuda_time_in_profiling_us}")
            print(f"[DEBUG] total_kernel_run_time_in_profiling_us: {total_kernel_run_time_in_profiling_us}")
            if self.config.coverage_reward.enable:
                final_reward += self.config.coverage_reward.weight * coverage

        return {
            "reward": final_reward,
            "speedup": speedup,
            "success": compiled and correctness,
            "correctness": correctness,
            "compiled": compiled,
            "score": final_reward,
            "profiling": profiling,
            "num_custom_kernel": num_custom_kernel,
            "num_total_kernels": num_total_kernels,
            "custom_kernel_cuda_time_in_profiling_us": custom_kernel_cuda_time_in_profiling_us,
            "total_kernel_run_time_in_profiling_us": total_kernel_run_time_in_profiling_us,
        }

    #!!! 可选
    def calculate_reward_speedup(self, result: Dict[str, Any]) -> Dict[str, Any]:
        penalty_score = self.penalty_score

        if result.get("status") != "completed":
            
            error_message = result.get("error_message", "Task failed")
            if error_message == "Task failed":
                error_message = result.get("error", "Task failed")
            print(f"[HybridClient] calculate_reward_like_kernel error_message: {error_message}")
            logger.debug(f"[HybridClient] Task failed result: {result}")

            return_result = {
                "reward": penalty_score,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "error": error_message,
            }

            for key in result.keys():
                if key not in return_result:
                    return_result[key] = result[key]

            return return_result
        # Server returned a decoy kernel; force penalty and carry the marker.
        # TODO Temporary disable decoy kernel detection
        if result.get("decoy_kernel", False):
            try:
                print("[HybridClient] decoy_kernel detected; forcing reward -1")
            except Exception:
                pass
            return {
                "reward": penalty_score,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "decoy_kernel": True,
                "error": "Reward hacking: Decoy kernel detected",
                "score": penalty_score,
            }
        correctness = result.get("correctness", False)
        speedup = result.get("speedup", 0.0)
        compiled = result.get("compiled", False)
        # In fact, profiling is always None here since it is actually inside metadata
        profiling = result.get("profiling", None)

        if speedup is None:
            speedup = 0.0

        # is_speedup_positive = speedup >= (1 + self.speedup_eps) # ignore too small speedup

        reward_speedup = speedup
        if speedup > self.speedup_reward_upper_bound:
            reward_speedup = self.speedup_reward_upper_bound
        
        if reward_speedup < self.speedup_reward_lower_bound:
            reward_speedup = 0.0

        reward = self.init_correct_weight * correctness + self.init_performance_weight * reward_speedup

        num_custom_kernel = 0
        num_total_kernels = 0
        custom_kernel_cuda_time_in_profiling_us = 0
        total_kernel_run_time_in_profiling_us = 0

        # if self.reward_config.coverage_reward.enable and correctness:
        final_reward = reward
        if correctness:
            coverage_dict = self.compute_coverage_reward(result)
            coverage = coverage_dict["coverage"]
            num_custom_kernel = coverage_dict["num_custom_kernel"]
            num_total_kernels = coverage_dict["num_total_kernels"]
            custom_kernel_cuda_time_in_profiling_us = coverage_dict["custom_kernel_cuda_time_in_profiling_us"]
            total_kernel_run_time_in_profiling_us = coverage_dict["total_kernel_run_time_in_profiling_us"]

            print(f"[DEBUG] coverage: {coverage}")
            print(f"[DEBUG] num_custom_kernel: {num_custom_kernel}")
            print(f"[DEBUG] num_total_kernels: {num_total_kernels}")
            print(f"[DEBUG] custom_kernel_cuda_time_in_profiling_us: {custom_kernel_cuda_time_in_profiling_us}")
            print(f"[DEBUG] total_kernel_run_time_in_profiling_us: {total_kernel_run_time_in_profiling_us}")

            if self.config.coverage_reward.enable:
                final_reward += self.config.coverage_reward.weight * coverage

        return {
            "reward": final_reward,
            "speedup": speedup,
            "success": compiled and correctness,
            "correctness": correctness,
            "compiled": compiled,
            "score": final_reward,
            "profiling": profiling,
            "num_custom_kernel": num_custom_kernel,
            "num_total_kernels": num_total_kernels,
            "custom_kernel_cuda_time_in_profiling_us": custom_kernel_cuda_time_in_profiling_us,
            "total_kernel_run_time_in_profiling_us": total_kernel_run_time_in_profiling_us,
        }


    def _preflight_validate(self, reference_code: str, kernel_code: str, entry_point: str) -> Tuple[bool, str]:
        """预备检查, 排除没有ModelNew入口的算子"""
        try:
            import regex
            #! 使用正则表达式检查是否有继承自 nn.Module 的 Model 或者 ModelNew 
            ref_required = f"class {entry_point}(nn.Module)"
            ker_required = f"class {entry_point}New(nn.Module)"
            ref_ok = bool(regex.search(fr"class {entry_point}\s*\(.*Module\s*\)", reference_code))
            ker_ok = bool(regex.search(fr"class {entry_point}New\s*\(.*Module\s*\)", kernel_code))
            # ref_ok = ref_required in (reference_code or "")
            # ker_ok = ker_required in (kernel_code or "")
            if ref_ok and ker_ok:
                return True, ""
            missing = []
            if not ref_ok:
                missing.append(ref_required)
            if not ker_ok:
                missing.append(ker_required)
            return False, ", ".join(missing)
        except Exception as e:
            logger.debug(f"preflight skipped due to error: {e}")
            return True, ""


    def _get_reward_func(self) -> Callable[..., Dict[str, Any]]:
        """Select reward function based on config; default to calculate_reward_like_kernel."""
        try:
            func = getattr(self, str(self.reward_func_name), None)
            if callable(func):
                return func # type: ignore
        except Exception:
            pass
        try:
            print(f"[HybridClient] invalid reward_func_name={self.reward_func_name}, fallback to calculate_reward_like_kernel")
        except Exception:
            pass
        return self.calculate_reward_like_kernel


    def compute_reward(self, task: Dict[str, Any], *, use_reference_cache: Optional[bool] = None, **_: Any ) -> Dict[str, Any]:
        penalty_score = self.penalty_score
        
        # !评估超时设定，kernelgym侧，env侧
        effective_timeout = int(self.timeout)
        effective_timeout_in_client = int(self.task_timeout_in_client)
        if effective_timeout_in_client < effective_timeout:     # !需要满足 client timeout >= server timeout
            print(f"[WARNING] task_timeout_in_client ({effective_timeout_in_client}s) < task_timeout ({effective_timeout}s)")
            print(f"[WARNING] Adjusting task_timeout_in_client to match task_timeout to respect timeout invariant")
            effective_timeout_in_client = effective_timeout
        
        kcode = task.get("kernel_code", "")
        ep = task.get("entry_point", "Model")
        ok, missing = self._preflight_validate(task.get("reference_code", ""), kcode, ep)
        if not ok:
            print(f"[HybridClient] preflight failed: missing {missing} entry_point={ep}")
            ret_data = {
                "reward": penalty_score,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "error": f"Client validation failed: missing {missing}",
            }
        else:
            #! 如果一个任务耗时特别长，可能会需要单独的设定保证不会破坏超时限定
            per_task_timeout_raw = task.get("task_timeout", None)
            per_task_timeout = effective_timeout if per_task_timeout_raw is None else int(per_task_timeout_raw)
            per_task_timeout_in_client_raw = task.get("task_timeout_in_client", None)
            per_task_timeout_in_client = effective_timeout_in_client if per_task_timeout_in_client_raw is None else int(per_task_timeout_in_client_raw)
            if per_task_timeout_in_client < per_task_timeout:
                per_task_timeout_in_client = per_task_timeout

            #! 构造请求
            payload = {
                "task_id": task.get("task_id"),
                "train_id": task.get("train_id", self.train_id),
                "reference_code": task.get("reference_code", ""),
                "kernel_code": kcode,
                "backend": self.reference_backend,
                "num_correct_trials": task.get("num_correct_trials", 5),
                "num_perf_trials": task.get("num_perf_trials", 100),
                "timeout": per_task_timeout,
                "priority": "normal",
                "entry_point": ep,
                "is_valid": task.get("is_valid", self.is_valid),                     # 是否需要开启 detect_decoy_kernel 验证
                "verbose_errors": task.get("verbose_errors", True),
                "enable_profiling": task.get("enable_profiling", True),
                "detect_decoy_kernel": task.get("detect_decoy_kernel", True),
                "reference_backend": task.get("reference_backend", None),
                "llm_messages": task.get("llm_messages", "")
            }

            #! 当算子需要验证时，强制开启 detect_decoy_kernel
            if payload["is_valid"]:
                logger.debug(f"Enforce detect decoy kernel if validate: {payload['detect_decoy_kernel']}")
                payload["detect_decoy_kernel"] = True

            # TODO. 还没搞懂啥东西
            ucache = task.get("use_reference_cache", use_reference_cache)
            if ucache:
                payload["use_reference_cache"] = True
                if task.get("uuid"):
                    payload["uuid"] = task["uuid"]
            
            #! 发送请求并获得回复
            ret_data = self._worker.submit_and_poll(payload, per_task_timeout_in_client, self.max_retries)


        #! 调用**奖励函数**对结果进行评估
        reward_func = self._get_reward_func()
        reward_summary = reward_func(ret_data)
        reward_result = {**(ret_data or {}), **(reward_summary or {})}      # 合并结果
        default_reward_result = {
            "reward": penalty_score,
            "speedup": 0.0,
            "success": False,
            "correctness": False,
            "compiled": False,
            "error": "Unknown error",
            "num_custom_kernel": 0,
            "num_total_kernels": 0,
            "custom_kernel_cuda_time_in_profiling_us": 0,
            "total_kernel_run_time_in_profiling_us": 0,
        }  

        return reward_result or default_reward_result # type: ignore[return-value]


    def get_reward_and_next_obs(self, task: dict, action: Any) -> tuple[float, dict]:
        #! 评估
        result = self.compute_reward(task=task, use_reference_cache=False)

        #! 有异常高值时会重新跑一遍 compute_reward
        if result.get("speedup", 0.0) > self.config.speedup_reward_upper_bound:
            print(f"[DEBUG] speedup is anomaly large, re-execute the environment")
            result = self.compute_reward(task=task, use_reference_cache=False)

        #! 剩下的是统计信息
        score = result.get("score", result.get("reward", 0.0))
        num_custom_kernel = result.get("num_custom_kernel", 0)
        num_total_kernels = result.get("num_total_kernels", 0)
        custom_kernel_cuda_time_in_profiling_us = result.get("custom_kernel_cuda_time_in_profiling_us", 0)
        total_kernel_run_time_in_profiling_us = result.get("total_kernel_run_time_in_profiling_us", 0)
        correctness = result.get("correctness", False)
        success = result.get("success", False)
        compiled = result.get("compiled", False)
        speedup = result.get("speedup", 0.0)
        status = result.get("status", "unknown")
        err_msg = result.get("error")

        #! num_coverage/time_coverage 参数处理
        num_coverage = 0
        if num_total_kernels > 0:
            num_coverage = num_custom_kernel / num_total_kernels
        time_coverage = 0
        if total_kernel_run_time_in_profiling_us > 0:
            time_coverage = custom_kernel_cuda_time_in_profiling_us / total_kernel_run_time_in_profiling_us

        meta_info = {
            "server_result": result,            #! 保存 kernelgym 反馈的数据，直接用于 Agent 的输入构造
            "correctness": correctness,
            "performance": speedup,
            "is_speedup_positive": (speedup >= 1.0 + self.config.speedup_eps),
            "is_decoy_kernel": result.get("decoy_kernel", False),
            "compilation": compiled,
            "success": success,
            "error": err_msg,
            "num_custom_kernel": num_custom_kernel,
            "num_total_kernels": num_total_kernels,
            "num_coverage": float(f"{num_coverage:.2f}"),
            "custom_kernel_cuda_time_in_profiling_us": custom_kernel_cuda_time_in_profiling_us,
            "total_kernel_run_time_in_profiling_us": total_kernel_run_time_in_profiling_us,
            "time_coverage": float(f"{time_coverage:.2f}")
        }
        
        print(f"[DEBUG] num_custom_kernel in reward manager: {num_custom_kernel}")
        print(f"[DEBUG] num_total_kernels in reward manager: {num_total_kernels}")
        print(f"[DEBUG] custom_kernel_cuda_time_in_profiling_us in reward manager: {custom_kernel_cuda_time_in_profiling_us}")
        print(f"[DEBUG] total_kernel_run_time_in_profiling_us in reward manager: {total_kernel_run_time_in_profiling_us}")
        
        #! 日志记录
        self.logger.info(f"[KernelEvalStatus] idx={0} status={status} compiled={compiled} correct={correctness} speedup={speedup} uuid={uuid} error={err_msg}")

        return score, meta_info


    def reset(self, task: dict | None = None, seed: int | None = None) -> Tuple[dict, dict]:
        """Reset the environment and return the initial observation."""
        if task is not None:
            self.task = task

        assert self.task is not None, "Task must be set before calling reset()"

        self.done = False
        self.current_turn = 0
        self.history = []
        self._last_error = None
        self._last_result = None

        self.session_uuid = uuid.uuid4().hex[:4]

        return self.task, {}


    def step(self, action: str, global_steps: int = 0) -> Tuple[Dict[str, Any], float, bool, dict]:
        self.history.append(action)

        #! kernelGYM 要求 task_id 为 problem_session_round 的形式，如果错误匹配，可能不会触发校验，直接走缓存。
        task_id = f"{self.task.get('problem_id', 'task')}_{self.session_uuid}_{global_steps}|{self.current_turn}"

        # self.message_passthrough
        llm_messages = ""
        if self.message_passthrough:
            assert "<|message_passtrhough|>" in action
            action, llm_messages = action.split("<|message_passtrhough|>")

        #! 构造 LLM 观测文本，重新构造一遍 task 对象，作为输入
        task = {
            "task_id": task_id,
            "train_id": self.train_id,
            "reference_code": self.reference_code,
            "kernel_code": action,
            "backend": self.reference_backend,
            "entry_point": self.entry_point,
            "use_reference_cache": False,
            "uuid": self.uuid or "",
            "is_valid": self.is_valid,
            "task_timeout": self.task_timeout,
            "task_timeout_in_client": self.task_timeout_in_client,
            "num_correct_trials": self.num_correct_trials,
            "num_perf_trials": self.num_perf_trials,
            "enable_profiling": self.enable_profiling,
            "verbose_errors": self.verbose_errors,
            "detect_decoy_kernel": self.detect_decoy_kernel,
            "reference_backend": self.reference_backend,
            "llm_messages": llm_messages
        }

        reward, meta_info = self.get_reward_and_next_obs(task, action=action)

        #! 观测值来自于 kernelgym 这个 server_result
        next_obs = meta_info["server_result"]

        self.meta_info_history.append(meta_info)
        # # import torch
        # import numpy as np
        # reward = np.random.randint(0,2)*np.random.randn()

        self.current_turn += 1
        if self.current_turn >= self.max_turns:
            self.done = True
            return {}, reward, self.done, self.task

        return next_obs, reward, self.done, self.task

    def close(self):
        self._worker

    @staticmethod
    def from_dict(env_args: dict) -> "KernelGymEnv":
        env_args = dict(env_args)  # avoid mutating caller's dict

        assert "task" in env_args
        task = env_args.pop("task")

        return KernelGymEnv(task=task, config=env_args['reward_config'])

    @staticmethod
    def is_multithread_safe() -> bool:
        """KernelGymEnv is stateless per instance; safe to use across threads."""
        return True
