import hydra

from rllm.agents.kernelgym_agent import KernelAgent
from rllm.data.dataset import Dataset, DatasetRegistry
from rllm.environments.kernelgym.kernelgym_env import KernelGymEnv
from rllm.trainer.agent_trainer import AgentTrainer


def _load_or_register(name: str, split: str, fallback_path: str) -> Dataset:
    """Load dataset from registry; if missing, load from file and auto-register.

    The verl training backend requires a ``_verl.parquet`` companion file.
    ``DatasetRegistry.register_dataset`` generates this automatically, whereas
    plain ``Dataset.load_data`` does not.  By auto-registering the JSONL/parquet
    data when the registry entry is absent, we ensure the verl data pipeline
    always has a valid parquet path.
    """
    ds = DatasetRegistry.load_dataset(name, split)
    if ds is not None:
        return ds

    # Fallback: load raw data file and register it so verl parquet is created
    raw = Dataset.load_data(fallback_path)
    ds = DatasetRegistry.register_dataset(
        name,
        raw.get_data(),
        split,
        source="local-file",
        description=f"Auto-registered from {fallback_path}",
        category="code",
    )
    return ds


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer_megatron", version_base=None)
def main(config):
    train_fallback = config.get("data", {}).get("train_files", "data/kernelbench_train.jsonl")
    val_fallback = config.get("data", {}).get("val_files", "data/kernelbench_val.jsonl")

    train_dataset = _load_or_register("kernelbench", "train", train_fallback)
    test_dataset = _load_or_register("kernelbench", "test", val_fallback)

    kernel_cfg = config.get("kernel", {})
    env_args = {
        "kernel_server_url": kernel_cfg.get("server_url", "http://localhost:8000"),
        "max_turns": config.get("rllm", {}).get("agent", {}).get("max_steps", 3),
        "backend": kernel_cfg.get("backend", "cuda"),
        "toolkit": kernel_cfg.get("toolkit", "kernelbench"),
        "backend_adapter": kernel_cfg.get("toolkit", "kernelbench"),
        "use_ray": kernel_cfg.get("use_ray", False),
        "num_correct_trials": kernel_cfg.get("num_correct_trials", 5),
        "num_perf_trials": kernel_cfg.get("num_perf_trials", 100),
        "timeout": kernel_cfg.get("timeout", 300),
        "reward_func_name": kernel_cfg.get("reward_func_name", "calculate_reward_like_kernel"),
        "reward_config": {
            k: v for k, v in {
                "rate_limit": kernel_cfg.get("rate_limit"),
                "max_concurrent": kernel_cfg.get("max_concurrent"),
                "acquire_timeout": kernel_cfg.get("acquire_timeout"),
                "enable_profiling": kernel_cfg.get("enable_profiling"),
                "detect_decoy_kernel": kernel_cfg.get("detect_decoy_kernel"),
                "reference_backend": kernel_cfg.get("reference_backend"),
            }.items() if v is not None
        },
    }

    trainer = AgentTrainer(
        agent_class=KernelAgent,
        env_class=KernelGymEnv,
        agent_args={},
        env_args=env_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
