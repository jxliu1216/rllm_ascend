import hydra
import os
os.environ["RLLM_HOME"]=os.path.abspath(os.path.join(os.path.dirname(__file__),"../../"))      #! 为了方便能读取到Dataset

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

# @hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer_megatron", version_base=None)
def main(config):
    import logging
    logging.basicConfig(level=logging.INFO)
    
    train_fallback = config.get("data", {}).get("train_files", "level_1-00000-of-00001.parquet")
    val_fallback = config.get("data", {}).get("val_files", "level_1-00000-of-00001.parquet")

    train_dataset = _load_or_register("drkernel_rl_data", "train", train_fallback)
    test_dataset = _load_or_register("drkernel_rl_data", "test", val_fallback)

    trainer = AgentTrainer(
        agent_class=KernelAgent,
        env_class=KernelGymEnv,
        agent_args={"message_passthrough": True},       # 是否
        env_args={"reward_config":config.get("reward_model", {}), "message_passthrough": True},
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()



if __name__ == "__main__":
    main()

