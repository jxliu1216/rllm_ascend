import hydra
import os
import re
import hashlib
from datetime import datetime, timezone
from omegaconf import open_dict
os.environ["RLLM_HOME"]=os.path.abspath(os.path.join(os.path.dirname(__file__),"../../"))      #! 为了方便能读取到Dataset

from rllm.agents.kernelgym_agent import KernelAgent
from rllm.data.dataset import Dataset, DatasetRegistry
from rllm.environments.kernelgym.kernelgym_env import KernelGymEnv
from rllm.trainer.agent_trainer import AgentTrainer


def _slug(s: str) -> str:
    s = (s or "unknown").strip()
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s or "unknown"


def _build_train_id(config, train_dataset: Dataset, val_dataset: Dataset) -> tuple[str, str, dict]:
    started_at = datetime.now(timezone.utc)
    started_at_iso = started_at.isoformat()
    started_at_compact = started_at.strftime("%Y%m%dT%H%M%SZ")

    train_file = str(config.get("data", {}).get("train_files", "") or "")
    val_file = str(config.get("data", {}).get("val_files", "") or "")
    train_name = train_dataset.name or os.path.basename(train_file) or "train"
    val_name = val_dataset.name or os.path.basename(val_file) or "val"

    dataset_tag = f"{_slug(train_name)}-vs-{_slug(val_name)}"
    dataset_fingerprint = hashlib.sha1(f"{train_file}|{val_file}".encode("utf-8")).hexdigest()[:8]
    train_id = f"kgtrain_{started_at_compact}_{dataset_tag}_{dataset_fingerprint}"
    dataset_meta = {
        "train_name": train_name,
        "train_path": train_file,
        "val_name": val_name,
        "val_path": val_file,
        "dataset_tag": dataset_tag,
        "dataset_fingerprint": dataset_fingerprint,
    }
    return train_id, started_at_iso, dataset_meta


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
    train_id, train_started_at, train_dataset_meta = _build_train_id(config, train_dataset, test_dataset)

    with open_dict(config):
        config.train_id = train_id
        if "reward_model" not in config or config.reward_model is None:
            config.reward_model = {}
        config.reward_model.train_id = train_id
        config.reward_model.train_started_at = train_started_at
        config.reward_model.train_dataset = train_dataset_meta

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
