"""Prepare KernelBench data for rllm + KernelGYM training.

Downloads the KernelBench dataset from HuggingFace and converts it into the
JSONL format expected by ``train_kernelgym.py``.

Usage:
    python -m examples.kernelgym.prepare_kernelbench_data

Output (default):
    data/kernelbench_train.jsonl   (level_1 + level_2, 200 problems)
    data/kernelbench_val.jsonl     (level_3,            50 problems)
"""

from __future__ import annotations

import json
import os

os.environ['HF_ENDPOINT']='https://hf-mirror.com'
# os.environ['HTTP_PROXY']='http://127.0.0.1:18080'
# os.environ['HTTPS_PROXY']='http://127.0.0.1:18080'
# os.environ['http_proxy']='http://127.0.0.1:18080'
# os.environ['https_proxy']='http://127.0.0.1:18080'
os.environ['NO_PROXY']='localhost,127.*.*.*,127.0.1.1,127.0.0.1,*.huawei.com,test.huaweisymantec.com,*-dev.huaweicloud.com,*-dev.myhuaweicloud.com,*.athuawei.com,*.chaspark.cn,*.chaspark.com,*.chaspark.net,*.hic.cloud,*.hisilicon.*,*.hisilicon.cn,*.huawei.cn,*.huawei.com,*.huaweimarine.com,*.huaweimossel.*,*.huaweistatic.cn,*.huaweistatic.com,*.hw3static.cn,*.hw3static.com,*.hwht.*,*.hwtelcloud.com,*.hwtrip.*,*.inhuawei.com,*.pinjiantrip.com,*.yinwang.com,*.yw-beta.com,*.yw-partners.com,*acm.chaspark.com,*cn-north-5-console.huaweicloud.com,*cn-north-5.myhuaweicloud.com,*cn-north-6.myhuaweicloud.com,*heds.huaweigsc.com,*irad.huaweigsc.com,*paper.chaspark.com,*papers.chaspark.com,*tool.chaspark.net,10.*,100.10*,100.11*,100.120.*,100.121.*,100.122.*,100.123.*,100.124.*,100.125.*,100.126.*,100.64.*,100.65.*,100.66.*,100.67.*,100.68.*,100.69.*,100.7*,100.8*,100.9*,127.0.0.1*,172.16.*,172.17.*,172.18.*,172.19.*,172.20.*,172.21.*,172.22.*,172.23.*,172.24.*,172.25.*,172.26.*,172.27.*,172.28.*,172.29.*,172.30.*,172.31.*,172.32.*,7.*,his.chaspark.com,wo-dr*.dbankcloud.cn,wo-dr*.dbankcloud.ru,wo.hicloud.com,.huawei.com'
# export NO_PROXY=localhost,127.*.*.*,127.0.1.1,127.0.0.1,*.huawei.com #,test.huaweisymantec.com,*-dev.huaweicloud.com,*-dev.myhuaweicloud.com,*.athuawei.com,*.chaspark.cn,*.chaspark.com,*.chaspark.net,*.hic.cloud,*.hisilicon.*,*.hisilicon.cn,*.huawei.cn,*.huawei.com,*.huaweimarine.com,*.huaweimossel.*,*.huaweistatic.cn,*.huaweistatic.com,*.hw3static.cn,*.hw3static.com,*.hwht.*,*.hwtelcloud.com,*.hwtrip.*,*.inhuawei.com,*.pinjiantrip.com,*.yinwang.com,*.yw-beta.com,*.yw-partners.com,*acm.chaspark.com,*cn-north-5-console.huaweicloud.com,*cn-north-5.myhuaweicloud.com,*cn-north-6.myhuaweicloud.com,*heds.huaweigsc.com,*irad.huaweigsc.com,*paper.chaspark.com,*papers.chaspark.com,*tool.chaspark.net,10.*,100.10*,100.11*,100.120.*,100.121.*,100.122.*,100.123.*,100.124.*,100.125.*,100.126.*,100.64.*,100.65.*,100.66.*,100.67.*,100.68.*,100.69.*,100.7*,100.8*,100.9*,127.0.0.1*,172.16.*,172.17.*,172.18.*,172.19.*,172.20.*,172.21.*,172.22.*,172.23.*,172.24.*,172.25.*,172.26.*,172.27.*,172.28.*,172.29.*,172.30.*,172.31.*,172.32.*,7.*,his.chaspark.com,wo-dr*.dbankcloud.cn,wo-dr*.dbankcloud.ru,wo.hicloud.com,.huawei.com
# export no_proxy=$NO_PROXY

os.environ["RLLM_HOME"]=os.path.abspath(os.path.join(os.path.dirname(__file__),"../../"))      #! 为了方便能读取到Dataset


from pathlib import Path

from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry

# - ``problem_id`` (str): Unique problem identifier.
# - ``reference_code`` (str): PyTorch reference implementation.
# - ``description`` (str, optional): Human-readable problem description.
# - ``entry_point`` (str, optional): Class name to evaluate (default "Model").

def _hf_row_to_record_cudallm(row: dict) -> dict:
    """Convert a HuggingFace row to the JSONL record format."""
    return {
        "task": {
            "problem_id": f"{row['extra_info']['uuid']}_{'+'.join(json.loads(row['extra_info']['ops']))}",
            "reference_code": row["reward_model"]["ground_truth"],
            "description": "",
            "prompt": row['prompt'][0]['content'],
            "entry_point": "Model",
        },
        "backend": "triton",
    }

def _hf_row_to_record_kerben(row: dict) -> dict:
    """Convert a HuggingFace row to the JSONL record format."""
    return {
        "task": {
            "problem_id": f"level{row['level']}_{row['problem_id']}_{row['name']}",
            "reference_code": row["code"],
            "description": "",
            "entry_point": "Model",
        },
        "backend": "cuda",
    }


def prepare_kernelbench_data(
    output_dir: str = "data",
    train_levels: tuple[int, ...] = (1,),
    val_levels: tuple[int, ...] = (1,),
    register: bool = True,
) -> tuple:
    """Download KernelBench and write train/val JSONL files.

    Args:
        output_dir: Directory for output JSONL files.
        train_levels: Levels to include in the training set (default: 1 + 2).
        val_levels: Levels for the validation set (default: 3).
        register: Whether to also register in the rllm DatasetRegistry.

    Returns:
        (train_dataset, val_dataset) rllm Dataset objects.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Download from HuggingFace ─────────────────────────────────────────
    split_map = {1: "level_1", 2: "level_2", 3: "level_3", 4: "level_4"}

    train_records: list[dict] = []
    # for lvl in train_levels:
    #     ds = load_dataset("ScalingIntelligence/KernelBench", split=split_map[lvl])
    #     train_records.extend(_hf_row_to_record(row) for row in ds)
    ds = load_dataset("./hf_dataset/drkernel-rl-data", split="train")
    train_records = [_hf_row_to_record_cudallm(row) for row in ds]

    val_records: list[dict] = []
    for lvl in val_levels:
        ds = load_dataset("./hf_dataset/kernelbench", split=split_map[lvl])
        val_records.extend(_hf_row_to_record_kerben(row) for row in ds)

    print(f"✅ Train records: {len(train_records)}  Val records: {len(val_records)}")

    # ── Write JSONL ───────────────────────────────────────────────────────
    train_path = os.path.join(output_dir, "drkernel_rl_data.jsonl")
    val_path = os.path.join(output_dir, "kernelbench_val.jsonl")

    for path, records in [(train_path, train_records), (val_path, val_records)]:
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"   Written: {path} ({len(records)} records)")

    # ── Register in rllm DatasetRegistry (optional) ───────────────────────
    if register:
        train_dataset = DatasetRegistry.register_dataset(
            "drkernel_rl_data", train_records, "train",
            source="ScalingIntelligence/KernelBench",
            description="Train Dataset from Dr.Kernel",
            category="code",
        )
        val_dataset = DatasetRegistry.register_dataset(
            "drkernel_rl_data", val_records, "test",
            source="ScalingIntelligence/KernelBench",
            description="KernelBench Benchmark Data",
            category="code",
        )
    else:
        from rllm.data.dataset import Dataset
        train_dataset = Dataset(data=train_records, name="drkernel_rl_data", split="train")
        val_dataset = Dataset(data=val_records, name="drkernel_rl_data", split="test")

    return train_dataset, val_dataset


if __name__ == "__main__":
    train_ds, val_ds = prepare_kernelbench_data()
    print(f"\n🎯 Train: {len(train_ds)} examples, Val: {len(val_ds)} examples")