---
license: mit
task_categories:
- text-generation
language:
- en
tags:
- reinforcement-learning
- triton
- kernel-generation
---

# DR.Kernel RL Dataset

[**Paper**](https://huggingface.co/papers/2602.05885) | [**Code**](https://github.com/hkust-nlp/KernelGYM) | [**Dataset**](https://huggingface.co/datasets/hkust-nlp/drkernel-rl-data)

[![Dataset](https://img.shields.io/badge/🤗%20Dataset-hkust--nlp/drkernel--rl--data-yellow)](https://huggingface.co/datasets/hkust-nlp/drkernel-rl-data)

This directory documents the format of `hkust-nlp/drkernel-rl-data`.

Unlike the cold-start SFT set, this RL dataset is primarily a **single-turn query pool** (plus reference metadata) used to launch multi-turn rollouts online in KernelGYM.

## Overview

- Purpose: provide RL training queries + reference code metadata for reward evaluation.
- Data form: one row per optimization task.
- Current local Parquet (`cuda_llm_rl_thinking_1025.parquet`) contains **71,996 rows**.

## Dataset Structure

The file is a Parquet table with the following columns:

| Field | Type | Description |
|---|---|---|
| `data_source` | `string` | Source tag (e.g., `cuda_llm`) |
| `prompt` | `list<struct<role: string, content: string>>` | Chat prompt used for generation (single user turn in this release) |
| `ability` | `string` | Task ability tag (e.g., `kernel_optimization`) |
| `reward_model` | `struct<ground_truth: string, style: string>` | Reward metadata; `ground_truth` is reference PyTorch code |
| `extra_info` | `struct<entry_point, level, module_name, ops, original_prompt, repo_name, type, uuid>` | Auxiliary metadata for rollout/reward tracking |

### Prompt / Reward Format

Each sample typically looks like:

```json
{
  "data_source": "cuda_llm",
  "prompt": [
    {
      "role": "user",
      "content": "You write custom Triton kernels ... Optimize Model -> ModelNew ..."
    }
  ],
  "ability": "kernel_optimization",
  "reward_model": {
    "style": "rule",
    "ground_truth": "import torch
...
class Model(nn.Module): ..."
  },
  "extra_info": {
    "entry_point": "Model",
    "uuid": "cuda_llm_763652",
    "level": "0",
    "module_name": "Model",
    "ops": "[\"torch.abs\", \"nn.Conv2d\"]",
    "original_prompt": [{"role": "user", "content": "..."}],
    "repo_name": "",
    "type": ""
  }
}
```

## How It Is Used in RL

At training time, the model receives `prompt` as initial context, then multi-turn feedback is generated online via KernelGYM:

1. Model generates candidate `ModelNew` code.
2. Kernel reward manager executes and evaluates against `reward_model.ground_truth` (Torch reference code).
3. Feedback (compile/correctness/speed/profiling) is fed back for next turns.
4. TRLOO/MRS/PR/PRS training consumes turn-level rewards.

Notes:

- In this release, `prompt` is single-turn (`len(prompt)=1`, role=`user`).
- Multi-turn trajectories are produced during rollout, not pre-stored in this RL parquet.
- `extra_info.entry_point` is used as the default evaluation entry class/function name.

## Usage

### Load with Hugging Face Datasets

```python
from datasets import load_dataset

ds = load_dataset("hkust-nlp/drkernel-rl-data", split="train")
print(ds.column_names)
# ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']
```

### RL Training with DR.Kernel Scripts

```bash
cd drkernel/kernel/scripts/rl

# 8B
bash 8b_trloo_mrs_pr_prs.sh

# 14B
bash 14b_trloo_mrs_pr_prs.sh
```

Typical dataset settings in RL configs:

```bash
TRAIN_DATASET=("hkust-nlp/drkernel-rl-data")
VALID_DATASET=("hkust-nlp/drkernel-validation-data")
# prompt_key defaults to prompt in trainer config
```

## Local Statistics (from `cuda_llm_rl_thinking_1025.parquet`)

| Metric | Value |
|---|---|
| Rows | 71,996 |
| Prompt list length | 1 for all rows |
| Prompt role pattern | `('user',)` for all rows |
| `ability` | `kernel_optimization` for all rows |
| `reward_model.style` | `rule` for all rows |
| `data_source` | `cuda_llm` for all rows |
| Non-empty `ground_truth` | 71,996 / 71,996 |

Length summary:

- User prompt chars: min 3887, p50 4379, p95 4927, max 8088
- Ground-truth chars: min 242, p50 734, p95 1282, max 4443

## Query Source and Attribution

- The optimization query/task source is based on:
  - [ByteDance-Seed/cudaLLM-data](https://huggingface.co/datasets/ByteDance-Seed/cudaLLM-data)
- We respect and acknowledge the original dataset authors and contributors.
- `hkust-nlp/drkernel-rl-data` focuses on RL-ready packaging and integration metadata (`reward_model`, `extra_info`) for KernelGYM-based training.

## Related Resources

| Resource | Link |
|---|---|
| DR.Kernel Paper | [arXiv:2602.05885](https://huggingface.co/papers/2602.05885) |
| KernelGYM Repo | [GitHub](https://github.com/hkust-nlp/KernelGYM) |
| DR.Kernel Training README | [`drkernel/README.md`](https://github.com/hkust-nlp/KernelGYM/blob/main/drkernel/README.md) |
| KernelGYM Root README | [`README.md`](https://github.com/hkust-nlp/KernelGYM/blob/main/README.md) |
| Query Source Dataset | [ByteDance-Seed/cudaLLM-data](https://huggingface.co/datasets/ByteDance-Seed/cudaLLM-data) |
| Cold-Start SFT Data | [hkust-nlp/drkernel-coldstart-8k](https://huggingface.co/datasets/hkust-nlp/drkernel-coldstart-8k) |
| Validation Data | [hkust-nlp/drkernel-validation-data](https://huggingface.co/datasets/hkust-nlp/drkernel-validation-data) |

## Citation

```bibtex
@article{liuetal2026,
  title={Dr.Kernel: Reinforcement Learning Done Right for Triton Kernel Generations},
  author={Wei Liu, Jiawei Xu, Yingru Li, Longtao Zheng, Tianjian Li, Qian Liu, Junxian He},
  journal={arXiv:2602.05885},
  year={2026}
}
```

## License

MIT License