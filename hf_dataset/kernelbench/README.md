---
dataset_info:
  features:
  - name: code
    dtype: string
  - name: level
    dtype: int64
  - name: name
    dtype: string
  - name: problem_id
    dtype: int64
  splits:
  - name: level_1
    num_bytes: 133215
    num_examples: 100
  - name: level_2
    num_bytes: 114519
    num_examples: 100
  - name: level_3
    num_bytes: 177082
    num_examples: 50
  - name: level_4
    num_bytes: 15591
    num_examples: 20
  download_size: 115926
  dataset_size: 440407
configs:
- config_name: default
  data_files:
  - split: level_1
    path: data/level_1-*
  - split: level_2
    path: data/level_2-*
  - split: level_3
    path: data/level_3-*
  - split: level_4
    path: data/level_4-*
---

# KernelBench
A benchmark designed to evaluate the ability of LLMs to generate efficient GPU kernels for optimizing neural network performance

## Version
[07-21-2025] This HF dataset version has been updated to v0.1

## Citation
```bibtex
@misc{ouyang2024kernelbench,
      title={KernelBench: Can LLMs Write GPU Kernels?}, 
      author={Anne Ouyang and Simon Guo and Azalia Mirhoseini},
      year={2024},
      url={https://scalingintelligence.stanford.edu/blogs/kernelbench/}, 
}
```
