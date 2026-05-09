# KernelGym + rLLM（Megatron）—— 协同训练快速开始

这个示例在**同一台机器**（共享加速器）上运行 **rLLM** 训练，并使用 **KernelGym** 作为奖励/评估后端。下面章节顺序为：**训练 Docker 镜像** → **KernelGym 服务** → **训练入口脚本**。

---

## 环境准备（Docker 镜像）

你们团队在 **`training_env/`** 下提供了一个自包含的构建目录：

| 路径 | 用途 |
|------|------|
| `training_env/Dockerfile` | 镜像配方（Ascend / CANN / rLLM 栈）。 |
| `training_env/pkgs/` | 构建过程中使用的依赖库**源码**（vendor 方式）。 |
| `training_env/command.txt` | 维护者预设的**完整 `docker build` 命令**（包含路径、标签与参数）。 |
| `training_env/` 中其他文件 | Dockerfile 或构建上下文引用的辅助资源。 |

**构建镜像**

如果你的机器上已经有 **`ascend-cann900-rllm:latest`**，可跳到[各组件运行位置](#各组件运行位置)。

1. 打开 `training_env/command.txt`，按其中内容原样执行命令（工作目录和上下文路径已在其中定义）。
2. 构建结果镜像标签应为 **`ascend-cann900-rllm:latest`**。

镜像准备好后，按集群要求启动任务或交互容器（设备挂载、`ASCEND_*` 环境变量等）。从**前置条件**开始的所有步骤都假设训练主机上已有该运行环境。

---

## 各组件运行位置

```text
rLLM (train_kernelgym_megatron_qwen30b.sh)
  └── Hydra 配置: reward_model.server_url → KernelGym HTTP API
        └── KernelGym (start_all_with_monitor.sh)
              ├── Redis
              ├── FastAPI (evaluate / health / …)
              ├── worker monitor
              └── NPU workers (GPU_DEVICES 中每个条目对应一个)
```

---

## 前置条件

### 1. 代码仓库

训练主机上需要同时具备两个仓库（以下为示例路径）：

| 角色 | 典型路径 |
| ---- | -------- |
| rLLM | `rllm-071/` |
| KernelGym（此分支） | `kernelGym-npu-main/` |

**下文会用到的脚本**

| 步骤 | 脚本 |
| ---- | ---- |
| 生成 `.env` | `kernelGym-npu-main/scripts/auto_configure.sh` |
| 启动服务 | `kernelGym-npu-main/start_all_with_monitor.sh` |
| 启动训练 | `rllm-071/examples/kernelgym/train_kernelgym_megatron_qwen30b.sh` |

### 2. Redis

KernelGym 需要主机上有 **redis-server**（建议同时安装 **redis-cli** 供启动脚本使用）。

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y redis-server redis-tools
```

**说明：**如果 Redis 尚未监听，且 `REDIS_HOST` 为 `localhost` / `127.0.0.1`，`start_all_with_monitor.sh` 可以拉起本地 `redis-server`。但前提仍是系统已安装 `redis-server` 可执行文件。

### 3. 端口对齐（KernelGym API 与训练脚本）

`train_kernelgym_megatron_qwen30b.sh` 中设置了：

```text
reward_model.server_url="http://127.0.0.1:8002"
```

因此，KernelGym 的 `.env` 中 **API_PORT** 必须为 **8002**，除非你同时把两端改为一致的 URL。

**关于 `8000` 端口与 vLLM：**`scripts/auto_configure.sh` 会从候选列表中选取第一个*空闲*端口（默认包含 `8000`–`8009`）。如果 `8000` 已被占用（例如 vLLM OpenAI 服务），会自动跳过。为了与本训练脚本保持稳定一致，建议自动配置后在 `.env` 中**固定 `API_PORT=8002`**（见下方快速开始）。

---

## 快速开始

### 1. 配置 KernelGym（首次运行或端口变更后）

```bash
cd /path/to/kernelGym-npu-main
bash scripts/auto_configure.sh --force
```

打开 `.env` 并至少确认以下项：

```bash
API_HOST=127.0.0.1          # 或你的可访问绑定地址 / 局域网 IP
API_PORT=8002               # 必须与 reward_model.server_url 一致
REDIS_HOST=localhost
REDIS_PORT=8001             # 示例；任一空闲端口均可
METRICS_PORT=8003           # 示例
GPU_DEVICES=[0,1,2,3,4,5,6,7]   # 按节点资源调整
```

如果 `auto_configure.sh` 写入了不同端口，可手动编辑 `.env`，或在释放端口/调整脚本候选端口池后再次使用 `--force` 重跑（详见 KernelGym 仓库 `README.md`）。

### 2. 启动 KernelGym（API + monitor + workers + 必要时 Redis）

```bash
cd /path/to/kernelGym-npu-main
chmod +x start_all_with_monitor.sh
./start_all_with_monitor.sh
```

默认日志目录为 `kernelGym-npu-main/logs/`（`api_server.log`、`worker_monitor.log`、`worker_npu_*.log`）。

**常用参数**（见 KernelGym `README.md`）：

```bash
./start_all_with_monitor.sh --force-config      # 使用现有 .env 重新执行 auto_configure
./start_all_with_monitor.sh --use-indexed-ports # 以 PORT0、PORT1、… 作为候选端口
```

### 3. 训练前连通性检查

```bash
curl -sS "http://127.0.0.1:8002/health" | head
curl -sS "http://127.0.0.1:8002/workers/status" | head
```

`/health` 应返回 JSON，`/workers/status` 应返回 worker 条目。若不符合预期，请查看 `logs/` 目录下日志文件。

### 4. 启动训练

`start_all_with_monitor.sh` 会在**后台**启动 KernelGym 相关进程，因此连通性检查通过后，你可以在同一终端（或任意其他终端）启动训练。

```bash
cd /path/to/rllm-071
bash examples/kernelgym/train_kernelgym_megatron_qwen30b.sh
```

该脚本会（重新）启动 Ray，并提交 `python3 -m examples.kernelgym.train_kernelgym`，其中包含 shell 文件定义的 Hydra overrides。

---

## 一行命令回顾

```bash
# KernelGym（后台）
cd /path/to/kernelGym-npu-main && bash scripts/auto_configure.sh --force
# 编辑 .env：API_PORT=8002，并修正 GPU_DEVICES
./start_all_with_monitor.sh

# rLLM 训练
cd /path/to/rllm-071 && bash examples/kernelgym/train_kernelgym_megatron_qwen30b.sh
```
