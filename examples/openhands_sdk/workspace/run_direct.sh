#!/bin/bash
# 直接运行 OpenHands Agent（不通过 Docker）
# 用法: ./run_direct.sh [任务描述文件路径]

set -e

# ============================================================================
# LLM 配置（根据你的实际情况修改）
# ============================================================================

# 示例 1: Anthropic Claude
# export LLM_BASE_URL="https://api.anthropic.com/v1"
# export LLM_API_KEY="sk-ant-api03-your-key"
# export LLM_MODEL="anthropic/claude-sonnet-4-20250514"

# 示例 2: OpenAI
# export LLM_BASE_URL="https://api.openai.com/v1"
# export LLM_API_KEY="sk-your-key"
# export LLM_MODEL="openai/gpt-4o"

# 示例 3: DeepSeek
# export LLM_BASE_URL="https://api.deepseek.com/v1"
# export LLM_API_KEY="sk-your-key"
# export LLM_MODEL="deepseek/deepseek-chat"

# 示例 4: 本地 vLLM
# export LLM_BASE_URL="http://localhost:8000/v1"
# export LLM_API_KEY="EMPTY"
# export LLM_MODEL="openai/your-model"

# 请取消注释并填写你的配置：
export LLM_BASE_URL="https://api.kimi.com/coding/v1"
export LLM_API_KEY="sk-kimi-VdAI9VWur5mGEE801IWQZlJ8YqzUN1BS6QNRv0a9A1CTXijkCBSBfrTshFop3cmL"
export LLM_MODEL="openai/kimi-for-coding"

# 禁用 SSL 验证（华为代理使用自签名证书）
export SSL_VERIFY="false"

# ============================================================================
# 工作区配置
# ============================================================================

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORKSPACE_BASE="${SCRIPT_DIR}/agent_workdir"

# ============================================================================
# 任务配置
# ============================================================================

export OPERATOR_BACKEND="ascendc"
export OPERATOR_ARCH="ascend910b1"
export MAX_ITERATIONS="9999"

# 自动推断算子名字：优先从参数传入，其次从工作目录下的 .py 文件推断
if [ -n "$1" ] && [ -f "$1" ]; then
    # 从传入的文件名提取算子名
    OPERATOR_NAME=$(basename "$1" .py)
    echo "[run_direct] 加载任务描述: $1 (算子名: ${OPERATOR_NAME})"
    cp "$1" "${WORKSPACE_BASE}/INSTRUCTIONS.md"
else
    # 从工作目录下查找唯一的 .py 算子文件（排除 __init__.py 等）
    PY_FILES=()
    for f in "${WORKSPACE_BASE}"/*.py; do
        if [ -f "$f" ] && [ "$(basename "$f")" != "__init__.py" ]; then
            PY_FILES+=("$f")
        fi
    done

    if [ ${#PY_FILES[@]} -eq 1 ]; then
        OPERATOR_NAME=$(basename "${PY_FILES[0]}" .py)
        echo "[run_direct] 自动推断算子名: ${OPERATOR_NAME} (文件: ${PY_FILES[0]})"
    elif [ ${#PY_FILES[@]} -gt 1 ]; then
        echo "[run_direct] 警告: 工作目录下有多个 .py 文件，无法自动推断算子名"
        OPERATOR_NAME="operator"
    else
        echo "[run_direct] 警告: 工作目录下没有找到算子 .py 文件，使用默认名字"
        OPERATOR_NAME="operator"
    fi
fi
export OPERATOR_NAME

# 设置输出目录
OUTPUT_DIR="output_${OPERATOR_NAME}"

# 替换 INSTRUCTIONS.md 中的变量占位符
INSTRUCTIONS_MD="${WORKSPACE_BASE}/INSTRUCTIONS.md"
if [ -f "${INSTRUCTIONS_MD}" ]; then
    sed -i "s/{op_name}/${OPERATOR_NAME}/g" "${INSTRUCTIONS_MD}"
    sed -i "s/{output_dir}/${OUTPUT_DIR}/g" "${INSTRUCTIONS_MD}"
    sed -i "s/{backend}/${OPERATOR_BACKEND}/g" "${INSTRUCTIONS_MD}"
    sed -i "s/{arch}/${OPERATOR_ARCH}/g" "${INSTRUCTIONS_MD}"
    echo "[run_direct] 已替换 INSTRUCTIONS.md 中的变量占位符"
fi

# ============================================================================
# 可选配置
# ============================================================================

# 启用 observer 上报到本地网关
export OBSERVER_API_URL="http://127.0.0.1:18859"

# 系统提示词路径
export SYSTEM_PROMPT_PATH="/tmp/rllm_minimal_system.j2"

# ============================================================================
# 验证配置
# ============================================================================

if [ -z "$LLM_BASE_URL" ] || [ -z "$LLM_API_KEY" ] || [ -z "$LLM_MODEL" ]; then
    echo "错误: 请先在脚本中配置 LLM_BASE_URL, LLM_API_KEY, LLM_MODEL"
    echo "编辑 $(basename "$0") 文件，取消注释并填写你的 API 配置"
    exit 1
fi

echo "========================================"
echo "OpenHands Direct Runner"
echo "========================================"
echo "LLM_BASE_URL: ${LLM_BASE_URL}"
echo "LLM_MODEL:    ${LLM_MODEL}"
echo "WORKSPACE:    ${WORKSPACE_BASE}"
echo "OPERATOR:     ${OPERATOR_NAME}"
echo "BACKEND:      ${OPERATOR_BACKEND}"
echo "OUTPUT_DIR:   ${OUTPUT_DIR}"
echo "========================================"

# ============================================================================
# 运行
# ============================================================================

cd "${SCRIPT_DIR}"
python3 -m rllm_entrypoint
