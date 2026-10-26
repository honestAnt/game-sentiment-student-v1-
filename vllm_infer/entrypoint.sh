#!/usr/bin/env bash
# vLLM OpenAI 兼容服务启动脚本：所有关键参数均可由 ENV 覆盖（docker -e / kubectl env）。
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/Qwen3-8B-AWQ}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-8B-AWQ}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
QUANTIZATION="${QUANTIZATION:-awq_marlin}"
DTYPE="${DTYPE:-bfloat16}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRITON_ATTN}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
ENABLE_REASONING_PARSER="${ENABLE_REASONING_PARSER:-1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
# 可选 API Key：非空则传给 vllm --api-key
VLLM_API_KEY="${VLLM_API_KEY:-}"
# 追加参数（空格分隔），例如：EXTRA_ARGS="--enforce-eager"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "错误: MODEL_PATH 不存在或不是目录: ${MODEL_PATH}" >&2
  echo "请将 Qwen/Qwen3-8B-AWQ 权重挂载到该路径（勿打进镜像）。" >&2
  exit 1
fi

args=(
  serve "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --dtype "${DTYPE}"
)

# 量化：sm_120 上标准 awq 常因强制 fp16 失败，默认 awq_marlin；设 QUANTIZATION= 空可关闭显式指定
if [[ -n "${QUANTIZATION}" ]]; then
  args+=(--quantization "${QUANTIZATION}")
fi

# Blackwell 消费卡（sm_120）现场报告：TRITON_ATTN + awq_marlin 可用；空则交给 vLLM 自动选
if [[ -n "${ATTENTION_BACKEND}" ]]; then
  args+=(--attention-backend "${ATTENTION_BACKEND}")
fi

if [[ "${ENABLE_REASONING_PARSER}" == "1" || "${ENABLE_REASONING_PARSER}" == "true" ]]; then
  # vLLM >=0.9 支持 --reasoning-parser qwen3（0.27.1 已包含）
  args+=(--reasoning-parser "${REASONING_PARSER}")
fi

if [[ "${TRUST_REMOTE_CODE}" == "1" || "${TRUST_REMOTE_CODE}" == "true" ]]; then
  args+=(--trust-remote-code)
fi

if [[ -n "${VLLM_API_KEY}" ]]; then
  args+=(--api-key "${VLLM_API_KEY}")
fi

# 透传容器额外参数（若镜像 ENTRYPOINT 被直接附加 CLI）
if [[ $# -gt 0 ]]; then
  args+=("$@")
fi

# EXTRA_ARGS：字符串拆成词（仅用于简单开关；复杂引号请改直接传 "$@"）
if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra_arr=(${EXTRA_ARGS})
  args+=("${extra_arr[@]}")
fi

echo "启动 vLLM: model=${MODEL_PATH} served_name=${SERVED_MODEL_NAME} port=${PORT} quant=${QUANTIZATION} gpu_mem=${GPU_MEMORY_UTILIZATION} max_len=${MAX_MODEL_LEN}"
exec vllm "${args[@]}"
