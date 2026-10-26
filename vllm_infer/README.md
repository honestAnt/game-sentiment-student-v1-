# Qwen3-8B-AWQ（vLLM）推理服务

在 `gpu-node-example`（RTX 5070 Ti / Blackwell `sm_120`）上以 vLLM 提供 OpenAI 兼容接口（`/v1/chat/completions`）。

## 基础镜像选择

| 方案 | 结论 |
|---|---|
| 偏好：`pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime` + pip 装 vLLM | **不采用**。vLLM 0.27.x 固定 PyTorch **2.13.0**，pip 会替换基础镜像的 torch，且自定义 CUDA 算子与 torch/CUDA 强绑定。 |
| 采用：`vllm/vllm-openai:v0.27.1` | **主方案**。Docker Hub 已核实该 tag 存在（含 `linux/amd64`）。内置匹配的 torch/CUDA/算子；同系列镜像 CUDA 为 13.0.x，适配驱动 595.91.07。 |
| 备选：`vllm/vllm-openai:cu130-nightly` | 文档对 Blackwell 偶有推荐；滚动标签，仅排障时用。 |

官方 HF 权重 ID：[`Qwen/Qwen3-8B-AWQ`](https://huggingface.co/Qwen/Qwen3-8B-AWQ)（已核实存在）。权重**外挂**，不打进镜像。

sm_120 注意：默认 `--quantization awq_marlin`、`--dtype bfloat16`、`--attention-backend TRITON_ATTN`（标准 `awq` 在消费级 Blackwell 上常因 fp16 校验失败）。可与 student_v8 BERT 共卡，故 `GPU_MEMORY_UTILIZATION` 默认 `0.80`。

## 下载模型（节点宿主机）

```bash
# 示例：落到与 embedding 一致的 /data/models 下
mkdir -p /data/models
huggingface-cli download Qwen/Qwen3-8B-AWQ --local-dir /data/models/Qwen3-8B-AWQ
```

## 构建镜像（不要在本任务中 push）

在 `vllm_infer/` 目录：

```bash
docker build --platform=linux/amd64 -f Dockerfile.amd \
  -t your-registry.example.com/ai/vllm:qwen3-8b-awq-v1 .
```

## 本地/节点 docker 运行

```bash
docker run --gpus all --rm -p 8000:8000 \
  --shm-size=8g \
  -v /data/models/Qwen3-8B-AWQ:/models/Qwen3-8B-AWQ:ro \
  -e GPU_MEMORY_UTILIZATION=0.80 \
  -e MAX_MODEL_LEN=8192 \
  your-registry.example.com/ai/vllm:qwen3-8b-awq-v1
```

常用可覆盖 ENV：`MODEL_PATH`、`SERVED_MODEL_NAME`、`PORT`、`MAX_MODEL_LEN`、`GPU_MEMORY_UTILIZATION`、`MAX_NUM_SEQS`、`QUANTIZATION`、`DTYPE`、`ATTENTION_BACKEND`、`REASONING_PARSER`、`ENABLE_REASONING_PARSER`、`VLLM_API_KEY`、`EXTRA_ARGS`。

## Kubernetes

清单：`k8s-deployment.yaml`（Deployment + Service）。

要点：`nodeSelector` → `gpu-node-example`；请求 `nvidia.com/gpu: 1`；容忍 `nvidia.com/gpu`；**不设** `runtimeClassName`；`/dev/shm` 用 `emptyDir`（Memory）；探活 `/health`；模型 `hostPath: /data/models/Qwen3-8B-AWQ`。

```bash
# 确认镜像已推 Harbor 且节点已有权重后：
# kubectl apply -f k8s-deployment.yaml
```

## 冒烟（chat/completions）

```bash
curl -s http://127.0.0.1:8000/health

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-8B-AWQ",
    "messages": [{"role":"user","content":"用一句话介绍你自己。"}],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

若设置了 `VLLM_API_KEY`，请求头加：`-H "Authorization: Bearer <key>"`。
