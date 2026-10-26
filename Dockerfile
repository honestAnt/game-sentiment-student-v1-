# 情感推理镜像（目标机器：RTX 5080 / Blackwell, sm_120；现役 student_v8）
#
# 关键约束：RTX 50 系列是 Blackwell 架构（compute capability 12.0），
# 必须用 CUDA 12.8 构建的 PyTorch（cu128 轮子）。
# 用默认 cu121/cu124 轮子会在推理时报错：
#   RuntimeError: no kernel image is available for execution on the device
#
# 双入口（entrypoint.sh 分发）：
#   batch: python infer_batch.py ...   离线批量推理（CSV -> jsonl）
#   api:    uvicorn infer_api:app      在线 HTTP 服务（POST /predict）
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

# torch cu128（Blackwell 支持）从阿里云 pytorch-wheels 镜像安装（官方源 download.pytorch.org
# 境内直连慢；若镜像未收录，改回 --index-url https://download.pytorch.org/whl/cu128），
# 其余依赖走阿里云 PyPI 镜像。版本与开发环境对齐：transformers 4.46.3 / tokenizers 0.20.3 / pandas 3.0.5。
RUN pip install --no-cache-dir torch==2.7.1+cu128 \
      -i https://mirrors.aliyun.com/pypi/simple/ \
      -f https://mirrors.aliyun.com/pytorch-wheels/cu128/ \
 && pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ \
      transformers==4.46.3 \
      tokenizers==0.20.3 \
      pandas==3.0.5 \
      numpy==1.26.4 \
      fastapi==0.141.1 \
      uvicorn==0.52.4

# 只拷贝推理必需代码；模型和数据不打进镜像，运行时 volume 挂载
# platform_input.py 是 v8 前缀输入的唯一真源（infer_batch/infer_api 均依赖），必须打入
COPY infer_batch.py infer_api.py config.py platform_input.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

# 用法示例（宿主机执行）：
#   批量：docker run --gpus all --rm -v $PWD/models:/app/models -v $PWD/data:/app/data \
#           sentiment-infer:v8 batch \
#           --model_dir models/student_v8 --input data/raw/daily.csv \
#           --output data/predictions/daily.jsonl --review_output data/candidates/lowconf.jsonl \
#           --neg_exempt
#   API： docker run --gpus all -d -p 8000:8000 -v $PWD/models:/app/models \
#           -e MODEL_DIR=models/student_v8 -e NEG_EXEMPT=1 \
#           sentiment-infer:v8 api
ENTRYPOINT ["./entrypoint.sh"]
