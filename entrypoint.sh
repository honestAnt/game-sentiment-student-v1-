#!/bin/sh
# 双入口分发：batch = 离线批量推理（infer_batch.py），api = 在线 HTTP 服务（infer_api.py）
# 其余命令原样透传（如 python -c "..." 做环境自检）
if [ $# -eq 0 ]; then
  echo "用法: docker run ... sentiment-infer:v5 batch <infer_batch.py 参数>"
  echo "      docker run -p 8000:8000 ... sentiment-infer:v5 api"
  exit 1
fi
case "$1" in
  api)
    shift
    exec uvicorn infer_api:app --host 0.0.0.0 --port "${PORT:-8000}" "$@"
    ;;
  batch)
    shift
    exec python infer_batch.py "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
