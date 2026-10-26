# bge-m3 Embedding 服务

OpenAI 兼容 `POST /v1/embeddings`，探活 `GET /health` / `GET /ready`。

设计：`docs/superpowers/specs/2026-09-23-bge-m3-embedding-service-design.md`

## 构建并推 Harbor（linux/amd64）

在仓库根目录 `game_sentiment_project`：

```bash
docker buildx build --platform linux/amd64 \
  -f embedding_bge_m3/Dockerfile.amd \
  -t your-registry.example.com/ai/embedding:bge-m3-v1 \
  --push .
```

本机无法 push 时先 `--load`，再在已登录镜像仓库的机器上推送。

## 运行

```bash
docker run --gpus all -p 8000:8000 \
  -v /data/models:/models:ro \
  your-registry.example.com/ai/embedding:bge-m3-v1
```

## 冒烟

```bash
curl -s localhost:8000/ready
curl -s localhost:8000/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{"input":"hello bge-m3"}'
```
