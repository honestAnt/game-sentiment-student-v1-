# -*- coding: utf-8 -*-
"""
infer_api.py
在线推理 HTTP 服务：与 infer_batch.py 同一模型、同一置信回退口径（max_prob < 阈值改判中性，
原始判定保留在 raw_sentiment）。适合逐条/小批量的在线调用；大批量离线任务仍建议走 infer_batch.py。

启动（2026-09-15 起现役 v8，默认即前缀 + 负面豁免工作点）：
    uvicorn infer_api:app --host 0.0.0.0 --port 8000

v8 系前缀模型（MODEL_DIR 目录名版本号 ≥ v8 时自动启用前缀，v9/v10… 同样命中；可用 PREFIX=off 强制关闭）：
请求体可选传 platforms / content_types 数组（与 texts 等长）用于拼受限前缀；
缺失时该条按 [未知] 前缀兜底。NEG_EXEMPT=1 为非对称回退（负面判定不回退，v8 工作点，
gold 0.8495 / 负召回 0.882；对 v5 是伤害，回退旧模型须显式 NEG_EXEMPT=0）。
回退 v5 旧口径：MODEL_DIR=models/student_v5 NEG_EXEMPT=0（前缀 auto 关闭）。

接口：
    GET  /health   存活 + 模型/设备/阈值/前缀/豁免信息
    POST /predict  {"texts": ["...", "..."], "threshold": 0.75,
                    "platforms": ["weibo", "bilibili"], "content_types": ["博文", "评论"]}
    每条结果含 predicted_sentiment / raw_sentiment / confidence / fallback_applied，
    以及 probabilities: {"负面": p0, "中性": p1, "正面": p2}（softmax 三分类全量概率）。
"""
import os
import time
from typing import List, Optional

from config import ID2LABEL, MAX_SEQ_LEN, MPS_HIGH_WATERMARK_RATIO, default_infer_batch_size
from platform_input import model_input_text, prefix_enabled

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", str(MPS_HIGH_WATERMARK_RATIO))

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from infer_batch import apply_conf_fallback, fallback_skip_mask, pick_device

MODEL_DIR = os.environ.get("MODEL_DIR", "models/student_v8")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.75"))
MAX_BATCH = int(os.environ.get("MAX_BATCH", "256"))
NEG_EXEMPT = os.environ.get("NEG_EXEMPT", "1") in ("1", "true", "True")
PREFIX_MODE = os.environ.get("PREFIX", "auto")
# 分片工作点候选：默认关，线上仍对微博评论做豁免回退。gold_v6 验证通过后再开。
WEIBO_COMMENT_RAW = os.environ.get("WEIBO_COMMENT_RAW", "0") in ("1", "true", "True")

USE_PREFIX = prefix_enabled(MODEL_DIR, PREFIX_MODE)

device = pick_device()
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", default_infer_batch_size(device)))
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()
if device == "cuda":
    model = model.half()

app = FastAPI(title="game-sentiment-infer")


class PredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=MAX_BATCH)
    threshold: Optional[float] = None
    platforms: Optional[List[str]] = Field(None, description="与 texts 等长，前缀模型用于拼平台前缀")
    content_types: Optional[List[str]] = Field(None, description="与 texts 等长，博文/评论")


def infer(texts: List[str]):
    labels, confidences, all_probs = [], [], []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        labels.extend(ID2LABEL[p.item()] for p in pred)
        confidences.extend(conf.float().cpu().tolist())
        all_probs.extend(probs.float().cpu().tolist())
    return labels, confidences, all_probs


@app.get("/health")
def health():
    return {
        "ok": True,
        "model_dir": MODEL_DIR,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else device,
        "conf_threshold": CONF_THRESHOLD,
        "neg_exempt": NEG_EXEMPT,
        "weibo_comment_raw": WEIBO_COMMENT_RAW,
        "prefix": USE_PREFIX,
        "max_seq_len": MAX_SEQ_LEN,
        "max_batch": MAX_BATCH,
        "batch_size": BATCH_SIZE,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    threshold = req.threshold if req.threshold is not None else CONF_THRESHOLD
    if not 0 <= threshold <= 1:
        raise HTTPException(status_code=400, detail="threshold 需在 [0,1] 区间")
    for name, arr in (("platforms", req.platforms), ("content_types", req.content_types)):
        if arr is not None and len(arr) != len(req.texts):
            raise HTTPException(status_code=400, detail=f"{name} 长度须与 texts 一致")

    texts = [t if t.strip() else "。" for t in req.texts]  # 空串占位，tokenizer 需要 >0 token
    platforms = req.platforms or [None] * len(texts)
    content_types = req.content_types or [None] * len(texts)
    if USE_PREFIX:
        texts = [
            model_input_text(t, p, ct) for t, p, ct in zip(texts, platforms, content_types)
        ]
    start = time.time()
    labels, confidences, all_probs = infer(texts)
    skip = fallback_skip_mask(platforms, content_types, weibo_comment_raw=WEIBO_COMMENT_RAW)
    final_labels, fallback_mask = apply_conf_fallback(
        labels, confidences, threshold, neg_exempt=NEG_EXEMPT, skip=skip
    )
    results = [
        {
            "predicted_sentiment": fl,
            "raw_sentiment": l,
            "confidence": round(c, 4),
            "fallback_applied": m,
            "probabilities": {
                ID2LABEL[i]: round(p, 4) for i, p in enumerate(pr)
            },
        }
        for fl, l, c, m, pr in zip(final_labels, labels, confidences, fallback_mask, all_probs)
    ]
    return {
        "count": len(results),
        "threshold": threshold,
        "elapsed_ms": round((time.time() - start) * 1000, 1),
        "results": results,
    }
