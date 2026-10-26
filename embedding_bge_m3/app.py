# -*- coding: utf-8 -*-
"""
bge-m3 embedding API（OpenAI 兼容子集）。

环境变量：
  MODEL_DIR      模型挂载根，默认 /models
  MODEL_SUBDIR   子目录名，默认 bge-m3；若 MODEL_DIR/MODEL_SUBDIR 存在则用之，
                 否则把 MODEL_DIR 当作模型根（兼容直接挂到权重目录）
  MODEL_NAME     响应里的 model 字段，默认 bge-m3
  MAX_BATCH      单次 input 条数上限，默认 64
  NORMALIZE      1 时 L2 normalize，默认 0
  PORT           由 entrypoint 读取，默认 8000
"""
from __future__ import annotations

import os
from typing import List, Optional, Union

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
MODEL_SUBDIR = os.environ.get("MODEL_SUBDIR", "bge-m3")
MODEL_NAME = os.environ.get("MODEL_NAME", "bge-m3")
MAX_BATCH = int(os.environ.get("MAX_BATCH", "64"))
NORMALIZE = os.environ.get("NORMALIZE", "0") in ("1", "true", "True")


def resolve_model_path(model_dir: str = MODEL_DIR, subdir: str = MODEL_SUBDIR) -> str:
    root = model_dir.rstrip("/")
    candidate = f"{root}/{subdir}" if subdir else root
    if subdir and os.path.isdir(candidate):
        return candidate
    return root


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


MODEL_PATH = resolve_model_path()
DEVICE = pick_device()

app = FastAPI(title="bge-m3-embedding")
_model: Optional[SentenceTransformer] = None
_ready = False
_load_error: Optional[str] = None


def get_model() -> SentenceTransformer:
    if _model is None or not _ready:
        raise HTTPException(status_code=503, detail=_load_error or "model not ready")
    return _model


@app.on_event("startup")
def load_model() -> None:
    global _model, _ready, _load_error
    try:
        if not os.path.isdir(MODEL_PATH):
            raise FileNotFoundError(f"model path not found: {MODEL_PATH}")
        m = SentenceTransformer(MODEL_PATH, device=DEVICE)
        if DEVICE == "cuda":
            # half can speed up; sentence-transformers handles dtype on encode
            pass
        _model = m
        _ready = True
        _load_error = None
    except Exception as e:  # noqa: BLE001 — startup must record any load failure
        _model = None
        _ready = False
        _load_error = str(e)


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = None


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]


class EmbeddingUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


@app.get("/health")
def health():
    return {
        "ok": True,
        "model_dir": MODEL_PATH,
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else DEVICE,
        "ready": _ready,
        "normalize": NORMALIZE,
        "max_batch": MAX_BATCH,
    }


@app.get("/ready")
def ready():
    if not _ready:
        raise HTTPException(status_code=503, detail=_load_error or "model not ready")
    return {"ok": True, "model_dir": MODEL_PATH, "device": DEVICE}


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def embeddings(req: EmbeddingRequest):
    texts: List[str]
    if isinstance(req.input, str):
        texts = [req.input]
    else:
        texts = list(req.input)

    if not texts or any(t is None for t in texts):
        raise HTTPException(status_code=400, detail="input must be a non-empty string or array")
    if len(texts) > MAX_BATCH:
        raise HTTPException(status_code=400, detail=f"input length exceeds MAX_BATCH={MAX_BATCH}")

    model = get_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=NORMALIZE,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    data = [
        EmbeddingData(index=i, embedding=vectors[i].astype(float).tolist())
        for i in range(len(texts))
    ]
    return EmbeddingResponse(
        data=data,
        model=MODEL_NAME,
        usage=EmbeddingUsage(prompt_tokens=0, total_tokens=0),
    )
