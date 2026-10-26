# -*- coding: utf-8 -*-
"""金标评估的模型推理部分（torch 侧），供 evaluate_gold_v6_gates / v7_gates 共用。

与 gold_gates.py 的分工：这里只负责「把记录跑成预测标签」，
指标口径与闸门判定全在 gold_gates.py（纯函数、可单测、无 torch）。
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from infer_batch import apply_conf_fallback, batch_infer, pick_device
from platform_input import derive_content_type, model_input_text, prefix_enabled


def build_texts(records: list[dict], use_prefix: bool) -> list[str]:
    if not use_prefix:
        return [r["text"] for r in records]
    return [
        model_input_text(r["text"], r.get("platform"), derive_content_type(r))
        for r in records
    ]


def predict(model_dir: str, records: list[dict], threshold: float = 0.75,
            prefix_mode: str = "auto", device: str | None = None) -> dict:
    """按现役工作点跑一遍，返回 raw / 一刀切回退 / 负面豁免 三套预测。

    三套都返回是因为工作点对比（v5 一刀切 vs v8 豁免）是历史上反复要用的证据，
    每次重跑模型太贵。
    """
    use_prefix = prefix_enabled(model_dir, prefix_mode)
    device = device or pick_device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    texts = build_texts(records, use_prefix)
    batch_size = 16 if device == "mps" else 64
    raw_preds, confs = batch_infer(texts, model, tokenizer, device, batch_size=batch_size)
    ex_preds, _ = apply_conf_fallback(raw_preds, confs, threshold, neg_exempt=True)
    fb_preds, _ = apply_conf_fallback(raw_preds, confs, threshold, neg_exempt=False)
    del model
    if device == "mps":
        torch.mps.empty_cache()
    return {
        "device": device,
        "use_prefix": use_prefix,
        "raw": raw_preds,
        "fb": fb_preds,
        "ex": ex_preds,
        "conf": confs,
    }
