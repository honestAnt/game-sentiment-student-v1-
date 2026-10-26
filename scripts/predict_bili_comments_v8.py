# -*- coding: utf-8 -*-
"""
scripts/predict_bili_comments_v8.py
对 data/raw 下的 dwd_crawl_bili_video_comments_details.csv 跑 student_v8 批量情感预测。

- 仅预测 bili_video_comment_text 字段（v8 前缀模型，拼 [B站评论] 前缀）
- 设备优先 MPS
- 工作点与线上口径一致：T0.75 低置信回退中性 + 负面豁免（infer_batch.apply_conf_fallback）
- 输出 Excel（openpyxl write_only 流式写，42 万行不会撑爆内存）：
  bili_video_comment_id / bili_video_bvid / data_task_id / 情感结果 / raw 判定 / 置信度

用法：
    .venv/bin/python scripts/predict_bili_comments_v8.py \
        --model_dir models/student_v8 \
        --input data/raw/dwd_crawl_bili_video_comments_details.csv \
        --output data/predictions/dwd_crawl_bili_video_comments_details__student_v8.xlsx
"""

import argparse
import os
import sys
import time

import pandas as pd
import torch
from openpyxl import Workbook
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ID2LABEL, MAX_SEQ_LEN
from infer_batch import apply_conf_fallback, pick_device
from platform_input import model_input_text

USECOLS = [
    "bili_video_comment_id",
    "bili_video_bvid",
    "data_task_id",
    "bili_video_comment_text",
]
CONF_THRESHOLD = 0.75


def batch_infer_sorted(texts, model, tokenizer, device, batch_size):
    """按文本长度排序分批（padding 最小化），推理后恢复原顺序。"""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    labels = [None] * len(texts)
    confidences = [None] * len(texts)
    n = len(order)
    t0 = time.time()
    for bi in range(0, n, batch_size):
        idx = order[bi:bi + batch_size]
        inputs = tokenizer(
            [texts[i] for i in idx],
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)
        conf, preds = probs.max(dim=-1)
        for j, i in enumerate(idx):
            labels[i] = ID2LABEL[preds[j].item()]
            confidences[i] = conf[j].item()
        done = min(bi + batch_size, n)
        if bi == 0 or done == n or (bi // batch_size) % 50 == 0:
            rate = done / (time.time() - t0)
            print(f"  scored {done}/{n}  ({rate:.0f} 条/s)", flush=True)
    return labels, confidences


def write_xlsx(path, frame, header):
    wb = Workbook(write_only=True)
    ws = wb.create_sheet("predictions")
    ws.append(header)
    for row in frame:
        ws.append(row)
    wb.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models/student_v8")
    parser.add_argument("--input", default="data/raw/dwd_crawl_bili_video_comments_details.csv")
    parser.add_argument("--output",
                        default="data/predictions/dwd_crawl_bili_video_comments_details__student_v8.xlsx")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--conf_threshold", type=float, default=CONF_THRESHOLD)
    args = parser.parse_args()

    device = pick_device()
    print(f"[推理设备] {device}  batch={args.batch_size}")
    if device != "mps":
        print("[警告] 未用到 MPS，速度会明显变慢")

    df = pd.read_csv(args.input, usecols=USECOLS, dtype={
        "bili_video_comment_id": str,
        "bili_video_bvid": str,
        "data_task_id": "Int64",
    })
    print(f"[待处理数据量] {len(df)} 条")

    texts = [
        model_input_text(t, "bilibili", "评论")
        for t in df["bili_video_comment_text"].fillna("").astype(str)
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    labels, confidences = batch_infer_sorted(
        texts, model, tokenizer, device, args.batch_size
    )

    final, fallback_mask = apply_conf_fallback(
        labels, confidences, args.conf_threshold, neg_exempt=True
    )
    print(f"[置信度回退] 阈值 {args.conf_threshold}（负面豁免），"
          f"改判中性 {sum(fallback_mask)} 条 ({sum(fallback_mask)/len(df):.2%})")

    dist = pd.Series(final).value_counts()
    print("[情感分布]")
    for k, v in dist.items():
        print(f"  {k}: {v} ({v/len(df):.2%})")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    rows = zip(
        df["bili_video_comment_id"],
        df["bili_video_bvid"],
        df["data_task_id"].astype(object).where(df["data_task_id"].notna(), None),
        final,
        labels,
        confidences,
    )
    write_xlsx(
        args.output,
        rows,
        ["bili_video_comment_id", "bili_video_bvid", "data_task_id",
         "predicted_sentiment", "raw_sentiment", "confidence"],
    )
    print(f"[结果已写入] {args.output}")


if __name__ == "__main__":
    main()
