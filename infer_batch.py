# -*- coding: utf-8 -*-
"""
infer_batch.py
生产环境批量推理脚本。处理每日增量数据（60万-200万/天量级），
GPU 批处理跑学生模型，输出情感预测结果。

同时会把低置信度样本单独存一份，作为下一轮主动学习的候选池，
形成"生产推理 -> 挖掘难例 -> LLM标注 -> 重新训练"的持续迭代闭环。

用法：
    python infer_batch.py --model_dir models/student_v1 \
        --input data/raw/bilibili_daily_20260908.csv \
        --output data/predictions/20260908_predictions.jsonl \
        --review_output data/candidates/20260908_low_confidence.jsonl

输出列：predicted_sentiment / confidence 之外，另带 p_negative / p_neutral / p_positive
三个 softmax 概率（label id 顺序 0 负面 / 1 中性 / 2 正面）。

置信度回退（--conf_threshold，默认 0.75）：softmax 最高概率低于阈值的样本
改判中性（金标验证：低置信样本 76% 为中性，回退后总 acc +3.8pt 且正面召回不降）。
原始 argmax 结果保留在 raw_sentiment 列供审计，传 0 可关闭回退。

v8 系前缀模型（--prefix，默认 auto=目录名版本号 ≥ v8 自动启用，v9/v10… 同样命中）：输入文本按 platform/content_type
拼受限前缀（platform_input.model_input_text），CSV 有 content_type 列则用之，缺省 [未知]。

非对称回退（--neg_exempt，v8 工作点）：低置信回退豁免负面判定——v5 的低置信带里是错猜，
回退大赚；v8 的低置信带里有大量真负面（gold 85 个金负样本 28 个 <0.75 且判定正确），
一刀切回退会摧毁负召回（0.882->0.647），豁免后 gold 总 0.8495 / 负召回 0.882。
v5 等旧模型勿开（同规则对 v5 是伤害，0.8343->0.8063）。

分片回退候选（--weibo_comment_raw，默认关）：微博评论保持 raw，博文/B站仍走豁免。
gold_eval_v6 上总 acc 0.8310（评论 0.7644），线上 API 需显式 WEIBO_COMMENT_RAW=1 才启用。
"""

import argparse
import json
import os
import time

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import ID2LABEL, LOW_CONFIDENCE_THRESHOLD, MAX_SEQ_LEN, default_infer_batch_size
from platform_input import derive_content_type, model_input_text, prefix_enabled


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def batch_infer(texts, model, tokenizer, device, batch_size=256):
    all_labels, all_confidences = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            confidences, preds = probs.max(dim=-1)
        all_labels.extend([ID2LABEL[p.item()] for p in preds])
        all_confidences.extend(confidences.cpu().tolist())
    return all_labels, all_confidences


def batch_infer_probs(texts, model, tokenizer, device, batch_size=16):
    """与 batch_infer 相同，额外返回每条 [P(负面), P(中性), P(正面)]。"""
    all_labels, all_confidences, all_probs = [], [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            confidences, preds = probs.max(dim=-1)
        all_labels.extend([ID2LABEL[p.item()] for p in preds])
        all_confidences.extend(confidences.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        done = min(i + batch_size, len(texts))
        if i == 0 or done == len(texts) or (i // batch_size) % 32 == 0:
            print(f"  scored {done}/{len(texts)}", flush=True)
    return all_labels, all_confidences, all_probs


def fallback_skip_mask(platforms, content_types, weibo_comment_raw=False):
    """分片回退开关：weibo_comment_raw=True 时仅微博评论跳过回退（B站评论/微博博文仍回退）。

    返回与输入等长的 bool 列表，True = 该行保持 raw。默认全 False，线上口径不变。
    """
    if len(platforms) != len(content_types):
        raise ValueError(
            f"skip mask length mismatch: platforms {len(platforms)} != content_types {len(content_types)}"
        )
    if not weibo_comment_raw:
        return [False] * len(platforms)
    return [p == "weibo" and ct == "评论" for p, ct in zip(platforms, content_types)]


def apply_conf_fallback(labels, confidences, threshold, neg_exempt=False, skip=None):
    """低置信回退到中性；neg_exempt=True 时负面判定豁免（v8 系前缀模型工作点）。

    skip: 可选等长 bool，True 的行不回退（微博评论 raw 工作点）。threshold<=0 时不回退。
    返回 (final_labels, fallback_mask)。纯函数，供批量/在线两条推理路径与单测共用。
    """
    n = len(labels)
    if skip is not None and len(skip) != n:
        raise ValueError(f"skip length {len(skip)} != labels length {n}")
    if threshold <= 0:
        return list(labels), [False] * n
    final, mask = [], []
    for i, (l, c) in enumerate(zip(labels, confidences)):
        if skip is not None and skip[i]:
            mask.append(False)
            final.append(l)
            continue
        hit = c < threshold and not (neg_exempt and l == "负面")
        mask.append(hit)
        final.append("中性" if hit else l)
    return final, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--input", required=True, help="CSV，需包含 text 列，建议也带 id / platform 列")
    parser.add_argument("--output", required=True)
    parser.add_argument("--review_output", required=True, help="低置信度样本，供下一轮主动学习使用")
    parser.add_argument("--conf_threshold", type=float, default=0.75,
                        help="max_prob 低于该值改判中性（0 关闭回退），默认 0.75")
    parser.add_argument("--neg_exempt", action="store_true",
                        help="非对称回退：负面判定不受低置信回退（v8 工作点，旧模型勿开）")
    parser.add_argument("--weibo_comment_raw", action="store_true",
                        help="微博评论不回退、保持 raw（分片工作点候选，默认关，不改线上）")
    parser.add_argument("--prefix", choices=["auto", "on", "off"], default="auto",
                        help="平台前缀输入：auto=模型目录名版本号 ≥ v8 自动启用，off=强制原文")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="推理批大小；默认 cuda=256 / mps=8 / cpu=16（16GB Mac 不要用 256）")
    args = parser.parse_args()

    device = pick_device()
    if args.batch_size is None:
        args.batch_size = default_infer_batch_size(device)
    print(f"[推理设备] {device}  batch={args.batch_size}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()
    if device == "cuda":
        model = model.half()   # FP16 推理，生产环境批量处理下速度和显存都有明显收益

    df = pd.read_csv(args.input)
    if "id" not in df.columns:
        df["id"] = range(len(df))
    if "platform" not in df.columns:
        df["platform"] = "unknown"

    use_prefix = prefix_enabled(args.model_dir, args.prefix)
    print(f"[前缀输入] {args.prefix} -> {'开（model_input_text 平台前缀）' if use_prefix else '关（原文）'}")
    raw_texts = df["text"].fillna("").astype(str).tolist()
    if use_prefix:
        ct_col = df["content_type"] if "content_type" in df.columns else [None] * len(df)
        texts = [
            model_input_text(t, p, None if pd.isna(ct) else str(ct))
            for t, p, ct in zip(raw_texts, df["platform"], ct_col)
        ]
    else:
        texts = raw_texts
    print(f"[待处理数据量] {len(texts)} 条")

    start = time.time()
    labels, confidences, probs = batch_infer_probs(texts, model, tokenizer, device, args.batch_size)
    elapsed = time.time() - start
    throughput = len(texts) / elapsed if elapsed > 0 else 0
    print(f"[推理完成] 耗时 {elapsed:.1f}s，吞吐量约 {throughput:.0f} 条/秒")

    df["predicted_sentiment"] = labels
    df["confidence"] = confidences
    df["p_negative"] = [p[0] for p in probs]
    df["p_neutral"] = [p[1] for p in probs]
    df["p_positive"] = [p[2] for p in probs]

    if args.conf_threshold > 0:
        skip = None
        if args.weibo_comment_raw:
            ct_col = df["content_type"] if "content_type" in df.columns else [None] * len(df)
            src_col = df["source_table"] if "source_table" in df.columns else [None] * len(df)
            cts = [
                derive_content_type({
                    "content_type": None if pd.isna(ct) else ct,
                    "source_table": None if pd.isna(src) else src,
                })
                for ct, src in zip(ct_col, src_col)
            ]
            skip = fallback_skip_mask(df["platform"].tolist(), cts, weibo_comment_raw=True)
        final, fallback_mask = apply_conf_fallback(
            labels, confidences, args.conf_threshold,
            neg_exempt=args.neg_exempt, skip=skip,
        )
        df["raw_sentiment"] = labels
        df["predicted_sentiment"] = final
        extra = "（负面豁免）" if args.neg_exempt else ""
        extra += "（微博评论 raw）" if args.weibo_comment_raw else ""
        print(f"[置信度回退] 阈值 {args.conf_threshold}{extra}，"
              f"改判中性 {sum(fallback_mask)} 条 ({sum(fallback_mask)/len(df):.2%})，"
              f"原始判定保留在 raw_sentiment 列")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_json(args.output, orient="records", lines=True, force_ascii=False)
    print(f"[结果已写入] {args.output}")

    # 挖掘低置信度样本，供下一轮主动学习标注使用
    # 候选池记录原始 argmax 判定（回退后的中性没有参考价值）
    low_conf_df = df[df["confidence"] < LOW_CONFIDENCE_THRESHOLD].copy()
    if "raw_sentiment" in low_conf_df.columns:
        low_conf_df["predicted_sentiment"] = low_conf_df["raw_sentiment"]
    os.makedirs(os.path.dirname(args.review_output), exist_ok=True)
    low_conf_df[["id", "text", "platform", "predicted_sentiment", "confidence"]].to_json(
        args.review_output, orient="records", lines=True, force_ascii=False
    )
    print(f"[低置信度样本] {len(low_conf_df)} 条 ({len(low_conf_df)/len(df):.2%})，已写入 {args.review_output}")
    print("提示：这批样本建议定期（如每周）送入 teacher_labeling.py 标注，滚动更新训练集")


if __name__ == "__main__":
    main()
