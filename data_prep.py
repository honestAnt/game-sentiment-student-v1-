# -*- coding: utf-8 -*-
"""
data_prep.py
从海量原始评论中挑出"值得送去给 LLM 精细标注"的样本，而不是全量标注。

两种模式：
1. 冷启动模式（cold_start）：还没有任何基线模型时，按平台/游戏品类/时间分层随机抽样
2. 主动学习模式（active_learning）：已经有基线模型（哪怕是粗糙的），
   跑一遍全量/大批量数据，把模型判断置信度低的样本挑出来，这些才是最该花标注预算的地方

用法：
    python data_prep.py --mode cold_start --input data/raw/bilibili_comments.csv \
        --output data/candidates/round1_candidates.jsonl --sample_size 30000

    python data_prep.py --mode active_learning --input data/raw/bilibili_comments_new.csv \
        --output data/candidates/round2_candidates.jsonl --sample_size 20000 \
        --baseline_model models/baseline
"""

import argparse
import json
import os
import re

import pandas as pd

from config import LOW_CONFIDENCE_THRESHOLD


def clean_text(text: str) -> str:
    """基础清洗：去 URL、去多余空白、统一表情符号写法。不要做过度清洗，
    很多情感信号就藏在标点、重复字符、emoji 里（比如"???"、"哈哈哈哈哈"）。"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+", "", text)          # 去 URL
    text = re.sub(r"\s+", " ", text).strip()            # 合并空白
    return text


def stratified_cold_start_sample(df: pd.DataFrame, sample_size: int,
                                  strat_cols=("platform", "game_category")) -> pd.DataFrame:
    """冷启动分层抽样：按平台、游戏品类分层，保证覆盖不同来源的语言风格差异。
    如果原始数据没有 game_category 这列，会自动退化为只按 platform 分层。"""
    strat_cols = [c for c in strat_cols if c in df.columns]
    if not strat_cols:
        return df.sample(n=min(sample_size, len(df)), random_state=42)

    groups = df.groupby(strat_cols)
    n_groups = len(groups)
    per_group = max(1, sample_size // n_groups)

    sampled = []
    for _, g in groups:
        sampled.append(g.sample(n=min(per_group, len(g)), random_state=42))
    result = pd.concat(sampled)

    # 如果分层后总量不够，从剩余数据里随机补齐
    if len(result) < sample_size:
        remaining = df.drop(result.index)
        extra_n = min(sample_size - len(result), len(remaining))
        if extra_n > 0:
            result = pd.concat([result, remaining.sample(n=extra_n, random_state=42)])
    return result


def active_learning_sample(df: pd.DataFrame, sample_size: int, baseline_model_path: str) -> pd.DataFrame:
    """用现有基线模型跑一遍数据，挑出置信度低的样本（模型自己也拿不准的）。
    这批样本对提升模型效果的边际收益最大，比随机抽样标注划算得多。"""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(baseline_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(baseline_model_path).to(device)
    model.eval()

    texts = df["text"].tolist()
    confidences = []
    batch_size = 128

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1).values.cpu().tolist()
            confidences.extend(max_probs)

    df = df.copy()
    df["baseline_confidence"] = confidences

    # 低置信度样本优先，同时保留少量高置信度样本做"回归测试"，防止模型在简单样本上退化
    low_conf = df[df["baseline_confidence"] < LOW_CONFIDENCE_THRESHOLD]
    high_conf_check = df[df["baseline_confidence"] >= LOW_CONFIDENCE_THRESHOLD]

    n_low = min(int(sample_size * 0.85), len(low_conf))
    n_high = min(sample_size - n_low, len(high_conf_check))

    result = pd.concat([
        low_conf.sample(n=n_low, random_state=42) if n_low > 0 else low_conf,
        high_conf_check.sample(n=n_high, random_state=42) if n_high > 0 else high_conf_check,
    ])
    print(f"[主动学习采样] 低置信度样本: {n_low} 条, 高置信度回归检查样本: {n_high} 条")
    return result


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """去重：完全相同文本只保留一条，避免刷屏内容（如"沙发""1"）占满标注配额"""
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    print(f"[去重] {before} -> {len(df)}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cold_start", "active_learning"], required=True)
    parser.add_argument("--input", required=True, help="原始 CSV，需包含列: text, platform (可选: game_category, id, timestamp)")
    parser.add_argument("--output", required=True, help="输出 jsonl 路径")
    parser.add_argument("--sample_size", type=int, default=20000)
    parser.add_argument("--baseline_model", default=None, help="active_learning 模式下必填")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    assert "text" in df.columns, "输入数据必须包含 text 列"
    if "id" not in df.columns:
        df["id"] = range(len(df))
    if "platform" not in df.columns:
        df["platform"] = "unknown"

    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() >= 2]   # 过滤空文本/极短噪声（单个标点之类）
    df = dedupe(df)

    if args.mode == "cold_start":
        sampled = stratified_cold_start_sample(df, args.sample_size)
    else:
        assert args.baseline_model, "active_learning 模式必须传 --baseline_model"
        sampled = active_learning_sample(df, args.sample_size, args.baseline_model)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for _, row in sampled.iterrows():
            record = {
                "id": str(row["id"]),
                "text": row["text"],
                "platform": row["platform"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[完成] 共挑选 {len(sampled)} 条候选样本，写入 {args.output}")


if __name__ == "__main__":
    main()
