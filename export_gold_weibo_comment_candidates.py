# -*- coding: utf-8 -*-
"""
export_gold_weibo_comment_candidates.py
从「微博情感双盲对比（2000条）.xlsx」的未人标行导出 weibo评论 金标扩样候选，交人工标注。

背景（2026-09-15 审计）：gold_eval_v5 的 weibo评论 片仅 66 条（±6 行 = ±9pt，单片否决晋级），
而本地人标池已枯竭（validation_gold 剩余 153 条 weibo评论 全部已在训练集）。
双盲 xlsx 有 1599 条未人标行（评论 1000 / 博文 599），与金标博文同源同分布，
取「类型=评论」行做扩样源。排除一切已用于训练/评估/抽检的文本，防泄漏。

用法：
    python export_gold_weibo_comment_candidates.py            # 默认 200 条
    python export_gold_weibo_comment_candidates.py -n 300

产出：data/candidates/gold_expansion_weibo_comment/
    candidates_<n>.xlsx   人工在「人工标注-杠」列填 中立/负向/正向 或 中性/负面/正面
    candidates_<n>.jsonl  候选行存档（id 带 ge6_ 前缀，与训练 id 空间隔离）

标注完成后用 merge_gold_weibo_comments.py 回收合并为 gold_eval_v6.jsonl。
"""

import argparse
import json
import os

import pandas as pd

GOLD_EXPANSION_DIR = os.path.join("data", "candidates", "gold_expansion_weibo_comment")
BLIND_XLSX = os.path.join("data", "raw", "微博情感双盲对比（2000条）.xlsx")

# 排除集：所有进过训练混合集 / 评估集 / 抽检的文本（按文本字节匹配，稳妥优先于召回）
EXCLUSION_FILES = [
    os.path.join("data", "labeled", f) for f in [
        "round3_train_mixed.jsonl", "round3_train_mixed_os.jsonl", "v5_train_mixed.jsonl",
        "v6_train_mixed.jsonl", "v7_train_mixed.jsonl", "v7_1_train_mixed.jsonl",
        "round5_labeled.jsonl",
    ]
] + [
    os.path.join("data", "validation", f) for f in [
        "validation_gold.jsonl", "gold_eval_v6.jsonl", "gold_eval_v7.jsonl",
        "gold_bili_neg_challenge.jsonl",
        "round4_holdout.jsonl", "v5_spotcheck_raw200.jsonl", "v5_spotcheck_raw200_teacher.jsonl",
        "v5_spotcheck_raw400.jsonl",
    ]
]

TEXT_COL = "文本"
TYPE_COL = "类型"
ID_COL = "序号"
HUMAN_LABEL_COL = "人工标注-杠"
CONTEXT_COL = "主贴内容（评论的上下文）"


def load_excluded_texts() -> set:
    excluded = set()
    for path in EXCLUSION_FILES:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    excluded.add(json.loads(line)["text"])
    return excluded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=200)
    parser.add_argument("--seed", type=int, default=48)
    args = parser.parse_args()

    excluded = load_excluded_texts()
    df = pd.read_excel(BLIND_XLSX)
    pool = df[df[HUMAN_LABEL_COL].isna() & (df[TYPE_COL] == "评论")].copy()
    n0 = len(pool)
    pool = pool[~pool[TEXT_COL].isin(excluded)]
    pool = pool.drop_duplicates(subset=[TEXT_COL])
    print(f"[候选池] 未人标评论 {n0} -> 排除已用文本/去重后 {len(pool)}")

    if len(pool) < args.num:
        print(f"[警告] 候选池仅 {len(pool)} 条，少于请求 {args.num}，全部导出")
    sampled = pool.sample(n=min(args.num, len(pool)), random_state=args.seed)

    os.makedirs(GOLD_EXPANSION_DIR, exist_ok=True)
    xlsx_out = os.path.join(GOLD_EXPANSION_DIR, f"candidates_{len(sampled)}.xlsx")
    jsonl_out = os.path.join(GOLD_EXPANSION_DIR, f"candidates_{len(sampled)}.jsonl")

    out_cols = [ID_COL, TYPE_COL, CONTEXT_COL, TEXT_COL, HUMAN_LABEL_COL]
    export = sampled[out_cols].copy()
    export[HUMAN_LABEL_COL] = None
    export.to_excel(xlsx_out, index=False)

    with open(jsonl_out, "w", encoding="utf-8") as f:
        for _, row in sampled.iterrows():
            rec = {
                "id": f"ge6_{row[ID_COL]}",
                "text": str(row[TEXT_COL]),
                "platform": "weibo",
                "content_type": "评论",
            }
            if pd.notna(row.get(CONTEXT_COL)):
                rec["context"] = str(row[CONTEXT_COL])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[完成] 候选 {len(sampled)} 条已导出：\n  {xlsx_out}（人工在「人工标注-杠」列填写）\n  {jsonl_out}")
    print("[提示] 标注用词：中立/中性、负向/负面、正向/正面 均可，回收脚本会归一化。")


if __name__ == "__main__":
    main()
