# -*- coding: utf-8 -*-
"""
export_gold_lol_comment_candidates.py
从本地 LoL 赛事微博评论 CSV 导出 weibo评论 金标扩样候选，交人工标注。

背景（2026-09-15 审计）：gold_eval_v5 的 weibo评论 片仅 66 条（±6 行 = ±9pt，单片否决晋级），
且 66 条全部为 LoL 域（id lol_ 前缀）——本源与之同域同源，扩样不引入分布偏移。
双盲 xlsx 池已枯竭（仅余 15 条，见 export_gold_weibo_comment_candidates.py）；
StarRocks 新拉需环境变量凭证。本地可用且血统清晰的最大池即本 CSV。

源数据：data/raw/英雄联盟赛事__weibo_post_comments_details.csv（108k，tier1 直评 92k）
  + 关联 英雄联盟赛事__weibo_post_details.csv 取主贴内容作标注上下文。

采样规则：tier=1 直评（回复的回复语义碎片多）；文本去重；排除集全量 10 源文本匹配
（docs/数据集对齐配置.md 第三节）；单 weibo_post_id 上限 3 条（防热帖评论相关性堆积）；
随机采样固定 seed。id 用 CSV 原始 id（唯一），回收后带 ge6_ 前缀与训练 id 空间隔离。

用法：
    python export_gold_lol_comment_candidates.py            # 默认 220 条
    python export_gold_lol_comment_candidates.py -n 300

产出：data/candidates/gold_expansion_weibo_comment/
    lol_candidates_<n>.xlsx   人工在「人工标注-杠」列填 中立/负向/正向 或 中性/负面/正面
    lol_candidates_<n>.jsonl  候选行存档

标注完成后用 merge_gold_weibo_comments.py --labeled_xlsx ... 回收合并为 gold_eval_v6.jsonl。
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

OUT_DIR = os.path.join("data", "candidates", "gold_expansion_weibo_comment")
COMMENTS_CSV = os.path.join("data", "raw", "英雄联盟赛事__weibo_post_comments_details.csv")
POSTS_CSV = os.path.join("data", "raw", "英雄联盟赛事__weibo_post_details.csv")

# 排除集：所有进过训练混合集 / 评估集 / 抽检 / 已导出候选的文本（docs 第三节全量口径，
# 按文本字节匹配——round2 本就出自 LoL 池，必须排除）
EXCLUSION_PATTERNS = [
    "data/labeled/round1_labeled.jsonl",
    "data/labeled/round2_labeled.jsonl",
    "data/labeled/round4_labeled.jsonl",
    "data/labeled/round4_train_slice.jsonl",
    "data/labeled/round5_labeled.jsonl",
    "data/labeled/round3_train_mixed.jsonl",
    "data/labeled/round3_train_mixed_os.jsonl",
    "data/labeled/v5_train_mixed.jsonl",
    "data/labeled/v6_train_mixed.jsonl",
    "data/labeled/v7_train_mixed.jsonl",
    "data/labeled/v7_1_train_mixed.jsonl",
    "data/labeled/round3/*.jsonl",
    "data/labeled/round3_full/*.jsonl",
    "data/labeled/round4-claude/*.jsonl",
    "data/validation/validation_gold.jsonl",
    "data/validation/gold_eval_v6.jsonl",
    "data/validation/gold_eval_v7.jsonl",
    "data/validation/gold_bili_neg_challenge.jsonl",
    "data/validation/round4_holdout.jsonl",
    "data/validation/v5_spotcheck_raw200.jsonl",
    "data/validation/v5_spotcheck_raw200_teacher.jsonl",
    "data/validation/v5_spotcheck_raw400.jsonl",
    "data/candidates/gold_expansion_weibo_comment/*.jsonl",
]

TEXT_COL = "文本"
TYPE_COL = "类型"
ID_COL = "序号"
HUMAN_LABEL_COL = "人工标注-杠"
CONTEXT_COL = "主贴内容（评论的上下文）"

MAX_PER_POST = 3
MIN_TEXT_LEN = 2
CONTEXT_TRUNC = 120


def load_excluded_texts() -> set:
    excluded = set()
    for pat in EXCLUSION_PATTERNS:
        for f in glob.glob(pat):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        excluded.add(json.loads(line)["text"])
    return excluded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=220)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--max_per_post", type=int, default=MAX_PER_POST)
    args = parser.parse_args()

    excluded = load_excluded_texts()
    print(f"[排除集] 已用文本 {len(excluded)} 条")

    c = pd.read_csv(COMMENTS_CSV)
    posts = pd.read_csv(POSTS_CSV, usecols=["weibo_post_id", "weibo_post_content_details"])
    post_text = dict(zip(posts["weibo_post_id"], posts["weibo_post_content_details"]))

    n0 = len(c)
    pool = c[c["weibo_post_comment_tier"] == 1].copy()
    n1 = len(pool)
    pool[TEXT_COL] = pool["weibo_post_comment_text"].astype(str).str.strip()
    pool = pool[pool[TEXT_COL].str.len() >= MIN_TEXT_LEN]
    n2 = len(pool)
    pool = pool[~pool[TEXT_COL].isin(excluded)]
    n3 = len(pool)
    pool = pool.drop_duplicates(subset=[TEXT_COL])
    print(f"[候选池] 全量 {n0} -> tier1 {n1} -> 长度过滤 {n2} -> 排除已用 {n3} -> 去重 {len(pool)}")

    rng = np.random.RandomState(args.seed)
    pool = pool.assign(_r=rng.rand(len(pool)))
    capped = (
        pool.sort_values(["weibo_post_id", "_r"])
        .groupby("weibo_post_id", group_keys=False).head(args.max_per_post)
        .drop(columns="_r")
    )
    sampled = capped.sample(n=min(args.num, len(capped)), random_state=args.seed).reset_index(drop=True)
    print(f"[采样] 单贴上限 {args.max_per_post}（贴内随机）后随机取 {len(sampled)} 条（seed {args.seed}）")

    sampled[CONTEXT_COL] = sampled["weibo_post_id"].map(post_text).apply(
        lambda t: (str(t)[:CONTEXT_TRUNC] + "…") if pd.notna(t) and len(str(t)) > CONTEXT_TRUNC
        else ("" if pd.isna(t) else str(t))
    )
    sampled[TYPE_COL] = "评论"

    os.makedirs(OUT_DIR, exist_ok=True)
    xlsx_out = os.path.join(OUT_DIR, f"lol_candidates_{len(sampled)}.xlsx")
    jsonl_out = os.path.join(OUT_DIR, f"lol_candidates_{len(sampled)}.jsonl")

    export = sampled[["id", TYPE_COL, CONTEXT_COL, TEXT_COL]].copy()
    export[HUMAN_LABEL_COL] = None
    export = export.rename(columns={"id": ID_COL})
    export[[ID_COL, TYPE_COL, CONTEXT_COL, TEXT_COL, HUMAN_LABEL_COL]].to_excel(xlsx_out, index=False)

    with open(jsonl_out, "w", encoding="utf-8") as f:
        for _, row in sampled.iterrows():
            f.write(json.dumps({
                "id": f"ge6_{row['id']}",
                "text": row[TEXT_COL],
                "platform": "weibo",
                "content_type": "评论",
                "context": row[CONTEXT_COL],
            }, ensure_ascii=False) + "\n")

    # 与双盲 15 条候选的 id 空间断言（同为 ge6_ 前缀，防回收时撞 id）
    blind_ids = set()
    for f in glob.glob(os.path.join(OUT_DIR, "candidates_*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            blind_ids |= {json.loads(line)["id"] for line in fh if line.strip()}
    new_ids = {f"ge6_{i}" for i in sampled["id"]}
    overlap = blind_ids & new_ids
    assert not overlap, f"与既有候选 id 冲突: {sorted(overlap)[:5]}"

    print(f"[完成] 候选 {len(sampled)} 条已导出：\n  {xlsx_out}（人工在「人工标注-杠」列填写）\n  {jsonl_out}")
    print("[提示] 标注用词：中立/中性、负向/负面、正向/正面 均可，回收脚本会归一化。")
    print("[提示] 标完后回收：python merge_gold_weibo_comments.py --labeled_xlsx <该xlsx>")


if __name__ == "__main__":
    main()
