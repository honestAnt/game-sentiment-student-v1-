# -*- coding: utf-8 -*-
"""
merge_gold_weibo_comments.py
回收人工标注完成的金标扩样 xlsx，合并进 gold_eval_v5 产出 gold_eval_v6。

标签归一化沿用 prepare_validation_gold.py 的映射（中立->中性 / 负向->负面 / 正向->正面）。
一次可传多个 xlsx，从 v5 一次写出 v6（禁止连跑两次覆盖）。

用法：
    python merge_gold_weibo_comments.py --labeled_xlsx \\
        data/candidates/gold_expansion_weibo_comment/lol_candidates_220.xlsx \\
        data/candidates/gold_expansion_weibo_comment/starrocks_candidates_220.xlsx

产出：data/validation/gold_eval_v6.jsonl（gold_eval_v5 原行不动 + 新行追加，id 带 ge6_ 前缀）
"""

import argparse
import json
import os
from collections import Counter

import pandas as pd

from config import LABEL2ID
from prepare_validation_gold import RAW_LABEL_NORMALIZE

# v5 已并入 v6/v7 后从仓库删除；若需重跑历史合并，用 git 取回 gold_eval_v5.jsonl。
GOLD_V5 = os.path.join("data", "validation", "gold_eval_v5.jsonl")
GOLD_V6 = os.path.join("data", "validation", "gold_eval_v6.jsonl")

TEXT_COL = "文本"
TYPE_COL = "类型"
ID_COL = "序号"
HUMAN_LABEL_COL = "人工标注-杠"

MIN_LABELED = 150   # 低于此数提示标注未完成，不产出 v6


def labeled_records_from_xlsx(path: str) -> tuple[list[dict], list[tuple]]:
    """从单份扩样 xlsx 抽出有效人标记录。空标签跳过；无法归一化进 invalid。"""
    df = pd.read_excel(path)
    recs, invalid = [], []
    for _, row in df.iterrows():
        raw = row.get(HUMAN_LABEL_COL)
        if pd.isna(raw):
            continue
        mapped = RAW_LABEL_NORMALIZE.get(str(raw).strip(), str(raw).strip())
        rid = row[ID_COL]
        if mapped not in LABEL2ID:
            invalid.append((rid, str(raw).strip()))
            continue
        recs.append({
            "raw_id": str(rid),
            "text": str(row[TEXT_COL]),
            "content_type": str(row[TYPE_COL]),
            "gold_label": mapped,
        })
    return recs, invalid


def append_new_gold(gold_rows: list[dict], incoming: list[dict]) -> tuple[list[dict], int, int]:
    """v5 原行原样保留；新行 id=ge6_{序号}。按文本、ge6_id 去重。"""
    gold_texts = {r["text"] for r in gold_rows}
    gold_ids = {r["id"] for r in gold_rows}
    new_rows, dup_text, dup_id = [], 0, 0
    for item in incoming:
        text, rid = item["text"], f"ge6_{item['raw_id']}"
        if text in gold_texts:
            dup_text += 1
            continue
        if rid in gold_ids:
            dup_id += 1
            continue
        row = {
            "id": rid,
            "text": text,
            "platform": "weibo",
            "content_type": item["content_type"],
            "gold_label": item["gold_label"],
        }
        new_rows.append(row)
        gold_texts.add(text)
        gold_ids.add(rid)
    return gold_rows + new_rows, dup_text, dup_id


def merge_gold_from_xlsx_paths(gold_rows: list[dict], xlsx_paths: list[str]) -> tuple[list[dict], dict]:
    incoming, invalid_all = [], []
    unlabeled_files = []
    for p in xlsx_paths:
        df = pd.read_excel(p)
        n_unlabeled = int(df[HUMAN_LABEL_COL].isna().sum()) if HUMAN_LABEL_COL in df.columns else 0
        if n_unlabeled:
            unlabeled_files.append((p, n_unlabeled, len(df)))
        recs, invalid = labeled_records_from_xlsx(p)
        incoming.extend(recs)
        invalid_all.extend(invalid)
    merged, dup_text, dup_id = append_new_gold(gold_rows, incoming)
    stats = {
        "n_labeled": len(incoming),
        "n_new": len(merged) - len(gold_rows),
        "n_invalid": len(invalid_all),
        "invalid": invalid_all,
        "dup_text": dup_text,
        "dup_id": dup_id,
        "unlabeled_files": unlabeled_files,
    }
    return merged, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_xlsx", nargs="+", required=True)
    parser.add_argument("--force", action="store_true", help="标注数不足 MIN_LABELED 时仍强制合并")
    parser.add_argument("--output", default=GOLD_V6)
    args = parser.parse_args()

    gold_rows = [json.loads(l) for l in open(GOLD_V5, encoding="utf-8") if l.strip()]
    merged, stats = merge_gold_from_xlsx_paths(gold_rows, args.labeled_xlsx)

    print(f"[回收] 有效人标 {stats['n_labeled']} 条（{len(args.labeled_xlsx)} 个文件）")
    for p, n_unlab, n_all in stats["unlabeled_files"]:
        print(f"[提示] {p} 有 {n_unlab}/{n_all} 行「人工标注-杠」为空，已跳过")
    if stats["invalid"]:
        print(f"[警告] {stats['n_invalid']} 条标签无法归一化，跳过：")
        for rid, raw in stats["invalid"]:
            print(f"  {rid}\t{raw}")
    if stats["n_labeled"] < MIN_LABELED and not args.force:
        raise SystemExit(f"[中止] 有效标注仅 {stats['n_labeled']} 条 < {MIN_LABELED}，疑似未标完；"
                         f"标完后重跑，或 --force 强制合并")
    if stats["dup_text"] or stats["dup_id"]:
        print(f"[去重] 文本重合跳过 {stats['dup_text']} 条、id 重合跳过 {stats['dup_id']} 条")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    slices = Counter((r["platform"], r["content_type"]) for r in merged)
    labels = Counter(r["gold_label"] for r in merged)
    print(f"[完成] gold_eval_v6 = {len(gold_rows)} 原行 + {stats['n_new']} 新行 = {len(merged)} 条 -> {args.output}")
    for sl, n in sorted(slices.items()):
        print(f"  {sl}: {n}")
    print(f"  标签 {dict(labels)}")


if __name__ == "__main__":
    main()
