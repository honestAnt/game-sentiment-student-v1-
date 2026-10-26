# -*- coding: utf-8 -*-
"""
prepare_validation_gold.py
把"微博情感双盲对比_2000条.xlsx"里真正有人工标注的行整理成标准 jsonl，
作为长期回归测试用的 gold validation set。

只保留"人工标注-杠"列非空的行——文件里 2000 条只有 401 条有真实人工标注，
其余 1599 条不含金标签，不能拿来做验证。

用法：
    python prepare_validation_gold.py \
        --input "/mnt/user-data/uploads/微博情感双盲对比_2000条_.xlsx" \
        --output data/validation/validation_gold.jsonl
"""

import argparse
import json

import pandas as pd

from config import LABEL2ID

# 原始Excel里的标注用词（中立/负向/正向）和项目标签体系（中性/负面/正面）不完全一致，
# 这里做归一化映射，而不是简单假设两边命名一样——这类命名不一致在多人协作标注时很常见，
# 归一化步骤必须显式做，不能默认跳过。
RAW_LABEL_NORMALIZE = {
    "中立": "中性", "中性": "中性",
    "负向": "负面", "负面": "负面",
    "正向": "正面", "正面": "正面",
}

TEXT_COL = "文本"
TYPE_COL = "类型"
ID_COL = "序号"
HUMAN_LABEL_COL = "人工标注-杠"
CONTEXT_COL = "主贴内容（评论的上下文）"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sheet_name", default=None, help="不填则自动取第一个sheet")
    args = parser.parse_args()

    df = pd.read_excel(args.input, sheet_name=args.sheet_name)
    if isinstance(df, dict):  # 没指定 sheet_name 时可能返回多个 sheet
        first_key = list(df.keys())[0]
        print(f"[提示] 未指定sheet，自动使用第一个: {first_key}")
        df = df[first_key]

    total = len(df)
    labeled = df[df[HUMAN_LABEL_COL].notna()].copy()
    print(f"[数据总量] {total} 条，其中有效人工标注 {len(labeled)} 条 ({len(labeled)/total:.1%})")

    # 归一化标签命名（中立->中性, 负向->负面, 正向->正面）
    raw_values = labeled[HUMAN_LABEL_COL].unique().tolist()
    unmapped = [v for v in raw_values if v not in RAW_LABEL_NORMALIZE]
    if unmapped:
        print(f"[警告] 发现无法归一化映射的原始标签值: {unmapped}，这些行会在下一步被跳过并提示")
    labeled[HUMAN_LABEL_COL] = labeled[HUMAN_LABEL_COL].map(
        lambda x: RAW_LABEL_NORMALIZE.get(x, x)
    )

    # 校验标签是否都在我们的标签体系里，不在的话打印出来提醒人工核对，而不是静默丢弃
    invalid = labeled[~labeled[HUMAN_LABEL_COL].isin(LABEL2ID.keys())]
    if len(invalid) > 0:
        print(f"[警告] 有 {len(invalid)} 条标签不在体系 {list(LABEL2ID.keys())} 内，已跳过，请人工核查：")
        print(invalid[[ID_COL, HUMAN_LABEL_COL]].to_string(index=False))
        labeled = labeled[labeled[HUMAN_LABEL_COL].isin(LABEL2ID.keys())]

    records = []
    for _, row in labeled.iterrows():
        record = {
            "id": str(row[ID_COL]),
            "text": str(row[TEXT_COL]),
            "platform": "weibo",
            "content_type": row[TYPE_COL],   # 博文 / 评论，保留下来方便后续分类型统计
            "gold_label": row[HUMAN_LABEL_COL],
        }
        if CONTEXT_COL in row and pd.notna(row[CONTEXT_COL]):
            record["context"] = str(row[CONTEXT_COL])
        records.append(record)

    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[完成] {len(records)} 条已写入 {args.output}")

    # 提醒数据缺口，避免用户误以为这份验证集覆盖全面
    type_counts = labeled[TYPE_COL].value_counts()
    print("\n[内容类型分布]")
    print(type_counts)
    if "评论" not in type_counts.index or type_counts.get("评论", 0) == 0:
        print("\n[重要提醒] 这份验证集里不包含任何'评论'类型的人工标注样本，"
              "只能验证模型在'博文'类型文本上的表现，评论类型仍需单独补充人工标注。")

    label_counts = labeled[HUMAN_LABEL_COL].value_counts()
    print("\n[标签分布]")
    print(label_counts)
    if len(label_counts) > 0:
        min_class = label_counts.idxmin()
        if label_counts.min() < 50:
            print(f"\n[提醒] '{min_class}'类别样本量仅 {label_counts.min()} 条，"
                  f"用这份数据算出来的该类别指标（如F1）置信度较低，仅供参考。")
    else:
        print("\n[警告] 没有任何有效标注数据可用，请检查标签归一化映射是否覆盖了所有原始标签值。")


if __name__ == "__main__":
    main()
