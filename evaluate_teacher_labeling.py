# -*- coding: utf-8 -*-
"""
evaluate_teacher_labeling.py
在人工标注的 gold validation set 上跑一遍 Qwen + DeepSeek 交叉标注流程，
统计：
  1. Qwen / DeepSeek 各自与人工标注的一致率、混淆矩阵
  2. 两模型互相的一致率
  3. "两模型一致"这个子集，与人工标注的吻合度有多高
     —— 这是最关键的指标，因为 teacher_labeling.py 实际写入训练集的
        就是这个子集，如果这个子集吻合度不高，说明 prompt 需要调整

用法：
    export DASHSCOPE_API_KEY=xxx
    export DEEPSEEK_API_KEY=xxx
    python evaluate_teacher_labeling.py \
        --input data/validation/validation_gold.jsonl \
        --output data/validation/validation_predictions.jsonl
"""

import argparse
import concurrent.futures
import json

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from config import LABEL2ID, LLM_CONCURRENCY
from teacher_labeling import (
    call_llm, deepseek_client, qwen_client,
    DEEPSEEK_MODEL_NAME, QWEN_MODEL_NAME,
)

LABELS_ORDER = list(LABEL2ID.keys())   # ["负面", "中性", "正面"] 之类，按config顺序展示混淆矩阵


def label_one_validation_sample(record: dict) -> dict:
    text, platform = record["text"], record.get("platform", "weibo")
    qwen_result = call_llm(qwen_client, QWEN_MODEL_NAME, text, platform)
    deepseek_result = call_llm(deepseek_client, DEEPSEEK_MODEL_NAME, text, platform)

    record["qwen_pred"] = qwen_result["sentiment"]
    record["deepseek_pred"] = deepseek_result["sentiment"]
    record["models_agree"] = (
        qwen_result["sentiment"] is not None
        and qwen_result["sentiment"] == deepseek_result["sentiment"]
    )
    return record


def print_report(y_true, y_pred, title):
    print(f"\n{'='*20} {title} {'='*20}")
    valid_mask = [p is not None for p in y_pred]
    y_true_f = [t for t, m in zip(y_true, valid_mask) if m]
    y_pred_f = [p for p, m in zip(y_pred, valid_mask) if m]
    n_invalid = len(y_true) - len(y_true_f)
    if n_invalid > 0:
        print(f"[提示] 有 {n_invalid} 条模型返回无效标签，已从统计中剔除")

    print(classification_report(y_true_f, y_pred_f, labels=LABELS_ORDER, zero_division=0, digits=3))
    print("混淆矩阵 (行=人工标注真值, 列=模型预测):")
    cm = confusion_matrix(y_true_f, y_pred_f, labels=LABELS_ORDER)
    header = "        " + "  ".join(f"{l:>6s}" for l in LABELS_ORDER)
    print(header)
    for i, row_label in enumerate(LABELS_ORDER):
        row_str = "  ".join(f"{v:>6d}" for v in cm[i])
        print(f"{row_label:>6s}  {row_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="prepare_validation_gold.py 产出的 jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    print(f"[验证集规模] {len(records)} 条")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as executor:
        futures = [executor.submit(label_one_validation_sample, r) for r in records]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="验证集标注中"):
            results.append(future.result())

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    df = pd.DataFrame(results)
    gold = df["gold_label"].tolist()

    # 1. 单模型 vs 人工标注
    print_report(gold, df["qwen_pred"].tolist(), "Qwen 单模型 vs 人工标注")
    print_report(gold, df["deepseek_pred"].tolist(), "DeepSeek 单模型 vs 人工标注")

    # 2. 两模型互相一致率
    agree_rate = df["models_agree"].mean()
    print(f"\n{'='*20} 两模型互相一致率 {'='*20}")
    print(f"Qwen 与 DeepSeek 判断完全一致: {df['models_agree'].sum()}/{len(df)} ({agree_rate:.1%})")

    # 3. 关键指标：两模型一致的子集，与人工标注的吻合度
    #    这个子集就是 teacher_labeling.py 实际会写入训练集的数据
    agreed_df = df[df["models_agree"]]
    if len(agreed_df) > 0:
        print_report(
            agreed_df["gold_label"].tolist(),
            agreed_df["qwen_pred"].tolist(),
            f"【核心指标】两模型一致子集({len(agreed_df)}条) vs 人工标注 —— 这就是会进入训练集的数据质量",
        )
        final_acc = (agreed_df["gold_label"] == agreed_df["qwen_pred"]).mean()
        print(f"\n>>> 训练集质量预估：两模型一致的样本中，{final_acc:.1%} 与人工标注相符 <<<")
        if final_acc < 0.85:
            print(">>> 建议：这个数字低于85%，说明prompt或标签体系需要调整，"
                  "不建议直接大规模标注，先看看下面按内容类型/标签的细分错误再改prompt。")
    else:
        print("\n[警告] 没有任何样本两模型判断一致，请检查API调用是否正常")

    # 4. 按内容类型（博文/评论）拆分看，帮助定位问题集中在哪
    if "content_type" in df.columns:
        print(f"\n{'='*20} 按内容类型拆分（仅两模型一致子集） {'='*20}")
        for ctype, sub in agreed_df.groupby("content_type"):
            acc = (sub["gold_label"] == sub["qwen_pred"]).mean()
            print(f"  {ctype}: {len(sub)} 条，准确率 {acc:.1%}")

    print(f"\n[详细结果已保存] {args.output}")


if __name__ == "__main__":
    main()
