# -*- coding: utf-8 -*-
import argparse
import json
from collections import Counter

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import ID2LABEL, LABEL2ID
from infer_batch import batch_infer
from platform_input import derive_content_type, model_input_text, prefix_enabled

CONF_THRESHOLD = 0.75   # 与部署口径 infer_batch --conf_threshold 默认值一致


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models/student_v1")
    parser.add_argument("--gold", default="data/validation/validation_gold.jsonl")
    parser.add_argument("--prefix", choices=["auto", "on", "off"], default="auto",
                        help="平台前缀输入：auto=目录版本≥v8 则启用，off=旧口径原文输入")
    args = parser.parse_args()

    use_prefix = prefix_enabled(args.model_dir, args.prefix)
    print(f"[前缀输入] {args.prefix} -> {'开（model_input_text 拼平台前缀）' if use_prefix else '关（原文）'}")

    records = [json.loads(l) for l in open(args.gold, encoding="utf-8") if l.strip()]
    if use_prefix:
        texts = [model_input_text(r["text"], r.get("platform"), derive_content_type(r)) for r in records]
    else:
        texts = [r["text"] for r in records]
    y_true = [r["gold_label"] for r in records]

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[评估设备] {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()
    infer_bs = 16 if device == "mps" else 64
    y_pred, confs = batch_infer(texts, model, tokenizer, device, batch_size=infer_bs)
    print(classification_report(y_true, y_pred, labels=list(LABEL2ID.keys()), zero_division=0, digits=3))
    print("混淆矩阵 行=人工 列=student")
    print(confusion_matrix(y_true, y_pred, labels=list(LABEL2ID.keys())))
    print(f"accuracy={accuracy_score(y_true, y_pred):.4f}")
    print(f"macro_f1={f1_score(y_true, y_pred, average='macro'):.4f}")
    counts = Counter(y_true)
    min_label, min_n = min(counts.items(), key=lambda kv: kv[1])
    print(f"gold n={len(records)} 分布={dict(counts)}")
    if min_n < 50:
        print(f"提示：gold「{min_label}」仅 {min_n} 条，该类别 F1 仅供参考。")

    band = Counter((p, t) for p, t, c in zip(y_pred, y_true, confs) if c < CONF_THRESHOLD)
    print(f"\n[低置信带] max_prob<{CONF_THRESHOLD} 共 {sum(band.values())} 条，(pred, gold) 构成：")
    for (p, t), n in band.most_common():
        print(f"  pred={p} gold={t} × {n}")


if __name__ == "__main__":
    main()
