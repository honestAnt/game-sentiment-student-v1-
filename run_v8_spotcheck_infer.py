# -*- coding: utf-8 -*-
"""
run_v8_spotcheck_infer.py
对 v8_iid_spotcheck 候选跑 student_v8 推理（前缀 auto），输出三分类概率分布，
供三工作点（raw / T0.75 一刀切 / T0.75 负面豁免）离线对拍。

产出：同目录 preds_<n>.jsonl，每行追加
    raw_label / raw_conf / probs=[P(负面),P(中性),P(正面)]
"""

import argparse
import json

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import MAX_SEQ_LEN
from infer_batch import batch_infer_probs, pick_device
from platform_input import model_input_text, prefix_enabled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model_dir", default="models/student_v8")
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    assert prefix_enabled(args.model_dir, "auto"), (
        f"{args.model_dir} 不是前缀模型（目录名版本号需 ≥ v8），本脚本按前缀口径打分"
    )

    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    texts = [
        model_input_text(r["text"], r.get("platform"), r.get("content_type"))
        for r in rows
    ]
    labels, confs, probs = batch_infer_probs(texts, model, tokenizer, device)

    for r, l, c, p in zip(rows, labels, confs, probs):
        r["raw_label"] = l
        r["raw_conf"] = round(c, 4)
        r["probs"] = [round(x, 4) for x in p]

    out = args.input.replace("candidates_", "preds_")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[完成] {len(rows)} 条 -> {out}")


if __name__ == "__main__":
    main()
