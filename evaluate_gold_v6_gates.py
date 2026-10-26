# -*- coding: utf-8 -*-
"""
evaluate_gold_v6_gates.py
在 gold_eval_v6 上按现役工作点（平台前缀 + neg_exempt T=0.75）评估候选模型，
比对基线三片闸门，产出与既有 gold_eval_v6_*_baseline.json 同结构的结果文件。

替代 v9 系列消融里的一次性脚本：口径（分片切分/指标/闸门）全部来自 gold_gates.py，
基线从 --baseline 的 json 读取而不是硬编码，避免每轮手抄 v8 数字抄错。

用法：
    python evaluate_gold_v6_gates.py --model_dir models/student_v9o --tag v9o \
        --recipe "s1=v9c mix lr=5e-6x2ep; then v7_1 bili lr=5e-6x1ep"

闸门 FAIL 时退出码为 1，便于串在脚本里自动停（不晋级就停，不要循环加数据）。
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from gold_gates import (
    GATES_V6, check_gates, format_gate_report, metrics, slice_metrics,
)
from infer_batch import apply_conf_fallback, batch_infer, pick_device
from platform_input import derive_content_type, model_input_text, prefix_enabled

DEFAULT_GOLD = os.path.join("data", "validation", "gold_eval_v6.jsonl")
DEFAULT_BASELINE = os.path.join("data", "validation", "gold_eval_v6_v8_baseline.json")
OUT_TMPL = os.path.join("data", "validation", "gold_eval_v6_{tag}_baseline.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--tag", required=True, help="产物命名，如 v9o -> gold_eval_v6_v9o_baseline.json")
    parser.add_argument("--gold", default=DEFAULT_GOLD)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE,
                        help="闸门基线 json（默认现役 v8）")
    parser.add_argument("--conf_threshold", type=float, default=0.75)
    parser.add_argument("--prefix", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--recipe", default="", help="配方备注，写进产物便于回溯")
    parser.add_argument("--out", default="", help="覆盖默认产物路径")
    args = parser.parse_args()

    records = [json.loads(l) for l in open(args.gold, encoding="utf-8") if l.strip()]
    golds = [r["gold_label"] for r in records]
    baseline = json.load(open(args.baseline, encoding="utf-8"))
    baseline_slices = baseline["slices_ex"]
    baseline_overall = baseline["overall"]["ex_acc"]

    use_prefix = prefix_enabled(args.model_dir, args.prefix)
    device = pick_device()
    print(f"[评估] {args.model_dir} × {args.gold} n={len(records)} device={device}")
    print(f"[工作点] 前缀={'开' if use_prefix else '关'} neg_exempt T={args.conf_threshold}")
    print(f"[基线] {args.baseline} 总 ex_acc={baseline_overall:.4f}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()
    if use_prefix:
        texts = [
            model_input_text(r["text"], r.get("platform"), derive_content_type(r))
            for r in records
        ]
    else:
        texts = [r["text"] for r in records]
    infer_bs = 16 if device == "mps" else 64
    raw_preds, confs = batch_infer(texts, model, tokenizer, device, batch_size=infer_bs)

    ex_preds, _ = apply_conf_fallback(raw_preds, confs, args.conf_threshold, neg_exempt=True)
    fb_preds, _ = apply_conf_fallback(raw_preds, confs, args.conf_threshold, neg_exempt=False)

    raw_overall = metrics(golds, raw_preds)
    fb_overall = metrics(golds, fb_preds)
    ex_overall = metrics(golds, ex_preds)
    slices_ex = slice_metrics(records, golds, ex_preds)
    gate_result = check_gates(slices_ex, baseline_slices, GATES_V6)

    print()
    print(format_gate_report(gate_result, ex_overall, baseline_overall))

    payload = {
        "model": args.model_dir,
        "device": device,
        "n_v6": len(records),
        "recipe": args.recipe,
        "threshold": args.conf_threshold,
        "overall": {
            "raw_acc": raw_overall["acc"],
            "fb_acc": fb_overall["acc"],
            "ex_acc": ex_overall["acc"],
            "raw_f1": raw_overall["macro_f1"],
            "ex_f1": ex_overall["macro_f1"],
            "ex_neg_rec": ex_overall["neg_rec"],
        },
        "slices_ex": slices_ex,
        "baseline_file": args.baseline,
        "gates": gate_result,
        "gates_pass": gate_result["passed"],
    }
    out = args.out or OUT_TMPL.format(tag=args.tag)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[产物] {out}")
    raise SystemExit(0 if gate_result["passed"] else 1)


if __name__ == "__main__":
    main()
