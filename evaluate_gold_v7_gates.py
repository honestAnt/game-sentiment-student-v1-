# -*- coding: utf-8 -*-
"""
evaluate_gold_v7_gates.py
v7 闸门评估：在**两个集合**上跑同一个模型，按 GATES_V7 判定晋级。

为什么要两个集合（见 docs/v10训练方案-先修尺子-2026-09-17.md）：
  gold_eval_v7.jsonl            分布真实 -> 量 acc（微博两片主判据、B站 acc 护栏）
  gold_bili_neg_challenge.jsonl 负面富集 -> 量 B站负召回（主判据）与负面精确率

v6 的「三片 acc ≥ v8」已证伪：B站片 91.5% 是中性，v8 只比「全判中性」多对 1 条，
该闸门与「抓负面」反向，选出了负召回垫底的 v9o。v7 把 B站 acc 降级为
「不低于基线 1.0pt」的护栏，主判据换成富集集上的负召回。

用法：
    # 1) 先用现役 v8 建基线（标注回收后只做一次）
    python evaluate_gold_v7_gates.py --model_dir models/student_v8 --tag v8 --write_baseline

    # 2) 之后每个候选模型都比这条基线
    python evaluate_gold_v7_gates.py --model_dir models/student_v10a --tag v10a \\
        --recipe "全量混集含 round6 B站，lr=5e-6x2ep"

闸门 FAIL 时退出码为 1（不晋级就停，不要用「再加 1000 条」循环）。
"""

import argparse
import json
import os

from gold_eval_infer import predict
from gold_gates import GATES_V7, check_gates, format_gate_report, metrics, slice_metrics

DEFAULT_GOLD = os.path.join("data", "validation", "gold_eval_v7.jsonl")
DEFAULT_CHALLENGE = os.path.join("data", "validation", "gold_bili_neg_challenge.jsonl")
DEFAULT_BASELINE = os.path.join("data", "validation", "gold_eval_v7_v8_baseline.json")
OUT_TMPL = os.path.join("data", "validation", "gold_eval_v7_{tag}_baseline.json")

BILI = "bilibili/评论"


def load_jsonl(path):
    if not os.path.exists(path):
        raise SystemExit(
            f"[中止] 找不到 {path}。v7 集合尚未回收，先完成人工标注再跑：\n"
            f"  python merge_gold_bili.py --iid_xlsx <...> --challenge_xlsx <...>"
        )
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--gold", default=DEFAULT_GOLD)
    parser.add_argument("--challenge", default=DEFAULT_CHALLENGE)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--conf_threshold", type=float, default=0.75)
    parser.add_argument("--prefix", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--recipe", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--write_baseline", action="store_true",
                        help="把本次结果写成 v7 基线（只该对现役 v8 用一次），跳过闸门判定")
    args = parser.parse_args()

    gold = load_jsonl(args.gold)
    chl = load_jsonl(args.challenge)
    gold_y = [r["gold_label"] for r in gold]
    chl_y = [r["gold_label"] for r in chl]

    print(f"[评估] {args.model_dir}")
    print(f"  分布真实集 {args.gold} n={len(gold)}")
    print(f"  负面富集集 {args.challenge} n={len(chl)}"
          f"（负面 {sum(1 for y in chl_y if y == '负面')} 条）")
    print(f"[工作点] neg_exempt T={args.conf_threshold} 前缀={args.prefix}")

    gold_pred = predict(args.model_dir, gold, args.conf_threshold, args.prefix)
    chl_pred = predict(args.model_dir, chl, args.conf_threshold, args.prefix,
                       device=gold_pred["device"])

    gold_slices = slice_metrics(gold, gold_y, gold_pred["ex"])
    gold_overall = metrics(gold_y, gold_pred["ex"])
    # 富集集全是 B站评论，直接整体算并挂到 BILI 键，与闸门的 slice 名对齐
    chl_metrics = {BILI: metrics(chl_y, chl_pred["ex"])}

    candidate = {"gold": gold_slices, "challenge": chl_metrics}
    payload = {
        "model": args.model_dir,
        "device": gold_pred["device"],
        "use_prefix": gold_pred["use_prefix"],
        "threshold": args.conf_threshold,
        "recipe": args.recipe,
        "n_gold": len(gold),
        "n_challenge": len(chl),
        "overall": {
            "raw_acc": metrics(gold_y, gold_pred["raw"])["acc"],
            "fb_acc": metrics(gold_y, gold_pred["fb"])["acc"],
            "ex_acc": gold_overall["acc"],
            "ex_f1": gold_overall["macro_f1"],
            "ex_neg_rec": gold_overall["neg_rec"],
        },
        "slices_ex": gold_slices,
        "challenge_ex": chl_metrics,
    }

    chl_b = chl_metrics[BILI]
    print(f"\n[富集集] B站负召回={chl_b['neg_rec']} 负面精确率={chl_b['neg_prec']} "
          f"（金负 {chl_b['n_neg_gold']} 条，1 条 = "
          f"{100/max(chl_b['n_neg_gold'], 1):.1f}pt）")

    if args.write_baseline:
        out = args.out or args.baseline
        payload["role"] = "baseline"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n[基线已写入] {out}")
        print("  后续候选模型用同一份基线比对，不要每轮重建基线")
        return

    if not os.path.exists(args.baseline):
        raise SystemExit(
            f"[中止] 基线 {args.baseline} 不存在。先对现役 v8 建基线：\n"
            f"  python evaluate_gold_v7_gates.py --model_dir models/student_v8 "
            f"--tag v8 --write_baseline"
        )
    base = json.load(open(args.baseline, encoding="utf-8"))
    baseline = {"gold": base["slices_ex"], "challenge": base["challenge_ex"]}
    gate_result = check_gates(candidate, baseline, GATES_V7)

    print()
    print(format_gate_report(gate_result, gold_overall, base["overall"]["ex_acc"]))

    payload["baseline_file"] = args.baseline
    payload["gates"] = gate_result
    payload["gates_pass"] = gate_result["passed"]
    out = args.out or OUT_TMPL.format(tag=args.tag)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[产物] {out}")
    if gate_result["passed"]:
        print("[提醒] 闸门过了不等于可以晋级：v7 起 i.i.d. 抽检是晋级必过项，"
              "v9o 就是只过 gold 不过 i.i.d.（见 v10 方案第 1.3 节）")
    raise SystemExit(0 if gate_result["passed"] else 1)


if __name__ == "__main__":
    main()
