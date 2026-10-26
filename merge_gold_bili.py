# -*- coding: utf-8 -*-
"""
merge_gold_bili.py
回收人工标注完成的 B站金标扩样 xlsx（export_gold_bili_candidates.py 的两份产物）。

为什么不复用 merge_gold_weibo_comments.py：那个脚本把 platform 硬编码成 "weibo"，
且只有「合并进 gold」一条路径；B站扩样是两个用途不同的集合，必须分开落盘。

两份表分开回收，产物也分开：

  A. iid 300       -> 追加进 gold_eval_v6，产出 data/validation/gold_eval_v7.jsonl
                      （分布真实，可与 v6 的 B站片续接）
  B. challenge 250 -> 独立落到 data/validation/gold_bili_neg_challenge.jsonl
                      （富集集，**不进 gold_v7**；混进去会污染 acc 的分布代表性，
                        它只用于量 B站负召回与负面精确率）

标签归一化沿用 prepare_validation_gold.RAW_LABEL_NORMALIZE（中立->中性 等）。
platform/content_type/domain/channel 从导出时的 jsonl 存档按「序号」回查，
不在 xlsx 里重新推导——xlsx 经人手编辑，不能当结构真源。

用法：
    # 两份都标完后一次回收
    python merge_gold_bili.py \
        --iid_xlsx data/candidates/gold_expansion_bili/bili_iid_candidates_300.xlsx \
        --challenge_xlsx data/candidates/gold_expansion_bili/bili_challenge_candidates_250.xlsx

    # 只回收其中一份（另一份还没标完）
    python merge_gold_bili.py --iid_xlsx <path>

回收后记得把 gold_eval_v7.jsonl 与 challenge 文件加进采样排除集
（已预置在 export_starrocks_iid_spotcheck.EXTRA_EXCLUSION_PATTERNS）。
"""

import argparse
import json
import os
from collections import Counter

import pandas as pd

from config import LABEL2ID
from prepare_validation_gold import RAW_LABEL_NORMALIZE

GOLD_V6 = os.path.join("data", "validation", "gold_eval_v6.jsonl")
GOLD_V7 = os.path.join("data", "validation", "gold_eval_v7.jsonl")
CHALLENGE_OUT = os.path.join("data", "validation", "gold_bili_neg_challenge.jsonl")

ID_COL = "序号"
TEXT_COL = "文本"
HUMAN_LABEL_COL = "人工标注-杠"

MIN_LABELED_RATIO = 0.9   # 标注覆盖率低于此值视为没标完，中止（可 --force 跳过）


def load_archive(jsonl_path: str) -> dict:
    """导出时的 jsonl 存档：序号(cid) -> 完整结构字段。"""
    archive = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            # id 形如 ge7b_<cid> / ge7c_<cid>，取 cid 作为与 xlsx「序号」的连接键
            cid = rec["id"].split("_", 1)[1]
            archive[cid] = rec
    return archive


def archive_path_for(xlsx_path: str) -> str:
    return os.path.splitext(xlsx_path)[0] + ".jsonl"


def labeled_rows(xlsx_path: str, archive: dict) -> tuple[list[dict], dict]:
    """抽出有效人标行，并按「序号」从存档回查结构字段。

    空标签跳过；标签无法归一化、或序号不在存档里的，进 issues 明示而不是静默丢。
    """
    df = pd.read_excel(xlsx_path)
    for col in (ID_COL, TEXT_COL, HUMAN_LABEL_COL):
        if col not in df.columns:
            raise ValueError(f"{xlsx_path} 缺列「{col}」，实际列={list(df.columns)}")

    rows, bad_label, missing, text_drift = [], [], [], []
    for _, row in df.iterrows():
        raw = row[HUMAN_LABEL_COL]
        if pd.isna(raw):
            continue
        cid = str(row[ID_COL]).strip()
        label = RAW_LABEL_NORMALIZE.get(str(raw).strip(), str(raw).strip())
        if label not in LABEL2ID:
            bad_label.append((cid, str(raw).strip()))
            continue
        rec = archive.get(cid)
        if rec is None:
            missing.append(cid)
            continue
        # xlsx 里的文本被改过就以存档为准，但要报出来（多为 Excel 自动格式化所致）
        if str(row[TEXT_COL]) != rec["text"]:
            text_drift.append(cid)
        out = dict(rec)
        out["gold_label"] = label
        rows.append(out)

    issues = {
        "n_total": len(df),
        "n_unlabeled": int(df[HUMAN_LABEL_COL].isna().sum()),
        "bad_label": bad_label,
        "missing_in_archive": missing,
        "text_drift": text_drift,
    }
    return rows, issues


def report_issues(path: str, issues: dict, force: bool):
    n_total, n_unlab = issues["n_total"], issues["n_unlabeled"]
    n_labeled = n_total - n_unlab
    print(f"[回收] {path}: 已标 {n_labeled}/{n_total}")
    if issues["bad_label"]:
        print(f"  [警告] {len(issues['bad_label'])} 条标签无法归一化，已跳过：")
        for cid, raw in issues["bad_label"][:10]:
            print(f"    {cid}\t{raw}")
    if issues["missing_in_archive"]:
        print(f"  [警告] {len(issues['missing_in_archive'])} 条「序号」不在 jsonl 存档里，已跳过："
              f"{issues['missing_in_archive'][:5]}")
    if issues["text_drift"]:
        print(f"  [提示] {len(issues['text_drift'])} 条 xlsx 文本与存档不一致，以存档为准"
              f"（通常是 Excel 自动格式化）")
    if n_total and n_labeled / n_total < MIN_LABELED_RATIO and not force:
        raise SystemExit(
            f"[中止] {path} 标注覆盖率 {n_labeled}/{n_total} < {MIN_LABELED_RATIO:.0%}，"
            f"疑似未标完；标完后重跑，或加 --force"
        )


def to_gold_record(rec: dict) -> dict:
    """统一成 gold_eval 的行结构（与 gold_eval_v6 字段对齐）。"""
    return {
        "id": rec["id"],
        "text": rec["text"],
        "platform": rec["platform"],
        "content_type": rec["content_type"],
        "gold_label": rec["gold_label"],
    }


def append_to_gold(gold_rows: list[dict], incoming: list[dict]) -> tuple[list[dict], dict]:
    """原 gold 行原样保留，新行按文本/id 去重后追加。"""
    texts = {r["text"] for r in gold_rows}
    ids = {r["id"] for r in gold_rows}
    new_rows, dup_text, dup_id = [], 0, 0
    for rec in incoming:
        if rec["text"] in texts:
            dup_text += 1
            continue
        if rec["id"] in ids:
            dup_id += 1
            continue
        row = to_gold_record(rec)
        new_rows.append(row)
        texts.add(row["text"])
        ids.add(row["id"])
    return gold_rows + new_rows, {"dup_text": dup_text, "dup_id": dup_id}


def summarize(rows: list[dict], title: str):
    labels = Counter(r["gold_label"] for r in rows)
    n = len(rows)
    print(f"  {title}: n={n} {dict(labels)}"
          + (f" 负面占比={labels['负面']/n:.1%}" if n else ""))
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iid_xlsx", help="随机集 xlsx，回收后并入 gold_v7")
    parser.add_argument("--challenge_xlsx", help="富集集 xlsx，回收为独立 challenge 文件")
    parser.add_argument("--gold_in", default=GOLD_V6)
    parser.add_argument("--gold_out", default=GOLD_V7)
    parser.add_argument("--challenge_out", default=CHALLENGE_OUT)
    parser.add_argument("--force", action="store_true", help="标注未覆盖满也强制回收")
    args = parser.parse_args()

    if not args.iid_xlsx and not args.challenge_xlsx:
        raise SystemExit("[中止] 至少要传 --iid_xlsx 或 --challenge_xlsx 之一")

    if args.challenge_xlsx:
        archive = load_archive(archive_path_for(args.challenge_xlsx))
        rows, issues = labeled_rows(args.challenge_xlsx, archive)
        report_issues(args.challenge_xlsx, issues, args.force)
        os.makedirs(os.path.dirname(args.challenge_out) or ".", exist_ok=True)
        with open(args.challenge_out, "w", encoding="utf-8") as f:
            for rec in rows:
                out = to_gold_record(rec)
                out["channel"] = rec.get("channel", "")
                out["domain"] = rec.get("domain", "")
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"[产出] challenge -> {args.challenge_out}")
        labels = summarize(rows, "challenge 总体")
        by_ch = {}
        for rec in rows:
            by_ch.setdefault(rec.get("channel", ""), []).append(rec)
        for ch, sub in sorted(by_ch.items()):
            summarize(sub, f"  通道「{ch}」")
        if labels["负面"] < 80:
            print(f"  [警告] challenge 集负面仅 {labels['负面']} 条，低于 ~100 的设计目标，"
                  f"B站负召回分辨力仍不足（1 条 = {100/max(labels['负面'],1):.1f}pt），"
                  f"建议再富集一批后再定闸门")

    if args.iid_xlsx:
        archive = load_archive(archive_path_for(args.iid_xlsx))
        rows, issues = labeled_rows(args.iid_xlsx, archive)
        report_issues(args.iid_xlsx, issues, args.force)
        gold_rows = [json.loads(l) for l in open(args.gold_in, encoding="utf-8") if l.strip()]
        merged, dedup = append_to_gold(gold_rows, rows)
        if dedup["dup_text"] or dedup["dup_id"]:
            print(f"  [去重] 文本重合 {dedup['dup_text']} 条、id 重合 {dedup['dup_id']} 条已跳过")
        os.makedirs(os.path.dirname(args.gold_out) or ".", exist_ok=True)
        with open(args.gold_out, "w", encoding="utf-8") as f:
            for r in merged:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_new = len(merged) - len(gold_rows)
        print(f"[产出] gold_v7 = {len(gold_rows)} 原行 + {n_new} 新行 = {len(merged)} 条 "
              f"-> {args.gold_out}")
        slices = Counter((r["platform"], r["content_type"]) for r in merged)
        for sl, n in sorted(slices.items()):
            sub = [r for r in merged if (r["platform"], r["content_type"]) == sl]
            neg = sum(1 for r in sub if r["gold_label"] == "负面")
            print(f"  {sl}: n={n} 负面={neg} ({neg/n:.1%})，1 条 = {100/n:.2f}pt")

    print("\n[下一步] 回收完后重算 v8 在新集合上的基线，再按新闸门定义训 v10：")
    print("  python evaluate_gold_v7_gates.py --model_dir models/student_v8 --tag v8 --write_baseline")


if __name__ == "__main__":
    main()
