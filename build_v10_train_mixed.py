# -*- coding: utf-8 -*-
"""v10 混训集：v7_1 + 全量 round6，并**去掉 v7_1 里 lol_* 行的整行复制**。

为什么做去重（2026-09-17 发现）：
`v7_1_train_mixed.jsonl` 有 569 行是整行复制（id/文本/标签全同），全部来自 `lol_*` 行：
  B站片   945 行 = 321 条唯一(1×) + 208 条唯一 LoL 评论各复制 3 遍(624 行)
  微博评论 858 行 = 552 条唯一(1×) + 153 条唯一 LoL 评论各复制 2 遍(306 行)
  微博博文 434 行 = 零重复
即 **B站训练片 56% 的行来自 176 条唯一的中性 LoL 电竞闲聊**（被 3× 放大的那批
负面只占 5.8%，未放大的占 8.4%）。这比学习率更能解释 v9 全系列的 B站 中性偏置；
v9o 的 stage2 回放（v9k_bili_replay_only）用的正是这份带 3× 复制的 945 行，
等于再叠一层曝光，它也确实成了全系列负召回最低的模型（3/17）。

注意：现役 v8 就是拿带复制的 v7_1 训出来的，所以去重是**需要验证的改动而非既定修复**。
本脚本产出的 v10a 与已有的 v9j（同语料、同 lr=5e-6×2ep、未去重）构成干净单变量对照。

用法：
    python build_v10_train_mixed.py                 # v10a：去重 + 全量 round6
    python build_v10_train_mixed.py --keep_dups     # 对照：不去重（应等于 v9_train_mixed）
    python build_v10_train_mixed.py --post_replay 1 # 若博文片掉分，加 1× v7_1 博文回放

产出：data/labeled/v10a_train_mixed.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from config import EMOTION_TAGS, LABEL2ID, LABELED_DIR, ROUND3_NEG_MAX_FRAC
from platform_input import PREFIXES, derive_content_type

NEG_MIN_FRAC = 0.21
NEG_MAX_FRAC = ROUND3_NEG_MAX_FRAC  # 0.26


def load_jsonl(path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dedup_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    """按 id 去整行复制，保留首次出现，保持原顺序。

    只在「同 id 的行内容完全一致」时才算复制；若同 id 出现不同文本或标签，
    那是标签冲突而非复制，必须报错而不是静默保留其中一条。
    """
    seen: dict[str, dict] = {}
    out, removed, conflicts = [], Counter(), []
    for row in rows:
        rid = row["id"]
        if rid in seen:
            prev = seen[rid]
            if (prev["text"], prev["final_label"]) != (row["text"], row["final_label"]):
                conflicts.append(rid)
            removed[(row.get("platform"), row.get("content_type"))] += 1
            continue
        seen[rid] = row
        out.append(row)
    if conflicts:
        raise SystemExit(
            f"[终止] {len(conflicts)} 个 id 同号不同内容（标签冲突，非整行复制），"
            f"不能自动去重：{conflicts[:5]}"
        )
    return out, dict(removed)


def backfill_content_type(row: dict) -> dict:
    out = dict(row)
    if not out.get("content_type"):
        ct = derive_content_type(out)
        if ct:
            out["content_type"] = ct
    return out


def slice_of(row: dict) -> tuple:
    return (row.get("platform"), row.get("content_type"))


def validate(rows: list[dict], allow_dups: bool) -> list[str]:
    errs = []
    if not rows:
        return ["混训集为空"]
    ids = [r["id"] for r in rows]
    if not allow_dups and len(ids) != len(set(ids)):
        errs.append(f"去重后仍有重复 id {len(ids) - len(set(ids))} 行")
    labels = Counter(r["final_label"] for r in rows)
    neg_frac = labels["负面"] / len(rows)
    if not NEG_MIN_FRAC - 1e-12 <= neg_frac <= NEG_MAX_FRAC + 1e-12:
        errs.append(f"负面占比 {neg_frac:.4f} 不在 [{NEG_MIN_FRAC}, {NEG_MAX_FRAC}]")
    for i, r in enumerate(rows):
        if r.get("final_label") not in LABEL2ID:
            errs.append(f"第{i}行 final_label 非法: {r.get('final_label')}")
            break
        if r.get("final_emotion") not in EMOTION_TAGS:
            errs.append(f"第{i}行 final_emotion 非法: {r.get('final_emotion')}")
            break
        if slice_of(r) not in PREFIXES:
            errs.append(f"第{i}行前缀不可解析 {slice_of(r)}")
            break
    return errs


def build(base: list[dict], round6: list[dict], dedup: bool,
          post_replay: int) -> tuple[list[dict], dict]:
    stats = {"base_in": len(base), "round6_in": len(round6)}
    if dedup:
        base, removed = dedup_rows(base)
        stats["dedup_removed"] = removed
        stats["dedup_removed_total"] = sum(removed.values())
    rows = [backfill_content_type(r) for r in base + round6]
    if post_replay:
        posts = [r for r in rows if slice_of(r) == ("weibo", "博文")]
        # 回放行沿用原 id：训练只按行取样，重复 id 是回放的既定语义，
        # 与「意外整行复制」区分靠 --post_replay 显式声明
        rows = rows + posts * post_replay
        stats["post_replay_rows"] = len(posts) * post_replay
    return rows, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep_dups", action="store_true",
                        help="不去重（对照组，结果应等于 v9_train_mixed）")
    parser.add_argument("--post_replay", type=int, default=0,
                        help="额外回放 N 份 weibo 博文（博文片掉分时用）")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    base = load_jsonl(Path(LABELED_DIR) / "v7_1_train_mixed.jsonl")
    round6 = []
    for p in sorted((Path(LABELED_DIR) / "round6").glob("shard_*.jsonl")):
        round6.extend(load_jsonl(p))
    if len(base) != 6895:
        raise SystemExit(f"v7_1 条数异常: {len(base)} != 6895")
    if len(round6) != 1200:
        raise SystemExit(f"round6 条数异常: {len(round6)} != 1200")
    if {r["id"] for r in base} & {r["id"] for r in round6}:
        raise SystemExit("v7_1 与 round6 id 重合，拒绝混训")

    dedup = not args.keep_dups
    rows, stats = build(base, round6, dedup, args.post_replay)

    print(f"[输入] v7_1 {stats['base_in']} + round6 {stats['round6_in']}")
    if dedup:
        print(f"[去重] 移除整行复制 {stats['dedup_removed_total']} 行:")
        for sl, n in sorted(stats["dedup_removed"].items(), key=lambda x: -x[1]):
            print(f"    {sl}: -{n}")
    else:
        print("[去重] 已关闭（对照组）")
    if args.post_replay:
        print(f"[回放] 额外 weibo 博文 {stats['post_replay_rows']} 行")

    errs = validate(rows, allow_dups=args.keep_dups or bool(args.post_replay))
    if errs:
        print("[校验未过]")
        for e in errs:
            print(f"  - {e}")
        raise SystemExit(1)

    labels = Counter(r["final_label"] for r in rows)
    print(f"\n[输出] n={len(rows)} {dict(labels)} 负面={labels['负面']/len(rows):.1%}")
    print("[片构成]")
    for sl, n in sorted(Counter(slice_of(r) for r in rows).items()):
        sub = [r for r in rows if slice_of(r) == sl]
        neg = sum(1 for r in sub if r["final_label"] == "负面")
        uniq = len({r["id"] for r in sub})
        print(f"    {sl}: n={n} 唯一={uniq} 负面={neg} ({neg/n:.1%})")

    name = args.out or ("v10a_train_mixed.jsonl" if dedup else "v10_keepdups_train_mixed.jsonl")
    out_path = Path(LABELED_DIR) / name if not os.path.dirname(name) else Path(name)
    os.makedirs(out_path.parent, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[完成] -> {out_path}")
    print("[下一步] 与 v9j（同语料、同 lr、未去重）构成单变量对照：")
    print("  python train_student.py --train_file data/labeled/v10a_train_mixed.jsonl \\")
    print("    --init_from models/student_v8 --output_dir models/student_v10a \\")
    print("    --prefix_granularity platform_ct --learning_rate 5e-6 --num_train_epochs 2")


if __name__ == "__main__":
    main()
