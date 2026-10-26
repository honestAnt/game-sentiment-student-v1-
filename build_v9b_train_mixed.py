# -*- coding: utf-8 -*-
"""v9b 消融混训：v7_1 + round6，仅对 round6 的 B站片做负面降采样。

背景：v9 全量 round6 后 B站金标 −2.8pt；错例多为中性玩梗/锐评被打成负面。
round6 B站 353 条负面 191（54%），抗议线索过密，负面先验泄漏到赛事玩梗。
本脚本只动 B站负向配额，微博片不动；采样后校验，不达标换 seed 重采，最多 3 轮。

用法:
    python build_v9b_train_mixed.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from config import EMOTION_TAGS, LABEL2ID, LABELED_DIR, ROUND3_NEG_MAX_FRAC
from platform_input import PREFIXES, derive_content_type

SEED = 49
MAX_ATTEMPTS = 3
OUT_NAME = "v9b_train_mixed.jsonl"
# 混训全集负面占比目标（与数据集对齐配置 / round3 经验一致）
NEG_MIN_FRAC = 0.21
NEG_MAX_FRAC = ROUND3_NEG_MAX_FRAC  # 0.26


@dataclass
class AttemptReport:
    attempt: int
    seed: int
    ok: bool
    errors: list[str] = field(default_factory=list)
    bili_before: dict = field(default_factory=dict)
    bili_after: dict = field(default_factory=dict)
    keep_neg: int = 0
    mixed_n: int = 0
    mixed_neg_frac: float = 0.0
    bili_neg_frac: float = 0.0


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _backfill_content_type(row: dict) -> dict:
    out = dict(row)
    if not out.get("content_type"):
        ct = derive_content_type(out)
        if ct:
            out["content_type"] = ct
    return out


def bili_neg_keep_bounds(
    n_non_neg: int,
    n_neg_avail: int,
    fixed_n: int,
    fixed_neg: int,
    neg_min: float = NEG_MIN_FRAC,
    neg_max: float = NEG_MAX_FRAC,
) -> tuple[int, int, list[str]]:
    """计算 B站可保留负面条数 [lo, hi]。

    - 片内：keep/(non_neg+keep) ≤ neg_max  → keep ≤ floor(non_neg*neg_max/(1-neg_max))
    - 全集：混训负面占比落在 [neg_min, neg_max]
    返回 (lo, hi, infeasible_reasons)；若不可行则 lo>hi。
    """
    reasons: list[str] = []
    if n_non_neg < 0 or n_neg_avail < 0:
        return 0, -1, ["non_neg/neg_avail 非法"]

    # 片内上限（输出占比 ≤ neg_max）
    hi_slice = int(n_non_neg * neg_max / (1.0 - neg_max)) if neg_max < 1 else n_neg_avail
    hi_slice = max(0, min(hi_slice, n_neg_avail))

    # 全集： (fixed_neg + k) / (fixed_n + n_non_neg + k) ∈ [neg_min, neg_max]
    # k ≥ ceil( (neg_min*(fixed_n+n_non_neg) - fixed_neg) / (1-neg_min) )
    denom_lo = 1.0 - neg_min
    raw_lo = (neg_min * (fixed_n + n_non_neg) - fixed_neg) / denom_lo
    lo_mix = int(math.ceil(raw_lo - 1e-12)) if raw_lo > 0 else 0
    lo_mix = max(0, lo_mix)

    # k ≤ floor( (neg_max*(fixed_n+n_non_neg) - fixed_neg) / (1-neg_max) )
    denom_hi = 1.0 - neg_max
    raw_hi = (neg_max * (fixed_n + n_non_neg) - fixed_neg) / denom_hi
    hi_mix = int(math.floor(raw_hi + 1e-12))
    hi_mix = max(0, hi_mix)

    lo = lo_mix
    hi = min(hi_slice, hi_mix, n_neg_avail)

    if hi_slice < lo_mix:
        reasons.append(
            f"片内上限 keep≤{hi_slice} 无法满足全集负面≥{neg_min:.0%} 所需 keep≥{lo_mix}"
        )
    if hi_mix < lo_mix:
        reasons.append(
            f"全集约束无解：neg∈[{neg_min:.0%},{neg_max:.0%}] 要求 keep∈[{lo_mix},{hi_mix}]"
        )
    if n_neg_avail < lo_mix:
        reasons.append(
            f"可用负面仅 {n_neg_avail} < 全集下限所需 {lo_mix}"
        )
    if lo > hi:
        if not reasons:
            reasons.append(f"可行区间为空 keep∈[{lo},{hi}]")
    return lo, hi, reasons


def downsample_bili_neg(
    rows: list[dict],
    keep_neg: int,
    seed: int,
) -> list[dict]:
    """保留全部非负 + 恰好 keep_neg 条负面（随机，seed 可控）。"""
    df = pd.DataFrame(rows)
    non_neg = df[df["final_label"] != "负面"]
    neg = df[df["final_label"] == "负面"]
    if keep_neg > len(neg):
        raise ValueError(f"keep_neg={keep_neg} > available={len(neg)}")
    if keep_neg < 0:
        raise ValueError(f"keep_neg={keep_neg}")
    keep = neg.sample(n=keep_neg, random_state=seed) if keep_neg else neg.iloc[0:0]
    out = pd.concat([non_neg, keep], ignore_index=True)
    return out.sample(frac=1, random_state=seed).to_dict(orient="records")


def downsample_slice_neg(
    rows: list[dict],
    neg_max_frac: float = NEG_MAX_FRAC,
    seed: int = SEED,
) -> list[dict]:
    """兼容旧接口：仅按片内输出占比 ≤ neg_max_frac 降采样。"""
    df = pd.DataFrame(rows)
    n_neg = int((df["final_label"] == "负面").sum())
    if not len(df) or n_neg / len(df) <= neg_max_frac:
        return rows
    n_non = int((df["final_label"] != "负面").sum())
    max_neg = int(n_non * neg_max_frac / (1.0 - neg_max_frac))
    max_neg = max(0, min(max_neg, n_neg))
    return downsample_bili_neg(rows, max_neg, seed)


def validate_bili_slice(
    kept: list[dict],
    original: list[dict],
    neg_max: float = NEG_MAX_FRAC,
) -> list[str]:
    errs: list[str] = []
    if not kept:
        return ["B站降采样结果为空"]
    labels = Counter(r["final_label"] for r in kept)
    neg_frac = labels.get("负面", 0) / len(kept)
    if neg_frac > neg_max + 1e-12:
        errs.append(f"B站负面占比 {neg_frac:.4f} > {neg_max:.2f}")

    orig_non = {r["id"] for r in original if r["final_label"] != "负面"}
    kept_non = {r["id"] for r in kept if r["final_label"] != "负面"}
    if kept_non != orig_non:
        missing = sorted(orig_non - kept_non)[:5]
        extra = sorted(kept_non - orig_non)[:5]
        errs.append(f"B站非负行被改动 missing={missing} extra={extra}")

    orig_neg_ids = {r["id"] for r in original if r["final_label"] == "负面"}
    kept_neg_ids = {r["id"] for r in kept if r["final_label"] == "负面"}
    if not kept_neg_ids.issubset(orig_neg_ids):
        errs.append("B站保留负面 id 不在原片内")

    ids = [r["id"] for r in kept]
    if len(ids) != len(set(ids)):
        errs.append("B站降采样后 id 重复")
    return errs


def validate_mixed(
    mixed: list[dict],
    base_ids: set[str],
    neg_min: float = NEG_MIN_FRAC,
    neg_max: float = NEG_MAX_FRAC,
) -> list[str]:
    errs: list[str] = []
    if not mixed:
        return ["混训集为空"]
    labels = Counter(r["final_label"] for r in mixed)
    neg_frac = labels.get("负面", 0) / len(mixed)
    if neg_frac < neg_min - 1e-12:
        errs.append(f"混训负面占比 {neg_frac:.4f} < 下限 {neg_min:.2f}")
    if neg_frac > neg_max + 1e-12:
        errs.append(f"混训负面占比 {neg_frac:.4f} > 上限 {neg_max:.2f}")

    ids = [r["id"] for r in mixed]
    if len(ids) != len(set(ids)):
        errs.append("混训 id 重复")

    r6_ids = [r["id"] for r in mixed if r["id"] not in base_ids]
    if set(r6_ids) & base_ids:
        errs.append("混训与 v7_1 出现 id 重合")

    for i, r in enumerate(mixed):
        if r.get("final_label") not in LABEL2ID:
            errs.append(f"第{i}行 final_label 非法: {r.get('final_label')}")
            break
        if r.get("final_emotion") not in EMOTION_TAGS:
            errs.append(f"第{i}行 final_emotion 非法: {r.get('final_emotion')}")
            break
        if (r.get("platform"), r.get("content_type")) not in PREFIXES:
            errs.append(
                f"第{i}行前缀不可解析 platform={r.get('platform')} ct={r.get('content_type')}"
            )
            break
        if not r.get("content_type"):
            errs.append(f"第{i}行缺少 content_type")
            break
    return errs


def build_attempt(
    base: list[dict],
    r6_weibo: list[dict],
    r6_bili: list[dict],
    keep_neg: int,
    seed: int,
    attempt: int,
) -> tuple[list[dict] | None, AttemptReport]:
    before = Counter(r["final_label"] for r in r6_bili)
    bili_kept = downsample_bili_neg(r6_bili, keep_neg, seed)
    after = Counter(r["final_label"] for r in bili_kept)
    bili_neg_frac = after.get("负面", 0) / max(1, len(bili_kept))

    errs = validate_bili_slice(bili_kept, r6_bili)
    mixed = [_backfill_content_type(r) for r in base + r6_weibo + bili_kept]
    base_ids = {r["id"] for r in base}
    errs.extend(validate_mixed(mixed, base_ids))

    labels = Counter(r["final_label"] for r in mixed)
    report = AttemptReport(
        attempt=attempt,
        seed=seed,
        ok=not errs,
        errors=errs,
        bili_before=dict(before),
        bili_after=dict(after),
        keep_neg=keep_neg,
        mixed_n=len(mixed),
        mixed_neg_frac=labels.get("负面", 0) / max(1, len(mixed)),
        bili_neg_frac=bili_neg_frac,
    )
    return (mixed if report.ok else None), report


def main() -> None:
    base_path = Path(LABELED_DIR) / "v7_1_train_mixed.jsonl"
    r6_dir = Path(LABELED_DIR) / "round6"
    out_path = Path(LABELED_DIR) / OUT_NAME

    base = _load_jsonl(base_path)
    r6: list[dict] = []
    for p in sorted(r6_dir.glob("shard_*.jsonl")):
        r6.extend(_load_jsonl(p))

    if len(base) != 6895:
        raise SystemExit(f"v7_1 条数异常: {len(base)} != 6895")
    if len(r6) != 1200:
        raise SystemExit(f"round6 条数异常: {len(r6)} != 1200")
    if {r["id"] for r in base} & {r["id"] for r in r6}:
        raise SystemExit("v7_1 与 round6 id 重合，拒绝混训")

    r6_bili = [r for r in r6 if r["platform"] == "bilibili"]
    r6_weibo = [r for r in r6 if r["platform"] == "weibo"]
    if len(r6_bili) + len(r6_weibo) != len(r6):
        raise SystemExit("round6 存在非 weibo/bilibili 平台行")

    fixed = base + r6_weibo
    fixed_n = len(fixed)
    fixed_neg = sum(1 for r in fixed if r["final_label"] == "负面")
    n_non = sum(1 for r in r6_bili if r["final_label"] != "负面")
    n_neg_avail = sum(1 for r in r6_bili if r["final_label"] == "负面")

    lo, hi, infeas = bili_neg_keep_bounds(n_non, n_neg_avail, fixed_n, fixed_neg)
    print(
        f"[bounds] B站 keep_neg ∈ [{lo}, {hi}] "
        f"(non_neg={n_non} neg_avail={n_neg_avail} fixed_n={fixed_n} fixed_neg={fixed_neg})"
    )
    if lo > hi:
        print("[终止] 降采样在约束下无可行解，不再重试。")
        for r in infeas:
            print(f"  - {r}")
        print(
            "原因总结：round6 B站负面过高，压到片内≤26% 后，"
            "混训全集负面会掉出 21–26% 目标带；或全集上下限互相矛盾。"
            "需改配额策略（例如丢弃更多 B站抗议负向并另补中性玩梗，"
            "或接受放宽片内/全集某一侧约束），而不是换 seed。"
        )
        raise SystemExit(2)

    # 优先取区间中点偏片内上限（多留一点难例负向，但仍在带内）
    keep_neg = min(hi, max(lo, (lo + hi) // 2))
    reports: list[AttemptReport] = []
    mixed: list[dict] | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        seed = SEED + attempt - 1
        print(f"\n[attempt {attempt}/{MAX_ATTEMPTS}] seed={seed} keep_neg={keep_neg}")
        mixed, report = build_attempt(
            base, r6_weibo, r6_bili, keep_neg=keep_neg, seed=seed, attempt=attempt
        )
        reports.append(report)
        print(
            f"  bili {report.bili_before} -> {report.bili_after} "
            f"bili_neg={report.bili_neg_frac:.4f} mix_n={report.mixed_n} "
            f"mix_neg={report.mixed_neg_frac:.4f}"
        )
        if report.ok:
            print(f"  [通过] attempt={attempt}")
            break
        print(f"  [未过] {report.errors}")
        mixed = None

    if mixed is None:
        print(f"\n[终止] {MAX_ATTEMPTS} 轮采样均未通过校验。")
        for rep in reports:
            print(f"  round{rep.attempt} seed={rep.seed}: {rep.errors}")
        print("原因总结：")
        print("  1. 可行 keep_neg 区间非空，但具体随机子集未通过字段/分布校验；")
        print("  2. 请检查 validate 报错（标签/情绪/前缀/id）；")
        print("  3. 若仅分布抖动，可增大 MAX_ATTEMPTS 或固定 keep_neg 后只换 seed；")
        print("  4. 若结构性失败（缺 content_type 等），换 seed 无效，需修数据。")
        raise SystemExit(1)

    os.makedirs(LABELED_DIR, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in mixed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    last = reports[-1]
    print(
        f"\n[OK] {out_path} n={last.mixed_n} "
        f"bili {last.bili_before}->{last.bili_after} "
        f"dropped_neg={last.bili_before.get('负面', 0) - last.bili_after.get('负面', 0)} "
        f"neg_frac={last.mixed_neg_frac:.4f} attempts={len(reports)}"
    )
    print(f"[slice] {dict(Counter((r['platform'], r['content_type']) for r in mixed))}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
