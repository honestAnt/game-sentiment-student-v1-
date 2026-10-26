# -*- coding: utf-8 -*-
"""从本地未进训池挖 v10c 中性难例/软负面候选（不连 StarRocks）。

种子模式来自 docs/v9j误差解剖-B站护栏-2026-09-18.md。
金标/challenge 文本只作排除，不进候选。
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import re
from collections import Counter

import numpy as np

OUT_DIR = os.path.join("data", "candidates", "v10c_calib")

EXCLUDE_PATS = [
    "data/labeled/v9_train_mixed.jsonl",
    "data/labeled/v10a_train_mixed.jsonl",
    "data/labeled/v10b_train_mixed.jsonl",
    "data/validation/gold_eval_v6.jsonl",
    "data/validation/gold_eval_v7.jsonl",
    "data/validation/gold_bili_neg_challenge.jsonl",
    "data/candidates/gold_expansion_bili/*.jsonl",
]

BUCKETS = {
    "esports_chat": re.compile(r"反派|职业生涯|粉丝|赛段|大黄|韩华|T1|BLG|TES|打点|分段"),
    "diff_vent": re.compile(r"最难打|难打|劝退|卡成|肝不动|肝死"),
    "trade_slang": re.compile(r"抬价|黑市|出号|收号|走中|可刀|打金|出金|收金"),
    "meme_scam": re.compile(r"诈骗是吧|脚呢|诈骗|骗我是吧|这是诈骗|被骗了吧"),
    "report_neutral": re.compile(r"举报|跑路|退款|下架|客服"),
    "trash_neutral": re.compile(r"垃圾|道具池"),
    "soft_neg": re.compile(r"差点意思|敷衍|失望|拉胯|退游|不玩了"),
    "planner_stmt": re.compile(r"策划"),
}

QUOTA = {
    "esports_chat": 40,
    "diff_vent": 25,
    "trade_slang": 20,
    "meme_scam": 15,
    "report_neutral": 35,
    "trash_neutral": 20,
    "soft_neg": 40,
    "planner_stmt": 30,
}


def load_excluded() -> set[str]:
    excl: set[str] = set()
    for pat in EXCLUDE_PATS:
        for path in glob.glob(pat):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        excl.add(json.loads(line)["text"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return excl


def iter_smoke(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def iter_lol1600_xlsx(path: str):
    """无 openpyxl 时用 zip+xml 读 inlineStr 的 B站评论 xlsx。"""
    import xml.etree.ElementTree as ET
    import zipfile

    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in root.findall(f".//{ns}sheetData/{ns}row"):
        vals = []
        for c in row.findall(f"{ns}c"):
            is_elem = c.find(f"{ns}is")
            v = c.find(f"{ns}v")
            if is_elem is not None:
                t = is_elem.find(f".//{ns}t")
                vals.append(t.text if t is not None else "")
            elif v is not None:
                vals.append(v.text or "")
            else:
                vals.append("")
        rows.append(vals)
    if not rows:
        return
    header = rows[0]
    try:
        text_i = header.index("评论文本")
    except ValueError:
        return
    for vals in rows[1:]:
        if text_i >= len(vals):
            continue
        yield {"text": vals[text_i], "platform": "bilibili", "content_type": "评论"}


def main():
    excl = load_excluded()
    print(f"[排除] {len(excl)}", flush=True)

    pool, seen = [], set()

    def add(text, platform, content_type, src):
        text = (text or "").strip()
        if not (4 <= len(text) <= 280):
            return
        if text in excl or text in seen:
            return
        # 优先评论；博文仅作补充
        if content_type not in ("评论", "博文"):
            content_type = "评论"
        hits = [k for k, p in BUCKETS.items() if p.search(text)]
        if not hits:
            return
        seen.add(text)
        cid = hashlib.md5(text.encode()).hexdigest()[:12]
        pool.append({
            "id": f"v10c_{src}_{cid}",
            "text": text,
            "platform": platform,
            "content_type": content_type,
            "source_table": src,
            "content_hash": hashlib.sha1(text.encode()).hexdigest()[:16],
            "buckets": hits,
        })

    lol_xlsx = "data/raw/LoL评论子agent标注（1600条）_B站.xlsx"
    if os.path.exists(lol_xlsx):
        n = 0
        try:
            for row in iter_lol1600_xlsx(lol_xlsx):
                add(row["text"], "bilibili", "评论", "lol1600_bili")
                n += 1
            print(f"[读] {lol_xlsx} rows≈{n} pool={len(pool)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[跳过 xlsx] {e}", flush=True)

    for smoke in ("data/raw/smoke_10k.csv", "data/raw/smoke_200.csv"):
        if not os.path.exists(smoke):
            continue
        n = 0
        for row in iter_smoke(smoke):
            text = row.get("text") or row.get("内容") or ""
            platform = row.get("platform") or "bilibili"
            ct = row.get("content_type") or "评论"
            if platform not in ("bilibili", "weibo"):
                continue
            if ct not in ("评论", "博文"):
                ct = "评论"
            add(text, platform, ct, os.path.basename(smoke).replace(".csv", ""))
            n += 1
        print(f"[读] {smoke} rows≈{n} pool={len(pool)}", flush=True)

    # 本地 B 站评论 CSV（补稀缺桶 + 平台偏好；非金标）
    for bcsv in sorted(glob.glob("data/raw/*bili_video_comments*.csv")):
        n = 0
        with open(bcsv, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                text = row.get("文本_清洗") or row.get("文本") or ""
                add(text, "bilibili", "评论", "bili_video_comments")
                n += 1
                if n >= 80000:
                    break
        print(f"[读] {bcsv} rows≈{n} pool={len(pool)}", flush=True)

    # 已有 jsonl 候选池里捞（round 候选若未进 v9 混集，排除集已挡）
    for pat in (
        "data/candidates/round2/pool_scored.jsonl",
        "data/candidates/round3/pool_scored.jsonl",
    ):
        if not os.path.exists(pat):
            continue
        with open(pat, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = rec.get("text") or ""
                platform = rec.get("platform") or "bilibili"
                ct = rec.get("content_type") or "评论"
                if platform not in ("bilibili", "weibo"):
                    continue
                add(text, platform, ct, os.path.basename(os.path.dirname(pat)))
        print(f"[读] {pat} pool={len(pool)}", flush=True)

    print(f"[池] {len(pool)} {Counter(r['source_table'] for r in pool)}", flush=True)
    print(f"[桶] {Counter(b for r in pool for b in r['buckets'])}", flush=True)
    print(f"[平台池] {Counter(r['platform'] for r in pool)}", flush=True)

    rng = np.random.RandomState(42)
    by_b = {k: [] for k in BUCKETS}
    for r in pool:
        for b in r["buckets"]:
            by_b[b].append(r)

    picked, used = [], set()
    for b, n in QUOTA.items():
        cand = [r for r in by_b[b] if r["text"] not in used]
        rng.shuffle(cand)
        # 优先 bilibili 评论，再 weibo 评论，最后博文
        def _prio(r):
            plat = 0 if r["platform"] == "bilibili" else 1
            ct = 0 if r.get("content_type") == "评论" else 1
            return (plat, ct)

        cand.sort(key=_prio)
        for r in cand[:n]:
            used.add(r["text"])
            rr = dict(r)
            rr["primary_bucket"] = b
            picked.append(rr)

    print(f"[抽样] {len(picked)} {Counter(r['primary_bucket'] for r in picked)}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    shard_size = 50
    for i in range(0, len(picked), shard_size):
        chunk = picked[i : i + shard_size]
        path = os.path.join(OUT_DIR, f"shard_{i // shard_size:02d}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in chunk:
                out = {k: r[k] for k in (
                    "id", "text", "platform", "content_type",
                    "source_table", "content_hash", "primary_bucket", "buckets",
                )}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"[写出] {path} n={len(chunk)}", flush=True)


if __name__ == "__main__":
    main()
