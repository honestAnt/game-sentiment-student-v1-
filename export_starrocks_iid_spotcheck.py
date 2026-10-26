# -*- coding: utf-8 -*-
"""
export_starrocks_iid_spotcheck.py
从 StarRocks DWD 层抽三片各 100 条全新 i.i.d. 数据，供 v8 工作点抽检
（raw / T0.75 一刀切 / T0.75 负面豁免 三工作点对拍，解决审计遗留的取舍问题）。

与金标扩样批（export_gold_*）的区别：本批目的是**真实流量分布抽检**而非金标扩样，
LoL 域不整批排除（真实流量含 LoL），但设单域上限防止单域主导；B站片天然覆盖
非 LoL 域（异环/原神/绝区零），与金标 B站片（LoL 域）互补。

三片：
    weibo博文  dwd_crawl_weibo_post_details_merged（valid，正文>=10字）
    weibo评论  comments join posts（tier=1 valid=1，>=2字，带主贴上下文）
    bili评论   bili comments tier=1 join video（带视频标题上下文）

排除集与金标扩样共用（EXCLUSION_PATTERNS 已含 data/candidates/*.jsonl，
即双盲15 + LoL220 + StarRocks220 全部已导出候选），文本精确匹配 + ge6_ id 双口径。

产出：data/validation/v8_iid_spotcheck/candidates_<n>.jsonl
    字段 id(iid1_ 前缀)/text/platform/content_type/domain/ctx
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from export_gold_starrocks_weibo_comments import (
    DOMAIN_PATTERNS, EXCLUSION_PATTERNS, connect, load_env_file,
)

OUT_DIR = os.path.join("data", "validation", "v8_iid_spotcheck")

# 在金标/训练排除集之上，再挡掉已抽检与 round6 训练候选，避免泄漏。
EXTRA_EXCLUSION_PATTERNS = [
    "data/validation/v8_iid_spotcheck/*.jsonl",
    "data/validation/v9o_iid_spotcheck/*.jsonl",
    "data/validation/v9k_iid_spotcheck/*.jsonl",
    "data/labeled/round6/*.jsonl",
    "data/candidates/round6/*.jsonl",
    "data/labeled/v9*.jsonl",
    # v10 步骤 A 的 B站金标扩样（候选 + 回收产物）
    "data/candidates/gold_expansion_bili/*.jsonl",
    "data/validation/gold_eval_v7*.jsonl",
    "data/validation/gold_bili_neg_challenge.jsonl",
]

WEIBO_POSTS_SQL = """
SELECT weibo_post_id, weibo_post_content_details, COALESCE(weibo_post_hashtag_topics, '')
FROM dwd_crawl_weibo_post_details_merged
WHERE weibo_post_valid_status = '1'
"""

WEIBO_COMMENTS_SQL = """
SELECT c.weibo_post_comment_id, c.weibo_post_comment_text,
       COALESCE(p.weibo_post_hashtag_topics, ''),
       COALESCE(SUBSTRING(p.weibo_post_content_details, 1, 200), '')
FROM dwd_crawl_weibo_post_comments_details_merged c
JOIN dwd_crawl_weibo_post_details_merged p
  ON c.weibo_post_id = p.weibo_post_id
WHERE c.weibo_post_comment_tier = '1' AND c.weibo_post_comment_valid_status = '1'
"""

BILI_COMMENTS_SQL = """
SELECT c.bili_video_comment_id, c.bili_video_comment_text,
       COALESCE(v.bili_video_title, ''), COALESCE(v.bili_video_topics, '')
FROM dwd_crawl_bili_video_comments_details c
JOIN dwd_crawl_bili_video_details v
  ON c.bili_video_bvid = v.bili_video_bvid
WHERE c.bili_video_comment_tier = 1 AND c.bili_video_comment_valid_status = 1
"""

CONTEXT_TRUNC = 120
MAX_PER_DOMAIN = 40
MIN_POST_LEN = 10
MIN_COMMENT_LEN = 2


def load_excluded() -> set:
    excluded = set()
    for pat in list(EXCLUSION_PATTERNS) + EXTRA_EXCLUSION_PATTERNS:
        for f in glob.glob(pat):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        try:
                            excluded.add(json.loads(line)["text"])
                        except (json.JSONDecodeError, KeyError):
                            continue
    return excluded


def assign_domain(*fields) -> str:
    import re
    for name, pat in DOMAIN_PATTERNS:
        if any(re.search(pat, f) for f in fields if f):
            return name
    return "其他"


def sample_slice(cur, sql, row_fn, excluded, per_slice, seed):
    """row_fn: (raw_row) -> dict(cid,pid,text,domain,ctx) | None"""
    cur.execute(sql)
    raw = cur.fetchall()
    recs, seen, n_text_hit = [], set(), 0
    for row in raw:
        r = row_fn(row)
        if r is None:
            continue
        if r["text"] in excluded or r["text"] in seen:
            n_text_hit += 1
            continue
        seen.add(r["text"])
        recs.append(r)
    print(f"  池 {len(raw)} -> 去重+排除后 {len(recs)}（排除命中 {n_text_hit}）")
    df = pd.DataFrame(recs)
    rs = np.random.RandomState(seed)
    df["_r1"] = rs.rand(len(df))
    df = (df.sort_values(["pid", "_r1"]).groupby("pid", group_keys=False).head(2))
    n_after_post = len(df)
    df = (df.sort_values(["domain", "_r1"]).groupby("domain", group_keys=False).head(MAX_PER_DOMAIN))
    sampled = df.sample(n=min(per_slice, len(df)), random_state=seed).reset_index(drop=True)
    if len(sampled) < per_slice:
        # 静默少抽会让「三片各 N 条」的配额说明和实际产物不一致（v9o 批 167 -> 153）
        print(
            f"  [警告] 该片只抽到 {len(sampled)}/{per_slice} 条："
            f"每帖限 2 条后 {n_after_post}，单域上限 MAX_PER_DOMAIN={MAX_PER_DOMAIN} 后 {len(df)}。"
            f" 需要足量请提高 MAX_PER_DOMAIN 或放宽排除集。"
        )
    return sampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_slice", type=int, default=100)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--id_prefix", default="iid1",
                        help="id 前缀，如 iid1 / iid2，实际写成 iidX_wb_/wc_/bc_")
    args = parser.parse_args()

    load_env_file()
    excluded = load_excluded()
    print(f"[排除集] 已用文本 {len(excluded)} 条")

    conn = connect()
    cur = conn.cursor()
    pref = args.id_prefix.rstrip("_")

    def weibo_post_fn(row):
        pid, content, topics = str(row[0]), str(row[1] or "").strip(), str(row[2] or "")
        if len(content) < MIN_POST_LEN:
            return None
        return {"cid": pid, "pid": pid, "text": content,
                "domain": assign_domain(topics, content),
                "ctx": content[:CONTEXT_TRUNC]}

    def weibo_comment_fn(row):
        cid, text = str(row[0]), str(row[1] or "").strip()
        if len(text) < MIN_COMMENT_LEN:
            return None
        topics, ctx = str(row[2] or ""), str(row[3] or "")
        return {"cid": cid, "pid": f"wp_{hash(row[3]) & 0xffffff}", "text": text,
                "domain": assign_domain(topics, ctx),
                "ctx": ctx[:CONTEXT_TRUNC]}

    def bili_comment_fn(row):
        cid, text = str(row[0]), str(row[1] or "").strip()
        if len(text) < MIN_COMMENT_LEN:
            return None
        title, topics = str(row[2] or ""), str(row[3] or "")
        return {"cid": cid, "pid": f"bv_{abs(hash(text)) & 0xffff}", "text": text,
                "domain": assign_domain(title, topics),
                "ctx": title[:CONTEXT_TRUNC]}

    slices = [
        ("weibo博文", WEIBO_POSTS_SQL, weibo_post_fn, f"{pref}_wb_"),
        ("weibo评论", WEIBO_COMMENTS_SQL, weibo_comment_fn, f"{pref}_wc_"),
        ("bili评论", BILI_COMMENTS_SQL, bili_comment_fn, f"{pref}_bc_"),
    ]

    all_rows = []
    for name, sql, fn, idp in slices:
        print(f"[抽样] {name}")
        s = sample_slice(cur, sql, fn, excluded, args.per_slice, args.seed)
        print(f"  域构成: {s['domain'].value_counts().to_dict()}")
        for _, row in s.iterrows():
            all_rows.append({
                "id": f"{idp}{row['cid']}", "text": row["text"],
                "platform": "weibo" if "_wb_" in idp or "_wc_" in idp else "bilibili",
                "content_type": "博文" if "_wb_" in idp else "评论",
                "domain": row["domain"], "ctx": row["ctx"],
            })
    conn.close()

    out = {"weibo博文": [], "weibo评论": [], "bili评论": []}
    for r in all_rows:
        key = f"{r['platform']}{r['content_type']}" if r["platform"] == "weibo" else "bili评论"
        out[key].append(r)

    os.makedirs(args.out_dir, exist_ok=True)
    n_total = sum(len(v) for v in out.values())
    path = os.path.join(args.out_dir, f"candidates_{n_total}.jsonl")
    ids = [r["id"] for r in all_rows]
    assert len(ids) == len(set(ids)), "id 冲突"
    texts = [r["text"] for r in all_rows]
    assert len(texts) == len(set(texts)) and not (set(texts) & excluded), "文本重合"
    with open(path, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[完成] {n_total} 条 -> {path}")
    print("[片构成]", {k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
