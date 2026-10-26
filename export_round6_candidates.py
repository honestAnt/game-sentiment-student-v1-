# -*- coding: utf-8 -*-
"""round6 候选导出：StarRocks DWD 多游戏三分片（评论短板蒸馏）。

配额（config.ROUND6_*）：weibo评论 50%（约 30% 难例=应援/低置信正面）+
weibo博文 25% + B站评论 25%（抗议/动员优先）。LoL 不整批排除，单域封顶。
金标/已训练/抽检文本精确匹配排除。student_v8 打分只进 sidecar，不进 teacher 上下文。

用法：
    python export_round6_candidates.py
产出：
    data/candidates/round6/shard_XX.jsonl
    data/candidates/round6/score_sidecar.jsonl
"""
import hashlib
import argparse
import glob
import json
import os

import pandas as pd
import pymysql
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    CANDIDATE_DIR,
    ROUND6_SAMPLE_SIZE,
    ROUND6_SEED,
    ROUND6_SHARD_SIZE,
    ROUND6_CHEER_PATTERN,
    ROUND6_PROTEST_PATTERN,
)
from export_gold_starrocks_weibo_comments import (
    DOMAIN_PATTERNS,
    EXCLUSION_PATTERNS,
    assign_domain,
    load_env_file,
)
from export_starrocks_candidates import candidate_record
from infer_batch import batch_infer
from platform_input import model_input_text
from sampling import build_round6_sample

OUT_DIR = os.path.join(CANDIDATE_DIR, "round6")
CONTEXT_TRUNC = 120
MIN_POST_LEN = 10
MIN_COMMENT_LEN = 2
MYSQL_CHEER = r"加油|fighting|恭喜|打call|yyds|夺冠|必胜|挺你|冲[!！鸭]"
MYSQL_PROTEST = r"抵制|联名|道歉|退钱|倒闭|圈钱|别买|卸载|别充|问责|骗钱|退游"

EMPTY_COLS = ["id", "pid", "text", "platform", "content_type", "source_table", "domain", "ctx"]

# 有界拉取：难例 REGEXP + id hash 抽样，避免全表 JOIN 把会话拖死。
WEIBO_COMMENTS_CHEER_SQL = """
SELECT c.weibo_post_comment_id, c.weibo_post_id, c.weibo_post_comment_text,
       COALESCE(p.weibo_post_hashtag_topics, ''),
       COALESCE(SUBSTRING(p.weibo_post_content_details, 1, 200), '')
FROM dwd_crawl_weibo_post_comments_details_merged c
JOIN dwd_crawl_weibo_post_details_merged p
  ON c.weibo_post_id = p.weibo_post_id
WHERE c.weibo_post_comment_tier = '1' AND c.weibo_post_comment_valid_status = '1'
  AND c.weibo_post_comment_text REGEXP %s
LIMIT %s
"""

WEIBO_COMMENTS_HASH_SQL = """
SELECT c.weibo_post_comment_id, c.weibo_post_id, c.weibo_post_comment_text,
       COALESCE(p.weibo_post_hashtag_topics, ''),
       COALESCE(SUBSTRING(p.weibo_post_content_details, 1, 200), '')
FROM dwd_crawl_weibo_post_comments_details_merged c
JOIN dwd_crawl_weibo_post_details_merged p
  ON c.weibo_post_id = p.weibo_post_id
WHERE c.weibo_post_comment_tier = '1' AND c.weibo_post_comment_valid_status = '1'
  AND CAST(c.weibo_post_comment_id AS BIGINT) %% %s = %s
LIMIT %s
"""

WEIBO_POSTS_HASH_SQL = """
SELECT weibo_post_id, weibo_post_content_details, COALESCE(weibo_post_hashtag_topics, '')
FROM dwd_crawl_weibo_post_details_merged
WHERE weibo_post_valid_status = '1'
  AND abs(murmur_hash3_32(weibo_post_id)) %% %s = %s
LIMIT %s
"""

WEIBO_POSTS_FALLBACK_SQL = """
SELECT weibo_post_id, weibo_post_content_details, COALESCE(weibo_post_hashtag_topics, '')
FROM dwd_crawl_weibo_post_details_merged
WHERE weibo_post_valid_status = '1'
LIMIT %s
"""

BILI_COMMENTS_PROTEST_SQL = """
SELECT c.bili_video_comment_id, c.bili_video_comment_text,
       COALESCE(v.bili_video_title, ''), COALESCE(v.bili_video_topics, '')
FROM dwd_crawl_bili_video_comments_details c
JOIN dwd_crawl_bili_video_details v
  ON c.bili_video_bvid = v.bili_video_bvid
WHERE c.bili_video_comment_tier = 1 AND c.bili_video_comment_valid_status = 1
  AND c.bili_video_comment_text REGEXP %s
LIMIT %s
"""

BILI_COMMENTS_HASH_SQL = """
SELECT c.bili_video_comment_id, c.bili_video_comment_text,
       COALESCE(v.bili_video_title, ''), COALESCE(v.bili_video_topics, '')
FROM dwd_crawl_bili_video_comments_details c
JOIN dwd_crawl_bili_video_details v
  ON c.bili_video_bvid = v.bili_video_bvid
WHERE c.bili_video_comment_tier = 1 AND c.bili_video_comment_valid_status = 1
  AND CAST(c.bili_video_comment_id AS BIGINT) %% %s = %s
LIMIT %s
"""

EXTRA_EXCLUDE = (
    "data/validation/v8_iid_spotcheck/*.jsonl",
    "data/labeled/v7b_train_mixed.jsonl",
    "data/labeled/v7_train_mixed.jsonl",
)


def connect_sr():
    load_env_file()
    from config import require_starrocks_env
    cfg = require_starrocks_env()
    return pymysql.connect(
        host=cfg["host"],
        port=cfg["port"],
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
        charset="utf8mb4",
        connect_timeout=10,
        read_timeout=180,
    )


def load_excluded_texts() -> set:
    excluded = set()
    for pat in list(EXCLUSION_PATTERNS) + list(EXTRA_EXCLUDE):
        for f in glob.glob(pat):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        excluded.add(json.loads(line)["text"])
    return excluded


def _assign(*fields) -> str:
    import re
    blob = " ".join(str(f or "") for f in fields)
    for name, pat in DOMAIN_PATTERNS:
        if re.search(pat, blob):
            return name
    return "其他"


def _cap_pool(df: pd.DataFrame, n_cap: int, priority: pd.Series, seed: int) -> pd.DataFrame:
    """优先保留 priority 行，再随机补到 n_cap。单帖/单域封顶放到 SQL 有界拉取，这里不再二次挤兑。"""
    if df.empty or n_cap <= 0:
        return df
    if len(df) <= n_cap:
        return df.reset_index(drop=True)
    pri = df[priority.reindex(df.index).fillna(False)]
    rest = df[~df["id"].isin(pri["id"])]
    keep = pri if len(pri) <= n_cap else pri.sample(n=n_cap, random_state=seed)
    still = n_cap - len(keep)
    if still > 0 and len(rest):
        keep = pd.concat([keep, rest.sample(n=min(still, len(rest)), random_state=seed)])
    return keep.reset_index(drop=True)


def _empty_frame():
    return pd.DataFrame(columns=EMPTY_COLS)


def _wc_rows(raw, excluded, seen):
    recs = []
    for cid, pid, text, topics, ctx in raw:
        text = str(text or "").strip()
        if len(text) < MIN_COMMENT_LEN or text in excluded or text in seen:
            continue
        seen.add(text)
        recs.append({
            "id": f"r6_wc_{cid}", "pid": str(pid), "text": text,
            "platform": "weibo", "content_type": "评论",
            "source_table": "dwd_crawl_weibo_post_comments_details_merged",
            "domain": assign_domain(str(topics), str(ctx)),
            "ctx": str(ctx)[:CONTEXT_TRUNC],
        })
    return recs


def _wp_rows(raw, excluded, seen):
    recs = []
    for pid, content, topics in raw:
        text = str(content or "").strip()
        if len(text) < MIN_POST_LEN or text in excluded or text in seen:
            continue
        seen.add(text)
        recs.append({
            "id": f"r6_wp_{pid}", "pid": str(pid), "text": text,
            "platform": "weibo", "content_type": "博文",
            "source_table": "dwd_crawl_weibo_post_details_merged",
            "domain": _assign(topics, text),
            "ctx": text[:CONTEXT_TRUNC],
        })
    return recs


def _bc_rows(raw, excluded, seen):
    recs = []
    for cid, text, title, topics in raw:
        text = str(text or "").strip()
        if len(text) < MIN_COMMENT_LEN or text in excluded or text in seen:
            continue
        seen.add(text)
        recs.append({
            "id": f"r6_bc_{cid}", "pid": f"bv_{abs(hash(title or cid)) & 0xffffff}",
            "text": text, "platform": "bilibili", "content_type": "评论",
            "source_table": "dwd_crawl_bili_video_comments_details",
            "domain": _assign(title, topics),
            "ctx": str(title or "")[:CONTEXT_TRUNC],
        })
    return recs


def fetch_frames(cur, excluded: set, seed: int):
    hash_mod, hash_rem = 23, seed % 23
    print("[拉取] weibo 评论（应援 REGEXP + hash 抽样）")
    seen = set()
    cur.execute(WEIBO_COMMENTS_CHEER_SQL, (MYSQL_CHEER, 4000))
    recs = _wc_rows(cur.fetchall(), excluded, seen)
    print(f"  cheer {len(recs)}", flush=True)
    cur.execute(WEIBO_COMMENTS_HASH_SQL, (hash_mod, hash_rem, 12000))
    recs += _wc_rows(cur.fetchall(), excluded, seen)
    wc = pd.DataFrame(recs) if recs else _empty_frame()
    print(f"  去重+排除后 {len(wc)}", flush=True)

    print("[拉取] weibo 博文（hash 抽样）")
    seen = set()
    cur.execute(WEIBO_POSTS_HASH_SQL, (hash_mod, hash_rem, 8000))
    raw = cur.fetchall()
    print(f"  hash raw {len(raw)}", flush=True)
    recs = _wp_rows(raw, excluded, seen)
    if len(recs) < 500:
        cur.execute(WEIBO_POSTS_FALLBACK_SQL, (8000,))
        raw = cur.fetchall()
        print(f"  fallback raw {len(raw)}", flush=True)
        recs = _wp_rows(raw, excluded, seen)
    wp = pd.DataFrame(recs) if recs else _empty_frame()
    print(f"  去重+排除后 {len(wp)}", flush=True)

    print("[拉取] bili 评论（抗议 REGEXP + hash 抽样）")
    seen = set()
    cur.execute(BILI_COMMENTS_PROTEST_SQL, (MYSQL_PROTEST, 2000))
    recs = _bc_rows(cur.fetchall(), excluded, seen)
    print(f"  protest {len(recs)}", flush=True)
    cur.execute(BILI_COMMENTS_HASH_SQL, (hash_mod, hash_rem, 8000))
    recs += _bc_rows(cur.fetchall(), excluded, seen)
    bc = pd.DataFrame(recs) if recs else _empty_frame()
    print(f"  去重+排除后 {len(bc)}", flush=True)
    return wc, wp, bc


def score_pool(df: pd.DataFrame, model_dir: str) -> pd.DataFrame:
    if df.empty:
        df["student_pred"] = []
        df["student_conf"] = []
        return df
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[打分] {len(df)} 条 student_v8 @ {device}")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    texts = [
        model_input_text(t, p, ct)
        for t, p, ct in zip(df["text"], df["platform"], df["content_type"])
    ]
    bs = 16 if device == "mps" else 64
    labels, confs = batch_infer(texts, model, tok, device, batch_size=bs)
    out = df.copy()
    out["student_pred"] = labels
    out["student_conf"] = confs
    return out


def write_shards(sample: pd.DataFrame, out_dir: str, shard_size: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    n = len(sample)
    n_shards = (n + shard_size - 1) // shard_size
    for i in range(n_shards):
        chunk = sample.iloc[i * shard_size:(i + 1) * shard_size]
        path = os.path.join(out_dir, f"shard_{i:02d}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for _, row in chunk.iterrows():
                f.write(json.dumps(candidate_record(row), ensure_ascii=False) + "\n")
    side = os.path.join(out_dir, "score_sidecar.jsonl")
    with open(side, "w", encoding="utf-8") as f:
        for _, row in sample.iterrows():
            rec = {
                "id": str(row["id"]),
                "student_pred": row.get("student_pred"),
                "student_conf": None if pd.isna(row.get("student_conf")) else float(row["student_conf"]),
            }
            if "domain" in row and pd.notna(row.get("domain")):
                rec["domain"] = row["domain"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[导出] {n} 条 -> {n_shards} 个分片 + sidecar，目录 {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=OUT_DIR)
    parser.add_argument("--model-dir", default=os.path.join("models", "student_v8"))
    parser.add_argument("--n", type=int, default=ROUND6_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=ROUND6_SEED)
    parser.add_argument("--comment-pool", type=int, default=8000)
    parser.add_argument("--post-pool", type=int, default=4000)
    parser.add_argument("--bili-pool", type=int, default=4000)
    args = parser.parse_args()

    excluded = load_excluded_texts()
    print(f"[排除集] 已用文本 {len(excluded)} 条")

    conn = connect_sr()
    cur = conn.cursor()
    wc, wp, bc = fetch_frames(cur, excluded, args.seed)
    conn.close()

    wc_pri = wc["text"].str.contains(ROUND6_CHEER_PATTERN, regex=True, na=False) if len(wc) else pd.Series(dtype=bool)
    bc_pri = bc["text"].str.contains(ROUND6_PROTEST_PATTERN, regex=True, na=False) if len(bc) else pd.Series(dtype=bool)
    wp_pri = pd.Series(False, index=wp.index) if len(wp) else pd.Series(dtype=bool)

    wc = _cap_pool(wc, args.comment_pool, wc_pri, args.seed)
    wp = _cap_pool(wp, args.post_pool, wp_pri, args.seed)
    bc = _cap_pool(bc, args.bili_pool, bc_pri, args.seed)
    print(f"[池] 评论 {len(wc)} / 博文 {len(wp)} / B站 {len(bc)}")

    pool = pd.concat([wc, wp, bc], ignore_index=True)
    pool["content_hash"] = pool["text"].map(lambda t: hashlib.md5(str(t).encode("utf-8")).hexdigest())
    pool = pool.drop_duplicates(subset=["content_hash"]).reset_index(drop=True)
    scored = score_pool(pool, args.model_dir)

    sample = build_round6_sample(scored, n_total=args.n, exclude_texts=excluded, seed=args.seed)
    if sample["text"].isin(excluded).any():
        raise SystemExit("采样后仍与排除集文本重合")
    if sample["content_hash"].duplicated().any():
        raise SystemExit("采样内部存在重复文本")
    if sample["id"].duplicated().any():
        raise SystemExit("采样内部存在重复 id")

    slices = sample.groupby(["platform", "content_type"]).size().to_dict()
    print(f"[采样] {len(sample)} 条，构成 {slices}")
    if "domain" in sample.columns:
        print("[域]", sample["domain"].value_counts().to_dict())
    write_shards(sample, args.output_dir, ROUND6_SHARD_SIZE)


if __name__ == "__main__":
    main()
