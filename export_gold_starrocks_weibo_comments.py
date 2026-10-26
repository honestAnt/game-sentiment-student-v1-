# -*- coding: utf-8 -*-
"""
export_gold_starrocks_weibo_comments.py
从 StarRocks DWD 层拉取多游戏/多域 weibo评论，导出金标扩样候选交人工标注。

表在环境变量 STARROCKS_DATABASE 下，默认表名：
    dwd_crawl_weibo_post_comments_details_merged（评论平表，带 weibo_post_id）
    dwd_crawl_weibo_post_details_merged（帖子平表，hashtag + 正文）
请按自己的仓表修改表名与字段。

采样规则：tier=1 直评 + valid；按帖子 hashtag+正文关键词做域标注；LoL 域整批排除
（另一批 lol_candidates_220 已覆盖该域）；单域上限（默认 50）强制多域混合；
单帖上限（默认 2）；排除集全量文本匹配；随机固定 seed。

凭证：STARROCKS_HOST / PORT / USER / PASSWORD / DATABASE 必须来自环境变量或项目根
.env（不覆盖已有环境变量）。不设默认主机、账号、库名。不要把 .env 提交到仓库。

用法：
    python export_gold_starrocks_weibo_comments.py            # 默认 220 条
    python export_gold_starrocks_weibo_comments.py -n 300

产出：data/candidates/gold_expansion_weibo_comment/
    starrocks_candidates_<n>.xlsx   人工在「人工标注-杠」列填 中立/负向/正向 或 中性/负面/正面
    starrocks_candidates_<n>.jsonl  候选行存档（id 带 ge6_ 前缀）

标注完成后用 merge_gold_weibo_comments.py --labeled_xlsx <xlsx...> 一次传入全部文件回收为 gold_eval_v6。
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import pymysql

OUT_DIR = os.path.join("data", "candidates", "gold_expansion_weibo_comment")
ENV_FILE = ".env"

COMMENTS_TBL = "dwd_crawl_weibo_post_comments_details_merged"
POSTS_TBL = "dwd_crawl_weibo_post_details_merged"

JOIN_SQL = f"""
SELECT c.weibo_post_comment_id, c.weibo_post_id, c.weibo_post_comment_text,
       COALESCE(p.weibo_post_hashtag_topics, ''),
       COALESCE(SUBSTRING(p.weibo_post_content_details, 1, 200), '')
FROM {COMMENTS_TBL} c
JOIN {POSTS_TBL} p ON c.weibo_post_id = p.weibo_post_id
WHERE c.weibo_post_comment_tier = '1' AND c.weibo_post_comment_valid_status = '1'
"""

EXCLUSION_PATTERNS = [
    "data/labeled/round1_labeled.jsonl",
    "data/labeled/round2_labeled.jsonl",
    "data/labeled/round4_labeled.jsonl",
    "data/labeled/round4_train_slice.jsonl",
    "data/labeled/round5_labeled.jsonl",
    "data/labeled/round3_train_mixed.jsonl",
    "data/labeled/round3_train_mixed_os.jsonl",
    "data/labeled/v5_train_mixed.jsonl",
    "data/labeled/v6_train_mixed.jsonl",
    "data/labeled/v7_train_mixed.jsonl",
    "data/labeled/v7_1_train_mixed.jsonl",
    "data/labeled/round3/*.jsonl",
    "data/labeled/round3_full/*.jsonl",
    "data/labeled/round4-claude/*.jsonl",
    "data/validation/validation_gold.jsonl",
    "data/validation/gold_eval_v6.jsonl",
    "data/validation/gold_eval_v7.jsonl",
    "data/validation/gold_bili_neg_challenge.jsonl",
    "data/validation/round4_holdout.jsonl",
    "data/validation/v5_spotcheck_raw200.jsonl",
    "data/validation/v5_spotcheck_raw200_teacher.jsonl",
    "data/validation/v5_spotcheck_raw400.jsonl",
    "data/candidates/gold_expansion_weibo_comment/*.jsonl",
]

# 域标注：hashtag + 主贴正文联合匹配（顺序即优先级）；LoL 单独列出用于整批排除
DOMAIN_PATTERNS = [
    ("LoL", r"英雄联盟|LPL|LCK|全球总决赛|MSI季中|S1[0-9]瑞士|召唤师|德玛西亚|云顶"),
    ("异环", r"异环|残虹"),
    ("归龙潮", r"归龙潮|龙吟狮醒|青龙元君|龙胤"),
    ("雾中/星回", r"雾中|朔望星回"),
    ("ChinaJoy", r"ChinaJoy|chinajoy|CJ骁龙"),
    ("数码硬件", r"K100|骁龙|一加|Redmi|红米|手机发布"),
    ("王者", r"王者荣耀|KPL"),
    ("原神", r"原神|提瓦特"),
    ("鸣潮", r"鸣潮|库洛"),
    ("绝区零", r"绝区零"),
    ("永劫", r"永劫无间"),
    ("第五人格", r"第五人格"),
]

ID_COL = "序号"
TYPE_COL = "类型"
CONTEXT_COL = "主贴内容（评论的上下文）"
TEXT_COL = "文本"
HUMAN_LABEL_COL = "人工标注-杠"

MAX_PER_POST = 2
MAX_PER_DOMAIN = 50
MIN_TEXT_LEN = 2
CONTEXT_TRUNC = 120


def load_env_file(path=ENV_FILE):
    """把 .env 里的 KEY=VALUE 注入 os.environ（不覆盖已有值）。"""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def connect():
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
    )


def load_excluded_texts() -> set:
    excluded = set()
    for pat in EXCLUSION_PATTERNS:
        for f in glob.glob(pat):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        excluded.add(json.loads(line)["text"])
    return excluded


def assign_domain(topics: str, ctx: str) -> str:
    for name, pat in DOMAIN_PATTERNS:
        if re.search(pat, topics) or re.search(pat, ctx):
            return name
    return "其他"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=220)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--max_per_domain", type=int, default=MAX_PER_DOMAIN)
    parser.add_argument("--max_per_post", type=int, default=MAX_PER_POST)
    args = parser.parse_args()

    conn = connect()
    cur = conn.cursor()
    cur.execute(JOIN_SQL)
    raw = cur.fetchall()
    conn.close()
    print(f"[拉取] tier1 有效评论 join 帖子: {len(raw)}")

    excluded = load_excluded_texts()
    print(f"[排除集] 已用文本 {len(excluded)} 条")

    # 既有候选的 ge6_ id（防与 LoL 批 comment_id 空间重叠；实测 0 冲突，防御性保留）
    existing_ids = set()
    for f in glob.glob(os.path.join(OUT_DIR, "*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            existing_ids |= {json.loads(line)["id"] for line in fh if line.strip()}

    recs, seen_text, n_id_hit = [], set(), 0
    for cid, pid, text, topics, ctx in raw:
        cid = str(cid)
        if f"ge6_{cid}" in existing_ids:
            n_id_hit += 1
            continue
        text = str(text).strip()
        if len(text) < MIN_TEXT_LEN or text in seen_text or text in excluded:
            continue
        seen_text.add(text)
        recs.append({"cid": cid, "pid": str(pid), "text": text,
                     "domain": assign_domain(str(topics), str(ctx)),
                     "ctx": str(ctx)[:CONTEXT_TRUNC]})
    if n_id_hit:
        print(f"[过滤] 与既有候选 id 重合排除 {n_id_hit} 条")
    print(f"[过滤] 去重+排除+长度后 {len(recs)} 条，域构成:",
          pd.Series([r["domain"] for r in recs]).value_counts().to_dict())

    recs = [r for r in recs if r["domain"] != "LoL"]

    df = pd.DataFrame(recs)
    df["_r1"] = np.random.RandomState(args.seed).rand(len(df))
    df = (df.sort_values(["pid", "_r1"])
          .groupby("pid", group_keys=False).head(args.max_per_post))
    df = (df.sort_values(["domain", "_r1"])
          .groupby("domain", group_keys=False).head(args.max_per_domain))
    sampled = df.sample(n=min(args.num, len(df)), random_state=args.seed).reset_index(drop=True)
    print(f"[采样] 单帖上限 {args.max_per_post}、单域上限 {args.max_per_domain} 后随机取 "
          f"{len(sampled)} 条（seed {args.seed}）")
    print("[采样域构成]:", sampled["domain"].value_counts().to_dict())

    # 终检：与既有候选（双盲15 / LoL批）id 空间零冲突，通过后才写文件
    new_ids = {f"ge6_{cid}" for cid in sampled["cid"]}
    overlap = existing_ids & new_ids
    assert not overlap, f"与既有候选 id 冲突: {sorted(overlap)[:5]}"

    os.makedirs(OUT_DIR, exist_ok=True)
    xlsx_out = os.path.join(OUT_DIR, f"starrocks_candidates_{len(sampled)}.xlsx")
    jsonl_out = os.path.join(OUT_DIR, f"starrocks_candidates_{len(sampled)}.jsonl")

    export = pd.DataFrame({
        ID_COL: sampled["cid"], TYPE_COL: "评论",
        CONTEXT_COL: sampled["ctx"], TEXT_COL: sampled["text"],
    })
    export[HUMAN_LABEL_COL] = None
    export.to_excel(xlsx_out, index=False)

    with open(jsonl_out, "w", encoding="utf-8") as f:
        for _, row in sampled.iterrows():
            f.write(json.dumps({
                "id": f"ge6_{row['cid']}", "text": row["text"],
                "platform": "weibo", "content_type": "评论",
                "domain": row["domain"], "context": row["ctx"],
            }, ensure_ascii=False) + "\n")

    print(f"[完成] 候选 {len(sampled)} 条已导出：\n  {xlsx_out}（人工在「人工标注-杠」列填写）\n  {jsonl_out}")
    print("[提示] 标注用词：中立/中性、负向/负面、正向/正面 均可，回收脚本会归一化。")


if __name__ == "__main__":
    main()
