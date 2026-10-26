# -*- coding: utf-8 -*-
"""从 StarRocks 抽样 round1 候选分片。密码只来自环境变量。"""
import argparse
import json
import os
import pandas as pd
import pymysql

from config import (
    CANDIDATE_DIR,
    COMMENT_SOURCE,
    REQUEST_ROW_MAX,
    REQUEST_ROW_MIN,
    ROUND1_N_SHARDS,
    ROUND1_SAMPLE_SIZE,
    ROUND1_SEED,
    ROUND1_SHARD_SIZE,
    require_starrocks_env,
)
from sampling import build_round1_sample, prepare_pool

JOIN_SQL = """
SELECT
  q.content_hash,
  q.request_content_text,
  q.request_data_source_clean_db_table_name_en AS source_table,
  r.sentiment_final_aliyun AS aliyun_final,
  r.positive_prob_aliyun,
  r.negative_prob_aliyun
FROM sentiment_aliyun_request q
LEFT JOIN sentiment_aliyun_request_result r
  ON q.request_data_source_clean_db_table_name_en = r.request_data_source_clean_db_table_name_en
 AND q.request_data_source_dedup_pk = r.request_data_source_dedup_pk
ORDER BY q.content_hash
"""

COUNT_SQL = "SELECT COUNT(*) AS n FROM sentiment_aliyun_request"


def candidate_record(row) -> dict:
    rec = {
        "id": str(row["id"]),
        "text": row["text"],
        "platform": row["platform"],
        "source_table": row["source_table"],
        "content_hash": str(row["content_hash"]),
    }
    if "content_type" in row and pd.notna(row.get("content_type")):
        rec["content_type"] = row["content_type"]
    return rec


def write_round1_outputs(sample_df: pd.DataFrame, out_dir: str, shard_size: int = ROUND1_SHARD_SIZE) -> None:
    os.makedirs(out_dir, exist_ok=True)
    n = len(sample_df)
    n_shards = (n + shard_size - 1) // shard_size
    for i in range(n_shards):
        chunk = sample_df.iloc[i * shard_size:(i + 1) * shard_size]
        path = os.path.join(out_dir, f"shard_{i:02d}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for _, row in chunk.iterrows():
                f.write(json.dumps(candidate_record(row), ensure_ascii=False) + "\n")
    side_path = os.path.join(out_dir, "aliyun_strata_sidecar.jsonl")
    with open(side_path, "w", encoding="utf-8") as f:
        for _, row in sample_df.iterrows():
            rec = {
                "id": str(row["id"]),
                "aliyun_final": int(row["aliyun_final"]) if pd.notna(row["aliyun_final"]) else None,
                "positive_prob_aliyun": None if pd.isna(row["positive_prob_aliyun"]) else float(row["positive_prob_aliyun"]),
                "negative_prob_aliyun": None if pd.isna(row["negative_prob_aliyun"]) else float(row["negative_prob_aliyun"]),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[导出] {n} 条 -> {n_shards} 个分片 + sidecar，目录 {out_dir}")


def fetch_joined_frame(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(COUNT_SQL)
        n = cur.fetchone()["n"]
    if n < REQUEST_ROW_MIN or n > REQUEST_ROW_MAX:
        raise SystemExit(
            f"sentiment_aliyun_request 行数为 {n}，不在 [{REQUEST_ROW_MIN}, {REQUEST_ROW_MAX}]，中止导出"
        )
    print(f"[核库] sentiment_aliyun_request 行数 {n}")
    with conn.cursor() as cur:
        cur.execute(JOIN_SQL)
        rows = cur.fetchall()
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=os.path.join(CANDIDATE_DIR, "round1"))
    args = parser.parse_args()
    cfg = require_starrocks_env()
    conn = pymysql.connect(
        host=cfg["host"],
        port=cfg["port"],
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
        charset=cfg["charset"],
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        raw = fetch_joined_frame(conn)
    finally:
        conn.close()
    pool = prepare_pool(raw)
    print(f"[池子] 清洗去重后 {len(pool)} 条")
    sampled = build_round1_sample(pool, sample_size=ROUND1_SAMPLE_SIZE, seed=ROUND1_SEED)
    print(
        f"[抽样] {len(sampled)} 条; "
        f"评论={(sampled['source_table']==COMMENT_SOURCE).sum()} "
        f"负向阿里云层={(sampled['aliyun_final']==3).sum()} "
        f"中性层={(sampled['aliyun_final']==2).sum()}"
    )
    if len(sampled) != ROUND1_SAMPLE_SIZE:
        raise SystemExit(f"抽样条数 {len(sampled)} != {ROUND1_SAMPLE_SIZE}")
    write_round1_outputs(sampled, args.output_dir, shard_size=ROUND1_SHARD_SIZE)
    n_files = len([x for x in os.listdir(args.output_dir) if x.startswith("shard_")])
    if n_files != ROUND1_N_SHARDS:
        raise SystemExit(f"分片数 {n_files} != {ROUND1_N_SHARDS}")


if __name__ == "__main__":
    main()
