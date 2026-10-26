# -*- coding: utf-8 -*-
"""round3 候选导出：B站 xlsx + 微博两 CSV，student 打分，(platform, content_type) 三分配额抽样。

与 round2 的区别：
- 数据源增加 data/raw/LoL评论子agent标注（1600条）_B站.xlsx（B站正面召回弱，浓度 40%）
- 排除文本：round1 全量 + round2 候选 + round3 试标已标 300 + 金标 918 + raw 抽检 400
- 抽样用 build_round3_sample：按 (platform, content_type) 配额（博文必须有意义占比，
  否则学生模型（纯文本输入）会被其他片口径带偏——v5 试标博文仅 8 条的教训）
- 难例仅按 conf<0.6 判定，不再按 p_neg 捞负面
"""
import argparse
import json
import os

import pandas as pd

from config import (
    CANDIDATE_DIR,
    LABELED_DIR,
    ROUND2_COMMENT_SOURCE,
    ROUND2_POST_SOURCE,
    ROUND3_FULL_N_SHARDS,
    ROUND3_FULL_SAMPLE_SIZE,
    ROUND3_FULL_SEED,
    ROUND3_FULL_SHARD_SIZE,
    ROUND3_SLICE_FRAC,
)
from data_prep import clean_text
from export_starrocks_candidates import candidate_record
from export_round2_candidates import (
    find_csv,
    pick_device,
    save_score_cache,
    score_pool,
    text_hash,
)
from sampling import build_round3_sample, is_low_conf

BILIBILI_XLSX = "LoL评论子agent标注（1600条）_B站.xlsx"
BILIBILI_SOURCE = "bilibili_comments"


def _texts_from_dir(path: str) -> set:
    texts = set()
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name.startswith("shard_") and name.endswith(".jsonl"):
                with open(os.path.join(path, name), encoding="utf-8") as f:
                    texts |= {json.loads(l)["text"] for l in f if l.strip()}
    return texts


def load_excluded_texts(round1_path, round2_dir, round3_dir, gold_path, spotcheck_path) -> set:
    texts = set()
    if os.path.exists(round1_path):
        with open(round1_path, encoding="utf-8") as f:
            texts |= {json.loads(l)["text"] for l in f if l.strip()}
    texts |= _texts_from_dir(round2_dir)
    texts |= _texts_from_dir(round3_dir)
    if os.path.exists(gold_path):
        with open(gold_path, encoding="utf-8") as f:
            texts |= {json.loads(l)["text"] for l in f if l.strip()}
    if os.path.exists(spotcheck_path):
        with open(spotcheck_path, encoding="utf-8") as f:
            texts |= {json.loads(l)["text"] for l in f if l.strip()}
    return texts


def load_bilibili_pool(raw_dir: str, excluded: set) -> pd.DataFrame:
    path = os.path.join(raw_dir, BILIBILI_XLSX)
    df = pd.read_excel(path)
    out = df[["id", "评论文本"]].rename(columns={"评论文本": "text"})
    out["text"] = out["text"].map(clean_text)
    out = out[out["text"].str.len() >= 2]
    out = out[~out["text"].isin(excluded)]
    out = out.drop_duplicates(subset=["text"], keep="first")
    out["source_table"] = BILIBILI_SOURCE
    out["platform"] = "bilibili"
    out["content_type"] = "评论"
    out["id"] = out["id"].astype(str)
    out["content_hash"] = out["text"].map(text_hash)
    return out[["id", "text", "platform", "content_type", "source_table", "content_hash"]].reset_index(drop=True)


def load_weibo_pool(raw_dir: str, excluded: set) -> pd.DataFrame:
    frames = []
    for csv_suffix, source, text_col, ctype in (
        ("weibo_post_comments_details.csv", ROUND2_COMMENT_SOURCE, "weibo_post_comment_text", "评论"),
        ("weibo_post_details.csv", ROUND2_POST_SOURCE, "weibo_post_content_details", "博文"),
    ):
        df = pd.read_csv(find_csv(raw_dir, csv_suffix))
        df["text"] = df[text_col].map(clean_text)
        df["source_table"] = source
        df["content_type"] = ctype
        df = df[df["text"].str.len() >= 2]
        df = df[~df["text"].isin(excluded)]
        df = df.drop_duplicates(subset=["text"], keep="first")
        frames.append(df[["id", "text", "source_table", "content_type"]])
    pool = pd.concat(frames, ignore_index=True)
    pool["id"] = pool["id"].astype(str)
    pool["platform"] = "weibo"
    pool["content_hash"] = pool["text"].map(text_hash)
    return pool[["id", "text", "platform", "content_type", "source_table", "content_hash"]].reset_index(drop=True)


def write_round3_outputs(sample_df: pd.DataFrame, out_dir: str, shard_size: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    n = len(sample_df)
    n_shards = (n + shard_size - 1) // shard_size
    for i in range(n_shards):
        chunk = sample_df.iloc[i * shard_size:(i + 1) * shard_size]
        with open(os.path.join(out_dir, f"shard_{i:02d}.jsonl"), "w", encoding="utf-8") as f:
            for _, row in chunk.iterrows():
                f.write(json.dumps(candidate_record(row), ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "student_score_sidecar.jsonl"), "w", encoding="utf-8") as f:
        for _, row in sample_df.iterrows():
            f.write(json.dumps({
                "id": str(row["id"]),
                "platform": row["platform"],
                "content_type": row["content_type"],
                "source_table": row["source_table"],
                "student_pred": row["student_pred"],
                "student_conf": float(row["student_conf"]),
                "hard": bool(is_low_conf(row["student_conf"])),
            }, ensure_ascii=False) + "\n")
    print(f"[导出] {n} 条 -> {n_shards} 个分片 + sidecar，目录 {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=os.path.join("data", "raw"))
    parser.add_argument("--model_dir", default=os.path.join("models", "student_v5"))
    parser.add_argument("--round1-labeled", default=os.path.join(LABELED_DIR, "round1_labeled.jsonl"))
    parser.add_argument("--round2-dir", default=os.path.join(CANDIDATE_DIR, "round2"))
    parser.add_argument("--round3-pilot-dir", default=os.path.join(LABELED_DIR, "round3"))
    parser.add_argument("--gold", default=os.path.join("data", "validation", "validation_gold.jsonl"))
    parser.add_argument("--spotcheck", default=os.path.join("data", "validation", "v5_spotcheck_raw400.jsonl"))
    parser.add_argument("--output-dir", default=os.path.join(CANDIDATE_DIR, "round3_full"))
    parser.add_argument("--cache", default=os.path.join(CANDIDATE_DIR, "round3_full", "pool_scored.jsonl"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()

    excluded = load_excluded_texts(
        args.round1_labeled, args.round2_dir, args.round3_pilot_dir, args.gold, args.spotcheck)
    print(f"[排除] 已标注/金标/抽检文本 {len(excluded)} 条")

    bilibili = load_bilibili_pool(args.raw_dir, excluded)
    weibo = load_weibo_pool(args.raw_dir, excluded)
    print(f"[池子] bilibili {len(bilibili)} | weibo {len(weibo)} "
          f"(评论 {(weibo['content_type'] == '评论').sum()} / "
          f"博文 {(weibo['content_type'] == '博文').sum()})")

    cache_path = args.cache
    if args.reuse_cache and os.path.exists(cache_path):
        scored = pd.DataFrame([json.loads(l) for l in open(cache_path, encoding="utf-8") if l.strip()])
        print(f"[缓存] 复用 {cache_path} 共 {len(scored)} 条")
    else:
        pool = pd.concat([bilibili, weibo], ignore_index=True)
        device = pick_device()
        scored = score_pool(pool, args.model_dir, device, args.batch_size)
        save_score_cache(scored, cache_path)
        print(f"[缓存] 写入 {cache_path}")

    sampled = build_round3_sample(
        scored,
        sample_size=ROUND3_FULL_SAMPLE_SIZE,
        seed=ROUND3_FULL_SEED,
        slice_fracs=ROUND3_SLICE_FRAC,
    )
    comp = sampled.groupby(["platform", "content_type"]).size().to_dict()
    n_hard = int(sampled["student_conf"].map(is_low_conf).sum())
    print(f"[抽样] {len(sampled)} 条; 构成 {comp}; "
          f"低置信难例={n_hard} ({n_hard/len(sampled):.0%}) "
          f"pred={sampled['student_pred'].value_counts().to_dict()}")
    write_round3_outputs(sampled, args.output_dir, ROUND3_FULL_SHARD_SIZE)
    n_files = len([x for x in os.listdir(args.output_dir) if x.startswith("shard_")])
    if len(sampled) != ROUND3_FULL_SAMPLE_SIZE:
        raise SystemExit(f"抽样条数 {len(sampled)} != {ROUND3_FULL_SAMPLE_SIZE}")
    if n_files != ROUND3_FULL_N_SHARDS:
        raise SystemExit(f"分片数 {n_files} != {ROUND3_FULL_N_SHARDS}")


if __name__ == "__main__":
    main()
