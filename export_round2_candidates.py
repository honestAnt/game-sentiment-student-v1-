# -*- coding: utf-8 -*-
"""从 data/raw 两份 LoL 微博 CSV 做 round2 主动学习抽样。"""
import argparse
import hashlib
import json
import os

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    CANDIDATE_DIR,
    LABELED_DIR,
    MAX_SEQ_LEN,
    ROUND2_COMMENT_SOURCE,
    ROUND2_N_SHARDS,
    ROUND2_POST_SOURCE,
    ROUND2_SAMPLE_SIZE,
    ROUND2_SEED,
    ROUND2_SHARD_SIZE,
    STUDENT_MODEL_OUTPUT,
)
from data_prep import clean_text
from export_starrocks_candidates import candidate_record
from infer_batch import batch_infer_probs
from sampling import build_round2_sample, is_hard_example

COMMENT_CSV_SUFFIX = "weibo_post_comments_details.csv"
POST_CSV_SUFFIX = "weibo_post_details.csv"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_round1_texts(path: str) -> set:
    texts = set()
    if not os.path.exists(path):
        return texts
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                texts.add(json.loads(line)["text"])
    return texts


def find_csv(raw_dir: str, suffix: str) -> str:
    matches = [
        os.path.join(raw_dir, n)
        for n in os.listdir(raw_dir)
        if n.endswith(suffix)
    ]
    if len(matches) != 1:
        raise SystemExit(f"在 {raw_dir} 应恰好找到 1 个 *{suffix}，实际 {len(matches)}")
    return matches[0]


def load_unique_pool(raw_dir: str, round1_texts: set) -> pd.DataFrame:
    comment_path = find_csv(raw_dir, COMMENT_CSV_SUFFIX)
    post_path = find_csv(raw_dir, POST_CSV_SUFFIX)
    comments = pd.read_csv(comment_path)
    posts = pd.read_csv(post_path)
    comments["text"] = comments["weibo_post_comment_text"].map(clean_text)
    comments["source_table"] = ROUND2_COMMENT_SOURCE
    posts["text"] = posts["weibo_post_content_details"].map(clean_text)
    posts["source_table"] = ROUND2_POST_SOURCE
    frames = []
    for df in (comments, posts):
        df = df[df["text"].str.len() >= 2]
        df = df[~df["text"].isin(round1_texts)]
        df = df.drop_duplicates(subset=["text"], keep="first")
        frames.append(df[["id", "text", "source_table"]])
    pool = pd.concat(frames, ignore_index=True)
    pool["id"] = pool["id"].astype(str)
    pool["platform"] = "weibo"
    pool["content_hash"] = pool["text"].map(text_hash)
    return pool.reset_index(drop=True)


def load_score_cache(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not rows:
        return None
    return pd.DataFrame(rows)


def save_score_cache(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def score_pool(pool: pd.DataFrame, model_dir: str, device: str, batch_size: int) -> pd.DataFrame:
    print(f"[推理] device={device} n={len(pool)} batch={batch_size} seq={MAX_SEQ_LEN}")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    texts = pool["text"].tolist()
    labels, confs, probs = batch_infer_probs(texts, model, tok, device, batch_size=batch_size)
    out = pool.copy()
    out["student_pred"] = labels
    out["student_conf"] = confs
    out["p_neg"] = [p[0] for p in probs]
    out["p_neu"] = [p[1] for p in probs]
    out["p_pos"] = [p[2] for p in probs]
    return out


def write_round2_outputs(sample_df: pd.DataFrame, out_dir: str, shard_size: int = ROUND2_SHARD_SIZE) -> None:
    os.makedirs(out_dir, exist_ok=True)
    n = len(sample_df)
    n_shards = (n + shard_size - 1) // shard_size
    for i in range(n_shards):
        chunk = sample_df.iloc[i * shard_size:(i + 1) * shard_size]
        path = os.path.join(out_dir, f"shard_{i:02d}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for _, row in chunk.iterrows():
                f.write(json.dumps(candidate_record(row), ensure_ascii=False) + "\n")
    side_path = os.path.join(out_dir, "student_score_sidecar.jsonl")
    with open(side_path, "w", encoding="utf-8") as f:
        for _, row in sample_df.iterrows():
            rec = {
                "id": str(row["id"]),
                "source_table": row["source_table"],
                "student_pred": row["student_pred"],
                "student_conf": float(row["student_conf"]),
                "p_neg": float(row["p_neg"]),
                "p_neu": float(row["p_neu"]),
                "p_pos": float(row["p_pos"]),
                "hard": bool(is_hard_example(row["student_conf"], row["student_pred"], row["p_neg"])),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[导出] {n} 条 -> {n_shards} 个分片 + sidecar，目录 {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=os.path.join("data", "raw"))
    parser.add_argument("--model_dir", default=STUDENT_MODEL_OUTPUT)
    parser.add_argument("--round1-labeled", default=os.path.join(LABELED_DIR, "round1_labeled.jsonl"))
    parser.add_argument("--output-dir", default=os.path.join(CANDIDATE_DIR, "round2"))
    parser.add_argument("--cache", default=os.path.join(CANDIDATE_DIR, "round2", "pool_scored.jsonl"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()

    round1_texts = load_round1_texts(args.round1_labeled)
    print(f"[round1] 已标注文本 {len(round1_texts)} 条，抽样时排除")

    cache_df = load_score_cache(args.cache) if args.reuse_cache else None
    if cache_df is not None:
        print(f"[缓存] 复用 {args.cache} 共 {len(cache_df)} 条")
        scored = cache_df
    else:
        pool = load_unique_pool(args.raw_dir, round1_texts)
        print(
            f"[池子] unique {len(pool)} "
            f"评论={(pool['source_table']==ROUND2_COMMENT_SOURCE).sum()} "
            f"博文={(pool['source_table']==ROUND2_POST_SOURCE).sum()}"
        )
        device = pick_device()
        scored = score_pool(pool, args.model_dir, device, args.batch_size)
        save_score_cache(scored, args.cache)
        print(f"[缓存] 写入 {args.cache}")

    sampled = build_round2_sample(scored, sample_size=ROUND2_SAMPLE_SIZE, seed=ROUND2_SEED)
    n_post = int((sampled["source_table"] == ROUND2_POST_SOURCE).sum())
    n_hard = int(
        sampled.apply(
            lambda r: is_hard_example(r["student_conf"], r["student_pred"], r["p_neg"]),
            axis=1,
        ).sum()
    )
    print(
        f"[抽样] {len(sampled)} 条; 评论={len(sampled)-n_post} 博文={n_post} "
        f"难例={n_hard} pred={sampled['student_pred'].value_counts().to_dict()}"
    )
    write_round2_outputs(sampled, args.output_dir, shard_size=ROUND2_SHARD_SIZE)
    n_files = len([x for x in os.listdir(args.output_dir) if x.startswith("shard_")])
    if len(sampled) != ROUND2_SAMPLE_SIZE:
        raise SystemExit(f"抽样条数 {len(sampled)} != {ROUND2_SAMPLE_SIZE}")
    if n_files != ROUND2_N_SHARDS:
        raise SystemExit(f"分片数 {n_files} != {ROUND2_N_SHARDS}")


if __name__ == "__main__":
    main()
