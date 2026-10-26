# -*- coding: utf-8 -*-
"""round5 候选导出：示例品牌舆情（B站视频+评论 / 小红书笔记+评论）。

新域新平台试点：
- 数据源 4 个 CSV（data/raw/示例品牌__*.csv），带 相关性(品牌/品类/无关) 与机器预标
  sentiment_final；只取相关行，文本列评论用 文本_清洗、详情用 标题+简介/内容 拼接
- 采样逻辑在 sampling.build_round5_pool / build_round5_sample：
  稀缺片（bili评论/bili视频/xhs笔记）全取，xhs评论按机器先验分层补足到总量
- 全池用 student_v5 打分留档 pool_scored.jsonl（新域置信分布供日后对比，不参与配额）
- 候选分片 data/candidates/round5/shard_XX.jsonl，六字段 schema 与 round3/round4 一致
"""
import argparse
import json
import os

import pandas as pd

from config import (
    CANDIDATE_DIR,
    LABELED_DIR,
    RAW_DATA_DIR,
    ROUND5_MCJ_BILI_COMMENTS_CSV,
    ROUND5_MCJ_BILI_VIDEOS_CSV,
    ROUND5_MCJ_XHS_COMMENTS_CSV,
    ROUND5_MCJ_XHS_NOTES_CSV,
    ROUND5_RELEVANCE,
    ROUND5_SEED,
    ROUND5_SHARD_SIZE,
    ROUND5_TOTAL_TARGET,
)
from export_starrocks_candidates import candidate_record
from sampling import build_round5_pool, build_round5_sample

# 文本排除：docs/数据集对齐配置.md 第三节的规范排除集（训练全集 + 金标/评估/留出/抽检）
TEXT_EXCLUDE_FILES = (
    os.path.join(LABELED_DIR, "round1_labeled.jsonl"),
    os.path.join(LABELED_DIR, "round2_labeled.jsonl"),
    os.path.join(LABELED_DIR, "round4_labeled.jsonl"),
    os.path.join("data", "validation", "validation_gold.jsonl"),
    os.path.join("data", "validation", "gold_eval_v6.jsonl"),
    os.path.join("data", "validation", "gold_eval_v7.jsonl"),
    os.path.join("data", "validation", "gold_bili_neg_challenge.jsonl"),
    os.path.join("data", "validation", "round4_holdout.jsonl"),
    os.path.join("data", "validation", "v5_spotcheck_raw400.jsonl"),
    os.path.join("data", "validation", "v5_spotcheck_raw200.jsonl"),
)
EXCLUDE_GLOBS = (
    os.path.join(LABELED_DIR, "round3", "shard_*.jsonl"),
    os.path.join(LABELED_DIR, "round3_full", "shard_*.jsonl"),
)


def load_exclude_texts() -> set:
    import glob as _glob
    paths = [p for p in TEXT_EXCLUDE_FILES if "*" not in p and os.path.exists(p)]
    paths += [p for g in EXCLUDE_GLOBS for p in _glob.glob(g)]
    texts = set()
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.add(json.loads(line)["text"])
    print(f"[排除] 训练/金标/评估/抽检文本 {len(texts)} 条")
    return texts


def _join_text(*parts) -> str:
    return "\n".join(p.strip() for p in parts if isinstance(p, str) and p.strip())


def load_mcj_frames(raw_dir: str, relevance=ROUND5_RELEVANCE) -> dict:
    """4 个 CSV -> 归一化候选行（相关性过滤在此完成）。"""
    bili_c = pd.read_csv(os.path.join(raw_dir, ROUND5_MCJ_BILI_COMMENTS_CSV), dtype=str)
    bili_v = pd.read_csv(os.path.join(raw_dir, ROUND5_MCJ_BILI_VIDEOS_CSV), dtype=str)
    xhs_c = pd.read_csv(os.path.join(raw_dir, ROUND5_MCJ_XHS_COMMENTS_CSV), dtype=str)
    xhs_n = pd.read_csv(os.path.join(raw_dir, ROUND5_MCJ_XHS_NOTES_CSV), dtype=str)

    frames = {
        "bili_comments": pd.DataFrame({
            "id": "mcj_bili_c_" + bili_c["评论ID"],
            "text": bili_c["文本_清洗"],
            "platform": "bilibili",
            "content_type": "评论",
            "source_table": "示例品牌__bili_video_comments_details_sentiment",
            "machine_sentiment": bili_c["sentiment_final"],
            "truncated": bili_c["truncated"],
        }),
        "bili_videos": pd.DataFrame({
            "id": "mcj_bili_v_" + bili_v["bvid"],
            "text": [_join_text(t, d) for t, d in zip(bili_v["标题"], bili_v["简介_清洗"])],
            "platform": "bilibili",
            "content_type": "简介",
            "source_table": "示例品牌__bili_video_details_sentiment",
            "machine_sentiment": bili_v["sentiment_final"],
            "truncated": bili_v["truncated"],
        }),
        "xhs_comments": pd.DataFrame({
            "id": "mcj_xhs_c_" + xhs_c["评论ID"],
            "text": xhs_c["文本_清洗"],
            "platform": "xhs",
            "content_type": "评论",
            "source_table": "示例品牌__xhs_note_comments_details_sentiment",
            "machine_sentiment": xhs_c["sentiment_final"],
            "truncated": xhs_c["truncated"],
        }),
        "xhs_notes": pd.DataFrame({
            "id": "mcj_xhs_n_" + xhs_n["笔记ID"],
            "text": [_join_text(t, c) for t, c in zip(xhs_n["标题"], xhs_n["内容_清洗"])],
            "platform": "xhs",
            "content_type": "笔记",
            "source_table": "示例品牌__xhs_note_details_sentiment",
            "machine_sentiment": xhs_n["sentiment_final"],
            "truncated": xhs_n["truncated"],
        }),
    }

    out = {}
    for name, df in frames.items():
        rel = None
        if name == "bili_comments":
            rel = bili_c["相关性"]
        elif name == "bili_videos":
            rel = bili_v["相关性"]
        elif name == "xhs_comments":
            rel = xhs_c["相关性"]
        else:
            rel = xhs_n["相关性"]
        kept = df[rel.isin(relevance)].reset_index(drop=True)
        print(f"[源] {name}: 相关行 {len(kept)}/{len(df)}")
        out[name] = kept
    return out


def score_pool_with_v5(pool_df: pd.DataFrame, model_dir: str) -> pd.DataFrame:
    """student_v5 全池打分，附加 student_pred / student_conf（留档用，不参与采样配额）。"""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from config import ID2LABEL, MAX_SEQ_LEN

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    texts = pool_df["text"].tolist()
    preds, confs = [], []
    bs = 64
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            inputs = tok(texts[i:i + bs], padding=True, truncation=True,
                         max_length=MAX_SEQ_LEN, return_tensors="pt").to(device)
            probs = torch.softmax(model(**inputs).logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            preds.extend(ID2LABEL[p.item()] for p in pred)
            confs.extend(conf.cpu().tolist())
    out = pool_df.copy()
    out["student_pred"] = preds
    out["student_conf"] = [round(c, 4) for c in confs]
    print(f"[打分] student_v5 完成 {len(out)} 条（device={device}）")
    return out


def write_pool_scored(pool_df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["id", "text", "platform", "content_type", "source_table", "content_hash",
            "machine_sentiment", "student_pred", "student_conf", "_source"]
    with open(path, "w", encoding="utf-8") as f:
        for _, row in pool_df[cols].iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def write_shards(sample_df: pd.DataFrame, out_dir: str, shard_size: int) -> int:
    os.makedirs(out_dir, exist_ok=True)
    n = len(sample_df)
    n_shards = (n + shard_size - 1) // shard_size
    for i in range(n_shards):
        chunk = sample_df.iloc[i * shard_size:(i + 1) * shard_size]
        with open(os.path.join(out_dir, f"shard_{i:02d}.jsonl"), "w", encoding="utf-8") as f:
            for _, row in chunk.iterrows():
                f.write(json.dumps(candidate_record(row), ensure_ascii=False) + "\n")
    print(f"[导出] {n} 条 -> {n_shards} 个分片，目录 {out_dir}")
    return n_shards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", default=os.path.join(CANDIDATE_DIR, "round5"))
    parser.add_argument("--model-dir", default="models/student_v5")
    parser.add_argument("--total-target", type=int, default=ROUND5_TOTAL_TARGET)
    parser.add_argument("--skip-scoring", action="store_true", help="跳过 v5 打分（只出分片）")
    args = parser.parse_args()

    exclude_texts = load_exclude_texts()
    frames = load_mcj_frames(args.raw_dir)
    pool = build_round5_pool(frames, exclude_texts)
    comp = pool.groupby(["platform", "content_type"]).size().to_dict()
    prior = pool["machine_sentiment"].value_counts().to_dict()
    print(f"[池] 过滤去重后 {len(pool)} 条，构成 {comp}，机器先验 {prior}")

    if not args.skip_scoring:
        scored = score_pool_with_v5(pool, args.model_dir)
        low_conf = (scored["student_conf"] < 0.75).mean()
        print(f"[打分] v5 在新域 conf<0.75 占比 {low_conf:.1%}（游戏域 gold 上为 31.8%）")
        write_pool_scored(scored, os.path.join(args.output_dir, "pool_scored.jsonl"))

    sample = build_round5_sample(pool, total_target=args.total_target, seed=ROUND5_SEED)
    if sample["text"].isin(exclude_texts).any():
        raise SystemExit("采样后仍与排除集文本重合")
    if sample["content_hash"].duplicated().any():
        raise SystemExit("采样内部存在重复文本")
    slices = sample.groupby(["platform", "content_type"]).size().to_dict()
    print(f"[采样] {len(sample)} 条，构成 {slices}，机器先验 {sample['machine_sentiment'].value_counts().to_dict()}")
    write_shards(sample, args.output_dir, ROUND5_SHARD_SIZE)


if __name__ == "__main__":
    main()
